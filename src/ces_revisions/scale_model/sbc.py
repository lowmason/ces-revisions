"""Simulation-based calibration of the scale model on the primary series' design.

Each replicate draws the parameters from the prior, simulates the revisions on the real
design (the windows, intervals, lapses, years, and months of the series), fits one NUTS
chain at the fit's target acceptance, and ranks the true value among evenly spaced
posterior draws. The rank arithmetic, the gamma-ECDF test (Säilynoja, Bürkner & Vehtari
2022), and the recovery statistics are the Stage 6 pilot's (``ces_revisions.pilot.sbc``).

Pre-registered pass rule (2026-11-17, before any replicate ran): every quantity in
QUANTITIES, the scalar sampled sites and the log likelihood at the true parameters (a
data-dependent test quantity, Modrák et al. 2023), passes the gamma-ECDF test at the
Bonferroni level FAMILY_ALPHA / len(QUANTITIES); and for beta, the estimand, the median
contraction is at least BETA_CONTRACTION_MIN and at most LARGE_Z_SHARE_MAX of the
replicates have a posterior z-score beyond LARGE_Z.
"""

import dataclasses
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import arviz as az
import jax
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median

from ces_revisions.devices import chain_method, has_nvidia_device
from ces_revisions.pilot.sbc import (
    gamma_critical_value,
    gamma_statistic,
    rank_of,
    recovery,
    replicate_seeds,
    thin,
)
from ces_revisions.scale_model.data import ScaleData
from ces_revisions.scale_model.model import (
    DEFAULT_PRIORS,
    OBSERVED_SITE,
    TARGET_ACCEPT_PROB,
    Priors,
    pointwise_log_likelihood,
    sample_prior,
    scale_model,
)

SCALAR_SITES = (
    "m",
    "gamma",
    "phi",
    "a",
    "tau_y",
    "beta",
    "kappa",
    "delta",
    "lam",
    "nu",
)
LOG_LIKELIHOOD = "log_likelihood"
QUANTITIES = (*SCALAR_SITES, LOG_LIKELIHOOD)
FAMILY_ALPHA = 0.05
BETA_CONTRACTION_MIN = 0.5
LARGE_Z = 3.0
LARGE_Z_SHARE_MAX = 0.05
SEED = sum(map(ord, "stage-1-scale-model-sbc"))
PRIOR_REFERENCE_SEED = sum(map(ord, "stage-1-scale-model-sbc-prior-reference"))
PRIOR_REFERENCE_DRAWS = 20_000
ECDF_SEED = sum(map(ord, "stage-1-scale-model-sbc-ecdf"))
# The family's critical value sits at alpha = 0.05 / 11, so its tail needs many simulations.
ECDF_SIMULATIONS = 100_000
MAX_TREE_DEPTH = 10
INIT_MEDIAN_DRAWS = 15
REPLICATE_FILE = "replicate-{index:04d}.json"
REPLICATE_GLOB = "replicate-*.json"
SUMMARY_FILE = "summary.json"
HOST_DEVICE_FLAG = "--xla_force_host_platform_device_count"


def prior_sds(data: ScaleData, priors: Priors) -> dict[str, float]:
    """The prior SD of each scalar site, from PRIOR_REFERENCE_DRAWS draws at a fixed seed."""
    draws = sample_prior(
        jax.random.PRNGKey(PRIOR_REFERENCE_SEED), PRIOR_REFERENCE_DRAWS, data, priors
    )
    return {name: float(np.std(draws[name], ddof=1)) for name in SCALAR_SITES}


def run_replicate(
    index: int,
    *,
    base_seed: int,
    data: ScaleData,
    num_warmup: int,
    num_samples: int,
    num_kept: int,
    priors: Priors = DEFAULT_PRIORS,
    target_accept_prob: float = TARGET_ACCEPT_PROB,
) -> dict:
    """One replicate: draw theta, simulate revisions on ``data``'s design, fit, and rank.

    Ranks use ``num_kept`` evenly spaced post-warmup draws, so they lie in 0..num_kept;
    recovery and the bulk ESS use every draw. The returned record is JSON-safe.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("SBC needs float64: call numpyro.enable_x64() first")
    started = time.perf_counter()
    seeds = replicate_seeds(base_seed, index)
    theta = sample_prior(jax.random.PRNGKey(seeds["prior"]), 1, data, priors)
    simulated_revision = Predictive(
        scale_model, posterior_samples=theta, return_sites=[OBSERVED_SITE]
    )(jax.random.PRNGKey(seeds["simulation"]), data, priors)[OBSERVED_SITE]
    simulated = dataclasses.replace(
        data, revision=np.asarray(simulated_revision)[0].astype(np.float64)
    )

    mcmc = MCMC(
        NUTS(
            scale_model,
            max_tree_depth=MAX_TREE_DEPTH,
            target_accept_prob=target_accept_prob,
            init_strategy=init_to_median(num_samples=INIT_MEDIAN_DRAWS),
        ),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
        chain_method=chain_method(1),
        progress_bar=False,
    )
    sampling_started = time.perf_counter()
    mcmc.run(
        jax.random.PRNGKey(seeds["mcmc"]),
        simulated,
        priors,
        simulated.revision,
        extra_fields=("diverging", "num_steps"),
    )
    samples = {
        name: np.asarray(value)
        for name, value in jax.block_until_ready(mcmc.get_samples()).items()
    }
    sampling_seconds = time.perf_counter() - sampling_started

    truth = {name: float(theta[name][0]) for name in SCALAR_SITES}
    kept = thin(np.arange(num_samples), num_kept)
    total_log_likelihood = jax.jit(
        jax.vmap(lambda params: pointwise_log_likelihood(params, simulated).sum())
    )
    truth[LOG_LIKELIHOOD] = float(
        total_log_likelihood({name: value for name, value in theta.items()})[0]
    )
    kept_draws = {name: samples[name][kept] for name in SCALAR_SITES}
    kept_draws[LOG_LIKELIHOOD] = np.asarray(
        total_log_likelihood({name: samples[name][kept] for name in theta})
    )
    rng = np.random.default_rng(seeds["ties"])
    sds = prior_sds(data, priors)
    extra = mcmc.get_extra_fields()
    num_steps = np.asarray(extra["num_steps"])
    return {
        "index": index,
        "seeds": seeds,
        "settings": {
            "series": list(data.series),
            "num_months": len(data.reference_months),
            "base_seed": base_seed,
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "num_kept": num_kept,
            "max_tree_depth": MAX_TREE_DEPTH,
            "target_accept_prob": target_accept_prob,
            "priors": dataclasses.asdict(priors),
            "prior_reference_seed": PRIOR_REFERENCE_SEED,
        },
        "truth": truth,
        "ranks": {
            name: rank_of(truth[name], kept_draws[name], rng) for name in QUANTITIES
        },
        "recovery": {
            name: recovery(truth[name], samples[name], sds[name])
            for name in SCALAR_SITES
        },
        "ess_bulk": {
            name: _json_float(az.ess(samples[name][None, :], method="bulk"))
            for name in SCALAR_SITES
        },
        "divergences": int(np.asarray(extra["diverging"]).sum()),
        "n_steps": {
            "mean": float(num_steps.mean()),
            "max": int(num_steps.max()),
            "saturated_share": float(np.mean(num_steps >= 2**MAX_TREE_DEPTH - 1)),
        },
        "seconds": {
            "sampling": sampling_seconds,
            "total": time.perf_counter() - started,
        },
    }


def summarize_sbc(
    records: list[dict], *, num_simulations: int = ECDF_SIMULATIONS
) -> dict:
    """The gamma-ECDF test of every quantity, beta's recovery, and the verdict."""
    if not records:
        raise ValueError("no replicate records to summarize")
    settings = records[0]["settings"]
    if any(record["settings"] != settings for record in records):
        raise ValueError("replicate records were run with different settings")
    num_kept = settings["num_kept"]
    alpha = FAMILY_ALPHA / len(QUANTITIES)
    critical = gamma_critical_value(
        len(records), num_kept, alpha, num_simulations=num_simulations, seed=ECDF_SEED
    )
    quantities = {}
    for name in QUANTITIES:
        ranks = np.array([record["ranks"][name] for record in records], dtype=np.int64)
        gamma = gamma_statistic(ranks, num_kept)
        quantities[name] = {
            "gamma": gamma,
            "critical": critical,
            "passed": bool(gamma >= critical),
        }
    z = np.array([record["recovery"]["beta"]["z"] for record in records])
    contraction = np.array(
        [record["recovery"]["beta"]["contraction"] for record in records]
    )
    beta_recovery = {
        "median_contraction": float(np.median(contraction)),
        "share_abs_z_above_3": float(np.mean(np.abs(z) > LARGE_Z)),
    }
    beta_recovery["passed"] = bool(
        beta_recovery["median_contraction"] >= BETA_CONTRACTION_MIN
        and beta_recovery["share_abs_z_above_3"] <= LARGE_Z_SHARE_MAX
    )
    failures = [name for name, entry in quantities.items() if not entry["passed"]]
    return {
        "settings": settings,
        "num_replicates": len(records),
        "indices": [record["index"] for record in records],
        "num_quantities": len(QUANTITIES),
        "alpha": alpha,
        "ecdf_simulations": num_simulations,
        "quantities": quantities,
        "ecdf_failures": failures,
        "beta_recovery": beta_recovery,
        "divergent_replicates": sum(record["divergences"] > 0 for record in records),
        "saturated_replicates": sum(
            record["n_steps"]["saturated_share"] > 0 for record in records
        ),
        "n_steps_mean": float(
            np.mean([record["n_steps"]["mean"] for record in records])
        ),
        "replicate_seconds_median": float(
            np.median([record["seconds"]["total"] for record in records])
        ),
        "passed": not failures and beta_recovery["passed"],
    }


def replicate_path(out_dir: Path, index: int) -> Path:
    return out_dir / REPLICATE_FILE.format(index=index)


def read_replicates(out_dir: Path) -> list[dict]:
    """Every replicate record in out_dir, by index."""
    return [
        json.loads(path.read_text()) for path in sorted(out_dir.glob(REPLICATE_GLOB))
    ]


def write_summary(out_dir: Path) -> dict:
    """Summarize every replicate in out_dir and write it to out_dir / SUMMARY_FILE."""
    summary = summarize_sbc(read_replicates(out_dir))
    _write_json(summary, out_dir / SUMMARY_FILE)
    return summary


def run_sbc(
    out_dir: Path,
    *,
    data: ScaleData,
    start: int,
    stop: int,
    workers: int,
    base_seed: int,
    num_warmup: int,
    num_samples: int,
    num_kept: int,
    priors: Priors = DEFAULT_PRIORS,
) -> dict:
    """Run replicates start..stop-1 in spawned single-device workers; summarize out_dir.

    A replicate whose file exists is skipped, so a stopped run resumes. A failed replicate
    writes no file and is listed under "failures", so a rerun retries it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        index
        for index in range(start, stop)
        if not replicate_path(out_dir, index).exists()
    ]
    failures = {}
    if pending:
        if workers > 1 and has_nvidia_device():
            raise ValueError("on a GPU host every worker would share one GPU")
        saved_flags = os.environ.get("XLA_FLAGS")
        kept_flags = [
            flag
            for flag in (saved_flags or "").split()
            if not flag.startswith(HOST_DEVICE_FLAG)
        ]
        os.environ["XLA_FLAGS"] = " ".join([*kept_flags, f"{HOST_DEVICE_FLAG}=1"])
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_worker,
            ) as pool:
                futures = {
                    pool.submit(
                        run_replicate,
                        index,
                        base_seed=base_seed,
                        data=data,
                        num_warmup=num_warmup,
                        num_samples=num_samples,
                        num_kept=num_kept,
                        priors=priors,
                    ): index
                    for index in pending
                }
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        _write_json(future.result(), replicate_path(out_dir, index))
                    except Exception as error:  # noqa: BLE001 - one failure must not stop the rest
                        failures[index] = repr(error)
        finally:
            if saved_flags is None:
                del os.environ["XLA_FLAGS"]
            else:
                os.environ["XLA_FLAGS"] = saved_flags
    summary = write_summary(out_dir)
    summary["failures"] = {
        str(index): error for index, error in sorted(failures.items())
    }
    summary["skipped"] = (stop - start) - len(pending)
    return summary


def _initialize_worker() -> None:
    numpyro.enable_x64()
    if jax.default_backend() == "cpu" and jax.local_device_count() != 1:
        raise RuntimeError(
            f"an SBC worker must see one host device, not {jax.local_device_count()}"
        )


def _write_json(record: dict, path: Path) -> None:
    """Write through a temporary file, so a killed process never leaves a partial file."""
    text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def _json_float(value) -> float | None:
    value = float(np.asarray(value))
    return value if np.isfinite(value) else None
