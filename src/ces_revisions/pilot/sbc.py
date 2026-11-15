"""Simulation-based calibration of the pilot: rank arithmetic, the ECDF test, and the runner.

When a parameter is drawn from its prior, a panel is simulated from it, and the model is
fit to that panel, the parameter's rank among L posterior draws is uniform on 0..L
(Talts et al. 2018). Säilynoja, Bürkner & Vehtari (2022) test that uniformity with the
statistic gamma: the smallest two-sided binomial tail probability of the ranks' empirical
CDF over the L interior evaluation points, compared with its own lower alpha-quantile
under uniform ranks. The rank arithmetic is NumPy on finished draws.

``run_replicate`` performs one replicate on the pilot: a prior draw, a panel simulated
with the template panel's gaps, one NUTS chain, and the ranks, recovery, and ESS of every
sampled site, the derived quantities, and the log likelihood as a data-dependent test
quantity (Modrák et al. 2023). ``run_sbc`` fans replicates over worker processes, one
JSON file per replicate, and ``summarize_sbc`` applies the pre-registered pass rule.
"""

import dataclasses
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, init_to_value

from ces_revisions.devices import chain_method, has_nvidia_device
from ces_revisions.kalman import kalman_filter
from ces_revisions.pilot.layout import LAYOUTS, Layout
from ces_revisions.pilot.model import (
    SAMPLED_SITES,
    TARGET_ACCEPT_PROB,
    derived,
    pilot_model,
    pilot_ssm,
    sample_prior,
)
from ces_revisions.pilot.panel import AggregatePanel
from ces_revisions.pilot.simulate import simulate_panel

# Simulated rank vectors per block in gamma_critical_value, which bounds its memory to
# a few (block, L) arrays however many simulations the caller asks for.
SIMULATION_BLOCK = 1_000


def rank_of(
    prior_value: float, draws: np.ndarray, rng: np.random.Generator | None = None
) -> int:
    """The number of posterior draws below the prior value, a rank in 0..len(draws).

    A draw equal to the prior value is a tie. Under exchangeability the prior value is
    equally likely to sit at each position among its ties, so the rank adds a uniform
    integer in 0..ties drawn from ``rng``; ties without an ``rng`` raise.
    """
    draws = np.asarray(draws, dtype=np.float64)
    if draws.ndim != 1 or draws.size == 0:
        raise ValueError("draws must be a non-empty 1-D array")
    if not np.isfinite(prior_value) or not np.isfinite(draws).all():
        raise ValueError("the prior value and every draw must be finite")
    below = int(np.count_nonzero(draws < prior_value))
    ties = int(np.count_nonzero(draws == prior_value))
    if ties == 0:
        return below
    if rng is None:
        raise ValueError(f"{ties} draws tie the prior value: pass a seeded rng")
    return below + int(rng.integers(0, ties + 1))


def thin(draws: np.ndarray, num_kept: int) -> np.ndarray:
    """``num_kept`` draws at evenly spaced positions, from the first draw on.

    Position j is ``j * len(draws) // num_kept``: a constant stride when ``num_kept``
    divides the draw count, and never a repeated draw.
    """
    draws = np.asarray(draws)
    if draws.ndim != 1:
        raise ValueError("draws must be a 1-D array")
    if not 1 <= num_kept <= draws.size:
        raise ValueError(f"num_kept must lie in 1..{draws.size}, not {num_kept}")
    return draws[np.arange(num_kept) * draws.size // num_kept]


def gamma_statistic(ranks: np.ndarray, num_draws: int) -> float:
    """Säilynoja, Bürkner & Vehtari's gamma for N ranks in 0..L, L = ``num_draws``.

    At each interior point z_i = i / (L + 1), i = 1..L, the count of ranks below i is
    Binomial(N, z_i) under uniform ranks; gamma is twice the smallest tail probability,
    lower P(X <= count) or upper P(X >= count) = 1 - P(X <= count - 1), over every i.
    Gamma can exceed 1, because both tails of the count's own atom can exceed one half.
    """
    ranks = _validated_ranks(ranks, num_draws)
    tails = _binomial_tails(ranks.size, num_draws)
    return float(_gammas(ranks[None, :], num_draws, tails)[0])


def gamma_critical_value(
    num_replicates: int,
    num_draws: int,
    alpha: float,
    *,
    num_simulations: int = 10_000,
    seed: int,
) -> float:
    """The alpha-quantile of gamma over simulated uniform ranks in 0..``num_draws``.

    The quantile is the inverse empirical CDF, so it is one of the simulated gammas:
    gamma takes discrete values, and the rule ``gamma >= critical`` then rejects
    uniform ranks with probability below alpha, up to simulation error. It rises with
    alpha: a larger alpha rejects more.
    """
    if num_replicates < 1 or num_draws < 1 or num_simulations < 1:
        raise ValueError("num_replicates, num_draws, and num_simulations must be >= 1")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1), not {alpha}")
    rng = np.random.default_rng(seed)
    tails = _binomial_tails(num_replicates, num_draws)
    gammas = []
    for start in range(0, num_simulations, SIMULATION_BLOCK):
        block = min(SIMULATION_BLOCK, num_simulations - start)
        ranks = rng.integers(0, num_draws + 1, size=(block, num_replicates))
        gammas.append(_gammas(ranks, num_draws, tails))
    return float(np.quantile(np.concatenate(gammas), alpha, method="inverted_cdf"))


def ecdf_test(
    ranks: np.ndarray,
    num_draws: int,
    alpha: float,
    *,
    seed: int,
    num_simulations: int = 10_000,
) -> dict:
    """Gamma for the ranks, its simulated critical value, and whether the ranks pass."""
    ranks = _validated_ranks(ranks, num_draws)
    gamma = gamma_statistic(ranks, num_draws)
    critical = gamma_critical_value(
        ranks.size, num_draws, alpha, num_simulations=num_simulations, seed=seed
    )
    return {"gamma": gamma, "critical": critical, "passed": gamma >= critical}


def recovery(prior_value: float, draws: np.ndarray, prior_sd: float) -> dict:
    """Posterior z-score and contraction of one parameter fit to one simulated panel.

    The z-score is (posterior mean - prior value) / posterior SD, and the contraction
    1 - posterior variance / prior variance; both use the draws' ddof=1 variance. A
    posterior with zero or non-finite variance (a constant chain) has neither: both are
    NaN, never an infinite z-score or a contraction of 1.
    """
    draws = np.asarray(draws, dtype=np.float64)
    if draws.ndim != 1 or draws.size < 2:
        raise ValueError("draws must be a 1-D array of at least two draws")
    if not prior_sd > 0:
        raise ValueError(f"prior_sd must be positive, not {prior_sd}")
    variance = draws.var(ddof=1)
    if not (np.isfinite(variance) and variance > 0):
        return {"z": float("nan"), "contraction": float("nan")}
    return {
        "z": float((draws.mean() - prior_value) / np.sqrt(variance)),
        "contraction": float(1.0 - variance / prior_sd**2),
    }


def _validated_ranks(ranks: np.ndarray, num_draws: int) -> np.ndarray:
    ranks = np.asarray(ranks)
    if ranks.ndim != 1 or ranks.size == 0:
        raise ValueError("ranks must be a non-empty 1-D array")
    if not np.issubdtype(ranks.dtype, np.integer):
        raise ValueError(f"ranks must be integers, not {ranks.dtype}")
    if num_draws < 1:
        raise ValueError(f"num_draws must be >= 1, not {num_draws}")
    if ranks.min() < 0 or ranks.max() > num_draws:
        raise ValueError(f"every rank must lie in 0..{num_draws}")
    return ranks.astype(np.int64)


def _binomial_tails(
    num_replicates: int, num_draws: int
) -> tuple[np.ndarray, np.ndarray]:
    """P(X_i <= k) and P(X_i >= k) for X_i ~ Binomial(N, i / (L + 1)), i = 1..L.

    Two (L, N + 1) tables indexed [i - 1, k]. The upper tail is summed from the top, not
    taken as one minus the lower tail, so a small upper tail keeps its relative accuracy.
    """
    z = np.arange(1, num_draws + 1) / (num_draws + 1)
    k = np.arange(num_replicates + 1)
    # log C(N, k) as a running sum of log((N - j + 1) / j), j = 1..k.
    log_comb = np.concatenate(
        [[0.0], np.cumsum(np.log((num_replicates - k[1:] + 1) / k[1:]))]
    )
    log_pmf = (
        log_comb + k * np.log(z)[:, None] + (num_replicates - k) * np.log1p(-z)[:, None]
    )
    pmf = np.exp(log_pmf)
    lower = np.cumsum(pmf, axis=1)
    upper = np.cumsum(pmf[:, ::-1], axis=1)[:, ::-1]
    return lower, upper


def _gammas(
    ranks: np.ndarray, num_draws: int, tails: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """Gamma for each row of an (S, N) matrix of integer ranks in 0..L."""
    num_rows = ranks.shape[0]
    bins = num_draws + 1
    # One histogram of ranks per row, counted in a single bincount over offset ranks.
    offsets = np.arange(num_rows)[:, None] * bins
    histogram = np.bincount((ranks + offsets).ravel(), minlength=num_rows * bins)
    # Column i - 1 counts the ranks below i, i = 1..L: an integer, never N * F-hat.
    below = np.cumsum(histogram.reshape(num_rows, bins)[:, :-1], axis=1)
    lower, upper = tails
    points = np.arange(num_draws)
    return 2.0 * np.minimum(lower[points, below], upper[points, below]).min(axis=1)


# The replicate runner.

DERIVED_QUANTITIES = (
    "sigma_noise_f",
    "sigma_noise_s",
    "sigma_noise_t",
    "share_s",
    "share_t",
)
# log p(y | theta), ranked at the true theta among the posterior draws: a data-dependent
# test quantity that is sensitive to miscalibration the marginal ranks miss.
LOG_LIKELIHOOD = "log_likelihood"
QUANTITIES = (*SAMPLED_SITES, *DERIVED_QUANTITIES, LOG_LIKELIHOOD)
# Pre-registered pass rule: every quantity passes the gamma-ECDF test at the Bonferroni
# level FAMILY_ALPHA / len(QUANTITIES); and for each news share, the median contraction
# is at least CONTRACTION_MIN and at most LARGE_Z_SHARE_MAX of the replicates have a
# posterior z-score beyond LARGE_Z.
FAMILY_ALPHA = 0.05
RECOVERY_QUANTITIES = ("share_s", "share_t")
CONTRACTION_MIN = 0.25
LARGE_Z = 3.0
LARGE_Z_SHARE_MAX = 0.05
# The prior draws behind every replicate's starting point (the prior median of each
# sampled site) and the prior SDs that scale recovery's contraction.
PRIOR_REFERENCE_SEED = sum(map(ord, "ces-revisions-stage-6-sbc-prior-reference"))
PRIOR_REFERENCE_DRAWS = 20_000
ECDF_SEED = sum(map(ord, "ces-revisions-stage-6-sbc-ecdf"))
# The family's critical value sits at alpha = 0.05 / 23, so its tail needs many simulations.
ECDF_SIMULATIONS = 100_000
MAX_TREE_DEPTH = 10
# The reduced panel: January 1999 to December 2021 holds 52 months before the May 2003
# shift, the 2020 crisis months, and the anchors of 2003-2021, so the shift, crisis, and
# post-T news parameters each meet data in every replicate.
FIRST_MONTH = date(1999, 1, 1)
LAST_MONTH = date(2021, 12, 1)
# Draws whose log likelihood is filtered together; bounds the (batch, T, n, n) buffers.
LOG_LIKELIHOOD_BATCH = 25
REPLICATE_FILE = "replicate-{index:04d}.json"
REPLICATE_GLOB = "replicate-*.json"
SUMMARY_FILE = "summary.json"
HOST_DEVICE_FLAG = "--xla_force_host_platform_device_count"


def replicate_seeds(base_seed: int, index: int) -> dict[str, int]:
    """Independent seeds for a replicate's prior draw, simulation, NUTS, and rank ties."""
    names = ("prior", "simulation", "mcmc", "ties")
    states = np.random.SeedSequence((base_seed, index)).generate_state(len(names))
    return {name: int(state) for name, state in zip(names, states, strict=True)}


def prior_reference(layout_name: str) -> dict:
    """Prior medians of the sampled sites and prior SDs of the sites and derived quantities.

    Estimated from PRIOR_REFERENCE_DRAWS ``sample_prior`` draws at a fixed seed, so every
    replicate starts NUTS at the same point and scales contraction by the same SDs.
    """
    draws = {
        name: np.asarray(values)
        for name, values in sample_prior(
            jax.random.PRNGKey(PRIOR_REFERENCE_SEED),
            PRIOR_REFERENCE_DRAWS,
            LAYOUTS[layout_name],
        ).items()
    }
    quantities = {**draws, **derived(draws)}
    return {
        "seed": PRIOR_REFERENCE_SEED,
        "num_draws": PRIOR_REFERENCE_DRAWS,
        "median": {name: float(np.median(draws[name])) for name in SAMPLED_SITES},
        "sd": {
            name: float(np.std(quantities[name], ddof=1))
            for name in (*SAMPLED_SITES, *DERIVED_QUANTITIES)
        },
    }


def run_replicate(
    index: int,
    *,
    base_seed: int,
    layout_name: str,
    panel: AggregatePanel,
    num_warmup: int,
    num_samples: int,
    num_kept: int,
    target_accept_prob: float = TARGET_ACCEPT_PROB,
) -> dict:
    """One SBC replicate: draw theta, simulate a panel with ``panel``'s gaps, fit, rank.

    One NUTS chain starts at the prior median. Ranks use ``num_kept`` evenly spaced
    post-warmup draws, so they lie in 0..num_kept; recovery and the bulk ESS use every
    draw. ``target_accept_prob`` should match the fit being calibrated. The returned
    record is JSON-safe.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("SBC needs float64: call numpyro.enable_x64() first")
    started = time.perf_counter()
    layout = LAYOUTS[layout_name]
    seeds = replicate_seeds(base_seed, index)
    reference = prior_reference(layout_name)
    theta = {
        name: float(values[0])
        for name, values in sample_prior(
            jax.random.PRNGKey(seeds["prior"]), 1, layout
        ).items()
    }
    simulated = dataclasses.replace(
        panel, y=simulate_panel(theta, panel, layout, seeds["simulation"])
    )

    mcmc = MCMC(
        NUTS(
            pilot_model,
            max_tree_depth=MAX_TREE_DEPTH,
            target_accept_prob=target_accept_prob,
            init_strategy=init_to_value(values=reference["median"]),
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
        layout,
        extra_fields=("diverging", "num_steps"),
    )
    samples = jax.block_until_ready(mcmc.get_samples())
    sampling_seconds = time.perf_counter() - sampling_started
    samples = {name: np.asarray(samples[name]) for name in SAMPLED_SITES}
    draws = {**samples, **derived(samples)}
    truth = {**theta, **{name: float(value) for name, value in derived(theta).items()}}

    recoveries = {
        name: recovery(truth[name], draws[name], reference["sd"][name])
        for name in (*SAMPLED_SITES, *DERIVED_QUANTITIES)
    }
    _require_finite_recovery(index, recoveries)

    kept = thin(np.arange(num_samples), num_kept)
    log_likelihoods = _log_likelihoods(
        {
            name: np.concatenate([[theta[name]], samples[name][kept]])
            for name in SAMPLED_SITES
        },
        simulated,
        layout,
    )
    truth[LOG_LIKELIHOOD] = float(log_likelihoods[0])
    kept_draws = {name: draws[name][kept] for name in draws}
    kept_draws[LOG_LIKELIHOOD] = log_likelihoods[1:]

    rng = np.random.default_rng(seeds["ties"])
    extra = mcmc.get_extra_fields()
    num_steps = np.asarray(extra["num_steps"])
    return {
        "index": index,
        "seeds": seeds,
        "settings": {
            "layout": layout_name,
            "base_seed": base_seed,
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "num_kept": num_kept,
            "max_tree_depth": MAX_TREE_DEPTH,
            "target_accept_prob": target_accept_prob,
            "first_month": panel.reference_months[0].isoformat(),
            "last_month": panel.reference_months[-1].isoformat(),
            "prior_reference_seed": reference["seed"],
        },
        "truth": truth,
        "ranks": {
            name: rank_of(truth[name], kept_draws[name], rng) for name in QUANTITIES
        },
        "recovery": recoveries,
        "ess_bulk": {
            name: _json_float(az.ess(draws[name][None, :], method="bulk"))
            for name in (*SAMPLED_SITES, *DERIVED_QUANTITIES)
        },
        "ess_bulk_kept": {
            name: _json_float(az.ess(kept_draws[name][None, :], method="bulk"))
            for name in QUANTITIES
        },
        "divergences": int(np.asarray(extra["diverging"]).sum()),
        "n_steps": {
            "mean": float(num_steps.mean()),
            "max": int(num_steps.max()),
            "saturated_share": float(np.mean(num_steps >= 2**MAX_TREE_DEPTH - 1)),
        },
        "step_size": float(mcmc.last_state.adapt_state.step_size),
        "seconds": {
            "sampling": sampling_seconds,
            "total": time.perf_counter() - started,
        },
    }


class NonFiniteRecovery(ValueError):
    """A replicate whose recovery cannot be computed finitely, so it cannot be recorded."""


def _require_finite_recovery(index: int, recoveries: dict) -> None:
    """Raise NonFiniteRecovery naming each quantity whose z or contraction is not finite."""
    bad = [
        name
        for name, entry in recoveries.items()
        if not (np.isfinite(entry["z"]) and np.isfinite(entry["contraction"]))
    ]
    if bad:
        raise NonFiniteRecovery(
            f"replicate {index} has non-finite recovery for {', '.join(bad)}: "
            "the posterior draws have zero or non-finite variance"
        )


def _log_likelihoods(params: dict, panel: AggregatePanel, layout: Layout) -> np.ndarray:
    """log p(y | theta) for each draw in params, (draws,) arrays per sampled site."""
    y = jnp.asarray(panel.y)

    def one(draw):
        return kalman_filter(pilot_ssm(draw, panel, layout), y).log_likelihood

    mapped = jax.jit(
        lambda draws: jax.lax.map(one, draws, batch_size=LOG_LIKELIHOOD_BATCH)
    )
    return np.asarray(
        mapped({name: jnp.asarray(value) for name, value in params.items()})
    )


def _json_float(value) -> float | None:
    value = float(np.asarray(value))
    return value if np.isfinite(value) else None


def replicate_path(out_dir: Path, index: int) -> Path:
    return out_dir / REPLICATE_FILE.format(index=index)


def run_sbc(
    out_dir: Path,
    *,
    layout_name: str,
    panel: AggregatePanel,
    start: int,
    stop: int,
    workers: int,
    base_seed: int,
    num_warmup: int,
    num_samples: int,
    num_kept: int,
    target_accept_prob: float = TARGET_ACCEPT_PROB,
) -> dict:
    """Run replicates start..stop-1 in spawned worker processes; summarize the out_dir.

    A replicate whose JSON file exists is skipped, so a stopped run resumes. Each worker
    sees one host device: XLA_FLAGS is set before the pool exists, so the spawned
    workers inherit it, and restored afterwards. A failed replicate writes no file and
    is listed under "failures", so a rerun retries it.

    On a host with an NVIDIA device every worker would share the first GPU and its
    preallocated memory, so ``workers`` must be 1 there. One replicate per GPU comes from
    one process per GPU, each under its own ``CUDA_VISIBLE_DEVICES`` with its own
    ``start`` and ``stop``, all writing to one ``out_dir``: a replicate's seeds depend
    only on ``base_seed`` and its index.
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
            raise ValueError(
                "on a GPU host every worker would share one GPU: run one worker per "
                "process and shard replicates across GPUs with start, stop, and "
                "CUDA_VISIBLE_DEVICES"
            )
        saved_flags = os.environ.get("XLA_FLAGS")
        os.environ["XLA_FLAGS"] = _single_device_flags(saved_flags)
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
                        layout_name=layout_name,
                        panel=panel,
                        num_warmup=num_warmup,
                        num_samples=num_samples,
                        num_kept=num_kept,
                        target_accept_prob=target_accept_prob,
                    ): index
                    for index in pending
                }
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        record = future.result()
                        _require_finite_recovery(index, record["recovery"])
                        _write_json(record, replicate_path(out_dir, index))
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


def _single_device_flags(flags: str | None) -> str:
    kept = [
        flag for flag in (flags or "").split() if not flag.startswith(HOST_DEVICE_FLAG)
    ]
    return " ".join([*kept, f"{HOST_DEVICE_FLAG}=1"])


def _initialize_worker() -> None:
    numpyro.enable_x64()
    if jax.default_backend() == "cpu" and jax.local_device_count() != 1:
        raise RuntimeError(
            f"an SBC worker must see one host device, not {jax.local_device_count()}"
        )


def _write_json(record: dict, path: Path) -> None:
    """Write through a temporary file, so a killed process never leaves a partial file."""
    text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    # The process id keeps hand-started shards that write one out_dir from sharing a name.
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def read_replicates(out_dir: Path) -> list[dict]:
    """Every replicate record in out_dir, by index."""
    return [
        json.loads(path.read_text()) for path in sorted(out_dir.glob(REPLICATE_GLOB))
    ]


def summarize_sbc(records: list[dict]) -> dict:
    """The gamma-ECDF test and recovery of every quantity, and the pre-registered verdict."""
    if not records:
        raise ValueError("no replicate records to summarize")
    settings = records[0]["settings"]
    if any(record["settings"] != settings for record in records):
        raise ValueError("replicate records were run with different settings")
    num_kept = settings["num_kept"]
    alpha = FAMILY_ALPHA / len(QUANTITIES)
    quantities = {}
    for name in QUANTITIES:
        ranks = np.array([record["ranks"][name] for record in records], dtype=np.int64)
        test = ecdf_test(
            ranks, num_kept, alpha, seed=ECDF_SEED, num_simulations=ECDF_SIMULATIONS
        )
        entry = {"ecdf": {key: test[key] for key in ("gamma", "critical")}}
        entry["ecdf"]["passed"] = bool(test["passed"])
        entry["ess_bulk_kept_min"] = _min_or_none(
            record["ess_bulk_kept"][name] for record in records
        )
        if name != LOG_LIKELIHOOD:
            z = np.array([record["recovery"][name]["z"] for record in records])
            contraction = np.array(
                [record["recovery"][name]["contraction"] for record in records]
            )
            entry["recovery"] = {
                "median_contraction": float(np.median(contraction)),
                "share_abs_z_above_3": float(np.mean(np.abs(z) > LARGE_Z)),
            }
            entry["ess_bulk_min"] = _min_or_none(
                record["ess_bulk"][name] for record in records
            )
        quantities[name] = entry
    recovery_checks = {
        name: quantities[name]["recovery"]["median_contraction"] >= CONTRACTION_MIN
        and quantities[name]["recovery"]["share_abs_z_above_3"] <= LARGE_Z_SHARE_MAX
        for name in RECOVERY_QUANTITIES
    }
    ecdf_passed = all(entry["ecdf"]["passed"] for entry in quantities.values())
    return {
        "settings": settings,
        "num_replicates": len(records),
        "indices": [record["index"] for record in records],
        "num_quantities": len(QUANTITIES),
        "alpha": alpha,
        "quantities": quantities,
        "ecdf_failures": [
            name for name, entry in quantities.items() if not entry["ecdf"]["passed"]
        ],
        "recovery_checks": recovery_checks,
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
        "passed": ecdf_passed and all(recovery_checks.values()),
    }


def write_summary(out_dir: Path) -> dict:
    """Summarize every replicate in out_dir and write it to out_dir / SUMMARY_FILE."""
    summary = summarize_sbc(read_replicates(out_dir))
    _write_json(summary, out_dir / SUMMARY_FILE)
    return summary


def _min_or_none(values) -> float | None:
    values = list(values)
    if any(value is None for value in values):
        return None
    return float(min(values))
