"""Run NUTS on a model and write its run directory.

A run starts each chain at a dispersed point placed from the caller's prior draws, samples
with NumPyro's NUTS at the run's tree depth and target acceptance, attaches the blocked log
likelihood to ArviZ's DataTree, checks the draws against Req 18, and records its provenance.
Provenance that reads the checkout is taken before sampling, so a missing input fails the
run before the expensive part. ``write_run`` stores the DataTree, the diagnostics, and the
provenance in a new directory that ``new_run_dir`` names and never reuses.
"""

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path

import arviz as az
import jax
import numpy as np
from numpyro.infer import MCMC, NUTS

from ces_revisions.devices import chain_method
from ces_revisions.runs.diagnostics import REQ18, diagnostics_record
from ces_revisions.runs.inference_data import (
    attach_log_likelihood,
    step_log_likelihood_draws,
    write_inference_data,
)
from ces_revisions.runs.inits import dispersed_init_params
from ces_revisions.runs.provenance import run_provenance
from ces_revisions.vintages import raw as vintage_raw

RUNS_DIR = vintage_raw.ROOT / "data" / "runs"
EXTRA_FIELDS = (
    "diverging",
    "energy",
    "num_steps",
    "accept_prob",
    "adapt_state.step_size",
    "potential_energy",
)
INFERENCE_DATA_FILE = "inference_data.nc"
DIAGNOSTICS_FILE = "diagnostics.json"
PROVENANCE_FILE = "provenance.json"


@dataclass(frozen=True)
class RunSettings:
    seed: int
    num_chains: int = 4
    num_warmup: int = 1000
    num_samples: int = 1000
    max_tree_depth: int = 10
    target_accept_prob: float = 0.8


@dataclass(frozen=True)
class RunResult:
    idata: object  # xarray DataTree
    diagnostics: dict
    provenance: dict
    sampling_seconds: float  # compilation, warmup, and sampling, until the draws exist


def run_nuts(
    model,
    model_args: tuple,
    *,
    settings: RunSettings,
    prior_draws: dict,
    step_fn: Callable,
    reference_months: Sequence[date],
    benchmark_year: np.ndarray,
    observed_any: np.ndarray,
    model_record: dict,
    arrays: dict[str, np.ndarray],
    root: Path = vintage_raw.ROOT,
) -> RunResult:
    """Sample ``model(*model_args)`` with NUTS from dispersed starts; diagnose and record.

    ``step_fn`` maps a dict of sampled sites to the model's (T,) per-step log likelihood;
    ``arrays`` are the data arrays the model saw, hashed into the provenance.
    """
    method = chain_method(settings.num_chains)
    init_params, init_record = dispersed_init_params(
        model, model_args, prior_draws, settings.num_chains, settings.seed
    )
    provenance = run_provenance(
        root=root,
        seeds={"run": settings.seed, "init": settings.seed, "mcmc_key": settings.seed},
        model=model_record,
        init_record=init_record,
        arrays=arrays,
        settings={
            **asdict(settings),
            "chain_method": method,
            "extra_fields": list(EXTRA_FIELDS),
        },
    )
    mcmc = MCMC(
        NUTS(
            model,
            max_tree_depth=settings.max_tree_depth,
            target_accept_prob=settings.target_accept_prob,
        ),
        num_warmup=settings.num_warmup,
        num_samples=settings.num_samples,
        num_chains=settings.num_chains,
        chain_method=method,
        progress_bar=False,
    )
    started_utc = datetime.now(UTC)
    started = time.perf_counter()
    mcmc.run(
        jax.random.PRNGKey(settings.seed),
        *model_args,
        init_params=init_params,
        extra_fields=EXTRA_FIELDS,
    )
    samples = jax.block_until_ready(mcmc.get_samples(group_by_chain=True))
    sampling_seconds = time.perf_counter() - started
    finished_utc = datetime.now(UTC)

    idata = az.from_numpyro(mcmc).map_over_datasets(lambda ds: ds.as_numpy())
    started = time.perf_counter()
    step_log_likelihood = step_log_likelihood_draws(step_fn, samples)
    log_likelihood_seconds = time.perf_counter() - started
    idata = attach_log_likelihood(
        idata, step_log_likelihood, reference_months, benchmark_year, observed_any
    )
    thresholds = replace(REQ18, max_tree_depth=settings.max_tree_depth)
    diagnostics = diagnostics_record(idata, thresholds)

    adapt_state = mcmc.last_state.adapt_state
    num_steps = np.asarray(mcmc.get_extra_fields(group_by_chain=True)["num_steps"])
    provenance["sampling"] = {
        "timings": {
            "init_readback_seconds": init_record["readback_seconds"],
            "sampling_seconds": sampling_seconds,
            "sampling_seconds_covers": "compilation, warmup, and sampling",
            "log_likelihood_seconds": log_likelihood_seconds,
            "started_utc": started_utc.isoformat(timespec="seconds"),
            "finished_utc": finished_utc.isoformat(timespec="seconds"),
        },
        "num_steps_sampling_total": int(num_steps.sum()),
        "inverse_mass_matrix": [
            {
                "sites": list(sites),
                "per_chain": np.asarray(diagonal).tolist(),
                "mean": np.asarray(diagonal).mean(axis=0).tolist(),
            }
            for sites, diagonal in adapt_state.inverse_mass_matrix.items()
        ],
        "final_step_size": np.asarray(adapt_state.step_size).tolist(),
    }
    return RunResult(
        idata=idata,
        diagnostics=diagnostics,
        provenance=provenance,
        sampling_seconds=sampling_seconds,
    )


def write_run(result: RunResult, run_dir: Path) -> Path:
    """Write the DataTree first, then the diagnostics and provenance JSON; return run_dir."""
    write_inference_data(result.idata, run_dir / INFERENCE_DATA_FILE)
    for name, record in (
        (DIAGNOSTICS_FILE, result.diagnostics),
        (PROVENANCE_FILE, result.provenance),
    ):
        text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False)
        (run_dir / name).write_text(text + "\n")
    return run_dir


def new_run_dir(label: str, seed: int, runs_dir: Path = RUNS_DIR) -> Path:
    """Create runs_dir/<label>-seed<seed>-<UTC timestamp>; raise if it already exists."""
    stamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    run_dir = runs_dir / f"{label}-seed{seed}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _utc_now() -> datetime:
    return datetime.now(UTC)
