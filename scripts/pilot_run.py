"""Run the Stage 6 aggregate pilot: prior predictive, fit, SBC, and the two decisions.

Every command assembles the pilot panel from the built Stage 3, 4, and 5 artifacts
(``ces_revisions.pilot.assembly.read_artifacts``). ``prior-predictive`` applies Req 15's
monthly constraint to complete panels simulated from the prior; ``fit`` samples the pilot
with four dispersed NUTS chains and writes a run directory with the April 2020 check;
``sbc`` and ``sbc-summary`` run and summarize simulation-based calibration, one JSON file
per replicate so a stopped run resumes; ``compare-seasonal`` compares fitted runs of the
seasonal layouts by blocked PSIS-LOO, residual autocorrelation, and measured cost, and
applies the pre-registered layout rule (``ces_revisions.pilot.seasonal``); ``choose-gpu``
applies the GPU rule (``ces_revisions.pilot.gpu_choice``) to probe-grid records; ``shares``
writes a small JSON of a fitted run's posterior news shares, and ``shares-report`` renders
the Stage 2 findings table from that JSON alone.

XLA reads its flags when JAX first touches a device, so they are set here before any JAX
operation: XLA's deterministic GPU operations on an NVIDIA host, and on a CPU host four
devices for the commands that run four chains or vectorize prior draws. SBC workers set
their own single device.
"""

import os
import sys

from ces_revisions.devices import DETERMINISTIC_GPU_FLAG, has_nvidia_device

FOUR_DEVICE_COMMANDS = ("fit", "prior-predictive")
HOST_DEVICE_FLAG = "--xla_force_host_platform_device_count"


def _configure_xla(argv: list[str]) -> None:
    """Append this command's XLA flags to XLA_FLAGS, keeping any already set.

    SBC's spawned workers re-run this module's top level with the parent's arguments,
    so it only ever appends, and leaves a device count already present alone.
    """
    flags = os.environ.get("XLA_FLAGS", "").split()
    if has_nvidia_device():
        if DETERMINISTIC_GPU_FLAG not in flags:
            flags.append(DETERMINISTIC_GPU_FLAG)
    elif (
        argv
        and argv[0] in FOUR_DEVICE_COMMANDS
        and not any(flag.startswith(HOST_DEVICE_FLAG) for flag in flags)
    ):
        flags.append(f"{HOST_DEVICE_FLAG}=4")
    os.environ["XLA_FLAGS"] = " ".join(flags)


_configure_xla(sys.argv[1:])

import numpyro

numpyro.enable_x64()

import argparse
import json
import time
from datetime import date
from pathlib import Path

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
from archive_inventory import replace_generated
from scipy.stats import chi2

from ces_revisions.kalman import kalman_filter
from ces_revisions.pilot.assembly import build_aggregate_panel, read_artifacts
from ces_revisions.pilot.checks import april_2020
from ces_revisions.pilot.dimensions import (
    pilot_dimensions,
    stage_dimensions,
)
from ces_revisions.pilot.gpu_choice import STEPS_PER_ITERATION, choose_gpu
from ces_revisions.pilot.layout import LAYOUTS, SEASONAL_PRIOR_SCALE, Layout
from ces_revisions.pilot.model import (
    INITIAL_MEAN,
    INITIAL_SD,
    SAMPLED_SITES,
    TARGET_ACCEPT_PROB,
    pilot_model,
    pilot_ssm,
    sample_prior,
    step_log_likelihood,
)
from ces_revisions.pilot.news_shares import (
    INTERVAL_PROB,
    SHARES,
    share_draws,
    shares_block,
    summarize,
)
from ces_revisions.pilot.panel import CLOSING_STAGES, AggregatePanel
from ces_revisions.pilot.sbc import FIRST_MONTH, LAST_MONTH, run_sbc, write_summary
from ces_revisions.pilot.seasonal import (
    REFERENCE_LAYOUT,
    choose_layout,
    grid_seconds,
    ljung_box_rejections,
)
from ces_revisions.pilot.simulate import (
    monthly_constraint,
    prior_predictive,
)
from ces_revisions.runs.fit import (
    DIAGNOSTICS_FILE,
    INFERENCE_DATA_FILE,
    PROVENANCE_FILE,
    RUNS_DIR,
    RunSettings,
    new_run_dir,
    run_nuts,
    write_run,
)
from ces_revisions.runs.inference_data import read_inference_data
from ces_revisions.runs.provenance import array_sha256
from ces_revisions.vintages.raw import ROOT, file_sha256

PRIOR_PREDICTIVE_SEED = sum(map(ord, "ces-revisions-stage-6-prior-predictive"))
FIT_SEED = sum(map(ord, "ces-revisions-stage-6-pilot-fit"))
# The prior draws that place the dispersed starts: a property of the prior, not the run.
INIT_DRAWS_SEED = sum(map(ord, "ces-revisions-stage-6-init-prior-draws"))
NUM_INIT_DRAWS = 4000
SBC_SEED = sum(map(ord, "ces-revisions-stage-6-sbc"))
SBC_REPLICATES = 100  # the registered count: sbc runs it, sbc-summary expects it
APRIL_2020_FILE = "april_2020.json"
MODEL_NAME = "stage6-pilot"
# The seasonal comparison's residual check: Ljung-Box over a year of lags.
LJUNG_BOX_LAGS = 12
LOO_BLOCKS = ("panel_month", "panel_benchmark_year")
RULE_BLOCK = "panel_benchmark_year"  # the seasonal rule's close-to-independent blocks
FINDINGS = ROOT / "specs" / "ces-revision-drivers-stage-2-findings.md"
SHARES_BLOCK_NAME = "stage-2-shares"
REFIT_COMMAND = (
    "uv run python scripts/pilot_run.py fit --layout dummy --label stage6-pilot"
)


def load_panel() -> AggregatePanel:
    return build_aggregate_panel(**read_artifacts())


def panel_arrays(panel: AggregatePanel) -> dict[str, np.ndarray]:
    """The arrays the model reads from the panel, hashed into each run's provenance."""
    return {
        "y": panel.y,
        "anchor_offset": panel.anchor_offset,
        "pre_2003": panel.pre_2003,
        "crisis": panel.crisis,
        "benchmark_year": panel.benchmark_year,
    }


def model_record(layout_name: str) -> dict:
    layout = LAYOUTS[layout_name]
    return {
        "name": MODEL_NAME,
        "layout": layout.name,
        "harmonics": layout.harmonics,
        "state_names": list(layout.state_names),
        "sampled_sites": list(SAMPLED_SITES),
        "seasonal_prior_scale": SEASONAL_PRIOR_SCALE[layout.name],
        "initial_mean": {name: INITIAL_MEAN[name] for name in layout.state_names},
        "initial_sd": {name: INITIAL_SD[name] for name in layout.state_names},
        "dimensions": pilot_dimensions(layout.name),
    }


def write_json(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False)
    path.write_text(text + "\n")


def run_prior_predictive(arguments: argparse.Namespace) -> int:
    layout = LAYOUTS[arguments.layout]
    panel = load_panel()
    started = time.perf_counter()
    result = prior_predictive(
        panel, layout, arguments.draws, arguments.seed, doubled=arguments.doubled
    )
    seconds = time.perf_counter() - started
    constraint = monthly_constraint(result)
    statistics = result["statistics"]
    record = {
        "layout": layout.name,
        "seed": arguments.seed,
        "doubled": arguments.doubled,
        "num_draws": arguments.draws,
        "seconds": seconds,
        "monthly_constraint": constraint,
        "median_abs_quantiles": {
            str(q): float(np.quantile(statistics["median_abs"], q))
            for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "statistics": {name: values.tolist() for name, values in statistics.items()},
    }
    write_json(record, arguments.out)
    print(json.dumps({"seconds": seconds, **constraint}))
    return 0


def run_fit(arguments: argparse.Namespace) -> int:
    layout = LAYOUTS[arguments.layout]
    panel = load_panel()
    prior_draws = {
        name: np.asarray(values)
        for name, values in sample_prior(
            jax.random.PRNGKey(INIT_DRAWS_SEED), NUM_INIT_DRAWS, layout
        ).items()
    }
    settings = RunSettings(
        seed=arguments.seed,
        num_chains=arguments.chains,
        num_warmup=arguments.warmup,
        num_samples=arguments.samples,
        max_tree_depth=arguments.max_tree_depth,
        target_accept_prob=arguments.target_accept,
    )
    result = run_nuts(
        pilot_model,
        (panel, layout),
        settings=settings,
        prior_draws=prior_draws,
        step_fn=lambda params: step_log_likelihood(params, panel, layout),
        reference_months=panel.reference_months,
        benchmark_year=panel.benchmark_year,
        observed_any=panel.observed_any(),
        model_record={
            **model_record(layout.name),
            "init_draws": {"seed": INIT_DRAWS_SEED, "num_draws": NUM_INIT_DRAWS},
        },
        arrays=panel_arrays(panel),
    )
    # The draws are on disk before the smoother runs, so a failure there loses nothing.
    run_dir = write_run(
        result, new_run_dir(arguments.label, arguments.seed, arguments.runs_dir)
    )
    posterior = result.idata["posterior"]
    samples = {name: np.asarray(posterior[name]) for name in SAMPLED_SITES}
    check = april_2020(samples, panel, layout, seed=arguments.seed)
    write_json(check, run_dir / APRIL_2020_FILE)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "sampling_seconds": result.sampling_seconds,
                "meets_req18": result.diagnostics["meets_req18"],
                "worst": result.diagnostics["worst"],
                "april_2020_ratio": check["ratio"],
                "april_2020_passed": check["passed"],
            }
        )
    )
    return 0


def run_sbc_command(arguments: argparse.Namespace) -> int:
    panel = load_panel().window(arguments.first, arguments.last)
    summary = run_sbc(
        arguments.out_dir,
        layout_name=arguments.layout,
        panel=panel,
        start=arguments.start,
        stop=arguments.replicates if arguments.stop is None else arguments.stop,
        workers=arguments.workers,
        base_seed=arguments.seed,
        num_warmup=arguments.warmup,
        num_samples=arguments.samples,
        num_kept=arguments.keep,
        target_accept_prob=arguments.target_accept,
    )
    print(json.dumps(brief_summary(summary)))
    return 0


def run_sbc_summary(arguments: argparse.Namespace) -> int:
    """Summarize the replicates in a directory, and fail when any expected one is missing.

    A replicate that failed leaves no file, so the summary's verdict covers only the
    rest. Naming the missing indices keeps a short run from reading as a passed one.
    """
    summary = write_summary(arguments.out_dir)
    missing = sorted(set(range(arguments.replicates)) - set(summary["indices"]))
    print(json.dumps({**brief_summary(summary), "missing": missing}))
    return 1 if missing else 0


def brief_summary(summary: dict) -> dict:
    keys = (
        "num_replicates",
        "passed",
        "ecdf_failures",
        "recovery_checks",
        "divergent_replicates",
        "replicate_seconds_median",
        "failures",
        "skipped",
    )
    return {key: summary[key] for key in keys if key in summary}


def standardized_errors(
    params: dict[str, float], panel: AggregatePanel, layout: Layout
) -> np.ndarray:
    """(T, 3) one-step prediction errors of the closing cells over their one-step SDs.

    NaN where the cell is not observed.
    """
    filtered = kalman_filter(pilot_ssm(params, panel, layout), jnp.asarray(panel.y))
    closing = len(CLOSING_STAGES)
    mean = np.asarray(filtered.one_step_mean)[:, :closing]
    variance = np.diagonal(np.asarray(filtered.one_step_cov), axis1=1, axis2=2)
    return (panel.y[:, :closing] - mean) / np.sqrt(variance[:, :closing])


def ljung_box(errors: np.ndarray, lags: int = LJUNG_BOX_LAGS) -> dict:
    """Ljung-Box Q over lags 1..lags of a monthly series with NaN gaps.

    The lag-h autocorrelation sums over every pair of observed cells h months apart,
    over the observed cells' sum of squares about their mean; n counts observed cells.
    """
    observed = ~np.isnan(errors)
    count = int(observed.sum())
    centered = np.where(observed, errors - errors[observed].mean(), 0.0)
    lag = np.arange(1, lags + 1)
    autocorrelation = np.array(
        [np.sum(centered[h:] * centered[:-h]) for h in lag]
    ) / np.sum(centered**2)
    q = count * (count + 2) * np.sum(autocorrelation**2 / (count - lag))
    return {
        "q": float(q),
        "p_value": float(chi2.sf(q, lags)),
        "lags": lags,
        "num_observed": count,
        "autocorrelation": autocorrelation.tolist(),
    }


def read_run(run_dir: Path) -> dict:
    return {
        "idata": read_inference_data(run_dir / INFERENCE_DATA_FILE),
        "diagnostics": json.loads((run_dir / DIAGNOSTICS_FILE).read_text()),
        "provenance": json.loads((run_dir / PROVENANCE_FILE).read_text()),
    }


def loo_block(idata, block: str) -> tuple[dict, np.ndarray]:
    """PSIS-LOO over one log-likelihood block: its summary and the pointwise elpd.

    Blocks come in the run's coordinate order, the same for every run of one panel.
    """
    loo = az.loo(idata, var_name=block, pointwise=True)
    pareto_k = np.asarray(loo.pareto_k)
    return {
        "elpd": float(loo.elpd),
        "se": float(loo.se),
        "p": float(loo.p),
        "num_blocks": int(loo.n_data_points),
        "pareto_k_max": float(pareto_k.max()),
        "pareto_k_good": float(loo.good_k),
        "pareto_k_above_good": int(np.sum(pareto_k > loo.good_k)),
        "pareto_k": pareto_k.tolist(),
    }, np.asarray(loo.elpd_i)


def seasonal_candidates(entries: list[dict], grid: dict) -> dict[str, dict]:
    """What the seasonal rule reads about each fitted layout, keyed by layout."""
    return {
        entry["layout"]: {
            "states": entry["dimensions"]["stage9_low"],
            "meets_req18": entry["meets_req18"],
            "april_2020_passed": entry["april_2020_passed"],
            "ljung_box_q": {
                stage: test["q"] for stage, test in entry["ljung_box"].items()
            },
            "ljung_box_rejections": ljung_box_rejections(entry["ljung_box"]),
            "seconds": grid_seconds(grid, entry["layout"]),
        }
        for entry in entries
    }


def run_compare_seasonal(arguments: argparse.Namespace) -> int:
    panel = load_panel()
    data_hash = array_sha256(panel_arrays(panel))
    grid = json.loads(arguments.grid.read_text())
    entries, pointwise = [], []
    for run_dir in arguments.run_dirs:
        run = read_run(run_dir)
        provenance = run["provenance"]
        if provenance["arrays"]["sha256"] != data_hash:
            raise ValueError(f"{run_dir} was fit to other data than the current panel")
        layout = LAYOUTS[provenance["model"]["layout"]]
        blocks, elpd = {}, {}
        for block in LOO_BLOCKS:
            blocks[block], elpd[block] = loo_block(run["idata"], block)
        posterior = run["idata"]["posterior"]
        medians = {name: float(np.median(posterior[name])) for name in SAMPLED_SITES}
        errors = standardized_errors(medians, panel, layout)
        diagnostics = run["diagnostics"]
        seconds = provenance["sampling"]["timings"]["sampling_seconds"]
        ess_bulk_min = diagnostics["worst"]["ess_bulk_min"]
        entries.append(
            {
                "run_dir": str(run_dir),
                "layout": layout.name,
                "loo": blocks,
                "ljung_box": {
                    stage: ljung_box(errors[:, k])
                    for k, stage in enumerate(CLOSING_STAGES)
                },
                "posterior_median": medians,
                "sampling_seconds": seconds,
                "ess_bulk_min": ess_bulk_min,
                "ess_bulk_min_per_second": None
                if ess_bulk_min is None
                else ess_bulk_min / seconds,
                "meets_req18": diagnostics["meets_req18"],
                "april_2020_passed": json.loads(
                    (run_dir / APRIL_2020_FILE).read_text()
                )["passed"],
                "dimensions": {
                    "pilot": pilot_dimensions(layout.name)["states"],
                    "stage7": stage_dimensions(layout.name, 7)["states"],
                    "stage9_low": stage_dimensions(layout.name, 9, "low")["states"],
                    "stage9_high": stage_dimensions(layout.name, 9, "high")["states"],
                },
            }
        )
        pointwise.append(elpd[RULE_BLOCK])
    layouts = [entry["layout"] for entry in entries]
    if REFERENCE_LAYOUT not in layouts or len(set(layouts)) != len(layouts):
        raise ValueError(
            "compare-seasonal needs one run per layout, the dummy included"
        )
    candidates = seasonal_candidates(entries, grid)
    rule = choose_layout(
        candidates,
        dict(zip(layouts, pointwise, strict=True)),
        {
            entry["layout"]: np.asarray(entry["loo"][RULE_BLOCK]["pareto_k"])
            for entry in entries
        },
    )
    write_json(
        {
            "panel_sha256": data_hash,
            "grid": {"path": str(arguments.grid), "label": grid["label"]},
            "runs": entries,
            "candidates": candidates,
            "rule": rule,
        },
        arguments.out,
    )
    print(json.dumps({"layout": rule["layout"], "reliable": rule["reliable"]}))
    return 0


def run_choose_gpu(arguments: argparse.Namespace) -> int:
    records = [json.loads(path.read_text()) for path in arguments.records]
    result = choose_gpu(records, arguments.layout, arguments.steps_per_iteration)
    write_json(
        {
            "layout": arguments.layout,
            "records": [str(path) for path in arguments.records],
            **result,
        },
        arguments.out,
    )
    choices = {
        f"{stage['stage']} {stage['wedge']}": stage["choice"]
        and stage["choice"]["host"]
        for stage in result["stages"]
    }
    print(json.dumps({"status": result["status"], "choices": choices}))
    return 0


def shares_record(run_dir: Path, panel_sha256: str) -> dict:
    """A fitted run's posterior news shares, summarized, for the panel it was fit to.

    The provenance is read first, so a run fit to other data stops before its
    InferenceData file opens.
    """
    provenance = json.loads((run_dir / PROVENANCE_FILE).read_text())
    if provenance["arrays"]["sha256"] != panel_sha256:
        raise ValueError(f"{run_dir} was fit to other data than the current panel")
    inference_data = run_dir / INFERENCE_DATA_FILE
    posterior = read_inference_data(inference_data)["posterior"].to_dataset()
    draws = share_draws(posterior)
    return {
        "run_dir": str(run_dir),
        "inference_data_sha256": file_sha256(inference_data),
        "panel_sha256": panel_sha256,
        "num_draws": len(draws["share_s"]),
        "interval_prob": INTERVAL_PROB,
        "sampling_seconds": provenance["sampling"]["timings"]["sampling_seconds"],
        **{key: summarize(values) for key, values in draws.items()},
    }


def run_shares(arguments: argparse.Namespace) -> int:
    if not arguments.run_dir.is_dir():
        print(
            f"{arguments.run_dir} is missing, and data/runs/ is gitignored. Refit "
            f"the pilot with `{REFIT_COMMAND}`, then run shares on the new run "
            "directory.",
            file=sys.stderr,
        )
        return 2
    panel_sha256 = array_sha256(panel_arrays(load_panel()))
    record = shares_record(arguments.run_dir, panel_sha256)
    write_json(record, arguments.out)
    print(json.dumps({key: record[key] for key in SHARES}))
    return 0


def run_shares_report(arguments: argparse.Namespace) -> int:
    record = json.loads(arguments.record.read_text())
    document = arguments.findings.read_text(encoding="utf-8")
    # Blank lines keep the table a block of its own beside the HTML-comment markers.
    body = f"\n{shares_block(record)}\n\n"
    document = replace_generated(document, SHARES_BLOCK_NAME, body)
    arguments.findings.write_text(document, encoding="utf-8")
    return 0


def month(text: str) -> date:
    """A YYYY-MM argument as the first day of that month."""
    return date.fromisoformat(f"{text}-01")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    prior = commands.add_parser("prior-predictive", help="Req 15's monthly constraint")
    prior.add_argument("--layout", choices=sorted(LAYOUTS), default="dummy")
    prior.add_argument("--draws", type=int, default=2000)
    prior.add_argument("--seed", type=int, default=PRIOR_PREDICTIVE_SEED)
    prior.add_argument("--doubled", action="store_true")
    prior.add_argument("--out", type=Path, required=True)
    prior.set_defaults(handler=run_prior_predictive)

    fit = commands.add_parser("fit", help="four dispersed NUTS chains and a run dir")
    fit.add_argument("--layout", choices=sorted(LAYOUTS), default="dummy")
    fit.add_argument("--seed", type=int, default=FIT_SEED)
    fit.add_argument("--chains", type=int, default=4)
    fit.add_argument("--warmup", type=int, default=1000)
    fit.add_argument("--samples", type=int, default=1000)
    fit.add_argument("--max-tree-depth", type=int, default=10)
    fit.add_argument("--target-accept", type=float, default=TARGET_ACCEPT_PROB)
    fit.add_argument("--label", default="stage6-pilot")
    fit.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    fit.set_defaults(handler=run_fit)

    sbc = commands.add_parser("sbc", help="simulation-based calibration replicates")
    sbc.add_argument("--layout", choices=sorted(LAYOUTS), default="dummy")
    sbc.add_argument("--replicates", type=int, default=SBC_REPLICATES)
    sbc.add_argument("--start", type=int, default=0)
    sbc.add_argument("--stop", type=int, default=None)
    sbc.add_argument("--workers", type=int, default=6)
    sbc.add_argument("--warmup", type=int, default=500)
    sbc.add_argument("--samples", type=int, default=990)
    sbc.add_argument("--keep", type=int, default=99)
    sbc.add_argument("--target-accept", type=float, default=TARGET_ACCEPT_PROB)
    sbc.add_argument("--seed", type=int, default=SBC_SEED)
    sbc.add_argument("--first", type=month, default=FIRST_MONTH)
    sbc.add_argument("--last", type=month, default=LAST_MONTH)
    sbc.add_argument("--out-dir", type=Path, required=True)
    sbc.set_defaults(handler=run_sbc_command)

    summary = commands.add_parser("sbc-summary", help="summarize an SBC directory")
    summary.add_argument("--out-dir", type=Path, required=True)
    summary.add_argument(
        "--replicates",
        type=int,
        default=SBC_REPLICATES,
        help="how many replicates the run should hold; exit 1 and name any missing",
    )
    summary.set_defaults(handler=run_sbc_summary)

    compare = commands.add_parser("compare-seasonal", help="compare fitted layouts")
    compare.add_argument("run_dirs", type=Path, nargs="+", metavar="RUN_DIR")
    compare.add_argument("--out", type=Path, required=True)
    compare.add_argument(
        "--grid", type=Path, required=True, help="gpu_probe_grid.py record for cost"
    )
    compare.set_defaults(handler=run_compare_seasonal)

    gpu = commands.add_parser("choose-gpu", help="choose the GPU type and count")
    gpu.add_argument("records", type=Path, nargs="+", metavar="GRID_RECORD")
    gpu.add_argument("--layout", choices=sorted(LAYOUTS), required=True)
    gpu.add_argument("--out", type=Path, required=True)
    gpu.add_argument(
        "--steps-per-iteration",
        type=int,
        default=STEPS_PER_ITERATION,
        help="leapfrog steps per iteration of the slowest chain, once a stage measures it",
    )
    gpu.set_defaults(handler=run_choose_gpu)

    shares = commands.add_parser("shares", help="the posterior news shares of a run")
    shares.add_argument("run_dir", type=Path, metavar="RUN_DIR")
    shares.add_argument("--out", type=Path, required=True)
    shares.set_defaults(handler=run_shares)

    report = commands.add_parser(
        "shares-report", help="render the Stage 2 shares table from a shares record"
    )
    report.add_argument("record", type=Path, metavar="RECORD")
    report.add_argument("--findings", type=Path, default=FINDINGS)
    report.set_defaults(handler=run_shares_report)

    arguments = parser.parse_args(argv)
    return arguments.handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
