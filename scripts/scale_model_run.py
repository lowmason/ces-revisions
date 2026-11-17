"""Run the Stage 1 scale model: prior predictive, fits, sensitivity, and SBC.

Every command builds the signed revisions from the built data/panel/ (run
`uv run python scripts/vintage_sources.py build` first). ``prior-predictive`` checks the
priors on the primary series' design against the recommendation's targets; ``fit`` samples
each named series at each named beta prior SD, writes a run directory under data/runs/, and
writes the small evidence JSON (and, at the default prior, the figures and the power-scaling
sensitivity) under docs/runs/stage-1-scale-model/; ``sensitivity`` applies the
recommendation's rule to the evidence of the three beta priors; ``sbc`` and ``sbc-summary``
run and summarize simulation-based calibration on the primary series' design.

XLA reads its flags when JAX first touches a device, so they are set here before any JAX
operation: four host devices for the commands that run four chains. SBC workers set their
own single device, and they re-run this module's top level, so it only ever appends.
"""

import os
import sys

from ces_revisions.devices import DETERMINISTIC_GPU_FLAG, has_nvidia_device

FOUR_DEVICE_COMMANDS = ("fit", "prior-predictive")
HOST_DEVICE_FLAG = "--xla_force_host_platform_device_count"


def _configure_xla(argv: list[str]) -> None:
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
import shutil
from pathlib import Path

import arviz as az
import jax
import matplotlib
import numpy as np
import polars as pl
import xarray as xr
from numpyro.infer import Predictive

from ces_revisions.collection_window import signed_revisions
from ces_revisions.runs.fit import RunSettings, new_run_dir, write_run
from ces_revisions.scale_model.checks import beta_sensitivity, prior_predictive_record
from ces_revisions.scale_model.data import PRIMARY_SERIES, SERIES, scale_data
from ces_revisions.scale_model.fit import (
    MODEL_NAME,
    REPORTED,
    SEED,
    estimate_record,
    fit_series,
    series_label,
)
from ces_revisions.scale_model.model import (
    DEFAULT_PRIORS,
    OBSERVED_SITE,
    SAMPLED_SITES,
    TARGET_ACCEPT_PROB,
    Priors,
    location_scale,
    scale_model,
)
from ces_revisions.scale_model.sbc import SEED as SBC_SEED
from ces_revisions.scale_model.sbc import run_sbc, write_summary
from ces_revisions.vintages.raw import PANEL_DIR, ROOT

matplotlib.use("Agg")

EVIDENCE_DIR = ROOT / "docs" / "runs" / "stage-1-scale-model"
RUNS_DIR = ROOT / "data" / "runs"
SBC_DIR = RUNS_DIR / "stage1-scale-sbc"
PRIOR_PREDICTIVE_SEED = sum(map(ord, f"{MODEL_NAME}-prior-predictive"))
BETA_PRIOR_SDS = (0.05, 0.1, 0.2)
# Parameters drawn in the trace and forest figures; the vectors are summarized apart.
FIGURE_PARAMETERS = (
    "beta",
    "gamma",
    "phi",
    "delta",
    "kappa",
    "lam",
    "tau_y",
    "nu",
    "a",
)
PSENSE_PARAMETERS = ("beta", "scale_ratio", "gamma", "delta", "tau_y", "nu")
REPORT_INTERVAL = {"stats.ci_prob": 0.94, "stats.ci_kind": "hdi"}


def load_revisions() -> pl.DataFrame:
    missing = [
        name
        for name in ("panel.parquet", "collection_days.parquet")
        if not (PANEL_DIR / name).exists()
    ]
    if missing:
        raise SystemExit(
            f"missing {', '.join(missing)} in {PANEL_DIR}: run "
            "`uv run python scripts/vintage_sources.py build` first"
        )
    return signed_revisions(
        pl.read_parquet(PANEL_DIR / "panel.parquet"),
        pl.read_parquet(PANEL_DIR / "collection_days.parquet"),
    )


def parse_series(text: str) -> tuple[str, str]:
    """``NSA:F_S`` to ("NSA", "F_S"); refuses a series the model does not fit."""
    series = tuple(text.split(":"))
    if series not in SERIES:
        raise argparse.ArgumentTypeError(
            f"series must be one of {[':'.join(item) for item in SERIES]}; got {text}"
        )
    return series


def write_json(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False)
    path.write_text(text + "\n")


def evidence_path(evidence_dir: Path, series: tuple[str, str], beta_sd: float) -> Path:
    return evidence_dir / series_label(series) / f"fit-beta{beta_sd}.json"


def run_prior_predictive(arguments: argparse.Namespace) -> int:
    data = scale_data(load_revisions(), PRIMARY_SERIES)
    draws = Predictive(scale_model, num_samples=arguments.draws)(
        jax.random.PRNGKey(PRIOR_PREDICTIVE_SEED), data, DEFAULT_PRIORS
    )
    params = {name: draws[name] for name in SAMPLED_SITES}
    _, log_sigma = jax.vmap(lambda draw: location_scale(draw, data))(params)
    revision = np.asarray(draws[OBSERVED_SITE])
    record = prior_predictive_record(revision, np.asarray(log_sigma), data)
    ratio = np.asarray(draws["scale_ratio"])
    record["seed"] = PRIOR_PREDICTIVE_SEED
    record["series"] = list(PRIMARY_SERIES)
    record["implied_scale_ratio_prior_q025_q50_q975"] = np.quantile(
        ratio, [0.025, 0.5, 0.975]
    ).tolist()
    write_json(record, arguments.out_dir / "prior-predictive.json")

    months = np.array(
        [np.datetime64(month, "ns") for month in data.reference_months],
        dtype="datetime64[ns]",
    )
    tree = xr.DataTree.from_dict(
        {
            "prior_predictive": xr.Dataset(
                {
                    "log10_abs_r": (
                        ("chain", "draw", "reference_month"),
                        np.log10(np.abs(revision) + 1.0)[None],
                    )
                },
                coords={
                    "chain": [0],
                    "draw": np.arange(arguments.draws),
                    "reference_month": months,
                },
            )
        }
    )
    plot = az.plot_ppc_dist(tree, group="prior_predictive", num_samples=100)
    plot.savefig(arguments.out_dir / "prior_predictive.png")
    print(
        json.dumps(
            {
                key: record[key]
                for key in (
                    "abs_revision_quantiles",
                    "share_abs_above_1000",
                    "scale_ratio_lag1_quantiles",
                    "abs_covers_target",
                    "ratio_within_target",
                )
            }
        )
    )
    return 0


def run_fit(arguments: argparse.Namespace) -> int:
    revisions = load_revisions()
    settings = RunSettings(
        seed=arguments.seed,
        num_warmup=arguments.warmup,
        num_samples=arguments.samples,
        target_accept_prob=TARGET_ACCEPT_PROB,
    )
    for series in arguments.series:
        data = scale_data(revisions, series)
        for beta_sd in arguments.beta_sd:
            priors = Priors(beta_sd=beta_sd)
            fit = fit_series(data, priors, settings)
            label = f"stage1-scale-{series_label(series)}-beta{beta_sd}"
            run_dir = write_run(
                fit.result, new_run_dir(label, settings.seed, arguments.runs_dir)
            )
            record = fit_evidence(fit, run_dir)
            write_json(record, evidence_path(arguments.evidence_dir, series, beta_sd))
            if beta_sd == DEFAULT_PRIORS.beta_sd:
                write_figures_and_psense(
                    fit.result.idata, arguments.evidence_dir / series_label(series)
                )
            print(
                json.dumps(
                    {
                        "series": ":".join(series),
                        "beta_sd": beta_sd,
                        "run_dir": str(run_dir),
                        "meets_req18": record["req18"]["meets_req18"],
                        "scale_ratio": record["estimates"]["scale_ratio"],
                    }
                )
            )
    return 0


def fit_evidence(fit, run_dir: Path) -> dict:
    """The small JSON that the report cites for one fit."""
    diagnostics = fit.result.diagnostics
    provenance = fit.result.provenance
    return {
        "run_dir": run_dir.name,
        "series": list(fit.data.series),
        "num_months": len(fit.data.reference_months),
        "first_month": fit.data.reference_months[0].isoformat(),
        "last_month": fit.data.reference_months[-1].isoformat(),
        "priors": provenance["model"]["priors"],
        "deviations_from_recommendation": provenance["model"][
            "deviations_from_recommendation"
        ],
        "settings": provenance["settings"],
        "sampling_seconds": fit.result.sampling_seconds,
        "req18": {
            key: diagnostics[key]
            for key in ("meets_req18", "checks", "worst", "divergences", "bfmi")
        },
        "estimates": estimate_record(fit.result.idata),
        "loo": fit.loo,
        "posterior_predictive_checks": fit.checks,
        "provenance": {
            "git": provenance["git"],
            "array_sha256": provenance["arrays"]["sha256"],
        },
    }


def write_figures_and_psense(idata, out_dir: Path) -> None:
    """The report's figures, the summary table, and the power-scaling sensitivity.

    The report's 94% HDI applies inside this function only: arviz-stats also takes the
    tail-ESS quantiles from stats.ci_prob, so a global change would move the Req 18
    tail-ESS check of every later fit in the same command.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with az.rc_context(REPORT_INTERVAL):
        _write_figures_and_psense(idata, out_dir)


def _write_figures_and_psense(idata, out_dir: Path) -> None:
    parameters = list(FIGURE_PARAMETERS)
    az.summary(idata, var_names=[*REPORTED, "c", "p", "u"]).to_csv(
        out_dir / "summary.csv"
    )
    az.plot_trace_dist(idata, var_names=parameters).savefig(out_dir / "trace.png")
    az.plot_forest(idata, var_names=parameters, combined=True).savefig(
        out_dir / "forest.png"
    )
    az.plot_forest(idata, var_names=["c", "u"], combined=True).savefig(
        out_dir / "forest_month_year.png"
    )
    az.plot_ppc_dist(idata, var_names=[OBSERVED_SITE], num_samples=100).savefig(
        out_dir / "posterior_predictive.png"
    )
    psense = az.psense_summary(idata, var_names=list(PSENSE_PARAMETERS))
    write_json(json.loads(psense.to_json(orient="index")), out_dir / "psense.json")
    az.plot_psense_dist(idata, var_names=["beta", "gamma", "tau_y", "nu"]).savefig(
        out_dir / "psense.png"
    )


def run_sensitivity(arguments: argparse.Namespace) -> int:
    records = {}
    for series in SERIES:
        summaries = {}
        for beta_sd in BETA_PRIOR_SDS:
            path = evidence_path(arguments.evidence_dir, series, beta_sd)
            if path.exists():
                evidence = json.loads(path.read_text())
                summaries[beta_sd] = evidence["estimates"]["beta"]
        if summaries:
            records[":".join(series)] = beta_sensitivity(summaries)
    write_json(records, arguments.evidence_dir / "sensitivity.json")
    print(
        json.dumps(
            {
                name: {
                    "shift": record["shift_0.1_to_0.2"],
                    "passed": record["passed"],
                }
                for name, record in records.items()
            }
        )
    )
    return 0 if all(record["passed"] for record in records.values()) else 1


def run_sbc_command(arguments: argparse.Namespace) -> int:
    data = scale_data(load_revisions(), PRIMARY_SERIES)
    summary = run_sbc(
        arguments.out_dir,
        data=data,
        start=arguments.start,
        stop=arguments.replicates if arguments.stop is None else arguments.stop,
        workers=arguments.workers,
        base_seed=arguments.seed,
        num_warmup=arguments.warmup,
        num_samples=arguments.samples,
        num_kept=arguments.keep,
    )
    print(json.dumps(brief_summary(summary)))
    return 0


def run_sbc_summary(arguments: argparse.Namespace) -> int:
    """Summarize the replicates, copy the summary to the evidence, and name any missing."""
    summary = write_summary(arguments.out_dir)
    missing = sorted(set(range(arguments.replicates)) - set(summary["indices"]))
    print(json.dumps({**brief_summary(summary), "missing": missing}))
    if missing:
        return 1
    arguments.evidence_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        arguments.out_dir / "summary.json", arguments.evidence_dir / "sbc-summary.json"
    )
    return 0


def brief_summary(summary: dict) -> dict:
    return {
        key: summary.get(key)
        for key in (
            "num_replicates",
            "passed",
            "ecdf_failures",
            "beta_recovery",
            "divergent_replicates",
            "failures",
            "replicate_seconds_median",
        )
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = root.add_subparsers(dest="command", required=True)

    prior = commands.add_parser("prior-predictive")
    prior.add_argument("--draws", type=int, default=500)
    prior.add_argument("--out-dir", type=Path, default=EVIDENCE_DIR)
    prior.set_defaults(handler=run_prior_predictive)

    fit = commands.add_parser("fit")
    fit.add_argument("--series", type=parse_series, nargs="+", default=list(SERIES))
    fit.add_argument(
        "--beta-sd", type=float, nargs="+", default=[DEFAULT_PRIORS.beta_sd]
    )
    fit.add_argument("--seed", type=int, default=SEED)
    fit.add_argument("--warmup", type=int, default=1000)
    fit.add_argument("--samples", type=int, default=1000)
    fit.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    fit.add_argument("--evidence-dir", type=Path, default=EVIDENCE_DIR)
    fit.set_defaults(handler=run_fit)

    sensitivity = commands.add_parser("sensitivity")
    sensitivity.add_argument("--evidence-dir", type=Path, default=EVIDENCE_DIR)
    sensitivity.set_defaults(handler=run_sensitivity)

    sbc = commands.add_parser("sbc")
    sbc.add_argument("--out-dir", type=Path, default=SBC_DIR)
    sbc.add_argument("--replicates", type=int, default=100)
    sbc.add_argument("--start", type=int, default=0)
    sbc.add_argument("--stop", type=int, default=None)
    sbc.add_argument("--workers", type=int, default=4)
    sbc.add_argument("--seed", type=int, default=SBC_SEED)
    sbc.add_argument("--warmup", type=int, default=500)
    sbc.add_argument("--samples", type=int, default=1000)
    sbc.add_argument("--keep", type=int, default=100)
    sbc.set_defaults(handler=run_sbc_command)

    summary = commands.add_parser("sbc-summary")
    summary.add_argument("--out-dir", type=Path, default=SBC_DIR)
    summary.add_argument("--replicates", type=int, default=100)
    summary.add_argument("--evidence-dir", type=Path, default=EVIDENCE_DIR)
    summary.set_defaults(handler=run_sbc_summary)
    return root


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
