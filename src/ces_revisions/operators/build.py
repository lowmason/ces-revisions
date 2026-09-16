"""Assemble and write deterministic roadmap Stage 5 operator artifacts."""

import json
from dataclasses import dataclass, fields
from pathlib import Path

import polars as pl

from ces_revisions.annual import raw as annual_raw
from ces_revisions.annual.build import build as build_annual
from ces_revisions.operators import SECTOR_ORDER
from ces_revisions.operators.benchmark import build_benchmark_fixtures
from ces_revisions.operators.post_march import (
    build_post_march_components,
    build_recoverability,
)
from ces_revisions.operators.seasonal import build_b_to_m_components
from ces_revisions.vintages import raw as vintage_raw
from ces_revisions.vintages.build import build as build_vintages

OUTPUT_DIR = vintage_raw.ROOT / "data" / "operators"


@dataclass(frozen=True)
class OperatorBuild:
    benchmark_fixtures: pl.DataFrame
    post_march_components: pl.DataFrame
    recoverability: pl.DataFrame
    b_to_m_components: pl.DataFrame


def assemble(
    levels: pl.DataFrame,
    stage_labels: pl.DataFrame,
    benchmarks: pl.DataFrame,
    reconstruction_events: pl.DataFrame,
    birth_death: pl.DataFrame,
) -> OperatorBuild:
    recoverability = build_recoverability(levels, birth_death)
    return OperatorBuild(
        benchmark_fixtures=build_benchmark_fixtures(
            levels, benchmarks, reconstruction_events
        ),
        post_march_components=build_post_march_components(levels, birth_death),
        recoverability=recoverability,
        b_to_m_components=build_b_to_m_components(levels, stage_labels, recoverability),
    )


def build() -> tuple[OperatorBuild, pl.DataFrame]:
    vintage = build_vintages()
    annual = build_annual()
    levels = vintage.panel.filter(
        (pl.col("source") == "cesvinall") & (pl.col("measure") == "level")
    )
    result = assemble(
        levels,
        vintage.stage_labels,
        annual.benchmarks,
        annual.reconstruction_events,
        annual.birth_death,
    )
    return result, levels


def _aggregation_check(levels: pl.DataFrame) -> dict:
    frame = levels.filter(pl.col("value_thousands").is_not_null())
    keys = ["release_month", "seasonal_status", "reference_month"]
    total = frame.filter(pl.col("sector") == "00").select(
        *keys, total=pl.col("value_thousands")
    )
    parts = (
        frame.filter(pl.col("sector").is_in(SECTOR_ORDER))
        .group_by(keys)
        .agg(parts=pl.col("value_thousands").sum(), sectors=pl.len())
    )
    checked = total.join(parts, on=keys).with_columns(
        residual=pl.col("total") - pl.col("parts")
    )
    if checked["sectors"].ne(len(SECTOR_ORDER)).any():
        raise ValueError("an aggregation cell does not contain eleven sectors")
    return {
        "cells": checked.height,
        "max_abs_residual_thousands": float(checked["residual"].abs().max()),
    }


def _checks(result: OperatorBuild, levels: pl.DataFrame) -> dict:
    wedge = result.benchmark_fixtures
    post = result.post_march_components
    recovery = result.recoverability
    identity = result.b_to_m_components["identity_residual_thousands"].drop_nulls()
    return {
        "aggregation": _aggregation_check(levels),
        "wedge": {
            "years": wedge["benchmark_year"].n_unique(),
            "cells": wedge.height,
            "reconstruction_supported_cells": wedge.filter(
                pl.col("reconstruction_supported")
            ).height,
            "max_abs_reproduction_residual_thousands": float(
                (
                    wedge["reproduced_level_thousands"]
                    - wedge["benchmark_level_thousands"]
                )
                .abs()
                .max()
            ),
            "max_abs_non_event_rounding_residual_thousands": float(
                wedge.filter(~pl.col("reconstruction_supported"))[
                    "rounding_residual_thousands"
                ]
                .abs()
                .max()
            ),
        },
        "post_march": {
            "years": post["benchmark_year"].n_unique(),
            "april_october_cells": post.filter(
                ~pl.col("additional_sample_receipts")
            ).height,
            "november_december_cells": post.filter(
                pl.col("additional_sample_receipts")
            ).height,
            "max_abs_reproduction_residual_thousands": float(
                post["residual_thousands"].abs().max()
            ),
        },
        "recoverability": {
            "cumulative_job_change_years": recovery.filter(
                pl.col("representation") == "cumulative_job_change"
            ).height,
            "inferred_years": recovery.filter(
                pl.col("series_status") == "inferred"
            ).height,
            "recovered_years": recovery.filter(
                pl.col("series_status") == "recovered"
            ).height,
        },
        "seasonal_mapping": {
            "max_abs_identity_residual_thousands": float(identity.abs().max())
        },
    }


def operator_test_record(result: OperatorBuild, levels: pl.DataFrame) -> dict:
    artifacts = {
        field.name: {
            "rows": getattr(result, field.name).height,
            "sha256": annual_raw.content_sha256(getattr(result, field.name)),
        }
        for field in fields(result)
    }
    return {
        "schema_version": 1,
        "input_manifests": {
            "stage3": vintage_raw.file_sha256(
                vintage_raw.RAW_DIR / vintage_raw.MANIFEST
            ),
            "stage4": annual_raw.file_sha256(annual_raw.RAW_DIR / annual_raw.MANIFEST),
        },
        "artifacts": artifacts,
        "checks": _checks(result, levels),
    }


def write(
    result: OperatorBuild,
    levels: pl.DataFrame,
    out_dir: Path = OUTPUT_DIR,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    for field in fields(result):
        getattr(result, field.name).write_parquet(out_dir / f"{field.name}.parquet")
    record = operator_test_record(result, levels)
    (out_dir / "operator-test-results.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record
