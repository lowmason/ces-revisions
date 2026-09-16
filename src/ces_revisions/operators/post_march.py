"""Req 10 post-March propagation and link-relative recoverability evidence."""

from datetime import date

import jax
import jax.numpy as jnp
import polars as pl
from jax.experimental.sparse import BCOO

from ces_revisions.operators import OBSERVATION_SECTOR_ORDER
from ces_revisions.operators.benchmark import BENCHMARK_YEARS, benchmark_release_month
from ces_revisions.vintages.months import month_range

ROUNDING_VARIANCE_THOUSANDS2 = 1.0 / 6.0
POST_MARCH_MONTHS = 9
FORMAL_POST_MARCH_MONTHS = 7


def post_march_link_operator(link_relatives: jax.Array) -> BCOO:
    """Map `[March anchor, revised BD]` through fixed matched-sample links."""
    links = jnp.asarray(link_relatives, dtype=jnp.float64)
    if links.ndim != 1 or links.size == 0:
        raise ValueError("link_relatives must be a nonempty vector")
    coefficient = jnp.zeros(links.size + 1, dtype=jnp.float64).at[0].set(1.0)
    rows = []
    for index in range(links.size):
        coefficient = coefficient * links[index]
        coefficient = coefficient.at[index + 1].set(1.0)
        rows.append(coefficient)
    return BCOO.fromdense(jnp.stack(rows))


def post_march_cumulative_operator(month_count: int) -> BCOO:
    """Map `[March anchor, sample changes, revised BD]` to monthly levels."""
    if month_count < 1:
        raise ValueError("month_count must be positive")
    lower = jnp.tril(jnp.ones((month_count, month_count), dtype=jnp.float64))
    dense = jnp.concatenate(
        [jnp.ones((month_count, 1), dtype=jnp.float64), lower, lower], axis=1
    )
    return BCOO.fromdense(dense)


def apply_post_march(
    march_anchor: float | jax.Array,
    revised_birth_death: jax.Array,
    *,
    link_relatives: jax.Array | None = None,
    sample_job_changes: jax.Array | None = None,
) -> jax.Array:
    """Apply exactly one Req 10 representation."""
    if (link_relatives is None) == (sample_job_changes is None):
        raise ValueError("supply exactly one post-March representation")
    revised = jnp.asarray(revised_birth_death, dtype=jnp.float64)
    anchor = jnp.asarray(march_anchor, dtype=jnp.float64).reshape(1)
    if revised.ndim != 1 or revised.size == 0:
        raise ValueError("revised_birth_death must be a nonempty vector")
    if link_relatives is not None:
        links = jnp.asarray(link_relatives, dtype=jnp.float64)
        if links.shape != revised.shape:
            raise ValueError("link_relatives and revised_birth_death must align")
        return post_march_link_operator(links) @ jnp.concatenate([anchor, revised])
    changes = jnp.asarray(sample_job_changes, dtype=jnp.float64)
    if changes.shape != revised.shape:
        raise ValueError("sample_job_changes and revised_birth_death must align")
    inputs = jnp.concatenate([anchor, changes, revised])
    return post_march_cumulative_operator(revised.size) @ inputs


POST_MARCH_SCHEMA = {
    "benchmark_year": pl.Int32,
    "sector": pl.String,
    "reference_month": pl.Date,
    "month_index": pl.Int8,
    "representation": pl.String,
    "series_status": pl.String,
    "additional_sample_receipts": pl.Boolean,
    "march_anchor_thousands": pl.Float64,
    "published_level_thousands": pl.Float64,
    "revised_birth_death_thousands": pl.Float64,
    "inferred_sample_job_change_thousands": pl.Float64,
    "reproduced_level_thousands": pl.Float64,
    "residual_thousands": pl.Float64,
    "march_anchor_cell_id": pl.Int64,
    "previous_level_cell_id": pl.Int64,
    "published_level_cell_id": pl.Int64,
    "birth_death_source_keys": pl.List(pl.String),
}


def _level_index(levels: pl.DataFrame) -> dict[tuple[str, date, date], dict]:
    nsa = levels.filter(
        (pl.col("seasonal_status") == "NSA") & pl.col("value_thousands").is_not_null()
    )
    return {
        (row["sector"], row["release_month"], row["reference_month"]): row
        for row in nsa.iter_rows(named=True)
    }


def _schedule_index(
    birth_death: pl.DataFrame, schedule_kind: str
) -> dict[tuple[int, str, date], dict]:
    frame = birth_death.filter(
        (pl.col("series_kind") == "published_schedule")
        & (pl.col("schedule_kind") == schedule_kind)
    )
    return {
        (row["benchmark_year"], row["sector"], row["reference_month"]): row
        for row in frame.iter_rows(named=True)
    }


def build_post_march_components(
    levels: pl.DataFrame, birth_death: pl.DataFrame
) -> pl.DataFrame:
    level_rows = _level_index(levels)
    revised = _schedule_index(birth_death, "post_benchmark")
    records = []
    for year in BENCHMARK_YEARS:
        release = benchmark_release_month(year)
        months = month_range(date(year, 4, 1), date(year, 12, 1))
        for sector in OBSERVATION_SECTOR_ORDER:
            anchor_row = level_rows[(sector, release, date(year, 3, 1))]
            previous = float(anchor_row["value_thousands"])
            previous_row = anchor_row
            sample_changes = []
            revised_values = []
            source_rows = []
            for month in months:
                level_row = level_rows[(sector, release, month)]
                schedule = revised[(year, sector, month)]
                revised_value = float(schedule["value_thousands"])
                sample_change = (
                    float(level_row["value_thousands"]) - previous - revised_value
                )
                sample_changes.append(sample_change)
                revised_values.append(revised_value)
                source_rows.append((month, previous_row, level_row, schedule))
                previous = float(level_row["value_thousands"])
                previous_row = level_row
            reproduced = apply_post_march(
                float(anchor_row["value_thousands"]),
                jnp.asarray(revised_values, dtype=jnp.float64),
                sample_job_changes=jnp.asarray(sample_changes, dtype=jnp.float64),
            )
            for index, (
                (month, previous_row, level_row, schedule),
                calculated,
            ) in enumerate(zip(source_rows, reproduced, strict=True), start=1):
                published = float(level_row["value_thousands"])
                records.append(
                    {
                        "benchmark_year": year,
                        "sector": sector,
                        "reference_month": month,
                        "month_index": index,
                        "representation": "cumulative_job_change",
                        "series_status": "inferred",
                        "additional_sample_receipts": month.month >= 11,
                        "march_anchor_thousands": float(anchor_row["value_thousands"]),
                        "published_level_thousands": published,
                        "revised_birth_death_thousands": revised_values[index - 1],
                        "inferred_sample_job_change_thousands": sample_changes[
                            index - 1
                        ],
                        "reproduced_level_thousands": float(calculated),
                        "residual_thousands": published - float(calculated),
                        "march_anchor_cell_id": anchor_row["cell_id"],
                        "previous_level_cell_id": previous_row["cell_id"],
                        "published_level_cell_id": level_row["cell_id"],
                        "birth_death_source_keys": schedule["source_keys"],
                    }
                )
    return pl.DataFrame(records, schema=POST_MARCH_SCHEMA).sort(
        "benchmark_year", "sector", "reference_month"
    )


RECOVERABILITY_SCHEMA = {
    "benchmark_year": pl.Int32,
    "representation": pl.String,
    "series_status": pl.String,
    "link_attempt_status": pl.String,
    "attempted_cells": pl.Int32,
    "within_rounding_cells": pl.Int32,
    "max_abs_residual_thousands": pl.Float64,
    "reconstruction_error_variance_thousands2": pl.Float64,
    "link_attempt_residuals_thousands": pl.List(pl.Float64),
    "level_cell_ids": pl.List(pl.Int64),
    "birth_death_source_keys": pl.List(pl.String),
    "reason": pl.String,
}


def build_recoverability(
    levels: pl.DataFrame, birth_death: pl.DataFrame
) -> pl.DataFrame:
    level_rows = _level_index(levels)
    initial_rows = birth_death.filter(
        (pl.col("series_kind") == "published_schedule")
        & (pl.col("schedule_kind") == "preliminary")
    )
    initial = {
        (row["sector"], row["reference_month"]): row
        for row in initial_rows.iter_rows(named=True)
    }
    revised = _schedule_index(birth_death, "post_benchmark")
    records = []
    for year in BENCHMARK_YEARS:
        if year == 2003:
            records.append(
                {
                    "benchmark_year": year,
                    "representation": "cumulative_job_change",
                    "series_status": "inferred",
                    "link_attempt_status": "inputs_missing",
                    "attempted_cells": 0,
                    "within_rounding_cells": 0,
                    "max_abs_residual_thousands": None,
                    "reconstruction_error_variance_thousands2": (
                        ROUNDING_VARIANCE_THOUSANDS2
                    ),
                    "link_attempt_residuals_thousands": [],
                    "level_cell_ids": [],
                    "birth_death_source_keys": [],
                    "reason": (
                        "BLS publishes the 2003 post-benchmark schedule but no "
                        "initial April–December 2003 schedule or matched-sample links."
                    ),
                }
            )
            continue
        release_old = date(year, 12, 1)
        release_new = benchmark_release_month(year)
        months = month_range(date(year, 4, 1), date(year, 10, 1))
        residuals = []
        level_cell_ids = set()
        birth_death_source_keys = set()
        for sector in OBSERVATION_SECTOR_ORDER:
            old_previous_row = level_rows[(sector, release_old, date(year, 3, 1))]
            new_anchor_row = level_rows[(sector, release_new, date(year, 3, 1))]
            level_cell_ids.update(
                [old_previous_row["cell_id"], new_anchor_row["cell_id"]]
            )
            old_previous = float(old_previous_row["value_thousands"])
            links = []
            revised_values = []
            targets = []
            for month in months:
                old_level_row = level_rows[(sector, release_old, month)]
                target_row = level_rows[(sector, release_new, month)]
                initial_row = initial[(sector, month)]
                revised_row = revised[(year, sector, month)]
                level_cell_ids.update([old_level_row["cell_id"], target_row["cell_id"]])
                birth_death_source_keys.update(initial_row["source_keys"])
                birth_death_source_keys.update(revised_row["source_keys"])
                old_level = float(old_level_row["value_thousands"])
                initial_bd = float(initial_row["value_thousands"])
                revised_bd = float(revised_row["value_thousands"])
                links.append((old_level - initial_bd) / old_previous)
                revised_values.append(revised_bd)
                targets.append(float(target_row["value_thousands"]))
                old_previous = old_level
            predicted = apply_post_march(
                float(new_anchor_row["value_thousands"]),
                jnp.asarray(revised_values, dtype=jnp.float64),
                link_relatives=jnp.asarray(links, dtype=jnp.float64),
            )
            residuals.extend(
                target - float(calculated)
                for target, calculated in zip(targets, predicted, strict=True)
            )
        within_rounding = sum(abs(value) <= 0.5 for value in residuals)
        if within_rounding == len(residuals):
            raise ValueError(
                f"{year} aggregate link proxy unexpectedly passed; revisit the "
                "source-recoverability decision before changing representation"
            )
        squared_mean = sum(value * value for value in residuals) / len(residuals)
        records.append(
            {
                "benchmark_year": year,
                "representation": "cumulative_job_change",
                "series_status": "inferred",
                "link_attempt_status": "failed",
                "attempted_cells": len(residuals),
                "within_rounding_cells": within_rounding,
                "max_abs_residual_thousands": max(abs(value) for value in residuals),
                "reconstruction_error_variance_thousands2": max(
                    squared_mean, ROUNDING_VARIANCE_THOUSANDS2
                ),
                "link_attempt_residuals_thousands": residuals,
                "level_cell_ids": sorted(level_cell_ids),
                "birth_death_source_keys": sorted(birth_death_source_keys),
                "reason": (
                    "Vintage levels and initial birth–death imply only an aggregate "
                    "link proxy; applying it with revised birth–death does not "
                    "reproduce every April–October level to rounding."
                ),
            }
        )
    return pl.DataFrame(records, schema=RECOVERABILITY_SCHEMA).sort("benchmark_year")
