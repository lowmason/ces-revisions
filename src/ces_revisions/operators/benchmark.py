"""Req 10 benchmark wedge and published-vintage validation fixtures."""

from collections.abc import Sequence
from datetime import date

import jax
import jax.numpy as jnp
import polars as pl
from jax.experimental.sparse import BCOO

from ces_revisions.vintages.months import month_range

BENCHMARK_YEARS = tuple(range(2003, 2026))
WINDOW_MONTHS = 12


def wedge_weights() -> jax.Array:
    return jnp.arange(1, WINDOW_MONTHS + 1, dtype=jnp.float64) / WINDOW_MONTHS


def wedge_operator() -> BCOO:
    weights = wedge_weights()
    march = jnp.zeros(WINDOW_MONTHS, dtype=jnp.float64).at[-1].set(1.0)
    dense = jnp.concatenate(
        [
            jnp.eye(WINDOW_MONTHS, dtype=jnp.float64) - jnp.outer(weights, march),
            weights[:, None],
        ],
        axis=1,
    )
    stored_elements = 3 * (WINDOW_MONTHS - 1) + 1
    return BCOO.fromdense(dense, nse=stored_elements)


def reconstruction_selector(support: Sequence[bool]) -> BCOO:
    flags = jnp.asarray(support, dtype=jnp.bool_)
    diagonal = jnp.arange(flags.size, dtype=jnp.int32)
    indices = jnp.stack([diagonal, diagonal], axis=1)
    return BCOO(
        (flags.astype(jnp.float64), indices),
        shape=(flags.size, flags.size),
    )


def apply_wedge(
    previous_levels: jax.Array,
    benchmark_anchor: float | jax.Array,
    *,
    reconstruction_terms: jax.Array | None = None,
    reconstruction_support: Sequence[bool] | None = None,
    rounding_residuals: jax.Array | None = None,
) -> jax.Array:
    """Apply Req 10 using the scope-adjusted March anchor before ``R kappa``.

    The archived benchmark-vintage March level may also contain a documented
    reconstruction.  Callers must pass the pre-reconstruction ``b_fin`` anchor
    here and supply that reconstruction separately.
    """
    previous = jnp.asarray(previous_levels, dtype=jnp.float64)
    if previous.shape != (WINDOW_MONTHS,):
        raise ValueError("previous_levels must hold April through March")
    inputs = jnp.concatenate(
        [previous, jnp.asarray(benchmark_anchor, dtype=jnp.float64).reshape(1)]
    )
    result = wedge_operator() @ inputs
    if (reconstruction_terms is None) != (reconstruction_support is None):
        raise ValueError(
            "reconstruction_terms and reconstruction_support must be supplied together"
        )
    if reconstruction_terms is not None:
        assert reconstruction_support is not None
        term = jnp.asarray(reconstruction_terms, dtype=jnp.float64)
        if term.shape != (WINDOW_MONTHS,):
            raise ValueError("reconstruction_terms must hold twelve months")
        if len(reconstruction_support) != WINDOW_MONTHS:
            raise ValueError("reconstruction_support must hold twelve months")
        result = result + reconstruction_selector(reconstruction_support) @ term
    if rounding_residuals is not None:
        rounding = jnp.asarray(rounding_residuals, dtype=jnp.float64)
        if rounding.shape != (WINDOW_MONTHS,):
            raise ValueError("rounding_residuals must hold twelve months")
        result = result + rounding
    return result


def benchmark_window(year: int) -> tuple[date, ...]:
    if year not in BENCHMARK_YEARS:
        raise ValueError(f"benchmark year outside 2003–2025: {year}")
    return tuple(month_range(date(year - 1, 4, 1), date(year, 3, 1)))


def prebenchmark_release_month(year: int) -> date:
    return date(year, 12, 1)


def benchmark_release_month(year: int) -> date:
    return date(year + 1, 1, 1)


BENCHMARK_FIXTURE_SCHEMA = {
    "benchmark_year": pl.Int32,
    "sector": pl.String,
    "reference_month": pl.Date,
    "prebenchmark_release_month": pl.Date,
    "benchmark_release_month": pl.Date,
    "previous_level_thousands": pl.Float64,
    "benchmark_level_thousands": pl.Float64,
    "wedge_anchor_thousands": pl.Float64,
    "published_revision_thousands": pl.Float64,
    "effective_revision_thousands": pl.Float64,
    "revision_status": pl.String,
    "wedge_weight": pl.Float64,
    "reconstruction_supported": pl.Boolean,
    "event_ids": pl.List(pl.String),
    "linear_wedge_level_thousands": pl.Float64,
    "reconstruction_term_thousands": pl.Float64,
    "rounding_residual_thousands": pl.Float64,
    "reproduced_level_thousands": pl.Float64,
    "previous_cell_id": pl.Int64,
    "benchmark_cell_id": pl.Int64,
    "benchmark_source_keys": pl.List(pl.String),
}


def _only(frame: pl.DataFrame, label: str) -> dict:
    if frame.height != 1:
        raise ValueError(f"expected one {label}, found {frame.height}")
    return frame.row(0, named=True)


def _benchmark_row(benchmarks: pl.DataFrame, year: int, sector: str) -> dict:
    row_kind = "anchor" if sector == "00" else "sector_contribution"
    return _only(
        benchmarks.filter(
            (pl.col("benchmark_year") == year)
            & (pl.col("benchmark_status") == "final")
            & (pl.col("row_kind") == row_kind)
            & (pl.col("sector") == sector)
        ),
        f"final benchmark row for {year}/{sector}",
    )


def _event_ids(
    events: pl.DataFrame, year: int, sector: str, reference_month: date
) -> list[str]:
    return sorted(
        row["event_id"]
        for row in events.filter(pl.col("benchmark_year") == year).iter_rows(named=True)
        if sector in row["sectors"]
        and (
            row["reference_start"] is None or reference_month >= row["reference_start"]
        )
        and (row["reference_end"] is None or reference_month <= row["reference_end"])
    )


def build_benchmark_fixtures(
    levels: pl.DataFrame,
    benchmarks: pl.DataFrame,
    reconstruction_events: pl.DataFrame,
) -> pl.DataFrame:
    nsa = levels.filter(pl.col("seasonal_status") == "NSA")
    records = []
    for year in BENCHMARK_YEARS:
        old_release = prebenchmark_release_month(year)
        new_release = benchmark_release_month(year)
        months = benchmark_window(year)
        for sector in (
            "00",
            "10",
            "20",
            "30",
            "40",
            "50",
            "55",
            "60",
            "65",
            "70",
            "80",
            "90",
        ):
            old = nsa.filter(
                (pl.col("sector") == sector)
                & (pl.col("release_month") == old_release)
                & pl.col("reference_month").is_in(months)
            ).sort("reference_month")
            new = nsa.filter(
                (pl.col("sector") == sector)
                & (pl.col("release_month") == new_release)
                & pl.col("reference_month").is_in(months)
            ).sort("reference_month")
            if old.height != WINDOW_MONTHS or new.height != WINDOW_MONTHS:
                raise ValueError(f"incomplete benchmark fixture for {year}/{sector}")
            benchmark = _benchmark_row(benchmarks, year, sector)
            previous_march = float(old[-1, "value_thousands"])
            march_gap = float(new[-1, "value_thousands"]) - previous_march
            published = benchmark["revision_thousands"]
            effective = march_gap if published is None else float(published)
            wedge_anchor = previous_march + effective
            status = "inferred_from_archive" if published is None else "published"
            for index, (old_row, new_row) in enumerate(
                zip(old.iter_rows(named=True), new.iter_rows(named=True), strict=True),
                start=1,
            ):
                event_ids = _event_ids(
                    reconstruction_events,
                    year,
                    sector,
                    old_row["reference_month"],
                )
                weight = index / WINDOW_MONTHS
                linear = float(old_row["value_thousands"]) + weight * (
                    wedge_anchor - previous_march
                )
                residual = float(new_row["value_thousands"]) - linear
                reconstruction = residual if event_ids else 0.0
                rounding = 0.0 if event_ids else residual
                records.append(
                    {
                        "benchmark_year": year,
                        "sector": sector,
                        "reference_month": old_row["reference_month"],
                        "prebenchmark_release_month": old_release,
                        "benchmark_release_month": new_release,
                        "previous_level_thousands": float(old_row["value_thousands"]),
                        "benchmark_level_thousands": float(new_row["value_thousands"]),
                        "wedge_anchor_thousands": wedge_anchor,
                        "published_revision_thousands": published,
                        "effective_revision_thousands": effective,
                        "revision_status": status,
                        "wedge_weight": weight,
                        "reconstruction_supported": bool(event_ids),
                        "event_ids": event_ids,
                        "linear_wedge_level_thousands": linear,
                        "reconstruction_term_thousands": reconstruction,
                        "rounding_residual_thousands": rounding,
                        "reproduced_level_thousands": linear
                        + reconstruction
                        + rounding,
                        "previous_cell_id": old_row["cell_id"],
                        "benchmark_cell_id": new_row["cell_id"],
                        "benchmark_source_keys": benchmark["source_keys"],
                    }
                )
    return pl.DataFrame(records, schema=BENCHMARK_FIXTURE_SCHEMA).sort(
        "benchmark_year", "sector", "reference_month"
    )
