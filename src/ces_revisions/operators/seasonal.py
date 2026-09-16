"""Req 9's joint NSA/SA measurement map and B→M observable components."""

from datetime import date

import jax.numpy as jnp
import polars as pl
from jax.experimental.sparse import BCOO


def seasonal_measurement_operator() -> BCOO:
    """Map mature `(x, q)` to `(NSA, SA) = (x, x-q)`."""
    return BCOO.fromdense(jnp.asarray([[1.0, 0.0], [1.0, -1.0]], dtype=jnp.float64))


def implied_adjustment_operator() -> BCOO:
    """Map a published `(NSA, SA)` pair to its additive adjustment."""
    return BCOO.fromdense(jnp.asarray([[1.0, -1.0]], dtype=jnp.float64))


B_TO_M_SCHEMA = {
    "sector": pl.String,
    "reference_month": pl.Date,
    "seasonal_status": pl.String,
    "b_release_month": pl.Date,
    "m_release_month": pl.Date,
    "b_status": pl.String,
    "m_status": pl.String,
    "b_value_thousands": pl.Float64,
    "m_value_thousands": pl.Float64,
    "b_nsa_thousands": pl.Float64,
    "b_sa_thousands": pl.Float64,
    "m_nsa_thousands": pl.Float64,
    "m_sa_thousands": pl.Float64,
    "b_adjustment_thousands": pl.Float64,
    "m_adjustment_thousands": pl.Float64,
    "observed_delta_thousands": pl.Float64,
    "paired_nsa_delta_thousands": pl.Float64,
    "seasonal_adjustment_delta_thousands": pl.Float64,
    "identity_residual_thousands": pl.Float64,
    "next_benchmark_year": pl.Int32,
    "next_wedge_weight": pl.Float64,
    "reconstruction_error_variance_thousands2": pl.Float64,
    "b_cell_ids": pl.List(pl.Int64),
    "m_cell_ids": pl.List(pl.Int64),
}


def _value_index(
    levels: pl.DataFrame,
) -> dict[tuple[str, date, date, str], dict]:
    return {
        (
            row["sector"],
            row["reference_month"],
            row["release_month"],
            row["seasonal_status"],
        ): row
        for row in levels.iter_rows(named=True)
        if row["value_thousands"] is not None
    }


def _pair_at_release(
    values: dict[tuple[str, date, date, str], dict],
    sector: str,
    reference_month: date,
    release_month: date,
) -> tuple[float | None, float | None, list[int]]:
    nsa = values.get((sector, reference_month, release_month, "NSA"))
    sa = values.get((sector, reference_month, release_month, "SA"))
    return (
        None if nsa is None else float(nsa["value_thousands"]),
        None if sa is None else float(sa["value_thousands"]),
        [row["cell_id"] for row in (nsa, sa) if row is not None],
    )


def _difference(later: float | None, earlier: float | None) -> float | None:
    return None if later is None or earlier is None else later - earlier


def build_b_to_m_components(
    levels: pl.DataFrame,
    stage_labels: pl.DataFrame,
    recoverability: pl.DataFrame,
) -> pl.DataFrame:
    labels = stage_labels.filter(
        (pl.col("source") == "cesvinall") & pl.col("release_stage").is_in(["B", "M"])
    )
    values = _value_index(levels)
    variances = {
        row["benchmark_year"]: row["reconstruction_error_variance_thousands2"]
        for row in recoverability.iter_rows(named=True)
    }
    records = []
    for key, group in labels.group_by(
        "sector", "reference_month", "seasonal_status", maintain_order=True
    ):
        sector, reference_month, seasonal_status = key
        by_stage = {row["release_stage"]: row for row in group.iter_rows(named=True)}
        if set(by_stage) != {"B", "M"}:
            raise ValueError(f"missing B or M label for {key}")
        b_label = by_stage["B"]
        m_label = by_stage["M"]
        b_release = b_label["release_month"]
        m_release = m_label["release_month"]
        b_nsa, b_sa, b_ids = _pair_at_release(
            values, sector, reference_month, b_release
        )
        m_nsa, m_sa, m_ids = _pair_at_release(
            values, sector, reference_month, m_release
        )
        b_value = b_nsa if seasonal_status == "NSA" else b_sa
        m_value = m_nsa if seasonal_status == "NSA" else m_sa
        b_adjustment = _difference(b_nsa, b_sa)
        m_adjustment = _difference(m_nsa, m_sa)
        observed_delta = _difference(m_value, b_value)
        paired_nsa_delta = _difference(m_nsa, b_nsa)
        seasonal_delta = _difference(m_adjustment, b_adjustment)
        stages_observed = b_label["status"] == m_label["status"] == "observed"
        if not stages_observed or None in (
            observed_delta,
            paired_nsa_delta,
            seasonal_delta,
        ):
            identity = None
        elif seasonal_status == "SA":
            identity = observed_delta - (paired_nsa_delta - seasonal_delta)
        else:
            identity = observed_delta - paired_nsa_delta
        candidate_year = m_release.year - 1
        eligible_wedge = (
            seasonal_status == "NSA"
            and 4 <= reference_month.month <= 10
            and candidate_year in variances
        )
        next_year = candidate_year if eligible_wedge else None
        variance = variances.get(next_year) if next_year is not None else None
        records.append(
            {
                "sector": sector,
                "reference_month": reference_month,
                "seasonal_status": seasonal_status,
                "b_release_month": b_release,
                "m_release_month": m_release,
                "b_status": b_label["status"],
                "m_status": m_label["status"],
                "b_value_thousands": b_value
                if b_label["status"] == "observed"
                else None,
                "m_value_thousands": m_value
                if m_label["status"] == "observed"
                else None,
                "b_nsa_thousands": b_nsa,
                "b_sa_thousands": b_sa,
                "m_nsa_thousands": m_nsa,
                "m_sa_thousands": m_sa,
                "b_adjustment_thousands": b_adjustment,
                "m_adjustment_thousands": m_adjustment,
                "observed_delta_thousands": observed_delta if stages_observed else None,
                "paired_nsa_delta_thousands": paired_nsa_delta,
                "seasonal_adjustment_delta_thousands": seasonal_delta,
                "identity_residual_thousands": identity,
                "next_benchmark_year": next_year,
                "next_wedge_weight": (
                    (reference_month.month - 3) / 12 if eligible_wedge else None
                ),
                "reconstruction_error_variance_thousands2": variance,
                "b_cell_ids": b_ids,
                "m_cell_ids": m_ids,
            }
        )
    return pl.DataFrame(records, schema=B_TO_M_SCHEMA).sort(
        "sector", "reference_month", "seasonal_status"
    )
