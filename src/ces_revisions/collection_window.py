"""Revision size against the collection window `D_t`, total nonfarm (roadmap Stage 1).

Each revision is the absolute change from the first estimate of a month's over-the-month change
to its second (F_S) or third (F_T), as BLS's revision table gives them, joined to the day count
of the month's first release. Reference months 2025-09 and 2025-10 drop, and March 2020 to
December 2021 form a period of their own. The regression fits the absolute revision of the main
period on `D_t`, a lapse indicator, an intercept, and reference-year and calendar-month dummies,
with standard errors clustered by reference year (CR1).
"""

from datetime import date

import numpy as np
import polars as pl

PANDEMIC_MONTHS = (date(2020, 3, 1), date(2021, 12, 1))
DROPPED_MONTHS = (date(2025, 9, 1), date(2025, 10, 1))
TRANSITIONS = {"F_S": "S", "F_T": "T"}
KEYS = ["reference_month", "seasonal_status"]


def absolute_revisions(changes: pl.DataFrame, days: pl.DataFrame) -> pl.DataFrame:
    """One row per month, seasonal status and transition whose two estimates are published."""
    estimates = changes.filter(
        pl.col("source") == "cesnaicsrev",
        ~pl.col("reference_month").is_in(DROPPED_MONTHS),
        pl.col("value_thousands").is_not_null(),
    )
    first = estimates.filter(pl.col("release_stage") == "F").select(
        *KEYS, first="value_thousands"
    )
    revisions = pl.concat(
        first.join(
            estimates.filter(pl.col("release_stage") == stage).select(
                *KEYS, later="value_thousands"
            ),
            on=KEYS,
            how="inner",
        ).select(
            *KEYS,
            transition=pl.lit(transition),
            absolute_revision=(pl.col("later") - pl.col("first")).abs(),
        )
        for transition, stage in TRANSITIONS.items()
    )
    joined = revisions.join(
        days.select("reference_month", "collection_days", "lapse_delayed"),
        on="reference_month",
        how="left",
    )
    if joined["collection_days"].null_count():
        raise ValueError("a revised month has no release in the day counts")
    pandemic = pl.col("reference_month").is_between(*PANDEMIC_MONTHS)
    return joined.with_columns(
        period=pl.when(pandemic).then(pl.lit("pandemic")).otherwise(pl.lit("main"))
    ).sort("seasonal_status", "transition", "reference_month")


def revisions_by_days(revisions: pl.DataFrame) -> pl.DataFrame:
    """The mean absolute revision for each period, status, transition and `D_t` value."""
    return (
        revisions.group_by("period", "seasonal_status", "transition", "collection_days")
        .agg(
            n=pl.len(),
            lapse_n=pl.col("lapse_delayed").sum(),
            mean_absolute_revision=pl.col("absolute_revision").mean(),
        )
        .sort("period", "seasonal_status", "transition", "collection_days")
    )


def ols_clustered(
    y: np.ndarray, x: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and their cluster-robust standard errors with the CR1 factor.

    The factor is G/(G-1) * (N-1)/(N-K), for G clusters, N observations and K regressors.
    """
    n, k = x.shape
    bread = np.linalg.inv(x.T @ x)
    beta = bread @ x.T @ y
    residuals = y - x @ beta
    labels = np.unique(groups)
    scores = np.stack(
        [x[groups == label].T @ residuals[groups == label] for label in labels]
    )
    g = len(labels)
    factor = g / (g - 1) * (n - 1) / (n - k)
    covariance = factor * bread @ (scores.T @ scores) @ bread
    return beta, np.sqrt(np.diag(covariance))


def _design(group: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The regressors and the reference years of one fit's main-period revisions.

    Columns: intercept, `D_t`, lapse indicator, a dummy for each reference year after the
    first, and a dummy for each calendar month after January.
    """
    year = group["reference_month"].dt.year().to_numpy()
    month = group["reference_month"].dt.month().to_numpy()
    columns = [
        np.ones(group.height),
        group["collection_days"].to_numpy(),
        group["lapse_delayed"].to_numpy(),
        *(year == later for later in np.unique(year)[1:]),
        *(month == later for later in range(2, 13)),
    ]
    return np.column_stack(columns).astype(float), year


def days_regression(revisions: pl.DataFrame) -> pl.DataFrame:
    """The `D_t` and lapse coefficients of each main-period fit, with clustered errors."""
    rows = []
    main = revisions.filter(pl.col("period") == "main")
    for (status, transition), group in main.group_by("seasonal_status", "transition"):
        x, year = _design(group)
        y = group["absolute_revision"].to_numpy().astype(float)
        beta, se = ols_clustered(y, x, year)
        rows.append(
            {
                "seasonal_status": status,
                "transition": transition,
                "n": group.height,
                "clusters": len(np.unique(year)),
                "days_coefficient": beta[1],
                "days_se": se[1],
                "lapse_coefficient": beta[2],
                "lapse_se": se[2],
            }
        )
    return pl.DataFrame(rows).sort("seasonal_status", "transition")
