"""Revision size against the collection window `D_t`, total nonfarm (roadmap Stage 1).

Each revision is the absolute change from the first estimate of a month's over-the-month change
to its second (F_S) or third (F_T), as BLS's revision table gives them, joined to the day count
of the month's first release. ``signed_revisions`` keeps the sign, later estimate less first,
for the Stage 1 scale model. Reference months 2025-09 and 2025-10 drop, and March 2020 to
December 2021 form a period of their own. The regression fits the absolute revision of the main
period on `D_t`, a lapse indicator, an intercept, and reference-year and calendar-month dummies,
with standard errors clustered by reference year (CR1). A second fit adds a dummy for a 5-week
survey interval, which the calendar ties closely to `D_t`.
"""

from datetime import date, timedelta

import numpy as np
import polars as pl
from scipy import stats

PANDEMIC_MONTHS = (date(2020, 3, 1), date(2021, 12, 1))
DROPPED_MONTHS = (date(2025, 9, 1), date(2025, 10, 1))
TRANSITIONS = {"F_S": "S", "F_T": "T"}
KEYS = ["reference_month", "seasonal_status"]
# The contrast that the confidence interval covers: 12 to 19 business days, the range of
# BLS's normal schedule in the main period.
CONTRAST_DAYS = 7
CONFIDENCE = 0.95


def _reference_week_start(year: int, month: int) -> date:
    """The Sunday that starts the week holding the 12th of the month."""
    twelfth = date(year, month, 12)
    return twelfth - timedelta(days=(twelfth.weekday() + 1) % 7)


def five_week_interval(reference_month: date) -> bool:
    """Whether 5 weeks, not 4, separate this month's reference week from the last month's.

    The reference week is the Sunday-to-Saturday week holding the 12th. This is a rule computed
    from the calendar, not a BLS designation.
    """
    year, month = reference_month.year, reference_month.month
    previous = (year - 1, 12) if month == 1 else (year, month - 1)
    gap = _reference_week_start(year, month) - _reference_week_start(*previous)
    return gap.days == 35


def absolute_revisions(changes: pl.DataFrame, days: pl.DataFrame) -> pl.DataFrame:
    """One row per month, seasonal status and transition whose two estimates are published."""
    revisions = signed_revisions(changes, days)
    return revisions.with_columns(pl.col("revision").abs()).rename(
        {"revision": "absolute_revision"}
    )


def signed_revisions(changes: pl.DataFrame, days: pl.DataFrame) -> pl.DataFrame:
    """The rows of ``absolute_revisions`` with the signed revision, later less first."""
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
            revision=pl.col("later") - pl.col("first"),
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
        period=pl.when(pandemic).then(pl.lit("pandemic")).otherwise(pl.lit("main")),
        five_week=pl.col("reference_month").map_elements(
            five_week_interval, return_dtype=pl.Boolean
        ),
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


FIT_SCHEMA = {"interval_coefficient": pl.Float64, "interval_se": pl.Float64}


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


def _design(
    group: pl.DataFrame, *, interval_control: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """The regressors and the reference years of one fit's main-period revisions.

    Columns: intercept, `D_t`, lapse indicator, a dummy for each reference year after the
    first, a dummy for each calendar month after January, and, with `interval_control`, the
    5-week dummy last.
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
    if interval_control:
        columns.append(group["five_week"].to_numpy())
    return np.column_stack(columns).astype(float), year


def days_regression(
    revisions: pl.DataFrame, *, interval_control: bool = False
) -> pl.DataFrame:
    """The `D_t` and lapse coefficients of each main-period fit, with clustered errors.

    `contrast_low` and `contrast_high` bound the change in absolute revision from 12 to 19
    business days, with a t quantile on G - 1 degrees of freedom for G clusters. With
    `interval_control`, the fit adds the 5-week dummy and reports its coefficient.
    """
    rows = []
    main = revisions.filter(pl.col("period") == "main")
    for (status, transition), group in main.group_by("seasonal_status", "transition"):
        x, year = _design(group, interval_control=interval_control)
        y = group["absolute_revision"].to_numpy().astype(float)
        beta, se = ols_clustered(y, x, year)
        clusters = len(np.unique(year))
        quantile = stats.t.ppf((1 + CONFIDENCE) / 2, clusters - 1)
        rows.append(
            {
                "seasonal_status": status,
                "transition": transition,
                "n": group.height,
                "clusters": clusters,
                "days_coefficient": beta[1],
                "days_se": se[1],
                "contrast_low": CONTRAST_DAYS * (beta[1] - quantile * se[1]),
                "contrast_high": CONTRAST_DAYS * (beta[1] + quantile * se[1]),
                "lapse_coefficient": beta[2],
                "lapse_se": se[2],
                "interval_coefficient": beta[-1] if interval_control else None,
                "interval_se": se[-1] if interval_control else None,
            }
        )
    return pl.DataFrame(rows, schema_overrides=FIT_SCHEMA).sort(
        "seasonal_status", "transition"
    )
