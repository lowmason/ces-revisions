"""Roadmap Stage 9: the window to first closing, the collection rate C1 and revision size.

The window to first closing `W_t` follows the owner's rule: "first closing, when BLS stops
collecting survey responses for the first estimate, falls one week before the release" (owner,
2026-11-17). Part (a) fits C1 on `W_t`, part (b) fits the absolute revision on C1, and O4 fits it
on `D_t` and on `W_t`. Each fit has reference-year and calendar-month effects and the 5-week
dummy, takes the main period only, and clusters its CR1 errors by reference year.
"""

from datetime import date

import numpy as np
import polars as pl
from scipy import stats

from ces_revisions.collection_window import (
    CONFIDENCE,
    DROPPED_MONTHS,
    PANDEMIC_MONTHS,
    five_week_interval,
    ols_clustered,
)
from ces_revisions.vintages.collection_rates import LABSTAT_SOURCE
from ces_revisions.vintages.federal_calendar import is_federal_holiday

FIRST_MONTH = date(2003, 1, 1)
LAST_MONTH = date(2026, 8, 1)
C1_SERIES = "CEU00000000C1"
WINDOW_CONTRAST_DAYS = 7
RATE_CONTRAST_POINTS = 10
WINDOWS = ("collection_days", "first_closing_days")
MAIN, PANDEMIC, LAPSE = "main", "pandemic", "lapse"
REVISION_COLUMNS = ("NSA_F_S", "NSA_F_T", "SA_F_S", "SA_F_T")
FIT_KEYS = ["seasonal_status", "transition"]
MONTH_COLUMNS = [
    "reference_month",
    "release_date",
    "first_closing_date",
    "collection_days",
    "first_closing_days",
    "c1",
    "c1_footnote",
    "five_week",
    "period",
]


def month_sample(days: pl.DataFrame, rates: pl.DataFrame) -> pl.DataFrame:
    """One row per release from FIRST_MONTH to LAST_MONTH, less the dropped months.

    Each row carries LABSTAT's C1 and its footnote, the 5-week dummy, and its period: pandemic
    inside PANDEMIC_MONTHS, lapse for a lapse-delayed release, else main. A main month with no
    C1 is a data fault, so it raises ValueError naming each one.
    """
    c1 = rates.filter(
        (pl.col("source") == LABSTAT_SOURCE) & (pl.col("series_id") == C1_SERIES)
    ).select("reference_month", c1="value", c1_footnote="footnote_code")
    pandemic = pl.col("reference_month").is_between(*PANDEMIC_MONTHS)
    sample = (
        days.filter(
            pl.col("reference_month").is_between(FIRST_MONTH, LAST_MONTH),
            ~pl.col("reference_month").is_in(DROPPED_MONTHS),
        )
        .join(c1, on="reference_month", how="left")
        .with_columns(
            five_week=pl.col("reference_month").map_elements(
                five_week_interval, return_dtype=pl.Boolean
            ),
            period=pl.when(pandemic)
            .then(pl.lit(PANDEMIC))
            .when(pl.col("lapse_delayed"))
            .then(pl.lit(LAPSE))
            .otherwise(pl.lit(MAIN)),
        )
        .select(MONTH_COLUMNS)
        .sort("reference_month")
    )
    missing = sample.filter((pl.col("period") == MAIN) & pl.col("c1").is_null())
    if missing.height:
        months = ", ".join(f"{month:%Y-%m}" for month in missing["reference_month"])
        raise ValueError(f"main-period months with no LABSTAT C1: {months}")
    return sample


def revision_sample(revisions: pl.DataFrame, months: pl.DataFrame) -> pl.DataFrame:
    """The revisions of the sample's months, with each month's window, C1 and period."""
    return (
        revisions.drop("period", "lapse_delayed")
        .join(
            months.select("reference_month", "first_closing_days", "c1", "period"),
            on="reference_month",
            how="inner",
        )
        .sort("seasonal_status", "transition", "reference_month")
    )


def _fit(frame: pl.DataFrame, outcome: str, regressor: str, contrast: int) -> dict:
    """OLS of outcome on regressor with year, calendar-month and 5-week terms.

    Design columns: intercept, the regressor, a dummy for each reference year after the first,
    a dummy for each calendar month after January, and the 5-week dummy last. The errors are
    CR1, clustered by reference year, and the contrast interval uses a t quantile on
    clusters - 1 degrees of freedom.
    """
    year = frame["reference_month"].dt.year().to_numpy()
    month = frame["reference_month"].dt.month().to_numpy()
    x = np.column_stack(
        [
            np.ones(frame.height),
            frame[regressor].to_numpy(),
            *(year == later for later in np.unique(year)[1:]),
            *(month == later for later in range(2, 13)),
            frame["five_week"].to_numpy(),
        ]
    ).astype(float)
    beta, se = ols_clustered(frame[outcome].to_numpy().astype(float), x, year)
    clusters = len(np.unique(year))
    quantile = stats.t.ppf((1 + CONFIDENCE) / 2, clusters - 1)
    return {
        "n": frame.height,
        "clusters": clusters,
        "coefficient": beta[1],
        "se": se[1],
        "contrast_low": contrast * (beta[1] - quantile * se[1]),
        "contrast_high": contrast * (beta[1] + quantile * se[1]),
    }


def _main(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(pl.col("period") == MAIN)


def rates_by_window(months: pl.DataFrame) -> pl.DataFrame:
    """The number of main-period months and their mean C1 for each window value."""
    return (
        _main(months)
        .group_by("first_closing_days")
        .agg(n=pl.len(), mean_c1=pl.col("c1").mean())
        .sort("first_closing_days")
    )


def rate_regression(months: pl.DataFrame) -> pl.DataFrame:
    """Part (a): C1 on the window to first closing, over the main period."""
    return pl.DataFrame(
        [_fit(_main(months), "c1", "first_closing_days", WINDOW_CONTRAST_DAYS)]
    )


def revision_regression(revisions: pl.DataFrame) -> pl.DataFrame:
    """Part (b): the absolute revision on C1, for each status and transition."""
    rows = [
        {
            "seasonal_status": status,
            "transition": transition,
            **_fit(group, "absolute_revision", "c1", RATE_CONTRAST_POINTS),
        }
        for (status, transition), group in _main(revisions).group_by(FIT_KEYS)
    ]
    return pl.DataFrame(rows).sort(FIT_KEYS)


def window_regressions(revisions: pl.DataFrame) -> pl.DataFrame:
    """O4: the absolute revision on each window, for each status and transition."""
    rows = [
        {
            "seasonal_status": status,
            "transition": transition,
            "window": window,
            **_fit(group, "absolute_revision", window, WINDOW_CONTRAST_DAYS),
        }
        for (status, transition), group in _main(revisions).group_by(FIT_KEYS)
        for window in WINDOWS
    ]
    return pl.DataFrame(rows).sort(*FIT_KEYS, "window")


def rate_quintiles(months: pl.DataFrame) -> pl.DataFrame:
    """Each main month's C1 less its reference year's mean, and the quintile of that deviation.

    The edges are the 20th to 80th percentiles with linear interpolation, and a deviation on an
    edge goes to the lower quintile.
    """
    main = (
        _main(months)
        .sort("reference_month")
        .with_columns(
            deviation=pl.col("c1")
            - pl.col("c1").mean().over(pl.col("reference_month").dt.year())
        )
    )
    deviation = main["deviation"].to_numpy()
    edges = np.quantile(deviation, [0.2, 0.4, 0.6, 0.8])
    quintile = np.searchsorted(edges, deviation, side="left") + 1
    return main.select("reference_month", "deviation").with_columns(
        quintile=pl.Series(quintile, dtype=pl.Int8)
    )


def revisions_by_quintile(
    revisions: pl.DataFrame, quintiles: pl.DataFrame
) -> pl.DataFrame:
    """For each quintile, status and transition: its months, deviation range and mean revision."""
    ranges = quintiles.group_by("quintile").agg(
        months=pl.len(),
        deviation_low=pl.col("deviation").min(),
        deviation_high=pl.col("deviation").max(),
    )
    return (
        _main(revisions)
        .join(quintiles.select("reference_month", "quintile"), on="reference_month")
        .group_by("quintile", *FIT_KEYS)
        .agg(n=pl.len(), mean_absolute_revision=pl.col("absolute_revision").mean())
        .join(ranges, on="quintile")
        .select(
            "quintile",
            *FIT_KEYS,
            "months",
            "deviation_low",
            "deviation_high",
            "n",
            "mean_absolute_revision",
        )
        .sort("quintile", *FIT_KEYS)
    )


def _wide(revisions: pl.DataFrame, index: str, value: pl.Expr) -> pl.DataFrame:
    """One column per status and transition, named as REVISION_COLUMNS, null where none."""
    wide = (
        revisions.with_columns(
            column=pl.col("seasonal_status") + "_" + pl.col("transition")
        )
        .group_by(index, "column")
        .agg(value=value)
        .pivot(on="column", index=index, values="value")
    )
    missing = [name for name in REVISION_COLUMNS if name not in wide.columns]
    wide = wide.with_columns(
        pl.lit(None, dtype=pl.Float64).alias(name) for name in missing
    )
    return wide.select(index, *REVISION_COLUMNS)


def lapse_releases(months: pl.DataFrame, revisions: pl.DataFrame) -> pl.DataFrame:
    """Each lapse-delayed month's release, window, C1 and four absolute revisions."""
    lapse = months.filter(pl.col("period") == LAPSE).select(
        "reference_month",
        "release_date",
        "first_closing_date",
        "collection_days",
        "first_closing_days",
        "c1",
        "c1_footnote",
    )
    wide = _wide(
        revisions.join(
            lapse.select("reference_month"), on="reference_month", how="semi"
        ),
        "reference_month",
        pl.col("absolute_revision").first(),
    )
    return lapse.join(wide, on="reference_month", how="left").sort("reference_month")


def pandemic_by_window(months: pl.DataFrame, revisions: pl.DataFrame) -> pl.DataFrame:
    """For each window value of the pandemic period: months, mean C1 and mean revisions."""
    base = (
        months.filter(pl.col("period") == PANDEMIC)
        .group_by("first_closing_days")
        .agg(months=pl.len(), mean_c1=pl.col("c1").mean())
    )
    wide = _wide(
        revisions.filter(pl.col("period") == PANDEMIC),
        "first_closing_days",
        pl.col("absolute_revision").mean(),
    )
    return base.join(wide, on="first_closing_days", how="left").sort(
        "first_closing_days"
    )


def _covered(months: pl.DataFrame) -> pl.DataFrame:
    return months.filter(pl.col("period").is_in([MAIN, PANDEMIC]))


def holiday_closings(months: pl.DataFrame) -> pl.DataFrame:
    """The main and pandemic months whose closing day by the rule is a federal holiday."""
    return (
        _covered(months)
        .filter(
            pl.col("first_closing_date").map_elements(
                is_federal_holiday, return_dtype=pl.Boolean
            )
        )
        .select("reference_month", "release_date", "first_closing_date")
        .sort("reference_month")
    )


def final_week_holidays(months: pl.DataFrame) -> pl.DataFrame:
    """Per calendar month: main and pandemic releases, and those with a final-week holiday.

    A holiday between the closing day and the release shortens `D_t - W_t` from 4 to 3.
    """
    return (
        _covered(months)
        .group_by(calendar_month=pl.col("reference_month").dt.month())
        .agg(
            releases=pl.len(),
            with_holiday=(
                (pl.col("collection_days") - pl.col("first_closing_days")) == 3
            ).sum(),
        )
        .sort("calendar_month")
    )
