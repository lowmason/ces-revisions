"""The Stage 9 samples and fits, on small synthetic frames."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from scipy import stats

from ces_revisions.collection_window import five_week_interval
from ces_revisions.first_closing import (
    C1_SERIES,
    RATE_CONTRAST_POINTS,
    WINDOW_CONTRAST_DAYS,
    final_week_holidays,
    holiday_closings,
    lapse_releases,
    month_sample,
    pandemic_by_window,
    rate_quintiles,
    rate_regression,
    rates_by_window,
    revision_regression,
    revision_sample,
    revisions_by_quintile,
    window_regressions,
)
from ces_revisions.vintages.collection_days import SCHEMA as DAYS_SCHEMA
from ces_revisions.vintages.collection_rates import SCHEMA as RATES_SCHEMA
from ces_revisions.vintages.months import add_months

STATUSES = ("NSA", "SA")
TRANSITIONS = ("F_S", "F_T")


def days_frame(months: list[date], *, lapse: tuple[date, ...] = ()) -> pl.DataFrame:
    """One release per month, on the first Friday after the month's end plus a week."""
    rows = []
    for month in months:
        release = add_months(month, 1) + timedelta(days=6)
        rows.append(
            (month, release, 14, month in lapse, release - timedelta(days=7), 10)
        )
    return pl.DataFrame(rows, schema=DAYS_SCHEMA, orient="row")


def rates_frame(values: dict[date, float]) -> pl.DataFrame:
    rows = [
        (C1_SERIES, month, "labstat", value, None) for month, value in values.items()
    ]
    rows += [("CEU00000000C2", month, "labstat", 99.0, None) for month in values]
    rows += [(C1_SERIES, month, "20250319165303", 1.0, None) for month in values]
    return pl.DataFrame(rows, schema=RATES_SCHEMA, orient="row")


def year_months(years: range) -> list[date]:
    return [date(year, month, 1) for year in years for month in range(1, 13)]


def window(month: date) -> int:
    """A window that varies within each year and is not a sum of year and month effects."""
    return 8 + (month.month * month.month + month.year) % 8


def fit_months(years: range, c1: dict[date, float]) -> pl.DataFrame:
    months = year_months(years)
    return pl.DataFrame(
        {
            "reference_month": months,
            "first_closing_days": [window(month) for month in months],
            "c1": [c1[month] for month in months],
            "five_week": [five_week_interval(month) for month in months],
            "period": ["main"] * len(months),
        }
    )


def fit_revisions(months: pl.DataFrame, outcome) -> pl.DataFrame:
    """One revision per month, status and transition, from outcome(row, status, transition)."""
    rows = []
    for row in months.iter_rows(named=True):
        for status in STATUSES:
            for transition in TRANSITIONS:
                rows.append(
                    {
                        "reference_month": row["reference_month"],
                        "seasonal_status": status,
                        "transition": transition,
                        "absolute_revision": outcome(row, status, transition),
                        "collection_days": row["first_closing_days"]
                        + (3 if row["reference_month"].month == 12 else 4),
                        "first_closing_days": row["first_closing_days"],
                        "c1": row["c1"],
                        "five_week": row["five_week"],
                        "period": row["period"],
                    }
                )
    return pl.DataFrame(rows)


def noisy_months(seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    months = year_months(range(2010, 2015))
    c1 = {month: 2 * window(month) + rng.normal(0, 3) for month in months}
    return fit_months(range(2010, 2015), c1)


def noisy_revisions(months: pl.DataFrame, seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return fit_revisions(
        months, lambda row, status, transition: 0.3 * row["c1"] + rng.normal(0, 5)
    )


def test_month_sample_labels_each_period_and_drops_the_dropped_months():
    main, pandemic, lapse = date(2010, 6, 1), date(2020, 6, 1), date(2013, 10, 1)
    months = [date(2002, 12, 1), main, lapse, pandemic, date(2025, 10, 1)]
    months.append(date(2026, 9, 1))
    sample = month_sample(
        days_frame(months, lapse=(lapse,)),
        rates_frame({month: 60.0 + index for index, month in enumerate(months)}),
    )
    assert sample.columns == [
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
    assert sample["reference_month"].to_list() == [main, lapse, pandemic]
    assert sample["period"].to_list() == ["main", "lapse", "pandemic"]
    assert sample["c1"].to_list() == [61.0, 62.0, 63.0]
    assert sample["five_week"].to_list() == [
        five_week_interval(month) for month in (main, lapse, pandemic)
    ]


def test_month_sample_refuses_a_main_month_without_c1():
    months = [date(2004, 5, 1), date(2004, 6, 1), date(2020, 5, 1)]
    with pytest.raises(ValueError, match="2004-05") as refusal:
        month_sample(days_frame(months), rates_frame({date(2004, 6, 1): 70.0}))
    assert "2020-05" not in str(refusal.value)


def test_revision_sample_takes_period_and_rate_from_the_months():
    lapse, main = date(2013, 10, 1), date(2013, 11, 1)
    months = month_sample(
        days_frame([lapse, main], lapse=(lapse,)),
        rates_frame({lapse: 83.5, main: 80.0}),
    )
    revisions = pl.DataFrame(
        {
            "reference_month": [lapse, main, date(2002, 12, 1)],
            "seasonal_status": ["NSA", "NSA", "NSA"],
            "transition": ["F_S", "F_S", "F_S"],
            "absolute_revision": [10.0, 20.0, 30.0],
            "collection_days": [14, 14, 14],
            "lapse_delayed": [True, False, False],
            "period": ["main", "main", "main"],
            "five_week": [False, False, False],
        }
    )
    sample = revision_sample(revisions, months)
    assert "lapse_delayed" not in sample.columns
    assert sample.sort("reference_month").select(
        "reference_month", "period", "c1", "first_closing_days"
    ).rows() == [(lapse, "lapse", 83.5, 10), (main, "main", 80.0, 10)]


def test_fit_recovers_an_exact_slope():
    months = year_months(range(2010, 2013))
    c1 = {
        month: 2 * window(month)
        + 3 * (month.year - 2010)
        + 0.5 * month.month
        + 5 * five_week_interval(month)
        for month in months
    }
    fit = rate_regression(fit_months(range(2010, 2013), c1))
    assert fit.height == 1
    assert fit["coefficient"][0] == pytest.approx(2)
    assert fit["n"][0] == 36
    assert fit["clusters"][0] == 3


def test_contrast_interval_uses_a_t_quantile_on_clusters_less_one():
    months = noisy_months()
    fits = [
        (rate_regression(months), WINDOW_CONTRAST_DAYS),
        (revision_regression(noisy_revisions(months)), RATE_CONTRAST_POINTS),
    ]
    assert fits[1][0].height == 4
    for fit, contrast in fits:
        for row in fit.iter_rows(named=True):
            assert row["se"] > 0
            quantile = stats.t.ppf(0.975, row["clusters"] - 1)
            middle = (row["contrast_low"] + row["contrast_high"]) / 2
            half = (row["contrast_high"] - row["contrast_low"]) / 2
            assert middle == pytest.approx(contrast * row["coefficient"])
            assert half == pytest.approx(contrast * quantile * row["se"])


def test_only_the_main_period_enters_the_fits():
    months = noisy_months()
    revisions = noisy_revisions(months)
    others = pl.DataFrame(
        {
            "reference_month": [date(2020, 6, 1), date(2013, 10, 1)],
            "first_closing_days": [9, 22],
            "c1": [1000.0, -1000.0],
            "five_week": [False, True],
            "period": ["pandemic", "lapse"],
        }
    )
    outliers = fit_revisions(others, lambda row, status, transition: 1e6)
    with_months = pl.concat([months, others])
    with_revisions = pl.concat([revisions, outliers])
    assert_frame_equal(rate_regression(with_months), rate_regression(months))
    assert_frame_equal(
        revision_regression(with_revisions), revision_regression(revisions)
    )
    assert_frame_equal(
        window_regressions(with_revisions), window_regressions(revisions)
    )


def test_window_regressions_fit_each_window_on_the_same_rows():
    months = noisy_months()
    revisions = fit_revisions(
        months,
        lambda row, status, transition: (
            2 * row["first_closing_days"] + 0.1 * row["reference_month"].month
        ),
    )
    fits = window_regressions(revisions)
    assert fits.height == 8
    for _, group in fits.group_by("seasonal_status", "transition"):
        assert group["n"].n_unique() == 1
        on_window = group.filter(pl.col("window") == "first_closing_days")
        assert on_window["coefficient"][0] == pytest.approx(2)


def test_quintile_edges_put_a_tie_in_the_lower_quintile():
    ten = [date(2010, month, 1) for month in range(1, 11)]
    months = pl.DataFrame(
        {
            "reference_month": ten,
            "c1": [float(v) for v in range(1, 11)],
            "period": "main",
        }
    )
    quintiles = rate_quintiles(months)
    assert quintiles["quintile"].dtype == pl.Int8
    assert quintiles["quintile"].to_list() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    six = months.head(6)
    # Six deviations -2.5 to 2.5 put the edges at -1.5, -0.5, 0.5 and 1.5, on four of them.
    assert rate_quintiles(six)["quintile"].to_list() == [1, 1, 2, 3, 4, 5]


def test_quintile_deviation_is_taken_within_the_reference_year():
    months = pl.DataFrame(
        {
            "reference_month": [date(2010, m, 1) for m in (1, 2, 3)]
            + [date(2011, m, 1) for m in (1, 2, 3)],
            "c1": [58.0, 60.0, 62.0, 78.0, 80.0, 82.0],
            "period": ["main"] * 6,
        }
    )
    deviations = rate_quintiles(months).sort("reference_month")["deviation"]
    assert deviations.to_list() == pytest.approx([-2, 0, 2, -2, 0, 2])


def test_lapse_releases_keep_missing_revisions_null():
    lapse = date(2013, 10, 1)
    months = month_sample(
        days_frame([lapse, date(2013, 11, 1)], lapse=(lapse,)),
        rates_frame({lapse: 83.5, date(2013, 11, 1): 80.0}),
    )
    revisions = pl.DataFrame(
        {
            "reference_month": [lapse, lapse],
            "seasonal_status": ["NSA", "SA"],
            "transition": ["F_S", "F_S"],
            "absolute_revision": [12.0, 7.0],
            "collection_days": [14, 14],
            "five_week": [False, False],
            "first_closing_days": [10, 10],
            "c1": [83.5, 83.5],
            "period": ["lapse", "lapse"],
        }
    )
    table = lapse_releases(months, revisions)
    assert table["reference_month"].to_list() == [lapse]
    row = table.row(0, named=True)
    assert (row["NSA_F_S"], row["SA_F_S"]) == (12.0, 7.0)
    assert row["NSA_F_T"] is None
    assert row["SA_F_T"] is None
    assert row["c1"] == 83.5


def test_holiday_closings_and_final_week_counts():
    months = pl.DataFrame(
        {
            "reference_month": [date(2010, 11, 1), date(2010, 12, 1)],
            "release_date": [date(2010, 12, 3), date(2011, 1, 7)],
            "first_closing_date": [date(2010, 11, 26), date(2010, 12, 31)],
            "collection_days": [14, 17],
            "first_closing_days": [10, 14],
            "period": ["main", "main"],
        }
    )
    closings = holiday_closings(months)
    assert closings.rows() == [
        (date(2010, 12, 1), date(2011, 1, 7), date(2010, 12, 31))
    ]
    counts = {
        row["calendar_month"]: (row["releases"], row["with_holiday"])
        for row in final_week_holidays(months).iter_rows(named=True)
    }
    assert counts == {11: (1, 0), 12: (1, 1)}


def test_revisions_by_quintile_counts_months_and_revisions():
    one, two, three = date(2010, 1, 1), date(2010, 2, 1), date(2010, 3, 1)
    quintiles = pl.DataFrame(
        {
            "reference_month": [one, two, three],
            "deviation": [-2.0, -1.0, 1.0],
            "quintile": pl.Series([1, 1, 2], dtype=pl.Int8),
        }
    )
    revisions = pl.DataFrame(
        {
            "reference_month": [one, two, three, three, date(2020, 6, 1)],
            "seasonal_status": ["NSA", "NSA", "NSA", "SA", "NSA"],
            "transition": ["F_S"] * 5,
            "absolute_revision": [10.0, 20.0, 30.0, 5.0, 1e6],
            "period": ["main", "main", "main", "main", "pandemic"],
        }
    )
    table = revisions_by_quintile(revisions, quintiles)
    assert table.columns == [
        "quintile",
        "seasonal_status",
        "transition",
        "months",
        "deviation_low",
        "deviation_high",
        "n",
        "mean_absolute_revision",
    ]
    assert table.rows() == [
        (1, "NSA", "F_S", 2, -2.0, -1.0, 2, 15.0),
        (2, "NSA", "F_S", 1, 1.0, 1.0, 1, 30.0),
        (2, "SA", "F_S", 1, 1.0, 1.0, 1, 5.0),
    ]


def test_pandemic_by_window_averages_each_window():
    pandemic = [date(2020, 4, 1), date(2020, 5, 1), date(2020, 6, 1)]
    months = pl.DataFrame(
        {
            "reference_month": [*pandemic, date(2019, 6, 1)],
            "first_closing_days": [9, 9, 12, 9],
            "c1": [50.0, 60.0, 70.0, 1000.0],
            "period": ["pandemic", "pandemic", "pandemic", "main"],
        }
    )
    revisions = pl.DataFrame(
        {
            "reference_month": [*pandemic, date(2019, 6, 1)],
            "seasonal_status": ["NSA"] * 4,
            "transition": ["F_S"] * 4,
            "absolute_revision": [10.0, 20.0, 30.0, 1e6],
            "first_closing_days": [9, 9, 12, 9],
            "period": ["pandemic", "pandemic", "pandemic", "main"],
        }
    )
    table = pandemic_by_window(months, revisions)
    assert table.columns == [
        "first_closing_days",
        "months",
        "mean_c1",
        "NSA_F_S",
        "NSA_F_T",
        "SA_F_S",
        "SA_F_T",
    ]
    assert table.rows() == [
        (9, 2, 55.0, 15.0, None, None, None),
        (12, 1, 70.0, 30.0, None, None, None),
    ]


def test_rates_by_window_counts_main_months_and_averages_c1():
    months = pl.DataFrame(
        {
            "reference_month": [date(2010, m, 1) for m in (1, 2, 3)]
            + [date(2020, 6, 1)],
            "first_closing_days": [9, 9, 12, 9],
            "c1": [50.0, 60.0, 70.0, 1000.0],
            "period": ["main", "main", "main", "pandemic"],
        }
    )
    assert rates_by_window(months).rows() == [(9, 2, 55.0), (12, 1, 70.0)]
