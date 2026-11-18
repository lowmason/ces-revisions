"""The collection-window day count `D_t`: federal business days after the 12th, before release."""

from collections import Counter
from datetime import date, timedelta

import polars as pl
import vintage_data

from ces_revisions.vintages.collection_days import (
    LAPSE_DELAYED_MONTHS,
    build_collection_days,
    collection_days,
    first_closing_date,
    first_closing_days,
)
from ces_revisions.vintages.federal_calendar import is_federal_holiday


def test_window_runs_from_after_the_12th_to_before_the_release():
    assert collection_days(date(2026, 8, 1), date(2026, 9, 4)) == 16


def test_december_window_skips_observed_saturday_holidays():
    assert collection_days(date(2010, 12, 1), date(2011, 1, 7)) == 17
    assert collection_days(date(2021, 12, 1), date(2022, 1, 7)) == 17


def test_window_starts_after_a_columbus_day_on_the_13th():
    assert collection_days(date(2014, 10, 1), date(2014, 11, 7)) == 18


def test_holidays_count_from_their_first_observance():
    assert collection_days(date(1985, 1, 1), date(1985, 2, 1)) == 14
    assert collection_days(date(1986, 1, 1), date(1986, 2, 7)) == 18
    assert collection_days(date(2021, 6, 1), date(2021, 7, 2)) == 13
    assert collection_days(date(2026, 6, 1), date(2026, 7, 2)) == 12


def test_lapse_days_count_as_business_days():
    assert collection_days(date(2026, 1, 1), date(2026, 2, 11)) == 20
    assert collection_days(date(2013, 9, 1), date(2013, 10, 22)) == 26


def test_every_release_has_one_day_count():
    frame = vintage_data.collection_days()
    assert frame.height == 571
    assert frame["reference_month"].is_unique().all()
    assert date(2025, 10, 1) not in frame["reference_month"].to_list()
    assert frame["collection_days"].null_count() == 0


def test_only_the_lapse_delayed_releases_are_flagged():
    frame = vintage_data.collection_days()
    assert LAPSE_DELAYED_MONTHS == (
        date(1995, 12, 1),
        date(2013, 9, 1),
        date(2013, 10, 1),
        date(2025, 9, 1),
        date(2025, 11, 1),
        date(2026, 1, 1),
    )
    assert frame.filter("lapse_delayed")["reference_month"].to_list() == list(
        LAPSE_DELAYED_MONTHS
    )


def test_built_day_counts_match_hand_counts():
    months = [date(1995, 12, 1), date(2013, 9, 1), date(2025, 9, 1), date(2026, 8, 1)]
    rows = vintage_data.collection_days().filter(
        pl.col("reference_month").is_in(months)
    )
    assert rows.sort("reference_month")["collection_days"].to_list() == [24, 26, 46, 16]


def test_day_counts_come_sorted_with_the_stated_schema():
    frame = build_collection_days(vintage_data.index().reverse())
    assert frame["reference_month"].is_sorted()
    assert frame.schema == {
        "reference_month": pl.Date,
        "release_date": pl.Date,
        "collection_days": pl.Int32,
        "lapse_delayed": pl.Boolean,
        "first_closing_date": pl.Date,
        "first_closing_days": pl.Int32,
    }


def test_first_closing_is_one_week_before_the_release():
    assert first_closing_date(date(2026, 7, 2)) == date(2026, 6, 25)


def test_window_to_first_closing_counts_through_the_closing_day():
    assert first_closing_days(date(2026, 6, 1), date(2026, 7, 2)) == 8
    assert first_closing_days(date(2026, 8, 1), date(2026, 9, 4)) == 12


def test_lapse_delayed_windows_follow_the_rule():
    months = [
        date(2013, 9, 1),
        date(2013, 10, 1),
        date(2025, 9, 1),
        date(2025, 11, 1),
        date(2026, 1, 1),
    ]
    rows = (
        vintage_data.collection_days()
        .filter(pl.col("reference_month").is_in(months))
        .sort("reference_month")
    )
    assert rows["first_closing_date"].to_list() == [
        date(2013, 10, 15),
        date(2013, 11, 1),
        date(2025, 11, 13),
        date(2025, 12, 9),
        date(2026, 2, 4),
    ]
    assert rows["first_closing_days"].to_list() == [22, 14, 42, 18, 16]


def _regular_releases_2003_2026() -> pl.DataFrame:
    return vintage_data.collection_days().filter(
        pl.col("reference_month").is_between(date(2003, 1, 1), date(2026, 8, 1)),
        ~pl.col("lapse_delayed"),
    )


def _final_week_has_a_holiday(closing: date, release: date) -> bool:
    days = (release - closing).days
    return any(
        is_federal_holiday(closing + timedelta(days=offset))
        for offset in range(1, days)
    )


def test_window_to_first_closing_is_four_days_short_or_three_with_a_holiday():
    rows = _regular_releases_2003_2026()
    assert rows.height == 278
    for closing, release, d_t, w_t in rows.select(
        "first_closing_date", "release_date", "collection_days", "first_closing_days"
    ).iter_rows():
        expected = 3 if _final_week_has_a_holiday(closing, release) else 4
        assert d_t - w_t == expected, (release, d_t, w_t)


def test_window_to_first_closing_counts_2003_2026():
    rows = _regular_releases_2003_2026()
    assert dict(Counter(rows["first_closing_days"].to_list())) == {
        8: 1,
        9: 26,
        10: 59,
        11: 36,
        12: 42,
        13: 52,
        14: 42,
        15: 20,
    }
    assert rows.filter(pl.col("first_closing_days") == 8)[
        "reference_month"
    ].to_list() == [date(2026, 6, 1)]
    difference = rows.with_columns(
        (pl.col("collection_days") - pl.col("first_closing_days")).alias("difference")
    )
    assert dict(Counter(difference["difference"].to_list())) == {4: 232, 3: 46}
    three = difference.filter(pl.col("difference") == 3)
    assert dict(Counter(month.month for month in three["reference_month"])) == {
        5: 13,
        6: 13,
        8: 10,
        12: 10,
    }


def test_six_closing_days_fall_on_a_holiday():
    rows = _regular_releases_2003_2026()
    on_holiday = [
        month
        for month, closing in rows.select(
            "reference_month", "first_closing_date"
        ).iter_rows()
        if is_federal_holiday(closing)
    ]
    assert on_holiday == [
        date(year, 12, 1) for year in (2004, 2009, 2010, 2015, 2020, 2021)
    ]
