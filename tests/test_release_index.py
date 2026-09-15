"""The release-date index of Employment Situation releases from January 1979."""

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest
import vintage_data

from ces_revisions.vintages import raw, release_index
from ces_revisions.vintages.months import add_months, month_range

FIXTURES = Path(__file__).parent / "fixtures" / "vintages"
ES_VINTAGES = raw.ROOT / "docs" / "inventory" / "es-vintages.csv"
RESCHEDULED = [
    date(1995, 12, 1),
    date(1998, 10, 1),
    date(2013, 9, 1),
    date(2013, 10, 1),
    date(2025, 9, 1),
    date(2025, 10, 1),
    date(2025, 11, 1),
    date(2026, 1, 1),
]


def test_add_months_crosses_year_boundaries_both_ways():
    assert add_months(date(2025, 11, 1), 2) == date(2026, 1, 1)
    assert add_months(date(2003, 5, 1), -1) == date(2003, 4, 1)
    assert add_months(date(2004, 1, 1), -13) == date(2002, 12, 1)


def test_month_range_includes_both_ends_and_is_empty_when_reversed():
    assert month_range(date(2025, 11, 1), date(2026, 1, 1)) == [
        date(2025, 11, 1),
        date(2025, 12, 1),
        date(2026, 1, 1),
    ]
    assert month_range(date(2026, 2, 1), date(2026, 1, 1)) == []


def test_historical_release_dates_read_the_employment_table_only():
    text = (FIXTURES / "histreleasedates-excerpt.txt").read_text(encoding="utf-8")
    dates = release_index.parse_historical_release_dates(text)
    assert dates[date(1959, 1, 1)] == date(1959, 2, 10)
    assert date(1959, 8, 1) not in dates
    assert dates[date(1995, 12, 1)] == date(1996, 1, 19)
    assert dates[date(1998, 10, 1)] == date(1998, 11, 5)
    assert dates[date(2000, 12, 1)] == date(2001, 1, 5)
    assert (min(dates), max(dates), len(dates)) == (
        date(1959, 1, 1),
        date(2000, 12, 1),
        45,
    )


def test_a_historical_row_without_twelve_dates_is_an_error():
    text = (
        "Release dates for national employment and unemployment estimates, 1957-2000\n"
        "   1999       February 5       March 5\n"
        "Release dates for the Consumer Price Index, 1953-2000\n"
    )
    with pytest.raises(ValueError, match="expected 12 release dates"):
        release_index.parse_historical_release_dates(text)


def test_philadelphia_fed_release_dates_read_iso_and_marked_cells():
    rows = [
        ("Release dates for the Employment Situation, 1966-2010", *[None] * 12),
        ("Year", "Jan.", "Feb.", *[None] * 10),
        ("1995", *["1995-02-03 00:00:00"] * 11, "1/19/96*"),
        ("1998", *[None] * 9, "11/5/98**", None, None),
        (
            "* Delayed due to Government shutdown and weather-related closing.",
            *[None] * 12,
        ),
    ]
    dates = release_index.parse_rtdsm_release_dates(rows)
    assert dates[date(1995, 1, 1)] == date(1995, 2, 3)
    assert dates[date(1995, 12, 1)] == date(1996, 1, 19)
    assert dates[date(1998, 10, 1)] == date(1998, 11, 5)
    assert len(dates) == 13


# --- The committed sources ------------------------------------------------------------------


def test_bls_historical_dates_agree_with_the_philadelphia_fed_copy_from_1979():
    historical = release_index.read_historical_release_dates()
    copy = release_index.read_rtdsm_release_dates()
    months = month_range(date(1979, 1, 1), date(2000, 12, 1))
    assert [historical[month] for month in months] == [copy[month] for month in months]


def test_archive_dates_agree_with_the_philadelphia_fed_copy_from_2001_on():
    archive = release_index.read_release_list(raw.RAW_DIR / raw.EMPSIT_RELEASES)
    copy = release_index.read_rtdsm_release_dates()
    # The copy's title says 1966-2010, but its last date is November 2010's release.
    assert (min(copy), max(copy)) == (date(1966, 1, 1), date(2010, 11, 1))
    months = month_range(date(2001, 1, 1), date(2010, 11, 1))
    assert [archive[month] for month in months] == [copy[month] for month in months]


def test_the_index_has_one_row_per_month_from_january_1979():
    latest = max(release_index.read_release_list(raw.RAW_DIR / raw.EMPSIT_RELEASES))
    months = vintage_data.index()["reference_month"].to_list()
    assert months == month_range(date(1979, 1, 1), latest)


def test_the_index_agrees_with_es_vintages_except_the_december_1999_repost():
    """The archive index links December 1999's release as a file of January 19, 2000; the
    release's own embargo line, BLS's historical dates, and the Philadelphia Fed copy all give
    January 7, 2000."""
    index = dict(
        vintage_data.index().select("reference_month", "release_date").iter_rows()
    )
    listed = release_index.read_release_list(ES_VINTAGES)
    differences = {
        month: (index[month], released)
        for month, released in listed.items()
        if index[month] != released
    }
    assert differences == {date(1999, 12, 1): (date(2000, 1, 7), date(2000, 1, 19))}


def test_february_1995_to_april_1999_releases_come_from_bls_historical_dates():
    rows = vintage_data.index().filter(
        pl.col("reference_month").is_between(date(1995, 2, 1), date(1999, 4, 1))
    )
    assert rows.height == 51
    assert rows["date_source"].unique().to_list() == ["histreleasedates"]
    assert rows["release_date"].null_count() == 0


def test_rescheduled_releases_keep_their_scheduled_dates():
    index = vintage_data.index()
    moved = index.filter(pl.col("off_schedule"))
    assert moved["reference_month"].to_list() == RESCHEDULED
    rows = {row["reference_month"]: row for row in moved.iter_rows(named=True)}
    october = rows[date(2025, 10, 1)]
    assert (
        october["status"],
        october["scheduled_date"],
        october["release_date"],
        october["published_date"],
    ) == ("canceled", date(2025, 11, 7), None, date(2025, 12, 16))
    assert rows[date(2013, 9, 1)]["scheduled_date"] is None
    assert rows[date(1995, 12, 1)]["scheduled_date"] == date(1996, 1, 5)
    assert index.filter(pl.col("status") == "canceled").height == 1
    unmoved = index.filter(~pl.col("off_schedule"))
    assert (unmoved["scheduled_date"] == unmoved["release_date"]).all()


def test_observable_at_is_the_release_time_in_eastern_time():
    rows = dict(
        vintage_data.index().select("reference_month", "observable_at").iter_rows()
    )
    assert rows[date(1998, 10, 1)] == datetime(1998, 11, 5, 18, 30, tzinfo=UTC)
    assert rows[date(2025, 8, 1)] == datetime(2025, 9, 5, 12, 30, tzinfo=UTC)
    assert rows[date(2026, 1, 1)] == datetime(2026, 2, 11, 13, 30, tzinfo=UTC)


def test_benchmark_releases_are_the_january_releases_from_2004():
    index = vintage_data.index()
    early = index.filter(pl.col("reference_month") < date(2003, 5, 1))
    assert early["benchmark_release"].null_count() == early.height
    later = index.filter(pl.col("reference_month") >= date(2003, 5, 1))
    assert later["benchmark_release"].null_count() == 0
    flagged = later.filter(pl.col("benchmark_release"))["reference_month"].to_list()
    assert flagged == [date(year, 1, 1) for year in range(2004, flagged[-1].year + 1)]


def test_no_calendar_month_holds_two_releases():
    released = vintage_data.index().filter(pl.col("status") == "released")
    assert released["release_date"].dt.truncate("1mo").n_unique() == released.height


def test_each_rescheduled_release_cites_a_bls_page():
    for row in release_index.read_reschedules().values():
        assert row["citation"].startswith("https://www.bls.gov/"), row
        assert row["status"] in ("released", "canceled"), row
