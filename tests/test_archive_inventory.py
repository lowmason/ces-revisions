"""scripts/archive_inventory.py: vintages, Internet Archive captures, and per-vintage statuses."""

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from archive_inventory import (
    FIRST_REFERENCE_MONTH,
    RELEASE_INDEX_URL,
    Vintage,
    fetch,
    missing_reference_months,
    parse_release_index,
    release_instant,
    select_vintages,
    vintage_rows,
    vintages_from_rows,
)

FIXTURES = Path(__file__).parent / "fixtures" / "archive_inventory"


def index_excerpt() -> str:
    return (FIXTURES / "empsit-index-excerpt.html").read_text(encoding="utf-8")


def vintage(reference_month: str, released: str) -> Vintage:
    return Vintage(
        date.fromisoformat(f"{reference_month}-01"),
        date.fromisoformat(released),
        f"https://www.bls.gov/news.release/archives/empsit_{reference_month}.htm",
    )


# --- Vintages ------------------------------------------------------------------------------


def test_release_index_yields_one_vintage_per_reference_month():
    vintages = parse_release_index(index_excerpt())
    assert [
        (f"{v.reference_month:%Y-%m}", v.release_date.isoformat()) for v in vintages
    ] == [
        ("2003-04", "2003-05-02"),
        ("2003-05", "2003-06-06"),
        ("2019-03", "2019-04-05"),
        ("2025-08", "2025-09-05"),
        ("2025-09", "2025-11-20"),
        ("2025-11", "2025-12-16"),
        ("2026-08", "2026-09-04"),
        ("2026-09", "2026-10-02"),
    ]


def test_release_urls_prefer_html_then_text_over_pdf():
    urls = {
        f"{v.reference_month:%Y-%m}": v.release_url
        for v in parse_release_index(index_excerpt())
    }
    assert (
        urls["2003-05"]
        == "https://www.bls.gov/news.release/history/empsit_06062003.txt"
    )
    assert (
        urls["2025-09"]
        == "https://www.bls.gov/news.release/archives/empsit_11202025.htm"
    )
    assert (
        urls["2019-03"]
        == "https://www.bls.gov/news.release/archives/empsit_04052019.htm"
    )


def test_two_release_dates_for_one_reference_month_are_an_error():
    html = (
        '<li><a href="/news.release/archives/empsit_09052025.htm">'
        "August 2025 Employment Situation</a></li>"
        '<li><a href="/news.release/archives/empsit_09122025.htm">'
        "August 2025 Employment Situation</a></li>"
    )
    with pytest.raises(ValueError, match="two releases for 2025-08"):
        parse_release_index(html)


def test_releases_scheduled_after_the_as_of_date_are_dropped():
    kept = select_vintages(
        parse_release_index(index_excerpt()), as_of=date(2026, 9, 13)
    )
    assert kept[-1].reference_month == date(2026, 8, 1)


def test_missing_reference_months_finds_the_absent_october_2025_release():
    vintages = [
        vintage("2025-08", "2025-09-05"),
        vintage("2025-09", "2025-11-20"),
        vintage("2025-11", "2025-12-16"),
    ]
    assert missing_reference_months(vintages, start=date(2025, 8, 1)) == [
        date(2025, 10, 1)
    ]


@pytest.mark.parametrize(
    ("released", "instant"),
    [
        ("2025-09-05", "2025-09-05T12:30:00+00:00"),
        ("2026-02-11", "2026-02-11T13:30:00+00:00"),
    ],
    ids=["daylight time", "standard time"],
)
def test_the_embargo_lifts_at_eight_thirty_eastern(released, instant):
    assert release_instant(date.fromisoformat(released)) == datetime.fromisoformat(
        instant
    )


def test_vintages_round_trip_through_csv_rows():
    vintages = parse_release_index(index_excerpt())
    assert vintages_from_rows(vintage_rows(vintages)) == vintages


@pytest.mark.network
def test_live_release_index_starts_the_modern_vintages_on_june_6_2003():
    html = fetch(RELEASE_INDEX_URL).decode("utf-8", "replace")
    vintages = select_vintages(
        parse_release_index(html), as_of=datetime.now(UTC).date()
    )
    first = next(v for v in vintages if v.reference_month == FIRST_REFERENCE_MONTH)
    assert first.release_date == date(2003, 6, 6)
    missing = missing_reference_months(vintages)
    assert date(2025, 10, 1) in missing
    assert all(month >= date(2025, 10, 1) for month in missing)
