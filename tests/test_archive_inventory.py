"""scripts/archive_inventory.py: vintages, Internet Archive captures, and per-vintage statuses."""

import io
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path

import archive_inventory
import pytest
from archive_inventory import (
    FIRST_REFERENCE_MONTH,
    RELEASE_INDEX_URL,
    Capture,
    Vintage,
    capture_from_row,
    capture_row,
    fetch,
    inspect_zip,
    list_captures,
    missing_reference_months,
    parse_cdx,
    parse_release_index,
    release_instant,
    retrying,
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


def make_zip(members: dict[str, bytes], date_time=(2026, 9, 1, 15, 33, 0)) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in members.items():
            archive.writestr(zipfile.ZipInfo(name, date_time=date_time), payload)
    return buffer.getvalue()


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


# --- Captures ------------------------------------------------------------------------------

# Member names in the Internet Archive's 2014-07-19 capture of ces.spec.other.zip.
OTHER_INPUTS_2014 = [
    "Fdumw96.dat",
    "NEW.readme.announcement.txt",
    "outliers.xls",
    "prior_adjustment_file.xls",
    "readme.other.txt",
    "DUMlp06.dat",
    "DUMlpel6.dat",
    "FDUM8606.dat",
    "Fdumel06.dat",
    "Fdumel96.dat",
    "Fdumpc96.dat",
    "Fdumpcw6.dat",
]


def test_inspect_zip_finds_prior_adjustments_and_outliers_among_the_2014_other_inputs():
    found = inspect_zip(make_zip(dict.fromkeys(OTHER_INPUTS_2014, b"x")))
    assert found["prior_adjustment"] and found["outliers"]
    assert found["specification"] == ""
    assert found["unexpected"] == ""


def test_inspect_zip_finds_specification_files_inside_a_folder():
    found = inspect_zip(make_zip({"ces.spec.ae/AE1011330000.spc": b"series{}"}))
    assert found["specification"]
    assert found["prior_adjustment"] == found["outliers"] == found["unexpected"] == ""


def test_inspect_zip_reports_a_member_no_readme_explains():
    found = inspect_zip(
        make_zip({"prior_adjustment_file.xlsx": b"x", "ces_input.dat": b"1 2"})
    )
    assert found["unexpected"] == "ces_input.dat"


def test_inspect_zip_explains_a_nested_copy_of_the_other_inputs_zip():
    # The Internet Archive's 2021-03-18 copy of ces.spec.other.zip nests an earlier copy of itself.
    nested = make_zip({"outliers.xlsx": b"draft AO"})
    found = inspect_zip(
        make_zip({"outliers.xlsx": b"AO", "ces.spec.other.zip": nested})
    )
    assert found["unexpected"] == ""
    assert (
        found["outliers"] == inspect_zip(make_zip({"outliers.xlsx": b"AO"}))["outliers"]
    )


def test_fingerprints_ignore_zip_timestamps_but_not_contents():
    members = {"outliers.xlsx": b"2020 AO"}
    first = inspect_zip(make_zip(members, date_time=(2025, 1, 1, 0, 0, 0)))
    rezipped = inspect_zip(make_zip(members, date_time=(2026, 1, 1, 0, 0, 0)))
    changed = inspect_zip(make_zip({"outliers.xlsx": b"2020 LS"}))
    assert first["outliers"] == rezipped["outliers"] != changed["outliers"]


def test_parse_cdx_reads_utc_timestamps_and_keeps_every_status():
    text = (
        "20140719134601 http://www.bls.gov/web/empsit/ces.spec.other.zip 200 "
        "ZSL4N7HMBW6G454PPITPXXHNHP34TVXH\n"
        "20170102045740 http://www.bls.gov/web/empsit/ces.spec.other.zip 301 "
        "3I42H3S6NNFQ2MSVX7XZKYAYSCX5QBYJ\n"
    )
    rows = parse_cdx(text)
    assert rows[0] == (
        datetime(2014, 7, 19, 13, 46, 1, tzinfo=UTC),
        "http://www.bls.gov/web/empsit/ces.spec.other.zip",
        "200",
        "ZSL4N7HMBW6G454PPITPXXHNHP34TVXH",
    )
    assert [status for _, _, status, _ in rows] == ["200", "301"]


def test_parse_cdx_of_no_captures_is_empty():
    assert parse_cdx("") == []


def test_parse_cdx_rejects_a_throttling_page():
    with pytest.raises(ValueError, match="not a CDX line"):
        parse_cdx("<html><body><p>Too many requests</p></body></html>")


def test_captures_round_trip_through_csv_rows():
    original = Capture(
        "other_inputs",
        "internet_archive",
        datetime(2014, 7, 19, 13, 46, 1, tzinfo=UTC),
        "https://web.archive.org/web/20140719134601id_/http://www.bls.gov/web/empsit/ces.spec.other.zip",
        "200",
        "ZSL4N7HMBW6G454PPITPXXHNHP34TVXH",
        prior_adjustment="ab12",
        outliers="cd34",
    )
    assert capture_from_row(capture_row(original)) == original


def test_retrying_returns_once_an_attempt_succeeds(monkeypatch):
    monkeypatch.setattr(archive_inventory, "sleep", lambda seconds: None)
    outcomes = iter([OSError("connection reset"), ValueError("throttled"), "payload"])

    def attempt():
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    assert retrying(attempt, what="test fetch") == "payload"


def test_retrying_gives_up_after_the_last_attempt(monkeypatch):
    monkeypatch.setattr(archive_inventory, "sleep", lambda seconds: None)

    def attempt():
        raise OSError("down")

    with pytest.raises(RuntimeError, match="test fetch failed after 5 attempts"):
        retrying(attempt, what="test fetch")


def test_collect_captures_records_an_unreplayable_copy_instead_of_failing(monkeypatch):
    artifact = archive_inventory.Artifact(
        "other_inputs",
        "https://www.bls.gov/web/empsit/ces.spec.other.zip",
        ("bls.gov/web/empsit/ces.spec.other.zip",),
    )
    original = "http://www.bls.gov/web/empsit/ces.spec.other.zip"
    listing = [
        (datetime(2015, 3, 21, 6, 17, 57, tzinfo=UTC), original, "200", "KEPT"),
        (datetime(2017, 1, 2, 4, 57, 40, tzinfo=UTC), original, "301", "MOVED"),
        (datetime(2020, 10, 18, 12, 35, 3, tzinfo=UTC), original, "200", "LOST"),
    ]
    payload = make_zip({"outliers.xlsx": b"AO"})

    def fake_fetch(url):
        if "20201018123503" in url:
            raise OSError("replay unavailable")
        return payload

    monkeypatch.setattr(archive_inventory, "ARTIFACTS", (artifact,))
    monkeypatch.setattr(archive_inventory, "list_captures", lambda key: listing)
    monkeypatch.setattr(archive_inventory, "fetch", fake_fetch)
    monkeypatch.setattr(archive_inventory, "sleep", lambda seconds: None)
    captures = archive_inventory.collect_captures(datetime(2026, 9, 13, 15, tzinfo=UTC))
    assert [(c.source, c.status, bool(c.outliers)) for c in captures] == [
        ("bls_current", "200", True),
        ("internet_archive", "200", True),
        ("internet_archive", "301", False),
        ("internet_archive", "unreplayable", False),
    ]
    assert (
        captures[1].url == f"https://web.archive.org/web/20150321061757id_/{original}"
    )


@pytest.mark.network
def test_live_cdx_lists_2014_captures_of_the_other_inputs_zip():
    rows = list_captures("bls.gov/web/empsit/ces.spec.other.zip")
    assert any(moment.year == 2014 and status == "200" for moment, _, status, _ in rows)
