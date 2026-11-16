"""scripts/archive_inventory.py: vintages, Internet Archive captures, and per-vintage statuses."""

import urllib.request
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import archive_inventory
import pytest
from archive_inventory import (
    BLS_SOURCE,
    CAPTURES_PATH,
    FILE_TYPES,
    FIRST_REFERENCE_MONTH,
    RELEASE_INDEX_URL,
    STATUSES,
    VINTAGES_PATH,
    Capture,
    Vintage,
    capture_from_row,
    capture_row,
    fetch,
    file_status,
    generated_block,
    inspect_zip,
    inventory_rows,
    list_captures,
    live_windows,
    member_release,
    missing_reference_months,
    newest_member,
    parse_cdx,
    parse_release_index,
    payload_digest,
    read_csv,
    release_instant,
    render_blocks,
    replace_generated,
    resolve_revisits,
    retrying,
    select_vintages,
    vintage_rows,
    vintages_from_rows,
)
from zip_fixtures import make_dated_zip, make_zip

FIXTURES = Path(__file__).parent / "fixtures" / "archive_inventory"


def index_excerpt() -> str:
    return (FIXTURES / "empsit-index-excerpt.html").read_text(encoding="utf-8")


def vintage(reference_month: str, released: str) -> Vintage:
    return Vintage(
        date.fromisoformat(f"{reference_month}-01"),
        date.fromisoformat(released),
        f"https://www.bls.gov/news.release/archives/empsit_{reference_month}.htm",
    )


def capture(timestamp: str, **fingerprints: str) -> Capture:
    return Capture(
        "other_inputs",
        "internet_archive",
        datetime.fromisoformat(timestamp),
        f"https://web.archive.org/web/{timestamp}",
        "200",
        "DIGEST",
        **fingerprints,
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


def test_releases_scheduled_after_now_are_dropped():
    kept = select_vintages(
        parse_release_index(index_excerpt()),
        now=datetime(2026, 9, 13, 15, tzinfo=UTC),
    )
    assert kept[-1].reference_month == date(2026, 8, 1)


def test_a_release_is_kept_only_once_its_embargo_lifts():
    vintages = parse_release_index(index_excerpt())
    # The September 2026 release is dated 2026-10-02; 00:30 UTC that day is still October 1 in
    # Washington, and the embargo lifts at 12:30 UTC.
    for moment, expected in (
        (datetime(2026, 10, 2, 0, 30, tzinfo=UTC), date(2026, 8, 1)),
        (datetime(2026, 10, 2, 12, 29, tzinfo=UTC), date(2026, 8, 1)),
        (datetime(2026, 10, 2, 12, 30, tzinfo=UTC), date(2026, 9, 1)),
    ):
        assert select_vintages(vintages, now=moment)[-1].reference_month == expected


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
    vintages = select_vintages(parse_release_index(html), now=datetime.now(UTC))
    first = next(v for v in vintages if v.reference_month == FIRST_REFERENCE_MONTH)
    assert first.release_date == date(2003, 6, 6)
    missing = missing_reference_months(vintages)
    assert date(2025, 10, 1) in missing
    assert all(month >= date(2025, 10, 1) for month in missing)


# --- Network -------------------------------------------------------------------------------

WITH_CONTACT = "ces-revisions/0.1.0 (someone@example.org)"
BARE = "ces-revisions/0.1.0"
REPLAY = "https://web.archive.org/web/20140719155542id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.bls.gov/web/empsit/ces.spec.ae.zip", WITH_CONTACT),
        ("https://download.bls.gov/pub/time.series/ce/ce.series", WITH_CONTACT),
        ("https://bls.gov/", WITH_CONTACT),
        (REPLAY, BARE),
        (
            "https://web.archive.org/cdx/search/cdx?url=bls.gov/web/empsit/ces.spec.ae.zip",
            BARE,
        ),
        ("https://bls.gov.example.com/", BARE),
        ("https://fraser.stlouisfed.org/title/employment-situation-4910", BARE),
        ("https://www.dol.gov/agencies/oasp", BARE),
    ],
)
def test_only_bls_hosts_get_the_contact_address(monkeypatch, url, expected):
    monkeypatch.setenv("BLS_CONTACT_EMAIL", "someone@example.org")
    assert archive_inventory.user_agent(url) == expected
    monkeypatch.delenv("BLS_CONTACT_EMAIL")
    assert archive_inventory.user_agent(url, "someone@example.org") == expected
    assert archive_inventory.user_agent(url) == BARE


@pytest.mark.parametrize(
    ("url", "is_bls"),
    [
        ("https://bls.gov/", True),
        ("https://www.bls.gov/web/empsit/ces.spec.ae.zip", True),
        ("https://download.bls.gov:443/pub/time.series/ce/ce.series", True),
        ("HTTPS://WWW.BLS.GOV/web/empsit/", True),
        ("https://notbls.gov/", False),
        ("https://www.bls.gov.example.com/", False),
        ("https://www.bls.gov@example.com/", False),
        ("https://example.com/?next=https://www.bls.gov/", False),
        ("https://example.com/www.bls.gov/", False),
        ("not a url", False),
    ],
)
def test_a_host_is_bls_when_it_is_bls_gov_or_a_subdomain(url, is_bls):
    assert archive_inventory.is_bls_host(url) is is_bls


def test_an_explicit_contact_beats_the_environment(monkeypatch):
    monkeypatch.setenv("BLS_CONTACT_EMAIL", "environment@example.org")
    url = "https://www.bls.gov/"
    assert archive_inventory.user_agent(url, "someone@example.org") == WITH_CONTACT
    assert archive_inventory.user_agent(url) == (
        "ces-revisions/0.1.0 (environment@example.org)"
    )


def test_an_explicit_blank_contact_is_not_replaced_by_the_environment(monkeypatch):
    monkeypatch.setenv("BLS_CONTACT_EMAIL", "environment@example.org")
    assert archive_inventory.user_agent("https://www.bls.gov/", "") == BARE


@pytest.mark.parametrize("contact", [None, ""])
def test_a_blank_contact_leaves_the_user_agent_bare(monkeypatch, contact):
    url = "https://www.bls.gov/"
    monkeypatch.setenv("BLS_CONTACT_EMAIL", "")
    assert archive_inventory.user_agent(url, contact) == BARE
    monkeypatch.delenv("BLS_CONTACT_EMAIL")
    assert archive_inventory.user_agent(url, contact) == BARE


class FakeResponse:
    headers: dict[str, str] = {}  # noqa: RUF012

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self) -> bytes:
        return b""


def test_fetch_sends_the_contact_to_bls_hosts_only(monkeypatch):
    monkeypatch.setenv("BLS_CONTACT_EMAIL", "someone@example.org")
    sent = []

    def fake_urlopen(request, timeout):
        sent.append(request.get_header("User-agent"))
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    fetch("https://www.bls.gov/web/empsit/ces.spec.ae.zip")
    fetch(REPLAY)
    assert sent == [WITH_CONTACT, BARE]


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


def test_inspect_zip_reports_an_unexplained_member_of_a_nested_zip():
    nested = make_zip({"outliers.xlsx": b"draft AO", "ces_input.dat": b"1 2"})
    found = inspect_zip(
        make_zip({"outliers.xlsx": b"AO", "ces.spec.other.zip": nested})
    )
    assert found["unexpected"] == "ces.spec.other.zip/ces_input.dat"


def test_fingerprints_ignore_zip_timestamps_but_not_contents():
    members = {"outliers.xlsx": b"2020 AO"}
    first = inspect_zip(make_zip(members, date_time=(2025, 1, 1, 0, 0, 0)))
    rezipped = inspect_zip(make_zip(members, date_time=(2026, 1, 1, 0, 0, 0)))
    changed = inspect_zip(make_zip({"outliers.xlsx": b"2020 LS"}))
    assert first["outliers"] == rezipped["outliers"] != changed["outliers"]


def test_newest_member_is_the_latest_member_date_not_the_last_member_written():
    payload = make_dated_zip(
        [
            ("readme.other.txt", (2021, 3, 1, 9, 0, 0), b"readme"),
            ("outliers.xlsx", (2021, 11, 3, 14, 22, 10), b"AO"),
            ("prior_adjustment_file.xlsx", (2021, 11, 2, 8, 0, 0), b"PA"),
        ]
    )
    # The ZIP's clock carries no timezone, so the value has no `Z`.
    assert newest_member(payload) == "2021-11-03T14:22:10"


def test_newest_member_dates_a_nested_zip_by_its_own_entry_only():
    # Same scope as the fingerprints: top-level members only. A nested ZIP counts by the date of
    # its entry in the outer ZIP, and the members inside it are not read.
    nested = make_dated_zip([("outliers.xlsx", (2030, 1, 1, 0, 0, 0), b"draft AO")])
    payload = make_dated_zip(
        [
            ("outliers.xlsx", (2021, 3, 16, 10, 0, 0), b"AO"),
            ("ces.spec.other.zip", (2021, 2, 1, 10, 0, 0), nested),
        ]
    )
    assert newest_member(payload) == "2021-03-16T10:00:00"


def test_newest_member_ignores_directory_entries():
    payload = make_dated_zip(
        [
            ("ces.spec.ae/", (2030, 1, 1, 0, 0, 0), b""),
            ("ces.spec.ae/AE1011330000.spc", (2026, 2, 3, 11, 5, 0), b"series{}"),
        ]
    )
    assert newest_member(payload) == "2026-02-03T11:05:00"


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
        newest_member="2014-07-03T16:20:00",
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


def test_collect_archive_copies_records_an_unreplayable_copy_instead_of_failing(
    monkeypatch,
):
    artifact = archive_inventory.Artifact(
        "other_inputs",
        "https://www.bls.gov/web/empsit/ces.spec.other.zip",
        ("bls.gov/web/empsit/ces.spec.other.zip",),
    )
    original = "http://www.bls.gov/web/empsit/ces.spec.other.zip"
    payload = make_zip({"outliers.xlsx": b"AO"})
    listing = [
        (
            datetime(2015, 3, 21, 6, 17, 57, tzinfo=UTC),
            original,
            "200",
            payload_digest(payload),
        ),
        (datetime(2017, 1, 2, 4, 57, 40, tzinfo=UTC), original, "301", "MOVED"),
        (datetime(2020, 10, 18, 12, 35, 3, tzinfo=UTC), original, "200", "LOST"),
    ]

    def fake_fetch(url):
        if "20201018123503" in url:
            raise OSError("replay unavailable")
        return payload

    monkeypatch.setattr(archive_inventory, "ARTIFACTS", (artifact,))
    monkeypatch.setattr(archive_inventory, "list_captures", lambda key: listing)
    monkeypatch.setattr(archive_inventory, "fetch", fake_fetch)
    monkeypatch.setattr(archive_inventory, "sleep", lambda seconds: None)
    copies = archive_inventory.collect_archive_copies()
    assert [
        (c.source, c.status, bool(c.outliers), c.newest_member) for c, _ in copies
    ] == [
        ("internet_archive", "200", True, "2026-09-01T15:33:00"),
        ("internet_archive", "301", False, ""),
        ("internet_archive", "unreplayable", False, ""),
    ]
    assert [held for _, held in copies] == [payload, None, None]
    assert (
        copies[0][0].url == f"https://web.archive.org/web/20150321061757id_/{original}"
    )


@pytest.mark.network
def test_live_cdx_lists_2014_captures_of_the_other_inputs_zip():
    rows = list_captures("bls.gov/web/empsit/ces.spec.other.zip")
    assert any(moment.year == 2014 and status == "200" for moment, _, status, _ in rows)


# --- Inventory -----------------------------------------------------------------------------


def test_monthly_windows_run_from_one_release_to_the_next():
    vintages = [
        vintage("2025-07", "2025-08-01"),
        vintage("2025-08", "2025-09-05"),
        vintage("2025-09", "2025-11-20"),
    ]
    windows = live_windows(vintages, annual=False)
    assert windows[date(2025, 8, 1)] == (
        release_instant(date(2025, 9, 5)),
        release_instant(date(2025, 11, 20)),
    )
    assert windows[date(2025, 9, 1)] == (release_instant(date(2025, 11, 20)), None)


def test_specification_windows_run_from_benchmark_release_to_benchmark_release():
    vintages = [
        vintage("2024-12", "2025-01-10"),
        vintage("2025-01", "2025-02-07"),
        vintage("2025-02", "2025-03-07"),
        vintage("2025-12", "2026-01-09"),
        vintage("2026-01", "2026-02-11"),
    ]
    windows = live_windows(vintages, annual=True)
    benchmark_2025 = (
        release_instant(date(2025, 2, 7)),
        release_instant(date(2026, 2, 11)),
    )
    assert windows[date(2025, 1, 1)] == benchmark_2025
    assert windows[date(2025, 12, 1)] == benchmark_2025
    assert windows[date(2026, 1, 1)] == (release_instant(date(2026, 2, 11)), None)


def test_a_capture_a_minute_before_the_embargo_belongs_to_the_previous_vintage():
    windows = live_windows(
        [vintage("2025-07", "2025-08-01"), vintage("2025-08", "2025-09-05")],
        annual=False,
    )
    before = capture("2025-09-05T12:29:00+00:00", outliers="a")
    after = capture("2025-09-05T12:31:00+00:00", outliers="b")
    assert file_status([before, after], "outliers", windows[date(2025, 7, 1)]) == (
        "internet_archive",
        before.url,
    )
    assert file_status([before, after], "outliers", windows[date(2025, 8, 1)]) == (
        "internet_archive",
        after.url,
    )


def test_file_status_prefers_the_bls_copy_and_ignores_captures_without_content():
    window = (datetime(2026, 9, 4, 12, 30, tzinfo=UTC), None)
    archived = capture("2026-09-05T00:00:00+00:00", outliers="same")
    current = Capture(
        "other_inputs",
        BLS_SOURCE,
        datetime(2026, 9, 13, 15, 0, tzinfo=UTC),
        "https://www.bls.gov/web/empsit/ces.spec.other.zip",
        "200",
        "DIGEST",
        outliers="same",
    )
    redirect = Capture(
        "other_inputs",
        "internet_archive",
        datetime(2026, 9, 6, tzinfo=UTC),
        "https://web.archive.org/web/redirect",
        "301",
        "DIGEST",
    )
    assert file_status([archived, current, redirect], "outliers", window) == (
        BLS_SOURCE,
        current.url,
    )


def test_file_status_flags_copies_that_disagree_within_one_window():
    window = (
        datetime(2021, 2, 5, 13, 30, tzinfo=UTC),
        datetime(2021, 3, 5, 13, 30, tzinfo=UTC),
    )
    first = capture("2021-02-10T00:00:00+00:00", prior_adjustment="a")
    second = capture("2021-02-20T00:00:00+00:00", prior_adjustment="b")
    assert file_status([second, first], "prior_adjustment", window) == (
        "conflicting_captures",
        f"{first.url} {second.url}",
    )


def test_file_status_without_a_copy_is_not_archived():
    window = (
        datetime(2008, 2, 1, 13, 30, tzinfo=UTC),
        datetime(2008, 3, 7, 13, 30, tzinfo=UTC),
    )
    assert file_status([], "specification", window) == ("not_archived", "")


def test_inventory_rows_cover_every_reference_month_and_mark_the_gap():
    vintages = [
        vintage("2003-04", "2003-05-02"),
        vintage("2003-05", "2003-06-06"),
        vintage("2003-06", "2003-07-03"),
        vintage("2003-08", "2003-09-05"),
    ]
    rows = inventory_rows(vintages, [])
    assert [row["reference_month"] for row in rows] == [
        "2003-05",
        "2003-06",
        "2003-07",
        "2003-08",
    ]
    gap = rows[2]
    assert (gap["release_date"], gap["specification"], gap["unrounded_nsa_inputs"]) == (
        "",
        "no_release",
        "no_release",
    )
    assert (rows[0]["prior_adjustment"], rows[0]["unrounded_nsa_inputs"]) == (
        "not_archived",
        "not_published",
    )


def test_files_complete_needs_all_three_file_types():
    vintages = [vintage("2003-05", "2003-06-06")]
    all_three = capture(
        "2003-06-10T00:00:00+00:00",
        specification="s",
        prior_adjustment="p",
        outliers="o",
    )
    two = capture("2003-06-10T00:00:00+00:00", specification="s", prior_adjustment="p")
    assert inventory_rows(vintages, [all_three])[0]["files_complete"] == "true"
    assert inventory_rows(vintages, [two])[0]["files_complete"] == "false"


def test_a_bls_copy_counts_as_present_for_each_file_type():
    vintages = [vintage("2003-05", "2003-06-06")]
    fetched = replace(
        capture(
            "2003-06-10T00:00:00+00:00",
            specification="s",
            prior_adjustment="p",
            outliers="o",
        ),
        source=BLS_SOURCE,
    )
    row = inventory_rows(vintages, [fetched])[0]
    assert [row[file_type] for file_type in FILE_TYPES] == [BLS_SOURCE] * 3
    assert row["files_complete"] == "true"
    assert BLS_SOURCE in STATUSES


def test_a_revisit_record_evidences_its_window_with_its_twin_fingerprints():
    vintages = [vintage("2003-05", "2003-06-06"), vintage("2003-06", "2003-07-03")]
    opened = replace(capture("2003-06-10T00:00:00+00:00", outliers="o"), digest="SAME")
    revisit = replace(capture("2003-07-10T00:00:00+00:00"), status="-", digest="SAME")
    rows = inventory_rows(vintages, [opened, revisit])
    assert [(row["outliers"], row["outliers_evidence"]) for row in rows] == [
        ("internet_archive", opened.url),
        ("internet_archive", revisit.url),
    ]


def test_a_revisit_record_without_an_opened_twin_evidences_nothing():
    vintages = [vintage("2003-05", "2003-06-06")]
    revisit = replace(capture("2003-06-10T00:00:00+00:00"), status="-", digest="ALONE")
    assert inventory_rows(vintages, [revisit])[0]["outliers"] == "not_archived"


def test_a_revisit_record_takes_its_twins_newest_member():
    opened = replace(
        capture("2003-06-10T00:00:00+00:00", outliers="o"),
        digest="SAME",
        newest_member="2003-06-04T10:00:00",
    )
    revisit = replace(capture("2003-07-10T00:00:00+00:00"), status="-", digest="SAME")
    assert resolve_revisits([opened, revisit])[1].newest_member == "2003-06-04T10:00:00"


SUMMER_2015 = [
    vintage("2015-06", "2015-07-02"),
    vintage("2015-07", "2015-08-07"),
    vintage("2015-08", "2015-09-04"),
    vintage("2015-09", "2015-10-02"),
]


def statuses(rows: list[dict[str, str]], file_type: str) -> dict[str, str]:
    """One file type's status by reference month, for the months that had a release."""
    return {
        row["reference_month"]: row[file_type] for row in rows if row["release_date"]
    }


def test_a_monthly_copy_counts_for_the_release_its_files_were_made_for():
    # The Internet Archive's copy of 2015-09-05, a day after the August release, holds files dated
    # 2015-08-04: they were made for the July release of 2015-08-07.
    stale = replace(
        capture("2015-09-05T16:47:51+00:00", prior_adjustment="p", outliers="o"),
        newest_member="2015-08-04T13:40:50",
    )
    rows = inventory_rows(SUMMER_2015, [stale])
    for file_type in ("prior_adjustment", "outliers"):
        assert statuses(rows, file_type) == {
            "2015-06": "not_archived",
            "2015-07": "internet_archive",
            "2015-08": "not_archived",
            "2015-09": "not_archived",
        }


def test_files_dated_on_a_release_day_belong_to_that_release():
    copy = replace(
        capture("2015-08-10T00:00:00+00:00", outliers="o"),
        newest_member="2015-08-07T07:00:00",
    )
    assert statuses(inventory_rows(SUMMER_2015, [copy]), "outliers")["2015-07"] == (
        "internet_archive"
    )


def test_a_monthly_copy_with_files_newer_than_every_release_evidences_none():
    copy = replace(
        capture("2015-10-20T00:00:00+00:00", outliers="o"),
        newest_member="2015-10-05T09:00:00",
    )
    assert set(statuses(inventory_rows(SUMMER_2015, [copy]), "outliers").values()) == {
        "not_archived"
    }


def test_specification_files_are_still_credited_by_the_copy_time():
    vintages = [vintage("2014-12", "2015-01-09"), vintage("2015-01", "2015-02-06")]
    # By member date these files would belong to the December release, before the benchmark.
    copy = replace(
        capture("2015-02-10T00:00:00+00:00", specification="s"),
        newest_member="2015-01-05T09:00:00",
    )
    assert statuses(inventory_rows(vintages, [copy]), "specification") == {
        "2014-12": "not_archived",
        "2015-01": "internet_archive",
    }


# --- Committed evidence: member dates against the live windows -----------------------------


def test_only_an_opened_copy_records_a_member_date():
    # The no-payload rule. Only a copy whose ZIP was downloaded and opened has members to date:
    # BLS's current file and an Internet Archive copy with status 200. A redirect (301), a refusal
    # (403), an `unreplayable` copy, and a revisit record (`-`), which `captures` never downloads,
    # record no date. A revisit takes its twin's date in `resolve_revisits`, as its fingerprints do.
    rows = read_csv(CAPTURES_PATH)
    assert rows
    assert [
        row["url"]
        for row in rows
        if bool(row["newest_member"]) != (row["status"] == "200")
    ] == []


# Days from a monthly copy's newest member to the release its files were made for. The evidence of
# 2026-11-15 spans 0 to 4 days over 33 copies, and releases are about 4 weeks apart.
BUILD_LEAD_DAYS = 7

# The specification copies whose newest member postdates the benchmark release whose window holds
# them. Every copy in the 2024 window has newest member 2024-03-06, 33 days after the 2024-02-02
# benchmark release, while every other benchmark year's copies predate their release by 1 to 11
# days. No copy between 2024-02-02 and 2024-03-06 survives to show whether the `.spc` files or only
# the readme changed, so 2024-01's specification status rests on files built after its release.
SPECIFICATIONS_BUILT_AFTER_THEIR_BENCHMARK = {
    f"https://web.archive.org/web/{stamp}id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip"
    for stamp in (
        "20240927002315",
        "20241127194400",
        "20241211002903",
        "20250101154028",
        "20250202014522",
    )
}


def test_every_copy_was_made_after_the_release_its_files_were_made_for():
    """Each copy's files must predate the copy and belong to a release already out by then.

    `inventory` credits a prior-adjustment or outlier copy to the release its files were made
    for, the first on or after its newest member's date (`member_release`), and not to the
    release whose window holds the copy: on the August 2015 and May 2026 release days, BLS
    posted as current a ZIP built for an earlier release (July 2015 and February 2026).

    Each copy that `file_status` can credit (status 200, or a revisit resolved to an opened twin)
    is checked. A ZIP member's time is the zipping machine's clock with no timezone, so member
    times are compared as calendar dates:

    - Every such copy: the newest member's date is on or before the copy's UTC date. For a clock
      at or west of UTC, as BLS's Eastern clock is, the local date of an instant never exceeds
      its UTC date.
    - A copy with any prior-adjustment or outlier fingerprint: the 8:30 a.m. Eastern embargo of
      the release its files were made for precedes the copy, so the files were not posted early.
      Their newest member's date is also at most `BUILD_LEAD_DAYS` before that release. A ZIP
      rebuilt after a release would be credited to the next release, about 4 weeks later, and
      fails here, as would a change in how far ahead BLS builds the files.

    A specification-only copy is credited by its own time under the annual benchmark window. Its
    newest member's date must also be on or before the benchmark release whose window holds the
    copy, since BLS builds the specification files before that release. The copies that break
    this are exactly `SPECIFICATIONS_BUILT_AFTER_THEIR_BENCHMARK`, so a new one fails here, and
    so does a change to the evidence for those.
    """
    vintages = vintages_from_rows(read_csv(VINTAGES_PATH))
    benchmarks = [vintage for vintage in vintages if vintage.reference_month.month == 1]
    captures = resolve_revisits(
        [capture_from_row(row) for row in read_csv(CAPTURES_PATH)]
    )
    credited = [
        capture
        for capture in captures
        if capture.status in ("200", "-")
        and (capture.specification or capture.prior_adjustment or capture.outliers)
    ]
    monthly = [copy for copy in credited if copy.prior_adjustment or copy.outliers]
    assert len(credited) > len(monthly) > 0
    outside = []
    built_after_benchmark = set()
    for copy in credited:
        if not copy.newest_member:
            outside.append(f"{copy.url}: credited but has no member date")
            continue
        member_day = datetime.fromisoformat(copy.newest_member).date()
        if member_day > copy.timestamp.date():
            outside.append(f"{copy.url}: files dated {member_day}, after the copy")
        if copy in monthly:
            release = member_release(copy, vintages)
            if release is None:
                outside.append(
                    f"{copy.url}: files dated {member_day}, after every release"
                )
            elif release_instant(release.release_date) > copy.timestamp:
                outside.append(
                    f"{copy.url}: files dated {member_day}, made for the release of"
                    f" {release.release_date}, which came after the copy"
                )
            elif (release.release_date - member_day).days > BUILD_LEAD_DAYS:
                outside.append(
                    f"{copy.url}: files dated {member_day}, more than {BUILD_LEAD_DAYS}"
                    f" days before the release of {release.release_date}"
                )
        else:
            opened = [
                benchmark
                for benchmark in benchmarks
                if release_instant(benchmark.release_date) <= copy.timestamp
            ]
            if opened and member_day > opened[-1].release_date:
                built_after_benchmark.add(copy.url)
    assert outside == []
    assert built_after_benchmark == SPECIFICATIONS_BUILT_AFTER_THEIR_BENCHMARK


def test_generated_blocks_are_replaced_in_place():
    document = (
        "intro\n<!-- BEGIN GENERATED archive-coverage -->\nold\n"
        "<!-- END GENERATED archive-coverage -->\noutro\n"
    )
    updated = replace_generated(document, "archive-coverage", "new\n")
    assert updated == document.replace("\nold\n", "\nnew\n")
    assert generated_block(updated, "archive-coverage") == "new\n"


def test_a_missing_generated_block_is_an_error():
    with pytest.raises(ValueError, match="archive-vintages"):
        replace_generated("no markers here", "archive-vintages", "body\n")


def test_rendered_vintage_rows_link_the_copy_that_evidences_them():
    kept = capture("2003-06-10T00:00:00+00:00", outliers="o")
    rows = inventory_rows([vintage("2003-05", "2003-06-06")], [kept])
    assert (
        f"| 2003-05 | 2003-06-06 | `not_archived` | `not_archived` | [`internet_archive`]({kept.url}) "
        "| `not_published` | false |"
    ) in render_blocks(rows)["archive-vintages"]
