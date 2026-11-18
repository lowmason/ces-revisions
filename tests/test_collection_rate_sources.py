"""Staging the collection-rate sources, the walk down a year's archive copies, and promotion."""

import hashlib
import http.client
import io
from datetime import UTC, datetime
from pathlib import Path

import collection_rate_sources as sources
import pytest
import vintage_sources

from ces_revisions.vintages import raw
from ces_revisions.vintages.collection_rates import ARCHIVE_YEARS, RECON_COPY

FIXTURES = Path(__file__).parent / "fixtures" / "vintages"
NOW = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)
LAST_MODIFIED = "Tue, 10 Nov 2026 13:30:00 GMT"
OLD_PAGE = "http://www.bls.gov/web/cesregrec.htm"
NEW_PAGE = "https://www.bls.gov/web/empsit/cesregrec.htm"
LABSTAT_BODY = (
    b"series_id                     \tyear\tperiod\t       value\tfootnote_codes\r\n"
    b"CEU00000000001                \t2003\tM01\t      130000\t\r\n"
    b"CEU00000000C1                 \t2003\tM01\t        65.2\t\r\n"
    b"CEU00000000C2                 \t2003\tM01\t        80.1\t\r\n"
    b"CEU00000000C3                 \t2003\tM01\t        90.3\t\r\n"
    b"CEU05000000RR                 \t2003\tM01\t         1.0\t\r\n"
    b"CEU05000000001                \t2003\tM01\t      110000\t\r\n"
)
FOOTNOTE_BODY = b"footnote_code\tfootnote_text\r\nE\tExample footnote.\r\n"


def valid_page() -> bytes:
    return (FIXTURES / "cesregrec-synthetic.htm").read_bytes()


INVALID_PAGE = b"<html><body><p>No table here.</p></body></html>"


class FakeResponse:
    """A response with a body, headers, and optionally a stream that ends too soon."""

    def __init__(
        self,
        body: bytes,
        headers: dict[str, str] | None = None,
        *,
        incomplete: bool = False,
    ) -> None:
        self.stream = io.BytesIO(body)
        self.headers = http.client.HTTPMessage()
        for name, value in (headers or {}).items():
            self.headers[name] = value
        self.incomplete = incomplete

    def read(self, amount: int = -1) -> bytes:
        chunk = self.stream.read(amount)
        if self.incomplete and not chunk:
            raise http.client.IncompleteRead(b"")
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeNetwork:
    """Answers requests from a table of bodies by URL, and records requests and pauses."""

    def __init__(self, monkeypatch, bodies: dict[str, bytes | FakeResponse]) -> None:
        self.bodies = bodies
        self.events: list[str] = []
        monkeypatch.setattr(sources.archive_inventory, "open_url", self.open_url)
        monkeypatch.setattr(sources, "sleep", self.sleep)
        monkeypatch.setattr(sources.archive_inventory, "sleep", self.sleep)

    def open_url(self, url: str, timeout: float, contact: str | None = None):
        self.events.append(url)
        body = self.bodies[url]
        if isinstance(body, FakeResponse):
            return body
        return FakeResponse(
            body,
            {"Content-Length": str(len(body)), "Last-Modified": LAST_MODIFIED},
        )

    def sleep(self, seconds: float) -> None:
        self.events.append(f"sleep {seconds}")

    @property
    def requests(self) -> list[str]:
        return [event for event in self.events if not event.startswith("sleep")]


def replay(stamp: str, original: str = OLD_PAGE) -> str:
    return sources.archive_inventory.REPLAY_URL.format(stamp=stamp, original=original)


def cdx(key: str) -> str:
    return sources.archive_inventory.CDX_URL.format(key=key)


def listing(*captures: tuple[str, str, str]) -> bytes:
    lines = [
        f"{stamp} {original} {status} DIGEST" for stamp, original, status in captures
    ]
    return ("\n".join(lines) + "\n").encode()


def recon_capture() -> tuple[str, str, str]:
    return (RECON_COPY, NEW_PAGE, "200")


def listings(old: list[tuple[str, str, str]], new: list[tuple[str, str, str]]):
    return {
        cdx(sources.PAGE_KEYS[0]): listing(*old),
        cdx(sources.PAGE_KEYS[1]): listing(*new),
    }


def stage_copy(staging: Path, stamp: str, page: bytes) -> None:
    sources.stage(
        staging,
        f"{raw.REGISTRY_COPIES}/{stamp}.htm",
        page,
        url=replay(stamp),
        fetched_at="2026-11-17T12:00:00Z",
    )


def staged_tree(staging: Path, *, skip_year: int | None = None) -> list[str]:
    """A complete staging tree: listings, a valid copy of each year, LABSTAT and footnotes."""
    old = [
        (f"{year}1215120000", OLD_PAGE, "200")
        for year in ARCHIVE_YEARS
        if year != skip_year
    ]
    for name, body in listings(old, [recon_capture()]).items():
        index = [cdx(key) for key in sources.PAGE_KEYS].index(name)
        (staging / "cdx").mkdir(parents=True, exist_ok=True)
        (staging / "cdx" / f"{index}.txt").write_bytes(body)
    stamps = [stamp for stamp, _, _ in old] + [RECON_COPY]
    for stamp in stamps:
        stage_copy(staging, stamp, valid_page())
    sources.stage(
        staging,
        raw.COLLECTION_RATES,
        sources.extract_series(io.BytesIO(LABSTAT_BODY), sources.SERIES),
        derived_from=f"{sources.LABSTAT_URL}, lines of {' '.join(sources.SERIES)}",
        derived_from_sha256=hashlib.sha256(LABSTAT_BODY).hexdigest(),
        last_modified=LAST_MODIFIED,
        fetched_at="2026-11-17T12:00:00Z",
    )
    sources.stage(
        staging,
        raw.CE_FOOTNOTES,
        FOOTNOTE_BODY,
        url=sources.FOOTNOTE_URL,
        last_modified=LAST_MODIFIED,
        fetched_at="2026-11-17T12:00:00Z",
    )
    return stamps


def raw_tree(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    existing = raw_dir / raw.RESCHEDULES
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"release_date,citation\n2025-11-20,example\n")
    row = vintage_sources.manifest_row(
        raw.RESCHEDULES,
        "",
        raw_dir=raw_dir,
        derived_from="hand-keyed from the citation on each row",
    )
    sources.archive_inventory.write_csv(
        raw_dir / raw.MANIFEST, [row], vintage_sources.MANIFEST_COLUMNS
    )
    return raw_dir


def snapshot(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_extract_keeps_the_header_and_the_four_series_byte_for_byte():
    lines = LABSTAT_BODY.splitlines(keepends=True)
    assert sources.extract_series(io.BytesIO(LABSTAT_BODY), sources.SERIES) == (
        lines[0] + lines[2] + lines[3] + lines[4] + lines[5]
    )


def test_short_download_stages_nothing(monkeypatch, tmp_path):
    short = FakeResponse(LABSTAT_BODY[:-10], {"Content-Length": str(len(LABSTAT_BODY))})
    FakeNetwork(monkeypatch, {sources.LABSTAT_URL: short})
    with pytest.raises(ValueError, match="bytes"):
        sources.stream_download(sources.LABSTAT_URL, tmp_path / "short")
    assert not (tmp_path / "short").exists()

    ended = FakeResponse(
        LABSTAT_BODY[:-10],
        {"Content-Length": str(len(LABSTAT_BODY))},
        incomplete=True,
    )
    FakeNetwork(monkeypatch, {sources.LABSTAT_URL: ended})
    with pytest.raises(ValueError, match="ended"):
        sources.stream_download(sources.LABSTAT_URL, tmp_path / "ended")
    assert not (tmp_path / "ended").exists()

    monkeypatch.setenv("BLS_CONTACT_EMAIL", "someone@example.org")
    for incomplete in (False, True):
        staging = tmp_path / f"staging-{incomplete}"
        cut = FakeResponse(
            LABSTAT_BODY[:-10],
            {"Content-Length": str(len(LABSTAT_BODY))},
            incomplete=incomplete,
        )
        network = FakeNetwork(monkeypatch, {sources.LABSTAT_URL: cut})
        with pytest.raises(RuntimeError):
            sources.fetch_labstat(NOW, staging)
        assert [path for path in staging.rglob("*") if path.is_file()] == []
        assert network.requests == [sources.LABSTAT_URL] * (
            sources.archive_inventory.ATTEMPTS
        )


def test_fetch_labstat_refuses_without_a_contact_address(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("BLS_CONTACT_EMAIL", raising=False)
    network = FakeNetwork(monkeypatch, {})
    assert sources.fetch_labstat(NOW, tmp_path / "staging") == 1
    assert network.requests == []
    assert "BLS_CONTACT_EMAIL" in capsys.readouterr().err


def test_fetch_labstat_stages_the_extract_footnotes_and_provenance(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("BLS_CONTACT_EMAIL", "someone@example.org")
    FakeNetwork(
        monkeypatch,
        {sources.LABSTAT_URL: LABSTAT_BODY, sources.FOOTNOTE_URL: FOOTNOTE_BODY},
    )
    staging = tmp_path / "staging"
    assert sources.fetch_labstat(NOW, staging) == 0
    assert (staging / "raw" / raw.COLLECTION_RATES).read_bytes() == (
        sources.extract_series(io.BytesIO(LABSTAT_BODY), sources.SERIES)
    )
    assert (staging / "raw" / raw.CE_FOOTNOTES).read_bytes() == FOOTNOTE_BODY
    assert list((staging / "download").iterdir()) == []
    rows = {
        row["file"]: row
        for row in sources.archive_inventory.read_csv(staging / sources.PROVENANCE)
    }
    extract = rows[raw.COLLECTION_RATES]
    assert extract["derived_from"].startswith(sources.LABSTAT_URL)
    assert extract["derived_from_sha256"] == hashlib.sha256(LABSTAT_BODY).hexdigest()
    assert extract["last_modified"] == LAST_MODIFIED
    assert extract["fetched_at"] == "2026-11-17T12:00:00Z"
    assert rows[raw.CE_FOOTNOTES]["url"] == sources.FOOTNOTE_URL


def test_year_candidates_take_200_captures_of_either_address_latest_first():
    parse = sources.archive_inventory.parse_cdx
    captures = parse(
        listing(
            ("20071230120000", OLD_PAGE, "200"),
            ("20100301120000", OLD_PAGE, "200"),
            ("20101201120000", NEW_PAGE, "200"),
            ("20100601120000", OLD_PAGE, "301"),
            ("20100901120000", NEW_PAGE, "404"),
            ("20100301120000", OLD_PAGE, "200"),
            ("20250319165303", NEW_PAGE, "200"),
        ).decode()
    )
    candidates = sources.year_candidates(captures)
    assert set(candidates) == set(ARCHIVE_YEARS)
    assert candidates[2010] == [
        ("20101201120000", NEW_PAGE),
        ("20100301120000", OLD_PAGE),
    ]
    assert all(candidates[year] == [] for year in ARCHIVE_YEARS if year != 2010)


def test_year_status_walks_down_to_the_next_copy_and_marks_exhausted_years_missing():
    candidates = {
        2010: [("20101201120000", OLD_PAGE), ("20100301120000", OLD_PAGE)],
        2011: [("20111201120000", OLD_PAGE), ("20110301120000", OLD_PAGE)],
        2012: [("20121201120000", OLD_PAGE), ("20120301120000", OLD_PAGE)],
        2013: [("20131201120000", OLD_PAGE)],
        2014: [],
    }
    staged = {
        "20101201120000": True,
        "20111201120000": False,
        "20121201120000": False,
        "20120301120000": False,
        "20131201120000": False,
    }
    assert sources.year_status(candidates, staged) == {
        2010: "20101201120000",
        2011: "pending",
        2012: "missing",
        2013: "missing",
        2014: "missing",
    }
    walked = {**staged, "20110301120000": True}
    assert sources.year_status(candidates, walked)[2011] == "20110301120000"


def test_fetch_archive_fetches_only_the_latest_copy_of_a_year_with_none_staged(
    monkeypatch, tmp_path
):
    old = [
        ("20100301120000", OLD_PAGE, "200"),
        ("20101201120000", OLD_PAGE, "200"),
        ("20100601120000", OLD_PAGE, "200"),
    ]
    bodies = listings(old, [recon_capture()])
    bodies[replay("20101201120000")] = valid_page()
    bodies[replay(RECON_COPY, NEW_PAGE)] = valid_page()
    network = FakeNetwork(monkeypatch, bodies)
    staging = tmp_path / "staging"
    assert sources.fetch_archive(NOW, staging) == 0
    assert network.requests == [
        cdx(sources.PAGE_KEYS[0]),
        cdx(sources.PAGE_KEYS[1]),
        replay("20101201120000"),
        replay(RECON_COPY, NEW_PAGE),
    ]
    pause = f"sleep {sources.archive_inventory.PAUSE_SECONDS}"
    assert network.events == [
        event for request in network.requests for event in (pause, request)
    ]
    assert (staging / "cdx" / "0.txt").read_bytes() == bodies[cdx(sources.PAGE_KEYS[0])]
    copies = staging / "raw" / raw.REGISTRY_COPIES
    assert sorted(path.name for path in copies.iterdir()) == [
        "20101201120000.htm",
        f"{RECON_COPY}.htm",
    ]
    rows = {
        row["file"]: row
        for row in sources.archive_inventory.read_csv(staging / sources.PROVENANCE)
    }
    assert rows[f"{raw.REGISTRY_COPIES}/20101201120000.htm"]["url"] == replay(
        "20101201120000"
    )


def test_fetch_archive_walks_down_a_year_whose_staged_copies_are_invalid(
    monkeypatch, tmp_path
):
    old = [
        ("20101201120000", OLD_PAGE, "200"),
        ("20111201120000", OLD_PAGE, "200"),
        ("20110901120000", OLD_PAGE, "200"),
        ("20110601120000", OLD_PAGE, "200"),
        ("20110301120000", OLD_PAGE, "200"),
    ]
    staging = tmp_path / "staging"
    stage_copy(staging, "20101201120000", valid_page())
    stage_copy(staging, "20111201120000", INVALID_PAGE)
    stage_copy(staging, RECON_COPY, valid_page())
    bodies = listings(old, [recon_capture()])
    bodies[replay("20110901120000")] = INVALID_PAGE
    bodies[replay("20110601120000")] = valid_page()
    network = FakeNetwork(monkeypatch, bodies)
    assert sources.fetch_archive(NOW, staging) == 0
    assert network.requests == [
        cdx(sources.PAGE_KEYS[0]),
        cdx(sources.PAGE_KEYS[1]),
        replay("20110901120000"),
        replay("20110601120000"),
    ]
    assert (
        sources.year_status(
            sources.year_candidates(sources.read_listing(staging)),
            sources.staged_copies(staging),
        )[2011]
        == "20110601120000"
    )


def test_fetch_archive_refuses_a_listing_without_the_recon_copy(monkeypatch, tmp_path):
    bodies = listings([("20101201120000", OLD_PAGE, "200")], [])
    network = FakeNetwork(monkeypatch, bodies)
    assert sources.fetch_archive(NOW, tmp_path / "staging") == 1
    assert network.requests == [cdx(sources.PAGE_KEYS[0]), cdx(sources.PAGE_KEYS[1])]


def test_prose_drops_tables_and_tags():
    prose = sources.page_prose(valid_page().decode())
    assert "first preliminary, second preliminary and third collection rates" in prose
    assert "March 7, 2025" in prose
    assert "55.1" not in prose
    assert "Registry Receipts" not in prose
    assert "<" not in prose


def test_promote_adds_rows_and_keeps_every_existing_row(tmp_path):
    staging = tmp_path / "staging"
    stamps = staged_tree(staging)
    raw_dir = raw_tree(tmp_path)
    existing = sources.archive_inventory.read_csv(raw_dir / raw.MANIFEST)
    assert sources.promote(staging, raw_dir) == 0
    rows = sources.archive_inventory.read_csv(raw_dir / raw.MANIFEST)
    files = [row["file"] for row in rows]
    assert files == sorted(files)
    assert existing[0] in rows
    expected = {raw.COLLECTION_RATES, raw.CE_FOOTNOTES} | {
        f"{raw.REGISTRY_COPIES}/{stamp}.htm" for stamp in stamps
    }
    assert set(files) == expected | {raw.RESCHEDULES}
    for file in expected:
        assert (raw_dir / file).read_bytes() == (staging / "raw" / file).read_bytes()
    by_file = {row["file"]: row for row in rows}
    assert by_file[raw.CE_FOOTNOTES]["url"] == sources.FOOTNOTE_URL
    assert by_file[raw.COLLECTION_RATES]["derived_from"].startswith(sources.LABSTAT_URL)
    assert vintage_sources.refresh_manifest(raw_dir) == len(rows)


def test_promote_refuses_while_a_year_is_pending(tmp_path):
    staging = tmp_path / "staging"
    staged_tree(staging)
    stage_copy(staging, "20151215120000", INVALID_PAGE)
    old = [
        (f"{year}1215120000", OLD_PAGE, "200") for year in ARCHIVE_YEARS if year != 2015
    ] + [("20151215120000", OLD_PAGE, "200"), ("20150601120000", OLD_PAGE, "200")]
    (staging / "cdx" / "0.txt").write_bytes(listing(*old))
    raw_dir = raw_tree(tmp_path)
    before = snapshot(raw_dir)
    assert sources.promote(staging, raw_dir) == 1
    assert snapshot(raw_dir) == before


def test_promote_refuses_to_overwrite_an_existing_source(tmp_path):
    staging = tmp_path / "staging"
    staged_tree(staging)
    raw_dir = raw_tree(tmp_path)
    (raw_dir / raw.COLLECTION_RATES).parent.mkdir(parents=True, exist_ok=True)
    (raw_dir / raw.COLLECTION_RATES).write_bytes(b"committed extract")
    before = snapshot(raw_dir)
    assert sources.promote(staging, raw_dir) == 1
    assert snapshot(raw_dir) == before


def test_download_without_a_content_length_is_refused_before_reading(
    monkeypatch, tmp_path
):
    unsized = FakeResponse(LABSTAT_BODY, {"Last-Modified": LAST_MODIFIED})
    FakeNetwork(monkeypatch, {sources.LABSTAT_URL: unsized})
    with pytest.raises(ValueError, match="Content-Length"):
        sources.stream_download(sources.LABSTAT_URL, tmp_path / "full")
    assert not (tmp_path / "full").exists()
    assert unsized.stream.tell() == 0


def test_fetch_labstat_refuses_a_file_without_every_series(monkeypatch, tmp_path):
    monkeypatch.setenv("BLS_CONTACT_EMAIL", "someone@example.org")
    without_rr = b"".join(
        line
        for line in LABSTAT_BODY.splitlines(keepends=True)
        if not line.startswith(b"CEU05000000RR")
    )
    FakeNetwork(
        monkeypatch,
        {sources.LABSTAT_URL: without_rr, sources.FOOTNOTE_URL: FOOTNOTE_BODY},
    )
    staging = tmp_path / "staging"
    with pytest.raises(ValueError, match="CEU05000000RR"):
        sources.fetch_labstat(NOW, staging)
    assert not (staging / "raw").exists()


def test_fetch_labstat_refuses_footnotes_without_their_header(monkeypatch, tmp_path):
    monkeypatch.setenv("BLS_CONTACT_EMAIL", "someone@example.org")
    FakeNetwork(
        monkeypatch,
        {
            sources.LABSTAT_URL: LABSTAT_BODY,
            sources.FOOTNOTE_URL: b"<html>error</html>",
        },
    )
    staging = tmp_path / "staging"
    with pytest.raises(ValueError, match="footnote"):
        sources.fetch_labstat(NOW, staging)
    assert not (staging / "raw" / raw.CE_FOOTNOTES).exists()


def test_fetch_archive_leaves_a_copy_that_fails_every_attempt_unstaged(
    monkeypatch, tmp_path, capsys
):
    old = [("20101201120000", OLD_PAGE, "200"), ("20100601120000", OLD_PAGE, "200")]
    bodies = listings(old, [recon_capture()])
    bodies[replay(RECON_COPY, NEW_PAGE)] = valid_page()
    network = FakeNetwork(monkeypatch, bodies)
    original_open = network.open_url

    def failing_open(url, timeout, contact=None):
        if url == replay("20101201120000"):
            network.events.append(url)
            raise OSError("replay unavailable")
        return original_open(url, timeout, contact)

    monkeypatch.setattr(sources.archive_inventory, "open_url", failing_open)
    staging = tmp_path / "staging"
    assert sources.fetch_archive(NOW, staging) == 0
    assert network.requests.count(replay("20101201120000")) == (
        sources.archive_inventory.ATTEMPTS
    )
    assert replay("20100601120000") not in network.requests
    assert sources.staged_copies(staging) == {RECON_COPY: True}
    assert "not staged" in capsys.readouterr().err


def test_prose_drops_scripts_and_styles():
    page = (
        "<html><head><style>p { color: red; }</style>"
        "<script>var first = 'preliminary';</script></head>"
        "<body><p>Collection rates by release.</p></body></html>"
    )
    assert sources.page_prose(page) == "Collection rates by release."


def test_status_exits_0_only_when_every_source_is_staged_and_no_year_pending(
    tmp_path, capsys
):
    assert sources.show_status(tmp_path / "empty") == 1
    staging = tmp_path / "staging"
    staged_tree(staging)
    stage_copy(staging, "20151201120000", INVALID_PAGE)
    assert sources.show_status(staging) == 0
    out = capsys.readouterr().out
    assert "2015: 20151215120000" in out
    assert "does not parse: 20151201120000" in out
    (staging / "raw" / raw.CE_FOOTNOTES).unlink()
    assert sources.show_status(staging) == 1
    assert f"{raw.CE_FOOTNOTES}: not staged" in capsys.readouterr().out


def test_status_refuses_an_invalid_recon_copy(tmp_path):
    staging = tmp_path / "staging"
    staged_tree(staging)
    stage_copy(staging, RECON_COPY, INVALID_PAGE)
    assert sources.show_status(staging) == 1


def test_prose_command_prints_each_chosen_copy(tmp_path, capsys):
    staging = tmp_path / "staging"
    stamps = staged_tree(staging, skip_year=2012)
    stage_copy(staging, "20151201120000", INVALID_PAGE)
    assert sources.show_prose(staging) == 0
    headers = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("==")
    ]
    assert headers == [f"== {stamp} ==" for stamp in stamps]


def test_promote_refuses_a_file_the_manifest_already_lists(tmp_path):
    staging = tmp_path / "staging"
    staged_tree(staging)
    raw_dir = raw_tree(tmp_path)
    rows = sources.archive_inventory.read_csv(raw_dir / raw.MANIFEST)
    listed = dict(rows[0], file=raw.CE_FOOTNOTES)
    sources.archive_inventory.write_csv(
        raw_dir / raw.MANIFEST, [*rows, listed], vintage_sources.MANIFEST_COLUMNS
    )
    before = snapshot(raw_dir)
    assert sources.promote(staging, raw_dir) == 1
    assert snapshot(raw_dir) == before


def test_promote_takes_no_copy_for_a_missing_year(tmp_path):
    staging = tmp_path / "staging"
    stamps = staged_tree(staging, skip_year=2012)
    raw_dir = raw_tree(tmp_path)
    assert sources.promote(staging, raw_dir) == 0
    copies = sorted(path.stem for path in (raw_dir / raw.REGISTRY_COPIES).iterdir())
    assert copies == sorted(stamps)
