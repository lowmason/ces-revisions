"""scripts/vintage_sources.py and the manifest of the committed sources in data/raw/."""

import csv
import subprocess
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pytest
import vintage_sources

from ces_revisions.vintages import raw


def manifest() -> list[dict[str, str]]:
    with (raw.RAW_DIR / raw.MANIFEST).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_bls_requests_name_a_contact_and_other_hosts_do_not(monkeypatch):
    monkeypatch.setenv("BLS_CONTACT_EMAIL", "someone@example.org")
    assert (
        vintage_sources.user_agent_for("https://www.bls.gov/web/empsit/cesvinall.zip")
        == "ces-revisions/0.1.0 (someone@example.org)"
    )
    assert (
        vintage_sources.user_agent_for("https://www.philadelphiafed.org/data.xlsx")
        == "ces-revisions/0.1.0"
    )
    assert (
        vintage_sources.user_agent_for("https://bls.gov.example.com/")
        == "ces-revisions/0.1.0"
    )


def test_comment_rows_keep_the_entries_below_the_header():
    rows = [
        ("NOTE ON DATA USAGE: the purpose of providing first and third", None),
        (None, None),
        ("Publication Date", "Adjustment"),
        ("June 2003", "With the release of May 2003 data on June 6, 2003, the CES"),
        (None, "Due to the 2025 lapse in appropriations, no estimates for October"),
        (None, None),
    ]
    assert vintage_sources.comment_rows(rows) == [
        {
            "sheet_row": "4",
            "publication_label": "June 2003",
            "adjustment": "With the release of May 2003 data on June 6, 2003, the CES",
        },
        {
            "sheet_row": "5",
            "publication_label": "",
            "adjustment": "Due to the 2025 lapse in appropriations, no estimates for October",
        },
    ]


def test_comment_rows_without_the_header_are_an_error():
    rows = [("NOTE ON DATA USAGE", None), ("June 2003", "With the release of May 2003")]
    with pytest.raises(ValueError, match="Publication Date"):
        vintage_sources.comment_rows(rows)


@pytest.mark.parametrize(
    ("url", "payload", "message"),
    [
        ("https://www.bls.gov/source.zip", b"<html>error</html>", "ZIP signature"),
        ("https://www.bls.gov/source.pdf", b"<html>error</html>", "PDF signature"),
        ("https://www.bls.gov/source.xls", b"<html>error</html>", "XLS signature"),
        ("https://www.bls.gov/source.xlsx", b"<html>error</html>", "XLSX signature"),
        (
            "https://www.bls.gov/source.htm",
            b"temporarily unavailable",
            "HTML signature",
        ),
        (
            "https://www.bls.gov/source.html",
            b"temporarily unavailable",
            "HTML signature",
        ),
    ],
)
def test_payload_validation_rejects_error_pages_and_wrong_file_types(
    url, payload, message
):
    with pytest.raises(ValueError, match=message):
        vintage_sources.validate_payload(url, payload)


@pytest.mark.parametrize(
    ("url", "payload"),
    [
        ("https://www.bls.gov/source.zip", b"PK\x03\x04archive"),
        ("https://www.bls.gov/source.pdf", b"%PDF-1.7\nbody"),
        (
            "https://www.bls.gov/source.xls",
            b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1workbook",
        ),
        ("https://www.bls.gov/source.xlsx", b"PK\x03\x04workbook"),
        ("https://www.bls.gov/source.htm", b"<!doctype html><html></html>"),
    ],
)
def test_payload_validation_accepts_expected_file_signatures(url, payload):
    vintage_sources.validate_payload(url, payload)


def test_payload_validation_rejects_an_empty_response():
    with pytest.raises(ValueError, match="empty payload"):
        vintage_sources.validate_payload("https://www.bls.gov/source.zip", b"")


def test_pdftotext_version_is_the_first_nonempty_output_line(monkeypatch):
    completed = subprocess.CompletedProcess(
        ["pdftotext", "-v"],
        0,
        stdout="",
        stderr="pdftotext version 26.06.0\nCopyright line\n",
    )
    monkeypatch.setattr(
        vintage_sources.subprocess, "run", lambda *args, **kwargs: completed
    )
    assert vintage_sources.pdftotext_version() == "pdftotext version 26.06.0"


def test_pdf_conversion_rejects_empty_derived_text(monkeypatch, tmp_path):
    source = tmp_path / "source.pdf"
    target = tmp_path / "source.txt"
    source.write_bytes(b"%PDF-1.7")

    def empty_conversion(command, check):
        Path(command[-1]).write_text("\n", encoding="utf-8")

    monkeypatch.setattr(vintage_sources.subprocess, "run", empty_conversion)
    with pytest.raises(ValueError, match="produced no text"):
        vintage_sources.convert_pdf(source, target)


def test_fetch_stops_before_downloading_without_a_contact_address(monkeypatch, capsys):
    monkeypatch.delenv("BLS_CONTACT_EMAIL", raising=False)

    def refuse(url: str) -> tuple[bytes, str]:
        raise AssertionError(f"fetched {url} without a contact address")

    monkeypatch.setattr(vintage_sources, "download", refuse)
    assert vintage_sources.fetch_sources(datetime(2026, 9, 15, tzinfo=UTC)) == 1
    assert "BLS_CONTACT_EMAIL" in capsys.readouterr().err


def test_fetch_validates_every_payload_before_replacing_sources(
    monkeypatch, tmp_path: Path
):
    raw_dir = tmp_path / "raw"
    existing_zip = raw_dir / raw.VINTAGE_FILES
    existing_zip.parent.mkdir(parents=True)
    existing_zip.write_bytes(b"old archive")
    existing_manifest = raw_dir / raw.MANIFEST
    existing_manifest.write_text("old manifest\n", encoding="utf-8")
    workbook_path = tmp_path / "cache" / "cesvin00.xlsx"
    responses = iter(
        [
            (b"PK\x03\x04new archive", "Fri, 06 Mar 2026 12:06:34 GMT"),
            (b"temporarily unavailable", ""),
        ]
    )

    monkeypatch.setenv("BLS_CONTACT_EMAIL", "someone@example.org")
    monkeypatch.setattr(vintage_sources.shutil, "which", lambda command: "/pdftotext")
    monkeypatch.setattr(
        vintage_sources, "pdftotext_version", lambda: "pdftotext version 26.06.0"
    )
    monkeypatch.setattr(vintage_sources, "download", lambda url: next(responses))

    with pytest.raises(ValueError, match="HTML signature"):
        vintage_sources.fetch_sources(
            datetime(2026, 9, 15, tzinfo=UTC),
            raw_dir=raw_dir,
            workbook_path=workbook_path,
        )

    assert existing_zip.read_bytes() == b"old archive"
    assert existing_manifest.read_text(encoding="utf-8") == "old manifest\n"
    assert not workbook_path.exists()


def test_refresh_manifest_rehashes_bytes_without_changing_provenance(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    source = raw_dir / raw.RESCHEDULES
    source.parent.mkdir(parents=True)
    source.write_text("old\n", encoding="utf-8")
    row = dict.fromkeys(vintage_sources.MANIFEST_COLUMNS, "")
    row.update(
        file=raw.RESCHEDULES,
        sha256="old hash",
        bytes="4",
        derived_from="hand-keyed from the citation on each row",
    )
    vintage_sources.archive_inventory.write_csv(
        raw_dir / raw.MANIFEST, [row], vintage_sources.MANIFEST_COLUMNS
    )

    source.write_text("new source bytes\n", encoding="utf-8")
    assert vintage_sources.refresh_manifest(raw_dir) == 1

    [refreshed] = vintage_sources.archive_inventory.read_csv(raw_dir / raw.MANIFEST)
    assert refreshed["sha256"] == raw.file_sha256(source)
    assert refreshed["bytes"] == str(source.stat().st_size)
    assert refreshed["derived_from"] == "hand-keyed from the citation on each row"
    assert refreshed["url"] == ""
    assert refreshed["fetched_at"] == ""


def test_refresh_manifest_rejects_an_unrecorded_source(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    vintage_sources.archive_inventory.write_csv(
        raw_dir / raw.MANIFEST, [], vintage_sources.MANIFEST_COLUMNS
    )
    (raw_dir / "unexpected.csv").write_text("value\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file set differs"):
        vintage_sources.refresh_manifest(raw_dir)


def test_manifest_command_reports_the_refreshed_file_count(monkeypatch, capsys):
    monkeypatch.setattr(vintage_sources, "refresh_manifest", lambda: 9)
    assert vintage_sources.main(["manifest"]) == 0
    assert capsys.readouterr().out == "manifest: 9 source files\n"


def test_historical_release_text_records_its_pdftotext_version():
    rows = {row["file"]: row for row in manifest()}
    assert rows[raw.HISTORICAL_RELEASE_DATES]["tool_version"].startswith(
        "pdftotext version "
    )
    assert all(
        not row["tool_version"]
        for file, row in rows.items()
        if file != raw.HISTORICAL_RELEASE_DATES
    )


def test_the_manifest_lists_every_committed_source_with_its_hash_and_size():
    rows = manifest()
    committed = sorted(
        str(path.relative_to(raw.RAW_DIR))
        for path in raw.RAW_DIR.rglob("*")
        if path.is_file() and path.name not in (raw.MANIFEST, ".DS_Store")
    )
    assert [row["file"] for row in rows] == committed
    for row in rows:
        path = raw.RAW_DIR / row["file"]
        assert row["sha256"] == raw.file_sha256(path), row["file"]
        assert row["bytes"] == str(path.stat().st_size), row["file"]


def test_downloads_record_their_origin_and_bls_file_dates():
    rows = {row["file"]: row for row in manifest()}
    for item in vintage_sources.DOWNLOADS:
        assert rows[item.file]["url"] == item.url
        assert rows[item.file]["fetched_at"], item.file
    # BLS dates its static files; its HTML pages and the Philadelphia Fed's files carry no
    # Last-Modified header, so their manifest rows leave it empty.
    for file in ("bls/cesvinall.zip", "bls/histreleasedates.pdf"):
        assert rows[file]["last_modified"].endswith(" GMT"), file
    comments = rows[raw.VINTAGE_COMMENTS]
    assert comments["derived_from"].startswith(vintage_sources.WORKBOOK_URL)
    assert len(comments["derived_from_sha256"]) == 64
    assert rows[raw.RESCHEDULES]["derived_from"].startswith("hand-keyed")


@pytest.mark.network
def test_bls_has_not_refreshed_the_vintage_files_since_the_manifest():
    """BLS refreshes cesvinall.zip about once a year; when this fails, fetch and rebuild."""
    rows = {row["file"]: row for row in manifest()}
    url = "https://www.bls.gov/web/empsit/cesvinall.zip"
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": vintage_sources.user_agent_for(url)}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        last_modified = response.headers["Last-Modified"]
    assert last_modified == rows["bls/cesvinall.zip"]["last_modified"]
