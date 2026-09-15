"""The Stage 4 fetch/build script and committed source manifest."""

import csv
from datetime import UTC, datetime
from pathlib import Path

import annual_sources
import polars as pl
import pytest

from ces_revisions.annual import raw


def manifest_rows() -> list[dict[str, str]]:
    with (raw.RAW_DIR / raw.MANIFEST).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_dotenv_supplies_the_contact_without_exposing_it(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("BLS_CONTACT_EMAIL", raising=False)
    env_file = tmp_path / ".project.env"
    env_file.write_text("BLS_CONTACT_EMAIL=someone@example.org\n", encoding="utf-8")
    contact = annual_sources.load_contact_email(env_file)
    assert contact == "someone@example.org"
    assert annual_sources.user_agent(contact) == (
        "ces-revisions/0.1.0 (someone@example.org)"
    )


def test_exported_contact_takes_precedence_over_dotenv(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("BLS_CONTACT_EMAIL", "exported@example.org")
    env_file = tmp_path / ".project.env"
    env_file.write_text("BLS_CONTACT_EMAIL=file@example.org\n", encoding="utf-8")
    assert annual_sources.load_contact_email(env_file) == "exported@example.org"


def test_contact_setting_must_look_like_an_email(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("BLS_CONTACT_EMAIL", "anonymous-client")
    assert annual_sources.load_contact_email(tmp_path / "missing.env") is None


def test_fetch_stops_without_the_bls_contact_setting(
    monkeypatch, capsys, tmp_path: Path
):
    monkeypatch.delenv("BLS_CONTACT_EMAIL", raising=False)
    monkeypatch.setattr(
        annual_sources,
        "download",
        lambda url, contact: (_ for _ in ()).throw(
            AssertionError(f"downloaded {url} as {contact}")
        ),
    )
    result = annual_sources.fetch_sources(
        datetime(2026, 9, 15, tzinfo=UTC),
        env_file=tmp_path / "missing.env",
    )
    captured = capsys.readouterr()
    assert result == 1
    assert "BLS_CONTACT_EMAIL" in captured.err
    assert "@" not in captured.err


def test_payload_validation_rejects_an_html_error_saved_as_a_pdf():
    with pytest.raises(ValueError, match="PDF signature"):
        annual_sources.validate_payload(
            "https://www.bls.gov/ces/publications/benchmark/example.pdf",
            b"<html>temporarily unavailable</html>",
        )


def test_retrying_download_backs_off_after_a_transient_network_error(monkeypatch):
    attempts = iter([OSError("connection refused"), (b"payload", "last modified")])
    pauses = []

    def flaky_download(url, contact):
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(annual_sources, "download", flaky_download)
    monkeypatch.setattr(annual_sources, "sleep", pauses.append)

    assert annual_sources.retrying_download(
        "https://web.archive.org/example", "someone@example.org"
    ) == (b"payload", "last modified")
    assert pauses == [annual_sources.PAUSE_SECONDS]


def test_archive_downloads_are_throttled_before_the_request(monkeypatch):
    pauses = []
    monkeypatch.setattr(
        annual_sources, "retrying_download", lambda url, contact: (b"payload", "")
    )
    monkeypatch.setattr(annual_sources, "sleep", pauses.append)

    assert annual_sources.fetch_catalog_source(
        "https://web.archive.org/example", "someone@example.org"
    ) == (b"payload", "")
    assert pauses == [annual_sources.PAUSE_SECONDS]


def test_fetch_validates_every_payload_before_replacing_sources(
    monkeypatch, tmp_path: Path
):
    raw_dir = tmp_path / "raw"
    existing = raw_dir / "bls" / "one.htm"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"old source")
    catalog = pl.DataFrame(
        {
            "status": ["available", "available"],
            "url": ["https://www.bls.gov/one.htm", "https://www.bls.gov/two.pdf"],
            "file": ["bls/one.htm", "bls/two.pdf"],
        }
    )
    monkeypatch.setattr(raw, "RAW_DIR", raw_dir)
    monkeypatch.setattr(raw, "read_source_catalog", lambda: catalog)
    monkeypatch.setattr(annual_sources.shutil, "which", lambda command: "/pdftotext")
    monkeypatch.setattr(
        annual_sources,
        "download",
        lambda url, contact: (
            (b"<html>new source</html>" if url.endswith(".htm") else b"error"),
            "",
        ),
    )
    monkeypatch.setenv("BLS_CONTACT_EMAIL", "someone@example.org")
    with pytest.raises(ValueError, match="PDF signature"):
        annual_sources.fetch_sources(
            datetime(2026, 9, 15, tzinfo=UTC),
            env_file=tmp_path / "missing.env",
        )
    assert existing.read_bytes() == b"old source"


def test_source_catalog_has_one_local_path_per_available_source():
    catalog = raw.read_source_catalog()
    available = catalog.filter(pl.col("status") == "available")
    assert available["source_id"].n_unique() == available.height
    assert available["file"].str.len_chars().gt(0).all()
    assert available["url"].str.starts_with("https://").all()


def test_2019_preliminary_evidence_uses_the_dated_bls_blog():
    row = raw.read_source_catalog().filter(pl.col("source_id") == "preliminary-2019")
    assert row.item(0, "url") == (
        "https://www.bls.gov/blog/2019/what-is-benchmarking-of-bureau-of-"
        "labor-statistics-employment-data.htm"
    )


def test_manifest_matches_every_committed_annual_source():
    rows = manifest_rows()
    files = sorted(
        str(path.relative_to(raw.RAW_DIR))
        for path in raw.RAW_DIR.rglob("*")
        if path.is_file() and path.name not in {raw.MANIFEST, ".DS_Store"}
    )
    assert [row["file"] for row in rows] == files
    for row in rows:
        path = raw.RAW_DIR / row["file"]
        assert row["sha256"] == raw.file_sha256(path)
        assert row["bytes"] == str(path.stat().st_size)


@pytest.mark.network
def test_live_qcew_header_has_not_changed():
    contact = annual_sources.load_contact_email()
    assert contact is not None
    payload, _ = annual_sources.download(annual_sources.QCEW_REVISIONS_URL, contact)
    assert payload.splitlines()[0].decode() == (
        "Year,Quarter,Area,Field,Initial Value,First Revised Value,"
        "Second Revised Value,Third Revised Value,Fourth Revised Value,Final Value"
    )
