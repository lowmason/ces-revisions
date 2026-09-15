"""scripts/vintage_sources.py and the manifest of the committed sources in data/raw/."""

import csv
import urllib.request

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
