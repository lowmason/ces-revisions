"""Fetch roadmap Stage 3's source files into data/raw/, or build the vintage panel from them.

    uv run python scripts/vintage_sources.py fetch   # network: refresh data/raw/ and its manifest
    uv run python scripts/vintage_sources.py build   # offline: write data/panel/ from data/raw/

BLS keeps one current copy of each file and overwrites it in place, so the committed files are
the raw archive Req 1 asks for. data/raw/manifest.csv records each file's origin, SHA-256, size,
and Last-Modified header. The 27 MB workbook holding the vintage files' Comments sheet stays in
the gitignored data/cache/; its entries are committed as a CSV beside the workbook's hash.
"""

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import archive_inventory
import fastexcel

from ces_revisions.vintages.build import build, write
from ces_revisions.vintages.raw import (
    DATA_DIR,
    EMPSIT_RELEASES,
    HISTORICAL_RELEASE_DATES,
    MANIFEST,
    PANEL_DIR,
    RAW_DIR,
    RESCHEDULES,
    VINTAGE_COMMENTS,
    file_sha256,
)

PHILADELPHIA_FED = (
    "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
    "real-time-data/data-files"
)


@dataclass(frozen=True)
class Download:
    file: str  # a path under data/raw/
    url: str


DOWNLOADS = (
    Download("bls/cesvinall.zip", "https://www.bls.gov/web/empsit/cesvinall.zip"),
    Download("bls/cesnaicsrev.htm", "https://www.bls.gov/web/empsit/cesnaicsrev.htm"),
    Download(
        "bls/histreleasedates.pdf", "https://www.bls.gov/bls/histreleasedates.pdf"
    ),
    Download(
        "philadelphiafed/employMvMd.xlsx", f"{PHILADELPHIA_FED}/xlsx/employMvMd.xlsx"
    ),
    Download(
        "philadelphiafed/release-dates-employment-situation.xls",
        f"{PHILADELPHIA_FED}/documentation/Release_-Dates-Employment_Situation-BLS.xls",
    ),
)
WORKBOOK_URL = "https://www.bls.gov/web/empsit/cesvin00.xlsx"
WORKBOOK_PATH = DATA_DIR / "cache" / "cesvin00.xlsx"
COMMENTS_SHEET = "Data Usage and Comments"
COMMENT_COLUMNS = ["sheet_row", "publication_label", "adjustment"]
MANIFEST_COLUMNS = [
    "file",
    "url",
    "sha256",
    "bytes",
    "last_modified",
    "fetched_at",
    "derived_from",
    "derived_from_sha256",
]


def user_agent_for(url: str) -> str:
    """BLS asks automated clients for a contact address; other hosts get none."""
    host = urlsplit(url).hostname or ""
    if host == "bls.gov" or host.endswith(".bls.gov"):
        return archive_inventory.user_agent()
    return "ces-revisions/0.1.0"


def download(url: str) -> tuple[bytes, str]:
    """The payload and its Last-Modified header."""
    request = urllib.request.Request(url, headers={"User-Agent": user_agent_for(url)})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read(), response.headers.get("Last-Modified", "")


def comment_rows(rows: list[tuple[str | None, ...]]) -> list[dict[str, str]]:
    """The entries below the Comments sheet's "Publication Date" header, with sheet rows."""
    header = next(
        (index for index, row in enumerate(rows) if row[0] == "Publication Date"), None
    )
    if header is None:
        raise ValueError(f'the {COMMENTS_SHEET} sheet has no "Publication Date" header')
    return [
        {
            "sheet_row": str(index + 1),
            "publication_label": row[0] or "",
            "adjustment": row[1],
        }
        for index, row in enumerate(rows)
        if index > header and row[1]
    ]


def extract_comments(workbook: Path) -> list[dict[str, str]]:
    sheet = fastexcel.read_excel(workbook).load_sheet(
        COMMENTS_SHEET, header_row=None, dtypes="string"
    )
    return comment_rows(sheet.to_polars().rows())


def manifest_row(file: str, fetched_at: str, **fields: str) -> dict[str, str]:
    path = RAW_DIR / file
    row = dict.fromkeys(MANIFEST_COLUMNS, "")
    row.update(
        file=file,
        sha256=file_sha256(path),
        bytes=str(path.stat().st_size),
        fetched_at=fetched_at,
        **fields,
    )
    return row


def fetch_sources(now: datetime) -> int:
    """Network step: refresh every source file, its derived text, and the manifest."""
    if not os.environ.get("BLS_CONTACT_EMAIL"):
        print(
            "set BLS_CONTACT_EMAIL to the contact address BLS asks automated clients for",
            file=sys.stderr,
        )
        return 1
    if shutil.which("pdftotext") is None:
        print("pdftotext (poppler) is required: brew install poppler", file=sys.stderr)
        return 1
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for item in DOWNLOADS:
        payload, last_modified = download(item.url)
        target = RAW_DIR / item.file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        rows.append(
            manifest_row(item.file, stamp, url=item.url, last_modified=last_modified)
        )
    pdf = RAW_DIR / "bls/histreleasedates.pdf"
    subprocess.run(
        ["pdftotext", "-layout", str(pdf), str(RAW_DIR / HISTORICAL_RELEASE_DATES)],
        check=True,
    )
    rows.append(
        manifest_row(
            HISTORICAL_RELEASE_DATES,
            stamp,
            derived_from="pdftotext -layout bls/histreleasedates.pdf",
            derived_from_sha256=file_sha256(pdf),
        )
    )
    payload, last_modified = download(WORKBOOK_URL)
    WORKBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    WORKBOOK_PATH.write_bytes(payload)
    archive_inventory.write_csv(
        RAW_DIR / VINTAGE_COMMENTS, extract_comments(WORKBOOK_PATH), COMMENT_COLUMNS
    )
    rows.append(
        manifest_row(
            VINTAGE_COMMENTS,
            stamp,
            last_modified=last_modified,
            derived_from=f"{WORKBOOK_URL}, sheet {COMMENTS_SHEET}",
            derived_from_sha256=file_sha256(WORKBOOK_PATH),
        )
    )
    index = download(archive_inventory.RELEASE_INDEX_URL)[0].decode("utf-8", "replace")
    vintages = archive_inventory.select_vintages(
        archive_inventory.parse_release_index(index), now=now
    )
    archive_inventory.write_csv(
        RAW_DIR / EMPSIT_RELEASES,
        archive_inventory.vintage_rows(vintages),
        archive_inventory.VINTAGE_COLUMNS,
    )
    rows.append(
        manifest_row(
            EMPSIT_RELEASES,
            stamp,
            derived_from=archive_inventory.RELEASE_INDEX_URL,
        )
    )
    rows.append(
        manifest_row(
            RESCHEDULES,
            "",
            derived_from="hand-keyed from the citation on each row",
        )
    )
    archive_inventory.write_csv(
        RAW_DIR / MANIFEST, sorted(rows, key=lambda row: row["file"]), MANIFEST_COLUMNS
    )
    return 0


def build_panel() -> int:
    """Offline step: write every Stage 3 artifact to data/panel/."""
    manifest = write(build(RAW_DIR), PANEL_DIR)
    for name, entry in manifest.items():
        print(f"{name}: {entry['rows']} rows")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["fetch", "build"])
    command = parser.parse_args(argv).command
    if command == "fetch":
        return fetch_sources(datetime.now(UTC).replace(microsecond=0))
    return build_panel()


if __name__ == "__main__":
    raise SystemExit(main())
