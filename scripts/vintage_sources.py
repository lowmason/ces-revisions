"""Fetch roadmap Stage 3's source files into data/raw/, or build the vintage panel from them.

    uv run python scripts/vintage_sources.py fetch     # network: refresh data/raw/ and its manifest
    uv run python scripts/vintage_sources.py manifest  # offline: rehash existing data/raw/
    uv run python scripts/vintage_sources.py build     # offline: write data/panel/ from data/raw/

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
import tempfile
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
    "tool_version",
]
ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
XLS_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def validate_payload(url: str, payload: bytes) -> None:
    """Reject empty responses and payloads that do not match their file type."""
    if not payload:
        raise ValueError(f"{url} returned an empty payload")
    suffix = Path(urlsplit(url).path).suffix.lower()
    prefix = payload.lstrip()[:4096].lower()
    checks = {
        ".zip": (payload.startswith(ZIP_SIGNATURES), "ZIP"),
        ".pdf": (payload.startswith(b"%PDF-"), "PDF"),
        ".xls": (payload.startswith(XLS_SIGNATURE), "XLS"),
        ".xlsx": (payload.startswith(ZIP_SIGNATURES), "XLSX"),
        ".htm": (b"<html" in prefix, "HTML"),
        ".html": (b"<html" in prefix, "HTML"),
    }
    if suffix in checks:
        valid, label = checks[suffix]
        if not valid:
            raise ValueError(f"{url} does not have a {label} signature")


def pdftotext_version() -> str:
    """Return the Poppler version line recorded beside layout-derived text."""
    completed = subprocess.run(
        ["pdftotext", "-v"], check=True, capture_output=True, text=True
    )
    lines = [
        line.strip() for line in (completed.stderr or completed.stdout).splitlines()
    ]
    try:
        return next(line for line in lines if line)
    except StopIteration as error:
        raise ValueError("pdftotext -v returned no version text") from error


def convert_pdf(source: Path, target: Path) -> None:
    """Create layout-preserving text and reject an empty conversion."""
    subprocess.run(["pdftotext", "-layout", str(source), str(target)], check=True)
    if not target.read_text(encoding="utf-8").strip():
        raise ValueError(f"pdftotext produced no text for {source}")


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


def manifest_row(
    file: str, fetched_at: str, *, raw_dir: Path = RAW_DIR, **fields: str
) -> dict[str, str]:
    path = raw_dir / file
    row = dict.fromkeys(MANIFEST_COLUMNS, "")
    row.update(
        file=file,
        sha256=file_sha256(path),
        bytes=str(path.stat().st_size),
        fetched_at=fetched_at,
        **fields,
    )
    return row


def promote_tree(staging_dir: Path, target_dir: Path) -> None:
    """Replace target files only after the caller has completed staging."""
    for staged in sorted(path for path in staging_dir.rglob("*") if path.is_file()):
        target = target_dir / staged.relative_to(staging_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        staged.replace(target)


def refresh_manifest(raw_dir: Path = RAW_DIR) -> int:
    """Rehash the recorded archive without changing acquisition provenance."""
    existing_rows = archive_inventory.read_csv(raw_dir / MANIFEST)
    existing = {row["file"]: row for row in existing_rows}
    files = sorted(
        str(path.relative_to(raw_dir))
        for path in raw_dir.rglob("*")
        if path.is_file() and path != raw_dir / MANIFEST and path.name != ".DS_Store"
    )
    if set(existing) != set(files):
        missing = sorted(set(existing) - set(files))
        unrecorded = sorted(set(files) - set(existing))
        raise ValueError(
            f"manifest file set differs: missing={missing}, unrecorded={unrecorded}"
        )

    rows = []
    for file in files:
        row = dict.fromkeys(MANIFEST_COLUMNS, "")
        row.update(existing[file])
        path = raw_dir / file
        row.update(
            file=file,
            sha256=file_sha256(path),
            bytes=str(path.stat().st_size),
        )
        rows.append(row)
    archive_inventory.write_csv(raw_dir / MANIFEST, rows, MANIFEST_COLUMNS)
    return len(rows)


def fetch_sources(
    now: datetime,
    *,
    raw_dir: Path = RAW_DIR,
    workbook_path: Path = WORKBOOK_PATH,
) -> int:
    """Refresh all sources only after a complete staged fetch validates."""
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
    tool_version = pdftotext_version()
    raw_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="ces-stage3-fetch-", dir=raw_dir.parent
    ) as directory:
        staging_root = Path(directory)
        staged_raw = staging_root / "raw"
        staged_workbook = staging_root / "cache" / workbook_path.name
        rows = []

        for item in DOWNLOADS:
            payload, last_modified = download(item.url)
            validate_payload(item.url, payload)
            target = staged_raw / item.file
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            rows.append(
                manifest_row(
                    item.file,
                    stamp,
                    raw_dir=staged_raw,
                    url=item.url,
                    last_modified=last_modified,
                )
            )

        pdf = staged_raw / "bls/histreleasedates.pdf"
        convert_pdf(pdf, staged_raw / HISTORICAL_RELEASE_DATES)
        rows.append(
            manifest_row(
                HISTORICAL_RELEASE_DATES,
                stamp,
                raw_dir=staged_raw,
                derived_from="pdftotext -layout bls/histreleasedates.pdf",
                derived_from_sha256=file_sha256(pdf),
                tool_version=tool_version,
            )
        )

        payload, last_modified = download(WORKBOOK_URL)
        validate_payload(WORKBOOK_URL, payload)
        staged_workbook.parent.mkdir(parents=True, exist_ok=True)
        staged_workbook.write_bytes(payload)
        archive_inventory.write_csv(
            staged_raw / VINTAGE_COMMENTS,
            extract_comments(staged_workbook),
            COMMENT_COLUMNS,
        )
        rows.append(
            manifest_row(
                VINTAGE_COMMENTS,
                stamp,
                raw_dir=staged_raw,
                last_modified=last_modified,
                derived_from=f"{WORKBOOK_URL}, sheet {COMMENTS_SHEET}",
                derived_from_sha256=file_sha256(staged_workbook),
            )
        )

        index_payload, _ = download(archive_inventory.RELEASE_INDEX_URL)
        validate_payload(archive_inventory.RELEASE_INDEX_URL, index_payload)
        vintages = archive_inventory.select_vintages(
            archive_inventory.parse_release_index(
                index_payload.decode("utf-8", "replace")
            ),
            now=now,
        )
        archive_inventory.write_csv(
            staged_raw / EMPSIT_RELEASES,
            archive_inventory.vintage_rows(vintages),
            archive_inventory.VINTAGE_COLUMNS,
        )
        rows.append(
            manifest_row(
                EMPSIT_RELEASES,
                stamp,
                raw_dir=staged_raw,
                derived_from=archive_inventory.RELEASE_INDEX_URL,
            )
        )

        manual_source = raw_dir / RESCHEDULES
        if not manual_source.is_file():
            raise FileNotFoundError(f"missing hand-maintained source {manual_source}")
        staged_manual = staged_raw / RESCHEDULES
        staged_manual.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manual_source, staged_manual)
        rows.append(
            manifest_row(
                RESCHEDULES,
                "",
                raw_dir=staged_raw,
                derived_from="hand-keyed from the citation on each row",
            )
        )
        archive_inventory.write_csv(
            staged_raw / MANIFEST,
            sorted(rows, key=lambda row: row["file"]),
            MANIFEST_COLUMNS,
        )

        workbook_path.parent.mkdir(parents=True, exist_ok=True)
        staged_workbook.replace(workbook_path)
        promote_tree(staged_raw, raw_dir)
    return 0


def build_panel() -> int:
    """Offline step: write every Stage 3 artifact to data/panel/."""
    manifest = write(build(RAW_DIR), PANEL_DIR, RAW_DIR)
    for name, entry in manifest["artifacts"].items():
        print(f"{name}: {entry['rows']} rows")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["fetch", "manifest", "build"])
    command = parser.parse_args(argv).command
    if command == "fetch":
        return fetch_sources(datetime.now(UTC).replace(microsecond=0))
    if command == "manifest":
        print(f"manifest: {refresh_manifest()} source files")
        return 0
    return build_panel()


if __name__ == "__main__":
    raise SystemExit(main())
