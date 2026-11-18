"""Stage the collection-rate sources of roadmap Stage 9, then promote them into data/raw/.

    uv run --env-file <main checkout>/.project.env \\
        python scripts/collection_rate_sources.py fetch-labstat  # network, owner's yes
    uv run python scripts/collection_rate_sources.py fetch-archive  # network, owner's yes
    uv run python scripts/collection_rate_sources.py status   # offline
    uv run python scripts/collection_rate_sources.py prose    # offline
    uv run python scripts/collection_rate_sources.py promote  # offline

`fetch-labstat` streams LABSTAT's full CES data file, keeps the header and the lines of C1, C2,
C3 and `CEU05000000RR`, and fetches `ce.footnote`. `fetch-archive` lists the Internet Archive's
copies of BLS's registry-receipts page and fetches, for each year of `ARCHIVE_YEARS`, the latest
copy, or walks a year whose staged copies are all invalid down to its next copy. Both write to
the gitignored staging folder, so the validity check and the reading of each copy's prose
happen offline before `promote`, the only command that adds these files to data/raw/.
"""

import argparse
import hashlib
import http.client
import os
import re
import shutil
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import sleep

import archive_inventory
import polars as pl
import vintage_sources

from ces_revisions.vintages import raw
from ces_revisions.vintages.collection_rates import (
    ARCHIVE_YEARS,
    RECON_COPY,
    labstat_values,
    registry_page_values,
)

LABSTAT_URL = "https://download.bls.gov/pub/time.series/ce/ce.data.0.AllCESSeries"
FOOTNOTE_URL = "https://download.bls.gov/pub/time.series/ce/ce.footnote"
SERIES = ("CEU00000000C1", "CEU00000000C2", "CEU00000000C3", "CEU05000000RR")
PAGE_KEYS = ("bls.gov/web/cesregrec.htm", "bls.gov/web/empsit/cesregrec.htm")
STAGING_DIR = raw.DATA_DIR / "cache" / "collection-rates"
PROVENANCE = "provenance.csv"
PROVENANCE_COLUMNS = [
    "file",
    "url",
    "last_modified",
    "fetched_at",
    "derived_from",
    "derived_from_sha256",
]
CHUNK_BYTES = 1 << 20
TIMEOUT_SECONDS = 300
PENDING, MISSING = "pending", "missing"


def stamp(now: datetime) -> str:
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def stage(staging: Path, file: str, payload: bytes, **provenance: str) -> None:
    """Write a staged file under staging/raw/ and record its provenance row."""
    target = staging / "raw" / file
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    path = staging / PROVENANCE
    rows = archive_inventory.read_csv(path) if path.is_file() else []
    row = dict.fromkeys(PROVENANCE_COLUMNS, "") | provenance | {"file": file}
    rows = [existing for existing in rows if existing["file"] != file] + [row]
    archive_inventory.write_csv(
        path, sorted(rows, key=lambda each: each["file"]), PROVENANCE_COLUMNS
    )


# --- LABSTAT -------------------------------------------------------------------------------


def extract_series(lines: Iterable[bytes], series: tuple[str, ...]) -> bytes:
    """The first line, and every later line whose first field is in series, byte for byte."""
    lines = iter(lines)
    kept = [next(lines, b"")]
    kept += [
        line
        for line in lines
        if line.split(b"\t", 1)[0].strip().decode("ascii", "replace") in series
    ]
    return b"".join(kept)


def stream_download(url: str, target: Path) -> tuple[str, str]:
    """Stream url to target: the body's SHA-256 and its Last-Modified header.

    A body whose length differs from Content-Length, or that http.client ends with
    IncompleteRead, raises ValueError, which `archive_inventory.retrying` retries, and leaves
    no file at target.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    written = 0
    complete = False
    try:
        with (
            archive_inventory.open_url(url, timeout=TIMEOUT_SECONDS) as response,
            target.open("wb") as handle,
        ):
            expected = response.headers.get("Content-Length")
            if expected is None:
                raise ValueError(f"{url} sent no Content-Length to check the bytes by")
            while chunk := response.read(CHUNK_BYTES):
                digest.update(chunk)
                handle.write(chunk)
                written += len(chunk)
            last_modified = response.headers.get("Last-Modified", "")
        if written != int(expected):
            raise ValueError(f"{url} sent {written} bytes of {expected}")
        complete = True
    except http.client.IncompleteRead as error:
        raise ValueError(f"{url} ended after {written} bytes") from error
    finally:
        if not complete:
            target.unlink(missing_ok=True)
    return digest.hexdigest(), last_modified


def download(url: str) -> tuple[bytes, str]:
    """A whole payload and its Last-Modified header."""
    with archive_inventory.open_url(url, timeout=TIMEOUT_SECONDS) as response:
        return response.read(), response.headers.get("Last-Modified", "")


def check_extract(extract: bytes) -> None:
    lines = extract.splitlines()
    if not lines or not lines[0].startswith(b"series_id"):
        raise ValueError("the LABSTAT extract does not start with its header line")
    present = {line.split(b"\t", 1)[0].strip().decode() for line in lines[1:]}
    missing = [series for series in SERIES if series not in present]
    if missing:
        raise ValueError(f"the LABSTAT file has no lines for {', '.join(missing)}")
    values = labstat_values(extract.decode("utf-8"))
    if values.filter(pl.col("series_id") == SERIES[0]).is_empty():
        raise ValueError(f"the LABSTAT extract has no monthly {SERIES[0]} value")


def fetch_labstat(now: datetime, staging: Path = STAGING_DIR) -> int:
    """Stage the C1-C3 extract of LABSTAT's full CES data file, and ce.footnote."""
    if not os.environ.get("BLS_CONTACT_EMAIL"):
        print(
            "set BLS_CONTACT_EMAIL to the contact address BLS asks automated clients for",
            file=sys.stderr,
        )
        return 1
    full = staging / "download" / LABSTAT_URL.rsplit("/", 1)[1]
    full_sha256, last_modified = archive_inventory.retrying(
        partial(stream_download, LABSTAT_URL, full), what=LABSTAT_URL
    )
    with full.open("rb") as handle:
        extract = extract_series(handle, SERIES)
    check_extract(extract)
    stage(
        staging,
        raw.COLLECTION_RATES,
        extract,
        last_modified=last_modified,
        fetched_at=stamp(now),
        derived_from=f"{LABSTAT_URL}, lines of {' '.join(SERIES)}",
        derived_from_sha256=full_sha256,
    )
    full.unlink()

    footnotes, footnotes_modified = archive_inventory.retrying(
        partial(download, FOOTNOTE_URL), what=FOOTNOTE_URL
    )
    if not footnotes.startswith(b"footnote_code"):
        raise ValueError(f"{FOOTNOTE_URL} does not start with its header line")
    stage(
        staging,
        raw.CE_FOOTNOTES,
        footnotes,
        url=FOOTNOTE_URL,
        last_modified=footnotes_modified,
        fetched_at=stamp(now),
    )
    print(f"staged {raw.COLLECTION_RATES} ({len(extract.splitlines())} lines)")
    print(f"staged {raw.CE_FOOTNOTES}")
    return 0


# --- Internet Archive ----------------------------------------------------------------------

type Listing = list[tuple[datetime, str, str, str]]


def year_candidates(listing: Listing) -> dict[int, list[tuple[str, str]]]:
    """Each year's `200` captures of either address, as (timestamp, original), latest first."""
    candidates: dict[int, list[tuple[str, str]]] = {year: [] for year in ARCHIVE_YEARS}
    seen = set()
    for moment, original, status, _ in sorted(listing, reverse=True):
        timestamp = f"{moment:%Y%m%d%H%M%S}"
        if status != "200" or moment.year not in candidates or timestamp in seen:
            continue
        seen.add(timestamp)
        candidates[moment.year].append((timestamp, original))
    return candidates


def year_status(
    candidates: dict[int, list[tuple[str, str]]], staged: dict[str, bool]
) -> dict[int, str]:
    """Each year's chosen copy: its latest valid staged copy, else missing or pending.

    A year is missing when every candidate is staged and none is valid, and pending while a
    candidate remains unstaged.
    """
    status = {}
    for year, captures in candidates.items():
        valid = [timestamp for timestamp, _ in captures if staged.get(timestamp)]
        if valid:
            status[year] = valid[0]
        elif all(timestamp in staged for timestamp, _ in captures):
            status[year] = MISSING
        else:
            status[year] = PENDING
    return status


def copy_file(timestamp: str) -> str:
    return f"{raw.REGISTRY_COPIES}/{timestamp}.htm"


def staged_copies(staging: Path) -> dict[str, bool]:
    """Each staged copy's timestamp, and whether its table parses into the three sections."""
    validity = {}
    for path in sorted((staging / "raw" / raw.REGISTRY_COPIES).glob("*.htm")):
        try:
            registry_page_values(
                path.read_text(encoding="utf-8", errors="replace"), path.stem
            )
        except ValueError:
            validity[path.stem] = False
        else:
            validity[path.stem] = True
    return validity


def read_listing(staging: Path) -> Listing | None:
    """The saved CDX listings of the last fetch-archive run, or None when there are none."""
    files = sorted((staging / "cdx").glob("*.txt"))
    if not files:
        return None
    return [
        capture
        for file in files
        for capture in archive_inventory.parse_cdx(file.read_text(encoding="utf-8"))
    ]


def list_page(staging: Path, index: int, key: str) -> Listing:
    """One CDX listing, saved before it is parsed so a later offline run can read it."""
    text = archive_inventory.fetch(archive_inventory.CDX_URL.format(key=key)).decode(
        "utf-8", "replace"
    )
    target = staging / "cdx" / f"{index}.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return archive_inventory.parse_cdx(text)


def fetch_copy(staging: Path, now: datetime, timestamp: str, original: str) -> bool:
    """Stage one archived copy, after the pause the Internet Archive asks for.

    Returns whether it parses, or False with nothing staged when every attempt fails.
    """
    url = archive_inventory.REPLAY_URL.format(stamp=timestamp, original=original)
    sleep(archive_inventory.PAUSE_SECONDS)
    try:
        payload, last_modified = archive_inventory.retrying(
            partial(download, url), what=url
        )
    except RuntimeError as error:
        print(f"{error}; not staged in this run", file=sys.stderr)
        return False
    stage(
        staging,
        copy_file(timestamp),
        payload,
        url=url,
        last_modified=last_modified,
        fetched_at=stamp(now),
    )
    return staged_copies(staging)[timestamp]


def fetch_archive(now: datetime, staging: Path = STAGING_DIR) -> int:
    """Stage the next archived copy of the registry page for each unresolved year."""
    listing: Listing = []
    for index, key in enumerate(PAGE_KEYS):
        sleep(archive_inventory.PAUSE_SECONDS)
        listing += archive_inventory.retrying(
            partial(list_page, staging, index, key), what=f"CDX listing of {key}"
        )
    recon = [
        original
        for moment, original, status, _ in listing
        if f"{moment:%Y%m%d%H%M%S}" == RECON_COPY and status == "200"
    ]
    if not recon:
        print(f"the listings have no capture {RECON_COPY}", file=sys.stderr)
        return 1

    candidates = year_candidates(listing)
    staged = staged_copies(staging)
    for year, status in year_status(candidates, staged).items():
        if status != PENDING:
            continue
        unstaged = [capture for capture in candidates[year] if capture[0] not in staged]
        tried = len(unstaged) < len(candidates[year])
        for timestamp, original in unstaged if tried else unstaged[:1]:
            if fetch_copy(staging, now, timestamp, original):
                break
    if RECON_COPY not in staged:
        fetch_copy(staging, now, RECON_COPY, recon[0])
    return 0


# --- Offline commands ----------------------------------------------------------------------


def chosen_copies(staging: Path) -> tuple[dict[int, str], dict[str, bool]] | None:
    listing = read_listing(staging)
    if listing is None:
        return None
    staged = staged_copies(staging)
    return year_status(year_candidates(listing), staged), staged


def show_status(staging: Path = STAGING_DIR) -> int:
    """Print what is staged. Exit 0 only when no year is pending and every source is staged."""
    found = chosen_copies(staging)
    if found is None:
        print("no CDX listing is staged: run fetch-archive")
        return 1
    statuses, staged = found
    for year, status in statuses.items():
        print(f"{year}: {status}")
    complete = PENDING not in statuses.values()
    for file in (raw.COLLECTION_RATES, raw.CE_FOOTNOTES):
        present = (staging / "raw" / file).is_file()
        print(f"{file}: {'staged' if present else 'not staged'}")
        complete = complete and present
    recon = staged.get(RECON_COPY)
    print(f"{RECON_COPY}: {'not staged' if recon is None else 'staged'}")
    for timestamp, valid in staged.items():
        if not valid:
            print(f"does not parse: {timestamp}")
    return 0 if complete and recon else 1


def promoted_copies(staging: Path) -> list[str]:
    """The copy files promote takes: each year's chosen copy, then the recon copy."""
    statuses, _ = chosen_copies(staging)
    chosen = [
        status for status in statuses.values() if status not in (PENDING, MISSING)
    ]
    return [copy_file(timestamp) for timestamp in [*chosen, RECON_COPY]]


def promoted_files(staging: Path) -> list[str]:
    return [raw.COLLECTION_RATES, raw.CE_FOOTNOTES, *promoted_copies(staging)]


def page_prose(page: str) -> str:
    """A page's text outside its tables, scripts and styles, with tags stripped."""
    for element in ("table", "script", "style"):
        page = re.sub(rf"(?is)<{element}\b.*?</{element}\s*>", " ", page)
    return raw.cell_text(page)


def show_prose(staging: Path = STAGING_DIR) -> int:
    """Print the prose of the recon copy and each year's chosen copy, for the owner's stop."""
    if chosen_copies(staging) is None:
        print("no CDX listing is staged: run fetch-archive", file=sys.stderr)
        return 1
    for file in promoted_copies(staging):
        path = staging / "raw" / file
        if not path.is_file():
            continue
        print(f"== {path.stem} ==")
        print(page_prose(path.read_text(encoding="utf-8", errors="replace")))
        print()
    return 0


def promote(staging: Path = STAGING_DIR, raw_dir: Path = raw.RAW_DIR) -> int:
    """Copy the staged sources into raw_dir and add their manifest rows; never overwrite."""
    if show_status(staging) != 0:
        print("promote refused: the staged sources are not complete", file=sys.stderr)
        return 1
    files = promoted_files(staging)
    manifest = archive_inventory.read_csv(raw_dir / raw.MANIFEST)
    recorded = {row["file"] for row in manifest}
    clashes = [file for file in files if (raw_dir / file).exists() or file in recorded]
    if clashes:
        print(f"promote refused: already in data/raw/: {clashes}", file=sys.stderr)
        return 1
    provenance = {
        row["file"]: row for row in archive_inventory.read_csv(staging / PROVENANCE)
    }
    added = []
    for file in files:
        target = raw_dir / file
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(staging / "raw" / file, target)
        fields = {
            name: provenance[file][name]
            for name in ("url", "last_modified", "derived_from", "derived_from_sha256")
        }
        added.append(
            vintage_sources.manifest_row(
                file, provenance[file]["fetched_at"], raw_dir=raw_dir, **fields
            )
        )
    archive_inventory.write_csv(
        raw_dir / raw.MANIFEST,
        sorted(manifest + added, key=lambda row: row["file"]),
        vintage_sources.MANIFEST_COLUMNS,
    )
    print(f"promoted {len(added)} files")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command",
        choices=["fetch-labstat", "fetch-archive", "status", "prose", "promote"],
    )
    command = parser.parse_args(argv).command
    now = datetime.now(UTC).replace(microsecond=0)
    if command == "fetch-labstat":
        return fetch_labstat(now)
    if command == "fetch-archive":
        return fetch_archive(now)
    if command == "status":
        return show_status()
    if command == "prose":
        return show_prose()
    return promote()


if __name__ == "__main__":
    raise SystemExit(main())
