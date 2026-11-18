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
    INNERMOST_TABLE,
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
# Replays that neither parse nor match their digest: kept for reading, never staged as copies.
UNVOUCHED = "unvouched"
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


class FramingError(Exception):
    """A response whose body has neither chunked framing nor a Content-Length.

    http.client cannot tell such a body cut off from a whole one. The server's framing does not
    change between attempts, so `archive_inventory.retrying` does not retry this error.
    """


FRAMING_HEADERS = ("Transfer-Encoding", "Content-Encoding", "Connection")


def body_length(url: str, response: http.client.HTTPResponse) -> int | None:
    """The body's Content-Length, or None when chunked framing marks the body's end.

    http.client raises IncompleteRead for a chunked body cut off before its last chunk, as it
    does for a body short of its Content-Length, so either framing shows a whole body. As in
    http.client, chunked framing wins over a Content-Length.
    """
    headers = response.headers
    if headers.get("Transfer-Encoding", "").lower() == "chunked":
        return None
    expected = headers.get("Content-Length")
    if expected is not None:
        return int(expected)
    framing = ", ".join(
        f"{name}: {headers.get(name, 'none')}" for name in FRAMING_HEADERS
    )
    raise FramingError(
        f"{url} answered HTTP/{response.version / 10:.1f} {response.status} with neither "
        f"chunked framing nor a Content-Length ({framing}); not retried"
    )


def stream_download(url: str, target: Path) -> tuple[str, str]:
    """Stream url to target: the body's SHA-256 and its Last-Modified header.

    A body whose length differs from Content-Length, or that http.client ends with
    IncompleteRead, raises ValueError, which `archive_inventory.retrying` retries, and leaves
    no file at target. A body with no framing to check raises FramingError (`body_length`).
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
            expected = body_length(url, response)
            while chunk := response.read(CHUNK_BYTES):
                digest.update(chunk)
                handle.write(chunk)
                written += len(chunk)
            last_modified = response.headers.get("Last-Modified", "")
        if expected is not None and written != expected:
            raise ValueError(f"{url} sent {written} bytes of {expected}")
        complete = True
    except http.client.IncompleteRead as error:
        raise ValueError(f"{url} ended after {written} bytes") from error
    finally:
        if not complete:
            target.unlink(missing_ok=True)
    return digest.hexdigest(), last_modified


def download(url: str) -> tuple[bytes, str]:
    """A whole payload and its Last-Modified header.

    A body cut off before its Content-Length, or ended by IncompleteRead, raises ValueError,
    which `archive_inventory.retrying` retries. A body with no framing to check raises
    FramingError (`body_length`).
    """
    try:
        with archive_inventory.open_url(url, timeout=TIMEOUT_SECONDS) as response:
            expected = body_length(url, response)
            payload = response.read()
            last_modified = response.headers.get("Last-Modified", "")
    except http.client.IncompleteRead as error:
        raise ValueError(f"{url} ended early") from error
    if expected is not None and len(payload) != expected:
        raise ValueError(f"{url} sent {len(payload)} bytes of {expected}")
    return payload, last_modified


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
    """Each year's chosen copy, found by walking its captures latest first.

    The walk passes over an invalid staged copy and takes the first valid one. An unstaged
    capture ends it with the year pending, so an older valid copy waits until every newer
    capture is checked. A year whose every capture is staged and invalid is missing.
    """
    status = {}
    for year, captures in candidates.items():
        status[year] = MISSING
        for timestamp, _ in captures:
            if timestamp not in staged:
                status[year] = PENDING
                break
            if staged[timestamp]:
                status[year] = timestamp
                break
    return status


def copy_file(timestamp: str) -> str:
    return f"{raw.REGISTRY_COPIES}/{timestamp}.htm"


def parses(page: bytes, timestamp: str) -> bool:
    """Whether a copy's table parses into the three sections."""
    try:
        registry_page_values(page.decode("utf-8", errors="replace"), timestamp)
    except ValueError:
        return False
    return True


def staged_copies(staging: Path) -> dict[str, bool]:
    """Each staged copy's timestamp, and whether its table parses into the three sections."""
    return {
        path.stem: parses(path.read_bytes(), path.stem)
        for path in sorted((staging / "raw" / raw.REGISTRY_COPIES).glob("*.htm"))
    }


def listing_file(staging: Path, index: int) -> Path:
    return staging / "cdx" / f"{index}.txt"


def read_listing(staging: Path) -> Listing | None:
    """The saved CDX listings of every address, or None while one is missing."""
    files = [listing_file(staging, index) for index in range(len(PAGE_KEYS))]
    if not all(file.is_file() for file in files):
        return None
    return [
        capture
        for file in files
        for capture in archive_inventory.parse_cdx(file.read_text(encoding="utf-8"))
    ]


def list_page(staging: Path, index: int, key: str) -> Listing:
    """One CDX listing, saved for a later offline run once it parses into captures.

    The text is saved beside the listing first, so a listing that fails to parse can be read,
    and it replaces the saved listing only when it holds captures: an empty or throttled answer
    raises ValueError, which `archive_inventory.retrying` retries.
    """
    payload, _ = download(archive_inventory.CDX_URL.format(key=key))
    text = payload.decode("utf-8", "replace")
    target = listing_file(staging, index)
    received = target.with_suffix(".received")
    received.parent.mkdir(parents=True, exist_ok=True)
    received.write_text(text, encoding="utf-8")
    captures = archive_inventory.parse_cdx(text)
    if not captures:
        raise ValueError(f"the CDX listing of {key} holds no captures")
    received.replace(target)
    return captures


def fetch_copy(
    staging: Path, now: datetime, timestamp: str, original: str, digest: str
) -> bool | None:
    """Stage one archived copy, after the pause the Internet Archive asks for.

    A replay that neither parses nor hashes to the listing's digest is not the copy, such as a
    throttled answer, so it counts as a failed attempt. Returns whether the staged copy parses,
    or None with nothing staged when every attempt fails.
    """
    url = archive_inventory.REPLAY_URL.format(stamp=timestamp, original=original)

    def attempt() -> tuple[bytes, str]:
        payload, last_modified = download(url)
        vouched = archive_inventory.payload_digest(payload) == digest
        if not vouched and not parses(payload, timestamp):
            kept = staging / UNVOUCHED / f"{timestamp}.htm"
            kept.parent.mkdir(parents=True, exist_ok=True)
            kept.write_bytes(payload)
            raise ValueError(f"{url} neither parses nor matches its digest {digest}")
        return payload, last_modified

    sleep(archive_inventory.PAUSE_SECONDS)
    try:
        payload, last_modified = archive_inventory.retrying(attempt, what=url)
    except RuntimeError as error:
        print(f"{error}; not staged in this run", file=sys.stderr)
        return None
    stage(
        staging,
        copy_file(timestamp),
        payload,
        url=url,
        last_modified=last_modified,
        fetched_at=stamp(now),
    )
    return parses(payload, timestamp)


def fetch_archive(now: datetime, staging: Path = STAGING_DIR) -> int:
    """Stage the next archived copy of the registry page for each unresolved year."""
    listing: Listing = []
    for index, key in enumerate(PAGE_KEYS):
        sleep(archive_inventory.PAUSE_SECONDS)
        listing += archive_inventory.retrying(
            partial(list_page, staging, index, key), what=f"CDX listing of {key}"
        )
    recon = [
        (original, digest)
        for moment, original, status, digest in listing
        if f"{moment:%Y%m%d%H%M%S}" == RECON_COPY and status == "200"
    ]
    if not recon:
        print(f"the listings have no capture {RECON_COPY}", file=sys.stderr)
        return 1

    digests = {f"{moment:%Y%m%d%H%M%S}": digest for moment, _, _, digest in listing}
    candidates = year_candidates(listing)
    staged = staged_copies(staging)
    failed_replays = 0
    for year, status in year_status(candidates, staged).items():
        if status != PENDING:
            continue
        tried = any(timestamp in staged for timestamp, _ in candidates[year])
        for timestamp, original in candidates[year]:
            if timestamp in staged:
                if staged[timestamp]:
                    break
                continue
            valid = fetch_copy(staging, now, timestamp, original, digests[timestamp])
            if valid is None:
                failed_replays += 1
            if valid is not False or not tried:
                break
    if RECON_COPY not in staged:
        recon_original, recon_digest = recon[0]
        if fetch_copy(staging, now, RECON_COPY, recon_original, recon_digest) is None:
            failed_replays += 1
    return 1 if failed_replays else 0


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
    for path in sorted((staging / UNVOUCHED).glob("*.htm")):
        if path.stem not in staged:
            print(f"unvouched: {path.stem}")
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
    """A page's text outside its innermost tables, scripts and styles, with tags stripped.

    A layout table's text stays, so the prose of a page that nests its registry table in one is
    read too.
    """
    for element in ("script", "style"):
        page = re.sub(rf"(?is)<{element}\b.*?</{element}\s*>", " ", page)
    return raw.cell_text(INNERMOST_TABLE.sub(" ", page))


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
    unrecorded = [file for file in files if file not in provenance]
    if unrecorded:
        print(f"promote refused: no provenance row for {unrecorded}", file=sys.stderr)
        return 1
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
    try:
        if command == "fetch-labstat":
            return fetch_labstat(now)
        if command == "fetch-archive":
            return fetch_archive(now)
    except FramingError as error:
        print(error, file=sys.stderr)
        return 1
    if command == "status":
        return show_status()
    if command == "prose":
        return show_prose()
    return promote()


if __name__ == "__main__":
    raise SystemExit(main())
