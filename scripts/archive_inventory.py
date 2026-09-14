"""Enumerate CES vintages and inventory the seasonal-adjustment files that survive for each.

Roadmap Stage 2 (specs/ces-revisions-roadmap.md) answers Req 9's open question: which vintages
have archived specification, prior-adjustment, and outlier files. BLS serves only the current
files (https://www.bls.gov/web/empsit/cesseasadj.htm), so a past vintage's files survive only
where the Internet Archive copied them. Network capture and offline derivation are separate
subcommands, so the inventory can be rebuilt from the committed evidence without the network:

    uv run python scripts/archive_inventory.py captures    # writes the evidence CSVs
    uv run python scripts/archive_inventory.py inventory   # derives the inventory from them

`captures` writes docs/inventory/es-vintages.csv and docs/inventory/archive-captures.csv.
`inventory` writes docs/inventory/archive-inventory.csv and regenerates the inventory tables in
docs/ces-revisions-review.md.
"""

import argparse
import base64
import csv
import hashlib
import io
import os
import re
import sys
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, replace
from datetime import UTC, date, datetime, time
from functools import partial
from pathlib import Path, PurePosixPath
from time import sleep
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
INVENTORY_DIR = ROOT / "docs" / "inventory"
REVIEW_PATH = ROOT / "docs" / "ces-revisions-review.md"
VINTAGES_PATH = INVENTORY_DIR / "es-vintages.csv"
CAPTURES_PATH = INVENTORY_DIR / "archive-captures.csv"
INVENTORY_PATH = INVENTORY_DIR / "archive-inventory.csv"

# Req 1 and Req 4: the vintage panel starts with the May 2003 publication vintage.
FIRST_REFERENCE_MONTH = date(2003, 5, 1)
RELEASE_INDEX_URL = "https://www.bls.gov/bls/news-release/empsit.htm"
MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
EASTERN = ZoneInfo("America/New_York")
# Employment Situation releases are embargoed until 8:30 a.m. Eastern time.
EMBARGO = time(8, 30)


# --- Vintages ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Vintage:
    """One Employment Situation release: the reference month it first estimates, and when."""

    reference_month: date  # the first day of the month
    release_date: date
    release_url: str


_ITEM = re.compile(r"(?s)<li(?:\s[^>]*)?>(.*?)</li>")
_TAG = re.compile(r"<[^>]+>")
# Most items link a site-relative path, but a few (March and July 2019) link the absolute URL.
_RELEASE_HREF = re.compile(
    r'href="(?:https?://www\.bls\.gov)?'
    r'(/news\.release/(?:archives|history)/empsit_(\d{2})(\d{2})(\d{4})\.(htm|txt|pdf))"'
)
_TITLE = re.compile(r"\b(" + "|".join(MONTHS) + r") (\d{4}) Employment Situation\b")
_FORMAT_PREFERENCE = {"htm": 0, "txt": 1, "pdf": 2}


def parse_release_index(html: str) -> list[Vintage]:
    """Parse the Employment Situation archive index into one vintage per reference month.

    The release date is the MMDDYYYY in each link, the index's only stable key. The reference
    month comes from the item's title, because delayed releases (September 2013 and September
    2025) break any fixed lag between reference month and release.
    """
    vintages: dict[date, Vintage] = {}
    for item in _ITEM.findall(html):
        title = _TITLE.search(_TAG.sub(" ", item))
        links = _RELEASE_HREF.findall(item)
        if title is None or not links:
            continue
        reference = date(int(title.group(2)), MONTHS.index(title.group(1)) + 1, 1)
        path, month, day, year, _ = min(
            links, key=lambda link: _FORMAT_PREFERENCE[link[4]]
        )
        vintage = Vintage(
            reference,
            date(int(year), int(month), int(day)),
            f"https://www.bls.gov{path}",
        )
        earlier = vintages.get(reference)
        if earlier is not None and earlier.release_date != vintage.release_date:
            raise ValueError(
                f"two releases for {reference:%Y-%m}: "
                f"{earlier.release_date} and {vintage.release_date}"
            )
        vintages[reference] = vintage
    return sorted(vintages.values(), key=lambda vintage: vintage.reference_month)


def select_vintages(vintages: list[Vintage], *, now: datetime) -> list[Vintage]:
    """Drop releases still under embargo at `now`: the index links scheduled releases early."""
    return [
        vintage for vintage in vintages if release_instant(vintage.release_date) <= now
    ]


def next_month(month: date) -> date:
    return date(month.year + month.month // 12, month.month % 12 + 1, 1)


def missing_reference_months(
    vintages: list[Vintage], *, start: date = FIRST_REFERENCE_MONTH
) -> list[date]:
    """Reference months from `start` through the latest vintage that no release first estimated."""
    released = {vintage.reference_month for vintage in vintages}
    missing, month = [], start
    while month <= vintages[-1].reference_month:
        if month not in released:
            missing.append(month)
        month = next_month(month)
    return missing


def release_instant(release_date: date) -> datetime:
    """When a release's embargo lifted, in UTC."""
    return datetime.combine(release_date, EMBARGO, tzinfo=EASTERN).astimezone(UTC)


VINTAGE_COLUMNS = ["reference_month", "release_date", "release_url"]


def vintage_rows(vintages: list[Vintage]) -> list[dict[str, str]]:
    return [
        {
            "reference_month": f"{vintage.reference_month:%Y-%m}",
            "release_date": vintage.release_date.isoformat(),
            "release_url": vintage.release_url,
        }
        for vintage in vintages
    ]


def vintages_from_rows(rows: list[dict[str, str]]) -> list[Vintage]:
    return [
        Vintage(
            date.fromisoformat(f"{row['reference_month']}-01"),
            date.fromisoformat(row["release_date"]),
            row["release_url"],
        )
        for row in rows
    ]


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --- Network -------------------------------------------------------------------------------


def user_agent() -> str:
    """BLS asks automated clients for a contact address, which BLS_CONTACT_EMAIL supplies."""
    contact = os.environ.get("BLS_CONTACT_EMAIL")
    return f"ces-revisions/0.1.0 ({contact})" if contact else "ces-revisions/0.1.0"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent()})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


# --- Captures ------------------------------------------------------------------------------

FILE_TYPES = ("specification", "prior_adjustment", "outliers")
_MEMBER_PATTERNS = {
    "specification": re.compile(r"\.spc$"),
    "prior_adjustment": re.compile(r"^prior_adjustment_file\.xlsx?$"),
    "outliers": re.compile(r"^outliers\.xlsx?$"),
}
# Members the ZIPs' readme files account for: the readmes and the calendar regressor files. The
# Internet Archive's 2021-03-18 copy of ces.spec.other.zip also nests an earlier copy of itself,
# which holds only the inputs the technical notes describe: the calendar regressor files, a readme,
# the prior-adjustment file, and the outlier file (https://www.bls.gov/web/empsit/cesseasadjtn.htm).
# `inspect_zip` opens a nested ZIP one level down, so its members face the same check.
_EXPLAINED = re.compile(
    r"^(readme[.\w]*\.txt|new\.readme\.announcement\.txt|f?dum\w*\.dat|ces\.spec\.other\.zip)$"
)


def _unexplained(name: str) -> bool:
    return not _EXPLAINED.match(name) and not any(
        pattern.search(name) for pattern in _MEMBER_PATTERNS.values()
    )


def _nested_names(payload: bytes) -> list[str]:
    """Member names of a ZIP nested inside a seasonal-adjustment ZIP."""
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            return [
                PurePosixPath(info.filename).name.lower()
                for info in archive.infolist()
                if not info.is_dir()
            ]
    except zipfile.BadZipFile:
        return ["unreadable nested zip"]


def inspect_zip(payload: bytes) -> dict[str, str]:
    """Fingerprint each Req 9 file type in a seasonal-adjustment ZIP and list unexplained members.

    A fingerprint hashes member names and CRC-32s, so re-zipping identical files under new
    timestamps keeps it and any change of contents alters it. An empty fingerprint means the
    file type is absent. Fingerprints use top-level members only, but a nested ZIP's members are
    checked for unexplained files too, and reported as `nested.zip/member`.
    """
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        members = [
            (PurePosixPath(info.filename).name.lower(), info.CRC) for info in infos
        ]
        nested = [
            f"{PurePosixPath(info.filename).name.lower()}/{name}"
            for info in infos
            if info.filename.lower().endswith(".zip")
            for name in _nested_names(archive.read(info))
            if _unexplained(name)
        ]
    result = {}
    for file_type, pattern in _MEMBER_PATTERNS.items():
        matched = sorted((name, crc) for name, crc in members if pattern.search(name))
        listing = "\n".join(f"{name} {crc:08x}" for name, crc in matched)
        result[file_type] = (
            hashlib.sha256(listing.encode()).hexdigest()[:16] if matched else ""
        )
    result["unexpected"] = ";".join(
        sorted([*(name for name, _ in members if _unexplained(name)), *nested])
    )
    return result


@dataclass(frozen=True)
class Capture:
    """One copy of a seasonal-adjustment ZIP: an Internet Archive capture or BLS's current file."""

    artifact: str
    source: str  # "internet_archive" or "bls_current"
    timestamp: datetime  # UTC: when the copy was made, or when BLS's file was fetched
    url: str  # where these bytes can be fetched again
    status: str
    # The base-32 SHA-1 of the payload, the Internet Archive's own digest format.
    digest: str
    specification: str = ""
    prior_adjustment: str = ""
    outliers: str = ""
    unexpected: str = ""


CAPTURE_COLUMNS = [field.name for field in fields(Capture)]
_TIMESTAMP = "%Y-%m-%dT%H:%M:%SZ"


def capture_row(capture: Capture) -> dict[str, str]:
    return {**asdict(capture), "timestamp": capture.timestamp.strftime(_TIMESTAMP)}


def capture_from_row(row: dict[str, str]) -> Capture:
    stamp = datetime.strptime(row["timestamp"], _TIMESTAMP).replace(tzinfo=UTC)
    return Capture(**{**row, "timestamp": stamp})


_CDX_LINE = re.compile(r"^(\d{14}) (\S+) (\S+) (\S+)$")


def parse_cdx(text: str) -> list[tuple[datetime, str, str, str]]:
    """Parse CDX lines of `timestamp original statuscode digest`, rejecting anything else.

    The CDX server can answer a throttled request with an HTML page, which must fail loudly
    rather than read as an empty list of captures.
    """
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        match = _CDX_LINE.match(line.strip())
        if match is None:
            raise ValueError(f"not a CDX line: {line[:80]!r}")
        stamp, original, status, digest = match.groups()
        moment = datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        rows.append((moment, original, status, digest))
    return rows


@dataclass(frozen=True)
class Artifact:
    name: str
    current_url: str
    archive_keys: tuple[str, ...]  # CDX URL keys, including the pre-2012 FTP location


# Only the all-employees specification set is inventoried: hours and earnings are out of scope,
# and the second-preliminary sets adjust detailed series that do not aggregate to total nonfarm.
ARTIFACTS = (
    Artifact(
        "all_employees_specifications",
        "https://www.bls.gov/web/empsit/ces.spec.ae.zip",
        (
            "bls.gov/web/empsit/ces.spec.ae.zip",
            "ftp.bls.gov/pub/suppl/empsit.ces.spec.ae.zip",
        ),
    ),
    Artifact(
        "other_inputs",
        "https://www.bls.gov/web/empsit/ces.spec.other.zip",
        (
            "bls.gov/web/empsit/ces.spec.other.zip",
            "ftp.bls.gov/pub/suppl/empsit.ces.spec.other.zip",
        ),
    ),
)
CDX_URL = "https://web.archive.org/cdx/search/cdx?url={key}&fl=timestamp,original,statuscode,digest"
REPLAY_URL = "https://web.archive.org/web/{stamp}id_/{original}"
ATTEMPTS = 5
PAUSE_SECONDS = 3.0


def retrying[T](attempt: Callable[[], T], *, what: str) -> T:
    """Retry `attempt` with a growing pause, because the Internet Archive throttles bursts."""
    retryable = (OSError, ValueError, zipfile.BadZipFile)
    for number in range(1, ATTEMPTS):
        try:
            return attempt()
        except retryable:
            sleep(PAUSE_SECONDS * number)
    try:
        return attempt()
    except retryable as error:
        raise RuntimeError(f"{what} failed after {ATTEMPTS} attempts") from error


def list_captures(key: str) -> list[tuple[datetime, str, str, str]]:
    return parse_cdx(fetch(CDX_URL.format(key=key)).decode("utf-8", "replace"))


def payload_digest(payload: bytes) -> str:
    return base64.b32encode(
        hashlib.sha1(payload, usedforsecurity=False).digest()
    ).decode()


def fetch_and_inspect(url: str) -> tuple[bytes, dict[str, str]]:
    payload = fetch(url)
    return payload, inspect_zip(payload)


def collect_captures(now: datetime) -> list[Capture]:
    """Fetch BLS's current copy and every Internet Archive capture of each artifact."""
    captures = []
    for artifact in ARTIFACTS:
        payload, found = retrying(
            partial(fetch_and_inspect, artifact.current_url), what=artifact.current_url
        )
        captures.append(
            Capture(
                artifact.name,
                "bls_current",
                now,
                artifact.current_url,
                "200",
                payload_digest(payload),
                **found,
            )
        )
        for key in artifact.archive_keys:
            for moment, original, status, digest in retrying(
                partial(list_captures, key), what=f"CDX listing of {key}"
            ):
                replay = REPLAY_URL.format(
                    stamp=f"{moment:%Y%m%d%H%M%S}", original=original
                )
                found, outcome = {}, status
                if status == "200":
                    sleep(PAUSE_SECONDS)
                    try:
                        _, found = retrying(
                            partial(fetch_and_inspect, replay), what=replay
                        )
                    except RuntimeError as error:
                        print(f"{error}; recorded as unreplayable", file=sys.stderr)
                        outcome = "unreplayable"
                captures.append(
                    Capture(
                        artifact.name,
                        "internet_archive",
                        moment,
                        replay,
                        outcome,
                        digest,
                        **found,
                    )
                )
    return captures


# --- Inventory -----------------------------------------------------------------------------

PRESENT = ("bls_current", "internet_archive")
STATUSES = (*PRESENT, "conflicting_captures", "not_archived", "no_release")
INVENTORY_COLUMNS = [
    "reference_month",
    "release_date",
    *FILE_TYPES,
    "unrounded_nsa_inputs",
    "files_complete",
    *(f"{file_type}_evidence" for file_type in FILE_TYPES),
]
_FILE_LABELS = {
    "specification": "Specification files",
    "prior_adjustment": "Prior-adjustment files",
    "outliers": "Outlier files",
}


def live_windows(
    vintages: list[Vintage], *, annual: bool
) -> dict[date, tuple[datetime, datetime | None]]:
    """When each vintage's files were the ones BLS served.

    Prior-adjustment and outlier files change with every release, so a vintage's window runs
    from its release to the next. Specification files change only with the annual benchmark,
    released with January estimates, so their window runs from one benchmark release to the
    next. A vintage before the first benchmark release in the list opens at the list's start.
    """
    opens = [release_instant(vintage.release_date) for vintage in vintages]
    boundaries = [
        index
        for index, vintage in enumerate(vintages)
        if not annual or vintage.reference_month.month == 1
    ]
    windows = {}
    for index, vintage in enumerate(vintages):
        start = max(
            (boundary for boundary in boundaries if boundary <= index), default=0
        )
        end = min(
            (boundary for boundary in boundaries if boundary > index), default=None
        )
        windows[vintage.reference_month] = (
            opens[start],
            None if end is None else opens[end],
        )
    return windows


def resolve_revisits(captures: list[Capture]) -> list[Capture]:
    """Give each revisit record the fingerprints of the opened copy with the same digest.

    The Internet Archive lists a copy identical to an earlier one as a revisit, with status `-`,
    and `captures` does not download it. The same digest means the same bytes, so a revisit
    evidences the same files at its own time. A revisit with no opened twin stays empty.
    """
    opened = {
        (capture.artifact, capture.digest): capture
        for capture in captures
        if capture.status == "200"
    }
    resolved = []
    for capture in captures:
        twin = opened.get((capture.artifact, capture.digest))
        if capture.status == "-" and twin is not None:
            copied = (*FILE_TYPES, "unexpected")
            capture = replace(capture, **{name: getattr(twin, name) for name in copied})
        resolved.append(capture)
    return resolved


def file_status(
    captures: list[Capture], file_type: str, window: tuple[datetime, datetime | None]
) -> tuple[str, str]:
    """One file type's status for one vintage, with the URL of the copy that evidences it.

    Opened copies count, and so do revisit records once `resolve_revisits` has filled them in.
    """
    start, end = window
    holding = sorted(
        (
            capture
            for capture in captures
            if capture.status in ("200", "-")
            and getattr(capture, file_type)
            and start <= capture.timestamp
            and (end is None or capture.timestamp < end)
        ),
        key=lambda capture: capture.timestamp,
    )
    if not holding:
        return "not_archived", ""
    if len({getattr(capture, file_type) for capture in holding}) > 1:
        return "conflicting_captures", " ".join(capture.url for capture in holding)
    chosen = min(holding, key=lambda capture: capture.source != "bls_current")
    return chosen.source, chosen.url


def inventory_rows(
    vintages: list[Vintage], captures: list[Capture]
) -> list[dict[str, str]]:
    """One row per reference month from May 2003 through the latest vintage."""
    captures = resolve_revisits(captures)
    windows = {
        "specification": live_windows(vintages, annual=True),
        "prior_adjustment": live_windows(vintages, annual=False),
        "outliers": live_windows(vintages, annual=False),
    }
    released = {vintage.reference_month: vintage for vintage in vintages}
    rows = []
    month = FIRST_REFERENCE_MONTH
    while month <= vintages[-1].reference_month:
        row = dict.fromkeys(INVENTORY_COLUMNS, "")
        row["reference_month"] = f"{month:%Y-%m}"
        vintage = released.get(month)
        if vintage is None:
            row.update(dict.fromkeys(FILE_TYPES, "no_release"))
            row.update(unrounded_nsa_inputs="no_release", files_complete="false")
        else:
            row["release_date"] = vintage.release_date.isoformat()
            for file_type in FILE_TYPES:
                row[file_type], row[f"{file_type}_evidence"] = file_status(
                    captures, file_type, windows[file_type][month]
                )
            # cesseasadjtn.htm: X-13 runs on unrounded NSA data, and BLS publishes rounded data.
            row["unrounded_nsa_inputs"] = "not_published"
            complete = all(row[file_type] in PRESENT for file_type in FILE_TYPES)
            row["files_complete"] = "true" if complete else "false"
        rows.append(row)
        month = next_month(month)
    return rows


def _status_cell(row: dict[str, str], file_type: str) -> str:
    status = row[file_type]
    if status in PRESENT:
        return f"[`{status}`]({row[f'{file_type}_evidence']})"
    return f"`{status}`"


def render_findings(rows: list[dict[str, str]]) -> str:
    released = [row for row in rows if row["release_date"]]
    gaps = [row["reference_month"] for row in rows if not row["release_date"]]
    lines = [
        (
            f"- Releases inventoried: {len(released)}, estimating reference months "
            f"{released[0]['reference_month']} through {released[-1]['reference_month']} "
            f"(released {released[0]['release_date']} to {released[-1]['release_date']})."
        ),
        f"- Reference months without a release of their own: {', '.join(gaps) or 'none'}.",
    ]
    for file_type, label in _FILE_LABELS.items():
        present = [
            row["reference_month"] for row in released if row[file_type] in PRESENT
        ]
        earliest = present[0] if present else "none"
        lines.append(
            f"- {label} survive for {len(present)} of {len(released)} releases; "
            f"the earliest such release estimates {earliest}."
        )
    complete = sum(row["files_complete"] == "true" for row in released)
    conflicts = sum(
        row[file_type] == "conflicting_captures"
        for row in released
        for file_type in FILE_TYPES
    )
    lines.append(
        f"- All three file types survive for {complete} of {len(released)} releases, "
        f"and {conflicts} cells are `conflicting_captures`."
    )
    lines.append("- Unrounded NSA inputs are `not_published` for every release.")
    return "\n".join(lines) + "\n"


def render_coverage(rows: list[dict[str, str]]) -> str:
    lines = [
        "| Reference year | Releases | Specification | Prior adjustment | Outliers | All three |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    years = sorted({row["reference_month"][:4] for row in rows})
    for year in [*years, "Total"]:
        subset = [
            row
            for row in rows
            if row["release_date"] and year in ("Total", row["reference_month"][:4])
        ]
        counts = [str(sum(row[t] in PRESENT for row in subset)) for t in FILE_TYPES]
        complete = sum(row["files_complete"] == "true" for row in subset)
        lines.append(f"| {year} | {len(subset)} | {' | '.join(counts)} | {complete} |")
    return "\n".join(lines) + "\n"


def render_vintages(rows: list[dict[str, str]]) -> str:
    lines = [
        "<details>",
        "<summary>One row per reference month from May 2003</summary>",
        "",
        (
            "| Reference month | Release | Specification | Prior adjustment | Outliers "
            "| Unrounded NSA inputs | All three |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        cells = [
            row["reference_month"],
            row["release_date"] or "none",
            *(_status_cell(row, file_type) for file_type in FILE_TYPES),
            f"`{row['unrounded_nsa_inputs']}`",
            row["files_complete"],
        ]
        lines.append(f"| {' | '.join(cells)} |")
    lines += ["", "</details>"]
    return "\n".join(lines) + "\n"


def render_blocks(rows: list[dict[str, str]]) -> dict[str, str]:
    """The generated blocks of the review's archive section, keyed by marker name."""
    blocks = {
        "archive-findings": render_findings(rows),
        "archive-coverage": render_coverage(rows),
        "archive-vintages": render_vintages(rows),
    }
    # Blank lines keep each table or list a block of its own beside the HTML-comment markers.
    return {name: f"\n{body}\n" for name, body in blocks.items()}


def _markers(name: str) -> tuple[str, str]:
    return f"<!-- BEGIN GENERATED {name} -->\n", f"<!-- END GENERATED {name} -->"


def _block_span(document: str, name: str) -> tuple[int, int]:
    begin, end = _markers(name)
    start = document.find(begin)
    stop = document.find(end, start) if start >= 0 else -1
    if stop < 0:
        raise ValueError(f"document has no generated block named {name!r}")
    return start + len(begin), stop


def generated_block(document: str, name: str) -> str:
    start, stop = _block_span(document, name)
    return document[start:stop]


def replace_generated(document: str, name: str, body: str) -> str:
    start, stop = _block_span(document, name)
    return document[:start] + body + document[stop:]


# --- Command line --------------------------------------------------------------------------


def capture_evidence() -> int:
    """Network step: write the vintage list and every copy's fingerprints."""
    now = datetime.now(UTC).replace(microsecond=0)
    index = retrying(partial(fetch, RELEASE_INDEX_URL), what=RELEASE_INDEX_URL)
    vintages = select_vintages(
        parse_release_index(index.decode("utf-8", "replace")), now=now
    )
    write_csv(VINTAGES_PATH, vintage_rows(vintages), VINTAGE_COLUMNS)
    captures = collect_captures(now)
    write_csv(
        CAPTURES_PATH, [capture_row(capture) for capture in captures], CAPTURE_COLUMNS
    )
    unexplained = [capture for capture in captures if capture.unexpected]
    for capture in unexplained:
        print(
            f"unexplained members in {capture.url}: {capture.unexpected}",
            file=sys.stderr,
        )
    return 1 if unexplained else 0


def derive_inventory() -> int:
    """Offline step: derive the inventory from the evidence and regenerate the review's tables."""
    vintages = vintages_from_rows(read_csv(VINTAGES_PATH))
    captures = [capture_from_row(row) for row in read_csv(CAPTURES_PATH)]
    rows = inventory_rows(vintages, captures)
    write_csv(INVENTORY_PATH, rows, INVENTORY_COLUMNS)
    document = REVIEW_PATH.read_text(encoding="utf-8")
    for name, body in render_blocks(rows).items():
        document = replace_generated(document, name, body)
    REVIEW_PATH.write_text(document, encoding="utf-8")
    return 0


COMMANDS = {"captures": capture_evidence, "inventory": derive_inventory}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=COMMANDS)
    return COMMANDS[parser.parse_args(argv).command]()


if __name__ == "__main__":
    raise SystemExit(main())
