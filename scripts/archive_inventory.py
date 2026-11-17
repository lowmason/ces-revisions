"""Enumerate CES vintages and inventory the seasonal-adjustment files that survive for each.

Roadmap Stage 2 (specs/completed/ces-revisions-roadmap.md) answers Req 9's open question: which vintages
have archived specification, prior-adjustment, and outlier files. BLS serves only the current
files (https://www.bls.gov/web/empsit/cesseasadj.htm), so a past vintage's files survive only
where the Internet Archive copied them. Network capture and offline derivation are separate
subcommands, so the inventory can be rebuilt from the committed evidence without the network:

    uv run python scripts/archive_inventory.py captures    # full run: BLS and every Archive copy
    uv run python scripts/archive_inventory.py current     # light run: BLS's current files only
    uv run python scripts/archive_inventory.py inventory   # derives the inventory from them
    uv run python -I scripts/archive_inventory.py adopt ZIP...  # offline: store a recorded local ZIP

`captures` and `current` merge what they observe into docs/inventory/archive-captures.csv, so no
recorded row is dropped, rewrite docs/inventory/es-vintages.csv, and store every ZIP they open
in data/seasonal/raw/ (specs/seasonal-adjustment-file-store.md). `adopt` stores a local ZIP whose
digest a committed row records, and uses only the standard library. `inventory` writes
docs/inventory/archive-inventory.csv and regenerates the inventory tables in
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
import tempfile
import urllib.request
import zipfile
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
from datetime import UTC, date, datetime, time
from functools import partial
from operator import attrgetter
from pathlib import Path, PurePosixPath
from time import sleep
from urllib.parse import urlsplit
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


USER_AGENT_PRODUCT = "ces-revisions/0.1.0"


def is_bls_host(url: str) -> bool:
    """True for bls.gov and its subdomains. The host decides, never the path."""
    host = urlsplit(url).hostname or ""
    return host == "bls.gov" or host.endswith(".bls.gov")


def user_agent(url: str, contact: str | None = None) -> str:
    """The User-Agent for a request to `url`.

    BLS asks automated clients for a contact address, which BLS_CONTACT_EMAIL supplies when
    `contact` is None. Only bls.gov hosts get it: every other host, the Internet Archive's
    replays of BLS pages included, gets the bare product name.
    """
    if contact is None:
        contact = os.environ.get("BLS_CONTACT_EMAIL")
    if contact and is_bls_host(url):
        return f"{USER_AGENT_PRODUCT} ({contact})"
    return USER_AGENT_PRODUCT


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent(url)})
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


def newest_member(payload: bytes) -> str:
    """The latest modification time among a ZIP's members, as ISO 8601 with no timezone.

    The scope is the fingerprints': top-level members that are not directories. A nested ZIP
    counts by its own entry's time, and its members are not read. ZIP times are the zipping
    machine's local clock and carry no timezone, so the value has no `Z` and is not UTC.
    """
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        # `date_time` is (year, month, day, hour, minute, second), so tuples sort as times do.
        times = [info.date_time for info in archive.infolist() if not info.is_dir()]
    if not times:
        return ""
    return "{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}".format(*max(times))


# The `source` of a row for a ZIP fetched from BLS itself.
BLS_SOURCE = "bls_fetched"


@dataclass(frozen=True)
class Capture:
    """One copy of a seasonal-adjustment ZIP: an Internet Archive capture or BLS's current file."""

    artifact: str
    source: str  # "internet_archive" or BLS_SOURCE
    timestamp: datetime  # UTC: when the copy was made, or when BLS's file was fetched
    url: str  # where these bytes were fetched
    status: str
    # The base-32 SHA-1 of the payload, the Internet Archive's own digest format.
    digest: str
    specification: str = ""
    prior_adjustment: str = ""
    outliers: str = ""
    unexpected: str = ""
    # The newest member's time on the zipping machine's clock (see `newest_member`), for an
    # opened copy only. It tests that a copy holds files made inside the window it is credited to.
    newest_member: str = ""


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


def fetch_and_inspect(
    url: str, digest: str | None = None
) -> tuple[bytes, dict[str, str]]:
    """Fetch a ZIP and fingerprint it. With a `digest`, bytes that do not hash to it are an error.

    The error is a ValueError, so `retrying` counts the attempt as failed: the Internet Archive
    answers a throttled replay with a page that is not the copy.
    """
    payload = fetch(url)
    if digest is not None and payload_digest(payload) != digest:
        raise ValueError(
            f"{url}: the bytes' digest {payload_digest(payload)} is not the listed {digest}"
        )
    return payload, {**inspect_zip(payload), "newest_member": newest_member(payload)}


def fetch_vintages(now: datetime) -> list[Vintage]:
    """The Employment Situation releases whose embargo has lifted at `now`."""
    index = retrying(partial(fetch, RELEASE_INDEX_URL), what=RELEASE_INDEX_URL)
    return select_vintages(
        parse_release_index(index.decode("utf-8", "replace")), now=now
    )


def fetch_bls_copies(now: datetime) -> list[tuple[Capture, bytes]]:
    """BLS's current copy of each artifact, as a row stamped `now` and the bytes it describes."""
    copies = []
    for artifact in ARTIFACTS:
        payload, found = retrying(
            partial(fetch_and_inspect, artifact.current_url), what=artifact.current_url
        )
        capture = Capture(
            artifact.name,
            BLS_SOURCE,
            now,
            artifact.current_url,
            "200",
            payload_digest(payload),
            **found,
        )
        copies.append((capture, payload))
    return copies


def collect_archive_copies() -> list[tuple[Capture, bytes | None]]:
    """Every Internet Archive capture of each artifact, each `200` copy replayed and opened.

    A replay counts only when its bytes hash to the digest the listing gives. One that fails
    every attempt becomes `unreplayable` with no bytes.
    """
    copies = []
    for artifact in ARTIFACTS:
        for key in artifact.archive_keys:
            for moment, original, status, digest in retrying(
                partial(list_captures, key), what=f"CDX listing of {key}"
            ):
                replay = REPLAY_URL.format(
                    stamp=f"{moment:%Y%m%d%H%M%S}", original=original
                )
                payload, found, outcome = None, {}, status
                if status == "200":
                    sleep(PAUSE_SECONDS)
                    try:
                        payload, found = retrying(
                            partial(fetch_and_inspect, replay, digest), what=replay
                        )
                    except RuntimeError as error:
                        print(f"{error}; not opened in this run", file=sys.stderr)
                        outcome = "unreplayable"
                capture = Capture(
                    artifact.name,
                    "internet_archive",
                    moment,
                    replay,
                    outcome,
                    digest,
                    **found,
                )
                copies.append((capture, payload))
    return copies


# --- File store ----------------------------------------------------------------------------

# Each artifact's ZIPs are kept once per digest (specs/seasonal-adjustment-file-store.md).
STORE_DIR = ROOT / "data" / "seasonal" / "raw"
STORE_MANIFEST = "manifest.csv"
STORE_MANIFEST_COLUMNS = ["file", "sha256", "bytes", "digest"]


def store_folder(artifact: str) -> str:
    """The artifact's folder in the store: its current URL's file name without `.zip`."""
    for known in ARTIFACTS:
        if known.name == artifact:
            return PurePosixPath(urlsplit(known.current_url).path).stem
    raise ValueError(f"unknown artifact {artifact!r}")


def stored_file(artifact: str, digest: str) -> str:
    """A stored ZIP's path relative to the store."""
    return f"{store_folder(artifact)}/{digest}.zip"


def evidence(capture: Capture) -> str:
    """What backs a status that a copy credits.

    A BLS copy's URL serves another file once BLS replaces it, so its stored ZIP is the evidence,
    as a path relative to the repository root. An Internet Archive copy's URL still serves the
    same bytes.
    """
    if capture.source == BLS_SOURCE:
        folder = STORE_DIR.relative_to(ROOT).as_posix()
        return f"{folder}/{stored_file(capture.artifact, capture.digest)}"
    return capture.url


class Staging:
    """New files held in a temporary folder until `promote` moves them into place."""

    def __init__(self, store_dir: Path, folder: Path):
        self.store_dir = store_dir
        self.folder = folder
        self._zip_rows: dict[str, dict[str, str]] = {}
        self._files: dict[Path, Path] = {}

    def _staged_zip(self, file: str) -> Path:
        return self.folder / "store" / file

    def has(self, artifact: str, digest: str) -> bool:
        file = stored_file(artifact, digest)
        return file in self._zip_rows or (self.store_dir / file).is_file()

    def put(self, artifact: str, payload: bytes) -> str:
        """Stage a ZIP unless its bytes are already stored. Returns its digest."""
        digest = payload_digest(payload)
        file = stored_file(artifact, digest)
        held = (
            self._staged_zip(file) if file in self._zip_rows else self.store_dir / file
        )
        if held.is_file():
            if held.read_bytes() != payload:
                raise ValueError(f"different payload under {file}")
            return digest
        staged = self._staged_zip(file)
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(payload)
        self._zip_rows[file] = {
            "file": file,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": str(len(payload)),
            "digest": digest,
        }
        return digest

    def path_for(self, target: Path) -> Path:
        """Where to write `target`'s new content. `promote` moves it to `target`."""
        if target not in self._files:
            staged = self.folder / "files" / str(len(self._files)) / target.name
            staged.parent.mkdir(parents=True)
            self._files[target] = staged
        return self._files[target]

    def promote(self) -> None:
        """Move everything staged into place, ZIPs first so no row names a missing file."""
        manifest = self.folder / STORE_MANIFEST
        manifest_path = self.store_dir / STORE_MANIFEST
        rows = dict(self._zip_rows)
        if manifest_path.is_file():
            rows.update({row["file"]: row for row in read_csv(manifest_path)})
        write_csv(
            manifest,
            [rows[file] for file in sorted(rows)],
            STORE_MANIFEST_COLUMNS,
        )
        moves = [
            (self._staged_zip(file), self.store_dir / file)
            for file in sorted(self._zip_rows)
        ]
        moves.append((manifest, manifest_path))
        moves.extend((staged, target) for target, staged in self._files.items())
        for staged, target in moves:
            target.parent.mkdir(parents=True, exist_ok=True)
            staged.replace(target)
        self._zip_rows.clear()
        self._files.clear()


@contextmanager
def staging(store_dir: Path) -> Iterator[Staging]:
    """Stage a run's new files beside the store. Nothing is promoted unless the caller says so."""
    store_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".staging-", dir=store_dir.parent
    ) as folder:
        yield Staging(store_dir, Path(folder))


def check_store(store_dir: Path) -> list[str]:
    """One message per file that breaks the store's rules. Empty when the store is sound."""
    manifest_path = store_dir / STORE_MANIFEST
    listed = (
        {row["file"]: row for row in read_csv(manifest_path)}
        if manifest_path.is_file()
        else {}
    )
    present = {
        path.relative_to(store_dir).as_posix()
        for path in store_dir.rglob("*")
        if path.is_file() and path.name != ".DS_Store"
    } - {STORE_MANIFEST}
    problems = []
    for file in sorted(present | set(listed)):
        if file not in present:
            problems.append(f"{file}: listed in the manifest but missing")
            continue
        payload = (store_dir / file).read_bytes()
        actual = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": str(len(payload)),
            "digest": payload_digest(payload),
        }
        reasons = []
        if file.endswith(".zip") and actual["digest"] != PurePosixPath(file).stem:
            reasons.append(f"its bytes have digest {actual['digest']}, not its name")
        if file not in listed:
            reasons.append("not in the manifest")
        reasons.extend(
            f"manifest {column} is {listed[file].get(column)}, the file's is {value}"
            for column, value in actual.items()
            if file in listed and listed[file].get(column) != value
        )
        if reasons:
            problems.append(f"{file}: {'; '.join(reasons)}")
    return problems


# --- Merge ---------------------------------------------------------------------------------

_OPENED_FIELDS = (
    "specification",
    "prior_adjustment",
    "outliers",
    "unexpected",
    "newest_member",
)


def capture_key(capture: Capture) -> tuple[str, str, str, datetime]:
    return (capture.artifact, capture.source, capture.url, capture.timestamp)


def _merge_pair(old: Capture, new: Capture) -> Capture:
    """The row that stands for a committed row and the observation that shares its key."""
    if old.digest != new.digest:
        raise ValueError(
            f"{old.url}: the listed digest {new.digest} differs from the committed {old.digest}"
        )
    if old.status == "200":
        if new.status == "200":
            for name in _OPENED_FIELDS:
                if getattr(old, name) != getattr(new, name):
                    raise ValueError(
                        f"{old.url}: the opened copy's {name} {getattr(new, name)!r} differs "
                        f"from the committed {getattr(old, name)!r}"
                    )
        return old
    if old.status == "payload_lost":
        return old
    return new


def merge_captures(committed: list[Capture], observed: list[Capture]) -> list[Capture]:
    """Merge a run's observations into the committed rows (specs/seasonal-adjustment-file-store.md).

    Rows with one key pair up as a multiset, the ith committed row with the ith observed row.
    A committed row with no partner stays, and an observed row with no partner is added. Any
    mismatch raises ValueError, so a run that merges writes nothing.
    """
    waiting: defaultdict[tuple, deque[Capture]] = defaultdict(deque)
    for new in observed:
        waiting[capture_key(new)].append(new)
    merged = []
    for old in committed:
        partners = waiting[capture_key(old)]
        merged.append(_merge_pair(old, partners.popleft()) if partners else old)
    merged.extend(new for partners in waiting.values() for new in partners)
    return sorted(merged, key=attrgetter("artifact", "source", "timestamp", "url"))


# --- Record --------------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidencePaths:
    """Where a network run reads and writes the committed evidence."""

    vintages: Path
    captures: Path
    store: Path

    @classmethod
    def at(cls, root: Path) -> EvidencePaths:
        return cls(
            root / "docs" / "inventory" / "es-vintages.csv",
            root / "docs" / "inventory" / "archive-captures.csv",
            root / "data" / "seasonal" / "raw",
        )


EVIDENCE = EvidencePaths(VINTAGES_PATH, CAPTURES_PATH, STORE_DIR)


def record_run(
    observed: list[tuple[Capture, bytes | None]],
    vintages: list[Vintage],
    paths: EvidencePaths,
) -> list[Capture]:
    """Merge a run's observations into the committed rows and write the run's evidence.

    The merge raises before anything is staged. Every opened ZIP, the vintage list, the merged
    rows and the store manifest then move into place together, ZIPs first, or not at all.
    Returns the merged rows.
    """
    committed = (
        [capture_from_row(row) for row in read_csv(paths.captures)]
        if paths.captures.is_file()
        else []
    )
    merged = merge_captures(committed, [capture for capture, _ in observed])
    with staging(paths.store) as run:
        for capture, payload in observed:
            if payload is not None:
                run.put(capture.artifact, payload)
        write_csv(run.path_for(paths.vintages), vintage_rows(vintages), VINTAGE_COLUMNS)
        write_csv(
            run.path_for(paths.captures),
            [capture_row(capture) for capture in merged],
            CAPTURE_COLUMNS,
        )
        run.promote()
    return merged


# --- Inventory -----------------------------------------------------------------------------

PRESENT = (BLS_SOURCE, "internet_archive")
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
    evidences the same files, with the same member dates. A specification revisit counts at its
    own time, and a prior-adjustment or outlier revisit counts by its member date
    (`monthly_moment`). A revisit with no opened twin stays empty.
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
            copied = (*FILE_TYPES, "unexpected", "newest_member")
            capture = replace(capture, **{name: getattr(twin, name) for name in copied})
        resolved.append(capture)
    return resolved


def member_release(capture: Capture, vintages: list[Vintage]) -> Vintage | None:
    """The release a copy's files were made for: the first dated on or after its newest member.

    BLS builds the prior-adjustment and outlier files a few days before each release, but it has
    posted an earlier release's build as current on a release day. The ZIP that the Internet
    Archive copied on 2015-09-05 was last modified 14 minutes after the August 2015 release, yet
    holds files dated 2015-08-04, made for the July release of 2015-08-07. A ZIP time has no
    timezone, so the newest member's calendar date is compared with the release dates, which are
    Eastern, and a file dated on a release's own day belongs to that release. None means the
    files are newer than every listed release.
    """
    made = date.fromisoformat(capture.newest_member[:10])
    return next((vintage for vintage in vintages if vintage.release_date >= made), None)


def monthly_moment(capture: Capture, vintages: list[Vintage]) -> datetime | None:
    """When a copy's prior-adjustment and outlier files went live: their release's embargo.

    A copy without a member date, as in evidence written before `captures` recorded them, counts
    at its own time. None leaves the copy out: its files belong to no listed release.
    """
    if not capture.newest_member:
        return capture.timestamp
    release = member_release(capture, vintages)
    return None if release is None else release_instant(release.release_date)


def _inside(window: tuple[datetime, datetime | None], moment: datetime | None) -> bool:
    start, end = window
    return moment is not None and start <= moment and (end is None or moment < end)


def file_status(
    captures: list[Capture],
    file_type: str,
    window: tuple[datetime, datetime | None],
    moment: Callable[[Capture], datetime | None] = attrgetter("timestamp"),
) -> tuple[str, str]:
    """One file type's status for one vintage, with the `evidence` of the copy behind it.

    Opened copies count, and so do revisit records once `resolve_revisits` has filled them in.
    `moment` gives the time a copy evidences, by default its own, and None leaves it out.
    """
    holding = sorted(
        (
            capture
            for capture in captures
            if capture.status in ("200", "-")
            and getattr(capture, file_type)
            and _inside(window, moment(capture))
        ),
        key=lambda capture: capture.timestamp,
    )
    if not holding:
        return "not_archived", ""
    if len({getattr(capture, file_type) for capture in holding}) > 1:
        return "conflicting_captures", " ".join(
            evidence(capture) for capture in holding
        )
    chosen = min(holding, key=lambda capture: capture.source != BLS_SOURCE)
    return chosen.source, evidence(chosen)


def inventory_rows(
    vintages: list[Vintage], captures: list[Capture]
) -> list[dict[str, str]]:
    """One row per reference month from May 2003 through the latest vintage.

    A specification copy evidences the benchmark window that holds the copy. A prior-adjustment
    or outlier copy evidences the release its files were made for (`monthly_moment`), which is
    earlier than the copy's own window when BLS posted an earlier release's build as current.
    """
    captures = resolve_revisits(captures)
    windows = {
        "specification": live_windows(vintages, annual=True),
        "prior_adjustment": live_windows(vintages, annual=False),
        "outliers": live_windows(vintages, annual=False),
    }
    by_files = partial(monthly_moment, vintages=vintages)
    moments = {
        "specification": attrgetter("timestamp"),
        "prior_adjustment": by_files,
        "outliers": by_files,
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
                    captures, file_type, windows[file_type][month], moments[file_type]
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
        target = row[f"{file_type}_evidence"]
        # The review sits in docs/, and a stored ZIP's evidence is relative to the repository root.
        link = f"../{target}" if status == BLS_SOURCE else target
        return f"[`{status}`]({link})"
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


def report_unexplained(merged: list[Capture]) -> bool:
    """Print each row with members no readme explains. True when there was one."""
    unexplained = [capture for capture in merged if capture.unexpected]
    for capture in unexplained:
        print(
            f"unexplained members in {capture.url}: {capture.unexpected}",
            file=sys.stderr,
        )
    return bool(unexplained)


def report_missing_zips(merged: list[Capture], store: Path) -> bool:
    """Print each `200` row whose ZIP is not in the store. True when there was one."""
    missing = [
        capture
        for capture in merged
        if capture.status == "200"
        and not (store / stored_file(capture.artifact, capture.digest)).is_file()
    ]
    for capture in missing:
        print(
            f"no stored ZIP for {capture.url} "
            f"({stored_file(capture.artifact, capture.digest)}); the row keeps its credit",
            file=sys.stderr,
        )
    return bool(missing)


def capture_evidence(now: datetime, paths: EvidencePaths = EVIDENCE) -> int:
    """Network step, full run: BLS's files and every Internet Archive copy.

    Exits 1 when a copy holds unexplained members or a `200` row still has no stored ZIP, after
    writing everything else.
    """
    vintages = fetch_vintages(now)
    observed = [*fetch_bls_copies(now), *collect_archive_copies()]
    merged = record_run(observed, vintages, paths)
    unexplained = report_unexplained(merged)
    missing = report_missing_zips(merged, paths.store)
    return 1 if unexplained or missing else 0


def fetch_current(now: datetime, paths: EvidencePaths = EVIDENCE) -> int:
    """Network step, light run: BLS's current files only, with no Internet Archive request."""
    vintages = fetch_vintages(now)
    merged = record_run(fetch_bls_copies(now), vintages, paths)
    return 1 if report_unexplained(merged) else 0


class AdoptRefusal(Exception):
    """A local ZIP that `adopt` will not store. The message says why."""


def _adopt_rows(
    payload: bytes, committed: list[Capture], store: Path, held: set[str]
) -> tuple[str, list[Capture]]:
    """The artifact of a ZIP `adopt` may store and the committed rows with its rows updated.

    `held` holds the stored files of the ZIPs the run has accepted so far, and gains this one's.
    Raises AdoptRefusal when no row, or rows under two artifacts, record the ZIP's digest, when
    the file is already held, or when a row's fingerprints or member date differ from the ZIP's.
    """
    digest = payload_digest(payload)
    matching = [
        index
        for index, capture in enumerate(committed)
        if capture.digest == digest and capture.status in ("200", "payload_lost")
    ]
    if not matching:
        raise AdoptRefusal(
            f"no committed row with status 200 or payload_lost records digest {digest}"
        )
    artifacts = sorted({committed[index].artifact for index in matching})
    if len(artifacts) > 1:
        raise AdoptRefusal(
            f"rows for digest {digest} name two artifacts: {', '.join(artifacts)}"
        )
    file = stored_file(artifacts[0], digest)
    if file in held:
        raise AdoptRefusal(f"this run already holds {file}")
    if (store / file).is_file():
        raise AdoptRefusal(f"the store already holds {file}")
    computed = {**inspect_zip(payload), "newest_member": newest_member(payload)}
    rows = list(committed)
    for index in matching:
        row = committed[index]
        adopted = replace(
            row,
            status="200",
            newest_member=row.newest_member or computed["newest_member"],
        )
        for name in _OPENED_FIELDS:
            if getattr(adopted, name) != computed[name]:
                raise AdoptRefusal(
                    f"the row for {row.url} at {row.timestamp.strftime(_TIMESTAMP)} records "
                    f"{name} {getattr(row, name)!r}, but the ZIP's is {computed[name]!r}"
                )
        rows[index] = adopted
    held.add(file)
    return artifacts[0], rows


def adopt_payloads(zips: list[Path], paths: EvidencePaths = EVIDENCE) -> int:
    """Offline step: store each local ZIP whose digest a committed row records.

    A ZIP's rows are the committed rows with its digest and status `200` or `payload_lost`.
    Their fingerprints and member date must equal the ZIP's. A row with no member date gets the
    ZIP's, and a `payload_lost` row becomes `200`. Every other row stays as it was, and no row is
    added. Every ZIP is checked before any is staged, so one refusal prints its reason to stderr,
    returns 1 and writes nothing. Otherwise the ZIPs and the updated captures move into place
    together. `es-vintages.csv` is never written.
    """
    committed = (
        [capture_from_row(row) for row in read_csv(paths.captures)]
        if paths.captures.is_file()
        else []
    )
    accepted: list[tuple[Path, str, bytes]] = []
    held: set[str] = set()
    for zip_path in zips:
        try:
            payload = zip_path.read_bytes()
            artifact, committed = _adopt_rows(payload, committed, paths.store, held)
        except (OSError, AdoptRefusal) as refusal:
            print(f"{zip_path}: {refusal}", file=sys.stderr)
            return 1
        accepted.append((zip_path, artifact, payload))
    with staging(paths.store) as run:
        stored = [
            (zip_path, stored_file(artifact, run.put(artifact, payload)))
            for zip_path, artifact, payload in accepted
        ]
        write_csv(
            run.path_for(paths.captures),
            [capture_row(capture) for capture in committed],
            CAPTURE_COLUMNS,
        )
        run.promote()
    for zip_path, file in stored:
        print(f"{zip_path}: stored {file}")
    return 0


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "captures", help="network, full run: BLS and every Archive copy"
    )
    commands.add_parser("current", help="network, light run: BLS's current files only")
    commands.add_parser(
        "inventory", help="offline: derive the inventory from the evidence"
    )
    adopt = commands.add_parser(
        "adopt", help="offline: store a local copy of a recorded ZIP"
    )
    adopt.add_argument("zips", nargs="+", type=Path, metavar="ZIP")
    arguments = parser.parse_args(argv)
    command = arguments.command
    if command == "inventory":
        return derive_inventory()
    if command == "adopt":
        return adopt_payloads(arguments.zips, EVIDENCE)
    now = datetime.now(UTC).replace(microsecond=0)
    if command == "captures":
        return capture_evidence(now)
    return fetch_current(now)


if __name__ == "__main__":
    raise SystemExit(main())
