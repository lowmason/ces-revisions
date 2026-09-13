"""Read docs/ces-revisions-review.md for structural tests: headings, sections, tables, and links."""

import re
from pathlib import Path
from urllib.parse import urlsplit

REVIEW_PATH = Path(__file__).resolve().parents[1] / "docs" / "ces-revisions-review.md"

EVIDENCE_LABELS = ("documented", "supported", "asserted", "contested", "not found")

# Req 20's primary sources are BLS, CRS, GAO, DOL/OMB, and the journal. The inventory also cites
# the custodians of other primary records: Congress and the Government Publishing Office, the
# National Archives, OPM, BEA, the Census Bureau, NBER's business-cycle dates, NOAA, and the
# Federal Reserve's real-time data sets. DOI links resolve journal articles.
PRIMARY_HOSTS = (
    "bls.gov",
    "dol.gov",
    "doleta.gov",
    "gao.gov",
    "congress.gov",
    "govinfo.gov",
    "house.gov",
    "senate.gov",
    "whitehouse.gov",
    "archives.gov",
    "opm.gov",
    "bea.gov",
    "census.gov",
    "nber.org",
    "noaa.gov",
    "federalreserve.gov",
    "philadelphiafed.org",
    "stlouisfed.org",
    "doi.org",
)
ARCHIVE_HOST = "web.archive.org"

_FENCE = re.compile(r"(?ms)^```.*?^```[ \t]*$")
_HEADING = re.compile(r"^(#{1,6}) (.+?)\s*$")
_LINK = re.compile(r"\]\((https?://[^)\s]+)\)")
_ARCHIVED = re.compile(r"^https?://web\.archive\.org/web/[^/]+/(https?://.+)$")
_UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")


def read_review() -> str:
    return REVIEW_PATH.read_text(encoding="utf-8")


def outside_fences(text: str) -> str:
    """`text` with fenced blocks blanked line for line, so they never read as headings or tables."""
    return _FENCE.sub(lambda match: "\n" * match.group(0).count("\n"), text)


def headings(text: str, level: int) -> list[str]:
    return [
        match.group(2)
        for line in outside_fences(text).splitlines()
        if (match := _HEADING.match(line)) and len(match.group(1)) == level
    ]


def section(text: str, title: str, level: int) -> str:
    """The body under heading `title` at `level`, up to the next heading at that level or above."""
    lines = outside_fences(text).splitlines()
    marker = f"{'#' * level} {title}"
    assert marker in lines, f"no heading {marker!r}"
    start = lines.index(marker) + 1
    end = next(
        (
            index
            for index in range(start, len(lines))
            if (match := _HEADING.match(lines[index])) and len(match.group(1)) <= level
        ),
        len(lines),
    )
    return "\n".join(lines[start:end])


def tables(text: str) -> list[list[dict[str, str]]]:
    """Every pipe table in `text`, each a list of rows keyed by the header cells."""
    found, block = [], []
    for line in [*outside_fences(text).splitlines(), ""]:
        if line.lstrip().startswith("|"):
            block.append(line)
            continue
        if len(block) >= 2:
            header = _cells(block[0])
            found.append(
                [dict(zip(header, _cells(row), strict=True)) for row in block[2:]]
            )
        block = []
    return found


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in _UNESCAPED_PIPE.split(line.strip().strip("|"))]


def links(text: str) -> list[str]:
    return _LINK.findall(text)


def is_primary(url: str) -> bool:
    """Whether `url` is on a primary-source host, or is an Internet Archive copy of such a page."""
    host = urlsplit(url).hostname or ""
    if host == ARCHIVE_HOST:
        archived = _ARCHIVED.match(url)
        return archived is not None and is_primary(archived.group(1))
    return any(
        host == primary or host.endswith(f".{primary}") for primary in PRIMARY_HOSTS
    )


def github_anchor(heading: str) -> str:
    """The fragment GitHub gives a heading: lowercase, punctuation dropped, spaces to hyphens."""
    return re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
