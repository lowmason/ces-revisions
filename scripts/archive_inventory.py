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

import csv
import os
import re
import urllib.request
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
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


def select_vintages(vintages: list[Vintage], *, as_of: date) -> list[Vintage]:
    """Drop releases after `as_of`: the index links scheduled releases before they happen."""
    return [vintage for vintage in vintages if vintage.release_date <= as_of]


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
