"""The release-date index: when each Employment Situation release was scheduled and published.

Release dates for reference months through December 2000 come from BLS's table of historical
release dates (bls/histreleasedates.txt, the text of histreleasedates.pdf), and later ones from
the Employment Situation archive index (bls/empsit-releases.csv). A hand-keyed table of the
eight releases that left their schedule, each row cited, adds the scheduled dates, the canceled
October 2025 release, and the one release not made at 8:30 a.m. Eastern. BLS publishes no
closing dates (Req 3), so the index holds none.
"""

import csv
import re
from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import fastexcel
import polars as pl

from ces_revisions.vintages.months import MONTH_NAMES, month_range
from ces_revisions.vintages.raw import (
    EMPSIT_RELEASES,
    HISTORICAL_RELEASE_DATES,
    RAW_DIR,
    RESCHEDULES,
    RTDSM_RELEASE_DATES,
)

FIRST_REFERENCE_MONTH = date(1979, 1, 1)
LAST_HISTORICAL_MONTH = date(2000, 12, 1)
# The May 2003 publication vintage, the first release in the vintage files.
FIRST_VINTAGE_FILE_MONTH = date(2003, 5, 1)
# That release carried the March 2002 benchmark, the last to arrive with May estimates:
# https://www.bls.gov/news.release/archives/empsit_06062003.pdf
LAST_MAY_BENCHMARK = date(2003, 5, 1)
# From the March 2003 benchmark on, benchmark revisions arrive with January estimates: every
# January release from 2004 in the vintage files revises the 21 months through the prior
# December, and no release from May to December 2003 revises more than two.
FIRST_JANUARY_BENCHMARK = date(2004, 1, 1)
EASTERN = ZoneInfo("America/New_York")
EMBARGO = time(8, 30)
PUBLICATION = "employment_situation"

_SECTION_START = "Release dates for national employment and unemployment estimates"
_SECTION_END = "Release dates for the Consumer Price Index"
_YEAR_ROW = re.compile(r"^\s*(\d{4})\s+(\S.*)$")
_DATE_TOKEN = re.compile(
    r"--|(" + "|".join(MONTH_NAMES) + r") (\d{1,2})(?:, (\d{4})\d?)?"
)

INDEX_SCHEMA = {
    "reference_month": pl.Date,
    "publication": pl.String,
    "scheduled_date": pl.Date,
    "release_date": pl.Date,
    "published_date": pl.Date,
    "observable_at": pl.Datetime("us", "UTC"),
    "status": pl.String,
    "off_schedule": pl.Boolean,
    "benchmark_release": pl.Boolean,
    "date_source": pl.String,
}


def parse_historical_release_dates(text: str) -> dict[date, date]:
    """Reference month to release date from the Employment Situation table of the BLS PDF.

    Each row holds a year and twelve cells, one per reference month, such as "February 2" or,
    in December's column, "January 11, 1980"; a dash marks an unknown date, and a footnote
    digit can trail a year, as in "January 19, 19961".
    """
    start = text.index(_SECTION_START)
    section = text[start : text.index(_SECTION_END, start)]
    dates = {}
    for line in section.splitlines():
        row = _YEAR_ROW.match(line)
        if row is None:
            continue
        tokens = list(_DATE_TOKEN.finditer(row.group(2)))
        if len(tokens) != 12:
            raise ValueError(f"expected 12 release dates in {line!r}")
        year = int(row.group(1))
        for month, token in enumerate(tokens, start=1):
            if token.group(0) == "--":
                continue
            released = date(
                int(token.group(3) or year),
                MONTH_NAMES.index(token.group(1)) + 1,
                int(token.group(2)),
            )
            dates[date(year, month, 1)] = released
    return dates


def parse_rtdsm_release_dates(rows: list[tuple[str | None, ...]]) -> dict[date, date]:
    """Reference month to release date from the Philadelphia Fed's copy of BLS's date file.

    Year rows hold twelve cells as ISO dates, except the two releases the file marks with
    asterisks, which read "1/19/96*" and "11/5/98**".
    """
    dates = {}
    for row in rows:
        head = (row[0] or "").strip()
        if not re.fullmatch(r"\d{4}", head):
            continue
        for month, cell in enumerate(row[1:13], start=1):
            text = (cell or "").strip().rstrip("*")
            if not text:
                continue
            if "/" in text:
                month_part, day, year = (int(part) for part in text.split("/"))
                released = date(1900 + year, month_part, day)
            else:
                released = date.fromisoformat(text[:10])
            dates[date(int(head), month, 1)] = released
    return dates


def read_historical_release_dates(raw_dir: Path = RAW_DIR) -> dict[date, date]:
    text = (raw_dir / HISTORICAL_RELEASE_DATES).read_text(encoding="utf-8")
    return parse_historical_release_dates(text)


def read_rtdsm_release_dates(raw_dir: Path = RAW_DIR) -> dict[date, date]:
    sheet = fastexcel.read_excel(raw_dir / RTDSM_RELEASE_DATES).load_sheet(
        "Release dates", header_row=None, dtypes="string"
    )
    return parse_rtdsm_release_dates(sheet.to_polars().rows())


def read_release_list(path: Path) -> dict[date, date]:
    """Reference month to release date from a CSV in the format of es-vintages.csv."""
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            date.fromisoformat(f"{row['reference_month']}-01"): date.fromisoformat(
                row["release_date"]
            )
            for row in csv.DictReader(handle)
        }


def read_reschedules(raw_dir: Path = RAW_DIR) -> dict[date, dict[str, str]]:
    with (raw_dir / RESCHEDULES).open(newline="", encoding="utf-8") as handle:
        return {
            date.fromisoformat(f"{row['reference_month']}-01"): row
            for row in csv.DictReader(handle)
        }


def _optional_date(text: str) -> date | None:
    return date.fromisoformat(text) if text else None


def build_release_index(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    """One row per reference month from January 1979 through the latest archived release.

    A release the hand-keyed table does not list is taken to have kept its schedule: its
    scheduled date is its release date, and it came out at 8:30 a.m. Eastern.
    """
    historical = read_historical_release_dates(raw_dir)
    archive = read_release_list(raw_dir / EMPSIT_RELEASES)
    moved = read_reschedules(raw_dir)
    rows, times = {}, {}
    for month in month_range(FIRST_REFERENCE_MONTH, max(archive)):
        if month <= LAST_HISTORICAL_MONTH:
            source, released = "histreleasedates", historical.get(month)
        else:
            source, released = "empsit_releases", archive.get(month)
        scheduled, status, times[month] = released, "released", EMBARGO
        change = moved.get(month)
        if change is not None:
            scheduled, status = (
                _optional_date(change["scheduled_date"]),
                change["status"],
            )
            if status == "canceled":
                released = None
            elif released != date.fromisoformat(change["release_date"]):
                raise ValueError(
                    f"{month:%Y-%m}: {source} gives {released}, "
                    f"{RESCHEDULES} {change['release_date']}"
                )
            else:
                times[month] = time.fromisoformat(change["release_time_et"])
        elif released is None:
            raise ValueError(f"no release date for {month:%Y-%m}")
        rows[month] = {
            "reference_month": month,
            "publication": PUBLICATION,
            "scheduled_date": scheduled,
            "release_date": released,
            "status": status,
            "off_schedule": change is not None,
            "benchmark_release": None
            if month < FIRST_VINTAGE_FILE_MONTH
            else month == LAST_MAY_BENCHMARK
            or (month.month == 1 and month >= FIRST_JANUARY_BENCHMARK),
            "date_source": source,
        }
    for month, row in rows.items():
        carrier = month
        if row["status"] == "canceled":
            published_with = moved[month]["published_with"]
            carrier = date.fromisoformat(f"{published_with}-01")
        row["published_date"] = rows[carrier]["release_date"]
        row["observable_at"] = datetime.combine(
            row["published_date"], times[carrier], EASTERN
        ).astimezone(UTC)
    return pl.DataFrame(list(rows.values()), schema=INDEX_SCHEMA)
