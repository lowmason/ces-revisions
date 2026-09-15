"""CES Table 1 active-report coverage proxy and published RSE regimes."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from functools import cache
from pathlib import Path

import polars as pl

from ces_revisions.annual import html_tables
from ces_revisions.annual.raw import RAW_DIR, cell_key, parse_number, raw_frame

REGIMES = "manual/sample-source-regimes.csv"
SECTORS = ("00", "10", "20", "30", "40", "50", "55", "60", "65", "70", "80", "90")
COVERAGE_NOTE = (
    "CES Table 1 counts active sample reports; its employee share is a proxy for "
    "usable linked coverage, not the matched sample used by the estimator."
)
_LEGACY_SECTOR_BY_TITLE = {
    "total": "00",
    "total nonfarm": "00",
    "natural resources and mining": "10",
    "mining and logging": "10",
    "construction": "20",
    "manufacturing": "30",
    "trade transportation and utilities": "40",
    "information": "50",
    "financial activities": "55",
    "professional and business services": "60",
    "education and health services": "65",
    "leisure and hospitality": "70",
    "other services": "80",
    "government": "90",
}
SAMPLE_SCHEMA = {
    "benchmark_year": pl.Int32,
    "reference_month": pl.Date,
    "frequency": pl.String,
    "sector": pl.String,
    "industry_title": pl.String,
    "benchmark_employment_thousands": pl.Float64,
    "active_ui_accounts": pl.Int64,
    "active_establishments": pl.Int64,
    "sample_employees_thousands": pl.Float64,
    "coverage_percent": pl.Float64,
    "coverage_measure": pl.String,
    "coverage_status": pl.String,
    "coverage_note": pl.String,
    "rse_value": pl.Float64,
    "rse_measure": pl.String,
    "rse_closing": pl.String,
    "rse_unit": pl.String,
    "rse_status": pl.String,
    "rse_definition_break": pl.Boolean,
    "missing_reason": pl.String,
    "published_series_note": pl.String,
    "publication_date": pl.Date,
    "observable_at": pl.Datetime("us", "UTC"),
    "source_files": pl.List(pl.String),
    "source_locators": pl.List(pl.String),
    "source_keys": pl.List(pl.String),
    "transformation": pl.String,
}


def _source_rows(raw_dir: Path) -> pl.DataFrame:
    return pl.read_csv(raw_dir / REGIMES, infer_schema_length=0)


def _read_page(path: Path) -> str:
    payload = path.read_bytes()
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload.decode("windows-1252")


@cache
def _table(raw_dir: Path, file: str, caption: str) -> html_tables.HtmlTable:
    return html_tables.find_table(_read_page(raw_dir / file), caption)


def _normalized_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _sector_rows(
    table: html_tables.HtmlTable, sector: str
) -> list[tuple[int, tuple[str, ...]]]:
    matches = []
    for index, row in enumerate(table.rows):
        if not row:
            continue
        code = re.match(r"^(\d{2})-(\d{6})(?:\s|$)", row[0])
        row_sector = (
            code.group(1)
            if code and code.group(2) == "000000"
            else _LEGACY_SECTOR_BY_TITLE.get(_normalized_title(row[0]))
        )
        if row_sector == sector:
            matches.append((index, row))
    return matches


def _sector_row(
    table: html_tables.HtmlTable, sector: str
) -> tuple[int, tuple[str, ...]]:
    matches = _sector_rows(table, sector)
    if len(matches) != 1:
        raise ValueError(
            f"expected one sector {sector} row in {table.caption!r}, found {len(matches)}"
        )
    return matches[0]


def _table_cells(
    raw_dir: Path, file: str, caption: str, table_key: str
) -> pl.DataFrame:
    return html_tables.table_raw_cells(
        _table(raw_dir, file, caption),
        source="bls",
        file=file,
        table_key=table_key,
    )


def _regime_raw_values(raw_dir: Path) -> pl.DataFrame:
    frame = _source_rows(raw_dir)
    records = [
        {
            "source": "manual_transcription",
            "file": REGIMES,
            "table_key": "sample_source_regimes",
            "row_key": str(row_number),
            "column_key": column,
            "text": row[column] or "",
        }
        for row_number, row in enumerate(frame.iter_rows(named=True), start=2)
        for column in frame.columns
    ]
    return raw_frame(records)


def raw_values(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    parts = [_regime_raw_values(raw_dir)]
    seen = set()
    for row in _source_rows(raw_dir).iter_rows(named=True):
        year = int(row["benchmark_year"])
        for kind in ("coverage", "rse"):
            if row[f"{kind}_status"] != "published":
                continue
            file = row[f"{kind}_file"]
            caption = row[f"{kind}_caption_regex"]
            key = (file, caption, kind, year)
            if key in seen:
                continue
            seen.add(key)
            parts.append(_table_cells(raw_dir, file, caption, f"sample_{kind}_{year}"))
    return pl.concat(parts).sort("cell_key")


def _publication(calendar: pl.DataFrame, year: int) -> dict:
    rows = calendar.filter(pl.col("publication_id") == f"benchmark_final_{year}")
    if rows.height != 1:
        raise ValueError(f"expected one final benchmark publication for {year}")
    return rows.row(0, named=True)


def _published_at(
    row: dict, calendar: pl.DataFrame
) -> tuple[date | None, datetime, list[str]]:
    if row["coverage_status"] == "archive_gap":
        instant = datetime.fromisoformat(row["evidence_observed_at"]).astimezone(UTC)
        return None, instant, []
    publication = _publication(calendar, int(row["benchmark_year"]))
    return (
        publication["publication_date"],
        publication["observable_at"],
        publication["source_keys"],
    )


def _required_integer(text: str, label: str) -> int:
    value = parse_number(text)
    if value is None or not value.is_integer():
        raise ValueError(
            f"{label} must be a published whole-number count, got {text!r}"
        )
    return int(value)


def _coverage(
    row: dict, sector: str, raw_dir: Path
) -> tuple[dict[str, object], list[str], str]:
    if row["coverage_status"] != "published":
        return (
            {
                "industry_title": None,
                "benchmark_employment_thousands": None,
                "active_ui_accounts": None,
                "active_establishments": None,
                "sample_employees_thousands": None,
                "coverage_percent": None,
            },
            [],
            "explicit_archive_gap",
        )
    table = _table(raw_dir, row["coverage_file"], row["coverage_caption_regex"])
    row_index, cells = _sector_row(table, sector)
    has_code = re.match(r"^\d{2}-\d{6}", cells[0]) is not None
    expected_columns = 7 if has_code else 6
    if len(cells) != expected_columns:
        raise ValueError(
            f"Table 1 for {row['benchmark_year']} has {len(cells)} columns, "
            f"expected {expected_columns}"
        )
    offset = 1 if has_code else 0
    values = {
        "industry_title": cells[offset],
        "benchmark_employment_thousands": parse_number(cells[offset + 1]),
        "active_ui_accounts": _required_integer(
            cells[offset + 2], "active UI accounts"
        ),
        "active_establishments": _required_integer(
            cells[offset + 3], "active establishments"
        ),
        "sample_employees_thousands": parse_number(cells[offset + 4]),
        "coverage_percent": parse_number(cells[offset + 5]),
    }
    benchmark = values["benchmark_employment_thousands"]
    sample = values["sample_employees_thousands"]
    percent = values["coverage_percent"]
    if benchmark is None or sample is None or percent is None:
        raise ValueError(f"Table 1 for {row['benchmark_year']} has a missing value")
    transformation = "parse_published_number"
    if round(sample / benchmark * 100) != percent:
        converted = sample / 1_000
        if round(converted / benchmark * 100) != percent:
            raise ValueError(
                f"Table 1 coverage identity fails for {row['benchmark_year']} "
                f"sector {sector}"
            )
        values["sample_employees_thousands"] = converted
        transformation = "sample_jobs_to_thousands"
    keys = [
        cell_key(
            row["coverage_file"],
            f"sample_coverage_{row['benchmark_year']}",
            str(row_index),
            str(column),
        )
        for column in range(offset, offset + 6)
    ]
    return values, keys, transformation


def _rse(row: dict, sector: str, raw_dir: Path) -> tuple[float | None, list[str]]:
    if row["rse_status"] != "published":
        return None, []
    if row["rse_measure"] == "level" and sector == "90":
        return None, []
    table = _table(raw_dir, row["rse_file"], row["rse_caption_regex"])
    matches = _sector_rows(table, sector)
    if not matches:
        return None, []
    if len(matches) != 1:
        raise ValueError(
            f"expected at most one sector {sector} row in {table.caption!r}, "
            f"found {len(matches)}"
        )
    row_index, cells = matches[0]
    column = int(row["rse_column_index"])
    if column >= len(cells):
        raise ValueError(
            f"RSE column {column} absent for {row['benchmark_year']} sector {sector}"
        )
    return parse_number(cells[column]), [
        cell_key(
            row["rse_file"],
            f"sample_rse_{row['benchmark_year']}",
            str(row_index),
            str(column),
        )
    ]


def _break_years(source: pl.DataFrame) -> set[int]:
    years = []
    previous = None
    for row in source.sort("benchmark_year").iter_rows(named=True):
        if row["rse_status"] != "published":
            continue
        definition = (row["rse_measure"], row["rse_closing"], row["rse_unit"])
        if previous is not None and definition != previous:
            years.append(int(row["benchmark_year"]))
        previous = definition
    return set(years)


def build_sample_panel(calendar: pl.DataFrame, raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    source = _source_rows(raw_dir)
    breaks = _break_years(source)
    rows = []
    for row_number, source_row in enumerate(source.iter_rows(named=True), start=2):
        year = int(source_row["benchmark_year"])
        publication_date, observable_at, publication_keys = _published_at(
            source_row, calendar
        )
        for sector in SECTORS:
            coverage, coverage_keys, coverage_transformation = _coverage(
                source_row, sector, raw_dir
            )
            rse_value, rse_keys = _rse(source_row, sector, raw_dir)
            regime_keys = [
                cell_key(
                    REGIMES,
                    "sample_source_regimes",
                    str(row_number),
                    column,
                )
                for column in (
                    "coverage_status",
                    "rse_status",
                    "rse_measure",
                    "rse_closing",
                    "rse_unit",
                    "published_series_note",
                    "published_series_source_files",
                    "published_series_source_locators",
                    "evidence_observed_at",
                )
            ]
            missing_reason = None
            if source_row["coverage_status"] == "archive_gap":
                missing_reason = "March 2012 Table 1 was not found in the archive"
            rows.append(
                {
                    "benchmark_year": year,
                    "reference_month": date(year, 3, 1),
                    "frequency": "annual",
                    "sector": sector,
                    **coverage,
                    "coverage_measure": "table_1_active_report_employee_share_proxy",
                    "coverage_status": source_row["coverage_status"],
                    "coverage_note": COVERAGE_NOTE,
                    "rse_value": rse_value,
                    "rse_measure": source_row["rse_measure"] or None,
                    "rse_closing": source_row["rse_closing"] or None,
                    "rse_unit": source_row["rse_unit"] or None,
                    "rse_status": source_row["rse_status"],
                    "rse_definition_break": year in breaks,
                    "missing_reason": missing_reason,
                    "published_series_note": source_row["published_series_note"]
                    or None,
                    "publication_date": publication_date,
                    "observable_at": observable_at,
                    "source_files": sorted(
                        {
                            file
                            for file in (
                                source_row["coverage_file"],
                                source_row["rse_file"],
                                REGIMES,
                                *(
                                    source_row["published_series_source_files"].split(
                                        ";"
                                    )
                                    if source_row["published_series_source_files"]
                                    else []
                                ),
                            )
                            if file
                        }
                    ),
                    "source_locators": [
                        locator
                        for locator in (
                            source_row["coverage_caption_regex"],
                            source_row["rse_caption_regex"],
                            *(
                                source_row["published_series_source_locators"].split(
                                    ";"
                                )
                                if source_row["published_series_source_locators"]
                                else []
                            ),
                        )
                        if locator
                    ],
                    "source_keys": sorted(
                        set(coverage_keys + rse_keys + regime_keys + publication_keys)
                    ),
                    "transformation": coverage_transformation,
                }
            )
    frame = pl.DataFrame(rows, schema=SAMPLE_SCHEMA).sort("benchmark_year", "sector")
    if frame.select("benchmark_year", "sector").n_unique() != frame.height:
        raise ValueError("duplicate annual sample rows")
    return frame
