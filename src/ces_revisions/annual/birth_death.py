"""Audited CES net birth-death schedules and forecast-versus-realized values."""

import re
from datetime import date
from pathlib import Path

import polars as pl

from ces_revisions.annual import html_tables
from ces_revisions.annual.raw import RAW_DIR, cell_key, parse_number, raw_frame

HISTORY = "bls/cesbd-history.htm"
CURRENT = "bls/cesbd-current.htm"
ANNUAL_CELLS = "manual/birth-death-annual-cells.csv"
RULES = "manual/birth-death-rules.csv"
MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}
SECTOR_BY_TITLE = {
    "mining and logging": "10",
    "natural resources and mining": "10",
    "construction": "20",
    "manufacturing": "30",
    "trade, transportation, and utilities": "40",
    "information": "50",
    "financial activities": "55",
    "professional and business services": "60",
    "private education and health services": "65",
    "education and health services": "65",
    "leisure and hospitality": "70",
    "other services": "80",
    "total nonfarm birth-death forecast": "00",
    "total nonfarm net birth-death forecast": "00",
    "total private net birth-death forecast": "00",
}
BIRTH_DEATH_SCHEMA = {
    "series_kind": pl.String,
    "benchmark_year": pl.Int32,
    "reference_month": pl.Date,
    "period_start": pl.Date,
    "period_end": pl.Date,
    "frequency": pl.String,
    "schedule_kind": pl.String,
    "sector": pl.String,
    "seasonal_status": pl.String,
    "value_kind": pl.String,
    "value_thousands": pl.Float64,
    "unit": pl.String,
    "publication_date": pl.Date,
    "observable_at": pl.Datetime("us", "UTC"),
    "source_file": pl.String,
    "source_locator": pl.String,
    "source_keys": pl.List(pl.String),
    "transformation": pl.String,
}


def _normalized(text: str) -> str:
    text = re.sub(r"\([^)]*\)|\[[^]]*\]", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _schedule_kind(header: tuple[str, ...]) -> str | None:
    label = _normalized(header[0]) if header else ""
    if label.startswith("preliminary"):
        return "preliminary"
    if label.startswith("post-benchmark"):
        return "post_benchmark"
    return None


def _table_year(table: html_tables.HtmlTable) -> int | None:
    match = re.search(r"\b(19|20)\d{2}\b", table.caption)
    return int(match.group()) if match else None


def _month_label(text: str) -> str:
    return _normalized(text).title()


def _selected_tables(raw_dir: Path):
    for file in (HISTORY, CURRENT):
        page = (raw_dir / file).read_text(encoding="utf-8")
        for table_number, table in enumerate(html_tables.parse_tables(page)):
            year = _table_year(table)
            header = next(
                (
                    row
                    for row in table.rows
                    if any(_month_label(cell) in MONTHS for cell in row)
                ),
                None,
            )
            if year is None or header is None:
                continue
            if file == CURRENT and "Supersector" not in header:
                continue
            if file == CURRENT and year == 2026:
                kind = "preliminary"
            elif file == CURRENT and year == 2025:
                kind = "post_benchmark"
            else:
                kind = _schedule_kind(header)
            if kind is None:
                continue
            historical = file == HISTORY and (
                (kind == "preliminary" and 2004 <= year <= 2025)
                or (kind == "post_benchmark" and 2003 <= year <= 2025)
            )
            current = file == CURRENT and (
                (year == 2026 and kind == "preliminary")
                or (year == 2025 and kind == "post_benchmark")
            )
            if historical or current:
                yield file, table_number, year, kind, table, header


def _sector(text: str) -> str | None:
    return SECTOR_BY_TITLE.get(_normalized(text))


def _monthly_cells(raw_dir: Path):
    for file, table_number, year, kind, table, header in _selected_tables(raw_dir):
        header_index = table.rows.index(header)
        month_columns = {
            column: MONTHS[_month_label(label)]
            for column, label in enumerate(header)
            if _month_label(label) in MONTHS
        }
        table_key = f"birth_death_{year}_{kind}"
        for row_index, row in enumerate(
            table.rows[header_index + 1 :], start=header_index + 1
        ):
            title_column = 1 if row and re.fullmatch(r"\d{2}-\d{6}.*", row[0]) else 0
            sector = _sector(row[title_column]) if len(row) > title_column else None
            if sector is None:
                continue
            for column, month in month_columns.items():
                if column >= len(row) or parse_number(row[column]) is None:
                    continue
                yield {
                    "file": file,
                    "table_number": table_number,
                    "table_key": table_key,
                    "row_index": row_index,
                    "column": column,
                    "year": year,
                    "schedule_kind": kind,
                    "reference_month": date(year, month, 1),
                    "sector": sector,
                    "text": row[column],
                }


def _csv_raw_values(path: Path, file: str, table_key: str) -> pl.DataFrame:
    frame = pl.read_csv(path, infer_schema_length=0)
    records = [
        {
            "source": "manual_transcription",
            "file": file,
            "table_key": table_key,
            "row_key": str(row_number),
            "column_key": column,
            "text": row[column] or "",
        }
        for row_number, row in enumerate(frame.iter_rows(named=True), start=2)
        for column in frame.columns
    ]
    return raw_frame(records)


def raw_values(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    schedule_records = [
        {
            "source": "bls",
            "file": cell["file"],
            "table_key": cell["table_key"],
            "row_key": str(cell["row_index"]),
            "column_key": str(cell["column"]),
            "text": cell["text"],
        }
        for cell in _monthly_cells(raw_dir)
    ]
    return pl.concat(
        [
            raw_frame(schedule_records),
            _csv_raw_values(raw_dir / ANNUAL_CELLS, ANNUAL_CELLS, "birth_death_annual"),
            _csv_raw_values(raw_dir / RULES, RULES, "birth_death_rules"),
        ]
    ).sort("cell_key")


def _one(frame: pl.DataFrame, **filters) -> dict:
    expression = pl.lit(True)
    for column, value in filters.items():
        expression &= pl.col(column) == value
    rows = frame.filter(expression)
    if rows.height != 1:
        raise ValueError(f"expected one row for {filters}, found {rows.height}")
    return rows.row(0, named=True)


def _schedule_rows(
    calendar: pl.DataFrame, release_index: pl.DataFrame, raw_dir: Path
) -> list[dict]:
    rows = []
    for cell in _monthly_cells(raw_dir):
        year = cell["year"]
        month = cell["reference_month"]
        if cell["schedule_kind"] == "preliminary":
            publication = _one(release_index, reference_month=month)
            publication_date = publication["published_date"]
            observable_at = publication["observable_at"]
            benchmark_year = year - 1
            publication_keys = [f"stage3::release_index::{month:%Y-%m}"]
        else:
            publication = _one(
                calendar,
                publication_id=f"benchmark_final_{year}",
            )
            publication_date = publication["publication_date"]
            observable_at = publication["observable_at"]
            benchmark_year = year
            publication_keys = publication["source_keys"]
        rows.append(
            {
                "series_kind": "published_schedule",
                "benchmark_year": benchmark_year,
                "reference_month": month,
                "period_start": month,
                "period_end": month,
                "frequency": "monthly",
                "schedule_kind": cell["schedule_kind"],
                "sector": cell["sector"],
                "seasonal_status": "NSA",
                "value_kind": (
                    "initial_forecast"
                    if cell["schedule_kind"] == "preliminary"
                    else "revised_forecast"
                ),
                "value_thousands": parse_number(cell["text"]),
                "unit": "thousands",
                "publication_date": publication_date,
                "observable_at": observable_at,
                "source_file": cell["file"],
                "source_locator": (
                    f"table {cell['table_number'] + 1}, "
                    f"{cell['schedule_kind']}, {month:%Y-%m}"
                ),
                "source_keys": [
                    cell_key(
                        cell["file"],
                        cell["table_key"],
                        str(cell["row_index"]),
                        str(cell["column"]),
                    )
                ]
                + publication_keys,
                "transformation": "parse_thousands",
            }
        )
    return rows


def _annual_rows(calendar: pl.DataFrame, raw_dir: Path) -> list[dict]:
    source = pl.read_csv(raw_dir / ANNUAL_CELLS, infer_schema_length=0)
    rows = []
    for row_number, row in enumerate(source.iter_rows(named=True), start=2):
        year = int(row["benchmark_year"])
        publication = _one(calendar, publication_id=f"benchmark_final_{year}")
        reference_month = (
            date.fromisoformat(row["reference_month"])
            if row["reference_month"]
            else None
        )
        period_start = reference_month or date(year - 1, 4, 1)
        period_end = reference_month or date(year, 3, 1)
        realized = parse_number(row["realized_text"])
        forecast = parse_number(row["forecast_text"])
        printed_difference = parse_number(row["difference_text"])
        if None in (realized, forecast, printed_difference):
            raise ValueError(
                f"missing annual birth-death value on CSV row {row_number}"
            )
        difference = realized - forecast
        if difference != printed_difference:
            raise ValueError(
                f"birth-death identity fails on CSV row {row_number}: "
                f"{realized} - {forecast} != {printed_difference}"
            )
        for value_kind, value, columns, transformation in (
            ("realized", realized, ("realized_text",), "parse_thousands"),
            ("forecast", forecast, ("forecast_text",), "parse_thousands"),
            (
                "realized_minus_forecast",
                difference,
                ("realized_text", "forecast_text", "difference_text"),
                "realized_minus_forecast",
            ),
        ):
            rows.append(
                {
                    "series_kind": "forecast_vs_realized",
                    "benchmark_year": year,
                    "reference_month": reference_month,
                    "period_start": period_start,
                    "period_end": period_end,
                    "frequency": row["frequency"],
                    "schedule_kind": None,
                    "sector": "00",
                    "seasonal_status": "NSA",
                    "value_kind": value_kind,
                    "value_thousands": value,
                    "unit": "thousands",
                    "publication_date": publication["publication_date"],
                    "observable_at": publication["observable_at"],
                    "source_file": row["source_file"],
                    "source_locator": row["source_locator"],
                    "source_keys": [
                        cell_key(
                            ANNUAL_CELLS,
                            "birth_death_annual",
                            str(row_number),
                            column,
                        )
                        for column in columns
                    ]
                    + publication["source_keys"],
                    "transformation": transformation,
                }
            )
    return rows


def _assert_private_totals(frame: pl.DataFrame) -> None:
    schedules = frame.filter(
        (pl.col("series_kind") == "published_schedule") & (pl.col("sector") != "90")
    )
    totals = schedules.filter(pl.col("sector") == "00").select(
        "schedule_kind", "reference_month", total=pl.col("value_thousands")
    )
    calculated = (
        schedules.filter(pl.col("sector") != "00")
        .group_by("schedule_kind", "reference_month")
        .agg(total_from_sectors=pl.col("value_thousands").sum())
    )
    failures = totals.join(calculated, on=["schedule_kind", "reference_month"]).filter(
        pl.col("total") != pl.col("total_from_sectors")
    )
    if failures.height:
        raise ValueError(f"birth-death sector totals fail for {failures.rows()}")


def _government_rows(frame: pl.DataFrame) -> pl.DataFrame:
    rule_key = cell_key(RULES, "birth_death_rules", "2", "value_text")
    return frame.filter(pl.col("sector") == "00").with_columns(
        sector=pl.lit("90"),
        value_thousands=pl.lit(0.0),
        source_file=pl.lit(RULES),
        source_locator=pl.lit("government-structural-zero"),
        source_keys=pl.concat_list(
            pl.lit([rule_key], dtype=pl.List(pl.String)), pl.col("source_keys")
        ),
        transformation=pl.lit("government_structural_zero"),
    )


def build_birth_death(
    calendar: pl.DataFrame,
    release_index: pl.DataFrame,
    raw_dir: Path = RAW_DIR,
) -> pl.DataFrame:
    rows = _schedule_rows(calendar, release_index, raw_dir)
    rows.extend(_annual_rows(calendar, raw_dir))
    frame = pl.DataFrame(rows, schema=BIRTH_DEATH_SCHEMA)
    _assert_private_totals(frame)
    frame = pl.concat([frame, _government_rows(frame)]).sort(
        "series_kind",
        "benchmark_year",
        "reference_month",
        "frequency",
        "sector",
        "value_kind",
    )
    keys = [
        "series_kind",
        "benchmark_year",
        "reference_month",
        "frequency",
        "sector",
        "value_kind",
    ]
    if frame.select(keys).n_unique() != frame.height:
        raise ValueError("duplicate birth-death rows")
    return frame
