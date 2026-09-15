"""Publication dates for Stage 4 values."""

from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from ces_revisions.annual.raw import PUBLICATION_DATES, RAW_DIR, cell_key, raw_frame

EASTERN = ZoneInfo("America/New_York")

PUBLICATION_SCHEMA = {
    "publication_id": pl.String,
    "publication_kind": pl.String,
    "benchmark_year": pl.Int32,
    "qcew_year": pl.Int32,
    "qcew_quarter": pl.Int8,
    "release_order": pl.Int8,
    "publication_date": pl.Date,
    "observable_at": pl.Datetime("us", "UTC"),
    "source_file": pl.String,
    "source_locator": pl.String,
    "source_keys": pl.List(pl.String),
}


def _utc_observable(day: date, clock: time) -> datetime:
    return datetime.combine(day, clock, EASTERN).astimezone(UTC)


def final_carrier_month(benchmark_year: int) -> date:
    if benchmark_year <= 2002:
        return date(benchmark_year + 1, 5, 1)
    return date(benchmark_year + 1, 1, 1)


def catalog_publications(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    frame = pl.read_csv(
        raw_dir / PUBLICATION_DATES,
        schema_overrides={
            "benchmark_year": pl.Int32,
            "qcew_year": pl.Int32,
            "qcew_quarter": pl.Int8,
            "release_order": pl.Int8,
            "publication_date": pl.Date,
        },
        try_parse_dates=True,
    )
    rows = []
    for row_number, row in enumerate(frame.iter_rows(named=True), start=2):
        clock_value = row["release_time_et"]
        clock = (
            clock_value
            if isinstance(clock_value, time)
            else time.fromisoformat(clock_value)
        )
        rows.append(
            {
                "publication_id": row["publication_id"],
                "publication_kind": row["publication_kind"],
                "benchmark_year": row["benchmark_year"],
                "qcew_year": row["qcew_year"],
                "qcew_quarter": row["qcew_quarter"],
                "release_order": row["release_order"],
                "publication_date": row["publication_date"],
                "observable_at": _utc_observable(row["publication_date"], clock),
                "source_file": row["source_file"],
                "source_locator": row["source_locator"],
                "source_keys": [
                    cell_key(
                        PUBLICATION_DATES,
                        "publication_dates",
                        str(row_number),
                        "publication_date",
                    )
                ],
            }
        )
    return pl.DataFrame(rows, schema=PUBLICATION_SCHEMA)


def final_benchmark_publications(release_index: pl.DataFrame) -> pl.DataFrame:
    by_month = {
        row["reference_month"]: row for row in release_index.iter_rows(named=True)
    }
    rows = []
    for year in range(1979, 2026):
        carrier = final_carrier_month(year)
        release = by_month.get(carrier)
        if release is None:
            raise ValueError(f"no Stage 3 release-index row for benchmark year {year}")
        rows.append(
            {
                "publication_id": f"benchmark_final_{year}",
                "publication_kind": "benchmark_final",
                "benchmark_year": year,
                "qcew_year": None,
                "qcew_quarter": None,
                "release_order": None,
                "publication_date": release["published_date"],
                "observable_at": release["observable_at"],
                "source_file": "data/panel/release_index.parquet",
                "source_locator": f"reference_month={carrier:%Y-%m}",
                "source_keys": [f"stage3::release_index::{carrier:%Y-%m}"],
            }
        )
    return pl.DataFrame(rows, schema=PUBLICATION_SCHEMA)


def build_publication_calendar(
    release_index: pl.DataFrame, raw_dir: Path = RAW_DIR
) -> pl.DataFrame:
    frame = pl.concat(
        [final_benchmark_publications(release_index), catalog_publications(raw_dir)]
    ).sort("publication_date", "publication_id")
    if frame["publication_id"].n_unique() != frame.height:
        raise ValueError("duplicate publication_id")
    preliminary = frame.filter(pl.col("publication_kind") == "benchmark_preliminary")
    if preliminary["benchmark_year"].to_list() != list(range(2000, 2027)):
        raise ValueError("preliminary benchmark publication calendar is incomplete")
    return frame


def publication_raw_values(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    frame = pl.read_csv(raw_dir / PUBLICATION_DATES, infer_schema_length=0)
    records = []
    for row_number, row in enumerate(frame.iter_rows(named=True), start=2):
        for column in frame.columns:
            records.append(
                {
                    "source": "manual_transcription",
                    "file": PUBLICATION_DATES,
                    "table_key": "publication_dates",
                    "row_key": str(row_number),
                    "column_key": column,
                    "text": row[column] or "",
                }
            )
    return raw_frame(records)
