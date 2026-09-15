"""Publication dates for Stage 4 values."""

from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from ces_revisions.annual.raw import (
    FINAL_CARRIER_OVERRIDES,
    PUBLICATION_DATES,
    QCEW_PUBLICATION_DATES,
    RAW_DIR,
    cell_key,
    raw_frame,
)

EASTERN = ZoneInfo("America/New_York")
_FINAL_CARRIER_OVERRIDES = {
    1979: date(1980, 6, 1),
    1980: date(1981, 6, 1),
    1989: date(1990, 8, 1),
}
_PUBLICATION_RAW_TABLES = {
    PUBLICATION_DATES: "publication_dates",
    QCEW_PUBLICATION_DATES: "qcew_publication_dates",
    FINAL_CARRIER_OVERRIDES: "final_carrier_overrides",
}

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
    "transformation": pl.String,
}


def _utc_observable(day: date, clock: time) -> datetime:
    return datetime.combine(day, clock, EASTERN).astimezone(UTC)


def _clock(value: str | time) -> time:
    return value if isinstance(value, time) else time.fromisoformat(value)


def final_carrier_month(benchmark_year: int) -> date:
    """Return the CES reference month whose release incorporated a benchmark."""
    if benchmark_year < 1979:
        raise ValueError("final benchmark carrier years begin in 1979")
    if benchmark_year in _FINAL_CARRIER_OVERRIDES:
        return _FINAL_CARRIER_OVERRIDES[benchmark_year]
    if benchmark_year <= 2002:
        return date(benchmark_year + 1, 5, 1)
    return date(benchmark_year + 1, 1, 1)


def _preliminary_publications(raw_dir: Path) -> pl.DataFrame:
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
    if set(frame["publication_kind"]) != {"benchmark_preliminary"}:
        raise ValueError(
            "publication-dates.csv may contain only benchmark preliminaries"
        )
    rows = []
    for row_number, row in enumerate(frame.iter_rows(named=True), start=2):
        clock = _clock(row["release_time_et"])
        rows.append(
            {
                "publication_id": row["publication_id"],
                "publication_kind": row["publication_kind"],
                "benchmark_year": row["benchmark_year"],
                "qcew_year": None,
                "qcew_quarter": None,
                "release_order": None,
                "publication_date": row["publication_date"],
                "observable_at": _utc_observable(row["publication_date"], clock),
                "source_file": row["source_file"],
                "source_locator": row["source_locator"],
                "source_keys": [
                    cell_key(
                        PUBLICATION_DATES,
                        "publication_dates",
                        str(row_number),
                        column,
                    )
                    for column in ("publication_date", "release_time_et")
                ],
                "transformation": "parse_publication_datetime",
            }
        )
    return pl.DataFrame(rows, schema=PUBLICATION_SCHEMA)


def _qcew_publications(raw_dir: Path) -> pl.DataFrame:
    frame = pl.read_csv(
        raw_dir / QCEW_PUBLICATION_DATES,
        schema_overrides={
            "qcew_year": pl.Int32,
            "qcew_quarter": pl.Int8,
            "news_release_date": pl.Date,
            "full_data_release_date": pl.Date,
        },
        try_parse_dates=True,
    )
    keys = frame.select("qcew_year", "qcew_quarter")
    if keys.n_unique() != frame.height:
        raise ValueError("duplicate QCEW publication quarter")
    expected = [
        (year, quarter) for year in range(2017, 2027) for quarter in range(1, 5)
    ]
    expected = expected[: expected.index((2026, 1)) + 1]
    if keys.rows() != expected:
        raise ValueError("QCEW publication quarters are incomplete or out of order")

    rows = []
    products = (
        ("qcew_news", "news_release_date", "news_release_time_et", "news release"),
        (
            "qcew_full_data",
            "full_data_release_date",
            "full_data_release_time_et",
            "full-data update",
        ),
    )
    for row_number, row in enumerate(frame.iter_rows(named=True), start=2):
        for kind, date_column, time_column, label in products:
            day = row[date_column]
            clock = _clock(row[time_column])
            rows.append(
                {
                    "publication_id": (
                        f"{kind}_{row['qcew_year']}_q{row['qcew_quarter']}"
                    ),
                    "publication_kind": kind,
                    "benchmark_year": None,
                    "qcew_year": row["qcew_year"],
                    "qcew_quarter": row["qcew_quarter"],
                    "release_order": None,
                    "publication_date": day,
                    "observable_at": _utc_observable(day, clock),
                    "source_file": row["source_file"],
                    "source_locator": f"{row['source_locator']}, {label}",
                    "source_keys": [
                        cell_key(
                            QCEW_PUBLICATION_DATES,
                            "qcew_publication_dates",
                            str(row_number),
                            column,
                        )
                        for column in (date_column, time_column)
                    ],
                    "transformation": "parse_publication_datetime",
                }
            )
    return pl.DataFrame(rows, schema=PUBLICATION_SCHEMA)


def catalog_publications(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    return pl.concat([_preliminary_publications(raw_dir), _qcew_publications(raw_dir)])


def _carrier_override_rows(raw_dir: Path) -> dict[int, tuple[int, date]]:
    frame = pl.read_csv(
        raw_dir / FINAL_CARRIER_OVERRIDES,
        schema_overrides={"benchmark_year": pl.Int32, "reference_month": pl.Date},
        try_parse_dates=True,
    )
    rows = {
        row["benchmark_year"]: (row_number, row["reference_month"])
        for row_number, row in enumerate(frame.iter_rows(named=True), start=2)
    }
    observed = {year: month for year, (_, month) in rows.items()}
    if observed != _FINAL_CARRIER_OVERRIDES:
        raise ValueError("final benchmark carrier overrides changed unexpectedly")
    return rows


def final_benchmark_publications(
    release_index: pl.DataFrame, raw_dir: Path = RAW_DIR
) -> pl.DataFrame:
    by_month = {
        row["reference_month"]: row for row in release_index.iter_rows(named=True)
    }
    overrides = _carrier_override_rows(raw_dir)
    rows = []
    for year in range(1979, 2026):
        carrier = final_carrier_month(year)
        release = by_month.get(carrier)
        if release is None:
            raise ValueError(f"no Stage 3 release-index row for benchmark year {year}")
        source_keys = [f"stage3::release_index::{carrier:%Y-%m}"]
        if year in overrides:
            row_number, _ = overrides[year]
            source_keys.append(
                cell_key(
                    FINAL_CARRIER_OVERRIDES,
                    "final_carrier_overrides",
                    str(row_number),
                    "reference_month",
                )
            )
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
                "source_keys": source_keys,
                "transformation": "join_stage3_release_index",
            }
        )
    return pl.DataFrame(rows, schema=PUBLICATION_SCHEMA)


def build_publication_calendar(
    release_index: pl.DataFrame, raw_dir: Path = RAW_DIR
) -> pl.DataFrame:
    frame = pl.concat(
        [
            final_benchmark_publications(release_index, raw_dir),
            catalog_publications(raw_dir),
        ]
    ).sort("publication_date", "publication_id")
    if frame["publication_id"].n_unique() != frame.height:
        raise ValueError("duplicate publication_id")
    preliminary = frame.filter(pl.col("publication_kind") == "benchmark_preliminary")
    if preliminary["benchmark_year"].to_list() != list(range(2000, 2027)):
        raise ValueError("preliminary benchmark publication calendar is incomplete")
    return frame


def publication_raw_values(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    records = []
    for file, table_key in _PUBLICATION_RAW_TABLES.items():
        frame = pl.read_csv(raw_dir / file, infer_schema_length=0)
        for row_number, row in enumerate(frame.iter_rows(named=True), start=2):
            for column in frame.columns:
                records.append(
                    {
                        "source": "manual_transcription",
                        "file": file,
                        "table_key": table_key,
                        "row_key": str(row_number),
                        "column_key": column,
                        "text": row[column] or "",
                    }
                )
    return raw_frame(records)
