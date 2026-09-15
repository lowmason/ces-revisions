"""National aggregate QCEW revisions and the March precision proxy."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import polars as pl

from ces_revisions.annual.raw import RAW_DIR, cell_key, parse_number, raw_frame

SOURCE = "bls/qcew-revisions.csv"
EXPECTED_COLUMNS = [
    "Year",
    "Quarter",
    "Area",
    "Field",
    "Initial Value",
    "First Revised Value",
    "Second Revised Value",
    "Third Revised Value",
    "Fourth Revised Value",
    "Final Value",
]
RELEASE_COLUMNS = [
    "Initial Value",
    "First Revised Value",
    "Second Revised Value",
    "Third Revised Value",
    "Fourth Revised Value",
]
RELEASE_COUNT = {1: 5, 2: 4, 3: 3, 4: 2}
MONTH = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}
QCEW_REVISION_SCHEMA = {
    "qcew_year": pl.Int32,
    "qcew_quarter": pl.Int8,
    "area": pl.String,
    "field": pl.String,
    "reference_month": pl.Date,
    "release_order": pl.Int8,
    "release_label": pl.String,
    "employment_jobs": pl.Int64,
    "revision_jobs": pl.Int64,
    "unit": pl.String,
    "seasonal_status": pl.String,
    "sequence_status": pl.String,
    "publication_date": pl.Date,
    "observable_at": pl.Datetime("us", "UTC"),
    "source_file": pl.String,
    "source_locator": pl.String,
    "source_keys": pl.List(pl.String),
    "transformation": pl.String,
}
QCEW_PRECISION_SCHEMA = {
    "benchmark_year": pl.Int32,
    "reference_month": pl.Date,
    "release_count": pl.Int8,
    "sequence_status": pl.String,
    "revision_precision_thousands": pl.Float64,
    "unit": pl.String,
    "publication_date": pl.Date,
    "observable_at": pl.Datetime("us", "UTC"),
    "source_keys": pl.List(pl.String),
    "transformation": pl.String,
}


def read_source(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    frame = pl.read_csv(raw_dir / SOURCE, infer_schema_length=0).with_row_index(
        "source_row", offset=2
    )
    if frame.columns[1:] != EXPECTED_COLUMNS:
        raise ValueError(f"QCEW revision header changed: {frame.columns[1:]!r}")
    return frame.with_columns(
        pl.col("Year").cast(pl.Int32),
        pl.col("Quarter").cast(pl.Int8),
    ).filter(
        (pl.col("Area") == "United States")
        & (pl.col("Year").cast(pl.Int32) >= 2017)
        & pl.col("Field").str.ends_with(" Employment")
    )


def _available_values(row: dict) -> list[tuple[int, str, int]]:
    quarter = int(row["Quarter"])
    available = []
    missing_seen = False
    for order, column in enumerate(RELEASE_COLUMNS[: RELEASE_COUNT[quarter]]):
        value = parse_number(row[column] or "")
        if value is None:
            missing_seen = True
            continue
        if missing_seen:
            raise ValueError(
                f"QCEW row {row['source_row']} has a revision after a missing release"
            )
        if not value.is_integer():
            raise ValueError(f"QCEW row {row['source_row']} is not whole jobs")
        available.append((order, column, int(value)))
    final = parse_number(row["Final Value"] or "")
    if final is not None and (not available or int(final) != available[-1][2]):
        raise ValueError(
            f"QCEW Final Value is not the last applicable revision on row "
            f"{row['source_row']}"
        )
    return available


def raw_values(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    frame = read_source(raw_dir)
    records = [
        {
            "source": "bls_qcew_revisions",
            "file": SOURCE,
            "table_key": "qcew_revisions",
            "row_key": str(row["source_row"]),
            "column_key": column,
            "text": "" if row[column] is None else str(row[column]),
        }
        for row in frame.iter_rows(named=True)
        for column in EXPECTED_COLUMNS
    ]
    return raw_frame(records)


def _publication(calendar: pl.DataFrame, year: int, quarter: int, order: int) -> dict:
    rows = calendar.filter(
        (pl.col("publication_kind") == "qcew_revision")
        & (pl.col("qcew_year") == year)
        & (pl.col("qcew_quarter") == quarter)
        & (pl.col("release_order") == order)
    )
    if rows.height != 1:
        raise ValueError(
            f"expected one QCEW publication for {(year, quarter, order)}, "
            f"found {rows.height}"
        )
    return rows.row(0, named=True)


def build_qcew_revisions(
    calendar: pl.DataFrame, raw_dir: Path = RAW_DIR
) -> pl.DataFrame:
    rows = []
    for source in read_source(raw_dir).iter_rows(named=True):
        year = int(source["Year"])
        quarter = int(source["Quarter"])
        month_name = source["Field"].removesuffix(" Employment")
        month = MONTH.get(month_name)
        if month is None or (month - 1) // 3 + 1 != quarter:
            raise ValueError(
                f"QCEW row {source['source_row']} has an invalid field/quarter pair"
            )
        available = _available_values(source)
        complete = len(available) == RELEASE_COUNT[quarter]
        previous = None
        previous_column = None
        for order, column, value in available:
            publication = _publication(calendar, year, quarter, order)
            keys = [
                cell_key(
                    SOURCE,
                    "qcew_revisions",
                    str(source["source_row"]),
                    column,
                )
            ]
            revision = None
            transformation = "parse_published_number"
            if previous is not None:
                revision = value - previous
                transformation = "successive_difference"
                keys.append(
                    cell_key(
                        SOURCE,
                        "qcew_revisions",
                        str(source["source_row"]),
                        previous_column,
                    )
                )
            keys.extend(publication["source_keys"])
            rows.append(
                {
                    "qcew_year": year,
                    "qcew_quarter": quarter,
                    "area": "United States",
                    "field": source["Field"],
                    "reference_month": date(year, month, 1),
                    "release_order": order,
                    "release_label": column,
                    "employment_jobs": value,
                    "revision_jobs": revision,
                    "unit": "jobs",
                    "seasonal_status": "NSA",
                    "sequence_status": "complete" if complete else "right_censored",
                    "publication_date": publication["publication_date"],
                    "observable_at": publication["observable_at"],
                    "source_file": SOURCE,
                    "source_locator": f"CSV row {source['source_row']}, {column}",
                    "source_keys": keys,
                    "transformation": transformation,
                }
            )
            previous, previous_column = value, column
    frame = pl.DataFrame(rows, schema=QCEW_REVISION_SCHEMA).sort(
        "reference_month", "release_order"
    )
    keys = ["reference_month", "release_order"]
    if frame.select(keys).n_unique() != frame.height:
        raise ValueError("duplicate national QCEW revision rows")
    return frame


def build_qcew_precision(revisions: pl.DataFrame) -> pl.DataFrame:
    rows = []
    march = revisions.filter(pl.col("field") == "March Employment")
    for year in sorted(march["qcew_year"].unique()):
        group = march.filter(pl.col("qcew_year") == year).sort("release_order")
        complete = group.height == RELEASE_COUNT[1]
        increments = group.filter(pl.col("release_order") > 0)["revision_jobs"]
        precision = None
        if complete:
            precision = math.sqrt(sum((value / 1_000) ** 2 for value in increments) / 4)
        latest = group.tail(1).row(0, named=True)
        rows.append(
            {
                "benchmark_year": year,
                "reference_month": date(year, 3, 1),
                "release_count": group.height,
                "sequence_status": "complete" if complete else "right_censored",
                "revision_precision_thousands": precision,
                "unit": "thousands",
                "publication_date": latest["publication_date"],
                "observable_at": latest["observable_at"],
                "source_keys": group["source_keys"]
                .explode(empty_as_null=True)
                .unique()
                .sort()
                .to_list(),
                "transformation": "rms_qcew_revision",
            }
        )
    return pl.DataFrame(rows, schema=QCEW_PRECISION_SCHEMA).sort("benchmark_year")
