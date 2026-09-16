"""Documented benchmark scope changes and reconstructions."""

from datetime import date
from pathlib import Path

import polars as pl

from ces_revisions.annual.raw import RAW_DIR, cell_key, parse_number, raw_frame

EVENTS = "manual/reconstruction-events.csv"


def raw_values(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    frame = pl.read_csv(raw_dir / EVENTS, infer_schema_length=0)
    records = []
    for row_number, row in enumerate(frame.iter_rows(named=True), start=2):
        for column in frame.columns:
            records.append(
                {
                    "source": "manual_transcription",
                    "file": EVENTS,
                    "table_key": "reconstruction_events",
                    "row_key": str(row_number),
                    "column_key": column,
                    "text": row[column] or "",
                }
            )
    return raw_frame(records)


def _optional_date(text: str | None) -> date | None:
    return date.fromisoformat(text) if text else None


def build_reconstruction_events(
    calendar: pl.DataFrame, raw_dir: Path = RAW_DIR
) -> pl.DataFrame:
    source = pl.read_csv(raw_dir / EVENTS, infer_schema_length=0)
    publications = {
        row["benchmark_year"]: row
        for row in calendar.filter(
            pl.col("publication_kind") == "benchmark_final"
        ).iter_rows(named=True)
    }
    rows = []
    for row_number, row in enumerate(source.iter_rows(named=True), start=2):
        year = int(row["benchmark_year"])
        publication = publications[year]
        rows.append(
            {
                "event_id": row["event_id"],
                "benchmark_year": year,
                "event_type": row["event_type"],
                "sectors": row["sectors"].split(";") if row["sectors"] else [],
                "series_codes": (
                    row["series_codes"].split(";") if row["series_codes"] else []
                ),
                "reference_start": _optional_date(row["reference_start"]),
                "reference_end": _optional_date(row["reference_end"]),
                "effect_thousands": parse_number(row["effect_thousands_text"] or ""),
                "footnote": int(row["footnote"]) if row["footnote"] else None,
                "description": row["description"],
                "publication_date": publication["publication_date"],
                "observable_at": publication["observable_at"],
                "source_file": row["source_file"],
                "source_locator": row["source_locator"],
                "source_keys": [
                    cell_key(
                        EVENTS,
                        "reconstruction_events",
                        str(row_number),
                        column,
                    )
                    for column in (
                        "event_type",
                        "effect_thousands_text",
                        "description",
                    )
                ]
                + publication["source_keys"],
                "transformation": "parse_thousands",
            }
        )
    return pl.DataFrame(rows).sort("benchmark_year", "event_id")
