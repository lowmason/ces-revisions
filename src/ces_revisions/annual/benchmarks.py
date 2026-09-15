"""CES final/preliminary benchmark revisions and major-industry contributions."""

import re
from pathlib import Path

import polars as pl

from ces_revisions.annual import html_tables
from ces_revisions.annual.raw import RAW_DIR, cell_key, parse_number, raw_frame

TECHNICAL_NOTES = "bls/cestn.htm"
PRELIMINARY_2026 = "bls/preliminary/prebmk-2026.htm"
SECTOR_CELLS = "manual/benchmark-sector-cells.csv"

BENCHMARK_SCHEMA = {
    "benchmark_year": pl.Int32,
    "benchmark_status": pl.String,
    "row_kind": pl.String,
    "sector": pl.String,
    "seasonal_status": pl.String,
    "footnotes": pl.List(pl.Int16),
    "revision_thousands": pl.Float64,
    "percent_revision": pl.Float64,
    "article_revision_thousands": pl.Float64,
    "article_difference_thousands": pl.Float64,
    "discrepancy_status": pl.String,
    "publication_date": pl.Date,
    "observable_at": pl.Datetime("us", "UTC"),
    "source_file": pl.String,
    "source_locator": pl.String,
    "source_keys": pl.List(pl.String),
    "transformation": pl.String,
}


def _year(text: str) -> int | None:
    match = re.match(r"^(\d{4})", text)
    return int(match.group(1)) if match else None


def _footnotes(*texts: str) -> list[int]:
    return sorted(
        {int(value) for text in texts for value in re.findall(r"\((\d+)\)", text)}
    )


def _table5(raw_dir: Path) -> html_tables.HtmlTable:
    page = (raw_dir / TECHNICAL_NOTES).read_text(encoding="utf-8")
    return html_tables.find_table(page, r"Table 5\..*preliminary and final benchmark")


def table5_raw_values(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    preliminary = html_tables.find_table(
        (raw_dir / PRELIMINARY_2026).read_text(encoding="utf-8"),
        r"Table 1\..*March 2026.*Preliminary Benchmark",
    )
    return pl.concat(
        [
            html_tables.table_raw_cells(
                _table5(raw_dir),
                source="bls",
                file=TECHNICAL_NOTES,
                table_key="table_5",
            ),
            html_tables.table_raw_cells(
                preliminary,
                source="bls",
                file=PRELIMINARY_2026,
                table_key="preliminary_2026_table_1",
            ),
        ]
    )


def preliminary_2026_values(
    raw_dir: Path = RAW_DIR,
) -> tuple[float, float, list[str]]:
    page = (raw_dir / PRELIMINARY_2026).read_text(encoding="utf-8")
    table = html_tables.find_table(
        page, r"Table 1\..*March 2026.*Preliminary Benchmark"
    )
    matches = [
        (row_index, row)
        for row_index, row in enumerate(table.rows)
        if row and row[0] == "Total nonfarm"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one 2026 total-nonfarm row, found {len(matches)}")
    row_index, row = matches[0]
    revision = parse_number(row[1])
    percent = parse_number(row[2])
    if revision is None or percent is None:
        raise ValueError("the 2026 total-nonfarm preliminary cells are not numeric")
    return (
        revision,
        percent,
        [
            cell_key(
                PRELIMINARY_2026,
                "preliminary_2026_table_1",
                str(row_index),
                column,
            )
            for column in ("1", "2")
        ],
    )


def benchmark_sector_raw_values(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    frame = pl.read_csv(raw_dir / SECTOR_CELLS, infer_schema_length=0)
    records = []
    for row_number, row in enumerate(frame.iter_rows(named=True), start=2):
        for column in frame.columns:
            records.append(
                {
                    "source": "manual_transcription",
                    "file": SECTOR_CELLS,
                    "table_key": "benchmark_sector_cells",
                    "row_key": str(row_number),
                    "column_key": column,
                    "text": row[column] or "",
                }
            )
    return raw_frame(records)


def _publication(calendar: pl.DataFrame, publication_id: str) -> dict:
    rows = calendar.filter(pl.col("publication_id") == publication_id)
    if rows.height != 1:
        raise ValueError(
            f"expected one publication {publication_id}, found {rows.height}"
        )
    return rows.row(0, named=True)


def _anchor_rows(calendar: pl.DataFrame, raw_dir: Path) -> list[dict]:
    table = _table5(raw_dir)
    rows = []
    article = pl.read_csv(raw_dir / SECTOR_CELLS, infer_schema_length=0)
    article_totals = {}
    for article_row_number, article_row in enumerate(
        article.iter_rows(named=True), start=2
    ):
        if article_row["sector"] == "00":
            article_totals[int(article_row["benchmark_year"])] = (
                parse_number(article_row["revision_text"]),
                article_row_number,
            )
    for row_index, cells in enumerate(table.rows):
        year = _year(cells[0]) if cells else None
        if year is None:
            continue
        final_percent = parse_number(cells[1]) if len(cells) > 2 else None
        final_revision = parse_number(cells[2]) if len(cells) > 2 else None
        preliminary_percent = parse_number(cells[3]) if len(cells) > 4 else None
        preliminary_revision = parse_number(cells[4]) if len(cells) > 4 else None
        for status, percent, revision, columns in (
            ("final", final_percent, final_revision, ("1", "2")),
            ("preliminary", preliminary_percent, preliminary_revision, ("3", "4")),
        ):
            if revision is None:
                continue
            publication = _publication(calendar, f"benchmark_{status}_{year}")
            article_entry = article_totals.get(year) if status == "final" else None
            article_revision = article_entry[0] if article_entry else None
            difference = (
                article_revision - revision if article_revision is not None else None
            )
            keys = [
                cell_key(TECHNICAL_NOTES, "table_5", str(row_index), column)
                for column in ("0", *columns)
            ]
            if article_entry:
                keys.append(
                    cell_key(
                        SECTOR_CELLS,
                        "benchmark_sector_cells",
                        str(article_entry[1]),
                        "revision_text",
                    )
                )
            keys.extend(publication["source_keys"])
            rows.append(
                {
                    "benchmark_year": year,
                    "benchmark_status": status,
                    "row_kind": "anchor",
                    "sector": "00",
                    "seasonal_status": "NSA",
                    "footnotes": sorted(
                        {
                            1,
                            *_footnotes(
                                cells[0],
                                *(cells[int(column)] for column in columns),
                            ),
                        }
                    ),
                    "revision_thousands": revision,
                    "percent_revision": percent,
                    "article_revision_thousands": article_revision,
                    "article_difference_thousands": difference,
                    "discrepancy_status": (
                        "unexplained" if year == 2025 and status == "final" else None
                    ),
                    "publication_date": publication["publication_date"],
                    "observable_at": publication["observable_at"],
                    "source_file": TECHNICAL_NOTES,
                    "source_locator": f"Table 5, row {year}, {status}",
                    "source_keys": keys,
                    "transformation": (
                        "benchmark_article_comparison"
                        if article_entry
                        else "parse_thousands"
                    ),
                }
            )
    if not any(
        row["benchmark_year"] == 2026 and row["benchmark_status"] == "preliminary"
        for row in rows
    ):
        revision, percent, keys = preliminary_2026_values(raw_dir)
        publication = _publication(calendar, "benchmark_preliminary_2026")
        keys.extend(publication["source_keys"])
        rows.append(
            {
                "benchmark_year": 2026,
                "benchmark_status": "preliminary",
                "row_kind": "anchor",
                "sector": "00",
                "seasonal_status": "NSA",
                "footnotes": [],
                "revision_thousands": revision,
                "percent_revision": percent,
                "article_revision_thousands": None,
                "article_difference_thousands": None,
                "discrepancy_status": None,
                "publication_date": publication["publication_date"],
                "observable_at": publication["observable_at"],
                "source_file": PRELIMINARY_2026,
                "source_locator": "Table 1, Total nonfarm",
                "source_keys": keys,
                "transformation": "parse_thousands",
            }
        )
    return rows


def _sector_rows(calendar: pl.DataFrame, raw_dir: Path) -> list[dict]:
    frame = pl.read_csv(raw_dir / SECTOR_CELLS, infer_schema_length=0)
    rows = []
    for row_number, row in enumerate(frame.iter_rows(named=True), start=2):
        if row["sector"] == "00":
            continue
        year = int(row["benchmark_year"])
        publication = _publication(calendar, f"benchmark_final_{year}")
        rows.append(
            {
                "benchmark_year": year,
                "benchmark_status": "final",
                "row_kind": "sector_contribution",
                "sector": row["sector"],
                "seasonal_status": "NSA",
                "footnotes": [],
                "revision_thousands": parse_number(row["revision_text"]),
                "percent_revision": parse_number(row["percent_text"]),
                "article_revision_thousands": None,
                "article_difference_thousands": None,
                "discrepancy_status": None,
                "publication_date": publication["publication_date"],
                "observable_at": publication["observable_at"],
                "source_file": row["source_file"],
                "source_locator": row["source_locator"],
                "source_keys": [
                    cell_key(
                        SECTOR_CELLS,
                        "benchmark_sector_cells",
                        str(row_number),
                        column,
                    )
                    for column in ("revision_text", "percent_text")
                ]
                + publication["source_keys"],
                "transformation": "parse_thousands",
            }
        )
    return rows


def build_benchmarks(calendar: pl.DataFrame, raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    rows = _anchor_rows(calendar, raw_dir) + _sector_rows(calendar, raw_dir)
    frame = pl.DataFrame(rows, schema=BENCHMARK_SCHEMA).sort(
        "benchmark_year", "benchmark_status", "row_kind", "sector"
    )
    keys = ["benchmark_year", "benchmark_status", "row_kind", "sector"]
    if frame.select(keys).n_unique() != frame.height:
        raise ValueError("duplicate benchmark rows")
    return frame
