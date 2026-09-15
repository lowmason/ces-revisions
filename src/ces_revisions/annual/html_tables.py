"""Minimal HTML table extraction for the stable BLS tables Stage 4 reads."""

import re
from dataclasses import dataclass
from html.parser import HTMLParser

import polars as pl

from ces_revisions.annual.raw import raw_frame


@dataclass(frozen=True)
class HtmlTable:
    caption: str
    rows: tuple[tuple[str, ...], ...]


def _clean(parts: list[str]) -> str:
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


class _Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[HtmlTable] = []
        self.in_table = False
        self.in_caption = False
        self.in_cell = False
        self.caption_parts: list[str] = []
        self.cell_parts: list[str] = []
        self.rows: list[tuple[str, ...]] = []
        self.row: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "table" and not self.in_table:
            self.in_table = True
            self.caption_parts, self.rows = [], []
        elif self.in_table and tag == "caption":
            self.in_caption = True
        elif self.in_table and tag == "tr":
            self.row = []
        elif self.in_table and tag in {"th", "td"}:
            self.in_cell = True
            self.cell_parts = []

    def handle_data(self, data: str) -> None:
        if self.in_caption:
            self.caption_parts.append(data)
        if self.in_cell:
            self.cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self.in_cell:
            if self.row is None:
                raise ValueError("table cell outside a row")
            self.row.append(_clean(self.cell_parts))
            self.in_cell = False
        elif tag == "tr" and self.in_table:
            if self.row:
                self.rows.append(tuple(self.row))
            self.row = None
        elif tag == "caption":
            self.in_caption = False
        elif tag == "table" and self.in_table:
            self.tables.append(HtmlTable(_clean(self.caption_parts), tuple(self.rows)))
            self.in_table = False


def parse_tables(page: str) -> tuple[HtmlTable, ...]:
    parser = _Parser()
    parser.feed(page)
    return tuple(parser.tables)


def find_table(page: str, caption: str) -> HtmlTable:
    pattern = re.compile(caption, re.IGNORECASE)
    matches = [table for table in parse_tables(page) if pattern.search(table.caption)]
    if len(matches) != 1:
        raise ValueError(
            f"expected one table matching {caption!r}, found {len(matches)}"
        )
    return matches[0]


def table_raw_cells(
    table: HtmlTable, *, source: str, file: str, table_key: str
) -> pl.DataFrame:
    records = [
        {
            "source": source,
            "file": file,
            "table_key": table_key,
            "row_key": str(row_index),
            "column_key": str(column_index),
            "text": text,
        }
        for row_index, row in enumerate(table.rows)
        for column_index, text in enumerate(row)
    ]
    return raw_frame(records)
