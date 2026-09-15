"""Minimal HTML table extraction for the stable BLS tables Stage 4 reads."""

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

import polars as pl

from ces_revisions.annual.raw import raw_frame


@dataclass(frozen=True)
class HtmlTable:
    caption: str
    rows: tuple[tuple[str, ...], ...]


def _clean(parts: list[str]) -> str:
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


@dataclass
class _TableState:
    fallback_caption: str | None
    caption_parts: list[str] = field(default_factory=list)
    rows: list[tuple[str, ...]] = field(default_factory=list)
    row: list[str] | None = None
    cell_parts: list[str] | None = None
    in_caption: bool = False


class _Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[HtmlTable] = []
        self.table_stack: list[_TableState] = []
        self.pending_caption: str | None = None
        self.title_tag: str | None = None
        self.title_parts: list[str] = []
        self.superscript_parts: list[str] | None = None

    @property
    def _table(self) -> _TableState | None:
        return self.table_stack[-1] if self.table_stack else None

    @staticmethod
    def _finish_cell(table: _TableState, *, keep_empty: bool) -> None:
        if table.cell_parts is None:
            return
        if table.row is None:
            table.row = []
        value = _clean(table.cell_parts)
        if value or keep_empty:
            table.row.append(value)
        table.cell_parts = None

    @classmethod
    def _finish_row(cls, table: _TableState) -> None:
        cls._finish_cell(table, keep_empty=True)
        if table.row:
            table.rows.append(tuple(table.row))
        table.row = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "sup":
            self.superscript_parts = []
            return
        if self.title_tag is None and tag in {
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "p",
        }:
            self.title_tag = tag
            self.title_parts = []
        if tag == "table":
            fallback = self.pending_caption
            self.pending_caption = None
            parent = self._table
            if (
                fallback is None
                and parent is not None
                and parent.fallback_caption is not None
                and not parent.caption_parts
                and not parent.rows
                and not parent.row
                and parent.cell_parts is None
            ):
                fallback = parent.fallback_caption
                parent.fallback_caption = None
            self.table_stack.append(_TableState(fallback))
            return
        table = self._table
        if table is None:
            return
        if tag == "caption":
            table.in_caption = True
        elif tag == "tr":
            if table.row is not None:
                self._finish_row(table)
            table.row = []
        elif tag in {"th", "td"}:
            if table.cell_parts is not None:
                self._finish_cell(table, keep_empty=False)
            if table.row is None:
                table.row = []
            table.cell_parts = []

    def handle_data(self, data: str) -> None:
        if self.superscript_parts is not None:
            self.superscript_parts.append(data)
            return
        if self.title_tag is not None:
            self.title_parts.append(data)
        table = self._table
        if table is None:
            return
        if table.in_caption:
            table.caption_parts.append(data)
        if table.cell_parts is not None:
            table.cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "sup" and self.superscript_parts is not None:
            marker = _clean(self.superscript_parts)
            if marker and not (marker.startswith("(") and marker.endswith(")")):
                marker = f"({marker})"
            if marker:
                if self.title_tag is not None:
                    self.title_parts.append(marker)
                table = self._table
                if table is not None:
                    if table.in_caption:
                        table.caption_parts.append(marker)
                    if table.cell_parts is not None:
                        table.cell_parts.append(marker)
            self.superscript_parts = None
            return
        table = self._table
        if table is not None:
            if tag in {"th", "td"}:
                self._finish_cell(table, keep_empty=True)
            elif tag == "tr":
                self._finish_row(table)
            elif tag == "caption":
                table.in_caption = False
            elif tag == "table":
                self._finish_row(table)
                self.table_stack.pop()
                native_caption = _clean(table.caption_parts)
                caption = native_caption or table.fallback_caption or ""
                if table.rows or native_caption:
                    self.tables.append(HtmlTable(caption, tuple(table.rows)))
                elif table.fallback_caption:
                    self.pending_caption = table.fallback_caption
        if tag == self.title_tag:
            title = _clean(self.title_parts)
            if re.match(r"^Table\s", title, re.IGNORECASE):
                self.pending_caption = title
            self.title_tag = None
            self.title_parts = []


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
