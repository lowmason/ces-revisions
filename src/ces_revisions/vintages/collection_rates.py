"""The CES collection rates C1, C2 and C3 from LABSTAT and the archived registry pages.

C1 (`CEU00000000C1`) is the first-closing collection rate, in percent, and C2 and C3 are the
rates of the second preliminary and third releases. LABSTAT publishes them with
`CEU05000000RR`, and every archived copy of BLS's registry-receipts page
(`cesregrec.htm`) is a snapshot of the history as BLS showed it that day. The comparison of each
copy with LABSTAT checks that no published value changed (spec D8).
"""

import re
from datetime import date
from pathlib import Path

import polars as pl

from ces_revisions.vintages import raw

SERIES_BY_SECTION = {
    "first preliminary": "CEU00000000C1",
    "second preliminary": "CEU00000000C2",
    "third": "CEU00000000C3",
}
LABSTAT_SOURCE = "labstat"
# The last copy of each calendar year that D8 compares, beside the copy of 2025-03-19, which the
# recon fetch read and which ends with the February 2025 data.
ARCHIVE_YEARS = range(2008, 2025)
RECON_COPY = "20250319165303"
SCHEMA = {
    "series_id": pl.String,
    "reference_month": pl.Date,
    "source": pl.String,
    "value": pl.Float64,
    "footnote_code": pl.String,
}

_MONTHLY_PERIOD = re.compile(r"M(0[1-9]|1[0-2])")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_LEADING_NUMBER = re.compile(r"^\d+(?:\.\d+)?")
_YEAR = re.compile(r"\d{4}")
# A table with no table inside it. From 2008 to 2019 BLS's page nests the registry table in a
# layout table, and a row read across both would merge the layout row with the registry
# table's first row.
INNERMOST_TABLE = re.compile(r"(?is)<table\b(?:(?!<table\b).)*?</table\s*>")


def labstat_values(text: str) -> pl.DataFrame:
    """The monthly values of a LABSTAT `ce.data` text: periods M01 to M12 with a number.

    The header line, annual averages (M13) and blank or `-` values give no row.
    """
    rows = []
    for line in text.splitlines():
        fields = [field.strip() for field in line.split("\t")]
        if len(fields) < 4:
            continue
        series_id, year, period, value = fields[:4]
        footnote = fields[4] if len(fields) > 4 else ""
        if not _MONTHLY_PERIOD.fullmatch(period) or not _NUMBER.fullmatch(value):
            continue
        rows.append(
            (
                series_id,
                date(int(year), int(period[1:]), 1),
                LABSTAT_SOURCE,
                float(value),
                footnote or None,
            )
        )
    return pl.DataFrame(rows, schema=SCHEMA, orient="row")


def registry_page_values(page: str, source: str) -> pl.DataFrame:
    """C1, C2 and C3 from the table rows of an archived registry-receipts page.

    Only `<tr>` rows of innermost tables count, so numbers and section names in the page's prose
    or its layout give nothing. A row with one non-empty cell that starts with a section's name
    opens that section, and a row whose first cell is a year gives up to 12 monthly values, read
    as each cell's leading number.
    """
    rows = []
    series_id = None
    table_rows = (
        row
        for table in INNERMOST_TABLE.findall(page)
        for row in raw._ROW.findall(table)
    )
    for row in table_rows:
        cells = [raw.cell_text(cell) for cell in raw._CELL.findall(row)]
        filled = [cell for cell in cells if cell]
        if len(filled) == 1:
            label = filled[0].lower()
            for section, series in SERIES_BY_SECTION.items():
                if label.startswith(section):
                    series_id = series
            continue
        if series_id is None or not cells or not _YEAR.fullmatch(cells[0]):
            continue
        year = int(cells[0])
        for month, cell in enumerate(cells[1:13], start=1):
            number = _LEADING_NUMBER.match(cell)
            if number:
                rows.append(
                    (series_id, date(year, month, 1), source, float(number[0]), None)
                )
    frame = pl.DataFrame(rows, schema=SCHEMA, orient="row")
    found = set(frame["series_id"].unique().to_list())
    missing = [
        section for section, series in SERIES_BY_SECTION.items() if series not in found
    ]
    if missing:
        raise ValueError(
            f"registry page {source} has no values in section(s): {', '.join(missing)}"
        )
    return frame


def build_collection_rates(raw_dir: Path = raw.RAW_DIR) -> pl.DataFrame:
    """LABSTAT's rows and every archived copy's rows, one per series, month and source."""
    parts = [
        labstat_values((raw_dir / raw.COLLECTION_RATES).read_text(encoding="utf-8"))
    ]
    for copy in sorted((raw_dir / raw.REGISTRY_COPIES).glob("*.htm")):
        page = copy.read_text(encoding="utf-8", errors="replace")
        parts.append(registry_page_values(page, copy.stem))
    return pl.concat(parts).sort("source", "series_id", "reference_month")


def _tenths(column: str) -> pl.Expr:
    return (pl.col(column) * 10).round(0).cast(pl.Int64)


def _paired(rates: pl.DataFrame) -> pl.DataFrame:
    """Each archived C1, C2 or C3 value beside LABSTAT's value for its series and month."""
    compared = rates.filter(pl.col("series_id").is_in(list(SERIES_BY_SECTION.values())))
    labstat = compared.filter(pl.col("source") == LABSTAT_SOURCE).select(
        "series_id", "reference_month", labstat_value="value"
    )
    copies = compared.filter(pl.col("source") != LABSTAT_SOURCE).select(
        "series_id", "reference_month", "source", copy_value="value"
    )
    return copies.join(labstat, on=["series_id", "reference_month"], how="left")


def archive_comparison(rates: pl.DataFrame) -> pl.DataFrame:
    """Per archived copy: values compared with LABSTAT, values that differ, values only in it.

    Values compare as integer tenths, as both sources print them to one decimal.
    """
    paired = _paired(rates)
    return (
        paired.group_by("source")
        .agg(
            compared=pl.col("labstat_value").is_not_null().sum().cast(pl.Int64),
            differing=(_tenths("copy_value") != _tenths("labstat_value"))
            .fill_null(False)
            .sum()
            .cast(pl.Int64),
            only_in_copy=pl.col("labstat_value").is_null().sum().cast(pl.Int64),
        )
        .sort("source")
    )


def archive_differences(rates: pl.DataFrame) -> pl.DataFrame:
    """Every archived value whose tenths differ from LABSTAT's, with both values."""
    return (
        _paired(rates)
        .filter(_tenths("copy_value") != _tenths("labstat_value"))
        .select("series_id", "reference_month", "source", "copy_value", "labstat_value")
        .sort("source", "series_id", "reference_month")
    )
