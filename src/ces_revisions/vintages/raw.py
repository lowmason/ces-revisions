"""The committed source files in data/raw/ and the immutable raw-value table read from them.

Req 1 keeps every published number in an immutable raw-value table beside a separate
transformations table. A raw cell is the text one cell of a source file holds, so every panel
value traces to the file, row, and column it was read from.
"""

import hashlib
import html
import re
import zipfile
from pathlib import Path

import fastexcel
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PANEL_DIR = DATA_DIR / "panel"

VINTAGE_FILES = "bls/cesvinall.zip"
REVISION_TABLE = "bls/cesnaicsrev.htm"
HISTORICAL_RELEASE_DATES = "bls/histreleasedates.txt"
EMPSIT_RELEASES = "bls/empsit-releases.csv"
VINTAGE_COMMENTS = "bls/cesvin00-comments.csv"
RTDSM_LEVELS = "philadelphiafed/employMvMd.xlsx"
RTDSM_RELEASE_DATES = "philadelphiafed/release-dates-employment-situation.xls"
RESCHEDULES = "manual/es-reschedules.csv"
MANIFEST = "manifest.csv"

# Total nonfarm and the eleven supersectors of Req 5, keyed by vintage-file code. These twelve
# members hold whole thousands; the other 214 carry decimals (docs/ces-revisions-review.md,
# Unrounded NSA inputs) and stay unread in the committed archive.
SECTORS = {
    "000000": "00",
    "100000": "10",
    "200000": "20",
    "300000": "30",
    "400000": "40",
    "500000": "50",
    "550000": "55",
    "600000": "60",
    "650000": "65",
    "700000": "70",
    "800000": "80",
    "900000": "90",
}
SUPERSECTORS = tuple(code for code in SECTORS.values() if code != "00")
SEASONAL_STATUSES = ("NSA", "SA")

# The revision table's twelve estimate columns, and the six revision columns of its summaries.
ESTIMATE_COLUMNS = tuple(
    f"{status}_{column}"
    for status in ("sa", "nsa")
    for column in (
        "1st",
        "2nd",
        "3rd",
        "2nd_minus_1st",
        "3rd_minus_2nd",
        "3rd_minus_1st",
    )
)
SUMMARY_COLUMNS = tuple(column for column in ESTIMATE_COLUMNS if "_minus_" in column)
TABLE_MONTHS = {
    "Jan.": 1,
    "Feb.": 2,
    "Mar.": 3,
    "Apr.": 4,
    "May": 5,
    "Jun.": 6,
    "Jul.": 7,
    "Aug.": 8,
    "Sep.": 9,
    "Oct.": 10,
    "Nov.": 11,
    "Dec.": 12,
}
_STATISTIC_ROWS = {"Mean revision": "mean", "Mean absolute revision": "mean_absolute"}
_TABLE = re.compile(r"(?s)<table\b.*?</table>")
_CAPTION = re.compile(r"(?s)<caption\b.*?</caption>")
_ROW = re.compile(r"(?s)<tr\b.*?</tr>")
_CELL = re.compile(r"(?s)<t[hd]\b[^>]*>(.*?)</t[hd]>")
_TAG = re.compile(r"<[^>]+>")
_SUMMARY_PERIOD = re.compile(r"\d{4} - (?:\d{4}|present)|Total All Periods")


def vintage_file_members() -> list[str]:
    return [
        f"tri_{code}_{status}.csv" for code in SECTORS for status in SEASONAL_STATUSES
    ]


def triangle_cells(payload: bytes, member: str) -> pl.DataFrame:
    """Every non-empty cell of one vintage-file member, keyed by release row and month column."""
    wide = pl.read_csv(payload, infer_schema_length=0)
    return (
        wide.unpivot(
            index=["year", "month"], variable_name="column_key", value_name="text"
        )
        .filter(pl.col("text").is_not_null())
        .select(
            source=pl.lit("cesvinall"),
            file=pl.lit(f"{VINTAGE_FILES}:{member}"),
            row_key=pl.format("{}-{}", "year", pl.col("month").str.zfill(2)),
            column_key="column_key",
            text="text",
        )
    )


def cell_text(fragment: str) -> str:
    """An HTML cell's text: tags dropped, entities decoded, and whitespace collapsed."""
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub(" ", fragment))).strip()


def revision_table_cells(page: str) -> pl.DataFrame:
    """Every data cell of the revision table page: months, yearly averages, and summaries."""
    records = []
    for table in _TABLE.findall(page):
        caption = _CAPTION.search(table)
        title = cell_text(caption.group(0)) if caption else ""
        rows = [
            [cell_text(cell) for cell in _CELL.findall(row)]
            for row in _ROW.findall(table)
        ]
        if "Revisions between over-the-month estimates," in title:
            for cells in rows:
                if len(cells) != 14 or not cells[1].isdigit():
                    continue
                if cells[0] in TABLE_MONTHS:
                    key = f"{cells[1]}-{TABLE_MONTHS[cells[0]]:02d}"
                elif cells[0] in _STATISTIC_ROWS:
                    key = f"{cells[1]}:{_STATISTIC_ROWS[cells[0]]}"
                else:
                    raise ValueError(f"unrecognized revision table row {cells[0]!r}")
                records += [
                    (key, column, text)
                    for column, text in zip(ESTIMATE_COLUMNS, cells[2:], strict=True)
                ]
        elif title.startswith("Summary of"):
            kind = "mean_absolute" if "ABSOLUTE" in title else "mean"
            for cells in rows:
                if len(cells) == 7 and _SUMMARY_PERIOD.fullmatch(cells[0]):
                    records += [
                        (f"summary:{kind}:{cells[0]}", column, text)
                        for column, text in zip(SUMMARY_COLUMNS, cells[1:], strict=True)
                    ]
    frame = pl.DataFrame(
        records, schema=["row_key", "column_key", "text"], orient="row"
    )
    return frame.select(
        source=pl.lit("cesnaicsrev"),
        file=pl.lit(REVISION_TABLE),
        row_key="row_key",
        column_key="column_key",
        text="text",
    )


def rtdsm_cells(wide: pl.DataFrame) -> pl.DataFrame:
    """Every non-empty cell of the EMPLOY matrix: observation rows by vintage columns."""
    return (
        wide.unpivot(index="DATE", variable_name="column_key", value_name="text")
        .filter(pl.col("text").is_not_null() & (pl.col("text") != ""))
        .select(
            source=pl.lit("rtdsm_employ"),
            file=pl.lit(f"{RTDSM_LEVELS}:employ"),
            row_key="DATE",
            column_key="column_key",
            text="text",
        )
    )


def read_rtdsm_matrix(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    sheet = fastexcel.read_excel(raw_dir / RTDSM_LEVELS).load_sheet(
        "employ", header_row=0, dtypes="string"
    )
    return sheet.to_polars()


def raw_values(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    """The immutable raw-value table: every cell Stage 3 reads, each with a stable cell_id."""
    with zipfile.ZipFile(raw_dir / VINTAGE_FILES) as archive:
        parts = [
            triangle_cells(archive.read(member), member)
            for member in vintage_file_members()
        ]
    parts.append(
        revision_table_cells((raw_dir / REVISION_TABLE).read_text(encoding="utf-8"))
    )
    parts.append(rtdsm_cells(read_rtdsm_matrix(raw_dir)))
    return (
        pl.concat(parts)
        .sort("source", "file", "row_key", "column_key")
        .with_row_index("cell_id")
        .with_columns(pl.col("cell_id").cast(pl.Int64))
    )


def read_vintage_comments(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    return pl.read_csv(raw_dir / VINTAGE_COMMENTS, infer_schema_length=0)


def content_sha256(frame: pl.DataFrame, *, chunk_rows: int = 500_000) -> str:
    """SHA-256 of the frame's CSV serialization with its header, hashed chunk by chunk."""
    digest = hashlib.sha256((",".join(frame.columns) + "\n").encode())
    for chunk in frame.iter_slices(chunk_rows):
        digest.update(chunk.write_csv(include_header=False).encode())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
