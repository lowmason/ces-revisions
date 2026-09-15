"""Paths, source cells, hashes, and named transformations for Stage 4."""

import hashlib
import re
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "annual"
RAW_DIR = DATA_DIR / "raw"
PANEL_DIR = DATA_DIR / "panel"
CACHE_DIR = DATA_DIR / "cache"
MANIFEST = "manifest.csv"
SOURCE_CATALOG = "source-catalog.csv"
PUBLICATION_DATES = "manual/publication-dates.csv"

RAW_SCHEMA = {
    "cell_key": pl.String,
    "source": pl.String,
    "file": pl.String,
    "table_key": pl.String,
    "row_key": pl.String,
    "column_key": pl.String,
    "text": pl.String,
}

TRANSFORMATIONS = pl.DataFrame(
    [
        ("identity", "Copy a published numeric value without rescaling."),
        (
            "parse_published_number",
            "Parse a published count or percentage without changing its unit.",
        ),
        ("parse_thousands", "Parse a CES value already printed in thousands."),
        (
            "benchmark_article_comparison",
            "Parse the Table 5 anchor and subtract it from the separately published article total.",
        ),
        (
            "qcew_jobs_to_thousands",
            "Divide a QCEW person-count employment value by 1,000.",
        ),
        (
            "sample_jobs_to_thousands",
            "Divide a CES sample employee count by 1,000 when the archived table cells are counts despite a thousands header.",
        ),
        (
            "successive_difference",
            "Subtract the preceding publication of the same QCEW cell.",
        ),
        (
            "rms_qcew_revision",
            "Root mean square of the four finalized Q1 March revision increments, in thousands.",
        ),
        (
            "realized_minus_forecast",
            "Subtract annual forecast birth-death from its published realized value.",
        ),
        (
            "government_structural_zero",
            "Set birth-death to zero because the model applies only to private industries.",
        ),
        (
            "explicit_archive_gap",
            "Materialize a null row where the source table is documented absent.",
        ),
    ],
    schema=["transformation", "description"],
    orient="row",
)

_MISSING = {"", "-", "--", "N/A", "NA", "Not applicable", "Not yet published"}


def cell_key(file: str, table_key: str, row_key: str, column_key: str) -> str:
    """A stable, human-readable key for one printed source cell."""
    return f"{file}::{table_key}::{row_key}::{column_key}"


def parse_number(text: str) -> float | None:
    """Parse BLS numeric text while preserving documented missing values."""
    cleaned = re.sub(r"\([A-Za-z0-9]+\)$", "", text.strip())
    cleaned = cleaned.translate(str.maketrans("−‐–", "---")).replace(",", "").strip()
    if cleaned in _MISSING:
        return None
    cleaned = cleaned.removeprefix("< ")
    return float(cleaned)


def raw_frame(records: list[dict[str, str]]) -> pl.DataFrame:
    """Build a sorted immutable raw-cell frame and reject ambiguous locations."""
    rows = []
    for record in records:
        row = dict(record)
        row["cell_key"] = cell_key(
            row["file"], row["table_key"], row["row_key"], row["column_key"]
        )
        rows.append(row)
    frame = pl.DataFrame(rows, schema=RAW_SCHEMA).sort("cell_key")
    if frame["cell_key"].n_unique() != frame.height:
        raise ValueError("duplicate raw cell keys")
    return frame


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_sha256(frame: pl.DataFrame) -> str:
    payload = frame.write_json().encode()
    return hashlib.sha256(payload).hexdigest()


def manifest_frame(raw_dir: Path = RAW_DIR, *, fetched_at: str) -> pl.DataFrame:
    """Hash every source/transcription below raw_dir except manifest.csv itself."""
    rows = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.name in {MANIFEST, ".DS_Store"}:
            continue
        rows.append(
            {
                "file": str(path.relative_to(raw_dir)),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "fetched_at": fetched_at,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "file": pl.String,
            "sha256": pl.String,
            "bytes": pl.Int64,
            "fetched_at": pl.String,
        },
    )


def read_source_catalog(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    return pl.read_csv(raw_dir / SOURCE_CATALOG, infer_schema_length=0)
