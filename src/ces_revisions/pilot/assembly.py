"""Assemble the pilot's aggregate panel from the Stage 3, 4, and 5 artifacts.

Each closing cell, a first, second, or third same-release over-the-month change of not
seasonally adjusted total nonfarm, is read from one source, chosen by its reference
month: BLS's revision table before May 2003, the vintage files from it, and the revision
table again where the vintage files end before the cell's release. The final March
benchmark anchors are Stage 5's wedge anchors, with earlier reconstructions kept apart as
an observation offset. A second or third estimate that a final benchmark release carried
is masked, because that release also revises the levels it is differenced from.

The EMPLOY helpers keep the evidence for taking the 1979 leg's changes from the revision
table: the repeated October 2025 vintage, and the cells where EMPLOY's same-vintage
seasonally adjusted changes disagree with the table.
"""

from pathlib import Path

import numpy as np
import polars as pl

from ces_revisions.pilot.panel import (
    ANCHOR_YEARS,
    CELL_SCHEMA,
    CLOSING_STAGES,
    COLUMNS,
    CRISIS_MONTHS,
    FIRST_MONTH,
    REGIME_SHIFT_MONTH,
    AggregatePanel,
    benchmark_year_of,
)
from ces_revisions.vintages import raw as vintage_raw
from ces_revisions.vintages.months import month_range

SECTOR = "00"
SEASONAL_STATUS = "NSA"
TABLE_SOURCE = "cesnaicsrev"
VINTAGE_FILE_SOURCE = "cesvinall"
EMPLOY_SOURCE = "rtdsm_employ"
ANCHOR_SOURCE = "benchmark_fixtures"
CARRIER_PUBLICATION = "benchmark_final"
# A final benchmark release revises the levels behind its second and third estimates;
# its first estimate is the new month's own change.
CARRIER_MASKED_STAGES = ("S", "T")
# October 2025 had no Employment Situation release, so its EMPLOY vintage repeats
# September's and resolves to the same release (specs/deferred_items.md, plan 5).
REPEATED_EMPLOY_VINTAGE = "rtdsm:EMPLOY25M10"
REPEATED_EMPLOY_ORIGINAL = "rtdsm:EMPLOY25M9"
MASK_REASONS = ("benchmark_carrier", "missing_vintage", "right_censored", "no_anchor")
ARTIFACT_PATHS = {
    "stage_changes": Path("data/panel/stage_changes.parquet"),
    "stage_labels": Path("data/panel/stage_labels.parquet"),
    "table_changes": Path("data/panel/panel.parquet"),
    "release_index": Path("data/panel/release_index.parquet"),
    "benchmark_fixtures": Path("data/operators/benchmark_fixtures.parquet"),
    "publication_calendar": Path("data/annual/panel/publication_calendar.parquet"),
}

_CELL = ["reference_month", "release_stage"]


def _total_nonfarm_closing(frame: pl.DataFrame, seasonal_status: str) -> pl.DataFrame:
    return frame.filter(
        (pl.col("sector") == SECTOR)
        & (pl.col("seasonal_status") == seasonal_status)
        & pl.col("release_stage").is_in(CLOSING_STAGES)
    )


def _source_labels(stage_labels: pl.DataFrame) -> pl.DataFrame:
    """One stage label per closing cell: the label of the source the cell is read from.

    The revision table's before May 2003 and the vintage files' from it, except where a
    vintage-file label is beyond the frontier: the table, which runs to the release
    index's last release, holds those cells.
    """
    closing = _total_nonfarm_closing(stage_labels, SEASONAL_STATUS)
    table = closing.filter(pl.col("source") == TABLE_SOURCE)
    files = closing.filter(
        (pl.col("source") == VINTAGE_FILE_SOURCE)
        & (pl.col("reference_month") >= REGIME_SHIFT_MONTH)
    )
    beyond = files.filter(pl.col("status") == "beyond_frontier").select(_CELL)
    chosen = pl.concat(
        [
            table.filter(pl.col("reference_month") < REGIME_SHIFT_MONTH),
            table.join(beyond, on=_CELL, how="semi"),
            files.filter(pl.col("status") != "beyond_frontier"),
        ]
    ).sort(_CELL)
    if chosen.select(_CELL).is_duplicated().any():
        raise ValueError("a closing cell has stage labels from two sources")
    return chosen


def _final_benchmark_dates(publication_calendar: pl.DataFrame) -> pl.DataFrame:
    return publication_calendar.filter(
        pl.col("publication_kind") == CARRIER_PUBLICATION
    ).select("publication_id", published_date="publication_date")


def carrier_cells(
    stage_labels: pl.DataFrame, publication_calendar: pl.DataFrame
) -> pl.DataFrame:
    """Sector 00 NSA closing-stage labels published in a final benchmark release.

    Every F, S, and T label, from the source its cell is read from, whose published date
    is a final benchmark publication date, with that publication's id. The panel masks
    the S and T rows. A missing vintage has no published date, so the 2003-gap cells
    the March 2002 benchmark release would have carried are not among them.
    """
    return (
        _source_labels(stage_labels)
        .join(
            _final_benchmark_dates(publication_calendar),
            on="published_date",
            how="inner",
        )
        .sort(_CELL)
    )


def _closing_cells(
    stage_changes: pl.DataFrame,
    stage_labels: pl.DataFrame,
    table_changes: pl.DataFrame,
    publication_calendar: pl.DataFrame,
) -> pl.DataFrame:
    values = pl.concat(
        [
            _total_nonfarm_closing(table_changes, SEASONAL_STATUS).select(
                *_CELL,
                source=pl.lit(TABLE_SOURCE),
                value_thousands=pl.col("value_thousands").cast(pl.Float64),
            ),
            _total_nonfarm_closing(stage_changes, SEASONAL_STATUS).select(
                *_CELL,
                source=pl.lit(VINTAGE_FILE_SOURCE),
                value_thousands=pl.col("change_thousands").cast(pl.Float64),
            ),
        ]
    )
    if values.select("source", *_CELL).is_duplicated().any():
        raise ValueError("a source holds two values for one closing cell")
    carried = (
        carrier_cells(stage_labels, publication_calendar)
        .filter(pl.col("release_stage").is_in(CARRIER_MASKED_STAGES))
        .select(*_CELL, carried=pl.lit(True))
    )
    cells = (
        _source_labels(stage_labels)
        .join(values, on=["source", *_CELL], how="left")
        .join(carried, on=_CELL, how="left")
        .with_columns(pl.col("carried").fill_null(False))
    )
    observed = pl.col("status") == "observed"
    mismatched = cells.filter(observed != pl.col("value_thousands").is_not_null())
    if mismatched.height:
        raise ValueError(
            f"{mismatched.height} closing cells disagree with their stage label's status"
        )
    unexpected = set(cells["status"]) - {
        "observed",
        "missing_vintage",
        "right_censored",
    }
    if unexpected:
        raise ValueError(f"unexpected closing-cell statuses {sorted(unexpected)}")
    return cells.select(
        "reference_month",
        column="release_stage",
        source="source",
        value_thousands="value_thousands",
        status="status",
        mask_reason=pl.when(~observed)
        .then("status")
        .when(pl.col("carried"))
        .then(pl.lit("benchmark_carrier"))
        .otherwise(pl.lit(None, dtype=pl.String)),
        published_date="published_date",
    )


def _anchor_rows(
    benchmark_fixtures: pl.DataFrame, release_index: pl.DataFrame
) -> pl.DataFrame:
    """The March anchor of each benchmark year, with earlier reconstructions' offset.

    A reconstruction shifts the vintage basis of every later benchmark release, while
    the wedge anchor removes only its own year's, so a year's anchor sits above the
    latent path by the March reconstruction terms of every earlier year.
    """
    marches = (
        benchmark_fixtures.filter(
            (pl.col("sector") == SECTOR) & (pl.col("wedge_weight") == 1.0)
        )
        .sort("benchmark_year")
        .with_columns(
            anchor_offset=pl.col("reconstruction_term_thousands")
            .cum_sum()
            .shift(1, fill_value=0.0)
        )
    )
    if marches["benchmark_year"].to_list() != list(ANCHOR_YEARS):
        raise ValueError(f"expected one March anchor for each year of {ANCHOR_YEARS}")
    if (marches["reference_month"].dt.month() != 3).any() or (
        marches["reference_month"].dt.year() != marches["benchmark_year"]
    ).any():
        raise ValueError("every anchor must be its benchmark year's March")
    releases = release_index.select(
        benchmark_release_month="reference_month", published_date="published_date"
    )
    return marches.join(releases, on="benchmark_release_month", how="left").select(
        "reference_month",
        "anchor_offset",
        column=pl.lit("anchor"),
        source=pl.lit(ANCHOR_SOURCE),
        value_thousands="wedge_anchor_thousands",
        status=pl.lit("anchor"),
        mask_reason=pl.lit(None, dtype=pl.String),
        published_date="published_date",
    )


def build_aggregate_panel(
    stage_changes: pl.DataFrame,
    stage_labels: pl.DataFrame,
    table_changes: pl.DataFrame,
    release_index: pl.DataFrame,
    benchmark_fixtures: pl.DataFrame,
    publication_calendar: pl.DataFrame,
) -> AggregatePanel:
    """The pilot panel over every reference month from January 1979 to the index's last."""
    months = month_range(FIRST_MONTH, release_index["reference_month"].max())
    grid = pl.DataFrame(
        {
            "reference_month": [month for month in months for _ in COLUMNS],
            "column": [column for _ in months for column in COLUMNS],
        },
        schema={"reference_month": pl.Date, "column": pl.String},
    )
    anchors = _anchor_rows(benchmark_fixtures, release_index)
    found = pl.concat(
        [
            _closing_cells(
                stage_changes, stage_labels, table_changes, publication_calendar
            ),
            anchors.drop("anchor_offset"),
        ]
    )
    cells = (
        grid.join(
            found, on=["reference_month", "column"], how="left", maintain_order="left"
        )
        .with_columns(
            status=pl.when(pl.col("column") == "anchor")
            .then(pl.col("status").fill_null("no_anchor"))
            .otherwise("status"),
            mask_reason=pl.when(
                (pl.col("column") == "anchor") & pl.col("status").is_null()
            )
            .then(pl.lit("no_anchor"))
            .otherwise("mask_reason"),
        )
        .select(list(CELL_SCHEMA))
        .cast(CELL_SCHEMA)
    )
    if cells.height != grid.height or cells["status"].is_null().any():
        raise ValueError("every panel cell needs exactly one source row")
    y = np.where(
        cells["mask_reason"].is_null().to_numpy(),
        cells["value_thousands"].fill_null(np.nan).to_numpy(),
        np.nan,
    ).reshape(len(months), len(COLUMNS))
    offsets = dict(
        zip(anchors["reference_month"], anchors["anchor_offset"], strict=True)
    )
    return AggregatePanel(
        reference_months=tuple(months),
        y=y,
        anchor_offset=np.array([offsets.get(month, 0.0) for month in months]),
        pre_2003=np.array([month < REGIME_SHIFT_MONTH for month in months]),
        crisis=np.array([month in CRISIS_MONTHS for month in months]),
        benchmark_year=np.array(
            [benchmark_year_of(month) for month in months], dtype=np.int64
        ),
        cells=cells,
    )


def read_artifacts(root: Path = vintage_raw.ROOT) -> dict[str, pl.DataFrame]:
    """The six inputs of build_aggregate_panel, read from the built artifacts under root.

    The revision table's changes are the long panel's cesnaicsrev rows.
    """
    frames = {
        name: pl.read_parquet(root / path)
        for name, path in ARTIFACT_PATHS.items()
        if name != "table_changes"
    }
    frames["table_changes"] = (
        pl.scan_parquet(root / ARTIFACT_PATHS["table_changes"])
        .filter(pl.col("source") == TABLE_SOURCE)
        .collect()
    )
    return {name: frames[name] for name in ARTIFACT_PATHS}


def employ_without_repeats(long_panel: pl.DataFrame) -> pl.DataFrame:
    """The long panel's EMPLOY rows without the repeated October 2025 vintage."""
    employ = long_panel.filter(pl.col("source") == EMPLOY_SOURCE)
    columns = ["reference_month", "release_month", "value_thousands"]
    repeat = employ.filter(pl.col("vintage_id") == REPEATED_EMPLOY_VINTAGE).select(
        columns
    )
    original = employ.filter(pl.col("vintage_id") == REPEATED_EMPLOY_ORIGINAL).select(
        columns
    )
    if repeat.is_empty() or not repeat.sort(columns).equals(original.sort(columns)):
        raise ValueError(
            f"{REPEATED_EMPLOY_VINTAGE} is not an exact repeat of "
            f"{REPEATED_EMPLOY_ORIGINAL}"
        )
    kept = employ.filter(pl.col("vintage_id") != REPEATED_EMPLOY_VINTAGE)
    doubled = (
        kept.group_by("release_month")
        .agg(pl.col("vintage_id").n_unique())
        .filter(pl.col("vintage_id") > 1)
    )
    if doubled.height:
        raise ValueError(
            f"release months still mapping to two EMPLOY vintages: "
            f"{sorted(doubled['release_month'])}"
        )
    return kept


def _months_after(later: str, earlier: str) -> pl.Expr:
    return (pl.col(later).dt.year() - pl.col(earlier).dt.year()) * 12 + (
        pl.col(later).dt.month().cast(pl.Int32)
        - pl.col(earlier).dt.month().cast(pl.Int32)
    )


def employ_table_disagreements(
    long_panel: pl.DataFrame, table_changes: pl.DataFrame
) -> pl.DataFrame:
    """Cells where EMPLOY's same-vintage SA F, S, or T change differs from the table's.

    An EMPLOY vintage's change is a month's level minus the prior month's in that
    vintage; its stage counts the months from the reference month to the vintage's
    release. Only cells both sources hold, from the same release, are compared.
    """
    employ = employ_without_repeats(long_panel)
    prior = employ.select(
        "vintage_id",
        reference_month=pl.col("reference_month").dt.offset_by("1mo"),
        prior_thousands="value_thousands",
    )
    stages = dict(enumerate(CLOSING_STAGES))
    changes = (
        employ.join(prior, on=["vintage_id", "reference_month"], how="inner")
        .with_columns(lag=_months_after("release_month", "reference_month"))
        .filter(pl.col("lag").is_in(list(stages)))
        .select(
            "reference_month",
            "release_month",
            "vintage_id",
            release_stage=pl.col("lag").replace_strict(stages, return_dtype=pl.String),
            employ_thousands=pl.col("value_thousands") - pl.col("prior_thousands"),
        )
    )
    table = (
        _total_nonfarm_closing(table_changes, "SA")
        .filter(pl.col("value_thousands").is_not_null())
        .select(
            "reference_month",
            "release_month",
            "release_stage",
            table_thousands="value_thousands",
        )
    )
    return (
        changes.join(
            table, on=["reference_month", "release_stage", "release_month"], how="inner"
        )
        .filter(pl.col("employ_thousands") != pl.col("table_thousands"))
        .select(
            "reference_month",
            "release_stage",
            "vintage_id",
            "employ_thousands",
            "table_thousands",
        )
        .sort("reference_month", "release_stage")
    )
