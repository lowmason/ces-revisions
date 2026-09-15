"""Stage 3 artifacts, each built at most once per test session from the sources in data/raw/."""

from functools import cache

import polars as pl

from ces_revisions.vintages import (
    differencing,
    panel,
    raw,
    release_index,
    revision_table,
    stages,
)
from ces_revisions.vintages.months import add_months


@cache
def raw_values() -> pl.DataFrame:
    return raw.raw_values()


@cache
def index() -> pl.DataFrame:
    return release_index.build_release_index()


@cache
def table_cells() -> pl.DataFrame:
    return revision_table.monthly_cells(raw_values())


@cache
def table_estimates() -> pl.DataFrame:
    return revision_table.estimates(table_cells())


@cache
def levels() -> pl.DataFrame:
    return panel.vintage_file_levels(raw_values(), index())


@cache
def rtdsm() -> pl.DataFrame:
    return panel.rtdsm_levels(raw_values(), index())


@cache
def table_changes() -> pl.DataFrame:
    return panel.revision_table_changes(raw_values(), index())


@cache
def labels() -> pl.DataFrame:
    return stages.build_stage_labels(
        levels(), table_changes(), index(), raw.read_vintage_comments()
    )


@cache
def long_panel() -> pl.DataFrame:
    return panel.assemble_panel(levels(), rtdsm(), table_changes(), labels())


@cache
def stage_changes() -> pl.DataFrame:
    first = add_months(stages.FIRST_SECTOR_MONTH, -1)
    within = differencing.same_release_changes(
        levels().filter(pl.col("reference_month") >= first)
    )
    return differencing.stage_changes(within, labels())
