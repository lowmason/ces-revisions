"""Stage 3 artifacts, each built at most once per test session from the sources in data/raw/."""

from functools import cache

import polars as pl

from ces_revisions.vintages import (
    panel,
    raw,
    release_index,
    revision_table,
)


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
