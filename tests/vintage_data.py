"""Stage 3 artifacts, each built at most once per test session from the sources in data/raw/."""

from functools import cache

import polars as pl

from ces_revisions.vintages import raw


@cache
def raw_values() -> pl.DataFrame:
    return raw.raw_values()
