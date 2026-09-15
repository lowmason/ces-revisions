"""One cached Stage 4 build shared by contract tests."""

from functools import cache

import polars as pl

from ces_revisions.annual.build import AnnualBuild, build


@cache
def result() -> AnnualBuild:
    return build()


def raw_values() -> pl.DataFrame:
    return result().raw_values


def artifact(name: str) -> pl.DataFrame:
    return getattr(result(), name)
