"""One cached Stage 5 build shared by operator artifact tests."""

from functools import cache

import annual_data
import polars as pl
import vintage_data

from ces_revisions.operators.build import OperatorBuild, assemble


@cache
def result() -> OperatorBuild:
    return assemble(
        vintage_data.levels(),
        vintage_data.labels(),
        annual_data.artifact("benchmarks"),
        annual_data.artifact("reconstruction_events"),
        annual_data.artifact("birth_death"),
    )


def artifact(name: str) -> pl.DataFrame:
    return getattr(result(), name)
