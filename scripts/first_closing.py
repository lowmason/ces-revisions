"""Roadmap Stage 9: the window to first closing and the collection rate C1, against revision size.

`check` (offline) reads the built data/panel/collection_rates.parquet and compares every C1, C2
and C3 value of each archived registry page with LABSTAT's value (spec D8). It prints each
copy's counts and every value that differs, and exits 1 on a difference the owner has not ruled
on. Build data/panel/ first with `uv run python scripts/vintage_sources.py build`.
"""

import argparse
import sys
from datetime import date

import polars as pl

from ces_revisions.vintages.collection_rates import (
    archive_comparison,
    archive_differences,
)
from ces_revisions.vintages.raw import PANEL_DIR

RATES = "collection_rates.parquet"
# Each archived value that differs from LABSTAT's, as (series, month, copy, copy value, LABSTAT
# value). The owner ruled on 2026-11-18 that the fits use LABSTAT's value (spec D8).
ACCEPTED_DIFFERENCES = frozenset(
    {("CEU00000000C1", date(2012, 6, 1), "20121114143627", 64.6, 72.9)}
)


def check_rates(rates: pl.DataFrame) -> int:
    """Print each copy's comparison with LABSTAT and each difference; 1 if one is not accepted."""
    for source, compared, differing, only_in_copy in archive_comparison(
        rates
    ).iter_rows():
        print(
            f"{source}: {compared} compared, {differing} differ, "
            f"{only_in_copy} only in the copy"
        )
    unaccepted = 0
    for series, month, source, copy_value, labstat_value in archive_differences(
        rates
    ).iter_rows():
        key = (series, month, source, round(copy_value, 1), round(labstat_value, 1))
        accepted = key in ACCEPTED_DIFFERENCES
        unaccepted += not accepted
        print(
            f"{series} {month:%Y-%m} {source}: {copy_value:.1f} against LABSTAT "
            f"{labstat_value:.1f}, {'accepted' if accepted else 'not accepted'}"
        )
    return 1 if unaccepted else 0


def check() -> int:
    path = PANEL_DIR / RATES
    if not path.exists():
        print(
            f"missing {RATES} in {PANEL_DIR}: run "
            "`uv run python scripts/vintage_sources.py build` first",
            file=sys.stderr,
        )
        return 1
    return check_rates(pl.read_parquet(path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["check"])
    parser.parse_args(argv)
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
