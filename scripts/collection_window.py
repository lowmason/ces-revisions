"""Roadmap Stage 1: revision size against the collection window, for total nonfarm.

`report` (offline) reads the built data/panel/ and rewrites the generated blocks of the Stage 1
findings file: the mean absolute revisions for each day count `D_t`, in the main period and in
March 2020 to December 2021, and the regression's coefficients. Build data/panel/ first with
`uv run python scripts/vintage_sources.py build`.
"""

import argparse
import sys
import time

import polars as pl
from archive_inventory import replace_generated

from ces_revisions.collection_window import (
    TRANSITIONS,
    absolute_revisions,
    days_regression,
    revisions_by_days,
)
from ces_revisions.vintages.raw import PANEL_DIR, ROOT

FINDINGS = ROOT / "specs" / "ces-revision-drivers-stage-1-findings.md"
INPUTS = ("panel.parquet", "collection_days.parquet")
STATUSES = ("NSA", "SA")
COLUMNS = [(status, transition) for status in STATUSES for transition in TRANSITIONS]
ARROWS = {"F_S": "F→S", "F_T": "F→T"}
NO_VALUE = "–"


def _count(cells: dict, days: int | None, transition: str) -> int:
    """The number of revisions in a row, which NSA and SA share."""
    found = {
        cells[key]["n"] for key in cells if key[0] == days and key[2] == transition
    }
    if len(found) > 1:
        raise ValueError(f"NSA and SA counts differ at D_t={days}, {transition}")
    return found.pop() if found else 0


def _row(label: str, cells: dict, days: int | None) -> str:
    lapse = max(cells[key]["lapse_n"] for key in cells if key[0] == days)
    means = [
        f"{cells[(days, status, transition)]['mean_absolute_revision']:.1f}"
        if (days, status, transition) in cells
        else NO_VALUE
        for status, transition in COLUMNS
    ]
    counts = [str(_count(cells, days, transition)) for transition in TRANSITIONS]
    return f"| {' | '.join([label, *counts, str(lapse), *means])} |"


def render_by_days(revisions: pl.DataFrame, period: str) -> str:
    """The mean absolute revision for each `D_t` value of a period, with an All row."""
    rows = revisions.filter(pl.col("period") == period)
    by_days = revisions_by_days(rows)
    overall = revisions_by_days(rows.with_columns(collection_days=pl.lit(None)))
    cells = {
        (row["collection_days"], row["seasonal_status"], row["transition"]): row
        for row in pl.concat([by_days, overall], how="vertical_relaxed").iter_rows(
            named=True
        )
    }
    names = [f"{status} {ARROWS[transition]}" for status, transition in COLUMNS]
    header = ["`D_t`", "n F→S", "n F→T", "Lapse-delayed", *names]
    lines = [f"| {' | '.join(header)} |", "|" + "---:|" * len(header)]
    for days in sorted(by_days["collection_days"].unique()):
        lines.append(_row(str(days), cells, days))
    lines.append(_row("All", cells, None))
    return "\n".join(lines)


def render_regression(revisions: pl.DataFrame) -> str:
    """The `D_t` and lapse coefficients of the four fits, in thousands."""
    header = [
        "Status",
        "Transition",
        "n",
        "Years",
        "`D_t` coefficient",
        "SE",
        "Lapse coefficient",
        "SE",
    ]
    lines = [f"| {' | '.join(header)} |", "|---|---|" + "---:|" * (len(header) - 2)]
    for row in days_regression(revisions).iter_rows(named=True):
        cells = [
            row["seasonal_status"],
            ARROWS[row["transition"]],
            str(row["n"]),
            str(row["clusters"]),
            *(
                f"{row[name]:.2f}"
                for name in ("days_coefficient", "days_se", "lapse_coefficient")
            ),
            f"{row['lapse_se']:.2f}",
        ]
        lines.append(f"| {' | '.join(cells)} |")
    return "\n".join(lines)


def report() -> int:
    started = time.perf_counter()
    missing = [name for name in INPUTS if not (PANEL_DIR / name).exists()]
    if missing:
        print(
            f"missing {', '.join(missing)} in {PANEL_DIR}: run "
            "`uv run python scripts/vintage_sources.py build` first",
            file=sys.stderr,
        )
        return 1
    revisions = absolute_revisions(
        pl.read_parquet(PANEL_DIR / "panel.parquet"),
        pl.read_parquet(PANEL_DIR / "collection_days.parquet"),
    )
    blocks = {
        "stage-1-main": render_by_days(revisions, "main"),
        "stage-1-pandemic": render_by_days(revisions, "pandemic"),
        "stage-1-regression": render_regression(revisions),
    }
    document = FINDINGS.read_text(encoding="utf-8")
    for name, body in blocks.items():
        # Blank lines keep each table a block of its own beside the HTML-comment markers.
        document = replace_generated(document, name, f"\n{body}\n\n")
    FINDINGS.write_text(document, encoding="utf-8")
    elapsed = time.perf_counter() - started
    print(f"report: {revisions.height} revisions, {elapsed:.1f} s")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["report"])
    parser.parse_args(argv)
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
