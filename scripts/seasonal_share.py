"""Outcome O9: the share of SA revision variance that comes from the change in seasonal adjustment.

`report` (offline) reads data/panel/accounting_decomposition.parquet and rewrites the generated
blocks of the O9 findings file: one table of the identity's terms and the three shares for each
transition, and one table of how many shares fall outside [0, 1]. Build data/panel/ first with
`uv run python scripts/vintage_sources.py build`.
"""

import argparse
import sys
import time

import polars as pl
from archive_inventory import replace_generated

from ces_revisions.seasonal_share import REGIME_LABELS, seasonal_shares
from ces_revisions.vintages.accounting import TRANSITIONS
from ces_revisions.vintages.raw import PANEL_DIR, ROOT

FINDINGS = ROOT / "specs" / "ces-revision-drivers-stage-3-findings.md"
INPUT = "accounting_decomposition.parquet"
# The titles of BLS's sample coverage tables (`ces_revisions.annual.sample`), which a test ties to
# each code.
SECTOR_TITLES = {
    "00": "Total nonfarm",
    "10": "Mining and logging",
    "20": "Construction",
    "30": "Manufacturing",
    "40": "Trade, transportation, and utilities",
    "50": "Information",
    "55": "Financial activities",
    "60": "Professional and business services",
    "65": "Education and health services",
    "70": "Leisure and hospitality",
    "80": "Other services",
    "90": "Government",
}
ARROWS = {f"{earlier}_{later}": f"{earlier}→{later}" for earlier, later in TRANSITIONS}
SHARES = (
    ("Ratio", "share_ratio"),
    ("Adjustment", "share_adjustment"),
    ("NSA", "share_nsa"),
)
BOUNDS_BLOCK = "stage-3-bounds"


def _block_name(transition: str) -> str:
    return "stage-3-" + transition.lower().replace("_", "-")


def _table(header: list[str], rows: list[list[str]], *, left: int) -> str:
    """A Markdown table whose first `left` columns align left and the others right."""
    rule = "|" + "---|" * left + "---:|" * (len(header) - left)
    lines = [f"| {' | '.join(header)} |", rule]
    lines.extend(f"| {' | '.join(cells)} |" for cells in rows)
    return "\n".join(lines)


def render_transition(shares: pl.DataFrame, transition: str) -> str:
    """One row per sector and regime of a transition: every identity term, then the shares.

    Variances and the covariance are in thousands squared. The identity gap prints in scientific
    notation, because at about 1e-11 it would read as 0.00.
    """
    header = [
        "Sector",
        "Source",
        "Regime",
        "Months",
        "Var SA",
        "Var NSA",
        "Var adjustment change",
        "Cov(NSA, adjustment change)",
        "Ratio share",
        "Adjustment share",
        "NSA share",
        "Identity gap",
    ]
    rows = [
        [
            f"{row['sector']} {SECTOR_TITLES[row['sector']]}",
            row["source"],
            REGIME_LABELS[row["regime"]],
            str(row["months"]),
            f"{row['var_revision_sa']:.1f}",
            f"{row['var_revision_nsa']:.1f}",
            f"{row['var_adjustment_change']:.1f}",
            f"{row['cov_revision_nsa_adjustment_change']:.1f}",
            f"{row['share_ratio']:.2f}",
            f"{row['share_adjustment']:.2f}",
            f"{row['share_nsa']:.2f}",
            f"{row['variance_identity_gap']:.1e}",
        ]
        for row in shares.filter(pl.col("transition") == transition).iter_rows(
            named=True
        )
    ]
    return _table(header, rows, left=3)


def render_bounds(shares: pl.DataFrame) -> str:
    """For each share and transition, how many rows fall below 0 or above 1, and the extremes."""
    header = ["Share", "Transition", "Rows", "Below 0", "Above 1", "Min", "Max"]
    groups = [
        (arrow, shares.filter(pl.col("transition") == transition))
        for transition, arrow in ARROWS.items()
    ] + [("All", shares)]
    rows = []
    for label, column in SHARES:
        for name, group in groups:
            values = group[column]
            rows.append(
                [
                    label,
                    name,
                    str(group.height),
                    str((values < 0).sum()),
                    str((values > 1).sum()),
                    f"{values.min():.2f}",
                    f"{values.max():.2f}",
                ]
            )
    return _table(header, rows, left=2)


def report() -> int:
    started = time.perf_counter()
    if not (PANEL_DIR / INPUT).exists():
        print(
            f"missing {INPUT} in {PANEL_DIR}: run "
            "`uv run python scripts/vintage_sources.py build` first",
            file=sys.stderr,
        )
        return 1
    shares = seasonal_shares(pl.read_parquet(PANEL_DIR / INPUT))
    blocks = {
        _block_name(transition): render_transition(shares, transition)
        for transition in ARROWS
    }
    blocks[BOUNDS_BLOCK] = render_bounds(shares)
    document = FINDINGS.read_text(encoding="utf-8")
    for name, body in blocks.items():
        # Blank lines keep each table a block of its own beside the HTML-comment markers.
        document = replace_generated(document, name, f"\n{body}\n\n")
    FINDINGS.write_text(document, encoding="utf-8")
    elapsed = time.perf_counter() - started
    print(f"report: {shares.height} rows, {elapsed:.1f} s")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["report"])
    parser.parse_args(argv)
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
