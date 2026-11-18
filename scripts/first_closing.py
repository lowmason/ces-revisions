"""Roadmap Stage 9: the window to first closing and the collection rate C1, against revision size.

`check` (offline) reads the built data/panel/collection_rates.parquet and compares every C1, C2
and C3 value of each archived registry page with LABSTAT's value (spec D8). It prints each
copy's counts and every value that differs, and exits 1 on a difference the owner has not ruled
on. `report` (offline) runs the same check, then rewrites the generated blocks of the Stage 9
findings file from the built data/panel/. Build data/panel/ first with
`uv run python scripts/vintage_sources.py build`.
"""

import argparse
import sys
import time
from datetime import date

import polars as pl
from archive_inventory import replace_generated

from ces_revisions.collection_window import absolute_revisions
from ces_revisions.first_closing import (
    MAIN,
    PANDEMIC,
    REVISION_COLUMNS,
    final_week_holidays,
    holiday_closings,
    lapse_releases,
    month_sample,
    pandemic_by_window,
    rate_quintiles,
    rate_regression,
    rates_by_window,
    revision_regression,
    revision_sample,
    revisions_by_quintile,
    window_regressions,
)
from ces_revisions.vintages.collection_rates import (
    ARCHIVE_YEARS,
    RECON_COPY,
    archive_comparison,
    archive_differences,
)
from ces_revisions.vintages.raw import PANEL_DIR, ROOT

FINDINGS = ROOT / "specs" / "ces-revision-drivers-stage-9-findings.md"
RATES = "collection_rates.parquet"
INPUTS = ("collection_days.parquet", "panel.parquet", RATES)
# Each archived value that differs from LABSTAT's, as (series, month, copy, copy value, LABSTAT
# value). The owner ruled on 2026-11-18 that the fits use LABSTAT's value (spec D8).
ACCEPTED_DIFFERENCES = frozenset(
    {("CEU00000000C1", date(2012, 6, 1), "20121114143627", 64.6, 72.9)}
)
ARROWS = {"F_S": "F→S", "F_T": "F→T"}
REVISION_NAMES = [f"{column[:-4]} {ARROWS[column[-3:]]}" for column in REVISION_COLUMNS]
NO_VALUE = "–"


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


def _missing(names: tuple[str, ...]) -> bool:
    """Whether a built input is missing, after naming the missing ones and the build command."""
    missing = [name for name in names if not (PANEL_DIR / name).exists()]
    if missing:
        print(
            f"missing {', '.join(missing)} in {PANEL_DIR}: run "
            "`uv run python scripts/vintage_sources.py build` first",
            file=sys.stderr,
        )
    return bool(missing)


def check() -> int:
    if _missing((RATES,)):
        return 1
    return check_rates(pl.read_parquet(PANEL_DIR / RATES))


def _table(header: list[str], rows: list[list[str]], *, left: int = 1) -> str:
    """A Markdown table whose first `left` columns align left and the rest right."""
    lines = [
        f"| {' | '.join(header)} |",
        "|" + "---|" * left + "---:|" * (len(header) - left),
    ]
    lines += [f"| {' | '.join(row)} |" for row in rows]
    return "\n".join(lines)


def _one(value: float | None) -> str:
    return NO_VALUE if value is None else f"{value:.1f}"


def _interval(row: dict) -> str:
    return f"{row['contrast_low']:.1f} to {row['contrast_high']:.1f}"


def _fit_cells(row: dict) -> list[str]:
    return [f"{row['coefficient']:.2f}", f"{row['se']:.2f}", _interval(row)]


def render_copies(rates: pl.DataFrame) -> str:
    """Each archive year's copy and the recon copy, with its values compared with LABSTAT."""
    comparison = {row[0]: row[1:] for row in archive_comparison(rates).iter_rows()}
    by_year = {int(source[:4]): source for source in comparison if source != RECON_COPY}
    rows = []
    for year, source in [
        *((year, by_year.get(year)) for year in ARCHIVE_YEARS),
        (int(RECON_COPY[:4]), RECON_COPY),
    ]:
        if source is None:
            rows.append([str(year), "no valid copy", NO_VALUE, NO_VALUE, NO_VALUE])
        else:
            rows.append([str(year), source, *map(str, comparison[source])])
    totals = [
        sum(counts[index] for counts in comparison.values()) for index in range(3)
    ]
    rows.append(["All", f"{len(comparison)} copies", *map(str, totals)])
    header = ["Year", "Copy", "Values compared", "Differing", "Only in the copy"]
    return _table(header, rows, left=2)


def render_holidays(months: pl.DataFrame) -> str:
    """The closing days on a federal holiday, then the final-week holidays by calendar month."""
    closings = _table(
        ["Reference month", "Release", "Closing day"],
        [
            [f"{month:%Y-%m}", str(release), str(closing)]
            for month, release, closing in holiday_closings(months).iter_rows()
        ],
        left=3,
    )
    weeks = final_week_holidays(months)
    rows = [
        [date(2000, month, 1).strftime("%B"), str(releases), str(with_holiday)]
        for month, releases, with_holiday in weeks.iter_rows()
    ]
    rows.append(["All", str(weeks["releases"].sum()), str(weeks["with_holiday"].sum())])
    counts = _table(["Calendar month", "Releases", "With a final-week holiday"], rows)
    return f"{closings}\n\n{counts}"


def render_rate_by_window(months: pl.DataFrame) -> str:
    """The main-period months and their mean C1 for each window value, with an All row."""
    rows = [
        [str(window), str(n), _one(mean)]
        for window, n, mean in rates_by_window(months).iter_rows()
    ]
    main = months.filter(pl.col("period") == MAIN)
    rows.append(["All", str(main.height), _one(main["c1"].mean())])
    return _table(["`W_t`", "Months", "Mean C1"], rows)


def render_rate_regression(months: pl.DataFrame) -> str:
    """Part (a): the window coefficient of C1, and the change from 8 to 15 days."""
    row = rate_regression(months).row(0, named=True)
    header = ["n", "Years", "`W_t` coefficient", "SE", "8→15 days, 95% CI"]
    return _table(
        header, [[str(row["n"]), str(row["clusters"]), *_fit_cells(row)]], left=0
    )


def render_revision_by_rate(revisions: pl.DataFrame, quintiles: pl.DataFrame) -> str:
    """The mean absolute revision for each quintile of C1 less its reference year's mean."""
    table = revisions_by_quintile(revisions, quintiles)
    rows = []
    for (quintile,), group in table.group_by("quintile", maintain_order=True):
        first = group.row(0, named=True)
        means = {
            f"{row['seasonal_status']}_{row['transition']}": row[
                "mean_absolute_revision"
            ]
            for row in group.iter_rows(named=True)
        }
        rows.append(
            [
                str(quintile),
                str(first["months"]),
                f"{first['deviation_low']:.1f} to {first['deviation_high']:.1f}",
                *(_one(means.get(column)) for column in REVISION_COLUMNS),
            ]
        )
    return _table(["Quintile", "Months", "Deviation", *REVISION_NAMES], rows)


def render_revision_regression(revisions: pl.DataFrame) -> str:
    """Part (b): the C1 coefficient of each fit, and the change for 10 points of C1."""
    header = ["Status", "Transition", "n", "Years", "C1 coefficient", "SE"]
    rows = [
        [
            row["seasonal_status"],
            ARROWS[row["transition"]],
            str(row["n"]),
            str(row["clusters"]),
            *_fit_cells(row),
        ]
        for row in revision_regression(revisions).iter_rows(named=True)
    ]
    return _table([*header, "+10 points, 95% CI"], rows, left=2)


def render_window_regression(revisions: pl.DataFrame) -> str:
    """O4: for each status and transition, the `D_t` and `W_t` fits side by side."""
    fits = window_regressions(revisions)
    rows = []
    for (status, transition), group in fits.group_by(
        "seasonal_status", "transition", maintain_order=True
    ):
        by_window = {row["window"]: row for row in group.iter_rows(named=True)}
        days, first = by_window["collection_days"], by_window["first_closing_days"]
        rows.append(
            [
                status,
                ARROWS[transition],
                str(days["n"]),
                str(days["clusters"]),
                *_fit_cells(days),
                *_fit_cells(first),
            ]
        )
    header = [
        "Status",
        "Transition",
        "n",
        "Years",
        "`D_t` coefficient",
        "SE",
        "`D_t` 7 days, 95% CI",
        "`W_t` coefficient",
        "SE",
        "`W_t` 7 days, 95% CI",
    ]
    return _table(header, rows, left=2)


def render_lapse(months: pl.DataFrame, revisions: pl.DataFrame) -> str:
    """The four lapse-delayed releases: dates, windows, C1 and its footnote, and revisions."""
    rows = []
    for row in lapse_releases(months, revisions).iter_rows(named=True):
        footnote = f" ({row['c1_footnote']})" if row["c1_footnote"] else ""
        rows.append(
            [
                f"{row['reference_month']:%Y-%m}",
                str(row["release_date"]),
                str(row["first_closing_date"]),
                str(row["collection_days"]),
                str(row["first_closing_days"]),
                f"{_one(row['c1'])}{footnote}",
                *(
                    NO_VALUE if row[column] is None else str(row[column])
                    for column in REVISION_COLUMNS
                ),
            ]
        )
    header = ["Reference month", "Release", "Closing day", "`D_t`", "`W_t`", "C1"]
    return _table([*header, *REVISION_NAMES], rows, left=3)


def render_pandemic(months: pl.DataFrame, revisions: pl.DataFrame) -> str:
    """March 2020 to December 2021: months, mean C1 and mean revisions per window, and All."""
    rows = [
        [
            str(row["first_closing_days"]),
            str(row["months"]),
            _one(row["mean_c1"]),
            *(_one(row[column]) for column in REVISION_COLUMNS),
        ]
        for row in pandemic_by_window(months, revisions).iter_rows(named=True)
    ]
    pandemic = months.filter(pl.col("period") == PANDEMIC)
    means = {
        f"{status}_{transition}": mean
        for status, transition, mean in revisions.filter(pl.col("period") == PANDEMIC)
        .group_by("seasonal_status", "transition")
        .agg(pl.col("absolute_revision").mean())
        .iter_rows()
    }
    rows.append(
        [
            "All",
            str(pandemic.height),
            _one(pandemic["c1"].mean()),
            *(_one(means.get(column)) for column in REVISION_COLUMNS),
        ]
    )
    return _table(["`W_t`", "Months", "Mean C1", *REVISION_NAMES], rows)


def report() -> int:
    started = time.perf_counter()
    if _missing(INPUTS):
        return 1
    rates = pl.read_parquet(PANEL_DIR / RATES)
    if check_rates(rates):
        print(
            "report: an archived value differs from LABSTAT without the owner's ruling; "
            "the findings are unchanged",
            file=sys.stderr,
        )
        return 1
    days = pl.read_parquet(PANEL_DIR / "collection_days.parquet")
    months = month_sample(days, rates)
    revisions = revision_sample(
        absolute_revisions(pl.read_parquet(PANEL_DIR / "panel.parquet"), days), months
    )
    blocks = {
        "stage-9-copies": render_copies(rates),
        "stage-9-holidays": render_holidays(months),
        "stage-9-rate-by-window": render_rate_by_window(months),
        "stage-9-rate-regression": render_rate_regression(months),
        "stage-9-revision-by-rate": render_revision_by_rate(
            revisions, rate_quintiles(months)
        ),
        "stage-9-revision-regression": render_revision_regression(revisions),
        "stage-9-window-regression": render_window_regression(revisions),
        "stage-9-lapse": render_lapse(months, revisions),
        "stage-9-pandemic": render_pandemic(months, revisions),
    }
    document = FINDINGS.read_text(encoding="utf-8")
    for name, body in blocks.items():
        # Blank lines keep each table a block of its own beside the HTML-comment markers.
        document = replace_generated(document, name, f"\n{body}\n\n")
    FINDINGS.write_text(document, encoding="utf-8")
    main = revisions.filter(pl.col("period") == MAIN)
    elapsed = time.perf_counter() - started
    print(
        f"report: {months.filter(pl.col('period') == MAIN).height} part (a) months, "
        f"{main.height} part (b) revisions, {elapsed:.1f} s"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["check", "report"])
    command = parser.parse_args(argv).command
    return check() if command == "check" else report()


if __name__ == "__main__":
    raise SystemExit(main())
