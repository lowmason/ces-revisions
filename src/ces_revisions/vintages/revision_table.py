"""BLS's revision table as data: estimates with their markers, and the averages it publishes.

For each month since January 1979 the page gives total nonfarm over-the-month changes at the
first, second, and third estimates, seasonally adjusted and not, their three pairwise
revisions, and each year's mean and mean absolute revision; two summary tables average the
revisions over the periods its footnote ** defines. Markers carry the page's footnotes: `A` for
values the 2025 lapse left unavailable, `B` for October 2025's first published estimates, which
the page counts as second estimates, `c` for corrected averages, and `NA` for the 2003 redesign.
"""

import re
from dataclasses import dataclass
from datetime import date
from fractions import Fraction

import polars as pl

STAGE_OF_ESTIMATE = {"1st": "F", "2nd": "S", "3rd": "T"}
# Each revision column and the later and earlier estimates it compares.
REVISION_TERMS = {
    "2nd_minus_1st": ("2nd", "1st"),
    "3rd_minus_2nd": ("3rd", "2nd"),
    "3rd_minus_1st": ("3rd", "1st"),
}
# The page prints zeros for March 2003's third estimate and April 2003's second and third, and
# NA in every revision they enter; a zero estimate whose every revision is NA is a placeholder.
# April 2003's first estimate also has only NA revisions, but it is a published change.
REVISIONS_OF_ESTIMATE = {
    estimate: tuple(
        column for column, terms in REVISION_TERMS.items() if estimate in terms
    )
    for estimate in STAGE_OF_ESTIMATE
}
# The periods of the page's footnote **. The later period, and the total, run through the
# latest month each revision is published for.
SUMMARY_PERIODS = {
    "1979 - 2003": {
        "2nd_minus_1st": (date(1979, 1, 1), date(2003, 3, 1)),
        "3rd_minus_2nd": (date(1979, 1, 1), date(2003, 2, 1)),
        "3rd_minus_1st": (date(1979, 1, 1), date(2003, 2, 1)),
    },
    "2003 - present": dict.fromkeys(REVISION_TERMS, (date(2003, 5, 1), None)),
    "Total All Periods": dict.fromkeys(REVISION_TERMS, (date(1979, 1, 1), None)),
}


@dataclass(frozen=True)
class Cell:
    value: int | None
    marker: str | None


def parse_cell(text: str) -> Cell:
    if text == "":
        return Cell(None, "blank")
    if text == "-(A)":
        return Cell(None, "A")
    if re.fullmatch(r"NA(?:\*\*\*)?", text):
        return Cell(None, "NA")
    match = re.fullmatch(r"(-?\d+)(?:\s*\((B|c)\))?", text)
    if match is None:
        raise ValueError(f"unrecognized revision table cell {text!r}")
    return Cell(int(match.group(1)), match.group(2))


def monthly_cells(raw: pl.DataFrame) -> pl.DataFrame:
    """One row per month and column of the page, with the parsed value and marker."""
    frame = raw.filter(
        (pl.col("source") == "cesnaicsrev")
        & pl.col("row_key").str.contains(r"^\d{4}-\d{2}$")
    )
    parsed = [parse_cell(text) for text in frame["text"]]
    return frame.select(
        "cell_id",
        reference_month=(pl.col("row_key") + "-01").str.to_date("%Y-%m-%d"),
        seasonal_status=pl.col("column_key")
        .str.extract(r"^(n?sa)_")
        .str.to_uppercase(),
        column=pl.col("column_key").str.extract(r"^n?sa_(.+)$"),
        text="text",
        value=pl.Series([cell.value for cell in parsed], dtype=pl.Int64),
        marker=pl.Series([cell.marker for cell in parsed], dtype=pl.String),
    )


def estimates(cells: pl.DataFrame) -> pl.DataFrame:
    """The first, second, and third estimates, with the 2003 placeholders set to missing."""
    revision_na = cells.filter(pl.col("column").is_in(list(REVISION_TERMS))).select(
        "reference_month",
        "seasonal_status",
        "column",
        na=(pl.col("marker") == "NA").fill_null(False),
    )
    parts = []
    for estimate, stage in STAGE_OF_ESTIMATE.items():
        placeholders = (
            revision_na.filter(pl.col("column").is_in(REVISIONS_OF_ESTIMATE[estimate]))
            .group_by("reference_month", "seasonal_status")
            .agg(placeholder=pl.col("na").all())
        )
        parts.append(
            cells.filter(pl.col("column") == estimate)
            .join(placeholders, on=["reference_month", "seasonal_status"], how="left")
            .with_columns(
                placeholder=pl.col("placeholder")
                & (pl.col("value") == 0).fill_null(False)
            )
            .select(
                "cell_id",
                "reference_month",
                "seasonal_status",
                release_stage=pl.lit(stage),
                value=pl.when(pl.col("placeholder"))
                .then(pl.lit(None, dtype=pl.Int64))
                .otherwise("value"),
                marker=pl.when(pl.col("placeholder"))
                .then(pl.lit("NA"))
                .otherwise("marker"),
            )
        )
    return pl.concat(parts).sort("reference_month", "seasonal_status", "release_stage")


def revision_checks(cells: pl.DataFrame) -> pl.DataFrame:
    """Each published revision beside the difference of the two estimates it compares."""
    values = cells.pivot(
        on="column", index=["reference_month", "seasonal_status"], values="value"
    )
    parts = [
        values.select(
            "reference_month",
            "seasonal_status",
            column=pl.lit(column),
            published=pl.col(column),
            difference=pl.col(later) - pl.col(earlier),
        )
        for column, (later, earlier) in REVISION_TERMS.items()
    ]
    return (
        pl.concat(parts)
        .drop_nulls(["published", "difference"])
        .with_columns(consistent=pl.col("published") == pl.col("difference"))
        .sort("reference_month", "seasonal_status", "column")
    )


def average(values: list[int], kind: str) -> Fraction:
    numbers = [abs(value) for value in values] if kind == "mean_absolute" else values
    return Fraction(sum(numbers), len(numbers))


def rounds_to(published: int, exact: Fraction) -> bool:
    """Whether `published` is `exact` rounded to an integer, taking either neighbor of a half.

    BLS rounds exact halves both ways: of the 61 yearly averages on the page that fall exactly
    halfway between two integers, 37 are rounded up and 24 down.
    """
    floor = exact.numerator // exact.denominator
    remainder = exact - floor
    if remainder == Fraction(1, 2):
        return published in (floor, floor + 1)
    return published == (floor + 1 if remainder > Fraction(1, 2) else floor)


def average_checks(raw: pl.DataFrame) -> pl.DataFrame:
    """Each published yearly or summary average beside the exact average of its revisions."""
    revisions = monthly_cells(raw).filter(
        pl.col("column").is_in(list(REVISION_TERMS)) & pl.col("value").is_not_null()
    )
    published = raw.filter(
        (pl.col("source") == "cesnaicsrev") & pl.col("row_key").str.contains(":mean")
    )
    rows = []
    for row_key, column_key, text in published.select(
        "row_key", "column_key", "text"
    ).iter_rows():
        cell = parse_cell(text)
        if cell.value is None:
            continue
        status, column = column_key.split("_", 1)
        if row_key.startswith("summary:"):
            _, kind, period = row_key.split(":", 2)
            first, last = SUMMARY_PERIODS[period][column]
        else:
            year, kind = row_key.split(":")
            first, last = date(int(year), 1, 1), date(int(year), 12, 1)
        window = revisions.filter(
            (pl.col("seasonal_status") == status.upper())
            & (pl.col("column") == column)
            & (pl.col("reference_month") >= first)
            & ((pl.col("reference_month") <= last) if last else pl.lit(True))
        )
        exact = average(window["value"].to_list(), kind)
        rows.append(
            {
                "row_key": row_key,
                "seasonal_status": status.upper(),
                "column": column,
                "kind": kind,
                "published": cell.value,
                "months": window.height,
                "exact": str(exact),
                "reproduced": rounds_to(cell.value, exact),
            }
        )
    return pl.DataFrame(rows)
