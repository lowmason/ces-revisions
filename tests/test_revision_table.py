"""BLS's revision table as data: cells, markers, estimates, and published averages."""

from datetime import date
from fractions import Fraction
from pathlib import Path

import polars as pl
import pytest
import vintage_data

from ces_revisions.vintages import raw, revision_table
from ces_revisions.vintages.revision_table import Cell, parse_cell, rounds_to

FIXTURES = Path(__file__).parent / "fixtures" / "vintages"


def excerpt_estimates() -> pl.DataFrame:
    page = (FIXTURES / "cesnaicsrev-excerpt.htm").read_text(encoding="utf-8")
    cells = raw.revision_table_cells(page).with_row_index("cell_id")
    return revision_table.estimates(revision_table.monthly_cells(cells))


def estimate(frame: pl.DataFrame, month: date, status: str, stage: str) -> tuple:
    row = frame.filter(
        (pl.col("reference_month") == month)
        & (pl.col("seasonal_status") == status)
        & (pl.col("release_stage") == stage)
    )
    return row["value"].item(), row["marker"].item()


@pytest.mark.parametrize(
    ("text", "cell"),
    [
        ("143", Cell(143, None)),
        ("-2722", Cell(-2722, None)),
        ("-105 (B)", Cell(-105, "B")),
        ("31(c)", Cell(31, "c")),
        ("-(A)", Cell(None, "A")),
        ("NA***", Cell(None, "NA")),
        ("NA", Cell(None, "NA")),
        ("", Cell(None, "blank")),
    ],
)
def test_parse_cell_reads_values_and_footnote_markers(text, cell):
    assert parse_cell(text) == cell


def test_an_unrecognized_cell_is_an_error():
    with pytest.raises(ValueError, match="unrecognized revision table cell"):
        parse_cell("12.5")


def test_the_2003_zeros_are_missing_estimates():
    frame = excerpt_estimates()
    assert estimate(frame, date(2003, 4, 1), "SA", "F") == (-48, None)
    assert estimate(frame, date(2003, 3, 1), "SA", "S") == (-124, None)
    assert estimate(frame, date(2003, 3, 1), "SA", "T") == (None, "NA")
    assert estimate(frame, date(2003, 4, 1), "NSA", "S") == (None, "NA")
    assert estimate(frame, date(2003, 4, 1), "NSA", "T") == (None, "NA")
    assert estimate(frame, date(2003, 5, 1), "NSA", "T") == (680, None)


def test_october_2025_has_no_first_estimate_and_a_marked_second():
    frame = excerpt_estimates()
    assert estimate(frame, date(2025, 10, 1), "SA", "F") == (None, "A")
    assert estimate(frame, date(2025, 10, 1), "SA", "S") == (-105, "B")
    assert estimate(frame, date(2025, 9, 1), "NSA", "S") == (None, "A")
    assert estimate(frame, date(2025, 9, 1), "NSA", "T") == (328, None)


@pytest.mark.parametrize(
    ("published", "exact", "expected"),
    [
        (78, Fraction(157, 2), True),
        (79, Fraction(157, 2), True),
        (77, Fraction(157, 2), False),
        (-4, Fraction(-9, 2), True),
        (-5, Fraction(-9, 2), True),
        (48, Fraction(12019, 250), True),
        (49, Fraction(12019, 250), False),
    ],
)
def test_rounds_to_takes_either_neighbor_of_an_exact_half(published, exact, expected):
    assert rounds_to(published, exact) is expected


# --- The committed page ---------------------------------------------------------------------


def test_every_published_average_reproduces_from_its_monthly_revisions():
    checks = revision_table.average_checks(vintage_data.raw_values())
    assert checks.filter(~pl.col("reproduced")).height == 0
    assert checks.filter(pl.col("row_key").str.starts_with("summary:")).height == 36


def test_bls_rounds_exact_half_yearly_averages_both_ways():
    checks = revision_table.average_checks(vintage_data.raw_values()).filter(
        pl.col("row_key").str.contains(r"^(?:19\d\d|20[01]\d|202[0-5]):")
    )
    ties = [
        (published, Fraction(exact))
        for published, exact in checks.select("published", "exact").iter_rows()
        if Fraction(exact).denominator == 2
    ]
    rounded_up = sum(published == exact + Fraction(1, 2) for published, exact in ties)
    assert (len(ties), rounded_up) == (61, 37)


def test_published_revisions_equal_their_estimates_difference_but_once():
    checks = revision_table.revision_checks(vintage_data.table_cells())
    inconsistent = checks.filter(~pl.col("consistent")).select(
        "reference_month", "seasonal_status", "column", "published", "difference"
    )
    assert inconsistent.rows() == [(date(2023, 9, 1), "SA", "3rd_minus_2nd", -34, -35)]


def test_markers_sit_only_where_the_page_footnotes_put_them():
    cells = vintage_data.table_cells()

    def months(marker: str) -> list[date]:
        rows = cells.filter(pl.col("marker") == marker)
        return sorted(set(rows["reference_month"]))

    assert months("NA") == [date(2003, 3, 1), date(2003, 4, 1)]
    assert months("A") == [date(2025, 9, 1), date(2025, 10, 1)]
    marked_b = cells.filter(pl.col("marker") == "B")
    assert sorted(marked_b.select("seasonal_status", "column").iter_rows()) == [
        ("NSA", "2nd"),
        ("SA", "2nd"),
    ]
