"""Req 1's same-release differencing and its reconciliation with BLS's revision table."""

from datetime import date

import polars as pl
import vintage_data

from ces_revisions.vintages import differencing


def level(vintage: str, month: date, value: int | None) -> dict:
    return {
        "source": "cesvinall",
        "sector": "00",
        "seasonal_status": "NSA",
        "vintage_id": f"cesvinall:{vintage}",
        "release_month": date.fromisoformat(f"{vintage}-01"),
        "reference_month": month,
        "value_thousands": value,
    }


def test_changes_difference_levels_within_one_vintage_only():
    levels = pl.DataFrame(
        [
            level("2003-06", date(2003, 4, 1), 100),
            level("2003-06", date(2003, 5, 1), 130),
            level("2003-06", date(2003, 6, 1), None),
            level("2003-07", date(2003, 5, 1), 131),
            level("2003-07", date(2003, 6, 1), 140),
        ]
    )
    changes = differencing.same_release_changes(levels).sort(
        "vintage_id", "reference_month"
    )
    assert changes.select(
        "vintage_id", "reference_month", "change_thousands"
    ).rows() == [
        ("cesvinall:2003-06", date(2003, 5, 1), 30),
        ("cesvinall:2003-06", date(2003, 6, 1), None),
        ("cesvinall:2003-07", date(2003, 6, 1), 9),
    ]


# --- The committed sources ------------------------------------------------------------------


def test_same_release_changes_reproduce_every_table_estimate_within_the_frontier():
    frame = differencing.reconcile(
        vintage_data.stage_changes(), vintage_data.table_estimates()
    )
    differs = frame.filter(pl.col("outcome") == "differs").select(
        "reference_month",
        "seasonal_status",
        "release_stage",
        "panel_thousands",
        "table_thousands",
    )
    assert differs.rows() == [(date(2003, 11, 1), "NSA", "T", 147, 46)]
    both_missing = frame.filter(pl.col("outcome") == "missing_in_both")
    assert set(both_missing.select("reference_month", "release_stage").iter_rows()) == {
        (date(2025, 9, 1), "S"),
        (date(2025, 10, 1), "F"),
    }
    outcomes = dict(frame.group_by("outcome").len().iter_rows())
    assert outcomes["reproduced"] == 1627
    assert set(outcomes) <= {
        "reproduced",
        "differs",
        "missing_in_both",
        "beyond_frontier",
        "right_censored",
    }


def test_the_table_takes_november_2003_third_estimate_across_two_releases():
    """November 2003's third estimate arrived in the February 2004 benchmark release. BLS's
    table differences it against October 2003 in the January 2004 release; Req 1's
    same-release change uses October in the February release."""
    levels = {
        (row["vintage_id"], row["reference_month"]): row["value_thousands"]
        for row in vintage_data.levels()
        .filter(
            (pl.col("sector") == "00")
            & (pl.col("seasonal_status") == "NSA")
            & pl.col("reference_month").is_between(date(2003, 10, 1), date(2003, 11, 1))
        )
        .iter_rows(named=True)
    }
    november = levels[("cesvinall:2004-01", date(2003, 11, 1))]
    assert november - levels[("cesvinall:2004-01", date(2003, 10, 1))] == 147
    assert november - levels[("cesvinall:2003-12", date(2003, 10, 1))] == 46


def test_panel_revisions_reproduce_the_table_revisions_but_three_cells():
    frame = differencing.revision_reconciliation(
        vintage_data.stage_changes(), vintage_data.table_cells()
    )
    differs = frame.filter(pl.col("outcome") == "differs")
    assert sorted(
        differs.select("reference_month", "seasonal_status", "column").iter_rows()
    ) == [
        (date(2003, 11, 1), "NSA", "3rd_minus_1st"),
        (date(2003, 11, 1), "NSA", "3rd_minus_2nd"),
        (date(2023, 9, 1), "SA", "3rd_minus_2nd"),
    ]
    assert frame.filter(pl.col("outcome") == "reproduced").height == 1617
