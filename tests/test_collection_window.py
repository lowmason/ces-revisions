"""Revision size against the collection window: which revisions enter, and in which period."""

from datetime import date
from functools import cache

import collection_window as report_script
import polars as pl
import pytest
import vintage_data
from archive_inventory import generated_block

from ces_revisions.collection_window import (
    DROPPED_MONTHS,
    absolute_revisions,
    days_regression,
    revisions_by_days,
)
from ces_revisions.vintages.collection_days import LAPSE_DELAYED_MONTHS
from ces_revisions.vintages.months import month_range


@cache
def revisions() -> pl.DataFrame:
    return absolute_revisions(
        vintage_data.table_changes(), vintage_data.collection_days()
    )


def test_missing_estimates_drop_rather_than_count_as_zero():
    frame = revisions()
    keys = set(frame.select("reference_month", "transition").iter_rows())
    assert (date(2003, 4, 1), "F_S") not in keys
    assert (date(2003, 3, 1), "F_T") not in keys
    assert (date(2003, 4, 1), "F_T") not in keys
    assert frame.filter(pl.col("reference_month").is_in(DROPPED_MONTHS)).is_empty()
    latest = frame.group_by("transition").agg(pl.col("reference_month").max())
    assert dict(latest.iter_rows()) == {
        "F_S": date(2026, 7, 1),
        "F_T": date(2026, 6, 1),
    }
    for status in ("NSA", "SA"):
        counts = frame.filter(pl.col("seasonal_status") == status)["transition"]
        assert (counts == "F_S").sum() > (counts == "F_T").sum()


def test_pandemic_months_are_reported_apart():
    pandemic = revisions().filter(pl.col("period") == "pandemic")
    months = pandemic["reference_month"].unique().sort().to_list()
    assert months == month_range(date(2020, 3, 1), date(2021, 12, 1))


def test_lapse_flag_reaches_the_revisions():
    flagged = revisions().filter("lapse_delayed")["reference_month"]
    assert flagged.unique().sort().to_list() == list(LAPSE_DELAYED_MONTHS)


def test_revision_is_the_absolute_change_from_the_first_estimate():
    frame = revisions().filter(
        pl.col("reference_month").is_in([date(1979, 1, 1), date(2025, 8, 1)])
    )
    values = {
        (row["reference_month"], row["seasonal_status"], row["transition"]): row[
            "absolute_revision"
        ]
        for row in frame.iter_rows(named=True)
    }
    assert values[(date(1979, 1, 1), "NSA", "F_S")] == 137
    assert values[(date(1979, 1, 1), "NSA", "F_T")] == 78
    assert values[(date(2025, 8, 1), "SA", "F_S")] == 26
    assert values[(date(2025, 8, 1), "SA", "F_T")] == 48


def test_only_the_revision_table_enters_from_the_long_panel():
    from_panel = absolute_revisions(
        vintage_data.long_panel(), vintage_data.collection_days()
    )
    assert from_panel.equals(revisions())


def synthetic_revisions() -> pl.DataFrame:
    """Three main-period months and one pandemic month for each status and transition."""
    base = {
        ("NSA", "F_S"): 10,
        ("NSA", "F_T"): 20,
        ("SA", "F_S"): 30,
        ("SA", "F_T"): 40,
    }
    months = [
        (date(2001, 1, 1), 13, False, 0, "main"),
        (date(2001, 2, 1), 13, True, 10, "main"),
        (date(2001, 3, 1), 14, False, 50, "main"),
        (date(2020, 4, 1), 15, False, 999, "pandemic"),
    ]
    rows = [
        {
            "reference_month": month,
            "seasonal_status": status,
            "transition": transition,
            "absolute_revision": start + step,
            "collection_days": days,
            "lapse_delayed": lapse,
            "period": period,
        }
        for (status, transition), start in base.items()
        for month, days, lapse, step, period in months
    ]
    return pl.DataFrame(rows, schema_overrides={"collection_days": pl.Int32})


def test_revisions_by_days_counts_lapses_and_averages():
    table = revisions_by_days(synthetic_revisions()).filter(
        pl.col("seasonal_status") == "NSA", pl.col("transition") == "F_S"
    )
    assert table.select(
        "period", "collection_days", "n", "lapse_n", "mean_absolute_revision"
    ).rows() == [
        ("main", 13, 2, 1, 15.0),
        ("main", 14, 1, 0, 60.0),
        ("pandemic", 15, 1, 0, 1009.0),
    ]


def test_report_lays_out_each_day_count_and_an_all_row():
    lines = report_script.render_by_days(synthetic_revisions(), "main").splitlines()
    assert (
        lines[0]
        == "| `D_t` | n F→S | n F→T | Lapse-delayed | NSA F→S | NSA F→T | SA F→S | SA F→T |"
    )
    assert lines[2:] == [
        "| 13 | 2 | 2 | 1 | 15.0 | 25.0 | 35.0 | 45.0 |",
        "| 14 | 1 | 1 | 0 | 60.0 | 70.0 | 80.0 | 90.0 |",
        "| All | 3 | 3 | 1 | 30.0 | 40.0 | 50.0 | 60.0 |",
    ]


def test_regression_matches_an_independent_statsmodels_fit():
    """Reference values from statsmodels OLS with cluster="year" errors (plan 12, Task 3 Step 7)."""
    expected = {
        ("NSA", "F_S"): (546, -1.039163, 1.120130, -29.802999, 9.247932),
        ("NSA", "F_T"): (544, -0.269608, 2.230070, -23.107881, 18.618386),
        ("SA", "F_S"): (546, -0.446174, 1.054940, -24.841061, 8.279678),
        ("SA", "F_T"): (544, 1.096002, 1.231889, -40.430822, 10.739625),
    }
    fits = days_regression(revisions())
    assert fits["clusters"].to_list() == [47, 47, 47, 47]
    for row in fits.iter_rows(named=True):
        n, *values = expected[(row["seasonal_status"], row["transition"])]
        assert row["n"] == n
        got = [
            row[name]
            for name in ("days_coefficient", "days_se", "lapse_coefficient", "lapse_se")
        ]
        assert got == pytest.approx(values, abs=1e-6)


def test_report_refuses_without_a_built_panel(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(report_script, "PANEL_DIR", tmp_path)
    assert report_script.main(["report"]) == 1
    assert "scripts/vintage_sources.py build" in capsys.readouterr().err


def test_a_month_without_a_day_count_is_refused():
    days = vintage_data.collection_days().filter(
        pl.col("reference_month") != date(2026, 6, 1)
    )
    with pytest.raises(ValueError, match="no release in the day counts"):
        absolute_revisions(vintage_data.table_changes(), days)


def test_report_lays_out_the_regression():
    lines = report_script.render_regression(revisions()).splitlines()
    assert (
        lines[0]
        == "| Status | Transition | n | Years | `D_t` coefficient | SE | Lapse coefficient | SE |"
    )
    assert lines[2] == "| NSA | F→S | 546 | 47 | -1.04 | 1.12 | -29.80 | 9.25 |"
    assert lines[5] == "| SA | F→T | 544 | 47 | 1.10 | 1.23 | -40.43 | 10.74 |"


def test_report_fills_the_three_generated_blocks(tmp_path, monkeypatch, capsys):
    vintage_data.table_changes().write_parquet(tmp_path / "panel.parquet")
    vintage_data.collection_days().write_parquet(tmp_path / "collection_days.parquet")
    findings = tmp_path / "findings.md"
    names = ("stage-1-main", "stage-1-pandemic", "stage-1-regression")
    findings.write_text(
        "".join(
            f"<!-- BEGIN GENERATED {name} -->\n<!-- END GENERATED {name} -->\n"
            for name in names
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(report_script, "PANEL_DIR", tmp_path)
    monkeypatch.setattr(report_script, "FINDINGS", findings)
    assert report_script.main(["report"]) == 0
    document = findings.read_text(encoding="utf-8")
    main = generated_block(document, "stage-1-main")
    assert "| All | 546 | 544 | 5 | 44.5 | 66.4 | 38.5 | 52.1 |" in main
    pandemic = generated_block(document, "stage-1-pandemic")
    assert "| All | 22 | 22 | 0 |" in pandemic
    regression = generated_block(document, "stage-1-regression")
    assert "| SA | F→T | 544 | 47 | 1.10 | 1.23 | -40.43 | 10.74 |" in regression
    assert "report: 2268 revisions" in capsys.readouterr().out
