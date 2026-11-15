"""Revision size against the collection window: which revisions enter, and in which period."""

from datetime import date
from functools import cache

import polars as pl
import vintage_data

from ces_revisions.collection_window import DROPPED_MONTHS, absolute_revisions
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
