"""The public national QCEW revision sequence and its March precision proxy."""

from math import sqrt

import polars as pl

from ces_revisions.annual import publications, qcew
from ces_revisions.vintages.release_index import build_release_index


def revisions() -> pl.DataFrame:
    calendar = publications.build_publication_calendar(build_release_index())
    return qcew.build_qcew_revisions(calendar)


def test_only_2017_plus_national_monthly_employment_is_ingested():
    frame = revisions()
    assert frame["area"].unique().to_list() == ["United States"]
    assert frame["reference_month"].min().year == 2017
    assert frame["field"].str.ends_with(" Employment").all()
    assert frame["reference_month"].n_unique() < 12 * 20


def test_completed_quarters_have_the_documented_release_count():
    frame = revisions().filter(pl.col("sequence_status") == "complete")
    counts = frame.group_by("qcew_year", "qcew_quarter", "field").agg(
        releases=pl.col("release_order").n_unique(),
        last_order=pl.col("release_order").max(),
    )
    expected = pl.DataFrame(
        {"qcew_quarter": [1, 2, 3, 4], "expected": [5, 4, 3, 2]},
        schema={"qcew_quarter": pl.Int8, "expected": pl.UInt32},
    )
    checked = counts.join(expected, on="qcew_quarter")
    assert (checked["releases"] == checked["expected"]).all()
    assert (checked["last_order"] == checked["expected"] - 1).all()


def test_final_value_is_an_alias_not_a_sixth_release():
    source = qcew.read_source()
    q1_complete = source.filter(
        (pl.col("Quarter") == 1)
        & pl.col("Final Value").cast(pl.String).str.contains(r"^-?\d")
    )
    assert (
        q1_complete["Final Value"].cast(pl.String)
        == q1_complete["Fourth Revised Value"].cast(pl.String)
    ).all()
    assert revisions().filter(pl.col("qcew_quarter") == 1)["release_order"].max() == 4


def test_values_and_successive_revisions_remain_in_jobs():
    frame = revisions()
    assert frame["unit"].unique().to_list() == ["jobs"]
    assert (frame["employment_jobs"].drop_nulls() % 1 == 0).all()
    group = frame.filter(
        (pl.col("qcew_year") == 2017)
        & (pl.col("qcew_quarter") == 1)
        & (pl.col("field") == "March Employment")
    ).sort("release_order")
    expected = group["employment_jobs"].diff()
    assert group["revision_jobs"].to_list() == expected.to_list()


def test_each_revision_uses_the_qcew_publication_clock():
    calendar = publications.build_publication_calendar(build_release_index())
    frame = revisions()
    row = frame.filter(
        (pl.col("qcew_year") == 2017)
        & (pl.col("qcew_quarter") == 1)
        & (pl.col("release_order") == 4)
    ).row(0, named=True)
    expected = calendar.filter(
        (pl.col("publication_kind") == "qcew_revision")
        & (pl.col("qcew_year") == 2017)
        & (pl.col("qcew_quarter") == 1)
        & (pl.col("release_order") == 4)
    ).row(0, named=True)
    assert row["publication_date"] == expected["publication_date"]
    assert row["observable_at"] == expected["observable_at"]


def test_qcew_calendar_has_exactly_the_releases_present_in_the_source():
    calendar = publications.build_publication_calendar(build_release_index())
    keys = ["qcew_year", "qcew_quarter", "release_order"]
    expected = revisions().select(keys).unique().sort(keys)
    actual = (
        calendar.filter(pl.col("publication_kind") == "qcew_revision")
        .select(keys)
        .unique()
        .sort(keys)
    )
    assert actual.equals(expected)
    qcew_dates = calendar.filter(pl.col("publication_kind") == "qcew_revision")
    for group in qcew_dates.partition_by("qcew_year", "qcew_quarter"):
        assert group.sort("release_order")["publication_date"].is_sorted()


def test_march_precision_is_rms_of_four_revision_increments_in_thousands():
    frame = revisions()
    precision = qcew.build_qcew_precision(frame)
    march = frame.filter(
        (pl.col("qcew_year") == 2017)
        & (pl.col("field") == "March Employment")
        & (pl.col("release_order") > 0)
    )
    expected = sqrt(sum((value / 1_000) ** 2 for value in march["revision_jobs"]) / 4)
    row = precision.filter(pl.col("benchmark_year") == 2017).row(0, named=True)
    assert row["sequence_status"] == "complete"
    assert row["release_count"] == 5
    assert row["revision_precision_thousands"] == expected


def test_latest_incomplete_march_sequence_is_right_censored():
    precision = qcew.build_qcew_precision(revisions())
    incomplete = precision.filter(pl.col("release_count") < 5)
    if incomplete.height:
        assert incomplete["sequence_status"].unique().to_list() == ["right_censored"]
        assert incomplete["revision_precision_thousands"].is_null().all()
    assert precision["observable_at"].is_not_null().all()
