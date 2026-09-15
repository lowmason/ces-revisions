"""Monthly and forecast-versus-realized CES net birth-death values."""

from datetime import date

import polars as pl

from ces_revisions.annual import birth_death, publications
from ces_revisions.vintages.release_index import build_release_index


def built() -> pl.DataFrame:
    index = build_release_index()
    calendar = publications.build_publication_calendar(index)
    return birth_death.build_birth_death(calendar, index)


def test_modern_schedule_vintages_are_complete():
    frame = built().filter(pl.col("series_kind") == "published_schedule")
    initial = frame.filter(
        (pl.col("schedule_kind") == "preliminary") & (pl.col("sector") == "00")
    )
    revised = frame.filter(
        (pl.col("schedule_kind") == "post_benchmark") & (pl.col("sector") == "00")
    )
    assert initial["reference_month"].dt.year().unique().sort().to_list() == list(
        range(2004, 2027)
    )
    assert revised["reference_month"].dt.year().unique().sort().to_list() == list(
        range(2003, 2026)
    )
    assert revised.group_by(pl.col("reference_month").dt.year()).len()[
        "len"
    ].unique().to_list() == [9]


def test_monthly_total_is_the_sum_of_private_supersectors():
    frame = built().filter(
        (pl.col("series_kind") == "published_schedule")
        & pl.col("sector").is_in(
            ["00", "10", "20", "30", "40", "50", "55", "60", "65", "70", "80"]
        )
    )
    totals = frame.filter(pl.col("sector") == "00").select(
        "schedule_kind", "reference_month", published=pl.col("value_thousands")
    )
    summed = (
        frame.filter(pl.col("sector") != "00")
        .group_by("schedule_kind", "reference_month")
        .agg(calculated=pl.col("value_thousands").sum())
    )
    checked = totals.join(summed, on=["schedule_kind", "reference_month"])
    assert (checked["published"] == checked["calculated"]).all()


def test_preliminary_months_use_their_first_employment_situation_release():
    index = build_release_index()
    frame = built()
    row = frame.filter(
        (pl.col("schedule_kind") == "preliminary")
        & (pl.col("reference_month") == date(2025, 7, 1))
        & (pl.col("sector") == "00")
    ).row(0, named=True)
    release = index.filter(pl.col("reference_month") == date(2025, 7, 1)).row(
        0, named=True
    )
    assert row["publication_date"] == release["published_date"]
    assert row["observable_at"] == release["observable_at"]


def test_post_benchmark_months_share_the_final_benchmark_publication():
    frame = built().filter(
        (pl.col("schedule_kind") == "post_benchmark")
        & (pl.col("benchmark_year") == 2025)
    )
    assert frame["publication_date"].unique().to_list() == [date(2026, 2, 11)]


def test_forecast_realized_rows_cover_2009_through_2025_and_balance():
    frame = built().filter(
        (pl.col("series_kind") == "forecast_vs_realized") & (pl.col("sector") == "00")
    )
    assert frame["benchmark_year"].unique().sort().to_list() == list(range(2009, 2026))
    wide = frame.pivot(
        on="value_kind",
        index=["benchmark_year", "reference_month", "frequency"],
        values="value_thousands",
    )
    assert (
        wide["realized_minus_forecast"] == wide["realized"] - wide["forecast"]
    ).all()


def test_printed_annual_totals_preserve_the_two_source_rounding_gaps():
    frame = built().filter(
        (pl.col("series_kind") == "forecast_vs_realized") & (pl.col("sector") == "00")
    )
    monthly = (
        frame.filter(pl.col("frequency") == "monthly")
        .group_by("benchmark_year", "value_kind")
        .agg(monthly_sum=pl.col("value_thousands").sum())
    )
    annual = frame.filter(pl.col("frequency") == "annual").select(
        "benchmark_year", "value_kind", annual_total=pl.col("value_thousands")
    )
    gaps = (
        monthly.join(annual, on=["benchmark_year", "value_kind"])
        .with_columns(gap=pl.col("monthly_sum") - pl.col("annual_total"))
        .filter(pl.col("gap") != 0)
        .sort("benchmark_year", "value_kind")
    )
    assert gaps.select("benchmark_year", "value_kind", "gap").rows() == [
        (2009, "forecast", 1.0),
        (2009, "realized", 1.0),
        (2010, "forecast", -1.0),
        (2010, "realized", -1.0),
    ]


def test_government_is_a_structural_zero_for_every_birth_death_key():
    frame = built()
    key = [
        "series_kind",
        "benchmark_year",
        "reference_month",
        "period_start",
        "period_end",
        "frequency",
        "schedule_kind",
        "value_kind",
    ]
    total_keys = frame.filter(pl.col("sector") == "00").select(key).unique()
    government = frame.filter(pl.col("sector") == "90")
    assert government["value_thousands"].eq(0.0).all()
    assert government["transformation"].unique().to_list() == [
        "government_structural_zero"
    ]
    assert government.select(key).unique().sort(key).equals(total_keys.sort(key))


def test_every_birth_death_row_is_nsa_dated_and_in_thousands():
    frame = built()
    assert frame["seasonal_status"].unique().to_list() == ["NSA"]
    assert frame["unit"].unique().to_list() == ["thousands"]
    assert frame["observable_at"].is_not_null().all()
