"""Annual CES Table 1 coverage proxy and published RSE regimes."""

from datetime import date

import polars as pl
import pytest

from ces_revisions.annual import publications, sample
from ces_revisions.vintages.release_index import build_release_index


def built() -> pl.DataFrame:
    calendar = publications.build_publication_calendar(build_release_index())
    return sample.build_sample_panel(calendar)


def test_required_table_1_counts_cannot_be_silently_missing():
    assert sample._required_integer("115,195", "active UI accounts") == 115195
    with pytest.raises(ValueError, match="active UI accounts"):
        sample._required_integer("", "active UI accounts")


def test_panel_has_one_annual_row_per_supersector_and_year():
    frame = built()
    assert frame.height == 24 * len(sample.SECTORS)
    assert frame["benchmark_year"].unique().sort().to_list() == list(range(2002, 2026))
    assert frame.group_by("benchmark_year").len()["len"].unique().to_list() == [
        len(sample.SECTORS)
    ]
    assert frame["frequency"].unique().to_list() == ["annual"]
    assert (
        frame["reference_month"]
        == pl.Series([date(year, 3, 1) for year in frame["benchmark_year"]])
    ).all()


def test_coverage_is_explicitly_the_active_report_proxy():
    frame = built()
    assert frame["coverage_measure"].unique().to_list() == [
        "table_1_active_report_employee_share_proxy"
    ]
    published = frame.filter(pl.col("coverage_status") == "published")
    calculated = (
        published["sample_employees_thousands"]
        / published["benchmark_employment_thousands"]
        * 100
    ).round(0)
    assert (calculated == published["coverage_percent"]).all()
    assert published["coverage_note"].str.contains("not the matched sample").all()


def test_2009_employee_counts_are_normalized_to_the_tables_stated_unit():
    rows = built().filter(pl.col("benchmark_year") == 2009)
    total = rows.filter(pl.col("sector") == "00").row(0, named=True)
    assert total["sample_employees_thousands"] == 38476.076
    assert total["coverage_percent"] == 29.0
    assert rows["transformation"].unique().to_list() == ["sample_jobs_to_thousands"]


def test_march_2012_is_an_explicit_archive_gap_not_an_interpolation():
    rows = built().filter(pl.col("benchmark_year") == 2012)
    assert rows["coverage_status"].unique().to_list() == ["archive_gap"]
    assert (
        rows.select(
            "benchmark_employment_thousands",
            "active_ui_accounts",
            "active_establishments",
            "sample_employees_thousands",
            "coverage_percent",
        )
        .null_count()
        .row(0)
        == (len(rows),) * 5
    )
    assert rows["missing_reason"].str.contains("March 2012 Table 1").all()
    assert rows["observable_at"].is_not_null().all()


def test_rse_regimes_and_definition_break_are_preserved():
    frame = built()
    level = frame.filter(pl.col("rse_measure") == "level")
    change = frame.filter(pl.col("rse_measure") == "one_month_change")
    gap = frame.filter(pl.col("benchmark_year").is_between(2010, 2015))
    assert level["benchmark_year"].unique().sort().to_list() == list(range(2002, 2010))
    assert level["rse_closing"].unique().to_list() == ["first_closing"]
    assert change["benchmark_year"].unique().sort().to_list() == list(range(2016, 2026))
    assert change["rse_closing"].unique().to_list() == ["not_stated"]
    assert gap["rse_value"].is_null().all()
    assert frame.filter(pl.col("rse_definition_break"))[
        "benchmark_year"
    ].unique().to_list() == [2016]


def test_historical_government_level_rse_is_published_as_unavailable():
    government = built().filter(
        (pl.col("rse_measure") == "level") & (pl.col("sector") == "90")
    )
    assert government.height == 8
    assert government["rse_value"].is_null().all()


def test_early_change_rse_preserves_source_level_sector_gaps():
    frame = built().filter(pl.col("benchmark_year").is_between(2016, 2017))
    assert frame.filter(pl.col("sector") == "00")["rse_value"].is_not_null().all()
    assert frame.filter(pl.col("sector") == "10")["rse_value"].is_null().all()
    assert frame.filter(pl.col("sector") == "90")["rse_value"].is_not_null().all()


def test_aggregate_government_rse_is_not_confused_with_detail_rows():
    government = built().filter(
        (pl.col("benchmark_year") >= 2018) & (pl.col("sector") == "90")
    )
    assert government.height == 8
    assert government["rse_value"].is_not_null().all()


def test_detailed_trade_rows_are_not_double_counted():
    assert set(built()["sector"]) == set(sample.SECTORS)
    assert not {"41", "42", "43", "44"} & set(built()["sector"])


def test_published_series_change_is_metadata_not_a_covariate():
    frame = built()
    assert "program_cut_covariate" not in frame.columns
    note = frame.filter(pl.col("benchmark_year") == 2025)[
        "published_series_note"
    ].unique()
    assert len(note) == 1
    assert "not definition-comparable" in note.item()
    assert "no series-structure change" in note.item()


def test_every_sample_row_has_an_as_of_timestamp():
    assert built()["observable_at"].is_not_null().all()
