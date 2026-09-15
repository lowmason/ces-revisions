"""Roadmap Stage 4's executable exit criteria across all annual artifacts."""

from dataclasses import replace

import annual_data
import polars as pl
import pytest

from ces_revisions.annual import contract
from ces_revisions.vintages.release_index import build_release_index


def test_complete_build_passes_the_stage_4_contract():
    contract.validate(annual_data.result(), build_release_index())


def test_contract_rejects_an_undated_value():
    result = annual_data.result()
    broken = (
        result.benchmarks.with_row_index("row")
        .with_columns(
            pl.when(pl.col("row") == 0)
            .then(None)
            .otherwise(pl.col("observable_at"))
            .alias("observable_at")
        )
        .drop("row")
    )
    with pytest.raises(ValueError, match="benchmarks has null observable_at"):
        contract.validate(replace(result, benchmarks=broken), build_release_index())


def test_contract_rejects_an_unknown_source_key():
    result = annual_data.result()
    broken = (
        result.qcew_precision.with_row_index("row")
        .with_columns(
            pl.when(pl.col("row") == 0)
            .then(pl.lit(["unknown::cell"], dtype=pl.List(pl.String)))
            .otherwise(pl.col("source_keys"))
            .alias("source_keys")
        )
        .drop("row")
    )
    with pytest.raises(ValueError, match="unknown source keys"):
        contract.validate(replace(result, qcew_precision=broken), build_release_index())


def test_contract_requires_source_keys_on_every_derived_artifact():
    result = annual_data.result()
    broken = result.benchmarks.drop("source_keys")
    with pytest.raises(ValueError, match="benchmarks is missing required columns"):
        contract.validate(replace(result, benchmarks=broken), build_release_index())


def test_contract_rejects_a_null_element_in_source_keys():
    result = annual_data.result()
    broken = (
        result.benchmarks.with_row_index("row")
        .with_columns(
            pl.when(pl.col("row") == 0)
            .then(pl.lit([None], dtype=pl.List(pl.String)))
            .otherwise(pl.col("source_keys"))
            .alias("source_keys")
        )
        .drop("row")
    )
    with pytest.raises(ValueError, match="benchmarks has a null source key"):
        contract.validate(replace(result, benchmarks=broken), build_release_index())


def test_contract_rejects_a_null_transformation():
    result = annual_data.result()
    broken = (
        result.qcew_precision.with_row_index("row")
        .with_columns(
            pl.when(pl.col("row") == 0)
            .then(None)
            .otherwise(pl.col("transformation"))
            .alias("transformation")
        )
        .drop("row")
    )
    with pytest.raises(ValueError, match="qcew_precision has a blank transformation"):
        contract.validate(replace(result, qcew_precision=broken), build_release_index())


def test_contract_requires_every_final_benchmark_anchor_year():
    result = annual_data.result()
    broken = result.benchmarks.filter(
        ~((pl.col("benchmark_year") == 1979) & (pl.col("benchmark_status") == "final"))
    )
    with pytest.raises(ValueError, match="final benchmark anchor years"):
        contract.validate(replace(result, benchmarks=broken), build_release_index())


def test_contract_requires_both_2025_reconstruction_events():
    result = annual_data.result()
    broken = result.reconstruction_events.filter(pl.col("benchmark_year") != 2025)
    with pytest.raises(ValueError, match="2025 reconstruction events"):
        contract.validate(
            replace(result, reconstruction_events=broken), build_release_index()
        )


def test_contract_preserves_the_2025_discrepancy_ruling():
    result = annual_data.result()
    broken = result.benchmarks.with_columns(
        pl.when(
            (pl.col("benchmark_year") == 2025)
            & (pl.col("benchmark_status") == "final")
            & (pl.col("row_kind") == "anchor")
            & (pl.col("sector") == "00")
        )
        .then(pl.lit("reconciled"))
        .otherwise(pl.col("discrepancy_status"))
        .alias("discrepancy_status")
    )
    with pytest.raises(ValueError, match="2025 benchmark discrepancy ruling"):
        contract.validate(replace(result, benchmarks=broken), build_release_index())


def test_contract_rejects_a_null_government_structural_zero():
    result = annual_data.result()
    broken = (
        result.birth_death.with_row_index("row")
        .with_columns(
            pl.when(pl.col("sector") == "90")
            .then(None)
            .otherwise(pl.col("value_thousands"))
            .alias("value_thousands")
        )
        .drop("row")
    )
    with pytest.raises(ValueError, match="government birth-death"):
        contract.validate(replace(result, birth_death=broken), build_release_index())


def test_contract_requires_the_rse_definition_break():
    result = annual_data.result()
    broken = result.sample_panel.with_columns(
        pl.lit(False).alias("rse_definition_break")
    )
    with pytest.raises(ValueError, match="RSE definition break"):
        contract.validate(replace(result, sample_panel=broken), build_release_index())


def test_contract_requires_the_complete_sample_year_sector_grid():
    result = annual_data.result()
    broken = result.sample_panel.filter(
        ~((pl.col("benchmark_year") == 2002) & (pl.col("sector") == "90"))
    )
    with pytest.raises(ValueError, match="sample year/sector coverage"):
        contract.validate(replace(result, sample_panel=broken), build_release_index())


def test_final_and_reconstruction_dates_equal_the_stage_3_carriers():
    result = annual_data.result()
    finals = result.publication_calendar.filter(
        pl.col("publication_kind") == "benchmark_final"
    )
    joined = result.reconstruction_events.join(
        finals.select(
            "benchmark_year",
            final_date=pl.col("publication_date"),
            final_observable=pl.col("observable_at"),
        ),
        on="benchmark_year",
    )
    assert (joined["publication_date"] == joined["final_date"]).all()
    assert (joined["observable_at"] == joined["final_observable"]).all()


def test_preliminary_benchmarks_use_only_stage_4_publication_evidence():
    preliminary = annual_data.artifact("publication_calendar").filter(
        pl.col("publication_kind") == "benchmark_preliminary"
    )
    assert preliminary["source_keys"].list.len().gt(0).all()
    assert (
        preliminary["source_keys"]
        .explode(empty_as_null=True)
        .str.starts_with("stage3::")
        .not_()
        .all()
    )


def test_no_annual_sample_measure_was_repeated_across_months():
    sample = annual_data.artifact("sample_panel")
    assert sample.select("benchmark_year", "sector").n_unique() == sample.height
    assert sample["frequency"].unique().to_list() == ["annual"]


def test_qcew_long_values_are_jobs_and_precision_is_thousands():
    assert annual_data.artifact("qcew_revisions")["unit"].unique().to_list() == ["jobs"]
    assert annual_data.artifact("qcew_precision")["unit"].unique().to_list() == [
        "thousands"
    ]
