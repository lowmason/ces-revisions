"""Cross-artifact invariants that define roadmap Stage 4 completion."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import polars as pl

from ces_revisions.annual.publications import final_carrier_month

if TYPE_CHECKING:
    from ces_revisions.annual.build import AnnualBuild

DERIVED_ARTIFACTS = (
    "publication_calendar",
    "benchmarks",
    "reconstruction_events",
    "birth_death",
    "qcew_revisions",
    "qcew_precision",
    "sample_panel",
)
REQUIRED_DERIVED_SCHEMA = {
    "publication_date": pl.Date,
    "observable_at": pl.Datetime("us", "UTC"),
    "source_keys": pl.List(pl.String),
    "transformation": pl.String,
}
SAMPLE_SECTORS = (
    "00",
    "10",
    "20",
    "30",
    "40",
    "50",
    "55",
    "60",
    "65",
    "70",
    "80",
    "90",
)
BIRTH_DEATH_KEY = (
    "series_kind",
    "benchmark_year",
    "reference_month",
    "period_start",
    "period_end",
    "frequency",
    "schedule_kind",
    "value_kind",
)
EXPECTED_2025_EVENTS = {
    "2025-central-bank-commercial-bank",
    "2025-taxi-exclusion",
}


def _validate_derived_schema(result: AnnualBuild) -> None:
    for name in DERIVED_ARTIFACTS:
        frame = getattr(result, name)
        missing = set(REQUIRED_DERIVED_SCHEMA) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
        wrong = {
            column: (frame.schema[column], expected)
            for column, expected in REQUIRED_DERIVED_SCHEMA.items()
            if frame.schema[column] != expected
        }
        if wrong:
            raise ValueError(f"{name} has invalid required column types: {wrong}")
        transformations = frame["transformation"]
        if (
            transformations.is_null().any()
            or transformations.str.strip_chars().eq("").any()
        ):
            raise ValueError(f"{name} has a blank transformation")


def _all_source_keys(frame: pl.DataFrame, name: str) -> set[str]:
    source_keys = frame["source_keys"]
    if source_keys.is_null().any() or source_keys.list.len().eq(0).any():
        raise ValueError(f"{name} has a derived row with no source keys")
    exploded = source_keys.explode(empty_as_null=True)
    if exploded.is_null().any():
        raise ValueError(f"{name} has a null source key")
    return set(exploded)


def _validate_provenance(result: AnnualBuild, release_index: pl.DataFrame) -> None:
    annual_keys = set(result.raw_values["cell_key"])
    stage3_keys = {
        f"stage3::release_index::{month:%Y-%m}"
        for month in release_index["reference_month"]
    }
    allowed = annual_keys | stage3_keys
    used_transformations = set()
    for name in DERIVED_ARTIFACTS:
        frame = getattr(result, name)
        unknown = _all_source_keys(frame, name) - allowed
        if unknown:
            raise ValueError(f"{name} has unknown source keys: {sorted(unknown)[:5]}")
        used_transformations.update(frame["transformation"])
    registered = set(result.transformations["transformation"])
    if not used_transformations <= registered:
        raise ValueError(
            f"unregistered transformations: {sorted(used_transformations - registered)}"
        )


def _validate_dates(result: AnnualBuild, release_index: pl.DataFrame) -> None:
    for name in DERIVED_ARTIFACTS:
        frame = getattr(result, name)
        if frame["observable_at"].is_null().any():
            raise ValueError(f"{name} has null observable_at")
    by_month = {
        row["reference_month"]: row for row in release_index.iter_rows(named=True)
    }
    finals = result.publication_calendar.filter(
        pl.col("publication_kind") == "benchmark_final"
    )
    final_by_year = {}
    for row in finals.iter_rows(named=True):
        expected = by_month[final_carrier_month(row["benchmark_year"])]
        if (
            row["publication_date"] != expected["published_date"]
            or row["observable_at"] != expected["observable_at"]
        ):
            raise ValueError(
                f"benchmark year {row['benchmark_year']} does not match Stage 3"
            )
        final_by_year[row["benchmark_year"]] = row
    final_artifacts = {
        "benchmarks": result.benchmarks.filter(pl.col("benchmark_status") == "final"),
        "reconstruction_events": result.reconstruction_events,
    }
    for name, frame in final_artifacts.items():
        for row in frame.iter_rows(named=True):
            expected = final_by_year[row["benchmark_year"]]
            if (
                row["publication_date"] != expected["publication_date"]
                or row["observable_at"] != expected["observable_at"]
            ):
                raise ValueError(
                    f"{name} benchmark year {row['benchmark_year']} has the wrong carrier"
                )
    prelim = result.publication_calendar.filter(
        pl.col("publication_kind") == "benchmark_preliminary"
    )
    if (
        prelim["source_keys"]
        .explode(empty_as_null=True)
        .str.starts_with("stage3::")
        .any()
    ):
        raise ValueError("preliminary benchmark dates must use Stage 4 evidence")
    prelim_by_year = {
        row["benchmark_year"]: row for row in prelim.iter_rows(named=True)
    }
    for row in result.benchmarks.filter(
        pl.col("benchmark_status") == "preliminary"
    ).iter_rows(named=True):
        expected = prelim_by_year[row["benchmark_year"]]
        if (
            row["publication_date"] != expected["publication_date"]
            or row["observable_at"] != expected["observable_at"]
        ):
            raise ValueError(
                f"preliminary benchmark year {row['benchmark_year']} has the wrong date"
            )


def _require_anchor_years(result: AnnualBuild) -> None:
    anchors = result.benchmarks.filter(
        (pl.col("row_kind") == "anchor") & (pl.col("sector") == "00")
    )
    expected = {
        "final": list(range(1979, 2026)),
        "preliminary": list(range(2000, 2027)),
    }
    for status, years in expected.items():
        observed = (
            anchors.filter(pl.col("benchmark_status") == status)["benchmark_year"]
            .sort()
            .to_list()
        )
        if observed != years:
            raise ValueError(f"{status} benchmark anchor years are incomplete")


def _validate_2025_rulings(result: AnnualBuild) -> None:
    benchmark = result.benchmarks.filter(
        (pl.col("benchmark_year") == 2025)
        & (pl.col("benchmark_status") == "final")
        & (pl.col("row_kind") == "anchor")
        & (pl.col("sector") == "00")
    )
    if benchmark.height != 1:
        raise ValueError("the 2025 final benchmark ruling is missing or duplicated")
    row = benchmark.row(0, named=True)
    expected = {
        "revision_thousands": -861.0,
        "article_revision_thousands": -862.0,
        "article_difference_thousands": -1.0,
        "discrepancy_status": "unexplained",
    }
    if (
        any(row[column] != value for column, value in expected.items())
        or 12 not in row["footnotes"]
    ):
        raise ValueError("the 2025 benchmark discrepancy ruling changed")

    events = result.reconstruction_events.filter(pl.col("benchmark_year") == 2025)
    if set(events["event_id"]) != EXPECTED_2025_EVENTS or events.height != len(
        EXPECTED_2025_EVENTS
    ):
        raise ValueError("the two required 2025 reconstruction events are incomplete")


def _validate_birth_death(result: AnnualBuild) -> None:
    government = result.birth_death.filter(pl.col("sector") == "90")
    if (
        government.is_empty()
        or government["value_thousands"].is_null().any()
        or government["value_thousands"].ne(0.0).any()
        or set(government["transformation"]) != {"government_structural_zero"}
    ):
        raise ValueError("government birth-death is not structurally zero")
    total_keys = (
        result.birth_death.filter(pl.col("sector") == "00")
        .select(BIRTH_DEATH_KEY)
        .unique()
        .sort(BIRTH_DEATH_KEY)
    )
    government_keys = government.select(BIRTH_DEATH_KEY).unique().sort(BIRTH_DEATH_KEY)
    if government.height != total_keys.height or not government_keys.equals(total_keys):
        raise ValueError("government birth-death does not cover every series key")


def _validate_sample(result: AnnualBuild) -> None:
    sample = result.sample_panel
    expected_keys = {
        (year, sector) for year in range(2002, 2026) for sector in SAMPLE_SECTORS
    }
    observed_keys = set(sample.select("benchmark_year", "sector").rows())
    if sample.height != len(expected_keys) or observed_keys != expected_keys:
        raise ValueError("annual sample year/sector coverage is incomplete")
    if set(sample["frequency"]) != {"annual"}:
        raise ValueError("sample measures must remain annual")
    if any(
        month != date(year, 3, 1)
        for year, month in sample.select(
            "benchmark_year", "reference_month"
        ).iter_rows()
    ):
        raise ValueError("annual sample rows must refer to March")
    missing_2012 = sample.filter(pl.col("benchmark_year") == 2012)
    if missing_2012.height != len(SAMPLE_SECTORS) or set(
        missing_2012["coverage_status"]
    ) != {"archive_gap"}:
        raise ValueError("March 2012 coverage must remain an archive gap")
    expected_breaks = {(2016, sector) for sector in SAMPLE_SECTORS}
    observed_breaks = set(
        sample.filter(pl.col("rse_definition_break"))
        .select("benchmark_year", "sector")
        .rows()
    )
    if (
        sample["rse_definition_break"].is_null().any()
        or observed_breaks != expected_breaks
    ):
        raise ValueError("the 2016 RSE definition break is incomplete")


def _validate_domain_contracts(result: AnnualBuild) -> None:
    _require_anchor_years(result)
    _validate_2025_rulings(result)
    _validate_birth_death(result)
    if set(result.qcew_revisions["unit"]) != {"jobs"}:
        raise ValueError("QCEW revision values must remain in jobs")
    if set(result.qcew_precision["unit"]) != {"thousands"}:
        raise ValueError("QCEW precision must be in thousands")
    _validate_sample(result)


def validate(result: AnnualBuild, release_index: pl.DataFrame) -> None:
    """Raise on any Stage 4 exit-criterion violation."""
    _validate_derived_schema(result)
    _validate_provenance(result, release_index)
    _validate_dates(result, release_index)
    _validate_domain_contracts(result)
