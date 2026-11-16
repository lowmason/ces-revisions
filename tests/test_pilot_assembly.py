"""The Stage 6 aggregate panel, assembled from the Stage 3, 4, and 5 artifacts."""

from datetime import date
from functools import cache
from pathlib import Path

import annual_data
import numpy as np
import operator_data
import polars as pl
import pytest
import vintage_data

from ces_revisions.pilot.assembly import (
    ARTIFACT_PATHS,
    MASK_REASONS,
    REPEATED_EMPLOY_ORIGINAL,
    REPEATED_EMPLOY_VINTAGE,
    build_aggregate_panel,
    carrier_cells,
    employ_table_disagreements,
    employ_without_repeats,
    read_artifacts,
)
from ces_revisions.pilot.panel import (
    ANCHOR_COLUMN,
    ANCHOR_YEARS,
    CELL_SCHEMA,
    COLUMNS,
    REGIME_SHIFT_MONTH,
    AggregatePanel,
    benchmark_year_of,
)
from ces_revisions.vintages.months import add_months, month_range


@cache
def inputs() -> dict[str, pl.DataFrame]:
    return {
        "stage_changes": vintage_data.stage_changes(),
        "stage_labels": vintage_data.labels(),
        "table_changes": vintage_data.table_changes(),
        "release_index": vintage_data.index(),
        "benchmark_fixtures": operator_data.artifact("benchmark_fixtures"),
        "publication_calendar": annual_data.artifact("publication_calendar"),
    }


@cache
def built() -> AggregatePanel:
    return build_aggregate_panel(**inputs())


def cell(panel: AggregatePanel, month: date, column: str) -> dict:
    return panel.cells.row(
        by_predicate=(pl.col("reference_month") == month)
        & (pl.col("column") == column),
        named=True,
    )


def value(panel: AggregatePanel, month: date, column: str) -> float:
    return panel.y[panel.reference_months.index(month), COLUMNS.index(column)]


def final_benchmark_dates() -> set[date]:
    calendar = inputs()["publication_calendar"]
    return set(
        calendar.filter(pl.col("publication_kind") == "benchmark_final")[
            "publication_date"
        ]
    )


def test_the_panel_spans_every_reference_month_from_1979_to_the_release_index():
    panel = built()
    assert panel.num_steps == 572
    assert panel.reference_months == tuple(
        month_range(date(1979, 1, 1), date(2026, 8, 1))
    )
    assert panel.reference_months[-1] == vintage_data.index()["reference_month"].max()
    assert panel.y.shape == (572, 4)
    assert panel.cells.height == 572 * 4
    assert dict(panel.cells.schema) == CELL_SCHEMA


def test_every_cell_has_one_row_and_is_nan_exactly_where_it_is_masked():
    panel = built()
    cells = panel.cells
    assert not cells.select("reference_month", "column").is_duplicated().any()
    assert cells["column"].to_list() == list(COLUMNS) * panel.num_steps
    masked = cells["mask_reason"].is_not_null().to_numpy()
    assert np.array_equal(np.isnan(panel.y).ravel(), masked)
    assert set(cells["mask_reason"].drop_nulls()) <= set(MASK_REASONS)
    counts = {
        (row["column"], row["mask_reason"]): row["len"]
        for row in cells.group_by("column", "mask_reason").len().iter_rows(named=True)
    }
    assert counts == {
        ("F", None): 571,
        ("F", "missing_vintage"): 1,
        ("S", None): 523,
        ("S", "benchmark_carrier"): 46,
        ("S", "missing_vintage"): 2,
        ("S", "right_censored"): 1,
        ("T", None): 522,
        ("T", "benchmark_carrier"): 46,
        ("T", "missing_vintage"): 2,
        ("T", "right_censored"): 2,
        ("anchor", None): 23,
        ("anchor", "no_anchor"): 549,
    }
    unobserved = cells.filter(
        pl.col("mask_reason").is_in(["missing_vintage", "right_censored"])
    )
    assert sorted(
        zip(unobserved["reference_month"], unobserved["column"], strict=True)
    ) == [
        (date(2003, 3, 1), "T"),
        (date(2003, 4, 1), "S"),
        (date(2003, 4, 1), "T"),
        (date(2025, 9, 1), "S"),
        (date(2025, 10, 1), "F"),
        (date(2026, 7, 1), "T"),
        (date(2026, 8, 1), "S"),
        (date(2026, 8, 1), "T"),
    ]
    assert unobserved["value_thousands"].is_null().all()


def test_each_closing_cell_is_read_from_one_source_chosen_by_reference_month():
    panel = built()
    closing = panel.cells.filter(pl.col("column") != "anchor")
    early = closing.filter(pl.col("reference_month") < REGIME_SHIFT_MONTH)
    assert set(early["source"]) == {"cesnaicsrev"}
    late = closing.filter(pl.col("reference_month") >= REGIME_SHIFT_MONTH)
    from_table = late.filter(pl.col("source") == "cesnaicsrev")
    frontier = inputs()["stage_labels"].filter(
        (pl.col("source") == "cesvinall")
        & (pl.col("sector") == "00")
        & (pl.col("seasonal_status") == "NSA")
        & (pl.col("status") == "beyond_frontier")
        & pl.col("release_stage").is_in(["F", "S", "T"])
    )
    assert from_table.height == frontier.height == 21
    assert sorted(
        zip(from_table["reference_month"], from_table["column"], strict=True)
    ) == sorted(
        zip(frontier["reference_month"], frontier["release_stage"], strict=True)
    )
    assert from_table["value_thousands"].is_not_null().all()
    assert (
        from_table["reference_month"].min(),
        from_table["reference_month"].max(),
    ) == (
        date(2025, 12, 1),
        date(2026, 8, 1),
    )
    assert set(late.filter(pl.col("source") != "cesnaicsrev")["source"]) == {
        "cesvinall"
    }

    keys = ["reference_month", "release_stage"]
    table = (
        inputs()["table_changes"]
        .filter(pl.col("seasonal_status") == "NSA")
        .select(*keys, source=pl.lit("cesnaicsrev"), expected="value_thousands")
    )
    files = (
        inputs()["stage_changes"]
        .filter(
            (pl.col("sector") == "00")
            & (pl.col("seasonal_status") == "NSA")
            & pl.col("release_stage").is_in(["F", "S", "T"])
        )
        .select(*keys, source=pl.lit("cesvinall"), expected="change_thousands")
    )
    compared = closing.rename({"column": "release_stage"}).join(
        pl.concat([table, files]), on=[*keys, "source"], how="left"
    )
    assert compared.height == closing.height
    assert (
        compared["value_thousands"].eq_missing(compared["expected"].cast(pl.Float64))
    ).all()
    # November 2003's third estimate is a cross-release change in the table (46); the
    # cell keeps the vintage file's same-release change.
    november = cell(panel, date(2003, 11, 1), "T")
    assert (november["source"], november["value_thousands"]) == ("cesvinall", 147.0)


def test_the_1979_leg_is_partially_observed_with_exactly_one_2003_shift():
    panel = built()
    months = np.array(panel.reference_months)
    assert np.array_equal(panel.pre_2003, months < REGIME_SHIFT_MONTH)
    assert np.count_nonzero(np.diff(panel.pre_2003.astype(np.int8))) == 1
    assert panel.reference_months[int(panel.pre_2003.sum())] == date(2003, 5, 1)
    leg = panel.y[panel.pre_2003, :ANCHOR_COLUMN]
    observed = (~np.isnan(leg)).sum(axis=0)
    assert leg.shape == (292, 3)
    assert observed.tolist() == [292, 268, 267]
    anchored = [
        month
        for month, anchor in zip(
            panel.reference_months, panel.y[:, ANCHOR_COLUMN], strict=True
        )
        if not np.isnan(anchor)
    ]
    assert anchored[0] == date(2003, 3, 1)
    assert all(month.year >= 2003 for month in anchored)
    for month, column in (
        (date(2003, 3, 1), "T"),
        (date(2003, 4, 1), "S"),
        (date(2003, 4, 1), "T"),
    ):
        assert np.isnan(value(panel, month, column))
        assert cell(panel, month, column)["mask_reason"] == "missing_vintage"
    assert panel.y[0].tolist()[:ANCHOR_COLUMN] == [-1681.0, -1818.0, -1759.0]


def test_final_benchmark_releases_mask_their_second_and_third_estimates():
    panel = built()
    finals = final_benchmark_dates()
    carriers = panel.cells.filter(pl.col("mask_reason") == "benchmark_carrier")
    assert set(carriers["column"]) == {"S", "T"}
    assert set(carriers["published_date"]) <= finals
    assert carriers["value_thousands"].is_not_null().all()
    assert carriers.filter(pl.col("reference_month") < REGIME_SHIFT_MONTH).height == 46
    assert carriers.filter(pl.col("reference_month") >= REGIME_SHIFT_MONTH).height == 46
    first = panel.cells.filter(pl.col("column") == "F")
    assert first.filter(pl.col("published_date").is_in(finals)).height == 47
    assert first.filter(pl.col("mask_reason") == "benchmark_carrier").is_empty()

    labels = carrier_cells(inputs()["stage_labels"], inputs()["publication_calendar"])
    assert labels["release_stage"].value_counts().sort("release_stage").rows() == [
        ("F", 47),
        ("S", 46),
        ("T", 46),
    ]
    masked = labels.filter(pl.col("release_stage") != "F")
    assert sorted(
        zip(masked["reference_month"], masked["release_stage"], strict=True)
    ) == (sorted(zip(carriers["reference_month"], carriers["column"], strict=True)))
    per_year = dict(masked.group_by("publication_id").len().iter_rows())
    expected = {
        f"benchmark_final_{year}": 2 for year in range(1979, 2026) if year != 2002
    }
    assert per_year == expected
    # The March 2002 benchmark arrived in the May 2003 release, whose second and third
    # estimates are the 2003-gap cells: missing, so undated and never carrier-masked.
    gap = labels.filter(pl.col("publication_id") == "benchmark_final_2002")
    assert gap.select("reference_month", "release_stage").rows() == [
        (date(2003, 5, 1), "F")
    ]
    assert cell(panel, date(2003, 12, 1), "S")["mask_reason"] == "benchmark_carrier"
    assert cell(panel, date(2003, 11, 1), "T")["published_date"] == date(2004, 2, 6)
    assert cell(panel, date(2004, 1, 1), "F")["mask_reason"] is None


def test_anchors_are_the_march_wedge_anchors_and_offsets_sum_earlier_reconstructions():
    panel = built()
    marches = (
        inputs()["benchmark_fixtures"]
        .filter((pl.col("sector") == "00") & (pl.col("wedge_weight") == 1.0))
        .sort("benchmark_year")
    )
    anchor_months = [date(year, 3, 1) for year in ANCHOR_YEARS]
    assert marches["reference_month"].to_list() == anchor_months
    rows = [panel.reference_months.index(month) for month in anchor_months]
    assert np.array_equal(
        panel.y[rows, ANCHOR_COLUMN], marches["wedge_anchor_thousands"].to_numpy()
    )
    assert int((~np.isnan(panel.y[:, ANCHOR_COLUMN])).sum()) == 23
    offsets = {
        month.year: float(panel.anchor_offset[row])
        for month, row in zip(anchor_months, rows, strict=True)
    }
    assert offsets == {
        **dict.fromkeys(range(2003, 2011), 0.0),
        **dict.fromkeys(range(2011, 2014), 4.0),
        **dict.fromkeys(range(2014, 2016), 470.0),
        **dict.fromkeys(range(2016, 2018), 443.0),
        **dict.fromkeys(range(2018, 2020), 446.0),
        **dict.fromkeys(range(2020, 2026), 430.0),
    }
    terms = dict(
        zip(
            marches["benchmark_year"],
            marches["reconstruction_term_thousands"],
            strict=True,
        )
    )
    assert offsets[2014] == terms[2010] + terms[2013] == 4.0 + 466.0
    assert offsets[2025] == sum(terms[year] for year in range(2003, 2025))
    elsewhere = np.ones(panel.num_steps, dtype=bool)
    elsewhere[rows] = False
    assert not panel.anchor_offset[elsewhere].any()
    anchor = cell(panel, date(2013, 3, 1), "anchor")
    assert (anchor["status"], anchor["source"], anchor["published_date"]) == (
        "anchor",
        "benchmark_fixtures",
        date(2014, 2, 7),
    )


def test_crisis_and_benchmark_year_flags_follow_the_reference_month():
    panel = built()
    crisis = [
        month
        for month, flag in zip(panel.reference_months, panel.crisis, strict=True)
        if flag
    ]
    assert crisis == [date(2020, month, 1) for month in range(3, 7)]
    assert panel.benchmark_year.tolist() == [
        benchmark_year_of(month) for month in panel.reference_months
    ]
    assert panel.benchmark_year[0] == 1979
    assert panel.benchmark_year[panel.reference_months.index(date(2003, 4, 1))] == 2004


def test_read_artifacts_reads_the_six_inputs_and_rebuilds_the_same_panel(
    tmp_path: Path,
):
    frames = {
        **inputs(),
        "table_changes": vintage_data.long_panel(),
    }
    for name, path in ARTIFACT_PATHS.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        frames[name].write_parquet(tmp_path / path)
    read = read_artifacts(tmp_path)
    assert list(read) == list(ARTIFACT_PATHS)
    assert set(read["table_changes"]["source"]) == {"cesnaicsrev"}
    assert read["table_changes"].height == vintage_data.table_changes().height
    panel, expected = build_aggregate_panel(**read), built()
    assert panel.reference_months == expected.reference_months
    for name in ("y", "anchor_offset", "pre_2003", "crisis", "benchmark_year"):
        assert getattr(panel, name).tobytes() == getattr(expected, name).tobytes()
    assert panel.cells.equals(expected.cells)


def test_the_repeated_october_2025_employ_vintage_is_dropped():
    long_panel = vintage_data.long_panel()
    employ = long_panel.filter(pl.col("source") == "rtdsm_employ")
    october = employ.filter(pl.col("vintage_id") == REPEATED_EMPLOY_VINTAGE)
    september = employ.filter(pl.col("vintage_id") == REPEATED_EMPLOY_ORIGINAL)
    assert october.height == september.height == 944
    assert october.sort("reference_month")["value_thousands"].equals(
        september.sort("reference_month")["value_thousands"]
    )
    kept = employ_without_repeats(long_panel)
    assert kept.height == employ.height - 944
    assert REPEATED_EMPLOY_VINTAGE not in set(kept["vintage_id"])
    assert kept["vintage_id"].n_unique() == 570
    assert (
        kept.group_by("release_month")
        .agg(pl.col("vintage_id").n_unique())["vintage_id"]
        .max()
        == 1
    )


def test_an_october_2025_employ_vintage_that_is_not_a_repeat_raises():
    long_panel = vintage_data.long_panel().filter(pl.col("source") == "rtdsm_employ")
    changed = long_panel.with_columns(
        value_thousands=pl.when(
            (pl.col("vintage_id") == REPEATED_EMPLOY_VINTAGE)
            & (pl.col("reference_month") == date(2025, 8, 1))
        )
        .then(pl.col("value_thousands") + 1)
        .otherwise("value_thousands")
    )
    with pytest.raises(ValueError, match="not an exact repeat"):
        employ_without_repeats(changed)
    with pytest.raises(ValueError, match="not an exact repeat"):
        employ_without_repeats(
            long_panel.filter(pl.col("vintage_id") != REPEATED_EMPLOY_VINTAGE)
        )


def test_another_doubled_employ_release_raises():
    long_panel = vintage_data.long_panel().filter(pl.col("source") == "rtdsm_employ")
    doubled = long_panel.filter(
        pl.col("vintage_id") == "rtdsm:EMPLOY24M6"
    ).with_columns(vintage_id=pl.lit("rtdsm:EMPLOY24M6bis"))
    with pytest.raises(ValueError, match="two EMPLOY vintages"):
        employ_without_repeats(pl.concat([long_panel, doubled]))


def test_employ_disagrees_with_the_revision_table_in_ten_pre_1997_and_three_2020_cells():
    disagreements = employ_table_disagreements(
        vintage_data.long_panel(), vintage_data.table_changes()
    )
    assert disagreements.columns == [
        "reference_month",
        "release_stage",
        "vintage_id",
        "employ_thousands",
        "table_thousands",
    ]
    assert disagreements.rows() == [
        (date(1981, 3, 1), "T", "rtdsm:EMPLOY81M6", 52, 53),
        (date(1982, 11, 1), "T", "rtdsm:EMPLOY83M2", -127, -110),
        (date(1983, 5, 1), "T", "rtdsm:EMPLOY83M8", 331, 320),
        (date(1983, 9, 1), "T", "rtdsm:EMPLOY83M12", 1103, 1116),
        (date(1983, 11, 1), "T", "rtdsm:EMPLOY84M2", 271, 268),
        (date(1984, 6, 1), "T", "rtdsm:EMPLOY84M9", 349, 367),
        (date(1984, 11, 1), "T", "rtdsm:EMPLOY85M2", 337, 340),
        (date(1984, 12, 1), "T", "rtdsm:EMPLOY85M3", 184, 187),
        (date(1996, 8, 1), "S", "rtdsm:EMPLOY96M10", 257, 241),
        (date(1996, 8, 1), "T", "rtdsm:EMPLOY96M11", 280, 263),
        (date(2020, 2, 1), "T", "rtdsm:EMPLOY20M5", 230, 251),
        (date(2020, 3, 1), "S", "rtdsm:EMPLOY20M5", -870, -881),
        (date(2020, 4, 1), "F", "rtdsm:EMPLOY20M5", -20500, -20537),
    ]
    pre_1997 = disagreements.filter(pl.col("reference_month") < date(1997, 1, 1))
    assert pre_1997.height == 10
    # Same release on both sides: each EMPLOY vintage holds the release that made the
    # table's estimate, the reference month plus the stage's offset.
    releases = dict(
        vintage_data.rtdsm().select("vintage_id", "release_month").unique().iter_rows()
    )
    for row in disagreements.iter_rows(named=True):
        offset = "FST".index(row["release_stage"])
        assert releases[row["vintage_id"]] == add_months(row["reference_month"], offset)
