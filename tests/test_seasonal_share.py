"""The seasonal-adjustment shares of SA revision variance (roadmap outcome O9)."""

import archive_inventory
import polars as pl
import pytest
import seasonal_share as seasonal_share_report
import vintage_data

from ces_revisions.annual import sample
from ces_revisions.seasonal_share import REGIME_LABELS, seasonal_shares
from ces_revisions.vintages.accounting import REGIMES, decomposition

SCHEMA = {
    "source": pl.String,
    "sector": pl.String,
    "transition": pl.String,
    "regime": pl.String,
    "months": pl.UInt32,
    "var_revision_sa": pl.Float64,
    "var_revision_nsa": pl.Float64,
    "var_adjustment_change": pl.Float64,
    "cov_revision_nsa_adjustment_change": pl.Float64,
    "variance_identity_gap": pl.Float64,
}


def decomposition_row(
    var_sa: float | None, var_adj: float, cov: float, gap: float = 0.0, **keys: str
) -> pl.DataFrame:
    """One decomposition row whose gap column is the identity's residual for its variances."""
    var_nsa = None if var_sa is None else var_sa - var_adj + 2 * cov - gap
    row = {
        "source": "cesvinall",
        "sector": "10",
        "transition": "F_S",
        "regime": "2023_present",
        "months": 34,
        "var_revision_sa": var_sa,
        "var_revision_nsa": var_nsa,
        "var_adjustment_change": var_adj,
        "cov_revision_nsa_adjustment_change": cov,
        "variance_identity_gap": gap,
        **keys,
    }
    return pl.DataFrame([row], schema=SCHEMA)


@pytest.fixture(scope="module")
def built() -> pl.DataFrame:
    return seasonal_shares(decomposition(vintage_data.identity_terms()))


def test_shares_of_a_hand_built_row():
    row = seasonal_shares(decomposition_row(100.0, 50.0, 20.0)).row(0, named=True)
    assert row["share_ratio"] == pytest.approx(0.5)
    assert row["share_adjustment"] == pytest.approx(0.3)
    assert row["share_nsa"] == pytest.approx(0.7)


def test_shares_are_not_clipped_to_the_unit_interval():
    row = seasonal_shares(decomposition_row(10.0, 60.0, 40.0)).row(0, named=True)
    assert row["share_ratio"] == pytest.approx(6.0)
    assert row["share_adjustment"] == pytest.approx(2.0)
    assert row["share_nsa"] == pytest.approx(-1.0)


@pytest.mark.parametrize("var_sa", [0.0, None])
def test_a_group_without_sa_variance_is_refused(var_sa):
    with pytest.raises(ValueError, match="cesvinall 10 F_S 2023_present"):
        seasonal_shares(decomposition_row(var_sa, 5.0, 1.0))


def test_regime_labels_follow_the_accounting_regimes():
    assert list(REGIME_LABELS) == [name for name, _, _ in REGIMES]
    assert REGIME_LABELS == {
        "pre_2003": "1979-01 to 2003-04",
        "2003_2019": "2003-05 to 2019-12",
        "2020_2022": "2020-01 to 2022-12",
        "2023_present": "2023-01 on",
    }


def test_the_two_splits_partition_the_built_variance(built):
    splits = built["share_adjustment"] + built["share_nsa"]
    complement = 1 - built["variance_identity_gap"] / built["var_revision_sa"]
    assert (splits - complement).abs().max() <= 1e-12
    assert (splits - 1).abs().max() <= 1e-12


def test_built_shares_run_in_time_order(built):
    assert built.height == 110
    assert built["transition"].rle().struct.field("value").to_list() == [
        "F_S",
        "S_T",
        "T_B",
    ]
    first_sector = built.filter(
        (pl.col("sector") == "00") & (pl.col("transition") == "F_S")
    )
    assert first_sector["regime"].to_list() == [
        "pre_2003",
        "2003_2019",
        "2020_2022",
        "2023_present",
    ]
    assert built.filter(
        (pl.col("transition") == "T_B") & (pl.col("regime") == "pre_2003")
    ).is_empty()


def test_built_shares_keep_values_outside_the_unit_interval(built):
    above = built.filter(pl.col("share_ratio") > 1)
    assert above.height == 12
    assert round(above["share_ratio"].max(), 2) == 9.93
    below = built.filter(pl.col("share_adjustment") < 0)
    assert below.height == 27
    assert set(below["transition"]) == {"F_S"}
    assert built.filter(pl.col("share_adjustment") > 1).height == 2


def test_the_splits_add_to_one_less_the_identity_gap_over_variance():
    row = seasonal_shares(decomposition_row(100.0, 50.0, 20.0, gap=2.0)).row(
        0, named=True
    )
    assert row["share_adjustment"] == pytest.approx(0.3)
    assert row["share_nsa"] == pytest.approx(0.68)
    assert row["share_adjustment"] + row["share_nsa"] == pytest.approx(0.98)


def test_the_refusal_names_the_offending_row_among_good_ones():
    table = pl.concat(
        [
            decomposition_row(100.0, 50.0, 20.0, sector="10"),
            decomposition_row(-1.0, 5.0, 1.0, sector="20", transition="T_B"),
        ]
    )
    with pytest.raises(ValueError, match="cesvinall 20 T_B 2023_present"):
        seasonal_shares(table)


def test_rows_are_ordered_by_transition_then_sector_then_regime():
    keys = [
        ("T_B", "10", "2023_present"),
        ("F_S", "20", "2003_2019"),
        ("S_T", "00", "2020_2022"),
        ("F_S", "10", "pre_2003"),
        ("F_S", "10", "2023_present"),
        ("F_S", "10", "2003_2019"),
    ]
    table = pl.concat(
        [
            decomposition_row(10.0, 5.0, 1.0, transition=t, sector=s, regime=r)
            for t, s, r in keys
        ]
    )
    ordered = seasonal_shares(table).select("transition", "sector", "regime").rows()
    assert ordered == [
        ("F_S", "10", "pre_2003"),
        ("F_S", "10", "2003_2019"),
        ("F_S", "10", "2023_present"),
        ("F_S", "20", "2003_2019"),
        ("S_T", "00", "2020_2022"),
        ("T_B", "10", "2023_present"),
    ]


def test_every_input_column_is_kept_before_the_three_shares():
    table = decomposition_row(100.0, 50.0, 20.0)
    shares = seasonal_shares(table)
    assert shares.columns == [
        *table.columns,
        "share_ratio",
        "share_adjustment",
        "share_nsa",
    ]
    assert list(shares.select(pl.col("^share_.*$")).schema.values()) == [pl.Float64] * 3
    assert shares.select(table.columns).equals(table)


def test_sector_titles_follow_the_coverage_tables():
    titles = seasonal_share_report.SECTOR_TITLES
    assert list(titles) == list(sample.SECTORS)
    for code, title in titles.items():
        assert sample._LEGACY_SECTOR_BY_TITLE[sample._normalized_title(title)] == code


def test_report_lays_out_one_row_per_sector_and_regime(built):
    first_to_second = seasonal_share_report.render_transition(built, "F_S").splitlines()
    assert len(first_to_second) == 39
    assert first_to_second[2].startswith(
        "| 00 Total nonfarm | cesnaicsrev | 1979-01 to 2003-04 | 291 |"
    )
    third_to_benchmark = seasonal_share_report.render_transition(built, "T_B")
    assert len(third_to_benchmark.splitlines()) == 38
    assert "cesnaicsrev" not in third_to_benchmark


def test_report_shows_the_gap_and_unclipped_shares():
    shares = seasonal_shares(decomposition_row(10.0, 60.0, 40.0, gap=1.5e-11))
    rendered = seasonal_share_report.render_transition(shares, "F_S")
    assert "| 6.00 | 2.00 | -1.00 | 1.5e-11 |" in rendered


def test_bounds_count_shares_outside_the_unit_interval(built):
    lines = seasonal_share_report.render_bounds(built).splitlines()
    for prefix in (
        "| Ratio | All | 110 | 0 | 12 |",
        "| Adjustment | F→S | 37 | 27 |",
        "| NSA | All | 110 | 2 | 27 |",
    ):
        assert any(line.startswith(prefix) for line in lines), prefix


def test_report_refuses_without_a_built_panel(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(seasonal_share_report, "PANEL_DIR", tmp_path)
    assert seasonal_share_report.main(["report"]) == 1
    assert "scripts/vintage_sources.py build" in capsys.readouterr().err


def test_report_fills_the_four_generated_blocks(tmp_path, monkeypatch, capsys, built):
    decomposition(vintage_data.identity_terms()).write_parquet(
        tmp_path / seasonal_share_report.INPUT
    )
    names = ["stage-3-f-s", "stage-3-s-t", "stage-3-t-b", "stage-3-bounds"]
    findings = tmp_path / "findings.md"
    findings.write_text(
        "".join(
            f"<!-- BEGIN GENERATED {name} -->\n<!-- END GENERATED {name} -->\n"
            for name in names
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(seasonal_share_report, "PANEL_DIR", tmp_path)
    monkeypatch.setattr(seasonal_share_report, "FINDINGS", findings)
    assert seasonal_share_report.main(["report"]) == 0
    expected = {
        "stage-3-f-s": seasonal_share_report.render_transition(built, "F_S"),
        "stage-3-s-t": seasonal_share_report.render_transition(built, "S_T"),
        "stage-3-t-b": seasonal_share_report.render_transition(built, "T_B"),
        "stage-3-bounds": seasonal_share_report.render_bounds(built),
    }
    document = findings.read_text(encoding="utf-8")
    for name, body in expected.items():
        assert archive_inventory.generated_block(document, name) == f"\n{body}\n\n"
    assert "report: 110 rows" in capsys.readouterr().out


TRANSITION_HEADER = [
    "Sector",
    "Source",
    "Regime",
    "Months",
    "Var SA",
    "Var NSA",
    "Var adjustment change",
    "Cov(NSA, adjustment change)",
    "Ratio share",
    "Adjustment share",
    "NSA share",
    "Identity gap",
]
BLOCK_NAMES = ["stage-3-f-s", "stage-3-s-t", "stage-3-t-b", "stage-3-bounds"]


def stage_report(tmp_path, monkeypatch, before="", after=""):
    """Point the report at the built decomposition and a findings file of four empty blocks."""
    decomposition(vintage_data.identity_terms()).write_parquet(
        tmp_path / seasonal_share_report.INPUT
    )
    blocks = "".join(
        f"<!-- BEGIN GENERATED {name} -->\n<!-- END GENERATED {name} -->\n"
        for name in BLOCK_NAMES
    )
    findings = tmp_path / "findings.md"
    findings.write_text(before + blocks + after, encoding="utf-8")
    monkeypatch.setattr(seasonal_share_report, "PANEL_DIR", tmp_path)
    monkeypatch.setattr(seasonal_share_report, "FINDINGS", findings)
    return findings


def test_report_prints_each_identity_term_in_its_column():
    shares = seasonal_shares(decomposition_row(10.26, 60.0, 40.0, sector="10"))
    lines = seasonal_share_report.render_transition(shares, "F_S").splitlines()
    assert lines[0] == f"| {' | '.join(TRANSITION_HEADER)} |"
    assert lines[1] == "|---|---|---|" + "---:|" * 9
    assert lines[2] == (
        "| 10 Mining and logging | cesvinall | 2023-01 on | 34 | 10.3 | 30.3 | 60.0 "
        "| 40.0 | 5.85 | 1.95 | -0.95 | 0.0e+00 |"
    )


def test_bounds_count_only_values_strictly_outside_the_unit_interval():
    table = pl.concat(
        [
            decomposition_row(10.0, 60.0, 40.0, transition="F_S"),
            decomposition_row(10.0, 10.0, 10.0, transition="S_T"),
            decomposition_row(100.0, 50.0, 20.0, transition="T_B"),
        ]
    )
    rendered = seasonal_share_report.render_bounds(seasonal_shares(table))
    assert rendered.splitlines() == [
        "| Share | Transition | Rows | Below 0 | Above 1 | Min | Max |",
        "|---|---|---:|---:|---:|---:|---:|",
        "| Ratio | F→S | 1 | 0 | 1 | 6.00 | 6.00 |",
        "| Ratio | S→T | 1 | 0 | 0 | 1.00 | 1.00 |",
        "| Ratio | T→B | 1 | 0 | 0 | 0.50 | 0.50 |",
        "| Ratio | All | 3 | 0 | 1 | 0.50 | 6.00 |",
        "| Adjustment | F→S | 1 | 0 | 1 | 2.00 | 2.00 |",
        "| Adjustment | S→T | 1 | 0 | 0 | 0.00 | 0.00 |",
        "| Adjustment | T→B | 1 | 0 | 0 | 0.30 | 0.30 |",
        "| Adjustment | All | 3 | 0 | 1 | 0.00 | 2.00 |",
        "| NSA | F→S | 1 | 1 | 0 | -1.00 | -1.00 |",
        "| NSA | S→T | 1 | 0 | 0 | 1.00 | 1.00 |",
        "| NSA | T→B | 1 | 0 | 0 | 0.70 | 0.70 |",
        "| NSA | All | 3 | 1 | 0 | -1.00 | 1.00 |",
    ]


def test_report_keeps_the_prose_around_the_blocks(tmp_path, monkeypatch):
    findings = stage_report(
        tmp_path, monkeypatch, before="Prose before.\n\n", after="\nProse after.\n"
    )
    assert seasonal_share_report.main(["report"]) == 0
    document = findings.read_text(encoding="utf-8")
    assert document.startswith("Prose before.\n\n<!-- BEGIN GENERATED stage-3-f-s -->")
    assert document.endswith("<!-- END GENERATED stage-3-bounds -->\n\nProse after.\n")


def test_report_run_twice_gives_the_same_file(tmp_path, monkeypatch):
    findings = stage_report(tmp_path, monkeypatch)
    assert seasonal_share_report.main(["report"]) == 0
    first = findings.read_text(encoding="utf-8")
    assert seasonal_share_report.main(["report"]) == 0
    assert findings.read_text(encoding="utf-8") == first
