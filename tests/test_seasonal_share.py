"""The seasonal-adjustment shares of SA revision variance (roadmap outcome O9)."""

import polars as pl
import pytest
import vintage_data

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
