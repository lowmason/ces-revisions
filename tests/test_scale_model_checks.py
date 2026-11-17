"""The scale model's predictive statistics, p-values, and log likelihood blocks."""

from datetime import date

import numpy as np
import pytest
from scipy import stats

from ces_revisions.scale_model.checks import (
    DAY_CELLS,
    acf_at_lag,
    beta_sensitivity,
    month_gaps,
    pearson_kurtosis,
    predictive_p_values,
    predictive_statistics,
    prior_predictive_record,
    sum_by_year,
)
from ces_revisions.scale_model.data import ScaleData


def months(*pairs) -> tuple[date, ...]:
    return tuple(date(year, month, 1) for year, month in pairs)


def test_month_gaps_pair_months_by_calendar_distance_not_position():
    reference = months((2019, 12), (2020, 1), (2020, 2), (2022, 1), (2022, 2))
    first, second = month_gaps(reference, 1)
    assert list(zip(first, second, strict=True)) == [(0, 1), (1, 2), (3, 4)]
    # December 2019 has no partner: December 2021 is not in the sample.
    first, second = month_gaps(reference, 24)
    assert list(zip(first, second, strict=True)) == [(1, 3), (2, 4)]
    first, second = month_gaps(reference, 7)
    assert first.size == 0 and second.size == 0


def test_acf_uses_only_the_pairs_at_the_lag():
    x = np.array([1.0, 3.0, 2.0, 10.0, -4.0])
    reference = months((2019, 12), (2020, 1), (2020, 2), (2022, 1), (2022, 2))
    centered = x - x.mean()
    pairs = [(0, 1), (1, 2), (3, 4)]
    expected = np.mean([centered[i] * centered[j] for i, j in pairs]) / np.mean(
        centered**2
    )
    assert acf_at_lag(x, reference, 1) == pytest.approx(expected)
    # Draws on leading axes are handled one series at a time.
    stacked = acf_at_lag(np.stack([x, 2 * x]), reference, 1)
    np.testing.assert_allclose(stacked, [expected, expected])
    # A lag with no pair is undefined, not an error.
    assert np.isnan(acf_at_lag(x, reference, 7))
    assert np.isnan(acf_at_lag(np.stack([x, x]), reference, 7)).all()


def test_pearson_kurtosis_matches_scipy():
    x = np.random.default_rng(3).standard_t(6, size=(2, 400))
    np.testing.assert_allclose(
        pearson_kurtosis(x), stats.kurtosis(x, axis=-1, fisher=False), rtol=1e-12
    )


def tiny_data() -> ScaleData:
    reference = months((2013, 1), (2013, 2), (2013, 8), (2014, 1), (2014, 8))
    return ScaleData(
        series=("NSA", "F_S"),
        reference_months=reference,
        years=(2013, 2014),
        revision=np.array([10.0, -20.0, 30.0, -40.0, 50.0]),
        collection_days=np.array([13.0, 15.0, 19.0, 14.0, 16.0]),
        window=np.zeros(5),
        year_window=np.zeros(2),
        five_week=np.array([0.0, 0.0, 1.0, 0.0, 1.0]),
        lapse=np.zeros(5),
        year=np.array([0, 0, 0, 1, 1]),
        month=np.array([0, 1, 7, 0, 7]),
    )


def test_statistics_average_absolute_revisions_by_month_interval_and_day_cell():
    data = tiny_data()
    r = data.revision
    values = predictive_statistics(r, np.zeros(5), np.ones(5), data)
    assert values["mean_abs_month_01"] == pytest.approx(25.0)
    assert values["mean_abs_month_08"] == pytest.approx(40.0)
    assert np.isnan(values["mean_abs_month_03"])
    assert values["mean_abs_five_week"] == pytest.approx(40.0)
    assert values["mean_abs_four_week"] == pytest.approx(70 / 3)
    assert values["mean_abs_days_le13"] == pytest.approx(10.0)
    assert values["mean_abs_days_ge19"] == pytest.approx(30.0)
    assert values["kurtosis_standardized"] == pytest.approx(pearson_kurtosis(r))
    assert set(DAY_CELLS) == {"le13", "14", "15", "16", "17", "18", "ge19"}


def test_statistics_standardize_by_each_draws_location_and_scale():
    data = tiny_data()
    mu = np.full((3, 5), 2.0)
    sigma = np.full((3, 5), 4.0)
    values = predictive_statistics(np.tile(data.revision, (3, 1)), mu, sigma, data)
    expected = pearson_kurtosis((data.revision - 2.0) / 4.0)
    np.testing.assert_allclose(values["kurtosis_standardized"], expected)
    assert values["mean_abs_month_01"].shape == (3,)


def test_p_value_is_the_share_of_replicated_values_at_or_above_the_observed():
    observed = {"a": np.array([1.0, 2.0, 3.0, 4.0]), "b": 0.5}
    replicated = {"a": np.array([0.0, 2.0, 5.0, 5.0]), "b": np.array([0.1, 0.6])}
    assert predictive_p_values(observed, replicated) == {"a": 0.75, "b": 0.5}


def test_p_value_is_none_when_the_statistic_is_undefined():
    observed = {"a": float("nan")}
    replicated = {"a": np.array([1.0, 2.0])}
    assert predictive_p_values(observed, replicated) == {"a": None}


def test_reference_year_blocks_sum_the_months_of_each_year():
    data = tiny_data()
    pointwise = np.arange(2 * 3 * 5, dtype=float).reshape(2, 3, 5)
    blocked = sum_by_year(pointwise, data)
    assert blocked.shape == (2, 3, 2)
    np.testing.assert_allclose(blocked[..., 0], pointwise[..., :3].sum(axis=-1))
    np.testing.assert_allclose(blocked[..., 1], pointwise[..., 3:].sum(axis=-1))


def test_prior_predictive_record_compares_quantiles_with_the_targets():
    data = tiny_data()
    # Draw 0 holds small and draw 1 large revisions; log scales step by 0.1 a month.
    revision = np.array(
        [[1.0, -1.5, 2.0, 1.0, -1.0], [400.0, -500.0, 350.0, 2000.0, 1.0]]
    )
    log_sigma = np.tile(np.arange(5) * 0.1, (2, 1))
    record = prior_predictive_record(revision, log_sigma, data)
    quantiles = record["abs_revision_quantiles"]
    assert quantiles["0.01"] == pytest.approx(np.quantile(np.abs(revision), 0.01))
    assert record["share_abs_above_1000"] == pytest.approx(0.1)
    # 2013-01 to 2013-02 is the one lag-1 pair, so each draw gives one ratio, exp(0.1).
    ratios = record["scale_ratio_lag1_quantiles"]
    assert ratios["0.5"] == pytest.approx(np.exp(0.1))
    assert record["abs_covers_target"]
    assert record["ratio_within_target"]
    assert record["targets"] == {
        "abs_revision": [2.0, 300.0],
        "scale_ratio": [0.3, 3.0],
    }


def test_sensitivity_passes_when_beta_moves_less_than_half_a_data_se():
    summaries = {
        0.05: {"mean": -0.02, "sd": 0.03},
        0.1: {"mean": -0.030, "sd": 0.035},
        0.2: {"mean": -0.040, "sd": 0.036},
    }
    record = beta_sensitivity(summaries)
    assert record["shift_0.1_to_0.2"] == pytest.approx(0.010)
    assert record["passed"]
    summaries[0.2] = {"mean": -0.050, "sd": 0.036}
    assert not beta_sensitivity(summaries)["passed"]
    with pytest.raises(ValueError, match="0.1 and 0.2"):
        beta_sensitivity({0.05: summaries[0.05], 0.1: summaries[0.1]})
