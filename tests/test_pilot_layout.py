"""Tests of the pilot's state layouts: transitions, shock loadings, observation rows."""

import math

import numpy as np
import pytest
from pilot_synthetic import STATE_NAMES

from ces_revisions.pilot.layout import (
    LAYOUTS,
    SEASONAL_PRIOR_SCALE,
    observation_matrix,
    seasonal_selector,
    seasonal_variance_factor,
    shock_loading,
    shock_names,
    transition_matrix,
)
from ces_revisions.pilot.panel import COLUMNS

SEED = sum(map(ord, "ces-revisions-stage-6-pilot-layout"))
NEWS = ("news_s", "news_t", "news_post")


@pytest.fixture(params=sorted(LAYOUTS))
def layout(request):
    return LAYOUTS[request.param]


def _at(layout, name):
    return layout.state_names.index(name)


def _seasonal_block(layout):
    first = len(("level", "slope", "lagged_value"))
    return slice(first, first + layout.seasonal_dim)


def test_layouts_have_the_contract_state_order_and_sizes():
    sizes = {
        name: (layout.state_dim, layout.seasonal_dim, layout.harmonics)
        for name, layout in LAYOUTS.items()
    }

    assert sizes == {"dummy": (17, 11, None), "trig": (17, 11, 6), "trig4": (14, 8, 4)}
    for name, layout in LAYOUTS.items():
        assert layout.name == name
        assert layout.state_names == STATE_NAMES[name]


def test_seasonal_paths_sum_to_zero_over_any_year_and_repeat_every_twelve_months(
    layout,
):
    transition = transition_matrix(layout)
    selector = seasonal_selector(layout)
    state = np.random.default_rng(SEED).normal(0.0, 1_000.0, layout.state_dim)
    gammas = []
    for _ in range(48):
        state = transition @ state
        gammas.append(selector @ state)
    gammas = np.array(gammas)
    yearly = np.convolve(gammas, np.ones(12), mode="valid")

    np.testing.assert_allclose(yearly, 0.0, atol=1e-8)
    np.testing.assert_allclose(gammas[12:], gammas[:-12], atol=1e-8)
    assert np.ptp(gammas) > 100.0  # a real seasonal pattern, not a vanishing one


def test_dummy_seasonal_is_minus_the_sum_of_the_last_eleven_months():
    layout = LAYOUTS["dummy"]
    block = transition_matrix(layout)[_seasonal_block(layout), _seasonal_block(layout)]
    expected = np.zeros((11, 11))
    expected[0, :] = -1.0
    expected[np.arange(1, 11), np.arange(10)] = 1.0

    np.testing.assert_array_equal(block, expected)


@pytest.mark.parametrize("name", ["trig", "trig4"])
def test_trigonometric_harmonics_rotate_with_period_twelve(name):
    layout = LAYOUTS[name]
    block = transition_matrix(layout)[_seasonal_block(layout), _seasonal_block(layout)]
    powers = [np.linalg.matrix_power(block, k) for k in range(1, 13)]

    np.testing.assert_allclose(powers[-1], np.eye(layout.seasonal_dim), atol=1e-12)
    assert all(
        not np.allclose(power, np.eye(layout.seasonal_dim)) for power in powers[:-1]
    )
    for j in range(1, min(layout.harmonics, 5) + 1):
        pair = block[2 * j - 2 : 2 * j, 2 * j - 2 : 2 * j]
        angle = 2 * math.pi * j / 12
        rotation = [
            [math.cos(angle), math.sin(angle)],
            [-math.sin(angle), math.cos(angle)],
        ]
        np.testing.assert_allclose(pair, rotation, rtol=0, atol=1e-15)
    if name == "trig":
        assert block[-1, -1] == -1.0
        assert np.count_nonzero(block[-1]) == np.count_nonzero(block[:, -1]) == 1


def test_selector_picks_gamma_from_the_seasonal_block(layout):
    picked = {
        name
        for name, weight in zip(
            layout.state_names, seasonal_selector(layout), strict=True
        )
        if weight
    }
    expected = (
        {"seasonal_0"}
        if layout.name == "dummy"
        else {name for name in layout.seasonal_names if not name.endswith("_sin")}
    )

    assert picked == expected
    assert set(seasonal_selector(layout)) == {0.0, 1.0}
    if layout.name == "trig":
        assert "harmonic_6" in picked


def test_transition_carries_the_trend_and_last_months_value(layout):
    transition = transition_matrix(layout)
    level, slope, lagged = (
        _at(layout, name) for name in ("level", "slope", "lagged_value")
    )
    level_row = np.zeros(layout.state_dim)
    level_row[[level, slope]] = 1.0
    lagged_row = seasonal_selector(layout).copy()
    lagged_row[level] = 1.0
    news = [_at(layout, name) for name in NEWS]

    np.testing.assert_array_equal(transition[level], level_row)
    np.testing.assert_array_equal(transition[slope], np.eye(layout.state_dim)[slope])
    np.testing.assert_array_equal(transition[lagged], lagged_row)
    assert not transition[news].any()  # news is white noise
    assert not transition[:, [lagged, *news]].any()  # nothing reads them a month later


def test_shock_loading_puts_each_news_shock_in_its_state_and_the_level(layout):
    loading = shock_loading(layout)
    names = shock_names(layout)
    seasonal_shocks = 1 if layout.name == "dummy" else layout.seasonal_dim
    loaded_by = {
        state: {names[k] for k in np.flatnonzero(loading[_at(layout, state)])}
        for state in layout.state_names
    }

    assert loading.shape == (layout.state_dim, 5 + seasonal_shocks)
    assert set(loading.ravel()) == {0.0, 1.0}
    assert loaded_by["level"] == {"eta_level", "eps_s", "eps_t", "eps_post"}
    assert loaded_by["slope"] == {"eta_slope"}
    assert loaded_by["lagged_value"] == set()
    assert loaded_by["news_s"] == {"eps_s"}
    assert loaded_by["news_t"] == {"eps_t"}
    assert loaded_by["news_post"] == {"eps_post"}
    seasonal = loading[_seasonal_block(layout), 2 : 2 + seasonal_shocks]
    if layout.name == "dummy":
        np.testing.assert_array_equal(seasonal, np.eye(11)[:, :1])  # slot 0 only
    else:
        np.testing.assert_array_equal(seasonal, np.eye(layout.seasonal_dim))


def test_observation_rows_are_the_change_less_the_news_still_to_come(layout):
    state = np.random.default_rng(SEED).normal(0.0, 100.0, layout.state_dim)
    value = state[_at(layout, "level")] + seasonal_selector(layout) @ state
    change = value - state[_at(layout, "lagged_value")]
    news_s, news_t, news_post = (state[_at(layout, name)] for name in NEWS)
    expected = {
        "F": change - news_s - news_t - news_post,
        "S": change - news_t - news_post,
        "T": change - news_post,
        "anchor": value,
    }

    np.testing.assert_allclose(
        observation_matrix(layout) @ state,
        [expected[column] for column in COLUMNS],
        rtol=1e-14,
    )


def _year_over_year_variance(layout):
    """Var(gamma_t - gamma_{t-12}) per unit shock variance, by a covariance recursion.

    The seasonal state at t - 12 is held fixed (zero covariance) and twelve months of
    unit shocks are propagated; C^12 = I makes the fixed part cancel in the difference.
    """
    block = transition_matrix(layout)[_seasonal_block(layout), _seasonal_block(layout)]
    loading = shock_loading(layout)[_seasonal_block(layout)]
    covariance = np.zeros_like(block)
    for _ in range(12):
        covariance = block @ covariance @ block.T + loading @ loading.T
    selector = seasonal_selector(layout)[_seasonal_block(layout)]
    return selector @ covariance @ selector


def test_seasonal_prior_scales_match_the_dummy_year_over_year_variance():
    factors = {
        name: seasonal_variance_factor(layout) for name, layout in LAYOUTS.items()
    }
    recursion = {
        name: _year_over_year_variance(layout) for name, layout in LAYOUTS.items()
    }

    assert factors == pytest.approx(
        {"dummy": 2.0, "trig": 72.0, "trig4": 48.0}, rel=1e-12
    )
    assert recursion == pytest.approx(factors, rel=1e-12)
    assert SEASONAL_PRIOR_SCALE["dummy"] == 50.0
    assert SEASONAL_PRIOR_SCALE["trig"] == pytest.approx(50.0 / 6.0, rel=1e-14)
    assert SEASONAL_PRIOR_SCALE["trig4"] == pytest.approx(
        50.0 / math.sqrt(24.0), rel=1e-14
    )
