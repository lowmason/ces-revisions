"""The April 2020 posterior check, against the dense oracle on a short crisis window."""

import dataclasses

import numpy as np
import pytest
from dense_reference import dense_reference
from pilot_synthetic import STATE_NAMES, TRUTH, crisis_panel, reference_ssm

from ces_revisions.pilot import checks
from ces_revisions.pilot.checks import (
    APRIL_2020,
    INTERVAL_Z,
    april_2020,
    band_check,
    change_row,
    news_row,
)
from ces_revisions.pilot.layout import LAYOUTS, observation_matrix
from ces_revisions.pilot.model import SAMPLED_SITES
from ces_revisions.pilot.panel import COLUMNS

SEED = sum(map(ord, "ces-revisions-stage-6-april-2020-check"))
NUM_CHAINS, NUM_DRAWS = 2, 5
NUM_KEPT = 4
# Positive sites are perturbed on the log scale, the rest additively, around TRUTH.
LOG_SITES = (
    "sigma_level",
    "sigma_slope",
    "sigma_seasonal",
    "sigma_news_s",
    "sigma_news_t",
    "sigma_news_annual",
    "sigma_anchor",
)


def fake_posterior() -> dict[str, np.ndarray]:
    """(chain, draw) arrays near TRUTH, distinct at every draw."""
    rng = np.random.default_rng(SEED)
    shape = (NUM_CHAINS, NUM_DRAWS)
    return {
        name: TRUTH[name] * np.exp(0.1 * rng.standard_normal(shape))
        if name in LOG_SITES
        else TRUTH[name] + 0.05 * rng.standard_normal(shape)
        for name in SAMPLED_SITES
    }


def oracle_rows(layout_name: str) -> np.ndarray:
    """dx and the news sum, written from the state names alone."""
    names = STATE_NAMES[layout_name]
    gamma = (
        ("seasonal_0",)
        if layout_name == "dummy"
        else tuple(
            name for name in names if name.endswith("_cos") or name == "harmonic_6"
        )
    )
    dx = np.zeros(len(names))
    dx[names.index("level")] = 1.0
    dx[[names.index(name) for name in gamma]] = 1.0
    dx[names.index("lagged_value")] = -1.0
    news = np.zeros(len(names))
    news[[names.index(name) for name in ("news_s", "news_t", "news_post")]] = 1.0
    return np.stack([dx, news])


@pytest.mark.parametrize("layout_name", sorted(LAYOUTS))
def test_change_and_news_rows(layout_name):
    layout = LAYOUTS[layout_name]
    expected = oracle_rows(layout_name)
    np.testing.assert_array_equal(change_row(layout), expected[0])
    np.testing.assert_array_equal(news_row(layout), expected[1])
    # The T print is the change less the post-T news; F is the change less all news.
    rows = observation_matrix(layout)
    news_post = np.eye(layout.state_dim)[layout.state_names.index("news_post")]
    np.testing.assert_array_equal(rows[COLUMNS.index("T")] + news_post, expected[0])
    np.testing.assert_array_equal(
        rows[COLUMNS.index("F")] + news_row(layout), expected[0]
    )


@pytest.mark.parametrize("layout_name", ["dummy", "trig4"])
def test_april_2020_matches_the_dense_smoother(layout_name):
    layout = LAYOUTS[layout_name]
    panel = crisis_panel(layout_name)
    samples = fake_posterior()
    result = april_2020(samples, panel, layout, num_draws=NUM_KEPT, seed=SEED)

    # Evenly spaced draws of the pooled (chain, draw) array: positions j * 10 // 4.
    kept = [0, 2, 5, 7]
    assert result["draw_indices"] == kept
    assert result["num_draws"] == NUM_KEPT
    assert result["non_finite_draws"] == 0
    step = panel.reference_months.index(APRIL_2020)
    rows = oracle_rows(layout_name)
    means, variances = [], []
    for index in kept:
        params = {
            name: float(values.reshape(-1)[index]) for name, values in samples.items()
        }
        smoothed = dense_reference(reference_ssm(params, panel, layout_name), panel.y)
        means.append(rows @ smoothed["smoothed_mean"][step])
        variances.append(np.diag(rows @ smoothed["smoothed_cov"][step] @ rows.T))
    means, variances = np.array(means), np.array(variances)
    for k, name in enumerate(("dx", "news_sum")):
        between = means[:, k].var()
        np.testing.assert_allclose(result[name]["mean"], means[:, k].mean(), rtol=1e-8)
        np.testing.assert_allclose(
            result[name]["sd"], np.sqrt(variances[:, k].mean() + between), rtol=1e-8
        )
        np.testing.assert_allclose(
            result[name]["sd_between_draws"], np.sqrt(between), rtol=1e-6
        )

    observed = panel.y[step, :3]
    assert result["observed"] == dict(
        zip(("F", "S", "T"), observed.tolist(), strict=True)
    )
    assert result["ratio"] == pytest.approx(
        result["dx"]["mean"] / observed[2], rel=1e-15
    )
    assert result["ratio_range"] == [0.95, 1.05]
    half_width = INTERVAL_Z * result["dx"]["sd"]
    interval = [result["dx"]["mean"] - half_width, result["dx"]["mean"] + half_width]
    assert result["interval_90"] == pytest.approx(interval, rel=1e-15)
    band = sorted([0.95 * observed[2], 1.05 * observed[2]])
    assert result["band"] == pytest.approx(band, rel=1e-15)
    assert result["passed"] == (band[0] <= interval[0] and interval[1] <= band[1])


THIRD = -20_000.0  # a third print at April 2020's scale; the band is [-21000, -19000]


@pytest.mark.parametrize(
    ("mean", "sd", "passed"),
    [
        (THIRD, 100.0, True),  # interval -20164.5 to -19835.5
        (-19_200.0, 100.0, True),  # interval -19364.5 to -19035.5
        (-19_100.0, 100.0, False),  # interval upper end -18935.5 leaves the band
        (THIRD, 5_600.0, False),  # the mean passes the ratio, the interval does not
        (-18_000.0, 0.0, False),  # a point mass outside the band
    ],
)
def test_band_check_needs_the_whole_interval_inside_the_band(mean, sd, passed):
    result = band_check(mean, sd, THIRD)
    assert result["ratio"] == pytest.approx(mean / THIRD, rel=1e-15)
    assert result["band"] == [-21_000.0, -19_000.0]
    assert result["interval_90"] == pytest.approx(
        [mean - INTERVAL_Z * sd, mean + INTERVAL_Z * sd], rel=1e-15
    )
    assert result["passed"] is passed


def test_april_2020_without_a_third_print_has_no_ratio():
    layout = LAYOUTS["dummy"]
    panel = crisis_panel("dummy")
    step = panel.reference_months.index(APRIL_2020)
    y = panel.y.copy()
    y[step, COLUMNS.index("T")] = np.nan
    result = april_2020(
        fake_posterior(),
        dataclasses.replace(panel, y=y),
        layout,
        num_draws=2,
        seed=SEED,
    )
    assert result["observed"]["T"] is None
    assert result["ratio"] is None
    assert result["passed"] is False


def test_a_draw_the_filter_cannot_evaluate_fails_the_check(monkeypatch):
    layout = LAYOUTS["dummy"]
    panel = crisis_panel("dummy")
    passing = {"ratio": 1.0, "interval_90": [0.0, 0.0], "band": [0.0, 0.0]}
    monkeypatch.setattr(
        checks, "band_check", lambda mean, sd, third: passing | {"passed": True}
    )
    samples = fake_posterior()
    result = april_2020(samples, panel, layout, num_draws=NUM_KEPT, seed=SEED)
    assert result["passed"] is True
    samples["sigma_level"][0, 0] = np.nan  # pooled draw 0, which the subsample keeps
    result = april_2020(samples, panel, layout, num_draws=NUM_KEPT, seed=SEED)
    assert result["non_finite_draws"] == 1
    assert result["passed"] is False
