"""Tests of the pilot's simulator, revision statistics, and prior-predictive check."""

import math

import numpy as np
import numpyro.distributions as dist
import pytest
from dense_reference import joint_moments
from pilot_synthetic import TRUTH, reference_ssm, shell_panel, synthetic_panel

from ces_revisions.pilot.layout import LAYOUTS
from ces_revisions.pilot.model import (
    NOISE_RAW_SITES,
    SAMPLED_SITES,
    prior_distributions,
)
from ces_revisions.pilot.panel import ANCHOR_COLUMN, ANCHOR_ROUNDING_VARIANCE
from ces_revisions.pilot.simulate import (
    monthly_constraint,
    prior_predictive,
    revision_statistics,
    simulate_complete_panels,
    simulate_panel,
)

SEED = sum(map(ord, "ces-revisions-stage-6-pilot-simulate"))
LAYOUT_NAMES = sorted(LAYOUTS)
# The whitened moment check: NUM_PANELS independent complete panels of MOMENT_STEPS
# months, whitened by the dense oracle's mean and covariance of every observed cell,
# must look like independent standard normals. Bounds are in sampling SDs, so they are
# loose for a correct simulator and tight for a wrong one (see EVIDENCE in the spike
# report): a 10% error in one noise or news scale moves the worst variance by > 10 SDs.
NUM_PANELS = 40_000
MOMENT_STEPS = 24
MEAN_Z_MAX = 4.5  # on each whitened coordinate's mean, in units of 1/sqrt(N)
VARIANCE_Z_MAX = 5.0  # on each whitened coordinate's variance, in units of sqrt(2/N)
CORRELATION_Z_MAX = 5.5  # on each off-diagonal covariance, in units of 1/sqrt(N)
PRIOR_DRAWS = 50


@pytest.fixture(params=LAYOUT_NAMES)
def layout_name(request):
    return request.param


def test_masked_panel_is_the_complete_panel_with_the_panel_gaps(layout_name):
    panel = synthetic_panel(layout_name)
    layout = LAYOUTS[layout_name]
    complete = simulate_panel(TRUTH, panel, layout, SEED, complete=True)
    masked = simulate_panel(TRUTH, panel, layout, SEED)
    anchored = ~np.isnan(panel.y[:, ANCHOR_COLUMN])

    assert complete.shape == masked.shape == panel.y.shape
    assert not np.isnan(complete[:, :ANCHOR_COLUMN]).any()
    np.testing.assert_array_equal(~np.isnan(complete[:, ANCHOR_COLUMN]), anchored)
    np.testing.assert_array_equal(np.isnan(masked), np.isnan(panel.y))
    observed = ~np.isnan(panel.y)
    np.testing.assert_array_equal(masked[observed], complete[observed])
    np.testing.assert_array_equal(simulate_panel(TRUTH, panel, layout, SEED), masked)
    batch = simulate_complete_panels(TRUTH, panel, layout, SEED, 3)
    np.testing.assert_array_equal(
        simulate_complete_panels(TRUTH, panel, layout, SEED, 3), batch
    )
    np.testing.assert_array_equal(np.isnan(batch[0]), np.isnan(complete))
    assert not np.array_equal(batch[1], batch[0])
    assert not np.array_equal(simulate_panel(TRUTH, panel, layout, SEED + 1), masked)


def test_simulated_panels_have_the_dense_oracles_moments(layout_name):
    panel = shell_panel(MOMENT_STEPS)
    layout = LAYOUTS[layout_name]
    _, _, mean, cov, _ = joint_moments(reference_ssm(TRUTH, panel, layout_name))
    cells = ~np.isnan(panel.y.ravel())
    chol = np.linalg.cholesky(cov[np.ix_(cells, cells)])
    panels = simulate_complete_panels(TRUTH, panel, layout, SEED, NUM_PANELS)
    draws = panels.reshape(NUM_PANELS, -1)[:, cells]
    white = np.linalg.solve(chol, (draws - mean[cells]).T).T
    white_cov = np.cov(white, rowvar=False)
    off_diagonal = white_cov - np.diag(np.diag(white_cov))

    assert panel.pre_2003.any() and panel.crisis.any()
    assert (~np.isnan(panel.y[:, ANCHOR_COLUMN])).sum() == 2
    assert np.abs(white.mean(axis=0)).max() * math.sqrt(NUM_PANELS) < MEAN_Z_MAX
    variance_z = np.abs(np.diag(white_cov) - 1.0) / math.sqrt(2.0 / NUM_PANELS)
    assert variance_z.max() < VARIANCE_Z_MAX
    assert np.abs(off_diagonal).max() * math.sqrt(NUM_PANELS) < CORRELATION_Z_MAX


def test_anchor_noise_is_a_standard_normal_scaled_by_the_anchor_sd():
    # The whitened check cannot see the anchor noise under the level's prior SD, so
    # isolate it: the simulator's draws do not depend on the parameters, so changing
    # sigma_anchor alone moves only the anchor cells, by sd * z with a shared z.
    panel = shell_panel(MOMENT_STEPS)
    layout = LAYOUTS["dummy"]
    anchored = ~np.isnan(panel.y[:, ANCHOR_COLUMN])
    scales = (40.0, 80.0, 160.0)
    runs = [
        simulate_complete_panels(
            TRUTH | {"sigma_anchor": scale}, panel, layout, SEED, NUM_PANELS
        )
        for scale in scales
    ]
    sd = [math.sqrt(scale**2 + ANCHOR_ROUNDING_VARIANCE) for scale in scales]
    anchors = [run[:, anchored, ANCHOR_COLUMN] for run in runs]
    z = (anchors[1] - anchors[0]) / (sd[1] - sd[0])

    for run in runs[1:]:
        np.testing.assert_array_equal(
            run[:, :, :ANCHOR_COLUMN], runs[0][:, :, :ANCHOR_COLUMN]
        )
    np.testing.assert_allclose(anchors[2] - anchors[0], z * (sd[2] - sd[0]), atol=1e-8)
    size = z.size
    assert abs(z.mean()) * math.sqrt(size) < MEAN_Z_MAX
    assert abs(z.var() - 1.0) / math.sqrt(2.0 / size) < VARIANCE_Z_MAX


def test_revision_statistics_use_rows_with_both_f_and_t():
    y = np.full((6, 4), np.nan)
    y[:, 0] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    y[:, 2] = [-290.0, 25.0, 30.0, np.nan, 90.0, 360.0]
    y[1, 0] = np.nan  # F missing: dropped
    revision = np.array([-300.0, 0.0, 40.0, 300.0])

    statistics = revision_statistics(y)

    assert statistics == pytest.approx(
        {
            "median_abs": 170.0,
            "share_above_250": 0.5,
            "mean": revision.mean(),
            "sd": revision.std(ddof=1),
            "count": 4,
        }
    )


def test_prior_predictive_reports_one_complete_panel_per_draw(layout_name):
    panel = synthetic_panel(layout_name)
    layout = LAYOUTS[layout_name]
    result = prior_predictive(panel, layout, PRIOR_DRAWS, SEED)
    again = prior_predictive(panel, layout, PRIOR_DRAWS, SEED)
    statistics = result["statistics"]

    assert (result["layout"], result["seed"], result["doubled"]) == (
        layout_name,
        SEED,
        False,
    )
    assert tuple(result["params"]) == SAMPLED_SITES
    assert set(statistics) == {"median_abs", "share_above_250", "mean", "sd", "count"}
    assert all(values.shape == (PRIOR_DRAWS,) for values in statistics.values())
    assert (statistics["count"] == panel.num_steps).all()  # every closing cell filled
    for name, values in statistics.items():
        np.testing.assert_array_equal(values, again["statistics"][name], err_msg=name)
    constraint = monthly_constraint(result)
    assert set(constraint) == {
        "share_median_in_range",
        "pooled_share_above_250",
        "passed",
    }
    assert 0.0 <= constraint["share_median_in_range"] <= 1.0


def test_doubled_prior_doubles_the_lognormal_and_noise_log_scales():
    panel = synthetic_panel()
    layout = LAYOUTS["dummy"]
    priors = prior_distributions(layout)
    base = prior_predictive(panel, layout, PRIOR_DRAWS, SEED)["params"]
    doubled = prior_predictive(panel, layout, PRIOR_DRAWS, SEED, doubled=True)["params"]
    lognormal = [
        name for name, prior in priors.items() if isinstance(prior, dist.LogNormal)
    ]

    assert lognormal == ["sigma_news_s", "sigma_news_t", "sigma_news_annual"]
    assert NOISE_RAW_SITES == ("noise_f_raw", "noise_s_raw", "noise_t_raw")
    for name in SAMPLED_SITES:
        if name in lognormal:
            loc = float(priors[name].loc)
            np.testing.assert_allclose(
                np.log(doubled[name]) - loc,
                2.0 * (np.log(base[name]) - loc),
                rtol=1e-10,
            )
        elif name in NOISE_RAW_SITES:
            # Each noise SD's log-scale deviation from its center doubles with its raw.
            np.testing.assert_allclose(doubled[name], 2.0 * base[name], rtol=1e-12)
        else:
            np.testing.assert_array_equal(doubled[name], base[name], err_msg=name)


def _result(median_abs, share_above_250, count):
    size = len(median_abs)
    return {
        "statistics": {
            "median_abs": np.asarray(median_abs, dtype=float),
            "share_above_250": np.asarray(share_above_250, dtype=float),
            "count": np.full(size, count),
        }
    }


def test_monthly_constraint_applies_both_req_15_bounds():
    inside = [20.0, 50.0, 100.0, 60.0, 30.0]
    passing = monthly_constraint(_result(inside, [0.0, 0.01, 0.02, 0.0, 0.02], 100))
    narrow = monthly_constraint(
        _result([19.9, 50.0, 100.1, 60.0, 30.0], [0.01] * 5, 100)
    )
    too_quiet = monthly_constraint(_result(inside, [0.0] * 5, 100))
    too_loud = monthly_constraint(_result(inside, [0.06] * 5, 100))

    assert passing == {
        "share_median_in_range": 1.0,
        "pooled_share_above_250": pytest.approx(0.01),
        "passed": True,
    }
    assert narrow["share_median_in_range"] == pytest.approx(0.6)
    assert not narrow["passed"]
    assert too_quiet["pooled_share_above_250"] == 0.0 and not too_quiet["passed"]
    assert too_loud["pooled_share_above_250"] == pytest.approx(0.06)
    assert not too_loud["passed"]
