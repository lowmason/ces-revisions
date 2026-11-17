"""The scale model's fit: the completed DataTree, its log prior, LOO, and interval summaries."""

import numpy as np
import pytest
from numpyro import distributions as dist
from test_scale_model import fixed_params, small_data

from ces_revisions.runs.fit import RunSettings
from ces_revisions.scale_model.fit import (
    SERIES_LABEL,
    fit_series,
    interval_summary,
    log_prior_draws,
    series_label,
)
from ces_revisions.scale_model.model import MONTH_SCALE_SD, SAMPLED_SITES, Priors


def by_chain(params: dict, chains: int = 2, draws: int = 3) -> dict:
    return {
        name: np.broadcast_to(
            np.asarray(value), (chains, draws, *np.shape(value))
        ).copy()
        for name, value in params.items()
    }


def test_log_prior_is_each_sites_prior_density_at_each_draw():
    params = {**fixed_params(), "nu": 6.0}
    values = log_prior_draws(by_chain(params), small_data(), Priors(beta_sd=0.2))
    assert set(values) == set(SAMPLED_SITES)
    assert values["beta"].shape == (2, 3)
    assert values["z_y"].shape == (2, 3, 2)
    assert values["p"].shape == (2, 3)
    expected_beta = float(dist.Normal(0.0, 0.2).log_prob(0.03))
    np.testing.assert_allclose(values["beta"], expected_beta)
    expected_nu = float(dist.Gamma(2.0, 0.1).log_prob(6.0))
    np.testing.assert_allclose(values["nu"], expected_nu)
    expected_c = float(
        dist.ZeroSumNormal(MONTH_SCALE_SD, event_shape=(12,)).log_prob(params["c"])
    )
    np.testing.assert_allclose(values["c"], expected_c)


def test_interval_summary_gives_equal_tailed_intervals():
    draws = np.arange(1, 1001, dtype=float)
    summary = interval_summary(draws)
    assert summary["mean"] == pytest.approx(500.5)
    assert summary["eti90"] == pytest.approx(list(np.quantile(draws, [0.05, 0.95])))
    assert summary["eti95"] == pytest.approx(list(np.quantile(draws, [0.025, 0.975])))


def test_series_labels_name_each_series_once():
    assert series_label(("NSA", "F_S")) == "nsa-f_s"
    assert len(set(SERIES_LABEL.values())) == 4


@pytest.mark.slow
def test_fit_completes_the_tree_for_checks_loo_and_sensitivity():
    data = small_data()
    fit = fit_series(
        data,
        Priors(),
        RunSettings(seed=11, num_warmup=60, num_samples=40, target_accept_prob=0.95),
    )
    tree = fit.result.idata
    groups = {name.strip("/") for name in tree.groups}
    assert {
        "posterior",
        "sample_stats",
        "observed_data",
        "posterior_predictive",
        "log_likelihood",
        "log_prior",
    } <= groups
    assert list(tree["log_likelihood"].data_vars) == ["r"]
    for group in ("observed_data", "posterior_predictive", "log_likelihood"):
        months = tree[group]["reference_month"].values
        assert months[0] == np.datetime64("2013-01-01")
    assert tree["posterior"]["c"].dims == ("chain", "draw", "calendar_month")
    assert tree["posterior"]["u"].dims == ("chain", "draw", "year")
    assert list(tree["posterior"]["year"].values) == [2013, 2014]
    assert tree["posterior_predictive"]["r"].shape == (4, 40, 5)
    # The log likelihood is the pointwise Student-t density at each draw.
    assert np.all(np.isfinite(tree["log_likelihood"]["r"].values))
    assert set(fit.loo) == {"month", "reference_year"}
    assert fit.loo["reference_year"]["num_blocks"] == 2
