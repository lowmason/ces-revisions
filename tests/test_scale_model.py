"""The scale model: its sites, priors, location and scale, and its pointwise log likelihood."""

from datetime import date

import jax
import numpy as np
import pytest
from numpyro import handlers
from scipy import stats

from ces_revisions.runs.inits import dispersed_init_params
from ces_revisions.scale_model.data import ScaleData
from ces_revisions.scale_model.model import (
    SAMPLED_SITES,
    Priors,
    location_scale,
    pointwise_log_likelihood,
    sample_prior,
    scale_model,
)


def small_data() -> ScaleData:
    """Five months across two years, one of them a lapse month."""
    return ScaleData(
        series=("NSA", "F_S"),
        reference_months=(
            date(2013, 1, 1),
            date(2013, 2, 1),
            date(2013, 9, 1),
            date(2014, 3, 1),
            date(2014, 12, 1),
        ),
        years=(2013, 2014),
        revision=np.array([-20.0, 30.0, -40.0, 50.0, -60.0]),
        collection_days=np.array([14.0, 16.0, 26.0, 13.0, 17.0]),
        window=np.array([-1.0, 1.0, 11.0, -2.0, 2.0]),
        year_window=np.array([0.5, -0.5]),
        five_week=np.array([1.0, 0.0, 1.0, 0.0, 0.0]),
        lapse=np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
        year=np.array([0, 0, 0, 1, 1]),
        month=np.array([0, 1, 8, 2, 11]),
    )


def fixed_params() -> dict:
    p = np.linspace(-5.5, 5.5, 12)
    c = np.linspace(-0.55, 0.55, 12)
    return {
        "m": 3.0,
        "p": p,
        "gamma": 2.0,
        "phi": -4.0,
        "a": np.log(40.0),
        "tau_y": 0.2,
        "z_y": np.array([1.0, -0.5]),
        "c": c,
        "beta": 0.03,
        "kappa": -0.05,
        "delta": 0.1,
        "lam": 0.4,
        "nu": 6.0,
    }


def test_location_and_log_scale_follow_the_recommendation():
    data, params = small_data(), fixed_params()
    mu, log_sigma = location_scale(params, data)
    expected_mu = (
        3.0 + params["p"][data.month] + 2.0 * data.window - 4.0 * data.five_week
    )
    expected_log_sigma = (
        np.log(40.0)
        + 0.2 * params["z_y"][data.year]
        + params["c"][data.month]
        + 0.03 * data.window
        - 0.05 * data.year_window[data.year]
        + 0.1 * data.five_week
        + 0.4 * data.lapse
    )
    np.testing.assert_allclose(mu, expected_mu, rtol=1e-12)
    np.testing.assert_allclose(log_sigma, expected_log_sigma, rtol=1e-12)


def test_pointwise_log_likelihood_is_the_student_t_density_of_each_month():
    data, params = small_data(), fixed_params()
    mu, log_sigma = location_scale(params, data)
    expected = stats.t.logpdf(
        data.revision, df=6.0, loc=np.asarray(mu), scale=np.exp(np.asarray(log_sigma))
    )
    np.testing.assert_allclose(
        pointwise_log_likelihood(params, data), expected, rtol=1e-10
    )


def test_model_log_likelihood_sums_the_pointwise_terms():
    data, params = small_data(), fixed_params()
    conditioned = handlers.substitute(scale_model, params)
    trace = handlers.trace(handlers.seed(conditioned, rng_seed=0)).get_trace(
        data, Priors(), data.revision
    )
    site = trace["r"]
    assert site["is_observed"]
    total = float(site["fn"].log_prob(site["value"]).sum())
    assert total == pytest.approx(float(pointwise_log_likelihood(params, data).sum()))


def test_scale_ratio_is_the_seven_day_contrast():
    data, params = small_data(), fixed_params()
    conditioned = handlers.substitute(scale_model, params)
    trace = handlers.trace(handlers.seed(conditioned, rng_seed=0)).get_trace(
        data, Priors(), data.revision
    )
    assert float(trace["scale_ratio"]["value"]) == pytest.approx(np.exp(7 * 0.03))


@pytest.mark.parametrize("beta_sd", [0.05, 0.1, 0.2])
def test_beta_prior_scale_is_a_setting(beta_sd):
    data = small_data()
    trace = handlers.trace(handlers.seed(scale_model, rng_seed=0)).get_trace(
        data, Priors(beta_sd=beta_sd), None
    )
    assert float(trace["beta"]["fn"].scale) == beta_sd


def test_prior_draws_hold_every_sampled_site_with_its_shape():
    data = small_data()
    draws = sample_prior(jax.random.PRNGKey(0), 200, data, Priors())
    assert set(draws) == set(SAMPLED_SITES)
    assert draws["z_y"].shape == (200, 2)
    assert draws["p"].shape == (200, 12)
    np.testing.assert_allclose(draws["c"].sum(axis=1), 0.0, atol=1e-9)
    np.testing.assert_allclose(draws["p"].sum(axis=1), 0.0, atol=1e-9)
    assert np.all(draws["tau_y"] > 0)
    assert np.all(draws["nu"] > 0)
    # Gamma(2, rate 0.1) has mean 20; HalfNormal(0.3) has mean 0.3 * sqrt(2 / pi).
    assert draws["nu"].mean() == pytest.approx(20.0, rel=0.2)
    assert draws["tau_y"].mean() == pytest.approx(0.3 * np.sqrt(2 / np.pi), rel=0.2)


def test_dispersed_starts_cover_every_site_at_finite_energy():
    data = small_data()
    draws = sample_prior(jax.random.PRNGKey(1), 500, data, Priors())
    init_params, record = dispersed_init_params(
        scale_model, (data, Priors(), data.revision), draws, 4, seed=7
    )
    assert set(init_params) == set(SAMPLED_SITES)
    assert np.all(np.isfinite(record["initial_potential_energy"]))
