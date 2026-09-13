"""Tests of the synthetic pilot for the Kalman engine."""

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from synthetic_pilot import (
    BENCHMARK_ROW,
    LAPSE_STEP,
    LEVEL_REGIME_START,
    MARCH,
    NUM_STEPS,
    PRIOR_MEDIANS,
    TRUTH,
    pilot_model,
    pilot_ssm,
    simulate_panel,
)

from ces_revisions.kalman import kalman_filter

SEED = sum(map(ord, "ces-revisions-stage-1-synthetic-pilot"))
# Req 18 MCMC thresholds, with energy-BFMI and tree-depth saturation made numeric.
NUM_CHAINS = 4
NUM_DRAWS = 1000
RHAT_MAX = 1.01
ESS_MIN = 400
BFMI_MIN = 0.3
MAX_TREE_DEPTH = 10
SATURATED_SHARE_MAX = 0.01
RECOVERY_SDS = 4  # the truth must lie within this many posterior SDs of the mean


@pytest.fixture(scope="module")
def panel():
    return jnp.asarray(simulate_panel(TRUTH, seed=SEED))


def test_panel_has_time_varying_rows_nan_cells_and_an_irregular_annual_row(panel):
    y = np.asarray(panel)
    steps = np.arange(NUM_STEPS)
    # Closing cells away from the ragged edge and the lapse month.
    interior_closing = np.delete(y[:-2, :BENCHMARK_ROW], LAPSE_STEP, axis=0)
    observation_matrix = np.asarray(pilot_ssm(TRUTH).observation_matrix)

    assert np.array_equal(~np.isnan(y[:, BENCHMARK_ROW]), steps % 12 == MARCH)
    assert np.isnan(y[LAPSE_STEP]).all()
    assert np.isnan(y[-1, 1:BENCHMARK_ROW]).all()
    assert np.isnan(interior_closing).any()
    assert not np.array_equal(
        observation_matrix[LEVEL_REGIME_START - 1],
        observation_matrix[LEVEL_REGIME_START],
    )


def test_pilot_samples_only_static_scales_and_factors_the_marginal_likelihood(panel):
    seeded = handlers.seed(pilot_model, jax.random.PRNGKey(SEED))
    trace = handlers.trace(seeded).get_trace(panel)
    sampled = {
        name: site
        for name, site in trace.items()
        if site["type"] == "sample" and not site["is_observed"]
    }
    scales = {name: site["value"] for name, site in sampled.items()}

    assert set(sampled) == set(PRIOR_MEDIANS)
    assert all(np.shape(value) == () for value in scales.values())
    assert isinstance(trace["panel"]["fn"], dist.Unit)
    np.testing.assert_allclose(
        trace["panel"]["fn"].log_factor,
        kalman_filter(pilot_ssm(scales), panel).log_likelihood,
    )


@pytest.mark.slow
def test_pilot_fit_meets_mcmc_thresholds_and_recovers_the_truth(panel):
    mcmc = MCMC(
        NUTS(pilot_model, max_tree_depth=MAX_TREE_DEPTH),
        num_warmup=NUM_DRAWS,
        num_samples=NUM_DRAWS,
        num_chains=NUM_CHAINS,
        chain_method="parallel",
        progress_bar=False,
    )
    fields = ("diverging", "energy", "num_steps")
    mcmc.run(jax.random.PRNGKey(SEED), panel, extra_fields=fields)
    # Diagnose through ArviZ's DataTree so R-hat and ESS use ArviZ's defaults: on raw
    # arrays, tail ESS needs an explicit quantile and does not reproduce the default.
    idata = az.from_numpyro(mcmc)
    posterior, stats = idata["posterior"], idata["sample_stats"]
    rhat = az.rhat(idata)
    bulk_ess = az.ess(idata, method="bulk")
    tail_ess = az.ess(idata, method="tail")

    assert set(posterior.data_vars) == set(PRIOR_MEDIANS)  # no latent state is sampled
    assert int(stats["diverging"].sum()) == 0
    assert np.all(np.asarray(az.bfmi(idata)["energy"]) > BFMI_MIN)
    saturated = stats["n_steps"] >= 2**MAX_TREE_DEPTH - 1
    assert float(saturated.mean()) < SATURATED_SHARE_MAX
    for name in PRIOR_MEDIANS:
        draws = np.asarray(posterior[name])
        assert draws.shape == (NUM_CHAINS, NUM_DRAWS)
        assert float(rhat[name]) < RHAT_MAX, name
        assert float(bulk_ess[name]) > ESS_MIN, name
        assert float(tail_ess[name]) > ESS_MIN, name
        assert abs(draws.mean() - TRUTH[name]) <= RECOVERY_SDS * draws.std(), name


@pytest.mark.slow
def test_pilot_draws_are_identical_under_a_fixed_seed(panel):
    def draws():
        mcmc = MCMC(
            NUTS(pilot_model),
            num_warmup=50,
            num_samples=50,
            num_chains=2,
            chain_method="parallel",
            progress_bar=False,
        )
        mcmc.run(jax.random.PRNGKey(SEED), panel)
        return mcmc.get_samples()

    first, second = draws(), draws()
    for name in PRIOR_MEDIANS:
        np.testing.assert_array_equal(first[name], second[name], err_msg=name)
