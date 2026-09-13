"""Tests of the synthetic pilot for the Kalman engine."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro import handlers
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
