"""Evidence for docs/decisions/engine.md: Dynamax's linear-Gaussian SSM on our inputs.

These tests pin observed Dynamax 1.0.2 behavior, not desired behavior. If one fails after a
Dynamax upgrade, the premise of the engine determination has changed: revisit the record.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from dense_reference import dense_reference, random_ssm, simulate
from dynamax.linear_gaussian_ssm.inference import lgssm_filter, make_lgssm_params

SEED = sum(map(ord, "ces-revisions-engine-determination"))
NUM_STEPS, STATE_DIM, OBS_DIM = 15, 3, 4
MISSING_PATTERNS = {
    "one missing cell": (3, 1),
    "fully missing step": (5, slice(None)),
    "irregular annual row": (np.arange(NUM_STEPS) % 12 != 2, OBS_DIM - 1),
}


def _to_dynamax(ssm):
    """Re-index a model for Dynamax.

    Dynamax puts its prior on the first state and its F_t maps z_t to z_{t+1}; ours puts the
    prior on x_{-1} and A_t maps x_{t-1} to x_t.
    """
    a, c, q = ssm["transition_matrix"], ssm["transition_offset"], ssm["transition_cov"]

    def shifted(array):  # Dynamax never uses the transition at its final step
        return jnp.asarray(np.concatenate([array[1:], array[-1:]]))

    return make_lgssm_params(
        initial_mean=jnp.asarray(a[0] @ ssm["initial_mean"] + c[0]),
        initial_cov=jnp.asarray(a[0] @ ssm["initial_cov"] @ a[0].T + q[0]),
        dynamics_weights=shifted(a),
        dynamics_cov=shifted(q),
        emissions_weights=jnp.asarray(ssm["observation_matrix"]),
        emissions_cov=jnp.asarray(ssm["observation_cov"]),
        dynamics_bias=shifted(c),
        emissions_bias=jnp.asarray(ssm["observation_offset"]),
    )


@pytest.fixture
def model_and_panel():
    rng = np.random.default_rng(SEED)
    ssm = random_ssm(rng, NUM_STEPS, STATE_DIM, OBS_DIM)
    return ssm, simulate(ssm, rng)


def test_dynamax_matches_the_dense_reference_on_time_varying_rows(model_and_panel):
    ssm, y = model_and_panel
    posterior = lgssm_filter(_to_dynamax(ssm), jnp.asarray(y))
    expected = dense_reference(ssm, y)["log_likelihood"]
    np.testing.assert_allclose(posterior.marginal_loglik, expected, rtol=1e-8)


@pytest.mark.parametrize("cells", MISSING_PATTERNS.values(), ids=list(MISSING_PATTERNS))
def test_dynamax_log_likelihood_is_nan_when_any_cell_is_missing(model_and_panel, cells):
    ssm, y = model_and_panel
    y = y.copy()
    y[cells] = np.nan
    posterior = lgssm_filter(_to_dynamax(ssm), jnp.asarray(y))
    assert np.isnan(posterior.marginal_loglik)
    assert np.isfinite(dense_reference(ssm, y)["log_likelihood"])  # the quantity exists


def test_dynamax_filter_has_no_step_terms_or_predicted_moments(model_and_panel):
    ssm, y = model_and_panel
    posterior = lgssm_filter(_to_dynamax(ssm), jnp.asarray(y))
    assert np.shape(posterior.marginal_loglik) == ()
    assert posterior.predicted_means is None
    assert posterior.predicted_covariances is None
