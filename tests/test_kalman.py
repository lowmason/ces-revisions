"""The Kalman engine against the dense oracle, plus its gradient and input contract."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dense_reference import dense_reference, random_ssm, simulate, time_invariant

from ces_revisions.kalman import FilterResult, LinearGaussianSSM, kalman_filter

SEED = sum(map(ord, "ces-revisions-kalman-engine"))
NUM_STEPS, STATE_DIM, OBS_DIM = 15, 3, 4
MISSING_STEP = 5
ANNUAL_ROW = OBS_DIM - 1  # observed only at steps 2 and 14, like a March benchmark
TOLERANCE = {"rtol": 1e-8, "atol": 1e-10}


def _with_gaps(y):
    """Scattered missing cells, a missing step, a ragged edge, and an annual-only row."""
    y = y.copy()
    y[3, 1] = np.nan
    y[9, 0] = np.nan
    y[MISSING_STEP, :] = np.nan
    y[-1, :2] = np.nan
    y[np.arange(len(y)) % 12 != 2, ANNUAL_ROW] = np.nan
    return y


def _random_case(*, time_varying):
    rng = np.random.default_rng(SEED)
    ssm = random_ssm(rng, NUM_STEPS, STATE_DIM, OBS_DIM)
    if not time_varying:
        ssm = time_invariant(ssm)
        return ssm, simulate(ssm, rng)
    return ssm, _with_gaps(simulate(ssm, rng))


def _engine(ssm):
    return LinearGaussianSSM(
        **{name: jnp.asarray(value) for name, value in ssm.items()}
    )


@pytest.fixture(
    params=[False, True], ids=["time-invariant, fully observed", "time-varying, gaps"]
)
def case(request):
    return _random_case(time_varying=request.param)


@pytest.fixture
def gappy_case():
    return _random_case(time_varying=True)


def test_filter_matches_the_dense_reference(case):
    ssm, y = case
    filtered = kalman_filter(_engine(ssm), jnp.asarray(y))
    reference = dense_reference(ssm, y)
    for field in FilterResult._fields:
        np.testing.assert_allclose(
            getattr(filtered, field), reference[field], **TOLERANCE, err_msg=field
        )


def test_fully_missing_step_adds_nothing_and_leaves_the_state_unconditioned(gappy_case):
    ssm, y = gappy_case
    filtered = kalman_filter(_engine(ssm), jnp.asarray(y))
    assert filtered.step_log_likelihood[MISSING_STEP] == 0.0
    np.testing.assert_array_equal(
        filtered.filtered_mean[MISSING_STEP], filtered.predicted_mean[MISSING_STEP]
    )
    np.testing.assert_array_equal(
        filtered.filtered_cov[MISSING_STEP], filtered.predicted_cov[MISSING_STEP]
    )


def test_log_likelihood_gradient_matches_a_dense_finite_difference(gappy_case):
    ssm, y = gappy_case
    engine_ssm = _engine(ssm)

    def engine_log_likelihood(scale):
        scaled = engine_ssm._replace(
            transition_cov=engine_ssm.transition_cov * scale,
            observation_cov=engine_ssm.observation_cov * scale,
        )
        return kalman_filter(scaled, jnp.asarray(y)).log_likelihood

    def dense_log_likelihood(scale):
        scaled = {
            **ssm,
            "transition_cov": ssm["transition_cov"] * scale,
            "observation_cov": ssm["observation_cov"] * scale,
        }
        return dense_reference(scaled, y)["log_likelihood"]

    step = 1e-6
    finite_difference = (
        dense_log_likelihood(1 + step) - dense_log_likelihood(1 - step)
    ) / (2 * step)
    gradient = jax.grad(engine_log_likelihood)(1.0)
    assert np.isfinite(gradient)
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-5)


def test_rejects_arrays_whose_shapes_disagree_with_the_panel(gappy_case):
    ssm, y = gappy_case
    short = {**ssm, "observation_cov": ssm["observation_cov"][:-1]}
    with pytest.raises(ValueError, match="observation_cov"):
        kalman_filter(_engine(short), jnp.asarray(y))


def test_rejects_float32_inputs(gappy_case):
    ssm, y = gappy_case
    with pytest.raises(TypeError, match="float64"):
        kalman_filter(_engine(ssm), jnp.asarray(y, dtype=jnp.float32))
