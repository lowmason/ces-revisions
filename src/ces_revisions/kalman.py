"""Marginalized Kalman engine: the linear-Gaussian state-space likelihood for NumPyro.

The state evolves as ``x_t = A_t x_{t-1} + c_t + w_t`` with ``w_t ~ N(0, Q_t)``, starting
from ``x_{-1} ~ N(m, P)``, and is observed as ``y_t = Z_t x_t + d_t + v_t`` with
``v_t ~ N(0, R_t)``. Every time-indexed array describes step ``t``: the transition into
``x_t`` and the observation of ``x_t``, so a time-varying process scale sits at the step it
shocks. A NaN in ``y`` marks a missing cell; an irregular row, such as an annual benchmark,
is a row whose other cells are NaN.

The filter's log likelihood integrates the states out, so a NumPyro model that adds it as a
factor never samples a state. All arrays must be float64 (call ``numpyro.enable_x64()``
before the first JAX operation), and every model array must be finite except
``observation_cov`` in a missing cell's row or column: those entries reach only that step's
``one_step_cov``, never the log likelihood, the filtered or smoothed moments, or gradients
with respect to the other arrays.
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpyro
from jax.scipy.linalg import cho_solve

# A Python float, not a JAX constant: a module-level JAX array would run a JAX
# operation at import time, before the caller could enable float64.
_LOG_2PI = math.log(2 * math.pi)


class LinearGaussianSSM(NamedTuple):
    """A time-varying linear-Gaussian state-space model: T steps, n states, p cells."""

    initial_mean: jax.Array  # (n,) mean of x_{-1}
    initial_cov: jax.Array  # (n, n)
    transition_matrix: jax.Array  # (T, n, n) A_t
    transition_offset: jax.Array  # (T, n) c_t
    transition_cov: jax.Array  # (T, n, n) Q_t
    observation_matrix: jax.Array  # (T, p, n) Z_t
    observation_offset: jax.Array  # (T, p) d_t
    observation_cov: jax.Array  # (T, p, p) R_t


class FilterResult(NamedTuple):
    """Kalman filter output, conditioning on the observed (non-NaN) cells only."""

    log_likelihood: jax.Array  # () log p(y)
    step_log_likelihood: jax.Array  # (T,) log p(y_t | y_<t); 0 if none observed
    predicted_mean: jax.Array  # (T, n) E[x_t | y_<t]
    predicted_cov: jax.Array  # (T, n, n)
    filtered_mean: jax.Array  # (T, n) E[x_t | y_<=t]
    filtered_cov: jax.Array  # (T, n, n)
    one_step_mean: jax.Array  # (T, p) E[y_t | y_<t], missing cells included
    one_step_cov: jax.Array  # (T, p, p) Cov[y_t | y_<t]


class SmootherResult(NamedTuple):
    """Rauch–Tung–Striebel smoother output."""

    smoothed_mean: jax.Array  # (T, n) E[x_t | y]
    smoothed_cov: jax.Array  # (T, n, n)


def kalman_filter(ssm: LinearGaussianSSM, y: jax.Array) -> FilterResult:
    """Filter the (T, p) panel y through ssm, conditioning only on its non-NaN cells."""
    ssm = LinearGaussianSSM(*(jnp.asarray(array) for array in ssm))
    y = jnp.asarray(y)
    _validate(ssm, y)
    observed = ~jnp.isnan(y)
    state_eye = jnp.eye(ssm.initial_mean.shape[0])

    def step(carry, inputs):
        mean, cov = carry
        a, c, q, z, d, r, y_t, observed_t = inputs
        predicted_mean = a @ mean + c
        predicted_cov = _symmetrize(a @ cov @ a.T + q)
        one_step_mean = z @ predicted_mean + d
        one_step_cov = _symmetrize(z @ predicted_cov @ z.T + r)

        # Condition on the observed cells only. A missing cell gets a zero emission row,
        # a zero innovation, and a unit diagonal, so the innovation covariance is block
        # diagonal in (observed block, identity). Its log-determinant and quadratic form
        # are exactly the observed block's, and the 2*pi term counts only observed cells.
        both_observed = observed_t[:, None] & observed_t[None, :]
        missing_diagonal = jnp.diag(jnp.where(observed_t, 0.0, 1.0))
        innovation_cov = jnp.where(both_observed, one_step_cov, 0.0) + missing_diagonal
        innovation = jnp.where(observed_t, y_t - one_step_mean, 0.0)
        observed_z = jnp.where(observed_t[:, None], z, 0.0)
        chol = jnp.linalg.cholesky(innovation_cov)
        gain = cho_solve((chol, True), observed_z @ predicted_cov).T
        filtered_mean = predicted_mean + gain @ innovation
        # Joseph form keeps the filtered covariance positive semidefinite. Its noise
        # term uses only the observed block of R_t, so no entry in a missing cell's row
        # or column can reach the filtered covariance, even a non-finite one.
        residual_map = state_eye - gain @ observed_z
        observed_r = jnp.where(both_observed, r, 0.0)
        filtered_cov = _symmetrize(
            residual_map @ predicted_cov @ residual_map.T + gain @ observed_r @ gain.T
        )

        num_observed = observed_t.sum()
        log_det = 2.0 * jnp.log(jnp.diag(chol)).sum()
        mahalanobis = innovation @ cho_solve((chol, True), innovation)
        step_log_likelihood = -0.5 * (num_observed * _LOG_2PI + log_det + mahalanobis)
        outputs = (
            step_log_likelihood,
            predicted_mean,
            predicted_cov,
            filtered_mean,
            filtered_cov,
            one_step_mean,
            one_step_cov,
        )
        return (filtered_mean, filtered_cov), outputs

    # Zero-fill missing cells so that no intermediate holds NaN. Masking the innovation
    # already keeps the likelihood and gradients finite, so this is defense in depth.
    y_filled = jnp.where(observed, y, 0.0)
    inputs = (
        ssm.transition_matrix,
        ssm.transition_offset,
        ssm.transition_cov,
        ssm.observation_matrix,
        ssm.observation_offset,
        ssm.observation_cov,
        y_filled,
        observed,
    )
    _, outputs = jax.lax.scan(step, (ssm.initial_mean, ssm.initial_cov), inputs)
    return FilterResult(outputs[0].sum(), *outputs)


def kalman_smoother(ssm: LinearGaussianSSM, filtered: FilterResult) -> SmootherResult:
    """Smooth a filter pass of the same model backward in time (Rauch–Tung–Striebel).

    Every predicted covariance after the first step must be positive definite.
    """
    transition_matrix = jnp.asarray(ssm.transition_matrix)

    def step(carry, inputs):
        next_smoothed_mean, next_smoothed_cov = carry
        mean, cov, next_a, next_predicted_mean, next_predicted_cov = inputs
        chol = jnp.linalg.cholesky(next_predicted_cov)
        gain = cho_solve((chol, True), next_a @ cov).T
        smoothed_mean = mean + gain @ (next_smoothed_mean - next_predicted_mean)
        smoothed_cov = _symmetrize(
            cov + gain @ (next_smoothed_cov - next_predicted_cov) @ gain.T
        )
        return (smoothed_mean, smoothed_cov), (smoothed_mean, smoothed_cov)

    last = (filtered.filtered_mean[-1], filtered.filtered_cov[-1])
    inputs = (
        filtered.filtered_mean[:-1],
        filtered.filtered_cov[:-1],
        transition_matrix[1:],
        filtered.predicted_mean[1:],
        filtered.predicted_cov[1:],
    )
    _, (means, covs) = jax.lax.scan(step, last, inputs, reverse=True)
    return SmootherResult(
        jnp.concatenate([means, last[0][None]]), jnp.concatenate([covs, last[1][None]])
    )


def kalman_factor(name: str, ssm: LinearGaussianSSM, y: jax.Array) -> FilterResult:
    """Add log p(y), with the states integrated out, as the NumPyro factor site ``name``.

    Returns the filter pass, so a caller can record per-step contributions or moments.
    """
    filtered = kalman_filter(ssm, y)
    numpyro.factor(name, filtered.log_likelihood)
    return filtered


def _validate(ssm: LinearGaussianSSM, y: jax.Array) -> None:
    if y.ndim != 2 or ssm.initial_mean.ndim != 1:
        raise ValueError(
            f"y must be (T, p) and initial_mean (n,); "
            f"got {y.shape} and {ssm.initial_mean.shape}"
        )
    (num_steps, obs_dim), state_dim = y.shape, ssm.initial_mean.shape[0]
    expected = {
        "initial_cov": (state_dim, state_dim),
        "transition_matrix": (num_steps, state_dim, state_dim),
        "transition_offset": (num_steps, state_dim),
        "transition_cov": (num_steps, state_dim, state_dim),
        "observation_matrix": (num_steps, obs_dim, state_dim),
        "observation_offset": (num_steps, obs_dim),
        "observation_cov": (num_steps, obs_dim, obs_dim),
    }
    for name, shape in expected.items():
        actual = getattr(ssm, name).shape
        if actual != shape:
            raise ValueError(
                f"{name} has shape {actual}; expected {shape} "
                f"for T={num_steps}, n={state_dim}, p={obs_dim}"
            )
    for name, array in (("y", y), *ssm._asdict().items()):
        if array.dtype != jnp.float64:
            raise TypeError(
                f"{name} is {array.dtype}; the Kalman engine needs float64. "
                "Call numpyro.enable_x64() before the first JAX operation."
            )


def _symmetrize(matrix: jax.Array) -> jax.Array:
    return 0.5 * (matrix + matrix.T)
