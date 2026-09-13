"""Dense Gaussian oracle for the Kalman engine.

Stacks every state and observation cell of a linear-Gaussian state-space model into one
joint normal, and answers each filtering and smoothing question by direct Gaussian
conditioning. It shares no code with ``ces_revisions.kalman`` and is quadratic in memory,
so use it only on tiny models. It follows the engine's conventions: the state evolves as
``x_t = A_t x_{t-1} + c_t + w_t`` from ``x_{-1} ~ N(m, P)`` and is observed as
``y_t = Z_t x_t + d_t + v_t``, and NaN marks a missing cell. Models are dicts of NumPy
arrays keyed by the ``LinearGaussianSSM`` field names.
"""

import numpy as np


def _random_spd(rng, dim):
    root = rng.normal(size=(dim, dim))
    return root @ root.T / dim + 0.5 * np.eye(dim)


def random_ssm(rng, num_steps, state_dim, obs_dim):
    """A time-varying model whose arrays are drawn independently at every step."""
    return {
        "initial_mean": rng.normal(size=state_dim),
        "initial_cov": _random_spd(rng, state_dim),
        "transition_matrix": 0.9 * np.eye(state_dim)
        + 0.2 * rng.normal(size=(num_steps, state_dim, state_dim)),
        "transition_offset": rng.normal(size=(num_steps, state_dim)),
        "transition_cov": np.stack(
            [_random_spd(rng, state_dim) for _ in range(num_steps)]
        ),
        "observation_matrix": rng.normal(size=(num_steps, obs_dim, state_dim)),
        "observation_offset": rng.normal(size=(num_steps, obs_dim)),
        "observation_cov": np.stack(
            [_random_spd(rng, obs_dim) for _ in range(num_steps)]
        ),
    }


def time_invariant(ssm):
    """The same model with every time-indexed array replaced by its step-0 value."""
    return {
        name: np.broadcast_to(value[0], value.shape).copy()
        if name.startswith(("transition", "observation"))
        else value
        for name, value in ssm.items()
    }


def simulate(ssm, rng):
    """Draw one fully observed panel of shape (T, p) from the model."""
    state = rng.multivariate_normal(ssm["initial_mean"], ssm["initial_cov"])
    rows = []
    for a, c, q, z, d, r in zip(
        ssm["transition_matrix"],
        ssm["transition_offset"],
        ssm["transition_cov"],
        ssm["observation_matrix"],
        ssm["observation_offset"],
        ssm["observation_cov"],
        strict=True,
    ):
        state = a @ state + c + rng.multivariate_normal(np.zeros(len(c)), q)
        rows.append(z @ state + d + rng.multivariate_normal(np.zeros(len(d)), r))
    return np.stack(rows)


def _block_diag(blocks):
    """Place (T, a, b) blocks along the diagonal of a (T*a, T*b) matrix."""
    num_steps, rows, cols = blocks.shape
    out = np.zeros((num_steps * rows, num_steps * cols))
    for t, block in enumerate(blocks):
        out[t * rows : (t + 1) * rows, t * cols : (t + 1) * cols] = block
    return out


def _joint_moments(ssm):
    """Moments of the stacked states X (T*n) and cells Y (T*p), and Cov(X, Y)."""
    a, c, q = ssm["transition_matrix"], ssm["transition_offset"], ssm["transition_cov"]
    num_steps, n = c.shape
    mean_x = np.zeros((num_steps, n))
    cov_x = np.zeros((num_steps * n, num_steps * n))
    prev_mean, prev_cov = ssm["initial_mean"], ssm["initial_cov"]
    prev_cross = np.zeros((n, 0))  # Cov(x_{t-1}, x_0..x_{t-1})
    for t in range(num_steps):
        here = slice(t * n, (t + 1) * n)
        mean_x[t] = a[t] @ prev_mean + c[t]
        cross = a[t] @ prev_cross  # Cov(x_t, x_0..x_{t-1})
        cov_x[here, : t * n] = cross
        cov_x[: t * n, here] = cross.T
        cov_x[here, here] = a[t] @ prev_cov @ a[t].T + q[t]
        prev_mean, prev_cov = mean_x[t], cov_x[here, here]
        prev_cross = cov_x[here, : (t + 1) * n]
    big_z = _block_diag(ssm["observation_matrix"])
    mean_y = big_z @ mean_x.ravel() + ssm["observation_offset"].ravel()
    cov_y = big_z @ cov_x @ big_z.T + _block_diag(ssm["observation_cov"])
    return mean_x.ravel(), cov_x, mean_y, cov_y, cov_x @ big_z.T


def _log_normal(value, mean, cov):
    if value.size == 0:
        return 0.0
    chol = np.linalg.cholesky(cov)
    white = np.linalg.solve(chol, value - mean)
    log_det = 2 * np.log(np.diag(chol)).sum()
    return -0.5 * (value.size * np.log(2 * np.pi) + log_det + white @ white)


def _condition(mean_a, cov_aa, cov_ab, mean_b, cov_bb, value_b):
    """Mean and covariance of a given b = value_b, for jointly Gaussian (a, b)."""
    if value_b.size == 0:
        return mean_a, cov_aa
    weights = np.linalg.solve(cov_bb, cov_ab.T).T
    return mean_a + weights @ (value_b - mean_b), cov_aa - weights @ cov_ab.T


def dense_reference(ssm, y):
    """Every FilterResult and SmootherResult field, by dense Gaussian conditioning."""
    mean_x, cov_x, mean_y, cov_y, cov_xy = _joint_moments(ssm)
    num_steps, p = y.shape
    n = len(ssm["initial_mean"])
    flat = y.ravel()
    observed = ~np.isnan(flat)
    everything = np.flatnonzero(observed)

    def observed_before(t):
        return np.flatnonzero(observed[: t * p])

    def log_likelihood_before(t):
        idx = observed_before(t)
        return _log_normal(flat[idx], mean_y[idx], cov_y[np.ix_(idx, idx)])

    def state_given(t, idx):
        here = slice(t * n, (t + 1) * n)
        return _condition(
            mean_x[here],
            cov_x[here, here],
            cov_xy[here, idx],
            mean_y[idx],
            cov_y[np.ix_(idx, idx)],
            flat[idx],
        )

    steps = []
    for t in range(num_steps):
        cells = slice(t * p, (t + 1) * p)
        before = observed_before(t)
        one_step = _condition(
            mean_y[cells],
            cov_y[cells, cells],
            cov_y[cells, before],
            mean_y[before],
            cov_y[np.ix_(before, before)],
            flat[before],
        )
        steps.append(
            (
                log_likelihood_before(t + 1) - log_likelihood_before(t),
                *state_given(t, before),
                *state_given(t, observed_before(t + 1)),
                *one_step,
                *state_given(t, everything),
            )
        )
    names = (
        "step_log_likelihood",
        "predicted_mean",
        "predicted_cov",
        "filtered_mean",
        "filtered_cov",
        "one_step_mean",
        "one_step_cov",
        "smoothed_mean",
        "smoothed_cov",
    )
    columns = (np.array(column) for column in zip(*steps, strict=True))
    return {
        "log_likelihood": log_likelihood_before(num_steps),
        **dict(zip(names, columns, strict=True)),
    }
