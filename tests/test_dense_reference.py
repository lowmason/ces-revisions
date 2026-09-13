"""The dense oracle agrees with a two-step scalar model worked by hand."""

import math

import numpy as np
from dense_reference import dense_reference

# x_{-1} ~ N(1, 2);  x_0 = 0.5 x_{-1} + 1 + N(0, 1);  x_1 = 2 x_0 - 1 + N(0, 3);
# y_t = 1.5 x_t + 0.5 + N(0, 0.25), with y_0 missing and y_1 = 4.
# By hand: x_0 ~ N(1.5, 1.5), x_1 ~ N(2, 9), y_0 ~ N(2.75, 3.625), y_1 ~ N(3.5, 20.5),
# and Cov(x_0, y_1) = 1.5 * 2 * 1.5 = 4.5. The two steps have different transitions, so
# these numbers pin the convention that A_t maps x_{t-1} to x_t.
HAND_MODEL = {
    "initial_mean": np.array([1.0]),
    "initial_cov": np.array([[2.0]]),
    "transition_matrix": np.array([[[0.5]], [[2.0]]]),
    "transition_offset": np.array([[1.0], [-1.0]]),
    "transition_cov": np.array([[[1.0]], [[3.0]]]),
    "observation_matrix": np.array([[[1.5]], [[1.5]]]),
    "observation_offset": np.array([[0.5], [0.5]]),
    "observation_cov": np.array([[[0.25]], [[0.25]]]),
}
HAND_PANEL = np.array([[np.nan], [4.0]])


def test_dense_reference_matches_a_hand_worked_two_step_model():
    reference = dense_reference(HAND_MODEL, HAND_PANEL)
    gain = 9.0 * 1.5 / 20.5  # Kalman gain for x_1 given y_1
    log_likelihood = -0.5 * (math.log(2 * math.pi) + math.log(20.5) + 0.5**2 / 20.5)

    np.testing.assert_allclose(reference["log_likelihood"], log_likelihood)
    np.testing.assert_allclose(reference["step_log_likelihood"], [0.0, log_likelihood])
    np.testing.assert_allclose(reference["predicted_mean"][:, 0], [1.5, 2.0])
    np.testing.assert_allclose(reference["predicted_cov"][:, 0, 0], [1.5, 9.0])
    np.testing.assert_allclose(
        reference["filtered_mean"][:, 0], [1.5, 2.0 + gain * 0.5]
    )
    np.testing.assert_allclose(
        reference["filtered_cov"][:, 0, 0], [1.5, 9.0 - gain * 13.5]
    )
    np.testing.assert_allclose(reference["one_step_mean"][:, 0], [2.75, 3.5])
    np.testing.assert_allclose(reference["one_step_cov"][:, 0, 0], [3.625, 20.5])
    np.testing.assert_allclose(
        reference["smoothed_mean"][:, 0], [1.5 + 4.5 / 20.5 * 0.5, 2.0 + gain * 0.5]
    )
    np.testing.assert_allclose(
        reference["smoothed_cov"][:, 0, 0], [1.5 - 4.5**2 / 20.5, 9.0 - gain * 13.5]
    )
