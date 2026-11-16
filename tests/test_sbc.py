"""Simulation-based calibration arithmetic, checked against exact rational oracles."""

import functools
import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from ces_revisions.pilot.sbc import (
    ecdf_test,
    gamma_critical_value,
    gamma_statistic,
    rank_of,
    recovery,
    thin,
)

SEED = sum(map(ord, "ces-revisions-stage-6-sbc"))
# A case small enough to enumerate every rank vector: 5 replicates, ranks in 0..4.
TINY_REPLICATES, TINY_DRAWS = 5, 4


@functools.cache
def _exact_tails(num_replicates, num_draws, point, count):
    """Exact lower P(X <= count) and upper 1 - P(X <= count - 1), X ~ Bin(N, z_point)."""
    z = Fraction(point, num_draws + 1)

    def cdf(k):
        return sum(
            math.comb(num_replicates, j) * z**j * (1 - z) ** (num_replicates - j)
            for j in range(k + 1)
        )

    return cdf(count), 1 - cdf(count - 1)


def brute_force_gamma(ranks, num_draws):
    """Gamma by the definition, one evaluation point at a time, in exact rationals."""
    smallest = None
    for point in range(1, num_draws + 1):
        count = sum(1 for rank in ranks if rank < point)
        lower, upper = _exact_tails(len(ranks), num_draws, point, count)
        smallest = (
            min(lower, upper) if smallest is None else min(smallest, lower, upper)
        )
    return 2 * smallest


def test_rank_of_counts_the_draws_below_the_prior_value():
    draws = np.array([0.1, 0.9, 0.3, 0.7])

    assert rank_of(0.5, draws) == 2
    assert rank_of(-1.0, draws) == 0
    assert rank_of(2.0, draws) == draws.size


def test_rank_of_breaks_ties_uniformly_over_the_tied_positions():
    rng = np.random.default_rng(SEED)
    num_draws, repeats = 4, 5_000
    ranks = [rank_of(1.0, np.ones(num_draws), rng) for _ in range(repeats)]
    counts = np.bincount(ranks, minlength=num_draws + 1)
    expected = repeats / (num_draws + 1)
    sd = math.sqrt(repeats * (1 / (num_draws + 1)) * (num_draws / (num_draws + 1)))

    assert counts.size == num_draws + 1
    assert np.abs(counts - expected).max() < 4 * sd
    partial = {rank_of(2.0, np.array([1.0, 2.0, 2.0, 3.0]), rng) for _ in range(200)}
    assert partial == {1, 2, 3}


def test_rank_of_repeats_for_a_seed_and_refuses_unbroken_ties_and_non_finite_input():
    draws = np.array([0.0, 1.0, 1.0, 1.0, 2.0])
    first = [rank_of(1.0, draws, np.random.default_rng(SEED)) for _ in range(3)]

    assert len(set(first)) == 1
    with pytest.raises(ValueError, match="tie"):
        rank_of(1.0, draws)
    with pytest.raises(ValueError, match="finite"):
        rank_of(np.nan, draws)
    with pytest.raises(ValueError, match="finite"):
        rank_of(0.5, np.array([0.0, np.inf]))
    with pytest.raises(ValueError, match="1-D"):
        rank_of(0.5, np.zeros((2, 2)))


def test_thin_keeps_evenly_spaced_draws_from_the_first():
    np.testing.assert_array_equal(thin(np.arange(1000), 100), np.arange(0, 1000, 10))
    np.testing.assert_array_equal(thin(np.arange(10), 3), [0, 3, 6])
    np.testing.assert_array_equal(thin(np.arange(7), 7), np.arange(7))
    np.testing.assert_array_equal(thin(np.arange(7.0), 1), [0.0])
    for num_kept in (0, 11):
        with pytest.raises(ValueError, match="num_kept"):
            thin(np.arange(10), num_kept)
    with pytest.raises(ValueError, match="1-D"):
        thin(np.zeros((4, 4)), 2)


def test_recovery_reports_the_posterior_z_score_and_contraction():
    result = recovery(2.0, np.array([1.0, 2.0, 3.0, 4.0, 5.0]), prior_sd=5.0)

    # Mean 3 and ddof=1 variance 2.5.
    assert result["z"] == pytest.approx(1.0 / math.sqrt(2.5), rel=1e-15)
    assert result["contraction"] == pytest.approx(0.9, rel=1e-15)
    with pytest.raises(ValueError, match="prior_sd"):
        recovery(2.0, np.array([1.0, 2.0]), prior_sd=0.0)


def test_recovery_of_a_constant_posterior_is_non_finite_not_perfect(recwarn):
    # A zero-variance chain would divide by zero: z = inf, and contraction = 1.0 would
    # read as perfect recovery. Both are NaN instead, with no numpy warning.
    for prior_value in (2.0, 3.0):
        result = recovery(prior_value, np.full(5, 3.0), prior_sd=5.0)

        assert math.isnan(result["z"])
        assert math.isnan(result["contraction"])
    assert not recwarn.list


def test_gamma_matches_the_exact_definition_on_every_tiny_rank_vector():
    vectors = itertools.product(range(TINY_DRAWS + 1), repeat=TINY_REPLICATES)
    for ranks in vectors:
        expected = float(brute_force_gamma(ranks, TINY_DRAWS))
        actual = gamma_statistic(np.array(ranks), TINY_DRAWS)
        assert actual == pytest.approx(expected, rel=1e-12), ranks


@pytest.mark.parametrize(
    ("num_replicates", "num_draws"), [(1, 1), (3, 9), (50, 20), (200, 99)]
)
def test_gamma_matches_the_exact_definition_on_random_rank_vectors(
    num_replicates, num_draws
):
    rng = np.random.default_rng(SEED + num_replicates)
    for _ in range(5):
        ranks = rng.integers(0, num_draws + 1, size=num_replicates)
        expected = float(brute_force_gamma(ranks.tolist(), num_draws))
        assert gamma_statistic(ranks, num_draws) == pytest.approx(expected, rel=1e-12)


def test_perfectly_uniform_ranks_pass_and_a_point_mass_fails():
    num_draws, copies = 19, 10
    uniform = np.repeat(np.arange(num_draws + 1), copies)
    point_mass = np.zeros(uniform.size, dtype=np.int64)
    # Every count of ranks below i equals its binomial mean N * z_i = copies * i exactly.
    np.random.default_rng(SEED).shuffle(uniform)

    passed = ecdf_test(uniform, num_draws, 0.05, seed=SEED)
    failed = ecdf_test(point_mass, num_draws, 0.05, seed=SEED)

    assert passed["gamma"] == pytest.approx(
        float(brute_force_gamma(uniform.tolist(), num_draws)), rel=1e-12
    )
    assert passed["gamma"] > 0.5
    assert passed["passed"] is True
    assert failed["passed"] is False
    assert failed["gamma"] < 1e-100
    assert set(passed) == {"gamma", "critical", "passed"}
    assert all(isinstance(passed[key], float) for key in ("gamma", "critical"))


def test_ranks_bunched_in_the_middle_fail():
    # An overdispersed posterior puts the prior value near its middle draw every time.
    rng = np.random.default_rng(SEED)
    num_draws = 99
    bunched = rng.binomial(num_draws, 0.5, size=200)

    assert ecdf_test(bunched, num_draws, 0.05, seed=SEED)["passed"] is False


def test_critical_value_is_the_exact_quantile_of_the_enumerated_null():
    vectors = list(itertools.product(range(TINY_DRAWS + 1), repeat=TINY_REPLICATES))
    exact = sorted(brute_force_gamma(ranks, TINY_DRAWS) for ranks in vectors)
    for alpha in (0.05, 0.1, 0.2):
        # The inverse-CDF quantile of the equally likely enumerated gammas.
        quantile = exact[math.ceil(alpha * len(exact)) - 1]
        size = sum(1 for gamma in exact if gamma < quantile) / len(exact)
        critical = gamma_critical_value(
            TINY_REPLICATES, TINY_DRAWS, alpha, num_simulations=20_000, seed=SEED
        )

        assert critical == pytest.approx(float(quantile), rel=1e-12), alpha
        assert size < alpha


def test_critical_value_repeats_for_a_seed_and_rises_with_alpha():
    def critical(alpha, seed=SEED):
        return gamma_critical_value(100, 50, alpha, num_simulations=4_000, seed=seed)

    assert critical(0.05) == critical(0.05)
    assert critical(0.01) <= critical(0.05) <= critical(0.10)
    assert critical(0.01) < critical(0.10)


@pytest.mark.parametrize(
    ("ranks", "num_draws", "message"),
    [
        (np.array([0, 1, 5]), 4, "0..4"),
        (np.array([-1, 1]), 4, "0..4"),
        (np.array([0.0, 1.0]), 4, "integers"),
        (np.array([], dtype=np.int64), 4, "non-empty"),
        (np.array([0, 1]), 0, "num_draws"),
    ],
)
def test_gamma_rejects_ranks_outside_0_to_l(ranks, num_draws, message):
    with pytest.raises(ValueError, match=message):
        gamma_statistic(ranks, num_draws)


def test_critical_value_rejects_alpha_outside_the_unit_interval():
    for alpha in (0.0, 1.0):
        with pytest.raises(ValueError, match="alpha"):
            gamma_critical_value(10, 10, alpha, seed=SEED)
