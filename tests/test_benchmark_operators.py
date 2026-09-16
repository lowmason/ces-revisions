"""Req 10's backward wedge against the archived published benchmark vintages."""

import annual_data
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import vintage_data
from jax.experimental.sparse import BCOO

from ces_revisions.operators.benchmark import (
    apply_wedge,
    build_benchmark_fixtures,
    reconstruction_selector,
    wedge_operator,
    wedge_weights,
)


def fixtures() -> pl.DataFrame:
    return build_benchmark_fixtures(
        vintage_data.levels(),
        annual_data.artifact("benchmarks"),
        annual_data.artifact("reconstruction_events"),
    )


def test_wedge_operator_has_the_req_10_affine_coefficients():
    weights = wedge_weights()
    np.testing.assert_allclose(weights, np.arange(1, 13) / 12)
    operator = wedge_operator()
    assert isinstance(operator, BCOO)
    assert operator.shape == (12, 13)
    previous = jnp.zeros(12, dtype=jnp.float64)
    result = jax.jit(lambda x: operator @ x)(
        jnp.concatenate([previous, jnp.asarray([-861.0])])
    )
    np.testing.assert_allclose(result, -861.0 * np.arange(1, 13) / 12)
    assert result[0] == -71.75
    assert result[-1] == -861.0


def test_benchmark_fixture_covers_every_year_sector_and_backward_month():
    frame = fixtures()
    assert frame.height == 3_312
    assert frame["benchmark_year"].unique().sort().to_list() == list(range(2003, 2026))
    assert frame.group_by("benchmark_year", "sector").len()["len"].eq(12).all()
    assert frame.select("benchmark_year", "sector", "reference_month").n_unique() == (
        frame.height
    )


def test_fixture_reproduces_each_published_nsa_benchmark_vintage_exactly():
    frame = fixtures()
    np.testing.assert_allclose(
        frame["reproduced_level_thousands"],
        frame["benchmark_level_thousands"],
        rtol=0.0,
        atol=1e-9,
    )
    assert frame.filter(pl.col("reconstruction_supported")).height == 215
    assert frame.filter(~pl.col("reconstruction_supported")).height == 3_097
    assert (
        frame.filter(~pl.col("reconstruction_supported"))[
            "reconstruction_term_thousands"
        ]
        == 0.0
    ).all()
    assert (
        frame.filter(pl.col("reconstruction_supported"))["rounding_residual_thousands"]
        == 0.0
    ).all()
    assert (
        frame.filter(~pl.col("reconstruction_supported"))["rounding_residual_thousands"]
        .abs()
        .max()
        == 21.0
    )
    assert np.isclose(
        frame.filter(pl.col("reconstruction_supported"))[
            "reconstruction_term_thousands"
        ]
        .abs()
        .max(),
        469.1666666667,
    )


def test_march_reconstruction_terms_have_documented_event_support():
    march = fixtures().filter(pl.col("wedge_weight") == 1.0)
    nonzero = march.filter(pl.col("reconstruction_term_thousands") != 0.0)
    assert nonzero["event_ids"].list.len().gt(0).all()
    assert set(nonzero["benchmark_year"]) == {
        2010,
        2013,
        2015,
        2017,
        2018,
        2019,
        2022,
        2024,
        2025,
    }
    inferred = march.filter(pl.col("revision_status") == "inferred_from_archive")
    assert inferred.select("benchmark_year", "sector").rows() == [(2023, "10")]


def test_reconstruction_support_respects_each_events_reference_range():
    state_transfer = fixtures().filter(
        (pl.col("benchmark_year") == 2018)
        & pl.col("sector").is_in(["65", "90"])
        & pl.col("event_ids").list.contains("2018-state-ownership-change")
    )
    assert state_transfer["reference_month"].dt.month().unique().sort().to_list() == [
        1,
        2,
        3,
    ]


def test_reconstruction_selector_cannot_emit_outside_documented_support():
    support = [False, True, False, True]
    operator = reconstruction_selector(support)
    assert isinstance(operator, BCOO)
    result = operator @ jnp.asarray([3.0, 4.0, 5.0, 6.0], dtype=jnp.float64)
    np.testing.assert_array_equal(result, [0.0, 4.0, 0.0, 6.0])


def test_apply_wedge_requires_twelve_months_and_adds_terms_separately():
    previous = jnp.arange(12.0, dtype=jnp.float64)
    support = [False] * 11 + [True]
    reconstructed = apply_wedge(
        previous,
        23.0,
        reconstruction_terms=jnp.ones(12, dtype=jnp.float64),
        reconstruction_support=support,
        rounding_residuals=jnp.full(12, 0.25, dtype=jnp.float64),
    )
    expected = np.asarray(
        wedge_operator() @ jnp.concatenate([previous, jnp.array([23.0])])
    )
    np.testing.assert_allclose(reconstructed[:-1], expected[:-1] + 0.25)
    np.testing.assert_allclose(reconstructed[-1], expected[-1] + 1.25)
