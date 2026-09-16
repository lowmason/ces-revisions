"""Req 10 post-March representations and their public-data determination."""

from functools import cache

import annual_data
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
import vintage_data
from jax.experimental.sparse import BCOO

from ces_revisions.operators.post_march import (
    ROUNDING_VARIANCE_THOUSANDS2,
    apply_post_march,
    build_post_march_components,
    build_recoverability,
    post_march_cumulative_operator,
    post_march_link_operator,
)


@cache
def components() -> pl.DataFrame:
    return build_post_march_components(
        vintage_data.levels(), annual_data.artifact("birth_death")
    )


@cache
def recoverability() -> pl.DataFrame:
    return build_recoverability(
        vintage_data.levels(), annual_data.artifact("birth_death")
    )


def test_link_relative_operator_matches_the_recursive_definition():
    links = jnp.asarray([1.01, 0.99, 1.02], dtype=jnp.float64)
    revised_bd = jnp.asarray([2.0, -1.0, 3.0], dtype=jnp.float64)
    operator = post_march_link_operator(links)
    assert isinstance(operator, BCOO)
    assert operator.shape == (3, 4)
    expected = []
    level = 100.0
    for link, adjustment in zip(links, revised_bd, strict=True):
        level = level * link + adjustment
        expected.append(level)
    actual = jax.jit(lambda x: operator @ x)(
        jnp.concatenate([jnp.asarray([100.0]), revised_bd])
    )
    np.testing.assert_allclose(actual, expected)


def test_cumulative_operator_adds_sample_and_revised_birth_death_changes():
    sample = jnp.asarray([4.0, -2.0, 1.0], dtype=jnp.float64)
    revised_bd = jnp.asarray([2.0, 3.0, -1.0], dtype=jnp.float64)
    operator = post_march_cumulative_operator(3)
    assert isinstance(operator, BCOO)
    assert operator.shape == (3, 7)
    actual = operator @ jnp.concatenate([jnp.asarray([100.0]), sample, revised_bd])
    np.testing.assert_array_equal(actual, [106.0, 107.0, 107.0])


def test_apply_requires_exactly_one_representation():
    revised_bd = jnp.ones(2, dtype=jnp.float64)
    with pytest.raises(ValueError, match="exactly one"):
        apply_post_march(100.0, revised_bd)
    with pytest.raises(ValueError, match="exactly one"):
        apply_post_march(
            100.0,
            revised_bd,
            link_relatives=jnp.ones(2),
            sample_job_changes=jnp.ones(2),
        )


def test_apply_uses_each_representation_without_mixing_inputs():
    revised_bd = jnp.asarray([2.0, 3.0], dtype=jnp.float64)
    by_link = apply_post_march(
        100.0, revised_bd, link_relatives=jnp.asarray([1.0, 1.0])
    )
    by_change = apply_post_march(
        100.0, revised_bd, sample_job_changes=jnp.asarray([4.0, -2.0])
    )
    np.testing.assert_array_equal(by_link, [102.0, 105.0])
    np.testing.assert_array_equal(by_change, [106.0, 107.0])


def test_inferred_cumulative_components_reproduce_every_published_target():
    frame = components()
    assert frame.height == 2_484
    formal = frame.filter(~pl.col("additional_sample_receipts"))
    assert formal.height == 1_932
    np.testing.assert_allclose(
        frame["reproduced_level_thousands"],
        frame["published_level_thousands"],
        rtol=0.0,
        atol=1e-9,
    )
    assert frame["representation"].unique().to_list() == ["cumulative_job_change"]
    assert frame["series_status"].unique().to_list() == ["inferred"]
    assert frame.filter(pl.col("additional_sample_receipts")).height == 552


def test_components_consume_revised_not_initial_birth_death_values():
    schedules = annual_data.artifact("birth_death").filter(
        pl.col("series_kind") == "published_schedule"
    )
    revised = schedules.filter(pl.col("schedule_kind") == "post_benchmark").select(
        "benchmark_year",
        "sector",
        "reference_month",
        expected=pl.col("value_thousands"),
    )
    checked = components().join(
        revised, on=["benchmark_year", "sector", "reference_month"]
    )
    assert (checked["revised_birth_death_thousands"] == checked["expected"]).all()
    initial = schedules.filter(pl.col("schedule_kind") == "preliminary").select(
        "sector",
        "reference_month",
        initial=pl.col("value_thousands"),
    )
    assert (
        checked.join(initial, on=["sector", "reference_month"])
        .select((pl.col("revised_birth_death_thousands") != pl.col("initial")).any())
        .item()
    )


def test_recoverability_selects_one_inferred_representation_for_every_year():
    frame = recoverability()
    assert frame.height == 23
    assert frame["benchmark_year"].to_list() == list(range(2003, 2026))
    assert frame["representation"].unique().to_list() == ["cumulative_job_change"]
    assert frame["series_status"].unique().to_list() == ["inferred"]
    assert frame.filter(pl.col("link_attempt_status") == "inputs_missing")[
        "benchmark_year"
    ].to_list() == [2003]
    assert frame.filter(pl.col("link_attempt_status") == "failed").height == 22
    assert frame["reconstruction_error_variance_thousands2"].gt(0.0).all()


@pytest.mark.parametrize(
    (
        "year",
        "attempted",
        "within_rounding",
        "maximum",
        "variance",
    ),
    [
        (2003, 0, 0, None, ROUNDING_VARIANCE_THOUSANDS2),
        (2004, 84, 19, 33.407275, 65.199745),
        (2020, 84, 3, 162.058501, 1607.645923),
        (2021, 84, 12, 179.079008, 1293.123145),
        (2025, 84, 11, 17.229047, 30.739396),
    ],
)
def test_recoverability_diagnostics_are_pinned(
    year, attempted, within_rounding, maximum, variance
):
    row = recoverability().filter(pl.col("benchmark_year") == year).row(0, named=True)
    assert row["attempted_cells"] == attempted
    assert row["within_rounding_cells"] == within_rounding
    if maximum is None:
        assert row["max_abs_residual_thousands"] is None
    else:
        assert np.isclose(row["max_abs_residual_thousands"], maximum, atol=1e-6)
    assert np.isclose(
        row["reconstruction_error_variance_thousands2"], variance, atol=1e-6
    )


def test_failed_link_attempts_retain_every_residual_and_source_key():
    failed = recoverability().filter(pl.col("link_attempt_status") == "failed")
    assert failed["link_attempt_residuals_thousands"].list.len().eq(84).all()
    assert failed["level_cell_ids"].list.len().gt(0).all()
    assert failed["birth_death_source_keys"].list.len().gt(0).all()
    for row in failed.iter_rows(named=True):
        residuals = np.asarray(row["link_attempt_residuals_thousands"])
        assert np.isclose(
            np.mean(residuals**2),
            row["reconstruction_error_variance_thousands2"],
        )


def test_nonzero_variance_is_equivalent_to_inferred_status():
    frame = recoverability()
    assert (
        (frame["reconstruction_error_variance_thousands2"] > 0.0)
        == (frame["series_status"] == "inferred")
    ).all()
