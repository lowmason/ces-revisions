"""Sparse aggregation, missing-vintage masks, and March selection."""

from datetime import date

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import vintage_data
from jax.experimental.sparse import BCOO

from ces_revisions.operators import OBSERVATION_SECTOR_ORDER, SECTOR_ORDER
from ces_revisions.operators.aggregation import (
    aggregate_sector_states,
    aggregation_operator,
)
from ces_revisions.operators.masks import (
    march_mask,
    march_selection_operator,
    mask_values,
    observed_mask,
    selection_operator,
)


def test_axis_orders_are_the_req_5_supersectors_and_published_total():
    assert SECTOR_ORDER == (
        "10",
        "20",
        "30",
        "40",
        "50",
        "55",
        "60",
        "65",
        "70",
        "80",
        "90",
    )
    assert OBSERVATION_SECTOR_ORDER == ("00", *SECTOR_ORDER)


def test_aggregation_operator_is_sparse_float64_and_jittable():
    operator = aggregation_operator()
    assert isinstance(operator, BCOO)
    assert operator.shape == (12, 11)
    assert operator.dtype == jnp.float64
    values = jnp.arange(1.0, 12.0, dtype=jnp.float64)
    result = jax.jit(lambda x: operator @ x)(values)
    np.testing.assert_array_equal(result[0], values.sum())
    np.testing.assert_array_equal(result[1:], values)


def test_aggregation_reproduces_every_published_total_nonfarm_level():
    levels = vintage_data.levels().filter(pl.col("value_thousands").is_not_null())
    keys = ["release_month", "seasonal_status", "reference_month"]
    wide = levels.pivot(on="sector", index=keys, values="value_thousands").sort(keys)
    assert wide.height == 496_310
    calculated = np.asarray(
        aggregate_sector_states(
            jnp.asarray(wide.select(*SECTOR_ORDER).to_numpy(), dtype=jnp.float64)
        )
    )
    np.testing.assert_array_equal(calculated[:, 0], wide["00"].to_numpy())
    np.testing.assert_array_equal(
        calculated[:, 1:], wide.select(*SECTOR_ORDER).to_numpy()
    )


def test_masks_preserve_each_missingness_reason_and_never_zero_fill():
    statuses = [
        "observed",
        "missing_vintage",
        "beyond_frontier",
        "right_censored",
    ]
    mask = observed_mask(statuses)
    np.testing.assert_array_equal(mask, [True, False, False, False])
    masked = mask_values(jnp.arange(4.0, dtype=jnp.float64), statuses)
    assert masked[0] == 0.0
    assert jnp.isnan(masked[1:]).all()
    selected = selection_operator(mask) @ jnp.arange(4.0, dtype=jnp.float64)
    np.testing.assert_array_equal(selected, [0.0])


def test_march_selection_is_sparse_and_keeps_only_march_rows():
    months = [date(2024, 2, 1), date(2024, 3, 1), date(2025, 3, 1)]
    np.testing.assert_array_equal(march_mask(months), [False, True, True])
    operator = march_selection_operator(months)
    assert isinstance(operator, BCOO)
    np.testing.assert_array_equal(
        operator @ jnp.asarray([2.0, 3.0, 4.0], dtype=jnp.float64),
        [3.0, 4.0],
    )
