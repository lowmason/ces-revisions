"""Sparse selectors for observed cells and irregular March benchmark rows."""

from collections.abc import Sequence
from datetime import date

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO

VALID_STATUSES = (
    "observed",
    "missing_vintage",
    "beyond_frontier",
    "right_censored",
)


def observed_mask(statuses: Sequence[str]) -> jax.Array:
    unknown = sorted(set(statuses) - set(VALID_STATUSES))
    if unknown:
        raise ValueError(f"unknown stage-label statuses: {unknown}")
    return jnp.asarray([status == "observed" for status in statuses], dtype=jnp.bool_)


def mask_values(values: jax.Array, statuses: Sequence[str]) -> jax.Array:
    array = jnp.asarray(values, dtype=jnp.float64)
    mask = observed_mask(statuses)
    if array.shape[-1] != mask.shape[0]:
        raise ValueError("values and statuses have different cell counts")
    return jnp.where(mask, array, jnp.nan)


def selection_operator(mask: Sequence[bool] | jax.Array) -> BCOO:
    flags = np.asarray(mask, dtype=bool)
    selected_columns = np.flatnonzero(flags).astype(np.int32, copy=False)
    selected_rows = np.arange(selected_columns.size, dtype=np.int32)
    indices = jnp.asarray(
        np.column_stack((selected_rows, selected_columns)), dtype=jnp.int32
    )
    data = jnp.ones(selected_columns.size, dtype=jnp.float64)
    return BCOO(
        (data, indices),
        shape=(selected_columns.size, flags.size),
    )


def march_mask(reference_months: Sequence[date]) -> jax.Array:
    return jnp.asarray(
        [month.month == 3 for month in reference_months], dtype=jnp.bool_
    )


def march_selection_operator(reference_months: Sequence[date]) -> BCOO:
    return selection_operator(march_mask(reference_months))
