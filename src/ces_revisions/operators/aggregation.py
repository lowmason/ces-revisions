"""Req 5 sector aggregation while retaining the published total as an observation."""

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

from ces_revisions.operators import SECTOR_ORDER


def aggregation_operator() -> BCOO:
    """Map eleven sector states to calculated total plus the eleven identities."""
    count = len(SECTOR_ORDER)
    dense = jnp.concatenate(
        [jnp.ones((1, count), dtype=jnp.float64), jnp.eye(count, dtype=jnp.float64)]
    )
    return BCOO.fromdense(dense, nse=2 * count)


def aggregate_sector_states(sector_values: jax.Array) -> jax.Array:
    """Apply the aggregation map to the last axis of a float64 array."""
    values = jnp.asarray(sector_values, dtype=jnp.float64)
    if values.ndim == 0 or values.shape[-1] != len(SECTOR_ORDER):
        raise ValueError(f"sector_values must end in {len(SECTOR_ORDER)} sectors")
    flat = values.reshape((-1, len(SECTOR_ORDER))).T
    aggregated = (aggregation_operator() @ flat).T
    return aggregated.reshape((*values.shape[:-1], len(SECTOR_ORDER) + 1))
