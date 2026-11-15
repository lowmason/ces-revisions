"""Pointwise log likelihood for blocked PSIS-LOO, and the DataTree's netCDF file.

The Kalman factor is one site, so ``az.from_numpyro`` sees one log likelihood per draw. The
filter's per-step terms ``log p(y_t | y_<t)`` sum to it exactly, so recomputing them at every
draw gives LOO two blockings with the same total: one block per reference month, and one
per benchmark year (Req 10's April-to-March window). A step with no observed cell adds an
exact zero and is dropped from the month blocking.
"""

from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

NETCDF_ENGINE = "h5netcdf"


def step_log_likelihood_draws(
    step_fn: Callable[[dict[str, jax.Array]], jax.Array],
    samples_by_chain: dict[str, np.ndarray],
    *,
    batch_size: int = 500,
) -> np.ndarray:
    """step_fn at every draw, as a (chain, draw, T) array.

    ``samples_by_chain`` holds (chain, draw, ...) arrays, as from
    ``mcmc.get_samples(group_by_chain=True)``. The draws are flattened and mapped in
    vmapped batches of ``batch_size`` under one jit, which bounds memory on a GPU.
    """
    leading = {tuple(np.shape(value)[:2]) for value in samples_by_chain.values()}
    if len(leading) != 1:
        raise ValueError(f"every site needs the same (chain, draw) axes; got {leading}")
    ((num_chains, num_draws),) = leading
    flat = {
        name: jnp.reshape(
            jnp.asarray(value), (num_chains * num_draws, *np.shape(value)[2:])
        )
        for name, value in samples_by_chain.items()
    }
    mapped = jax.jit(lambda params: jax.lax.map(step_fn, params, batch_size=batch_size))
    values = np.asarray(mapped(flat), dtype=np.float64)
    return values.reshape(num_chains, num_draws, -1)


def attach_log_likelihood(
    idata,
    step_log_likelihood: np.ndarray,
    reference_months: Sequence[date],
    benchmark_year: np.ndarray,
    observed_any: np.ndarray,
    name: str = "panel",
) -> xr.DataTree:
    """A copy of idata with a log_likelihood group blocked by month and benchmark year.

    ``{name}_month`` is (chain, draw, reference_month) over the steps with an observed
    cell, and ``{name}_benchmark_year`` is (chain, draw, benchmark_year): the sum of those
    steps within each benchmark year. Raises ValueError if a dropped step's log likelihood
    is not exactly zero, since then neither blocking would sum to the model's.
    """
    step_log_likelihood = np.asarray(step_log_likelihood, dtype=np.float64)
    observed_any = np.asarray(observed_any, dtype=bool)
    benchmark_year = np.asarray(benchmark_year, dtype=np.int64)
    posterior = idata["posterior"]
    num_chains, num_draws = posterior.sizes["chain"], posterior.sizes["draw"]
    num_steps = len(reference_months)
    if step_log_likelihood.shape != (num_chains, num_draws, num_steps):
        raise ValueError(
            f"step_log_likelihood must be ({num_chains}, {num_draws}, {num_steps}); "
            f"got {step_log_likelihood.shape}"
        )
    if observed_any.shape != (num_steps,) or benchmark_year.shape != (num_steps,):
        raise ValueError(f"observed_any and benchmark_year must be ({num_steps},)")
    if np.any(step_log_likelihood[..., ~observed_any] != 0.0):
        raise ValueError("a step without an observed cell has a nonzero log likelihood")

    month_values = step_log_likelihood[..., observed_any]
    observed_years = benchmark_year[observed_any]
    years = np.unique(observed_years)
    year_values = np.stack(
        [month_values[..., observed_years == year].sum(axis=-1) for year in years],
        axis=-1,
    )
    months = np.array(
        [np.datetime64(month, "ns") for month in reference_months],
        dtype="datetime64[ns]",
    )[observed_any]
    sample_coords = {
        "chain": posterior["chain"].values,
        "draw": posterior["draw"].values,
    }
    log_likelihood = xr.Dataset(
        {
            f"{name}_month": (("chain", "draw", "reference_month"), month_values),
            f"{name}_benchmark_year": (
                ("chain", "draw", "benchmark_year"),
                year_values,
            ),
        },
        coords={**sample_coords, "reference_month": months, "benchmark_year": years},
    )
    tree = idata.copy()
    tree["log_likelihood"] = xr.DataTree(log_likelihood)
    return tree


def write_inference_data(idata, path: Path) -> None:
    """Write every group of the DataTree to one netCDF file through h5netcdf."""
    idata.to_netcdf(path, engine=NETCDF_ENGINE)


def read_inference_data(path: Path) -> xr.DataTree:
    """Read a DataTree written by write_inference_data fully into memory."""
    with xr.open_datatree(path, engine=NETCDF_ENGINE) as tree:
        return tree.load()
