"""Dispersed NUTS starting points, placed from the caller's prior draws.

Req 18 asks for four dispersed chains. Each chain starts at a different point of the prior's
bulk in NUTS's unconstrained space: site by site, the prior median plus a fixed multiple of
the prior SD there. The multiples are permuted independently for each site, so no chain
starts in the same tail of every site at once, and the R-hat that later compares the chains
compares starts that disagree.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from numpyro.distributions import biject_to
from numpyro.infer import NUTS

# Prior SDs from the unconstrained prior median at which the four chains start.
DISPERSION_OFFSETS = (-1.5, -0.5, 0.5, 1.5)


def chain_offsets(num_chains: int) -> np.ndarray:
    """DISPERSION_OFFSETS for four chains, else num_chains evenly spaced in [-1.5, 1.5]."""
    if num_chains < 2:
        raise ValueError(f"dispersed starts need at least two chains; got {num_chains}")
    if num_chains == len(DISPERSION_OFFSETS):
        return np.array(DISPERSION_OFFSETS)
    return np.linspace(DISPERSION_OFFSETS[0], DISPERSION_OFFSETS[-1], num_chains)


def dispersed_init_params(
    model,
    model_args: tuple,
    prior_draws: dict[str, np.ndarray],
    num_chains: int,
    seed: int,
    *,
    scale: float = 1.0,
) -> tuple[dict[str, jax.Array], dict]:
    """Unconstrained starts, (num_chains, ...) per sampled site, and their JSON record.

    ``prior_draws`` maps every sampled site to constrained draws from its prior, with the
    draws on the leading axis. Chain c starts at median + offset * SD * scale, where the
    median and SD are taken over the draws in unconstrained space and the offsets are a
    seeded per-site permutation of ``chain_offsets(num_chains)``. Raises ValueError unless
    every chain's initial potential energy is finite.
    """
    offsets = chain_offsets(num_chains)
    supports = sample_site_supports(model, model_args, seed)
    if set(prior_draws) != set(supports):
        raise ValueError(
            f"prior_draws must hold exactly the sampled sites {sorted(supports)}; "
            f"got {sorted(prior_draws)}"
        )
    rng = np.random.default_rng(seed)
    init_params, sites = {}, {}
    for name in sorted(supports):
        transform = biject_to(supports[name])
        unconstrained = np.asarray(transform.inv(jnp.asarray(prior_draws[name])))
        median = np.median(unconstrained, axis=0)
        sd = np.std(unconstrained, axis=0)
        # Identical draws give a rounding-error SD, not zero, so test the spread itself.
        spread = np.ptp(unconstrained, axis=0)
        if not np.all(np.isfinite(median) & np.isfinite(sd) & (spread > 0)):
            raise ValueError(
                f"site {name!r} needs finite, not all equal, unconstrained prior draws; "
                f"got median {median} and SD {sd}"
            )
        site_offsets = rng.permutation(offsets)
        chain_axis = site_offsets.reshape(num_chains, *(1,) * median.ndim)
        starts = median + chain_axis * sd * scale
        init_params[name] = jnp.asarray(starts)
        sites[name] = {
            "prior_median_unconstrained": median.tolist(),
            "prior_sd_unconstrained": sd.tolist(),
            "offsets": site_offsets.tolist(),
            "unconstrained": starts.tolist(),
            "constrained": np.asarray(transform(init_params[name])).tolist(),
        }

    started = time.perf_counter()
    energies = initial_potential_energies(model, model_args, init_params, seed)
    readback_seconds = time.perf_counter() - started
    if not np.all(np.isfinite(energies)):
        raise ValueError(
            f"every chain must start at a finite potential energy; got {energies.tolist()}"
        )
    record = {
        "strategy": "dispersed_prior_quantiles",
        "offsets": offsets.tolist(),
        "scale": float(scale),
        "seed": int(seed),
        "num_chains": num_chains,
        "num_prior_draws": {name: len(prior_draws[name]) for name in sorted(supports)},
        "sites": sites,
        "initial_potential_energy": energies.tolist(),
        "readback_seconds": readback_seconds,
    }
    return init_params, record


def sample_site_supports(model, model_args: tuple, seed: int) -> dict:
    """The support of each unobserved sample site, from one seeded trace of the model."""
    trace = handlers.trace(handlers.seed(model, rng_seed=seed)).get_trace(*model_args)
    return {
        name: site["fn"].support
        for name, site in trace.items()
        if site["type"] == "sample" and not site["is_observed"]
    }


def initial_potential_energies(
    model, model_args: tuple, init_params: dict[str, jax.Array], seed: int
) -> np.ndarray:
    """Each chain's potential energy at its start, read back from NUTS's own init.

    A throwaway kernel: a batched ``init`` vmaps the kernel's sample function in place,
    so the kernel that samples must be a fresh one.
    """
    num_chains = len(next(iter(init_params.values())))
    keys = jax.random.split(jax.random.PRNGKey(seed), num_chains)
    state = NUTS(model).init(keys, 0, init_params, model_args, {})
    return np.asarray(state.potential_energy)
