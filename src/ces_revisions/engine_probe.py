"""Time the Kalman engine's value and gradient on this host.

Run as ``python -m ces_revisions.engine_probe``. The probe builds a synthetic float64
model at the requested dimensions and times ``jax.value_and_grad`` of the filter's log
likelihood with respect to two log scales, one multiplying every transition covariance
and one every observation covariance, vmapped over a batch of parameter draws. Each run
writes one JSON record, so runs on the Mac and on each cloud size compare field by field.
"""

import argparse
import importlib.metadata
import json
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro

from ces_revisions.devices import has_nvidia_device
from ces_revisions.kalman import LinearGaussianSSM, kalman_filter

# Each transition matrix is this multiple of a random orthogonal matrix, so every
# eigenvalue has exactly this modulus and the state process is stable.
TRANSITION_RADIUS = 0.95
# Standard deviation of the log-scale draws around the scales the model was built at.
LOG_SCALE_SD = 0.1


def synthetic_problem(
    num_steps: int, state_dim: int, obs_dim: int, missing_share: float, seed: int
) -> tuple[LinearGaussianSSM, np.ndarray]:
    """A time-varying float64 model and a panel simulated from it, with NaN cells.

    Each time-indexed array varies by step, every covariance is positive definite, and
    exactly ``round(missing_share * num_steps * obs_dim)`` panel cells are NaN.
    """
    rng = np.random.default_rng(seed)
    orthogonal, _ = np.linalg.qr(rng.normal(size=(num_steps, state_dim, state_dim)))
    ssm = LinearGaussianSSM(
        initial_mean=np.zeros(state_dim),
        initial_cov=np.eye(state_dim),
        transition_matrix=TRANSITION_RADIUS * orthogonal,
        transition_offset=rng.normal(size=(num_steps, state_dim)),
        transition_cov=_random_spd(rng, num_steps, state_dim),
        observation_matrix=rng.normal(size=(num_steps, obs_dim, state_dim))
        / np.sqrt(state_dim),
        observation_offset=rng.normal(size=(num_steps, obs_dim)),
        observation_cov=_random_spd(rng, num_steps, obs_dim),
    )
    panel = _simulate(ssm, rng)
    missing = rng.choice(
        panel.size, size=round(missing_share * panel.size), replace=False
    )
    panel.flat[missing] = np.nan
    return ssm, panel


def time_batches(
    ssm: LinearGaussianSSM,
    panel: np.ndarray,
    batch_sizes: list[int],
    repeats: int,
    seed: int,
) -> list[dict[str, float]]:
    """Seconds per call of the vmapped value and gradient, one entry per batch size.

    A batch size's first call traces and compiles, and is reported as its compile time;
    the median and interquartile range cover the ``repeats`` calls after it. Each call
    pulls its result to the host, so a GPU's asynchronous dispatch cannot hide work.
    """
    ssm = LinearGaussianSSM(*(jnp.asarray(array) for array in ssm))
    panel = jnp.asarray(panel)
    value_and_grad = jax.jit(
        jax.vmap(jax.value_and_grad(_log_likelihood), in_axes=(0, None, None))
    )
    rng = np.random.default_rng(seed)
    timings = []
    for batch in batch_sizes:
        log_scales = jnp.asarray(rng.normal(scale=LOG_SCALE_SD, size=(batch, 2)))
        compile_seconds = _call_seconds(value_and_grad, log_scales, ssm, panel)
        seconds = [
            _call_seconds(value_and_grad, log_scales, ssm, panel)
            for _ in range(repeats)
        ]
        lower, median, upper = np.percentile(seconds, [25, 50, 75])
        timings.append(
            {
                "batch": batch,
                "compile_seconds": compile_seconds,
                "median_seconds": float(median),
                "iqr_seconds": float(upper - lower),
            }
        )
    return timings


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, time the engine, and write the JSON record."""
    args = _parse_args(argv)
    numpyro.enable_x64()  # before the first JAX operation: the engine needs float64
    started = datetime.now(UTC)
    ssm, panel = synthetic_problem(
        args.steps, args.states, args.cells, args.missing_share, args.seed
    )
    record = {
        "label": args.label,
        "dimensions": {"steps": args.steps, "states": args.states, "cells": args.cells},
        "missing_share": args.missing_share,
        "seed": args.seed,
        "repeats": args.repeats,
        "batches": time_batches(ssm, panel, args.batch, args.repeats, args.seed),
        "backend": jax.default_backend(),
        "versions": _jax_versions(),
        "nvidia_driver": _nvidia_driver(),
        "timestamp_utc": started.isoformat(timespec="seconds"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m ces_revisions.engine_probe",
        description="Time the Kalman engine's value and gradient on this host.",
    )
    parser.add_argument("--steps", type=int, required=True, help="time steps T")
    parser.add_argument("--states", type=int, required=True, help="state dimension n")
    parser.add_argument("--cells", type=int, required=True, help="cells p per step")
    parser.add_argument(
        "--batch", type=int, nargs="+", required=True, help="batch sizes to time"
    )
    parser.add_argument(
        "--missing-share", type=float, default=0.2, help="share of NaN panel cells"
    )
    parser.add_argument(
        "--repeats", type=int, default=10, help="timed calls per batch after compiling"
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for model and draws")
    parser.add_argument("--label", required=True, help="host label, e.g. mac or l4")
    parser.add_argument("--out", type=Path, required=True, help="JSON record to write")
    return parser.parse_args(argv)


def _random_spd(rng: np.random.Generator, num_steps: int, dim: int) -> np.ndarray:
    root = rng.normal(size=(num_steps, dim, dim))
    return root @ root.transpose(0, 2, 1) / dim + 0.5 * np.eye(dim)


def _simulate(ssm: LinearGaussianSSM, rng: np.random.Generator) -> np.ndarray:
    state = rng.multivariate_normal(ssm.initial_mean, ssm.initial_cov)
    rows = []
    for a, c, q, z, d, r in zip(
        ssm.transition_matrix,
        ssm.transition_offset,
        ssm.transition_cov,
        ssm.observation_matrix,
        ssm.observation_offset,
        ssm.observation_cov,
        strict=True,
    ):
        state = a @ state + c + np.linalg.cholesky(q) @ rng.normal(size=c.shape)
        rows.append(z @ state + d + np.linalg.cholesky(r) @ rng.normal(size=d.shape))
    return np.stack(rows)


def _log_likelihood(
    log_scales: jax.Array, ssm: LinearGaussianSSM, panel: jax.Array
) -> jax.Array:
    scaled = ssm._replace(
        transition_cov=jnp.exp(log_scales[0]) * ssm.transition_cov,
        observation_cov=jnp.exp(log_scales[1]) * ssm.observation_cov,
    )
    return kalman_filter(scaled, panel).log_likelihood


def _call_seconds(function: Callable[..., object], *args: object) -> float:
    start = time.perf_counter()
    # Pulling the result to the host waits for the device to finish computing it.
    jax.tree.map(np.asarray, function(*args))
    return time.perf_counter() - start


def _jax_versions() -> dict[str, str]:
    """Versions of jax, jaxlib, and any installed JAX CUDA plugin."""
    versions = {}
    for dist in importlib.metadata.distributions():
        name = dist.name.lower().replace("_", "-")
        if name in {"jax", "jaxlib"} or name.startswith("jax-cuda"):
            versions[name] = dist.version
    return dict(sorted(versions.items()))


def _nvidia_driver() -> str | None:
    """The NVIDIA driver version, or None on a host without an NVIDIA device."""
    # A CPU size can have the driver's nvidia-smi, which fails there without a GPU.
    if not has_nvidia_device():
        return None
    query = ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    output = subprocess.run(query, capture_output=True, text=True, check=True).stdout
    return output.splitlines()[0].strip()


if __name__ == "__main__":
    main()
