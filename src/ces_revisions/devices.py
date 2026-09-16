"""Device facts, so one checkout runs on the Mac's CPU and on a cloud GPU host."""

from pathlib import Path

import jax

NVIDIA_DEVICE = Path("/dev/nvidia0")


def has_nvidia_device() -> bool:
    """Whether the host exposes an NVIDIA GPU, whether or not JAX can use it."""
    return NVIDIA_DEVICE.exists()


def chain_method(num_chains: int) -> str:
    """NumPyro's chain method: parallel with a device per chain, otherwise vectorized."""
    return "parallel" if jax.local_device_count() >= num_chains else "vectorized"
