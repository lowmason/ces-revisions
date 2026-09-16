"""Device facts, so one checkout runs on the Mac's CPU and on a cloud GPU host."""

from pathlib import Path

import jax

NVIDIA_DEVICE = Path("/dev/nvidia0")
# XLA's switch for run-to-run determinism on GPU, named as in the XLA revision that
# jaxlib 0.11.1 pins (openxla/xla dcf304bc, xla/debug_options_flags.cc).
DETERMINISTIC_GPU_FLAG = "--xla_gpu_deterministic_ops=true"


def has_nvidia_device() -> bool:
    """Whether the host exposes an NVIDIA GPU, whether or not JAX can use it."""
    return NVIDIA_DEVICE.exists()


def chain_method(num_chains: int) -> str:
    """NumPyro's chain method: parallel with a device per chain, otherwise vectorized."""
    return "parallel" if jax.local_device_count() >= num_chains else "vectorized"
