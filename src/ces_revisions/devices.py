"""Device facts, so one checkout runs on the Mac's CPU and on a rented GPU host."""

import importlib.metadata
import shutil
import subprocess
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


def jax_versions() -> dict[str, str]:
    """Versions of jax, jaxlib, and any installed JAX CUDA plugin."""
    versions = {}
    for dist in importlib.metadata.distributions():
        name = dist.name.lower().replace("_", "-")
        if name in {"jax", "jaxlib"} or name.startswith("jax-cuda"):
            versions[name] = dist.version
    return dict(sorted(versions.items()))


def nvidia_driver() -> str | None:
    """The NVIDIA driver version, or None on a host without nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return None
    query = ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    output = subprocess.run(query, capture_output=True, text=True, check=True).stdout
    return output.splitlines()[0].strip()
