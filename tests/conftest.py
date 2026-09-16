"""Session numerics policy, applied before any test module performs a JAX operation.

Float64 because Req 17 runs the Kalman recursions in 64-bit JAX; four host devices so the
synthetic pilot's four NUTS chains run in parallel on a CPU host; and, on an NVIDIA host,
XLA's deterministic GPU operations, so a fixed seed repeats its draws. All three settings
silently do nothing once a JAX operation has run, which is why they live here rather than
in a test module. The GPU flag goes into XLA_FLAGS first because set_host_device_count
keeps the flags it finds there.
"""

import os

import numpyro

from ces_revisions.devices import DETERMINISTIC_GPU_FLAG, has_nvidia_device

if has_nvidia_device():
    flags = os.environ.get("XLA_FLAGS", "").split()
    os.environ["XLA_FLAGS"] = " ".join([*flags, DETERMINISTIC_GPU_FLAG])
numpyro.set_host_device_count(4)
numpyro.enable_x64()
