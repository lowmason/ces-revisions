"""Session numerics policy, applied before any test module performs a JAX operation.

Float64 because Req 17 runs the Kalman recursions in 64-bit JAX; four host devices so the
synthetic pilot's four NUTS chains run in parallel. Both calls silently do nothing once a
JAX operation has run, which is why they live here rather than in a test module.
"""

import numpyro

numpyro.set_host_device_count(4)
numpyro.enable_x64()
