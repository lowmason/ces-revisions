"""NumPyro's chain method follows the devices JAX sees on this host."""

import jax

from ces_revisions.devices import chain_method, has_nvidia_device


def test_chains_run_in_parallel_when_every_chain_has_a_device():
    assert chain_method(jax.local_device_count()) == "parallel"


def test_chains_are_vectorized_when_devices_are_fewer_than_chains():
    assert chain_method(jax.local_device_count() + 1) == "vectorized"


def test_four_chains_run_in_parallel_on_cpu_hosts_and_vectorized_on_one_gpu():
    # tests/conftest.py gives CPU hosts four devices; each GPU size has a single GPU.
    expected = "vectorized" if has_nvidia_device() else "parallel"
    assert chain_method(4) == expected
