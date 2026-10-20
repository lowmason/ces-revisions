"""NumPyro's chain method follows the devices JAX sees on this host."""

import jax
import pytest

from ces_revisions.devices import chain_method, has_nvidia_device


def test_chains_run_in_parallel_when_every_chain_has_a_device():
    assert chain_method(jax.local_device_count()) == "parallel"


def test_chains_are_vectorized_when_devices_are_fewer_than_chains():
    assert chain_method(jax.local_device_count() + 1) == "vectorized"


@pytest.mark.skipif(
    has_nvidia_device(), reason="a GPU host has as many devices as the model's GPUs"
)
def test_four_chains_run_in_parallel_on_cpu_hosts():
    # tests/conftest.py gives CPU hosts four devices, one per chain of the synthetic pilot.
    assert chain_method(4) == "parallel"
