"""The pinned stack imports, and the session runs float64 JAX on the host's devices."""

import importlib
import os
import sys

import jax
import jax.numpy as jnp
import pytest

from ces_revisions.devices import DETERMINISTIC_GPU_FLAG, has_nvidia_device

# One importable module per pinned distribution: runtime jax, numpy, numpyro, arviz,
# polars, and fastexcel, the Excel reader behind Stage 3's workbooks; dynamax from the dev
# group, as engine-determination evidence only.
STACK_MODULES = [
    "jax",
    "numpy",
    "numpyro",
    "arviz",
    "polars",
    "fastexcel",
    "dynamax.linear_gaussian_ssm",
]


def test_interpreter_is_python_3_14_or_newer():
    assert sys.version_info >= (3, 14)


@pytest.mark.parametrize("module", STACK_MODULES)
def test_stack_module_imports(module):
    importlib.import_module(module)


def test_session_runs_float64_jax_on_the_expected_devices():
    assert jnp.zeros(1).dtype == jnp.float64
    if has_nvidia_device():  # a GPU size: JAX runs on the GPU, deterministically
        assert jax.default_backend() == "gpu"
        assert DETERMINISTIC_GPU_FLAG in os.environ["XLA_FLAGS"].split()
    else:  # the Mac or dev: the four host devices from tests/conftest.py
        assert jax.local_device_count() == 4
