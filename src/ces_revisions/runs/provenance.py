"""What a run needs recorded to be repeated: code, data, libraries, host, and arrays.

The record names the commit and whether the working tree differed from it, the source
manifests and built-artifact manifests the run's inputs came from, the operator test record
that every run copies (scripts/operator_artifacts.py writes it), the exact library and host
it ran on, and a hash of the arrays the model saw.
"""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import jax
import numpy as np

from ces_revisions.devices import jax_versions, nvidia_driver
from ces_revisions.vintages import raw as vintage_raw

LIBRARIES = (
    "numpyro",
    "arviz",
    "arviz-base",
    "arviz-stats",
    "xarray",
    "numpy",
    "polars",
    "h5netcdf",
    "h5py",
)
SOURCE_MANIFESTS = {
    "stage3": Path("data/raw/manifest.csv"),
    "stage4": Path("data/annual/raw/manifest.csv"),
}
ARTIFACT_MANIFESTS = {
    "stage3": Path("data/panel/manifest.json"),
    "stage4": Path("data/annual/panel/manifest.json"),
}
OPERATOR_TEST_RECORD = Path("data/operators/operator-test-results.json")


def git_state(root: Path) -> dict:
    """The checked-out commit and whether tracked or untracked files differ from it."""
    commit = _git(root, "rev-parse", "HEAD").strip()
    changes = _git(root, "status", "--porcelain", "--untracked-files=normal")
    return {"commit": commit, "dirty": bool(changes.strip())}


def library_versions() -> dict:
    """Python, JAX (with any CUDA plugin), and the inference, data, and storage stack."""
    versions = {"python": platform.python_version(), **jax_versions()}
    versions |= {name: importlib.metadata.version(name) for name in LIBRARIES}
    return dict(sorted(versions.items()))


def host_record() -> dict:
    """The platform, JAX's backend and devices, the NVIDIA driver, and XLA_FLAGS."""
    devices = jax.devices()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_executable": sys.executable,
        "jax_backend": jax.default_backend(),
        "device_kinds": sorted({device.device_kind for device in devices}),
        "device_count": len(devices),
        "nvidia_driver": nvidia_driver(),
        "xla_flags": os.environ.get("XLA_FLAGS"),
    }


def data_inputs(root: Path) -> dict:
    """SHA-256 of the committed source manifests, and the built-artifact manifests if any."""
    return {
        "source_manifest_sha256": {
            stage: vintage_raw.file_sha256(root / path)
            for stage, path in SOURCE_MANIFESTS.items()
        },
        "artifact_manifests": {
            stage: json.loads((root / path).read_text())
            if (root / path).is_file()
            else None
            for stage, path in ARTIFACT_MANIFESTS.items()
        },
    }


def operator_test_record(root: Path) -> dict:
    """The whole Stage 5 operator test record; every run copies it into its provenance."""
    path = root / OPERATOR_TEST_RECORD
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; rebuild it with "
            "`uv run python scripts/operator_artifacts.py build`"
        )
    return json.loads(path.read_text())


def array_sha256(arrays: dict[str, np.ndarray]) -> str:
    """One SHA-256 over the arrays' sorted names, dtypes, shapes, and C-order bytes."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        if array.dtype.hasobject:
            raise TypeError(
                f"array {name!r} has object dtype; its bytes are not its values"
            )
        header = json.dumps([name, array.dtype.str, list(array.shape)])
        digest.update(header.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def run_provenance(
    *,
    root: Path,
    seeds: dict,
    model: dict,
    init_record: dict,
    arrays: dict[str, np.ndarray],
    settings: dict,
) -> dict:
    """The run's provenance record; JSON-serializable when its arguments are."""
    return {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_state(root),
        "libraries": library_versions(),
        "host": host_record(),
        "data_inputs": data_inputs(root),
        "operator_test_results": operator_test_record(root),
        "seeds": seeds,
        "model": model,
        "settings": settings,
        "init": init_record,
        "arrays": {
            "sha256": array_sha256(arrays),
            "shapes": {name: list(np.shape(arrays[name])) for name in sorted(arrays)},
            "dtypes": {
                name: np.asarray(arrays[name]).dtype.str for name in sorted(arrays)
            },
        },
    }


def _git(root: Path, *args: str) -> str:
    command = ["git", "-C", str(root), *args]
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout
