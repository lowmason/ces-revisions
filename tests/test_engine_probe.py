"""The engine probe's synthetic model and its JSON record, at tiny dimensions."""

import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta

import jax
import numpy as np

from ces_revisions import engine_probe
from ces_revisions.devices import has_nvidia_device
from ces_revisions.engine_probe import synthetic_problem

NUM_STEPS, STATE_DIM, OBS_DIM = 6, 3, 4
MISSING_SHARE = 0.25


def test_synthetic_problem_is_float64_stable_positive_definite_and_time_varying():
    ssm, panel = synthetic_problem(NUM_STEPS, STATE_DIM, OBS_DIM, MISSING_SHARE, seed=0)

    assert panel.shape == (NUM_STEPS, OBS_DIM)
    assert all(np.asarray(array).dtype == np.float64 for array in (*ssm, panel))
    assert np.abs(np.linalg.eigvals(ssm.transition_matrix)).max() < 1
    for cov in (ssm.initial_cov[None], ssm.transition_cov, ssm.observation_cov):
        assert np.linalg.eigvalsh(cov).min() > 0
    for name in ("transition_matrix", "transition_cov", "observation_matrix"):
        steps = getattr(ssm, name)
        assert not np.allclose(steps[0], steps[1]), name


def test_synthetic_problem_sets_the_requested_share_of_cells_to_nan():
    _, panel = synthetic_problem(NUM_STEPS, STATE_DIM, OBS_DIM, MISSING_SHARE, seed=0)

    assert np.isnan(panel).sum() == round(MISSING_SHARE * NUM_STEPS * OBS_DIM)


def test_probe_module_writes_a_timing_record(tmp_path):
    out = tmp_path / "probe.json"
    command = [sys.executable, "-m", "ces_revisions.engine_probe"]
    command += ["--steps", str(NUM_STEPS), "--states", str(STATE_DIM)]
    command += ["--cells", str(OBS_DIM), "--batch", "1", "2", "--repeats", "3"]
    command += ["--label", "test", "--out", str(out)]
    subprocess.run(command, check=True)
    record = json.loads(out.read_text())

    assert record["label"] == "test"
    assert record["dimensions"] == {
        "steps": NUM_STEPS,
        "states": STATE_DIM,
        "cells": OBS_DIM,
    }
    assert [entry["batch"] for entry in record["batches"]] == [1, 2]
    for entry in record["batches"]:
        assert entry["compile_seconds"] > 0
        assert entry["median_seconds"] > 0
        assert entry["iqr_seconds"] >= 0
    assert record["backend"] == jax.default_backend()
    for name in ("jax", "jaxlib"):
        assert record["versions"][name] == importlib.metadata.version(name)
    assert (record["nvidia_driver"] is None) == (not has_nvidia_device())
    assert datetime.fromisoformat(record["timestamp_utc"]).utcoffset() == timedelta(0)


def _stub_nvidia_smi(monkeypatch, tmp_path, body):
    stub = tmp_path / "nvidia-smi"
    stub.write_text(f"#!/bin/sh\n{body}\n")
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")


def test_the_driver_is_not_asked_for_without_an_nvidia_device(monkeypatch, tmp_path):
    # A CPU size can have the driver's nvidia-smi, which fails there without a GPU.
    _stub_nvidia_smi(monkeypatch, tmp_path, "exit 9")
    monkeypatch.setattr(engine_probe, "has_nvidia_device", lambda: False)

    assert engine_probe._nvidia_driver() is None


def test_the_driver_version_comes_from_nvidia_smi_on_an_nvidia_host(
    monkeypatch, tmp_path
):
    _stub_nvidia_smi(monkeypatch, tmp_path, "echo 580.173.02")
    monkeypatch.setattr(engine_probe, "has_nvidia_device", lambda: True)

    assert engine_probe._nvidia_driver() == "580.173.02"
