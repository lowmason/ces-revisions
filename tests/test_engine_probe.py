"""The engine probe's synthetic model and its JSON record, at tiny dimensions."""

import importlib.metadata
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta

import jax
import numpy as np

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
    assert (record["nvidia_driver"] is None) == (shutil.which("nvidia-smi") is None)
    assert datetime.fromisoformat(record["timestamp_utc"]).utcoffset() == timedelta(0)
