"""The pilot CLI's pieces that need no built artifact: XLA flags, Ljung-Box, arguments."""

import json
import os
import subprocess
import sys
from datetime import date

import numpy as np
import pilot_run
import pytest
from scipy.stats import chi2

from ces_revisions.devices import DETERMINISTIC_GPU_FLAG
from ces_revisions.pilot import sbc
from ces_revisions.pilot.dimensions import STAGE_CONFIGURATIONS, stage_dimensions
from ces_revisions.pilot.model import TARGET_ACCEPT_PROB
from ces_revisions.vintages import raw as vintage_raw

SEED = sum(map(ord, "ces-revisions-stage-6-pilot-run"))
COUNT_FLAG = "--xla_force_host_platform_device_count"


@pytest.mark.parametrize(
    ("argv", "nvidia", "before", "after"),
    [
        (["fit"], False, "", f"{COUNT_FLAG}=4"),
        (["prior-predictive", "--out", "x"], False, "", f"{COUNT_FLAG}=4"),
        (["sbc", "--out-dir", "x"], False, "", ""),
        (["compare-seasonal", "fit"], False, "", ""),
        (["fit"], False, f"{COUNT_FLAG}=1", f"{COUNT_FLAG}=1"),
        (["fit"], True, "", DETERMINISTIC_GPU_FLAG),
        (["sbc"], True, DETERMINISTIC_GPU_FLAG, DETERMINISTIC_GPU_FLAG),
    ],
)
def test_xla_flags_follow_the_command_and_host(
    monkeypatch, argv, nvidia, before, after
):
    monkeypatch.setenv("XLA_FLAGS", before)
    monkeypatch.setattr(pilot_run, "has_nvidia_device", lambda: nvidia)
    pilot_run._configure_xla(argv)
    assert os.environ["XLA_FLAGS"] == after


def brute_force_ljung_box(series: np.ndarray, lags: int) -> float:
    n = series.size
    centered = series - series.mean()
    total = 0.0
    for h in range(1, lags + 1):
        r = sum(centered[t] * centered[t - h] for t in range(h, n)) / np.sum(
            centered**2
        )
        total += r**2 / (n - h)
    return n * (n + 2) * total


def test_ljung_box_matches_its_formula_without_gaps():
    series = np.random.default_rng(SEED).standard_normal(300)
    result = pilot_run.ljung_box(series)
    q = brute_force_ljung_box(series, 12)
    assert result["q"] == pytest.approx(q, rel=1e-12)
    assert result["p_value"] == pytest.approx(chi2.sf(q, 12), rel=1e-12)
    assert result["num_observed"] == 300


def gapped_series(rng: np.random.Generator, phi: float) -> np.ndarray:
    """500 months of AR(1) noise with coefficient phi and 40 random gaps."""
    shocks = rng.standard_normal(500)
    series = np.zeros(500)
    for t in range(500):
        series[t] = (phi * series[t - 1] if t else 0.0) + shocks[t]
    series[rng.choice(500, size=40, replace=False)] = np.nan
    return series


def test_ljung_box_with_gaps_has_its_size_and_detects_autocorrelation():
    rng = np.random.default_rng(SEED)
    white = [pilot_run.ljung_box(gapped_series(rng, 0.0)) for _ in range(200)]
    assert all(result["num_observed"] == 460 for result in white)
    rejected = np.mean([result["p_value"] < 0.05 for result in white])
    assert 0.02 <= rejected <= 0.09
    autoregressive = [pilot_run.ljung_box(gapped_series(rng, 0.5)) for _ in range(20)]
    assert all(result["p_value"] < 1e-6 for result in autoregressive)


def test_a_gap_contributes_to_no_lag():
    """Filling every gap with the observed mean leaves each autocorrelation unchanged."""
    series = gapped_series(np.random.default_rng(SEED), 0.3)
    filled = np.where(np.isnan(series), np.nanmean(series), series)
    centered = filled - filled.mean()
    expected = [
        np.sum(centered[h:] * centered[:-h]) / np.sum(centered**2) for h in range(1, 13)
    ]
    np.testing.assert_allclose(
        pilot_run.ljung_box(series)["autocorrelation"], expected, rtol=1e-12
    )


def test_month_arguments_are_first_days():
    assert pilot_run.month("2003-05") == date(2003, 5, 1)


def test_the_cli_starts_in_a_fresh_process():
    result = subprocess.run(
        [sys.executable, "scripts/pilot_run.py", "--help"],
        cwd=vintage_raw.ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    for command in (
        "prior-predictive",
        "fit",
        "sbc",
        "sbc-summary",
        "compare-seasonal",
        "choose-gpu",
    ):
        assert command in result.stdout


def grid_entry(stage, wedge, steps, batches):
    dims = stage_dimensions("dummy", stage, wedge, steps)
    return {**dims, "batches": [{"batch": b, **timing} for b, timing in batches]}


def test_choose_gpu_writes_the_rule_for_the_given_records(tmp_path):
    no_fit = {"error": "skipped: compiled memory exceeds 40e9 bytes"}
    entries = []
    for stage, wedge in STAGE_CONFIGURATIONS:
        fit = [(1, {"median_seconds": 0.2}), (4, no_fit)]
        entries.append(grid_entry(stage, wedge, 572, fit))
        entries.append(grid_entry(stage, wedge, 280, [(1, {"median_seconds": 0.05})]))
    grid = {
        "label": "a100",
        "hourly_price_usd": 1.99,
        "four_gpu_hourly_price_usd": 7.96,
        "entries": entries,
    }
    path = tmp_path / "a100.json"
    path.write_text(json.dumps(grid))
    out = tmp_path / "choice.json"
    parser_arguments = ["choose-gpu", str(path), "--layout", "trig", "--out", str(out)]
    assert pilot_run.main(parser_arguments) == 0
    written = json.loads(out.read_text())
    assert written["layout"] == "trig"  # reads the dummy's entries: same dimensions
    assert written["records"] == [str(path)]
    assert written["status"] == "chosen"
    assert written["steps_per_iteration"] == 127
    assert len(written["stages"]) == len(STAGE_CONFIGURATIONS)
    for result in written["stages"]:
        assert result["choice"]["host"] == "a100"
        assert result["choice"]["gpus"] == 4  # batch 4 did not fit: one GPU per chain
    measured = [*parser_arguments, "--steps-per-iteration", "31"]
    assert pilot_run.main(measured) == 0
    assert json.loads(out.read_text())["steps_per_iteration"] == 31


def test_seasonal_candidates_carry_what_the_rule_reads():
    tests = {
        "F": {"q": 85.0, "p_value": 1e-9},
        "S": {"q": 50.0, "p_value": 1e-5},
        "T": {"q": 9.0, "p_value": 0.7},
    }
    entry = {
        "layout": "trig",
        "dimensions": {"stage9_low": 247},
        "meets_req18": True,
        "april_2020_passed": False,
        "ljung_box": tests,
    }
    cost = stage_dimensions("dummy", 9, "low", 572)
    grid = {"entries": [cost | {"batches": [{"batch": 1, "median_seconds": 2.99}]}]}
    assert pilot_run.seasonal_candidates([entry], grid) == {
        "trig": {
            "states": 247,
            "meets_req18": True,
            "april_2020_passed": False,
            "ljung_box_q": {"F": 85.0, "S": 50.0, "T": 9.0},
            "ljung_box_rejections": 2,
            "seconds": 2.99,
        }
    }


class RecordingPanel:
    """Stands in for the assembled panel: records the window the command asks for."""

    def __init__(self):
        self.windows = []

    def window(self, first, last):
        self.windows.append((first, last))
        return ("windowed", first, last)


def test_sbc_defaults_slice_the_panel_and_pass_the_registered_counts(
    monkeypatch, tmp_path, capsys
):
    panel, calls = RecordingPanel(), []

    def fake_run_sbc(out_dir, **keywords):
        calls.append((out_dir, keywords))
        return {
            "num_replicates": 2,
            "passed": False,
            "failures": {"1": "ValueError('non-finite recovery')"},
            "skipped": 0,
            "quantities": "not printed",
        }

    monkeypatch.setattr(pilot_run, "load_panel", lambda: panel)
    monkeypatch.setattr(pilot_run, "run_sbc", fake_run_sbc)
    assert pilot_run.main(["sbc", "--out-dir", str(tmp_path)]) == 0
    assert panel.windows == [(sbc.FIRST_MONTH, sbc.LAST_MONTH)]
    assert (sbc.FIRST_MONTH, sbc.LAST_MONTH) == (date(1999, 1, 1), date(2021, 12, 1))
    [(out_dir, keywords)] = calls
    assert out_dir == tmp_path
    assert keywords["panel"] == ("windowed", sbc.FIRST_MONTH, sbc.LAST_MONTH)
    assert keywords["start"] == 0
    assert keywords["stop"] == 100
    assert keywords["num_warmup"] == 500
    assert keywords["num_samples"] == 990
    assert keywords["num_kept"] == 99
    assert keywords["target_accept_prob"] == TARGET_ACCEPT_PROB == 0.95
    assert keywords["layout_name"] == "dummy"
    # A failed replicate is reported, not hidden, and does not fail the command.
    printed = json.loads(capsys.readouterr().out)
    assert printed["failures"] == {"1": "ValueError('non-finite recovery')"}
    assert "quantities" not in printed


def test_sbc_stop_defaults_to_the_replicate_count_and_can_be_overridden(
    monkeypatch, tmp_path
):
    calls = []
    monkeypatch.setattr(pilot_run, "load_panel", RecordingPanel)
    monkeypatch.setattr(
        pilot_run, "run_sbc", lambda out_dir, **keywords: calls.append(keywords) or {}
    )
    base = ["sbc", "--out-dir", str(tmp_path)]
    pilot_run.main([*base, "--replicates", "7"])
    pilot_run.main([*base, "--replicates", "7", "--start", "2", "--stop", "4"])
    assert [(call["start"], call["stop"]) for call in calls] == [(0, 7), (2, 4)]


def fake_summary(indices):
    return {
        "num_replicates": len(indices),
        "indices": list(indices),
        "passed": True,
        "ecdf_failures": [],
        "recovery_checks": {"share_s": True, "share_t": True},
        "divergent_replicates": 0,
        "replicate_seconds_median": 1.0,
    }


def test_sbc_summary_names_the_replicates_that_are_missing(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(
        pilot_run, "write_summary", lambda out_dir: fake_summary([0, 2, 3])
    )
    argv = ["sbc-summary", "--out-dir", str(tmp_path), "--replicates", "5"]
    assert pilot_run.main(argv) == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["missing"] == [1, 4]
    assert printed["num_replicates"] == 3


def test_sbc_summary_succeeds_when_every_replicate_is_there(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(
        pilot_run, "write_summary", lambda out_dir: fake_summary(range(5))
    )
    argv = ["sbc-summary", "--out-dir", str(tmp_path), "--replicates", "5"]
    assert pilot_run.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["missing"] == []


def test_sbc_summary_expects_the_registered_hundred_replicates_by_default(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(
        pilot_run, "write_summary", lambda out_dir: fake_summary(range(99))
    )
    assert pilot_run.main(["sbc-summary", "--out-dir", str(tmp_path)]) == 1
    assert json.loads(capsys.readouterr().out)["missing"] == [99]
