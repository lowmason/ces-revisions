"""The pilot's posterior news shares: pooled draws, their summary, and the two commands."""

import hashlib
import json

import numpy as np
import pilot_run
import pytest
import xarray as xr

from ces_revisions.pilot.model import SAMPLED_SITES, derived
from ces_revisions.pilot.news_shares import (
    SHARES,
    share_draws,
    shares_block,
    summarize,
)
from ces_revisions.runs.fit import INFERENCE_DATA_FILE, PROVENANCE_FILE
from ces_revisions.runs.inference_data import write_inference_data
from ces_revisions.runs.provenance import array_sha256
from ces_revisions.vintages import raw as vintage_raw

RECORD = {
    "run_dir": "data/runs/example-run",
    "inference_data_sha256": "b" * 64,
    "panel_sha256": "c" * 64,
    "num_draws": 4000,
    "interval_prob": 0.9,
    "sampling_seconds": 1659.6018,
    "share_s": {"median": 0.7234, "lower": 0.6, "upper": 0.81},
    "share_t": {"median": 0.652, "lower": 0.3, "upper": 0.8466},
}
EMPTY_FINDINGS = (
    "# Findings\n\n"
    "<!-- BEGIN GENERATED stage-2-shares -->\n"
    "<!-- END GENERATED stage-2-shares -->\n"
)


def chain_sites(chain: int) -> dict[str, float]:
    """One chain's constant sampled sites: only the news SDs differ between chains."""
    sites = dict.fromkeys(SAMPLED_SITES, 0.5)
    sites["sigma_news_s"] = 1.0 + chain
    sites["sigma_news_t"] = 2.0 + 0.5 * chain
    return sites


def test_share_draws_pool_every_chain_and_draw():
    chains = [chain_sites(chain) for chain in range(4)]
    posterior = xr.Dataset(
        {
            name: (("chain", "draw"), np.array([[sites[name]] * 3 for sites in chains]))
            for name in SAMPLED_SITES
        }
    )
    draws = share_draws(posterior)
    assert set(draws) == set(SHARES)
    for key, values in draws.items():
        assert values.shape == (12,)
        assert np.unique(np.round(values, 10)).size == 4
        expected = [derived(sites)[key] for sites in chains]
        np.testing.assert_allclose(np.unique(values), np.sort(expected), rtol=1e-12)


def test_summary_is_the_median_and_central_interval():
    assert summarize(np.linspace(0, 1, 101)) == pytest.approx(
        {"median": 0.5, "lower": 0.05, "upper": 0.95}
    )


def test_the_block_names_the_revision_each_share_splits():
    lines = shares_block(RECORD).splitlines()
    assert "| F→S | `share_s` | 0.723 | 0.600–0.810 |" in lines
    assert any(line.startswith("| S→T | `share_t` |") for line in lines)


def test_shares_refuse_a_run_fit_to_other_data(tmp_path):
    (tmp_path / "provenance.json").write_text(
        json.dumps({"arrays": {"sha256": "f" * 64}})
    )
    with pytest.raises(ValueError, match="other data"):
        pilot_run.shares_record(tmp_path, "0" * 64)


def test_shares_name_the_refit_when_the_run_is_gone(tmp_path, capsys):
    status = pilot_run.main(
        ["shares", str(tmp_path / "missing"), "--out", str(tmp_path / "x.json")]
    )
    assert status == 2
    assert "pilot_run.py fit --layout dummy --label stage6-pilot" in (
        capsys.readouterr().err
    )


def test_shares_report_needs_only_the_record(tmp_path):
    record = tmp_path / "news-shares.json"
    findings = tmp_path / "findings.md"
    record.write_text(json.dumps(RECORD))
    findings.write_text(EMPTY_FINDINGS)
    status = pilot_run.main(["shares-report", str(record), "--findings", str(findings)])
    assert status == 0
    assert "| F→S | `share_s` | 0.723 | 0.600–0.810 |" in findings.read_text()


# --- The shapes the plan's tests leave: other intervals, a whole run, and the commands ---

PANEL_ARRAYS = {"y": np.arange(3.0)}
PANEL_SHA256 = array_sha256(PANEL_ARRAYS)
SAMPLING_SECONDS = 12.5
NUM_CHAINS, NUM_DRAWS = 4, 3


def write_run(run_dir, panel_sha256=PANEL_SHA256):
    """A small run directory, a constant posterior per chain, and return its posterior."""
    run_dir.mkdir()
    posterior = xr.Dataset(
        {
            name: (
                ("chain", "draw"),
                np.array(
                    [
                        [chain_sites(chain)[name]] * NUM_DRAWS
                        for chain in range(NUM_CHAINS)
                    ]
                ),
            )
            for name in SAMPLED_SITES
        }
    )
    write_inference_data(
        xr.DataTree.from_dict({"posterior": posterior}), run_dir / INFERENCE_DATA_FILE
    )
    provenance = {
        "arrays": {"sha256": panel_sha256},
        "sampling": {"timings": {"sampling_seconds": SAMPLING_SECONDS}},
    }
    (run_dir / PROVENANCE_FILE).write_text(json.dumps(provenance))
    return posterior


def use_panel(monkeypatch):
    """Stand in for the built panel, which hermetic tests do not have."""
    monkeypatch.setattr(pilot_run, "load_panel", lambda: None)
    monkeypatch.setattr(pilot_run, "panel_arrays", lambda panel: PANEL_ARRAYS)


def refuse_panel(monkeypatch):
    def load_panel():
        raise AssertionError("the command loaded the panel")

    monkeypatch.setattr(pilot_run, "load_panel", load_panel)


def test_summary_interval_follows_the_probability_and_the_median_is_not_the_mean():
    assert summarize(np.array([0.0, 1.0, 2.0, 3.0, 100.0]), prob=0.5) == (
        pytest.approx({"median": 2.0, "lower": 1.0, "upper": 3.0})
    )


def test_the_block_states_the_interval_the_draws_and_the_run():
    lines = shares_block(RECORD).splitlines()
    assert lines[0] == "| Revision | Share | Median | 90% interval |"
    assert "| S→T | `share_t` | 0.652 | 0.300–0.847 |" in lines
    note = lines[-1]
    assert "4000" in note
    assert "`data/runs/example-run`" in note
    assert "1659.6 seconds" in note
    assert "80% interval" in shares_block(RECORD | {"interval_prob": 0.8})


def test_shares_record_summarizes_the_posterior_of_a_run_fit_to_the_panel(tmp_path):
    run_dir = tmp_path / "run"
    posterior = write_run(run_dir)
    record = pilot_run.shares_record(run_dir, PANEL_SHA256)
    draws = share_draws(posterior)
    digest = hashlib.sha256((run_dir / INFERENCE_DATA_FILE).read_bytes())
    assert record["run_dir"] == str(run_dir)
    assert record["inference_data_sha256"] == digest.hexdigest()
    assert record["panel_sha256"] == PANEL_SHA256
    assert record["num_draws"] == NUM_CHAINS * NUM_DRAWS
    assert record["interval_prob"] == 0.9
    assert record["sampling_seconds"] == SAMPLING_SECONDS
    for key in SHARES:
        assert record[key] == pytest.approx(summarize(draws[key]))


def test_shares_write_sorted_json_for_the_current_panel(tmp_path, monkeypatch):
    use_panel(monkeypatch)
    run_dir, out = tmp_path / "run", tmp_path / "out" / "news-shares.json"
    write_run(run_dir)
    assert pilot_run.main(["shares", str(run_dir), "--out", str(out)]) == 0
    record = pilot_run.shares_record(run_dir, PANEL_SHA256)
    assert out.read_text() == json.dumps(record, indent=2, sort_keys=True) + "\n"


def test_the_shares_command_refuses_a_run_fit_to_other_data(tmp_path, monkeypatch):
    use_panel(monkeypatch)
    run_dir, out = tmp_path / "run", tmp_path / "news-shares.json"
    write_run(run_dir, panel_sha256="f" * 64)
    with pytest.raises(ValueError, match="other data"):
        pilot_run.main(["shares", str(run_dir), "--out", str(out)])
    assert not out.exists()


def test_shares_stop_on_a_missing_run_before_loading_the_panel(tmp_path, monkeypatch):
    refuse_panel(monkeypatch)
    missing, out = tmp_path / "missing", tmp_path / "x.json"
    assert pilot_run.main(["shares", str(missing), "--out", str(out)]) == 2


def test_shares_report_replaces_a_stale_block_and_is_repeatable(tmp_path):
    record, findings = tmp_path / "news-shares.json", tmp_path / "findings.md"
    record.write_text(json.dumps(RECORD))
    findings.write_text(
        "intro\n"
        "<!-- BEGIN GENERATED stage-2-shares -->\nstale\n"
        "<!-- END GENERATED stage-2-shares -->\n"
        "outro\n"
    )
    argv = ["shares-report", str(record), "--findings", str(findings)]
    assert pilot_run.main(argv) == 0
    first = findings.read_text()
    assert first.startswith(
        "intro\n<!-- BEGIN GENERATED stage-2-shares -->\n\n| Revision | Share |"
    )
    assert first.endswith("\n\n<!-- END GENERATED stage-2-shares -->\noutro\n")
    assert "stale" not in first
    assert pilot_run.main(argv) == 0
    assert findings.read_text() == first


def test_shares_report_never_loads_the_panel(tmp_path, monkeypatch):
    refuse_panel(monkeypatch)
    record, findings = tmp_path / "news-shares.json", tmp_path / "findings.md"
    record.write_text(json.dumps(RECORD))
    findings.write_text(EMPTY_FINDINGS)
    argv = ["shares-report", str(record), "--findings", str(findings)]
    assert pilot_run.main(argv) == 0


def test_shares_report_defaults_to_the_stage_2_findings_file(tmp_path, monkeypatch):
    record, findings = tmp_path / "news-shares.json", tmp_path / "findings.md"
    record.write_text(json.dumps(RECORD))
    findings.write_text(EMPTY_FINDINGS)
    monkeypatch.setattr(pilot_run, "FINDINGS", findings)
    assert pilot_run.main(["shares-report", str(record)]) == 0
    assert "| F→S | `share_s` | 0.723 | 0.600–0.810 |" in findings.read_text()


def test_the_default_findings_file_is_the_stage_2_findings():
    stage_2_findings = "specs/ces-revision-drivers-stage-2-findings.md"
    assert pilot_run.FINDINGS == vintage_raw.ROOT / stage_2_findings
