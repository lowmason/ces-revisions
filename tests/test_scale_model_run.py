"""The scale model's command line: series names, evidence paths, and the sensitivity rule."""

import argparse
import json

import arviz as az
import pytest
import scale_model_run
from test_scale_model import small_data

from ces_revisions.runs.fit import RunSettings
from ces_revisions.scale_model.fit import fit_series
from ces_revisions.scale_model.model import Priors


def test_series_parse_from_status_and_transition():
    assert scale_model_run.parse_series("NSA:F_S") == ("NSA", "F_S")
    with pytest.raises(argparse.ArgumentTypeError, match="series must be one of"):
        scale_model_run.parse_series("NSA:S_T")


def test_evidence_is_filed_by_series_and_beta_prior(tmp_path):
    path = scale_model_run.evidence_path(tmp_path, ("SA", "F_T"), 0.2)
    assert path == tmp_path / "sa-f_t" / "fit-beta0.2.json"


def write_evidence(directory, series, beta_sd, mean):
    path = scale_model_run.evidence_path(directory, series, beta_sd)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"estimates": {"beta": {"mean": mean, "sd": 0.03}}}))


def test_sensitivity_command_applies_the_rule_to_each_series(tmp_path, capsys):
    for beta_sd, mean in ((0.05, -0.02), (0.1, -0.04), (0.2, -0.045)):
        write_evidence(tmp_path, ("NSA", "F_S"), beta_sd, mean)
    code = scale_model_run.main(["sensitivity", "--evidence-dir", str(tmp_path)])
    record = json.loads((tmp_path / "sensitivity.json").read_text())
    assert code == 0
    assert set(record) == {"NSA:F_S"}
    assert record["NSA:F_S"]["shift_0.1_to_0.2"] == pytest.approx(0.005)
    assert '"passed": true' in capsys.readouterr().out


def test_sensitivity_command_fails_when_a_series_moves_too_far(tmp_path):
    for beta_sd, mean in ((0.1, -0.04), (0.2, -0.07)):
        write_evidence(tmp_path, ("SA", "F_S"), beta_sd, mean)
    code = scale_model_run.main(["sensitivity", "--evidence-dir", str(tmp_path)])
    assert code == 1


@pytest.mark.slow
def test_figures_leave_the_global_interval_settings_alone(tmp_path):
    before = (az.rcParams["stats.ci_prob"], az.rcParams["stats.ci_kind"])
    fit = fit_series(
        small_data(),
        Priors(),
        RunSettings(seed=13, num_warmup=60, num_samples=40, target_accept_prob=0.95),
    )
    scale_model_run.write_figures_and_psense(fit.result.idata, tmp_path)
    assert (az.rcParams["stats.ci_prob"], az.rcParams["stats.ci_kind"]) == before
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "trace.png").exists()


def test_sbc_summary_with_a_missing_replicate_leaves_the_evidence_alone(
    tmp_path, monkeypatch
):
    out_dir, evidence_dir = tmp_path / "sbc", tmp_path / "evidence"
    out_dir.mkdir()
    (out_dir / "summary.json").write_text("{}")
    monkeypatch.setattr(
        scale_model_run,
        "write_summary",
        lambda directory: {"indices": [0, 1], "num_replicates": 2},
    )
    code = scale_model_run.main(
        [
            "sbc-summary",
            "--out-dir",
            str(out_dir),
            "--replicates",
            "3",
            "--evidence-dir",
            str(evidence_dir),
        ]
    )
    assert code == 1
    assert not (evidence_dir / "sbc-summary.json").exists()
