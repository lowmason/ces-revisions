"""The pilot's posterior news shares: pooled draws, their summary, and the two commands."""

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
