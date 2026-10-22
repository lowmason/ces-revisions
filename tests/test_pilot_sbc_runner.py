"""The SBC replicate runner, its resumable fan-out, and the pre-registered summary."""

import json
import math

import numpy as np
import pytest
from pilot_synthetic import synthetic_panel

from ces_revisions.pilot import sbc
from ces_revisions.pilot.model import SAMPLED_SITES
from ces_revisions.pilot.sbc import (
    DERIVED_QUANTITIES,
    LOG_LIKELIHOOD,
    QUANTITIES,
    prior_reference,
    read_replicates,
    replicate_path,
    replicate_seeds,
    run_replicate,
    run_sbc,
    summarize_sbc,
)

SEED = sum(map(ord, "ces-revisions-stage-6-sbc-runner"))
NUM_KEPT = 19
NUM_REPLICATES = 200
# A short NUTS run: enough to exercise every step, not to converge.
TINY = {"num_warmup": 40, "num_samples": 30, "num_kept": 10}


def fake_records(
    rng: np.random.Generator,
    *,
    contraction: float = 0.6,
    z_sd: float = 1.0,
    point_mass: str | None = None,
) -> list[dict]:
    """Replicate records with uniform ranks, except ``point_mass``'s, all at rank 0."""
    settings = {"layout": "dummy", "num_kept": NUM_KEPT}
    records = []
    for index in range(NUM_REPLICATES):
        ranks = {
            name: 0 if name == point_mass else int(rng.integers(0, NUM_KEPT + 1))
            for name in QUANTITIES
        }
        records.append(
            {
                "index": index,
                "settings": settings,
                "ranks": ranks,
                "recovery": {
                    name: {
                        "z": float(z_sd * rng.standard_normal()),
                        "contraction": contraction,
                    }
                    for name in (*SAMPLED_SITES, *DERIVED_QUANTITIES)
                },
                "ess_bulk": dict.fromkeys((*SAMPLED_SITES, *DERIVED_QUANTITIES), 500.0),
                "ess_bulk_kept": dict.fromkeys(QUANTITIES, 19.0),
                "divergences": int(index % 50 == 0),
                "n_steps": {"mean": 15.0, "max": 31, "saturated_share": 0.0},
                "seconds": {"sampling": 1.0, "total": 2.0},
            }
        )
    return records


def test_quantities_are_the_sites_the_derived_quantities_and_the_log_likelihood():
    assert QUANTITIES == (*SAMPLED_SITES, *DERIVED_QUANTITIES, LOG_LIKELIHOOD)
    assert len(QUANTITIES) == 23
    assert "sigma_news_post" not in QUANTITIES  # sigma_news_annual's exact rescaling


def test_replicate_seeds_are_deterministic_and_distinct():
    first = replicate_seeds(SEED, 3)
    assert first == replicate_seeds(SEED, 3)
    assert len(set(first.values())) == len(first)
    assert set(first.values()).isdisjoint(replicate_seeds(SEED, 4).values())
    assert set(first.values()).isdisjoint(replicate_seeds(SEED + 1, 3).values())


def test_prior_reference_matches_the_priors():
    reference = prior_reference("dummy")
    median, sd = reference["median"], reference["sd"]
    # HalfNormal(s): median s * 0.6745, SD s * sqrt(1 - 2 / pi); LogNormal medians.
    halfnormal_median = 0.6744897501960817
    assert median["sigma_level"] == pytest.approx(150 * halfnormal_median, rel=0.03)
    assert sd["sigma_level"] == pytest.approx(
        150 * math.sqrt(1 - 2 / math.pi), rel=0.03
    )
    assert median["sigma_news_s"] == pytest.approx(40.0, rel=0.02)
    assert median["kappa_crisis"] == pytest.approx(math.log(50.0), abs=0.02)
    assert median["rho_noise"] == pytest.approx(0.0, abs=0.01)
    assert sd["shift_mean"] == pytest.approx(15.0, rel=0.03)
    assert set(sd) == {*SAMPLED_SITES, *DERIVED_QUANTITIES}
    assert all(value > 0 for value in sd.values())


def test_summary_passes_uniform_ranks_with_recovered_news_shares():
    summary = summarize_sbc(fake_records(np.random.default_rng(SEED)))
    assert summary["num_replicates"] == NUM_REPLICATES
    assert summary["num_quantities"] == len(QUANTITIES)
    assert summary["alpha"] == pytest.approx(0.05 / len(QUANTITIES))
    assert summary["ecdf_failures"] == []
    assert summary["recovery_checks"] == {"share_s": True, "share_t": True}
    assert summary["divergent_replicates"] == 4
    assert summary["passed"] is True
    assert "recovery" not in summary["quantities"][LOG_LIKELIHOOD]
    json.dumps(summary, allow_nan=False)


def test_summary_fails_a_point_mass_rank():
    summary = summarize_sbc(
        fake_records(np.random.default_rng(SEED), point_mass="sigma_slope")
    )
    assert summary["ecdf_failures"] == ["sigma_slope"]
    assert summary["passed"] is False


@pytest.mark.parametrize(
    ("contraction", "z_sd"),
    [(0.2, 1.0), (0.6, 2.0)],
    ids=["weak-contraction", "poor-coverage"],
)
def test_summary_fails_unrecovered_news_shares(contraction, z_sd):
    summary = summarize_sbc(
        fake_records(np.random.default_rng(SEED), contraction=contraction, z_sd=z_sd)
    )
    assert summary["recovery_checks"] == {"share_s": False, "share_t": False}
    assert summary["passed"] is False


def test_summary_refuses_mixed_settings():
    records = fake_records(np.random.default_rng(SEED))
    records[1] = {**records[1], "settings": {"layout": "trig4", "num_kept": NUM_KEPT}}
    with pytest.raises(ValueError, match="different settings"):
        summarize_sbc(records)


def test_single_device_flags_replace_any_host_device_count():
    flags = sbc._single_device_flags(
        "--xla_force_host_platform_device_count=4 --xla_gpu_deterministic_ops=true"
    )
    assert flags.split() == [
        "--xla_gpu_deterministic_ops=true",
        "--xla_force_host_platform_device_count=1",
    ]
    assert sbc._single_device_flags(None) == "--xla_force_host_platform_device_count=1"


def test_run_sbc_skips_replicates_already_written(tmp_path, monkeypatch):
    """With every replicate file present, no worker starts and the files are summarized."""
    records = fake_records(np.random.default_rng(SEED))[:3]
    for record in records:
        replicate_path(tmp_path, record["index"]).write_text(json.dumps(record))

    def no_pool(*args, **kwargs):
        raise AssertionError("no replicate should run")

    monkeypatch.setattr(sbc, "ProcessPoolExecutor", no_pool)
    summary = run_sbc(
        tmp_path,
        layout_name="dummy",
        panel=None,
        start=0,
        stop=3,
        workers=2,
        base_seed=SEED,
        **TINY,
    )
    assert summary["skipped"] == 3
    assert summary["indices"] == [0, 1, 2]
    assert json.loads((tmp_path / sbc.SUMMARY_FILE).read_text())["indices"] == [0, 1, 2]
    assert [record["index"] for record in read_replicates(tmp_path)] == [0, 1, 2]


def test_run_sbc_refuses_several_workers_on_a_gpu_host(tmp_path, monkeypatch):
    def no_pool(*args, **kwargs):
        raise AssertionError("no worker should start")

    monkeypatch.setattr(sbc, "has_nvidia_device", lambda: True)
    monkeypatch.setattr(sbc, "ProcessPoolExecutor", no_pool)
    with pytest.raises(ValueError, match="share one GPU"):
        run_sbc(
            tmp_path,
            layout_name="dummy",
            panel=None,
            start=0,
            stop=2,
            workers=2,
            base_seed=SEED,
            **TINY,
        )


@pytest.mark.slow
def test_run_replicate_records_ranks_recovery_and_is_seeded():
    panel = synthetic_panel("dummy")
    record = run_replicate(2, base_seed=SEED, layout_name="dummy", panel=panel, **TINY)
    json.dumps(record, allow_nan=False)
    assert set(record["ranks"]) == set(QUANTITIES)
    assert all(0 <= rank <= TINY["num_kept"] for rank in record["ranks"].values())
    assert set(record["recovery"]) == {*SAMPLED_SITES, *DERIVED_QUANTITIES}
    assert record["seeds"] == replicate_seeds(SEED, 2)
    assert record["settings"]["num_kept"] == TINY["num_kept"]
    again = run_replicate(2, base_seed=SEED, layout_name="dummy", panel=panel, **TINY)
    for key in ("truth", "ranks", "recovery", "divergences", "n_steps", "step_size"):
        assert again[key] == record[key], key


@pytest.mark.slow
def test_run_sbc_spawns_workers_and_resumes(tmp_path):
    panel = synthetic_panel("dummy")
    arguments = {
        "layout_name": "dummy",
        "panel": panel,
        "workers": 2,
        "base_seed": SEED,
        **TINY,
    }
    first = run_sbc(tmp_path, start=0, stop=2, **arguments)
    assert first["failures"] == {}
    assert first["skipped"] == 0
    assert first["indices"] == [0, 1]
    written = {index: replicate_path(tmp_path, index).read_text() for index in (0, 1)}
    resumed = run_sbc(tmp_path, start=0, stop=3, **arguments)
    assert resumed["skipped"] == 2
    assert resumed["indices"] == [0, 1, 2]
    assert {
        index: replicate_path(tmp_path, index).read_text() for index in (0, 1)
    } == written
    # The worker's replicate equals the same replicate run in this process.
    local = run_replicate(2, base_seed=SEED, layout_name="dummy", panel=panel, **TINY)
    remote = json.loads(replicate_path(tmp_path, 2).read_text())
    assert remote["ranks"] == local["ranks"]
    assert remote["truth"] == pytest.approx(local["truth"], rel=1e-12)
