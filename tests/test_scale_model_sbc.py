"""SBC of the scale model: the replicate record and the pre-registered pass rule."""

import numpy as np
import pytest
from test_scale_model import small_data

from ces_revisions.scale_model.sbc import (
    BETA_CONTRACTION_MIN,
    QUANTITIES,
    run_replicate,
    summarize_sbc,
)

# Simulations for the critical value in these tests; the summary command uses 100,000.
FAST = 2_000


def fake_record(index: int, *, beta_z: float = 0.0) -> dict:
    """A replicate whose ranks, over 100 replicates, cover 0..99 exactly once each."""
    return {
        "index": index,
        "settings": {"num_kept": 99, "num_months": 258},
        "ranks": {
            name: (37 * index + 11 * position) % 100
            for position, name in enumerate(QUANTITIES)
        },
        "recovery": {
            "beta": {"z": beta_z, "contraction": 0.8},
        },
        "divergences": 0,
        "n_steps": {"mean": 20.0, "saturated_share": 0.0},
        "seconds": {"total": 5.0},
    }


def test_uniform_ranks_and_good_recovery_pass():
    summary = summarize_sbc(
        [fake_record(index) for index in range(100)], num_simulations=FAST
    )
    assert summary["passed"]
    assert summary["ecdf_failures"] == []
    assert summary["num_quantities"] == len(QUANTITIES)
    assert summary["alpha"] == pytest.approx(0.05 / len(QUANTITIES))


def test_ranks_piled_at_one_end_fail_that_quantity():
    rng = np.random.default_rng(1)
    records = [fake_record(index) for index in range(100)]
    for record in records:
        record["ranks"]["beta"] = int(rng.integers(0, 10))
    summary = summarize_sbc(records, num_simulations=FAST)
    assert not summary["passed"]
    assert summary["ecdf_failures"] == ["beta"]


def test_large_beta_z_scores_fail_the_recovery_check():
    records = [
        fake_record(index, beta_z=4.0 if index < 10 else 0.0) for index in range(100)
    ]
    summary = summarize_sbc(records, num_simulations=FAST)
    assert not summary["beta_recovery"]["passed"]
    assert not summary["passed"]


def test_weak_contraction_fails_the_recovery_check():
    records = [fake_record(index) for index in range(100)]
    for record in records:
        record["recovery"]["beta"]["contraction"] = BETA_CONTRACTION_MIN / 2
    assert not summarize_sbc(records, num_simulations=FAST)["passed"]


def test_records_with_different_settings_are_refused():
    records = [fake_record(index) for index in range(3)]
    records[1]["settings"] = {"num_kept": 49, "num_months": 258}
    with pytest.raises(ValueError, match="settings"):
        summarize_sbc(records, num_simulations=FAST)


@pytest.mark.slow
def test_a_replicate_ranks_every_quantity_within_the_kept_draws():
    record = run_replicate(
        3, base_seed=17, data=small_data(), num_warmup=50, num_samples=40, num_kept=20
    )
    assert set(record["ranks"]) == set(QUANTITIES)
    assert all(0 <= rank <= 20 for rank in record["ranks"].values())
    assert set(record["truth"]) == set(QUANTITIES)
    assert np.isfinite(record["recovery"]["beta"]["z"])
    again = run_replicate(
        3, base_seed=17, data=small_data(), num_warmup=50, num_samples=40, num_kept=20
    )
    assert again["ranks"] == record["ranks"]
