"""The pilot model on the real aggregate panel: structure, prior predictive, determinism.

The panel comes from the session's cached Stage 3, 4, and 5 builds, so these tests need
no gitignored artifact. The fits are short and run at a capped tree depth: they check that
a seed repeats its draws, not Req 18, which the full-length run in scripts/pilot_run.py
checks.
"""

import json
import time
from datetime import date
from functools import cache

import annual_data
import jax
import numpy as np
import operator_data
import pytest
import vintage_data
from numpyro import handlers

from ces_revisions.kalman import kalman_filter
from ces_revisions.pilot.assembly import build_aggregate_panel
from ces_revisions.pilot.dimensions import SEASONAL_DIMS, pilot_dimensions
from ces_revisions.pilot.layout import LAYOUTS
from ces_revisions.pilot.model import (
    SAMPLED_SITES,
    pilot_model,
    pilot_ssm,
    sample_prior,
    step_log_likelihood,
)
from ces_revisions.pilot.panel import ANCHOR_COLUMN, ANCHOR_YEARS, AggregatePanel
from ces_revisions.pilot.simulate import monthly_constraint, prior_predictive
from ces_revisions.runs import provenance
from ces_revisions.runs.fit import RunSettings, run_nuts
from ces_revisions.vintages import raw as vintage_raw

SEED = sum(map(ord, "ces-revisions-stage-6-pilot-fit-test"))
PRIOR_PREDICTIVE_SEED = sum(map(ord, "ces-revisions-stage-6-prior-predictive"))
NUM_PRIOR_PREDICTIVE_DRAWS = 1000
# Shorter than Req 18's run; depth 5 bounds each draw at 31 leapfrog steps.
SHORT = {"num_chains": 2, "num_warmup": 100, "num_samples": 100, "max_tree_depth": 5}


@cache
def real_panel() -> AggregatePanel:
    return build_aggregate_panel(
        stage_changes=vintage_data.stage_changes(),
        stage_labels=vintage_data.labels(),
        table_changes=vintage_data.table_changes(),
        release_index=vintage_data.index(),
        benchmark_fixtures=operator_data.artifact("benchmark_fixtures"),
        publication_calendar=annual_data.artifact("publication_calendar"),
    )


def prior_medians(layout_name: str) -> dict[str, float]:
    draws = sample_prior(jax.random.PRNGKey(SEED), 2000, LAYOUTS[layout_name])
    return {name: float(np.median(values)) for name, values in draws.items()}


def test_seasonal_dimensions_agree_with_the_layouts():
    for name, layout in LAYOUTS.items():
        assert SEASONAL_DIMS[name] == layout.seasonal_dim
        assert pilot_dimensions(name)["states"] == layout.state_dim


def test_the_likelihood_sees_irregular_anchor_rows_and_missing_cells():
    panel = real_panel()
    anchored = ~np.isnan(panel.y[:, ANCHOR_COLUMN])
    anchor_months = [
        month
        for month, kept in zip(panel.reference_months, anchored, strict=True)
        if kept
    ]
    assert anchor_months == [date(year, 3, 1) for year in ANCHOR_YEARS]
    assert len(anchor_months) == 23
    closing = panel.y[:, :ANCHOR_COLUMN]
    assert np.isnan(closing).sum(axis=0).tolist() == [1, 49, 50]
    # Every month has some observed cell, so every step adds to the likelihood.
    assert panel.observed_any().all()
    steps = np.asarray(
        step_log_likelihood(prior_medians("dummy"), panel, LAYOUTS["dummy"])
    )
    assert steps.shape == (panel.num_steps,)
    assert np.isfinite(steps).all()
    assert (steps != 0.0).all()


@pytest.mark.parametrize("layout_name", sorted(LAYOUTS))
def test_the_model_samples_only_the_scalar_sites_and_no_state(layout_name):
    panel, layout = real_panel(), LAYOUTS[layout_name]
    trace = handlers.trace(handlers.seed(pilot_model, rng_seed=SEED)).get_trace(
        panel, layout
    )
    latent = [
        name
        for name, site in trace.items()
        if site["type"] == "sample" and not site["is_observed"]
    ]
    assert tuple(latent) == SAMPLED_SITES
    assert all(np.shape(trace[name]["value"]) == () for name in latent)
    others = {name: site for name, site in trace.items() if name not in latent}
    assert list(others) == ["panel"]
    params = {name: trace[name]["value"] for name in latent}
    expected = kalman_filter(pilot_ssm(params, panel, layout), panel.y).log_likelihood
    factor = others["panel"]["fn"].log_prob(others["panel"]["value"])
    np.testing.assert_allclose(factor, expected, rtol=1e-12)


def test_the_prior_predictive_meets_the_monthly_constraint_on_the_real_panel():
    started = time.perf_counter()
    result = prior_predictive(
        real_panel(),
        LAYOUTS["dummy"],
        NUM_PRIOR_PREDICTIVE_DRAWS,
        PRIOR_PREDICTIVE_SEED,
    )
    seconds = time.perf_counter() - started
    constraint = monthly_constraint(result)
    assert constraint["passed"], constraint
    assert seconds < 60.0


@pytest.fixture
def provenance_root(tmp_path, monkeypatch):
    """The source manifests and a stand-in operator record; the checkout's git state."""
    root = tmp_path / "root"
    for relative in provenance.SOURCE_MANIFESTS.values():
        (root / relative).parent.mkdir(parents=True)
        (root / relative).write_bytes((vintage_raw.ROOT / relative).read_bytes())
    record = root / provenance.OPERATOR_TEST_RECORD
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"schema_version": 1}))
    state = provenance.git_state(vintage_raw.ROOT)
    monkeypatch.setattr(provenance, "git_state", lambda root: state)
    return root


@pytest.mark.slow
def test_a_seed_repeats_its_draws_on_the_real_panel(provenance_root):
    panel, layout = real_panel(), LAYOUTS["dummy"]
    prior_draws = {
        name: np.asarray(values)
        for name, values in sample_prior(jax.random.PRNGKey(SEED), 2000, layout).items()
    }

    def short_fit():
        return run_nuts(
            pilot_model,
            (panel, layout),
            settings=RunSettings(seed=SEED, **SHORT),
            prior_draws=prior_draws,
            step_fn=lambda params: step_log_likelihood(params, panel, layout),
            reference_months=panel.reference_months,
            benchmark_year=panel.benchmark_year,
            observed_any=panel.observed_any(),
            model_record={"name": "stage6-pilot", "layout": layout.name},
            arrays={"y": panel.y},
            root=provenance_root,
        )

    first, second = short_fit(), short_fit()
    for name in SAMPLED_SITES:
        a, b = (np.asarray(fit.idata["posterior"][name]) for fit in (first, second))
        assert a.shape == (SHORT["num_chains"], SHORT["num_samples"])
        assert np.array_equal(a, b), name
    for name in ("panel_month", "panel_benchmark_year"):
        assert np.array_equal(
            first.idata["log_likelihood"][name], second.idata["log_likelihood"][name]
        )
    assert first.provenance["init"] == {
        **second.provenance["init"],
        "readback_seconds": first.provenance["init"]["readback_seconds"],
    }
