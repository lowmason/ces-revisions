"""The model-agnostic run harness, exercised on the Stage 1 synthetic pilot."""

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, date, datetime

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from numpyro.infer import log_likelihood
from numpyro.infer.util import potential_energy
from synthetic_pilot import (
    LAPSE_STEP,
    NUM_STEPS,
    PRIOR_MEDIANS,
    TRUTH,
    pilot_model,
    pilot_ssm,
    simulate_panel,
)

from ces_revisions.devices import chain_method
from ces_revisions.kalman import kalman_filter
from ces_revisions.runs import fit, provenance
from ces_revisions.runs.diagnostics import REQ18, Thresholds, diagnostics_record
from ces_revisions.runs.fit import RunSettings, new_run_dir, run_nuts, write_run
from ces_revisions.runs.inference_data import (
    attach_log_likelihood,
    read_inference_data,
    step_log_likelihood_draws,
    write_inference_data,
)
from ces_revisions.runs.inits import DISPERSION_OFFSETS, dispersed_init_params
from ces_revisions.vintages import raw as vintage_raw

SEED = sum(map(ord, "ces-revisions-stage-6-run-harness"))
NUM_PRIOR_DRAWS = 2000
# The synthetic pilot's priors are LogNormal(log median, 1); see tests/synthetic_pilot.py.
PRIOR_LOG_SD = 1.0
FIRST_MONTH = date(1979, 1, 1)
# Fake DataTree for the diagnostics: four chains of 1000 iid draws pass every Req 18 check.
FAKE_CHAINS, FAKE_DRAWS = 4, 1000
SATURATION_STEPS = 2**REQ18.max_tree_depth - 1
SAMPLE_STAT_CHECKS = ("bfmi", "divergences", "tree_depth")
DRAW_CHECKS = ("rhat", "ess_bulk", "ess_tail")
# A stand-in Stage 5 operator test record, so provenance tests need no built artifact.
OPERATOR_RECORD = {"schema_version": 1, "checks": {"wedge": {"years": 23}}}


@pytest.fixture(scope="module")
def panel():
    return jnp.asarray(simulate_panel(TRUTH, seed=SEED))


@pytest.fixture(scope="module")
def prior_draws():
    rng = np.random.default_rng(SEED)
    return {
        name: np.exp(rng.normal(np.log(median), PRIOR_LOG_SD, NUM_PRIOR_DRAWS))
        for name, median in PRIOR_MEDIANS.items()
    }


@pytest.fixture(scope="module")
def reference_months():
    return tuple(
        date(FIRST_MONTH.year + t // 12, t % 12 + 1, 1) for t in range(NUM_STEPS)
    )


@pytest.fixture(scope="module")
def benchmark_year(reference_months):
    # Req 10's April-to-March benchmark year, computed here so the test needs no pilot code.
    return np.array(
        [
            month.year + 1 if month.month >= 4 else month.year
            for month in reference_months
        ],
        dtype=np.int64,
    )


def copy_source_manifests(root):
    """Copy the committed Stage 3 and Stage 4 source manifests under root."""
    for relative in provenance.SOURCE_MANIFESTS.values():
        (root / relative).parent.mkdir(parents=True)
        (root / relative).write_bytes((vintage_raw.ROOT / relative).read_bytes())


@pytest.fixture
def provenance_root(tmp_path, monkeypatch):
    """A root with the source manifests and a stand-in operator record, but no built
    artifacts; git_state reports the checkout's own state, since the root is no repo."""
    root = tmp_path / "root"
    copy_source_manifests(root)
    record_path = root / provenance.OPERATOR_TEST_RECORD
    record_path.parent.mkdir(parents=True)
    record_path.write_text(json.dumps(OPERATOR_RECORD))
    state = provenance.git_state(vintage_raw.ROOT)
    monkeypatch.setattr(provenance, "git_state", lambda root: state)
    return root


def step_fn_for(panel):
    return lambda params: kalman_filter(pilot_ssm(params), panel).step_log_likelihood


def near_truth_samples(num_chains, num_draws, seed):
    """Posterior-like (chain, draw) draws: the truth times a small lognormal jitter."""
    rng = np.random.default_rng(seed)
    return {
        name: value * np.exp(0.05 * rng.normal(size=(num_chains, num_draws)))
        for name, value in TRUTH.items()
    }


def factor_log_likelihood(panel, samples_by_chain):
    """The model's own factor log likelihood per draw, through NumPyro, not step_fn."""
    num_chains, num_draws = next(iter(samples_by_chain.values())).shape[:2]
    flat = {
        name: jnp.reshape(jnp.asarray(value), (num_chains * num_draws,))
        for name, value in samples_by_chain.items()
    }
    values = np.asarray(log_likelihood(pilot_model, flat, panel)["panel"])
    return values.reshape(num_chains, num_draws)


def fake_idata(
    seed, *, shift_chain=0.0, ar=0.0, random_walk_energy=False, divergent=0, saturated=0
):
    """A DataTree with a scalar and a vector posterior variable and NUTS sample stats."""
    rng = np.random.default_rng(seed)
    shape = (FAKE_CHAINS, FAKE_DRAWS)
    alpha = rng.normal(size=shape)
    if ar:
        for t in range(1, FAKE_DRAWS):
            alpha[:, t] = ar * alpha[:, t - 1] + np.sqrt(1 - ar**2) * alpha[:, t]
    alpha[0] += shift_chain
    energy = rng.normal(size=shape)
    if random_walk_energy:
        energy = energy.cumsum(axis=1)
    diverging = np.zeros(shape, dtype=bool)
    diverging.flat[rng.choice(diverging.size, divergent, replace=False)] = True
    n_steps = np.full(shape, 7, dtype=np.int64)
    n_steps.flat[rng.choice(n_steps.size, saturated, replace=False)] = SATURATION_STEPS
    return az.from_dict(
        {
            "posterior": {"alpha": alpha, "beta": rng.normal(size=(*shape, 2))},
            "sample_stats": {
                "diverging": diverging,
                "energy": energy,
                "n_steps": n_steps,
                "acceptance_rate": np.full(shape, 0.8),
                "step_size": np.full(shape, 0.5),
            },
        }
    )


# ---- dispersed starts ----


def test_dispersed_starts_bracket_the_prior_median_and_differ_per_chain(
    panel, prior_draws
):
    init_params, record = dispersed_init_params(
        pilot_model, (panel,), prior_draws, 4, SEED
    )

    assert set(init_params) == set(PRIOR_MEDIANS)
    for name, draws in prior_draws.items():
        starts = np.asarray(init_params[name])
        median, sd = np.median(np.log(draws)), np.std(np.log(draws))
        assert starts.shape == (4,)
        assert starts.min() < median < starts.max(), name
        assert len(set(starts.tolist())) == 4, name
        np.testing.assert_allclose(
            np.sort((starts - median) / sd), DISPERSION_OFFSETS, rtol=0, atol=1e-12
        )
        site = record["sites"][name]
        np.testing.assert_allclose(site["constrained"], np.exp(starts), rtol=1e-12)
        np.testing.assert_allclose(site["unconstrained"], starts, rtol=0, atol=0)
    # The offsets are permuted per site, so no chain is the extreme start of every site.
    assert len({tuple(record["sites"][name]["offsets"]) for name in PRIOR_MEDIANS}) > 1
    json.dumps(record, allow_nan=False)


def test_initial_potential_energy_readback_matches_numpyro(panel, prior_draws):
    init_params, record = dispersed_init_params(
        pilot_model, (panel,), prior_draws, 4, SEED
    )
    expected = [
        float(
            potential_energy(
                pilot_model,
                (panel,),
                {},
                {name: value[chain] for name, value in init_params.items()},
            )
        )
        for chain in range(4)
    ]

    np.testing.assert_allclose(record["initial_potential_energy"], expected, rtol=1e-10)
    assert record["readback_seconds"] > 0


def test_dispersed_starts_are_seeded(panel, prior_draws):
    first, _ = dispersed_init_params(pilot_model, (panel,), prior_draws, 4, SEED)
    again, _ = dispersed_init_params(pilot_model, (panel,), prior_draws, 4, SEED)
    other, _ = dispersed_init_params(pilot_model, (panel,), prior_draws, 4, SEED + 1)

    for name in PRIOR_MEDIANS:
        np.testing.assert_array_equal(first[name], again[name])
    assert any(not np.array_equal(first[name], other[name]) for name in PRIOR_MEDIANS)


def test_other_chain_counts_use_evenly_spaced_offsets(panel, prior_draws):
    _, record = dispersed_init_params(pilot_model, (panel,), prior_draws, 3, SEED)

    assert record["offsets"] == [-1.5, 0.0, 1.5]
    with pytest.raises(ValueError, match="at least two chains"):
        dispersed_init_params(pilot_model, (panel,), prior_draws, 1, SEED)


def test_a_non_finite_start_raises(panel, prior_draws):
    # A thousand prior SDs out, a scale overflows and the potential energy is not finite.
    with pytest.raises(ValueError, match="finite potential energy"):
        dispersed_init_params(pilot_model, (panel,), prior_draws, 4, SEED, scale=1000.0)


def test_prior_draws_must_cover_exactly_the_sampled_sites(panel, prior_draws):
    missing = {
        name: draws for name, draws in prior_draws.items() if name != "sigma_slope"
    }
    extra = {**prior_draws, "sigma_unused": prior_draws["sigma_slope"]}
    constant = {**prior_draws, "sigma_slope": np.full(NUM_PRIOR_DRAWS, 3.0)}

    for draws in (missing, extra):
        with pytest.raises(ValueError, match="exactly the sampled sites"):
            dispersed_init_params(pilot_model, (panel,), draws, 4, SEED)
    with pytest.raises(ValueError, match="not all equal"):
        dispersed_init_params(pilot_model, (panel,), constant, 4, SEED)


# ---- diagnostics ----


def test_diagnostics_record_passes_iid_draws_and_is_json_safe():
    record = diagnostics_record(fake_idata(SEED))

    assert record["meets_req18"] is True
    assert set(record["checks"]) == {*DRAW_CHECKS, *SAMPLE_STAT_CHECKS}
    assert all(value is True for value in record["checks"].values())
    assert set(record["variables"]) == {"alpha", "beta[0]", "beta[1]"}
    assert record["thresholds"] == {
        "rhat_max": 1.01,
        "ess_min": 400.0,
        "bfmi_min": 0.3,
        "max_tree_depth": 10,
        "saturated_share_max": 0.01,
        "divergences_max": 0,
    }
    assert record["rhat_method"] == "rank"
    assert len(record["bfmi"]) == FAKE_CHAINS
    assert record["divergences"] == {"total": 0, "per_chain": [0] * FAKE_CHAINS}
    assert record["tree_depth"]["saturation_steps"] == SATURATION_STEPS
    assert record["tree_depth"]["n_steps_max"] == 7
    assert record["acceptance_rate"] == pytest.approx([0.8] * FAKE_CHAINS)
    assert record["step_size"] == pytest.approx([0.5] * FAKE_CHAINS)
    for counts in record["rank_histograms"]["counts"].values():
        assert np.shape(counts) == (FAKE_CHAINS, record["rank_histograms"]["bins"])
        assert np.sum(counts, axis=1).tolist() == [FAKE_DRAWS] * FAKE_CHAINS
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize(
    ("failure", "check"),
    [
        ({"divergent": 1}, "divergences"),
        ({"random_walk_energy": True}, "bfmi"),
        ({"saturated": 81}, "tree_depth"),  # 81 of 4000 draws: just over 2%
        ({"saturated": 40}, "tree_depth"),  # exactly 1%: not "under 1%"
    ],
)
def test_one_failing_sample_stat_fails_only_its_check(failure, check):
    record = diagnostics_record(fake_idata(SEED, **failure))

    assert record["meets_req18"] is False
    assert {name for name, passed in record["checks"].items() if not passed} == {check}


@pytest.mark.parametrize(
    ("failure", "checks"),
    [
        ({"shift_chain": 1.0}, {"rhat"}),
        ({"ar": 0.99}, {"ess_bulk", "ess_tail"}),
    ],
)
def test_poorly_mixed_draws_fail_their_checks(failure, checks):
    record = diagnostics_record(fake_idata(SEED, **failure))

    assert record["meets_req18"] is False
    assert all(record["checks"][name] is False for name in checks)
    # R-hat and ESS share the between-chain variance, so they fail together; the sample
    # statistics are untouched.
    assert all(record["checks"][name] is True for name in SAMPLE_STAT_CHECKS)


def test_rank_histogram_shows_a_shifted_chain_in_the_upper_ranks():
    record = diagnostics_record(fake_idata(SEED, shift_chain=1.0))
    counts = np.array(record["rank_histograms"]["counts"]["alpha"])
    half = counts.shape[1] // 2

    assert counts[0, half:].sum() > 3 * counts[0, :half].sum()
    assert counts.sum() == FAKE_CHAINS * FAKE_DRAWS


def test_a_nan_statistic_fails_its_check_and_serializes_as_null():
    idata = fake_idata(SEED)
    idata["posterior"]["alpha"] = xr.full_like(idata["posterior"]["alpha"], 1.0)
    record = diagnostics_record(idata)

    assert record["variables"]["alpha"]["rhat"] is None
    assert record["checks"]["rhat"] is False
    assert record["worst"]["rhat_max"] is None
    json.dumps(record, allow_nan=False)


def test_saturation_is_measured_against_the_given_tree_depth():
    idata = fake_idata(SEED)  # every draw takes 7 steps: depth 3 is saturated
    record = diagnostics_record(idata, Thresholds(max_tree_depth=3))

    assert record["tree_depth"]["saturated_share"] == 1.0
    assert record["checks"]["tree_depth"] is False


def test_diagnostics_record_requires_the_nuts_sample_stats():
    idata = fake_idata(SEED)
    idata["sample_stats"] = xr.DataTree(
        idata["sample_stats"].to_dataset().drop_vars("step_size")
    )
    with pytest.raises(ValueError, match="step_size"):
        diagnostics_record(idata)


# ---- log likelihood blocks and the netCDF file ----


@pytest.fixture(scope="module")
def blocked(panel, reference_months, benchmark_year):
    samples = near_truth_samples(2, 25, SEED)
    idata = az.from_dict({"posterior": samples})
    observed_any = ~np.isnan(np.asarray(panel)).all(axis=1)
    step_ll = step_log_likelihood_draws(step_fn_for(panel), samples, batch_size=7)
    tree = attach_log_likelihood(
        idata, step_ll, reference_months, benchmark_year, observed_any
    )
    return idata, tree, samples, step_ll, observed_any


def test_step_log_likelihood_draws_match_the_filter_draw_by_draw(panel, blocked):
    _, _, samples, step_ll, _ = blocked

    assert step_ll.shape == (2, 25, NUM_STEPS)
    for chain, draw in ((0, 0), (1, 24)):
        params = {name: value[chain, draw] for name, value in samples.items()}
        expected = kalman_filter(pilot_ssm(params), panel).step_log_likelihood
        np.testing.assert_allclose(step_ll[chain, draw], expected, rtol=1e-12, atol=0)


def test_log_likelihood_blocks_have_the_observed_months_and_benchmark_years(
    blocked, reference_months, benchmark_year
):
    idata, tree, _, _, observed_any = blocked
    months = tree["log_likelihood"]["panel_month"]
    years = tree["log_likelihood"]["panel_benchmark_year"]

    assert "log_likelihood" not in idata.children  # the input tree is not modified
    assert months.dims == ("chain", "draw", "reference_month")
    assert years.dims == ("chain", "draw", "benchmark_year")
    assert months.sizes["reference_month"] == NUM_STEPS - 1 == observed_any.sum()
    lapse = np.datetime64(reference_months[LAPSE_STEP], "ns")
    assert lapse not in months["reference_month"].values
    assert years["benchmark_year"].dtype == np.int64
    assert years["benchmark_year"].values.tolist() == sorted(
        set(benchmark_year.tolist())
    )


def test_both_blocks_sum_to_the_factor_log_likelihood_at_every_draw(panel, blocked):
    _, tree, samples, _, _ = blocked
    expected = factor_log_likelihood(panel, samples)
    group = tree["log_likelihood"]

    np.testing.assert_allclose(
        group["panel_month"].sum("reference_month"), expected, rtol=0, atol=1e-8
    )
    np.testing.assert_allclose(
        group["panel_benchmark_year"].sum("benchmark_year"), expected, rtol=0, atol=1e-8
    )


def test_loo_runs_on_both_blocks(blocked):
    _, tree, _, _, _ = blocked

    for name, size in (("panel_month", NUM_STEPS - 1), ("panel_benchmark_year", 11)):
        loo = az.loo(tree, var_name=name)
        assert np.isfinite(loo.elpd), name
        assert len(loo.pareto_k.values) == size, name


def test_a_dropped_step_must_have_zero_log_likelihood(
    blocked, reference_months, benchmark_year
):
    idata, _, _, step_ll, observed_any = blocked
    wrongly_dropped = observed_any.copy()
    wrongly_dropped[0] = False

    with pytest.raises(ValueError, match="nonzero log likelihood"):
        attach_log_likelihood(
            idata, step_ll, reference_months, benchmark_year, wrongly_dropped
        )


def test_inference_data_round_trips_groups_values_dtypes_and_coords(blocked, tmp_path):
    _, tree, _, _, _ = blocked
    tree = tree.copy()
    tree["sample_stats"] = fake_idata(SEED)["sample_stats"].isel(
        chain=slice(2), draw=slice(25)
    )
    path = tmp_path / "inference_data.nc"
    write_inference_data(tree, path)
    back = read_inference_data(path)

    assert back.groups == tree.groups
    for group in tree.groups[1:]:
        original, read = tree[group].to_dataset(), back[group].to_dataset()
        xr.testing.assert_equal(read, original)
        for name in (*original.data_vars, *original.coords):
            assert read[name].dtype == original[name].dtype, (group, name)
    assert back["sample_stats"]["n_steps"].dtype == np.int64
    assert back["log_likelihood"]["benchmark_year"].dtype == np.int64
    assert back["log_likelihood"]["reference_month"].dtype == np.dtype("datetime64[ns]")


# ---- provenance ----


def test_git_state_reads_the_checkout_and_refuses_a_directory_outside_git(tmp_path):
    state = provenance.git_state(vintage_raw.ROOT)
    head = subprocess.run(
        ["git", "-C", str(vintage_raw.ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert state["commit"] == head
    assert len(head) == 40 and set(head) <= set("0123456789abcdef")
    assert isinstance(state["dirty"], bool)
    with pytest.raises(subprocess.CalledProcessError):
        provenance.git_state(tmp_path)


def test_library_versions_and_host_record():
    versions = provenance.library_versions()
    host = provenance.host_record()

    for name in (*provenance.LIBRARIES, "jax", "jaxlib"):
        assert versions[name] == importlib.metadata.version(name), name
    assert versions["python"] == ".".join(map(str, sys.version_info[:3]))
    assert host["jax_backend"] == jax.default_backend()
    assert host["device_count"] == jax.device_count()
    assert set(host) >= {"platform", "machine", "device_kinds", "xla_flags"}


def test_data_inputs_hash_the_source_manifests_and_read_built_manifests(tmp_path):
    root = vintage_raw.ROOT
    record = provenance.data_inputs(root)
    for stage, path in provenance.SOURCE_MANIFESTS.items():
        digest = hashlib.sha256((root / path).read_bytes()).hexdigest()
        assert record["source_manifest_sha256"][stage] == digest
    for stage, path in provenance.ARTIFACT_MANIFESTS.items():
        built = (
            json.loads((root / path).read_text()) if (root / path).is_file() else None
        )
        assert record["artifact_manifests"][stage] == built

    copy_source_manifests(tmp_path)
    unbuilt = provenance.data_inputs(tmp_path)
    assert unbuilt["source_manifest_sha256"] == record["source_manifest_sha256"]
    assert unbuilt["artifact_manifests"] == {"stage3": None, "stage4": None}


def test_operator_test_record_is_the_whole_json_file_and_required(tmp_path):
    path = tmp_path / provenance.OPERATOR_TEST_RECORD
    with pytest.raises(FileNotFoundError, match="operator_artifacts.py build"):
        provenance.operator_test_record(tmp_path)
    written = {"schema_version": 1, "checks": {"wedge": {"years": 23}}}
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(written))

    assert provenance.operator_test_record(tmp_path) == written


def test_array_sha256_is_stable_and_sensitive_to_names_dtypes_shapes_and_values():
    y = np.arange(12.0).reshape(3, 4)
    flags = np.array([True, False, True])
    digest = provenance.array_sha256({"y": y, "flags": flags})

    assert digest == provenance.array_sha256({"flags": flags.copy(), "y": y.copy()})
    assert digest == provenance.array_sha256(
        {"y": np.asfortranarray(y), "flags": flags}
    )
    changed_value = y.copy()
    changed_value[2, 3] += 1e-12
    for variant in (
        {"z": y, "flags": flags},
        {"y": y.astype(np.float32), "flags": flags},
        {"y": y.reshape(4, 3), "flags": flags},
        {"y": changed_value, "flags": flags},
    ):
        assert provenance.array_sha256(variant) != digest
    with pytest.raises(TypeError, match="object dtype"):
        provenance.array_sha256({"months": np.array([date(1979, 1, 1)], dtype=object)})


def test_run_provenance_records_every_section_and_serializes(panel, provenance_root):
    record = provenance.run_provenance(
        root=provenance_root,
        seeds={"run": SEED},
        model={"name": "stage1-synthetic-pilot"},
        init_record={"strategy": "test"},
        arrays={"y": np.asarray(panel)},
        settings={"num_chains": 4},
    )

    assert set(record) == {
        "created_utc",
        "git",
        "libraries",
        "host",
        "data_inputs",
        "operator_test_results",
        "seeds",
        "model",
        "settings",
        "init",
        "arrays",
    }
    assert record["operator_test_results"] == OPERATOR_RECORD
    assert record["git"] == provenance.git_state(vintage_raw.ROOT)
    assert record["data_inputs"]["artifact_manifests"] == {
        "stage3": None,
        "stage4": None,
    }
    assert record["arrays"]["sha256"] == provenance.array_sha256(
        {"y": np.asarray(panel)}
    )
    assert record["arrays"]["shapes"] == {"y": [NUM_STEPS, 4]}
    json.dumps(record, allow_nan=False)


# ---- run directories and the end-to-end run ----


def test_new_run_dir_is_named_by_label_seed_and_time_and_never_reused(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        fit, "_utc_now", lambda: datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
    )
    run_dir = new_run_dir("pilot", 7, runs_dir=tmp_path / "runs")

    assert run_dir == tmp_path / "runs" / "pilot-seed7-20261021T120000Z"
    assert run_dir.is_dir()
    with pytest.raises(FileExistsError):
        new_run_dir("pilot", 7, runs_dir=tmp_path / "runs")


def test_the_harness_imports_no_pilot_module():
    modules = ["diagnostics", "fit", "inference_data", "inits", "provenance"]
    imports = "; ".join(f"import ces_revisions.runs.{name}" for name in modules)
    check = "import sys; print(sorted(m for m in sys.modules if 'ces_revisions.pilot' in m))"
    output = subprocess.run(
        [sys.executable, "-c", f"{imports}; {check}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    assert output.strip() == "[]"


@pytest.mark.slow
def test_run_nuts_writes_a_complete_run_directory(
    panel, prior_draws, reference_months, benchmark_year, provenance_root, tmp_path
):
    settings = RunSettings(seed=SEED, num_warmup=150, num_samples=100)
    observed_any = ~np.isnan(np.asarray(panel)).all(axis=1)
    result = run_nuts(
        pilot_model,
        (panel,),
        settings=settings,
        prior_draws=prior_draws,
        step_fn=step_fn_for(panel),
        reference_months=reference_months,
        benchmark_year=benchmark_year,
        observed_any=observed_any,
        model_record={"name": "stage1-synthetic-pilot", "prior_medians": PRIOR_MEDIANS},
        arrays={"y": np.asarray(panel)},
        root=provenance_root,
    )
    run_dir = write_run(
        result, new_run_dir("synthetic-pilot", SEED, runs_dir=tmp_path / "runs")
    )
    back = read_inference_data(run_dir / fit.INFERENCE_DATA_FILE)
    written_provenance = json.loads((run_dir / fit.PROVENANCE_FILE).read_text())
    sampling = written_provenance["sampling"]
    posterior = result.idata["posterior"]

    assert sorted(path.name for path in run_dir.iterdir()) == sorted(
        [fit.DIAGNOSTICS_FILE, fit.INFERENCE_DATA_FILE, fit.PROVENANCE_FILE]
    )
    assert back.groups == result.idata.groups
    for group in result.idata.groups[1:]:
        xr.testing.assert_equal(
            back[group].to_dataset(), result.idata[group].to_dataset()
        )
    assert (
        json.loads((run_dir / fit.DIAGNOSTICS_FILE).read_text()) == result.diagnostics
    )
    assert set(posterior.data_vars) == set(PRIOR_MEDIANS)
    assert posterior.sizes == {"chain": 4, "draw": 100}
    assert written_provenance["operator_test_results"] == OPERATOR_RECORD
    assert written_provenance["settings"]["chain_method"] == chain_method(4)
    assert written_provenance["settings"]["max_tree_depth"] == 10
    assert len(written_provenance["init"]["initial_potential_energy"]) == 4
    assert sampling["timings"]["sampling_seconds"] == result.sampling_seconds > 0
    (mass,) = sampling["inverse_mass_matrix"]
    assert mass["sites"] == sorted(PRIOR_MEDIANS)
    assert np.shape(mass["per_chain"]) == (4, len(PRIOR_MEDIANS))
    np.testing.assert_allclose(mass["mean"], np.mean(mass["per_chain"], axis=0))
    assert len(sampling["final_step_size"]) == 4
    assert result.diagnostics["thresholds"] == asdict(REQ18)
    samples = {name: np.asarray(posterior[name]) for name in PRIOR_MEDIANS}
    np.testing.assert_allclose(
        result.idata["log_likelihood"]["panel_month"].sum("reference_month"),
        factor_log_likelihood(panel, samples),
        rtol=0,
        atol=1e-8,
    )
