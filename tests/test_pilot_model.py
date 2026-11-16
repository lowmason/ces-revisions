"""Tests of the pilot model: its arrays, the dense oracle, its trace, and its flags."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from dense_reference import dense_reference
from numpyro import handlers
from numpyro.infer.util import log_density
from pilot_synthetic import (
    ANCHOR_OFFSETS,
    CARRIER_STEPS,
    CRISIS_STEPS,
    MISSING_STEP,
    STATE_NAMES,
    TRUTH,
    reference_ssm,
    synthetic_panel,
)

from ces_revisions.kalman import (
    FilterResult,
    LinearGaussianSSM,
    SmootherResult,
    kalman_filter,
    kalman_smoother,
)
from ces_revisions.pilot.layout import LAYOUTS, SEASONAL_PRIOR_SCALE
from ces_revisions.pilot.model import (
    INITIAL_MEAN,
    INITIAL_SD,
    SAMPLED_SITES,
    derived,
    pilot_model,
    pilot_ssm,
    prior_distributions,
    sample_prior,
    step_log_likelihood,
)
from ces_revisions.pilot.panel import ANCHOR_COLUMN

SEED = sum(map(ord, "ces-revisions-stage-6-pilot-model"))
LAYOUT_NAMES = sorted(LAYOUTS)
REAL_MAGNITUDE_STEPS = 60
NUM_PRIOR_DRAWS = 3
# Every field must agree with the dense oracle to 1e-8 of its natural scale: a
# covariance entry to 1e-8 sqrt(C_ii C_jj) (rtol 1e-8 on the diagonal, and an exact zero
# held to the same scale), a mean to 1e-8 (|m_i| + sqrt(C_ii)), and the log likelihood
# per step to rtol 1e-8 with a 1e-8 nat floor for the empty step.
DENSE_TOLERANCE = 1e-8
MEAN_COV_FIELDS = {
    "predicted_mean": "predicted_cov",
    "filtered_mean": "filtered_cov",
    "one_step_mean": "one_step_cov",
    "smoothed_mean": "smoothed_cov",
}
# pilot_ssm and the entry-by-entry reference agree to round-off of each array's scale.
ARRAY_TOLERANCE = 1e-12
# D2: the crisis multiplier scales the level shock and the S and T news, not post-T news.
SCALED_BLOCK = ("level", "news_s", "news_t")


@pytest.fixture(params=LAYOUT_NAMES)
def layout_name(request):
    return request.param


def _prior_draws(layout_name, num_draws=NUM_PRIOR_DRAWS):
    draws = sample_prior(jax.random.PRNGKey(SEED), num_draws, LAYOUTS[layout_name])
    return [
        {name: float(value[k]) for name, value in draws.items()}
        for k in range(num_draws)
    ]


def _engine_moments(params, panel, layout_name):
    ssm = pilot_ssm(params, panel, LAYOUTS[layout_name])
    filtered = kalman_filter(ssm, jnp.asarray(panel.y))
    smoothed = kalman_smoother(ssm, filtered)
    return {
        **{name: np.asarray(value) for name, value in filtered._asdict().items()},
        **{name: np.asarray(value) for name, value in smoothed._asdict().items()},
    }


def _assert_matches_dense(engine, dense):
    assert set(engine) == {*FilterResult._fields, *SmootherResult._fields}
    np.testing.assert_allclose(
        engine["log_likelihood"], dense["log_likelihood"], rtol=DENSE_TOLERANCE
    )
    np.testing.assert_allclose(
        engine["step_log_likelihood"],
        dense["step_log_likelihood"],
        rtol=DENSE_TOLERANCE,
        atol=DENSE_TOLERANCE,
    )
    for mean_field, cov_field in MEAN_COV_FIELDS.items():
        variance = np.diagonal(dense[cov_field], axis1=1, axis2=2)
        assert (variance > 0).all(), cov_field
        cov_scale = np.sqrt(variance[:, :, None] * variance[:, None, :])
        cov_error = np.abs(engine[cov_field] - dense[cov_field]) / cov_scale
        mean_scale = np.abs(dense[mean_field]) + np.sqrt(variance)
        mean_error = np.abs(engine[mean_field] - dense[mean_field]) / mean_scale
        assert cov_error.max() <= DENSE_TOLERANCE, (cov_field, cov_error.max())
        assert mean_error.max() <= DENSE_TOLERANCE, (mean_field, mean_error.max())


def test_synthetic_panel_has_every_feature_the_pilot_must_handle():
    panel = synthetic_panel()
    anchored = ~np.isnan(panel.y[:, ANCHOR_COLUMN])
    closing = panel.y[:, :ANCHOR_COLUMN]

    assert panel.pre_2003.any() and not panel.pre_2003.all()
    assert np.flatnonzero(panel.crisis).tolist() == list(CRISIS_STEPS)
    assert anchored.sum() == 3
    np.testing.assert_array_equal(panel.anchor_offset[anchored], ANCHOR_OFFSETS[:3])
    assert not panel.anchor_offset[~anchored].any()
    assert np.isnan(panel.y[MISSING_STEP]).all()
    for step in CARRIER_STEPS:
        assert not np.isnan(closing[step, 0]) and np.isnan(closing[step, 1:]).all()
    assert np.isnan(closing[-1, 1:]).all() and np.isnan(closing[-2, 2])
    assert np.nanmin(panel.y[:, ANCHOR_COLUMN]) > 80_000.0  # real magnitude levels


def test_pilot_arrays_match_the_entry_by_entry_reference(layout_name):
    panel = synthetic_panel(layout_name)
    for params in [TRUTH, *_prior_draws(layout_name)]:
        ssm = pilot_ssm(params, panel, LAYOUTS[layout_name])
        reference = reference_ssm(params, panel, layout_name)
        assert set(reference) == set(LinearGaussianSSM._fields)
        for field, expected in reference.items():
            np.testing.assert_allclose(
                getattr(ssm, field),
                expected,
                rtol=ARRAY_TOLERANCE,
                atol=ARRAY_TOLERANCE * np.abs(expected).max(),
                err_msg=field,
            )


def test_filter_and_smoother_match_the_dense_reference(layout_name):
    panel = synthetic_panel(layout_name)
    engine = _engine_moments(TRUTH, panel, layout_name)
    dense = dense_reference(reference_ssm(TRUTH, panel, layout_name), panel.y)

    _assert_matches_dense(engine, dense)


def test_real_magnitude_filter_and_smoother_match_the_dense_reference(layout_name):
    panel = synthetic_panel(layout_name, REAL_MAGNITUDE_STEPS)
    closing = panel.y[:, :ANCHOR_COLUMN]

    assert np.nanmin(panel.y[:, ANCHOR_COLUMN]) > 80_000.0
    assert np.nanmax(closing) - np.nanmin(closing) > 1_000.0  # seasonal swings
    assert (~np.isnan(panel.y[:, ANCHOR_COLUMN])).sum() == len(ANCHOR_OFFSETS)
    for params in [TRUTH, *_prior_draws(layout_name)]:
        engine = _engine_moments(params, panel, layout_name)
        dense = dense_reference(reference_ssm(params, panel, layout_name), panel.y)
        _assert_matches_dense(engine, dense)


def test_model_samples_exactly_the_scalar_sites_and_factors_the_filter(layout_name):
    panel = synthetic_panel(layout_name)
    layout = LAYOUTS[layout_name]
    seeded = handlers.seed(pilot_model, jax.random.PRNGKey(SEED))
    trace = handlers.trace(seeded).get_trace(panel, layout)
    sampled = [
        name
        for name, site in trace.items()
        if site["type"] == "sample" and not site["is_observed"]
    ]
    observed = [
        name
        for name, site in trace.items()
        if site["type"] == "sample" and site["is_observed"]
    ]
    params = {name: trace[name]["value"] for name in sampled}

    assert tuple(sampled) == SAMPLED_SITES
    assert len(SAMPLED_SITES) == 17
    assert all(np.shape(value) == () for value in params.values())
    assert observed == ["panel"]
    assert {site["type"] for site in trace.values()} == {"sample"}  # no deterministic
    assert isinstance(trace["panel"]["fn"], dist.Unit)
    np.testing.assert_allclose(
        trace["panel"]["fn"].log_factor,
        kalman_filter(
            pilot_ssm(params, panel, layout), jnp.asarray(panel.y)
        ).log_likelihood,
        rtol=1e-12,
    )


def _expected_priors(layout_name):
    return {
        "sigma_level": (dist.HalfNormal, {"scale": 150.0}),
        "sigma_slope": (dist.HalfNormal, {"scale": 15.0}),
        "sigma_seasonal": (
            dist.HalfNormal,
            {"scale": SEASONAL_PRIOR_SCALE[layout_name]},
        ),
        "sigma_news_s": (dist.LogNormal, {"loc": math.log(40.0), "scale": 0.40}),
        "sigma_news_t": (dist.LogNormal, {"loc": math.log(35.0), "scale": 0.45}),
        "sigma_news_annual": (dist.LogNormal, {"loc": math.log(300.0), "scale": 0.60}),
        "noise_f_raw": (dist.Normal, {"loc": 0.0, "scale": 1.0}),
        "noise_s_raw": (dist.Normal, {"loc": 0.0, "scale": 1.0}),
        "noise_t_raw": (dist.Normal, {"loc": 0.0, "scale": 1.0}),
        "mean_news_s": (dist.Normal, {"loc": 0.0, "scale": 15.0}),
        "mean_news_t": (dist.Normal, {"loc": 0.0, "scale": 15.0}),
        "mean_news_annual": (dist.Normal, {"loc": 0.0, "scale": 100.0}),
        "sigma_anchor": (dist.HalfNormal, {"scale": 100.0}),
        "kappa_crisis": (dist.Normal, {"loc": math.log(50.0), "scale": 0.5}),
        "shift_log_scale": (dist.Normal, {"loc": 0.0, "scale": 0.5}),
        "shift_mean": (dist.Normal, {"loc": 0.0, "scale": 15.0}),
    }


def test_priors_are_the_contract_table(layout_name):
    priors = prior_distributions(LAYOUTS[layout_name])
    rho = priors["rho_noise"]

    assert tuple(priors) == SAMPLED_SITES
    for name, (kind, parameters) in _expected_priors(layout_name).items():
        assert type(priors[name]) is kind, name
        for parameter, value in parameters.items():
            assert float(getattr(priors[name], parameter)) == pytest.approx(value), name
    assert isinstance(rho.base_dist, dist.Normal)
    assert (float(rho.base_dist.loc), float(rho.base_dist.scale)) == (0.0, 0.2)
    assert (float(rho.low), float(rho.high)) == (-0.8, 0.8)


def test_initial_state_covers_every_layout_with_the_contract_values():
    for names in STATE_NAMES.values():
        assert set(names) <= set(INITIAL_MEAN) == set(INITIAL_SD)
    expected = {"level": (89_408.0, 5_000.0), "slope": (150.0, 300.0)}
    expected |= {"lagged_value": (89_408.0, 5_000.0)}
    expected |= {name: (0.0, 100.0) for name in ("news_s", "news_t", "news_post")}
    for name in INITIAL_MEAN:
        mean, sd = expected.get(name, (0.0, 2_000.0))  # every seasonal state
        assert (INITIAL_MEAN[name], INITIAL_SD[name]) == (mean, sd), name


def _arrays(params, panel, layout_name):
    ssm = pilot_ssm(params, panel, LAYOUTS[layout_name])
    return {name: np.asarray(value) for name, value in ssm._asdict().items()}


def _state_index(layout_name, names):
    return [STATE_NAMES[layout_name].index(name) for name in names]


def test_shift_log_scale_moves_only_pre_2003_news_and_noise_scales(layout_name):
    panel = synthetic_panel(layout_name)
    pre = panel.pre_2003
    base = _arrays(TRUTH, panel, layout_name)
    moved = _arrays(TRUTH | {"shift_log_scale": -0.4}, panel, layout_name)
    news_s, news_t, news_post = _state_index(
        layout_name, ("news_s", "news_t", "news_post")
    )
    level = _state_index(layout_name, ("level",))[0]
    others = [
        k
        for k in range(len(STATE_NAMES[layout_name]))
        if k not in (level, news_s, news_t, news_post)
    ]

    for name in base:
        if name not in ("transition_cov", "observation_cov"):
            np.testing.assert_array_equal(moved[name], base[name], err_msg=name)
    for name in ("transition_cov", "observation_cov"):
        np.testing.assert_array_equal(moved[name][~pre], base[name][~pre], err_msg=name)
    q_base, q_moved = base["transition_cov"][pre], moved["transition_cov"][pre]
    assert (q_moved[:, news_s, news_s] != q_base[:, news_s, news_s]).all()
    assert (q_moved[:, news_t, news_t] != q_base[:, news_t, news_t]).all()
    np.testing.assert_array_equal(q_moved[:, news_post], q_base[:, news_post])
    np.testing.assert_array_equal(q_moved[:, others], q_base[:, others])

    def non_news_level(q):
        return q[:, level, level] - q[:, news_s, news_s] - q[:, news_t, news_t]

    np.testing.assert_allclose(
        non_news_level(q_moved), non_news_level(q_base), rtol=1e-12
    )
    r_base, r_moved = base["observation_cov"][pre], moved["observation_cov"][pre]
    assert (r_moved[:, :3, :3] != r_base[:, :3, :3]).all()
    np.testing.assert_array_equal(r_moved[:, ANCHOR_COLUMN], r_base[:, ANCHOR_COLUMN])


def test_shift_mean_moves_only_pre_2003_news_offsets(layout_name):
    panel = synthetic_panel(layout_name)
    pre = panel.pre_2003
    base = _arrays(TRUTH, panel, layout_name)
    moved = _arrays(TRUTH | {"shift_mean": 9.0}, panel, layout_name)
    shifted = _state_index(layout_name, ("level", "news_s", "news_t"))
    change = moved["transition_offset"] - base["transition_offset"]

    for name in base:
        if name != "transition_offset":
            np.testing.assert_array_equal(moved[name], base[name], err_msg=name)
    assert not change[~pre].any()
    # shift_mean moves from -6 to 9: each pre-2003 S and T news mean by 15, the level by 30.
    expected = np.broadcast_to([2 * 15.0, 15.0, 15.0], (pre.sum(), 3))
    np.testing.assert_allclose(change[pre][:, shifted], expected, rtol=1e-12)
    unshifted = [k for k in range(change.shape[1]) if k not in shifted]
    assert not change[pre][:, unshifted].any()


def test_kappa_crisis_moves_only_the_level_and_closing_news_shocks_of_crisis_steps(
    layout_name,
):
    panel = synthetic_panel(layout_name)
    crisis = panel.crisis
    base = _arrays(TRUTH, panel, layout_name)
    moved = _arrays(TRUTH | {"kappa_crisis": math.log(80.0)}, panel, layout_name)
    block = _state_index(layout_name, SCALED_BLOCK)
    others = [k for k in range(len(STATE_NAMES[layout_name])) if k not in block]

    for name in base:
        if name != "transition_cov":
            np.testing.assert_array_equal(moved[name], base[name], err_msg=name)
    q_base, q_moved = base["transition_cov"], moved["transition_cov"]
    np.testing.assert_array_equal(q_moved[~crisis], q_base[~crisis])
    # Every row and column outside the level and the S and T news, news_post's included.
    np.testing.assert_array_equal(q_moved[crisis][:, others], q_base[crisis][:, others])
    np.testing.assert_array_equal(
        q_moved[crisis][:, :, others], q_base[crisis][:, :, others]
    )
    # kappa_crisis moves from log 30 to log 80, so every scaled variance grows by
    # (80/30)^2. The level's variance also holds the unscaled post-T news variance.
    level = SCALED_BLOCK.index("level")
    post = STATE_NAMES[layout_name].index("news_post")
    post_var = q_base[crisis][:, post, post]
    unscaled = np.zeros((crisis.sum(), len(block), len(block)))
    unscaled[:, level, level] = post_var
    in_block = np.ix_(np.flatnonzero(crisis), block, block)
    np.testing.assert_allclose(
        q_moved[in_block],
        (80.0 / 30.0) ** 2 * (q_base[in_block] - unscaled) + unscaled,
        rtol=1e-12,
    )
    assert (q_moved[in_block] != q_base[in_block]).any(axis=(1, 2)).all()


def _explicit_derived(p):
    noise_f = 0.5 * p["sigma_news_s"] * math.exp(0.7 * p["noise_f_raw"])
    noise_s = 0.5 * p["sigma_news_t"] * math.exp(0.7 * p["noise_s_raw"])
    noise_t = 0.5 * p["sigma_news_t"] * math.exp(0.7 * p["noise_t_raw"])
    rho = p["rho_noise"]
    # e_s - e_f = (rho - 1) z_f + z_s and e_t - e_s = (rho - 1) e_s + z_t.
    revision_s = (rho - 1) ** 2 * noise_f**2 + noise_s**2
    e_s = rho**2 * noise_f**2 + noise_s**2
    revision_t = (rho - 1) ** 2 * e_s + noise_t**2
    return {
        "sigma_noise_f": noise_f,
        "sigma_noise_s": noise_s,
        "sigma_noise_t": noise_t,
        "sigma_news_post": p["sigma_news_annual"] / math.sqrt(12),
        "share_s": p["sigma_news_s"] ** 2 / (p["sigma_news_s"] ** 2 + revision_s),
        "share_t": p["sigma_news_t"] ** 2 / (p["sigma_news_t"] ** 2 + revision_t),
    }


def test_derived_quantities_work_on_floats_numpy_and_jax_arrays():
    draws = sample_prior(jax.random.PRNGKey(SEED), 5, LAYOUTS["dummy"])
    as_numpy = {name: np.asarray(value) for name, value in draws.items()}
    expected = [
        _explicit_derived({name: float(value[k]) for name, value in as_numpy.items()})
        for k in range(5)
    ]
    from_numpy = derived(as_numpy)
    from_jax = derived(draws)
    from_jit = jax.jit(derived)(draws)

    assert set(from_numpy) == set(expected[0])
    assert all(type(value) is np.ndarray for value in from_numpy.values())
    assert all(isinstance(value, jax.Array) for value in from_jax.values())
    assert derived(TRUTH) == pytest.approx(_explicit_derived(TRUTH), rel=1e-14)
    for name in expected[0]:
        column = [row[name] for row in expected]
        np.testing.assert_allclose(from_numpy[name], column, rtol=1e-13, err_msg=name)
        np.testing.assert_allclose(from_jax[name], column, rtol=1e-13, err_msg=name)
        np.testing.assert_allclose(from_jit[name], column, rtol=1e-13, err_msg=name)
    assert ((from_numpy["share_s"] > 0) & (from_numpy["share_s"] < 1)).all()
    gradient = jax.grad(lambda p: derived(p)["share_t"])(TRUTH)
    assert all(np.isfinite(float(value)) for value in gradient.values())


def test_sample_prior_draws_every_site_on_its_support():
    draws = sample_prior(jax.random.PRNGKey(SEED), 200, LAYOUTS["trig4"])
    again = sample_prior(jax.random.PRNGKey(SEED), 200, LAYOUTS["trig4"])

    assert tuple(draws) == SAMPLED_SITES
    assert all(value.shape == (200,) for value in draws.values())
    for name in SAMPLED_SITES:
        np.testing.assert_array_equal(draws[name], again[name])
        if name.startswith("sigma_"):
            assert (draws[name] > 0).all(), name
    assert (np.abs(draws["rho_noise"]) < 0.8).all()


def test_step_log_likelihood_batches_under_jit_and_vmap(layout_name):
    panel = synthetic_panel(layout_name)
    layout = LAYOUTS[layout_name]
    draws = sample_prior(jax.random.PRNGKey(SEED), NUM_PRIOR_DRAWS, layout)
    batched = jax.jit(jax.vmap(lambda p: step_log_likelihood(p, panel, layout)))(draws)

    assert batched.shape == (NUM_PRIOR_DRAWS, panel.num_steps)
    for k in range(NUM_PRIOR_DRAWS):
        params = {name: value[k] for name, value in draws.items()}
        eager = step_log_likelihood(params, panel, layout)
        total = kalman_filter(pilot_ssm(params, panel, layout), jnp.asarray(panel.y))
        np.testing.assert_allclose(batched[k], eager, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(eager.sum(), total.log_likelihood, rtol=1e-12)
        assert float(eager[MISSING_STEP]) == 0.0


def test_log_density_gradient_is_finite_at_prior_draws(layout_name):
    panel = synthetic_panel(layout_name)
    layout = LAYOUTS[layout_name]

    def log_joint(params):
        return log_density(pilot_model, (panel, layout), {}, params)[0]

    value_and_grad = jax.jit(jax.value_and_grad(log_joint))
    for params in _prior_draws(layout_name):
        value, gradient = value_and_grad(params)
        assert np.isfinite(float(value))
        assert set(gradient) == set(SAMPLED_SITES)
        assert all(np.isfinite(float(entry)) for entry in gradient.values())
