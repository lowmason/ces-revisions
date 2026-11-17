"""Fit one series of the scale model, and complete its DataTree for the checks.

``fit_series`` samples through the run harness (``ces_revisions.runs.run_nuts``: dispersed
starts, Req 18 diagnostics, provenance), then replaces the harness's generic groups with
this model's: the pointwise log likelihood as ``r`` over ``reference_month``, the observed
and replicated revisions on the same dates, the log prior of each sampled site at each draw
(for power-scaling sensitivity), and named month and year dimensions. Only the pointwise
term is stored, so ``az.loo`` finds one variable; the reference-year blocks are its sums,
which is exact because months are conditionally independent given the parameters.
"""

from dataclasses import asdict, dataclass, replace

import arviz as az
import jax
import numpy as np
import xarray as xr
from numpyro import handlers
from numpyro.infer import Predictive

from ces_revisions.runs.fit import RunResult, RunSettings, run_nuts
from ces_revisions.scale_model.checks import (
    predictive_p_values,
    predictive_statistics,
    sum_by_year,
)
from ces_revisions.scale_model.data import NUM_MONTHS, SAMPLE_START, SERIES, ScaleData
from ces_revisions.scale_model.model import (
    OBSERVED_SITE,
    SAMPLED_SITES,
    TARGET_ACCEPT_PROB,
    Priors,
    location_scale,
    pointwise_log_likelihood,
    sample_prior,
    scale_model,
)
from ces_revisions.vintages import raw as vintage_raw

MODEL_NAME = "stage-1-scale-model"
SEED = sum(map(ord, MODEL_NAME))
INIT_DRAWS_SEED = sum(map(ord, f"{MODEL_NAME}-init"))
PREDICTIVE_SEED = sum(map(ord, f"{MODEL_NAME}-posterior-predictive"))
NUM_INIT_DRAWS = 1000
INTERVALS = {"eti90": 0.90, "eti95": 0.95}
PARETO_K_MAX = 0.7
SERIES_LABEL = {series: f"{series[0].lower()}-{series[1].lower()}" for series in SERIES}
# Extra dimensions of the posterior's vector variables.
VECTOR_DIMS = {
    "p": "calendar_month",
    "c": "calendar_month",
    "z_y": "year",
    "u": "year",
}
DEVIATIONS = (
    (
        "no benchmark-cell term omega: from 2004 its cell is one calendar month in every "
        "year but 2003, so one observation would identify it against that month's effect"
    ),
    "the window is centered, and its year mean taken, over non-lapse months only",
    (
        "month effects on log scale have prior SD 0.4, not 0.5: at 0.5 the prior's "
        "central 90% of adjacent-month scale ratios was 0.27-3.7, outside the "
        "recommendation's target of about 0.3-3"
    ),
    (
        "a lapse month's within-year window is 0: a shutdown, not the calendar, set its "
        "18-26 days, and with them it leveraged beta and gamma; on log scale the lapse "
        "indicator absorbs it, and on the mean it keeps only its month and 5-week terms"
    ),
)
REPORTED = (
    "beta",
    "scale_ratio",
    "gamma",
    "phi",
    "delta",
    "kappa",
    "lam",
    "tau_y",
    "nu",
    "a",
    "m",
)


@dataclass(frozen=True)
class SeriesFit:
    data: ScaleData
    priors: Priors
    result: RunResult  # result.idata is the completed DataTree
    loo: dict
    checks: dict


def series_label(series: tuple[str, str]) -> str:
    return SERIES_LABEL[series]


def model_record(data: ScaleData, priors: Priors) -> dict:
    return {
        "name": MODEL_NAME,
        "series": list(data.series),
        "sample_start": SAMPLE_START.isoformat(),
        "num_months": len(data.reference_months),
        "num_years": len(data.years),
        "sampled_sites": list(SAMPLED_SITES),
        "priors": asdict(priors),
        "deviations_from_recommendation": list(DEVIATIONS),
        "init_draws": {"seed": INIT_DRAWS_SEED, "num_draws": NUM_INIT_DRAWS},
    }


def data_arrays(data: ScaleData) -> dict[str, np.ndarray]:
    """The arrays the model reads, hashed into the run's provenance."""
    return {
        "revision": data.revision,
        "window": data.window,
        "year_window": data.year_window,
        "five_week": data.five_week,
        "lapse": data.lapse,
        "year": data.year,
        "month": data.month,
    }


def fit_series(
    data: ScaleData,
    priors: Priors,
    settings: RunSettings,
    root=vintage_raw.ROOT,
) -> SeriesFit:
    """Sample one series with NUTS and complete its DataTree, LOO, and predictive checks."""
    prior_draws = sample_prior(
        jax.random.PRNGKey(INIT_DRAWS_SEED), NUM_INIT_DRAWS, data, priors
    )
    result = run_nuts(
        scale_model,
        (data, priors, data.revision),
        settings=settings,
        prior_draws=prior_draws,
        step_fn=lambda params: pointwise_log_likelihood(params, data),
        reference_months=data.reference_months,
        benchmark_year=np.array([data.years[index] for index in data.year]),
        observed_any=np.ones(len(data.reference_months), dtype=bool),
        model_record=model_record(data, priors),
        arrays=data_arrays(data),
        root=root,
    )
    posterior = result.idata["posterior"]
    samples = {name: np.asarray(posterior[name]) for name in SAMPLED_SITES}
    replicated, mu, sigma = posterior_predictive(samples, data, priors)
    idata = complete_inference_data(
        result.idata, data, replicated, log_prior_draws(samples, data, priors)
    )
    return SeriesFit(
        data=data,
        priors=priors,
        result=replace(result, idata=idata),
        loo=loo_record(idata, data),
        checks=predictive_check_record(data, replicated, mu, sigma),
    )


def _flatten(samples_by_chain: dict) -> tuple[dict, tuple[int, int]]:
    leading = next(iter(samples_by_chain.values())).shape[:2]
    flat = {
        name: np.reshape(value, (-1, *np.shape(value)[2:]))
        for name, value in samples_by_chain.items()
    }
    return flat, leading


def posterior_predictive(
    samples_by_chain: dict, data: ScaleData, priors: Priors
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replicated revisions and each draw's mu and sigma, each (chain, draw, T)."""
    flat, leading = _flatten(samples_by_chain)
    draws = Predictive(
        scale_model, posterior_samples=flat, return_sites=[OBSERVED_SITE]
    )(jax.random.PRNGKey(PREDICTIVE_SEED), data, priors)
    mu, log_sigma = jax.vmap(lambda params: location_scale(params, data))(flat)
    shape = (*leading, len(data.reference_months))
    return (
        np.asarray(draws[OBSERVED_SITE]).reshape(shape),
        np.asarray(mu).reshape(shape),
        np.exp(np.asarray(log_sigma)).reshape(shape),
    )


def log_prior_draws(
    samples_by_chain: dict, data: ScaleData, priors: Priors
) -> dict[str, np.ndarray]:
    """Each sampled site's prior log density at each draw, (chain, draw, ...).

    A site with an event dimension (the zero-sum month effects) gives one value per draw;
    the year plate gives one per year.
    """
    flat, leading = _flatten(samples_by_chain)

    def one(params):
        trace = handlers.trace(handlers.substitute(scale_model, params)).get_trace(
            data, priors, data.revision
        )
        return {
            name: trace[name]["fn"].log_prob(trace[name]["value"])
            for name in SAMPLED_SITES
        }

    values = jax.vmap(one)(flat)
    return {
        name: np.asarray(value).reshape(*leading, *np.shape(value)[1:])
        for name, value in values.items()
    }


def complete_inference_data(
    idata, data: ScaleData, replicated: np.ndarray, log_prior: dict
) -> xr.DataTree:
    """The harness's DataTree with this model's groups, dimensions, and coordinates."""
    months = np.array(
        [np.datetime64(month, "ns") for month in data.reference_months],
        dtype="datetime64[ns]",
    )
    posterior = idata["posterior"]
    sample = {
        "chain": posterior["chain"].values,
        "draw": posterior["draw"].values,
    }
    named = {
        **sample,
        "calendar_month": np.arange(1, NUM_MONTHS + 1),
        "year": np.array(data.years),
    }

    def sample_dims(name: str, ndim: int) -> tuple[str, ...]:
        extra = (VECTOR_DIMS[name],) if ndim > 2 else ()
        return ("chain", "draw", *extra)

    posterior_ds = xr.Dataset(
        {
            name: (sample_dims(name, posterior[name].ndim), posterior[name].values)
            for name in posterior.data_vars
        },
        coords=named,
        attrs=dict(posterior.attrs),
    )
    pointwise = idata["log_likelihood"]["panel_month"]
    if not np.array_equal(pointwise["reference_month"].values, months):
        raise ValueError(
            "the harness's log likelihood is not in the data's month order"
        )
    by_month = ("chain", "draw", "reference_month")
    tree = idata.copy()
    tree["posterior"] = xr.DataTree(posterior_ds)
    tree["observed_data"] = xr.DataTree(
        xr.Dataset(
            {OBSERVED_SITE: (("reference_month",), data.revision)},
            coords={"reference_month": months},
        )
    )
    tree["posterior_predictive"] = xr.DataTree(
        xr.Dataset(
            {OBSERVED_SITE: (by_month, replicated)},
            coords={**sample, "reference_month": months},
        )
    )
    tree["log_likelihood"] = xr.DataTree(
        xr.Dataset(
            {OBSERVED_SITE: (by_month, pointwise.values)},
            coords={**sample, "reference_month": months},
        )
    )
    tree["log_prior"] = xr.DataTree(
        xr.Dataset(
            {
                name: (sample_dims(name, value.ndim), value)
                for name, value in log_prior.items()
            },
            coords=named,
        )
    )
    return tree


def loo_record(idata, data: ScaleData) -> dict:
    """PSIS-LOO with one block per reference month and one per reference year."""
    pointwise = idata["log_likelihood"][OBSERVED_SITE]
    by_year = xr.Dataset(
        {
            OBSERVED_SITE: (
                ("chain", "draw", "reference_year"),
                sum_by_year(pointwise.values, data),
            )
        },
        coords={
            "chain": pointwise["chain"].values,
            "draw": pointwise["draw"].values,
            "reference_year": np.array(data.years),
        },
    )
    year_tree = xr.DataTree.from_dict(
        {
            "posterior": idata["posterior"].to_dataset(),
            "log_likelihood": by_year,
        }
    )
    month_labels = [month.isoformat()[:7] for month in data.reference_months]
    return {
        "month": _loo_entry(idata, month_labels),
        "reference_year": _loo_entry(year_tree, [str(year) for year in data.years]),
    }


def _loo_entry(tree, labels: list[str]) -> dict:
    loo = az.loo(tree, pointwise=True, var_name=OBSERVED_SITE)
    pareto_k = np.asarray(loo.pareto_k).ravel()
    high = [
        {"block": label, "pareto_k": float(k)}
        for label, k in zip(labels, pareto_k, strict=True)
        if not k <= PARETO_K_MAX
    ]
    return {
        "elpd": float(loo.elpd),
        "se": float(loo.se),
        "p_loo": float(loo.p),
        "num_blocks": len(labels),
        "pareto_k_max": float(np.max(pareto_k)),
        "pareto_k_above_0_7": high,
    }


def interval_summary(draws: np.ndarray) -> dict:
    """Mean, SD, and the equal-tailed intervals of INTERVALS over every draw."""
    draws = np.asarray(draws, dtype=np.float64).ravel()
    summary = {"mean": float(draws.mean()), "sd": float(draws.std(ddof=1))}
    for name, probability in INTERVALS.items():
        tail = (1 - probability) / 2
        summary[name] = np.quantile(draws, [tail, 1 - tail]).tolist()
    return summary


def estimate_record(idata) -> dict:
    """Interval summaries of the reported parameters, and P(scale ratio > 1)."""
    posterior = idata["posterior"]
    record = {name: interval_summary(posterior[name].values) for name in REPORTED}
    record["scale_ratio"]["probability_above_1"] = float(
        np.mean(posterior["scale_ratio"].values > 1)
    )
    return record


def predictive_check_record(
    data: ScaleData, replicated: np.ndarray, mu: np.ndarray, sigma: np.ndarray
) -> dict:
    """Each check statistic: its observed value, replicated quantiles, and p-value."""
    num_months = len(data.reference_months)
    flat = (-1, num_months)
    mu, sigma = mu.reshape(flat), sigma.reshape(flat)
    replicated = replicated.reshape(flat)
    observed = predictive_statistics(
        np.broadcast_to(data.revision, replicated.shape), mu, sigma, data
    )
    simulated = predictive_statistics(replicated, mu, sigma, data)
    p_values = predictive_p_values(observed, simulated)
    record = {}
    for name, p_value in p_values.items():
        if p_value is None:
            continue
        record[name] = {
            "observed": float(np.mean(observed[name])),
            "replicated_q05_q50_q95": np.quantile(
                simulated[name], [0.05, 0.5, 0.95]
            ).tolist(),
            "p_value": p_value,
        }
    return record


def default_settings(seed: int = SEED) -> RunSettings:
    return RunSettings(seed=seed, target_accept_prob=TARGET_ACCEPT_PROB)
