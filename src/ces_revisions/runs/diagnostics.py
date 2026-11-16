"""Req 18's MCMC diagnostics as one JSON-safe record.

Req 18 asks for rank-normalized R-hat below 1.01, bulk and tail ESS above 400, zero
post-warmup divergences, acceptable energy-BFMI, and no systematic tree-depth saturation.
The last two are numeric here as in the Stage 1 synthetic pilot's test: every chain's
energy-BFMI above 0.3, and under 1% of draws at the maximum tree depth. Every check is a
comparison that a NaN fails, so a statistic ArviZ cannot compute never passes.
"""

import math
from dataclasses import asdict, dataclass

import arviz as az
import numpy as np
from scipy.stats import rankdata

RHAT_METHOD = "rank"  # ArviZ's rank-normalized split R-hat, as Req 18 names it
SAMPLE_STATS = ("diverging", "energy", "n_steps", "acceptance_rate", "step_size")


@dataclass(frozen=True)
class Thresholds:
    rhat_max: float = 1.01
    ess_min: float = 400.0
    bfmi_min: float = 0.3
    max_tree_depth: int = 10  # the sampler's own setting: saturation is relative to it
    saturated_share_max: float = 0.01
    divergences_max: int = 0


REQ18 = Thresholds()


def diagnostics_record(
    idata, thresholds: Thresholds = REQ18, *, rank_bins: int = 20
) -> dict:
    """R-hat, ESS, BFMI, divergences, tree depth, and rank histograms, with Req 18 checks.

    Each element of a non-scalar posterior variable is reported as ``name[i,j]``. The
    rank histograms count, per chain, the draws whose rank among all chains' pooled draws
    falls in each of ``rank_bins`` equal bins; a well-mixed chain fills them evenly.
    """
    posterior, stats = idata["posterior"], idata["sample_stats"]
    missing = [name for name in SAMPLE_STATS if name not in stats.data_vars]
    if missing:
        raise ValueError(f"sample_stats lacks {missing}")
    if not posterior.data_vars:
        raise ValueError("the posterior has no variables")
    rhat = az.rhat(idata, method=RHAT_METHOD)
    ess_bulk = az.ess(idata, method="bulk")
    ess_tail = az.ess(idata, method="tail")
    variables, rank_counts = {}, {}
    for name in posterior.data_vars:
        draws = np.asarray(posterior[name])
        for index in np.ndindex(draws.shape[2:]):
            element = f"{name}[{','.join(map(str, index))}]" if index else name
            variables[element] = {
                "rhat": _json_float(np.asarray(rhat[name])[index]),
                "ess_bulk": _json_float(np.asarray(ess_bulk[name])[index]),
                "ess_tail": _json_float(np.asarray(ess_tail[name])[index]),
            }
            rank_counts[element] = _rank_histogram(draws[(..., *index)], rank_bins)

    diverging = np.asarray(stats["diverging"]).astype(bool)
    n_steps = np.asarray(stats["n_steps"])
    saturated = n_steps >= 2**thresholds.max_tree_depth - 1
    bfmi = [_json_float(value) for value in np.asarray(az.bfmi(idata)["energy"])]
    record = {
        "thresholds": asdict(thresholds),
        "rhat_method": RHAT_METHOD,
        "num_chains": int(diverging.shape[0]),
        "num_draws": int(diverging.shape[1]),
        "variables": variables,
        "bfmi": bfmi,
        "divergences": {
            "total": int(diverging.sum()),
            "per_chain": diverging.sum(axis=1).tolist(),
        },
        "tree_depth": {
            "saturation_steps": 2**thresholds.max_tree_depth - 1,
            "saturated_share": float(saturated.mean()),
            "saturated_per_chain": saturated.sum(axis=1).tolist(),
            "n_steps_max": int(n_steps.max()),
            "n_steps_mean_per_chain": n_steps.mean(axis=1).tolist(),
        },
        "acceptance_rate": np.asarray(stats["acceptance_rate"]).mean(axis=1).tolist(),
        "step_size": np.asarray(stats["step_size"])[:, -1].tolist(),
        "rank_histograms": {"bins": rank_bins, "counts": rank_counts},
    }
    statistics = {
        key: [entry[key] for entry in variables.values()]
        for key in ("rhat", "ess_bulk", "ess_tail")
    }
    checks = {
        "rhat": _all_below(statistics["rhat"], thresholds.rhat_max),
        "ess_bulk": _all_above(statistics["ess_bulk"], thresholds.ess_min),
        "ess_tail": _all_above(statistics["ess_tail"], thresholds.ess_min),
        "bfmi": _all_above(bfmi, thresholds.bfmi_min),
        "divergences": record["divergences"]["total"] <= thresholds.divergences_max,
        "tree_depth": record["tree_depth"]["saturated_share"]
        < thresholds.saturated_share_max,
    }
    record["worst"] = {
        "rhat_max": _worst(statistics["rhat"], max),
        "ess_bulk_min": _worst(statistics["ess_bulk"], min),
        "ess_tail_min": _worst(statistics["ess_tail"], min),
        "bfmi_min": _worst(bfmi, min),
    }
    record["checks"] = checks
    record["meets_req18"] = all(checks.values())
    return record


def _rank_histogram(draws: np.ndarray, bins: int) -> list[list[int]]:
    """Per-chain counts of pooled ranks in equal bins; draws is (chain, draw)."""
    ranks = rankdata(draws, axis=None).reshape(draws.shape)  # 1..N, ties averaged
    bin_index = np.minimum(((ranks - 1) * bins / ranks.size).astype(np.int64), bins - 1)
    return [np.bincount(chain, minlength=bins).tolist() for chain in bin_index]


def _json_float(value) -> float | None:
    """A Python float, or None for NaN or infinity, so json.dumps(allow_nan=False) works."""
    value = float(value)
    return value if math.isfinite(value) else None


def _all_below(values: list[float | None], bound: float) -> bool:
    return all(value is not None and value < bound for value in values)


def _all_above(values: list[float | None], bound: float) -> bool:
    return all(value is not None and value > bound for value in values)


def _worst(values: list[float | None], pick) -> float | None:
    if any(value is None for value in values):
        return None
    return pick(values)
