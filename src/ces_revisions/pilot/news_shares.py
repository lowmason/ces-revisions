"""The pilot's posterior news shares: what part of each revision's variance is news.

``derived()`` (``model.py``) defines two shares for every posterior draw: the news
variance's share of the F→S revision's variance (``share_s``) and of the S→T revision's
(``share_t``), without the rounding variance. ``share_draws`` pools the draws of every
chain, ``summarize`` reduces them to a median and a central interval, and
``shares_block`` renders the Stage 2 findings table from a committed record of the
summaries, so the table needs neither the run directory nor the panel.
"""

import numpy as np
import xarray as xr

from ces_revisions.pilot.model import SAMPLED_SITES, derived

INTERVAL_PROB = 0.90
# The revision that each share splits. share_t is the news share of the S→T revision, not
# of F→T, which Stage 1 reports.
SHARES = {"share_s": "F→S", "share_t": "S→T"}


def share_draws(posterior: xr.Dataset) -> dict[str, np.ndarray]:
    """Each share's draws, with ``chain`` and ``draw`` pooled into one axis."""
    pooled = posterior[list(SAMPLED_SITES)].stack(sample=("chain", "draw"))
    params = {name: pooled[name].to_numpy() for name in SAMPLED_SITES}
    shares = derived(params)
    return {key: np.asarray(shares[key]) for key in SHARES}


def summarize(draws: np.ndarray, prob: float = INTERVAL_PROB) -> dict[str, float]:
    """The median and the central ``prob`` interval of the draws."""
    median, lower, upper = np.quantile(draws, [0.5, (1 - prob) / 2, (1 + prob) / 2])
    return {"median": float(median), "lower": float(lower), "upper": float(upper)}


def shares_block(record: dict) -> str:
    """The Stage 2 findings table for a committed shares record."""
    rows = [
        f"| {revision} | `{key}` | {record[key]['median']:.3f} "
        f"| {record[key]['lower']:.3f}–{record[key]['upper']:.3f} |"
        for key, revision in SHARES.items()
    ]
    note = (
        f"The shares summarize {record['num_draws']} pooled posterior draws of "
        f"`{record['run_dir']}`, which took {record['sampling_seconds']:.1f} seconds "
        "to sample."
    )
    return "\n".join(
        [
            f"| Revision | Share | Median | {record['interval_prob']:.0%} interval |",
            "|---|---|---:|---:|",
            *rows,
            "",
            note,
        ]
    )
