"""The pre-registered rule that chooses the seasonal layout for Stages 7-17.

Req 6 keeps the dummy recursion unless "diagnostics favor" a trigonometric state. The rule
compares each alternative with the dummy on benchmark-year PSIS-LOO blocks, which are
close to independent, so their elpd difference has an honest standard error:

(a) an alternative that meets Req 18, passes its April 2020 check, and whose elpd beats
    the dummy's by more than SE_MULTIPLE standard errors replaces it (the largest such
    difference wins);
(b) otherwise an alternative with fewer states replaces it when its elpd is within
    SE_MULTIPLE standard errors, it meets Req 18, it passes its April 2020 check, its
    Ljung-Box Q is at most LJUNG_BOX_Q_RATIO_MAX times the dummy's in every closing
    column, and its measured value-and-gradient time is at least COST_SAVING_MIN below
    the dummy's;
(c) otherwise the dummy stays.

A block whose Pareto k exceeds PARETO_K_MAX in any layout makes PSIS unreliable there, so
the rule runs twice, on every block and on the blocks reliable in every layout. When the
two runs choose different layouts, the comparison is unreliable and the dummy stays.
"""

import numpy as np

from ces_revisions.pilot.dimensions import matches_layout

REFERENCE_LAYOUT = "dummy"
SE_MULTIPLE = 2.0
COST_SAVING_MIN = 0.25
PARETO_K_MAX = 0.7
LJUNG_BOX_ALPHA = 0.01
# Clause (b)'s residual guard: Q under the white-noise null has a few units of SD, while a
# layout that misses the seasonal pattern multiplies it.
LJUNG_BOX_Q_RATIO_MAX = 1.25
# Clause (b)'s cost: one value-and-gradient at the Stage 9 low-wedge dimensions over the
# full 572 months, batch 1, from a gpu_probe_grid.py record. Stage 9 is the largest model
# the layout choice sizes; the high wedge waits on Stage 8's choice.
COST_ENTRY = {"stage": 9, "wedge": "low", "steps": 572}
COST_BATCH = 1


def elpd_difference(
    alternative: np.ndarray, reference: np.ndarray, keep: np.ndarray
) -> dict:
    """The summed pointwise elpd difference over the kept blocks and its standard error.

    As in ``az.compare``, the standard error is sqrt(B) times the SD of the B pointwise
    differences.
    """
    difference = (np.asarray(alternative) - np.asarray(reference))[keep]
    return {
        "diff": float(difference.sum()),
        "se": float(np.sqrt(difference.size) * difference.std(ddof=1)),
        "num_blocks": int(difference.size),
    }


def ljung_box_rejections(ljung_box: dict[str, dict]) -> int:
    """How many closing columns' Ljung-Box tests reject at LJUNG_BOX_ALPHA."""
    return sum(test["p_value"] < LJUNG_BOX_ALPHA for test in ljung_box.values())


def verdict(candidates: dict[str, dict], differences: dict[str, dict]) -> dict:
    """The layout the rule chooses, with the clause that chose it.

    ``candidates`` maps every layout, the reference included, to its ``states`` (int),
    ``meets_req18`` (bool), ``april_2020_passed`` (bool), ``ljung_box_q`` (closing
    column to Q), and ``seconds`` (the measured value-and-gradient time, or None when it
    was not measured). ``differences`` maps each alternative to its ``elpd_difference``
    against the reference.
    """
    reference = candidates[REFERENCE_LAYOUT]
    better = {
        name: difference["diff"]
        for name, difference in differences.items()
        if difference["diff"] > SE_MULTIPLE * difference["se"]
        and _eligible(candidates[name])
    }
    if better:
        return {"layout": max(better, key=better.get), "clause": "a"}
    cheaper = [
        name
        for name, difference in differences.items()
        if candidates[name]["states"] < reference["states"]
        and abs(difference["diff"]) <= SE_MULTIPLE * difference["se"]
        and _eligible(candidates[name])
        and _residuals_hold(candidates[name], reference)
        and _saving(candidates[name], reference) >= COST_SAVING_MIN
    ]
    if cheaper:
        return {
            "layout": min(cheaper, key=lambda name: candidates[name]["seconds"]),
            "clause": "b",
        }
    return {"layout": REFERENCE_LAYOUT, "clause": "c"}


def choose_layout(
    candidates: dict[str, dict],
    pointwise: dict[str, np.ndarray],
    pareto_k: dict[str, np.ndarray],
) -> dict:
    """The rule on every block and on the reliable blocks, and the layout it settles on.

    ``pointwise`` and ``pareto_k`` map every layout to its benchmark-year blocks' elpd and
    Pareto k, in the same block order.
    """
    reference = np.asarray(pointwise[REFERENCE_LAYOUT])
    reliable = np.all(
        [np.asarray(k) <= PARETO_K_MAX for k in pareto_k.values()], axis=0
    )
    alternatives = [name for name in candidates if name != REFERENCE_LAYOUT]
    runs = {}
    for label, keep in (
        ("all_blocks", np.ones(reference.size, dtype=bool)),
        ("reliable_blocks", reliable),
    ):
        differences = {
            name: elpd_difference(pointwise[name], reference, keep)
            for name in alternatives
        }
        runs[label] = {
            "differences": differences,
            "verdict": verdict(candidates, differences),
        }
    agree = (
        runs["all_blocks"]["verdict"]["layout"]
        == runs["reliable_blocks"]["verdict"]["layout"]
    )
    return {
        **runs,
        "unreliable_blocks": int((~reliable).sum()),
        "reliable": agree,
        "layout": runs["all_blocks"]["verdict"]["layout"]
        if agree
        else REFERENCE_LAYOUT,
    }


def grid_seconds(grid: dict, layout_name: str) -> float | None:
    """The COST_ENTRY median seconds at a layout's dimensions in a grid record, if any."""
    for entry in grid["entries"]:
        if matches_layout(entry, layout_name) and all(
            entry[key] == value for key, value in COST_ENTRY.items()
        ):
            for batch in entry["batches"]:
                if batch["batch"] == COST_BATCH and "median_seconds" in batch:
                    return batch["median_seconds"]
    return None


def _eligible(candidate: dict) -> bool:
    return candidate["meets_req18"] and candidate["april_2020_passed"]


def _residuals_hold(candidate: dict, reference: dict) -> bool:
    """Whether the candidate's Q is within the ratio of the reference's in every column."""
    return all(
        candidate["ljung_box_q"][column] <= LJUNG_BOX_Q_RATIO_MAX * q
        for column, q in reference["ljung_box_q"].items()
    )


def _saving(candidate: dict, reference: dict) -> float:
    """The candidate's fractional time saving over the reference; -inf when unmeasured."""
    if candidate["seconds"] is None or reference["seconds"] is None:
        return -np.inf
    return 1.0 - candidate["seconds"] / reference["seconds"]
