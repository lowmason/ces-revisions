"""The decision line and Markdown table rows of docs/decisions/seasonal-state.md.

Each comes from the committed Stage 6 JSON: the seasonal comparison and the Mac
probe-grid record that it names. The decision record pastes them, and
``tests/test_seasonal_state_decision.py`` checks that every one is in the record, so the
record cannot drift from its evidence. Run ``python -m ces_revisions.pilot.decision_tables
COMPARISON`` to print them. The GPU half of the decision moved to outcome O6 (roadmap
Stage 2), so no GPU table is built here.
"""

import json
import sys
from pathlib import Path

from ces_revisions.pilot.dimensions import matches_layout
from ces_revisions.pilot.seasonal import REFERENCE_LAYOUT
from ces_revisions.vintages import raw as vintage_raw

SEASONAL_HEADER = (
    "| Layout | Stage 9 low states | elpd vs dummy, all blocks "
    "| elpd vs dummy, reliable blocks | Req 18 | April 2020 "
    "| Ljung–Box Q (F, S, T) | Seconds |"
)
TIMING_HEADER = (
    "| Host | Stage | Wedge | Steps | States | Cells | Batch | Seconds | Compiled GB |"
)


def decision_lines(comparison: dict) -> list[str]:
    """The sentence that states the decision: the layout the rule chose."""
    layout = comparison["rule"]["layout"]
    return [
        f"The seasonal state for O6 and every later fit of this model is `{layout}`."
    ]


def seasonal_rows(comparison: dict) -> list[str]:
    """One row per fitted layout: its elpd difference, Req 18, residuals, and cost."""
    rule = comparison["rule"]
    rows = []
    for entry in comparison["runs"]:
        layout = entry["layout"]
        candidate = comparison["candidates"][layout]
        if layout == REFERENCE_LAYOUT:
            every, reliable = "reference", "reference"
        else:
            every = _difference(rule["all_blocks"]["differences"][layout])
            reliable = _difference(rule["reliable_blocks"]["differences"][layout])
        rows.append(
            f"| `{layout}` | {candidate['states']} | {every} | {reliable} "
            f"| {_yes(candidate['meets_req18'])} "
            f"| {_passed(candidate['april_2020_passed'])} "
            f"| {', '.join(f'{q:.0f}' for q in candidate['ljung_box_q'].values())} "
            f"| {_number(candidate['seconds'], '.3f')} |"
        )
    return rows


def timing_rows(records: list[dict], layout_name: str) -> list[str]:
    """One row per host, grid entry at the layout's dimensions, and batch.

    A skipped batch shows "skipped", and one skipped before compiling shows no memory.
    """
    rows = []
    for record in records:
        for entry in record["entries"]:
            if not matches_layout(entry, layout_name):
                continue
            for batch in entry["batches"]:
                memory = batch["memory_analysis"]
                total = None if memory is None else memory["total_bytes"] / 1e9
                rows.append(
                    f"| {record['label']} | {entry['stage']} | {entry['wedge']} "
                    f"| {entry['steps']} | {entry['states']} | {entry['cells']} "
                    f"| {batch['batch']} "
                    f"| {_number(batch.get('median_seconds'), '.3f', 'skipped')} "
                    f"| {_number(total, '.1f')} |"
                )
    return rows


def _difference(difference: dict) -> str:
    return f"{difference['diff']:+.1f} ± {difference['se']:.1f}"


def _number(value: float | None, spec: str, missing: str = "—") -> str:
    return missing if value is None else format(value, spec)


def _yes(flag: bool) -> str:
    return "yes" if flag else "no"


def _passed(flag: bool) -> str:
    return "passed" if flag else "failed"


def main(argv: list[str] | None = None) -> None:
    (comparison_path,) = map(Path, sys.argv[1:] if argv is None else argv)
    comparison = json.loads(comparison_path.read_text())
    layout = comparison["rule"]["layout"]
    mac = json.loads((vintage_raw.ROOT / comparison["grid"]["path"]).read_text())
    sections = [
        decision_lines(comparison),
        [
            SEASONAL_HEADER,
            "|---|---:|---|---|---|---|---|---:|",
            *seasonal_rows(comparison),
        ],
        [
            TIMING_HEADER,
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
            *timing_rows([mac], layout),
        ],
    ]
    print("\n\n".join("\n".join(section) for section in sections))


if __name__ == "__main__":
    main()
