# Keep the dummy seasonal state

- **Status:** Accepted
- **Date:** 2026-11-15
- **Deciders:** Lowell Mason
- **Blast radius:** the seasonal block of every sector's state in O6 and the later model outcomes of [`specs/ces-revision-drivers-roadmap.md`](../../specs/ces-revision-drivers-roadmap.md)

## Context

Req 6 of [`specs/completed/ces-revisions.md`](../../specs/completed/ces-revisions.md) uses the dummy seasonal recursion and says that a trigonometric seasonal state may replace it if diagnostics favor it. Roadmap Stage 2 moved the GPU choice that plan 11 first paired with this decision to outcome O6 (`specs/ces-revision-drivers-roadmap.md:24`), so this record holds the seasonal state alone.

Plan 11 fitted the aggregate total-nonfarm pilot under three layouts with the same data, seeds, and sampler settings: `dummy` (11 states), `trig` (harmonics 1–5 and the Nyquist harmonic, 11 states), and `trig4` (harmonics 1–4, 8 states). It timed the engine's value and gradient at the Stage 7–9 dimensions on the Mac. The rule is code (`src/ces_revisions/pilot/seasonal.py`), fixed by plan 11 before the runs recorded here. A planning spike under an earlier model had already applied the seasonal rule's clauses to these layouts (trig +15.4 ± 9.4, trig4 −466 ± 70 on benchmark-year blocks). The reliable-block rerun, and the April 2020 and Ljung–Box conditions, were set after that spike evidence was read.

## Decision

The seasonal state for O6 and every later fit of this model is `dummy`.

### The seasonal rule

The rule compares each alternative with the dummy on benchmark-year PSIS-LOO blocks. An alternative can replace the dummy only if it meets Req 18 and passes its own April 2020 check. It replaces the dummy if its elpd is more than 2 standard errors better. Otherwise an alternative with fewer states replaces it if its elpd is within 2 standard errors, its Ljung–Box Q is at most 1.25 times the dummy's in every closing column, and its measured value and gradient at Stage 9 low, $`T=572`$, is at least 25% cheaper on the Mac. The rule runs on every block and again on the blocks whose Pareto $`k`$ is at most 0.7 in every layout. If the two runs choose different layouts, the dummy stays. The two runs agreed (`rule.reliable` is true). Both ended at clause (c), so the dummy stays. 2 of the 49 benchmark-year blocks were unreliable (`rule.unreliable_blocks`), which leaves 47 for the second run.

| Layout | Stage 9 low states | elpd vs dummy, all blocks | elpd vs dummy, reliable blocks | Req 18 | April 2020 | Ljung–Box Q (F, S, T) | Seconds |
|---|---:|---|---|---|---|---|---:|
| `dummy` | 247 | reference | reference | yes | passed | 84, 49, 61 | 2.935 |
| `trig` | 247 | +12.8 ± 8.6 | +10.5 ± 8.5 | yes | passed | 43, 28, 41 | 2.935 |
| `trig4` | 214 | -470.8 ± 67.6 | -517.0 ± 26.1 | yes | passed | 2303, 1526, 1519 | 2.358 |

### The cost on the Mac

Each row is one value and gradient of the engine at the chosen layout's dimensions, timed on a synthetic float64 model by `scripts/gpu_probe_grid.py`. "Batch" counts chains or SBC replicates run at once. "Compiled GB" is XLA's memory analysis of the compiled batch. A skipped batch did not fit the Mac's 20 GB guard. One with no compiled size was skipped without compiling, after a smaller batch did not fit. The `trig` layout has the dummy's dimensions, so the dummy's entries serve both. The record is `docs/decisions/cloud-gpu-probe/mac-grid.json`. The Stage 7, 8, and 9 in the table are the parked roadmap's model stages, which O6 now covers.

| Host | Stage | Wedge | Steps | States | Cells | Batch | Seconds | Compiled GB |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| mac | 7 | low | 572 | 190 | 37 | 1 | 1.277 | 2.0 |
| mac | 7 | low | 572 | 190 | 37 | 4 | 2.886 | 6.0 |
| mac | 7 | low | 280 | 190 | 37 | 1 | 0.642 | 1.0 |
| mac | 7 | low | 280 | 190 | 37 | 4 | 1.341 | 2.9 |
| mac | 8 | low | 572 | 203 | 85 | 1 | 2.066 | 2.7 |
| mac | 8 | low | 572 | 203 | 85 | 4 | 4.208 | 8.4 |
| mac | 8 | low | 280 | 203 | 85 | 1 | 1.022 | 1.3 |
| mac | 8 | low | 280 | 203 | 85 | 4 | 1.918 | 4.1 |
| mac | 8 | high | 572 | 324 | 217 | 1 | 3.617 | 8.7 |
| mac | 8 | high | 572 | 324 | 217 | 4 | skipped | 26.9 |
| mac | 8 | high | 280 | 324 | 217 | 1 | 1.703 | 4.3 |
| mac | 8 | high | 280 | 324 | 217 | 4 | 6.019 | 13.2 |
| mac | 9 | low | 572 | 247 | 145 | 1 | 2.935 | 4.7 |
| mac | 9 | low | 572 | 247 | 145 | 4 | 7.033 | 14.5 |
| mac | 9 | low | 280 | 247 | 145 | 1 | 1.402 | 2.3 |
| mac | 9 | low | 280 | 247 | 145 | 4 | 3.350 | 7.1 |
| mac | 9 | high | 572 | 368 | 277 | 1 | 5.403 | 12.1 |
| mac | 9 | high | 572 | 368 | 277 | 4 | skipped | 37.4 |
| mac | 9 | high | 280 | 368 | 277 | 1 | 2.586 | 5.9 |
| mac | 9 | high | 280 | 368 | 277 | 4 | 8.670 | 18.3 |

## Consequences

- O6 and the later model outcomes build each sector's seasonal block with the chosen layout from `src/ces_revisions/pilot/layout.py`, and size their models with `src/ces_revisions/pilot/dimensions.py`.
- No layout removes the pilot's residual seasonality at lags 11 and 12: every layout's Ljung–Box test rejects at 1% in all three closing columns (`runs[].ljung_box` in `docs/runs/stage-6/seasonal-comparison.json`). O17's posterior predictive suite owns it.

## Alternatives considered

- **`trig`.** It beat the dummy by +12.8 ± 8.6 on all 49 blocks and +10.5 ± 8.5 on the 47 reliable blocks, inside 2 standard errors on both, so clause (a) did not apply. It has the dummy's 247 states at Stage 9 low and saves no time, so clause (b), which needs fewer states, did not apply either. It meets Req 18 and passes its April 2020 check. Its Ljung–Box Q is lower than the dummy's (43, 28, 41 against 84, 49, 61), yet its test still rejects at 1%. Clause (c) kept it out.
- **`trig4`.** It lost to the dummy by −470.8 ± 67.6 on all blocks and −517.0 ± 26.1 on the reliable blocks, far outside 2 standard errors, so clause (b)'s elpd condition failed. Its Ljung–Box Q is 25 to 31 times the dummy's, far above the 1.25 limit. Its 214 states save 19.7% of the dummy's time per value and gradient, below the 25% minimum. It meets Req 18 and passes its April 2020 check. Clause (c) kept it out.

## Revisit triggers

- A posterior predictive check in O17 traces a misfit to the seasonal block.
