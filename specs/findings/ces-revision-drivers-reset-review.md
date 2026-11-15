# Reset review for the ces-revision-drivers roadmap

Reviewed 2026-11-15 at `1c8f15e`, read-only. It replaces [the old roadmap](../completed/ces-revisions-roadmap.md) (5 of 27 stages ticked) with [specs/ces-revision-drivers-roadmap.md](../ces-revision-drivers-roadmap.md).

## What works today

| Piece | Real data | Artifact | Evidence |
|---|---|---|---|
| Release-date index (plan 5) | yes | `data/panel/release_index.parquet`: 572 releases, 1979-02-02 to 2026-09-04 | reproduces BLS's dates |
| Vintage panel and same-release F/S/T changes (plan 5) | yes | `data/panel/panel.parquet` (6.34 M rows), total nonfarm 1979-01 to 2026-08; `stage_changes.parquet`, 12 sectors, 2003-05 to 2026-08 | reproduces every row of BLS's revision table |
| SA = NSA + seasonal accounting decomposition (plan 5) | yes | `data/panel/accounting_decomposition.parquet` (110 rows) | Req 9 identity test |
| Data and archive inventory (plan 2) | yes | `docs/ces-revisions-review.md`, `docs/inventory/` | structural tests |
| Benchmark, birth–death, QCEW and sample tables (plan 6) | yes | `data/annual/panel/` | `observable_at` test |
| Benchmark wedge, post-March and B→M operators (plan 8) | yes | `data/operators/` | reproduces the published B vintage, 2003–2025 |
| Kalman engine (plan 1) | no, synthetic only on `main` | `src/ces_revisions/kalman.py` | dense oracle |
| Aggregate news/noise pilot (plan 11, Tasks 1–12 of 14) | yes, branch `claude/ces-revisions-stage-6-bd9ab7` only | `docs/runs/stage-6/` on that branch | Req 18 met, 1,660 s sampling, SBC 100 of 100 |
| Fetch scripts | inputs only | `data/raw/`, `data/annual/raw/` | — |

No piece reported revision size against the collection window.

## Causes, largest first

1. **Scope: the old order follows the model.** The first stage that reports a driver's effect was Stage 17, behind 11 unticked stages. The old roadmap's reason ("the estimand layer cannot be validated before the Gaussian core's news/noise and wedge-rank identification is proven") holds for the Req 8 coefficient, not for a descriptive answer. Stages 1–5 took one calendar day each, so pace per stage was not the cause.
2. **Scope: the cloud GPU detour.** A resume on 2026-09-13 made Stage 6 wait for an AWS environment. No commit reached `main` from 2026-09-17 to 2026-10-20. The reversal deleted 6,012 lines, including a 5,195-line plan. Work on the deleted codex branches cannot be counted.
3. **Scope: final-standard qualities before the first answer.** In Stages 1–16, SBC is named in 8 entries, prior predictive in 7, provenance in 5, `observable_at` in 4, and content hashes or manifests in 3.
4. **Process: plans outweigh code.** The Stage 1–6 plans have 1,727 to 10,383 lines. Plans on `main` hold 18,480 lines against 6,720 lines of non-test source. Plan churn (29,818 lines) is more than twice all code churn (13,360). Plan 11 needed 7 revision commits and 8 owner decisions before its first task.

External limits were not a main cause. No rate limit is measured. BLS publishes no first-closing dates, so `D_t` is a proxy.
