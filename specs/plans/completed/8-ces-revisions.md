# Stage 5 — Benchmark, Aggregation, and Seasonal-Mapping Operators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via subagent-driven-development (the default) — or executing-plans when your human partner chose inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status: COMPLETE (2026-09-16)** — executed via executing-plans; nothing deferred

**Goal:** Ship the sparse, float64 JAX operator layer that later CES state-space models use for sector aggregation, benchmark wedge-back, post-March propagation, joint NSA/SA measurement, B→M components, March selection, and missing-vintage masking, with every empirical operator checked against the committed Stage 3 and Stage 4 BLS evidence.

**Architecture:** Add a focused `ces_revisions.operators` package. Small JAX `BCOO` builders express the mathematical maps; separate Polars builders align the committed release-vintage data to those maps and write deterministic diagnostic artifacts under the gitignored `data/operators/`. Benchmark validation is release-centric: for benchmark year `y`, the December `y` vintage is the pre-benchmark input and the January `y+1` vintage is the published benchmark output. Because the public archive contains levels and net birth–death forecasts but no matched-sample link relatives, the chosen post-March representation for all 2003–2025 years is an inferred cumulative job-change operator, with a positive reconstruction-error variance recorded for every year.

**Tech Stack:** Python 3.14; JAX 0.11.1 with `jax.experimental.sparse.BCOO` and float64 enabled before the first JAX operation; Polars 1.44.2; NumPy 2.5.3 for test assertions; pytest 9.1.1; ruff 0.16.7; the existing Stage 3 and Stage 4 offline source builders. No new dependency.

## Global Constraints

- **Scope:** Implement only roadmap Stage 5, plus the two omitted 2018 rows required to make Stage 4's reconstruction-event table the promised single source of documented reconstructions. Do not add a benchmark likelihood, latent states, priors, a fit, X-13 reproduction, or a final B→M probabilistic composition; those remain in Stages 8, 9, and 25.
- **Python:** Keep `requires-python = ">=3.14"`; add no dependency and do not change `pyproject.toml` or `uv.lock`.
- **JAX policy:** Every numerical operator consumes and returns float64 JAX arrays. `tests/conftest.py` remains the one place that calls `numpyro.enable_x64()` before JAX work. The public operator constructors return `jax.experimental.sparse.BCOO`; tests may densify only to inspect a small matrix.
- **Axis contracts:** Sector-state input order is exactly `("10", "20", "30", "40", "50", "55", "60", "65", "70", "80", "90")`; reconciled observation order is exactly `("00", *SECTOR_ORDER)`. NSA/SA observation order is exactly `("NSA", "SA")`; latent seasonal-state order is exactly `("x", "q")`.
- **Benchmark-year clock:** For benchmark year `y`, use the April `y-1` through March `y` backward window, the December `y` pre-benchmark vintage, and the January `y+1` benchmark vintage. The post-March window is April through December `y`; November and December carry `additional_sample_receipts=True` and are not counted in the April–October recoverability gate.
- **Req 10 wedge:** The sparse map is `(I - W e_March') y_old + W b_fin`, with `W=(1/12,...,1)`, where `b_fin` is the scope-comparable benchmark anchor before separately documented reconstruction. `Rκ` may be nonzero only where a row of Stage 4's `reconstruction_events` names that benchmark year and sector and its closed `reference_start`/`reference_end` interval contains the month; a null bound is open. The aggregate residual left by applying a basic-cell BLS procedure to rounded supersector data is stored separately as `rounding_residual_thousands`; the resulting published March level, not `b_fin`, initializes the post-March map.
- **Req 10 representation decision:** The selected empirical representation is `cumulative_job_change` for every benchmark year 2003–2025, and every selected series is `inferred`. The link-relative representation remains a tested public JAX interface, but no empirical year is assigned to it. Exactly one representation may be supplied to an application call.
- **Reconstruction variance:** `ROUNDING_VARIANCE_THOUSANDS2 = 1 / 6`, the variance of the difference of two independently rounded whole-thousand levels. For 2004–2025 use `max(mean(link_attempt_residual**2), 1/6)` over the 84 April–October sector cells; use `1/6` for 2003, whose old forecast table is absent. The value is positive exactly when `series_status == "inferred"`.
- **Req 11:** Birth–death quantities are NSA thousands. Revised (`post_benchmark`) birth–death values enter the selected post-March operator; never add them to an SA change. Government remains a structural zero supplied by Stage 4.
- **Req 5:** The published total remains its own observation. The aggregation operator maps the eleven sector states to a twelve-row vector containing the calculated total plus each sector; it does not replace or silently overwrite the published total.
- **Req 9:** The per-cell mature mapping is `NSA=x`, `SA=x-q`, and implied adjustment is `NSA-SA=q`. For a release-to-release pair, assert `ΔSA = ΔNSA - Δ(NSA-SA)` wherever all four published levels exist.
- **Missingness:** Preserve all four Stage 3 statuses: `observed`, `missing_vintage`, `beyond_frontier`, and `right_censored`. Only `observed` is selected into an observation vector; masked values are NaN, not zero.
- **Sources and provenance:** Do not modify `data/raw/` or downloaded files under `data/annual/raw/bls/`. The only source edit is the auditable append to `data/annual/raw/manual/reconstruction-events.csv`, followed by the existing offline Stage 4 manifest command. Every derived operator row retains the source cell IDs/keys it consumed.
- **Artifacts:** Derived parquet and JSON live under gitignored `data/operators/`. `operator-test-results.json` is deterministic—no wall-clock timestamp—and records both input manifest hashes, row counts, content hashes, and the Stage 5 gate summaries. Stage 6 and later run harnesses copy this whole JSON object into run provenance.
- **Tests:** Every new test is hermetic and unmarked. The default gate is `uv run pytest -m "not slow and not network"`; the existing network and slow tests remain deselected.
- **Ruff and Markdown:** Keep the default Ruff rules plus `extend-select = ["I", "B", "UP"]`. Use the repository's GitHub-renderable math conventions in this plan and in `docs/decisions/link-relatives.md`.

---

## Source and numbering

- Source spec: [`specs/ces-revisions.md`](../../ces-revisions.md), Req 5, Req 9, Req 10, Req 11, Req 17 step (1), and Verification bullets 2–3.
- Roadmap stage: [`specs/ces-revisions-roadmap.md`](../../ces-revisions-roadmap.md), Stage 5.
- Inputs: completed plans 5 and 6 under `specs/plans/completed/`, the Stage 3 source archive and builders, and the Stage 4 benchmark/reconstruction/birth–death builders.
- Plan id: existing ids are 1, 2, 5, 6, and 7; therefore this is plan **8**.
- Retirement: this shared spec does not retire with Stage 5. On completion, tick Stage 5 and add the Stage 5 completion stamp to the spec's Rollout note; then resume the roadmap.

## Scope check

The stage names four mathematical areas, but they are one deployable subsystem: each is a sparse observation/preprocessing operator over the same sector, release, seasonal-status, and missingness axes, and Stages 6–9 consume them through one provenance record. Splitting the work into independent plans would duplicate axis ordering, masks, source alignment, and artifact hashing. The modules remain separate, and each task below has its own focused test cycle and review gate.

Stage 3 already implements and tests Req 17's same-release differencing operator. Stage 5 composes with that implementation and reruns its tests in the full verification gate; it must not introduce a second differencing path. The Stage 5 provenance record hashes the Stage 3 source manifest alongside the new operator artifacts so later runs identify the exact differencing inputs they inherited.

## Planning evidence (2026-09-16)

- `main` is clean. The fast hermetic baseline is **324 passed, 17 deselected**, with the two already-documented Dynamax dependency warnings.
- Stage 3 builds 6,342,530 panel rows and 37,032 stage labels. Its eleven supersectors sum exactly to total nonfarm in all **496,310** nonmissing release/status/reference-month cells.
- Stage 4 builds 327 benchmark rows, 14 reconstruction-event rows, and 7,074 birth–death rows. The 2018 benchmark article and Stage 3 Comments sheet document two further reconstruction events—wholesale recoding and a state-to-private ownership change—but Stage 4's structured event table omits them. The operator support test needs those two rows, bringing the event table to 16 rows.
- The December-to-January adjacent vintage pair supplies all twelve backward-window levels for all twelve sectors in every benchmark year 2003–2025, including 2003. The resulting NSA fixture has `23 * 12 * 12 = 3,312` rows.
- The Stage 4 article revision and the archive's March release-to-release gap disagree only in documented reconstruction/adjustment cells, plus the already-preserved 2025 article/Table-5 one-thousand discrepancy. One article cell, 2023 mining and logging, prints only footnote `(1)` and has a null numeric revision; its effective March gap must be read from the archive and labeled `inferred_from_archive`.
- After the 2018 events are added, 215 fixture rows lie inside a documented event's sector **and reference-date range**, and 3,097 do not. Assigning supported residuals to the combined reconstruction remainder `Rκ` and other residuals to aggregate `ε_round` leaves **no** unexplained reconstruction term. `Rκ` is support-bounded rather than a causal estimate of any individual event because the articles do not identify a basic-series allocation within a supported aggregate cell. The largest non-event aggregate rounding residual is 21.0 thousand; the largest event-supported term is 469.1666667 thousand.
- The vintage triangles store published levels, not matched-sample link relatives. Stage 4 stores initial and revised birth–death forecasts. For 2004–2025 an aggregate link proxy can be inferred as `(old_level_m - initial_BD_m) / old_level_{m-1}`, but applying it to the new March anchor with revised birth–death fails the ±0.5-thousand publication-rounding gate in every year. The 2003 initial April–December schedule is not published at all.
- Therefore all 23 years select the cumulative representation. For each target benchmark vintage, infer `sample_job_change_m = level_m - level_{m-1} - revised_BD_m`; the sparse cumulative operator then reproduces all 2,484 April–December sector cells exactly, of which 1,932 April–October cells form the formal exit gate. This is an operator reconstruction, not evidence that the private matched-sample link was observed.
- The recorded recoverability diagnostics are:

| Year | Link attempt | Attempted | Within ±0.5 | Maximum absolute residual | Reconstruction variance |
|---:|---|---:|---:|---:|---:|
| 2003 | inputs missing | 0 | 0 | — | 0.166667 |
| 2004 | failed | 84 | 19 | 33.407275 | 65.199745 |
| 2005 | failed | 84 | 20 | 61.890801 | 228.730081 |
| 2006 | failed | 84 | 15 | 23.731192 | 51.300815 |
| 2007 | failed | 84 | 16 | 34.228674 | 58.660143 |
| 2008 | failed | 84 | 17 | 22.098039 | 18.249807 |
| 2009 | failed | 84 | 13 | 40.264512 | 78.661208 |
| 2010 | failed | 84 | 12 | 31.813475 | 91.563345 |
| 2011 | failed | 84 | 16 | 20.359022 | 35.259417 |
| 2012 | failed | 84 | 20 | 78.412368 | 520.337201 |
| 2013 | failed | 84 | 11 | 37.113240 | 48.080295 |
| 2014 | failed | 84 | 19 | 26.630662 | 68.188332 |
| 2015 | failed | 84 | 8 | 63.751727 | 331.085586 |
| 2016 | failed | 84 | 13 | 32.440472 | 81.254024 |
| 2017 | failed | 84 | 14 | 8.913966 | 7.778542 |
| 2018 | failed | 84 | 11 | 24.771480 | 93.717486 |
| 2019 | failed | 84 | 17 | 13.991568 | 16.178823 |
| 2020 | failed | 84 | 3 | 162.058501 | 1607.645923 |
| 2021 | failed | 84 | 12 | 179.079008 | 1293.123145 |
| 2022 | failed | 84 | 5 | 43.910696 | 157.995551 |
| 2023 | failed | 84 | 14 | 74.270533 | 433.893346 |
| 2024 | failed | 84 | 18 | 38.587326 | 108.426410 |
| 2025 | failed | 84 | 11 | 17.229047 | 30.739396 |

## Execution notes

- **Track the handoff first:** This plan is the execution source of truth. If it is still untracked, commit it on the starting branch with `git add specs/plans/8-ces-revisions.md` and `git commit -m "docs: plan Stage 5 operators"` before creating an execution worktree.
- **Isolation:** At execution time, invoke `using-git-worktrees` and create a `codex/` feature branch from the then-current `main`; do not execute this multi-task plan in the user's primary checkout.
- **Inputs are offline:** Tasks 1–6 use only committed Stage 3/4 sources. The manifest refresh command is offline. Do not fetch replacement BLS or Internet Archive files during this plan.
- **Review boundary:** Each task ends in a focused green test cycle and a commit suitable for a task-scoped review. Do not batch neighboring tasks into one commit.
- **Stop conditions:** Stop and report exact source rows instead of changing a constant when the 2018 events cannot be reproduced from the committed article, the 3,312-cell wedge fixture changes, a purported reconstruction falls outside documented sector/date support, any link proxy unexpectedly passes, revised birth–death does not join uniquely, the NSA/SA identity fails, or an existing Stage 3/4 contract regresses.

## File structure

| Path | Responsibility | Task |
|---|---|---:|
| `data/annual/raw/manual/reconstruction-events.csv` | Add the two documented 2018 reconstruction events | 1 |
| `data/annual/raw/manifest.csv` | Rehash the edited manual source through the existing offline command | 1 |
| `tests/test_reconstructions.py` | Pin the complete 16-event set and the two 2018 rows | 1 |
| `src/ces_revisions/operators/__init__.py` | Package boundary and axis-order exports | 2 |
| `src/ces_revisions/operators/aggregation.py` | Sparse sector-to-total reconciliation operator | 2 |
| `src/ces_revisions/operators/masks.py` | Missing-vintage, observed-cell, and March selection operators | 2 |
| `tests/test_operator_foundations.py` | Sparse/JIT, full aggregation, mask, and selector tests | 2 |
| `src/ces_revisions/operators/benchmark.py` | Wedge matrix, reconstruction support, and published benchmark fixtures | 3 |
| `tests/test_benchmark_operators.py` | Formula, article/archive alignment, reconstruction support, and 2003–2025 fixture tests | 3 |
| `src/ces_revisions/operators/post_march.py` | Link-relative and cumulative operators, empirical components, and recoverability table | 4 |
| `tests/test_post_march_operators.py` | Both representations, exclusivity, revised birth–death, residual, and variance tests | 4 |
| `src/ces_revisions/operators/seasonal.py` | Joint NSA/SA maps and release-specific B→M components | 5 |
| `tests/test_seasonal_operators.py` | Seasonal identity and B→M component/mask tests | 5 |
| `src/ces_revisions/operators/build.py` | Assemble/write Stage 5 artifacts and deterministic test record | 6 |
| `scripts/operator_artifacts.py` | Offline build CLI | 6 |
| `tests/operator_data.py` | Session-cached Stage 5 build for tests | 6 |
| `tests/test_operator_build.py` | Artifact, manifest, and test-record round trip | 6 |
| `docs/decisions/link-relatives.md` | Accepted public-data representation and per-year determination | 6 |
| `.gitignore` | Ignore rebuilt `data/operators/` | 6 |
| `README.md`, `AGENTS.md`, `CLAUDE.md` | Document the operator layer, command, contracts, and decision | 6 |

### Task 1: Complete the Structured Reconstruction Inventory for 2018

**Files:**
- Modify: `tests/test_reconstructions.py`
- Modify: `data/annual/raw/manual/reconstruction-events.csv`
- Modify, generated offline: `data/annual/raw/manifest.csv`

**Interfaces:**
- Consumes: `annual.reconstructions.build_reconstruction_events(calendar, raw_dir) -> pl.DataFrame` and its existing generic CSV parser.
- Produces: event ids `2018-wholesale-recoding` and `2018-state-ownership-change`; event support on sectors `40,60` and `65,90`, respectively; a 16-row reconstruction-event table whose publication clock remains the final 2018 benchmark release.

- [x] **Step 1: Write the failing 2018 event tests**

In `tests/test_reconstructions.py`, add the two ids to `expected` and add this test:

```python
def test_2018_reconstructions_cover_both_documented_sector_transfers():
    rows = built().filter(pl.col("benchmark_year") == 2018)
    assert rows.select("event_id", "sectors", "effect_thousands").rows() == [
        ("2018-state-ownership-change", ["65", "90"], 17.0),
        ("2018-wholesale-recoding", ["40", "60"], 336.0),
    ]
    assert rows["reference_start"].to_list() == [
        date(2018, 1, 1),
        date(1990, 1, 1),
    ]
```

Add `from datetime import date` above the Polars import. The complete `expected` set must include the existing fourteen ids plus `"2018-state-ownership-change"` and `"2018-wholesale-recoding"`.

- [x] **Step 2: Run the focused test to verify it fails**

Run: `uv run pytest tests/test_reconstructions.py -q`

Expected: `test_expected_documented_events_exist_once` and `test_2018_reconstructions_cover_both_documented_sector_transfers` fail because the built frame has no 2018 rows.

- [x] **Step 3: Append the exact source rows**

> Deviation: The two description fields were CSV-quoted because their source text contains commas; blank footnote fields are normalized to null and pinned by tests.

Append these rows to `data/annual/raw/manual/reconstruction-events.csv`, keeping the file sorted by benchmark year and event id:

```csv
2018-state-ownership-change,2018,classification_reconstruction,65;90,65-624120;90-922999,2018-01-01,2018-03-01,17,,A state employer was reclassified to private ownership; about 17 thousand moved from other state government to services for the elderly and persons with disabilities and the affected series were rebuilt for January through March 2018.,bls/benchmark/ces-benchmark-revision-2018.pdf,"Ownership change in state government, PDF page 19"
2018-wholesale-recoding,2018,classification_reconstruction,40;60,41-425120,1990-01-01,2018-03-01,336,,About 336 thousand moved out of wholesale trade agents and brokers into wholesale trade, retail trade, transportation and warehousing, and professional and business services; affected basic series were rebuilt generally to January 1990.,bls/benchmark/ces-benchmark-revision-2018.pdf,"Wholesale trade recoding, PDF pages 15–19"
```

Do not assign a Table 5 footnote: the 2018 article documents the events directly and its Table 3 footnote says the reconstruction is already incorporated.

- [x] **Step 4: Refresh the Stage 4 manifest offline**

Run: `uv run python scripts/annual_sources.py manifest`

Expected: exit 0; `data/annual/raw/manifest.csv` is rewritten, and its row for `manual/reconstruction-events.csv` has a new SHA-256. No network request occurs.

- [x] **Step 5: Run the focused and Stage 4 contract tests**

Run:

```bash
uv run pytest tests/test_reconstructions.py tests/test_annual_raw.py tests/test_annual_contract.py -q
uv run ruff format tests/test_reconstructions.py
uv run ruff check tests/test_reconstructions.py src/ces_revisions/annual
```

Expected: all focused tests pass; `build_reconstruction_events()` returns 16 unique ids, the 2018 rows use the final-benchmark publication timestamp, and the manifest test passes.

- [x] **Step 6: Commit the completed event inventory**

```bash
git add tests/test_reconstructions.py data/annual/raw/manual/reconstruction-events.csv data/annual/raw/manifest.csv
git commit -m "fix(annual): record 2018 benchmark reconstructions"
```

### Task 2: Establish Sparse Axis, Aggregation, and Mask Contracts

**Files:**
- Create: `src/ces_revisions/operators/__init__.py`
- Create: `src/ces_revisions/operators/aggregation.py`
- Create: `src/ces_revisions/operators/masks.py`
- Test: `tests/test_operator_foundations.py`

**Interfaces:**
- Consumes: `ces_revisions.vintages.raw.SUPERSECTORS`, Stage 3 level columns, stage-label statuses, and Python `date` reference months.
- Produces:
  - `SECTOR_ORDER: tuple[str, ...]` and `OBSERVATION_SECTOR_ORDER: tuple[str, ...]`.
  - `aggregation_operator() -> BCOO`, shape `(12, 11)`.
  - `aggregate_sector_states(sector_values: jax.Array) -> jax.Array`, preserving arbitrary leading dimensions and replacing the last 11-sector axis with the 12-row observation axis.
  - `observed_mask(statuses) -> jax.Array`, `mask_values(values, statuses) -> jax.Array`, `selection_operator(mask) -> BCOO`, `march_mask(reference_months) -> jax.Array`, and `march_selection_operator(reference_months) -> BCOO`.

- [x] **Step 1: Write the failing sparse-foundation tests**

Create `tests/test_operator_foundations.py`:

```python
"""Sparse aggregation, missing-vintage masks, and March selection."""

from datetime import date

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from jax.experimental.sparse import BCOO

import vintage_data
from ces_revisions.operators import OBSERVATION_SECTOR_ORDER, SECTOR_ORDER
from ces_revisions.operators.aggregation import (
    aggregate_sector_states,
    aggregation_operator,
)
from ces_revisions.operators.masks import (
    march_mask,
    march_selection_operator,
    mask_values,
    observed_mask,
    selection_operator,
)


def test_axis_orders_are_the_req_5_supersectors_and_published_total():
    assert SECTOR_ORDER == (
        "10",
        "20",
        "30",
        "40",
        "50",
        "55",
        "60",
        "65",
        "70",
        "80",
        "90",
    )
    assert OBSERVATION_SECTOR_ORDER == ("00", *SECTOR_ORDER)


def test_aggregation_operator_is_sparse_float64_and_jittable():
    operator = aggregation_operator()
    assert isinstance(operator, BCOO)
    assert operator.shape == (12, 11)
    assert operator.dtype == jnp.float64
    values = jnp.arange(1.0, 12.0, dtype=jnp.float64)
    result = jax.jit(lambda x: operator @ x)(values)
    np.testing.assert_array_equal(result[0], values.sum())
    np.testing.assert_array_equal(result[1:], values)


def test_aggregation_reproduces_every_published_total_nonfarm_level():
    levels = vintage_data.levels().filter(pl.col("value_thousands").is_not_null())
    keys = ["release_month", "seasonal_status", "reference_month"]
    wide = levels.pivot(on="sector", index=keys, values="value_thousands").sort(keys)
    assert wide.height == 496_310
    calculated = np.asarray(
        aggregate_sector_states(
            jnp.asarray(wide.select(*SECTOR_ORDER).to_numpy(), dtype=jnp.float64)
        )
    )
    np.testing.assert_array_equal(calculated[:, 0], wide["00"].to_numpy())
    np.testing.assert_array_equal(
        calculated[:, 1:], wide.select(*SECTOR_ORDER).to_numpy()
    )


def test_masks_preserve_each_missingness_reason_and_never_zero_fill():
    statuses = [
        "observed",
        "missing_vintage",
        "beyond_frontier",
        "right_censored",
    ]
    mask = observed_mask(statuses)
    np.testing.assert_array_equal(mask, [True, False, False, False])
    masked = mask_values(jnp.arange(4.0, dtype=jnp.float64), statuses)
    assert masked[0] == 0.0
    assert jnp.isnan(masked[1:]).all()
    selected = selection_operator(mask) @ jnp.arange(4.0, dtype=jnp.float64)
    np.testing.assert_array_equal(selected, [0.0])


def test_march_selection_is_sparse_and_keeps_only_march_rows():
    months = [date(2024, 2, 1), date(2024, 3, 1), date(2025, 3, 1)]
    np.testing.assert_array_equal(march_mask(months), [False, True, True])
    operator = march_selection_operator(months)
    assert isinstance(operator, BCOO)
    np.testing.assert_array_equal(
        operator @ jnp.asarray([2.0, 3.0, 4.0], dtype=jnp.float64),
        [3.0, 4.0],
    )
```

- [x] **Step 2: Run the tests to verify import failure**

Run: `uv run pytest tests/test_operator_foundations.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'ces_revisions.operators'`.

- [x] **Step 3: Create the package boundary and aggregation operator**

Create `src/ces_revisions/operators/__init__.py`:

```python
"""Sparse deterministic operators shared by CES data preparation and models."""

from ces_revisions.vintages.raw import SUPERSECTORS

SECTOR_ORDER = tuple(SUPERSECTORS)
OBSERVATION_SECTOR_ORDER = ("00", *SECTOR_ORDER)

__all__ = ["OBSERVATION_SECTOR_ORDER", "SECTOR_ORDER"]
```

Create `src/ces_revisions/operators/aggregation.py`:

```python
"""Req 5 sector aggregation while retaining the published total as an observation."""

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

from ces_revisions.operators import SECTOR_ORDER


def aggregation_operator() -> BCOO:
    """Map eleven sector states to calculated total plus the eleven identities."""
    count = len(SECTOR_ORDER)
    dense = jnp.concatenate(
        [jnp.ones((1, count), dtype=jnp.float64), jnp.eye(count, dtype=jnp.float64)]
    )
    return BCOO.fromdense(dense)


def aggregate_sector_states(sector_values: jax.Array) -> jax.Array:
    """Apply the aggregation map to the last axis of a float64 array."""
    values = jnp.asarray(sector_values, dtype=jnp.float64)
    if values.ndim == 0 or values.shape[-1] != len(SECTOR_ORDER):
        raise ValueError(f"sector_values must end in {len(SECTOR_ORDER)} sectors")
    flat = values.reshape((-1, len(SECTOR_ORDER))).T
    aggregated = (aggregation_operator() @ flat).T
    return aggregated.reshape((*values.shape[:-1], len(SECTOR_ORDER) + 1))
```

- [x] **Step 4: Implement explicit observation and March masks**

Create `src/ces_revisions/operators/masks.py`:

```python
"""Sparse selectors for observed cells and irregular March benchmark rows."""

from collections.abc import Sequence
from datetime import date

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO

VALID_STATUSES = (
    "observed",
    "missing_vintage",
    "beyond_frontier",
    "right_censored",
)


def observed_mask(statuses: Sequence[str]) -> jax.Array:
    unknown = sorted(set(statuses) - set(VALID_STATUSES))
    if unknown:
        raise ValueError(f"unknown stage-label statuses: {unknown}")
    return jnp.asarray([status == "observed" for status in statuses], dtype=jnp.bool_)


def mask_values(values: jax.Array, statuses: Sequence[str]) -> jax.Array:
    array = jnp.asarray(values, dtype=jnp.float64)
    mask = observed_mask(statuses)
    if array.shape[-1] != mask.shape[0]:
        raise ValueError("values and statuses have different cell counts")
    return jnp.where(mask, array, jnp.nan)


def selection_operator(mask: Sequence[bool] | jax.Array) -> BCOO:
    flags = np.asarray(mask, dtype=bool)
    dense = jnp.eye(flags.size, dtype=jnp.float64)[flags]
    return BCOO.fromdense(dense)


def march_mask(reference_months: Sequence[date]) -> jax.Array:
    return jnp.asarray(
        [month.month == 3 for month in reference_months], dtype=jnp.bool_
    )


def march_selection_operator(reference_months: Sequence[date]) -> BCOO:
    return selection_operator(march_mask(reference_months))
```

- [x] **Step 5: Run and format the foundation tests**

Run:

```bash
uv run pytest tests/test_operator_foundations.py -q
uv run ruff format src/ces_revisions/operators tests/test_operator_foundations.py
uv run ruff check src/ces_revisions/operators tests/test_operator_foundations.py
```

Expected: 5 tests pass; the full 496,310-cell aggregation test has zero residual; Ruff is clean.

- [x] **Step 6: Commit the sparse foundation**

```bash
git add src/ces_revisions/operators tests/test_operator_foundations.py
git commit -m "feat(operators): add aggregation and observation masks"
```

### Task 3: Build and Verify the Benchmark Wedge Fixture

**Files:**
- Create: `src/ces_revisions/operators/benchmark.py`
- Test: `tests/test_benchmark_operators.py`

**Interfaces:**
- Consumes: Stage 3 `levels`, Stage 4 `benchmarks` and `reconstruction_events`, `SECTOR_ORDER`, and `vintages.months.month_range()`.
- Produces:
  - `wedge_weights() -> jax.Array`, `wedge_operator() -> BCOO`, and `apply_wedge(previous_levels, benchmark_level, reconstruction_terms=None, reconstruction_support=None, rounding_residuals=None) -> jax.Array`.
  - `reconstruction_selector(support) -> BCOO`.
  - `build_benchmark_fixtures(levels, benchmarks, reconstruction_events) -> pl.DataFrame`, exactly 3,312 rows with release/source lineage and separate reconstruction and aggregate-rounding terms.

- [x] **Step 1: Write the failing wedge and fixture tests**

Create `tests/test_benchmark_operators.py`:

```python
"""Req 10's backward wedge against the archived published benchmark vintages."""

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from jax.experimental.sparse import BCOO

import annual_data
import vintage_data
from ces_revisions.operators.benchmark import (
    apply_wedge,
    build_benchmark_fixtures,
    reconstruction_selector,
    wedge_operator,
    wedge_weights,
)


def fixtures() -> pl.DataFrame:
    return build_benchmark_fixtures(
        vintage_data.levels(),
        annual_data.artifact("benchmarks"),
        annual_data.artifact("reconstruction_events"),
    )


def test_wedge_operator_has_the_req_10_affine_coefficients():
    weights = wedge_weights()
    np.testing.assert_allclose(weights, np.arange(1, 13) / 12)
    operator = wedge_operator()
    assert isinstance(operator, BCOO)
    assert operator.shape == (12, 13)
    previous = jnp.zeros(12, dtype=jnp.float64)
    result = jax.jit(lambda x: operator @ x)(
        jnp.concatenate([previous, jnp.asarray([-861.0])])
    )
    np.testing.assert_allclose(result, -861.0 * np.arange(1, 13) / 12)
    assert result[0] == -71.75
    assert result[-1] == -861.0


def test_benchmark_fixture_covers_every_year_sector_and_backward_month():
    frame = fixtures()
    assert frame.height == 3_312
    assert frame["benchmark_year"].unique().sort().to_list() == list(range(2003, 2026))
    assert frame.group_by("benchmark_year", "sector").len()["len"].eq(12).all()
    assert frame.select("benchmark_year", "sector", "reference_month").n_unique() == (
        frame.height
    )


def test_fixture_reproduces_each_published_nsa_benchmark_vintage_exactly():
    frame = fixtures()
    np.testing.assert_allclose(
        frame["reproduced_level_thousands"],
        frame["benchmark_level_thousands"],
        rtol=0.0,
        atol=1e-9,
    )
    assert frame.filter(pl.col("reconstruction_supported")).height == 215
    assert frame.filter(~pl.col("reconstruction_supported")).height == 3_097
    assert (
        frame.filter(~pl.col("reconstruction_supported"))[
            "reconstruction_term_thousands"
        ]
        == 0.0
    ).all()
    assert (
        frame.filter(pl.col("reconstruction_supported"))["rounding_residual_thousands"]
        == 0.0
    ).all()
    assert (
        frame.filter(~pl.col("reconstruction_supported"))["rounding_residual_thousands"]
        .abs()
        .max()
        == 21.0
    )
    assert np.isclose(
        frame.filter(pl.col("reconstruction_supported"))[
            "reconstruction_term_thousands"
        ]
        .abs()
        .max(),
        469.1666666667,
    )


def test_march_reconstruction_terms_have_documented_event_support():
    march = fixtures().filter(pl.col("wedge_weight") == 1.0)
    nonzero = march.filter(pl.col("reconstruction_term_thousands") != 0.0)
    assert nonzero["event_ids"].list.len().gt(0).all()
    assert set(nonzero["benchmark_year"]) == {
        2010,
        2013,
        2015,
        2017,
        2018,
        2019,
        2022,
        2024,
        2025,
    }
    inferred = march.filter(pl.col("revision_status") == "inferred_from_archive")
    assert inferred.select("benchmark_year", "sector").rows() == [(2023, "10")]


def test_reconstruction_support_respects_each_events_reference_range():
    state_transfer = fixtures().filter(
        (pl.col("benchmark_year") == 2018)
        & pl.col("sector").is_in(["65", "90"])
        & pl.col("event_ids").list.contains("2018-state-ownership-change")
    )
    assert state_transfer["reference_month"].dt.month().unique().sort().to_list() == [
        1,
        2,
        3,
    ]


def test_reconstruction_selector_cannot_emit_outside_documented_support():
    support = [False, True, False, True]
    operator = reconstruction_selector(support)
    assert isinstance(operator, BCOO)
    result = operator @ jnp.asarray([3.0, 4.0, 5.0, 6.0], dtype=jnp.float64)
    np.testing.assert_array_equal(result, [0.0, 4.0, 0.0, 6.0])


def test_apply_wedge_requires_twelve_months_and_adds_terms_separately():
    previous = jnp.arange(12.0, dtype=jnp.float64)
    support = [False] * 11 + [True]
    reconstructed = apply_wedge(
        previous,
        23.0,
        reconstruction_terms=jnp.ones(12, dtype=jnp.float64),
        reconstruction_support=support,
        rounding_residuals=jnp.full(12, 0.25, dtype=jnp.float64),
    )
    expected = np.asarray(
        wedge_operator() @ jnp.concatenate([previous, jnp.array([23.0])])
    )
    np.testing.assert_allclose(reconstructed[:-1], expected[:-1] + 0.25)
    np.testing.assert_allclose(reconstructed[-1], expected[-1] + 1.25)
```

- [x] **Step 2: Run the tests to verify the missing module**

Run: `uv run pytest tests/test_benchmark_operators.py -q`

Expected: collection fails because `ces_revisions.operators.benchmark` does not exist.

- [x] **Step 3: Implement the sparse wedge and release clock**

Create `src/ces_revisions/operators/benchmark.py` with the imports, constants, and public numerical functions below:

```python
"""Req 10 benchmark wedge and published-vintage validation fixtures."""

from collections.abc import Sequence
from datetime import date

import jax
import jax.numpy as jnp
import polars as pl
from jax.experimental.sparse import BCOO

from ces_revisions.vintages.months import month_range

BENCHMARK_YEARS = tuple(range(2003, 2026))
WINDOW_MONTHS = 12


def wedge_weights() -> jax.Array:
    return jnp.arange(1, WINDOW_MONTHS + 1, dtype=jnp.float64) / WINDOW_MONTHS


def wedge_operator() -> BCOO:
    weights = wedge_weights()
    march = jnp.zeros(WINDOW_MONTHS, dtype=jnp.float64).at[-1].set(1.0)
    dense = jnp.concatenate(
        [
            jnp.eye(WINDOW_MONTHS, dtype=jnp.float64) - jnp.outer(weights, march),
            weights[:, None],
        ],
        axis=1,
    )
    return BCOO.fromdense(dense)


def reconstruction_selector(support: Sequence[bool]) -> BCOO:
    flags = jnp.asarray(support, dtype=jnp.bool_)
    return BCOO.fromdense(jnp.diag(flags.astype(jnp.float64)))


def apply_wedge(
    previous_levels: jax.Array,
    benchmark_level: float | jax.Array,
    *,
    reconstruction_terms: jax.Array | None = None,
    reconstruction_support: Sequence[bool] | None = None,
    rounding_residuals: jax.Array | None = None,
) -> jax.Array:
    previous = jnp.asarray(previous_levels, dtype=jnp.float64)
    if previous.shape != (WINDOW_MONTHS,):
        raise ValueError("previous_levels must hold April through March")
    inputs = jnp.concatenate(
        [previous, jnp.asarray(benchmark_level, dtype=jnp.float64).reshape(1)]
    )
    result = wedge_operator() @ inputs
    if (reconstruction_terms is None) != (reconstruction_support is None):
        raise ValueError(
            "reconstruction_terms and reconstruction_support must be supplied together"
        )
    if reconstruction_terms is not None:
        assert reconstruction_support is not None
        term = jnp.asarray(reconstruction_terms, dtype=jnp.float64)
        if term.shape != (WINDOW_MONTHS,):
            raise ValueError("reconstruction_terms must hold twelve months")
        if len(reconstruction_support) != WINDOW_MONTHS:
            raise ValueError("reconstruction_support must hold twelve months")
        result = result + reconstruction_selector(reconstruction_support) @ term
    if rounding_residuals is not None:
        rounding = jnp.asarray(rounding_residuals, dtype=jnp.float64)
        if rounding.shape != (WINDOW_MONTHS,):
            raise ValueError("rounding_residuals must hold twelve months")
        result = result + rounding
    return result


def benchmark_window(year: int) -> tuple[date, ...]:
    if year not in BENCHMARK_YEARS:
        raise ValueError(f"benchmark year outside 2003–2025: {year}")
    return tuple(month_range(date(year - 1, 4, 1), date(year, 3, 1)))


def prebenchmark_release_month(year: int) -> date:
    return date(year, 12, 1)


def benchmark_release_month(year: int) -> date:
    return date(year + 1, 1, 1)
```

- [x] **Step 4: Implement the audited benchmark fixture builder**

In the same module, add the schema and builder. Use Python records deliberately: there are only 3,312 rows, and the explicit loop makes the release clock and source lineage reviewable.

```python
BENCHMARK_FIXTURE_SCHEMA = {
    "benchmark_year": pl.Int32,
    "sector": pl.String,
    "reference_month": pl.Date,
    "prebenchmark_release_month": pl.Date,
    "benchmark_release_month": pl.Date,
    "previous_level_thousands": pl.Float64,
    "benchmark_level_thousands": pl.Float64,
    "published_revision_thousands": pl.Float64,
    "effective_revision_thousands": pl.Float64,
    "revision_status": pl.String,
    "wedge_weight": pl.Float64,
    "reconstruction_supported": pl.Boolean,
    "event_ids": pl.List(pl.String),
    "linear_wedge_level_thousands": pl.Float64,
    "reconstruction_term_thousands": pl.Float64,
    "rounding_residual_thousands": pl.Float64,
    "reproduced_level_thousands": pl.Float64,
    "previous_cell_id": pl.Int64,
    "benchmark_cell_id": pl.Int64,
    "benchmark_source_keys": pl.List(pl.String),
}


def _only(frame: pl.DataFrame, label: str) -> dict:
    if frame.height != 1:
        raise ValueError(f"expected one {label}, found {frame.height}")
    return frame.row(0, named=True)


def _benchmark_row(benchmarks: pl.DataFrame, year: int, sector: str) -> dict:
    row_kind = "anchor" if sector == "00" else "sector_contribution"
    return _only(
        benchmarks.filter(
            (pl.col("benchmark_year") == year)
            & (pl.col("benchmark_status") == "final")
            & (pl.col("row_kind") == row_kind)
            & (pl.col("sector") == sector)
        ),
        f"final benchmark row for {year}/{sector}",
    )


def _event_ids(
    events: pl.DataFrame, year: int, sector: str, reference_month: date
) -> list[str]:
    return sorted(
        row["event_id"]
        for row in events.filter(pl.col("benchmark_year") == year).iter_rows(named=True)
        if sector in row["sectors"]
        and (
            row["reference_start"] is None or reference_month >= row["reference_start"]
        )
        and (row["reference_end"] is None or reference_month <= row["reference_end"])
    )


def build_benchmark_fixtures(
    levels: pl.DataFrame,
    benchmarks: pl.DataFrame,
    reconstruction_events: pl.DataFrame,
) -> pl.DataFrame:
    nsa = levels.filter(pl.col("seasonal_status") == "NSA")
    records = []
    for year in BENCHMARK_YEARS:
        old_release = prebenchmark_release_month(year)
        new_release = benchmark_release_month(year)
        months = benchmark_window(year)
        for sector in (
            "00",
            "10",
            "20",
            "30",
            "40",
            "50",
            "55",
            "60",
            "65",
            "70",
            "80",
            "90",
        ):
            old = nsa.filter(
                (pl.col("sector") == sector)
                & (pl.col("release_month") == old_release)
                & pl.col("reference_month").is_in(months)
            ).sort("reference_month")
            new = nsa.filter(
                (pl.col("sector") == sector)
                & (pl.col("release_month") == new_release)
                & pl.col("reference_month").is_in(months)
            ).sort("reference_month")
            if old.height != WINDOW_MONTHS or new.height != WINDOW_MONTHS:
                raise ValueError(f"incomplete benchmark fixture for {year}/{sector}")
            benchmark = _benchmark_row(benchmarks, year, sector)
            march_gap = float(new[-1, "value_thousands"] - old[-1, "value_thousands"])
            published = benchmark["revision_thousands"]
            effective = march_gap if published is None else float(published)
            status = "inferred_from_archive" if published is None else "published"
            for index, (old_row, new_row) in enumerate(
                zip(old.iter_rows(named=True), new.iter_rows(named=True), strict=True),
                start=1,
            ):
                event_ids = _event_ids(
                    reconstruction_events,
                    year,
                    sector,
                    old_row["reference_month"],
                )
                weight = index / WINDOW_MONTHS
                linear = float(old_row["value_thousands"]) + weight * effective
                residual = float(new_row["value_thousands"]) - linear
                reconstruction = residual if event_ids else 0.0
                rounding = 0.0 if event_ids else residual
                records.append(
                    {
                        "benchmark_year": year,
                        "sector": sector,
                        "reference_month": old_row["reference_month"],
                        "prebenchmark_release_month": old_release,
                        "benchmark_release_month": new_release,
                        "previous_level_thousands": float(old_row["value_thousands"]),
                        "benchmark_level_thousands": float(new_row["value_thousands"]),
                        "published_revision_thousands": published,
                        "effective_revision_thousands": effective,
                        "revision_status": status,
                        "wedge_weight": weight,
                        "reconstruction_supported": bool(event_ids),
                        "event_ids": event_ids,
                        "linear_wedge_level_thousands": linear,
                        "reconstruction_term_thousands": reconstruction,
                        "rounding_residual_thousands": rounding,
                        "reproduced_level_thousands": linear
                        + reconstruction
                        + rounding,
                        "previous_cell_id": old_row["cell_id"],
                        "benchmark_cell_id": new_row["cell_id"],
                        "benchmark_source_keys": benchmark["source_keys"],
                    }
                )
    return pl.DataFrame(records, schema=BENCHMARK_FIXTURE_SCHEMA).sort(
        "benchmark_year", "sector", "reference_month"
    )
```

- [x] **Step 5: Run the fixture tests and inspect only named discrepancies**

Run:

```bash
uv run pytest tests/test_benchmark_operators.py -q
uv run ruff format src/ces_revisions/operators/benchmark.py tests/test_benchmark_operators.py
uv run ruff check src/ces_revisions/operators tests/test_benchmark_operators.py
```

Expected: 7 tests pass. If the 215/3,097 support split, 21.0 rounding envelope, 469.1666667 supported maximum, or 2023/sector-10 inferred cell changes, stop and report the exact source rows; do not widen a tolerance or relabel a non-event residual as reconstruction.

- [x] **Step 6: Commit the wedge and fixture**

```bash
git add src/ces_revisions/operators/benchmark.py tests/test_benchmark_operators.py
git commit -m "feat(operators): add benchmark wedge fixtures"
```

### Task 4: Implement Both Post-March Maps and Settle Recoverability

**Files:**
- Create: `src/ces_revisions/operators/post_march.py`
- Test: `tests/test_post_march_operators.py`

**Interfaces:**
- Consumes: Stage 3 NSA release levels; Stage 4 `published_schedule` birth–death rows; the release clock from `operators.benchmark`.
- Produces:
  - `post_march_link_operator(link_relatives) -> BCOO`, shape `(n, 1+n)` over `[March anchor, revised BD]`.
  - `post_march_cumulative_operator(month_count) -> BCOO`, shape `(n, 1+2n)` over `[March anchor, sample job changes, revised BD]`.
  - `apply_post_march(anchor, revised_birth_death, *, link_relatives=None, sample_job_changes=None) -> jax.Array`, requiring exactly one representation.
  - `build_post_march_components(levels, birth_death) -> pl.DataFrame`, 2,484 April–December rows.
  - `build_recoverability(levels, birth_death) -> pl.DataFrame`, one row per 2003–2025 benchmark year, with the formula and values pinned in Planning evidence plus the complete failed-attempt residual vector and source lineage.

- [x] **Step 1: Write the failing numerical and exclusivity tests**

Create `tests/test_post_march_operators.py` with the imports and first four tests:

```python
"""Req 10 post-March representations and their public-data determination."""

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
from jax.experimental.sparse import BCOO

import annual_data
import vintage_data
from ces_revisions.operators.post_march import (
    ROUNDING_VARIANCE_THOUSANDS2,
    apply_post_march,
    build_post_march_components,
    build_recoverability,
    post_march_cumulative_operator,
    post_march_link_operator,
)


@cache
def components() -> pl.DataFrame:
    return build_post_march_components(
        vintage_data.levels(), annual_data.artifact("birth_death")
    )


@cache
def recoverability() -> pl.DataFrame:
    return build_recoverability(
        vintage_data.levels(), annual_data.artifact("birth_death")
    )


def test_link_relative_operator_matches_the_recursive_definition():
    links = jnp.asarray([1.01, 0.99, 1.02], dtype=jnp.float64)
    revised_bd = jnp.asarray([2.0, -1.0, 3.0], dtype=jnp.float64)
    operator = post_march_link_operator(links)
    assert isinstance(operator, BCOO)
    assert operator.shape == (3, 4)
    expected = []
    level = 100.0
    for link, adjustment in zip(links, revised_bd, strict=True):
        level = level * link + adjustment
        expected.append(level)
    actual = jax.jit(lambda x: operator @ x)(
        jnp.concatenate([jnp.asarray([100.0]), revised_bd])
    )
    np.testing.assert_allclose(actual, expected)


def test_cumulative_operator_adds_sample_and_revised_birth_death_changes():
    sample = jnp.asarray([4.0, -2.0, 1.0], dtype=jnp.float64)
    revised_bd = jnp.asarray([2.0, 3.0, -1.0], dtype=jnp.float64)
    operator = post_march_cumulative_operator(3)
    assert isinstance(operator, BCOO)
    assert operator.shape == (3, 7)
    actual = operator @ jnp.concatenate([jnp.asarray([100.0]), sample, revised_bd])
    np.testing.assert_array_equal(actual, [106.0, 107.0, 107.0])


def test_apply_requires_exactly_one_representation():
    revised_bd = jnp.ones(2, dtype=jnp.float64)
    with pytest.raises(ValueError, match="exactly one"):
        apply_post_march(100.0, revised_bd)
    with pytest.raises(ValueError, match="exactly one"):
        apply_post_march(
            100.0,
            revised_bd,
            link_relatives=jnp.ones(2),
            sample_job_changes=jnp.ones(2),
        )


def test_apply_uses_each_representation_without_mixing_inputs():
    revised_bd = jnp.asarray([2.0, 3.0], dtype=jnp.float64)
    by_link = apply_post_march(
        100.0, revised_bd, link_relatives=jnp.asarray([1.0, 1.0])
    )
    by_change = apply_post_march(
        100.0, revised_bd, sample_job_changes=jnp.asarray([4.0, -2.0])
    )
    np.testing.assert_array_equal(by_link, [102.0, 105.0])
    np.testing.assert_array_equal(by_change, [106.0, 107.0])
```

- [x] **Step 2: Add the failing empirical component and determination tests**

Append to `tests/test_post_march_operators.py`:

```python
def test_inferred_cumulative_components_reproduce_every_published_target():
    frame = components()
    assert frame.height == 2_484
    formal = frame.filter(~pl.col("additional_sample_receipts"))
    assert formal.height == 1_932
    np.testing.assert_allclose(
        frame["reproduced_level_thousands"],
        frame["published_level_thousands"],
        rtol=0.0,
        atol=1e-9,
    )
    assert frame["representation"].unique().to_list() == ["cumulative_job_change"]
    assert frame["series_status"].unique().to_list() == ["inferred"]
    assert frame.filter(pl.col("additional_sample_receipts")).height == 552


def test_components_consume_revised_not_initial_birth_death_values():
    schedules = annual_data.artifact("birth_death").filter(
        pl.col("series_kind") == "published_schedule"
    )
    revised = schedules.filter(pl.col("schedule_kind") == "post_benchmark").select(
        "benchmark_year",
        "sector",
        "reference_month",
        expected=pl.col("value_thousands"),
    )
    checked = components().join(
        revised, on=["benchmark_year", "sector", "reference_month"]
    )
    assert (checked["revised_birth_death_thousands"] == checked["expected"]).all()
    initial = schedules.filter(pl.col("schedule_kind") == "preliminary").select(
        "sector",
        "reference_month",
        initial=pl.col("value_thousands"),
    )
    assert (
        checked.join(initial, on=["sector", "reference_month"])
        .select((pl.col("revised_birth_death_thousands") != pl.col("initial")).any())
        .item()
    )


def test_recoverability_selects_one_inferred_representation_for_every_year():
    frame = recoverability()
    assert frame.height == 23
    assert frame["benchmark_year"].to_list() == list(range(2003, 2026))
    assert frame["representation"].unique().to_list() == ["cumulative_job_change"]
    assert frame["series_status"].unique().to_list() == ["inferred"]
    assert frame.filter(pl.col("link_attempt_status") == "inputs_missing")[
        "benchmark_year"
    ].to_list() == [2003]
    assert frame.filter(pl.col("link_attempt_status") == "failed").height == 22
    assert frame["reconstruction_error_variance_thousands2"].gt(0.0).all()


@pytest.mark.parametrize(
    (
        "year",
        "attempted",
        "within_rounding",
        "maximum",
        "variance",
    ),
    [
        (2003, 0, 0, None, ROUNDING_VARIANCE_THOUSANDS2),
        (2004, 84, 19, 33.407275, 65.199745),
        (2020, 84, 3, 162.058501, 1607.645923),
        (2021, 84, 12, 179.079008, 1293.123145),
        (2025, 84, 11, 17.229047, 30.739396),
    ],
)
def test_recoverability_diagnostics_are_pinned(
    year, attempted, within_rounding, maximum, variance
):
    row = recoverability().filter(pl.col("benchmark_year") == year).row(0, named=True)
    assert row["attempted_cells"] == attempted
    assert row["within_rounding_cells"] == within_rounding
    if maximum is None:
        assert row["max_abs_residual_thousands"] is None
    else:
        assert np.isclose(row["max_abs_residual_thousands"], maximum, atol=1e-6)
    assert np.isclose(
        row["reconstruction_error_variance_thousands2"], variance, atol=1e-6
    )


def test_failed_link_attempts_retain_every_residual_and_source_key():
    failed = recoverability().filter(pl.col("link_attempt_status") == "failed")
    assert failed["link_attempt_residuals_thousands"].list.len().eq(84).all()
    assert failed["level_cell_ids"].list.len().gt(0).all()
    assert failed["birth_death_source_keys"].list.len().gt(0).all()
    for row in failed.iter_rows(named=True):
        residuals = np.asarray(row["link_attempt_residuals_thousands"])
        assert np.isclose(
            np.mean(residuals**2),
            row["reconstruction_error_variance_thousands2"],
        )


def test_nonzero_variance_is_equivalent_to_inferred_status():
    frame = recoverability()
    assert (
        (frame["reconstruction_error_variance_thousands2"] > 0.0)
        == (frame["series_status"] == "inferred")
    ).all()
```

- [x] **Step 3: Run the tests to verify the missing module**

Run: `uv run pytest tests/test_post_march_operators.py -q`

Expected: collection fails because `ces_revisions.operators.post_march` does not exist.

- [x] **Step 4: Implement the two sparse representations and the exclusive application API**

Create `src/ces_revisions/operators/post_march.py` with:

```python
"""Req 10 post-March propagation and link-relative recoverability evidence."""

from datetime import date

import jax
import jax.numpy as jnp
import polars as pl
from jax.experimental.sparse import BCOO

from ces_revisions.operators import OBSERVATION_SECTOR_ORDER
from ces_revisions.operators.benchmark import BENCHMARK_YEARS, benchmark_release_month
from ces_revisions.vintages.months import month_range

ROUNDING_VARIANCE_THOUSANDS2 = 1.0 / 6.0
POST_MARCH_MONTHS = 9
FORMAL_POST_MARCH_MONTHS = 7


def post_march_link_operator(link_relatives: jax.Array) -> BCOO:
    """Map `[March anchor, revised BD]` through fixed matched-sample links."""
    links = jnp.asarray(link_relatives, dtype=jnp.float64)
    if links.ndim != 1 or links.size == 0:
        raise ValueError("link_relatives must be a nonempty vector")
    coefficient = jnp.zeros(links.size + 1, dtype=jnp.float64).at[0].set(1.0)
    rows = []
    for index in range(links.size):
        coefficient = coefficient * links[index]
        coefficient = coefficient.at[index + 1].set(1.0)
        rows.append(coefficient)
    return BCOO.fromdense(jnp.stack(rows))


def post_march_cumulative_operator(month_count: int) -> BCOO:
    """Map `[March anchor, sample changes, revised BD]` to monthly levels."""
    if month_count < 1:
        raise ValueError("month_count must be positive")
    lower = jnp.tril(jnp.ones((month_count, month_count), dtype=jnp.float64))
    dense = jnp.concatenate(
        [jnp.ones((month_count, 1), dtype=jnp.float64), lower, lower], axis=1
    )
    return BCOO.fromdense(dense)


def apply_post_march(
    march_anchor: float | jax.Array,
    revised_birth_death: jax.Array,
    *,
    link_relatives: jax.Array | None = None,
    sample_job_changes: jax.Array | None = None,
) -> jax.Array:
    """Apply exactly one Req 10 representation."""
    if (link_relatives is None) == (sample_job_changes is None):
        raise ValueError("supply exactly one post-March representation")
    revised = jnp.asarray(revised_birth_death, dtype=jnp.float64)
    anchor = jnp.asarray(march_anchor, dtype=jnp.float64).reshape(1)
    if revised.ndim != 1 or revised.size == 0:
        raise ValueError("revised_birth_death must be a nonempty vector")
    if link_relatives is not None:
        links = jnp.asarray(link_relatives, dtype=jnp.float64)
        if links.shape != revised.shape:
            raise ValueError("link_relatives and revised_birth_death must align")
        return post_march_link_operator(links) @ jnp.concatenate([anchor, revised])
    changes = jnp.asarray(sample_job_changes, dtype=jnp.float64)
    if changes.shape != revised.shape:
        raise ValueError("sample_job_changes and revised_birth_death must align")
    inputs = jnp.concatenate([anchor, changes, revised])
    return post_march_cumulative_operator(revised.size) @ inputs
```

- [x] **Step 5: Implement inferred cumulative components with source lineage**

Add this schema, indexing helper, and builder to `post_march.py`:

```python
POST_MARCH_SCHEMA = {
    "benchmark_year": pl.Int32,
    "sector": pl.String,
    "reference_month": pl.Date,
    "month_index": pl.Int8,
    "representation": pl.String,
    "series_status": pl.String,
    "additional_sample_receipts": pl.Boolean,
    "march_anchor_thousands": pl.Float64,
    "published_level_thousands": pl.Float64,
    "revised_birth_death_thousands": pl.Float64,
    "inferred_sample_job_change_thousands": pl.Float64,
    "reproduced_level_thousands": pl.Float64,
    "residual_thousands": pl.Float64,
    "march_anchor_cell_id": pl.Int64,
    "previous_level_cell_id": pl.Int64,
    "published_level_cell_id": pl.Int64,
    "birth_death_source_keys": pl.List(pl.String),
}


def _level_index(levels: pl.DataFrame) -> dict[tuple[str, date, date], dict]:
    nsa = levels.filter(
        (pl.col("seasonal_status") == "NSA") & pl.col("value_thousands").is_not_null()
    )
    return {
        (row["sector"], row["release_month"], row["reference_month"]): row
        for row in nsa.iter_rows(named=True)
    }


def _schedule_index(
    birth_death: pl.DataFrame, schedule_kind: str
) -> dict[tuple[int, str, date], dict]:
    frame = birth_death.filter(
        (pl.col("series_kind") == "published_schedule")
        & (pl.col("schedule_kind") == schedule_kind)
    )
    return {
        (row["benchmark_year"], row["sector"], row["reference_month"]): row
        for row in frame.iter_rows(named=True)
    }


def build_post_march_components(
    levels: pl.DataFrame, birth_death: pl.DataFrame
) -> pl.DataFrame:
    level_rows = _level_index(levels)
    revised = _schedule_index(birth_death, "post_benchmark")
    records = []
    for year in BENCHMARK_YEARS:
        release = benchmark_release_month(year)
        months = month_range(date(year, 4, 1), date(year, 12, 1))
        for sector in OBSERVATION_SECTOR_ORDER:
            anchor_row = level_rows[(sector, release, date(year, 3, 1))]
            previous = float(anchor_row["value_thousands"])
            previous_row = anchor_row
            sample_changes = []
            revised_values = []
            source_rows = []
            for month in months:
                level_row = level_rows[(sector, release, month)]
                schedule = revised[(year, sector, month)]
                revised_value = float(schedule["value_thousands"])
                sample_change = (
                    float(level_row["value_thousands"]) - previous - revised_value
                )
                sample_changes.append(sample_change)
                revised_values.append(revised_value)
                source_rows.append((month, previous_row, level_row, schedule))
                previous = float(level_row["value_thousands"])
                previous_row = level_row
            reproduced = apply_post_march(
                float(anchor_row["value_thousands"]),
                jnp.asarray(revised_values, dtype=jnp.float64),
                sample_job_changes=jnp.asarray(sample_changes, dtype=jnp.float64),
            )
            for index, (
                (month, previous_row, level_row, schedule),
                calculated,
            ) in enumerate(zip(source_rows, reproduced, strict=True), start=1):
                published = float(level_row["value_thousands"])
                records.append(
                    {
                        "benchmark_year": year,
                        "sector": sector,
                        "reference_month": month,
                        "month_index": index,
                        "representation": "cumulative_job_change",
                        "series_status": "inferred",
                        "additional_sample_receipts": month.month >= 11,
                        "march_anchor_thousands": float(anchor_row["value_thousands"]),
                        "published_level_thousands": published,
                        "revised_birth_death_thousands": revised_values[index - 1],
                        "inferred_sample_job_change_thousands": sample_changes[
                            index - 1
                        ],
                        "reproduced_level_thousands": float(calculated),
                        "residual_thousands": published - float(calculated),
                        "march_anchor_cell_id": anchor_row["cell_id"],
                        "previous_level_cell_id": previous_row["cell_id"],
                        "published_level_cell_id": level_row["cell_id"],
                        "birth_death_source_keys": schedule["source_keys"],
                    }
                )
    return pl.DataFrame(records, schema=POST_MARCH_SCHEMA).sort(
        "benchmark_year", "sector", "reference_month"
    )
```

- [x] **Step 6: Implement the per-year link attempt and variance rule**

Add to `post_march.py`:

```python
RECOVERABILITY_SCHEMA = {
    "benchmark_year": pl.Int32,
    "representation": pl.String,
    "series_status": pl.String,
    "link_attempt_status": pl.String,
    "attempted_cells": pl.Int32,
    "within_rounding_cells": pl.Int32,
    "max_abs_residual_thousands": pl.Float64,
    "reconstruction_error_variance_thousands2": pl.Float64,
    "link_attempt_residuals_thousands": pl.List(pl.Float64),
    "level_cell_ids": pl.List(pl.Int64),
    "birth_death_source_keys": pl.List(pl.String),
    "reason": pl.String,
}


def build_recoverability(
    levels: pl.DataFrame, birth_death: pl.DataFrame
) -> pl.DataFrame:
    level_rows = _level_index(levels)
    initial_rows = birth_death.filter(
        (pl.col("series_kind") == "published_schedule")
        & (pl.col("schedule_kind") == "preliminary")
    )
    initial = {
        (row["sector"], row["reference_month"]): row
        for row in initial_rows.iter_rows(named=True)
    }
    revised = _schedule_index(birth_death, "post_benchmark")
    records = []
    for year in BENCHMARK_YEARS:
        if year == 2003:
            records.append(
                {
                    "benchmark_year": year,
                    "representation": "cumulative_job_change",
                    "series_status": "inferred",
                    "link_attempt_status": "inputs_missing",
                    "attempted_cells": 0,
                    "within_rounding_cells": 0,
                    "max_abs_residual_thousands": None,
                    "reconstruction_error_variance_thousands2": (
                        ROUNDING_VARIANCE_THOUSANDS2
                    ),
                    "link_attempt_residuals_thousands": [],
                    "level_cell_ids": [],
                    "birth_death_source_keys": [],
                    "reason": (
                        "BLS publishes the 2003 post-benchmark schedule but no "
                        "initial April–December 2003 schedule or matched-sample links."
                    ),
                }
            )
            continue
        release_old = date(year, 12, 1)
        release_new = benchmark_release_month(year)
        months = month_range(date(year, 4, 1), date(year, 10, 1))
        residuals = []
        level_cell_ids = set()
        birth_death_source_keys = set()
        for sector in OBSERVATION_SECTOR_ORDER:
            old_previous_row = level_rows[(sector, release_old, date(year, 3, 1))]
            new_anchor_row = level_rows[(sector, release_new, date(year, 3, 1))]
            level_cell_ids.update(
                [old_previous_row["cell_id"], new_anchor_row["cell_id"]]
            )
            old_previous = float(old_previous_row["value_thousands"])
            links = []
            revised_values = []
            targets = []
            for month in months:
                old_level_row = level_rows[(sector, release_old, month)]
                target_row = level_rows[(sector, release_new, month)]
                initial_row = initial[(sector, month)]
                revised_row = revised[(year, sector, month)]
                level_cell_ids.update([old_level_row["cell_id"], target_row["cell_id"]])
                birth_death_source_keys.update(initial_row["source_keys"])
                birth_death_source_keys.update(revised_row["source_keys"])
                old_level = float(old_level_row["value_thousands"])
                initial_bd = float(initial_row["value_thousands"])
                revised_bd = float(revised_row["value_thousands"])
                links.append((old_level - initial_bd) / old_previous)
                revised_values.append(revised_bd)
                targets.append(float(target_row["value_thousands"]))
                old_previous = old_level
            predicted = apply_post_march(
                float(new_anchor_row["value_thousands"]),
                jnp.asarray(revised_values, dtype=jnp.float64),
                link_relatives=jnp.asarray(links, dtype=jnp.float64),
            )
            residuals.extend(
                target - float(calculated)
                for target, calculated in zip(targets, predicted, strict=True)
            )
        within_rounding = sum(abs(value) <= 0.5 for value in residuals)
        if within_rounding == len(residuals):
            raise ValueError(
                f"{year} aggregate link proxy unexpectedly passed; revisit the "
                "source-recoverability decision before changing representation"
            )
        squared_mean = sum(value * value for value in residuals) / len(residuals)
        records.append(
            {
                "benchmark_year": year,
                "representation": "cumulative_job_change",
                "series_status": "inferred",
                "link_attempt_status": "failed",
                "attempted_cells": len(residuals),
                "within_rounding_cells": within_rounding,
                "max_abs_residual_thousands": max(abs(value) for value in residuals),
                "reconstruction_error_variance_thousands2": max(
                    squared_mean, ROUNDING_VARIANCE_THOUSANDS2
                ),
                "link_attempt_residuals_thousands": residuals,
                "level_cell_ids": sorted(level_cell_ids),
                "birth_death_source_keys": sorted(birth_death_source_keys),
                "reason": (
                    "Vintage levels and initial birth–death imply only an aggregate "
                    "link proxy; applying it with revised birth–death does not "
                    "reproduce every April–October level to rounding."
                ),
            }
        )
    return pl.DataFrame(records, schema=RECOVERABILITY_SCHEMA).sort("benchmark_year")
```

- [x] **Step 7: Run the post-March gate**

Run:

```bash
uv run pytest tests/test_post_march_operators.py -q
uv run ruff format src/ces_revisions/operators/post_march.py tests/test_post_march_operators.py
uv run ruff check src/ces_revisions/operators tests/test_post_march_operators.py
```

Expected: all tests pass; 23 recoverability rows select one representation; the 1,932 April–October cells and 552 November/December cells reproduce exactly; every variance is positive.

- [x] **Step 8: Commit the post-March determination code**

```bash
git add src/ces_revisions/operators/post_march.py tests/test_post_march_operators.py
git commit -m "feat(operators): determine post-March representation"
```

### Task 5: Add the Joint NSA/SA Map and Release-Specific B→M Components

**Files:**
- Create: `src/ces_revisions/operators/seasonal.py`
- Test: `tests/test_seasonal_operators.py`

**Interfaces:**
- Consumes: Stage 3 `levels` and `stage_labels`; Task 4's recoverability table.
- Produces:
  - `seasonal_measurement_operator() -> BCOO`, the `(NSA, SA) <- (x, q)` block `[[1, 0], [1, -1]]`.
  - `implied_adjustment_operator() -> BCOO`, the `NSA-SA` row `[1, -1]`.
  - `build_b_to_m_components(levels, stage_labels, recoverability) -> pl.DataFrame`, keyed by `(sector, reference_month, seasonal_status)` and carrying the B/M status, same-release NSA/SA pairs, implied-adjustment change, NSA next-wedge weight where applicable, and the recoverability variance.

The component table is release-specific. NSA `M` is the second benchmark after `T`; SA `M` is the first benchmark vintage after the reference month leaves the five-year SA window. The paired NSA value used beside an SA `M` therefore comes from the SA-M release, not from the earlier NSA-M stage.

- [x] **Step 1: Write the failing seasonal matrix tests**

Create `tests/test_seasonal_operators.py`:

```python
"""Req 9 joint NSA/SA maps and release-specific B→M components."""

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from jax.experimental.sparse import BCOO

import annual_data
import vintage_data
from ces_revisions.operators.post_march import build_recoverability
from ces_revisions.operators.seasonal import (
    build_b_to_m_components,
    implied_adjustment_operator,
    seasonal_measurement_operator,
)


@cache
def components() -> pl.DataFrame:
    recoverability = build_recoverability(
        vintage_data.levels(), annual_data.artifact("birth_death")
    )
    return build_b_to_m_components(
        vintage_data.levels(), vintage_data.labels(), recoverability
    )


def test_seasonal_measurement_map_uses_one_state_not_a_second_free_path():
    operator = seasonal_measurement_operator()
    assert isinstance(operator, BCOO)
    assert operator.shape == (2, 2)
    result = jax.jit(lambda z: operator @ z)(
        jnp.asarray([150.0, 12.0], dtype=jnp.float64)
    )
    np.testing.assert_array_equal(result, [150.0, 138.0])


def test_implied_adjustment_is_published_nsa_minus_sa():
    operator = implied_adjustment_operator()
    assert isinstance(operator, BCOO)
    assert operator.shape == (1, 2)
    result = operator @ jnp.asarray([150.0, 138.0], dtype=jnp.float64)
    np.testing.assert_array_equal(result, [12.0])


def test_b_to_m_components_have_one_unique_row_per_axis_cell():
    frame = components()
    key = ["sector", "reference_month", "seasonal_status"]
    assert frame.select(key).n_unique() == frame.height
    assert set(frame["seasonal_status"]) == {"NSA", "SA"}
    assert set(frame["b_status"]) <= {
        "observed",
        "missing_vintage",
        "beyond_frontier",
        "right_censored",
    }
    assert set(frame["m_status"]) <= {
        "observed",
        "missing_vintage",
        "beyond_frontier",
        "right_censored",
    }


def test_sa_b_to_m_identity_uses_pairs_from_each_rows_own_release():
    observed = components().filter(
        (pl.col("seasonal_status") == "SA")
        & (pl.col("b_status") == "observed")
        & (pl.col("m_status") == "observed")
    )
    np.testing.assert_allclose(
        observed["observed_delta_thousands"],
        observed["paired_nsa_delta_thousands"]
        - observed["seasonal_adjustment_delta_thousands"],
        rtol=0.0,
        atol=1e-9,
    )
    np.testing.assert_allclose(observed["identity_residual_thousands"], 0.0)


def test_nsa_next_wedge_component_exists_only_for_april_through_october():
    frame = components()
    weighted = frame.filter(pl.col("next_wedge_weight").is_not_null())
    assert weighted["seasonal_status"].unique().to_list() == ["NSA"]
    assert weighted["reference_month"].dt.month().is_between(4, 10).all()
    expected = (weighted["reference_month"].dt.month() - 3) / 12
    np.testing.assert_allclose(weighted["next_wedge_weight"], expected)
    assert weighted["reconstruction_error_variance_thousands2"].gt(0.0).all()


def test_right_censored_m_rows_have_no_fabricated_value():
    rows = components().filter(pl.col("m_status") == "right_censored")
    assert rows.height > 0
    assert rows["m_value_thousands"].is_null().all()
    assert rows["observed_delta_thousands"].is_null().all()


def test_sa_and_nsa_m_horizons_remain_distinct():
    sample = components().filter(
        (pl.col("sector") == "00") & (pl.col("reference_month") == pl.date(2003, 5, 1))
    )
    release_by_status = dict(
        sample.select("seasonal_status", "m_release_month").iter_rows()
    )
    assert release_by_status["NSA"] != release_by_status["SA"]
```

- [x] **Step 2: Run the tests to verify the missing module**

Run: `uv run pytest tests/test_seasonal_operators.py -q`

Expected: collection fails because `ces_revisions.operators.seasonal` does not exist.

- [x] **Step 3: Implement the two sparse seasonal maps**

Create `src/ces_revisions/operators/seasonal.py`:

```python
"""Req 9's joint NSA/SA measurement map and B→M observable components."""

from datetime import date

import jax.numpy as jnp
import polars as pl
from jax.experimental.sparse import BCOO


def seasonal_measurement_operator() -> BCOO:
    """Map mature `(x, q)` to `(NSA, SA) = (x, x-q)`."""
    return BCOO.fromdense(jnp.asarray([[1.0, 0.0], [1.0, -1.0]], dtype=jnp.float64))


def implied_adjustment_operator() -> BCOO:
    """Map a published `(NSA, SA)` pair to its additive adjustment."""
    return BCOO.fromdense(jnp.asarray([[1.0, -1.0]], dtype=jnp.float64))
```

- [x] **Step 4: Implement the release-specific B→M table**

Append the complete schema and builder below to `seasonal.py`:

```python
B_TO_M_SCHEMA = {
    "sector": pl.String,
    "reference_month": pl.Date,
    "seasonal_status": pl.String,
    "b_release_month": pl.Date,
    "m_release_month": pl.Date,
    "b_status": pl.String,
    "m_status": pl.String,
    "b_value_thousands": pl.Float64,
    "m_value_thousands": pl.Float64,
    "b_nsa_thousands": pl.Float64,
    "b_sa_thousands": pl.Float64,
    "m_nsa_thousands": pl.Float64,
    "m_sa_thousands": pl.Float64,
    "b_adjustment_thousands": pl.Float64,
    "m_adjustment_thousands": pl.Float64,
    "observed_delta_thousands": pl.Float64,
    "paired_nsa_delta_thousands": pl.Float64,
    "seasonal_adjustment_delta_thousands": pl.Float64,
    "identity_residual_thousands": pl.Float64,
    "next_benchmark_year": pl.Int32,
    "next_wedge_weight": pl.Float64,
    "reconstruction_error_variance_thousands2": pl.Float64,
    "b_cell_ids": pl.List(pl.Int64),
    "m_cell_ids": pl.List(pl.Int64),
}


def _value_index(
    levels: pl.DataFrame,
) -> dict[tuple[str, date, date, str], dict]:
    return {
        (
            row["sector"],
            row["reference_month"],
            row["release_month"],
            row["seasonal_status"],
        ): row
        for row in levels.iter_rows(named=True)
        if row["value_thousands"] is not None
    }


def _pair_at_release(
    values: dict[tuple[str, date, date, str], dict],
    sector: str,
    reference_month: date,
    release_month: date,
) -> tuple[float | None, float | None, list[int]]:
    nsa = values.get((sector, reference_month, release_month, "NSA"))
    sa = values.get((sector, reference_month, release_month, "SA"))
    return (
        None if nsa is None else float(nsa["value_thousands"]),
        None if sa is None else float(sa["value_thousands"]),
        [row["cell_id"] for row in (nsa, sa) if row is not None],
    )


def _difference(later: float | None, earlier: float | None) -> float | None:
    return None if later is None or earlier is None else later - earlier


def build_b_to_m_components(
    levels: pl.DataFrame,
    stage_labels: pl.DataFrame,
    recoverability: pl.DataFrame,
) -> pl.DataFrame:
    labels = stage_labels.filter(
        (pl.col("source") == "cesvinall") & pl.col("release_stage").is_in(["B", "M"])
    )
    values = _value_index(levels)
    variances = {
        row["benchmark_year"]: row["reconstruction_error_variance_thousands2"]
        for row in recoverability.iter_rows(named=True)
    }
    records = []
    for key, group in labels.group_by(
        "sector", "reference_month", "seasonal_status", maintain_order=True
    ):
        sector, reference_month, seasonal_status = key
        by_stage = {row["release_stage"]: row for row in group.iter_rows(named=True)}
        if set(by_stage) != {"B", "M"}:
            raise ValueError(f"missing B or M label for {key}")
        b_label = by_stage["B"]
        m_label = by_stage["M"]
        b_release = b_label["release_month"]
        m_release = m_label["release_month"]
        b_nsa, b_sa, b_ids = _pair_at_release(
            values, sector, reference_month, b_release
        )
        m_nsa, m_sa, m_ids = _pair_at_release(
            values, sector, reference_month, m_release
        )
        b_value = b_nsa if seasonal_status == "NSA" else b_sa
        m_value = m_nsa if seasonal_status == "NSA" else m_sa
        b_adjustment = _difference(b_nsa, b_sa)
        m_adjustment = _difference(m_nsa, m_sa)
        observed_delta = _difference(m_value, b_value)
        paired_nsa_delta = _difference(m_nsa, b_nsa)
        seasonal_delta = _difference(m_adjustment, b_adjustment)
        stages_observed = b_label["status"] == m_label["status"] == "observed"
        if not stages_observed or None in (
            observed_delta,
            paired_nsa_delta,
            seasonal_delta,
        ):
            identity = None
        elif seasonal_status == "SA":
            identity = observed_delta - (paired_nsa_delta - seasonal_delta)
        else:
            identity = observed_delta - paired_nsa_delta
        candidate_year = m_release.year - 1
        eligible_wedge = (
            seasonal_status == "NSA"
            and 4 <= reference_month.month <= 10
            and candidate_year in variances
        )
        next_year = candidate_year if eligible_wedge else None
        variance = variances.get(next_year) if next_year is not None else None
        records.append(
            {
                "sector": sector,
                "reference_month": reference_month,
                "seasonal_status": seasonal_status,
                "b_release_month": b_release,
                "m_release_month": m_release,
                "b_status": b_label["status"],
                "m_status": m_label["status"],
                "b_value_thousands": b_value
                if b_label["status"] == "observed"
                else None,
                "m_value_thousands": m_value
                if m_label["status"] == "observed"
                else None,
                "b_nsa_thousands": b_nsa,
                "b_sa_thousands": b_sa,
                "m_nsa_thousands": m_nsa,
                "m_sa_thousands": m_sa,
                "b_adjustment_thousands": b_adjustment,
                "m_adjustment_thousands": m_adjustment,
                "observed_delta_thousands": observed_delta if stages_observed else None,
                "paired_nsa_delta_thousands": paired_nsa_delta,
                "seasonal_adjustment_delta_thousands": seasonal_delta,
                "identity_residual_thousands": identity,
                "next_benchmark_year": next_year,
                "next_wedge_weight": (
                    (reference_month.month - 3) / 12 if eligible_wedge else None
                ),
                "reconstruction_error_variance_thousands2": variance,
                "b_cell_ids": b_ids,
                "m_cell_ids": m_ids,
            }
        )
    return pl.DataFrame(records, schema=B_TO_M_SCHEMA).sort(
        "sector", "reference_month", "seasonal_status"
    )
```

The explicit branch above is intentional: it keeps the NSA and SA identities auditable and avoids conditional-expression precedence hiding which residual is being tested.

- [x] **Step 5: Run the seasonal and B→M tests**

Run:

```bash
uv run pytest tests/test_seasonal_operators.py -q
uv run ruff format src/ces_revisions/operators/seasonal.py tests/test_seasonal_operators.py
uv run ruff check src/ces_revisions/operators tests/test_seasonal_operators.py
```

Expected: 7 tests pass; every observed SA B→M row satisfies the accounting identity exactly; right-censored M rows remain null; only NSA April–October rows carry a next-wedge coefficient and variance.

- [x] **Step 6: Commit the seasonal and B→M components**

```bash
git add src/ces_revisions/operators/seasonal.py tests/test_seasonal_operators.py
git commit -m "feat(operators): add seasonal and B-to-M maps"
```

### Task 6: Assemble Artifacts, Record the Decision, and Pass the Stage Gate

**Files:**
- Create: `src/ces_revisions/operators/build.py`
- Create: `scripts/operator_artifacts.py`
- Create: `tests/operator_data.py`
- Create: `tests/test_operator_build.py`
- Create: `docs/decisions/link-relatives.md`
- Modify: `.gitignore`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: all Task 2–5 builders, Stage 3/4 builds, and both committed source manifests.
- Produces:
  - `OperatorBuild(benchmark_fixtures, post_march_components, recoverability, b_to_m_components)`.
  - `assemble(levels, stage_labels, benchmarks, reconstruction_events, birth_death) -> OperatorBuild` for cached tests and downstream callers.
  - `build() -> OperatorBuild` from committed sources.
  - `operator_test_record(result, levels) -> dict` and `write(result, levels, out_dir=OUTPUT_DIR) -> dict`.
  - `data/operators/{benchmark_fixtures,post_march_components,recoverability,b_to_m_components}.parquet` plus deterministic `operator-test-results.json`.
  - `scripts/operator_artifacts.py build [--out-dir PATH]`.

- [x] **Step 1: Write the failing build, record, CLI, and decision tests**

Create `tests/operator_data.py`:

```python
"""One cached Stage 5 build shared by operator artifact tests."""

from functools import cache

import polars as pl

import annual_data
import vintage_data
from ces_revisions.operators.build import OperatorBuild, assemble


@cache
def result() -> OperatorBuild:
    return assemble(
        vintage_data.levels(),
        vintage_data.labels(),
        annual_data.artifact("benchmarks"),
        annual_data.artifact("reconstruction_events"),
        annual_data.artifact("birth_death"),
    )


def artifact(name: str) -> pl.DataFrame:
    return getattr(result(), name)
```

Create `tests/test_operator_build.py`:

```python
"""Stage 5 artifact round trip and deterministic provenance record."""

import json
import re
from dataclasses import fields
from pathlib import Path

import polars as pl

import operator_artifacts
import operator_data
import vintage_data
from ces_revisions.operators.build import operator_test_record, write


def test_assembled_artifacts_have_the_stage_gate_rows():
    result = operator_data.result()
    assert result.benchmark_fixtures.height == 3_312
    assert result.post_march_components.height == 2_484
    assert result.recoverability.height == 23
    assert result.b_to_m_components.height > 0
    assert result.recoverability["benchmark_year"].to_list() == list(range(2003, 2026))


def test_operator_test_record_is_deterministic_and_complete():
    first = operator_test_record(operator_data.result(), vintage_data.levels())
    second = operator_test_record(operator_data.result(), vintage_data.levels())
    assert first == second
    assert "generated_at" not in first
    assert first["schema_version"] == 1
    assert set(first["input_manifests"]) == {"stage3", "stage4"}
    assert all(len(value) == 64 for value in first["input_manifests"].values())
    assert first["checks"] == {
        "aggregation": {"cells": 496_310, "max_abs_residual_thousands": 0.0},
        "wedge": {
            "years": 23,
            "cells": 3_312,
            "reconstruction_supported_cells": 215,
            "max_abs_reproduction_residual_thousands": 0.0,
            "max_abs_non_event_rounding_residual_thousands": 21.0,
        },
        "post_march": {
            "years": 23,
            "april_october_cells": 1_932,
            "november_december_cells": 552,
            "max_abs_reproduction_residual_thousands": 0.0,
        },
        "recoverability": {
            "cumulative_job_change_years": 23,
            "inferred_years": 23,
            "recovered_years": 0,
        },
        "seasonal_mapping": {"max_abs_identity_residual_thousands": 0.0},
    }


def test_write_round_trips_every_frame_and_json_record(tmp_path: Path):
    result = operator_data.result()
    record = write(result, vintage_data.levels(), tmp_path)
    assert json.loads((tmp_path / "operator-test-results.json").read_text()) == record
    for field in fields(result):
        path = tmp_path / f"{field.name}.parquet"
        assert path.is_file()
        assert pl.read_parquet(path).equals(getattr(result, field.name))
        assert (
            record["artifacts"][field.name]["rows"]
            == getattr(result, field.name).height
        )
        assert len(record["artifacts"][field.name]["sha256"]) == 64


def test_offline_cli_writes_the_same_artifact_set(tmp_path: Path):
    assert operator_artifacts.main(["build", "--out-dir", str(tmp_path)]) == 0
    assert (tmp_path / "operator-test-results.json").is_file()
    assert set(path.name for path in tmp_path.glob("*.parquet")) == {
        f"{field.name}.parquet" for field in fields(operator_data.result())
    }


def test_written_determination_has_one_row_per_year_and_the_selected_contract():
    text = Path("docs/decisions/link-relatives.md").read_text(encoding="utf-8")
    assert "**Status:** Accepted" in text
    assert "`cumulative_job_change` for all 23 benchmark years" in text
    rows = re.findall(r"^\| (20\d{2}) \|", text, flags=re.MULTILINE)
    assert rows == [str(year) for year in range(2003, 2026)]
    for row in operator_data.artifact("recoverability").iter_rows(named=True):
        assert f"| {row['benchmark_year']} | cumulative job change | inferred |" in text
```

- [x] **Step 2: Run the tests to verify the missing build module**

Run: `uv run pytest tests/test_operator_build.py -q`

Expected: collection fails because `ces_revisions.operators.build` does not exist.

- [x] **Step 3: Implement the Stage 5 assembly and summaries**

Create `src/ces_revisions/operators/build.py` with the assembly types and helpers:

```python
"""Assemble and write deterministic roadmap Stage 5 operator artifacts."""

import json
from dataclasses import dataclass, fields
from pathlib import Path

import polars as pl

from ces_revisions.annual import raw as annual_raw
from ces_revisions.annual.build import build as build_annual
from ces_revisions.operators import SECTOR_ORDER
from ces_revisions.operators.benchmark import build_benchmark_fixtures
from ces_revisions.operators.post_march import (
    build_post_march_components,
    build_recoverability,
)
from ces_revisions.operators.seasonal import build_b_to_m_components
from ces_revisions.vintages import raw as vintage_raw
from ces_revisions.vintages.build import build as build_vintages

OUTPUT_DIR = vintage_raw.ROOT / "data" / "operators"


@dataclass(frozen=True)
class OperatorBuild:
    benchmark_fixtures: pl.DataFrame
    post_march_components: pl.DataFrame
    recoverability: pl.DataFrame
    b_to_m_components: pl.DataFrame


def assemble(
    levels: pl.DataFrame,
    stage_labels: pl.DataFrame,
    benchmarks: pl.DataFrame,
    reconstruction_events: pl.DataFrame,
    birth_death: pl.DataFrame,
) -> OperatorBuild:
    recoverability = build_recoverability(levels, birth_death)
    return OperatorBuild(
        benchmark_fixtures=build_benchmark_fixtures(
            levels, benchmarks, reconstruction_events
        ),
        post_march_components=build_post_march_components(levels, birth_death),
        recoverability=recoverability,
        b_to_m_components=build_b_to_m_components(levels, stage_labels, recoverability),
    )


def build() -> tuple[OperatorBuild, pl.DataFrame]:
    vintage = build_vintages()
    annual = build_annual()
    levels = vintage.panel.filter(
        (pl.col("source") == "cesvinall") & (pl.col("measure") == "level")
    )
    result = assemble(
        levels,
        vintage.stage_labels,
        annual.benchmarks,
        annual.reconstruction_events,
        annual.birth_death,
    )
    return result, levels


def _aggregation_check(levels: pl.DataFrame) -> dict:
    frame = levels.filter(pl.col("value_thousands").is_not_null())
    keys = ["release_month", "seasonal_status", "reference_month"]
    total = frame.filter(pl.col("sector") == "00").select(
        *keys, total=pl.col("value_thousands")
    )
    parts = (
        frame.filter(pl.col("sector").is_in(SECTOR_ORDER))
        .group_by(keys)
        .agg(parts=pl.col("value_thousands").sum(), sectors=pl.len())
    )
    checked = total.join(parts, on=keys).with_columns(
        residual=pl.col("total") - pl.col("parts")
    )
    if checked["sectors"].ne(len(SECTOR_ORDER)).any():
        raise ValueError("an aggregation cell does not contain eleven sectors")
    return {
        "cells": checked.height,
        "max_abs_residual_thousands": float(checked["residual"].abs().max()),
    }


def _checks(result: OperatorBuild, levels: pl.DataFrame) -> dict:
    wedge = result.benchmark_fixtures
    post = result.post_march_components
    recovery = result.recoverability
    identity = result.b_to_m_components["identity_residual_thousands"].drop_nulls()
    return {
        "aggregation": _aggregation_check(levels),
        "wedge": {
            "years": wedge["benchmark_year"].n_unique(),
            "cells": wedge.height,
            "reconstruction_supported_cells": wedge.filter(
                pl.col("reconstruction_supported")
            ).height,
            "max_abs_reproduction_residual_thousands": float(
                (
                    wedge["reproduced_level_thousands"]
                    - wedge["benchmark_level_thousands"]
                )
                .abs()
                .max()
            ),
            "max_abs_non_event_rounding_residual_thousands": float(
                wedge.filter(~pl.col("reconstruction_supported"))[
                    "rounding_residual_thousands"
                ]
                .abs()
                .max()
            ),
        },
        "post_march": {
            "years": post["benchmark_year"].n_unique(),
            "april_october_cells": post.filter(
                ~pl.col("additional_sample_receipts")
            ).height,
            "november_december_cells": post.filter(
                pl.col("additional_sample_receipts")
            ).height,
            "max_abs_reproduction_residual_thousands": float(
                post["residual_thousands"].abs().max()
            ),
        },
        "recoverability": {
            "cumulative_job_change_years": recovery.filter(
                pl.col("representation") == "cumulative_job_change"
            ).height,
            "inferred_years": recovery.filter(
                pl.col("series_status") == "inferred"
            ).height,
            "recovered_years": recovery.filter(
                pl.col("series_status") == "recovered"
            ).height,
        },
        "seasonal_mapping": {
            "max_abs_identity_residual_thousands": float(identity.abs().max())
        },
    }
```

- [x] **Step 4: Implement deterministic hashing and writing**

Append to `operators/build.py`:

```python
def operator_test_record(result: OperatorBuild, levels: pl.DataFrame) -> dict:
    artifacts = {
        field.name: {
            "rows": getattr(result, field.name).height,
            "sha256": annual_raw.content_sha256(getattr(result, field.name)),
        }
        for field in fields(result)
    }
    return {
        "schema_version": 1,
        "input_manifests": {
            "stage3": vintage_raw.file_sha256(
                vintage_raw.RAW_DIR / vintage_raw.MANIFEST
            ),
            "stage4": annual_raw.file_sha256(annual_raw.RAW_DIR / annual_raw.MANIFEST),
        },
        "artifacts": artifacts,
        "checks": _checks(result, levels),
    }


def write(
    result: OperatorBuild,
    levels: pl.DataFrame,
    out_dir: Path = OUTPUT_DIR,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    for field in fields(result):
        getattr(result, field.name).write_parquet(out_dir / f"{field.name}.parquet")
    record = operator_test_record(result, levels)
    (out_dir / "operator-test-results.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record
```

If `annual_raw.content_sha256()` proves sensitive to a Polars struct/list serialization change, keep that shared function and fix the shared test; do not invent a second hashing convention for Stage 5.

- [x] **Step 5: Add the offline CLI and output ignore rule**

> Deviation: The standalone CLI enables JAX float64 before importing the operator build because it cannot inherit pytest's process-level initialization; a fresh-interpreter test pins the ordering.

Create `scripts/operator_artifacts.py`:

```python
"""Build deterministic Stage 5 operator fixtures and their provenance record."""

import argparse
from pathlib import Path

from ces_revisions.operators import build as operator_build


def build_artifacts(out_dir: Path) -> int:
    result, levels = operator_build.build()
    record = operator_build.write(result, levels, out_dir)
    for name, entry in record["artifacts"].items():
        print(f"{name}: {entry['rows']} rows")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--out-dir", type=Path, default=operator_build.OUTPUT_DIR)
    arguments = parser.parse_args(argv)
    return build_artifacts(arguments.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
```

Append to `.gitignore`:

```gitignore

# Stage 5 deterministic operator artifacts are rebuilt from the committed Stage 3/4 sources
data/operators/
```

- [x] **Step 6: Write the accepted link-relative determination**

Create `docs/decisions/link-relatives.md` with the following complete decision. Keep the six-decimal values below aligned with the generated parquet; the test record retains full float64 values.

````markdown
# Use inferred cumulative job changes for every public post-March operator

- **Status:** Accepted
- **Date:** 2026-09-16
- **Deciders:** Lowell Mason
- **Blast radius:** roadmap Stages 8, 17, 20, 23, and 24; every model or report that uses post-March or B→M observations

## Context

Req 10 of [`specs/ces-revisions.md`](../../specs/ces-revisions.md) preserves BLS's post-March calculation as either (1) a recursion in previously published matched-sample link relatives and revised net birth–death forecasts or (2) a linear cumulative operator when only job-change contributions are available. The two representations may not be mixed. Roadmap Stage 5 must determine, for each benchmark year 2003–2025, whether the required public series is recovered or inferred and must assign positive reconstruction-error variance to inferred years.

The committed vintage triangles contain published NSA and SA levels, not matched-sample employment sums, weighted link relatives, or sample-only job changes. The Stage 4 birth–death table contains the initial January–December schedule from 2004 onward and each revised post-benchmark April–December schedule. It contains no initial April–December 2003 schedule.

For 2004–2025 we tested the strongest aggregate link proxy public data permit:

```math
\widehat R^{old}_{s,m}
=\frac{E^{old}_{s,m}-BD^{old}_{s,m}}{E^{old}_{s,m-1}},
\qquad
\widehat E^B_{s,m}
=\widehat E^B_{s,m-1}\widehat R^{old}_{s,m}+BD^{new}_{s,m}.
```

Each attempt used all twelve sectors and April–October, 84 cells per year. No year reproduced all cells within ±0.5 thousand. The failure is expected: BLS applies weighted link-relative estimation below the published supersector aggregate, and whole-thousand aggregate levels do not identify those private matched-sample inputs. The pandemic and later method-change years show especially large residuals.

## Decision

Use `cumulative_job_change` for all 23 benchmark years. Every series is `inferred`; no historical link-relative series is labeled recovered. For each published benchmark vintage infer the sample contribution as

```math
\widehat{SC}_{s,m}=E^B_{s,m}-E^B_{s,m-1}-BD^{new}_{s,m},
```

and represent the path as the sparse cumulative map

```math
E^B_{s,m}=E^B_{s,\mathrm{Mar}(y)}
+\sum_{h=\mathrm{Apr}(y)}^m
\left(\widehat{SC}_{s,h}+BD^{new}_{s,h}\right).
```

Here $`E^B_{s,\mathrm{Mar}(y)}`$ is the archived published March level after any
documented reconstruction and rounding residual, rather than the scope-comparable wedge anchor.

This identity reproduces the published April–December benchmark vintage and explicitly consumes revised birth–death values. It is an operator reconstruction of the published path, not an observation of BLS's private matched-sample link relatives. Later fits propagate the per-year reconstruction-error variance below rather than fixing the inferred components as error-free.

For 2004–2025 the variance is the mean squared residual from the failed aggregate link-proxy attempt, floored at `1/6` thousand squared. The floor is the variance of the difference of two independently rounded whole-thousand levels. The 2003 initial schedule is absent, so its variance is the `1/6` floor. The recoverability parquet retains all 84 cell residuals and their level-cell and birth–death source keys for every attempted year; the table below is only the readable summary.

| Year | Representation | Status | Link attempt | Cells within ±0.5 / attempted | Max residual | Variance |
|---:|---|---|---|---:|---:|---:|
| 2003 | cumulative job change | inferred | inputs missing | 0 / 0 | — | 0.166667 |
| 2004 | cumulative job change | inferred | failed | 19 / 84 | 33.407275 | 65.199745 |
| 2005 | cumulative job change | inferred | failed | 20 / 84 | 61.890801 | 228.730081 |
| 2006 | cumulative job change | inferred | failed | 15 / 84 | 23.731192 | 51.300815 |
| 2007 | cumulative job change | inferred | failed | 16 / 84 | 34.228674 | 58.660143 |
| 2008 | cumulative job change | inferred | failed | 17 / 84 | 22.098039 | 18.249807 |
| 2009 | cumulative job change | inferred | failed | 13 / 84 | 40.264512 | 78.661208 |
| 2010 | cumulative job change | inferred | failed | 12 / 84 | 31.813475 | 91.563345 |
| 2011 | cumulative job change | inferred | failed | 16 / 84 | 20.359022 | 35.259417 |
| 2012 | cumulative job change | inferred | failed | 20 / 84 | 78.412368 | 520.337201 |
| 2013 | cumulative job change | inferred | failed | 11 / 84 | 37.113240 | 48.080295 |
| 2014 | cumulative job change | inferred | failed | 19 / 84 | 26.630662 | 68.188332 |
| 2015 | cumulative job change | inferred | failed | 8 / 84 | 63.751727 | 331.085586 |
| 2016 | cumulative job change | inferred | failed | 13 / 84 | 32.440472 | 81.254024 |
| 2017 | cumulative job change | inferred | failed | 14 / 84 | 8.913966 | 7.778542 |
| 2018 | cumulative job change | inferred | failed | 11 / 84 | 24.771480 | 93.717486 |
| 2019 | cumulative job change | inferred | failed | 17 / 84 | 13.991568 | 16.178823 |
| 2020 | cumulative job change | inferred | failed | 3 / 84 | 162.058501 | 1607.645923 |
| 2021 | cumulative job change | inferred | failed | 12 / 84 | 179.079008 | 1293.123145 |
| 2022 | cumulative job change | inferred | failed | 5 / 84 | 43.910696 | 157.995551 |
| 2023 | cumulative job change | inferred | failed | 14 / 84 | 74.270533 | 433.893346 |
| 2024 | cumulative job change | inferred | failed | 18 / 84 | 38.587326 | 108.426410 |
| 2025 | cumulative job change | inferred | failed | 11 / 84 | 17.229047 | 30.739396 |

## Consequences

- **Positive:** The public operator is exactly reproducible from committed data, revised birth–death enters explicitly, no private BLS series is fabricated, and every inferred year carries uncertainty into Stage 8.
- **Negative:** The components cannot identify the multiplicative matched-sample link separately from aggregation and source rounding. Their variance is an operator-reconstruction variance, not QCEW benchmark error or sampling variance.
- **Neutral:** November and December remain in the artifact with `additional_sample_receipts=True`; the April–October reproduction gate does not pretend their later sample receipts are a fixed link.

## Alternatives considered

- **Label the aggregate proxy recovered.** Rejected: every year fails at least 64 of 84 cells at publication rounding, and the required matched-sample inputs are not in a public source.
- **Infer link relatives from the target benchmark path and revised birth–death.** Algebraically possible but observationally identical to the chosen cumulative contributions; calling the quotient a recovered link would add interpretation without information.
- **Use both representations by year or sector.** Rejected by Req 10 and unnecessary: no year passes the recovery gate.
- **Treat inferred contributions as fixed without variance.** Rejected: it would turn rounded, underidentified public aggregates into private production inputs by assumption.

## Revisit triggers

Revisit if BLS supplies historical matched-sample weighted link relatives or sample-only job-change contributions with a documented vintage clock. A later source may replace a year's `inferred` row with `recovered` only if the link-relative operator reproduces every April–October published level within ±0.5 thousand and its reconstruction-error variance becomes exactly zero.
````

- [x] **Step 7: Document the package, command, and downstream contract**

In `README.md`, add this section after “Annual benchmark-source tables”:

````markdown
### Deterministic operators

Roadmap Stage 5 turns the Stage 3 vintage panel and Stage 4 annual tables into
sparse float64 JAX operators and audited Polars fixtures. Rebuild the gitignored
artifacts with:

```bash
uv run python scripts/operator_artifacts.py build
```

The build writes the 2003–2025 benchmark-wedge fixtures, inferred post-March
components, the link-relative recoverability table, release-specific B→M
components, and `operator-test-results.json` under `data/operators/`. The
[link-relative decision](docs/decisions/link-relatives.md) records why every
public year uses an inferred cumulative job-change representation. Later model
runs copy the JSON record into their provenance.
````

In both `AGENTS.md` and `CLAUDE.md`, replace the first sentence in `## Purpose` that says the package “so far holds” only the Kalman engine with:

```markdown
The Python package holds the marginalized Kalman engine (`src/ces_revisions/kalman.py`), the Stage 3 vintage panel (`src/ces_revisions/vintages/`), the Stage 4 annual-source tables (`src/ces_revisions/annual/`), and the sparse deterministic operator layer (`src/ces_revisions/operators/`).
```

Add this layout/tooling bullet in both files immediately after the Kalman bullet:

```markdown
- `src/ces_revisions/operators/` owns only deterministic preprocessing and sparse float64 JAX maps. Axis order is fixed in `operators/__init__.py`; benchmark years use December-to-January adjacent vintages; the selected public post-March representation is recorded in `docs/decisions/link-relatives.md`. Rebuild its gitignored parquet and `operator-test-results.json` with `uv run python scripts/operator_artifacts.py build`; every later run copies that record into provenance.
```

Add `uv run python scripts/operator_artifacts.py build` to each file's command block.

- [x] **Step 8: Run the focused artifact and document tests**

Run:

```bash
uv run pytest tests/test_operator_build.py tests/test_benchmark_operators.py tests/test_post_march_operators.py tests/test_seasonal_operators.py -q
uv run python scripts/operator_artifacts.py build
```

Expected: all focused tests pass; the CLI prints four artifact row counts and writes `data/operators/operator-test-results.json`; its five `checks` blocks equal the dictionary pinned in `test_operator_test_record_is_deterministic_and_complete`.

- [x] **Step 9: Run the full Stage 5 verification gate**

Run:

```bash
uv run ruff format src/ces_revisions/operators scripts/operator_artifacts.py tests README.md AGENTS.md CLAUDE.md docs/decisions/link-relatives.md specs/plans/8-ces-revisions.md
uv run ruff check src/ces_revisions/operators scripts/operator_artifacts.py tests
uv run ruff format --check
uv run pytest -m "not slow and not network"
uv run pytest -m slow
git diff --check
```

Expected: Ruff and `git diff --check` are clean; every fast hermetic test and every slow test passes; no network test runs; no file under `data/operators/` appears in `git status`.

- [x] **Step 10: Commit the assembled Stage 5 deliverable**

```bash
git add .gitignore README.md AGENTS.md CLAUDE.md docs/decisions/link-relatives.md scripts/operator_artifacts.py src/ces_revisions/operators/build.py tests/operator_data.py tests/test_operator_build.py
git commit -m "feat(operators): assemble Stage 5 artifacts"
```

## Completion record — 2026-09-16

All six tasks and all 42 tracked steps are complete. The implementation commits are:

- `333885a` — record the two omitted 2018 reconstruction events.
- `64d5379` — establish aggregation, missingness, and March-selection contracts.
- `13f02d7` — add the benchmark wedge and audited 2003–2025 fixtures.
- `bad6129` — implement both post-March maps and the all-years-inferred determination.
- `aa43c4b` — add joint NSA/SA and release-specific B→M maps.
- `08e7b5e` — assemble the artifacts, provenance record, CLI, decision, and documentation.

### Approved deviations

1. The two appended reconstruction-event descriptions are CSV-quoted because they contain
   commas. The generic parser also normalizes their blank footnote fields to null, which is
   pinned by the reconstruction tests.
2. `scripts/operator_artifacts.py` enables JAX float64 before importing the operator build.
   Pytest still enables float64 centrally, but the standalone CLI cannot rely on pytest's
   process initialization. A fresh-interpreter regression test pins this ordering.

### Whole-plan review resolution

The post-implementation review found no critical issues. Its three important findings and one
minor finding were resolved in four additional commits:

- `d4e29d0` separates the scope-adjusted `b_fin` wedge anchor from the archived post-
  reconstruction March level and runs every year/sector fixture through the public API.
- `f522b69` gives every traced sparse constructor fixed structural metadata and adds direct JIT
  tests for aggregation, wedge reconstruction, and both post-March representations.
- `ad54bad` constructs observation selectors directly from sparse coordinates, avoiding the
  quadratic dense identity allocation on the 37,032-cell panel.
- `40e3aaa` verifies the CLI's float64 initialization in a fresh interpreter.

A follow-up review also resolved two gaps in that first fix pass: every public seasonal
constructor now has trace-stable sparse metadata, and reconstruction support uses structural
BCOO exclusion so unsupported NaN terms cannot contaminate eager or JIT results. The spec,
decision record, and a 276-series cross-artifact test now distinguish the scope-comparable wedge
anchor from the published post-reconstruction March level. Stage 8 remains routed to
`brainstorming` because probabilistic B→M composition and variance injection are outside Stage 5.

### Roadmap and deferred-item audit

Stages 6 and 7 remain valid unchanged. Stage 8 now consumes positive reconstruction-error
variance for all 23 inferred benchmark years but remains routed to `brainstorming` for its
probabilistic B→M composition and variance-injection design; Stage 9 must preserve those terms
through the joint NSA/SA map. Stages 17, 20, 23, and 24 now explicitly carry, audit, cite, or
disclose the all-years-inferred determination. No Stage 5 work was skipped, no review finding
remains unresolved, and no new deferred item was created. Existing deferred items remain
assigned to their documented future stages or revisit triggers.

## Completion gate

Before invoking the writing-plans Plan Completion Protocol, verify these observable outcomes from fresh commands and generated artifacts:

1. `operator-test-results.json` reports 496,310 exact aggregation cells; 23 wedge years, 3,312 cells, and 215 date-bounded reconstruction-supported cells; 1,932 April–October and 552 flagged November/December post-March cells; 23 inferred cumulative years and zero recovered years; zero seasonal identity residual.
2. Every fixture row satisfies `reproduced_level_thousands == benchmark_level_thousands`. Every nonzero reconstruction term has at least one `event_id`; no non-event row has a reconstruction term.
3. `docs/decisions/link-relatives.md` and the recoverability parquet each have one row for every year 2003–2025, select only `cumulative_job_change`, and carry positive variance exactly for inferred rows.
4. The public API raises when both or neither post-March representations are supplied. The empirical component table contains only the cumulative representation and consumes Stage 4's revised birth–death values.
5. B→M right-censored values stay null; NSA and SA mature horizons stay distinct; every observed SA row satisfies the joint-measurement identity.
6. No model fit, prior, likelihood, X-13 wrapper, or new dependency entered the branch.

At plan completion:

- run the resolve-before-defer gate and fix every unblocked skipped step or review finding;
- tick completed plan steps and add deviations/skips only after that partition is final;
- tick Stage 5 in `specs/ces-revisions-roadmap.md` and add this exact Rollout-note stamp to `specs/ces-revisions.md` with the execution date: `Stage 5: COMPLETE (YYYY-MM-DD) — implemented by plan 8 (specs/plans/completed/8-ces-revisions.md). Next: resume the roadmap.`;
- re-validate roadmap Stages 6–9, 17, 20, 23, and 24 against the all-years-inferred decision. In particular, Stage 8 must propagate nonzero reconstruction variance for every 2003–2025 year, not only selected historical years;
- update `specs/deferred_items.md` by ticking any earlier item this work closes and appending only genuinely deferred items under `## 8-ces-revisions — <date>`;
- run the deferred-backlog statistics and present any required triage proposal;
- move this plan to `specs/plans/completed/8-ces-revisions.md`, repair its relative links one directory deeper, and commit the retirement as `chore(specs): retire plan 8`.
