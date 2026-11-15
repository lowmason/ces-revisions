# Stage 2 — Finish the Aggregate News/Noise Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via subagent-driven-development (the default) — or executing-plans when your human partner chose inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

> Roadmap: specs/ces-revision-drivers-roadmap.md, Stage 2 — on plan completion, tick the stage and
> re-validate later stages against what shipped.

**Goal:** Bring plan 11's aggregate pilot and run harness onto this branch without its GPU work, record the seasonal-state decision, and report the posterior news shares of the total-nonfarm revisions in `specs/ces-revision-drivers-stage-2-findings.md`.

**Architecture:** Task 1 merges plan 11's branch and checks that the merged code reproduces plan 11's committed evidence. Task 2 runs plan 11's Task 14 without its GPU half. Task 3 adds `ces_revisions.pilot.news_shares` and two `scripts/pilot_run.py` subcommands: `shares` reads the pilot's gitignored run directory and writes a small committed JSON, and `shares-report` renders the findings table from that JSON alone. Task 4 writes the findings. Task 5 reviews the whole branch and retires plans 11 and 13.

**Tech Stack:** Python 3.14, JAX, NumPyro, ArviZ 1.x and xarray as plan 11 locks them, NumPy, pytest, ruff. No new dependency.

**Spec:** none. The plan's source is the Stage 2 entry of the roadmap (`specs/ces-revision-drivers-roadmap.md:71-79`), its choices (`:18`, `:24`), and plan 11 (`specs/plans/11-ces-revisions.md` on branch `claude/ces-revisions-stage-6-bd9ab7`). Before Task 1, read plan 11 with `git show claude/ces-revisions-stage-6-bd9ab7:specs/plans/11-ces-revisions.md`. References such as `11:254` are its line numbers. Executors read all three.

## Global Constraints

- **Plan 11's constraints** (`11:16-33`) hold for every file this plan touches. Two lines do not: Owner gates, because no task here rents a GPU, and Scope, whose stage numbers belong to the parked roadmap.
- **No GPU work** (roadmap `:24`): no Lambda instance, no `choose-gpu` run and no `docs/runs/stage-6/gpu-choice.json`. Plan 11's GPU code ships as built and waits for O6 (F7).
- **History:** merge, never rebase or squash, on this branch and in the PR to `main`. Plan 11's run records cite their commits, for example `c0fddf5` with `"dirty": false` in `docs/runs/stage-6/pilot-fit/provenance.json`. A rewrite orphans the commits that the evidence names.
- **Ignored workspaces:** `data/runs/` and `.sdd/` stay gitignored, and no `inference_data.nc` is committed. New evidence goes under `docs/runs/stage-6/` (F6).
- **Branch:** commit on this plan's branch, `claude/cool-goodall-842bfb`, cut from `origin/main` at `9861d27` (PR 13). Nothing is committed on `main` (D1).
- **Style:** stage files by explicit path. Run `uv run ruff format` and `uv run ruff check` on the files each commit touches. New Markdown follows CLAUDE.md's rules for `specs/`.

## Review Focus

1. A run directory fitted to other data than the current panel, as after a BLS refresh rebuilds `data/panel/`: `shares` refuses, as `compare-seasonal` does, and reports no shares for data that the findings do not describe (Task 3).
2. A clone with no `data/runs/`, such as any fresh clone or this worktree once archived: `shares` stops and names plan 11's refit command, and `shares-report` still renders the table from the committed JSON (Task 3).
3. Draws taken from one chain, or left grouped by chain: the summary pools the 1,000 draws of all four chains (Task 3).
4. `share_t` read as an F→T share, because Stage 1 reports F→T: it is the news share of the S→T revision, and the table says so (Task 3).
5. A merged tree that computes other evidence than the branch did: `compare-seasonal` on the copied run directories reproduces the committed `seasonal-comparison.json` byte for byte (Task 1).

## Decisions

- **D1** (owner, brief, 2026-11-15): commit nothing on `main`. Pushing is the owner's call. Applied in: Global Constraints, Task 5.
- **D2** (owner, brief, 2026-11-15): plan 11's branch stays untouched until this plan is approved. Task 1 merges it and never rewrites it. Applied in: Task 1.

## Defaults settled in this plan

These wait for the owner's review.

- **F1 (process):** execute inline with executing-plans in one fresh session, as Stage 1 did (roadmap `:23`). Every task works on Task 1's merged tree, and Tasks 4 and 5 need the tasks before them, so subagent dispatch gains little. Keep test-first for Tasks 2 and 3, and one whole-branch Opus review in Task 5, which is also plan 11's missing final review. If the owner agrees, the header names executing-plans.
- **F2 (merge direction):** Task 1 merges plan 11's branch into this plan's branch. Merging `main` into plan 11's branch gives the same tree, but it changes plan 11's branch before the owner's merge.
- **F3 (interval):** the median and the central 90% interval (the 5th to 95th percentile) of the 4,000 pooled draws. Plan 11's April 2020 check uses the same 90%. ArviZ's 94% HDI is the alternative.
- **F4 (stamp):** the Stage 2 stamp names plan 13 with plan 11. Both plans retire in one commit, each with its history entry.
- **F5 (parked files):** the parked roadmap and spec get nothing beyond the Req 6 and Req 7 amendment that the merge carries. The parked roadmap says "Do not resume this file", so its Stage 6 stays unticked and the spec's Rollout note gets no completion line.
- **F6 (evidence path):** `docs/runs/stage-6/` keeps the parked stage number, because plan 11 and the run records cite those paths. `news-shares.json` joins it.
- **F7 (GPU code):** `pilot/gpu_choice.py`, `pilot/dimensions.py`, `scripts/gpu_probe_grid.py`, the `choose-gpu` subcommand and their tests reach `main` unused, for O6. Removing them would make O6 restore and re-review them.
- **F8 (what the shares mean):** the findings report `share_s` and `share_t` as `derived()` defines them (`src/ces_revisions/pilot/model.py`): the news variance's share of each revision's variance, without the rounding variance. The 2003 shift scales the S and T news and all three noise SDs by one factor, so the shares hold before and after May 2003. The crisis multiplier scales only the news in March–June 2020, so the shares are larger there. The findings state both as limits and report no separate pre-2003 or crisis pair.

## Planning evidence (2026-11-15)

- Drift: plan 11's branch is 22 commits ahead of `origin/main` (`9861d27`) and 29 behind. The roadmap's "2 behind" (`:75`) was measured against an older local `main`.
- `git merge-tree` of the two is clean. Rename detection carries plan 11's Task 1 amendment to `specs/completed/ces-revisions.md`, and `specs/ces-revisions.md` does not come back.
- On a throwaway commit of that merge, `uv lock --check` passes, and the fast tier gives `679 passed, 22 deselected` in about 5.5 minutes, and the slow tier gives `18 passed, 683 deselected` in about 5 minutes. After the three builds, `compare-seasonal` on the three run directories reproduces the committed `seasonal-comparison.json` byte for byte, in 80 s with the builds. So the rebuilt panel still has the hash that the runs were fitted to.
- Plan 11's ledger (`.sdd/11-ces-revisions/progress.md`) records Tasks 1–12 complete and Task 13 stopped after Step 1. Its Minor findings wait for a final review that never ran.
- The ignored copies: the main checkout holds `data/runs/` and `.sdd/11-ces-revisions/`. This plan's worktree sits at plan 11's old worktree path, and it still holds identical copies of both (`diff -rq`). The pilot's `inference_data.nc` has the SHA-256 `b0480b9795f80b5d56bf32163b01a1e62c44381fba99f94aff7f9a759165c0af`. `main` already ignores `data/runs/` (`63d923d`).
- `main` already ticked plan 5's evidence-tests item and plan 1's `A_t`/`Z_t` item (`specs/history.md`, `/deferred — 2026-11-15`). Plan 11's Execution notes and Retirement lines on them (`11:260`, `11:54-64`) no longer apply.
- `derived()` applied to the pilot's 4,000 posterior draws gives `share_s` 0.7232 (90%: 0.5532–0.8526) and `share_t` 0.6520 (90%: 0.3002–0.8466).

---

### Task 1: Merge plan 11 and check its evidence

**Files:** the merge commit brings plan 11's 54 files. No other file changes.

**Interfaces:**
- Produces: `ces_revisions.pilot`, `ces_revisions.runs` and `scripts/pilot_run.py` on this branch. The run directories in `data/runs/`, plan 11's ledger in `.sdd/11-ces-revisions/`, and the built `data/panel/`, `data/annual/panel/` and `data/operators/`.

- [ ] **Step 1: Merge.** Run `git merge --no-ff claude/ces-revisions-stage-6-bd9ab7 -m "Merge plan 11's aggregate pilot for roadmap Stage 2"`. Expected: no conflict. `specs/ces-revisions.md` does not exist, and `grep -c 'Lowell, plan 11, 2026-10-21' specs/completed/ces-revisions.md` prints 2.
- [ ] **Step 2: Check the ignored workspaces.** If `data/runs/` is missing, run `cp -Rp /Users/lowell/Projects/ces-revisions/data/runs data/`. If `.sdd/11-ces-revisions/` is missing, run `mkdir -p .sdd` and `cp -Rp /Users/lowell/Projects/ces-revisions/.sdd/. .sdd/`, which also copies the `.gitignore` of `*`. Expected: `diff -rq data/runs /Users/lowell/Projects/ces-revisions/data/runs`, the same `diff` for `.sdd/11-ces-revisions`, `cmp data/runs/stage6-pilot-seed2995-20261022T020454Z/provenance.json docs/runs/stage-6/pilot-fit/provenance.json` and `git status --short` print nothing. `shasum -a 256` of that run's `inference_data.nc` prints the hash in Planning evidence.
- [ ] **Step 3: Sync and build.** Run `uv sync --locked`, then the three builds of `11:257`. Expected: each build finishes.
- [ ] **Step 4: Run the tiers.** Run `uv run pytest -m "not slow and not network" -q`, then `uv run pytest -m slow -q`. Expected: `679 passed, 22 deselected`, then `18 passed, 683 deselected`. A failure stops the task. Fix it in its own commit, with a Deviation note here.
- [ ] **Step 5: Reproduce the seasonal comparison** (Review Focus 5). Run plan 11's Task 12 Step 3 commands (`11:9577-9580`) with `--out /tmp/seasonal-comparison.json`, then `cmp /tmp/seasonal-comparison.json docs/runs/stage-6/seasonal-comparison.json`. Expected: no output. A difference stops the plan. Report the `diff`.

### Task 2: The seasonal-state decision record (plan 11's Task 14, re-scoped)

**Files:**
- Create: `src/ces_revisions/pilot/decision_tables.py`, `docs/decisions/seasonal-state.md`
- Test: `tests/test_seasonal_state_decision.py`
- Modify: `CLAUDE.md`, `README.md`, `specs/plans/11-ces-revisions.md` (a Deviation note under its Task 14)

**Interfaces:**
- Consumes: `docs/runs/stage-6/seasonal-comparison.json`, and the Mac record that its `grid.path` names (`docs/decisions/cloud-gpu-probe/mac-grid.json`).
- Produces: `decision_lines(comparison: dict) -> list[str]`, `seasonal_rows(comparison: dict) -> list[str]` and `timing_rows(records: list[dict], layout_name: str) -> list[str]`, and `python -m ces_revisions.pilot.decision_tables COMPARISON`, which prints the line and the two tables. Plan 11's `gpu_rows` and `grid_records` are not built.

Run plan 11's Task 14 Steps 2–10 (`11:9744-10366`) with these changes.

- [ ] **Step 1: Skip plan 11's Step 1,** the GPU rule. Under plan 11's Task 14 heading, add `> Deviation: re-scoped to the seasonal-state decision alone by roadmap Stage 2 (specs/ces-revision-drivers-roadmap.md:75). Step 1, the GPU table and gpu-choice.json moved to O6. Plan 13's Task 2 executed the rest.`
- [ ] **Step 2: Write the tests** of plan 11's Step 2, changed:
  - `test_decision_lines_name_the_layout`: ``decision_lines(COMPARISON) == ["The seasonal state for O6 and every later fit of this model is `dummy`."]``
  - Drop `test_an_incomplete_gpu_measurement_has_no_decision_lines` and `test_gpu_rows_escape_dollars_and_mark_unmeasured_cells`. Keep the seasonal-rows and timing-rows tests as written.
  - `test_the_decision_record_states_its_evidence`: read the Mac record from the comparison's `grid.path`. Assert that every line of `decision_lines`, `seasonal_rows` and `timing_rows([mac], layout)` is in the record, and that the timing rows are not empty.
- [ ] **Step 3: Run them to verify they fail.** Run `uv run pytest tests/test_seasonal_state_decision.py -q`. Expected: `ModuleNotFoundError: No module named 'ces_revisions.pilot.decision_tables'`.
- [ ] **Step 4: Write the module** of plan 11's Step 4, without `gpu_rows`, `grid_records` and the GPU lines, with a `python -m` entry that takes only COMPARISON. Run the Step 3 command. Expected: 3 passed and 1 failed, the record test, with `FileNotFoundError`.
- [ ] **Step 5: Write the record** from plan 11's Step 5 template (`11:10141-10204`), changed:
  - Title: one that states the layout the rule chose, for example "Keep the dummy seasonal state".
  - Blast radius: the seasonal block of every sector's state in O6 and the later model outcomes.
  - Context: in the first paragraph, drop the sentences from "Roadmap Stage 6 makes that choice" through the D5 sentence. In the second, drop each GPU mention: the Lambda instances, `gpu_choice.py` and the GPU rule's constants. Point the spec link to `../../specs/completed/ces-revisions.md`. Add that roadmap Stage 2 moved the GPU choice to O6 (`specs/ces-revision-drivers-roadmap.md:24`).
  - Decision: the layout line only. Drop the paragraph on Stages 10–17.
  - Keep "The seasonal rule" as written.
  - Replace "The GPU measurement" and "The GPU rule" with "The cost on the Mac": the first paragraph of "The GPU measurement" without its GPU sentences, one sentence saying that the table's Stage 7, 8 and 9 are the parked roadmap's model stages, which O6 now covers, then the timing table of the Mac record.
  - Consequences: the first bullet, with "Stages 7–17" read as "O6 and the later model outcomes". Drop the GPU bullet. The Ljung–Box bullet names O17 for Stages 18 and 21.
  - Alternatives considered: the layout bullets only. The GPU, one-GPU and parallel-filter bullets go to the O6 deferred item (Task 5).
  - Revisit triggers: only "A posterior predictive check in O17 traces a misfit to the seasonal block."
- [ ] **Step 6: Run the test to verify it passes.** Run the Step 3 command. Expected: 4 passed.
- [ ] **Step 7: Update CLAUDE.md and README.md** as plan 11's Step 7 states, changed:
  - In the first new CLAUDE.md bullet, replace "and the two pre-registered decision rules (`seasonal.py`, and `gpu_choice.py`, which chooses a GPU for each Stage 7–9 model) recorded in `docs/decisions/seasonal-state.md`" with "the pre-registered seasonal rule (`seasonal.py`), recorded in `docs/decisions/seasonal-state.md`, and the GPU rule (`gpu_choice.py`), which waits for O6".
  - In the third bullet, the test sentence names only `docs/runs/stage-6/seasonal-comparison.json` and the Mac grid record that it names.
  - Drop the CLAUDE.md "GPU type and count" edit and the README "Stage 6 … measures them" edit. Their old text is gone from `main`, which already names O6.
  - For the README status tree, replace `README.md:36-40`, from `docs/decisions/` through `engine_probe.py times the engine`, with the block below.

```text
docs/decisions/                       decision records (engine.md: the Kalman engine;
                                      link-relatives.md: the post-March operator;
                                      seasonal-state.md: the pilot's seasonal state)
  cloud-gpu-probe/                    engine timing records: the Mac baseline and the Mac grid
docs/runs/stage-6/                    the aggregate pilot's committed evidence
src/ces_revisions/                    Python package (src layout): kalman.py is the state-space engine,
                                      vintages/ and annual/ build the Stage 3/4 inputs, operators/ the
                                      Stage 5 maps, pilot/ the aggregate pilot, runs/ the run harness,
                                      devices.py picks the chain method, engine_probe.py times the engine
```

- [ ] **Step 8: Run every tier** as plan 11's Step 8 states. Expected: the fast tier gives `683 passed, 22 deselected`, and the slow tier and ruff pass.
- [ ] **Step 9: Commit** the files above with `git commit -m "Record the seasonal-state decision without the GPU choice"`.

### Task 3: The news shares

**Files:**
- Create: `src/ces_revisions/pilot/news_shares.py`, `docs/runs/stage-6/news-shares.json`
- Modify: `scripts/pilot_run.py`, `CLAUDE.md` (the two subcommands in the `scripts/pilot_run.py` bullet, and one Commands line: `uv run python scripts/pilot_run.py shares-report docs/runs/stage-6/news-shares.json   # rewrite the Stage 2 findings table from the committed record (offline)`)
- Test: `tests/test_pilot_news_shares.py`

**Interfaces:**
- Consumes: `derived(params: dict) -> dict` and `SAMPLED_SITES` (`pilot/model.py`). In `scripts/pilot_run.py`: `load_panel`, `panel_arrays`, `read_run`, and `array_sha256` (`runs/provenance.py`).
- Produces, in `ces_revisions.pilot.news_shares`:
  - `INTERVAL_PROB = 0.90`, and `SHARES = {"share_s": "F→S", "share_t": "S→T"}`, the revision that each share splits.
  - `share_draws(posterior: xr.Dataset) -> dict[str, np.ndarray]`: `derived()` applied to the sampled sites, with `chain` and `draw` pooled into one axis, for each key of `SHARES`.
  - `summarize(draws: np.ndarray, prob: float = INTERVAL_PROB) -> dict[str, float]`: `median`, `lower` and `upper`, the `np.quantile` values at 0.5, (1 − prob)/2 and (1 + prob)/2.
  - `shares_block(record: dict) -> str`: a table with the columns Revision, Share, Median and 90% interval, one row per share with values to three decimals, then one line that names the number of draws, the run directory and `sampling_seconds` to one decimal.
- Produces, in `scripts/pilot_run.py`:
  - `shares_record(run_dir: Path, panel_sha256: str) -> dict`. It reads `provenance.json` first, and raises `ValueError` naming "other data" when its `arrays.sha256` differs, before it opens the netCDF file. The record holds `run_dir`, `inference_data_sha256`, `panel_sha256`, `num_draws`, `interval_prob`, `sampling_seconds` (from `sampling.timings` of the provenance) and the `summarize` result of each share.
  - `shares RUN_DIR --out PATH` writes the record as sorted, indented JSON. It checks `RUN_DIR` before it loads the panel. When `RUN_DIR` is missing, it returns 2 and prints plan 11's refit command, `uv run python scripts/pilot_run.py fit --layout dummy --label stage6-pilot`, on stderr.
  - `shares-report RECORD [--findings PATH]` reads only the JSON. It rewrites the generated block `stage-2-shares` of the findings file (default `specs/ces-revision-drivers-stage-2-findings.md`) with `archive_inventory.replace_generated`, as `scripts/collection_window.py` does.

- [ ] **Step 1: Write the failing tests** in `tests/test_pilot_news_shares.py`:
  - `test_share_draws_pool_every_chain_and_draw` (Review Focus 3): a posterior Dataset of 4 chains × 3 draws, in which each chain holds its own constant set of sampled sites. Each output has shape `(12,)` and 4 distinct values, and each chain's value equals `derived()` of that chain's sites.
  - `test_summary_is_the_median_and_central_interval`: `summarize(np.linspace(0, 1, 101)) == pytest.approx({"median": 0.5, "lower": 0.05, "upper": 0.95})`
  - `test_the_block_names_the_revision_each_share_splits` (Review Focus 4): with `share_s` at median 0.7234, lower 0.6 and upper 0.81, the block holds ``| F→S | `share_s` | 0.723 | 0.600–0.810 |``, and the `share_t` row starts ``| S→T | `share_t` |``.
  - `test_shares_refuse_a_run_fit_to_other_data` (Review Focus 1): a `tmp_path` run directory that holds only a `provenance.json` whose `arrays.sha256` is `"f" * 64`. `pilot_run.shares_record(run_dir, "0" * 64)` raises `ValueError` matching "other data".
  - `test_shares_name_the_refit_when_the_run_is_gone` (Review Focus 2): `pilot_run.main(["shares", str(tmp_path / "missing"), "--out", str(tmp_path / "x.json")]) == 2`, and `capsys`'s stderr holds `pilot_run.py fit --layout dummy --label stage6-pilot`.
  - `test_shares_report_needs_only_the_record` (Review Focus 2): with a record JSON and a findings file that holds an empty `stage-2-shares` block, both in `tmp_path`, `pilot_run.main(["shares-report", <record>, "--findings", <findings>]) == 0`, and the file holds the F→S row.
- [ ] **Step 2: Run them to verify they fail.** Run `uv run pytest tests/test_pilot_news_shares.py -q`. Expected: `ModuleNotFoundError: No module named 'ces_revisions.pilot.news_shares'`.
- [ ] **Step 3: Implement** the module and the two subcommands as the Interfaces state.
- [ ] **Step 4: Run the tests to verify they pass.** Run the Step 2 command. Expected: 6 passed.
- [ ] **Step 5: Make the real record.** Run `uv run python scripts/pilot_run.py shares data/runs/stage6-pilot-seed2995-20261022T020454Z --out docs/runs/stage-6/news-shares.json`. Expected: `num_draws` 4000, `sampling_seconds` 1659.6018157910003, `panel_sha256` starting `c197e079`, `inference_data_sha256` starting `b0480b97`, `share_s` 0.7232 (0.5532–0.8526) and `share_t` 0.6520 (0.3002–0.8466) to four decimals. Another value stops the task. Report it.
- [ ] **Step 6: Commit** the five files with `git commit -m "Report the pilot's posterior news shares"`.

### Task 4: The findings file

**Files:** Create `specs/ces-revision-drivers-stage-2-findings.md`.

- [ ] **Step 1: Write the skeleton** with the sections Question, Answer, Data and model, Results, Limits and Measured cost. Results holds the empty block between `<!-- BEGIN GENERATED stage-2-shares -->` and `<!-- END GENERATED stage-2-shares -->`.
- [ ] **Step 2: Render the table.** Run `uv run python scripts/pilot_run.py shares-report docs/runs/stage-6/news-shares.json` twice. Expected: the block holds the two rows of Task 3 Step 5, and `shasum` of the file is the same after each run.
- [ ] **Step 3: Write the prose.**
  - Answer: two or three sentences from the generated numbers.
  - Data and model: plan 11's pilot, with the dummy layout (`docs/decisions/seasonal-state.md`), Req 18 met (`docs/runs/stage-6/pilot-fit/diagnostics.json`), and the reduced-panel SBC's share contraction and |z| > 3 share (`docs/runs/stage-6/sbc-summary.json`).
  - Limits: the two of F8, and the excluded rounding variance. The shares are model-implied variance shares, not a split of each month's revision. The model has one aggregate NSA series, and benchmark and seasonal revisions stay outside it (O6). SBC ran on the reduced 1999–2021 panel only, and the doubled-prior rerun missed Req 15's constraint (`docs/runs/stage-6/prior-predictive-doubled.json`), so full-panel SBC and prior sensitivity wait for O17.
  - Measured cost: the generated sampling time, which covers compilation, warmup and sampling of four chains on the Mac's CPU on 2026-10-22, and the sessions that Stage 2 used, counted at completion.
- [ ] **Step 4: Verify.** Run `uv run pytest -m "not slow and not network" -q`, `uv run ruff check` and `uv run ruff format --check`. Expected: all pass.
- [ ] **Step 5: Commit** with `git commit -m "Write the Stage 2 news/noise findings"`.

### Task 5: Final review and completion

**Files:** Modify `specs/plans/11-ces-revisions.md`, this plan, `specs/deferred_items.md`, `specs/ces-revision-drivers-roadmap.md` and `specs/history.md`. Move both plans to `specs/plans/completed/`.

- [ ] **Step 1: Review the whole branch.** Dispatch code-reviewer (Opus) over `9861d27..HEAD`, against plan 11 (Tasks 1–12, and Task 14 as Task 2 re-scopes it) and this plan. Put the Minor findings of `.sdd/11-ces-revisions/progress.md` in the dispatch. Handle each finding with receiving-code-review, and commit each fix on its own.
- [ ] **Step 2: Run the completion gate** (writing-plans' Plan Completion Protocol, step 1) for both plans. Plan 11's leftovers are Task 13, and Step 1 and the GPU half of Task 14. Defer them to O6.
- [ ] **Step 3: Mark up plan 11.** Tick the steps done in Tasks 1–12 and 14. Under Task 13 and under Step 1 of Task 14, add `> Skipped: GPU work moved to O6 by roadmap Stage 2 (:24) → deferred`. Add the ledger's deviations, such as the owner decision of Task 7 (`progress.md:15`). Replace its `Roadmap:` lines (`11:5-6`) with the two stamp lines of this plan's header, which name this roadmap's Stage 2. Point its spec and roadmap links to `specs/completed/` from `specs/plans/completed/`, and add its status line. Its Retirement block (`11:54-64`) changes as follows:
  - Item 1: drop the plan 5 rewrite. Append `## 11-ces-revisions — YYYY-MM-DD` with its three items and these Revisit-if lines: for `σ_slope`, "O6's fit or O17's prior-predictive gate still leaves it above 3 prior scales". For the doubled prior, "O17's prior-sensitivity run". For the residual seasonality, "O17's posterior predictive suite traces a misfit to the seasonal block". Add a fourth item: plan 11's Task 13 and the GPU half of Task 14 (the \$40 Lambda measurement, the GPU rule, `gpu-choice.json`, and D8's parallel-filter note). Size: plan. Covered by O6. Done when: the stage that delivers O6 ticks. Add the unfixed review findings.
  - Item 2: plan 11's Handoff section points to this plan's.
  - Items 3 and 4: not done (F5).
- [ ] **Step 4: Tick the two planned items** in `specs/deferred_items.md` with `→ done YYYY-MM-DD`: plan 1's dispersed chain starts (`runs/inits.py`, with each run's `init` provenance) and plan 5's repeated EMPLOY vintage (`employ_without_repeats`).
- [ ] **Step 5: Mark up this plan** and write its Handoff section. Stage 8 consumes the findings file. Its table is the generated block that `shares-report` rewrites from the committed `docs/runs/stage-6/news-shares.json`, so Stage 8 needs no run directory. Name the traps: the run directory is gitignored, a refit (plan 11's Task 10 Step 5) followed by `shares` regenerates the JSON, `share_t` splits the S→T revision, and `docs/runs/stage-6/` keeps the parked stage number. Name the differences: the 29-commit drift, and the deferred items that go to O6 and O17 by their section.
- [ ] **Step 6: Tick Stage 2 and stamp it** in the roadmap:\
  `> Stage 2: COMPLETE (YYYY-MM-DD) — implemented by plan 13 (specs/plans/completed/13-ces-revision-drivers-stage-2.md) with plan 11 (specs/plans/completed/11-ces-revisions.md). Its Handoff section names what Stage 8 consumes.`\
  `> Next: resume the roadmap.`
- [ ] **Step 7: Retire.** Report `deferred_stats.py`'s summary, run `deferral_scan.py` on both plans, append a history entry for each plan, and `git mv` both plans to `specs/plans/completed/` in one `chore(specs): retire plans 11 and 13` commit.
- [ ] **Step 8: Finish** with finishing-a-development-branch. The PR merges with a merge commit. Before the worktree is archived, copy this plan's `.sdd/` ledger to the main checkout's `.sdd/`.
