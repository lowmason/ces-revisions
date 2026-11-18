# Stage 3 — Seasonal Share of SA Revisions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via executing-plans (default F1). Steps use checkbox (`- [ ]`) syntax for tracking.

> Roadmap: specs/ces-revision-drivers-roadmap.md, Stage 3 — on plan completion, tick the stage and re-validate later stages against what shipped.

Process (default F1): Stage 1's lighter process, without plan 15's two additions. Test first for the share computation and the findings generator. executing-plans' mutant gate at each task and its final whole-branch Opus review, with its Codex seat, stay. No per-task reviewers, no subagent dispatch, and no code blocks in this plan.

**Goal:** Report, for each sector, transition and regime of the built accounting decomposition, the share of SA revision variance that comes from the change in seasonal adjustment, with the identity gap shown, in `specs/ces-revision-drivers-stage-3-findings.md` (O9).

**Architecture:** A new module, `ces_revisions.seasonal_share`, reads the 110-row `data/panel/accounting_decomposition.parquet` that plan 5's vintage build writes. It adds three shares to each row and leaves `ces_revisions.vintages.accounting` and the build artifact unchanged. `scripts/seasonal_share.py report` writes one generated table per transition and one table of the shares' bounds into the findings file, as Stages 1 and 9 do.

**Tech Stack:** Python 3.14, Polars, pytest, ruff. No new dependency.

**Spec:** none. The plan's source is the Stage 3 entry of the roadmap (`specs/ces-revision-drivers-roadmap.md:90-98`, as of `7dfe067`), Stage 8's Exit (`:146`) and the Req 9 identity in `src/ces_revisions/vintages/accounting.py`. Executors read all three.

## Global Constraints

- **The identity (accounting module docstring).** `r^S = r^N − Δa`, so within a group `Var(r^S) = Var(r^N) + Var(Δa) − 2 Cov(r^N, Δa)`. `variance_identity_gap` is that equation's float residual, at most about 1.5e-11. It is rounding error, not a fault.
- **The three shares (default F2).** With `V = var_revision_sa` and `C = cov_revision_nsa_adjustment_change`:
  - `share_ratio = var_adjustment_change / V`
  - `share_adjustment = (var_adjustment_change − C) / V`, the covariance split, which is `Cov(r^S, −Δa) / V`
  - `share_nsa = (var_revision_nsa − C) / V`, its complement: `share_adjustment + share_nsa = 1 − variance_identity_gap / V`
  None is bounded in [0, 1]. Never clip, drop or flag a row for its value.
- **Input.** `data/panel/accounting_decomposition.parquet` only, rebuilt offline by `uv run python scripts/vintage_sources.py build`. Copy nothing from the main checkout. `ces_revisions.vintages.accounting`, the build and `data/raw/` stay unchanged.
- **Rows.** 110: 108 `cesvinall` rows (12 sectors × F→S, S→T, T→B × 2003-05 to 2019-12, 2020 to 2022, 2023 on) and 2 `cesnaicsrev` rows (total nonfarm, F→S and S→T, 1979-01 to 2003-04). Regimes sort in the order of `accounting.REGIMES`, never alphabetically, which would put `pre_2003` last.
- **No interval (default F3).** The shares are descriptive and carry no interval. The findings say so, as Stage 8's Exit expects.
- **No model run.** Stage 3 fits no NumPyro model, so the roadmap's Where models run line (`:29`) sends nothing to Lambda.
- **Naming.** Committed text outside `specs/` names this work by O9 or by the findings file, never as "Stage 3" alone: CLAUDE.md's "Stage 3" means the parked roadmap's vintage panel.
- **Tests.** New tests are hermetic and unmarked. The gate is `uv run pytest -m "not slow and not network"`.
- **Style.** `uv run ruff check` and `uv run ruff format --check` pass. The findings file follows CLAUDE.md's GitHub-renderable Markdown rules.

## Review Focus

1. Shares above 1 or below 0, such as the total-nonfarm S→T ratio of 9.93 before 2003 and the 27 F→S covariance splits below 0: rendered as computed, never clipped or dropped (Tasks 1 and 2).
2. The identity gap of about 1e-11: shown in scientific notation, so it never reads as `0.00` (Task 2).
3. Regime order: each sector's rows run from 1979 or 2003 forward, and T→B has no row before 2003 (Tasks 1 and 2).
4. A group with a null or zero SA revision variance, which a later vintage refresh with a 1-month regime would give: refused with the row named, never a division to `inf` or `NaN` (Task 1).
5. A checkout with no `data/panel/`: the report exits non-zero and names the build command (Task 2).

## Decisions

None.

## Defaults settled in this plan

Each is put to the owner at plan review as a flagged default.

- **F1 (brief): process.** Stage 1's process (roadmap `:23`, the owner's choice for Stage 1 only), without plan 15's two additions (`specs/plans/completed/15-ces-revision-drivers-stage-9.md:51`). executing-plans runs inline. Test first covers the share computation and the findings generator. executing-plans' own gates stay, as in plans 12 (D3) and 15: the mutant gate at each task and the final whole-branch Opus review with its Codex seat. No per-task reviewer, no subagent, and test steps as names with inline assertions. Effect if wrong: the plan carries more or less review than the owner wants.
- **F2 (brief): the share.** The findings report the plain ratio and the covariance split side by side, and the NSA split beside them, and state that none is bounded in [0, 1]. On the 110 rows, the ratio is above 1 in 12 rows (maximum 9.93). The covariance split is below 0 in 27 rows, all of them F→S, and above 1 in 2. The two splits add to 1 up to the identity gap, which makes the split the one that partitions `Var(r^S)`. Effect if wrong: O9's headline numbers change meaning.
- **F3 (brief): generated tables, no interval.** `scripts/seasonal_share.py report` rebuilds every table offline from `data/panel/`, so Stage 8 can regenerate them. The shares carry no interval: an interval needs the month-level terms, which `data/panel/` does not hold. Effect if wrong: Stage 8 cannot regenerate the numbers, or calls a result with an interval descriptive.
- **F4: sector titles.** Each row reads `<code> <title>`, from a `SECTOR_TITLES` map in the script. A test ties every title to its code through `_LEGACY_SECTOR_BY_TITLE` and `_normalized_title` in `ces_revisions.annual.sample`, which plan 6 keyed from BLS's sample coverage tables in `data/annual/raw/`. So the titles are those tables' titles: "Education and health services", not "Private education and health services". Effect if wrong: a reader sees a title that differs from the one BLS uses elsewhere.
- **F5: table layout.** One generated block per transition (`stage-3-f-s`, 37 rows; `stage-3-s-t`, 37; `stage-3-t-b`, 36) and one bounds block (`stage-3-bounds`). Each transition row shows every identity term, so a reader can check a share and see its base. Variances and the covariance print to one decimal in thousands squared, shares to two decimals, and the gap as `.1e`.
- **F6: measured cost.** The same as plan 15's F10: `uv sync`, then `rm -rf data/panel`, then `/usr/bin/time -p` on `scripts/vintage_sources.py build` and on `scripts/seasonal_share.py report`. The planning session's 35.9 s build included creating the venv, so it is not the cost.

---

### Task 1: The three shares

**Files:**
- Create: `src/ces_revisions/seasonal_share.py`
- Test: `tests/test_seasonal_share.py`

**Interfaces:**
- Consumes: `REGIMES` and `TRANSITIONS` in `ces_revisions.vintages.accounting`. The decomposition frame, in tests from `accounting.decomposition(vintage_data.identity_terms())`.
- Produces: `REGIME_LABELS: dict[str, str]`, derived from `REGIMES`: `{"pre_2003": "1979-01 to 2003-04", "2003_2019": "2003-05 to 2019-12", "2020_2022": "2020-01 to 2022-12", "2023_present": "2023-01 on"}`. `seasonal_shares(table: pl.DataFrame) -> pl.DataFrame`: every input column plus `share_ratio`, `share_adjustment` and `share_nsa` (Float64), sorted by transition in `TRANSITIONS` order, then sector, then regime in `REGIMES` order. It raises `ValueError` naming `<source> <sector> <transition> <regime>` for a row whose `var_revision_sa` is null or not above 0.

- [ ] **Step 1: Write the failing tests** in `tests/test_seasonal_share.py`, with a helper `decomposition_row(var_sa, var_adj, cov, gap=0.0, **keys)` that sets `var_revision_nsa = var_sa − var_adj + 2·cov + gap` and the given gap:
  - `test_shares_of_a_hand_built_row`: var_sa 100, var_adj 50, cov 20 give `share_ratio == 0.5`, `share_adjustment == 0.3` and `share_nsa == 0.7`, each with `pytest.approx`
  - `test_shares_are_not_clipped_to_the_unit_interval`: var_sa 10, var_adj 60, cov 40 give 6.0, 2.0 and −1.0
  - `test_a_group_without_sa_variance_is_refused`, parametrized over `var_revision_sa` 0.0 and None: `pytest.raises(ValueError, match="cesvinall 10 F_S 2023_present")`
  - `test_regime_labels_follow_the_accounting_regimes`: `list(REGIME_LABELS) == [name for name, _, _ in REGIMES]` and the four labels as in Interfaces
  - `test_the_two_splits_partition_the_built_variance`: on the built table, `(share_adjustment + share_nsa − (1 − variance_identity_gap / var_revision_sa)).abs().max() <= 1e-12` and `(share_adjustment + share_nsa − 1).abs().max() <= 1e-12`
  - `test_built_shares_run_in_time_order`: height 110; the `transition` column's run order is `F_S`, `S_T`, `T_B`; sector `00` F→S regimes are `["pre_2003", "2003_2019", "2020_2022", "2023_present"]`; no `T_B` row has regime `pre_2003`
  - `test_built_shares_keep_values_outside_the_unit_interval`: 12 rows with `share_ratio > 1` and `round(max, 2) == 9.93`; 27 rows with `share_adjustment < 0`, all `F_S`; 2 rows with `share_adjustment > 1`
- [ ] **Step 2: Run the tests to verify they fail.** Run: `uv run pytest tests/test_seasonal_share.py -q`. Expected: a collection error, `ModuleNotFoundError: No module named 'ces_revisions.seasonal_share'`.
- [ ] **Step 3: Implement `REGIME_LABELS` and `seasonal_shares`** in `src/ces_revisions/seasonal_share.py`. The module docstring states the three shares and that none is bounded. Sort with `replace_strict` ranks from `TRANSITIONS` and `REGIMES`.
- [ ] **Step 4: Run the tests to verify they pass.** Run: the Step 2 command. Expected: 8 passed.
- [ ] **Step 5: Commit.** `git add src/ces_revisions/seasonal_share.py tests/test_seasonal_share.py`, then `git commit -m "Add the seasonal-adjustment shares of SA revision variance"`.
- [ ] **Step 6: Prove the tests.** Run executing-plans' mutant gate on this task's diff, which needs the clean tree of Step 5. Expected: every mutant killed, or each survivor ruled in the ledger.

### Task 2: The report and the findings file

**Files:**
- Create: `scripts/seasonal_share.py`
- Create: `specs/ces-revision-drivers-stage-3-findings.md`
- Modify: `CLAUDE.md` (one Commands line after the `first_closing.py report` line: `uv run python scripts/seasonal_share.py report  # rewrite the O9 findings tables from data/panel/ (offline; run the vintage build first)`)
- Test: `tests/test_seasonal_share.py`

**Interfaces:**
- Consumes: `seasonal_shares` and `REGIME_LABELS` (Task 1). `replace_generated` in `archive_inventory`. `PANEL_DIR` and `ROOT` in `ces_revisions.vintages.raw`.
- Produces, in `scripts/seasonal_share.py`:
  - `FINDINGS = ROOT / "specs" / "ces-revision-drivers-stage-3-findings.md"` and `INPUT = "accounting_decomposition.parquet"`.
  - `SECTOR_TITLES: dict[str, str]`: `00` Total nonfarm, `10` Mining and logging, `20` Construction, `30` Manufacturing, `40` Trade, transportation, and utilities, `50` Information, `55` Financial activities, `60` Professional and business services, `65` Education and health services, `70` Leisure and hospitality, `80` Other services, `90` Government.
  - `render_transition(shares: pl.DataFrame, transition: str) -> str`: a Markdown table with the columns Sector, Source, Regime, Months, Var SA, Var NSA, Var adjustment change, Cov(NSA, adjustment change), Ratio share, Adjustment share, NSA share and Identity gap, one row per row of that transition.
  - `render_bounds(shares: pl.DataFrame) -> str`: one row per share (Ratio, Adjustment, NSA) and transition (F→S, S→T, T→B, All), with Rows, Below 0, Above 1, Min and Max.
  - `report() -> int` and `main(argv: list[str] | None = None) -> int` with the one command `report`. It writes the blocks `stage-3-f-s`, `stage-3-s-t`, `stage-3-t-b` and `stage-3-bounds`, each body framed by blank lines as in `scripts/collection_window.py`, and prints `report: 110 rows, <s> s`. With no input it prints to stderr a line that names `uv run python scripts/vintage_sources.py build` and returns 1.

- [ ] **Step 1: Write the failing tests** in `tests/test_seasonal_share.py`, importing the script as `import seasonal_share as seasonal_share_report` and `from ces_revisions.annual import sample`:
  - `test_sector_titles_follow_the_coverage_tables`: `list(SECTOR_TITLES) == list(sample.SECTORS)` and, for each pair, `sample._LEGACY_SECTOR_BY_TITLE[sample._normalized_title(title)] == code`
  - `test_report_lays_out_one_row_per_sector_and_regime`: on the built shares, `render_transition(shares, "F_S")` has 39 lines and its third line starts with `| 00 Total nonfarm | cesnaicsrev | 1979-01 to 2003-04 | 291 |`; `render_transition(shares, "T_B")` has 38 lines and no `cesnaicsrev`
  - `test_report_shows_the_gap_and_unclipped_shares`: on the hand-built row var_sa 10, var_adj 60, cov 40, gap 1.5e-11, the rendered row holds `| 6.00 | 2.00 | -1.00 | 1.5e-11 |`
  - `test_bounds_count_shares_outside_the_unit_interval`: on the built shares, `render_bounds` holds the rows `| Ratio | All | 110 | 0 | 12 |`, `| Adjustment | F→S | 37 | 27 |` and `| NSA | All | 110 | 2 | 27 |` as line prefixes
  - `test_report_refuses_without_a_built_panel(tmp_path, monkeypatch, capsys)`: with `PANEL_DIR` set to `tmp_path`, `main(["report"]) == 1` and stderr contains `scripts/vintage_sources.py build`
  - `test_report_fills_the_four_generated_blocks(tmp_path, monkeypatch, capsys)`: write the built decomposition to `tmp_path / INPUT` and a findings file with the four empty marker pairs; `main(["report"]) == 0`; each block, read back with `archive_inventory.generated_block`, equals `f"\n{render(...)}\n\n"` for its renderer; stdout contains `report: 110 rows`
- [ ] **Step 2: Run the tests to verify they fail.** Run: `uv run pytest tests/test_seasonal_share.py -q`. Expected: a collection error, `ModuleNotFoundError: No module named 'seasonal_share'`.
- [ ] **Step 3: Implement `scripts/seasonal_share.py`** as its Interfaces state, after the shape of `scripts/collection_window.py`. Its docstring names the `report` command, its input and the build that comes first.
- [ ] **Step 4: Run the tests to verify they pass.** Run: the Step 2 command. Expected: all pass.
- [ ] **Step 5: Write the findings skeleton** with the sections Question, Answer, Data and shares, Results (the three transition blocks under `### F→S`, `### S→T` and `### T→B`, then the bounds block under `### Shares outside [0, 1]`, each between `<!-- BEGIN GENERATED <name> -->` and `<!-- END GENERATED <name> -->`), Limits and Measured cost. The opening paragraph names O9, the roadmap, plan 16 and the command, and says not to edit the tables by hand, as the Stage 1 findings do.
- [ ] **Step 6: Make the real run (F6).** Run: `uv sync`, `rm -rf data/panel`, `/usr/bin/time -p uv run python scripts/vintage_sources.py build`, then `/usr/bin/time -p uv run python scripts/seasonal_share.py report`. Expected: the build lists `accounting_decomposition: 110 rows`. The report prints `report: 110 rows`. The bounds block shows the F2 counts: ratio above 1 in 12 rows (F→S 4, S→T 2, T→B 6) with maximum 9.93, and adjustment share below 0 in 27 rows, all F→S, and above 1 in 2. No gap prints above `1.6e-11`.
- [ ] **Step 7: Write the findings prose** from the generated numbers. The Answer gives the result in two or three sentences, says that the shares are descriptive with no interval, and says that none is bounded in [0, 1]. Data and shares defines the identity, the three shares and why the two splits add to 1. Limits names:
  - the benchmark release: December F→S, November S→T and every T→B take their later value from the January benchmark release, which also revises the seasonal factors (compare the Stage 1 findings' Benchmark effects line)
  - what the negative F→S splits mean: there the adjustment change moves with the NSA revision, so it offsets part of the NSA revision rather than adding variance
  - the short regimes: 36 months for 2020 to 2022 and 34 for 2023 on
  - rounding: the inputs are whole thousands, which matters most where `Var SA` is small (the smallest is about 1.0)
  - before 2003, only total nonfarm, from the revision table, and no T→B

  Measured cost gives the two wall times from Step 6 and "1 session" (or the count used).
- [ ] **Step 8: Add the CLAUDE.md Commands line** given under Files.
- [ ] **Step 9: Verify.** Run: `uv run pytest -m "not slow and not network" -q`, `uv run ruff check` and `uv run ruff format --check`. Expected: all pass. Rerun the report and check that `shasum` of the findings file is unchanged, which shows the blocks are stable.
- [ ] **Step 10: Commit.** `git add scripts/seasonal_share.py specs/ces-revision-drivers-stage-3-findings.md CLAUDE.md tests/test_seasonal_share.py`, then `git commit -m "Report the seasonal-adjustment share of SA revision variance"`.
- [ ] **Step 11: Prove the tests.** Run executing-plans' mutant gate on this task's diff, which needs the clean tree of Step 10. Expected: every mutant killed, or each survivor ruled in the ledger.
