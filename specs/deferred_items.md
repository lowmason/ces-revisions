# Deferred items

## 1-ces-revisions — 2026-09-12
- [x] Review Minor: gradient checks along `A_t` and `Z_t` (whole-branch
      review of plan 1, triaged defer). `tests/test_kalman.py` checks the
      log-likelihood gradient only along a joint scaling of `transition_cov`
      and `observation_cov`, which covers Stage 1's scale-only pilot. A stage
      that samples parameters inside `transition_matrix` or
      `observation_matrix` (factor loadings, for example) needs
      finite-difference checks along those arrays against
      `tests/dense_reference.py`. See `specs/plans/completed/1-ces-revisions.md`.
      Size: quick-fix. Done when: the first stage that samples a parameter
      entering `transition_matrix` or `observation_matrix` adds a dense-oracle
      gradient check along it.
      → acknowledged 2026-11-15 (claude/roadmap-reset-faster-answers-881246): roadmap reset 2026-11-15. Waits for O6 in specs/ces-revision-drivers-roadmap.md, the first model that samples parameters inside those matrices
      → done 2026-11-15 (/deferred quick fix)
- [x] Review recommendation: dispersed chain starts for Req 18 (whole-branch
      review of plan 1, triaged defer). NumPyro's default
      `init_to_uniform(radius=2)` starts each LogNormal scale in roughly
      [0.14, 7.4], below five of the six true scales in
      `tests/synthetic_pilot.py`. The pilot still passes, but Req 18 asks for
      four dispersed chains in reported fits, and the run harness that owns
      initialization arrives in roadmap Stage 6. See
      `specs/plans/completed/1-ces-revisions.md`. Size: quick-fix. Done when:
      the Stage 6 run harness starts the four chains from dispersed points
      and records the initialization strategy with each run.
      → acknowledged 2026-11-15 (claude/roadmap-reset-faster-answers-881246): roadmap reset 2026-11-15. Done in plan 11, which Stage 2 of specs/ces-revision-drivers-roadmap.md finishes
      → planned 2026-11-15: specs/plans/13-ces-revision-drivers-stage-2.md
      → done 2026-11-15 (plan 13)

## 2-ces-revisions — 2026-09-13
- [x] Review Important: test the monthly live window against member dates
      (whole-branch review of plan 2, triaged defer). `scripts/archive_inventory.py`
      credits a prior-adjustment or outlier copy to the release whose 8:30 a.m.
      Eastern embargo precedes it, but BLS's seasonal adjustment files page says
      only that those files change with every monthly release, not when they are
      posted. The committed evidence fits the rule: no fingerprint repeats across
      11 neighboring pairs of windows, and the 2021-11-05 copy's files are dated
      2021-11-03. Only a direct check would catch a copy taken before BLS posted a
      release's files: record each copy's newest ZIP member date in
      `docs/inventory/archive-captures.csv` and test that it falls after the
      previous release and before the copy's own time, which needs a `captures`
      run. See `specs/plans/completed/2-ces-revisions.md`. Size: quick-fix.
      Done when: `captures` records member dates and a test checks every copy
      against them.
      → acknowledged 2026-11-15 (claude/roadmap-reset-faster-answers-881246): roadmap reset 2026-11-15. Waits for a captures run
      → done 2026-11-15 (/deferred quick fix)
- [ ] Review Minor: show the copies behind a `conflicting_captures` cell
      (whole-branch review of plan 2, triaged defer). `_status_cell` in
      `scripts/archive_inventory.py` links evidence only for present statuses, so
      a `conflicting_captures` cell in the review's per-vintage table would name
      none of the copies that disagree; no such cell exists today. Size:
      quick-fix. Revisit if: `docs/inventory/archive-inventory.csv` shows a
      `conflicting_captures` status.
- [ ] BLS request: first-closing dates, matched-sample counts, and lapse
      operations (user decisions of 2026-09-14). Req 3 uses two labeled proxies
      because BLS publishes neither input: the release date ends the collection
      window `D_t`, and Table 1's active-report coverage share stands in for
      usable linked coverage. A request drafted on 2026-09-14 for the user to
      send asks BLS for historical first-closing dates and matched-sample counts
      by supersector, and optionally for the six *not found* lapse values in the
      `shutdown_*` rows of `docs/ces-revisions-review.md`. Size: quick-fix.
      Revisit if: BLS answers; then Stage 12 swaps in first-closing dates,
      Stage 4 swaps in matched-sample counts, and Stage 13 codes any lapse
      values the answer documents.

## 5-ces-revisions — 2026-09-14
- [ ] BLS's next vintage-file refresh (plan 5, Deviation 1). The committed
      `data/raw/bls/cesvinall.zip` of 2026-03-06 ends with the January 2026
      benchmark release, so Stage 3 labels later vintage-file stages
      `beyond_frontier`, and their estimates reconcile only inside the revision
      table. BLS refreshes the files about once a year, weeks after the
      benchmark release. Absorbing a refresh: land the fetch-hardening item
      below, run `scripts/vintage_sources.py fetch` and `build`, check that
      every newly held table estimate reproduces, teach
      `comment_release_months` in `src/ces_revisions/vintages/stages.py` any
      new Comments wording, and re-pin the frontier, the Comments entries, and
      the M rows the frontier leaves unchecked in `tests/test_stage_labels.py`
      and the 1627 and 1617 reproduced counts in `tests/test_differencing.py`.
      See `specs/plans/completed/5-ces-revisions.md`. Size: quick-fix. Revisit
      if: `test_bls_has_not_refreshed_the_vintage_files_since_the_manifest`
      fails under `uv run pytest -m network`.
- [x] Review Minor: two EMPLOY vintages map to one release (whole-branch
      review of plan 5, triaged defer). `rtdsm_levels` in
      `src/ces_revisions/vintages/panel.py` joins each EMPLOY vintage to the
      release made in its calendar month or the latest before it. October 2025
      had no release, so EMPLOY25M10 resolves to release 2025-08 as EMPLOY25M9
      does, and the panel's `rtdsm_employ` rows hold that release twice under
      two `vintage_id`s, whose values agree because the October vintage
      repeats September's. No Stage 3 test differences or pins these levels.
      Size: quick-fix. Done when: the first stage that reads the
      `rtdsm_employ` rows (roadmap Stage 6 consumes the 1979 leg) flags or
      drops the repeat vintage.
      → acknowledged 2026-11-15 (claude/roadmap-reset-faster-answers-881246): roadmap reset 2026-11-15. Done in plan 11, which Stage 2 of specs/ces-revision-drivers-roadmap.md finishes
      → planned 2026-11-15: specs/plans/13-ces-revision-drivers-stage-2.md
      → done 2026-11-15 (plan 13)
- [x] Review recommendations: evidence tests for two planning findings
      (whole-branch review of plan 5, triaged defer). Two findings in plan 5's
      Planning evidence rest on one-off queries: EMPLOY's same-vintage
      seasonally adjusted changes differ from the revision table in 10 cells
      between 1981 and 1996, the reason the 1979 leg's changes come from the
      table; and between B and M the not seasonally adjusted value moves in
      1,775 of 1,836 April-to-October unit-months but in 46 of 756
      January-to-March and 35 of 504 November-to-December, the evidence for
      Deviation 5's stage rule. Tests reproducing both from
      `tests/vintage_data.py`'s frames, in `tests/test_vintage_panel.py` and
      `tests/test_stage_labels.py`, would keep those choices auditable. Size:
      quick-fix. Done when: tests pin the 10 cells and the three counts.
      → acknowledged 2026-11-15 (claude/roadmap-reset-faster-answers-881246): roadmap reset 2026-11-15. Half done in plan 11 (the ten EMPLOY cells), which Stage 2 of specs/ces-revision-drivers-roadmap.md finishes
      → done 2026-11-15 (/deferred quick fix)

## ces-revisions-roadmap — 2026-11-15

The owner parked this roadmap and replaced it with the outcome-first
`specs/ces-revision-drivers-roadmap.md`, for the reasons in
`specs/findings/ces-revision-drivers-reset-review.md`. "Spec" below means the
parked `specs/completed/ces-revisions.md`, and O-numbers are outcomes of the new
roadmap. Stage 6 is not here, because Stage 2 of the new roadmap finishes its
plan 11.
- [ ] Parked Stage 7, Gaussian closing-stage core across supersectors
      (`specs/completed/ces-revisions-roadmap.md`, parked 2026-11-15 by the
      outcome-first reset). Widens plan 11's aggregate pilot to the eleven
      supersectors with the common factor, cross-sector news factors, shared
      noise and reconciliation. Covered by O6 in the Track B Ranked list.
      Size: plan. Done when: the stage of
      `specs/ces-revision-drivers-roadmap.md` that delivers O6 ticks.
- [ ] Parked Stage 8, benchmark anchors, wedge, post-March recursion and
      mature horizon in the model (`specs/completed/ces-revisions-roadmap.md`,
      parked 2026-11-15 by the outcome-first reset). Enters the preliminary
      and final benchmarks, the B vintage through the wedge operator in
      `src/ces_revisions/operators/`, the post-March recursion with its
      reconstruction error, and right-censored M. Covered by O6. Size: design.
      Done when: the stage of `specs/ces-revision-drivers-roadmap.md` that
      delivers O6 ticks.
- [ ] Parked Stage 9, joint NSA/SA measurement and the model-based seasonal
      decomposition (`specs/completed/ces-revisions-roadmap.md`, parked
      2026-11-15 by the outcome-first reset). Adds the SA vintages through one
      mature seasonal component (spec Req 9, channel (a)). Covered by O6, and
      O9 is its descriptive counterpart. Size: plan. Done when: the stage of
      `specs/ces-revision-drivers-roadmap.md` that delivers O6 ticks.
- [ ] Parked Stage 10, Student-t news, noise and process innovations
      (`specs/completed/ces-revisions-roadmap.md`, parked 2026-11-15 by the
      outcome-first reset). Comparator (1) of spec Req 8. Covered by O7. Size:
      plan. Done when: the stage of `specs/ces-revision-drivers-roadmap.md`
      that delivers O7 ticks.
- [ ] Parked Stage 11, stochastic volatility per transition and the sampler
      benchmark (`specs/completed/ces-revisions-roadmap.md`, parked 2026-11-15
      by the outcome-first reset). Comparator (2) of spec Req 8, with the Req
      17 step (5) benchmark record. Covered by O7. Size: plan. Done when: the
      stage of `specs/ces-revision-drivers-roadmap.md` that delivers O7 ticks.
- [ ] Parked Stage 12, collection-window panel, first-stage C1 model and macro
      controls (`specs/completed/ces-revisions-roadmap.md`, parked 2026-11-15
      by the outcome-first reset). Spec Req 3, collection and macro rows.
      Stage 1 of the new roadmap builds the business-day count. The collection
      rates C1–C3 and RR are O3, and the calendar-predicted C1 split and the
      macro controls enter the model under O7. Size: plan. Done when: the
      stages of `specs/ces-revision-drivers-roadmap.md` that deliver O3 and O7
      have all ticked.
- [ ] Parked Stage 13, institutional capacity panel and shutdown events,
      FY2009 on (`specs/completed/ces-revisions-roadmap.md`, parked 2026-11-15
      by the outcome-first reset). Spec Req 3 institutional rows and Req 13
      inputs. The funding and staffing panel is O11. The four-field
      shutdown-events table enters the model under O7, and the descriptive
      party-composition table belongs to the report, O16. Size: plan. Done
      when: the stages of `specs/ces-revision-drivers-roadmap.md` that deliver
      O11, O7 and O16 have all ticked.
- [ ] Parked Stage 14, seasonal-adjustment file store, seasonal-flag panel,
      birth–death covariates and interventions table
      (`specs/completed/ces-revisions-roadmap.md`, parked 2026-11-15 by the
      outcome-first reset). Spec Req 3 seasonal rows, Req 11 and Req 14. The
      file store is part of O10, the birth–death columns are part of O13, and
      the seasonal flags and the interventions enter the model under O7. Size:
      plan. Done when: the stages of `specs/ces-revision-drivers-roadmap.md`
      that deliver O10, O13 and O7 have all ticked.
- [ ] Parked Stage 15, closing-stage scale layer
      (`specs/completed/ces-revisions-roadmap.md`, parked 2026-11-15 by the
      outcome-first reset). The F→S, S→T and seasonal rows of the spec Req 8
      map, with the Req 12 hierarchy and the Req 13 and Req 14 terms. Covered
      by O7. Size: plan. Done when: the stage of
      `specs/ces-revision-drivers-roadmap.md` that delivers O7 ticks.
- [ ] Parked Stage 16, annual-stage scale, capacity factor and the full
      prior-predictive gate (`specs/completed/ces-revisions-roadmap.md`,
      parked 2026-11-15 by the outcome-first reset). Completes comparator (3)
      of spec Req 8 and passes the complete Req 15 prior-predictive plan.
      Covered by O7, with the gate under O17. Size: plan. Done when: the
      stages of `specs/ces-revision-drivers-roadmap.md` that deliver O7 and
      O17 have all ticked.
- [ ] Parked Stage 17, first fit of comparator (3), the kernel decision and
      the hierarchy geometry (`specs/completed/ces-revisions-roadmap.md`,
      parked 2026-11-15 by the outcome-first reset). Spec Req 17 step (5) and
      Req 12. Covered by O7. Size: plan. Done when: the stage of
      `specs/ces-revision-drivers-roadmap.md` that delivers O7 ticks.
- [ ] Parked Stage 18, posterior predictive checks and full-panel SBC
      (`specs/completed/ces-revisions-roadmap.md`, parked 2026-11-15 by the
      outcome-first reset). Spec Req 18. Covered by O17 in the Track B Ranked
      list. Size: plan. Done when: the stage of
      `specs/ces-revision-drivers-roadmap.md` that delivers O17 ticks.
- [ ] Parked Stage 19, nested comparison of comparators (0)–(3)
      (`specs/completed/ces-revisions-roadmap.md`, parked 2026-11-15 by the
      outcome-first reset). Blocked PSIS-LOO, leave-future-out and
      leave-one-benchmark-year-out under spec Req 18. Covered by O8. Size:
      plan. Done when: the stage of `specs/ces-revision-drivers-roadmap.md`
      that delivers O8 ticks.
- [ ] Parked Stage 20, pseudo-real-time evaluation, leakage audit and the
      predictive-scale artifact (`specs/completed/ces-revisions-roadmap.md`,
      parked 2026-11-15 by the outcome-first reset). Spec Req 18 and Req 19,
      about 280 refits at past release dates. Covered by O8. Size: design.
      Done when: the stage of `specs/ces-revision-drivers-roadmap.md` that
      delivers O8 ticks.
- [ ] Parked Stage 21, sensitivity suite and identification checks
      (`specs/completed/ces-revisions-roadmap.md`, parked 2026-11-15 by the
      outcome-first reset). Spec Req 16 and the variants that Req 6, 7, 13, 14
      and 18 list. No outcome of the new roadmap covers it. Size: plan. Done
      when: the sensitivity registry, the capacity overlap table and the
      (a)/(b)/(c) decomposition tables that the parked stage names exist for
      the joint model of O7.
- [ ] Parked Stage 22, non-marginal sensitivities: exact interval censoring
      and unknown change points (`specs/completed/ces-revisions-roadmap.md`,
      parked 2026-11-15 by the outcome-first reset). Spec Req 15 and Req 14.
      No outcome of the new roadmap covers it. Size: design. Done when: the
      censored reduced-panel model and the change-point model are registered
      as sensitivity-only fits beside the joint model of O7.
- [ ] Parked Stage 23, verified literature review and bibliography
      (`specs/completed/ces-revisions-roadmap.md`, parked 2026-11-15 by the
      outcome-first reset). Spec Req 20. Committed as Stage 7 of the new
      roadmap (O15). Size: plan. Done when: Stage 7 of
      `specs/ces-revision-drivers-roadmap.md` ticks.
- [ ] Parked Stage 24, report (`specs/completed/ces-revisions-roadmap.md`,
      parked 2026-11-15 by the outcome-first reset). Spec Req 19, Req 16 and
      Req 13. Committed as Stage 8 of the new roadmap (O16). Size: plan. Done
      when: Stage 8 of `specs/ces-revision-drivers-roadmap.md` ticks.
- [ ] Parked Stage 25, X-13 vintage reproduction
      (`specs/completed/ces-revisions-roadmap.md`, parked 2026-11-15 by the
      outcome-first reset). Reproduces the X-13 factors for the 29
      archive-complete releases (spec Req 9, channel (b)). Covered by O10 in
      the Track C Ranked list. Size: design. Done when: the stage of
      `specs/ces-revision-drivers-roadmap.md` that delivers O10 ticks.
- [ ] Parked Stage 26, channel (b) observation and the re-reported
      decomposition (`specs/completed/ces-revisions-roadmap.md`, parked
      2026-11-15 by the outcome-first reset). Spec Req 9, channel (b), in the
      model. Covered by O10 and O7. Size: plan. Done when: the stages of
      `specs/ces-revision-drivers-roadmap.md` that deliver O10 and O7 have all
      ticked.
- [ ] Parked Stage 27, FY1979 institutional-panel extension and the full-leg
      capacity refits (`specs/completed/ces-revisions-roadmap.md`, parked
      2026-11-15 by the outcome-first reset). Spec Req 3 and Req 13. Covered
      by O12 in the Track D Ranked list, with the refits under O7. Size:
      design. Done when: the stages of `specs/ces-revision-drivers-roadmap.md`
      that deliver O12 and O7 have all ticked.

## claude/deferred-063daa Step 4b review — 2026-11-15
- [x] Keep each `bls_current` copy as evidence across `captures` runs
      (code-reviewer's Declined to judge line in the Step 4b review of branch
      `claude/deferred-063daa`, carried by the owner). Each run of
      `scripts/archive_inventory.py captures` replaces the earlier `bls_current`
      rows in `docs/inventory/archive-captures.csv` and keeps no payload. The
      run of 2026-11-15 dropped the 2026-09-13 copy of `ces.spec.other.zip`, so
      2026-08 went from complete to `not_archived`. 2026-10 is one of the 29
      archive-complete releases that O10 of
      `specs/ces-revision-drivers-roadmap.md` would reproduce, and it rests only
      on BLS's current file. It drops at the next run unless the Internet
      Archive copies it first. The behavior predates the branch (plan 2). The
      fix keeps the earlier rows as dated evidence. It also stores each fetched
      payload, because O10 needs the files and not only their fingerprints.
      Where the payloads live is the open decision, under CLAUDE.md's rules for
      `data/raw/`. Size: design. Done when: a later `captures` run keeps every
      earlier `bls_current` row and its payload, and a test shows that a
      release credited by such a copy keeps its status after that run.
      → planned 2026-11-16: specs/plans/14-seasonal-adjustment-file-store.md
      → done 2026-11-17 (plan 14)

## 12-ces-revision-drivers-stage-1 — 2026-11-15
- [ ] Declined review line: `absolute_revisions` in
      `src/ces_revisions/collection_window.py` pairs the first and later
      estimates on `reference_month` and `seasonal_status` only. That holds
      while its source, BLS's revision table (`cesnaicsrev`), carries total
      nonfarm alone (final review of plan 12; the owner ruled defer for O2).
      O2 measures the 11 supersectors from vintage-file changes
      (`data/panel/stage_changes.parquet`), whose rows add `sector`, so the
      join keys and the source filter must take `sector` and that source, with
      a test that two sectors' estimates for one month never pair. Size:
      quick-fix. Revisit if: a stage of
      `specs/ces-revision-drivers-roadmap.md` lists O2 in its Outcomes line.

## 11-ces-revisions — 2026-11-15
- [ ] The pilot's `sigma_slope` posterior conflicts with its prior (plan 11,
      planning choice 9). The pilot fit's run directory
      (`data/runs/stage6-pilot-seed2995-20261022T020454Z`, gitignored; its
      `inference_data.nc` has SHA-256 `b0480b97…`) gives a posterior median of
      65.9 and an SD of 6.4 for `sigma_slope`, against its HalfNormal(15)
      prior, whose median is 10.1 and SD 9.0. The median is 4.4 times the
      prior's scale. The fit still meets Req 18 and the April 2020 check, and
      the news shares do not read `sigma_slope`. The committed
      `docs/runs/stage-6/pilot-fit/diagnostics.json` holds only its R-hat and
      ESS. Size: design. Revisit if: O6's fit or O17's prior-predictive gate
      still leaves it above 3 prior scales.
- [ ] The doubled-prior rerun misses Req 15's monthly constraint (plan 11,
      planning choice 9). `docs/runs/stage-6/prior-predictive-doubled.json`
      shows 13.3% of months with an F→T revision above 250 thousand, where the
      allowed range is 0.1% to 5%. It also shows a median |F→T| revision in
      20 to 100 thousand in 66.9% of prior draws, where the minimum is 80%. The
      base prior passes (2.9% and 94.8%). Size: design. Revisit if: O17's
      prior-sensitivity run starts, meaning a stage of
      `specs/ces-revision-drivers-roadmap.md` lists O17 in its Outcomes line.
- [ ] Residual seasonality at lags 11 and 12 in every layout (plan 11,
      planning choice 9). `docs/runs/stage-6/seasonal-comparison.json` gives
      the Ljung–Box Q (F, S, T) of 84, 49 and 61 for `dummy`, 43, 28 and 41 for
      `trig`, and 2303, 1526 and 1519 for `trig4`. Every test rejects at 1%.
      The decision record `docs/decisions/seasonal-state.md` keeps the dummy
      state anyway. Size: design. Revisit if: O17's posterior predictive suite
      traces a misfit to the seasonal block.
- [ ] Plan 11's GPU work (Task 13, and Task 14 Step 1, descoped by roadmap
      Stage 2, `specs/ces-revision-drivers-roadmap.md:24`). It covers the
      \$40 Lambda measurement of one on-demand instance each of A10, A100 40 GB
      and GH200 (Task 13), the GPU rule applied by
      `scripts/pilot_run.py choose-gpu` that writes
      `docs/runs/stage-6/gpu-choice.json` (Task 14 Step 1), and decision D8,
      which keeps the parallel-in-time Kalman filter out until a model is
      infeasible. The GPU code ships unused (`pilot/gpu_choice.py`,
      `pilot/dimensions.py`, `scripts/gpu_probe_grid.py`). Only the owner rents
      Lambda GPUs. The GPU rule has four open points from plan 11's reviews:
      the `fit_usd` tie rule and the record-order tie-break are unspecified,
      a `--max-bytes` skip counts as infeasible like a real out-of-memory,
      `gpu_choice.py`'s chain-method strings are labels only, and the free-form
      `--label` makes matching a GPU type by label fragile. See
      `specs/plans/completed/11-ces-revisions.md`. Size: plan. Done when: the
      stage that delivers O6 ticks.
- [ ] Review Minor findings left open by plan 11's per-task reviews and by the
      whole-branch review that plan 13 ran (the owner ruled carry as one item,
      2026-11-15). None hurts a person today. In
      `src/ces_revisions/pilot/`: `panel.py` has no hermetic test of
      `window()`; `__init__.py`'s docstring forward-references later modules;
      `model.py` ties the shock-SD column order to `layout.shock_names` order
      implicitly and has no test of its own for `TARGET_ACCEPT_PROB` and the
      `NOISE_*` constants; `simulate.py`'s `revision_statistics` gives NaN and
      a RuntimeWarning when no row has both F and T; `sbc.py` has no named
      finite check for the truth log likelihood, step size and truth values,
      does not enforce the 1999–2021 window or the counts, hard-codes 3 in the
      key `share_abs_z_above_3`, and no test pins the recovery thresholds
      (0.25 and 5%) or the ECDF alpha of 0.05/23; `seasonal.py` gives a NaN
      SE when fewer than two blocks are reliable. In `src/ces_revisions/runs/`:
      `fit.py` measures `meets_req18` at the run's own maximum tree depth, has
      no float64 guard in `run_nuts`, and `diagnostics_record` also checks
      deterministic sites, which the pilot lacks; `inits.py` labels its
      strategy "dispersed_prior_quantiles" for a median-plus-or-minus-k-SD
      method. In tests: `tests/pilot_synthetic.py`'s `shell_panel` fails
      unclearly above about 39 steps and `crisis_panel`'s only anchor has
      offset 0.0; the `sigma_seasonal` prior test in
      `tests/test_pilot_layout.py` is circular; `tests/test_pilot_simulate.py`
      cites a spike report that is not in the repo; two test warnings leak
      (ArviZ's RuntimeWarning in the NaN-statistic test and a Pareto-k
      UserWarning in `test_loo_runs_on_both_blocks`). In `scripts/pilot_run.py`:
      it imports `devices`, and so JAX, before `_configure_xla`, and no test
      runs `run_compare_seasonal` end to end. One more point comes from plan
      13's review: `shares` checks only the panel hash, so a `trig` run fit to
      the same panel would give shares for `trig`. Size: quick-fix. Done when: a
      plan for O17 lists each point as fixed, dropped on the owner's word, or
      kept with a reason.

## 14-seasonal-adjustment-file-store — 2026-11-17
- [ ] Drop the contact address when a BLS request is redirected off bls.gov
      (plan 14, Task 1 ruling on a declined review line). `user_agent` in
      `scripts/archive_inventory.py` decides from the request URL, and the
      three fetchers (`archive_inventory.fetch`, `scripts/vintage_sources.py`,
      `scripts/annual_sources.py`) send that header once. urllib's
      `HTTPRedirectHandler` copies every non-content header, the
      `User-Agent` included, to the redirect target. So a bls.gov response
      that redirects to another host would send `BLS_CONTACT_EMAIL` there.
      No such BLS redirect is known. The fix is a redirect handler that
      recomputes the header for each hop with `user_agent`. Size: quick-fix.
      Done when: every fetcher recomputes the `User-Agent` on each redirect
      hop, and a test shows that a bls.gov URL redirected to a non-BLS host
      sends only the bare product name to that host.
- [ ] Lift the 2026-09-13 `payload_lost` row to `200` when `captures` stores
      its bytes (plan 14's final review, a Declined to judge line; the owner
      chose a watch item). The BLS other-inputs row of 2026-09-13 in
      `docs/inventory/archive-captures.csv` has status `payload_lost` and
      digest `JRW4ECRTZ3X22BDLK7YHUZ6U3M7F5WUG`. If the Internet Archive
      ever lists a copy with that digest, `captures` stores the ZIP, but
      merge point 3 (`specs/completed/seasonal-adjustment-file-store.md`,
      Merge rule) keeps the BLS row at `payload_lost`, and `adopt` then
      refuses the ZIP as already stored. No credit is lost, because the
      Archive row credits release 2026-08. But
      `docs/ces-revisions-review.md` and the `data/seasonal/raw/` bullet in
      `CLAUDE.md` say that `adopt` sets the row to `200`. The fix amends merge
      point 3 so that a stored digest lifts the row, in
      `_merge_pair` or `_adopt_rows` of `scripts/archive_inventory.py`.
      Size: quick-fix. Revisit if: a row with digest
      `JRW4ECRTZ3X22BDLK7YHUZ6U3M7F5WUG` and status `200` appears in
      `docs/inventory/archive-captures.csv`.
- [ ] Retry a truncated replay instead of stopping the run (plan 14's final
      review, a Declined to judge line; the owner chose a watch item). A
      truncated response raises `http.client.IncompleteRead`, which is not an
      `OSError`, so `retrying` in `scripts/archive_inventory.py` does not
      retry it. A full `captures` or `current` run then stops and writes
      nothing. The behavior predates plan 14, and no run has met it. The fix
      catches `http.client.IncompleteRead` beside `OSError` in `retrying`,
      with a test. Size: quick-fix. Revisit if: a `captures` or `current` run
      stops on `http.client.IncompleteRead`.
- [ ] Keep the review's summary counts in step with the monthly `current`
      run (plan 14's final re-review, an observation). The summary bullets
      at the top of `docs/ces-revisions-review.md` hand-write the release
      count ("Of 281 releases from May 2003, specification files survive
      for 117") and the archive-complete count ("Only the 29 releases that
      keep all three file types"). `inventory` regenerates the tables below
      them after each monthly `current` run, but not these bullets, and no
      test checks them. The staleness predates plan 14. Size: quick-fix.
      Done when: those counts are either written by `inventory` as generated
      text, or dated to the run they describe, and the `data/seasonal/raw/`
      bullet in `CLAUDE.md` says which.
