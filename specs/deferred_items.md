# Deferred items

## 1-ces-revisions — 2026-09-12
- [ ] Review Minor: gradient checks along `A_t` and `Z_t` (whole-branch
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
- [ ] Review recommendation: dispersed chain starts for Req 18 (whole-branch
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

## 2-ces-revisions — 2026-09-13
- [ ] Review Important: test the monthly live window against member dates
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
- [x] Review Minor: releases before May 1999 in `es-vintages.csv` (whole-branch
      review of plan 2, triaged defer). `parse_release_index` in
      `scripts/archive_inventory.py` reads only links with a four-digit year, so
      the 49 releases from February 1995 to April 1999, whose links use MMDDYY,
      are missing from `docs/inventory/es-vintages.csv`, and each older date needs
      checking against the archived schedules, since the December 1999 item links
      a file dated 2000-01-19. Stage 2 needs only May 2003 on; Stage 3's
      release-date index owns the earlier years. Size: quick-fix. Done when:
      Stage 3's release-date index covers the 1995–1999 releases with checked
      dates, or records those years as out of scope. → done in plan 5
- [ ] Review Minor: show the copies behind a `conflicting_captures` cell
      (whole-branch review of plan 2, triaged defer). `_status_cell` in
      `scripts/archive_inventory.py` links evidence only for present statuses, so
      a `conflicting_captures` cell in the review's per-vintage table would name
      none of the copies that disagree; no such cell exists today. Size:
      quick-fix. Revisit if: `docs/inventory/archive-inventory.csv` shows a
      `conflicting_captures` status.
- [x] Spec-owner finding: usable linked coverage is not published (completion
      report of plan 2). Req 3 asks for usable linked employment coverage and
      RSE by supersector, harmonized across the organization, UI-account, and
      worksite definition switches. Table 1 of the CES technical notes counts
      active sample reports, not the matched sample the estimator uses; every
      copy from March 2002 to March 2025 has the same columns, so no switch was
      found; and no copy holds the March 2012 table. The RSE measure does
      change: relative standard errors of levels through 2009, separate
      standard error pages from the 2009 benchmark, and a 1-month change from
      2017 (the `linked_coverage` and `relative_standard_error` rows of
      `docs/ces-revisions-review.md`). Roadmap Stage 4 builds the sample panel,
      and its exit's "definition-break flag at each documented switch" would
      pass vacuously. Size: design. Done when: the spec names the coverage
      measure Stage 4 builds, and Stage 4's exit flags the RSE measure changes
      and the missing 2012 table. Spec half done 2026-09-14: Req 3 names the
      coverage-share proxy and the RSE break flags (user decision); Stage 4's
      exit waits for the roadmap resume. → done in plan 6
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
- [x] Review Important and Minors: harden `fetch` before the next refresh → done in plan 7
      (whole-branch review of plan 5; deferred by user decision, 2026-09-14).
      `fetch_sources` in `scripts/vintage_sources.py` overwrites each file in
      `data/raw/` as it downloads and accepts whatever the server returns, so
      a failure partway, or an error page served in a file's place, leaves the
      committed sources out of step with `data/raw/manifest.csv`; meanwhile,
      `git checkout -- data/raw` restores them. The work: download into a
      staging directory, check each payload's file signature (ZIP, PDF, XLS,
      XLSX, HTML), and move files into `data/raw/` only when all pass; add an
      offline command that rehashes the manifest, since editing
      `data/raw/manual/es-reschedules.csv`, the one hand-editable source, fails
      the manifest test until a full network fetch; and record the `pdftotext`
      version, since the layout of `bls/histreleasedates.txt` can vary by
      poppler version. Size: plan. Done when: `fetch` stages and checks its
      downloads, an offline command rehashes the manifest, and the manifest
      names the `pdftotext` version.
- [ ] Review Minor: two EMPLOY vintages map to one release (whole-branch
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
- [ ] Review recommendations: evidence tests for two planning findings
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
      outcome-first reset). Reproduces the X-13 factors for the 28
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
