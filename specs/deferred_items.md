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
- [ ] Review Minor: releases before May 1999 in `es-vintages.csv` (whole-branch
      review of plan 2, triaged defer). `parse_release_index` in
      `scripts/archive_inventory.py` reads only links with a four-digit year, so
      the 49 releases from February 1995 to April 1999, whose links use MMDDYY,
      are missing from `docs/inventory/es-vintages.csv`, and each older date needs
      checking against the archived schedules, since the December 1999 item links
      a file dated 2000-01-19. Stage 2 needs only May 2003 on; Stage 3's
      release-date index owns the earlier years. Size: quick-fix. Done when:
      Stage 3's release-date index covers the 1995–1999 releases with checked
      dates, or records those years as out of scope.
- [ ] Review Minor: show the copies behind a `conflicting_captures` cell
      (whole-branch review of plan 2, triaged defer). `_status_cell` in
      `scripts/archive_inventory.py` links evidence only for present statuses, so
      a `conflicting_captures` cell in the review's per-vintage table would name
      none of the copies that disagree; no such cell exists today. Size:
      quick-fix. Revisit if: `docs/inventory/archive-inventory.csv` shows a
      `conflicting_captures` status.
- [ ] Spec-owner finding: usable linked coverage is not published (completion
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
      exit waits for the roadmap resume.
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
