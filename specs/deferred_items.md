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
- [ ] Review Important and Minors: harden `fetch` before the next refresh
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
