# Deferred items

## 2-ces-revisions — 2026-09-13
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

## 5-ces-revisions — 2026-09-14
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
