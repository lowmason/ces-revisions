# Development history

## /deferred — 2026-11-15

Closed deferred items:
- "Review Minor: gradient checks along `A_t` and `Z_t` (whole-branch" (deferred by 1-ces-revisions): dense-oracle finite-difference checks of the gradient along `transition_matrix` and `observation_matrix` in `tests/test_kalman.py`.
- "Review Important: test the monthly live window against member dates" (deferred by 2-ces-revisions): `captures` records each copy's newest ZIP member date. Prior-adjustment and outlier copies are credited to the first release on or after it, which moves two copies to 2015-07 and 2026-02. A test caps every copy's build lead at 7 days.
- "Review recommendations: evidence tests for two planning findings" (deferred by 5-ces-revisions): tests pin plan 5's EMPLOY-versus-table cells and its B-to-M counts.

## 12-ces-revision-drivers-stage-1 — 2026-11-15
Plan: specs/plans/completed/12-ces-revision-drivers-stage-1.md · Spec: none · Roadmap: specs/ces-revision-drivers-roadmap.md, Stage 1

Differences from the design:
- Ruling (Task 1): business_days takes the holidays of start.year through end.year, not end.year + 1 — a December 31 observance before end sits in a year no later than end's, so the extra year is never read and no test could tell the two apart — if wrong, a December window would count one day too many, which the 2010-12 and 2021-12 tests would catch
- Ruling (Task 3): the findings file adds an Answer section after Question, beyond Step 5's section list — Step 8 asks for the answer in two or three sentences, and a reader looks for it first — if wrong, the section moves under Results with no other change
- Ruling (final review): flag 2025-09 too, so LAPSE_DELAYED_MONTHS holds six months and the plan test test_only_the_five_lapse_delayed_releases_are_flagged expects six — roadmap :20 reads "Lapse days count as business days, and the affected releases are flagged", and the 2025 lapse moved the September 2025 release from October 3 to November 20 (data/raw/manual/es-reschedules.csv, docs/ces-revisions-review.md:100); D1 listed the five releases that stay in the table and the regression, and the analysis drops 2025-09, so the tables and the regression do not change — if wrong, a later stage that joins collection_days reads September 2025 as lapse-delayed, and reverting is one constant and one test tuple
- Ruling (final review): test_lapse_flag_reaches_the_revisions compares against LAPSE_DELAYED_MONTHS minus DROPPED_MONTHS — it still pins the same five months in the revisions, so it is no weaker — if wrong, nothing changes, as the five months are the same
- Ruling (final review): rename test_only_the_five_lapse_delayed_releases_are_flagged to test_only_the_lapse_delayed_releases_are_flagged — under the ruling above it checks six months, and a name that says five misleads — if wrong, only the name differs from the plan
- Deviation (Task 1): the holiday span ends at `end.year`. Source: plan, Task 1.
- Deviation (Task 1): four shape tests added by the mutant step. Source: plan, Task 1.
- Deviation (Task 2): the day-count file flags six lapse-delayed months, September 2025 included; the analysis keeps five. Source: plan, Task 2.
- Deviation (Task 2): one test added, and the final review replaced one mutant and added two. Source: plan, Task 2.
- Deviation (Task 3): the lapse test compares against the six months minus the dropped months. Source: plan, Task 3.
- Deviation (Task 3): the findings file has an Answer section. Source: plan, Task 3.
- Deviation (Task 3): the run was timed with `/usr/bin/time -p`. Source: plan, Task 3.
- Deviation (Task 3): a Limits line on lapse-delayed later prints, added at the completion gate. Source: plan, Task 3.
- Deviation (Task 3): nine tests added by the mutant step. Source: plan, Task 3.
- Deferred: 1 item. Source: specs/deferred_items.md, § 12-ces-revision-drivers-stage-1.
