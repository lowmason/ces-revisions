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

## 11-ces-revisions — 2026-11-15
Plan: specs/plans/completed/11-ces-revisions.md · Spec: specs/completed/ces-revisions.md (retired) · Roadmap: specs/ces-revision-drivers-roadmap.md, Stage 2

Differences from the design:
- Deviation (Task 6): `uv.lock` moved from revision 3 to 5, the first relock under uv 0.13. Source: plan, Task 6.
- Deviation (Task 7): at the owner's word, the non-finite-recovery crash and three Minor findings were fixed, though the plan's code did neither: NaN on zero variance, `NonFiniteRecovery`, `ECDF_SIMULATIONS` of 100,000, a process ID in the summary's temporary name, and `chain_method(1)`. Source: plan, Task 7.
- Deviation (Task 8): the "infeasible" GPU gate stays manual, in Task 14 Step 1 and the Task 13 header. Both moved to O6. Source: plan, Task 8.
- Deviation (Task 13): only Step 1 ran, at `97736a3`, and the rented-GPU steps moved to O6. Source: plan, Task 13.
- Deviation (Task 14): re-scoped to the seasonal-state decision alone by roadmap Stage 2. Plan 13's Task 2 executed it, and its Step 1 and the GPU rows of the exit check moved to O6. Source: plan, Task 14.
- Decision D1 (spec amendment): every closing cell, 1979–2026, enters as a same-release over-the-month change, and the March anchors enter as levels on the latent path. Source: plan, `## Decisions`.
- Decision D2 (spec amendment): the crisis multiplier scales the non-news level shock and the S and T news shocks in March–June 2020, and not the slope, seasonal, post-T news, or noise terms. Source: plan, `## Decisions`.
- Decision D6 (spec amendment): Task 1's spec wording on the crisis multiplier, the NSA closing cells and the Rollout note line is approved. Source: plan, `## Decisions`.
- Deferred: 5 items. Source: specs/deferred_items.md, § 11-ces-revisions.

## 13-ces-revision-drivers-stage-2 — 2026-11-15
Plan: specs/plans/completed/13-ces-revision-drivers-stage-2.md · Spec: none · Roadmap: specs/ces-revision-drivers-roadmap.md, Stage 2

Differences from the design:
- Ruling (Task 2): added 3 tests beyond the plan's 4 (positive elpd gain and Req 18 failure, a non-dummy rule layout, more than one grid record) — mutants.md step 2 asks for each rule's input shapes, and mutant 1 (decision_lines names the reference layout) survives the plan's tests alone — the fast tier reads 686, not the plan's 683; a wrong expected count if the owner compares it.
- Ruling (Task 3): added 10 tests after Step 4's run (the plan lists 6, Step 4 expected 6 passed, which held) — mutants.md step 2 asks for each rule's input shapes (other interval, skewed summary, block header and note, a whole run directory, both commands with a stand-in panel, a stale findings block, the default path) — the Task 3 test file holds 16 tests, not 6, and the fast tier reads 702, not 686+6
- Deviation (Task 2): three tests added to the plan's four, and the tiers ran inside `task-done` after Step 9's commit, so the fast tier read 686, not 683. Source: plan, Task 2.
- Deviation (Task 3): ten tests added after Step 4's run, and one from the final review, so the file holds 17. Source: plan, Task 3.
- Deviation (Task 4): the final review corrected the Limits line on the crisis multiplier, and F8. Source: plan, Task 4.
- Deviation (Task 5): the Codex review was skipped at the owner's word (D4). Three commits fix the three Important findings, a fourth adds the findings-table test (D5), and `sbc` keeps exit 0 on a failed replicate. Source: plan, Task 5.
- Deviation (Task 5): the executor stopped before Step 8, the owner's merge or PR choice. Source: plan, Task 5.

Closed earlier deferred items:
- "Review recommendation: dispersed chain starts for Req 18 (whole-branch" (deferred by 1-ces-revisions).
- "Review Minor: two EMPLOY vintages map to one release (whole-branch" (deferred by 5-ces-revisions).

## 14-seasonal-adjustment-file-store — 2026-11-17
Plan: specs/plans/completed/14-seasonal-adjustment-file-store.md · Spec: specs/completed/seasonal-adjustment-file-store.md (retired)

Differences from the design:
- Ruling (Task 1): Task 1 concern 1 / review Important 1 (README.md:76-78) -> reversed the earlier Task 7 ruling; fixed now by controller hand fix (one file, prose only), counts as fix attempt 1.
- Ruling (Task 2): Task 2 finding 5 accepted — a subprocess `git check-attr` test is hermetic (no network), so the spec:202 Unreachable line does not hold — if wrong, one test depends on git being installed.
- Ruling, contradicts spec (Task 6): Task 6 finding 2 — spec:294 vs the 55 pinned URLs: the plan test governs — spec:198 says "Revisit rows (status `-`) and every repeat of a digest resolve to that one file." — the five Archive rows sharing UMPG resolve to the adopted ZIP; spec:294 is left as written (approved design text) — if wrong, the pinned list is 5 short and Task 8's update corrects it.
- Ruling (Task 7): Task 7 concern 1 (Stage 14 bullet says coverage contains the inventory "once a full captures run has stored the Internet Archive copies that the file store does not hold yet") accepted — the brief's plain claim is false until Task 8 — Task 8 drops the "once ..." clause after a clean run; if Task 8 is deferred, the clause stays true.
- Deviation (Task 1): `README.md:76-78` was fixed in Task 1 too (ce357c6). Source: plan, Task 1.
- Deviation (Task 6): the pin held 55 URLs, not every Archive `200` row (`spec:294`). Source: plan, Task 6.
- Deviation (Task 7): the Stage 14 bullet kept a "once a full `captures` run…" clause until Task 8. Source: plan, Task 7.
- Deviation (Task 8): the counts at `review:132` and `review:488` are dated to the run of 2026-11-17 (owner, completion gate). Source: plan, Task 8.
- Deviation (Task 9): the Codex seat was skipped; the final review's two fixes are 3edf644 and 232b91b. Source: plan, Task 9.
- Decision D3 (spec amendment): split the store's `.gitattributes` rule into `-text` for every file and `-diff` for the ZIPs. Source: plan, `## Decisions`.
- Deferred: 4 items. Source: specs/deferred_items.md, § 14-seasonal-adjustment-file-store.

Closed earlier deferred items:
- "Keep each `bls_current` copy as evidence across `captures` runs" (deferred by claude/deferred-063daa Step 4b review).

## /deferred — 2026-11-17

Closed deferred items:
- "Drop the contact address when a BLS request is redirected off bls.gov" (deferred by 14-seasonal-adjustment-file-store): `ContactRedirectHandler` and `open_url` in `scripts/archive_inventory.py` set the User-Agent of each hop, and all three fetchers open their URLs through them.
- "Keep the review's summary counts in step with the monthly `current`" (deferred by 14-seasonal-adjustment-file-store): `inventory` writes the review's counts inside figure markers (`render_figures`), so no count is hand-typed or dated.
