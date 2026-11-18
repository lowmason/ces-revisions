# Stage 9 findings: the window to first closing, the first-closing collection rate and revision size, total nonfarm

Outcomes O3 and O4 of the [ces-revision-drivers roadmap](ces-revision-drivers-roadmap.md), Stage 9, implemented by [plan 15](plans/completed/15-ces-revision-drivers-stage-9.md). `uv run python scripts/first_closing.py report` writes the nine tables below from the built `data/panel/`. Do not edit them by hand.

## Question

Does a longer window to first closing raise the first-closing collection rate C1? How do the mean absolute F→S and F→T revisions of total nonfarm employment move with C1? And how does the Stage 1 answer change when the window ends at first closing, not at the release, 2003–2026?

## Answer

**O3 (a).** A longer window to first closing goes with a higher first-closing collection rate. Within a reference year and calendar month, and with the 5-week control, each added business day goes with a C1 2.20 points higher (SE 0.26). The 95% interval for the change from 8 to 15 days is 11.7 to 19.2 points, against a mean C1 of 69.9 in the main period.

**O3 (b).** The data show no clear association between C1 and revision size. For 10 more points of C1, the 95% intervals are -14.4 to 3.1 thousand (NSA F→S), -15.5 to 3.6 (NSA F→T), -9.2 to 5.1 (SA F→S) and -7.7 to 16.8 (SA F→T), against mean absolute revisions of 28 to 48 thousand. Three of the four point estimates are negative, but each interval includes zero. This is an association, not an effect of C1 (Limits).

**O4.** Ending the window at first closing changes the Stage 1 answer very little. For each status and transition, the 95% intervals for a 7-day change with `W_t` and with `D_t` differ by at most 6 thousand at either end. For NSA F→S they are -36.5 to 28.1 thousand with `W_t` and -37.1 to 24.6 with `D_t`. Both windows give uninformative answers on 2003–2026: 23 year clusters cannot separate a moderate effect from none. The gap between the two windows changes only in the 45 releases of the fit sample with a holiday in the final week, so close slopes are expected.

## Data and window

- **Revisions.** As in Stage 1, a revision is the absolute difference between the second (F→S) or third (F→T) estimate of total nonfarm's over-the-month change and the first, NSA and SA, in thousands, from BLS's revision table (`absolute_revisions` in `ces_revisions.collection_window`). In the main period the mean absolute revision is 43.3 thousand for NSA F→S, 47.6 for NSA F→T, 28.0 for SA F→S and 41.8 for SA F→T.
- **C1.** LABSTAT's series `CEU00000000C1` is the first-closing collection rate: the percent of the actively reporting units on the CES sample registry whose report BLS received for the first preliminary estimate. `data/raw/bls/ce-collection-rates.tsv` holds the C1, C2, C3 and `CEU05000000RR` lines of LABSTAT's `ce.data.0.AllCESSeries`, fetched on 2026-11-18 from the file of 2026-11-10, beside `data/raw/bls/ce.footnote`. In the main period C1 runs from 47.4 to 86.1, with a mean of 69.9.
- **Archived values (D8).** The last Internet Archive copy of BLS's registry-receipts page (`cesregrec.htm`) in each year from 2008 to 2024, and the copy of 2025-03-19, are in `data/raw/internet-archive/cesregrec/`. Each copy shows the history from 1981 as BLS showed it that day. The check compared 23,355 C1, C2 and C3 values in the 18 copies with LABSTAT's, to one decimal (first table below). One value differs: C1 for June 2012 is 64.6 in the copy of 2012-11-14, and 72.9 in LABSTAT and in every later copy. The owner accepted the difference on 2026-11-18, and the fits use LABSTAT's 72.9. No copy holds a value that LABSTAT lacks.
- **Window to first closing `W_t`.** The closing day follows the owner's rule: "first closing, when BLS stops collecting survey responses for the first estimate, falls one week before the release" (owner, 2026-11-17). So the closing day is the release date less 7 calendar days. `W_t` counts the federal business days from the 13th of the reference month through the closing day (`first_closing_days` in `data/panel/collection_days.parquet`). `D_t` counts the same days through the day before the release, so `D_t − W_t` is 4, or 3 when a holiday falls in the release's final week. In the 278 releases of 2003-01 to 2026-08 that no lapse delayed, `W_t` runs from 8 to 15.
- **Window start (D3).** BLS's [response-rate page](https://www.bls.gov/osmr/response-rates/) gives "10–16 business days" of collection before the first preliminary release. No start rule tried reproduces that range: from the 13th the window is 8–15, from the 12th 9–15, from the Monday of the reference week 13–18, and from the Sunday after the reference week 8–13. A start that shifts every release by the same number of days changes no slope, so `W_t` starts on the 13th, as `D_t` does.
- **Samples.** The main period is the reference months 2003-01 to 2026-08 with a release, less September and October 2025 (dropped, as in Stage 1), March 2020 to December 2021 (reported apart) and the four lapse-delayed months (D2). It holds 256 months in 23 reference years. Part (a) takes all 256, and no main month lacks C1. Part (b) and O4 take each main month and transition with a published revision: 254 F→S and 252 F→T for each status. With no lapse month in the fits, the fits have no lapse indicator.
- **Fits.** Each fit has reference-year and calendar-month effects and the 5-week dummy `F_t` (`five_week_interval` in `ces_revisions.collection_window`). Standard errors cluster by reference year with the CR1 factor, and each 95% interval uses a t quantile on 22 degrees of freedom (23 clusters less one). A statsmodels refit of the 13 fits agreed with these values to within 1e-12.

## Results

### Archived copies compared with LABSTAT

<!-- BEGIN GENERATED stage-9-copies -->

| Year | Copy | Values compared | Differing | Only in the copy |
|---|---|---:|---:|---:|
| 2008 | 20081204164446 | 995 | 0 | 0 |
| 2009 | 20090831195859 | 1022 | 0 | 0 |
| 2010 | 20100615144206 | 1052 | 0 | 0 |
| 2011 | 20111205012811 | 1106 | 0 | 0 |
| 2012 | 20121114143627 | 1139 | 1 | 0 |
| 2013 | 20131109043233 | 1175 | 0 | 0 |
| 2014 | 20141110094945 | 1211 | 0 | 0 |
| 2015 | 20150905111214 | 1241 | 0 | 0 |
| 2016 | 20161214012544 | 1286 | 0 | 0 |
| 2017 | 20170523103601 | 1301 | 0 | 0 |
| 2018 | 20180719105627 | 1343 | 0 | 0 |
| 2019 | 20191212004825 | 1394 | 0 | 0 |
| 2020 | 20201219101554 | 1430 | 0 | 0 |
| 2021 | 20211228210841 | 1466 | 0 | 0 |
| 2022 | 20221209230559 | 1502 | 0 | 0 |
| 2023 | 20231207233658 | 1535 | 0 | 0 |
| 2024 | 20241231185714 | 1574 | 0 | 0 |
| 2025 | 20250319165303 | 1583 | 0 | 0 |
| All | 18 copies | 23355 | 1 | 0 |

<!-- END GENERATED stage-9-copies -->

### Closing days on a federal holiday

In six releases the closing day by the rule is itself a federal holiday (D6). The rule is kept as stated, and the business-day count is unaffected. The second table counts the releases of the main period and of March 2020 to December 2021 with a holiday in the final week.

<!-- BEGIN GENERATED stage-9-holidays -->

| Reference month | Release | Closing day |
|---|---|---|
| 2004-12 | 2005-01-07 | 2004-12-31 |
| 2009-12 | 2010-01-08 | 2010-01-01 |
| 2010-12 | 2011-01-07 | 2010-12-31 |
| 2015-12 | 2016-01-08 | 2016-01-01 |
| 2020-12 | 2021-01-08 | 2021-01-01 |
| 2021-12 | 2022-01-07 | 2021-12-31 |

| Calendar month | Releases | With a final-week holiday |
|---|---:|---:|
| January | 23 | 0 |
| February | 24 | 0 |
| March | 24 | 0 |
| April | 24 | 0 |
| May | 24 | 13 |
| June | 24 | 13 |
| July | 24 | 0 |
| August | 24 | 10 |
| September | 21 | 0 |
| October | 21 | 0 |
| November | 22 | 0 |
| December | 23 | 10 |
| All | 278 | 46 |

<!-- END GENERATED stage-9-holidays -->

### C1 by window to first closing

<!-- BEGIN GENERATED stage-9-rate-by-window -->

| `W_t` | Months | Mean C1 |
|---|---:|---:|
| 8 | 1 | 54.4 |
| 9 | 22 | 64.5 |
| 10 | 56 | 67.3 |
| 11 | 34 | 68.5 |
| 12 | 38 | 70.5 |
| 13 | 49 | 71.9 |
| 14 | 39 | 73.3 |
| 15 | 17 | 74.6 |
| All | 256 | 69.9 |

<!-- END GENERATED stage-9-rate-by-window -->

### Part (a): C1 on the window

```math
C1_t = \alpha + \beta W_t + \delta F_t + \mu_{y(t)} + \nu_{m(t)} + \varepsilon_t
```

<!-- BEGIN GENERATED stage-9-rate-regression -->

| n | Years | `W_t` coefficient | SE | 8→15 days, 95% CI |
|---:|---:|---:|---:|---:|
| 256 | 23 | 2.20 | 0.26 | 11.7 to 19.2 |

<!-- END GENERATED stage-9-rate-regression -->

### Revision size by C1 quintile

The quintiles cut C1 less its reference year's mean over the 256 part (a) months, the within-year variation that the fit uses. Each cell is the mean absolute revision in thousands.

<!-- BEGIN GENERATED stage-9-revision-by-rate -->

| Quintile | Months | Deviation | NSA F→S | NSA F→T | SA F→S | SA F→T |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 52 | -16.1 to -4.0 | 54.7 | 57.2 | 31.0 | 39.0 |
| 2 | 51 | -4.0 to -1.4 | 41.9 | 44.0 | 26.8 | 37.5 |
| 3 | 51 | -1.3 to 1.7 | 35.4 | 44.1 | 23.4 | 39.3 |
| 4 | 52 | 1.8 to 4.5 | 41.9 | 47.6 | 28.4 | 43.4 |
| 5 | 50 | 4.5 to 10.3 | 42.4 | 44.8 | 30.5 | 50.2 |

<!-- END GENERATED stage-9-revision-by-rate -->

### Part (b): revision size on C1

```math
|r_t| = \alpha + \theta\, C1_t + \delta F_t + \mu_{y(t)} + \nu_{m(t)} + \varepsilon_t
```

<!-- BEGIN GENERATED stage-9-revision-regression -->

| Status | Transition | n | Years | C1 coefficient | SE | +10 points, 95% CI |
|---|---|---:|---:|---:|---:|---:|
| NSA | F→S | 254 | 23 | -0.56 | 0.42 | -14.4 to 3.1 |
| NSA | F→T | 252 | 23 | -0.60 | 0.46 | -15.5 to 3.6 |
| SA | F→S | 254 | 23 | -0.20 | 0.34 | -9.2 to 5.1 |
| SA | F→T | 252 | 23 | 0.46 | 0.59 | -7.7 to 16.8 |

<!-- END GENERATED stage-9-revision-regression -->

### O4: the Stage 1 regression with the window to first closing

```math
|r_t| = \alpha + \beta X_t + \delta F_t + \mu_{y(t)} + \nu_{m(t)} + \varepsilon_t, \qquad X_t \in \{D_t, W_t\}
```

`X_t` is `D_t` in one fit and `W_t` in the other, on the same rows. `W_t − D_t` is constant except in the 45 releases of the fit sample with a holiday in the final week, so the two slopes can differ only through those releases. The holiday table counts 46, because it also covers the pandemic period, which adds May 2021. This `D_t` fit on 2003–2026 belongs to Stage 9 only. It does not replace the Stage 1 tables.

<!-- BEGIN GENERATED stage-9-window-regression -->

| Status | Transition | n | Years | `D_t` coefficient | SE | `D_t` 7 days, 95% CI | `W_t` coefficient | SE | `W_t` 7 days, 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NSA | F→S | 254 | 23 | -0.89 | 2.13 | -37.1 to 24.6 | -0.60 | 2.23 | -36.5 to 28.1 |
| NSA | F→T | 252 | 23 | -0.69 | 2.16 | -36.3 to 26.6 | -0.21 | 2.33 | -35.4 to 32.4 |
| SA | F→S | 254 | 23 | -0.51 | 1.42 | -24.1 to 17.0 | -0.43 | 1.56 | -25.6 to 19.7 |
| SA | F→T | 252 | 23 | -0.28 | 1.60 | -25.3 to 21.3 | 0.10 | 1.75 | -24.7 to 26.0 |

<!-- END GENERATED stage-9-window-regression -->

### Lapse-delayed releases

<!-- BEGIN GENERATED stage-9-lapse -->

| Reference month | Release | Closing day | `D_t` | `W_t` | C1 | NSA F→S | NSA F→T | SA F→S | SA F→T |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2013-09 | 2013-10-22 | 2013-10-15 | 26 | 22 | 77.9 | 14 | 2 | 15 | 27 |
| 2013-10 | 2013-11-08 | 2013-11-01 | 18 | 14 | 83.5 | 31 | 29 | 4 | 4 |
| 2025-11 | 2025-12-16 | 2025-12-09 | 22 | 18 | 73.8 (E) | 2 | 25 | 8 | 23 |
| 2026-01 | 2026-02-11 | 2026-02-04 | 20 | 16 | 69.5 | 14 | 19 | 4 | 30 |

<!-- END GENERATED stage-9-lapse -->

### March 2020 to December 2021

<!-- BEGIN GENERATED stage-9-pandemic -->

| `W_t` | Months | Mean C1 | NSA F→S | NSA F→T | SA F→S | SA F→T |
|---|---:|---:|---:|---:|---:|---:|
| 9 | 4 | 66.1 | 80.0 | 101.2 | 56.2 | 181.2 |
| 10 | 3 | 70.7 | 83.3 | 80.3 | 82.7 | 66.7 |
| 11 | 2 | 68.4 | 211.0 | 458.0 | 155.5 | 460.0 |
| 12 | 4 | 71.1 | 97.2 | 109.8 | 112.2 | 144.2 |
| 13 | 3 | 72.8 | 59.3 | 99.3 | 137.7 | 224.0 |
| 14 | 3 | 74.5 | 41.7 | 34.0 | 50.0 | 55.7 |
| 15 | 3 | 75.5 | 62.3 | 92.3 | 99.0 | 145.7 |
| All | 22 | 71.2 | 85.0 | 121.7 | 95.1 | 168.1 |

<!-- END GENERATED stage-9-pandemic -->

## Limits

- **The rule is the owner's.** The closing day comes only from the owner's rule: "first closing, when BLS stops collecting survey responses for the first estimate, falls one week before the release" (owner, 2026-11-17). BLS's "10–16 business days" is not reproduced (D3).
- **Holidays.** In 6 releases the closing day by the rule is a federal holiday (D6). In 46 a holiday falls in the release's final week, which makes `D_t − W_t` 3, not 4.
- **Lapse-delayed months.** The four lapse-delayed months, 2013-09, 2013-10, 2025-11 and 2026-01, are out of every fit (D2) and in a table of their own. The rule puts the 2013-09 closing day, 2013-10-15, inside the lapse of October 1–16, 2013, when BLS suspended all survey operations.
- **O3 (b) is an association.** C1 moves for reasons other than the window, for example the shift to automated collection that the archived page names as the cause of its upward trend. Part (b) cannot separate those reasons from an effect of C1. C1 and the revisions are for total nonfarm only.
- **Footnote `E`.** LABSTAT marks C1 for September and November 2025 with footnote `E`, "Impacted by 2025 lapse in appropriations" (`data/raw/bls/ce.footnote`). Neither month is in a fit: the sample drops September 2025, and November 2025 is lapse-delayed.
- **The archive check.** One archived C1 differs from LABSTAT's, for June 2012, and the fits use LABSTAT's value (owner, 2026-11-18). The copies start in 2008, so the check cannot see a change made before the first copy, or a change undone between two copies.
- **Later prints.** The lapse flag marks a month's first release only. In the main period, 3 F→S revisions (August 2013, and August and December 2025) and 4 F→T revisions (July and August 2013, and July and August 2025) take their later estimate from a release that a lapse delayed. They enter the fits unflagged (D2).
- **The second release's window.** The F→S revision also depends on how long collection ran before the second estimate. The fits do not include that window.
- **Benchmark effects.** From 2004 on, the November third estimate and the December second estimate come from the January benchmark release. So November's F→T and December's F→S revisions include part of the benchmark revision. The calendar-month effects absorb only its average.
- **Standard errors.** The 23 reference-year clusters allow correlation inside a year, but not between December and the next January. Cluster-robust errors from 23 clusters are themselves uncertain.
- **Rounding.** The revision table's estimates are rounded to thousands, and C1 to one decimal.

## Measured cost

- **Wall time of the real run** (on this Mac, from a worktree with no `data/panel/`): `scripts/vintage_sources.py build` took 8.35 s and `scripts/first_closing.py report` 1.43 s, measured with `/usr/bin/time -p`.
- **Network fetches**, each timed apart with `/usr/bin/time -p`: `fetch-labstat` took 37.85 s, after a first run that stopped in 31.28 s with nothing staged, because LABSTAT's answer had no `Content-Length`. `fetch-archive` took 190.94 s for 2 listings and 18 copies.
- **Sessions:** 4, one for each pull request: the session that resumed the roadmap for Stage 9 (PR 22), the one that wrote the spec (PR 23), the one that wrote plan 15 (PR 24), and the one that executed plan 15 on 2026-11-18.
