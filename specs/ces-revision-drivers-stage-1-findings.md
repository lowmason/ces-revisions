# Stage 1 findings: collection window and revision size, total nonfarm

Outcome O1 of the [ces-revision-drivers roadmap](ces-revision-drivers-roadmap.md), Stage 1, implemented by [plan 12](plans/completed/12-ces-revision-drivers-stage-1.md). `uv run python scripts/collection_window.py report` writes the four tables below from the built `data/panel/`. Do not edit them by hand.

## Question

How do the mean absolute F→S and F→T revisions of total nonfarm employment change with the length of the collection window before the first print, 1979–2026?

## Answer

Within the windows that BLS's normal schedule gives, 12 to 19 federal business days, the data show no clear effect of window length on revision size. They also cannot rule out a moderate one. Most of the variation in `D_t` comes from the calendar: months with a 5-week survey interval have windows about 2.5 business days longer. With a control for that interval, the 95% interval for the change from 12 to 19 days is about -23 to +20 thousand for F→S, against a mean revision of 39 to 45 thousand. For F→T it is -71 to +57 thousand (NSA) and -24 to +23 thousand (SA), so the F→T fits are uninformative. Without the control, the F→S interval is narrower, -23 to +9 thousand (NSA), but that fit cannot separate extra collection days from a 5-week month. The five releases in the sample that a lapse in appropriations delayed have smaller revisions: their indicator is from -23 to -40 thousand, about three standard errors from zero in three of the four fits. But five releases cannot separate a longer window from the other effects of a lapse (Limits).

## Data and day count

- **Revisions.** BLS's revision table (`cesnaicsrev`) gives the first, second and third estimates of each month's over-the-month change in total nonfarm, NSA and SA, in thousands, for reference months from January 1979. These are the table's rows in the built `data/panel/panel.parquet`. A revision is the absolute difference between the second (F→S) or third (F→T) estimate and the first. It exists only where both estimates are published, so the `A` and `NA` markers (March and April 2003, the 2025 lapse) give no revision, never a zero. F→T has 2 fewer revisions than F→S in the main period: March 2003's third estimate is `NA`, and the T series ends at June 2026, a month before the S series.
- **Day count `D_t`.** For each release, `D_t` counts the federal business days from the first business day after the 12th of the reference month up to, but not including, the release date (`data/panel/collection_days.parquet`, written by `scripts/vintage_sources.py build`). The holidays are the legal public holidays of 5 U.S.C. 6103(a), computed by rule: the King holiday from 1986, Juneteenth from 2021, and a holiday on a Saturday or Sunday observed on the Friday before or the Monday after. Inauguration Day, a holiday only in the Washington area, and closures by executive order are not holidays (owner, plan 12, D5). Days of a lapse in appropriations count as business days.
- **Lapse flag.** The five releases in the sample that a lapse delayed carry an indicator: December 1995, September and October 2013, November 2025 and January 2026 (owner, plan 12, D1). The 2025 lapse also delayed the September 2025 release, which the sample drops, and the day-count file flags it too. The lapses of January 2018 and 2018–19 delayed no release.
- **Survey interval.** A month has a 5-week interval when 35 days, not 28, separate the Sunday that starts the week holding its 12th from the same Sunday of the month before (`five_week_interval` in `ces_revisions.collection_window`). This is a rule computed from the calendar, not a BLS designation. CES seasonal adjustment has a special adjustment for this interval, so it is a known source of month-to-month differences. In the main period, 190 of the 546 F→S revisions are 5-week months, with a mean `D_t` of 17.4 business days against 14.9 for the others.
- **Sample.** Reference months September and October 2025 are dropped. March 2020 to December 2021 is reported apart and is not in the regression (owner, plan 12, D6).

## Results

### Main period

<!-- BEGIN GENERATED stage-1-main -->

| `D_t` | n F→S | n F→T | Lapse-delayed | NSA F→S | NSA F→T | SA F→S | SA F→T |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 1 | 1 | 0 | 70.0 | 84.0 | 37.0 | 26.0 |
| 13 | 61 | 61 | 0 | 47.5 | 54.1 | 38.5 | 49.6 |
| 14 | 117 | 117 | 0 | 50.9 | 73.0 | 42.9 | 55.3 |
| 15 | 73 | 73 | 0 | 36.0 | 65.5 | 34.2 | 43.8 |
| 16 | 89 | 88 | 0 | 39.1 | 49.9 | 33.8 | 48.5 |
| 17 | 92 | 92 | 0 | 43.0 | 70.1 | 38.7 | 52.7 |
| 18 | 83 | 83 | 1 | 48.9 | 70.1 | 42.1 | 56.9 |
| 19 | 26 | 25 | 0 | 45.7 | 109.3 | 39.5 | 67.6 |
| 20 | 1 | 1 | 1 | 14.0 | 19.0 | 4.0 | 30.0 |
| 22 | 1 | 1 | 1 | 2.0 | 25.0 | 8.0 | 23.0 |
| 24 | 1 | 1 | 1 | 11.0 | 8.0 | 10.0 | 6.0 |
| 26 | 1 | 1 | 1 | 14.0 | 2.0 | 15.0 | 27.0 |
| All | 546 | 544 | 5 | 44.5 | 66.4 | 38.5 | 52.1 |

<!-- END GENERATED stage-1-main -->

### March 2020 to December 2021

<!-- BEGIN GENERATED stage-1-pandemic -->

| `D_t` | n F→S | n F→T | Lapse-delayed | NSA F→S | NSA F→T | SA F→S | SA F→T |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 13 | 4 | 4 | 0 | 80.0 | 101.2 | 56.2 | 181.2 |
| 14 | 3 | 3 | 0 | 83.3 | 80.3 | 82.7 | 66.7 |
| 15 | 3 | 3 | 0 | 141.0 | 310.3 | 111.7 | 325.0 |
| 16 | 3 | 3 | 0 | 129.3 | 141.3 | 141.7 | 174.0 |
| 17 | 3 | 3 | 0 | 59.3 | 99.3 | 137.7 | 224.0 |
| 18 | 3 | 3 | 0 | 41.7 | 34.0 | 50.0 | 55.7 |
| 19 | 3 | 3 | 0 | 62.3 | 92.3 | 99.0 | 145.7 |
| All | 22 | 22 | 0 | 85.0 | 121.7 | 95.1 | 168.1 |

<!-- END GENERATED stage-1-pandemic -->

### Regression

One specification is fit four times, once for each seasonal status and transition, on the main period:

```math
|r_t| = \alpha + \beta D_t + \gamma L_t + \mu_{y(t)} + \nu_{m(t)} + \varepsilon_t
```

Here `|r_t|` is the absolute revision in thousands, `L_t` the lapse indicator, `μ_{y(t)}` a reference-year effect and `ν_{m(t)}` a calendar-month effect. Standard errors cluster by reference year with the CR1 factor (owner, plan 12, D7), and the table gives β and γ with them. A statsmodels refit of the four designs agreed with these values to within 1e-11. The 95% interval column bounds `7β`, the change from 12 to 19 business days, with a t quantile on 46 degrees of freedom (47 clusters less one).

<!-- BEGIN GENERATED stage-1-regression -->

| Status | Transition | n | Years | `D_t` coefficient | SE | 12→19 days, 95% CI | Lapse coefficient | SE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NSA | F→S | 546 | 47 | -1.04 | 1.12 | -23.1 to 8.5 | -29.80 | 9.25 |
| NSA | F→T | 544 | 47 | -0.27 | 2.23 | -33.3 to 29.5 | -23.11 | 18.62 |
| SA | F→S | 546 | 47 | -0.45 | 1.05 | -18.0 to 11.7 | -24.84 | 8.28 |
| SA | F→T | 544 | 47 | 1.10 | 1.23 | -9.7 to 25.0 | -40.43 | 10.74 |

<!-- END GENERATED stage-1-regression -->

### Regression with the survey-interval control

The same four fits add `δ F_t`, where `F_t` is 1 for a month with a 5-week survey interval. After the year and month effects, `D_t` keeps a residual standard deviation of 1.8 business days. `F_t` accounts for about half of that variation, which leaves 1.26 days. So the standard errors of β grow, most for NSA F→T. A statsmodels refit of these four designs agreed with these values to within 1e-6.

<!-- BEGIN GENERATED stage-1-regression-interval -->

| Status | Transition | n | Years | `D_t` coefficient | SE | 12→19 days, 95% CI | 5-week coefficient | SE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NSA | F→S | 546 | 47 | -0.28 | 1.48 | -22.8 to 18.9 | -3.72 | 4.72 |
| NSA | F→T | 544 | 47 | -0.98 | 4.55 | -70.9 to 57.2 | 3.48 | 13.23 |
| SA | F→S | 546 | 47 | 0.15 | 1.35 | -18.0 to 20.1 | -2.92 | 3.96 |
| SA | F→T | 544 | 47 | -0.09 | 1.66 | -24.0 to 22.7 | 5.83 | 4.93 |

<!-- END GENERATED stage-1-regression-interval -->

## Limits

- **`D_t` is a proxy.** BLS publishes no first-closing dates, so the release date stands in for the end of first-closing collection. The [Stage 9 findings](ces-revision-drivers-stage-9-findings.md) repeat the regression with the window to first closing. That window follows the owner's rule (owner, 2026-11-17): first closing, when BLS stops collecting survey responses for the first estimate, falls one week before the release.
- **The window varies little.** 542 of the 546 main-period releases have `D_t` from 12 to 19, and the cells for 12, 20, 22, 24 and 26 hold one release each. The slope rests on that narrow range. It does not show what a window much longer than the schedule's would do.
- **The window is mostly calendar.** The weekday of the 12th sets both the reference week and the release date, so `D_t` follows the calendar closely. The 5-week control removes the largest calendar effect it shares. Other calendar effects that the year and month dummies do not take in may remain.
- **The second release's window.** The F→S revision also depends on how long collection ran before the second estimate. The fits do not include that window.
- **Five lapse releases.** Four of their windows hold days of a lapse, which count as business days: BLS was closed in the December 1995 and October 2013 lapses, and whether it closed in 2026 is not found. The November 2025 window began the day after the 2025 lapse ended, and BLS extended that month's collection period. The indicator keeps these releases from driving the slope, but it also takes in every other effect of a lapse.
- **Later prints.** The flag marks a month's first release only. In the main period, 4 F→S revisions (November 1995, August 2013, and August and December 2025) and 5 F→T revisions (October 1995, July and August 2013, and July and August 2025) take their later estimate from a release that a lapse delayed. They enter the table and the regression unflagged (owner, plan 12, final review).
- **Benchmark effects.** From 2004 on, the November third estimate and the December second estimate come from the January benchmark release (plan 5's reconciliation). So November's F→T and December's F→S revisions include part of the benchmark revision. The calendar-month effects absorb only its average (owner, plan 12, D8). Before 2003 this is unchecked.
- **Standard errors.** The 47 reference-year clusters allow correlation inside a year, but not between December and the next January.
- **Rounding.** The table's estimates are rounded to thousands, so a revision under 1,000 jobs reads as 0 or 1.

## Measured cost

- **Wall time of the real run** (on this Mac, from a worktree with no `data/panel/`): `scripts/vintage_sources.py build` took 7.22 s and `scripts/collection_window.py report` 0.38 s, measured with `/usr/bin/time -p`.
- **Sessions:** 1, the session that wrote and executed plan 12 on 2026-11-15.
