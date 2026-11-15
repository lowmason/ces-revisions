# Stage 1 findings: collection window and revision size, total nonfarm

Outcome O1 of the [ces-revision-drivers roadmap](ces-revision-drivers-roadmap.md), Stage 1, implemented by [plan 12](plans/12-ces-revision-drivers-stage-1.md). `uv run python scripts/collection_window.py report` writes the three tables below from the built `data/panel/`. Do not edit them by hand.

## Question

How do the mean absolute F→S and F→T revisions of total nonfarm employment change with the length of the collection window before the first print, 1979–2026?

## Answer

Within the windows that BLS's normal schedule gives, 12 to 19 federal business days, the length of the window shows no clear effect on revision size. In all four fits, the `D_t` coefficient is between -1.04 and 1.10 thousand per business day, and each lies within one standard error of zero. The five releases in the sample that a lapse in appropriations delayed have smaller revisions: their indicator is from -23 to -40 thousand, about three standard errors from zero in three of the four fits. But five releases cannot separate a longer window from the other effects of a lapse (Limits).

## Data and day count

- **Revisions.** BLS's revision table (`cesnaicsrev`) gives the first, second and third estimates of each month's over-the-month change in total nonfarm, NSA and SA, in thousands, for reference months from January 1979. These are the table's rows in the built `data/panel/panel.parquet`. A revision is the absolute difference between the second (F→S) or third (F→T) estimate and the first. It exists only where both estimates are published, so the `A` and `NA` markers (March and April 2003, the 2025 lapse) give no revision, never a zero. F→T has 2 fewer revisions than F→S in the main period: March 2003's third estimate is `NA`, and the T series ends at June 2026, a month before the S series.
- **Day count `D_t`.** For each release, `D_t` counts the federal business days from the first business day after the 12th of the reference month up to, but not including, the release date (`data/panel/collection_days.parquet`, written by `scripts/vintage_sources.py build`). The holidays are the legal public holidays of 5 U.S.C. 6103(a), computed by rule: the King holiday from 1986, Juneteenth from 2021, and a holiday on a Saturday or Sunday observed on the Friday before or the Monday after. Inauguration Day, a holiday only in the Washington area, and closures by executive order are not holidays (owner, plan 12, D5). Days of a lapse in appropriations count as business days.
- **Lapse flag.** The five releases in the sample that a lapse delayed carry an indicator: December 1995, September and October 2013, November 2025 and January 2026 (owner, plan 12, D1). The 2025 lapse also delayed the September 2025 release, which the sample drops, and the day-count file flags it too. The lapses of January 2018 and 2018–19 delayed no release.
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

Here `|r_t|` is the absolute revision in thousands, `L_t` the lapse indicator, `μ_{y(t)}` a reference-year effect and `ν_{m(t)}` a calendar-month effect. Standard errors cluster by reference year with the CR1 factor (owner, plan 12, D7), and the table gives β and γ with them. A statsmodels refit of the four designs agreed with these values to within 1e-11.

<!-- BEGIN GENERATED stage-1-regression -->

| Status | Transition | n | Years | `D_t` coefficient | SE | Lapse coefficient | SE |
|---|---|---:|---:|---:|---:|---:|---:|
| NSA | F→S | 546 | 47 | -1.04 | 1.12 | -29.80 | 9.25 |
| NSA | F→T | 544 | 47 | -0.27 | 2.23 | -23.11 | 18.62 |
| SA | F→S | 546 | 47 | -0.45 | 1.05 | -24.84 | 8.28 |
| SA | F→T | 544 | 47 | 1.10 | 1.23 | -40.43 | 10.74 |

<!-- END GENERATED stage-1-regression -->

## Limits

- **`D_t` is a proxy.** BLS publishes no first-closing dates, so the release date stands in for the end of first-closing collection. Outcome O4 replaces it with BLS's dates when they arrive.
- **The window varies little.** 542 of the 546 main-period releases have `D_t` from 12 to 19, and the cells for 12, 20, 22, 24 and 26 hold one release each. The slope rests on that narrow range. It does not show what a window much longer than the schedule's would do.
- **Five lapse releases.** Four of their windows hold days of a lapse, which count as business days: BLS was closed in the December 1995 and October 2013 lapses, and whether it closed in 2026 is not found. The November 2025 window began the day after the 2025 lapse ended, and BLS extended that month's collection period. The indicator keeps these releases from driving the slope, but it also takes in every other effect of a lapse.
- **Benchmark effects.** From 2004 on, the November third estimate and the December second estimate come from the January benchmark release (plan 5's reconciliation). So November's F→T and December's F→S revisions include part of the benchmark revision. The calendar-month effects absorb only its average (owner, plan 12, D8). Before 2003 this is unchecked.
- **Standard errors.** The 47 reference-year clusters allow correlation inside a year, but not between December and the next January.
- **Rounding.** The table's estimates are rounded to thousands, so a revision under 1,000 jobs reads as 0 or 1.

## Measured cost

- **Wall time of the real run** (on this Mac, from a worktree with no `data/panel/`): `scripts/vintage_sources.py build` took 7.22 s and `scripts/collection_window.py report` 0.38 s, measured with `/usr/bin/time -p`.
- **Sessions:** 1, the session that wrote and executed plan 12 on 2026-11-15.
