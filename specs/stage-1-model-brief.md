# Brief for /recommend-probabilistic-model: collection window and revision scale, total nonfarm

Paste this after `/recommend-probabilistic-model`. It is the problem statement for Stage 1 (outcome O1) of `specs/ces-revision-drivers-roadmap.md`. The current answer and its generated tables are in `specs/ces-revision-drivers-stage-1-findings.md`; the code is `src/ces_revisions/collection_window.py`.

**Sample decision (owner, 2026-11-17):** model reference months from 2003 on only, because of the probability-sample redesign. The figures below cover 1979–2026. The 2003+ figures are in `specs/stage-1-scale-model/recommendation.md` §2.

## Question

Does the length of the CES collection window before the first print change the **scale** (spread, not mean) of payroll revisions to total nonfarm employment, 1979–2026? The quantity of interest is the change in revision scale from 12 to 19 federal business days, the range of BLS's normal schedule, with honest uncertainty. The question is inferential (uncertainty about one association), not predictive.

Causal framing, taken as given here: `D_t` is set by the calendar (the weekday of the 12th and the release rule), so it is treated as as-good-as-random **conditional on** calendar effects: reference year, calendar month, and the 4- vs 5-week survey interval. Do not redesign the identification; choose the likelihood and structure that best estimate the scale association under that conditioning. Flag it if a candidate model would quietly change the conditioning set.

## Data (all on disk, offline)

- Source: BLS's revision table `cesnaicsrev`, rows in `data/panel/panel.parquet` (build with `uv run python scripts/vintage_sources.py build`). `absolute_revisions(...)` in `ces_revisions.collection_window` joins them to the day counts in `data/panel/collection_days.parquet`.
- Unit: one reference month. Four outcome series per month: {NSA, SA} × {F→S, F→T}. F→S = second estimate minus first; F→T = third minus first. So F→T = F→S + S→T, the series share the first estimate, and they are correlated (|r| correlations 0.31–0.82; NSA–SA F→S is 0.82).
- Signed revisions are available, not only absolute ones. Main period, in thousands:

| Series | n | mean | sd | skew | kurtosis | share exactly 0 | max \|r\| |
|---|---:|---:|---:|---:|---:|---:|---:|
| NSA F→S | 546 | 2.3 | 59.8 | 0.19 | 4.6 | 0.009 | 283 |
| SA F→S | 546 | 0.3 | 53.5 | 0.15 | 5.7 | 0.011 | 285 |
| NSA F→T | 544 | 10.8 | 115.2 | -0.15 | 18.1 | 0.011 | 839 |
| SA F→T | 544 | 8.8 | 71.0 | 0.17 | 6.5 | 0.006 | 383 |

- Signed means are near zero; tails are heavy, extreme for NSA F→T. Values are rounded to whole thousands.
- The scale drifts by era (probability-sample redesign around 2003). Mean |r| in thousands, by 1979–89 / 1990–2002 / 2003–26: NSA F→S 55.1 / 38.1 / 42.9; SA F→S 58.1 / 39.9 / 27.7; NSA F→T 92.8 / 75.8 / 47.1; SA F→T 74.2 / 50.7 / 41.5. Note NSA F→S does not fall after 2003 while SA F→S does.
- Regressors:
  - `D_t`: federal business days from the first business day after the 12th up to the release date. Integer, 542 of 546 values in 12–19. After year and month effects its residual SD is 1.8 days. After also controlling for the 5-week flag it is 1.26 days. This thin variation is the binding constraint.
  - `five_week`: 1 when 35 days separate consecutive reference weeks (computed rule, not a BLS flag). 190 of 546 months. Correlation with `D_t` is 0.62; 5-week months average 17.4 days vs 14.9.
  - `lapse_delayed`: 5 releases delayed by a lapse in appropriations (Dec 1995, Sep and Oct 2013, Nov 2025, Jan 2026), with `D_t` of 18–26.
  - Reference year (47 levels) and calendar month (12 levels).
- Sample rules: drop reference months 2025-09 and 2025-10. March 2020–December 2021 (22 months, revisions 2–3× larger) are held out and reported apart.
- Known contamination: from 2004, November's F→T and December's F→S include part of the annual benchmark revision. SA revisions also carry concurrent seasonal-factor revisions that have nothing to do with the window, so NSA is the cleaner test.

## Methods already tried (all frequentist, on |r| unless stated)

1. OLS of |r| on `D_t`, lapse, year and calendar-month fixed effects, CR1 standard errors clustered by year (47 clusters). Each of the four series fit apart. `D_t` coefficient −1.04 to 1.10 thousand per day, all within one SE of zero. Lapse coefficient −23 to −40 thousand. Verified against statsmodels to 1e-11.
2. The same plus the 5-week dummy. `D_t` coefficients move to near zero and SEs grow (NSA F→T 2.23 → 4.55). 95% interval for 12 → 19 days (t, 46 df): F→S about −23 to +20 thousand; F→T NSA −71 to +57, SA −24 to +23. The 5-week coefficient itself is small and imprecise (−3.7 ± 4.7 for NSA F→S). Verified against statsmodels to 1e-6.
3. `log(1+|r|)` as the outcome: same conclusion, `D_t` within about 1 SE of zero.
4. Robustness subsets, each with the null intact: drop November and December; drop lapse rows and one-observation `D_t` cells; post-mid-2003 subsample only.

Weaknesses of what was tried:
- OLS on |r| models the mean of a folded, heavy-tailed variable rather than the scale of the signed revision.
- Additive year effects shift the level but cannot capture the era change in scale.
- The four series are fit separately, which ignores their correlation and the F→T = F→S + S→T nesting.
- 47 year fixed effects spend many parameters on about 12 observations each.
- The window before the second print, which also drives F→S, is omitted.

## What to recommend

Recommend the likelihood and structure for a Bayesian model in NumPyro (`bayesian-workflow` takes the handoff). Specifically weigh:

- Modeling **signed** revisions with a scale that depends on covariates (location-scale model, e.g. log σ_t = calendar terms + β·D_t), against a Gamma or log-normal GLM on |r|. Include heavy-tailed choices (Student-t with estimated ν) given kurtosis 4.6–18.
- How to represent the drift in scale over time: year effects as hierarchical (partial pooling) against fixed, against a smooth trend or random walk on log σ, against era breaks at known redesigns.
- Whether to fit the four series jointly (shared covariate effects with series-specific scales, correlated errors, or explicit nesting of F→S inside F→T), or keep NSA F→S as the primary series and others as secondary.
- How to treat the lapse releases (5 observations), the Nov/Dec benchmark cells, and the rounding to thousands.
- Priors informed by external facts: the β of interest should be weakly informative on a plausible scale (about a few percent of σ per day). `bls-data-context` covers CES revision and seasonal-adjustment facts.
- Whether this stand-alone model or the project's existing state-space machinery (`src/ces_revisions/kalman.py`, where covariates enter the revision variance) is the better home. The Stage 6 pilot's lesson is that NUTS needs `target_accept_prob` 0.95.

Constraints: Python 3.14, NumPyro/JAX float64, ArviZ 1.x, Polars. The Mac fits small models in minutes; GPUs are rented by hand and are not needed at this size. Report the 12 → 19-day contrast on the scale of σ (ratio or thousands) with a posterior interval, and say which model comparison (LOO) or posterior predictive checks would settle the choice.
