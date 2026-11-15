# Stage 2 findings: news and noise in the total-nonfarm revisions

Outcome O5 of the [ces-revision-drivers roadmap](ces-revision-drivers-roadmap.md), Stage 2, implemented by [plan 13](plans/completed/13-ces-revision-drivers-stage-2.md) with [plan 11](plans/completed/11-ces-revisions.md). `uv run python scripts/pilot_run.py shares-report docs/runs/stage-6/news-shares.json` writes the table below from the committed `docs/runs/stage-6/news-shares.json`. Do not edit it by hand.

## Question

How much of the variance of each total-nonfarm payroll revision is news, and how much is noise? The revisions are the first-to-second print (F→S) and the second-to-third print (S→T) of each month's over-the-month change in employment, from 1979 to 2026.

## Answer

News is the larger part of each revision's variance, by the posterior median. The F→S news share is 0.723, with a 90% interval from 0.553 to 0.853. The S→T news share is 0.652, with a wider 90% interval from 0.300 to 0.847, so the data do not rule out that noise is the larger part of the S→T revision.

## Data and model

- **Model.** The model is plan 11's aggregate pilot: a linear-Gaussian state-space model of total-nonfarm employment, not seasonally adjusted. It takes the same-release F, S and T changes of every reference month from 1979, one composite shift in the news and noise scales at May 2003, and the 23 final March benchmark anchors. A Kalman filter integrates out every latent state. Four NUTS chains sample the 17 static parameters.
- **News and noise.** The model splits each revision's variance into two sources. News terms are new information that the earlier print did not have. Noise terms are the measurement errors of the prints. The news share is the news variance divided by the sum of the news variance and the noise variance. `derived()` in `src/ces_revisions/pilot/model.py` computes it for each posterior draw: `share_s` for the F→S revision and `share_t` for the S→T revision.
- **Seasonal state.** The fit uses the dummy layout, as `docs/decisions/seasonal-state.md` records.
- **Fit quality.** The fit meets Req 18 (`docs/runs/stage-6/pilot-fit/diagnostics.json`). The largest R-hat, a check that the chains agree, is 1.004 against a limit of 1.01. The smallest bulk and tail effective sample sizes are 1694 and 1470 against a floor of 400. No draw diverged, no tree saturated, and the smallest BFMI is 0.97.
- **Calibration.** Simulation-based calibration (SBC) fits the model to panels simulated from its own prior and checks that it recovers the parameters. The reduced-panel SBC passed: 100 replicates over 1999–2021 (`docs/runs/stage-6/sbc-summary.json`). For both shares, no replicate had a posterior z-score beyond 3. The median contraction, which is 1 minus the ratio of posterior to prior variance, is 0.854 for `share_s` and 0.581 for `share_t`. The floor is 0.25. The data inform the F→S share more than the S→T share, as the two intervals also show.

## Results

<!-- BEGIN GENERATED stage-2-shares -->

| Revision | Share | Median | 90% interval |
|---|---|---:|---:|
| F→S | `share_s` | 0.723 | 0.553–0.853 |
| S→T | `share_t` | 0.652 | 0.300–0.847 |

The shares summarize 4000 pooled posterior draws of `data/runs/stage6-pilot-seed2995-20261022T020454Z`, which took 1659.6 seconds to sample.

<!-- END GENERATED stage-2-shares -->

`share_t` is the news share of the S→T revision. It is not an F→T share, which Stage 1 reports, so the two cannot be compared directly.

## Limits

- **Model-implied shares.** The table gives variance shares that the model implies. It does not split the revision of any single month into news and noise.
- **Rounding variance.** The shares leave out the variance that rounding adds to the published figures.
- **The 2003 shift.** The May 2003 shift scales the S and T news and all three noise standard deviations by one factor. So the shares are the same before and after May 2003. The table has no separate pre-2003 pair.
- **The crisis months.** The crisis multiplier scales the S and T news shocks and the non-news level shock in March to June 2020, and no noise term. So the shares are larger in those four months. The table gives no separate crisis pair.
- **One series.** The model has one aggregate NSA series. Benchmark revisions and seasonal-adjustment revisions stay outside it. Outcome O6 puts them inside the model.
- **Validation.** SBC ran on the reduced 1999–2021 panel only. The prior-predictive rerun with every prior scale doubled missed Req 15's monthly constraint: 13.3% of months had an F→T revision above 250 thousand, against a ceiling of 5% (`docs/runs/stage-6/prior-predictive-doubled.json`). Full-panel SBC and a prior-sensitivity run wait for outcome O17.

## Measured cost

- **Sampling time.** The fit took 1659.6 seconds, about 27.7 minutes. That time covers compilation, warmup and sampling of four chains on the Mac's CPU on 2026-10-22 (`docs/runs/stage-6/pilot-fit/provenance.json`). The generated line above gives the same figure.
- **Sessions.** Stage 2 used two sessions: one wrote plan 13, and one executed it.
