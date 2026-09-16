# Use inferred cumulative job changes for every public post-March operator

- **Status:** Accepted
- **Date:** 2026-09-16
- **Deciders:** Lowell Mason
- **Blast radius:** roadmap Stages 8, 17, 20, 23, and 24; every model or report that uses post-March or B→M observations

## Context

Req 10 of [`specs/ces-revisions.md`](../../specs/ces-revisions.md) preserves BLS's post-March calculation as either (1) a recursion in previously published matched-sample link relatives and revised net birth–death forecasts or (2) a linear cumulative operator when only job-change contributions are available. The two representations may not be mixed. Roadmap Stage 5 must determine, for each benchmark year 2003–2025, whether the required public series is recovered or inferred and must assign positive reconstruction-error variance to inferred years.

The committed vintage triangles contain published NSA and SA levels, not matched-sample employment sums, weighted link relatives, or sample-only job changes. The Stage 4 birth–death table contains the initial January–December schedule from 2004 onward and each revised post-benchmark April–December schedule. It contains no initial April–December 2003 schedule.

For 2004–2025 we tested the strongest aggregate link proxy public data permit:

```math
\widehat R^{old}_{s,m}
=\frac{E^{old}_{s,m}-BD^{old}_{s,m}}{E^{old}_{s,m-1}},
\qquad
\widehat E^B_{s,m}
=\widehat E^B_{s,m-1}\widehat R^{old}_{s,m}+BD^{new}_{s,m}.
```

Each attempt used all twelve sectors and April–October, 84 cells per year. No year reproduced all cells within ±0.5 thousand. The failure is expected: BLS applies weighted link-relative estimation below the published supersector aggregate, and whole-thousand aggregate levels do not identify those private matched-sample inputs. The pandemic and later method-change years show especially large residuals.

## Decision

Use `cumulative_job_change` for all 23 benchmark years. Every series is `inferred`; no historical link-relative series is labeled recovered. For each published benchmark vintage infer the sample contribution as

```math
\widehat{SC}_{s,m}=E^B_{s,m}-E^B_{s,m-1}-BD^{new}_{s,m},
```

and represent the path as the sparse cumulative map

```math
E^B_{s,m}=b^{fin}_{s,y}
+\sum_{h=\mathrm{Apr}(y)}^m
\left(\widehat{SC}_{s,h}+BD^{new}_{s,h}\right).
```

This identity reproduces the published April–December benchmark vintage and explicitly consumes revised birth–death values. It is an operator reconstruction of the published path, not an observation of BLS's private matched-sample link relatives. Later fits propagate the per-year reconstruction-error variance below rather than fixing the inferred components as error-free.

For 2004–2025 the variance is the mean squared residual from the failed aggregate link-proxy attempt, floored at `1/6` thousand squared. The floor is the variance of the difference of two independently rounded whole-thousand levels. The 2003 initial schedule is absent, so its variance is the `1/6` floor. The recoverability parquet retains all 84 cell residuals and their level-cell and birth–death source keys for every attempted year; the table below is only the readable summary.

| Year | Representation | Status | Link attempt | Cells within ±0.5 / attempted | Max residual | Variance |
|---:|---|---|---|---:|---:|---:|
| 2003 | cumulative job change | inferred | inputs missing | 0 / 0 | — | 0.166667 |
| 2004 | cumulative job change | inferred | failed | 19 / 84 | 33.407275 | 65.199745 |
| 2005 | cumulative job change | inferred | failed | 20 / 84 | 61.890801 | 228.730081 |
| 2006 | cumulative job change | inferred | failed | 15 / 84 | 23.731192 | 51.300815 |
| 2007 | cumulative job change | inferred | failed | 16 / 84 | 34.228674 | 58.660143 |
| 2008 | cumulative job change | inferred | failed | 17 / 84 | 22.098039 | 18.249807 |
| 2009 | cumulative job change | inferred | failed | 13 / 84 | 40.264512 | 78.661208 |
| 2010 | cumulative job change | inferred | failed | 12 / 84 | 31.813475 | 91.563345 |
| 2011 | cumulative job change | inferred | failed | 16 / 84 | 20.359022 | 35.259417 |
| 2012 | cumulative job change | inferred | failed | 20 / 84 | 78.412368 | 520.337201 |
| 2013 | cumulative job change | inferred | failed | 11 / 84 | 37.113240 | 48.080295 |
| 2014 | cumulative job change | inferred | failed | 19 / 84 | 26.630662 | 68.188332 |
| 2015 | cumulative job change | inferred | failed | 8 / 84 | 63.751727 | 331.085586 |
| 2016 | cumulative job change | inferred | failed | 13 / 84 | 32.440472 | 81.254024 |
| 2017 | cumulative job change | inferred | failed | 14 / 84 | 8.913966 | 7.778542 |
| 2018 | cumulative job change | inferred | failed | 11 / 84 | 24.771480 | 93.717486 |
| 2019 | cumulative job change | inferred | failed | 17 / 84 | 13.991568 | 16.178823 |
| 2020 | cumulative job change | inferred | failed | 3 / 84 | 162.058501 | 1607.645923 |
| 2021 | cumulative job change | inferred | failed | 12 / 84 | 179.079008 | 1293.123145 |
| 2022 | cumulative job change | inferred | failed | 5 / 84 | 43.910696 | 157.995551 |
| 2023 | cumulative job change | inferred | failed | 14 / 84 | 74.270533 | 433.893346 |
| 2024 | cumulative job change | inferred | failed | 18 / 84 | 38.587326 | 108.426410 |
| 2025 | cumulative job change | inferred | failed | 11 / 84 | 17.229047 | 30.739396 |

## Consequences

- **Positive:** The public operator is exactly reproducible from committed data, revised birth–death enters explicitly, no private BLS series is fabricated, and every inferred year carries uncertainty into Stage 8.
- **Negative:** The components cannot identify the multiplicative matched-sample link separately from aggregation and source rounding. Their variance is an operator-reconstruction variance, not QCEW benchmark error or sampling variance.
- **Neutral:** November and December remain in the artifact with `additional_sample_receipts=True`; the April–October reproduction gate does not pretend their later sample receipts are a fixed link.

## Alternatives considered

- **Label the aggregate proxy recovered.** Rejected: every year fails at least 64 of 84 cells at publication rounding, and the required matched-sample inputs are not in a public source.
- **Infer link relatives from the target benchmark path and revised birth–death.** Algebraically possible but observationally identical to the chosen cumulative contributions; calling the quotient a recovered link would add interpretation without information.
- **Use both representations by year or sector.** Rejected by Req 10 and unnecessary: no year passes the recovery gate.
- **Treat inferred contributions as fixed without variance.** Rejected: it would turn rounded, underidentified public aggregates into private production inputs by assumption.

## Revisit triggers

Revisit if BLS supplies historical matched-sample weighted link relatives or sample-only job-change contributions with a documented vintage clock. A later source may replace a year's `inferred` row with `recovered` only if the link-relative operator reproduces every April–October published level within ±0.5 thousand and its reconstruction-error variance becomes exactly zero.
