# Stage 3 findings: the seasonal-adjustment share of SA revision variance

Outcome O9 of the [ces-revision-drivers roadmap](ces-revision-drivers-roadmap.md), Stage 3, implemented by [plan 16](plans/16-ces-revision-drivers-stage-3.md). `uv run python scripts/seasonal_share.py report` writes the four tables below from the built `data/panel/`. Do not edit them by hand.

## Question

For each sector, transition and regime of the accounting decomposition, what share of the variance of the seasonally adjusted (SA) revision comes from the change in seasonal adjustment?

A revision is the later value of a month's over-the-month employment change minus the earlier value, in thousands of jobs. The three transitions are F→S (first estimate to second), S→T (second to third) and T→B (third to the value after the annual benchmark). The regimes are the reference months before 2003-05, 2003-05 to 2019-12, 2020-01 to 2022-12 and 2023-01 on.

## Answer

The change in seasonal adjustment carries a large part of the SA revision variance in S→T and T→B, and little or none in F→S. Across the 11 supersectors, the median adjustment share is 0.32 to 0.57 for S→T and 0.66 to 0.82 for T→B, depending on the regime. For F→S it is -0.28 to -0.01.

These shares are descriptive. They carry no interval, and none of the three shares is bounded in [0, 1]. The plain ratio is above 1 in 12 of 110 rows (largest 9.93). The adjustment share is below 0 in 27 rows, all F→S, and above 1 in 2.

## Data and shares

- **Input.** `data/panel/accounting_decomposition.parquet`, written by `scripts/vintage_sources.py build`. It has 110 rows. 108 come from BLS's vintage files (`cesvinall`): 12 sectors (total nonfarm and 11 supersectors), 3 transitions and the 3 regimes from 2003-05. 2 come from BLS's revision table (`cesnaicsrev`): total nonfarm, F→S and S→T, 1979-01 to 2003-04.
- **Identity.** For each month, the implied seasonal adjustment is the unadjusted (NSA) change less the SA change. Its change between two estimates is $`\Delta a`$. The SA revision $`r^S`$ equals the NSA revision $`r^N`$ less $`\Delta a`$. So within a group of months:

```math
\operatorname{Var}(r^S) = \operatorname{Var}(r^N) + \operatorname{Var}(\Delta a) - 2\operatorname{Cov}(r^N, \Delta a)
```

- **Identity gap.** `variance_identity_gap` is the float residual of that equation. The largest in the tables is 1.5e-11 in thousands squared, which is rounding error. The tables print it in scientific notation, because in fixed point it would read as 0.00.
- **The three shares.** Let $`V`$ be the variance of $`r^S`$ and $`C`$ the covariance of $`r^N`$ with $`\Delta a`$.
  - **Ratio share:** $`\operatorname{Var}(\Delta a) / V`$. It ignores the covariance.
  - **Adjustment share:** $`(\operatorname{Var}(\Delta a) - C) / V`$, which equals $`\operatorname{Cov}(r^S, -\Delta a) / V`$.
  - **NSA share:** $`(\operatorname{Var}(r^N) - C) / V`$, which equals $`\operatorname{Cov}(r^S, r^N) / V`$.
- **Why the two splits add to 1.** $`\operatorname{Var}(r^S) = \operatorname{Cov}(r^S, r^N) + \operatorname{Cov}(r^S, -\Delta a)`$, because $`r^S = r^N - \Delta a`$. So the adjustment share and the NSA share add to $`1 - \text{gap} / V`$, which is 1 to within rounding. The ratio share leaves out the covariance, so it does not partition $`V`$ and can exceed 1.
- **Units.** Variances and the covariance are in thousands squared. Shares have no unit.

## Results

### F→S

<!-- BEGIN GENERATED stage-3-f-s -->

| Sector | Source | Regime | Months | Var SA | Var NSA | Var adjustment change | Cov(NSA, adjustment change) | Ratio share | Adjustment share | NSA share | Identity gap |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00 Total nonfarm | cesnaicsrev | 1979-01 to 2003-04 | 291 | 4222.0 | 3795.9 | 304.2 | -60.9 | 0.07 | 0.09 | 0.91 | -1.8e-12 |
| 00 Total nonfarm | cesvinall | 2003-05 to 2019-12 | 200 | 1155.8 | 2897.2 | 1065.0 | 1403.2 | 0.92 | -0.29 | 1.29 | 2.3e-13 |
| 00 Total nonfarm | cesvinall | 2020-01 to 2022-12 | 36 | 8434.9 | 8296.0 | 3478.5 | 1669.8 | 0.41 | 0.21 | 0.79 | 0.0e+00 |
| 00 Total nonfarm | cesvinall | 2023-01 on | 34 | 2039.7 | 4610.4 | 1138.0 | 1854.4 | 0.56 | -0.35 | 1.35 | 2.3e-13 |
| 10 Mining and logging | cesvinall | 2003-05 to 2019-12 | 200 | 3.6 | 5.2 | 1.4 | 1.5 | 0.37 | -0.03 | 1.03 | 1.3e-15 |
| 10 Mining and logging | cesvinall | 2020-01 to 2022-12 | 36 | 10.4 | 13.9 | 1.5 | 2.5 | 0.15 | -0.10 | 1.10 | 1.8e-15 |
| 10 Mining and logging | cesvinall | 2023-01 on | 34 | 2.2 | 2.6 | 1.2 | 0.8 | 0.54 | 0.20 | 0.80 | 0.0e+00 |
| 20 Construction | cesvinall | 2003-05 to 2019-12 | 200 | 29.9 | 43.9 | 18.4 | 16.2 | 0.62 | 0.07 | 0.93 | 2.8e-14 |
| 20 Construction | cesvinall | 2020-01 to 2022-12 | 36 | 47.7 | 59.0 | 8.8 | 10.1 | 0.18 | -0.03 | 1.03 | 2.8e-14 |
| 20 Construction | cesvinall | 2023-01 on | 34 | 40.0 | 72.7 | 13.8 | 23.2 | 0.34 | -0.24 | 1.24 | -7.1e-15 |
| 30 Manufacturing | cesvinall | 2003-05 to 2019-12 | 200 | 54.6 | 70.0 | 12.9 | 14.2 | 0.24 | -0.02 | 1.02 | 7.1e-15 |
| 30 Manufacturing | cesvinall | 2020-01 to 2022-12 | 36 | 96.1 | 124.2 | 11.0 | 19.6 | 0.11 | -0.09 | 1.09 | 1.4e-14 |
| 30 Manufacturing | cesvinall | 2023-01 on | 34 | 21.9 | 47.0 | 8.8 | 16.9 | 0.40 | -0.37 | 1.37 | 7.1e-15 |
| 40 Trade, transportation, and utilities | cesvinall | 2003-05 to 2019-12 | 200 | 194.9 | 465.6 | 137.6 | 204.2 | 0.71 | -0.34 | 1.34 | -2.8e-14 |
| 40 Trade, transportation, and utilities | cesvinall | 2020-01 to 2022-12 | 36 | 2534.1 | 3499.6 | 427.2 | 696.3 | 0.17 | -0.11 | 1.11 | -4.5e-13 |
| 40 Trade, transportation, and utilities | cesvinall | 2023-01 on | 34 | 429.5 | 1371.7 | 654.9 | 798.6 | 1.52 | -0.33 | 1.33 | -1.1e-13 |
| 50 Information | cesvinall | 2003-05 to 2019-12 | 200 | 17.8 | 41.3 | 8.3 | 15.9 | 0.47 | -0.43 | 1.43 | -2.1e-14 |
| 50 Information | cesvinall | 2020-01 to 2022-12 | 36 | 95.5 | 102.4 | 10.0 | 8.4 | 0.10 | 0.02 | 0.98 | -2.8e-14 |
| 50 Information | cesvinall | 2023-01 on | 34 | 34.6 | 62.6 | 13.0 | 20.5 | 0.38 | -0.22 | 1.22 | -7.1e-15 |
| 55 Financial activities | cesvinall | 2003-05 to 2019-12 | 200 | 15.8 | 25.2 | 5.0 | 7.2 | 0.32 | -0.14 | 1.14 | 3.6e-15 |
| 55 Financial activities | cesvinall | 2020-01 to 2022-12 | 36 | 48.1 | 80.3 | 9.0 | 20.6 | 0.19 | -0.24 | 1.24 | 7.1e-15 |
| 55 Financial activities | cesvinall | 2023-01 on | 34 | 26.7 | 69.3 | 14.9 | 28.8 | 0.56 | -0.52 | 1.52 | -2.1e-14 |
| 60 Professional and business services | cesvinall | 2003-05 to 2019-12 | 200 | 167.1 | 312.1 | 82.6 | 113.8 | 0.49 | -0.19 | 1.19 | 2.3e-13 |
| 60 Professional and business services | cesvinall | 2020-01 to 2022-12 | 36 | 343.4 | 431.6 | 156.4 | 122.3 | 0.46 | 0.10 | 0.90 | -5.7e-14 |
| 60 Professional and business services | cesvinall | 2023-01 on | 34 | 237.4 | 570.9 | 200.3 | 266.9 | 0.84 | -0.28 | 1.28 | -2.0e-13 |
| 65 Education and health services | cesvinall | 2003-05 to 2019-12 | 200 | 97.9 | 231.6 | 59.4 | 96.5 | 0.61 | -0.38 | 1.38 | 7.1e-14 |
| 65 Education and health services | cesvinall | 2020-01 to 2022-12 | 36 | 264.1 | 335.9 | 126.8 | 99.3 | 0.48 | 0.10 | 0.90 | 2.3e-13 |
| 65 Education and health services | cesvinall | 2023-01 on | 34 | 82.4 | 167.2 | 90.1 | 87.5 | 1.09 | 0.03 | 0.97 | -8.5e-14 |
| 70 Leisure and hospitality | cesvinall | 2003-05 to 2019-12 | 200 | 96.8 | 194.7 | 59.1 | 78.5 | 0.61 | -0.20 | 1.20 | 5.7e-14 |
| 70 Leisure and hospitality | cesvinall | 2020-01 to 2022-12 | 36 | 2494.6 | 2390.8 | 574.6 | 235.4 | 0.23 | 0.14 | 0.86 | 0.0e+00 |
| 70 Leisure and hospitality | cesvinall | 2023-01 on | 34 | 187.4 | 273.6 | 73.7 | 80.0 | 0.39 | -0.03 | 1.03 | 5.7e-14 |
| 80 Other services | cesvinall | 2003-05 to 2019-12 | 200 | 13.2 | 23.7 | 4.3 | 7.4 | 0.33 | -0.23 | 1.23 | -2.3e-14 |
| 80 Other services | cesvinall | 2020-01 to 2022-12 | 36 | 49.7 | 60.9 | 16.6 | 13.9 | 0.33 | 0.05 | 0.95 | 0.0e+00 |
| 80 Other services | cesvinall | 2023-01 on | 34 | 13.1 | 35.0 | 8.1 | 15.0 | 0.62 | -0.53 | 1.53 | -3.6e-15 |
| 90 Government | cesvinall | 2003-05 to 2019-12 | 200 | 257.1 | 977.9 | 318.0 | 519.4 | 1.24 | -0.78 | 1.78 | -4.0e-13 |
| 90 Government | cesvinall | 2020-01 to 2022-12 | 36 | 1096.0 | 1611.5 | 501.1 | 508.3 | 0.46 | -0.01 | 1.01 | 0.0e+00 |
| 90 Government | cesvinall | 2023-01 on | 34 | 382.3 | 1585.8 | 657.5 | 930.5 | 1.72 | -0.71 | 1.71 | 2.8e-13 |

<!-- END GENERATED stage-3-f-s -->

### S→T

<!-- BEGIN GENERATED stage-3-s-t -->

| Sector | Source | Regime | Months | Var SA | Var NSA | Var adjustment change | Cov(NSA, adjustment change) | Ratio share | Adjustment share | NSA share | Identity gap |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00 Total nonfarm | cesnaicsrev | 1979-01 to 2003-04 | 290 | 1573.5 | 15962.6 | 15631.5 | 15010.3 | 9.93 | 0.39 | 0.61 | -1.5e-11 |
| 00 Total nonfarm | cesvinall | 2003-05 to 2019-12 | 200 | 1136.1 | 436.6 | 878.7 | 89.6 | 0.77 | 0.69 | 0.31 | 2.3e-13 |
| 00 Total nonfarm | cesvinall | 2020-01 to 2022-12 | 36 | 14319.0 | 8192.2 | 5468.1 | -329.3 | 0.38 | 0.40 | 0.60 | 3.6e-12 |
| 00 Total nonfarm | cesvinall | 2023-01 on | 34 | 2142.9 | 277.5 | 1757.0 | -54.2 | 0.82 | 0.85 | 0.15 | 4.5e-13 |
| 10 Mining and logging | cesvinall | 2003-05 to 2019-12 | 200 | 3.1 | 1.6 | 1.9 | 0.2 | 0.60 | 0.54 | 0.46 | -4.4e-16 |
| 10 Mining and logging | cesvinall | 2020-01 to 2022-12 | 36 | 5.4 | 3.5 | 2.1 | 0.1 | 0.38 | 0.36 | 0.64 | -8.9e-16 |
| 10 Mining and logging | cesvinall | 2023-01 on | 34 | 1.0 | 0.6 | 0.9 | 0.3 | 0.94 | 0.66 | 0.34 | 3.3e-16 |
| 20 Construction | cesvinall | 2003-05 to 2019-12 | 200 | 63.6 | 21.4 | 44.8 | 1.3 | 0.70 | 0.68 | 0.32 | 1.1e-13 |
| 20 Construction | cesvinall | 2020-01 to 2022-12 | 36 | 75.8 | 55.5 | 20.2 | -0.1 | 0.27 | 0.27 | 0.73 | 1.4e-14 |
| 20 Construction | cesvinall | 2023-01 on | 34 | 25.3 | 9.3 | 12.9 | -1.5 | 0.51 | 0.57 | 0.43 | 3.6e-15 |
| 30 Manufacturing | cesvinall | 2003-05 to 2019-12 | 200 | 41.5 | 16.9 | 28.4 | 2.0 | 0.69 | 0.64 | 0.36 | 1.4e-14 |
| 30 Manufacturing | cesvinall | 2020-01 to 2022-12 | 36 | 76.7 | 70.3 | 24.1 | 8.9 | 0.31 | 0.20 | 0.80 | 1.4e-14 |
| 30 Manufacturing | cesvinall | 2023-01 on | 34 | 14.1 | 17.7 | 6.5 | 5.0 | 0.46 | 0.11 | 0.89 | 0.0e+00 |
| 40 Trade, transportation, and utilities | cesvinall | 2003-05 to 2019-12 | 200 | 123.0 | 81.1 | 101.9 | 30.0 | 0.83 | 0.58 | 0.42 | -8.5e-14 |
| 40 Trade, transportation, and utilities | cesvinall | 2020-01 to 2022-12 | 36 | 991.0 | 861.9 | 509.3 | 190.1 | 0.51 | 0.32 | 0.68 | -5.7e-13 |
| 40 Trade, transportation, and utilities | cesvinall | 2023-01 on | 34 | 285.6 | 49.1 | 253.2 | 8.4 | 0.89 | 0.86 | 0.14 | -5.7e-14 |
| 50 Information | cesvinall | 2003-05 to 2019-12 | 200 | 9.9 | 10.9 | 5.1 | 3.0 | 0.51 | 0.21 | 0.79 | 0.0e+00 |
| 50 Information | cesvinall | 2020-01 to 2022-12 | 36 | 27.3 | 20.4 | 10.3 | 1.7 | 0.38 | 0.32 | 0.68 | 0.0e+00 |
| 50 Information | cesvinall | 2023-01 on | 34 | 8.9 | 5.5 | 5.1 | 0.9 | 0.58 | 0.48 | 0.52 | -1.8e-15 |
| 55 Financial activities | cesvinall | 2003-05 to 2019-12 | 200 | 14.1 | 8.7 | 8.3 | 1.5 | 0.59 | 0.49 | 0.51 | -1.8e-15 |
| 55 Financial activities | cesvinall | 2020-01 to 2022-12 | 36 | 29.5 | 20.4 | 8.3 | -0.4 | 0.28 | 0.30 | 0.70 | 3.6e-15 |
| 55 Financial activities | cesvinall | 2023-01 on | 34 | 10.8 | 9.0 | 4.7 | 1.5 | 0.43 | 0.30 | 0.70 | 1.8e-15 |
| 60 Professional and business services | cesvinall | 2003-05 to 2019-12 | 200 | 138.2 | 117.1 | 89.7 | 34.3 | 0.65 | 0.40 | 0.60 | 1.4e-13 |
| 60 Professional and business services | cesvinall | 2020-01 to 2022-12 | 36 | 273.3 | 61.6 | 339.9 | 64.1 | 1.24 | 1.01 | -0.01 | -5.7e-14 |
| 60 Professional and business services | cesvinall | 2023-01 on | 34 | 432.9 | 93.9 | 357.4 | 9.2 | 0.83 | 0.80 | 0.20 | 0.0e+00 |
| 65 Education and health services | cesvinall | 2003-05 to 2019-12 | 200 | 58.7 | 32.3 | 41.0 | 7.3 | 0.70 | 0.57 | 0.43 | 2.8e-14 |
| 65 Education and health services | cesvinall | 2020-01 to 2022-12 | 36 | 344.4 | 235.1 | 108.8 | -0.2 | 0.32 | 0.32 | 0.68 | 0.0e+00 |
| 65 Education and health services | cesvinall | 2023-01 on | 34 | 77.0 | 57.4 | 40.3 | 10.4 | 0.52 | 0.39 | 0.61 | 0.0e+00 |
| 70 Leisure and hospitality | cesvinall | 2003-05 to 2019-12 | 200 | 106.8 | 57.8 | 71.8 | 11.4 | 0.67 | 0.57 | 0.43 | -4.3e-14 |
| 70 Leisure and hospitality | cesvinall | 2020-01 to 2022-12 | 36 | 2873.5 | 2035.2 | 597.8 | -120.2 | 0.21 | 0.25 | 0.75 | 0.0e+00 |
| 70 Leisure and hospitality | cesvinall | 2023-01 on | 34 | 207.8 | 31.4 | 182.8 | 3.2 | 0.88 | 0.86 | 0.14 | 2.8e-14 |
| 80 Other services | cesvinall | 2003-05 to 2019-12 | 200 | 8.6 | 9.0 | 6.3 | 3.3 | 0.73 | 0.34 | 0.66 | -1.1e-14 |
| 80 Other services | cesvinall | 2020-01 to 2022-12 | 36 | 121.8 | 102.6 | 17.5 | -0.8 | 0.14 | 0.15 | 0.85 | -1.4e-14 |
| 80 Other services | cesvinall | 2023-01 on | 34 | 6.1 | 5.3 | 3.8 | 1.5 | 0.63 | 0.38 | 0.62 | 8.9e-16 |
| 90 Government | cesvinall | 2003-05 to 2019-12 | 200 | 182.7 | 128.1 | 120.6 | 33.0 | 0.66 | 0.48 | 0.52 | -5.7e-14 |
| 90 Government | cesvinall | 2020-01 to 2022-12 | 36 | 444.6 | 142.7 | 349.7 | 23.9 | 0.79 | 0.73 | 0.27 | -5.7e-14 |
| 90 Government | cesvinall | 2023-01 on | 34 | 570.9 | 50.4 | 478.1 | -21.2 | 0.84 | 0.87 | 0.13 | 1.1e-13 |

<!-- END GENERATED stage-3-s-t -->

### T→B

<!-- BEGIN GENERATED stage-3-t-b -->

| Sector | Source | Regime | Months | Var SA | Var NSA | Var adjustment change | Cov(NSA, adjustment change) | Ratio share | Adjustment share | NSA share | Identity gap |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00 Total nonfarm | cesvinall | 2003-05 to 2019-12 | 200 | 1793.8 | 877.7 | 1172.1 | 128.0 | 0.65 | 0.58 | 0.42 | 1.4e-12 |
| 00 Total nonfarm | cesvinall | 2020-01 to 2022-12 | 36 | 21903.7 | 856.8 | 20437.2 | -304.9 | 0.93 | 0.95 | 0.05 | -7.3e-12 |
| 00 Total nonfarm | cesvinall | 2023-01 on | 34 | 3412.3 | 1295.9 | 1761.1 | -177.7 | 0.52 | 0.57 | 0.43 | -4.5e-13 |
| 10 Mining and logging | cesvinall | 2003-05 to 2019-12 | 200 | 2.3 | 1.1 | 1.9 | 0.4 | 0.81 | 0.66 | 0.34 | -1.3e-15 |
| 10 Mining and logging | cesvinall | 2020-01 to 2022-12 | 36 | 8.0 | 3.3 | 5.7 | 0.5 | 0.71 | 0.65 | 0.35 | 1.8e-15 |
| 10 Mining and logging | cesvinall | 2023-01 on | 34 | 2.2 | 0.7 | 3.2 | 0.8 | 1.43 | 1.07 | -0.07 | -8.9e-16 |
| 20 Construction | cesvinall | 2003-05 to 2019-12 | 200 | 75.2 | 27.2 | 58.1 | 5.1 | 0.77 | 0.71 | 0.29 | -4.3e-14 |
| 20 Construction | cesvinall | 2020-01 to 2022-12 | 36 | 93.5 | 20.0 | 97.4 | 11.9 | 1.04 | 0.91 | 0.09 | -2.8e-14 |
| 20 Construction | cesvinall | 2023-01 on | 34 | 33.7 | 11.8 | 23.9 | 1.0 | 0.71 | 0.68 | 0.32 | -7.1e-15 |
| 30 Manufacturing | cesvinall | 2003-05 to 2019-12 | 200 | 39.2 | 14.6 | 30.8 | 3.1 | 0.79 | 0.71 | 0.29 | -7.1e-15 |
| 30 Manufacturing | cesvinall | 2020-01 to 2022-12 | 36 | 80.8 | 16.9 | 82.9 | 9.5 | 1.03 | 0.91 | 0.09 | 0.0e+00 |
| 30 Manufacturing | cesvinall | 2023-01 on | 34 | 33.6 | 14.3 | 15.3 | -2.0 | 0.45 | 0.51 | 0.49 | 2.1e-14 |
| 40 Trade, transportation, and utilities | cesvinall | 2003-05 to 2019-12 | 200 | 206.9 | 62.9 | 153.9 | 4.9 | 0.74 | 0.72 | 0.28 | -8.5e-14 |
| 40 Trade, transportation, and utilities | cesvinall | 2020-01 to 2022-12 | 36 | 1883.7 | 138.3 | 1708.9 | -18.2 | 0.91 | 0.92 | 0.08 | 0.0e+00 |
| 40 Trade, transportation, and utilities | cesvinall | 2023-01 on | 34 | 633.7 | 108.0 | 465.3 | -30.2 | 0.73 | 0.78 | 0.22 | -1.1e-13 |
| 50 Information | cesvinall | 2003-05 to 2019-12 | 200 | 13.4 | 8.5 | 7.1 | 1.2 | 0.53 | 0.45 | 0.55 | 0.0e+00 |
| 50 Information | cesvinall | 2020-01 to 2022-12 | 36 | 35.6 | 16.4 | 25.0 | 2.9 | 0.70 | 0.62 | 0.38 | 0.0e+00 |
| 50 Information | cesvinall | 2023-01 on | 34 | 36.9 | 6.3 | 28.5 | -1.1 | 0.77 | 0.80 | 0.20 | 0.0e+00 |
| 55 Financial activities | cesvinall | 2003-05 to 2019-12 | 200 | 22.5 | 18.3 | 10.7 | 3.2 | 0.47 | 0.33 | 0.67 | 2.5e-14 |
| 55 Financial activities | cesvinall | 2020-01 to 2022-12 | 36 | 24.0 | 16.9 | 18.3 | 5.6 | 0.76 | 0.53 | 0.47 | 3.6e-15 |
| 55 Financial activities | cesvinall | 2023-01 on | 34 | 17.7 | 13.4 | 8.5 | 2.1 | 0.48 | 0.36 | 0.64 | 3.6e-15 |
| 60 Professional and business services | cesvinall | 2003-05 to 2019-12 | 200 | 166.7 | 91.8 | 117.7 | 21.4 | 0.71 | 0.58 | 0.42 | 2.8e-14 |
| 60 Professional and business services | cesvinall | 2020-01 to 2022-12 | 36 | 1102.5 | 175.5 | 707.7 | -109.6 | 0.64 | 0.74 | 0.26 | 2.3e-13 |
| 60 Professional and business services | cesvinall | 2023-01 on | 34 | 367.2 | 159.4 | 225.6 | 8.9 | 0.61 | 0.59 | 0.41 | 5.7e-14 |
| 65 Education and health services | cesvinall | 2003-05 to 2019-12 | 200 | 71.9 | 28.6 | 55.5 | 6.1 | 0.77 | 0.69 | 0.31 | 5.7e-14 |
| 65 Education and health services | cesvinall | 2020-01 to 2022-12 | 36 | 458.0 | 101.9 | 394.7 | 19.3 | 0.86 | 0.82 | 0.18 | -1.1e-13 |
| 65 Education and health services | cesvinall | 2023-01 on | 34 | 64.6 | 106.2 | 114.3 | 78.0 | 1.77 | 0.56 | 0.44 | 4.3e-14 |
| 70 Leisure and hospitality | cesvinall | 2003-05 to 2019-12 | 200 | 135.7 | 64.7 | 105.0 | 17.0 | 0.77 | 0.65 | 0.35 | -5.7e-14 |
| 70 Leisure and hospitality | cesvinall | 2020-01 to 2022-12 | 36 | 5498.9 | 961.0 | 5000.3 | 231.2 | 0.91 | 0.87 | 0.13 | 2.7e-12 |
| 70 Leisure and hospitality | cesvinall | 2023-01 on | 34 | 437.9 | 101.2 | 352.3 | 7.8 | 0.80 | 0.79 | 0.21 | 0.0e+00 |
| 80 Other services | cesvinall | 2003-05 to 2019-12 | 200 | 11.7 | 7.4 | 5.3 | 0.5 | 0.45 | 0.41 | 0.59 | 3.6e-15 |
| 80 Other services | cesvinall | 2020-01 to 2022-12 | 36 | 133.8 | 86.9 | 106.3 | 29.7 | 0.79 | 0.57 | 0.43 | 2.8e-14 |
| 80 Other services | cesvinall | 2023-01 on | 34 | 17.9 | 11.5 | 7.0 | 0.3 | 0.39 | 0.38 | 0.62 | 0.0e+00 |
| 90 Government | cesvinall | 2003-05 to 2019-12 | 200 | 158.8 | 16.8 | 166.0 | 12.0 | 1.05 | 0.97 | 0.03 | 8.5e-14 |
| 90 Government | cesvinall | 2020-01 to 2022-12 | 36 | 2652.2 | 131.5 | 2190.4 | -165.1 | 0.83 | 0.89 | 0.11 | -1.8e-12 |
| 90 Government | cesvinall | 2023-01 on | 34 | 316.1 | 15.6 | 323.4 | 11.5 | 1.02 | 0.99 | 0.01 | 5.7e-14 |

<!-- END GENERATED stage-3-t-b -->

### Shares outside [0, 1]

<!-- BEGIN GENERATED stage-3-bounds -->

| Share | Transition | Rows | Below 0 | Above 1 | Min | Max |
|---|---|---:|---:|---:|---:|---:|
| Ratio | F→S | 37 | 0 | 4 | 0.07 | 1.72 |
| Ratio | S→T | 37 | 0 | 2 | 0.14 | 9.93 |
| Ratio | T→B | 36 | 0 | 6 | 0.39 | 1.77 |
| Ratio | All | 110 | 0 | 12 | 0.07 | 9.93 |
| Adjustment | F→S | 37 | 27 | 0 | -0.78 | 0.21 |
| Adjustment | S→T | 37 | 0 | 1 | 0.11 | 1.01 |
| Adjustment | T→B | 36 | 0 | 1 | 0.33 | 1.07 |
| Adjustment | All | 110 | 27 | 2 | -0.78 | 1.07 |
| NSA | F→S | 37 | 0 | 27 | 0.79 | 1.78 |
| NSA | S→T | 37 | 1 | 0 | -0.01 | 0.89 |
| NSA | T→B | 36 | 1 | 0 | -0.07 | 0.67 |
| NSA | All | 110 | 2 | 27 | -0.07 | 1.78 |

<!-- END GENERATED stage-3-bounds -->

## Limits

- **No interval.** The shares are descriptive. The decomposition table holds group variances, not the month-level terms that an interval needs.
- **Shares outside [0, 1].** The tables show them as computed. No row is clipped, dropped or flagged. The 9.93 ratio is total nonfarm S→T before 2003: the NSA revision and the adjustment change nearly cancel there, so the SA variance is small beside both.
- **Negative F→S splits.** In 27 F→S rows, the adjustment share is below 0 and the NSA share is above 1. There the adjustment change moves with the NSA revision. It offsets part of the NSA revision instead of adding variance. This is not an error.
- **The benchmark release.** The December F→S, the November S→T and every T→B take their later value from the January benchmark release, which also revises the seasonal factors. So part of those shares comes from the benchmark, not from the monthly adjustment (compare the [Stage 1 findings](ces-revision-drivers-stage-1-findings.md), Benchmark effects).
- **Short regimes.** The 2020 to 2022 regime has 36 months and the 2023 on regime has 34. A share from so few months is noisy, and it carries no interval.
- **Rounding.** The inputs are whole thousands. This matters most where the SA variance is small. The smallest is about 1.0, for mining and logging S→T in 2023 on.
- **Before 2003.** Only total nonfarm has rows, from the revision table, and there is no T→B row.

## Measured cost

- **Wall time of the real run** (on this Mac, from a worktree with no `data/panel/`, after `uv sync`): `scripts/vintage_sources.py build` took 32.94 s and `scripts/seasonal_share.py report` 1.38 s, measured with `/usr/bin/time -p`.
- **Sessions:** 2, the session that wrote plan 16 (PR 27) and the session that executed it, both on 2026-11-18.
