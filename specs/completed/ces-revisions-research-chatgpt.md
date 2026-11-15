# The changing scale of U.S. payroll-employment revisions

## Critical literature review, gap analysis, and a Bayesian multi-vintage research design

**Research cutoff:** September 12, 2026\
**Scope:** National Current Employment Statistics (CES) all-employees estimates, with supersector detail where it identifies mechanisms. Numbers are thousands of jobs unless otherwise stated.

## Executive summary

The evidence does **not** support a simple story that CES revisions have steadily become larger. In BLS's own total-nonfarm table, the mean absolute revision (MAR) to the seasonally adjusted monthly change fell after the May 2003 production break from 48,000 to 33,000 at first-to-second release and from 61,000 to 51,000 at first-to-third release. Yet the second-to-third MAR rose from 29,000 to 34,000, and 2020–21 produced exceptional tails. These comparisons are descriptive: May 2003 simultaneously introduced the probability sample, NAICS, concurrent seasonal adjustment, and other changes.

The four stages have different causes. The first two revisions mostly add late sample receipts and recalculate concurrent seasonal factors. The annual benchmark instead reanchors March not-seasonally-adjusted employment to QCEW-dominated universe counts; net birth–death and other estimation errors are embedded in that gap. Wedge-back then imposes a highly correlated, nearly rank-one revision across April–February, while post-March months are rebuilt from the new level with old sample links and updated birth–death estimates. Annual re-seasonalization can change five years of seasonally adjusted history. Consequently, a single “revision” series mixes incompatible estimands.

There is credible CES evidence that shorter collection intervals raise late-reporting rates, and weak, old evidence that late reporting is associated with absolute revision. There is no long-run published decomposition of total seasonally adjusted revisions into an NSA sample component and a seasonal-factor component. Nor did this search find a study linking BLS appropriations or staffing to CES revision scale. Funding, FTE, and sample size are slow-moving and collinear; party control is even less credible as a causal regressor.

The proposed study jointly models NSA and SA vintages by sector. A latent concept-consistent employment path is observed through stage-specific news and noise equations; a higher-precision but fallible March benchmark enters irregularly; the published wedge is an observation operator, not twelve independent shocks. Student-*t* measurement errors have stochastic log-scale driven by collection days/rates, operational capacity, design regimes, and pandemic indicators. Hierarchical shrinkage pools driver effects across supersectors. This design can estimate changing revision dispersion and tail risk while propagating uncertainty about “truth”; causal claims for annual capacity covariates should remain explicitly limited.

## 1. Scope, definitions, and evidentiary standard

### 1.1 Four noninterchangeable revision stages

Let $`Y_{s,t\mid v}`$ be the published employment **level** for sector $`s`$, reference month $`t`$, in release vintage $`v`$. Let $`v_F(t),v_S(t),v_T(t)`$ denote the release dates on which month $`t`$ is first, second, and third (final sample-based), and define the over-the-month change actually visible in each release as

```math
d^j_{s,t}=Y_{s,t\mid v_j(t)}-Y_{s,t-1\mid v_j(t)},\qquad j\in\{F,S,T\}.
```

The scheduled-revision estimands in the official table are therefore

```math
r^{FS}_{s,t}=d^S_{s,t}-d^F_{s,t},\qquad
r^{ST}_{s,t}=d^T_{s,t}-d^S_{s,t},
```

not merely changes in the level for month $`t`$. In an ordinary release, for example, $`r^{FS}_t=(Y_{t\mid v_S}-Y_{t\mid v_F})-(Y_{t-1\mid v_S}-Y_{t-1\mid v_F})`$; the prior-month term changes too. Let $`Y^{B}_{s,t}`$ denote the first comprehensive series incorporating the final March-reference benchmark and $`Y^{P}_{s,t\mid v}`$ later post-benchmark vintages. The benchmark/post-benchmark level estimands are $`r^{TB}_{s,t}=Y^{B}_{s,t}-Y_{s,t\mid v_T(t)}`$ and $`r^{BP}_{s,t\mid v}=Y^{P}_{s,t\mid v}-Y^{B}_{s,t}`$, with their published-change analogues obtained by differencing within a single vintage.

The principal estimands are scales—standard deviations, MARs, or Student-*t* scale parameters—of these objects. For monthly releases the policy-relevant object is usually the revision to the **over-the-month change**, not the level. Benchmark publications usually report the March **level** discrepancy and a percentage of employment. Post-benchmark revisions span both levels and changes. Comparisons below preserve those units rather than manufacture a common metric.

One timing convention avoids a false fifth source of information: the March anchor and the April–February wedge appear in the **same comprehensive benchmark vintage**. The information arrival is `T→B` at March; the wedge is its deterministic backward propagation and is reported separately under stage 4 because the user-facing revision vector is different. `B→P/M` denotes genuinely later rebenchmarking, re-seasonalization, reconstruction, and post-March maturation. The model therefore treats the wedge as an operator, not a new stochastic shock.

The evidence labels used in this review are:

- **Documented:** a production rule, administrative fact, or statistic directly reported by BLS or another primary source.
- **Supported:** an empirical association or experimental comparison with an explicit design.
- **Asserted:** a plausible mechanism stated without an identifying analysis of revision magnitude.
- **Contested:** competing evidence or an estimand that does not support the claimed interpretation.
- **Not found:** no directly responsive study was located in the searched agency, journal, working-paper, and grey-literature sources; this is not proof of nonexistence.

### 1.2 Magnitude at the two sample-based stages

The longest internally consistent official summary is BLS's 1979–present table. It excludes later benchmark, re-seasonalization, and reconstruction effects, and therefore answers only stages 1 and 2.

| Period | SA MAR: second − first | SA MAR: third − second | SA MAR: third − first | NSA MAR: second − first | NSA MAR: third − second | NSA MAR: third − first |
|---|---:|---:|---:|---:|---:|---:|
| Jan. 1979–early 2003 | 48 | 29 | 61 | 46 | 53 | 83 |
| May 2003–present | 33 | 34 | 51 | 46 | 18 | 53 |
| Full published span | 41 | 31 | 57 | 46 | 36 | 68 |

Source: [BLS, “Nonfarm Payroll Employment: Revisions between over-the-month estimates, 1979–present”](https://www.bls.gov/web/empsit/cesnaicsrev.htm), updated September 4, 2026. The break is BLS's production-method split; missing 2003 observations differ by comparison.

Three facts matter. First, the post-2003 first-to-third SA MAR is about 16% lower, but this is a two-regime average, not evidence of a secular trend. Second, SA and NSA stage patterns differ: the post-2003 second-to-third NSA MAR fell sharply while its SA counterpart rose slightly. That is exactly why subtracting an NSA MAR from an SA MAR is not a decomposition; absolute values discard covariance and concurrent factors react to revised NSA inputs. Third, tails are regime dependent. The 2020 and 2021 first-to-third SA MARs were 130,000 and 181,000, respectively, versus 34,000 in 2019 and 28,000 in 2022. Recent annual MARs (51,000 in 2023, 48,000 in 2024, and 58,000 in 2025) show neither a return to the 2020–21 tail nor a monotone decline.

Aggregating BLS's annual rows into economically meaningful periods makes that nonmonotonicity clearer. These are reproducible arithmetic from the official annual table, not separately published BLS estimates.

| Period | SA 2−1 | SA 3−2 | SA 3−1 | NSA 2−1 | NSA 3−2 | NSA 3−1 |
|---|---:|---:|---:|---:|---:|---:|
| 2004–07 | 28.3 | 26.3 | 40.3 | 44.3 | 13.8 | 50.8 |
| 2008–09 | 33.5 | 44.5 | 64.0 | 46.5 | 24.5 | 46.0 |
| 2010–14 | 26.4 | 26.2 | 41.4 | 40.8 | 13.4 | 43.6 |
| 2015–19 | 22.4 | 20.2 | 29.0 | 38.2 | 14.0 | 41.6 |
| 2020–21 | 89.5 | 83.0 | 155.5 | 79.5 | 49.5 | 116.0 |
| 2022–25 | 30.5 | 40.3 | 46.3 | 48.5 | 13.8 | 52.3 |

The post-2022 SA 3−2 average remains twice the 2015–19 value while NSA 3−2 does not. That is a useful seasonal-factor diagnostic, not proof of a factor share.

### 1.3 Which source can enter which stage?

| Revision source | First → second | Second → third | Third → annual benchmark | Post-benchmark / wedge-back |
|---|---|---|---|---|
| Late sample receipts / collection completeness | Direct, principal NSA input | Direct, principal NSA input | Only through errors remaining in the third estimate | New late receipts matter for the most recent months; historical sample links are largely reused |
| Concurrent seasonal-factor revision | Direct for SA; absent from NSA | Direct for SA; absent from NSA | Annual re-seasonalization can revise five years of SA data | Later annual factor/model changes can revise SA history again |
| Net birth–death model | Ordinary monthly forecast is held fixed across closings, so no direct NSA contribution; flag exceptional corrections | Same | Forecast error is embedded in the CES–universe gap; sometimes substantively dominant | Revised forecasts/adjustments are applied after reanchoring; from 2025 the April–October post-benchmark method became more current-sensitive |
| QCEW/universe realignment | None | None | Defining operation, but the universe count is itself measured and revised | The March discrepancy is distributed backward; the new level propagates forward |
| NAICS, sample redesign, reweighting, reconstruction | Usually no routine role | Annual sample rotation can affect a release comparison when introduced | Benchmark/sample update and classification conversions may alter sector detail | Reconstructions can revise history beyond ordinary windows |

This crosswalk follows the [CES Handbook calculation chapter](https://www.bls.gov/opub/hom/ces/calculation.htm), [CES revision FAQ](https://www.bls.gov/web/empsit/cesfaq.htm), and [vintage-data documentation](https://www.bls.gov/web/empsit/cesvininfo.htm). It also shows why no single covariate should be forced to explain every stage.

## 2. Literature review by candidate driver

### 2.1 Seasonality

#### What is known

**The 2003 method comparison is unusually informative, but the realized break is not.** Before May 2003 CES projected seasonal factors for six months and updated them semiannually. BLS ran a controlled parallel exercise in which NSA inputs and model choices were held common. For March 1998–March 2001, concurrent adjustment reduced the total-nonfarm first-release-to-final-benchmarked MAR in monthly change from 77,973 to 64,973 (−13,000). For March 1998–March 2002 it reduced the first-to-second MAR from 37,000 to 34,000 and the first-to-third MAR from 48,000 to 36,000. Eight of nine major industry divisions improved in the first comparison. This is evidence for the **seasonal algorithm**, not for the composite May 2003 production break ([Kropf, Manning, Mueller, and Scott 2002](https://www.bls.gov/osmr/research-papers/2002/pdf/st020110.pdf)).

The realized May 2003 break simultaneously changed four consequential features: SIC to NAICS, the quota/cutoff sample to the probability sample, projected to concurrent seasonal adjustment, and treatment of federal employment. BLS itself describes the published history as reflecting all four plus the regular benchmark ([BLS 2003 notice](https://www.bls.gov/opub/ted/2003/jun/wk2/art03.htm)). The pre/post MAR table therefore cannot attribute its lower first-to-third scale to concurrent adjustment.

**X-12 to X-13ARIMA-SEATS is not evidence that CES adopted SEATS decomposition.** CES moved production to X-13ARIMA-SEATS in 2015, added automatic regARIMA model selection using TRAMO in 2017, and continues to describe the decomposition as X-11 within the X-13 suite. The upgrade expanded diagnostics, regressors, and model-selection capability; no CES study located here estimates its effect on payroll-revision scale ([CES history](https://www.bls.gov/opub/hom/ces/history.htm); [seasonal-adjustment technical notes](https://www.bls.gov/web/empsit/cesseasadjtn.htm)). Treating a software-name change as a measured revision break would be an unsupported assertion.

**Calendar adjustments mostly target industry series and hours/earnings, with weak evidence for total-nonfarm revision scale.** CES's reference pay period always includes the 12th, but adjacent reference intervals contain four or five weeks. BLS regARIMA research evaluated interval regressors on fit, smoothness, stability, and revision behavior, leading to production changes beginning in 1996 ([Cano, Getz, Kropf, Scott, and Stamas 1996](https://www.bls.gov/osmr/research-papers/1996/st960190.htm)). The evidence is mixed rather than uniformly favorable: 68 series preferred an interval adjustment, but for the total-employment level the adjusted series had slightly larger median and 85th-percentile revisions (0.014% and 0.031%) than the unadjusted series (0.012% and 0.024%); construction was excluded after worse smoothness. Differences were generally below 0.01 percentage point. Labor Day, Easter/religious-holiday, and weather/outlier adjustments appear in annual technical material. Their clearest effects are in construction and in hours/earnings; this search found no credible estimate of their separate contribution to total-nonfarm stage-1 or stage-2 variance.

**COVID was a genuine seasonal-identification failure mode.** For the 2020 annual run, BLS split pre- and post-pandemic processing and used additive outliers so the collapse would not contaminate earlier seasonal patterns. For total nonfarm, the March 2020 “normal” seasonal movement rose from 360,000 under the old model to 474,000 in the ordinary annual run and 670,000 under the intervention specification. For the 2021 benchmark BLS standardized additive-outlier, level-shift, and temporary-change selection; 68% of all-employees series preferred one of those intervention types by AICc (571 of 834 series). BLS reports more stable adjustment and smaller revisions under the revised treatment, but does not decompose total monthly revisions into sample and factor contributions ([Hudson, Mercurio, and Kropf 2022](https://www.bls.gov/opub/mlr/2022/article/the-challenges-of-seasonal-adjustment-for-the-current-employment-statistics-survey-during-the-covid-19-pandemic.htm)). This supports a 2020–22 scale/tail regime, not a permanent post-COVID coefficient.

The March 2021 benchmark is a vivid nondecomposition example: the NSA total-nonfarm benchmark revision was only −7,000 while the corresponding SA revision was +374,000; in 2020 they were −121,000 and −250,000 ([BLS 2022 explanation](https://www.bls.gov/blog/2022/what-s-going-on-with-the-large-revisions-in-seasonally-adjusted-employment-estimates.htm)). Those differences show that annual seasonal recomputation can dominate the SA revision even when the March NSA anchor barely moves, but they do not quantify the seasonal share of routine first→second or second→third revisions.

#### The sample-versus-factor decomposition

Wright's real-time analysis is the closest published decomposition. Comparing first releases with a May 2013 vintage, he reports standard deviations of revisions to monthly change of 93,000 for NSA payrolls, 111,000 for SA payrolls, and 81,000 for the implied seasonal component (NSA minus SA), with a 0.19 correlation between that implied factor revision and the NSA revision. He identifies revised NSA inputs, asymmetric end filters/forecast extension, and movement of the ten-year estimation window as distinct channels. His recursive correction reduced an RMSPE from 71,100 to 67,300, but not significantly ([Wright 2013](https://doi.org/10.1353/eca.2013.0017)).

That exercise is important but not a clean production-accounting identity. The “factor” proxy inherits revised NSA data, benchmark revisions, changing windows, and specification changes; detailed real-time NSA inputs and unrounded components were unavailable. Loya's BLS seam-effect study likewise emphasizes that changing NSA vintages obstruct exact isolation. Excluding benchmark-release Januaries, it finds a total-nonfarm mean absolute fourth-versus-third over-the-month seam effect of only 9,626 jobs over January 2005–September 2012—7.6% of the September 2012 monthly change—but calls for constant-vintage NSA reconstruction ([Loya 2014](https://www.bls.gov/osmr/research-papers/2014/pdf/st140090.pdf)).

The substantive interpretation is contested. Wright argues that endpoint behavior around the Great Recession displaced more than 100,000 jobs between first and second halves of some years. Tiller and Evans simulate CES-like series and find X-11 reasonably robust to continuous recession ramps, with serious distortion mainly under discontinuous level shifts that intervention variables can mitigate ([Tiller and Evans 2014](https://www.bls.gov/osmr/research-papers/2014/pdf/st140180.pdf)). Because the counterfactual seasonal is unobserved, neither establishes the realized share of total revisions caused by seasonal adjustment.

#### Claims audit by revision stage

- **First → second and second → third:** documented inputs are additional reports and recomputed concurrent factors. Controlled pre-2003 experiments support modestly smaller MAR under concurrency. **Not found:** a long-span, constant-NSA-vintage accounting decomposition, by stage and sector.
- **Third → benchmark:** NSA benchmark realignment and annual SA recomputation occur together in published SA vintages; five years may change. Claims that a large SA benchmark revision is “seasonal” require a counterfactual NSA run and are otherwise unsubstantiated.
- **Post-benchmark:** subsequent annual re-seasonalizations produce historical changes even after NSA is fixed. Their scale is demonstrably nonzero but small in Loya's limited seam estimand; it is not the same as the full post-benchmark revision.
- **Contested:** simple SA-minus-NSA absolute revisions are not additive contributions. The covariance between sample revision and factor response must be estimated.

### 2.2 Collection interval and nonresponse

#### Calendar-generated variation and the three closings

BLS starts collecting a month's establishment data on the first business day after the 12th and continues through the Monday before the Employment Situation release. The historical first window therefore varies from roughly 9 to 16 collection days with the weekday of the 12th, holidays, the scheduled release date, and operative closing rules. Collection continues for the second and third releases. In 2024, average collection rates were 60.4%, 89.0%, and 90.9% at the first, second, and third releases, respectively ([GAO 2026](https://files.gao.gov/reports/GAO-26-107538/index.html); [Handwerker 2025](https://www.congress.gov/crs_external_products/IF/PDF/IF13084/IF13084.1.pdf)).

That calendar gives the sharpest publicly available variation among the five candidate drivers. A reproducible covariate should be computed from the archived release schedule, not encoded only as month-of-year:

```math
D_t=\#\{\text{federal business days from the first business day after the 12th through first closing}\}.
```

Separate indicators should flag federal holidays, nonstandard releases, and lapses in appropriations. The number of days is plausibly predetermined, but the release schedule is not literally randomized and its effect may interact with industry payroll frequency and reporting mode.

#### Direct CES evidence

Copeland's early probability-sample studies remain the closest direct evidence. In January 2001–December 2002 data, eight of the ten months with only nine collection days and eight of ten with ten days had the largest late-reporting rates. Correlations between collection interval and late reporting ranged from about −0.30 to −0.53 across establishments and −0.33 to −0.71 when weighted by employment. Late reporters represented 11%–35% of establishments and 13%–43% of employment among units eventually observed by third closing, implying that late reporters were often larger. In pooled cells the correlation between late-reporting rate and absolute relative first-to-third revision was only about 0.18 and was heterogeneous by industry ([Copeland 2003a](https://www.bls.gov/osmr/research-papers/2003/pdf/st030370.pdf)).

This is **supported but weak** evidence: four industries, roughly two years, a redesigned survey still in transition, simple correlations, and a relative-revision outcome rather than a modern variance model. It does not establish that shortening the window raises total-nonfarm revision dispersion, much less separate first-to-second from second-to-third scale.

Copeland and Valliant's peer-reviewed imputation experiment confirms the modest leverage of observables available at the time. Across construction, manufacturing, mining, and wholesale, average absolute first-to-third relative revisions ranged from 0.08% to 0.30%; alternative late-report imputations reduced absolute revisions only 6%–8% in three industries and essentially not at all in mining, with no statistically significant aggregate improvement ([Copeland and Valliant 2007](https://www.scb.se/contentassets/ca21efb41fee47d293bbee5bf7be7fb3/imputing-for-late-reporting-in-the-u.s.-current-employment-statistics-survey.pdf)). The paper shows that late-report information matters, but not that a simple imputation can remove its revision tail.

Robertson's 2003–12 production profile supplies the best national distributional benchmark: first-to-third absolute revisions were 15,000 at the 25th percentile, 38,000 at the median, 66,000 at the 75th, 115,000 at the 90th, and 169,000 at the 99th; 96% were under 0.1% of employment. Roughly 71% of employment-weighted receipts were present at first close, 20% arrived between first and second, and only 2.4% between second and third. Government—especially local government and education—was a systematic late-reporting exception; size, region, pay frequency, and mode did not yield a general directional rule ([Robertson 2013](https://www.bls.gov/osmr/research-papers/2013/pdf/ec130070.pdf)).

Copeland's companion paper makes the estimand clearer. Holding production rules fixed, revisions between closings arise from the different link relatives contributed by late reporters; the estimator assumes response is ignorable within adjustment cells. Annual CES–QCEW differences cannot identify nonresponse bias alone because they also contain sampling, frame, birth–death, and reporting-concept errors ([Copeland 2003b](https://www.bls.gov/osmr/research-papers/2003/pdf/st030360.pdf)). That warning remains current.

Huff and Gershunskaya match CES units to QCEW for 2004–08 and partition an approximation to total error into nonresponse, sampling, CES-versus-QCEW reporting, and a frame/birth–death-like residual. Their national-private nonresponse components vary sharply in sign and magnitude (approximately +219, −206, +29, +280, and +273 thousand across the five years), and lower response did not mechanically imply larger nonresponse bias. The residual labeled birth–death is not the production model's forecast error, and five annual observations cannot establish a scale law ([Huff and Gershunskaya 2009](https://www.bls.gov/osmr/research-papers/2009/pdf/st090050.pdf)).

#### Secular collection-rate change

First-closing collection initially improved: the annual rate rose from roughly 39.6% in 1981 to 78.1% in 2014, aided by prompting and electronic modes ([Johnson 2016](https://www.bls.gov/opub/mlr/2016/article/one-hundred-years-of-current-employment-statistics-data-collection.htm)). It then fell. GAO reports a third-release CES response rate decline from about 62% in fiscal 2015 to 42% in fiscal 2025; this response concept differs from monthly collection, which counts usable receipts from active reporters. BLS appropriately cautions that response rates do not map one-for-one into nonresponse bias ([BLS response-rate portal](https://www.bls.gov/osmr/response-rates/)).

| Year | First close | Second close | Third close |
|---:|---:|---:|---:|
| 1981 | 39.6% | — | — |
| 1983 | 45.0% | 79.7% | 89.2% |
| 1990 | 52.2% | 82.1% | 90.8% |
| 2000 | 58.6% | 74.0% | 78.9% |
| 2009 | 73.3% | 90.5% | 93.1% |
| 2014 | 78.1% | 95.3% | 96.9% |
| 2024 | 60.4% | 89.0% | 90.9% |

The early rows are annual historical summaries and the last row is the 2024 monthly-series average; they are not guaranteed to share every operational denominator detail. Still, all three closings declined from their mid-2010s peaks, with the first close falling most.

The four public monthly series—`CEU00000000C1`, `C2`, `C3`, and total-private third-release response `CEU05000000RR`—were added to the public database in December 2024, but carry historical observations. The C1–C3 monthly history begins in January 2000; Johnson's published annual table extends selected rates to 1981. The consistently defined response series begins later. Any empirical design must preserve the distinction and record definition changes ([BLS 2024 collection-series notice](https://www.bls.gov/ces/notices/2024/ces-collection-rates-have-been-added-to-the-online-public-database.htm)).

Recent evidence is deliberately two-sided. Leduc, Oliveira, and Paulson plot annual mean absolute first-to-second payroll-growth revisions from 1990: 2020–21 is exceptional, but 2022–24 is roughly in line with the 1990–2019 average despite a much lower response rate ([2025](https://www.frbsf.org/research-and-insights/publications/economic-letter/2025/03/do-low-survey-response-rates-threaten-data-dependence/)). That is useful descriptive evidence against a simple contemporaneous monotone relation. It does not condition on collection days, separate response from collection, distinguish NSA receipts from seasonal-factor revision, or analyze the second-to-third and benchmark stages.

GAO likewise documents the response decline and occasional large revisions, but finds that BLS met its monthly- and benchmark-revision accuracy goals in every fiscal year 2020–25 after temporarily relaxing a pandemic-period target. BLS's CES nonresponse assessments were not public because linking respondents and nonrespondents to a comprehensive comparator proved difficult; a new study is under way. Handwerker reports no stable late-reporter pattern by size, industry, geography, pay frequency, or collection mode, although government reports have recently accounted for some large monthly revisions. Thus “response rates fell, therefore revisions rose” is asserted more often than demonstrated.

The remaining CES-specific nonresponse literature does not fill that gap. Kratzke's linked CES–QCEW analysis addresses average weekly earnings rather than employment revisions ([Kratzke 2013](https://www.bls.gov/osmr/research-papers/2013/pdf/st130160.pdf)); Dixon and Tucker model late revisions with a nonproduction quantile approach ([2016](https://www.bls.gov/osmr/research-papers/2016/pdf/st160080.pdf)); and Dixon finds area-characteristic response models with low explanatory power ([2018](https://www.bls.gov/osmr/research-papers/2018/pdf/st180100.pdf)). These are relevant evidence about selection, not estimates of the requested variance effect.

#### Claims audit by revision stage

- **First → second:** strongest hypothesized effect. Collection days and first-close coverage should enter the scale equation separately; the latter is partly endogenous to shocks and operational strain.
- **Second → third:** only the incremental 89.0%→90.9% average coverage in 2024 is relevant, but a small share can have high leverage if reporters are large or trend-different. No modern study estimates that leverage distribution.
- **Third → benchmark:** permanent nonresponse and attrition may contribute, but the benchmark gap cannot isolate them. QCEW and CES reporting errors may also be correlated.
- **Post-benchmark:** late reports can change recent post-March months, whereas the historical wedge is mechanical. Collection rates should not be assigned to the whole wedge vector.
- **Not found:** a CES paper relating monthly first-close collection days or coverage to a conditional variance of stage-specific revisions after controlling for seasonality, industry mix, and macro volatility.

### 2.3 BLS funding

#### What is documented

The appropriations record contains genuine operational shocks, but not a demonstrated CES revision-production function.

| Episode | Documented statistical-program effect | What can be inferred about CES revision magnitude? |
|---|---|---|
| FY2008 shortfall | Enacted BLS funding was \$544.3 million, \$30.2 million (5.3%) below the request. BLS discontinued metropolitan hours/earnings and employment estimates for the 65 smallest MSAs—about 3,900 monthly series—froze hiring, and delayed a CES project intended to improve response and reduce revisions; FY2009 restored \$19.2 million. | Direct evidence that budget pressure changes scope and improvement work; no national-CES revision estimate. |
| FY2013 sequestration | More than \$30 million (about 5%) was removed. BLS eliminated Mass Layoff Statistics, green-jobs products, and international labor comparisons and imposed staffing/operational constraints. | Program loss is observed; a national CES scale effect was neither estimated nor clearly predicted. |
| FY2014 constraints | Federal statistical-program documentation anticipated curtailed QCEW activity and a small degradation in QCEW quality and CES-frame accuracy. | A relevant benchmark/frame channel, but the effect was forecast, not measured. |
| Oct. 2013 lapse | CES operations and releases were suspended October 1–16. | The delayed September estimate had 83.5% collection and BLS found no discernible effect; timeliness changed, not demonstrably accuracy. |
| Jan. 2018 lapse | Three-day lapse; data collection continued or was recovered. | BLS reported no discernible total-nonfarm effect. |
| Dec. 2018–Jan. 2019 partial lapse | Labor/BLS already had full-year appropriations and did not shut down. Other agencies' furloughs entered CES as real employment changes and may have affected contractor reporting. | It is incorrect to code this as a BLS production shutdown. The contractor-data effect is unquantified. |
| Oct. 1–Nov. 12, 2025 lapse | BLS collection and releases were suspended; the September report moved from October 3 to November 20, and no stand-alone October report was issued. | September collection reached 80.2% because electronic receipts continued and the window was extended. This is a major timing intervention, but its direction is not “lower response”; no mature causal evaluation was found by the research cutoff. |
| Jan. 31–Feb. 3, 2026 lapse | A short partial lapse delayed the January Employment Situation from February 6 to February 11. The DOL contingency plan listed 2,055 BLS employees, one retained and 2,054 furloughed during a lapse. | Another timeliness intervention with little leverage for a quality effect; it should be coded separately from the 2025 lapse. The February–April 2026 DHS-only lapse did not close BLS. |

Primary documentation: [BLS's FY2008 program-impact statement](https://www.bls.gov/bls/budgetimpact.htm), [BLS sequestration notice](https://www.bls.gov/bls/sequester_info.htm), [BLS's FY2014 enacted-budget effects](https://www.bls.gov/bls/budget2014_enacted.htm), [OMB's FY2014 statistical-program report](https://obamawhitehouse.archives.gov/omb/statistical-programs-2014), [BLS 2013 shutdown FAQ](https://www.bls.gov/bls/shutdown_2013_empsit_qa.pdf), [BLS on the 2018–19 partial lapse](https://www.bls.gov/bls/what-impact-did-the-lapse-of-appropriation-for-some-federal-agencies-have-on-january-employment-data.htm), [BLS's 2025/26 revised-release calendar](https://www.bls.gov/bls/2025-lapse-revised-release-dates.htm), and the [DOL lapse plan](https://www.dol.gov/sites/dolgov/files/general/plans/dol-contingency-plan.pdf).

The resource trend is nonetheless material. The table below reconciles general-fund and Unemployment Trust Fund budget authority where available; FTE is the CBJ fiscal-year-average concept, not OPM headcount. FY2026 FTE is a plan, not a realized count.

| FY | Nominal budget authority (\$m) | BLS FTE | FY | Nominal budget authority (\$m) | BLS FTE |
|---:|---:|---:|---:|---:|---:|
| 2009 | 597.2 | 2,210 | 2018 | 612.0 | 2,022 |
| 2010 | 611.4 | 2,341 | 2019 | 615.0 | 2,057 |
| 2011 | 610.2 | 2,313 | 2020 | 655.0 | 1,961 |
| 2012 | 609.1 | 2,304 | 2021 | 655.0 | 1,965 |
| 2013 | 577.2 | 2,254 | 2022 | 688.0 | 1,949 |
| 2014 | 592.2 | 2,165 | 2023 | 698.0 | 2,023 |
| 2015 | 592.2 | 2,166 | 2024 | 698.0 | 2,058 |
| 2016 | 609.0 | 2,141 | 2025 | 704.0 | 2,019 |
| 2017 | 609.0 | 2,185 | 2026 | 708.5 | 1,773 planned |

Sources: [DOL/BLS transition brief](https://www.dol.gov/sites/dolgov/files/general/foia/presidential-transition-docs/bls.pdf) and [FY2027 DOL Congressional Budget Justification](https://www.dol.gov/sites/dolgov/files/general/budget/2027/CBJ-2027-V3-01.pdf). The apparent nominal increase is misleading. ASA's consistent FY2009-dollar series is \$597.2 million in 2009, \$516.0 million in 2021, \$494.8 million in 2022, \$505.5 million in 2023, \$493.6 million in 2024, and \$485.7 million in 2025—a 2009–25 real decline of 18.7% ([ASA 2025 system report](https://www.amstat.org/docs/default-source/amstat-documents/the-nations-data-at-risk-2025/The-Nations-Data-at-Risk-2025-Report.pdf)). OMB's Public Budget Database generally captures general-fund authority but can omit the trust-fund transfer; it is not directly interchangeable with the table above.

Nominal BLS budget authority must be deflated before analysis and reconciled across enacted, annualized continuing-resolution, sequestration, rescission, and supplemental amounts. Program-level “Labor Force Statistics” or CES obligations are preferable to agency totals when definitions are stable. A continuing resolution is not itself a cut; its operational content depends on duration, anomalies, and the prior-year base. Likewise, a shutdown mainly changes collection and release timing unless it forces a durable loss of staff or sample.

CR exposure is common rather than a rare quasi-experiment. CRS counts 134 interim continuing resolutions during FY1998–2025, four full-year CRs, and roughly five CRs covering 118 days per fiscal year on average ([Saturno et al., CRS R46595](https://www.congress.gov/crs_external_products/R/PDF/R46595/R46595.7.pdf)). The relevant hand-built covariates are days under CR, number of extensions, full-year status, and BLS-specific anomalies—not a binary “CR year.” No study located here links those measures to CES revision scale.

#### What is asserted, contested, and absent

ASA and COPAFS reports plausibly warn that declining real resources erode survey response, modernization, detail, redundancy, and expert staff ([COPAFS/Friends of BLS 2021–25 priorities](https://copafs.org/wp-content/uploads/2021/02/BLS_Priorities2021-2025-Final.pdf)). They are valuable system documentation and advocacy, not identified estimates of payroll revision variance. GAO's 2025 federal-statistics forum reports the same mechanisms from experts, while GAO's 2026 Jobs Report audit supplies an important counterpoint: visible collection problems coexisted with attainment of every stated CES accuracy target in FY2020–25. BLS's monthly target was that the first-to-third NSA growth-rate revision stay within 0.1 percentage point in at least 10 of 12 months (temporarily 8 during January 2020–June 2021), and its benchmark target was a five-year mean absolute revision below 0.4 percentage point. Those binary thresholds do not test whether dispersion rose within the acceptable band. Historical testimony similarly records lost outputs after cuts but does not isolate quality conditional on output survival. Published national/state/area CES detail did fall from 24,511 estimates in FY2022 to 22,049 in FY2025, illustrating that output breadth can absorb pressure while the headline target remains met.

**Not found:** a peer-reviewed or agency study regressing CES revision scale on real appropriations, continuing resolutions, sequestration, or shutdown exposure. More broadly, the federal-statistics literature documents output cancellations, delayed releases, reduced granularity, and modernization backlogs more convincingly than it measures error in surviving series.

Nor did this search locate a documented budget-driven reduction in the **national CES probability sample** comparable to the FY2008 loss of local series. Where BLS protected the principal national indicator by reducing detail, delaying improvements, centralizing collection, or curtailing QCEW work, assigning a hidden national sample cut would be inference, not documentation.

Party composition of the House and Senate appropriations process can be reported as descriptive context or used to stratify a figure. With fewer than forty annual observations, long political regimes, budget trends, divided government, and macro-policy feedback, it is not a defensible causal slope. No such coefficient belongs in the primary model.

#### Claims audit by revision stage

- Funding could operate through staff effort, initiation, follow-up, frame maintenance, sample size, and seasonal/model research; none is a stage-invariant channel.
- The most proximal monthly estimand is a funding shock's effect on first-close collection conditional on calendar days, followed by first-to-second scale. Annual benchmark effects would be lagged through QCEW and frame operations.
- Shutdown indicators should be treated as case studies or known timing interventions, not pooled with annual real funding.
- Slow real appropriations cannot separately identify effects of staffing and sample size without strong priors or external instruments. The design below estimates a capacity association, not a funding multiplier.

### 2.4 Sample design, composition, and collection mode

#### Design history and the 2003 break

The legacy CES was a decentralized quota/cutoff sample tilted toward large, continuing establishments. Sample size expanded from about 160,000 establishments in 1975 to 425,000 in 1989, chiefly to support service-sector detail. By the mid-1990s, BLS comparisons in ten large states found sampled firms to be roughly ten years older than the establishment universe; simulations produced large benchmark errors absent manual/bias adjustments. The 1991 benchmark controversy accelerated an ASA review, although BLS attributes that particular revision to payroll-processor reporting changes in the benchmark source, not to the sample design ([Kelter 2016](https://doi.org/10.21916/mlr.2016.35)).

Probability-sample development began in 1995, was completed in 1997, and was tested from 1998 through 2002. Production conversion was staggered: wholesale trade in 2000; mining, construction, and manufacturing in 2001; transportation/public utilities, retail, and finance in 2002; services in 2003. Thus “the 2003 sample break” is both useful shorthand and inaccurate as a sharp national treatment date. Government remained largely a high-coverage legacy design.

The modern design is a stratified simple random sample of worksites clustered by state UI account. Each state receives a fixed allocation, then optimum allocation distributes units across supersectors and size classes to minimize variance of monthly total-nonfarm change subject to state-reliability bounds. Large/certainty units overlap heavily. The full sample is redrawn annually from the QCEW-derived longitudinal frame, a new-birth sample is drawn midyear, about two-thirds overlaps year to year, most noncertainty units remain two to four years, and attrition must be replaced through initiation. Quarterly implementation by supersector began in August 2014, shortening the frame-to-production lag ([Kelter 2016](https://doi.org/10.21916/mlr.2016.35)).

Published headcounts require caution. Historical figures may count establishments, UI accounts, reporting businesses/agencies, or worksites. Current BLS material reports roughly 119,000 businesses and government agencies representing about 622,000 worksites, covering around 26% of universe employment ([CES FAQ](https://www.bls.gov/web/empsit/cesfaq.htm); [benchmark technical notes](https://www.bls.gov/web/empsit/cestn.htm)). Allocation is highly nonproportional: in March 2025, UI accounts below 10 employees were 73.1% of universe accounts and 10.4% of universe employment but 33.9% of sample accounts and only 0.3% of sampled employment; accounts with at least 1,000 employees were 0.2%/29.3% of the universe but 6.2%/69.4% of the sample. A regression that splices counts or ignores this composition would manufacture a sample-size trend.

#### What is known about revision magnitude

The official post-2003 table is consistent with an improvement in first-to-third revision scale, but it cannot assign the change to probability sampling. SA MAR fell 61,000→51,000 and NSA MAR 83,000→53,000; however, NAICS and concurrent adjustment changed simultaneously, industries converted in different earlier years, and macro volatility differs across the two eras. Within the modern sample, BLS publishes relative standard errors and coverage by supersector, but this search found no national panel study estimating the elasticity of **revision variance** with respect to usable linked sample, employment coverage, or size-class composition.

Evidence from lower geographies supports only the general mechanism: smaller MSAs, which tend to have smaller samples, show larger absolute percentage benchmark revisions. Over 2005–15, MSAs below 100,000 employment typically had absolute revisions around 1.1%–2.1%, versus roughly 0.4%–1.1% for MSAs above one million ([Robertson 2017](https://www.bls.gov/opub/mlr/2017/article/benchmarking-the-current-employment-statistics-survey-perspectives-on-current-research.htm)). Geography, industry mix, estimator structure, and population volatility all co-vary with sample size, so this is not a national-CES sample-size coefficient.

BLS's 2015 report to Congress simulated increasing the state/area sample by 47,000 UI accounts to 190,000 at a cost of about \$9 million: projected sampling standard errors fell 10%–12% for small MSAs, 6%–8% for medium MSAs, and about 5% for large MSAs. A larger 85,000-account expansion with cell minima produced much larger gains for small MSAs but less than 1% for large areas. This is direct design evidence for **sampling precision**, not for close-to-close or benchmark revision variance; treating it as the latter changes the estimand.

#### Collection mode and attrition

CES moved from mail and state collection toward centralized CATI, touchtone data entry (TDE), web, fax, and EDI. During the redesign, TDE was the volume backbone, while EDI concentrated large multi-establishment reporters ([Rosen, Manning, Harrell, and Skuta 1999](https://www.bls.gov/osmr/research-papers/1999/pdf/st990030.pdf)). The 2015 report's 2003–12 employment-weighted first-closing receipt shares were roughly 22.9% CATI, 8.5% EDI, 43.9% web, 11.9% TDE/state, and 12.8% other ([BLS 2015 report to Congress](https://www.bls.gov/sae/additional-resources/bls-report-to-congress-on-ces-methodology-for-metropolitan-statistical-areas-2015.pdf)); a 2016 account using a different denominator reported 44% EDI, 28% CATI, 17% web, 4% fax, 3% touchtone, and 4% other ([Robertson 2016](https://www.bls.gov/osmr/research-papers/2016/st160050.htm)). Their divergence is a warning to retain both denominator and closing stage. These are operational summaries, not a public monthly mode panel.

Clayton's redesign account reports that automated mixed-mode collection raised first-close response by about 20 percentage points and reduced average preliminary monthly revisions about 38% ([1997](https://www.bls.gov/osmr/research-papers/1997/pdf/st970030.pdf)). It is the strongest historical numeric mode claim, but it is a program before/after comparison during a broader redesign, not a controlled estimate, and it predates NAICS and the probability sample.

Attrition is material but old evidence is the only quantified public evidence found. Among active December 2000 units in Copeland's study, 69%–75% reported in all 12 subsequent months, 11%–15% attrited, and 14%–18% responded episodically; over a separate 18-month window only 47%–57% were complete reporters and 12%–16% attrited. Even among consistent reporters, only 23%–29% always made first close ([Copeland 2003b](https://www.bls.gov/osmr/research-papers/2003/pdf/st030360.pdf)). No comparable modern public paradata panel was found.

Mode can affect timeliness and respondent composition, but it is not randomly assigned: large payroll processors dominate EDI, reluctant/new units receive CATI, and web eligibility and initiation change over time. Handwerker's review finds no consistent modern pattern in lateness by mode, industry, size, geography, or pay frequency. **Not found:** a CES study mapping the EDI/web/TDE/fax/CATI mix, attrition, or sample age to stage-specific revision dispersion after selection adjustment.

#### Birth–death and benchmark heterogeneity

The birth–death model belongs with sample/frame coverage, but its revision role is stage specific. Monthly forecasts fill the lag in observing openings and closures; the ordinary forecast is held fixed across first, second, and third closings, so it does not directly generate their NSA revisions. At annual benchmark, model error is part of the CES–QCEW discrepancy. In March 2009, total nonfarm was revised −902,000; BLS's retrospective table reports that the revision would have been about −185,000 without the cumulative modeled net birth–death contribution, making model miss material in that episode ([2009 benchmark article](https://www.bls.gov/ces/publications/benchmark/ces-benchmark-revision-2009.pdf); [birth–death FAQ](https://www.bls.gov/web/empsit/cesbdqa.htm)).

BLS does not calculate a distinct seasonally adjusted “birth–death contribution.” The NSA modeled amount enters the series before concurrent adjustment, so its SA impact depends on the full series and factor run. Any paper or nowcast that adds an NSA birth–death amount mechanically to an SA payroll change is asserting a decomposition BLS does not produce.

That example should not be generalized mechanically. Model effects are strongly seasonal and sector heterogeneous; the “revision without birth–death” is a counterfactual accounting calculation, not a causal error decomposition, because sample and modeled components interact. BLS research found quarterly updating reduced post-benchmark birth–death revisions in five of seven 2003–09 comparisons, and the production method later adopted quarterly inputs ([Clausen 2011](https://www.bls.gov/osmr/research-papers/2011/st110020.htm)). Persistent forecast misses after 2020 prompted a current-sample adjustment for April–October post-benchmark estimates in 2025 and a broader current-information change in January 2026 ([CES birth–death FAQ](https://www.bls.gov/web/empsit/cesbdqa.htm)). These are credible change points for benchmark/post-benchmark scale, not for the within-month late-reporting component.

Recent BLS model research quantifies the forecast problem more directly. Grieves, Mance, and Witt report monthly supersector birth–death RMSE of 27,100 for the production ARIMA model, 20,100 for modified regARIMA, and 10,900 for a selected-model average; in 2020 the corresponding figures were 95,300, 69,600, and 33,200. Twelve-month total-private RMSE was 293,000 for production, 214,000 for modified regARIMA, and 167,000 for the preferred combination ([Grieves, Mance, and Witt 2023](https://www.bls.gov/osmr/research-papers/2023/pdf/st230010.pdf)). This is strong evidence of sector-heterogeneous, heavy-tailed benchmark risk and motivates hierarchical shrinkage. It still does not allocate the observed benchmark gap between birth–death, QCEW error, and sample/reporting error.

#### NAICS, reweighting, and reconstruction

NAICS revisions alter cell definitions, weights, and detailed histories. The 2003 SIC→NAICS conversion was fundamental; later NAICS versions were introduced with benchmark releases, including NAICS 2017 and NAICS 2022. At total nonfarm, pure recoding should net to zero, but reconstruction choices, scope changes, and estimation-cell aggregation can affect detailed and occasionally aggregate histories. BLS commonly reconstructs affected series to 1990 and can revise data outside the ordinary 21-month NSA window ([Manning and Stewart 2017](https://www.bls.gov/opub/mlr/2017/article/benchmarking-the-current-employment-statistics-national-estimates.htm); [CES classification history](https://www.bls.gov/ces/naics/home.htm)).

Consequently, NAICS-release vintages should be marked as concept breaks. They are not ordinary draws from a stationary measurement equation, and treating their revisions as “error corrections” overstates variance.

#### Claims audit by revision stage

- **First → second / second → third:** usable linked sample and its late-reporter composition are relevant; total enrolled sample is a weaker proxy. Mode and size composition may matter through reporting delay, but public evidence is absent.
- **Third → benchmark:** frame coverage, sample composition, estimator, permanent nonresponse, reporting differences, and birth–death error all enter. The March gap does not decompose them without matched microdata.
- **Post-benchmark:** wedge-back is mechanical; post-March estimates inherit the reanchored level, old sample ratios, revised birth–death amounts, late receipts, and later sample introduction.
- **Contested:** “the larger the sample, the smaller the revision” is a design intuition, not an established time-series relation; optimum allocation and composition matter more than a raw worksite count.

### 2.5 Staffing

#### Observable quantities

Three imperfect sources can be assembled:

1. **OPM FedScope** supplies BLS on-board employment snapshots and characteristics, generally from the late 1990s onward. Counts are people, not FTE, and reorganizations, bureau coding, duty station, employment status, and September-versus-quarter snapshots require reconciliation.
2. **BLS/DOL congressional budget justifications (CBJs)** report enacted, estimated, and requested agency FTE—often with activity/program tables and narrative on hiring constraints. These are fiscal-year averages or plans, not necessarily realized CES production labor.
3. **OMB statistical-program reports, Federal Employment Reports, and archived personnel tables** extend further back but have classification and coverage breaks. Contractor and state-workforce-agency effort is mostly absent.

The appropriate primary series is realized BLS FTE from successive CBJs, cross-checked against FedScope September headcount. A second, narrower series should code the Labor Force Statistics/CES activity when comparable. Each observation needs provenance and a definition-break flag; filling gaps by linear interpolation would falsely create monthly information.

Agency FTE is a particularly noisy CES-capacity proxy. Robertson's FY2015 production accounting put the CES budget near \$61 million: \$26 million federal (42%), \$8 million state (14%), \$25 million contractor (41%), and \$2 million other (3%). Roughly 595 workyears comprised 181 federal, 84 state, and 330 contractor workyears; only 94 federal workyears were in the CES program office, and staff compensation accounted for about 97% of budgeted CES costs ([Robertson 2016](https://www.bls.gov/osmr/research-papers/2016/st160050.htm)). A bureau-wide FTE series therefore misses most contractor/state labor and can move differently from CES-specific capacity.

FedScope's legacy annual snapshots begin in September 1998 and become quarterly in December 2007; the replacement portal provides monthly records. Recent OPM on-board counts—2,330 in FY2023, 2,321 in FY2024, 2,165 in FY2025, and 1,846 by July 1, 2026—are headcounts, while the CBJ table above reports fiscal-year-average FTE. These are related but not spliceable concepts ([ASA midyear 2026 update](https://www.amstat.org/docs/default-source/amstat-documents/FedStatHealth_MidYearUpdate.pdf)).

#### What the literature establishes

Agency documents connect staffing constraints to delayed modernization, reduced follow-up/initiation, curtailed detail, and discontinued outputs. For example, sequestration imposed a hiring freeze and curtailed operations; expert reports from ASA, COPAFS, and GAO describe loss of specialist knowledge and resilience. These are real production channels.

They do **not** establish a CES elasticity. This review found no CES paper and no persuasive cross-agency panel linking statistical-agency FTE to revision magnitude, conditional bias, or tail risk. Cross-agency output counts are not comparable quality measures; staff may be cut in response to automation, and agencies protect principal indicators by sacrificing lower-priority products. The FY2020–25 coexistence of falling CES response and achieved revision targets illustrates that buffering.

#### Claims audit by revision stage

- Staffing is plausibly most proximal to initiation, follow-up, editing, outlier handling, frame/QCEW processing, and model maintenance. Those mediators should be modeled before any direct staff coefficient.
- Monthly interpolation of annual FTE would create pseudo-replication. Identify any capacity association from annual deviations, discrete hiring freezes, or program-specific shocks—not 480 nominal “monthly” observations.
- BLS headcount omits Census work on CPS, state partners' QCEW/CES effort, contractors, payroll-processor systems, and skill mix. A null BLS-FTE coefficient would not imply staffing is irrelevant.
- **Not found:** published evidence that lower BLS staffing increased any of the four CES revision scales. This is an open empirical question, not a documented fact.

### 2.6 Cross-driver synthesis: annual benchmark and post-benchmark revisions

#### Third release → preliminary and final benchmark

The preliminary national benchmark, usually announced in August or September when first-quarter QCEW becomes available, is a signal of the March discrepancy; it does **not** revise the official CES series. The final comprehensive March-reference benchmark is incorporated with January estimates the following February. BLS publishes final revisions from 1979 and preliminary revisions from 2000 ([2025 CES technical notes, table 5](https://www.bls.gov/web/empsit/cestn.htm)).

The percent scale has been broadly stable at small values but with episodic large errors. Final absolute March revisions averaged approximately 0.25% in 1979–99 and 0.22% in 2000–25 when calculated from BLS's one-decimal published percentages; rounding and changing employment levels make those approximate. The 2007–16 official MAR was 0.2%, with a range from −0.7% to +0.3% ([Manning and Stewart 2017](https://doi.org/10.21916/mlr.2017.25)). The most recent sequence was −0.3% in 2019, −0.1% in 2020, less than 0.05% in 2021, +0.3% in 2022, −0.1% in 2023, −0.4% (−598,000) in 2024, and −0.5% (−861,000 after scope adjustment; −862,000 headline NSA difference) in 2025. The March 2026 preliminary estimate is −79,000 (−0.1%); no final estimate existed at the research cutoff ([BLS preliminary release, August 28, 2026](https://www.bls.gov/news.release/prebmk.nr0.htm)).

Thus Decker's statement that revisions “have increased in recent years” is defensible for the 2024–25 benchmark levels and some recent monthly windows, but not as a four-decade monotone trend ([Decker 2026](https://doi.org/10.3386/w34924)). The 2024 preliminary revision was −818,000 versus a −598,000 final; for 2025 the analogous figures were −911,000 and −861,000. The preliminary benchmark is highly informative but is not a zero-error proxy for the final benchmark. A model should use it as a noisy advance measurement.

The benchmark is also not literal truth. QCEW and other benchmark inputs have coverage, classification, imputation, processing, and employer-reporting errors; CES and QCEW can share reporters and payroll systems, violating independence. BLS therefore describes the revision as the difference between two separately produced, errorful counts ([Robertson 2017](https://www.bls.gov/opub/mlr/2017/article/benchmarking-the-current-employment-statistics-survey-perspectives-on-current-research.htm)). The 1991 episode—driven substantially by a change in how UI counts were reported—and later reconstructions show that a benchmark-source change can reshape an entire historical path.

#### Wedge-back and forward propagation

For national NSA employment, let $`d_{s,y}=B_{s,y,Mar}-y^{(3)}_{s,y,Mar}`$. Holding scope changes aside, the published backward revision approximately applies

```math
w_m d_{s,y},\qquad
w_m=(1/12,2/12,\ldots,11/12,1)
```

to April through the following March. The previous March remains fixed. This linear wedge assumes error accumulated steadily; it does not estimate the monthly timing of error. It turns one March discrepancy into eleven mechanically correlated historical revisions. Treating those observations as twelve independent errors would severely overstate information ([CES FAQ](https://www.bls.gov/web/empsit/cesfaq.htm)).

After March, BLS applies previously derived monthly sample link relatives to the new level and substitutes updated net birth–death values; later sample receipts and the annual sample update affect the newest months. In a typical benchmark release, about 21 months of NSA data change, while new specifications and factors can revise five years of SA data. Reconstructions, NAICS conversions, scope corrections, and errors can reach further back ([CES calculation chapter](https://www.bls.gov/opub/hom/ces/calculation.htm)). “Post-benchmark revision” therefore contains at least four objects: a propagated level shift, altered birth–death values, new sample information, and re-seasonalization.

The wedge has a mechanically large effect on monthly changes when the March gap is large. Ignoring cell aggregation and reconstructions, the −861,000 March 2025 gap implies about −71,750 per monthly change from April through March; the −598,000 March 2024 gap implies about −49,833. These are deterministic translations of the published March gaps, not estimates of the true month in which error occurred.

#### What the research has and has not measured

Haltom, Mitchell, and Tallman's real-time reconstruction shows that annual benchmarks account for the largest enduring changes in the payroll **level**, while monthly and benchmark revisions both materially alter monthly growth. Their 1990–2005 Granger-style tests find lagged benchmark revisions informative for the next benchmarked series, but the sample is short, uses pre-modern and transitional methods, and does not identify the revision-scale drivers or the business-cycle mechanism ([Haltom, Mitchell, and Tallman 2005](https://fraser.stlouisfed.org/files/docs/publications/frbatlreview/rev_frbatl_2005_v90no2.pdf)).

Dey and Loewenstein's quarterly-benchmark research stresses that a small March error can conceal offsetting monthly errors and that CES/QCEW reporting-period seasonality differs. Robertson's later simulation is the strongest direct challenge to the linear wedge. For 2006–15, mean CES-minus-QCEW quarterly artifacts after current benchmarking were −447,500 in Q1, −96,200 in Q2, +292,000 in Q3, and +260,700 in Q4—3.9, 0.8, 2.5, and 2.3 CES three-month standard errors. A quarterly-endpoint procedure reduced them to −15,500, −13,900, +9,200, and +52,400 ([Robertson 2021](https://www.bls.gov/osmr/research-papers/2021/st210070.htm)). This is persuasive evidence that the maintained linear error path is wrong on average, but it is an aggregate simulation, not a realized vintage-by-vintage estimate of the latent monthly discrepancy ([Dey and Loewenstein 2017](https://www.bls.gov/opub/mlr/2017/article/a-quarterly-benchmarking-procedure-for-the-current-employment-statistics-program.htm); [Groen 2012](https://www.scb.se/contentassets/ca21efb41fee47d293bbee5bf7be7fb3/sources-of-error-in-survey-and-administrative-data-the-importance-of-reporting-procedures.pdf)).

Brave, Gascon, Kluender, and Walstrum explicitly model state-level revisions in state space and reduce average benchmark-revision MAE about 11% (14% when averaged with existing models). States supply cross-sectional variation and quarterly QCEW anchors; the paper does not estimate national driver effects or distinguish all four vintage stages ([Brave et al. 2021](https://doi.org/10.1016/j.ijforecast.2021.02.006)). Cajner et al. combine CES with ADP microdata in a state-space model and place roughly equal weight on the two signals, demonstrating the value of an external measurement; access to proprietary payroll microdata limits replication ([Cajner et al. 2022](https://www.nber.org/papers/w26033)).

The strongest conclusion is a gap: no located paper models the national first, second, third, preliminary-benchmark, final-benchmark, wedge, and later-reseasonalized vintages jointly while allowing the **scale** of news and noise to depend on collection, seasonal, sample, budget, and staffing covariates.

### 2.7 CES in the general data-revision literature

| Framework | Principal result / estimand | CES application achieved | CES question still open |
|---|---|---|---|
| Mankiw–Shapiro “news versus noise” | Tests whether revisions are orthogonal to early or later estimates; early U.S. GNP revisions looked more like news. | Provides the canonical information-set distinction. | Has not separated CES news/noise across all four stages or modeled their conditional scales. |
| Croushore–Stark real-time data | Shows that model results and forecasts depend on the vintage actually available. | Philadelphia Fed's real-time payroll files make aggregate pseudo-real-time work possible. | Aggregate files do not provide the full NSA/SA sectoral production inputs needed for driver decomposition. |
| Aruoba revision properties | Macro revisions are often biased, large relative to published change, and predictable; simple “well-behaved news” assumptions fail. | Includes broad U.S. macro indicators and motivates diagnostic tests on payroll vintages. | Does not trace CES production channels or four-stage heteroskedasticity. |
| Jacobs–van Norden state space | Represents multiple vintages as noisy/news-bearing measurements of a latent value and identifies covariance restrictions jointly. | Direct conceptual foundation for this proposal; subsequent work applies it mainly to national accounts. | No complete CES multi-stage, sector-hierarchical implementation was found. |
| Kishor–Koenig real-time Kalman filtering | Uses a vintage VAR/state-space structure to produce efficient real-time estimates without simply declaring the latest release truth. | Has influenced payroll/state nowcasting and motivates filtering a ragged vintage panel. | Constant or parsimonious error structures do not answer which CES drivers change revision scale. |
| Fixler–Grimm BEA revision accounting | Compares revision stages and source-data/method changes using MAR and bias metrics. | Supplies a useful template for stage-specific official-quality scorecards. | BEA GDP vintages and revision schedules differ fundamentally from CES sample closings and annual wedge. |
| Faust–Rogers–Wright cross-country GDP | Finds U.S. GDP revisions closer to news than many other G7 revisions and shows institutional heterogeneity. | Warns against transferring one agency/country error model to another. | No analogous cross-program or cross-agency CES design identifies funding/staff mechanisms. |
| Clements Bayesian VAR with stochastic volatility | Extends real-time forecasting to time-varying revision variance and density evaluation. | Establishes that stochastic revision volatility can improve real-time macro density forecasts. | Not CES-specific; no benchmark operator, NSA/SA joint measurement, or sector pooling. |

Core sources are [Mankiw and Shapiro (1986)](https://doi.org/10.3386/w1939), [Croushore and Stark (2001)](https://doi.org/10.1016/S0304-4076%2801%2900072-0), [Aruoba (2008)](https://doi.org/10.1111/j.1538-4616.2008.00115.x), [Jacobs and van Norden (2011)](https://doi.org/10.1016/j.jeconom.2010.04.010), [Kishor and Koenig (2012)](https://doi.org/10.1198/jbes.2010.08169), [Fixler and Grimm (2005)](https://apps.bea.gov/scb/pdf/2005/02February/0205_NIPAs.pdf), and [Faust, Rogers, and Wright (2005)](https://doi.org/10.1353/mcb.2005.0029).

The smaller CES-specific efficiency literature is closer to the outcome but still misses the requested estimand. Neumark and Wascher find pre-redesign preliminary payroll estimates inefficient and reduce unpredictable benchmark variance using contemporaneous information ([1991](https://ideas.repec.org/a/bes/jnlbes/v9y1991i2p197-205.html)); Phillips and Nordlund find cyclical and seasonal predictability in benchmark revisions ([2012](https://doi.org/10.1016/j.econlet.2011.12.118)). Both target conditional means/efficiency, not time-varying scale under modern production. Dorfman, Li, and Zhang's hierarchical Bayesian model forecasts the sign and magnitude of Jobs Report revisions but does not treat the four stages as noisy observations of a common latent path ([2025](https://doi.org/10.1515/bejm-2024-0145)).

Two 2026 Federal Reserve studies sharpen the secular-trend claim. Pinheiro and Quinlan report a mean of −4,510 and an interquartile range of −59,000 to +57,000 for current-minus-third monthly-change revisions over 1980–2025, with no mean break; that terminal endpoint mixes benchmark, annual seasonal, and later reconstruction changes ([2026](https://doi.org/10.26509/frbc-ec-202612)). Lubik and Titcomb report standard deviations of initial-to-current revisions of 116,300 in 1964–83, 111,600 in 1983–2000, and 106,000 in 2000–25, while mean bias falls more sharply ([2026](https://www.richmondfed.org/publications/research/economic_brief/2026/eb_26-01)). Those results argue against a long-run explosion in unconditional scale; neither separates the four production stages or conditions variance on the five candidate drivers.

The news/noise labels are restrictions, not findings. For CES, a late report can be news relative to the first-close information set yet contain reporting error; a final benchmark can reveal CES news while adding QCEW noise; and concurrent seasonal adjustment makes a new observation revise several estimated factors. Those mixed channels are precisely why a full covariance model and sensitivity analysis are required.

**Grey and private-nowcasting literature.** ADP's redesigned National Employment Report is an external payroll signal, not a CES forecast: since 2022 it weights a matched weekly ADP client sample to QCEW industry × state × size cells, and prior-month revisions incorporate late client activity. ADP explicitly says the redesigned product no longer targets the BLS payroll release ([ADP technical notes and FAQ](https://adpemploymentreport.com/)). That makes it useful as an error-correlated auxiliary observation, as in Cajner et al., but not evidence about why BLS revisions change scale. Private-nowcaster commentary located in this search generally compares forecast errors or headline surprises; it does not identify CES late-sample, seasonal, birth–death, funding, or staffing mechanisms. It is therefore labelled grey evidence and not used to establish a driver coefficient.

## 3. Gap analysis

### 3.1 Open questions

| Open question | Closest existing work | Why it falls short | Data required | Public-source feasibility |
|---|---|---|---|---|
| 1. Has the conditional **scale** of each of the four revision stages changed secularly? | BLS 1979–present monthly table; benchmark table; Haltom et al.; Decker | Mostly unconditional MARs, two coarse regimes, mixed level/change estimands, little tail modeling | Full release-by-reference-month panel, stage labels, concept breaks, sector weights | **High from 2003; medium before 2003.** Aggregate SA reaches 1964 in RTDSM and official stage revisions 1979; detailed NSA/SA vintages start May 2003. |
| 2. How much of stage-1 and stage-2 SA revision is revised NSA sample versus recomputed factor? | BLS concurrent experiment; Wright; Loya | Experiment is pre-2003; Wright's implied factor mixes benchmark/window effects; Loya studies a seam effect | Exact NSA input at each closing, each closing's X-13 specs/prior/outlier files, unrounded outputs | **Medium after 2003, low historically.** Reconstructable only where input files were archived; internal unrounded series may be necessary. |
| 3. Does the calendar collection interval change revision dispersion? | Copeland 2003; BLS/CRS/GAO collection summaries | Two-year, four-industry correlations; no modern scale regression; release timing and holidays not fully controlled | Release calendar, closing calendar, business-day count, C1–C3, sector revisions | **High for aggregate; medium by sector.** Days are hand-buildable; historical cell-level rates are not public. |
| 4. Is first-close coverage predictive after controlling for days and macro shocks? | Leduc et al. 2025; GAO 2026 | Leduc emphasizes first revisions and annual plots; coverage is endogenous; response differs from collection | Monthly C1, expected C1 from calendar/mode, revision panel, volatility controls | **High descriptively; medium causally.** Use calendar-predicted coverage or residualization, not naïve slopes. |
| 5. Which late reporters drive tails—size, industry, government, mode, or pay frequency? | Copeland; Handwerker; 2015 BLS report | Old micro study; modern public evidence says no stable pattern but gives no distributional estimates | Restricted CES paradata and micro reports linked across closings, QCEW size/industry | **Low publicly; high only under BLS restricted access/cooperation.** |
| 6. How do birth–death errors contribute to annual and post-benchmark revision scale by sector? | BLS birth–death tables; Clausen; Decker | Accounting counterfactuals, limited years, aggregate interactions, method changes | Vintage forecasts, realized matched QCEW birth/death flows, sample component, sector cells | **Medium aggregate; low for exact micro decomposition.** Published forecasts and benchmarks exist, realized production internals do not fully. |
| 7. What is the monthly timing of error concealed by the linear wedge? | Dey–Loewenstein quarterly benchmarking; Groen | Method experiment, not a national historical latent monthly-error series | QCEW quarterly/monthly employment vintages, CES vintages, reporting-period bridge | **Medium from current final QCEW; low real-time.** Full detailed QCEW vintage archive is absent. |
| 8. How much uncertainty should be assigned to QCEW as benchmark “truth”? | Robertson; Huff–Gershunskaya; BLS QCEW revision tables | Matched work covers 2004–08; public QCEW revision data are aggregate and begin 2017 | All QCEW publication vintages by industry plus processing/reclassification flags | **Low for long detailed panel; medium for 2017+ aggregate sensitivity.** |
| 9. Did probability sampling reduce revision scale, separately from NAICS and concurrent SA? | BLS controlled SA experiment; Kelter design history; BLS pre/post table | 2000–03 stagger plus common 2003 changes; treatment dates differ by industry; no clean untreated group | Industry conversion dates, SIC/NAICS bridges, SA counterfactual, vintage revisions | **Low for point identification.** A staggered descriptive design is possible; causal separation is not credible without internal parallel estimates. |
| 10. Did quarterly sample implementation (2014) or later NAICS/X-13/TRAMO changes alter variance? | CES history and benchmark articles | Production documentation, no outcome evaluation; interventions cluster with other changes | Sector-specific release vintages, exact implementation map, specification files | **Medium.** Suitable for shrinkage/change-point analysis, not many independent events. |
| 11. Did 2020–22 produce a temporary volatility state or a permanent parameter shift? | Hudson et al.; Tiller–Evans; BLS annual MARs | Seasonal-method papers do not jointly model sample and benchmark tails | NSA/SA vintage panel, interventions, business closures, response/collection, BD forecasts | **High after 2003.** Heavy tails and known interventions are estimable; permanent effects remain weakly identified. |
| 12. Do real appropriations, staffing, and sample capacity predict revision scale? | BLS/OMB/GAO operational records; ASA/COPAFS; no direct study found | Slow, trending, mutually collinear; annual definitions change; programs protect headline series | Harmonized real BA/obligations, realized FTE, CES/QCEW program effort, usable linked coverage | **Medium for a capacity index; low for separate causal slopes.** Substantial hand coding required. |
| 13. What did sequestration and shutdowns do to CES data quality? | BLS episode notices; GAO | Very few events; collection windows and macro conditions differ; some lapses did not close BLS | Event-specific paradata, electronic receipts, staff hours, comparable control series | **Medium as case studies, low as pooled causal estimation.** |
| 14. Can news and noise be separately identified across all CES stages? | Jacobs–van Norden; Kishor–Koenig; Brave et al.; Cajner et al. | Frameworks are GDP, state CES, or two-signal national models; no full national stage system | Joint sector NSA/SA vintages, preliminary/final benchmarks, external signals, covariance restrictions | **High computationally after 2003; identification is assumption-sensitive.** |
| 15. Does the preliminary benchmark optimally predict the final benchmark and subsequent re-benchmarking? | BLS preliminary/final table since 2000 | Official table is descriptive; small annual $`n`$; QCEW vintage evolution not modeled | Preliminary and final benchmark gaps, QCEW aggregate revision vintages, industry shares | **High descriptively, medium inferentially.** Pool sectors and use informative priors. |
| 16. How much do reporting-mode changes alter revision variance? | Clayton; Rosen et al.; 2015 BLS mode summary | Mode assignment is selected; no continuous public mix series | Monthly weighted mode mix and transitions, close-specific receipts, sample characteristics | **Low.** Must be hand-built from archived reports or obtained internally. |

To be explicit about the evidence status: **no directly responsive study was located** for a modern closing-specific NSA/factor variance decomposition, a calendar-day conditional-scale estimate, a CES appropriations/FTE scale relation, or a national four-stage Jacobs–van Norden model. Literature **does exist but is old or on a different estimand** for late reporting (mostly 2001–12), preliminary-estimate efficiency (pre-NAICS), and sample redesign (the confounded 2000–03 transition). COVID seasonal-intervention research exists, but it studies model choice and estimated seasonal movement rather than the resulting stage-specific revision variance.

### 3.2 The three gaps worth pursuing first

**1. Joint NSA/SA scale decomposition at stages 1 and 2.** This is the most direct unanswered production question and is feasible for the modern vintage era. It converts a recurring argument—“seasonals versus late sample”—into an estimable covariance decomposition. It also prevents the mistake of reading SA-minus-NSA MARs as shares. Even an exact 2012–present panel, aligned to archived X-13 inputs, would advance materially beyond Wright and Loya.

**2. Calendar days, collection completeness, and late-report leverage.** Collection days provide high-frequency, sharply varying information that budgets and staffing lack. The goal should be a variance effect, with calendar-predicted C1 separated from realized C1. Public aggregate work is feasible immediately; a restricted-data extension could identify whether large, government, or particular-mode reporters generate the tails.

**3. A four-stage, sector-hierarchical news/noise model with a benchmark operator.** Existing studies either stop at the third release, treat a latest vintage as truth, or model state benchmarks without the full national production sequence. Encoding preliminary/final QCEW information, the backward wedge, forward level propagation, and annual re-seasonalization would make the latent estimand coherent. It also supplies a principled place for birth–death and capacity covariates while exposing which effects are not identified.

Funding and staffing are substantively important but should be second-wave work. With roughly forty annual points and three collinear trends, a model can estimate at most a strongly regularized operational-capacity association. It cannot credibly recover separate structural effects of a budget dollar, an FTE, and a sampled worksite.

### 3.3 Data availability inventory

| Object / driver | Public series or artifact | Vintage coverage and frequency | Earliest usable date | Access route | Constraint / hand build |
|---|---|---|---|---|---|
| Aggregate first/second/third revisions | SA and NSA total-nonfarm monthly changes and pairwise revisions | Monthly; all three releases | Jan. 1979 | [BLS revision table](https://www.bls.gov/web/empsit/cesnaicsrev.htm) | Excludes benchmark, annual SA, and reconstruction revisions; 2003 gap. |
| Full aggregate SA vintages | EMPLOY vintage matrix plus extracted first/second/third changes | Monthly release vintages | Dec. 1964 vintages | [Philadelphia Fed RTDSM](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/employ) | Total nonfarm SA only; definition/method history must be coded. Release dates begin Feb. 1966. |
| Detailed national CES vintages | SA and NSA levels/changes to three-digit NAICS | Every Employment Situation release | First published vintages in May 2003; reference history can be older | [BLS CES vintage files](https://www.bls.gov/web/empsit/cesvindata.htm) | Modern concept only; large ZIP; comments flag some nonstandard revisions. |
| Final and preliminary benchmarks | Total-nonfarm level and percent; preliminary-final difference | Annual; final 1979+; turnkey preliminary table 2000+ | 1979 final; archived preliminary announcements potentially Dec. 1991+ | [CES benchmark technical table 5](https://www.bls.gov/web/empsit/cestn.htm) and [archive](https://www.bls.gov/web/empsit/cesbmkarch.htm) | Preliminary does not revise the series. Pre-2000 announcements and sector tables require annual extraction plus scope/reconstruction flags. |
| Post-benchmark/wedge vintages | Every CES vintage before/after benchmark | Monthly vintage panel; annual intervention | May 2003 detailed | BLS vintage files | Wedge weights and post-March components must be reconstructed from adjacent vintages; separate SA from NSA. |
| QCEW current history | Quarterly/monthly employment by geography, ownership, industry | Quarterly releases; current/revised history | 1975 totals; NAICS detail 1990+ | [QCEW downloadable files](https://www.bls.gov/cew/downloadable-data-files.htm) | Current historical files overwrite prior vintages. 1975–89 NAICS files are ownership totals only. |
| QCEW revision evidence | National/state counts across QCEW releases | Quarterly release sequence | 2017 Q1 | [BLS QCEW revisions CSV](https://www.bls.gov/cew/revisions/) | **No full public QCEW vintage archive with detailed industry.** The new file is aggregate national/state only. |
| ALFRED QCEW | Selected QCEW/BED-related county or MSA series | Series dependent | Series dependent | [ALFRED](https://alfred.stlouisfed.org/) | Coverage is limited to selected county/MSA series and lacks the detailed national industry vintage panel required here. |
| Collection completeness | `CEU00000000C1`, `C2`, `C3`; older annual table | Monthly public series / annual historical snapshots | Jan. 2000 monthly; selected annual rates from 1981 | BLS public database; [series notice](https://www.bls.gov/ces/notices/2024/ces-collection-rates-have-been-added-to-the-online-public-database.htm); Johnson (2016) | Definitions/legacy operations need metadata; aggregate only; do not interpolate the annual pre-2000 observations into monthly values. |
| CES response rate | `CEU05000000RR`, total private at third release | Monthly | 2009 | BLS public database and [OSMR portal](https://www.bls.gov/osmr/response-rates/) | Not interchangeable with collection rate; no close-1 response series. |
| Collection interval | Reference date, release date, closing date, federal holidays | Monthly, hand-derived | Feb. 1966 release dates; robust modern era after 1979 | Philly Fed release-date file; BLS archived schedules | Build business-day count and nonstandard-release flags; verify historical closing rule changes. |
| Seasonal factors/specifications | X-13 specification, input, prior-adjustment, regression, outlier files | Current monthly first/second runs; annual reviews | Availability varies; best in recent years | [CES seasonal files](https://www.bls.gov/web/empsit/cesseasadj.htm) | Historical archive is incomplete; unrounded NSA inputs and exact third/annual runs may require BLS access. |
| Calendar regressors | Four/five-week interval, Easter/religious holidays, Labor Day, outliers | Monthly, series-specific | Production varies by series; 1996 research baseline | BLS annual specs and prior-adjustment files | Extract from files; treatment changes and industry applicability must be coded. |
| Net birth–death forecasts | Published monthly/quarterly forecast tables and benchmark counterfactual table | Monthly forecasts; annual benchmark comparisons | Full modern model 2003/04 | [CES net birth–death page](https://www.bls.gov/web/empsit/cesbd.htm) and FAQ | Older vintage forecasts need archived pages; realized component and sample interaction are not fully public. |
| Sample size/coverage/RSE | Annual benchmark technical tables by sector; historical narrative totals | Annual snapshots | Consistent modern tables roughly 2003+ | Benchmark archive; CES Handbook | Definitions switch among organizations, UI accounts, establishments, worksites, linked sample, and employment coverage; extract/harmonize manually. |
| Size-class/industry allocation | Sample/universe employment by UI size; adequacy/publication review | Annual/current snapshots | Modern probability era | Benchmark technical tables and notices | No turnkey monthly panel; cell allocation and attrition require internal microdata. |
| Collection-mode mix | CATI, EDI, web, TDE/state, fax/other tables | Isolated study/report snapshots | Late 1990s | BLS OSMR papers and 2015 report to Congress | **Hand-build** from archived reports; no public continuous weighted mode series found. |
| Real BLS appropriations | Enacted/actual BA and obligations; GDP implicit price deflator | Annual fiscal year | At least late 1970s from archived CBJs/OMB; consistency varies | DOL/BLS [budget and performance](https://www.bls.gov/bls/budget-and-performance.htm), OMB historical tables, Congress.gov | **Hand-build and reconcile** General Fund plus UTF, enacted versus request/CR annualization, sequestration, rescissions, supplementals, and program boundaries. |
| BLS staffing | Agency/program FTE in CBJs; OPM on-board headcount | Annual FTE; quarterly/September headcount | CBJs extend decades; FedScope practical late 1990s+ | DOL CBJs; [OPM FedScope](https://www.fedscope.opm.gov/) | **Hand-build.** FTE ≠ headcount; contractors and state staff excluded; organization codes change. |
| Shutdown / operational shock flags | Official lapse dates, revised release dates, program notices | Event/month | 1980s archives; key CES events 2013+ | BLS, OPM, CRS, GAO | Code whether BLS itself closed, collection stopped, release delayed, and window extended; do not use a generic federal-shutdown dummy. |
| Congressional party composition | House/Senate majority and appropriations chairs | Congress/fiscal year | Entire study span | House Clerk, Senate historical office, Congress.gov | **Hand-build, descriptive only.** Fiscal-year mapping, divided government, CRs, and endogeneity defeat a simple causal slope. |
| Macro/tail controls | Recession dates, unemployment claims, strikes, severe weather, payroll growth | Monthly/weekly | Long histories | NBER, DOL/BLS, NOAA, FRED/ALFRED | Align to the information set at each release; final macro vintages can leak future information. |
| Private payroll signals | ADP-derived research files/products; Homebase or processor data where licensed | Weekly/monthly, proprietary-vintage dependent | Mostly 2000s+ | FEDS/NBER papers; vendor agreements | Grey/proprietary; selection and definition changes; not a public long-run substitute for CES. |

The central archive limitation deserves emphasis: BLS now publishes aggregate national/state QCEW revision data beginning in 2017, but it still does not expose the full historical, industry-detailed QCEW vintage cube. Current downloadable files are overwritten with revised history. ALFRED does not fill that hole.

## 4. Bayesian research design

This is one implementable specification, not a menu. The core estimation sample is May 2003 onward, when detailed NSA and SA vintages are available under the modern framework. Aggregate total-nonfarm releases from January 1979 are appended as a partially observed upper level, with a single composite pre/post-2003 regime effect. The eleven mutually exclusive BLS supersectors are modeled; total nonfarm is their sum, subject only to published rounding/reconciliation error.

### 4.1 Data layout and fixed stage definitions

Store one long table keyed by:

```text
(sector, reference_month, release_date, vintage_id,
 release_stage, seasonal_status, value_thousands, concept_regime)
```

Construct over-the-month changes with a sparse **release-vintage differencing operator** $`D_{t,j}`$ that subtracts the prior-month value in the same release file. Do not difference separately stored “first,” “second,” and “third” level labels across reference months; those labels occupy different release vintages and would not reproduce BLS's official revision table. The state-space likelihood remains on levels, while all validation against official stage MARs applies $`D_{t,j}`$ first.

Define five measurement states and four transitions:

1. `F`: first release;
2. `S`: second release;
3. `T`: third/final sample-based release;
4. `B`: first release incorporating the **final** comprehensive March-reference benchmark;
5. `M`: a fixed mature horizon—second annual benchmark for NSA and the first frozen five-year horizon for SA, with later reconstructions separately flagged.

The preliminary August/September benchmark is not a sixth employment vintage because it does not revise the series. It is an auxiliary observation of the future final March benchmark discrepancy. Never define `M` as “latest available”: that gives older observations more revision opportunity and induces vintage-age heteroskedasticity by construction.

### 4.2 Latent employment process

Let $`x_{s,t}`$ be concept-consistent NSA employment in thousands for sector $`s`$, and decompose it into a smooth employment state and a mature seasonal/calendar component $`q_{s,t}`$:

```math
x_{s,t}=\ell_{s,t}+q_{s,t},\qquad
q_{s,t}=\gamma_{s,t}+c_{s,t}'\delta_s,
```

```math
\ell_{s,t}=\ell_{s,t-1}+g_{s,t-1}+\lambda_s f_t+\eta^{\ell}_{s,t},
\qquad
g_{s,t}=g_{s,t-1}+\eta^{g}_{s,t},
```

```math
f_t=\phi_f f_{t-1}+\eta^f_t,
\qquad
\gamma_{s,t}=-\sum_{h=1}^{11}\gamma_{s,t-h}+\eta^{\gamma}_{s,t}.
```

Here $`c_{s,t}`$ contains only concept-consistent calendar regressors needed for the NSA process, not the revision-scale drivers. A trigonometric seasonal state can replace the dummy-seasonal recursion if diagnostics favor it, but the production analysis below still uses the observed vintage-specific CES adjustment. Innovation covariance has one common labor-market factor $`f_t`$ plus sector idiosyncratic terms. This prevents a national turning point from being misclassified automatically as eleven independent measurement errors.

Real COVID employment movements must not be absorbed by the measurement-scale regime. Let $`\mathcal C_x=`$ March–June 2020 and use scale-mixture process innovations

```math
\eta^f_t\sim t_5\!\left(0,
\sigma_f\exp\{\kappa_x\mathbf 1(t\in\mathcal C_x)\}\right),\qquad
\eta^\ell_{s,t}\sim t_5\!\left(0,
\sigma_{\ell,s}\exp\{\kappa_x\mathbf 1(t\in\mathcal C_x)\}\right),
\qquad \kappa_x\sim N(\log 50,0.5^2).
```

At the total-equivalent baseline these scales are centered near 150, so the crisis scale is roughly 7,500 and can generate the approximately 20-million April 2020 movement without declaring it revision error. This known crisis transition is distinct from the COVID indicator in the **measurement** log-scale equation. A sensitivity run replaces the multiplier with unrestricted March–June 2020 level-jump states.

For total nonfarm,

```math
x_{0,t}=\sum_{s=1}^{11}x_{s,t},
```

and the reported total receives a tight rounding/reconciliation likelihood. Pre-2003 missing sector vintages are integrated out; they do not fabricate sector histories.

### 4.3 Multi-vintage news and noise

For the monthly closing stages $`j\in\{F,S,T\}`$, use a sequential-revelation form of the Jacobs–van Norden model:

```math
y^N_{s,t,j}
=x_{s,t}-\sum_{k\in\mathcal F(j)}n^N_{s,t,k}+e^N_{s,t,j},
\tag{1}
```

where $`n^N_{s,t,k}`$ is information (“news”) first revealed at a later stage and $`e^N_{s,t,j}`$ is transitory measurement noise. Because $`x`$ is the mature concept-consistent state, $`\mathcal F(F)=\{S,T,B,M\}`$, $`\mathcal F(S)=\{T,B,M\}`$, and $`\mathcal F(T)=\{B,M\}`$. The `B` increment is not a free monthly shock: it is induced by the annual QCEW observation and wedge operator in equations (6)–(7); `M` is induced by the later benchmark/reconstruction operator. For the adjacent monthly closings, equation (1) implies

```math
y^N_{s,t,j+1}-y^N_{s,t,j}
=n^N_{s,t,j+1}+e^N_{s,t,j+1}-e^N_{s,t,j}.
```

Allow one cross-sector news factor at each transition:

```math
n^N_{s,t,k}=a^N_{s,k}q^N_{t,k}+\tilde n^N_{s,t,k}.
```

This is essential: total-nonfarm revision scale is not the root-sum-square of independent sector scales. Loadings are sign-identified by requiring the total-private-weighted mean loading to be positive.

The primary specification sets news increments mutually orthogonal conditional on covariates and the latent economic state, and noise orthogonal to $`x`$ and all news. Closing-stage transitory noise follows one common parsimonious recursion,

```math
e^N_{s,t,j}=\rho_e e^N_{s,t,j-1}+\zeta^N_{s,t,j},
\qquad \zeta^N_{s,t,j}\perp \{x,n,e_{s,t,j-1}\},\quad |\rho_e|<0.8,
```

with innovations independent across sectors conditional on the published-total reconciliation equation. This supplies persistence without an unrestricted Cholesky covariance that could relabel news as noise. Sensitivity models allow stage-specific $`\rho_e`$, a tightly regularized residual Cholesky correlation, and QCEW/third-release error correlation; report how the allocation changes.

### 4.4 The estimand: covariates in scale, with stochastic volatility

Both news and noise are centered Student-*t*, with drivers entering log scale:

```math
n^N_{s,t,k}\sim t_{\nu^n_k}\!\left(m^n_{s,k},\sigma^n_{s,t,k}\right),
\qquad
e^N_{s,t,j}\sim t_{\nu^e_j}\!\left(0,\sigma^e_{s,t,j}\right),
```

```math
\log \sigma^n_{s,t,k}
=\alpha^n_{s,k}+z^n_{s,t,k}{}'\beta^n_{s,k}
+\rho^n_{s,k}h^n_{t,k},
\tag{2}
```

```math
h^n_{t,k}=\phi^n_k h^n_{t-1,k}+\omega^n_k u^n_{t,k},
\qquad u^n_{t,k}\sim N(0,1),
\tag{3}
```

with an analogous, more tightly regularized equation for noise. Thus the target is explicitly the conditional **scale**, while $`m^n_{s,k}`$ absorbs any mean revision. A Gaussian baseline is the nested model obtained by fixing $`\nu`$ large and $`h=0`$.

Use Student-*t*, not Gaussian, in the principal model: 2020–21 and occasional reporting/reconstruction failures are too influential for a Gaussian likelihood. But do not let a single low degree of freedom replace a volatility model. $`\nu_k`$ captures isolated tails; $`h_{t,k}`$ captures clustered revision volatility; covariates explain systematic scale shifts. Report all three.

The standardized covariate mapping is fixed in advance:

| Transition/component | Scale covariates in $`z`$ | Identification source |
|---|---|---|
| `F→S` NSA news | First-window business days; calendar-predicted C1; residual realized C1; usable linked employment coverage/RSE; latent absolute growth and release-time claims volatility; capacity factor; COVID regime | Within-year calendar variation; residual coverage association; sector/time pooling |
| `S→T` NSA news | Incremental C3−C2 collection; C2 level; sample adequacy; latent growth; capacity; COVID | Incremental late receipts; stage contrast |
| Seasonal-factor news at both closings | Change in implied seasonal component; endpoint leverage; four/five-week interval; Easter/Labor Day/outlier flags; annual-specification regime; COVID | Joint NSA/SA identity and archived X-13 inputs |
| `T→B` annual news | Relative net birth–death forecast; sector sample coverage/RSE; third-release response and annual attrition proxy; QCEW revision-precision proxy; capacity factor; recession/turning-point growth; method indicators | Sector-year variation and annual time variation—small effective $`n`$ |
| `B→M` | Updated birth–death difference; preliminary−final benchmark surprise; annual SA specification change; reconstruction/NAICS flag; benchmark discrepancy magnitude | Within-release mechanical operators and later vintage changes |

`Residual realized C1` is the residual from a first-stage collection equation using business days, holidays, scheduled-release features, sector, and month effects. Its coefficient is associational: economic shocks can make reports both late and unusual. Calendar days supply the cleaner variation. Lagged real budget or FTE never receives 12-fold information merely because it is repeated across months; likelihood and cross-validation are blocked by fiscal year.

### 4.5 Joint NSA/SA measurement and an exact variance decomposition

Benchmark and late-sample mechanics are estimated on NSA data. SA vintages are then linked to the **same** release-specific NSA measurement state. In the core model use the mature component $`q_{s,t}=\gamma_{s,t}+c_{s,t}'\delta_s`$ from section 4.2; there is no second free seasonal state. Let $`n^A_{s,t,k}`$ be seasonal-factor news revealed at stage $`k`$:

```math
y^S_{s,t,j}
=x_{s,t}-q_{s,t}
-\sum_{k>j}\left(n^N_{s,t,k}-n^A_{s,t,k}\right)
+e^N_{s,t,j}-e^A_{s,t,j}.
\tag{4}
```

The pair $`(n^N,n^A)`$ has a bivariate Student-*t* scale-mixture representation and an estimated 2×2 correlation at each stage. The factor observation implied by the paired published levels is

```math
A_{s,t,j}=y^N_{s,t,j}-y^S_{s,t,j}
=q_{s,t}-\sum_{k\in\mathcal F(j)}n^A_{s,t,k}+e^A_{s,t,j}.
```

This pins $`q`$ to the NSA seasonal/calendar dynamics and prevents trend, mature seasonality, and factor-news from drifting independently. The covariance captures the fact that revised NSA inputs change the concurrently estimated factor. Where vintage-specific X-13 factors/specifications can be reproduced, their implied adjustment is observed with only rounding error and strongly informs $`n^A`$. Elsewhere the factor path is still observed through NSA-minus-SA, but the algorithm-versus-input split is latent and relies on priors calibrated to the reproducible overlap.

For monthly changes, the published data also give an auditable identity. Let $`d^N_{s,t,j}=D_{t,j}y^N`$ and $`d^S_{s,t,j}=D_{t,j}y^S`$ use the two levels in the same release vintage, and define the implied adjustment $`a_{s,t,j}=d^N_{s,t,j}-d^S_{s,t,j}`$. Then

```math
r^S_{s,t,j\to j+1}=r^N_{s,t,j\to j+1}
-\left(a_{s,t,j+1}-a_{s,t,j}\right),
```

and

```math
\operatorname{Var}(r^S)=
\operatorname{Var}(r^N)+\operatorname{Var}(\Delta_j a)
-2\operatorname{Cov}(r^N,\Delta_j a).
\tag{5}
```

Equation (5), estimated conditionally on sector, stage, and regime, is the requested sample-versus-seasonal decomposition. It is an accounting decomposition of published revision variance, not a claim that factor revision is exogenous to sample revision.

### 4.6 Preliminary/final benchmark as irregular higher-precision observations

For March of benchmark year $`y`$, recover the preliminary implied level $`b^{pre}_{s,y}`$ from the announced discrepancy and the contemporaneous CES estimate, then specify

```math
b^{pre}_{s,y}=x_{s,Mar(y)}+\xi^{Q,pre}_{s,y},
```

```math
b^{final}_{s,y}=x_{s,Mar(y)}+\xi^{Q,final}_{s,y},
\qquad
\sigma^{Q,final}_{s,y}<\sigma^{T}_{s,Mar(y)}\ \text{in prior probability, not by fiat}.
\tag{6}
```

The two QCEW errors are correlated because the final count revises the preliminary source. The 2017+ public aggregate QCEW revision sequence informs the **maturation component** and preliminary/final correlation, not total QCEW measurement error; the latter remains prior- and sensitivity-identified because no external truth series exists. Earlier periods rely entirely on hierarchical extrapolation and bounds. QCEW is a high-precision observation, not an error-free terminal value.

Let $`I_y=(Apr(y-1),\ldots,Mar(y))`$, $`W=(1/12,\ldots,11/12,1)'`$, and $`e_{Mar}`$ select March. For the ordinary backward benchmark window, the published NSA vector is represented by the known affine operator

```math
\mathbf y^B_{s,I_y}
=\left(I-W e_{Mar}'\right)\mathbf y^{preB}_{s,I_y}
+W b^{final}_{s,y}+R_{s,y}\kappa_{s,y}+\epsilon^{round}_{s,y}.
\tag{7}
```

$`R\kappa`$ is nonzero only for documented scope changes or reconstructions. Equation (7) makes the wedge's rank-one covariance explicit. The benchmarked monthly vector is not entered as twelve independent observations.

For April–October after March, preserve the multiplicative sample-link calculation. With $`R^{old}_{s,m}`$ the previously published matched-sample link relative, define the deterministic recursion

```math
E^B_{s,Mar(y)}=b^{final}_{s,y},\qquad
E^B_{s,m}=E^B_{s,m-1}R^{old}_{s,m}+BD^{new}_{s,m},
\quad m>Mar(y),
\tag{8}
```

or $`\mathbf y^B_{s,post(y)}=H_y(b^{final}_{s,y},\mathbf R^{old}_{s,y},\mathbf{BD}^{new}_{s,y})+R^{post}_{s,y}\kappa^{post}_{s,y}+\epsilon^{post}_{s,y}`$, where $`H_y`$ is the known recursively constructed operator. If only job-change contributions rather than link relatives are stored, a linear cumulative operator is valid, but the two representations must not be mixed. If historical link relatives cannot be recovered exactly, infer them from adjacent NSA vintages and published birth–death tables, then propagate that reconstruction error rather than fixing a guessed series. November/December and the new-sample introduction receive explicit indicators. Annual SA reprocessing is applied after (7)–(8) through equation (4).

The fourth transition is made observable by fixing the maturity rule in section 4.1 and extracting that vintage, not by silently treating `B` as terminal. For the same reference-month vector $`J_y`$, define

```math
\mathbf Y^M_{s,J_y}
=\mathcal M_y\!\left(\mathbf x_s,b^{final}_{s,y},b^{final}_{s,y+1},
\mathbf R_s,\mathbf{BD}_s,\mathbf q_s;\mathcal I_y\right)
+\mathbf n^M_{s,y}+\mathbf e^M_{s,y}.
\tag{8a}
```

The known operator $`\mathcal M_y`$ applies the next benchmark's wedge to months that were post-March in `B`, the observed revised birth–death inputs, and—on the SA side—the successive annual adjustment maps; $`\mathcal I_y`$ holds NAICS/reconstruction flags. Residual mature news $`\mathbf n^M`$ and transitory error $`\mathbf e^M`$ use the `B→M` scale equation in section 4.4. The modeled fourth-stage outcome is the observed $`\mathbf Y^M-\mathbf Y^B`$. For right-censored recent years with no fixed mature vintage yet, integrate $`\mathbf Y^M`$ out and issue a posterior predictive distribution rather than shorten the maturity horizon.

### 4.7 Hierarchy across supersectors

For driver $`p`$, transition $`k`$, and sector $`s`$,

```math
\beta_{s,k,p}=\bar\beta_{k,p}+\psi_{k,p}\tilde\beta_{s,k,p},
\qquad \tilde\beta_{s,k,p}\sim N(0,1).
\tag{9}
```

Use non-centered parameterization for (9) by default. With eleven sectors and weak annual information, a centered hierarchy produces a funnel when $`\tau`$ is near zero. Centered parameterization is acceptable only for blocks with strong within-sector monthly information and should be chosen after comparing geometry, not because it is more familiar. Constrain sector deviations to sum to zero when $`\bar\beta`$ is interpreted as the employment-weighted national effect.

Pooling is driver and stage specific. Birth–death and benchmark coefficients may vary substantially across construction, professional/business services, and education/health; collection-day effects may pool more tightly. The posterior for $`\psi_{k,p}`$, not separate unregularized sector regressions, determines that heterogeneity.

### 4.8 Operational capacity and confounding

Do not estimate unrestricted slopes for budget, FTE, headcount, sample size, and mode mix together. In the primary model, use a single annual latent operational-capacity factor $`K_y`$:

```math
v_{r,y}=a_r+\lambda_r K_y+\epsilon_{r,y},
\qquad
K_y=\phi_K K_{y-1}+\eta^K_y,
\tag{10}
```

where $`v`$ comprises log real enacted BLS budget (General Fund plus UTF), appropriation-funded FTE, and—when definition-consistent—CES federal/state/contract workyears. Fix the budget loading positive to orient the factor. Missing indicators are integrated out; a discontinuous measurement definition receives its own intercept. Usable linked employment coverage is **not** also loaded into $`K_y`$: its sector-month residual enters the stage-scale equation directly, avoiding double use of the same variation.

Only one shared $`K_y`$ coefficient is estimated across monthly closing stages, with tightly shrunk stage deviations. That coefficient is learned from about forty annual changes, not hundreds of monthly rows. A sensitivity model replaces $`K_y`$ with separate regularized covariates; if their signs and magnitudes are prior-sensitive, report them as unidentified.

The variation supporting each primary parameter is therefore:

- collection-day effect: within-year calendar shifts, conditional on month/holiday and macro state;
- residual collection effect: month-level departures from calendar prediction, associative;
- seasonal-factor effect: within-reference-month differences between paired NSA and SA vintages plus specification interventions;
- sample adequacy: sector-year differences in linked coverage/RSE, partially confounded with industry volatility;
- birth–death effect: sector-year forecast exposure and method changes, only annual/post-benchmark;
- capacity effect: slow annual deviations and a handful of operational shocks, weakly identified;
- 2003 effect: one composite intercept shift only.

Party control is omitted from the likelihood. A descriptive table may group posterior residual revision scales by appropriating-chamber composition, with no causal label.

### 4.9 Structural change

Use documented interventions, not an unconstrained search for dozens of breaks:

- May 2003: one composite regime shift in aggregate intercept and scale. Do not label it NAICS, probability-sample, or concurrent-SA separately.
- March 2020–December 2021: a pandemic level/tail regime and BLS intervention-treatment indicators; 2022 is a separate normalization regime.
- August 2014 quarterly sample implementation, 2015 X-13, 2017 TRAMO, NAICS releases, 2025/26 birth–death changes, and shutdown-affected releases: candidate scale interventions under strong shrinkage.

Residual drift in a small subset of coefficients follows

```math
\beta_{p,t}=\beta_{p,t-1}+\sqrt{q_p}\,u_{p,t},
```

Use a noncentered normal-gamma prior in the spirit of Bitto and Frühwirth-Schnatter,

```math
\sqrt{q_p}\mid \xi_p^2\sim N(0,\xi_p^2),\qquad
\xi_p^2\sim\operatorname{Gamma}(0.10,40)
```

(shape–rate parameterization), with sensitivity at shapes 0.05 and 0.50. It places a sharp spike near zero while allowing occasional drift; unsupported time variation collapses to a constant. In the base model only the collection-day effect and common scale intercept may drift; allowing every sector-driver coefficient to vary over time is not identified. Discrete unknown change points are reserved for sensitivity analysis because their posterior will otherwise rediscover 2020 and absorb heavy tails.

### 4.10 Prior specification in thousands of jobs

All continuous predictors in (2) are standardized over the modern estimation sample. Binary interventions remain 0/1. Let $`w_s`$ be sector $`s`$'s average 2015–19 share of total-nonfarm employment and use $`w_s^{0.7}`$ to translate aggregate scale centers to sectors. The 0.7 exponent permits less-than-proportional diversification; give it $`N(0.7,0.15^2)`$, truncated to $`[0.3,1]`$, rather than fix it in the final run.

The following priors are deliberately on natural units:

| Parameter | Prior | Rationale |
|---|---|---|
| Baseline total-nonfarm news scale, `F→S` | $`\exp(\alpha^n_{0,S})\sim\operatorname{LogNormal}(\log 40,0.40^2)`$ | Post-2003 official SA/NSA MARs are roughly 33–46; Student-*t* scale is not MAR, so keep broad. |
| Baseline total-nonfarm news scale, `S→T` | $`\exp(\alpha^n_{0,T})\sim\operatorname{LogNormal}(\log 35,0.45^2)`$ | Centers near the 34 SA and 18 NSA MARs while allowing factor covariance. |
| Annual `T→B` March-level news scale | $`\exp(\alpha^n_{0,B})\sim\operatorname{LogNormal}(\log 300,0.60^2)`$ | A 0.2% error at 150 million jobs is 300; accommodates 2009, 2024, and 2025 without making them routine. |
| `B→M` residual news scale after known operator | $`\exp(\alpha^n_{0,M})\sim\operatorname{LogNormal}(\log 75,0.60^2)`$ | Covers QCEW maturation, later benchmark, birth–death, and SA changes after removing mechanical level propagation. |
| Sector baseline scales | $`\log\sigma_{s,k}\sim N(\log(\sigma_{0,k}w_s^{0.7}),0.6^2)`$ | Partial pooling with large allowance for sector heterogeneity. |
| Noise scale | $`\exp(\alpha^e_{s,j})\sim\operatorname{LogNormal}(\log(0.5\sigma_{s,j}),0.7^2)`$ | Begins below news but does not force that ordering. |
| Monthly factor-news scale, total | $`\sigma^A_{0,S},\sigma^A_{0,T}\sim\operatorname{LogNormal}(\log 15,0.70^2)`$ | Loya's 9.6 seam MAR is narrower than full closing-factor news; Wright's broader 81 SD includes benchmark/window effects. |
| Annual factor/specification news, total | $`\sigma^A_{0,B},\sigma^A_{0,M}\sim\operatorname{LogNormal}(\log 75,0.70^2)`$ | Allows large annual/COVID revisions without importing them into routine closings. |
| NSA-news/factor-news correlation | stage-specific Cholesky factor $`L^{NA}_k\sim\operatorname{LKJ}(2)`$ | Weak shrinkage toward zero; sign is not fixed because factor response can reinforce or offset NSA revision. |
| Factor transitory noise | $`\sigma^{eA}_{0,j}\sim\operatorname{HalfNormal}(10)`$ | Small relative to systematic factor news, but not fixed at zero. |
| Mean monthly news | $`m^n_{0,k}\sim N(0,15^2)`$; sector means scale by $`w_s^{0.7}`$ | Official mean revisions are much smaller than MARs; prevents scale from absorbing bias. |
| Mean annual benchmark news | $`m^n_{0,B}\sim N(0,100^2)`$ | Allows persistent benchmark bias without expecting it. |
| Student-*t* degrees of freedom | $`\nu_k=2+\operatorname{Exponential}(0.1)`$ | Finite variance, substantial prior mass on heavy tails, and a path toward near-Gaussian behavior. |
| Final QCEW/benchmark error, total | $`\sigma^{Q,final}_0\sim\operatorname{HalfNormal}(100)`$ | QCEW is more precise than monthly CES in prior probability but demonstrably revised. |
| Additional preliminary-benchmark error | $`\sigma^{Q,pre-extra}_0\sim\operatorname{HalfNormal}(150)`$ | Covers observed preliminary-to-final gaps such as 220 in 2024. |
| Preliminary/final QCEW correlation | $`\operatorname{atanh}(\rho_Q)\sim N(1,0.7^2)`$, truncated to $`0<\rho_Q<0.99`$ | Same underlying QCEW records imply a positive prior; unlike a symmetric LKJ prior, this actually imposes that information. |
| SV persistence | $`(\phi_k+1)/2\sim\operatorname{Beta}(20,1.5)`$ | Persistent but stationary scale states; do not hard-code a random walk. |
| SV innovation | $`\omega_k\sim\operatorname{HalfNormal}(0.15)`$ on log scale | One innovation SD implies about a 16% scale move before persistence. |
| Sector SV loading | $`\rho_{s,k}\sim\operatorname{LogNormal}(0,0.25^2)`$, normalized to weighted mean 1 | Shared volatility with modest sector departures. |
| Common growth factor | $`(\phi_f+1)/2\sim\operatorname{Beta}(5,3)`$; $`\sigma_f\sim\operatorname{HalfNormal}(150)`$ | Allows modest serial persistence and aggregate-equivalent monthly innovations in thousands. |
| Common-factor loadings | $`\lambda_s\sim N(1,0.5^2)`$ with employment-weighted mean fixed at 1 and weighted sign positive | Fixes scale/sign while allowing heterogeneous sector exposure. |
| Cross-stage transitory noise | $`\rho_e\sim N(0,0.20^2)`$ truncated to $`(-0.8,0.8)`$; no residual innovation correlation in the primary model | One shared persistence parameter preserves news/noise identification; $`L_e\sim\operatorname{LKJ}(10)`$ only in sensitivity analysis. |
| Capacity dynamics | $`(\phi_K+1)/2\sim\operatorname{Beta}(10,2)`$; $`\sigma_K\sim\operatorname{HalfNormal}(0.15)`$ | Persistent but stationary annual standardized capacity. |
| Capacity loadings/errors | budget loading fixed to +1; other $`\lambda_r\sim N(1,0.5^2)`$; $`\sigma_{v,r}\sim\operatorname{HalfNormal}(0.25)`$ | Orients and scales $`K_y`$ and acknowledges noisy, nonidentical resource measures. |
| Local-level innovation, total-equivalent | $`\sigma_{\eta^\ell,0}\sim\operatorname{HalfNormal}(150)`$ | Monthly latent employment growth can move materially without being revision error. |
| Slope innovation, total-equivalent | $`\sigma_{\eta^g,0}\sim\operatorname{HalfNormal}(15)`$ | Favors gradual trend-growth change outside documented breaks. |
| Seasonal-state innovation, total-equivalent | $`\sigma_{\eta^\gamma,0}\sim\operatorname{HalfNormal}(50)`$ | Permits evolving NSA seasonality; production factor vintages provide stronger information. |
| Hierarchical sector dispersion | $`\psi_{k,p}\sim\operatorname{HalfNormal}(0.15)`$ | A one-SD sector departure usually changes scale by less than about 35%. |

For the many candidate **global mean** scale coefficients $`\bar\beta_{k,p}`$, use the regularized horseshoe; sector deviations remain governed by $`\psi_{k,p}`$ in (9):

```math
\bar\beta_{k,p}=z_{k,p}\tau^{HS}_k\tilde\lambda_{k,p},\quad
z_{k,p}\sim N(0,1),\quad
\lambda_{k,p}\sim C^+(0,1),
```

```math
\tilde\lambda_{k,p}^2=
\frac{c_k^2\lambda_{k,p}^2}{c_k^2+(\tau^{HS}_k)^2\lambda_{k,p}^2},\quad
\tau^{HS}_k\sim\operatorname{HalfStudent}\text{-}t_3(0,0.10),\quad
c_k\sim\operatorname{HalfNormal}(0.5).
\tag{11}
```

On log scale, 0.25 corresponds to multiplying revision scale by 1.28 for a one-SD covariate change; the slab makes multipliers beyond roughly 2 uncommon but possible. The global 0.10 scale encodes the belief that only a few of the correlated operational variables have independent explanatory power. Report results under global scales 0.05 and 0.20.

In the marginalized production model, represent integer-thousand publication rounding as independent zero-mean error with variance $`1/12`$ thousand-squared—the variance of $`U(-0.5,0.5)`$. This approximation is negligible relative to substantive scales and preserves conditional linear-Gaussian filtering. Run exact interval censoring $`[y-0.5,y+0.5)`$ only as a non-marginal sensitivity check on a reduced panel (or with a suitable non-Gaussian filter); do not claim simultaneous exact censoring and analytic Kalman marginalization. Decimal industry values use their actual publication precision.

### 4.11 Prior predictive checks

Before conditioning on outcomes, simulate complete 1979–2026 vintage panels and require the prior to satisfy broad—not fitted—constraints:

- median monthly first-to-third absolute revisions usually in 20–100 thousand, with nonnegligible but not routine probability above 250 thousand;
- annual March revisions commonly below 0.5%, occasionally near 1%, and rarely multi-million outside a pandemic-sized latent shock;
- sector revisions aggregate to plausible national revisions under the estimated common factor;
- a two-SD change in any ordinary covariate does not imply a 10-fold scale change unless it escapes the horseshoe slab;
- simulated pandemic tails can reproduce 2020–21 without permanently raising the post-2022 baseline;
- wedge simulations produce the exact monotone April–March profile absent explicit reconstruction residuals;
- QCEW error remains smaller than third-release CES error in most, not all, prior draws.

If these fail, adjust priors before inspecting coefficient posteriors. Because the baseline centers use official historical MARs, call this weak empirical calibration and rerun prior sensitivity with doubled log-scale SDs.

### 4.12 Identification assumptions and failure modes

The news/noise decomposition requires more than a state-space label:

1. **Nested information sets.** Later releases contain earlier information plus new receipts/source data. A methodological reconstruction that discards or redefines earlier information violates this and must be an intervention, not ordinary news.
2. **Concept consistency.** Vintages must be bridged to a common NAICS/scope concept. Otherwise classification change is inseparable from latent employment.
3. **Orthogonality restrictions.** Conditional news is orthogonal to the earlier release information set; transitory noise is orthogonal to latent truth and news. Multiple vintages identify covariance components only under these zero restrictions and the cross-stage Cholesky structure.
4. **Benchmark anchor.** Final QCEW is more precise but not exact, and its error distribution is anchored by external QCEW revision evidence. If benchmark and CES errors share payroll-processor or classification error, independence fails; estimate a correlation sensitivity model.
5. **Revision policy is ignorable conditional on flags.** BLS intervention/model changes responding to unusual data can correlate with latent error. COVID outlier decisions and reconstructions therefore receive explicit indicators.
6. **Scale/process separation.** Large true employment moves are allowed in the latent state and enter the scale equation. Otherwise process volatility will be mislabeled measurement volatility.
7. **Missing vintages are ignorable conditional on lapse/reconstruction flags.** The 2003 and 2025 missing or nonstandard releases are not ordinary random missingness.

With only the covariance of two releases, news and noise are not separately identified. The third release, benchmark signal, fixed mature horizon, sector hierarchy, and external QCEW precision create overidentifying information, but conclusions remain conditional on restrictions 1–7. Present posterior decomposition under at least: (a) independent QCEW/CES error, (b) positive shared-error prior, and (c) latest mature vintage treated as another noisy measure rather than truth.

What remains unidentified from roughly forty annual observations is equally important. Separate causal slopes for real appropriations, FTE, contractor workyears, sample size, and party control are not recoverable. The model can estimate one regularized capacity association and show its prior sensitivity. It cannot say how many jobs of revision one additional BLS employee prevents.

### 4.13 Validation and model comparison

#### Posterior predictive checks targeted to the estimand

- stage × sector × regime MAR, median absolute revision, SD, 90th/95th/99th absolute quantiles, skew, and counts above 100/250/500 thousand at total nonfarm;
- rolling 24-, 60-, and 120-month dispersion, with separate 2020–22 panels;
- correlation of $`r^{12}`$ with $`r^{23}`$, and of NSA sample with seasonal-factor revision;
- cross-sector covariance and the distribution of the aggregate-to-sector-rss ratio;
- preliminary-to-final benchmark surprise, March benchmark percentage error, and sequences of same-sign annual errors;
- exact April–March wedge shape and residual deviations after removing the operator;
- coverage of published first/second/third values by posterior real-time intervals.

#### Comparators

Fit, in order: (0) constant-variance Gaussian measurement error; (1) constant-variance Student-*t*; (2) Student-*t* plus SV; (3) full covariate-SV hierarchy. Use PSIS-LOO on independent innovation/reference-month blocks, not raw wedge cells, and inspect Pareto $`k`$. Because adjacent vintages and annual vectors are dependent, supplement PSIS with leave-future-out and leave-one-benchmark-year-out log predictive density. A covariate “wins” only if it improves held-out scale/tail calibration, not merely in-sample likelihood.

#### Pseudo-real-time validation

At each historical release date, truncate every series and covariate to what was then observable. Predict: (i) second release from first; (ii) third from first+second; (iii) the preliminary and final March benchmark; and (iv) the benchmarked monthly path. Use expanding windows with prespecified evaluation blocks 2008–09, 2013, 2019, 2020–22, and 2023–26. Score log predictive density, CRPS, interval coverage/width, absolute-revision calibration, and exceedance Brier scores. Refit or sequentially update; never feed revised collection, QCEW, or macro controls to an earlier pseudo-vintage.

#### MCMC and calibration diagnostics

- four dispersed chains; rank-normalized $`\hat R<1.01`$, bulk and tail ESS >400 for all reported functionals;
- zero divergences after warmup, acceptable energy-BFMI, no systematic maximum-tree-depth hits;
- trace/rank plots for scale, $`\nu`$, SV persistence, capacity, and hierarchy SDs;
- simulation-based calibration on reduced and full synthetic panels; recovery checks specifically for news/noise allocation and wedge rank;
- posterior/prior overlap and likelihood profiles for capacity coefficients;
- sensitivity to centered/non-centered hierarchies, global horseshoe scale, QCEW/CES error correlation, and removal of each crisis episode.

### 4.14 Python/JAX implementation

1. **Polars ingestion.** Read BLS CSV/XLSX-extracted files into the long schema; attach release calendars, stage, concept, benchmark year, and provenance. Use lazy joins and categorical keys. Persist an immutable raw-value table and a separate transformations table so every revision is reproducible.
2. **Operator construction.** Build sparse JAX arrays for aggregation, monthly differencing, the April–March wedge $`W`$, forward cumulative matrix $`G`$, seasonal mapping, and missing-vintage masks. Unit-test the operators against published benchmark articles before fitting.
3. **Gaussian pilot in Dynamax.** Fit the linear-Gaussian version, including irregular benchmark observations and missing data, to debug state dimensions and obtain reasonable mass-matrix initialization.
4. **Full NumPyro model.** Run NUTS in 64-bit JAX. Represent Student-*t* errors as scale mixtures,

   ```math
   \lambda_i\sim\operatorname{Gamma}(\nu/2,\nu/2),\qquad
   \epsilon_i\mid\lambda_i\sim N(0,\sigma_i^2/\lambda_i).
   ```

   Conditional on $`\lambda`$, SV, and nonlinear scale parameters, the employment/seasonal state is linear Gaussian. Use a differentiable Kalman filter/smoother—Dynamax components or a custom `jax.lax.scan`—to integrate $`x,\ell,g,\gamma,f`$ analytically and pass the marginal log likelihood to NumPyro.
5. **Why marginalization matters.** Sampling thousands of highly correlated employment levels jointly with persistent SV and hierarchical scales creates severe posterior curvature and slow NUTS trajectories. Marginalizing the linear state is the single most important efficiency choice. The local Student-*t* mixing variables remain, but are conditionally independent and far less correlated with the regression hierarchy. If that dimension is still prohibitive, compare a small finite normal mixture whose discrete states are summed by forward recursion.
6. **BlackJAX only if warranted.** If NumPyro NUTS mixes poorly for the SV/local-scale block, use BlackJAX for a blocked kernel: NUTS on global coefficients and hierarchy scales, elliptical or specialized updates for Gaussian SV states, and Gibbs updates for gamma mixing scales. Do not write a custom kernel before the marginalized NumPyro benchmark establishes the bottleneck.
7. **ArviZ.** Convert NumPyro draws with log-likelihood grouped by reference month/benchmark year; use ArviZ for rank plots, ESS/$`\hat R`$, PSIS diagnostics, LOO comparison, and posterior predictive summaries. Store seeds, data hashes, JAX/NumPyro versions, and benchmark-operator tests with each run.

### 4.15 What this model adds beyond an event study or heteroskedastic regression

A frequentist event study can estimate an average discontinuity around 2003 or 2020, and a heteroskedastic regression can relate squared revisions to collection rates. Neither, by itself, determines a latent employment path, respects the nested vintage information sets, separates news from transitory noise, treats QCEW as fallible, or propagates the single March discrepancy through the correlated wedge and later re-seasonalization.

The Bayesian model yields posterior probabilities for stage- and sector-specific scale multipliers, tail exceedances, and news shares; shrinks weak capacity effects honestly toward zero; borrows strength across sectors without declaring them identical; and produces a real-time posterior for employment and its future revisions. It can say, for example, “given a short collection window and this sector mix, the probability that the first-to-third revision exceeds 100,000 is *p*, of which *q* is attributable to late-sample news under the stated covariance restrictions.” It cannot turn trending appropriations or party control into causal effects. Its advantage is coherent uncertainty and explicit assumptions, not immunity from identification.

## 5. Annotated bibliography

Source labels distinguish peer review from agency documentation and grey literature. BLS *Monthly Labor Review* articles are labelled agency journal articles; OSMR Statistical Survey Papers and conference papers are not assumed to be peer-reviewed.

### 5.1 Core CES data and production documentation

- **U.S. Bureau of Labor Statistics. 2026. “Nonfarm Payroll Employment: Revisions between Over-the-Month Estimates, 1979–Present.” Updated September 4.** [Data and notes](https://www.bls.gov/web/empsit/cesnaicsrev.htm). *Agency documentation/data.* Primary long-run SA/NSA first-, second-, and third-release revision table; excludes benchmarks, later seasonal revisions, and reconstructions.

- **U.S. Bureau of Labor Statistics. 2026. “CES Vintage Data Information” and “CES Vintage Data.”** [Documentation](https://www.bls.gov/web/empsit/cesvininfo.htm); [files](https://www.bls.gov/web/empsit/cesvindata.htm). *Agency documentation/data.* Detailed published SA and NSA vintage matrices beginning with the May 2003 publication vintage.

- **U.S. Bureau of Labor Statistics. 2026. “Current Employment Statistics: Calculation.” *Handbook of Methods*.** [Chapter](https://www.bls.gov/opub/hom/ces/calculation.htm). *Agency methodology.* Authoritative source for link-relative estimation, closings, benchmarking, wedge-back, reconstruction, and postbenchmark mechanics.

- **U.S. Bureau of Labor Statistics. 2026. “Technical Notes for the CES National Benchmark.”** [Technical notes and tables](https://www.bls.gov/web/empsit/cestn.htm). *Agency methodology/data.* Historical final/preliminary total-nonfarm benchmark table and current sample/benchmark definitions.

- **U.S. Bureau of Labor Statistics. 2026. “CES Seasonal Adjustment Technical Notes.”** [Technical notes](https://www.bls.gov/web/empsit/cesseasadjtn.htm); [input/specification files](https://www.bls.gov/web/empsit/cesseasadj.htm). *Agency methodology/data.* Documents X-13/X-11 production, interval and holiday regressors, outlier treatment, and available inputs.

- **U.S. Bureau of Labor Statistics. 2026. “CES Frequently Asked Questions” and “CES Net Birth–Death Model Frequently Asked Questions.”** [CES FAQ](https://www.bls.gov/web/empsit/cesfaq.htm); [birth–death FAQ](https://www.bls.gov/web/empsit/cesbdqa.htm). *Agency methodology.* Concise authoritative statements on sample scope, closings, benchmark/wedge timing, and which birth–death quantities do and do not revise.

- **U.S. Bureau of Labor Statistics. 2024. “CES Collection Rates Have Been Added to the Online Public Database.” December 6.** [Notice](https://www.bls.gov/ces/notices/2024/ces-collection-rates-have-been-added-to-the-online-public-database.htm). *Agency documentation/data.* Defines C1, C2, C3, and total-private third-release response series; crucially, collection and response are different concepts.

- **U.S. Bureau of Labor Statistics. 2026. “QCEW Data Revisions.”** [Data and documentation](https://www.bls.gov/cew/revisions/). *Agency documentation/data.* Aggregate national/state QCEW vintage changes from 2017; useful for benchmark-error priors but not a detailed historical industry-vintage archive.

- **Federal Reserve Bank of Philadelphia. 2026. “Real-Time Data Set for Macroeconomists: Payroll Employment.”** [EMPLOY documentation and files](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/employ). *Federal Reserve data documentation.* Aggregate SA vintages and release dates extend the pseudo-real-time record before the detailed BLS archive.

- **Federal Reserve Bank of St. Louis. n.d. “Archival Federal Reserve Economic Data (ALFRED).”** [Portal](https://alfred.stlouisfed.org/). *Federal Reserve data portal.* Useful for vintage macro controls, but QCEW holdings do not supply the needed national industry vintage cube.

- **Strifas, Sharon. 2003. “Revisions to the Current Employment Statistics National Estimates Effective May 2003.”** [BLS benchmark article](https://www.bls.gov/ces/publications/benchmark/ces-benchmark-revision-2002.pdf). *Agency documentation.* Primary source documenting the joint NAICS, probability-sample, benchmark, estimator, and concurrent-SA production change.

- **Fett, Nicholas A. 2015. “Comparing with the Original: A Look at Current Employment Statistics Vintage Data.” *Monthly Labor Review*, April.** [doi:10.21916/mlr.2015.12](https://doi.org/10.21916/mlr.2015.12). *Agency journal article.* Explains the vintage archive and separates routine closing revisions from nonroutine revisions.

### 5.2 Seasonality, sample design, collection, and nonresponse

- **Kropf, Jurgen, Christopher Manning, Kirk Mueller, and Stuart Scott. 2002. “Concurrent Seasonal Adjustment for Industry Employment Statistics.” Proceedings of the Joint Statistical Meetings.** [BLS Statistical Survey Paper](https://www.bls.gov/osmr/research-papers/2002/st020110.htm). *Agency conference/research paper.* Controlled projected-versus-concurrent factor experiment; the strongest isolated estimate of the seasonal rule's effect on MAR.

- **Manning, Christopher D. 2003. “Concurrent Seasonal Adjustment for National CES Survey.” *Monthly Labor Review* 126(10): 39–43.** [Article](https://www.bls.gov/opub/mlr/2003/article/concurrent-seasonal-adjustment-for-national-ces-survey.htm). *Agency journal article.* Describes production implementation and the intended revision tradeoff.

- **Morisi, Teresa L. 2003. “Recent Changes in the National Current Employment Statistics Survey.” *Monthly Labor Review* 126(6): 3–13.** [Article](https://www.bls.gov/opub/mlr/2003/06/art1full.pdf). *Agency journal article.* Contemporaneous account of the bundled 2003 redesign and therefore of its identification problem.

- **Loya, Brenda. 2014. “Examining the Impact of Updating More Months of Concurrent Seasonally Adjusted Industry Estimates in the Current Employment Statistics.”** [BLS Statistical Survey Paper](https://www.bls.gov/osmr/research-papers/2014/st140090.htm). *Agency research paper.* Isolates a limited factor-run seam under controlled inputs; does not decompose ordinary closing revisions.

- **Cano, Stephanie, Patricia Getz, Jurgen Kropf, Stuart Scott, and George Stamas. 1996. “Adjusting for a Calendar Effect in Employment Time Series.” Proceedings of the ASA Survey Research Methods Section.** [BLS paper](https://www.bls.gov/osmr/research-papers/1996/st960190.htm). *Agency conference paper.* Primary CES evidence on four- versus five-week reference intervals, including mixed revision and construction results.

- **Hudson, Nicole, Jeannine Mercurio, and Jurgen Kropf. 2022. “The Challenges of Seasonal Adjustment for the Current Employment Statistics Survey during the COVID-19 Pandemic.” *Monthly Labor Review*, May.** [doi:10.21916/mlr.2022.14](https://doi.org/10.21916/mlr.2022.14). *Agency journal article.* Documents split runs, additive outliers, temporary changes, level shifts, and large pandemic seasonal instability.

- **U.S. Bureau of Labor Statistics. 2022. “What's Going on with the Large Revisions in Seasonally Adjusted Employment Estimates?” *Commissioner's Corner*, February 14.** [Blog post](https://www.bls.gov/blog/2022/what-s-going-on-with-the-large-revisions-in-seasonally-adjusted-employment-estimates.htm). *Agency explanatory documentation.* Provides the 2020–21 NSA-versus-SA benchmark examples; not a causal decomposition.

- **Tiller, Richard B., and Thomas D. Evans. 2014. “Seasonal Adjustment and the Great Recession: Implications for Statistical Agencies.”** [BLS Statistical Survey Paper](https://www.bls.gov/osmr/research-papers/2014/st140180.htm). *Agency research paper.* Simulation evidence that continuous recession ramps are less damaging to X-11 than abrupt level shifts.

- **Wright, Jonathan H. 2013. “Unseasonal Seasonals?” *Brookings Papers on Economic Activity* 44(2): 65–110.** [doi:10.1353/eca.2013.0017](https://doi.org/10.1353/eca.2013.0017). *Research article.* Closest published real-time NSA/implied-seasonal revision decomposition, albeit with benchmark and moving-window contamination.

- **Mullins, John P. 2016. “One Hundred Years of Current Employment Statistics—an Overview of Survey Advancements.” *Monthly Labor Review*, August.** [doi:10.21916/mlr.2016.39](https://doi.org/10.21916/mlr.2016.39). *Agency journal article.* Concise integrated history of sampling, collection, estimation, benchmarking, and seasonal adjustment.

- **Johnson, Nicholas. 2016. “One Hundred Years of Current Employment Statistics Data Collection.” *Monthly Labor Review*, January.** [doi:10.21916/mlr.2016.2](https://doi.org/10.21916/mlr.2016.2). *Agency journal article.* Long-run collection-rate snapshots and collection-mode history.

- **Kelter, Laura A. 2016. “One Hundred Years of Current Employment Statistics—the History of CES Sample Design.” *Monthly Labor Review*, August.** [doi:10.21916/mlr.2016.35](https://doi.org/10.21916/mlr.2016.35). *Agency journal article.* Definitive history of the quota/cutoff-to-probability transition and modern allocation.

- **Clayton, Richard L. 1997. “Implementation of the Current Employment Statistics Redesign: Data Collection.” Proceedings of the ASA Survey Research Methods Section: 295–297.** [BLS paper](https://www.bls.gov/osmr/research-papers/1997/st970030.htm). *Agency conference paper.* Historical automated-mode implementation and the strongest numeric before/after revision-reduction claim, without causal identification.

- **Rosen, Richard J., Christopher D. Manning, Louis J. Harrell Jr., and Douglas A. Skuta. 1999. “Data Collection Issues Related to Implementing the Redesigned Current Employment Statistics Survey.”** [BLS Statistical Survey Paper](https://www.bls.gov/osmr/research-papers/1999/st990030.htm). *Agency conference/research paper.* Documents CATI, TDE, fax, EDI, and web architecture and their operational selection.

- **Copeland, Kennon R. 2003. “Reporting Patterns in the Current Employment Statistics Survey.” Proceedings of the ASA Survey Research Methods Section: 1052–1057.** [Paper](https://www.bls.gov/osmr/research-papers/2003/pdf/st030370.pdf). *Agency conference paper.* Closest direct evidence linking collection days, late-report rates, and absolute revisions; small and pre-modern sample.

- **Copeland, Kennon R. 2003. “Nonresponse Adjustment in the Current Employment Statistics Survey.” Proceedings of the Federal Committee on Statistical Methodology Research Conference: 80–90.** [Paper](https://www.fcsm.gov/assets/files/docs/2003FCSM_Copeland.pdf). *Agency conference paper.* Establishes the closing-to-closing link-relative and ignorability logic and warns against treating benchmark gaps as pure nonresponse bias.

- **Copeland, Kennon R., and Richard Valliant. 2007. “Imputing for Late Reporting in the U.S. Current Employment Statistics Survey.” *Journal of Official Statistics* 23(1): 69–90.** [Article](https://www.scb.se/contentassets/ca21efb41fee47d293bbee5bf7be7fb3/imputing-for-late-reporting-in-the-u.s.-current-employment-statistics-survey.pdf). *Peer-reviewed.* Direct imputation experiment finds only modest, statistically weak aggregate revision gains.

- **Robertson, Kenneth W. 2013. “A Working Paper Presenting a Profile of Revisions in the Current Employment Statistics Program.” BLS Working Paper 466.** [Paper](https://www.bls.gov/osmr/research-papers/2013/ec130070.htm). *Agency working paper.* Best national evidence on revision quantiles, receipt timing, and government/education late reporting in 2003–12.

- **Huff, Larry L., and Julie B. Gershunskaya. 2009. “Components of Error Analysis in the Current Employment Statistics Survey.”** [BLS Statistical Survey Paper](https://www.bls.gov/osmr/research-papers/2009/st090050.htm). *Agency research/conference paper.* Matched CES–QCEW decomposition into sampling, nonresponse, reporting, and frame/birth components over five annual cycles.

- **Dixon, John S., and Clyde Tucker. 2016. “Using Quantile Regression to Model Revisions Due to Late Reporting in the Current Employment Statistics Survey.”** [BLS Statistical Survey Paper](https://www.bls.gov/osmr/research-papers/2016/st160080.htm). *Agency research paper.* Distributional late-reporting work under a nonproduction estimator, not a full stage-scale model.

- **Dixon, John S. 2018. “Using Area Characteristics to Model Nonresponse and Late Reporting in the Current Employment Statistics Survey.”** [BLS Statistical Survey Paper](https://www.bls.gov/osmr/research-papers/2018/st180100.htm). *Agency research paper.* Finds weak and unstable area-level prediction of response timing.

- **Kratzke, Diem-Tran. 2013. “Nonresponse Bias Analysis of Average Weekly Earnings in the Current Employment Statistics Survey.”** [BLS Statistical Survey Paper](https://www.bls.gov/osmr/research-papers/2013/st130160.htm). *Agency research paper.* CES-specific linked-data nonresponse analysis, but the estimand is earnings rather than payroll revisions.

- **U.S. Bureau of Labor Statistics. 2015. *BLS Report to Congress on the Current Employment Statistics Methodology for Metropolitan Statistical Areas*.** [Report](https://www.bls.gov/sae/additional-resources/bls-report-to-congress-on-ces-methodology-for-metropolitan-statistical-areas-2015.pdf). *Agency report.* Sample-expansion simulations and mode/receipt summaries; informs sampling precision, not national revision variance.

### 5.3 Benchmarking, QCEW, birth–death, and postbenchmark research

- **Manning, Christopher D., and John R. Stewart. 2017. “Benchmarking the Current Employment Statistics National Estimates.” *Monthly Labor Review*, October.** [doi:10.21916/mlr.2017.25](https://doi.org/10.21916/mlr.2017.25). *Agency journal article.* Authoritative comprehensive-benchmark, reconstruction, wedge, and postbenchmark estimator account.

- **Robertson, Kenneth W. 2017. “Benchmarking the Current Employment Statistics Survey: Perspectives on Current Research.” *Monthly Labor Review*, November.** [doi:10.21916/mlr.2017.27](https://doi.org/10.21916/mlr.2017.27). *Agency journal article.* Treats CES and QCEW as separately produced, fallible sources and summarizes benchmark-research limits.

- **Loewenstein, Mark A., and Matthew Dey. 2017. “A Quarterly Benchmarking Procedure for the Current Employment Statistics Program.” *Monthly Labor Review*, November.** [doi:10.21916/mlr.2017.28](https://doi.org/10.21916/mlr.2017.28). *Agency journal article.* Shows why a small March gap can conceal offsetting monthly errors and motivates quarterly anchors.

- **Dey, Matthew, and Mark A. Loewenstein. 2017. “Quarterly Benchmarking for the Current Employment Survey.” BLS Working Paper 496.** [Paper](https://www.bls.gov/osmr/research-papers/2017/ec170040.htm). *Agency working paper.* Technical derivation and simulations underlying the quarterly-benchmark proposal.

- **Groen, Jeffrey A. 2012. “Sources of Error in Survey and Administrative Data: The Importance of Reporting Procedures.” *Journal of Official Statistics* 28(2): 173–198.** [Article](https://www.scb.se/contentassets/ca21efb41fee47d293bbee5bf7be7fb3/sources-of-error-in-survey-and-administrative-data-the-importance-of-reporting-procedures.pdf). *Peer-reviewed.* Quantifies reporting-procedure, nonresponse, coverage, and sampling contributions to CES–QCEW differences.

- **Robertson, Kenneth W. 2021. “A Better Benchmark Process for the U.S. Current Employment Statistics Survey.”** [BLS Statistical Survey Paper](https://www.bls.gov/osmr/research-papers/2021/st210070.htm). *Agency research paper.* Strongest direct simulation evidence against the linear annual wedge's quarterly error path.

- **Clausen, Nathan. 2011. “Forecasting Birth/Death Residuals on a Quarterly Basis.”** [BLS Statistical Survey Paper](https://www.bls.gov/osmr/research-papers/2011/st110020.htm). *Agency research paper.* Shows that quarterly forecast updates reduced postbenchmark birth–death revisions in most study years.

- **Grieves, Chris, Steve Mance, and Collin Witt. 2023. “Predicting the Effect of Business Births and Deaths on the Current Employment Statistics Survey: Using Sample Information to Minimize Coverage Error.”** [BLS Statistical Survey Paper](https://www.bls.gov/osmr/research-papers/2023/st230010.htm). *Agency research paper.* Modern supersector/pandemic evidence on birth–death forecast RMSE and model averaging.

- **U.S. Bureau of Labor Statistics. 2009. “The Current Employment Statistics Program's 2009 Benchmark Revision.”** [Benchmark article](https://www.bls.gov/ces/publications/benchmark/ces-benchmark-revision-2009.pdf). *Agency documentation.* Primary source for the exceptional −902,000 benchmark and birth–death counterfactual accounting.

- **Brave, Scott A., Charles S. Gascon, William Kluender, and Thomas Walstrum. 2021. “Predicting Benchmarked US State Employment Data in Real Time.” *International Journal of Forecasting* 37(3): 1261–1275.** [doi:10.1016/j.ijforecast.2021.02.006](https://doi.org/10.1016/j.ijforecast.2021.02.006). *Peer-reviewed.* State-level state-space benchmark model; does not encompass all national CES vintage stages.

### 5.4 CES revision econometrics and external signals

- **Neumark, David, and William L. Wascher. 1991. “Can We Improve Upon Preliminary Estimates of Payroll Employment Growth?” *Journal of Business & Economic Statistics* 9(2): 197–205.** [doi:10.1080/07350015.1991.10509845](https://doi.org/10.1080/07350015.1991.10509845). *Peer-reviewed.* Finds predictable preliminary-to-benchmarked revisions; entirely pre-probability-sample, pre-NAICS, and pre-concurrent-SA.

- **Haltom, Nicholas L., Vanessa D. Mitchell, and Ellis W. Tallman. 2005. “Payroll Employment Data: Measuring the Effects of Annual Benchmark Revisions.” *Federal Reserve Bank of Atlanta Economic Review* 90(2): 1–23.** [Article](https://fraser.stlouisfed.org/files/docs/publications/frbatlreview/rev_frbatl_2005_v90no2.pdf). *Federal Reserve research/grey literature.* Real-time evidence that benchmarks dominate persistent level changes; short transitional sample.

- **Phillips, Keith R., and James Nordlund. 2012. “The Efficiency of the Benchmark Revisions to the Current Employment Statistics (CES) Data.” *Economics Letters* 115(3): 431–434.** [doi:10.1016/j.econlet.2011.12.118](https://doi.org/10.1016/j.econlet.2011.12.118). *Peer-reviewed.* Finds cyclical/seasonal mean predictability, not a conditional-scale relationship.

- **Dorfman, Jeffrey H., Wenying Li, and Jingfang Zhang. 2025. “Forecasting Revisions to U.S. Jobs Report Data.” *B.E. Journal of Macroeconomics* 25(2): 825–845.** [doi:10.1515/bejm-2024-0145](https://doi.org/10.1515/bejm-2024-0145). *Peer-reviewed.* Closest hierarchical Bayesian revision forecast; does not formulate all four stages as a news/noise measurement system.

- **Cajner, Tomaz, Leland D. Crane, Ryan A. Decker, Adrian Hamins-Puertolas, Christopher Kurz, and Tyler Radler. 2018. “Using Payroll Processor Microdata to Measure Aggregate Labor Market Activity.” Finance and Economics Discussion Series 2018-005. Board of Governors of the Federal Reserve System.** [doi:10.17016/FEDS.2018.005](https://doi.org/10.17016/FEDS.2018.005). *Federal Reserve working paper.* Establishes the real-time information content of proprietary payroll-processor microdata for CES measurement.

- **Cajner, Tomaz, Leland D. Crane, Ryan A. Decker, Adrian Hamins-Puertolas, and Christopher Kurz. 2022. “Improving the Accuracy of Economic Measurement with Multiple Data Sources: The Case of Payroll Employment Data.” In *Big Data for Twenty-First-Century Economic Statistics*, 147–170. University of Chicago Press/NBER.** [Working-paper version](https://www.nber.org/papers/w26033). *Peer-reviewed book chapter / prior working paper.* Combines CES and proprietary ADP signals in a latent-state model but does not decompose CES production stages.

- **Decker, Ryan A. 2026. “Measuring Payroll Employment: A Note on the Current Employment Statistics.” NBER Working Paper 34924.** [doi:10.3386/w34924](https://doi.org/10.3386/w34924). *Working paper.* Current synthesis of payroll measurement, recent revisions, and reform proposals; not peer-reviewed as of the cutoff.

- **Leduc, Sylvain, Luiz Edgard Oliveira, and Caroline Paulson. 2025. “Do Low Survey Response Rates Threaten Data Dependence?” *FRBSF Economic Letter* 2025-07, March 31.** [Letter](https://www.frbsf.org/research-and-insights/publications/economic-letter/2025/03/do-low-survey-response-rates-threaten-data-dependence/). *Federal Reserve research.* Descriptive first-revision MARs show pandemic exceptionalism and post-2022 normalization despite lower response.

- **Pinheiro, Nicholas R., and Rory G. Quinlan. 2026. “BLS Benchmark Revisions: Is This Time Different?” *Federal Reserve Bank of Cleveland Economic Commentary* 2026-12.** [doi:10.26509/frbc-ec-202612](https://doi.org/10.26509/frbc-ec-202612). *Federal Reserve research.* Long current-minus-third endpoint shows no mean break, but mixes benchmark, SA, and reconstruction stages.

- **Lubik, Thomas A., and Jacob Titcomb. 2026. “How Data Revisions and Uncertainty Affect Monetary Policy.” *Federal Reserve Bank of Richmond Economic Brief* 26-01.** [Brief](https://www.richmondfed.org/publications/research/economic_brief/2026/eb_26-01). *Federal Reserve research.* Reports only a modest long-run decline in initial-to-current revision SD and does not isolate mechanisms.

- **ADP Research Institute and Stanford Digital Economy Lab. 2026. “ADP National Employment Report: Technical Notes and FAQs.”** [Methodology](https://adpemploymentreport.com/). *Private grey literature.* Since 2022 an independent QCEW-weighted payroll measure, not a CES forecast; revisions incorporate late client activity.

### 5.5 General real-time revision and state-space literature

- **Mankiw, N. Gregory, and Matthew D. Shapiro. 1986. “News or Noise? An Analysis of GNP Revisions.” *Survey of Current Business* 66(May): 20–25; NBER Working Paper 1939.** [Working-paper version](https://doi.org/10.3386/w1939). *Agency journal article / working paper.* Canonical information-set orthogonality distinction; not applied to CES production stages.

- **Croushore, Dean, and Tom Stark. 2001. “A Real-Time Data Set for Macroeconomists.” *Journal of Econometrics* 105(1): 111–130.** [doi:10.1016/S0304-4076(01)00072-0](https://doi.org/10.1016/S0304-4076%2801%2900072-0). *Peer-reviewed.* Establishes vintage-correct pseudo-real-time analysis and the RTDSM infrastructure.

- **Aruoba, S. Borağan. 2008. “Data Revisions Are Not Well Behaved.” *Journal of Money, Credit and Banking* 40(2–3): 319–340.** [doi:10.1111/j.1538-4616.2008.00115.x](https://doi.org/10.1111/j.1538-4616.2008.00115.x). *Peer-reviewed.* Demonstrates bias, predictability, heteroskedasticity, and non-Gaussian revisions across macro series.

- **Jacobs, Jan P. A. M., and Simon van Norden. 2011. “Modeling Data Revisions: Measurement Error and Dynamics of ‘True’ Values.” *Journal of Econometrics* 161(2): 101–109.** [doi:10.1016/j.jeconom.2010.04.010](https://doi.org/10.1016/j.jeconom.2010.04.010). *Peer-reviewed.* Direct foundation for the proposed multi-vintage latent-state news/noise model.

- **Kishor, N. Kundan, and Evan F. Koenig. 2012. “VAR Estimation and Forecasting When Data Are Subject to Revision.” *Journal of Business & Economic Statistics* 30(2): 181–190.** [doi:10.1198/jbes.2010.08169](https://doi.org/10.1198/jbes.2010.08169). *Peer-reviewed.* Kalman-filtered estimation with latent true values; motivates state marginalization and real-time forecasting.

- **Fixler, Dennis J., and Bruce T. Grimm. 2005. “Reliability of the NIPA Estimates of U.S. Economic Activity.” *Survey of Current Business* 85(2): 8–19.** [Article](https://apps.bea.gov/scb/pdf/2005/02February/0205_NIPAs.pdf). *Agency journal article.* Template for stage-specific MAR/bias accounting, though NIPA revision schedules differ from CES.

- **Faust, Jon, John H. Rogers, and Jonathan H. Wright. 2005. “News and Noise in G-7 GDP Announcements.” *Journal of Money, Credit and Banking* 37(3): 403–419.** [doi:10.1353/mcb.2005.0029](https://doi.org/10.1353/mcb.2005.0029). *Peer-reviewed.* Shows institutional/country variation in news/noise properties and cautions against importing a generic error model.

- **Clements, Michael P., and Ana Beatriz Galvão. 2023. “Density Forecasting with Bayesian Vector Autoregressive Models under Macroeconomic Data Uncertainty.” *Journal of Applied Econometrics* 38(2): 164–185.** [doi:10.1002/jae.2944](https://doi.org/10.1002/jae.2944). *Peer-reviewed.* Extends latent-revision forecasting to stochastic volatility; lacks CES-specific benchmark and seasonal operators.

### 5.6 Funding, staffing, shutdowns, and statistical-system capacity

- **Robertson, Kenneth W. 2016. “Management of Business Surveys and Production of Economic Statistics: Perspectives from the U.S. Bureau of Labor Statistics Current Employment Statistics Survey.”** [BLS Statistical Survey Paper](https://www.bls.gov/osmr/research-papers/2016/st160050.htm). *Agency research paper.* Most direct CES production-capacity source, including federal/state/contract workyears, spending, and mode mix.

- **U.S. Bureau of Labor Statistics. 2008. “Impact of the 2008 Federal Budget on the Availability and Quality of Data from the Bureau of Labor Statistics.”** [Statement](https://www.bls.gov/bls/budgetimpact.htm). *Agency documentation.* Primary record of CES series eliminations, hiring freeze, and delayed response/revision improvement work.

- **U.S. Bureau of Labor Statistics. 2013. “BLS 2013 Sequestration Information.” March 4.** [Notice](https://www.bls.gov/bls/sequester_info.htm). *Agency documentation.* Documents the greater-than-\$30-million cut and lost programs; does not measure CES revision effects.

- **U.S. Bureau of Labor Statistics. 2014. “Effects of the Fiscal Year 2014 Enacted Budget on BLS Programs.”** [Notice](https://www.bls.gov/bls/budget2014_enacted.htm). *Agency documentation.* Records curtailed QCEW collection and the agency's anticipated small degradation in frame accuracy; an ex ante quality claim, not a measured CES effect.

- **U.S. Bureau of Labor Statistics. 2013. “Frequently Asked Questions: The Impact of the Partial Federal Government Shutdown on the Employment Situation for October 2013.” November 8.** [FAQ](https://www.bls.gov/bls/shutdown_2013_empsit_qa.pdf). *Agency documentation.* Records suspension/delay and BLS's finding of no discernible CES-estimate effect.

- **U.S. Bureau of Labor Statistics. 2019. “What Impact Did the Lapse of Appropriation for Some Federal Agencies Have on January Employment Data?” Updated April 5.** [Notice](https://www.bls.gov/bls/what-impact-did-the-lapse-of-appropriation-for-some-federal-agencies-have-on-january-employment-data.htm). *Agency documentation.* Establishes that BLS was funded during the 2018–19 partial lapse.

- **U.S. Bureau of Labor Statistics. 2026. “Revised News Release Dates Following the 2025 and 2026 Lapses in Appropriations.”** [Calendar](https://www.bls.gov/bls/2025-lapse-revised-release-dates.htm). *Agency documentation.* Defines the nonstandard release schedule needed for event and collection-window coding.

- **U.S. Department of Labor. 2026. “Bureau of Labor Statistics Lapse Plan Summary Overview.” In *DOL Contingency Plan for Operations in the Absence of Appropriations*.** [Plan](https://www.dol.gov/sites/dolgov/files/general/plans/dol-contingency-plan.pdf). *Agency documentation.* Lists 2,055 employees before a lapse, one retained, and 2,054 furloughed; a contingency count, not realized annual FTE.

- **U.S. Department of Labor, Bureau of Labor Statistics. 2016. *Bureau of Labor Statistics: Presidential Transition Brief*.** [Brief](https://www.dol.gov/sites/dolgov/files/general/foia/presidential-transition-docs/bls.pdf). *Agency documentation.* Reconciled FY2009–16 budget-authority/FTE history and resource chart.

- **U.S. Department of Labor, Bureau of Labor Statistics. 2026. “FY 2027 Congressional Budget Justification: Bureau of Labor Statistics.” In *Departmental Management … FY 2027 Congressional Budget Justification*.** [Justification](https://www.dol.gov/sites/dolgov/files/general/budget/2027/CBJ-2027-V3-01.pdf). *Agency budget documentation.* Recent budget-authority/FTE tables; requested and planned figures are not realized staffing.

- **U.S. Office of Personnel Management. n.d. “Federal Workforce Data.”** [Current portal](https://data.opm.gov/); [legacy FedScope](https://www.fedscope.opm.gov/). *Agency administrative data.* BLS on-board headcount, not fiscal-year-average FTE or CES-specific capacity.

- **Bowen, Claire McKay, Constance Citro, Michelle Crosby, Steve Pierson, Nancy Potok, and Zachary Seeskin. 2025. *The Nation's Data at Risk: 2025 Report*. American Statistical Association.** [Report](https://www.amstat.org/docs/default-source/amstat-documents/the-nations-data-at-risk-2025/The-Nations-Data-at-Risk-2025-Report.pdf). *Professional-association grey literature.* Consistent real-resource/staffing trends and risk assessment; descriptive and advocacy-oriented, not causal CES evidence.

- **American Statistical Association. 2026. *The Nation's Data at Risk: Mid-Year Update*.** [Update](https://www.amstat.org/docs/default-source/amstat-documents/FedStatHealth_MidYearUpdate.pdf). *Professional-association grey literature.* Supplies recent OPM headcount snapshots and program-risk updates; definitions must be reconciled to CBJ FTE.

- **Council of Professional Associations on Federal Statistics and Friends of BLS. 2021. *Bureau of Labor Statistics: Priorities for 2021–2025*.** [Report](https://copafs.org/wp-content/uploads/2021/02/BLS_Priorities2021-2025-Final.pdf). *Professional-association grey literature.* Documents real funding erosion and modernization priorities; assertions about future quality are not revision estimates.

- **U.S. Government Accountability Office. 2025. *Expert Views on the Federal Statistical System*. GAO-25-107124.** [Report](https://www.gao.gov/products/gao-25-107124). *Government report.* Expert-forum synthesis on declining response, costs, staffing, modernization, and resilience; views rather than causal estimates.

- **U.S. Government Accountability Office. 2026. *Federal Statistics: Stakeholders Said Jobs Report Generally Meets Their Needs, but Opportunities Exist to Improve Data Quality*. GAO-26-107538.** [Report](https://www.gao.gov/products/gao-26-107538). *Government report.* Current CES/CPS response and performance-target audit; key counterevidence to simple resource-to-quality claims.

- **Saturno, James V., Megan S. Lynch, Bill Heniff Jr., Drew C. Aherne, and Justin Murray. 2025. *Continuing Resolutions: Overview of Components and Practices*. Congressional Research Service R46595, updated March 27.** [Report](https://crsreports.congress.gov/product/pdf/R/R46595). *Government/CRS report.* Supplies CR definitions, counts, and durations for hand-built exposure variables; not a statistical-quality study.

- **Handwerker, Elizabeth Weber. 2025. *Current Employment Survey Monthly Revisions*. Congressional Research Service IF13084, August 14.** [In Focus](https://www.congress.gov/crs_external_products/IF/PDF/IF13084/IF13084.1.pdf). *Government/CRS report.* Current concise summary of closing rates, revision mechanisms, and public concerns.

- **Office of Management and Budget. 2014. *Statistical Programs of the United States Government: Fiscal Year 2014*.** [Report](https://obamawhitehouse.archives.gov/omb/statistical-programs-2014). *Government report.* Contemporaneous cross-agency funding context and anticipated QCEW/CES-frame resource risk.

### 5.7 Bayesian specification and validation methods

- **Bitto, Angela, and Sylvia Frühwirth-Schnatter. 2019. “Achieving Shrinkage in a Time-Varying Parameter Model Framework.” *Journal of Econometrics* 210(1): 75–97.** [doi:10.1016/j.jeconom.2018.11.006](https://doi.org/10.1016/j.jeconom.2018.11.006). *Peer-reviewed.* Justifies shrinkage on TVP innovation variances so unsupported drift collapses toward static coefficients.

- **Piironen, Juho, and Aki Vehtari. 2017. “Sparsity Information and Regularization in the Horseshoe and Other Shrinkage Priors.” *Electronic Journal of Statistics* 11(2): 5018–5051.** [doi:10.1214/17-EJS1337SI](https://doi.org/10.1214/17-EJS1337SI). *Peer-reviewed.* Defines the regularized horseshoe used for weak, collinear operational drivers.

- **Papaspiliopoulos, Omiros, Gareth O. Roberts, and Martin Sköld. 2007. “A General Framework for the Parametrization of Hierarchical Models.” *Statistical Science* 22(1): 59–73.** [doi:10.1214/088342307000000014](https://doi.org/10.1214/088342307000000014). *Peer-reviewed.* Canonical centered-versus-noncentered parameterization guidance.

- **Vehtari, Aki, Andrew Gelman, and Jonah Gabry. 2017. “Practical Bayesian Model Evaluation Using Leave-One-Out Cross-Validation and WAIC.” *Statistics and Computing* 27(5): 1413–1432.** [doi:10.1007/s11222-016-9696-4](https://doi.org/10.1007/s11222-016-9696-4). *Peer-reviewed.* Basis for PSIS-LOO, here applied to independent innovation/reference-month blocks rather than wedge cells.

- **Vehtari, Aki, Daniel Simpson, Andrew Gelman, Yuling Yao, and Jonah Gabry. 2024. “Pareto Smoothed Importance Sampling.” *Journal of Machine Learning Research* 25(72): 1–58.** [Article](https://www.jmlr.org/papers/v25/19-556.html). *Peer-reviewed.* Current PSIS method and Pareto-$`k`$ diagnostic reference.

- **Talts, Sean, Michael Betancourt, Daniel Simpson, Aki Vehtari, and Andrew Gelman. 2018. “Validating Bayesian Inference Algorithms with Simulation-Based Calibration.” arXiv:1804.06788.** [Preprint](https://arxiv.org/abs/1804.06788). *Methodological preprint.* Supports recovery tests for news/noise allocation, wedge rank, and scale coefficients.

- **Gelman, Andrew, Daniel Simpson, and Michael Betancourt. 2017. “The Prior Can Often Only Be Understood in the Context of the Likelihood.” *Entropy* 19(10): 555.** [doi:10.3390/e19100555](https://doi.org/10.3390/e19100555). *Peer-reviewed.* Supports calibration through prior predictive job-count revisions rather than abstract coefficient scales.
