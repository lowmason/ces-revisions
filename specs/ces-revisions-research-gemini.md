# U.S. CES Payroll Revisions: Literature, Open Gaps, and a Bayesian State-Space Research Design

## Executive summary

The strongest empirical fact is that the **magnitude of early CES revisions changed materially around the 2003 methodological break, but not uniformly across revision stages**. BLS’s 1979–present revision archive shows that mean absolute revisions to seasonally adjusted total-nonfarm monthly change fell from 48,000 to 33,000 jobs for first-to-second estimates comparing 1979–2003 with 2003–present, while second-to-third revisions increased slightly, from 29,000 to 34,000. For not-seasonally-adjusted data, first-to-second mean absolute revisions were essentially unchanged at 46,000, whereas second-to-third revisions collapsed from 53,000 to 18,000. First-to-third absolute revisions fell from 61,000 to 51,000 SA and from 83,000 to 53,000 NSA. Because probability sampling, NAICS conversion, and concurrent seasonal adjustment arrived almost simultaneously, this is a **regime break, not causal identification of any single reform**. Recent years also show that improvement was not monotonic: SA total-nonfarm first-to-third mean absolute revisions were roughly 51,000 in 2023, 48,000 in 2024, and 58,000 in 2025.

The best causal-like evidence concerns **concurrent seasonal adjustment**. BLS’s preimplementation experiment found first-to-second SA mean absolute revisions of 34,000 under concurrent adjustment versus 37,000 under projected factors, and first-to-third revisions of 36,000 versus 48,000. Concurrent adjustment also reduced first-print-to-final-benchmarked revision magnitudes for total nonfarm and eight of nine SIC divisions. By contrast, I found **no published CES study that cleanly decomposes every monthly SA revision into an NSA late-sample component plus a seasonal-factor component over a long vintage panel**.

For collection, the institutional mechanism is clear: first estimates have only 10–16 business days of collection; responses continue for two subsequent months. BLS research confirms that late responders generate revisions and that their behavior is heterogeneous by industry and establishment size, while lower aggregate response rates do not mechanically imply larger nonresponse bias. But I found **no published study exploiting calendar-driven variation in collection days to estimate its effect on revision variance**.

Evidence is considerably weaker for **funding, staffing, and sample size**. Shutdowns provide discrete operational shocks—CES collection was suspended during October 1–16, 2013, and the 2025 lapse again disrupted BLS operations—but there is no credible published estimate tying appropriations or BLS FTEs to CES revision scale.

The most valuable research project is therefore a **joint NSA/SA, multi-vintage Bayesian state-space model** that treats revision *scale* as the estimand, separates news from noise, incorporates the annual QCEW benchmark and deterministic wedge-back, partially pools effects across supersectors, and places collection interval, response/collection rates, birth-death error, sample measures, and agency-capacity variables directly in the log variance.

## Literature review

### Revision stages and what is actually known about magnitude

The literature is asymmetrical. Monthly first/second/third revisions are unusually well documented because BLS publishes a continuous total-nonfarm revision table going back to 1979 and, since May 2003, detailed CES vintage matrices. Annual benchmarks are also documented year by year. What is largely missing is a published longitudinal analysis that treats **revision dispersion itself as the dependent variable across all four stages**, especially at the supersector level. BLS’s vintage documentation explicitly distinguishes monthly revisions, annual benchmarks, and historical reconstructions, and preserves all published vintages from May 2003 forward.

| Revision stage | Principal mechanical sources | Evidence on magnitude over time |
|---|---|---|
| **First → second** | Newly arriving matched-sample reports; revised current/prior NSA estimates; concurrent recomputation of SA factors | TNF SA mean absolute revision fell from **48k in 1979–2003 to 33k in 2003–present**; NSA stayed at **46k → 46k**. This is suggestive that the post-2003 improvement was not simply “more complete NSA sample at first closing.” |
| **Second → third** | Additional late reports; revised matched sample; concurrent SA factor update | TNF SA mean absolute revision moved **29k → 34k**, but NSA fell **53k → 18k**. The divergence is strong evidence that seasonal-factor movement can be quantitatively important at this stage, although the two MAEs cannot simply be subtracted because the components covary. |
| **Third → annual benchmark** | QCEW universe realignment; accumulated sampling/nonresponse/reporting/frame and birth-death error; annual model changes; five-year seasonal readjustment; classification/reconstruction changes | BLS reports the annual benchmark differences, but I found no published study estimating a stable time series of the *scale* of this stage conditional on its separate sources. Recent birth-death forecast errors have been important enough to motivate explicit model modifications. |
| **Post-benchmark / wedge-back** | Linear distribution of the March benchmark discrepancy over intervening months; annual reseasonalization; historical reconstructions and reclassification | This stage is partly deterministic rather than new information. CES benchmarking wedges the March discrepancy back over the preceding April–March interval, while the annual process also re-seasonally adjusts recent history. Published work describes the mechanics but does not, to my knowledge, routinely report the variance share attributable separately to wedge, reseasonalization, and reconstruction. |

The 2003 breakpoint deserves unusual caution. The BLS archive’s own split at 2003 is operationally convenient but coincides with completion of probability-sample implementation, the transition to NAICS, and concurrent seasonal adjustment. Probability sampling was developed beginning in 1995, tested in 1998–2002, and phased into national production between June 2000 and June 2003; the final services conversion occurred in 2003. Thus, a simple “post-2003” variance coefficient is meaningful as a **composite regime effect**, not as an estimate of any one reform.

Recent observations are also inconsistent with a story of smooth secular convergence toward ever-smaller revisions. BLS’s published annual summaries show SA TNF first-to-third mean absolute revisions of about 51,000 in 2023, 48,000 in 2024, and 58,000 in 2025; the latter years also had negative average revisions. Those are still below the pre-2003 long-run mean absolute first-to-third revision of 61,000, but the recent increase is material.

The preliminary March 2026 benchmark is also a useful reminder that annual benchmark errors are episodic: the published preliminary total-nonfarm adjustment was only **−79,000, or −0.1 percent**, much smaller than several recent benchmark episodes. That observation should not be read as a permanent restoration of small benchmark errors.

### Seasonality

**What is known.** The 2003 switch to concurrent seasonal adjustment is the one candidate driver for which BLS conducted an explicit counterfactual comparison. Manning’s BLS experiment reran historical vintages under both projected-factor and concurrent approaches. For March 1998–March 2002, first-to-second mean absolute revisions to SA TNF monthly change were 37,000 under projected factors and 34,000 under concurrent adjustment; first-to-third revisions were 48,000 versus 36,000. Looking from the first print to the eventual benchmarked series, concurrent adjustment reduced mean absolute revisions for TNF and eight of nine major SIC divisions; the TNF statistic fell from roughly 78,000 to 65,000.

This is stronger evidence than a before/after 2003 comparison because both methods were evaluated on the same underlying historical sample observations. The result also refutes a common ex ante concern that allowing the seasonal factor to move every month necessarily makes monthly revisions larger. In the experiment it did not; first-to-third revision dispersion fell materially.

Current CES production calculates new seasonal factors each month for the current and preceding two revised months, using X-13ARIMA-SEATS. Models are selected annually; during the year the model specification is held fixed while data and factors update concurrently. At the annual benchmark, CES reviews specifications and re-seasonally adjusts the most recent five years.

Calendar treatment is more sophisticated than a generic month-of-year seasonal factor. CES uses REGARIMA variables for the **4-versus-5-week survey-interval effect**, arising because the pay periods containing the 12th can be four or five weeks apart; it also treats floating Good Friday/Easter and Labor Day effects where significant. Hours and earnings additionally have length-of-pay-period and 10/11-day treatments. These adjustments are methodologically plausible channels through which the calendar can alter vintage-to-vintage SA changes, but I found no paper estimating how many thousands of jobs in the historical TNF revision distribution are attributable specifically to revisions of these calendar coefficients.

The COVID episode generated a genuine seasonal-adjustment structural problem rather than merely extreme residuals. BLS research subsequently changed CES outlier treatment: through the 2021 benchmark the annual procedure had relied on additive/point outliers; beginning in 2022 the production review could also retain temporary-change and level-shift outliers after COVID disrupted normal seasonal patterns. The current technical documentation explicitly identifies April–July 2020 and other pandemic observations in example specifications.

The **SA-versus-NSA revision statistics are especially informative**. After 2003, first-to-second MAE is 33,000 SA versus 46,000 NSA, but second-to-third MAE is 34,000 SA versus only 18,000 NSA. This cannot be interpreted as “seasonality contributes 16,000” because absolute values are nonlinear and NSA-sample and factor revisions are correlated. It does establish, however, that the second-to-third SA revision cannot reasonably be treated as just a late-sample revision with negligible seasonal-factor contribution.

**What is asserted without adequate evidence.** It is often asserted that concurrent adjustment itself explains the post-2003 decline in CES revision magnitude. The contemporaneous experiment supports a contribution, but the historical breakpoint bundles concurrent adjustment with probability sampling and NAICS. Likewise, “COVID made seasonal adjustment unreliable” is too broad: BLS documented concrete outlier/model problems and changed treatment, but that does not by itself quantify the fraction of total CES vintage revision variance caused by seasonal adjustment.

**What is contested or unresolved.** I found current documentation that CES uses X-13ARIMA-SEATS and historical documentation of X-12-era procedures, but **no CES study that isolates the effect of the X-12→X-13 migration on revision scale**. More importantly, I found no long-sample published decomposition of

```math
r^{SA}_{t,k}
=
r^{NSA}_{t,k}
+
\Delta_k\!\left(SA_t-NSA_t\right),
```

where the second term is the vintage change in the published seasonal correction. The necessary SA and NSA vintages are available post-May 2003, making this an unusually tractable gap.

### Collection interval

**What is known.** BLS begins CES collection after the establishment’s reference pay period containing the 12th has ended. The first estimate is produced after only **10–16 business days** of collection, depending on that month’s Employment Situation release calendar. Collection then continues for approximately two months; the second and third estimates incorporate the newly arriving sample.

That institutional timing makes late reporting the obvious principal source of first/second/third **NSA** revisions. Dixon and Tucker explicitly define late reporters as establishments missing the initial publication but entering a subsequent vintage and distinguish them from permanent nonresponders, who can bias an estimate but do not mechanically create a later revision because they never report. Their 2010–14 linked CES/QCEW analysis found substantial heterogeneity in late-response behavior by industry, firm size, and month, with some patterns switching over time—a plausible reason revision direction and magnitude are difficult to predict.

Copeland and Valliant’s earlier Journal of Official Statistics work on imputing for late CES reporting is the closest peer-reviewed methodological precursor: it asks whether information on historical reporting patterns can improve estimates before all respondents arrive. Dixon and Tucker explicitly situate their work against it.

Response rates and revision variance must not be conflated. Huff and Gershunskaya decomposed CES–QCEW discrepancies for 2002–06 into sampling, potential nonresponse, reporting, and frame-related components. They found that response rates were lower in the later frames they studied, but **no evidence that the lower response rate was associated with larger nonresponse bias**; two low-response industries showed large reporting errors, but the authors called that relationship tentative. They also warn that CES and QCEW inhabit somewhat different reporting/measurement systems, so a QCEW-based decomposition is not a literal decomposition into pure CES errors.

BLS’s current response-rate program repeats this warning: lower response rates do not have a one-to-one relationship with nonresponse bias, although response and collection-rate trends remain useful indicators of survey operations. Public BLS charts now provide recent first-preliminary and later-closing CES collection/response information, with a visible decline in first-closing performance over portions of the post-2016 period rather than a stable rate.

**The central unexploited design is the calendar.** The number of usable collection days varies because the 12th moves across weekdays and because the scheduled Employment Situation release moves across the calendar. That generates sharp month-to-month variation while agency staffing, appropriations, and sample design move slowly. I found **no published CES paper estimating a variance model in which first-closing revision scale is a function of actual collection days**, nor one using scheduled collection days as a quasi-exogenous source of variation for first-closing completeness. The Dixon–Tucker study examines who reports late; it does not identify the effect of giving the survey an additional collection day.

The 2013 shutdown provides a much larger interruption. All BLS operations, including CES collection, were suspended October 1–16. BLS noted that this interrupted data that normally would have contributed to the August final, September second-preliminary, and October preliminary estimates. This is a clean operational event but a poor standalone causal experiment because it is a single episode and coincides with an economically and politically unusual period.

The 2025 lapse in appropriations, from October 1 through November 12, provides another unusual interruption and materially disrupted the release calendar. It therefore belongs in the revision panel as a separately flagged operational regime rather than being treated as an ordinary low-response month.

### BLS funding

The literature is much weaker here than public commentary suggests.

**What is known.** BLS budget justifications provide annual budget authority/request information, programmatic changes, and agency or activity-level FTE information; these documents are the appropriate raw material for constructing a real BLS-resource series. The public BLS budget-and-performance archive is preferable to newspaper characterizations of whether a budget was “cut,” because requested, enacted, annualized-CR, and actual obligations are different objects.

Budget stress can visibly affect statistical output. The 2013 shutdown halted BLS survey operations for more than two weeks and delayed the September Employment Situation; BLS reported that only three of roughly 2,400 agency employees worked full time during the shutdown. Sequestration-era BLS materials also document discontinuation or contraction of statistical activities outside CES, demonstrating that fiscal constraints can reach production rather than merely administrative overhead. The key point for this project is that I found no documentation showing that the **national CES sample itself was cut in 2013 in a way suitable for attributing a change in payroll revision variance to sequestration**.

The 2018–19 partial federal shutdown should **not** be coded as a BLS shutdown analogous to 2013 or 2025. The lapse affected agencies whose FY2019 funding had not been enacted; Labor-HHS-Education funding, including DOL/BLS, had already received full-year appropriations. Thus it can be useful as a negative-control political episode but not as a direct CES collection interruption. CRS’s contemporaneous description of the FY2019 shutdown emphasizes that it applied to only the unfunded portion of government.

The 2025 lapse, in contrast, did affect BLS operations and releases, so a modern sample contains at least two genuine BLS shutdown shocks—2013 and 2025—plus one prominent shutdown that did not directly close BLS.

**What is not known.** I found no peer-reviewed or BLS working paper that estimates

```math
\operatorname{Var}(r^{CES}_t)\quad\text{as a function of real BLS appropriations},
```

or that causally links a one-percent funding change to first-, second-, benchmark-, or post-benchmark CES revision dispersion. GAO’s recent work on the federal statistical system reports expert concern about resources and statistical-system capacity, but this is institutional evidence, not identification of a funding-to-error production function.

This distinction matters because annual real appropriations have severe identification problems. They trend; they are jointly determined with staffing and program choices; Congress may appropriate in response to perceived statistical needs; inflation measures matter; and a one-year appropriation can affect systems with multiyear lags. A bivariate correlation between “real BLS budget” and absolute CES revision would therefore be close to uninterpretable.

Continuing resolutions should be coded separately from nominal appropriations. A CR may constrain hiring, contracting, and new initiatives without changing the eventually enacted annual appropriation. But again, I found no published CES-quality study that translates CR exposure into measured revision dispersion.

Partisan control is still weaker. Chamber composition is potentially useful **descriptive stratification** for budget histories, but it is almost perfectly entangled with time, macroeconomic regimes, presidents, congressional cycles, and budget institutions. Without a much stronger design—such as an externally generated appropriations discontinuity affecting BLS but not neighboring statistical activities—it should not be assigned a causal coefficient.

### Sample

There is strong historical evidence that sample **design** mattered; much less evidence that incremental sample **size** predicts monthly revision magnitude.

The old CES was effectively a collection of cut-off/quota samples concentrated on large establishments. The 1994 ASA review, prompted in part by concern following the unusually large March 1991 benchmark revision, recommended probability sampling. BLS’s subsequent internal work found that old sample units were on average roughly a decade older than establishments in the universe and that simulated estimates from the continuing quota sample could produce very large benchmark discrepancies.

Development of the probability design began in 1995, was completed in 1997, and was production-tested from 1998 through 2002. Implementation was phased in from June 2000 through June 2003: wholesale trade first, then mining/construction/manufacturing, then transportation/utilities/retail/FIRE, then services.

The current private-sector design is a stratified simple random sample of worksites clustered by UI account, with strata defined by state, industry, and establishment size and allocation chosen to minimize variance subject to geographic reliability objectives. Larger firms are sampled at higher rates but design weights correct for unequal selection probabilities. Government is handled differently because CES obtains much higher direct coverage.

As of the March 2025 benchmark documentation, the sample covered roughly **119,000 businesses and government agencies and 622,000 worksites**, representing approximately **26 percent of universe payroll employment**. Coverage is heterogeneous across sectors; this is precisely why raw worksite count is an inferior sample-quality covariate to effective sample employment, matched-sample weight concentration, or a design-effect analogue.

Sample allocation has been revisited after the original redesign. BLS describes reallocations that attempted to improve national efficiency without allowing unacceptable deterioration in smaller-state precision. This means sample composition is endogenous to expected variance: finding that industries receiving more sample have larger raw revisions could simply reflect BLS allocating more sample to intrinsically difficult domains.

Collection mode also evolved substantially. By the mid-2010s CES was using centralized computer-assisted telephone collection, EDI for large reporters, web reporting, touchtone data entry, fax, and state channels, with electronic collection increasingly important. A mode effect on revision scale is plausible because modes differ in timeliness, respondent burden, automated edits, and establishment composition. But Dixon and Tucker explicitly identify response mode as a potentially relevant omitted covariate rather than presenting an estimate of its effect.

The birth-death model belongs partly under “sample” because it exists to handle employment change in units that cannot yet enter a lagged sampling frame. CES’s matched-sample estimator adds a net birth-death forecast to sample growth. This is a qualitatively different error source from ordinary sampling variance.

The birth-death model’s vulnerability at turning points is well documented. BLS’s recent research found that historical seasonal ARIMA relationships broke down during extreme labor-market changes and that approaches using current sample information substantially improved prediction, including over Great Recession and pandemic episodes. Persistent post-2020 benchmark errors led BLS to modify the model, first in post-benchmark processing and subsequently in monthly production.

A particularly important institutional fact for revision attribution is that the monthly birth-death forecast is generally **held fixed through the two subsequent monthly revisions**. Thus first→second and second→third differences are overwhelmingly sample/seasonal-adjustment phenomena, not repeated birth-death reforecasting. Birth-death error emerges most clearly when QCEW information arrives at benchmark and when new post-benchmark forecasts are generated.

The benchmark itself is not an error-free oracle. Groen’s linked CES/QCEW work shows that QCEW and CES can differ because of establishment reporting procedures, QCEW imputation, payroll frequency, and establishment concepts. At the aggregate level, the largest CES/QCEW growth differences occurred around November and January, and roughly three-fourths of monthly growth differences in his analysis were attributable to reporting differences. This argues strongly against fixing benchmark measurement variance at zero in a state-space model.

### Staffing and broader revision-model literature

BLS staffing is conceptually distinct from appropriations, but empirically they are difficult to separate. OPM/FedScope supplies agency employment counts, while BLS congressional budget justifications supply budgeted or actual FTE measures. Headcount and FTE should not be conflated: the relevant production-input variable is probably FTE, and more specifically program-level FTE, but consistent CES-specific FTE histories are not published as a clean monthly series. BLS budget documentation is therefore likely to require manual extraction.

The 2013 shutdown shows the extreme margin clearly—almost the entire agency ceased work and CES collection stopped—but says little about the marginal effect of 20 or 50 ordinary-year FTE. I found **no CES paper and no convincing federal-statistical-system paper estimating the causal elasticity of revision variance with respect to statistical-agency staffing**. GAO material is useful evidence that expertise, staffing, technology, and resource constraints matter institutionally; it is not a measured CES quality production function.

That absence contrasts sharply with the mature generic literature on revisions.

Mankiw and Shapiro introduced the canonical **news-versus-noise** distinction for preliminary GNP: under a “news” model, early estimates are efficient forecasts and subsequent revisions reflect newly arriving information; under “noise,” preliminary releases contain measurement error that later revisions remove.

Croushore and Stark made real-time macroeconomic analysis operational by constructing vintage datasets and showing that empirical conclusions can depend on which vintage is used. ALFRED subsequently institutionalized vintage access for many U.S. series, although source-agency archival limitations constrain what it can preserve.

Aruoba showed that revisions in major U.S. macro series frequently violate simple “well-behaved” assumptions: revisions can have nonzero means, substantial magnitude, and forecastable components.

Jacobs and van Norden's contribution is particularly relevant here. They formulate multiple data vintages in a state-space system in which latent “true” values and different kinds of revision errors are jointly identified from cross-vintage covariance restrictions, allowing mixtures of news and noise rather than forcing the analyst to choose one ex ante.

Kishor and Koenig likewise use a state-space/real-time approach for macroeconomic series subject to revision and explicitly include employment-related applications, demonstrating that modeling the revision process can improve real-time estimation and forecasting. But their objective is not a CES institutional decomposition into first/second/third/benchmark/wedge stages, nor do they model collection days, response rates, seasonal-factor changes, birth-death error, or BLS capacity as **scale covariates**.

Fixler and Grimm’s BEA work studies GDP revisions, predictability, and turning-point reliability; later BEA state-space work explicitly treats the “true” state of the economy as unobserved. Those papers are conceptually close to the proposed latent-payroll framework but operate on national accounts, whose source-data and annual-revision structure differ substantially from CES.

Faust, Rogers, and Wright show that GDP revision behavior differs considerably across G-7 statistical systems and that a meaningful part of some countries’ revisions was predictable. This is important caution against assuming that a revision is automatically “news.”

**Bottom line:** I found no published application that takes the Jacobs–van Norden framework all the way to a **CES-specific, four-stage, sector-hierarchical, variance-regression model with NSA/SA decomposition and institutional covariates**. Kishor–Koenig is the nearest macroeconometric relative; BLS’s late-response, seasonal-adjustment, benchmark, and birth-death papers provide the institutional pieces. They have not been assembled into one statistical model.

## Gap analysis

### Open questions

| Open question | Closest existing work | Why it falls short | Data needed | Public-source feasibility |
|---|---|---|---|---|
| **Does an extra first-closing collection day reduce the scale of first→second revisions?** | BLS vintage documentation; Dixon–Tucker late-response analysis. | No calendar-based identification; estimand is late-response bias/behavior rather than conditional revision variance. | Exact reference-period completion date, release date, usable business days, first/second vintages. | **High.** Calendar and releases can be hand-built; CES TNF/supersector vintages exist post-2003. |
| **Does first-closing collection/response rate predict first→second revision dispersion conditional on collection days?** | BLS response-rate dashboard; Huff–Gershunskaya. | Existing work emphasizes bias, not revision scale; older error decomposition predates COVID and much of the response decline. | Closing-specific response/collection rates, vintage revisions, sector information if available. | **Medium-high after 2016; lower earlier.** |
| **How much of each SA monthly revision is NSA sample revision versus seasonal-factor revision?** | Manning concurrent-SA experiment; BLS SA technical files. | Counterfactual method comparison is pre-2003; no long-run exact vintage decomposition found. | Paired SA and NSA vintage levels; archived specification/outlier files. | **High for aggregate decomposition from May 2003; medium for reproducing exact X-13 factors.** |
| **Has revision scale itself changed continuously since 2003 rather than just pre/post-2003?** | BLS 1979-present TNF revision table. | Mostly descriptive averages; no stochastic-volatility or covariate model, and no comparable pre-2003 full industry vintage panel. | Monthly revisions by stage, sector, date. | **Very high post-2003; TNF only is much longer.** |
| **How much annual benchmark error is attributable to birth-death forecast error rather than ordinary sample/frame/reporting error?** | BLS birth-death research and benchmark notes. | BLS documents mechanisms and recent forecast errors but does not provide a long, sector-by-year variance attribution dataset. | Historical BD forecasts, benchmark “actual” net births/deaths, QCEW/CES March gaps, sector. | **Medium.** Substantial hand assembly from annual benchmark releases. |
| **What fraction of post-benchmark historical revision is mechanical wedge-back versus new seasonal factors versus classification/reconstruction?** | CES benchmarking and SA documentation. | Mechanics are documented; empirical variance decomposition is not. | Third-vintage path, benchmarked NSA path, benchmarked SA path, reconstruction flags. | **High post-2003**, except some reconstruction metadata need manual coding. |
| **Did sample composition/coverage changes alter revision dispersion?** | CES sample-design history and probability-sample redesign literature. | Strong evidence that the quota design was biased; little evidence relating marginal modern sample-size/composition changes to vintage revision scale. | Sample worksite counts, covered employment, weights/design effects, sector allocations. | **Medium-low.** Published annual snapshots exist; a monthly design-history series does not. |
| **Does collection mode affect revision scale?** | CES collection-history documentation; Dixon–Tucker notes mode as a possible covariate. | No longitudinal causal/variance study found; mode is highly confounded with establishment size. | Mode shares by month/sector and closing, establishment-size mix. | **Low-medium publicly; likely much better with internal microdata.** |
| **Do real BLS appropriations predict CES revision scale?** | BLS CBJs; GAO statistical-system resource assessments. | **No CES study found.** Funding is annual, trending, endogenous, and collinear with staffing/sample. | Enacted real budget, CES budget if separable, CR/shutdown indicators, revision panel. | **Medium data feasibility; low causal identification.** |
| **Does BLS staffing predict revision scale?** | CBJ/FedScope administrative staffing series; shutdown evidence. | **No CES study found.** Headcount/FTE distinction and program assignment are major problems; slow variation. | Agency and preferably CES-specific FTE, vacancies/turnover, revision panel. | **Medium for agency totals; low for CES-specific historical FTE.** |
| **Can the 2003 decline be separately attributed to NAICS, concurrent SA, and probability sampling?** | Manning; CES sample-design history. | The interventions overlap almost exactly, and the detailed vintage archive begins at the break. | External counterfactuals or internal parallel-production datasets. | **Low with public data.** This is fundamentally underidentified. |
| **Did COVID permanently change the revision-generating process?** | BLS COVID seasonal-adjustment research; recent birth-death research. | Mechanism-specific studies exist, but no unified post-COVID revision-scale model. | 2003–present vintages plus pandemic outlier/model flags, BD errors, response metrics. | **High.** |
| **Are benchmark revisions “news” or “noise,” and does the mixture vary by sector/regime?** | Jacobs–van Norden; Kishor–Koenig; BLS CES/QCEW error studies. | Generic framework exists; CES-specific cross-vintage identification has not been carried through all benchmark stages. | Full vintage triangle plus benchmark observations. | **High post-2003.** |

The first three gaps are the highest return.

The **collection-calendar design** is attractive because it has much stronger identifying variation than funding or staffing. The number of days between the end of the reference period and first closing moves sharply month to month for mechanical calendar reasons. Conditional on month-of-year, release conventions, holidays, strike/pandemic flags, and sector, it offers a credible source of variation for the *scale* of first→second revisions. It also directly tests the operational story that everyone assumes but, as far as I can determine, nobody has estimated in this form.

The **NSA/seasonal-factor decomposition** is nearly as valuable because it is feasible with public vintage data and addresses an ambiguity in nearly every discussion of CES revisions. The post-2003 aggregate statistics already show dramatically different NSA and SA second→third behavior. An exact joint-vintage decomposition would tell you whether recent large SA revisions reflect late sample, unstable seasonal factors, or covariance between them.

The third high-value gap is **benchmark/birth-death attribution**. BLS has already established that the old birth-death forecasting relationship became unreliable after 2020 and has changed the model. A sector-hierarchical model can ask the quantitatively sharper question: how much of the tail risk in March benchmark revisions is predictable from ex ante birth-death exposure and sector composition?

Funding and staffing are worth retaining, but primarily as **capacity covariates and sensitivity analyses**, not as headline causal results. Forty years of monthly observations do not create forty years of independent information about an annual trending budget series.

## Data availability inventory

| Driver / outcome | Public series and vintage coverage | Frequency / earliest useful date | Access and constraints |
|---|---|---|---|
| **TNF first/second/third revisions** | BLS table of revisions between monthly estimates, including SA and NSA summary statistics | Monthly; **1979–present** | Direct BLS revision table. Best long history for aggregate scale; does not provide the full sector vintage state needed for all models. |
| **Full CES employment vintage panel** | Published vintage matrices for TNF and supersectors; detailed industries also available | Full publication vintages from **May 2003** | BLS CES vintage files. For earlier reference months, May 2003 is effectively the earliest preserved vintage rather than the original historical release. |
| **CES NSA/SA pairs** | Both SA and NSA levels and changes are retained in CES vintage products | Monthly, May 2003 onward for true vintage reconstruction | Core input for exact sample-versus-seasonal revision decomposition. |
| **Seasonal-adjustment specifications** | Current X-13 specification files, prior-adjustment files, recent outliers; historical annual specification tables | Monthly current files; annual historical material from the post-2003 era, with archival coverage uneven by file type | BLS CES seasonal-adjustment page. Exact reproduction requires unrounded production NSA inputs and prior adjustments; published rounded data will not reproduce production values exactly. |
| **Benchmark revisions** | Annual CES benchmark articles and technical tables; preliminary March benchmark before final incorporation | Annual; archived modern benchmark articles cover the NAICS/probability-sample era | BLS benchmark archive. The **March** label is the benchmark reference month, not necessarily publication month. March 2026 preliminary TNF adjustment was −79k. |
| **QCEW levels** | Monthly employment within quarterly UI census files; long historical series | Quarterly files, monthly employment observations | **Critical constraint:** BLS does not provide a comprehensive public archive of successive QCEW vintages. Current/revised historical QCEW is observable; the contemporaneously available QCEW vintage generally is not. |
| **ALFRED QCEW** | Vintage functionality exists for selected FRED/ALFRED QCEW series | Series-specific | Insufficient substitute for a national detailed-industry QCEW vintage archive: coverage is concentrated in selected county/MSA aggregates and lacks the full industry-by-vintage panel required here. ALFRED also notes that vintage depth depends on what source agencies preserved. |
| **Birth-death forecasts/errors** | Monthly model forecasts plus annual benchmark articles and recent “actual versus forecast” discussions | Modern probability-sample era; strongest documentation in recent years | Requires hand-building a consistent historical table across changing benchmark pages. Current BLS research provides model-change dates and recent forecast-error evidence. |
| **First/second/final collection or response rates** | BLS establishment-survey response/collection-rate dashboards | Recent monthly history, prominently **2016 onward** in current charts | Public aggregate CES series; historical and sector-level detail before the dashboard period is patchier and may require archived releases or BLS special tabulations. |
| **Collection interval** | No ready-made regression covariate; exactly constructible from reference-period/release calendars | Monthly; potentially much longer than the 2003 vintage panel | **Hand-build.** Compute business days from pay-period/reference timing to first close/release, add federal holidays and shutdown days. The 10–16-day institutional range is documented. |
| **CES sample size / coverage** | Benchmark technical notes, Handbook, budget documents, historical MLR articles | Annual or redesign snapshots | **Hand-build.** Current benchmark gives businesses, worksites, employment coverage and industry coverage; no clean monthly historical design panel. |
| **Collection-mode mix** | Historical CES data-collection articles and budget documents | Irregular snapshots | **Hand-build.** EDI/web/CATI/touchtone/fax shares are not supplied as a consistent monthly public time series. |
| **NAICS/design/redesign events** | Handbook and MLR historical documentation | Event dates | Easy categorical coding; 2000–03 probability-sample phase-in and 2003 NAICS/concurrent-SA break are documented. |
| **BLS nominal appropriations / budget authority** | Congressional Budget Justifications, DOL appropriations documents, BLS budget archive | Annual fiscal year | **Hand-build.** Prefer enacted/actual authority to presidential request; deflate consistently. CR exposure and rescissions should be separate variables. |
| **BLS real appropriations** | Derived, not directly supplied | Annual | Deflate nominal authority using CPI-U, GDP deflator, or federal compensation/input deflator; report sensitivity because the choice changes “real” agency-resource trends. |
| **Shutdown/CR exposure** | 2013 and 2025 BLS operational notices; federal appropriations histories | Event/day | High feasibility. Distinguish 2013 and 2025 direct BLS shutdowns from the 2018–19 lapse that did not directly close BLS. |
| **BLS staffing** | OPM FedScope headcount plus CBJ FTE tables | Monthly/quarterly headcount where available; annual FTE | **Hand-build.** FedScope headcount is not FTE; historical CES-program FTE is much harder than agency total. CBJs are the likely authoritative annual source. |
| **Party composition** | Congressional composition / appropriations committee control | Congress/fiscal year | Easy to construct, but use **descriptively only**. The time-series variation is too low-frequency and confounded for a credible standalone causal slope. |

A practical implication follows from the QCEW constraint: the research can model the **benchmark observation BLS actually published** and its effect on CES vintages, but it cannot, from public data alone, perfectly recreate the information set BLS had at every historical QCEW production vintage. That limitation is especially relevant if one tries to label all benchmark revisions as CES error; some portion reflects revisions or measurement imperfections in the administrative benchmark itself. Groen and Huff–Gershunskaya provide strong reasons not to impose a zero-error QCEW assumption.

## Bayesian research design

### Estimand and panel

Use monthly observations at the CES supersector level $`s=1,\ldots,S`$, with total nonfarm generated by summing sector latent levels rather than estimated as an independent unrelated process. Let $`t`$ denote reference month and $`k\in\{1,2,3\}`$ the first, second, and third sample-based vintages.

The primary estimands are:

```math
\sigma^{R}_{s,t,12},
\qquad
\sigma^{R}_{s,t,23},
\qquad
\sigma^{R}_{s,Y,3B},
\qquad
\sigma^{R}_{s,t,BP},
```

the conditional scales of the first→second, second→third, third→benchmark, and benchmark→post-benchmark revision processes.

The main scientific parameters are **effects on log scale**, not conditional means:

```math
\log \sigma^{R}_{s,t,k}
=
\alpha_{s,k}
+
X_{t,k}'\beta_{s,k}
+
h_{s,t,k}/2 .
```

All continuous $`X`$'s should be standardized before fitting, but effects should also be transformed back into interpretable scale ratios. For example, report

```math
\exp(\beta_{\text{days}})
```

as the multiplicative change in revision scale associated with a one-standard-deviation increase in collection days and then translate that back to the effect of one additional business day.

Mean equations should include only enough structure to absorb persistent signed bias. The central claim is about variance/scale. A conventional regression of signed revisions on collection days would answer the wrong question.

### Latent true path and news/noise decomposition

Let $`x_{s,t}`$ be latent employment in thousands of jobs under a consistent CES/QCEW target concept. Evolve it as

```math
x_{s,t}=x_{s,t-1}+g_{s,t},
```

```math
g_{s,t}=\mu_{s,t}+q_{s,t},
\qquad
\mu_{s,t}=\mu_{s,t-1}+\kappa_s u_{s,t},
```

with robust innovations for $`q`$ during extreme episodes. A local trend is preferable to forcing a stationary AR process on employment levels.

For each reference month, use a **triangular information-arrival representation**:

```math
y^{NSA}_{s,t,1}
=
x_{s,t}
-
N^{12}_{s,t}
-
N^{23}_{s,t}
-
N^{3B}_{s,t}
+
Z_{s,t,1},
```

```math
y^{NSA}_{s,t,2}
=
x_{s,t}
-
N^{23}_{s,t}
-
N^{3B}_{s,t}
+
Z_{s,t,2},
```

```math
y^{NSA}_{s,t,3}
=
x_{s,t}
-
N^{3B}_{s,t}
+
Z_{s,t,3},
```

and, when the March administrative benchmark arrives,

```math
B_{s,Y}
=
x_{s,\mathrm{Mar}(Y)}
+
Z^B_{s,Y}.
```

Here $`N^{12}`$, $`N^{23}`$, and $`N^{3B}`$ are information increments not available at the prior vintage—“news”—whereas $`Z_k`$ are vintage-specific noise components.

This parameterization yields, for example,

```math
y_2-y_1
=
N^{12}
+
(Z_2-Z_1),
```

so the observed revision can contain both genuinely new information and correction of noise.

Identification follows the Jacobs–van Norden logic. The key restrictions are:

```math
\operatorname{Cov}
\left(
N^{12}_{t},y_{1,t}\mid \mathcal I_1
\right)=0,
```

and analogously for later news increments: information that genuinely arrives after a vintage must be orthogonal to the efficient estimate based on the earlier information set. Noise errors satisfy

```math
\operatorname{Cov}(Z_{k,t},x_t)=0
```

conditional on modeled covariates. Cross-vintage covariance patterns and the higher-precision benchmark then identify the news/noise mixture.

These assumptions are substantive, not innocuous. They break if BLS analysts knowingly smooth preliminary data toward priors in a way correlated with subsequent information; if late responders are systematically predictable from the preliminary estimate and that predictability is omitted; if benchmark reporting error covaries with CES survey error; or if revisions include deterministic methodological reconstructions that are mistakenly treated as stochastic information arrivals. The latter is why wedge-back and seasonal-factor revisions should enter explicitly rather than being dumped into $`N^{3B}`$.

### Heavy tails and stochastic volatility

Use Student-$`t`$, not Gaussian, innovations as the primary specification:

```math
N^{j}_{s,t}
\sim
t_{\nu^N_j}(0,\sigma^N_{s,t,j}),
```

```math
Z_{s,t,k}
\sim
t_{\nu^Z_k}(0,\sigma^Z_{s,t,k}).
```

Set

```math
\nu_j = 2+\tilde\nu_j,\qquad
\tilde\nu_j\sim\operatorname{Exponential}(1/8),
```

giving substantial prior mass to economically meaningful heavy tails while ensuring finite variance.

The reason is empirical rather than aesthetic: CES revision behavior contains recession, pandemic, shutdown, strike, and benchmark episodes that a Gaussian variance model will tend to explain by inflating persistent volatility for many surrounding months. BLS’s own pandemic seasonal-adjustment work treats several 2020 observations as special outlier regimes.

Residual time variation in revision scale is

```math
h_{s,t,k}
=
\rho_k h_{s,t-1,k}
+
\tau_k\epsilon_{s,t,k},
\qquad
\epsilon\sim N(0,1),
```

with $`h`$ constrained to have mean zero so that $`\alpha+X\beta`$ retains an interpretable baseline.

Use

```math
\rho_k\sim\operatorname{Beta}(20,2)
```

after mapping to $`0<\rho<1`$, and

```math
\tau_k\sim\operatorname{HalfNormal}(0.15).
```

A 0.15 innovation standard deviation on log volatility allows meaningful but not explosive month-to-month movement.

### Covariates by revision stage

The variance design should reflect the institutional timing rather than putting every driver indiscriminately into every stage.

For **first→second**:

```math
X_{t,12} =
[
\text{collection days},
\text{first-close collection rate},
\text{reference-calendar variables},
\text{shutdown},
\text{sample coverage},
\text{capacity variables}
].
```

Collection days and first-closing rate receive the strongest ex ante role.

For **second→third**:

```math
X_{t,23} =
[
\text{additional collection days},
\text{second-close completeness},
\text{seasonal-factor instability},
\text{sample coverage},
\text{capacity}
].
```

Given the very small post-2003 NSA second→third MAE relative to SA, a seasonal-factor state should be especially relevant here.

For **third→benchmark**:

```math
X_{Y,3B}=
[
\text{BD forecast exposure/error predictor},
\text{sector birth/death intensity},
\text{sample coverage},
\text{frame age},
\text{QCEW reporting/imputation indicators},
\text{capacity}
].
```

Funding or staffing should not be expected to act mechanically at only this stage; their coefficients represent broad production capacity.

For **post-benchmark**:

```math
X_{t,BP}
=
[
\text{wedge loading},
\text{seasonal specification change},
\text{historical reconstruction flag},
\text{NAICS revision flag}
].
```

Do not interpret the wedge coefficient as “error.” It is primarily a propagation mechanism.

### Annual benchmark and wedge-back

Treat the March QCEW benchmark as an irregularly timed, higher-precision observation rather than literal truth:

```math
B_{s,Y}
\sim
t_{\nu_B}
\left(
x_{s,\mathrm{Mar}(Y)},
\sigma_{B,s,Y}
\right).
```

This respects evidence that CES and QCEW differ in reporting procedures and that QCEW itself contains imputation and reporting imperfections.

Let the raw discrepancy at March be

```math
\delta_{s,Y}
=
B_{s,Y}
-
y^{NSA}_{s,\mathrm{Mar}(Y),3}.
```

The benchmarked historical path has a known wedge operator:

```math
\widetilde y_{s,t}
=
y^{NSA}_{s,t,3}
+
\lambda_{t,Y}\delta_{s,Y}
+
R_{s,t,Y},
```

where $`\lambda_{t,Y}`$ increases linearly from the month after the preceding March benchmark to $`1`$ at the current March benchmark, and $`R`$ is a residual reconstruction/reclassification component. BLS describes the wedge as distributing the discrepancy across the intervening April–March period.

This is preferable to treating all post-benchmark changes as independent observations. It recognizes that many historical revisions are algebraic consequences of one March discrepancy.

The model should then attach annual seasonal readjustment separately:

```math
y^{SA,PB}_{s,t}
=
\widetilde y^{NSA}_{s,t}
+
A^{PB}_{s,t},
```

where $`A^{PB}`$ is the post-benchmark seasonal correction.

### Joint NSA/SA modeling

This is the cleanest way to separate sample and seasonal-factor revisions.

For every observed vintage define the published seasonal correction in job levels,

```math
A_{s,t,k}
=
y^{SA}_{s,t,k}-y^{NSA}_{s,t,k}.
```

This identity is valid regardless of whether X-13 internally used additive or multiplicative decomposition; it is merely the difference between the two published employment levels.

Then model NSA employment and the seasonal-correction process jointly:

```math
y^{NSA}_{s,t,k}
=
x^{NSA}_{s,t}
+
e^{NSA}_{s,t,k},
```

```math
A_{s,t,k}
=
a_{s,t}
+
e^{SAF}_{s,t,k}.
```

Consequently,

```math
r^{SA}_{k\rightarrow k+1}
=
r^{NSA}_{k\rightarrow k+1}
+
\left(A_{k+1}-A_k\right)
```

**exactly** in published job units.

That gives three directly reportable quantities for each stage and month:

```math
r^{sample}=r^{NSA},
```

```math
r^{factor}=\Delta A,
```

```math
2\operatorname{Cov}(r^{sample},r^{factor})
```

as the covariance contribution to total SA revision variance.

This is superior to subtracting SA and NSA mean absolute revisions, which has no variance-decomposition interpretation.

The seasonal-factor scale equation should include 4/5-week indicators, moving-holiday treatments, specification-change indicators, annual-reestimation months, COVID outlier regimes, and an indicator for whether the relevant detailed series is directly or indirectly adjusted. BLS documentation provides these model classifications.

### Sector hierarchy

For every scale coefficient,

```math
\beta_{s,j,k}
=
\bar\beta_{j,k}
+
\omega_{j,k}z_{s,j,k},
\qquad
z_{s,j,k}\sim N(0,1).
```

Use

```math
\omega_{j,k}\sim\operatorname{HalfNormal}(0.15)
```

for most drivers and

```math
\omega_{j,k}\sim\operatorname{HalfNormal}(0.30)
```

for birth-death and benchmark-related coefficients, where sector heterogeneity has stronger institutional justification.

This lets construction, leisure/hospitality, professional/business services, and other sectors differ without estimating unrelated coefficients from only a few dozen annual benchmark observations.

Use a **non-centered parameterization by default**:

```math
\beta_{s,j,k}=\bar\beta_{j,k}+\omega_{j,k}z_{s,j,k}.
```

With only about two dozen annual benchmarks post-2003 and potentially weak sector heterogeneity, centered hierarchical parameterizations will create classic funnel geometries. Centering can be reconsidered only for strongly identified monthly parameters after pilot fits.

### Structural change

Do **not** estimate three separate 2003 treatment effects for “NAICS,” “concurrent SA,” and “probability sample.” Public data do not identify them. Use one composite regime break:

```math
I(t\ge \text{May/June 2003})
```

for the long TNF history, while using Manning's same-data counterfactual experiment as external evidence specifically about concurrent seasonal adjustment.

For the full sector model, whose reliable vintage panel begins in May 2003, the main structural-change treatment should instead concern 2020–22 and documented redesign/model events.

Use a shrinkage time-varying coefficient formulation:

```math
\beta_{j,t}
=
\beta_{j,t-1}
+
\sqrt{\theta_j}\,\eta_{j,t},
```

with strong shrinkage on $`\theta_j`$. The Bitto–Frühwirth-Schnatter principle is appropriate: most coefficients should behave as constant unless the data supply strong evidence for genuine temporal drift.

Add explicit intervention points for March/April 2020 and the 2022 seasonal-outlier methodology regime rather than demanding that a generic random walk absorb the pandemic. Birth-death methodology change dates receive analogous indicators.

This gives a model with both known change points and sparse residual parameter drift—not a free time-varying-parameter model that can explain any observation after the fact.

### Priors on natural job units

For supersector monthly latent growth innovations, start with

```math
\sigma_{g,s}
\sim
\operatorname{HalfStudent}\text{-}t_4(0,100)
\quad\text{thousand jobs}.
```

For smooth trend innovation,

```math
\kappa_s
\sim
\operatorname{HalfNormal}(20)
\quad\text{thousand jobs/month}.
```

For baseline supersector revision scales:

```math
\alpha_{s,12}
\sim
N(\log 12,\;0.8^2),
```

```math
\alpha_{s,23}
\sim
N(\log 10,\;0.8^2).
```

Ten roughly independent sector errors of 10–12 thousand imply an aggregate scale in the several-tens-of-thousands range, consistent with the post-2003 TNF revision magnitudes without hard-coding the observed TNF MAE.

For annual benchmark news:

```math
\alpha_{s,3B}
\sim
N(\log 50,\;1.0^2),
```

which is deliberately much wider. The prior allows sector benchmark discrepancies ranging from single-digit thousands to well over 100,000 before Student-$`t`$ tails are invoked.

For QCEW benchmark measurement noise:

```math
\log \sigma_{B,s}
\sim
N(\log 5,\;0.8^2).
```

That encodes “higher precision” rather than “perfect truth.”

For high-frequency pre-specified drivers—collection days and collection rate—use

```math
\bar\beta_{j,k}\sim N(0,0.25^2).
```

A coefficient of $`0.25`$ means a one-SD covariate change multiplies revision scale by $`e^{0.25}\approx1.28`$; coefficients large enough to double scale, $`|\beta|\gtrsim0.69`$, are possible but not casually expected.

For the larger candidate set—appropriations, staffing, sample coverage, mode shares, calendar controls, redesign flags—use a **regularized horseshoe**:

```math
\beta_j\sim
N(0,\tau^2\tilde\lambda_j^2),
```

```math
\lambda_j\sim C^+(0,1),
```

with a slab scale

```math
c=0.7
```

on log scale and global $`\tau`$ calibrated to an a priori expectation that roughly three to five of the candidate scale predictors have non-negligible effects.

The 0.7 slab corresponds roughly to allowing factor-of-two effects without permitting completely uncontrolled variance explosions.

### Confounding and what is actually identified

The best-identified coefficient should be **collection interval**. Its identification comes from within-year, month-to-month calendar variation after controlling for seasonal month, holidays, and release conventions.

First-closing response/collection rate is also high frequency but less cleanly identified: economic stress may simultaneously lower response and increase true employment heterogeneity. Therefore it should be interpreted primarily as a predictive operational covariate unless collection days or other scheduling variables are used to explain its exogenous component.

Seasonal-factor revision is directly observed after the NSA/SA transformation above, so its decomposition is algebraically identified rather than inferred from a regression coefficient.

Birth-death exposure is identified mainly from annual cross-sector and cross-year benchmark variation. That gives perhaps twenty-odd modern benchmark years per sector, which is exactly why partial pooling is necessary.

**Appropriations, staffing, and sample size are not separately well identified.** They move slowly, trend together, and share a common technological/productivity trend. Adding 480 monthly observations does not turn 40 annual budget realizations into 480 independent budget experiments.

The main model should therefore detrend each annual capacity series and include:

```math
\{\widetilde{\text{real budget}},
\widetilde{\text{FTE}},
\widetilde{\text{sample coverage}}\}
```

with aggressive regularized-horseshoe shrinkage. Report their posterior dependence and prior sensitivity explicitly. If the marginal posterior for one changes sign when another is omitted, the scientifically correct conclusion is **“not separately identified.”**

A useful derived quantity is a low-frequency agency-capacity index:

```math
C_Y=w_1 B_Y+w_2 FTE_Y+w_3 S_Y,
```

but that should be presented as a descriptive latent factor, not as proof that an additional appropriation dollar causally lowers CES error.

Party control should not enter the main scale regression. Report benchmark/revision distributions grouped descriptively by appropriations-era political configuration only. A causal party coefficient is not defensible from this single national time series.

### Prior predictive checks

Before seeing posterior results, simulate complete 2003–present vintage panels.

The checks should ask whether the priors generate:

1. plausible TNF first→second and second→third revisions, with most ordinary-month absolute revisions in the tens rather than hundreds of thousands;
2. occasional 100,000-plus monthly revisions without assigning large probability to them every year;
3. annual benchmark errors naturally in the hundreds of thousands at TNF after aggregation, with rare much larger values;
4. pandemic-like extreme observations without forcing permanently high volatility;
5. decreasing measurement uncertainty, on average, from first to third vintage;
6. benchmark observations that are more precise but not perfectly equal to latent truth.

The BLS 1979–present distribution can calibrate these checks, but final prior scales should be fixed before evaluating the covariate coefficients used for substantive inference.

### Validation

Posterior predictive checking should target the **estimand**, not generic residual normality.

Compare observed and replicated distributions of

```math
|r_{12}|,\quad |r_{23}|,\quad |r_{3B}|,\quad |r_{BP}|
```

by year, supersector, collection-day quintile, response-rate quintile, and recession/pandemic regime.

Also check the 50th, 75th, 90th, 95th, and 99th percentiles; autocorrelation of absolute revisions; runs of same-sign revisions; cross-sector covariance; SA-minus-NSA revision distributions; and the shape of the wedge around each annual benchmark.

Fit at least these nested specifications:

```math
M_0:\text{constant variance},
```

```math
M_1:\text{stochastic volatility only},
```

```math
M_2:\text{SV + operational/seasonal covariates},
```

```math
M_3:\text{SV + full hierarchical drivers}.
```

Use PSIS-LOO on monthly sector blocks, inspecting Pareto-$`k`$ diagnostics, but do **not** treat naive independent-observation LOO as sufficient for a time series. Supplement it with leave-future-out or contiguous block validation.

The decisive pseudo-real-time experiment is rolling-origin. At each historical release date, expose the model only to vintages and benchmark information that would actually have existed then. Evaluate:

```math
\log p(r_{\text{next revision}}\mid\mathcal I_t),
```

CRPS or interval score for future revision magnitude, calibration of 50/80/95-percent revision intervals, and posterior accuracy for latent payroll after subsequent benchmark information arrives.

This is especially important for birth-death variables: using a later benchmark’s “actual” birth-death error as a contemporaneous predictor would be look-ahead leakage.

MCMC acceptance criteria should include split $`\hat R<1.01`$, bulk and tail ESS appropriate to the number of posterior draws, zero unexplained divergences after warm-up, acceptable E-BFMI, no systematic maximum-treedepth saturation, and rank/energy plots in ArviZ. Variance components, tail degrees of freedom, hierarchical scales, and 2020–22 states deserve particular attention.

### Python/JAX implementation

Store the vintage triangle in **Polars** in long form:

`reference_month, release_date, stage, sector, SA_flag, employment, revision, collection_days, closing_rate, sample_metrics, benchmark_year, benchmark_flag, method_flags`.

Preserve release date separately from reference month; otherwise pseudo-real-time validation becomes error-prone.

Use **NumPyro/NUTS** for the full Student-$`t`$, hierarchical, stochastic-volatility specification. Implement time recursions with `jax.lax.scan` rather than Python loops.

The largest computational decision is whether to sample every $`x_{s,t}`$. For a 23-year monthly panel, multiple sectors, two adjustment states, and several latent revision processes, explicit latent-state NUTS can become unnecessarily expensive.

For the **linear-Gaussian conditional submodel**, analytically marginalize $`x_{1:T}`$ with Kalman filtering/smoothing. **Dynamax** is well suited to that component. Then NUTS samples hyperparameters, hierarchy coefficients, stochastic-volatility states, and any non-Gaussian mixing variables rather than several thousand strongly correlated employment levels.

The Student-$`t`$ innovations can be represented as normal scale mixtures,

```math
e_t\mid\lambda_t
\sim N(0,\sigma_t^2/\lambda_t),
```

```math
\lambda_t\sim\operatorname{Gamma}(\nu/2,\nu/2),
```

which keeps the employment state conditionally Gaussian. That opens the possibility of Kalman-marginalizing $`x`$ conditional on the local scales.

This marginalization is likely the single most important sampling-efficiency improvement. A direct centered NUTS fit of thousands of latent levels, stochastic volatilities, hierarchical sector coefficients, and heavy-tail scales is exactly the geometry one should avoid.

Use **BlackJAX** only if profiling shows that a custom blocked kernel is worthwhile—for example, alternating HMC updates for static parameters with a specialized Gaussian-state update. It is not necessary merely because the project uses JAX.

Use **ArviZ** for rank plots, ESS, $`\hat R`$, BFMI, PSIS-LOO, posterior predictive summaries, and model comparison.

A sensible development sequence is: Gaussian constant-variance marginalized model → Gaussian SV model → joint NSA/SA model → Student-$`t`$ scale mixture → sector hierarchy → slow-moving capacity covariates. The first four stages should reproduce the published BLS revision moments before funding or staffing is allowed anywhere near the model.

### What this model adds beyond a heteroskedastic regression

A frequentist heteroskedastic regression can certainly estimate whether

```math
\log |r_t|
```

is associated with collection days. An event study can describe shutdowns or a 2003 break. Neither is intrinsically invalid.

What the proposed model adds is a **single probabilistic accounting system** in which:

- latent employment uncertainty is carried through every vintage rather than treating the eventual release as unquestioned truth;
- news and noise are separated using cross-vintage restrictions rather than all revision being labeled “error”;
- QCEW benchmarks are high-precision observations rather than exact terminal values;
- annual benchmark discrepancies propagate through a known wedge operator;
- NSA sample revisions and seasonal-factor revisions are separated exactly and jointly;
- sector coefficients pool according to the available information;
- heavy tails and stochastic volatility distinguish isolated extreme revisions from persistent changes in reliability;
- weakly identified capacity coefficients remain visibly prior-sensitive rather than producing spuriously precise $`t`$-statistics;
- and the output is a **posterior predictive distribution for the next revision**, conditional on the actual collection calendar and current survey conditions.

The genuinely Bayesian advantage is therefore not that frequentist state-space methods are incapable of these components. It is that the hierarchical, weakly identified, irregular-vintage system can propagate uncertainty and shrinkage coherently enough to support statements such as:

> given this month’s collection interval, observed first-closing completeness, sector mix, and current volatility state, there is a posterior probability $`p`$ that the first payroll print will ultimately be revised by more than 75,000 jobs, of which $`q`$ percent of posterior revision variance is attributable to late-sample information and $`r`$ percent to seasonal-factor revision.

That is much closer to the decision problem faced by a real-time payroll nowcaster than either an average event-study coefficient or a regression on signed revisions.

## Annotated bibliography

**Aruoba, S. Boragan. 2008. “Data Revisions Are Not Well Behaved.” *Journal of Money, Credit and Banking* 40.** Peer-reviewed. Establishes that macroeconomic revisions can have nonzero means, large dispersion, and predictable structure; important warning against assuming CES revisions are either unbiased or pure news.

**Bureau of Labor Statistics. “Comparing with the original: a look at Current Employment Statistics vintage data.” *Monthly Labor Review*, 2015.** Agency analytical documentation. Defines the CES vintage archive, the 10–16-day first collection window, monthly revision stages, benchmarking, and historical reconstructions; essential data-source reference.

**Bureau of Labor Statistics. “Nonfarm Payroll Employment: Revisions between over-the-month estimates, 1979–present.”** Agency statistical documentation. The central public source for long-run total-nonfarm revision magnitude by first→second and second→third stage, in both SA and NSA form.

**Bureau of Labor Statistics. “Technical Notes for the CES-National Benchmark.”** Agency technical documentation. Current sample coverage, annual QCEW benchmarking, birth-death updates, reliability discussion, and benchmark/reconstruction mechanics.

**Bureau of Labor Statistics. *Handbook of Methods: Current Employment Statistics—National, Calculation*.** Agency methods documentation. Gives the matched-sample estimator, birth-death addition, seasonal-adjustment machinery, and calendar-effect treatments.

**Bureau of Labor Statistics. “CES Seasonal Adjustment Technical Notes.”** Agency technical documentation. Especially valuable for reproducing monthly factor changes: X-13 inputs, fixed-within-year specifications, annual five-year reseasonalization, prior adjustments, and COVID-era AO/TC/LS treatment.

**Bureau of Labor Statistics. “Seasonal Adjustment Files and Documentation.”** Agency data/documentation. Provides current X-13 specification inputs, adjustment classifications, and archived model information; key source for building the seasonal-factor component.

**Bureau of Labor Statistics. “The challenges of seasonal adjustment for the Current Employment Statistics survey during the COVID-19 pandemic.” *Monthly Labor Review*, 2022.** Agency analytical article. Documents why pandemic movements strained standard outlier treatment and motivated changes in CES seasonal-adjustment handling.

**Bureau of Labor Statistics. “One hundred years of Current Employment Statistics—the history of CES sample design.” *Monthly Labor Review*, 2016.** Agency analytical history. Best concise source for quota-sample weaknesses, ASA review, probability-sample development, 2000–03 phase-in, and subsequent allocation changes.

**Butani, Shail J., George Stamas, and J. Michael Brick. 1997. “Sample Redesign for the Current Employment Statistics Survey.” BLS statistical survey paper.** Agency research. Contemporary technical treatment of the probability-sample redesign and its design tradeoffs.

**Copeland, Kennon R., and Richard Valliant. 2007. “Imputing for Late Reporting in the U.S. Current Employment Statistics Survey.” *Journal of Official Statistics* 23(1): 69–90.** Peer-reviewed. Directly addresses the late-reporting problem underlying early CES revisions and is the closest published precursor to a predictive late-response correction model.

**Croushore, Dean, and Tom Stark. 2001. “A Real-Time Data Set for Macroeconomists.” *Journal of Econometrics* 105.** Peer-reviewed. Foundation for treating data vintage as part of the dataset rather than nuisance metadata; directly motivates pseudo-real-time CES validation.

**Dixon, John, and Clyde Tucker. 2016. “Using Quantile Regression to Model Revisions Due To Late Reporting in the Current Employment Statistics Survey.” BLS Office of Survey Methods Research paper.** Agency working paper. Most directly relevant CES study of late responders and revisions; linked 2010–14 CES/QCEW data reveal industry, size, and temporal heterogeneity but do not estimate the calendar-length effect on revision variance.

**Faust, Jon, John H. Rogers, and Jonathan H. Wright. “News and Noise in G-7 GDP Announcements.” Federal Reserve working-paper version; subsequently *Journal of Money, Credit and Banking*.** Peer-reviewed/central-bank working-paper lineage. Cross-country evidence that revision predictability and news/noise mixtures are institutional rather than universal constants.

**Fixler, Dennis J., and Bruce T. Grimm. 2003. “Revisions, Rationality, and Turning Points in GDP.” *Survey of Current Business*.** Agency research. Examines GDP revision predictability and reliability around turning points; useful benchmark for analogous CES questions.

**Fixler, Dennis J., et al. “News, Noise, and Estimates of the ‘True’ Unobserved State of the Economy.” BEA/Federal Reserve research.** Agency/working-paper research. Particularly relevant methodological bridge from conventional revision statistics to a latent-state interpretation.

**Groen, Jeffrey A. 2011. “Seasonal Differences in Employment between Survey and Administrative Data.” BLS Working Paper 443; related published work in *Journal of Official Statistics*.** Agency working paper / peer-reviewed lineage. Critical evidence that CES–QCEW discrepancies reflect reporting procedures and seasonal differences, not merely CES sampling error.

**Huff, Larry, and Julie Gershunskaya. 2009. “Components of Error Analysis in the Current Employment Statistics Survey.” BLS Office of Survey Methods Research.** Agency research. Decomposes CES–QCEW discrepancies and finds no simple relationship between lower response rates and higher nonresponse bias; crucial caution for any response-rate variance regression.

**Jacobs, Jan P.A.M., and Simon van Norden. 2011. “Modeling Data Revisions: Measurement Error and Dynamics of ‘True’ Values.” *Journal of Econometrics* 161.** Peer-reviewed. The core methodological template for the proposed latent-truth, multi-vintage news/noise decomposition.

**Kishor, N. Kundan, and Evan F. Koenig. 2012. “VAR Estimation and Forecasting When Data Are Subject to Revision.” *Journal of Business & Economic Statistics*.** Peer-reviewed. Demonstrates state-space treatment of real-time revisions and employment-related data; closest econometric predecessor, but without CES institutional variance drivers.

**Mankiw, N. Gregory, and Matthew D. Shapiro. 1986. “News or Noise? An Analysis of GNP Revisions.” NBER working paper / *Survey of Current Business*.** Working-paper/agency-publication lineage. Canonical news-versus-noise framework and the conceptual starting point for identification restrictions on CES vintage errors.

**Manning, Christopher. 2003. “Concurrent Seasonal Adjustment for National CES Survey.” *Monthly Labor Review*.** Agency analytical research. The strongest direct evidence on the magnitude effect of a CES methodological change: same-sample comparison of projected versus concurrent seasonal adjustment showing reduced revision dispersion.

**Bureau of Labor Statistics, Office of Survey Methods Research. “Predicting the Effect of Business Births and Deaths on Payroll Employment.” 2023.** Agency research. Demonstrates breakdown of historical birth-death relationships in extreme cycles and improved performance from current-sample information; directly motivates sector-specific benchmark-scale covariates.

**Bureau of Labor Statistics. “Business Birth-Death Model Frequently Asked Questions” and recent CES benchmark documentation.** Agency methods documentation. Establishes that post-2020 benchmark errors motivated model modification and documents how recent forecasts are incorporated.

**Bureau of Labor Statistics. “Federal Government Shutdown: October 2013 Employment Situation News Release.” 2013.** Agency operational documentation. Rare direct observation of exogenous survey-production disruption: CES data collection stopped from October 1 through October 16.

**Bureau of Labor Statistics. “Release dates impacted by the 2025/2026 lapse in appropriations.”** Agency operational documentation. Establishes a second modern BLS shutdown/lapse episode suitable for explicit coding in the vintage panel.

**U.S. Government Accountability Office. “Expert Views on the Federal Statistical System.” 2025.** Government oversight report. Relevant to resource, staffing, modernization, and institutional-capacity concerns; useful context but not causal evidence connecting BLS funding to CES revisions.

**Federal Reserve Bank of St. Louis, ALFRED.** Public real-time data infrastructure. Essential general-purpose vintage source, but not a substitute for a detailed national QCEW vintage archive; source-agency preservation determines historical depth.

**Bitto, Angela, and Sylvia Frühwirth-Schnatter. 2019. “Achieving Shrinkage in a Time-Varying Parameter Model Framework.” *Journal of Econometrics* 210(1): 75–97.** Peer-reviewed. Appropriate prior architecture for allowing structural evolution in volatility coefficients without turning every coefficient into an unrestricted random walk.

**Piironen, Juho, and Aki Vehtari. 2017. “Sparsity Information and Regularization in the Horseshoe and Other Shrinkage Priors.” *Electronic Journal of Statistics* 11.** Peer-reviewed. Basis for the proposed regularized-horseshoe treatment of the highly collinear funding/staffing/sample candidate drivers.

The literature therefore supports several individual mechanisms very well—late reporting, concurrent seasonal adjustment, probability-sample redesign, benchmark realignment, birth-death failure at turning points—but **does not yet supply the object most relevant for payroll nowcasting: a stage-specific model of the conditional scale of future CES revisions, with the latent true path, annual benchmark, seasonal-factor component, sector hierarchy, and production-system covariates estimated jointly**.
