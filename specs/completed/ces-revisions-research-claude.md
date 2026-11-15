# CES Payroll Revision Magnitude: Critical Literature Review, Gap Analysis, and a Bayesian State-Space Research Design

## 1. Executive Summary

The literature robustly characterizes the *sign and mean bias* of CES revisions but is remarkably thin on the *scale* (second moments) and how that scale has evolved — which is precisely your estimand. The canonical CES-specific work (Neumark-Wascher 1991; Aruoba 2008 JMCB; Goto-Jacobs-Sinclair-van Norden 2023 JAE) either predates NAICS/COVID or targets mean bias, news/noise classification, or nowcasting rather than revision *dispersion*. No published work applies a Jacobs–van Norden or Kishor–Koenig state-space decomposition to CES across all four revision stages with covariate-driven variance. That is the gap your design fills.

The 2025–2026 events are first-order evidence and are dated throughout. The **final March 2025 benchmark** (published Feb 11, 2026) cut total nonfarm by **−862,000 NSA (−0.5%)** — or **−861,000** after accounting for a monetary-authorities/commercial-banks data reconstruction — and **−898,000 seasonally adjusted (−0.6%)**. The **preliminary benchmark (Sept 9, 2025) was −911,000 (−0.6%)**, the largest absolute preliminary benchmark on record (back to 2002, per CNBC, and >50% larger than 2024's). BLS attributed the overstatement primarily to **response error and nonresponse error** ("businesses reported less employment to the QCEW than they reported to the CES survey," and nonrespondents reported still less to QCEW). The **43-day Oct 1–Nov 12, 2025 shutdown** (longest in U.S. history) forced cancellation of the October Employment Situation (CES October data folded into November's Dec 16 release); the household-survey October data are permanently lost because CPS cannot be collected retroactively. Budget and staffing stress is documented and severe: FY2026 enacted BLS budget ~\$708.5M, but on-board FTE fell to 1,773 (≈−14% vs FY2024's 2,058), and ASA states BLS "lost an estimated 22% in staffing since FY24."

Three drivers dominate the *magnitude* signal empirically: (i) net birth-death forecast error (the main benchmark driver post-2020); (ii) first-closing collection completeness (first-closing collection rate averaged 60.4% in 2024, down from a 2015 peak; the response rate fell from ~58% pre-2020 to 43% in 2024); and (iii) seasonal-factor instability during COVID. Funding/staffing and sample size are slow-moving, trending, and mutually collinear — honestly, only weakly identified from ~40 years of monthly data. The proposed design is a multi-vintage state-space model with a latent employment path, vintage-specific measurement equations across all four stages, stochastic volatility with covariates entering the log-scale, Student-t measurement innovations, partial pooling of driver coefficients across NAICS supersectors, and a regularized-horseshoe prior on the collinear covariates — implemented in NumPyro/Dynamax with analytic marginalization of the linear-Gaussian latent state.

## 2. Literature Review, Organized by the Five Drivers

### Revision-stage taxonomy and the decomposition running through this review

BLS publishes, for reference month *m*: first preliminary (≈first Friday of *m*+1), second preliminary (*m*+2), and third/final sample-based (*m*+3), then the annual benchmark (preliminary in Aug/Sept, final the following February with January's Employment Situation). Wedge-back linearly distributes the March benchmark error across the intervening months (21 months NSA, back to January 2021 SA in the 2025 benchmark), assuming error accumulated at a steady rate. The four stages you specify map to: (1) first→second (one-month revision); (2) second→third (two-month revision, "final" sample-based); (3) third→benchmark (incl. the preliminary Aug/Sept benchmark and the final March benchmark); (4) post-benchmark wedge-back/interpolation on intervening months.

Sources of revision the literature decomposes: (i) late sample reporting/collection completeness; (ii) seasonal-factor revision (concurrent recalculation each month); (iii) net birth-death model error; (iv) benchmark realignment to the QCEW universe; (v) NAICS/sample redesign/reweighting.

**Quantitative anchors (magnitude, seasonally adjusted, thousands):**

- **CRS IF13084 (Aug 14, 2025):** since 2003, the average *absolute* revision from first preliminary to third/final was **51,000**; the signed mean was **+9,000**. Largest negative: **−672,000 (March 2020)**; largest positives **+437,000** and **+389,000** (Nov, Dec 2021).
- **Benchmark magnitude, CRS IF12827:** from 2002 through 2023 the benchmark revision averaged **255,000** in absolute value; largest **−902,000 (March 2009)**, smallest **−7,000 (March 2021)**.
- **Preliminary benchmark percentage (HAAWKS, grey, Aug 2026):** 2016–2020 average absolute preliminary total-nonfarm revision **0.12%**; 2021–2025 average **0.34%**; total private **0.18%→0.46%**. Largest negative **−0.7% in 2025**.
- **Recent benchmark series (preliminary → final):** 2023 −306k prelim; 2024 −818k prelim / **−598k final**; 2025 **−911k prelim / −862k NSA (−861k adjusted) / −898k SA final**.
- **BLS 10-year context:** absolute benchmark revisions have averaged **0.2%** of total nonfarm over the prior decade (range <0.05% to 0.3%), so 2025's −0.6% is roughly 3× the recent norm.

The magnitude has risen materially in the 2020s, but disentangling how much is birth-death error vs. seasonal instability vs. collection completeness is exactly what the literature has NOT done systematically.

### Driver 1 — SEASONALITY

**What is known.** CES moved to concurrent seasonal adjustment in **June 2003** (with May 2003 first preliminary estimates), *simultaneously* with the SIC→NAICS 2002 conversion, the probability-sample redesign, birth-death modeling, and a new federal-government estimation method — all at once, and therefore confounded (BLS History; "Concurrent Seasonal Adjustment for Industry Employment Statistics," Oct 21 2002). Concurrent SA recalculates factors every month using all data through the current month; BLS documented this was *expected* to produce "smaller revisions from the first preliminary estimates to the final benchmarked estimates, than the semiannual updates." BLS uses X-13ARIMA-SEATS (adopted for CPI in Jan 2015, later extended to employment; earlier X-12-ARIMA) and controls for the 4- vs 5-week survey-interval ("variable survey interval") effect during adjustment. Additive vs. multiplicative: prior to the March 2002 benchmark (June 2003) all CES series used multiplicative models; a new processing system then enabled additive adjustment where it fit better.

**What is asserted without strong evidence.** The relative size of seasonal-factor revision vs. sample revision within the one- and two-month revision is not cleanly decomposed in any public, current paper. BLS's own concurrent-SA guide (Table 2) reported mean absolute revisions by industry division from first preliminary to final benchmarked, but that is pre-NAICS-era research and not maintained. There is no current published NSA-vs-SA decomposition of the monthly revision that isolates the seasonal-factor contribution.

**What is contested.** COVID-era seasonal-factor instability and outlier/level-shift handling. Abeln & Jacobs (2022, *J. Business Cycle Research*) study COVID and seasonal adjustment generally. The extent to which 2020–2022 outlier/level-shift treatment inflated subsequent seasonal-factor revisions in CES specifically is asserted but not quantified in peer-reviewed CES work.

### Driver 2 — COLLECTION INTERVAL

**What is known.** First closing is the last Friday of the reference month; first preliminary estimates rest on **10–16 days** of collection, varying with the calendar placement of the 12th and the release date. Historical first-closing collection was ~74% (Li & Lahiri, OSMR 2006); by **2024 the first-closing collection rate averaged 60.4%, second 89.0%, third 90.9%** (CRS IF13084, citing BLS series CEU00000000C1/C2/C3). The first-closing collection rate rose 1981–2015 (mail→telephone/web/electronic modes) then declined after 2015. Distinctly, the **response rate** (which includes refusals) fell from **~58% pre-2020 to 43% in 2024** (CRS; SF Fed EL 2025-07; series CEU05000000RR). BLS OSMR research (cited in CRS) finds *no consistent late-reporter pattern* by mode/industry/size/geography in the private sector, but **government units are systematically late** — and nearly half the unusually large May/June 2025 revisions were government (largely local and state government education).

**What is asserted / thin.** The explicit link from collection interval (days) or first-closing rate to one-month revision *dispersion* is intuited but not formally estimated in current literature. Li & Lahiri (2006, OSMR) modeled first-vs-third-closing discrepancies but for mean improvement, not variance, and pre-COVID. This is a clean open question and, importantly, the SF Fed (March 2025) finding that *monthly* revision dispersion is "in line with pre-pandemic averages" sits in tension with the record *annual* benchmark — a tension the four-stage design resolves.

### Driver 3 — BLS FUNDING

**Time series (verified from BLS Congressional Budget Justifications; nominal total budget authority = general funds + ~\$68M Unemployment Trust Fund transfer):** FY2010 ~\$611M (2,341 FTE); FY2012 ~\$609M; **FY2013 ~\$577M** (post-sequestration; BLS sequestration page: >\$30M / 5% cut, which eliminated Measuring Green Jobs and the International Labor Comparisons program); FY2014–15 ~\$592M; FY2016–17 ~\$609M; FY2018 ~\$612M; FY2019 ~\$615M; FY2020–21 ~\$655M; FY2022 ~\$688M; FY2023 ~\$698M; **FY2024 ~\$698M (2,058 FTE)**; **FY2025 ~\$704M (2,019 FTE)** (Full-Year CR, P.L.119-4; ASA: "\$698 million in FY24 to \$704 million in FY25"); **FY2026 enacted ~\$708.5M but FTE 1,773** (FY2027 CBJ; Further Consolidated Appropriations Act 2026, P.L.119-75, Feb 3 2026 — Congress rejected the proposed cut and the BLS→Commerce reorganization). The FY2026 President's request was **\$648M total budget authority / \$580M general-funds-only** (the two figures reconcile: \$580M excludes the ~\$68M trust-fund transfer), with requested FTE **1,851**.

*Caveat:* House and Senate FY2026 committee **BLS-specific** dollar marks could not be verified from CRS toplines (which report only department-level DOL discretionary figures); flag as unverified.

**2025 program cuts (documented, BLS notices):** CPI collection suspended in Lincoln NE and Provo UT (April 2025) and Buffalo NY (June 2025); numerous PPI estimates discontinued; restricted-use FSRDC datasets suspended (June 2025). A January 2025 hiring freeze hit BLS hard; a partial exemption allowed CPI price-collector hiring from September 2025. These are CPI/PPI cuts; **no public documentation shows a CES sample reduction or suspended CES series in 2025 beyond the routine Feb-2025 annual sample review** (which discontinued/combined some low-sample industries) — state this explicitly rather than inferring a CES cut.

**Causal claim status.** Whether appropriations/budget stress *causally* degrades CES data quality is asserted by advocacy/grey sources (CFM "Penny wise, data foolish"; Center for American Progress; ASA "Nation's Data at Risk 2025") but not credibly identified anywhere. Partisan composition of appropriators is descriptive only and confounded with secular trend — not causal.

### Driver 4 — SAMPLE

**What is known.** CES completed a multiyear redesign **2000–2003** from a quota/cutoff sample to a probability-based, stratified simple random sample of worksites clustered by UI account number, phased in by industry (wholesale 2000; mining/construction/manufacturing 2001; TPU/FIRE/retail 2002; services 2003; government not converted due to high existing coverage) (MLR Feb 2006; BLS History). Sample size has drifted: ~140,000 businesses/440,000 worksites (c. 2011); ~122,000 businesses/666,000 worksites (FY2024 CBJ); ~119,000–121,000 businesses/~622,000 worksites (2024–2026). Strata sampling rates are set by **optimum allocation** to minimize variance on the primary estimate; the sample is reviewed/reallocated every 5 years. CES covers >26% of universe employment, yielding small variance on the total-nonfarm level.

**What is thin.** A documented empirical relationship between sample size/composition and revision *variance* over time does not exist in the public literature. OSMR researchers (Gershunskaya, Kropf, Mueller, Stamas, Getz, Butani, and colleagues) have modeled CES small-domain estimation and nonresponse, but not a longitudinal sample-size→revision-variance link.

### Driver 5 — STAFFING

FTE series above (2,341 in FY2010 → **1,773 enacted FY2026**). ASA: **"BLS has lost an estimated 22% in staffing since FY24"**; the FY26 request of 1,851 FTE is a 10% cut from FY24's 2,058; roughly one-third of ~36 leadership positions vacant, including positions tied to employment estimates (and the Commissioner, following McEntarfer's Aug 1, 2025 firing; Wiatrowski acting). There is essentially no peer-reviewed literature relating statistical-agency staffing to revision behavior; the claim is plausible and topical but empirically unestablished — a genuine "no one has looked at this."

### Situating CES in the broader revision literature

- **Mankiw-Shapiro (1986, "News or Noise?", NBER w1939 / Survey of Current Business)** and **Mankiw-Runkle-Shapiro (1984, JME 14:15–27)** — GNP/money stock, not payrolls. Framework, not application.
- **Aruoba (2008, "Data Revisions Are Not Well Behaved," JMCB 40(2-3):319–340)** — includes payroll employment among US series; finds revisions have non-zero mean (biased), are large, and predictable — violating the noise-only benchmark. Peer-reviewed; the closest thing to a payroll-revision-properties paper, but on mean/predictability, not evolving scale.
- **Jacobs & van Norden (2011, J. Econometrics 161:101–109)** — state-space model separating news vs. noise measurement error with a latent "true" value; the tradition your design extends. Not applied to CES specifically.
- **Kishor & Koenig (2012, JBES 30(2):181–190)** — real-time Kalman filtering / VAR-with-revision; methodology, applied to GDP.
- **Goto, Jacobs, Sinclair & van Norden (2023, JAE 38(7):1007–1017, "Employment Reconciliation and Nowcasting")** — THE key CES-adjacent paper: builds a latent US employment estimate reconciling payroll and household surveys and incorporating the payroll revision process; finds the reconciled series is close to benchmarked payrolls but in real time puts *near-zero weight on the household survey*. Peer-reviewed. It does NOT model revision *scale* as covariate-driven or across all four stages with SV.
- **Nalewaik (FEDS 2007-34, "News, Noise, and Estimates of the 'True' Unobserved State")**, **Fixler-Grimm (BEA)**, **Faust-Rogers-Wright (2005, JMCB 37:403–419, G-7 GDP)** — GDP/GDI, cross-country; not payrolls.
- **Croushore-Stark (Philadelphia Fed Real-Time Data Set / ALFRED)** — infrastructure; payroll vintages available but revision-scale modeling not done. **Aruoba-Diebold-Scotti** ADS index uses payroll growth as an input but is a coincident-activity model, not a revision-scale model.
- **Neumark & Wascher (1991)** and **Stark (2011)** — find payroll revisions biased.
- Grey literature (labeled): SF Fed EL 2025-07 (Leduc-Oliveira-Paulson: revisions "in line with pre-pandemic averages" as of March 2025, with an Aug 29, 2025 update); Wells Fargo, RBC, CFA Institute EI blog, HAAWKS, ADP, macromostly/MishTalk substacks; PIIE (Wilcox, Oct 2025); Upjohn (Horrigan, 2025); AEI/COSM; CFM.

**Bottom line for Part A:** The frameworks exist; their application to CES revision *magnitude across all four stages with covariate-driven second moments* does not.

## 3. Gap Analysis

| # | Open question | Closest existing work | Why it falls short | Data needed | Feasibility (public) |
|---|---|---|---|---|---|
| 1 | How has the *variance* of the one-month (1st→2nd) revision evolved 1979–2026? | CRS IF13084 (mean abs = 51k since 2003) | Reports means/extremes, not modeled time-varying variance; no covariates | BLS 1979–present revision tables (CEU vintage series) | High — public |
| 2 | Decompose SA monthly revision into NSA-sample vs. seasonal-factor components | BLS concurrent-SA guide (2002, Table 2) | Pre-NAICS; not maintained; no variance decomposition | NSA and SA vintages jointly, all closings | Medium — NSA vintages exist; needs reconstruction |
| 3 | Does first-closing collection rate / collection-interval days drive 1-month revision dispersion? | Li & Lahiri (OSMR 2006) | Mean, not variance; pre-COVID | CEU C1/C2/C3 collection-rate series + calendar of closings | High |
| 4 | Net birth-death forecast error contribution to benchmark magnitude, by sector, over time | BLS Birth-Death FAQ; Clausen (OSMR 2011) | Descriptive tables; no probabilistic model of error scale | BLS birth-death forecasts vs realized (tables 8–10 in benchmark articles) | Medium — hand-build from annual articles |
| 5 | Causal effect of appropriations/staffing on CES revision scale | CFM, CAP, ASA (grey) | No identification; collinear/trending | FY appropriations + FTE (built) + revision panel | Low — not credibly identified |
| 6 | Relationship of sample size/composition to revision variance | MLR 2006; OSMR sample-design papers | No longitudinal variance link | Annual sample sizes by stratum (largely unpublished) | Low–Medium |
| 7 | Full news/noise decomposition of CES across all 4 stages | Jacobs–van Norden (2011); Goto et al. (2023) | JvN not applied to CES; Goto et al. stops at 2nd revision + benchmark, no SV | Multi-vintage panel + benchmark | High (this proposal) |
| 8 | COVID seasonal-factor instability effect on revision scale | Abeln-Jacobs (2022) | General, not CES revision-scale | NSA/SA vintages 2018–2023 | Medium |

**Data availability inventory (per driver):**

- **Revisions:** BLS `cesnaicsrev.htm` (1979–present, over-the-month, SA and NSA); `cesvininfo.htm` (industry vintages since 2003). Monthly. BLS public.
- **Collection/response rates:** CEU00000000C1/C2/C3 (collection, three closings); CEU05000000RR (response). Public; earliest ~2000 for collection, monthly.
- **Birth-death:** annual benchmark articles (tables of forecast vs realized net birth-death by supersector — e.g., 2025 article: actual net birth-death Apr 2024–Mar 2025 was ~117k below forecast) — must be hand-compiled year by year. FAQ documents methodology changes (2011 quarterly update; 2024 current-sample modification for the post-benchmark window; 2025→2026 extension incorporating current sample info into first preliminary estimates plus a WLR regression component).
- **Benchmark:** annual benchmark articles + preliminary benchmark releases (national and state/area). Public.
- **Seasonal factors:** published SA and NSA series; projected-factor archives pre-2003. Concurrent factors are not separately archived by vintage — reconstruct from SA/NSA vintage pairs.
- **QCEW vintages:** **BLS does not publicly archive QCEW vintages**; ALFRED's QCEW coverage is limited to selected county/MSA series without industry detail. This is a real constraint on treating the benchmark target as a revisable observation.
- **Appropriations/FTE:** hand-built from Congressional Budget Justifications ("Amounts Available for Obligation" and "Appropriation History" tables, FY2012–FY2027) and the ASA report; no single clean public series exists.

**Most-worth-pursuing gaps.** (1) The SA-decomposition (Gap 2) and the collection-interval→dispersion link (Gap 3) are both feasible from public data and directly address your estimand — these are "the literature is old/pre-NAICS," not "impossible." (2) The full four-stage news/noise decomposition with covariate-driven scale (Gap 7) is genuinely novel — "no one has looked at this." Funding/staffing causality (Gap 5) is honestly not identifiable and should be framed as descriptive association with explicit shrinkage.

## 4. Bayesian Research Design (implementation-ready specification)

### 4.1 Estimand and top-level structure

The target is the **scale** of revisions at each stage, and how covariates shift it. Covariates enter the **variance / measurement-density scale**, not (only) a conditional mean.

Latent state: true (benchmark-consistent) NSA employment *level* by supersector *s*, month *t*: **`θ_{s,t}`**. True over-the-month change: **`Δθ_{s,t}`**.

Vintage-specific measurement equations. For each reference month *t*, four observed vintages of the over-the-month change: y¹ (first preliminary), y² (second), y³ (third/final sample-based), and the benchmark-implied change y^B (annual, higher precision). Following Jacobs–van Norden, each vintage = true value + a **news** component (correlated with future information, uncorrelated with earlier vintages) + a **noise** component (classical measurement error, uncorrelated with the true value). Identification of news vs. noise comes from the **correlation structure across the four releases**: noise revisions are predictable from and correlated with earlier vintages; news revisions are not.

### 4.2 Measurement density with stochastic volatility + covariates

For release *r* ∈ {1,2,3}, model the revision increment r→r+1 with a Student-t density whose log-scale is covariate- and time-driven:

- `(y^{r+1}_{s,t} − y^{r}_{s,t}) ~ StudentT(ν_r, μ_r [persistent bias, ≈0 for news], exp(h^r_{s,t}))`
- `h^r_{s,t} = α^r_s + x_{t}·β^r_s + φ_r·h^r_{s,t−1} + σ_η·η_{s,t}` (SV on the log-scale)

where x_t contains: first-closing collection-rate deficit (60%−rate), collection-interval days (calendar-driven), a COVID indicator/spline (2020–2022), a post-2015 collection-decline slope, log real appropriations, FTE, and log sample size. Student-t (ν_r estimated, small ν captures fat tails from strikes/turning points, e.g. the −672k March 2020 and +437k Nov 2021 revisions) vs. Gaussian is tested via LOO. The SV term captures persistent volatility clustering (2008–09, 2020–21) that covariates alone miss.

### 4.3 Benchmark as irregular high-precision observation + wedge-back

The annual benchmark enters as a measurement of θ at March with small measurement variance (QCEW near-universe), but *not zero* — QCEW itself has response/nonresponse error (BLS explicitly attributed the 2025 gap to response + nonresponse error) and is itself revised (unarchived; modeled as a nuisance). The **wedge-back** is modeled explicitly: the accumulated benchmark error is distributed across the intervening months. Rather than BLS's deterministic linear wedge, use a **random-walk bridge** (Brownian bridge) between consecutive benchmarks so the interpolation error has a posterior, letting you test whether the linear-accumulation assumption holds. Net birth-death forecast error enters as a sector-specific mean shift in the third→benchmark stage with its own scale (calibrated to the benchmark-article forecast-vs-realized tables).

### 4.4 Hierarchy across NAICS supersectors

Partial pooling of β^r_s (and α^r_s, μ_r) across the ~11 supersectors, since birth-death and benchmark error are sector-heterogeneous (2025 final-benchmark contributions: leisure & hospitality −176k, professional & business services −158k, retail −126k, wholesale −110k, manufacturing −95k). **Non-centered parameterization** for all hierarchical coefficients (β^r_s = β^r + τ^r·z_s, z_s ~ N(0,1)) — essential given the funnel geometry that SV + hierarchy produce in NUTS. Total nonfarm is modeled as the sum of supersectors plus an aggregation-consistency term (aggregate variance < sum, due to negative cross-sector covariances and the larger effective sample share at the aggregate).

### 4.5 Seasonality: joint NSA/SA modeling

Model NSA and SA vintages **jointly**. The SA revision = NSA-sample revision + seasonal-factor revision. Represent the concurrent seasonal factor as a latent log-additive state `γ_{s,t}` with its own innovation variance; the SA vintage = NSA vintage × exp(γ). Because concurrent SA re-estimates γ each month, the seasonal-factor revision component is identified from the difference between SA and NSA revisions at the same closing. This directly answers Gap 2. A COVID variance-inflation multiplier on γ's innovation captures 2020–2022 seasonal instability.

### 4.6 Structural change

- **2003 (NAICS + concurrent SA + probability sample + birth-death + federal method, simultaneous):** cannot be separately identified — treat as a single confounded change point with a level/scale break; explicitly state you cannot attribute the break to any one component. Do not pretend otherwise.
- **2020–2022:** time-varying parameters with shrinkage (Bitto & Frühwirth-Schnatter 2019 triple-gamma, or a horseshoe on TVP innovations) plus explicit outlier/level-shift dummies, rather than hard change points, so the model can absorb COVID without discarding data.
- **Sample redesigns / birth-death changes (2011 quarterly update; 2024/2025–26 current-sample + WLR regression):** dated change points on the third→benchmark scale.

### 4.7 Identification and confounding — be candid

Appropriations, staffing, and sample size are slow-moving, trending, mutually collinear → their separate slopes are **weakly identified**. Collection-interval days vary sharply within year → **well identified** off high-frequency variation. First-closing rate has both a secular trend (weakly identified vs. budget) and idiosyncratic monthly variation (identified). Strategy:

- **Regularized horseshoe** (Piironen–Vehtari) on the block of slow-moving collinear covariates {log real appropriations, FTE, log sample size, post-2015 slope}, so the model selects at most one or two carrying signal and shrinks the rest to ~0, honestly reflecting that ~40 years of monthly data cannot separate them.
- Well-identified within-year covariates (collection days, first-closing deficit) get weakly-informative Normal priors, not horseshoe.
- **Party-of-appropriators:** descriptive grouping only; enter as a varying intercept with a tight prior, explicitly *not* interpreted as a causal slope. No defensible causal design exists here without an instrument.
- **Breaking assumptions to state:** news/noise identification requires news uncorrelated with earlier vintages and noise uncorrelated with the true state. Correlated sampling errors across closings (the *same* late reporters) or benchmark error correlated with sample error (both driven by the same nonresponding firms — which BLS's 2025 statement directly implies) **break** the clean decomposition. Model this with a nonzero covariance between third-stage noise and benchmark error, and report sensitivity.

### 4.8 Priors (natural units: thousands of jobs)

- Log-scale intercepts α^r_s: one-month revision historical MAD ≈ 51k total nonfarm ⇒ per-supersector scale ~10–30k. α^r ~ Normal(log 20, 0.5); supersector deviations τ^r ~ HalfNormal(0.5).
- SV persistence φ_r ~ Beta(8,2) (persistent, stationary); σ_η ~ HalfNormal(0.3).
- Student-t dof ν_r ~ Gamma(2, 0.1) shifted to >2 (finite variance), allowing heavy tails.
- Well-identified covariate slopes β ~ Normal(0, 0.3) on standardized covariates (effect on log-scale).
- Horseshoe block: global τ ~ HalfCauchy(0, τ₀), τ₀ set so effective nonzero coefficients ≈1–2; local λ_j ~ HalfCauchy(0,1); regularization slab c² ~ InvGamma.
- Benchmark measurement scale: HalfNormal centered near QCEW-implied ~0.1–0.2% of level, with a right tail to admit the response/nonresponse error BLS flagged.
- Seasonal innovation σ_γ: tight (HalfNormal(0.02) on the log-factor) except a COVID inflation multiplier ~ LogNormal.
- **Prior predictive check plan:** simulate revision panels; verify implied one-month total-nonfarm revision MAD lands in the 30–70k range and benchmark revisions in the 0.1–0.7% range; reject priors that generate implausible ±500k routine monthly revisions.

### 4.9 Validation

- **PPCs targeted at dispersion:** posterior predictive distribution of |revision| by stage and era (pre-2003, 2003–2019, 2020–2022, 2023–2026); check coverage of the 51k mean-abs and the −672k/−911k tails.
- **LOO/PSIS** vs. a constant-variance (no-SV, no-covariate) Gaussian baseline and vs. a Gaussian-SV (no-t) model; report Pareto-k (turning-point months will be influential).
- **Pseudo-real-time / out-of-sample:** refit through each year *y*, predict year *y*+1 revision scale; evaluate CRPS on realized |revisions|.
- **MCMC diagnostics:** R-hat < 1.01, ESS, divergences (non-centered parameterization essential), energy/BFMI for the SV blocks.

### 4.10 Python/JAX implementation notes

- **NumPyro (NUTS)** for the full model with SV + horseshoe + Student-t.
- **Dynamax** for the linear-Gaussian core where the latent employment path and seasonal state can be **marginalized analytically** via the Kalman filter — do this: marginalizing θ (and γ where log-linearized) dramatically improves geometry and ESS versus sampling the full latent path. Sample only the non-Gaussian pieces (SV log-vols, dof, horseshoe scales) in NUTS, conditioning on the marginal likelihood from the Kalman filter. **This is the single most important efficiency decision.** (The Student-t density can be handled as a Gaussian scale-mixture, preserving conditional linear-Gaussian structure so the Kalman marginalization still applies.)
- **Blackjax** if a custom kernel is warranted (SV-specific ancillary/sufficient interweaving, or a tempered kernel for the multimodal horseshoe).
- **ArviZ** for LOO/PSIS, R-hat, PPC plots.
- **Polars** for the vintage panel (lazy joins across CEU vintage series, collection-rate series, hand-built budget/FTE and benchmark-article tables), keyed by (supersector, reference month, vintage).

### 4.11 What this buys you over frequentist alternatives

A frequentist event-study around 2003/2020 or a heteroskedastic (GARCH) regression on revisions could estimate variance shifts but cannot: (i) **separately identify news vs. noise** across four releases within a coherent latent-state model; (ii) **propagate benchmark and wedge-back uncertainty** into the intervening-month state with full posteriors; (iii) **partially pool** sector coefficients while honestly shrinking collinear budget/staffing effects via the horseshoe (avoiding the false precision a fixed-effects OLS reports on collinear trends); (iv) deliver **calibrated predictive distributions of future revision scale** (a nowcast-of-the-revision) directly usable as the measurement variance in your payroll nowcast; and (v) yield a **posterior probability** that, e.g., the post-2020 birth-death error scale exceeds its 2003–2019 level, rather than a reject/fail-to-reject verdict.

## 5. Annotated Bibliography

**Peer-reviewed**

- Aruoba, S.B. (2008). "Data Revisions Are Not Well Behaved." *JMCB* 40(2-3):319–340. Payroll among US series; revisions biased, large, predictable. Closest payroll-properties paper; mean not scale.
- Jacobs, J.P.A.M. & van Norden, S. (2011). "Modeling data revisions: measurement error and dynamics of 'true' values." *J. Econometrics* 161:101–109. State-space news/noise tradition this design extends; not applied to CES.
- Kishor, N.K. & Koenig, E.F. (2012). "VAR estimation and forecasting when data are subject to revision." *JBES* 30(2):181–190. Real-time Kalman/VAR; GDP.
- Goto, E., Jacobs, J., Sinclair, T., van Norden, S. (2023). "Employment Reconciliation and Nowcasting." *JAE* 38(7):1007–1017. Latent US employment reconciling payroll+household with revision process; near-zero real-time weight on household survey. Most CES-relevant; no covariate-driven scale.
- Mankiw, N.G. & Shapiro, M.D. (1986). "News or Noise? An Analysis of GNP Revisions." NBER w1939 / Survey of Current Business. Origin of news/noise; GNP.
- Mankiw, Runkle & Shapiro (1984). *JME* 14:15–27. Money-stock rationality.
- Faust, Rogers & Wright (2005). "News and Noise in G-7 GDP Announcements." *JMCB* 37:403–419. Cross-country GDP.
- Jacobs, Sarferaz, Sturm & van Norden (2022). "Can GDP Measurement Be Further Improved? Data Revision and Reconciliation." *JBES* 40(1):423–431. Multi-release news/noise identification; GDP.
- Abeln, B. & Jacobs, J. (2022). "COVID-19 and Seasonal Adjustment." *J. Business Cycle Research*. Seasonal adjustment under COVID.
- Neumark & Wascher (1991); Stark (2011). Payroll revisions biased.
- Bitto, A. & Frühwirth-Schnatter, S. (2019). "Achieving shrinkage in a time-varying parameter model framework." *J. Econometrics*. TVP shrinkage for the design.

**Agency documentation (BLS/Census/CRS/DOL)**

- BLS, "Current Employment Statistics Preliminary Benchmark (National) — March 2025," USDL-25-1352 (Sept 9, 2025). −911k (−0.6%); response + nonresponse error.
- BLS, "CES National Benchmark Article" / "CES Benchmark Announcement" (cesbmk.htm, Feb 11, 2026). Final March 2025: −862k NSA (−861k adjusted), −898k SA; birth-death forecast-error tables (net −117k Apr'24–Mar'25); wedge-back description.
- BLS, "Employment Situation — January 2026" (Feb 11, 2026, archive) and "February 2026" (DOL). Benchmark incorporation; furloughed workers counted as employed for the pay period including the 12th.
- BLS, "Revised news release dates following the 2025 and 2026 lapses in appropriations." Oct'25 Employment Situation cancelled; CES October folded into November (Dec 16, 2025).
- BLS, MLR (May 2026), "Addressing missing consumer expenditure data due to the 2025 lapse in appropriations." Shutdown data-handling.
- BLS, CES Birth-Death Model FAQ; Clausen, N. (OSMR 2011), "Forecasting Birth/Death Residuals on a Quarterly Basis." 2011 quarterly update; 2024/2025–26 current-sample + WLR regression modifications.
- BLS, "Concurrent Seasonal Adjustment for Industry Employment Statistics" (Oct 21, 2002) + concurrent-SA guide. 2003 transition; Table 2 magnitudes (pre-NAICS).
- BLS History (CES-National); MLR Feb 2006 "Employment Measures" (probability-sample redesign 2000–2003).
- BLS Congressional Budget Justifications FY2012–FY2027; BLS sequestration page. Appropriations/FTE series; FY2013 sequestration (>\$30M/5%; eliminated Green Jobs, ILC).
- Li, B.T. & Lahiri, P. (OSMR 2006), "Improving CES Preliminary Employment Estimate through Statistical Model." First-vs-third closing; ~74% first-closing.
- CRS IF13084 (Aug 14, 2025), monthly revisions (51k mean abs since 2003; +9k signed; −672k Mar 2020; collection rates 60.4/89/90.9%; response 58%→43%); CRS IF12827, benchmark revisions (255k avg abs 2002–2023; −902k Mar 2009; −7k Mar 2021).
- Census, X-13ARIMA-SEATS documentation.

**Grey literature (labeled)**

- SF Fed EL 2025-07 (Leduc, Oliveira, Paulson, Mar 31, 2025; Aug 29, 2025 update): monthly revisions "in line with pre-pandemic averages."
- ASA, "The Nation's Data at Risk: 2025" (BLS report): 22% staffing loss since FY24; 1,851 FY26 requested FTE; budget path \$698M→\$704M→\$648M request; purchasing-power erosion.
- Center for American Progress (2025): FY26 request framing; CPS sample-cut risk.
- CFM, "Penny wise, data foolish" (2025): funding cuts and revision errors (asserts causality — treat skeptically).
- PIIE/Wilcox (Oct 2025); Upjohn/Horrigan (2025); AEI/COSM; HAAWKS (Aug 2026); Wells Fargo (Sept 2025); RBC; CFA Institute EI blog (2026); macromostly, MishTalk, USAFacts. Commentary/market analysis and benchmark magnitude framing.

## 6. Caveats

- **Final vs. preliminary March 2025 benchmark:** −911k preliminary (Sept 2025); final −898k SA / **−862k NSA (−861k after a monetary-authorities/commercial-banks data reconstruction)**. The **−598k** figure that circulates is the **2024 final** benchmark, not 2025 — do not conflate.
- **CES-specific 2025 cuts:** documented collection cuts in 2025 are to **CPI and PPI**, not CES. Beyond the routine Feb-2025 annual sample review (discontinuing/combining low-sample industries), no public source confirms a 2025 CES sample reduction or suspended CES series — stated as a non-finding rather than inferred.
- **FY2026 chamber marks:** House/Senate committee **BLS-specific** dollar figures could not be verified (CRS reports only department-level DOL toplines); FY2026 was ultimately enacted at ~\$708.5M (P.L.119-75).
- **FTE figures** mix "authorized" vs. "actual" across CBJ vintages; year-to-year FTE comparisons carry that caveat. ASA's "~22% since FY24" reflects on-board staffing (hiring freeze + attrition + workforce actions), steeper than the budgeted FTE line.
- **QCEW vintages are not publicly archived**; the benchmark-as-revisable-observation component relies on reconstruction and is the weakest data link.
- **Funding/staffing causal effects** on revision scale are not identifiable from available data; the design deliberately shrinks them and reports associations, not causes.
- **Apparent SF Fed vs. record-benchmark contradiction is not real:** SF Fed (March 2025) measures *monthly* first→second revision dispersion (stable), while the record benchmark measures *annual* level error (elevated by birth-death and response/nonresponse error). The four-stage design is precisely what reconciles the two.
