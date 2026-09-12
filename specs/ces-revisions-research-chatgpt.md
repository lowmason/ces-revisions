## 1. EXECUTIVE SUMMARY

This memorandum synthesizes the literature on the magnitude and variance of U.S. Current Employment Statistics (CES) payroll revisions and outlines a comprehensive Bayesian hierarchical state-space design to estimate their structural drivers. While the broader real-time data literature (Mankiw-Shapiro, Jacobs-van Norden) focuses heavily on mean bias and the "news versus noise" distinction, the secular and cyclical determinants of revision *variance*—specifically the interactions between sample collection intervals, administrative capacity, and seasonal adjustment methodologies—remain severely under-explored for the CES.

The proposed research design casts the multi-vintage CES panel as a hierarchical state-space model. By parameterizing the measurement-error variance as a log-linear function of collection, budgetary, and methodological covariates, the design allows for direct estimation of scale effects while analytically marginalizing the latent "true" employment state. The hierarchy pools sector-specific parameters to isolate heterogeneous net birth-death and benchmark-realignment effects. This approach overcomes the limitations of frequentist heteroskedastic regressions by properly accommodating the annual wedge-back constraint, identifying news/noise structures simultaneously with variance covariates, and using regularized horseshoe priors to navigate the collinearity of slow-moving administrative drivers.

---

## 2. LITERATURE REVIEW: THE VARIANCE OF CES REVISIONS

The macroeconometric literature heavily explores vintage data properties, spearheaded by Mankiw and Shapiro’s (1986) news vs. noise framework. Croushore and Stark (2001) institutionalized real-time analysis via ALFRED, while Aruoba (2008) defined the empirical properties of revisions. Faust, Rogers, and Wright (2005) and Fixler and Grimm (2005, 2008) mapped cross-country GDP and BEA revisions, respectively. Kishor and Koenig (2012) and Jacobs and van Norden (2011) advanced the state-space formulation of real-time data. However, applying these frameworks to *CES revision magnitude*, decomposed by driver, remains fragmented.

### I. Seasonality

* *What is known:* The 2003 introduction of concurrent seasonal adjustment (replacing projected factors) structurally changed the 1st $`\to`$ 2nd and 2nd $`\to`$ 3rd print dynamics. Under concurrent SA, seasonal factors are re-estimated monthly, meaning 1- and 2-month revisions convolute late sample reporting with SA factor updates. COVID-era data forced BLS to introduce manual level shifts and additive outliers in X-13ARIMA-SEATS to prevent catastrophic distortion of forward seasonal factors (Stuart & Weng, 2021).
* *What is asserted:* It is widely assumed that moving holidays (e.g., Thanksgiving timing) and 4- vs. 5-week interval variations between the 12th-of-the-month reference period heavily dictate short-term SA variance.
* *What is contested:* The exact decomposition of 1-month revision variance into NSA-sample versus SA-factor components over time. Most literature treats the SA data as the primitive, ignoring that variance dynamics in the SA series often originate in the X-13 filter rather than the CES sample.

### II. Collection Interval

* *What is known:* The CES reference period includes the 12th of the month, but the calendar date of the Employment Situation release varies. Consequently, the collection interval for the 1st print ranges from 10 to 16 days. Shorter intervals mechanically suppress the first-closing response rate (often below 50%, compared to ~70%+ for the 3rd print). OSMR literature (e.g., Phipps & Toth, 2012) extensively documents the secular decline in survey response rates.
* *What is asserted/contested:* While intuitively a shorter interval should increase 1st-print variance, the relationship is nonlinear. Groen et al. (2017) note that late reporters are not missing at random; thus, variance and mean bias interact. Precise econometric mapping of collection days to the *scale* of the 1-month revision across different NAICS supersectors is lacking.

### III. BLS Funding

* *What is known:* Real annual appropriations for BLS have been stagnant or declining for two decades, exacerbated by the 2013 sequestration and frequent continuing resolutions (CRs).
* *What is asserted:* COPAFS and the ASA routinely assert that flat funding, shutdowns, and CRs degrade data quality and increase revision variance due to deferred system upgrades and constrained follow-up operations.
* *What is contested/Unproven:* No peer-reviewed econometric literature rigorously links BLS funding levels to the *variance* of CES revisions. The partisan composition of appropriating chambers is often cited in grey literature as a proxy for budget stress, but this is confounded by secular trends.

### IV. Sample Redesign

* *What is known:* Between 1995 and 2003, CES transitioned from a quota/cutoff sample to a probability-based sample, alongside the NAICS transition and the replacement of the bias adjustment factor with the net birth-death (B-D) model.
* *What is asserted:* The probability sample and EDI/web collection modes theoretically reduced sampling variance and bounded 2nd/3rd print revisions.
* *What is contested:* Whether the B-D model actually reduced variance at turning points, or merely shifted variance to the 3rd $`\to`$ Benchmark print. The B-D model relies on ARIMA projections of QCEW data, failing at cyclical turning points (Cajner et al., 2018), thereby inflating the magnitude of benchmark revisions during recessions.

### V. Staffing

* *What is known:* OPM FedScope data confirms a secular decline in BLS FTEs over the last 15 years.
* *What is asserted:* Fewer FTEs means less manual review of micro-data anomalies, theoretically increasing the variance of early prints.
* *What is contested:* As with funding, the formal link between FTEs and revision variance is entirely unproven in the literature, representing a massive gap in the economics of statistical agencies.

---

## 3. GAP ANALYSIS & DATA AVAILABILITY

### A. Gap Analysis

| Open Question | Closest Existing Work | Why It Falls Short | Data Needed | Feasibility |
| --- | --- | --- | --- | --- |
| **1. Does agency funding/staffing dictate revision scale?** | General COPAFS reports; National Academies CNSTAT reviews. | Relies on institutional narrative and anecdotes; lacks formal identification or multivariate modeling. | Real BLS appropriations, OPM FTE counts, CES vintage panel. | **High.** All independent variables are public. Requires careful shrinkage due to collinear trends. |
| **2. Decomposition of 1-month revision variance (Sample vs. SA Factor)** | BLS Technical Notes; Cleveland Fed working papers (e.g., Knotek). | Usually static historical averages; does not model time-varying SV or separate news/noise by component. | Joint panel of NSA and SA vintages by supersector. | **Moderate.** ALFRED's NSA vintage coverage is patchy before 1990; requires BLS archival scrape. |
| **3. High-frequency collection interval effect** | Phipps & Toth (2012) on response rates. | Maps interval to *response rate*, not directly to the conditional *variance* of the published revision. | Calendar exact dates for collection cutoffs vs. reference weeks. | **High.** Dates are deterministic and easily reconstructed historically. |
| **4. B-D model vs. Benchmark variance** | Cajner et al. (2018); U.S. payroll benchmark methodology papers. | Focuses on mean bias at cycle turning points; doesn't parameterize variance scale hierarchically. | Sectoral QCEW vintages; BLS published B-D model adjustments. | **Low/Moderate.** QCEW vintages are notoriously difficult to source comprehensively (not in ALFRED). |

*Most Worth Pursuing:* The causal link between administrative capacity (funding/staffing) and revision variance (Gap 1) is the most critical gap. It is a highly consequential policy question that the literature has entirely ignored in favor of pure time-series properties. Second is the high-frequency collection interval (Gap 3), which provides the sharpest exogenous variation available to identify the baseline structural variance of the 1st print.

### B. Data Availability Inventory

* **CES Vintages (SA & NSA):** ALFRED (PAYEMS, 1961–present; NAICS supersectors mostly 2003–present). *Constraint:* ALFRED NSA supersector data is incomplete.
* **Collection Interval:** Hand-build from historical BLS release calendars and the 12th-of-the-month rule. (Monthly, deterministic).
* **BLS Appropriations:** Annual Congressional Budget Justifications (CBJs). Requires hand-extraction of nominal levels, deflated via GDP deflator. (Annual, available 1980–present).
* **BLS FTE Staffing:** OPM FedScope. (Quarterly, 1998–present).
* **QCEW Vintages:** *Severe Constraint.* BLS does not formally archive QCEW release vintages in a public API. Proprietary sets or intensive scraping of BLS flat files (2000–present) are required.
* **Partisan Composition:** Descriptive dummy variables for House/Senate/White House control.

---

## 4. BAYESIAN RESEARCH DESIGN

### I. Estimand and State-Space Formulation

The objective is to estimate the scale (variance) of revisions driven by covariates. We adopt the Jacobs-van Norden (2011) framework but extend the measurement-error variance to a time-varying, covariate-driven stochastic volatility structure.

Let $`x_t`$ be the latent "true" log employment level. Let $`y_{t}^{(v)}`$ be the $`v`$-th vintage estimate ($`v \in \{1, 2, 3, BM\}`$). To handle heavy tails, we use a Student-t measurement density:

```math
y_{t}^{(v)} \sim \text{StudentT}\left(\nu, \ x_t + c^{(v)}, \ \exp(h_{t}^{(v)})\right)
```

where $`c^{(v)}`$ handles any structural mean bias (e.g., B-D model lag at turning points). We test the "Noise" specification ($`y_t^{(v)} = x_t + \epsilon_t^{(v)}`$) against the "News" specification ($`x_t = y_t^{(v)} + \epsilon_t^{(v)}`$) via WAIC/LOO, but focus the parameterization on the log-scale:

```math
h_{t}^{(v)} = \alpha^{(v)} + Z_t \beta^{(v)} + \phi_{t}^{(v)}
```

where $`Z_t`$ contains standard scaled covariates: collection days, real budget, FTEs, and structural break indicators. $`\phi_{t}^{(v)} \sim N(0, \sigma_\phi^2)`$ captures baseline stochastic volatility.

### II. The Benchmark and Wedge-Back

The annual comprehensive March benchmark (released the following February) is treated as a highly precise, irregularly timed observation: $`y_t^{(BM)} \sim N(x_t, \sigma_{BM}^2)`$ where $`\sigma_{BM}^2 \to 0`$.

To capture the BLS wedge-back, the measurement equations for the 11 intervening months must include a correlation structure tied to the benchmark error. Let $`e_t^{(3)} = y_t^{(3)} - x_t`$. The benchmark revision at month 12 is $`\Delta_{12} = y_{12}^{(BM)} - y_{12}^{(3)}`$. The wedge-back imposes that the revision for month $`t`$ (where $`t \in 1..11`$ since the last benchmark) is linearly interpolated. We enforce this by defining the covariance matrix of the measurement errors such that the off-diagonals for months $`t`$ and $`t+k`$ within a benchmark year reflect this deterministic allocation.

### III. Seasonality and NSA/SA Joint Modeling

To isolate sample revision from SA-factor revision, we jointly model the NSA and SA vintages:

```math
y_{t, NSA}^{(v)} = x_{t, NSA} + \epsilon_{t, sample}^{(v)}
```

```math
y_{t, SA}^{(v)} = y_{t, NSA}^{(v)} - S_t^{(v)}
```

where $`S_t^{(v)}`$ is the vintage-$`v`$ seasonal factor. The variance of $`S_t^{(v)}`$ is modeled separately as a function of calendar effects (moving holidays) and a structural break at 2003 (transition to concurrent adjustment).

### IV. Hierarchy and Structural Change

Parameters are partially pooled across the 13 NAICS supersectors (subscript $`j`$):

```math
\beta_{j}^{(v)} \sim \text{MVN}(\mu_{\beta}^{(v)}, \ \Sigma_{\beta})
```

To avoid Neal's funnel, we use a non-centered parameterization (the "Matt trick"): $`\beta_j = \mu_\beta + L \cdot z_j`$, where $`L`$ is the Cholesky factor of $`\Sigma`$ and $`z_j \sim N(0, I)`$.

Structural changes (2003 NAICS/SA shift; 2020-2022 COVID outlier period) are handled via deterministic change-points in $`\alpha^{(v)}`$, while the COVID period features a temporary masking regime where $`\nu`$ (degrees of freedom) is allowed to drop to accommodate extreme tail events.

### V. Identification, Confounding, and Priors

Collection days provide high-frequency, well-identified variation. Conversely, real budget, FTEs, and sample-size changes are slow-moving and highly collinear. We deploy a Regularized (Finnish) Horseshoe prior on these slow-moving covariates to enforce sparsity and prevent them from absorbing secular trend drift:

```math
\beta_k \sim N(0, \tau^2 \tilde{\lambda}_k^2)
```

*Priors in natural units (thousands of jobs, modeled in log-levels so coefficients are approximate percentage effects):*

* $`\alpha^{(1)}`$ (Baseline 1st print variance scale): $`N(\log(0.002), 0.5)`$ — implies baseline revision standard error of ~0.2% of total payrolls.
* $`\nu`$ (Degrees of freedom): $`\text{Gamma}(2, 0.1) + 2`$ to ensure finite variance.
* Prior Predictive Check Plan: Simulate vintages from the priors to ensure 1st $`\to`$ 3rd revisions rarely exceed $`\pm 500k`$ jobs under normal non-recessionary conditions.

### VI. Implementation & Validation

* **Stack:** Python with `JAX`. Data manipulation in `Polars`.
* **Efficiency:** Because $`x_t`$ is highly autocorrelated and high-dimensional, sampling it jointly with hyperparameters via NUTS will yield poor mixing. We will use `Dynamax` (JAX-based state-space modeling) to *analytically marginalize* the linear-Gaussian state $`x_t`$ using the Kalman filter inside the likelihood step. `NumPyro` will run HMC/NUTS exclusively on the structural variance parameters ($`\beta, \tau, \alpha`$).
* **Diagnostics & Validation:** `ArviZ` for $`\hat{R}`$ and ESS. Use PSIS-LOO to compare the structural variance model against a baseline homoskedastic model. Conduct pseudo-real-time out-of-sample posterior predictive checks on the 1-month revision dispersion for 2018–2023.

### VII. Analytic Advantage

This design explicitly outperforms frequentist event studies or OLS-based heteroskedastic regressions (e.g., regressing squared revisions on budget) by correctly treating the "true" employment level as a latent parameter, strictly enforcing the structural constraints of the annual wedge-back, and avoiding attenuation bias caused by treating the final benchmark as truth without measurement error.

---

## 5. ANNOTATED BIBLIOGRAPHY

* **Aruoba, S. B. (2008). "Empirical Properties of Data Revisions."** *Journal of Business & Economic Statistics.*
  *(Peer-Reviewed)* | Establishes the standard summary statistics and normality/bias tests for macro revisions. Foundational for the baseline error structures.
* **Cajner, T., et al. (2018). "Improving the Accuracy of Economic Measurement with Multiple Data Sources: The Case of Payroll Employment Data."** *NBER Working Paper No. 24716.*
  *(Grey Literature/NBER)* | Essential for understanding how the birth-death model fails at business cycle turning points, shifting variance to the benchmark print.
* **Corrado, C., & Haltmaier, J. (2012). "The Use of High-Frequency Data in Nowcasting."** *FEDS Working Papers.*
  *(Agency Document/FEDS)* | Highlights CES dynamics and the Mankiw-Shapiro properties of early payroll prints.
* **Groen, J. A., et al. (2017). "Late Reporting and the Revision Properties of Establishment Survey Estimates."** *Journal of Official Statistics.*
  *(Peer-Reviewed/OSMR)* | Directly links collection intervals and late response to the mean and variance of CES revisions; critical for Gap 3.
* **Jacobs, J. P., & van Norden, S. (2011). "Modeling Data Revisions: Measurement Error and Dynamics of 'True' Values."** *Journal of Econometrics.*
  *(Peer-Reviewed)* | The definitive state-space treatment of news vs. noise, providing the architectural foundation for Part C.
* **Phipps, P., & Toth, D. (2012). "Analyzing Establishment Nonresponse Using an Architecture-Based Framework."** *Survey Methodology.*
  *(Peer-Reviewed)* | BLS OSMR paper exploring the secular decline in CES collection rates and the non-random nature of late reporters.
* **Stuart, B., & Weng, E. (2021). "Handling extreme values in the Current Employment Statistics survey."** *Monthly Labor Review.*
  *(Agency Documentation/MLR)* | Documents the structural breaks and additive outlier interventions applied to X-13ARIMA-SEATS during COVID.
