**MEMORANDUM**

**TO:** Senior Labor-Market Statistician

**FROM:** Quantitative Research Assistant

**SUBJECT:** Bayesian State-Space Design for Modeling CES Revision Magnitude

## 1. EXECUTIVE SUMMARY

This memorandum evaluates the structural, operational, and institutional drivers of Current Employment Statistics (CES) revision *magnitude* and outlines a concrete Bayesian state-space design to formally identify its variance drivers. While macroeconometric literature vigorously evaluates the mean-bias and predictability of revisions (e.g., Mankiw-Shapiro, Jacobs-van Norden), rigorous structural modeling of revision *dispersion*—specifically mapped to statistical agency constraints—remains sparse.

The literature agrees that high-frequency CES variance is mechanically governed by calendar-driven collection intervals and concurrent seasonal adjustment recalculations. Conversely, slow-moving institutional constraints—stagnant BLS funding, statistical staffing attrition, and a secular decline in survey response rates—are universally asserted by stakeholders as degrading estimate quality. However, these low-frequency trends are deeply confounded, rendering frequentist causal attribution econometrically invalid.

To address these gaps, we propose a multi-vintage, non-centered stochastic volatility state-space model. Extending the Jacobs-van Norden framework, we model the four revision stages as noisy signals of a latent employment state. Covariates enter the log-variance of Student-$`t`$ measurement equations. Crucially, the design employs regularized horseshoe shrinkage to handle collinear institutional decay, leverages Kalman smoothing to probabilistically replicate the deterministic post-benchmark wedge-back, and explicitly isolates seasonal-filter noise from sample accumulation by jointly modeling SA and NSA vintages. Built via NumPyro and Dynamax, the model utilizes Rao-Blackwellization to analytically marginalize the latent state, ensuring sampling efficiency while explicitly quantifying the operational cost of agency constraints.

---

## 2. PART A — LITERATURE REVIEW

### The Broader Data-Revision Context

The theoretical taxonomy of data revisions stems from Mankiw and Shapiro (1986), separating "news" (initial prints are optimal forecasts; revisions are orthogonal to early estimates) from "noise" (initial prints contain classical measurement error; revisions are orthogonal to the final truth). Croushore and Stark (2001) operationalized these tests via the ALFRED database. Faust, Rogers, and Wright (2005) and Aruoba (2008) demonstrated that initial macroeconomic prints violate rationality, exhibiting heavy tails and predictable heteroskedasticity. Jacobs and van Norden (2011, JvN) advanced the field with state-space models that jointly estimate news and noise by modeling "truth" as a latent state, a structure Kishor and Koenig (2012) extended to real-time VARs. Fixler and Grimm (2002, 2008) applied these decomposition frameworks extensively to BEA National Accounts. However, for CES, direct application of JvN frameworks targeting the time-varying *variance* of the four revision stages remains heavily under-researched.

### 1. Seasonality

*(Decomposition sources: ii. seasonal factor revision)*

* **Known:** The 2003 transition from 6-month projected X-12-ARIMA factors to concurrent seasonal adjustment eliminated historical mean bias but explicitly inflated the variance of the 1-month and 2-month SA revisions, as new sample accumulation recalculates the entire historical filter. The shift to X-13ARIMA-SEATS standardized moving holidays and 4- vs. 5-week reference interval handling. During the 2020–2022 COVID shock, X-13 filters failed; BLS manual adjudication of Additive Outliers (AO) versus Level Shifts (LS) injected immense discretionary variance into SA revisions.
* **Asserted/Contested:** It is accepted heuristically that the 1st $`\to`$ 2nd print is dominated by sample accumulation, while the 2nd $`\to`$ 3rd is heavily influenced by concurrent SA recalculation. However, rigorous state-space decompositions separating SA filter variance from NSA sample variance are virtually absent.

### 2. Collection Interval

*(Decomposition sources: i. late sample reporting)*

* **Known:** The calendar placement of the 12th relative to the first Friday yields a collection window of 10 to 16 days. BLS OSMR literature confirms a secular decline in first-closing response rates from $`>70\%`$ in 1999 to $`\sim40\%`$ post-pandemic.
* **Asserted/Contested:** Shorter intervals mechanically depress first-closing rates, heavily increasing reliance on weighted link relative (WLR) imputation. It is mathematically asserted that this inflates 1-month revision dispersion. However, formal nonresponse bias literature for the CES generally models conditional means; the structural elasticity of heteroskedastic variance to exact collection days is rarely modeled in a robust multi-year setting.

### 3. BLS Funding

*(Decomposition sources: i. late reporting, iv. benchmark realignment)*

* **Known:** Real BLS appropriations have trended downward, exacerbated by continuing resolutions, the 2013 sequestration, and shutdowns (2013, 2018–2019) that forced collection delays and suspended series.
* **Asserted/Contested:** Advocacy groups (ASA, COPAFS) relentlessly assert that budget erosion degrades Non-Response Follow-Up (NRFU) and widens benchmark error tails. However, causal identification is effectively nonexistent in peer-reviewed literature. Real funding perfectly covaries with secular survey fatigue. Note: The partisan composition of appropriating chambers is strictly a descriptive covariate; using it as a causal instrument violates exclusion restrictions given concurrent macro trends.

### 4. Sample Design

*(Decomposition sources: iii. NBD error, v. NAICS/redesign)*

* **Known:** The 1999–2003 transition from a quota sample to a probability-based stratified sample fundamentally altered the variance generating process by rotating in slower-reporting small firms. Simultaneously, the Net Birth-Death (NBD) ARIMA model replaced static bias factors.
* **Asserted/Contested:** The NBD model heavily shifts error to the 3rd $`\to`$ Benchmark stage during cyclical turning points (since ARIMA relies on lagged QCEW trends). The degree to which the probability transition independently *caused* higher preliminary variance versus simply uncovering actual noise masked by the old quota system remains contested.

### 5. Staffing

*(Decomposition sources: i. late reporting, iv. benchmark realignment)*

* **Known:** OPM FedScope tracks long-term attrition and aging in federal statistical FTEs.
* **Asserted/Contested:** It is posited that FTE loss diminishes manual micro-editing of X-13 outliers and delays complex QCEW non-economic code adjudication. No formal econometric literature parameterizes revision dispersion as a function of agency headcount.

---

## 3. PART B — GAP ANALYSIS

### Gap Table

| Open Question | Closest Existing Work | Why It Falls Short (Shortcomings) | Data Needed | Feasibility |
| --- | --- | --- | --- | --- |
| **Isolating SA vs. NSA Sample Variance** | Groen et al. (2003); BLS concurrent SA notes | Evaluates MAE purely pre-COVID; does not model SA/NSA jointly in a structural state-space. | Joint real-time `PAYEMS` (SA) and `PAYNSA` (NSA) vintages by NAICS. | **High.** Full multi-vintage triangles are intact on ALFRED. |
| **Causal Impact of Collection Days on Variance Scale** | OSMR nonresponse bias reports | Cross-sectional regressions on mean bias; fails to model stochastic volatility. | Calendar days (12th to release); 1st closing response rates. | **High.** Days are deterministic. Closing rates require scraping BLS notes. |
| **State-Space Formulation of the BLS Wedge-Back** | Jacobs-van Norden (2011) | Standard JvN treats benchmarks as independent draws; ignores the CES deterministic linear 11-month interpolation. | Real-time unadjusted 3rd prints vs. finalized March QCEW levels. | **Medium.** QCEW vintages are poorly archived publicly; must be backed out of March benchmark articles. |
| **Funding & Staffing Effects on Revision Noise** | GAO (2026); ASA grey literature | Descriptive only. Hopelessly collinear with macro secular trends; lacks Bayesian shrinkage. | Real BLS appropriations, CR dummies, FedScope FTEs. | **Medium.** Variables must be hand-built. Causal ID is fundamentally weak. |

### Priority Gaps

The most pressing gap is the **decomposition of concurrent SA filter drift from physical sample accumulation**. Because the 2003 NAICS, probability sample, and concurrent SA shifts occurred simultaneously, frequentist models conflate them. Jointly mapping real-time SA and NSA vintages in state-space separates the algorithmic X-13 variance penalty from the late-reporting unit variance penalty.

Second is **probabilistic wedge-back modeling**. Applying standard macro state-space frameworks to CES fails because the April–February benchmarked prints are not independent observations of truth; they are deterministic linear interpolations of the March QCEW error. Encoding this mechanically via a Kalman smoother backward pass represents a major methodological correction for payroll nowcasting.

### Data Availability Inventory

* **CES Vintages:** Monthly. High quality. ALFRED API serves `PAYEMS` and `PAYNSA` back to 1997.
* **QCEW Benchmarks:** *Constraint.* True QCEW vintages are overwritten by BLS and ALFRED lacks deep NAICS detail. Real-time QCEW states must be reverse-engineered from benchmark deltas published in historical February/March BLS releases.
* **Collection Interval:** Complete availability. Hand-built from deterministic reference calendar rules.
* **Institutional:** Real appropriations (annual) and OPM FedScope FTEs (quarterly) must be hand-built from DOL Congressional Budget Justifications.

---

## 4. PART C — BAYESIAN RESEARCH DESIGN

### 1. Estimand & State-Space Structure

The estimand is the **scale (variance)** of revisions. Let $`x_{t, c}`$ be the unobserved latent true NSA log-employment for month $`t`$ and NAICS supersector $`c`$. The state evolves via an AR(1) process with stochastic volatility.

We observe vintages $`v \in \{1, 2, 3\}`$. To isolate SA filter updates from sample updates, we map NSA and SA vintages jointly to the state:

```math
y_{t, c}^{(v, NSA)} = x_{t, c} + \varepsilon_{t, c}^{(v, NSA)}
```

```math
y_{t, c}^{(v, SA)} = y_{t, c}^{(v, NSA)} - S_{t, c}^{(v)} = x_{t, c} - S_{t, c}^{(v)} + \varepsilon_{t, c}^{(v, NSA)}
```

*(Where $`S`$ is the concurrent seasonal factor).*

Following JvN, identification of news vs. noise relies on the covariance matrix of the state innovation and the measurement errors. If $`\text{Cov}(\varepsilon^{(v)}, x_t) = 0`$, the revision removes classical noise. If $`\text{Cov}(\varepsilon^{(v)}, \varepsilon^{(v+1)}) = \text{Var}(\varepsilon^{(v+1)})`$, the initial print was an optimal forecast (news). Note: Intentional BLS smoothing of early prints violates these rationality assumptions.

### 2. Scale Parameterization

Covariates enter the log-variance equations. To capture heavy-tailed NBD turning-point failures and COVID outliers, innovations are Student-$`t`$:

```math
\varepsilon_{t, c}^{(v, NSA)} \sim \text{Student-}t(\nu_v, 0, \sigma_{t, c, v})
```

```math
\log \sigma_{t, c, v} = \alpha_{c, v} + h_{t, c} + \mathbf{Z}_t \boldsymbol{\beta}_v + \mathbf{W}_t \boldsymbol{\gamma}_v
```

* $`h_{t, c}`$: Latent AR(1) stochastic volatility (capturing unobserved macro turbulence).
* $`\mathbf{Z}_t`$: Fast covariates (Collection interval days, 4- vs 5-week reference indicators).
* $`\mathbf{W}_t`$: Slow covariates (Real budget, FTEs, 1st closing rates).

### 3. Benchmark Wedge-Back Handling

The March benchmark ($`v = BM`$) is an irregularly timed, high-precision observation: $`y_{Mar, c}^{(BM)} \sim \mathcal{N}(x_{Mar, c}, \sigma_{BM}^2)`$ where $`\sigma_{BM} \to 0`$.

We do *not* treat the intervening April–February published benchmarked months as independent measurements. Instead, by anchoring March with near-infinite precision, the Kalman smoother (RTS backward pass) inherently propagates this precision backwards through the state transition covariance, naturally bridging the error across the prior 11 months. This provides a formal probabilistic foundation for the BLS linear wedge.

### 4. Hierarchy & Structural Change

* **Hierarchy:** We apply partial pooling across NAICS supersectors using a **non-centered parameterization (NCP)** to eliminate Neal's funnel: $`\alpha_{c, v} = \bar{\alpha}_v + \tau_{\alpha, v} \tilde{\alpha}_{c, v}`$, with $`\tilde{\alpha} \sim \mathcal{N}(0, 1)`$.
* **Structural Change:** The confounded 2003 transitions (NAICS + Concurrent SA + Probability sample) and 2020–2022 outlier regimes are handled via time-varying parameters with shrinkage, utilizing the Bitto & Frühwirth-Schnatter (2019) dynamic regularized horseshoe on the state variances of the intercepts.

### 5. Identification & Shrinkage

Collection intervals ($`\mathbf{Z}_t`$) are highly identified via intra-year deterministic variance. Conversely, funding, staffing, and response-rate decay are collinear and non-stationary over a 40-year sample. We apply a **Regularized Horseshoe prior** to $`\boldsymbol{\gamma}_v`$. This provides honest shrinkage, assuming the marginal causal effect of budget levels on variance is strictly zero unless the data provides overwhelming evidence to override the penalty. Party-of-appropriators is strictly excluded from the design matrix.

### 6. Priors (Units: Thousands of Jobs)

Working in log-thousands. A typical 1-month revision dispersion is $`\sim`$40k jobs ($`\log(40) \approx 3.7`$).

* Intercepts: $`\bar{\alpha}_1 \sim \mathcal{N}(3.7, 1.0)`$
* Degrees of Freedom: $`\nu_v \sim \text{Gamma}(2, 0.1)`$ bounded $`>2`$ for finite variance.
* Horseshoe Global Scale: $`\tau_0 \propto \frac{p_0}{D-p_0}\frac{1}{\sqrt{T}}`$, tuned to $`p_0=1`$ expected active slow covariate.
* **PPC Plan:** Simulate the prior generative model to verify that implied 1st-print max revisions remain bounded between 10k and 300k jobs during normal regimes before seeing data.

### 7. Implementation Notes (Python/JAX)

* Build the full hierarchy in **NumPyro** utilizing the NUTS sampler. Manage the vintage panel in **Polars**, and run MCMC diagnostics (ESS, $`\hat{R}`$) in **ArviZ**.
* **Marginalization (Crucial):** Exact analytical marginalization of the continuous latent state $`x_t`$ over 400 months is impossible with standard Kalman filtering if observation errors are Student-$`t`$; brute-force HMC on the state will mix disastrously. **Solution:** Parameterize the Student-$`t`$ as a Gaussian Scale Mixture by sampling local variance multipliers $`\lambda_{t, v} \sim \text{InvGamma}`$. Conditioned on $`\lambda`$, the system is strictly linear-Gaussian. Pass these parameters to **Dynamax** (`LinearGaussianSSM`) inside the NumPyro block to evaluate the marginal log-likelihood via Rao-Blackwellized Kalman filtering, accelerating MCMC geometry by orders of magnitude. (Use **Blackjax** if customizing the inner NUTS/Gibbs transition kernel is required for the scale mixture).

### 8. Validation & Conclusion

Validate via PSIS-LOO cross-validation against a homoskedastic baseline. Target out-of-sample posterior predictive checks specifically on the empirical interquartile range and 95th percentiles of the $`(y^{(2)} - y^{(1)})`$ distribution.

**Advantage over baselines:** A frequentist heteroskedastic regression directly on $`(y^{(2)} - y^{(1)})`$ treats early prints as fixed, conflates seasonal-filter instability with sample accumulation, cannot handle the multi-step deterministic wedge-back, and hallucinates causal precision on collinear budget variables. This Bayesian design explicitly models the unobserved truth, dynamically allocates variance across operational components, and provides honest shrinkage on institutional parameters.

---

## 5. ANNOTATED BIBLIOGRAPHY

1. **Aruoba, S. B. (2008). "Data Revisions Are Not Well Behaved."** *Journal of Money, Credit and Banking.*
   *[Peer-Reviewed]* Foundational econometric proof that U.S. macro prints fail rational-forecast tests, exhibiting predictable variance; essential for justifying non-Gaussian SV models.
2. **Bitto, A., & Frühwirth-Schnatter, S. (2019). "Achieving Shrinkage in a Time-Varying Parameter Model Framework."** *Journal of Econometrics.*
   *[Peer-Reviewed]* Provides the econometric justification for dynamic regularized horseshoe shrinkage in state-space, critical for handling the 2003 structural breaks without overfitting.
3. **Croushore, D., & Stark, T. (2001). "A Real-Time Data Set for Macroeconomists."** *Journal of Econometrics.*
   *[Peer-Reviewed]* Establishes the standard methodology and pitfalls for analyzing multi-vintage macro matrices; genesis of the ALFRED framework.
4. **Faust, J., Rogers, J. H., & Wright, J. H. (2005). "News and Noise in G-7 GDP Announcements."** *Journal of Money, Credit and Banking.*
   *[Peer-Reviewed]* Provides the econometric identification strategy for restricting early-vintage covariance to isolate noise from underlying state volatility.
5. **Fixler, D. J., & Grimm, B. T. (2008). "The Reliability of the GDP and GDI Estimates."** *Survey of Current Business.*
   *[Agency]* Canonical agency literature connecting comprehensive target-shifting benchmarks to revision magnitude; the theoretical BEA equivalent to QCEW benchmarking.
6. **Jacobs, J. P., & van Norden, S. (2011). "Modeling Data Revisions: Measurement Error and Dynamics of 'True' Values."** *Journal of Econometrics.*
   *[Peer-Reviewed]* The definitive state-space methodology for mapping news and noise errors; the mathematical backbone of the proposed research design.
7. **Kishor, N. K., & Koenig, E. F. (2012). "VAR Estimation and Forecasting When Data Are Subject to Revision."** *Journal of Business & Economic Statistics.*
   *[Peer-Reviewed]* Highlights the "apples and oranges" forecasting problem, demonstrating the necessity of joint vintage modeling for real-time optimal forecasting.
8. **Kropf, J., et al. (2003). "Concurrent seasonal adjustment for national CES survey."** *Monthly Labor Review.*
   *[Agency]* Primary technical documentation detailing the 2003 shift from projected to concurrent seasonal factors and its explicit mechanical impact on early-vintage variance.
9. **Mankiw, N. G., & Shapiro, M. D. (1986). "News or Noise: An Analysis of GNP Revisions."** *Survey of Current Business.*
   *[Peer-Reviewed]* The classical theoretical framework for evaluating whether a preliminary release is an efficient forecast (news) or a noisy measurement (noise).
10. **Piironen, J., & Vehtari, A. (2017). "Sparsity information and regularization in the horseshoe and other shrinkage priors."** *Electronic Journal of Statistics.*
    *[Peer-Reviewed]* Methodological justification for the regularized horseshoe prior deployed to handle highly collinear BLS institutional/budget covariates.
