# ces-revisions — Design Spec

> For agentic workers: REQUIRED NEXT SKILL: derive-roadmap — do not plan
> this spec directly and do not split it into per-subsystem plans.

A Bayesian multi-vintage state-space model of the **scale** of U.S. CES payroll-employment
revisions across four stages — first → second print, second → third, third → annual benchmark,
and post-benchmark wedge-back — in which collection-interval, seasonal-factor, sample,
birth–death, and institutional-capacity covariates enter the log-scale of Student-*t* news and
noise measurement errors on a latent, concept-consistent employment path. Eleven NAICS
supersectors are partially pooled; the annual benchmark enters as a fallible, irregularly timed
anchor whose published wedge-back is a known operator; NSA and SA vintages are modeled jointly
so seasonal-factor revision is separated from sample revision. Alongside the model, the project
delivers a primary-source-verified literature review and data inventory reconciling the three
research drafts in `specs/`.

Design provenance: this spec synthesizes the research brief
[`specs/ces-revisions-prompt.md`](ces-revisions-prompt.md) with three independent, unverified,
first-pass responses to it — [`ces-revisions-research-chatgpt.md`](ces-revisions-research-chatgpt.md),
[`ces-revisions-research-claude.md`](ces-revisions-research-claude.md), and
[`ces-revisions-research-gemini.md`](ces-revisions-research-gemini.md). None was adjudicated
before synthesis, so every accept/reject was made in the 2026-09-12 triage session, not
inherited. Locators: **(prompt Part X)** for the brief, **(chatgpt §n)**, **(claude §n)**,
**(gemini §n)** for the drafts. Markers: **(chosen)** / **(rejected)** record triage
adjudications and the five user decisions of 2026-09-12; **(open — resolved by verification, not
argument)** marks a matter of fact outside this repo, each discharged by a named Verification
bullet. Numeric facts quoted below are those on which at least two drafts and the
`bls-data-context` reference agree; everything else is the written finding's job (Req 20).

## Motivation — the gaps

The literature characterizes the **sign and mean bias** of CES revisions well and their
**conditional scale** hardly at all (prompt Part A; chatgpt §1.2, §2.7; claude §1):

- BLS's own 1979–present table gives the only long, consistent view of stages 1–2, and it is a
  two-regime average, not a trend: SA mean absolute revisions of 48/29/61 thousand
  (second−first / third−second / third−first) before May 2003 and 33/34/51 after, with 2020–21
  first-to-third MARs of 130 and 181 thousand against 34 in 2019 (chatgpt §1.2). The May 2003
  break bundled NAICS, the probability sample, concurrent seasonal adjustment, and a new
  federal method, so the table cannot attribute its change to any one of them (chatgpt §2.1;
  claude §2 Driver 1).
- No published work decomposes the SA monthly revision into an NSA-sample component and a
  seasonal-factor component at production granularity. Wright (2013) is the closest —
  a real-time implied-factor proxy contaminated by benchmark and window effects — and Loya
  (2014) isolates only a fourth-vs-third seam of ~9.6 thousand (chatgpt §2.1). Absolute SA and
  NSA MARs are not additive contributions; the covariance must be estimated (chatgpt §1.2).
- Collection days vary 9–16 with the calendar and are the one sharply varying driver.
  Copeland (2003) links short windows to late reporting, but the link to revision
  **dispersion** rests on pooled correlations of ~0.18 over two years and four industries
  (chatgpt §2.2). Collection rates in 2024 averaged 60.4 / 89.0 / 90.9 percent at the three
  closings (C1–C3 series, monthly from January 2000); the distinct third-release response rate
  fell from ~58 percent pre-2020 to ~43 percent in 2024 (chatgpt §2.2; claude §2 Driver 2).
  The gemini draft's "~40 percent first-closing rate" conflates the two (rejected).
- The annual benchmark is the difference between two separately produced, errorful counts:
  QCEW is revised, BLS does not archive its vintages, and ALFRED's QCEW holdings lack industry
  detail. The published wedge distributes one March gap linearly over eleven months, and
  Robertson (2021) shows that linear error path is wrong on average (chatgpt §2.6, §3.3;
  claude §3). The March 2025 final benchmark was −862 thousand NSA (−861 after a scope
  reconstruction), −898 SA; the preliminary was −911 (chatgpt §2.6; claude §1;
  `bls-data-context`).
- No study — peer-reviewed or agency — relates real BLS appropriations, continuing
  resolutions, sequestration, shutdown exposure, or FTE to any CES revision scale. Budget
  authority, FTE, and sample size are slow, trending, collinear, and definition-broken
  (chatgpt §2.3, §2.5; claude §2 Drivers 3, 5). The 2018–19 lapse did not close BLS at all
  (chatgpt §2.3).
- The Jacobs–van Norden news/noise state-space tradition, Kishor–Koenig real-time filtering,
  and Clements–Galvão stochastic-volatility extensions exist, but no national implementation
  models the first, second, third, preliminary-benchmark, final-benchmark, wedge, and mature
  vintages jointly with covariate-driven scale (chatgpt §2.7; claude §2 "Situating CES").

The three drafts disagree with each other on facts (collection vs response rates; whether the
2018–19 lapse closed BLS; whether 2025 program cuts touched CES; FTE concepts) and none checked
its numbers against primary sources — the second half of this project's motivation.

## Core principle

The estimand is the **conditional scale** of stage-specific revisions of a latent,
concept-consistent employment path. Every published number enters through a measurement
equation that states how BLS produced it — the three closings as nested information sets, the
benchmark as a fallible anchor, the wedge and post-March recursion as known operators, the
mature vintage as a fixed rule — so that the model never fabricates information the data do not
carry. Every covariate coefficient is identified off *named* variation (within-year calendar,
sector-year exposure, annual capacity deviations) or shrunk honestly toward zero, and the model
says which is which.

## Requirements

### Data foundation

**Req 1 — Vintage panel.** One long Polars table keyed by
`(sector, reference_month, release_date, vintage_id, release_stage, seasonal_status,
value_thousands, concept_regime)`, with an immutable raw-value table and a separate
transformations table so every derived number is reproducible (chatgpt §4.1, §4.14).
Sources: BLS CES vintage files (SA and NSA, supersector detail, every Employment Situation
release from the May 2003 publication vintage; nonstandard releases flagged from the file
comments); the BLS 1979–present revision table and the Philadelphia Fed RTDSM `EMPLOY` matrix
for the total-nonfarm aggregate leg; the CES benchmark technical table (final revisions 1979+,
preliminary 2000+) and archived benchmark articles for sector benchmark contributions
(chatgpt §3.3; claude §3). Over-the-month changes are constructed only with a **same-release
differencing operator** that subtracts the prior month's value in the same release file; stage
labels are never differenced across release files (chosen — chatgpt §4.1; that is the only
construction that reproduces the official revision table).

**Req 2 — Stage definitions.** Five measurement states and four transitions (chosen —
chatgpt §4.1; the four-vintage-with-terminal-benchmark layout of claude §4.1 and gemini §1
rejected because it leaves stage 4 unobservable): `F` first release; `S` second; `T`
third/final sample-based; `B` the first release incorporating the **final** March benchmark;
`M` a **fixed** mature horizon — for NSA the vintage of the second annual benchmark after the
reference month; for SA the first benchmark vintage in which the reference month has left
BLS's five-year SA revision window — with later reconstructions separately flagged. `M` is never "latest
available", which would give older months more revision opportunity and manufacture
vintage-age heteroskedasticity. The preliminary August/September benchmark is **not** a sixth
vintage: it does not revise the series and enters as an auxiliary observation of the final
March discrepancy (Req 10). Reference months whose `M` vintage does not yet exist are
right-censored: `M` is integrated out and reported as a posterior predictive, never shortened.

**Req 3 — Covariate panel.** Hand-built, provenance-tagged, with a definition-break flag per
series; annual series are **never** interpolated to monthly (chatgpt §2.5, §3.3; claude §3):

- Collection window: $`D_t`$ = federal business days from the first business day after the 12th
  through first closing, computed from archived release and closing schedules, with separate
  indicators for federal holidays, nonstandard releases, and lapses in appropriations
  (chatgpt §2.2). Collection rates `CEU00000000C1/C2/C3` (monthly, January 2000+) and the
  total-private third-release response rate `CEU05000000RR` (2009+), never conflated
  (chatgpt §2.2; the gemini §2 figure rejected). Realized first-close coverage is split into a
  calendar-predicted component (a first-stage equation on business days, holidays, release
  features, sector, and month effects) and its residual; only the residual enters the scale
  equation, as an associational term (chatgpt §4.4).
- Seasonal: four/five-week interval, Easter/Labor Day/outlier flags, annual-specification
  regime, BLS COVID intervention-treatment indicators, from the CES seasonal-adjustment files
  (chatgpt §2.1, §4.4).
- Net birth–death: published monthly forecasts by supersector and the benchmark-article
  forecast-vs-realized tables; zero for government (chatgpt §2.4; claude §3; user decision Q4).
- Sample: annual benchmark technical tables — usable linked employment coverage and RSE by
  supersector — harmonized across the organization/UI-account/worksite definition switches
  (chatgpt §2.4, §3.3).
- Institutional: real BLS budget authority (General Fund plus Unemployment Trust Fund
  transfer, deflated), fiscal-year-average FTE from successive Congressional Budget
  Justifications cross-checked against OPM September headcount, CES federal/state/contractor
  workyears where reported; continuing-resolution days, extensions, and full-year status;
  shutdown flags coded for whether **BLS itself** closed, collection stopped, the release was
  delayed, and the window extended — never a generic federal-shutdown dummy (chatgpt §2.3,
  §2.5, §4.12; claude §2 Driver 3). Reach: FY2009–present in v1 (chosen — user decision Q5;
  reconciled series exist in the transition brief and FY2027 CBJ); extension to FY1979 from
  archived CBJs and OMB historical tables is a later stage (Out of scope for v1).
- Party composition of the appropriating chambers: descriptive grouping only, stored for a
  residual-scale table, **never** in the likelihood (chosen — chatgpt §4.8; prompt Part C;
  the tight varying intercept of claude §4.7 rejected — a coefficient in the likelihood is
  still a coefficient readers will quote).
- Macro/tail controls (recession dates, claims, strikes, severe weather) aligned to the
  information set at each release date (chatgpt §3.3).

**Req 4 — Estimation window.** Supersector NSA and SA vintages from the May 2003 publication
vintage forward, plus the total-nonfarm F/S/T revision series from January 1979 appended as a
partially observed upper level with one composite pre/post-May-2003 regime shift in intercept
and scale (chosen — user decision Q2; chatgpt §4). Pre-2003 sector vintages are integrated out,
never fabricated. The 1964+ RTDSM SA-only extension is rejected for v1 (no NSA counterpart;
quota-sample method history to code).

**Req 5 — Sector set and aggregation.** The eleven mutually exclusive BLS supersectors,
government included, in one hierarchy (chosen — user decision Q4; a separate government leg and
a private-only estimand rejected). Government's non-probability design, systematic late
reporting, and absent birth–death component are carried by sector-specific deviations on the
collection and coverage effects and a zero birth–death covariate. Total nonfarm is the sum of
the eleven sector states with a tight rounding/reconciliation likelihood (chatgpt §4.2); the
published total is also observed in its own right so the aggregate leg of Req 4 binds.

### Model

**Req 6 — Latent employment process.** For sector $`s`$ and month $`t`$, concept-consistent
NSA employment in thousands decomposes into a smooth state and a mature seasonal/calendar
component (chosen — chatgpt §4.2; gemini §1's stationary AR(1) on a log level rejected as
misspecified and non-additive):

```math
x_{s,t}=\ell_{s,t}+q_{s,t},\qquad
q_{s,t}=\gamma_{s,t}+c_{s,t}'\delta_s,
```

```math
\ell_{s,t}=\ell_{s,t-1}+g_{s,t-1}+\lambda_s f_t+\eta^{\ell}_{s,t},\qquad
g_{s,t}=g_{s,t-1}+\eta^{g}_{s,t},\qquad
f_t=\phi_f f_{t-1}+\eta^f_t,
```

```math
\gamma_{s,t}=-\sum_{h=1}^{11}\gamma_{s,t-h}+\eta^{\gamma}_{s,t},
\qquad
x_{0,t}=\sum_{s=1}^{11}x_{s,t}.
```

$`c_{s,t}`$ holds only concept-consistent calendar regressors for the NSA process, not the
revision-scale drivers. The common labor-market factor $`f_t`$ with sector loadings
$`\lambda_s`$ (employment-weighted mean fixed at 1, weighted sign positive) is required so that a
national turning point is not read as eleven independent measurement errors. Process
innovations are $`t_5`$ scale mixtures whose scale carries a known crisis multiplier
$`\exp\{\kappa_x\mathbf 1(t\in\text{Mar–Jun 2020})\}`$, $`\kappa_x\sim N(\log 50,0.5^2)`$, so
the ~20-million April 2020 movement lives in the latent state rather than in measurement error;
a sensitivity run replaces the multiplier with unrestricted level-jump states. A trigonometric
seasonal state may replace the dummy recursion if diagnostics favor it.

**Req 7 — Multi-vintage news and noise.** Sequential-revelation Jacobs–van Norden measurement
for the closing stages $`j\in\{F,S,T\}`$ (chosen — chatgpt §4.3; claude §4.1 and gemini §1
state the same restrictions less specifically):

```math
y^N_{s,t,j}=x_{s,t}-\sum_{k\in\mathcal F(j)}n^N_{s,t,k}+e^N_{s,t,j},
\qquad
\mathcal F(F)=\{S,T,B,M\},\ \mathcal F(S)=\{T,B,M\},\ \mathcal F(T)=\{B,M\},
```

where $`n^N_{s,t,k}`$ is news first revealed at stage $`k`$ and $`e^N_{s,t,j}`$ transitory
noise. The `B` and `M` increments are **not** free monthly shocks: they are induced by the
benchmark observations and operators of Req 10. Each transition carries one cross-sector news
factor, $`n^N_{s,t,k}=a^N_{s,k}q^N_{t,k}+\tilde n^N_{s,t,k}`$, sign-identified by a positive
total-private-weighted mean loading, so total-nonfarm revision scale is not the root-sum-square
of sector scales. Closing-stage noise follows one shared recursion
$`e^N_{s,t,j}=\rho_e e^N_{s,t,j-1}+\zeta^N_{s,t,j}`$ with $`\lvert\rho_e\rvert<0.8`$ and
innovations independent across sectors given the published-total reconciliation. The primary
specification sets news increments mutually orthogonal conditional on covariates and the latent
state, and noise orthogonal to $`x`$ and all news. Sensitivity models allow stage-specific
$`\rho_e`$, a tightly regularized residual correlation (LKJ(10)), and a nonzero third-stage
noise / benchmark-error correlation (chatgpt §4.3; claude §4.7 — BLS attributed the 2025 gap to
response and nonresponse error shared by CES and QCEW reporters).

**Req 8 — The estimand: covariates in scale with stochastic volatility.** News and noise are
centered Student-*t* with drivers entering log scale (chosen — chatgpt §4.4; the per-sector,
per-stage volatility paths of claude §4.2 and gemini §2 rejected as unidentified on ~280
months and hostile to sampling):

```math
n^N_{s,t,k}\sim t_{\nu^n_k}\!\left(m^n_{s,k},\sigma^n_{s,t,k}\right),
\qquad
e^N_{s,t,j}\sim t_{\nu^e_j}\!\left(0,\sigma^e_{s,t,j}\right),
```

```math
\log\sigma^n_{s,t,k}=\alpha^n_{s,k}+z_{s,t,k}'\beta^n_{s,k}+\rho^n_{s,k}h^n_{t,k},
\qquad
h^n_{t,k}=\phi^n_k h^n_{t-1,k}+\omega^n_k u^n_{t,k},\quad u^n_{t,k}\sim N(0,1),
```

with an analogous, more tightly regularized equation for noise. One volatility path per
transition, loaded onto sectors by $`\rho^n_{s,k}`$ (normalized to weighted mean 1). $`\nu_k`$
captures isolated tails, $`h_{t,k}`$ clustered volatility, and the covariates systematic scale
shifts; all three are reported. The Gaussian constant-variance baseline is the nested model
with $`\nu`$ large and $`h\equiv0`$. The covariate-to-transition map is fixed in advance
(chosen — chatgpt §4.4; the single shared covariate vector with a "post-2015 decline slope" of
claude §4.2 rejected — a deterministic trend dummy is exactly the collinear term the shrinkage
prior should not have to carry):

| Transition / component | Scale covariates | Identification source |
|---|---|---|
| `F→S` NSA news | first-window business days $`D_t`$; calendar-predicted C1; residual realized C1; usable linked coverage/RSE; latent absolute growth and release-time claims volatility; capacity factor $`K_y`$; COVID regime | within-year calendar variation; residual coverage association; sector/time pooling |
| `S→T` NSA news | incremental C3−C2 collection; C2 level; sample adequacy; latent growth; $`K_y`$; COVID | incremental late receipts; stage contrast |
| seasonal-factor news, both closings | change in implied seasonal component; endpoint leverage; four/five-week interval; Easter/Labor Day/outlier flags; annual-specification regime; COVID | joint NSA/SA identity (Req 9) and archived X-13 inputs |
| `T→B` annual news | relative net birth–death forecast (zero for government); sector coverage/RSE; third-release response and annual attrition proxy; QCEW revision-precision proxy; $`K_y`$; recession/turning-point growth; method indicators | sector-year and annual variation — small effective $`n`$ |
| `B→M` | updated birth–death difference; preliminary−final benchmark surprise; annual SA specification change; reconstruction/NAICS flag; benchmark discrepancy magnitude | within-release operators and later vintage changes |

Continuous predictors are standardized over the modern sample; interventions stay 0/1.
Lagged annual covariates never receive twelve-fold information merely by being repeated across
months: likelihood contributions and cross-validation folds are blocked by fiscal year
(chatgpt §4.4, §4.8).

**Req 9 — Joint NSA/SA measurement and the seasonal decomposition.** SA vintages are linked to
the **same** release-specific NSA measurement through the single mature seasonal component
$`q_{s,t}`$ of Req 6 — there is no second free seasonal state (chosen — chatgpt §4.5; the
multiplicative $`\text{SA}=\text{NSA}\times\exp(\gamma)`$ of claude §4.5 rejected because it
breaks conditional linear-Gaussianity and the NSA−SA difference is observed additively
anyway; gemini §1's $`\text{SA}=\text{NSA}-S`$ accepted in spirit, superseded). With
$`n^A_{s,t,k}`$ the seasonal-factor news revealed at stage $`k`$:

```math
y^S_{s,t,j}=x_{s,t}-q_{s,t}
-\sum_{k\in\mathcal F(j)}\left(n^N_{s,t,k}-n^A_{s,t,k}\right)
+e^N_{s,t,j}-e^A_{s,t,j},
\qquad
A_{s,t,j}=y^N_{s,t,j}-y^S_{s,t,j}.
```

$`(n^N,n^A)`$ has a bivariate Student-*t* scale-mixture representation with an estimated
per-stage 2×2 correlation (LKJ(2)), capturing that revised NSA inputs move the concurrently
estimated factor. For same-release monthly changes $`d^N,d^S`$ and implied adjustment
$`a=d^N-d^S`$, the published data give the auditable identity

```math
r^S_{s,t,j\to j+1}=r^N_{s,t,j\to j+1}-\left(a_{s,t,j+1}-a_{s,t,j}\right),
\qquad
\operatorname{Var}(r^S)=\operatorname{Var}(r^N)+\operatorname{Var}(\Delta_j a)
-2\operatorname{Cov}(r^N,\Delta_j a),
```

estimated conditionally on sector, stage, and regime — the sample-versus-seasonal
decomposition the prompt asks for, as an accounting decomposition of published revision
variance, not a claim that factor revision is exogenous. Two observation channels for the
factor path (chosen — user decision Q3: design for both, pairs first): (a) the published pair
$`A_{s,t,j}`$ everywhere, under which the algorithm-versus-input split of $`n^A`$ is latent and
prior-identified; (b) wherever vintage-specific X-13 factors and specifications can be
reproduced from BLS's archived seasonal-adjustment files, the reproduced adjustment is observed
with rounding error only and strongly informs $`n^A`$. Channel (b) is a later stage. Which
vintages have archived specification, prior-adjustment, and outlier files, and whether the
unrounded NSA inputs are recoverable, is **(open — resolved by verification, not argument)**:
an archive inventory, not a design argument, settles it.

**Req 10 — Benchmark observations, wedge-back, and the post-benchmark operator.** For
benchmark year $`y`$ (chosen — chatgpt §4.6; gemini §3's $`\sigma_{BM}\to0`$ rejected as
contradicted by documented QCEW revisions, and its smoother-only treatment rejected because it
leaves the published benchmarked months unmodeled; claude §4.3's Brownian-bridge wedge is
accepted as a **derived output**, not a measurement — see below):

```math
b^{pre}_{s,y}=x_{s,\mathrm{Mar}(y)}+\xi^{pre}_{s,y},
\qquad
b^{fin}_{s,y}=x_{s,\mathrm{Mar}(y)}+\xi^{fin}_{s,y},
\qquad
\sigma^{fin}_{s,y}<\sigma^{T}_{s,\mathrm{Mar}(y)}\ \text{in prior probability, not by fiat.}
```

The two QCEW errors are positively correlated (the final count revises the preliminary
source). BLS's 2017+ public aggregate QCEW revision sequence informs the maturation component
and the preliminary/final correlation only; total QCEW error is prior- and
sensitivity-identified because no external truth series exists. For the ordinary backward window
$`I_y=(\mathrm{Apr}(y-1),\ldots,\mathrm{Mar}(y))`$, $`W=(1/12,\ldots,11/12,1)'`$, and
$`e_{\mathrm{Mar}}`$ selecting March, the published NSA vector is the known affine operator

```math
\mathbf y^B_{s,I_y}
=\left(I-W e_{\mathrm{Mar}}'\right)\mathbf y^{T}_{s,I_y}
+W\,b^{fin}_{s,y}+R_{s,y}\kappa_{s,y}+\epsilon^{round}_{s,y},
```

with $`R\kappa`$ nonzero only for documented scope changes or reconstructions. This makes the
wedge's rank-one covariance explicit: the benchmarked vector is never entered as twelve
independent observations. After March, the multiplicative sample-link calculation is preserved
as a deterministic recursion in the previously published matched-sample link relatives
$`R^{old}_{s,m}`$ and the updated birth–death values,

```math
E^B_{s,\mathrm{Mar}(y)}=b^{fin}_{s,y},\qquad
E^B_{s,m}=E^B_{s,m-1}R^{old}_{s,m}+BD^{new}_{s,m},\quad m>\mathrm{Mar}(y),
```

with explicit indicators for November/December and new-sample introduction. If only
job-change contributions are stored, a linear cumulative operator is valid, but the two
representations are never mixed. Whether historical link relatives are recoverable from the
vintage files or must be inferred from adjacent NSA vintages and published birth–death tables
(propagating that reconstruction error rather than fixing a guessed series) is **(open —
resolved by verification, not argument)**. The `B→M` transition applies the next benchmark's
wedge to months that were post-March in `B`, the observed revised birth–death inputs, and — on
the SA side — the successive annual adjustment maps, through a known operator
$`\mathcal M_y`$ plus residual mature news $`\mathbf n^M`$ and transitory error $`\mathbf e^M`$
on the `B→M` scale equation; the modeled stage-4 outcome is the observed $`\mathbf Y^M-\mathbf Y^B`$.
Annual SA reprocessing is applied after these operators through Req 9. Separately, the smoothed
posterior of the latent path $`x_{s,t}`$ between consecutive March anchors **is** the
probabilistic wedge (claude §4.3; gemini §3); the model reports it against BLS's linear
$`W`$ so the linear-accumulation assumption is tested (Robertson 2021), and this comparison is a
required output, not an alternative measurement equation.

**Req 11 — Net birth–death.** The ordinary monthly forecast is held fixed across the three
closings, so it has no `F→S` or `S→T` role; its forecast error is embedded in the `T→B`
discrepancy and its revised values enter the post-March operator (chatgpt §2.4, §4.6;
claude §4.3 folded in — the sector-specific `T→B` mean shift claude proposes is the stage mean
$`m^n_{s,B}`$, calibrated to the benchmark-article forecast-vs-realized tables). The relative
forecast is a `T→B` scale covariate (Req 8); NSA birth–death amounts are never added to SA
changes. Method changes — quarterly updating (2011), the 2025 April–October current-sample
adjustment, the January 2026 current-information ARIMA modification — are dated interventions
on the `T→B` and `B→M` scales under shrinkage (Req 14).

**Req 12 — Hierarchy across supersectors.** For driver $`p`$, transition $`k`$, sector $`s`$:
$`\beta_{s,k,p}=\bar\beta_{k,p}+\psi_{k,p}\tilde\beta_{s,k,p}`$, $`\tilde\beta\sim N(0,1)`$ —
non-centered by default (chosen — chatgpt §4.7; claude §4.4; gemini §4 agree). Centered
parameterization is admissible only for blocks with strong within-sector monthly information
and only after comparing sampler geometry, never by familiarity. Sector deviations are
constrained to sum to zero where $`\bar\beta`$ is interpreted as the employment-weighted
national effect. Pooling scale $`\psi_{k,p}`$ is driver- and stage-specific, so birth–death
and benchmark coefficients may vary widely across construction, professional/business
services, and education/health while collection-day effects pool tightly; the posterior of
$`\psi`$, not separate unregularized sector regressions, determines heterogeneity.

**Req 13 — Operational capacity, shrinkage, and what is identified.** No unrestricted slopes
for budget, FTE, headcount, sample size, and mode mix together (chosen — chatgpt §4.8; the
separate-slow-covariate horseshoe as *primary* model of claude §4.7 and gemini §5 rejected —
separate slopes are not recoverable from ~40 annual points, and reporting them as primary
invites misreading). The primary model carries one annual latent capacity factor

```math
v_{r,y}=a_r+\lambda_r K_y+\epsilon_{r,y},
\qquad
K_y=\phi_K K_{y-1}+\eta^K_y,
```

over log real enacted budget authority, appropriation-funded FTE, and — when
definition-consistent — CES workyears, budget loading fixed positive, missing indicators
integrated out, each definition break with its own intercept. Usable linked coverage is not
loaded into $`K_y`$ (its sector-month residual enters Req 8 directly). One shared $`K_y`$
coefficient across monthly closing stages with tightly shrunk stage deviations, learned from
annual changes. Except for the two well-identified within-year covariates — collection days
and the first-close residual — which take $`N(0,0.3^2)`$ priors on standardized scale
(claude §4.7, accepted for this block), the global-mean scale coefficients $`\bar\beta_{k,p}`$
— the operational, sample, birth–death, and intervention block — carry the regularized
horseshoe (Piironen–Vehtari; the regularized form is mandatory for sampler geometry):

```math
\bar\beta_{k,p}=z_{k,p}\tau^{HS}_k\tilde\lambda_{k,p},\quad
z_{k,p}\sim N(0,1),\quad
\lambda_{k,p}\sim C^+(0,1),\quad
\tilde\lambda_{k,p}^2=\frac{c_k^2\lambda_{k,p}^2}{c_k^2+(\tau^{HS}_k)^2\lambda_{k,p}^2},
```

$`\tau^{HS}_k\sim\text{Half-Student-}t_3(0,0.10)`$, $`c_k\sim\text{HalfNormal}(0.5)`$, with
results reported at global scales 0.05 and 0.20. A sensitivity model replaces $`K_y`$ with
separately regularized covariates; if their signs and magnitudes are prior-sensitive they are
reported as unidentified. The variation supporting each primary parameter is stated in the
report: collection-day effect (within-year calendar shifts, conditional on month/holiday and
macro state); residual collection effect (associational); seasonal-factor effect (paired
NSA/SA vintages plus specification interventions); sample adequacy (sector-year coverage/RSE,
partly confounded with industry volatility); birth–death (sector-year forecast exposure and
method changes, annual only); capacity (slow annual deviations and a handful of operational
shocks, weakly identified); 2003 (one composite intercept/scale shift). Shutdowns are case
studies or timing interventions, never pooled with annual funding. The model cannot say how
many jobs of revision one additional BLS employee prevents, and the report says so.

**Req 14 — Structural change.** Documented interventions only, no unconstrained break search
(chosen — chatgpt §4.9; claude §4.6 and gemini §4 accepted in spirit, superseded by the
narrower drift set): May 2003 as one composite regime shift, never labeled NAICS,
probability-sample, or concurrent-SA separately; March 2020–December 2021 as a pandemic
level/tail regime with BLS intervention-treatment indicators, 2022 a separate normalization
regime; August 2014 quarterly sample implementation, 2015 X-13, 2017 TRAMO, NAICS releases,
the 2025/26 birth–death changes, and shutdown-affected releases as candidate scale
interventions under strong shrinkage. Residual drift is permitted only in the collection-day
effect and the common scale intercept:

```math
\beta_{p,t}=\beta_{p,t-1}+\sqrt{q_p}\,u_{p,t},\qquad
\sqrt{q_p}\mid\xi_p^2\sim N(0,\xi_p^2),\qquad
\xi_p^2\sim\operatorname{Gamma}(0.10,40)
```

(shape–rate; the non-centered normal-gamma of Bitto & Frühwirth-Schnatter 2019, with
sensitivity at shapes 0.05 and 0.50), so unsupported time variation collapses to a constant.
Time variation in every sector-driver coefficient is not identified and is not attempted.
Unknown discrete change points are sensitivity-only, because their posterior otherwise
rediscovers 2020 and absorbs heavy tails.

**Req 15 — Priors in thousands of jobs, and the prior predictive plan.** Sector scale centers
are translated from aggregate centers by $`w_s^{\theta}`$ with $`w_s`$ the sector's 2015–19
employment share and $`\theta\sim N(0.7,0.15^2)`$ truncated to $`[0.3,1]`$ (chosen — chatgpt
§4.10 as the one prior table; claude §4.8's shorter set rejected for consistency; gemini §6's
log-thousands rejected with Req 6). Integer-thousand publication rounding is independent
zero-mean error with variance $`1/12`$ in the marginalized model; exact interval censoring is a
non-marginal sensitivity on a reduced panel only.

| Parameter | Prior | Rationale |
|---|---|---|
| baseline total-nonfarm news scale, `F→S` | $`\exp(\alpha^n_{0,S})\sim\operatorname{LogNormal}(\log 40,0.40^2)`$ | post-2003 SA/NSA MARs are 33–46; a *t* scale is not a MAR, so broad |
| baseline news scale, `S→T` | $`\exp(\alpha^n_{0,T})\sim\operatorname{LogNormal}(\log 35,0.45^2)`$ | centers near the 34 SA / 18 NSA MARs, allowing factor covariance |
| annual `T→B` March-level news scale | $`\exp(\alpha^n_{0,B})\sim\operatorname{LogNormal}(\log 300,0.60^2)`$ | 0.2 percent of ~150 million is 300; admits 2009/2024/2025 without making them routine |
| `B→M` residual news scale | $`\exp(\alpha^n_{0,M})\sim\operatorname{LogNormal}(\log 75,0.60^2)`$ | QCEW maturation, later benchmark, birth–death, and SA changes after the known operator |
| sector baseline scales | $`\log\sigma_{s,k}\sim N(\log(\sigma_{0,k}w_s^{\theta}),0.6^2)`$ | partial pooling with wide sector allowance |
| noise scale | $`\exp(\alpha^e_{s,j})\sim\operatorname{LogNormal}(\log(0.5\,\sigma_{s,j}),0.7^2)`$ | starts below news; ordering not forced |
| monthly factor-news scale, total | $`\sigma^A_{0,S},\sigma^A_{0,T}\sim\operatorname{LogNormal}(\log 15,0.70^2)`$ | between Loya's 9.6 seam MAR and Wright's contaminated 81 SD |
| annual factor/specification news, total | $`\sigma^A_{0,B},\sigma^A_{0,M}\sim\operatorname{LogNormal}(\log 75,0.70^2)`$ | large annual/COVID revisions without importing them into closings |
| NSA-news / factor-news correlation | per-stage Cholesky $`L^{NA}_k\sim\operatorname{LKJ}(2)`$ | weak shrinkage to zero; sign free |
| factor transitory noise | $`\sigma^{eA}_{0,j}\sim\operatorname{HalfNormal}(10)`$ | small relative to factor news, not zero |
| mean monthly news | $`m^n_{0,k}\sim N(0,15^2)`$, sectors scaled by $`w_s^{\theta}`$ | official mean revisions are far below MARs; keeps bias out of scale |
| mean annual benchmark news | $`m^n_{0,B}\sim N(0,100^2)`$ | allows persistent benchmark bias without expecting it |
| Student-*t* degrees of freedom | $`\nu_k=2+\operatorname{Exponential}(0.1)`$ | finite variance, mass on heavy tails (chosen over the shifted Gamma(2, 0.1) of claude/gemini — same support, more tail mass) |
| final QCEW/benchmark error, total | $`\sigma^{fin}_0\sim\operatorname{HalfNormal}(100)`$ | more precise than monthly CES in prior probability, demonstrably revised |
| additional preliminary-benchmark error | $`\sigma^{pre\text{-}extra}_0\sim\operatorname{HalfNormal}(150)`$ | covers preliminary-to-final gaps such as 220 in 2024 |
| preliminary/final QCEW correlation | $`\operatorname{atanh}(\rho_Q)\sim N(1,0.7^2)`$, truncated to $`0<\rho_Q<0.99`$ | same underlying records; imposes the positive prior an LKJ would not |
| SV persistence | $`(\phi_k+1)/2\sim\operatorname{Beta}(20,1.5)`$ | persistent, stationary; no hard-coded random walk |
| SV innovation | $`\omega_k\sim\operatorname{HalfNormal}(0.15)`$ | one SD ≈ 16 percent scale move before persistence |
| sector SV loading | $`\rho_{s,k}\sim\operatorname{LogNormal}(0,0.25^2)`$, weighted mean 1 | shared volatility with modest sector departures |
| common growth factor | $`(\phi_f+1)/2\sim\operatorname{Beta}(5,3)`$; $`\sigma_f\sim\operatorname{HalfNormal}(150)`$ | modest persistence; aggregate-equivalent monthly innovations |
| common-factor loadings | $`\lambda_s\sim N(1,0.5^2)`$, weighted mean 1, weighted sign positive | fixes scale and sign, heterogeneous exposure |
| cross-stage transitory noise | $`\rho_e\sim N(0,0.20^2)`$ truncated to $`(-0.8,0.8)`$; no residual correlation in the primary model | preserves news/noise identification; $`L_e\sim\operatorname{LKJ}(10)`$ in sensitivity only |
| capacity dynamics | $`(\phi_K+1)/2\sim\operatorname{Beta}(10,2)`$; $`\sigma_K\sim\operatorname{HalfNormal}(0.15)`$ | persistent, stationary annual capacity |
| capacity loadings / errors | budget loading fixed at +1; other $`\lambda_r\sim N(1,0.5^2)`$; $`\sigma_{v,r}\sim\operatorname{HalfNormal}(0.25)`$ | orients and scales $`K_y`$; resource measures are noisy and nonidentical |
| local-level innovation, total-equivalent | $`\sigma_{\eta^\ell,0}\sim\operatorname{HalfNormal}(150)`$ | latent growth can move materially without being revision error |
| slope innovation, total-equivalent | $`\sigma_{\eta^g,0}\sim\operatorname{HalfNormal}(15)`$ | gradual trend change outside documented breaks |
| seasonal-state innovation, total-equivalent | $`\sigma_{\eta^\gamma,0}\sim\operatorname{HalfNormal}(50)`$ | evolving NSA seasonality; production factor vintages inform more strongly |
| hierarchical sector dispersion | $`\psi_{k,p}\sim\operatorname{HalfNormal}(0.15)`$ | a one-SD sector departure changes scale by under ~35 percent |
| well-identified within-year slopes | $`\bar\beta\sim N(0,0.3^2)`$ on standardized covariates (collection days, first-close residual) | not horseshoe — these are identified (claude §4.7 accepted for this block) |

Prior predictive checks simulate complete 1979–2026 vintage panels before any outcome is
conditioned on, and require, as broad rather than fitted constraints (chatgpt §4.11; claude
§4.8; gemini §6 — union): median monthly first-to-third absolute revisions usually in 20–100
thousand with non-negligible but non-routine mass above 250; annual March revisions commonly
below 0.5 percent, occasionally near 1 percent, and multi-million only under a pandemic-sized
latent shock; sector revisions aggregating to plausible national revisions under the common
factor; a two-SD covariate change not implying a ten-fold scale change unless it escapes the
horseshoe slab; pandemic tails reproducible without permanently raising the post-2022 baseline;
wedge simulations reproducing the exact monotone April–March profile absent explicit
reconstruction residuals; QCEW error smaller than third-release CES error in most, not all,
draws. Failures adjust priors before any coefficient posterior is inspected. Because the
centers use official historical MARs, this is weak empirical calibration and prior sensitivity
is rerun with doubled log-scale SDs.

**Req 16 — Identification assumptions and failure modes.** Stated in the report as numbered
assumptions with what breaks each (prompt Part C; chatgpt §4.12): (1) nested information sets —
a reconstruction that discards earlier information is an intervention, not news; (2) concept
consistency — vintages bridged to a common NAICS/scope concept, else classification change is
inseparable from latent employment; (3) orthogonality — conditional news orthogonal to the
earlier release's information set, transitory noise orthogonal to latent truth and news;
(4) the benchmark anchor is precise but not exact, and shared payroll-processor or
classification error between CES and QCEW breaks independence — hence the correlation
sensitivity; (5) revision policy is ignorable conditional on flags — COVID outlier decisions and
reconstructions get explicit indicators; (6) scale/process separation — large true movements
live in the latent state; (7) missing vintages (2003 gap, 2025 lapse) are ignorable
conditional on lapse/reconstruction flags, never ordinary random missingness. With two releases
alone news and noise are not separately identified; the third release, benchmark signals, fixed
mature horizon, sector hierarchy, and external QCEW precision overidentify, conditional on
(1)–(7). The posterior decomposition is presented under at least (a) independent QCEW/CES
error, (b) a positive shared-error prior, and (c) the mature vintage as another noisy measure
rather than truth.

### Inference and validation

**Req 17 — Inference.** Student-*t* errors are Gaussian scale mixtures,
$`\lambda_i\sim\operatorname{Gamma}(\nu/2,\nu/2)`$, $`\epsilon_i\mid\lambda_i\sim N(0,\sigma_i^2/\lambda_i)`$
(the InvGamma variance form of gemini §7 is the same representation). Conditional on the
mixing variables, the volatility states, and the nonlinear scale parameters, the employment,
trend, seasonal, and common-factor states are linear-Gaussian and are **marginalized
analytically** by a differentiable Kalman filter, whose log marginal likelihood is the NumPyro
likelihood term; NUTS samples only the static parameters, mixing variables, and volatility
paths (chosen — chatgpt §4.14; claude §4.10; gemini §7; and the house state-space rule:
marginalize whenever conditionally linear-Gaussian). Order of work: (1) sparse JAX operators
for aggregation, same-release differencing, the wedge $`W`$, the post-March cumulative
operator, seasonal mapping, and missing-vintage masks, unit-tested against published benchmark
articles before any fit; (2) a linear-Gaussian pilot with irregular benchmark rows and missing
data to debug state dimension and initialize mass matrices; (3) the full model in 64-bit JAX
with non-centered innovations wherever a path is sampled; (4) if the mixing-variable dimension
is prohibitive, a small finite normal mixture summed by forward recursion; (5) BlackJAX only
after the marginalized NumPyro benchmark identifies the bottleneck — a blocked kernel (NUTS on
globals and hierarchy scales, specialized updates for volatility states, Gibbs for mixing
scales), never a custom kernel first. Whether Dynamax's linear-Gaussian SSM supports the
time-varying emission rows, missing cells, and irregular annual observations these operators
need, or a hand-written Kalman `scan` in NumPyro is required, is **(open — resolved by
verification, not argument)**; so is Python 3.14 support for each of NumPyro, JAX, Dynamax,
ArviZ, and Polars (project rule in CLAUDE.md). ArviZ InferenceData carries the log-likelihood
grouped by reference month and benchmark year; seeds, data hashes, library versions, and the
operator test results are stored with every run.

**Req 18 — Validation.** (chatgpt §4.13 as the superset; claude §4.9's era-specific checks and
CRPS folded in; gemini §8's raw pointwise LOO rejected because wedge cells are not
exchangeable.) Posterior predictive checks targeted at the estimand: stage × sector × regime
(pre-2003, 2003–2019, 2020–2022, 2023–present) MAR, median absolute revision, SD, 90th/95th/99th
absolute quantiles, skew, and counts above 100/250/500 thousand at total nonfarm; rolling
24/60/120-month dispersion with separate 2020–22 panels; correlations of `F→S` with `S→T`
revisions and of NSA-sample with seasonal-factor revision; cross-sector covariance and the
aggregate-to-sector-rss ratio; preliminary-to-final benchmark surprise and same-sign runs; the
exact April–March wedge shape and residuals after the operator; coverage of published
first/second/third values by posterior real-time intervals. Comparators, fitted in order:
(0) constant-variance Gaussian, (1) constant-variance Student-*t*, (2) Student-*t* plus SV,
(3) the full covariate-SV hierarchy; PSIS-LOO on independent innovation/reference-month
blocks with Pareto-$`k`$ inspected, supplemented by leave-future-out and
leave-one-benchmark-year-out log predictive density; a covariate "wins" only on held-out
scale/tail calibration. Pseudo-real-time: at each historical release date truncate every series
and covariate to what was then observable and predict the second release from the first, the
third from the first two, the preliminary and final March benchmark, and the benchmarked
monthly path, on expanding windows with prespecified evaluation blocks 2008–09, 2013, 2019,
2020–22, and 2023–present, scored by log predictive density, CRPS, interval coverage/width,
absolute-revision calibration, and exceedance Brier scores; revised collection, QCEW, or
macro controls are never fed to an earlier pseudo-vintage. MCMC: four dispersed chains,
rank-normalized $`\hat R<1.01`$, bulk and tail ESS above 400 for every reported functional, zero
post-warmup divergences, acceptable energy-BFMI, no systematic tree-depth saturation; rank
plots for scale, $`\nu`$, SV persistence, capacity, and hierarchy SDs; simulation-based
calibration on reduced and full synthetic panels with recovery checks specifically for
news/noise allocation and wedge rank; posterior/prior overlap and likelihood profiles for
capacity coefficients; sensitivity to centered/non-centered hierarchies, horseshoe global
scale, QCEW/CES error correlation, and removal of each crisis episode. Prior predictive,
posterior predictive, PIT/coverage calibration, prior sensitivity, and the fixed-shape report
follow the house Bayesian workflow.

**Req 19 — Outputs and claims.** The report delivers (chatgpt §4.15; claude §4.11; gemini
§8): posterior distributions of stage- and sector-specific scale multipliers, tail exceedance
probabilities, and news shares; the estimated seasonal-versus-sample decomposition (Req 9) by
sector, stage, and regime; the posterior latent path versus the linear wedge (Req 10); the
posterior probability that, for example, the post-2020 birth–death error scale exceeds its
2003–2019 level; a calibrated predictive distribution of future revision scale usable directly
as the measurement variance in a payroll nowcast; the capacity association with its prior
sensitivity, labeled as an association; and the descriptive party-composition table with no
causal label. The report states plainly what a frequentist event study or heteroskedastic
regression could not deliver — a latent path, nested information sets, news separated from
transitory noise, a fallible QCEW, one March discrepancy propagated through the correlated
wedge — and equally plainly that it cannot turn trending appropriations into causal effects.

### Written finding

**Req 20 — Verified literature review and data inventory.** A GitHub-renderable document at
`docs/ces-revisions-review.md`, following the Markdown conventions in CLAUDE.md, delivering the
brief's Parts A and B (chosen — user decision Q1; model-only and inventory-only scopes
rejected): the literature organized by the five candidate drivers with the revision-source
decomposition running through it; the gap table (one row per open question — question,
closest work, why it falls short, data needed, public feasibility); the data-availability
inventory per driver (series, vintage coverage, frequency, earliest date, access route,
hand-build constraints); and an annotated bibliography with full citations labeled
peer-reviewed / agency documentation / grey literature. Rules: every numeric claim, date, and
citation is verified against a primary source (BLS, CRS, GAO, DOL/OMB, the journal) or
explicitly marked *unverified*; each claim carries one of the evidence labels *documented /
supported / asserted / contested / not found* (chatgpt §1.1); disagreements among the three
drafts are resolved on the record — at minimum the collection-versus-response conflation, the
2018–19 lapse, whether 2025 program cuts touched CES, the FTE authorized-versus-actual and
FTE-versus-headcount concepts, and the 2024/2025 preliminary-versus-final benchmark figures.
The inventory's findings on which vintages, archives, and link relatives exist discharge the
(open) items of Reqs 9, 10, and 17 as far as public sources allow; where a fact turns on a
package's behavior it is discharged by running the package, not by reading about it.

## Verification — observable outcomes

- [ ] The same-release differencing operator applied to the Req 1 panel reproduces every row of
      BLS's 1979–present revision table (SA and NSA, all three pairwise MARs) to publication
      rounding, including the May 2003 gap handling.
- [ ] The wedge operator applied to the stored `T` vintage and the published final benchmark
      reproduces the published `B` vintage for every benchmark year 2003–2025 to rounding
      (e.g. the −861 thousand March 2025 gap yields the published −71.75 thousand per-month
      profile), with documented reconstructions the only nonzero $`R\kappa`$ terms; the
      post-March recursion reproduces the published April–October `B` months where link
      relatives are recoverable.
- [ ] Discharge of (open) in Req 10: a written determination of which benchmark years' link
      relatives are recoverable from vintage files and which are inferred, with the
      reconstruction error propagated in the fitted model.
- [ ] Discharge of (open) in Req 9: an inventory of BLS's archived seasonal-adjustment files by
      vintage (specification, prior-adjustment, outlier files; unrounded inputs present or
      absent), committed with the written finding.
- [ ] Discharge of (open) in Req 17: each of NumPyro, JAX, Dynamax, ArviZ, and Polars installs
      and imports under Python 3.14 via `uv add`, recorded in `pyproject.toml`/`uv.lock`; and a
      pilot fit demonstrates either Dynamax handling time-varying emission rows, missing
      cells, and irregular annual observations, or the adopted hand-written Kalman `scan`
      passing the same pilot.
- [ ] Prior predictive simulation satisfies every Req 15 constraint before the first
      conditional fit; the check is re-run with doubled log-scale SDs.
- [ ] Simulation-based calibration on synthetic panels recovers news/noise allocation, wedge
      rank, and scale coefficients with calibrated rank histograms.
- [ ] Final fits meet Req 18's MCMC thresholds; the diagnostics JSON and interpreted report
      artifacts exist for every reported model.
- [ ] Nested comparison (0)–(3) on blocked PSIS-LOO, leave-future-out, and
      leave-one-benchmark-year-out is reported with Pareto-$`k`$; every retained covariate
      improves held-out scale/tail calibration, not merely in-sample likelihood.
- [ ] Pseudo-real-time scores on the five prespecified blocks are reported for all four
      prediction targets, with a leakage audit showing no revised covariate reached an
      earlier pseudo-vintage.
- [ ] The seasonal-versus-sample decomposition (Req 9 identity) is reported by sector, stage,
      and regime with posterior intervals, alongside the posterior latent path versus the
      linear wedge for every benchmark year.
- [ ] The capacity coefficient is reported with posterior/prior overlap, at horseshoe global
      scales 0.05/0.10/0.20, and under the separate-covariate sensitivity, with any
      prior-sensitive slope labeled unidentified; party composition appears only in a
      descriptive table.
- [ ] `docs/ces-revisions-review.md` exists; every numeric claim carries a primary-source
      citation or an *unverified* flag; the five named draft disagreements are resolved on the
      record; `uv run ruff format --check` passes on it and on this spec.
- [ ] The FY2009+ institutional panel and the collection-window panel are committed with
      provenance and definition-break flags and contain no annual-to-monthly interpolation.

## Out of scope

- ADP or other private payroll-processor signals as auxiliary observations (chatgpt §2.6;
  claude §2) — proprietary, and not a revision-scale driver.
- State and metropolitan (SAE) revisions, and quarterly-benchmarking redesign proposals for
  BLS production (Dey–Loewenstein; Robertson 2021) beyond using their evidence to test the
  wedge.
- Causal claims about appropriations, staffing, or party control; any instrument-based design.
- Restricted-access CES microdata or paradata — which late reporters generate the tails is
  answerable only with BLS cooperation (chatgpt §3.1 Q5).
- Hours and earnings revisions.
- X-13 reproduction as a v1 requirement (designed for in Req 9; a later stage gated on the
  archive inventory), the FY1979 institutional panel (Req 3; a later stage), and the 1964
  RTDSM extension (Req 4; rejected).
- Reconstructing the full historical QCEW vintage cube; only the 2017+ public revision file
  informs QCEW maturation.

## Rollout note

Two (open) items block the first model fit and open this rollout: Python 3.14 support for the
inference stack and the Dynamax-versus-hand-written-Kalman question (Req 17). The archive
inventory (Req 9) and link-relative recoverability (Req 10) gate later stages, not the first.
The written finding (Req 20) is an investigation stage whose exit artifact is a document, and
much of the data inventory it produces is information later model stages consume — derive-roadmap
should sequence by that information order rather than by the drafts' value ranking. The three
research drafts and the prompt retire alongside this spec when its last stage completes; they
are its inputs. Stage stamps and the roadmap reference are added to this note by derive-roadmap.

Stage 1: COMPLETE (2026-09-12) — implemented by plan 1 (specs/plans/completed/1-ces-revisions.md). Next: resume the roadmap.

Stage 2: COMPLETE (2026-09-13) — implemented by plan 2 (specs/plans/completed/2-ces-revisions.md). Next: resume the roadmap.
