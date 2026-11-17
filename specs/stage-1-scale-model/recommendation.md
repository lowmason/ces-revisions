# Recommendation: stage-1-scale-model

The problem brief is [`specs/stage-1-model-brief.md`](../stage-1-model-brief.md). The current frequentist answer is in [`specs/ces-revision-drivers-stage-1-findings.md`](../ces-revision-drivers-stage-1-findings.md). **Sample: reference months from January 2003 on** (owner, 2026-11-17, because of the probability-sample redesign), less the held-out March 2020–December 2021 and the dropped 2025-09 and 2025-10.

## Summary

Fit a Bayesian **location-scale Student-t regression on the signed revision**. Log σ gets hierarchical reference-year effects, unpooled calendar-month effects, and the window, 5-week, lapse and benchmark-cell terms. The mean gets calendar-month effects and the same window and 5-week terms, so a shift in the mean is not misread as a change in scale. Fit each series by itself first, with NSA F→S as the primary series. Then fit one joint model that pools the window effect. Use a stand-alone NumPyro model, not the Kalman engine. The estimate to report is `exp(7β)`, the ratio of revision scale at 19 business days to 12, with its posterior interval.

## 1. Problem framing

- **Task type:** regression, but distributional: the covariates act on the **scale** of a continuous signed outcome.
- **The question being asked:** uncertainty and inference about one association, the per-day effect of the collection window `D_t` on revision scale. The causal design is taken as given from the brief. `D_t` is treated as assigned by the calendar, conditional on reference year, calendar month and the 5-week survey interval. §4 shows how the model keeps that conditioning set.
- **Decision driver:** whether Stage 1's answer ("no clear effect within 12 to 19 days") holds on the scale that the project's design says drivers act on, and how tight the bound is. The answer decides whether Stage 9 (O3) and Track B keep `D_t` as a variance covariate.

## 2. Data characterization

- **Primary data:** 258 reference months for F→S and 256 for F→T, across 23 reference years (2003–2026, with 2021 entirely held out). There are four signed outcomes per month, {NSA, SA} × {F→S, F→T}, in thousands, rounded to whole thousands. The regressors are:
  - `D_t`: 12 to 26 business days.
  - `five_week`: 90 months.
  - `lapse_delayed`: 4 months (September and October 2013, November 2025, January 2026).
  - Reference year and calendar month.

  Signals measured on 2026-11-17 from `data/panel/`:

| Series | n | mean | sd | kurtosis raw | kurtosis, leave-one-out rescaled | lag-1 / lag-12 ACF of r | lag-1 / lag-12 ACF of \|r\| |
|---|---:|---:|---:|---:|---:|---:|---:|
| NSA F→S | 258 | 9.0 | 56.8 | 4.5 | 4.9 | −0.07 / 0.12 | 0.05 / 0.06 |
| SA F→S | 258 | 4.0 | 36.2 | 4.6 | 4.9 | 0.01 / 0.09 | 0.03 / 0.02 |
| NSA F→T | 256 | 8.8 | 62.2 | 4.0 | 4.4 | 0.01 / 0.14 | 0.01 / 0.12 |
| SA F→T | 256 | 3.0 | 54.1 | 4.9 | 3.4 | **0.35** / 0.20 | 0.21 / 0.27 |

  The rescaled kurtosis divides each revision by the RMS of the 24 neighbouring months, without the month itself.

  - **Little drift.** Mean |r| for NSA F→S by 2003–09 / 2010–19 / 2022–26 is 44.1 / 39.4 / 48.5. SA F→S is 30.4 / 24.5 / 30.7. Year effects need to absorb a mild U shape, not a trend.
  - **The tails are moderate and real.** Kurtosis stays at 4–5 after removing local scale, so a Student-t with ν near 5–10 is justified, not just insurance.
  - **Strong calendar-month pattern in scale.** NSA F→S mean |r| by month runs from 25 to 26 in March–April up to 70 in August, 60 in September and 61 in December. All four series have the same shape. So the calendar-month effects on log σ carry much of the explained variance.
  - **Small positive mean.** NSA revisions average +9 thousand. The mean has a weak 12-month echo (lag-12 ACF 0.09–0.20).
  - **SA F→T is serially dependent.** Its lag-1 ACF is 0.35. A likely cause is concurrent seasonal adjustment: one revision of the seasonal factors moves neighbouring months together. Conditional independence fails for this series, which is one more reason NSA is the primary series.
  - **Cross-series dependence.** The |r| correlations are 0.41–0.90 (NSA F→S with NSA F→T is 0.90). F→T = F→S + S→T, so the two transitions share the F→S part.
  - **Identifying variation is thin.** After the year and month effects, `D_t` has a residual SD of 1.84 days, and 1.35 days after the 5-week flag. For a Gaussian likelihood, the data-only SE of β on log σ is about `1/(√(2n)·1.35) ≈ 0.033` per day (n = 258), and somewhat larger under a Student-t. This is the yardstick for the priors and the sensitivity check in §5.
  - **`D_t` varies by calendar month.** In 2003+, its mean ranges from 15.2 (May, June) to 16.8 (September), and the share of 5-week months ranges from 0.00 (March) to 0.48 (August). Pooled month effects would let that pattern into β (§4).
  - **Rounding.** About 1–2% of values are exactly 0. Rounding to 1 thousand adds variance of 1/12 against σ near 25–70 thousand, which is negligible.
- **External / auxiliary information:**
  - The benchmark moved to the January release from 2004. November's F→T and December's F→S then include part of the benchmark revision, so they get an indicator. The December spike in mean |r| fits this.
  - Stage 9 (O3) will bring first-closing collection rates, the mediator between `D_t` and revision scale. Build the model so the rate can enter as one more scale covariate.
  - Stage 6's pilot gives the harness conventions and the prior centre for revision noise in total nonfarm.

## 3. Candidate methods

| Method | Why a candidate | Key assumption | Trade-off |
|--------|-----------------|----------------|-----------|
| **A. Location-scale Student-t on signed r. Log σ: hierarchical year + unpooled month + covariates** (distributional regression) | It puts β on log σ directly, which is the estimand. Hierarchical year effects mirror the OLS year fixed effects but pool 23 years with about 11 months each. A Student-t fits the measured kurtosis of 4–5 | Given μ and σ, revisions are independent across months. This holds for three series, not SA F→T | About 50 parameters per series, cheap on NUTS. Year effects need a non-centered parameterization |
| **A′. As A, with a yearly random walk on log σ** | A smoother drift that borrows from adjacent years | Scale changes slowly between years | With little drift and a two-year gap at 2020–21, it is unlikely to beat A. Compare by LOO |
| **B. Gamma GLM on \|r\| with log link, year and month effects** | The closest step from the tried OLS. Multiplicative effects on mean \|r\|, which is proportional to σ when the error is symmetric about zero | Signed mean is zero | Drops the sign, so it cannot separate a mean shift from a scale change. That is a real issue here, because NSA's mean is +9 |

## 4. Recommendation

- **Recommended:** **A**, fit first to each series by itself, with NSA F→S as the primary series. **Because:**
  - The estimand is a scale effect, and A models log σ directly.
  - Kurtosis of 4–5 that survives local rescaling calls for a Student-t, and A estimates ν.
  - NSA's positive mean (+9) and its 12-month echo need a modeled location, which B cannot have.
- **How A keeps the conditioning set:**
  1. **Years.** Hierarchical year effects pool years, so between-year variation in `D_t` could leak into β. **Center `D_t` within reference year** (`D̃_t = D_t − mean_year(D)`) and add the year mean `D̄_y` as its own scale term (Mundlak 1978). β then rests only on within-year variation, as in the OLS with year fixed effects.
  2. **Months.** `D_t` and the 5-week share vary by calendar month. So the **month effects on log σ and on μ are unpooled**: sum-to-zero with a fixed wide prior, not a learned pooling scale. Each month has about 21 observations, enough to estimate them without pooling.
  3. **Mean versus scale.** Put `D̃_t` and `F_t` in the **location** μ as well. Then a window effect on the mean does not widen `r − μ` and get read as a scale effect.
- **Explicit assumptions:**
  - Given μ and σ, revisions are conditionally independent across months, for NSA F→S, SA F→S and NSA F→T.
  - The effect of the window is log-linear in days over the observed range.
  - The benchmark-contaminated cells (November F→T and December F→S, from 2004) differ only in scale, through an indicator.
  - The 4 lapse months are modeled by an indicator, not dropped.
- **What would change this:**
  - If LOO prefers A′ by more than about 2 SE of the ELPD difference, use the random walk.
  - If posterior predictive checks show lag-1 dependence in |r| for a primary series, add an AR(1) term on the signed residual.
  - For SA F→T, include an AR(1) term from the start, or report the series as secondary with that caveat.
  - If the joint model puts the pooling scale τ_β clearly away from 0, report β by series.

## 5. Regularization & model selection

- **Complexity knobs:**
  - the year-effect scale `τ_y` on log σ (or `σ_h` for A′)
  - ν of the Student-t
  - in the joint model, the pooling scale `τ_β` of the series-specific window effects

  The month effects are deliberately unpooled (§4).
- **Selection criterion:** **PSIS-LOO ELPD**, with the log likelihood grouped by reference month (pointwise) and by reference year (a stricter check). The question is inferential, not forecasting, so leave-future-out CV is not needed. Pareto-k above 0.7 on the lapse months or January 2024 (−220 thousand, the largest NSA F→S revision) is expected. Report it, and do not exclude data on that basis.
- **Prior sensitivity, with its yardstick:** the data-only SE of β is about 0.033 per day (§2). Refit with β's prior SD at 0.05, 0.1 (the default) and 0.2. At 0.1 the prior is about 3 times the data SE, so it shrinks the posterior SD by only about 5%. At 0.05 it shrinks it by about 17% **by construction**, which is not evidence that the data fail to determine β. The criterion is that the posterior mean of β moves by less than half a data SE (about 0.017) between the 0.1 and 0.2 runs. If it moves more, say in the findings that the prior shapes the answer.
- (Pre-fit specification only. The post-fit LOO comparison belongs to `bayesian-workflow`.)

## 6. Specification for handoff (C4)

Notation: `t` is the reference month, `y(t)` its reference year, `m(t)` its calendar month, `k` the series. `D̃_t` is `D_t` centered within the reference year. `D̄_y` is the year's mean `D_t` less the sample mean. `F_t` is the 5-week flag, `L_t` the lapse flag, and `B_{t,k}` the benchmark-cell flag (November F→T and December F→S, from 2004).

- **Likelihood family:** `r_{t,k} ~ StudentT(ν_k, μ_{t,k}, σ_{t,k})`, with the signed revision in thousands.
- **Location:**

  ```math
  \mu_{t,k} = m_k + p_{k,m(t)} + \gamma_k \tilde D_t + \phi_k F_t
  ```

- **Scale:**

  ```math
  \log \sigma_{t,k} = a_k + u_{k,y(t)} + c_{k,m(t)} + \beta_k \tilde D_t + \kappa_k \bar D_{y(t)} + \delta_k F_t + \lambda_k L_t + \omega_k B_{t,k}
  ```

- **Candidate priors** (log-σ effects are on the log scale, so 0.1 is about a 10% change in scale):

| Parameter | Prior | Note |
|---|---|---|
| `a_k` | Normal(log 40, 0.5) | Observed σ is 36–62 thousand |
| `u_{k,y}` | non-centered `τ_y · z_y`, `z_y ~ Normal(0,1)`, `τ_y ~ HalfNormal(0.3)` | Era means vary by about ±0.15 in log |
| `c_{k,m}` | Normal(0, 0.5), sum to zero over months | Unpooled. Observed month spread is about ±0.5 in log |
| `β_k` | **Normal(0, 0.1)** per day | Data SE is about 0.033. Run sensitivity at 0.05 and 0.2 (§5) |
| `κ_k` | Normal(0, 0.1) | Between-year window term (Mundlak) |
| `δ_k` | Normal(0, 0.25) | 5-week, on scale |
| `λ_k` | Normal(0, 0.5) | 4 observations, so the prior regularizes |
| `ω_k` | Normal(0.3, 0.5) | Benchmark cells, centred on a larger scale |
| `ν_k` | Gamma(2, 0.1) | Juárez–Steel. The measured kurtosis suggests ν near 5–10 |
| `m_k` | Normal(0, 10) thousand | |
| `p_{k,m}` | Normal(0, 10) thousand, sum to zero | Unpooled, for the same reason as `c` |
| `γ_k` | Normal(0, 5) thousand per day | Window effect on the mean |
| `φ_k` | Normal(0, 10) thousand | 5-week effect on the mean |

- **Structure:**
  1. **Fit 1, the primary fit:** each of the four series fit by itself. NSA F→S is the headline. Report `exp(7β_k)` with 90% and 95% intervals, and `γ_k` beside it. For SA F→T, add an AR(1) term on the signed residual, `ε_t = ρ ε_{t−1} + η_t`, or label the series secondary.
  2. **Fit 2, the joint fit:** reparameterize F→T as the increment S→T, so F→S and S→T are separate draws. Pool the window effect, `β_k = β + τ_β η_k` with `τ_β ~ HalfNormal(0.05)`, non-centered. Share the year effects with series deviations, `u_{k,y} = u_y + s_k v_{k,y}`. Fit NSA and SA separately, and accept the NSA–SA correlation as left out of the model.
- **Regularization & selection plan:**
  - Partial pooling over years, and in Fit 2 over β across series, is the regularizer. The month effects stay unpooled on purpose.
  - Compare A with A′ by PSIS-LOO, grouped by month and by year. Fit B only as a cross-check against the OLS, not for LOO, because its outcome differs.
  - **Before the fit**, run prior predictive checks: simulated |r| should cover about 2 to 300 thousand, and simulated month-to-month scale ratios should stay within about 0.3 to 3.
  - **After the fit**, run posterior predictive checks:
    - kurtosis of `(r − μ̂)/σ̂`
    - mean |r| by calendar month (August, September, December against March–April)
    - ACF of r and |r| at lags 1 and 12
    - mean |r| by 5-week status and by `D_t` cell
  - Run SBC on Fit 1 for NSA F→S at a small replicate count before trusting β's interval. `src/ces_revisions/pilot/sbc.py` is the precedent.
- **Execution notes for this repo:**
  - Use `ces_revisions.runs`: `run_nuts` with `dispersed_init_params`, and `diagnostics_record` for Req 18.
  - `attach_log_likelihood` blocks by reference month and by an integer `benchmark_year` array, and expects pilot-style steps with an `observed_any` mask. Passing the reference year as that array would work mechanically, but the output group would be named `_benchmark_year`. Write a small reference-year grouping for this model instead, or rename it on output.
  - Start at `target_accept_prob = 0.95`, the pilot's lesson. Run `numpyro.enable_x64()` first, and pass `chain_method(n)` from `ces_revisions.devices`.
  - Write runs to `data/runs/` and the small JSON evidence to `docs/runs/`.
  - There is no latent revision level to filter, so `kalman.py` does not apply.

## 7. References (verified)

The citations come from the skill's curated family files, whose §refs and notebook paths are recorded as Gate-A verified. The sections were not re-read against a local copy of the books on 2026-11-17.

- PML1 §11.6 — robust regression (Student-t and Laplace likelihoods); PML2 §15.2.
- PML1 §13.2.4.4 / PML2 §14.3.2.1 — quantile and distributional regression (mean and scale as functions of predictors).
- PML2 §15.5 — hierarchical models; PML2 §15.5.2 — non-centered parameterization.
- PML2 §29.12 — structural time series (the random-walk component in A′).
- pyprobml: `notebooks/book1/11/linregRobustDemoCombined.ipynb`, `notebooks/book2/15/linreg_hierarchical_non_centered_numpyro.ipynb`, `notebooks/book2/29/sts.ipynb`.
- Mundlak, Y. (1978), "On the pooling of time series and cross section data", *Econometrica* 46(1), 69–85. Cited from the literature, not from the skill's index.

## 8. Next steps / handoff

- **Runs next:** `bayesian-workflow` in a fresh session: `/clear`, then invoke it against this file. The fit runs on the Mac in minutes. No GPU is needed.
- **Bracketing skills:** `explore-data` is upstream and was done in this session (§2). `validate-data` is downstream, before any findings change. `track-model-experiments` keeps the ledger of A, A′ and the prior-sensitivity runs.
- **Iterate:** `tune-hyperparameters` is not needed. Every knob is a prior or a structure compared by LOO.
- **Open questions:**
  1. **The second release's window.** The F→S revision also depends on how long collection ran before the second estimate. Adding that window as a scale covariate changes the conditioning set, so it is the owner's decision.
  2. **Where the result lands.** Does the Bayesian estimate replace the OLS tables in the Stage 1 findings, or sit beside them? Should the OLS also move to 2003+? The roadmap's exit test names the 1979–2026 OLS specification, so this is the owner's decision.
  3. **SA F→T's lag-1 dependence.** Confirm that it comes from concurrent seasonal-factor revisions (`bls-data-context`) before choosing between the AR(1) term and a secondary label.
