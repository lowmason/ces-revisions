# Stage 6 — Aggregate Linear-Gaussian Pilot and Run Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via subagent-driven-development (the default) — or executing-plans when your human partner chose inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

> Roadmap: specs/ces-revisions-roadmap.md, Stage 6 — on plan completion, tick the stage and
> re-validate later stages against what shipped.

**Goal:** Fit the marginalized linear-Gaussian pilot to the real total-nonfarm vintages under Req 18's thresholds, and ship the run harness that every later fit reuses. The pilot takes same-release F, S, and T changes from 1979, one composite May 2003 shift, and the 23 final March anchors. Calibrate it by prior predictive and reduced-panel SBC. Choose the seasonal state, and the Lambda GPU type and count for each Stage 7–9 model, from measured cost.

**Architecture:** A new `ces_revisions.pilot` package builds the aggregate observation panel from the Stage 3, 4, and 5 artifacts. It defines three seasonal layouts and the Jacobs–van Norden state-space model on top of `kalman.py`, simulates synthetic panels from that model, runs SBC, and holds the two pre-registered decision rules (seasonal layout and per-stage GPU). A new, model-agnostic `ces_revisions.runs` package owns dispersed chain starts, NUTS, the grouped log likelihood, the Req 18 diagnostics, provenance, and the run directory. `scripts/pilot_run.py` drives the prior predictive, the fit, SBC, and both decisions. `scripts/gpu_probe_grid.py` times the engine at the Stage 7–9 dimensions on each host. One owner-run task rents the Lambda GPUs. The decision record is written from the committed evidence.

**Tech Stack:** Python 3.14; JAX 0.11.1 (float64); NumPyro 0.21.0 (NUTS); ArviZ 1.3.0 (xarray `DataTree`); xarray 2026.7.0; h5netcdf 1.8.1 and h5py 3.16.0 (new); SciPy 1.18.1 (already locked through ArviZ, now declared); Polars 1.44.2; NumPy 2.5.3; pytest 9.1.1; ruff 0.16.7.

**Spec:** [`specs/ces-revisions.md`](../ces-revisions.md) Req 17 (step 2), Req 4, Req 6, Req 7, Req 15, Req 18; roadmap [`specs/ces-revisions-roadmap.md`](../ces-revisions-roadmap.md) Stage 6. Executors read both.

## Global Constraints

- **Python:** keep `requires-python = ">=3.14"` and the uv `required-version = "~=0.13.1"`.
- **Dependencies:** h5netcdf and h5py are the only new runtime packages (D4). SciPy is declared because the pilot imports it directly; it is already in `uv.lock` through ArviZ, so the lock gains no package. Each declared package has an import test in `tests/test_stack.py`. No plotting library and no BlackJAX.
- **Marginalization (Req 17):** "the employment, trend, seasonal, and common-factor states are linear-Gaussian and are **marginalized analytically** by a differentiable Kalman filter, whose log marginal likelihood is the NumPyro likelihood term". Every latent state, the news states included, is integrated out by `kalman_factor`. NUTS samples only the 17 static scalars.
- **Precision:** every JAX array is float64. Tests get `numpyro.enable_x64()` from `tests/conftest.py`. Each script sets `XLA_FLAGS` and calls `numpyro.enable_x64()` before its first JAX operation.
- **MCMC (Req 18):** "four dispersed chains, rank-normalized $`\hat R<1.01`$, bulk and tail ESS above 400 for every reported functional, zero post-warmup divergences, acceptable energy-BFMI, no systematic tree-depth saturation". As in Stage 1, "acceptable" means energy-BFMI above 0.3 in every chain, and "no saturation" means under 1% of draws at the maximum tree depth of 10. The reported functionals are every sampled site.
- **Chains:** NumPyro's chain method always comes from `chain_method(num_chains)` in `src/ces_revisions/devices.py`. Never write `"parallel"`, and never write a GPU count into code.
- **Seeds:** descriptive, `sum(map(ord, "<name>"))`, never a bare literal.
- **Data:** the pilot reads only the committed Stage 3, 4, and 5 sources, through their offline builds. NSA only (Stage 9 owns SA). Sector `"00"` only. EMPLOY is never fitted.
- **Measurement (D1):** every closing cell is a same-release over-the-month change, and the anchors are levels on the latent path. Each cell has one source.
- **Crisis multiplier (D2):** `exp(kappa_crisis)` scales the non-news level shock and the S and T news shocks in March–June 2020. It does not scale the slope, seasonal, post-T news, or noise terms.
- **Run artifacts:** run directories go to the gitignored `data/runs/`. The pilot's small JSON evidence is committed under `docs/runs/stage-6/`. No InferenceData file is committed.
- **Tests:** flat under `tests/`, with bare-name helpers and no `tests/__init__.py`. MCMC tests carry `@pytest.mark.slow`. No test hits the network.
- **Ruff:** every commit step runs `uv run ruff format` and `uv run ruff check` on the files it touches, by explicit path. Stage files by explicit path, never `git add -A`.
- **Markdown:** CLAUDE.md's rules for `specs/` apply to every new `.md` file: display math in ```` ```math ```` fences, inline math in `` $`…`$ ``, literal dollars as `\$`, real headings, hard breaks as a trailing `\`, and pseudo-math with `_` in code spans.
- **Owner gates:** only Lowell rents, pays for, and terminates Lambda instances. No agent creates an account, enters a credential, or starts a paid resource.
- **Scope:** Stage 6 only. No sector states, common factor, cross-sector news factors, wedge or B/M rows, SA rows, Student-*t* or stochastic volatility, covariates, or BlackJAX. Stages 7–17 own them.

## Decisions

- **D1** (owner, planning questions, 2026-10-21): every closing cell, 1979–2026, enters as a same-release over-the-month change, and the March anchors enter as levels on the latent path. Applied in: Global Constraints, Task 1 (spec amendment), Task 2, Task 4.
- **D2** (owner, planning questions, 2026-10-21; narrowed after the spike, 2026-10-21): the crisis multiplier scales the non-news level shock and the S and T news shocks in March–June 2020, and not the slope, seasonal, post-T news, or noise terms. The first answer also scaled the post-T news. The spike showed that this left April 2020's latent change almost unidentified (total SD 5,634), and Lowell narrowed it. Applied in: Global Constraints, Task 1 (spec amendment), Task 4, Task 5, Task 7.
- **D3** (owner, planning questions, 2026-10-21): the GPU measurement rents one single-GPU on-demand instance each of A10, A100 40 GB, and GH200 from Lambda, with a \$40 cap on the total. Applied in: Task 13.
- **D4** (owner, planning questions, 2026-10-21): h5netcdf and h5py join the runtime dependencies so that a run's InferenceData is one file. Rank histograms are numeric, and plotting waits for Stage 10. Applied in: Global Constraints, Task 6.
- **D5** (owner, plan review, 2026-10-21): the GPU rule chooses a GPU type and count for each Stage 7–9 model, not one for every stage: Stage 7, and Stages 8 and 9 under each wedge reading. Stages 10–17 keep Stage 9's Kalman states and cells (Req 17 samples their mixing variables and volatility paths), so they take Stage 9's choice at the wedge Stage 8 settles, rerun with their own measured steps per iteration. The probe grid adds Stage 8's two wedge readings, so it has 20 entries. This narrows the roadmap's "GPU type and count for Stages 7–17" to one choice per model. The Handoff section names it for the resume. Applied in: Task 8, Task 9, Task 12, Task 13, Task 14.
- **D6** (owner, plan review, 2026-10-21): Task 1's spec wording is approved as shown: the Req 6 sentences on what the crisis multiplier scales, the Req 7 sentences that put NSA closing cells in the same-release change form and leave Req 9's SA rows to Stage 9, and the Rollout note line. Applied in: Task 1 (spec amendment).
- **D7** (owner, plan review, 2026-10-21): the 4-GPU row's SBC projection counts one hand-started shard per GPU, 25 rounds for 100 replicates, so that row's SBC can pass where one GPU's cannot. A configuration stays feasible only when its fit takes at most 48 hours and its SBC run at most 14 days. Applied in: Task 8, Task 14.
- **D8** (owner, plan review, 2026-10-21): the parallel-in-time Kalman filter stays out of Stage 6, and Stage 6 measures the engine as built. A toy-model check at review found that Dynamax's parallel filter returns a NaN log likelihood when the state has a deterministic part, as the pilot's does, and that it has no masking for missing cells. A two-pass form, a parallel scan for the filtered moments and then every month's log density at once, matched the sequential filter. If a model is infeasible, the roadmap's resume takes the parallel filter up. Applied in: Task 14.

---

## Source and numbering

- Source spec: [`specs/ces-revisions.md`](../ces-revisions.md): Req 4, Req 6, Req 7, Req 15, Req 17 step (2), Req 18.
- Roadmap stage: [`specs/ces-revisions-roadmap.md`](../ces-revisions-roadmap.md), Stage 6.
- Inputs: the Stage 1 engine (`src/ces_revisions/kalman.py`, `tests/dense_reference.py`), the Stage 3 vintage panel (`ces_revisions.vintages`), the Stage 4 annual tables (`ces_revisions.annual`), the Stage 5 operators (`ces_revisions.operators`), and the plan 3 device rule and engine probe (`devices.py`, `engine_probe.py`).
- **Plan ID 11.** `entry_check.py` reports 9 as the next free ID, but it cannot see the abandoned `codex/plan-10-cloud-gpu-runner` branch. The roadmap's 2026-10-20 amendment gives plans 9 and 10 to that branch and says that the next new plan takes ID 11, so that no ID is reused.
- **Retirement:** `specs/ces-revisions.md` does not retire with this plan, because later stages still implement it. On completion, run writing-plans' Plan Completion Protocol, step 3a included:
  1. In `specs/deferred_items.md`, rewrite plan 5's evidence-tests item as Execution notes give it, and append a `## 11-ces-revisions — YYYY-MM-DD` section with the three items that planning choice 9 records, each in the backlog's schema (`Size: design`):
     - the `σ_slope` conflict: the pilot's posterior median and SD of `sigma_slope` from `docs/runs/stage-6/pilot-fit/`, against its HalfNormal(15) prior. Revisit if: Stage 10's $`t_5`$ process innovations or Stage 16's prior-predictive gate still leave it above 3 prior scales.
     - the doubled-prior failure: the two figures in `docs/runs/stage-6/prior-predictive-doubled.json`. Revisit if: Stage 16's full prior-predictive gate runs.
     - the residual seasonality at lags 11–12: the Ljung–Box Q of each layout from `docs/runs/stage-6/seasonal-comparison.json`. Revisit if: Stage 18's posterior predictive suite or Stage 21's sensitivity suite traces a misfit to the seasonal block.
  2. Add `## Handoff to later stages` to this plan. Only Stage 7 names Stage 6 in its roadmap `Consumes`, so the section has one bullet, for Stage 7: the `ces_revisions.runs` and `ces_revisions.pilot` names it imports, `TARGET_ACCEPT_PROB` 0.95, the `news_post` stand-in that Stage 8 replaces, the chosen layout and each Stage 7–9 model's GPU from `docs/decisions/seasonal-state.md` (D5 narrows the roadmap's one GPU for Stages 7–17), the per-GPU SBC shards (`run_sbc` takes one worker on a GPU host), plan 1's open `A_t`/`Z_t` item, and the plan 11 section of `specs/deferred_items.md`. Add any deviation from execution.
  3. Tick Stage 6 in `specs/ces-revisions-roadmap.md`, and put this stamp under its entry:\
     `> Stage 6: COMPLETE (YYYY-MM-DD) — implemented by plan 11 (specs/plans/completed/11-ces-revisions.md). Its Handoff section names what Stage 7 consumes.`\
     `> Next: resume the roadmap.`
  4. Add the same completion line to the spec's Rollout note, after the plan 11 amendment note, as earlier stages did:\
     `Stage 6: COMPLETE (YYYY-MM-DD) — implemented by plan 11 (specs/plans/completed/11-ces-revisions.md). Next: resume the roadmap.`

## Scope check

Stage 6 names five deliverables: the pilot model, the run harness, three harnesses (simulator, SBC, prior predictive), the seasonal-state decision, and the GPU cost measurement. They form one subsystem. The harness is model-agnostic and tested on the Stage 1 synthetic pilot, but its only consumer in this stage is the pilot. The SBC, the prior predictive, and both decisions all run the pilot model. Splitting them into separate plans would duplicate the panel assembly and the model. Each task below still has its own test cycle and review gate, and the harness lives in its own package, so Stage 7 can import it without the pilot.

## The pilot model

This section states the model that Tasks 3–5 implement. Values are thousands of jobs, NSA, total nonfarm. One time step is one reference month from January 1979 to the release index's last month (August 2026), so $`T=572`$.

**State.** The state vector is, in this order,
`level, slope, lagged_value, <seasonal block>, news_s, news_t, news_post`.
With $`\gamma_t`$ the seasonal block's current seasonal value and $`x_t=\ell_t+\gamma_t`$:

```math
\ell_t=\ell_{t-1}+g_{t-1}+\eta^{\ell}_t+\varepsilon^S_t+\varepsilon^T_t+\varepsilon^P_t,\qquad
g_t=g_{t-1}+\eta^g_t,\qquad
\text{lagged\_value}_t=x_{t-1},
```

```math
\text{news\_s}_t=\varepsilon^S_t,\qquad\text{news\_t}_t=\varepsilon^T_t,\qquad\text{news\_post}_t=\varepsilon^P_t .
```

The news shocks enter the truth's own innovation, which is the Jacobs–van Norden reading of Req 7. The engine has no cross-covariance between state and observation noise, so the news must be states. The news states are white noise, so the same shock appears in the level and in its own state.

**The post-T news is a stand-in.** Req 7 splits the news after the third print into `B` and `M` increments that the benchmark rows and Req 10's operators induce. Stage 6 has no benchmark rows, so `news_post` stands in for both: a free monthly shock with the Req 15 `T→B` prior spread evenly over twelve months, whose yearly total the March anchors pin. Task 1's Req 7 amendment states this for Stages 6–7, and Stage 8's wedge rows replace it.

**Seasonal layouts.** `dummy` is Req 6's recursion $`\gamma_t=-\sum_{h=1}^{11}\gamma_{t-h}+\eta^\gamma_t`$ (11 states, one shock). `trig` holds harmonics 1–5 as rotating pairs plus the Nyquist harmonic (11 states, an independent shock on each). `trig4` holds harmonics 1–4 only (8 states). The seasonal prior scale is HalfNormal(50) for the dummy (Req 15). For the trigonometric layouts it is rescaled so that one unit of shock variance implies the same $`\operatorname{Var}(\gamma_t-\gamma_{t-12})`$ as in the dummy. The exact per-unit factors are $`v=2`$ for the dummy, $`72`$ for trig, and $`48`$ for trig4, so the scales are $`50/6`$ and $`50/\sqrt{24}`$.

**Shock scales.** With $`p_t=\mathbf 1(t<\text{May 2003})`$ and $`c_t=\mathbf 1(t\in\text{Mar–Jun 2020})`$:

```math
\operatorname{sd}(\eta^\ell_t)=\sigma_\ell e^{\kappa c_t},\quad
\operatorname{sd}(\eta^g_t)=\sigma_g,\quad
\operatorname{sd}(\eta^\gamma_t)=\sigma_\gamma,\quad
\operatorname{sd}(\varepsilon^S_t)=\sigma_S e^{\lambda p_t}e^{\kappa c_t},\quad
\operatorname{sd}(\varepsilon^T_t)=\sigma_T e^{\lambda p_t}e^{\kappa c_t},\quad
\operatorname{sd}(\varepsilon^P_t)=\sigma_P/\sqrt{12}.
```

The S and T news have means $`m_S+\mu p_t`$ and $`m_T+\mu p_t`$, and the post-T news has mean $`m_P/12`$. $`\sigma_P`$ and $`m_P`$ are `sigma_news_annual` and `mean_news_annual`, the annual scale and mean of the news revealed after the third print. The level's offset is the sum of the three news means. $`\lambda`$ (`shift_log_scale`) and $`\mu`$ (`shift_mean`) are Req 4's one composite May 2003 shift in scale and intercept.

**Observations.** Four columns, with $`\Delta x_t=x_t-\text{lagged\_value}_t`$:

```math
y^F_t=\Delta x_t-n^S_t-n^T_t-n^P_t+e^F_t,\quad
y^S_t=\Delta x_t-n^T_t-n^P_t+e^S_t,\quad
y^T_t=\Delta x_t-n^P_t+e^T_t,\quad
y^A_t=x_t+o_t+\xi_t .
```

The anchor column $`y^A`$ is observed only in the 23 Marches 2003–2025. Its offset $`o_t`$ is the sum of the earlier benchmark years' reconstruction terms, because each documented reconstruction shifts the vintage basis while the latent path stays on one concept. The closing noise follows Req 7's recursion $`e^F=z^F`$, $`e^S=\rho e^F+z^S`$, $`e^T=\rho e^S+z^T`$, with $`\operatorname{sd}(z^j_t)=\sigma^e_j e^{\lambda p_t}`$, plus the rounding variance $`2/12`$ of a same-release change (Req 15). The anchor variance is $`\sigma_{\text{anchor}}^2+1/12`$, with $`\sigma_{\text{anchor}}`$ = `sigma_anchor`, the Req 15 final-benchmark error.

**Priors.** The 17 sampled sites:

| Site | Prior | Source |
|---|---|---|
| `sigma_level` | HalfNormal(150) | Req 15 local-level row |
| `sigma_slope` | HalfNormal(15) | Req 15 slope row |
| `sigma_seasonal` | HalfNormal(layout scale) | Req 15 seasonal row, rescaled |
| `sigma_news_s` | LogNormal(log 40, 0.40) | Req 15 `F→S` row |
| `sigma_news_t` | LogNormal(log 35, 0.45) | Req 15 `S→T` row |
| `sigma_news_annual` | LogNormal(log 300, 0.60); monthly SD is /√12 | Req 15 `T→B` row |
| `noise_f_raw`, `noise_s_raw`, `noise_t_raw` | Normal(0, 1), non-centered | Req 15 noise row |
| `rho_noise` | TruncatedNormal(0, 0.20, ±0.8) | Req 15 cross-stage row |
| `mean_news_s`, `mean_news_t` | Normal(0, 15) | Req 15 mean monthly news |
| `mean_news_annual` | Normal(0, 100); monthly mean is /12 | Req 15 mean annual news |
| `sigma_anchor` | HalfNormal(100) | Req 15 final benchmark error |
| `kappa_crisis` | Normal(log 50, 0.5) | Req 6 |
| `shift_log_scale` | Normal(0, 0.5) | Req 4 |
| `shift_mean` | Normal(0, 15) | Req 4 |

The noise SDs are $`\sigma^e_F=0.5\,\sigma_S e^{0.7 z_F}`$ and $`\sigma^e_{S,T}=0.5\,\sigma_T e^{0.7 z_{S,T}}`$: Req 15's LogNormal(log(0.5 σ), 0.7) noise row. F's noise is centered on half the scale of the `F→S` news that next revises it, and S's on half the `S→T` scale. The news that next revises T is the annual post-T news, which only the March anchors pin, so T's noise is also centered on half $`\sigma_T`$. That is this plan's reading of Req 15's $`\sigma_{s,j}`$ for T. The news shares are $`\sigma_S^2/(\sigma_S^2+\operatorname{Var}(e^S-e^F))`$ and $`\sigma_T^2/(\sigma_T^2+\operatorname{Var}(e^T-e^S))`$ at the post-2003 scale.

## Planning evidence (2026-10-21)

All figures come from a spike in a scratch clone at `a2f6937`. The spike implemented the code in this plan and ran every evidence step. Two models ran. The **first model** applied D2 as first answered, with κ on the post-T news as well. The **plan's model** narrows D2 (Decisions). The fit, April 2020, and LOO figures below come from the plan's model. The seasonal comparison and the SBC ran under the first model, and Tasks 11 and 12 rerun them under the plan's model. The prior predictive is the same under both models, because the post-T news cancels out of an F-to-T revision.

### Assembly

- $`T=572`$ (1979-01 to 2026-08). Observed cells: F 571, S 523, T 522, anchor 23.
- 92 S and T cells are carrier-masked: 46 before May 2003 and 46 from it. Each benchmark-final year from 1979 to 2025 masks 2 cells, except 2002. The 2002 final carried the two 2003-gap cells, which are already missing.
- From May 2003, 21 closing cells beyond the vintage files' frontier come from the revision table. Each cell has one source.
- `anchor_offset` by year: 2003–2010 = 0, 2011–2013 = 4, 2014–2015 = 470, 2016–2017 = 443, 2018–2019 = 446, 2020–2025 = 430.
- The first anchor, March 2003, falls on a `pre_2003` step.

### EMPLOY

- EMPLOY25M10 repeats EMPLOY25M9 in 944 of 944 values. Dropping it leaves 382,392 of 383,336 rows.
- 13 same-release cells disagree with the revision table, not 10 as plan 5's evidence line and its deferred item say. Ten are before 1997, and three are in EMPLOY20M5:

| Month | Stage | EMPLOY | Table |
|---|---|---:|---:|
| 1981-03 | T | 52 | 53 |
| 1982-11 | T | −127 | −110 |
| 1983-05 | T | 331 | 320 |
| 1983-09 | T | 1103 | 1116 |
| 1983-11 | T | 271 | 268 |
| 1984-06 | T | 349 | 367 |
| 1984-11 | T | 337 | 340 |
| 1984-12 | T | 184 | 187 |
| 1996-08 | S | 257 | 241 |
| 1996-08 | T | 280 | 263 |
| 2020-02 | T | 230 | 251 |
| 2020-03 | S | −870 | −881 |
| 2020-04 | F | −20500 | −20537 |

**Prior predictive** (dummy, 2,000 draws, about 1 s). Req 15's monthly constraint, read as "at least 80% of draws have a median absolute F→T revision in 20–100, and the pooled share of months above 250 lies in 0.001–0.05":
- Normal priors: share 0.9475, pooled 0.02885. Passes.
- Doubled log-scale SDs, the noise row's included (Req 15's sensitivity rerun): 0.6685 and 0.1333. Fails both bounds. Stage 16 owns the prior-sensitivity gate, so this is recorded, not acted on.

**Pilot fit** (dummy, 4 × (1,000 + 1,000), seed 2995).
- At NumPyro's default target acceptance of 0.8, the first model's fit fails Req 18: 7 divergences in the neck where `sigma_news_t` is small and trades against `noise_t_raw` (posterior correlation −0.83).
- At 0.95 both models meet Req 18. Under the plan's model (D2 as narrowed): R-hat at most 1.0038, minimum bulk ESS 1694, minimum tail ESS 1470, BFMI 0.97–1.0, 0 divergences, 1,648 s of sampling on the M4 Max. The 0.95 setting costs about 1.55 times the leapfrog steps of 0.8. A refit with the same seed is bitwise identical.
- Posterior medians under the plan's model: `σ_level` 110, **`σ_slope` 65.9 ± 6.4**, `σ_seasonal` 31.0, `σ_news_s` 46.8, `σ_news_t` 17.2, `σ_annual` 425, ρ 0.249, κ 4.05, `shift_log_scale` 0.218, `shift_mean` 11.8, `share_s` 0.72 (90% interval 0.55–0.85), `share_t` 0.65 (0.30–0.85).
- `σ_slope` sits 4.4 prior scales above its HalfNormal(15) prior, under both models. The likely cause is the 2020–21 recovery months outside the fixed crisis window: a Gaussian slope must bend to follow them. Req 6's $`t_5`$ process innovations (Stage 10) are the spec's answer. Stage 16's prior-sensitivity gate rechecks it.
- **April 2020.** Under the first model, the smoothed latent change had mean −19,689 against a third print of −19,756, but a total SD of 5,634, while its SD between draws was only 20. κ multiplied the post-T news SD by about 53 in each crisis month, and the March 2021 anchor pins only the year's total, so the four crisis months shared the whole benchmark year's news freely. Under the plan's model the change has mean −19,744 and total SD 122. Its 90% interval, −19,944 to −19,543, lies inside the ±5% band of −20,744 to −18,768.

**LOO and residuals** (plan's model). Benchmark-year elpd −9443 ± 230, against −9451 ± 233 for the first model. Pareto k is 1.78 for benchmark year 2021 (April 2020 to March 2021) and 0.71 for 2022. Month elpd −9440, with k 1.22 at April 2020 and 0.73 at July 2020, the first month after the crisis window. Ljung–Box Q(12) on the standardized one-step errors is 84, 49, and 61 in F, S, and T, and it rejects at 1% in all three. The lag-11 autocorrelation is about 0.18 in each column, and the lag-12 autocorrelation is 0.13–0.17.

**Seasonal comparison** (first model; all three layouts meet Req 18 at 0.95).
- trig − dummy: +15.4 ± 9.4 on benchmark-year blocks (1.64 SE), and +11.2 ± 8.5 (1.32 SE) without block 2021. trig roughly halves Q but still rejects in all three columns. It has the same 11 states as the dummy.
- trig4 − dummy: −466 ± 70. trig4 lacks harmonics 5 and 6.
- Measured on the Mac, trig4 saves 18% per value and gradient at Stage 9 low (2.445 s against 2.989 s) and 18% at Stage 7, not the 35% that a cubic projection gives. It also took 21% more leapfrog steps. Clause (b) of the rule fails on cost as well as on elpd.
- Verdict under the pre-registered rule (Task 8): keep the dummy.

**SBC** (first model; 100 replicates, 2003-05 to 2025-12, $`T=272`$, target 0.8, 500 warmup, 990 draws thinned to 99, 6 workers, 2 h 38 min). All 24 γ-ECDF tests pass at the Bonferroni critical value 0.000119. `share_s`: median contraction 0.853, |z| > 3 in 1% of replicates. `share_t`: contraction 0.605, |z| > 3 in 1%. 0 divergences. Median 555 s per replicate.
- Three of the 24 tests carried no information: the shift parameters met no pre-2003 data in that window, and `sigma_news_post` is an exact rescaling of `sigma_news_annual`. The plan's SBC (Task 11) fixes all three: it uses January 1999 to December 2021, which spans the May 2003 shift, the 2020 crisis, and 19 anchors, and it drops `sigma_news_post`, leaving 23 quantities. It also runs at the fit's target acceptance of 0.95, which costs about 1.55 times the leapfrog steps, so expect about 4 hours.
- No replicate drew `σ_slope` near the posterior's 65. SBC checks calibration under the prior, not the misfit behind that value.

**Cost on the Mac** (M4 Max CPU, 36 GiB).
- A pilot value and gradient takes 14 ms ($`n=17`$, $`p=4`$, $`T=572`$). In the first model's fit, the cost per leapfrog step was 13.4 ms over 130,585 steps of the slowest chain. Warmup costs 1.8–2.3 times the sampling steps, and iterations 0–100 average 310–580 steps.
- Stage 7–9 dimensions (states, cells):

| Stage | dummy | trig4 |
|---|---|---|
| 7 | 190, 37 | 157, 37 |
| 8, low wedge | 203, 85 | 170, 85 |
| 8, high wedge | 324, 217 | 291, 217 |
| 9, low wedge | 247, 145 | 214, 145 |
| 9, high wedge | 368, 277 | 335, 277 |

- Batch-1 value and gradient at $`T=572`$: dummy Stage 7 1.35 s, Stage 8 low 2.13 s, Stage 8 high 3.91 s, Stage 9 low 2.99 s, Stage 9 high 5.64 s. Batch 4 needs 26.9 GB of compiled memory at Stage 8 high and 37.4 GB at Stage 9 high, so neither ran under the Mac's 20 GB guard. One 4-chain Stage 9 high fit would take 685–725 hours sequentially on the Mac. The Mac is not viable past Stage 7, which is why the GPU measurement exists.
- On every Lambda target, the cp314 wheels exist for jaxlib 0.11.1 (manylinux `x86_64` and `aarch64`), `jax-cuda13-plugin` 0.11.1 (both), and `jax-cuda13-pjrt` 0.11.1 (both), and every package in `uv.lock` has an aarch64 build. So the GH200's aarch64 host can run `uv sync --extra cuda`.

**Planning choices** (mine, with reasons; each is open to change at plan review):
1. **Target acceptance 0.95** for the fit and every SBC replicate (`TARGET_ACCEPT_PROB` in `model.py`). Evidence: the 0.8 fit fails Req 18.
2. **The April 2020 check needs the whole 90% interval** of the smoothed change inside ±5% of the third print, not only the mean. The spike's first check passed a mean of −19,689 against −19,756 while the total SD was 5,634. That pass said nothing about identification.
3. **The SBC window and quantities** as described above.
4. **The seasonal rule is code** (`pilot/seasonal.py`). An alternative layout must meet Req 18 and pass its own April 2020 check. Clause (b) also needs its Ljung–Box Q at most 1.25 times the dummy's in every closing column: the spike's dummy rejects in all three columns, so a count of rejections could not bind. The rule runs on all benchmark-year blocks and again on the blocks with Pareto k ≤ 0.7 in every layout. If the two runs choose different layouts, the dummy stays and the decision record says the comparison is unreliable.
5. **The GPU rule is code** (`pilot/gpu_choice.py`). It runs once for each Stage 7–9 model (D5), on that model's own dimensions: a fit over $`T=572`$ and an SBC replicate over 280 months. It sizes every model at 127 leapfrog steps per iteration, the top of the 31–127 band, because Stages 7–9 sample far more parameters than the pilot. A grid entry belongs to a layout by its dimensions, so `trig` reads the dummy's timings. A configuration is one GPU running the four chains as a batch, or four GPUs with one chain each where Lambda rents that GPU type as a 4-GPU instance (A100 only, at planning time). It is feasible when its fit takes at most 48 hours and its SBC at most 14 days (D7). SBC is projected for the shipped runner: one replicate per GPU at a time, in whole rounds. `run_sbc` refuses more than one worker on a GPU host, so a 4-GPU instance runs four `pilot_run.py sbc --workers 1` shards, each under its own `CUDA_VISIBLE_DEVICES` with its own `--start` and `--stop`, into one output directory. A batched-runner projection is reported beside it and decides nothing. For each model, the cheapest feasible configuration in dollars per fit wins. If any model has every configuration measured and none feasible, the roadmap's resume re-validates Stages 7–17. A record that lacks an entry the rule needs gives the status "unmeasured", which decides nothing. "Infeasible" is recorded only when every D3 GPU type has a record, or Lowell confirms that a missing type could not be measured.
6. **SciPy is declared**, not new. `diagnostics.py` uses `scipy.stats.rankdata`, and `pilot_run.py` uses `scipy.stats.chi2`.
7. **`jax_versions()` and `nvidia_driver()` move from `engine_probe.py` to `devices.py`**, and `engine_probe.batched_value_and_grad()` becomes public, so that the harness and the grid script import no private names.
8. **The `A_t`/`Z_t` gradient-check item stays open.** No sampled parameter enters `transition_matrix` or `observation_matrix`: κ, λ, μ, and ρ enter only covariances and offsets. That item's closure condition is therefore not met.
9. **Recorded, not acted on:** the doubled-prior failure (Stage 16), the `σ_slope` conflict (Stages 10 and 16), and the residual seasonality at lags 11–12 (Stage 18's posterior predictive suite and Stage 21's sensitivity suite). The completion protocol records them as plan 11 deferred items (Retirement above).
10. **The post-T news stand-in and T's noise center** are this plan's readings of Req 7 and Req 15, stated in the model section. Lowell approved Task 1's Req 7 wording, which states the stand-in, at plan review (D6).

## File structure

Created:
- `src/ces_revisions/pilot/__init__.py`: package docstring.
- `src/ces_revisions/pilot/panel.py`: `AggregatePanel` and the panel constants.
- `src/ces_revisions/pilot/assembly.py`: builds the panel from the Stage 3–5 artifacts; EMPLOY repeat and disagreement evidence.
- `src/ces_revisions/pilot/layout.py`: the three seasonal layouts and the fixed system matrices.
- `src/ces_revisions/pilot/model.py`: priors, `pilot_ssm`, `pilot_model`, derived quantities, `TARGET_ACCEPT_PROB`.
- `src/ces_revisions/pilot/simulate.py`: an explicit NumPy simulator, revision statistics, the prior predictive, and Req 15's monthly constraint.
- `src/ces_revisions/pilot/checks.py`: the April 2020 posterior check.
- `src/ces_revisions/pilot/sbc.py`: γ-ECDF math and the replicate runner.
- `src/ces_revisions/pilot/dimensions.py`: Stage 7–9 state and cell counts and the measurement grid.
- `src/ces_revisions/pilot/seasonal.py`: the pre-registered seasonal-layout rule.
- `src/ces_revisions/pilot/gpu_choice.py`: fit and SBC projections and the per-stage GPU rule.
- `src/ces_revisions/pilot/decision_tables.py`: the decision record's lines and tables, from the committed JSON.
- `src/ces_revisions/runs/__init__.py`, `inits.py`, `diagnostics.py`, `inference_data.py`, `provenance.py`, `fit.py`: the model-agnostic run harness.
- `scripts/pilot_run.py`: the CLI (`prior-predictive`, `fit`, `sbc`, `sbc-summary`, `compare-seasonal`, `choose-gpu`).
- `scripts/gpu_probe_grid.py`: times the engine at the Stage 7–9 dimensions on one host.
- `tests/pilot_synthetic.py`: a bare-name helper with small synthetic panels and a reference SSM.
- `tests/test_pilot_assembly.py`, `test_pilot_layout.py`, `test_pilot_model.py`, `test_pilot_simulate.py`, `test_run_harness.py`, `test_pilot_checks.py`, `test_sbc.py`, `test_pilot_sbc_runner.py`, `test_pilot_seasonal.py`, `test_pilot_gpu_choice.py`, `test_pilot_dimensions.py`, `test_pilot_run.py`, `test_pilot_fit.py`, `test_seasonal_state_decision.py`.
- `docs/runs/stage-6/`: the committed evidence JSON.
- `docs/decisions/cloud-gpu-probe/mac-grid.json`, `a10-grid.json`, `a100-grid.json`, `gh200-grid.json`: the grid records.
- `docs/decisions/seasonal-state.md`: the decision record.

Modified:
- `specs/ces-revisions.md`: Req 6 and Req 7 amendments (Task 1).
- `src/ces_revisions/devices.py`, `src/ces_revisions/engine_probe.py`: the host helpers move, and the batched function becomes public (Task 6).
- `pyproject.toml`, `uv.lock`, `tests/test_stack.py`, `.gitignore` (Task 6).
- `tests/dense_reference.py`: `_joint_moments` becomes `joint_moments` (Task 5).
- `CLAUDE.md`, `README.md` (Task 14).
- `specs/deferred_items.md`, `specs/ces-revisions-roadmap.md`, `specs/ces-revisions.md` (Rollout note): the completion protocol (Retirement and Execution notes above).

## Execution notes

- **Long runs.** Task 10's fit takes about 30 minutes. Task 11's SBC takes about 4 hours on six workers. Task 12's two extra fits take about 35 minutes each. Run each in the background with output to a log, and do not run another timing-sensitive job beside it. SBC writes one JSON file per replicate, so a stopped run resumes with the same command.
- **Built artifacts.** The scripts read `data/panel/`, `data/annual/panel/`, and `data/operators/`. On a fresh checkout, build them first: `uv run python scripts/vintage_sources.py build`, then `uv run python scripts/annual_sources.py build`, then `uv run python scripts/operator_artifacts.py build`. Tests need none of them, because they build the same frames in memory.
- **A clean tree for evidence runs.** `run_provenance` records `git status`, untracked files included. Before Tasks 10–12, commit every code change, this plan file, and any checkbox ticks in it. Each evidence task commits its JSON before its next run starts, so each committed run record shows `"dirty": false`.
- **Expected numbers.** The evidence tasks give the spike's values as expectations. A new value that misses an acceptance gate stops the task: report it rather than tune past it.
- **Deferred items.** The completion protocol ticks plan 1's dispersed-chain-starts item and plan 5's EMPLOY-repeat item. Plan 5's evidence-tests item stays unticked, because its B-to-M half (the three movement counts behind Deviation 5's stage rule) is still open. Plan 1's `A_t`/`Z_t` item stays open (planning choice 8). In `specs/deferred_items.md`, replace the whole evidence-tests item:

```markdown
- [ ] Review recommendations: evidence tests for two planning findings
      (whole-branch review of plan 5, triaged defer). Two findings in plan 5's
      Planning evidence rest on one-off queries: EMPLOY's same-vintage
      seasonally adjusted changes differ from the revision table in 10 cells
      between 1981 and 1996, the reason the 1979 leg's changes come from the
      table; and between B and M the not seasonally adjusted value moves in
      1,775 of 1,836 April-to-October unit-months but in 46 of 756
      January-to-March and 35 of 504 November-to-December, the evidence for
      Deviation 5's stage rule. Tests reproducing both from
      `tests/vintage_data.py`'s frames, in `tests/test_vintage_panel.py` and
      `tests/test_stage_labels.py`, would keep those choices auditable. Size:
      quick-fix. Done when: tests pin the 10 cells and the three counts.
```

with:

```markdown
- [ ] Review recommendations: evidence tests for two planning findings
      (whole-branch review of plan 5, triaged defer). Two findings in plan 5's
      Planning evidence rest on one-off queries: EMPLOY's same-vintage
      seasonally adjusted changes differ from the revision table in 13 cells,
      10 between 1981 and 1996 and 3 in EMPLOY20M5 (2020), the reason the 1979
      leg's changes come from the table; and between B and M the not
      seasonally adjusted value moves in 1,775 of 1,836 April-to-October
      unit-months but in 46 of 756 January-to-March and 35 of 504
      November-to-December, the evidence for Deviation 5's stage rule. Tests
      reproducing both from `tests/vintage_data.py`'s frames, in
      `tests/test_vintage_panel.py` and `tests/test_stage_labels.py`, would
      keep those choices auditable. EMPLOY half done in plan 11:
      `tests/test_pilot_assembly.py::test_employ_disagrees_with_the_revision_table_in_ten_pre_1997_and_three_2020_cells`.
      Size: quick-fix. Done when: tests pin the three B-to-M counts.
```

## Tasks

### Task 1: Spec amendment for D1 and D2

**Files:**
- Modify: `specs/ces-revisions.md` (Req 6, Req 7, Rollout note)

**Interfaces:**
- Consumes: Decisions D1, D2, and D6.
- Produces: the spec text that Tasks 2–5 implement. The spec's status line and every other requirement stay as they are.

Lowell approved this wording at plan review (D6): the D1 and D2 text, and the Req 7 sentences that replace the level display with the change form, attach news and noise to the month's change, scope the form to NSA cells, and state the Stages 6–7 post-T news stand-in. Apply the wording exactly.

- [ ] **Step 1: Amend Req 6's crisis multiplier (D2)**

In `specs/ces-revisions.md`, Req 6, replace:

```markdown
a sensitivity run replaces the multiplier with unrestricted level-jump states. A trigonometric
seasonal state may replace the dummy recursion if diagnostics favor it.
```

with:

```markdown
a sensitivity run replaces the multiplier with unrestricted level-jump states.
The multiplier scales the non-news level innovation and the `F→S` and `S→T` news increments,
which enter the truth's own innovation (Req 7). It does not scale the slope or seasonal
innovations, the transitory noise, or the post-third-print news, whose yearly total the
benchmark rows pin.
(Lowell, plan 11, 2026-10-21) A trigonometric seasonal state may replace the dummy recursion
if diagnostics favor it.
```

- [ ] **Step 2: Amend Req 7's measurement form (D1)**

In `specs/ces-revisions.md`, Req 7, replace:

```markdown
where $`n^N_{s,t,k}`$ is news first revealed at stage $`k`$ and $`e^N_{s,t,j}`$ transitory
noise. The `B` and `M` increments are **not** free monthly shocks: they are induced by the
```

with:

```markdown
where $`n^N_{s,t,k}`$ is news first revealed at stage $`k`$ and $`e^N_{s,t,j}`$ transitory
noise. The display above fixes which news each closing stage still lacks. The closing
stages enter the likelihood not as those levels but as the same-release over-the-month
change, the published change from $`t-1`$ to $`t`$ in the release that first prints stage
$`j`$ for month $`t`$:
$`d^N_{s,t,j}=\Delta x_{s,t}-\sum_{k\in\mathcal F(j)}n^N_{s,t,k}+e^N_{s,t,j}`$,
with $`n^N_{s,t,k}`$ and $`e^N_{s,t,j}`$ read as the news and noise of month $`t`$'s change.
So a print's news and noise attach to that month's change, and its distance from the truth
does not carry into later months' prints. This holds for every NSA closing cell, the 1979+
total-nonfarm leg of Req 4 included. Stage 9 settles whether Req 9's SA rows take the same
form. The news increments enter the latent process's own innovation (the Jacobs–van Norden
form), so each closing change is the true change less the news still to come, plus noise.
Benchmark anchors stay level observations of $`x`$. Before the wedge rows enter
(Stages 6–7), the news that a month's third print still lacks is one monthly increment, with
the Req 15 `T→B` prior spread evenly over twelve months, and the March anchors pin its yearly
total. (Lowell, plan 11, 2026-10-21)
From Stage 8 the `B` and `M` increments are **not** free monthly shocks: they are induced by the
```

- [ ] **Step 3: Note the amendment in the Rollout note**

In `specs/ces-revisions.md`, append to the end of the Rollout note, after the Stage 5 stamp:

```markdown

Amended 2026-10-21 at plan 11 (user decisions): Req 7 states that closing stages enter as
same-release over-the-month changes in the Jacobs–van Norden form, that benchmark anchors stay
levels, and that the post-third-print news is one monthly increment until Stage 8's wedge
rows (Req 10's B-vintage rows) induce the `B` and `M` increments. Req 6 states which
innovations the crisis multiplier scales. Stage 6 implements these. Stages 7–17 inherit them.
```

- [ ] **Step 4: Check the rendering and commit**

Run: `uv run ruff format --check specs/ces-revisions.md`
Expected: `1 file already formatted` (ruff scans Markdown only for Python code blocks, so this checks that nothing else changed shape).

Check by eye that each inline math span opens with `` $` `` and closes with `` `$ ``, and that no line starts with `=` or `-` inside the new text.

```bash
git add specs/ces-revisions.md
git commit -m "Amend Req 6 and Req 7 for the Stage 6 pilot decisions"
```

### Task 2: The aggregate observation panel

**Files:**
- Create: `src/ces_revisions/pilot/__init__.py`
- Create: `src/ces_revisions/pilot/panel.py`
- Create: `src/ces_revisions/pilot/assembly.py`
- Test: `tests/test_pilot_assembly.py`

**Interfaces:**
- Consumes: `ces_revisions.vintages` (`stage_changes` from the vintage files, `stage_labels` for both sources, `revision_table_changes` for the 1979 leg, `release_index`, the long panel, `REGIME_SHIFT_MONTH`), Stage 5's `benchmark_fixtures`, and Stage 4's `publication_calendar`. Tests build them through `tests/vintage_data.py`, `tests/annual_data.py`, and `tests/operator_data.py`; scripts read the built parquet.
- Produces:
  - `AggregatePanel` (frozen dataclass): `reference_months: tuple[date, ...]`, `y: np.ndarray` (T, 4) float64 with NaN for missing, `anchor_offset` (T,) float64, `pre_2003` (T,) bool, `crisis` (T,) bool, `benchmark_year` (T,) int64, `cells: pl.DataFrame` (`CELL_SCHEMA`); properties and methods `num_steps`, `observed_any() -> np.ndarray`, `window(first: date, last: date) -> AggregatePanel`.
  - Constants in `panel.py`: `FIRST_MONTH = date(1979, 1, 1)`, `CLOSING_STAGES = ("F", "S", "T")`, `COLUMNS = ("F", "S", "T", "anchor")`, `ANCHOR_COLUMN = 3`, `CRISIS_MONTHS` (March–June 2020), `ANCHOR_YEARS = range(2003, 2026)`, `CLOSING_ROUNDING_VARIANCE = 2/12`, `ANCHOR_ROUNDING_VARIANCE = 1/12`, `CELL_SCHEMA`, and `benchmark_year_of(month: date) -> int`.
  - In `assembly.py`: `build_aggregate_panel(stage_changes, stage_labels, table_changes, release_index, benchmark_fixtures, publication_calendar) -> AggregatePanel`, `carrier_cells(stage_labels, publication_calendar) -> pl.DataFrame`, `read_artifacts(root: Path = vintage_raw.ROOT) -> dict[str, pl.DataFrame]` (keys in `build_aggregate_panel`'s parameter order, so `build_aggregate_panel(**read_artifacts())` works), `employ_without_repeats(long_panel) -> pl.DataFrame`, `employ_table_disagreements(long_panel, table_changes) -> pl.DataFrame`, and `MASK_REASONS`, `CARRIER_MASKED_STAGES`.

This task carries two of Stage 6's exit tests: the 1979 leg is partially observed with exactly one 2003 shift (Req 4), and the EMPLOY evidence (the repeated EMPLOY25M10 vintage is dropped, and the 13 disagreements are pinned).

- [ ] **Step 1: Write the failing panel tests**

Create `tests/test_pilot_assembly.py`:

```python
"""The Stage 6 aggregate panel, assembled from the Stage 3, 4, and 5 artifacts."""

from datetime import date
from functools import cache
from pathlib import Path

import annual_data
import numpy as np
import operator_data
import polars as pl
import pytest
import vintage_data

from ces_revisions.pilot.assembly import (
    ARTIFACT_PATHS,
    MASK_REASONS,
    REPEATED_EMPLOY_ORIGINAL,
    REPEATED_EMPLOY_VINTAGE,
    build_aggregate_panel,
    carrier_cells,
    employ_table_disagreements,
    employ_without_repeats,
    read_artifacts,
)
from ces_revisions.pilot.panel import (
    ANCHOR_COLUMN,
    ANCHOR_YEARS,
    CELL_SCHEMA,
    COLUMNS,
    REGIME_SHIFT_MONTH,
    AggregatePanel,
    benchmark_year_of,
)
from ces_revisions.vintages.months import add_months, month_range


@cache
def inputs() -> dict[str, pl.DataFrame]:
    return {
        "stage_changes": vintage_data.stage_changes(),
        "stage_labels": vintage_data.labels(),
        "table_changes": vintage_data.table_changes(),
        "release_index": vintage_data.index(),
        "benchmark_fixtures": operator_data.artifact("benchmark_fixtures"),
        "publication_calendar": annual_data.artifact("publication_calendar"),
    }


@cache
def built() -> AggregatePanel:
    return build_aggregate_panel(**inputs())


def cell(panel: AggregatePanel, month: date, column: str) -> dict:
    return panel.cells.row(
        by_predicate=(pl.col("reference_month") == month)
        & (pl.col("column") == column),
        named=True,
    )


def value(panel: AggregatePanel, month: date, column: str) -> float:
    return panel.y[panel.reference_months.index(month), COLUMNS.index(column)]


def final_benchmark_dates() -> set[date]:
    calendar = inputs()["publication_calendar"]
    return set(
        calendar.filter(pl.col("publication_kind") == "benchmark_final")[
            "publication_date"
        ]
    )


def test_the_panel_spans_every_reference_month_from_1979_to_the_release_index():
    panel = built()
    assert panel.num_steps == 572
    assert panel.reference_months == tuple(
        month_range(date(1979, 1, 1), date(2026, 8, 1))
    )
    assert panel.reference_months[-1] == vintage_data.index()["reference_month"].max()
    assert panel.y.shape == (572, 4)
    assert panel.cells.height == 572 * 4
    assert dict(panel.cells.schema) == CELL_SCHEMA


def test_every_cell_has_one_row_and_is_nan_exactly_where_it_is_masked():
    panel = built()
    cells = panel.cells
    assert not cells.select("reference_month", "column").is_duplicated().any()
    assert cells["column"].to_list() == list(COLUMNS) * panel.num_steps
    masked = cells["mask_reason"].is_not_null().to_numpy()
    assert np.array_equal(np.isnan(panel.y).ravel(), masked)
    assert set(cells["mask_reason"].drop_nulls()) <= set(MASK_REASONS)
    counts = {
        (row["column"], row["mask_reason"]): row["len"]
        for row in cells.group_by("column", "mask_reason").len().iter_rows(named=True)
    }
    assert counts == {
        ("F", None): 571,
        ("F", "missing_vintage"): 1,
        ("S", None): 523,
        ("S", "benchmark_carrier"): 46,
        ("S", "missing_vintage"): 2,
        ("S", "right_censored"): 1,
        ("T", None): 522,
        ("T", "benchmark_carrier"): 46,
        ("T", "missing_vintage"): 2,
        ("T", "right_censored"): 2,
        ("anchor", None): 23,
        ("anchor", "no_anchor"): 549,
    }
    unobserved = cells.filter(
        pl.col("mask_reason").is_in(["missing_vintage", "right_censored"])
    )
    assert sorted(
        zip(unobserved["reference_month"], unobserved["column"], strict=True)
    ) == [
        (date(2003, 3, 1), "T"),
        (date(2003, 4, 1), "S"),
        (date(2003, 4, 1), "T"),
        (date(2025, 9, 1), "S"),
        (date(2025, 10, 1), "F"),
        (date(2026, 7, 1), "T"),
        (date(2026, 8, 1), "S"),
        (date(2026, 8, 1), "T"),
    ]
    assert unobserved["value_thousands"].is_null().all()


def test_each_closing_cell_is_read_from_one_source_chosen_by_reference_month():
    panel = built()
    closing = panel.cells.filter(pl.col("column") != "anchor")
    early = closing.filter(pl.col("reference_month") < REGIME_SHIFT_MONTH)
    assert set(early["source"]) == {"cesnaicsrev"}
    late = closing.filter(pl.col("reference_month") >= REGIME_SHIFT_MONTH)
    from_table = late.filter(pl.col("source") == "cesnaicsrev")
    frontier = inputs()["stage_labels"].filter(
        (pl.col("source") == "cesvinall")
        & (pl.col("sector") == "00")
        & (pl.col("seasonal_status") == "NSA")
        & (pl.col("status") == "beyond_frontier")
        & pl.col("release_stage").is_in(["F", "S", "T"])
    )
    assert from_table.height == frontier.height == 21
    assert sorted(
        zip(from_table["reference_month"], from_table["column"], strict=True)
    ) == sorted(
        zip(frontier["reference_month"], frontier["release_stage"], strict=True)
    )
    assert from_table["value_thousands"].is_not_null().all()
    assert (
        from_table["reference_month"].min(),
        from_table["reference_month"].max(),
    ) == (
        date(2025, 12, 1),
        date(2026, 8, 1),
    )
    assert set(late.filter(pl.col("source") != "cesnaicsrev")["source"]) == {
        "cesvinall"
    }

    keys = ["reference_month", "release_stage"]
    table = (
        inputs()["table_changes"]
        .filter(pl.col("seasonal_status") == "NSA")
        .select(*keys, source=pl.lit("cesnaicsrev"), expected="value_thousands")
    )
    files = (
        inputs()["stage_changes"]
        .filter(
            (pl.col("sector") == "00")
            & (pl.col("seasonal_status") == "NSA")
            & pl.col("release_stage").is_in(["F", "S", "T"])
        )
        .select(*keys, source=pl.lit("cesvinall"), expected="change_thousands")
    )
    compared = closing.rename({"column": "release_stage"}).join(
        pl.concat([table, files]), on=[*keys, "source"], how="left"
    )
    assert compared.height == closing.height
    assert (
        compared["value_thousands"].eq_missing(compared["expected"].cast(pl.Float64))
    ).all()
    # November 2003's third estimate is a cross-release change in the table (46); the
    # cell keeps the vintage file's same-release change.
    november = cell(panel, date(2003, 11, 1), "T")
    assert (november["source"], november["value_thousands"]) == ("cesvinall", 147.0)


def test_the_1979_leg_is_partially_observed_with_exactly_one_2003_shift():
    panel = built()
    months = np.array(panel.reference_months)
    assert np.array_equal(panel.pre_2003, months < REGIME_SHIFT_MONTH)
    assert np.count_nonzero(np.diff(panel.pre_2003.astype(np.int8))) == 1
    assert panel.reference_months[int(panel.pre_2003.sum())] == date(2003, 5, 1)
    leg = panel.y[panel.pre_2003, :ANCHOR_COLUMN]
    observed = (~np.isnan(leg)).sum(axis=0)
    assert leg.shape == (292, 3)
    assert observed.tolist() == [292, 268, 267]
    anchored = [
        month
        for month, anchor in zip(
            panel.reference_months, panel.y[:, ANCHOR_COLUMN], strict=True
        )
        if not np.isnan(anchor)
    ]
    assert anchored[0] == date(2003, 3, 1)
    assert all(month.year >= 2003 for month in anchored)
    for month, column in (
        (date(2003, 3, 1), "T"),
        (date(2003, 4, 1), "S"),
        (date(2003, 4, 1), "T"),
    ):
        assert np.isnan(value(panel, month, column))
        assert cell(panel, month, column)["mask_reason"] == "missing_vintage"
    assert panel.y[0].tolist()[:ANCHOR_COLUMN] == [-1681.0, -1818.0, -1759.0]


def test_final_benchmark_releases_mask_their_second_and_third_estimates():
    panel = built()
    finals = final_benchmark_dates()
    carriers = panel.cells.filter(pl.col("mask_reason") == "benchmark_carrier")
    assert set(carriers["column"]) == {"S", "T"}
    assert set(carriers["published_date"]) <= finals
    assert carriers["value_thousands"].is_not_null().all()
    assert carriers.filter(pl.col("reference_month") < REGIME_SHIFT_MONTH).height == 46
    assert carriers.filter(pl.col("reference_month") >= REGIME_SHIFT_MONTH).height == 46
    first = panel.cells.filter(pl.col("column") == "F")
    assert first.filter(pl.col("published_date").is_in(finals)).height == 47
    assert first.filter(pl.col("mask_reason") == "benchmark_carrier").is_empty()

    labels = carrier_cells(inputs()["stage_labels"], inputs()["publication_calendar"])
    assert labels["release_stage"].value_counts().sort("release_stage").rows() == [
        ("F", 47),
        ("S", 46),
        ("T", 46),
    ]
    masked = labels.filter(pl.col("release_stage") != "F")
    assert sorted(
        zip(masked["reference_month"], masked["release_stage"], strict=True)
    ) == (sorted(zip(carriers["reference_month"], carriers["column"], strict=True)))
    per_year = dict(masked.group_by("publication_id").len().iter_rows())
    expected = {
        f"benchmark_final_{year}": 2 for year in range(1979, 2026) if year != 2002
    }
    assert per_year == expected
    # The March 2002 benchmark arrived in the May 2003 release, whose second and third
    # estimates are the 2003-gap cells: missing, so undated and never carrier-masked.
    gap = labels.filter(pl.col("publication_id") == "benchmark_final_2002")
    assert gap.select("reference_month", "release_stage").rows() == [
        (date(2003, 5, 1), "F")
    ]
    assert cell(panel, date(2003, 12, 1), "S")["mask_reason"] == "benchmark_carrier"
    assert cell(panel, date(2003, 11, 1), "T")["published_date"] == date(2004, 2, 6)
    assert cell(panel, date(2004, 1, 1), "F")["mask_reason"] is None


def test_anchors_are_the_march_wedge_anchors_and_offsets_sum_earlier_reconstructions():
    panel = built()
    marches = (
        inputs()["benchmark_fixtures"]
        .filter((pl.col("sector") == "00") & (pl.col("wedge_weight") == 1.0))
        .sort("benchmark_year")
    )
    anchor_months = [date(year, 3, 1) for year in ANCHOR_YEARS]
    assert marches["reference_month"].to_list() == anchor_months
    rows = [panel.reference_months.index(month) for month in anchor_months]
    assert np.array_equal(
        panel.y[rows, ANCHOR_COLUMN], marches["wedge_anchor_thousands"].to_numpy()
    )
    assert int((~np.isnan(panel.y[:, ANCHOR_COLUMN])).sum()) == 23
    offsets = {
        month.year: float(panel.anchor_offset[row])
        for month, row in zip(anchor_months, rows, strict=True)
    }
    assert offsets == {
        **dict.fromkeys(range(2003, 2011), 0.0),
        **dict.fromkeys(range(2011, 2014), 4.0),
        **dict.fromkeys(range(2014, 2016), 470.0),
        **dict.fromkeys(range(2016, 2018), 443.0),
        **dict.fromkeys(range(2018, 2020), 446.0),
        **dict.fromkeys(range(2020, 2026), 430.0),
    }
    terms = dict(
        zip(
            marches["benchmark_year"],
            marches["reconstruction_term_thousands"],
            strict=True,
        )
    )
    assert offsets[2014] == terms[2010] + terms[2013] == 4.0 + 466.0
    assert offsets[2025] == sum(terms[year] for year in range(2003, 2025))
    elsewhere = np.ones(panel.num_steps, dtype=bool)
    elsewhere[rows] = False
    assert not panel.anchor_offset[elsewhere].any()
    anchor = cell(panel, date(2013, 3, 1), "anchor")
    assert (anchor["status"], anchor["source"], anchor["published_date"]) == (
        "anchor",
        "benchmark_fixtures",
        date(2014, 2, 7),
    )


def test_crisis_and_benchmark_year_flags_follow_the_reference_month():
    panel = built()
    crisis = [
        month
        for month, flag in zip(panel.reference_months, panel.crisis, strict=True)
        if flag
    ]
    assert crisis == [date(2020, month, 1) for month in range(3, 7)]
    assert panel.benchmark_year.tolist() == [
        benchmark_year_of(month) for month in panel.reference_months
    ]
    assert panel.benchmark_year[0] == 1979
    assert panel.benchmark_year[panel.reference_months.index(date(2003, 4, 1))] == 2004


def test_read_artifacts_reads_the_six_inputs_and_rebuilds_the_same_panel(
    tmp_path: Path,
):
    frames = {
        **inputs(),
        "table_changes": vintage_data.long_panel(),
    }
    for name, path in ARTIFACT_PATHS.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        frames[name].write_parquet(tmp_path / path)
    read = read_artifacts(tmp_path)
    assert list(read) == list(ARTIFACT_PATHS)
    assert set(read["table_changes"]["source"]) == {"cesnaicsrev"}
    assert read["table_changes"].height == vintage_data.table_changes().height
    panel, expected = build_aggregate_panel(**read), built()
    assert panel.reference_months == expected.reference_months
    for name in ("y", "anchor_offset", "pre_2003", "crisis", "benchmark_year"):
        assert getattr(panel, name).tobytes() == getattr(expected, name).tobytes()
    assert panel.cells.equals(expected.cells)


def test_the_repeated_october_2025_employ_vintage_is_dropped():
    long_panel = vintage_data.long_panel()
    employ = long_panel.filter(pl.col("source") == "rtdsm_employ")
    october = employ.filter(pl.col("vintage_id") == REPEATED_EMPLOY_VINTAGE)
    september = employ.filter(pl.col("vintage_id") == REPEATED_EMPLOY_ORIGINAL)
    assert october.height == september.height == 944
    assert october.sort("reference_month")["value_thousands"].equals(
        september.sort("reference_month")["value_thousands"]
    )
    kept = employ_without_repeats(long_panel)
    assert kept.height == employ.height - 944
    assert REPEATED_EMPLOY_VINTAGE not in set(kept["vintage_id"])
    assert kept["vintage_id"].n_unique() == 570
    assert (
        kept.group_by("release_month")
        .agg(pl.col("vintage_id").n_unique())["vintage_id"]
        .max()
        == 1
    )


def test_an_october_2025_employ_vintage_that_is_not_a_repeat_raises():
    long_panel = vintage_data.long_panel().filter(pl.col("source") == "rtdsm_employ")
    changed = long_panel.with_columns(
        value_thousands=pl.when(
            (pl.col("vintage_id") == REPEATED_EMPLOY_VINTAGE)
            & (pl.col("reference_month") == date(2025, 8, 1))
        )
        .then(pl.col("value_thousands") + 1)
        .otherwise("value_thousands")
    )
    with pytest.raises(ValueError, match="not an exact repeat"):
        employ_without_repeats(changed)
    with pytest.raises(ValueError, match="not an exact repeat"):
        employ_without_repeats(
            long_panel.filter(pl.col("vintage_id") != REPEATED_EMPLOY_VINTAGE)
        )


def test_another_doubled_employ_release_raises():
    long_panel = vintage_data.long_panel().filter(pl.col("source") == "rtdsm_employ")
    doubled = long_panel.filter(
        pl.col("vintage_id") == "rtdsm:EMPLOY24M6"
    ).with_columns(vintage_id=pl.lit("rtdsm:EMPLOY24M6bis"))
    with pytest.raises(ValueError, match="two EMPLOY vintages"):
        employ_without_repeats(pl.concat([long_panel, doubled]))


def test_employ_disagrees_with_the_revision_table_in_ten_pre_1997_and_three_2020_cells():
    disagreements = employ_table_disagreements(
        vintage_data.long_panel(), vintage_data.table_changes()
    )
    assert disagreements.columns == [
        "reference_month",
        "release_stage",
        "vintage_id",
        "employ_thousands",
        "table_thousands",
    ]
    assert disagreements.rows() == [
        (date(1981, 3, 1), "T", "rtdsm:EMPLOY81M6", 52, 53),
        (date(1982, 11, 1), "T", "rtdsm:EMPLOY83M2", -127, -110),
        (date(1983, 5, 1), "T", "rtdsm:EMPLOY83M8", 331, 320),
        (date(1983, 9, 1), "T", "rtdsm:EMPLOY83M12", 1103, 1116),
        (date(1983, 11, 1), "T", "rtdsm:EMPLOY84M2", 271, 268),
        (date(1984, 6, 1), "T", "rtdsm:EMPLOY84M9", 349, 367),
        (date(1984, 11, 1), "T", "rtdsm:EMPLOY85M2", 337, 340),
        (date(1984, 12, 1), "T", "rtdsm:EMPLOY85M3", 184, 187),
        (date(1996, 8, 1), "S", "rtdsm:EMPLOY96M10", 257, 241),
        (date(1996, 8, 1), "T", "rtdsm:EMPLOY96M11", 280, 263),
        (date(2020, 2, 1), "T", "rtdsm:EMPLOY20M5", 230, 251),
        (date(2020, 3, 1), "S", "rtdsm:EMPLOY20M5", -870, -881),
        (date(2020, 4, 1), "F", "rtdsm:EMPLOY20M5", -20500, -20537),
    ]
    pre_1997 = disagreements.filter(pl.col("reference_month") < date(1997, 1, 1))
    assert pre_1997.height == 10
    # Same release on both sides: each EMPLOY vintage holds the release that made the
    # table's estimate, the reference month plus the stage's offset.
    releases = dict(
        vintage_data.rtdsm().select("vintage_id", "release_month").unique().iter_rows()
    )
    for row in disagreements.iter_rows(named=True):
        offset = "FST".index(row["release_stage"])
        assert releases[row["vintage_id"]] == add_months(row["reference_month"], offset)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_pilot_assembly.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'ces_revisions.pilot'`.

- [ ] **Step 3: Write the package, the panel dataclass, and the assembly**

Create `src/ces_revisions/pilot/__init__.py`:

```python
"""Roadmap Stage 6: the aggregate total-nonfarm linear-Gaussian pilot.

``panel`` defines the observation panel, ``assembly`` builds it from the Stage 3, 4, and 5
artifacts, ``layout`` and ``model`` define the state-space model and its priors,
``simulate`` draws synthetic panels from it, ``sbc`` checks its calibration, and
``dimensions`` sizes the Stage 7-9 models for the GPU measurement.
"""
```

Create `src/ces_revisions/pilot/panel.py`:

```python
"""The aggregate total-nonfarm observation panel of the Stage 6 pilot.

One row per reference month from January 1979, and four cells per row: the first,
second, and third same-release over-the-month changes of not seasonally adjusted total
nonfarm employment, and the final March benchmark anchor, all in thousands. A NaN cell
is not observed: never published, missing in the 2003 gap or the 2025 lapse, beyond the
release index, a non-March anchor cell, or masked because a benchmark release carried it.
"""

import itertools
from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

from ces_revisions.vintages.panel import REGIME_SHIFT_MONTH

FIRST_MONTH = date(1979, 1, 1)
CLOSING_STAGES = ("F", "S", "T")
COLUMNS = (*CLOSING_STAGES, "anchor")
ANCHOR_COLUMN = COLUMNS.index("anchor")
# Req 6's known crisis window, March through June 2020.
CRISIS_MONTHS = tuple(date(2020, month, 1) for month in range(3, 7))
# Benchmark years with a final anchor on the vintage-file basis (Stage 5's wedge fixtures).
ANCHOR_YEARS = tuple(range(2003, 2026))
# A same-release change differences two independently rounded whole-thousand levels;
# an anchor is one rounded level (Req 15's 1/12).
CLOSING_ROUNDING_VARIANCE = 2.0 / 12.0
ANCHOR_ROUNDING_VARIANCE = 1.0 / 12.0
CELL_SCHEMA = {
    "reference_month": pl.Date,
    "column": pl.String,
    "source": pl.String,
    "value_thousands": pl.Float64,
    "status": pl.String,
    "mask_reason": pl.String,
    "published_date": pl.Date,
}

__all__ = [
    "ANCHOR_COLUMN",
    "ANCHOR_ROUNDING_VARIANCE",
    "ANCHOR_YEARS",
    "CELL_SCHEMA",
    "CLOSING_ROUNDING_VARIANCE",
    "CLOSING_STAGES",
    "COLUMNS",
    "CRISIS_MONTHS",
    "FIRST_MONTH",
    "REGIME_SHIFT_MONTH",
    "AggregatePanel",
    "benchmark_year_of",
]


def benchmark_year_of(month: date) -> int:
    """The benchmark year whose April-to-March window holds ``month`` (Req 10's I_y)."""
    return month.year + 1 if month.month >= 4 else month.year


@dataclass(frozen=True)
class AggregatePanel:
    """The pilot's observations and the per-month flags the model reads.

    ``y`` holds the published values exactly; the anchor cell's offset for earlier
    reconstructions is kept apart in ``anchor_offset`` so the model adds it as an
    observation offset. ``cells`` is the audit table: one row per (month, column).
    """

    reference_months: tuple[date, ...]
    y: np.ndarray  # (T, 4) float64
    anchor_offset: np.ndarray  # (T,) float64; zero outside anchor months
    pre_2003: np.ndarray  # (T,) bool; reference month before REGIME_SHIFT_MONTH
    crisis: np.ndarray  # (T,) bool; reference month in CRISIS_MONTHS
    benchmark_year: np.ndarray  # (T,) int64
    cells: pl.DataFrame  # CELL_SCHEMA

    def __post_init__(self) -> None:
        num_steps = len(self.reference_months)
        if self.y.shape != (num_steps, len(COLUMNS)) or self.y.dtype != np.float64:
            raise ValueError(
                f"y must be float64 of shape ({num_steps}, {len(COLUMNS)})"
            )
        for name, dtype in (
            ("anchor_offset", np.float64),
            ("pre_2003", np.bool_),
            ("crisis", np.bool_),
            ("benchmark_year", np.int64),
        ):
            array = getattr(self, name)
            if array.shape != (num_steps,) or array.dtype != dtype:
                raise ValueError(
                    f"{name} must be {np.dtype(dtype)} of shape ({num_steps},)"
                )
        months = self.reference_months
        if any(
            (later.year - earlier.year) * 12 + later.month - earlier.month != 1
            for earlier, later in itertools.pairwise(months)
        ):
            raise ValueError("reference_months must be consecutive months")
        if dict(self.cells.schema) != CELL_SCHEMA:
            raise ValueError(f"cells must have schema {CELL_SCHEMA}")

    @property
    def num_steps(self) -> int:
        return len(self.reference_months)

    def observed_any(self) -> np.ndarray:
        """Steps with at least one observed cell; LOO drops the rest (they add zero)."""
        return ~np.isnan(self.y).all(axis=1)

    def window(self, first: date, last: date) -> AggregatePanel:
        """The panel restricted to reference months first..last, both included."""
        keep = np.array([first <= month <= last for month in self.reference_months])
        return AggregatePanel(
            reference_months=tuple(
                month
                for month, kept in zip(self.reference_months, keep, strict=True)
                if kept
            ),
            y=self.y[keep].copy(),
            anchor_offset=self.anchor_offset[keep].copy(),
            pre_2003=self.pre_2003[keep].copy(),
            crisis=self.crisis[keep].copy(),
            benchmark_year=self.benchmark_year[keep].copy(),
            cells=self.cells.filter(pl.col("reference_month").is_between(first, last)),
        )
```

Create `src/ces_revisions/pilot/assembly.py`:

```python
"""Assemble the pilot's aggregate panel from the Stage 3, 4, and 5 artifacts.

Each closing cell, a first, second, or third same-release over-the-month change of not
seasonally adjusted total nonfarm, is read from one source, chosen by its reference
month: BLS's revision table before May 2003, the vintage files from it, and the revision
table again where the vintage files end before the cell's release. The final March
benchmark anchors are Stage 5's wedge anchors, with earlier reconstructions kept apart as
an observation offset. A second or third estimate that a final benchmark release carried
is masked, because that release also revises the levels it is differenced from.

The EMPLOY helpers keep the evidence for taking the 1979 leg's changes from the revision
table: the repeated October 2025 vintage, and the cells where EMPLOY's same-vintage
seasonally adjusted changes disagree with the table.
"""

from pathlib import Path

import numpy as np
import polars as pl

from ces_revisions.pilot.panel import (
    ANCHOR_YEARS,
    CELL_SCHEMA,
    CLOSING_STAGES,
    COLUMNS,
    CRISIS_MONTHS,
    FIRST_MONTH,
    REGIME_SHIFT_MONTH,
    AggregatePanel,
    benchmark_year_of,
)
from ces_revisions.vintages import raw as vintage_raw
from ces_revisions.vintages.months import month_range

SECTOR = "00"
SEASONAL_STATUS = "NSA"
TABLE_SOURCE = "cesnaicsrev"
VINTAGE_FILE_SOURCE = "cesvinall"
EMPLOY_SOURCE = "rtdsm_employ"
ANCHOR_SOURCE = "benchmark_fixtures"
CARRIER_PUBLICATION = "benchmark_final"
# A final benchmark release revises the levels behind its second and third estimates;
# its first estimate is the new month's own change.
CARRIER_MASKED_STAGES = ("S", "T")
# October 2025 had no Employment Situation release, so its EMPLOY vintage repeats
# September's and resolves to the same release (specs/deferred_items.md, plan 5).
REPEATED_EMPLOY_VINTAGE = "rtdsm:EMPLOY25M10"
REPEATED_EMPLOY_ORIGINAL = "rtdsm:EMPLOY25M9"
MASK_REASONS = ("benchmark_carrier", "missing_vintage", "right_censored", "no_anchor")
ARTIFACT_PATHS = {
    "stage_changes": Path("data/panel/stage_changes.parquet"),
    "stage_labels": Path("data/panel/stage_labels.parquet"),
    "table_changes": Path("data/panel/panel.parquet"),
    "release_index": Path("data/panel/release_index.parquet"),
    "benchmark_fixtures": Path("data/operators/benchmark_fixtures.parquet"),
    "publication_calendar": Path("data/annual/panel/publication_calendar.parquet"),
}

_CELL = ["reference_month", "release_stage"]


def _total_nonfarm_closing(frame: pl.DataFrame, seasonal_status: str) -> pl.DataFrame:
    return frame.filter(
        (pl.col("sector") == SECTOR)
        & (pl.col("seasonal_status") == seasonal_status)
        & pl.col("release_stage").is_in(CLOSING_STAGES)
    )


def _source_labels(stage_labels: pl.DataFrame) -> pl.DataFrame:
    """One stage label per closing cell: the label of the source the cell is read from.

    The revision table's before May 2003 and the vintage files' from it, except where a
    vintage-file label is beyond the frontier: the table, which runs to the release
    index's last release, holds those cells.
    """
    closing = _total_nonfarm_closing(stage_labels, SEASONAL_STATUS)
    table = closing.filter(pl.col("source") == TABLE_SOURCE)
    files = closing.filter(
        (pl.col("source") == VINTAGE_FILE_SOURCE)
        & (pl.col("reference_month") >= REGIME_SHIFT_MONTH)
    )
    beyond = files.filter(pl.col("status") == "beyond_frontier").select(_CELL)
    chosen = pl.concat(
        [
            table.filter(pl.col("reference_month") < REGIME_SHIFT_MONTH),
            table.join(beyond, on=_CELL, how="semi"),
            files.filter(pl.col("status") != "beyond_frontier"),
        ]
    ).sort(_CELL)
    if chosen.select(_CELL).is_duplicated().any():
        raise ValueError("a closing cell has stage labels from two sources")
    return chosen


def _final_benchmark_dates(publication_calendar: pl.DataFrame) -> pl.DataFrame:
    return publication_calendar.filter(
        pl.col("publication_kind") == CARRIER_PUBLICATION
    ).select("publication_id", published_date="publication_date")


def carrier_cells(
    stage_labels: pl.DataFrame, publication_calendar: pl.DataFrame
) -> pl.DataFrame:
    """Sector 00 NSA closing-stage labels published in a final benchmark release.

    Every F, S, and T label, from the source its cell is read from, whose published date
    is a final benchmark publication date, with that publication's id. The panel masks
    the S and T rows. A missing vintage has no published date, so the 2003-gap cells
    the March 2002 benchmark release would have carried are not among them.
    """
    return (
        _source_labels(stage_labels)
        .join(
            _final_benchmark_dates(publication_calendar),
            on="published_date",
            how="inner",
        )
        .sort(_CELL)
    )


def _closing_cells(
    stage_changes: pl.DataFrame,
    stage_labels: pl.DataFrame,
    table_changes: pl.DataFrame,
    publication_calendar: pl.DataFrame,
) -> pl.DataFrame:
    values = pl.concat(
        [
            _total_nonfarm_closing(table_changes, SEASONAL_STATUS).select(
                *_CELL,
                source=pl.lit(TABLE_SOURCE),
                value_thousands=pl.col("value_thousands").cast(pl.Float64),
            ),
            _total_nonfarm_closing(stage_changes, SEASONAL_STATUS).select(
                *_CELL,
                source=pl.lit(VINTAGE_FILE_SOURCE),
                value_thousands=pl.col("change_thousands").cast(pl.Float64),
            ),
        ]
    )
    if values.select("source", *_CELL).is_duplicated().any():
        raise ValueError("a source holds two values for one closing cell")
    carried = (
        carrier_cells(stage_labels, publication_calendar)
        .filter(pl.col("release_stage").is_in(CARRIER_MASKED_STAGES))
        .select(*_CELL, carried=pl.lit(True))
    )
    cells = (
        _source_labels(stage_labels)
        .join(values, on=["source", *_CELL], how="left")
        .join(carried, on=_CELL, how="left")
        .with_columns(pl.col("carried").fill_null(False))
    )
    observed = pl.col("status") == "observed"
    mismatched = cells.filter(observed != pl.col("value_thousands").is_not_null())
    if mismatched.height:
        raise ValueError(
            f"{mismatched.height} closing cells disagree with their stage label's status"
        )
    unexpected = set(cells["status"]) - {
        "observed",
        "missing_vintage",
        "right_censored",
    }
    if unexpected:
        raise ValueError(f"unexpected closing-cell statuses {sorted(unexpected)}")
    return cells.select(
        "reference_month",
        column="release_stage",
        source="source",
        value_thousands="value_thousands",
        status="status",
        mask_reason=pl.when(~observed)
        .then("status")
        .when(pl.col("carried"))
        .then(pl.lit("benchmark_carrier"))
        .otherwise(pl.lit(None, dtype=pl.String)),
        published_date="published_date",
    )


def _anchor_rows(
    benchmark_fixtures: pl.DataFrame, release_index: pl.DataFrame
) -> pl.DataFrame:
    """The March anchor of each benchmark year, with earlier reconstructions' offset.

    A reconstruction shifts the vintage basis of every later benchmark release, while
    the wedge anchor removes only its own year's, so a year's anchor sits above the
    latent path by the March reconstruction terms of every earlier year.
    """
    marches = (
        benchmark_fixtures.filter(
            (pl.col("sector") == SECTOR) & (pl.col("wedge_weight") == 1.0)
        )
        .sort("benchmark_year")
        .with_columns(
            anchor_offset=pl.col("reconstruction_term_thousands")
            .cum_sum()
            .shift(1, fill_value=0.0)
        )
    )
    if marches["benchmark_year"].to_list() != list(ANCHOR_YEARS):
        raise ValueError(f"expected one March anchor for each year of {ANCHOR_YEARS}")
    if (marches["reference_month"].dt.month() != 3).any() or (
        marches["reference_month"].dt.year() != marches["benchmark_year"]
    ).any():
        raise ValueError("every anchor must be its benchmark year's March")
    releases = release_index.select(
        benchmark_release_month="reference_month", published_date="published_date"
    )
    return marches.join(releases, on="benchmark_release_month", how="left").select(
        "reference_month",
        "anchor_offset",
        column=pl.lit("anchor"),
        source=pl.lit(ANCHOR_SOURCE),
        value_thousands="wedge_anchor_thousands",
        status=pl.lit("anchor"),
        mask_reason=pl.lit(None, dtype=pl.String),
        published_date="published_date",
    )


def build_aggregate_panel(
    stage_changes: pl.DataFrame,
    stage_labels: pl.DataFrame,
    table_changes: pl.DataFrame,
    release_index: pl.DataFrame,
    benchmark_fixtures: pl.DataFrame,
    publication_calendar: pl.DataFrame,
) -> AggregatePanel:
    """The pilot panel over every reference month from January 1979 to the index's last."""
    months = month_range(FIRST_MONTH, release_index["reference_month"].max())
    grid = pl.DataFrame(
        {
            "reference_month": [month for month in months for _ in COLUMNS],
            "column": [column for _ in months for column in COLUMNS],
        },
        schema={"reference_month": pl.Date, "column": pl.String},
    )
    anchors = _anchor_rows(benchmark_fixtures, release_index)
    found = pl.concat(
        [
            _closing_cells(
                stage_changes, stage_labels, table_changes, publication_calendar
            ),
            anchors.drop("anchor_offset"),
        ]
    )
    cells = (
        grid.join(
            found, on=["reference_month", "column"], how="left", maintain_order="left"
        )
        .with_columns(
            status=pl.when(pl.col("column") == "anchor")
            .then(pl.col("status").fill_null("no_anchor"))
            .otherwise("status"),
            mask_reason=pl.when(
                (pl.col("column") == "anchor") & pl.col("status").is_null()
            )
            .then(pl.lit("no_anchor"))
            .otherwise("mask_reason"),
        )
        .select(list(CELL_SCHEMA))
        .cast(CELL_SCHEMA)
    )
    if cells.height != grid.height or cells["status"].is_null().any():
        raise ValueError("every panel cell needs exactly one source row")
    y = np.where(
        cells["mask_reason"].is_null().to_numpy(),
        cells["value_thousands"].fill_null(np.nan).to_numpy(),
        np.nan,
    ).reshape(len(months), len(COLUMNS))
    offsets = dict(
        zip(anchors["reference_month"], anchors["anchor_offset"], strict=True)
    )
    return AggregatePanel(
        reference_months=tuple(months),
        y=y,
        anchor_offset=np.array([offsets.get(month, 0.0) for month in months]),
        pre_2003=np.array([month < REGIME_SHIFT_MONTH for month in months]),
        crisis=np.array([month in CRISIS_MONTHS for month in months]),
        benchmark_year=np.array(
            [benchmark_year_of(month) for month in months], dtype=np.int64
        ),
        cells=cells,
    )


def read_artifacts(root: Path = vintage_raw.ROOT) -> dict[str, pl.DataFrame]:
    """The six inputs of build_aggregate_panel, read from the built artifacts under root.

    The revision table's changes are the long panel's cesnaicsrev rows.
    """
    frames = {
        name: pl.read_parquet(root / path)
        for name, path in ARTIFACT_PATHS.items()
        if name != "table_changes"
    }
    frames["table_changes"] = (
        pl.scan_parquet(root / ARTIFACT_PATHS["table_changes"])
        .filter(pl.col("source") == TABLE_SOURCE)
        .collect()
    )
    return {name: frames[name] for name in ARTIFACT_PATHS}


def employ_without_repeats(long_panel: pl.DataFrame) -> pl.DataFrame:
    """The long panel's EMPLOY rows without the repeated October 2025 vintage."""
    employ = long_panel.filter(pl.col("source") == EMPLOY_SOURCE)
    columns = ["reference_month", "release_month", "value_thousands"]
    repeat = employ.filter(pl.col("vintage_id") == REPEATED_EMPLOY_VINTAGE).select(
        columns
    )
    original = employ.filter(pl.col("vintage_id") == REPEATED_EMPLOY_ORIGINAL).select(
        columns
    )
    if repeat.is_empty() or not repeat.sort(columns).equals(original.sort(columns)):
        raise ValueError(
            f"{REPEATED_EMPLOY_VINTAGE} is not an exact repeat of "
            f"{REPEATED_EMPLOY_ORIGINAL}"
        )
    kept = employ.filter(pl.col("vintage_id") != REPEATED_EMPLOY_VINTAGE)
    doubled = (
        kept.group_by("release_month")
        .agg(pl.col("vintage_id").n_unique())
        .filter(pl.col("vintage_id") > 1)
    )
    if doubled.height:
        raise ValueError(
            f"release months still mapping to two EMPLOY vintages: "
            f"{sorted(doubled['release_month'])}"
        )
    return kept


def _months_after(later: str, earlier: str) -> pl.Expr:
    return (pl.col(later).dt.year() - pl.col(earlier).dt.year()) * 12 + (
        pl.col(later).dt.month().cast(pl.Int32)
        - pl.col(earlier).dt.month().cast(pl.Int32)
    )


def employ_table_disagreements(
    long_panel: pl.DataFrame, table_changes: pl.DataFrame
) -> pl.DataFrame:
    """Cells where EMPLOY's same-vintage SA F, S, or T change differs from the table's.

    An EMPLOY vintage's change is a month's level minus the prior month's in that
    vintage; its stage counts the months from the reference month to the vintage's
    release. Only cells both sources hold, from the same release, are compared.
    """
    employ = employ_without_repeats(long_panel)
    prior = employ.select(
        "vintage_id",
        reference_month=pl.col("reference_month").dt.offset_by("1mo"),
        prior_thousands="value_thousands",
    )
    stages = dict(enumerate(CLOSING_STAGES))
    changes = (
        employ.join(prior, on=["vintage_id", "reference_month"], how="inner")
        .with_columns(lag=_months_after("release_month", "reference_month"))
        .filter(pl.col("lag").is_in(list(stages)))
        .select(
            "reference_month",
            "release_month",
            "vintage_id",
            release_stage=pl.col("lag").replace_strict(stages, return_dtype=pl.String),
            employ_thousands=pl.col("value_thousands") - pl.col("prior_thousands"),
        )
    )
    table = (
        _total_nonfarm_closing(table_changes, "SA")
        .filter(pl.col("value_thousands").is_not_null())
        .select(
            "reference_month",
            "release_month",
            "release_stage",
            table_thousands="value_thousands",
        )
    )
    return (
        changes.join(
            table, on=["reference_month", "release_stage", "release_month"], how="inner"
        )
        .filter(pl.col("employ_thousands") != pl.col("table_thousands"))
        .select(
            "reference_month",
            "release_stage",
            "vintage_id",
            "employ_thousands",
            "table_thousands",
        )
        .sort("reference_month", "release_stage")
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_pilot_assembly.py -q`
Expected: `12 passed`. The first test takes about 40 s, because it builds the cached operator and vintage frames.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format src/ces_revisions/pilot/__init__.py src/ces_revisions/pilot/panel.py src/ces_revisions/pilot/assembly.py tests/test_pilot_assembly.py
uv run ruff check src/ces_revisions/pilot/__init__.py src/ces_revisions/pilot/panel.py src/ces_revisions/pilot/assembly.py tests/test_pilot_assembly.py
git add src/ces_revisions/pilot/__init__.py src/ces_revisions/pilot/panel.py src/ces_revisions/pilot/assembly.py tests/test_pilot_assembly.py
git commit -m "Assemble the Stage 6 aggregate observation panel"
```

### Task 3: Seasonal layouts and the synthetic test panels

**Files:**
- Create: `src/ces_revisions/pilot/layout.py`
- Create: `tests/pilot_synthetic.py`
- Test: `tests/test_pilot_layout.py`

**Interfaces:**
- Consumes: `COLUMNS`, `AggregatePanel`, `CELL_SCHEMA`, and the other `panel.py` constants (Task 2); `tests/dense_reference.simulate`.
- Produces:
  - `Layout` (frozen dataclass: `name`, `harmonics: int | None`, `state_names: tuple[str, ...]`; properties `state_dim`, `seasonal_dim`), `LAYOUTS: dict[str, Layout]` with keys `"dummy"`, `"trig"`, `"trig4"`.
  - `transition_matrix(layout) -> np.ndarray` (n, n), `shock_loading(layout) -> np.ndarray` (n, k), `observation_matrix(layout) -> np.ndarray` (4, n), `seasonal_selector(layout) -> np.ndarray` (n,), `shock_names(layout) -> tuple[str, ...]`, `seasonal_variance_factor(layout) -> float`, `SEASONAL_PRIOR_SCALE: dict[str, float]`, and `PERIOD`, `TREND_STATES`, `NEWS_STATES`, `TREND_SHOCKS`, `NEWS_SHOCKS`, `DUMMY_SEASONAL_PRIOR_SCALE`.
  - `tests/pilot_synthetic.py`: `STATE_NAMES`, `TRUTH` (one value per sampled site), the synthetic panel's layout constants (`NUM_STEPS`, `CRISIS_STEPS`, `MISSING_STEP`, `CARRIER_STEPS`, `MISSING_CELLS`, `ANCHOR_OFFSETS`, and the `CRISIS_*` constants of `crisis_panel`), `reference_ssm(params, panel, layout_name)` (an oracle written from state names alone, independent of `layout.py` and `model.py`), `synthetic_panel(...)`, `shell_panel(...)`, and `crisis_panel(layout_name)`.

The system matrices hold no parameter. Every sampled scalar enters only a covariance or an offset, which is why plan 1's `A_t`/`Z_t` gradient-check item stays open (planning choice 8).

- [ ] **Step 1: Write the synthetic-panel helper**

Create `tests/pilot_synthetic.py`:

```python
"""A small synthetic aggregate panel for the Stage 6 pilot, and an independent model.

``reference_ssm`` writes the pilot's state-space arrays entry by entry from the model's
written definition (state names, transitions, shock covariances, observation rows, and
Req 7's noise recursion), sharing no code with ``ces_revisions.pilot.layout`` or
``ces_revisions.pilot.model``. Tests compare ``pilot_ssm`` with it array by array, and run
the dense oracle on it.

``synthetic_panel`` packs every feature the pilot must handle into a short window: a
pre-2003 to post-2003 boundary, two crisis steps, March anchors with nonzero offsets, a
fully missing step, F-only steps like a benchmark carrier's, scattered missing cells, and a
ragged edge. Its values are drawn from ``reference_ssm`` by the dense oracle's simulator,
not by the pilot's own ``simulate_panel``. The crisis flags are set by step, not by date:
the real crisis (2020) and the 2003 boundary cannot share a short window.
"""

import math
from datetime import date

import numpy as np
import polars as pl
from dense_reference import simulate

from ces_revisions.pilot.panel import (
    ANCHOR_COLUMN,
    ANCHOR_ROUNDING_VARIANCE,
    CELL_SCHEMA,
    CLOSING_ROUNDING_VARIANCE,
    COLUMNS,
    REGIME_SHIFT_MONTH,
    AggregatePanel,
    benchmark_year_of,
)

SEED = sum(map(ord, "ces-revisions-stage-6-synthetic-panel"))
FIRST_MONTH = date(2002, 1, 1)  # steps 0-15 are before REGIME_SHIFT_MONTH (2003-05)
NUM_STEPS = 30
CRISIS_STEPS = (20, 21)
MISSING_STEP = 9  # no cell observed at all
CARRIER_STEPS = (5, 17)  # F observed, S and T masked as a benchmark carrier's would be
MISSING_CELLS = ((12, 2), (23, 0))  # (step, column) gaps
# The anchor offset at each successive March: earlier reconstructions add up.
ANCHOR_OFFSETS = (0.0, -312.0, 175.0, 268.0, -41.0)

# A draw away from the prior medians, so a swapped or mis-scaled term shows up.
TRUTH = {
    "sigma_level": 60.0,
    "sigma_slope": 8.0,
    "sigma_seasonal": 30.0,
    "sigma_news_s": 45.0,
    "sigma_news_t": 30.0,
    "sigma_news_annual": 350.0,
    "noise_f_raw": 0.3,
    "noise_s_raw": -0.2,
    "noise_t_raw": 0.1,
    "rho_noise": 0.25,
    "mean_news_s": 5.0,
    "mean_news_t": -3.0,
    "mean_news_annual": 40.0,
    "sigma_anchor": 80.0,
    "kappa_crisis": math.log(30.0),
    "shift_log_scale": 0.3,
    "shift_mean": -6.0,
}

_TREND = ("level", "slope", "lagged_value")
_NEWS = ("news_s", "news_t", "news_post")
_PAIRS = {"trig": 5, "trig4": 4}  # cos/sin harmonic pairs j = 1..pairs
STATE_NAMES = {
    "dummy": (*_TREND, *(f"seasonal_{k}" for k in range(11)), *_NEWS),
    "trig": (
        *_TREND,
        *(f"harmonic_{j}_{part}" for j in range(1, 6) for part in ("cos", "sin")),
        "harmonic_6",
        *_NEWS,
    ),
    "trig4": (
        *_TREND,
        *(f"harmonic_{j}_{part}" for j in range(1, 5) for part in ("cos", "sin")),
        *_NEWS,
    ),
}


def _months(first: date, num_steps: int) -> tuple[date, ...]:
    index = first.year * 12 + first.month - 1
    return tuple(
        date((index + t) // 12, (index + t) % 12 + 1, 1) for t in range(num_steps)
    )


def _gamma_states(layout_name: str) -> tuple[str, ...]:
    """The seasonal states whose sum is gamma_t."""
    if layout_name == "dummy":
        return ("seasonal_0",)
    names = STATE_NAMES[layout_name]
    return tuple(
        name for name in names if name.endswith("_cos") or name == "harmonic_6"
    )


def reference_ssm(params: dict, panel: AggregatePanel, layout_name: str) -> dict:
    """The pilot's arrays, written entry by entry, as a dense_reference model dict."""
    names = STATE_NAMES[layout_name]
    at = {name: k for k, name in enumerate(names)}
    n, num_steps = len(names), panel.num_steps
    seasonal = [name for name in names if name not in (*_TREND, *_NEWS)]

    a = np.zeros((n, n))
    a[at["level"], at["level"]] = a[at["level"], at["slope"]] = 1.0
    a[at["slope"], at["slope"]] = 1.0
    a[at["lagged_value"], at["level"]] = 1.0
    for name in _gamma_states(layout_name):
        a[at["lagged_value"], at[name]] = 1.0
    if layout_name == "dummy":
        for k in range(11):
            a[at["seasonal_0"], at[f"seasonal_{k}"]] = -1.0
        for k in range(1, 11):
            a[at[f"seasonal_{k}"], at[f"seasonal_{k - 1}"]] = 1.0
    else:
        for j in range(1, _PAIRS[layout_name] + 1):
            cos, sin = f"harmonic_{j}_cos", f"harmonic_{j}_sin"
            angle = 2 * math.pi * j / 12
            a[at[cos], at[cos]] = a[at[sin], at[sin]] = math.cos(angle)
            a[at[cos], at[sin]] = math.sin(angle)
            a[at[sin], at[cos]] = -math.sin(angle)
        if layout_name == "trig":
            a[at["harmonic_6"], at["harmonic_6"]] = -1.0

    offset = np.zeros((num_steps, n))
    shock_cov = np.zeros((num_steps, n, n))
    obs_cov = np.zeros((num_steps, 4, 4))
    noise_sd = {
        "f": 0.5 * params["sigma_news_s"] * math.exp(0.7 * params["noise_f_raw"]),
        "s": 0.5 * params["sigma_news_t"] * math.exp(0.7 * params["noise_s_raw"]),
        "t": 0.5 * params["sigma_news_t"] * math.exp(0.7 * params["noise_t_raw"]),
    }
    rho = params["rho_noise"]
    for t in range(num_steps):
        pre, crisis = float(panel.pre_2003[t]), float(panel.crisis[t])
        crisis_scale = math.exp(params["kappa_crisis"] * crisis)
        shift_scale = math.exp(params["shift_log_scale"] * pre)
        news_mean = {
            "news_s": params["mean_news_s"] + params["shift_mean"] * pre,
            "news_t": params["mean_news_t"] + params["shift_mean"] * pre,
            "news_post": params["mean_news_annual"] / 12,
        }
        news_var = {
            "news_s": (params["sigma_news_s"] * shift_scale * crisis_scale) ** 2,
            "news_t": (params["sigma_news_t"] * shift_scale * crisis_scale) ** 2,
            "news_post": params["sigma_news_annual"] ** 2 / 12,
        }
        level_var = (params["sigma_level"] * crisis_scale) ** 2
        offset[t, at["level"]] = sum(news_mean.values())
        shock_cov[t, at["level"], at["level"]] = level_var + sum(news_var.values())
        for name in _NEWS:
            offset[t, at[name]] = news_mean[name]
            shock_cov[t, at[name], at[name]] = news_var[name]
            shock_cov[t, at[name], at["level"]] = news_var[name]
            shock_cov[t, at["level"], at[name]] = news_var[name]
        shock_cov[t, at["slope"], at["slope"]] = params["sigma_slope"] ** 2
        shocked = ["seasonal_0"] if layout_name == "dummy" else seasonal
        for name in shocked:
            shock_cov[t, at[name], at[name]] = params["sigma_seasonal"] ** 2

        var_f, var_s, var_t = ((noise_sd[k] * shift_scale) ** 2 for k in "fst")
        e_s_var = rho**2 * var_f + var_s
        obs_cov[t, :3, :3] = [
            [var_f, rho * var_f, rho**2 * var_f],
            [rho * var_f, e_s_var, rho * e_s_var],
            [rho**2 * var_f, rho * e_s_var, rho**2 * e_s_var + var_t],
        ]
        obs_cov[t, :3, :3] += CLOSING_ROUNDING_VARIANCE * np.eye(3)
        obs_cov[t, 3, 3] = params["sigma_anchor"] ** 2 + ANCHOR_ROUNDING_VARIANCE

    z = np.zeros((4, n))
    for row in range(4):
        z[row, at["level"]] = 1.0
        for name in _gamma_states(layout_name):
            z[row, at[name]] = 1.0
    for row, missed in enumerate((_NEWS, _NEWS[1:], _NEWS[2:])):
        z[row, at["lagged_value"]] = -1.0
        for name in missed:
            z[row, at[name]] = -1.0
    obs_offset = np.zeros((num_steps, 4))
    obs_offset[:, 3] = panel.anchor_offset

    mean0 = {"level": 89_408.0, "slope": 150.0, "lagged_value": 89_408.0}
    sd0 = {"level": 5_000.0, "slope": 300.0, "lagged_value": 5_000.0}
    sd0 |= dict.fromkeys(seasonal, 2_000.0) | dict.fromkeys(_NEWS, 100.0)
    return {
        "initial_mean": np.array([mean0.get(name, 0.0) for name in names]),
        "initial_cov": np.diag([sd0[name] ** 2 for name in names]),
        "transition_matrix": np.broadcast_to(a, (num_steps, n, n)).copy(),
        "transition_offset": offset,
        "transition_cov": shock_cov,
        "observation_matrix": np.broadcast_to(z, (num_steps, 4, n)).copy(),
        "observation_offset": obs_offset,
        "observation_cov": obs_cov,
    }


def shell_panel(
    num_steps: int = NUM_STEPS,
    first_month: date = FIRST_MONTH,
    crisis_steps: tuple[int, ...] = CRISIS_STEPS,
) -> AggregatePanel:
    """A panel with every flag set and 0.0 placeholders in each cell that will be observed.

    Anchors sit at every March, with ANCHOR_OFFSETS in turn; the closing cells are all
    zero, so a simulator with ``complete=True`` fills them.
    """
    months = _months(first_month, num_steps)
    march = np.array([month.month == 3 for month in months])
    anchor_offset = np.zeros(num_steps)
    anchor_offset[march] = ANCHOR_OFFSETS[: march.sum()]
    y = np.zeros((num_steps, len(COLUMNS)))
    y[~march, ANCHOR_COLUMN] = np.nan
    return _panel(months, y, anchor_offset, crisis_steps)


def synthetic_panel(
    layout_name: str = "dummy",
    num_steps: int = NUM_STEPS,
    *,
    params: dict = TRUTH,
    seed: int = SEED,
) -> AggregatePanel:
    """A panel drawn from ``reference_ssm`` and then masked with every gap pattern."""
    shell = shell_panel(num_steps)
    draws = simulate(
        reference_ssm(params, shell, layout_name), np.random.default_rng(seed)
    )
    y = np.where(np.isnan(shell.y), np.nan, draws)
    y[MISSING_STEP] = np.nan
    for step in CARRIER_STEPS:
        y[step, 1:ANCHOR_COLUMN] = np.nan
    for step, column in MISSING_CELLS:
        y[step, column] = np.nan
    y[-1, 1:ANCHOR_COLUMN] = np.nan  # ragged edge: the latest month has only F
    y[-2, 2] = np.nan  # and the month before it has no T yet
    return _panel(shell.reference_months, y, shell.anchor_offset, CRISIS_STEPS)


# A window around the 2020 crisis for the April 2020 check: 2019-07 through 2020-12, with
# the crisis flags on March-June 2020 by date and an anchor at March 2020.
CRISIS_FIRST_MONTH = date(2019, 7, 1)
CRISIS_NUM_STEPS = 18
CRISIS_WINDOW_STEPS = (8, 9, 10, 11)


def crisis_panel(
    layout_name: str = "dummy", *, params: dict = TRUTH, seed: int = SEED
) -> AggregatePanel:
    """A fully observed panel over CRISIS_FIRST_MONTH's 18 months, from ``reference_ssm``.

    The latest month has only its F print, like the real panel's ragged edge.
    """
    shell = shell_panel(CRISIS_NUM_STEPS, CRISIS_FIRST_MONTH, CRISIS_WINDOW_STEPS)
    draws = simulate(
        reference_ssm(params, shell, layout_name), np.random.default_rng(seed)
    )
    y = np.where(np.isnan(shell.y), np.nan, draws)
    y[-1, 1:ANCHOR_COLUMN] = np.nan
    return _panel(shell.reference_months, y, shell.anchor_offset, CRISIS_WINDOW_STEPS)


def _panel(months, y, anchor_offset, crisis_steps) -> AggregatePanel:
    num_steps = len(months)
    crisis = np.zeros(num_steps, dtype=bool)
    crisis[list(crisis_steps)] = True
    rows = [
        {
            "reference_month": month,
            "column": column,
            "source": "synthetic",
            "value_thousands": None if np.isnan(y[t, k]) else float(y[t, k]),
            "status": "synthetic",
            "mask_reason": "synthetic_gap" if np.isnan(y[t, k]) else None,
            "published_date": None,
        }
        for t, month in enumerate(months)
        for k, column in enumerate(COLUMNS)
    ]
    return AggregatePanel(
        reference_months=months,
        y=y.astype(np.float64),
        anchor_offset=anchor_offset.astype(np.float64),
        pre_2003=np.array([month < REGIME_SHIFT_MONTH for month in months]),
        crisis=crisis,
        benchmark_year=np.array([benchmark_year_of(month) for month in months]),
        cells=pl.DataFrame(rows, schema=CELL_SCHEMA),
    )
```

- [ ] **Step 2: Write the failing layout tests**

Create `tests/test_pilot_layout.py`:

```python
"""Tests of the pilot's state layouts: transitions, shock loadings, observation rows."""

import math

import numpy as np
import pytest
from pilot_synthetic import STATE_NAMES

from ces_revisions.pilot.layout import (
    LAYOUTS,
    SEASONAL_PRIOR_SCALE,
    observation_matrix,
    seasonal_selector,
    seasonal_variance_factor,
    shock_loading,
    shock_names,
    transition_matrix,
)
from ces_revisions.pilot.panel import COLUMNS

SEED = sum(map(ord, "ces-revisions-stage-6-pilot-layout"))
NEWS = ("news_s", "news_t", "news_post")


@pytest.fixture(params=sorted(LAYOUTS))
def layout(request):
    return LAYOUTS[request.param]


def _at(layout, name):
    return layout.state_names.index(name)


def _seasonal_block(layout):
    first = len(("level", "slope", "lagged_value"))
    return slice(first, first + layout.seasonal_dim)


def test_layouts_have_the_contract_state_order_and_sizes():
    sizes = {
        name: (layout.state_dim, layout.seasonal_dim, layout.harmonics)
        for name, layout in LAYOUTS.items()
    }

    assert sizes == {"dummy": (17, 11, None), "trig": (17, 11, 6), "trig4": (14, 8, 4)}
    for name, layout in LAYOUTS.items():
        assert layout.name == name
        assert layout.state_names == STATE_NAMES[name]


def test_seasonal_paths_sum_to_zero_over_any_year_and_repeat_every_twelve_months(
    layout,
):
    transition = transition_matrix(layout)
    selector = seasonal_selector(layout)
    state = np.random.default_rng(SEED).normal(0.0, 1_000.0, layout.state_dim)
    gammas = []
    for _ in range(48):
        state = transition @ state
        gammas.append(selector @ state)
    gammas = np.array(gammas)
    yearly = np.convolve(gammas, np.ones(12), mode="valid")

    np.testing.assert_allclose(yearly, 0.0, atol=1e-8)
    np.testing.assert_allclose(gammas[12:], gammas[:-12], atol=1e-8)
    assert np.ptp(gammas) > 100.0  # a real seasonal pattern, not a vanishing one


def test_dummy_seasonal_is_minus_the_sum_of_the_last_eleven_months():
    layout = LAYOUTS["dummy"]
    block = transition_matrix(layout)[_seasonal_block(layout), _seasonal_block(layout)]
    expected = np.zeros((11, 11))
    expected[0, :] = -1.0
    expected[np.arange(1, 11), np.arange(10)] = 1.0

    np.testing.assert_array_equal(block, expected)


@pytest.mark.parametrize("name", ["trig", "trig4"])
def test_trigonometric_harmonics_rotate_with_period_twelve(name):
    layout = LAYOUTS[name]
    block = transition_matrix(layout)[_seasonal_block(layout), _seasonal_block(layout)]
    powers = [np.linalg.matrix_power(block, k) for k in range(1, 13)]

    np.testing.assert_allclose(powers[-1], np.eye(layout.seasonal_dim), atol=1e-12)
    assert all(
        not np.allclose(power, np.eye(layout.seasonal_dim)) for power in powers[:-1]
    )
    for j in range(1, min(layout.harmonics, 5) + 1):
        pair = block[2 * j - 2 : 2 * j, 2 * j - 2 : 2 * j]
        angle = 2 * math.pi * j / 12
        rotation = [
            [math.cos(angle), math.sin(angle)],
            [-math.sin(angle), math.cos(angle)],
        ]
        np.testing.assert_allclose(pair, rotation, rtol=0, atol=1e-15)
    if name == "trig":
        assert block[-1, -1] == -1.0
        assert np.count_nonzero(block[-1]) == np.count_nonzero(block[:, -1]) == 1


def test_selector_picks_gamma_from_the_seasonal_block(layout):
    picked = {
        name
        for name, weight in zip(
            layout.state_names, seasonal_selector(layout), strict=True
        )
        if weight
    }
    expected = (
        {"seasonal_0"}
        if layout.name == "dummy"
        else {name for name in layout.seasonal_names if not name.endswith("_sin")}
    )

    assert picked == expected
    assert set(seasonal_selector(layout)) == {0.0, 1.0}
    if layout.name == "trig":
        assert "harmonic_6" in picked


def test_transition_carries_the_trend_and_last_months_value(layout):
    transition = transition_matrix(layout)
    level, slope, lagged = (
        _at(layout, name) for name in ("level", "slope", "lagged_value")
    )
    level_row = np.zeros(layout.state_dim)
    level_row[[level, slope]] = 1.0
    lagged_row = seasonal_selector(layout).copy()
    lagged_row[level] = 1.0
    news = [_at(layout, name) for name in NEWS]

    np.testing.assert_array_equal(transition[level], level_row)
    np.testing.assert_array_equal(transition[slope], np.eye(layout.state_dim)[slope])
    np.testing.assert_array_equal(transition[lagged], lagged_row)
    assert not transition[news].any()  # news is white noise
    assert not transition[:, [lagged, *news]].any()  # nothing reads them a month later


def test_shock_loading_puts_each_news_shock_in_its_state_and_the_level(layout):
    loading = shock_loading(layout)
    names = shock_names(layout)
    seasonal_shocks = 1 if layout.name == "dummy" else layout.seasonal_dim
    loaded_by = {
        state: {names[k] for k in np.flatnonzero(loading[_at(layout, state)])}
        for state in layout.state_names
    }

    assert loading.shape == (layout.state_dim, 5 + seasonal_shocks)
    assert set(loading.ravel()) == {0.0, 1.0}
    assert loaded_by["level"] == {"eta_level", "eps_s", "eps_t", "eps_post"}
    assert loaded_by["slope"] == {"eta_slope"}
    assert loaded_by["lagged_value"] == set()
    assert loaded_by["news_s"] == {"eps_s"}
    assert loaded_by["news_t"] == {"eps_t"}
    assert loaded_by["news_post"] == {"eps_post"}
    seasonal = loading[_seasonal_block(layout), 2 : 2 + seasonal_shocks]
    if layout.name == "dummy":
        np.testing.assert_array_equal(seasonal, np.eye(11)[:, :1])  # slot 0 only
    else:
        np.testing.assert_array_equal(seasonal, np.eye(layout.seasonal_dim))


def test_observation_rows_are_the_change_less_the_news_still_to_come(layout):
    state = np.random.default_rng(SEED).normal(0.0, 100.0, layout.state_dim)
    value = state[_at(layout, "level")] + seasonal_selector(layout) @ state
    change = value - state[_at(layout, "lagged_value")]
    news_s, news_t, news_post = (state[_at(layout, name)] for name in NEWS)
    expected = {
        "F": change - news_s - news_t - news_post,
        "S": change - news_t - news_post,
        "T": change - news_post,
        "anchor": value,
    }

    np.testing.assert_allclose(
        observation_matrix(layout) @ state,
        [expected[column] for column in COLUMNS],
        rtol=1e-14,
    )


def _year_over_year_variance(layout):
    """Var(gamma_t - gamma_{t-12}) per unit shock variance, by a covariance recursion.

    The seasonal state at t - 12 is held fixed (zero covariance) and twelve months of
    unit shocks are propagated; C^12 = I makes the fixed part cancel in the difference.
    """
    block = transition_matrix(layout)[_seasonal_block(layout), _seasonal_block(layout)]
    loading = shock_loading(layout)[_seasonal_block(layout)]
    covariance = np.zeros_like(block)
    for _ in range(12):
        covariance = block @ covariance @ block.T + loading @ loading.T
    selector = seasonal_selector(layout)[_seasonal_block(layout)]
    return selector @ covariance @ selector


def test_seasonal_prior_scales_match_the_dummy_year_over_year_variance():
    factors = {
        name: seasonal_variance_factor(layout) for name, layout in LAYOUTS.items()
    }
    recursion = {
        name: _year_over_year_variance(layout) for name, layout in LAYOUTS.items()
    }

    assert factors == pytest.approx(
        {"dummy": 2.0, "trig": 72.0, "trig4": 48.0}, rel=1e-12
    )
    assert recursion == pytest.approx(factors, rel=1e-12)
    assert SEASONAL_PRIOR_SCALE["dummy"] == 50.0
    assert SEASONAL_PRIOR_SCALE["trig"] == pytest.approx(50.0 / 6.0, rel=1e-14)
    assert SEASONAL_PRIOR_SCALE["trig4"] == pytest.approx(
        50.0 / math.sqrt(24.0), rel=1e-14
    )
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/test_pilot_layout.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'ces_revisions.pilot.layout'`.

- [ ] **Step 4: Write the layouts**

Create `src/ces_revisions/pilot/layout.py`:

```python
"""The pilot's state layouts: the state vector, its transition, and its observation rows.

The state is ``level, slope, lagged_value, <seasonal block>, news_s, news_t, news_post``.
``level`` and ``slope`` form a local linear trend; ``lagged_value`` copies last month's
value ``x_{t-1} = level_{t-1} + gamma_{t-1}``, so a same-release change is a difference of
two states; and the three news states are white noise holding this month's second-print,
third-print, and post-third-print news, the same shocks that move ``level``
(Jacobs and van Norden's news inside the truth's innovation). The seasonal block is the
only part that differs between layouts: a dummy seasonal ("dummy"), a full trigonometric
seasonal ("trig"), or a trigonometric seasonal with four harmonics ("trig4").

All matrices are NumPy float64 and time-invariant; ``model.pilot_ssm`` broadcasts them over
the reference months and adds the time-varying shock and noise scales.
"""

import math
from dataclasses import dataclass

import numpy as np

from ces_revisions.pilot.panel import COLUMNS

PERIOD = 12  # months in a seasonal cycle
TREND_STATES = ("level", "slope", "lagged_value")
NEWS_STATES = ("news_s", "news_t", "news_post")
TREND_SHOCKS = ("eta_level", "eta_slope")
NEWS_SHOCKS = ("eps_s", "eps_t", "eps_post")
# Req 15's HalfNormal scale for the dummy seasonal shock SD.
DUMMY_SEASONAL_PRIOR_SCALE = 50.0


@dataclass(frozen=True)
class Layout:
    """One seasonal layout of the pilot's state vector.

    ``harmonics`` is None for the dummy seasonal and the highest harmonic for the
    trigonometric layouts (6 includes the Nyquist harmonic, a single state).
    """

    name: str
    harmonics: int | None
    state_names: tuple[str, ...]

    @property
    def state_dim(self) -> int:
        return len(self.state_names)

    @property
    def seasonal_dim(self) -> int:
        return self.state_dim - len(TREND_STATES) - len(NEWS_STATES)

    @property
    def seasonal_names(self) -> tuple[str, ...]:
        first = len(TREND_STATES)
        return self.state_names[first : first + self.seasonal_dim]


def _trigonometric_names(harmonics: int) -> tuple[str, ...]:
    pairs = min(harmonics, PERIOD // 2 - 1)
    names = [
        f"harmonic_{j}_{part}" for j in range(1, pairs + 1) for part in ("cos", "sin")
    ]
    if harmonics == PERIOD // 2:
        names.append(f"harmonic_{PERIOD // 2}")
    return tuple(names)


def _layout(name: str, harmonics: int | None, seasonal: tuple[str, ...]) -> Layout:
    return Layout(name, harmonics, (*TREND_STATES, *seasonal, *NEWS_STATES))


LAYOUTS = {
    "dummy": _layout("dummy", None, tuple(f"seasonal_{k}" for k in range(PERIOD - 1))),
    "trig": _layout("trig", PERIOD // 2, _trigonometric_names(PERIOD // 2)),
    "trig4": _layout("trig4", 4, _trigonometric_names(4)),
}


def _index(layout: Layout, name: str) -> int:
    return layout.state_names.index(name)


def _seasonal_slice(layout: Layout) -> slice:
    first = len(TREND_STATES)
    return slice(first, first + layout.seasonal_dim)


def _seasonal_transition(layout: Layout) -> np.ndarray:
    """The seasonal block's own transition, (S, S)."""
    size = layout.seasonal_dim
    block = np.zeros((size, size))
    if layout.harmonics is None:
        # gamma_t = -(gamma_{t-1} + ... + gamma_{t-11}); the other slots shift down by one.
        block[0, :] = -1.0
        block[np.arange(1, size), np.arange(size - 1)] = 1.0
        return block
    for j in range(1, min(layout.harmonics, PERIOD // 2 - 1) + 1):
        frequency = 2.0 * math.pi * j / PERIOD
        cos, sin = math.cos(frequency), math.sin(frequency)
        first = 2 * (j - 1)
        block[first : first + 2, first : first + 2] = [[cos, sin], [-sin, cos]]
    if layout.harmonics == PERIOD // 2:
        block[-1, -1] = -1.0  # the Nyquist harmonic alternates sign every month
    return block


def _seasonal_selector_block(layout: Layout) -> np.ndarray:
    """The (S,) row that reads gamma_t off the seasonal block."""
    if layout.harmonics is None:
        selector = np.zeros(layout.seasonal_dim)
        selector[0] = 1.0
        return selector
    return np.array(
        [0.0 if name.endswith("_sin") else 1.0 for name in layout.seasonal_names]
    )


def _seasonal_loading_block(layout: Layout) -> np.ndarray:
    """The (S, number of seasonal shocks) loading of the seasonal shocks."""
    if layout.harmonics is None:
        loading = np.zeros((layout.seasonal_dim, 1))
        loading[0, 0] = 1.0  # one shock, on the current month's slot
        return loading
    return np.eye(layout.seasonal_dim)  # every harmonic state has its own shock


def seasonal_selector(layout: Layout) -> np.ndarray:
    """The (n,) row over the whole state that picks gamma_t."""
    selector = np.zeros(layout.state_dim)
    selector[_seasonal_slice(layout)] = _seasonal_selector_block(layout)
    return selector


def shock_names(layout: Layout) -> tuple[str, ...]:
    """The shock vector loaded by ``shock_loading``, in column order."""
    count = _seasonal_loading_block(layout).shape[1]
    seasonal = tuple(f"eta_seasonal_{k}" for k in range(count))
    return (*TREND_SHOCKS, *seasonal, *NEWS_SHOCKS)


def transition_matrix(layout: Layout) -> np.ndarray:
    """The (n, n) transition A: x_t = A x_{t-1} + c_t + L w_t."""
    level, slope, lagged = (_index(layout, name) for name in TREND_STATES)
    seasonal = _seasonal_slice(layout)
    matrix = np.zeros((layout.state_dim, layout.state_dim))
    matrix[level, [level, slope]] = 1.0  # level_t = level_{t-1} + slope_{t-1} + shocks
    matrix[slope, slope] = 1.0
    matrix[lagged, level] = 1.0  # lagged_value_t = level_{t-1} + gamma_{t-1}
    matrix[lagged, seasonal] = _seasonal_selector_block(layout)
    matrix[seasonal, seasonal] = _seasonal_transition(layout)
    # The news rows stay zero: each month's news is a fresh shock.
    return matrix


def shock_loading(layout: Layout) -> np.ndarray:
    """The (n, k) loading L of the shocks in ``shock_names`` order onto the states."""
    names = shock_names(layout)
    level, slope = _index(layout, "level"), _index(layout, "slope")
    loading = np.zeros((layout.state_dim, len(names)))
    loading[level, names.index("eta_level")] = 1.0
    loading[slope, names.index("eta_slope")] = 1.0
    seasonal_block = _seasonal_loading_block(layout)
    first_shock = len(TREND_SHOCKS)
    loading[
        _seasonal_slice(layout), first_shock : first_shock + seasonal_block.shape[1]
    ] = seasonal_block
    for state, shock in zip(NEWS_STATES, NEWS_SHOCKS, strict=True):
        # Each news shock is both this month's news state and part of the level's move.
        loading[_index(layout, state), names.index(shock)] = 1.0
        loading[level, names.index(shock)] = 1.0
    return loading


def observation_matrix(layout: Layout) -> np.ndarray:
    """The (4, n) rows of the F, S, T same-release changes and the March anchor.

    With dx_t = level_t + gamma_t - lagged_value_t, a print is the true change less the
    news still to come: F misses all three news shocks, S misses T's and post-T's, and T
    misses post-T's. The anchor observes the level x_t itself.
    """
    gamma = seasonal_selector(layout)
    value = gamma.copy()
    value[_index(layout, "level")] = 1.0
    change = value.copy()
    change[_index(layout, "lagged_value")] = -1.0
    news = {
        state: np.eye(layout.state_dim)[_index(layout, state)] for state in NEWS_STATES
    }
    rows = {
        "F": change - news["news_s"] - news["news_t"] - news["news_post"],
        "S": change - news["news_t"] - news["news_post"],
        "T": change - news["news_post"],
        "anchor": value,
    }
    return np.stack([rows[column] for column in COLUMNS])


def seasonal_variance_factor(layout: Layout) -> float:
    """Var(gamma_t - gamma_{t-12}) per unit seasonal shock variance, exactly.

    Every layout's seasonal block C satisfies C^12 = I (its eigenvalues are twelfth roots
    of unity), so gamma_t - gamma_{t-12} = s' sum_{h=0}^{11} C^h L w_{t-h} whatever the
    state at t - 12, and its variance per unit shock variance is sum_h ||s' C^h L||^2.
    Dummy: gamma_t - gamma_{t-12} = w_t - w_{t-1}, so 2. Trigonometric: each harmonic
    adds 12 (a rotation keeps the shock's norm for 12 months), so 72 for six harmonics
    and 48 for four.
    """
    block = _seasonal_transition(layout)
    if not np.allclose(
        np.linalg.matrix_power(block, PERIOD), np.eye(layout.seasonal_dim)
    ):
        raise ValueError(f"layout {layout.name!r} is not periodic with period {PERIOD}")
    selector = _seasonal_selector_block(layout)
    loading = _seasonal_loading_block(layout)
    return float(
        sum(
            np.sum((selector @ np.linalg.matrix_power(block, lag) @ loading) ** 2)
            for lag in range(PERIOD)
        )
    )


# Each layout's sigma_seasonal prior implies the dummy's Var(gamma_t - gamma_{t-12}) per
# unit shock: scale = 50 * sqrt(v(dummy) / v(layout)), with v(dummy) = 2, v(trig) = 72,
# and v(trig4) = 48 (``seasonal_variance_factor``), so trig = 50 / 6 and
# trig4 = 50 / sqrt(24).
SEASONAL_PRIOR_SCALE = {
    name: DUMMY_SEASONAL_PRIOR_SCALE
    * math.sqrt(
        seasonal_variance_factor(LAYOUTS["dummy"]) / seasonal_variance_factor(layout)
    )
    for name, layout in LAYOUTS.items()
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_pilot_layout.py -q`
Expected: `20 passed`.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format src/ces_revisions/pilot/layout.py tests/pilot_synthetic.py tests/test_pilot_layout.py
uv run ruff check src/ces_revisions/pilot/layout.py tests/pilot_synthetic.py tests/test_pilot_layout.py
git add src/ces_revisions/pilot/layout.py tests/pilot_synthetic.py tests/test_pilot_layout.py
git commit -m "Add the pilot's three seasonal layouts and synthetic test panels"
```

### Task 4: The pilot model

**Files:**
- Create: `src/ces_revisions/pilot/model.py`
- Test: `tests/test_pilot_model.py`

**Interfaces:**
- Consumes: `LinearGaussianSSM`, `FilterResult`, `kalman_filter`, `kalman_factor` (`kalman.py`); the layout functions and constants (Task 3); `AggregatePanel` and the rounding variances (Task 2); `tests/dense_reference.dense_reference`; `tests/pilot_synthetic` (Task 3).
- Produces:
  - `SAMPLED_SITES: tuple[str, ...]` (17 names), `INITIAL_MEAN` and `INITIAL_SD` (dicts by state name), `MONTHS_PER_YEAR`, `NOISE_CENTER`, `NOISE_LOG_SCALE`, `NOISE_RAW_SITES` (the three standard normal noise sites), `TARGET_ACCEPT_PROB = 0.95`.
  - `prior_distributions(layout) -> dict[str, dist.Distribution]`, `derived(params: dict) -> dict` (scalars or arrays, NumPy or JAX), `pilot_ssm(params, panel, layout) -> LinearGaussianSSM`, `pilot_model(panel, layout) -> FilterResult`, `sample_prior(key, num_samples, layout) -> dict[str, jax.Array]`, `draw_distributions(priors, key, num_samples) -> dict[str, jax.Array]`, `step_log_likelihood(params, panel, layout) -> jax.Array` (T,).

This task carries Stage 6's likelihood exit test: the model's trace holds exactly the 17 scalar sites and one factor equal to the filter's log likelihood, so no latent state is a sampled site, and the likelihood includes irregular anchor rows and NaN cells. The dense-parity tests compare the engine with the Stage 1 dense oracle at `rtol=1e-8`, on a short synthetic panel and on one at real magnitude.

- [ ] **Step 1: Write the failing model tests**

Create `tests/test_pilot_model.py`:

```python
"""Tests of the pilot model: its arrays, the dense oracle, its trace, and its flags."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from dense_reference import dense_reference
from numpyro import handlers
from numpyro.infer.util import log_density
from pilot_synthetic import (
    ANCHOR_OFFSETS,
    CARRIER_STEPS,
    CRISIS_STEPS,
    MISSING_STEP,
    STATE_NAMES,
    TRUTH,
    reference_ssm,
    synthetic_panel,
)

from ces_revisions.kalman import (
    FilterResult,
    LinearGaussianSSM,
    SmootherResult,
    kalman_filter,
    kalman_smoother,
)
from ces_revisions.pilot.layout import LAYOUTS, SEASONAL_PRIOR_SCALE
from ces_revisions.pilot.model import (
    INITIAL_MEAN,
    INITIAL_SD,
    SAMPLED_SITES,
    derived,
    pilot_model,
    pilot_ssm,
    prior_distributions,
    sample_prior,
    step_log_likelihood,
)
from ces_revisions.pilot.panel import ANCHOR_COLUMN

SEED = sum(map(ord, "ces-revisions-stage-6-pilot-model"))
LAYOUT_NAMES = sorted(LAYOUTS)
REAL_MAGNITUDE_STEPS = 60
NUM_PRIOR_DRAWS = 3
# Every field must agree with the dense oracle to 1e-8 of its natural scale: a
# covariance entry to 1e-8 sqrt(C_ii C_jj) (rtol 1e-8 on the diagonal, and an exact zero
# held to the same scale), a mean to 1e-8 (|m_i| + sqrt(C_ii)), and the log likelihood
# per step to rtol 1e-8 with a 1e-8 nat floor for the empty step.
DENSE_TOLERANCE = 1e-8
MEAN_COV_FIELDS = {
    "predicted_mean": "predicted_cov",
    "filtered_mean": "filtered_cov",
    "one_step_mean": "one_step_cov",
    "smoothed_mean": "smoothed_cov",
}
# pilot_ssm and the entry-by-entry reference agree to round-off of each array's scale.
ARRAY_TOLERANCE = 1e-12
# D2: the crisis multiplier scales the level shock and the S and T news, not post-T news.
SCALED_BLOCK = ("level", "news_s", "news_t")


@pytest.fixture(params=LAYOUT_NAMES)
def layout_name(request):
    return request.param


def _prior_draws(layout_name, num_draws=NUM_PRIOR_DRAWS):
    draws = sample_prior(jax.random.PRNGKey(SEED), num_draws, LAYOUTS[layout_name])
    return [
        {name: float(value[k]) for name, value in draws.items()}
        for k in range(num_draws)
    ]


def _engine_moments(params, panel, layout_name):
    ssm = pilot_ssm(params, panel, LAYOUTS[layout_name])
    filtered = kalman_filter(ssm, jnp.asarray(panel.y))
    smoothed = kalman_smoother(ssm, filtered)
    return {
        **{name: np.asarray(value) for name, value in filtered._asdict().items()},
        **{name: np.asarray(value) for name, value in smoothed._asdict().items()},
    }


def _assert_matches_dense(engine, dense):
    assert set(engine) == {*FilterResult._fields, *SmootherResult._fields}
    np.testing.assert_allclose(
        engine["log_likelihood"], dense["log_likelihood"], rtol=DENSE_TOLERANCE
    )
    np.testing.assert_allclose(
        engine["step_log_likelihood"],
        dense["step_log_likelihood"],
        rtol=DENSE_TOLERANCE,
        atol=DENSE_TOLERANCE,
    )
    for mean_field, cov_field in MEAN_COV_FIELDS.items():
        variance = np.diagonal(dense[cov_field], axis1=1, axis2=2)
        assert (variance > 0).all(), cov_field
        cov_scale = np.sqrt(variance[:, :, None] * variance[:, None, :])
        cov_error = np.abs(engine[cov_field] - dense[cov_field]) / cov_scale
        mean_scale = np.abs(dense[mean_field]) + np.sqrt(variance)
        mean_error = np.abs(engine[mean_field] - dense[mean_field]) / mean_scale
        assert cov_error.max() <= DENSE_TOLERANCE, (cov_field, cov_error.max())
        assert mean_error.max() <= DENSE_TOLERANCE, (mean_field, mean_error.max())


def test_synthetic_panel_has_every_feature_the_pilot_must_handle():
    panel = synthetic_panel()
    anchored = ~np.isnan(panel.y[:, ANCHOR_COLUMN])
    closing = panel.y[:, :ANCHOR_COLUMN]

    assert panel.pre_2003.any() and not panel.pre_2003.all()
    assert np.flatnonzero(panel.crisis).tolist() == list(CRISIS_STEPS)
    assert anchored.sum() == 3
    np.testing.assert_array_equal(panel.anchor_offset[anchored], ANCHOR_OFFSETS[:3])
    assert not panel.anchor_offset[~anchored].any()
    assert np.isnan(panel.y[MISSING_STEP]).all()
    for step in CARRIER_STEPS:
        assert not np.isnan(closing[step, 0]) and np.isnan(closing[step, 1:]).all()
    assert np.isnan(closing[-1, 1:]).all() and np.isnan(closing[-2, 2])
    assert np.nanmin(panel.y[:, ANCHOR_COLUMN]) > 80_000.0  # real magnitude levels


def test_pilot_arrays_match_the_entry_by_entry_reference(layout_name):
    panel = synthetic_panel(layout_name)
    for params in [TRUTH, *_prior_draws(layout_name)]:
        ssm = pilot_ssm(params, panel, LAYOUTS[layout_name])
        reference = reference_ssm(params, panel, layout_name)
        assert set(reference) == set(LinearGaussianSSM._fields)
        for field, expected in reference.items():
            np.testing.assert_allclose(
                getattr(ssm, field),
                expected,
                rtol=ARRAY_TOLERANCE,
                atol=ARRAY_TOLERANCE * np.abs(expected).max(),
                err_msg=field,
            )


def test_filter_and_smoother_match_the_dense_reference(layout_name):
    panel = synthetic_panel(layout_name)
    engine = _engine_moments(TRUTH, panel, layout_name)
    dense = dense_reference(reference_ssm(TRUTH, panel, layout_name), panel.y)

    _assert_matches_dense(engine, dense)


def test_real_magnitude_filter_and_smoother_match_the_dense_reference(layout_name):
    panel = synthetic_panel(layout_name, REAL_MAGNITUDE_STEPS)
    closing = panel.y[:, :ANCHOR_COLUMN]

    assert np.nanmin(panel.y[:, ANCHOR_COLUMN]) > 80_000.0
    assert np.nanmax(closing) - np.nanmin(closing) > 1_000.0  # seasonal swings
    assert (~np.isnan(panel.y[:, ANCHOR_COLUMN])).sum() == len(ANCHOR_OFFSETS)
    for params in [TRUTH, *_prior_draws(layout_name)]:
        engine = _engine_moments(params, panel, layout_name)
        dense = dense_reference(reference_ssm(params, panel, layout_name), panel.y)
        _assert_matches_dense(engine, dense)


def test_model_samples_exactly_the_scalar_sites_and_factors_the_filter(layout_name):
    panel = synthetic_panel(layout_name)
    layout = LAYOUTS[layout_name]
    seeded = handlers.seed(pilot_model, jax.random.PRNGKey(SEED))
    trace = handlers.trace(seeded).get_trace(panel, layout)
    sampled = [
        name
        for name, site in trace.items()
        if site["type"] == "sample" and not site["is_observed"]
    ]
    observed = [
        name
        for name, site in trace.items()
        if site["type"] == "sample" and site["is_observed"]
    ]
    params = {name: trace[name]["value"] for name in sampled}

    assert tuple(sampled) == SAMPLED_SITES
    assert len(SAMPLED_SITES) == 17
    assert all(np.shape(value) == () for value in params.values())
    assert observed == ["panel"]
    assert {site["type"] for site in trace.values()} == {"sample"}  # no deterministic
    assert isinstance(trace["panel"]["fn"], dist.Unit)
    np.testing.assert_allclose(
        trace["panel"]["fn"].log_factor,
        kalman_filter(
            pilot_ssm(params, panel, layout), jnp.asarray(panel.y)
        ).log_likelihood,
        rtol=1e-12,
    )


def _expected_priors(layout_name):
    return {
        "sigma_level": (dist.HalfNormal, {"scale": 150.0}),
        "sigma_slope": (dist.HalfNormal, {"scale": 15.0}),
        "sigma_seasonal": (
            dist.HalfNormal,
            {"scale": SEASONAL_PRIOR_SCALE[layout_name]},
        ),
        "sigma_news_s": (dist.LogNormal, {"loc": math.log(40.0), "scale": 0.40}),
        "sigma_news_t": (dist.LogNormal, {"loc": math.log(35.0), "scale": 0.45}),
        "sigma_news_annual": (dist.LogNormal, {"loc": math.log(300.0), "scale": 0.60}),
        "noise_f_raw": (dist.Normal, {"loc": 0.0, "scale": 1.0}),
        "noise_s_raw": (dist.Normal, {"loc": 0.0, "scale": 1.0}),
        "noise_t_raw": (dist.Normal, {"loc": 0.0, "scale": 1.0}),
        "mean_news_s": (dist.Normal, {"loc": 0.0, "scale": 15.0}),
        "mean_news_t": (dist.Normal, {"loc": 0.0, "scale": 15.0}),
        "mean_news_annual": (dist.Normal, {"loc": 0.0, "scale": 100.0}),
        "sigma_anchor": (dist.HalfNormal, {"scale": 100.0}),
        "kappa_crisis": (dist.Normal, {"loc": math.log(50.0), "scale": 0.5}),
        "shift_log_scale": (dist.Normal, {"loc": 0.0, "scale": 0.5}),
        "shift_mean": (dist.Normal, {"loc": 0.0, "scale": 15.0}),
    }


def test_priors_are_the_contract_table(layout_name):
    priors = prior_distributions(LAYOUTS[layout_name])
    rho = priors["rho_noise"]

    assert tuple(priors) == SAMPLED_SITES
    for name, (kind, parameters) in _expected_priors(layout_name).items():
        assert type(priors[name]) is kind, name
        for parameter, value in parameters.items():
            assert float(getattr(priors[name], parameter)) == pytest.approx(value), name
    assert isinstance(rho.base_dist, dist.Normal)
    assert (float(rho.base_dist.loc), float(rho.base_dist.scale)) == (0.0, 0.2)
    assert (float(rho.low), float(rho.high)) == (-0.8, 0.8)


def test_initial_state_covers_every_layout_with_the_contract_values():
    for names in STATE_NAMES.values():
        assert set(names) <= set(INITIAL_MEAN) == set(INITIAL_SD)
    expected = {"level": (89_408.0, 5_000.0), "slope": (150.0, 300.0)}
    expected |= {"lagged_value": (89_408.0, 5_000.0)}
    expected |= {name: (0.0, 100.0) for name in ("news_s", "news_t", "news_post")}
    for name in INITIAL_MEAN:
        mean, sd = expected.get(name, (0.0, 2_000.0))  # every seasonal state
        assert (INITIAL_MEAN[name], INITIAL_SD[name]) == (mean, sd), name


def _arrays(params, panel, layout_name):
    ssm = pilot_ssm(params, panel, LAYOUTS[layout_name])
    return {name: np.asarray(value) for name, value in ssm._asdict().items()}


def _state_index(layout_name, names):
    return [STATE_NAMES[layout_name].index(name) for name in names]


def test_shift_log_scale_moves_only_pre_2003_news_and_noise_scales(layout_name):
    panel = synthetic_panel(layout_name)
    pre = panel.pre_2003
    base = _arrays(TRUTH, panel, layout_name)
    moved = _arrays(TRUTH | {"shift_log_scale": -0.4}, panel, layout_name)
    news_s, news_t, news_post = _state_index(
        layout_name, ("news_s", "news_t", "news_post")
    )
    level = _state_index(layout_name, ("level",))[0]
    others = [
        k
        for k in range(len(STATE_NAMES[layout_name]))
        if k not in (level, news_s, news_t, news_post)
    ]

    for name in base:
        if name not in ("transition_cov", "observation_cov"):
            np.testing.assert_array_equal(moved[name], base[name], err_msg=name)
    for name in ("transition_cov", "observation_cov"):
        np.testing.assert_array_equal(moved[name][~pre], base[name][~pre], err_msg=name)
    q_base, q_moved = base["transition_cov"][pre], moved["transition_cov"][pre]
    assert (q_moved[:, news_s, news_s] != q_base[:, news_s, news_s]).all()
    assert (q_moved[:, news_t, news_t] != q_base[:, news_t, news_t]).all()
    np.testing.assert_array_equal(q_moved[:, news_post], q_base[:, news_post])
    np.testing.assert_array_equal(q_moved[:, others], q_base[:, others])

    def non_news_level(q):
        return q[:, level, level] - q[:, news_s, news_s] - q[:, news_t, news_t]

    np.testing.assert_allclose(
        non_news_level(q_moved), non_news_level(q_base), rtol=1e-12
    )
    r_base, r_moved = base["observation_cov"][pre], moved["observation_cov"][pre]
    assert (r_moved[:, :3, :3] != r_base[:, :3, :3]).all()
    np.testing.assert_array_equal(r_moved[:, ANCHOR_COLUMN], r_base[:, ANCHOR_COLUMN])


def test_shift_mean_moves_only_pre_2003_news_offsets(layout_name):
    panel = synthetic_panel(layout_name)
    pre = panel.pre_2003
    base = _arrays(TRUTH, panel, layout_name)
    moved = _arrays(TRUTH | {"shift_mean": 9.0}, panel, layout_name)
    shifted = _state_index(layout_name, ("level", "news_s", "news_t"))
    change = moved["transition_offset"] - base["transition_offset"]

    for name in base:
        if name != "transition_offset":
            np.testing.assert_array_equal(moved[name], base[name], err_msg=name)
    assert not change[~pre].any()
    # shift_mean moves from -6 to 9: each pre-2003 S and T news mean by 15, the level by 30.
    expected = np.broadcast_to([2 * 15.0, 15.0, 15.0], (pre.sum(), 3))
    np.testing.assert_allclose(change[pre][:, shifted], expected, rtol=1e-12)
    unshifted = [k for k in range(change.shape[1]) if k not in shifted]
    assert not change[pre][:, unshifted].any()


def test_kappa_crisis_moves_only_the_level_and_closing_news_shocks_of_crisis_steps(
    layout_name,
):
    panel = synthetic_panel(layout_name)
    crisis = panel.crisis
    base = _arrays(TRUTH, panel, layout_name)
    moved = _arrays(TRUTH | {"kappa_crisis": math.log(80.0)}, panel, layout_name)
    block = _state_index(layout_name, SCALED_BLOCK)
    others = [k for k in range(len(STATE_NAMES[layout_name])) if k not in block]

    for name in base:
        if name != "transition_cov":
            np.testing.assert_array_equal(moved[name], base[name], err_msg=name)
    q_base, q_moved = base["transition_cov"], moved["transition_cov"]
    np.testing.assert_array_equal(q_moved[~crisis], q_base[~crisis])
    # Every row and column outside the level and the S and T news, news_post's included.
    np.testing.assert_array_equal(q_moved[crisis][:, others], q_base[crisis][:, others])
    np.testing.assert_array_equal(
        q_moved[crisis][:, :, others], q_base[crisis][:, :, others]
    )
    # kappa_crisis moves from log 30 to log 80, so every scaled variance grows by
    # (80/30)^2. The level's variance also holds the unscaled post-T news variance.
    level = SCALED_BLOCK.index("level")
    post = STATE_NAMES[layout_name].index("news_post")
    post_var = q_base[crisis][:, post, post]
    unscaled = np.zeros((crisis.sum(), len(block), len(block)))
    unscaled[:, level, level] = post_var
    in_block = np.ix_(np.flatnonzero(crisis), block, block)
    np.testing.assert_allclose(
        q_moved[in_block],
        (80.0 / 30.0) ** 2 * (q_base[in_block] - unscaled) + unscaled,
        rtol=1e-12,
    )
    assert (q_moved[in_block] != q_base[in_block]).any(axis=(1, 2)).all()


def _explicit_derived(p):
    noise_f = 0.5 * p["sigma_news_s"] * math.exp(0.7 * p["noise_f_raw"])
    noise_s = 0.5 * p["sigma_news_t"] * math.exp(0.7 * p["noise_s_raw"])
    noise_t = 0.5 * p["sigma_news_t"] * math.exp(0.7 * p["noise_t_raw"])
    rho = p["rho_noise"]
    # e_s - e_f = (rho - 1) z_f + z_s and e_t - e_s = (rho - 1) e_s + z_t.
    revision_s = (rho - 1) ** 2 * noise_f**2 + noise_s**2
    e_s = rho**2 * noise_f**2 + noise_s**2
    revision_t = (rho - 1) ** 2 * e_s + noise_t**2
    return {
        "sigma_noise_f": noise_f,
        "sigma_noise_s": noise_s,
        "sigma_noise_t": noise_t,
        "sigma_news_post": p["sigma_news_annual"] / math.sqrt(12),
        "share_s": p["sigma_news_s"] ** 2 / (p["sigma_news_s"] ** 2 + revision_s),
        "share_t": p["sigma_news_t"] ** 2 / (p["sigma_news_t"] ** 2 + revision_t),
    }


def test_derived_quantities_work_on_floats_numpy_and_jax_arrays():
    draws = sample_prior(jax.random.PRNGKey(SEED), 5, LAYOUTS["dummy"])
    as_numpy = {name: np.asarray(value) for name, value in draws.items()}
    expected = [
        _explicit_derived({name: float(value[k]) for name, value in as_numpy.items()})
        for k in range(5)
    ]
    from_numpy = derived(as_numpy)
    from_jax = derived(draws)
    from_jit = jax.jit(derived)(draws)

    assert set(from_numpy) == set(expected[0])
    assert all(type(value) is np.ndarray for value in from_numpy.values())
    assert all(isinstance(value, jax.Array) for value in from_jax.values())
    assert derived(TRUTH) == pytest.approx(_explicit_derived(TRUTH), rel=1e-14)
    for name in expected[0]:
        column = [row[name] for row in expected]
        np.testing.assert_allclose(from_numpy[name], column, rtol=1e-13, err_msg=name)
        np.testing.assert_allclose(from_jax[name], column, rtol=1e-13, err_msg=name)
        np.testing.assert_allclose(from_jit[name], column, rtol=1e-13, err_msg=name)
    assert ((from_numpy["share_s"] > 0) & (from_numpy["share_s"] < 1)).all()
    gradient = jax.grad(lambda p: derived(p)["share_t"])(TRUTH)
    assert all(np.isfinite(float(value)) for value in gradient.values())


def test_sample_prior_draws_every_site_on_its_support():
    draws = sample_prior(jax.random.PRNGKey(SEED), 200, LAYOUTS["trig4"])
    again = sample_prior(jax.random.PRNGKey(SEED), 200, LAYOUTS["trig4"])

    assert tuple(draws) == SAMPLED_SITES
    assert all(value.shape == (200,) for value in draws.values())
    for name in SAMPLED_SITES:
        np.testing.assert_array_equal(draws[name], again[name])
        if name.startswith("sigma_"):
            assert (draws[name] > 0).all(), name
    assert (np.abs(draws["rho_noise"]) < 0.8).all()


def test_step_log_likelihood_batches_under_jit_and_vmap(layout_name):
    panel = synthetic_panel(layout_name)
    layout = LAYOUTS[layout_name]
    draws = sample_prior(jax.random.PRNGKey(SEED), NUM_PRIOR_DRAWS, layout)
    batched = jax.jit(jax.vmap(lambda p: step_log_likelihood(p, panel, layout)))(draws)

    assert batched.shape == (NUM_PRIOR_DRAWS, panel.num_steps)
    for k in range(NUM_PRIOR_DRAWS):
        params = {name: value[k] for name, value in draws.items()}
        eager = step_log_likelihood(params, panel, layout)
        total = kalman_filter(pilot_ssm(params, panel, layout), jnp.asarray(panel.y))
        np.testing.assert_allclose(batched[k], eager, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(eager.sum(), total.log_likelihood, rtol=1e-12)
        assert float(eager[MISSING_STEP]) == 0.0


def test_log_density_gradient_is_finite_at_prior_draws(layout_name):
    panel = synthetic_panel(layout_name)
    layout = LAYOUTS[layout_name]

    def log_joint(params):
        return log_density(pilot_model, (panel, layout), {}, params)[0]

    value_and_grad = jax.jit(jax.value_and_grad(log_joint))
    for params in _prior_draws(layout_name):
        value, gradient = value_and_grad(params)
        assert np.isfinite(float(value))
        assert set(gradient) == set(SAMPLED_SITES)
        assert all(np.isfinite(float(entry)) for entry in gradient.values())
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_pilot_model.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'ces_revisions.pilot.model'`.

- [ ] **Step 3: Write the model**

Create `src/ces_revisions/pilot/model.py`:

```python
"""The Stage 6 pilot model: priors, the state-space arrays, and the NumPyro model.

``pilot_ssm`` turns one draw of the 17 sampled scalars into the time-varying
``LinearGaussianSSM`` over the panel's reference months, and ``pilot_model`` samples them
and adds the panel's marginal log likelihood with ``kalman_factor``, so no state is ever a
sampled site. Shock scales vary in time through two flags: before May 2003 the news and
noise SDs are multiplied by ``exp(shift_log_scale)`` and the S and T news means move by
``shift_mean`` (Req 4's 2003 regime shift); in March-June 2020 the non-news level shock and
the S and T news shocks are multiplied by ``exp(kappa_crisis)`` (decision D2). The post-T
news is not: the March anchors pin only its yearly total, so a multiplied post-T shock would
let the crisis months absorb the whole benchmark year's news. Revision noise follows Req 7's
within-month recursion e_f = z_f, e_s = rho e_f + z_s, e_t = rho e_s + z_t.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from ces_revisions.kalman import (
    FilterResult,
    LinearGaussianSSM,
    kalman_factor,
    kalman_filter,
)
from ces_revisions.pilot.layout import (
    LAYOUTS,
    NEWS_SHOCKS,
    NEWS_STATES,
    SEASONAL_PRIOR_SCALE,
    TREND_SHOCKS,
    Layout,
    observation_matrix,
    shock_loading,
    shock_names,
    transition_matrix,
)
from ces_revisions.pilot.panel import (
    ANCHOR_COLUMN,
    ANCHOR_ROUNDING_VARIANCE,
    CLOSING_ROUNDING_VARIANCE,
    CLOSING_STAGES,
    AggregatePanel,
)

MONTHS_PER_YEAR = 12
# NUTS's target acceptance for this model, for the fit and every SBC replicate alike. At
# NumPyro's 0.8 the real-panel fit diverged where sigma_news_t is small and trades off
# against noise_t_raw; at 0.95 it met Req 18 (plan 11, planning evidence).
TARGET_ACCEPT_PROB = 0.95

SAMPLED_SITES = (
    "sigma_level",
    "sigma_slope",
    "sigma_seasonal",
    "sigma_news_s",
    "sigma_news_t",
    "sigma_news_annual",
    "noise_f_raw",
    "noise_s_raw",
    "noise_t_raw",
    "rho_noise",
    "mean_news_s",
    "mean_news_t",
    "mean_news_annual",
    "sigma_anchor",
    "kappa_crisis",
    "shift_log_scale",
    "shift_mean",
)

# The state at December 1978 (x_{-1}), as independent normals, in thousands of jobs.
# lagged_value and the news states at t = -1 enter no later state or observation.
_SEASONAL_STATES = sorted(
    {name for layout in LAYOUTS.values() for name in layout.seasonal_names}
)
INITIAL_MEAN = {
    # Dec 1978 NSA total nonfarm in vintage cesvinall:2026-01.
    "level": 89_408.0,
    # Between the 1979-2019 average monthly gain (129) and 1979's (167), cesvinall:2026-01.
    "slope": 150.0,
    # x_{-2}, Nov 1978, is within a few hundred thousand of Dec 1978.
    "lagged_value": 89_408.0,
    # Seasonal effects sum to about zero over a year.
    **dict.fromkeys(_SEASONAL_STATES, 0.0),
    # The news states hold only the current month's shocks; x_{-1}'s enter nothing.
    **dict.fromkeys(NEWS_STATES, 0.0),
}
INITIAL_SD = {
    # Covers any vintage's Dec 1978 level, including every benchmark revision since.
    "level": 5_000.0,
    # Twice the mean: covers 1978's 359 thousand a month and a flat trend alike.
    "slope": 300.0,
    # As the level.
    "lagged_value": 5_000.0,
    # NSA seasonal swings in total nonfarm are one to two million.
    **dict.fromkeys(_SEASONAL_STATES, 2_000.0),
    # Wide against any monthly news SD; it enters nothing later.
    **dict.fromkeys(NEWS_STATES, 100.0),
}

# Req 15 noise row: a stage's noise SD is centered on half the SD of the news that next
# revises it, with a log-scale spread of 0.7. F's next news is S's and S's is T's. T's is
# the annual post-T news, which only the March anchors pin, so T's noise is centered on
# T's own news SD, as S's is (plan 11's reading of Req 15's sigma_{s,j}).
NOISE_CENTER = 0.5
NOISE_LOG_SCALE = 0.7
_NOISE_NEWS = {
    "sigma_noise_f": ("noise_f_raw", "sigma_news_s"),
    "sigma_noise_s": ("noise_s_raw", "sigma_news_t"),
    "sigma_noise_t": ("noise_t_raw", "sigma_news_t"),
}
# The standard normal sites behind the noise SDs: NOISE_LOG_SCALE times one is its SD's
# log-scale deviation from the center.
NOISE_RAW_SITES = tuple(raw for raw, _ in _NOISE_NEWS.values())


def prior_distributions(layout: Layout) -> dict[str, dist.Distribution]:
    """The prior of every sampled site, in ``SAMPLED_SITES`` order."""
    priors = {
        "sigma_level": dist.HalfNormal(150.0),  # Req 15 local-level row, non-news part
        "sigma_slope": dist.HalfNormal(15.0),
        "sigma_seasonal": dist.HalfNormal(SEASONAL_PRIOR_SCALE[layout.name]),
        "sigma_news_s": dist.LogNormal(math.log(40.0), 0.40),
        "sigma_news_t": dist.LogNormal(math.log(35.0), 0.45),
        "sigma_news_annual": dist.LogNormal(math.log(300.0), 0.60),  # post-T, per year
        "noise_f_raw": dist.Normal(0.0, 1.0),  # non-centered noise SDs; see derived()
        "noise_s_raw": dist.Normal(0.0, 1.0),
        "noise_t_raw": dist.Normal(0.0, 1.0),
        "rho_noise": dist.TruncatedNormal(0.0, 0.20, low=-0.8, high=0.8),
        "mean_news_s": dist.Normal(0.0, 15.0),
        "mean_news_t": dist.Normal(0.0, 15.0),
        "mean_news_annual": dist.Normal(0.0, 100.0),  # post-T, per year
        "sigma_anchor": dist.HalfNormal(100.0),
        "kappa_crisis": dist.Normal(math.log(50.0), 0.5),
        "shift_log_scale": dist.Normal(0.0, 0.5),
        "shift_mean": dist.Normal(0.0, 15.0),
    }
    return {name: priors[name] for name in SAMPLED_SITES}


def _namespace(params: dict):
    """jax.numpy when any value is a JAX array (or tracer), NumPy otherwise."""
    return jnp if any(isinstance(value, jax.Array) for value in params.values()) else np


def derived(params: dict) -> dict:
    """The deterministic quantities of a draw: noise SDs, post-T news SD, news shares.

    Works elementwise on scalars or arrays of draws, NumPy or JAX. The shares are the
    news variance's share of each revision's variance at post-2003 scale, without the
    rounding variance: Var(e_s - e_f) = (rho - 1)^2 s_f^2 + s_s^2 and
    Var(e_t - e_s) = (rho - 1)^2 (rho^2 s_f^2 + s_s^2) + s_t^2.
    """
    xp = _namespace(params)
    noise = {
        name: NOISE_CENTER * params[news] * xp.exp(NOISE_LOG_SCALE * params[raw])
        for name, (raw, news) in _NOISE_NEWS.items()
    }
    rho = params["rho_noise"]
    var_f, var_s, var_t = (noise[name] ** 2 for name in _NOISE_NEWS)
    noise_var_s = (rho - 1.0) ** 2 * var_f + var_s
    noise_var_t = (rho - 1.0) ** 2 * (rho**2 * var_f + var_s) + var_t
    news_var_s, news_var_t = params["sigma_news_s"] ** 2, params["sigma_news_t"] ** 2
    return {
        **noise,
        "sigma_news_post": params["sigma_news_annual"] / math.sqrt(MONTHS_PER_YEAR),
        "share_s": news_var_s / (news_var_s + noise_var_s),
        "share_t": news_var_t / (news_var_t + noise_var_t),
    }


def _closing_covariance(rho: jax.Array, sd: jax.Array) -> jax.Array:
    """Req 7's (T, 3, 3) closing-noise covariance M diag(sd^2) M' plus rounding."""
    zero, one = jnp.zeros_like(rho), jnp.ones_like(rho)
    recursion = jnp.stack(
        [
            jnp.stack([one, zero, zero]),
            jnp.stack([rho, one, zero]),
            jnp.stack([rho**2, rho, one]),
        ]
    )
    scaled = recursion[None] * sd[:, None, :]  # M diag(sd), per step
    rounding = CLOSING_ROUNDING_VARIANCE * jnp.eye(len(CLOSING_STAGES))
    return scaled @ jnp.swapaxes(scaled, 1, 2) + rounding


def pilot_ssm(
    params: dict[str, jax.Array], panel: AggregatePanel, layout: Layout
) -> LinearGaussianSSM:
    """The pilot's time-varying state-space model for one draw of the sampled sites."""
    num_steps, state_dim = panel.num_steps, layout.state_dim
    params = {
        name: jnp.asarray(value, dtype=jnp.float64) for name, value in params.items()
    }
    noise = derived(params)
    pre_2003 = jnp.asarray(panel.pre_2003, dtype=jnp.float64)
    crisis_scale = jnp.exp(
        params["kappa_crisis"] * jnp.asarray(panel.crisis, jnp.float64)
    )
    shift_scale = jnp.exp(params["shift_log_scale"] * pre_2003)

    # Shock SDs in shock_names order: eta_level, eta_slope, seasonal shocks, eps_s, eps_t,
    # eps_post. D2: the crisis multiplier scales the non-news level shock and the S and T
    # news shocks, never the slope, seasonal, or post-T news shocks.
    seasonal_shocks = len(shock_names(layout)) - len(TREND_SHOCKS) - len(NEWS_SHOCKS)
    ones = jnp.ones(num_steps)
    shock_sd = jnp.stack(
        [
            params["sigma_level"] * crisis_scale,
            params["sigma_slope"] * ones,
            *[params["sigma_seasonal"] * ones] * seasonal_shocks,
            params["sigma_news_s"] * shift_scale * crisis_scale,
            params["sigma_news_t"] * shift_scale * crisis_scale,
            noise["sigma_news_post"] * ones,
        ],
        axis=1,
    )
    loading = jnp.asarray(shock_loading(layout))
    transition_cov = jnp.einsum("ik,tk,jk->tij", loading, shock_sd**2, loading)

    # News enters the truth with its mean, so the level's offset is the news offsets' sum.
    news_mean = jnp.stack(
        [
            params["mean_news_s"] + params["shift_mean"] * pre_2003,
            params["mean_news_t"] + params["shift_mean"] * pre_2003,
            params["mean_news_annual"] / MONTHS_PER_YEAR * ones,
        ],
        axis=1,
    )
    names = layout.state_names
    transition_offset = (
        jnp.zeros((num_steps, state_dim))
        .at[:, names.index("level")]
        .set(news_mean.sum(axis=1))
        .at[:, [names.index(state) for state in NEWS_STATES]]
        .set(news_mean)
    )

    noise_sd = (
        jnp.stack([noise[name] for name in _NOISE_NEWS])[None, :] * shift_scale[:, None]
    )
    closing_cov = _closing_covariance(params["rho_noise"], noise_sd)
    anchor_var = params["sigma_anchor"] ** 2 + ANCHOR_ROUNDING_VARIANCE
    num_closing = len(CLOSING_STAGES)
    observation_cov = (
        jnp.zeros((num_steps, ANCHOR_COLUMN + 1, ANCHOR_COLUMN + 1))
        .at[:, :num_closing, :num_closing]
        .set(closing_cov)
        .at[:, ANCHOR_COLUMN, ANCHOR_COLUMN]
        .set(anchor_var)
    )
    observation_offset = (
        jnp.zeros((num_steps, ANCHOR_COLUMN + 1))
        .at[:, ANCHOR_COLUMN]
        .set(jnp.asarray(panel.anchor_offset))
    )

    def over_steps(matrix):
        return jnp.broadcast_to(jnp.asarray(matrix), (num_steps, *matrix.shape))

    return LinearGaussianSSM(
        initial_mean=jnp.array([INITIAL_MEAN[name] for name in names]),
        initial_cov=jnp.diag(jnp.array([INITIAL_SD[name] for name in names]) ** 2),
        transition_matrix=over_steps(transition_matrix(layout)),
        transition_offset=transition_offset,
        transition_cov=transition_cov,
        observation_matrix=over_steps(observation_matrix(layout)),
        observation_offset=observation_offset,
        observation_cov=observation_cov,
    )


def pilot_model(panel: AggregatePanel, layout: Layout) -> FilterResult:
    """Sample the 17 scalars; add the panel's marginal log likelihood as site "panel"."""
    params = {
        name: numpyro.sample(name, prior)
        for name, prior in prior_distributions(layout).items()
    }
    return kalman_factor(
        "panel", pilot_ssm(params, panel, layout), jnp.asarray(panel.y)
    )


def sample_prior(
    key: jax.Array, num_samples: int, layout: Layout
) -> dict[str, jax.Array]:
    """Independent prior draws of every sampled site, shape (num_samples,) each."""
    return draw_distributions(prior_distributions(layout), key, num_samples)


def draw_distributions(
    priors: dict[str, dist.Distribution], key: jax.Array, num_samples: int
) -> dict[str, jax.Array]:
    """Independent draws from each named distribution, one split key per name."""
    keys = jax.random.split(key, len(priors))
    return {
        name: prior.sample(site_key, (num_samples,))
        for (name, prior), site_key in zip(priors.items(), keys, strict=True)
    }


def step_log_likelihood(
    params: dict[str, jax.Array], panel: AggregatePanel, layout: Layout
) -> jax.Array:
    """The (T,) per-step log likelihood log p(y_t | y_<t) of one draw."""
    ssm = pilot_ssm(params, panel, layout)
    return kalman_filter(ssm, jnp.asarray(panel.y)).step_log_likelihood
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_pilot_model.py -q`
Expected: `34 passed`.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format src/ces_revisions/pilot/model.py tests/test_pilot_model.py
uv run ruff check src/ces_revisions/pilot/model.py tests/test_pilot_model.py
git add src/ces_revisions/pilot/model.py tests/test_pilot_model.py
git commit -m "Add the marginalized Jacobs-van Norden pilot model"
```

### Task 5: The simulator and the prior predictive

**Files:**
- Create: `src/ces_revisions/pilot/simulate.py`
- Modify: `tests/dense_reference.py` (rename `_joint_moments` to `joint_moments`)
- Test: `tests/test_pilot_simulate.py`

**Interfaces:**
- Consumes: `SAMPLED_SITES`, `INITIAL_MEAN`, `INITIAL_SD`, `MONTHS_PER_YEAR`, `NOISE_RAW_SITES`, `prior_distributions`, `derived`, `draw_distributions` (Task 4); the layout functions (Task 3); `AggregatePanel` (Task 2).
- Produces: `simulate_panel(params, panel, layout, seed, *, complete=False) -> np.ndarray` (T, 4), an explicit NumPy recursion independent of `pilot_ssm`; `simulate_complete_panels(params, panel, layout, seed, num_panels) -> np.ndarray`; `revision_statistics(y) -> dict[str, float]`; `prior_predictive(panel, layout, num_draws, seed, *, doubled=False) -> dict`; `monthly_constraint(prior_predictive_result) -> dict` with keys `share_median_in_range`, `pooled_share_above_250`, `passed`; constants `MEDIAN_RANGE = (20, 100)`, `MEDIAN_SHARE_MIN = 0.8`, `LARGE_REVISION = 250`, `LARGE_SHARE_RANGE = (0.001, 0.05)`, `SENSITIVITY_FACTOR = 2`. The doubled run multiplies every LogNormal prior's log-scale SD by `SENSITIVITY_FACTOR`, and the SD of each `NOISE_RAW_SITES` prior too, because the noise SDs are LogNormal through those sites.

- [ ] **Step 1: Make the dense oracle's moment helper public**

The simulator's moment test needs the oracle's joint moments. In `tests/dense_reference.py`, rename the function `_joint_moments` to `joint_moments` and its one call in `dense_reference`:

```diff
diff --git a/tests/dense_reference.py b/tests/dense_reference.py
index ad6f41d..98c8034 100644
--- a/tests/dense_reference.py
+++ b/tests/dense_reference.py
@@ -73,7 +73,7 @@ def _block_diag(blocks):
     return out
 
 
-def _joint_moments(ssm):
+def joint_moments(ssm):
     """Moments of the stacked states X (T*n) and cells Y (T*p), and Cov(X, Y)."""
     a, c, q = ssm["transition_matrix"], ssm["transition_offset"], ssm["transition_cov"]
     num_steps, n = c.shape
@@ -115,7 +115,7 @@ def _condition(mean_a, cov_aa, cov_ab, mean_b, cov_bb, value_b):
 
 def dense_reference(ssm, y):
     """Every FilterResult and SmootherResult field, by dense Gaussian conditioning."""
-    mean_x, cov_x, mean_y, cov_y, cov_xy = _joint_moments(ssm)
+    mean_x, cov_x, mean_y, cov_y, cov_xy = joint_moments(ssm)
     num_steps, p = y.shape
     n = len(ssm["initial_mean"])
     flat = y.ravel()
```

- [ ] **Step 2: Write the failing simulator tests**

Create `tests/test_pilot_simulate.py`:

```python
"""Tests of the pilot's simulator, revision statistics, and prior-predictive check."""

import math

import numpy as np
import numpyro.distributions as dist
import pytest
from dense_reference import joint_moments
from pilot_synthetic import TRUTH, reference_ssm, shell_panel, synthetic_panel

from ces_revisions.pilot.layout import LAYOUTS
from ces_revisions.pilot.model import (
    NOISE_RAW_SITES,
    SAMPLED_SITES,
    prior_distributions,
)
from ces_revisions.pilot.panel import ANCHOR_COLUMN, ANCHOR_ROUNDING_VARIANCE
from ces_revisions.pilot.simulate import (
    monthly_constraint,
    prior_predictive,
    revision_statistics,
    simulate_complete_panels,
    simulate_panel,
)

SEED = sum(map(ord, "ces-revisions-stage-6-pilot-simulate"))
LAYOUT_NAMES = sorted(LAYOUTS)
# The whitened moment check: NUM_PANELS independent complete panels of MOMENT_STEPS
# months, whitened by the dense oracle's mean and covariance of every observed cell,
# must look like independent standard normals. Bounds are in sampling SDs, so they are
# loose for a correct simulator and tight for a wrong one (see EVIDENCE in the spike
# report): a 10% error in one noise or news scale moves the worst variance by > 10 SDs.
NUM_PANELS = 40_000
MOMENT_STEPS = 24
MEAN_Z_MAX = 4.5  # on each whitened coordinate's mean, in units of 1/sqrt(N)
VARIANCE_Z_MAX = 5.0  # on each whitened coordinate's variance, in units of sqrt(2/N)
CORRELATION_Z_MAX = 5.5  # on each off-diagonal covariance, in units of 1/sqrt(N)
PRIOR_DRAWS = 50


@pytest.fixture(params=LAYOUT_NAMES)
def layout_name(request):
    return request.param


def test_masked_panel_is_the_complete_panel_with_the_panel_gaps(layout_name):
    panel = synthetic_panel(layout_name)
    layout = LAYOUTS[layout_name]
    complete = simulate_panel(TRUTH, panel, layout, SEED, complete=True)
    masked = simulate_panel(TRUTH, panel, layout, SEED)
    anchored = ~np.isnan(panel.y[:, ANCHOR_COLUMN])

    assert complete.shape == masked.shape == panel.y.shape
    assert not np.isnan(complete[:, :ANCHOR_COLUMN]).any()
    np.testing.assert_array_equal(~np.isnan(complete[:, ANCHOR_COLUMN]), anchored)
    np.testing.assert_array_equal(np.isnan(masked), np.isnan(panel.y))
    observed = ~np.isnan(panel.y)
    np.testing.assert_array_equal(masked[observed], complete[observed])
    np.testing.assert_array_equal(simulate_panel(TRUTH, panel, layout, SEED), masked)
    batch = simulate_complete_panels(TRUTH, panel, layout, SEED, 3)
    np.testing.assert_array_equal(
        simulate_complete_panels(TRUTH, panel, layout, SEED, 3), batch
    )
    np.testing.assert_array_equal(np.isnan(batch[0]), np.isnan(complete))
    assert not np.array_equal(batch[1], batch[0])
    assert not np.array_equal(simulate_panel(TRUTH, panel, layout, SEED + 1), masked)


def test_simulated_panels_have_the_dense_oracles_moments(layout_name):
    panel = shell_panel(MOMENT_STEPS)
    layout = LAYOUTS[layout_name]
    _, _, mean, cov, _ = joint_moments(reference_ssm(TRUTH, panel, layout_name))
    cells = ~np.isnan(panel.y.ravel())
    chol = np.linalg.cholesky(cov[np.ix_(cells, cells)])
    panels = simulate_complete_panels(TRUTH, panel, layout, SEED, NUM_PANELS)
    draws = panels.reshape(NUM_PANELS, -1)[:, cells]
    white = np.linalg.solve(chol, (draws - mean[cells]).T).T
    white_cov = np.cov(white, rowvar=False)
    off_diagonal = white_cov - np.diag(np.diag(white_cov))

    assert panel.pre_2003.any() and panel.crisis.any()
    assert (~np.isnan(panel.y[:, ANCHOR_COLUMN])).sum() == 2
    assert np.abs(white.mean(axis=0)).max() * math.sqrt(NUM_PANELS) < MEAN_Z_MAX
    variance_z = np.abs(np.diag(white_cov) - 1.0) / math.sqrt(2.0 / NUM_PANELS)
    assert variance_z.max() < VARIANCE_Z_MAX
    assert np.abs(off_diagonal).max() * math.sqrt(NUM_PANELS) < CORRELATION_Z_MAX


def test_anchor_noise_is_a_standard_normal_scaled_by_the_anchor_sd():
    # The whitened check cannot see the anchor noise under the level's prior SD, so
    # isolate it: the simulator's draws do not depend on the parameters, so changing
    # sigma_anchor alone moves only the anchor cells, by sd * z with a shared z.
    panel = shell_panel(MOMENT_STEPS)
    layout = LAYOUTS["dummy"]
    anchored = ~np.isnan(panel.y[:, ANCHOR_COLUMN])
    scales = (40.0, 80.0, 160.0)
    runs = [
        simulate_complete_panels(
            TRUTH | {"sigma_anchor": scale}, panel, layout, SEED, NUM_PANELS
        )
        for scale in scales
    ]
    sd = [math.sqrt(scale**2 + ANCHOR_ROUNDING_VARIANCE) for scale in scales]
    anchors = [run[:, anchored, ANCHOR_COLUMN] for run in runs]
    z = (anchors[1] - anchors[0]) / (sd[1] - sd[0])

    for run in runs[1:]:
        np.testing.assert_array_equal(
            run[:, :, :ANCHOR_COLUMN], runs[0][:, :, :ANCHOR_COLUMN]
        )
    np.testing.assert_allclose(anchors[2] - anchors[0], z * (sd[2] - sd[0]), atol=1e-8)
    size = z.size
    assert abs(z.mean()) * math.sqrt(size) < MEAN_Z_MAX
    assert abs(z.var() - 1.0) / math.sqrt(2.0 / size) < VARIANCE_Z_MAX


def test_revision_statistics_use_rows_with_both_f_and_t():
    y = np.full((6, 4), np.nan)
    y[:, 0] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    y[:, 2] = [-290.0, 25.0, 30.0, np.nan, 90.0, 360.0]
    y[1, 0] = np.nan  # F missing: dropped
    revision = np.array([-300.0, 0.0, 40.0, 300.0])

    statistics = revision_statistics(y)

    assert statistics == pytest.approx(
        {
            "median_abs": 170.0,
            "share_above_250": 0.5,
            "mean": revision.mean(),
            "sd": revision.std(ddof=1),
            "count": 4,
        }
    )


def test_prior_predictive_reports_one_complete_panel_per_draw(layout_name):
    panel = synthetic_panel(layout_name)
    layout = LAYOUTS[layout_name]
    result = prior_predictive(panel, layout, PRIOR_DRAWS, SEED)
    again = prior_predictive(panel, layout, PRIOR_DRAWS, SEED)
    statistics = result["statistics"]

    assert (result["layout"], result["seed"], result["doubled"]) == (
        layout_name,
        SEED,
        False,
    )
    assert tuple(result["params"]) == SAMPLED_SITES
    assert set(statistics) == {"median_abs", "share_above_250", "mean", "sd", "count"}
    assert all(values.shape == (PRIOR_DRAWS,) for values in statistics.values())
    assert (statistics["count"] == panel.num_steps).all()  # every closing cell filled
    for name, values in statistics.items():
        np.testing.assert_array_equal(values, again["statistics"][name], err_msg=name)
    constraint = monthly_constraint(result)
    assert set(constraint) == {
        "share_median_in_range",
        "pooled_share_above_250",
        "passed",
    }
    assert 0.0 <= constraint["share_median_in_range"] <= 1.0


def test_doubled_prior_doubles_the_lognormal_and_noise_log_scales():
    panel = synthetic_panel()
    layout = LAYOUTS["dummy"]
    priors = prior_distributions(layout)
    base = prior_predictive(panel, layout, PRIOR_DRAWS, SEED)["params"]
    doubled = prior_predictive(panel, layout, PRIOR_DRAWS, SEED, doubled=True)["params"]
    lognormal = [
        name for name, prior in priors.items() if isinstance(prior, dist.LogNormal)
    ]

    assert lognormal == ["sigma_news_s", "sigma_news_t", "sigma_news_annual"]
    assert NOISE_RAW_SITES == ("noise_f_raw", "noise_s_raw", "noise_t_raw")
    for name in SAMPLED_SITES:
        if name in lognormal:
            loc = float(priors[name].loc)
            np.testing.assert_allclose(
                np.log(doubled[name]) - loc,
                2.0 * (np.log(base[name]) - loc),
                rtol=1e-10,
            )
        elif name in NOISE_RAW_SITES:
            # Each noise SD's log-scale deviation from its center doubles with its raw.
            np.testing.assert_allclose(doubled[name], 2.0 * base[name], rtol=1e-12)
        else:
            np.testing.assert_array_equal(doubled[name], base[name], err_msg=name)


def _result(median_abs, share_above_250, count):
    size = len(median_abs)
    return {
        "statistics": {
            "median_abs": np.asarray(median_abs, dtype=float),
            "share_above_250": np.asarray(share_above_250, dtype=float),
            "count": np.full(size, count),
        }
    }


def test_monthly_constraint_applies_both_req_15_bounds():
    inside = [20.0, 50.0, 100.0, 60.0, 30.0]
    passing = monthly_constraint(_result(inside, [0.0, 0.01, 0.02, 0.0, 0.02], 100))
    narrow = monthly_constraint(
        _result([19.9, 50.0, 100.1, 60.0, 30.0], [0.01] * 5, 100)
    )
    too_quiet = monthly_constraint(_result(inside, [0.0] * 5, 100))
    too_loud = monthly_constraint(_result(inside, [0.06] * 5, 100))

    assert passing == {
        "share_median_in_range": 1.0,
        "pooled_share_above_250": pytest.approx(0.01),
        "passed": True,
    }
    assert narrow["share_median_in_range"] == pytest.approx(0.6)
    assert not narrow["passed"]
    assert too_quiet["pooled_share_above_250"] == 0.0 and not too_quiet["passed"]
    assert too_loud["pooled_share_above_250"] == pytest.approx(0.06)
    assert not too_loud["passed"]
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/test_pilot_simulate.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'ces_revisions.pilot.simulate'`.

- [ ] **Step 4: Write the simulator**

Create `src/ces_revisions/pilot/simulate.py`:

```python
"""Synthetic panels from the pilot model, and Req 15's prior-predictive monthly check.

``simulate_panel`` runs the model's equations forward as an explicit NumPy recursion, with
its own seasonal recursions, so it shares no code with ``model.pilot_ssm`` beyond the
priors' derived quantities and the initial-state constants; a test checks its moments
against the dense oracle. Rounding enters as Gaussian noise with the model's rounding
variances, exactly as the model treats it.

``prior_predictive`` draws parameters from the prior, simulates a complete panel per draw
on the real panel's flags, and records each draw's F-to-T revision statistics;
``monthly_constraint`` applies Req 15's monthly constraint at total nonfarm to them.
"""

import math

import jax
import numpy as np
import numpyro.distributions as dist

from ces_revisions.pilot.layout import PERIOD, Layout
from ces_revisions.pilot.model import (
    INITIAL_MEAN,
    INITIAL_SD,
    MONTHS_PER_YEAR,
    NOISE_RAW_SITES,
    derived,
    draw_distributions,
    prior_distributions,
)
from ces_revisions.pilot.panel import (
    ANCHOR_COLUMN,
    ANCHOR_ROUNDING_VARIANCE,
    CLOSING_ROUNDING_VARIANCE,
    COLUMNS,
    AggregatePanel,
)

F_COLUMN, T_COLUMN = COLUMNS.index("F"), COLUMNS.index("T")
# Req 15's monthly constraint at total nonfarm, operationalized: at least 80% of prior
# draws put the median |F->T| revision in [20, 100] thousand, and the pooled share of
# months with |F->T| above 250 thousand lies in [0.1%, 5%].
MEDIAN_RANGE = (20.0, 100.0)
MEDIAN_SHARE_MIN = 0.80
LARGE_REVISION = 250.0
LARGE_SHARE_RANGE = (0.001, 0.05)
# Req 15's sensitivity rerun doubles every LogNormal prior's log-scale SD. The noise SDs
# are LogNormal through their non-centered standard normal sites, so those double too.
SENSITIVITY_FACTOR = 2.0
# The standard normal draws of each month, in draw order; the seasonal shocks follow.
_MONTHLY_DRAWS = (
    "eps_s",
    "eps_t",
    "eps_post",
    "eta_level",
    "eta_slope",
    "z_f",
    "z_s",
    "z_t",
    "rounding_f",
    "rounding_s",
    "rounding_t",
    "anchor",
)


def _seasonal_gamma(layout: Layout, seasonal: np.ndarray) -> np.ndarray:
    """gamma from a (draws, S) seasonal block."""
    if layout.harmonics is None:
        return seasonal[:, 0]
    keep = [not name.endswith("_sin") for name in layout.seasonal_names]
    return seasonal[:, keep].sum(axis=1)


def _seasonal_step(
    layout: Layout, seasonal: np.ndarray, shocks: np.ndarray
) -> np.ndarray:
    """One month of the seasonal recursion; shocks are (draws, number of shocks), scaled."""
    if layout.harmonics is None:
        # gamma_t = -(gamma_{t-1} + ... + gamma_{t-11}) + w_t, then every slot ages a month.
        gamma = -seasonal.sum(axis=1) + shocks[:, 0]
        return np.column_stack([gamma, seasonal[:, :-1]])
    moved = np.empty_like(seasonal)
    for j in range(1, min(layout.harmonics, PERIOD // 2 - 1) + 1):
        angle = 2.0 * math.pi * j / PERIOD
        cos, sin = seasonal[:, 2 * j - 2], seasonal[:, 2 * j - 1]
        moved[:, 2 * j - 2] = math.cos(angle) * cos + math.sin(angle) * sin
        moved[:, 2 * j - 1] = -math.sin(angle) * cos + math.cos(angle) * sin
    if layout.harmonics == PERIOD // 2:
        moved[:, -1] = -seasonal[:, -1]
    return moved + shocks


def _simulate_complete(
    params: dict[str, np.ndarray], panel: AggregatePanel, layout: Layout, rng
) -> np.ndarray:
    """(draws, T, 4) panels: every closing cell, and anchors at the panel's anchor months."""
    p = {name: np.asarray(value, dtype=np.float64) for name, value in params.items()}
    noise = derived(p)
    num_draws, num_steps = p["sigma_level"].shape[0], panel.num_steps
    seasonal_shocks = 1 if layout.harmonics is None else layout.seasonal_dim

    def initial(name, size=None):
        return rng.normal(INITIAL_MEAN[name], INITIAL_SD[name], size)

    # lagged_value and the news states at t = -1 enter no later state, so only the
    # level, slope, and seasonal block of x_{-1} are drawn.
    level = initial("level", num_draws)
    slope = initial("slope", num_draws)
    seasonal = np.column_stack(
        [initial(name, num_draws) for name in layout.seasonal_names]
    )
    value = level + _seasonal_gamma(layout, seasonal)

    rounding_sd = math.sqrt(CLOSING_ROUNDING_VARIANCE)
    anchor_sd = np.sqrt(p["sigma_anchor"] ** 2 + ANCHOR_ROUNDING_VARIANCE)
    y = np.full((num_draws, num_steps, len(COLUMNS)), np.nan)
    anchored = ~np.isnan(panel.y[:, ANCHOR_COLUMN])
    for t in range(num_steps):
        pre, crisis = float(panel.pre_2003[t]), float(panel.crisis[t])
        crisis_scale = np.exp(p["kappa_crisis"] * crisis)
        shift_scale = np.exp(p["shift_log_scale"] * pre)
        z = dict(
            zip(
                _MONTHLY_DRAWS,
                rng.standard_normal((len(_MONTHLY_DRAWS), num_draws)),
                strict=True,
            )
        )
        seasonal_z = rng.standard_normal((num_draws, seasonal_shocks))

        # D2: the crisis multiplier scales the S and T news and the non-news level shock.
        news_s = (
            p["mean_news_s"]
            + p["shift_mean"] * pre
            + p["sigma_news_s"] * shift_scale * crisis_scale * z["eps_s"]
        )
        news_t = (
            p["mean_news_t"]
            + p["shift_mean"] * pre
            + p["sigma_news_t"] * shift_scale * crisis_scale * z["eps_t"]
        )
        news_post = (
            p["mean_news_annual"] / MONTHS_PER_YEAR
            + noise["sigma_news_post"] * z["eps_post"]
        )
        eta_level = p["sigma_level"] * crisis_scale * z["eta_level"]
        level, slope = (
            level + slope + eta_level + news_s + news_t + news_post,
            slope + p["sigma_slope"] * z["eta_slope"],
        )
        seasonal = _seasonal_step(
            layout, seasonal, p["sigma_seasonal"][:, None] * seasonal_z
        )
        previous, value = value, level + _seasonal_gamma(layout, seasonal)
        change = value - previous

        # Req 7: e_f = z_f, e_s = rho e_f + z_s, e_t = rho e_s + z_t.
        e_f = noise["sigma_noise_f"] * shift_scale * z["z_f"]
        e_s = p["rho_noise"] * e_f + noise["sigma_noise_s"] * shift_scale * z["z_s"]
        e_t = p["rho_noise"] * e_s + noise["sigma_noise_t"] * shift_scale * z["z_t"]
        y[:, t, 0] = (
            change - news_s - news_t - news_post + e_f + rounding_sd * z["rounding_f"]
        )
        y[:, t, 1] = change - news_t - news_post + e_s + rounding_sd * z["rounding_s"]
        y[:, t, 2] = change - news_post + e_t + rounding_sd * z["rounding_t"]
        if anchored[t]:
            y[:, t, ANCHOR_COLUMN] = (
                value + panel.anchor_offset[t] + anchor_sd * z["anchor"]
            )
    return y


def simulate_panel(
    params: dict[str, float],
    panel: AggregatePanel,
    layout: Layout,
    seed: int,
    *,
    complete: bool = False,
) -> np.ndarray:
    """One (T, 4) panel drawn from the model at params, with x_{-1} from its prior.

    NaN wherever ``panel.y`` is NaN, unless ``complete``: then every closing cell of
    every month is filled, and anchors appear at the panel's anchor months only. The
    same seed draws the same values either way.
    """
    y = simulate_complete_panels(params, panel, layout, seed, 1)[0]
    if not complete:
        y[np.isnan(panel.y)] = np.nan
    return y


def simulate_complete_panels(
    params: dict[str, float],
    panel: AggregatePanel,
    layout: Layout,
    seed: int,
    num_panels: int,
) -> np.ndarray:
    """num_panels independent complete (T, 4) panels at the same params, in one pass.

    The draws depend on num_panels: the first panel is not ``simulate_panel``'s.
    """
    batch = {name: np.full(num_panels, float(value)) for name, value in params.items()}
    return _simulate_complete(batch, panel, layout, np.random.default_rng(seed))


def revision_statistics(y: np.ndarray) -> dict[str, float]:
    """F-to-T revision statistics over the rows where both F and T are observed."""
    both = ~np.isnan(y[:, F_COLUMN]) & ~np.isnan(y[:, T_COLUMN])
    revision = y[both, T_COLUMN] - y[both, F_COLUMN]
    size = np.abs(revision)
    return {
        "median_abs": float(np.median(size)),
        "share_above_250": float(np.mean(size > LARGE_REVISION)),
        "mean": float(revision.mean()),
        "sd": float(revision.std(ddof=1)),
        "count": int(both.sum()),
    }


def _widened(priors: dict[str, dist.Distribution]) -> dict[str, dist.Distribution]:
    widened = {}
    for name, prior in priors.items():
        if isinstance(prior, dist.LogNormal):
            widened[name] = dist.LogNormal(prior.loc, SENSITIVITY_FACTOR * prior.scale)
        elif name in NOISE_RAW_SITES:
            widened[name] = dist.Normal(prior.loc, SENSITIVITY_FACTOR * prior.scale)
        else:
            widened[name] = prior
    return widened


def prior_predictive(
    panel: AggregatePanel,
    layout: Layout,
    num_draws: int,
    seed: int,
    *,
    doubled: bool = False,
) -> dict:
    """F-to-T revision statistics of one complete simulated panel per prior draw.

    ``doubled`` doubles every LogNormal prior's log-scale SD and the SD of each
    ``NOISE_RAW_SITES`` prior (Req 15's sensitivity rerun).
    Returns the draws' parameters and statistics as arrays of shape (num_draws,).
    """
    priors = prior_distributions(layout)
    if doubled:
        priors = _widened(priors)
    params = {
        name: np.asarray(value)
        for name, value in draw_distributions(
            priors, jax.random.PRNGKey(seed), num_draws
        ).items()
    }
    panels = _simulate_complete(params, panel, layout, np.random.default_rng(seed))
    rows = [revision_statistics(y) for y in panels]
    return {
        "layout": layout.name,
        "seed": seed,
        "doubled": doubled,
        "num_draws": num_draws,
        "params": params,
        "statistics": {name: np.array([row[name] for row in rows]) for name in rows[0]},
    }


def monthly_constraint(prior_predictive_result: dict) -> dict:
    """Req 15's monthly constraint at total nonfarm, applied to a prior_predictive result."""
    statistics = prior_predictive_result["statistics"]
    median = statistics["median_abs"]
    low, high = MEDIAN_RANGE
    share_median_in_range = float(np.mean((median >= low) & (median <= high)))
    counts = statistics["count"]
    pooled = float(np.sum(statistics["share_above_250"] * counts) / np.sum(counts))
    passed = (
        share_median_in_range >= MEDIAN_SHARE_MIN
        and LARGE_SHARE_RANGE[0] <= pooled <= LARGE_SHARE_RANGE[1]
    )
    return {
        "share_median_in_range": share_median_in_range,
        "pooled_share_above_250": pooled,
        "passed": bool(passed),
    }
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_pilot_simulate.py tests/test_dense_reference.py tests/test_kalman.py -q`
Expected: all pass (`13` in `tests/test_pilot_simulate.py`).

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format src/ces_revisions/pilot/simulate.py tests/test_pilot_simulate.py tests/dense_reference.py
uv run ruff check src/ces_revisions/pilot/simulate.py tests/test_pilot_simulate.py tests/dense_reference.py
git add src/ces_revisions/pilot/simulate.py tests/test_pilot_simulate.py tests/dense_reference.py
git commit -m "Add the pilot simulator and Req 15's prior predictive"
```

### Task 6: The run harness

**Files:**
- Create: `src/ces_revisions/runs/__init__.py`, `src/ces_revisions/runs/inits.py`, `src/ces_revisions/runs/diagnostics.py`, `src/ces_revisions/runs/inference_data.py`, `src/ces_revisions/runs/provenance.py`, `src/ces_revisions/runs/fit.py`
- Modify: `src/ces_revisions/devices.py`, `src/ces_revisions/engine_probe.py`
- Modify: `pyproject.toml`, `uv.lock`, `tests/test_stack.py`, `.gitignore`
- Test: `tests/test_run_harness.py`

**Interfaces:**
- Consumes: `tests/synthetic_pilot.py` (the Stage 1 synthetic pilot, so the harness is tested on a model that is not the Stage 6 pilot), `kalman_filter`, `chain_method`, `vintage_raw.ROOT`, and the operator record `data/operators/operator-test-results.json` that `scripts/operator_artifacts.py build` writes.
- Produces:
  - `devices.jax_versions() -> dict[str, str]` and `devices.nvidia_driver() -> str | None` (moved from `engine_probe.py`); `engine_probe.batched_value_and_grad()` (public).
  - `runs.inits`: `DISPERSION_OFFSETS = (-1.5, -0.5, 0.5, 1.5)`, `chain_offsets(num_chains) -> np.ndarray`, `dispersed_init_params(model, model_args, prior_draws, num_chains, seed, *, scale=1.0) -> tuple[dict[str, jax.Array], dict]`, `sample_site_supports(model, model_args, seed) -> dict`, `initial_potential_energies(model, model_args, init_params, seed) -> np.ndarray`.
  - `runs.diagnostics`: `Thresholds` (frozen: `rhat_max=1.01`, `ess_min=400.0`, `bfmi_min=0.3`, `max_tree_depth=10`, `saturated_share_max=0.01`, `divergences_max=0`), `REQ18`, `diagnostics_record(idata, thresholds=REQ18, *, rank_bins=20) -> dict` (JSON-safe, with `checks` and `meets_req18`).
  - `runs.inference_data`: `step_log_likelihood_draws(step_fn, samples_by_chain, *, batch_size=500) -> np.ndarray` (chain, draw, T); `attach_log_likelihood(idata, step_log_likelihood, reference_months, benchmark_year, observed_any, name="panel") -> xr.DataTree` (variables `f"{name}_month"` over observed steps and `f"{name}_benchmark_year"`); `write_inference_data(idata, path)`, `read_inference_data(path) -> xr.DataTree`.
  - `runs.provenance`: `git_state(root)`, `library_versions()`, `host_record()`, `data_inputs(root)`, `operator_test_record(root)`, `array_sha256(arrays) -> str`, `run_provenance(*, root, seeds, model, init_record, arrays, settings) -> dict`.
  - `runs.fit`: `RUNS_DIR` (`data/runs`), `RunSettings` (frozen: `seed`, `num_chains=4`, `num_warmup=1000`, `num_samples=1000`, `max_tree_depth=10`, `target_accept_prob=0.8`), `RunResult` (`idata`, `diagnostics`, `provenance`, `sampling_seconds`), `run_nuts(model, model_args, *, settings, prior_draws, step_fn, reference_months, benchmark_year, observed_any, model_record, arrays, root=vintage_raw.ROOT) -> RunResult`, `write_run(result, run_dir) -> Path`, `new_run_dir(label, seed, runs_dir=RUNS_DIR) -> Path`, and the file names `INFERENCE_DATA_FILE`, `DIAGNOSTICS_FILE`, `PROVENANCE_FILE`.

`RunSettings` keeps NumPyro's 0.8 as its default because the harness is model-agnostic; the pilot passes its own `TARGET_ACCEPT_PROB`. This task closes plan 1's dispersed-chain-starts item: every run starts four chains at per-site permutations of ±0.5 and ±1.5 prior SDs in unconstrained space, checks that each start has a finite potential energy, and records the strategy in its provenance.

- [ ] **Step 1: Add the dependencies**

Run:

```bash
uv add "h5netcdf>=1.8.1" "h5py>=3.16.0" "scipy>=1.18.1"
```

Expected: `pyproject.toml` gains the three lines below, and `uv.lock` gains h5netcdf and h5py (SciPy is already locked):

```diff
diff --git a/pyproject.toml b/pyproject.toml
index 3788256..34e98ee 100644
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -10,11 +10,14 @@ requires-python = ">=3.14"
 dependencies = [
     "arviz>=1.3.0",
     "fastexcel>=0.21.0",
+    "h5netcdf>=1.8.1",
+    "h5py>=3.16.0",
     "jax>=0.11.1",
     "numpy>=2.5.3",
     "numpyro>=0.21.0",
     "polars>=1.44.2",
     "python-dotenv>=1.0",
+    "scipy>=1.18.1",
 ]
 
 [project.optional-dependencies]
```

- [ ] **Step 2: Extend the stack test and ignore run directories**

Edit `tests/test_stack.py`:

```diff
diff --git a/tests/test_stack.py b/tests/test_stack.py
index 04a2fb6..d119166 100644
--- a/tests/test_stack.py
+++ b/tests/test_stack.py
@@ -11,8 +11,10 @@ import pytest
 from ces_revisions.devices import DETERMINISTIC_GPU_FLAG, has_nvidia_device
 
 # One importable module per pinned distribution: runtime jax, numpy, numpyro, arviz,
-# polars, and fastexcel, the Excel reader behind Stage 3's workbooks; dynamax from the dev
-# group, as engine-determination evidence only.
+# polars, and fastexcel, the Excel reader behind Stage 3's workbooks; h5netcdf and h5py,
+# the netCDF backend that writes each run's DataTree; scipy, for the run diagnostics' ranks
+# and the Ljung-Box p-value; dynamax from the dev group, as engine-determination evidence
+# only.
 STACK_MODULES = [
     "jax",
     "numpy",
@@ -20,6 +22,9 @@ STACK_MODULES = [
     "arviz",
     "polars",
     "fastexcel",
+    "h5netcdf",
+    "h5py",
+    "scipy",
     "dynamax.linear_gaussian_ssm",
 ]
 
```

Append to `.gitignore`:

```diff
diff --git a/.gitignore b/.gitignore
index 3c60f7e..fe32a72 100644
--- a/.gitignore
+++ b/.gitignore
@@ -28,3 +28,6 @@ data/annual/panel/
 
 # Stage 5 deterministic operator artifacts are rebuilt from the committed Stage 3/4 sources
 data/operators/
+
+# Stage 6 run directories (InferenceData, diagnostics, provenance) are rebuilt by each fit
+data/runs/
```

- [ ] **Step 3: Move the host helpers into `devices.py`**

Edit `src/ces_revisions/devices.py`:

```diff
diff --git a/src/ces_revisions/devices.py b/src/ces_revisions/devices.py
index bfc68b3..a99e23a 100644
--- a/src/ces_revisions/devices.py
+++ b/src/ces_revisions/devices.py
@@ -1,5 +1,8 @@
 """Device facts, so one checkout runs on the Mac's CPU and on a rented GPU host."""
 
+import importlib.metadata
+import shutil
+import subprocess
 from pathlib import Path
 
 import jax
@@ -18,3 +21,22 @@ def has_nvidia_device() -> bool:
 def chain_method(num_chains: int) -> str:
     """NumPyro's chain method: parallel with a device per chain, otherwise vectorized."""
     return "parallel" if jax.local_device_count() >= num_chains else "vectorized"
+
+
+def jax_versions() -> dict[str, str]:
+    """Versions of jax, jaxlib, and any installed JAX CUDA plugin."""
+    versions = {}
+    for dist in importlib.metadata.distributions():
+        name = dist.name.lower().replace("_", "-")
+        if name in {"jax", "jaxlib"} or name.startswith("jax-cuda"):
+            versions[name] = dist.version
+    return dict(sorted(versions.items()))
+
+
+def nvidia_driver() -> str | None:
+    """The NVIDIA driver version, or None on a host without nvidia-smi."""
+    if shutil.which("nvidia-smi") is None:
+        return None
+    query = ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
+    output = subprocess.run(query, capture_output=True, text=True, check=True).stdout
+    return output.splitlines()[0].strip()
```

Edit `src/ces_revisions/engine_probe.py`:

```diff
diff --git a/src/ces_revisions/engine_probe.py b/src/ces_revisions/engine_probe.py
index ce4b98f..8378423 100644
--- a/src/ces_revisions/engine_probe.py
+++ b/src/ces_revisions/engine_probe.py
@@ -8,10 +8,7 @@ writes one JSON record, so runs on the Mac and on each GPU host compare field by
 """
 
 import argparse
-import importlib.metadata
 import json
-import shutil
-import subprocess
 import time
 from collections.abc import Callable
 from datetime import UTC, datetime
@@ -22,6 +19,7 @@ import jax.numpy as jnp
 import numpy as np
 import numpyro
 
+from ces_revisions.devices import jax_versions, nvidia_driver
 from ces_revisions.kalman import LinearGaussianSSM, kalman_filter
 
 # Each transition matrix is this multiple of a random orthogonal matrix, so every
@@ -60,6 +58,17 @@ def synthetic_problem(
     return ssm, panel
 
 
+def batched_value_and_grad() -> Callable[..., object]:
+    """The jitted value and gradient the probe times, vmapped over a (batch, 2) array.
+
+    It maps ``(log_scales, ssm, panel)`` to the batch's log likelihoods and their
+    gradients with respect to the two log scales; ``ssm`` and ``panel`` are shared.
+    """
+    return jax.jit(
+        jax.vmap(jax.value_and_grad(_log_likelihood), in_axes=(0, None, None))
+    )
+
+
 def time_batches(
     ssm: LinearGaussianSSM,
     panel: np.ndarray,
@@ -75,9 +84,7 @@ def time_batches(
     """
     ssm = LinearGaussianSSM(*(jnp.asarray(array) for array in ssm))
     panel = jnp.asarray(panel)
-    value_and_grad = jax.jit(
-        jax.vmap(jax.value_and_grad(_log_likelihood), in_axes=(0, None, None))
-    )
+    value_and_grad = batched_value_and_grad()
     rng = np.random.default_rng(seed)
     timings = []
     for batch in batch_sizes:
@@ -115,8 +122,8 @@ def main(argv: list[str] | None = None) -> None:
         "repeats": args.repeats,
         "batches": time_batches(ssm, panel, args.batch, args.repeats, args.seed),
         "backend": jax.default_backend(),
-        "versions": _jax_versions(),
-        "nvidia_driver": _nvidia_driver(),
+        "versions": jax_versions(),
+        "nvidia_driver": nvidia_driver(),
         "timestamp_utc": started.isoformat(timespec="seconds"),
     }
     args.out.parent.mkdir(parents=True, exist_ok=True)
@@ -185,24 +192,5 @@ def _call_seconds(function: Callable[..., object], *args: object) -> float:
     return time.perf_counter() - start
 
 
-def _jax_versions() -> dict[str, str]:
-    """Versions of jax, jaxlib, and any installed JAX CUDA plugin."""
-    versions = {}
-    for dist in importlib.metadata.distributions():
-        name = dist.name.lower().replace("_", "-")
-        if name in {"jax", "jaxlib"} or name.startswith("jax-cuda"):
-            versions[name] = dist.version
-    return dict(sorted(versions.items()))
-
-
-def _nvidia_driver() -> str | None:
-    """The NVIDIA driver version, or None on a host without nvidia-smi."""
-    if shutil.which("nvidia-smi") is None:
-        return None
-    query = ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
-    output = subprocess.run(query, capture_output=True, text=True, check=True).stdout
-    return output.splitlines()[0].strip()
-
-
 if __name__ == "__main__":
     main()
```

Run: `uv run pytest tests/test_stack.py tests/test_engine_probe.py tests/test_devices.py -q`
Expected: all pass.

- [ ] **Step 4: Write the failing harness tests**

Create `tests/test_run_harness.py`:

```python
"""The model-agnostic run harness, exercised on the Stage 1 synthetic pilot."""

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, date, datetime

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from numpyro.infer import log_likelihood
from numpyro.infer.util import potential_energy
from synthetic_pilot import (
    LAPSE_STEP,
    NUM_STEPS,
    PRIOR_MEDIANS,
    TRUTH,
    pilot_model,
    pilot_ssm,
    simulate_panel,
)

from ces_revisions.devices import chain_method
from ces_revisions.kalman import kalman_filter
from ces_revisions.runs import fit, provenance
from ces_revisions.runs.diagnostics import REQ18, Thresholds, diagnostics_record
from ces_revisions.runs.fit import RunSettings, new_run_dir, run_nuts, write_run
from ces_revisions.runs.inference_data import (
    attach_log_likelihood,
    read_inference_data,
    step_log_likelihood_draws,
    write_inference_data,
)
from ces_revisions.runs.inits import DISPERSION_OFFSETS, dispersed_init_params
from ces_revisions.vintages import raw as vintage_raw

SEED = sum(map(ord, "ces-revisions-stage-6-run-harness"))
NUM_PRIOR_DRAWS = 2000
# The synthetic pilot's priors are LogNormal(log median, 1); see tests/synthetic_pilot.py.
PRIOR_LOG_SD = 1.0
FIRST_MONTH = date(1979, 1, 1)
# Fake DataTree for the diagnostics: four chains of 1000 iid draws pass every Req 18 check.
FAKE_CHAINS, FAKE_DRAWS = 4, 1000
SATURATION_STEPS = 2**REQ18.max_tree_depth - 1
SAMPLE_STAT_CHECKS = ("bfmi", "divergences", "tree_depth")
DRAW_CHECKS = ("rhat", "ess_bulk", "ess_tail")
# A stand-in Stage 5 operator test record, so provenance tests need no built artifact.
OPERATOR_RECORD = {"schema_version": 1, "checks": {"wedge": {"years": 23}}}


@pytest.fixture(scope="module")
def panel():
    return jnp.asarray(simulate_panel(TRUTH, seed=SEED))


@pytest.fixture(scope="module")
def prior_draws():
    rng = np.random.default_rng(SEED)
    return {
        name: np.exp(rng.normal(np.log(median), PRIOR_LOG_SD, NUM_PRIOR_DRAWS))
        for name, median in PRIOR_MEDIANS.items()
    }


@pytest.fixture(scope="module")
def reference_months():
    return tuple(
        date(FIRST_MONTH.year + t // 12, t % 12 + 1, 1) for t in range(NUM_STEPS)
    )


@pytest.fixture(scope="module")
def benchmark_year(reference_months):
    # Req 10's April-to-March benchmark year, computed here so the test needs no pilot code.
    return np.array(
        [
            month.year + 1 if month.month >= 4 else month.year
            for month in reference_months
        ],
        dtype=np.int64,
    )


def copy_source_manifests(root):
    """Copy the committed Stage 3 and Stage 4 source manifests under root."""
    for relative in provenance.SOURCE_MANIFESTS.values():
        (root / relative).parent.mkdir(parents=True)
        (root / relative).write_bytes((vintage_raw.ROOT / relative).read_bytes())


@pytest.fixture
def provenance_root(tmp_path, monkeypatch):
    """A root with the source manifests and a stand-in operator record, but no built
    artifacts; git_state reports the checkout's own state, since the root is no repo."""
    root = tmp_path / "root"
    copy_source_manifests(root)
    record_path = root / provenance.OPERATOR_TEST_RECORD
    record_path.parent.mkdir(parents=True)
    record_path.write_text(json.dumps(OPERATOR_RECORD))
    state = provenance.git_state(vintage_raw.ROOT)
    monkeypatch.setattr(provenance, "git_state", lambda root: state)
    return root


def step_fn_for(panel):
    return lambda params: kalman_filter(pilot_ssm(params), panel).step_log_likelihood


def near_truth_samples(num_chains, num_draws, seed):
    """Posterior-like (chain, draw) draws: the truth times a small lognormal jitter."""
    rng = np.random.default_rng(seed)
    return {
        name: value * np.exp(0.05 * rng.normal(size=(num_chains, num_draws)))
        for name, value in TRUTH.items()
    }


def factor_log_likelihood(panel, samples_by_chain):
    """The model's own factor log likelihood per draw, through NumPyro, not step_fn."""
    num_chains, num_draws = next(iter(samples_by_chain.values())).shape[:2]
    flat = {
        name: jnp.reshape(jnp.asarray(value), (num_chains * num_draws,))
        for name, value in samples_by_chain.items()
    }
    values = np.asarray(log_likelihood(pilot_model, flat, panel)["panel"])
    return values.reshape(num_chains, num_draws)


def fake_idata(
    seed, *, shift_chain=0.0, ar=0.0, random_walk_energy=False, divergent=0, saturated=0
):
    """A DataTree with a scalar and a vector posterior variable and NUTS sample stats."""
    rng = np.random.default_rng(seed)
    shape = (FAKE_CHAINS, FAKE_DRAWS)
    alpha = rng.normal(size=shape)
    if ar:
        for t in range(1, FAKE_DRAWS):
            alpha[:, t] = ar * alpha[:, t - 1] + np.sqrt(1 - ar**2) * alpha[:, t]
    alpha[0] += shift_chain
    energy = rng.normal(size=shape)
    if random_walk_energy:
        energy = energy.cumsum(axis=1)
    diverging = np.zeros(shape, dtype=bool)
    diverging.flat[rng.choice(diverging.size, divergent, replace=False)] = True
    n_steps = np.full(shape, 7, dtype=np.int64)
    n_steps.flat[rng.choice(n_steps.size, saturated, replace=False)] = SATURATION_STEPS
    return az.from_dict(
        {
            "posterior": {"alpha": alpha, "beta": rng.normal(size=(*shape, 2))},
            "sample_stats": {
                "diverging": diverging,
                "energy": energy,
                "n_steps": n_steps,
                "acceptance_rate": np.full(shape, 0.8),
                "step_size": np.full(shape, 0.5),
            },
        }
    )


# ---- dispersed starts ----


def test_dispersed_starts_bracket_the_prior_median_and_differ_per_chain(
    panel, prior_draws
):
    init_params, record = dispersed_init_params(
        pilot_model, (panel,), prior_draws, 4, SEED
    )

    assert set(init_params) == set(PRIOR_MEDIANS)
    for name, draws in prior_draws.items():
        starts = np.asarray(init_params[name])
        median, sd = np.median(np.log(draws)), np.std(np.log(draws))
        assert starts.shape == (4,)
        assert starts.min() < median < starts.max(), name
        assert len(set(starts.tolist())) == 4, name
        np.testing.assert_allclose(
            np.sort((starts - median) / sd), DISPERSION_OFFSETS, rtol=0, atol=1e-12
        )
        site = record["sites"][name]
        np.testing.assert_allclose(site["constrained"], np.exp(starts), rtol=1e-12)
        np.testing.assert_allclose(site["unconstrained"], starts, rtol=0, atol=0)
    # The offsets are permuted per site, so no chain is the extreme start of every site.
    assert len({tuple(record["sites"][name]["offsets"]) for name in PRIOR_MEDIANS}) > 1
    json.dumps(record, allow_nan=False)


def test_initial_potential_energy_readback_matches_numpyro(panel, prior_draws):
    init_params, record = dispersed_init_params(
        pilot_model, (panel,), prior_draws, 4, SEED
    )
    expected = [
        float(
            potential_energy(
                pilot_model,
                (panel,),
                {},
                {name: value[chain] for name, value in init_params.items()},
            )
        )
        for chain in range(4)
    ]

    np.testing.assert_allclose(record["initial_potential_energy"], expected, rtol=1e-10)
    assert record["readback_seconds"] > 0


def test_dispersed_starts_are_seeded(panel, prior_draws):
    first, _ = dispersed_init_params(pilot_model, (panel,), prior_draws, 4, SEED)
    again, _ = dispersed_init_params(pilot_model, (panel,), prior_draws, 4, SEED)
    other, _ = dispersed_init_params(pilot_model, (panel,), prior_draws, 4, SEED + 1)

    for name in PRIOR_MEDIANS:
        np.testing.assert_array_equal(first[name], again[name])
    assert any(not np.array_equal(first[name], other[name]) for name in PRIOR_MEDIANS)


def test_other_chain_counts_use_evenly_spaced_offsets(panel, prior_draws):
    _, record = dispersed_init_params(pilot_model, (panel,), prior_draws, 3, SEED)

    assert record["offsets"] == [-1.5, 0.0, 1.5]
    with pytest.raises(ValueError, match="at least two chains"):
        dispersed_init_params(pilot_model, (panel,), prior_draws, 1, SEED)


def test_a_non_finite_start_raises(panel, prior_draws):
    # A thousand prior SDs out, a scale overflows and the potential energy is not finite.
    with pytest.raises(ValueError, match="finite potential energy"):
        dispersed_init_params(pilot_model, (panel,), prior_draws, 4, SEED, scale=1000.0)


def test_prior_draws_must_cover_exactly_the_sampled_sites(panel, prior_draws):
    missing = {
        name: draws for name, draws in prior_draws.items() if name != "sigma_slope"
    }
    extra = {**prior_draws, "sigma_unused": prior_draws["sigma_slope"]}
    constant = {**prior_draws, "sigma_slope": np.full(NUM_PRIOR_DRAWS, 3.0)}

    for draws in (missing, extra):
        with pytest.raises(ValueError, match="exactly the sampled sites"):
            dispersed_init_params(pilot_model, (panel,), draws, 4, SEED)
    with pytest.raises(ValueError, match="not all equal"):
        dispersed_init_params(pilot_model, (panel,), constant, 4, SEED)


# ---- diagnostics ----


def test_diagnostics_record_passes_iid_draws_and_is_json_safe():
    record = diagnostics_record(fake_idata(SEED))

    assert record["meets_req18"] is True
    assert set(record["checks"]) == {*DRAW_CHECKS, *SAMPLE_STAT_CHECKS}
    assert all(value is True for value in record["checks"].values())
    assert set(record["variables"]) == {"alpha", "beta[0]", "beta[1]"}
    assert record["thresholds"] == {
        "rhat_max": 1.01,
        "ess_min": 400.0,
        "bfmi_min": 0.3,
        "max_tree_depth": 10,
        "saturated_share_max": 0.01,
        "divergences_max": 0,
    }
    assert record["rhat_method"] == "rank"
    assert len(record["bfmi"]) == FAKE_CHAINS
    assert record["divergences"] == {"total": 0, "per_chain": [0] * FAKE_CHAINS}
    assert record["tree_depth"]["saturation_steps"] == SATURATION_STEPS
    assert record["tree_depth"]["n_steps_max"] == 7
    assert record["acceptance_rate"] == pytest.approx([0.8] * FAKE_CHAINS)
    assert record["step_size"] == pytest.approx([0.5] * FAKE_CHAINS)
    for counts in record["rank_histograms"]["counts"].values():
        assert np.shape(counts) == (FAKE_CHAINS, record["rank_histograms"]["bins"])
        assert np.sum(counts, axis=1).tolist() == [FAKE_DRAWS] * FAKE_CHAINS
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize(
    ("failure", "check"),
    [
        ({"divergent": 1}, "divergences"),
        ({"random_walk_energy": True}, "bfmi"),
        ({"saturated": 81}, "tree_depth"),  # 81 of 4000 draws: just over 2%
        ({"saturated": 40}, "tree_depth"),  # exactly 1%: not "under 1%"
    ],
)
def test_one_failing_sample_stat_fails_only_its_check(failure, check):
    record = diagnostics_record(fake_idata(SEED, **failure))

    assert record["meets_req18"] is False
    assert {name for name, passed in record["checks"].items() if not passed} == {check}


@pytest.mark.parametrize(
    ("failure", "checks"),
    [
        ({"shift_chain": 1.0}, {"rhat"}),
        ({"ar": 0.99}, {"ess_bulk", "ess_tail"}),
    ],
)
def test_poorly_mixed_draws_fail_their_checks(failure, checks):
    record = diagnostics_record(fake_idata(SEED, **failure))

    assert record["meets_req18"] is False
    assert all(record["checks"][name] is False for name in checks)
    # R-hat and ESS share the between-chain variance, so they fail together; the sample
    # statistics are untouched.
    assert all(record["checks"][name] is True for name in SAMPLE_STAT_CHECKS)


def test_rank_histogram_shows_a_shifted_chain_in_the_upper_ranks():
    record = diagnostics_record(fake_idata(SEED, shift_chain=1.0))
    counts = np.array(record["rank_histograms"]["counts"]["alpha"])
    half = counts.shape[1] // 2

    assert counts[0, half:].sum() > 3 * counts[0, :half].sum()
    assert counts.sum() == FAKE_CHAINS * FAKE_DRAWS


def test_a_nan_statistic_fails_its_check_and_serializes_as_null():
    idata = fake_idata(SEED)
    idata["posterior"]["alpha"] = xr.full_like(idata["posterior"]["alpha"], 1.0)
    record = diagnostics_record(idata)

    assert record["variables"]["alpha"]["rhat"] is None
    assert record["checks"]["rhat"] is False
    assert record["worst"]["rhat_max"] is None
    json.dumps(record, allow_nan=False)


def test_saturation_is_measured_against_the_given_tree_depth():
    idata = fake_idata(SEED)  # every draw takes 7 steps: depth 3 is saturated
    record = diagnostics_record(idata, Thresholds(max_tree_depth=3))

    assert record["tree_depth"]["saturated_share"] == 1.0
    assert record["checks"]["tree_depth"] is False


def test_diagnostics_record_requires_the_nuts_sample_stats():
    idata = fake_idata(SEED)
    idata["sample_stats"] = xr.DataTree(
        idata["sample_stats"].to_dataset().drop_vars("step_size")
    )
    with pytest.raises(ValueError, match="step_size"):
        diagnostics_record(idata)


# ---- log likelihood blocks and the netCDF file ----


@pytest.fixture(scope="module")
def blocked(panel, reference_months, benchmark_year):
    samples = near_truth_samples(2, 25, SEED)
    idata = az.from_dict({"posterior": samples})
    observed_any = ~np.isnan(np.asarray(panel)).all(axis=1)
    step_ll = step_log_likelihood_draws(step_fn_for(panel), samples, batch_size=7)
    tree = attach_log_likelihood(
        idata, step_ll, reference_months, benchmark_year, observed_any
    )
    return idata, tree, samples, step_ll, observed_any


def test_step_log_likelihood_draws_match_the_filter_draw_by_draw(panel, blocked):
    _, _, samples, step_ll, _ = blocked

    assert step_ll.shape == (2, 25, NUM_STEPS)
    for chain, draw in ((0, 0), (1, 24)):
        params = {name: value[chain, draw] for name, value in samples.items()}
        expected = kalman_filter(pilot_ssm(params), panel).step_log_likelihood
        np.testing.assert_allclose(step_ll[chain, draw], expected, rtol=1e-12, atol=0)


def test_log_likelihood_blocks_have_the_observed_months_and_benchmark_years(
    blocked, reference_months, benchmark_year
):
    idata, tree, _, _, observed_any = blocked
    months = tree["log_likelihood"]["panel_month"]
    years = tree["log_likelihood"]["panel_benchmark_year"]

    assert "log_likelihood" not in idata.children  # the input tree is not modified
    assert months.dims == ("chain", "draw", "reference_month")
    assert years.dims == ("chain", "draw", "benchmark_year")
    assert months.sizes["reference_month"] == NUM_STEPS - 1 == observed_any.sum()
    lapse = np.datetime64(reference_months[LAPSE_STEP], "ns")
    assert lapse not in months["reference_month"].values
    assert years["benchmark_year"].dtype == np.int64
    assert years["benchmark_year"].values.tolist() == sorted(
        set(benchmark_year.tolist())
    )


def test_both_blocks_sum_to_the_factor_log_likelihood_at_every_draw(panel, blocked):
    _, tree, samples, _, _ = blocked
    expected = factor_log_likelihood(panel, samples)
    group = tree["log_likelihood"]

    np.testing.assert_allclose(
        group["panel_month"].sum("reference_month"), expected, rtol=0, atol=1e-8
    )
    np.testing.assert_allclose(
        group["panel_benchmark_year"].sum("benchmark_year"), expected, rtol=0, atol=1e-8
    )


def test_loo_runs_on_both_blocks(blocked):
    _, tree, _, _, _ = blocked

    for name, size in (("panel_month", NUM_STEPS - 1), ("panel_benchmark_year", 11)):
        loo = az.loo(tree, var_name=name)
        assert np.isfinite(loo.elpd), name
        assert len(loo.pareto_k.values) == size, name


def test_a_dropped_step_must_have_zero_log_likelihood(
    blocked, reference_months, benchmark_year
):
    idata, _, _, step_ll, observed_any = blocked
    wrongly_dropped = observed_any.copy()
    wrongly_dropped[0] = False

    with pytest.raises(ValueError, match="nonzero log likelihood"):
        attach_log_likelihood(
            idata, step_ll, reference_months, benchmark_year, wrongly_dropped
        )


def test_inference_data_round_trips_groups_values_dtypes_and_coords(blocked, tmp_path):
    _, tree, _, _, _ = blocked
    tree = tree.copy()
    tree["sample_stats"] = fake_idata(SEED)["sample_stats"].isel(
        chain=slice(2), draw=slice(25)
    )
    path = tmp_path / "inference_data.nc"
    write_inference_data(tree, path)
    back = read_inference_data(path)

    assert back.groups == tree.groups
    for group in tree.groups[1:]:
        original, read = tree[group].to_dataset(), back[group].to_dataset()
        xr.testing.assert_equal(read, original)
        for name in (*original.data_vars, *original.coords):
            assert read[name].dtype == original[name].dtype, (group, name)
    assert back["sample_stats"]["n_steps"].dtype == np.int64
    assert back["log_likelihood"]["benchmark_year"].dtype == np.int64
    assert back["log_likelihood"]["reference_month"].dtype == np.dtype("datetime64[ns]")


# ---- provenance ----


def test_git_state_reads_the_checkout_and_refuses_a_directory_outside_git(tmp_path):
    state = provenance.git_state(vintage_raw.ROOT)
    head = subprocess.run(
        ["git", "-C", str(vintage_raw.ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert state["commit"] == head
    assert len(head) == 40 and set(head) <= set("0123456789abcdef")
    assert isinstance(state["dirty"], bool)
    with pytest.raises(subprocess.CalledProcessError):
        provenance.git_state(tmp_path)


def test_library_versions_and_host_record():
    versions = provenance.library_versions()
    host = provenance.host_record()

    for name in (*provenance.LIBRARIES, "jax", "jaxlib"):
        assert versions[name] == importlib.metadata.version(name), name
    assert versions["python"] == ".".join(map(str, sys.version_info[:3]))
    assert host["jax_backend"] == jax.default_backend()
    assert host["device_count"] == jax.device_count()
    assert set(host) >= {"platform", "machine", "device_kinds", "xla_flags"}


def test_data_inputs_hash_the_source_manifests_and_read_built_manifests(tmp_path):
    root = vintage_raw.ROOT
    record = provenance.data_inputs(root)
    for stage, path in provenance.SOURCE_MANIFESTS.items():
        digest = hashlib.sha256((root / path).read_bytes()).hexdigest()
        assert record["source_manifest_sha256"][stage] == digest
    for stage, path in provenance.ARTIFACT_MANIFESTS.items():
        built = (
            json.loads((root / path).read_text()) if (root / path).is_file() else None
        )
        assert record["artifact_manifests"][stage] == built

    copy_source_manifests(tmp_path)
    unbuilt = provenance.data_inputs(tmp_path)
    assert unbuilt["source_manifest_sha256"] == record["source_manifest_sha256"]
    assert unbuilt["artifact_manifests"] == {"stage3": None, "stage4": None}


def test_operator_test_record_is_the_whole_json_file_and_required(tmp_path):
    path = tmp_path / provenance.OPERATOR_TEST_RECORD
    with pytest.raises(FileNotFoundError, match="operator_artifacts.py build"):
        provenance.operator_test_record(tmp_path)
    written = {"schema_version": 1, "checks": {"wedge": {"years": 23}}}
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(written))

    assert provenance.operator_test_record(tmp_path) == written


def test_array_sha256_is_stable_and_sensitive_to_names_dtypes_shapes_and_values():
    y = np.arange(12.0).reshape(3, 4)
    flags = np.array([True, False, True])
    digest = provenance.array_sha256({"y": y, "flags": flags})

    assert digest == provenance.array_sha256({"flags": flags.copy(), "y": y.copy()})
    assert digest == provenance.array_sha256(
        {"y": np.asfortranarray(y), "flags": flags}
    )
    changed_value = y.copy()
    changed_value[2, 3] += 1e-12
    for variant in (
        {"z": y, "flags": flags},
        {"y": y.astype(np.float32), "flags": flags},
        {"y": y.reshape(4, 3), "flags": flags},
        {"y": changed_value, "flags": flags},
    ):
        assert provenance.array_sha256(variant) != digest
    with pytest.raises(TypeError, match="object dtype"):
        provenance.array_sha256({"months": np.array([date(1979, 1, 1)], dtype=object)})


def test_run_provenance_records_every_section_and_serializes(panel, provenance_root):
    record = provenance.run_provenance(
        root=provenance_root,
        seeds={"run": SEED},
        model={"name": "stage1-synthetic-pilot"},
        init_record={"strategy": "test"},
        arrays={"y": np.asarray(panel)},
        settings={"num_chains": 4},
    )

    assert set(record) == {
        "created_utc",
        "git",
        "libraries",
        "host",
        "data_inputs",
        "operator_test_results",
        "seeds",
        "model",
        "settings",
        "init",
        "arrays",
    }
    assert record["operator_test_results"] == OPERATOR_RECORD
    assert record["git"] == provenance.git_state(vintage_raw.ROOT)
    assert record["data_inputs"]["artifact_manifests"] == {
        "stage3": None,
        "stage4": None,
    }
    assert record["arrays"]["sha256"] == provenance.array_sha256(
        {"y": np.asarray(panel)}
    )
    assert record["arrays"]["shapes"] == {"y": [NUM_STEPS, 4]}
    json.dumps(record, allow_nan=False)


# ---- run directories and the end-to-end run ----


def test_new_run_dir_is_named_by_label_seed_and_time_and_never_reused(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        fit, "_utc_now", lambda: datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
    )
    run_dir = new_run_dir("pilot", 7, runs_dir=tmp_path / "runs")

    assert run_dir == tmp_path / "runs" / "pilot-seed7-20261021T120000Z"
    assert run_dir.is_dir()
    with pytest.raises(FileExistsError):
        new_run_dir("pilot", 7, runs_dir=tmp_path / "runs")


def test_the_harness_imports_no_pilot_module():
    modules = ["diagnostics", "fit", "inference_data", "inits", "provenance"]
    imports = "; ".join(f"import ces_revisions.runs.{name}" for name in modules)
    check = "import sys; print(sorted(m for m in sys.modules if 'ces_revisions.pilot' in m))"
    output = subprocess.run(
        [sys.executable, "-c", f"{imports}; {check}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    assert output.strip() == "[]"


@pytest.mark.slow
def test_run_nuts_writes_a_complete_run_directory(
    panel, prior_draws, reference_months, benchmark_year, provenance_root, tmp_path
):
    settings = RunSettings(seed=SEED, num_warmup=150, num_samples=100)
    observed_any = ~np.isnan(np.asarray(panel)).all(axis=1)
    result = run_nuts(
        pilot_model,
        (panel,),
        settings=settings,
        prior_draws=prior_draws,
        step_fn=step_fn_for(panel),
        reference_months=reference_months,
        benchmark_year=benchmark_year,
        observed_any=observed_any,
        model_record={"name": "stage1-synthetic-pilot", "prior_medians": PRIOR_MEDIANS},
        arrays={"y": np.asarray(panel)},
        root=provenance_root,
    )
    run_dir = write_run(
        result, new_run_dir("synthetic-pilot", SEED, runs_dir=tmp_path / "runs")
    )
    back = read_inference_data(run_dir / fit.INFERENCE_DATA_FILE)
    written_provenance = json.loads((run_dir / fit.PROVENANCE_FILE).read_text())
    sampling = written_provenance["sampling"]
    posterior = result.idata["posterior"]

    assert sorted(path.name for path in run_dir.iterdir()) == sorted(
        [fit.DIAGNOSTICS_FILE, fit.INFERENCE_DATA_FILE, fit.PROVENANCE_FILE]
    )
    assert back.groups == result.idata.groups
    for group in result.idata.groups[1:]:
        xr.testing.assert_equal(
            back[group].to_dataset(), result.idata[group].to_dataset()
        )
    assert (
        json.loads((run_dir / fit.DIAGNOSTICS_FILE).read_text()) == result.diagnostics
    )
    assert set(posterior.data_vars) == set(PRIOR_MEDIANS)
    assert posterior.sizes == {"chain": 4, "draw": 100}
    assert written_provenance["operator_test_results"] == OPERATOR_RECORD
    assert written_provenance["settings"]["chain_method"] == chain_method(4)
    assert written_provenance["settings"]["max_tree_depth"] == 10
    assert len(written_provenance["init"]["initial_potential_energy"]) == 4
    assert sampling["timings"]["sampling_seconds"] == result.sampling_seconds > 0
    (mass,) = sampling["inverse_mass_matrix"]
    assert mass["sites"] == sorted(PRIOR_MEDIANS)
    assert np.shape(mass["per_chain"]) == (4, len(PRIOR_MEDIANS))
    np.testing.assert_allclose(mass["mean"], np.mean(mass["per_chain"], axis=0))
    assert len(sampling["final_step_size"]) == 4
    assert result.diagnostics["thresholds"] == asdict(REQ18)
    samples = {name: np.asarray(posterior[name]) for name in PRIOR_MEDIANS}
    np.testing.assert_allclose(
        result.idata["log_likelihood"]["panel_month"].sum("reference_month"),
        factor_log_likelihood(panel, samples),
        rtol=0,
        atol=1e-8,
    )
```

- [ ] **Step 5: Run them to verify they fail**

Run: `uv run pytest tests/test_run_harness.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'ces_revisions.runs'`.

- [ ] **Step 6: Write the harness package**

Create `src/ces_revisions/runs/__init__.py`:

```python
"""The model-agnostic NUTS run harness: starts, sampling, diagnostics, and provenance.

``inits`` places dispersed chain starts from the caller's prior draws, ``fit`` runs NUTS
and writes a run directory, ``diagnostics`` checks the draws against Req 18's MCMC
thresholds, ``inference_data`` attaches the per-month and per-benchmark-year log
likelihood that blocked PSIS-LOO reads and stores the DataTree as netCDF, and
``provenance`` records the code, data, library, and host state of a run. Nothing here
knows a particular model: a model is a NumPyro function, its arguments, and a function
from a parameter dict to its per-step log likelihood.
"""
```

Create `src/ces_revisions/runs/inits.py`:

```python
"""Dispersed NUTS starting points, placed from the caller's prior draws.

Req 18 asks for four dispersed chains. Each chain starts at a different point of the prior's
bulk in NUTS's unconstrained space: site by site, the prior median plus a fixed multiple of
the prior SD there. The multiples are permuted independently for each site, so no chain
starts in the same tail of every site at once, and the R-hat that later compares the chains
compares starts that disagree.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from numpyro.distributions import biject_to
from numpyro.infer import NUTS

# Prior SDs from the unconstrained prior median at which the four chains start.
DISPERSION_OFFSETS = (-1.5, -0.5, 0.5, 1.5)


def chain_offsets(num_chains: int) -> np.ndarray:
    """DISPERSION_OFFSETS for four chains, else num_chains evenly spaced in [-1.5, 1.5]."""
    if num_chains < 2:
        raise ValueError(f"dispersed starts need at least two chains; got {num_chains}")
    if num_chains == len(DISPERSION_OFFSETS):
        return np.array(DISPERSION_OFFSETS)
    return np.linspace(DISPERSION_OFFSETS[0], DISPERSION_OFFSETS[-1], num_chains)


def dispersed_init_params(
    model,
    model_args: tuple,
    prior_draws: dict[str, np.ndarray],
    num_chains: int,
    seed: int,
    *,
    scale: float = 1.0,
) -> tuple[dict[str, jax.Array], dict]:
    """Unconstrained starts, (num_chains, ...) per sampled site, and their JSON record.

    ``prior_draws`` maps every sampled site to constrained draws from its prior, with the
    draws on the leading axis. Chain c starts at median + offset * SD * scale, where the
    median and SD are taken over the draws in unconstrained space and the offsets are a
    seeded per-site permutation of ``chain_offsets(num_chains)``. Raises ValueError unless
    every chain's initial potential energy is finite.
    """
    offsets = chain_offsets(num_chains)
    supports = sample_site_supports(model, model_args, seed)
    if set(prior_draws) != set(supports):
        raise ValueError(
            f"prior_draws must hold exactly the sampled sites {sorted(supports)}; "
            f"got {sorted(prior_draws)}"
        )
    rng = np.random.default_rng(seed)
    init_params, sites = {}, {}
    for name in sorted(supports):
        transform = biject_to(supports[name])
        unconstrained = np.asarray(transform.inv(jnp.asarray(prior_draws[name])))
        median = np.median(unconstrained, axis=0)
        sd = np.std(unconstrained, axis=0)
        # Identical draws give a rounding-error SD, not zero, so test the spread itself.
        spread = np.ptp(unconstrained, axis=0)
        if not np.all(np.isfinite(median) & np.isfinite(sd) & (spread > 0)):
            raise ValueError(
                f"site {name!r} needs finite, not all equal, unconstrained prior draws; "
                f"got median {median} and SD {sd}"
            )
        site_offsets = rng.permutation(offsets)
        chain_axis = site_offsets.reshape(num_chains, *(1,) * median.ndim)
        starts = median + chain_axis * sd * scale
        init_params[name] = jnp.asarray(starts)
        sites[name] = {
            "prior_median_unconstrained": median.tolist(),
            "prior_sd_unconstrained": sd.tolist(),
            "offsets": site_offsets.tolist(),
            "unconstrained": starts.tolist(),
            "constrained": np.asarray(transform(init_params[name])).tolist(),
        }

    started = time.perf_counter()
    energies = initial_potential_energies(model, model_args, init_params, seed)
    readback_seconds = time.perf_counter() - started
    if not np.all(np.isfinite(energies)):
        raise ValueError(
            f"every chain must start at a finite potential energy; got {energies.tolist()}"
        )
    record = {
        "strategy": "dispersed_prior_quantiles",
        "offsets": offsets.tolist(),
        "scale": float(scale),
        "seed": int(seed),
        "num_chains": num_chains,
        "num_prior_draws": {name: len(prior_draws[name]) for name in sorted(supports)},
        "sites": sites,
        "initial_potential_energy": energies.tolist(),
        "readback_seconds": readback_seconds,
    }
    return init_params, record


def sample_site_supports(model, model_args: tuple, seed: int) -> dict:
    """The support of each unobserved sample site, from one seeded trace of the model."""
    trace = handlers.trace(handlers.seed(model, rng_seed=seed)).get_trace(*model_args)
    return {
        name: site["fn"].support
        for name, site in trace.items()
        if site["type"] == "sample" and not site["is_observed"]
    }


def initial_potential_energies(
    model, model_args: tuple, init_params: dict[str, jax.Array], seed: int
) -> np.ndarray:
    """Each chain's potential energy at its start, read back from NUTS's own init.

    A throwaway kernel: a batched ``init`` vmaps the kernel's sample function in place,
    so the kernel that samples must be a fresh one.
    """
    num_chains = len(next(iter(init_params.values())))
    keys = jax.random.split(jax.random.PRNGKey(seed), num_chains)
    state = NUTS(model).init(keys, 0, init_params, model_args, {})
    return np.asarray(state.potential_energy)
```

Create `src/ces_revisions/runs/diagnostics.py`:

```python
"""Req 18's MCMC diagnostics as one JSON-safe record.

Req 18 asks for rank-normalized R-hat below 1.01, bulk and tail ESS above 400, zero
post-warmup divergences, acceptable energy-BFMI, and no systematic tree-depth saturation.
The last two are numeric here as in the Stage 1 synthetic pilot's test: every chain's
energy-BFMI above 0.3, and under 1% of draws at the maximum tree depth. Every check is a
comparison that a NaN fails, so a statistic ArviZ cannot compute never passes.
"""

import math
from dataclasses import asdict, dataclass

import arviz as az
import numpy as np
from scipy.stats import rankdata

RHAT_METHOD = "rank"  # ArviZ's rank-normalized split R-hat, as Req 18 names it
SAMPLE_STATS = ("diverging", "energy", "n_steps", "acceptance_rate", "step_size")


@dataclass(frozen=True)
class Thresholds:
    rhat_max: float = 1.01
    ess_min: float = 400.0
    bfmi_min: float = 0.3
    max_tree_depth: int = 10  # the sampler's own setting: saturation is relative to it
    saturated_share_max: float = 0.01
    divergences_max: int = 0


REQ18 = Thresholds()


def diagnostics_record(
    idata, thresholds: Thresholds = REQ18, *, rank_bins: int = 20
) -> dict:
    """R-hat, ESS, BFMI, divergences, tree depth, and rank histograms, with Req 18 checks.

    Each element of a non-scalar posterior variable is reported as ``name[i,j]``. The
    rank histograms count, per chain, the draws whose rank among all chains' pooled draws
    falls in each of ``rank_bins`` equal bins; a well-mixed chain fills them evenly.
    """
    posterior, stats = idata["posterior"], idata["sample_stats"]
    missing = [name for name in SAMPLE_STATS if name not in stats.data_vars]
    if missing:
        raise ValueError(f"sample_stats lacks {missing}")
    if not posterior.data_vars:
        raise ValueError("the posterior has no variables")
    rhat = az.rhat(idata, method=RHAT_METHOD)
    ess_bulk = az.ess(idata, method="bulk")
    ess_tail = az.ess(idata, method="tail")
    variables, rank_counts = {}, {}
    for name in posterior.data_vars:
        draws = np.asarray(posterior[name])
        for index in np.ndindex(draws.shape[2:]):
            element = f"{name}[{','.join(map(str, index))}]" if index else name
            variables[element] = {
                "rhat": _json_float(np.asarray(rhat[name])[index]),
                "ess_bulk": _json_float(np.asarray(ess_bulk[name])[index]),
                "ess_tail": _json_float(np.asarray(ess_tail[name])[index]),
            }
            rank_counts[element] = _rank_histogram(draws[(..., *index)], rank_bins)

    diverging = np.asarray(stats["diverging"]).astype(bool)
    n_steps = np.asarray(stats["n_steps"])
    saturated = n_steps >= 2**thresholds.max_tree_depth - 1
    bfmi = [_json_float(value) for value in np.asarray(az.bfmi(idata)["energy"])]
    record = {
        "thresholds": asdict(thresholds),
        "rhat_method": RHAT_METHOD,
        "num_chains": int(diverging.shape[0]),
        "num_draws": int(diverging.shape[1]),
        "variables": variables,
        "bfmi": bfmi,
        "divergences": {
            "total": int(diverging.sum()),
            "per_chain": diverging.sum(axis=1).tolist(),
        },
        "tree_depth": {
            "saturation_steps": 2**thresholds.max_tree_depth - 1,
            "saturated_share": float(saturated.mean()),
            "saturated_per_chain": saturated.sum(axis=1).tolist(),
            "n_steps_max": int(n_steps.max()),
            "n_steps_mean_per_chain": n_steps.mean(axis=1).tolist(),
        },
        "acceptance_rate": np.asarray(stats["acceptance_rate"]).mean(axis=1).tolist(),
        "step_size": np.asarray(stats["step_size"])[:, -1].tolist(),
        "rank_histograms": {"bins": rank_bins, "counts": rank_counts},
    }
    statistics = {
        key: [entry[key] for entry in variables.values()]
        for key in ("rhat", "ess_bulk", "ess_tail")
    }
    checks = {
        "rhat": _all_below(statistics["rhat"], thresholds.rhat_max),
        "ess_bulk": _all_above(statistics["ess_bulk"], thresholds.ess_min),
        "ess_tail": _all_above(statistics["ess_tail"], thresholds.ess_min),
        "bfmi": _all_above(bfmi, thresholds.bfmi_min),
        "divergences": record["divergences"]["total"] <= thresholds.divergences_max,
        "tree_depth": record["tree_depth"]["saturated_share"]
        < thresholds.saturated_share_max,
    }
    record["worst"] = {
        "rhat_max": _worst(statistics["rhat"], max),
        "ess_bulk_min": _worst(statistics["ess_bulk"], min),
        "ess_tail_min": _worst(statistics["ess_tail"], min),
        "bfmi_min": _worst(bfmi, min),
    }
    record["checks"] = checks
    record["meets_req18"] = all(checks.values())
    return record


def _rank_histogram(draws: np.ndarray, bins: int) -> list[list[int]]:
    """Per-chain counts of pooled ranks in equal bins; draws is (chain, draw)."""
    ranks = rankdata(draws, axis=None).reshape(draws.shape)  # 1..N, ties averaged
    bin_index = np.minimum(((ranks - 1) * bins / ranks.size).astype(np.int64), bins - 1)
    return [np.bincount(chain, minlength=bins).tolist() for chain in bin_index]


def _json_float(value) -> float | None:
    """A Python float, or None for NaN or infinity, so json.dumps(allow_nan=False) works."""
    value = float(value)
    return value if math.isfinite(value) else None


def _all_below(values: list[float | None], bound: float) -> bool:
    return all(value is not None and value < bound for value in values)


def _all_above(values: list[float | None], bound: float) -> bool:
    return all(value is not None and value > bound for value in values)


def _worst(values: list[float | None], pick) -> float | None:
    if any(value is None for value in values):
        return None
    return pick(values)
```

Create `src/ces_revisions/runs/inference_data.py`:

```python
"""Pointwise log likelihood for blocked PSIS-LOO, and the DataTree's netCDF file.

The Kalman factor is one site, so ``az.from_numpyro`` sees one log likelihood per draw. The
filter's per-step terms ``log p(y_t | y_<t)`` sum to it exactly, so recomputing them at every
draw gives LOO two blockings with the same total: one block per reference month, and one
per benchmark year (Req 10's April-to-March window). A step with no observed cell adds an
exact zero and is dropped from the month blocking.
"""

from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

NETCDF_ENGINE = "h5netcdf"


def step_log_likelihood_draws(
    step_fn: Callable[[dict[str, jax.Array]], jax.Array],
    samples_by_chain: dict[str, np.ndarray],
    *,
    batch_size: int = 500,
) -> np.ndarray:
    """step_fn at every draw, as a (chain, draw, T) array.

    ``samples_by_chain`` holds (chain, draw, ...) arrays, as from
    ``mcmc.get_samples(group_by_chain=True)``. The draws are flattened and mapped in
    vmapped batches of ``batch_size`` under one jit, which bounds memory on a GPU.
    """
    leading = {tuple(np.shape(value)[:2]) for value in samples_by_chain.values()}
    if len(leading) != 1:
        raise ValueError(f"every site needs the same (chain, draw) axes; got {leading}")
    ((num_chains, num_draws),) = leading
    flat = {
        name: jnp.reshape(
            jnp.asarray(value), (num_chains * num_draws, *np.shape(value)[2:])
        )
        for name, value in samples_by_chain.items()
    }
    mapped = jax.jit(lambda params: jax.lax.map(step_fn, params, batch_size=batch_size))
    values = np.asarray(mapped(flat), dtype=np.float64)
    return values.reshape(num_chains, num_draws, -1)


def attach_log_likelihood(
    idata,
    step_log_likelihood: np.ndarray,
    reference_months: Sequence[date],
    benchmark_year: np.ndarray,
    observed_any: np.ndarray,
    name: str = "panel",
) -> xr.DataTree:
    """A copy of idata with a log_likelihood group blocked by month and benchmark year.

    ``{name}_month`` is (chain, draw, reference_month) over the steps with an observed
    cell, and ``{name}_benchmark_year`` is (chain, draw, benchmark_year): the sum of those
    steps within each benchmark year. Raises ValueError if a dropped step's log likelihood
    is not exactly zero, since then neither blocking would sum to the model's.
    """
    step_log_likelihood = np.asarray(step_log_likelihood, dtype=np.float64)
    observed_any = np.asarray(observed_any, dtype=bool)
    benchmark_year = np.asarray(benchmark_year, dtype=np.int64)
    posterior = idata["posterior"]
    num_chains, num_draws = posterior.sizes["chain"], posterior.sizes["draw"]
    num_steps = len(reference_months)
    if step_log_likelihood.shape != (num_chains, num_draws, num_steps):
        raise ValueError(
            f"step_log_likelihood must be ({num_chains}, {num_draws}, {num_steps}); "
            f"got {step_log_likelihood.shape}"
        )
    if observed_any.shape != (num_steps,) or benchmark_year.shape != (num_steps,):
        raise ValueError(f"observed_any and benchmark_year must be ({num_steps},)")
    if np.any(step_log_likelihood[..., ~observed_any] != 0.0):
        raise ValueError("a step without an observed cell has a nonzero log likelihood")

    month_values = step_log_likelihood[..., observed_any]
    observed_years = benchmark_year[observed_any]
    years = np.unique(observed_years)
    year_values = np.stack(
        [month_values[..., observed_years == year].sum(axis=-1) for year in years],
        axis=-1,
    )
    months = np.array(
        [np.datetime64(month, "ns") for month in reference_months],
        dtype="datetime64[ns]",
    )[observed_any]
    sample_coords = {
        "chain": posterior["chain"].values,
        "draw": posterior["draw"].values,
    }
    log_likelihood = xr.Dataset(
        {
            f"{name}_month": (("chain", "draw", "reference_month"), month_values),
            f"{name}_benchmark_year": (
                ("chain", "draw", "benchmark_year"),
                year_values,
            ),
        },
        coords={**sample_coords, "reference_month": months, "benchmark_year": years},
    )
    tree = idata.copy()
    tree["log_likelihood"] = xr.DataTree(log_likelihood)
    return tree


def write_inference_data(idata, path: Path) -> None:
    """Write every group of the DataTree to one netCDF file through h5netcdf."""
    idata.to_netcdf(path, engine=NETCDF_ENGINE)


def read_inference_data(path: Path) -> xr.DataTree:
    """Read a DataTree written by write_inference_data fully into memory."""
    with xr.open_datatree(path, engine=NETCDF_ENGINE) as tree:
        return tree.load()
```

Create `src/ces_revisions/runs/provenance.py`:

```python
"""What a run needs recorded to be repeated: code, data, libraries, host, and arrays.

The record names the commit and whether the working tree differed from it, the source
manifests and built-artifact manifests the run's inputs came from, the operator test record
that every run copies (scripts/operator_artifacts.py writes it), the exact library and host
it ran on, and a hash of the arrays the model saw.
"""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import jax
import numpy as np

from ces_revisions.devices import jax_versions, nvidia_driver
from ces_revisions.vintages import raw as vintage_raw

LIBRARIES = (
    "numpyro",
    "arviz",
    "arviz-base",
    "arviz-stats",
    "xarray",
    "numpy",
    "polars",
    "h5netcdf",
    "h5py",
)
SOURCE_MANIFESTS = {
    "stage3": Path("data/raw/manifest.csv"),
    "stage4": Path("data/annual/raw/manifest.csv"),
}
ARTIFACT_MANIFESTS = {
    "stage3": Path("data/panel/manifest.json"),
    "stage4": Path("data/annual/panel/manifest.json"),
}
OPERATOR_TEST_RECORD = Path("data/operators/operator-test-results.json")


def git_state(root: Path) -> dict:
    """The checked-out commit and whether tracked or untracked files differ from it."""
    commit = _git(root, "rev-parse", "HEAD").strip()
    changes = _git(root, "status", "--porcelain", "--untracked-files=normal")
    return {"commit": commit, "dirty": bool(changes.strip())}


def library_versions() -> dict:
    """Python, JAX (with any CUDA plugin), and the inference, data, and storage stack."""
    versions = {"python": platform.python_version(), **jax_versions()}
    versions |= {name: importlib.metadata.version(name) for name in LIBRARIES}
    return dict(sorted(versions.items()))


def host_record() -> dict:
    """The platform, JAX's backend and devices, the NVIDIA driver, and XLA_FLAGS."""
    devices = jax.devices()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_executable": sys.executable,
        "jax_backend": jax.default_backend(),
        "device_kinds": sorted({device.device_kind for device in devices}),
        "device_count": len(devices),
        "nvidia_driver": nvidia_driver(),
        "xla_flags": os.environ.get("XLA_FLAGS"),
    }


def data_inputs(root: Path) -> dict:
    """SHA-256 of the committed source manifests, and the built-artifact manifests if any."""
    return {
        "source_manifest_sha256": {
            stage: vintage_raw.file_sha256(root / path)
            for stage, path in SOURCE_MANIFESTS.items()
        },
        "artifact_manifests": {
            stage: json.loads((root / path).read_text())
            if (root / path).is_file()
            else None
            for stage, path in ARTIFACT_MANIFESTS.items()
        },
    }


def operator_test_record(root: Path) -> dict:
    """The whole Stage 5 operator test record; every run copies it into its provenance."""
    path = root / OPERATOR_TEST_RECORD
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; rebuild it with "
            "`uv run python scripts/operator_artifacts.py build`"
        )
    return json.loads(path.read_text())


def array_sha256(arrays: dict[str, np.ndarray]) -> str:
    """One SHA-256 over the arrays' sorted names, dtypes, shapes, and C-order bytes."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        if array.dtype.hasobject:
            raise TypeError(
                f"array {name!r} has object dtype; its bytes are not its values"
            )
        header = json.dumps([name, array.dtype.str, list(array.shape)])
        digest.update(header.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def run_provenance(
    *,
    root: Path,
    seeds: dict,
    model: dict,
    init_record: dict,
    arrays: dict[str, np.ndarray],
    settings: dict,
) -> dict:
    """The run's provenance record; JSON-serializable when its arguments are."""
    return {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_state(root),
        "libraries": library_versions(),
        "host": host_record(),
        "data_inputs": data_inputs(root),
        "operator_test_results": operator_test_record(root),
        "seeds": seeds,
        "model": model,
        "settings": settings,
        "init": init_record,
        "arrays": {
            "sha256": array_sha256(arrays),
            "shapes": {name: list(np.shape(arrays[name])) for name in sorted(arrays)},
            "dtypes": {
                name: np.asarray(arrays[name]).dtype.str for name in sorted(arrays)
            },
        },
    }


def _git(root: Path, *args: str) -> str:
    command = ["git", "-C", str(root), *args]
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout
```

Create `src/ces_revisions/runs/fit.py`:

```python
"""Run NUTS on a model and write its run directory.

A run starts each chain at a dispersed point placed from the caller's prior draws, samples
with NumPyro's NUTS at the run's tree depth and target acceptance, attaches the blocked log
likelihood to ArviZ's DataTree, checks the draws against Req 18, and records its provenance.
Provenance that reads the checkout is taken before sampling, so a missing input fails the
run before the expensive part. ``write_run`` stores the DataTree, the diagnostics, and the
provenance in a new directory that ``new_run_dir`` names and never reuses.
"""

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path

import arviz as az
import jax
import numpy as np
from numpyro.infer import MCMC, NUTS

from ces_revisions.devices import chain_method
from ces_revisions.runs.diagnostics import REQ18, diagnostics_record
from ces_revisions.runs.inference_data import (
    attach_log_likelihood,
    step_log_likelihood_draws,
    write_inference_data,
)
from ces_revisions.runs.inits import dispersed_init_params
from ces_revisions.runs.provenance import run_provenance
from ces_revisions.vintages import raw as vintage_raw

RUNS_DIR = vintage_raw.ROOT / "data" / "runs"
EXTRA_FIELDS = (
    "diverging",
    "energy",
    "num_steps",
    "accept_prob",
    "adapt_state.step_size",
    "potential_energy",
)
INFERENCE_DATA_FILE = "inference_data.nc"
DIAGNOSTICS_FILE = "diagnostics.json"
PROVENANCE_FILE = "provenance.json"


@dataclass(frozen=True)
class RunSettings:
    seed: int
    num_chains: int = 4
    num_warmup: int = 1000
    num_samples: int = 1000
    max_tree_depth: int = 10
    target_accept_prob: float = 0.8


@dataclass(frozen=True)
class RunResult:
    idata: object  # xarray DataTree
    diagnostics: dict
    provenance: dict
    sampling_seconds: float  # compilation, warmup, and sampling, until the draws exist


def run_nuts(
    model,
    model_args: tuple,
    *,
    settings: RunSettings,
    prior_draws: dict,
    step_fn: Callable,
    reference_months: Sequence[date],
    benchmark_year: np.ndarray,
    observed_any: np.ndarray,
    model_record: dict,
    arrays: dict[str, np.ndarray],
    root: Path = vintage_raw.ROOT,
) -> RunResult:
    """Sample ``model(*model_args)`` with NUTS from dispersed starts; diagnose and record.

    ``step_fn`` maps a dict of sampled sites to the model's (T,) per-step log likelihood;
    ``arrays`` are the data arrays the model saw, hashed into the provenance.
    """
    method = chain_method(settings.num_chains)
    init_params, init_record = dispersed_init_params(
        model, model_args, prior_draws, settings.num_chains, settings.seed
    )
    provenance = run_provenance(
        root=root,
        seeds={"run": settings.seed, "init": settings.seed, "mcmc_key": settings.seed},
        model=model_record,
        init_record=init_record,
        arrays=arrays,
        settings={
            **asdict(settings),
            "chain_method": method,
            "extra_fields": list(EXTRA_FIELDS),
        },
    )
    mcmc = MCMC(
        NUTS(
            model,
            max_tree_depth=settings.max_tree_depth,
            target_accept_prob=settings.target_accept_prob,
        ),
        num_warmup=settings.num_warmup,
        num_samples=settings.num_samples,
        num_chains=settings.num_chains,
        chain_method=method,
        progress_bar=False,
    )
    started_utc = datetime.now(UTC)
    started = time.perf_counter()
    mcmc.run(
        jax.random.PRNGKey(settings.seed),
        *model_args,
        init_params=init_params,
        extra_fields=EXTRA_FIELDS,
    )
    samples = jax.block_until_ready(mcmc.get_samples(group_by_chain=True))
    sampling_seconds = time.perf_counter() - started
    finished_utc = datetime.now(UTC)

    idata = az.from_numpyro(mcmc).map_over_datasets(lambda ds: ds.as_numpy())
    started = time.perf_counter()
    step_log_likelihood = step_log_likelihood_draws(step_fn, samples)
    log_likelihood_seconds = time.perf_counter() - started
    idata = attach_log_likelihood(
        idata, step_log_likelihood, reference_months, benchmark_year, observed_any
    )
    thresholds = replace(REQ18, max_tree_depth=settings.max_tree_depth)
    diagnostics = diagnostics_record(idata, thresholds)

    adapt_state = mcmc.last_state.adapt_state
    num_steps = np.asarray(mcmc.get_extra_fields(group_by_chain=True)["num_steps"])
    provenance["sampling"] = {
        "timings": {
            "init_readback_seconds": init_record["readback_seconds"],
            "sampling_seconds": sampling_seconds,
            "sampling_seconds_covers": "compilation, warmup, and sampling",
            "log_likelihood_seconds": log_likelihood_seconds,
            "started_utc": started_utc.isoformat(timespec="seconds"),
            "finished_utc": finished_utc.isoformat(timespec="seconds"),
        },
        "num_steps_sampling_total": int(num_steps.sum()),
        "inverse_mass_matrix": [
            {
                "sites": list(sites),
                "per_chain": np.asarray(diagonal).tolist(),
                "mean": np.asarray(diagonal).mean(axis=0).tolist(),
            }
            for sites, diagonal in adapt_state.inverse_mass_matrix.items()
        ],
        "final_step_size": np.asarray(adapt_state.step_size).tolist(),
    }
    return RunResult(
        idata=idata,
        diagnostics=diagnostics,
        provenance=provenance,
        sampling_seconds=sampling_seconds,
    )


def write_run(result: RunResult, run_dir: Path) -> Path:
    """Write the DataTree first, then the diagnostics and provenance JSON; return run_dir."""
    write_inference_data(result.idata, run_dir / INFERENCE_DATA_FILE)
    for name, record in (
        (DIAGNOSTICS_FILE, result.diagnostics),
        (PROVENANCE_FILE, result.provenance),
    ):
        text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False)
        (run_dir / name).write_text(text + "\n")
    return run_dir


def new_run_dir(label: str, seed: int, runs_dir: Path = RUNS_DIR) -> Path:
    """Create runs_dir/<label>-seed<seed>-<UTC timestamp>; raise if it already exists."""
    stamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    run_dir = runs_dir / f"{label}-seed{seed}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _utc_now() -> datetime:
    return datetime.now(UTC)
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/test_run_harness.py -q -m "not slow"`
Expected: `31 passed, 1 deselected`.

Run: `uv run pytest tests/test_run_harness.py -q -m slow`
Expected: `1 passed`. It runs `run_nuts` end to end on the Stage 1 synthetic pilot and checks the run directory.

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff format src/ces_revisions/runs/ src/ces_revisions/devices.py src/ces_revisions/engine_probe.py tests/test_run_harness.py tests/test_stack.py
uv run ruff check src/ces_revisions/runs/ src/ces_revisions/devices.py src/ces_revisions/engine_probe.py tests/test_run_harness.py tests/test_stack.py
git add src/ces_revisions/runs/__init__.py src/ces_revisions/runs/inits.py src/ces_revisions/runs/diagnostics.py src/ces_revisions/runs/inference_data.py src/ces_revisions/runs/provenance.py src/ces_revisions/runs/fit.py src/ces_revisions/devices.py src/ces_revisions/engine_probe.py tests/test_run_harness.py tests/test_stack.py pyproject.toml uv.lock .gitignore
git commit -m "Add the model-agnostic NUTS run harness"
```

### Task 7: The April 2020 check and the SBC library

**Files:**
- Create: `src/ces_revisions/pilot/checks.py`
- Create: `src/ces_revisions/pilot/sbc.py`
- Test: `tests/test_pilot_checks.py`, `tests/test_sbc.py`, `tests/test_pilot_sbc_runner.py`

**Interfaces:**
- Consumes: `kalman_filter`, `kalman_smoother`; `NEWS_STATES`, `Layout`, `LAYOUTS`, `seasonal_selector` (Task 3); `SAMPLED_SITES`, `TARGET_ACCEPT_PROB`, `derived`, `pilot_model`, `pilot_ssm`, `sample_prior` (Task 4); `simulate_panel` (Task 5); `AggregatePanel` (Task 2).
- Produces:
  - `checks`: `APRIL_2020`, `RATIO_RANGE = (0.95, 1.05)`, `INTERVAL_Z`, `change_row(layout)`, `news_row(layout)`, `april_2020(samples, panel, layout, *, num_draws=200, seed) -> dict` (keys include `dx`, `news_sum`, `observed`, `non_finite_draws`, `ratio`, `interval_90`, `band`, `passed`), `band_check(mean, sd, third) -> dict`.
  - `sbc` math: `rank_of(prior_value, draws, rng=None) -> int`, `thin(draws, num_kept) -> np.ndarray`, `gamma_statistic(ranks, num_draws) -> float`, `gamma_critical_value(num_replicates, num_draws, alpha, *, num_simulations=10_000, seed) -> float`, `ecdf_test(ranks, num_draws, alpha, *, seed, num_simulations=10_000) -> dict`, `recovery(prior_value, draws, prior_sd) -> dict`.
  - `sbc` runner: `QUANTITIES` (23: the 17 sites, 5 derived quantities, and the log likelihood), `FIRST_MONTH = date(1999, 1, 1)`, `LAST_MONTH = date(2021, 12, 1)`, the pre-registered rule constants (`FAMILY_ALPHA`, `CONTRACTION_MIN`, `LARGE_Z`, `LARGE_Z_SHARE_MAX`), `replicate_seeds`, `prior_reference(layout_name)`, `run_replicate(index, *, base_seed, layout_name, panel, num_warmup, num_samples, num_kept, target_accept_prob=TARGET_ACCEPT_PROB) -> dict`, `run_sbc(out_dir, *, layout_name, panel, start, stop, workers, base_seed, num_warmup, num_samples, num_kept, target_accept_prob=TARGET_ACCEPT_PROB) -> dict`, `read_replicates(out_dir)`, `summarize_sbc(records) -> dict`, `write_summary(out_dir) -> dict`. On a host with an NVIDIA device `run_sbc` refuses more than one worker, because every worker would share the first GPU; one replicate per GPU comes from one `--workers 1` shard per GPU, each under its own `CUDA_VISIBLE_DEVICES`.

The April 2020 check passes only when every subsampled draw filters to finite moments and the smoothed change's whole central 90% interval lies within ±5% of the third print (planning choice 2). SBC's rule is pre-registered: every quantity passes the γ-ECDF test of Säilynoja, Bürkner, and Vehtari (2022) at the Bonferroni level `FAMILY_ALPHA / len(QUANTITIES)`, and for each news share the median contraction is at least 0.25 and at most 5% of replicates have |z| > 3.

- [ ] **Step 1: Write the failing check and SBC tests**

Create `tests/test_pilot_checks.py`:

```python
"""The April 2020 posterior check, against the dense oracle on a short crisis window."""

import dataclasses

import numpy as np
import pytest
from dense_reference import dense_reference
from pilot_synthetic import STATE_NAMES, TRUTH, crisis_panel, reference_ssm

from ces_revisions.pilot import checks
from ces_revisions.pilot.checks import (
    APRIL_2020,
    INTERVAL_Z,
    april_2020,
    band_check,
    change_row,
    news_row,
)
from ces_revisions.pilot.layout import LAYOUTS, observation_matrix
from ces_revisions.pilot.model import SAMPLED_SITES
from ces_revisions.pilot.panel import COLUMNS

SEED = sum(map(ord, "ces-revisions-stage-6-april-2020-check"))
NUM_CHAINS, NUM_DRAWS = 2, 5
NUM_KEPT = 4
# Positive sites are perturbed on the log scale, the rest additively, around TRUTH.
LOG_SITES = (
    "sigma_level",
    "sigma_slope",
    "sigma_seasonal",
    "sigma_news_s",
    "sigma_news_t",
    "sigma_news_annual",
    "sigma_anchor",
)


def fake_posterior() -> dict[str, np.ndarray]:
    """(chain, draw) arrays near TRUTH, distinct at every draw."""
    rng = np.random.default_rng(SEED)
    shape = (NUM_CHAINS, NUM_DRAWS)
    return {
        name: TRUTH[name] * np.exp(0.1 * rng.standard_normal(shape))
        if name in LOG_SITES
        else TRUTH[name] + 0.05 * rng.standard_normal(shape)
        for name in SAMPLED_SITES
    }


def oracle_rows(layout_name: str) -> np.ndarray:
    """dx and the news sum, written from the state names alone."""
    names = STATE_NAMES[layout_name]
    gamma = (
        ("seasonal_0",)
        if layout_name == "dummy"
        else tuple(
            name for name in names if name.endswith("_cos") or name == "harmonic_6"
        )
    )
    dx = np.zeros(len(names))
    dx[names.index("level")] = 1.0
    dx[[names.index(name) for name in gamma]] = 1.0
    dx[names.index("lagged_value")] = -1.0
    news = np.zeros(len(names))
    news[[names.index(name) for name in ("news_s", "news_t", "news_post")]] = 1.0
    return np.stack([dx, news])


@pytest.mark.parametrize("layout_name", sorted(LAYOUTS))
def test_change_and_news_rows(layout_name):
    layout = LAYOUTS[layout_name]
    expected = oracle_rows(layout_name)
    np.testing.assert_array_equal(change_row(layout), expected[0])
    np.testing.assert_array_equal(news_row(layout), expected[1])
    # The T print is the change less the post-T news; F is the change less all news.
    rows = observation_matrix(layout)
    news_post = np.eye(layout.state_dim)[layout.state_names.index("news_post")]
    np.testing.assert_array_equal(rows[COLUMNS.index("T")] + news_post, expected[0])
    np.testing.assert_array_equal(
        rows[COLUMNS.index("F")] + news_row(layout), expected[0]
    )


@pytest.mark.parametrize("layout_name", ["dummy", "trig4"])
def test_april_2020_matches_the_dense_smoother(layout_name):
    layout = LAYOUTS[layout_name]
    panel = crisis_panel(layout_name)
    samples = fake_posterior()
    result = april_2020(samples, panel, layout, num_draws=NUM_KEPT, seed=SEED)

    # Evenly spaced draws of the pooled (chain, draw) array: positions j * 10 // 4.
    kept = [0, 2, 5, 7]
    assert result["draw_indices"] == kept
    assert result["num_draws"] == NUM_KEPT
    assert result["non_finite_draws"] == 0
    step = panel.reference_months.index(APRIL_2020)
    rows = oracle_rows(layout_name)
    means, variances = [], []
    for index in kept:
        params = {
            name: float(values.reshape(-1)[index]) for name, values in samples.items()
        }
        smoothed = dense_reference(reference_ssm(params, panel, layout_name), panel.y)
        means.append(rows @ smoothed["smoothed_mean"][step])
        variances.append(np.diag(rows @ smoothed["smoothed_cov"][step] @ rows.T))
    means, variances = np.array(means), np.array(variances)
    for k, name in enumerate(("dx", "news_sum")):
        between = means[:, k].var()
        np.testing.assert_allclose(result[name]["mean"], means[:, k].mean(), rtol=1e-8)
        np.testing.assert_allclose(
            result[name]["sd"], np.sqrt(variances[:, k].mean() + between), rtol=1e-8
        )
        np.testing.assert_allclose(
            result[name]["sd_between_draws"], np.sqrt(between), rtol=1e-6
        )

    observed = panel.y[step, :3]
    assert result["observed"] == dict(
        zip(("F", "S", "T"), observed.tolist(), strict=True)
    )
    assert result["ratio"] == pytest.approx(
        result["dx"]["mean"] / observed[2], rel=1e-15
    )
    assert result["ratio_range"] == [0.95, 1.05]
    half_width = INTERVAL_Z * result["dx"]["sd"]
    interval = [result["dx"]["mean"] - half_width, result["dx"]["mean"] + half_width]
    assert result["interval_90"] == pytest.approx(interval, rel=1e-15)
    band = sorted([0.95 * observed[2], 1.05 * observed[2]])
    assert result["band"] == pytest.approx(band, rel=1e-15)
    assert result["passed"] == (band[0] <= interval[0] and interval[1] <= band[1])


THIRD = -20_000.0  # a third print at April 2020's scale; the band is [-21000, -19000]


@pytest.mark.parametrize(
    ("mean", "sd", "passed"),
    [
        (THIRD, 100.0, True),  # interval -20164.5 to -19835.5
        (-19_200.0, 100.0, True),  # interval -19364.5 to -19035.5
        (-19_100.0, 100.0, False),  # interval upper end -18935.5 leaves the band
        (THIRD, 5_600.0, False),  # the mean passes the ratio, the interval does not
        (-18_000.0, 0.0, False),  # a point mass outside the band
    ],
)
def test_band_check_needs_the_whole_interval_inside_the_band(mean, sd, passed):
    result = band_check(mean, sd, THIRD)
    assert result["ratio"] == pytest.approx(mean / THIRD, rel=1e-15)
    assert result["band"] == [-21_000.0, -19_000.0]
    assert result["interval_90"] == pytest.approx(
        [mean - INTERVAL_Z * sd, mean + INTERVAL_Z * sd], rel=1e-15
    )
    assert result["passed"] is passed


def test_april_2020_without_a_third_print_has_no_ratio():
    layout = LAYOUTS["dummy"]
    panel = crisis_panel("dummy")
    step = panel.reference_months.index(APRIL_2020)
    y = panel.y.copy()
    y[step, COLUMNS.index("T")] = np.nan
    result = april_2020(
        fake_posterior(),
        dataclasses.replace(panel, y=y),
        layout,
        num_draws=2,
        seed=SEED,
    )
    assert result["observed"]["T"] is None
    assert result["ratio"] is None
    assert result["passed"] is False


def test_a_draw_the_filter_cannot_evaluate_fails_the_check(monkeypatch):
    layout = LAYOUTS["dummy"]
    panel = crisis_panel("dummy")
    passing = {"ratio": 1.0, "interval_90": [0.0, 0.0], "band": [0.0, 0.0]}
    monkeypatch.setattr(
        checks, "band_check", lambda mean, sd, third: passing | {"passed": True}
    )
    samples = fake_posterior()
    result = april_2020(samples, panel, layout, num_draws=NUM_KEPT, seed=SEED)
    assert result["passed"] is True
    samples["sigma_level"][0, 0] = np.nan  # pooled draw 0, which the subsample keeps
    result = april_2020(samples, panel, layout, num_draws=NUM_KEPT, seed=SEED)
    assert result["non_finite_draws"] == 1
    assert result["passed"] is False
```

Create `tests/test_sbc.py`:

```python
"""Simulation-based calibration arithmetic, checked against exact rational oracles."""

import functools
import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from ces_revisions.pilot.sbc import (
    ecdf_test,
    gamma_critical_value,
    gamma_statistic,
    rank_of,
    recovery,
    thin,
)

SEED = sum(map(ord, "ces-revisions-stage-6-sbc"))
# A case small enough to enumerate every rank vector: 5 replicates, ranks in 0..4.
TINY_REPLICATES, TINY_DRAWS = 5, 4


@functools.cache
def _exact_tails(num_replicates, num_draws, point, count):
    """Exact lower P(X <= count) and upper 1 - P(X <= count - 1), X ~ Bin(N, z_point)."""
    z = Fraction(point, num_draws + 1)

    def cdf(k):
        return sum(
            math.comb(num_replicates, j) * z**j * (1 - z) ** (num_replicates - j)
            for j in range(k + 1)
        )

    return cdf(count), 1 - cdf(count - 1)


def brute_force_gamma(ranks, num_draws):
    """Gamma by the definition, one evaluation point at a time, in exact rationals."""
    smallest = None
    for point in range(1, num_draws + 1):
        count = sum(1 for rank in ranks if rank < point)
        lower, upper = _exact_tails(len(ranks), num_draws, point, count)
        smallest = (
            min(lower, upper) if smallest is None else min(smallest, lower, upper)
        )
    return 2 * smallest


def test_rank_of_counts_the_draws_below_the_prior_value():
    draws = np.array([0.1, 0.9, 0.3, 0.7])

    assert rank_of(0.5, draws) == 2
    assert rank_of(-1.0, draws) == 0
    assert rank_of(2.0, draws) == draws.size


def test_rank_of_breaks_ties_uniformly_over_the_tied_positions():
    rng = np.random.default_rng(SEED)
    num_draws, repeats = 4, 5_000
    ranks = [rank_of(1.0, np.ones(num_draws), rng) for _ in range(repeats)]
    counts = np.bincount(ranks, minlength=num_draws + 1)
    expected = repeats / (num_draws + 1)
    sd = math.sqrt(repeats * (1 / (num_draws + 1)) * (num_draws / (num_draws + 1)))

    assert counts.size == num_draws + 1
    assert np.abs(counts - expected).max() < 4 * sd
    partial = {rank_of(2.0, np.array([1.0, 2.0, 2.0, 3.0]), rng) for _ in range(200)}
    assert partial == {1, 2, 3}


def test_rank_of_repeats_for_a_seed_and_refuses_unbroken_ties_and_non_finite_input():
    draws = np.array([0.0, 1.0, 1.0, 1.0, 2.0])
    first = [rank_of(1.0, draws, np.random.default_rng(SEED)) for _ in range(3)]

    assert len(set(first)) == 1
    with pytest.raises(ValueError, match="tie"):
        rank_of(1.0, draws)
    with pytest.raises(ValueError, match="finite"):
        rank_of(np.nan, draws)
    with pytest.raises(ValueError, match="finite"):
        rank_of(0.5, np.array([0.0, np.inf]))
    with pytest.raises(ValueError, match="1-D"):
        rank_of(0.5, np.zeros((2, 2)))


def test_thin_keeps_evenly_spaced_draws_from_the_first():
    np.testing.assert_array_equal(thin(np.arange(1000), 100), np.arange(0, 1000, 10))
    np.testing.assert_array_equal(thin(np.arange(10), 3), [0, 3, 6])
    np.testing.assert_array_equal(thin(np.arange(7), 7), np.arange(7))
    np.testing.assert_array_equal(thin(np.arange(7.0), 1), [0.0])
    for num_kept in (0, 11):
        with pytest.raises(ValueError, match="num_kept"):
            thin(np.arange(10), num_kept)
    with pytest.raises(ValueError, match="1-D"):
        thin(np.zeros((4, 4)), 2)


def test_recovery_reports_the_posterior_z_score_and_contraction():
    result = recovery(2.0, np.array([1.0, 2.0, 3.0, 4.0, 5.0]), prior_sd=5.0)

    # Mean 3 and ddof=1 variance 2.5.
    assert result["z"] == pytest.approx(1.0 / math.sqrt(2.5), rel=1e-15)
    assert result["contraction"] == pytest.approx(0.9, rel=1e-15)
    with pytest.raises(ValueError, match="prior_sd"):
        recovery(2.0, np.array([1.0, 2.0]), prior_sd=0.0)


def test_gamma_matches_the_exact_definition_on_every_tiny_rank_vector():
    vectors = itertools.product(range(TINY_DRAWS + 1), repeat=TINY_REPLICATES)
    for ranks in vectors:
        expected = float(brute_force_gamma(ranks, TINY_DRAWS))
        actual = gamma_statistic(np.array(ranks), TINY_DRAWS)
        assert actual == pytest.approx(expected, rel=1e-12), ranks


@pytest.mark.parametrize(
    ("num_replicates", "num_draws"), [(1, 1), (3, 9), (50, 20), (200, 99)]
)
def test_gamma_matches_the_exact_definition_on_random_rank_vectors(
    num_replicates, num_draws
):
    rng = np.random.default_rng(SEED + num_replicates)
    for _ in range(5):
        ranks = rng.integers(0, num_draws + 1, size=num_replicates)
        expected = float(brute_force_gamma(ranks.tolist(), num_draws))
        assert gamma_statistic(ranks, num_draws) == pytest.approx(expected, rel=1e-12)


def test_perfectly_uniform_ranks_pass_and_a_point_mass_fails():
    num_draws, copies = 19, 10
    uniform = np.repeat(np.arange(num_draws + 1), copies)
    point_mass = np.zeros(uniform.size, dtype=np.int64)
    # Every count of ranks below i equals its binomial mean N * z_i = copies * i exactly.
    np.random.default_rng(SEED).shuffle(uniform)

    passed = ecdf_test(uniform, num_draws, 0.05, seed=SEED)
    failed = ecdf_test(point_mass, num_draws, 0.05, seed=SEED)

    assert passed["gamma"] == pytest.approx(
        float(brute_force_gamma(uniform.tolist(), num_draws)), rel=1e-12
    )
    assert passed["gamma"] > 0.5
    assert passed["passed"] is True
    assert failed["passed"] is False
    assert failed["gamma"] < 1e-100
    assert set(passed) == {"gamma", "critical", "passed"}
    assert all(isinstance(passed[key], float) for key in ("gamma", "critical"))


def test_ranks_bunched_in_the_middle_fail():
    # An overdispersed posterior puts the prior value near its middle draw every time.
    rng = np.random.default_rng(SEED)
    num_draws = 99
    bunched = rng.binomial(num_draws, 0.5, size=200)

    assert ecdf_test(bunched, num_draws, 0.05, seed=SEED)["passed"] is False


def test_critical_value_is_the_exact_quantile_of_the_enumerated_null():
    vectors = list(itertools.product(range(TINY_DRAWS + 1), repeat=TINY_REPLICATES))
    exact = sorted(brute_force_gamma(ranks, TINY_DRAWS) for ranks in vectors)
    for alpha in (0.05, 0.1, 0.2):
        # The inverse-CDF quantile of the equally likely enumerated gammas.
        quantile = exact[math.ceil(alpha * len(exact)) - 1]
        size = sum(1 for gamma in exact if gamma < quantile) / len(exact)
        critical = gamma_critical_value(
            TINY_REPLICATES, TINY_DRAWS, alpha, num_simulations=20_000, seed=SEED
        )

        assert critical == pytest.approx(float(quantile), rel=1e-12), alpha
        assert size < alpha


def test_critical_value_repeats_for_a_seed_and_rises_with_alpha():
    def critical(alpha, seed=SEED):
        return gamma_critical_value(100, 50, alpha, num_simulations=4_000, seed=seed)

    assert critical(0.05) == critical(0.05)
    assert critical(0.01) <= critical(0.05) <= critical(0.10)
    assert critical(0.01) < critical(0.10)


@pytest.mark.parametrize(
    ("ranks", "num_draws", "message"),
    [
        (np.array([0, 1, 5]), 4, "0..4"),
        (np.array([-1, 1]), 4, "0..4"),
        (np.array([0.0, 1.0]), 4, "integers"),
        (np.array([], dtype=np.int64), 4, "non-empty"),
        (np.array([0, 1]), 0, "num_draws"),
    ],
)
def test_gamma_rejects_ranks_outside_0_to_l(ranks, num_draws, message):
    with pytest.raises(ValueError, match=message):
        gamma_statistic(ranks, num_draws)


def test_critical_value_rejects_alpha_outside_the_unit_interval():
    for alpha in (0.0, 1.0):
        with pytest.raises(ValueError, match="alpha"):
            gamma_critical_value(10, 10, alpha, seed=SEED)
```

Create `tests/test_pilot_sbc_runner.py`:

```python
"""The SBC replicate runner, its resumable fan-out, and the pre-registered summary."""

import json
import math

import numpy as np
import pytest
from pilot_synthetic import synthetic_panel

from ces_revisions.pilot import sbc
from ces_revisions.pilot.model import SAMPLED_SITES
from ces_revisions.pilot.sbc import (
    DERIVED_QUANTITIES,
    LOG_LIKELIHOOD,
    QUANTITIES,
    prior_reference,
    read_replicates,
    replicate_path,
    replicate_seeds,
    run_replicate,
    run_sbc,
    summarize_sbc,
)

SEED = sum(map(ord, "ces-revisions-stage-6-sbc-runner"))
NUM_KEPT = 19
NUM_REPLICATES = 200
# A short NUTS run: enough to exercise every step, not to converge.
TINY = {"num_warmup": 40, "num_samples": 30, "num_kept": 10}


def fake_records(
    rng: np.random.Generator,
    *,
    contraction: float = 0.6,
    z_sd: float = 1.0,
    point_mass: str | None = None,
) -> list[dict]:
    """Replicate records with uniform ranks, except ``point_mass``'s, all at rank 0."""
    settings = {"layout": "dummy", "num_kept": NUM_KEPT}
    records = []
    for index in range(NUM_REPLICATES):
        ranks = {
            name: 0 if name == point_mass else int(rng.integers(0, NUM_KEPT + 1))
            for name in QUANTITIES
        }
        records.append(
            {
                "index": index,
                "settings": settings,
                "ranks": ranks,
                "recovery": {
                    name: {
                        "z": float(z_sd * rng.standard_normal()),
                        "contraction": contraction,
                    }
                    for name in (*SAMPLED_SITES, *DERIVED_QUANTITIES)
                },
                "ess_bulk": dict.fromkeys((*SAMPLED_SITES, *DERIVED_QUANTITIES), 500.0),
                "ess_bulk_kept": dict.fromkeys(QUANTITIES, 19.0),
                "divergences": int(index % 50 == 0),
                "n_steps": {"mean": 15.0, "max": 31, "saturated_share": 0.0},
                "seconds": {"sampling": 1.0, "total": 2.0},
            }
        )
    return records


def test_quantities_are_the_sites_the_derived_quantities_and_the_log_likelihood():
    assert QUANTITIES == (*SAMPLED_SITES, *DERIVED_QUANTITIES, LOG_LIKELIHOOD)
    assert len(QUANTITIES) == 23
    assert "sigma_news_post" not in QUANTITIES  # sigma_news_annual's exact rescaling


def test_replicate_seeds_are_deterministic_and_distinct():
    first = replicate_seeds(SEED, 3)
    assert first == replicate_seeds(SEED, 3)
    assert len(set(first.values())) == len(first)
    assert set(first.values()).isdisjoint(replicate_seeds(SEED, 4).values())
    assert set(first.values()).isdisjoint(replicate_seeds(SEED + 1, 3).values())


def test_prior_reference_matches_the_priors():
    reference = prior_reference("dummy")
    median, sd = reference["median"], reference["sd"]
    # HalfNormal(s): median s * 0.6745, SD s * sqrt(1 - 2 / pi); LogNormal medians.
    halfnormal_median = 0.6744897501960817
    assert median["sigma_level"] == pytest.approx(150 * halfnormal_median, rel=0.03)
    assert sd["sigma_level"] == pytest.approx(
        150 * math.sqrt(1 - 2 / math.pi), rel=0.03
    )
    assert median["sigma_news_s"] == pytest.approx(40.0, rel=0.02)
    assert median["kappa_crisis"] == pytest.approx(math.log(50.0), abs=0.02)
    assert median["rho_noise"] == pytest.approx(0.0, abs=0.01)
    assert sd["shift_mean"] == pytest.approx(15.0, rel=0.03)
    assert set(sd) == {*SAMPLED_SITES, *DERIVED_QUANTITIES}
    assert all(value > 0 for value in sd.values())


def test_summary_passes_uniform_ranks_with_recovered_news_shares():
    summary = summarize_sbc(fake_records(np.random.default_rng(SEED)))
    assert summary["num_replicates"] == NUM_REPLICATES
    assert summary["num_quantities"] == len(QUANTITIES)
    assert summary["alpha"] == pytest.approx(0.05 / len(QUANTITIES))
    assert summary["ecdf_failures"] == []
    assert summary["recovery_checks"] == {"share_s": True, "share_t": True}
    assert summary["divergent_replicates"] == 4
    assert summary["passed"] is True
    assert "recovery" not in summary["quantities"][LOG_LIKELIHOOD]
    json.dumps(summary, allow_nan=False)


def test_summary_fails_a_point_mass_rank():
    summary = summarize_sbc(
        fake_records(np.random.default_rng(SEED), point_mass="sigma_slope")
    )
    assert summary["ecdf_failures"] == ["sigma_slope"]
    assert summary["passed"] is False


@pytest.mark.parametrize(
    ("contraction", "z_sd"),
    [(0.2, 1.0), (0.6, 2.0)],
    ids=["weak-contraction", "poor-coverage"],
)
def test_summary_fails_unrecovered_news_shares(contraction, z_sd):
    summary = summarize_sbc(
        fake_records(np.random.default_rng(SEED), contraction=contraction, z_sd=z_sd)
    )
    assert summary["recovery_checks"] == {"share_s": False, "share_t": False}
    assert summary["passed"] is False


def test_summary_refuses_mixed_settings():
    records = fake_records(np.random.default_rng(SEED))
    records[1] = {**records[1], "settings": {"layout": "trig4", "num_kept": NUM_KEPT}}
    with pytest.raises(ValueError, match="different settings"):
        summarize_sbc(records)


def test_single_device_flags_replace_any_host_device_count():
    flags = sbc._single_device_flags(
        "--xla_force_host_platform_device_count=4 --xla_gpu_deterministic_ops=true"
    )
    assert flags.split() == [
        "--xla_gpu_deterministic_ops=true",
        "--xla_force_host_platform_device_count=1",
    ]
    assert sbc._single_device_flags(None) == "--xla_force_host_platform_device_count=1"


def test_run_sbc_skips_replicates_already_written(tmp_path, monkeypatch):
    """With every replicate file present, no worker starts and the files are summarized."""
    records = fake_records(np.random.default_rng(SEED))[:3]
    for record in records:
        replicate_path(tmp_path, record["index"]).write_text(json.dumps(record))

    def no_pool(*args, **kwargs):
        raise AssertionError("no replicate should run")

    monkeypatch.setattr(sbc, "ProcessPoolExecutor", no_pool)
    summary = run_sbc(
        tmp_path,
        layout_name="dummy",
        panel=None,
        start=0,
        stop=3,
        workers=2,
        base_seed=SEED,
        **TINY,
    )
    assert summary["skipped"] == 3
    assert summary["indices"] == [0, 1, 2]
    assert json.loads((tmp_path / sbc.SUMMARY_FILE).read_text())["indices"] == [0, 1, 2]
    assert [record["index"] for record in read_replicates(tmp_path)] == [0, 1, 2]


def test_run_sbc_refuses_several_workers_on_a_gpu_host(tmp_path, monkeypatch):
    def no_pool(*args, **kwargs):
        raise AssertionError("no worker should start")

    monkeypatch.setattr(sbc, "has_nvidia_device", lambda: True)
    monkeypatch.setattr(sbc, "ProcessPoolExecutor", no_pool)
    with pytest.raises(ValueError, match="share one GPU"):
        run_sbc(
            tmp_path,
            layout_name="dummy",
            panel=None,
            start=0,
            stop=2,
            workers=2,
            base_seed=SEED,
            **TINY,
        )


@pytest.mark.slow
def test_run_replicate_records_ranks_recovery_and_is_seeded():
    panel = synthetic_panel("dummy")
    record = run_replicate(2, base_seed=SEED, layout_name="dummy", panel=panel, **TINY)
    json.dumps(record, allow_nan=False)
    assert set(record["ranks"]) == set(QUANTITIES)
    assert all(0 <= rank <= TINY["num_kept"] for rank in record["ranks"].values())
    assert set(record["recovery"]) == {*SAMPLED_SITES, *DERIVED_QUANTITIES}
    assert record["seeds"] == replicate_seeds(SEED, 2)
    assert record["settings"]["num_kept"] == TINY["num_kept"]
    again = run_replicate(2, base_seed=SEED, layout_name="dummy", panel=panel, **TINY)
    for key in ("truth", "ranks", "recovery", "divergences", "n_steps", "step_size"):
        assert again[key] == record[key], key


@pytest.mark.slow
def test_run_sbc_spawns_workers_and_resumes(tmp_path):
    panel = synthetic_panel("dummy")
    arguments = {
        "layout_name": "dummy",
        "panel": panel,
        "workers": 2,
        "base_seed": SEED,
        **TINY,
    }
    first = run_sbc(tmp_path, start=0, stop=2, **arguments)
    assert first["failures"] == {}
    assert first["skipped"] == 0
    assert first["indices"] == [0, 1]
    written = {index: replicate_path(tmp_path, index).read_text() for index in (0, 1)}
    resumed = run_sbc(tmp_path, start=0, stop=3, **arguments)
    assert resumed["skipped"] == 2
    assert resumed["indices"] == [0, 1, 2]
    assert {
        index: replicate_path(tmp_path, index).read_text() for index in (0, 1)
    } == written
    # The worker's replicate equals the same replicate run in this process.
    local = run_replicate(2, base_seed=SEED, layout_name="dummy", panel=panel, **TINY)
    remote = json.loads(replicate_path(tmp_path, 2).read_text())
    assert remote["ranks"] == local["ranks"]
    assert remote["truth"] == pytest.approx(local["truth"], rel=1e-12)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_pilot_checks.py tests/test_sbc.py tests/test_pilot_sbc_runner.py -q`
Expected: collection errors, `ModuleNotFoundError: No module named 'ces_revisions.pilot.checks'` and `'ces_revisions.pilot.sbc'`.

- [ ] **Step 3: Write the check and the SBC module**

Create `src/ces_revisions/pilot/checks.py`:

```python
"""Posterior checks of the fitted pilot on the panel's most extreme month.

April 2020 moved total nonfarm by about twenty million jobs, two orders of magnitude more
than any month's revision. The crisis multiplier lets the S and T news grow with the level
shock that month (decision D2), so the smoothed latent change should follow the third print
rather than be pulled toward the trend, and the panel should pin it down rather than leave
it to the prior. ``april_2020`` reads the smoothed change, the smoothed news, and the
published prints there, for a subsample of posterior draws.
"""

from datetime import date

import jax
import jax.numpy as jnp
import numpy as np

from ces_revisions.kalman import kalman_filter, kalman_smoother
from ces_revisions.pilot.layout import NEWS_STATES, Layout, seasonal_selector
from ces_revisions.pilot.model import SAMPLED_SITES, pilot_ssm
from ces_revisions.pilot.panel import CLOSING_STAGES, COLUMNS, AggregatePanel
from ces_revisions.pilot.sbc import thin

APRIL_2020 = date(2020, 4, 1)
# Pre-registered: every subsampled draw filters to finite moments, and the smoothed latent
# change's central 90% interval, its mean plus or minus INTERVAL_Z total posterior SDs,
# lies within 5% of the third print. A mean inside the band alone would pass a change the
# panel leaves almost unidentified.
RATIO_RANGE = (0.95, 1.05)
INTERVAL_Z = 1.6448536269514722  # the standard normal's 0.95 quantile
# Draws filtered and smoothed together under lax.map; bounds the (batch, T, n, n) buffers.
BATCH_SIZE = 20


def change_row(layout: Layout) -> np.ndarray:
    """The (n,) row reading the latent change dx_t = level_t + gamma_t - lagged_value_t."""
    names = layout.state_names
    row = seasonal_selector(layout)
    row[names.index("level")] += 1.0
    row[names.index("lagged_value")] -= 1.0
    return row


def news_row(layout: Layout) -> np.ndarray:
    """The (n,) row summing the month's S, T, and post-T news states."""
    row = np.zeros(layout.state_dim)
    row[[layout.state_names.index(name) for name in NEWS_STATES]] = 1.0
    return row


def april_2020(
    samples: dict,
    panel: AggregatePanel,
    layout: Layout,
    *,
    num_draws: int = 200,
    seed: int,
) -> dict:
    """The smoothed latent change and news at April 2020 against the published prints.

    ``samples`` maps every sampled site to its draws, with any leading (chain, draw)
    shape. ``num_draws`` evenly spaced draws of the pooled chains are filtered and
    smoothed. A quantity's ``sd`` is its total posterior SD, the smoother's own variance
    averaged over draws plus the variance of the smoothed means across draws;
    ``sd_between_draws`` is the second part alone. ``seed`` is recorded for the run's
    provenance; the subsample itself is deterministic.
    """
    step = panel.reference_months.index(APRIL_2020)
    flat = {name: np.asarray(samples[name]).reshape(-1) for name in SAMPLED_SITES}
    total = len(flat[SAMPLED_SITES[0]])
    kept = thin(np.arange(total), min(num_draws, total))
    draws = {name: jnp.asarray(values[kept]) for name, values in flat.items()}
    rows = jnp.asarray(np.stack([change_row(layout), news_row(layout)]))
    y = jnp.asarray(panel.y)

    def moments(params):
        ssm = pilot_ssm(params, panel, layout)
        smoothed = kalman_smoother(ssm, kalman_filter(ssm, y))
        mean = rows @ smoothed.smoothed_mean[step]
        variance = jnp.einsum("ri,ij,rj->r", rows, smoothed.smoothed_cov[step], rows)
        return mean, variance

    mapped = jax.jit(lambda params: jax.lax.map(moments, params, batch_size=BATCH_SIZE))
    means, variances = (np.asarray(array) for array in mapped(draws))
    finite = np.isfinite(means).all(axis=1) & np.isfinite(variances).all(axis=1)
    observed = {
        stage: _optional(panel.y[step, COLUMNS.index(stage)])
        for stage in CLOSING_STAGES
    }
    summaries = {
        name: _summary(means[finite, k], variances[finite, k])
        for k, name in enumerate(("dx", "news_sum"))
    }
    third = observed["T"]
    check = (
        band_check(summaries["dx"]["mean"], summaries["dx"]["sd"], third)
        if third is not None and finite.any()
        else {"ratio": None, "interval_90": None, "band": None, "passed": False}
    )
    # A draw the filter cannot evaluate fails the check rather than drop out of it.
    check["passed"] = check["passed"] and bool(finite.all())
    return {
        "reference_month": APRIL_2020.isoformat(),
        "layout": layout.name,
        "seed": int(seed),
        "num_draws": len(kept),
        "draw_indices": kept.tolist(),
        "non_finite_draws": int((~finite).sum()),
        **summaries,
        "observed": observed,
        "ratio_range": list(RATIO_RANGE),
        **check,
    }


def band_check(mean: float, sd: float, third: float) -> dict:
    """Whether mean plus or minus INTERVAL_Z sd lies inside RATIO_RANGE times the print."""
    interval = [mean - INTERVAL_Z * sd, mean + INTERVAL_Z * sd]
    band = sorted(third * bound for bound in RATIO_RANGE)
    return {
        "ratio": mean / third,
        "interval_90": interval,
        "band": band,
        "passed": band[0] <= interval[0] and interval[1] <= band[1],
    }


def _summary(means: np.ndarray, variances: np.ndarray) -> dict:
    if means.size == 0:
        return {"mean": None, "sd": None, "sd_between_draws": None}
    between = float(means.var()) if means.size > 1 else 0.0
    return {
        "mean": float(means.mean()),
        "sd": float(np.sqrt(variances.mean() + between)),
        "sd_between_draws": float(np.sqrt(between)),
    }


def _optional(value: float) -> float | None:
    return None if np.isnan(value) else float(value)
```

Create `src/ces_revisions/pilot/sbc.py`:

```python
"""Simulation-based calibration of the pilot: rank arithmetic, the ECDF test, and the runner.

When a parameter is drawn from its prior, a panel is simulated from it, and the model is
fit to that panel, the parameter's rank among L posterior draws is uniform on 0..L
(Talts et al. 2018). Säilynoja, Bürkner & Vehtari (2022) test that uniformity with the
statistic gamma: the smallest two-sided binomial tail probability of the ranks' empirical
CDF over the L interior evaluation points, compared with its own lower alpha-quantile
under uniform ranks. The rank arithmetic is NumPy on finished draws.

``run_replicate`` performs one replicate on the pilot: a prior draw, a panel simulated
with the template panel's gaps, one NUTS chain, and the ranks, recovery, and ESS of every
sampled site, the derived quantities, and the log likelihood as a data-dependent test
quantity (Modrák et al. 2023). ``run_sbc`` fans replicates over worker processes, one
JSON file per replicate, and ``summarize_sbc`` applies the pre-registered pass rule.
"""

import dataclasses
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, init_to_value

from ces_revisions.devices import has_nvidia_device
from ces_revisions.kalman import kalman_filter
from ces_revisions.pilot.layout import LAYOUTS, Layout
from ces_revisions.pilot.model import (
    SAMPLED_SITES,
    TARGET_ACCEPT_PROB,
    derived,
    pilot_model,
    pilot_ssm,
    sample_prior,
)
from ces_revisions.pilot.panel import AggregatePanel
from ces_revisions.pilot.simulate import simulate_panel

# Simulated rank vectors per block in gamma_critical_value, which bounds its memory to
# a few (block, L) arrays however many simulations the caller asks for.
SIMULATION_BLOCK = 1_000


def rank_of(
    prior_value: float, draws: np.ndarray, rng: np.random.Generator | None = None
) -> int:
    """The number of posterior draws below the prior value, a rank in 0..len(draws).

    A draw equal to the prior value is a tie. Under exchangeability the prior value is
    equally likely to sit at each position among its ties, so the rank adds a uniform
    integer in 0..ties drawn from ``rng``; ties without an ``rng`` raise.
    """
    draws = np.asarray(draws, dtype=np.float64)
    if draws.ndim != 1 or draws.size == 0:
        raise ValueError("draws must be a non-empty 1-D array")
    if not np.isfinite(prior_value) or not np.isfinite(draws).all():
        raise ValueError("the prior value and every draw must be finite")
    below = int(np.count_nonzero(draws < prior_value))
    ties = int(np.count_nonzero(draws == prior_value))
    if ties == 0:
        return below
    if rng is None:
        raise ValueError(f"{ties} draws tie the prior value: pass a seeded rng")
    return below + int(rng.integers(0, ties + 1))


def thin(draws: np.ndarray, num_kept: int) -> np.ndarray:
    """``num_kept`` draws at evenly spaced positions, from the first draw on.

    Position j is ``j * len(draws) // num_kept``: a constant stride when ``num_kept``
    divides the draw count, and never a repeated draw.
    """
    draws = np.asarray(draws)
    if draws.ndim != 1:
        raise ValueError("draws must be a 1-D array")
    if not 1 <= num_kept <= draws.size:
        raise ValueError(f"num_kept must lie in 1..{draws.size}, not {num_kept}")
    return draws[np.arange(num_kept) * draws.size // num_kept]


def gamma_statistic(ranks: np.ndarray, num_draws: int) -> float:
    """Säilynoja, Bürkner & Vehtari's gamma for N ranks in 0..L, L = ``num_draws``.

    At each interior point z_i = i / (L + 1), i = 1..L, the count of ranks below i is
    Binomial(N, z_i) under uniform ranks; gamma is twice the smallest tail probability,
    lower P(X <= count) or upper P(X >= count) = 1 - P(X <= count - 1), over every i.
    Gamma can exceed 1, because both tails of the count's own atom can exceed one half.
    """
    ranks = _validated_ranks(ranks, num_draws)
    tails = _binomial_tails(ranks.size, num_draws)
    return float(_gammas(ranks[None, :], num_draws, tails)[0])


def gamma_critical_value(
    num_replicates: int,
    num_draws: int,
    alpha: float,
    *,
    num_simulations: int = 10_000,
    seed: int,
) -> float:
    """The alpha-quantile of gamma over simulated uniform ranks in 0..``num_draws``.

    The quantile is the inverse empirical CDF, so it is one of the simulated gammas:
    gamma takes discrete values, and the rule ``gamma >= critical`` then rejects
    uniform ranks with probability below alpha, up to simulation error. It rises with
    alpha: a larger alpha rejects more.
    """
    if num_replicates < 1 or num_draws < 1 or num_simulations < 1:
        raise ValueError("num_replicates, num_draws, and num_simulations must be >= 1")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1), not {alpha}")
    rng = np.random.default_rng(seed)
    tails = _binomial_tails(num_replicates, num_draws)
    gammas = []
    for start in range(0, num_simulations, SIMULATION_BLOCK):
        block = min(SIMULATION_BLOCK, num_simulations - start)
        ranks = rng.integers(0, num_draws + 1, size=(block, num_replicates))
        gammas.append(_gammas(ranks, num_draws, tails))
    return float(np.quantile(np.concatenate(gammas), alpha, method="inverted_cdf"))


def ecdf_test(
    ranks: np.ndarray,
    num_draws: int,
    alpha: float,
    *,
    seed: int,
    num_simulations: int = 10_000,
) -> dict:
    """Gamma for the ranks, its simulated critical value, and whether the ranks pass."""
    ranks = _validated_ranks(ranks, num_draws)
    gamma = gamma_statistic(ranks, num_draws)
    critical = gamma_critical_value(
        ranks.size, num_draws, alpha, num_simulations=num_simulations, seed=seed
    )
    return {"gamma": gamma, "critical": critical, "passed": gamma >= critical}


def recovery(prior_value: float, draws: np.ndarray, prior_sd: float) -> dict:
    """Posterior z-score and contraction of one parameter fit to one simulated panel.

    The z-score is (posterior mean - prior value) / posterior SD, and the contraction
    1 - posterior variance / prior variance; both use the draws' ddof=1 variance.
    """
    draws = np.asarray(draws, dtype=np.float64)
    if draws.ndim != 1 or draws.size < 2:
        raise ValueError("draws must be a 1-D array of at least two draws")
    if not prior_sd > 0:
        raise ValueError(f"prior_sd must be positive, not {prior_sd}")
    variance = draws.var(ddof=1)
    return {
        "z": float((draws.mean() - prior_value) / np.sqrt(variance)),
        "contraction": float(1.0 - variance / prior_sd**2),
    }


def _validated_ranks(ranks: np.ndarray, num_draws: int) -> np.ndarray:
    ranks = np.asarray(ranks)
    if ranks.ndim != 1 or ranks.size == 0:
        raise ValueError("ranks must be a non-empty 1-D array")
    if not np.issubdtype(ranks.dtype, np.integer):
        raise ValueError(f"ranks must be integers, not {ranks.dtype}")
    if num_draws < 1:
        raise ValueError(f"num_draws must be >= 1, not {num_draws}")
    if ranks.min() < 0 or ranks.max() > num_draws:
        raise ValueError(f"every rank must lie in 0..{num_draws}")
    return ranks.astype(np.int64)


def _binomial_tails(
    num_replicates: int, num_draws: int
) -> tuple[np.ndarray, np.ndarray]:
    """P(X_i <= k) and P(X_i >= k) for X_i ~ Binomial(N, i / (L + 1)), i = 1..L.

    Two (L, N + 1) tables indexed [i - 1, k]. The upper tail is summed from the top, not
    taken as one minus the lower tail, so a small upper tail keeps its relative accuracy.
    """
    z = np.arange(1, num_draws + 1) / (num_draws + 1)
    k = np.arange(num_replicates + 1)
    # log C(N, k) as a running sum of log((N - j + 1) / j), j = 1..k.
    log_comb = np.concatenate(
        [[0.0], np.cumsum(np.log((num_replicates - k[1:] + 1) / k[1:]))]
    )
    log_pmf = (
        log_comb + k * np.log(z)[:, None] + (num_replicates - k) * np.log1p(-z)[:, None]
    )
    pmf = np.exp(log_pmf)
    lower = np.cumsum(pmf, axis=1)
    upper = np.cumsum(pmf[:, ::-1], axis=1)[:, ::-1]
    return lower, upper


def _gammas(
    ranks: np.ndarray, num_draws: int, tails: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """Gamma for each row of an (S, N) matrix of integer ranks in 0..L."""
    num_rows = ranks.shape[0]
    bins = num_draws + 1
    # One histogram of ranks per row, counted in a single bincount over offset ranks.
    offsets = np.arange(num_rows)[:, None] * bins
    histogram = np.bincount((ranks + offsets).ravel(), minlength=num_rows * bins)
    # Column i - 1 counts the ranks below i, i = 1..L: an integer, never N * F-hat.
    below = np.cumsum(histogram.reshape(num_rows, bins)[:, :-1], axis=1)
    lower, upper = tails
    points = np.arange(num_draws)
    return 2.0 * np.minimum(lower[points, below], upper[points, below]).min(axis=1)


# The replicate runner.

DERIVED_QUANTITIES = (
    "sigma_noise_f",
    "sigma_noise_s",
    "sigma_noise_t",
    "share_s",
    "share_t",
)
# log p(y | theta), ranked at the true theta among the posterior draws: a data-dependent
# test quantity that is sensitive to miscalibration the marginal ranks miss.
LOG_LIKELIHOOD = "log_likelihood"
QUANTITIES = (*SAMPLED_SITES, *DERIVED_QUANTITIES, LOG_LIKELIHOOD)
# Pre-registered pass rule: every quantity passes the gamma-ECDF test at the Bonferroni
# level FAMILY_ALPHA / len(QUANTITIES); and for each news share, the median contraction
# is at least CONTRACTION_MIN and at most LARGE_Z_SHARE_MAX of the replicates have a
# posterior z-score beyond LARGE_Z.
FAMILY_ALPHA = 0.05
RECOVERY_QUANTITIES = ("share_s", "share_t")
CONTRACTION_MIN = 0.25
LARGE_Z = 3.0
LARGE_Z_SHARE_MAX = 0.05
# The prior draws behind every replicate's starting point (the prior median of each
# sampled site) and the prior SDs that scale recovery's contraction.
PRIOR_REFERENCE_SEED = sum(map(ord, "ces-revisions-stage-6-sbc-prior-reference"))
PRIOR_REFERENCE_DRAWS = 20_000
ECDF_SEED = sum(map(ord, "ces-revisions-stage-6-sbc-ecdf"))
MAX_TREE_DEPTH = 10
# The reduced panel: January 1999 to December 2021 holds 52 months before the May 2003
# shift, the 2020 crisis months, and the anchors of 2003-2021, so the shift, crisis, and
# post-T news parameters each meet data in every replicate.
FIRST_MONTH = date(1999, 1, 1)
LAST_MONTH = date(2021, 12, 1)
# Draws whose log likelihood is filtered together; bounds the (batch, T, n, n) buffers.
LOG_LIKELIHOOD_BATCH = 25
REPLICATE_FILE = "replicate-{index:04d}.json"
REPLICATE_GLOB = "replicate-*.json"
SUMMARY_FILE = "summary.json"
HOST_DEVICE_FLAG = "--xla_force_host_platform_device_count"


def replicate_seeds(base_seed: int, index: int) -> dict[str, int]:
    """Independent seeds for a replicate's prior draw, simulation, NUTS, and rank ties."""
    names = ("prior", "simulation", "mcmc", "ties")
    states = np.random.SeedSequence((base_seed, index)).generate_state(len(names))
    return {name: int(state) for name, state in zip(names, states, strict=True)}


def prior_reference(layout_name: str) -> dict:
    """Prior medians of the sampled sites and prior SDs of the sites and derived quantities.

    Estimated from PRIOR_REFERENCE_DRAWS ``sample_prior`` draws at a fixed seed, so every
    replicate starts NUTS at the same point and scales contraction by the same SDs.
    """
    draws = {
        name: np.asarray(values)
        for name, values in sample_prior(
            jax.random.PRNGKey(PRIOR_REFERENCE_SEED),
            PRIOR_REFERENCE_DRAWS,
            LAYOUTS[layout_name],
        ).items()
    }
    quantities = {**draws, **derived(draws)}
    return {
        "seed": PRIOR_REFERENCE_SEED,
        "num_draws": PRIOR_REFERENCE_DRAWS,
        "median": {name: float(np.median(draws[name])) for name in SAMPLED_SITES},
        "sd": {
            name: float(np.std(quantities[name], ddof=1))
            for name in (*SAMPLED_SITES, *DERIVED_QUANTITIES)
        },
    }


def run_replicate(
    index: int,
    *,
    base_seed: int,
    layout_name: str,
    panel: AggregatePanel,
    num_warmup: int,
    num_samples: int,
    num_kept: int,
    target_accept_prob: float = TARGET_ACCEPT_PROB,
) -> dict:
    """One SBC replicate: draw theta, simulate a panel with ``panel``'s gaps, fit, rank.

    One NUTS chain starts at the prior median. Ranks use ``num_kept`` evenly spaced
    post-warmup draws, so they lie in 0..num_kept; recovery and the bulk ESS use every
    draw. ``target_accept_prob`` should match the fit being calibrated. The returned
    record is JSON-safe.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("SBC needs float64: call numpyro.enable_x64() first")
    started = time.perf_counter()
    layout = LAYOUTS[layout_name]
    seeds = replicate_seeds(base_seed, index)
    reference = prior_reference(layout_name)
    theta = {
        name: float(values[0])
        for name, values in sample_prior(
            jax.random.PRNGKey(seeds["prior"]), 1, layout
        ).items()
    }
    simulated = dataclasses.replace(
        panel, y=simulate_panel(theta, panel, layout, seeds["simulation"])
    )

    mcmc = MCMC(
        NUTS(
            pilot_model,
            max_tree_depth=MAX_TREE_DEPTH,
            target_accept_prob=target_accept_prob,
            init_strategy=init_to_value(values=reference["median"]),
        ),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
        progress_bar=False,
    )
    sampling_started = time.perf_counter()
    mcmc.run(
        jax.random.PRNGKey(seeds["mcmc"]),
        simulated,
        layout,
        extra_fields=("diverging", "num_steps"),
    )
    samples = jax.block_until_ready(mcmc.get_samples())
    sampling_seconds = time.perf_counter() - sampling_started
    samples = {name: np.asarray(samples[name]) for name in SAMPLED_SITES}
    draws = {**samples, **derived(samples)}
    truth = {**theta, **{name: float(value) for name, value in derived(theta).items()}}

    kept = thin(np.arange(num_samples), num_kept)
    log_likelihoods = _log_likelihoods(
        {
            name: np.concatenate([[theta[name]], samples[name][kept]])
            for name in SAMPLED_SITES
        },
        simulated,
        layout,
    )
    truth[LOG_LIKELIHOOD] = float(log_likelihoods[0])
    kept_draws = {name: draws[name][kept] for name in draws}
    kept_draws[LOG_LIKELIHOOD] = log_likelihoods[1:]

    rng = np.random.default_rng(seeds["ties"])
    extra = mcmc.get_extra_fields()
    num_steps = np.asarray(extra["num_steps"])
    return {
        "index": index,
        "seeds": seeds,
        "settings": {
            "layout": layout_name,
            "base_seed": base_seed,
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "num_kept": num_kept,
            "max_tree_depth": MAX_TREE_DEPTH,
            "target_accept_prob": target_accept_prob,
            "first_month": panel.reference_months[0].isoformat(),
            "last_month": panel.reference_months[-1].isoformat(),
            "prior_reference_seed": reference["seed"],
        },
        "truth": truth,
        "ranks": {
            name: rank_of(truth[name], kept_draws[name], rng) for name in QUANTITIES
        },
        "recovery": {
            name: recovery(truth[name], draws[name], reference["sd"][name])
            for name in (*SAMPLED_SITES, *DERIVED_QUANTITIES)
        },
        "ess_bulk": {
            name: _json_float(az.ess(draws[name][None, :], method="bulk"))
            for name in (*SAMPLED_SITES, *DERIVED_QUANTITIES)
        },
        "ess_bulk_kept": {
            name: _json_float(az.ess(kept_draws[name][None, :], method="bulk"))
            for name in QUANTITIES
        },
        "divergences": int(np.asarray(extra["diverging"]).sum()),
        "n_steps": {
            "mean": float(num_steps.mean()),
            "max": int(num_steps.max()),
            "saturated_share": float(np.mean(num_steps >= 2**MAX_TREE_DEPTH - 1)),
        },
        "step_size": float(mcmc.last_state.adapt_state.step_size),
        "seconds": {
            "sampling": sampling_seconds,
            "total": time.perf_counter() - started,
        },
    }


def _log_likelihoods(params: dict, panel: AggregatePanel, layout: Layout) -> np.ndarray:
    """log p(y | theta) for each draw in params, (draws,) arrays per sampled site."""
    y = jnp.asarray(panel.y)

    def one(draw):
        return kalman_filter(pilot_ssm(draw, panel, layout), y).log_likelihood

    mapped = jax.jit(
        lambda draws: jax.lax.map(one, draws, batch_size=LOG_LIKELIHOOD_BATCH)
    )
    return np.asarray(
        mapped({name: jnp.asarray(value) for name, value in params.items()})
    )


def _json_float(value) -> float | None:
    value = float(np.asarray(value))
    return value if np.isfinite(value) else None


def replicate_path(out_dir: Path, index: int) -> Path:
    return out_dir / REPLICATE_FILE.format(index=index)


def run_sbc(
    out_dir: Path,
    *,
    layout_name: str,
    panel: AggregatePanel,
    start: int,
    stop: int,
    workers: int,
    base_seed: int,
    num_warmup: int,
    num_samples: int,
    num_kept: int,
    target_accept_prob: float = TARGET_ACCEPT_PROB,
) -> dict:
    """Run replicates start..stop-1 in spawned worker processes; summarize the out_dir.

    A replicate whose JSON file exists is skipped, so a stopped run resumes. Each worker
    sees one host device: XLA_FLAGS is set before the pool exists, so the spawned
    workers inherit it, and restored afterwards. A failed replicate writes no file and
    is listed under "failures", so a rerun retries it.

    On a host with an NVIDIA device every worker would share the first GPU and its
    preallocated memory, so ``workers`` must be 1 there. One replicate per GPU comes from
    one process per GPU, each under its own ``CUDA_VISIBLE_DEVICES`` with its own
    ``start`` and ``stop``, all writing to one ``out_dir``: a replicate's seeds depend
    only on ``base_seed`` and its index.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        index
        for index in range(start, stop)
        if not replicate_path(out_dir, index).exists()
    ]
    failures = {}
    if pending:
        if workers > 1 and has_nvidia_device():
            raise ValueError(
                "on a GPU host every worker would share one GPU: run one worker per "
                "process and shard replicates across GPUs with start, stop, and "
                "CUDA_VISIBLE_DEVICES"
            )
        saved_flags = os.environ.get("XLA_FLAGS")
        os.environ["XLA_FLAGS"] = _single_device_flags(saved_flags)
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_worker,
            ) as pool:
                futures = {
                    pool.submit(
                        run_replicate,
                        index,
                        base_seed=base_seed,
                        layout_name=layout_name,
                        panel=panel,
                        num_warmup=num_warmup,
                        num_samples=num_samples,
                        num_kept=num_kept,
                        target_accept_prob=target_accept_prob,
                    ): index
                    for index in pending
                }
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        record = future.result()
                    except Exception as error:  # noqa: BLE001 - one failure must not stop the rest
                        failures[index] = repr(error)
                        continue
                    _write_json(record, replicate_path(out_dir, index))
        finally:
            if saved_flags is None:
                del os.environ["XLA_FLAGS"]
            else:
                os.environ["XLA_FLAGS"] = saved_flags
    summary = write_summary(out_dir)
    summary["failures"] = {
        str(index): error for index, error in sorted(failures.items())
    }
    summary["skipped"] = (stop - start) - len(pending)
    return summary


def _single_device_flags(flags: str | None) -> str:
    kept = [
        flag for flag in (flags or "").split() if not flag.startswith(HOST_DEVICE_FLAG)
    ]
    return " ".join([*kept, f"{HOST_DEVICE_FLAG}=1"])


def _initialize_worker() -> None:
    numpyro.enable_x64()
    if jax.default_backend() == "cpu" and jax.local_device_count() != 1:
        raise RuntimeError(
            f"an SBC worker must see one host device, not {jax.local_device_count()}"
        )


def _write_json(record: dict, path: Path) -> None:
    """Write through a temporary file, so a killed process never leaves a partial file."""
    text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def read_replicates(out_dir: Path) -> list[dict]:
    """Every replicate record in out_dir, by index."""
    return [
        json.loads(path.read_text()) for path in sorted(out_dir.glob(REPLICATE_GLOB))
    ]


def summarize_sbc(records: list[dict]) -> dict:
    """The gamma-ECDF test and recovery of every quantity, and the pre-registered verdict."""
    if not records:
        raise ValueError("no replicate records to summarize")
    settings = records[0]["settings"]
    if any(record["settings"] != settings for record in records):
        raise ValueError("replicate records were run with different settings")
    num_kept = settings["num_kept"]
    alpha = FAMILY_ALPHA / len(QUANTITIES)
    quantities = {}
    for name in QUANTITIES:
        ranks = np.array([record["ranks"][name] for record in records], dtype=np.int64)
        test = ecdf_test(ranks, num_kept, alpha, seed=ECDF_SEED)
        entry = {"ecdf": {key: test[key] for key in ("gamma", "critical")}}
        entry["ecdf"]["passed"] = bool(test["passed"])
        entry["ess_bulk_kept_min"] = _min_or_none(
            record["ess_bulk_kept"][name] for record in records
        )
        if name != LOG_LIKELIHOOD:
            z = np.array([record["recovery"][name]["z"] for record in records])
            contraction = np.array(
                [record["recovery"][name]["contraction"] for record in records]
            )
            entry["recovery"] = {
                "median_contraction": float(np.median(contraction)),
                "share_abs_z_above_3": float(np.mean(np.abs(z) > LARGE_Z)),
            }
            entry["ess_bulk_min"] = _min_or_none(
                record["ess_bulk"][name] for record in records
            )
        quantities[name] = entry
    recovery_checks = {
        name: quantities[name]["recovery"]["median_contraction"] >= CONTRACTION_MIN
        and quantities[name]["recovery"]["share_abs_z_above_3"] <= LARGE_Z_SHARE_MAX
        for name in RECOVERY_QUANTITIES
    }
    ecdf_passed = all(entry["ecdf"]["passed"] for entry in quantities.values())
    return {
        "settings": settings,
        "num_replicates": len(records),
        "indices": [record["index"] for record in records],
        "num_quantities": len(QUANTITIES),
        "alpha": alpha,
        "quantities": quantities,
        "ecdf_failures": [
            name for name, entry in quantities.items() if not entry["ecdf"]["passed"]
        ],
        "recovery_checks": recovery_checks,
        "divergent_replicates": sum(record["divergences"] > 0 for record in records),
        "saturated_replicates": sum(
            record["n_steps"]["saturated_share"] > 0 for record in records
        ),
        "n_steps_mean": float(
            np.mean([record["n_steps"]["mean"] for record in records])
        ),
        "replicate_seconds_median": float(
            np.median([record["seconds"]["total"] for record in records])
        ),
        "passed": ecdf_passed and all(recovery_checks.values()),
    }


def write_summary(out_dir: Path) -> dict:
    """Summarize every replicate in out_dir and write it to out_dir / SUMMARY_FILE."""
    summary = summarize_sbc(read_replicates(out_dir))
    _write_json(summary, out_dir / SUMMARY_FILE)
    return summary


def _min_or_none(values) -> float | None:
    values = list(values)
    if any(value is None for value in values):
        return None
    return float(min(values))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_pilot_checks.py tests/test_sbc.py tests/test_pilot_sbc_runner.py -q -m "not slow"`
Expected: `43 passed, 2 deselected`.

Run: `uv run pytest tests/test_pilot_sbc_runner.py -q -m slow`
Expected: `2 passed`. They run two short replicates in worker processes and check the resume path.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format src/ces_revisions/pilot/checks.py src/ces_revisions/pilot/sbc.py tests/test_pilot_checks.py tests/test_sbc.py tests/test_pilot_sbc_runner.py
uv run ruff check src/ces_revisions/pilot/checks.py src/ces_revisions/pilot/sbc.py tests/test_pilot_checks.py tests/test_sbc.py tests/test_pilot_sbc_runner.py
git add src/ces_revisions/pilot/checks.py src/ces_revisions/pilot/sbc.py tests/test_pilot_checks.py tests/test_sbc.py tests/test_pilot_sbc_runner.py
git commit -m "Add the April 2020 check and the pilot's SBC"
```

### Task 8: The decision rules and the probe grid

**Files:**
- Create: `src/ces_revisions/pilot/dimensions.py`, `src/ces_revisions/pilot/seasonal.py`, `src/ces_revisions/pilot/gpu_choice.py`
- Create: `scripts/gpu_probe_grid.py`
- Test: `tests/test_pilot_dimensions.py`, `tests/test_pilot_seasonal.py`, `tests/test_pilot_gpu_choice.py`

**Interfaces:**
- Consumes: `engine_probe.synthetic_problem`, `engine_probe.time_batches`, `engine_probe.batched_value_and_grad`, `devices.jax_versions`, `devices.nvidia_driver`, `DETERMINISTIC_GPU_FLAG`, `has_nvidia_device` (Task 6).
- Produces:
  - `dimensions`: `SEASONAL_DIMS = {"dummy": 11, "trig": 11, "trig4": 8}`, `pilot_dimensions(layout_name, num_steps=572) -> dict`, `stage_dimensions(layout_name, stage, wedge="low", num_steps=572) -> dict` with keys `steps`, `states`, `cells`, `stage`, `layout`, `wedge`, `matches_layout(entry, layout_name) -> bool` (a grid entry has the layout's Stage 7–9 states and cells), `STAGE_CONFIGURATIONS` (Stage 7 low, Stage 8 low and high, Stage 9 low and high), and `MEASUREMENT_GRID` (dummy and trig4, every model of `STAGE_CONFIGURATIONS`, at 572 and 280 steps: 20 entries).
  - `seasonal`: `REFERENCE_LAYOUT = "dummy"`, `SE_MULTIPLE = 2.0`, `COST_SAVING_MIN = 0.25`, `PARETO_K_MAX = 0.7`, `LJUNG_BOX_ALPHA = 0.01`, `LJUNG_BOX_Q_RATIO_MAX = 1.25`, `COST_ENTRY`, `elpd_difference(alternative, reference, keep) -> dict`, `ljung_box_rejections(ljung_box) -> int`, `verdict(candidates, differences) -> dict` (each candidate has `states`, `meets_req18`, `april_2020_passed`, `ljung_box_q`, `seconds`), `choose_layout(candidates, pointwise, pareto_k) -> dict`, `grid_seconds(grid, layout_name) -> float | None`.
  - `gpu_choice`: `NUM_CHAINS`, `ITERATIONS`, `STEPS_PER_ITERATION = 127`, `STEP_BAND`, `LOCKSTEP_FACTOR`, `FIT_HOURS_MAX = 48.0`, `FIT_STEPS = 572`, `SBC_STEPS = 280`, `SBC_REPLICATES`, `SBC_ITERATIONS`, `SBC_DAYS_MAX = 14.0`, `BATCH_LOCKSTEP_FACTORS`, `batch_timings(record, layout_name, target) -> dict[int, float | None] | None` (`target` gives `stage`, `wedge`, and `steps`), `configurations(record, layout_name, stage, wedge, steps_per_iteration=127) -> list[dict]`, `batched_sbc(sbc, gpus, steps_per_iteration=127) -> dict | None`, `choose_stage(records, layout_name, stage, wedge, steps_per_iteration=127) -> dict` with keys `stage`, `wedge`, `configurations`, `band_fit_hours`, `status` ("chosen", "infeasible", or "unmeasured"), and `choice`, and `choose_gpu(records, layout_name, steps_per_iteration=127) -> dict` with keys `steps_per_iteration`, `stages` (one `choose_stage` result per model of `STAGE_CONFIGURATIONS`), and the overall `status`.
  - `scripts/gpu_probe_grid.py`: `--label`, `--out`, `--batch`, `--repeats`, `--steps`, `--grid-filter`, `--grid-json`, `--missing-share`, `--seed`, `--max-bytes`, `--hourly-price`, `--four-gpu-price`. `--four-gpu-price` is the whole 4-GPU instance's hourly price, and the script refuses one below twice `--hourly-price`. One JSON record per host: `label`, `hourly_price_usd`, `four_gpu_hourly_price_usd`, `host`, `versions`, `entries`, `timestamp_utc`. On a GPU the device's `bytes_limit` caps each batch, so batches that cannot fit are recorded as skipped rather than crashing the run, and larger batches are skipped without compiling. On a host with an NVIDIA device, the script refuses to run unless JAX's backend is the GPU.

`NUM_CHAINS` is Req 18's chain count, not a GPU count. For each Stage 7–9 model (D5), the rule considers one GPU running all chains, and one GPU per chain where the record carries the price of a 4-GPU instance of the same type. `BATCH_LOCKSTEP_FACTORS` come from the planning spike's 100 SBC replicates: the expected largest of b replicates' mean steps per iteration, over their average.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pilot_dimensions.py`:

```python
"""State and cell counts of the pilot and of the Stage 7-9 models, and the grid probe."""

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import gpu_probe_grid
import jax
import pytest
from gpu_probe_grid import DEFAULT_STEPS, measure_entry, measurement_entries

from ces_revisions.pilot.dimensions import (
    MEASUREMENT_GRID,
    PILOT_CELLS,
    SEASONAL_DIMS,
    matches_layout,
    pilot_dimensions,
    stage_dimensions,
)

# (layout, stage, wedge) -> (states, cells), counted by hand from the term list.
EXPECTED = {
    ("dummy", 7, "low"): (190, 37),
    ("dummy", 8, "low"): (203, 85),
    ("dummy", 8, "high"): (324, 217),
    ("dummy", 9, "low"): (247, 145),
    ("dummy", 9, "high"): (368, 277),
    ("trig", 7, "low"): (190, 37),
    ("trig4", 7, "low"): (157, 37),
    ("trig4", 8, "low"): (170, 85),
    ("trig4", 8, "high"): (291, 217),
    ("trig4", 9, "low"): (214, 145),
    ("trig4", 9, "high"): (335, 277),
}


def test_pilot_dimensions_count_trend_seasonal_and_news_states():
    assert pilot_dimensions("dummy") == {
        "steps": 572,
        "states": 17,
        "cells": PILOT_CELLS,
        "stage": 6,
        "layout": "dummy",
        "wedge": None,
    }
    assert pilot_dimensions("trig")["states"] == 17
    assert pilot_dimensions("trig4", num_steps=280)["states"] == 14
    assert pilot_dimensions("trig4", num_steps=280)["steps"] == 280
    assert SEASONAL_DIMS == {"dummy": 11, "trig": 11, "trig4": 8}


@pytest.mark.parametrize(("key", "expected"), EXPECTED.items(), ids=str)
def test_stage_dimensions_match_the_hand_count(key, expected):
    layout_name, stage, wedge = key
    dims = stage_dimensions(layout_name, stage, wedge, num_steps=280)

    assert (dims["states"], dims["cells"]) == expected
    assert dims == {
        "steps": 280,
        "states": expected[0],
        "cells": expected[1],
        "stage": stage,
        "layout": layout_name,
        "wedge": wedge,
    }


def test_stage_dimensions_default_to_the_full_span_and_the_low_wedge():
    assert stage_dimensions("dummy", 9) == stage_dimensions("dummy", 9, "low", 572)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("weekly", 7), "layout"),
        (("dummy", 6), "stage"),
        (("dummy", 10), "stage"),
        (("dummy", 9, "medium"), "wedge"),
        (("dummy", 7, "high"), "Stage 7"),
    ],
)
def test_stage_dimensions_reject_unknown_configurations(arguments, message):
    with pytest.raises(ValueError, match=message):
        stage_dimensions(*arguments)


def test_pilot_dimensions_reject_an_unknown_layout():
    with pytest.raises(ValueError, match="layout"):
        pilot_dimensions("weekly")


def test_measurement_grid_lists_both_layouts_every_stage_7_to_9_model_and_both_spans():
    configurations = [
        (entry["layout"], entry["stage"], entry["wedge"], entry["steps"])
        for entry in MEASUREMENT_GRID
    ]

    assert configurations == [
        (layout_name, stage, wedge, steps)
        for layout_name in ("dummy", "trig4")
        for stage, wedge in (
            (7, "low"),
            (8, "low"),
            (8, "high"),
            (9, "low"),
            (9, "high"),
        )
        for steps in (572, 280)
    ]
    assert len(MEASUREMENT_GRID) == 20
    for entry in MEASUREMENT_GRID:
        expected = EXPECTED[(entry["layout"], entry["stage"], entry["wedge"])]
        assert (entry["states"], entry["cells"]) == expected
    assert max(entry["states"] for entry in MEASUREMENT_GRID) == 368
    assert max(entry["cells"] for entry in MEASUREMENT_GRID) == 277


def test_grid_entries_default_to_the_measurement_grid():
    assert measurement_entries() == list(MEASUREMENT_GRID)
    assert measurement_entries(list(DEFAULT_STEPS)) == list(MEASUREMENT_GRID)


def test_grid_entries_take_other_steps_and_filter_by_configuration():
    filtered = measurement_entries([6], ["layout=trig4", "stage=7"])
    either_stage = measurement_entries([572], ["stage=7", "stage=9", "wedge=high"])

    assert filtered == [stage_dimensions("trig4", 7, "low", 6)]
    assert either_stage == [
        stage_dimensions(layout_name, 9, "high", 572)
        for layout_name in ("dummy", "trig4")
    ]


def test_grid_entries_reject_a_bad_filter_and_steps_beside_a_grid():
    grid = [{"steps": 6, "states": 3, "cells": 4}]

    assert measurement_entries(grid=grid) == [
        {
            "layout": None,
            "stage": None,
            "wedge": None,
            "steps": 6,
            "states": 3,
            "cells": 4,
        }
    ]
    with pytest.raises(ValueError, match="KEY=VALUE"):
        measurement_entries(filters=["sector=00"])
    with pytest.raises(ValueError, match="KEY=VALUE"):
        measurement_entries(filters=["layout"])
    with pytest.raises(ValueError, match="not both"):
        measurement_entries([6], grid=grid)
    with pytest.raises(ValueError, match="states"):
        measurement_entries(grid=[{"steps": 6, "states": 0, "cells": 4}])


def test_grid_probe_writes_one_record_with_timings_and_memory(tmp_path):
    grid = [
        {
            "layout": "tiny",
            "stage": 7,
            "wedge": "low",
            "steps": 6,
            "states": 3,
            "cells": 4,
        },
        {
            "layout": "tiny",
            "stage": 9,
            "wedge": "high",
            "steps": 5,
            "states": 4,
            "cells": 2,
        },
    ]
    grid_path = tmp_path / "grid.json"
    grid_path.write_text(json.dumps(grid))
    out = tmp_path / "grid-probe.json"
    command = [sys.executable, str(Path("scripts") / "gpu_probe_grid.py")]
    command += ["--label", "test", "--out", str(out), "--grid-json", str(grid_path)]
    command += ["--batch", "1", "2", "--repeats", "2", "--grid-filter", "layout=tiny"]
    command += ["--hourly-price", "1.29", "--four-gpu-price", "5.16"]
    subprocess.run(command, check=True, cwd=Path(__file__).parents[1])
    record = json.loads(out.read_text())

    assert set(record) == {
        "label",
        "hourly_price_usd",
        "four_gpu_hourly_price_usd",
        "host",
        "versions",
        "entries",
        "timestamp_utc",
    }
    assert record["label"] == "test"
    assert record["hourly_price_usd"] == 1.29
    assert record["four_gpu_hourly_price_usd"] == 5.16
    assert record["host"]["backend"] == jax.default_backend()
    assert record["host"]["device_kind"] == jax.devices()[0].device_kind
    assert (record["host"]["nvidia_driver"] is None) == (
        shutil.which("nvidia-smi") is None
    )
    for name in ("jax", "jaxlib"):
        assert record["versions"][name] == importlib.metadata.version(name)
    assert datetime.fromisoformat(record["timestamp_utc"]).utcoffset() == timedelta(0)
    assert [
        {
            key: entry[key]
            for key in ("layout", "stage", "wedge", "steps", "states", "cells")
        }
        for entry in record["entries"]
    ] == grid
    for entry in record["entries"]:
        assert [batch["batch"] for batch in entry["batches"]] == [1, 2]
        assert entry["repeats"] == 2
        assert (entry["peak_bytes_in_use"] is None) == (
            jax.devices()[0].memory_stats() is None
        )
        for batch in entry["batches"]:
            assert batch["compile_seconds"] > 0
            assert batch["median_seconds"] > 0
            assert batch["iqr_seconds"] >= 0
            assert batch["memory_analysis"]["total_bytes"] > 0
        smaller, larger = (batch["memory_analysis"] for batch in entry["batches"])
        assert larger["temp_size_in_bytes"] > smaller["temp_size_in_bytes"]


def test_grid_probe_refuses_a_filter_that_matches_nothing(tmp_path):
    command = [sys.executable, str(Path("scripts") / "gpu_probe_grid.py")]
    command += ["--label", "test", "--out", str(tmp_path / "none.json")]
    command += ["--grid-filter", "layout=weekly"]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).parents[1],
    )

    assert result.returncode == 2
    assert "no grid entry matches" in result.stderr
    assert not (tmp_path / "none.json").exists()


def test_grid_probe_skips_batches_beyond_the_memory_budget():
    entry = {"layout": "tiny", "stage": 7, "wedge": "low"}
    entry |= {"steps": 6, "states": 3, "cells": 4}
    unbounded = measure_entry(entry, [1, 2, 8], 1, 0.25, seed=0)
    totals = [batch["memory_analysis"]["total_bytes"] for batch in unbounded["batches"]]
    bounded = measure_entry(entry, [1, 2, 8], 1, 0.25, seed=0, max_bytes=totals[0])

    assert totals == sorted(totals)
    assert totals[0] < totals[1]
    assert all("error" not in batch for batch in unbounded["batches"])
    assert [batch.get("error") for batch in bounded["batches"]] == [
        None,
        f"skipped: compiled memory exceeds {totals[0]} bytes",
        "skipped: batch 2 did not fit in memory",
    ]
    assert bounded["batches"][0]["median_seconds"] > 0
    assert "median_seconds" not in bounded["batches"][1]
    assert bounded["batches"][1]["memory_analysis"]["total_bytes"] == totals[1]
    # Batch 8 follows a batch that did not fit, so it is skipped without compiling.
    assert bounded["batches"][2]["memory_analysis"] is None
    assert bounded["max_bytes"] == totals[0]


def test_trig_entries_are_the_dummys_and_trig4_entries_are_its_own():
    for entry in MEASUREMENT_GRID:
        own = entry["layout"]
        assert matches_layout(entry, own)
        assert matches_layout(entry, "trig") == (own == "dummy")
        assert matches_layout(entry, "dummy") == (own == "dummy")
        assert matches_layout(entry, "trig4") == (own == "trig4")


def test_an_entry_outside_stages_7_to_9_matches_no_layout():
    custom = {"layout": "tiny", "stage": None, "wedge": None}
    assert not matches_layout(custom | {"steps": 6, "states": 3, "cells": 4}, "dummy")
    stage_7_high = stage_dimensions("dummy", 7) | {"wedge": "high"}
    assert not matches_layout(stage_7_high, "dummy")


@pytest.mark.parametrize(
    "prices",
    [["--four-gpu-price", "1.99"], ["--four-gpu-price", "7.96"]],
    ids=["per-gpu-price", "no-hourly-price"],
)
def test_grid_probe_refuses_a_per_gpu_four_gpu_price(tmp_path, capsys, prices):
    out = tmp_path / "a100.json"
    command = ["--label", "a100", "--out", str(out), *prices]
    if prices[1] == "1.99":
        command += ["--hourly-price", "1.99"]
    with pytest.raises(SystemExit) as raised:
        gpu_probe_grid.main(command)
    assert raised.value.code == 2
    assert "whole 4-GPU instance" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.skipif(jax.default_backend() == "gpu", reason="the GPU backend is present")
def test_grid_probe_refuses_a_cpu_backend_beside_an_nvidia_device(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(gpu_probe_grid, "has_nvidia_device", lambda: True)
    monkeypatch.setenv("XLA_FLAGS", os.environ.get("XLA_FLAGS", ""))
    out = tmp_path / "cpu.json"
    with pytest.raises(SystemExit) as raised:
        gpu_probe_grid.main(["--label", "a10", "--out", str(out)])
    assert raised.value.code == 2
    assert "JAX's backend is 'cpu'" in capsys.readouterr().err
    assert not out.exists()
```

Create `tests/test_pilot_seasonal.py`:

```python
"""The pre-registered seasonal-layout rule, clause by clause, on hand-made comparisons."""

import numpy as np
import pytest

from ces_revisions.pilot.dimensions import stage_dimensions
from ces_revisions.pilot.seasonal import (
    COST_SAVING_MIN,
    LJUNG_BOX_Q_RATIO_MAX,
    PARETO_K_MAX,
    choose_layout,
    elpd_difference,
    grid_seconds,
    ljung_box_rejections,
    verdict,
)


def candidate(states, seconds, *, req18=True, april=True, q=(85.0, 50.0, 62.0)):
    return {
        "states": states,
        "meets_req18": req18,
        "april_2020_passed": april,
        "ljung_box_q": dict(zip(("F", "S", "T"), q, strict=True)),
        "seconds": seconds,
    }


DUMMY = candidate(247, 3.0)


def test_elpd_difference_sums_the_kept_blocks_with_az_compare_se():
    alternative = np.array([1.0, 2.0, 4.0, -1.0])
    reference = np.array([0.0, 1.0, 1.0, 0.0])
    keep = np.array([True, True, True, False])
    result = elpd_difference(alternative, reference, keep)
    # Kept differences 1, 1, 3: sum 5, SD (ddof 1) sqrt(4/3), se sqrt(3) * sqrt(4/3) = 2.
    assert result == {"diff": 5.0, "se": pytest.approx(2.0, rel=1e-15), "num_blocks": 3}


def test_ljung_box_rejections_count_columns_below_one_percent():
    tests = {"F": {"p_value": 0.001}, "S": {"p_value": 0.01}, "T": {"p_value": 0.2}}
    assert ljung_box_rejections(tests) == 1


def test_clause_a_takes_the_largest_difference_beyond_two_se():
    candidates = {
        "dummy": DUMMY,
        "trig": candidate(247, 3.0),
        "trig4": candidate(214, 2.0),
    }
    differences = {
        "trig": {"diff": 30.0, "se": 10.0},
        "trig4": {"diff": 25.0, "se": 5.0},
    }
    assert verdict(candidates, differences) == {"layout": "trig", "clause": "a"}


@pytest.mark.parametrize("failing", [{"req18": False}, {"april": False}])
def test_clause_a_needs_req18_and_the_april_check(failing):
    candidates = {"dummy": DUMMY, "trig": candidate(247, 3.0, **failing)}
    assert verdict(candidates, {"trig": {"diff": 30.0, "se": 10.0}})["clause"] == "c"


def test_a_difference_of_exactly_two_se_does_not_trigger_clause_a():
    candidates = {"dummy": DUMMY, "trig": candidate(247, 3.0)}
    assert verdict(candidates, {"trig": {"diff": 20.0, "se": 10.0}})["clause"] == "c"


@pytest.mark.parametrize(
    ("trig4", "difference", "clause"),
    [
        (candidate(214, 2.25), {"diff": -15.0, "se": 10.0}, "b"),  # saving 0.25
        (candidate(214, 2.26), {"diff": -15.0, "se": 10.0}, "c"),  # saving below 0.25
        (candidate(214, 2.0), {"diff": -25.0, "se": 10.0}, "c"),  # beyond two SE
        (candidate(214, 2.0, req18=False), {"diff": 0.0, "se": 10.0}, "c"),
        (candidate(214, 2.0, april=False), {"diff": 0.0, "se": 10.0}, "c"),
        # Q at the ratio in every column passes; one column above it fails.
        (candidate(214, 2.0, q=(106.25, 62.5, 77.5)), {"diff": 0.0, "se": 10.0}, "b"),
        (candidate(214, 2.0, q=(85.0, 62.6, 62.0)), {"diff": 0.0, "se": 10.0}, "c"),
        (
            candidate(214, 2.0, q=(2286.0, 1514.0, 1507.0)),
            {"diff": 0.0, "se": 10.0},
            "c",
        ),
        (candidate(214, None), {"diff": 0.0, "se": 10.0}, "c"),  # cost unmeasured
        (candidate(247, 1.0), {"diff": 0.0, "se": 10.0}, "c"),  # no fewer states
    ],
)
def test_clause_b_needs_every_condition(trig4, difference, clause):
    assert 1 - 2.25 / DUMMY["seconds"] == pytest.approx(COST_SAVING_MIN)
    assert LJUNG_BOX_Q_RATIO_MAX * 85.0 == 106.25
    candidates = {"dummy": DUMMY, "trig4": trig4}
    result = verdict(candidates, {"trig4": difference})
    assert result["clause"] == clause
    assert result["layout"] == ("trig4" if clause == "b" else "dummy")


def test_choose_layout_drops_blocks_unreliable_in_any_layout():
    candidates = {"dummy": DUMMY, "trig": candidate(247, 3.0)}
    pointwise = {
        "dummy": np.zeros(4),
        "trig": np.array([3.0, 3.0, 3.0, 4.0]),
    }
    pareto_k = {
        "dummy": np.array([0.1, 0.2, 0.3, 1.9]),
        "trig": np.array([0.1, 0.2, PARETO_K_MAX, 0.4]),
    }
    result = choose_layout(candidates, pointwise, pareto_k)
    assert result["unreliable_blocks"] == 1
    assert result["all_blocks"]["differences"]["trig"]["num_blocks"] == 4
    assert result["reliable_blocks"]["differences"]["trig"]["num_blocks"] == 3
    # All four blocks: diff 13, se sqrt(4) * 0.5 = 1, so clause a picks trig. The three
    # reliable blocks differ by exactly 3 each: diff 9, se 0, so clause a again.
    assert result["all_blocks"]["verdict"] == {"layout": "trig", "clause": "a"}
    assert result["reliable_blocks"]["verdict"] == {"layout": "trig", "clause": "a"}
    assert result["reliable"] is True
    assert result["layout"] == "trig"


def test_choose_layout_keeps_the_dummy_when_the_verdicts_disagree():
    candidates = {"dummy": DUMMY, "trig": candidate(247, 3.0)}
    pointwise = {
        "dummy": np.zeros(4),
        "trig": np.array([3.0, 3.0, 3.0, -40.0]),
    }
    pareto_k = {"dummy": np.array([0.1, 0.1, 0.1, 1.2]), "trig": np.full(4, 0.1)}
    result = choose_layout(candidates, pointwise, pareto_k)
    # The unreliable block's -40 sinks trig on all blocks; without it, clause a holds.
    assert result["all_blocks"]["verdict"]["layout"] == "dummy"
    assert result["reliable_blocks"]["verdict"] == {"layout": "trig", "clause": "a"}
    assert result["reliable"] is False
    assert result["layout"] == "dummy"


def test_choose_layout_agrees_on_the_layout_whatever_the_clause():
    candidates = {"dummy": DUMMY, "trig4": candidate(214, 2.0)}
    pointwise = {"dummy": np.zeros(4), "trig4": np.array([3.0, 3.0, 3.1, -10.0])}
    pareto_k = {"dummy": np.array([0.1, 0.1, 0.1, 1.2]), "trig4": np.full(4, 0.1)}
    result = choose_layout(candidates, pointwise, pareto_k)
    # All blocks: diff -0.9 within 2 * 13.0, so clause b on cost. The three reliable
    # blocks: diff 9.1 beyond 2 * 0.1, so clause a. Both choose trig4.
    assert result["all_blocks"]["verdict"] == {"layout": "trig4", "clause": "b"}
    assert result["reliable_blocks"]["verdict"] == {"layout": "trig4", "clause": "a"}
    assert result["reliable"] is True
    assert result["layout"] == "trig4"


def grid_entry(layout, stage, wedge, steps, batches):
    return {**stage_dimensions(layout, stage, wedge, steps), "batches": batches}


def test_grid_seconds_reads_the_stage_9_low_full_span_batch_1_median():
    grid = {
        "entries": [
            grid_entry("dummy", 9, "low", 280, [{"batch": 1, "median_seconds": 1.5}]),
            grid_entry("dummy", 9, "high", 572, [{"batch": 1, "median_seconds": 5.6}]),
            grid_entry(
                "dummy",
                9,
                "low",
                572,
                [
                    {"batch": 4, "median_seconds": 10.7},
                    {"batch": 1, "median_seconds": 2.99},
                ],
            ),
            grid_entry("trig4", 9, "low", 572, [{"batch": 1, "error": "skipped"}]),
        ]
    }
    assert grid_seconds(grid, "dummy") == 2.99
    assert grid_seconds(grid, "trig") == 2.99  # trig has the dummy's dimensions
    assert grid_seconds(grid, "trig4") is None  # skipped cells carry no median
```

Create `tests/test_pilot_gpu_choice.py`:

```python
"""The GPU projection and per-stage choice rule, on hand-made probe-grid records."""

import math

import pytest

from ces_revisions.pilot.dimensions import STAGE_CONFIGURATIONS, stage_dimensions
from ces_revisions.pilot.gpu_choice import (
    BATCH_LOCKSTEP_FACTORS,
    batch_timings,
    batched_sbc,
    choose_gpu,
    choose_stage,
    configurations,
)

FIT_STEPS = 2_000 * 127  # iterations times leapfrog steps per iteration
SBC_STEPS = 1_490 * 127
FIT_TARGET = {"stage": 9, "wedge": "high", "steps": 572}


def entry(stage, wedge, steps, batches, layout_name="dummy"):
    """A grid entry with the layout's dimensions; None marks a batch that did not fit."""
    tried = [
        {"batch": b, "median_seconds": s}
        if s is not None
        else {"batch": b, "error": "x"}
        for b, s in batches.items()
    ]
    return {**stage_dimensions(layout_name, stage, wedge, steps), "batches": tried}


def record(label, price, fit, sbc=None, four_gpu_price=None, layout_name="dummy"):
    """A record with Stage 9 high's fit entry, optionally its SBC entry, and one entry of
    another stage, which the Stage 9 high rule must ignore."""
    entries = [entry(9, "high", 572, fit, layout_name)]
    if sbc is not None:
        entries.append(entry(9, "high", 280, sbc, layout_name))
    entries.append(entry(9, "low", 572, {1: 99.0}, layout_name))
    return {
        "label": label,
        "hourly_price_usd": price,
        "four_gpu_hourly_price_usd": four_gpu_price,
        "entries": entries,
    }


A100_SBC = {1: 0.05, 4: 0.15, 16: 0.4}
A10 = record("a10", 1.29, {1: 0.5, 4: None}, {1: 0.08, 4: 0.2})
A100 = record("a100", 1.99, {1: 0.2, 4: 0.3}, A100_SBC, four_gpu_price=7.96)
MAC = record("mac", None, {1: 5.6, 4: 20.0}, {1: 2.0})

# Relative cost of each Stage 7-9 model, for records that cover the whole grid.
SCALE = {(7, "low"): 1.0, (8, "low"): 1.2, (8, "high"): 2.5, (9, "low"): 1.5}
SCALE[(9, "high")] = 3.0


def full_record(label, price, fit4, fit1, sbc1, *, fits_from=None, skip=()):
    """A record over every Stage 7-9 model: batch 4 takes ``fit4`` times the model's
    scale, and does not fit from scale ``fits_from`` up; ``skip`` drops models."""
    entries = []
    for stage, wedge in STAGE_CONFIGURATIONS:
        if (stage, wedge) in skip:
            continue
        scale = SCALE[(stage, wedge)]
        batch4 = None if fits_from is not None and scale >= fits_from else fit4 * scale
        entries.append(entry(stage, wedge, 572, {1: fit1 * scale, 4: batch4}))
        entries.append(entry(stage, wedge, 280, {1: sbc1 * scale}))
    return {
        "label": label,
        "hourly_price_usd": price,
        "four_gpu_hourly_price_usd": None,
        "entries": entries,
    }


A10_FULL = full_record("a10", 1.29, 0.2, 0.15, 0.03, fits_from=2.5)
GH200_FULL = full_record("gh200", 2.29, 0.15, 0.12, 0.02)


def test_batch_timings_read_every_tried_batch_at_the_layouts_dimensions():
    assert batch_timings(A10, "dummy", FIT_TARGET) == {1: 0.5, 4: None}
    # trig has the dummy's dimensions, so it reads the dummy's entries; trig4 does not.
    assert batch_timings(A10, "trig", FIT_TARGET) == {1: 0.5, 4: None}
    assert batch_timings(A10, "trig4", FIT_TARGET) is None
    trig4 = record("a10", 1.29, {1: 0.4}, layout_name="trig4")
    assert batch_timings(trig4, "trig4", FIT_TARGET) == {1: 0.4}
    assert batch_timings(trig4, "dummy", FIT_TARGET) is None


def test_configurations_price_one_gpu_and_a_rented_four_gpu_instance():
    vectorized, parallel = configurations(A100, "dummy", 9, "high")
    hours = FIT_STEPS * 1.44 * 0.3 / 3_600
    assert vectorized == {
        "stage": 9,
        "wedge": "high",
        "host": "a100",
        "gpus": 1,
        "chain_method": "vectorized",
        "batch": 4,
        "hourly_price_usd": 1.99,
        "seconds": 0.3,
        "fit_hours": pytest.approx(hours, rel=1e-12),
        "fit_usd": pytest.approx(hours * 1.99, rel=1e-12),
        "sbc_seconds": 0.05,
        # The shipped runner: one replicate at a time on the one GPU, 100 rounds.
        "sbc_days": pytest.approx(100 * SBC_STEPS * 0.05 / 86_400, rel=1e-12),
        "sbc_batched": batched_sbc(A100_SBC, gpus=1),
        "measured": True,
        "feasible": True,
    }
    hours = FIT_STEPS * 0.2 / 3_600
    assert (parallel["gpus"], parallel["chain_method"], parallel["batch"]) == (
        4,
        "parallel",
        1,
    )
    assert parallel["fit_hours"] == pytest.approx(hours, rel=1e-12)
    # The 4-GPU instance's own hourly price, not four times the 1-GPU price.
    assert parallel["fit_usd"] == pytest.approx(hours * 7.96, rel=1e-12)
    # One replicate per GPU: ceil(100 / 4) = 25 rounds.
    assert parallel["sbc_days"] == pytest.approx(25 * SBC_STEPS * 0.05 / 86_400)


def test_a_gpu_type_without_a_four_gpu_instance_has_no_parallel_row():
    rows = configurations(A10, "dummy", 9, "high")
    assert [row["chain_method"] for row in rows] == ["vectorized"]


def test_a_batch_that_did_not_fit_is_measured_and_infeasible():
    (vectorized,) = configurations(A10, "dummy", 9, "high")
    assert vectorized["seconds"] is None
    assert vectorized["fit_hours"] is None
    assert vectorized["fit_usd"] is None
    assert vectorized["measured"] is True
    assert vectorized["feasible"] is False


def test_batched_sbc_takes_the_fastest_batch_with_whole_rounds():
    result = batched_sbc({**A100_SBC, 64: None}, gpus=1)
    rounds = {1: 100, 4: 25, 16: math.ceil(100 / 16)}
    days = {
        batch: rounds[batch]
        * SBC_STEPS
        * BATCH_LOCKSTEP_FACTORS[batch]
        * seconds
        / 86_400
        for batch, seconds in A100_SBC.items()
    }
    best = min(days, key=days.get)
    assert result == {"replicates_per_gpu": best, "days": pytest.approx(days[best])}
    assert batched_sbc(None, gpus=1) is None
    assert batched_sbc({1: None}, gpus=1) is None


def test_the_cheapest_feasible_configuration_wins_and_unpriced_hosts_are_ignored():
    result = choose_stage([MAC, A10, A100], "dummy", 9, "high")
    assert (result["stage"], result["wedge"]) == (9, "high")
    assert {row["host"] for row in result["configurations"]} == {"a10", "a100"}
    assert result["status"] == "chosen"
    choice = result["choice"]
    assert (choice["host"], choice["gpus"], choice["chain_method"]) == (
        "a100",
        1,
        "vectorized",
    )
    assert set(result["band_fit_hours"]) == {"a10", "a100"}
    assert result["band_fit_hours"]["a100"]["31"][0] == pytest.approx(
        2_000 * 31 * 1.44 * 0.3 / 3_600, rel=1e-12
    )


def test_each_stage_gets_the_cheapest_gpu_that_runs_it():
    result = choose_gpu([MAC, A10_FULL, GH200_FULL], "dummy")
    assert result["status"] == "chosen"
    assert [(r["stage"], r["wedge"]) for r in result["stages"]] == list(
        STAGE_CONFIGURATIONS
    )
    chosen = {(r["stage"], r["wedge"]): r["choice"]["host"] for r in result["stages"]}
    # The A10 is cheaper wherever batch 4 fits its memory; the larger models need the
    # GH200.
    assert chosen == {
        (7, "low"): "a10",
        (8, "low"): "a10",
        (8, "high"): "gh200",
        (9, "low"): "a10",
        (9, "high"): "gh200",
    }


def test_trig_reads_the_dummys_entries_and_gets_the_same_choices():
    records = [A10_FULL, GH200_FULL]
    assert choose_gpu(records, "trig") == choose_gpu(records, "dummy")
    assert choose_gpu(records, "trig4")["status"] == "unmeasured"


def test_a_configuration_whose_sbc_is_too_slow_is_not_chosen():
    # x is cheaper per fit, but one replicate at a time takes 17.5 days of SBC.
    cheap = record("x", 1.0, {1: 1.0, 4: 0.3}, {1: 0.08})
    dear = record("y", 2.0, {1: 1.0, 4: 0.3}, {1: 0.05})
    (row,) = configurations(cheap, "dummy", 9, "high")
    assert row["fit_hours"] <= 48
    assert row["sbc_days"] == pytest.approx(17.5, abs=0.1)
    assert row["feasible"] is False
    result = choose_stage([cheap, dear], "dummy", 9, "high")
    assert (result["status"], result["choice"]["host"]) == ("chosen", "y")


def test_one_infeasible_stage_makes_the_overall_status_infeasible():
    result = choose_gpu([A10_FULL], "dummy")
    statuses = {(r["stage"], r["wedge"]): r["status"] for r in result["stages"]}
    assert statuses == {
        (7, "low"): "chosen",
        (8, "low"): "chosen",
        (8, "high"): "infeasible",
        (9, "low"): "chosen",
        (9, "high"): "infeasible",
    }
    assert result["status"] == "infeasible"


@pytest.mark.parametrize(
    "records",
    [
        [],  # nothing measured
        [MAC],  # no priced host
        [record("a10", 1.29, {1: 0.2, 4: 0.3})],  # no SBC entry
        [record("a10", 1.29, {1: 0.2}, {1: 0.05})],  # batch 4 was never tried
        [A100, record("a10", 1.29, {1: 0.2, 4: 0.3})],  # one host incomplete
    ],
)
def test_an_incomplete_measurement_is_unmeasured_not_infeasible(records):
    result = choose_stage(records, "dummy", 9, "high")
    assert (result["status"], result["choice"]) == ("unmeasured", None)


def test_a_stage_missing_from_one_record_makes_the_overall_status_unmeasured():
    partial = full_record("gh200", 2.29, 0.15, 0.12, 0.02, skip={(8, "high")})
    result = choose_gpu([A10_FULL, partial], "dummy")
    statuses = {(r["stage"], r["wedge"]): r["status"] for r in result["stages"]}
    assert statuses[(8, "high")] == "unmeasured"
    assert result["status"] == "unmeasured"


def test_an_unmeasured_stage_outranks_an_infeasible_one():
    # Stage 7 is missing from the GH200 record, and Stage 9 high fits neither GPU.
    partial = full_record(
        "gh200", 2.29, 0.15, 0.12, 0.02, fits_from=3.0, skip={(7, "low")}
    )
    result = choose_gpu([A10_FULL, partial], "dummy")
    statuses = {(r["stage"], r["wedge"]): r["status"] for r in result["stages"]}
    assert (statuses[(7, "low")], statuses[(9, "high")]) == ("unmeasured", "infeasible")
    assert result["status"] == "unmeasured"


def test_measured_steps_per_iteration_replace_the_sizing_assumption():
    result = choose_gpu([A10_FULL, GH200_FULL], "dummy", steps_per_iteration=31)
    assert result["steps_per_iteration"] == 31
    stage_7 = result["stages"][0]["configurations"][0]
    assert stage_7["fit_hours"] == pytest.approx(2_000 * 31 * 1.44 * 0.2 / 3_600)
    assert stage_7["sbc_days"] == pytest.approx(100 * 1_490 * 31 * 0.03 / 86_400)
    assert choose_gpu([A10_FULL], "dummy")["steps_per_iteration"] == 127


def test_a_tie_in_dollars_goes_to_fewer_gpus():
    # Vectorized: 1.44 * 0.4 s at $2.00; parallel: 0.144 s at $8.00. Equal per fit.
    tied = record("a100", 2.0, {1: 0.144, 4: 0.4}, {1: 0.05}, four_gpu_price=8.0)
    vectorized, parallel = configurations(tied, "dummy", 9, "high")
    assert vectorized["fit_usd"] == pytest.approx(parallel["fit_usd"], rel=1e-12)
    assert choose_stage([tied], "dummy", 9, "high")["choice"]["gpus"] == 1
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_pilot_dimensions.py tests/test_pilot_seasonal.py tests/test_pilot_gpu_choice.py -q`
Expected: collection errors, `ModuleNotFoundError` for `gpu_probe_grid`, `ces_revisions.pilot.dimensions`, `ces_revisions.pilot.seasonal`, and `ces_revisions.pilot.gpu_choice`.

- [ ] **Step 3: Write the modules and the grid script**

Create `src/ces_revisions/pilot/dimensions.py`:

```python
"""State and cell counts of the pilot and of the Stage 7-9 models, for the GPU measurement.

The engine's cost per step is cubic in the state dimension and grows with the number of
observed cells, so the Stage 6 cost measurement times the engine at the dimensions the
later models will have. Each count below names its terms; ``S`` is the seasonal block's
size under the chosen layout, which is the largest share of each sector's state.
"""

# Seasonal states per series: the dummy recursion and full trigonometric form carry 11,
# the reduced trigonometric form (harmonics 1-4, no Nyquist term) carries 8.
SEASONAL_DIMS = {"dummy": 11, "trig": 11, "trig4": 8}
# The pilot's cells per month: the F, S, and T closing changes and the March anchor.
PILOT_CELLS = 4
# Level, slope, and the lagged value x_{t-1} that turns levels into same-release changes.
TREND_STATES = 3
# Per-series news states at Stage 6 and 7: S news, T news, and post-T news.
CLOSING_NEWS_STATES = 3
# From Stage 8 the post-T news splits into its benchmark (B) and post-benchmark (M) parts.
BENCHMARK_NEWS_STATES = 4
# Stage 9's sector-specific factor news n^A, one state per news stage (S, T, B, M).
FACTOR_NEWS_STATES = 4
SECTORS = 11  # Req 5's eleven supersectors
SERIES = 12  # the eleven supersectors and total nonfarm
COMMON_FACTORS = 1  # the common level factor
CLOSINGS = 3  # F, S, and T rows per series
# Wedge "high" carries the benchmark's backward window as 11 lagged months per sector.
WEDGE_WINDOW = 11
STAGES = (7, 8, 9)
WEDGES = ("low", "high")
# Every Stage 7-9 model the GPU measurement times: Stage 7 has no wedge rows, and Stage 8
# chooses the wedge reading that Stage 9 inherits, so both readings are timed.
STAGE_CONFIGURATIONS = ((7, "low"), (8, "low"), (8, "high"), (9, "low"), (9, "high"))
# 1979-01 through 2026-08, the aggregate leg's full span, and 2003-05 through 2026-08, the
# sector panel's span (Req 4: no sector vintage precedes May 2003).
FULL_STEPS = 572
SECTOR_STEPS = 280


def pilot_dimensions(layout_name: str, num_steps: int = FULL_STEPS) -> dict:
    """The Stage 6 pilot's dimensions: one series with trend, seasonal, and news states."""
    seasonal = _seasonal_dim(layout_name)
    return {
        "steps": num_steps,
        "states": TREND_STATES + seasonal + CLOSING_NEWS_STATES,
        "cells": PILOT_CELLS,
        "stage": 6,
        "layout": layout_name,
        "wedge": None,
    }


def stage_dimensions(
    layout_name: str, stage: int, wedge: str = "low", num_steps: int = FULL_STEPS
) -> dict:
    """The Stage 7, 8, or 9 model's state and cell counts under a layout and wedge.

    Stage 7 has no wedge rows, so it accepts only ``wedge="low"``, which adds nothing.
    """
    seasonal = _seasonal_dim(layout_name)
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, not {stage}")
    if wedge not in WEDGES:
        raise ValueError(f"wedge must be one of {WEDGES}, not {wedge!r}")
    if stage == 7 and wedge != "low":
        raise ValueError("Stage 7 has no wedge rows: wedge must be 'low'")

    # Every series carries its F, S, and T closing rows; total nonfarm adds its anchor.
    closing_cells = SERIES * CLOSINGS + 1
    if stage == 7:
        per_sector = TREND_STATES + seasonal + CLOSING_NEWS_STATES
        # One common factor and the two cross-sector news factors, for S and T.
        factors = COMMON_FACTORS + 2
        states = SECTORS * per_sector + factors
        cells = closing_cells
    else:
        per_sector = TREND_STATES + seasonal + BENCHMARK_NEWS_STATES
        if stage == 9:
            per_sector += FACTOR_NEWS_STATES
        # One common factor and the four cross-sector news factors, for S, T, B, and M.
        factors = COMMON_FACTORS + BENCHMARK_NEWS_STATES
        states = SECTORS * per_sector + factors
        # Per series: the B and M rows, then the preliminary and final benchmark rows.
        cells = closing_cells + SERIES * 2 + SERIES * 2
        if wedge == "high":
            states += SECTORS * WEDGE_WINDOW
            # The benchmark's backward window enters as an 11-month B vector per series.
            cells += SERIES * WEDGE_WINDOW
        if stage == 9:
            # Stage 9 adds the seasonally adjusted F, S, T, B, and M rows per series.
            cells += SERIES * 5
    return {
        "steps": num_steps,
        "states": states,
        "cells": cells,
        "stage": stage,
        "layout": layout_name,
        "wedge": wedge,
    }


def matches_layout(entry: dict, layout_name: str) -> bool:
    """Whether a probe-grid entry has a layout's Stage 7-9 dimensions.

    The entry's states and cells must equal ``stage_dimensions`` at its own stage, wedge,
    and steps. ``trig`` and the dummy have equal dimensions, so each reads the other's
    entries; an entry without a Stage 7-9 configuration matches no layout.
    """
    try:
        expected = stage_dimensions(
            layout_name, entry.get("stage"), entry.get("wedge"), entry["steps"]
        )
    except ValueError:
        return False
    return all(entry[key] == expected[key] for key in ("states", "cells"))


def _seasonal_dim(layout_name: str) -> int:
    if layout_name not in SEASONAL_DIMS:
        raise ValueError(
            f"layout must be one of {tuple(SEASONAL_DIMS)}, not {layout_name!r}"
        )
    return SEASONAL_DIMS[layout_name]


# The configurations the Stage 6 cost measurement times on each GPU type: the dummy (whose
# dimensions ``trig`` shares) and ``trig4``, every Stage 7-9 model, at the full span (a
# fit) and the sector span (an SBC replicate).
MEASUREMENT_GRID = tuple(
    stage_dimensions(layout_name, stage, wedge, num_steps)
    for layout_name in ("dummy", "trig4")
    for stage, wedge in STAGE_CONFIGURATIONS
    for num_steps in (FULL_STEPS, SECTOR_STEPS)
)
```

Create `src/ces_revisions/pilot/seasonal.py`:

```python
"""The pre-registered rule that chooses the seasonal layout for Stages 7-17.

Req 6 keeps the dummy recursion unless "diagnostics favor" a trigonometric state. The rule
compares each alternative with the dummy on benchmark-year PSIS-LOO blocks, which are
close to independent, so their elpd difference has an honest standard error:

(a) an alternative that meets Req 18, passes its April 2020 check, and whose elpd beats
    the dummy's by more than SE_MULTIPLE standard errors replaces it (the largest such
    difference wins);
(b) otherwise an alternative with fewer states replaces it when its elpd is within
    SE_MULTIPLE standard errors, it meets Req 18, it passes its April 2020 check, its
    Ljung-Box Q is at most LJUNG_BOX_Q_RATIO_MAX times the dummy's in every closing
    column, and its measured value-and-gradient time is at least COST_SAVING_MIN below
    the dummy's;
(c) otherwise the dummy stays.

A block whose Pareto k exceeds PARETO_K_MAX in any layout makes PSIS unreliable there, so
the rule runs twice, on every block and on the blocks reliable in every layout. When the
two runs choose different layouts, the comparison is unreliable and the dummy stays.
"""

import numpy as np

from ces_revisions.pilot.dimensions import matches_layout

REFERENCE_LAYOUT = "dummy"
SE_MULTIPLE = 2.0
COST_SAVING_MIN = 0.25
PARETO_K_MAX = 0.7
LJUNG_BOX_ALPHA = 0.01
# Clause (b)'s residual guard: Q under the white-noise null has a few units of SD, while a
# layout that misses the seasonal pattern multiplies it.
LJUNG_BOX_Q_RATIO_MAX = 1.25
# Clause (b)'s cost: one value-and-gradient at the Stage 9 low-wedge dimensions over the
# full 572 months, batch 1, from a gpu_probe_grid.py record. Stage 9 is the largest model
# the layout choice sizes; the high wedge waits on Stage 8's choice.
COST_ENTRY = {"stage": 9, "wedge": "low", "steps": 572}
COST_BATCH = 1


def elpd_difference(
    alternative: np.ndarray, reference: np.ndarray, keep: np.ndarray
) -> dict:
    """The summed pointwise elpd difference over the kept blocks and its standard error.

    As in ``az.compare``, the standard error is sqrt(B) times the SD of the B pointwise
    differences.
    """
    difference = (np.asarray(alternative) - np.asarray(reference))[keep]
    return {
        "diff": float(difference.sum()),
        "se": float(np.sqrt(difference.size) * difference.std(ddof=1)),
        "num_blocks": int(difference.size),
    }


def ljung_box_rejections(ljung_box: dict[str, dict]) -> int:
    """How many closing columns' Ljung-Box tests reject at LJUNG_BOX_ALPHA."""
    return sum(test["p_value"] < LJUNG_BOX_ALPHA for test in ljung_box.values())


def verdict(candidates: dict[str, dict], differences: dict[str, dict]) -> dict:
    """The layout the rule chooses, with the clause that chose it.

    ``candidates`` maps every layout, the reference included, to its ``states`` (int),
    ``meets_req18`` (bool), ``april_2020_passed`` (bool), ``ljung_box_q`` (closing
    column to Q), and ``seconds`` (the measured value-and-gradient time, or None when it
    was not measured). ``differences`` maps each alternative to its ``elpd_difference``
    against the reference.
    """
    reference = candidates[REFERENCE_LAYOUT]
    better = {
        name: difference["diff"]
        for name, difference in differences.items()
        if difference["diff"] > SE_MULTIPLE * difference["se"]
        and _eligible(candidates[name])
    }
    if better:
        return {"layout": max(better, key=better.get), "clause": "a"}
    cheaper = [
        name
        for name, difference in differences.items()
        if candidates[name]["states"] < reference["states"]
        and abs(difference["diff"]) <= SE_MULTIPLE * difference["se"]
        and _eligible(candidates[name])
        and _residuals_hold(candidates[name], reference)
        and _saving(candidates[name], reference) >= COST_SAVING_MIN
    ]
    if cheaper:
        return {
            "layout": min(cheaper, key=lambda name: candidates[name]["seconds"]),
            "clause": "b",
        }
    return {"layout": REFERENCE_LAYOUT, "clause": "c"}


def choose_layout(
    candidates: dict[str, dict],
    pointwise: dict[str, np.ndarray],
    pareto_k: dict[str, np.ndarray],
) -> dict:
    """The rule on every block and on the reliable blocks, and the layout it settles on.

    ``pointwise`` and ``pareto_k`` map every layout to its benchmark-year blocks' elpd and
    Pareto k, in the same block order.
    """
    reference = np.asarray(pointwise[REFERENCE_LAYOUT])
    reliable = np.all(
        [np.asarray(k) <= PARETO_K_MAX for k in pareto_k.values()], axis=0
    )
    alternatives = [name for name in candidates if name != REFERENCE_LAYOUT]
    runs = {}
    for label, keep in (
        ("all_blocks", np.ones(reference.size, dtype=bool)),
        ("reliable_blocks", reliable),
    ):
        differences = {
            name: elpd_difference(pointwise[name], reference, keep)
            for name in alternatives
        }
        runs[label] = {
            "differences": differences,
            "verdict": verdict(candidates, differences),
        }
    agree = (
        runs["all_blocks"]["verdict"]["layout"]
        == runs["reliable_blocks"]["verdict"]["layout"]
    )
    return {
        **runs,
        "unreliable_blocks": int((~reliable).sum()),
        "reliable": agree,
        "layout": runs["all_blocks"]["verdict"]["layout"]
        if agree
        else REFERENCE_LAYOUT,
    }


def grid_seconds(grid: dict, layout_name: str) -> float | None:
    """The COST_ENTRY median seconds at a layout's dimensions in a grid record, if any."""
    for entry in grid["entries"]:
        if matches_layout(entry, layout_name) and all(
            entry[key] == value for key, value in COST_ENTRY.items()
        ):
            for batch in entry["batches"]:
                if batch["batch"] == COST_BATCH and "median_seconds" in batch:
                    return batch["median_seconds"]
    return None


def _eligible(candidate: dict) -> bool:
    return candidate["meets_req18"] and candidate["april_2020_passed"]


def _residuals_hold(candidate: dict, reference: dict) -> bool:
    """Whether the candidate's Q is within the ratio of the reference's in every column."""
    return all(
        candidate["ljung_box_q"][column] <= LJUNG_BOX_Q_RATIO_MAX * q
        for column, q in reference["ljung_box_q"].items()
    )


def _saving(candidate: dict, reference: dict) -> float:
    """The candidate's fractional time saving over the reference; -inf when unmeasured."""
    if candidate["seconds"] is None or reference["seconds"] is None:
        return -np.inf
    return 1.0 - candidate["seconds"] / reference["seconds"]
```

Create `src/ces_revisions/pilot/gpu_choice.py`:

```python
"""Project Stage 7-9 fit and SBC cost from probe-grid records; choose a GPU for each stage.

Each rented host's record (``scripts/gpu_probe_grid.py --hourly-price``) times one value
and gradient of the engine at the Stage 7-9 dimensions, for batches of chains or SBC
replicates run at once. A grid entry belongs to a layout when its states and cells are
that layout's at the entry's stage, wedge, and steps, so ``trig``, whose dimensions equal
the dummy's, reads the dummy's entries.

The rule chooses a GPU type and count for each Stage 7-9 model of STAGE_CONFIGURATIONS
(decision D5), from that model's own dimensions: a fit over FIT_STEPS months and an SBC
replicate over SBC_STEPS. Stages 10-17 sample more parameters but keep Stage 9's Kalman
states and cells (Req 17). While they also keep Stage 9's engine and NUTS kernel, they
take Stage 9's choice at the wedge Stage 8 settles, rerun with their own measured steps
per iteration.

A fit runs NUM_CHAINS chains of ITERATIONS NUTS iterations, and every leapfrog step costs
one value and gradient. On one GPU the chains run as one batch (NumPyro's "vectorized"
chain method), so each iteration waits for the chain with the deepest tree. With one GPU
per chain they run side by side ("parallel") at batch 1; that configuration exists only
where Lambda rents the GPU type as a NUM_CHAINS-GPU instance, whose hourly price the
record carries (``--four-gpu-price``).

SBC runs SBC_REPLICATES single-chain replicates over SBC_STEPS, the reduced panel's span.
The projection assumes one replicate per GPU at a time, at batch 1, in
ceil(SBC_REPLICATES / GPUs) rounds. That is the shipped runner's throughput
(``ces_revisions.pilot.sbc.run_sbc``): it runs one replicate per process and one worker
on a GPU host, so a multi-GPU instance runs one shard per GPU under its own
``CUDA_VISIBLE_DEVICES``. The record also projects a batched runner, which Stage 7 would
have to build, for reference; that projection decides nothing.

The pre-registered rule, for each stage:

1. A configuration is measured when its record holds the stage's fit and SBC entries at
   the layout's dimensions with the batches it needs. A batch that did not fit device
   memory counts as measured and infeasible.
2. A configuration is feasible when its fit takes at most FIT_HOURS_MAX hours and its SBC
   at most SBC_DAYS_MAX days.
3. When every configuration is measured, the cheapest feasible one in dollars per fit, to
   the cent, wins ("chosen"); a tie goes to fewer GPUs. With none feasible the stage's
   status is "infeasible".
4. When no host is priced or any configuration is unmeasured, the status is
   "unmeasured" and nothing is chosen: the measurement is incomplete, not a finding.

The overall status is "unmeasured" if any stage's is, else "infeasible" if any stage's
is, and then the roadmap's resume re-validates Stages 7-17 before it routes Stage 7;
otherwise it is "chosen".
"""

import math

from ces_revisions.pilot.dimensions import (
    FULL_STEPS,
    SECTOR_STEPS,
    STAGE_CONFIGURATIONS,
    matches_layout,
)

NUM_CHAINS = 4
ITERATIONS = 2_000  # 1,000 warmup and 1,000 sampling iterations per chain (Req 18 runs)
# Leapfrog steps per iteration of the slowest chain. The pilot's 17 parameters took about
# 65 over warmup and sampling; Stages 7-9 sample far more parameters, so the rule sizes
# for depth-7 trees, the top of the 31-127 band the projections report.
STEPS_PER_ITERATION = 127
STEP_BAND = (31, 63, 127)
# A vectorized batch takes each iteration's deepest tree across chains; the pilot's
# lockstep total over its slowest chain's was 94.2 / 65.3 steps per iteration.
LOCKSTEP_FACTOR = 1.44
FIT_HOURS_MAX = 48.0
FIT_STEPS = FULL_STEPS  # a fit covers 1979-01 through 2026-08
SBC_STEPS = SECTOR_STEPS  # an SBC replicate covers the sector panel's span
SBC_REPLICATES = 100
SBC_ITERATIONS = 1_490  # 500 warmup and 990 sampling iterations, one chain each
SBC_DAYS_MAX = 14.0
# The batched reference projection's lockstep factor by replicates per batch: over the
# planning spike's 100 SBC replicates, the expected largest of b replicates' mean steps
# per iteration over their average. Trees also vary within a replicate, so these
# understate the lockstep cost.
BATCH_LOCKSTEP_FACTORS = {1: 1.0, 4: 1.331, 16: 1.637, 64: 1.991}
SECONDS_PER_HOUR = 3_600.0
SECONDS_PER_DAY = 86_400.0


def batch_timings(
    record: dict, layout_name: str, target: dict
) -> dict[int, float | None] | None:
    """Median seconds of every batch tried at a target, by batch size.

    ``target`` gives the entry's stage, wedge, and steps. None when the record holds no
    entry at the target and the layout's dimensions. A batch that did not fit device
    memory maps to None.
    """
    for entry in record["entries"]:
        if matches_layout(entry, layout_name) and all(
            entry[key] == value for key, value in target.items()
        ):
            return {
                batch["batch"]: batch.get("median_seconds")
                for batch in entry["batches"]
            }
    return None


def configurations(
    record: dict,
    layout_name: str,
    stage: int,
    wedge: str,
    steps_per_iteration: int = STEPS_PER_ITERATION,
) -> list[dict]:
    """The host's configurations for one stage, each with its fit cost and SBC days."""
    fit = batch_timings(
        record, layout_name, {"stage": stage, "wedge": wedge, "steps": FIT_STEPS}
    )
    sbc = batch_timings(
        record, layout_name, {"stage": stage, "wedge": wedge, "steps": SBC_STEPS}
    )
    shapes = [
        (1, "vectorized", NUM_CHAINS, LOCKSTEP_FACTOR, record["hourly_price_usd"])
    ]
    if record.get("four_gpu_hourly_price_usd") is not None:
        shapes.append(
            (NUM_CHAINS, "parallel", 1, 1.0, record["four_gpu_hourly_price_usd"])
        )
    sbc_seconds = None if sbc is None else sbc.get(1)
    rows = []
    for gpus, chain_method, batch, factor, price in shapes:
        seconds = None if fit is None else fit.get(batch)
        hours = (
            None
            if seconds is None
            else ITERATIONS * steps_per_iteration * factor * seconds / SECONDS_PER_HOUR
        )
        days = (
            None
            if sbc_seconds is None
            else math.ceil(SBC_REPLICATES / gpus)
            * SBC_ITERATIONS
            * steps_per_iteration
            * sbc_seconds
            / SECONDS_PER_DAY
        )
        measured = fit is not None and batch in fit and sbc is not None and 1 in sbc
        rows.append(
            {
                "stage": stage,
                "wedge": wedge,
                "host": record["label"],
                "gpus": gpus,
                "chain_method": chain_method,
                "batch": batch,
                "hourly_price_usd": price,
                "seconds": seconds,
                "fit_hours": hours,
                "fit_usd": None if hours is None else hours * price,
                "sbc_seconds": sbc_seconds,
                "sbc_days": days,
                "sbc_batched": batched_sbc(sbc, gpus, steps_per_iteration),
                "measured": measured,
                "feasible": hours is not None
                and hours <= FIT_HOURS_MAX
                and days is not None
                and days <= SBC_DAYS_MAX,
            }
        )
    return rows


def batched_sbc(
    sbc: dict[int, float | None] | None,
    gpus: int,
    steps_per_iteration: int = STEPS_PER_ITERATION,
) -> dict | None:
    """The fastest batched-runner SBC projection over the measured batches, if any."""
    options = [
        {
            "replicates_per_gpu": batch,
            "days": math.ceil(SBC_REPLICATES / (batch * gpus))
            * SBC_ITERATIONS
            * steps_per_iteration
            * BATCH_LOCKSTEP_FACTORS[batch]
            * seconds
            / SECONDS_PER_DAY,
        }
        for batch, seconds in (sbc or {}).items()
        if seconds is not None and batch in BATCH_LOCKSTEP_FACTORS
    ]
    return min(options, key=lambda option: option["days"], default=None)


def choose_stage(
    records: list[dict],
    layout_name: str,
    stage: int,
    wedge: str,
    steps_per_iteration: int = STEPS_PER_ITERATION,
) -> dict:
    """One stage's configurations over every priced host, its band, status, and choice."""
    priced = [record for record in records if record["hourly_price_usd"] is not None]
    rows = [
        row
        for record in priced
        for row in configurations(
            record, layout_name, stage, wedge, steps_per_iteration
        )
    ]
    band = {
        record["label"]: {
            str(steps): [
                row["fit_hours"]
                for row in configurations(record, layout_name, stage, wedge, steps)
            ]
            for steps in STEP_BAND
        }
        for record in priced
    }
    feasible = [row for row in rows if row["feasible"]]
    if not rows or not all(row["measured"] for row in rows):
        status, choice = "unmeasured", None
    elif not feasible:
        status, choice = "infeasible", None
    else:
        status = "chosen"
        choice = min(feasible, key=lambda row: (round(row["fit_usd"], 2), row["gpus"]))
    return {
        "stage": stage,
        "wedge": wedge,
        "configurations": rows,
        "band_fit_hours": band,
        "status": status,
        "choice": choice,
    }


def choose_gpu(
    records: list[dict],
    layout_name: str,
    steps_per_iteration: int = STEPS_PER_ITERATION,
) -> dict:
    """The rule for every Stage 7-9 model, and the overall status.

    ``steps_per_iteration`` replaces the rule's sizing assumption when a later stage has
    measured its own leapfrog steps per iteration.
    """
    stages = [
        choose_stage(records, layout_name, stage, wedge, steps_per_iteration)
        for stage, wedge in STAGE_CONFIGURATIONS
    ]
    statuses = {result["status"] for result in stages}
    if "unmeasured" in statuses:
        status = "unmeasured"
    elif "infeasible" in statuses:
        status = "infeasible"
    else:
        status = "chosen"
    return {
        "steps_per_iteration": steps_per_iteration,
        "stages": stages,
        "status": status,
    }
```

Create `scripts/gpu_probe_grid.py`:

```python
"""Time the Kalman engine at the Stage 7-9 model dimensions on this host.

For each configuration of ``MEASUREMENT_GRID`` (or of a ``--grid-json`` override) the
script builds the engine probe's synthetic model at that configuration's steps, states,
and cells, and times the vmapped value and gradient at each batch size, where the batch
stands for the chains or SBC replicates run at once. Each batch size is timed on its own,
so one that exhausts device memory is recorded, larger ones are skipped without
compiling, and the next configuration still runs. Each compiled batch also records XLA's
compiled memory analysis, which needs no device statistics and so sizes memory on the Mac
too. The JSON record is rewritten after every configuration, so an interrupted run on a
rented GPU keeps its finished entries. On a host with an NVIDIA device the script refuses
to run unless JAX's backend is the GPU, so a driver or plugin fault cannot produce a CPU
record under a GPU label.
"""

import argparse
import json
import os
import platform
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import jax
import numpy as np
import numpyro

from ces_revisions import engine_probe
from ces_revisions.devices import (
    DETERMINISTIC_GPU_FLAG,
    has_nvidia_device,
    jax_versions,
    nvidia_driver,
)
from ces_revisions.kalman import LinearGaussianSSM
from ces_revisions.pilot.dimensions import MEASUREMENT_GRID, stage_dimensions

DEFAULT_BATCHES = (1, 4, 16, 64)
DEFAULT_REPEATS = 5
DEFAULT_STEPS = (572, 280)
# The engine probe's default share of NaN cells; the filter's cost does not depend on it.
MISSING_SHARE = 0.2
SEED = sum(map(ord, "ces-revisions-stage-6-gpu-probe-grid"))
FILTER_KEYS = ("layout", "stage", "wedge")
DIMENSION_KEYS = ("steps", "states", "cells")


def measurement_entries(
    steps: list[int] | None = None,
    filters: list[str] | None = None,
    grid: list[dict] | None = None,
) -> list[dict]:
    """The configurations to measure, in grid order.

    Without ``grid``, every (layout, stage, wedge) configuration of MEASUREMENT_GRID at
    each of ``steps`` (default 572 and 280, which reproduces the grid). A ``grid`` lists
    complete entries and takes no ``steps``. A filter ``key=value`` keeps entries whose
    key has one of the given values: values of one key are alternatives, keys combine.
    """
    if grid is None:
        configurations = dict.fromkeys(
            (entry["layout"], entry["stage"], entry["wedge"])
            for entry in MEASUREMENT_GRID
        )
        entries = [
            stage_dimensions(layout_name, stage, wedge, num_steps)
            for layout_name, stage, wedge in configurations
            for num_steps in steps or DEFAULT_STEPS
        ]
    elif steps is not None:
        raise ValueError(
            "a grid lists its own steps: give --steps or --grid-json, not both"
        )
    else:
        entries = [_validated_grid_entry(entry) for entry in grid]
    wanted = _parse_filters(filters or [])
    return [
        entry
        for entry in entries
        if all(str(entry.get(key)) in values for key, values in wanted.items())
    ]


def measure_entry(
    entry: dict,
    batch_sizes: list[int],
    repeats: int,
    missing_share: float,
    seed: int,
    max_bytes: int | None = None,
) -> dict:
    """Timings and memory of the engine's value and gradient at one entry's dimensions.

    A batch runs only when its compiled memory fits the budget: the smaller of
    ``max_bytes`` and the device's ``bytes_limit``, whichever exist. A CPU reports no
    limit, so ``max_bytes`` keeps a large batch from swapping the host. Once a batch has
    not fit, larger batches are skipped without compiling, so their memory analysis is
    None.
    """
    ssm, panel = engine_probe.synthetic_problem(
        entry["steps"], entry["states"], entry["cells"], missing_share, seed
    )
    limits = [max_bytes, _device_memory_stats().get("bytes_limit")]
    budget = min((limit for limit in limits if limit is not None), default=None)
    batches, exhausted_at = [], None
    for batch in batch_sizes:
        if exhausted_at is not None and batch >= exhausted_at:
            batches.append(
                {
                    "batch": batch,
                    "memory_analysis": None,
                    "error": f"skipped: batch {exhausted_at} did not fit in memory",
                }
            )
            continue
        memory = _compiled_memory(ssm, panel, batch)
        result = {"batch": batch, "memory_analysis": memory}
        if budget is not None and memory["total_bytes"] > budget:
            result["error"] = f"skipped: compiled memory exceeds {budget} bytes"
            exhausted_at = batch
        else:
            try:
                (timing,) = engine_probe.time_batches(
                    ssm, panel, [batch], repeats, seed
                )
                result.update(timing)
            except jax.errors.JaxRuntimeError as error:
                if "RESOURCE_EXHAUSTED" not in str(error):
                    raise
                result["error"] = f"RESOURCE_EXHAUSTED: {str(error).splitlines()[0]}"
                exhausted_at = batch
        batches.append(result)
    stats = _device_memory_stats()
    return {
        **entry,
        "missing_share": missing_share,
        "seed": seed,
        "repeats": repeats,
        "max_bytes": max_bytes,
        "batches": batches,
        # The device's high-water mark for the whole process so far, read after this
        # entry's largest batch: an earlier, larger entry raises it for later ones.
        "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
        "load_average": list(os.getloadavg()),
    }


def host_record() -> dict:
    """The backend, devices, driver, and XLA flags the timings were taken under."""
    device = jax.devices()[0]
    return {
        "backend": jax.default_backend(),
        "device_kind": device.device_kind,
        "device_count": jax.device_count(),
        "bytes_limit": _device_memory_stats().get("bytes_limit"),
        "nvidia_driver": nvidia_driver(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
    }


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, measure every entry, and write the JSON record."""
    parser = _parser()
    args = parser.parse_args(argv)
    grid = json.loads(args.grid_json.read_text()) if args.grid_json else None
    try:
        entries = measurement_entries(args.steps, args.grid_filter, grid)
    except ValueError as error:
        parser.error(str(error))
    if not entries:
        parser.error("no grid entry matches the filters")
    if args.four_gpu_price is not None and (
        args.hourly_price is None or args.four_gpu_price < 2 * args.hourly_price
    ):
        parser.error(
            "--four-gpu-price is the whole 4-GPU instance's hourly price, 4 times the "
            "per-GPU price Lambda lists, and needs --hourly-price"
        )
    # The fits run with XLA's deterministic GPU operations, so the timings do too. Like
    # float64, the flag must be set before the first JAX operation.
    flags = os.environ.get("XLA_FLAGS", "").split()
    if has_nvidia_device() and DETERMINISTIC_GPU_FLAG not in flags:
        os.environ["XLA_FLAGS"] = " ".join([*flags, DETERMINISTIC_GPU_FLAG])
    numpyro.enable_x64()
    if has_nvidia_device() and jax.default_backend() != "gpu":
        parser.error(
            f"an NVIDIA device is present but JAX's backend is "
            f"{jax.default_backend()!r}: check nvidia-smi and uv sync --extra cuda"
        )
    started = datetime.now(UTC)
    record = {
        "label": args.label,
        "hourly_price_usd": args.hourly_price,
        "four_gpu_hourly_price_usd": args.four_gpu_price,
        "host": host_record(),
        "versions": jax_versions(),
        "entries": [],
        "timestamp_utc": started.isoformat(timespec="seconds"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write(record, args.out)
    for entry in entries:
        measured = measure_entry(
            entry,
            args.batch,
            args.repeats,
            args.missing_share,
            args.seed,
            args.max_bytes,
        )
        record["entries"].append(measured)
        _write(record, args.out)
        print(_summary(measured), flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/gpu_probe_grid.py",
        description="Time the Kalman engine at the Stage 7-9 model dimensions.",
    )
    parser.add_argument("--label", required=True, help="host label, e.g. mac or a100")
    parser.add_argument("--out", type=Path, required=True, help="JSON record to write")
    parser.add_argument(
        "--batch",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCHES),
        help="batch sizes to time (chains or SBC replicates run at once)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="timed calls per batch after compiling",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help=f"time steps per grid configuration (default {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--grid-filter",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help=f"keep entries matching every key; keys {FILTER_KEYS}",
    )
    parser.add_argument(
        "--grid-json",
        type=Path,
        default=None,
        help="JSON list of entries with steps, states, and cells, replacing the grid",
    )
    parser.add_argument(
        "--missing-share",
        type=float,
        default=MISSING_SHARE,
        help="share of NaN panel cells",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED, help="seed for model and draws"
    )
    parser.add_argument(
        "--max-bytes",
        type=_byte_count,
        default=None,
        help="skip a batch whose compiled memory exceeds this many bytes (e.g. 14e9)",
    )
    parser.add_argument(
        "--hourly-price",
        type=float,
        default=None,
        help="the rented instance's on-demand USD per hour, as billed; none on the Mac",
    )
    parser.add_argument(
        "--four-gpu-price",
        type=float,
        default=None,
        help=(
            "on-demand USD per hour of the whole 4-GPU instance of the same GPU type "
            "(4 times Lambda's per-GPU price), if Lambda rents one; none otherwise"
        ),
    )
    return parser


def _byte_count(text: str) -> int:
    return int(float(text))


def _parse_filters(filters: list[str]) -> dict[str, set[str]]:
    wanted = defaultdict(set)
    for item in filters:
        key, separator, value = item.partition("=")
        if not separator or key not in FILTER_KEYS:
            raise ValueError(
                f"a filter is KEY=VALUE with KEY in {FILTER_KEYS}: {item!r}"
            )
        wanted[key].add(value)
    return dict(wanted)


def _validated_grid_entry(entry: dict) -> dict:
    for key in DIMENSION_KEYS:
        value = entry.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"grid entry {entry} needs a positive integer {key!r}")
    return {key: entry.get(key) for key in FILTER_KEYS} | entry


def _device_memory_stats() -> dict:
    """The first device's memory statistics; empty where it reports none (a CPU)."""
    return jax.devices()[0].memory_stats() or {}


def _compiled_memory(
    ssm: LinearGaussianSSM, panel: np.ndarray, batch: int
) -> dict[str, int]:
    """XLA's memory analysis of the batched value and gradient, compiled from shapes."""
    value_and_grad = engine_probe.batched_value_and_grad()
    log_scales = jax.ShapeDtypeStruct((batch, 2), panel.dtype)
    ssm_shapes, panel_shape = jax.tree.map(
        lambda array: jax.ShapeDtypeStruct(array.shape, array.dtype), (ssm, panel)
    )
    lowered = value_and_grad.lower(log_scales, ssm_shapes, panel_shape)
    analysis = lowered.compile().memory_analysis()
    fields = {
        name: int(getattr(analysis, name))
        for name in dir(analysis)
        if name.endswith("_in_bytes") and not name.startswith("host_")
    }
    fields["total_bytes"] = (
        fields["argument_size_in_bytes"]
        + fields["output_size_in_bytes"]
        + fields["temp_size_in_bytes"]
        - fields["alias_size_in_bytes"]
    )
    return fields


def _write(record: dict, out: Path) -> None:
    out.write_text(json.dumps(record, indent=2) + "\n")


def _summary(entry: dict) -> str:
    timings = ", ".join(
        f"batch {batch['batch']}: "
        + (
            f"{batch['median_seconds']:.4g} s"
            if "median_seconds" in batch
            else "skipped"
        )
        for batch in entry["batches"]
    )
    name = f"{entry.get('layout')} stage {entry.get('stage')} {entry.get('wedge')}"
    dims = f"T={entry['steps']} n={entry['states']} p={entry['cells']}"
    return f"{name} {dims}: {timings}"


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_pilot_dimensions.py tests/test_pilot_seasonal.py tests/test_pilot_gpu_choice.py -q`
Expected: `70 passed`.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format src/ces_revisions/pilot/dimensions.py src/ces_revisions/pilot/seasonal.py src/ces_revisions/pilot/gpu_choice.py scripts/gpu_probe_grid.py tests/test_pilot_dimensions.py tests/test_pilot_seasonal.py tests/test_pilot_gpu_choice.py
uv run ruff check src/ces_revisions/pilot/dimensions.py src/ces_revisions/pilot/seasonal.py src/ces_revisions/pilot/gpu_choice.py scripts/gpu_probe_grid.py tests/test_pilot_dimensions.py tests/test_pilot_seasonal.py tests/test_pilot_gpu_choice.py
git add src/ces_revisions/pilot/dimensions.py src/ces_revisions/pilot/seasonal.py src/ces_revisions/pilot/gpu_choice.py scripts/gpu_probe_grid.py tests/test_pilot_dimensions.py tests/test_pilot_seasonal.py tests/test_pilot_gpu_choice.py
git commit -m "Add the seasonal and GPU decision rules and the probe grid"
```

### Task 9: The pilot CLI

**Files:**
- Create: `scripts/pilot_run.py`
- Test: `tests/test_pilot_run.py`, `tests/test_pilot_fit.py`

**Interfaces:**
- Consumes: everything from Tasks 2–8.
- Produces: `scripts/pilot_run.py` with the commands `prior-predictive`, `fit`, `sbc`, `sbc-summary`, `compare-seasonal`, and `choose-gpu`; its module-level `_configure_xla(argv)`, `ljung_box(errors, lags=12) -> dict`, `standardized_errors(params, panel, layout)`, `seasonal_candidates(entries, grid) -> dict[str, dict]`, `month(text) -> date`, and `main(argv) -> int`. `fit` writes a run directory (`inference_data.nc`, `diagnostics.json`, `provenance.json`, `april_2020.json`) under `data/runs/`. `compare-seasonal RUN_DIR... --grid PATH --out PATH` writes the comparison and the rule's verdict; it reads each run directory's `april_2020.json`. `choose-gpu GRID_RECORD... --layout NAME --out PATH [--steps-per-iteration N]` writes the GPU rule's result and prints its overall status and each Stage 7–9 model's chosen host.

The CLI sets `XLA_FLAGS` before its first JAX import: XLA's deterministic GPU flag on an NVIDIA host, and four host devices on a CPU host for `fit` and `prior-predictive`. SBC workers set their own single device.

- [ ] **Step 1: Write the failing CLI tests**

Create `tests/test_pilot_run.py`:

```python
"""The pilot CLI's pieces that need no built artifact: XLA flags, Ljung-Box, arguments."""

import json
import os
import subprocess
import sys
from datetime import date

import numpy as np
import pilot_run
import pytest
from scipy.stats import chi2

from ces_revisions.devices import DETERMINISTIC_GPU_FLAG
from ces_revisions.pilot.dimensions import STAGE_CONFIGURATIONS, stage_dimensions
from ces_revisions.vintages import raw as vintage_raw

SEED = sum(map(ord, "ces-revisions-stage-6-pilot-run"))
COUNT_FLAG = "--xla_force_host_platform_device_count"


@pytest.mark.parametrize(
    ("argv", "nvidia", "before", "after"),
    [
        (["fit"], False, "", f"{COUNT_FLAG}=4"),
        (["prior-predictive", "--out", "x"], False, "", f"{COUNT_FLAG}=4"),
        (["sbc", "--out-dir", "x"], False, "", ""),
        (["compare-seasonal", "fit"], False, "", ""),
        (["fit"], False, f"{COUNT_FLAG}=1", f"{COUNT_FLAG}=1"),
        (["fit"], True, "", DETERMINISTIC_GPU_FLAG),
        (["sbc"], True, DETERMINISTIC_GPU_FLAG, DETERMINISTIC_GPU_FLAG),
    ],
)
def test_xla_flags_follow_the_command_and_host(
    monkeypatch, argv, nvidia, before, after
):
    monkeypatch.setenv("XLA_FLAGS", before)
    monkeypatch.setattr(pilot_run, "has_nvidia_device", lambda: nvidia)
    pilot_run._configure_xla(argv)
    assert os.environ["XLA_FLAGS"] == after


def brute_force_ljung_box(series: np.ndarray, lags: int) -> float:
    n = series.size
    centered = series - series.mean()
    total = 0.0
    for h in range(1, lags + 1):
        r = sum(centered[t] * centered[t - h] for t in range(h, n)) / np.sum(
            centered**2
        )
        total += r**2 / (n - h)
    return n * (n + 2) * total


def test_ljung_box_matches_its_formula_without_gaps():
    series = np.random.default_rng(SEED).standard_normal(300)
    result = pilot_run.ljung_box(series)
    q = brute_force_ljung_box(series, 12)
    assert result["q"] == pytest.approx(q, rel=1e-12)
    assert result["p_value"] == pytest.approx(chi2.sf(q, 12), rel=1e-12)
    assert result["num_observed"] == 300


def gapped_series(rng: np.random.Generator, phi: float) -> np.ndarray:
    """500 months of AR(1) noise with coefficient phi and 40 random gaps."""
    shocks = rng.standard_normal(500)
    series = np.zeros(500)
    for t in range(500):
        series[t] = (phi * series[t - 1] if t else 0.0) + shocks[t]
    series[rng.choice(500, size=40, replace=False)] = np.nan
    return series


def test_ljung_box_with_gaps_has_its_size_and_detects_autocorrelation():
    rng = np.random.default_rng(SEED)
    white = [pilot_run.ljung_box(gapped_series(rng, 0.0)) for _ in range(200)]
    assert all(result["num_observed"] == 460 for result in white)
    rejected = np.mean([result["p_value"] < 0.05 for result in white])
    assert 0.02 <= rejected <= 0.09
    autoregressive = [pilot_run.ljung_box(gapped_series(rng, 0.5)) for _ in range(20)]
    assert all(result["p_value"] < 1e-6 for result in autoregressive)


def test_a_gap_contributes_to_no_lag():
    """Filling every gap with the observed mean leaves each autocorrelation unchanged."""
    series = gapped_series(np.random.default_rng(SEED), 0.3)
    filled = np.where(np.isnan(series), np.nanmean(series), series)
    centered = filled - filled.mean()
    expected = [
        np.sum(centered[h:] * centered[:-h]) / np.sum(centered**2) for h in range(1, 13)
    ]
    np.testing.assert_allclose(
        pilot_run.ljung_box(series)["autocorrelation"], expected, rtol=1e-12
    )


def test_month_arguments_are_first_days():
    assert pilot_run.month("2003-05") == date(2003, 5, 1)


def test_the_cli_starts_in_a_fresh_process():
    result = subprocess.run(
        [sys.executable, "scripts/pilot_run.py", "--help"],
        cwd=vintage_raw.ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    for command in (
        "prior-predictive",
        "fit",
        "sbc",
        "sbc-summary",
        "compare-seasonal",
        "choose-gpu",
    ):
        assert command in result.stdout


def grid_entry(stage, wedge, steps, batches):
    dims = stage_dimensions("dummy", stage, wedge, steps)
    return {**dims, "batches": [{"batch": b, **timing} for b, timing in batches]}


def test_choose_gpu_writes_the_rule_for_the_given_records(tmp_path):
    no_fit = {"error": "skipped: compiled memory exceeds 40e9 bytes"}
    entries = []
    for stage, wedge in STAGE_CONFIGURATIONS:
        fit = [(1, {"median_seconds": 0.2}), (4, no_fit)]
        entries.append(grid_entry(stage, wedge, 572, fit))
        entries.append(grid_entry(stage, wedge, 280, [(1, {"median_seconds": 0.05})]))
    grid = {
        "label": "a100",
        "hourly_price_usd": 1.99,
        "four_gpu_hourly_price_usd": 7.96,
        "entries": entries,
    }
    path = tmp_path / "a100.json"
    path.write_text(json.dumps(grid))
    out = tmp_path / "choice.json"
    parser_arguments = ["choose-gpu", str(path), "--layout", "trig", "--out", str(out)]
    assert pilot_run.main(parser_arguments) == 0
    written = json.loads(out.read_text())
    assert written["layout"] == "trig"  # reads the dummy's entries: same dimensions
    assert written["records"] == [str(path)]
    assert written["status"] == "chosen"
    assert written["steps_per_iteration"] == 127
    assert len(written["stages"]) == len(STAGE_CONFIGURATIONS)
    for result in written["stages"]:
        assert result["choice"]["host"] == "a100"
        assert result["choice"]["gpus"] == 4  # batch 4 did not fit: one GPU per chain
    measured = [*parser_arguments, "--steps-per-iteration", "31"]
    assert pilot_run.main(measured) == 0
    assert json.loads(out.read_text())["steps_per_iteration"] == 31


def test_seasonal_candidates_carry_what_the_rule_reads():
    tests = {
        "F": {"q": 85.0, "p_value": 1e-9},
        "S": {"q": 50.0, "p_value": 1e-5},
        "T": {"q": 9.0, "p_value": 0.7},
    }
    entry = {
        "layout": "trig",
        "dimensions": {"stage9_low": 247},
        "meets_req18": True,
        "april_2020_passed": False,
        "ljung_box": tests,
    }
    cost = stage_dimensions("dummy", 9, "low", 572)
    grid = {"entries": [cost | {"batches": [{"batch": 1, "median_seconds": 2.99}]}]}
    assert pilot_run.seasonal_candidates([entry], grid) == {
        "trig": {
            "states": 247,
            "meets_req18": True,
            "april_2020_passed": False,
            "ljung_box_q": {"F": 85.0, "S": 50.0, "T": 9.0},
            "ljung_box_rejections": 2,
            "seconds": 2.99,
        }
    }
```

Create `tests/test_pilot_fit.py`:

```python
"""The pilot model on the real aggregate panel: structure, prior predictive, determinism.

The panel comes from the session's cached Stage 3, 4, and 5 builds, so these tests need
no gitignored artifact. The fits are short and run at a capped tree depth: they check that
a seed repeats its draws, not Req 18, which the full-length run in scripts/pilot_run.py
checks.
"""

import json
import time
from datetime import date
from functools import cache

import annual_data
import jax
import numpy as np
import operator_data
import pytest
import vintage_data
from numpyro import handlers

from ces_revisions.kalman import kalman_filter
from ces_revisions.pilot.assembly import build_aggregate_panel
from ces_revisions.pilot.dimensions import SEASONAL_DIMS, pilot_dimensions
from ces_revisions.pilot.layout import LAYOUTS
from ces_revisions.pilot.model import (
    SAMPLED_SITES,
    pilot_model,
    pilot_ssm,
    sample_prior,
    step_log_likelihood,
)
from ces_revisions.pilot.panel import ANCHOR_COLUMN, ANCHOR_YEARS, AggregatePanel
from ces_revisions.pilot.simulate import monthly_constraint, prior_predictive
from ces_revisions.runs import provenance
from ces_revisions.runs.fit import RunSettings, run_nuts
from ces_revisions.vintages import raw as vintage_raw

SEED = sum(map(ord, "ces-revisions-stage-6-pilot-fit-test"))
PRIOR_PREDICTIVE_SEED = sum(map(ord, "ces-revisions-stage-6-prior-predictive"))
NUM_PRIOR_PREDICTIVE_DRAWS = 1000
# Shorter than Req 18's run; depth 5 bounds each draw at 31 leapfrog steps.
SHORT = {"num_chains": 2, "num_warmup": 100, "num_samples": 100, "max_tree_depth": 5}


@cache
def real_panel() -> AggregatePanel:
    return build_aggregate_panel(
        stage_changes=vintage_data.stage_changes(),
        stage_labels=vintage_data.labels(),
        table_changes=vintage_data.table_changes(),
        release_index=vintage_data.index(),
        benchmark_fixtures=operator_data.artifact("benchmark_fixtures"),
        publication_calendar=annual_data.artifact("publication_calendar"),
    )


def prior_medians(layout_name: str) -> dict[str, float]:
    draws = sample_prior(jax.random.PRNGKey(SEED), 2000, LAYOUTS[layout_name])
    return {name: float(np.median(values)) for name, values in draws.items()}


def test_seasonal_dimensions_agree_with_the_layouts():
    for name, layout in LAYOUTS.items():
        assert SEASONAL_DIMS[name] == layout.seasonal_dim
        assert pilot_dimensions(name)["states"] == layout.state_dim


def test_the_likelihood_sees_irregular_anchor_rows_and_missing_cells():
    panel = real_panel()
    anchored = ~np.isnan(panel.y[:, ANCHOR_COLUMN])
    anchor_months = [
        month
        for month, kept in zip(panel.reference_months, anchored, strict=True)
        if kept
    ]
    assert anchor_months == [date(year, 3, 1) for year in ANCHOR_YEARS]
    assert len(anchor_months) == 23
    closing = panel.y[:, :ANCHOR_COLUMN]
    assert np.isnan(closing).sum(axis=0).tolist() == [1, 49, 50]
    # Every month has some observed cell, so every step adds to the likelihood.
    assert panel.observed_any().all()
    steps = np.asarray(
        step_log_likelihood(prior_medians("dummy"), panel, LAYOUTS["dummy"])
    )
    assert steps.shape == (panel.num_steps,)
    assert np.isfinite(steps).all()
    assert (steps != 0.0).all()


@pytest.mark.parametrize("layout_name", sorted(LAYOUTS))
def test_the_model_samples_only_the_scalar_sites_and_no_state(layout_name):
    panel, layout = real_panel(), LAYOUTS[layout_name]
    trace = handlers.trace(handlers.seed(pilot_model, rng_seed=SEED)).get_trace(
        panel, layout
    )
    latent = [
        name
        for name, site in trace.items()
        if site["type"] == "sample" and not site["is_observed"]
    ]
    assert tuple(latent) == SAMPLED_SITES
    assert all(np.shape(trace[name]["value"]) == () for name in latent)
    others = {name: site for name, site in trace.items() if name not in latent}
    assert list(others) == ["panel"]
    params = {name: trace[name]["value"] for name in latent}
    expected = kalman_filter(pilot_ssm(params, panel, layout), panel.y).log_likelihood
    factor = others["panel"]["fn"].log_prob(others["panel"]["value"])
    np.testing.assert_allclose(factor, expected, rtol=1e-12)


def test_the_prior_predictive_meets_the_monthly_constraint_on_the_real_panel():
    started = time.perf_counter()
    result = prior_predictive(
        real_panel(),
        LAYOUTS["dummy"],
        NUM_PRIOR_PREDICTIVE_DRAWS,
        PRIOR_PREDICTIVE_SEED,
    )
    seconds = time.perf_counter() - started
    constraint = monthly_constraint(result)
    assert constraint["passed"], constraint
    assert seconds < 60.0


@pytest.fixture
def provenance_root(tmp_path, monkeypatch):
    """The source manifests and a stand-in operator record; the checkout's git state."""
    root = tmp_path / "root"
    for relative in provenance.SOURCE_MANIFESTS.values():
        (root / relative).parent.mkdir(parents=True)
        (root / relative).write_bytes((vintage_raw.ROOT / relative).read_bytes())
    record = root / provenance.OPERATOR_TEST_RECORD
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"schema_version": 1}))
    state = provenance.git_state(vintage_raw.ROOT)
    monkeypatch.setattr(provenance, "git_state", lambda root: state)
    return root


@pytest.mark.slow
def test_a_seed_repeats_its_draws_on_the_real_panel(provenance_root):
    panel, layout = real_panel(), LAYOUTS["dummy"]
    prior_draws = {
        name: np.asarray(values)
        for name, values in sample_prior(jax.random.PRNGKey(SEED), 2000, layout).items()
    }

    def short_fit():
        return run_nuts(
            pilot_model,
            (panel, layout),
            settings=RunSettings(seed=SEED, **SHORT),
            prior_draws=prior_draws,
            step_fn=lambda params: step_log_likelihood(params, panel, layout),
            reference_months=panel.reference_months,
            benchmark_year=panel.benchmark_year,
            observed_any=panel.observed_any(),
            model_record={"name": "stage6-pilot", "layout": layout.name},
            arrays={"y": panel.y},
            root=provenance_root,
        )

    first, second = short_fit(), short_fit()
    for name in SAMPLED_SITES:
        a, b = (np.asarray(fit.idata["posterior"][name]) for fit in (first, second))
        assert a.shape == (SHORT["num_chains"], SHORT["num_samples"])
        assert np.array_equal(a, b), name
    for name in ("panel_month", "panel_benchmark_year"):
        assert np.array_equal(
            first.idata["log_likelihood"][name], second.idata["log_likelihood"][name]
        )
    assert first.provenance["init"] == {
        **second.provenance["init"],
        "readback_seconds": first.provenance["init"]["readback_seconds"],
    }
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_pilot_run.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'pilot_run'`.

- [ ] **Step 3: Write the CLI**

Create `scripts/pilot_run.py`:

```python
"""Run the Stage 6 aggregate pilot: prior predictive, fit, SBC, and the two decisions.

Every command assembles the pilot panel from the built Stage 3, 4, and 5 artifacts
(``ces_revisions.pilot.assembly.read_artifacts``). ``prior-predictive`` applies Req 15's
monthly constraint to complete panels simulated from the prior; ``fit`` samples the pilot
with four dispersed NUTS chains and writes a run directory with the April 2020 check;
``sbc`` and ``sbc-summary`` run and summarize simulation-based calibration, one JSON file
per replicate so a stopped run resumes; ``compare-seasonal`` compares fitted runs of the
seasonal layouts by blocked PSIS-LOO, residual autocorrelation, and measured cost, and
applies the pre-registered layout rule (``ces_revisions.pilot.seasonal``); ``choose-gpu``
applies the GPU rule (``ces_revisions.pilot.gpu_choice``) to probe-grid records.

XLA reads its flags when JAX first touches a device, so they are set here before any JAX
operation: XLA's deterministic GPU operations on an NVIDIA host, and on a CPU host four
devices for the commands that run four chains or vectorize prior draws. SBC workers set
their own single device.
"""

import os
import sys

from ces_revisions.devices import DETERMINISTIC_GPU_FLAG, has_nvidia_device

FOUR_DEVICE_COMMANDS = ("fit", "prior-predictive")
HOST_DEVICE_FLAG = "--xla_force_host_platform_device_count"


def _configure_xla(argv: list[str]) -> None:
    """Append this command's XLA flags to XLA_FLAGS, keeping any already set.

    SBC's spawned workers re-run this module's top level with the parent's arguments,
    so it only ever appends, and leaves a device count already present alone.
    """
    flags = os.environ.get("XLA_FLAGS", "").split()
    if has_nvidia_device():
        if DETERMINISTIC_GPU_FLAG not in flags:
            flags.append(DETERMINISTIC_GPU_FLAG)
    elif (
        argv
        and argv[0] in FOUR_DEVICE_COMMANDS
        and not any(flag.startswith(HOST_DEVICE_FLAG) for flag in flags)
    ):
        flags.append(f"{HOST_DEVICE_FLAG}=4")
    os.environ["XLA_FLAGS"] = " ".join(flags)


_configure_xla(sys.argv[1:])

import numpyro

numpyro.enable_x64()

import argparse
import json
import time
from datetime import date
from pathlib import Path

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import chi2

from ces_revisions.kalman import kalman_filter
from ces_revisions.pilot.assembly import build_aggregate_panel, read_artifacts
from ces_revisions.pilot.checks import april_2020
from ces_revisions.pilot.dimensions import (
    pilot_dimensions,
    stage_dimensions,
)
from ces_revisions.pilot.gpu_choice import STEPS_PER_ITERATION, choose_gpu
from ces_revisions.pilot.layout import LAYOUTS, SEASONAL_PRIOR_SCALE, Layout
from ces_revisions.pilot.model import (
    INITIAL_MEAN,
    INITIAL_SD,
    SAMPLED_SITES,
    TARGET_ACCEPT_PROB,
    pilot_model,
    pilot_ssm,
    sample_prior,
    step_log_likelihood,
)
from ces_revisions.pilot.panel import CLOSING_STAGES, AggregatePanel
from ces_revisions.pilot.sbc import FIRST_MONTH, LAST_MONTH, run_sbc, write_summary
from ces_revisions.pilot.seasonal import (
    REFERENCE_LAYOUT,
    choose_layout,
    grid_seconds,
    ljung_box_rejections,
)
from ces_revisions.pilot.simulate import (
    monthly_constraint,
    prior_predictive,
)
from ces_revisions.runs.fit import (
    DIAGNOSTICS_FILE,
    INFERENCE_DATA_FILE,
    PROVENANCE_FILE,
    RUNS_DIR,
    RunSettings,
    new_run_dir,
    run_nuts,
    write_run,
)
from ces_revisions.runs.inference_data import read_inference_data
from ces_revisions.runs.provenance import array_sha256

PRIOR_PREDICTIVE_SEED = sum(map(ord, "ces-revisions-stage-6-prior-predictive"))
FIT_SEED = sum(map(ord, "ces-revisions-stage-6-pilot-fit"))
# The prior draws that place the dispersed starts: a property of the prior, not the run.
INIT_DRAWS_SEED = sum(map(ord, "ces-revisions-stage-6-init-prior-draws"))
NUM_INIT_DRAWS = 4000
SBC_SEED = sum(map(ord, "ces-revisions-stage-6-sbc"))
APRIL_2020_FILE = "april_2020.json"
MODEL_NAME = "stage6-pilot"
# The seasonal comparison's residual check: Ljung-Box over a year of lags.
LJUNG_BOX_LAGS = 12
LOO_BLOCKS = ("panel_month", "panel_benchmark_year")
RULE_BLOCK = "panel_benchmark_year"  # the seasonal rule's close-to-independent blocks


def load_panel() -> AggregatePanel:
    return build_aggregate_panel(**read_artifacts())


def panel_arrays(panel: AggregatePanel) -> dict[str, np.ndarray]:
    """The arrays the model reads from the panel, hashed into each run's provenance."""
    return {
        "y": panel.y,
        "anchor_offset": panel.anchor_offset,
        "pre_2003": panel.pre_2003,
        "crisis": panel.crisis,
        "benchmark_year": panel.benchmark_year,
    }


def model_record(layout_name: str) -> dict:
    layout = LAYOUTS[layout_name]
    return {
        "name": MODEL_NAME,
        "layout": layout.name,
        "harmonics": layout.harmonics,
        "state_names": list(layout.state_names),
        "sampled_sites": list(SAMPLED_SITES),
        "seasonal_prior_scale": SEASONAL_PRIOR_SCALE[layout.name],
        "initial_mean": {name: INITIAL_MEAN[name] for name in layout.state_names},
        "initial_sd": {name: INITIAL_SD[name] for name in layout.state_names},
        "dimensions": pilot_dimensions(layout.name),
    }


def write_json(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False)
    path.write_text(text + "\n")


def run_prior_predictive(arguments: argparse.Namespace) -> int:
    layout = LAYOUTS[arguments.layout]
    panel = load_panel()
    started = time.perf_counter()
    result = prior_predictive(
        panel, layout, arguments.draws, arguments.seed, doubled=arguments.doubled
    )
    seconds = time.perf_counter() - started
    constraint = monthly_constraint(result)
    statistics = result["statistics"]
    record = {
        "layout": layout.name,
        "seed": arguments.seed,
        "doubled": arguments.doubled,
        "num_draws": arguments.draws,
        "seconds": seconds,
        "monthly_constraint": constraint,
        "median_abs_quantiles": {
            str(q): float(np.quantile(statistics["median_abs"], q))
            for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "statistics": {name: values.tolist() for name, values in statistics.items()},
    }
    write_json(record, arguments.out)
    print(json.dumps({"seconds": seconds, **constraint}))
    return 0


def run_fit(arguments: argparse.Namespace) -> int:
    layout = LAYOUTS[arguments.layout]
    panel = load_panel()
    prior_draws = {
        name: np.asarray(values)
        for name, values in sample_prior(
            jax.random.PRNGKey(INIT_DRAWS_SEED), NUM_INIT_DRAWS, layout
        ).items()
    }
    settings = RunSettings(
        seed=arguments.seed,
        num_chains=arguments.chains,
        num_warmup=arguments.warmup,
        num_samples=arguments.samples,
        max_tree_depth=arguments.max_tree_depth,
        target_accept_prob=arguments.target_accept,
    )
    result = run_nuts(
        pilot_model,
        (panel, layout),
        settings=settings,
        prior_draws=prior_draws,
        step_fn=lambda params: step_log_likelihood(params, panel, layout),
        reference_months=panel.reference_months,
        benchmark_year=panel.benchmark_year,
        observed_any=panel.observed_any(),
        model_record={
            **model_record(layout.name),
            "init_draws": {"seed": INIT_DRAWS_SEED, "num_draws": NUM_INIT_DRAWS},
        },
        arrays=panel_arrays(panel),
    )
    # The draws are on disk before the smoother runs, so a failure there loses nothing.
    run_dir = write_run(
        result, new_run_dir(arguments.label, arguments.seed, arguments.runs_dir)
    )
    posterior = result.idata["posterior"]
    samples = {name: np.asarray(posterior[name]) for name in SAMPLED_SITES}
    check = april_2020(samples, panel, layout, seed=arguments.seed)
    write_json(check, run_dir / APRIL_2020_FILE)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "sampling_seconds": result.sampling_seconds,
                "meets_req18": result.diagnostics["meets_req18"],
                "worst": result.diagnostics["worst"],
                "april_2020_ratio": check["ratio"],
                "april_2020_passed": check["passed"],
            }
        )
    )
    return 0


def run_sbc_command(arguments: argparse.Namespace) -> int:
    panel = load_panel().window(arguments.first, arguments.last)
    summary = run_sbc(
        arguments.out_dir,
        layout_name=arguments.layout,
        panel=panel,
        start=arguments.start,
        stop=arguments.replicates if arguments.stop is None else arguments.stop,
        workers=arguments.workers,
        base_seed=arguments.seed,
        num_warmup=arguments.warmup,
        num_samples=arguments.samples,
        num_kept=arguments.keep,
        target_accept_prob=arguments.target_accept,
    )
    print(json.dumps(brief_summary(summary)))
    return 0


def run_sbc_summary(arguments: argparse.Namespace) -> int:
    print(json.dumps(brief_summary(write_summary(arguments.out_dir))))
    return 0


def brief_summary(summary: dict) -> dict:
    keys = (
        "num_replicates",
        "passed",
        "ecdf_failures",
        "recovery_checks",
        "divergent_replicates",
        "replicate_seconds_median",
        "failures",
        "skipped",
    )
    return {key: summary[key] for key in keys if key in summary}


def standardized_errors(
    params: dict[str, float], panel: AggregatePanel, layout: Layout
) -> np.ndarray:
    """(T, 3) one-step prediction errors of the closing cells over their one-step SDs.

    NaN where the cell is not observed.
    """
    filtered = kalman_filter(pilot_ssm(params, panel, layout), jnp.asarray(panel.y))
    closing = len(CLOSING_STAGES)
    mean = np.asarray(filtered.one_step_mean)[:, :closing]
    variance = np.diagonal(np.asarray(filtered.one_step_cov), axis1=1, axis2=2)
    return (panel.y[:, :closing] - mean) / np.sqrt(variance[:, :closing])


def ljung_box(errors: np.ndarray, lags: int = LJUNG_BOX_LAGS) -> dict:
    """Ljung-Box Q over lags 1..lags of a monthly series with NaN gaps.

    The lag-h autocorrelation sums over every pair of observed cells h months apart,
    over the observed cells' sum of squares about their mean; n counts observed cells.
    """
    observed = ~np.isnan(errors)
    count = int(observed.sum())
    centered = np.where(observed, errors - errors[observed].mean(), 0.0)
    lag = np.arange(1, lags + 1)
    autocorrelation = np.array(
        [np.sum(centered[h:] * centered[:-h]) for h in lag]
    ) / np.sum(centered**2)
    q = count * (count + 2) * np.sum(autocorrelation**2 / (count - lag))
    return {
        "q": float(q),
        "p_value": float(chi2.sf(q, lags)),
        "lags": lags,
        "num_observed": count,
        "autocorrelation": autocorrelation.tolist(),
    }


def read_run(run_dir: Path) -> dict:
    return {
        "idata": read_inference_data(run_dir / INFERENCE_DATA_FILE),
        "diagnostics": json.loads((run_dir / DIAGNOSTICS_FILE).read_text()),
        "provenance": json.loads((run_dir / PROVENANCE_FILE).read_text()),
    }


def loo_block(idata, block: str) -> tuple[dict, np.ndarray]:
    """PSIS-LOO over one log-likelihood block: its summary and the pointwise elpd.

    Blocks come in the run's coordinate order, the same for every run of one panel.
    """
    loo = az.loo(idata, var_name=block, pointwise=True)
    pareto_k = np.asarray(loo.pareto_k)
    return {
        "elpd": float(loo.elpd),
        "se": float(loo.se),
        "p": float(loo.p),
        "num_blocks": int(loo.n_data_points),
        "pareto_k_max": float(pareto_k.max()),
        "pareto_k_good": float(loo.good_k),
        "pareto_k_above_good": int(np.sum(pareto_k > loo.good_k)),
        "pareto_k": pareto_k.tolist(),
    }, np.asarray(loo.elpd_i)


def seasonal_candidates(entries: list[dict], grid: dict) -> dict[str, dict]:
    """What the seasonal rule reads about each fitted layout, keyed by layout."""
    return {
        entry["layout"]: {
            "states": entry["dimensions"]["stage9_low"],
            "meets_req18": entry["meets_req18"],
            "april_2020_passed": entry["april_2020_passed"],
            "ljung_box_q": {
                stage: test["q"] for stage, test in entry["ljung_box"].items()
            },
            "ljung_box_rejections": ljung_box_rejections(entry["ljung_box"]),
            "seconds": grid_seconds(grid, entry["layout"]),
        }
        for entry in entries
    }


def run_compare_seasonal(arguments: argparse.Namespace) -> int:
    panel = load_panel()
    data_hash = array_sha256(panel_arrays(panel))
    grid = json.loads(arguments.grid.read_text())
    entries, pointwise = [], []
    for run_dir in arguments.run_dirs:
        run = read_run(run_dir)
        provenance = run["provenance"]
        if provenance["arrays"]["sha256"] != data_hash:
            raise ValueError(f"{run_dir} was fit to other data than the current panel")
        layout = LAYOUTS[provenance["model"]["layout"]]
        blocks, elpd = {}, {}
        for block in LOO_BLOCKS:
            blocks[block], elpd[block] = loo_block(run["idata"], block)
        posterior = run["idata"]["posterior"]
        medians = {name: float(np.median(posterior[name])) for name in SAMPLED_SITES}
        errors = standardized_errors(medians, panel, layout)
        diagnostics = run["diagnostics"]
        seconds = provenance["sampling"]["timings"]["sampling_seconds"]
        ess_bulk_min = diagnostics["worst"]["ess_bulk_min"]
        entries.append(
            {
                "run_dir": str(run_dir),
                "layout": layout.name,
                "loo": blocks,
                "ljung_box": {
                    stage: ljung_box(errors[:, k])
                    for k, stage in enumerate(CLOSING_STAGES)
                },
                "posterior_median": medians,
                "sampling_seconds": seconds,
                "ess_bulk_min": ess_bulk_min,
                "ess_bulk_min_per_second": None
                if ess_bulk_min is None
                else ess_bulk_min / seconds,
                "meets_req18": diagnostics["meets_req18"],
                "april_2020_passed": json.loads(
                    (run_dir / APRIL_2020_FILE).read_text()
                )["passed"],
                "dimensions": {
                    "pilot": pilot_dimensions(layout.name)["states"],
                    "stage7": stage_dimensions(layout.name, 7)["states"],
                    "stage9_low": stage_dimensions(layout.name, 9, "low")["states"],
                    "stage9_high": stage_dimensions(layout.name, 9, "high")["states"],
                },
            }
        )
        pointwise.append(elpd[RULE_BLOCK])
    layouts = [entry["layout"] for entry in entries]
    if REFERENCE_LAYOUT not in layouts or len(set(layouts)) != len(layouts):
        raise ValueError(
            "compare-seasonal needs one run per layout, the dummy included"
        )
    candidates = seasonal_candidates(entries, grid)
    rule = choose_layout(
        candidates,
        dict(zip(layouts, pointwise, strict=True)),
        {
            entry["layout"]: np.asarray(entry["loo"][RULE_BLOCK]["pareto_k"])
            for entry in entries
        },
    )
    write_json(
        {
            "panel_sha256": data_hash,
            "grid": {"path": str(arguments.grid), "label": grid["label"]},
            "runs": entries,
            "candidates": candidates,
            "rule": rule,
        },
        arguments.out,
    )
    print(json.dumps({"layout": rule["layout"], "reliable": rule["reliable"]}))
    return 0


def run_choose_gpu(arguments: argparse.Namespace) -> int:
    records = [json.loads(path.read_text()) for path in arguments.records]
    result = choose_gpu(records, arguments.layout, arguments.steps_per_iteration)
    write_json(
        {
            "layout": arguments.layout,
            "records": [str(path) for path in arguments.records],
            **result,
        },
        arguments.out,
    )
    choices = {
        f"{stage['stage']} {stage['wedge']}": stage["choice"]
        and stage["choice"]["host"]
        for stage in result["stages"]
    }
    print(json.dumps({"status": result["status"], "choices": choices}))
    return 0


def month(text: str) -> date:
    """A YYYY-MM argument as the first day of that month."""
    return date.fromisoformat(f"{text}-01")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    prior = commands.add_parser("prior-predictive", help="Req 15's monthly constraint")
    prior.add_argument("--layout", choices=sorted(LAYOUTS), default="dummy")
    prior.add_argument("--draws", type=int, default=2000)
    prior.add_argument("--seed", type=int, default=PRIOR_PREDICTIVE_SEED)
    prior.add_argument("--doubled", action="store_true")
    prior.add_argument("--out", type=Path, required=True)
    prior.set_defaults(handler=run_prior_predictive)

    fit = commands.add_parser("fit", help="four dispersed NUTS chains and a run dir")
    fit.add_argument("--layout", choices=sorted(LAYOUTS), default="dummy")
    fit.add_argument("--seed", type=int, default=FIT_SEED)
    fit.add_argument("--chains", type=int, default=4)
    fit.add_argument("--warmup", type=int, default=1000)
    fit.add_argument("--samples", type=int, default=1000)
    fit.add_argument("--max-tree-depth", type=int, default=10)
    fit.add_argument("--target-accept", type=float, default=TARGET_ACCEPT_PROB)
    fit.add_argument("--label", default="stage6-pilot")
    fit.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    fit.set_defaults(handler=run_fit)

    sbc = commands.add_parser("sbc", help="simulation-based calibration replicates")
    sbc.add_argument("--layout", choices=sorted(LAYOUTS), default="dummy")
    sbc.add_argument("--replicates", type=int, default=100)
    sbc.add_argument("--start", type=int, default=0)
    sbc.add_argument("--stop", type=int, default=None)
    sbc.add_argument("--workers", type=int, default=6)
    sbc.add_argument("--warmup", type=int, default=500)
    sbc.add_argument("--samples", type=int, default=990)
    sbc.add_argument("--keep", type=int, default=99)
    sbc.add_argument("--target-accept", type=float, default=TARGET_ACCEPT_PROB)
    sbc.add_argument("--seed", type=int, default=SBC_SEED)
    sbc.add_argument("--first", type=month, default=FIRST_MONTH)
    sbc.add_argument("--last", type=month, default=LAST_MONTH)
    sbc.add_argument("--out-dir", type=Path, required=True)
    sbc.set_defaults(handler=run_sbc_command)

    summary = commands.add_parser("sbc-summary", help="summarize an SBC directory")
    summary.add_argument("--out-dir", type=Path, required=True)
    summary.set_defaults(handler=run_sbc_summary)

    compare = commands.add_parser("compare-seasonal", help="compare fitted layouts")
    compare.add_argument("run_dirs", type=Path, nargs="+", metavar="RUN_DIR")
    compare.add_argument("--out", type=Path, required=True)
    compare.add_argument(
        "--grid", type=Path, required=True, help="gpu_probe_grid.py record for cost"
    )
    compare.set_defaults(handler=run_compare_seasonal)

    gpu = commands.add_parser("choose-gpu", help="choose the GPU type and count")
    gpu.add_argument("records", type=Path, nargs="+", metavar="GRID_RECORD")
    gpu.add_argument("--layout", choices=sorted(LAYOUTS), required=True)
    gpu.add_argument("--out", type=Path, required=True)
    gpu.add_argument(
        "--steps-per-iteration",
        type=int,
        default=STEPS_PER_ITERATION,
        help="leapfrog steps per iteration of the slowest chain, once a stage measures it",
    )
    gpu.set_defaults(handler=run_choose_gpu)

    arguments = parser.parse_args(argv)
    return arguments.handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_pilot_run.py tests/test_pilot_fit.py -q -m "not slow"`
Expected: `20 passed, 1 deselected`.

Run: `uv run pytest tests/test_pilot_fit.py -q -m slow`
Expected: `1 passed`. It runs two short, depth-capped fits from one seed and checks that they repeat.

- [ ] **Step 5: Run the whole fast tier**

Run: `uv run pytest -m "not slow and not network" -q`
Expected: `625 passed, 21 deselected`, as in the spike.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format scripts/pilot_run.py tests/test_pilot_run.py tests/test_pilot_fit.py
uv run ruff check scripts/pilot_run.py tests/test_pilot_run.py tests/test_pilot_fit.py
git add scripts/pilot_run.py tests/test_pilot_run.py tests/test_pilot_fit.py
git commit -m "Add the Stage 6 pilot CLI"
```

### Task 10: The prior predictive and the pilot fit

**Files:**
- Create: `docs/runs/stage-6/prior-predictive.json`, `docs/runs/stage-6/prior-predictive-doubled.json`
- Create: `docs/runs/stage-6/pilot-fit/diagnostics.json`, `docs/runs/stage-6/pilot-fit/provenance.json`, `docs/runs/stage-6/pilot-fit/april_2020.json`

**Interfaces:**
- Consumes: `scripts/pilot_run.py` (Task 9) and the built Stage 3–5 artifacts.
- Produces: the committed evidence for three Stage 6 exit items: the prior predictive meets Req 15's monthly constraint before the fit, the pilot meets Req 18 with diagnostics and provenance in its run directory, and the April 2020 movement is absorbed by the latent state. The run directory under `data/runs/` keeps the pilot InferenceData (`inference_data.nc`, about 21 MB), which Task 12 reads.

Req 15 requires the prior predictive to run before the fit, so commit Steps 2 and 3 before Step 5.

- [ ] **Step 1: Build the artifacts and check that the tree is clean**

Commit this plan file and any checkbox ticks in it first: `run_provenance` counts untracked and modified files.

```bash
uv run python scripts/vintage_sources.py build
uv run python scripts/annual_sources.py build
uv run python scripts/operator_artifacts.py build
git status --short
```

Expected: the builds finish, and `git status --short` prints nothing.

- [ ] **Step 2: Run the prior predictive**

```bash
mkdir -p docs/runs/stage-6 data/runs
uv run python scripts/pilot_run.py prior-predictive --out docs/runs/stage-6/prior-predictive.json
```

Expected: `"passed": true`, with `share_median_in_range` 0.9475 and `pooled_share_above_250` 0.02885. The seed is fixed, so the values repeat exactly. A failure stops the plan here: Req 15 says that failures adjust priors before any fit, and that is a decision for Lowell.

- [ ] **Step 3: Run the doubled-SD sensitivity**

```bash
uv run python scripts/pilot_run.py prior-predictive --doubled --out docs/runs/stage-6/prior-predictive-doubled.json
```

Expected: `"passed": false`, with `share_median_in_range` 0.6685 and `pooled_share_above_250` 0.1333. This is Req 15's sensitivity rerun, with every LogNormal log-scale SD doubled, the noise row's included. It is recorded for Stage 16's prior-sensitivity gate and does not stop the plan.

- [ ] **Step 4: Commit the prior predictive**

```bash
git add docs/runs/stage-6/prior-predictive.json docs/runs/stage-6/prior-predictive-doubled.json
git commit -m "Record the Stage 6 prior predictive"
git status --short
```

Expected: `git status --short` prints nothing, so the fit records `"dirty": false`.

- [ ] **Step 5: Fit the pilot**

Run in the background, about 30 minutes on the M4 Max:

```bash
uv run python scripts/pilot_run.py fit --layout dummy --label stage6-pilot > data/runs/stage6-pilot.log 2>&1
```

Expected: the last line of the log is a JSON summary with `"meets_req18": true` and `"april_2020_passed": true`. The April 2020 check passes only if every subsampled draw filters to finite moments. In the spike, the plan's model gave R-hat at most 1.0038, minimum bulk ESS 1694, 0 divergences, and an April 2020 change of −19,744 with total SD 122. The run directory is `data/runs/stage6-pilot-seed2995-<UTC timestamp>/`.

If `meets_req18` is false, or the April 2020 check fails, stop and report the summary. Do not change the target acceptance, the priors, or the check to make it pass.

- [ ] **Step 6: Commit the fit evidence**

```bash
RUN=$(ls -d data/runs/stage6-pilot-seed2995-* | tail -1)
grep '"dirty"' "$RUN/provenance.json"
mkdir -p docs/runs/stage-6/pilot-fit
cp "$RUN/diagnostics.json" "$RUN/provenance.json" "$RUN/april_2020.json" docs/runs/stage-6/pilot-fit/
git add docs/runs/stage-6/pilot-fit/
git commit -m "Record the Stage 6 pilot fit"
```

Expected: the `grep` prints `"dirty": false`.

### Task 11: Reduced-panel SBC

**Files:**
- Create: `docs/runs/stage-6/sbc-summary.json`

**Interfaces:**
- Consumes: `scripts/pilot_run.py sbc` and `sbc-summary` (Task 9).
- Produces: the committed evidence for Stage 6's SBC exit item: reduced-panel SBC recovers the news/noise allocation (Req 18 bullet 7, reduced half).

- [ ] **Step 1: Run the replicates**

Run in the background, about 4 hours with six workers:

```bash
uv run python scripts/pilot_run.py sbc --out-dir data/runs/stage6-sbc > data/runs/stage6-sbc.log 2>&1
```

The defaults are 100 replicates, January 1999 to December 2021, 500 warmup and 990 draws thinned to 99, and target acceptance 0.95. Each replicate writes `data/runs/stage6-sbc/replicate-NNNN.json`. If the run stops, the same command resumes it and skips finished replicates.

- [ ] **Step 2: Summarize**

```bash
uv run python scripts/pilot_run.py sbc-summary --out-dir data/runs/stage6-sbc
```

Expected: `"passed": true`, `"num_replicates": 100`, no ECDF failures, and both recovery checks passed. In the spike's window and settings the share contractions were 0.853 and 0.605 and the |z| > 3 shares were 0.01 each. A new window and target acceptance give new values; the rule's bars are contraction ≥ 0.25 and |z| > 3 in at most 5% of replicates.

If `passed` is false, stop and report `ecdf_failures` and `recovery_checks`. A failed SBC is a finding about the model, not a reason to change the rule.

- [ ] **Step 3: Commit the summary**

```bash
cp data/runs/stage6-sbc/summary.json docs/runs/stage-6/sbc-summary.json
git add docs/runs/stage-6/sbc-summary.json
git commit -m "Record the Stage 6 reduced-panel SBC"
```

### Task 12: The Mac grid and the seasonal comparison

**Files:**
- Create: `docs/decisions/cloud-gpu-probe/mac-grid.json`
- Create: `docs/runs/stage-6/seasonal-comparison.json`
- Create: `docs/runs/stage-6/seasonal-trig/diagnostics.json` and `april_2020.json`, and the same two files under `docs/runs/stage-6/seasonal-trig4/`

**Interfaces:**
- Consumes: `scripts/gpu_probe_grid.py` (Task 8), `scripts/pilot_run.py fit` and `compare-seasonal` (Task 9), and Task 10's dummy run directory.
- Produces: the Mac grid record that clause (b) of the seasonal rule reads, and the comparison whose `rule.layout` names the layout for Stages 7–17.

Run nothing else while Step 1 runs: it is a timing measurement.

- [ ] **Step 1: Time the engine on the Mac**

About 12 minutes:

```bash
uv run python scripts/gpu_probe_grid.py --label mac --out docs/decisions/cloud-gpu-probe/mac-grid.json --batch 1 4 --repeats 3 --max-bytes 20e9
```

Expected: 20 entries. Four batch-4 cells at $`T=572`$ are skipped on memory: Stage 9 high (37.4 GB and 33.1 GB compiled for dummy and trig4) and Stage 8 high (26.9 GB and 23.2 GB). Batch-1 medians near 1.35 s (dummy Stage 7), 2.13 s (dummy Stage 8 low), 3.91 s (dummy Stage 8 high), 2.99 s (dummy Stage 9 low), and 2.45 s (trig4 Stage 9 low). Host load changes these by up to about 40%.

Commit the record now, so that the next fits record a clean tree:

```bash
git add docs/decisions/cloud-gpu-probe/mac-grid.json
git commit -m "Record the Mac engine timing grid"
git status --short
```

Expected: `git status --short` prints nothing.

- [ ] **Step 2: Fit the two trigonometric layouts**

Run one after the other in the background, about 35 minutes each:

```bash
uv run python scripts/pilot_run.py fit --layout trig --label seasonal-trig > data/runs/seasonal-trig.log 2>&1
uv run python scripts/pilot_run.py fit --layout trig4 --label seasonal-trig4 > data/runs/seasonal-trig4.log 2>&1
```

Expected: both report `"meets_req18": true`. If a layout fails Req 18 or its April 2020 check, keep going: the rule needs both for an alternative to win, so that layout cannot be chosen.

- [ ] **Step 3: Apply the rule**

```bash
DUMMY=$(ls -d data/runs/stage6-pilot-seed2995-* | tail -1)
TRIG=$(ls -d data/runs/seasonal-trig-seed2995-* | tail -1)
TRIG4=$(ls -d data/runs/seasonal-trig4-seed2995-* | tail -1)
uv run python scripts/pilot_run.py compare-seasonal "$DUMMY" "$TRIG" "$TRIG4" --grid docs/decisions/cloud-gpu-probe/mac-grid.json --out docs/runs/stage-6/seasonal-comparison.json
```

Expected: `{"layout": "dummy", "reliable": true}`, as in the spike. The spike ran this comparison under the first model (trig +15.4 ± 9.4, trig4 −466 ± 70 on benchmark-year blocks). The narrowed D2 changes every layout's elpd a little, so expect the margins to move. Whatever the rule returns is the decision: Task 14 records it. If it returns `trig`, the GPU rule reads the dummy's grid entries, because `trig` has the dummy's dimensions.

- [ ] **Step 4: Commit the evidence**

```bash
TRIG=$(ls -d data/runs/seasonal-trig-seed2995-* | tail -1)
TRIG4=$(ls -d data/runs/seasonal-trig4-seed2995-* | tail -1)
mkdir -p docs/runs/stage-6/seasonal-trig docs/runs/stage-6/seasonal-trig4
cp "$TRIG/diagnostics.json" "$TRIG/april_2020.json" docs/runs/stage-6/seasonal-trig/
cp "$TRIG4/diagnostics.json" "$TRIG4/april_2020.json" docs/runs/stage-6/seasonal-trig4/
git add docs/runs/stage-6/seasonal-comparison.json docs/runs/stage-6/seasonal-trig/ docs/runs/stage-6/seasonal-trig4/
git commit -m "Record the seasonal-layout comparison"
```

### Task 13: The Lambda GPU measurement (owner-run)

**Files:**
- Create: `docs/decisions/cloud-gpu-probe/a10-grid.json`, `docs/decisions/cloud-gpu-probe/a100-grid.json`, `docs/decisions/cloud-gpu-probe/gh200-grid.json`

**Interfaces:**
- Consumes: `scripts/gpu_probe_grid.py` (Task 8), committed on the branch.
- Produces: one grid record per GPU type, each with the price Lowell paid, for Task 14's GPU rule.

**Owner gate.** Lowell runs this task. Only Lowell rents, pays for, and terminates the instances (D3). The executor's part is Step 1 and Step 10. **Executor: after Step 1, stop and wait for Lowell. Do not start Task 14 until Lowell hands back at least one priced record. If no GPU type could be measured, stop and report. If Task 14's GPU rule later returns "infeasible" while a D3 type has no record, Task 14 stops and asks Lowell whether to measure that type before anything is recorded.**

Lambda bills by the minute until an instance is terminated in its console or API. An OS shutdown does not stop billing, and termination deletes the disk, so copy the record off first. The cap is \$40 for all three instances. Each instance has a time box of 3 hours from launch: at the planning prices, three full boxes cost \$16.71.

| Label | Lambda instance to rent | Price seen at planning | 4-GPU instance of the same type (not rented) |
|---|---|---|---|
| `a10` | 1× A10 (24 GB) | \$1.29 per hour | none |
| `a100` | 1× A100 (40 GB, SXM or PCIe) | \$1.99 per hour | 4× A100 40 GB PCIe, \$7.96 per hour |
| `gh200` | 1× GH200 (96 GB, aarch64 host) | \$2.29 per hour | none |

The last column gives the GPU rule its one-GPU-per-chain configuration. Only its price is used: the rule assumes that each of its GPUs runs batch 1 at the measured 1-GPU speed, which is an approximation when the measured A100 is SXM and the 4× instance is PCIe.

- [ ] **Step 1: Bundle the branch (executor)**

```bash
git bundle create /tmp/ces-revisions.bundle HEAD
git bundle verify /tmp/ces-revisions.bundle
```

Expected: `/tmp/ces-revisions.bundle is okay`. The bundle carries the committed code, so the instance needs no GitHub credentials. Hand the path to Lowell.

- [ ] **Step 2: Launch one instance (Lowell)**

Before the launch, check the budget: the spend so far plus the console's hourly price times 3 hours must stay at or under \$40. Then, in the Lambda console, launch one on-demand instance of the next type in the table, with the default Lambda Stack image and Lowell's SSH key. Note its IP address, the hourly price shown, the launch time, and, if the console lists a 4-GPU instance of the same GPU type, that instance's whole hourly price: 4 times the per-GPU price the console shows (\$7.96 for the A100 at planning).

- [ ] **Step 3: Check the driver (Lowell)**

```bash
ssh ubuntu@<ip> nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```

Expected: driver 580 or later, the minimum for JAX's CUDA 13 wheels. If it is older, first upgrade Lambda Stack's own packages, then restart the instance from the Lambda console and check again:

```bash
ssh ubuntu@<ip> 'sudo apt-get update && sudo apt-get -y dist-upgrade'
```

If the driver is still older than 580, install Ubuntu's open-module driver, restart, and check once more. Use the open module: the GH200 supports only NVIDIA's open GPU kernel modules.

```bash
ssh ubuntu@<ip> 'sudo apt-get install -y nvidia-driver-580-server-open'
```

If `nvidia-smi` still does not report 580 or later, terminate the instance (Step 8) and record the type as not measured, with the reason.

- [ ] **Step 4: Install uv and the project (Lowell)**

```bash
scp /tmp/ces-revisions.bundle ubuntu@<ip>:
ssh ubuntu@<ip>
curl -LsSf https://astral.sh/uv/0.13.1/install.sh | sh
source "$HOME/.local/bin/env"
git clone ces-revisions.bundle ces-revisions
cd ces-revisions
uv sync --extra cuda
uv run python -c "import jax; print(jax.devices())"
```

Expected: `uv sync` installs `jax-cuda13-plugin` 0.11.1, and the last command prints one `CudaDevice`. On the GH200, uv installs the aarch64 wheels, which exist for every locked package.

- [ ] **Step 5: Run a preflight (Lowell)**

One small entry, about a minute, checks that the GPU path works before the full grid:

```bash
uv run python scripts/gpu_probe_grid.py --label preflight --out /tmp/preflight.json --grid-filter stage=7 layout=dummy --steps 280 --batch 1 --repeats 1
uv run python -c "import json; r = json.load(open('/tmp/preflight.json')); print(r['host']['backend'], r['entries'][0]['batches'][0].get('median_seconds'))"
```

Expected: `gpu` and a number of seconds. If the script refuses to run because JAX's backend is not the GPU, or no number comes back, stop: terminate the instance (Step 8) and record the type as not measured, with the reason.

- [ ] **Step 6: Run the grid (Lowell)**

Replace `<label>` and `<price>` from the table and the console. Add `--four-gpu-price <price4>` for any type whose 4-GPU instance price you noted in Step 2, as the whole instance's hourly price; at planning time that is only `a100`. Steps 6 and 7 run from `~/ces-revisions`. `nohup` keeps the run alive if the SSH session drops; after a reconnect, run `cd ~/ces-revisions && tail -f grid.log` to see the log again:

```bash
nohup uv run python scripts/gpu_probe_grid.py --label <label> --out <label>-grid.json --batch 1 4 16 64 --repeats 5 --hourly-price <price> > grid.log 2>&1 &
tail -f grid.log
```

The script caps every batch at the device's memory and records a batch that cannot fit as skipped; larger batches after it are skipped without compiling. It rewrites the record after every entry, so a stopped run keeps its finished entries. Expect 15–60 minutes for the 20 entries. Stop `tail` with Ctrl-C when the 20th entry's line appears; the run itself keeps going until then.

- [ ] **Step 7: Check and copy the record off (Lowell)**

```bash
cd ~/ces-revisions && uv run python -c "import json; r = json.load(open('<label>-grid.json')); print(r['host']['backend'], len(r['entries']), r['hourly_price_usd'], r['four_gpu_hourly_price_usd'])"
```

Expected: `gpu`, `20`, and the prices you passed. If fewer than 20 entries are there and the 3-hour box allows, run Step 6 again in full: it rewrites the record. If the record is still incomplete, do not copy it: record the type as not measured, with the reason. A complete record is copied from the Mac, in the worktree:

```bash
scp ubuntu@<ip>:ces-revisions/<label>-grid.json docs/decisions/cloud-gpu-probe/
```

- [ ] **Step 8: Terminate the instance (Lowell)**

In the Lambda console, terminate the instance no later than 3 hours after its launch, whatever state the run is in, and check that the instance list shows none running.

- [ ] **Step 9: Repeat (Lowell)**

Repeat Steps 2–8 for the other two GPU types. Then tell the executor which records were copied and which types were not measured, with the reasons. Task 14's decision record names every type that was not measured.

- [ ] **Step 10: Commit the records (executor)**

```bash
uv run python -c "import json, sys; [print(p, len(r['entries']), r['hourly_price_usd'], r.get('four_gpu_hourly_price_usd'), r['host']['backend'], r['host']['device_kind']) for p in sys.argv[1:] for r in [json.load(open(p))]]" docs/decisions/cloud-gpu-probe/*-grid.json
git add docs/decisions/cloud-gpu-probe/*-grid.json
git commit -m "Record the Lambda GPU grid measurements"
```

Expected: each Lambda record shows 20 entries, its price, the `gpu` backend, and its device kind. `mac-grid.json` is already committed and shows no price.

### Task 14: The seasonal-state decision record and the docs

> Deviation: re-scoped to the seasonal-state decision alone by roadmap Stage 2 (specs/ces-revision-drivers-roadmap.md:75). Step 1, the GPU table and gpu-choice.json moved to O6. Plan 13's Task 2 executed the rest.

**Files:**
- Create: `src/ces_revisions/pilot/decision_tables.py`
- Create: `docs/runs/stage-6/gpu-choice.json`
- Create: `docs/decisions/seasonal-state.md`
- Test: `tests/test_seasonal_state_decision.py`
- Modify: `CLAUDE.md`, `README.md`

**Interfaces:**
- Consumes: `docs/runs/stage-6/seasonal-comparison.json` (Task 12), the grid records (Tasks 12 and 13), `scripts/pilot_run.py choose-gpu` (Task 9), the Task 10 and 11 evidence, and, from Task 8, `matches_layout` (`dimensions`), `FIT_HOURS_MAX` and `SBC_DAYS_MAX` (`gpu_choice`), and `REFERENCE_LAYOUT` (`seasonal`).
- Produces: `decision_tables` (`decision_lines(comparison, gpu_choice)`, `seasonal_rows(comparison)`, `timing_rows(records, layout_name)`, `gpu_rows(gpu_choice)`, `grid_records(gpu_choice, root=vintage_raw.ROOT)`, each returning `list[str]`, and a `python -m` entry that prints them); `docs/decisions/seasonal-state.md`, which records the seasonal state, the engine's measured value-and-gradient time on each GPU type at the Stage 7–9 dimensions of the chosen layout, and the GPU type and count for Stages 7–17 (roadmap Stage 6, Produces).

- [ ] **Step 1: Apply the GPU rule**

With `<layout>` the `rule.layout` of `docs/runs/stage-6/seasonal-comparison.json`:

```bash
uv run python scripts/pilot_run.py choose-gpu docs/decisions/cloud-gpu-probe/*-grid.json --layout <layout> --out docs/runs/stage-6/gpu-choice.json
```

Expected: a JSON line with `"status": "chosen"` and one chosen host for each Stage 7–9 model (D5). The Mac record has no price, so the rule skips it. `trig` reads the dummy's grid entries, because the two layouts have the same dimensions.

- `"status": "unmeasured"` means a priced record lacks an entry or a batch the rule needs for some model, or no record is priced. Stop and report: the measurement is incomplete, and nothing may be recorded as a finding.
- `"status": "infeasible"` means that, for at least one model, every configuration of the records given was measured and none runs its fit within 48 hours and its SBC within 14 days. First check that `a10`, `a100`, and `gh200` each have a committed record. If any has none, stop and report to Lowell before recording anything: "infeasible" is a finding only when every D3 type was measured, or when Lowell confirms in chat that each missing type could not be measured. Then the decision record says so, and the roadmap's resume re-validates Stages 7–17 before it routes Stage 7 (roadmap Stage 6, Exit).

- [ ] **Step 2: Write the failing decision test**

Create `tests/test_seasonal_state_decision.py`:

```python
"""The seasonal-state decision record states exactly what its committed evidence says."""

import json

import pytest

from ces_revisions.pilot.decision_tables import (
    decision_lines,
    gpu_rows,
    grid_records,
    seasonal_rows,
    timing_rows,
)
from ces_revisions.vintages import raw as vintage_raw

DECISION = vintage_raw.ROOT / "docs" / "decisions" / "seasonal-state.md"
EVIDENCE = vintage_raw.ROOT / "docs" / "runs" / "stage-6"

COMPARISON = {
    "runs": [{"layout": "dummy"}, {"layout": "trig4"}],
    "candidates": {
        "dummy": {
            "states": 247,
            "meets_req18": True,
            "april_2020_passed": True,
            "ljung_box_q": {"F": 85.2, "S": 49.6, "T": 61.7},
            "seconds": 2.989,
        },
        "trig4": {
            "states": 214,
            "meets_req18": True,
            "april_2020_passed": False,
            "ljung_box_q": {"F": 2286.4, "S": 1514.0, "T": 1507.3},
            "seconds": None,
        },
    },
    "rule": {
        "layout": "dummy",
        "all_blocks": {"differences": {"trig4": {"diff": -466.36, "se": 70.4}}},
        "reliable_blocks": {"differences": {"trig4": {"diff": -520.68, "se": 26.64}}},
    },
}
CONFIGURATION = {
    "stage": 9,
    "wedge": "high",
    "host": "a100",
    "gpus": 1,
    "chain_method": "vectorized",
    "batch": 4,
    "hourly_price_usd": 1.99,
    "seconds": 0.3,
    "fit_hours": 30.48,
    "fit_usd": 60.6552,
    "sbc_seconds": 0.05,
    "sbc_days": 10.95,
    "sbc_batched": {"replicates_per_gpu": 16, "days": 5.2},
    "measured": True,
    "feasible": True,
}


def stage(number, wedge, status, choice=None):
    return {"stage": number, "wedge": wedge, "status": status, "choice": choice}


def test_decision_lines_name_the_layout_and_each_models_gpu():
    a10 = CONFIGURATION | {"stage": 7, "wedge": "low", "host": "a10"}
    gpu_choice = {
        "status": "infeasible",
        "stages": [
            stage(7, "low", "chosen", a10),
            stage(9, "high", "infeasible"),
        ],
    }
    assert decision_lines(COMPARISON, gpu_choice) == [
        "The seasonal state for Stages 7–17 is `dummy`.",
        "Stage 7: 1 × `a10`, with NumPyro's `vectorized` chain method.",
        (
            "Stage 9, high wedge: no measured GPU configuration runs its fit within "
            "48 hours and its SBC within 14 days."
        ),
    ]


@pytest.mark.parametrize(
    "gpu_choice",
    [
        {"status": "unmeasured", "stages": [stage(8, "low", "unmeasured")]},
        # A model left unmeasured is never written up as infeasible, whatever the
        # overall status says.
        {
            "status": "infeasible",
            "stages": [stage(7, "low", "unmeasured"), stage(9, "high", "infeasible")],
        },
    ],
)
def test_an_incomplete_gpu_measurement_has_no_decision_lines(gpu_choice):
    with pytest.raises(ValueError, match="incomplete"):
        decision_lines(COMPARISON, gpu_choice)


def test_seasonal_rows_show_both_block_sets_the_checks_and_the_cost():
    assert seasonal_rows(COMPARISON) == [
        "| `dummy` | 247 | reference | reference | yes | passed | 85, 50, 62 | 2.989 |",
        (
            "| `trig4` | 214 | -466.4 ± 70.4 | -520.7 ± 26.6 | yes | failed "
            "| 2286, 1514, 1507 | — |"
        ),
    ]


def test_timing_rows_cover_every_batch_at_the_layouts_dimensions():
    memory = {"memory_analysis": {"total_bytes": 37_401_276_160}}
    record = {
        "label": "mac",
        "entries": [
            {
                "layout": "dummy",
                "stage": 9,
                "wedge": "high",
                "steps": 572,
                "states": 368,
                "cells": 277,
                "batches": [
                    {"batch": 1, "median_seconds": 5.6374, **memory},
                    {"batch": 4, "error": "skipped", **memory},
                    {"batch": 16, "error": "skipped", "memory_analysis": None},
                ],
            },
            {
                "layout": "trig4",
                "stage": 9,
                "wedge": "high",
                "steps": 572,
                "states": 335,
                "cells": 277,
                "batches": [{"batch": 1, "median_seconds": 4.9, **memory}],
            },
        ],
    }
    rows = [
        "| mac | 9 | high | 572 | 368 | 277 | 1 | 5.637 | 37.4 |",
        "| mac | 9 | high | 572 | 368 | 277 | 4 | skipped | 37.4 |",
        "| mac | 9 | high | 572 | 368 | 277 | 16 | skipped | — |",
    ]
    assert timing_rows([record], "dummy") == rows
    assert timing_rows([record], "trig") == rows  # trig has the dummy's dimensions


def test_gpu_rows_escape_dollars_and_mark_unmeasured_cells():
    unmeasured = CONFIGURATION | {
        "seconds": None,
        "fit_hours": None,
        "fit_usd": None,
        "feasible": False,
    }
    unmeasured |= {"sbc_seconds": None, "sbc_days": None, "sbc_batched": None}
    stage_8 = unmeasured | {"stage": 8, "wedge": "low"}
    gpu_choice = {
        "stages": [
            {"configurations": [CONFIGURATION]},
            {"configurations": [stage_8]},
        ]
    }
    assert gpu_rows(gpu_choice) == [
        (
            "| Stage 9, high wedge | a100 | 1 | `vectorized` | 4 | 0.300 | 30.5 "
            "| \\$60.66 | 10.9 | 5.2 | yes |"
        ),
        "| Stage 8, low wedge | a100 | 1 | `vectorized` | 4 | — | — | — | — | — | no |",
    ]


def test_the_decision_record_states_its_evidence():
    text = DECISION.read_text()
    comparison = json.loads((EVIDENCE / "seasonal-comparison.json").read_text())
    gpu_choice = json.loads((EVIDENCE / "gpu-choice.json").read_text())
    layout = comparison["rule"]["layout"]
    assert gpu_choice["layout"] == layout
    assert "**Status:** Accepted" in text
    timings = timing_rows(grid_records(gpu_choice), layout)
    configurations = gpu_rows(gpu_choice)
    # Empty tables would pass the check below vacuously.
    assert timings
    assert configurations
    expected = [
        *decision_lines(comparison, gpu_choice),
        *seasonal_rows(comparison),
        *timings,
        *configurations,
    ]
    missing = [line for line in expected if line not in text]
    assert missing == []
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run pytest tests/test_seasonal_state_decision.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'ces_revisions.pilot.decision_tables'`.

- [ ] **Step 4: Write the table module**

Create `src/ces_revisions/pilot/decision_tables.py`:

```python
"""The decision lines and Markdown table rows of docs/decisions/seasonal-state.md.

Each comes from the committed Stage 6 JSON: the seasonal comparison, the GPU choice, and
the probe-grid records. The decision record pastes them, and
``tests/test_seasonal_state_decision.py`` checks that every one is in the record, so the
record cannot drift from its evidence. Run ``python -m ces_revisions.pilot.decision_tables
COMPARISON GPU_CHOICE`` to print them.
"""

import json
import sys
from pathlib import Path

from ces_revisions.pilot.dimensions import matches_layout
from ces_revisions.pilot.gpu_choice import FIT_HOURS_MAX, SBC_DAYS_MAX
from ces_revisions.pilot.seasonal import REFERENCE_LAYOUT
from ces_revisions.vintages import raw as vintage_raw

SEASONAL_HEADER = (
    "| Layout | Stage 9 low states | elpd vs dummy, all blocks "
    "| elpd vs dummy, reliable blocks | Req 18 | April 2020 "
    "| Ljung–Box Q (F, S, T) | Seconds |"
)
TIMING_HEADER = (
    "| Host | Stage | Wedge | Steps | States | Cells | Batch | Seconds | Compiled GB |"
)
GPU_HEADER = (
    "| Model | Host | GPUs | Chain method | Batch | Seconds | Fit hours | Fit cost "
    "| SBC days | Batched SBC days | Feasible |"
)


def model_name(stage: int, wedge: str) -> str:
    """A Stage 7-9 model's name: Stage 7 has no wedge rows, Stages 8 and 9 have two."""
    return "Stage 7" if stage == 7 else f"Stage {stage}, {wedge} wedge"


def decision_lines(comparison: dict, gpu_choice: dict) -> list[str]:
    """The sentences that state the decision: the layout, then each model's GPU."""
    statuses = {gpu_choice["status"]} | {r["status"] for r in gpu_choice["stages"]}
    if "unmeasured" in statuses:
        raise ValueError("the GPU measurement is incomplete: no decision to record")
    lines = [f"The seasonal state for Stages 7–17 is `{comparison['rule']['layout']}`."]
    for result in gpu_choice["stages"]:
        name = model_name(result["stage"], result["wedge"])
        choice = result["choice"]
        if result["status"] == "chosen":
            lines.append(
                f"{name}: {choice['gpus']} × `{choice['host']}`, with NumPyro's "
                f"`{choice['chain_method']}` chain method."
            )
        else:
            lines.append(
                f"{name}: no measured GPU configuration runs its fit within "
                f"{FIT_HOURS_MAX:g} hours and its SBC within {SBC_DAYS_MAX:g} days."
            )
    return lines


def seasonal_rows(comparison: dict) -> list[str]:
    """One row per fitted layout: its elpd difference, Req 18, residuals, and cost."""
    rule = comparison["rule"]
    rows = []
    for entry in comparison["runs"]:
        layout = entry["layout"]
        candidate = comparison["candidates"][layout]
        if layout == REFERENCE_LAYOUT:
            every, reliable = "reference", "reference"
        else:
            every = _difference(rule["all_blocks"]["differences"][layout])
            reliable = _difference(rule["reliable_blocks"]["differences"][layout])
        rows.append(
            f"| `{layout}` | {candidate['states']} | {every} | {reliable} "
            f"| {_yes(candidate['meets_req18'])} "
            f"| {_passed(candidate['april_2020_passed'])} "
            f"| {', '.join(f'{q:.0f}' for q in candidate['ljung_box_q'].values())} "
            f"| {_number(candidate['seconds'], '.3f')} |"
        )
    return rows


def timing_rows(records: list[dict], layout_name: str) -> list[str]:
    """One row per host, grid entry at the layout's dimensions, and batch.

    A skipped batch shows "skipped", and one skipped before compiling shows no memory.
    """
    rows = []
    for record in records:
        for entry in record["entries"]:
            if not matches_layout(entry, layout_name):
                continue
            for batch in entry["batches"]:
                memory = batch["memory_analysis"]
                total = None if memory is None else memory["total_bytes"] / 1e9
                rows.append(
                    f"| {record['label']} | {entry['stage']} | {entry['wedge']} "
                    f"| {entry['steps']} | {entry['states']} | {entry['cells']} "
                    f"| {batch['batch']} "
                    f"| {_number(batch.get('median_seconds'), '.3f', 'skipped')} "
                    f"| {_number(total, '.1f')} |"
                )
    return rows


def gpu_rows(gpu_choice: dict) -> list[str]:
    """One row per model, priced host, and configuration the GPU rule weighed."""
    return [
        f"| {model_name(row['stage'], row['wedge'])} | {row['host']} | {row['gpus']} "
        f"| `{row['chain_method']}` | {row['batch']} "
        f"| {_number(row['seconds'], '.3f')} | {_number(row['fit_hours'], '.1f')} "
        f"| {_dollars(row['fit_usd'])} | {_number(row['sbc_days'], '.1f')} "
        f"| {_number((row['sbc_batched'] or {}).get('days'), '.1f')} "
        f"| {_yes(row['feasible'])} |"
        for result in gpu_choice["stages"]
        for row in result["configurations"]
    ]


def grid_records(gpu_choice: dict, root: Path = vintage_raw.ROOT) -> list[dict]:
    """The probe-grid records the GPU choice read, in its order."""
    return [json.loads((root / path).read_text()) for path in gpu_choice["records"]]


def _difference(difference: dict) -> str:
    return f"{difference['diff']:+.1f} ± {difference['se']:.1f}"


def _number(value: float | None, spec: str, missing: str = "—") -> str:
    return missing if value is None else format(value, spec)


def _dollars(value: float | None) -> str:
    # A literal dollar is escaped so that GitHub does not read a pair as math.
    return "—" if value is None else f"\\${value:.2f}"


def _yes(flag: bool) -> str:
    return "yes" if flag else "no"


def _passed(flag: bool) -> str:
    return "passed" if flag else "failed"


def main(argv: list[str] | None = None) -> None:
    comparison_path, choice_path = map(Path, sys.argv[1:] if argv is None else argv)
    comparison = json.loads(comparison_path.read_text())
    gpu_choice = json.loads(choice_path.read_text())
    layout = comparison["rule"]["layout"]
    sections = [
        decision_lines(comparison, gpu_choice),
        [
            SEASONAL_HEADER,
            "|---|---:|---|---|---|---|---|---:|",
            *seasonal_rows(comparison),
        ],
        [
            TIMING_HEADER,
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
            *timing_rows(grid_records(gpu_choice), layout),
        ],
        [
            GPU_HEADER,
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
            *gpu_rows(gpu_choice),
        ],
    ]
    print("\n\n".join("\n".join(section) for section in sections))


if __name__ == "__main__":
    main()
```

Run: `uv run pytest tests/test_seasonal_state_decision.py -q`
Expected: 6 passed and 1 failed. The failure is `test_the_decision_record_states_its_evidence`, with `FileNotFoundError` for `docs/decisions/seasonal-state.md`.

- [ ] **Step 5: Write the decision record**

Print the decision lines and the three tables:

```bash
uv run python -m ces_revisions.pilot.decision_tables docs/runs/stage-6/seasonal-comparison.json docs/runs/stage-6/gpu-choice.json
```

Create `docs/decisions/seasonal-state.md` from the template below. Paste the printed lines and tables where the template says, and replace each other `<…>` field with the value it names from the committed JSON. Replace the title with one that states the decision, for example "Keep the dummy seasonal state, and run Stage 7 on one A10 and Stage 9 on one GH200".

```markdown
# Choose the seasonal state and each stage's GPU for Stages 7–17

- **Status:** Accepted
- **Date:** <the commit date, YYYY-MM-DD>
- **Deciders:** Lowell Mason
- **Blast radius:** roadmap Stages 7–17: the seasonal block of every sector's state, which is the largest share of the state dimension, and the GPU type and count that each fit and SBC run rents from Lambda

## Context

Req 6 of [`specs/ces-revisions.md`](../../specs/ces-revisions.md) uses the dummy seasonal recursion and says that a trigonometric seasonal state may replace it if diagnostics favor it. Roadmap Stage 6 makes that choice together with the GPU measurement, because the seasonal block largely fixes the Stage 7–9 state dimension. Float64 throughput, memory, and the number of chains or SBC replicates run at once then set each stage's GPU (Req 17). Decision D5 of plan 11 chooses a GPU for each Stage 7–9 model rather than one for all stages, so a small model does not rent the largest model's GPU.

Plan 11 fitted the aggregate total-nonfarm pilot under three layouts with the same data, seeds, and sampler settings: `dummy` (11 states), `trig` (harmonics 1–5 and the Nyquist harmonic, 11 states), and `trig4` (harmonics 1–4, 8 states). It timed the engine's value and gradient at the Stage 7–9 dimensions on the Mac and on one Lambda instance of each GPU type it rented. Both rules are code (`src/ces_revisions/pilot/seasonal.py`, `src/ces_revisions/pilot/gpu_choice.py`), fixed by plan 11 before the runs recorded here. A planning spike under an earlier model had already applied the seasonal rule's clauses to these layouts (trig +15.4 ± 9.4, trig4 −466 ± 70 on benchmark-year blocks). The reliable-block rerun, the April 2020 and Ljung–Box conditions, and the GPU rule's step and lockstep constants were set after that spike evidence was read.

## Decision

<paste the lines that `uv run python -m ces_revisions.pilot.decision_tables docs/runs/stage-6/seasonal-comparison.json docs/runs/stage-6/gpu-choice.json` prints first: the layout line as one paragraph, then the five model lines as a bullet list>

Stages 10–17 keep Stage 9's Kalman states and cells: Req 17 samples their mixing variables and volatility paths rather than adding states. So each of them rents Stage 9's GPU at the wedge reading Stage 8 settles, and reruns the rule with its own measured steps per iteration. This holds while a stage keeps Stage 9's engine and NUTS kernel.

### The seasonal rule

The rule compares each alternative with the dummy on benchmark-year PSIS-LOO blocks. An alternative can replace the dummy only if it meets Req 18 and passes its own April 2020 check. It replaces the dummy if its elpd is more than 2 standard errors better. Otherwise an alternative with fewer states replaces it if its elpd is within 2 standard errors, its Ljung–Box Q is at most 1.25 times the dummy's in every closing column, and its measured value and gradient at Stage 9 low, $`T=572`$, is at least 25% cheaper on the Mac. The rule runs on every block and again on the blocks whose Pareto $`k`$ is at most 0.7 in every layout. If the two runs choose different layouts, the dummy stays. <state here whether the two runs agreed (`rule.reliable` in `docs/runs/stage-6/seasonal-comparison.json`), the clause each run used, and how many blocks were unreliable (`rule.unreliable_blocks`)>.

<paste the seasonal table that the command prints>

### The GPU measurement

Each row is one value and gradient of the engine at the chosen layout's dimensions, timed on a synthetic float64 model by `scripts/gpu_probe_grid.py`. "Batch" counts chains or SBC replicates run at once. "Compiled GB" is XLA's memory analysis of the compiled batch. A skipped batch did not fit the device's memory or the Mac's 20 GB guard. One with no compiled size was skipped without compiling, after a smaller batch did not fit. The `trig` layout has the dummy's dimensions, so the dummy's entries serve both. The records are in `docs/decisions/cloud-gpu-probe/`.

<name each D3 GPU type (A10, A100 40 GB, GH200) that was not measured, and why: not available, a driver that did not reach 580, or a record left incomplete. If every type was measured, write "Every D3 GPU type was measured.">

<paste the timing table that the command prints>

### The GPU rule

The rule runs once for each Stage 7–9 model: Stage 7, and Stages 8 and 9 under each wedge reading. A fit runs 4 chains of 2,000 NUTS iterations over all 572 months. The rule sizes every model at 127 leapfrog steps per iteration of the slowest chain: the pilot's 17 parameters took about 65, and Stages 7–9 sample many more. On one GPU the 4 chains run as one batch and wait for the deepest tree in each iteration (the pilot measured 1.44 times the slowest chain's steps). With one GPU per chain they run side by side at batch 1, on a 4-GPU instance. That configuration exists only for a GPU type that Lambda rents as one, priced at that instance's hourly rate. SBC runs 100 single-chain replicates of 1,490 iterations at the model's dimensions over 280 months. The projection assumes one replicate per GPU at a time: `run_sbc` takes one worker on a GPU host, and a multi-GPU instance runs one `pilot_run.py sbc --workers 1` shard per GPU under its own `CUDA_VISIBLE_DEVICES`. A configuration so needs ceil(100 / GPUs) rounds. A configuration is feasible if its fit takes at most 48 hours and its SBC at most 14 days. For each model, the cheapest feasible configuration, in dollars per fit at the price each instance was billed, wins. The "Batched SBC days" column projects a batched runner that Stage 7 would have to build. It decides nothing.

<paste the GPU table that the command prints>

<state the overall `status` in `docs/runs/stage-6/gpu-choice.json`, and for each model in `stages` its status, its chosen configuration's `sbc_days`, and the batched projection (`sbc_batched`). For each model whose status is "infeasible", say which limit each configuration missed, and give the cheapest fit-feasible configuration and its batched SBC days. Add "The roadmap's resume re-validates Stages 7–17 before it routes Stage 7." only when every D3 type has a record, or Lowell has confirmed that each missing type could not be measured.>

<state `band_fit_hours` for each model's chosen host, or for every priced host where the model's status is "infeasible": the fit hours at 31, 63, and 127 steps per iteration, so a reader sees how the projection moves if Stage 7–9 trees are shallower than the rule assumes>

## Consequences

- Stages 7–17 build each sector's seasonal block with the chosen layout from `src/ces_revisions/pilot/layout.py`, and size their models with `src/ces_revisions/pilot/dimensions.py`.
- Fits and SBC runs for Stages 7–9 rent their own model's GPU type and count by hand from Lambda, as CLAUDE.md describes, and Stages 10–17 rent Stage 9's at the wedge Stage 8 settles. The code takes its chain method from `chain_method(num_chains)`, so it never names a GPU count.
- <state whether every layout's Ljung–Box test rejects at 1% (`runs[].ljung_box` in `docs/runs/stage-6/seasonal-comparison.json`). If so, write: "No layout removes the pilot's residual seasonality at lags 11 and 12. Stage 18's posterior predictive suite and Stage 21's sensitivity suite own it.">

## Alternatives considered

- <one bullet for each layout the rule did not choose, `dummy` included when it lost: its elpd difference on both block sets, its measured saving, its Req 18 and April 2020 results, and the clause that kept it out>.
- **The other GPU types and counts.** <for each model, one line per GPU type that the rule did not choose, from the GPU table: its cheapest feasible configuration's cost per fit, or why it had none>.
- **One GPU for every stage.** The roadmap first asked for one GPU type and count for Stages 7–17. Sized for the largest model, that choice would rent the largest model's GPU for the smaller ones. D5 chooses per model instead. One GPU for every stage would keep all runs on one architecture, but the posterior does not depend on the hardware beyond floating-point rounding.
- **A parallel-in-time filter.** The engine runs the months one after another, so one chain at batch 1, as in an SBC replicate, may leave a GPU mostly idle. An associative-scan filter (Särkkä and García-Fernández, 2021) takes about 2 × log₂T rounds instead of T steps, at several times the float64 arithmetic. Dynamax's version cannot run this model as it stands (plan 11, D8). Stage 6 measures the engine as built, and the roadmap's resume takes the parallel filter up if a model is infeasible.

## Revisit triggers

- A Stage 7, 8, or 9 model's measured leapfrog steps per iteration are far from 127. Then rerun `scripts/pilot_run.py choose-gpu` with `--steps-per-iteration` set to the measured value.
- A later stage changes the engine or kernel that Stage 9 ran: Req 17 step (4)'s finite-mixture fallback at Stage 10, or a filter remedy or blocked kernel from Stage 17's sampler benchmark (`docs/decisions/engine.md`). Then time the changed engine at Stage 9's dimensions before that stage chooses its GPU. `--steps-per-iteration` does not cover a slower value and gradient.
- Lambda's prices or GPU types change.
- A posterior predictive check in Stage 18 traces a misfit to the seasonal block.
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run pytest tests/test_seasonal_state_decision.py -q`
Expected: `7 passed`.

- [ ] **Step 7: Update CLAUDE.md and README.md**

In `CLAUDE.md`, under "Layout and tooling", add after the `operators/` bullet:

```markdown
- `src/ces_revisions/pilot/` implements roadmap Stage 6: the aggregate total-nonfarm panel (`assembly.py`, `panel.py`), the three seasonal layouts (`layout.py`), the marginalized Jacobs–van Norden pilot (`model.py`), its simulator and prior predictive (`simulate.py`), SBC (`sbc.py`), the April 2020 check (`checks.py`), the Stage 7–9 dimensions (`dimensions.py`), and the two pre-registered decision rules (`seasonal.py`, and `gpu_choice.py`, which chooses a GPU for each Stage 7–9 model) recorded in `docs/decisions/seasonal-state.md`. NUTS runs at `TARGET_ACCEPT_PROB = 0.95` from `model.py`: at 0.8 the pilot diverges.
- `src/ces_revisions/runs/` is the model-agnostic run harness every fit uses: dispersed chain starts (`inits.py`), NUTS and the run directory (`fit.py`), Req 18 diagnostics (`diagnostics.py`), the log likelihood grouped by reference month and benchmark year (`inference_data.py`), and provenance (`provenance.py`). Run directories go to the gitignored `data/runs/`; small JSON evidence is committed under `docs/runs/`.
- `scripts/pilot_run.py` drives the pilot (`prior-predictive`, `fit`, `sbc`, `sbc-summary`, `compare-seasonal`, `choose-gpu`). On a GPU host `pilot_run.py sbc` refuses more than one worker. On a 4-GPU instance, start one shard per GPU, each under its own `CUDA_VISIBLE_DEVICES` with `--workers 1` and its own `--start` and `--stop`, all into one `--out-dir`, and run `sbc-summary` once every shard has finished. A replicate's seeds depend only on `--seed` and its index, so the split does not change the results (D7). `scripts/gpu_probe_grid.py` times the engine at the Stage 7–9 dimensions on one host and writes one record; on a rented GPU pass `--hourly-price`, and `--four-gpu-price` where Lambda rents a 4-GPU instance of the same type. `tests/test_seasonal_state_decision.py` checks `docs/decisions/seasonal-state.md` against `docs/runs/stage-6/seasonal-comparison.json`, `gpu-choice.json`, and the `*-grid.json` records that `gpu-choice.json` lists, and fails if they change. The rest of `docs/runs/stage-6/` is Stage 6's committed exit evidence, which no test guards. Write new measurements to `data/runs/`; re-measuring the committed ones means rewriting that record.
```

In `CLAUDE.md`, in the "Commands" block, add after the `operator_artifacts.py build` line:

```bash
uv run python scripts/pilot_run.py prior-predictive --out data/runs/prior-predictive.json   # Req 15 check before any fit; docs/runs/stage-6/ holds the committed run
uv run python scripts/pilot_run.py fit --layout dummy --label stage6-pilot   # 4-chain pilot fit to data/runs/ (~30 min on the Mac)
uv run python scripts/gpu_probe_grid.py --label mac --out data/runs/mac-grid.json --batch 1 4 --max-bytes 20e9   # time the engine at the Stage 7-9 dimensions
mkdir -p data/runs && for gpu in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/pilot_run.py sbc --workers 1 --start $((gpu * 25)) --stop $((gpu * 25 + 25)) --out-dir data/runs/sbc > data/runs/sbc-$gpu.log 2>&1 & done   # SBC on a 4-GPU instance, one shard per GPU
uv run python scripts/pilot_run.py sbc-summary --out-dir data/runs/sbc   # summarize once every shard has finished
```

In `README.md`, in the `Status` tree, replace the five lines that run from `docs/decisions/` through `devices.py picks the chain method, engine_probe.py times the engine` with:

```text
docs/decisions/                       decision records (engine.md: the Kalman engine;
                                      link-relatives.md: the post-March operator;
                                      seasonal-state.md: the seasonal state and the GPU)
  cloud-gpu-probe/                    engine timing records: the Mac and each rented GPU type
docs/runs/stage-6/                    the Stage 6 pilot's committed evidence
src/ces_revisions/                    Python package (src layout): kalman.py is the state-space engine,
                                      vintages/ and annual/ build the Stage 3/4 inputs, operators/ the
                                      Stage 5 maps, pilot/ the Stage 6 pilot, runs/ the run harness,
                                      devices.py picks the chain method, engine_probe.py times the engine
```

Then correct the sentences that this plan makes stale. Each old text occurs once in its file.

In `CLAUDE.md`, replace:

```text
and the sparse deterministic operator layer (`src/ces_revisions/operators/`).
```

with:

```text
the sparse deterministic operator layer (`src/ces_revisions/operators/`), the Stage 6 aggregate pilot (`src/ces_revisions/pilot/`), and the model-agnostic run harness (`src/ces_revisions/runs/`).
```

In `CLAUDE.md`, replace:

```text
fastexcel (Polars' Excel reader, for Stage 3's workbooks), and python-dotenv (the Stage 4 network-command contact configuration).
```

with:

```text
fastexcel (Polars' Excel reader, for Stage 3's workbooks), python-dotenv (the Stage 4 network-command contact configuration), SciPy (ranks for the Req 18 diagnostics and the Ljung–Box p-value), and h5netcdf with h5py (the netCDF backend that writes each run's `DataTree` to one file).
```

In `CLAUDE.md`, replace:

```text
the `DETERMINISTIC_GPU_FLAG` that `tests/conftest.py` adds to `XLA_FLAGS`, and `chain_method(num_chains)`
```

with:

```text
the `DETERMINISTIC_GPU_FLAG` that `tests/conftest.py` adds to `XLA_FLAGS`, `jax_versions()` and `nvidia_driver()`, which run and grid records copy, and `chain_method(num_chains)`
```

In `CLAUDE.md`, replace:

```text
Records are committed to `docs/decisions/cloud-gpu-probe/`, which holds the Mac baseline.
```

with:

```text
Records are committed to `docs/decisions/cloud-gpu-probe/`, which holds the Mac baseline beside the Stage 6 grid records from `scripts/gpu_probe_grid.py`.
```

In `CLAUDE.md`, replace:

```text
The GPU type and count follow what the model requires, measured with the engine probe at roadmap Stage 6;
```

with:

```text
The GPU type and count follow what each model requires: roadmap Stage 6 measured them, and `docs/decisions/seasonal-state.md` records the choice for each Stage 7–9 model;
```

In `CLAUDE.md`, replace:

```text
# the synthetic pilot's four-chain NUTS fit and the Stage 3 build (~30 s)
```

with:

```text
# the MCMC fits, the SBC smoke runs, and the Stage 3 build (a few minutes)
```

In `README.md`, replace:

```text
Stage 6 of the [roadmap](specs/ces-revisions-roadmap.md) measures them with the engine probe.
```

with:

```text
Stage 6 of the [roadmap](specs/ces-revisions-roadmap.md) measured them, and [`docs/decisions/seasonal-state.md`](docs/decisions/seasonal-state.md) records the choice for each stage.
```

In `README.md`, replace:

```text
with fastexcel to read the source workbooks.
```

with:

```text
with fastexcel to read the source workbooks, SciPy for ranks and test statistics, and h5netcdf with h5py to write each run's InferenceData.
```


- [ ] **Step 8: Run every test tier**

```bash
uv run pytest -m "not slow and not network" -q
uv run pytest -m slow -q
uv run ruff check
uv run ruff format --check
```

Expected: the fast tier gives `632 passed, 21 deselected`; the slow tier passes in a few minutes; ruff reports no problem.

- [ ] **Step 9: Check the exit criteria**

Check each row of the exit table below against the committed evidence, and report any row that fails.

- [ ] **Step 10: Lint and commit**

```bash
uv run ruff format src/ces_revisions/pilot/decision_tables.py tests/test_seasonal_state_decision.py
uv run ruff check src/ces_revisions/pilot/decision_tables.py tests/test_seasonal_state_decision.py
git add src/ces_revisions/pilot/decision_tables.py docs/runs/stage-6/gpu-choice.json docs/decisions/seasonal-state.md tests/test_seasonal_state_decision.py CLAUDE.md README.md
git commit -m "Record the seasonal-state and GPU decision for Stages 7-17"
```

## Exit criteria

| Roadmap Stage 6 exit item | Where it is met |
|---|---|
| The pilot meets Req 18, with diagnostics JSON and provenance in the run directory (Req 17 step 2) | Task 10 Step 5; `docs/runs/stage-6/pilot-fit/diagnostics.json` (`meets_req18`) and `provenance.json` |
| A test asserts irregular anchor rows, NaN cells, and no latent state among sampled sites | `tests/test_pilot_model.py::test_model_samples_exactly_the_scalar_sites_and_factors_the_filter` (synthetic panels); `tests/test_pilot_fit.py::test_the_likelihood_sees_irregular_anchor_rows_and_missing_cells` and `::test_the_model_samples_only_the_scalar_sites_and_no_state` (real panel) |
| A test asserts the 1979 leg is partially observed with exactly one 2003 shift (Req 4) | `tests/test_pilot_assembly.py::test_the_1979_leg_is_partially_observed_with_exactly_one_2003_shift` |
| The repeated EMPLOY25M10 vintage is flagged or dropped before fitting | `employ_without_repeats`; `tests/test_pilot_assembly.py::test_the_repeated_october_2025_employ_vintage_is_dropped` |
| The pre-1997 EMPLOY/revision-table disagreements are pinned | `tests/test_pilot_assembly.py::test_employ_disagrees_with_the_revision_table_in_ten_pre_1997_and_three_2020_cells` pins all 13: the 10 pre-1997 cells and 3 in EMPLOY20M5 |
| The prior predictive meets Req 15's monthly constraint before the fit | Task 10 Step 2; `docs/runs/stage-6/prior-predictive.json`; `tests/test_pilot_fit.py::test_the_prior_predictive_meets_the_monthly_constraint_on_the_real_panel` |
| The April 2020 movement is absorbed by the latent state under the crisis multiplier (Req 6) | Task 10 Step 5; `docs/runs/stage-6/pilot-fit/april_2020.json` (`passed`: every subsampled draw finite, and the whole 90% interval within ±5% of the third print) |
| Reduced-panel SBC recovers the news/noise allocation (bullet 7, reduced half) | Task 11; `docs/runs/stage-6/sbc-summary.json` (`passed`) |
| The seasonal-state decision is committed with the GPU cost measurement and the GPU type and count | Task 14; `docs/decisions/seasonal-state.md` (one GPU type and count per Stage 7–9 model, D5), `tests/test_seasonal_state_decision.py` |
| The dispersed-chain-starts item is closed | Task 6; `runs/inits.py`, recorded in every run's provenance (`init`) |
| The `A_t`/`Z_t` gradient-check item is closed if the pilot samples a parameter in `transition_matrix` or `observation_matrix` | Not triggered: no sampled parameter enters either matrix (planning choice 8). The item stays open for Stage 7's loadings |
| If the cost measurement makes Stage 7–9 fits or their SBC infeasible, resume re-validates Stages 7–17 | Task 14 Step 1; `gpu-choice.json` (`status`: "infeasible" triggers it), stated in the decision record |
