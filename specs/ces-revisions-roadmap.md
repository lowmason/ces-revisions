# ces-revisions — Roadmap

> For agentic workers: REQUIRED SKILL: derive-roadmap — resume via its
> reconcile step; route each unticked stage per its ROUTING line; never plan
> this document wholesale.

Source spec: [`specs/ces-revisions.md`](ces-revisions.md) (`237b426`, 2026-09-12). Derived
2026-09-12 at fresh entry; 27 stages. Resumed 2026-09-13 at `cd835c5`: Stage 1 complete per its
stamp, and Stages 2–27 re-validated against what it shipped.

## Gap analysis

Search boundary: the whole repository at `237b426` — `src/ces_revisions/__init__.py` is the
two-line scaffold `main()`, `tests/test_smoke.py` holds two smoke tests, `pyproject.toml` has
`dependencies = []`, and there is no `docs/`, no data, and no `specs/deferred_items.md`. The
table is that entry snapshot; a Note dated 2026-09-13 records what a later stage has shipped.

| Req | Verdict | Evidence | Note |
|---|---|---|---|
| 1 Vintage panel | missing | none found; `pyproject.toml:9` `dependencies = []` | — |
| 2 Stage definitions | missing | none found | — |
| 3 Covariate panel | missing | none found | FY2009+ in v1; FY1979 is Stage 27 |
| 4 Estimation window | missing | none found | — |
| 5 Sector set / aggregation | missing | none found | — |
| 6 Latent process | missing | none found | — |
| 7 News/noise measurement | missing | none found | — |
| 8 Scale equation + covariate map | missing | none found | — |
| 9 Joint NSA/SA + decomposition | missing | none found | channel (b) is Stages 25–26; carries an (open) |
| 10 Benchmark observations + operators | missing | none found | carries an (open) |
| 11 Birth–death | missing | none found | — |
| 12 Hierarchy | missing | none found | — |
| 13 Capacity + shrinkage | missing | none found | — |
| 14 Structural change | missing | none found | — |
| 15 Priors + prior predictive | missing | none found | — |
| 16 Identification assumptions | missing | none found | — |
| 17 Inference | missing | no NumPyro/JAX/Dynamax in `pyproject.toml` | two (open) items block the first fit. 2026-09-13: Stage 1 (plan 1) discharged both and shipped the engine (`src/ces_revisions/kalman.py`, `docs/decisions/engine.md`); steps (1)–(5) remain in Stages 5, 6, 10, 17; "InferenceData" is read as the ArviZ 1.x `DataTree` (`docs/decisions/engine.md:32`); a scratch probe on an M4 Max CPU, not the fit target, timed one dense value+grad at 0.54 s (T=280, n=150, p=70) and about 1.1 s at n=250, p=100 or at T=570; Stage 6 re-measures on the cloud GPU |
| 18 Validation | missing | none found | pytest `network`/`slow` markers exist as infrastructure only. 2026-09-13: Stage 1's slow tier asserts the MCMC thresholds on the synthetic pilot, with energy-BFMI above 0.3 per chain and under 1% of draws at maximum tree depth 10 (`tests/test_synthetic_pilot.py:30-34`) — the reading later stages inherit unless a plan records a Deviation |
| 19 Outputs | missing | none found | — |
| 20 Written finding | missing | no `docs/` | — |
| — greeting console script | in-code-but-not-in-spec | `src/ces_revisions/__init__.py:1-2`; `[project.scripts]` | `uv init` scaffold; flagged, not a defect |

Three ambiguities were resolved with the user at entry and govern the stages below: (a) Req 15's
prior-predictive gate is read as layer-local prior predictive checks before each pilot/comparator
fit, with the complete Req 15 plan gating the first conditional fit of the full model (Stage 16);
(b) Req 18's covariate retention is a reporting label under the fixed Req 8 map, computed in
Stage 19 — no covariate is removed and no refit-based scheme exists; (c) the two non-marginal
sensitivities named by Req 14 and Req 15 are one late brainstorming stage (Stage 22), not parked.

## Stages

Sequencing follows the spec's Rollout note where it states an order — Req 17's two (open)
items open the roadmap; Req 17's own order of work (operators tested before any fit, the pilot,
the full model, BlackJAX only after a benchmark) is the backbone of Stages 5–17; Req 18's
comparator order (0)→(3) is the model-layer order; the X-13 channel and the FY1979 panel are
last, as Out of scope names them. It diverges from the drafts' value ranking and the spec's
Motivation deliberately: the collection-window and capacity drivers land at Stages 12–16 and the
seasonal decomposition at Stages 9 and 17, because the comparators are nested and the estimand
layer cannot be validated before the Gaussian core's news/noise and wedge-rank identification
is proven. The written finding is split by information order: its data and archive inventory
(Stage 2) runs alongside Stage 1 because the covariate stages code its rulings, while the
literature review (Stage 23) gates nothing but the report. Bullet numbers cite the spec's
Verification bullets in file order (1–14). Decision records live under `docs/decisions/`.
Stages 12–14 are independent of Stages 6–11, and Stage 23 of Stages 6–22; an orchestrator may
run them alongside. The cloud GPU environment that later fits run on is its own brainstorming cycle
outside this roadmap (user decision, 2026-09-13) — no spec requirement covers it — running
alongside Stages 2–5 and gating Stage 6.

- [x] Stage 1: Inference stack and Kalman engine on Python 3.14\
      Objective: Discharge Req 17's two (open) items by pinning the inference stack under Python 3.14 and shipping the marginalized Kalman engine — filter likelihood, smoother, per-step log-density contributions, one-step-ahead moments — as a tested NumPyro term proven on a synthetic pilot.\
      Spec: Req 17; Rollout note\
      Gap closed: Req 17 (engine; both (open) items)\
      Consumes: the spec and the scaffold only.\
      Produces: stack pins in `pyproject.toml`/`uv.lock` with an import test per package; the Kalman engine module (`src/ces_revisions/kalman.py`) over time-varying transition/emission matrices and offsets, NaN cells, irregular rows, and per-cell time-varying observation and process covariances, exposing the smoother, per-time-step contributions, and one-step-ahead moments; its dense-reference and marginalization-site tests; the synthetic pilot test (slow-marked); the engine determination at `docs/decisions/engine.md`; the dense oracle `tests/dense_reference.py`, the parity gate for any engine change or gradient check; the engine's input contracts (float64 arrays; a missing cell's `observation_cov` entries reach only `one_step_cov`; the smoother needs positive-definite predicted covariances after the first step — `kalman.py` docstrings).\
      Exit: `uv sync` on 3.14 with the pins recorded and every import test passing, any package failing on 3.14 recorded with the fallback taken; the synthetic pilot passes on the determined engine with time-varying rows, NaN cells, irregular annual rows, four chains, zero divergences (bullet 5); the dense-reference test matches likelihood and smoother moments to tolerance; no latent state among sampled sites; the engine determination is committed.\
      ROUTING: writing-plans

- [x] Stage 2: Data and archive inventory\
      Objective: Write the inventory half of the Req 20 finding — per-driver data inventory with publication lags, the Req 9 archive inventory, and the five draft-disagreement rulings.\
      Spec: Req 20 (inventory, rulings); Req 9 (open); Req 3; Rollout note\
      Gap closed: Req 9 ((open) archive inventory); Req 20 (inventory half)\
      Consumes: the spec, the prompt, and the three research drafts in `specs/`; the CLAUDE.md Markdown conventions. Independent of Stage 1.\
      Produces: `docs/ces-revisions-review.md` with the inventory sections complete and the literature sections stubbed — the data inventory (one row per Req 3 series: vintage coverage, frequency, earliest date, access route, hand-build constraint, first-publication lag); the archive inventory (per vintage: specification, prior-adjustment, and outlier files present or absent; unrounded NSA inputs recoverable or not); the disagreement rulings (the five named, each with a primary-source citation).\
      Exit: the archive inventory lists every vintage May 2003–present with the three file types and unrounded-input status (bullet 4); the data inventory has one row per Req 3 series with a publication-lag column; each ruling has its own cited subsection; `uv run ruff format --check` passes on the document (bullet 13, inventory half).\
      ROUTING: writing-plans

- [ ] Stage 3: Vintage panel, stage labels, and same-release differencing\
      Objective: Build the Req 1 raw-value, transformations, and long tables for supersector SA and NSA vintages from May 2003 and the 1979+ total-nonfarm leg, label stages per Req 2, and ship the differencing operator that reproduces the official revision table.\
      Spec: Req 1; Req 2; Req 4; Req 5; Req 9 (identity)\
      Gap closed: Req 1 (vintage sources); Req 2 (data half); Req 4 (data half); Req 5 (data half); Req 9 (accounting identity)\
      Consumes: From Stage 1: Polars in the stack pins.\
      Produces: the vintage panel (immutable raw-value parquet with content hash; transformations parquet; long panel keyed per Req 1); the raw vintage file archive (every fetched release file, content-hashed); the stage-label table (F/S/T/B/M per Req 2, right-censored-M, nonstandard-release, `concept_regime`, missing-vintage flags for the 2003 gap and the 2025 lapse); the 1979+ aggregate leg (SA and NSA F/S/T same-release changes from the BLS revision table; RTDSM SA level vintages); the release-date index (release, closing, and publication dates); the differencing module; the accounting-decomposition table (the Req 9 identity by sector, stage, regime); recorded fixtures for the hermetic tier.\
      Exit: the differencing module reproduces every row of the BLS 1979–present revision table — SA and NSA, all three pairwise MARs — to rounding, including the 2003 gap (bullet 1); re-ingest reproduces every panel value from raw plus transformations; every stage-label row carries exactly one stage, M is right-censored where its vintage does not exist, and the preliminary benchmark is not a stage (Req 2); no sector vintage precedes May 2003 and the aggregate leg carries `concept_regime` (Req 4); the eleven sector values sum to the published total within rounding in every release file (Req 5); the Req 9 identity holds on every same-release pair; live fetches carry the `network` marker and the default tier passes on fixtures.\
      ROUTING: writing-plans

- [ ] Stage 4: Benchmark, birth–death, QCEW-revision, and sample tables\
      Objective: Ingest the annual-source tables the operators and covariates read — benchmark revisions and sector contributions with publication dates, reconstruction events, birth–death forecasts and forecast-vs-realized rows, the 2017+ QCEW revision sequence, and coverage/RSE by supersector-year.\
      Spec: Req 1; Req 3 (sample rows); Req 10 (inputs); Req 11\
      Gap closed: Req 1 (benchmark sources); Req 3 (sample panel); Req 11 (data half)\
      Consumes: From Stage 3: the release-date index. From Stage 2: the data inventory.\
      Produces: the benchmark table (preliminary 2000+ and final 1979+ March revisions, sector contributions 2003+, a publication date per figure); the reconstruction-events table (the single source of "documented reconstruction"); the birth–death table (monthly forecasts by supersector, revised values, forecast-vs-realized rows; zero for government); the QCEW revision table (2017+ aggregate sequence with a revision-precision proxy per benchmark year); the sample panel (coverage/RSE by supersector-year harmonized across definition switches); every value with provenance and an `observable_at` date.\
      Exit: a final row exists for every benchmark year 1979–2025 and a preliminary row for 2000–2025, each with a publication date; every reconstruction event and benchmark publication date exists in Stage 3's release-date index; the birth–death table is zero for government in every row (Req 11, data half); the sample panel carries a definition-break flag at each documented switch (Req 3); a test asserts every `observable_at` is populated.\
      ROUTING: writing-plans

- [ ] Stage 5: Benchmark, aggregation, and seasonal-mapping operators\
      Objective: Implement and unit-test the Req 17 step (1) operator set against published benchmark articles before any fit, settling link-relative recoverability per benchmark year.\
      Spec: Req 10; Req 5; Req 9; Req 11; Req 17 (step 1)\
      Gap closed: Req 10 (operator half; (open) determination); Req 5 (operator half); Req 11 (operator half); Req 17 (step 1)\
      Consumes: From Stage 3: the vintage panel, stage-label table, raw file archive, differencing module. From Stage 4: the benchmark, reconstruction-events, and birth–death tables. From Stage 1: JAX in the stack pins.\
      Produces: the operators module (sparse JAX: aggregation with reconciliation; the wedge and its reconstruction terms; the post-March recursion in both Req 10 representations; the B→M operator components; the seasonal mapping; missing-vintage masks); fixtures of published B vintages; the recoverability table (per benchmark year: representation, recovered or inferred series, reconstruction-error variance for inferred years); the written determination at `docs/decisions/link-relatives.md`; the operator-test-results record every later run stores in its provenance.\
      Exit: the wedge operator reproduces the published B vintage for every benchmark year 2003–2025 to rounding with nonzero reconstruction terms only for documented reconstructions, and the post-March recursion is attempted for every year with the table recording pass/fail and residuals, reproducing April–October B months for every year marked recoverable (bullet 2); the written determination exists with one row per year and nonzero reconstruction-error variance exactly for inferred years (bullet 3, determination half); a test asserts the two representations are never applied to the same year (Req 10); aggregating sector vintages reproduces the published total (Req 5); the recursion consumes revised birth–death values (Req 11).\
      ROUTING: writing-plans

- [ ] Stage 6: Aggregate linear-Gaussian pilot and run harness\
      Objective: Run Req 17 step (2) on the real total-nonfarm aggregate — 2003+ closings and the 1979+ leg with its composite 2003 shift, March anchor rows only — to debug state layout and sampler geometry, and ship the run harness every later fit reuses.\
      Spec: Req 17 (step 2); Req 4; Req 6; Req 7; Req 15; Req 18\
      Gap closed: Req 4 (model half); Req 6 (aggregate); Req 7 (aggregate); Req 17 (step 2)\
      Consumes: From Stage 1: the engine module, the engine determination, the dense oracle. From Stage 3: the vintage panel (total-nonfarm rows), stage-label table, 1979+ leg, release-date index. From Stage 4: the benchmark table (final March anchors). From Stage 5: the operators module (March selection, masks), the operator-test-results record. Outside this roadmap: the cloud GPU environment and its decision record.\
      Produces: the aggregate pilot model (Req 6 process including the crisis multiplier, Req 7 F/S/T news and noise, the 1979 leg as same-release change observations with one 2003 shift, March anchors as irregular rows, no wedge rows); the run harness (InferenceData writer with log-likelihood grouped by reference month and benchmark year; diagnostics JSON; provenance of seeds, data hashes, versions, and the operator-test record); the synthetic-panel simulator, SBC harness, and prior-predictive harness at aggregate scope; the seasonal-state decision at `docs/decisions/seasonal-state.md`, recording the engine's value-and-gradient time on the cloud GPU at the Stage 7–9 state and cell dimensions of the chosen layout (the seasonal block is the largest share of each sector's state, so its choice largely fixes those dimensions); the pilot InferenceData.\
      Exit: the pilot meets Req 18 thresholds with diagnostics JSON and provenance in the run directory (Req 17 step 2); a test asserts the likelihood contains irregular anchor rows and NaN cells and no latent state among sampled sites; a test asserts the 1979 leg is partially observed with exactly one 2003 shift (Req 4); the layer-local prior predictive satisfies the Req 15 monthly constraint at total nonfarm before the fit; the April 2020 movement is absorbed by the latent state under the crisis multiplier (Req 6); reduced-panel SBC recovers the news/noise allocation (bullet 7, reduced half); the seasonal-state decision is committed with the GPU cost measurement; the dispersed-chain-starts item in `specs/deferred_items.md` (plan 1) is closed, and its `A_t`/`Z_t` gradient-check item too if the pilot samples a parameter entering `transition_matrix` or `observation_matrix`. If the cost measurement makes the Stage 7–9 fits or their SBC infeasible, resume re-validates Stages 7–17 before routing Stage 7.\
      ROUTING: writing-plans

- [ ] Stage 7: Gaussian closing-stage core across supersectors\
      Objective: Widen the pilot to the eleven supersectors with the common factor, cross-sector news factors, shared noise recursion, and reconciliation — comparator (0)'s closing-stage half.\
      Spec: Req 5; Req 6; Req 7; Req 4; Req 15; Req 18\
      Gap closed: Req 5 (model half); Req 6; Req 7 (closing stages)\
      Consumes: From Stage 6: the pilot model, run harness, the three harnesses, the seasonal-state decision. From Stage 3: the vintage panel (sector NSA rows), stage-label table. From Stage 5: the operators module (aggregation, masks). From Stage 1: the dense oracle.\
      Produces: the sector Gaussian core and the comparator-settings registry (from here every model is one module with nested settings); the harnesses extended to sector panels; the pre-2003 sector-state decision at `docs/decisions/pre-2003-states.md`; fitted InferenceData and diagnostics.\
      Exit: the fit meets Req 18 thresholds (bullet 8, comparator (0) closing half); a test asserts sector states sum to the total within the reconciliation tolerance, the published total is observed in its own right, and no sector observation row precedes May 2003 (Req 5, Req 4); loading and news-factor sign constraints hold in every draw (Req 6, Req 7); the layer-local prior predictive shows sector revisions aggregating plausibly before the fit; SBC on synthetic sector panels recovers the news/noise allocation and the news-factor share (bullet 7, sector half); the `A_t`/`Z_t` gradient-check item in `specs/deferred_items.md` (plan 1) is closed if Stage 6 left it open.\
      ROUTING: writing-plans

- [ ] Stage 8: Benchmark anchors, wedge, post-March recursion, and mature horizon in the model\
      Objective: Enter the preliminary and final benchmarks as correlated fallible anchors, the B vintage through the wedge operator, the post-March recursion with propagated reconstruction error, the B→M operator with right-censored M, and the T→B stage mean — completing comparator (0) NSA and producing the latent path against the linear wedge.\
      Spec: Req 10; Req 2; Req 7; Req 11; Req 15; Req 18\
      Gap closed: Req 10 (model half; propagation); Req 2 (model half); Req 11 (model half); Req 7 (B and M increments)\
      Consumes: From Stage 7: the sector core, comparator settings, harnesses, run harness. From Stage 5: the operators module (wedge, recursion, B→M components, masks), the recoverability table. From Stage 4: the benchmark, QCEW revision, and birth–death tables. From Stage 3: the stage-label table (B and M rows). From Stage 1: the smoother.\
      Produces: comparator (0) NSA complete; the probabilistic-wedge function and the latent-path-vs-wedge table (first version, every benchmark year); the third-stage-noise/benchmark-error correlation as a setting; the B→M composition record at `docs/decisions/b-to-m.md`; fitted InferenceData and diagnostics.\
      Exit: the fit meets thresholds (bullet 8, comparator (0) NSA); a test asserts reconstruction-error variance is nonzero exactly for inferred years and that zeroing it changes the likelihood (bullet 3 closed); a test asserts B rows enter only through the wedge operator with the post-operator residual within rounding for every year (Req 10); right-censored M is integrated out and reported as a posterior predictive, and the preliminary benchmark is an auxiliary observation (Req 2); the T→B stage-mean prior center matches the forecast-vs-realized rows (Req 11); the layer-local prior predictive satisfies the Req 15 wedge and annual-March constraints before the fit; SBC recovers the wedge rank (bullet 7, wedge half); the latent-path-vs-wedge table exists for 2003–2025 (Req 10 output, first version).\
      ROUTING: brainstorming — the T→B half is determined; the open question is the B→M operator's composition and how reconstruction error enters for inferred years.

- [ ] Stage 9: Joint NSA/SA measurement and the model-based seasonal decomposition\
      Objective: Add SA vintages as observations of the same release-specific measurement through the single mature seasonal component (Req 9 channel (a)), completing comparator (0) and the model-based decomposition.\
      Spec: Req 9; Req 4; Req 15; Req 18\
      Gap closed: Req 9 (model half, channel (a)); Req 4 (SA rows of the 1979 leg)\
      Consumes: From Stage 8: comparator (0) NSA, settings, harnesses, run harness. From Stage 5: the operators module (seasonal mapping). From Stage 3: the vintage panel (SA rows), the 1979 leg SA rows, the accounting-decomposition table, the stage-label table (SA-side M). From Stage 1: the smoother.\
      Produces: comparator (0) joint; the decomposition function and the model-based decomposition table (by sector, stage, regime, with intervals) on comparator (0); settings updated; fitted InferenceData and diagnostics.\
      Exit: the fit meets thresholds (bullet 8, comparator (0) joint); a test asserts SA rows share the seasonal component with NSA rows and no second seasonal state exists (Req 9); the model's observation mapping reproduces the accounting-decomposition table on the published panel (Req 9); the pre-2003 SA change rows enter through the Req 9 equation (Req 4, model half closed); SBC recovers the sign of the per-stage NSA/factor correlation; the decomposition table exists for comparator (0) (bullet 11, first version).\
      ROUTING: writing-plans

- [ ] Stage 10: Student-t news, noise, and process innovations\
      Objective: Replace Gaussian errors with Student-t scale mixtures — news, noise, the bivariate NSA/factor news, and the process innovations — sampling mixing variables with the linear states marginalized: comparator (1).\
      Spec: Req 8; Req 6; Req 9; Req 17 (steps 3–4); Req 15; Req 18\
      Gap closed: Req 8 (Student-t layer); Req 17 (steps 3–4)\
      Consumes: From Stage 9: comparator (0) joint, settings, harnesses, run harness. From Stage 1: the engine module, the dense oracle.\
      Produces: comparator (1) (64-bit, non-centered innovations); the mixing-representation decision at `docs/decisions/mixing.md`, with the engine extension and its dense-reference test if the finite-mixture fallback is adopted; fitted InferenceData and diagnostics.\
      Exit: the fit meets thresholds in 64-bit (bullet 8, comparator (1)); a test asserts fixing the degrees of freedom large reproduces comparator (0) joint's log-likelihood (Req 8 nesting); the decision record exists, with the triggering benchmark if the fallback was adopted (Req 17 step 4); SBC recovers the degrees of freedom and the allocation on heavy-tailed synthetic panels; the layer-local prior predictive satisfies the Req 15 tail constraint before the fit; rank plots for the degrees of freedom exist.\
      ROUTING: writing-plans

- [ ] Stage 11: Stochastic volatility per transition\
      Objective: Add one persistent log-volatility path per transition with sector loadings, sampled non-centered — comparator (2) — and record the sampler benchmark.\
      Spec: Req 8; Req 15; Req 17 (step 5 record); Req 18\
      Gap closed: Req 8 (SV layer)\
      Consumes: From Stage 10: comparator (1), settings, harnesses, run harness, the mixing decision.\
      Produces: comparator (2); the sampler benchmark record (wall time per effective sample by parameter block) at `docs/decisions/sampler-benchmark.md`; fitted InferenceData and diagnostics.\
      Exit: the fit meets thresholds with rank plots for SV persistence and innovation (bullet 8, comparator (2)); a test asserts fixing the volatility path to zero reproduces comparator (1)'s log-likelihood (Req 8 nesting); loadings have employment-weighted mean one (Req 8); SBC recovers the volatility-path ranks; the layer-local prior predictive reproduces pandemic tails without raising the post-2022 baseline before the fit; the benchmark record exists.\
      ROUTING: writing-plans

- [ ] Stage 12: Collection-window panel, first-stage C1 model, and macro controls\
      Objective: Build the release-date-keyed covariate panels — the business-day collection window with its indicators, C1/C2/C3 and RR as distinct series with the calendar-predicted/residual C1 split, and macro/tail controls aligned to each release's information set.\
      Spec: Req 3 (collection and macro rows)\
      Gap closed: Req 3 (collection and macro rows)\
      Consumes: From Stage 2: the data inventory, the collection-versus-response ruling. From Stage 3: the release-date index, the stage-label table (nonstandard-release flags). Independent of Stages 6–11.\
      Produces: the collection-window panel (business days, holiday/nonstandard/lapse indicators, C1/C2/C3, RR, calendar-predicted and residual C1) with the first-stage C1 model and its validation; the macro-controls table keyed by release date; `observable_at` per value; provenance and definition-break flags.\
      Exit: the panel is committed with provenance and flags and no monthly row carries an interpolated annual value (bullet 14, collection half); a test asserts RR and C3 are distinct series and predicted plus residual equals realized C1; a test asserts the first-stage design contains exactly the Req 3 regressor set, with an out-of-sample fit statistic recorded; the business-day count exists for every release since May 2003; every macro-control value's `observable_at` is no later than its release date (Req 3).\
      ROUTING: writing-plans

- [ ] Stage 13: Institutional capacity panel and shutdown events, FY2009–present\
      Objective: Build the fiscal-year institutional panel, the four-field shutdown-events table, and the descriptive-only party-composition table.\
      Spec: Req 3 (institutional rows); Req 13 (inputs)\
      Gap closed: Req 3 (institutional rows)\
      Consumes: From Stage 2: the data inventory (institutional rows), the rulings on the 2018–19 lapse, 2025 program cuts, and FTE concepts. From Stage 3: the release-date index. Independent of Stages 6–12.\
      Produces: the institutional panel (FY2009+, per-series provenance, definition-break flag, `observable_at`); the shutdown-events table (BLS closed, collection stopped, release delayed, window extended — the single home for shutdown coding); the party-composition table (separate file, no monthly key).\
      Exit: the panel is committed with fiscal-year keys only and no monthly expansion (bullet 14, institutional half); the 2018–19 lapse row's four fields match Stage 2's ruling and citation; no generic federal-shutdown column exists; the party table is not joinable to any monthly panel (Req 3).\
      ROUTING: writing-plans

- [ ] Stage 14: Seasonal-adjustment file store, seasonal-flag panel, birth–death covariates, and interventions table\
      Objective: Fetch and store BLS's archived seasonal-adjustment files, build the seasonal-flag panel and the birth–death covariate columns, and date the Req 14 intervention set with shutdown rows joined from Stage 13.\
      Spec: Req 3 (seasonal rows); Req 11; Req 14\
      Gap closed: Req 3 (seasonal rows); Req 11 (covariate columns); Req 14 (data half)\
      Consumes: From Stage 2: the archive inventory. From Stage 3: the release-date index, stage-label table. From Stage 4: the birth–death and reconstruction-events tables. From Stage 13: the shutdown-events table.\
      Produces: the seasonal-adjustment file store (specification, prior-adjustment, and outlier files per vintage, content-hashed, coverage scoped by the archive inventory); the seasonal-flag panel; the birth–death covariate columns (relative forecast, forecast-vs-realized; zero for government); the interventions table (the documented Req 14 set, shutdown-affected releases from Stage 13, NAICS/reconstruction flags from Stage 4); `observable_at` per value.\
      Exit: every seasonal-flag row for a vintage the archive inventory marks incomplete carries a missing flag rather than a silent absence; the file store's coverage equals the archive inventory; the government birth–death covariate is zero in every row (Req 11); the interventions table contains every Req 14 documented event with a source, and every shutdown row resolves to its four-field Stage 13 row (Req 14, data half).\
      ROUTING: writing-plans

- [ ] Stage 15: Closing-stage scale layer\
      Objective: Enter the F→S, S→T, and seasonal-factor rows of the Req 8 map — capacity factor excluded — with the sector hierarchy, the well-identified and horseshoe blocks, drift on the two permitted coefficients, and the interventions, gated by a layer-local prior predictive and SBC.\
      Spec: Req 8; Req 12; Req 13; Req 14; Req 5; Req 15; Req 18\
      Gap closed: Req 8 (closing rows); Req 12 (closing rows); Req 13 (closing-row priors); Req 14 (model half); Req 5 (government deviations)\
      Consumes: From Stage 11: comparator (2), settings, harnesses, run harness. From Stage 12: the collection-window panel, the first-stage C1 model, the macro-controls table. From Stage 14: the seasonal-flag panel, the interventions table. From Stage 4: the sample panel. From Stage 3: the accounting-decomposition table.\
      Produces: the closing-stage scale model; the design tables (standardized per transition; the capacity column absent pending Stage 16); the run harness extended with fiscal-year blocking of likelihood contributions and cross-validation folds; the three harnesses extended to the scale layer. No real-data fit is reported from this stage.\
      Exit: tests assert the Req 12 non-centered hierarchy with sum-to-zero deviations on every closing-row sector coefficient; the Req 13 prior assignment on every closing-row covariate other than the capacity factor; drift only on the two Req 14 coefficients and no break search; government's own collection and coverage deviations (Req 5); annual covariates contributing once per fiscal year (Req 8); the layer-local prior predictive satisfies the Req 15 monthly constraints; layer-local SBC recovers the closing-row scale coefficients (bullet 7, closing-row part).\
      ROUTING: writing-plans

- [ ] Stage 16: Annual-stage scale, capacity factor, and the full prior-predictive gate\
      Objective: Complete comparator (3) — the T→B and B→M rows, the capacity factor on closing and annual rows, the birth–death and shutdown interventions, the separate-covariate setting — and pass the complete Req 15 prior-predictive plan before its first conditional fit.\
      Spec: Req 8; Req 11; Req 12; Req 13; Req 14; Req 15; Req 18\
      Gap closed: Req 8 (complete); Req 11 (interventions); Req 12 (annual rows); Req 13; Req 14 (interventions); Req 15 (gate)\
      Consumes: From Stage 15: the closing-stage scale model, design tables, harnesses, run harness. From Stage 13: the institutional panel, the shutdown-events table. From Stage 14: the birth–death covariate columns, the interventions table. From Stage 4: the benchmark, QCEW revision, and sample tables. From Stage 12: the collection-window panel (RR).\
      Produces: comparator (3), complete and unfitted; the design tables completed; the prior-predictive report at `docs/prior-predictive.md`; the gated prior table (Req 15 as amended by the gate, each change a Deviation) — the table every later stage cites; the harnesses extended to annual rows and the capacity factor.\
      Exit: tests assert comparators (0)–(3) are obtainable from comparator (3) by nested settings (Req 18); one shared capacity coefficient with shrunk stage deviations on closing and annual rows, missing indicators integrated out, and no unrestricted slow slope beside it (Req 13); the Req 12 hierarchy on T→B and B→M sector coefficients; the birth–death method changes as dated interventions on annual rows only (Req 11, Req 14); party composition absent from model inputs (Req 3); the prior-predictive report shows every Req 15 constraint satisfied on complete 1979–2026 panels with the doubled-SD rerun committed, before any conditional fit of comparator (3) (bullet 6 closed); layer-local SBC recovers the annual-row coefficients and the capacity loadings (bullet 7, annual part).\
      ROUTING: writing-plans

- [ ] Stage 17: First fit of comparator (3), sampler decision, and hierarchy geometry\
      Objective: Make the first conditional fit of the full model, benchmark the sampler by block, decide the kernel per Req 17 step (5), and run the Req 12 geometry comparison on real geometry.\
      Spec: Req 17 (step 5); Req 12; Req 18\
      Gap closed: Req 17 (step 5); Req 12 (geometry comparison)\
      Consumes: From Stage 16: comparator (3), the gated prior table, design tables, harnesses, run harness. From Stage 11: the sampler benchmark record. From Stage 1: the engine determination and its revisit triggers.\
      Produces: the reported comparator-(3) fit (InferenceData, diagnostics JSON) under the kernel the decision names; the sampler benchmark record extended; the kernel decision at `docs/decisions/kernel.md`; the geometry-comparison record at `docs/decisions/geometry.md`; the decomposition table and the latent-path-vs-wedge table on comparator (3).\
      Exit: the fit meets Req 18 thresholds with rank plots for scale, capacity, and hierarchy SDs (bullet 8, comparator (3)); the kernel decision exists with its benchmark evidence, and BlackJAX appears in the pins with a 3.14 import test if and only if adopted (Req 17 step 5); the geometry record names every centered block and a test asserts centered parameterization appears only in blocks it favored (Req 12); the two tables exist on comparator (3) (bullet 11, on the reported model). If the benchmark demands a blocked kernel or trips an `engine.md` revisit trigger beyond one plan's scope, the plan's Scope Check splits it out and resume re-validates.\
      ROUTING: writing-plans

- [ ] Stage 18: Posterior predictive checks and full-panel simulation-based calibration\
      Objective: Build the Req 18 posterior-predictive suite on the reported fit and close bullet 7 with full-panel SBC.\
      Spec: Req 18\
      Gap closed: Req 18 (PPC, SBC)\
      Consumes: From Stage 17: the reported fit, the kernel decision. From Stage 16: the harnesses, the gated prior table. From Stage 3: the vintage panel, stage-label table. From Stage 1: the engine module (one-step-ahead moments, smoother).\
      Produces: the validation module (PPC half); the PPC tables; the SBC report.\
      Exit: the PPC tables cover every Req 18 statistic by stage, sector, and regime with the 2020–22 panels separate, on the reported fit; the SBC report on full synthetic panels shows calibrated rank histograms for the scale coefficients and recovers the news/noise allocation and the wedge rank (bullet 7 closed).\
      ROUTING: writing-plans

- [ ] Stage 19: Nested comparison (0)–(3)\
      Objective: Refit comparators (0)–(3) from nested settings under one harness and the gated prior table, compare them on blocked PSIS-LOO, leave-future-out, and leave-one-benchmark-year-out, and label covariate retention.\
      Spec: Req 18; Req 8\
      Gap closed: Req 18 (comparison, retention)\
      Consumes: From Stage 17: the reported fit, the kernel decision. From Stage 16: comparator (3), settings, the gated prior table, the run harness (fiscal-year blocking). From Stage 18: the validation module.\
      Produces: the comparator fits (0)–(3) refit under one harness; the comparison table (ELPD differences with SE and Pareto-k over the three schemes); the covariate-retention table — a reporting label under the fixed Req 8 map, no covariate removed, no refit; the validation module (comparison half).\
      Exit: the comparison table exists over the three schemes with Pareto-k reported and values above 0.7 flagged, and every covariate carries a retained/not-retained label from held-out scale/tail calibration (bullet 9); diagnostics JSON exists for every model entering the comparison (bullet 8, comparators); a test asserts LOO blocks are innovation/reference-month blocks nested within fiscal-year folds, so no wedge cell is split and no annual covariate leaks twelve-fold (Req 18, Req 8).\
      ROUTING: writing-plans

- [ ] Stage 20: Pseudo-real-time evaluation and leakage audit\
      Objective: Refit on expanding windows at historical release dates with every series, covariate, and the first-stage C1 model truncated to the then-observable vintage; score the four targets on the five blocks; export the predictive-scale artifact.\
      Spec: Req 18; Req 19; Req 3\
      Gap closed: Req 18 (pseudo-real-time); Req 19 (predictive-scale artifact)\
      Consumes: From Stage 17: the reported fit, the kernel decision. From Stage 16: comparator (3), settings, run harness. From Stage 18: the validation module. From Stage 3: the vintage panel, stage-label table, release-date index. From Stage 12: the collection-window panel and the first-stage C1 model. From Stages 4, 13, 14: the covariate panels with `observable_at`. From Stage 5: the operators module.\
      Produces: the pseudo-vintage truncation module (refits the first-stage C1 model per pseudo-vintage); the leakage audit log covering raw and derived columns; the score tables; the predictive-scale artifact with its as-of date.\
      Exit: the score tables exist for the five blocks, four targets, and five scoring rules (bullet 10); the audit log lists per pseudo-vintage the maximum `observable_at` of every input including derived columns and the first-stage fit window, and a test asserts none exceeds the pseudo-vintage date (bullet 10); the predictive-scale artifact exists (Req 19).\
      ROUTING: brainstorming — how roughly 280 release-date refits are made feasible (date subset, warm starts, approximations) is open.

- [ ] Stage 21: Sensitivity suite and identification checks\
      Objective: Run the sensitivity variants enumerated in Req 6, 7, 13, 14, 16, and 18 as settings of comparator (3), with the capacity overlap table and the (a)/(b)/(c) decompositions.\
      Spec: Req 16; Req 7; Req 6; Req 13; Req 14; Req 15; Req 18\
      Gap closed: Req 16; Req 13 (sensitivity half); Req 7 (sensitivities); Req 14 (sensitivities)\
      Consumes: From Stage 17: the reported fit, the kernel decision, the geometry record. From Stage 16: comparator (3), settings, the gated prior table, run harness. From Stage 9: the decomposition function. From Stage 18: the validation module.\
      Produces: the sensitivity registry (variant, setting, run, InferenceData, diagnostics); the capacity overlap table (posterior/prior overlap and likelihood profiles at the three global scales and under the separate-covariate setting, with prior sensitivity); the decomposition tables under (a), (b), (c).\
      Exit: every listed variant has InferenceData meeting thresholds or a recorded failure; the capacity overlap table exists with prior-sensitive slopes flagged unidentified (bullet 12, fitting half); the decomposition appears under (a), (b), and (c) (Req 16); the crisis-episode-removal, drift-shape, horseshoe-scale, correlation, and centered-hierarchy runs are registered.\
      ROUTING: writing-plans

- [ ] Stage 22: Non-marginal sensitivities — exact censoring and unknown change points\
      Objective: Build the reduced-panel exact interval-censoring model (Req 15) and the unknown-change-point sensitivity (Req 14) as their own model classes, registered as sensitivities.\
      Spec: Req 15; Req 14; Req 18\
      Gap closed: Req 15 (non-marginal sensitivity); Req 14 (change-point sensitivity)\
      Consumes: From Stage 16: comparator (3), the gated prior table, run harness. From Stage 17: the reported fit. From Stage 21: the sensitivity registry.\
      Produces: the censored reduced-panel model and the change-point model with their registry entries.\
      Exit: each model fits its panel meeting thresholds, with latent states among the sampled sites for the censored model; both are registered as sensitivity-only and neither alters the reported model.\
      ROUTING: brainstorming

- [ ] Stage 23: Verified literature review and bibliography\
      Objective: Complete `docs/ces-revisions-review.md` — the literature by driver, the gap table, and the annotated bibliography — verified against primary sources.\
      Spec: Req 20\
      Gap closed: Req 20 (complete)\
      Consumes: From Stage 2: the review document with its inventory sections and rulings. From Stage 5: the written link-relative determination (cited). From Stage 1: the engine determination (cited as Req 17's discharge). The drafts and the prompt. Independent of Stages 6–22; may run any time after Stage 5.\
      Produces: the finished review document.\
      Exit: the four Req 20 parts exist; every numeric claim, date, and citation carries a primary-source citation or an unverified flag and an evidence label; the bibliography labels each source's status; `uv run ruff format --check` passes (bullet 13 closed).\
      ROUTING: writing-plans

- [ ] Stage 24: Report\
      Objective: Generate the interpreted report delivering Req 19's outputs, Req 16's assumptions, and Req 13's identification statements from the Stage 17–22 artifacts, with an interpreted report artifact per reported model.\
      Spec: Req 19; Req 16; Req 13; Req 18; Req 9; Req 10\
      Gap closed: Req 19; Req 16 (statement); Req 13 (reporting); Req 18 (reporting)\
      Consumes: From Stage 17: the reported fit, the two tables on comparator (3). From Stage 18: the PPC tables, the SBC report. From Stage 19: the comparison and retention tables. From Stage 20: the score tables, the predictive-scale artifact. From Stage 21: the sensitivity registry, the capacity overlap table, the (a)/(b)/(c) tables. From Stage 22, if it ran: its registry entries. From Stage 13: the party-composition table. From Stage 23: the review document.\
      Produces: the report under `docs/` (figures and tables generated by code from the run artifacts); the interpreted report artifacts, one per reported model, in the house fixed shape.\
      Exit: the report exists and `uv run ruff format --check` passes on it; it shows the decomposition by sector, stage, and regime with intervals and the latent path against the linear wedge for every benchmark year (bullet 11 closed); capacity is presented at the three global scales and under the separate-covariate sensitivity, labeled an association, party composition only descriptive, and a test asserts no party variable is a site in any model (bullet 12 closed); an interpreted artifact exists for every model in the registry and the comparison table (bullet 8 closed); the seven Req 16 assumptions with their failure modes, the Req 13 identification sources and disclaimers, and the Req 19 outputs and can/cannot statements are present.\
      ROUTING: writing-plans

- [ ] Stage 25: X-13 vintage reproduction\
      Objective: For every vintage the archive inventory marks complete, reproduce the vintage-specific X-13 factors from the stored files and record match-to-published diagnostics.\
      Spec: Req 9 (channel (b)); Out of scope\
      Gap closed: Req 9 (channel (b), reproduction half)\
      Consumes: From Stage 2: the archive inventory. From Stage 14: the seasonal-adjustment file store. From Stage 3: the vintage panel (NSA rows), the raw file archive. From Stage 1: the stack pins, which include no X-13 wrapper.\
      Produces: the X-13 reproduction module, with any X-13 wrapper it adopts pinned under a Python 3.14 import test; the reproduced-factor table per vintage with match diagnostics and an explicit channel-(a) flag for partial-archive vintages.\
      Exit: for every archive-complete vintage a test shows the reproduced SA series matches the published SA vintage to rounding, and partial-archive vintages carry the channel-(a) flag; the table states how many vintages reproduce. If none do, resume parks Stage 26.\
      ROUTING: brainstorming

- [ ] Stage 26: Channel (b) observation and re-reported decomposition\
      Objective: Enter the reproduced adjustment as a rounding-error observation where available, swap in the archived-input seasonal covariates, and re-report the decomposition beside channel (a).\
      Spec: Req 9 (channel (b)); Req 8; Req 18\
      Gap closed: Req 9 (channel (b), model half)\
      Consumes: From Stage 25: the reproduced-factor table. From Stage 16: comparator (3), settings, the gated prior table. From Stage 17: the reported fit, the kernel decision. From Stage 9: the decomposition function. From Stage 24: the report.\
      Produces: the channel-(b) setting and fit; the re-reported decomposition with a diff against channel (a).\
      Exit: the channel-(b) fit meets thresholds; the decomposition is re-reported with channel-(b) intervals beside channel (a), stating which vintages channel (b) covers (bullet 11 re-closed).\
      ROUTING: writing-plans

- [ ] Stage 27: FY1979 institutional-panel extension\
      Objective: Extend the institutional panel to FY1979 from archived Congressional Budget Justifications and OMB tables with per-break intercepts, and refit the capacity sensitivity over the full 1979+ leg.\
      Spec: Req 3; Req 13; Out of scope\
      Gap closed: Req 3 (FY1979 reach); Req 13 (full-leg capacity)\
      Consumes: From Stage 13: the institutional panel and its conventions. From Stage 2: the data inventory (pre-2009 rows), the FTE ruling. From Stage 16: comparator (3), settings, the gated prior table. From Stage 17: the kernel decision. From Stage 21: the sensitivity registry, the capacity overlap table. From Stage 24: the report.\
      Produces: the panel extended to FY1979 with definition-break intercepts and `observable_at`; refits at the three global scales and the separate-covariate setting; the updated capacity overlap table and re-reported capacity section.\
      Exit: the extended panel passes Stage 13's fiscal-year-key and no-expansion tests (bullet 14 re-closed); each pre-2009 break carries its own intercept (Req 13); the refits meet thresholds and the capacity overlap table is re-reported (bullet 12 re-closed).\
      ROUTING: brainstorming

## Stage-spec stamp

Every stage spec's Rollout note carries this line, which writing-plans copies verbatim into the
stage plan's header:

> Roadmap: specs/ces-revisions-roadmap.md, Stage N — on plan completion, tick the stage and
> re-validate later stages against what shipped.

On completion the stamp becomes authoritative:

> Stage N: COMPLETE (YYYY-MM-DD) — implemented by plan <id> (path). Next: resume the roadmap.

## Completion

Retirement is gated behind a conformance audit of the accumulated system: re-run the gap rubric
over every numbered requirement of `specs/ces-revisions.md` with evidence per verdict
(implementing stage and plan, Deviation notes, `specs/deferred_items.md` entries), and confirm
every Verification bullet 1–14 is discharged by a ticked stage. Unmet requirements exit exactly
two ways — a new stage, or conscious deferral with a written why. On retirement, this roadmap,
the spec, the prompt, and the three research drafts move to `specs/completed/` together; the
drafts and prompt are the spec's inputs.
