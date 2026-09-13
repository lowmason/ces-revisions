# Use a hand-written masked Kalman filter as the state-space engine

- **Status:** Accepted
- **Date:** 2026-09-12
- **Deciders:** Lowell Mason
- **Blast radius:** the `ces_revisions` package, and every roadmap stage that marginalizes the linear-Gaussian states (Stages 6–22, 26, and 27)

## Context

Req 17 of [`specs/ces-revisions.md`](../../specs/ces-revisions.md) marginalizes the employment, trend, seasonal, and common-factor states analytically. It uses a differentiable Kalman filter whose log marginal likelihood is the NumPyro likelihood term. Req 17 leaves two matters open, to be resolved by verification rather than argument. First, does each of NumPyro, JAX, Dynamax, ArviZ, and Polars support Python 3.14? Second, does Dynamax's linear-Gaussian SSM support the time-varying emission rows, missing cells, and irregular annual observations the model needs, or is a hand-written Kalman `scan` required? Req 20 adds that a fact about a package's behavior is settled by running the package.

The engine must accept:
- time-varying transition and observation matrices and offsets;
- per-cell time-varying observation and process covariances;
- NaN cells;
- irregular rows. An annual benchmark row is NaN outside March.

It must expose:
- the smoother;
- per-time-step log-likelihood contributions, which Stage 6 groups by reference month and benchmark year;
- one-step-ahead moments.

### Python 3.14

The stack was resolved with `uv add` on CPython 3.14.0 (macOS arm64) and is imported by `tests/test_stack.py`:

| Package | Locked version | Group | Result on Python 3.14 |
|---|---|---|---|
| jax, jaxlib | 0.11.1 | runtime | Imports; float64 NUTS runs on four host devices |
| numpy | 2.5.3 | runtime | Imports |
| numpyro | 0.21.0 | runtime | Imports |
| arviz (arviz-base 1.3.0, arviz-stats 1.3.2, arviz-plots 1.3.1) | 1.3.0 | runtime | Imports; `az.from_numpyro` returns an xarray `DataTree`, the ArviZ 1.x successor to `InferenceData` |
| polars | 1.44.2 | runtime | Imports |
| dynamax (with tfp-nightly 0.26.0.dev20260912) | 1.0.2 | dev | Imports with three warnings: `jax.core.pytype_aval_mappings` is deprecated (via tfp-nightly), `parallel_inference.py` has an invalid escape sequence, and `asyncio.iscoroutinefunction` is slated for removal in Python 3.16 (via fastcore) |

No package failed on Python 3.14, so no fallback was taken. BlackJAX is not pinned: Req 17 step (5) defers it until the marginalized NumPyro benchmark (roadmap Stage 17).

### Dynamax on this project's inputs

`tests/test_engine_determination.py` runs Dynamax's `lgssm_filter` on a 15-step model with three states and four cells. It compares the results against a NumPy-only dense Gaussian oracle, `tests/dense_reference.py`:

- **Time-varying arrays work.** Time-varying transition, observation, offset, and covariance arrays are supported. With no missing cell, Dynamax matches the oracle to `rtol=1e-8` once its convention is re-indexed. Dynamax puts its prior on the first state, and its `F_t` maps `z_t` to `z_{t+1}`.
- **Missing cells break it.** One missing cell, one fully missing step, or an annual-only row each makes the log likelihood NaN. Dynamax's linear-Gaussian package has no NaN or masking code path.
- **Outputs are incomplete.** The filter returns the log likelihood only as a scalar, and returns `predicted_means` and `predicted_covariances` as `None`. It provides no per-step contributions and no one-step-ahead moments.

## Decision

We will marginalize the linear-Gaussian states with the hand-written Kalman filter and Rauch–Tung–Striebel smoother in `src/ces_revisions/kalman.py`, entered into NumPyro models through `kalman_factor`. Dynamax stays in the dev dependency group, only as the evidence in `tests/test_engine_determination.py`.

## Consequences

- **Positive:**
  - One masking path handles NaN cells, fully missing steps, and irregular rows, and the log density is exact over the observed cells. The filter and smoother match the dense oracle in `tests/test_kalman.py` at tolerance `rtol=1e-8`; the planning run observed agreement near 1e-13. The log-likelihood gradient matches a finite difference with NaN cells present.
  - The engine returns per-step contributions and one-step-ahead moments for Stage 6's grouped log likelihood and Stage 18's predictive checks.
  - The synthetic pilot in `tests/test_synthetic_pilot.py` covers time-varying emission rows, NaN cells, a ragged real-time edge, a missing month, and a March-only benchmark row. It samples six static scales with four chains and zero divergences. R-hat is at most 1.0011, bulk ESS at least 5001, and tail ESS at least 2580. Energy-BFMI runs from 0.96 to 1.16, and no draw used more than 15 leapfrog steps. Every true scale is recovered within 1.6 posterior SDs.
  - No nightly TensorFlow Probability build enters the runtime dependencies.
  - Arrays are indexed by the step they describe, so a crisis multiplier on process noise sits at the months it shocks.
- **Negative:**
  - The project owns about 200 lines of numerical code (masking, Joseph-form update, smoother) and the tests that keep it honest.
  - The filter is dense. Each step costs `O(n^3)`, with full `(T, n, n)` and `(T, p, p)` covariance stacks, which grows expensive as the full model approaches roughly 150 states over several hundred months.
  - There is no parallel-in-time (associative scan) variant like Dynamax's `parallel_inference.py`.
  - The smoother requires every predicted covariance after the first step to be positive definite.
- **Neutral / follow-on:**
  - If Stage 10 adopts Req 17 step (4)'s finite-mixture fallback, it extends this engine with its own dense-reference test.
  - Stage 17's sampler benchmark measures the engine's cost.

## Alternatives considered

- **Dynamax `lgssm_filter` and `lgssm_smoother` as they are.** Rejected: any missing cell makes the log likelihood NaN, and the filter exposes neither per-step contributions nor one-step-ahead moments. `tests/test_engine_determination.py` verifies all three facts.
- **Dynamax with pre-masked inputs.** This option would zero the observation rows of missing cells in a time-varying `H_t`, give those cells a unit diagonal in `R_t`, zero-fill `y`, and add back `0.5 log(2π)` for each masked cell. Dynamax accepts the `(T, ·, ·)` arrays this needs, so the scalar log likelihood and the smoother are within reach. Per-step contributions and one-step-ahead moments would still require re-implementing the recursion. The option also adds a hand-maintained constant correction, and puts tfp-nightly and a fastcore web-server dependency chain into the runtime environment. It was not built.
- **Sampling the states with NUTS through a non-centered `scan`.** Rejected by Req 17 and the house state-space rule. The states are conditionally linear-Gaussian, and hundreds of autocorrelated states per month are the geometry NUTS handles worst.

## Trade-offs & reversibility

This is a two-way door. Models reach the engine only through `LinearGaussianSSM`, `kalman_filter`, `kalman_smoother`, and `kalman_factor`, so replacing the engine is a local change, and the dense-reference tests are the parity gate a replacement must pass.

Revisit this decision if either trigger occurs:
- a test in `tests/test_engine_determination.py` fails after a Dynamax upgrade, which would mean per-cell missing data or per-step outputs may have arrived;
- Stage 17's benchmark attributes sampling time to the dense filter. Candidate remedies are sequential per-cell updates, a square-root form, or a parallel scan.
