# Stage 1 — Inference Stack and Kalman Engine Implementation Plan

**Status: COMPLETE (2026-09-12)** — executed via executing-plans; deferred items in specs/deferred_items.md

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via subagent-driven-development (the default) — or executing-plans when your human partner chose inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

> Roadmap: specs/ces-revisions-roadmap.md, Stage 1 — on plan completion, tick the stage and
> re-validate later stages against what shipped.

**Goal:** Discharge Req 17's two (open) items. Pin NumPyro, JAX, Dynamax, ArviZ, and Polars under Python 3.14 with an import test per package. Ship the marginalized Kalman engine: filter log likelihood, smoother, per-step contributions, and one-step-ahead moments over time-varying rows, NaN cells, and irregular annual rows. Prove it with a dense-reference oracle and a four-chain synthetic NUTS pilot, and commit the engine determination.

**Architecture:** `src/ces_revisions/kalman.py` is a hand-written, NaN-masked Kalman filter and Rauch–Tung–Striebel smoother over a `LinearGaussianSSM` NamedTuple. Every array is time-indexed and float64. `kalman_factor` adds the filter's log likelihood to a NumPyro model, so no state is ever a sampled site. Correctness is gated by a NumPy-only dense Gaussian oracle in `tests/`. Dynamax, run against the same oracle, is dev-only evidence for the decision record `docs/decisions/engine.md`.

**Tech Stack:** Python 3.14, uv, JAX 0.11.1 (float64), NumPyro 0.21.0, ArviZ 1.3.0, Polars 1.44.2 (pinned now, first used in Stage 3), Dynamax 1.0.2 (dev group), pytest, ruff.

**Source:** [`specs/ces-revisions.md`](../../ces-revisions.md) Req 17, the Rollout note, and Verification bullet 5; [`specs/ces-revisions-roadmap.md`](../../ces-revisions-roadmap.md) Stage 1.

**Retirement:** When this plan retires to `specs/plans/completed/`, `specs/ces-revisions.md` does **not** retire with it. The spec, roadmap, prompt, and research drafts retire together under the roadmap's Completion section. At completion, tick Stage 1 in the roadmap and append this authoritative stamp to the spec's Rollout note:\
`Stage 1: COMPLETE (YYYY-MM-DD) — implemented by plan 1 (specs/plans/completed/1-ces-revisions.md). Next: resume the roadmap.`

## Planning evidence (2026-09-12)

This plan's code was run end to end in a scratch clone of `0b1ad7c` before it was written. Deviations from these outcomes are signals, not noise:

- `uv add` resolved the stack on CPython 3.14.0 (macOS arm64): jax/jaxlib 0.11.1, numpy 2.5.3, numpyro 0.21.0, arviz 1.3.0 (arviz-base 1.3.0, arviz-stats 1.3.2, arviz-plots 1.3.1), polars 1.44.2, dynamax 1.0.2 (pulling tfp-nightly 0.26.0.dev20260912). Every package imported, and a float64 four-chain NUTS run worked with `chain_method="parallel"`.
- Dynamax 1.0.2 `lgssm_filter` supports time-varying arrays. It matches the dense oracle once re-indexed. But it returns a NaN log likelihood for one missing cell, a fully missing step, or an annual-only row. Its `marginal_loglik` is a scalar, and `predicted_means`/`predicted_covariances` are `None`. Its linear-Gaussian package contains no NaN or masking code path.
- The engine matched the dense oracle to about 1e-13 absolute on every field. The full suite passed: 26 fast tests and 2 slow ones, with the pilot fit taking about 12 s. `ruff format --check`, `ruff check`, and `uv sync --locked` were clean.

## Execution notes

- Execute on a feature branch or worktree (using-git-worktrees), for example `stage-1-kalman-engine`. `main` is the default branch.
- `tests/dense_reference.py` and `tests/synthetic_pilot.py` are helper modules that tests import by bare name. pytest's default `prepend` import mode puts `tests/` on `sys.path`, and ruff sorts these imports with third-party ones. Do not add `tests/__init__.py`.
- Every commit step runs `uv run ruff format` and `uv run ruff check` first, and stages files by explicit path. The repo root has an untracked `.DS_Store`, so never `git add -A`.
- Importing Dynamax emits three warnings: a `jax.core.pytype_aval_mappings` deprecation, an invalid escape sequence, and an `asyncio.iscoroutinefunction` deprecation. They are expected and recorded in the decision record. Do not filter them.
- **Stop conditions.** Stop and report rather than improvise when:
  - any `tests/test_engine_determination.py` outcome differs from this plan;
  - any package fails to resolve or import on 3.14;
  - a pilot threshold fails.

  Never loosen a tolerance, threshold, prior, or seed to make a test pass. Debug with systematic-debugging.

## Global Constraints

- **Python:** `requires-python = ">=3.14"` (`.python-version` is `3.14`). Every package must install and import under Python 3.14 via `uv add`, recorded in `pyproject.toml`/`uv.lock` (Verification bullet 5).
- **Marginalization (Req 17):** the linear-Gaussian states "are **marginalized analytically** by a differentiable Kalman filter, whose log marginal likelihood is the NumPyro likelihood term; NUTS samples only the static parameters, mixing variables, and volatility paths". No state is ever a sampled site.
- **Precision (Req 17):** "the full model in 64-bit JAX". Engine inputs must be float64. `numpyro.enable_x64()` and `numpyro.set_host_device_count(4)` run before the first JAX operation.
- **MCMC (Req 18):** "four dispersed chains, rank-normalized $`\hat R<1.01`$, bulk and tail ESS above 400 for every reported functional, zero post-warmup divergences, acceptable energy-BFMI, no systematic tree-depth saturation". This plan makes the last two numeric: energy-BFMI above 0.3 in every chain, and fewer than 1% of draws at maximum tree depth 10.
- **BlackJAX (Req 17 step 5):** "BlackJAX only after the marginalized NumPyro benchmark identifies the bottleneck". It is not pinned in this stage.
- **Dependency placement:** runtime `jax`, `numpy`, `numpyro`, `arviz`, `polars`; dev group `dynamax`. Dynamax is evidence only and is never imported from `src/`.
- **Seeds:** descriptive, `sum(map(ord, "<name>"))`, never a bare literal.
- **Tests:** MCMC tests carry the `slow` marker, and an unregistered marker is a collection error. The fast hermetic tier is `uv run pytest -m "not slow and not network"`.
- **Ruff:** the default rules plus `extend-select = ["I", "B", "UP"]`. Do not switch to `select`. `uv run ruff format` also formats Python blocks inside Markdown.
- **Markdown** (`docs/decisions/engine.md`, `CLAUDE.md`, `README.md`) stays GitHub-renderable per CLAUDE.md:
  - display math in ```` ```math ```` fences and inline math as $`…`$; never `\(…\)`, `\[…\]`, `$$…$$`, or bare `$…$`;
  - literal dollars written as `\$`;
  - real headings;
  - hard breaks as a trailing `\`;
  - pseudo-math containing `_` in code spans.
- **Scope:** Stage 1 ships:
  - the pins;
  - the engine and its tests;
  - the synthetic pilot;
  - one decision record, at `docs/decisions/engine.md` (path fixed by the roadmap).

  The run harness, InferenceData/DataTree writer, diagnostics JSON, provenance, prior-predictive harness, and any real data belong to Stage 6 and later. Do not build them here.

---

## File structure

| Path | Responsibility | Task |
|---|---|---|
| `pyproject.toml`, `uv.lock` | Stack pins: runtime and dev group | 1 |
| `tests/conftest.py` | Session numerics policy: float64 and four host devices, before any JAX op | 1 |
| `tests/test_stack.py` | One import test per pinned package; interpreter and policy checks | 1 |
| `tests/dense_reference.py` | NumPy-only dense Gaussian oracle; random time-varying models; simulator | 2 |
| `tests/test_dense_reference.py` | Oracle self-test against a hand-worked two-step model | 2 |
| `tests/test_engine_determination.py` | Executable evidence of Dynamax's behavior on masked panels | 2 |
| `src/ces_revisions/kalman.py` | `LinearGaussianSSM`, `kalman_filter`, `kalman_smoother`, `kalman_factor` | 3, 4, 5 |
| `tests/test_kalman.py` | Engine vs. oracle, fully missing step, gradient, input contract | 3, 4 |
| `tests/synthetic_pilot.py` | CES-shaped synthetic panel simulator, `pilot_ssm`, `pilot_model` | 5 |
| `tests/test_synthetic_pilot.py` | Panel structure and sampled sites (fast); four-chain fit and determinism (slow) | 5, 6 |
| `docs/decisions/engine.md` | The engine determination and the Python 3.14 record | 7 |
| `CLAUDE.md`, `README.md` | Replace the stale "no dependencies / placeholder package" statements | 7 |

---

### Task 1: Pin the inference stack on Python 3.14

**Files:**
- Modify: `pyproject.toml` and `uv.lock` (through `uv add` only)
- Create: `tests/test_stack.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Consumes: the scaffold. `tests/test_smoke.py` must keep passing.
- Produces: runtime dependencies `arviz`, `jax`, `numpy`, `numpyro`, `polars`, and dev dependency `dynamax`. Every later test runs with float64 JAX on four host CPU devices, set once in `tests/conftest.py`.

- [x] **Step 1: Write the failing import test**

Create `tests/test_stack.py`:

```python
"""The pinned stack imports on this interpreter, and the session runs float64 JAX."""

import importlib
import sys

import jax
import jax.numpy as jnp
import pytest

# One importable module per pinned distribution: runtime jax, numpy, numpyro, arviz,
# and polars; dynamax from the dev group, as engine-determination evidence only.
STACK_MODULES = [
    "jax",
    "numpy",
    "numpyro",
    "arviz",
    "polars",
    "dynamax.linear_gaussian_ssm",
]


def test_interpreter_is_python_3_14_or_newer():
    assert sys.version_info >= (3, 14)


@pytest.mark.parametrize("module", STACK_MODULES)
def test_stack_module_imports(module):
    importlib.import_module(module)


def test_session_runs_float64_jax_on_four_host_devices():
    assert jnp.zeros(1).dtype == jnp.float64
    assert jax.local_device_count() == 4
```

- [x] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_stack.py -q`\
Expected: a collection error, `ModuleNotFoundError: No module named 'jax'`.

- [x] **Step 3: Add the stack with uv**

```bash
uv add jax numpy numpyro arviz polars
```

```bash
uv add --dev dynamax
```

Expected `pyproject.toml` change (version floors may be newer if releases landed since 2026-09-12; that is fine, and Task 7 records what locked):

```diff
-dependencies = []
+dependencies = [
+    "arviz>=1.3.0",
+    "jax>=0.11.1",
+    "numpy>=2.5.3",
+    "numpyro>=0.21.0",
+    "polars>=1.44.2",
+]
@@
 dev = [
+    "dynamax>=1.0.2",
     "pytest>=9.1.1",
     "ruff>=0.16.7",
 ]
```

If any package fails to resolve or install on Python 3.14, stop and report. The roadmap's exit requires the failure and the fallback taken to be recorded, and that is a decision for your human partner.

- [x] **Step 4: Run the tests to see the numerics policy fail**

Run: `uv run pytest tests/test_stack.py -q`\
Expected: 7 passed, 1 failed. `test_session_runs_float64_jax_on_four_host_devices` fails on `assert jnp.zeros(1).dtype == jnp.float64` (the dtype is float32), because nothing has enabled x64 yet.

- [x] **Step 5: Write the session numerics policy**

Create `tests/conftest.py`:

```python
"""Session numerics policy, applied before any test module performs a JAX operation.

Float64 because Req 17 runs the Kalman recursions in 64-bit JAX; four host devices so the
synthetic pilot's four NUTS chains run in parallel. Both calls silently do nothing once a
JAX operation has run, which is why they live here rather than in a test module.
"""

import numpyro

numpyro.set_host_device_count(4)
numpyro.enable_x64()
```

- [x] **Step 6: Run the whole suite**

Run: `uv run pytest -q`\
Expected: `10 passed` (8 stack tests and 2 smoke tests), plus the three Dynamax import warnings.

> Deviation: 10 passed with 2 warnings rather than 3 once bytecode is cached. The `SyntaxWarning` from dynamax's `parallel_inference.py` fires only when that file is compiled; a cold compile shows all three.

- [x] **Step 7: Confirm the lock and lint**

Run: `uv sync --locked && uv run ruff format && uv run ruff check`\
Expected: `uv sync` reports no changes, and ruff reports `All checks passed!`.

- [x] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock tests/conftest.py tests/test_stack.py
git commit -m "Pin the inference stack on Python 3.14 with import tests"
```

> Deviation: every commit in this plan keeps the planned subject line and adds a body and a Co-Authored-By trailer, following the repository's commit convention.

---

### Task 2: Dense Gaussian oracle and Dynamax evidence

**Files:**
- Create: `tests/dense_reference.py`
- Create: `tests/test_dense_reference.py`
- Create: `tests/test_engine_determination.py`

**Interfaces:**
- Consumes: Task 1's pins and `tests/conftest.py`.
- Produces, all in `tests/dense_reference.py` and all NumPy-only:
  - **Model representation.** A model is a `dict[str, np.ndarray]` keyed by `initial_mean` `(n,)`, `initial_cov` `(n, n)`, `transition_matrix` `(T, n, n)`, `transition_offset` `(T, n)`, `transition_cov` `(T, n, n)`, `observation_matrix` `(T, p, n)`, `observation_offset` `(T, p)`, and `observation_cov` `(T, p, p)`. These are exactly Task 3's `LinearGaussianSSM` field names.
  - **Convention.** The state follows `x_t = A_t x_{t-1} + c_t + w_t` from `x_{-1} ~ N(initial_mean, initial_cov)` and is observed as `y_t = Z_t x_t + d_t + v_t`. NaN marks a missing cell.
  - `random_ssm(rng, num_steps, state_dim, obs_dim) -> dict` builds a random time-varying model.
  - `time_invariant(ssm) -> dict` returns the same model with every time-indexed array set to its step-0 value.
  - `simulate(ssm, rng) -> np.ndarray` draws a fully observed `(T, p)` panel.
  - `dense_reference(ssm, y) -> dict` returns the keys `log_likelihood` `()`, `step_log_likelihood` `(T,)`, `predicted_mean`, `predicted_cov`, `filtered_mean`, `filtered_cov`, `one_step_mean` `(T, p)`, `one_step_cov` `(T, p, p)`, `smoothed_mean`, and `smoothed_cov`.

- [x] **Step 1: Write the oracle's self-test**

The oracle is only as good as its own check. This test pins it to numbers worked by hand. Its two steps have different transitions, so it also pins the indexing convention. Create `tests/test_dense_reference.py`:

```python
"""The dense oracle agrees with a two-step scalar model worked by hand."""

import math

import numpy as np
from dense_reference import dense_reference

# x_{-1} ~ N(1, 2);  x_0 = 0.5 x_{-1} + 1 + N(0, 1);  x_1 = 2 x_0 - 1 + N(0, 3);
# y_t = 1.5 x_t + 0.5 + N(0, 0.25), with y_0 missing and y_1 = 4.
# By hand: x_0 ~ N(1.5, 1.5), x_1 ~ N(2, 9), y_0 ~ N(2.75, 3.625), y_1 ~ N(3.5, 20.5),
# and Cov(x_0, y_1) = 1.5 * 2 * 1.5 = 4.5. The two steps have different transitions, so
# these numbers pin the convention that A_t maps x_{t-1} to x_t.
HAND_MODEL = {
    "initial_mean": np.array([1.0]),
    "initial_cov": np.array([[2.0]]),
    "transition_matrix": np.array([[[0.5]], [[2.0]]]),
    "transition_offset": np.array([[1.0], [-1.0]]),
    "transition_cov": np.array([[[1.0]], [[3.0]]]),
    "observation_matrix": np.array([[[1.5]], [[1.5]]]),
    "observation_offset": np.array([[0.5], [0.5]]),
    "observation_cov": np.array([[[0.25]], [[0.25]]]),
}
HAND_PANEL = np.array([[np.nan], [4.0]])


def test_dense_reference_matches_a_hand_worked_two_step_model():
    reference = dense_reference(HAND_MODEL, HAND_PANEL)
    gain = 9.0 * 1.5 / 20.5  # Kalman gain for x_1 given y_1
    log_likelihood = -0.5 * (math.log(2 * math.pi) + math.log(20.5) + 0.5**2 / 20.5)

    np.testing.assert_allclose(reference["log_likelihood"], log_likelihood)
    np.testing.assert_allclose(reference["step_log_likelihood"], [0.0, log_likelihood])
    np.testing.assert_allclose(reference["predicted_mean"][:, 0], [1.5, 2.0])
    np.testing.assert_allclose(reference["predicted_cov"][:, 0, 0], [1.5, 9.0])
    np.testing.assert_allclose(
        reference["filtered_mean"][:, 0], [1.5, 2.0 + gain * 0.5]
    )
    np.testing.assert_allclose(
        reference["filtered_cov"][:, 0, 0], [1.5, 9.0 - gain * 13.5]
    )
    np.testing.assert_allclose(reference["one_step_mean"][:, 0], [2.75, 3.5])
    np.testing.assert_allclose(reference["one_step_cov"][:, 0, 0], [3.625, 20.5])
    np.testing.assert_allclose(
        reference["smoothed_mean"][:, 0], [1.5 + 4.5 / 20.5 * 0.5, 2.0 + gain * 0.5]
    )
    np.testing.assert_allclose(
        reference["smoothed_cov"][:, 0, 0], [1.5 - 4.5**2 / 20.5, 9.0 - gain * 13.5]
    )
```

- [x] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_dense_reference.py -q`\
Expected: a collection error, `ModuleNotFoundError: No module named 'dense_reference'`.

- [x] **Step 3: Write the oracle**

The oracle stacks all states and cells into one joint normal, then answers each question by conditioning. It never runs a recursion, so it shares no failure mode with the engine. Create `tests/dense_reference.py`:

```python
"""Dense Gaussian oracle for the Kalman engine.

Stacks every state and observation cell of a linear-Gaussian state-space model into one
joint normal, and answers each filtering and smoothing question by direct Gaussian
conditioning. It shares no code with ``ces_revisions.kalman`` and is quadratic in memory,
so use it only on tiny models. It follows the engine's conventions: the state evolves as
``x_t = A_t x_{t-1} + c_t + w_t`` from ``x_{-1} ~ N(m, P)`` and is observed as
``y_t = Z_t x_t + d_t + v_t``, and NaN marks a missing cell. Models are dicts of NumPy
arrays keyed by the ``LinearGaussianSSM`` field names.
"""

import numpy as np


def _random_spd(rng, dim):
    root = rng.normal(size=(dim, dim))
    return root @ root.T / dim + 0.5 * np.eye(dim)


def random_ssm(rng, num_steps, state_dim, obs_dim):
    """A time-varying model whose arrays are drawn independently at every step."""
    return {
        "initial_mean": rng.normal(size=state_dim),
        "initial_cov": _random_spd(rng, state_dim),
        "transition_matrix": 0.9 * np.eye(state_dim)
        + 0.2 * rng.normal(size=(num_steps, state_dim, state_dim)),
        "transition_offset": rng.normal(size=(num_steps, state_dim)),
        "transition_cov": np.stack(
            [_random_spd(rng, state_dim) for _ in range(num_steps)]
        ),
        "observation_matrix": rng.normal(size=(num_steps, obs_dim, state_dim)),
        "observation_offset": rng.normal(size=(num_steps, obs_dim)),
        "observation_cov": np.stack(
            [_random_spd(rng, obs_dim) for _ in range(num_steps)]
        ),
    }


def time_invariant(ssm):
    """The same model with every time-indexed array replaced by its step-0 value."""
    return {
        name: np.broadcast_to(value[0], value.shape).copy()
        if name.startswith(("transition", "observation"))
        else value
        for name, value in ssm.items()
    }


def simulate(ssm, rng):
    """Draw one fully observed panel of shape (T, p) from the model."""
    state = rng.multivariate_normal(ssm["initial_mean"], ssm["initial_cov"])
    rows = []
    for a, c, q, z, d, r in zip(
        ssm["transition_matrix"],
        ssm["transition_offset"],
        ssm["transition_cov"],
        ssm["observation_matrix"],
        ssm["observation_offset"],
        ssm["observation_cov"],
        strict=True,
    ):
        state = a @ state + c + rng.multivariate_normal(np.zeros(len(c)), q)
        rows.append(z @ state + d + rng.multivariate_normal(np.zeros(len(d)), r))
    return np.stack(rows)


def _block_diag(blocks):
    """Place (T, a, b) blocks along the diagonal of a (T*a, T*b) matrix."""
    num_steps, rows, cols = blocks.shape
    out = np.zeros((num_steps * rows, num_steps * cols))
    for t, block in enumerate(blocks):
        out[t * rows : (t + 1) * rows, t * cols : (t + 1) * cols] = block
    return out


def _joint_moments(ssm):
    """Moments of the stacked states X (T*n) and cells Y (T*p), and Cov(X, Y)."""
    a, c, q = ssm["transition_matrix"], ssm["transition_offset"], ssm["transition_cov"]
    num_steps, n = c.shape
    mean_x = np.zeros((num_steps, n))
    cov_x = np.zeros((num_steps * n, num_steps * n))
    prev_mean, prev_cov = ssm["initial_mean"], ssm["initial_cov"]
    prev_cross = np.zeros((n, 0))  # Cov(x_{t-1}, x_0..x_{t-1})
    for t in range(num_steps):
        here = slice(t * n, (t + 1) * n)
        mean_x[t] = a[t] @ prev_mean + c[t]
        cross = a[t] @ prev_cross  # Cov(x_t, x_0..x_{t-1})
        cov_x[here, : t * n] = cross
        cov_x[: t * n, here] = cross.T
        cov_x[here, here] = a[t] @ prev_cov @ a[t].T + q[t]
        prev_mean, prev_cov = mean_x[t], cov_x[here, here]
        prev_cross = cov_x[here, : (t + 1) * n]
    big_z = _block_diag(ssm["observation_matrix"])
    mean_y = big_z @ mean_x.ravel() + ssm["observation_offset"].ravel()
    cov_y = big_z @ cov_x @ big_z.T + _block_diag(ssm["observation_cov"])
    return mean_x.ravel(), cov_x, mean_y, cov_y, cov_x @ big_z.T


def _log_normal(value, mean, cov):
    if value.size == 0:
        return 0.0
    chol = np.linalg.cholesky(cov)
    white = np.linalg.solve(chol, value - mean)
    log_det = 2 * np.log(np.diag(chol)).sum()
    return -0.5 * (value.size * np.log(2 * np.pi) + log_det + white @ white)


def _condition(mean_a, cov_aa, cov_ab, mean_b, cov_bb, value_b):
    """Mean and covariance of a given b = value_b, for jointly Gaussian (a, b)."""
    if value_b.size == 0:
        return mean_a, cov_aa
    weights = np.linalg.solve(cov_bb, cov_ab.T).T
    return mean_a + weights @ (value_b - mean_b), cov_aa - weights @ cov_ab.T


def dense_reference(ssm, y):
    """Every FilterResult and SmootherResult field, by dense Gaussian conditioning."""
    mean_x, cov_x, mean_y, cov_y, cov_xy = _joint_moments(ssm)
    num_steps, p = y.shape
    n = len(ssm["initial_mean"])
    flat = y.ravel()
    observed = ~np.isnan(flat)
    everything = np.flatnonzero(observed)

    def observed_before(t):
        return np.flatnonzero(observed[: t * p])

    def log_likelihood_before(t):
        idx = observed_before(t)
        return _log_normal(flat[idx], mean_y[idx], cov_y[np.ix_(idx, idx)])

    def state_given(t, idx):
        here = slice(t * n, (t + 1) * n)
        return _condition(
            mean_x[here],
            cov_x[here, here],
            cov_xy[here, idx],
            mean_y[idx],
            cov_y[np.ix_(idx, idx)],
            flat[idx],
        )

    steps = []
    for t in range(num_steps):
        cells = slice(t * p, (t + 1) * p)
        before = observed_before(t)
        one_step = _condition(
            mean_y[cells],
            cov_y[cells, cells],
            cov_y[cells, before],
            mean_y[before],
            cov_y[np.ix_(before, before)],
            flat[before],
        )
        steps.append(
            (
                log_likelihood_before(t + 1) - log_likelihood_before(t),
                *state_given(t, before),
                *state_given(t, observed_before(t + 1)),
                *one_step,
                *state_given(t, everything),
            )
        )
    names = (
        "step_log_likelihood",
        "predicted_mean",
        "predicted_cov",
        "filtered_mean",
        "filtered_cov",
        "one_step_mean",
        "one_step_cov",
        "smoothed_mean",
        "smoothed_cov",
    )
    columns = (np.array(column) for column in zip(*steps, strict=True))
    return {
        "log_likelihood": log_likelihood_before(num_steps),
        **dict(zip(names, columns, strict=True)),
    }
```

- [x] **Step 4: Run the self-test to verify it passes**

Run: `uv run pytest tests/test_dense_reference.py -q`\
Expected: `1 passed`.

- [x] **Step 5: Write the Dynamax evidence tests**

These tests characterize a third-party library. They pin what Dynamax does, not what we want, so they have no red phase. Create `tests/test_engine_determination.py`:

```python
"""Evidence for docs/decisions/engine.md: Dynamax's linear-Gaussian SSM on our inputs.

These tests pin observed Dynamax 1.0.2 behavior, not desired behavior. If one fails after a
Dynamax upgrade, the premise of the engine determination has changed: revisit the record.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from dense_reference import dense_reference, random_ssm, simulate
from dynamax.linear_gaussian_ssm.inference import lgssm_filter, make_lgssm_params

SEED = sum(map(ord, "ces-revisions-engine-determination"))
NUM_STEPS, STATE_DIM, OBS_DIM = 15, 3, 4
MISSING_PATTERNS = {
    "one missing cell": (3, 1),
    "fully missing step": (5, slice(None)),
    "irregular annual row": (np.arange(NUM_STEPS) % 12 != 2, OBS_DIM - 1),
}


def _to_dynamax(ssm):
    """Re-index a model for Dynamax.

    Dynamax puts its prior on the first state and its F_t maps z_t to z_{t+1}; ours puts the
    prior on x_{-1} and A_t maps x_{t-1} to x_t.
    """
    a, c, q = ssm["transition_matrix"], ssm["transition_offset"], ssm["transition_cov"]

    def shifted(array):  # Dynamax never uses the transition at its final step
        return jnp.asarray(np.concatenate([array[1:], array[-1:]]))

    return make_lgssm_params(
        initial_mean=jnp.asarray(a[0] @ ssm["initial_mean"] + c[0]),
        initial_cov=jnp.asarray(a[0] @ ssm["initial_cov"] @ a[0].T + q[0]),
        dynamics_weights=shifted(a),
        dynamics_cov=shifted(q),
        emissions_weights=jnp.asarray(ssm["observation_matrix"]),
        emissions_cov=jnp.asarray(ssm["observation_cov"]),
        dynamics_bias=shifted(c),
        emissions_bias=jnp.asarray(ssm["observation_offset"]),
    )


@pytest.fixture
def model_and_panel():
    rng = np.random.default_rng(SEED)
    ssm = random_ssm(rng, NUM_STEPS, STATE_DIM, OBS_DIM)
    return ssm, simulate(ssm, rng)


def test_dynamax_matches_the_dense_reference_on_time_varying_rows(model_and_panel):
    ssm, y = model_and_panel
    posterior = lgssm_filter(_to_dynamax(ssm), jnp.asarray(y))
    expected = dense_reference(ssm, y)["log_likelihood"]
    np.testing.assert_allclose(posterior.marginal_loglik, expected, rtol=1e-8)


@pytest.mark.parametrize("cells", MISSING_PATTERNS.values(), ids=list(MISSING_PATTERNS))
def test_dynamax_log_likelihood_is_nan_when_any_cell_is_missing(model_and_panel, cells):
    ssm, y = model_and_panel
    y = y.copy()
    y[cells] = np.nan
    posterior = lgssm_filter(_to_dynamax(ssm), jnp.asarray(y))
    assert np.isnan(posterior.marginal_loglik)
    assert np.isfinite(dense_reference(ssm, y)["log_likelihood"])  # the quantity exists


def test_dynamax_filter_has_no_step_terms_or_predicted_moments(model_and_panel):
    ssm, y = model_and_panel
    posterior = lgssm_filter(_to_dynamax(ssm), jnp.asarray(y))
    assert np.shape(posterior.marginal_loglik) == ()
    assert posterior.predicted_means is None
    assert posterior.predicted_covariances is None
```

- [x] **Step 6: Run the evidence tests**

Run: `uv run pytest tests/test_engine_determination.py -v`\
Expected: `5 passed`. The five tests are the time-varying match, the three missing-cell patterns, and the scalar-log-likelihood check.

**If any test fails, stop.** Report which outcome differs before starting Task 3. The engine determination rests on these facts, and a different Dynamax outcome reopens the decision for your human partner.

- [x] **Step 7: Lint and run the fast tier**

Run: `uv run ruff format && uv run ruff check && uv run pytest -m "not slow and not network" -q`\
Expected: `All checks passed!` and `16 passed`.

- [x] **Step 8: Commit**

```bash
git add tests/dense_reference.py tests/test_dense_reference.py tests/test_engine_determination.py
git commit -m "Add a dense Gaussian oracle and pin Dynamax's behavior on masked panels"
```

---

### Task 3: Kalman filter with NaN-cell masking

**Files:**
- Create: `src/ces_revisions/kalman.py`
- Create: `tests/test_kalman.py`

**Interfaces:**
- Consumes: from Task 2, `dense_reference`, `random_ssm`, `simulate`, and `time_invariant` in `tests/dense_reference.py`, using the dict keys and conventions listed there.
- Produces:
  - **`LinearGaussianSSM(NamedTuple)`** with fields `initial_mean` `(n,)`, `initial_cov` `(n, n)`, `transition_matrix` `(T, n, n)`, `transition_offset` `(T, n)`, `transition_cov` `(T, n, n)`, `observation_matrix` `(T, p, n)`, `observation_offset` `(T, p)`, and `observation_cov` `(T, p, p)`. Every array is float64 and has the same convention as the oracle. Index `t` describes the transition into `x_t` and the observation of `x_t`.
  - **`FilterResult(NamedTuple)`** with fields `log_likelihood` `()`, `step_log_likelihood` `(T,)`, `predicted_mean` `(T, n)`, `predicted_cov` `(T, n, n)`, `filtered_mean` `(T, n)`, `filtered_cov` `(T, n, n)`, `one_step_mean` `(T, p)`, and `one_step_cov` `(T, p, p)`.
  - **`kalman_filter(ssm: LinearGaussianSSM, y: jax.Array) -> FilterResult`**, where `y` is `(T, p)` with NaN for missing cells. It raises `ValueError` naming the offending field on a shape mismatch, and `TypeError` mentioning `float64` on any non-float64 input.
  - **Why the extra outputs.** The per-step contributions and one-step-ahead moments exist so Stage 6 can build the log likelihood grouped by reference month and benchmark year: `az.from_numpyro` produces no `log_likelihood` group for a `numpyro.factor` model. Stage 18 uses them for predictive checks. This task builds no harness.

Design notes for the implementer and reviewer:
- **Masking.** A missing cell gets a zero emission row, a zero innovation, and a unit diagonal in the innovation covariance. That covariance is then block diagonal in (observed block, identity), so its log-determinant and quadratic form are exactly the observed block's. The `2*pi` term counts observed cells only, which makes the log density exact, with no constant to correct. The NaN fixtures in the oracle test are what catch a mistake here. A pilot fit would not, because a constant cancels in the NUTS acceptance ratio.
- **Finite gradients.** `y` is zero-filled where NaN before any arithmetic, so NaN never reaches a gradient.
- **Stability.** The Joseph-form update keeps filtered covariances positive semidefinite over long panels with precise anchors.
- **Import side effects.** `_LOG_2PI` is a Python float. A module-level JAX constant would run a JAX operation at import, before callers can enable float64.

- [x] **Step 1: Write the failing tests**

Create `tests/test_kalman.py`:

```python
"""The Kalman engine against the dense oracle, plus its gradient and input contract."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dense_reference import dense_reference, random_ssm, simulate, time_invariant

from ces_revisions.kalman import FilterResult, LinearGaussianSSM, kalman_filter

SEED = sum(map(ord, "ces-revisions-kalman-engine"))
NUM_STEPS, STATE_DIM, OBS_DIM = 15, 3, 4
MISSING_STEP = 5
ANNUAL_ROW = OBS_DIM - 1  # observed only at steps 2 and 14, like a March benchmark
TOLERANCE = {"rtol": 1e-8, "atol": 1e-10}


def _with_gaps(y):
    """Scattered missing cells, a missing step, a ragged edge, and an annual-only row."""
    y = y.copy()
    y[3, 1] = np.nan
    y[9, 0] = np.nan
    y[MISSING_STEP, :] = np.nan
    y[-1, :2] = np.nan
    y[np.arange(len(y)) % 12 != 2, ANNUAL_ROW] = np.nan
    return y


def _random_case(*, time_varying):
    rng = np.random.default_rng(SEED)
    ssm = random_ssm(rng, NUM_STEPS, STATE_DIM, OBS_DIM)
    if not time_varying:
        ssm = time_invariant(ssm)
        return ssm, simulate(ssm, rng)
    return ssm, _with_gaps(simulate(ssm, rng))


def _engine(ssm):
    return LinearGaussianSSM(
        **{name: jnp.asarray(value) for name, value in ssm.items()}
    )


@pytest.fixture(
    params=[False, True], ids=["time-invariant, fully observed", "time-varying, gaps"]
)
def case(request):
    return _random_case(time_varying=request.param)


@pytest.fixture
def gappy_case():
    return _random_case(time_varying=True)


def test_filter_matches_the_dense_reference(case):
    ssm, y = case
    filtered = kalman_filter(_engine(ssm), jnp.asarray(y))
    reference = dense_reference(ssm, y)
    for field in FilterResult._fields:
        np.testing.assert_allclose(
            getattr(filtered, field), reference[field], **TOLERANCE, err_msg=field
        )


def test_fully_missing_step_adds_nothing_and_leaves_the_state_unconditioned(gappy_case):
    ssm, y = gappy_case
    filtered = kalman_filter(_engine(ssm), jnp.asarray(y))
    assert filtered.step_log_likelihood[MISSING_STEP] == 0.0
    np.testing.assert_array_equal(
        filtered.filtered_mean[MISSING_STEP], filtered.predicted_mean[MISSING_STEP]
    )
    np.testing.assert_array_equal(
        filtered.filtered_cov[MISSING_STEP], filtered.predicted_cov[MISSING_STEP]
    )


def test_log_likelihood_gradient_matches_a_dense_finite_difference(gappy_case):
    ssm, y = gappy_case
    engine_ssm = _engine(ssm)

    def engine_log_likelihood(scale):
        scaled = engine_ssm._replace(
            transition_cov=engine_ssm.transition_cov * scale,
            observation_cov=engine_ssm.observation_cov * scale,
        )
        return kalman_filter(scaled, jnp.asarray(y)).log_likelihood

    def dense_log_likelihood(scale):
        scaled = {
            **ssm,
            "transition_cov": ssm["transition_cov"] * scale,
            "observation_cov": ssm["observation_cov"] * scale,
        }
        return dense_reference(scaled, y)["log_likelihood"]

    step = 1e-6
    finite_difference = (
        dense_log_likelihood(1 + step) - dense_log_likelihood(1 - step)
    ) / (2 * step)
    gradient = jax.grad(engine_log_likelihood)(1.0)
    assert np.isfinite(gradient)
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-5)


def test_rejects_arrays_whose_shapes_disagree_with_the_panel(gappy_case):
    ssm, y = gappy_case
    short = {**ssm, "observation_cov": ssm["observation_cov"][:-1]}
    with pytest.raises(ValueError, match="observation_cov"):
        kalman_filter(_engine(short), jnp.asarray(y))


def test_rejects_float32_inputs(gappy_case):
    ssm, y = gappy_case
    with pytest.raises(TypeError, match="float64"):
        kalman_filter(_engine(ssm), jnp.asarray(y, dtype=jnp.float32))
```

> Deviation: after the whole-branch review, `tests/test_kalman.py` gained `test_missing_cells_observation_noise_reaches_only_one_step_cov`, and `test_rejects_float32_inputs` was parametrized over `y`, `initial_mean`, and `observation_cov` so it covers the model fields as well as the panel.

- [x] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_kalman.py -q`\
Expected: a collection error, `ModuleNotFoundError: No module named 'ces_revisions.kalman'`.

- [x] **Step 3: Write the filter**

Create `src/ces_revisions/kalman.py`:

```python
"""Marginalized Kalman engine: the linear-Gaussian state-space likelihood for NumPyro.

The state evolves as ``x_t = A_t x_{t-1} + c_t + w_t`` with ``w_t ~ N(0, Q_t)``, starting
from ``x_{-1} ~ N(m, P)``, and is observed as ``y_t = Z_t x_t + d_t + v_t`` with
``v_t ~ N(0, R_t)``. Every time-indexed array describes step ``t``: the transition into
``x_t`` and the observation of ``x_t``, so a time-varying process scale sits at the step it
shocks. A NaN in ``y`` marks a missing cell; an irregular row, such as an annual benchmark,
is a row whose other cells are NaN.

The filter's log likelihood integrates the states out, so a NumPyro model that adds it as a
factor never samples a state. All arrays must be float64 (call ``numpyro.enable_x64()``
before the first JAX operation), and every covariance must be finite, including in the rows
and columns of missing cells.
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve

# A Python float, not a JAX constant: a module-level JAX array would run a JAX
# operation at import time, before the caller could enable float64.
_LOG_2PI = math.log(2 * math.pi)


class LinearGaussianSSM(NamedTuple):
    """A time-varying linear-Gaussian state-space model: T steps, n states, p cells."""

    initial_mean: jax.Array  # (n,) mean of x_{-1}
    initial_cov: jax.Array  # (n, n)
    transition_matrix: jax.Array  # (T, n, n) A_t
    transition_offset: jax.Array  # (T, n) c_t
    transition_cov: jax.Array  # (T, n, n) Q_t
    observation_matrix: jax.Array  # (T, p, n) Z_t
    observation_offset: jax.Array  # (T, p) d_t
    observation_cov: jax.Array  # (T, p, p) R_t


class FilterResult(NamedTuple):
    """Kalman filter output, conditioning on the observed (non-NaN) cells only."""

    log_likelihood: jax.Array  # () log p(y)
    step_log_likelihood: jax.Array  # (T,) log p(y_t | y_<t); 0 if none observed
    predicted_mean: jax.Array  # (T, n) E[x_t | y_<t]
    predicted_cov: jax.Array  # (T, n, n)
    filtered_mean: jax.Array  # (T, n) E[x_t | y_<=t]
    filtered_cov: jax.Array  # (T, n, n)
    one_step_mean: jax.Array  # (T, p) E[y_t | y_<t], missing cells included
    one_step_cov: jax.Array  # (T, p, p) Cov[y_t | y_<t]


def kalman_filter(ssm: LinearGaussianSSM, y: jax.Array) -> FilterResult:
    """Filter the (T, p) panel y through ssm, conditioning only on its non-NaN cells."""
    ssm = LinearGaussianSSM(*(jnp.asarray(array) for array in ssm))
    y = jnp.asarray(y)
    _validate(ssm, y)
    observed = ~jnp.isnan(y)
    state_eye = jnp.eye(ssm.initial_mean.shape[0])

    def step(carry, inputs):
        mean, cov = carry
        a, c, q, z, d, r, y_t, observed_t = inputs
        predicted_mean = a @ mean + c
        predicted_cov = _symmetrize(a @ cov @ a.T + q)
        one_step_mean = z @ predicted_mean + d
        one_step_cov = _symmetrize(z @ predicted_cov @ z.T + r)

        # Condition on the observed cells only. A missing cell gets a zero emission row,
        # a zero innovation, and a unit diagonal, so the innovation covariance is block
        # diagonal in (observed block, identity). Its log-determinant and quadratic form
        # are exactly the observed block's, and the 2*pi term counts only observed cells.
        both_observed = observed_t[:, None] & observed_t[None, :]
        missing_diagonal = jnp.diag(jnp.where(observed_t, 0.0, 1.0))
        innovation_cov = jnp.where(both_observed, one_step_cov, 0.0) + missing_diagonal
        innovation = jnp.where(observed_t, y_t - one_step_mean, 0.0)
        observed_z = jnp.where(observed_t[:, None], z, 0.0)
        chol = jnp.linalg.cholesky(innovation_cov)
        gain = cho_solve((chol, True), observed_z @ predicted_cov).T
        filtered_mean = predicted_mean + gain @ innovation
        # Joseph form keeps the filtered covariance positive semidefinite.
        residual_map = state_eye - gain @ observed_z
        filtered_cov = _symmetrize(
            residual_map @ predicted_cov @ residual_map.T + gain @ r @ gain.T
        )

        num_observed = observed_t.sum()
        log_det = 2.0 * jnp.log(jnp.diag(chol)).sum()
        mahalanobis = innovation @ cho_solve((chol, True), innovation)
        step_log_likelihood = -0.5 * (num_observed * _LOG_2PI + log_det + mahalanobis)
        outputs = (
            step_log_likelihood,
            predicted_mean,
            predicted_cov,
            filtered_mean,
            filtered_cov,
            one_step_mean,
            one_step_cov,
        )
        return (filtered_mean, filtered_cov), outputs

    # Zero-filling keeps NaN out of the arithmetic, so gradients stay finite.
    y_filled = jnp.where(observed, y, 0.0)
    inputs = (
        ssm.transition_matrix,
        ssm.transition_offset,
        ssm.transition_cov,
        ssm.observation_matrix,
        ssm.observation_offset,
        ssm.observation_cov,
        y_filled,
        observed,
    )
    _, outputs = jax.lax.scan(step, (ssm.initial_mean, ssm.initial_cov), inputs)
    return FilterResult(outputs[0].sum(), *outputs)


def _validate(ssm: LinearGaussianSSM, y: jax.Array) -> None:
    if y.ndim != 2 or ssm.initial_mean.ndim != 1:
        raise ValueError(
            f"y must be (T, p) and initial_mean (n,); "
            f"got {y.shape} and {ssm.initial_mean.shape}"
        )
    (num_steps, obs_dim), state_dim = y.shape, ssm.initial_mean.shape[0]
    expected = {
        "initial_cov": (state_dim, state_dim),
        "transition_matrix": (num_steps, state_dim, state_dim),
        "transition_offset": (num_steps, state_dim),
        "transition_cov": (num_steps, state_dim, state_dim),
        "observation_matrix": (num_steps, obs_dim, state_dim),
        "observation_offset": (num_steps, obs_dim),
        "observation_cov": (num_steps, obs_dim, obs_dim),
    }
    for name, shape in expected.items():
        actual = getattr(ssm, name).shape
        if actual != shape:
            raise ValueError(
                f"{name} has shape {actual}; expected {shape} "
                f"for T={num_steps}, n={state_dim}, p={obs_dim}"
            )
    for name, array in (("y", y), *ssm._asdict().items()):
        if array.dtype != jnp.float64:
            raise TypeError(
                f"{name} is {array.dtype}; the Kalman engine needs float64. "
                "Call numpyro.enable_x64() before the first JAX operation."
            )


def _symmetrize(matrix: jax.Array) -> jax.Array:
    return 0.5 * (matrix + matrix.T)
```

> Deviation (plan-level defect, found by the whole-branch review): the Joseph noise term used the full `R_t`, so a non-finite entry in a missing cell's row or column made the filtered and smoothed covariances and the gradients NaN, silently at the last step. The shipped code masks `R_t` to the observed block. The Finite gradients design note's rationale was also wrong: masking the innovation already keeps gradients finite, so zero-filling is defense in depth, and the code comment now says so.

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_kalman.py -v`\
Expected: `6 passed`:
- both `test_filter_matches_the_dense_reference` cases;
- the fully missing step;
- the gradient;
- both input-contract tests.

- [x] **Step 5: Lint and run the fast tier**

Run: `uv run ruff format && uv run ruff check && uv run pytest -m "not slow and not network" -q`\
Expected: `All checks passed!` and `22 passed`.

- [x] **Step 6: Commit**

```bash
git add src/ces_revisions/kalman.py tests/test_kalman.py
git commit -m "Add the NaN-masked Kalman filter, checked against the dense oracle"
```

---

### Task 4: Rauch–Tung–Striebel smoother

**Files:**
- Modify: `src/ces_revisions/kalman.py`, adding `SmootherResult` and `kalman_smoother`
- Modify: `tests/test_kalman.py`

**Interfaces:**
- Consumes: Task 3's `LinearGaussianSSM`, `FilterResult`, and `kalman_filter`.
- Produces:
  - `SmootherResult(NamedTuple)` with fields `smoothed_mean` `(T, n)` and `smoothed_cov` `(T, n, n)`.
  - `kalman_smoother(ssm: LinearGaussianSSM, filtered: FilterResult) -> SmootherResult`, which smooths an existing filter pass of the same model, so callers never filter twice. It requires every predicted covariance after step 0 to be positive definite.
  - Stage 8 consumes this as the probabilistic wedge between March anchors, and Stage 9 for the seasonal decomposition.

- [x] **Step 1: Write the failing test**

In `tests/test_kalman.py`, replace the `ces_revisions.kalman` import with:

```python
from ces_revisions.kalman import (
    FilterResult,
    LinearGaussianSSM,
    SmootherResult,
    kalman_filter,
    kalman_smoother,
)
```

Then insert this test immediately after `test_filter_matches_the_dense_reference`:

```python
def test_smoother_matches_the_dense_reference(case):
    ssm, y = case
    engine_ssm = _engine(ssm)
    smoothed = kalman_smoother(engine_ssm, kalman_filter(engine_ssm, jnp.asarray(y)))
    reference = dense_reference(ssm, y)
    for field in SmootherResult._fields:
        np.testing.assert_allclose(
            getattr(smoothed, field), reference[field], **TOLERANCE, err_msg=field
        )
```

- [x] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_kalman.py -q`\
Expected: a collection error, `ImportError: cannot import name 'SmootherResult' from 'ces_revisions.kalman'`.

- [x] **Step 3: Write the smoother**

In `src/ces_revisions/kalman.py`, insert this class immediately after the `FilterResult` class:

```python
class SmootherResult(NamedTuple):
    """Rauch–Tung–Striebel smoother output."""

    smoothed_mean: jax.Array  # (T, n) E[x_t | y]
    smoothed_cov: jax.Array  # (T, n, n)
```

Then insert this function immediately after `kalman_filter`, before `_validate`:

```python
def kalman_smoother(ssm: LinearGaussianSSM, filtered: FilterResult) -> SmootherResult:
    """Smooth a filter pass of the same model backward in time (Rauch–Tung–Striebel).

    Every predicted covariance after the first step must be positive definite.
    """
    transition_matrix = jnp.asarray(ssm.transition_matrix)

    def step(carry, inputs):
        next_smoothed_mean, next_smoothed_cov = carry
        mean, cov, next_a, next_predicted_mean, next_predicted_cov = inputs
        chol = jnp.linalg.cholesky(next_predicted_cov)
        gain = cho_solve((chol, True), next_a @ cov).T
        smoothed_mean = mean + gain @ (next_smoothed_mean - next_predicted_mean)
        smoothed_cov = _symmetrize(
            cov + gain @ (next_smoothed_cov - next_predicted_cov) @ gain.T
        )
        return (smoothed_mean, smoothed_cov), (smoothed_mean, smoothed_cov)

    last = (filtered.filtered_mean[-1], filtered.filtered_cov[-1])
    inputs = (
        filtered.filtered_mean[:-1],
        filtered.filtered_cov[:-1],
        transition_matrix[1:],
        filtered.predicted_mean[1:],
        filtered.predicted_cov[1:],
    )
    _, (means, covs) = jax.lax.scan(step, last, inputs, reverse=True)
    return SmootherResult(
        jnp.concatenate([means, last[0][None]]), jnp.concatenate([covs, last[1][None]])
    )
```

The smoother gain at step `t` pairs the filtered moments at `t` with `transition_matrix[t + 1]` and the predicted moments at `t + 1`. That is why the scan inputs are offset by one.

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_kalman.py -v`\
Expected: `8 passed`, with both `test_smoother_matches_the_dense_reference` cases added.

- [x] **Step 5: Lint and run the fast tier**

Run: `uv run ruff format && uv run ruff check && uv run pytest -m "not slow and not network" -q`\
Expected: `All checks passed!` and `24 passed`.

- [x] **Step 6: Commit**

```bash
git add src/ces_revisions/kalman.py tests/test_kalman.py
git commit -m "Add the Rauch-Tung-Striebel smoother, checked against the dense oracle"
```

---

### Task 5: NumPyro factor term and the synthetic pilot model

**Files:**
- Modify: `src/ces_revisions/kalman.py`, adding `import numpyro` and `kalman_factor`
- Create: `tests/synthetic_pilot.py`
- Create: `tests/test_synthetic_pilot.py` (fast tests only)

**Interfaces:**
- Consumes: Tasks 3–4 `LinearGaussianSSM` and `kalman_filter`.
- Produces:
  - `kalman_factor(name: str, ssm: LinearGaussianSSM, y: jax.Array) -> FilterResult`. It registers `numpyro.factor(name, filtered.log_likelihood)` and returns the filter pass.
  - `tests/synthetic_pilot.py`, which exports:
    - constants `NUM_STEPS = 120`, `MARCH = 2`, `LEVEL_REGIME_START = 60`, `CRISIS_STEPS`, `LAPSE_STEP = 100`, `CLOSING_ROWS`, and `BENCHMARK_ROW = 3`;
    - `TRUTH` and `PRIOR_MEDIANS`, both `dict[str, float]` over the six scale names `sigma_level`, `sigma_slope`, `sigma_first`, `sigma_second`, `sigma_third`, and `sigma_benchmark`;
    - `simulate_panel(truth, seed) -> np.ndarray` of shape `(120, 4)`;
    - `pilot_ssm(scales) -> LinearGaussianSSM`;
    - `pilot_model(y)`, whose only unobserved sites are the six scalar scales and whose factor site is `"panel"`.

Pilot design: the pilot mirrors CES structure at toy scale, and the fixture test in Step 2 guards that structure against being simplified away.
- **State.** A local linear trend with state `(level_t, slope_t, level_{t-1})`.
- **Closing rows.** Three rows measure changes before step 60 and levels from step 60 on, a time-varying emission row like the 1979 aggregate leg.
- **Benchmark row.** A benchmark row is observed only in March, an irregular annual row.
- **Missingness.** The panel has random missing cells, a ragged real-time edge, and a fully missing lapse month.
- **Crisis window.** A known window multiplies the level-innovation and closing-noise scales, exercising time-varying process and observation covariances.
- **Independence.** The simulator is an explicit recursion that is independent of `pilot_ssm`, so a wrong matrix in `pilot_ssm` shows up as failed recovery.

- [x] **Step 1: Write the pilot model module**

Create `tests/synthetic_pilot.py`:

```python
"""A synthetic pilot for the Kalman engine: CES-shaped structure and six sampled scales.

A local linear trend, in thousands of jobs, is observed by three closing rows (first,
second, and third print) that measure month-over-month changes before LEVEL_REGIME_START
and levels from then on (a time-varying emission row), and by an annual benchmark row
observed only in March. The panel has scattered missing cells, a ragged real-time edge, and
one fully missing month, and a known crisis window scales the level innovations and the
closing-row noise. Only the six scales are sampled; ``kalman_factor`` integrates out the
states.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from ces_revisions.kalman import LinearGaussianSSM, kalman_factor

NUM_STEPS = 120  # ten years of reference months; t = 0 is a January
MARCH = 2  # benchmark months are the steps with t % 12 == MARCH
LEVEL_REGIME_START = 60  # closing rows observe changes before this step, levels after
CRISIS_STEPS = range(86, 90)  # a known window standing in for March–June 2020
CRISIS_PROCESS_MULTIPLIER = 10.0  # on the level-innovation SD inside the window
CRISIS_NOISE_MULTIPLIER = 3.0  # on the closing-row noise SD inside the window
LAPSE_STEP = 100  # a month with no release at all
MISSING_CELL_RATE = 0.05  # share of closing-row cells dropped at random
INITIAL_SD = (50.0, 5.0, 50.0)  # zero-mean prior SDs of the state at t = -1
CLOSING_ROWS = ("sigma_first", "sigma_second", "sigma_third")
BENCHMARK_ROW = len(CLOSING_ROWS)

TRUTH = {
    "sigma_level": 20.0,
    "sigma_slope": 2.0,
    "sigma_first": 30.0,
    "sigma_second": 20.0,
    "sigma_third": 12.0,
    "sigma_benchmark": 8.0,
}
# Broad LogNormal(log median, 1) priors, not centered on TRUTH; one SD is a factor of e.
PRIOR_MEDIANS = {
    "sigma_level": 15.0,
    "sigma_slope": 3.0,
    "sigma_first": 15.0,
    "sigma_second": 15.0,
    "sigma_third": 15.0,
    "sigma_benchmark": 15.0,
}


def _in_crisis(steps):
    return (steps >= CRISIS_STEPS.start) & (steps < CRISIS_STEPS.stop)


def simulate_panel(truth, seed):
    """Draw the (NUM_STEPS, 4) panel by explicit recursion, independently of pilot_ssm."""
    rng = np.random.default_rng(seed)
    steps = np.arange(NUM_STEPS)
    crisis = _in_crisis(steps)
    level_sd = truth["sigma_level"] * np.where(crisis, CRISIS_PROCESS_MULTIPLIER, 1.0)
    level_prev = rng.normal(0.0, INITIAL_SD[0])
    slope_prev = rng.normal(0.0, INITIAL_SD[1])
    levels, changes = np.empty(NUM_STEPS), np.empty(NUM_STEPS)
    for t in steps:
        level = level_prev + slope_prev + rng.normal(0.0, level_sd[t])
        slope = slope_prev + rng.normal(0.0, truth["sigma_slope"])
        levels[t], changes[t] = level, level - level_prev
        level_prev, slope_prev = level, slope

    y = np.full((NUM_STEPS, BENCHMARK_ROW + 1), np.nan)
    closing_target = np.where(steps < LEVEL_REGIME_START, changes, levels)
    noise_scale = np.where(crisis, CRISIS_NOISE_MULTIPLIER, 1.0)
    for row, name in enumerate(CLOSING_ROWS):
        y[:, row] = closing_target + rng.normal(0.0, truth[name] * noise_scale)
    march = steps % 12 == MARCH
    benchmark_noise = rng.normal(0.0, truth["sigma_benchmark"], march.sum())
    y[march, BENCHMARK_ROW] = levels[march] + benchmark_noise

    closing = y[:, :BENCHMARK_ROW]  # a view, so these writes land in y
    closing[rng.random(closing.shape) < MISSING_CELL_RATE] = np.nan
    closing[-1, 1:] = np.nan  # ragged edge: the latest month has only its first print
    closing[-2, 2] = np.nan  # and the month before it has no third print yet
    y[LAPSE_STEP] = np.nan
    return y


def pilot_ssm(scales):
    """The pilot as a LinearGaussianSSM over the state (level_t, slope_t, level_{t-1})."""
    steps = jnp.arange(NUM_STEPS)
    crisis = _in_crisis(steps)
    transition = jnp.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    process_sd = jnp.stack(
        [
            scales["sigma_level"] * jnp.where(crisis, CRISIS_PROCESS_MULTIPLIER, 1.0),
            jnp.broadcast_to(scales["sigma_slope"], (NUM_STEPS,)),
            jnp.zeros(NUM_STEPS),  # the lagged level is carried exactly
        ],
        axis=1,
    )

    level_row = jnp.array([1.0, 0.0, 0.0])
    change_row = jnp.array([1.0, 0.0, -1.0])
    measures_changes = (steps < LEVEL_REGIME_START)[:, None]
    closing_row = jnp.where(measures_changes, change_row, level_row)
    benchmark_row = jnp.broadcast_to(level_row, (NUM_STEPS, 3))
    observation_rows = [closing_row] * len(CLOSING_ROWS) + [benchmark_row]
    noise_scale = jnp.where(crisis, CRISIS_NOISE_MULTIPLIER, 1.0)
    observation_sd = jnp.stack(
        [
            *(scales[name] * noise_scale for name in CLOSING_ROWS),
            jnp.broadcast_to(scales["sigma_benchmark"], (NUM_STEPS,)),
        ],
        axis=1,
    )
    return LinearGaussianSSM(
        initial_mean=jnp.zeros(3),
        initial_cov=jnp.diag(jnp.square(jnp.array(INITIAL_SD))),
        transition_matrix=jnp.broadcast_to(transition, (NUM_STEPS, 3, 3)),
        transition_offset=jnp.zeros((NUM_STEPS, 3)),
        transition_cov=jax.vmap(jnp.diag)(process_sd**2),
        observation_matrix=jnp.stack(observation_rows, axis=1),
        observation_offset=jnp.zeros((NUM_STEPS, BENCHMARK_ROW + 1)),
        observation_cov=jax.vmap(jnp.diag)(observation_sd**2),
    )


def pilot_model(y):
    """Sample the six scales; add the panel's marginal log likelihood as site "panel"."""
    scales = {
        name: numpyro.sample(name, dist.LogNormal(math.log(median), 1.0))
        for name, median in PRIOR_MEDIANS.items()
    }
    kalman_factor("panel", pilot_ssm(scales), y)
```

> Deviation (plan-level claim, no code change): the whole-branch review found that the Independence bullet above overstates the recovery check. A four-SD bound catches a gross error in `pilot_ssm`, such as swapped change and level rows, but probably not a crisis multiplier applied to the variance instead of the SD. The dense oracle, not the pilot, is the engine's correctness gate.

- [x] **Step 2: Write the fast pilot tests**

The second test is the marginalization-site test the roadmap's exit names: no latent state among the sampled sites. Create `tests/test_synthetic_pilot.py`:

```python
"""Tests of the synthetic pilot for the Kalman engine."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from synthetic_pilot import (
    BENCHMARK_ROW,
    LAPSE_STEP,
    LEVEL_REGIME_START,
    MARCH,
    NUM_STEPS,
    PRIOR_MEDIANS,
    TRUTH,
    pilot_model,
    pilot_ssm,
    simulate_panel,
)

from ces_revisions.kalman import kalman_filter

SEED = sum(map(ord, "ces-revisions-stage-1-synthetic-pilot"))


@pytest.fixture(scope="module")
def panel():
    return jnp.asarray(simulate_panel(TRUTH, seed=SEED))


def test_panel_has_time_varying_rows_nan_cells_and_an_irregular_annual_row(panel):
    y = np.asarray(panel)
    steps = np.arange(NUM_STEPS)
    # Closing cells away from the ragged edge and the lapse month.
    interior_closing = np.delete(y[:-2, :BENCHMARK_ROW], LAPSE_STEP, axis=0)
    observation_matrix = np.asarray(pilot_ssm(TRUTH).observation_matrix)

    assert np.array_equal(~np.isnan(y[:, BENCHMARK_ROW]), steps % 12 == MARCH)
    assert np.isnan(y[LAPSE_STEP]).all()
    assert np.isnan(y[-1, 1:BENCHMARK_ROW]).all()
    assert np.isnan(interior_closing).any()
    assert not np.array_equal(
        observation_matrix[LEVEL_REGIME_START - 1],
        observation_matrix[LEVEL_REGIME_START],
    )


def test_pilot_samples_only_static_scales_and_factors_the_marginal_likelihood(panel):
    seeded = handlers.seed(pilot_model, jax.random.PRNGKey(SEED))
    trace = handlers.trace(seeded).get_trace(panel)
    sampled = {
        name: site
        for name, site in trace.items()
        if site["type"] == "sample" and not site["is_observed"]
    }
    scales = {name: site["value"] for name, site in sampled.items()}

    assert set(sampled) == set(PRIOR_MEDIANS)
    assert all(np.shape(value) == () for value in scales.values())
    assert isinstance(trace["panel"]["fn"], dist.Unit)
    np.testing.assert_allclose(
        trace["panel"]["fn"].log_factor,
        kalman_filter(pilot_ssm(scales), panel).log_likelihood,
    )
```

- [x] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/test_synthetic_pilot.py -q`\
Expected: a collection error, `ImportError: cannot import name 'kalman_factor' from 'ces_revisions.kalman'`.

- [x] **Step 4: Write the factor term**

In `src/ces_revisions/kalman.py`, add `import numpyro` to the imports, so the import block reads:

```python
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpyro
from jax.scipy.linalg import cho_solve
```

Then insert this function immediately after `kalman_smoother`, before `_validate`:

```python
def kalman_factor(name: str, ssm: LinearGaussianSSM, y: jax.Array) -> FilterResult:
    """Add log p(y), with the states integrated out, as the NumPyro factor site ``name``.

    Returns the filter pass, so a caller can record per-step contributions or moments.
    """
    filtered = kalman_filter(ssm, y)
    numpyro.factor(name, filtered.log_likelihood)
    return filtered
```

- [x] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_synthetic_pilot.py -v`\
Expected: `2 passed`.

- [x] **Step 6: Lint and run the fast tier**

Run: `uv run ruff format && uv run ruff check && uv run pytest -m "not slow and not network" -q`\
Expected: `All checks passed!` and `26 passed`.

- [x] **Step 7: Commit**

```bash
git add src/ces_revisions/kalman.py tests/synthetic_pilot.py tests/test_synthetic_pilot.py
git commit -m "Add kalman_factor and a synthetic pilot that samples only static scales"
```

---

### Task 6: Synthetic pilot fit under Req 18's MCMC thresholds

**Files:**
- Modify: `tests/test_synthetic_pilot.py`

**Interfaces:**
- Consumes: Task 5's `pilot_model`, `simulate_panel`, `TRUTH`, `PRIOR_MEDIANS`, and the module-scoped `panel` fixture.
- Produces: the slow acceptance evidence for Verification bullet 5, cited by Task 7's decision record. The pilot passes on the determined engine with time-varying rows, NaN cells, irregular annual rows, four chains, and zero divergences.

This is an acceptance test of a model that already exists, so it has no red phase. The planning spike's values (see Task 7 Step 1) clear every threshold by a wide margin. If an assertion fails, stop and debug. Do not loosen a threshold, prior, draw count, or seed.

- [x] **Step 1: Add the imports, thresholds, and slow tests**

In `tests/test_synthetic_pilot.py`, add `import arviz as az` as the first import. Add `from numpyro.infer import MCMC, NUTS` immediately after `from numpyro import handlers`. The import block then begins:

```python
import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
```

Replace the `SEED = ...` line with:

```python
SEED = sum(map(ord, "ces-revisions-stage-1-synthetic-pilot"))
# Req 18 MCMC thresholds, with energy-BFMI and tree-depth saturation made numeric.
NUM_CHAINS = 4
NUM_DRAWS = 1000
RHAT_MAX = 1.01
ESS_MIN = 400
BFMI_MIN = 0.3
MAX_TREE_DEPTH = 10
SATURATED_SHARE_MAX = 0.01
RECOVERY_SDS = 4  # the truth must lie within this many posterior SDs of the mean
```

Append these two tests at the end of the file:

```python
@pytest.mark.slow
def test_pilot_fit_meets_mcmc_thresholds_and_recovers_the_truth(panel):
    mcmc = MCMC(
        NUTS(pilot_model, max_tree_depth=MAX_TREE_DEPTH),
        num_warmup=NUM_DRAWS,
        num_samples=NUM_DRAWS,
        num_chains=NUM_CHAINS,
        chain_method="parallel",
        progress_bar=False,
    )
    fields = ("diverging", "energy", "num_steps")
    mcmc.run(jax.random.PRNGKey(SEED), panel, extra_fields=fields)
    # Diagnose through ArviZ's DataTree so R-hat and ESS use ArviZ's defaults: on raw
    # arrays, tail ESS needs an explicit quantile and does not reproduce the default.
    idata = az.from_numpyro(mcmc)
    posterior, stats = idata["posterior"], idata["sample_stats"]
    rhat = az.rhat(idata)
    bulk_ess = az.ess(idata, method="bulk")
    tail_ess = az.ess(idata, method="tail")

    assert set(posterior.data_vars) == set(PRIOR_MEDIANS)  # no latent state is sampled
    assert int(stats["diverging"].sum()) == 0
    assert np.all(np.asarray(az.bfmi(idata)["energy"]) > BFMI_MIN)
    saturated = stats["n_steps"] >= 2**MAX_TREE_DEPTH - 1
    assert float(saturated.mean()) < SATURATED_SHARE_MAX
    for name in PRIOR_MEDIANS:
        draws = np.asarray(posterior[name])
        assert draws.shape == (NUM_CHAINS, NUM_DRAWS)
        assert float(rhat[name]) < RHAT_MAX, name
        assert float(bulk_ess[name]) > ESS_MIN, name
        assert float(tail_ess[name]) > ESS_MIN, name
        assert abs(draws.mean() - TRUTH[name]) <= RECOVERY_SDS * draws.std(), name


@pytest.mark.slow
def test_pilot_draws_are_identical_under_a_fixed_seed(panel):
    def draws():
        mcmc = MCMC(
            NUTS(pilot_model),
            num_warmup=50,
            num_samples=50,
            num_chains=2,
            chain_method="parallel",
            progress_bar=False,
        )
        mcmc.run(jax.random.PRNGKey(SEED), panel)
        return mcmc.get_samples()

    first, second = draws(), draws()
    for name in PRIOR_MEDIANS:
        np.testing.assert_array_equal(first[name], second[name], err_msg=name)
```

Two API facts behind this code, both verified on arviz 1.3.0 and numpyro 0.21.0:
- NumPyro's extra field `num_steps` appears in `az.from_numpyro`'s `sample_stats` as `n_steps`, with the leapfrog count `2**depth - 1` at saturation.
- `az.ess(array, method="tail")` on a raw array raises `TypeError: ... missing 1 required positional argument: 'prob'`, which is why every diagnostic goes through the DataTree.

- [x] **Step 2: Run the slow tier**

Run: `uv run pytest -m slow -v`\
Expected: `2 passed, 26 deselected`. The fit takes about 12 s and the determinism check about 3 s on an Apple M4 Max; the first run includes JIT compilation.

- [x] **Step 3: Confirm the fast tier is unchanged**

Run: `uv run ruff format && uv run ruff check && uv run pytest -m "not slow and not network" -q`\
Expected: `All checks passed!` and `26 passed, 2 deselected`.

- [x] **Step 4: Commit**

```bash
git add tests/test_synthetic_pilot.py
git commit -m "Fit the synthetic pilot with four NUTS chains under Req 18's thresholds"
```

---

### Task 7: Engine decision record and documentation

**Files:**
- Create: `docs/decisions/engine.md`
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: the evidence from Tasks 1–6: the locked versions, `tests/test_engine_determination.py`, `tests/test_kalman.py`, and `tests/test_synthetic_pilot.py`.
- Produces: `docs/decisions/engine.md`, the engine determination later stages cite (roadmap Stages 6, 10, and 17).

- [x] **Step 1: Collect the evidence the record quotes**

Run each and compare with the planning values.

```bash
uv tree --depth 1
```

Expected runtime and dev entries: `arviz v1.3.0`, `jax v0.11.1`, `numpy v2.5.3`, `numpyro v0.21.0`, `polars v1.44.2`, `dynamax v1.0.2 (group: dev)`.

```bash
uv pip list 2>/dev/null | grep -E "^(jaxlib|arviz-base|arviz-stats|arviz-plots|tfp-nightly) "
```

Expected: `arviz-base 1.3.0`, `arviz-plots 1.3.1`, `arviz-stats 1.3.2`, `jaxlib 0.11.1`, `tfp-nightly 0.26.0.dev20260912`.

```bash
grep -rn -i -E "isnan|mask" .venv/lib/python3.14/site-packages/dynamax/linear_gaussian_ssm --include='*.py' | grep -v _test.py
```

Expected: no output.

```bash
PYTHONPATH=tests uv run python - <<'EOF' 2>/dev/null
import numpyro

numpyro.set_host_device_count(4)
numpyro.enable_x64()

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC, NUTS
from synthetic_pilot import PRIOR_MEDIANS, TRUTH, pilot_model, simulate_panel

seed = sum(map(ord, "ces-revisions-stage-1-synthetic-pilot"))
panel = jnp.asarray(simulate_panel(TRUTH, seed=seed))
mcmc = MCMC(NUTS(pilot_model, max_tree_depth=10), num_warmup=1000, num_samples=1000,
            num_chains=4, chain_method="parallel", progress_bar=False)
mcmc.run(jax.random.PRNGKey(seed), panel, extra_fields=("diverging", "energy", "num_steps"))
idata = az.from_numpyro(mcmc)
post, stats = idata["posterior"], idata["sample_stats"]
rhat, bulk, tail = az.rhat(idata), az.ess(idata, method="bulk"), az.ess(idata, method="tail")
print("divergences", int(stats["diverging"].sum()),
      "bfmi", np.round(np.asarray(az.bfmi(idata)["energy"]), 2),
      "max n_steps", int(stats["n_steps"].max()))
for name in PRIOR_MEDIANS:
    draws = np.asarray(post[name])
    z = abs(draws.mean() - TRUTH[name]) / draws.std()
    print(f"{name:16s} rhat {float(rhat[name]):.4f} bulk {float(bulk[name]):5.0f} "
          f"tail {float(tail[name]):5.0f} |z| {z:.2f}")
EOF
```

Expected (planning spike, Apple M4 Max):

```text
divergences 0 bfmi [1.01 0.96 1.05 1.16] max n_steps 15
sigma_level      rhat 1.0008 bulk  5025 tail  3161 |z| 1.51
sigma_slope      rhat 0.9996 bulk  5286 tail  3174 |z| 1.24
sigma_first      rhat 1.0008 bulk  5316 tail  3029 |z| 1.15
sigma_second     rhat 1.0005 bulk  5147 tail  3067 |z| 0.66
sigma_third      rhat 1.0011 bulk  5001 tail  3337 |z| 0.29
sigma_benchmark  rhat 1.0002 bulk  5117 tail  2580 |z| 0.37
```

If any version or diagnostic differs, use the values you observe in Step 2's table and Consequences bullet, and note the difference as a `> Deviation:` under this step.

- [x] **Step 2: Write the decision record**

This is the repository's first `docs/` directory, so create it first:

```bash
mkdir -p docs/decisions
```

Then create `docs/decisions/engine.md` with the content below. If you execute on a date other than 2026-09-12, set **Date** to that day.

````markdown
# Use a hand-written masked Kalman filter as the state-space engine

- **Status:** Accepted
- **Date:** 2026-09-12
- **Deciders:** Lowell Mason
- **Blast radius:** the `ces_revisions` package, and every roadmap stage that marginalizes the linear-Gaussian states (Stages 6–22 and 26)

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
````

> Deviation: the committed record differs from this text. Its blast radius lists Stages 6–11, 15–21, 26, and 27, and Stage 22's change-point model, and its per-step cost is `O(n^3 + p^3)`; both corrections came from the whole-branch review, which superseded an execution-time edit that had only added Stage 27. After the review it also gives the observed oracle agreement (1.1e-13), the oracle's `atol`, the gradient check's direction, the missing-cell covariance contract, the `SyntaxWarning` caveat, and the pilot's asserted thresholds separately from one run's observed values, and it describes the pre-masked Dynamax alternative more exactly.

- [x] **Step 3: Update CLAUDE.md**

In `CLAUDE.md`, make these four edits.

Replace the sentence

> The Python package is still the `uv init --package` scaffold; the substance so far lives in `specs/`.

with

> The Python package so far holds the marginalized Kalman engine (`src/ces_revisions/kalman.py`); the design lives in `specs/` (the spec `specs/ces-revisions.md`, staged by `specs/ces-revisions-roadmap.md`) and decision records in `docs/decisions/`.

Replace the bullet

```markdown
- Python 3.14 (`.python-version`; `requires-python = ">=3.14"`). No runtime dependencies yet.
```

with

```markdown
- Python 3.14 (`.python-version`; `requires-python = ">=3.14"`). Runtime dependencies are JAX, NumPy, NumPyro, ArviZ 1.x (`az.from_numpyro` returns an xarray `DataTree`), and Polars; Dynamax is a dev-only dependency kept as evidence for `docs/decisions/engine.md`.
```

Replace the bullet

```markdown
- The design in `specs/` targets NumPyro (NUTS), Dynamax, BlackJAX, ArviZ, and Polars. None are installed; confirm each supports Python 3.14 when adding it.
```

with

```markdown
- `src/ces_revisions/kalman.py` is the hand-written, NaN-masked Kalman filter and smoother chosen in `docs/decisions/engine.md`. Models add its log likelihood with `kalman_factor`, so states are never sampled sites. All JAX work is float64: `numpyro.enable_x64()` (and `numpyro.set_host_device_count`) must run before the first JAX operation, which `tests/conftest.py` does for the test session. BlackJAX waits for the sampler benchmark (roadmap Stage 17); confirm Python 3.14 support for any package you add.
```

In the Commands code block, add this line after the `uv run pytest -m "not slow and not network"` line:

```bash
uv run pytest -m slow                           # the synthetic pilot's four-chain NUTS fit (~15 s)
```

- [x] **Step 4: Update README.md**

In `README.md`, make these three edits.

Replace

> Early stage: the literature review and research design are in progress, and the Python package is a placeholder.

with

> Early stage: the research design is specified and staged, and the Python package holds the marginalized Kalman engine that later model stages build on.

Replace the `text` tree block's contents with:

```text
specs/
  ces-revisions.md                    design spec
  ces-revisions-roadmap.md            staged implementation roadmap
  ces-revisions-prompt.md             research brief: scope, required outputs, design requirements
  ces-revisions-research-chatgpt.md   literature review, gap analysis, and design draft
  ces-revisions-research-claude.md    literature review, gap analysis, and design draft
  ces-revisions-research-gemini.md    literature review, gap analysis, and design draft
  plans/                              implementation plans, one per roadmap stage
docs/decisions/                       decision records (engine.md: the Kalman engine)
src/ces_revisions/                    Python package (src layout); kalman.py is the state-space engine
```

Replace

> The planned modeling stack is NumPyro, Dynamax, BlackJAX, ArviZ, and Polars; none are dependencies yet.

with

> The modeling stack is JAX, NumPyro, ArviZ, and Polars on Python 3.14. The state-space engine is hand-written rather than taken from Dynamax, for the reasons recorded in [`docs/decisions/engine.md`](docs/decisions/engine.md).

- [x] **Step 5: Verify the stage exit**

Run:

```bash
uv sync --locked && uv run ruff format --check && uv run ruff check && uv run pytest -q
```

Expected: the lockfile is unchanged, the formatter reports every file already formatted, lint prints `All checks passed!`, and pytest reports `28 passed`.

Then check each roadmap Stage 1 exit item against its evidence:
- **Pins and imports.** `uv sync` on 3.14 with the pins recorded, and every import test passing: Task 1 and `tests/test_stack.py`.
- **No failed package.** No package failed on 3.14; the decision record's table records this.
- **Synthetic pilot.** The pilot passes with time-varying rows, NaN cells, irregular annual rows, four chains, and zero divergences: `tests/test_synthetic_pilot.py` (slow).
- **Dense reference.** The dense-reference tests match the likelihood and smoother moments to tolerance: `tests/test_kalman.py`.
- **No sampled state.** No latent state appears among the sampled sites: `test_pilot_samples_only_static_scales_and_factors_the_marginal_likelihood`, and the slow fit's site assertion.
- **Decision record.** The engine determination is committed: `docs/decisions/engine.md`.

> Deviation: the stage exit passed as planned with 28 tests. After the review fixes the suite has 32 (two poisoning cases and two more float64-guard cases), all passing.

- [x] **Step 6: Commit**

```bash
git add docs/decisions/engine.md CLAUDE.md README.md
git commit -m "Record the Kalman engine determination and update project docs"
```
