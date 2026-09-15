# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Purpose

Research project on the **magnitude** (scale, not mean bias) of U.S. CES payroll revisions across four stages — first → second print, second → third, third → annual benchmark, and post-benchmark wedge-back — plus a Bayesian state-space design in which collection-interval, seasonal-adjustment, sample, BLS-funding, and staffing covariates enter the revision *variance*. The Python package so far holds the marginalized Kalman engine (`src/ces_revisions/kalman.py`); the design lives in `specs/` (the spec `specs/ces-revisions.md`, staged by `specs/ces-revisions-roadmap.md`), decision records in `docs/decisions/`, and the Req 20 written finding in `docs/ces-revisions-review.md`, with its evidence in `docs/inventory/`.

## Specs

- `specs/ces-revisions-prompt.md` — the research brief: required sections, the five candidate drivers, and the design requirements. Treat it as the statement of scope.
- `specs/ces-revisions-research-{chatgpt,Codex,gemini}.md` — three independent AI-generated responses to that brief. They are unverified drafts: figures, dates, and citations have not been checked against primary BLS sources, and the documents disagree with each other (e.g. the gemini draft puts the post-pandemic first-closing rate near 40%, while the chatgpt and Codex drafts give 60.4% for 2024 — collection and response rates are easy to conflate). Verify a claim against primary sources before building on it. Drafts are replaced wholesale when regenerated; re-normalize a replacement to the conventions below before committing it.

Markdown in `specs/` is kept GitHub-renderable. Freshly pasted AI output usually arrives with `$…$`/`$$…$$` or `\(…\)`/`\[…\]` math and needs converting. When adding or editing:

- Put display math in ```` ```math ```` fences and inline math in `` $`…`$ ``. Do not use `\(…\)`, `\[…\]`, `$$…$$`, or bare `$…$`: GitHub ignores the backslash forms, and Markdown escapes/emphasis can corrupt TeX inside bare dollars (`\%` loses its backslash; a lone `=` or `-` line in unfenced math becomes a setext heading).
- Escape literal dollar amounts as `\$` (e.g. `\$544.3 million`, `(\$m)`); an unescaped pair on one line is read as math, and a `|` inside it splits table cells.
- Use real headings (`##`, `###`), not bold-only lines, and write hard line breaks as a trailing `\` rather than two trailing spaces, which whitespace cleanup silently deletes.
- Put non-TeX pseudo-math containing `_` (e.g. `θ_{s,t}`) in code spans so underscores are not read as emphasis.

## Layout and tooling

- `src/ces_revisions/` is an installable package (src layout, `uv_build` backend). `main()` in `__init__.py` is exposed as the `ces-revisions` console script via `[project.scripts]`. `uv_build` expects the module at `src/ces_revisions`, matching `name = "ces-revisions"` in `pyproject.toml`; rename both together.
- Python 3.14 (`.python-version`; `requires-python = ">=3.14"`). Runtime dependencies are JAX, NumPy, NumPyro, ArviZ 1.x (`az.from_numpyro` returns an xarray `DataTree`), and Polars; Dynamax is a dev-only dependency kept as evidence for `docs/decisions/engine.md`.
- `src/ces_revisions/kalman.py` is the hand-written, NaN-masked Kalman filter and smoother chosen in `docs/decisions/engine.md`. Models add its log likelihood with `kalman_factor`, so states are never sampled sites. All JAX work is float64: `numpyro.enable_x64()` (and `numpyro.set_host_device_count`) must run before the first JAX operation, which `tests/conftest.py` does for the test session. BlackJAX waits for the sampler benchmark (roadmap Stage 17); confirm Python 3.14 support for any package you add.
- `scripts/` holds research tools outside the package; `pythonpath = ["scripts"]` in `[tool.pytest.ini_options]` lets tests import them by bare name. `scripts/archive_inventory.py captures` (network) records every Employment Situation release and every BLS and Internet Archive copy of the seasonal-adjustment files in `docs/inventory/`, and `scripts/archive_inventory.py inventory` (offline) derives the archive inventory from that evidence and regenerates its tables in `docs/ces-revisions-review.md`. Never hand-edit those CSVs or the generated blocks.

## Commands

```bash
uv sync                                         # create/update .venv; installs the package (editable) and the dev group
uv run ces-revisions                            # run the console entry point
uv run pytest                                   # run all tests
uv run pytest -m "not slow and not network"     # fast, hermetic tier to run on every change
uv run pytest -m slow                           # the synthetic pilot's four-chain NUTS fit (~15 s)
uv run pytest tests/test_smoke.py::test_main_prints_greeting   # run a single test
uv run ruff check                               # lint (--fix applies safe fixes)
uv run ruff format                              # format (--check to verify without writing)
uv run python scripts/archive_inventory.py inventory   # rebuild the archive inventory from docs/inventory/ (offline)
uv add <package>                                # add a runtime dependency (updates pyproject.toml + uv.lock)
uv add --dev <package>                          # add to the `dev` dependency group
```

pytest and ruff are dev dependencies. Ruff lints with its default rule set — already broad in ruff 0.16 (pyflakes, pylint, simplify, and partial bugbear/pyupgrade, among others) — extended with every `I` (import sorting), `B` (bugbear), and `UP` (pyupgrade, targeting Python 3.14 via `requires-python`) rule through `extend-select` in `[tool.ruff.lint]`. Don't switch that to `select`, which replaces the defaults rather than adding to them; tests live in `tests/`, and `[tool.pytest.ini_options]` registers the `network` (live BLS fetches, layout canaries) and `slow` (MCMC, end-to-end runs) markers and turns an unregistered marker into a collection error, so a misspelled marker can't slip past `-m` deselection. `ruff format` also scans Markdown (including `specs/`, `README.md`, and this file) and reformats Python code blocks inside it.
