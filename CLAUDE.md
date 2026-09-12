# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Research project on the **magnitude** (scale, not mean bias) of U.S. CES payroll revisions across four stages — first → second print, second → third, third → annual benchmark, and post-benchmark wedge-back — plus a Bayesian state-space design in which collection-interval, seasonal-adjustment, sample, BLS-funding, and staffing covariates enter the revision *variance*. The Python package is still the `uv init --package` scaffold; the substance so far lives in `specs/`.

## Specs

- `specs/ces-revisions-prompt.md` — the research brief: required sections, the five candidate drivers, and the design requirements. Treat it as the statement of scope.
- `specs/ces-revisions-research-{chatgpt,claude,gemini}.md` — three independent AI-generated responses to that brief. They are unverified drafts: figures, dates, and citations have not been checked against primary BLS sources, and the documents disagree with each other (e.g. 13 vs. ~11 NAICS supersectors). Verify a claim against primary sources before building on it. The filename labels may not match the tool that produced each file — the `gemini` file originally contained ChatGPT Deep Research citation markers.

Markdown in `specs/` is kept GitHub-renderable. When editing it:

- Put display math in ```` ```math ```` fences and inline math in `` $`…`$ ``. Do not use `\(…\)`, `\[…\]`, or bare `$…$`: GitHub ignores the first two, and Markdown escapes/emphasis can corrupt TeX inside bare dollars (a lone `=` or `-` line in unfenced math even becomes a setext heading).
- Escape literal dollar amounts as `\$`.
- Put non-TeX pseudo-math containing `_` (e.g. `θ_{s,t}`) in code spans so underscores are not read as emphasis.

## Layout and tooling

- `src/ces_revisions/` is an installable package (src layout, `uv_build` backend). `main()` in `__init__.py` is exposed as the `ces-revisions` console script via `[project.scripts]`. `uv_build` expects the module at `src/ces_revisions`, matching `name = "ces-revisions"` in `pyproject.toml`; rename both together.
- Python 3.14 (`.python-version`; `requires-python = ">=3.14"`). No runtime dependencies yet.
- The design in `specs/` targets NumPyro (NUTS), Dynamax, BlackJAX, ArviZ, and Polars. None are installed; confirm each supports Python 3.14 when adding it.

## Commands

```bash
uv sync                  # create/update .venv and install the package in editable mode
uv run ces-revisions     # run the console entry point
uv add <package>         # add a runtime dependency (updates pyproject.toml + uv.lock)
uv add --dev <package>   # add a dev dependency
```

No test runner, linter, or formatter is configured yet. When one is added, record its commands here, including how to run a single test.
