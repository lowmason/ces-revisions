# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Research project on the **magnitude** (scale, not mean bias) of U.S. CES payroll revisions across four stages — first → second print, second → third, third → annual benchmark, and post-benchmark wedge-back — plus a Bayesian state-space design in which collection-interval, seasonal-adjustment, sample, BLS-funding, and staffing covariates enter the revision *variance*. The Python package is still the `uv init --package` scaffold; the substance so far lives in `specs/`.

## Specs

- `specs/ces-revisions-prompt.md` — the research brief: required sections, the five candidate drivers, and the design requirements. Treat it as the statement of scope.
- `specs/ces-revisions-research-{chatgpt,claude,gemini}.md` — three independent AI-generated responses to that brief. They are unverified drafts: figures, dates, and citations have not been checked against primary BLS sources, and the documents disagree with each other (e.g. the gemini draft puts the post-pandemic first-closing rate near 40%, while the chatgpt and claude drafts give 60.4% for 2024 — collection and response rates are easy to conflate). Verify a claim against primary sources before building on it. Drafts are replaced wholesale when regenerated; re-normalize a replacement to the conventions below before committing it.

Markdown in `specs/` is kept GitHub-renderable. Freshly pasted AI output usually arrives with `$…$`/`$$…$$` or `\(…\)`/`\[…\]` math and needs converting. When adding or editing:

- Put display math in ```` ```math ```` fences and inline math in `` $`…`$ ``. Do not use `\(…\)`, `\[…\]`, `$$…$$`, or bare `$…$`: GitHub ignores the backslash forms, and Markdown escapes/emphasis can corrupt TeX inside bare dollars (`\%` loses its backslash; a lone `=` or `-` line in unfenced math becomes a setext heading).
- Escape literal dollar amounts as `\$` (e.g. `\$544.3 million`, `(\$m)`); an unescaped pair on one line is read as math, and a `|` inside it splits table cells.
- Use real headings (`##`, `###`), not bold-only lines, and write hard line breaks as a trailing `\` rather than two trailing spaces, which whitespace cleanup silently deletes.
- Put non-TeX pseudo-math containing `_` (e.g. `θ_{s,t}`) in code spans so underscores are not read as emphasis.

## Layout and tooling

- `src/ces_revisions/` is an installable package (src layout, `uv_build` backend). `main()` in `__init__.py` is exposed as the `ces-revisions` console script via `[project.scripts]`. `uv_build` expects the module at `src/ces_revisions`, matching `name = "ces-revisions"` in `pyproject.toml`; rename both together.
- Python 3.14 (`.python-version`; `requires-python = ">=3.14"`). No runtime dependencies yet.
- The design in `specs/` targets NumPyro (NUTS), Dynamax, BlackJAX, ArviZ, and Polars. None are installed; confirm each supports Python 3.14 when adding it.

## Commands

```bash
uv sync                                         # create/update .venv; installs the package (editable) and the dev group
uv run ces-revisions                            # run the console entry point
uv run pytest                                   # run all tests (exit code 5 = no tests collected)
uv run pytest path/to/test_file.py::test_name   # run a single test
uv run ruff check                               # lint (--fix applies safe fixes)
uv run ruff format                              # format (--check to verify without writing)
uv add <package>                                # add a runtime dependency (updates pyproject.toml + uv.lock)
uv add --dev <package>                          # add to the `dev` dependency group
```

pytest and ruff are dev dependencies. Ruff lints with its default rule set — already broad in ruff 0.16 (pyflakes, pylint, simplify, and partial bugbear/pyupgrade, among others) — extended with every `I` (import sorting), `B` (bugbear), and `UP` (pyupgrade, targeting Python 3.14 via `requires-python`) rule through `extend-select` in `[tool.ruff.lint]`. Don't switch that to `select`, which replaces the defaults rather than adding to them; pytest has no `[tool.pytest.ini_options]` section and there are no tests yet. `ruff format` also scans Markdown (including `specs/`, `README.md`, and this file) and reformats Python code blocks inside it.
