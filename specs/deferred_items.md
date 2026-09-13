# Deferred items

## 1-ces-revisions — 2026-09-12
- [ ] Review Minor: gradient checks along `A_t` and `Z_t` (whole-branch
      review of plan 1, triaged defer). `tests/test_kalman.py` checks the
      log-likelihood gradient only along a joint scaling of `transition_cov`
      and `observation_cov`, which covers Stage 1's scale-only pilot. A stage
      that samples parameters inside `transition_matrix` or
      `observation_matrix` (factor loadings, for example) needs
      finite-difference checks along those arrays against
      `tests/dense_reference.py`. See `specs/plans/completed/1-ces-revisions.md`.
      Size: quick-fix. Done when: the first stage that samples a parameter
      entering `transition_matrix` or `observation_matrix` adds a dense-oracle
      gradient check along it.
- [ ] Review recommendation: dispersed chain starts for Req 18 (whole-branch
      review of plan 1, triaged defer). NumPyro's default
      `init_to_uniform(radius=2)` starts each LogNormal scale in roughly
      [0.14, 7.4], below five of the six true scales in
      `tests/synthetic_pilot.py`. The pilot still passes, but Req 18 asks for
      four dispersed chains in reported fits, and the run harness that owns
      initialization arrives in roadmap Stage 6. See
      `specs/plans/completed/1-ces-revisions.md`. Size: quick-fix. Done when:
      the Stage 6 run harness starts the four chains from dispersed points
      and records the initialization strategy with each run.
