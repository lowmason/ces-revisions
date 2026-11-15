# Development history

## /deferred — 2026-11-15

Closed deferred items:
- "Review Minor: gradient checks along `A_t` and `Z_t` (whole-branch" (deferred by 1-ces-revisions): dense-oracle finite-difference checks of the gradient along `transition_matrix` and `observation_matrix` in `tests/test_kalman.py`.
- "Review Important: test the monthly live window against member dates" (deferred by 2-ces-revisions): `captures` records each copy's newest ZIP member date. Prior-adjustment and outlier copies are credited to the first release on or after it, which moves two copies to 2015-07 and 2026-02. A test caps every copy's build lead at 7 days.
- "Review recommendations: evidence tests for two planning findings" (deferred by 5-ces-revisions): tests pin plan 5's EMPLOY-versus-table cells and its B-to-M counts.
