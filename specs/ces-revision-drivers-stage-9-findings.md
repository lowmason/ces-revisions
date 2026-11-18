# Stage 9 findings: the window to first closing, the first-closing collection rate and revision size, total nonfarm

Outcomes O3 and O4 of the [ces-revision-drivers roadmap](ces-revision-drivers-roadmap.md), Stage 9, implemented by [plan 15](plans/completed/15-ces-revision-drivers-stage-9.md). `uv run python scripts/first_closing.py report` writes the nine tables below from the built `data/panel/`. Do not edit them by hand.

## Question

Does a longer window to first closing raise the first-closing collection rate C1? How do the mean absolute F→S and F→T revisions of total nonfarm employment move with C1? And how does the Stage 1 answer change when the window ends at first closing, not at the release, 2003–2026?

## Answer

To be written from the tables of the real run.

## Data and window

To be written from the tables of the real run.

## Results

### Archived copies compared with LABSTAT

<!-- BEGIN GENERATED stage-9-copies -->
<!-- END GENERATED stage-9-copies -->

### Closing days on a federal holiday

<!-- BEGIN GENERATED stage-9-holidays -->
<!-- END GENERATED stage-9-holidays -->

### C1 by window to first closing

<!-- BEGIN GENERATED stage-9-rate-by-window -->
<!-- END GENERATED stage-9-rate-by-window -->

### Part (a): C1 on the window

```math
C1_t = \alpha + \beta W_t + \delta F_t + \mu_{y(t)} + \nu_{m(t)} + \varepsilon_t
```

<!-- BEGIN GENERATED stage-9-rate-regression -->
<!-- END GENERATED stage-9-rate-regression -->

### Revision size by C1 quintile

<!-- BEGIN GENERATED stage-9-revision-by-rate -->
<!-- END GENERATED stage-9-revision-by-rate -->

### Part (b): revision size on C1

```math
|r_t| = \alpha + \theta\, C1_t + \delta F_t + \mu_{y(t)} + \nu_{m(t)} + \varepsilon_t
```

<!-- BEGIN GENERATED stage-9-revision-regression -->
<!-- END GENERATED stage-9-revision-regression -->

### O4: the Stage 1 regression with the window to first closing

```math
|r_t| = \alpha + \beta X_t + \delta F_t + \mu_{y(t)} + \nu_{m(t)} + \varepsilon_t, \qquad X_t \in \{D_t, W_t\}
```

<!-- BEGIN GENERATED stage-9-window-regression -->
<!-- END GENERATED stage-9-window-regression -->

### Lapse-delayed releases

<!-- BEGIN GENERATED stage-9-lapse -->
<!-- END GENERATED stage-9-lapse -->

### March 2020 to December 2021

<!-- BEGIN GENERATED stage-9-pandemic -->
<!-- END GENERATED stage-9-pandemic -->

## Limits

To be written from the tables of the real run.

## Measured cost

To be written after the real run.
