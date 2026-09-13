# CES payroll revisions: data inventory, archive inventory, and draft rulings

This document is the written finding that Req 20 of [`specs/ces-revisions.md`](../specs/ces-revisions.md) requires. Roadmap Stage 2 wrote its inventory half: the data-availability inventory, the seasonal-adjustment archive inventory, and rulings on the five disagreements among the research drafts that Req 20 names. Roadmap Stage 23 completes the literature by driver, the gap table, and the annotated bibliography.

## Evidence labels and citations

The three research drafts in `specs/` are inputs to this document, never its sources. Every numeric claim, date, and citation here was checked against a primary source (BLS, CRS, GAO, DOL or OMB, or the journal) or is marked *unverified*, and each cited page carries the date it was accessed. Each claim carries one evidence label, in the sense `specs/ces-revisions.md` adopts from the drafts:

- *documented*: a primary source reports the rule, fact, or statistic directly.
- *supported*: an explicit empirical design backs it, or it is computed from primary data.
- *asserted*: a source states it without evidence that would establish it.
- *contested*: the evidence conflicts, or does not support the claimed reading.
- *not found*: no primary source was located, which does not prove none exists.

## Summary of findings

**Status:** in progress — plan 2, Task 8.

## Literature by driver

**Status:** stub — roadmap Stage 23 writes this section: the literature on the magnitude of CES revisions, organized by the five candidate drivers (seasonality, collection interval, BLS funding, sample, and staffing), with the decomposition of revision sources running through it.

## Gap table

**Status:** stub — roadmap Stage 23 writes this section: one row per open question, giving the question, the closest existing work, why it falls short, the data it needs, and its feasibility from public sources.

## Data-availability inventory

**Status:** in progress — plan 2, Tasks 5 to 7.

Each subsection below is one group of the series Req 3 names, with one row per series. The columns are:

- **Vintage coverage**: whether values as first published can still be recovered, and from where.
- **Frequency**: the native frequency. Req 3 never interpolates an annual series to monthly.
- **Earliest date**: the first observation the access route provides.
- **Access route**: where the series is obtained.
- **Hand-build constraint**: what must be assembled by hand, and why.
- **First-publication lag**: when a value first became public relative to its reference period, which sets its `observable_at` date in roadmap Stages 4, 12 to 14, and 20.
- **Evidence** and **Citation**: the row's evidence label, and its primary sources or *unverified*.

### Collection window

### Seasonal

### Net birth–death

### Sample

### Institutional

### Party composition

### Macro and tail controls

## Seasonal-adjustment archive inventory

Req 9 leaves open which vintages have archived specification, prior-adjustment, and outlier files, and whether the unrounded NSA inputs are recoverable. This section settles both for every Employment Situation release from the May 2003 publication vintage onward, from BLS's pages and the Internet Archive's copies of BLS's files.

### What BLS publishes

The [CES seasonal adjustment files page](https://www.bls.gov/web/empsit/cesseasadj.htm) (accessed 2026-09-13) serves only the input files now in use. Three ZIP files hold the X-13ARIMA-SEATS specification files for series adjusted at the first preliminary estimates, among them `ces.spec.ae.zip` for all employees. Three more hold the files for series adjusted independently at the second preliminary estimates, and `ces.spec.other.zip` holds the calendar regressor files, the prior-adjustment file, and the recent-outliers file. The page's section on the availability of historical input files says only when the files change: the specification and prior-adjustment files each year, and the prior-adjustment and recent-outlier files with every monthly release. It links no earlier versions. The model specification tables for 2003 through 2013, which give each series' adjustment mode and calendar treatments but not its specification file, are in the [archived benchmark articles](https://www.bls.gov/web/empsit/cesbmkarch.htm) (accessed 2026-09-13). In March 2011 the page linked the same four ZIP files at `ftp.bls.gov/pub/suppl/`, as the [Internet Archive's copy of the page](https://web.archive.org/web/20110312035934/http://www.bls.gov:80/web/empsit/cesseasadj.htm) shows. Evidence: *documented*.

### Method

[`scripts/archive_inventory.py`](../scripts/archive_inventory.py) builds the inventory in two steps, whose outputs are committed in [`docs/inventory/`](inventory/):

1. `captures` reads the [Employment Situation archive index](https://www.bls.gov/bls/news-release/empsit.htm) into `es-vintages.csv`, one release per reference month, dropping links to releases scheduled after the day it runs. It fetches BLS's current `ces.spec.ae.zip` and `ces.spec.other.zip`, lists every copy of each that the Internet Archive holds at the current and pre-2012 locations, and downloads each copy. It writes each file's fingerprints of its specification, prior-adjustment, and outlier members, by name and CRC-32, to `archive-captures.csv`, and marks a listed copy the Internet Archive cannot serve as `unreplayable`.
2. `inventory` works offline from those two files. A release's prior-adjustment and outlier files were current from its 8:30 a.m. Eastern embargo until the next release. Its specification files were current from the benchmark release, which carries January estimates, until the next benchmark release, because the [seasonal adjustment technical notes](https://www.bls.gov/web/empsit/cesseasadjtn.htm) (accessed 2026-09-13) say all controllable variables remain fixed during the year. A copy made inside a release's window evidences that release's file.

Each file type gets one status per release: `bls_current` when BLS serves the file today, `internet_archive` when a copy from inside the window holds it, `conflicting_captures` when copies inside one window hold different contents, `not_archived` when no copy does, and `no_release` for a reference month with no release of its own. Only the all-employees specification set is inventoried, because hours and earnings are out of scope and the second-preliminary sets adjust detailed series that do not aggregate to total nonfarm. A revisit record, which the Internet Archive lists with status `-` for a copy identical to an earlier one, is not downloaded; each of the five for `ces.spec.ae.zip` falls in the same specification window as the earlier copy it repeats.

### Findings

<!-- BEGIN GENERATED archive-findings -->

- Releases inventoried: 279, estimating reference months 2003-05 through 2026-08 (released 2003-06-06 to 2026-09-04).
- Reference months without a release of their own: 2025-10.
- Specification files survive for 115 of 279 releases; the earliest such release estimates 2014-01.
- Prior-adjustment files survive for 29 of 279 releases; the earliest such release estimates 2014-06.
- Outlier files survive for 29 of 279 releases; the earliest such release estimates 2014-06.
- All three file types survive for 28 of 279 releases, and 0 cells are `conflicting_captures`.
- Unrounded NSA inputs are `not_published` for every release.

<!-- END GENERATED archive-findings -->

### Coverage by year

Releases by the year of their reference month, with how many keep each file type as a `bls_current` or `internet_archive` copy.

<!-- BEGIN GENERATED archive-coverage -->

| Reference year | Releases | Specification | Prior adjustment | Outliers | All three |
|---:|---:|---:|---:|---:|---:|
| 2003 | 8 | 0 | 0 | 0 | 0 |
| 2004 | 12 | 0 | 0 | 0 | 0 |
| 2005 | 12 | 0 | 0 | 0 | 0 |
| 2006 | 12 | 0 | 0 | 0 | 0 |
| 2007 | 12 | 0 | 0 | 0 | 0 |
| 2008 | 12 | 0 | 0 | 0 | 0 |
| 2009 | 12 | 0 | 0 | 0 | 0 |
| 2010 | 12 | 0 | 0 | 0 | 0 |
| 2011 | 12 | 0 | 0 | 0 | 0 |
| 2012 | 12 | 0 | 0 | 0 | 0 |
| 2013 | 12 | 0 | 0 | 0 | 0 |
| 2014 | 12 | 12 | 1 | 1 | 1 |
| 2015 | 12 | 12 | 2 | 2 | 2 |
| 2016 | 12 | 12 | 2 | 2 | 2 |
| 2017 | 12 | 12 | 2 | 2 | 2 |
| 2018 | 12 | 12 | 1 | 1 | 1 |
| 2019 | 12 | 0 | 0 | 0 | 0 |
| 2020 | 12 | 12 | 1 | 1 | 1 |
| 2021 | 12 | 12 | 3 | 3 | 3 |
| 2022 | 12 | 0 | 1 | 1 | 0 |
| 2023 | 12 | 0 | 0 | 0 | 0 |
| 2024 | 12 | 12 | 3 | 3 | 3 |
| 2025 | 11 | 11 | 7 | 7 | 7 |
| 2026 | 8 | 8 | 6 | 6 | 6 |
| Total | 279 | 115 | 29 | 29 | 28 |

<!-- END GENERATED archive-coverage -->

### Per-vintage inventory

Each surviving file's status links the copy that evidences it, and [`archive-inventory.csv`](inventory/archive-inventory.csv) holds the same rows with the evidence URLs.

<!-- BEGIN GENERATED archive-vintages -->

<details>
<summary>One row per reference month from May 2003</summary>

| Reference month | Release | Specification | Prior adjustment | Outliers | Unrounded NSA inputs | All three |
|---|---|---|---|---|---|---|
| 2003-05 | 2003-06-06 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2003-06 | 2003-07-03 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2003-07 | 2003-08-01 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2003-08 | 2003-09-05 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2003-09 | 2003-10-03 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2003-10 | 2003-11-07 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2003-11 | 2003-12-05 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2003-12 | 2004-01-09 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2004-01 | 2004-02-06 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2004-02 | 2004-03-05 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2004-03 | 2004-04-02 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2004-04 | 2004-05-07 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2004-05 | 2004-06-04 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2004-06 | 2004-07-02 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2004-07 | 2004-08-06 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2004-08 | 2004-09-03 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2004-09 | 2004-10-08 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2004-10 | 2004-11-05 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2004-11 | 2004-12-03 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2004-12 | 2005-01-07 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2005-01 | 2005-02-04 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2005-02 | 2005-03-04 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2005-03 | 2005-04-01 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2005-04 | 2005-05-06 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2005-05 | 2005-06-03 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2005-06 | 2005-07-08 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2005-07 | 2005-08-05 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2005-08 | 2005-09-02 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2005-09 | 2005-10-07 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2005-10 | 2005-11-04 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2005-11 | 2005-12-02 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2005-12 | 2006-01-06 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2006-01 | 2006-02-03 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2006-02 | 2006-03-10 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2006-03 | 2006-04-07 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2006-04 | 2006-05-05 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2006-05 | 2006-06-02 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2006-06 | 2006-07-07 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2006-07 | 2006-08-04 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2006-08 | 2006-09-01 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2006-09 | 2006-10-06 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2006-10 | 2006-11-03 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2006-11 | 2006-12-08 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2006-12 | 2007-01-05 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2007-01 | 2007-02-02 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2007-02 | 2007-03-09 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2007-03 | 2007-04-06 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2007-04 | 2007-05-04 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2007-05 | 2007-06-01 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2007-06 | 2007-07-06 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2007-07 | 2007-08-03 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2007-08 | 2007-09-07 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2007-09 | 2007-10-05 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2007-10 | 2007-11-02 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2007-11 | 2007-12-07 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2007-12 | 2008-01-04 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2008-01 | 2008-02-01 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2008-02 | 2008-03-07 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2008-03 | 2008-04-04 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2008-04 | 2008-05-02 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2008-05 | 2008-06-06 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2008-06 | 2008-07-03 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2008-07 | 2008-08-01 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2008-08 | 2008-09-05 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2008-09 | 2008-10-03 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2008-10 | 2008-11-07 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2008-11 | 2008-12-05 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2008-12 | 2009-01-09 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2009-01 | 2009-02-06 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2009-02 | 2009-03-06 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2009-03 | 2009-04-03 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2009-04 | 2009-05-08 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2009-05 | 2009-06-05 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2009-06 | 2009-07-02 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2009-07 | 2009-08-07 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2009-08 | 2009-09-04 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2009-09 | 2009-10-02 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2009-10 | 2009-11-06 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2009-11 | 2009-12-04 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2009-12 | 2010-01-08 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2010-01 | 2010-02-05 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2010-02 | 2010-03-05 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2010-03 | 2010-04-02 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2010-04 | 2010-05-07 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2010-05 | 2010-06-04 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2010-06 | 2010-07-02 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2010-07 | 2010-08-06 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2010-08 | 2010-09-03 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2010-09 | 2010-10-08 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2010-10 | 2010-11-05 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2010-11 | 2010-12-03 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2010-12 | 2011-01-07 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2011-01 | 2011-02-04 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2011-02 | 2011-03-04 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2011-03 | 2011-04-01 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2011-04 | 2011-05-06 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2011-05 | 2011-06-03 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2011-06 | 2011-07-08 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2011-07 | 2011-08-05 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2011-08 | 2011-09-02 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2011-09 | 2011-10-07 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2011-10 | 2011-11-04 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2011-11 | 2011-12-02 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2011-12 | 2012-01-06 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2012-01 | 2012-02-03 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2012-02 | 2012-03-09 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2012-03 | 2012-04-06 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2012-04 | 2012-05-04 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2012-05 | 2012-06-01 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2012-06 | 2012-07-06 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2012-07 | 2012-08-03 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2012-08 | 2012-09-07 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2012-09 | 2012-10-05 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2012-10 | 2012-11-02 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2012-11 | 2012-12-07 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2012-12 | 2013-01-04 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2013-01 | 2013-02-01 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2013-02 | 2013-03-08 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2013-03 | 2013-04-05 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2013-04 | 2013-05-03 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2013-05 | 2013-06-07 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2013-06 | 2013-07-05 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2013-07 | 2013-08-02 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2013-08 | 2013-09-06 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2013-09 | 2013-10-22 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2013-10 | 2013-11-08 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2013-11 | 2013-12-06 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2013-12 | 2014-01-10 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2014-01 | 2014-02-07 | [`internet_archive`](https://web.archive.org/web/20140719155542id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2014-02 | 2014-03-07 | [`internet_archive`](https://web.archive.org/web/20140719155542id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2014-03 | 2014-04-04 | [`internet_archive`](https://web.archive.org/web/20140719155542id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2014-04 | 2014-05-02 | [`internet_archive`](https://web.archive.org/web/20140719155542id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2014-05 | 2014-06-06 | [`internet_archive`](https://web.archive.org/web/20140719155542id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2014-06 | 2014-07-03 | [`internet_archive`](https://web.archive.org/web/20140719155542id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | [`internet_archive`](https://web.archive.org/web/20140719134601id_/http://www.bls.gov/web/empsit/ces.spec.other.zip) | [`internet_archive`](https://web.archive.org/web/20140719134601id_/http://www.bls.gov/web/empsit/ces.spec.other.zip) | `not_published` | true |
| 2014-07 | 2014-08-01 | [`internet_archive`](https://web.archive.org/web/20140719155542id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2014-08 | 2014-09-05 | [`internet_archive`](https://web.archive.org/web/20140719155542id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2014-09 | 2014-10-03 | [`internet_archive`](https://web.archive.org/web/20140719155542id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2014-10 | 2014-11-07 | [`internet_archive`](https://web.archive.org/web/20140719155542id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2014-11 | 2014-12-05 | [`internet_archive`](https://web.archive.org/web/20140719155542id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2014-12 | 2015-01-09 | [`internet_archive`](https://web.archive.org/web/20140719155542id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2015-01 | 2015-02-06 | [`internet_archive`](https://web.archive.org/web/20150321060819id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2015-02 | 2015-03-06 | [`internet_archive`](https://web.archive.org/web/20150321060819id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | [`internet_archive`](https://web.archive.org/web/20150321061757id_/http://www.bls.gov/web/empsit/ces.spec.other.zip) | [`internet_archive`](https://web.archive.org/web/20150321061757id_/http://www.bls.gov/web/empsit/ces.spec.other.zip) | `not_published` | true |
| 2015-03 | 2015-04-03 | [`internet_archive`](https://web.archive.org/web/20150321060819id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2015-04 | 2015-05-08 | [`internet_archive`](https://web.archive.org/web/20150321060819id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2015-05 | 2015-06-05 | [`internet_archive`](https://web.archive.org/web/20150321060819id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2015-06 | 2015-07-02 | [`internet_archive`](https://web.archive.org/web/20150321060819id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2015-07 | 2015-08-07 | [`internet_archive`](https://web.archive.org/web/20150321060819id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2015-08 | 2015-09-04 | [`internet_archive`](https://web.archive.org/web/20150321060819id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | [`internet_archive`](https://web.archive.org/web/20150905164751id_/http://www.bls.gov/web/empsit/ces.spec.other.zip) | [`internet_archive`](https://web.archive.org/web/20150905164751id_/http://www.bls.gov/web/empsit/ces.spec.other.zip) | `not_published` | true |
| 2015-09 | 2015-10-02 | [`internet_archive`](https://web.archive.org/web/20150321060819id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2015-10 | 2015-11-06 | [`internet_archive`](https://web.archive.org/web/20150321060819id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2015-11 | 2015-12-04 | [`internet_archive`](https://web.archive.org/web/20150321060819id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2015-12 | 2016-01-08 | [`internet_archive`](https://web.archive.org/web/20150321060819id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2016-01 | 2016-02-05 | [`internet_archive`](https://web.archive.org/web/20160825180042id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2016-02 | 2016-03-04 | [`internet_archive`](https://web.archive.org/web/20160825180042id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2016-03 | 2016-04-01 | [`internet_archive`](https://web.archive.org/web/20160825180042id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2016-04 | 2016-05-06 | [`internet_archive`](https://web.archive.org/web/20160825180042id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2016-05 | 2016-06-03 | [`internet_archive`](https://web.archive.org/web/20160825180042id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2016-06 | 2016-07-08 | [`internet_archive`](https://web.archive.org/web/20160825180042id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2016-07 | 2016-08-05 | [`internet_archive`](https://web.archive.org/web/20160825180042id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | [`internet_archive`](https://web.archive.org/web/20160825173250id_/http://www.bls.gov/web/empsit/ces.spec.other.zip) | [`internet_archive`](https://web.archive.org/web/20160825173250id_/http://www.bls.gov/web/empsit/ces.spec.other.zip) | `not_published` | true |
| 2016-08 | 2016-09-02 | [`internet_archive`](https://web.archive.org/web/20160825180042id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2016-09 | 2016-10-07 | [`internet_archive`](https://web.archive.org/web/20160825180042id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2016-10 | 2016-11-04 | [`internet_archive`](https://web.archive.org/web/20160825180042id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2016-11 | 2016-12-02 | [`internet_archive`](https://web.archive.org/web/20160825180042id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2016-12 | 2017-01-06 | [`internet_archive`](https://web.archive.org/web/20160825180042id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip) | [`internet_archive`](https://web.archive.org/web/20170110014114id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | [`internet_archive`](https://web.archive.org/web/20170110014114id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | `not_published` | true |
| 2017-01 | 2017-02-03 | [`internet_archive`](https://web.archive.org/web/20170507080910id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2017-02 | 2017-03-10 | [`internet_archive`](https://web.archive.org/web/20170507080910id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2017-03 | 2017-04-07 | [`internet_archive`](https://web.archive.org/web/20170507080910id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2017-04 | 2017-05-05 | [`internet_archive`](https://web.archive.org/web/20170507080910id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | [`internet_archive`](https://web.archive.org/web/20170507081200id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | [`internet_archive`](https://web.archive.org/web/20170507081200id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | `not_published` | true |
| 2017-05 | 2017-06-02 | [`internet_archive`](https://web.archive.org/web/20170507080910id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | [`internet_archive`](https://web.archive.org/web/20170606104922id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | [`internet_archive`](https://web.archive.org/web/20170606104922id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | `not_published` | true |
| 2017-06 | 2017-07-07 | [`internet_archive`](https://web.archive.org/web/20170507080910id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2017-07 | 2017-08-04 | [`internet_archive`](https://web.archive.org/web/20170507080910id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2017-08 | 2017-09-01 | [`internet_archive`](https://web.archive.org/web/20170507080910id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2017-09 | 2017-10-06 | [`internet_archive`](https://web.archive.org/web/20170507080910id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2017-10 | 2017-11-03 | [`internet_archive`](https://web.archive.org/web/20170507080910id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2017-11 | 2017-12-08 | [`internet_archive`](https://web.archive.org/web/20170507080910id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2017-12 | 2018-01-05 | [`internet_archive`](https://web.archive.org/web/20170507080910id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2018-01 | 2018-02-02 | [`internet_archive`](https://web.archive.org/web/20190111220437id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2018-02 | 2018-03-09 | [`internet_archive`](https://web.archive.org/web/20190111220437id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2018-03 | 2018-04-06 | [`internet_archive`](https://web.archive.org/web/20190111220437id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2018-04 | 2018-05-04 | [`internet_archive`](https://web.archive.org/web/20190111220437id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2018-05 | 2018-06-01 | [`internet_archive`](https://web.archive.org/web/20190111220437id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2018-06 | 2018-07-06 | [`internet_archive`](https://web.archive.org/web/20190111220437id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2018-07 | 2018-08-03 | [`internet_archive`](https://web.archive.org/web/20190111220437id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2018-08 | 2018-09-07 | [`internet_archive`](https://web.archive.org/web/20190111220437id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2018-09 | 2018-10-05 | [`internet_archive`](https://web.archive.org/web/20190111220437id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2018-10 | 2018-11-02 | [`internet_archive`](https://web.archive.org/web/20190111220437id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2018-11 | 2018-12-07 | [`internet_archive`](https://web.archive.org/web/20190111220437id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2018-12 | 2019-01-04 | [`internet_archive`](https://web.archive.org/web/20190111220437id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | [`internet_archive`](https://web.archive.org/web/20190111160920id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | [`internet_archive`](https://web.archive.org/web/20190111160920id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | `not_published` | true |
| 2019-01 | 2019-02-01 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2019-02 | 2019-03-08 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2019-03 | 2019-04-05 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2019-04 | 2019-05-03 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2019-05 | 2019-06-07 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2019-06 | 2019-07-05 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2019-07 | 2019-08-02 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2019-08 | 2019-09-06 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2019-09 | 2019-10-04 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2019-10 | 2019-11-01 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2019-11 | 2019-12-06 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2019-12 | 2020-01-10 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2020-01 | 2020-02-07 | [`internet_archive`](https://web.archive.org/web/20201018125113id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2020-02 | 2020-03-06 | [`internet_archive`](https://web.archive.org/web/20201018125113id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2020-03 | 2020-04-03 | [`internet_archive`](https://web.archive.org/web/20201018125113id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2020-04 | 2020-05-08 | [`internet_archive`](https://web.archive.org/web/20201018125113id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2020-05 | 2020-06-05 | [`internet_archive`](https://web.archive.org/web/20201018125113id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2020-06 | 2020-07-02 | [`internet_archive`](https://web.archive.org/web/20201018125113id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2020-07 | 2020-08-07 | [`internet_archive`](https://web.archive.org/web/20201018125113id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2020-08 | 2020-09-04 | [`internet_archive`](https://web.archive.org/web/20201018125113id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2020-09 | 2020-10-02 | [`internet_archive`](https://web.archive.org/web/20201018125113id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | [`internet_archive`](https://web.archive.org/web/20201018123503id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | [`internet_archive`](https://web.archive.org/web/20201018123503id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | `not_published` | true |
| 2020-10 | 2020-11-06 | [`internet_archive`](https://web.archive.org/web/20201018125113id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2020-11 | 2020-12-04 | [`internet_archive`](https://web.archive.org/web/20201018125113id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2020-12 | 2021-01-08 | [`internet_archive`](https://web.archive.org/web/20201018125113id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2021-01 | 2021-02-05 | [`internet_archive`](https://web.archive.org/web/20210318044659id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2021-02 | 2021-03-05 | [`internet_archive`](https://web.archive.org/web/20210318044659id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | [`internet_archive`](https://web.archive.org/web/20210318011343id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | [`internet_archive`](https://web.archive.org/web/20210318011343id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | `not_published` | true |
| 2021-03 | 2021-04-02 | [`internet_archive`](https://web.archive.org/web/20210318044659id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2021-04 | 2021-05-07 | [`internet_archive`](https://web.archive.org/web/20210318044659id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2021-05 | 2021-06-04 | [`internet_archive`](https://web.archive.org/web/20210318044659id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2021-06 | 2021-07-02 | [`internet_archive`](https://web.archive.org/web/20210318044659id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2021-07 | 2021-08-06 | [`internet_archive`](https://web.archive.org/web/20210318044659id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2021-08 | 2021-09-03 | [`internet_archive`](https://web.archive.org/web/20210318044659id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | [`internet_archive`](https://web.archive.org/web/20210917004822id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | [`internet_archive`](https://web.archive.org/web/20210917004822id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | `not_published` | true |
| 2021-09 | 2021-10-08 | [`internet_archive`](https://web.archive.org/web/20210318044659id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2021-10 | 2021-11-05 | [`internet_archive`](https://web.archive.org/web/20210318044659id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | [`internet_archive`](https://web.archive.org/web/20211105131924id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | [`internet_archive`](https://web.archive.org/web/20211105131924id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | `not_published` | true |
| 2021-11 | 2021-12-03 | [`internet_archive`](https://web.archive.org/web/20210318044659id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2021-12 | 2022-01-07 | [`internet_archive`](https://web.archive.org/web/20210318044659id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2022-01 | 2022-02-04 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2022-02 | 2022-03-04 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2022-03 | 2022-04-01 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2022-04 | 2022-05-06 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2022-05 | 2022-06-03 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2022-06 | 2022-07-08 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2022-07 | 2022-08-05 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2022-08 | 2022-09-02 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2022-09 | 2022-10-07 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2022-10 | 2022-11-04 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2022-11 | 2022-12-02 | `not_archived` | [`internet_archive`](https://web.archive.org/web/20221205142739id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | [`internet_archive`](https://web.archive.org/web/20221205142739id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | `not_published` | false |
| 2022-12 | 2023-01-06 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2023-01 | 2023-02-03 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2023-02 | 2023-03-10 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2023-03 | 2023-04-07 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2023-04 | 2023-05-05 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2023-05 | 2023-06-02 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2023-06 | 2023-07-07 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2023-07 | 2023-08-04 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2023-08 | 2023-09-01 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2023-09 | 2023-10-06 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2023-10 | 2023-11-03 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2023-11 | 2023-12-08 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2023-12 | 2024-01-05 | `not_archived` | `not_archived` | `not_archived` | `not_published` | false |
| 2024-01 | 2024-02-02 | [`internet_archive`](https://web.archive.org/web/20240927002315id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2024-02 | 2024-03-08 | [`internet_archive`](https://web.archive.org/web/20240927002315id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2024-03 | 2024-04-05 | [`internet_archive`](https://web.archive.org/web/20240927002315id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2024-04 | 2024-05-03 | [`internet_archive`](https://web.archive.org/web/20240927002315id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2024-05 | 2024-06-07 | [`internet_archive`](https://web.archive.org/web/20240927002315id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2024-06 | 2024-07-05 | [`internet_archive`](https://web.archive.org/web/20240927002315id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2024-07 | 2024-08-02 | [`internet_archive`](https://web.archive.org/web/20240927002315id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2024-08 | 2024-09-06 | [`internet_archive`](https://web.archive.org/web/20240927002315id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | [`internet_archive`](https://web.archive.org/web/20240926232604id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | [`internet_archive`](https://web.archive.org/web/20240926232604id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | `not_published` | true |
| 2024-09 | 2024-10-04 | [`internet_archive`](https://web.archive.org/web/20240927002315id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2024-10 | 2024-11-01 | [`internet_archive`](https://web.archive.org/web/20240927002315id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2024-11 | 2024-12-06 | [`internet_archive`](https://web.archive.org/web/20240927002315id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | [`internet_archive`](https://web.archive.org/web/20241211002914id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | [`internet_archive`](https://web.archive.org/web/20241211002914id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | `not_published` | true |
| 2024-12 | 2025-01-10 | [`internet_archive`](https://web.archive.org/web/20240927002315id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | [`internet_archive`](https://web.archive.org/web/20250202014501id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | [`internet_archive`](https://web.archive.org/web/20250202014501id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | `not_published` | true |
| 2025-01 | 2025-02-07 | [`internet_archive`](https://web.archive.org/web/20250208075248id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | [`internet_archive`](https://web.archive.org/web/20250208074215id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | [`internet_archive`](https://web.archive.org/web/20250208074215id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | `not_published` | true |
| 2025-02 | 2025-03-07 | [`internet_archive`](https://web.archive.org/web/20250208075248id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | [`internet_archive`](https://web.archive.org/web/20250327025357id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | [`internet_archive`](https://web.archive.org/web/20250327025357id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | `not_published` | true |
| 2025-03 | 2025-04-04 | [`internet_archive`](https://web.archive.org/web/20250208075248id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | [`internet_archive`](https://web.archive.org/web/20250404181957id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | [`internet_archive`](https://web.archive.org/web/20250404181957id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | `not_published` | true |
| 2025-04 | 2025-05-02 | [`internet_archive`](https://web.archive.org/web/20250208075248id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2025-05 | 2025-06-06 | [`internet_archive`](https://web.archive.org/web/20250208075248id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2025-06 | 2025-07-03 | [`internet_archive`](https://web.archive.org/web/20250208075248id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2025-07 | 2025-08-01 | [`internet_archive`](https://web.archive.org/web/20250208075248id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | [`internet_archive`](https://web.archive.org/web/20250802202747id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | [`internet_archive`](https://web.archive.org/web/20250802202747id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | `not_published` | true |
| 2025-08 | 2025-09-05 | [`internet_archive`](https://web.archive.org/web/20250208075248id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | [`internet_archive`](https://web.archive.org/web/20250911000514id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | [`internet_archive`](https://web.archive.org/web/20250911000514id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | `not_published` | true |
| 2025-09 | 2025-11-20 | [`internet_archive`](https://web.archive.org/web/20250208075248id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | [`internet_archive`](https://web.archive.org/web/20251122085546id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | [`internet_archive`](https://web.archive.org/web/20251122085546id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | `not_published` | true |
| 2025-10 | none | `no_release` | `no_release` | `no_release` | `no_release` | false |
| 2025-11 | 2025-12-16 | [`internet_archive`](https://web.archive.org/web/20250208075248id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | [`internet_archive`](https://web.archive.org/web/20251220081843id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | [`internet_archive`](https://web.archive.org/web/20251220081843id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | `not_published` | true |
| 2025-12 | 2026-01-09 | [`internet_archive`](https://web.archive.org/web/20250208075248id_/https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2026-01 | 2026-02-11 | [`bls_current`](https://www.bls.gov/web/empsit/ces.spec.ae.zip) | [`internet_archive`](https://web.archive.org/web/20260214200550id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | [`internet_archive`](https://web.archive.org/web/20260214200550id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | `not_published` | true |
| 2026-02 | 2026-03-06 | [`bls_current`](https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2026-03 | 2026-04-03 | [`bls_current`](https://www.bls.gov/web/empsit/ces.spec.ae.zip) | [`internet_archive`](https://web.archive.org/web/20260502095306id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | [`internet_archive`](https://web.archive.org/web/20260502095306id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | `not_published` | true |
| 2026-04 | 2026-05-08 | [`bls_current`](https://www.bls.gov/web/empsit/ces.spec.ae.zip) | [`internet_archive`](https://web.archive.org/web/20260516123319id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | [`internet_archive`](https://web.archive.org/web/20260516123319id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | `not_published` | true |
| 2026-05 | 2026-06-05 | [`bls_current`](https://www.bls.gov/web/empsit/ces.spec.ae.zip) | [`internet_archive`](https://web.archive.org/web/20260613101136id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | [`internet_archive`](https://web.archive.org/web/20260613101136id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | `not_published` | true |
| 2026-06 | 2026-07-02 | [`bls_current`](https://www.bls.gov/web/empsit/ces.spec.ae.zip) | `not_archived` | `not_archived` | `not_published` | false |
| 2026-07 | 2026-08-07 | [`bls_current`](https://www.bls.gov/web/empsit/ces.spec.ae.zip) | [`internet_archive`](https://web.archive.org/web/20260819192341id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | [`internet_archive`](https://web.archive.org/web/20260819192341id_/https://www.bls.gov/web/empsit/ces.spec.other.zip) | `not_published` | true |
| 2026-08 | 2026-09-04 | [`bls_current`](https://www.bls.gov/web/empsit/ces.spec.ae.zip) | [`bls_current`](https://www.bls.gov/web/empsit/ces.spec.other.zip) | [`bls_current`](https://www.bls.gov/web/empsit/ces.spec.other.zip) | `not_published` | true |

</details>

<!-- END GENERATED archive-vintages -->

### Unrounded NSA inputs

BLS does not publish them. The [seasonal adjustment technical notes](https://www.bls.gov/web/empsit/cesseasadjtn.htm) describe the input data file of NSA estimates that X-13ARIMA-SEATS reads, from which BLS first removes strikes and other prior adjustments, and they say seasonal adjustment runs on unrounded data while the data BLS publishes are rounded. No copy of either ZIP file, current or archived, holds an input data file: the `unexpected` column of [`archive-captures.csv`](inventory/archive-captures.csv) is empty, so every member is a specification file, a calendar regressor file, a readme, the prior-adjustment file, the outlier file, or, in the [Internet Archive's copy of `ces.spec.other.zip` from 2021-03-18](https://web.archive.org/web/20210318011343/https://www.bls.gov/web/empsit/ces.spec.other.zip), an earlier copy of that ZIP file nested inside it, which held only the same calendar regressor files, readme, and prior-adjustment file and an earlier outlier file when opened on 2026-09-13. The prior-adjustment file is the one unrounded input BLS does publish, according to the technical notes' footnote to figure 2. Evidence: *documented*.

The NSA employment levels BLS publishes are whole thousands for total nonfarm and its aggregates, but not for more detailed industries. In [`ce.data.0.AllCESSeries`](https://download.bls.gov/pub/time.series/ce/ce.data.0.AllCESSeries), none of the 307 values that total nonfarm and each of the eleven supersectors carry from January 2003 on has a nonzero decimal. In the CES vintage data file [`cesvinall.zip`](https://www.bls.gov/web/empsit/cesvinall.zip) (both accessed 2026-09-13), 21 of the 113 NSA files hold whole thousands: those for total nonfarm, total private, goods-producing, service-providing, private service-providing, the eleven supersectors, durable and nondurable goods, and federal, state, and local government. The other 92, all for industries below those aggregates, carry a nonzero decimal in 7,491,962 values, listed file by file below. Req 15's rounding variance of 1/12, in thousands squared, therefore holds for the supersector levels that Reqs 4 and 5 model, which are published in whole thousands, while a level published to one decimal place has rounding variance 1/1200. Evidence: *supported*.

<details>
<summary>The 92 NSA vintage files with nonzero decimals</summary>

| Vintage file | CES industry | Values with a nonzero decimal |
|---|---|---:|
| `tri_102100_NSA.csv` | Mining, quarrying, and oil and gas extraction | 169,685 |
| `tri_102110_NSA.csv` | Oil and gas extraction | 126,324 |
| `tri_102120_NSA.csv` | Mining (except oil and gas) | 71,554 |
| `tri_102130_NSA.csv` | Support activities for mining | 73,352 |
| `tri_202360_NSA.csv` | Construction of buildings | 73,392 |
| `tri_202370_NSA.csv` | Heavy and civil engineering construction | 72,721 |
| `tri_202380_NSA.csv` | Specialty trade contractors | 115,762 |
| `tri_313210_NSA.csv` | Wood product manufacturing | 74,408 |
| `tri_313270_NSA.csv` | Nonmetallic mineral product manufacturing | 190,523 |
| `tri_313310_NSA.csv` | Primary metal manufacturing | 71,463 |
| `tri_313320_NSA.csv` | Fabricated metal product manufacturing | 73,774 |
| `tri_313330_NSA.csv` | Machinery manufacturing | 72,705 |
| `tri_313340_NSA.csv` | Computer and electronic product manufacturing | 71,148 |
| `tri_313350_NSA.csv` | Electrical equipment, appliance, and component manufacturing | 73,080 |
| `tri_313360_NSA.csv` | Transportation equipment manufacturing | 73,925 |
| `tri_313370_NSA.csv` | Furniture and related product manufacturing | 73,241 |
| `tri_313390_NSA.csv` | Miscellaneous manufacturing | 73,204 |
| `tri_323110_NSA.csv` | Food manufacturing | 71,850 |
| `tri_323130_NSA.csv` | Textile mills | 75,224 |
| `tri_323140_NSA.csv` | Textile product mills | 74,823 |
| `tri_323150_NSA.csv` | Apparel manufacturing | 73,129 |
| `tri_323220_NSA.csv` | Paper manufacturing | 73,303 |
| `tri_323230_NSA.csv` | Printing and related support activities | 73,611 |
| `tri_323240_NSA.csv` | Petroleum and coal products manufacturing | 73,236 |
| `tri_323250_NSA.csv` | Chemical manufacturing | 71,031 |
| `tri_323260_NSA.csv` | Plastics and rubber products manufacturing | 73,235 |
| `tri_323290_NSA.csv` | Beverage, tobacco, and leather and allied product manufacturing | 51,117 |
| `tri_414200_NSA.csv` | Wholesale trade | 226,385 |
| `tri_414230_NSA.csv` | Merchant wholesalers, durable goods | 75,028 |
| `tri_414240_NSA.csv` | Merchant wholesalers, nondurable goods | 74,404 |
| `tri_414250_NSA.csv` | Wholesale trade agents and brokers | 73,660 |
| `tri_420000_NSA.csv` | Retail trade | 222,321 |
| `tri_424410_NSA.csv` | Motor vehicle and parts dealers | 72,408 |
| `tri_424440_NSA.csv` | Building material and garden equipment and supplies dealers | 72,556 |
| `tri_424450_NSA.csv` | Food and beverage retailers | 73,488 |
| `tri_424490_NSA.csv` | Furniture, home furnishings, electronics, and appliance retailers | 14,072 |
| `tri_424550_NSA.csv` | General merchandise retailers | 14,282 |
| `tri_424560_NSA.csv` | Health and personal care retailers | 14,010 |
| `tri_424570_NSA.csv` | Gasoline stations and fuel dealers | 13,874 |
| `tri_424580_NSA.csv` | Clothing, clothing accessories, shoe, and jewelry retailers | 13,846 |
| `tri_424590_NSA.csv` | Sporting goods, hobby, musical instrument, book, and miscellaneous retailers | 14,026 |
| `tri_430000_NSA.csv` | Transportation and warehousing | 126,916 |
| `tri_434810_NSA.csv` | Air transportation | 73,303 |
| `tri_434820_NSA.csv` | Rail transportation | 200,566 |
| `tri_434830_NSA.csv` | Water transportation | 72,636 |
| `tri_434840_NSA.csv` | Truck transportation | 73,125 |
| `tri_434850_NSA.csv` | Transit and ground passenger transportation | 72,884 |
| `tri_434860_NSA.csv` | Pipeline transportation | 73,614 |
| `tri_434870_NSA.csv` | Scenic and sightseeing transportation | 73,500 |
| `tri_434880_NSA.csv` | Support activities for transportation | 72,556 |
| `tri_434920_NSA.csv` | Couriers and messengers | 71,292 |
| `tri_434930_NSA.csv` | Warehousing and storage | 72,051 |
| `tri_442200_NSA.csv` | Utilities | 151,845 |
| `tri_505120_NSA.csv` | Motion picture and sound recording industries | 73,287 |
| `tri_505130_NSA.csv` | Publishing industries | 13,803 |
| `tri_505160_NSA.csv` | Broadcasting and content providers | 13,767 |
| `tri_505170_NSA.csv` | Telecommunications | 73,315 |
| `tri_505180_NSA.csv` | Computing infrastructure providers, data processing, web hosting, and related services | 73,700 |
| `tri_505190_NSA.csv` | Web search portals, libraries, archives, and other information services | 73,016 |
| `tri_555200_NSA.csv` | Finance and insurance | 72,797 |
| `tri_555210_NSA.csv` | Monetary authorities-central bank | 72,072 |
| `tri_555220_NSA.csv` | Credit intermediation and related activities | 71,391 |
| `tri_555230_NSA.csv` | Securities, commodity contracts, funds, trusts, and other financial vehicles, investments, and related activities | 73,089 |
| `tri_555240_NSA.csv` | Insurance carriers and related activities | 73,370 |
| `tri_555300_NSA.csv` | Real estate and rental and leasing | 73,052 |
| `tri_555310_NSA.csv` | Real estate | 73,793 |
| `tri_555320_NSA.csv` | Rental and leasing services | 70,831 |
| `tri_555330_NSA.csv` | Lessors of nonfinancial intangible assets (except copyrighted works) | 69,916 |
| `tri_605400_NSA.csv` | Professional, scientific, and technical services | 72,611 |
| `tri_605500_NSA.csv` | Management of companies and enterprises | 72,303 |
| `tri_605600_NSA.csv` | Administrative and support and waste management and remediation services | 73,389 |
| `tri_605610_NSA.csv` | Administrative and support services | 72,285 |
| `tri_605620_NSA.csv` | Waste management and remediation services | 73,220 |
| `tri_656100_NSA.csv` | Private educational services | 72,833 |
| `tri_656200_NSA.csv` | Health care and social assistance | 73,513 |
| `tri_656210_NSA.csv` | Ambulatory health care services | 72,034 |
| `tri_656220_NSA.csv` | Hospitals | 74,278 |
| `tri_656230_NSA.csv` | Nursing and residential care facilities | 71,016 |
| `tri_656240_NSA.csv` | Social assistance | 72,620 |
| `tri_707100_NSA.csv` | Arts, entertainment, and recreation | 71,463 |
| `tri_707110_NSA.csv` | Performing arts, spectator sports, and related industries | 73,519 |
| `tri_707120_NSA.csv` | Museums, historical sites, and similar institutions | 70,945 |
| `tri_707130_NSA.csv` | Amusement, gambling, and recreation industries | 74,424 |
| `tri_707200_NSA.csv` | Accommodation and food services | 75,262 |
| `tri_707210_NSA.csv` | Accommodation | 125,968 |
| `tri_707220_NSA.csv` | Food services and drinking places | 72,344 |
| `tri_808110_NSA.csv` | Repair and maintenance | 74,576 |
| `tri_808120_NSA.csv` | Personal and laundry services | 72,907 |
| `tri_808130_NSA.csv` | Religious, grantmaking, civic, professional, and similar organizations | 72,477 |
| `tri_909110_NSA.csv` | Federal, except U.S. Postal Service | 222,288 |
| `tri_909220_NSA.csv` | State government, excluding education | 171,352 |
| `tri_909320_NSA.csv` | Local government, excluding education | 173,668 |

</details>

### Consequences for later stages

- **Req 9 channel (b), roadmap Stages 25 and 26.** Only the releases counted above as keeping all three file types can seed a vintage-specific X-13 reproduction, and none of them has unrounded inputs, so Stage 25 can at best match published SA values to rounding from rounded inputs. Every other release carries the channel-(a) flag.
- **Roadmap Stage 14.** The seasonal-adjustment file store fetches exactly the copies [`archive-inventory.csv`](inventory/archive-inventory.csv) marks `bls_current` or `internet_archive`, so its coverage equals this inventory, and it records every other release as missing rather than silently absent.
- **Roadmap Stage 3.** [`es-vintages.csv`](inventory/es-vintages.csv) lists release dates only. Stage 3's release-date index adds closing and publication dates and must agree with it on every release, including the September 2025 release on 2025-11-20 and the absence of an October 2025 release.
- **The seasonal-flag panel of Req 3.** Calendar regressors and specification regimes are observable release by release only where specification files survive. Elsewhere the model specification tables in the benchmark articles are the fallback, and they cover 2003 through 2013.

## Draft disagreements resolved

Req 20 names five disagreements among the research drafts. Each subsection states what each draft says, cites the primary sources that settle the question, and gives the ruling, its evidence label, and its consequence for later stages.

### Collection rate versus response rate

**Status:** in progress — plan 2, Task 5.

### The December 2018 to January 2019 lapse in appropriations

**Status:** in progress — plan 2, Task 7.

### Whether the 2025 program cuts reached CES

**Status:** in progress — plan 2, Task 6.

### FTE concepts: authorized versus actual, and FTE versus headcount

**Status:** in progress — plan 2, Task 7.

### The 2024 and 2025 preliminary and final benchmark revisions

**Status:** in progress — plan 2, Task 6.

## Annotated bibliography

**Status:** stub — roadmap Stage 23 writes this section: full citations, each with a one-line note on its relevance and labeled peer-reviewed, agency documentation, or grey literature.
