# Seasonal-adjustment file store — Design Spec

> For agentic workers: REQUIRED NEXT SKILL: writing-plans. This spec is one plan.

Status: approved design, 2026-11-16.\
Source: the deferred item "Keep each `bls_current` copy as evidence across `captures` runs" in
`specs/deferred_items.md` (section `claude/deferred-063daa Step 4b review — 2026-11-15`), picked
as a Design item in the `/deferred` pass of 2026-11-16.\
Owner decisions: the nine in [Decisions](#decisions), all of 2026-11-16.

## Intent

`scripts/archive_inventory.py captures` records every copy of BLS's two seasonal-adjustment ZIPs:
the copy that BLS serves now and every copy that the Internet Archive holds. Each run rewrites
`docs/inventory/archive-captures.csv` from scratch and keeps no ZIP. A copy fetched from BLS's
live URL therefore exists only until the next run. When BLS replaces the file, nobody can fetch
those bytes again.

This already cost one release. The run of 2026-11-15 dropped the BLS copy of
`ces.spec.other.zip` fetched on 2026-09-13. Release 2026-08 went from archive-complete to
`not_archived`, and its prior-adjustment and outlier files are now lost. No Internet Archive copy
falls in its window, and no local copy exists.

Outcome O10 of `specs/ces-revision-drivers-roadmap.md` reproduces BLS's X-13 seasonal factors
for the 29 archive-complete releases. It needs the files themselves, not only their fingerprints.
Release 2026-10 is one of the 29, and it rested only on BLS's current file.

The fix keeps every recorded copy as dated evidence and stores every ZIP that `captures` opens in
a committed, content-addressed file store. A light command also makes it cheap to fetch BLS's
files after each monthly release, so that future releases stop falling into gaps.

## Background

How `captures` works today (plan 2, `specs/plans/completed/2-ces-revisions.md`, and the
member-date change of 2026-11-15):

1. It reads BLS's Employment Situation index into `docs/inventory/es-vintages.csv`.
2. For each artifact (`all_employees_specifications` = `ces.spec.ae.zip`, `other_inputs` =
   `ces.spec.other.zip`), it fetches BLS's live file as one `bls_current` row. It then lists
   every Internet Archive copy through the CDX index and downloads each copy with status 200.
3. `inspect_zip` fingerprints the specification, prior-adjustment and outlier members by name and
   CRC-32. `newest_member` records the latest member time. Then the ZIP is discarded.
4. It writes all rows to `archive-captures.csv`, which replaces the earlier file.

`inventory` is offline. It credits each copy to a release window and writes
`docs/inventory/archive-inventory.csv` and the generated blocks of
`docs/ces-revisions-review.md`. It credits specification copies by copy time under annual
benchmark windows, and prior-adjustment and outlier copies by member date (`member_release`).

Facts found while writing this design:

- The rescue fetch of 2026-11-16 (owner approved) saved both current ZIPs outside the repo, in
  `/Users/lowell/Projects/ces-sa-rescue/2026-11-16/`. `ces.spec.other.zip` is 85,005 bytes with
  digest `GVSJRQMPDUNDJSKLZHWEUAMDANPU7KOU`, and `ces.spec.ae.zip` is 89,460 bytes with digest
  `UMPGBDVBRNEGZEZI5FGRTGMFDVXETU4X`. Both equal the digests of the 2026-11-15 `bls_current`
  rows, so they are the same bytes. `fetch-record.json` beside them holds each file's SHA-256,
  `Last-Modified` header and fetch time.
- The committed CSV has 40 distinct opened payloads (30 of `ces.spec.other.zip`, 10 of
  `ces.spec.ae.zip`). At about 85 KB each, storing all of them takes about 3.5 MB. Each later
  month adds about one new payload.
- The two rows dropped on 2026-11-15 survive in commit `1f1061a`. The 2026-09-13
  `ces.spec.ae.zip` row has digest `UMPG…`, the same bytes as the rescued file. The 2026-09-13
  `ces.spec.other.zip` row has digest `JRW4ECRTZ3X22BDLK7YHUZ6U3M7F5WUG`, fingerprints
  `70f4db3c9e3b4761` and `635b9e68f7634c73`, and no member date. Nobody holds its bytes.
- `data/raw/` can't hold the store. `vintage_sources.py fetch` rebuilds
  `data/raw/manifest.csv` from its own download list, and `refresh_manifest` fails on any file it
  has no record of (`tests/test_vintage_sources.py` pins this).
- Three things assume that `bls_current` means "current": the review's Method prose ("when BLS
  serves the file today"), the comment on `Capture.url` ("where these bytes can be fetched
  again"), and `_status_cell`, which links each cell to BLS's live URL.
- `archive_inventory.user_agent()` puts `BLS_CONTACT_EMAIL` in every request, including each CDX
  listing and replay at web.archive.org. `vintage_sources.user_agent_for` sends it to BLS hosts
  only, and it calls `archive_inventory.user_agent()` for them. Stage 4's
  `annual_sources.user_agent(contact_email)` puts it in every request too, and
  `data/annual/raw/source-catalog.csv` points 39 sources at web.archive.org, 2 at
  fraser.stlouisfed.org and 2 at www.dol.gov. `tests/test_annual_sources.py` pins that format,
  and `CLAUDE.md` describes it ("includes it in every HTTP `User-Agent`").
- No code checks that a replayed Archive copy's SHA-1 equals its CDX digest. The review's prose
  says every copy matched on 2026-11-15.

## Goals

1. No run of any command drops a recorded row from `archive-captures.csv`.
2. Every ZIP that a run opens is stored in `data/seasonal/raw/`, named by its digest, before the
   CSV row that references it is written.
3. Every release that the inventory marks `bls_fetched` or `internet_archive` resolves to a
   stored ZIP, so O10 needs no network for its inputs. The only exceptions are Archive copies
   that a full run could not download again (see [Merge rule](#merge-rule)). A test names them,
   and the list is empty after a clean run.
4. A light command fetches only BLS's two ZIPs and the release index, so a fetch after each
   monthly release takes seconds.
5. The contact address goes only to BLS hosts, from every fetcher in the repo:
   `archive_inventory.py`, `vintage_sources.py` and `annual_sources.py`.
6. The 2026-11-15 rows get their payloads from the rescued files, and the two 2026-09-13 rows
   come back.

## Non-goals

- Other ZIPs on BLS's page (`ces.spec.aehe.zip`, `ces.spec.nonae.zip` and the second-preliminary
  sets). Plan 2 scoped them out.
- Scheduling the light command. The owner runs it by hand after each release.
- A per-release index of input files for X-13, the seasonal-flag panel, and the X-13
  reproduction itself. These stay with parked Stages 14 and 25 and outcome O10.
- Changes to how `inventory` credits copies to windows.
- The deferred item "show the copies behind a `conflicting_captures` cell". It keeps its own
  revisit condition.

## Anti-goals

The design fails, even if it runs, if any of these happens:

- A run silently drops or changes a row whose ZIP is stored.
- The store accepts bytes whose digest no recorded row names.
- A test touches the network.
- The contact address reaches any host other than BLS.
- A stored ZIP can change without a test failing.

## Constraints

- The store follows the `data/raw/` rules: committed, manifest-hashed, and written only by its
  script. Nobody edits it by hand.
- Network commands run only with the owner's explicit yes, each time. The plan can't include a
  network run as an unattended step.
- `vintage_sources.py` imports `archive_inventory`, so `archive_inventory` must not import
  `vintage_sources`. The shared host rule for the User-Agent belongs in `archive_inventory`, and
  `annual_sources.py` imports it from there as `vintage_sources.py` does.
- `CLAUDE.md` forbids hand edits of the CSVs in `docs/inventory/` and of the generated blocks.
  The one-time migration below runs as code.
- Timing: the rescued files protect release 2026-10. The next window that needs a fetch opens at
  BLS's December release and closes at the January 2027 release. Neither date was confirmed
  offline. The light command should land before that window closes. Otherwise release 2026-11
  depends on the Internet Archive.

## Approaches

How a run keeps earlier rows:

- **A. One CSV, merged on every run (chosen).** Each network command reads the existing
  `archive-captures.csv`, keeps every earlier row, and adds the new ones. One evidence file and
  one merge rule cover both sources. An Internet Archive row whose copy the Archive later
  deletes stays, because its ZIP is stored. The cost is a merge rule that needs care.
- **B. A BLS log and an Archive snapshot.** BLS fetches go to an append-only `bls-fetches.csv`,
  and `archive-captures.csv` stays a fresh rebuild of the CDX listing. The BLS guarantee is
  trivial to test. But stored Archive ZIPs still need an Archive-side merge, or a deleted copy
  leaves an orphan ZIP. `inventory` and the tests would read two files. Rejected: once Archive
  ZIPs are stored, every row is worth keeping, so B needs a merge anyway.
- **C. A dated snapshot per run, unioned by `inventory`.** Rejected without a vote: it repeats
  about 70 Archive rows on every run.

Where the ZIPs live (the item's open decision):

- **`data/seasonal/raw/` with its own manifest (chosen).** It mirrors Stage 4's
  `data/annual/raw/`, leaves Stage 3's fetch untouched, and becomes parked Stage 14's file store.
- **Inside `data/raw/`.** Rejected: `vintage_sources.py fetch` and `manifest` would have to carry
  rows they don't fetch, which couples Stage 3's refresh to `captures`.
- **`docs/inventory/payloads/`.** Rejected: it puts binary data in `docs/`, which holds none
  today.

## Decisions

The owner made these on 2026-11-16:

1. **Rescue.** Fetch both current ZIPs from bls.gov with the contact address, outside the repo.
   Done, and both match their committed digests.
2. **Scope.** Store every ZIP that `captures` opens, BLS and Internet Archive copies alike, not
   only the BLS copies.
3. **Cadence.** Add a light fetch command. The owner runs it after each release. Scheduling it
   stays out of scope.
4. **Location.** A new committed `data/seasonal/raw/` with its own `manifest.csv`.
5. **The 2026-09-13 rows.** Restore both. The specifications row gets its stored ZIP. The
   other-inputs row gets status `payload_lost`, so it credits nothing: 2026-08 stays
   `not_archived`, and the archive-complete count stays at 29.
6. **Email.** The User-Agent fix is part of this spec.
7. **Approach.** A, one merged CSV.
8. **A copy that a full run can't download again.** Keep, credit, retry. A committed Archive
   row with status `200` and no stored ZIP stays unchanged and keeps its credit when a run can't
   get its bytes. The run stores everything else, exits 1, and names those rows. A later run
   retries them. A row is never demoted, so a throttled run can't lower the count of 29.
9. **Stage 4's fetcher.** `annual_sources.py` moves to the same BLS-only host rule in this spec.

## Design

### The file store

```text
data/seasonal/raw/
  manifest.csv              file, sha256, bytes, digest
  ces.spec.ae/<digest>.zip
  ces.spec.other/<digest>.zip
```

- `<digest>` is the base-32 SHA-1 of the ZIP's bytes, the same value as the CSV's `digest`
  column and the Internet Archive's own digest. A row links to its ZIP with no new key.
- The folder is the ZIP's name without `.zip`, taken from each `Artifact`'s current URL. Rows
  from the pre-2012 FTP keys of an artifact use the same folder.
- `manifest.csv` has one row per stored ZIP: the path relative to `data/seasonal/raw/`, the
  SHA-256, the byte count, and the digest. Rows are sorted by path.
- A ZIP is stored once per digest. A second copy with the same bytes reuses it. Revisit rows
  (status `-`) and every repeat of a digest resolve to that one file.
- A stored ZIP never changes or disappears. Writing a different payload under an existing name
  is an error.
- `.gitattributes` gets `data/seasonal/raw/** -text` and `data/seasonal/raw/*/*.zip -diff`, as
  `data/annual/raw/` has, so `manifest.csv` diffs as text (owner, plan 14, 2026-11-16).
- Each network command stages its new ZIPs, the merged CSV and the new manifest in a temporary
  folder under `data/seasonal/`. It promotes them only when the run has finished, in the same
  way as `vintage_sources.promote_tree`. ZIPs are promoted before the CSV and the manifest, so
  no committed row references a missing file.

### Evidence rows and labels

- **Source label.** `bls_current` becomes `bls_fetched`: a copy fetched from BLS's live URL at
  the row's `timestamp`. The inventory status of the same name means "credited to a copy fetched
  from BLS and held in the store". The label changes in the code, both CSVs, the tests and the
  review's prose. The text of the deferred item is not changed.
- **`url`.** Its meaning becomes "where these bytes were fetched". For an Internet Archive row,
  that URL still serves the same bytes. For a `bls_fetched` row, it doesn't once BLS replaces
  the file. The stored ZIP is the evidence.
- **Inventory evidence.** For a `bls_fetched` status, the `*_evidence` column of
  `archive-inventory.csv` holds the stored ZIP's repo-relative path. The review links it as
  `../data/seasonal/raw/…`. An `internet_archive` status keeps the replay URL.
- **One row per fetch.** Each BLS fetch adds a row, even when its digest repeats an earlier one.
  A specification copy is credited by copy time, so each fetch can credit a window.
- **New status `payload_lost`.** It marks a row whose copy was opened but whose bytes nobody
  holds. Only the restored 2026-09-13 other-inputs row has it. `file_status` already counts only
  statuses `200` and `-`, so this row credits nothing. It records no member date.
- **Preference.** When a BLS copy and an Archive copy both credit a window with the same
  fingerprint, `file_status` keeps preferring the BLS copy, as it does today.

### Merge rule

Each network command merges its new observations into the committed rows:

1. A row's key is `(artifact, source, url, timestamp)`.
2. A committed row whose key the run doesn't observe stays unchanged. This covers every earlier
   BLS fetch and any Archive copy that the CDX index no longer lists.
3. A committed row with a stored ZIP or status `payload_lost` never changes. The one exception
   is `adopt` (below).
4. A committed row with status `200` and no stored ZIP changes only when the run opens its copy
   and the bytes hash to the row's digest. The row then gets its ZIP, and the fingerprints and
   member date that the run computes must equal the row's. If the run can't get the bytes, the
   row stays unchanged and keeps its credit (decision 8). This covers a copy that the CDX index
   no longer lists, a replay that fails every attempt, and bytes that don't match the digest.
   The run still promotes everything else, then exits 1 and names those rows.
5. A committed row with status `301`, `403`, `-` or `unreplayable` takes the new observation.
   This is how an `unreplayable` copy that later opens becomes status `200`.
6. A CDX listing whose digest for a key differs from the committed row's digest stops the run
   with an error. It writes nothing. This is about the listed digest. Replayed bytes that don't
   match the digest fall under point 4.
7. Rows the CDX index lists twice stay as two rows. The 2017-01-02 redirects are listed twice
   today, so the merge matches rows as a multiset. If the Archive listing is unchanged, the
   Archive rows come out byte for byte the same.
8. Output rows are sorted deterministically: by artifact, then source, then timestamp, then URL.

A replayed Archive copy counts as opened only when its SHA-1 equals the CDX digest. A mismatch
counts as a failed attempt, so `retrying` tries again. After the last attempt, a new row gets
status `unreplayable`, and a committed row follows point 4.

### Commands

- **`captures`** (network, full run). It refreshes `es-vintages.csv`, fetches BLS's two ZIPs,
  lists and opens every Archive copy, stores every opened ZIP, and merges. It exits 1 when a
  committed `200` row still has no stored ZIP (merge point 4), and also when a copy holds
  unexplained members, as today. In both cases it promotes its results first.
- **`current`** (network, light run). It refreshes `es-vintages.csv`, fetches BLS's two ZIPs,
  stores them, and appends one `bls_fetched` row per artifact. It sends no request to the
  Internet Archive and leaves every Archive row byte for byte the same.
- **`adopt <zip>…`** (offline). It stores a local ZIP only when its digest equals the digest of
  a recorded row, and that digest has no stored ZIP yet. It refuses anything else and adds no
  row. It checks that the fingerprints and the member date it computes equal the row's. A row
  that recorded no member date (the 2026-09-13 rows) gets the computed one. If the lost `JRW4…`
  bytes ever turn up, `adopt` stores them and sets that row's status from `payload_lost` to
  `200`. This is the only change merge point 3 allows.
- **`inventory`** (offline). This is unchanged except for the new labels and evidence links. It
  reads only the CSVs and does not open the store.

Every fetcher in the repo sends the contact address only to bls.gov hosts. The host rule moves
into `archive_inventory` (for example a `user_agent(url)`). `vintage_sources.user_agent_for` and
`annual_sources.py`'s fetch both call it, and every other host gets `ces-revisions/0.1.0`.

### One-time migration

The plan runs these offline steps as code, in this order, before any network run. Steps 1 and 2
land in one commit. Between them, the restored `ces.spec.ae.zip` row has status `200` and no
member date, which fails `test_only_an_opened_copy_records_a_member_date`.

1. Restore the two 2026-09-13 rows from commit `1f1061a`, relabeled `bls_fetched`. The
   other-inputs row gets status `payload_lost`.
2. Run `adopt` on the two rescued files. `UMPG…` becomes the payload of both `ces.spec.ae.zip`
   rows, of 2026-09-13 and 2026-11-15. `GVSJ…` becomes the payload of the 2026-11-15
   `ces.spec.other.zip` row.
3. Run `inventory`. Release 2026-10 stays archive-complete, now with a stored-ZIP link. The
   count stays at 29.

After step 3, the Archive rows with status 200 still have no stored ZIP. The first full
`captures` run with the owner's yes stores them (see [Sequence](#sequence)).

### Documents

- `docs/ces-revisions-review.md`, hand-written parts:
  - Method step 1 says that `captures` stores each ZIP it opens and keeps earlier rows. It
    describes `current` and `adopt`.
  - The status list defines `bls_fetched` and `payload_lost` in place of `bls_current`.
  - The coverage caption uses the new label.
  - Counts in the Method prose that the restored rows or the full run change ("In all 33
    copies…") are updated after the full run.
  - The Stage 14 bullet under "Consequences for later stages" says the file store exists in
    `data/seasonal/raw/`, and that Stage 14 builds its flag panel on it.
  - The release 2026-08 loss is stated once, with its `payload_lost` row.
- `CLAUDE.md`: a paragraph for `data/seasonal/raw/`, with the same rules as `data/raw/`, and
  `current` and `adopt` in the Commands block. The sentence that says `annual_sources.py fetch`
  puts the address "in every HTTP `User-Agent`" changes to the BLS-only rule.

## Tests

The tests that show the item is done:

1. **Two runs keep a release's status (hermetic).** Run a network command with BLS serving ZIP X
   at time t1, then again with BLS serving ZIP Y at t2. Monkeypatch `fetch` and the listing.
   After the second run, the t1 row is byte for byte unchanged, ZIP X is in the store, and the
   release that X credits has the same status and evidence as after the first run.
2. **Committed evidence.**
   - Every row with status `200` resolves to a stored ZIP, except the rows in a pinned list of
     copies that a full run could not download again. The list is empty after a clean run, and
     any change to it fails the test.
   - Every stored ZIP is referenced by at least one row.
   - Each stored ZIP's SHA-1 equals its name, and its SHA-256 and size equal its manifest row.
   - The manifest lists exactly the files present.
   - Every present cell in `archive-inventory.csv` resolves to a stored ZIP.
   - Release 2026-10 is archive-complete with `bls_fetched` evidence.
   - Both 2026-09-13 rows are present, and the other-inputs row has status `payload_lost`.

Other tests:

- Merge rule: each numbered point above. This includes the digest-mismatch error, the
  `unreplayable` to `200` change, a committed `200` row that keeps its credit when its replay
  fails or its copy is no longer listed, and the multiset match of rows listed twice.
- Replay check: bytes whose SHA-1 differs from the CDX digest count as a failed attempt.
- `current`: it makes no Internet Archive request, appends one row per artifact, and leaves the
  Archive rows unchanged.
- `adopt`: it stores a matching ZIP, refuses an unknown digest and an already stored one, and
  adds no row.
- Store: the staged write leaves the committed tree untouched when a run fails part way.
- User-Agent: bls.gov hosts get the contact, and web.archive.org, fraser.stlouisfed.org and
  www.dol.gov get none, from all three fetchers. The pinned format in
  `tests/test_annual_sources.py` changes to match.
- The existing tests keep passing after the label change, including
  `test_every_copy_was_made_after_the_release_its_files_were_made_for` and
  `test_only_an_opened_copy_records_a_member_date`.

## Sequence

1. Build, migration and tests, all offline. The committed-evidence test of every `200` row
   waits for step 2, because the Archive ZIPs are not stored until then. Either it lands with
   step 2, or it is written to skip rows from before the store until step 2 lands. The planner
   picks.
2. The first full `captures` run, with the owner's explicit yes. It stores about 40 ZIPs, about
   3.5 MB. Commit the store, the CSVs, the regenerated blocks and the updated prose counts
   together.
3. From then on, the owner runs `current` after each monthly release and then `inventory`. The
   first such window opens at BLS's December 2026 release.

After step 2, the rescue folder `/Users/lowell/Projects/ces-sa-rescue/` can be deleted by the
owner. Nothing in the repo reads it.

## Relation to parked stages and O10

- **Parked Stage 14** produces "the seasonal-adjustment file store (specification,
  prior-adjustment, and outlier files per vintage, content-hashed, coverage scoped by the
  archive inventory)". This spec delivers the store and its content hashing. Every credited copy
  is stored, so the store's coverage contains the archive inventory's coverage. Stage 14 keeps
  the seasonal-flag panel, the birth–death columns, the interventions table, and any per-release
  index of input files. Its member-date exit point was closed on 2026-11-15.
- **Parked Stage 25** consumes "the seasonal-adjustment file store" from Stage 14. It gets it
  from this spec instead, at the same path. Nothing in Stage 25's text conflicts.
- **O10** reads its inputs from `data/seasonal/raw/` through `archive-inventory.csv` and
  `archive-captures.csv`, with no network.

## Left to the builder

The design does not settle these. A sensible builder decides them without changing what gets
built:

- Function and module names, and whether the store code stays in `archive_inventory.py` or
  becomes a small module beside it.
- How the restore step reads the two rows from commit `1f1061a`.
- The exact form of the staged write.
- How the review renders a `bls_fetched` link.
- Whether the step 1 committed-evidence test is deferred or written with a pre-store exception.

Calls made in this design, which the owner can still overrule:

- The label `bls_fetched` and the status `payload_lost`.
- The command names `current` and `adopt`.
- The merge key and sort order, and the multiset match of rows that the CDX index lists twice.
- `inventory` stays CSV-only and does not open the store.
- A replayed Archive copy must hash to its CDX digest, or the attempt counts as failed.
- A run that finds unexplained members or rows without a ZIP still promotes, then exits 1.
- `adopt` may turn the `payload_lost` row back into a `200` row if its bytes turn up.
