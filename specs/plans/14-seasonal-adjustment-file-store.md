# Seasonal-Adjustment File Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via subagent-driven-development (the default) — or executing-plans when your human partner chose inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep every recorded copy of BLS's two seasonal-adjustment ZIPs as dated evidence. Store every ZIP that `captures` opens in a committed, content-addressed store, add the light `current` and the offline `adopt` commands, and send the contact address only to BLS hosts.

**Architecture:** The store, merge and command code stays in `scripts/archive_inventory.py`. That module keeps importing only the standard library, so `adopt` can read the rescued ZIPs under `python -I`. A network run first collects its observations, then merges them into the committed rows (`merge_captures`), which raises before anything is written. It then stages the new ZIPs, `es-vintages.csv`, the merged `archive-captures.csv` and the store manifest in a temporary folder under `data/seasonal/`, and promotes ZIPs before the files that name them. The BLS label lives in one constant, `BLS_SOURCE`, so Task 6 flips `bls_current` to `bls_fetched` in code and data in one commit.

**Tech Stack:** Python 3.14 standard library (`zipfile`, `hashlib`, `csv`, `tempfile`, `urllib.request`), pytest, ruff. No new dependency.

**Spec:** `specs/seasonal-adjustment-file-store.md`, approved 2026-11-16. References such as `spec:228` are its line numbers, and `review:131` is a line of `docs/ces-revisions-review.md` at this plan's base. Executors read the spec and this plan.

## Global Constraints

- **Network:** a network command runs only after the owner's explicit yes, each time (`spec:122-123`, D2). Only Task 8 runs one. No test touches the network. New tests replace `archive_inventory.fetch`, `archive_inventory.list_captures` and `archive_inventory.sleep`, and replace `urllib.request.urlopen` where they test headers.
- **Imports:** `archive_inventory` imports only the standard library and never imports `vintage_sources` (`spec:124-126`). `vintage_sources.py` and `annual_sources.py` take the host rule from it.
- **Contact address:** only `bls.gov` and its subdomains get `BLS_CONTACT_EMAIL` (`spec:92-93`, `spec:275-277`). Never print or commit the address.
- **No hand edits:** never hand-edit `docs/inventory/*.csv`, the review's generated blocks, or anything under `data/seasonal/raw/` (`spec:120-121`, `spec:127-128`). A one-time change runs as code.
- **Mismatches:** where the spec says computed fingerprints or member dates "must equal" a row's, a mismatch in a network run raises `ValueError` naming the row's URL, and the run writes nothing. `adopt` refuses instead: it prints the reason to stderr, returns 1 and writes nothing.
- **Leave alone:** the text of the parked Stage 14 item in `specs/deferred_items.md` (D1). Also leave the text of the item "Keep each `bls_current` copy as evidence across `captures` runs", apart from its markers (`spec:212-213`).
- **Timing:** `current` must reach `main` before BLS's January 2027 release (`spec:129-132`).
- **Branch:** start the executing worktree from `claude/captures-evidence-design-0fce4f`, not from `main`. That branch carries the spec commits `3827a64` and `5867868` and this plan. Never commit in the main checkout, `/Users/lowell/Projects/ces-revisions`, because other sessions switch its branch.
- **Contact file:** `.project.env` is gitignored and exists only in the main checkout. Network commands read it by its absolute path, `/Users/lowell/Projects/ces-revisions/.project.env`, through `uv run --env-file`. uv stops with an error if the file is missing.
- **Style:** stage files by explicit path. Run `uv run ruff format` and `uv run ruff check` on the files each commit touches. Markdown follows CLAUDE.md's rules for `specs/`. The fast suite, `uv run pytest -m "not slow and not network"`, passes at every commit.

## Review Focus

1. A throttled full run. A committed `200` row whose replay fails keeps its row and its credit, and the archive-complete count does not fall. The run promotes everything else, exits 1 and names the row (Task 4).
2. The contact address leaking through a replay URL whose path holds `www.bls.gov`, or through a lookalike host such as `bls.gov.example.com` (Task 1).
3. BLS serving the same bytes on two runs: two rows, one ZIP and no "different payload" error (Task 4).
4. A run that fails part way, in the network loop or while staging. The committed CSVs, the store and its manifest stay byte for byte the same, and no `.staging-*` folder is left (Tasks 2 and 4).
5. An unchanged CDX listing that includes doubled rows, as the 2017-01-02 redirects are today. The Archive rows come out byte for byte the same (Task 4).

## Decisions

- **D1** (owner, brief, 2026-11-16): the parked Stage 14 item's text in `specs/deferred_items.md` stays unedited. The spec's section "Relation to parked stages and O10" records which half of Stage 14 this plan delivers. Applied in: Global Constraints, Task 9.
- **D2** (owner, brief, 2026-11-16): each network step needs the owner's yes each time: the first full `captures` run with the new code, and `current` after each release. Applied in: Global Constraints, Task 8.

## Files

- `scripts/archive_inventory.py`: the host rule (Task 1), a `File store` section (Task 2), a `Merge` section (Task 3), the reworked capture collection and commands (Tasks 4 and 5), and the label and evidence switch (Task 6).
- `scripts/vintage_sources.py` and `scripts/annual_sources.py`: they call the host rule (Task 1).
- `tests/zip_fixtures.py`: new. It holds `make_zip` and `make_dated_zip`, moved from `tests/test_archive_inventory.py:58-70` (Task 2).
- `tests/test_archive_store.py`: new. It covers the store, the merge rule and the commands, all hermetic (Tasks 2 to 5).
- `tests/test_archive_inventory.py`: the host-rule tests (Task 1), the rewritten unreplayable test (Task 4), and the committed-evidence tests of the store (Task 6).
- `data/seasonal/raw/`: new committed store (Task 6), filled by the full run (Task 8).
- `.gitattributes`, `.gitignore`, `CLAUDE.md`, `docs/ces-revisions-review.md`: Tasks 1, 2, 7 and 8.

---

### Task 1: Send the contact address only to BLS hosts

**Files:**
- Modify: `scripts/archive_inventory.py:187-199` (Network section)
- Modify: `scripts/vintage_sources.py:127-132`
- Modify: `scripts/annual_sources.py:22`, `scripts/annual_sources.py:34-45`
- Modify: `CLAUDE.md:36`
- Test: `tests/test_archive_inventory.py`, `tests/test_vintage_sources.py:36-49`, `tests/test_annual_sources.py:24-32`

**Interfaces:**
- Produces: `archive_inventory.USER_AGENT_PRODUCT = "ces-revisions/0.1.0"`, `archive_inventory.is_bls_host(url: str) -> bool` and `archive_inventory.user_agent(url: str, contact: str | None = None) -> str`. `contact=None` reads `BLS_CONTACT_EMAIL` from the environment. `annual_sources.user_agent` and `annual_sources.USER_AGENT_PRODUCT` are removed.

- [ ] **Step 1: Write the failing tests**

In `tests/test_archive_inventory.py`, under `# --- Network ---`:

```python
WITH_CONTACT = "ces-revisions/0.1.0 (someone@example.org)"
BARE = "ces-revisions/0.1.0"
REPLAY = "https://web.archive.org/web/20140719155542id_/http://www.bls.gov/web/empsit/ces.spec.ae.zip"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.bls.gov/web/empsit/ces.spec.ae.zip", WITH_CONTACT),
        ("https://download.bls.gov/pub/time.series/ce/ce.series", WITH_CONTACT),
        ("https://bls.gov/", WITH_CONTACT),
        (REPLAY, BARE),
        (
            "https://web.archive.org/cdx/search/cdx?url=bls.gov/web/empsit/ces.spec.ae.zip",
            BARE,
        ),
        ("https://bls.gov.example.com/", BARE),
        ("https://fraser.stlouisfed.org/title/employment-situation-4910", BARE),
        ("https://www.dol.gov/agencies/oasp", BARE),
    ],
)
def test_only_bls_hosts_get_the_contact_address(monkeypatch, url, expected):
    monkeypatch.setenv("BLS_CONTACT_EMAIL", "someone@example.org")
    assert archive_inventory.user_agent(url) == expected
    monkeypatch.delenv("BLS_CONTACT_EMAIL")
    assert archive_inventory.user_agent(url, "someone@example.org") == expected
    assert archive_inventory.user_agent(url) == BARE
```

Add `test_fetch_sends_the_contact_to_bls_hosts_only(monkeypatch)`. It replaces `urllib.request.urlopen` with a fake that records `request.get_header("User-agent")` and returns an empty response. It then calls `archive_inventory.fetch` on the bls.gov ZIP URL and on `REPLAY`, and asserts the recorded headers are `[WITH_CONTACT, BARE]`.

In `tests/test_annual_sources.py`, replace the `user_agent` assertion at lines 30-32 with `test_annual_downloads_send_the_contact_to_bls_hosts_only(monkeypatch)`. It uses the same fake `urlopen` and calls `annual_sources.download(url, "someone@example.org")` for `annual_sources.QCEW_REVISIONS_URL`, a `fraser.stlouisfed.org` URL, a `www.dol.gov` URL and a `web.archive.org` replay URL. The headers are `[WITH_CONTACT, BARE, BARE, BARE]`. Keep the dotenv assertion at line 29.

In `tests/test_vintage_sources.py:36-49`, add `assert vintage_sources.user_agent_for(REPLAY_URL) == "ces-revisions/0.1.0"`, where `REPLAY_URL` is the replay URL above.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_archive_inventory.py tests/test_annual_sources.py tests/test_vintage_sources.py -m "not network" -q`
Expected: FAIL. `user_agent()` takes no `url`, and `annual_sources.download` sends the contact to every host.

- [ ] **Step 3: Implement the host rule**

`is_bls_host` takes `urlsplit(url).hostname` and accepts `bls.gov` or a name that ends in `.bls.gov`. Move that test from `vintage_sources.user_agent_for` into it. `user_agent` returns `f"{USER_AGENT_PRODUCT} ({contact})"` only for a BLS host and a non-empty contact. `fetch` sends `user_agent(url)`. `vintage_sources.user_agent_for(url)` returns `archive_inventory.user_agent(url)`. `annual_sources.download(url, contact_email)` sends `archive_inventory.user_agent(url, contact_email)`. `load_contact_email` and its checks stay.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_archive_inventory.py tests/test_annual_sources.py tests/test_vintage_sources.py -m "not network" -q`
Expected: PASS.

- [ ] **Step 5: Update `CLAUDE.md:36`**

Replace "and includes it in every HTTP `User-Agent`" with "and sends it in the `User-Agent` of requests to bls.gov hosts only, through `archive_inventory.user_agent`. Every other host gets `ces-revisions/0.1.0`".

- [ ] **Step 6: Commit**

```bash
git add scripts/archive_inventory.py scripts/vintage_sources.py scripts/annual_sources.py tests/test_archive_inventory.py tests/test_vintage_sources.py tests/test_annual_sources.py CLAUDE.md
git commit -m "Send the contact address to BLS hosts only, from all three fetchers"
```

### Task 2: The content-addressed file store

**Files:**
- Modify: `scripts/archive_inventory.py` (new section `# --- File store ---` after `# --- Captures ---`)
- Modify: `.gitattributes`, `.gitignore`
- Create: `tests/zip_fixtures.py`, `tests/test_archive_store.py`
- Modify: `tests/test_archive_inventory.py:58-70` (import the ZIP builders from `zip_fixtures`)

**Interfaces:**
- Consumes: `payload_digest(payload: bytes) -> str` (`archive_inventory.py:393`), `ARTIFACTS`.
- Produces:
  - `STORE_DIR = ROOT / "data" / "seasonal" / "raw"`, `STORE_MANIFEST = "manifest.csv"` and `STORE_MANIFEST_COLUMNS = ["file", "sha256", "bytes", "digest"]`.
  - `store_folder(artifact: str) -> str`: the artifact's `current_url` file name without `.zip`, so `"ces.spec.ae"` or `"ces.spec.other"`. An unknown artifact raises `ValueError`.
  - `stored_file(artifact: str, digest: str) -> str`: `f"{store_folder(artifact)}/{digest}.zip"`, relative to the store.
  - `class Staging` with `has(artifact: str, digest: str) -> bool`, `put(artifact: str, payload: bytes) -> str`, `path_for(target: Path) -> Path` and `promote() -> None`.
  - `staging(store_dir: Path) -> Iterator[Staging]`, a `contextlib.contextmanager`.
  - `check_store(store_dir: Path) -> list[str]`: one message per problem, empty when the store is sound.

- [ ] **Step 1: Move the ZIP builders**

Move `make_zip` and `make_dated_zip` unchanged to `tests/zip_fixtures.py`, and import them in `tests/test_archive_inventory.py`. Run `uv run pytest tests/test_archive_inventory.py -m "not network" -q`. Expected: PASS.

- [ ] **Step 2: Write the failing tests in `tests/test_archive_store.py`**

Each test uses `store = tmp_path / "data" / "seasonal" / "raw"`.

- `test_a_stored_zip_is_named_by_its_digest_in_its_artifact_folder`: `put("other_inputs", payload)` inside `staging(store)`, then `promote()`. Assert the returned digest `== payload_digest(payload)`, `(store / "ces.spec.other" / f"{digest}.zip").read_bytes() == payload`, and `stored_file("all_employees_specifications", "D") == "ces.spec.ae/D.zip"`.
- `test_the_same_bytes_are_stored_once`: put the same payload twice in one run and again in a second run. Assert one ZIP file and one manifest row.
- `test_different_bytes_under_a_stored_name_are_an_error`: write other bytes at the name the payload's digest gives, then `put`. Assert `pytest.raises(ValueError, match="different payload")`.
- `test_the_manifest_keeps_committed_rows_and_adds_staged_zips`: run 1 stores A and run 2 stores B. The manifest's header is `STORE_MANIFEST_COLUMNS`. Its rows are sorted by `file`, and each row has `sha256` equal to `hashlib.sha256(payload).hexdigest()`, `bytes` equal to `str(len(payload))` and `digest` equal to `payload_digest(payload)`. A's row is byte for byte what run 1 wrote.
- `test_promote_moves_zips_before_the_files_that_name_them`: wrap `pathlib.Path.replace` to record each target. Stage one ZIP and `path_for(tmp_path / "docs" / "inventory" / "archive-captures.csv")`, then promote. Assert every `.zip` target comes before every `.csv` target.
- `test_a_failed_run_leaves_the_committed_tree_untouched`: commit one ZIP and a CSV in an earlier run. In a new run, put another ZIP, write a staged CSV, then raise `RuntimeError` inside the `with`. Assert the store's files, the manifest bytes and the CSV bytes are unchanged, and `list(store.parent.glob(".staging-*")) == []`.
- `test_check_store_finds_a_changed_an_unlisted_and_a_missing_zip`: a sound store gives `[]`. Each of the following gives one message naming the file: one changed byte in a stored ZIP, an extra ZIP not in the manifest, and a deleted listed ZIP.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_archive_store.py -q`
Expected: FAIL with `ImportError` on `staging`.

- [ ] **Step 4: Implement the store**

- `staging` creates `store_dir.parent` if needed and opens `tempfile.TemporaryDirectory(prefix=".staging-", dir=store_dir.parent)`. The folder goes away on any exit.
- `put` computes the digest. If the committed store or this run already holds that name, the bytes must be equal, else it raises `ValueError(f"different payload under {file}")`, and nothing new is staged. Otherwise it writes the ZIP into the staging folder.
- `has` is true for a committed or a staged ZIP.
- `promote` writes the new manifest first. The manifest keeps every committed row as it is, without rehashing, adds one row per staged ZIP, and is sorted by `file`. A missing manifest reads as no rows. `promote` then moves every staged ZIP, and after that every other staged file and the manifest, with `Path.replace`.
- `check_store` reports four problems: a ZIP whose `payload_digest` is not its file name, a manifest row whose `sha256`, `bytes` or `digest` differs from the file, a file that is not in the manifest, and a manifest row with no file. It ignores `.DS_Store`.

- [ ] **Step 5: Add the repository rules**

Add these rules to `.gitattributes`, with a comment:

```text
data/seasonal/raw/** -text
data/seasonal/raw/*/*.zip -diff
```

The rules follow the `data/annual/raw/` pattern, so `manifest.csv` diffs as text. Add `data/seasonal/.staging-*/` to `.gitignore`, with a comment that a killed run can leave one.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_archive_store.py tests/test_archive_inventory.py -m "not network" -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/archive_inventory.py .gitattributes .gitignore tests/zip_fixtures.py tests/test_archive_store.py tests/test_archive_inventory.py
git commit -m "Add the content-addressed store for seasonal-adjustment ZIPs"
```

### Task 3: The merge rule

**Files:**
- Modify: `scripts/archive_inventory.py` (new section `# --- Merge ---` after the file store)
- Test: `tests/test_archive_store.py`

**Interfaces:**
- Consumes: `Capture`.
- Produces: `capture_key(capture: Capture) -> tuple[str, str, str, datetime]`, which is `(artifact, source, url, timestamp)`, and `merge_captures(committed: list[Capture], observed: list[Capture]) -> list[Capture]`.

The rule is `spec:228-251`, in this order:

1. Pair committed and observed rows by `capture_key` as multisets, in their given order (point 7).
2. A pair whose digests differ raises `ValueError` naming the URL (point 6).
3. A committed row with status `200` stays. If the observed status is also `200`, its `specification`, `prior_adjustment`, `outliers`, `unexpected` and `newest_member` must equal the committed row's, else `ValueError` (points 3 and 4).
4. A committed row with status `payload_lost` stays (point 3).
5. A committed row with any other status takes the observed row (point 5).
6. An unpaired committed row stays, and an unpaired observed row is added (point 2).
7. The result is sorted by `(artifact, source, timestamp, url)` with Python's stable sort, so doubled rows keep their order (point 8).

- [ ] **Step 1: Write the failing tests**

Build rows with a helper, `row(**changes) -> Capture`, which applies `replace` to one `internet_archive` row with status `200`, digest `"D"` and `outliers="o"`.

- `test_a_committed_row_the_run_does_not_observe_stays`: `merge_captures([bls, archived], []) == [bls, archived]`.
- `test_an_observed_row_with_a_new_key_is_added`.
- `test_a_committed_opened_row_keeps_its_credit_when_its_replay_fails`: committed `row()` against observed `row(status="unreplayable", outliers="")` gives `[row()]`.
- `test_an_opened_copy_with_other_fingerprints_is_an_error`: committed `row()` against observed `row(outliers="x")` raises `ValueError` matching the URL.
- `test_a_payload_lost_row_never_changes`.
- `test_a_row_without_a_payload_takes_the_new_observation`: parametrize over the committed statuses `"301"`, `"403"`, `"-"` and `"unreplayable"`. The observed row is `row()`, and the result is `[row()]`.
- `test_a_listed_digest_that_differs_from_the_committed_one_is_an_error`: the observed row is `row(digest="E")`.
- `test_rows_listed_twice_pair_as_a_multiset`: with committed `[r, r]`, observing `[r]` gives `[r, r]`, observing `[r, r]` gives `[r, r]`, and observing `[r, r, r]` gives `[r, r, r]`.
- `test_merged_rows_sort_by_artifact_source_timestamp_url`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_archive_store.py -q`
Expected: FAIL with `ImportError` on `merge_captures`.

- [ ] **Step 3: Implement `capture_key` and `merge_captures`**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_archive_store.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/archive_inventory.py tests/test_archive_store.py
git commit -m "Merge each run's copies into the committed evidence rows"
```

### Task 4: `captures` and `current` keep every row and store every opened copy

**Files:**
- Modify: `scripts/archive_inventory.py` (Captures: replace `collect_captures`, `archive_inventory.py:399-450`; Inventory: `PRESENT`, `file_status`; Command line: `archive_inventory.py:754-798`)
- Test: `tests/test_archive_store.py`, `tests/test_archive_inventory.py:361-395` and `tests/test_archive_inventory.py:456-479`

**Interfaces:**
- Consumes: `staging`, `Staging`, `stored_file` (Task 2), `merge_captures` (Task 3).
- Produces:
  - `BLS_SOURCE = "bls_current"`. Task 6 flips it. The capture code, `PRESENT`, `STATUSES` and `file_status`'s preference use it in place of the literal.
  - `fetch_and_inspect(url: str, digest: str | None = None) -> tuple[bytes, dict[str, str]]`. It raises `ValueError` when `digest` is given and `payload_digest(payload) != digest`, so `retrying` counts the attempt as failed (`spec:253-255`).
  - `fetch_vintages(now: datetime) -> list[Vintage]`: the release index through `retrying`, parsed and passed through `select_vintages`.
  - `fetch_bls_copies(now: datetime) -> list[tuple[Capture, bytes]]`: one `BLS_SOURCE` row per artifact, with status `200`, time `now` and its payload.
  - `collect_archive_copies() -> list[tuple[Capture, bytes | None]]`: every CDX row of every key. A `200` row is replayed with the CDX digest check. A replay that fails every attempt becomes `unreplayable`, with no payload.
  - `@dataclass(frozen=True) class EvidencePaths` with the fields `vintages: Path`, `captures: Path` and `store: Path`, and `@classmethod at(cls, root: Path) -> EvidencePaths`. `at` gives `root / "docs/inventory/es-vintages.csv"`, `root / "docs/inventory/archive-captures.csv"` and `root / "data/seasonal/raw"`. `EVIDENCE = EvidencePaths(VINTAGES_PATH, CAPTURES_PATH, STORE_DIR)`.
  - `record_run(observed: list[tuple[Capture, bytes | None]], vintages: list[Vintage], paths: EvidencePaths) -> list[Capture]`. It reads the committed rows (a missing CSV reads as no rows) and calls `merge_captures`. Inside `staging(paths.store)`, it puts each payload, writes `es-vintages.csv` and the merged CSV to `path_for`, and promotes. It returns the merged rows.
  - `capture_evidence(now: datetime, paths: EvidencePaths = EVIDENCE) -> int`: `fetch_vintages`, `fetch_bls_copies`, `collect_archive_copies`, then `record_run`. It prints to stderr each merged row with unexplained members and each `200` row whose `stored_file` is missing from `paths.store`. It returns 1 if it printed any.
  - `fetch_current(now: datetime, paths: EvidencePaths = EVIDENCE) -> int`: `fetch_vintages`, `fetch_bls_copies`, then `record_run`. It sends no Internet Archive request, reports unexplained members only, and returns 1 if there are any.
  - `main` uses argparse subcommands `captures`, `current`, `adopt` (Task 5) and `inventory`. `captures` and `current` pass `datetime.now(UTC).replace(microsecond=0)`. `collect_captures` and `COMMANDS` are removed.

- [ ] **Step 1: Write the failing tests**

In `tests/test_archive_store.py`, add a `fake_network(monkeypatch, *, bls, listing, replays)` helper.

- It replaces `archive_inventory.fetch` with a function that serves three things: `RELEASE_INDEX_URL` from `tests/fixtures/archive_inventory/empsit-index-excerpt.html`, each artifact's `current_url` from `bls`, and each replay URL from `replays`, where a value that is an exception is raised. The function appends each URL to a list that the helper returns.
- It replaces `list_captures` with `listing.get(key, [])`, and replaces `sleep` with a no-op.
- It limits `ARTIFACTS` to the `other_inputs` artifact with its one current CDX key.

Paths come from `EvidencePaths.at(tmp_path)`.

- `test_two_runs_keep_the_first_copy_and_its_release_status` (spec Test 1, `spec:316-319`). Parametrize it over `fetch_current` and `capture_evidence`, with an empty listing, so it shows the deferred item's "Done when" for both commands. ZIP X holds `prior_adjustment_file.xlsx` and `outliers.xlsx`, both dated 2026-09-29, so they were made for the release of 2026-10-02. ZIP Y has the same members with other bytes, dated 2026-10-30. Run the command at 2026-10-05 15:00 UTC serving X, then at 2026-11-10 15:00 UTC serving Y. After the second run, X's CSV line is byte for byte the line after the first run, and both ZIPs are in the store. `inventory_rows` gives reference month 2026-09 the same `prior_adjustment`, `outliers` and `*_evidence` values after both runs.
- `test_current_makes_no_internet_archive_request_and_keeps_the_archive_rows`. The committed CSV holds one Archive row with status `200` and one with status `301`. `list_captures` raises `AssertionError`. After `fetch_current`, no fetched URL has host `web.archive.org`, the two Archive lines are unchanged, and one `BLS_SOURCE` row was added.
- `test_the_same_bls_bytes_on_two_runs_add_two_rows_and_one_zip` (Review Focus 3).
- `test_a_full_run_stores_every_opened_copy`. The listing holds a `200` copy, a `301` and a revisit `-` with the `200` copy's digest. `capture_evidence` returns 0, and every `200` row's `stored_file` exists.
- `test_a_throttled_full_run_keeps_every_credit_and_exits_1` (Review Focus 1). The committed CSV holds Archive row A with status `200`, no stored ZIP, and a member date in the 2026-09 window. The listing lists A and a new copy B, whose files were made for the release of 2026-09-04 (reference month 2026-08), so B does not conflict with A. A's replay raises `OSError`, and B opens. The run returns 1, and stderr names A's URL. A's line is unchanged, B's ZIP is stored, and `inventory_rows` gives A's release the same status before and after.
- `test_replayed_bytes_that_miss_their_digest_count_as_failed_attempts`. The listing gives the digest `"NOTTHEPAYLOAD"` for a `200` copy. Its replay URL is fetched `ATTEMPTS` times, and the row becomes `unreplayable` with no stored ZIP.
- `test_an_unchanged_listing_gives_byte_identical_archive_rows` (Review Focus 5). The listing doubles one `301` row, and `capture_evidence` runs twice. The Archive lines are identical after the two runs, and the doubled row is two lines.
- `test_a_run_that_fails_while_fetching_writes_nothing` (Review Focus 4). `list_captures` raises `RuntimeError`, and `capture_evidence` raises it. The committed CSVs and the store are unchanged, and there is no `.staging-*` folder.
- `test_a_listed_digest_that_contradicts_a_committed_row_writes_nothing`. `capture_evidence` raises `ValueError`, and nothing is written.

In `tests/test_archive_inventory.py:361-395`, rename the test to `test_collect_archive_copies_records_an_unreplayable_copy_instead_of_failing` and call `collect_archive_copies()`. Give the `KEPT` copy the digest `payload_digest(payload)`. The expected rows lose the BLS row, and each item is a `(Capture, payload)` pair. At lines 456-479, use `BLS_SOURCE` in place of the `"bls_current"` literal.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_archive_store.py tests/test_archive_inventory.py -m "not network" -q`
Expected: FAIL with `ImportError` on `EvidencePaths`.

- [ ] **Step 3: Implement the collection functions, `record_run`, both commands and `main`**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -m "not slow and not network" -q`
Expected: PASS. The committed CSVs are not touched.

- [ ] **Step 5: Commit**

```bash
git add scripts/archive_inventory.py tests/test_archive_store.py tests/test_archive_inventory.py
git commit -m "Keep earlier rows and store each opened ZIP in captures and current"
```

### Task 5: `adopt` stores a local copy of a recorded ZIP

**Files:**
- Modify: `scripts/archive_inventory.py` (Command line)
- Test: `tests/test_archive_store.py`

**Interfaces:**
- Consumes: `EvidencePaths`, `staging`, `stored_file`, `inspect_zip`, `newest_member`.
- Produces: `adopt_payloads(zips: list[Path], paths: EvidencePaths = EVIDENCE) -> int` and the subcommand `adopt ZIP [ZIP ...]`.

Behavior (`spec:266-271`):

- `adopt` reads and checks every given ZIP before it stages any.
- The matching rows for a ZIP are the committed rows with its digest and status `200` or `payload_lost`.
- It refuses when there is no matching row, when the matching rows span two artifacts, or when the store or this run already holds `stored_file`.
- For each matching row, `adopt` compares the four fingerprint fields with the computed ones (`specification`, `prior_adjustment`, `outliers`, `unexpected`). It refuses when they differ.
- If the row has a `newest_member`, that value must equal the computed one, else `adopt` refuses. If it is empty, it takes the computed one.
- A matching row with status `payload_lost` becomes `200`.
- Revisit rows and every other row stay unchanged, and `adopt` adds no row.
- A run with no refusal stages the ZIPs and the updated `archive-captures.csv`, then promotes them. It does not write `es-vintages.csv`.

- [ ] **Step 1: Write the failing tests**

- `test_adopt_stores_a_recorded_zip_and_fills_an_empty_member_date`: it returns 0, the ZIP is at `stored_file`, the row has the computed `newest_member`, and the row count is unchanged.
- `test_adopt_refuses_a_digest_no_row_records`: it returns 1, and the CSV bytes and the store are unchanged.
- `test_adopt_refuses_a_digest_already_stored`.
- `test_adopt_refuses_bytes_whose_fingerprints_differ_from_the_row`: the row has the payload's digest but `outliers="0000000000000000"`.
- `test_adopt_turns_a_payload_lost_row_into_200`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_archive_store.py -q -k adopt`
Expected: FAIL with `ImportError` on `adopt_payloads`.

- [ ] **Step 3: Implement `adopt_payloads` and the subcommand**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_archive_store.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/archive_inventory.py tests/test_archive_store.py
git commit -m "Add adopt: store a local copy of a recorded seasonal-adjustment ZIP"
```

### Task 6: Migrate the committed evidence

This task makes one commit, because the label flip, the restored rows and the adopted ZIPs fail the tests apart (`spec:281-291`).

**Files:**
- Modify: `scripts/archive_inventory.py` (`BLS_SOURCE`, the `Capture` field comments at lines 292-294, `file_status`, `_status_cell`)
- Modify: `tests/test_archive_inventory.py` (a new section `# --- Committed evidence: the file store ---`)
- Regenerate: `docs/inventory/archive-captures.csv`, `docs/inventory/archive-inventory.csv` and the generated blocks of `docs/ces-revisions-review.md`
- Create, by `adopt`: `data/seasonal/raw/manifest.csv`, `data/seasonal/raw/ces.spec.ae/UMPGBDVBRNEGZEZI5FGRTGMFDVXETU4X.zip` and `data/seasonal/raw/ces.spec.other/GVSJRQMPDUNDJSKLZHWEUAMDANPU7KOU.zip`

**Interfaces:**
- Produces:
  - `BLS_SOURCE = "bls_fetched"`.
  - `evidence(capture: Capture) -> str`. For `BLS_SOURCE` it is `f"{STORE_DIR.relative_to(ROOT).as_posix()}/{stored_file(capture.artifact, capture.digest)}"`, and for any other source it is `capture.url`. `file_status` returns `evidence(chosen)` and joins `evidence(...)` for `conflicting_captures`.
  - `_status_cell` links a `BLS_SOURCE` cell to `../<evidence>`, and an `internet_archive` cell to its URL as now.
  - The comment on `Capture.url` becomes "where these bytes were fetched" (`spec:213-216`).

- [ ] **Step 1: Write the failing tests**

The new section in `tests/test_archive_inventory.py` holds two constants and seven tests:

- `ARCHIVE_COMPLETE = 29`.
- `AWAITING_A_STORED_ZIP: set[str]`: the URLs of the copies with status `200` and no stored ZIP. Step 6 fills it, and Task 8 empties it (`spec:86-89`).
- `test_the_store_matches_its_manifest`: `check_store(STORE_DIR) == []`.
- `test_every_stored_zip_is_named_by_a_recorded_row`.
- `test_every_opened_copy_has_a_stored_zip_except_the_pinned_ones`: the set of `200` rows without `STORE_DIR / stored_file(...)` equals `AWAITING_A_STORED_ZIP`. The comparison is exact, so any change fails.
- `test_every_present_inventory_cell_resolves_to_a_stored_zip`.
  - A `bls_fetched` cell's evidence is a file under `ROOT`.
  - An `internet_archive` cell's URL names a row whose `(artifact, digest)` has a stored ZIP. A revisit row carries its twin's digest, so the same lookup covers it. A cell is exempt only when a row in `AWAITING_A_STORED_ZIP` has the same artifact and digest.
- `test_release_2026_10_is_archive_complete_on_stored_bls_copies`: all three statuses are `bls_fetched`, `files_complete == "true"`, and each evidence path is a file.
- `test_the_bls_copies_of_2026_09_13_are_restored`.
  - The `ces.spec.ae.zip` row has status `200`, digest `UMPGBDVBRNEGZEZI5FGRTGMFDVXETU4X` and `newest_member == "2026-02-09T15:30:44"`.
  - The `ces.spec.other.zip` row has status `payload_lost`, digest `JRW4ECRTZ3X22BDLK7YHUZ6U3M7F5WUG`, fingerprints `70f4db3c9e3b4761` and `635b9e68f7634c73`, and no member date.
  - Release 2026-08's `prior_adjustment` and `outliers` are `not_archived`.
  - The count of `files_complete == "true"` equals `ARCHIVE_COMPLETE`.
- `test_rendered_bls_cells_link_the_stored_zip`: a synthetic `BLS_SOURCE` copy renders `` [`bls_fetched`](../data/seasonal/raw/ces.spec.other/<digest>.zip) `` in `render_blocks(rows)["archive-vintages"]`.

Update `test_file_status_prefers_the_bls_copy_and_ignores_captures_without_content` to expect `(BLS_SOURCE, evidence(current))`.

- [ ] **Step 2: Flip `BLS_SOURCE` and add `evidence`**

Run: `uv run pytest tests/test_archive_inventory.py tests/test_review_document.py -m "not network" -q`
Expected: FAIL, because the committed CSVs still say `bls_current`.

- [ ] **Step 3: Relabel and restore the rows**

Make a fresh folder with `mktemp -d` and write `migrate_captures.py` alone in it, so no other file sits beside the script on `sys.path`. Run `uv run python <folder>/migrate_captures.py` from the worktree root. Put the Step 6 helper in such a folder too.

```python
"""Plan 14, Task 6: relabel the BLS rows and restore the two BLS rows of 2026-09-13 (commit 1f1061a)."""

import sys
from datetime import UTC, datetime

sys.path.insert(0, "scripts")
import archive_inventory as ai  # noqa: E402

FETCHED = datetime(2026, 9, 13, 17, 11, 32, tzinfo=UTC)
committed = [
    ai.capture_from_row(
        {
            **row,
            "source": ai.BLS_SOURCE
            if row["source"] == "bls_current"
            else row["source"],
        }
    )
    for row in ai.read_csv(ai.CAPTURES_PATH)
]
restored = [
    ai.Capture(
        "all_employees_specifications",
        ai.BLS_SOURCE,
        FETCHED,
        "https://www.bls.gov/web/empsit/ces.spec.ae.zip",
        "200",
        "UMPGBDVBRNEGZEZI5FGRTGMFDVXETU4X",
        specification="8f91d65e24c47302",
    ),
    ai.Capture(
        "other_inputs",
        ai.BLS_SOURCE,
        FETCHED,
        "https://www.bls.gov/web/empsit/ces.spec.other.zip",
        "payload_lost",
        "JRW4ECRTZ3X22BDLK7YHUZ6U3M7F5WUG",
        prior_adjustment="70f4db3c9e3b4761",
        outliers="635b9e68f7634c73",
    ),
]
merged = ai.merge_captures(committed, restored)
ai.write_csv(ai.CAPTURES_PATH, [ai.capture_row(c) for c in merged], ai.CAPTURE_COLUMNS)
print(f"{len(committed)} -> {len(merged)} rows")
```

Expected: `74 -> 76 rows`.

- [ ] **Step 4: Adopt the rescued files**

First check that `/Users/lowell/Projects/ces-sa-rescue/2026-11-16/` holds both ZIPs. If it does not, stop and ask the owner. Run from the worktree root:

```bash
uv run python -I scripts/archive_inventory.py adopt /Users/lowell/Projects/ces-sa-rescue/2026-11-16/ces.spec.ae.zip /Users/lowell/Projects/ces-sa-rescue/2026-11-16/ces.spec.other.zip
```

Expected: exit 0. `data/seasonal/raw/manifest.csv` has two rows, with the SHA-256 values `805f6afecb353ee89af2911ce55eb506f325f8c7b51b585cd5ef046928eb863e` (`ces.spec.ae`) and `155dccf7afb0fa20e5b7b6612b82af0e4f9ce63ac7276920b1ff7b2812fe91fb` (`ces.spec.other`), which match `fetch-record.json`. Seven rows match `UMPG…`: the two BLS rows and five Internet Archive copies from 2026-02-14 to 2026-08-19. Their fingerprints and member dates must agree, and only the 2026-09-13 row's empty member date changes.

- [ ] **Step 5: Regenerate the inventory**

Run: `uv run python scripts/archive_inventory.py inventory`
Expected: exit 0. In `archive-inventory.csv`, `bls_current` cells become `bls_fetched`, with `data/seasonal/raw/…` evidence.

- [ ] **Step 6: Pin the copies that await the full run**

Set `AWAITING_A_STORED_ZIP` to the sorted URLs of the CSV's `200` rows without a stored ZIP. Generate the list with a short script that uses `read_csv`, `stored_file` and `STORE_DIR`. Expected: 55 URLs. That is the 60 `internet_archive` rows with status `200`, minus the five that share the adopted `UMPG…` digest.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest -m "not slow and not network" -q`
Expected: PASS, including `test_only_an_opened_copy_records_a_member_date` and `test_every_copy_was_made_after_the_release_its_files_were_made_for`.

- [ ] **Step 8: Commit**

```bash
git add scripts/archive_inventory.py tests/test_archive_inventory.py docs/inventory/archive-captures.csv docs/inventory/archive-inventory.csv docs/ces-revisions-review.md data/seasonal/raw
git commit -m "Store the rescued BLS ZIPs and restore the BLS copies of 2026-09-13"
```

### Task 7: Documents

**Files:**
- Modify: `docs/ces-revisions-review.md` (hand-written parts only)
- Modify: `CLAUDE.md`

The content comes from `spec:296-310`.

- [ ] **Step 1: Update the review's prose**

- In Method step 1 (`review:131`), state four things. `captures` stores each ZIP it opens in `data/seasonal/raw/`, named by its digest. It keeps every earlier row. A replayed copy counts only when its SHA-1 equals its CDX digest. The step also describes `current` and `adopt`.
- In the status list (`review:134`), `bls_fetched` replaces `bls_current`: it is a copy fetched from BLS's live URL and held in the store. Also define `payload_lost`.
- State the release 2026-08 loss once, after the status list, citing its `payload_lost` row.
- In the coverage caption (`review:152`), use `bls_fetched`.
- At `review:486`, drop the claim that BLS serves the copies now. Task 8 recounts that paragraph.
- Rewrite the Stage 14 bullet (`review:593`). The file store exists in `data/seasonal/raw/`, and its coverage contains this inventory's. Stage 14 builds its flag panel on it.
- Leave the counts in Method step 2 ("In all 33 copies…") for Task 8.

- [ ] **Step 2: Update `CLAUDE.md`**

- Line 33 names `current`, `adopt` and the store.
- A new bullet after line 36 covers `data/seasonal/raw/`. It is committed and hashed in `manifest.csv`, and only `archive_inventory.py`'s `captures`, `current` and `adopt` write it. Never hand-edit it. Each network command needs the owner's yes. `current` runs after each monthly release, followed by `inventory`.
- Add these three lines to the Commands block:

```bash
uv run --env-file .project.env python scripts/archive_inventory.py captures  # network, owner's yes: record and store every BLS and Internet Archive copy (from the main checkout; a worktree passes its absolute path)
uv run --env-file .project.env python scripts/archive_inventory.py current   # network, owner's yes: after each release, fetch and store BLS's two ZIPs
uv run python -I scripts/archive_inventory.py adopt <zip>...                 # offline: store a local copy of a recorded ZIP
```

- [ ] **Step 3: Verify**

Run: `uv run pytest tests/test_review_document.py -q && uv run ruff format --check docs/ces-revisions-review.md CLAUDE.md`
Expected: PASS, and the format check reports no changes.

- [ ] **Step 4: Commit**

```bash
git add docs/ces-revisions-review.md CLAUDE.md
git commit -m "Describe the file store, current and adopt in the review and CLAUDE.md"
```

### Task 8: The first full `captures` run (owner gate)

**Files:**
- Regenerate: `data/seasonal/raw/`, `docs/inventory/*.csv` and the review's generated blocks
- Modify: `tests/test_archive_inventory.py` (`AWAITING_A_STORED_ZIP`, `ARCHIVE_COMPLETE`), and the prose counts at `review:132` and `review:486`

- [ ] **Step 1: STOP and ask the owner**

Ask: "Run `captures` now? It sends about 100 requests to bls.gov and web.archive.org, with the contact address to bls.gov only, and stores about 40 ZIPs (about 3.5 MB)." Go on only after an explicit yes. If the owner declines, leave this task unchecked, go to Task 9, and let the completion gate defer this task.

- [ ] **Step 2: Check the contact address without printing it**

Run: `uv run --env-file /Users/lowell/Projects/ces-revisions/.project.env python -c "import os; print(bool(os.environ.get('BLS_CONTACT_EMAIL')))"`
Expected: `True`. Otherwise stop and ask the owner.

- [ ] **Step 3: Run it**

Run it in the background, because it takes several minutes:

```bash
uv run --env-file /Users/lowell/Projects/ces-revisions/.project.env python scripts/archive_inventory.py captures
```

Record the exit code and every stderr line. Exit 1 means one of two things:

- Stderr names `200` rows with no stored ZIP. This is the throttled case: keep going.
- Stderr names unexplained members. This is a finding about BLS's files: stop and report to the owner.

Stop and report in two more cases. One is a new copy that fails `test_no_copy_of_a_seasonal_adjustment_zip_holds_an_unexplained_file`. The other is a new copy that fails `test_every_copy_was_made_after_the_release_its_files_were_made_for`. Never change `_EXPLAINED`, `BUILD_LEAD_DAYS` or `SPECIFICATIONS_BUILT_AFTER_THEIR_BENCHMARK` without the owner's yes.

- [ ] **Step 4: Check that no committed row changed**

Run: `uv run python scripts/archive_inventory.py inventory`, then `git diff -U0 docs/inventory/archive-captures.csv | grep '^-[^-]'`
Expected: no output, so every change is an added row. The only allowed removed line belongs to a row that was `unreplayable` and is now `200` (merge point 5). There is none at this plan's base.

- [ ] **Step 5: Update the pinned evidence**

Set `AWAITING_A_STORED_ZIP` to the URLs that the run named, which is an empty set after a clean run. If the archive-complete count rose, set `ARCHIVE_COMPLETE` to the new count. If it fell, stop and report to the owner.

- [ ] **Step 6: Update the prose counts**

Recount the copy counts at `review:132` and `review:486` from `archive-captures.csv` with a short script, and edit the prose to match.

- [ ] **Step 7: Verify and commit everything together**

Run: `uv run pytest -m "not slow and not network" -q`
Expected: PASS.

```bash
git add data/seasonal/raw docs/inventory/archive-captures.csv docs/inventory/es-vintages.csv docs/inventory/archive-inventory.csv docs/ces-revisions-review.md tests/test_archive_inventory.py
git commit -m "Store every opened Internet Archive copy from the first full captures run"
```

- [ ] **Step 8: Tell the owner the rescue folder can go**

Tell the owner that `/Users/lowell/Projects/ces-sa-rescue/` can now be deleted (`spec:361-362`). Do not delete it yourself.

### Task 9: Whole-branch review and completion

- [ ] **Step 1: Review the whole branch**

Dispatch code-reviewer (Opus) over the range from the merge base with `origin/main` to `HEAD`, against the spec and this plan. Fix or answer each finding.

- [ ] **Step 2: Run the Plan Completion Protocol of writing-plans**

Tick the item "Keep each `bls_current` copy as evidence across `captures` runs" with `→ done <date>`. Its "Done when" is met by Task 4's two-run test, which covers both commands, and by Task 8. If Task 8 was deferred, defer the full run as a new item instead. Leave the parked Stage 14 item's text unchanged (D1). Retire this plan and the spec.
