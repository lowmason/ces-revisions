# Vintage Source Fetch Hardening Implementation Plan

**Status: COMPLETE (2026-09-15)** — executed via executing-plans; nothing deferred

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via subagent-driven-development (the default) — or executing-plans when your human partner chose inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Stage 3 source refresh safe against partial downloads and error-page payloads, add an offline manifest rehash command for the hand-maintained source, and record the Poppler version that produced the committed historical-release text.

**Architecture:** Keep `scripts/vintage_sources.py` as the single Stage 3 acquisition entry point, but split its work into validation, staging, promotion, and manifest-refresh boundaries. A network fetch writes every downloaded and derived artifact into a temporary tree on the same filesystem as `data/raw/`, validates all payloads and derived text, writes the complete staged manifest, and only then promotes files into the committed archive. The offline `manifest` command rehashes the existing file set while preserving acquisition and derivation provenance.

**Tech Stack:** Python 3.14; standard-library `argparse`, `datetime`, `pathlib`, `shutil`, `subprocess`, `tempfile`, and URL tools; the existing `fastexcel` and CSV helpers; pytest 9.1.1; ruff 0.16.7; and `pdftotext` from Poppler for the network fetch only.

## Global Constraints

- **Scope:** Implement only the open fetch-hardening item from plan 5 in `specs/deferred_items.md`. Do not change vintage parsing, stage labels, differencing, panel schemas, or any roadmap stage.
- **Raw archive:** `data/raw/` remains committed and immutable except through `scripts/vintage_sources.py fetch`, or through the documented hand edit of `data/raw/manual/es-reschedules.csv` followed by `manifest`.
- **Transaction boundary:** No file below `data/raw/` and no cached `cesvin00.xlsx` may be replaced until every network payload, workbook extraction, release-index parse, PDF conversion, and staged manifest write has succeeded.
- **Validation:** Reject empty content and mismatched ZIP, PDF, XLS, XLSX, HTML, or HTM signatures before parsing or promotion. This is a file-type guard, not a promise that the upstream schema is unchanged; existing parser and canary tests keep owning schema drift.
- **Manual source:** Preserve `manual/es-reschedules.csv` byte-for-byte during `fetch`; it is copied into the staged snapshot only so the new manifest hashes the complete archive.
- **Manifest provenance:** The offline rehash changes only `sha256` and `bytes`. It preserves `url`, `last_modified`, `fetched_at`, `derived_from`, `derived_from_sha256`, and `tool_version`, and it fails if the archive's file set differs from the manifest.
- **Poppler provenance:** Add a `tool_version` manifest column. Only `bls/histreleasedates.txt` populates it, with the first line of `pdftotext -v`; a fetch records the version that performed that same fetch's conversion.
- **Network identity:** Preserve the existing `BLS_CONTACT_EMAIL` requirement and host-specific `User-Agent`; never print or persist the address.
- **Dependencies:** Add no package dependency. The Stage 4 fetcher is a behavioral precedent, not a shared-module refactor.
- **Testing:** All new tests are hermetic. Do not make a live fetch while implementing this plan; the existing live freshness canary remains `@pytest.mark.network`.
- **Deferred ownership:** Leave Q1 (member-date windows) open for Stage 14 and both halves of Q2 open for Stages 6 and 8. Completing this plan closes only the fetch-hardening item.

---

## Source and numbering

This spec-less plan implements the selected `/deferred` item **“Review Important and Minors: harden `fetch` before the next refresh”** under `## 5-ces-revisions — 2026-09-14` in [`specs/deferred_items.md`](../../deferred_items.md). It is plan **7**, the next integer after completed plan 6.

This work is not roadmap Stage 5. Stage 5 builds benchmark, aggregation, and seasonal-mapping operators and does not fetch Stage 3 sources. On completion, do not tick or edit a roadmap stage. The Plan Completion Protocol must tick the source item as:

```markdown
- [x] Review Important and Minors: harden `fetch` before the next refresh … → done in plan 7
```

## Scope check

The payload guards, staging transaction, manifest rehash, and Poppler provenance form one acquisition-integrity change: each protects the same refresh operation and shares the same manifest contract. Splitting them would either leave `fetch` able to commit unvalidated bytes or leave the hand-maintained source without a supported rehash path. Q1 and Q2 remain separate because their roadmap stages consume different data and already own their closure tests.

## Planning evidence (2026-09-15)

- `fetch_sources()` currently writes each response directly into `data/raw/`; a failure on a later response or an error page therefore leaves earlier files replaced while `manifest.csv` still describes the old or incomplete set.
- `scripts/annual_sources.py` already proves the local pattern: validate payloads, write a temporary tree, and promote only after all validation and PDF extraction succeeds. Reuse the pattern locally without coupling the Stage 3 and Stage 4 source catalogs.
- Stage 3 downloads five committed payload types: ZIP, HTML, PDF, XLSX, and XLS. It also downloads the gitignored XLSX comments workbook and the HTML Employment Situation index before deriving committed CSVs.
- `manual/es-reschedules.csv` is the only hand-editable Stage 3 source. The current manifest test correctly fails after an edit, but `vintage_sources.py` has no offline command that can acknowledge the new bytes without repeating every network request.
- `bls/histreleasedates.txt` is layout-sensitive output of `pdftotext -layout`; the current manifest records the PDF hash but not the Poppler version. The planning host reports `pdftotext version 26.06.0`.
- The working tree contains an unrelated untracked `AGENTS.md`; execution must preserve it and must stage only the paths named by this plan.

## Execution notes

- **Plan artifact:** Ensure this plan is tracked on the starting branch before creating an execution worktree. Do not include the unrelated untracked `AGENTS.md` in any commit.
- **Isolation:** Before Task 1, use `using-git-worktrees` and create an isolated `codex/vintage-fetch-hardening` branch from the current branch.
- **Tests first:** Use `test-driven-development` for each task. Observe the named failure before adding implementation.
- **Verification:** Before every commit, run the task's focused tests plus `uv run ruff check` and `uv run ruff format --check` on the touched Python and Markdown files.
- **Stop conditions:** Stop rather than weakening validation if a committed source does not match its URL's expected type, `pdftotext` emits empty text, the current manifest and archive name different files, or the local Poppler version cannot be established.
- **No network refresh:** This plan hardens the next refresh; it does not perform it. Use mocked bytes for transaction tests and keep the live canary deselected.

## File structure

### Modify

- `scripts/vintage_sources.py` — file-signature guards, Poppler-version capture, staged fetch transaction, offline manifest rehash, and the new `manifest` CLI branch.
- `tests/test_vintage_sources.py` — validation, transaction rollback, Poppler provenance, rehash, file-set, and CLI tests.
- `data/raw/manifest.csv` — add the `tool_version` column and record the Poppler version for the existing `histreleasedates.txt` derivation.
- `README.md` — document the Stage 3 `fetch`, `manifest`, and `build` commands and their network/offline boundaries.
- `CLAUDE.md` — add the offline Stage 3 manifest command and the staging/validation invariant to the repository guidance.
- `specs/deferred_items.md` — completion-protocol markup only, after all tasks and review are complete.

### Create

None. The change stays within the existing Stage 3 acquisition boundary.

---

### Task 1: Validate Every Download Type and Capture the Poppler Version

**Files:**

- Modify: `scripts/vintage_sources.py`
- Modify: `tests/test_vintage_sources.py`

**Interfaces:**

- Consumes: a source URL and downloaded `bytes`; the installed `pdftotext` executable.
- Produces: `validate_payload(url: str, payload: bytes) -> None`, `pdftotext_version() -> str`, and `convert_pdf(source: Path, target: Path) -> None`.

- [x] **Step 1: Write failing payload-signature and Poppler-helper tests**

Add `subprocess` and `from pathlib import Path` to the imports in
`tests/test_vintage_sources.py`, then add:

```python
@pytest.mark.parametrize(
    ("url", "payload", "message"),
    [
        ("https://www.bls.gov/source.zip", b"<html>error</html>", "ZIP signature"),
        ("https://www.bls.gov/source.pdf", b"<html>error</html>", "PDF signature"),
        ("https://www.bls.gov/source.xls", b"<html>error</html>", "XLS signature"),
        ("https://www.bls.gov/source.xlsx", b"<html>error</html>", "XLSX signature"),
        (
            "https://www.bls.gov/source.htm",
            b"temporarily unavailable",
            "HTML signature",
        ),
        (
            "https://www.bls.gov/source.html",
            b"temporarily unavailable",
            "HTML signature",
        ),
    ],
)
def test_payload_validation_rejects_error_pages_and_wrong_file_types(
    url, payload, message
):
    with pytest.raises(ValueError, match=message):
        vintage_sources.validate_payload(url, payload)


@pytest.mark.parametrize(
    ("url", "payload"),
    [
        ("https://www.bls.gov/source.zip", b"PK\x03\x04archive"),
        ("https://www.bls.gov/source.pdf", b"%PDF-1.7\nbody"),
        (
            "https://www.bls.gov/source.xls",
            b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1workbook",
        ),
        ("https://www.bls.gov/source.xlsx", b"PK\x03\x04workbook"),
        ("https://www.bls.gov/source.htm", b"<!doctype html><html></html>"),
    ],
)
def test_payload_validation_accepts_expected_file_signatures(url, payload):
    vintage_sources.validate_payload(url, payload)


def test_payload_validation_rejects_an_empty_response():
    with pytest.raises(ValueError, match="empty payload"):
        vintage_sources.validate_payload("https://www.bls.gov/source.zip", b"")


def test_pdftotext_version_is_the_first_nonempty_output_line(monkeypatch):
    completed = subprocess.CompletedProcess(
        ["pdftotext", "-v"],
        0,
        stdout="",
        stderr="pdftotext version 26.06.0\nCopyright line\n",
    )
    monkeypatch.setattr(
        vintage_sources.subprocess, "run", lambda *args, **kwargs: completed
    )
    assert vintage_sources.pdftotext_version() == "pdftotext version 26.06.0"


def test_pdf_conversion_rejects_empty_derived_text(monkeypatch, tmp_path):
    source = tmp_path / "source.pdf"
    target = tmp_path / "source.txt"
    source.write_bytes(b"%PDF-1.7")

    def empty_conversion(command, check):
        Path(command[-1]).write_text("\n", encoding="utf-8")

    monkeypatch.setattr(vintage_sources.subprocess, "run", empty_conversion)
    with pytest.raises(ValueError, match="produced no text"):
        vintage_sources.convert_pdf(source, target)
```

- [x] **Step 2: Run the focused tests and confirm the missing-interface failure**

Run:

```bash
uv run pytest \
  tests/test_vintage_sources.py::test_payload_validation_rejects_error_pages_and_wrong_file_types \
  tests/test_vintage_sources.py::test_payload_validation_accepts_expected_file_signatures \
  tests/test_vintage_sources.py::test_payload_validation_rejects_an_empty_response \
  tests/test_vintage_sources.py::test_pdftotext_version_is_the_first_nonempty_output_line \
  tests/test_vintage_sources.py::test_pdf_conversion_rejects_empty_derived_text -v
```

Expected: FAIL because `validate_payload`, `pdftotext_version`, and `convert_pdf` do not exist.

- [x] **Step 3: Implement the file guards and Poppler helpers**

Add these constants and functions to `scripts/vintage_sources.py` after `MANIFEST_COLUMNS`:

```python
ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
XLS_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def validate_payload(url: str, payload: bytes) -> None:
    """Reject empty responses and payloads that do not match their file type."""
    if not payload:
        raise ValueError(f"{url} returned an empty payload")
    suffix = Path(urlsplit(url).path).suffix.lower()
    prefix = payload.lstrip()[:4096].lower()
    checks = {
        ".zip": (payload.startswith(ZIP_SIGNATURES), "ZIP"),
        ".pdf": (payload.startswith(b"%PDF-"), "PDF"),
        ".xls": (payload.startswith(XLS_SIGNATURE), "XLS"),
        ".xlsx": (payload.startswith(ZIP_SIGNATURES), "XLSX"),
        ".htm": (b"<html" in prefix, "HTML"),
        ".html": (b"<html" in prefix, "HTML"),
    }
    if suffix in checks:
        valid, label = checks[suffix]
        if not valid:
            raise ValueError(f"{url} does not have a {label} signature")


def pdftotext_version() -> str:
    """Return the Poppler version line recorded beside layout-derived text."""
    completed = subprocess.run(
        ["pdftotext", "-v"], check=True, capture_output=True, text=True
    )
    lines = [
        line.strip() for line in (completed.stderr or completed.stdout).splitlines()
    ]
    try:
        return next(line for line in lines if line)
    except StopIteration as error:
        raise ValueError("pdftotext -v returned no version text") from error


def convert_pdf(source: Path, target: Path) -> None:
    """Create layout-preserving text and reject an empty conversion."""
    subprocess.run(["pdftotext", "-layout", str(source), str(target)], check=True)
    if not target.read_text(encoding="utf-8").strip():
        raise ValueError(f"pdftotext produced no text for {source}")
```

Append `"tool_version"` to `MANIFEST_COLUMNS`; Task 3 migrates the committed CSV after the fetch path writes the field.

- [x] **Step 4: Run the focused tests and the existing Stage 3 source tests**

Run:

```bash
uv run pytest tests/test_vintage_sources.py -m "not network" -v
```

Expected: PASS. The current committed manifest may not yet contain `tool_version`; no assertion requires it until Task 3.

- [x] **Step 5: Lint and commit the validation boundary**

Run:

```bash
uv run ruff format scripts/vintage_sources.py tests/test_vintage_sources.py
uv run ruff check scripts/vintage_sources.py tests/test_vintage_sources.py
git diff --check
git add scripts/vintage_sources.py tests/test_vintage_sources.py
git commit -m "fix(data): validate vintage source payloads"
```

Expected: formatting and lint pass; the commit contains only the two named files.

---

### Task 2: Stage the Complete Fetch Before Promoting Any Source

**Files:**

- Modify: `scripts/vintage_sources.py`
- Modify: `tests/test_vintage_sources.py`

**Interfaces:**

- Consumes: Task 1's `validate_payload()`, `pdftotext_version()`, and `convert_pdf()`; the existing `DOWNLOADS`, comments extractor, release-index parser, and manual reschedule file.
- Produces: `manifest_row(..., raw_dir: Path)`, `promote_tree(staging_dir: Path, target_dir: Path) -> None`, and `fetch_sources(now, *, raw_dir=..., workbook_path=...) -> int` with an all-validation-before-promotion contract.

- [x] **Step 1: Write the regression test for a failure after an earlier successful download**

Add this regression test to `tests/test_vintage_sources.py`:

```python
def test_fetch_validates_every_payload_before_replacing_sources(
    monkeypatch, tmp_path: Path
):
    raw_dir = tmp_path / "raw"
    existing_zip = raw_dir / raw.VINTAGE_FILES
    existing_zip.parent.mkdir(parents=True)
    existing_zip.write_bytes(b"old archive")
    existing_manifest = raw_dir / raw.MANIFEST
    existing_manifest.write_text("old manifest\n", encoding="utf-8")
    workbook_path = tmp_path / "cache" / "cesvin00.xlsx"
    responses = iter(
        [
            (b"PK\x03\x04new archive", "Fri, 06 Mar 2026 12:06:34 GMT"),
            (b"temporarily unavailable", ""),
        ]
    )

    monkeypatch.setenv("BLS_CONTACT_EMAIL", "someone@example.org")
    monkeypatch.setattr(vintage_sources.shutil, "which", lambda command: "/pdftotext")
    monkeypatch.setattr(
        vintage_sources, "pdftotext_version", lambda: "pdftotext version 26.06.0"
    )
    monkeypatch.setattr(vintage_sources, "download", lambda url: next(responses))

    with pytest.raises(ValueError, match="HTML signature"):
        vintage_sources.fetch_sources(
            datetime(2026, 9, 15, tzinfo=UTC),
            raw_dir=raw_dir,
            workbook_path=workbook_path,
        )

    assert existing_zip.read_bytes() == b"old archive"
    assert existing_manifest.read_text(encoding="utf-8") == "old manifest\n"
    assert not workbook_path.exists()
```

- [x] **Step 2: Run the regression test and verify the destructive partial update**

Run:

```bash
uv run pytest \
  tests/test_vintage_sources.py::test_fetch_validates_every_payload_before_replacing_sources -v
```

Expected: FAIL with an unexpected `raw_dir` keyword because the current
`fetch_sources()` exposes no isolated destination and therefore has no testable
staging boundary.

- [x] **Step 3: Make manifest rows and promotion operate on explicit roots**

Import `tempfile` in `scripts/vintage_sources.py`. Replace `manifest_row()` and add `promote_tree()`:

```python
def manifest_row(
    file: str, fetched_at: str, *, raw_dir: Path = RAW_DIR, **fields: str
) -> dict[str, str]:
    path = raw_dir / file
    row = dict.fromkeys(MANIFEST_COLUMNS, "")
    row.update(
        file=file,
        sha256=file_sha256(path),
        bytes=str(path.stat().st_size),
        fetched_at=fetched_at,
        **fields,
    )
    return row


def promote_tree(staging_dir: Path, target_dir: Path) -> None:
    """Replace target files only after the caller has completed staging."""
    for staged in sorted(path for path in staging_dir.rglob("*") if path.is_file()):
        target = target_dir / staged.relative_to(staging_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        staged.replace(target)
```

- [x] **Step 4: Replace the direct-write fetch with a staged transaction**

Replace `fetch_sources()` with:

```python
def fetch_sources(
    now: datetime,
    *,
    raw_dir: Path = RAW_DIR,
    workbook_path: Path = WORKBOOK_PATH,
) -> int:
    """Refresh all sources only after a complete staged fetch validates."""
    if not os.environ.get("BLS_CONTACT_EMAIL"):
        print(
            "set BLS_CONTACT_EMAIL to the contact address BLS asks automated clients for",
            file=sys.stderr,
        )
        return 1
    if shutil.which("pdftotext") is None:
        print("pdftotext (poppler) is required: brew install poppler", file=sys.stderr)
        return 1

    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    tool_version = pdftotext_version()
    raw_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="ces-stage3-fetch-", dir=raw_dir.parent
    ) as directory:
        staging_root = Path(directory)
        staged_raw = staging_root / "raw"
        staged_workbook = staging_root / "cache" / workbook_path.name
        rows = []

        for item in DOWNLOADS:
            payload, last_modified = download(item.url)
            validate_payload(item.url, payload)
            target = staged_raw / item.file
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            rows.append(
                manifest_row(
                    item.file,
                    stamp,
                    raw_dir=staged_raw,
                    url=item.url,
                    last_modified=last_modified,
                )
            )

        pdf = staged_raw / "bls/histreleasedates.pdf"
        convert_pdf(pdf, staged_raw / HISTORICAL_RELEASE_DATES)
        rows.append(
            manifest_row(
                HISTORICAL_RELEASE_DATES,
                stamp,
                raw_dir=staged_raw,
                derived_from="pdftotext -layout bls/histreleasedates.pdf",
                derived_from_sha256=file_sha256(pdf),
                tool_version=tool_version,
            )
        )

        payload, last_modified = download(WORKBOOK_URL)
        validate_payload(WORKBOOK_URL, payload)
        staged_workbook.parent.mkdir(parents=True, exist_ok=True)
        staged_workbook.write_bytes(payload)
        archive_inventory.write_csv(
            staged_raw / VINTAGE_COMMENTS,
            extract_comments(staged_workbook),
            COMMENT_COLUMNS,
        )
        rows.append(
            manifest_row(
                VINTAGE_COMMENTS,
                stamp,
                raw_dir=staged_raw,
                last_modified=last_modified,
                derived_from=f"{WORKBOOK_URL}, sheet {COMMENTS_SHEET}",
                derived_from_sha256=file_sha256(staged_workbook),
            )
        )

        index_payload, _ = download(archive_inventory.RELEASE_INDEX_URL)
        validate_payload(archive_inventory.RELEASE_INDEX_URL, index_payload)
        vintages = archive_inventory.select_vintages(
            archive_inventory.parse_release_index(
                index_payload.decode("utf-8", "replace")
            ),
            now=now,
        )
        archive_inventory.write_csv(
            staged_raw / EMPSIT_RELEASES,
            archive_inventory.vintage_rows(vintages),
            archive_inventory.VINTAGE_COLUMNS,
        )
        rows.append(
            manifest_row(
                EMPSIT_RELEASES,
                stamp,
                raw_dir=staged_raw,
                derived_from=archive_inventory.RELEASE_INDEX_URL,
            )
        )

        manual_source = raw_dir / RESCHEDULES
        if not manual_source.is_file():
            raise FileNotFoundError(f"missing hand-maintained source {manual_source}")
        staged_manual = staged_raw / RESCHEDULES
        staged_manual.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manual_source, staged_manual)
        rows.append(
            manifest_row(
                RESCHEDULES,
                "",
                raw_dir=staged_raw,
                derived_from="hand-keyed from the citation on each row",
            )
        )
        archive_inventory.write_csv(
            staged_raw / MANIFEST,
            sorted(rows, key=lambda row: row["file"]),
            MANIFEST_COLUMNS,
        )

        workbook_path.parent.mkdir(parents=True, exist_ok=True)
        staged_workbook.replace(workbook_path)
        promote_tree(staged_raw, raw_dir)
    return 0
```

This implementation deliberately validates and derives everything before the two promotion calls. Promotion is not expected to recover from a filesystem or power failure; it closes the recorded failure modes of a later download, parser, or conversion invalidating an earlier source.

- [x] **Step 5: Run the transaction regression and all hermetic source tests**

> Deviation: Whole-plan review added successful-promotion and late-parser-failure coverage beyond the planned early-payload regression.

Run:

```bash
uv run pytest tests/test_vintage_sources.py -m "not network" -v
```

Expected: PASS, including proof that a valid first payload remains staged when a later HTML payload is invalid and that neither the old archive nor its manifest changes.

- [x] **Step 6: Lint and commit the staged fetch**

Run:

```bash
uv run ruff format scripts/vintage_sources.py tests/test_vintage_sources.py
uv run ruff check scripts/vintage_sources.py tests/test_vintage_sources.py
git diff --check
git add scripts/vintage_sources.py tests/test_vintage_sources.py
git commit -m "fix(data): stage vintage source refreshes"
```

Expected: formatting and lint pass; the commit contains only the two named files.

---

### Task 3: Add the Offline Rehash Command and Complete Provenance

**Files:**

- Modify: `scripts/vintage_sources.py`
- Modify: `tests/test_vintage_sources.py`
- Modify: `data/raw/manifest.csv`
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Interfaces:**

- Consumes: the existing `data/raw/manifest.csv` file set and provenance fields.
- Produces: `refresh_manifest(raw_dir: Path = RAW_DIR) -> int`; `vintage_sources.py manifest`; a manifest with `tool_version`; and command documentation.

- [x] **Step 1: Write failing offline-rehash, file-set, CLI, and committed-provenance tests**

Add these tests to `tests/test_vintage_sources.py`:

```python
def test_refresh_manifest_rehashes_bytes_without_changing_provenance(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    source = raw_dir / raw.RESCHEDULES
    source.parent.mkdir(parents=True)
    source.write_text("old\n", encoding="utf-8")
    row = dict.fromkeys(vintage_sources.MANIFEST_COLUMNS, "")
    row.update(
        file=raw.RESCHEDULES,
        sha256="old hash",
        bytes="4",
        derived_from="hand-keyed from the citation on each row",
    )
    vintage_sources.archive_inventory.write_csv(
        raw_dir / raw.MANIFEST, [row], vintage_sources.MANIFEST_COLUMNS
    )

    source.write_text("new source bytes\n", encoding="utf-8")
    assert vintage_sources.refresh_manifest(raw_dir) == 1

    [refreshed] = vintage_sources.archive_inventory.read_csv(raw_dir / raw.MANIFEST)
    assert refreshed["sha256"] == raw.file_sha256(source)
    assert refreshed["bytes"] == str(source.stat().st_size)
    assert refreshed["derived_from"] == "hand-keyed from the citation on each row"
    assert refreshed["url"] == ""
    assert refreshed["fetched_at"] == ""


def test_refresh_manifest_rejects_an_unrecorded_source(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    vintage_sources.archive_inventory.write_csv(
        raw_dir / raw.MANIFEST, [], vintage_sources.MANIFEST_COLUMNS
    )
    (raw_dir / "unexpected.csv").write_text("value\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file set differs"):
        vintage_sources.refresh_manifest(raw_dir)


def test_manifest_command_reports_the_refreshed_file_count(monkeypatch, capsys):
    monkeypatch.setattr(vintage_sources, "refresh_manifest", lambda: 9)
    assert vintage_sources.main(["manifest"]) == 0
    assert capsys.readouterr().out == "manifest: 9 source files\n"


def test_historical_release_text_records_its_pdftotext_version():
    rows = {row["file"]: row for row in manifest()}
    assert rows[raw.HISTORICAL_RELEASE_DATES]["tool_version"].startswith(
        "pdftotext version "
    )
    assert all(
        not row["tool_version"]
        for file, row in rows.items()
        if file != raw.HISTORICAL_RELEASE_DATES
    )
```

- [x] **Step 2: Run the new tests and confirm the missing command/schema failures**

Run:

```bash
uv run pytest \
  tests/test_vintage_sources.py::test_refresh_manifest_rehashes_bytes_without_changing_provenance \
  tests/test_vintage_sources.py::test_refresh_manifest_rejects_an_unrecorded_source \
  tests/test_vintage_sources.py::test_manifest_command_reports_the_refreshed_file_count \
  tests/test_vintage_sources.py::test_historical_release_text_records_its_pdftotext_version -v
```

Expected: FAIL because `refresh_manifest()` and the `manifest` command do not exist and the committed CSV lacks `tool_version`.

- [x] **Step 3: Implement provenance-preserving offline rehashing**

> Deviation: Whole-plan review narrowed the manifest exclusion to the root path so a nested `manifest.csv` cannot escape the exact file-set check.

Add this function before `fetch_sources()` in `scripts/vintage_sources.py`:

```python
def refresh_manifest(raw_dir: Path = RAW_DIR) -> int:
    """Rehash the recorded archive without changing acquisition provenance."""
    existing_rows = archive_inventory.read_csv(raw_dir / MANIFEST)
    existing = {row["file"]: row for row in existing_rows}
    files = sorted(
        str(path.relative_to(raw_dir))
        for path in raw_dir.rglob("*")
        if path.is_file() and path.name not in {MANIFEST, ".DS_Store"}
    )
    if set(existing) != set(files):
        missing = sorted(set(existing) - set(files))
        unrecorded = sorted(set(files) - set(existing))
        raise ValueError(
            f"manifest file set differs: missing={missing}, unrecorded={unrecorded}"
        )

    rows = []
    for file in files:
        row = dict.fromkeys(MANIFEST_COLUMNS, "")
        row.update(existing[file])
        path = raw_dir / file
        row.update(
            file=file,
            sha256=file_sha256(path),
            bytes=str(path.stat().st_size),
        )
        rows.append(row)
    archive_inventory.write_csv(raw_dir / MANIFEST, rows, MANIFEST_COLUMNS)
    return len(rows)
```

Update the module docstring's command list to include:

```text
    uv run python scripts/vintage_sources.py manifest  # offline: rehash existing data/raw/
```

Replace `main()` with the three-way dispatch:

```python
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["fetch", "manifest", "build"])
    command = parser.parse_args(argv).command
    if command == "fetch":
        return fetch_sources(datetime.now(UTC).replace(microsecond=0))
    if command == "manifest":
        print(f"manifest: {refresh_manifest()} source files")
        return 0
    return build_panel()
```

- [x] **Step 4: Migrate the committed manifest schema and current Poppler provenance**

Append `tool_version` to the header of `data/raw/manifest.csv`, append an empty field to every other row, and make the historical-release text row exactly:

```csv
bls/histreleasedates.txt,,17b3a0fa2e487c695ae7765b4213390f9f529f1469e696144b3faa056849b6da,33971,,2026-09-15T01:28:27Z,pdftotext -layout bls/histreleasedates.pdf,03b2f2f36a7fdafe36bf34895f0fe5ac3bc118821cd4db8f00806e9ff1a059cc,pdftotext version 26.06.0
```

The header becomes:

```csv
file,url,sha256,bytes,last_modified,fetched_at,derived_from,derived_from_sha256,tool_version
```

Then run the offline command once to prove the migrated manifest is canonical and idempotent:

```bash
cp data/raw/manifest.csv /tmp/ces-revisions-plan7-manifest.csv
uv run python scripts/vintage_sources.py manifest
cmp /tmp/ces-revisions-plan7-manifest.csv data/raw/manifest.csv
```

Expected: `manifest: 9 source files`; `cmp` exits 0 because the rehash does
not alter the already-correct migrated file.

- [x] **Step 5: Document the three Stage 3 source commands and invariants**

Expand `README.md`'s Getting started command block to:

```bash
uv sync                                      # create .venv and install the package
uv run ces-revisions                         # run the entry point
uv run python scripts/vintage_sources.py fetch     # network: staged, validated refresh
uv run python scripts/vintage_sources.py manifest  # offline: rehash existing sources
uv run python scripts/vintage_sources.py build     # offline: build data/panel/
```

Immediately after that block, add:

```markdown
The Stage 3 fetch validates every payload and completes PDF/workbook-derived
artifacts in a temporary tree before replacing anything in `data/raw/`.
`manual/es-reschedules.csv` is the only hand-editable raw source; run the
offline `manifest` command after changing it.
```

In `CLAUDE.md`, replace the Stage 3 source-command paragraph with wording that states `fetch` stages and validates the complete refresh before promotion and that `manifest` is the only offline rehash path after editing `manual/es-reschedules.csv`. Add this command beside the existing Stage 3 commands:

```bash
uv run python scripts/vintage_sources.py manifest  # rehash existing data/raw/ after the permitted manual edit (offline)
```

- [x] **Step 6: Run the focused and full hermetic verification gates**

Run:

```bash
uv run pytest tests/test_vintage_sources.py -m "not network" -v
uv run pytest -m "not slow and not network"
uv run ruff format scripts/vintage_sources.py tests/test_vintage_sources.py README.md CLAUDE.md specs/plans/7-vintage-fetch-hardening.md
uv run ruff format --check
uv run ruff check
git diff --check
```

Expected: every command passes. The broad pytest run still builds against the unchanged committed Stage 3 bytes; only the manifest schema/provenance changed.

- [x] **Step 7: Commit the offline command, provenance, and documentation**

Run:

```bash
git add \
  scripts/vintage_sources.py \
  tests/test_vintage_sources.py \
  data/raw/manifest.csv \
  README.md \
  CLAUDE.md
git commit -m "feat(data): add offline vintage manifest refresh"
```

Expected: the commit contains exactly the five named files and does not include `AGENTS.md`.

---

## Completion gate

After Task 3:

1. Use `requesting-code-review` for a whole-plan review against this file and resolve every finding under `receiving-code-review`.
2. Use `verification-before-completion` and rerun the full commands from Task 3 Step 6 from a clean working tree, excluding only the tracked plan completion markup that follows.
3. Run the writing-plans Plan Completion Protocol. Tick the fetch-hardening item in `specs/deferred_items.md` with `→ done in plan 7`; do not tick Q1, Q2, or any roadmap stage.
4. Mark every plan checkbox, add the complete status header, and move this file to `specs/plans/completed/7-vintage-fetch-hardening.md` in the retirement commit. There is no standalone spec to retire.
5. Run the deferred-backlog stats helper and report the new closure rate and age spread before `finishing-a-development-branch` offers integration choices.
