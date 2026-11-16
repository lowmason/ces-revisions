"""scripts/archive_inventory.py: the content-addressed store for seasonal-adjustment ZIPs."""

import ast
import csv
import hashlib
import pathlib
import re
import shutil
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, date, datetime, timedelta
from operator import attrgetter
from pathlib import Path
from urllib.parse import urlsplit

import archive_inventory
import pytest
from archive_inventory import (
    ATTEMPTS,
    BLS_SOURCE,
    CAPTURE_COLUMNS,
    CAPTURES_PATH,
    EVIDENCE,
    FILE_TYPES,
    PAUSE_SECONDS,
    RELEASE_INDEX_URL,
    ROOT,
    STORE_DIR,
    STORE_MANIFEST,
    STORE_MANIFEST_COLUMNS,
    VINTAGE_COLUMNS,
    VINTAGES_PATH,
    Capture,
    EvidencePaths,
    Vintage,
    adopt_payloads,
    capture_evidence,
    capture_from_row,
    capture_key,
    capture_row,
    check_store,
    fetch_and_inspect,
    fetch_bls_copies,
    fetch_current,
    fetch_vintages,
    inspect_zip,
    inventory_rows,
    main,
    merge_captures,
    newest_member,
    parse_release_index,
    payload_digest,
    read_csv,
    record_run,
    select_vintages,
    staging,
    store_folder,
    stored_file,
    vintage_rows,
    write_csv,
)
from zip_fixtures import make_dated_zip, make_zip

AE = "all_employees_specifications"
OTHER = "other_inputs"


def zip_a() -> bytes:
    return make_zip({"outliers.xlsx": b"AO"})


def zip_b() -> bytes:
    return make_zip({"ces.spec.ae/AE1011330000.spc": b"series{}"})


def store_of(tmp_path: Path) -> Path:
    return tmp_path / "data" / "seasonal" / "raw"


def store_zips(store: Path, *artifact_and_payload: tuple[str, bytes]) -> None:
    """Commit the ZIPs in one run, the way a network command does."""
    with staging(store) as run:
        for artifact, payload in artifact_and_payload:
            run.put(artifact, payload)
        run.promote()


def snapshot(folder: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(folder)): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


# --- Names ---------------------------------------------------------------------------------


def test_the_store_lives_under_data_seasonal_raw():
    assert STORE_DIR == ROOT / "data" / "seasonal" / "raw"


def test_an_artifact_folder_is_its_current_url_file_name_without_zip():
    assert store_folder(AE) == "ces.spec.ae"
    assert store_folder(OTHER) == "ces.spec.other"


def test_an_unknown_artifact_has_no_folder():
    with pytest.raises(ValueError, match="unknown artifact"):
        store_folder("hours_and_earnings")
    with pytest.raises(ValueError, match="unknown artifact"):
        stored_file("hours_and_earnings", "D")


# --- Storing -------------------------------------------------------------------------------


def test_a_stored_zip_is_named_by_its_digest_in_its_artifact_folder(tmp_path):
    store = tmp_path / "data" / "seasonal" / "raw"
    payload = zip_a()
    with staging(store) as run:
        digest = run.put(OTHER, payload)
        run.promote()
    assert digest == payload_digest(payload)
    assert (store / "ces.spec.other" / f"{digest}.zip").read_bytes() == payload
    assert stored_file("all_employees_specifications", "D") == "ces.spec.ae/D.zip"


def test_the_same_bytes_are_stored_once(tmp_path):
    store = tmp_path / "data" / "seasonal" / "raw"
    payload = zip_a()
    with staging(store) as run:
        run.put(OTHER, payload)
        run.put(OTHER, payload)
        run.promote()
    with staging(store) as run:
        run.put(OTHER, payload)
        run.promote()
    assert len(list(store.rglob("*.zip"))) == 1
    assert len(read_csv(store / STORE_MANIFEST)) == 1


def test_the_same_bytes_under_two_artifacts_are_two_files(tmp_path):
    store = store_of(tmp_path)
    payload = zip_a()
    store_zips(store, (AE, payload), (OTHER, payload))
    digest = payload_digest(payload)
    assert sorted(snapshot(store)) == [
        f"ces.spec.ae/{digest}.zip",
        f"ces.spec.other/{digest}.zip",
        STORE_MANIFEST,
    ]


def test_has_is_true_for_a_staged_and_for_a_committed_zip_only(tmp_path):
    store = store_of(tmp_path)
    committed, staged = zip_a(), zip_b()
    store_zips(store, (OTHER, committed))
    with staging(store) as run:
        assert run.has(OTHER, payload_digest(committed))
        assert not run.has(OTHER, payload_digest(staged))
        run.put(OTHER, staged)
        assert run.has(OTHER, payload_digest(staged))
        assert not run.has(AE, payload_digest(staged))


def test_different_bytes_under_a_stored_name_are_an_error(tmp_path):
    store = tmp_path / "data" / "seasonal" / "raw"
    payload = zip_a()
    named = store / stored_file(OTHER, payload_digest(payload))
    named.parent.mkdir(parents=True)
    named.write_bytes(b"other bytes")
    with staging(store) as run:
        with pytest.raises(ValueError, match="different payload"):
            run.put(OTHER, payload)
        run.promote()
    assert named.read_bytes() == b"other bytes"
    assert [path.name for path in store.rglob("*.zip")] == [named.name]


def test_different_bytes_under_a_name_staged_in_the_run_are_an_error(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(archive_inventory, "payload_digest", lambda payload: "SAME")
    store = store_of(tmp_path)
    with staging(store) as run:
        run.put(OTHER, zip_a())
        with pytest.raises(ValueError, match="different payload under ces.spec.other"):
            run.put(OTHER, zip_b())
        run.promote()
    assert (store / "ces.spec.other" / "SAME.zip").read_bytes() == zip_a()


def test_a_stored_name_with_same_length_but_different_bytes_is_an_error(tmp_path):
    store = store_of(tmp_path)
    payload = zip_a()
    named = store / stored_file(OTHER, payload_digest(payload))
    named.parent.mkdir(parents=True)
    named.write_bytes(bytes(len(payload)))
    with staging(store) as run, pytest.raises(ValueError, match="different payload"):
        run.put(OTHER, payload)


def test_the_staging_folder_sits_beside_the_store(tmp_path):
    store = store_of(tmp_path)
    with staging(store):
        assert [
            path.name.startswith(".staging-") for path in store.parent.iterdir()
        ] == [True]


def test_the_manifest_keeps_committed_rows_and_adds_staged_zips(tmp_path):
    store = tmp_path / "data" / "seasonal" / "raw"
    payload_a, payload_b = zip_a(), zip_b()
    store_zips(store, (OTHER, payload_a))
    row_a = (store / STORE_MANIFEST).read_text().splitlines()[1]
    store_zips(store, (AE, payload_b))

    lines = (store / STORE_MANIFEST).read_text().splitlines()
    assert lines[0] == ",".join(STORE_MANIFEST_COLUMNS)
    rows = read_csv(store / STORE_MANIFEST)
    expected = [
        {
            "file": stored_file(artifact, payload_digest(payload)),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": str(len(payload)),
            "digest": payload_digest(payload),
        }
        for artifact, payload in ((AE, payload_b), (OTHER, payload_a))
    ]
    assert rows == expected
    assert [row["file"] for row in rows] == sorted(row["file"] for row in rows)
    assert lines[2] == row_a


def manifest_files(store: Path) -> list[str]:
    return [row["file"] for row in read_csv(store / STORE_MANIFEST)]


def test_the_manifest_rows_are_sorted_whatever_order_the_zips_arrive_in(tmp_path):
    ae_file = stored_file(AE, payload_digest(zip_b()))
    other_file = stored_file(OTHER, payload_digest(zip_a()))
    assert ae_file < other_file

    in_one_run = store_of(tmp_path / "one")
    store_zips(in_one_run, (OTHER, zip_a()), (AE, zip_b()))
    assert manifest_files(in_one_run) == [ae_file, other_file]

    across_runs = store_of(tmp_path / "two")
    store_zips(across_runs, (AE, zip_b()))
    store_zips(across_runs, (OTHER, zip_a()))
    assert manifest_files(across_runs) == [ae_file, other_file]

    # A new row that sorts between two committed rows.
    before, after = zip_a(), make_zip({"later.xlsx": b"later"})
    middle_run = store_of(tmp_path / "three")
    store_zips(middle_run, (AE, before), (AE, after))
    store_zips(middle_run, (AE, zip_b()))
    assert manifest_files(middle_run) == [
        stored_file(AE, payload_digest(before)),
        ae_file,
        stored_file(AE, payload_digest(after)),
    ]


def test_a_later_run_leaves_the_earlier_zips_of_the_same_folder_in_place(tmp_path):
    store = store_of(tmp_path)
    earlier = [zip_a(), make_zip({"later.xlsx": b"later"})]
    store_zips(store, *((AE, payload) for payload in earlier))
    store_zips(store, (AE, zip_b()))
    store_zips(store, (AE, zip_b()))
    for payload in earlier:
        stored = store / stored_file(AE, payload_digest(payload))
        assert stored.read_bytes() == payload
    assert len(list((store / "ces.spec.ae").glob("*.zip"))) == 3
    assert check_store(store) == []


def test_the_manifest_is_not_rehashed_from_the_committed_zips(tmp_path):
    store = store_of(tmp_path)
    store_zips(store, (OTHER, zip_a()))
    rows = read_csv(store / STORE_MANIFEST)
    rows[0]["sha256"] = "0" * 64
    write_csv(store / STORE_MANIFEST, rows, STORE_MANIFEST_COLUMNS)
    store_zips(store, (AE, zip_b()))
    kept = {row["file"]: row for row in read_csv(store / STORE_MANIFEST)}
    assert kept[rows[0]["file"]]["sha256"] == "0" * 64


def test_leaving_the_run_without_promote_stores_nothing(tmp_path):
    store = store_of(tmp_path)
    with staging(store) as run:
        run.put(OTHER, zip_a())
    assert not list(store.rglob("*"))
    assert list(store.parent.glob(".staging-*")) == []


# --- Promoting -----------------------------------------------------------------------------


def test_promote_moves_zips_before_the_files_that_name_them(tmp_path, monkeypatch):
    moved: list[Path] = []
    replace = pathlib.Path.replace

    def recording_replace(self, target):
        moved.append(Path(target))
        return replace(self, target)

    monkeypatch.setattr(pathlib.Path, "replace", recording_replace)
    store = tmp_path / "data" / "seasonal" / "raw"
    with staging(store) as run:
        run.put(AE, zip_b())
        run.put(OTHER, zip_a())
        run.path_for(
            tmp_path / "docs" / "inventory" / "archive-captures.csv"
        ).write_text("file\n")
        run.promote()
    suffixes = [target.suffix for target in moved]
    assert sorted(suffixes) == [".csv", ".csv", ".zip", ".zip"]
    last_zip = max(i for i, suffix in enumerate(suffixes) if suffix == ".zip")
    first_csv = min(i for i, suffix in enumerate(suffixes) if suffix == ".csv")
    assert last_zip < first_csv


def test_promote_writes_staged_files_to_their_targets_and_creates_their_folders(
    tmp_path,
):
    store = store_of(tmp_path)
    target = tmp_path / "docs" / "inventory" / "es-vintages.csv"
    with staging(store) as run:
        staged = run.path_for(target)
        assert run.path_for(target) == staged
        assert staged != target
        staged.write_text("reference_month\n")
        assert not target.exists()
        run.promote()
    assert target.read_text() == "reference_month\n"


def test_a_failed_run_leaves_the_committed_tree_untouched(tmp_path):
    store = tmp_path / "data" / "seasonal" / "raw"
    inventory = tmp_path / "docs" / "inventory" / "archive-captures.csv"
    with staging(store) as run:
        run.put(OTHER, zip_a())
        run.path_for(inventory).write_text("run 1\n")
        run.promote()
    before = snapshot(tmp_path)

    with pytest.raises(RuntimeError, match="boom"), staging(store) as run:
        run.put(AE, zip_b())
        run.path_for(inventory).write_text("run 2\n")
        raise RuntimeError("boom")

    assert snapshot(tmp_path) == before
    assert list(store.parent.glob(".staging-*")) == []


# --- Checking ------------------------------------------------------------------------------


def sound_store(tmp_path: Path) -> tuple[Path, str, str]:
    """A store of two ZIPs; returns it with the two ZIPs' file names."""
    store = store_of(tmp_path)
    store_zips(store, (AE, zip_b()), (OTHER, zip_a()))
    return (
        store,
        stored_file(AE, payload_digest(zip_b())),
        stored_file(OTHER, payload_digest(zip_a())),
    )


def test_a_sound_store_has_no_problems(tmp_path):
    store, _, _ = sound_store(tmp_path)
    assert check_store(store) == []


def test_a_missing_store_and_an_empty_store_have_no_problems(tmp_path):
    store = store_of(tmp_path)
    assert check_store(store) == []
    store.mkdir(parents=True)
    assert check_store(store) == []


def test_check_store_ignores_ds_store_files(tmp_path):
    store, _, _ = sound_store(tmp_path)
    (store / ".DS_Store").write_bytes(b"finder")
    (store / "ces.spec.ae" / ".DS_Store").write_bytes(b"finder")
    assert check_store(store) == []


def test_check_store_finds_a_changed_an_unlisted_and_a_missing_zip(tmp_path):
    store, ae_file, other_file = sound_store(tmp_path)

    changed = bytearray((store / ae_file).read_bytes())
    changed[-1] ^= 1
    (store / ae_file).write_bytes(bytes(changed))
    problems = check_store(store)
    assert len(problems) == 1
    assert ae_file in problems[0]
    (store / ae_file).write_bytes(zip_b())
    assert check_store(store) == []

    extra = make_zip({"extra.xlsx": b"extra"})
    extra_file = stored_file(OTHER, payload_digest(extra))
    (store / extra_file).write_bytes(extra)
    problems = check_store(store)
    assert len(problems) == 1
    assert extra_file in problems[0]
    assert "not in the manifest" in problems[0]
    (store / extra_file).unlink()

    (store / other_file).unlink()
    problems = check_store(store)
    assert len(problems) == 1
    assert other_file in problems[0]
    assert "missing" in problems[0]


@pytest.mark.parametrize("column", ["sha256", "bytes", "digest"])
def test_check_store_finds_a_manifest_row_that_differs_from_its_file(tmp_path, column):
    store, ae_file, _ = sound_store(tmp_path)
    rows = read_csv(store / STORE_MANIFEST)
    for row in rows:
        if row["file"] == ae_file:
            row[column] = "9" * 5
    write_csv(store / STORE_MANIFEST, rows, STORE_MANIFEST_COLUMNS)
    problems = check_store(store)
    assert len(problems) == 1
    assert ae_file in problems[0]
    assert column in problems[0]


def test_check_store_finds_a_zip_whose_bytes_do_not_hash_to_its_name(tmp_path):
    store, ae_file, _ = sound_store(tmp_path)
    renamed = "ces.spec.ae/NOTTHEDIGEST.zip"
    (store / ae_file).rename(store / renamed)
    rows = read_csv(store / STORE_MANIFEST)
    for row in rows:
        if row["file"] == ae_file:
            row["file"] = renamed
    write_csv(store / STORE_MANIFEST, rows, STORE_MANIFEST_COLUMNS)
    problems = check_store(store)
    assert len(problems) == 1
    assert renamed in problems[0]


def test_check_store_finds_a_stray_file_that_is_not_a_zip(tmp_path):
    store, _, _ = sound_store(tmp_path)
    (store / "ces.spec.ae" / "notes.txt").write_text("hand edit")
    (store / "ces.spec.ae" / STORE_MANIFEST).write_text("not the manifest")
    problems = check_store(store)
    assert len(problems) == 2
    assert "ces.spec.ae/manifest.csv" in problems[0]
    assert "ces.spec.ae/notes.txt" in problems[1]


def test_check_store_without_a_manifest_finds_every_zip_unlisted(tmp_path):
    store, ae_file, other_file = sound_store(tmp_path)
    (store / STORE_MANIFEST).unlink()
    problems = check_store(store)
    assert [ae_file in problems[0], other_file in problems[1]] == [True, True]
    assert len(problems) == 2


# --- Merging -------------------------------------------------------------------------------

ARCHIVE_URL = (
    "https://web.archive.org/web/20210318120000id_/https://www.bls.gov/ces.spec.ae.zip"
)
FINGERPRINTS = (
    "specification",
    "prior_adjustment",
    "outliers",
    "unexpected",
    "newest_member",
)


def row(**changes) -> Capture:
    """One opened Internet Archive copy, with `changes` applied."""
    base = Capture(
        artifact=AE,
        source="internet_archive",
        timestamp=datetime(2021, 3, 18, 12, 0, tzinfo=UTC),
        url=ARCHIVE_URL,
        status="200",
        digest="D",
        outliers="o",
    )
    return replace(base, **changes)


def test_a_capture_key_is_artifact_source_url_timestamp():
    capture = row()
    assert capture_key(capture) == (
        AE,
        "internet_archive",
        ARCHIVE_URL,
        datetime(2021, 3, 18, 12, 0, tzinfo=UTC),
    )


def test_a_committed_row_the_run_does_not_observe_stays():
    bls = row(source="bls_current", url="https://www.bls.gov/ces.spec.ae.zip")
    archived = row()
    assert merge_captures([bls, archived], []) == [bls, archived]


def test_an_observed_row_with_a_new_key_is_added():
    committed = row()
    later = row(timestamp=datetime(2021, 4, 1, tzinfo=UTC))
    other_url = row(url=ARCHIVE_URL + "?other")
    other_artifact = row(artifact=OTHER)
    other_source = row(source="bls_current")
    for added in (later, other_url, other_artifact, other_source):
        merged = merge_captures([committed], [added])
        assert len(merged) == 2
        assert committed in merged
        assert added in merged
    assert merge_captures([], [committed]) == [committed]


def test_a_committed_opened_row_keeps_its_credit_when_its_replay_fails():
    assert merge_captures([row()], [row(status="unreplayable", outliers="")]) == [row()]


@pytest.mark.parametrize("observed_status", ["301", "403", "-", "unreplayable"])
def test_a_committed_opened_row_stays_against_any_observation_without_a_payload(
    observed_status,
):
    observed = row(status=observed_status, outliers="")
    assert merge_captures([row()], [observed]) == [row()]


def test_an_opened_copy_with_other_fingerprints_is_an_error():
    with pytest.raises(ValueError, match=re.escape(ARCHIVE_URL)):
        merge_captures([row()], [row(outliers="x")])


@pytest.mark.parametrize("column", FINGERPRINTS)
def test_an_opened_copy_with_any_other_fingerprint_or_member_date_is_an_error(column):
    committed = row(**{column: "same"})
    with pytest.raises(ValueError, match=re.escape(ARCHIVE_URL)):
        merge_captures([committed], [row(**{column: "other"})])


@pytest.mark.parametrize("column", FINGERPRINTS)
def test_an_opened_copy_with_the_same_fingerprints_leaves_the_row_unchanged(column):
    committed = row(**{column: "same"})
    assert merge_captures([committed], [row(**{column: "same"})]) == [committed]


def test_a_payload_lost_row_never_changes():
    lost = row(status="payload_lost", outliers="")
    assert merge_captures([lost], [row()]) == [lost]
    assert merge_captures([lost], [row(outliers="x")]) == [lost]
    assert merge_captures([lost], [row(status="unreplayable", outliers="")]) == [lost]
    assert merge_captures([lost], []) == [lost]


@pytest.mark.parametrize("committed_status", ["301", "403", "-", "unreplayable"])
def test_a_row_without_a_payload_takes_the_new_observation(committed_status):
    committed = row(status=committed_status, outliers="")
    assert merge_captures([committed], [row()]) == [row()]


@pytest.mark.parametrize("committed_status", ["301", "403", "-", "unreplayable"])
def test_a_row_without_a_payload_takes_an_observation_that_has_none_either(
    committed_status,
):
    committed = row(status=committed_status, outliers="")
    observed = row(
        status="unreplayable" if committed_status != "unreplayable" else "301"
    )
    assert merge_captures([committed], [observed]) == [observed]


@pytest.mark.parametrize("committed_status", ["302", "404", "503"])
def test_a_row_with_any_other_status_takes_the_new_observation(committed_status):
    committed = row(status=committed_status, outliers="")
    assert merge_captures([committed], [row()]) == [row()]


def test_a_listed_digest_that_differs_from_the_committed_one_is_an_error():
    with pytest.raises(ValueError, match=re.escape(ARCHIVE_URL)):
        merge_captures([row()], [row(digest="E")])


@pytest.mark.parametrize(
    "committed_status", ["200", "payload_lost", "301", "403", "-", "unreplayable"]
)
def test_a_listed_digest_is_checked_whatever_the_committed_status(committed_status):
    committed = row(status=committed_status)
    with pytest.raises(ValueError, match=re.escape(ARCHIVE_URL)):
        merge_captures([committed], [row(digest="E")])


@pytest.mark.parametrize("observed_status", ["301", "403", "-", "unreplayable"])
def test_a_listed_digest_is_checked_whatever_the_observed_status(observed_status):
    with pytest.raises(ValueError, match=re.escape(ARCHIVE_URL)):
        merge_captures([row()], [row(status=observed_status, digest="E")])


def test_rows_listed_twice_pair_as_a_multiset():
    r = row()
    assert merge_captures([r, r], [r]) == [r, r]
    assert merge_captures([r, r], [r, r]) == [r, r]
    assert merge_captures([r, r], [r, r, r]) == [r, r, r]


def test_rows_listed_twice_for_the_first_time_stay_two_rows():
    r = row(status="301", outliers="")
    assert merge_captures([], [r, r]) == [r, r]
    assert merge_captures([r], [r, r]) == [r, r]


def test_the_ith_committed_row_pairs_with_the_ith_observed_row():
    committed = [row(status="301", outliers=""), row()]
    observed = [row(status="403", outliers=""), row(status="unreplayable", outliers="")]
    assert merge_captures(committed, observed) == [observed[0], row()]


def test_an_extra_committed_row_under_a_key_stays_and_an_extra_observed_row_is_added():
    first = row(status="301", outliers="")
    second = row(status="403", outliers="")
    third = row(status="unreplayable", outliers="")
    assert merge_captures([first, second], [third]) == [third, second]
    assert merge_captures([first], [second, third]) == [second, third]


def test_the_inputs_are_left_as_they_were():
    committed = [row(status="301", outliers="")]
    observed = [row(), row(url=ARCHIVE_URL + "?other")]
    merge_captures(committed, observed)
    assert committed == [row(status="301", outliers="")]
    assert observed == [row(), row(url=ARCHIVE_URL + "?other")]


def test_merged_rows_sort_by_artifact_source_timestamp_url():
    early = datetime(2020, 1, 1, tzinfo=UTC)
    late = datetime(2021, 1, 1, tzinfo=UTC)
    in_order = [
        row(artifact=AE, source="bls_current", timestamp=late, url="https://a"),
        row(artifact=AE, source="internet_archive", timestamp=early, url="https://z"),
        row(artifact=AE, source="internet_archive", timestamp=late, url="https://a"),
        row(artifact=AE, source="internet_archive", timestamp=late, url="https://b"),
        row(artifact=OTHER, source="bls_current", timestamp=early, url="https://a"),
    ]
    assert merge_captures([], list(reversed(in_order))) == in_order
    assert merge_captures(list(reversed(in_order)), []) == in_order


def test_doubled_rows_keep_their_order_through_the_sort():
    first = row(status="403", outliers="")
    second = row(status="301", outliers="")
    assert merge_captures([first, second], []) == [first, second]
    assert merge_captures([second, first], []) == [second, first]


def test_an_unchanged_listing_returns_the_committed_rows_as_they_were():
    rows = [
        row(source="bls_current", url="https://www.bls.gov/ces.spec.ae.zip"),
        row(status="301", outliers=""),
        row(status="301", outliers=""),
        row(timestamp=datetime(2021, 4, 1, tzinfo=UTC)),
        row(artifact=OTHER, status="payload_lost", outliers=""),
    ]
    once = merge_captures(rows, rows)
    assert once == merge_captures(rows, [])
    assert merge_captures(once, rows) == once


# --- Commands ------------------------------------------------------------------------------

OTHER_URL = "https://www.bls.gov/web/empsit/ces.spec.other.zip"
OTHER_KEY = "bls.gov/web/empsit/ces.spec.other.zip"
ORIGINAL = "http://www.bls.gov/web/empsit/ces.spec.other.zip"
INDEX_EXCERPT = (
    Path(__file__).parent
    / "fixtures"
    / "archive_inventory"
    / "empsit-index-excerpt.html"
)
T1 = datetime(2026, 10, 5, 15, tzinfo=UTC)
T2 = datetime(2026, 11, 10, 15, tzinfo=UTC)
BOTH_COMMANDS = pytest.mark.parametrize(
    "command", [fetch_current, capture_evidence], ids=["current", "captures"]
)


def replay_url(moment: datetime) -> str:
    return archive_inventory.REPLAY_URL.format(
        stamp=f"{moment:%Y%m%d%H%M%S}", original=ORIGINAL
    )


def listed(moment: datetime, status: str, digest: str):
    """One line of the Internet Archive's listing, as `list_captures` returns it."""
    return (moment, ORIGINAL, status, digest)


def archive_copy(moment: datetime, status: str, digest: str, **changes) -> Capture:
    return Capture(
        OTHER,
        "internet_archive",
        moment,
        replay_url(moment),
        status,
        digest,
        **changes,
    )


def monthly_zip(made: tuple[int, ...], tag: bytes) -> bytes:
    """Prior-adjustment and outlier files dated `made`, so made for the release that follows."""
    return make_dated_zip(
        [
            ("prior_adjustment_file.xlsx", made, b"P" + tag),
            ("outliers.xlsx", made, b"O" + tag),
        ]
    )


def opened(payload: bytes) -> dict[str, str]:
    """The fingerprints and member date that opening `payload` records."""
    return {**inspect_zip(payload), "newest_member": newest_member(payload)}


def fake_network(monkeypatch, *, bls, listing=None, replays=None) -> list[str]:
    """Serve BLS, the release index and the Internet Archive from memory.

    `bls` maps a current URL, and `replays` a replay URL, to bytes, to an exception to raise, or
    to a list of these to serve one after another. `listing` maps a CDX key to its lines.
    Returns the list that collects every URL fetched.
    """
    listing, replays = listing or {}, replays or {}
    fetched: list[str] = []
    index = INDEX_EXCERPT.read_bytes()

    def fake_fetch(url):
        fetched.append(url)
        if url == RELEASE_INDEX_URL:
            return index
        served = bls[url] if url in bls else replays[url]
        if isinstance(served, list):
            served = served.pop(0)
        if isinstance(served, Exception):
            raise served
        return served

    other = next(a for a in archive_inventory.ARTIFACTS if a.name == OTHER)
    monkeypatch.setattr(
        archive_inventory,
        "ARTIFACTS",
        (replace(other, archive_keys=other.archive_keys[:1]),),
    )
    monkeypatch.setattr(archive_inventory, "fetch", fake_fetch)
    monkeypatch.setattr(
        archive_inventory, "list_captures", lambda key: listing.get(key, [])
    )
    monkeypatch.setattr(archive_inventory, "sleep", lambda seconds: None)
    return fetched


def capture_lines(paths: EvidencePaths) -> list[str]:
    return paths.captures.read_text(encoding="utf-8").splitlines()[1:]


def lines_of(paths: EvidencePaths, source: str) -> list[str]:
    column = CAPTURE_COLUMNS.index("source")
    return [
        line
        for line in capture_lines(paths)
        if next(csv.reader([line]))[column] == source
    ]


def commit(paths: EvidencePaths, *captures: Capture) -> None:
    """Commit rows in the order a run writes them."""
    ordered = sorted(captures, key=attrgetter("artifact", "source", "timestamp", "url"))
    write_csv(paths.captures, [capture_row(c) for c in ordered], CAPTURE_COLUMNS)


OLD_VINTAGE = Vintage(
    date(2026, 8, 1),
    date(2026, 9, 4),
    "https://www.bls.gov/news.release/archives/empsit_09042026.htm",
)


def commit_everything(paths: EvidencePaths) -> Capture:
    """Commit an old vintage list, one stored ZIP and the one Archive row that names it."""
    payload = zip_b()
    kept = archive_copy(
        datetime(2015, 3, 21, 6, 17, 57, tzinfo=UTC),
        "200",
        payload_digest(payload),
        outliers="o",
        newest_member="2015-03-20T12:00:00",
    )
    write_csv(paths.vintages, vintage_rows([OLD_VINTAGE]), VINTAGE_COLUMNS)
    commit(paths, kept)
    store_zips(paths.store, (OTHER, payload))
    return kept


def no_staging_left(tmp_path: Path) -> bool:
    return not list(tmp_path.rglob(".staging-*"))


def credit(paths: EvidencePaths, month: str = "2026-09") -> dict[str, str]:
    """What the committed rows say about one release's files, as the inventory derives it."""
    vintages = select_vintages(
        parse_release_index(INDEX_EXCERPT.read_text(encoding="utf-8")), now=T2
    )
    committed = [capture_from_row(r) for r in read_csv(paths.captures)]
    found = next(
        r for r in inventory_rows(vintages, committed) if r["reference_month"] == month
    )
    return {
        name: found[name]
        for name in (*FILE_TYPES, *(f"{t}_evidence" for t in FILE_TYPES))
    }


def stored(paths: EvidencePaths, artifact: str, payload: bytes) -> bool:
    return (paths.store / stored_file(artifact, payload_digest(payload))).is_file()


@BOTH_COMMANDS
def test_two_runs_keep_the_first_copy_and_its_release_status(
    monkeypatch, tmp_path, command
):
    paths = EvidencePaths.at(tmp_path)
    x = monthly_zip((2026, 9, 29, 12, 0, 0), b"X")
    y = monthly_zip((2026, 10, 30, 12, 0, 0), b"Y")
    fake_network(monkeypatch, bls={OTHER_URL: x})
    command(T1, paths)
    after_first = capture_lines(paths)
    first_credit = credit(paths)
    fake_network(monkeypatch, bls={OTHER_URL: y})
    command(T2, paths)
    assert after_first[0] in capture_lines(paths)
    assert stored(paths, OTHER, x)
    assert stored(paths, OTHER, y)
    assert credit(paths) == first_credit


@BOTH_COMMANDS
def test_the_first_copy_credits_the_release_its_files_were_made_for(
    monkeypatch, tmp_path, command
):
    paths = EvidencePaths.at(tmp_path)
    fake_network(
        monkeypatch, bls={OTHER_URL: monthly_zip((2026, 9, 29, 12, 0, 0), b"X")}
    )
    command(T1, paths)
    assert credit(paths) == {
        "specification": "not_archived",
        "prior_adjustment": BLS_SOURCE,
        "outliers": BLS_SOURCE,
        "specification_evidence": "",
        "prior_adjustment_evidence": OTHER_URL,
        "outliers_evidence": OTHER_URL,
    }
    assert credit(paths, "2026-08")["outliers"] == "not_archived"


@BOTH_COMMANDS
def test_the_second_run_adds_its_own_row_beside_the_first(
    monkeypatch, tmp_path, command
):
    paths = EvidencePaths.at(tmp_path)
    fake_network(
        monkeypatch, bls={OTHER_URL: monthly_zip((2026, 9, 29, 12, 0, 0), b"X")}
    )
    command(T1, paths)
    fake_network(
        monkeypatch, bls={OTHER_URL: monthly_zip((2026, 10, 30, 12, 0, 0), b"Y")}
    )
    command(T2, paths)
    assert [(r["source"], r["timestamp"]) for r in read_csv(paths.captures)] == [
        (BLS_SOURCE, "2026-10-05T15:00:00Z"),
        (BLS_SOURCE, "2026-11-10T15:00:00Z"),
    ]


def test_current_makes_no_internet_archive_request_and_keeps_the_archive_rows(
    monkeypatch, tmp_path
):
    paths = EvidencePaths.at(tmp_path)
    kept = archive_copy(
        datetime(2026, 9, 10, 12, tzinfo=UTC),
        "200",
        "KEPT",
        outliers="o",
        newest_member="2026-09-01T12:00:00",
    )
    moved = archive_copy(datetime(2017, 1, 2, 4, 57, 40, tzinfo=UTC), "301", "MOVED")
    commit(paths, kept, moved)
    archive_lines = lines_of(paths, "internet_archive")
    fetched = fake_network(monkeypatch, bls={OTHER_URL: zip_a()})

    def refuse(key):
        raise AssertionError(f"listed {key}")

    monkeypatch.setattr(archive_inventory, "list_captures", refuse)
    fetch_current(T1, paths)
    assert not any(urlsplit(url).hostname == "web.archive.org" for url in fetched)
    assert lines_of(paths, "internet_archive") == archive_lines
    assert len(archive_lines) == 2
    assert len(lines_of(paths, BLS_SOURCE)) == 1


def test_current_fetches_only_the_release_index_and_the_current_files(
    monkeypatch, tmp_path
):
    fetched = fake_network(monkeypatch, bls={OTHER_URL: zip_a()})
    fetch_current(T1, EvidencePaths.at(tmp_path))
    assert fetched == [RELEASE_INDEX_URL, OTHER_URL]


def test_current_does_not_report_a_committed_200_row_that_has_no_stored_zip(
    monkeypatch, tmp_path, capsys
):
    paths = EvidencePaths.at(tmp_path)
    commit(paths, archive_copy(datetime(2026, 9, 10, 12, tzinfo=UTC), "200", "KEPT"))
    fake_network(monkeypatch, bls={OTHER_URL: zip_a()})
    assert fetch_current(T1, paths) == 0
    assert capsys.readouterr().err == ""


@BOTH_COMMANDS
def test_the_same_bls_bytes_on_two_runs_add_two_rows_and_one_zip(
    monkeypatch, tmp_path, command
):
    paths = EvidencePaths.at(tmp_path)
    payload = zip_a()
    fake_network(monkeypatch, bls={OTHER_URL: payload})
    command(T1, paths)
    command(T2, paths)
    assert len(read_csv(paths.captures)) == 2
    assert len(list(paths.store.rglob("*.zip"))) == 1


@BOTH_COMMANDS
@pytest.mark.parametrize("down", [RELEASE_INDEX_URL, OTHER_URL])
def test_a_bls_fetch_that_fails_every_attempt_stops_the_run_and_writes_nothing(
    monkeypatch, tmp_path, command, down
):
    paths = EvidencePaths.at(tmp_path)
    commit_everything(paths)
    before = snapshot(tmp_path)
    fake_network(monkeypatch, bls={OTHER_URL: zip_a()})
    serving = archive_inventory.fetch

    def fetch_unless_down(url):
        if url == down:
            raise OSError("down")
        return serving(url)

    monkeypatch.setattr(archive_inventory, "fetch", fetch_unless_down)
    with pytest.raises(RuntimeError, match="failed after 5 attempts"):
        command(T1, paths)
    assert snapshot(tmp_path) == before
    assert no_staging_left(tmp_path)


def test_a_full_run_stores_every_opened_copy(monkeypatch, tmp_path):
    paths = EvidencePaths.at(tmp_path)
    payload = zip_b()
    copied = datetime(2026, 9, 10, 12, tzinfo=UTC)
    fake_network(
        monkeypatch,
        bls={OTHER_URL: zip_a()},
        listing={
            OTHER_KEY: [
                listed(copied, "200", payload_digest(payload)),
                listed(datetime(2017, 1, 2, tzinfo=UTC), "301", "MOVED"),
                listed(
                    datetime(2026, 9, 11, 12, tzinfo=UTC), "-", payload_digest(payload)
                ),
            ]
        },
        replays={replay_url(copied): payload},
    )
    assert capture_evidence(T1, paths) == 0
    for found in read_csv(paths.captures):
        if found["status"] == "200":
            assert (
                paths.store / stored_file(found["artifact"], found["digest"])
            ).is_file()


def test_a_full_run_records_each_copy_it_lists_and_leaves_a_sound_store(
    monkeypatch, tmp_path
):
    paths = EvidencePaths.at(tmp_path)
    payload = zip_b()
    copied = datetime(2026, 9, 10, 12, tzinfo=UTC)
    revisit = datetime(2026, 9, 11, 12, tzinfo=UTC)
    fake_network(
        monkeypatch,
        bls={OTHER_URL: zip_a()},
        listing={
            OTHER_KEY: [
                listed(copied, "200", payload_digest(payload)),
                listed(datetime(2017, 1, 2, tzinfo=UTC), "301", "MOVED"),
                listed(revisit, "-", payload_digest(payload)),
            ]
        },
        replays={replay_url(copied): payload},
    )
    capture_evidence(T1, paths)
    assert [(r["source"], r["status"]) for r in read_csv(paths.captures)] == [
        (BLS_SOURCE, "200"),
        ("internet_archive", "301"),
        ("internet_archive", "200"),
        ("internet_archive", "-"),
    ]
    assert stored(paths, OTHER, payload)
    assert stored(paths, OTHER, zip_a())
    assert check_store(paths.store) == []


def test_a_throttled_full_run_keeps_every_credit_and_exits_1(
    monkeypatch, tmp_path, capsys
):
    paths = EvidencePaths.at(tmp_path)
    lost = archive_copy(
        datetime(2026, 10, 3, 12, tzinfo=UTC),
        "200",
        "DIGESTA",
        outliers="fpA",
        newest_member="2026-09-29T12:00:00",
    )
    commit(paths, lost)
    line_of_lost = capture_lines(paths)[0]
    first_credit = credit(paths)
    new_zip = make_dated_zip([("outliers.xlsx", (2026, 9, 1, 12, 0, 0), b"OB")])
    new_copy = datetime(2026, 9, 10, 12, tzinfo=UTC)
    fake_network(
        monkeypatch,
        bls={OTHER_URL: make_zip({"readme.txt": b"hello"})},
        listing={
            OTHER_KEY: [
                listed(lost.timestamp, "200", "DIGESTA"),
                listed(new_copy, "200", payload_digest(new_zip)),
            ]
        },
        replays={
            lost.url: OSError("throttled"),
            replay_url(new_copy): new_zip,
        },
    )
    assert capture_evidence(T1, paths) == 1
    assert lost.url in capsys.readouterr().err
    assert line_of_lost in capture_lines(paths)
    assert stored(paths, OTHER, new_zip)
    assert credit(paths) == first_credit


def test_replayed_bytes_that_miss_their_digest_count_as_failed_attempts(
    monkeypatch, tmp_path
):
    paths = EvidencePaths.at(tmp_path)
    copied = datetime(2026, 9, 10, 12, tzinfo=UTC)
    fetched = fake_network(
        monkeypatch,
        bls={OTHER_URL: zip_a()},
        listing={OTHER_KEY: [listed(copied, "200", "NOTTHEPAYLOAD")]},
        replays={replay_url(copied): zip_b()},
    )
    capture_evidence(T1, paths)
    assert fetched.count(replay_url(copied)) == ATTEMPTS
    found = next(r for r in read_csv(paths.captures) if r["source"] != BLS_SOURCE)
    assert found["status"] == "unreplayable"
    assert not stored(paths, OTHER, zip_b())


def test_bytes_that_miss_the_digest_once_are_retried(monkeypatch, tmp_path):
    paths = EvidencePaths.at(tmp_path)
    copied = datetime(2026, 9, 10, 12, tzinfo=UTC)
    payload = zip_b()
    fetched = fake_network(
        monkeypatch,
        bls={OTHER_URL: zip_a()},
        listing={OTHER_KEY: [listed(copied, "200", payload_digest(payload))]},
        replays={replay_url(copied): [b"<html>throttled</html>", payload]},
    )
    assert capture_evidence(T1, paths) == 0
    assert fetched.count(replay_url(copied)) == 2
    found = next(r for r in read_csv(paths.captures) if r["source"] != BLS_SOURCE)
    assert found["status"] == "200"
    assert stored(paths, OTHER, payload)


def test_an_unchanged_listing_gives_byte_identical_archive_rows(monkeypatch, tmp_path):
    paths = EvidencePaths.at(tmp_path)
    redirect = datetime(2017, 1, 2, 4, 57, 40, tzinfo=UTC)
    fake_network(
        monkeypatch,
        bls={OTHER_URL: zip_a()},
        listing={
            OTHER_KEY: [
                listed(redirect, "301", "MOVED"),
                listed(redirect, "301", "MOVED"),
            ]
        },
    )
    capture_evidence(T1, paths)
    after_first = lines_of(paths, "internet_archive")
    capture_evidence(T2, paths)
    assert lines_of(paths, "internet_archive") == after_first
    assert len(after_first) == 2


def test_an_unchanged_listing_with_an_opened_copy_replays_it_and_keeps_its_row(
    monkeypatch, tmp_path
):
    paths = EvidencePaths.at(tmp_path)
    payload = zip_b()
    copied = datetime(2026, 9, 10, 12, tzinfo=UTC)
    fetched = fake_network(
        monkeypatch,
        bls={OTHER_URL: zip_a()},
        listing={OTHER_KEY: [listed(copied, "200", payload_digest(payload))]},
        replays={replay_url(copied): payload},
    )
    capture_evidence(T1, paths)
    after_first = lines_of(paths, "internet_archive")
    assert capture_evidence(T2, paths) == 0
    assert lines_of(paths, "internet_archive") == after_first
    assert fetched.count(replay_url(copied)) == 2


def test_a_doubled_200_row_stays_two_rows_with_one_zip(monkeypatch, tmp_path):
    paths = EvidencePaths.at(tmp_path)
    payload = zip_b()
    copied = datetime(2026, 9, 10, 12, tzinfo=UTC)
    fake_network(
        monkeypatch,
        bls={OTHER_URL: zip_a()},
        listing={
            OTHER_KEY: [listed(copied, "200", payload_digest(payload))] * 2,
        },
        replays={replay_url(copied): payload},
    )
    capture_evidence(T1, paths)
    after_first = lines_of(paths, "internet_archive")
    assert capture_evidence(T2, paths) == 0
    assert lines_of(paths, "internet_archive") == after_first
    assert len(after_first) == 2
    assert len(list(paths.store.rglob("*.zip"))) == 2


def test_a_replay_that_fails_in_an_unexpected_way_writes_nothing(monkeypatch, tmp_path):
    paths = EvidencePaths.at(tmp_path)
    commit_everything(paths)
    before = snapshot(tmp_path)
    copied = datetime(2026, 9, 10, 12, tzinfo=UTC)
    fake_network(
        monkeypatch,
        bls={OTHER_URL: zip_a()},
        listing={OTHER_KEY: [listed(copied, "200", payload_digest(zip_b()))]},
        replays={replay_url(copied): KeyError("not an Archive failure")},
    )
    with pytest.raises(KeyError):
        capture_evidence(T1, paths)
    assert snapshot(tmp_path) == before
    assert no_staging_left(tmp_path)


def test_a_run_that_fails_while_fetching_writes_nothing(monkeypatch, tmp_path):
    paths = EvidencePaths.at(tmp_path)
    commit_everything(paths)
    before = snapshot(tmp_path)
    fake_network(monkeypatch, bls={OTHER_URL: zip_a()})

    def fail(key):
        raise RuntimeError("the listing is down")

    monkeypatch.setattr(archive_inventory, "list_captures", fail)
    with pytest.raises(RuntimeError, match="the listing is down"):
        capture_evidence(T1, paths)
    assert snapshot(tmp_path) == before
    assert no_staging_left(tmp_path)


def test_a_run_that_fails_while_staging_writes_nothing(monkeypatch, tmp_path):
    paths = EvidencePaths.at(tmp_path)
    commit_everything(paths)
    copied = datetime(2026, 9, 10, 12, tzinfo=UTC)
    replayed = make_zip({"outliers.xlsx": b"AO-other"})
    # A stored ZIP whose bytes are not its name: staging the copy that names it raises, after
    # BLS's own ZIP has been staged.
    corrupt = paths.store / stored_file(OTHER, payload_digest(replayed))
    corrupt.write_bytes(b"not the bytes of its name")
    before = snapshot(tmp_path)
    fake_network(
        monkeypatch,
        bls={OTHER_URL: zip_a()},
        listing={OTHER_KEY: [listed(copied, "200", payload_digest(replayed))]},
        replays={replay_url(copied): replayed},
    )
    with pytest.raises(ValueError, match="different payload under"):
        capture_evidence(T1, paths)
    assert snapshot(tmp_path) == before
    assert not stored(EvidencePaths.at(tmp_path), OTHER, zip_a())
    assert no_staging_left(tmp_path)


def test_nothing_is_written_before_the_run_is_promoted(monkeypatch, tmp_path):
    paths = EvidencePaths.at(tmp_path)
    commit_everything(paths)
    before = snapshot(tmp_path)
    fake_network(monkeypatch, bls={OTHER_URL: zip_a()})

    def refuse(self):
        raise RuntimeError("stopped before the move")

    monkeypatch.setattr(archive_inventory.Staging, "promote", refuse)
    with pytest.raises(RuntimeError, match="stopped before the move"):
        fetch_current(T1, paths)
    assert snapshot(tmp_path) == before
    assert no_staging_left(tmp_path)


def test_a_listed_digest_that_contradicts_a_committed_row_writes_nothing(
    monkeypatch, tmp_path
):
    paths = EvidencePaths.at(tmp_path)
    kept = commit_everything(paths)
    before = snapshot(tmp_path)
    contradicting = zip_a()
    fake_network(
        monkeypatch,
        bls={OTHER_URL: zip_a()},
        listing={
            OTHER_KEY: [listed(kept.timestamp, "200", payload_digest(contradicting))]
        },
        replays={kept.url: contradicting},
    )
    with pytest.raises(ValueError, match=re.escape(kept.url)):
        capture_evidence(T1, paths)
    assert snapshot(tmp_path) == before
    assert no_staging_left(tmp_path)


def test_an_opened_copy_that_contradicts_the_committed_fingerprints_writes_nothing(
    monkeypatch, tmp_path
):
    paths = EvidencePaths.at(tmp_path)
    payload = zip_a()
    recorded = archive_copy(
        datetime(2026, 9, 10, 12, tzinfo=UTC),
        "200",
        payload_digest(payload),
        **{**opened(payload), "outliers": "someotherfp"},
    )
    commit(paths, recorded)
    before = snapshot(tmp_path)
    fake_network(
        monkeypatch,
        bls={OTHER_URL: zip_b()},
        listing={OTHER_KEY: [listed(recorded.timestamp, "200", recorded.digest)]},
        replays={recorded.url: payload},
    )
    with pytest.raises(ValueError, match=re.escape(recorded.url)):
        capture_evidence(T1, paths)
    assert snapshot(tmp_path) == before


def test_a_committed_200_row_without_a_zip_gets_its_zip_when_its_bytes_match(
    monkeypatch, tmp_path, capsys
):
    paths = EvidencePaths.at(tmp_path)
    payload = zip_a()
    recorded = archive_copy(
        datetime(2026, 9, 10, 12, tzinfo=UTC),
        "200",
        payload_digest(payload),
        **opened(payload),
    )
    commit(paths, recorded)
    line = capture_lines(paths)[0]
    fake_network(
        monkeypatch,
        bls={OTHER_URL: zip_b()},
        listing={OTHER_KEY: [listed(recorded.timestamp, "200", recorded.digest)]},
        replays={recorded.url: payload},
    )
    assert capture_evidence(T1, paths) == 0
    assert capsys.readouterr().err == ""
    assert line in capture_lines(paths)
    assert stored(paths, OTHER, payload)


MISSING_ZIP = "no stored ZIP for"


@pytest.mark.parametrize(
    "situation", ["unlisted", "replay fails", "bytes miss the digest"]
)
def test_a_committed_200_row_that_still_has_no_zip_is_named_and_the_run_exits_1(
    monkeypatch, tmp_path, capsys, situation
):
    paths = EvidencePaths.at(tmp_path)
    recorded = archive_copy(
        datetime(2026, 9, 10, 12, tzinfo=UTC),
        "200",
        "DIGESTA",
        outliers="fpA",
        newest_member="2026-09-01T12:00:00",
    )
    commit(paths, recorded)
    line = capture_lines(paths)[0]
    listing = (
        {}
        if situation == "unlisted"
        else {OTHER_KEY: [listed(recorded.timestamp, "200", "DIGESTA")]}
    )
    replays = {
        "replay fails": {recorded.url: OSError("throttled")},
        "bytes miss the digest": {recorded.url: zip_b()},
    }.get(situation, {})
    fake_network(
        monkeypatch, bls={OTHER_URL: zip_a()}, listing=listing, replays=replays
    )
    assert capture_evidence(T1, paths) == 1
    assert f"{MISSING_ZIP} {recorded.url}" in capsys.readouterr().err
    assert line in capture_lines(paths)
    assert len(lines_of(paths, BLS_SOURCE)) == 1


@BOTH_COMMANDS
def test_unexplained_members_are_reported_after_the_run_is_promoted(
    monkeypatch, tmp_path, capsys, command
):
    paths = EvidencePaths.at(tmp_path)
    payload = make_zip({"mystery.dat": b"?"})
    fake_network(monkeypatch, bls={OTHER_URL: payload})
    assert command(T1, paths) == 1
    err = capsys.readouterr().err
    assert OTHER_URL in err
    assert "mystery.dat" in err
    assert stored(paths, OTHER, payload)
    assert len(capture_lines(paths)) == 1


def test_a_full_run_reports_unexplained_members_of_an_archive_copy(
    monkeypatch, tmp_path, capsys
):
    paths = EvidencePaths.at(tmp_path)
    payload = make_zip({"mystery.dat": b"?"})
    copied = datetime(2026, 9, 10, 12, tzinfo=UTC)
    fake_network(
        monkeypatch,
        bls={OTHER_URL: zip_a()},
        listing={OTHER_KEY: [listed(copied, "200", payload_digest(payload))]},
        replays={replay_url(copied): payload},
    )
    assert capture_evidence(T1, paths) == 1
    err = capsys.readouterr().err
    assert replay_url(copied) in err
    assert "mystery.dat" in err
    assert MISSING_ZIP not in err


def test_a_full_run_replays_with_a_pause_before_each_200_copy_only(
    monkeypatch, tmp_path
):
    events: list[str] = []
    first, second = zip_a(), zip_b()
    moments = [datetime(2026, 9, day, 12, tzinfo=UTC) for day in (10, 11, 12)]
    served = {replay_url(moments[0]): first, replay_url(moments[2]): second}
    listing = [
        listed(moments[0], "200", payload_digest(first)),
        listed(moments[1], "301", "MOVED"),
        listed(moments[2], "200", payload_digest(second)),
    ]
    monkeypatch.setattr(archive_inventory, "list_captures", lambda key: listing)
    monkeypatch.setattr(
        archive_inventory,
        "fetch",
        lambda url: events.append(url) or served[url],
    )
    monkeypatch.setattr(
        archive_inventory, "sleep", lambda seconds: events.append(f"sleep {seconds}")
    )
    monkeypatch.setattr(
        archive_inventory,
        "ARTIFACTS",
        (archive_inventory.Artifact(OTHER, OTHER_URL, (OTHER_KEY,)),),
    )
    archive_inventory.collect_archive_copies()
    pause = f"sleep {PAUSE_SECONDS}"
    assert events == [pause, replay_url(moments[0]), pause, replay_url(moments[2])]


def forbid_fetch(monkeypatch) -> None:
    """Fail any request, so a regression in a listing-only test cannot reach the network."""

    def forbidden(url):
        raise AssertionError(f"no request expected: {url}")

    monkeypatch.setattr(archive_inventory, "fetch", forbidden)


def forbid_pause_and_listing(monkeypatch) -> None:
    """Fail on a pause or an Archive listing, which a BLS-only path must never reach.

    A test that patches `fetch` for such a path calls this too, so a regression cannot sleep
    for real through `retrying` or list the Internet Archive.
    """

    def forbidden(*args):
        raise AssertionError("no pause or Archive listing expected")

    monkeypatch.setattr(archive_inventory, "sleep", forbidden)
    monkeypatch.setattr(archive_inventory, "list_captures", forbidden)


def test_every_archive_key_of_every_artifact_is_listed_in_order(monkeypatch):
    keys: list[str] = []
    forbid_fetch(monkeypatch)
    monkeypatch.setattr(archive_inventory, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        archive_inventory,
        "list_captures",
        lambda key: keys.append(key) or [(T1, ORIGINAL, "301", key)],
    )
    copies = archive_inventory.collect_archive_copies()
    assert keys == [
        key for artifact in archive_inventory.ARTIFACTS for key in artifact.archive_keys
    ]
    assert [(c.artifact, c.digest) for c, held in copies] == [
        (artifact.name, key)
        for artifact in archive_inventory.ARTIFACTS
        for key in artifact.archive_keys
    ]
    assert all(held is None for _, held in copies)


def test_a_listing_that_fails_once_is_retried(monkeypatch):
    calls: list[str] = []

    def flaky(key):
        calls.append(key)
        if len(calls) == 1:
            raise OSError("reset")
        return [(T1, ORIGINAL, "301", "MOVED")]

    forbid_fetch(monkeypatch)
    monkeypatch.setattr(archive_inventory, "list_captures", flaky)
    monkeypatch.setattr(archive_inventory, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        archive_inventory,
        "ARTIFACTS",
        (archive_inventory.Artifact(OTHER, OTHER_URL, (OTHER_KEY,)),),
    )
    assert len(archive_inventory.collect_archive_copies()) == 1
    assert calls == [OTHER_KEY, OTHER_KEY]


def test_fetch_and_inspect_returns_bytes_that_match_the_digest(monkeypatch):
    forbid_pause_and_listing(monkeypatch)
    payload = zip_a()
    monkeypatch.setattr(archive_inventory, "fetch", lambda url: payload)
    found, fingerprints = fetch_and_inspect(
        "https://example.test/x", payload_digest(payload)
    )
    assert found == payload
    assert fingerprints == opened(payload)


def test_fetch_and_inspect_without_a_digest_takes_any_bytes(monkeypatch):
    forbid_pause_and_listing(monkeypatch)
    payload = zip_a()
    monkeypatch.setattr(archive_inventory, "fetch", lambda url: payload)
    assert fetch_and_inspect("https://example.test/x")[0] == payload
    assert fetch_and_inspect("https://example.test/x", None)[0] == payload


def test_fetch_and_inspect_rejects_bytes_that_miss_the_digest(monkeypatch):
    forbid_pause_and_listing(monkeypatch)
    monkeypatch.setattr(archive_inventory, "fetch", lambda url: zip_a())
    with pytest.raises(ValueError, match=re.escape("https://example.test/x")):
        fetch_and_inspect("https://example.test/x", "NOTTHEPAYLOAD")


def test_fetch_vintages_drops_releases_still_under_embargo(monkeypatch):
    forbid_pause_and_listing(monkeypatch)
    monkeypatch.setattr(
        archive_inventory, "fetch", lambda url: INDEX_EXCERPT.read_bytes()
    )
    before = fetch_vintages(datetime(2026, 9, 13, 15, tzinfo=UTC))
    after = fetch_vintages(T1)
    assert max(v.release_date for v in before) == date(2026, 9, 4)
    assert max(v.release_date for v in after) == date(2026, 10, 2)


def test_fetch_vintages_retries_a_failed_fetch(monkeypatch):
    calls: list[str] = []

    def flaky(url):
        calls.append(url)
        if len(calls) == 1:
            raise OSError("reset")
        return INDEX_EXCERPT.read_bytes()

    monkeypatch.setattr(archive_inventory, "fetch", flaky)
    monkeypatch.setattr(archive_inventory, "sleep", lambda seconds: None)
    assert fetch_vintages(T1)
    assert calls == [RELEASE_INDEX_URL, RELEASE_INDEX_URL]


def test_fetch_bls_copies_gives_one_current_row_for_each_artifact(monkeypatch):
    forbid_pause_and_listing(monkeypatch)
    payloads = {
        artifact.current_url: make_zip({"outliers.xlsx": artifact.name.encode()})
        for artifact in archive_inventory.ARTIFACTS
    }
    monkeypatch.setattr(archive_inventory, "fetch", lambda url: payloads[url])
    copies = fetch_bls_copies(T1)
    assert [capture.artifact for capture, _ in copies] == [
        artifact.name for artifact in archive_inventory.ARTIFACTS
    ]
    for capture, payload in copies:
        assert payload == payloads[capture.url]
        assert capture == Capture(
            capture.artifact,
            BLS_SOURCE,
            T1,
            capture.url,
            "200",
            payload_digest(payload),
            **opened(payload),
        )


def test_evidence_paths_sit_under_a_root_and_the_default_is_the_repository():
    root = Path("/some/root")
    assert EvidencePaths.at(root) == EvidencePaths(
        root / "docs" / "inventory" / "es-vintages.csv",
        root / "docs" / "inventory" / "archive-captures.csv",
        root / "data" / "seasonal" / "raw",
    )
    assert EVIDENCE == EvidencePaths(VINTAGES_PATH, CAPTURES_PATH, STORE_DIR)
    assert EvidencePaths.at(ROOT) == EVIDENCE
    with pytest.raises(FrozenInstanceError):
        EVIDENCE.store = root


def test_record_run_reads_no_rows_from_a_missing_csv_and_returns_the_merged_rows(
    tmp_path,
):
    paths = EvidencePaths.at(tmp_path)
    payload = zip_a()
    fresh = Capture(
        OTHER,
        BLS_SOURCE,
        T1,
        OTHER_URL,
        "200",
        payload_digest(payload),
        **opened(payload),
    )
    plain = archive_copy(datetime(2017, 1, 2, tzinfo=UTC), "301", "MOVED")
    merged = record_run([(fresh, payload), (plain, None)], [OLD_VINTAGE], paths)
    assert merged == [fresh, plain]
    assert [capture_from_row(r) for r in read_csv(paths.captures)] == merged
    assert [r["reference_month"] for r in read_csv(paths.vintages)] == ["2026-08"]
    assert stored(paths, OTHER, payload)
    assert len(list(paths.store.rglob("*.zip"))) == 1


def test_a_run_rewrites_the_vintage_list_from_the_release_index(monkeypatch, tmp_path):
    paths = EvidencePaths.at(tmp_path)
    write_csv(paths.vintages, vintage_rows([OLD_VINTAGE]), VINTAGE_COLUMNS)
    fake_network(monkeypatch, bls={OTHER_URL: zip_a()})
    fetch_current(T1, paths)
    rows = read_csv(paths.vintages)
    assert rows == vintage_rows(
        select_vintages(
            parse_release_index(INDEX_EXCERPT.read_text(encoding="utf-8")), now=T1
        )
    )
    assert rows[-1]["release_date"] == "2026-10-02"


def test_an_unreplayable_copy_is_noted_on_stderr_with_its_replay_url(
    monkeypatch, tmp_path, capsys
):
    copied = datetime(2026, 9, 10, 12, tzinfo=UTC)
    fake_network(
        monkeypatch,
        bls={OTHER_URL: zip_a()},
        listing={OTHER_KEY: [listed(copied, "200", "DIGESTA")]},
        replays={replay_url(copied): OSError("throttled")},
    )
    assert capture_evidence(T1, EvidencePaths.at(tmp_path)) == 0
    err = capsys.readouterr().err
    assert f"{replay_url(copied)} failed after {ATTEMPTS} attempts" in err
    assert "recorded as unreplayable" in err
    assert MISSING_ZIP not in err


def test_current_stores_each_artifacts_zip_in_its_own_folder(monkeypatch, tmp_path):
    forbid_pause_and_listing(monkeypatch)
    paths = EvidencePaths.at(tmp_path)
    payloads = {
        artifact.current_url: make_zip({"outliers.xlsx": artifact.name.encode()})
        for artifact in archive_inventory.ARTIFACTS
    }
    monkeypatch.setattr(
        archive_inventory,
        "fetch",
        lambda url: (
            INDEX_EXCERPT.read_bytes() if url == RELEASE_INDEX_URL else payloads[url]
        ),
    )
    assert fetch_current(T1, paths) == 0
    for artifact in archive_inventory.ARTIFACTS:
        payload = payloads[artifact.current_url]
        assert stored(paths, artifact.name, payload)
        assert [
            r["url"] for r in read_csv(paths.captures) if r["artifact"] == artifact.name
        ] == [artifact.current_url]
    assert len(list(paths.store.rglob("*.zip"))) == len(archive_inventory.ARTIFACTS)
    assert check_store(paths.store) == []


def test_record_run_stores_each_zip_under_its_own_artifact(tmp_path):
    paths = EvidencePaths.at(tmp_path)
    for_ae, for_other = zip_a(), zip_b()
    held = [
        (
            Capture(
                artifact,
                BLS_SOURCE,
                T1,
                f"https://www.bls.gov/{artifact}.zip",
                "200",
                payload_digest(payload),
            ),
            payload,
        )
        for artifact, payload in ((AE, for_ae), (OTHER, for_other))
    ]
    record_run(held, [OLD_VINTAGE], paths)
    assert stored(paths, AE, for_ae)
    assert stored(paths, OTHER, for_other)
    assert not stored(paths, OTHER, for_ae)
    assert not stored(paths, AE, for_other)


def test_main_runs_the_command_it_names_with_the_time_now(monkeypatch):
    forbid_fetch(monkeypatch)
    forbid_pause_and_listing(monkeypatch)
    calls: list[tuple] = []
    for name, result in (("capture_evidence", 0), ("fetch_current", 1)):
        monkeypatch.setattr(
            archive_inventory,
            name,
            lambda *args, name=name, result=result: (
                calls.append((name, *args)) or result
            ),
        )
    monkeypatch.setattr(
        archive_inventory,
        "derive_inventory",
        lambda *args: calls.append(("derive_inventory", *args)) or 0,
    )
    assert main(["captures"]) == 0
    assert main(["current"]) == 1
    assert main(["inventory"]) == 0
    assert [call[0] for call in calls] == [
        "capture_evidence",
        "fetch_current",
        "derive_inventory",
    ]
    for _, now in calls[:2]:
        assert now.tzinfo is UTC
        assert now.microsecond == 0
        assert abs(datetime.now(UTC) - now) < timedelta(minutes=1)
    assert calls[2] == ("derive_inventory",)


@pytest.mark.parametrize("argv", [[], ["fetch"], ["captures", "extra"]])
def test_main_refuses_a_missing_or_unknown_command(monkeypatch, argv, capsys):
    def forbidden(*args):
        raise AssertionError("no command should run")

    # A wrong parser must fail here, never reach the network.
    for name in ("fetch", "capture_evidence", "fetch_current", "derive_inventory"):
        monkeypatch.setattr(archive_inventory, name, forbidden)
    with pytest.raises(SystemExit) as stopped:
        main(argv)
    assert stopped.value.code == 2


# --- Adopting ------------------------------------------------------------------------------

MOMENT = datetime(2026, 11, 15, 12, tzinfo=UTC)
FAKE_FINGERPRINT = "0000000000000000"
FAKE_MEMBER_DATE = "2020-01-01T00:00:00"


@pytest.fixture
def offline(monkeypatch) -> None:
    """`adopt` is offline: a request, a pause or an Archive listing fails the test."""
    forbid_fetch(monkeypatch)
    forbid_pause_and_listing(monkeypatch)


def rescued(tmp_path: Path, payload: bytes, name: str = "rescued.zip") -> Path:
    """A local copy of `payload`, the way a rescued file sits outside the repository."""
    file = tmp_path / "rescue" / name
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(payload)
    return file


def recorded(payload: bytes, artifact: str = OTHER, **changes) -> Capture:
    """A committed BLS row for `payload`'s digest, with what opening `payload` records."""
    base = Capture(
        artifact,
        BLS_SOURCE,
        MOMENT,
        f"https://www.bls.gov/{artifact}.zip",
        "200",
        payload_digest(payload),
        **opened(payload),
    )
    return replace(base, **changes)


def adopted_file(paths: EvidencePaths, artifact: str, payload: bytes) -> Path:
    return paths.store / stored_file(artifact, payload_digest(payload))


def refused(tmp_path: Path, paths: EvidencePaths, files: list[Path], capsys) -> str:
    """Run `adopt`, check that it exits 1 and changes no file at all, and return its stderr."""
    before = snapshot(tmp_path)
    assert adopt_payloads(files, paths) == 1
    assert snapshot(tmp_path) == before
    return capsys.readouterr().err


def test_adopt_stores_a_recorded_zip_and_fills_an_empty_member_date(
    tmp_path, offline, capsys
):
    paths = EvidencePaths.at(tmp_path)
    payload = zip_a()
    digest = payload_digest(payload)
    bare = recorded(payload, newest_member="")
    dated = archive_copy(
        datetime(2021, 3, 18, 12, tzinfo=UTC), "200", digest, **opened(payload)
    )
    revisit = archive_copy(datetime(2021, 3, 19, 12, tzinfo=UTC), "-", digest)
    elsewhere = recorded(zip_b(), status="payload_lost", newest_member="")
    commit(paths, bare, dated, revisit, elsewhere)
    write_csv(paths.vintages, vintage_rows([OLD_VINTAGE]), VINTAGE_COLUMNS)
    vintages = paths.vintages.read_bytes()
    before = read_csv(paths.captures)
    file = rescued(tmp_path, payload)

    assert adopt_payloads([file], paths) == 0

    assert adopted_file(paths, OTHER, payload).read_bytes() == payload
    assert read_csv(paths.captures) == [
        {**old, "newest_member": newest_member(payload)}
        if old == capture_row(bare)
        else old
        for old in before
    ]
    assert [r["file"] for r in read_csv(paths.store / STORE_MANIFEST)] == [
        stored_file(OTHER, digest)
    ]
    assert check_store(paths.store) == []
    assert paths.vintages.read_bytes() == vintages
    assert stored_file(OTHER, digest) in capsys.readouterr().out
    assert no_staging_left(tmp_path)


def test_adopt_refuses_a_digest_no_row_records(tmp_path, offline, capsys):
    paths = EvidencePaths.at(tmp_path)
    payload = zip_a()
    # Same fingerprints and member date as the ZIP, so only the digest sets it apart.
    commit(paths, recorded(payload, digest="ANOTHERDIGEST"))
    file = rescued(tmp_path, payload)
    assert str(file) in refused(tmp_path, paths, [file], capsys)
    assert not paths.store.exists()


@pytest.mark.parametrize("status", ["-", "301", "403", "404", "unreplayable"])
def test_adopt_refuses_a_digest_that_only_rows_with_another_status_record(
    tmp_path, offline, capsys, status
):
    paths = EvidencePaths.at(tmp_path)
    payload = zip_a()
    commit(paths, recorded(payload, status=status))
    file = rescued(tmp_path, payload)
    assert str(file) in refused(tmp_path, paths, [file], capsys)
    assert not paths.store.exists()


def test_adopt_refuses_a_digest_already_stored(tmp_path, offline, capsys):
    paths = EvidencePaths.at(tmp_path)
    payload = zip_a()
    commit(paths, recorded(payload, newest_member=""))
    store_zips(paths.store, (OTHER, payload))
    file = rescued(tmp_path, payload)
    err = refused(tmp_path, paths, [file], capsys)
    assert str(file) in err
    assert stored_file(OTHER, payload_digest(payload)) in err


def test_adopt_refuses_the_same_bytes_given_twice_and_stores_neither(
    tmp_path, offline, capsys
):
    paths = EvidencePaths.at(tmp_path)
    payload = zip_a()
    commit(paths, recorded(payload, newest_member=""))
    first, second = (
        rescued(tmp_path, payload, "a.zip"),
        rescued(tmp_path, payload, "b.zip"),
    )
    err = refused(tmp_path, paths, [first, second], capsys)
    assert str(second) in err
    assert str(first) not in err
    assert not paths.store.exists()


def test_adopt_refuses_bytes_whose_fingerprints_differ_from_the_row(
    tmp_path, offline, capsys
):
    paths = EvidencePaths.at(tmp_path)
    payload = zip_a()
    commit(paths, recorded(payload, outliers="0000000000000000"))
    file = rescued(tmp_path, payload)
    err = refused(tmp_path, paths, [file], capsys)
    assert str(file) in err
    assert "outliers" in err


@pytest.mark.parametrize("field", FINGERPRINTS[:4])
def test_adopt_refuses_a_difference_in_any_one_fingerprint(
    tmp_path, offline, capsys, field
):
    paths = EvidencePaths.at(tmp_path)
    payload = zip_a()
    commit(paths, recorded(payload, **{field: "stray.dat"}))
    file = rescued(tmp_path, payload)
    err = refused(tmp_path, paths, [file], capsys)
    assert str(file) in err
    assert field in err


def test_adopt_refuses_a_member_date_that_differs_from_the_row(
    tmp_path, offline, capsys
):
    paths = EvidencePaths.at(tmp_path)
    payload = zip_a()
    commit(paths, recorded(payload, newest_member=FAKE_MEMBER_DATE))
    file = rescued(tmp_path, payload)
    err = refused(tmp_path, paths, [file], capsys)
    assert str(file) in err
    assert "newest_member" in err


@pytest.mark.parametrize("bad_year", [2015, 2018, 2021])
@pytest.mark.parametrize(
    ("field", "wrong"),
    [("outliers", FAKE_FINGERPRINT), ("newest_member", FAKE_MEMBER_DATE)],
)
def test_adopt_checks_every_row_that_has_the_digest(
    tmp_path, offline, capsys, bad_year, field, wrong
):
    paths = EvidencePaths.at(tmp_path)
    payload = zip_a()
    commit(
        paths,
        *(
            recorded(
                payload,
                timestamp=datetime(year, 3, 18, 12, tzinfo=UTC),
                **({field: wrong} if year == bad_year else {}),
            )
            for year in (2015, 2018, 2021)
        ),
    )
    file = rescued(tmp_path, payload)
    assert str(file) in refused(tmp_path, paths, [file], capsys)


def test_adopt_refuses_rows_with_the_digest_under_two_artifacts(
    tmp_path, offline, capsys
):
    paths = EvidencePaths.at(tmp_path)
    payload = zip_a()
    commit(paths, recorded(payload, AE), recorded(payload, OTHER))
    file = rescued(tmp_path, payload)
    assert str(file) in refused(tmp_path, paths, [file], capsys)
    assert not paths.store.exists()


def test_adopt_turns_a_payload_lost_row_into_200(tmp_path, offline):
    paths = EvidencePaths.at(tmp_path)
    payload = zip_a()
    lost = recorded(payload, status="payload_lost", newest_member="")
    also_lost = recorded(
        payload,
        status="payload_lost",
        timestamp=datetime(2026, 9, 13, 12, tzinfo=UTC),
    )
    still_lost = recorded(zip_b(), status="payload_lost", newest_member="")
    commit(paths, lost, also_lost, still_lost)
    before = read_csv(paths.captures)
    assert adopt_payloads([rescued(tmp_path, payload)], paths) == 0
    assert read_csv(paths.captures) == [
        {**old, "status": "200", "newest_member": newest_member(payload)}
        if old["digest"] == payload_digest(payload)
        else old
        for old in before
    ]
    assert adopted_file(paths, OTHER, payload).read_bytes() == payload


def test_adopt_stores_each_zip_under_its_own_artifact_in_one_run(tmp_path, offline):
    paths = EvidencePaths.at(tmp_path)
    for_ae, for_other = zip_a(), zip_b()
    commit(
        paths,
        recorded(for_ae, AE, newest_member=""),
        recorded(for_other, OTHER, newest_member=""),
    )
    files = [
        rescued(tmp_path, for_ae, "ae.zip"),
        rescued(tmp_path, for_other, "other.zip"),
    ]
    assert adopt_payloads(files, paths) == 0
    assert adopted_file(paths, AE, for_ae).read_bytes() == for_ae
    assert adopted_file(paths, OTHER, for_other).read_bytes() == for_other
    assert len(list(paths.store.rglob("*.zip"))) == 2
    assert len(read_csv(paths.store / STORE_MANIFEST)) == 2
    assert [r["newest_member"] for r in read_csv(paths.captures)] == [
        newest_member(for_ae),
        newest_member(for_other),
    ]
    assert check_store(paths.store) == []


@pytest.mark.parametrize("bad_first", [True, False], ids=["bad first", "bad last"])
def test_adopt_checks_every_zip_before_it_stores_any(
    tmp_path, offline, capsys, bad_first
):
    paths = EvidencePaths.at(tmp_path)
    good, unknown = zip_a(), zip_b()
    commit(paths, recorded(good, newest_member=""))
    files = [rescued(tmp_path, good, "good.zip"), rescued(tmp_path, unknown, "bad.zip")]
    err = refused(tmp_path, paths, files[::-1] if bad_first else files, capsys)
    assert str(files[1]) in err
    assert str(files[0]) not in err
    assert not paths.store.exists()


def test_adopt_refuses_a_file_it_cannot_read(tmp_path, offline, capsys):
    paths = EvidencePaths.at(tmp_path)
    commit(paths, recorded(zip_a()))
    missing = tmp_path / "rescue" / "missing.zip"
    assert str(missing) in refused(tmp_path, paths, [missing], capsys)


def test_adopt_refuses_every_zip_when_no_captures_csv_exists(tmp_path, offline, capsys):
    paths = EvidencePaths.at(tmp_path)
    file = rescued(tmp_path, zip_a())
    assert str(file) in refused(tmp_path, paths, [file], capsys)
    assert not paths.captures.exists()
    assert not paths.store.exists()


def test_main_passes_the_adopt_zip_paths_and_the_evidence_to_adopt_payloads(
    monkeypatch,
):
    calls: list[tuple] = []
    monkeypatch.setattr(
        archive_inventory,
        "adopt_payloads",
        lambda *args: calls.append(args) or 1,
    )
    assert main(["adopt", "a.zip", "dir/b.zip"]) == 1
    assert calls == [([Path("a.zip"), Path("dir/b.zip")], EVIDENCE)]


def test_main_adopt_works_on_the_evidence_paths_it_names(
    tmp_path, monkeypatch, offline
):
    paths = EvidencePaths.at(tmp_path)
    monkeypatch.setattr(archive_inventory, "EVIDENCE", paths)
    payload = zip_a()
    commit(paths, recorded(payload, newest_member=""))
    assert main(["adopt", str(rescued(tmp_path, payload))]) == 0
    assert adopted_file(paths, OTHER, payload).read_bytes() == payload


def test_main_refuses_adopt_without_a_zip(monkeypatch):
    def forbidden(*args):
        raise AssertionError("adopt should not run")

    monkeypatch.setattr(archive_inventory, "adopt_payloads", forbidden)
    with pytest.raises(SystemExit) as stopped:
        main(["adopt"])
    assert stopped.value.code == 2


def test_adopt_runs_in_an_isolated_interpreter_on_a_copy_of_the_script_alone(tmp_path):
    # The copy has no sibling module to import, and its evidence paths sit under `tmp_path`.
    script = tmp_path / "scripts" / "archive_inventory.py"
    script.parent.mkdir()
    shutil.copy(ROOT / "scripts" / "archive_inventory.py", script)
    paths = EvidencePaths.at(tmp_path)
    payload = zip_a()
    commit(paths, recorded(payload, newest_member=""))
    file = rescued(tmp_path, payload)

    def adopt() -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-I", str(script), "adopt", str(file)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )

    first = adopt()
    assert first.stderr == ""
    assert first.returncode == 0
    assert adopted_file(paths, OTHER, payload).read_bytes() == payload
    assert [r["newest_member"] for r in read_csv(paths.captures)] == [
        newest_member(payload)
    ]
    # The store now holds the ZIP, so the same command refuses and exits 1.
    second = adopt()
    assert second.returncode == 1
    assert str(file) in second.stderr
    assert "Traceback" not in second.stderr


def test_archive_inventory_imports_only_the_standard_library():
    tree = ast.parse((ROOT / "scripts" / "archive_inventory.py").read_text("utf-8"))
    imported = {
        name
        for node in ast.walk(tree)
        for name in (
            [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else [("." * node.level) + (node.module or "")]
            if isinstance(node, ast.ImportFrom)
            else []
        )
    }
    assert {name.split(".")[0] for name in imported} <= sys.stdlib_module_names


# --- Repository rules ----------------------------------------------------------------------


def git_attributes(*paths: str) -> dict[str, dict[str, str]]:
    """What `git check-attr` reports for `text` and `diff` on each path, from the repository root."""
    result = subprocess.run(
        ["git", "check-attr", "text", "diff", "--", *paths],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    attributes: dict[str, dict[str, str]] = {}
    for line in result.stdout.splitlines():
        path, name, value = line.split(": ")
        attributes.setdefault(path, {})[name] = value
    return attributes


def test_git_keeps_stored_zips_binary_and_the_manifest_a_text_diff():
    manifest = "data/seasonal/raw/manifest.csv"
    zips = [
        "data/seasonal/raw/ces.spec.ae/DIGEST.zip",
        "data/seasonal/raw/ces.spec.other/DIGEST.zip",
    ]
    attributes = git_attributes(manifest, *zips)
    assert attributes[manifest] == {"text": "unset", "diff": "unspecified"}
    for stored in zips:
        assert attributes[stored] == {"text": "unset", "diff": "unset"}
