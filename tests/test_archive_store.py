"""scripts/archive_inventory.py: the content-addressed store for seasonal-adjustment ZIPs."""

import hashlib
import pathlib
import subprocess
from pathlib import Path

import archive_inventory
import pytest
from archive_inventory import (
    ROOT,
    STORE_DIR,
    STORE_MANIFEST,
    STORE_MANIFEST_COLUMNS,
    check_store,
    payload_digest,
    read_csv,
    staging,
    store_folder,
    stored_file,
    write_csv,
)
from zip_fixtures import make_zip

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
