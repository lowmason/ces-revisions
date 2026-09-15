"""Stage 4's complete build and content-hashed artifact manifest."""

import json
from dataclasses import fields

import polars as pl

from ces_revisions.annual import build, raw
from ces_revisions.vintages import raw as vintage_raw


def test_build_exposes_every_stage_4_artifact():
    result = build.build()
    assert [field.name for field in fields(result)] == [
        "raw_values",
        "transformations",
        "publication_calendar",
        "benchmarks",
        "reconstruction_events",
        "birth_death",
        "qcew_revisions",
        "qcew_precision",
        "sample_panel",
    ]
    for field in fields(result):
        frame = getattr(result, field.name)
        assert isinstance(frame, pl.DataFrame)
        assert not frame.is_empty(), field.name
    assert result.raw_values["cell_key"].n_unique() == result.raw_values.height


def test_write_round_trips_every_frame_and_hashes_both_source_archives(tmp_path):
    result = build.build()
    out = tmp_path / "panel"
    manifest = build.write(result, out_dir=out)
    assert manifest["annual_source_manifest_sha256"] == raw.file_sha256(
        raw.RAW_DIR / raw.MANIFEST
    )
    assert manifest["stage3_source_manifest_sha256"] == vintage_raw.file_sha256(
        vintage_raw.RAW_DIR / vintage_raw.MANIFEST
    )
    for field in fields(result):
        expected = getattr(result, field.name)
        actual = pl.read_parquet(out / f"{field.name}.parquet")
        assert actual.equals(expected), field.name
        assert manifest["artifacts"][field.name] == {
            "rows": expected.height,
            "sha256": raw.content_sha256(expected),
        }
    assert json.loads((out / "manifest.json").read_text()) == manifest


def test_built_outputs_live_only_below_the_gitignored_panel_directory():
    assert raw.PANEL_DIR == raw.ROOT / "data" / "annual" / "panel"
    assert not str(raw.RAW_DIR).startswith(str(raw.PANEL_DIR))
