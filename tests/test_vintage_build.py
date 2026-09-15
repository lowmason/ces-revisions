"""The Stage 3 build writes every artifact from the committed sources, with a manifest."""

import json
from dataclasses import fields
from functools import cache

import polars as pl
import pytest
import vintage_data
from polars.testing import assert_frame_equal

from ces_revisions.vintages import accounting, differencing, panel, raw, revision_table
from ces_revisions.vintages.build import VintageBuild, build, write

FRAMES = [field.name for field in fields(VintageBuild)]
# Frames whose last step is a join or pivot, which promise no row order.
UNORDERED = {"stage_changes", "reconciliation", "revision_reconciliation"}


@pytest.fixture(scope="module")
def result() -> VintageBuild:
    return build()


@cache
def composed() -> dict[str, pl.DataFrame]:
    """Each frame as tests/vintage_data.py composes it for the fast tier's exit tests."""
    changes = vintage_data.stage_changes()
    return {
        "raw_values": vintage_data.raw_values(),
        "transformations": panel.TRANSFORMATIONS,
        "release_index": vintage_data.index(),
        "panel": vintage_data.long_panel(),
        "stage_labels": vintage_data.labels(),
        "stage_changes": changes,
        "reconciliation": differencing.reconcile(
            changes, vintage_data.table_estimates()
        ),
        "revision_reconciliation": differencing.revision_reconciliation(
            changes, vintage_data.table_cells()
        ),
        "average_checks": revision_table.average_checks(vintage_data.raw_values()),
        "accounting_decomposition": accounting.decomposition(
            vintage_data.identity_terms()
        ),
    }


@pytest.mark.slow
def test_build_writes_every_artifact_and_its_manifest(result, tmp_path):
    manifest = write(result, tmp_path)
    assert (
        json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == manifest
    )
    assert manifest["source_manifest_sha256"] == raw.file_sha256(
        raw.RAW_DIR / raw.MANIFEST
    )
    assert list(manifest["artifacts"]) == FRAMES
    for name in FRAMES:
        frame = getattr(result, name)
        assert manifest["artifacts"][name]["rows"] == frame.height
        written = pl.read_parquet(tmp_path / f"{name}.parquet")
        assert written.equals(frame), name


@pytest.mark.slow
@pytest.mark.parametrize("name", FRAMES)
def test_build_writes_the_frame_the_fast_tier_checks(result, name):
    """Each built frame equals the one tests/vintage_data.py composes for the exit tests."""
    assert_frame_equal(
        getattr(result, name), composed()[name], check_row_order=name not in UNORDERED
    )
