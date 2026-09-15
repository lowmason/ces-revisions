"""The Stage 3 build writes every artifact from the committed sources, with a manifest."""

import json
from dataclasses import fields

import polars as pl
import pytest

from ces_revisions.vintages.build import VintageBuild, build, write


@pytest.mark.slow
def test_build_writes_every_artifact_and_its_manifest(tmp_path):
    result = build()
    manifest = write(result, tmp_path)
    assert (
        json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == manifest
    )
    for field in fields(VintageBuild):
        frame = getattr(result, field.name)
        assert manifest[field.name]["rows"] == frame.height
        written = pl.read_parquet(tmp_path / f"{field.name}.parquet")
        assert written.equals(frame), field.name
