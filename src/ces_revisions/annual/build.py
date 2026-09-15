"""Build every Stage 4 artifact from the two committed source archives."""

import json
from dataclasses import dataclass, fields
from pathlib import Path

import polars as pl

from ces_revisions.annual import (
    benchmarks,
    birth_death,
    contract,
    publications,
    qcew,
    raw,
    reconstructions,
    sample,
)
from ces_revisions.vintages import raw as vintage_raw
from ces_revisions.vintages.release_index import build_release_index


@dataclass(frozen=True)
class AnnualBuild:
    raw_values: pl.DataFrame
    transformations: pl.DataFrame
    publication_calendar: pl.DataFrame
    benchmarks: pl.DataFrame
    reconstruction_events: pl.DataFrame
    birth_death: pl.DataFrame
    qcew_revisions: pl.DataFrame
    qcew_precision: pl.DataFrame
    sample_panel: pl.DataFrame


def _raw_values(raw_dir: Path) -> pl.DataFrame:
    parts = [
        publications.publication_raw_values(raw_dir),
        benchmarks.table5_raw_values(raw_dir),
        benchmarks.benchmark_sector_raw_values(raw_dir),
        reconstructions.raw_values(raw_dir),
        birth_death.raw_values(raw_dir),
        qcew.raw_values(raw_dir),
        sample.raw_values(raw_dir),
    ]
    frame = pl.concat(parts).sort("cell_key")
    duplicates = frame.group_by("cell_key").len().filter(pl.col("len") > 1)
    if duplicates.height:
        raise ValueError(f"duplicate Stage 4 raw cells: {duplicates.rows()}")
    return frame


def build(
    raw_dir: Path = raw.RAW_DIR,
    stage3_raw_dir: Path = vintage_raw.RAW_DIR,
) -> AnnualBuild:
    release_index = build_release_index(stage3_raw_dir)
    calendar = publications.build_publication_calendar(release_index, raw_dir)
    qcew_revisions = qcew.build_qcew_revisions(calendar, raw_dir)
    result = AnnualBuild(
        raw_values=_raw_values(raw_dir),
        transformations=raw.TRANSFORMATIONS,
        publication_calendar=calendar,
        benchmarks=benchmarks.build_benchmarks(calendar, raw_dir),
        reconstruction_events=reconstructions.build_reconstruction_events(
            calendar, raw_dir
        ),
        birth_death=birth_death.build_birth_death(calendar, release_index, raw_dir),
        qcew_revisions=qcew_revisions,
        qcew_precision=qcew.build_qcew_precision(qcew_revisions),
        sample_panel=sample.build_sample_panel(calendar, raw_dir),
    )
    contract.validate(result, release_index)
    return result


def write(
    result: AnnualBuild,
    out_dir: Path = raw.PANEL_DIR,
    raw_dir: Path = raw.RAW_DIR,
    stage3_raw_dir: Path = vintage_raw.RAW_DIR,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for field in fields(result):
        frame = getattr(result, field.name)
        frame.write_parquet(out_dir / f"{field.name}.parquet")
        artifacts[field.name] = {
            "rows": frame.height,
            "sha256": raw.content_sha256(frame),
        }
    manifest = {
        "annual_source_manifest_sha256": raw.file_sha256(raw_dir / raw.MANIFEST),
        "stage3_source_manifest_sha256": vintage_raw.file_sha256(
            stage3_raw_dir / vintage_raw.MANIFEST
        ),
        "artifacts": artifacts,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
