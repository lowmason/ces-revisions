"""Build every Stage 3 artifact from the committed sources, and write them to data/panel/."""

import json
from dataclasses import dataclass, fields
from pathlib import Path

import polars as pl

from ces_revisions.vintages import (
    accounting,
    differencing,
    panel,
    raw,
    release_index,
    revision_table,
    stages,
)
from ces_revisions.vintages.months import add_months


@dataclass(frozen=True)
class VintageBuild:
    raw_values: pl.DataFrame
    transformations: pl.DataFrame
    release_index: pl.DataFrame
    panel: pl.DataFrame
    stage_labels: pl.DataFrame
    stage_changes: pl.DataFrame
    reconciliation: pl.DataFrame
    revision_reconciliation: pl.DataFrame
    average_checks: pl.DataFrame
    accounting_decomposition: pl.DataFrame


def build(raw_dir: Path = raw.RAW_DIR) -> VintageBuild:
    values = raw.raw_values(raw_dir)
    index = release_index.build_release_index(raw_dir)
    cells = revision_table.monthly_cells(values)
    table_estimates = revision_table.estimates(cells)
    levels = panel.vintage_file_levels(values, index)
    changes = panel.revision_table_changes(values, index)
    labels = stages.build_stage_labels(
        levels, changes, index, raw.read_vintage_comments(raw_dir)
    )
    first = add_months(stages.FIRST_SECTOR_MONTH, -1)
    within = differencing.same_release_changes(
        levels.filter(pl.col("reference_month") >= first)
    )
    at_stages = differencing.stage_changes(within, labels)
    terms = accounting.identity_terms(
        accounting.accounting_changes(at_stages, table_estimates)
    )
    return VintageBuild(
        raw_values=values,
        transformations=panel.TRANSFORMATIONS,
        release_index=index,
        panel=panel.assemble_panel(
            levels, panel.rtdsm_levels(values, index), changes, labels
        ),
        stage_labels=labels,
        stage_changes=at_stages,
        reconciliation=differencing.reconcile(at_stages, table_estimates),
        revision_reconciliation=differencing.revision_reconciliation(at_stages, cells),
        average_checks=revision_table.average_checks(values),
        accounting_decomposition=accounting.decomposition(terms),
    )


def write(result: VintageBuild, out_dir: Path = raw.PANEL_DIR) -> dict[str, dict]:
    """Write each frame as parquet, with a manifest of row counts and content hashes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for field in fields(result):
        frame = getattr(result, field.name)
        frame.write_parquet(out_dir / f"{field.name}.parquet")
        manifest[field.name] = {
            "rows": frame.height,
            "sha256": raw.content_sha256(frame),
        }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
