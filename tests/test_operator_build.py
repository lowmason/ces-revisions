"""Stage 5 artifact round trip and deterministic provenance record."""

import json
import os
import re
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import operator_artifacts
import operator_data
import polars as pl
import vintage_data

from ces_revisions.operators.build import operator_test_record, write


def test_assembled_artifacts_have_the_stage_gate_rows():
    result = operator_data.result()
    assert result.benchmark_fixtures.height == 3_312
    assert result.post_march_components.height == 2_484
    assert result.recoverability.height == 23
    assert result.b_to_m_components.height > 0
    assert result.recoverability["benchmark_year"].to_list() == list(range(2003, 2026))


def test_operator_test_record_is_deterministic_and_complete():
    first = operator_test_record(operator_data.result(), vintage_data.levels())
    second = operator_test_record(operator_data.result(), vintage_data.levels())
    assert first == second
    assert "generated_at" not in first
    assert first["schema_version"] == 1
    assert set(first["input_manifests"]) == {"stage3", "stage4"}
    assert all(len(value) == 64 for value in first["input_manifests"].values())
    assert first["checks"] == {
        "aggregation": {"cells": 496_310, "max_abs_residual_thousands": 0.0},
        "wedge": {
            "years": 23,
            "cells": 3_312,
            "reconstruction_supported_cells": 215,
            "max_abs_reproduction_residual_thousands": 0.0,
            "max_abs_non_event_rounding_residual_thousands": 21.0,
        },
        "post_march": {
            "years": 23,
            "april_october_cells": 1_932,
            "november_december_cells": 552,
            "max_abs_reproduction_residual_thousands": 0.0,
        },
        "recoverability": {
            "cumulative_job_change_years": 23,
            "inferred_years": 23,
            "recovered_years": 0,
        },
        "seasonal_mapping": {"max_abs_identity_residual_thousands": 0.0},
    }


def test_write_round_trips_every_frame_and_json_record(tmp_path: Path):
    result = operator_data.result()
    record = write(result, vintage_data.levels(), tmp_path)
    assert json.loads((tmp_path / "operator-test-results.json").read_text()) == record
    for field in fields(result):
        path = tmp_path / f"{field.name}.parquet"
        assert path.is_file()
        assert pl.read_parquet(path).equals(getattr(result, field.name))
        assert (
            record["artifacts"][field.name]["rows"]
            == getattr(result, field.name).height
        )
        assert len(record["artifacts"][field.name]["sha256"]) == 64


def test_offline_cli_writes_the_same_artifact_set(tmp_path: Path):
    assert operator_artifacts.main(["build", "--out-dir", str(tmp_path)]) == 0
    assert (tmp_path / "operator-test-results.json").is_file()
    assert {path.name for path in tmp_path.glob("*.parquet")} == {
        f"{field.name}.parquet" for field in fields(operator_data.result())
    }


def test_cli_module_enables_x64_in_a_fresh_interpreter():
    environment = os.environ.copy()
    environment["JAX_ENABLE_X64"] = "false"
    command = (
        "import jax, sys; "
        "assert not jax.config.x64_enabled; "
        "sys.path.insert(0, 'scripts'); "
        "import operator_artifacts; "
        "assert jax.config.x64_enabled"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_written_determination_has_one_row_per_year_and_the_selected_contract():
    text = Path("docs/decisions/link-relatives.md").read_text(encoding="utf-8")
    assert "**Status:** Accepted" in text
    assert "`cumulative_job_change` for all 23 benchmark years" in text
    rows = re.findall(r"^\| (20\d{2}) \|", text, flags=re.MULTILINE)
    assert rows == [str(year) for year in range(2003, 2026)]
    for row in operator_data.artifact("recoverability").iter_rows(named=True):
        assert f"| {row['benchmark_year']} | cumulative job change | inferred |" in text
