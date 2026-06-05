from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "scripts" / "run_near_degenerate_jhtdb_pipeline.py"
FINAL_CSV = ROOT / "data" / "results" / "near_degenerate_multi_tube_packing_attack_summary.csv"
DICTIONARY_JSON = ROOT / "data" / "results" / "near_degenerate_column_dictionary.json"
REPRODUCTION_MAP = ROOT / "data" / "near_degenerate_same_parent_reproduction_map.md"
PARENT_CACHE_RESULTS = ROOT.parent / "benchmarks" / "jhtdb_response_family" / "results"
PUBLISHED_INPUT_PACK = ROOT / "data" / "inputs" / "near_degenerate_published"


def run_pipeline(
    *args: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PIPELINE), *args],
        cwd=str(cwd or ROOT),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_dry_run_orchestrator_does_not_fail_and_writes_manifest(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    manifest = results_dir / "near_degenerate_pipeline_manifest.json"
    completed = run_pipeline(
        "--published-cases",
        "--dry-run",
        "--results-dir",
        str(results_dir),
        "--manifest",
        str(manifest),
    )

    assert completed.returncode == 0, completed.stderr
    assert "[stage:skeleton]" in completed.stdout
    assert "[stage:plots]" in completed.stdout
    assert manifest.exists()

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["dataset"] == "isotropic1024coarse"
    assert payload["published_mode"] is True
    assert len(payload["stages"]) >= 7
    assert payload["stages"][0]["status"] == "dry_run"


def test_published_live_dry_run_uses_bundled_inputs_and_public_testing_token(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    manifest = results_dir / "near_degenerate_pipeline_manifest.json"
    env = os.environ.copy()
    for key in [
        "JHTDB_TOKEN",
        "JHTDB_AUTH_TOKEN",
        "JHTDB_API_TOKEN",
        "JHTDB_USERNAME",
        "JHTDB_PASSWORD",
    ]:
        env.pop(key, None)

    completed = run_pipeline(
        "--published-cases",
        "--require-jhtdb",
        "--dry-run",
        "--results-dir",
        str(results_dir),
        "--manifest",
        str(manifest),
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    assert str(PUBLISHED_INPUT_PACK) in completed.stdout
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["live_jhtdb_config"] == "public_testing_token"
    assert payload["published_input_pack"] is not None
    assert payload["published_input_pack"]["used_default_pack"] is True


def test_column_dictionary_covers_every_final_csv_column() -> None:
    dictionary = json.loads(DICTIONARY_JSON.read_text(encoding="utf-8"))
    documented = {entry["column"] for entry in dictionary["columns"]}
    actual = set(pd.read_csv(FINAL_CSV, nrows=0).columns)

    assert documented == actual
    aliases = {entry["name"] for entry in dictionary.get("legacy_aliases", [])}
    assert "split_pair_path_proxy" in aliases


def test_reproduction_map_covers_required_sections_and_columns() -> None:
    text = REPRODUCTION_MAP.read_text(encoding="utf-8")

    required_sections = [
        "# Near-degenerate same-parent attack: reproduction map",
        "## 1. Run order",
        "## 2. Producer scripts",
        "## 3. Input artifacts",
        "## 4. Output artifacts",
        "## 5. Column dictionary",
        "## 6. Paper-symbol mapping",
        "## 7. Known limits",
    ]
    for heading in required_sections:
        assert heading in text

    lower = text.lower()
    assert "mechanism testing" in lower
    assert "does not prove the continuum inequalities" in lower

    column_section = text.split("## 6. Paper-symbol mapping", maxsplit=1)[0]
    for column in pd.read_csv(FINAL_CSV, nrows=0).columns:
        assert f"`{column}`" in column_section


def test_attack_summary_normalizer_restores_public_csv_contract(tmp_path: Path) -> None:
    module_name = "near_degenerate_public_pipeline_test"
    spec = importlib.util.spec_from_file_location(module_name, PIPELINE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)

    published_columns = list(pd.read_csv(FINAL_CSV, nrows=0).columns)
    summary_csv = tmp_path / FINAL_CSV.name
    extra_columns = ["attack_family", "sampled_phi_pairs"]
    row = {column: f"value_{index}" for index, column in enumerate(published_columns, start=1)}
    row.update({"attack_family": "same_parent_split", "sampled_phi_pairs": "42"})
    pd.DataFrame([row], columns=[*published_columns, *extra_columns]).to_csv(summary_csv, index=False)

    dropped = module._normalize_public_attack_summary(summary_csv, root=ROOT)
    normalized = pd.read_csv(summary_csv, dtype=str)

    assert list(normalized.columns) == published_columns
    assert dropped == extra_columns


def test_nonpublished_pipeline_fails_honestly_without_scan_inputs(tmp_path: Path) -> None:
    completed = run_pipeline(
        "--candidate",
        "lambda_abs2_omega_top2:ti14:c413:lag4",
        "--results-dir",
        str(tmp_path / "results"),
    )

    assert completed.returncode == 2
    assert "ERROR:" in completed.stderr
    assert "Provide --cache-dir or explicit --scan" in completed.stderr


@pytest.mark.skipif(not PARENT_CACHE_RESULTS.exists(), reason="parent benchmark cache/results not available")
def test_cached_pipeline_reruns_from_public_folder_and_validates(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    manifest = results_dir / "near_degenerate_pipeline_manifest.json"
    completed = run_pipeline(
        "--published-cases",
        "--use-existing-cache",
        "--cache-dir",
        str(PARENT_CACHE_RESULTS),
        "--results-dir",
        str(results_dir),
        "--manifest",
        str(manifest),
    )

    assert completed.returncode == 0, completed.stderr
    assert (results_dir / "near_degenerate_multi_tube_packing_attack_summary.csv").exists()
    assert (results_dir / "near_degenerate_attack_eta_vs_M.png").exists()
    assert (results_dir / "near_degenerate_attack_fphys_vs_M.png").exists()
    assert (results_dir / "near_degenerate_attack_results_figure.png").exists()
    assert "Verified near_degenerate_multi_tube_packing_attack_summary.csv" in completed.stdout
    assert manifest.exists()
    assert "WARNING:" not in completed.stdout

    final_csv = pd.read_csv(results_dir / "near_degenerate_multi_tube_packing_attack_summary.csv", nrows=0)
    bundled_csv = pd.read_csv(FINAL_CSV, nrows=0)
    assert list(final_csv.columns) == list(bundled_csv.columns)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    comparison = payload.get("published_comparison")
    assert comparison is not None
    assert comparison["matches_bundled_csv"] is True

    reused = run_pipeline(
        "--published-cases",
        "--use-existing-cache",
        "--cache-dir",
        str(PARENT_CACHE_RESULTS),
        "--results-dir",
        str(results_dir),
        "--manifest",
        str(manifest),
        "--stop-after",
        "membership",
    )
    assert reused.returncode == 0, reused.stderr
