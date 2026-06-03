from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT / "scripts"))

import amplification_lab as lab
import build_ledger_lab


def test_synthetic_mode_runs_without_cached_data() -> None:
    summary = lab.run_all("synthetic")
    assert summary["same_parent"]["all_split_rows_merge_back"] is True
    assert summary["same_parent"]["eta_min"] == 1.0
    assert summary["tail_ladder"]["best_denominator_counts"] == "D5=5"
    assert summary["arr"]["dominant_deficit_source"] == "renewal_exposure"
    assert len(summary["conclusion"]) == 5
    assert "how_to_break_it" in summary["conclusion"][0]


def test_cached_mode_is_available_by_default() -> None:
    assert lab.data_available() is True
    summary = lab.run_all("cached")
    assert summary["same_parent"]["all_split_rows_merge_back"] is True
    assert summary["arr"]["dominant_deficit_source"] == "renewal_exposure"


def test_cached_ledger_worksheets_expose_decisive_columns() -> None:
    same = lab.same_parent_ledger("cached")
    assert {"created_labels", "collapse_fraction", "free_amplification_survives"}.issubset(same.columns)
    assert not same["free_amplification_survives"].any()

    renewal = lab.renewal_ledger("cached")
    assert {"pays_physical_charge", "passes_proxy_gap", "free_jitter_candidate"}.issubset(renewal.columns)
    assert not renewal["free_jitter_candidate"].any()

    tail = lab.tail_ladder_ledger("cached")
    assert {"best_ratio_recomputed", "unresolved_family_count", "is_paid_below_one"}.issubset(tail.columns)
    assert (tail["best_ratio"] - tail["best_ratio_recomputed"]).abs().max() < 1e-9

    arr = lab.arr_ledger("cached")
    assert float(arr.iloc[-1]["ratio"]) < 1.0

    coherent = lab.coherent_ledger("cached")
    assert {"net_fraction_of_positive", "door4_status"}.issubset(coherent.columns)


def test_artifact_table_names_expected_public_data_paths() -> None:
    table = lab.artifact_table()
    assert not table.empty
    assert table["path"].str.startswith(("data/results/", "outputs/", "paper/", "notebooks/")).all()
    assert "data/results/tail_denominator_ladder_audit_top2_top3_rows.csv" in set(table["path"])


def test_write_summary_json_and_synthetic_results(tmp_path) -> None:
    out = lab.write_summary_json("synthetic", tmp_path / "summary.json")
    assert out.exists()
    synthetic_dir = lab.write_synthetic_results(tmp_path / "synthetic_results")
    assert (synthetic_dir / "near_degenerate_multi_tube_packing_attack_summary.csv").exists()
    assert (synthetic_dir / "arr_deficit_attribution_audit_c185_final81_summary.csv").exists()


def test_public_plots_construct_without_cached_data() -> None:
    for plotter in (
        lab.plot_same_parent,
        lab.plot_renewal,
        lab.plot_tail_arr,
        lab.plot_routing_summary,
    ):
        fig = plotter("synthetic")
        assert fig.axes


def test_ledger_lab_builds_from_real_cached_artifacts(tmp_path, monkeypatch) -> None:
    out_file = tmp_path / "index.html"
    monkeypatch.setattr(build_ledger_lab, "OUT_DIR", tmp_path)
    monkeypatch.setattr(build_ledger_lab, "OUT_FILE", out_file)

    assert build_ledger_lab.main() == 0
    html = out_file.read_text(encoding="utf-8")

    assert "near_degenerate_multi_tube_packing_attack_summary.csv" in html
    assert "renewal_cascade_shape_topsource_phi_env_act_joint_hammer_r0p1_summary.csv" in html
    assert "arr_deficit_attribution_audit_c185_final81_summary.csv" in html
    assert '"natural": 28' in html
    assert '"labeled": 224' in html
    assert "No synthetic rows are used by this visual." in html
