"""Targeted Source Growth Dichotomy audit for high-Q material tube candidates.

This script is intentionally offline: it consumes the existing sparse-skeleton,
radial-thickening, and first-shell closure CSVs and classifies each candidate
against the current finite-thickness closure ledger.

The output is not a proof.  It answers the empirical question:

    Does source growth charge a measured shape/deformation channel, a measured
    non-shape channel, or an unmeasured tail/residual channel?

The current available shape quantity is the particle-cloud shape ledger
(`shape_stretch_penalty`) plus first-shell shear where available.  The exact
continuum Cauchy--Green quantity remains a proof-facing target, so this script
labels it as a proxy.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_RESULTS = Path("benchmarks/jhtdb_response_family/results")


def _num(row: pd.Series | dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _pos(value: float) -> float:
    return max(0.0, float(value))


def _safe_div(num: float, den: float) -> float:
    if abs(den) < 1e-15:
        return math.inf if num > 0 else 0.0
    return num / den


def _dominant_branch(values: dict[str, float], pass_c1: bool) -> str:
    if not pass_c1:
        return "tail_or_missing_ledger"
    return max(values.items(), key=lambda item: item[1])[0]


def _audit_row(
    *,
    family: str,
    candidate_id: str,
    stage: str,
    source: str,
    t0: float,
    t1: float,
    radius_dx: float,
    num_particles: int,
    G: float,
    Q: float,
    R_reported: float,
    R_shape_cg_proxy: float,
    R_shear: float = 0.0,
    R_xi: float = 0.0,
    R_frame: float = 0.0,
    R_D: float = 0.0,
    R_B: float = 0.0,
    R_spread: float = 0.0,
    R_tail_proxy: float = 0.0,
    E_nu: float = 0.0,
    alpha: float = np.nan,
) -> dict[str, Any]:
    R_shape_branch = _pos(R_shape_cg_proxy) + _pos(R_shear)
    R_alignment_frame = _pos(R_xi) + _pos(R_frame)
    R_deactivation = _pos(R_D)
    R_negative = _pos(R_B)
    R_spread_branch = _pos(R_spread)
    R_tail_branch = _pos(R_tail_proxy)
    E_branch = _pos(E_nu)

    ledger_total = (
        R_shape_branch
        + R_alignment_frame
        + R_deactivation
        + R_negative
        + R_spread_branch
        + R_tail_branch
        + E_branch
    )
    delta_dich = ledger_total - G
    pass_c1 = delta_dich >= 0.0
    branch_values = {
        "shape_deformation": R_shape_branch,
        "alignment_frame": R_alignment_frame,
        "deactivation": R_deactivation,
        "negative_stretch": R_negative,
        "spread": R_spread_branch,
        "tail_residual": R_tail_branch + E_branch,
    }
    dominant = _dominant_branch(branch_values, pass_c1)

    return {
        "family": family,
        "candidate_id": candidate_id,
        "stage": stage,
        "source": source,
        "t0": t0,
        "t1": t1,
        "radius_dx": radius_dx,
        "num_particles": num_particles,
        "G": G,
        "Q_original": Q,
        "R_reported": R_reported,
        "alpha": alpha,
        "R_shape_CG_proxy": _pos(R_shape_cg_proxy),
        "R_shear": _pos(R_shear),
        "R_xi": _pos(R_xi),
        "R_frame": _pos(R_frame),
        "R_D": _pos(R_D),
        "R_B": _pos(R_B),
        "R_spread": _pos(R_spread),
        "R_tail_proxy": _pos(R_tail_proxy),
        "E_nu": _pos(E_nu),
        "R_shape_branch": R_shape_branch,
        "R_alignment_frame_branch": R_alignment_frame,
        "R_deactivation_branch": R_deactivation,
        "R_negative_stretch_branch": R_negative,
        "R_spread_branch": R_spread_branch,
        "R_tail_residual_branch": R_tail_branch + E_branch,
        "ledger_total_C1": ledger_total,
        "delta_dich_C1": delta_dich,
        "C_min": _safe_div(max(0.0, G - _pos(E_nu)), ledger_total),
        "dichotomy_pass_C1": pass_c1,
        "dominant_branch": dominant,
        "dangerous_tail_or_missing": dominant == "tail_or_missing_ledger",
    }


def _row_from_catalog(row: pd.Series, stage: str) -> dict[str, Any]:
    R_spread = max(
        _num(row, "ancestor_spread_over_final"),
        _num(row, "volume_growth_rel"),
    )
    return _audit_row(
        family=str(row["family"]),
        candidate_id=str(row["candidate_id"]),
        stage=stage,
        source="skeleton_closure_catalog",
        t0=_num(row, "t0"),
        t1=_num(row, "t1"),
        radius_dx=_num(row, "tube_radius_dx"),
        num_particles=int(_num(row, "num_particles")),
        G=_num(row, "G"),
        Q=_num(row, "Q"),
        R_reported=_num(row, "R"),
        R_shape_cg_proxy=_num(row, "shape_stretch_penalty"),
        R_D=_num(row, "D_rel"),
        R_B=_num(row, "B_growth_rel"),
        R_spread=R_spread,
    )


def _shell_row(shell: pd.Series, core: pd.Series) -> dict[str, Any]:
    shape = _num(core, "shape_stretch_penalty") + _num(shell, "delta_R_shape")
    D = _num(core, "D_rel") + _num(shell, "delta_R_D")
    B = _num(core, "B_growth_rel") + _num(shell, "delta_R_B")
    spread = max(
        _num(core, "ancestor_spread_over_final") + _num(shell, "delta_R_spread"),
        _num(core, "volume_growth_rel") + _num(shell, "delta_R_volume"),
    )
    xi = max(
        _num(shell, "delta_vorticity_rotation"),
        _num(shell, "delta_alignment_loss"),
    )
    return _audit_row(
        family=str(shell["family"]),
        candidate_id=str(shell["candidate_id"]),
        stage="first_shell_cumulative_0p1dx",
        source="skeleton_closure_shell_catalog",
        t0=_num(shell, "t0"),
        t1=_num(shell, "t1"),
        radius_dx=0.1,
        num_particles=int(_num(shell, "particle_count_cumulative_first_shell")),
        G=_num(shell, "G_cumulative_first_shell"),
        Q=_num(shell, "Q_cumulative_first_shell"),
        R_reported=_num(shell, "R_cumulative_first_shell"),
        R_shape_cg_proxy=shape,
        R_shear=_num(shell, "delta_tube_shear"),
        R_xi=xi,
        R_frame=_num(shell, "delta_strain_frame_rotation"),
        R_D=D,
        R_B=B,
        R_spread=spread,
    )


def build_audit_rows(details: pd.DataFrame, shell: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    core_by_key: dict[tuple[str, str], pd.Series] = {}

    for (family, candidate_id), part in details.groupby(["family", "candidate_id"], sort=False):
        part = part.sort_values("tube_radius_dx")
        core = part.iloc[0]
        core_by_key[(str(family), str(candidate_id))] = core
        rows.append(_row_from_catalog(core, "sparse_skeleton"))

        finite = part[part["tube_radius_dx"].astype(float) > 0.0]
        if not finite.empty:
            first_finite = finite.iloc[0]
            rows.append(_row_from_catalog(first_finite, "first_catalog_finite"))
            max_q_finite = finite.sort_values("Q", ascending=False).iloc[0]
            if float(max_q_finite["tube_radius_dx"]) != float(first_finite["tube_radius_dx"]):
                rows.append(_row_from_catalog(max_q_finite, "max_Q_catalog_finite"))

    for _, shell_row in shell.iterrows():
        key = (str(shell_row["family"]), str(shell_row["candidate_id"]))
        core = core_by_key.get(key)
        if core is not None:
            rows.append(_shell_row(shell_row, core))

    return pd.DataFrame(rows)


def build_summary(rows: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for (family, candidate_id), part in rows.groupby(["family", "candidate_id"], sort=False):
        sparse = part[part["stage"] == "sparse_skeleton"].iloc[0]
        shell = part[part["stage"] == "first_shell_cumulative_0p1dx"]
        finite = part[part["stage"] == "first_catalog_finite"]

        if float(sparse["Q_original"]) > 1.0 and not shell.empty:
            row = shell.iloc[0].copy()
            selection_reason = "sparse_Q_gt_1_closed_by_first_shell"
        elif float(sparse["Q_original"]) > 1.0 and not finite.empty:
            row = finite.iloc[0].copy()
            selection_reason = "sparse_Q_gt_1_closed_by_first_catalog_finite"
        else:
            row = sparse.copy()
            selection_reason = "sparse_ledger_already_Q_le_1"

        row["selection_reason"] = selection_reason
        row["Q_sparse"] = float(sparse["Q_original"])
        row["G_sparse"] = float(sparse["G"])
        row["C_min_sparse"] = float(sparse["C_min"])
        selected.append(row)

    return pd.DataFrame(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the targeted Source Growth Dichotomy audit.")
    parser.add_argument(
        "--details-csv",
        type=Path,
        default=DEFAULT_RESULTS / "skeleton_closure_catalog_top2_top3_details.csv",
    )
    parser.add_argument(
        "--shell-csv",
        type=Path,
        default=DEFAULT_RESULTS / "skeleton_closure_shell_catalog_top2_top3.csv",
    )
    parser.add_argument(
        "--output-rows-csv",
        type=Path,
        default=DEFAULT_RESULTS / "targeted_dichotomy_audit_top2_top3_rows.csv",
    )
    parser.add_argument(
        "--output-summary-csv",
        type=Path,
        default=DEFAULT_RESULTS / "targeted_dichotomy_audit_top2_top3_summary.csv",
    )
    args = parser.parse_args()

    details = pd.read_csv(args.details_csv)
    shell = pd.read_csv(args.shell_csv)
    rows = build_audit_rows(details, shell)
    summary = build_summary(rows)

    args.output_rows_csv.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output_rows_csv, index=False)
    summary.to_csv(args.output_summary_csv, index=False)

    cols = [
        "family",
        "candidate_id",
        "stage",
        "G",
        "Q_original",
        "ledger_total_C1",
        "delta_dich_C1",
        "C_min",
        "dominant_branch",
        "dangerous_tail_or_missing",
    ]
    print("Selected dichotomy rows:")
    print(summary[cols].sort_values("C_min", ascending=False).to_string(index=False))
    print()
    print("Branch counts:")
    print(summary["dominant_branch"].value_counts().to_string())
    print()
    print(f"rows_csv={args.output_rows_csv}")
    print(f"summary_csv={args.output_summary_csv}")


if __name__ == "__main__":
    main()
