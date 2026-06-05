from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.jhtdb_response_family.config import JHTDBAccessConfig
from benchmarks.jhtdb_response_family.global_hotspot_ancestry_scan import _components, _threshold_mask
from benchmarks.jhtdb_response_family.jhtdb_client import make_flow_client
from benchmarks.jhtdb_response_family.material_image_runaway_audit import _query_patch, _source_center
from benchmarks.jhtdb_response_family.relaxed_patch_stretch_budget_audit import _point_metrics
from benchmarks.jhtdb_response_family.sampling import patch_points
from benchmarks.jhtdb_response_family.skeleton_radial_thickening_profile import (
    _score_tube,
    _stable_seed,
    _tube_particles,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Catalog finite-radius closure of sparse REAL-NS hotspot skeleton candidates."
    )
    parser.add_argument(
        "--scan",
        nargs=3,
        action="append",
        metavar=("FAMILY", "RUN_DIR", "CANDIDATE_CSV"),
        required=True,
        help="Candidate source: family label, material heat-age run directory, global ancestry scan CSV.",
    )
    parser.add_argument("--windows-csv", type=Path, required=True)
    parser.add_argument("--parent-core-id", default="0:0", help="Only used to center each support cube.")
    parser.add_argument("--dataset", default="isotropic1024coarse")
    parser.add_argument("--source-mode", choices=("synthetic", "soap", "pyjhtdb", "auto"), default="pyjhtdb")
    parser.add_argument("--patch-points-per-axis", type=int, default=49)
    parser.add_argument("--patch-spacing", type=float, default=2.0 * np.pi / 1024.0 * 4.0)
    parser.add_argument("--min-q", type=float, default=0.8)
    parser.add_argument("--max-candidates-per-scan", type=int, default=6)
    parser.add_argument("--tube-radii-dx", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--points-per-center", type=int, default=16)
    parser.add_argument("--substep-dt", type=float, default=0.005)
    parser.add_argument("--max-points-per-request", type=int, default=2048)
    parser.add_argument("--eta", type=float, default=1e-12)
    parser.add_argument("--rng-seed", type=int, default=1729)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    args = parser.parse_args()

    access = JHTDBAccessConfig(
        dataset=args.dataset,
        source_mode=args.source_mode,
        velocity_sinterp="Lag4",
        gradient_sinterp="FD4Lag4",
        tinterp="PCHIP",
        max_points_per_request=args.max_points_per_request,
    )
    client = make_flow_client(access)
    detail_rows: list[dict[str, float | int | str | bool]] = []

    for family, run_dir_raw, candidate_csv_raw in args.scan:
        run_dir = Path(run_dir_raw)
        candidate_csv = Path(candidate_csv_raw)
        candidates = _selected_candidates(candidate_csv, min_q=float(args.min_q), max_n=int(args.max_candidates_per_scan))
        if candidates.empty:
            continue
        starts = pd.read_csv(run_dir / "material_heat_age_starts.csv")
        center = _source_center(starts, args.parent_core_id)
        grid_points, _center_index = patch_points(
            center,
            patch_points_per_axis=args.patch_points_per_axis,
            patch_spacing=args.patch_spacing,
            domain_length=access.domain_length,
        )
        shape = (args.patch_points_per_axis,) * 3
        snapshot_cache: dict[tuple[float, str], dict[str, np.ndarray]] = {}
        for _, candidate in candidates.iterrows():
            detail_rows.extend(
                _profile_candidate(
                    family=family,
                    candidate=candidate,
                    args=args,
                    access=access,
                    client=client,
                    grid_points=grid_points,
                    shape=shape,
                    snapshot_cache=snapshot_cache,
                )
            )

    details = pd.DataFrame(detail_rows)
    summary = _summary(details)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.output_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)

    print("skeleton_closure_catalog", flush=True)
    if summary.empty:
        print("no catalog rows", flush=True)
    else:
        cols = [
            "family",
            "candidate_id",
            "Q_sparse_scan",
            "r_close_dx",
            "Q_0dx",
            "Q_0p25dx",
            "Q_0p5dx",
            "Q_1dx",
            "Q_2dx",
            "dominant_R_at_closure",
            "dominant_R_at_2dx",
        ]
        print(summary[cols].sort_values("Q_sparse_scan", ascending=False).to_string(index=False), flush=True)
    print(f"\ndetails={args.output_csv}", flush=True)
    print(f"summary={args.summary_csv}", flush=True)


def _selected_candidates(candidate_csv: Path, *, min_q: float, max_n: int) -> pd.DataFrame:
    candidates = pd.read_csv(candidate_csv)
    selected = candidates[(candidates["Q"] >= min_q) | (candidates["Q"] > 1.0)].copy()
    if selected.empty:
        return selected
    selected = selected.sort_values(["Q", "G"], ascending=[False, False]).head(max_n).reset_index(drop=True)
    return selected


def _profile_candidate(
    *,
    family: str,
    candidate: pd.Series,
    args: argparse.Namespace,
    access: JHTDBAccessConfig,
    client,
    grid_points: np.ndarray,
    shape: tuple[int, int, int],
    snapshot_cache: dict[tuple[float, str], dict[str, np.ndarray]],
) -> list[dict[str, float | int | str | bool]]:
    t1 = float(candidate["t1"])
    mode = str(candidate["threshold_mode"])
    component_id = int(candidate["component_id"])
    key = (t1, mode)
    if key not in snapshot_cache:
        data = _query_patch(client, access, time=t1, points=grid_points, args=args)
        snapshot_cache[key] = _point_metrics(data.gradient)
    metrics = snapshot_cache[key]
    mask = _threshold_mask(mode, np.maximum(metrics["lambda"], 0.0), metrics["omega_norm"])
    components = _components(mask, shape)
    if component_id >= len(components):
        raise SystemExit(f"{family}: component {component_id} missing at t1={t1} mode={mode}; only {len(components)} components")
    skeleton_indices = components[component_id]
    skeleton_points = grid_points[skeleton_indices]
    rows = []
    for radius_dx in [float(v) for v in args.tube_radii_dx]:
        final_points = _tube_particles(
            skeleton_points,
            radius=radius_dx * float(args.patch_spacing),
            points_per_center=int(args.points_per_center),
            seed=_stable_seed(args.rng_seed, family, str(candidate["candidate_id"]), radius_dx),
            domain_length=access.domain_length,
        )
        score_args = argparse.Namespace(**vars(args))
        score_args.t0 = float(candidate["t0"])
        score_args.t1 = t1
        score_args.threshold_mode = mode
        score_args.component_id = component_id
        row = _score_tube(score_args, access, client, final_points, radius_dx, len(skeleton_points))
        row.update(
            {
                "family": family,
                "candidate_id": str(candidate["candidate_id"]),
                "scan_Q": float(candidate["Q"]),
                "scan_G": float(candidate["G"]),
                "scan_R": float(candidate["R"]),
                "scan_dominant_R": str(candidate["dominant_R"]),
                "scan_num_particles": int(candidate["num_particles"]),
            }
        )
        rows.append(row)
    return rows


def _summary(details: pd.DataFrame) -> pd.DataFrame:
    if details.empty:
        return pd.DataFrame()
    rows = []
    for (family, candidate_id), part in details.groupby(["family", "candidate_id"], sort=False):
        part = part.sort_values("tube_radius_dx")
        sparse = part.iloc[0]
        closed = part[part["Q"] < 1.0]
        closure = closed.iloc[0] if not closed.empty else None
        row = {
            "family": family,
            "candidate_id": candidate_id,
            "t0": float(sparse["t0"]),
            "t1": float(sparse["t1"]),
            "threshold_mode": sparse["threshold_mode"],
            "component_id": int(sparse["component_id"]),
            "component_size": int(sparse["skeleton_particles"]),
            "particle_count_sparse": int(sparse["scan_num_particles"]),
            "Q_sparse_scan": float(sparse["scan_Q"]),
            "G_sparse_scan": float(sparse["scan_G"]),
            "R_sparse_scan": float(sparse["scan_R"]),
            "dominant_R_sparse_scan": sparse["scan_dominant_R"],
            "Q_0dx": _q_at(part, 0.0),
            "Q_0p25dx": _q_at(part, 0.25),
            "Q_0p5dx": _q_at(part, 0.5),
            "Q_1dx": _q_at(part, 1.0),
            "Q_1p5dx": _q_at(part, 1.5),
            "Q_2dx": _q_at(part, 2.0),
            "dominant_R_at_2dx": _dominant_at(part, 2.0),
            "max_Q_after_zero": float(part[part["tube_radius_dx"] > 0.0]["Q"].max()) if (part["tube_radius_dx"] > 0.0).any() else np.nan,
            "min_Q_after_zero": float(part[part["tube_radius_dx"] > 0.0]["Q"].min()) if (part["tube_radius_dx"] > 0.0).any() else np.nan,
        }
        if closure is None:
            row.update({"r_close_dx": np.nan, "Q_at_close": np.nan, "dominant_R_at_closure": "unclosed"})
        else:
            row.update(
                {
                    "r_close_dx": float(closure["tube_radius_dx"]),
                    "Q_at_close": float(closure["Q"]),
                    "dominant_R_at_closure": str(closure["dominant_R"]),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _q_at(part: pd.DataFrame, radius: float) -> float:
    match = part[np.isclose(part["tube_radius_dx"], radius)]
    if match.empty:
        return np.nan
    return float(match.iloc[0]["Q"])


def _dominant_at(part: pd.DataFrame, radius: float) -> str:
    match = part[np.isclose(part["tube_radius_dx"], radius)]
    if match.empty:
        return ""
    return str(match.iloc[0]["dominant_R"])


if __name__ == "__main__":
    main()
