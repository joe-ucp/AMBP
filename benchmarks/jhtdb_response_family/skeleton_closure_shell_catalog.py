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
from benchmarks.jhtdb_response_family.skeleton_closure_shell_decomposition import _annulus_points, _score_set
from benchmarks.jhtdb_response_family.skeleton_radial_thickening_profile import _stable_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Catalog the first-shell closure mechanism for high-Q sparse REAL-NS skeletons."
    )
    parser.add_argument(
        "--scan",
        nargs=3,
        action="append",
        metavar=("FAMILY", "RUN_DIR", "CANDIDATE_CSV"),
        required=True,
        help="Candidate source: family label, material heat-age run directory, global ancestry scan CSV.",
    )
    parser.add_argument("--parent-core-id", default="0:0", help="Only used to center each support cube.")
    parser.add_argument("--dataset", default="isotropic1024coarse")
    parser.add_argument("--source-mode", choices=("synthetic", "soap", "pyjhtdb", "auto"), default="pyjhtdb")
    parser.add_argument("--patch-points-per-axis", type=int, default=49)
    parser.add_argument("--patch-spacing", type=float, default=2.0 * np.pi / 1024.0 * 4.0)
    parser.add_argument("--min-q", type=float, default=0.8)
    parser.add_argument("--max-candidates-per-scan", type=int, default=6)
    parser.add_argument("--first-shell-radius-dx", type=float, default=0.1)
    parser.add_argument("--points-per-shell", type=int, default=16)
    parser.add_argument("--substep-dt", type=float, default=0.005)
    parser.add_argument("--max-points-per-request", type=int, default=2048)
    parser.add_argument("--eta", type=float, default=1e-12)
    parser.add_argument("--rng-seed", type=int, default=1729)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "benchmarks/jhtdb_response_family/results/"
            "skeleton_closure_shell_catalog_top2_top3.csv"
        ),
    )
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
    rows: list[dict[str, float | int | str | bool]] = []

    for family, run_dir_raw, candidate_csv_raw in args.scan:
        run_dir = Path(run_dir_raw)
        candidate_csv = Path(candidate_csv_raw)
        candidates = _selected_candidates(
            candidate_csv,
            min_q=float(args.min_q),
            max_n=int(args.max_candidates_per_scan),
        )
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
            rows.append(
                _candidate_row(
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

    out = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    print("skeleton_closure_shell_catalog", flush=True)
    if out.empty:
        print("no shell catalog rows", flush=True)
    else:
        cols = [
            "family",
            "candidate_id",
            "Q_core",
            "Q_first_shell",
            "Q_cumulative_first_shell",
            "delta_Q_core_to_cumulative",
            "delta_R_shape",
            "delta_R_spread",
            "delta_R_D",
            "delta_R_B",
            "dominant_first_shell_channel",
        ]
        print(out[cols].sort_values("Q_core", ascending=False).to_string(index=False), flush=True)
    print(f"\noutput={args.output_csv}", flush=True)


def _selected_candidates(candidate_csv: Path, *, min_q: float, max_n: int) -> pd.DataFrame:
    candidates = pd.read_csv(candidate_csv)
    selected = candidates[(candidates["Q"] >= min_q) | (candidates["Q"] > 1.0)].copy()
    if selected.empty:
        return selected
    return selected.sort_values(["Q", "G"], ascending=[False, False]).head(max_n).reset_index(drop=True)


def _candidate_row(
    *,
    family: str,
    candidate: pd.Series,
    args: argparse.Namespace,
    access: JHTDBAccessConfig,
    client,
    grid_points: np.ndarray,
    shape: tuple[int, int, int],
    snapshot_cache: dict[tuple[float, str], dict[str, np.ndarray]],
) -> dict[str, float | int | str | bool]:
    t1 = float(candidate["t1"])
    t0 = float(candidate["t0"])
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

    skeleton_points = grid_points[components[component_id]]
    shell_points = _annulus_points(
        skeleton_points,
        inner_radius=0.0,
        outer_radius=float(args.first_shell_radius_dx) * float(args.patch_spacing),
        points_per_center=int(args.points_per_shell),
        seed=_stable_seed(args.rng_seed, "shell_catalog", family, str(candidate["candidate_id"])),
        domain_length=access.domain_length,
    )
    cumulative_points = np.vstack([skeleton_points, shell_points])

    score_args = argparse.Namespace(**vars(args))
    score_args.t0 = t0
    score_args.t1 = t1
    score_args.threshold_mode = mode
    score_args.component_id = component_id

    core = _score_set(score_args, access, client, skeleton_points, "core", 0.0, 0.0, 0.0, len(skeleton_points))
    shell = _score_set(
        score_args,
        access,
        client,
        shell_points,
        "shell",
        0.0,
        float(args.first_shell_radius_dx),
        float(args.first_shell_radius_dx),
        len(skeleton_points),
    )
    cumulative = _score_set(
        score_args,
        access,
        client,
        cumulative_points,
        "cumulative",
        0.0,
        float(args.first_shell_radius_dx),
        float(args.first_shell_radius_dx),
        len(skeleton_points),
    )

    deltas = {
        "shape": float(cumulative["shape_stretch_total"]) - float(core["shape_stretch_total"]),
        "spread": float(cumulative["ancestor_spread_over_final"]) - float(core["ancestor_spread_over_final"]),
        "D": float(cumulative["D_rel"]) - float(core["D_rel"]),
        "B": float(cumulative["B_growth_rel"]) - float(core["B_growth_rel"]),
        "volume": float(cumulative["volume_growth_rel"]) - float(core["volume_growth_rel"]),
        "shear": float(cumulative["tube_shear_ratio"]) - float(core["tube_shear_ratio"]),
        "vorticity_rotation": float(cumulative["vorticity_direction_rotation_mean_rad"]) - float(core["vorticity_direction_rotation_mean_rad"]),
        "strain_frame_rotation": float(cumulative["strain_frame_rotation_mean_rad"]) - float(core["strain_frame_rotation_mean_rad"]),
        "alignment_loss": float(cumulative["alignment_loss"]) - float(core["alignment_loss"]),
    }
    dominant_delta = max(deltas, key=deltas.get)

    return {
        "family": family,
        "candidate_id": str(candidate["candidate_id"]),
        "t0": t0,
        "t1": t1,
        "threshold_mode": mode,
        "component_id": component_id,
        "component_size": int(len(skeleton_points)),
        "particle_count_core": int(core["num_particles"]),
        "particle_count_first_shell": int(shell["num_particles"]),
        "particle_count_cumulative_first_shell": int(cumulative["num_particles"]),
        "scan_Q": float(candidate["Q"]),
        "scan_G": float(candidate["G"]),
        "scan_R": float(candidate["R"]),
        "Q_core": float(core["Q"]),
        "Q_first_shell": float(shell["Q"]),
        "Q_cumulative_first_shell": float(cumulative["Q"]),
        "G_core": float(core["G"]),
        "G_first_shell": float(shell["G"]),
        "G_cumulative_first_shell": float(cumulative["G"]),
        "R_core": float(core["R"]),
        "R_first_shell": float(shell["R"]),
        "R_cumulative_first_shell": float(cumulative["R"]),
        "delta_Q_core_to_shell": float(core["Q"]) - float(shell["Q"]),
        "delta_Q_core_to_cumulative": float(core["Q"]) - float(cumulative["Q"]),
        "delta_G": float(cumulative["G"]) - float(core["G"]),
        "delta_R": float(cumulative["R"]) - float(core["R"]),
        "delta_R_shape": deltas["shape"],
        "delta_R_spread": deltas["spread"],
        "delta_R_D": deltas["D"],
        "delta_R_B": deltas["B"],
        "delta_R_volume": deltas["volume"],
        "delta_tube_shear": deltas["shear"],
        "delta_vorticity_rotation": deltas["vorticity_rotation"],
        "delta_strain_frame_rotation": deltas["strain_frame_rotation"],
        "delta_alignment_loss": deltas["alignment_loss"],
        "dominant_first_shell_channel": dominant_delta,
        "dominant_first_shell_value": deltas[dominant_delta],
        "core_dominant_R": str(core["dominant_R"]),
        "shell_dominant_R": str(shell["dominant_R"]),
        "cumulative_dominant_R": str(cumulative["dominant_R"]),
        "first_shell_reduces_Q": bool(float(cumulative["Q"]) < float(core["Q"])),
        "first_shell_closes_Q_gt_1": bool(float(core["Q"]) > 1.0 and float(cumulative["Q"]) < 1.0),
        "shape_dominated_delta": bool(dominant_delta == "shape"),
    }


if __name__ == "__main__":
    main()
