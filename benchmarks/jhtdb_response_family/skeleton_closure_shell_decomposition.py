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
from benchmarks.jhtdb_response_family.skeleton_closure_mechanism_audit import _score_mechanism
from benchmarks.jhtdb_response_family.skeleton_radial_thickening_profile import _stable_seed


CHANNELS = [
    "shape_stretch_total",
    "ancestor_spread_over_final",
    "D_rel",
    "B_growth_rel",
    "volume_growth_rel",
    "tube_shear_ratio",
    "vorticity_direction_rotation_mean_rad",
    "strain_frame_rotation_mean_rad",
    "alignment_loss",
    "cauchy_green_log10",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a sparse REAL-NS closure tube into core, annuli, and nested cumulative material shells."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        default=Path(
            "benchmarks/jhtdb_response_family/results/"
            "material_heat_age_audit_patch17_h14_top2_live_recentered"
        ),
    )
    parser.add_argument("--parent-core-id", default="0:0", help="Only used to center the support cube.")
    parser.add_argument("--dataset", default="isotropic1024coarse")
    parser.add_argument("--source-mode", choices=("synthetic", "soap", "pyjhtdb", "auto"), default="pyjhtdb")
    parser.add_argument("--patch-points-per-axis", type=int, default=49)
    parser.add_argument("--patch-spacing", type=float, default=2.0 * np.pi / 1024.0 * 4.0)
    parser.add_argument("--t0", type=float, default=0.2400000000000001)
    parser.add_argument("--t1", type=float, default=0.3200000000000002)
    parser.add_argument("--threshold-mode", default="lambda_abs2_omega_top2")
    parser.add_argument("--component-id", type=int, default=413)
    parser.add_argument("--shell-radii-dx", nargs="+", type=float, default=[0.0, 0.1, 0.25, 0.5])
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
            "skeleton_closure_shell_decomposition_top2_c413.csv"
        ),
    )
    args = parser.parse_args()

    radii = sorted(float(v) for v in args.shell_radii_dx)
    if not radii or abs(radii[0]) > 1e-12:
        raise SystemExit("--shell-radii-dx must start at 0")

    starts = pd.read_csv(args.run_dir / "material_heat_age_starts.csv")
    access = JHTDBAccessConfig(
        dataset=args.dataset,
        source_mode=args.source_mode,
        velocity_sinterp="Lag4",
        gradient_sinterp="FD4Lag4",
        tinterp="PCHIP",
        max_points_per_request=args.max_points_per_request,
    )
    client = make_flow_client(access)
    center = _source_center(starts, args.parent_core_id)
    grid_points, _center_index = patch_points(
        center,
        patch_points_per_axis=args.patch_points_per_axis,
        patch_spacing=args.patch_spacing,
        domain_length=access.domain_length,
    )
    snapshot = _query_patch(client, access, time=float(args.t1), points=grid_points, args=args)
    metrics = _point_metrics(snapshot.gradient)
    mask = _threshold_mask(str(args.threshold_mode), np.maximum(metrics["lambda"], 0.0), metrics["omega_norm"])
    components = _components(mask, (args.patch_points_per_axis,) * 3)
    if int(args.component_id) >= len(components):
        raise SystemExit(f"component {args.component_id} not found; only {len(components)} components")
    skeleton_points = grid_points[components[int(args.component_id)]]

    shells: list[tuple[float, float, np.ndarray]] = []
    for inner, outer in zip(radii[:-1], radii[1:]):
        shell_points = _annulus_points(
            skeleton_points,
            inner_radius=inner * float(args.patch_spacing),
            outer_radius=outer * float(args.patch_spacing),
            points_per_center=int(args.points_per_shell),
            seed=_stable_seed(args.rng_seed, "shell", args.component_id, inner, outer),
            domain_length=access.domain_length,
        )
        shells.append((inner, outer, shell_points))

    rows = []
    core_row = _score_set(args, access, client, skeleton_points, "core", 0.0, 0.0, 0.0, len(skeleton_points))
    rows.append(core_row)

    cumulative_parts = [skeleton_points]
    previous_cumulative = core_row
    for inner, outer, shell_points in shells:
        shell_row = _score_set(args, access, client, shell_points, "shell", inner, outer, outer, len(skeleton_points))
        shell_row["Q_drop_from_previous_cumulative"] = np.nan
        shell_row["delta_G_from_previous_cumulative"] = np.nan
        shell_row["delta_R_from_previous_cumulative"] = np.nan
        rows.append(shell_row)

        cumulative_parts.append(shell_points)
        cumulative_points = np.vstack(cumulative_parts)
        cumulative_row = _score_set(
            args,
            access,
            client,
            cumulative_points,
            "cumulative",
            0.0,
            outer,
            outer,
            len(skeleton_points),
        )
        _add_incremental_columns(cumulative_row, previous_cumulative)
        cumulative_row["added_shell_inner_dx"] = inner
        cumulative_row["added_shell_outer_dx"] = outer
        rows.append(cumulative_row)
        previous_cumulative = cumulative_row

    out = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    cols = [
        "set_type",
        "shell_inner_dx",
        "shell_outer_dx",
        "num_particles",
        "G",
        "R",
        "Q",
        "dominant_R",
        "Q_drop_from_previous_cumulative",
        "dominant_delta_channel",
        "shape_stretch_total",
        "tube_shear_ratio",
        "vorticity_direction_rotation_mean_rad",
        "strain_frame_rotation_mean_rad",
        "alignment_loss",
        "ancestor_spread_over_final",
        "D_rel",
        "B_growth_rel",
    ]
    print("skeleton_closure_shell_decomposition", flush=True)
    print(out[cols].to_string(index=False), flush=True)
    print(f"\noutput={args.output_csv}", flush=True)


def _score_set(
    args: argparse.Namespace,
    access: JHTDBAccessConfig,
    client,
    final_points: np.ndarray,
    set_type: str,
    shell_inner_dx: float,
    shell_outer_dx: float,
    label_radius_dx: float,
    skeleton_count: int,
) -> dict[str, float | int | str]:
    score_args = argparse.Namespace(**vars(args))
    # The role string inside _score_mechanism uses component id and radius.
    # Offset shell-only labels slightly to avoid human ambiguity; the cache key
    # still includes exact point coordinates.
    radius_label = float(label_radius_dx)
    if set_type == "shell":
        radius_label += 1000.0 + float(shell_inner_dx)
    row = _score_mechanism(score_args, access, client, final_points, radius_label, skeleton_count)
    row.update(
        {
            "set_type": set_type,
            "cache_radius_label": radius_label,
            "radius_dx": float(label_radius_dx),
            "radius_physical": float(label_radius_dx) * float(args.patch_spacing),
            "shell_inner_dx": float(shell_inner_dx),
            "shell_outer_dx": float(shell_outer_dx),
            "cumulative_outer_dx": float(shell_outer_dx) if set_type == "cumulative" else np.nan,
            "alignment_loss": max(-float(row.get("alignment_max_mean_delta", 0.0)), 0.0),
            "cauchy_green_log10": float(np.log10(max(float(row.get("cauchy_green_anisotropy", 0.0)), 1e-12))),
            "Q_drop_from_previous_cumulative": np.nan,
            "delta_G_from_previous_cumulative": np.nan,
            "delta_R_from_previous_cumulative": np.nan,
            "dominant_delta_channel": "",
            "dominant_delta_value": np.nan,
        }
    )
    return row


def _add_incremental_columns(row: dict[str, float | int | str], previous: dict[str, float | int | str]) -> None:
    row["Q_drop_from_previous_cumulative"] = float(previous["Q"]) - float(row["Q"])
    row["delta_G_from_previous_cumulative"] = float(row["G"]) - float(previous["G"])
    row["delta_R_from_previous_cumulative"] = float(row["R"]) - float(previous["R"])
    deltas = {}
    for channel in CHANNELS:
        deltas[channel] = float(row.get(channel, 0.0)) - float(previous.get(channel, 0.0))
        row[f"delta_{channel}"] = deltas[channel]
    label = max(deltas, key=deltas.get)
    row["dominant_delta_channel"] = label
    row["dominant_delta_value"] = deltas[label]


def _annulus_points(
    centers: np.ndarray,
    *,
    inner_radius: float,
    outer_radius: float,
    points_per_center: int,
    seed: int,
    domain_length: float,
) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if outer_radius <= inner_radius:
        return np.empty((0, 3), dtype=float)
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(max(1, int(points_per_center)), 3))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    inner3 = float(inner_radius) ** 3
    outer3 = float(outer_radius) ** 3
    radii = (inner3 + rng.random(len(directions)) * (outer3 - inner3)) ** (1.0 / 3.0)
    offsets = directions * radii.reshape(-1, 1)
    points = (centers[:, None, :] + offsets[None, :, :]) % domain_length
    return points.reshape((-1, 3))


if __name__ == "__main__":
    main()
