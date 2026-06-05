from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.jhtdb_response_family.cache import cached_query_points
from benchmarks.jhtdb_response_family.config import JHTDBAccessConfig
from benchmarks.jhtdb_response_family.global_hotspot_ancestry_scan import (
    _components,
    _shape_stretch_penalty,
    _threshold_mask,
    _time_grid,
    _unwrap,
)
from benchmarks.jhtdb_response_family.jhtdb_client import make_flow_client
from benchmarks.jhtdb_response_family.material_image_runaway_audit import (
    _advect_cloud,
    _occupied_voxel_volume,
    _periodic_norm,
    _query_patch,
    _rel,
    _source_center,
)
from benchmarks.jhtdb_response_family.relaxed_patch_stretch_budget_audit import _point_metrics
from benchmarks.jhtdb_response_family.sampling import patch_points


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Radial material thickening profile around a sparse hotspot skeleton."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--windows-csv", type=Path, required=True)
    parser.add_argument("--parent-core-id", default="0:0", help="Only used to center the support cube.")
    parser.add_argument("--dataset", default="isotropic1024coarse")
    parser.add_argument("--source-mode", choices=("synthetic", "soap", "pyjhtdb", "auto"), default="pyjhtdb")
    parser.add_argument("--patch-points-per-axis", type=int, default=49)
    parser.add_argument("--patch-spacing", type=float, default=2.0 * np.pi / 1024.0 * 4.0)
    parser.add_argument("--t0", type=float, required=True)
    parser.add_argument("--t1", type=float, required=True)
    parser.add_argument("--threshold-mode", default="lambda_abs2_omega_top2")
    parser.add_argument("--component-id", type=int, required=True)
    parser.add_argument("--tube-radii-dx", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--points-per-center", type=int, default=16)
    parser.add_argument("--substep-dt", type=float, default=0.005)
    parser.add_argument("--max-points-per-request", type=int, default=2048)
    parser.add_argument("--eta", type=float, default=1e-12)
    parser.add_argument("--rng-seed", type=int, default=1729)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

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
    shape = (args.patch_points_per_axis,) * 3

    # Loading the time grid here is intentional: it catches mismatched windows
    # files before an expensive particle integration starts.
    available = set(np.round(_time_grid(args.windows_csv), 12))
    for time in (args.t0, args.t1):
        if np.round(float(time), 12) not in available:
            raise SystemExit(f"time {time} not present in {args.windows_csv}")

    final_snapshot = _query_patch(client, access, time=float(args.t1), points=grid_points, args=args)
    final_patch_metrics = _point_metrics(final_snapshot.gradient)
    mask = _threshold_mask(
        str(args.threshold_mode),
        np.maximum(final_patch_metrics["lambda"], 0.0),
        final_patch_metrics["omega_norm"],
    )
    components = _components(mask, shape)
    if int(args.component_id) >= len(components):
        raise SystemExit(f"component {args.component_id} not present; only {len(components)} components")

    skeleton_indices = components[int(args.component_id)]
    skeleton_points = grid_points[skeleton_indices]
    rows = []
    for radius_dx in [float(v) for v in args.tube_radii_dx]:
        final_points = _tube_particles(
            skeleton_points,
            radius=float(radius_dx) * float(args.patch_spacing),
            points_per_center=int(args.points_per_center),
            seed=_stable_seed(args.rng_seed, args.component_id, radius_dx),
            domain_length=access.domain_length,
        )
        rows.append(_score_tube(args, access, client, final_points, radius_dx, len(skeleton_points)))

    out = pd.DataFrame(rows)
    output_csv = args.output_csv or args.run_dir / "skeleton_radial_thickening_profile.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    print("skeleton_radial_thickening_profile", flush=True)
    print(
        out[
            [
                "tube_radius_dx",
                "num_particles",
                "G",
                "R",
                "Q",
                "dominant_R",
                "omega_growth_rel",
                "Splus_growth_rel",
                "B_growth_rel",
                "D_rel",
                "volume_growth_rel",
                "ancestor_spread_over_final",
                "shape_stretch_penalty",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(f"\noutput={output_csv}", flush=True)


def _score_tube(
    args: argparse.Namespace,
    access: JHTDBAccessConfig,
    client,
    final_points: np.ndarray,
    radius_dx: float,
    skeleton_count: int,
) -> dict[str, float | int | str | bool]:
    final_data = cached_query_points(
        client,
        access,
        time=float(args.t1),
        points=final_points,
        role=(
            "skeleton_thickening:"
            f"t0={args.t0:.12g}:t1={args.t1:.12g}:mode={args.threshold_mode}:"
            f"component={args.component_id}:r={radius_dx:g}:final"
        ),
    )
    final_metrics = _point_metrics(final_data.gradient)
    ancestor_points = _advect_cloud(
        final_points,
        t0=float(args.t1),
        t1=float(args.t0),
        args=args,
        access=access,
        client=client,
        window_id=f"skeleton_thickening:c{args.component_id}:r={radius_dx:g}:backward",
    )
    finite = np.isfinite(ancestor_points).all(axis=1)
    final_points = final_points[finite]
    ancestor_points = ancestor_points[finite] % access.domain_length
    final_metrics = {key: np.asarray(value)[finite] for key, value in final_metrics.items()}
    ancestor_data = cached_query_points(
        client,
        access,
        time=float(args.t0),
        points=ancestor_points,
        role=(
            "skeleton_thickening:"
            f"t0={args.t0:.12g}:t1={args.t1:.12g}:mode={args.threshold_mode}:"
            f"component={args.component_id}:r={radius_dx:g}:ancestor"
        ),
    )
    ancestor_metrics = _point_metrics(ancestor_data.gradient)
    returned_points = _advect_cloud(
        ancestor_points,
        t0=float(args.t0),
        t1=float(args.t1),
        args=args,
        access=access,
        client=client,
        window_id=f"skeleton_thickening:c{args.component_id}:r={radius_dx:g}:return",
    )
    return_error = _periodic_norm(returned_points - final_points, access.domain_length)
    final_diameter = max(_max_pairwise(final_points, access.domain_length), 1e-12)

    voxel_volume = float(args.patch_spacing) ** 3
    lam0 = np.asarray(ancestor_metrics["lambda"], dtype=float)
    lam1 = np.asarray(final_metrics["lambda"], dtype=float)
    pos0 = np.maximum(lam0, 0.0)
    pos1 = np.maximum(lam1, 0.0)
    neg0 = np.maximum(-lam0, 0.0)
    neg1 = np.maximum(-lam1, 0.0)
    s0 = float(np.sum(pos0) * voxel_volume)
    s1 = float(np.sum(pos1) * voxel_volume)
    b0 = float(np.sum(neg0) * voxel_volume)
    b1 = float(np.sum(neg1) * voxel_volume)
    omega0 = float(np.max(ancestor_metrics["omega_norm"]))
    omega1 = float(np.max(final_metrics["omega_norm"]))

    omega_growth = _rel(omega1 - omega0, omega0)
    s_growth = _rel(s1 - s0, s0)
    b_growth = _rel(max(b1 - b0, 0.0), s0)
    d_rel = float(np.sum(np.maximum(pos0 - pos1, 0.0)) * voxel_volume) / max(s0, 1e-12)
    ancestor_volume = _occupied_voxel_volume(ancestor_points, args.patch_spacing, access.domain_length)
    final_volume = _occupied_voxel_volume(final_points, args.patch_spacing, access.domain_length)
    volume_growth = _rel(max(final_volume - ancestor_volume, 0.0), ancestor_volume)
    ancestor_spread = _radius_gyration(ancestor_points, access.domain_length)
    final_spread = _radius_gyration(final_points, access.domain_length)
    ancestor_spread_over_final = ancestor_spread / max(final_spread, 1e-12)
    shape_stretch = _shape_stretch_penalty(ancestor_points, final_points, access.domain_length)
    g = min(max(omega_growth, 0.0), max(s_growth, 0.0))
    channels = {
        "B_growth": b_growth,
        "D_material": d_rel,
        "volume_growth": volume_growth,
        "ancestor_spread_over_final": ancestor_spread_over_final,
        "shape_stretch_penalty": shape_stretch,
    }
    r = max(channels.values())
    dominant = max(channels, key=channels.get)
    q = g / (r + float(args.eta))

    final_cov = _cov_eigenvalues(final_points, access.domain_length)
    ancestor_cov = _cov_eigenvalues(ancestor_points, access.domain_length)
    return {
        "t0": float(args.t0),
        "t1": float(args.t1),
        "threshold_mode": str(args.threshold_mode),
        "component_id": int(args.component_id),
        "skeleton_particles": int(skeleton_count),
        "tube_radius_dx": float(radius_dx),
        "tube_radius_physical": float(radius_dx) * float(args.patch_spacing),
        "num_particles": int(len(final_points)),
        "particle_retention_fraction": float(np.mean(finite)) if len(finite) else 0.0,
        "omega_max_t0": omega0,
        "omega_max_t1": omega1,
        "omega_growth_rel": omega_growth,
        "Splus_t0": s0,
        "Splus_t1": s1,
        "Splus_growth_rel": s_growth,
        "B_t0": b0,
        "B_t1": b1,
        "B_growth_rel": b_growth,
        "D_rel": d_rel,
        "ancestor_volume_t0": ancestor_volume,
        "final_volume_t1": final_volume,
        "volume_growth_rel": volume_growth,
        "ancestor_spread_t0": ancestor_spread,
        "final_spread_t1": final_spread,
        "ancestor_spread_over_final": ancestor_spread_over_final,
        "shape_stretch_penalty": shape_stretch,
        "G": g,
        "R": r,
        "Q": q,
        "dominant_R": dominant,
        "final_radius_gyration": final_spread,
        "final_q90_radius": _q_radius(final_points, access.domain_length, 0.90),
        "final_max_pairwise": _max_pairwise(final_points, access.domain_length),
        "ancestor_radius_gyration": ancestor_spread,
        "ancestor_q90_radius": _q_radius(ancestor_points, access.domain_length, 0.90),
        "ancestor_max_pairwise": _max_pairwise(ancestor_points, access.domain_length),
        "final_cov_eval_0": final_cov[0],
        "final_cov_eval_1": final_cov[1],
        "final_cov_eval_2": final_cov[2],
        "ancestor_cov_eval_0": ancestor_cov[0],
        "ancestor_cov_eval_1": ancestor_cov[1],
        "ancestor_cov_eval_2": ancestor_cov[2],
        "return_error_mean_over_diameter": float(np.mean(return_error) / final_diameter) if return_error.size else np.nan,
        "return_error_max_over_diameter": float(np.max(return_error) / final_diameter) if return_error.size else np.nan,
    }


def _tube_particles(
    centers: np.ndarray,
    *,
    radius: float,
    points_per_center: int,
    seed: int,
    domain_length: float,
) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if radius <= 0.0:
        return centers.copy()
    rng = np.random.default_rng(seed)
    offsets = _ball_offsets(radius, max(1, int(points_per_center)), rng)
    points = (centers[:, None, :] + offsets[None, :, :]) % domain_length
    return np.vstack([centers, points.reshape(-1, 3)])


def _ball_offsets(radius: float, count: int, rng: np.random.Generator) -> np.ndarray:
    directions = rng.normal(size=(count, 3))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    radii = radius * rng.random(count) ** (1.0 / 3.0)
    return directions * radii.reshape(-1, 1)


def _stable_seed(*parts: object) -> int:
    data = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little") % (2**32)


def _radius_gyration(points: np.ndarray, domain_length: float) -> float:
    if len(points) == 0:
        return 0.0
    unwrapped = _unwrap(points, domain_length)
    center = np.mean(unwrapped, axis=0)
    return float(np.sqrt(np.mean(np.sum((unwrapped - center.reshape(1, 3)) ** 2, axis=1))))


def _q_radius(points: np.ndarray, domain_length: float, q: float) -> float:
    if len(points) == 0:
        return 0.0
    unwrapped = _unwrap(points, domain_length)
    center = np.mean(unwrapped, axis=0)
    radius = np.sqrt(np.sum((unwrapped - center.reshape(1, 3)) ** 2, axis=1))
    return float(np.quantile(radius, q))


def _max_pairwise(points: np.ndarray, domain_length: float) -> float:
    if len(points) < 2:
        return 0.0
    points = np.asarray(points, dtype=float)
    max_dist = 0.0
    block = 256
    for start in range(0, len(points), block):
        delta = points[start : start + block, None, :] - points[None, :, :]
        dist = _periodic_norm(delta, domain_length)
        max_dist = max(max_dist, float(np.max(dist)))
    return max_dist


def _cov_eigenvalues(points: np.ndarray, domain_length: float) -> np.ndarray:
    if len(points) < 4:
        return np.asarray([0.0, 0.0, 0.0], dtype=float)
    unwrapped = _unwrap(points, domain_length)
    evals = np.linalg.eigvalsh(np.cov(unwrapped.T))
    return np.asarray(np.clip(evals, 0.0, None), dtype=float)


if __name__ == "__main__":
    main()
