from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.jhtdb_response_family.cache import cached_query_points
from benchmarks.jhtdb_response_family.config import JHTDBAccessConfig
from benchmarks.jhtdb_response_family.features import strain_rotation, vorticity_from_gradient
from benchmarks.jhtdb_response_family.global_hotspot_ancestry_scan import (
    _components,
    _shape_stretch_penalty,
    _threshold_mask,
)
from benchmarks.jhtdb_response_family.jhtdb_client import make_flow_client
from benchmarks.jhtdb_response_family.material_image_runaway_audit import (
    _advect_cloud,
    _occupied_voxel_volume,
    _query_patch,
    _rel,
    _source_center,
)
from benchmarks.jhtdb_response_family.relaxed_patch_stretch_budget_audit import _point_metrics
from benchmarks.jhtdb_response_family.sampling import patch_points
from benchmarks.jhtdb_response_family.skeleton_radial_thickening_profile import (
    _max_pairwise,
    _periodic_norm,
    _q_radius,
    _radius_gyration,
    _stable_seed,
    _tube_particles,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decompose the finite-radius REAL-NS closure channel around a sparse amplification skeleton."
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
    parser.add_argument(
        "--windows-csv",
        type=Path,
        default=Path(
            "benchmarks/jhtdb_response_family/results/"
            "material_heat_age_audit_patch17_h14_top3_isolated_runaway_search_n81_windows.csv"
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
    parser.add_argument(
        "--tube-radii-dx",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2, 0.25, 0.35, 0.5, 1.0, 2.0],
    )
    parser.add_argument("--points-per-center", type=int, default=16)
    parser.add_argument("--substep-dt", type=float, default=0.005)
    parser.add_argument("--max-points-per-request", type=int, default=2048)
    parser.add_argument("--eta", type=float, default=1e-12)
    parser.add_argument("--rng-seed", type=int, default=1729)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "benchmarks/jhtdb_response_family/results/"
            "skeleton_closure_mechanism_top2_c413.csv"
        ),
    )
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
    final_snapshot = _query_patch(client, access, time=float(args.t1), points=grid_points, args=args)
    final_patch_metrics = _point_metrics(final_snapshot.gradient)
    mask = _threshold_mask(
        str(args.threshold_mode),
        np.maximum(final_patch_metrics["lambda"], 0.0),
        final_patch_metrics["omega_norm"],
    )
    components = _components(mask, (args.patch_points_per_axis,) * 3)
    if int(args.component_id) >= len(components):
        raise SystemExit(f"component {args.component_id} not found; only {len(components)} components")
    skeleton_indices = components[int(args.component_id)]
    skeleton_points = grid_points[skeleton_indices]

    rows = []
    for radius_dx in [float(v) for v in args.tube_radii_dx]:
        final_points = _tube_particles(
            skeleton_points,
            radius=radius_dx * float(args.patch_spacing),
            points_per_center=int(args.points_per_center),
            seed=_stable_seed(args.rng_seed, "mechanism", str(args.component_id), radius_dx),
            domain_length=access.domain_length,
        )
        rows.append(_score_mechanism(args, access, client, final_points, radius_dx, len(skeleton_points)))

    out = pd.DataFrame(rows).sort_values("radius_dx").reset_index(drop=True)
    out = _add_impulse_columns(out)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    cols = [
        "radius_dx",
        "num_particles",
        "G",
        "R",
        "Q",
        "dominant_R",
        "shape_stretch_total",
        "axial_stretch",
        "cross_section_area_growth",
        "volume_jacobian_growth",
        "cauchy_green_anisotropy",
        "vorticity_direction_rotation_mean_rad",
        "strain_frame_rotation_mean_rad",
        "closure_impulse_channel",
    ]
    print("skeleton_closure_mechanism_audit", flush=True)
    print(out[cols].to_string(index=False), flush=True)
    print(f"\noutput={args.output_csv}", flush=True)


def _score_mechanism(
    args: argparse.Namespace,
    access: JHTDBAccessConfig,
    client,
    final_points: np.ndarray,
    radius_dx: float,
    skeleton_count: int,
) -> dict[str, float | int | str]:
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
    final_metrics = _rich_metrics(final_data.gradient)
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
    ancestor_metrics = _rich_metrics(ancestor_data.gradient)
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

    deformation = _deformation_diagnostics(ancestor_points, final_points, access.domain_length)
    rotations = _rotation_diagnostics(ancestor_metrics, final_metrics)
    geometry = _cloud_geometry(ancestor_points, final_points, access.domain_length)

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

    row: dict[str, float | int | str] = {
        "t0": float(args.t0),
        "t1": float(args.t1),
        "threshold_mode": str(args.threshold_mode),
        "component_id": int(args.component_id),
        "skeleton_particles": int(skeleton_count),
        "radius_dx": float(radius_dx),
        "radius_physical": float(radius_dx) * float(args.patch_spacing),
        "num_particles": int(len(final_points)),
        "particle_retention_fraction": float(np.mean(finite)) if len(finite) else 0.0,
        "omega_growth_rel": omega_growth,
        "Splus_growth_rel": s_growth,
        "B_growth_rel": b_growth,
        "D_rel": d_rel,
        "volume_growth_rel": volume_growth,
        "ancestor_spread_over_final": ancestor_spread_over_final,
        "shape_stretch_total": shape_stretch,
        "G": g,
        "R": r,
        "Q": q,
        "dominant_R": dominant,
        "return_error_mean_over_diameter": float(np.mean(return_error) / final_diameter) if return_error.size else np.nan,
        "return_error_max_over_diameter": float(np.max(return_error) / final_diameter) if return_error.size else np.nan,
        "ancestor_radius_gyration": ancestor_spread,
        "final_radius_gyration": final_spread,
        "ancestor_q90_radius": _q_radius(ancestor_points, access.domain_length, 0.90),
        "final_q90_radius": _q_radius(final_points, access.domain_length, 0.90),
        "ancestor_max_pairwise": _max_pairwise(ancestor_points, access.domain_length),
        "final_max_pairwise": _max_pairwise(final_points, access.domain_length),
    }
    row.update(deformation)
    row.update(rotations)
    row.update(geometry)
    return row


def _rich_metrics(gradient: np.ndarray) -> dict[str, np.ndarray]:
    gradient = np.asarray(gradient, dtype=float).reshape((-1, 3, 3))
    omega = vorticity_from_gradient(gradient)
    omega_norm = np.linalg.norm(omega, axis=1)
    xi = omega / np.maximum(omega_norm[:, None], 1e-12)
    strain = np.asarray([strain_rotation(g)[0] for g in gradient])
    sxi = np.einsum("nij,nj->ni", strain, xi)
    lam = np.einsum("ni,ni->n", xi, sxi)
    evals = np.empty((len(gradient), 3), dtype=float)
    evecs = np.empty((len(gradient), 3, 3), dtype=float)
    for i, mat in enumerate(strain):
        vals, vecs = np.linalg.eigh(mat)
        order = np.argsort(vals)
        evals[i] = vals[order]
        evecs[i] = vecs[:, order]
    return {
        "lambda": lam,
        "omega": omega,
        "omega_norm": omega_norm,
        "xi": xi,
        "strain": strain,
        "strain_evals": evals,
        "strain_evecs": evecs,
        "strain_max_evec": evecs[:, :, 2],
        "alignment_max": np.abs(np.einsum("ni,ni->n", xi, evecs[:, :, 2])),
    }


def _deformation_diagnostics(
    ancestor_points: np.ndarray, final_points: np.ndarray, domain_length: float
) -> dict[str, float]:
    x0 = _centered_unwrapped(ancestor_points, domain_length)
    x1 = _centered_unwrapped(final_points, domain_length)
    if len(x0) < 4:
        return {
            "axial_stretch": 0.0,
            "transverse_stretch_mean": 0.0,
            "cross_section_area_growth": 0.0,
            "volume_jacobian_growth": 0.0,
            "signed_volume_jacobian_growth": 0.0,
            "cauchy_green_anisotropy": 0.0,
            "tube_shear_ratio": 0.0,
            "deformation_singular_0": np.nan,
            "deformation_singular_1": np.nan,
            "deformation_singular_2": np.nan,
        }
    # Least-squares material map from ancestor cloud to final cloud.
    a, *_ = np.linalg.lstsq(x0, x1, rcond=None)
    f = a.T
    singular = np.linalg.svd(f, compute_uv=False)
    singular = np.sort(np.clip(singular, 1e-12, None))
    det_f = float(np.linalg.det(f))
    area = float(singular[0] * singular[1])
    volume = float(np.prod(singular))
    cov0 = _cov_eigenvectors(ancestor_points, domain_length)[1]
    local_f = cov0.T @ f @ cov0
    diag_norm = float(np.linalg.norm(np.diag(np.diag(local_f))))
    offdiag = local_f - np.diag(np.diag(local_f))
    shear = float(np.linalg.norm(offdiag) / max(diag_norm, 1e-12))
    return {
        "axial_stretch": max(float(singular[2] - 1.0), 0.0),
        "transverse_stretch_mean": max(float(np.sqrt(area) - 1.0), 0.0),
        "cross_section_area_growth": max(area - 1.0, 0.0),
        "volume_jacobian_growth": max(volume - 1.0, 0.0),
        "signed_volume_jacobian_growth": det_f - 1.0,
        "cauchy_green_anisotropy": max(float((singular[2] / singular[0]) ** 2 - 1.0), 0.0),
        "tube_shear_ratio": shear,
        "deformation_singular_0": float(singular[0]),
        "deformation_singular_1": float(singular[1]),
        "deformation_singular_2": float(singular[2]),
    }


def _rotation_diagnostics(ancestor_metrics: dict[str, np.ndarray], final_metrics: dict[str, np.ndarray]) -> dict[str, float]:
    xi0 = np.asarray(ancestor_metrics["xi"], dtype=float)
    xi1 = np.asarray(final_metrics["xi"], dtype=float)
    e0 = np.asarray(ancestor_metrics["strain_max_evec"], dtype=float)
    e1 = np.asarray(final_metrics["strain_max_evec"], dtype=float)
    xi_dot = np.clip(np.abs(np.einsum("ni,ni->n", xi0, xi1)), 0.0, 1.0)
    e_dot = np.clip(np.abs(np.einsum("ni,ni->n", e0, e1)), 0.0, 1.0)
    xi_angle = np.arccos(xi_dot)
    e_angle = np.arccos(e_dot)
    align0 = np.asarray(ancestor_metrics["alignment_max"], dtype=float)
    align1 = np.asarray(final_metrics["alignment_max"], dtype=float)
    return {
        "vorticity_direction_rotation_mean_rad": float(np.mean(xi_angle)) if xi_angle.size else np.nan,
        "vorticity_direction_rotation_max_rad": float(np.max(xi_angle)) if xi_angle.size else np.nan,
        "strain_frame_rotation_mean_rad": float(np.mean(e_angle)) if e_angle.size else np.nan,
        "strain_frame_rotation_max_rad": float(np.max(e_angle)) if e_angle.size else np.nan,
        "alignment_max_mean_t0": float(np.mean(align0)) if align0.size else np.nan,
        "alignment_max_mean_t1": float(np.mean(align1)) if align1.size else np.nan,
        "alignment_max_mean_delta": float(np.mean(align1) - np.mean(align0)) if align0.size and align1.size else np.nan,
    }


def _cloud_geometry(
    ancestor_points: np.ndarray, final_points: np.ndarray, domain_length: float
) -> dict[str, float]:
    eval0, _evec0 = _cov_eigenvectors(ancestor_points, domain_length)
    eval1, _evec1 = _cov_eigenvectors(final_points, domain_length)
    cond0 = float(eval0[-1] / max(eval0[0], 1e-12))
    cond1 = float(eval1[-1] / max(eval1[0], 1e-12))
    line0 = float(np.sqrt(max(eval0[0] + eval0[1], 0.0)) / max(np.sqrt(eval0[2]), 1e-12))
    line1 = float(np.sqrt(max(eval1[0] + eval1[1], 0.0)) / max(np.sqrt(eval1[2]), 1e-12))
    return {
        "ancestor_cov_eval_0": float(eval0[0]),
        "ancestor_cov_eval_1": float(eval0[1]),
        "ancestor_cov_eval_2": float(eval0[2]),
        "final_cov_eval_0": float(eval1[0]),
        "final_cov_eval_1": float(eval1[1]),
        "final_cov_eval_2": float(eval1[2]),
        "ancestor_shape_condition": cond0,
        "final_shape_condition": cond1,
        "line_residual_proxy_t0": line0,
        "line_residual_proxy_t1": line1,
        "curvature_proxy_growth": max(line1 / max(line0, 1e-12) - 1.0, 0.0),
    }


def _cov_eigenvectors(points: np.ndarray, domain_length: float) -> tuple[np.ndarray, np.ndarray]:
    centered = _centered_unwrapped(points, domain_length)
    if len(centered) < 2:
        return np.asarray([1e-12, 1e-12, 1e-12], dtype=float), np.eye(3)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-12, None)
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def _centered_unwrapped(points: np.ndarray, domain_length: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return points.reshape(0, 3)
    base = points[0]
    unwrapped = base.reshape(1, 3) + ((points - base.reshape(1, 3) + 0.5 * domain_length) % domain_length - 0.5 * domain_length)
    return unwrapped - np.mean(unwrapped, axis=0, keepdims=True)


def _add_impulse_columns(out: pd.DataFrame) -> pd.DataFrame:
    if out.empty:
        return out
    channels = [
        "B_growth_rel",
        "D_rel",
        "volume_growth_rel",
        "ancestor_spread_over_final",
        "shape_stretch_total",
        "axial_stretch",
        "cross_section_area_growth",
        "volume_jacobian_growth",
        "cauchy_green_anisotropy",
        "tube_shear_ratio",
        "vorticity_direction_rotation_mean_rad",
        "strain_frame_rotation_mean_rad",
        "curvature_proxy_growth",
    ]
    baseline = out.iloc[0]
    impulse_names = []
    for channel in channels:
        name = f"impulse_{channel}"
        impulse_names.append(name)
        denom = np.maximum(out["radius_dx"].to_numpy(dtype=float), 1e-12)
        out[name] = (out[channel].to_numpy(dtype=float) - float(baseline[channel])) / denom
        out.loc[np.isclose(out["radius_dx"], 0.0), name] = 0.0
    labels = []
    values = []
    for _, row in out.iterrows():
        radius = float(row["radius_dx"])
        if radius <= 0.0:
            labels.append("")
            values.append(0.0)
            continue
        vals = {name.replace("impulse_", ""): float(row[name]) for name in impulse_names}
        label = max(vals, key=vals.get)
        labels.append(label)
        values.append(vals[label])
    out["closure_impulse_channel"] = labels
    out["closure_impulse_value"] = values
    return out


if __name__ == "__main__":
    main()
