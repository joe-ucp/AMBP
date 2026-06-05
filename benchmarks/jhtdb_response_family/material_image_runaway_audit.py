from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.jhtdb_response_family.cache import cached_query_points
from benchmarks.jhtdb_response_family.config import JHTDBAccessConfig
from benchmarks.jhtdb_response_family.jhtdb_client import make_flow_client
from benchmarks.jhtdb_response_family.relaxed_patch_stretch_budget_audit import _point_metrics
from benchmarks.jhtdb_response_family.sampling import patch_points


DEFAULT_WINDOWS = ("w12_02_14", "w2_02_04")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Particle-cloud material-image isolated-runaway audit for selected "
            "REAL-NS counterexample windows."
        )
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--windows-csv", type=Path, required=True)
    parser.add_argument("--window-label", action="append", default=[])
    parser.add_argument("--parent-core-id", default="0:0")
    parser.add_argument("--dataset", default="isotropic1024coarse")
    parser.add_argument("--source-mode", choices=("synthetic", "soap", "pyjhtdb", "auto"), default="pyjhtdb")
    parser.add_argument("--patch-points-per-axis", type=int, default=49)
    parser.add_argument("--patch-spacing", type=float, default=2.0 * np.pi / 1024.0 * 4.0)
    parser.add_argument("--lambda-threshold-fraction", type=float, default=0.10)
    parser.add_argument("--lambda-threshold-absolute", type=float, default=None)
    parser.add_argument(
        "--seed-radius",
        type=float,
        default=None,
        help="Optional periodic radius around the parent path position used to localize the seed component.",
    )
    parser.add_argument("--substep-dt", type=float, default=0.005)
    parser.add_argument("--max-points-per-request", type=int, default=2048)
    parser.add_argument("--eta", type=float, default=1e-12)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    paths = pd.read_csv(args.run_dir / "material_heat_age_paths.csv")
    starts = pd.read_csv(args.run_dir / "material_heat_age_starts.csv")
    windows = _load_windows(args)

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

    rows = []
    for window in windows:
        rows.append(_audit_window(window, args, access, client, paths, starts, grid_points, shape))

    out = pd.DataFrame(rows)
    output_csv = args.output_csv or args.run_dir / "material_image_runaway_audit.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    print("material_image_runaway_audit", flush=True)
    print(out.to_string(index=False), flush=True)
    print(f"\nrows={output_csv}", flush=True)


def _load_windows(args: argparse.Namespace) -> list[dict[str, float | str]]:
    windows_df = pd.read_csv(args.windows_csv)
    labels = set(args.window_label or DEFAULT_WINDOWS)
    windows_df = windows_df[windows_df["window_label"].isin(labels)]
    if windows_df.empty:
        raise SystemExit(f"no requested windows found in {args.windows_csv}: {sorted(labels)}")
    return [
        {"window_id": str(row.window_label), "t0": float(row.time0), "t1": float(row.time1)}
        for row in windows_df.itertuples(index=False)
    ]


def _audit_window(
    window: dict[str, float | str],
    args: argparse.Namespace,
    access: JHTDBAccessConfig,
    client,
    paths: pd.DataFrame,
    starts: pd.DataFrame,
    grid_points: np.ndarray,
    shape: tuple[int, int, int],
) -> dict[str, float | int | str | bool]:
    window_id = str(window["window_id"])
    t0 = float(window["t0"])
    t1 = float(window["t1"])
    parent = _path_row(paths, args.parent_core_id, t0)
    if parent is None:
        raise SystemExit(f"parent core {args.parent_core_id} has no path row at t0={t0}")
    parent_position = np.asarray([parent["x"], parent["y"], parent["z"]], dtype=float)

    grid0 = _query_patch(client, access, time=t0, points=grid_points, args=args)
    metrics0 = _point_metrics(grid0.gradient)
    lam_pos0 = np.maximum(metrics0["lambda"], 0.0)
    threshold = (
        float(args.lambda_threshold_absolute)
        if args.lambda_threshold_absolute is not None
        else float(args.lambda_threshold_fraction) * float(np.nanmax(lam_pos0))
    )
    active = lam_pos0 > threshold
    parent_distances = _periodic_norm(grid_points - parent_position.reshape(1, 3), access.domain_length)
    if args.seed_radius is not None:
        active &= parent_distances <= float(args.seed_radius)
    nearest_parent = int(np.argmin(_periodic_norm(grid_points - parent_position.reshape(1, 3), access.domain_length)))
    anchor = _anchor_index(grid_points, active, parent_position, access.domain_length)
    component_mask = _connected_component(active.reshape(shape), np.unravel_index(anchor, shape)).ravel()
    seed_points = grid_points[component_mask].copy()
    if len(seed_points) == 0:
        raise SystemExit(f"empty seed component for {window_id} at threshold {threshold}")

    seed_metrics = {key: value[component_mask] for key, value in metrics0.items()}
    final_points = _advect_cloud(
        seed_points,
        t0=t0,
        t1=t1,
        args=args,
        access=access,
        client=client,
        window_id=window_id,
    )
    finite = np.isfinite(final_points).all(axis=1)
    retained_seed = seed_points[finite]
    retained_final = final_points[finite] % access.domain_length
    retained_metrics0 = {key: value[finite] for key, value in seed_metrics.items()}

    final_data = cached_query_points(
        client,
        access,
        time=t1,
        points=retained_final,
        role=f"material_image_runaway:window={window_id}:parent={args.parent_core_id}:final_cloud",
    )
    final_metrics = _point_metrics(final_data.gradient)

    voxel_volume = float(args.patch_spacing) ** 3
    lam0 = np.asarray(retained_metrics0["lambda"], dtype=float)
    lam1 = np.asarray(final_metrics["lambda"], dtype=float)
    pos0 = np.maximum(lam0, 0.0)
    pos1 = np.maximum(lam1, 0.0)
    neg0 = np.maximum(-lam0, 0.0)
    neg1 = np.maximum(-lam1, 0.0)

    s0 = float(np.sum(pos0) * voxel_volume)
    s1 = float(np.sum(pos1) * voxel_volume)
    b0 = float(np.sum(neg0) * voxel_volume)
    b1 = float(np.sum(neg1) * voxel_volume)
    d_rel = float(np.sum(np.maximum(pos0 - pos1, 0.0)) * voxel_volume) / max(s0, 1e-12)
    omega0 = float(np.max(np.asarray(retained_metrics0["omega_norm"], dtype=float)))
    omega1 = float(np.max(np.asarray(final_metrics["omega_norm"], dtype=float)))
    omega_growth_rel = _rel(omega1 - omega0, omega0)
    s_growth_rel = _rel(s1 - s0, s0)
    b_growth_rel = _rel(max(b1 - b0, 0.0), s0)

    material_volume_t0 = _occupied_voxel_volume(retained_seed, args.patch_spacing, access.domain_length)
    material_volume_t1 = _occupied_voxel_volume(retained_final, args.patch_spacing, access.domain_length)
    volume_growth_rel = _rel(max(material_volume_t1 - material_volume_t0, 0.0), material_volume_t0)

    daughter_ids = _descendants_alive(starts, args.parent_core_id, t1)
    daughter_fraction = _daughter_splus_fraction(paths, daughter_ids, t1, retained_final, pos1, voxel_volume, s1, args, access)
    outside_fraction = _outside_material_splus_fraction(
        client,
        access,
        t1,
        grid_points,
        retained_final,
        args,
        window_id,
        s1,
        voxel_volume,
    )

    g_m = min(max(omega_growth_rel, 0.0), max(s_growth_rel, 0.0))
    channels = {
        "volume_growth": volume_growth_rel,
        "B_growth": b_growth_rel,
        "D_material": d_rel,
        "outside_material_Splus": outside_fraction,
    }
    r_m = max(channels.values())
    dominant = max(channels, key=channels.get)
    q_m = g_m / (r_m + args.eta)

    return {
        "window_id": window_id,
        "parent_core_id": args.parent_core_id,
        "descendant_ids": ";".join(daughter_ids),
        "num_seed_particles": int(len(seed_points)),
        "particle_retention_fraction": float(np.mean(finite)) if len(finite) else 0.0,
        "seed_lambda_threshold": threshold,
        "seed_lambda_threshold_fraction": float(args.lambda_threshold_fraction),
        "seed_radius": float(args.seed_radius) if args.seed_radius is not None else np.nan,
        "parent_lambda_t0": float(parent["lambda"]),
        "parent_omega_t0": float(parent["omega_norm"]),
        "nearest_parent_grid_lambda_t0": float(metrics0["lambda"][nearest_parent]),
        "nearest_parent_grid_omega_t0": float(metrics0["omega_norm"][nearest_parent]),
        "nearest_parent_grid_active": bool(active[nearest_parent]),
        "nearest_parent_grid_in_seed_component": bool(component_mask[nearest_parent]),
        "anchor_distance_to_parent": float(
            _periodic_norm(grid_points[anchor] - parent_position, access.domain_length)
        ),
        "anchor_lambda_t0": float(metrics0["lambda"][anchor]),
        "anchor_omega_t0": float(metrics0["omega_norm"][anchor]),
        "patch_points_per_axis": int(args.patch_points_per_axis),
        "patch_spacing": float(args.patch_spacing),
        "material_volume_t0": material_volume_t0,
        "material_volume_t1": material_volume_t1,
        "volume_growth_rel": volume_growth_rel,
        "omega_max_material_t0": omega0,
        "omega_max_material_t1": omega1,
        "omega_growth_rel": omega_growth_rel,
        "Splus_material_t0": s0,
        "Splus_material_t1": s1,
        "Splus_growth_rel": s_growth_rel,
        "B_material_t0": b0,
        "B_material_t1": b1,
        "B_growth_rel": b_growth_rel,
        "D_material_rel": d_rel,
        "daughter_Splus_fraction": daughter_fraction,
        "outside_material_Splus_fraction": outside_fraction,
        "G_M": g_m,
        "R_M": r_m,
        "Q_M": q_m,
        "dominant_R_channel": dominant,
        "counterexample_candidate_Q_gt_1": bool(g_m > 0.05 and q_m > 1.0),
        "counterexample_candidate_R_lt_0p1G": bool(g_m > 0.05 and r_m < 0.1 * g_m),
        "measurement_note": (
            "particle cloud from connected lambda+ component; S/B/D are material-particle "
            "weighted; volume is occupied-grid support proxy; outside S+ is sampled-patch proxy"
        ),
    }


def _query_patch(client, access: JHTDBAccessConfig, *, time: float, points: np.ndarray, args: argparse.Namespace):
    return cached_query_points(
        client,
        access,
        time=float(time),
        points=points,
        role=(
            "relaxed_patch_stretch_budget:"
            f"n={args.patch_points_per_axis}:dx={args.patch_spacing:.12g}:"
            "psi=0.005:center=source_birth"
        ),
    )


def _advect_cloud(
    points: np.ndarray,
    *,
    t0: float,
    t1: float,
    args: argparse.Namespace,
    access: JHTDBAccessConfig,
    client,
    window_id: str,
) -> np.ndarray:
    out = np.asarray(points, dtype=float).copy()
    total = float(t1 - t0)
    if abs(total) <= 1e-15:
        return out
    direction = 1.0 if total > 0.0 else -1.0
    step = abs(float(args.substep_dt))
    if step <= 0.0:
        raise ValueError("--substep-dt must be positive")
    t = float(t0)
    remaining = abs(total)
    substep_index = 0
    while remaining > 1e-12:
        h = direction * min(step, remaining)
        out = _rk4_step(out, t, h, args, access, client, window_id, substep_index)
        out %= access.domain_length
        t += h
        remaining -= abs(h)
        substep_index += 1
    return out


def _rk4_step(
    points: np.ndarray,
    t: float,
    h: float,
    args: argparse.Namespace,
    access: JHTDBAccessConfig,
    client,
    window_id: str,
    substep_index: int,
) -> np.ndarray:
    k1 = _velocity(client, access, t, points, window_id, substep_index, "k1")
    k2 = _velocity(client, access, t + 0.5 * h, points + 0.5 * h * k1, window_id, substep_index, "k2")
    k3 = _velocity(client, access, t + 0.5 * h, points + 0.5 * h * k2, window_id, substep_index, "k3")
    k4 = _velocity(client, access, t + h, points + h * k3, window_id, substep_index, "k4")
    return points + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _velocity(client, access: JHTDBAccessConfig, time: float, points: np.ndarray, window_id: str, substep: int, stage: str) -> np.ndarray:
    wrapped = np.asarray(points, dtype=float) % access.domain_length
    data = cached_query_points(
        client,
        access,
        time=float(time),
        points=wrapped,
        role=f"material_image_runaway:window={window_id}:substep={substep}:{stage}:velocity",
    )
    return np.asarray(data.velocity, dtype=float)


def _anchor_index(points: np.ndarray, active: np.ndarray, position: np.ndarray, domain_length: float) -> int:
    distances = _periodic_norm(points - position.reshape(1, 3), domain_length)
    nearest = int(np.argmin(distances))
    if bool(active[nearest]):
        return nearest
    active_indices = np.flatnonzero(active)
    if active_indices.size == 0:
        raise SystemExit("no active lambda+ points available for seed component")
    return int(active_indices[np.argmin(distances[active_indices])])


def _connected_component(active: np.ndarray, anchor: tuple[int, int, int]) -> np.ndarray:
    if not bool(active[anchor]):
        raise ValueError("anchor must be active")
    out = np.zeros_like(active, dtype=bool)
    queue: deque[tuple[int, int, int]] = deque([anchor])
    out[anchor] = True
    shape = active.shape
    while queue:
        i, j, k = queue.popleft()
        for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            ni, nj, nk = i + di, j + dj, k + dk
            if ni < 0 or nj < 0 or nk < 0 or ni >= shape[0] or nj >= shape[1] or nk >= shape[2]:
                continue
            if active[ni, nj, nk] and not out[ni, nj, nk]:
                out[ni, nj, nk] = True
                queue.append((ni, nj, nk))
    return out


def _occupied_voxel_volume(points: np.ndarray, spacing: float, domain_length: float) -> float:
    if len(points) == 0:
        return 0.0
    bins = np.floor((np.asarray(points, dtype=float) % domain_length) / float(spacing)).astype(np.int64)
    unique = np.unique(bins, axis=0)
    return float(len(unique)) * float(spacing) ** 3


def _outside_material_splus_fraction(
    client,
    access: JHTDBAccessConfig,
    time: float,
    grid_points: np.ndarray,
    material_points: np.ndarray,
    args: argparse.Namespace,
    window_id: str,
    s_material_t1: float,
    voxel_volume: float,
) -> float:
    grid = cached_query_points(
        client,
        access,
        time=float(time),
        points=grid_points,
        role=(
            "relaxed_patch_stretch_budget:"
            f"n={args.patch_points_per_axis}:dx={args.patch_spacing:.12g}:"
            "psi=0.005:center=source_birth"
        ),
    )
    metrics = _point_metrics(grid.gradient)
    grid_bins = _voxel_keys(grid_points, args.patch_spacing, access.domain_length)
    material_bins = set(_voxel_keys(material_points, args.patch_spacing, access.domain_length))
    outside = np.asarray([key not in material_bins for key in grid_bins], dtype=bool)
    s_outside = float(np.sum(np.maximum(metrics["lambda"][outside], 0.0)) * voxel_volume)
    return s_outside / max(s_material_t1, 1e-12)


def _daughter_splus_fraction(
    paths: pd.DataFrame,
    daughter_ids: list[str],
    time: float,
    points: np.ndarray,
    pos: np.ndarray,
    voxel_volume: float,
    s_total: float,
    args: argparse.Namespace,
    access: JHTDBAccessConfig,
) -> float:
    if not daughter_ids or len(points) == 0:
        return 0.0
    mask = np.zeros(len(points), dtype=bool)
    for core_id in daughter_ids:
        row = _path_row(paths, core_id, time)
        if row is None:
            continue
        center = np.asarray([row["x"], row["y"], row["z"]], dtype=float)
        radius = max(float(row.get("local_max_radius", 0.0) or 0.0), float(args.patch_spacing))
        mask |= _periodic_norm(points - center.reshape(1, 3), access.domain_length) <= radius
    return float(np.sum(pos[mask]) * voxel_volume) / max(s_total, 1e-12)


def _voxel_keys(points: np.ndarray, spacing: float, domain_length: float) -> list[tuple[int, int, int]]:
    bins = np.floor((np.asarray(points, dtype=float) % domain_length) / float(spacing)).astype(np.int64)
    return [tuple(int(value) for value in row) for row in bins]


def _descendants_alive(starts: pd.DataFrame, parent_core_id: str, time: float) -> list[str]:
    children: dict[str, list[tuple[str, float]]] = {}
    for row in starts.itertuples(index=False):
        parent = getattr(row, "parent_core_id", "")
        if parent is None or (isinstance(parent, float) and np.isnan(parent)):
            continue
        parent_id = str(parent)
        if not parent_id or parent_id.lower() == "nan":
            continue
        children.setdefault(parent_id, []).append((str(getattr(row, "core_id")), float(getattr(row, "birth_time"))))

    out: list[str] = []

    def collect(core_id: str) -> None:
        for child_id, birth_time in children.get(core_id, []):
            if birth_time <= time:
                out.append(child_id)
                collect(child_id)

    collect(parent_core_id)
    return out


def _path_row(paths: pd.DataFrame, core_id: str, time: float) -> pd.Series | None:
    part = paths[paths["core_id"].astype(str) == str(core_id)].copy()
    if part.empty:
        return None
    times = pd.to_numeric(part["time"], errors="coerce")
    idx = (times - float(time)).abs().idxmin()
    if abs(float(times.loc[idx]) - float(time)) > 1e-9:
        return None
    return part.loc[idx]


def _source_center(starts: pd.DataFrame, parent_core_id: str) -> np.ndarray:
    row = starts[starts["core_id"].astype(str) == str(parent_core_id)]
    if row.empty:
        raise SystemExit(f"missing parent core in starts: {parent_core_id}")
    first = row.iloc[0]
    return np.asarray([first["source_birth_x"], first["source_birth_y"], first["source_birth_z"]], dtype=float)


def _periodic_norm(delta: np.ndarray, domain_length: float) -> np.ndarray:
    wrapped = (np.asarray(delta, dtype=float) + 0.5 * domain_length) % domain_length - 0.5 * domain_length
    return np.linalg.norm(wrapped, axis=-1)


def _rel(delta: float, baseline: float) -> float:
    return float(delta) / max(abs(float(baseline)), 1e-12)


if __name__ == "__main__":
    main()
