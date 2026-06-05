from __future__ import annotations

import argparse
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.jhtdb_response_family.cache import cached_query_points
from benchmarks.jhtdb_response_family.config import JHTDBAccessConfig
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


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    time_index: int
    t1: float
    t0: float
    lag_intervals: int
    threshold_mode: str
    component_id: int
    component_indices: np.ndarray
    final_score: float
    final_omega_max: float
    final_splus: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Global backward ancestry scan for isolated REAL-NS runaway hotspots. "
            "Final-time components are detected across the whole support cube and "
            "then advected backward as particle clouds."
        )
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--windows-csv", type=Path, required=True)
    parser.add_argument("--parent-core-id", default="0:0", help="Only used to center the existing support cube.")
    parser.add_argument("--dataset", default="isotropic1024coarse")
    parser.add_argument("--source-mode", choices=("synthetic", "soap", "pyjhtdb", "auto"), default="pyjhtdb")
    parser.add_argument("--patch-points-per-axis", type=int, default=49)
    parser.add_argument("--patch-spacing", type=float, default=2.0 * np.pi / 1024.0 * 4.0)
    parser.add_argument("--lags", nargs="+", type=int, default=[2, 4, 8, 12])
    parser.add_argument("--threshold-modes", nargs="+", default=[
        "joint_top0p5",
        "joint_top1",
        "joint_top2",
        "lambda_abs2_omega_top2",
        "lambda_frac0p2_omega_top2",
    ])
    parser.add_argument("--min-component-particles", type=int, default=3)
    parser.add_argument("--max-component-particles", type=int, default=800)
    parser.add_argument("--max-candidates", type=int, default=40)
    parser.add_argument("--substep-dt", type=float, default=0.005)
    parser.add_argument("--max-points-per-request", type=int, default=2048)
    parser.add_argument("--eta", type=float, default=1e-12)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--catalog-csv", type=Path, default=None)
    args = parser.parse_args()

    starts = pd.read_csv(args.run_dir / "material_heat_age_starts.csv")
    times = _time_grid(args.windows_csv)
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

    snapshots = _load_snapshots(times, grid_points, args, access, client)
    candidates = _enumerate_candidates(snapshots, times, args, shape)
    catalog = _catalog(candidates, args)
    selected = sorted(candidates, key=lambda c: c.final_score, reverse=True)[: max(int(args.max_candidates), 0)]
    rows = [_score_candidate(c, snapshots, grid_points, args, access, client) for c in selected]
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Q", "G"], ascending=[False, False]).reset_index(drop=True)

    output_csv = args.output_csv or args.run_dir / "global_hotspot_ancestry_scan.csv"
    catalog_csv = args.catalog_csv or args.run_dir / "global_hotspot_ancestry_scan_catalog.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    catalog.to_csv(catalog_csv, index=False)

    print("global_hotspot_ancestry_scan", flush=True)
    if out.empty:
        print("no scored candidates", flush=True)
    else:
        cols = [
            "t0",
            "t1",
            "lag_intervals",
            "threshold_mode",
            "component_id",
            "num_particles",
            "omega_growth_rel",
            "Splus_growth_rel",
            "G",
            "B_growth_rel",
            "D_rel",
            "ancestor_spread_over_final",
            "shape_stretch_penalty",
            "R",
            "Q",
            "dominant_R",
        ]
        print(out[cols].head(25).to_string(index=False), flush=True)
    print(f"\nscored_rows={output_csv}", flush=True)
    print(f"candidate_catalog={catalog_csv}", flush=True)


def _time_grid(windows_csv: Path) -> np.ndarray:
    windows = pd.read_csv(windows_csv)
    vals = pd.concat([windows["time0"], windows["time1"]], ignore_index=True)
    return np.asarray(sorted(pd.to_numeric(vals, errors="coerce").dropna().unique()), dtype=float)


def _load_snapshots(times: np.ndarray, grid_points: np.ndarray, args: argparse.Namespace, access: JHTDBAccessConfig, client) -> list[dict[str, object]]:
    snapshots = []
    for time in times:
        data = _query_patch(client, access, time=float(time), points=grid_points, args=args)
        metrics = _point_metrics(data.gradient)
        snapshots.append({"time": float(time), "metrics": metrics})
    return snapshots


def _enumerate_candidates(
    snapshots: list[dict[str, object]],
    times: np.ndarray,
    args: argparse.Namespace,
    shape: tuple[int, int, int],
) -> list[Candidate]:
    out: list[Candidate] = []
    voxel_volume = float(args.patch_spacing) ** 3
    lags = sorted({int(lag) for lag in args.lags if int(lag) > 0})
    for time_index, snapshot in enumerate(snapshots):
        t1 = float(snapshot["time"])
        valid_lags = [lag for lag in lags if time_index - lag >= 0]
        if not valid_lags:
            continue
        metrics = snapshot["metrics"]
        lam = np.asarray(metrics["lambda"], dtype=float)
        lam_pos = np.maximum(lam, 0.0)
        omega = np.asarray(metrics["omega_norm"], dtype=float)
        for mode in args.threshold_modes:
            mask = _threshold_mask(mode, lam_pos, omega)
            components = _components(mask, shape)
            kept = 0
            for component_id, indices in enumerate(components):
                n = int(len(indices))
                if n < int(args.min_component_particles) or n > int(args.max_component_particles):
                    continue
                splus = float(np.sum(lam_pos[indices]) * voxel_volume)
                omega_max = float(np.max(omega[indices]))
                final_score = _final_score(splus, omega_max, n)
                for lag in valid_lags:
                    out.append(
                        Candidate(
                            candidate_id=f"{mode}:ti{time_index}:c{component_id}:lag{lag}",
                            time_index=time_index,
                            t1=t1,
                            t0=float(times[time_index - lag]),
                            lag_intervals=lag,
                            threshold_mode=mode,
                            component_id=component_id,
                            component_indices=indices,
                            final_score=final_score,
                            final_omega_max=omega_max,
                            final_splus=splus,
                        )
                    )
                kept += 1
            if kept == 0:
                continue
    return out


def _threshold_mask(mode: str, lam_pos: np.ndarray, omega: np.ndarray) -> np.ndarray:
    if mode == "joint_top0p5":
        return (lam_pos >= np.quantile(lam_pos, 0.995)) & (omega >= np.quantile(omega, 0.995))
    if mode == "joint_top1":
        return (lam_pos >= np.quantile(lam_pos, 0.99)) & (omega >= np.quantile(omega, 0.99))
    if mode == "joint_top2":
        return (lam_pos >= np.quantile(lam_pos, 0.98)) & (omega >= np.quantile(omega, 0.98))
    if mode == "lambda_abs2_omega_top2":
        return (lam_pos > 2.0) & (omega >= np.quantile(omega, 0.98))
    if mode == "lambda_frac0p2_omega_top2":
        return (lam_pos > 0.2 * float(np.nanmax(lam_pos))) & (omega >= np.quantile(omega, 0.98))
    raise ValueError(f"unknown threshold mode: {mode}")


def _components(mask: np.ndarray, shape: tuple[int, int, int]) -> list[np.ndarray]:
    cube = np.asarray(mask, dtype=bool).reshape(shape)
    seen = np.zeros(shape, dtype=bool)
    components: list[np.ndarray] = []
    for start in np.argwhere(cube):
        start_t = tuple(int(v) for v in start)
        if seen[start_t]:
            continue
        queue: deque[tuple[int, int, int]] = deque([start_t])
        seen[start_t] = True
        points: list[tuple[int, int, int]] = []
        while queue:
            i, j, k = queue.popleft()
            points.append((i, j, k))
            for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                ni, nj, nk = i + di, j + dj, k + dk
                if ni < 0 or nj < 0 or nk < 0 or ni >= shape[0] or nj >= shape[1] or nk >= shape[2]:
                    continue
                if cube[ni, nj, nk] and not seen[ni, nj, nk]:
                    seen[ni, nj, nk] = True
                    queue.append((ni, nj, nk))
        flat = np.asarray([np.ravel_multi_index(p, shape) for p in points], dtype=int)
        components.append(flat)
    return components


def _score_candidate(
    candidate: Candidate,
    snapshots: list[dict[str, object]],
    grid_points: np.ndarray,
    args: argparse.Namespace,
    access: JHTDBAccessConfig,
    client,
) -> dict[str, float | int | str | bool]:
    final_points = grid_points[candidate.component_indices].copy()
    ancestor_points = _advect_cloud(
        final_points,
        t0=candidate.t1,
        t1=candidate.t0,
        args=args,
        access=access,
        client=client,
        window_id=f"global_hotspot:{candidate.candidate_id}:backward",
    )
    finite = np.isfinite(ancestor_points).all(axis=1)
    final_points = final_points[finite]
    ancestor_points = ancestor_points[finite] % access.domain_length
    final_indices = candidate.component_indices[finite]

    final_metrics_all = snapshots[candidate.time_index]["metrics"]
    final_metrics = {key: np.asarray(value)[final_indices] for key, value in final_metrics_all.items()}
    ancestor_data = cached_query_points(
        client,
        access,
        time=candidate.t0,
        points=ancestor_points,
        role=f"global_hotspot_ancestry:ancestor:{candidate.candidate_id}",
    )
    ancestor_metrics = _point_metrics(ancestor_data.gradient)

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
    omega0 = float(np.max(np.asarray(ancestor_metrics["omega_norm"], dtype=float))) if len(ancestor_points) else np.nan
    omega1 = float(np.max(np.asarray(final_metrics["omega_norm"], dtype=float))) if len(final_points) else np.nan
    omega_growth = _rel(omega1 - omega0, omega0)
    s_growth = _rel(s1 - s0, s0)
    b_growth = _rel(max(b1 - b0, 0.0), s0)
    d_rel = float(np.sum(np.maximum(pos0 - pos1, 0.0)) * voxel_volume) / max(s0, 1e-12)

    ancestor_volume = _occupied_voxel_volume(ancestor_points, args.patch_spacing, access.domain_length)
    final_volume = _occupied_voxel_volume(final_points, args.patch_spacing, access.domain_length)
    volume_growth = _rel(max(final_volume - ancestor_volume, 0.0), ancestor_volume)
    ancestor_spread = _spread(ancestor_points, access.domain_length)
    final_spread = _spread(final_points, access.domain_length)
    ancestor_spread_over_final = ancestor_spread / max(final_spread, 1e-12)
    shape_stretch = _shape_stretch_penalty(ancestor_points, final_points, access.domain_length)

    g = min(max(omega_growth, 0.0), max(s_growth, 0.0))
    channels = {
        "B_growth": b_growth,
        "D_material": d_rel,
        "ancestor_spread_over_final": ancestor_spread_over_final,
        "shape_stretch_penalty": shape_stretch,
        "volume_growth": volume_growth,
    }
    r = max(channels.values())
    dominant = max(channels, key=channels.get)
    q = g / (r + args.eta)

    return {
        "t0": candidate.t0,
        "t1": candidate.t1,
        "lag_intervals": candidate.lag_intervals,
        "threshold_mode": candidate.threshold_mode,
        "component_id": candidate.component_id,
        "candidate_id": candidate.candidate_id,
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
        "counterexample_candidate_Q_gt_1": bool(g > 0.05 and q > 1.0),
        "counterexample_candidate_R_lt_0p1G": bool(g > 0.05 and r < 0.1 * g),
        "final_score": candidate.final_score,
        "final_omega_max_catalog": candidate.final_omega_max,
        "final_Splus_catalog": candidate.final_splus,
    }


def _shape_stretch_penalty(ancestor_points: np.ndarray, final_points: np.ndarray, domain_length: float) -> float:
    if len(ancestor_points) < 4 or len(final_points) < 4:
        return 0.0
    cond0 = _shape_condition(ancestor_points, domain_length)
    cond1 = _shape_condition(final_points, domain_length)
    return max(cond1 / max(cond0, 1e-12) - 1.0, 0.0)


def _shape_condition(points: np.ndarray, domain_length: float) -> float:
    unwrapped = _unwrap(points, domain_length)
    cov = np.cov(unwrapped.T)
    evals = np.linalg.eigvalsh(cov)
    evals = np.clip(evals, 1e-12, None)
    return float(evals[-1] / evals[0])


def _spread(points: np.ndarray, domain_length: float) -> float:
    if len(points) == 0:
        return 0.0
    unwrapped = _unwrap(points, domain_length)
    center = np.mean(unwrapped, axis=0)
    return float(np.sqrt(np.mean(np.sum((unwrapped - center.reshape(1, 3)) ** 2, axis=1))))


def _unwrap(points: np.ndarray, domain_length: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    base = points[0]
    return base.reshape(1, 3) + ((points - base.reshape(1, 3) + 0.5 * domain_length) % domain_length - 0.5 * domain_length)


def _final_score(splus: float, omega_max: float, n: int) -> float:
    return float(np.log1p(max(splus, 0.0)) * np.log1p(max(omega_max, 0.0)) * np.sqrt(max(n, 1)))


def _catalog(candidates: list[Candidate], args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for c in candidates:
        rows.append(
            {
                "candidate_id": c.candidate_id,
                "time_index": c.time_index,
                "t0": c.t0,
                "t1": c.t1,
                "lag_intervals": c.lag_intervals,
                "threshold_mode": c.threshold_mode,
                "component_id": c.component_id,
                "num_particles": int(len(c.component_indices)),
                "final_score": c.final_score,
                "final_omega_max": c.final_omega_max,
                "final_Splus": c.final_splus,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("final_score", ascending=False).reset_index(drop=True)
        out["selected_for_ancestry"] = False
        out.loc[out.index[: max(int(args.max_candidates), 0)], "selected_for_ancestry"] = True
    return out


if __name__ == "__main__":
    main()
