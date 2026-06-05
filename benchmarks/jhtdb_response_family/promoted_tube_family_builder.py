"""Build promoted tube-family membership tables for Top2/Top3 candidates.

The output is the canonical particle-level object needed before pairwise tube
overlap can be audited.  It reconstructs each candidate's final skeleton
component, promotes each skeleton center to a finite-radius material tube,
advects the tube particles through the window, and records source weights.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import csv
import re
import json
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.jhtdb_response_family.cache import cached_query_points  # noqa: E402
from benchmarks.jhtdb_response_family.config import JHTDBAccessConfig  # noqa: E402
from benchmarks.jhtdb_response_family.global_hotspot_ancestry_scan import _components, _threshold_mask  # noqa: E402
from benchmarks.jhtdb_response_family.jhtdb_client import make_flow_client  # noqa: E402
from benchmarks.jhtdb_response_family.material_image_runaway_audit import _advect_cloud, _query_patch, _source_center  # noqa: E402
from benchmarks.jhtdb_response_family.relaxed_patch_stretch_budget_audit import _point_metrics  # noqa: E402
from benchmarks.jhtdb_response_family.sampling import patch_points  # noqa: E402


DEFAULT_RESULTS = Path("benchmarks/jhtdb_response_family/results")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-canonical", action="store_true", help="Scan results/ and emit canonical registry + membership CSVs (conservative)")
    parser.add_argument(
        "--scan",
        nargs=3,
        action="append",
        metavar=("FAMILY", "RUN_DIR", "CANDIDATE_CSV"),
        required=False,
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_RESULTS / "targeted_dichotomy_audit_top2_top3_summary.csv",
    )
    parser.add_argument("--candidate-id", action="append", default=[])
    parser.add_argument("--max-candidates", type=int, default=0, help="Optional cap after filtering; 0 means all.")
    parser.add_argument("--parent-core-id", default="0:0")
    parser.add_argument("--dataset", default="isotropic1024coarse")
    parser.add_argument("--source-mode", choices=("synthetic", "soap", "pyjhtdb", "auto"), default="pyjhtdb")
    parser.add_argument("--patch-points-per-axis", type=int, default=49)
    parser.add_argument("--patch-spacing", type=float, default=2.0 * np.pi / 1024.0 * 4.0)
    parser.add_argument("--tube-radius-dx", type=float, default=0.1)
    parser.add_argument("--points-per-tube", type=int, default=4)
    parser.add_argument("--time-samples", type=int, default=5)
    parser.add_argument("--max-tubes-per-candidate", type=int, default=0, help="Optional deterministic cap; 0 means all.")
    parser.add_argument("--component-id-override", type=int, default=None, help="Smoke-test helper; do not use for real rows.")
    parser.add_argument("--substep-dt", type=float, default=0.005)
    parser.add_argument("--max-points-per-request", type=int, default=2048)
    parser.add_argument("--rng-seed", type=int, default=1729)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--write-parts", action="store_true", help="Write one resumable part per candidate tube.")
    parser.add_argument("--parts-dir", type=Path, default=None)
    parser.add_argument("--manifest-csv", type=Path, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--combine-only", action="store_true", help="Combine completed manifest parts into --output-csv.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_RESULTS / "promoted_tube_family_membership_top2_top3.csv",
    )
    args = parser.parse_args()
    args.parts_dir = args.parts_dir or args.output_csv.with_name(f"{args.output_csv.stem}_parts")
    args.manifest_csv = args.manifest_csv or args.output_csv.with_name(f"{args.output_csv.stem}_manifest.csv")

    if args.combine_only:
        _combine_completed_parts(args.manifest_csv, args.output_csv)
        return
    if args.emit_canonical:
        _emit_canonical_registry()
        return
    if not args.scan:
        raise SystemExit("--scan is required unless --combine-only is used")

    summary = pd.read_csv(args.summary_csv)
    if args.candidate_id:
        keep = set(str(v) for v in args.candidate_id)
        summary = summary[summary["candidate_id"].astype(str).isin(keep)].copy()
    if int(args.max_candidates) > 0:
        summary = summary.head(int(args.max_candidates)).copy()
    scans = {family: (Path(run_dir), Path(candidate_csv)) for family, run_dir, candidate_csv in args.scan}
    access = JHTDBAccessConfig(
        dataset=args.dataset,
        source_mode=args.source_mode,
        velocity_sinterp="Lag4",
        gradient_sinterp="FD4Lag4",
        tinterp="PCHIP",
        max_points_per_request=args.max_points_per_request,
    )
    client = make_flow_client(access)
    _component_id.override = args.component_id_override

    if args.write_parts:
        _write_part_outputs(summary, scans, args, access, client)
        _combine_completed_parts(args.manifest_csv, args.output_csv)
        out = pd.read_csv(args.output_csv) if args.output_csv.exists() else pd.DataFrame()
    else:
        rows: list[dict[str, Any]] = []
        for _, candidate in summary.iterrows():
            rows.extend(_candidate_membership_rows(candidate, scans, args, access, client))
        out = pd.DataFrame(rows)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output_csv, index=False)
    print("promoted_tube_family_builder", flush=True)
    print(f"rows={len(out)} candidates={out['candidate_id'].nunique() if not out.empty else 0}", flush=True)
    print(f"output={args.output_csv}", flush=True)
    if args.write_parts:
        print(f"manifest={args.manifest_csv}", flush=True)
        print(f"parts_dir={args.parts_dir}", flush=True)


def _candidate_membership_rows(
    candidate: pd.Series,
    scans: dict[str, tuple[Path, Path]],
    args: argparse.Namespace,
    access: JHTDBAccessConfig,
    client,
) -> list[dict[str, Any]]:
    prepared = _prepare_candidate_family(candidate, scans, args, access, client)
    if prepared is None:
        return []
    family, final_points, ancestor_points, tube_index, material_index = prepared
    times = np.linspace(float(candidate["t0"]), float(candidate["t1"]), max(2, int(args.time_samples)))
    return _membership_rows_for_points(
        candidate,
        family,
        final_points,
        ancestor_points,
        tube_index,
        material_index,
        times,
        args,
        access,
        client,
    )


def _prepare_candidate_family(
    candidate: pd.Series,
    scans: dict[str, tuple[Path, Path]],
    args: argparse.Namespace,
    access: JHTDBAccessConfig,
    client,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    family = str(candidate["family"])
    if family not in scans:
        raise SystemExit(f"no --scan supplied for family {family}")
    print(f"prepare_load_starts {candidate['candidate_id']}", flush=True)
    run_dir, _candidate_csv = scans[family]
    starts = pd.read_csv(run_dir / "material_heat_age_starts.csv")
    center = _source_center(starts, args.parent_core_id)
    grid_points, _center_index = patch_points(
        center,
        patch_points_per_axis=int(args.patch_points_per_axis),
        patch_spacing=float(args.patch_spacing),
        domain_length=access.domain_length,
    )
    print(f"prepare_component_query {candidate['candidate_id']} n={len(grid_points)}", flush=True)
    skeleton_points = _candidate_skeleton_points(candidate, args, access, client, grid_points)
    print(f"prepare_component_done {candidate['candidate_id']} skeleton={len(skeleton_points)}", flush=True)
    if skeleton_points.size == 0:
        return None
    if int(args.max_tubes_per_candidate) > 0 and len(skeleton_points) > int(args.max_tubes_per_candidate):
        seed = _stable_seed(args.rng_seed, str(candidate["candidate_id"]), "tube_cap")
        rng = np.random.default_rng(seed)
        take = np.sort(rng.choice(len(skeleton_points), size=int(args.max_tubes_per_candidate), replace=False))
        skeleton_points = skeleton_points[take]

    final_points, tube_index, material_index = _tube_family_particles(
        skeleton_points,
        radius=float(args.tube_radius_dx) * float(args.patch_spacing),
        points_per_tube=int(args.points_per_tube),
        seed=_stable_seed(args.rng_seed, str(candidate["candidate_id"]), float(args.tube_radius_dx)),
        domain_length=access.domain_length,
    )
    print(f"prepare_backward_advect {candidate['candidate_id']} particles={len(final_points)}", flush=True)
    ancestor_points = _advect_cloud(
        final_points,
        t0=float(candidate["t1"]),
        t1=float(candidate["t0"]),
        args=args,
        access=access,
        client=client,
        window_id=f"promoted_tube_family:{candidate['candidate_id']}:backward:r={float(args.tube_radius_dx):g}",
    )
    finite = np.isfinite(ancestor_points).all(axis=1)
    final_points = final_points[finite] % access.domain_length
    ancestor_points = ancestor_points[finite] % access.domain_length
    tube_index = tube_index[finite]
    material_index = material_index[finite]
    print(f"prepare_backward_done {candidate['candidate_id']} retained={len(final_points)}", flush=True)
    return family, final_points, ancestor_points, tube_index, material_index


def _membership_rows_for_points(
    candidate: pd.Series,
    family: str,
    final_points: np.ndarray,
    ancestor_points: np.ndarray,
    tube_index: np.ndarray,
    material_index: np.ndarray,
    times: np.ndarray,
    args: argparse.Namespace,
    access: JHTDBAccessConfig,
    client,
) -> list[dict[str, Any]]:
    dt_weight = _sample_dt(times)
    rows: list[dict[str, Any]] = []
    for time_index, time in enumerate(times):
        if abs(float(time) - float(candidate["t0"])) < 1e-14:
            points = ancestor_points
        elif abs(float(time) - float(candidate["t1"])) < 1e-14:
            points = final_points
        else:
            points = _advect_cloud(
                ancestor_points,
                t0=float(candidate["t0"]),
                t1=float(time),
                args=args,
                access=access,
                client=client,
                window_id=(
                    f"promoted_tube_family:{candidate['candidate_id']}:"
                    f"sample={float(time):.12g}:r={float(args.tube_radius_dx):g}"
                ),
            )
        data = cached_query_points(
            client,
            access,
            time=float(time),
            points=np.asarray(points, dtype=float) % access.domain_length,
            role=(
                f"promoted_tube_family:{candidate['candidate_id']}:"
                f"sample={float(time):.12g}:r={float(args.tube_radius_dx):g}:n={len(points)}"
            ),
        )
        metrics = _point_metrics(data.gradient)
        lambda_plus = np.maximum(np.asarray(metrics["lambda"], dtype=float), 0.0)
        omega_norm = np.asarray(metrics["omega_norm"], dtype=float)
        sigma_plus = lambda_plus * omega_norm * omega_norm
        source_weight = sigma_plus * (float(args.patch_spacing) ** 3) * dt_weight[time_index]
        for local_id, point in enumerate(np.asarray(points, dtype=float) % access.domain_length):
            tube_local = int(tube_index[local_id])
            material_local = int(material_index[local_id])
            component_id = _component_id(candidate)
            rows.append(
                {
                    "candidate_id": str(candidate["candidate_id"]),
                    "family": family,
                    "tube_id": f"{candidate['candidate_id']}:r{float(args.tube_radius_dx):g}:tube{tube_local}",
                    "tube_radius_dx": float(args.tube_radius_dx),
                    "time_index": int(time_index),
                    "time": float(time),
                    "point_id": int(local_id),
                    "material_particle_id": f"{candidate['candidate_id']}:p{material_local}",
                    "material_label": f"{candidate['candidate_id']}:p{material_local}",
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2]),
                    "sigma_plus": float(sigma_plus[local_id]),
                    "omega_norm": float(omega_norm[local_id]),
                    "lambda_plus": float(lambda_plus[local_id]),
                    "source_weight": float(source_weight[local_id]),
                    "tube_membership": 1,
                    "ancestor_component": component_id,
                    "ancestor_id": f"{candidate['candidate_id']}:ancestor:c{component_id}",
                    "final_component": component_id,
                    "descendant_id": f"{candidate['candidate_id']}:final:c{component_id}",
                    "material_family_id": f"{candidate['candidate_id']}:r{float(args.tube_radius_dx):g}",
                    "active_flag": bool(lambda_plus[local_id] > 0.0),
                }
            )
    return rows


def _write_part_outputs(
    summary: pd.DataFrame,
    scans: dict[str, tuple[Path, Path]],
    args: argparse.Namespace,
    access: JHTDBAccessConfig,
    client,
) -> None:
    args.parts_dir.mkdir(parents=True, exist_ok=True)
    completed = _completed_manifest_keys(args.manifest_csv) if args.resume else set()
    next_part = _next_part_index(args.manifest_csv, args.parts_dir, args.output_csv.stem)
    for _, candidate in summary.iterrows():
        try:
            candidate_id = str(candidate["candidate_id"])
            prepare_key = _manifest_key(candidate_id, "__prepare__", args)
            prepare_started = _utc_now()
            _append_manifest_row(
                args.manifest_csv,
                {
                    "candidate_id": candidate_id,
                    "tube_id": "__prepare__",
                    "window": f"{float(candidate['t0']):.6g}_to_{float(candidate['t1']):.6g}",
                    "radius_dx": float(args.tube_radius_dx),
                    "num_points_expected": 0,
                    "num_points_written": 0,
                    "status": "running",
                    "cache_key": prepare_key,
                    "part_path": "",
                    "part_index": -1,
                    "started_at": prepare_started,
                    "finished_at": "",
                    "error": "",
                },
            )
            print(f"prepare_start {candidate_id}", flush=True)
            prepared = _prepare_candidate_family(candidate, scans, args, access, client)
            if prepared is None:
                continue
            _append_manifest_row(
                args.manifest_csv,
                {
                    "candidate_id": candidate_id,
                    "tube_id": "__prepare__",
                    "window": f"{float(candidate['t0']):.6g}_to_{float(candidate['t1']):.6g}",
                    "radius_dx": float(args.tube_radius_dx),
                    "num_points_expected": 0,
                    "num_points_written": 0,
                    "status": "prepared",
                    "cache_key": prepare_key,
                    "part_path": "",
                    "part_index": -1,
                    "started_at": prepare_started,
                    "finished_at": _utc_now(),
                    "error": "",
                },
            )
            print(f"prepare_done {candidate_id}", flush=True)
            family, final_points, ancestor_points, tube_index, material_index = prepared
            for tube_local in sorted(int(v) for v in np.unique(tube_index)):
                tube_id = f"{candidate['candidate_id']}:r{float(args.tube_radius_dx):g}:tube{tube_local}"
                manifest_key = _manifest_key(str(candidate["candidate_id"]), tube_id, args)
                if manifest_key in completed:
                    continue
                part_path = args.parts_dir / f"{args.output_csv.stem}_part_{next_part:03d}.csv"
                next_part += 1
                started_at = _utc_now()
                try:
                    mask = tube_index == tube_local
                    times = np.linspace(float(candidate["t0"]), float(candidate["t1"]), max(2, int(args.time_samples)))
                    rows = _membership_rows_for_points(
                        candidate,
                        family,
                        final_points[mask],
                        ancestor_points[mask],
                        tube_index[mask],
                        material_index[mask],
                        times,
                        args,
                        access,
                        client,
                    )
                    pd.DataFrame(rows).to_csv(part_path, index=False)
                    _append_manifest_row(
                        args.manifest_csv,
                        {
                            "candidate_id": str(candidate["candidate_id"]),
                            "tube_id": tube_id,
                            "window": f"{float(candidate['t0']):.6g}_to_{float(candidate['t1']):.6g}",
                            "radius_dx": float(args.tube_radius_dx),
                            "num_points_expected": int(np.sum(mask) * len(times)),
                            "num_points_written": int(len(rows)),
                            "status": "complete",
                            "cache_key": manifest_key,
                            "part_path": str(part_path),
                            "part_index": int(next_part - 1),
                            "started_at": started_at,
                            "finished_at": _utc_now(),
                            "error": "",
                        },
                    )
                    print(f"part_done {part_path} rows={len(rows)}", flush=True)
                except Exception as exc:
                    _append_manifest_row(
                        args.manifest_csv,
                        {
                            "candidate_id": str(candidate["candidate_id"]),
                            "tube_id": tube_id,
                            "window": f"{float(candidate['t0']):.6g}_to_{float(candidate['t1']):.6g}",
                            "radius_dx": float(args.tube_radius_dx),
                            "num_points_expected": int(np.sum(tube_index == tube_local) * int(args.time_samples)),
                            "num_points_written": 0,
                            "status": "error",
                            "cache_key": manifest_key,
                            "part_path": str(part_path),
                            "part_index": int(next_part - 1),
                            "started_at": started_at,
                            "finished_at": _utc_now(),
                            "error": repr(exc),
                        },
                    )
                    print(f"part_error {tube_id}: {exc!r}", flush=True)
        except Exception as exc:
            candidate_id = str(candidate.get("candidate_id", "unknown"))
            _append_manifest_row(
                args.manifest_csv,
                {
                    "candidate_id": candidate_id,
                    "tube_id": "",
                    "window": f"{float(candidate['t0']):.6g}_to_{float(candidate['t1']):.6g}" if "t0" in candidate else "",
                    "radius_dx": float(args.tube_radius_dx),
                    "num_points_expected": 0,
                    "num_points_written": 0,
                    "status": "error",
                    "cache_key": _manifest_key(candidate_id, "", args),
                    "part_path": "",
                    "part_index": -1,
                    "started_at": _utc_now(),
                    "finished_at": _utc_now(),
                    "error": repr(exc),
                },
            )
            print(f"candidate_error {candidate_id}: {exc!r}", flush=True)


def _completed_manifest_keys(manifest_csv: Path) -> set[str]:
    if not manifest_csv.exists():
        return set()
    manifest = pd.read_csv(manifest_csv)
    if manifest.empty or "cache_key" not in manifest.columns or "status" not in manifest.columns:
        return set()
    complete = manifest[manifest["status"].astype(str).eq("complete")]
    completed: set[str] = set()
    for _, row in complete.iterrows():
        tube_id = str(row.get("tube_id", ""))
        if tube_id == "__prepare__":
            completed.add(str(row["cache_key"]))
            continue
        part_path = str(row.get("part_path", ""))
        if not part_path:
            continue
        path = Path(part_path)
        if not path.exists():
            continue
        try:
            part = pd.read_csv(path, usecols=["tube_id"])
        except Exception:
            continue
        if tube_id in set(part["tube_id"].astype(str)):
            completed.add(str(row["cache_key"]))
    return completed


def _next_part_index(manifest_csv: Path, parts_dir: Path, output_stem: str) -> int:
    indices: list[int] = []
    if manifest_csv.exists():
        manifest = pd.read_csv(manifest_csv)
        if "part_index" in manifest.columns:
            indices.extend(int(v) for v in pd.to_numeric(manifest["part_index"], errors="coerce").dropna() if int(v) >= 0)
    for path in parts_dir.glob(f"{output_stem}_part_*.csv"):
        suffix = path.stem.rsplit("_part_", 1)[-1]
        if suffix.isdigit():
            indices.append(int(suffix))
    return max(indices, default=-1) + 1


def _append_manifest_row(manifest_csv: Path, row: dict[str, Any]) -> None:
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    frame.to_csv(manifest_csv, mode="a", header=not manifest_csv.exists(), index=False)


def _combine_completed_parts(manifest_csv: Path, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not manifest_csv.exists():
        pd.DataFrame().to_csv(output_csv, index=False)
        return
    manifest = pd.read_csv(manifest_csv)
    if manifest.empty:
        pd.DataFrame().to_csv(output_csv, index=False)
        return
    complete = manifest[manifest["status"].astype(str).eq("complete")].copy()
    if complete.empty:
        pd.DataFrame().to_csv(output_csv, index=False)
        return
    complete["finished_at_sort"] = complete["finished_at"].astype(str)
    complete = complete.sort_values("part_index")
    frames = []
    for part_path in complete["part_path"].astype(str).drop_duplicates():
        path = Path(part_path)
        if path.exists():
            frames.append(pd.read_csv(path))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    row_keys = [col for col in ("candidate_id", "tube_id", "time_index", "point_id") if col in out.columns]
    if row_keys:
        out = out.drop_duplicates(row_keys, keep="last")
    out.to_csv(output_csv, index=False)


def _manifest_key(candidate_id: str, tube_id: str, args: argparse.Namespace) -> str:
    data = "|".join(
        [
            candidate_id,
            tube_id,
            f"r={float(args.tube_radius_dx):.12g}",
            f"samples={int(args.time_samples)}",
            f"points={int(args.points_per_tube)}",
            f"patch={int(args.patch_points_per_axis)}",
        ]
    )
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:24]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _candidate_skeleton_points(candidate: pd.Series, args: argparse.Namespace, access: JHTDBAccessConfig, client, grid_points: np.ndarray) -> np.ndarray:
    shape = (int(args.patch_points_per_axis),) * 3
    snapshot = _query_patch(client, access, time=float(candidate["t1"]), points=grid_points, args=args)
    metrics = _point_metrics(snapshot.gradient)
    mask = _threshold_mask(
        str(candidate["candidate_id"]).split(":")[0],
        np.maximum(metrics["lambda"], 0.0),
        metrics["omega_norm"],
    )
    components = _components(mask, shape)
    component_id = _component_id(candidate)
    if component_id >= len(components):
        if args.component_id_override is not None:
            lambda_plus = np.maximum(metrics["lambda"], 0.0)
            take_count = max(1, int(args.max_tubes_per_candidate) if int(args.max_tubes_per_candidate) > 0 else 1)
            take = np.argsort(lambda_plus)[::-1][:take_count]
            return np.asarray(grid_points[take], dtype=float)
        raise SystemExit(f"component {component_id} missing for {candidate['candidate_id']} at t1={candidate['t1']}")
    return np.asarray(grid_points[components[component_id]], dtype=float)


def _component_id(candidate: pd.Series) -> int:
    override = getattr(_component_id, "override", None)
    if override is not None:
        return int(override)
    if "component_id" in candidate.index and pd.notna(candidate["component_id"]):
        return int(candidate["component_id"])
    for part in str(candidate["candidate_id"]).split(":"):
        if part.startswith("c") and part[1:].isdigit():
            return int(part[1:])
    raise ValueError(f"could not parse component id from {candidate['candidate_id']}")


def _tube_family_particles(
    centers: np.ndarray,
    *,
    radius: float,
    points_per_tube: int,
    seed: int,
    domain_length: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.asarray(centers, dtype=float)
    rng = np.random.default_rng(seed)
    all_points: list[np.ndarray] = []
    tube_ids: list[int] = []
    material_ids: list[int] = []
    for tube_id, center in enumerate(centers):
        if radius > 0.0:
            offsets = _ball_offsets(radius, max(1, int(points_per_tube)), rng)
            points = np.vstack([center.reshape(1, 3), center.reshape(1, 3) + offsets])
        else:
            points = center.reshape(1, 3)
        start = len(material_ids)
        count = len(points)
        all_points.append(points % domain_length)
        tube_ids.extend([tube_id] * count)
        material_ids.extend(range(start, start + count))
    if not all_points:
        return np.empty((0, 3)), np.asarray([], dtype=int), np.asarray([], dtype=int)
    return np.vstack(all_points), np.asarray(tube_ids, dtype=int), np.asarray(material_ids, dtype=int)


def _ball_offsets(radius: float, count: int, rng: np.random.Generator) -> np.ndarray:
    directions = rng.normal(size=(count, 3))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    radii = radius * rng.random(count) ** (1.0 / 3.0)
    return directions * radii.reshape(-1, 1)


def _sample_dt(times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    if len(times) == 1:
        return np.ones(1, dtype=float)
    weights = np.empty(len(times), dtype=float)
    weights[0] = 0.5 * (times[1] - times[0])
    weights[-1] = 0.5 * (times[-1] - times[-2])
    if len(times) > 2:
        weights[1:-1] = 0.5 * (times[2:] - times[:-2])
    return np.maximum(weights, 0.0)


def _stable_seed(*parts: object) -> int:
    data = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little") % (2**32)


# ---- Canonical registry emission (conservative merge of existing promoted membership files)
CANON_REGISTRY = DEFAULT_RESULTS / "promoted_tube_family_registry.csv"
CANON_MEMBERSHIP = DEFAULT_RESULTS / "promoted_tube_family_membership_canonical.csv"
CANON_TEX = DEFAULT_RESULTS / "promoted_tube_family_summary.tex"


def _discover_membership_files(results_dir: Path):
    return sorted(results_dir.glob("promoted_tube_family_membership*.csv"))


def _read_rows(path: Path):
    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception:
        return []


def _parse_radius_from_candidate(candidate_id: str, row: dict):
    if not candidate_id:
        return ''
    m = re.search(r'([0-9])p([0-9]+)dx', candidate_id)
    if m:
        return f"0.{m.group(2)}"
    m = re.search(r'r0p([0-9]+)', candidate_id)
    if m:
        return f"0.{m.group(1)}"
    return row.get('tube_radius_dx','') or row.get('radius_dx','') or ''


def _stable_family_id(candidate_id: str, r_over_dx: str, case: str, window: str) -> str:
    base = str(candidate_id or '')
    key = f"{base}::r={r_over_dx or ''}::case={case or ''}::w={window or ''}"
    return re.sub(r'\s+', '_', key)


def _normalize_member(row: dict, src: str):
    out = {k: '' for k in [
        'family_id','member_id','time_index','particle_x','particle_y','particle_z',
        'source_weight','is_core','is_shell','is_tail','is_boundary','tube_id',
        'pack_id','shell_label','ancestry_id','renewal_id','source_files'
    ]}
    candidate = row.get('candidate_id') or row.get('check_id') or row.get('family') or ''
    r = _parse_radius_from_candidate(candidate, row)
    case = row.get('case','')
    window = row.get('time') or row.get('window','')
    out['family_id'] = _stable_family_id(candidate, r, case, window)
    out['member_id'] = row.get('material_particle_id') or row.get('material_label') or row.get('particle_id') or row.get('member_id') or ''
    out['time_index'] = row.get('time_index') or row.get('time') or ''
    out['particle_x'] = row.get('x') or row.get('particle_x') or ''
    out['particle_y'] = row.get('y') or row.get('particle_y') or ''
    out['particle_z'] = row.get('z') or row.get('particle_z') or ''
    out['source_weight'] = row.get('source_weight') or row.get('sigma_plus') or ''
    out['is_core'] = row.get('is_core') or row.get('core') or ''
    out['is_shell'] = row.get('is_shell') or row.get('shell') or ''
    out['is_tail'] = row.get('is_tail') or row.get('tail') or ''
    out['is_boundary'] = row.get('is_boundary') or row.get('boundary') or ''
    out['tube_id'] = row.get('tube_id') or ''
    out['pack_id'] = row.get('pack_id') or ''
    out['shell_label'] = row.get('shell_label') or ''
    out['ancestry_id'] = row.get('ancestor_id') or row.get('ancestry_id') or ''
    out['renewal_id'] = row.get('renewal_id') or ''
    out['source_files'] = src
    return out


def _validate_family(members: list[dict]) -> list[str]:
    warnings = []
    ids = defaultdict(list)
    for m in members:
        ids[m.get('member_id','')].append(m)
    for mid, rows in ids.items():
        if mid == '':
            warnings.append('missing_member_id')
        elif len(rows) > 1:
            times = set(r.get('time_index','') for r in rows)
            if len(times) == 1:
                warnings.append(f'duplicate_member_id_no_time:{mid}')
    for m in members:
        sw = m.get('source_weight','')
        if sw:
            try:
                if float(sw) < 0:
                    warnings.append('negative_source_weight')
                    break
            except Exception:
                warnings.append('unparseable_source_weight')
                break
    return warnings


def _emit_canonical_registry():
    results_dir = DEFAULT_RESULTS
    results_dir.mkdir(parents=True, exist_ok=True)
    files = _discover_membership_files(results_dir)
    registry = {}
    canonical_members = []

    for f in files:
        fname = f.name
        rows = _read_rows(f)
        for r in rows:
            candidate = r.get('candidate_id') or r.get('check_id') or r.get('family') or ''
            r_over_dx = _parse_radius_from_candidate(candidate, r)
            case = r.get('case','')
            window = r.get('time') or r.get('window','')
            fid = _stable_family_id(candidate, r_over_dx, case, window)
            member = _normalize_member(r, src=fname)
            canonical_members.append(member)
            if fid not in registry:
                registry[fid] = {
                    'family_id': fid,
                    'candidate_id': candidate,
                    'r_over_dx': r_over_dx,
                    'case': case,
                    'window': window,
                    'status': 'direct',
                    'source_files': fname,
                    'missing_inputs': '',
                    'validation_warnings': '',
                    'member_count': 1,
                }
            else:
                entry = registry[fid]
                entry['source_files'] = ';'.join(sorted(set(entry['source_files'].split(';') + [fname])))
                entry['member_count'] = int(entry.get('member_count',0)) + 1

    # discover candidate tokens from results CSV filenames to create reconstructed placeholders
    all_csvs = list(results_dir.glob('*.csv'))
    cand_tokens = set()
    for f in all_csvs:
        m = re.search(r'(_|^)c([0-9]{1,4})(_|\.|$)', f.name)
        if m:
            cand_tokens.add(f'c{m.group(2)}')
    for cand in sorted(cand_tokens):
        if any(reg['candidate_id'] == cand for reg in registry.values()):
            continue
        fid = _stable_family_id(cand, '', '', '')
        registry[fid] = {
            'family_id': fid,
            'candidate_id': cand,
            'r_over_dx': '',
            'case': '',
            'window': '',
            'status': 'reconstructed',
            'source_files': '',
            'missing_inputs': 'members,coordinates',
            'validation_warnings': '',
            'member_count': 0,
        }

    # validation
    fam_map = defaultdict(list)
    for m in canonical_members:
        fam_map[m['family_id']].append(m)
    for fid, entry in registry.items():
        members = fam_map.get(fid, [])
        warnings = _validate_family(members)
        entry['validation_warnings'] = ';'.join(warnings)

    # write canonical membership and registry
    with open(CANON_MEMBERSHIP, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'family_id','member_id','time_index','particle_x','particle_y','particle_z',
            'source_weight','is_core','is_shell','is_tail','is_boundary','tube_id',
            'pack_id','shell_label','ancestry_id','renewal_id','source_files'
        ])
        writer.writeheader()
        for m in canonical_members:
            writer.writerow(m)

    with open(CANON_REGISTRY, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'family_id','candidate_id','r_over_dx','case','window','status',
            'source_files','missing_inputs','validation_warnings','member_count'
        ])
        writer.writeheader()
        for fid, reg in sorted(registry.items()):
            writer.writerow(reg)

    counts = defaultdict(int)
    for reg in registry.values():
        counts[reg['status']] += 1
    with open(CANON_TEX, 'w') as f:
        f.write('% Auto-generated promoted tube family registry summary\n')
        f.write('\\begin{tabular}{lr}\n')
        f.write('Status & Count\\\\\n')
        f.write('\\hline\n')
        for k in sorted(counts.keys()):
            f.write(f"{k} & {counts[k]}\\\\\n")
        f.write('\\end{tabular}\n')

    print('Wrote', CANON_REGISTRY, CANON_MEMBERSHIP, CANON_TEX)


if __name__ == "__main__":
    main()
