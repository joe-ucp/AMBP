"""Replay packing-weighted identity renewal on real promoted tube-family artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_RESULTS = Path("benchmarks/jhtdb_response_family/results")
DOMAIN_LENGTH = 2.0 * np.pi


def effective_counts(masses: np.ndarray, *, eps: float = 1e-12) -> dict[str, float]:
    masses = np.asarray(masses, dtype=float)
    masses[~np.isfinite(masses)] = 0.0
    masses = np.clip(masses, 0.0, None)
    total = float(np.sum(masses))
    if total <= eps:
        return {
            "N_eff_2": 0.0,
            "N_eff_half": 0.0,
            "H_pack_2": float("-inf"),
            "H_pack_half": float("-inf"),
        }
    p = masses / (total + eps)
    neff2 = float((total * total) / (float(np.sum(masses * masses)) + eps))
    neffhalf = float(np.square(float(np.sum(np.sqrt(np.clip(p, 0.0, None))))))
    return {
        "N_eff_2": neff2,
        "N_eff_half": neffhalf,
        "H_pack_2": float(np.log(max(neff2, eps))),
        "H_pack_half": float(np.log(max(neffhalf, eps))),
    }


def closure_ledger_sum(
    *,
    spread: float,
    shape: float,
    renew_total: float,
    tail: float,
    deactivation: float,
) -> float:
    return spread + shape + renew_total + tail + deactivation


def unclassified_fraction(
    pack_star: float,
    ledger_sum: float,
    *,
    c_pack: float = 1.0,
    eps: float = 1e-12,
) -> float:
    if pack_star <= eps:
        return 0.0
    excess = max(0.0, pack_star - c_pack * ledger_sum)
    return float(excess / (pack_star + eps))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membership-csv", type=Path, action="append", required=True)
    parser.add_argument("--overlap-summary-csv", type=Path, action="append", default=[])
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_RESULTS / "real_packing_weighted_renewal_audit_summary.csv",
    )
    parser.add_argument("--dx", type=float, default=DOMAIN_LENGTH / 1024.0)
    parser.add_argument("--voxel-dx", type=float, default=0.0, help="Voxel size in dx units; 0 uses tube_radius_dx.")
    parser.add_argument("--n0", type=float, default=1.0)
    parser.add_argument("--h-steps", type=int, default=1)
    parser.add_argument("--unclassified-tolerance", type=float, default=0.1)
    parser.add_argument("--c-pack", type=float, default=1.0)
    args = parser.parse_args()

    overlap_maps = _overlap_summary_maps(args.overlap_summary_csv)
    rows: list[dict[str, Any]] = []
    for membership_csv in args.membership_csv:
        membership = pd.read_csv(membership_csv)
        overlap_map = overlap_maps.get(_membership_stem(membership_csv), {})
        rows.extend(
            audit_membership_table(
                membership,
                membership_csv=membership_csv,
                overlap_by_candidate=overlap_map,
                dx=float(args.dx),
                voxel_dx=float(args.voxel_dx),
                n0=float(args.n0),
                h_steps=int(args.h_steps),
                unclassified_tolerance=float(args.unclassified_tolerance),
                c_pack=float(args.c_pack),
            )
        )

    out = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print("real_packing_weighted_renewal_audit", flush=True)
    print(f"rows={len(out)} output={args.output_csv}", flush=True)
    if not out.empty:
        print(out.to_string(index=False), flush=True)


def _membership_stem(path: Path) -> str:
    return path.stem


def _overlap_summary_maps(paths: list[Path]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty or "candidate_id" not in df.columns or "R_pack" not in df.columns:
            continue
        stem = path.stem
        if stem.startswith("tube_pair_overlap_audit_"):
            stem = stem.replace("tube_pair_overlap_audit_", "promoted_tube_family_membership_")
        if stem.endswith("_summary"):
            stem = stem[: -len("_summary")]
        out[stem] = {str(row["candidate_id"]): float(row["R_pack"]) for _, row in df.iterrows()}
    return out


def audit_membership_table(
    membership: pd.DataFrame,
    *,
    membership_csv: Path,
    overlap_by_candidate: dict[str, float],
    dx: float,
    voxel_dx: float,
    n0: float,
    h_steps: int,
    unclassified_tolerance: float,
    c_pack: float,
) -> list[dict[str, Any]]:
    if membership.empty:
        return []

    data = membership.copy()
    if "active_flag" in data.columns:
        data = data[data["active_flag"].astype(str).str.lower().isin({"true", "1"})].copy()
    data["source_weight"] = pd.to_numeric(data["source_weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    data = data[data["source_weight"] > 0.0].copy()
    if data.empty:
        return []

    radius_dx = float(data["tube_radius_dx"].dropna().iloc[0]) if "tube_radius_dx" in data.columns else 0.1
    voxel_size = (voxel_dx if voxel_dx > 0.0 else radius_dx) * dx

    rows: list[dict[str, Any]] = []
    for candidate_id, candidate in data.groupby(data["candidate_id"].astype(str), sort=False):
        tubes = sorted(candidate["tube_id"].astype(str).unique())
        cell_map, tube_to_idx = _build_cell_map(candidate, tubes=tubes, voxel_size=voxel_size)
        if not cell_map:
            continue

        ledgers = _compute_voxel_ledgers(cell_map, n_tubes=len(tubes), n0=n0, h=h_steps)
        tube_masses = candidate.groupby(candidate["tube_id"].astype(str))["source_weight"].sum()
        masses = np.asarray([float(tube_masses.get(t, 0.0)) for t in tubes], dtype=float)
        eff = effective_counts(masses)

        pack_overlap = float(overlap_by_candidate.get(str(candidate_id), np.nan))
        pack_star = ledgers["R_pack_voxel"] if np.isnan(pack_overlap) else pack_overlap

        spread_star = _r_spread_proxy(candidate)
        shape_star = 0.0
        tail_star = 0.0
        deact_star = ledgers["R_D"]

        renew_total = ledgers["R_renew_mult"] + ledgers["R_renew_pack_id"]
        ledger_sum = closure_ledger_sum(
            spread=spread_star,
            shape=shape_star,
            renew_total=renew_total,
            tail=tail_star,
            deactivation=deact_star,
        )
        unclassified = unclassified_fraction(pack_star, ledger_sum, c_pack=c_pack)
        door1b_status = _door1b_status(
            r_pack=pack_star,
            renew_pack_id=ledgers["R_renew_pack_id"],
            unclassified=unclassified,
            tolerance=unclassified_tolerance,
        )

        rows.append(
            {
                "membership_csv": str(membership_csv),
                "candidate_id": str(candidate_id),
                "radius_dx": radius_dx,
                "num_tubes": len(tubes),
                "R_pack_voxel": ledgers["R_pack_voxel"],
                "R_pack_overlap_audit": pack_overlap,
                "R_pack": pack_star,
                "N_eff_2": eff["N_eff_2"],
                "N_eff_half": eff["N_eff_half"],
                "R_renew_mult": ledgers["R_renew_mult"],
                "R_renew_id": ledgers["R_renew_id"],
                "R_renew_pack_id": ledgers["R_renew_pack_id"],
                "R_renew_total": renew_total,
                "R_spread": spread_star,
                "R_shape": shape_star,
                "R_D": deact_star,
                "R_tail": tail_star,
                "ledger_sum": ledger_sum,
                "unclassified_fraction": unclassified,
                "door1b_status": door1b_status,
            }
        )
    return rows


def _build_cell_map(
    candidate: pd.DataFrame,
    *,
    tubes: list[str],
    voxel_size: float,
) -> tuple[dict[tuple[int, tuple[int, int, int]], dict[int, float]], dict[str, int]]:
    tube_to_idx = {tube_id: idx for idx, tube_id in enumerate(tubes)}
    cell_map: dict[tuple[int, tuple[int, int, int]], dict[int, float]] = {}
    for _, row in candidate.iterrows():
        t = int(row["time_index"])
        tube_idx = tube_to_idx[str(row["tube_id"])]
        voxel = _voxel_key(float(row["x"]), float(row["y"]), float(row["z"]), voxel_size=voxel_size)
        key = (t, voxel)
        cell = cell_map.setdefault(key, {})
        cell[tube_idx] = cell.get(tube_idx, 0.0) + float(row["source_weight"])
    return cell_map, tube_to_idx


def _compute_voxel_ledgers(
    cell_map: dict[tuple[int, tuple[int, int, int]], dict[int, float]],
    *,
    n_tubes: int,
    n0: float,
    h: int,
    eps: float = 1e-12,
) -> dict[str, float]:
    total_source = float(sum(sum(weights.values()) for weights in cell_map.values()))
    if total_source <= eps:
        return {
            "R_pack_voxel": 0.0,
            "R_renew_mult": 0.0,
            "R_renew_id": 0.0,
            "R_renew_pack_id": 0.0,
            "R_D": 0.0,
        }

    pack_numer = 0.0
    renew_mult_numer = 0.0
    renew_id_numer = 0.0
    renew_pack_id_numer = 0.0
    deact_numer = 0.0
    deact_denom = 0.0

    by_voxel: dict[tuple[int, int, int], dict[int, dict[int, float]]] = {}
    for (time_index, voxel), weights in cell_map.items():
        by_voxel.setdefault(voxel, {})[time_index] = weights

    for voxel_times in by_voxel.values():
        sorted_times = sorted(voxel_times.keys())
        time_to_idx = {t: i for i, t in enumerate(sorted_times)}
        for t0 in sorted_times:
            weights0 = voxel_times[t0]
            sigma0 = float(sum(weights0.values()))
            present0 = _present_tubes(weights0)
            m0 = float(len(present0))
            pack_numer += sigma0 * max(m0 - n0, 0.0)

            target = t0 + h
            if target not in time_to_idx:
                continue
            weights1 = voxel_times[target]
            sigma1 = float(sum(weights1.values()))
            present1 = _present_tubes(weights1)
            m1 = float(len(present1))

            p0 = _identity_distribution(present0, n_tubes)
            p1 = _identity_distribution(present1, n_tubes)
            tv = 0.5 * float(np.sum(np.abs(p1 - p0)))

            renew_mult_numer += sigma0 * abs(m1 - m0)
            renew_id_numer += sigma0 * tv
            renew_pack_id_numer += sigma0 * max(m0 - n0, 0.0) * tv

            if m0 > n0:
                sigma_pers = min(sigma0, sigma1)
                deact_numer += sigma_pers
                deact_denom += sigma0

    r_d = 0.0 if deact_denom <= eps else max(0.0, 1.0 - deact_numer / (deact_denom + eps))
    return {
        "R_pack_voxel": pack_numer / (total_source + eps),
        "R_renew_mult": renew_mult_numer / (total_source + eps),
        "R_renew_id": renew_id_numer / (total_source + eps),
        "R_renew_pack_id": renew_pack_id_numer / (total_source + eps),
        "R_D": r_d,
    }


def _present_tubes(weights: dict[int, float]) -> list[int]:
    return [tube_idx for tube_idx, weight in weights.items() if weight > 0.0]


def _identity_distribution(present: list[int], n_tubes: int) -> np.ndarray:
    p = np.zeros(n_tubes, dtype=float)
    if not present:
        return p
    share = 1.0 / float(len(present))
    for idx in present:
        p[idx] = share
    return p


def _r_spread_proxy(candidate: pd.DataFrame) -> float:
    tubes = sorted(candidate["tube_id"].astype(str).unique())
    times = sorted(int(v) for v in candidate["time_index"].astype(int).unique())
    if len(times) < 2:
        return 0.0
    start = times[0]
    end = times[-1]

    def diam(points: np.ndarray) -> float:
        if len(points) <= 1:
            return 0.0
        d = 0.0
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                d = max(d, float(np.linalg.norm(points[i] - points[j])))
        return d

    seeds: list[np.ndarray] = []
    finals: list[np.ndarray] = []
    for tube_id in tubes:
        sub = candidate[candidate["tube_id"].astype(str).eq(tube_id)]
        start_pts = sub[sub["time_index"].astype(int).eq(start)][["x", "y", "z"]].to_numpy(dtype=float)
        end_pts = sub[sub["time_index"].astype(int).eq(end)][["x", "y", "z"]].to_numpy(dtype=float)
        if len(start_pts):
            seeds.append(start_pts.mean(axis=0))
        if len(end_pts):
            finals.append(end_pts.mean(axis=0))
    if not seeds or not finals:
        return 0.0
    d_in = diam(np.asarray(seeds))
    d_out = diam(np.asarray(finals))
    if d_out <= 0.0:
        return 0.0
    return float(max(0.0, np.log((d_in + 1.0) / (d_out + 1.0))))


def _door1b_status(
    *,
    r_pack: float,
    renew_pack_id: float,
    unclassified: float,
    tolerance: float,
) -> str:
    if r_pack <= 1e-12:
        return "no_packing_to_charge"
    if unclassified <= tolerance:
        return "packing_charged"
    if renew_pack_id > 0.0 and unclassified > tolerance:
        return "renewal_undercharged"
    return "uncharged_packing"


def _voxel_key(x: float, y: float, z: float, *, voxel_size: float) -> tuple[int, int, int]:
    coords = np.asarray([x, y, z], dtype=float) % DOMAIN_LENGTH
    return tuple(int(np.floor(c / voxel_size)) for c in coords)


if __name__ == "__main__":
    main()
