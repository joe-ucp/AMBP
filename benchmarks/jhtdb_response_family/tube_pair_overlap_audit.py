"""Audit source-weighted pairwise overlap in promoted tube-family tables."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_RESULTS = Path("benchmarks/jhtdb_response_family/results")
DOMAIN_LENGTH = 2.0 * np.pi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membership-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_RESULTS / "tube_pair_overlap_audit_summary.csv")
    parser.add_argument("--pair-output-csv", type=Path, default=None)
    parser.add_argument("--dx", type=float, default=DOMAIN_LENGTH / 1024.0)
    parser.add_argument("--overlap-radius-dx", type=float, default=0.0)
    parser.add_argument("--significance-tau", type=float, default=1e-3)
    parser.add_argument("--persistence-threshold", type=float, default=0.6)
    parser.add_argument("--deactivation-ratio", type=float, default=0.5)
    args = parser.parse_args()

    membership = pd.read_csv(args.membership_csv)
    pair_rows = build_pair_rows(
        membership,
        dx=float(args.dx),
        overlap_radius_dx=float(args.overlap_radius_dx),
        significance_tau=float(args.significance_tau),
        persistence_threshold=float(args.persistence_threshold),
        deactivation_ratio=float(args.deactivation_ratio),
    )
    summary = build_summary_rows(pair_rows, membership)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    pair_output = args.pair_output_csv or args.output_csv.with_name(f"{args.output_csv.stem}_pairs.csv")
    pair_output.parent.mkdir(parents=True, exist_ok=True)
    pair_rows.to_csv(pair_output, index=False)

    print("tube_pair_overlap_audit", flush=True)
    print(f"rows={len(summary)} pairs={len(pair_rows)}", flush=True)
    print(f"output={args.output_csv}", flush=True)
    print(f"pair_output={pair_output}", flush=True)


def build_pair_rows(
    membership: pd.DataFrame,
    *,
    dx: float,
    overlap_radius_dx: float,
    significance_tau: float,
    persistence_threshold: float,
    deactivation_ratio: float,
) -> pd.DataFrame:
    if membership.empty:
        return pd.DataFrame(columns=_pair_columns())

    data = membership.copy()
    if "active_flag" in data.columns:
        data = data[data["active_flag"].astype(str).str.lower().isin({"true", "1"})].copy()
    if data.empty:
        return pd.DataFrame(columns=_pair_columns())

    radius_dx = _radius_dx(data, fallback=overlap_radius_dx)
    overlap_radius = (overlap_radius_dx if overlap_radius_dx > 0.0 else 2.0 * radius_dx) * dx
    rows: list[dict[str, Any]] = []

    for candidate_id, candidate in data.groupby(data["candidate_id"].astype(str), sort=False):
        tubes = sorted(candidate["tube_id"].astype(str).unique())
        tube_mass = candidate.groupby(candidate["tube_id"].astype(str))["source_weight"].sum().to_dict()
        centroid = _tube_time_centroids(candidate)
        time_count = int(candidate["time_index"].nunique())
        for tube_i, tube_j in itertools.combinations(tubes, 2):
            pair_overlap, overlap_times = _pair_overlap(candidate, tube_i, tube_j, overlap_radius)
            mi = float(tube_mass.get(tube_i, 0.0))
            mj = float(tube_mass.get(tube_j, 0.0))
            scale = float(np.sqrt(max(mi, 0.0) * max(mj, 0.0)))
            k_ij = pair_overlap / scale if scale > 0.0 else 0.0
            significant = pair_overlap > float(significance_tau) * scale if scale > 0.0 else False
            persistence = len(overlap_times) / max(time_count, 1)
            charge = _pair_charge(
                candidate,
                tube_i,
                tube_j,
                centroid,
                overlap_times,
                overlap_radius,
                persistence,
                persistence_threshold,
                deactivation_ratio,
            )
            if not significant:
                charge = "none"
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "tube_i": tube_i,
                    "tube_j": tube_j,
                    "O_ij": pair_overlap,
                    "K_ij": k_ij,
                    "source_mass_i": mi,
                    "source_mass_j": mj,
                    "significant": bool(significant),
                    "overlap_time_count": len(overlap_times),
                    "overlap_persistence": persistence,
                    "pair_charge": charge,
                }
            )
    return pd.DataFrame(rows, columns=_pair_columns())


def build_summary_rows(pair_rows: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate_id",
        "num_tubes",
        "num_significant_pairs",
        "max_weighted_degree",
        "source_weighted_degree_mean",
        "redundant_pair_fraction",
        "spread_pair_fraction",
        "shape_pair_fraction",
        "renewal_pair_fraction",
        "tail_pair_fraction",
        "deactivation_pair_fraction",
        "unclassified_pair_fraction",
        "R_pack",
        "dominant_pair_charge",
    ]
    if membership.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for candidate_id, candidate in membership.groupby(membership["candidate_id"].astype(str), sort=False):
        candidate_pairs = pair_rows[pair_rows["candidate_id"].astype(str).eq(str(candidate_id))].copy()
        significant = candidate_pairs[candidate_pairs["significant"].astype(bool)].copy()
        total_source = float(candidate["source_weight"].clip(lower=0.0).sum())
        r_pack = float(significant["O_ij"].sum()) / total_source if total_source > 0.0 else 0.0
        charge_mass = significant.groupby("pair_charge")["O_ij"].sum() if not significant.empty else pd.Series(dtype=float)
        dominant = str(charge_mass.idxmax()) if not charge_mass.empty and charge_mass.max() > 0.0 else "none"
        degree = _weighted_degree(significant)
        degree_values = list(degree.values())
        denom = max(len(significant), 1)
        rows.append(
            {
                "candidate_id": str(candidate_id),
                "num_tubes": int(candidate["tube_id"].nunique()),
                "num_significant_pairs": int(len(significant)),
                "max_weighted_degree": max(degree_values, default=0.0),
                "source_weighted_degree_mean": float(np.mean(degree_values)) if degree_values else 0.0,
                "redundant_pair_fraction": _charge_fraction(significant, "redundant", denom),
                "spread_pair_fraction": _charge_fraction(significant, "spread", denom),
                "shape_pair_fraction": _charge_fraction(significant, "shape", denom),
                "renewal_pair_fraction": _charge_fraction(significant, "renewal", denom),
                "tail_pair_fraction": _charge_fraction(significant, "tail", denom),
                "deactivation_pair_fraction": _charge_fraction(significant, "deactivation", denom),
                "unclassified_pair_fraction": _charge_fraction(significant, "unclassified", denom),
                "R_pack": r_pack,
                "dominant_pair_charge": dominant,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _pair_overlap(candidate: pd.DataFrame, tube_i: str, tube_j: str, overlap_radius: float) -> tuple[float, set[int]]:
    overlap = 0.0
    overlap_times: set[int] = set()
    for time_index, at_time in candidate.groupby("time_index", sort=False):
        left = at_time[at_time["tube_id"].astype(str).eq(tube_i)]
        right = at_time[at_time["tube_id"].astype(str).eq(tube_j)]
        if left.empty or right.empty:
            continue
        left_xyz = left[["x", "y", "z"]].to_numpy(dtype=float)
        right_xyz = right[["x", "y", "z"]].to_numpy(dtype=float)
        distances = np.linalg.norm(left_xyz[:, None, :] - right_xyz[None, :, :], axis=2)
        hits = distances <= overlap_radius
        if not hits.any():
            continue
        left_w = left["source_weight"].clip(lower=0.0).to_numpy(dtype=float)
        right_w = right["source_weight"].clip(lower=0.0).to_numpy(dtype=float)
        weights = np.minimum(left_w[:, None], right_w[None, :])
        overlap += float(weights[hits].sum())
        overlap_times.add(int(time_index))
    return overlap, overlap_times


def _pair_charge(
    candidate: pd.DataFrame,
    tube_i: str,
    tube_j: str,
    centroid: dict[tuple[str, int], np.ndarray],
    overlap_times: set[int],
    overlap_radius: float,
    persistence: float,
    persistence_threshold: float,
    deactivation_ratio: float,
) -> str:
    if not overlap_times:
        return "unclassified"
    if persistence < persistence_threshold:
        return "renewal"
    if _source_decay(candidate, tube_i, tube_j) <= deactivation_ratio:
        return "deactivation"

    times = sorted(candidate["time_index"].astype(int).unique())
    start = times[0]
    end = times[-1]
    start_i = centroid.get((tube_i, start))
    start_j = centroid.get((tube_j, start))
    end_i = centroid.get((tube_i, end))
    end_j = centroid.get((tube_j, end))
    if start_i is None or start_j is None or end_i is None or end_j is None:
        return "unclassified"
    start_distance = float(np.linalg.norm(start_i - start_j))
    end_distance = float(np.linalg.norm(end_i - end_j))
    close = 2.0 * overlap_radius
    if start_distance <= close and end_distance <= close:
        return "redundant"
    if start_distance > close and end_distance <= close:
        return "spread"
    if start_distance <= close and end_distance > close:
        return "shape"
    return "unclassified"


def _tube_time_centroids(candidate: pd.DataFrame) -> dict[tuple[str, int], np.ndarray]:
    centroids: dict[tuple[str, int], np.ndarray] = {}
    grouped = candidate.groupby([candidate["tube_id"].astype(str), candidate["time_index"].astype(int)], sort=False)
    for (tube_id, time_index), rows in grouped:
        centroids[(str(tube_id), int(time_index))] = rows[["x", "y", "z"]].mean().to_numpy(dtype=float)
    return centroids


def _source_decay(candidate: pd.DataFrame, tube_i: str, tube_j: str) -> float:
    times = sorted(candidate["time_index"].astype(int).unique())
    if len(times) < 2:
        return 1.0
    start = candidate[candidate["time_index"].astype(int).eq(times[0])]
    end = candidate[candidate["time_index"].astype(int).eq(times[-1])]
    start_mass = float(start[start["tube_id"].astype(str).isin([tube_i, tube_j])]["source_weight"].clip(lower=0.0).sum())
    end_mass = float(end[end["tube_id"].astype(str).isin([tube_i, tube_j])]["source_weight"].clip(lower=0.0).sum())
    return end_mass / start_mass if start_mass > 0.0 else 1.0


def _radius_dx(data: pd.DataFrame, *, fallback: float) -> float:
    if fallback > 0.0:
        return fallback
    if "tube_radius_dx" in data.columns and not data["tube_radius_dx"].dropna().empty:
        return float(data["tube_radius_dx"].dropna().iloc[0])
    return 0.1


def _weighted_degree(significant: pd.DataFrame) -> dict[str, float]:
    degree: dict[str, float] = {}
    for _, row in significant.iterrows():
        weight = float(row["K_ij"])
        degree[str(row["tube_i"])] = degree.get(str(row["tube_i"]), 0.0) + weight
        degree[str(row["tube_j"])] = degree.get(str(row["tube_j"]), 0.0) + weight
    return degree


def _charge_fraction(significant: pd.DataFrame, charge: str, denom: int) -> float:
    if significant.empty:
        return 0.0
    return float(significant["pair_charge"].astype(str).eq(charge).sum()) / float(denom)


def _pair_columns() -> list[str]:
    return [
        "candidate_id",
        "tube_i",
        "tube_j",
        "O_ij",
        "K_ij",
        "source_mass_i",
        "source_mass_j",
        "significant",
        "overlap_time_count",
        "overlap_persistence",
        "pair_charge",
    ]


if __name__ == "__main__":
    main()
