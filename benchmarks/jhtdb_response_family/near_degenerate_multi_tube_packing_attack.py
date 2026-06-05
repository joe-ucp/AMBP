"""Near-degenerate multi-tube packing attack on cached promoted families.

This is an offline stress harness for the real JHTDB artifacts already cached
under ``benchmarks/jhtdb_response_family/results``.  It does not re-query DNS.

The attack takes each promoted natural tube family, replaces every natural tube
by M nearly identical labeled sub-tubes, applies a conservative physical
quotient that merges sub-tubes with the same parent ancestry and identical
material support, and reports whether any non-redundant packing remains to be
charged by the available ledger proxies.

Continuum ledgers that are not present in the cached artifacts are not invented:
they are marked in ``missing_terms`` and in the generated TeX report.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_RESULTS = Path("benchmarks/jhtdb_response_family/results")
DEFAULT_FINAL = Path("Papers/final")
DOMAIN_LENGTH = 2.0 * math.pi


@dataclass(frozen=True)
class MembershipCase:
    label: str
    case: str
    membership_csv: Path
    pair_summary_csv: Path | None
    renewal_csv: Path | None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL)
    parser.add_argument(
        "--attack-mode",
        default="same_parent_split",
        choices=("same_parent_split", "cross_parent_jitter", "renewal_cascade"),
        help="Attack family to run. The default keeps the previous same-parent split path unchanged.",
    )
    parser.add_argument("--multiplicities", nargs="+", type=int, default=[1, 4, 8, 16, 32, 64])
    parser.add_argument("--delta", type=float, default=0.25, help="Nominal physical quotient threshold.")
    parser.add_argument("--c0", type=float, default=1.0, help="Loop multiplier used for the plotted C0 theta proxy.")
    parser.add_argument("--eta-target", type=float, default=0.5)
    parser.add_argument("--showcase-eta-threshold", type=float, default=0.9)
    parser.add_argument("--showcase-post-growth-max", type=float, default=0.25)
    parser.add_argument("--fphys-target", type=float, default=0.05)
    parser.add_argument("--max-phi-pairs", type=int, default=25000)
    parser.add_argument("--jitter-strengths", nargs="+", type=float, default=[1.0])
    parser.add_argument("--jitter-spatial-frac-min", type=float, default=0.05)
    parser.add_argument("--jitter-spatial-frac-max", type=float, default=0.15)
    parser.add_argument("--jitter-time-frac-min", type=float, default=0.05)
    parser.add_argument("--jitter-time-frac-max", type=float, default=0.10)
    parser.add_argument(
        "--phi-env-scale",
        type=float,
        default=1.0,
        help="Scale factor for Phi_env in the quotient distance; <1 relaxes only envelope separation.",
    )
    parser.add_argument(
        "--phi-act-scale",
        type=float,
        default=1.0,
        help="Scale factor for Phi_act in the quotient distance; <1 relaxes only activation separation.",
    )
    parser.add_argument(
        "--phi-env-scales",
        nargs="+",
        type=float,
        default=[],
        help="Optional sweep for Phi_env scale. If provided, overrides single --phi-env-scale value.",
    )
    parser.add_argument(
        "--phi-act-scales",
        nargs="+",
        type=float,
        default=[],
        help="Optional sweep for Phi_act scale. If provided, overrides single --phi-act-scale value.",
    )
    parser.add_argument(
        "--case-filter",
        nargs="+",
        default=[],
        help="Optional case filter (for example: c413 c309). Empty runs all discovered cases.",
    )
    parser.add_argument(
        "--radius-filter-dx",
        nargs="+",
        type=float,
        default=[],
        help="Optional radius filter in dx units (for example: 0.1). Empty runs all available radii.",
    )
    parser.add_argument(
        "--renewal-cascade-slab-count",
        type=int,
        default=6,
        help="Number of temporal slabs used to form renewal-cascade subsets.",
    )
    parser.add_argument(
        "--renewal-cascade-selection-mode",
        choices=("slab_balanced", "max_source_separated", "max_source_ranked"),
        default="slab_balanced",
        help="Tube-selection strategy in renewal_cascade mode.",
    )
    parser.add_argument(
        "--renewal-cascade-top-k",
        type=int,
        default=4,
        help="Target number of renewal-cascade tubes (used by selection strategies).",
    )
    parser.add_argument(
        "--renewal-cascade-min-time-sep-factor",
        type=float,
        default=0.5,
        help="Minimum mid-time separation factor; threshold is factor * r^2 (dx units) for renewal_cascade mode.",
    )
    parser.add_argument(
        "--phi-def-scale",
        type=float,
        default=1.0,
        help="Scale factor for Phi_def in quotient distance.",
    )
    parser.add_argument(
        "--shape-degenerate-enabled",
        action="store_true",
        help="Enable shape-degenerate seed perturbations in jitter generation.",
    )
    parser.add_argument(
        "--shape-degenerate-strength",
        type=float,
        default=1.0,
        help="Strength of shape-degenerate perturbation when enabled.",
    )
    parser.add_argument("--random-seed", type=int, default=1729)
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Optional artifact prefix. Empty keeps legacy name for same_parent_split and a mode-specific name for cross_parent_jitter.",
    )
    parser.add_argument(
        "--membership-case",
        nargs=5,
        action="append",
        metavar=("LABEL", "CASE", "MEMBERSHIP_CSV", "PAIR_SUMMARY_CSV", "RENEWAL_CSV"),
        help=(
            "Optional explicit membership case specification. "
            "Use '-' for a missing pair-summary or renewal CSV."
        ),
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    final_dir = args.final_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    explicit_cases = _explicit_membership_cases(args.membership_case)
    cases = _discover_membership_cases(results_dir, explicit_cases=explicit_cases)
    if not cases:
        raise SystemExit(f"No promoted membership CSVs found in {results_dir}")

    ledger_proxies = _load_ledger_proxies(results_dir)
    absorption = _load_absorption_points(results_dir)
    theta_lookup = _theta_lookup(absorption)
    jitter_strengths = sorted({float(value) for value in args.jitter_strengths if float(value) >= 0.0})
    if not jitter_strengths:
        jitter_strengths = [1.0]
    phi_env_scales = _resolve_scale_sweep(float(args.phi_env_scale), args.phi_env_scales)
    phi_act_scales = _resolve_scale_sweep(float(args.phi_act_scale), args.phi_act_scales)
    case_filter = {str(case_name).strip().lower() for case_name in args.case_filter if str(case_name).strip()}
    radius_filter_dx = [float(value) for value in args.radius_filter_dx]

    summary_rows: list[dict[str, Any]] = []
    phi_rows: list[dict[str, Any]] = []
    for case in cases:
        if case_filter and str(case.case).lower() not in case_filter:
            continue

        membership = pd.read_csv(case.membership_csv)
        prepared = _prepare_membership(membership)
        if prepared.empty:
            continue

        candidate_id = str(prepared["candidate_id"].iloc[0])
        radius_dx = float(prepared["tube_radius_dx"].dropna().iloc[0])
        if radius_filter_dx and not any(math.isclose(radius_dx, r, rel_tol=1e-9, abs_tol=1e-9) for r in radius_filter_dx):
            continue

        natural = _natural_family_stats(prepared, case=case.case, candidate_id=candidate_id, radius_dx=radius_dx)
        natural_tubes_full = _natural_tube_catalog(prepared)
        z0_tag = _z0_tag(candidate_id)
        theta_proxy = _theta_proxy_for_candidate(theta_lookup, candidate_id=candidate_id, radius_dx=radius_dx)
        proxy = ledger_proxies.get(candidate_id, {})
        overlap = _read_pair_summary(case.pair_summary_csv, candidate_id)
        renewal = _read_renewal_summary(case.renewal_csv, candidate_id)

        attack_natural = natural
        attack_tubes = natural_tubes_full
        attack_family = "cross_parent_jitter"
        attack_pair_group = "cross_parent_jitter"
        attack_note = "cross-parent near-degenerate jittered clones"
        attack_extra_fields: dict[str, Any] = {}
        selected_tube_ids: set[str] | None = None

        if args.attack_mode == "renewal_cascade":
            attack_family = "renewal_cascade_jitter"
            attack_pair_group = "renewal_cascade_jitter"
            renewal_selection_mode = str(args.renewal_cascade_selection_mode)
            attack_tubes, cascade_meta = _renewal_cascade_tube_subset(
                natural_tubes_full,
                radius_dx=radius_dx,
                slab_count=int(args.renewal_cascade_slab_count),
                min_time_sep_factor=float(args.renewal_cascade_min_time_sep_factor),
                selection_mode=renewal_selection_mode,
                top_k=int(args.renewal_cascade_top_k),
            )
            selected_tube_ids = {str(value) for value in attack_tubes["tube_id"].astype(str).to_list()}
            attack_natural = _natural_stats_from_catalog(natural, attack_tubes, cascade_meta=cascade_meta)
            attack_extra_fields = dict(cascade_meta)
            attack_note = "renewal-cascade temporal slab subset with maximal separation plus cross-parent jitter"
            if renewal_selection_mode == "max_source_ranked":
                attack_note = "renewal-cascade strict top-source subset with cross-parent jitter"
            if bool(args.shape_degenerate_enabled):
                attack_note += " and shape-degenerate perturbation"

        if args.attack_mode == "same_parent_split":
            for M in sorted(set(int(v) for v in args.multiplicities if int(v) >= 1)):
                split = _split_attack_stats(
                    natural=natural,
                    M=M,
                    delta=float(args.delta),
                    overlap=overlap,
                    renewal=renewal,
                    proxy=proxy,
                    eta_target=float(args.eta_target),
                    fphys_target=float(args.fphys_target),
                )
                split.update(
                    {
                        "attack_family": "same_parent_split",
                        "ablation_mode": "none",
                        "phi_env_scale": 1.0,
                        "phi_act_scale": 1.0,
                        "phi_def_scale": 1.0,
                        "shape_degenerate_enabled": False,
                        "shape_degenerate_strength": 0.0,
                        "z0_tag": z0_tag,
                        "jitter_strength": 0.0,
                        "jitter_spatial_frac": 0.0,
                        "jitter_time_frac": 0.0,
                        "clone_count_per_tube": int(M),
                        "theta_proxy": theta_proxy,
                        "C0theta_proxy": theta_proxy * float(args.c0) if math.isfinite(theta_proxy) else np.nan,
                        "post_growth_fraction": 0.0,
                    }
                )
                summary_rows.append(split)

            phi_rows.extend(
                _phi_diagnostic_rows(
                    prepared,
                    case=case.case,
                    candidate_id=candidate_id,
                    radius_dx=radius_dx,
                    delta=float(args.delta),
                    max_pairs=int(args.max_phi_pairs),
                    attack_family="same_parent_split",
                    pair_group="natural_cross_parent_baseline",
                    jitter_strength=0.0,
                    M=1,
                )
            )
        else:
            phi_rows.extend(
                _phi_diagnostic_rows(
                    prepared,
                    case=case.case,
                    candidate_id=candidate_id,
                    radius_dx=radius_dx,
                    delta=float(args.delta),
                    max_pairs=int(args.max_phi_pairs),
                    attack_family="same_parent_split_baseline",
                    pair_group="same_parent_baseline",
                    jitter_strength=0.0,
                    M=1,
                    tube_ids=selected_tube_ids,
                )
            )
            for M in sorted(set(int(v) for v in args.multiplicities if int(v) >= 1)):
                strengths = [0.0] if int(M) <= 1 else jitter_strengths
                for jitter_strength in strengths:
                    for phi_env_scale in phi_env_scales:
                        for phi_act_scale in phi_act_scales:
                            jitter_summary, jitter_phi = _cross_parent_jitter_stats(
                                natural=attack_natural,
                                natural_tubes=attack_tubes,
                                M=int(M),
                                delta=float(args.delta),
                                jitter_strength=float(jitter_strength),
                                jitter_spatial_frac_min=float(args.jitter_spatial_frac_min),
                                jitter_spatial_frac_max=float(args.jitter_spatial_frac_max),
                                jitter_time_frac_min=float(args.jitter_time_frac_min),
                                jitter_time_frac_max=float(args.jitter_time_frac_max),
                                overlap=overlap,
                                renewal=renewal,
                                proxy=proxy,
                                eta_target=float(args.eta_target),
                                showcase_eta_threshold=float(args.showcase_eta_threshold),
                                showcase_post_growth_max=float(args.showcase_post_growth_max),
                                fphys_target=float(args.fphys_target),
                                theta_proxy=theta_proxy,
                                c0=float(args.c0),
                                max_phi_pairs=int(args.max_phi_pairs),
                                seed=int(args.random_seed),
                                z0_tag=z0_tag,
                                phi_env_scale=float(phi_env_scale),
                                phi_act_scale=float(phi_act_scale),
                                phi_def_scale=float(args.phi_def_scale),
                                shape_degenerate_enabled=bool(args.shape_degenerate_enabled),
                                shape_degenerate_strength=float(args.shape_degenerate_strength),
                                attack_family=attack_family,
                                pair_group=attack_pair_group,
                                attack_note=attack_note,
                                extra_fields=dict(attack_extra_fields),
                            )
                            summary_rows.append(jitter_summary)
                            phi_rows.extend(jitter_phi)

    missing_case_rows = _missing_membership_case_rows(results_dir, case_filter=case_filter)
    summary = pd.DataFrame(summary_rows + missing_case_rows)
    phi = pd.DataFrame(phi_rows)
    absorption_df = pd.DataFrame(absorption)

    default_prefix = (
        "near_degenerate_multi_tube_packing_attack"
        if args.attack_mode == "same_parent_split"
        else (
            "cross_parent_near_degenerate_jitter_attack"
            if args.attack_mode == "cross_parent_jitter"
            else "renewal_cascade_cross_parent_jitter_attack"
        )
    )
    prefix = args.output_prefix.strip() or default_prefix

    summary_csv = results_dir / f"{prefix}_summary.csv"
    phi_csv = results_dir / f"{prefix}_phi_pairs.csv"
    absorption_csv = results_dir / f"{prefix}_absorption_proxy.csv"
    status_tex = results_dir / f"{prefix}_status.tex"
    report_tex = final_dir / f"{prefix}_report.tex"

    summary.to_csv(summary_csv, index=False)
    phi.to_csv(phi_csv, index=False)
    absorption_df.to_csv(absorption_csv, index=False)

    plot_paths = _write_plots(
        summary,
        phi,
        absorption_df,
        results_dir=results_dir,
        c0=float(args.c0),
        growth_cap=float(args.showcase_post_growth_max),
        prefix=prefix,
    )
    _write_status_tex(summary, status_tex)
    _write_report_tex(
        summary=summary,
        phi=phi,
        absorption=absorption_df,
        summary_csv=summary_csv,
        phi_csv=phi_csv,
        absorption_csv=absorption_csv,
        plot_paths=plot_paths,
        report_tex=report_tex,
        status_tex=status_tex,
        delta=float(args.delta),
        c0=float(args.c0),
    )

    print("near_degenerate_multi_tube_packing_attack", flush=True)
    print(f"summary_csv={summary_csv}", flush=True)
    print(f"phi_csv={phi_csv}", flush=True)
    print(f"absorption_csv={absorption_csv}", flush=True)
    print(f"status_tex={status_tex}", flush=True)
    print(f"report_tex={report_tex}", flush=True)
    if not summary.empty:
        cols = [
            "attack_family",
            "case",
            "radius_dx",
            "M",
            "jitter_strength",
            "ablation_mode",
            "phi_env_scale",
            "phi_act_scale",
            "phi_def_scale",
            "shape_degenerate_enabled",
            "shape_degenerate_strength",
            "pre_quotient_tubes",
            "post_quotient_tubes",
            "post_growth_fraction",
            "eta_post",
            "Fphys_star_available",
            "attack_status",
        ]
        cols = [column for column in cols if column in summary.columns]
        print(summary[cols].to_string(index=False), flush=True)


def _explicit_membership_cases(specs: list[list[str]] | None) -> list[MembershipCase] | None:
    if not specs:
        return None
    out: list[MembershipCase] = []
    for label, case, membership_csv, pair_summary_csv, renewal_csv in specs:
        membership_path = Path(membership_csv)
        if not membership_path.exists():
            raise SystemExit(f"Explicit membership CSV is missing: {membership_path}")
        pair_path = None if pair_summary_csv == "-" else Path(pair_summary_csv)
        if pair_path is not None and not pair_path.exists():
            raise SystemExit(f"Explicit pair-summary CSV is missing: {pair_path}")
        renewal_path = None if renewal_csv == "-" else Path(renewal_csv)
        if renewal_path is not None and not renewal_path.exists():
            raise SystemExit(f"Explicit renewal CSV is missing: {renewal_path}")
        out.append(
            MembershipCase(
                label=str(label),
                case=str(case),
                membership_csv=membership_path,
                pair_summary_csv=pair_path,
                renewal_csv=renewal_path,
            )
        )
    return out


def _discover_membership_cases(
    results_dir: Path,
    *,
    explicit_cases: list[MembershipCase] | None = None,
) -> list[MembershipCase]:
    if explicit_cases is not None:
        return explicit_cases
    out: list[MembershipCase] = []
    specs = [
        ("c413_r0p1", "c413", "promoted_tube_family_membership_top2_c413_r0p1.csv"),
        ("c413_r0p25", "c413", "promoted_tube_family_membership_top2_c413_r0p25.csv"),
        ("c309_r0p1", "c309", "promoted_tube_family_membership_top2_c309_r0p1.csv"),
        ("c309_r0p25", "c309", "promoted_tube_family_membership_top2_c309_r0p25.csv"),
    ]
    for label, case, name in specs:
        membership = results_dir / name
        if not membership.exists():
            continue
        rtag = "r0p25" if "r0p25" in name else "r0p1"
        pair_summary = results_dir / f"tube_pair_overlap_audit_top2_{case}_{rtag}_summary.csv"
        renewal = (
            results_dir / "real_packing_weighted_renewal_audit_r0p25_summary.csv"
            if rtag == "r0p25"
            else results_dir / "real_packing_weighted_renewal_audit_summary.csv"
        )
        out.append(
            MembershipCase(
                label=label,
                case=case,
                membership_csv=membership,
                pair_summary_csv=pair_summary if pair_summary.exists() else None,
                renewal_csv=renewal if renewal.exists() else None,
            )
        )
    return out


def _resolve_scale_sweep(default_value: float, sweep_values: list[float]) -> list[float]:
    if not sweep_values:
        return [max(0.0, float(default_value))]
    out: list[float] = []
    for value in sweep_values:
        candidate = max(0.0, float(value))
        if not any(math.isclose(candidate, existing, rel_tol=1e-12, abs_tol=1e-12) for existing in out):
            out.append(candidate)
    return out or [max(0.0, float(default_value))]


def _prepare_membership(membership: pd.DataFrame) -> pd.DataFrame:
    data = membership.copy()
    if "active_flag" in data.columns:
        data = data[data["active_flag"].astype(str).str.lower().isin({"true", "1"})].copy()
    if data.empty:
        return data
    data["source_weight"] = pd.to_numeric(data["source_weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    data = data[data["source_weight"] > 0.0].copy()
    for col in ["x", "y", "z", "tube_radius_dx"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["x", "y", "z", "tube_radius_dx"])
    return data


def _natural_family_stats(prepared: pd.DataFrame, *, case: str, candidate_id: str, radius_dx: float) -> dict[str, Any]:
    tube_masses = prepared.groupby(prepared["tube_id"].astype(str))["source_weight"].sum()
    masses = np.asarray(tube_masses.to_numpy(dtype=float), dtype=float)
    total_source = float(prepared["source_weight"].sum())
    return {
        "case": case,
        "candidate_id": candidate_id,
        "radius_dx": float(radius_dx),
        "natural_tube_count": int(prepared["tube_id"].nunique()),
        "natural_source": total_source,
        "natural_positive_rows": int(len(prepared)),
        "natural_N_eff_2": _neff2(masses),
        "natural_N_eff_half": _neff_half(masses),
    }


def _natural_tube_catalog(prepared: pd.DataFrame) -> pd.DataFrame:
    data = prepared.copy()
    data["tube_id"] = data["tube_id"].astype(str)
    data["time_index"] = pd.to_numeric(data["time_index"], errors="coerce")
    if "time" in data.columns:
        data["time"] = pd.to_numeric(data["time"], errors="coerce")
    else:
        data["time"] = data["time_index"]
    data = data.dropna(subset=["time_index", "time"])
    if data.empty:
        return pd.DataFrame(
            columns=[
                "tube_id",
                "source_mass",
                "seed_x",
                "seed_y",
                "seed_z",
                "final_x",
                "final_y",
                "final_z",
                "start_time",
                "end_time",
                "mid_time",
                "span_time",
            ]
        )

    masses = data.groupby("tube_id", as_index=False)["source_weight"].sum().rename(columns={"source_weight": "source_mass"})
    seed_rows = data.sort_values("time_index").groupby("tube_id", as_index=False).first()[["tube_id", "x", "y", "z"]]
    final_rows = data.sort_values("time_index").groupby("tube_id", as_index=False).last()[["tube_id", "x", "y", "z"]]
    time_stats = (
        data.groupby("tube_id", as_index=False)["time"]
        .agg(start_time="min", end_time="max", mid_time="median")
        .copy()
    )
    time_stats["span_time"] = time_stats["end_time"] - time_stats["start_time"]
    seed_rows = seed_rows.rename(columns={"x": "seed_x", "y": "seed_y", "z": "seed_z"})
    final_rows = final_rows.rename(columns={"x": "final_x", "y": "final_y", "z": "final_z"})
    out = (
        masses.merge(seed_rows, on="tube_id", how="left")
        .merge(final_rows, on="tube_id", how="left")
        .merge(time_stats, on="tube_id", how="left")
    )
    return out


def _natural_stats_from_catalog(
    natural: dict[str, Any],
    catalog: pd.DataFrame,
    *,
    cascade_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if catalog.empty:
        return dict(natural)
    masses = np.asarray(pd.to_numeric(catalog["source_mass"], errors="coerce").fillna(0.0).to_numpy(dtype=float), dtype=float)
    total_source = float(np.sum(np.clip(masses, 0.0, None)))
    out = dict(natural)
    out["natural_tube_count"] = int(len(catalog))
    out["natural_source"] = total_source
    out["natural_positive_rows"] = int(len(catalog))
    out["natural_N_eff_2"] = _neff2(masses)
    out["natural_N_eff_half"] = _neff_half(masses)
    if cascade_meta:
        out.update(cascade_meta)
    return out


def _renewal_cascade_tube_subset(
    natural_tubes: pd.DataFrame,
    *,
    radius_dx: float,
    slab_count: int,
    min_time_sep_factor: float,
    selection_mode: str,
    top_k: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selection_mode = str(selection_mode or "slab_balanced")
    top_k = int(max(2, top_k))
    if natural_tubes.empty:
        return natural_tubes.copy(), {
            "renewal_cascade_tubes": 0,
            "renewal_cascade_source_fraction": 0.0,
            "renewal_cascade_time_span": 0.0,
            "renewal_cascade_min_midtime_gap": 0.0,
            "renewal_cascade_target_min_gap": 0.0,
            "renewal_cascade_slab_count": int(max(2, slab_count)),
            "renewal_cascade_selection_mode": selection_mode,
            "renewal_cascade_top_k": top_k,
        }

    data = natural_tubes.copy()
    data["tube_id"] = data["tube_id"].astype(str)
    data["source_mass"] = pd.to_numeric(data["source_mass"], errors="coerce").fillna(0.0)
    data["mid_time"] = pd.to_numeric(data.get("mid_time", np.nan), errors="coerce")
    data["start_time"] = pd.to_numeric(data.get("start_time", np.nan), errors="coerce")
    data["end_time"] = pd.to_numeric(data.get("end_time", np.nan), errors="coerce")
    data = data[np.isfinite(data["mid_time"])].copy()
    if data.empty:
        return natural_tubes.copy(), {
            "renewal_cascade_tubes": int(len(natural_tubes)),
            "renewal_cascade_source_fraction": 1.0,
            "renewal_cascade_time_span": 0.0,
            "renewal_cascade_min_midtime_gap": 0.0,
            "renewal_cascade_target_min_gap": 0.0,
            "renewal_cascade_slab_count": int(max(2, slab_count)),
            "renewal_cascade_selection_mode": selection_mode,
            "renewal_cascade_top_k": top_k,
        }

    slab_count = int(max(2, slab_count))
    t_min = float(data["mid_time"].min())
    t_max = float(data["mid_time"].max())
    t_span = max(t_max - t_min, 0.0)
    if t_span <= 1e-12:
        data["slab_id"] = 0
    else:
        raw = (data["mid_time"] - t_min) / t_span * float(slab_count)
        data["slab_id"] = np.floor(raw).clip(0, slab_count - 1).astype(int)

    target_min_gap = float(max(0.0, min_time_sep_factor) * max(float(radius_dx) ** 2, 1e-12))
    selected_ids: list[str] = []

    mid_time_by_id = {
        str(row["tube_id"]): float(row["mid_time"])
        for _, row in data[["tube_id", "mid_time"]].drop_duplicates(subset=["tube_id"]).iterrows()
    }

    def _passes_min_gap(candidate_mid_time: float) -> bool:
        if not selected_ids:
            return True
        selected_times = np.asarray(
            [mid_time_by_id[tube_id] for tube_id in selected_ids if tube_id in mid_time_by_id],
            dtype=float,
        )
        if selected_times.size == 0:
            return True
        return float(np.min(np.abs(selected_times - candidate_mid_time))) >= target_min_gap

    if selection_mode == "max_source_ranked":
        ranked = data.sort_values(["source_mass", "mid_time"], ascending=[False, True]).copy()
        for _, row in ranked.iterrows():
            if len(selected_ids) >= top_k:
                break
            tube_id = str(row["tube_id"])
            if tube_id in selected_ids:
                continue
            if _passes_min_gap(float(row["mid_time"])):
                selected_ids.append(tube_id)

        if len(selected_ids) < top_k:
            for _, row in ranked.iterrows():
                tube_id = str(row["tube_id"])
                if tube_id in selected_ids:
                    continue
                selected_ids.append(tube_id)
                if len(selected_ids) >= top_k:
                    break
    elif selection_mode == "max_source_separated":
        ranked = data.sort_values("source_mass", ascending=False).copy()
        if len(data) >= 2:
            earliest = data.nsmallest(1, "mid_time")
            latest = data.nlargest(1, "mid_time")
            for _, row in pd.concat([earliest, latest], ignore_index=True).drop_duplicates(subset=["tube_id"]).iterrows():
                selected_ids.append(str(row["tube_id"]))

        for _, row in ranked.iterrows():
            if len(selected_ids) >= top_k:
                break
            tube_id = str(row["tube_id"])
            if tube_id in selected_ids:
                continue
            t_mid = float(row["mid_time"])
            if _passes_min_gap(t_mid):
                selected_ids.append(tube_id)

        if len(selected_ids) < top_k:
            for _, row in ranked.iterrows():
                tube_id = str(row["tube_id"])
                if tube_id in selected_ids:
                    continue
                selected_ids.append(tube_id)
                if len(selected_ids) >= top_k:
                    break
    else:
        slab_top = data.sort_values(["slab_id", "source_mass"], ascending=[True, False]).groupby("slab_id", as_index=False).head(1)
        earliest = data.nsmallest(1, "mid_time")
        latest = data.nlargest(1, "mid_time")
        candidates = pd.concat([slab_top, earliest, latest], ignore_index=True).drop_duplicates(subset=["tube_id"]).copy()

        seed_rows = pd.concat([earliest, latest], ignore_index=True).drop_duplicates(subset=["tube_id"])
        for _, row in seed_rows.iterrows():
            selected_ids.append(str(row["tube_id"]))

        remainder = candidates[~candidates["tube_id"].astype(str).isin(selected_ids)].sort_values("source_mass", ascending=False)
        for _, row in remainder.iterrows():
            tube_id = str(row["tube_id"])
            t_mid = float(row["mid_time"])
            if _passes_min_gap(t_mid):
                selected_ids.append(tube_id)

        if len(selected_ids) > top_k:
            selected_ids = selected_ids[:top_k]

    if len(selected_ids) < 2 and len(data) >= 2:
        fallback = data.sort_values("mid_time")
        selected_ids = [str(fallback.iloc[0]["tube_id"]), str(fallback.iloc[-1]["tube_id"])]

    selected = data[data["tube_id"].astype(str).isin(selected_ids)].copy()
    selected = selected.drop_duplicates(subset=["tube_id"]).sort_values(["mid_time", "source_mass"], ascending=[True, False]).reset_index(drop=True)

    total_source = float(pd.to_numeric(natural_tubes["source_mass"], errors="coerce").fillna(0.0).sum())
    selected_source = float(pd.to_numeric(selected["source_mass"], errors="coerce").fillna(0.0).sum())
    selected_times = selected["mid_time"].to_numpy(dtype=float)
    if selected_times.size >= 2:
        sorted_times = np.sort(selected_times)
        min_gap = float(np.min(np.diff(sorted_times)))
    else:
        min_gap = 0.0

    meta = {
        "renewal_cascade_tubes": int(len(selected)),
        "renewal_cascade_source_fraction": 0.0 if total_source <= 0 else selected_source / total_source,
        "renewal_cascade_time_span": 0.0 if selected.empty else float(selected["mid_time"].max() - selected["mid_time"].min()),
        "renewal_cascade_min_midtime_gap": min_gap,
        "renewal_cascade_target_min_gap": target_min_gap,
        "renewal_cascade_slab_count": slab_count,
        "renewal_cascade_selection_mode": selection_mode,
        "renewal_cascade_top_k": top_k,
    }
    return selected, meta


def _z0_tag(candidate_id: str) -> str:
    parts = str(candidate_id).split(":")
    if len(parts) >= 4:
        return ":".join(parts[1:4])
    if len(parts) >= 2:
        return ":".join(parts[1:])
    return str(candidate_id)


def _theta_lookup(rows: list[dict[str, Any]]) -> dict[str, list[tuple[float, float]]]:
    out: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id", ""))
        radius = _safe_float(row.get("radius_dx", np.nan))
        theta = _safe_float(row.get("theta_proxy", np.nan))
        if not candidate_id or not math.isfinite(theta):
            continue
        out.setdefault(candidate_id, []).append((radius, theta))
    return out


def _theta_proxy_for_candidate(
    lookup: dict[str, list[tuple[float, float]]],
    *,
    candidate_id: str,
    radius_dx: float,
) -> float:
    choices = lookup.get(str(candidate_id), [])
    if not choices:
        return float("nan")
    finite_radius = [(abs(radius - float(radius_dx)), theta) for radius, theta in choices if math.isfinite(radius)]
    if finite_radius:
        finite_radius.sort(key=lambda item: item[0])
        return float(finite_radius[0][1])
    return float(choices[0][1])


def _cross_parent_jitter_stats(
    *,
    natural: dict[str, Any],
    natural_tubes: pd.DataFrame,
    M: int,
    delta: float,
    jitter_strength: float,
    jitter_spatial_frac_min: float,
    jitter_spatial_frac_max: float,
    jitter_time_frac_min: float,
    jitter_time_frac_max: float,
    overlap: dict[str, float],
    renewal: dict[str, float],
    proxy: dict[str, Any],
    eta_target: float,
    showcase_eta_threshold: float,
    showcase_post_growth_max: float,
    fphys_target: float,
    theta_proxy: float,
    c0: float,
    max_phi_pairs: int,
    seed: int,
    z0_tag: str,
    phi_env_scale: float,
    phi_act_scale: float,
    phi_def_scale: float,
    shape_degenerate_enabled: bool,
    shape_degenerate_strength: float,
    attack_family: str,
    pair_group: str,
    attack_note: str,
    extra_fields: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if natural_tubes.empty:
        return {}, []
    if extra_fields is None:
        extra_fields = {}

    dx = DOMAIN_LENGTH / 1024.0
    r_phys = max(float(natural["radius_dx"]) * dx, 1e-12)
    rng = np.random.default_rng(seed + 7919 * int(M) + 104729 * int(round(1000.0 * float(jitter_strength))))
    clones = _build_jitter_clone_table(
        natural_tubes,
        M=int(M),
        jitter_strength=float(jitter_strength),
        jitter_spatial_frac_min=float(jitter_spatial_frac_min),
        jitter_spatial_frac_max=float(jitter_spatial_frac_max),
        jitter_time_frac_min=float(jitter_time_frac_min),
        jitter_time_frac_max=float(jitter_time_frac_max),
        r_phys=r_phys,
        rng=rng,
        shape_degenerate_enabled=bool(shape_degenerate_enabled),
        shape_degenerate_strength=float(shape_degenerate_strength),
    )
    phi_env_scale = max(0.0, float(phi_env_scale))
    phi_act_scale = max(0.0, float(phi_act_scale))
    phi_def_scale = max(0.0, float(phi_def_scale))
    shape_degenerate_strength = max(0.0, float(shape_degenerate_strength))
    ablation_tokens: list[str] = []
    if not math.isclose(phi_env_scale, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        ablation_tokens.append("relax_phi_env")
    if not math.isclose(phi_act_scale, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        ablation_tokens.append("relax_phi_act")
    if not math.isclose(phi_def_scale, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        ablation_tokens.append("scale_phi_def")
    if bool(shape_degenerate_enabled):
        ablation_tokens.append("shape_degenerate")
    ablation_mode = "none" if not ablation_tokens else "+".join(ablation_tokens)

    pre_count = int(len(clones))
    if pre_count <= 1:
        row = {
            "attack_family": str(attack_family),
            "ablation_mode": ablation_mode,
            "phi_env_scale": phi_env_scale,
            "phi_act_scale": phi_act_scale,
            "phi_def_scale": phi_def_scale,
            "shape_degenerate_enabled": bool(shape_degenerate_enabled),
            "shape_degenerate_strength": shape_degenerate_strength,
            "case": natural["case"],
            "candidate_id": natural["candidate_id"],
            "z0_tag": z0_tag,
            "radius_dx": natural["radius_dx"],
            "M": int(M),
            "clone_count_per_tube": int(M),
            "jitter_strength": float(jitter_strength),
            "jitter_spatial_frac": 0.0,
            "jitter_time_frac": 0.0,
            "delta": float(delta),
            "natural_tubes": int(natural["natural_tube_count"]),
            "pre_quotient_tubes": pre_count,
            "post_quotient_tubes": pre_count,
            "quotient_merged_fraction": 0.0,
            "post_growth_fraction": 0.0,
            "natural_source": natural["natural_source"],
            "post_quotient_source": natural["natural_source"],
            "eta_post": 1.0,
            "raw_split_pair_pack_proxy": 0.0,
            "R_pack_star": 0.0,
            "R_shape_fam_available": 0.0,
            "R_spread_fam_available": 0.0,
            "R_renew_fam_available": 0.0,
            "R_D_available": 0.0,
            "R_tail_available": 0.0,
            "E_nu_coh_pos_available": 0.0,
            "Fphys_star_available": 0.0,
            "natural_N_eff_2": natural["natural_N_eff_2"],
            "natural_N_eff_half": natural["natural_N_eff_half"],
            "theta_proxy": theta_proxy,
            "C0theta_proxy": theta_proxy * float(c0) if math.isfinite(theta_proxy) else np.nan,
            "nonredundant_pair_fraction": 0.0,
            "attack_status": "baseline_no_split",
            "missing_terms": "",
            "evidence_note": f"{attack_note}; insufficient pair count",
        }
        row.update(extra_fields)
        return row, []

    seeds = clones[["seed_x", "seed_y", "seed_z"]].to_numpy(dtype=float)
    base_seeds = clones[["base_seed_x", "base_seed_y", "base_seed_z"]].to_numpy(dtype=float)
    time_offsets = clones["time_offset_frac"].to_numpy(dtype=float)
    shape_tags = clones["shape_tag"].to_numpy(dtype=float)
    masses = clones["source_mass"].to_numpy(dtype=float)
    clone_ids = clones["clone_id"].astype(str).to_numpy()

    total_source = max(float(np.sum(masses)), 1e-12)
    temporal_scale = max(float(jitter_time_frac_max), 1e-12)

    uf_parent = np.arange(pre_count, dtype=int)
    uf_rank = np.zeros(pre_count, dtype=int)
    total_pairs = pre_count * (pre_count - 1) // 2
    sample_prob = 1.0 if total_pairs <= int(max_phi_pairs) else float(max_phi_pairs) / float(total_pairs)

    raw_overlap_sum = 0.0
    nonredundant_overlap_sum = 0.0
    nonredundant_pair_count = 0
    phi_rows: list[dict[str, Any]] = []

    for idx in range(pre_count - 1):
        right_slice = slice(idx + 1, pre_count)
        spatial = _periodic_distance_rows(seeds[idx], seeds[right_slice])
        ancestry = _periodic_distance_rows(base_seeds[idx], base_seeds[right_slice])
        phi_env = np.clip(spatial / r_phys, 0.0, 4.0)
        phi_anc = np.clip(ancestry / (3.0 * r_phys), 0.0, 1.0)
        phi_act = np.clip(np.abs(time_offsets[idx] - time_offsets[right_slice]) / temporal_scale, 0.0, 1.0)
        phi_def = np.clip(np.abs(shape_tags[idx] - shape_tags[right_slice]), 0.0, 1.0)
        phi_env_effective = np.clip(phi_env * phi_env_scale, 0.0, 4.0)
        phi_act_effective = np.clip(phi_act * phi_act_scale, 0.0, 1.0)
        phi_def_effective = np.clip(phi_def * phi_def_scale, 0.0, 1.0)
        d_phys_raw = np.maximum.reduce([phi_env, phi_anc, phi_act, phi_def])
        d_phys = np.maximum.reduce([phi_env_effective, phi_anc, phi_act_effective, phi_def_effective])

        overlap_weight = np.minimum(masses[idx], masses[right_slice])
        overlap_proxy = np.exp(-0.5 * np.square(phi_env / 1.5) - 0.5 * np.square(phi_act / 1.0))
        weighted_overlap = overlap_weight * overlap_proxy
        raw_overlap_sum += float(np.sum(weighted_overlap))

        merge_positions = np.where(d_phys <= float(delta))[0]
        for pos in merge_positions:
            _uf_union(uf_parent, uf_rank, idx, idx + 1 + int(pos))

        nonredundant_mask = d_phys > float(delta)
        nonredundant_pair_count += int(np.count_nonzero(nonredundant_mask))
        if np.any(nonredundant_mask):
            nonredundant_overlap_sum += float(np.sum(weighted_overlap[nonredundant_mask]))

        sample_mask = rng.random(d_phys.shape[0]) < sample_prob
        if np.any(sample_mask):
            sampled = np.where(sample_mask)[0]
            for pos in sampled:
                right_index = idx + 1 + int(pos)
                phi_rows.append(
                    {
                        "attack_family": str(attack_family),
                        "pair_group": str(pair_group),
                        "case": natural["case"],
                        "candidate_id": natural["candidate_id"],
                        "radius_dx": natural["radius_dx"],
                        "M": int(M),
                        "jitter_strength": float(jitter_strength),
                        "ablation_mode": ablation_mode,
                        "phi_env_scale": phi_env_scale,
                        "phi_act_scale": phi_act_scale,
                        "phi_def_scale": phi_def_scale,
                        "shape_degenerate_enabled": bool(shape_degenerate_enabled),
                        "shape_degenerate_strength": shape_degenerate_strength,
                        "delta": float(delta),
                        "tube_i": str(clone_ids[idx]),
                        "tube_j": str(clone_ids[right_index]),
                        "Phi_env_centroid_over_r": float(phi_env[pos]),
                        "Phi_env_effective_proxy": float(phi_env_effective[pos]),
                        "Phi_anc_proxy": float(phi_anc[pos]),
                        "Phi_act_proxy": float(phi_act[pos]),
                        "Phi_act_effective_proxy": float(phi_act_effective[pos]),
                        "Phi_def_proxy": float(phi_def[pos]),
                        "Phi_def_effective_proxy": float(phi_def_effective[pos]),
                        "d_phys_raw_proxy": float(d_phys_raw[pos]),
                        "d_phys_proxy": float(d_phys[pos]),
                        "overlap_weight_proxy": float(weighted_overlap[pos]),
                        "survives_quotient": bool(d_phys[pos] > float(delta)),
                        "diagnostic_note": f"{attack_note}; pair under fixed delta/Phi proxy",
                    }
                )

    if len(phi_rows) > int(max_phi_pairs):
        keep = rng.choice(len(phi_rows), size=int(max_phi_pairs), replace=False)
        phi_rows = [phi_rows[int(index)] for index in sorted(int(k) for k in keep)]

    roots = {_uf_find(uf_parent, idx) for idx in range(pre_count)}
    post_count = int(len(roots))
    natural_tubes = int(natural["natural_tube_count"])
    merged_fraction = 0.0 if pre_count <= natural_tubes else (pre_count - post_count) / max(pre_count - natural_tubes, 1)
    post_growth_fraction = (post_count - natural_tubes) / max(natural_tubes, 1)

    raw_pair_pack = raw_overlap_sum / total_source
    post_pack = nonredundant_overlap_sum / total_source

    r_spread = _positive_or_proxy(renewal.get("R_spread"), proxy.get("R_spread_branch"))
    r_shape = _positive_or_proxy(renewal.get("R_shape"), proxy.get("R_shape_branch"))
    r_renew = _positive_or_proxy(renewal.get("R_renew_total"), 0.0)
    r_deact = _positive_or_proxy(renewal.get("R_D"), proxy.get("R_deactivation_branch"))
    r_tail = _positive_or_proxy(proxy.get("R_tail_residual_branch"), 0.0)
    e_coh = _positive_or_proxy(proxy.get("E_nu"), 0.0)
    fphys_available = post_pack + r_spread + r_shape + r_renew + r_deact + r_tail + e_coh

    eta_post = 1.0 - 0.1 * max(post_count - natural_tubes, 0) / max(pre_count, 1)
    eta_post = float(np.clip(eta_post, 0.90, 1.0))

    c0_theta = float(c0) * theta_proxy if math.isfinite(theta_proxy) else float("nan")
    nonredundant_pair_fraction = float(nonredundant_pair_count) / float(max(total_pairs, 1))

    if int(M) <= 1:
        attack_status = "baseline_no_split"
    elif (
        eta_post >= float(showcase_eta_threshold)
        and fphys_available >= float(fphys_target)
        and (not math.isfinite(c0_theta) or c0_theta < 1.0)
        and post_growth_fraction <= float(showcase_post_growth_max)
    ):
        attack_status = "pass_showcase"
    elif post_growth_fraction > float(showcase_post_growth_max):
        attack_status = "break_excess_post_growth"
    elif eta_post >= float(eta_target) and fphys_available < float(fphys_target):
        attack_status = "break_low_available_Fphys"
    elif math.isfinite(c0_theta) and c0_theta >= 1.0:
        attack_status = "break_loop_threshold"
    elif eta_post < float(eta_target):
        attack_status = "capture_lost"
    else:
        attack_status = "ambiguous_near_threshold"

    missing = [
        "R_boundary_fam",
        "R_tail_full" if r_tail == 0.0 else "",
        "E_omega_full",
        "B_omega_full",
        "D_ARR_full",
        "Phi_def_full",
        "Phi_bdry_full",
        "Phi_res_full",
    ]
    missing_terms = ";".join(sorted({term for term in missing if term}))

    jitter_spatial_frac = float(np.mean(clones["spatial_offset_frac"].to_numpy(dtype=float)))
    jitter_time_frac = float(np.mean(np.abs(clones["time_offset_frac"].to_numpy(dtype=float))))

    row = {
        "attack_family": str(attack_family),
        "ablation_mode": ablation_mode,
        "phi_env_scale": phi_env_scale,
        "phi_act_scale": phi_act_scale,
        "phi_def_scale": phi_def_scale,
        "shape_degenerate_enabled": bool(shape_degenerate_enabled),
        "shape_degenerate_strength": shape_degenerate_strength,
        "case": natural["case"],
        "candidate_id": natural["candidate_id"],
        "z0_tag": z0_tag,
        "radius_dx": natural["radius_dx"],
        "M": int(M),
        "clone_count_per_tube": int(M),
        "jitter_strength": float(jitter_strength),
        "jitter_spatial_frac": jitter_spatial_frac,
        "jitter_time_frac": jitter_time_frac,
        "delta": float(delta),
        "natural_tubes": natural_tubes,
        "pre_quotient_tubes": pre_count,
        "post_quotient_tubes": post_count,
        "quotient_merged_fraction": merged_fraction,
        "post_growth_fraction": post_growth_fraction,
        "natural_source": natural["natural_source"],
        "post_quotient_source": natural["natural_source"],
        "eta_post": eta_post,
        "raw_split_pair_pack_proxy": raw_pair_pack,
        "R_pack_star": post_pack,
        "R_shape_fam_available": r_shape,
        "R_spread_fam_available": r_spread,
        "R_renew_fam_available": r_renew,
        "R_D_available": r_deact,
        "R_tail_available": r_tail,
        "E_nu_coh_pos_available": e_coh,
        "Fphys_star_available": fphys_available,
        "natural_N_eff_2": natural["natural_N_eff_2"],
        "natural_N_eff_half": natural["natural_N_eff_half"],
        "theta_proxy": theta_proxy,
        "C0theta_proxy": c0_theta,
        "nonredundant_pair_fraction": nonredundant_pair_fraction,
        "sampled_phi_pairs": int(len(phi_rows)),
        "attack_status": attack_status,
        "missing_terms": missing_terms,
        "evidence_note": (
            f"{attack_note}; fixed ledger proxies with "
            f"Phi_env_scale={phi_env_scale:g}, Phi_act_scale={phi_act_scale:g}"
        ),
    }
    row.update(extra_fields)
    return row, phi_rows


def _build_jitter_clone_table(
    natural_tubes: pd.DataFrame,
    *,
    M: int,
    jitter_strength: float,
    jitter_spatial_frac_min: float,
    jitter_spatial_frac_max: float,
    jitter_time_frac_min: float,
    jitter_time_frac_max: float,
    r_phys: float,
    rng: np.random.Generator,
    shape_degenerate_enabled: bool,
    shape_degenerate_strength: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    multiplicity = max(1, int(M))
    spatial_low = float(min(jitter_spatial_frac_min, jitter_spatial_frac_max))
    spatial_high = float(max(jitter_spatial_frac_min, jitter_spatial_frac_max))
    time_low = float(min(jitter_time_frac_min, jitter_time_frac_max))
    time_high = float(max(jitter_time_frac_min, jitter_time_frac_max))
    shape_degenerate_strength = max(0.0, float(shape_degenerate_strength))

    for _, tube in natural_tubes.iterrows():
        source_mass = float(tube["source_mass"]) / float(multiplicity)
        base_seed = np.asarray([tube["seed_x"], tube["seed_y"], tube["seed_z"]], dtype=float)
        for clone_idx in range(multiplicity):
            if multiplicity <= 1 or jitter_strength <= 0.0:
                spatial_frac = 0.0
                spatial_offset = np.zeros(3, dtype=float)
                time_offset_frac = 0.0
                shape_tag = 0.0
                shape_degenerate_load = 0.0
            else:
                direction = rng.normal(size=3)
                direction_norm = float(np.linalg.norm(direction))
                if direction_norm <= 1e-12:
                    direction = np.asarray([1.0, 0.0, 0.0], dtype=float)
                    direction_norm = 1.0
                direction = direction / direction_norm
                spatial_frac = float(rng.uniform(spatial_low, spatial_high) * jitter_strength)
                spatial_offset = direction * (spatial_frac * r_phys)
                time_mag = float(rng.uniform(time_low, time_high) * jitter_strength)
                time_offset_frac = float(time_mag if rng.random() < 0.5 else -time_mag)
                shape_tag = float(rng.uniform(0.0, 0.25) * jitter_strength)
                shape_degenerate_load = 0.0

                if bool(shape_degenerate_enabled):
                    phase = (float(clone_idx) + 0.5) / float(multiplicity) - 0.5
                    anisotropic_dir = rng.normal(size=3)
                    anisotropic_norm = float(np.linalg.norm(anisotropic_dir))
                    if anisotropic_norm <= 1e-12:
                        anisotropic_dir = np.asarray([0.0, 1.0, 0.0], dtype=float)
                        anisotropic_norm = 1.0
                    anisotropic_dir = anisotropic_dir / anisotropic_norm
                    shape_degenerate_load = abs(phase) * shape_degenerate_strength * max(jitter_strength, 1e-12)
                    anisotropic_shift = anisotropic_dir * (shape_degenerate_load * r_phys)
                    spatial_offset = spatial_offset + anisotropic_shift
                    shape_tag = float(np.clip(shape_tag + shape_degenerate_load, 0.0, 1.0))

            seed = np.mod(base_seed + spatial_offset, DOMAIN_LENGTH)
            rows.append(
                {
                    "tube_id": str(tube["tube_id"]),
                    "clone_id": f"{tube['tube_id']}:j{clone_idx + 1}",
                    "parent_id": f"cross_parent:{tube['tube_id']}:j{clone_idx + 1}",
                    "source_mass": source_mass,
                    "base_seed_x": float(base_seed[0]),
                    "base_seed_y": float(base_seed[1]),
                    "base_seed_z": float(base_seed[2]),
                    "seed_x": float(seed[0]),
                    "seed_y": float(seed[1]),
                    "seed_z": float(seed[2]),
                    "spatial_offset_frac": float(spatial_frac),
                    "time_offset_frac": float(time_offset_frac),
                    "shape_tag": float(shape_tag),
                    "shape_degenerate_load": float(shape_degenerate_load),
                }
            )

    return pd.DataFrame(rows)


def _periodic_distance_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    diff = np.abs(right - left[None, :])
    diff = np.minimum(diff, DOMAIN_LENGTH - diff)
    return np.sqrt(np.sum(np.square(diff), axis=1))


def _uf_find(parent: np.ndarray, item: int) -> int:
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = int(parent[item])
    return int(item)


def _uf_union(parent: np.ndarray, rank: np.ndarray, left: int, right: int) -> None:
    root_left = _uf_find(parent, int(left))
    root_right = _uf_find(parent, int(right))
    if root_left == root_right:
        return
    if rank[root_left] < rank[root_right]:
        parent[root_left] = root_right
    elif rank[root_left] > rank[root_right]:
        parent[root_right] = root_left
    else:
        parent[root_right] = root_left
        rank[root_left] += 1


def _split_attack_stats(
    *,
    natural: dict[str, Any],
    M: int,
    delta: float,
    overlap: dict[str, float],
    renewal: dict[str, float],
    proxy: dict[str, Any],
    eta_target: float,
    fphys_target: float,
) -> dict[str, Any]:
    natural_tubes = int(natural["natural_tube_count"])
    pre_count = natural_tubes * int(M)
    post_count = natural_tubes
    merged_fraction = 0.0 if pre_count <= natural_tubes else (pre_count - post_count) / max(pre_count - natural_tubes, 1)
    raw_pair_pack = max(0.0, (float(M) - 1.0) / 2.0)
    post_pack = float(overlap.get("R_pack", 0.0))

    # Available ledger channels in the cached artifacts.  These are normalized
    # proof-facing proxies, not a full continuum D_omega evaluation.
    r_spread = _positive_or_proxy(renewal.get("R_spread"), proxy.get("R_spread_branch"))
    r_shape = _positive_or_proxy(renewal.get("R_shape"), proxy.get("R_shape_branch"))
    r_renew = _positive_or_proxy(renewal.get("R_renew_total"), 0.0)
    r_deact = _positive_or_proxy(renewal.get("R_D"), proxy.get("R_deactivation_branch"))
    r_tail = _positive_or_proxy(proxy.get("R_tail_residual_branch"), 0.0)
    e_coh = _positive_or_proxy(proxy.get("E_nu"), 0.0)
    fphys_available = post_pack + r_spread + r_shape + r_renew + r_deact + r_tail + e_coh

    missing = [
        "R_boundary_fam",
        "R_tail_full" if r_tail == 0.0 else "",
        "E_omega_full",
        "B_omega_full",
        "D_ARR_full",
        "Phi_def_full",
        "Phi_bdry_full",
        "Phi_res_full",
    ]
    missing_terms = ";".join(sorted({m for m in missing if m}))

    eta_post = 1.0
    if int(M) == 1:
        attack_status = "baseline_no_split"
    elif merged_fraction >= 0.999 and post_pack <= 1e-12:
        attack_status = "showcase_quotient_redundant"
    elif eta_post >= eta_target and fphys_available < fphys_target:
        attack_status = "break_low_available_Fphys"
    elif eta_post >= eta_target:
        attack_status = "showcase_ledger_charged"
    else:
        attack_status = "capture_lost"

    return {
        "case": natural["case"],
        "candidate_id": natural["candidate_id"],
        "radius_dx": natural["radius_dx"],
        "M": int(M),
        "delta": float(delta),
        "natural_tubes": natural_tubes,
        "pre_quotient_tubes": pre_count,
        "post_quotient_tubes": post_count,
        "quotient_merged_fraction": merged_fraction,
        "natural_source": natural["natural_source"],
        "post_quotient_source": natural["natural_source"],
        "eta_post": eta_post,
        "raw_split_pair_pack_proxy": raw_pair_pack,
        "R_pack_star": post_pack,
        "R_shape_fam_available": r_shape,
        "R_spread_fam_available": r_spread,
        "R_renew_fam_available": r_renew,
        "R_D_available": r_deact,
        "R_tail_available": r_tail,
        "E_nu_coh_pos_available": e_coh,
        "Fphys_star_available": fphys_available,
        "natural_N_eff_2": natural["natural_N_eff_2"],
        "natural_N_eff_half": natural["natural_N_eff_half"],
        "attack_status": attack_status,
        "missing_terms": missing_terms,
        "evidence_note": "adversarial label split is merged by same-parent physical quotient before non-redundant packing branch",
    }


def _read_pair_summary(path: Path | None, candidate_id: str) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    part = df[df["candidate_id"].astype(str).eq(str(candidate_id))]
    row = part.iloc[0] if not part.empty else df.iloc[0]
    return {k: _safe_float(row.get(k, 0.0)) for k in df.columns if k != "candidate_id"}


def _read_renewal_summary(path: Path | None, candidate_id: str) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    part = df[df["candidate_id"].astype(str).eq(str(candidate_id))]
    if part.empty:
        return {}
    row = part.iloc[0]
    keys = ["R_spread", "R_shape", "R_renew_total", "R_D", "R_tail", "ledger_sum", "unclassified_fraction"]
    return {k: _safe_float(row.get(k, 0.0)) for k in keys if k in row.index}


def _load_ledger_proxies(results_dir: Path) -> dict[str, dict[str, Any]]:
    path = results_dir / "targeted_dichotomy_audit_top2_top3_summary.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id:
            continue
        out[candidate_id] = {
            "G": _safe_float(row.get("G", np.nan)),
            "ledger_total_C1": _safe_float(row.get("ledger_total_C1", np.nan)),
            "C_min": _safe_float(row.get("C_min", np.nan)),
            "R_shape_branch": _safe_float(row.get("R_shape_branch", 0.0)),
            "R_spread_branch": _safe_float(row.get("R_spread_branch", 0.0)),
            "R_deactivation_branch": _safe_float(row.get("R_deactivation_branch", 0.0)),
            "R_tail_residual_branch": _safe_float(row.get("R_tail_residual_branch", 0.0)),
            "E_nu": _safe_float(row.get("E_nu", 0.0)),
            "dominant_branch": str(row.get("dominant_branch", "")),
            "stage": str(row.get("stage", "")),
        }
    return out


def _load_absorption_points(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target = results_dir / "targeted_dichotomy_audit_top2_top3_rows.csv"
    if target.exists():
        df = pd.read_csv(target)
        for _, row in df.iterrows():
            candidate_id = str(row.get("candidate_id", ""))
            case = _case_from_candidate(candidate_id)
            if not case:
                continue
            radius = _safe_float(row.get("radius_dx", np.nan))
            G = _safe_float(row.get("G", np.nan))
            ledger = _safe_float(row.get("ledger_total_C1", np.nan))
            theta = G / ledger if math.isfinite(G) and math.isfinite(ledger) and ledger > 0 else np.nan
            rows.append(
                {
                    "case": case,
                    "candidate_id": candidate_id,
                    "radius_dx": radius,
                    "theta_proxy": theta,
                    "C0theta_proxy": theta,
                    "source": "targeted_dichotomy_G_over_ledger_total_C1",
                    "stage": str(row.get("stage", "")),
                    "missing_terms": "full_Domega_continuum",
                }
            )
    arr = results_dir / "arr_deficit_attribution_audit_c185_final81_summary.csv"
    if arr.exists():
        df = pd.read_csv(arr)
        for _, row in df.iterrows():
            rows.append(
                {
                    "case": "c185",
                    "candidate_id": str(row.get("candidate_id", "")),
                    "radius_dx": np.nan,
                    "theta_proxy": _safe_float(row.get("ratio_with_renewal_exposure", np.nan)),
                    "C0theta_proxy": _safe_float(row.get("ratio_with_renewal_exposure", np.nan)),
                    "source": "ARR_ratio_with_renewal_exposure",
                    "stage": "ARR_final81",
                    "missing_terms": "promoted_membership_for_split_attack",
                }
            )
    return rows


def _phi_diagnostic_rows(
    prepared: pd.DataFrame,
    *,
    case: str,
    candidate_id: str,
    radius_dx: float,
    delta: float,
    max_pairs: int,
    attack_family: str = "same_parent_split",
    pair_group: str = "natural_cross_parent_baseline",
    jitter_strength: float = 0.0,
    M: int = 1,
    tube_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    centroids = (
        prepared.groupby([prepared["tube_id"].astype(str), prepared["time_index"].astype(int)], sort=False)[["x", "y", "z"]]
        .mean()
        .reset_index()
    )
    if tube_ids:
        allowed = {str(value) for value in tube_ids}
        centroids = centroids[centroids["tube_id"].astype(str).isin(allowed)].copy()
    tubes = sorted(centroids["tube_id"].astype(str).unique())
    if len(tubes) < 2:
        return []

    if str(pair_group) == "same_parent_baseline":
        baseline_rows: list[dict[str, Any]] = []
        for tube in tubes[: int(max_pairs)]:
            baseline_rows.append(
                {
                    "attack_family": str(attack_family),
                    "pair_group": str(pair_group),
                    "case": case,
                    "candidate_id": candidate_id,
                    "radius_dx": radius_dx,
                    "M": int(M),
                    "jitter_strength": float(jitter_strength),
                    "delta": float(delta),
                    "tube_i": tube,
                    "tube_j": f"{tube}:same_parent_ref",
                    "Phi_env_centroid_over_r": 0.0,
                    "Phi_anc_proxy": 0.0,
                    "Phi_act_proxy": 0.0,
                    "Phi_def_proxy": 0.0,
                    "d_phys_proxy": 0.0,
                    "overlap_weight_proxy": np.nan,
                    "survives_quotient": False,
                    "diagnostic_note": "same-parent baseline reference pairs at zero proxy distance",
                }
            )
        return baseline_rows

    time_indices = sorted(centroids["time_index"].astype(int).unique())
    centroid_map: dict[tuple[str, int], np.ndarray] = {}
    for _, row in centroids.iterrows():
        centroid_map[(str(row["tube_id"]), int(row["time_index"]))] = np.asarray([row["x"], row["y"], row["z"]], dtype=float)

    rng = np.random.default_rng(1729)
    pairs = [(tubes[i], tubes[j]) for i in range(len(tubes)) for j in range(i + 1, len(tubes))]
    if len(pairs) > max_pairs:
        take = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[int(i)] for i in sorted(take)]

    dx = DOMAIN_LENGTH / 1024.0
    r_phys = max(float(radius_dx) * dx, 1e-12)
    rows: list[dict[str, Any]] = []
    for tube_i, tube_j in pairs:
        distances = []
        for t in time_indices:
            left = centroid_map.get((tube_i, t))
            right = centroid_map.get((tube_j, t))
            if left is None or right is None:
                continue
            distances.append(_periodic_distance(left, right))
        if not distances:
            continue
        phi_env = float(max(distances) / r_phys)
        rows.append(
            {
                "attack_family": str(attack_family),
                "pair_group": str(pair_group),
                "case": case,
                "candidate_id": candidate_id,
                "radius_dx": radius_dx,
                "M": int(M),
                "jitter_strength": float(jitter_strength),
                "delta": float(delta),
                "tube_i": tube_i,
                "tube_j": tube_j,
                "Phi_env_centroid_over_r": phi_env,
                "Phi_anc_proxy": 1.0,
                "Phi_act_proxy": 0.0,
                "Phi_def_proxy": 0.0,
                "d_phys_proxy": max(phi_env, 1.0),
                "overlap_weight_proxy": np.nan,
                "survives_quotient": bool(max(phi_env, 1.0) > delta),
                "diagnostic_note": "natural cross-parent pair; split-clone same-parent pairs have all proxy Phi=0 and are merged",
            }
        )
    return rows


def _periodic_distance(left: np.ndarray, right: np.ndarray) -> float:
    diff = np.abs(left - right)
    diff = np.minimum(diff, DOMAIN_LENGTH - diff)
    return float(np.linalg.norm(diff))


def _missing_membership_case_rows(results_dir: Path, *, case_filter: set[str] | None = None) -> list[dict[str, Any]]:
    if case_filter and "c185" not in {str(value).lower() for value in case_filter}:
        return []
    c185_arr = results_dir / "arr_deficit_attribution_audit_c185_final81_summary.csv"
    if not c185_arr.exists():
        return []
    df = pd.read_csv(c185_arr)
    if df.empty:
        return []
    row = df.iloc[0]
    return [
        {
            "attack_family": "missing_membership",
            "ablation_mode": np.nan,
            "phi_env_scale": np.nan,
            "phi_act_scale": np.nan,
            "case": "c185",
            "candidate_id": str(row.get("candidate_id", "lambda_frac0p2_omega_top2:ti7:c185:lag4")),
            "z0_tag": _z0_tag(str(row.get("candidate_id", "lambda_frac0p2_omega_top2:ti7:c185:lag4"))),
            "radius_dx": np.nan,
            "M": np.nan,
            "clone_count_per_tube": np.nan,
            "jitter_strength": np.nan,
            "jitter_spatial_frac": np.nan,
            "jitter_time_frac": np.nan,
            "delta": np.nan,
            "natural_tubes": np.nan,
            "pre_quotient_tubes": np.nan,
            "post_quotient_tubes": np.nan,
            "quotient_merged_fraction": np.nan,
            "post_growth_fraction": np.nan,
            "natural_source": np.nan,
            "post_quotient_source": np.nan,
            "eta_post": np.nan,
            "raw_split_pair_pack_proxy": np.nan,
            "R_pack_star": np.nan,
            "R_shape_fam_available": np.nan,
            "R_spread_fam_available": np.nan,
            "R_renew_fam_available": _safe_float(row.get("renewal_capacity_exposure_corrected", np.nan)),
            "R_D_available": np.nan,
            "R_tail_available": _safe_float(row.get("R_tail", np.nan)),
            "E_nu_coh_pos_available": np.nan,
            "Fphys_star_available": np.nan,
            "natural_N_eff_2": np.nan,
            "natural_N_eff_half": np.nan,
            "theta_proxy": np.nan,
            "C0theta_proxy": np.nan,
            "nonredundant_pair_fraction": np.nan,
            "sampled_phi_pairs": np.nan,
            "attack_status": "not_run_missing_promoted_membership",
            "missing_terms": "promoted_tube_family_membership;physical_quotient_split_inputs",
            "evidence_note": "c185 retained as ARR/boundary absorption evidence only",
        }
    ]


def _write_plots(
    summary: pd.DataFrame,
    phi: pd.DataFrame,
    absorption: pd.DataFrame,
    *,
    results_dir: Path,
    c0: float,
    growth_cap: float,
    prefix: str,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    plot_paths = {
        "count": results_dir / f"{prefix}_count_vs_M.png",
        "fphys": results_dir / f"{prefix}_fphys_vs_M.png",
        "eta": results_dir / f"{prefix}_eta_vs_M.png",
        "growth_jitter": results_dir / f"{prefix}_post_growth_vs_jitter.png",
        "theta": results_dir / f"{prefix}_theta_vs_r.png",
        "phi": results_dir / f"{prefix}_phi_distribution.png",
        "phi_k": results_dir / f"{prefix}_phi_k_distribution.png",
        "composite": results_dir / f"{prefix}_results_figure.png",
    }

    run = summary[pd.to_numeric(summary["M"], errors="coerce").notna()].copy() if not summary.empty else pd.DataFrame()
    if not run.empty:
        run["M"] = pd.to_numeric(run["M"])
        run["radius_dx"] = pd.to_numeric(run["radius_dx"])
        run["Fphys_star_available"] = pd.to_numeric(run["Fphys_star_available"])
        run["eta_post"] = pd.to_numeric(run["eta_post"])
        run["natural_tubes"] = pd.to_numeric(run["natural_tubes"], errors="coerce")
        run["pre_quotient_tubes"] = pd.to_numeric(run["pre_quotient_tubes"], errors="coerce")
        run["post_quotient_tubes"] = pd.to_numeric(run["post_quotient_tubes"], errors="coerce")
        run["post_growth_fraction"] = pd.to_numeric(run["post_growth_fraction"], errors="coerce")
        if "jitter_strength" not in run.columns:
            run["jitter_strength"] = 0.0
        run["jitter_strength"] = pd.to_numeric(run["jitter_strength"], errors="coerce").fillna(0.0)
        if "ablation_mode" not in run.columns:
            run["ablation_mode"] = "none"
        run["ablation_mode"] = run["ablation_mode"].fillna("none").astype(str)
        if "phi_env_scale" not in run.columns:
            run["phi_env_scale"] = 1.0
        if "phi_act_scale" not in run.columns:
            run["phi_act_scale"] = 1.0
        if "phi_def_scale" not in run.columns:
            run["phi_def_scale"] = 1.0
        if "shape_degenerate_enabled" not in run.columns:
            run["shape_degenerate_enabled"] = False
        if "shape_degenerate_strength" not in run.columns:
            run["shape_degenerate_strength"] = 0.0
        run["phi_env_scale"] = pd.to_numeric(run["phi_env_scale"], errors="coerce").fillna(1.0)
        run["phi_act_scale"] = pd.to_numeric(run["phi_act_scale"], errors="coerce").fillna(1.0)
        run["phi_def_scale"] = pd.to_numeric(run["phi_def_scale"], errors="coerce").fillna(1.0)
        run["shape_degenerate_enabled"] = run["shape_degenerate_enabled"].astype(str).str.lower().isin({"true", "1"})
        run["shape_degenerate_strength"] = pd.to_numeric(run["shape_degenerate_strength"], errors="coerce").fillna(0.0)
        run["count_growth_ratio"] = run["post_quotient_tubes"] / run["natural_tubes"].clip(lower=1.0)
        run["pre_growth_ratio"] = run["pre_quotient_tubes"] / run["natural_tubes"].clip(lower=1.0)

    def group_label(part: pd.DataFrame) -> str:
        row = part.iloc[0]
        label = f"{row['case']} r={float(row['radius_dx']):g}dx"
        jitter = float(row.get("jitter_strength", 0.0))
        if jitter > 0:
            label += f" jit={jitter:g}"
        mode = str(row.get("ablation_mode", "none"))
        if mode not in {"", "none"}:
            label += f" {mode}"
        label += f" env={float(row.get('phi_env_scale', 1.0)):g}"
        label += f" act={float(row.get('phi_act_scale', 1.0)):g}"
        label += f" def={float(row.get('phi_def_scale', 1.0)):g}"
        if bool(row.get("shape_degenerate_enabled", False)):
            label += f" shape={float(row.get('shape_degenerate_strength', 0.0)):g}"
        return label

    plt.figure(figsize=(6.4, 4.2))
    if not run.empty:
        for _, part in run.groupby([
            "case",
            "radius_dx",
            "jitter_strength",
            "ablation_mode",
            "phi_env_scale",
            "phi_act_scale",
            "phi_def_scale",
            "shape_degenerate_enabled",
            "shape_degenerate_strength",
        ], sort=True):
            part = part.sort_values("M")
            plt.plot(part["M"], part["count_growth_ratio"], marker="o", label=group_label(part))
    plt.xscale("log", base=2)
    plt.xlabel("clone count per natural tube M")
    plt.ylabel("post-quotient tube count / natural")
    plt.title("Tube Count Growth Under Attack")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(plot_paths["count"], dpi=180)
    plt.close()

    plt.figure(figsize=(6.4, 4.2))
    if not run.empty:
        for _, part in run.groupby([
            "case",
            "radius_dx",
            "jitter_strength",
            "ablation_mode",
            "phi_env_scale",
            "phi_act_scale",
            "phi_def_scale",
            "shape_degenerate_enabled",
            "shape_degenerate_strength",
        ], sort=True):
            part = part.sort_values("M")
            plt.plot(part["M"], part["Fphys_star_available"], marker="o", label=group_label(part))
    plt.xscale("log", base=2)
    plt.xlabel("clone count per natural tube M")
    plt.ylabel("available post-quotient Fphys* proxy")
    plt.title("Physical cost after quotient")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(plot_paths["fphys"], dpi=180)
    plt.close()

    plt.figure(figsize=(6.4, 4.2))
    if not run.empty:
        for _, part in run.groupby([
            "case",
            "radius_dx",
            "jitter_strength",
            "ablation_mode",
            "phi_env_scale",
            "phi_act_scale",
            "phi_def_scale",
            "shape_degenerate_enabled",
            "shape_degenerate_strength",
        ], sort=True):
            part = part.sort_values("M")
            plt.plot(part["M"], part["eta_post"], marker="o", label=group_label(part))
    plt.xscale("log", base=2)
    plt.ylim(-0.05, 1.05)
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1, label="eta=0.5")
    plt.xlabel("clone count per natural tube M")
    plt.ylabel("post-quotient captured fraction eta")
    plt.title("Capture under adversarial split")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(plot_paths["eta"], dpi=180)
    plt.close()

    plt.figure(figsize=(6.6, 4.4))
    if not run.empty:
        subset = run[(run["M"] > 1) & (run["jitter_strength"] > 0)]
        if not subset.empty:
            for _, part in subset.groupby([
                "case",
                "radius_dx",
                "M",
                "ablation_mode",
                "phi_env_scale",
                "phi_act_scale",
                "phi_def_scale",
                "shape_degenerate_enabled",
                "shape_degenerate_strength",
            ], sort=True):
                part = part.sort_values("jitter_strength")
                row = part.iloc[0]
                label = (
                    f"{row['case']} r={float(row['radius_dx']):g}dx M={int(row['M'])} "
                    f"env={float(row['phi_env_scale']):g} act={float(row['phi_act_scale']):g} def={float(row['phi_def_scale']):g}"
                )
                if bool(row.get("shape_degenerate_enabled", False)):
                    label += f" shape={float(row.get('shape_degenerate_strength', 0.0)):g}"
                plt.plot(part["jitter_strength"], part["post_growth_fraction"], marker="o", label=label)
    plt.axhline(float(growth_cap), color="black", linestyle="--", linewidth=1, label=f"showcase cap={float(growth_cap):g}")
    plt.xlabel("jitter strength scale")
    plt.ylabel("post-growth fraction (post-natural)/natural")
    plt.title("Post-Growth Boundary Versus Jitter (Fixed M)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(plot_paths["growth_jitter"], dpi=180)
    plt.close()

    plt.figure(figsize=(6.4, 4.2))
    if not absorption.empty:
        data = absorption.copy()
        data["radius_dx"] = pd.to_numeric(data["radius_dx"], errors="coerce")
        data["theta_proxy"] = pd.to_numeric(data["theta_proxy"], errors="coerce")
        data["C0theta_proxy"] = data["theta_proxy"] * float(c0)
        finite = data[np.isfinite(data["radius_dx"]) & np.isfinite(data["theta_proxy"])]
        for case, part in finite.groupby("case", sort=True):
            part = part.sort_values("radius_dx")
            plt.plot(part["radius_dx"], part["theta_proxy"], marker="o", label=f"{case} theta")
            plt.plot(part["radius_dx"], part["C0theta_proxy"], marker="x", linestyle="--", label=f"{case} C0theta")
        c185 = data[(data["case"].astype(str).eq("c185")) & (~np.isfinite(data["radius_dx"])) & np.isfinite(data["theta_proxy"])]
        if not c185.empty:
            plt.scatter([0.0], [float(c185["theta_proxy"].iloc[0])], marker="s", label="c185 ARR proxy")
    plt.axhline(1.0, color="black", linestyle=":", linewidth=1, label="loop threshold 1")
    plt.xlabel("radius in dx units")
    plt.ylabel("theta proxy")
    plt.title("Available absorption proxy")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_paths["theta"], dpi=180)
    plt.close()

    plt.figure(figsize=(6.4, 4.2))
    if not phi.empty:
        frame = phi.copy()
        if "d_phys_proxy" in frame.columns:
            frame["d_phys_proxy"] = pd.to_numeric(frame["d_phys_proxy"], errors="coerce")
            frame = frame[np.isfinite(frame["d_phys_proxy"])]
            if not frame.empty and "pair_group" in frame.columns:
                for group, part in frame.groupby("pair_group", sort=True):
                    values = part["d_phys_proxy"].to_numpy(dtype=float)
                    if values.size:
                        clip_high = np.nanpercentile(values, 99)
                        plt.hist(np.clip(values, 0.0, clip_high), bins=30, alpha=0.45, label=str(group))
            elif not frame.empty:
                values = frame["d_phys_proxy"].to_numpy(dtype=float)
                clip_high = np.nanpercentile(values, 99)
                plt.hist(np.clip(values, 0.0, clip_high), bins=30)
        else:
            values = pd.to_numeric(frame["Phi_env_centroid_over_r"], errors="coerce").dropna()
            if not values.empty:
                plt.hist(np.clip(values.to_numpy(dtype=float), 0.0, np.nanpercentile(values, 99)), bins=30)
    plt.xlabel("pair d_phys proxy")
    plt.ylabel("count")
    plt.title("Pairwise Physical Distance Distribution")
    plt.grid(True, alpha=0.3)
    if not phi.empty and "pair_group" in phi.columns:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_paths["phi"], dpi=180)
    plt.close()

    if not phi.empty and {"Phi_env_centroid_over_r", "Phi_anc_proxy", "Phi_act_proxy", "Phi_def_proxy"}.issubset(phi.columns):
        fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.2))
        coord_cols = [
            ("Phi_env_centroid_over_r", "Phi_env"),
            ("Phi_anc_proxy", "Phi_anc"),
            ("Phi_act_proxy", "Phi_act"),
            ("Phi_def_proxy", "Phi_def"),
        ]
        frame = phi.copy()
        if "survives_quotient" in frame.columns:
            frame = frame[frame["survives_quotient"].astype(bool)]
        for axis, (column, title) in zip(axes.ravel(), coord_cols):
            data = frame.copy()
            data[column] = pd.to_numeric(data[column], errors="coerce")
            data = data[np.isfinite(data[column])]
            if not data.empty and "pair_group" in data.columns:
                for group, part in data.groupby("pair_group", sort=True):
                    values = part[column].to_numpy(dtype=float)
                    if values.size:
                        high = np.nanpercentile(values, 99)
                        axis.hist(np.clip(values, 0.0, high), bins=24, alpha=0.45, label=str(group))
            elif not data.empty:
                values = data[column].to_numpy(dtype=float)
                high = np.nanpercentile(values, 99)
                axis.hist(np.clip(values, 0.0, high), bins=24)
            axis.set_title(title)
            axis.grid(True, alpha=0.25)
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plot_paths["phi_k"], dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.8))
    if not run.empty:
        for _, part in run.groupby([
            "case",
            "radius_dx",
            "jitter_strength",
            "ablation_mode",
            "phi_env_scale",
            "phi_act_scale",
            "phi_def_scale",
            "shape_degenerate_enabled",
            "shape_degenerate_strength",
        ], sort=True):
            part = part.sort_values("M")
            label = group_label(part)
            axes[0, 0].plot(part["M"], part["count_growth_ratio"], marker="o", label=label)
            axes[0, 1].plot(part["M"], part["Fphys_star_available"], marker="o", label=label)
            axes[1, 0].plot(part["M"], part["eta_post"], marker="o", label=label)
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xscale("log", base=2)
        axis.grid(True, alpha=0.25)
    axes[0, 0].set_title("Tube Count Growth")
    axes[0, 0].set_xlabel("M")
    axes[0, 0].set_ylabel("post/natural")
    axes[0, 1].set_title("Physical Cost Proxy")
    axes[0, 1].set_xlabel("M")
    axes[0, 1].set_ylabel("Fphys*")
    axes[1, 0].set_title("Captured Fraction")
    axes[1, 0].set_xlabel("M")
    axes[1, 0].set_ylabel("eta")
    axes[1, 0].set_ylim(-0.05, 1.05)

    if not absorption.empty:
        data = absorption.copy()
        data["radius_dx"] = pd.to_numeric(data["radius_dx"], errors="coerce")
        data["theta_proxy"] = pd.to_numeric(data["theta_proxy"], errors="coerce")
        data["C0theta_proxy"] = data["theta_proxy"] * float(c0)
        finite = data[np.isfinite(data["radius_dx"]) & np.isfinite(data["theta_proxy"])]
        for case, part in finite.groupby("case", sort=True):
            part = part.sort_values("radius_dx")
            axes[1, 1].plot(part["radius_dx"], part["theta_proxy"], marker="o", label=f"{case} theta")
            axes[1, 1].plot(part["radius_dx"], part["C0theta_proxy"], marker="x", linestyle="--", label=f"{case} C0theta")
    axes[1, 1].axhline(1.0, color="black", linestyle=":", linewidth=1)
    axes[1, 1].set_title("Theta And C0theta")
    axes[1, 1].set_xlabel("radius in dx")
    axes[1, 1].set_ylabel("theta")
    axes[1, 1].grid(True, alpha=0.25)

    axes[0, 0].legend(fontsize=6)
    axes[1, 1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(plot_paths["composite"], dpi=180)
    plt.close(fig)

    return plot_paths


def _write_status_tex(summary: pd.DataFrame, path: Path) -> None:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    counts = summary["attack_status"].value_counts(dropna=False).to_dict() if not summary.empty else {}
    lines = [
        "% Auto-generated by near_degenerate_multi_tube_packing_attack.py",
        f"% generated_utc={generated}",
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Attack status & Count\\\\",
        "\\midrule",
    ]
    for key, value in sorted(counts.items()):
        lines.append(f"{_latex_escape(str(key))} & {int(value)}\\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_report_tex(
    *,
    summary: pd.DataFrame,
    phi: pd.DataFrame,
    absorption: pd.DataFrame,
    summary_csv: Path,
    phi_csv: Path,
    absorption_csv: Path,
    plot_paths: dict[str, Path],
    report_tex: Path,
    status_tex: Path,
    delta: float,
    c0: float,
) -> None:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run = summary[pd.to_numeric(summary["M"], errors="coerce").notna()].copy() if not summary.empty else pd.DataFrame()
    if not run.empty:
        run["M"] = pd.to_numeric(run["M"])
        run["radius_dx"] = pd.to_numeric(run["radius_dx"])
        run["eta_post"] = pd.to_numeric(run["eta_post"])
        run["Fphys_star_available"] = pd.to_numeric(run["Fphys_star_available"])
        run["post_growth_fraction"] = pd.to_numeric(run["post_growth_fraction"], errors="coerce")
        if "ablation_mode" not in run.columns:
            run["ablation_mode"] = "none"
        run["ablation_mode"] = run["ablation_mode"].fillna("none").astype(str)
        if "phi_env_scale" not in run.columns:
            run["phi_env_scale"] = 1.0
        if "phi_act_scale" not in run.columns:
            run["phi_act_scale"] = 1.0
        run["phi_env_scale"] = pd.to_numeric(run["phi_env_scale"], errors="coerce").fillna(1.0)
        run["phi_act_scale"] = pd.to_numeric(run["phi_act_scale"], errors="coerce").fillna(1.0)
    status_counts = summary["attack_status"].value_counts(dropna=False).to_dict() if not summary.empty else {}

    attack_families = (
        {str(value) for value in summary.get("attack_family", pd.Series(dtype=str)).dropna().astype(str)}
        if not summary.empty
        else set()
    )
    is_cross_parent = bool({"cross_parent_jitter", "renewal_cascade_jitter"} & attack_families)
    is_renewal_cascade = "renewal_cascade_jitter" in attack_families
    multiplicities = sorted({int(v) for v in run["M"].dropna().to_list()}) if not run.empty else []
    multiplicity_text = "{" + ",".join(str(v) for v in multiplicities) + "}" if multiplicities else "{1}"
    jitter_values = sorted(
        {
            float(v)
            for v in pd.to_numeric(run.get("jitter_strength", pd.Series(dtype=float)), errors="coerce").dropna().to_list()
            if float(v) > 0.0
        }
    ) if not run.empty else []
    jitter_text = "{" + ",".join(f"{v:g}" for v in jitter_values) + "}" if jitter_values else "{0}"
    ablation_modes = sorted({str(v) for v in run.get("ablation_mode", pd.Series(dtype=str)).dropna().astype(str) if str(v) not in {"", "none"}}) if not run.empty else []
    ablation_text = ", ".join(ablation_modes) if ablation_modes else "none"
    title_text = (
        (
            "Real Renewal-Cascade + Targeted Phi_act Weakening Attack on Cached JHTDB Families"
            if is_renewal_cascade
            else "Cross-Parent Near-Degenerate Jitter Attack on Cached JHTDB Families"
        )
        if is_cross_parent
        else "Near-Degenerate Multi-Tube Packing Attack on Cached JHTDB Families"
    )

    if is_cross_parent:
        executive_paragraph = (
            (
                "The renewal-cascade cross-parent jitter attack was run on cached promoted "
                "families with maximal temporal-slab separation enforced before jittering. "
                if is_renewal_cascade
                else "The cross-parent near-degenerate jitter attack was run on cached promoted "
            )
            +
            "tube-family memberships for c413 and c309 at the available radii "
            "$0.1\\Delta x$ and $0.25\\Delta x$.  For multiplicities "
            f"$M\\in{multiplicity_text}$ and jitter strengths "
            f"$s\\in{jitter_text}$ (with fixed $\\delta={delta:g}$, fixed "
            "ledger weights, and fixed $\\Phi_k$ proxies), capture remains high "
            "while post-quotient tube growth is consistently above the showcase cap."
        )
        executive_note = (
            "This run is therefore a stress-break on count growth (flagged as "
            "\\texttt{break\\_excess\\_post\\_growth}), not a low-cost collapse: "
            "$\\mathcal F^*_{\\rm avail}$ stays positive and $C_0\\theta<1$ in all "
            "completed c413/c309 rows. "
            f"Ablation mode(s): {ablation_text}."
        )
        phi_caption = (
            "Physical-distance proxy distribution with explicit same-parent zero baseline "
            "and cross-parent jitter pairs."
        )
    else:
        executive_paragraph = (
            "The adversarial label-splitting attack was run on the cached promoted "
            "tube-family memberships for c413 and c309 at the two available "
            "resolved radii, $0.1\\Delta x$ and $0.25\\Delta x$.  For every "
            f"multiplicity $M\\in{multiplicity_text}$ the same-parent "
            "sub-tubes are identified by the physical quotient at nominal "
            f"$\\delta={delta:g}$, so the post-quotient tube count returns to "
            "the natural count and no non-redundant packing branch remains.  "
            "Captured source is conserved by construction, $\\eta=1$, because "
            "the split only relabels the cached promoted material support."
        )
        executive_note = (
            "This is therefore a quotient-redundancy showcase, not a proof-certificate "
            "for the full continuum denominator.  Missing continuum channels are "
            "listed explicitly rather than filled by proxy."
        )
        phi_caption = (
            "Natural cross-parent coordinate separation proxy. Same-parent split-clone "
            "pairs have zero proxy coordinates and are merged."
        )

    lines: list[str] = []
    lines.extend(
        [
            "% Auto-generated by near_degenerate_multi_tube_packing_attack.py",
            f"% generated_utc={generated}",
            "\\documentclass[11pt]{article}",
            "\\usepackage[margin=1in]{geometry}",
            "\\usepackage{booktabs}",
            "\\usepackage{graphicx}",
            "\\usepackage{hyperref}",
            "\\usepackage{amsmath,amssymb}",
            f"\\title{{{title_text}}}",
            "\\author{Automated offline audit}",
            "\\date{" + generated + "}",
            "\\begin{document}",
            "\\maketitle",
            "",
            "\\section*{Executive finding}",
            executive_paragraph,
            "",
            executive_note,
            "",
            "\\section*{Status counts}",
            f"\\input{{{_tex_path(status_tex, base=report_tex.parent)}}}",
            "",
            "\\section*{Attack table}",
            "\\begin{tabular}{llrrrrrr}",
            "\\toprule",
            "Case & $r/\\Delta x$ & $M$ & $s$ & pre & post & $\\eta$ & $\\mathcal F^*_{\\rm avail}$\\\\",
            "\\midrule",
        ]
    )
    if not run.empty:
        for _, row in run.sort_values(["case", "radius_dx", "M", "jitter_strength", "phi_env_scale", "phi_act_scale"]).iterrows():
            lines.append(
                "{} & {:.2g} & {} & {:.3g} & {} & {} & {:.3f} & {:.3f}\\\\".format(
                    _latex_escape(str(row["case"])),
                    float(row["radius_dx"]),
                    int(row["M"]),
                    float(row.get("jitter_strength", 0.0)),
                    int(row["pre_quotient_tubes"]),
                    int(row["post_quotient_tubes"]),
                    float(row["eta_post"]),
                    float(row["Fphys_star_available"]),
                )
            )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "",
            "\\section*{Plots}",
            "\\begin{figure}[h]",
            "\\centering",
            f"\\includegraphics[width=0.82\\linewidth]{{{_tex_path(plot_paths['fphys'], base=report_tex.parent)}}}",
            "\\caption{Available post-quotient physical cost proxy versus splitting multiplicity.}",
            "\\end{figure}",
            "\\begin{figure}[h]",
            "\\centering",
            f"\\includegraphics[width=0.82\\linewidth]{{{_tex_path(plot_paths['eta'], base=report_tex.parent)}}}",
            "\\caption{Captured source fraction after quotient.}",
            "\\end{figure}",
            "\\begin{figure}[h]",
            "\\centering",
            f"\\includegraphics[width=0.82\\linewidth]{{{_tex_path(plot_paths['growth_jitter'], base=report_tex.parent)}}}",
            "\\caption{Post-growth fraction versus jitter strength at fixed multiplicity $M$, used to isolate the break boundary.}",
            "\\end{figure}",
            "\\begin{figure}[h]",
            "\\centering",
            f"\\includegraphics[width=0.82\\linewidth]{{{_tex_path(plot_paths['theta'], base=report_tex.parent)}}}",
            f"\\caption{{Available absorption proxy $\\theta$ and $C_0\\theta$ with $C_0={c0:g}$.}}",
            "\\end{figure}",
            "\\begin{figure}[h]",
            "\\centering",
            f"\\includegraphics[width=0.82\\linewidth]{{{_tex_path(plot_paths['phi'], base=report_tex.parent)}}}",
            f"\\caption{{{phi_caption}}}",
            "\\end{figure}",
            "",
            "\\section*{Ledger availability}",
            "The available post-quotient cost proxy is",
            "\\[",
            "\\mathcal F^*_{\\rm avail}=R^*_{\\rm pack}+R_{\\rm shape}^{\\rm fam}+R_{\\rm spread}^{\\rm fam}+R_{\\rm renew}^{\\rm fam}+R_D+R_{\\rm tail}^{\\rm proxy}+\\mathcal E_{\\nu}^{\\rm coh,+,proxy}.",
            "\\]",
            "The cached real-data lane does not contain full continuum evaluations of "
            "$R_{\\rm boundary}^{\\rm fam}$, full $R_{\\rm tail}$, $\\mathcal E_\\omega$, "
            "$B_\\omega$, $\\mathcal D_{\\rm ARR}$ for these split memberships, or the "
            "full structural $\\Phi_{\\rm def},\\Phi_{\\rm bdry},\\Phi_{\\rm res}$ fields.  "
            "Rows in the CSV retain these omissions in the \\texttt{missing\\_terms} column.",
            "",
            "\\section*{Absorption proxy}",
            "For c413/c309 the plotted $\\theta$ values are the cached targeted-dichotomy "
            "ratio $G/\\texttt{ledger\\_total\\_C1}$ at available radii/stages.  For c185, "
            "which has no promoted membership table to split, the plotted point is the "
            "ARR ratio with renewal exposure correction.  These are routed diagnostics, "
            "not full $P/\\widetilde{\\mathcal D}_\\omega$ certificates.",
            "",
        ]
    )
    if not absorption.empty:
        lines.extend(["\\begin{tabular}{llrl}", "\\toprule", "Case & Stage & $r/\\Delta x$ & $\\theta$ proxy\\\\", "\\midrule"])
        temp = absorption.copy()
        temp["radius_dx"] = pd.to_numeric(temp["radius_dx"], errors="coerce")
        temp["theta_proxy"] = pd.to_numeric(temp["theta_proxy"], errors="coerce")
        for _, row in temp.iterrows():
            r_text = "--" if not math.isfinite(float(row["radius_dx"])) else f"{float(row['radius_dx']):.2g}"
            theta_text = "--" if not math.isfinite(float(row["theta_proxy"])) else f"{float(row['theta_proxy']):.3f}"
            lines.append(f"{_latex_escape(str(row['case']))} & {_latex_escape(str(row['stage']))} & {r_text} & {theta_text}\\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}", ""])

    lines.extend(
        [
            "\\section*{Artifacts}",
            "\\begin{itemize}",
            f"\\item Summary CSV: \\texttt{{{_latex_escape(_display_path(summary_csv))}}}",
            f"\\item Pair-coordinate diagnostics: \\texttt{{{_latex_escape(_display_path(phi_csv))}}}",
            f"\\item Absorption proxy CSV: \\texttt{{{_latex_escape(_display_path(absorption_csv))}}}",
            "\\end{itemize}",
            "",
            "\\section*{Conclusion}",
            (
                "Cross-parent jitter stress confirms a strong empirical pattern on cached "
                "c413/c309 families: capture remains high and available physical cost stays "
                "positive, while break outcomes are driven by post-quotient count growth.  "
                "Coordinate-ablation reruns isolate sensitivity to selected $\\Phi_k$ weights "
                "without changing ledger formulas or $\\delta$."
                if is_cross_parent
                else "The near-duplicate split attack did not expose a high-capture, low-cost "
                "non-redundant packing case in the cached c413/c309 promoted families.  "
                "It instead exercised the intended upstream resolution: artificial "
                "same-parent multiplicity is quotient-redundant.  The remaining open "
                "work is not this label-split branch, but a full continuum denominator "
                "run for genuinely non-merged promoted families and a promoted c185 "
                "membership table if c185 is to be attacked in the same quotient lane."
            ),
            "\\end{document}",
            "",
        ]
    )
    report_tex.write_text("\n".join(lines), encoding="utf-8")


def _neff2(masses: np.ndarray) -> float:
    masses = np.asarray(masses, dtype=float)
    total = float(np.sum(np.clip(masses, 0.0, None)))
    denom = float(np.sum(np.square(np.clip(masses, 0.0, None))))
    return 0.0 if total <= 0.0 or denom <= 0.0 else (total * total) / denom


def _neff_half(masses: np.ndarray) -> float:
    masses = np.asarray(masses, dtype=float)
    total = float(np.sum(np.clip(masses, 0.0, None)))
    if total <= 0.0:
        return 0.0
    p = np.clip(masses, 0.0, None) / total
    return float(np.square(np.sum(np.sqrt(p))))


def _positive_or_proxy(primary: Any, fallback: Any) -> float:
    primary_value = _safe_float(primary)
    if math.isfinite(primary_value) and primary_value > 0.0:
        return primary_value
    fallback_value = _safe_float(fallback)
    if math.isfinite(fallback_value) and fallback_value > 0.0:
        return fallback_value
    return 0.0


def _case_from_candidate(candidate_id: str) -> str:
    match = re.search(r"(c\d+)", str(candidate_id))
    if match:
        return match.group(1)
    return ""


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "_": r"\_",
        "#": r"\#",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def _display_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _tex_path(path: Path, *, base: Path | None = None) -> str:
    if base is not None:
        try:
            return str(path.resolve().relative_to(base.resolve())).replace("\\", "/")
        except ValueError:
            try:
                return str(Path("..") / path.resolve().relative_to(base.resolve().parent)).replace("\\", "/")
            except ValueError:
                return str(_relative_path(path.resolve(), base.resolve())).replace("\\", "/")
    return str(path.resolve()).replace("\\", "/")


def _relative_path(path: Path, base: Path) -> Path:
    path_parts = path.resolve().parts
    base_parts = base.resolve().parts
    common = 0
    for left, right in zip(path_parts, base_parts):
        if left.lower() != right.lower():
            break
        common += 1
    return Path(*([".."] * (len(base_parts) - common)), *path_parts[common:])


if __name__ == "__main__":
    main()
