from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = DATA / "results"
OUTPUTS = ROOT / "outputs"


@dataclass(frozen=True)
class ArtifactSet:
    name: str
    csvs: tuple[Path, ...]
    figures: tuple[Path, ...] = ()


NEAR_DEGENERATE = ArtifactSet(
    name="same-parent quotient redundancy",
    csvs=(
        RESULTS / "near_degenerate_multi_tube_packing_attack_summary.csv",
        RESULTS / "near_degenerate_multi_tube_packing_attack_absorption_proxy.csv",
    ),
    figures=(
        RESULTS / "near_degenerate_attack_eta_vs_M.png",
        RESULTS / "near_degenerate_attack_fphys_vs_M.png",
        RESULTS / "near_degenerate_attack_results_figure.png",
    ),
)

RENEWAL_CASCADE = ArtifactSet(
    name="renewal cascade jitter",
    csvs=(
        RESULTS / "renewal_cascade_shape_topsource_phi_env_act_joint_hammer_r0p1_summary.csv",
        RESULTS / "renewal_cascade_shape_topsource_phi_env_act_joint_hammer_r0p1_absorption_proxy.csv",
    ),
    figures=(
        RESULTS / "renewal_cascade_shape_topsource_phi_env_act_joint_hammer_r0p1_results_figure.png",
        RESULTS / "renewal_cascade_shape_topsource_phi_env_act_joint_hammer_r0p1_post_growth_vs_jitter.png",
        RESULTS / "renewal_cascade_shape_topsource_phi_env_act_joint_hammer_r0p1_fphys_vs_M.png",
    ),
)

TAIL_LADDER = ArtifactSet(
    name="tail denominator ladder",
    csvs=(
        RESULTS / "tail_denominator_ladder_audit_top2_top3_rows.csv",
        RESULTS / "tail_denominator_ladder_audit_top2_top3_summary.csv",
    ),
)

ARR_DEFICIT = ArtifactSet(
    name="ARR deficit attribution",
    csvs=(RESULTS / "arr_deficit_attribution_audit_c185_final81_summary.csv",),
)

COHERENT_RESIDUAL = ArtifactSet(
    name="coherent residual attribution",
    csvs=(RESULTS / "coherent_viscous_residual_attribution_top2_c413.csv",),
)

ALL_ARTIFACTS = (NEAR_DEGENERATE, RENEWAL_CASCADE, TAIL_LADDER, ARR_DEFICIT, COHERENT_RESIDUAL)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def artifact_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for artifact in ALL_ARTIFACTS:
        for kind, paths in (("csv", artifact.csvs), ("figure", artifact.figures)):
            for path in paths:
                rows.append(
                    {
                        "demo": artifact.name,
                        "kind": kind,
                        "path": rel(path),
                        "exists": path.exists(),
                        "bytes": path.stat().st_size if path.exists() else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def data_available() -> bool:
    required = [path for artifact in ALL_ARTIFACTS for path in artifact.csvs]
    return all(path.exists() for path in required)


def require_cached_data() -> None:
    missing = artifact_table()
    missing = missing[(missing["kind"].eq("csv")) & (~missing["exists"])]
    if not missing.empty:
        raise FileNotFoundError(
            "Cached evidence CSVs are not bundled. Place the release artifacts in public/data/results.\n\n"
            + missing[["demo", "path"]].to_string(index=False)
        )


def read_csv(path: Path) -> pd.DataFrame:
    require_cached_data()
    return pd.read_csv(path)


def same_parent_summary(mode: str = "cached") -> pd.DataFrame:
    if mode in {"synthetic", "fixture"}:
        return _fixture_same_parent()
    df = read_csv(NEAR_DEGENERATE.csvs[0])
    return (
        df.groupby(["case", "radius_dx", "M"], as_index=False)
        .agg(
            pre_quotient_tubes=("pre_quotient_tubes", "max"),
            post_quotient_tubes=("post_quotient_tubes", "max"),
            eta_post=("eta_post", "min"),
            R_pack_star=("R_pack_star", "max"),
            Fphys_star_available=("Fphys_star_available", "max"),
            attack_status=("attack_status", lambda values: ",".join(sorted(set(map(str, values))))),
        )
        .sort_values(["case", "radius_dx", "M"])
    )


def same_parent_scorecard(mode: str = "cached") -> dict[str, object]:
    rows = same_parent_summary(mode)
    split = rows[rows["M"] > 1]
    return {
        "rows": int(len(rows)),
        "all_split_rows_merge_back": bool((split["post_quotient_tubes"] < split["pre_quotient_tubes"]).all()),
        "eta_min": float(rows["eta_post"].min()),
        "max_R_pack_star_after_quotient": float(split["R_pack_star"].max()),
        "statuses": ", ".join(sorted(set(rows["attack_status"].astype(str)))),
    }


def same_parent_ledger(mode: str = "cached") -> pd.DataFrame:
    rows = same_parent_summary(mode).copy()
    rows["created_labels"] = rows["pre_quotient_tubes"] - rows["post_quotient_tubes"]
    rows["collapse_fraction"] = np.where(
        rows["pre_quotient_tubes"].gt(0),
        rows["created_labels"] / rows["pre_quotient_tubes"],
        np.nan,
    )
    rows["free_amplification_survives"] = (
        rows["M"].gt(1)
        & rows["post_quotient_tubes"].gt(rows["pre_quotient_tubes"] * 0.5)
        & rows["R_pack_star"].gt(0)
    )
    return rows[
        [
            "case",
            "radius_dx",
            "M",
            "pre_quotient_tubes",
            "post_quotient_tubes",
            "created_labels",
            "collapse_fraction",
            "eta_post",
            "R_pack_star",
            "Fphys_star_available",
            "free_amplification_survives",
        ]
    ]


def same_parent_explanation(mode: str = "cached") -> pd.DataFrame:
    rows = same_parent_ledger(mode)
    split = rows[rows["M"] > 1]
    return pd.DataFrame(
        [
            {
                "calculation": "label inflation",
                "formula": "pre_quotient_tubes / post_quotient_tubes",
                "observed": float((split["pre_quotient_tubes"] / split["post_quotient_tubes"]).max()),
                "interpretation": "the attack creates many labels before quotienting",
            },
            {
                "calculation": "post-quotient residual packing",
                "formula": "max R_pack_star after quotient",
                "observed": float(split["R_pack_star"].max()),
                "interpretation": "same-parent copies do not survive as nonredundant packing",
            },
            {
                "calculation": "source capture",
                "formula": "min eta_post",
                "observed": float(split["eta_post"].min()),
                "interpretation": "source is retained while labels merge back",
            },
        ]
    )


def renewal_summary(mode: str = "cached") -> pd.DataFrame:
    if mode in {"synthetic", "fixture"}:
        return _fixture_renewal()
    return read_csv(RENEWAL_CASCADE.csvs[0])


def renewal_scorecard(mode: str = "cached") -> dict[str, object]:
    df = renewal_summary(mode)
    stressed = df[df["M"].astype(float) > 1].copy()
    return {
        "rows": int(len(df)),
        "stressed_rows": int(len(stressed)),
        "eta_min_stressed": float(stressed["eta_post"].min()),
        "post_growth_max": float(stressed["post_growth_fraction"].max()),
        "Fphys_min_stressed": float(stressed["Fphys_star_available"].min()),
        "C0theta_max": float(stressed["C0theta_proxy"].max()),
        "statuses": ", ".join(sorted(set(stressed["attack_status"].astype(str)))),
    }


def renewal_ledger(mode: str = "cached") -> pd.DataFrame:
    df = renewal_summary(mode).copy()
    df["M"] = df["M"].astype(float)
    df["pays_physical_charge"] = df["Fphys_star_available"].gt(0)
    df["passes_proxy_gap"] = df["C0theta_proxy"].lt(1)
    df["free_jitter_candidate"] = (
        df["M"].gt(1)
        & df["eta_post"].gt(0.9)
        & df["Fphys_star_available"].le(0)
        & df["R_pack_star"].gt(0)
    )
    cols = [
        "case",
        "ablation_mode",
        "M",
        "jitter_strength",
        "eta_post",
        "post_growth_fraction",
        "R_shape_fam_available",
        "R_spread_fam_available",
        "R_renew_fam_available",
        "Fphys_star_available",
        "C0theta_proxy",
        "pays_physical_charge",
        "passes_proxy_gap",
        "free_jitter_candidate",
    ]
    available = [col for col in cols if col in df.columns]
    return df[available].sort_values(["case", "ablation_mode", "M"]).reset_index(drop=True)


def renewal_explanation(mode: str = "cached") -> pd.DataFrame:
    rows = renewal_ledger(mode)
    stressed = rows[rows["M"] > 1]
    return pd.DataFrame(
        [
            {
                "calculation": "capture survives jitter",
                "formula": "min eta_post for M > 1",
                "observed": float(stressed["eta_post"].min()),
                "interpretation": "the perturbation keeps a large source fraction",
            },
            {
                "calculation": "physical ledger charge",
                "formula": "min Fphys_star_available for M > 1",
                "observed": float(stressed["Fphys_star_available"].min()),
                "interpretation": "the high-capture perturbation is not free",
            },
            {
                "calculation": "proxy loop margin",
                "formula": "max C0theta_proxy for M > 1",
                "observed": float(stressed["C0theta_proxy"].max()),
                "interpretation": "in these rows the proxy absorption coefficient stays below one",
            },
        ]
    )


def tail_ladder_rows(mode: str = "cached") -> pd.DataFrame:
    if mode in {"synthetic", "fixture"}:
        return _fixture_tail_rows()
    return read_csv(TAIL_LADDER.csvs[0])


def tail_ladder_scorecard(mode: str = "cached") -> dict[str, object]:
    rows = tail_ladder_rows(mode)
    if mode == "cached":
        summary = read_csv(TAIL_LADDER.csvs[1]).iloc[0]
        return {
            "rows": int(summary["rows"]),
            "best_denominator_counts": str(summary["best_denominator_counts"]),
            "best_ratio_min": float(summary["best_ratio_min"]),
            "best_ratio_median": float(summary["best_ratio_median"]),
            "best_ratio_max": float(summary["best_ratio_max"]),
            "unresolved_D3_rows": int(summary["unresolved_D3_rows"]),
            "unresolved_D6_rows": int(summary["unresolved_D6_rows"]),
            "all_lt_1": bool(summary["best_ratio_all_lt_1"]),
            "worst_candidate": str(rows.sort_values("best_ratio", ascending=False).iloc[0]["candidate_id"]),
        }
    return {
        "rows": int(len(rows)),
        "best_denominator_counts": ";".join(
            f"{name}={count}" for name, count in rows["best_denominator"].value_counts().sort_index().items()
        ),
        "best_ratio_min": float(rows["best_ratio"].min()),
        "best_ratio_median": float(rows["best_ratio"].median()),
        "best_ratio_max": float(rows["best_ratio"].max()),
        "unresolved_D3_rows": int(rows["ratio_D3"].isna().sum()),
        "unresolved_D6_rows": int(rows["ratio_D6"].isna().sum()),
        "all_lt_1": bool(rows["best_ratio"].lt(1.0).all()),
        "worst_candidate": str(rows.sort_values("best_ratio", ascending=False).iloc[0]["candidate_id"]),
    }


def tail_ladder_ledger(mode: str = "cached") -> pd.DataFrame:
    rows = tail_ladder_rows(mode).copy()
    ratio_cols = [col for col in rows.columns if col.startswith("ratio_D")]
    rows["available_denominators"] = rows[ratio_cols].notna().sum(axis=1)
    rows["best_ratio_recomputed"] = rows[ratio_cols].min(axis=1, skipna=True)
    rows["has_below_one_closure"] = rows["best_ratio_recomputed"].lt(1)
    rows["unresolved_family_count"] = rows[ratio_cols].isna().sum(axis=1)
    return rows[
        [
            "candidate_id",
            "support",
            "tail_sum_SK",
            "available_denominators",
            "unresolved_family_count",
            "best_denominator",
            "best_ratio",
            "best_ratio_recomputed",
            "has_below_one_closure",
            "denominator_family_pass",
        ]
    ].sort_values("best_ratio", ascending=False)


def tail_ladder_explanation(mode: str = "cached") -> pd.DataFrame:
    rows = tail_ladder_ledger(mode)
    return pd.DataFrame(
        [
            {
                "calculation": "best available denominator",
                "formula": "argmin_D ratio_D",
                "observed": str(rows["best_denominator"].value_counts().idxmax()),
                "interpretation": "the tail pressure consistently chooses the same least-bad tested row",
            },
            {
                "calculation": "best ratio range",
                "formula": "min/median/max best_ratio",
                "observed": f"{rows['best_ratio'].min():.3g} / {rows['best_ratio'].median():.3g} / {rows['best_ratio'].max():.3g}",
                "interpretation": "every tested ratio stays over budget; no below-one closure appears",
            },
            {
                "calculation": "unresolved rows",
                "formula": "count missing ratio_D entries",
                "observed": int(rows["unresolved_family_count"].sum()),
                "interpretation": "D3 and D6 stay visible as missing denominator families",
            },
        ]
    )


def arr_table(mode: str = "cached") -> pd.DataFrame:
    if mode in {"synthetic", "fixture"}:
        return _fixture_arr()
    return read_csv(ARR_DEFICIT.csvs[0])


def arr_scorecard(mode: str = "cached") -> dict[str, object]:
    row = arr_table(mode).iloc[0]
    return {
        "candidate": str(row["candidate_id"]),
        "support": int(row["support"]),
        "ratio_current": float(row["ratio_current"]),
        "ratio_with_renewal_exposure": float(row["ratio_with_renewal_exposure"]),
        "dominant_deficit_source": str(row["dominant_deficit_source"]),
        "renewal_fraction": float(row["renewal_fraction"]),
        "renewal_transition_count": int(row["renewal_transition_count"]),
    }


def arr_ledger(mode: str = "cached") -> pd.DataFrame:
    row = arr_table(mode).iloc[0].copy()
    demand = float(row["R_tail"])
    total = float(row["ARR_total_current"])
    deficit = float(row["deficit"])
    renewal_gain = float(row["renewal_capacity_exposure_corrected"]) - float(row["renewal_capacity_current"])
    repaired_capacity = total + renewal_gain
    return pd.DataFrame(
        [
            {
                "step": "current demand/capacity ratio",
                "formula": "R_tail / ARR_total_current",
                "numerator": demand,
                "denominator": total,
                "ratio": float(row["ratio_current"]),
                "interpretation": "slightly underpaid before naming the missing channel",
            },
            {
                "step": "deficit",
                "formula": "R_tail - ARR_total_current",
                "numerator": deficit,
                "denominator": float(row["R_tail"]),
                "ratio": float(row["deficit_fraction"]),
                "interpretation": "the gap is small but real",
            },
            {
                "step": "renewal-corrected demand/capacity ratio",
                "formula": "R_tail / (ARR_total_current + renewal_gain)",
                "numerator": demand,
                "denominator": repaired_capacity,
                "ratio": float(row["ratio_with_renewal_exposure"]),
                "interpretation": "the named renewal exposure pushes demand/capacity below one",
            },
        ]
    )


def coherent_table(mode: str = "cached") -> pd.DataFrame:
    if mode in {"synthetic", "fixture"}:
        return _fixture_coherent()
    return read_csv(COHERENT_RESIDUAL.csvs[0])


def coherent_scorecard(mode: str = "cached") -> dict[str, object]:
    df = coherent_table(mode)
    return {
        "rows": int(len(df)),
        "statuses": ", ".join(f"{k}={v}" for k, v in df["door4_status"].value_counts().sort_index().items()),
        "max_coherent_residual_ratio": float(df["coherent_residual_ratio"].max()),
        "min_Q": float(df["Q"].min()),
        "max_Q": float(df["Q"].max()),
    }


def coherent_ledger(mode: str = "cached") -> pd.DataFrame:
    df = coherent_table(mode).copy()
    df["net_fraction_of_positive"] = np.where(
        df["positive_residual_mass"].gt(0),
        df["signed_residual_net"] / df["positive_residual_mass"],
        np.nan,
    )
    return df[
        [
            "radius_dx",
            "positive_residual_mass",
            "negative_residual_mass",
            "signed_residual_net",
            "net_fraction_of_positive",
            "coherent_positive_residual",
            "damping_capacity",
            "renewal_deactivation_capacity",
            "coherent_residual_ratio",
            "door4_status",
            "Q",
        ]
    ]


def coherent_explanation(mode: str = "cached") -> pd.DataFrame:
    rows = coherent_ledger(mode)
    return pd.DataFrame(
        [
            {
                "calculation": "largest positive coherent ratio",
                "formula": "max coherent_residual_ratio",
                "observed": float(rows["coherent_residual_ratio"].max()),
                "interpretation": "positive residual exists but keeps its attribution label",
            },
            {
                "calculation": "signed cancellation",
                "formula": "min signed_residual_net / positive_residual_mass",
                "observed": float(rows["net_fraction_of_positive"].min()),
                "interpretation": "some radii are net damping rather than uncharged positive source",
            },
            {
                "calculation": "status labels",
                "formula": "door4_status value counts",
                "observed": ", ".join(f"{k}={v}" for k, v in rows["door4_status"].value_counts().sort_index().items()),
                "interpretation": "the residual is classified instead of buried",
            },
        ]
    )


def conclusion_summary(mode: str = "cached") -> pd.DataFrame:
    summary = run_all(mode, include_conclusion=False)
    same = summary["same_parent"]
    renewal = summary["renewal"]
    tail = summary["tail_ladder"]
    arr = summary["arr"]
    coherent = summary["coherent"]
    return pd.DataFrame(
        [
            {
                "claim": "Same-parent splits are not new evidence of source packing.",
                "current_read": (
                    "All split rows merge back under quotienting."
                    if same["all_split_rows_merge_back"]
                    else "At least one split row survives quotienting."
                ),
                "implication": "Multiplicity alone is not a free amplification channel.",
                "how_to_break_it": "Find a same-parent split with sustained post-quotient multiplicity and nonzero residual packing.",
            },
            {
                "claim": "Cross-parent jitter remains physically charged.",
                "current_read": (
                    f"Minimum stressed Fphys is {renewal['Fphys_min_stressed']:.3g}; "
                    f"max C0theta proxy is {renewal['C0theta_max']:.3g}."
                ),
                "implication": "A high-capture perturbation must still pay shape, renewal, separation, or oscillation cost.",
                "how_to_break_it": "Produce high eta with vanishing final physical charge and no quotient redundancy.",
            },
            {
                "claim": "Tail mass is not hidden; it is assigned to named denominator pressure or left unresolved.",
                "current_read": (
                    f"Best tail denominator profile: {tail['best_denominator_counts']}; "
                    f"best ratios stay {tail['best_ratio_min']:.3g} to {tail['best_ratio_max']:.3g}."
                ),
                "implication": "The notebook exposes named over-budget tail pressure instead of absorbing it silently.",
                "how_to_break_it": "Find annular tail mass with no finite-overlap parent payment and no named denominator or unresolved row.",
            },
            {
                "claim": "ARR deficits are repairable only when the missing channel is named.",
                "current_read": (
                    f"ARR pressure ratio changes from {arr['ratio_current']:.4g} to "
                    f"{arr['ratio_with_renewal_exposure']:.4g} after {arr['dominant_deficit_source']}."
                ),
                "implication": "A near miss becomes useful only if the deficit is attributed to a real ledger channel.",
                "how_to_break_it": "Find a persistent ARR deficit that is not renewal, tail, boundary, coherent residual, or physical charge.",
            },
            {
                "claim": "Coherent residuals do not become free positive source.",
                "current_read": f"Observed coherent labels: {coherent['statuses']}.",
                "implication": "The residual bookkeeping remains falsifiable because every positive row keeps its label.",
                "how_to_break_it": "Find a positive coherent residual that is not sign-balanced, damping-paid, renewal-paid, or explicitly matrix-paid.",
            },
        ]
    )


def routing_board() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "attack": "same-parent split",
                "pressure": "fake label multiplicity",
                "routing": "quotient-redundant",
                "headline": "pre-count explodes; post-count returns to natural tubes; eta=1",
            },
            {
                "attack": "renewal-cascade cross-parent jitter",
                "pressure": "high capture with perturbed cross-parent packets",
                "routing": "physical charge / post-growth, not free hiding",
                "headline": "capture stays high, but Fphys remains positive and C0theta proxy stays < 1",
            },
            {
                "attack": "tail denominator ladder",
                "pressure": "arbitrary annular tails",
                "routing": "unresolved denominator family",
                "headline": "D5 is the least-bad tested denominator, but every best ratio stays above 1",
            },
            {
                "attack": "ARR c185 final81",
                "pressure": "small ARR deficit",
                "routing": "renewal exposure",
                "headline": "renewal exposure can move an ARR pressure ratio below 1",
            },
            {
                "attack": "coherent residual",
                "pressure": "positive viscous residual",
                "routing": "signed damping / renewal-deactivation / sign balance",
                "headline": "finite-radius rows do not become an uncharged coherent positive source",
            },
        ]
    )


def run_all(mode: str = "cached", include_conclusion: bool = True) -> dict[str, object]:
    if mode not in {"synthetic", "fixture", "cached"}:
        raise ValueError("mode must be 'cached' or 'synthetic'")
    if mode == "cached":
        require_cached_data()
    summary: dict[str, object] = {
        "same_parent": same_parent_scorecard(mode),
        "renewal": renewal_scorecard(mode),
        "tail_ladder": tail_ladder_scorecard(mode),
        "arr": arr_scorecard(mode),
        "coherent": coherent_scorecard(mode),
    }
    if include_conclusion:
        summary["conclusion"] = conclusion_summary(mode).to_dict(orient="records")
    return summary


def compact_scorecard(mode: str = "cached") -> pd.DataFrame:
    summary = run_all(mode, include_conclusion=False)
    return pd.DataFrame(
        [
            {
                "demo": "same-parent quotient",
                "signal": "split copies merge back",
                "value": "yes" if summary["same_parent"]["all_split_rows_merge_back"] else "no",
            },
            {
                "demo": "renewal jitter",
                "signal": "min final physical charge",
                "value": f"{summary['renewal']['Fphys_min_stressed']:.3g}",
            },
            {
                "demo": "tail ladder",
                "signal": "best denominator family",
                "value": str(summary["tail_ladder"]["best_denominator_counts"]),
            },
            {
                "demo": "ARR deficit",
                "signal": "before -> after named repair",
                "value": (
                    f"{summary['arr']['ratio_current']:.3g} -> "
                    f"{summary['arr']['ratio_with_renewal_exposure']:.3g}"
                ),
            },
            {
                "demo": "coherent residual",
                "signal": "max residual ratio",
                "value": f"{summary['coherent']['max_coherent_residual_ratio']:.3g}",
            },
        ]
    )


def plot_same_parent(mode: str = "cached"):
    import matplotlib.pyplot as plt

    df = same_parent_summary(mode)
    grouped = (
        df.groupby("M", as_index=False)
        .agg(pre=("pre_quotient_tubes", "mean"), post=("post_quotient_tubes", "mean"))
        .sort_values("M")
    )
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(grouped["M"], grouped["pre"], marker="o", linewidth=2.4, label="before quotient")
    ax.plot(grouped["M"], grouped["post"], marker="o", linewidth=2.4, label="after quotient")
    ax.set_xscale("log")
    ax.set_xlabel("split multiplier M")
    ax.set_ylabel("mean tube count")
    ax.set_title("Same-parent split attack collapses under quotienting")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_renewal(mode: str = "cached"):
    import matplotlib.pyplot as plt

    df = renewal_summary(mode).copy()
    df["M"] = df["M"].astype(float)
    grouped = (
        df.groupby("M", as_index=False)
        .agg(eta=("eta_post", "min"), fphys=("Fphys_star_available", "min"), growth=("post_growth_fraction", "max"))
        .sort_values("M")
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    axes[0].plot(grouped["M"], grouped["eta"], marker="o", linewidth=2.4, color="#276fbf")
    axes[0].set_xscale("log")
    axes[0].set_ylim(0, max(1.05, grouped["eta"].max() * 1.05))
    axes[0].set_title("capture stays high")
    axes[0].set_xlabel("stress multiplier M")
    axes[0].set_ylabel("min eta")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(grouped["M"], grouped["fphys"], marker="o", linewidth=2.4, color="#a23e48")
    axes[1].set_xscale("log")
    axes[1].set_title("but physical charge remains")
    axes[1].set_xlabel("stress multiplier M")
    axes[1].set_ylabel("min Fphys")
    axes[1].grid(True, alpha=0.25)
    fig.suptitle("Renewal jitter is not free hiding", y=1.03)
    fig.tight_layout()
    return fig


def plot_tail_arr(mode: str = "cached"):
    import matplotlib.pyplot as plt

    tail = tail_ladder_rows(mode).copy()
    tail = tail.sort_values("best_ratio", ascending=True).tail(12)
    arr = arr_scorecard(mode)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), gridspec_kw={"width_ratios": [1.35, 0.8]})
    colors = ["#558b6e" if denom == "D5" else "#7a7a7a" for denom in tail["best_denominator"].astype(str)]
    labels = [f"tail case {idx}" for idx in range(1, len(tail) + 1)]
    axes[0].barh(labels, tail["best_ratio"], color=colors)
    axes[0].axvline(1.0, color="#222222", linewidth=1.2, linestyle="--")
    axes[0].set_xlabel("best tested tail ratio (>1 means unresolved)")
    axes[0].set_title("tail pressure stays named and above 1")
    axes[0].grid(True, axis="x", alpha=0.25)

    axes[1].bar(
        ["current demand/capacity", "renewal-corrected demand/capacity"],
        [arr["ratio_current"], arr["ratio_with_renewal_exposure"]],
        color=["#a23e48", "#558b6e"],
    )
    axes[1].axhline(1.0, color="#222222", linewidth=1.2, linestyle="--")
    axes[1].set_ylabel("ARR pressure ratio (demand / capacity)")
    axes[1].set_title("ARR repair check")
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_routing_summary(mode: str = "cached"):
    import matplotlib.pyplot as plt

    rows = pd.DataFrame(
        [
            {"pressure": "split labels", "route": "quotient"},
            {"pressure": "cross-parent jitter", "route": "physical charge"},
            {"pressure": "annular tail", "route": "denominator"},
            {"pressure": "ARR deficit", "route": "renewal exposure"},
            {"pressure": "coherent residual", "route": "signed / matrix row"},
        ]
    )
    y = np.arange(len(rows))
    colors = ["#276fbf", "#a23e48", "#558b6e", "#f4a261", "#6d597a"]
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.hlines(y, 0.18, 0.82, color="#d0d0d0", linewidth=2)
    ax.scatter(np.full(len(rows), 0.18), y, s=95, color="#333333", zorder=3)
    ax.scatter(np.full(len(rows), 0.82), y, s=130, color=colors, zorder=3)
    for idx, row in rows.iterrows():
        ax.text(0.14, idx, row["pressure"], va="center", ha="right", fontsize=10)
        ax.text(0.86, idx, row["route"], va="center", ha="left", fontsize=10, weight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.invert_yaxis()
    ax.set_title("Every pressure keeps a name")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def write_summary_json(mode: str = "cached", output: Path | None = None) -> Path:
    output = output or OUTPUTS / f"amplification_payment_summary_{mode}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(run_all(mode), indent=2, sort_keys=True), encoding="utf-8")
    return output


def _write_fixture_results(output_dir: Path | None = None) -> Path:
    """Generate toy CSVs in the same shape as the public lab expects.

    These are smoke-test artifacts only. They are not JHTDB evidence.
    """
    output_dir = output_dir or OUTPUTS / "fixture_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    _fixture_same_parent().to_csv(output_dir / NEAR_DEGENERATE.csvs[0].name, index=False)
    _fixture_renewal().to_csv(output_dir / RENEWAL_CASCADE.csvs[0].name, index=False)
    _fixture_tail_rows().to_csv(output_dir / TAIL_LADDER.csvs[0].name, index=False)
    pd.DataFrame([tail_ladder_scorecard("fixture")]).to_csv(output_dir / TAIL_LADDER.csvs[1].name, index=False)
    _fixture_arr().to_csv(output_dir / ARR_DEFICIT.csvs[0].name, index=False)
    _fixture_coherent().to_csv(output_dir / COHERENT_RESIDUAL.csvs[0].name, index=False)
    return output_dir


def write_synthetic_results(output_dir: Path | None = None) -> Path:
    """Write schema-compatible smoke-test CSVs; these are not JHTDB evidence."""
    output_dir = output_dir or OUTPUTS / "synthetic_results"
    return _write_fixture_results(output_dir)


def _fixture_same_parent() -> pd.DataFrame:
    rows = []
    for case, natural in [("demoA", 12), ("demoB", 20)]:
        for m in [1, 4, 8, 16, 32, 64]:
            rows.append(
                {
                    "case": case,
                    "radius_dx": 0.1,
                    "M": float(m),
                    "pre_quotient_tubes": float(natural * m),
                    "post_quotient_tubes": float(natural),
                    "eta_post": 1.0,
                    "R_pack_star": 0.0 if m > 1 else 0.0,
                    "Fphys_star_available": 2.0 + 0.1 * (case == "demoB"),
                    "attack_status": "baseline_no_split" if m == 1 else "showcase_quotient_redundant",
                }
            )
    return pd.DataFrame(rows)


def _fixture_renewal() -> pd.DataFrame:
    rows = []
    for case in ["demoA", "demoB"]:
        for m in [1, 8, 16]:
            rows.append(
                {
                    "case": case,
                    "ablation_mode": "fixture_shape_degenerate",
                    "M": m,
                    "jitter_strength": 0.0 if m == 1 else 1.5,
                    "post_quotient_tubes": 4 if m == 1 else 7 + m,
                    "eta_post": 1.0 if m == 1 else 0.94 - 0.002 * m,
                    "post_growth_fraction": 0.0 if m == 1 else 0.75 + 0.08 * m,
                    "Fphys_star_available": 2.1 if m == 1 else 3.5 + 0.2 * m,
                    "C0theta_proxy": 0.31,
                    "attack_status": "baseline_no_split" if m == 1 else "break_excess_post_growth",
                }
            )
    return pd.DataFrame(rows)


def _fixture_tail_rows() -> pd.DataFrame:
    rows = []
    for i, ratio in enumerate([1.4, 2.5, 5.8, 12.0, 22.0], start=1):
        rows.append(
            {
                "candidate_id": f"fixture_tail_{i}",
                "support": 33 + 16 * (i % 3),
                "tail_sum_SK": 0.5 + 0.03 * i,
                "ratio_D1": 1000.0 * ratio,
                "ratio_D2": 10.0 * ratio,
                "ratio_D3": np.nan,
                "ratio_D4": 15.0 * ratio,
                "ratio_D5": ratio,
                "ratio_D6": np.nan,
                "best_denominator": "D5",
                "best_ratio": ratio,
                "denominator_family_pass": "bounded_not_lt_1",
            }
        )
    return pd.DataFrame(rows)


def _fixture_arr() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": "fixture_arr_c185_like",
                "support": 81,
                "ratio_current": 1.0217,
                "ratio_with_renewal_exposure": 0.9671,
                "dominant_deficit_source": "renewal_exposure",
                "renewal_fraction": 0.89,
                "renewal_transition_count": 4,
            }
        ]
    )


def _fixture_coherent() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"radius_dx": 0.0, "door4_status": "sign_balanced_oscillation", "coherent_residual_ratio": 0.74, "Q": 1.8},
            {
                "radius_dx": 0.1,
                "door4_status": "renewal_deactivation_associated_residual",
                "coherent_residual_ratio": 0.34,
                "Q": 0.98,
            },
            {"radius_dx": 0.25, "door4_status": "signed_damping", "coherent_residual_ratio": 0.08, "Q": 0.32},
            {"radius_dx": 0.5, "door4_status": "signed_damping", "coherent_residual_ratio": 0.03, "Q": 0.31},
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the public amplification-payment lab summary.")
    parser.add_argument("--mode", choices=("cached", "synthetic"), default="cached")
    parser.add_argument("--write-json", action="store_true")
    parser.add_argument("--write-synthetic-results", action="store_true")
    args = parser.parse_args(argv)

    if args.write_synthetic_results:
        path = write_synthetic_results()
        print(f"wrote synthetic smoke CSVs to {rel(path)}")

    summary = run_all(args.mode)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.write_json:
        print(f"wrote {rel(write_summary_json(args.mode))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
