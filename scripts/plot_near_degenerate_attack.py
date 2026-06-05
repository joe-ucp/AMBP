from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
DEFAULT_INPUT_CSV = RESULTS / "near_degenerate_multi_tube_packing_attack_summary.csv"
ETA_NAME = "near_degenerate_attack_eta_vs_M.png"
FPHYS_NAME = "near_degenerate_attack_fphys_vs_M.png"
COMBINED_NAME = "near_degenerate_attack_results_figure.png"

REQUIRED_COLUMNS = {
    "case",
    "radius_dx",
    "M",
    "eta_post",
    "Fphys_star_available",
    "pre_quotient_tubes",
    "post_quotient_tubes",
}

GROUP_COLORS = {
    ("c309", 0.10): "#276fbf",
    ("c309", 0.25): "#3d8bfd",
    ("c413", 0.10): "#a23e48",
    ("c413", 0.25): "#d16b74",
}


def format_radius(radius_dx: float) -> str:
    return f"{radius_dx:g}dx"


def label_for(case: str, radius_dx: float) -> str:
    return f"{case} r={format_radius(radius_dx)}"


def load_plot_rows(input_csv: Path = DEFAULT_INPUT_CSV) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing bundled CSV: {input_csv}")

    df = pd.read_csv(input_csv)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"CSV is missing required columns: {joined}")

    plot_df = df.dropna(subset=sorted(REQUIRED_COLUMNS)).copy()
    if plot_df.empty:
        raise ValueError("CSV has no complete same-parent plot rows after dropping missing values.")

    plot_df["case"] = plot_df["case"].astype(str)
    plot_df["radius_dx"] = plot_df["radius_dx"].astype(float)
    plot_df["M"] = plot_df["M"].astype(float)
    for col in (
        "eta_post",
        "Fphys_star_available",
        "pre_quotient_tubes",
        "post_quotient_tubes",
    ):
        plot_df[col] = plot_df[col].astype(float)

    plot_df = plot_df.sort_values(["case", "radius_dx", "M"]).reset_index(drop=True)
    return plot_df


def output_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return (
        output_dir / ETA_NAME,
        output_dir / FPHYS_NAME,
        output_dir / COMBINED_NAME,
    )


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def grouped_rows(df: pd.DataFrame) -> list[tuple[tuple[str, float], pd.DataFrame]]:
    return list(df.groupby(["case", "radius_dx"], sort=True))


def maybe_set_log2_xscale(ax: plt.Axes, values: pd.Series) -> None:
    positive = values[values.gt(0)]
    if positive.nunique() > 1 and positive.min() >= 1:
        ax.set_xscale("log", base=2)


def plot_eta(ax: plt.Axes, groups: list[tuple[tuple[str, float], pd.DataFrame]]) -> None:
    for key, group in groups:
        case, radius_dx = key
        color = GROUP_COLORS.get((case, radius_dx), None)
        ax.plot(
            group["M"],
            group["eta_post"],
            marker="o",
            linewidth=2.2,
            markersize=5,
            color=color,
            label=label_for(case, radius_dx),
        )
    maybe_set_log2_xscale(ax, pd.concat([group["M"] for _, group in groups], ignore_index=True))
    ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1.1)
    ax.set_xlabel("M")
    ax.set_ylabel("post-quotient captured fraction eta")
    ax.set_title("Capture under adversarial label split")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)


def plot_fphys(ax: plt.Axes, groups: list[tuple[tuple[str, float], pd.DataFrame]]) -> None:
    for key, group in groups:
        case, radius_dx = key
        color = GROUP_COLORS.get((case, radius_dx), None)
        ax.plot(
            group["M"],
            group["Fphys_star_available"],
            marker="o",
            linewidth=2.2,
            markersize=5,
            color=color,
            label=label_for(case, radius_dx),
        )
    maybe_set_log2_xscale(ax, pd.concat([group["M"] for _, group in groups], ignore_index=True))
    ax.set_xlabel("M")
    ax.set_ylabel("available post-quotient Fphys* proxy")
    ax.set_title("Physical cost after quotient")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)


def plot_tube_counts(ax: plt.Axes, groups: list[tuple[tuple[str, float], pd.DataFrame]]) -> None:
    for key, group in groups:
        case, radius_dx = key
        color = GROUP_COLORS.get((case, radius_dx), None)
        ax.plot(group["M"], group["pre_quotient_tubes"], marker="o", linewidth=2.1, color=color)
        ax.plot(group["M"], group["post_quotient_tubes"], marker="o", linewidth=2.1, linestyle="--", color=color)
    maybe_set_log2_xscale(ax, pd.concat([group["M"] for _, group in groups], ignore_index=True))
    ax.set_xlabel("M")
    ax.set_ylabel("tube count")
    ax.set_title("Quotient Merges All Same-Parent Clones")
    ax.grid(True, alpha=0.25)
    ax.legend(
        handles=[
            Line2D([0], [0], color="#444444", linewidth=2.1, marker="o", label="pre-quotient tubes"),
            Line2D([0], [0], color="#444444", linewidth=2.1, linestyle="--", marker="o", label="post-quotient tubes"),
        ],
        frameon=False,
        fontsize=8,
        loc="upper left",
    )


def write_eta_plot(df: pd.DataFrame, output_path: Path) -> None:
    groups = grouped_rows(df)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    plot_eta(ax, groups)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_fphys_plot(df: pd.DataFrame, output_path: Path) -> None:
    groups = grouped_rows(df)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    plot_fphys(ax, groups)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_combined_plot(df: pd.DataFrame, output_path: Path) -> None:
    groups = grouped_rows(df)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    plot_fphys(axes[0], groups)
    axes[0].set_title("Physical Cost Remains Flat")
    plot_tube_counts(axes[1], groups)
    plot_eta(axes[2], groups)
    axes[2].set_title("eta = 1.0 (Perfectly Conserved)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_plots(output_dir: Path = RESULTS, input_csv: Path = DEFAULT_INPUT_CSV) -> tuple[Path, Path, Path]:
    df = load_plot_rows(input_csv)
    eta_png, fphys_png, combined_png = output_paths(output_dir)
    write_eta_plot(df, eta_png)
    write_fphys_plot(df, fphys_png)
    write_combined_plot(df, combined_png)
    return eta_png, fphys_png, combined_png


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate bundled near-degenerate attack figures from the summary CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS,
        help="directory to write the PNGs into (defaults to data/results)",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="summary CSV to plot (defaults to data/results/near_degenerate_multi_tube_packing_attack_summary.csv)",
    )
    args = parser.parse_args(argv)

    eta_png, fphys_png, combined_png = write_plots(args.output_dir, args.input_csv)
    print(f"Wrote {display_path(eta_png)}")
    print(f"Wrote {display_path(fphys_png)}")
    print(f"Wrote {display_path(combined_png)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
