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
from benchmarks.jhtdb_response_family.jhtdb_client import make_flow_client
from benchmarks.jhtdb_response_family.sampling import patch_points


def main() -> None:
    parser = argparse.ArgumentParser(description="Relaxed patch-local stretch budget without renewal birth gates.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--dataset", default="isotropic1024coarse")
    parser.add_argument("--source-mode", choices=("synthetic", "soap", "pyjhtdb", "auto"), default="pyjhtdb")
    parser.add_argument("--patch-points-per-axis", type=int, default=17)
    parser.add_argument("--patch-spacing", type=float, default=2.0 * np.pi / 1024.0 * 4.0)
    parser.add_argument("--psi-threshold", type=float, default=0.005)
    parser.add_argument("--time-stride", type=int, default=8)
    parser.add_argument("--center-mode", choices=("source_birth", "birth"), default="source_birth")
    parser.add_argument("--max-points-per-request", type=int, default=2048)
    args = parser.parse_args()

    starts = pd.read_csv(args.run_dir / "material_heat_age_starts.csv")
    paths = pd.read_csv(args.run_dir / "material_heat_age_paths.csv")
    if starts.empty or paths.empty:
        raise SystemExit("run directory must contain nonempty material heat-age starts and paths")
    root = starts.sort_values(["chain_index", "birth_time"]).iloc[0]
    if args.center_mode == "birth" and {"birth_x", "birth_y", "birth_z"}.issubset(starts.columns):
        center = np.asarray([root["birth_x"], root["birth_y"], root["birth_z"]], dtype=float)
    else:
        center = np.asarray([root["source_birth_x"], root["source_birth_y"], root["source_birth_z"]], dtype=float)

    points, _center_index = patch_points(
        center,
        patch_points_per_axis=args.patch_points_per_axis,
        patch_spacing=args.patch_spacing,
        domain_length=2.0 * np.pi,
    )
    times = np.sort(pd.to_numeric(paths["time"], errors="coerce").dropna().unique())
    times = times[:: max(int(args.time_stride), 1)]
    if times.size == 0:
        raise SystemExit("no finite path times found")

    access = JHTDBAccessConfig(
        dataset=args.dataset,
        source_mode=args.source_mode,
        velocity_sinterp="Lag4",
        gradient_sinterp="FD4Lag4",
        tinterp="PCHIP",
        max_points_per_request=args.max_points_per_request,
    )
    client = make_flow_client(access)

    rows = []
    peak_rows = []
    shape = (args.patch_points_per_axis,) * 3
    for time in times:
        data = cached_query_points(
            client,
            access,
            time=float(time),
            points=points,
            role=(
                "relaxed_patch_stretch_budget:"
                f"n={args.patch_points_per_axis}:dx={args.patch_spacing:.12g}:"
                f"psi={args.psi_threshold:.12g}:center={args.center_mode}"
            ),
        )
        metrics = _point_metrics(data.gradient)
        local_mask = _local_lambda_maxima(metrics["lambda"], shape)
        active = local_mask & (metrics["lambda"] > 0.0) & (metrics["psi_pos"] >= args.psi_threshold)
        rows.append(_budget_row(float(time), active, metrics))
        for idx in np.flatnonzero(active):
            peak_rows.append(
                {
                    "time": float(time),
                    "point_index": int(idx),
                    "x": float(points[idx, 0]),
                    "y": float(points[idx, 1]),
                    "z": float(points[idx, 2]),
                    "lambda": float(metrics["lambda"][idx]),
                    "omega_norm": float(metrics["omega_norm"][idx]),
                    "psi_pos": float(metrics["psi_pos"][idx]),
                }
            )

    timeseries = pd.DataFrame(rows)
    peaks = pd.DataFrame(peak_rows)
    timeseries = _add_integrals(timeseries)
    summary = _summary(timeseries, peaks, args, center)

    ts_path = args.run_dir / "relaxed_patch_stretch_budget_timeseries.csv"
    peaks_path = args.run_dir / "relaxed_patch_stretch_budget_peaks.csv"
    summary_path = args.run_dir / "relaxed_patch_stretch_budget_summary.csv"
    timeseries.to_csv(ts_path, index=False)
    peaks.to_csv(peaks_path, index=False)
    summary.to_csv(summary_path, index=False)
    chart = _write_plot(timeseries, args.run_dir)

    print("relaxed_patch_stretch_budget", flush=True)
    print("\nsummary", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nkey times", flush=True)
    cols = [
        "time",
        "N_active",
        "S_lambda_pos",
        "M_lambda",
        "S_over_M_pos",
        "omega_max_active",
        "omega_max_patch",
        "cumulative_S",
        "cumulative_M_pos",
    ]
    print(_key_times(timeseries)[cols].to_string(index=False), flush=True)
    print(f"\ntimeseries={ts_path}", flush=True)
    print(f"peaks={peaks_path}", flush=True)
    print(f"summary={summary_path}", flush=True)
    if chart is not None:
        print(f"chart={chart}", flush=True)


def _point_metrics(gradient: np.ndarray) -> dict[str, np.ndarray]:
    gradient = np.asarray(gradient, dtype=float).reshape((-1, 3, 3))
    omega = vorticity_from_gradient(gradient)
    omega_norm = np.linalg.norm(omega, axis=1)
    xi = omega / np.maximum(omega_norm[:, None], 1e-12)
    strain = np.asarray([strain_rotation(g)[0] for g in gradient])
    sxi = np.einsum("nij,nj->ni", strain, xi)
    lam = np.einsum("ni,ni->n", xi, sxi)
    psi_pos = np.maximum(lam, 0.0) / np.maximum(omega_norm, 1e-12)
    return {"lambda": lam, "omega_norm": omega_norm, "psi_pos": psi_pos}


def _local_lambda_maxima(values: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    cube = np.asarray(values, dtype=float).reshape(shape)
    out = np.ones(shape, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if di == dj == dk == 0:
                    continue
                shifted = np.roll(cube, shift=(di, dj, dk), axis=(0, 1, 2))
                valid = np.ones(shape, dtype=bool)
                if di > 0:
                    valid[:di, :, :] = False
                elif di < 0:
                    valid[di:, :, :] = False
                if dj > 0:
                    valid[:, :dj, :] = False
                elif dj < 0:
                    valid[:, dj:, :] = False
                if dk > 0:
                    valid[:, :, :dk] = False
                elif dk < 0:
                    valid[:, :, dk:] = False
                out &= (~valid) | (cube >= shifted)
    return out.ravel()


def _budget_row(time: float, active: np.ndarray, metrics: dict[str, np.ndarray]) -> dict[str, float | int]:
    lam = metrics["lambda"]
    omega = metrics["omega_norm"]
    psi = metrics["psi_pos"]
    lam_pos_all = np.maximum(lam, 0.0)
    lam_active = lam[active]
    lam_pos = np.maximum(lam_active, 0.0)
    omega_active = omega[active]
    s_pos = float(np.sum(lam_pos))
    m_pos = float(np.max(lam_pos)) if lam_pos.size else 0.0
    m_lambda = float(np.max(lam_active)) if lam_active.size else np.nan
    return {
        "time": time,
        "N_active": int(np.sum(active)),
        "N_positive_patch": int(np.sum(lam > 0.0)),
        "S_lambda_pos": s_pos,
        "M_lambda": m_lambda,
        "M_lambda_pos": m_pos,
        "S_over_M_pos": s_pos / m_pos if m_pos > 0.0 else np.nan,
        "M_over_S_pos": m_pos / s_pos if s_pos > 0.0 else np.nan,
        "omega_max_active": float(np.max(omega_active)) if omega_active.size else np.nan,
        "omega_max_patch": float(np.max(omega)) if omega.size else np.nan,
        "psi_pos_max_active": float(np.max(psi[active])) if np.any(active) else np.nan,
        "S_lambda_pos_patch_all_positive": float(np.sum(lam_pos_all)),
        "M_lambda_patch": float(np.max(lam)) if lam.size else np.nan,
    }


def _add_integrals(timeseries: pd.DataFrame) -> pd.DataFrame:
    out = timeseries.sort_values("time").copy()
    for col, target in [
        ("S_lambda_pos", "cumulative_S"),
        ("M_lambda_pos", "cumulative_M_pos"),
        ("omega_max_patch", "cumulative_omega_max_patch"),
        ("omega_max_active", "cumulative_omega_max_active"),
    ]:
        vals = pd.to_numeric(out[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        times = pd.to_numeric(out["time"], errors="coerce").to_numpy(dtype=float)
        integ = np.zeros_like(vals)
        if len(vals) > 1:
            dt = np.diff(times)
            area = 0.5 * (vals[:-1] + vals[1:]) * dt
            integ[1:] = np.cumsum(area)
        out[target] = integ
    return out


def _summary(
    timeseries: pd.DataFrame,
    peaks: pd.DataFrame,
    args: argparse.Namespace,
    center: np.ndarray,
) -> pd.DataFrame:
    ts = timeseries.sort_values("time")
    terminal = ts.iloc[-1]
    return pd.DataFrame(
        [
            {
                "patch_points_per_axis": int(args.patch_points_per_axis),
                "patch_spacing": float(args.patch_spacing),
                "psi_threshold": float(args.psi_threshold),
                "time_stride": int(args.time_stride),
                "center_mode": args.center_mode,
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "center_z": float(center[2]),
                "time_start": float(ts["time"].iloc[0]),
                "time_end": float(ts["time"].iloc[-1]),
                "sample_count": int(len(ts)),
                "peak_count_total": int(len(peaks)),
                "N_active_max": int(ts["N_active"].max()),
                "S_lambda_pos_max": float(ts["S_lambda_pos"].max()),
                "time_S_lambda_pos_max": float(ts.loc[ts["S_lambda_pos"].idxmax(), "time"]),
                "M_lambda_max": float(ts["M_lambda"].max()),
                "time_M_lambda_max": float(ts.loc[ts["M_lambda"].idxmax(), "time"]),
                "omega_max_patch_max": float(ts["omega_max_patch"].max()),
                "time_omega_max_patch_max": float(ts.loc[ts["omega_max_patch"].idxmax(), "time"]),
                "terminal_N_active": int(terminal["N_active"]),
                "terminal_S_lambda_pos": float(terminal["S_lambda_pos"]),
                "terminal_M_lambda": float(terminal["M_lambda"]),
                "terminal_S_over_M_pos": float(terminal["S_over_M_pos"])
                if pd.notna(terminal["S_over_M_pos"])
                else np.nan,
                "terminal_omega_max_patch": float(terminal["omega_max_patch"]),
                "cumulative_S": float(terminal["cumulative_S"]),
                "cumulative_M_pos": float(terminal["cumulative_M_pos"]),
                "cumulative_omega_max_patch": float(terminal["cumulative_omega_max_patch"]),
            }
        ]
    )


def _key_times(timeseries: pd.DataFrame) -> pd.DataFrame:
    ts = timeseries.sort_values("time")
    idxs = {ts.index[0], ts.index[-1]}
    for col in ("S_lambda_pos", "M_lambda", "omega_max_patch"):
        idxs.add(ts[col].idxmax())
    return ts.loc[sorted(idxs)].sort_values("time")


def _write_plot(timeseries: pd.DataFrame, run_dir: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    out = run_dir / "relaxed_patch_stretch_budget.png"
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    axes[0, 0].plot(timeseries["time"], timeseries["S_lambda_pos"], marker="o", label="S local maxima")
    axes[0, 0].plot(timeseries["time"], timeseries["M_lambda"], marker="o", label="M local maxima")
    axes[0, 0].plot(
        timeseries["time"],
        timeseries["S_lambda_pos_patch_all_positive"],
        marker="o",
        label="sum all lambda+ points",
        alpha=0.55,
    )
    axes[0, 0].set_title("Relaxed Stretch Budget")
    axes[0, 0].legend(fontsize=7)

    axes[0, 1].plot(timeseries["time"], timeseries["N_active"], marker="o", label="local maxima")
    axes[0, 1].plot(timeseries["time"], timeseries["N_positive_patch"], marker="o", label="all lambda+ points")
    axes[0, 1].set_title("Active Count")
    axes[0, 1].legend(fontsize=7)

    axes[1, 0].plot(timeseries["time"], timeseries["S_over_M_pos"], marker="o", label="S/M")
    axes[1, 0].plot(timeseries["time"], timeseries["M_over_S_pos"], marker="o", label="M/S")
    axes[1, 0].set_title("Participation")
    axes[1, 0].legend(fontsize=7)

    axes[1, 1].plot(timeseries["time"], timeseries["omega_max_patch"], marker="o", label="patch omega max")
    axes[1, 1].plot(timeseries["time"], timeseries["omega_max_active"], marker="o", label="active omega max")
    axes[1, 1].set_title("Patch Sup Norm")
    axes[1, 1].legend(fontsize=7)
    for ax in axes.ravel():
        ax.set_xlabel("time")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()
