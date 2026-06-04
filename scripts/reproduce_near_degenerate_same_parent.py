from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


FINAL_CSV_NAME = "near_degenerate_multi_tube_packing_attack_summary.csv"
FINAL_REQUIRED_COLUMNS = {
    "case",
    "candidate_id",
    "radius_dx",
    "M",
    "natural_tubes",
    "pre_quotient_tubes",
    "post_quotient_tubes",
    "eta_post",
    "Fphys_star_available",
    "R_pack_star",
    "R_shape_fam_available",
    "R_spread_fam_available",
    "R_renew_fam_available",
    "R_D_available",
    "R_tail_available",
    "E_nu_coh_pos_available",
    "attack_status",
}
FINAL_COMPONENT_COLUMNS = [
    "R_pack_star",
    "R_spread_fam_available",
    "R_shape_fam_available",
    "R_renew_fam_available",
    "R_D_available",
    "R_tail_available",
    "E_nu_coh_pos_available",
]
EXPECTED_FPHYS_BY_FAMILY = {
    ("c413", 0.10): 2.4175185860354174,
    ("c413", 0.25): 2.4175185860354174,
    ("c309", 0.10): 2.1199817390292925,
    ("c309", 0.25): 2.117638477825343,
}
EXPECTED_ROWS = 25
KEY_COLUMNS = ["candidate_id", "case", "radius_dx", "M", "delta"]
PIPELINE_SCRIPT_ORDER = [
    "skeleton_closure_catalog.py",
    "targeted_dichotomy_audit.py",
    "promoted_tube_family_builder.py",
    "tube_pair_overlap_audit.py",
    "real_packing_weighted_renewal_audit.py",
    "near_degenerate_multi_tube_packing_attack.py",
    "plot_near_degenerate_attack.py",
]
JHTDB_ENV_VARS = [
    "JHTDB_TOKEN",
    "JHTDB_AUTH_TOKEN",
    "JHTDB_API_TOKEN",
    "JHTDB_USERNAME",
    "JHTDB_PASSWORD",
]
COMPONENT_PATTERNS = {
    "overlap": ["tube_pair_overlap_audit_top2_*_summary.csv"],
    "dichotomy": ["targeted_dichotomy_audit_top2_top3_summary.csv"],
    "renewal": [
        "real_packing_weighted_renewal_audit_summary.csv",
        "real_packing_weighted_renewal_audit_r0p25_summary.csv",
    ],
}
COMPONENT_COLUMN_ALIASES = {
    "R_pack_star": ["R_pack_star", "post_pack", "post_pack_proxy", "raw_split_pair_pack_proxy"],
    "R_spread_fam_available": ["R_spread_fam_available", "r_spread", "R_spread"],
    "R_shape_fam_available": ["R_shape_fam_available", "r_shape", "R_shape"],
    "R_renew_fam_available": ["R_renew_fam_available", "r_renew", "R_renew"],
    "R_D_available": ["R_D_available", "r_deact", "R_deact_available"],
    "R_tail_available": ["R_tail_available", "r_tail", "R_tail"],
    "E_nu_coh_pos_available": ["E_nu_coh_pos_available", "e_coh", "E_nu_coh_pos", "E_coh_pos"],
}


def find_repo_root(start: Path | None = None) -> Path:
    start_path = (start or Path(__file__).resolve()).resolve()
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "data" / "results" / FINAL_CSV_NAME).exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find repo root above {start_path} containing data/results/{FINAL_CSV_NAME}."
    )


def load_csv(path: Path, required_columns: set[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    frame = pd.read_csv(path)
    if required_columns:
        missing = sorted(required_columns - set(frame.columns))
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"{path} is missing required columns: {joined}")
    return frame


def positive_or_proxy(values: list[object]) -> float:
    finite_values: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        finite_values.append(numeric)
        if numeric > 0.0:
            return numeric
    if finite_values:
        return finite_values[0]
    return 0.0


def compute_fphys_from_components(row: pd.Series) -> float:
    return float(sum(positive_or_proxy([row.get(column)]) for column in FINAL_COMPONENT_COLUMNS))


def verify_final_csv_shape(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) != EXPECTED_ROWS:
        raise AssertionError(f"Expected {EXPECTED_ROWS} rows in {FINAL_CSV_NAME}; found {len(df)}.")
    numeric = df.copy()
    for column in (
        "radius_dx",
        "M",
        "natural_tubes",
        "pre_quotient_tubes",
        "post_quotient_tubes",
        "eta_post",
        "Fphys_star_available",
    ):
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    completed = numeric.dropna(
        subset=[
            "case",
            "candidate_id",
            "radius_dx",
            "M",
            "natural_tubes",
            "pre_quotient_tubes",
            "post_quotient_tubes",
            "Fphys_star_available",
        ]
    ).copy()
    if completed.empty:
        raise AssertionError("The final CSV has no completed same-parent rows after dropping incomplete entries.")
    return completed


def verify_three_fphys_values(df: pd.DataFrame, tolerance: float) -> dict[tuple[str, float], float]:
    observed = (
        df.groupby(["case", "radius_dx"], sort=True)["Fphys_star_available"]
        .first()
        .to_dict()
    )
    unique_values = sorted({float(value) for value in observed.values()})
    if len(unique_values) != 3:
        raise AssertionError(
            "Completed same-parent rows should have exactly 3 unique numeric "
            f"Fphys_star_available values; found {len(unique_values)} values: {unique_values}."
        )
    for family, expected in EXPECTED_FPHYS_BY_FAMILY.items():
        actual = float(observed.get(family, np.nan))
        if not math.isfinite(actual):
            raise AssertionError(f"Missing completed family for {family}.")
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
            raise AssertionError(
                f"Unexpected Fphys_star_available for {family}: expected {expected}, found {actual}."
            )
    return observed


def verify_m_invariance(df: pd.DataFrame, tolerance: float) -> None:
    grouped = df.groupby(["case", "candidate_id", "radius_dx"], sort=True)
    for family, group in grouped:
        if group["Fphys_star_available"].nunique(dropna=True) != 1:
            raise AssertionError(
                f"Fphys_star_available varies with M inside family {family}, "
                "but same_parent_split should keep it invariant."
            )
        if group["pre_quotient_tubes"].nunique(dropna=True) <= 1:
            raise AssertionError(
                f"pre_quotient_tubes does not change with M inside family {family}."
            )
        if not np.allclose(
            group["post_quotient_tubes"].to_numpy(dtype=float),
            group["natural_tubes"].to_numpy(dtype=float),
            atol=tolerance,
            rtol=0.0,
        ):
            raise AssertionError(
                f"post_quotient_tubes does not match natural_tubes for family {family}."
            )


def discover_component_files(
    results_dir: Path,
    benchmark_results_dir: Path | None,
    cache_dir: Path | None,
) -> dict[str, list[Path]]:
    search_roots: list[Path] = []
    for candidate in (results_dir, benchmark_results_dir, cache_dir):
        if candidate and candidate.exists():
            resolved = candidate.resolve()
            if resolved not in search_roots:
                search_roots.append(resolved)

    discovered: dict[str, list[Path]] = {name: [] for name in COMPONENT_PATTERNS}
    for name, patterns in COMPONENT_PATTERNS.items():
        for root in search_roots:
            for pattern in patterns:
                for path in sorted(root.rglob(pattern)):
                    if path not in discovered[name]:
                        discovered[name].append(path)
    return discovered


def match_component_rows(frame: pd.DataFrame, final_row: pd.Series, tolerance: float) -> pd.DataFrame:
    matched = frame.copy()
    shared_keys = [column for column in KEY_COLUMNS if column in frame.columns and column in final_row.index]
    if not shared_keys:
        return matched.iloc[0:0]
    for column in shared_keys:
        target = final_row[column]
        if pd.isna(target):
            continue
        if column in {"radius_dx", "M", "delta"}:
            values = pd.to_numeric(matched[column], errors="coerce")
            matched = matched[np.isclose(values, float(target), atol=tolerance, rtol=0.0)]
        else:
            matched = matched[matched[column].astype(str) == str(target)]
        if matched.empty:
            return matched
    return matched


def pick_component_value(
    component_name: str,
    matched_rows: list[pd.Series],
) -> float | None:
    candidates: list[object] = []
    aliases = COMPONENT_COLUMN_ALIASES[component_name]
    for row in matched_rows:
        for alias in aliases:
            if alias in row.index:
                candidates.append(row[alias])
    if not candidates:
        return None
    return positive_or_proxy(candidates)


def verify_component_sums_if_available(
    df: pd.DataFrame,
    results_dir: Path,
    benchmark_results_dir: Path | None,
    cache_dir: Path | None,
    tolerance: float,
) -> tuple[bool, str]:
    discovered = discover_component_files(results_dir, benchmark_results_dir, cache_dir)
    missing_groups = [name for name, paths in discovered.items() if not paths]
    if missing_groups:
        missing = ", ".join(missing_groups)
        return False, f"Skipped upstream recomputation because component summary files were not found for: {missing}."

    loaded = {
        name: [load_csv(path) for path in paths]
        for name, paths in discovered.items()
    }
    for _, final_row in df.iterrows():
        matched_rows: list[pd.Series] = []
        for frames in loaded.values():
            for frame in frames:
                matches = match_component_rows(frame, final_row, tolerance)
                if len(matches) == 1:
                    matched_rows.append(matches.iloc[0])
                elif len(matches) > 1:
                    raise AssertionError(
                        "Upstream component match was ambiguous for "
                        f"{final_row['candidate_id']} r={final_row['radius_dx']} M={final_row['M']}."
                    )
        if not matched_rows:
            raise AssertionError(
                "No upstream component rows matched final row "
                f"{final_row['candidate_id']} r={final_row['radius_dx']} M={final_row['M']}."
            )
        pieces: dict[str, float] = {}
        for component_name in FINAL_COMPONENT_COLUMNS:
            value = pick_component_value(component_name, matched_rows)
            if value is None:
                raise AssertionError(
                    f"Matched upstream summaries do not expose {component_name} for "
                    f"{final_row['candidate_id']} r={final_row['radius_dx']} M={final_row['M']}."
                )
            pieces[component_name] = value
        recomputed = float(sum(pieces.values()))
        actual = float(final_row["Fphys_star_available"])
        if not math.isclose(recomputed, actual, rel_tol=0.0, abs_tol=tolerance):
            raise AssertionError(
                "Upstream component recomputation does not match final Fphys_star_available for "
                f"{final_row['candidate_id']} r={final_row['radius_dx']} M={final_row['M']}: "
                f"expected {actual}, recomputed {recomputed}."
            )
    summary = []
    for name, paths in discovered.items():
        summary.extend(str(path) for path in paths)
    return True, "Verified upstream component sums using: " + ", ".join(summary)


def find_script(root: Path, filename: str) -> Path | None:
    matches = sorted(root.rglob(filename))
    return matches[0] if matches else None


def build_pipeline_commands(
    root: Path,
    results_dir: Path,
    benchmark_results_dir: Path | None,
    cache_dir: Path | None,
) -> list[dict[str, object]]:
    steps: list[dict[str, object]] = []
    for filename in PIPELINE_SCRIPT_ORDER:
        script_path = find_script(root, filename)
        if script_path is None:
            steps.append(
                {
                    "name": filename,
                    "script": None,
                    "command": None,
                    "prerequisites": [f"Missing script: {filename}"],
                    "requires_jhtdb": filename == "skeleton_closure_catalog.py",
                }
            )
            continue

        command = [sys.executable, str(script_path)]
        prerequisites = [f"Script exists: {script_path}"]
        requires_jhtdb = filename == "skeleton_closure_catalog.py"

        if filename == "near_degenerate_multi_tube_packing_attack.py":
            command.extend(["--attack-mode", "same_parent_split"])
            if benchmark_results_dir:
                command.extend(["--results-dir", str(benchmark_results_dir)])
            if cache_dir:
                command.extend(["--cache-dir", str(cache_dir)])
        elif filename == "plot_near_degenerate_attack.py":
            command.extend(["--output-dir", str(results_dir)])
        elif filename == "promoted_tube_family_builder.py":
            prerequisites.append("This builder must cover c413/c309 at radius_dx 0.1 and 0.25.")

        if benchmark_results_dir:
            prerequisites.append(f"Benchmark results dir: {benchmark_results_dir}")
        if cache_dir:
            prerequisites.append(f"Cache dir: {cache_dir}")

        steps.append(
            {
                "name": filename,
                "script": script_path,
                "command": command,
                "prerequisites": prerequisites,
                "requires_jhtdb": requires_jhtdb,
            }
        )
    return steps


def run_or_print_commands(
    commands: list[dict[str, object]],
    root: Path,
    dry_run: bool,
    strict: bool,
) -> None:
    for step in commands:
        name = str(step["name"])
        command = step["command"]
        print(f"[pipeline] step: {name}")
        for prereq in step["prerequisites"]:
            print(f"  prerequisite: {prereq}")
        if command is None:
            message = f"Skipping {name} because the script is not available."
            if strict:
                raise FileNotFoundError(message)
            print(f"  {message}")
            continue
        if step["requires_jhtdb"] and not any(os.environ.get(key) for key in JHTDB_ENV_VARS):
            message = (
                "Skipping JHTDB-dependent step because no JHTDB credentials were detected in the environment. "
                "Public users need their own JHTDB access/token or must rely on cached artifacts."
            )
            if strict:
                raise EnvironmentError(message)
            print(f"  {message}")
            continue

        rendered = subprocess.list2cmdline([str(part) for part in command])
        print(f"  command: {rendered}")
        if dry_run:
            continue
        completed = subprocess.run(command, cwd=root, check=False)
        if completed.returncode != 0:
            message = f"{name} exited with status {completed.returncode}."
            if strict:
                raise RuntimeError(message)
            print(f"  {message}")


def write_explanation_report(
    root: Path,
    results_dir: Path,
    benchmark_results_dir: Path | None,
    cache_dir: Path | None,
    write_report: Path | None,
) -> str:
    final_csv = results_dir / FINAL_CSV_NAME
    benchmark_note = (
        str(benchmark_results_dir)
        if benchmark_results_dir and benchmark_results_dir.exists()
        else "not present in this public checkout"
    )
    cache_note = str(cache_dir) if cache_dir else "not specified"
    report = f"""# Near-Degenerate Same-Parent Reproduction Bridge

## What this file is

`{final_csv}` is an offline derived summary, not a direct JHTDB export and not synthetic demo data.
The attack summary is produced by `benchmarks/jhtdb_response_family/near_degenerate_multi_tube_packing_attack.py`
in `same_parent_split` mode after upstream cached artifacts already exist.
The attack script itself does not query JHTDB.

## Upstream lineage

The public summary sits downstream of JHTDB-derived caches rooted in the `isotropic1024coarse` workflow.
At a high level the lineage is:

1. JHTDB / cache restore for promoted membership and skeleton intermediates.
2. `skeleton_closure_catalog.py`
3. `targeted_dichotomy_audit.py`
4. `promoted_tube_family_builder.py` for c413/c309 at radius_dx 0.1 and 0.25
5. `tube_pair_overlap_audit.py`
6. `real_packing_weighted_renewal_audit.py` for r0.1 and r0.25
7. `near_degenerate_multi_tube_packing_attack.py --attack-mode same_parent_split`
8. `scripts/plot_near_degenerate_attack.py` for figure regeneration

## Why there are 25 rows

There are 24 completed same-parent rows from:

- 2 physical candidates: c413 and c309
- 2 radii: 0.1 and 0.25
- 6 multiplicities: M = 1, 4, 8, 16, 32, 64

The 25th row is an incomplete c185 placeholder with `attack_status=not_run_missing_promoted_membership`.
That row is retained as evidence of a missing-promoted-membership branch rather than silently removed.

## Why exactly three Fphys values exist

`Fphys_star_available` is flat in `same_parent_split` mode because M changes label multiplicity before the physical quotient,
but the post-quotient physical ledger bundle stays the same within each candidate/radius family.

The completed rows collapse to exactly three values:

- c413 at both radii: `2.4175185860354174`
- c309 at radius 0.1: `2.1199817390292925`
- c309 at radius 0.25: `2.117638477825343`

So the repeated decimals are evidence of family-level upstream ledger bundles, not of synthetic filler rows.

## Files needed to recompute Fphys from cached summaries

- `tube_pair_overlap_audit_top2_*_summary.csv`
- `targeted_dichotomy_audit_top2_top3_summary.csv`
- `real_packing_weighted_renewal_audit_summary.csv`
- `real_packing_weighted_renewal_audit_r0p25_summary.csv`

The final Fphys proxy is the positive-or-proxy sum of:

- `R_pack_star`
- `R_spread_fam_available`
- `R_shape_fam_available`
- `R_renew_fam_available`
- `R_D_available`
- `R_tail_available`
- `E_nu_coh_pos_available`

## Files needed to regenerate from DNS / JHTDB

Full regeneration requires:

- the upstream benchmark scripts
- a JHTDB-backed cache restore path or your own JHTDB access/token
- benchmark results directories such as `{benchmark_note}`
- any required cache directory such as `{cache_note}`

The public package can audit the derived lineage and verify the bundled CSV.
It should not be read as claiming that DNS regeneration is possible from the public bundle alone.
"""
    if write_report:
        write_report.parent.mkdir(parents=True, exist_ok=True)
        write_report.write_text(report, encoding="utf-8")
    return report


def default_results_dir(root: Path) -> Path:
    public_results = root / "data" / "results"
    benchmark_results = root / "benchmarks" / "jhtdb_response_family" / "results"
    if public_results.exists():
        return public_results
    if benchmark_results.exists():
        return benchmark_results
    return public_results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit or optionally orchestrate the near-degenerate same-parent attack data lineage."
    )
    discovered_root = find_repo_root()
    parser.add_argument("--root", type=Path, default=discovered_root, help="repo root (defaults to this public repo)")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="results directory (defaults to data/results, or benchmarks/.../results if data/results is absent)",
    )
    parser.add_argument(
        "--benchmark-results-dir",
        type=Path,
        default=None,
        help="benchmark results directory for upstream cached summaries",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="optional cache directory to search for upstream summary CSVs",
    )
    parser.add_argument("--verify-derived", action="store_true", help="verify the bundled derived summary CSV")
    parser.add_argument("--explain", action="store_true", help="print a human-readable lineage report")
    parser.add_argument("--run-pipeline", action="store_true", help="attempt the upstream pipeline when scripts exist")
    parser.add_argument("--dry-run", action="store_true", help="print pipeline commands without executing them")
    parser.add_argument("--strict", action="store_true", help="treat missing prerequisites as hard failures")
    parser.add_argument(
        "--write-report",
        type=Path,
        default=None,
        help="optional markdown report path, e.g. reports/near_degenerate_same_parent_reproduction.md",
    )
    parser.add_argument("--tolerance", type=float, default=1e-9, help="absolute comparison tolerance")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    results_dir = (args.results_dir.resolve() if args.results_dir else default_results_dir(root).resolve())
    benchmark_results_dir = args.benchmark_results_dir.resolve() if args.benchmark_results_dir else None
    cache_dir = args.cache_dir.resolve() if args.cache_dir else None

    if not any((args.verify_derived, args.explain, args.run_pipeline)):
        args.verify_derived = True

    if args.explain:
        report = write_explanation_report(
            root=root,
            results_dir=results_dir,
            benchmark_results_dir=benchmark_results_dir,
            cache_dir=cache_dir,
            write_report=args.write_report,
        )
        print(report)
        if args.write_report:
            print(f"Wrote report to {args.write_report}")

    if args.verify_derived:
        final_csv = results_dir / FINAL_CSV_NAME
        df = load_csv(final_csv, FINAL_REQUIRED_COLUMNS)
        completed = verify_final_csv_shape(df)
        verify_three_fphys_values(completed, args.tolerance)
        verify_m_invariance(completed, args.tolerance)

        recomputed = completed.apply(compute_fphys_from_components, axis=1)
        deltas = (recomputed - completed["Fphys_star_available"]).abs()
        if float(deltas.max()) > args.tolerance:
            raise AssertionError(
                "Direct component recomputation from the final CSV columns does not match "
                f"Fphys_star_available within tolerance {args.tolerance}."
            )

        upstream_verified, upstream_message = verify_component_sums_if_available(
            df=completed,
            results_dir=results_dir,
            benchmark_results_dir=benchmark_results_dir,
            cache_dir=cache_dir,
            tolerance=args.tolerance,
        )
        print(f"Verified {FINAL_CSV_NAME}: {len(df)} rows total, {len(completed)} completed same-parent rows.")
        print("Verified exactly three completed Fphys_star_available plateaus and M-invariance within families.")
        print("Verified pre_quotient_tubes changes with M and post_quotient_tubes matches natural_tubes.")
        print(
            "Verified direct Fphys recomputation from bundled component columns "
            f"within tolerance {args.tolerance}."
        )
        print(upstream_message)
        if args.strict and not upstream_verified:
            raise FileNotFoundError(upstream_message)

    if args.run_pipeline:
        commands = build_pipeline_commands(
            root=root,
            results_dir=results_dir,
            benchmark_results_dir=benchmark_results_dir,
            cache_dir=cache_dir,
        )
        run_or_print_commands(commands, root=root, dry_run=args.dry_run, strict=args.strict)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
