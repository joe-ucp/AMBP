from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmarks" / "jhtdb_response_family"
DEFAULT_RESULTS_DIR = ROOT / "data" / "results"
DEFAULT_OUTPUTS_DIR = ROOT / "outputs"
DEFAULT_PUBLISHED_RESULTS_DIR = DEFAULT_OUTPUTS_DIR / "near_degenerate_public_validation"
DEFAULT_PUBLISHED_INPUTS_DIR = ROOT / "data" / "inputs" / "near_degenerate_published"
FINAL_CSV_NAME = "near_degenerate_multi_tube_packing_attack_summary.csv"
COLUMN_DICTIONARY_NAME = "near_degenerate_column_dictionary.json"
PUBLISHED_INPUT_PACK_NAME = "published_input_pack.json"
FINAL_PLOT_NAMES = [
    "near_degenerate_attack_eta_vs_M.png",
    "near_degenerate_attack_fphys_vs_M.png",
    "near_degenerate_attack_results_figure.png",
]
PIPELINE_STAGES = ["skeleton", "dichotomy", "membership", "overlap", "renewal", "attack", "plots"]
JHTDB_ENV_VARS = [
    "JHTDB_TOKEN",
    "JHTDB_AUTH_TOKEN",
    "JHTDB_API_TOKEN",
    "JHTDB_USERNAME",
    "JHTDB_PASSWORD",
]


class PipelineError(RuntimeError):
    """Raised when the public pipeline cannot proceed honestly."""


@dataclass(frozen=True)
class CandidateSpec:
    case: str
    candidate_id: str
    family_label: str


@dataclass(frozen=True)
class MembershipArtifact:
    candidate: CandidateSpec
    radius_dx: float
    membership_csv: Path
    overlap_summary_csv: Path
    overlap_pairs_csv: Path


@dataclass(frozen=True)
class ScanSource:
    family_label: str
    run_dir: Path
    candidate_csv: Path


@dataclass(frozen=True)
class PublishedInputPack:
    pack_dir: Path
    manifest_path: Path
    windows_csv: Path | None
    scan_sources: list[ScanSource]


def find_repo_root(start: Path | None = None) -> Path:
    start_path = (start or Path(__file__).resolve()).resolve()
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "README.md").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError(f"Could not find the public repo root above {start_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or audit the public near-degenerate same-parent JHTDB pipeline."
    )
    parser.add_argument(
        "--published-cases",
        action="store_true",
        help="use the bundled published c413/c309 same-parent configuration",
    )
    parser.add_argument("--candidate", help="exact candidate_id to run instead of the published bundle")
    parser.add_argument(
        "--candidate-tag",
        help="candidate token such as c132; the runner selects the strongest matching candidate row",
    )
    parser.add_argument(
        "--scan",
        nargs=3,
        action="append",
        metavar=("FAMILY", "RUN_DIR", "CANDIDATE_CSV"),
        help="explicit scan source override used when live skeleton/membership stages must run",
    )
    parser.add_argument("--windows-csv", type=Path, default=None, help="required for a fresh live skeleton run")
    parser.add_argument("--dry-run", action="store_true", help="print commands and prerequisites without executing")
    parser.add_argument(
        "--use-existing-cache",
        action="store_true",
        help="restore matching upstream artifacts from --cache-dir or discovered roots when available",
    )
    parser.add_argument(
        "--require-jhtdb",
        action="store_true",
        help="prefer the live JHTDB path for missing upstream stages instead of relying only on restored cache artifacts",
    )
    parser.add_argument(
        "--stop-after",
        choices=PIPELINE_STAGES,
        default=None,
        help="stop after the named stage (skeleton, dichotomy, membership, overlap, renewal, attack, plots)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "directory for pipeline artifacts; defaults to "
            "outputs/near_degenerate_public_validation for published runs"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="restored upstream artifact root to search recursively",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="manifest output path; defaults to <results-dir>/near_degenerate_pipeline_manifest.json",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat hash drift or ambiguous restored artifacts as fatal instead of warning",
    )
    return parser.parse_args(argv)


def _candidate_family_label(candidate_id: str) -> str:
    lowered = str(candidate_id).lower()
    if "top3" in lowered:
        return "top3"
    if "top2" in lowered:
        return "top2"
    return "top2"


def _candidate_case(candidate_id: str) -> str:
    text = str(candidate_id)
    for token in text.split(":"):
        if token.startswith("c") and token[1:].isdigit():
            return token
    raise PipelineError(f"Could not parse a case token like c413 from candidate_id={candidate_id!r}")


def _slugify(value: str) -> str:
    out = []
    for ch in str(value):
        out.append(ch if ch.isalnum() else "_")
    return "".join(out).strip("_") or "candidate"


def _radius_tag(radius_dx: float) -> str:
    text = f"{radius_dx:.6g}".rstrip("0").rstrip(".")
    return "r" + text.replace(".", "p")


def _command_text(parts: list[object]) -> str:
    return subprocess.list2cmdline([str(part) for part in parts])


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _live_jhtdb_config_source() -> str:
    if any(os.environ.get(key) for key in JHTDB_ENV_VARS):
        return "env_user_token"
    return "public_testing_token"


def _has_live_jhtdb_config() -> bool:
    return _live_jhtdb_config_source() != "none"


def _published_candidate_specs(root: Path) -> list[CandidateSpec]:
    bundled = root / "data" / "results" / FINAL_CSV_NAME
    if not bundled.exists():
        raise PipelineError(f"Bundled published summary is missing: {bundled}")
    df = pd.read_csv(bundled)
    rows = []
    for case in ("c413", "c309"):
        part = df[df["case"].astype(str).eq(case)].copy()
        if part.empty:
            raise PipelineError(f"The bundled summary does not contain a published {case} row.")
        candidate_id = str(part.iloc[0]["candidate_id"])
        rows.append(
            CandidateSpec(
                case=case,
                candidate_id=candidate_id,
                family_label=_candidate_family_label(candidate_id),
            )
        )
    return rows


def _published_summary_columns(root: Path) -> list[str]:
    dictionary_path = root / "data" / "results" / COLUMN_DICTIONARY_NAME
    if dictionary_path.exists():
        payload = json.loads(dictionary_path.read_text(encoding="utf-8"))
        columns = [str(entry["column"]) for entry in payload.get("columns", []) if entry.get("column")]
        if columns:
            return columns

    bundled = root / "data" / "results" / FINAL_CSV_NAME
    if bundled.exists():
        with bundled.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
        if header:
            return [str(column) for column in header]

    raise PipelineError(
        "Could not determine the published final CSV column contract from the bundled "
        "dictionary or summary CSV."
    )


def _normalize_public_attack_summary(summary_csv: Path, *, root: Path) -> list[str]:
    published_columns = _published_summary_columns(root)
    bundled_summary = root / "data" / "results" / FINAL_CSV_NAME
    if bundled_summary.exists():
        newline_bytes = bundled_summary.read_bytes()
    else:
        newline_bytes = summary_csv.read_bytes()
    line_terminator = "\r\n" if b"\r\n" in newline_bytes else "\n"
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = [column for column in published_columns if column not in fieldnames]
        if missing:
            raise PipelineError(
                "The regenerated attack summary is missing published columns: " + ", ".join(missing)
            )
        extra = [column for column in fieldnames if column not in published_columns]
        rows = [{column: row.get(column, "") for column in published_columns} for row in reader]

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=published_columns, lineterminator=line_terminator)
        writer.writeheader()
        writer.writerows(rows)
    return extra


def _discover_scan_sources(search_roots: list[Path]) -> list[ScanSource]:
    discovered: list[ScanSource] = []
    seen: set[tuple[Path, Path]] = set()
    for root in search_roots:
        if not root.exists():
            continue
        for candidate_csv in sorted(root.rglob("*global_hotspot_ancestry_scan*.csv")):
            if "catalog" in candidate_csv.name:
                continue
            try:
                with candidate_csv.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    fieldnames = set(reader.fieldnames or [])
            except OSError:
                continue
            if not {"candidate_id", "Q", "G", "R", "t0", "t1", "threshold_mode", "component_id"}.issubset(fieldnames):
                continue
            prefix = candidate_csv.name.split("_global_hotspot_ancestry_scan", 1)[0]
            family_label = "top3" if "_top3_" in candidate_csv.name else "top2"
            run_dir_name = f"{prefix}_live_recentered"
            run_dir = candidate_csv.parent / run_dir_name
            if not run_dir.exists():
                matches = sorted(root.rglob(run_dir_name))
                run_dir = matches[0] if matches else run_dir
            key = (candidate_csv.resolve(), run_dir.resolve() if run_dir.exists() else run_dir)
            if key in seen:
                continue
            seen.add(key)
            discovered.append(
                ScanSource(
                    family_label=family_label,
                    run_dir=run_dir,
                    candidate_csv=candidate_csv,
                )
            )
    return discovered


def _resolve_candidate_from_tag(tag: str, search_roots: list[Path]) -> CandidateSpec:
    matches: list[tuple[float, float, str, ScanSource]] = []
    for source in _discover_scan_sources(search_roots):
        df = pd.read_csv(source.candidate_csv)
        part = df[df["candidate_id"].astype(str).str.contains(str(tag), case=False, regex=False)].copy()
        if part.empty:
            continue
        part["Q"] = pd.to_numeric(part["Q"], errors="coerce").fillna(float("-inf"))
        part["G"] = pd.to_numeric(part["G"], errors="coerce").fillna(float("-inf"))
        row = part.sort_values(["Q", "G"], ascending=[False, False]).iloc[0]
        candidate_id = str(row["candidate_id"])
        matches.append((float(row["Q"]), float(row["G"]), candidate_id, source))
    if not matches:
        raise PipelineError(
            f"No candidate rows containing {tag!r} were found. "
            "Provide --cache-dir or explicit --scan sources so the public runner can inspect candidate CSVs."
        )
    matches.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    _, _, candidate_id, source = matches[0]
    return CandidateSpec(
        case=_candidate_case(candidate_id),
        candidate_id=candidate_id,
        family_label=source.family_label,
    )


def _load_published_input_pack(root: Path) -> PublishedInputPack | None:
    manifest_path = (root / "data" / "inputs" / "near_degenerate_published" / PUBLISHED_INPUT_PACK_NAME).resolve()
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    pack_dir = manifest_path.parent.resolve()
    windows_value = payload.get("windows_csv")
    windows_csv = (pack_dir / str(windows_value)).resolve() if windows_value else None
    scan_sources: list[ScanSource] = []
    for entry in payload.get("scans", []):
        scan_sources.append(
            ScanSource(
                family_label=str(entry["family"]),
                run_dir=(pack_dir / str(entry["run_dir"])).resolve(),
                candidate_csv=(pack_dir / str(entry["candidate_csv"])).resolve(),
            )
        )
    return PublishedInputPack(
        pack_dir=pack_dir,
        manifest_path=manifest_path,
        windows_csv=windows_csv,
        scan_sources=scan_sources,
    )


def _apply_published_input_pack_defaults(
    root: Path,
    args: argparse.Namespace,
    *,
    published_mode: bool,
) -> PublishedInputPack | None:
    if not published_mode:
        return None
    pack = _load_published_input_pack(root)
    if pack is None:
        return None
    if not args.scan and pack.scan_sources:
        args.scan = [
            [source.family_label, str(source.run_dir), str(source.candidate_csv)]
            for source in pack.scan_sources
        ]
    if args.windows_csv is None and pack.windows_csv is not None:
        args.windows_csv = pack.windows_csv
    return pack


def _path_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _effective_results_dir(root: Path, args: argparse.Namespace, published_mode: bool, candidates: list[CandidateSpec]) -> Path:
    if args.results_dir is not None:
        return args.results_dir.resolve()
    if published_mode:
        return DEFAULT_PUBLISHED_RESULTS_DIR.resolve()
    slug = _slugify(candidates[0].candidate_id if len(candidates) == 1 else "_".join(spec.case for spec in candidates))
    return (DEFAULT_OUTPUTS_DIR / f"near_degenerate_pipeline_{slug}").resolve()


def _manifest_path(results_dir: Path, args: argparse.Namespace) -> Path:
    if args.manifest is not None:
        return args.manifest.resolve()
    return results_dir / "near_degenerate_pipeline_manifest.json"


def _search_roots(root: Path, cache_dir: Path | None) -> list[Path]:
    candidates: list[Path] = []
    if cache_dir is not None:
        resolved = cache_dir.resolve()
        candidates.append(resolved)
        if (resolved / "results").exists():
            candidates.append((resolved / "results").resolve())
    local_results = (root / "benchmarks" / "jhtdb_response_family" / "results").resolve()
    if local_results.exists():
        candidates.append(local_results)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _filtered_scan_csv_path(results_dir: Path, family_label: str) -> Path:
    return results_dir / "inputs" / f"selected_{family_label}_candidates.csv"


def _resolve_live_scan_sources(
    candidates: list[CandidateSpec],
    results_dir: Path,
    search_roots: list[Path],
    explicit_scans: list[list[str]] | None,
    *,
    dry_run: bool,
) -> dict[str, ScanSource]:
    discovered = []
    if explicit_scans:
        for family, run_dir_raw, candidate_csv_raw in explicit_scans:
            discovered.append(
                ScanSource(
                    family_label=str(family),
                    run_dir=Path(run_dir_raw),
                    candidate_csv=Path(candidate_csv_raw),
                )
            )
    else:
        discovered = _discover_scan_sources(search_roots)

    mapping: dict[str, ScanSource] = {}
    by_candidate = {spec.candidate_id: spec for spec in candidates}
    for source in discovered:
        if not source.candidate_csv.exists():
            continue
        df = pd.read_csv(source.candidate_csv, usecols=["candidate_id"])
        available = set(df["candidate_id"].astype(str))
        for candidate_id in by_candidate:
            if candidate_id in available and candidate_id not in mapping:
                mapping[candidate_id] = source

    missing = [spec for spec in candidates if spec.candidate_id not in mapping]
    if missing and not dry_run:
        joined = ", ".join(spec.candidate_id for spec in missing)
        raise PipelineError(
            "Could not discover live scan sources for: "
            f"{joined}. Provide --cache-dir or explicit --scan FAMILY RUN_DIR CANDIDATE_CSV inputs."
        )
    if missing and dry_run:
        for spec in missing:
            family = spec.family_label
            mapping[spec.candidate_id] = ScanSource(
                family_label=family,
                run_dir=results_dir / "inputs" / f"discover_{family}_run_dir",
                candidate_csv=_filtered_scan_csv_path(results_dir, family),
            )
    return mapping


def _write_filtered_scan_csvs(
    candidates: list[CandidateSpec],
    scan_sources: dict[str, ScanSource],
    results_dir: Path,
) -> dict[str, Path]:
    grouped: dict[tuple[str, Path], list[str]] = {}
    for spec in candidates:
        source = scan_sources[spec.candidate_id]
        key = (source.family_label, source.candidate_csv.resolve())
        grouped.setdefault(key, []).append(spec.candidate_id)

    outputs: dict[str, Path] = {}
    for (family_label, candidate_csv), candidate_ids in grouped.items():
        df = pd.read_csv(candidate_csv)
        selected = df[df["candidate_id"].astype(str).isin(set(candidate_ids))].copy()
        if selected.empty:
            raise PipelineError(f"{candidate_csv} did not contain the expected selected rows: {candidate_ids}")
        output_csv = _filtered_scan_csv_path(results_dir, family_label)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(output_csv, index=False)
        for candidate_id in candidate_ids:
            outputs[candidate_id] = output_csv
    return outputs


def _membership_artifacts(candidates: list[CandidateSpec], results_dir: Path) -> list[MembershipArtifact]:
    artifacts: list[MembershipArtifact] = []
    for candidate in candidates:
        for radius_dx in (0.1, 0.25):
            radius_tag = _radius_tag(radius_dx)
            stem = f"{candidate.family_label}_{candidate.case}_{radius_tag}"
            artifacts.append(
                MembershipArtifact(
                    candidate=candidate,
                    radius_dx=radius_dx,
                    membership_csv=results_dir / f"promoted_tube_family_membership_{stem}.csv",
                    overlap_summary_csv=results_dir / f"tube_pair_overlap_audit_{stem}_summary.csv",
                    overlap_pairs_csv=results_dir / f"tube_pair_overlap_audit_{stem}_summary_pairs.csv",
                )
            )
    return artifacts


def _renewal_outputs(results_dir: Path) -> dict[str, Path]:
    return {
        "r0p1": results_dir / "real_packing_weighted_renewal_audit_summary.csv",
        "r0p25": results_dir / "real_packing_weighted_renewal_audit_r0p25_summary.csv",
    }


def _restore_by_name(
    outputs: list[Path],
    search_roots: list[Path],
    *,
    dry_run: bool,
    strict: bool,
) -> tuple[list[Path], list[str]]:
    restored: list[Path] = []
    notes: list[str] = []
    for destination in outputs:
        if destination.exists():
            continue
        matches: list[Path] = []
        for root in search_roots:
            if not root.exists():
                continue
            matches.extend(path for path in root.rglob(destination.name) if path.is_file())
        if not matches:
            continue
        matches = sorted({path.resolve() for path in matches}, key=lambda path: str(path).lower())
        if len(matches) > 1:
            hashes = {_sha256(path) for path in matches}
            if strict and len(hashes) > 1:
                joined = ", ".join(str(path) for path in matches)
                raise PipelineError(f"Ambiguous restored artifacts for {destination.name}: {joined}")
            notes.append(
                f"Multiple restored matches found for {destination.name}; using {matches[0]}."
            )
        source = matches[0]
        restored.append(source)
        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    return restored, notes


def _verify_inputs_exist(paths: list[Path], *, label: str) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise PipelineError(f"{label} is missing required input files: {joined}")


def _run_commands(
    *,
    stage: str,
    commands: list[list[object]],
    cwd: Path,
    dry_run: bool,
) -> None:
    for command in commands:
        print(f"  command: {_command_text(command)}")
        if dry_run:
            continue
        completed = subprocess.run(command, cwd=cwd, check=False)
        if completed.returncode != 0:
            raise PipelineError(f"Stage {stage!r} failed with exit status {completed.returncode}.")


def _stage_record(
    *,
    stage: str,
    commands: list[list[object]],
    inputs: list[Path],
    outputs: list[Path],
    source: str,
    notes: list[str],
    used_jhtdb: bool,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "commands": [_command_text(command) for command in commands],
        "inputs": [str(path) for path in inputs],
        "outputs": [str(path) for path in outputs],
        "output_hashes": {str(path): _sha256(path) for path in outputs if path.exists() and not dry_run},
        "source": source,
        "used_jhtdb": used_jhtdb,
        "status": "dry_run" if dry_run else "completed",
        "notes": notes,
    }


def _print_stage_header(stage: str, *, mode_note: str) -> None:
    print(f"[stage:{stage}] {mode_note}")


def _stop_requested(target: str | None, stage: str) -> bool:
    if target is None:
        return False
    return PIPELINE_STAGES.index(stage) >= PIPELINE_STAGES.index(target)


def _build_manifest(
    *,
    root: Path,
    results_dir: Path,
    manifest_path: Path,
    args: argparse.Namespace,
    candidates: list[CandidateSpec],
    stage_records: list[dict[str, Any]],
    published_mode: bool,
) -> dict[str, Any]:
    published_input_pack = _load_published_input_pack(root) if published_mode else None
    published_input_pack_info: dict[str, Any] | None = None
    if published_input_pack is not None:
        configured_scans = []
        for family, run_dir, candidate_csv in args.scan or []:
            run_path = Path(run_dir).resolve()
            csv_path = Path(candidate_csv).resolve()
            configured_scans.append(
                {
                    "family": str(family),
                    "run_dir": str(run_path),
                    "candidate_csv": str(csv_path),
                    "from_bundled_pack": _path_within(run_path, published_input_pack.pack_dir)
                    and _path_within(csv_path, published_input_pack.pack_dir),
                }
            )
        windows_csv = args.windows_csv.resolve() if args.windows_csv else None
        published_input_pack_info = {
            "pack_dir": str(published_input_pack.pack_dir),
            "manifest_path": str(published_input_pack.manifest_path),
            "used_default_pack": bool(
                (windows_csv is not None and published_input_pack.windows_csv is not None and windows_csv == published_input_pack.windows_csv)
                or any(item["from_bundled_pack"] for item in configured_scans)
            ),
            "windows_csv": str(windows_csv) if windows_csv is not None else None,
            "windows_from_bundled_pack": bool(
                windows_csv is not None
                and published_input_pack.windows_csv is not None
                and windows_csv == published_input_pack.windows_csv
            ),
            "scan_sources": configured_scans,
        }
    final_csv = results_dir / FINAL_CSV_NAME
    final_plot_hashes = {
        str(results_dir / name): _sha256(results_dir / name)
        for name in FINAL_PLOT_NAMES
        if (results_dir / name).exists()
    }
    bundled_final = root / "data" / "results" / FINAL_CSV_NAME
    bundled_hash = _sha256(bundled_final) if bundled_final.exists() else None
    final_hash = _sha256(final_csv) if final_csv.exists() else None
    comparison: dict[str, Any] | None = None
    if bundled_hash and final_hash and published_mode:
        comparison = {
            "bundled_path": str(bundled_final),
            "bundled_sha256": bundled_hash,
            "generated_sha256": final_hash,
            "matches_bundled_csv": bundled_hash == final_hash,
        }

    all_inputs = sorted({path for record in stage_records for path in record["inputs"]})
    all_outputs = sorted({path for record in stage_records for path in record["outputs"]})

    manifest = {
        "timestamp_utc": _iso_now(),
        "git_commit": _git_commit(root),
        "python_version": sys.version,
        "platform": platform.platform(),
        "command_line": " ".join([sys.executable, *sys.argv]),
        "dataset": "isotropic1024coarse",
        "published_mode": published_mode,
        "candidate_ids": [spec.candidate_id for spec in candidates],
        "candidate_cases": [spec.case for spec in candidates],
        "radii": [0.1, 0.25],
        "multiplicities": [1, 4, 8, 16, 32, 64],
        "live_jhtdb_config": _live_jhtdb_config_source(),
        "published_input_pack": published_input_pack_info,
        "results_dir": str(results_dir),
        "cache_dir": str(args.cache_dir.resolve()) if args.cache_dir else None,
        "input_files": all_inputs,
        "output_files": all_outputs,
        "stages": stage_records,
        "final_csv": str(final_csv),
        "final_csv_hash": final_hash,
        "final_plot_hashes": final_plot_hashes,
        "published_comparison": comparison,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = find_repo_root()

    if args.candidate and args.candidate_tag:
        raise PipelineError("Use either --candidate or --candidate-tag, not both.")

    search_roots = _search_roots(root, args.cache_dir)
    published_mode = args.published_cases or not (args.candidate or args.candidate_tag)

    if args.candidate_tag:
        candidates = [_resolve_candidate_from_tag(args.candidate_tag, search_roots)]
        published_mode = False
    elif args.candidate:
        candidates = [
            CandidateSpec(
                case=_candidate_case(args.candidate),
                candidate_id=str(args.candidate),
                family_label=_candidate_family_label(args.candidate),
            )
        ]
        published_mode = False
    else:
        candidates = _published_candidate_specs(root)
        published_mode = True

    published_input_pack = _apply_published_input_pack_defaults(root, args, published_mode=published_mode)
    if published_mode and args.require_jhtdb and published_input_pack is None and (not args.scan or args.windows_csv is None):
        raise PipelineError(
            "The published live JHTDB lane needs the bundled small input pack under "
            f"{DEFAULT_PUBLISHED_INPUTS_DIR}. Restore that folder or provide explicit --scan and --windows-csv inputs."
        )

    results_dir = _effective_results_dir(root, args, published_mode, candidates)
    manifest_path = _manifest_path(results_dir, args)
    memberships = _membership_artifacts(candidates, results_dir)
    renewal_outputs = _renewal_outputs(results_dir)
    script_root = root
    python_exe = sys.executable
    stage_records: list[dict[str, Any]] = []
    live_scan_sources: dict[str, ScanSource] | None = None
    filtered_scan_csvs: dict[str, Path] | None = None
    live_jhtdb_config = _live_jhtdb_config_source()

    def ensure_live_scan_inputs(*, allow_placeholder: bool = False) -> tuple[dict[str, ScanSource], dict[str, Path]]:
        nonlocal live_scan_sources, filtered_scan_csvs
        if live_scan_sources is None:
            live_scan_sources = _resolve_live_scan_sources(
                candidates,
                results_dir,
                search_roots,
                args.scan,
                dry_run=allow_placeholder,
            )
        if filtered_scan_csvs is None:
            if allow_placeholder:
                filtered_scan_csvs = {
                    spec.candidate_id: _filtered_scan_csv_path(results_dir, spec.family_label) for spec in candidates
                }
            else:
                filtered_scan_csvs = _write_filtered_scan_csvs(candidates, live_scan_sources, results_dir)
        return live_scan_sources, filtered_scan_csvs

    def stage_skeleton() -> None:
        outputs = [
            results_dir / "skeleton_closure_catalog_top2_top3_details.csv",
            results_dir / "skeleton_closure_catalog_top2_top3_summary.csv",
        ]
        notes: list[str] = [
            "This stage either queries live JHTDB or reuses restored sparse-skeleton closure outputs.",
            "Dataset expectation: isotropic1024coarse.",
        ]
        if published_input_pack is not None and args.scan:
            notes.append(f"Bundled published input pack available at {published_input_pack.pack_dir}.")
        restored = []
        if args.use_existing_cache:
            restored, restore_notes = _restore_by_name(outputs, search_roots, dry_run=args.dry_run, strict=args.strict)
            notes.extend(restore_notes)
        source = "existing_artifact" if all(path.exists() for path in outputs) else "generated"
        if restored and all(path.exists() or args.dry_run for path in outputs):
            source = "restored_cache"
        commands: list[list[object]] = []
        need_live = not all(path.exists() for path in outputs)
        if need_live or args.dry_run:
            scan_sources, filtered_csvs = ensure_live_scan_inputs(allow_placeholder=args.dry_run)
            if need_live and not _has_live_jhtdb_config() and not args.dry_run:
                raise PipelineError(
                    "Skeleton outputs are missing and no explicit JHTDB credentials were found. "
                    "Use --use-existing-cache with restored skeleton artifacts or rerun with your own JHTDB access configured."
                )
            if args.windows_csv is None and need_live and not args.dry_run:
                raise PipelineError(
                    "A fresh skeleton run needs --windows-csv. Use the bundled published input pack under "
                    f"{DEFAULT_PUBLISHED_INPUTS_DIR} or provide an explicit live windows CSV."
                )
            if need_live and args.require_jhtdb and not _has_live_jhtdb_config() and not args.dry_run:
                raise PipelineError(
                    "Live JHTDB was requested via --require-jhtdb but no explicit JHTDB credentials were found "
                    f"in {', '.join(JHTDB_ENV_VARS)}."
                )
            command = [python_exe, str(BENCHMARK_DIR / "skeleton_closure_catalog.py")]
            seen_scan_triplets: set[tuple[str, str, str]] = set()
            for spec in candidates:
                source_spec = scan_sources[spec.candidate_id]
                triplet = (
                    source_spec.family_label,
                    str(source_spec.run_dir),
                    str(filtered_csvs[spec.candidate_id]),
                )
                if triplet in seen_scan_triplets:
                    continue
                seen_scan_triplets.add(triplet)
                command.extend(["--scan", *triplet])
            command.extend(
                [
                    "--windows-csv",
                    str(args.windows_csv or (results_dir / "inputs" / "missing_windows.csv")),
                    "--output-csv",
                    str(outputs[0]),
                    "--summary-csv",
                    str(outputs[1]),
                    "--dataset",
                    "isotropic1024coarse",
                    "--source-mode",
                    "pyjhtdb",
                ]
            )
            commands.append(command)
            notes.append(
                "If the outputs are not already restored, this stage uses live JHTDB "
                f"with config source {live_jhtdb_config} plus the selected windows CSV."
            )
        _print_stage_header("skeleton", mode_note=notes[0])
        print(f"  outputs: {', '.join(str(path) for path in outputs)}")
        if commands:
            _run_commands(stage="skeleton", commands=commands, cwd=script_root, dry_run=args.dry_run or not need_live)
            if need_live and not args.dry_run:
                _verify_inputs_exist(outputs, label="skeleton stage")
                source = "live_jhtdb"
        skeleton_inputs = [args.windows_csv.resolve()] if args.windows_csv else []
        if filtered_scan_csvs:
            skeleton_inputs.extend(filtered_scan_csvs[spec.candidate_id] for spec in candidates)
        stage_records.append(
            _stage_record(
                stage="skeleton",
                commands=commands,
                inputs=skeleton_inputs,
                outputs=outputs,
                source=source,
                notes=notes,
                used_jhtdb=source == "live_jhtdb",
                dry_run=args.dry_run,
            )
        )

    def stage_dichotomy() -> None:
        shell_output = results_dir / "skeleton_closure_shell_catalog_top2_top3.csv"
        outputs = [
            shell_output,
            results_dir / "targeted_dichotomy_audit_top2_top3_rows.csv",
            results_dir / "targeted_dichotomy_audit_top2_top3_summary.csv",
        ]
        notes = [
            "This stage consumes skeleton + first-shell closure CSVs and writes the targeted dichotomy rows/summary.",
            "It can rerun publicly once those upstream CSVs exist in the results directory.",
        ]
        restored_shell = []
        if args.use_existing_cache and not shell_output.exists():
            restored_shell, shell_notes = _restore_by_name([shell_output], search_roots, dry_run=args.dry_run, strict=args.strict)
            notes.extend(shell_notes)
        inputs = [
            results_dir / "skeleton_closure_catalog_top2_top3_details.csv",
            shell_output,
        ]
        commands: list[list[object]] = []
        source = "generated"
        if not shell_output.exists():
            if not inputs[0].exists() and not args.dry_run:
                raise PipelineError(
                    "The shell-catalog step needs skeleton_closure_catalog_top2_top3_details.csv first. "
                    "Restore or rerun the skeleton stage before continuing."
                )
            if not _has_live_jhtdb_config() and not args.dry_run:
                raise PipelineError(
                    "The shell-catalog output is missing and no explicit JHTDB credentials were found. "
                    "Restore skeleton_closure_shell_catalog_top2_top3.csv via --cache-dir or rerun with live JHTDB access."
                )
            scan_sources, filtered_csvs = ensure_live_scan_inputs(allow_placeholder=args.dry_run)
            shell_command = [python_exe, str(BENCHMARK_DIR / "skeleton_closure_shell_catalog.py")]
            seen_scan_triplets: set[tuple[str, str, str]] = set()
            for spec in candidates:
                source_spec = scan_sources[spec.candidate_id]
                triplet = (
                    source_spec.family_label,
                    str(source_spec.run_dir),
                    str(filtered_csvs[spec.candidate_id]),
                )
                if triplet in seen_scan_triplets:
                    continue
                seen_scan_triplets.add(triplet)
                shell_command.extend(["--scan", *triplet])
            shell_command.extend(
                [
                    "--output-csv",
                    str(shell_output),
                    "--dataset",
                    "isotropic1024coarse",
                    "--source-mode",
                    "pyjhtdb",
                ]
            )
            commands.append(shell_command)
            source = "live_jhtdb"
        if (inputs[0].exists() and (shell_output.exists() or commands)) or args.dry_run:
            commands.append(
                [
                    python_exe,
                    str(BENCHMARK_DIR / "targeted_dichotomy_audit.py"),
                    "--details-csv",
                    str(inputs[0]),
                    "--shell-csv",
                    str(inputs[1]),
                    "--output-rows-csv",
                    str(outputs[1]),
                    "--output-summary-csv",
                    str(outputs[2]),
                ]
            )
            if source != "live_jhtdb":
                source = "rerun_from_csv"
        elif args.use_existing_cache:
            restored, restore_notes = _restore_by_name(outputs[1:], search_roots, dry_run=args.dry_run, strict=args.strict)
            notes.extend(restore_notes)
            if restored:
                source = "restored_cache"
        if not commands and source != "restored_cache" and not args.dry_run:
            raise PipelineError(
                "The dichotomy stage needs either restored skeleton/shell outputs or the live upstream run that produces them."
            )
        if args.dry_run and not commands and source != "restored_cache":
            commands.append(
                [
                    python_exe,
                    str(BENCHMARK_DIR / "targeted_dichotomy_audit.py"),
                    "--details-csv",
                    str(inputs[0]),
                    "--shell-csv",
                    str(inputs[1]),
                    "--output-rows-csv",
                    str(outputs[0]),
                    "--output-summary-csv",
                    str(outputs[1]),
                ]
            )
        _print_stage_header("dichotomy", mode_note=notes[0])
        print(f"  inputs: {', '.join(str(path) for path in inputs)}")
        print(f"  outputs: {', '.join(str(path) for path in outputs)}")
        _run_commands(stage="dichotomy", commands=commands, cwd=script_root, dry_run=args.dry_run or source == "restored_cache")
        stage_records.append(
            _stage_record(
                stage="dichotomy",
                commands=commands,
                inputs=inputs,
                outputs=outputs,
                source=source,
                notes=notes,
                used_jhtdb=source == "live_jhtdb",
                dry_run=args.dry_run,
            )
        )

    def stage_membership() -> None:
        notes = [
            "This stage builds promoted tube-family memberships for the selected candidates/radii.",
            "It is the last expensive stage before the purely CSV-based overlap/renewal/attack reruns.",
        ]
        if published_input_pack is not None and args.scan:
            notes.append(f"Bundled published scan inputs available at {published_input_pack.pack_dir}.")
        commands: list[list[object]] = []
        restored, restore_notes = ([], [])
        if args.use_existing_cache:
            restored, restore_notes = _restore_by_name(
                [artifact.membership_csv for artifact in memberships],
                search_roots,
                dry_run=args.dry_run,
                strict=args.strict,
            )
            notes.extend(restore_notes)
        existing_outputs = all(artifact.membership_csv.exists() for artifact in memberships)
        if restored and all(artifact.membership_csv.exists() or args.dry_run for artifact in memberships):
            source = "restored_cache"
        elif existing_outputs:
            source = "existing_outputs"
        else:
            source = "generated"
        need_live = not existing_outputs
        if need_live or args.dry_run:
            scan_sources, filtered_csvs = ensure_live_scan_inputs(allow_placeholder=args.dry_run)
            if need_live and not _has_live_jhtdb_config() and not args.dry_run:
                raise PipelineError(
                    "Membership outputs are missing and no explicit JHTDB credentials were found. "
                    "Restore membership CSVs via --cache-dir or rerun with live JHTDB access."
                )
            if need_live and args.require_jhtdb and not _has_live_jhtdb_config() and not args.dry_run:
                raise PipelineError(
                    "Live JHTDB was requested via --require-jhtdb but no explicit JHTDB credentials were found."
                )
            for artifact in memberships:
                source_spec = scan_sources[artifact.candidate.candidate_id]
                commands.append(
                    [
                        python_exe,
                        str(BENCHMARK_DIR / "promoted_tube_family_builder.py"),
                        "--scan",
                        source_spec.family_label,
                        str(source_spec.run_dir),
                        str(filtered_csvs[artifact.candidate.candidate_id]),
                        "--summary-csv",
                        str(results_dir / "targeted_dichotomy_audit_top2_top3_summary.csv"),
                        "--candidate-id",
                        artifact.candidate.candidate_id,
                        "--tube-radius-dx",
                        f"{artifact.radius_dx:g}",
                        "--output-csv",
                        str(artifact.membership_csv),
                        "--dataset",
                        "isotropic1024coarse",
                        "--source-mode",
                        "pyjhtdb",
                    ]
                )
        if not commands and source not in {"restored_cache", "existing_outputs"} and not args.dry_run:
            raise PipelineError(
                "Membership outputs are missing. Provide restored membership CSVs via --cache-dir or the live scan inputs needed to rebuild them."
            )
        _print_stage_header("membership", mode_note=notes[0])
        print("  outputs:")
        for artifact in memberships:
            print(f"    {artifact.membership_csv}")
        _run_commands(stage="membership", commands=commands, cwd=script_root, dry_run=args.dry_run or source == "restored_cache")
        if commands and not args.dry_run:
            source = "live_jhtdb"
        stage_records.append(
            _stage_record(
                stage="membership",
                commands=commands,
                inputs=[results_dir / "targeted_dichotomy_audit_top2_top3_summary.csv"],
                outputs=[artifact.membership_csv for artifact in memberships],
                source=source,
                notes=notes,
                used_jhtdb=source == "live_jhtdb",
                dry_run=args.dry_run,
            )
        )

    def stage_overlap() -> None:
        notes = [
            "This stage reruns the source-weighted overlap audit directly from promoted membership CSVs.",
            "No live JHTDB access is used here; restored memberships are enough.",
        ]
        inputs = [artifact.membership_csv for artifact in memberships]
        commands: list[list[object]] = []
        source = "rerun_from_csv"
        if all(path.exists() for path in inputs):
            for artifact in memberships:
                commands.append(
                    [
                        python_exe,
                        str(BENCHMARK_DIR / "tube_pair_overlap_audit.py"),
                        "--membership-csv",
                        str(artifact.membership_csv),
                        "--output-csv",
                        str(artifact.overlap_summary_csv),
                        "--pair-output-csv",
                        str(artifact.overlap_pairs_csv),
                    ]
                )
        elif args.use_existing_cache:
            restored, restore_notes = _restore_by_name(
                [path for artifact in memberships for path in (artifact.overlap_summary_csv, artifact.overlap_pairs_csv)],
                search_roots,
                dry_run=args.dry_run,
                strict=args.strict,
            )
            notes.extend(restore_notes)
            if restored:
                source = "restored_cache"
        if not commands and source != "restored_cache" and not args.dry_run:
            raise PipelineError(
                "Overlap rerun needs membership CSVs. Restore them first or rerun the membership stage."
            )
        if args.dry_run and not commands and source != "restored_cache":
            for artifact in memberships:
                commands.append(
                    [
                        python_exe,
                        str(BENCHMARK_DIR / "tube_pair_overlap_audit.py"),
                        "--membership-csv",
                        str(artifact.membership_csv),
                        "--output-csv",
                        str(artifact.overlap_summary_csv),
                        "--pair-output-csv",
                        str(artifact.overlap_pairs_csv),
                    ]
                )
        _print_stage_header("overlap", mode_note=notes[0])
        _run_commands(stage="overlap", commands=commands, cwd=script_root, dry_run=args.dry_run or source == "restored_cache")
        stage_records.append(
            _stage_record(
                stage="overlap",
                commands=commands,
                inputs=inputs,
                outputs=[path for artifact in memberships for path in (artifact.overlap_summary_csv, artifact.overlap_pairs_csv)],
                source=source,
                notes=notes,
                used_jhtdb=False,
                dry_run=args.dry_run,
            )
        )

    def stage_renewal() -> None:
        notes = [
            "This stage reruns the packing-weighted renewal audit from membership + overlap outputs.",
            "It stays completely offline once those CSVs exist.",
        ]
        commands: list[list[object]] = []
        source = "rerun_from_csv"
        radius_groups = {
            "r0p1": [artifact for artifact in memberships if abs(artifact.radius_dx - 0.1) < 1e-9],
            "r0p25": [artifact for artifact in memberships if abs(artifact.radius_dx - 0.25) < 1e-9],
        }
        can_run = all(
            artifact.membership_csv.exists() and artifact.overlap_summary_csv.exists()
            for artifact in memberships
        )
        if can_run:
            for key, artifacts in radius_groups.items():
                command = [python_exe, str(BENCHMARK_DIR / "real_packing_weighted_renewal_audit.py")]
                for artifact in artifacts:
                    command.extend(["--membership-csv", str(artifact.membership_csv)])
                for artifact in artifacts:
                    command.extend(["--overlap-summary-csv", str(artifact.overlap_summary_csv)])
                command.extend(["--output-csv", str(renewal_outputs[key])])
                commands.append(command)
        elif args.use_existing_cache:
            restored, restore_notes = _restore_by_name(
                list(renewal_outputs.values()),
                search_roots,
                dry_run=args.dry_run,
                strict=args.strict,
            )
            notes.extend(restore_notes)
            if restored:
                source = "restored_cache"
        if not commands and source != "restored_cache" and not args.dry_run:
            raise PipelineError(
                "Renewal rerun needs both membership and overlap outputs. Restore them or rerun the earlier stages."
            )
        if args.dry_run and not commands and source != "restored_cache":
            for key, artifacts in radius_groups.items():
                command = [python_exe, str(BENCHMARK_DIR / "real_packing_weighted_renewal_audit.py")]
                for artifact in artifacts:
                    command.extend(["--membership-csv", str(artifact.membership_csv)])
                for artifact in artifacts:
                    command.extend(["--overlap-summary-csv", str(artifact.overlap_summary_csv)])
                command.extend(["--output-csv", str(renewal_outputs[key])])
                commands.append(command)
        _print_stage_header("renewal", mode_note=notes[0])
        _run_commands(stage="renewal", commands=commands, cwd=script_root, dry_run=args.dry_run or source == "restored_cache")
        stage_records.append(
            _stage_record(
                stage="renewal",
                commands=commands,
                inputs=[artifact.membership_csv for artifact in memberships] + [artifact.overlap_summary_csv for artifact in memberships],
                outputs=list(renewal_outputs.values()),
                source=source,
                notes=notes,
                used_jhtdb=False,
                dry_run=args.dry_run,
            )
        )

    def stage_attack() -> None:
        arr_summary = results_dir / "arr_deficit_attribution_audit_c185_final81_summary.csv"
        notes = [
            "This stage reruns the same_parent_split attack and rewrites the final public summary CSV.",
            "It consumes membership, overlap, renewal, and (for the bundled published lane) the c185 ARR summary.",
        ]
        if published_mode and args.use_existing_cache and not arr_summary.exists():
            restored, restore_notes = _restore_by_name([arr_summary], search_roots, dry_run=args.dry_run, strict=args.strict)
            notes.extend(restore_notes)
        if published_mode and not arr_summary.exists() and not args.dry_run:
            raise PipelineError(
                "The published lane needs arr_deficit_attribution_audit_c185_final81_summary.csv "
                "to retain the c185 placeholder row. Restore it with --cache-dir."
            )
        inputs = [artifact.membership_csv for artifact in memberships]
        inputs.extend(artifact.overlap_summary_csv for artifact in memberships)
        inputs.extend(list(renewal_outputs.values()))
        if published_mode:
            inputs.append(arr_summary)
        if not args.dry_run:
            _verify_inputs_exist(inputs, label="attack stage")

        command = [
            python_exe,
            str(BENCHMARK_DIR / "near_degenerate_multi_tube_packing_attack.py"),
            "--results-dir",
            str(results_dir),
            "--final-dir",
            str(results_dir / "reports"),
            "--attack-mode",
            "same_parent_split",
            "--output-prefix",
            "near_degenerate_multi_tube_packing_attack",
            "--multiplicities",
            "1",
            "4",
            "8",
            "16",
            "32",
            "64",
        ]
        for artifact in memberships:
            renewal_key = "r0p25" if abs(artifact.radius_dx - 0.25) < 1e-9 else "r0p1"
            command.extend(
                [
                    "--membership-case",
                    f"{artifact.candidate.case}_{_radius_tag(artifact.radius_dx)}",
                    artifact.candidate.case,
                    str(artifact.membership_csv),
                    str(artifact.overlap_summary_csv),
                    str(renewal_outputs[renewal_key]),
                ]
            )
        if not published_mode:
            command.extend(["--case-filter", *(sorted({candidate.case for candidate in candidates}))])
        _print_stage_header("attack", mode_note=notes[0])
        _run_commands(stage="attack", commands=[command], cwd=script_root, dry_run=args.dry_run)
        if published_mode and not args.dry_run:
            dropped_columns = _normalize_public_attack_summary(results_dir / FINAL_CSV_NAME, root=root)
            if dropped_columns:
                note = (
                    "Normalized the final summary back to the published public CSV schema and "
                    "dropped internal diagnostic columns: " + ", ".join(dropped_columns)
                )
                notes.append(note)
                print(f"[stage:attack] {note}")
        stage_records.append(
            _stage_record(
                stage="attack",
                commands=[command],
                inputs=inputs,
                outputs=[
                    results_dir / "near_degenerate_multi_tube_packing_attack_summary.csv",
                    results_dir / "near_degenerate_multi_tube_packing_attack_phi_pairs.csv",
                    results_dir / "near_degenerate_multi_tube_packing_attack_absorption_proxy.csv",
                    results_dir / "near_degenerate_multi_tube_packing_attack_status.tex",
                ],
                source="rerun_from_csv",
                notes=notes,
                used_jhtdb=False,
                dry_run=args.dry_run,
            )
        )

    def stage_plots() -> None:
        final_csv = results_dir / FINAL_CSV_NAME
        if not args.dry_run:
            _verify_inputs_exist([final_csv], label="plot stage")
        command = [
            python_exe,
            str(ROOT / "scripts" / "plot_near_degenerate_attack.py"),
            "--input-csv",
            str(final_csv),
            "--output-dir",
            str(results_dir),
        ]
        notes = [
            "This stage regenerates the three public PNGs from the final summary CSV.",
            "It uses the public plotting script, not the benchmark-internal figure writer.",
        ]
        _print_stage_header("plots", mode_note=notes[0])
        _run_commands(stage="plots", commands=[command], cwd=script_root, dry_run=args.dry_run)
        stage_records.append(
            _stage_record(
                stage="plots",
                commands=[command],
                inputs=[final_csv],
                outputs=[results_dir / name for name in FINAL_PLOT_NAMES],
                source="rerun_from_csv",
                notes=notes,
                used_jhtdb=False,
                dry_run=args.dry_run,
            )
        )

    def stage_verify_published() -> None:
        if not published_mode:
            return
        final_csv = results_dir / FINAL_CSV_NAME
        if not args.dry_run:
            _verify_inputs_exist([final_csv], label="verification stage")
        command = [
            python_exe,
            str(ROOT / "scripts" / "reproduce_near_degenerate_same_parent.py"),
            "--results-dir",
            str(results_dir),
            "--verify-derived",
            "--explain",
        ]
        notes = [
            "This stage runs the public derived verifier against the regenerated final CSV.",
            "It is only valid for the published c413/c309/c185 bundle.",
        ]
        print("[stage:verify] " + notes[0])
        _run_commands(stage="verify", commands=[command], cwd=script_root, dry_run=args.dry_run)
        stage_records.append(
            _stage_record(
                stage="verify",
                commands=[command],
                inputs=[final_csv],
                outputs=[],
                source="public_verifier",
                notes=notes,
                used_jhtdb=False,
                dry_run=args.dry_run,
            )
        )

    try:
        stage_skeleton()
        if _stop_requested(args.stop_after, "skeleton"):
            manifest = _build_manifest(
                root=root,
                results_dir=results_dir,
                manifest_path=manifest_path,
                args=args,
                candidates=candidates,
                stage_records=stage_records,
                published_mode=published_mode,
            )
            print(f"Wrote manifest to {manifest_path}")
            return 0

        stage_dichotomy()
        if _stop_requested(args.stop_after, "dichotomy"):
            manifest = _build_manifest(
                root=root,
                results_dir=results_dir,
                manifest_path=manifest_path,
                args=args,
                candidates=candidates,
                stage_records=stage_records,
                published_mode=published_mode,
            )
            print(f"Wrote manifest to {manifest_path}")
            return 0

        stage_membership()
        if _stop_requested(args.stop_after, "membership"):
            manifest = _build_manifest(
                root=root,
                results_dir=results_dir,
                manifest_path=manifest_path,
                args=args,
                candidates=candidates,
                stage_records=stage_records,
                published_mode=published_mode,
            )
            print(f"Wrote manifest to {manifest_path}")
            return 0

        stage_overlap()
        if _stop_requested(args.stop_after, "overlap"):
            manifest = _build_manifest(
                root=root,
                results_dir=results_dir,
                manifest_path=manifest_path,
                args=args,
                candidates=candidates,
                stage_records=stage_records,
                published_mode=published_mode,
            )
            print(f"Wrote manifest to {manifest_path}")
            return 0

        stage_renewal()
        if _stop_requested(args.stop_after, "renewal"):
            manifest = _build_manifest(
                root=root,
                results_dir=results_dir,
                manifest_path=manifest_path,
                args=args,
                candidates=candidates,
                stage_records=stage_records,
                published_mode=published_mode,
            )
            print(f"Wrote manifest to {manifest_path}")
            return 0

        stage_attack()
        if _stop_requested(args.stop_after, "attack"):
            manifest = _build_manifest(
                root=root,
                results_dir=results_dir,
                manifest_path=manifest_path,
                args=args,
                candidates=candidates,
                stage_records=stage_records,
                published_mode=published_mode,
            )
            print(f"Wrote manifest to {manifest_path}")
            return 0

        stage_plots()
        stage_verify_published()
        manifest = _build_manifest(
            root=root,
            results_dir=results_dir,
            manifest_path=manifest_path,
            args=args,
            candidates=candidates,
            stage_records=stage_records,
            published_mode=published_mode,
        )
        print(f"Wrote manifest to {manifest_path}")
        comparison = manifest.get("published_comparison")
        if comparison and comparison.get("matches_bundled_csv") is False:
            message = (
                "The regenerated published final CSV does not match the bundled public CSV hash. "
                "Inspect the manifest for the generated and bundled SHA256 values."
            )
            if args.strict:
                raise PipelineError(message)
            print(f"WARNING: {message}")
        return 0
    except PipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
