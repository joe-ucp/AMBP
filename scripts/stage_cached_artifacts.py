from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
sys.path.insert(0, str(ROOT / "notebooks"))

import amplification_lab as lab


def expected_names() -> list[str]:
    table = lab.artifact_table()
    return sorted(set(table["path"].str.rsplit("/", n=1).str[-1]))


def index_source(source: Path) -> dict[str, list[Path]]:
    indexed: dict[str, list[Path]] = {}
    for path in source.rglob("*"):
        if path.is_file():
            indexed.setdefault(path.name, []).append(path)
    return indexed


def stage(source: Path, dry_run: bool = False) -> tuple[list[str], list[str]]:
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    RESULTS.mkdir(parents=True, exist_ok=True)
    indexed = index_source(source)
    copied: list[str] = []
    missing: list[str] = []
    for name in expected_names():
        matches = indexed.get(name, [])
        if not matches:
            missing.append(name)
            continue
        if len(matches) > 1:
            options = "\n".join(f"  - {path}" for path in sorted(matches))
            raise ValueError(f"Multiple source artifacts named {name}; choose a source folder without duplicates:\n{options}")
        src = matches[0]
        dst = RESULTS / name
        if not dry_run:
            shutil.copy2(src, dst)
        copied.append(name)
    return copied, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage public cached artifacts by expected filename.")
    parser.add_argument("source", type=Path, help="Folder containing cached evidence artifacts.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    copied, missing = stage(args.source, args.dry_run)
    verb = "would copy" if args.dry_run else "copied"
    print(f"{verb} {len(copied)} artifacts into data/results")
    for name in copied:
        print(f"  + {name}")
    if missing:
        print(f"\nmissing {len(missing)} expected artifacts")
        for name in missing:
            print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
