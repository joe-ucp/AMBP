from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

import amplification_lab as lab


def main() -> int:
    table = lab.artifact_table()
    print(table.to_string(index=False))
    missing_csv = table[(table["kind"].eq("csv")) & (~table["exists"])]
    if missing_csv.empty:
        print("\nAll required cached evidence CSVs are present.")
        return 0
    print("\nCached evidence CSVs are missing.")
    print("Use `python scripts/run_lab.py --mode synthetic` only for a no-data smoke test.")
    print("Synthetic mode is not JHTDB evidence.")
    print("To reproduce paper evidence, place release CSV artifacts in `data/results/`.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
