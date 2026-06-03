from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "amplification_payment_lab.ipynb"
OUTPUT = ROOT / "outputs" / "amplification_payment_lab_executed.ipynb"


def main() -> int:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(nb, timeout=180, kernel_name="python3")
    client.execute(cwd=str(ROOT))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, OUTPUT)
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
