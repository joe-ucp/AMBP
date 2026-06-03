# Amplification Must Be Paid: Public Lab

This is the runnable companion package for the amplification-payment paper. It
is meant to be copied, opened, and run without the parent repository.

It contains the paper, a notebook, command-line runners, smoke tests, and an
included curated evidence cache in `data/results/`.

The claim under inspection is a proposed Third Law of Navier-Stokes ledger
behavior:

> amplification must be paid.

An attempted growth path should route into quotient redundancy, source payment,
physical charge, denominator payment, or an explicit unresolved row. The public
lab is built so a reader can run the attacks and look for an uncharged route.

## What This Package Shows

The lab recreates the three clearest evidence views from the paper:

1. Same-parent quotient redundancy: split labels merge back under quotienting.
2. Renewal-cascade jitter: perturbed cross-parent packets still pay physical
   charge.
3. Tail/ARR routing: annular tails and ARR deficits are assigned to named rows
   or left visibly unresolved.

It also includes a coherent residual attribution check.

This is an evidence notebook and reproducibility package for that claim. The
paper gives the accompanying ledger formalism; the notebook makes the pressure
tests runnable.

## Quickstart

From this folder:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_lab.py
python scripts/build_ledger_lab.py
python scripts/execute_notebook.py
python -m pytest -q
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

Cached mode is the default and the cached evidence artifacts are bundled.
Synthetic mode is only a no-data smoke test and is not JHTDB evidence:

```bash
python scripts/run_lab.py --mode synthetic
```

Open the notebook:

```bash
jupyter notebook notebooks/amplification_payment_lab.ipynb
```

Open the local visual:

```text
ledger_lab/index.html
```

The Ledger Lab is generated from the bundled cached evidence artifacts only.
It embeds no synthetic rows.

## Cached Evidence

The curated cached artifacts needed by the notebook are included in
`data/results/`. They are small derived summary artifacts, not raw JHTDB cache
dumps. Check them with:

```bash
python scripts/check_artifacts.py
python scripts/run_lab.py --write-json
```

The artifact inventory, row counts, byte counts, and SHA256 hashes are recorded
in `data/ARTIFACTS.md`.

If you want to refresh them from a larger local cache, stage by filename:

```bash
python scripts/stage_cached_artifacts.py C:\path\to\artifact-cache
```

The staging script searches recursively and copies only the expected artifact
names into `data/results/`.

## Package Layout

- `LICENSE`: MIT license for original package material.
- `NOTICE.md`: JHTDB-derived artifact attribution and evidence fence.
- `CITATION.cff`: citation metadata for the public lab.
- `paper/amplification_must_be_paid.pdf`: current conditional framework paper.
- `paper/amplification_must_be_paid.tex`: source for the current framework paper.
- `paper/technical_gates_k3_k7.pdf`: trimmed technical ledger for K3--K7.
- `paper/technical_gates_k3_k7.tex`: source for the technical gate ledger.
- `notebooks/amplification_payment_lab.ipynb`: transparent run-and-read lab.
- `notebooks/amplification_lab.py`: standalone analysis helpers used by the notebook and scripts.
- `ledger_lab/index.html`: local animated ledger visual generated from cached evidence.
- `scripts/run_lab.py`: CLI summary runner with a conclusion payload.
- `scripts/build_ledger_lab.py`: builds the local Ledger Lab visual from cached CSV artifacts.
- `scripts/execute_notebook.py`: executes the notebook into `outputs/`.
- `scripts/check_artifacts.py`: cached-artifact inventory check.
- `scripts/stage_cached_artifacts.py`: copies expected artifacts from a cache/release folder.
- `tests/`: public smoke tests.
- `data/results/`: curated cached evidence artifacts used by default.
- `data/ARTIFACTS.md`: derived artifact provenance and integrity manifest.
- `outputs/`: generated summaries and executed notebooks.

## What Counts as a Break

Each notebook run ends with a conclusion table. For every displayed mechanism,
it gives a current read, an implication, and a concrete way to break the claim.

The intended standard is simple: no escape route gets hidden in prose. It is
quotient-paid, source-paid, physical-charge-paid, denominator-assigned, or still
honestly unresolved.

## Evidence Fence

The computational results attack the mechanism. They make the proposed Third
Law falsifiable: a reader can rerun the checks, inspect the rows, and look for
an uncharged amplification path.

## Data Attribution

The curated artifacts in `data/results/` are derived summary evidence artifacts
from experiments using the Johns Hopkins Turbulence Databases (JHTDB). JHTDB
states that its data are made available under the Open Data Commons Attribution
License (ODC-By), which requires attribution. See `NOTICE.md`.
