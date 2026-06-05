# Amplification Must Be Paid: Public Lab

This is the runnable companion package for the amplification-payment paper. It
is meant to be copied, opened, and run without the parent repository.

It contains the paper, a notebook, command-line runners, smoke tests, and an
included bundle of small derived demo rows in `data/results/`.

This repo is not a full reproduction package. It is a public attack surface
for the amplification-payment ledger: a small, runnable place to inspect the
mechanism, try to break it, and see whether an uncharged route appears.

The claim under inspection is a proposed Third Law of Navier-Stokes ledger
behavior:

> amplification must be paid.

An attempted growth path should route into quotient redundancy, source payment,
physical charge, denominator assignment, or an explicit unresolved row. The public
lab is built so a reader can run the attacks and look for an uncharged route.

## What This Package Shows

The lab recreates the three clearest evidence views from the paper:

1. Same-parent quotient redundancy: split labels merge back under quotienting.
2. Renewal-cascade jitter: perturbed cross-parent packets still pay physical
   charge.
3. Tail/ARR routing: annular tails and ARR deficits are assigned to named rows
   or left visibly unresolved.

It also includes a coherent residual attribution check.

This is a public mechanism notebook, not a local data warehouse dump. The paper
gives the accompanying ledger formalism; the notebook makes the pressure tests
runnable.

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

Cached mode is the default and the bundled demo CSVs are included. Synthetic
mode is only a no-data smoke test and is not JHTDB data:

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

The Ledger Lab is generated from the bundled demo CSVs only.
It embeds no synthetic rows.

## Demo Rows

The bundled CSVs used by the notebook are included in `data/results/`. They are
small derived summary artifacts, not raw JHTDB cache dumps, and they do not
carry proof load. List them with:

```bash
python scripts/list_artifacts.py
python scripts/plot_near_degenerate_attack.py
python scripts/reproduce_near_degenerate_same_parent.py --explain --verify-derived
python scripts/run_lab.py --write-json
```

`data/ARTIFACTS.md` records row counts, byte counts, and hashes for orientation.
It is not an enforced reproducibility gate.

If you want to refresh them from a larger local cache, stage by filename:

```bash
python scripts/stage_cached_artifacts.py C:\path\to\artifact-cache
```

The staging script searches recursively and copies only the expected artifact
names into `data/results/`.

To audit or regenerate the near-degenerate same-parent lineage:

```bash
python scripts/reproduce_near_degenerate_same_parent.py --explain --verify-derived
python scripts/reproduce_near_degenerate_same_parent.py --dry-run --run-pipeline
```

The bundled public CSV is a derived offline summary from JHTDB-derived cached
artifacts. It is not synthetic and not a direct JHTDB export. Fresh JHTDB
replay uses the bundled published seed pack and the public JHTDB testing token;
cache-backed replay requires restored upstream artifacts.

The canonical reviewer-facing checklist for this CSV now lives in
`data/near_degenerate_same_parent_reproduction_map.md`. It is the public bridge
from producer scripts to input/output artifacts to column meanings to the
paper-side K3 / final-ledger mapping.

## End-To-End Near-Degenerate Same-Parent Reproduction

The public repo now exposes the upstream near-degenerate stage scripts under
`benchmarks/jhtdb_response_family/` together with a top-level public runner:

```bash
python scripts/run_near_degenerate_jhtdb_pipeline.py --published-cases --dry-run --results-dir outputs/near_degenerate_public_validation
python scripts/run_near_degenerate_jhtdb_pipeline.py --published-cases --use-existing-cache --cache-dir PATH_TO_RESTORED_UPSTREAM_RESULTS --results-dir outputs/near_degenerate_public_validation
python scripts/run_near_degenerate_jhtdb_pipeline.py --published-cases --require-jhtdb --results-dir outputs/near_degenerate_public_validation
python scripts/reproduce_near_degenerate_same_parent.py --explain --verify-derived
```

The layers are intentionally separate:

- Plot regeneration: `python scripts/plot_near_degenerate_attack.py`
  Regenerates the three bundled PNGs from the final summary CSV only.
- Derived summary audit: `python scripts/reproduce_near_degenerate_same_parent.py --explain --verify-derived`
  Explains and verifies the existing final CSV and, when component summaries are present, recomputes the available `Fphys*` sum from upstream summaries.
- Full pipeline runner: `python scripts/run_near_degenerate_jhtdb_pipeline.py ...`
  Restores or reruns the upstream component summaries and then reruns the overlap, renewal, attack, plot, and published-verifier chain.

There are now three practical rerun paths:

- Final-CSV plotting only: regenerate PNGs from the bundled public summary CSV.
- Cache-backed replay: point `--cache-dir` at restored upstream outputs and rerun the published lineage offline.
- Fresh published live replay: use the bundled seed pack under `data/inputs/near_degenerate_published/` together with the public JHTDB testing-token fallback.

The public bundle includes:

- `data/near_degenerate_same_parent_reproduction_map.md`
- `data/results/near_degenerate_column_dictionary.md`
- `data/results/near_degenerate_column_dictionary.json`

The public runner writes:

- `<results-dir>/near_degenerate_pipeline_manifest.json` (default published path: `outputs/near_degenerate_public_validation/near_degenerate_pipeline_manifest.json`)
- regenerated CSVs, figures, and intermediate stage outputs under the chosen `--results-dir`

Important constraints:

- The repo does not ship the 10GB+ raw or intermediate cache artifacts needed for a completely fresh long-form rerun.
- The published `c413` / `c309` live lane now ships a small seed-input pack under `data/inputs/near_degenerate_published/`.
- Private JHTDB credentials are optional for the published live lane: when `JHTDB_TOKEN` is not set, the public runner falls back to the bounded public testing token already defined in `benchmarks/jhtdb_response_family/config.py`.
- The public testing token may be rate-limited, and fresh live reruns can take a while.
- Other custom live reruns may still benefit from your own token or restored cache artifacts.
- Cached reproduction works when you point `--cache-dir` at restored upstream artifacts.
- Published reruns now default to `outputs/near_degenerate_public_validation/` so they do not churn tracked files under `data/results/`.
- Full reruns can be long-running.
- The bundled public summary CSV is a numerical exhibit and audit target, not a standalone proof of the conjecture.

The intended next step is for skeptics to generate their own rows from their
own turbulence data or JHTDB cache and see whether the ledger fails.

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
- `ledger_lab/index.html`: local animated ledger visual generated from bundled demo rows.
- `scripts/run_lab.py`: CLI summary runner with a conclusion payload.
- `scripts/build_ledger_lab.py`: builds the local Ledger Lab visual from cached CSV artifacts.
- `scripts/execute_notebook.py`: executes the notebook into `outputs/`.
- `scripts/list_artifacts.py`: simple inventory helper for bundled demo rows.
- `scripts/plot_near_degenerate_attack.py`: regenerates the bundled near-degenerate attack PNGs from the summary CSV.
- `scripts/run_near_degenerate_jhtdb_pipeline.py`: public end-to-end runner for the near-degenerate same-parent lineage.
- `scripts/reproduce_near_degenerate_same_parent.py`: audits or optionally rebuilds the data lineage behind the bundled same-parent attack CSV.
- `scripts/stage_cached_artifacts.py`: copies expected artifacts from a cache/release folder.
- `benchmarks/jhtdb_response_family/`: public copy of the near-degenerate stage scripts and helper modules needed for end-to-end reruns.
- `tests/`: public smoke tests.
- `data/results/`: bundled derived demo artifacts used by default.
- `data/inputs/near_degenerate_published/`: small published scan/window/start inputs for the fresh live c413/c309 lane.
- `data/ARTIFACTS.md`: derived artifact inventory and provenance notes.
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

The curated artifacts in `data/results/` are derived summary demo artifacts
from experiments using the Johns Hopkins Turbulence Databases (JHTDB). JHTDB
states that its data are made available under the Open Data Commons Attribution
License (ODC-By), which requires attribution. See `NOTICE.md`.
