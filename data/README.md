# Data Folder

This folder contains the bundled derived demo artifacts used by the public
notebook. These are small summary artifacts, not raw JHTDB cache dumps.
See `ARTIFACTS.md` for row counts, byte counts, hashes, and provenance notes.

Synthetic mode is only a schema-compatible no-data smoke test and is not JHTDB
data.

Bundled demo CSV artifacts:

- `near_degenerate_multi_tube_packing_attack_summary.csv`
- `near_degenerate_multi_tube_packing_attack_absorption_proxy.csv`
- `renewal_cascade_shape_topsource_phi_env_act_joint_hammer_r0p1_summary.csv`
- `renewal_cascade_shape_topsource_phi_env_act_joint_hammer_r0p1_absorption_proxy.csv`
- `tail_denominator_ladder_audit_top2_top3_rows.csv`
- `tail_denominator_ladder_audit_top2_top3_summary.csv`
- `arr_deficit_attribution_audit_c185_final81_summary.csv`
- `coherent_viscous_residual_attribution_top2_c413.csv`

## Regenerating / Auditing the Near-Degenerate Same-Parent Attack

Use the public reproduction bridge to inspect where
`near_degenerate_multi_tube_packing_attack_summary.csv` comes from:

```bash
python scripts/reproduce_near_degenerate_same_parent.py --explain --verify-derived
python scripts/reproduce_near_degenerate_same_parent.py --dry-run --run-pipeline
```

The bundled CSV is an offline derived summary from JHTDB-derived cached
artifacts. It is not synthetic and not a direct JHTDB export. Full DNS/JHTDB
regeneration requires the upstream benchmark pipeline plus compatible seed
inputs or local caches.

`scripts/plot_near_degenerate_attack.py` regenerates figures from the final
summary CSV. `scripts/reproduce_near_degenerate_same_parent.py` audits or
rebuilds the data lineage that produced that CSV.

## End-To-End Near-Degenerate Same-Parent Reproduction

Use the public orchestrator when you want the full published c413/c309
same-parent lineage instead of just the final CSV audit:

```bash
python scripts/run_near_degenerate_jhtdb_pipeline.py --published-cases --dry-run --results-dir outputs/near_degenerate_public_validation
python scripts/run_near_degenerate_jhtdb_pipeline.py --published-cases --use-existing-cache --cache-dir PATH_TO_RESTORED_UPSTREAM_RESULTS --results-dir outputs/near_degenerate_public_validation
python scripts/run_near_degenerate_jhtdb_pipeline.py --published-cases --require-jhtdb --results-dir outputs/near_degenerate_public_validation
python scripts/reproduce_near_degenerate_same_parent.py --explain --verify-derived
```

What each layer does:

- Plot regeneration: rewrites the bundled PNGs from `near_degenerate_multi_tube_packing_attack_summary.csv`.
- Derived summary audit: checks the public final CSV and explains the lineage behind its repeated `Fphys*` values.
- Full pipeline runner: restores or reruns the upstream stage outputs, writes a provenance manifest, regenerates the final CSV, regenerates the plots, and then reruns the public verifier.

The three practical public paths are:

- final-CSV plotting only
- cache-backed replay from restored upstream artifacts
- fresh published live replay using `data/inputs/near_degenerate_published/`

Related public artifacts:

- `data/results/near_degenerate_column_dictionary.md`
- `data/results/near_degenerate_column_dictionary.json`
- `<results-dir>/near_degenerate_pipeline_manifest.json` when the orchestrator runs

Important limitations:

- This folder does not contain the huge raw/intermediate cache artifacts needed for a full fresh rerun.
- The published `c413` / `c309` live lane ships a small bundled input pack under `data/inputs/near_degenerate_published/`.
- Private JHTDB credentials are optional for that published live lane because the JHTDB client falls back to the public testing token when `JHTDB_TOKEN` is not set.
- The public testing token may be rate-limited, so fresh live reruns can be slow.
- Broader custom live reruns may still need restored caches or your own token/input bundle.
- Published reruns should target `outputs/near_degenerate_public_validation/` rather than rewriting tracked files in `data/results/`.
- The bundled final CSV remains a numerical exhibit, not a stand-alone proof certificate.

Optional figure artifacts used by the notebook when present:

- `near_degenerate_attack_eta_vs_M.png`
- `near_degenerate_attack_fphys_vs_M.png`
- `near_degenerate_attack_results_figure.png`
- `renewal_cascade_shape_topsource_phi_env_act_joint_hammer_r0p1_results_figure.png`
- `renewal_cascade_shape_topsource_phi_env_act_joint_hammer_r0p1_post_growth_vs_jitter.png`
- `renewal_cascade_shape_topsource_phi_env_act_joint_hammer_r0p1_fphys_vs_M.png`

## Attribution

These artifacts are derived from experiments using the Johns Hopkins Turbulence
Databases (JHTDB). JHTDB data are made available under the Open Data Commons
Attribution License (ODC-By), which requires attribution.

Use this short attribution when reusing these artifacts:

Data-derived summaries use data obtained from the Johns Hopkins Turbulence
Databases (JHTDB), https://flow.pha.jhu.edu.

See `../NOTICE.md` for citation details.
