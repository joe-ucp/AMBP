# Data Folder

This folder contains the curated cached evidence artifacts used by the public
notebook. These are derived summary artifacts, not raw JHTDB cache dumps.
See `ARTIFACTS.md` for row counts, byte counts, SHA256 hashes, and the
artifact-level evidence fence.

Synthetic mode is only a schema-compatible no-data smoke test and is not JHTDB
evidence.

Required cached CSV artifacts:

- `near_degenerate_multi_tube_packing_attack_summary.csv`
- `near_degenerate_multi_tube_packing_attack_absorption_proxy.csv`
- `renewal_cascade_shape_topsource_phi_env_act_joint_hammer_r0p1_summary.csv`
- `renewal_cascade_shape_topsource_phi_env_act_joint_hammer_r0p1_absorption_proxy.csv`
- `tail_denominator_ladder_audit_top2_top3_rows.csv`
- `tail_denominator_ladder_audit_top2_top3_summary.csv`
- `arr_deficit_attribution_audit_c185_final81_summary.csv`
- `coherent_viscous_residual_attribution_top2_c413.csv`

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
