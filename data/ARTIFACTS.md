# Cached Artifact Manifest

Generated for the public lab release on 2026-06-02.

The files below are small derived summary artifacts used by the notebook and
CLI checks. They are not raw JHTDB cache dumps and do not replace the original
Johns Hopkins Turbulence Databases. Upstream data attribution requirements
continue to apply; see `../NOTICE.md`.

CSV row and column counts are included so reviewers can spot accidental schema
or truncation changes. SHA256 hashes are for the bytes currently bundled in
`data/results/`.

| Artifact | Kind | Rows | Columns | Bytes | SHA256 |
| --- | --- | ---: | ---: | ---: | --- |
| `arr_deficit_attribution_audit_c185_final81_summary.csv` | CSV | 1 | 24 | 901 | `dbd7080ff88b49d6f9a9da4d09114e29ec3336b8a2d8d8e4a00121915aa52680` |
| `coherent_viscous_residual_attribution_top2_c413.csv` | CSV | 4 | 21 | 2240 | `2b1dfd03359d57c21b0ab028befd9be383ee8f63e2ce3481f50f679cc7250216` |
| `near_degenerate_multi_tube_packing_attack_absorption_proxy.csv` | CSV | 11 | 8 | 1954 | `9a4d1c348d69a667ba87f81640bc392d1ac3b982b77fe4580cd78a6bda6d4ad9` |
| `near_degenerate_multi_tube_packing_attack_summary.csv` | CSV | 25 | 26 | 12121 | `dfd9740a01c32c1a273e63d1c4bab31fde69baee5366010c955b1b3ea2fd5bea` |
| `renewal_cascade_shape_topsource_phi_env_act_joint_hammer_r0p1_absorption_proxy.csv` | CSV | 11 | 8 | 1954 | `9a4d1c348d69a667ba87f81640bc392d1ac3b982b77fe4580cd78a6bda6d4ad9` |
| `renewal_cascade_shape_topsource_phi_env_act_joint_hammer_r0p1_summary.csv` | CSV | 96 | 51 | 80726 | `7ac8d8e9156e7de10cfdc5f582ee6c60cb7d160f0f25a86dc2ec169c8cf79528` |
| `tail_denominator_ladder_audit_top2_top3_rows.csv` | CSV | 15 | 12 | 2781 | `2d06589ceb8bd27b11ca79c7174547b43178506b731b1a427a992839aaa3929b` |
| `tail_denominator_ladder_audit_top2_top3_summary.csv` | CSV | 1 | 26 | 695 | `3aa721b37e8de7d20e37aee583838128a3537fc07ff3ecb28ff5c642cd6fcf97` |
| `near_degenerate_attack_eta_vs_M.png` | Figure |  |  | 49299 | `6614de1d72b4613d3d6f8d1438670319d7cbe2d10e44b2a1d5f5d85138e2066f` |
| `near_degenerate_attack_fphys_vs_M.png` | Figure |  |  | 48721 | `a452eaa8d2e0c990e19788edf426e85d237f575b08d2f538d5bc55a3b283b29c` |
| `near_degenerate_attack_results_figure.png` | Figure |  |  | 62479 | `226c1c5112b99403e3d768c19727c9ae9e6ffe8e69eabc00493a2686e151fb28` |
| `renewal_cascade_shape_topsource_phi_env_act_joint_hammer_r0p1_fphys_vs_M.png` | Figure |  |  | 273455 | `b9851700e07e597ce3d1b2e733787f610176dddd68525eff30b523fcc820eecf` |
| `renewal_cascade_shape_topsource_phi_env_act_joint_hammer_r0p1_post_growth_vs_jitter.png` | Figure |  |  | 199300 | `1d5f453e0cfaa82485f215c47b728888b726f4cfb6fa31744af4807d3f92175d` |
| `renewal_cascade_shape_topsource_phi_env_act_joint_hammer_r0p1_results_figure.png` | Figure |  |  | 418387 | `8568ff90e56d1694b4e56eb263568219558b8b75a41e3013dbe0ac1bde2a8ee8` |

## Evidence Fence

The cached rows are mechanism-test evidence for the public notebook. They do
not prove the continuum closure estimates named in the paper. In particular,
the tail denominator audit records named denominator pressure with unresolved
`D3` and `D6` families; it is not a below-one closure of every tail row.
