# Near-Degenerate Summary Column Dictionary

This dictionary covers every column in `data/results/near_degenerate_multi_tube_packing_attack_summary.csv`.

The bundled published CSV has 24 completed `c413`/`c309` same-parent rows plus 1 incomplete `c185` placeholder row with `attack_status=not_run_missing_promoted_membership`. That placeholder is intentional; it marks a missing upstream membership input instead of silently dropping the case.

| Column | Meaning | Formula / code source | Kind | Depends on `M` | Depends on candidate/radius | Upstream source |
| --- | --- | --- | --- | --- | --- | --- |
| `case` | Physical case token such as `c413` or `c309`. | `natural["case"]` in the final attack row. | `raw` | No | Yes | `near_degenerate_multi_tube_packing_attack.py` |
| `candidate_id` | Exact ancestry-scan candidate identifier. | Passed through from the selected hotspot candidate row. | `raw` | No | Yes | `global_hotspot_ancestry_scan*.csv` |
| `radius_dx` | Promoted tube radius in units of `dx`. | Passed through from the promoted membership family. | `raw` | No | Yes | `promoted_tube_family_membership_*` |
| `M` | Split multiplicity used in the same-parent attack. | Loop variable in `_split_attack_stats`. | `raw` | Yes | No | attack arguments |
| `delta` | Quotient threshold for deciding whether a pair survives as non-redundant. | Attack parameter `delta`, published default `0.25`. | `raw` | No | No | attack arguments |
| `natural_tubes` | Number of natural promoted tubes before splitting. | Unique `tube_id` count in the promoted membership CSV. | `derived` | No | Yes | `promoted_tube_family_builder.py` |
| `pre_quotient_tubes` | Tube count after splitting but before quotienting. | `natural_tubes * M` in `same_parent_split`. | `derived` | Yes | Yes | `near_degenerate_multi_tube_packing_attack.py` |
| `post_quotient_tubes` | Tube count after quotienting. | Count of quotient roots; in the published same-parent rows it returns to `natural_tubes`. | `derived` | Yes | Yes | `near_degenerate_multi_tube_packing_attack.py` |
| `quotient_merged_fraction` | Fraction of extra split clones merged away. | `(pre - post) / (pre - natural)` with `0` when `pre <= natural`. | `derived` | Yes | Yes | `near_degenerate_multi_tube_packing_attack.py` |
| `natural_source` | Total positive source weight of the natural family. | Sum of `source_weight` over the promoted membership rows. | `cached` | No | Yes | `promoted_tube_family_membership_*` |
| `post_quotient_source` | Total source weight remaining after quotient. | In `same_parent_split` it is set equal to `natural_source`. | `derived` | No | Yes | `near_degenerate_multi_tube_packing_attack.py` |
| `eta_post` | Post-quotient captured-source fraction. | Hard-coded to `1.0` in `same_parent_split` because relabeling preserves source. | `derived` | No | No | `near_degenerate_multi_tube_packing_attack.py` |
| `raw_split_pair_pack_proxy` | Diagnostic pre-quotient split pressure. | `max(0, (M - 1) / 2)` in `same_parent_split`. | `diagnostic` | Yes | No | `near_degenerate_multi_tube_packing_attack.py` |
| `R_pack_star` | Post-quotient non-redundant packing cost. | `overlap_summary["R_pack"]`; zero in the published same-parent rows because quotienting removes the artificial duplicates. | `derived` | No | Yes | `tube_pair_overlap_audit_*_summary.csv` |
| `R_shape_fam_available` | Available family-level shape charge. | `positive_or_proxy(renewal["R_shape"], dichotomy["R_shape_branch"])`. | `derived` | No | Yes | renewal summary, then dichotomy fallback |
| `R_spread_fam_available` | Available family-level spread charge. | `positive_or_proxy(renewal["R_spread"], dichotomy["R_spread_branch"])`. | `derived` | No | Yes | renewal summary, then dichotomy fallback |
| `R_renew_fam_available` | Available family-level renewal charge. | `positive_or_proxy(renewal["R_renew_total"], 0.0)`. | `derived` | No | Yes | `real_packing_weighted_renewal_audit*_summary.csv` |
| `R_D_available` | Available deactivation charge. | `positive_or_proxy(renewal["R_D"], dichotomy["R_deactivation_branch"])`. | `derived` | No | Yes | renewal summary, then dichotomy fallback |
| `R_tail_available` | Available tail or residual charge. | `positive_or_proxy(dichotomy["R_tail_residual_branch"], 0.0)`. | `derived` | No | Yes | `targeted_dichotomy_audit_top2_top3_summary.csv` |
| `E_nu_coh_pos_available` | Available coherent positive viscous residual proxy. | `positive_or_proxy(dichotomy["E_nu"], 0.0)`. | `derived` | No | Yes | `targeted_dichotomy_audit_top2_top3_summary.csv` |
| `Fphys_star_available` | Available post-quotient physical-cost proxy. | `R_pack_star + R_spread_fam_available + R_shape_fam_available + R_renew_fam_available + R_D_available + R_tail_available + E_nu_coh_pos_available`. | `derived` | No | Yes | final attack row |
| `natural_N_eff_2` | Order-2 effective tube count of the natural family. | `_neff2(masses)` from natural per-tube source masses. | `diagnostic` | No | Yes | `near_degenerate_multi_tube_packing_attack.py` |
| `natural_N_eff_half` | Order-1/2 effective tube count of the natural family. | `_neff_half(masses)` from natural per-tube source masses. | `diagnostic` | No | Yes | `near_degenerate_multi_tube_packing_attack.py` |
| `attack_status` | Human-readable row classification. | Published same-parent rows use values such as `baseline_no_split`, `showcase_quotient_redundant`, and `not_run_missing_promoted_membership`. | `diagnostic` | Yes | Yes | `near_degenerate_multi_tube_packing_attack.py` |
| `missing_terms` | Explicitly omitted continuum terms. | Semicolon-joined missing-term list built by the attack script. | `diagnostic` | No | Yes | `near_degenerate_multi_tube_packing_attack.py` |
| `evidence_note` | Short prose note explaining why the row is evidence-bearing. | Fixed explanatory text written by the attack script. | `diagnostic` | Yes | No | `near_degenerate_multi_tube_packing_attack.py` |

## Reviewer Notes

`quotient_merged_fraction` is `0` at `M=1` because there are no extra clones to merge. It is `1` for the published `M>1` same-parent rows because every extra split label is merged back to its parent under the quotient.

`R_pack_star` is the post-quotient non-redundant packing cost, not the pre-quotient split pressure. In the published same-parent rows it is `0` because the quotient removes the artificial same-parent duplicates before any non-redundant packing branch survives.

`R_shape_fam_available` comes first from the renewal summary if a positive `R_shape` exists there; otherwise it falls back to `R_shape_branch` from the targeted dichotomy summary. For `c309`, both sources are `0`, so the public row stays `0`.

`Fphys_star_available` is exactly:

`R_pack_star + R_spread_fam_available + R_shape_fam_available + R_renew_fam_available + R_D_available + R_tail_available + E_nu_coh_pos_available`

`M` does not enter that sum in the published `same_parent_split` lane because the split only duplicates labels. After quotienting, the surviving physical family and its available ledger pieces are the same for every `M`.

`split_pair_path_proxy` is not a column in the current public summary CSV. Treat it as a legacy reviewer-facing alias for split-pair diagnostics. The current public outputs expose that information as:

- `raw_split_pair_pack_proxy` for the pre-quotient split pressure.
- `near_degenerate_multi_tube_packing_attack_phi_pairs.csv` for pair-level quotient geometry diagnostics.
