# Near-degenerate same-parent attack: reproduction map

This is the public reviewer map for `data/results/near_degenerate_multi_tube_packing_attack_summary.csv`.
It answers three questions in one place:

1. which code path produces the CSV
2. what each CSV column means
3. which paper-side object, gate, or evidence-table concept each column is tied to

This file is intentionally an artifact checklist, not a proof claim. The same-parent CSV is a mechanism-test output for the K3 quotient-or-charge lane. It does not prove the continuum inequalities, and it does not claim that the open K3-K7 technical gates are already closed.

## 1. Run order

Shortest audit of the bundled CSV:

```bash
python scripts/reproduce_near_degenerate_same_parent.py --explain --verify-derived
```

Dry-run of the published public pipeline:

```bash
python scripts/run_near_degenerate_jhtdb_pipeline.py --published-cases --dry-run --results-dir outputs/near_degenerate_public_validation
```

Cache-backed replay from restored upstream artifacts:

```bash
python scripts/run_near_degenerate_jhtdb_pipeline.py --published-cases --use-existing-cache --cache-dir PATH_TO_RESTORED_UPSTREAM_RESULTS --results-dir outputs/near_degenerate_public_validation
```

Fresh published live replay with the bundled seed-input pack:

```bash
python scripts/run_near_degenerate_jhtdb_pipeline.py --published-cases --require-jhtdb --results-dir outputs/near_degenerate_public_validation
```

Underlying stage order for the published lane:

1. seed the published c413/c309 lane from `data/inputs/near_degenerate_published/`
2. run `benchmarks/jhtdb_response_family/skeleton_closure_catalog.py`
3. run `benchmarks/jhtdb_response_family/targeted_dichotomy_audit.py`
4. run `benchmarks/jhtdb_response_family/promoted_tube_family_builder.py`
5. run `benchmarks/jhtdb_response_family/tube_pair_overlap_audit.py`
6. run `benchmarks/jhtdb_response_family/real_packing_weighted_renewal_audit.py`
7. run `benchmarks/jhtdb_response_family/near_degenerate_multi_tube_packing_attack.py --attack-mode same_parent_split`
8. normalize the regenerated summary back to the public contract in `scripts/run_near_degenerate_jhtdb_pipeline.py`
9. regenerate plots with `scripts/plot_near_degenerate_attack.py`
10. verify and explain the result with `scripts/reproduce_near_degenerate_same_parent.py --verify-derived --explain`

Direct producer note:

- `benchmarks/jhtdb_response_family/near_degenerate_multi_tube_packing_attack.py` is the script that writes the summary CSV with `summary.to_csv(summary_csv, index=False)`.
- `scripts/run_near_degenerate_jhtdb_pipeline.py` is the public orchestrator that builds the attack command, writes the manifest, and drops internal-only columns so the regenerated CSV matches the published public schema.
- `scripts/reproduce_near_degenerate_same_parent.py` verifies and explains the CSV; it is not the script that generates the CSV rows.

## 2. Producer scripts

| Path | Role in the lineage | Public output or effect |
| --- | --- | --- |
| `scripts/run_near_degenerate_jhtdb_pipeline.py` | Public runner for the published c413/c309 same-parent lane. | Writes `<results-dir>/near_degenerate_pipeline_manifest.json` and orchestrates the full replay. |
| `benchmarks/jhtdb_response_family/near_degenerate_multi_tube_packing_attack.py` | Direct summary producer for the same-parent attack. | Writes `near_degenerate_multi_tube_packing_attack_summary.csv`, `near_degenerate_multi_tube_packing_attack_phi_pairs.csv`, `near_degenerate_multi_tube_packing_attack_absorption_proxy.csv`, and `near_degenerate_multi_tube_packing_attack_status.tex`. |
| `scripts/plot_near_degenerate_attack.py` | Plot-only postprocessor. | Rewrites the three bundled PNGs from the final summary CSV. |
| `scripts/reproduce_near_degenerate_same_parent.py` | Public verifier and explanation bridge. | Checks row count, `Fphys_star_available`, M-invariance, and upstream component sums when component summaries are available. |
| `benchmarks/jhtdb_response_family/real_packing_weighted_renewal_audit.py` | Upstream family-level renewal/source-charge stage. | Produces the `real_packing_weighted_renewal_audit*.csv` summaries used by the attack row assembler. |
| `benchmarks/jhtdb_response_family/tube_pair_overlap_audit.py` | Upstream post-membership overlap audit. | Produces `tube_pair_overlap_audit_*_summary.csv` and pair diagnostics consumed by the attack script. |
| `benchmarks/jhtdb_response_family/promoted_tube_family_builder.py` | Upstream promoted-family builder. | Produces `promoted_tube_family_membership_*` CSVs for the c413/c309 finite-radius families. |
| `benchmarks/jhtdb_response_family/targeted_dichotomy_audit.py` | Upstream targeted dichotomy / branch-proxy stage. | Produces fallback branch proxies such as `R_shape_branch`, `R_spread_branch`, `R_deactivation_branch`, `R_tail_residual_branch`, and `E_nu`. |
| `benchmarks/jhtdb_response_family/skeleton_closure_catalog.py` | Earliest live/cache-restored stage in the published lane. | Produces upstream skeleton/catalog inputs needed before the later public stages can be rebuilt. |

## 3. Input artifacts

Bundled small public seed-input pack:

| Path | Status | Used by | Notes |
| --- | --- | --- | --- |
| `data/inputs/near_degenerate_published/published_input_pack.json` | bundled | public orchestrator | Declares the published c413/c309 seed-input pack. |
| `data/inputs/near_degenerate_published/material_heat_age_audit_patch17_h14_top2_published_candidates.csv` | bundled | live published lane | Selected ancestry-scan rows for the published c413/c309 candidates. |
| `data/inputs/near_degenerate_published/material_heat_age_audit_patch17_h14_top2_live_recentered/material_heat_age_starts.csv` | bundled | live published lane | Start table used by the live recentered run. |
| `data/inputs/near_degenerate_published/material_heat_age_audit_patch17_h14_top3_real_ns_scan_n81_windows.csv` | bundled | live published lane | Windows CSV defining the published time grid. |

Bundled attack-stage evidence inputs:

| Path | Status | Used by | Notes |
| --- | --- | --- | --- |
| `data/results/arr_deficit_attribution_audit_c185_final81_summary.csv` | bundled | attack stage | Retains the incomplete c185 placeholder row instead of silently dropping it. |
| `data/results/near_degenerate_column_dictionary.json` | bundled | documentation and tests | Machine-readable column contract for the published summary schema. |
| `data/results/near_degenerate_column_dictionary.md` | bundled | reviewer-facing dictionary | Human-readable dictionary for the published summary schema. |

Generated or restored upstream artifacts needed before the final attack CSV can be rebuilt:

| Path or pattern | Status | Produced by | Notes |
| --- | --- | --- | --- |
| `targeted_dichotomy_audit_top2_top3_summary.csv` | generated or restored | `targeted_dichotomy_audit.py` | Supplies branch-proxy fallback terms. |
| `promoted_tube_family_membership_top2_c413_r0p1.csv` | generated or restored | `promoted_tube_family_builder.py` | Membership input for the c413, `r/dx = 0.1` family. |
| `promoted_tube_family_membership_top2_c413_r0p25.csv` | generated or restored | `promoted_tube_family_builder.py` | Membership input for the c413, `r/dx = 0.25` family. |
| `promoted_tube_family_membership_top2_c309_r0p1.csv` | generated or restored | `promoted_tube_family_builder.py` | Membership input for the c309, `r/dx = 0.1` family. |
| `promoted_tube_family_membership_top2_c309_r0p25.csv` | generated or restored | `promoted_tube_family_builder.py` | Membership input for the c309, `r/dx = 0.25` family. |
| `tube_pair_overlap_audit_top2_*_summary.csv` | generated or restored | `tube_pair_overlap_audit.py` | Supplies `R_pack` for the row assembler. |
| `real_packing_weighted_renewal_audit_summary.csv` | generated or restored | `real_packing_weighted_renewal_audit.py` | Supplies the `r0p1` renewal/shape/spread/deactivation family terms. |
| `real_packing_weighted_renewal_audit_r0p25_summary.csv` | generated or restored | `real_packing_weighted_renewal_audit.py` | Supplies the `r0p25` renewal/shape/spread/deactivation family terms. |

External or large prerequisites that are intentionally not bundled:

- the large raw/intermediate JHTDB cache tree
- restored upstream benchmark results for long offline reruns
- live JHTDB access when no compatible cache is available

## 4. Output artifacts

Bundled public outputs:

| Path | Meaning |
| --- | --- |
| `data/results/near_degenerate_multi_tube_packing_attack_summary.csv` | Published final same-parent summary CSV. |
| `data/results/near_degenerate_multi_tube_packing_attack_absorption_proxy.csv` | Absorption-side proxy CSV written by the attack script. |
| `data/results/near_degenerate_attack_eta_vs_M.png` | Plot of `eta_post` versus `M`. |
| `data/results/near_degenerate_attack_fphys_vs_M.png` | Plot of `Fphys_star_available` versus `M`. |
| `data/results/near_degenerate_attack_results_figure.png` | Combined summary figure regenerated from the final CSV. |

Run-specific outputs under the chosen `--results-dir`:

| Path or pattern | Meaning |
| --- | --- |
| `<results-dir>/near_degenerate_pipeline_manifest.json` | Public provenance manifest written by the orchestrator. |
| `<results-dir>/near_degenerate_multi_tube_packing_attack_summary.csv` | Regenerated final summary CSV. |
| `<results-dir>/near_degenerate_multi_tube_packing_attack_phi_pairs.csv` | Pair-level quotient geometry diagnostics. |
| `<results-dir>/near_degenerate_multi_tube_packing_attack_absorption_proxy.csv` | Absorption-side proxy CSV. |
| `<results-dir>/near_degenerate_multi_tube_packing_attack_status.tex` | TeX status table emitted by the attack script. |
| `<results-dir>/reports/near_degenerate_multi_tube_packing_attack_report.tex` | Generated report TeX file for the rerun. |

## 5. Column dictionary

Every header in `data/results/near_degenerate_multi_tube_packing_attack_summary.csv` appears exactly once below.

| Column | Class | Producer / source | Meaning |
| --- | --- | --- | --- |
| `case` | implementation label | final attack row from `near_degenerate_multi_tube_packing_attack.py` | Physical case token such as `c413`, `c309`, or the incomplete `c185` placeholder row. |
| `candidate_id` | implementation label | ancestry-scan candidate carried through the membership and attack rows | Exact hotspot candidate identifier for the attacked promoted family. |
| `radius_dx` | implementation parameter | promoted membership CSV | Promoted-tube radius measured in grid spacings `dx`. |
| `M` | attack parameter | `_split_attack_stats(..., M=...)` | Same-parent split multiplicity used before quotienting. |
| `delta` | attack parameter | attack CLI / `_split_attack_stats(..., delta=...)` | Quotient threshold used to decide whether a pair survives as non-redundant. |
| `natural_tubes` | derived count | promoted membership table | Natural promoted-tube count before any adversarial split. |
| `pre_quotient_tubes` | derived count | `_split_attack_stats` | Tube count after label-splitting and before quotienting. |
| `post_quotient_tubes` | derived count | `_split_attack_stats` | Tube count after the same-parent quotient merges artificial clones. |
| `quotient_merged_fraction` | quotient proxy | `_split_attack_stats` | Fraction of extra split clones that are merged away by the quotient. |
| `natural_source` | cached source measurement | promoted membership table | Total positive source weight of the natural promoted family. |
| `post_quotient_source` | derived source measurement | `_split_attack_stats` | Source weight remaining after quotient; for the published same-parent lane it matches `natural_source`. |
| `eta_post` | paper-facing proxy | `_split_attack_stats` | Post-quotient captured-source fraction. |
| `raw_split_pair_pack_proxy` | diagnostic proxy | `_split_attack_stats` | Pre-quotient split pressure created by artificial duplicate labels. |
| `R_pack_star` | paper-facing proxy | overlap summary via `_read_pair_summary` | Post-quotient non-redundant packing cost available from the overlap audit. |
| `R_shape_fam_available` | paper-facing proxy | renewal summary with dichotomy fallback | Available family-level shape charge carried into the final physical-cost proxy. |
| `R_spread_fam_available` | paper-facing proxy | renewal summary with dichotomy fallback | Available family-level spread / separation charge carried into the final physical-cost proxy. |
| `R_renew_fam_available` | paper-facing proxy | renewal summary | Available family-level renewal charge carried into the final physical-cost proxy. |
| `R_D_available` | paper-facing proxy | renewal summary with dichotomy fallback | Available deactivation charge carried into the final physical-cost proxy. |
| `R_tail_available` | paper-facing proxy | dichotomy summary | Available tail or residual branch charge carried into the final physical-cost proxy. |
| `E_nu_coh_pos_available` | paper-facing proxy | dichotomy summary | Available coherent positive viscous residual proxy carried into the final physical-cost proxy. |
| `Fphys_star_available` | paper-facing proxy | final attack row assembled in `_split_attack_stats` | Sum of the available post-quotient physical-charge proxies used for the public same-parent evidence rows. |
| `natural_N_eff_2` | diagnostic concentration metric | `_neff2(masses)` in the attack script | Order-2 effective tube count of the natural family, weighted by source mass. |
| `natural_N_eff_half` | diagnostic concentration metric | `_neff_half(masses)` in the attack script | Order-1/2 effective tube count of the natural family, weighted by source mass. |
| `attack_status` | diagnostic status | final attack row | Human-readable row classification such as `baseline_no_split`, `showcase_quotient_redundant`, or `not_run_missing_promoted_membership`. |
| `missing_terms` | evidence fence | final attack row | Explicit semicolon-delimited list of continuum terms not evaluated in the cached public lane. |
| `evidence_note` | evidence fence | final attack row | Short prose note describing why the row is evidence-bearing. |

Related but not present in the final public summary CSV:

- pairwise `Phi_*` diagnostics live in `near_degenerate_multi_tube_packing_attack_phi_pairs.csv`, not in the final summary CSV
- the legacy reviewer label `split_pair_path_proxy` is not a current public column; the current public outputs expose that information as `raw_split_pair_pack_proxy` plus the pair-level `phi_pairs` CSV

## 6. Paper-symbol mapping

This table maps CSV columns to the paper-side object, gate, or evidence-table concept they are meant to illuminate. When a row is marked "implementation-only", that means the public CSV stores a finite-run label or diagnostic rather than a theorem-level symbol.

| CSV column or group | Paper symbol or concept | Paper source | Code variable / function | Mapping note |
| --- | --- | --- | --- | --- |
| `case` | finite candidate family labels `c413`, `c309`, `c185` used in the public evidence table | `paper/amplification_must_be_paid.tex` evidence table for the public attacks and audits | `natural["case"]` and `_missing_membership_case_rows(...)` | Evidence-table label, not a theorem symbol. |
| `candidate_id` | implementation-only ancestry-scan candidate identity | no paper formula symbol; this is a pipeline identifier | `natural["candidate_id"]` | Needed to trace provenance back to the ancestry-scan CSV, not to state a theorem. |
| `radius_dx` | finite-radius tag `r / dx` used in the public evidence table | `paper/amplification_must_be_paid.tex` evidence table references the `0.1 Delta x` and `0.25 Delta x` rows | `natural["radius_dx"]` | Experimental radius tag, not the continuum limit variable by itself. |
| `M` | same-parent split multiplicity shown in the evidence table (`M = 1, 4, 8, 16, 32, 64`) | `paper/amplification_must_be_paid.tex` same-parent split entry in the evidence table | loop over `args.multiplicities` in `_split_attack_stats` | Attack parameter for the empirical stress test, not a theorem-side invariant. |
| `delta` | quotient threshold `delta` in the K3 final quotient relation | `paper/amplification_must_be_paid.tex` and `paper/technical_gates_k3_k7.tex` around `d_phys^{final}` and quantitative quotient redundancy | attack CLI `--delta`, then `_split_attack_stats(..., delta=...)` | Exact attack parameter, not a derived proxy. |
| `natural_tubes`, `pre_quotient_tubes`, `post_quotient_tubes`, `quotient_merged_fraction` | quotient-redundancy bookkeeping for the `sim_delta^{final}` collapse and the nonredundant edge set | `paper/technical_gates_k3_k7.tex` quantitative final quotient redundancy and Gate K3 | `_split_attack_stats` | These are finite-family implementation summaries of the quotient branch, not standalone paper symbols. |
| `natural_source`, `post_quotient_source` | captured promoted source entering the `eta P(r)` statements | `paper/technical_gates_k3_k7.tex` quantitative final quotient redundancy and K3 mass lower bound | natural family stats plus `_split_attack_stats` | Rowwise finite-family source totals. They are not the paper's global `P(r)` symbol. |
| `eta_post` | realized captured-source fraction corresponding to the paper's `eta`-level capture | `paper/technical_gates_k3_k7.tex` quantitative quotient-redundancy definition and Gate K3 | `_split_attack_stats` | Rowwise realized fraction. In the published same-parent lane it is hard-coded to `1.0` because relabeling preserves source. |
| `raw_split_pair_pack_proxy` | implementation-only pre-quotient duplicate-pressure diagnostic | no direct paper symbol in the final ledger | `raw_pair_pack = max(0, (M - 1) / 2)` in `_split_attack_stats` | Diagnostic only. It is not a theorem-side row in `F_phys^{final}`. |
| `R_pack_star` | `R_{pack,delta}^{*,final}` or the nonredundant packing term in Gate K3 | `paper/technical_gates_k3_k7.tex` Gate K3 final quantitative quotient-or-charge | `post_pack = overlap.get("R_pack", 0.0)` | Public same-parent rows show `0` here because the quotient removes the artificial duplicates before a nonredundant packing branch survives. |
| `R_shape_fam_available` | shape component of the final physical ledger | `paper/amplification_must_be_paid.tex` final physical ledger and `paper/technical_gates_k3_k7.tex` branch-action shape terms | `r_shape = positive_or_proxy(renewal["R_shape"], proxy["R_shape_branch"])` | Family-level available proxy, not a full continuum proof term. |
| `R_spread_fam_available` | spread / separation component of the final physical ledger | `paper/amplification_must_be_paid.tex` spread/separation component of `F_phys` and `paper/technical_gates_k3_k7.tex` separation branch-action terms | `r_spread = positive_or_proxy(renewal["R_spread"], proxy["R_spread_branch"])` | Code uses `spread`; the paper language is spread / separation. |
| `R_renew_fam_available` | renewal component of the final physical ledger | `paper/amplification_must_be_paid.tex` final physical ledger and `paper/technical_gates_k3_k7.tex` `R_{renew}` branch-action terms | `r_renew = positive_or_proxy(renewal["R_renew_total"], 0.0)` | Family-level renewal proxy from the renewal audit summary. |
| `R_D_available` | deactivation channel corresponding conceptually to `M_deact(r)` | `paper/amplification_must_be_paid.tex` source deactivation row and `paper/technical_gates_k3_k7.tex` deactivation row in `D(r)` | `r_deact = positive_or_proxy(renewal["R_D"], proxy["R_deactivation_branch"])` | Code uses `R_D` / `R_deactivation_branch` as implementation names. Paper-side concept is the deactivation row. |
| `R_tail_available` | ARR / tail-routing contribution | `paper/amplification_must_be_paid.tex` annular renewal/tail reservoir `D_ARR(r)` and tail discussion | `r_tail = positive_or_proxy(proxy["R_tail_residual_branch"], 0.0)` | Tail-residual proxy only. It is not a full continuum tail proof evaluation. |
| `E_nu_coh_pos_available` | coherent positive viscous residual `E_nu^{coh,+}(r)` | `paper/amplification_must_be_paid.tex` and `paper/technical_gates_k3_k7.tex` coherent viscous residual row | `e_coh = positive_or_proxy(proxy["E_nu"], 0.0)` | Proxy pulled from the targeted dichotomy summary. |
| `Fphys_star_available` | rowwise public proxy for `F_phys^{final}` | `paper/amplification_must_be_paid.tex` and `paper/technical_gates_k3_k7.tex` `F_phys^{final}` | `fphys_available = post_pack + r_spread + r_shape + r_renew + r_deact + r_tail + e_coh` in `_split_attack_stats` | Available-proxy sum only. The script explicitly records missing continuum channels in `missing_terms`. |
| `natural_N_eff_2`, `natural_N_eff_half` | implementation-only concentration diagnostics | no named paper symbol | `_neff2(masses)` and `_neff_half(masses)` | Useful for auditing concentration of source mass in the natural family, but not a theorem object in the papers. |
| `attack_status` | public evidence-table classifier | `paper/amplification_must_be_paid.tex` evidence-table interpretations | `_split_attack_stats` and `_missing_membership_case_rows(...)` | Human-readable status, not a proof-side variable. |
| `missing_terms` | explicit evidence fence for unresolved continuum channels | `paper/amplification_must_be_paid.tex` and `paper/technical_gates_k3_k7.tex` open technical obligations and partial-closure framing | `_split_attack_stats` and `_missing_membership_case_rows(...)` | This is where the CSV tells you which continuum pieces were not computed. |
| `evidence_note` | short prose interpretation of the row's role | public evidence-table framing | `_split_attack_stats` and `_missing_membership_case_rows(...)` | Explanation field only, not a mathematical object. |

Pair-level note:

- the summary CSV does not contain direct `Phi_osc`, `Phi_env`, `Phi_act`, or `Phi_def` columns
- the pairwise quotient-geometry diagnostics live in `near_degenerate_multi_tube_packing_attack_phi_pairs.csv`
- for the published same-parent lane, same-parent baseline pairs are recorded there as zero-distance references and are merged by the quotient

## 7. Known limits

- This is mechanism testing and reproduction support, not a completed proof.
- The public same-parent summary CSV does not prove continuum inequalities.
- The attack script itself does not re-query JHTDB; full reruns depend on restored cache artifacts or a live JHTDB lane upstream of the attack stage.
- The public bundle does not include the large raw/intermediate cache tree.
- `Fphys_star_available` is an available-proxy sum, not a full continuum evaluation of every physical-charge channel.
- `missing_terms` is part of the contract: unresolved continuum pieces stay visible instead of being filled by proxy.
- `R_tail_available` is a tail-residual proxy, not a full tail proof term.
- `R_D_available` is an implementation-side deactivation proxy name that points toward the paper's deactivation channel; it is not a claim that the full deactivation theorem row has been proved from this CSV alone.
- The published `c185` row is intentionally incomplete. It records `attack_status=not_run_missing_promoted_membership` so the missing promoted-membership input stays explicit.
- The final public summary schema intentionally omits some internal diagnostic columns from the benchmark-local rerun and normalizes the regenerated CSV back to the published contract.

Bottom line:

- The code path is in the repo.
- The small published input pack is in the repo.
- The manifest path is in the repo.
- The column meanings are in the repo.
- The paper-to-code bridge is in the repo.
- The large raw/cache artifacts are intentionally not bundled.
