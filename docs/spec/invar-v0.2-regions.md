# Invar v0.2: region rules

Status: DRAFT; section 8 onward of the v0.2 chapter `docs/spec/invar-v0.2-axes.md`, kept in its own file.
Counts: statements=7 citations=8
Size record: tokens=867 estimator=bytes/3 limit=6000
## 8. Rule levels

[NA8.1] This file holds section 8 onward of `docs/spec/invar-v0.2-axes.md`, follows its section 1 conventions and has its own 6,000-token budget. `tests/spec_examples.rs::SPECS`
[NA8.2] A C level selects the rules a region's plan may use (NA3.2), so each rule a plan names in `rule_identities` and each law of the normal-form route has exactly one lowest C level by the table below, and a test checks the table against the code row for row. `src/axes/rules.rs::RULES`
[NA8.3] Admission refuses a candidate plan that names a rule above its region's C level, or a rule the table lacks, with the obligation `CONTRACT_LEVEL` and evidence `unknown`, and the region runs its serial plan; the serial plan itself is never refused. `src/axes/rules.rs::gate`
[NA8.4] The normal-form route runs a whole program, so it applies no law above the lowest C level any function declares, and otherwise falls back with reason `contract-level`. `src/axes/rules.rs::laws_admitted`
[NA8.5] Every region record of `invar-explain-v6` carries `rule_levels`, the lowest C level of each entry of `rule_identities` in order, and a region refused by NA8.3 also carries `c_refused`, the refused rule and its level, null when the table lacks it. `schema/explain-v6.schema.json`
[NA8.6] Removing `rule_levels` and `c_refused` with the section 6 fields still turns the v6 document of an undeclared program into its v5 document byte for byte. `tests/numeric_tiers/explain_v6.rs::undeclared_documents_project_to_the_committed_v5_goldens`
[NA8.7] A contract that an intrinsic names at its call or that a fold tier applies (NA3.10) is not a region rule, and its level is the contract table's in section 3. `src/axes.rs::CONTRACTS`

| Rule | Level | Plan |
| --- | --- | --- |
| `counted-wrapping-map-v1` | c0 | map |
| `total-u32-map-fusion-v1` | c0 | fused map pair |
| `total-u32-dag-fusion-v1` | c0 | fused map chain |
| `sealed-wrapping-u32-laws-v1` | c0 | wrapping reduction |
| `ordered-affine-pair-reduction-v1` | c0 | affine pair reduction |
| `wrapping-affine-recurrence-v1` | c0 | affine scan |
| `pnf-l3-checked-sum-carry-v1` | c0 | checked sum carry |
| `pnf-l7-select-records-v1` | c0 | select with companions |
| L1 to L20 | c0 | normal-form route |

Note: every rule on `main` is exact, so no source program reaches `CONTRACT_LEVEL` today; the gate guards the rules that later levels add, and its tests tag a rule above C0 to reach it.
