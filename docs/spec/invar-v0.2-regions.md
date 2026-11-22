# Invar v0.2: region rules

Status: DRAFT; section 8 onward of the v0.2 chapter `docs/spec/invar-v0.2-axes.md`, kept in its own file.
Counts: statements=15 citations=16
Size record: tokens=1589 estimator=bytes/3 limit=6000
## 8. Rule levels

[NA8.1] This file holds section 8 onward of `docs/spec/invar-v0.2-axes.md`, follows its section 1 conventions and has its own 6,000-token budget. `tests/spec_examples.rs::SPECS`
[NA8.2] A C level selects the rules a region's plan may use (NA3.2), so each rule a plan names in `rule_identities` and each law of the normal-form route has exactly one lowest C level by the table below, and a test checks the table against the code row for row. `src/axes/rules.rs::RULES`
[NA8.3] Admission refuses a candidate plan that names a rule above its region's C level, or a rule the table lacks, with the obligation `CONTRACT_LEVEL` and evidence `unknown`, and the region runs its serial plan; the serial plan itself is never refused. `src/axes/rules.rs::gate`
[NA8.4] The normal-form route runs a whole program, so it applies no law above the lowest C level any function or loop declares, and otherwise falls back with reason `contract-level`. `src/axes/rules.rs::laws_admitted`
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

## 9. Loop regions

[NA9.1] `#[region(...)]` may also precede a loop statement, a counted, stepped or for-each `for`, a `while`, or a labelled loop before its label, with the keys, safe defaults, refusals and codes of NA4.1 to NA4.6; `#[numeric]` before a loop, `#[region]` before any other statement and a repeated attribute on one loop are `UNSUPPORTED_FEATURE`. `src/front/items.rs::declared_loop`
[NA9.2] A loop declaration governs every region inside that loop, its nested loops included, the innermost declared loop containing a region governs it, and a region in no declared loop takes its function's levels. `src/axes/loops.rs::region_levels`
[NA9.3] A loop declaration is complete: an absent key is the safe level, not the enclosing declaration's, so a loop may raise or lower its function's levels. `tests/numeric_tiers/loop_regions.rs::a_loop_declaration_governs_its_loop_and_the_loops_inside_it`
[NA9.4] The folds of a declared loop take the tier its own contract selects by NA4.7, and the strict tier when it names no fold contract, whatever its function declares. `tests/numeric_tiers/loop_regions.rs::a_declared_loop_folds_at_its_tier_on_every_engine`
[NA9.5] The governing levels are the region record's axis fields and the level the gate of NA8.3 applies. `src/axes/rules/tests.rs::a_loop_level_gates_its_own_plan`
[NA9.6] A loop declaration reaches no callee, a closure lifted from inside a declared loop is at the safe levels, and a loop declared inside a closure body keeps its declaration in the lifted function. `tests/numeric_tiers/loop_regions.rs::a_loop_declaration_reaches_neither_callee_nor_closure`
[NA9.7] A loop declaration at the safe levels admits the same plan, receipt and identity as no declaration. `tests/numeric_tiers/loop_regions.rs::a_safe_loop_declaration_admits_the_same_plan_and_identity`
[NA9.8] `--max-c` and `--max-n` refuse a loop declared above the cap with `CONTRACT_CAP` at its attribute, `numeric_contracts.levels` counts every declared loop, and `--region-report` adds one row per declared loop with its `loop` span. `tests/numeric_tiers/loop_regions.rs::caps_and_the_region_report_cover_declared_loops`
