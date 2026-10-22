# M4-07 - Bounded finite-state and matrix recurrence summaries

Problem/result: extend the M2-04 affine recurrence path with two small compiler-internal summary families: finite-state transition tables and fixed-size matrix transitions. This broadens eligible recursive/loop computations while making the feasibility ceiling explicit and retaining sequential fallback.

Dependencies: M2-04, M4-04.

Contract and implementation: introduce versioned `SummaryFamily` rules with exact identity, ordered composition, state action, numeric policy, and source-observation mode. The initial finite-state family permits `K <= 16` states and immutable transition tables; the matrix family permits dense dimension `d <= 4` with u32 coefficients/state and operations modulo 2^32. This exact ring supplies the matrix composition law; arbitrary floating/checked matrices are not admitted. Table entries and matrix coefficients may depend on constants or immutable input, but not on the evolving state in a way that requires synthesizing a new table or matrix during execution. Reject unbounded tables, `K > 16`, `d > 4`, state-dependent coefficients, unknown shape, effects, cleanup, trapping arithmetic, and non-associative composition. A cost-feasibility guard estimates table `O(K)` or matrix `O(d^3)` composition/application plus temporary/residency cost; false or unknown selects the original loop. Route accepted summaries through the required typed interaction encoding and extraction to ordered scan/reduction operations. Support final-state-only and per-step observations separately.

Positive cases: `K=1,2,16`, `d=1,2,4`, zero/singleton/odd/ragged lengths, nonidentity seeds, segmented independent instances, exhaustive small-state table composition, matrix identity/associativity fixtures, and source/reference agreement for every observed prefix. Negative cases: `K=17`, `d=5`, unbounded or runtime-sized tables, coefficients depending on the evolving state, checked overflow, strict floating reassociation, non-associative operators, hidden calls/effects, aliasing, and a cost guard that is false. Every negative case retains serial execution and a reason code.

Acceptance interface and evidence: add `issue_m4_07_bounded_summary_families`; first require `cargo test --locked -- --list | rg -q 'issue_m4_07_bounded_summary_families'` (a missing match fails), then run `cargo test --locked issue_m4_07_bounded_summary_families`. Record family/rule IDs, bounds, identity/composition proofs, state/prefix comparisons, cost estimates, extraction artifacts, mutation tests, and fallback results. This compiler-internal issue has no GPU qualification; device lowering may consume an admitted summary later, but no GPU run is performed now.

Exclusions/stop: no universal recurrence synthesis, symbolic theorem prover, arbitrary semiring inference, matrices larger than `4x4`, tables larger than 16 states, state-dependent coefficient discovery, or performance claim. Stop when both bounded families are deterministic, cost-guarded, and source-order correct.

Shared acceptance: follow `docs/implementation-plan.md`, require a nonzero relevant test selection and run Debug/Release. Explicit GPU qualification must fail on missing hardware; ignored-only or mock-only success does not complete a hardware gate. Source-invalid programs remain errors in every engine; only optimization-specific unknown legality selects a lawful fallback.

## Current implementation scope (October 2, 2026)

`summary_source::host::run_source` executes verified typed table/matrix scans
on actually reached ordinary-source loops, retaining original fuel, allocation,
cleanup and ordered fallback. It publishes only a private same-source completion
after rechecking runtime inputs, state, extents and loans. Its private logical
budget includes both immutable preparation and typed execution. This is CPU
component functionality; native/device lowering and calibrated baseline gates
remain separate.

Device lowering (October 14, 2026): `Prepared::device_plan` lowers a prepared candidate through its freshly verified typed scan to the versioned `invar-bounded-summary-scan-v3` plan (`crates/invar-device/src/summary.rs`).
Both targets run prefixes as one ordered look-back scan; Metal bounds every wait and lets a waiting threadgroup reduce an unpublished predecessor itself, because Metal has no inter-threadgroup forward-progress guarantee. Both match the CPU oracle bit for bit at every observed prefix for `K = 1, 2, 3, 16` and `d = 1, 2, 3, 4`.
`K = 17`, `d = 5` and state-dependent coefficients stay serial with the reason codes `summary_family_bound` and `summary_state_dependent`.
Device results come from the runtime API; reached-source publication stays on the CPU host until M4-06 routes it.
Against device-to-device copy bandwidth measured on the same GPU, the largest admitted scans reach 0.770 to 0.836 (kernel unchanged since v2; final-head CUDA rerun pending) on the RTX 5070 Ti and 0.584 to 1.082 on the Apple M4 (Release and Debug).
The section 10 summaries row (at least 0.50 of copy bandwidth on both targets) is therefore met on both gate hosts; the M5-01 library-ratio and held-out gates are not yet measured.

## Independent baseline (from M5-01)

CUDA: CUB `DeviceScan` with a custom associative operator implementing the same table (K <= 16) or u32 matrix (d <= 4) composition. Metal: a reviewed handwritten Metal scan pinned by content hash. Its gate is measured through the M5-01 harness against the frozen `bench/manifest-v1.json` entry for this family and against the M5-01 held-out program set, on the RTX 5070 Ti (sm_120) and the Apple M4 (local Metal); the agent-written same-structure fixture is secondary and never sets the gate.

Amended 2026-10-14: the CUDA performance gates in this contract are measured on an NVIDIA RTX 5070 Ti (sm_120, driver 580.178.04) and, for Metal, on the Apple M4, replacing the earlier RTX A6000 target. No threshold value changes. Every fraction of peak is taken against a reference measured on that same GPU (device-to-device copy bandwidth, or the measured compute peak where this contract defines one), never a datasheet number.
