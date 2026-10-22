# M1-01 — Typed control-flow MIR and independent evaluator

Problem/result: M0 has a checked tree interpreter but no stable typed control-flow boundary for native lowering or later extraction. Produce one typed MIR for the frozen source subset, plus an evaluator independent of the M0 tree evaluator, with source-order observations preserved.

Dependencies: M0-09.

Contract and implementation: lower checked functions into explicit blocks, typed values, branches, loops, calls, moves, loans, cleanup edges, and source spans. Keep checked arithmetic, trap categories, fuel semantics where applicable, and ordered effects explicit. Reject malformed MIR at construction and verification boundaries; unknown facts are not permissions. The MIR evaluator must not call the existing reference evaluator as its oracle. Keep all code safe Rust and use typed outcomes for traps, resource limits, and internal errors.

Positive cases: scalar and array examples, recursion, short-circuiting, loops, moves, call loans, cleanup, and every required trap agree with the M0 reference on completed runs and exact source provenance. Negative cases: invalid ownership joins, missing blocks, wrong types, malformed terminators, stale spans, and unsupported effects are rejected; divergent runs are bounded rather than declared equivalent.

Acceptance interface and evidence: add the future test name `issue_m1_01_typed_mir_independent_evaluator`; run `cargo test --locked issue_m1_01_typed_mir_independent_evaluator`. Record MIR schema/version, verifier results, differential corpus seeds, completed-run observations, and bounded failures. This test name is a deliverable, not an existing test. Compare both evaluators only on completed bounded cases and include shared-bug probes.

Exclusions/stop: no LLVM, interaction nets, optimization, new source syntax, native ABI, or GPU. Stop when the MIR verifier and independent evaluator have reviewable evidence for the M0 corpus and no semantic rule is hidden in lowering.

## Concrete integration requirements

Represent control flow with explicit block parameters/terminators, places and source spans; do not encode moves as host Rust moves. Verify dominance, types, initialized places and successor arity. Include checked MIN/-1 division, MIN%-1 == 0, RHS-before-index evaluation and an unselected trapping branch in source/MIR differential fixtures.

Follow the shared acceptance interface in `docs/implementation-plan.md`: list and verify a nonzero test selection, run the issue suite in Debug and Release, and retain full milestone Cargo gates. Commands are required deliverables, not existing functionality in the planning checkout.
