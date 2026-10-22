# M5-02 - Adversarial full-pipeline and device qualification

Problem/result: qualify the complete source-to-dispatch pipeline against adversarial semantics and the pinned device contract before timing acceptance.

Dependencies: M4-06, M4-07, M5-01.

Contract and implementation: run source, independent evaluator, MIR/native serial, interaction extraction, admitted CPU, generated CUDA, and library-assisted references where applicable. Cover all manifest families, boundary shapes, aliases, malformed caches, invalid layouts, completion failures, unsupported devices, and every source trap/fallback rule. Require nonzero counts in each required class and report generated versus library-assisted outputs separately. Device qualification requires the M3 `cuda` feature and exact preflight; CPU-only CI must still execute deterministic semantic tests.

Positive cases: qualified devices reproduce independent results, ownership/completion traces, numeric policy, source order, and required failure behavior across the full pipeline. Negative cases: pointer-chasing dependencies, overflow-sensitive regrouping, strict floating reassociation, malformed IR/cache, divergent barriers, device OOM, launch failure, timeout, and missing hardware reject or preserve CPU fallback.

Acceptance interface and evidence: add `issue_m5_02_full_pipeline_qualification`; run `cargo test --locked issue_m5_02_full_pipeline_qualification` and require a nonzero test count (zero-test success fails). Run the M5-01 CPU command and the separate `cargo run --locked --features cuda --bin invar-bench -- --manifest bench/manifest-v1.json --track generated --require-gpu --report target/m5-qualification.json` only after authorized device preflight. Record commit, issue IDs, digests, toolchains, device identity, nonzero counts, failure injections, and pending hardware checks.

Exclusions/stop: no timing gate, family removal, silent tolerance change, or missing-hardware pass. Complete only when every required pipeline stage is qualified with evidence. A pending/failed hardware stage permits a checkpointed stop but leaves this issue open and blocks M5-03.

Bound corpus size, device memory, compilation time, generated artifacts, and run deadline. A partial qualification must list exact families and target conditions; it cannot be summarized as full-pipeline success.
