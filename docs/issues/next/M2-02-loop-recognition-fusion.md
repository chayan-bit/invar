# M2-02 — Ordinary loop recognition and fused map extraction

Problem/result: recognize one canonical counted loop and extract a lawful fused map while retaining a serial plan.

Dependencies: M2-01.

Contract and implementation: match only fixed-array counted loops with explicit bounds, independent element accesses, no hidden loop-carried dependence, total nontrapping body, no unknown calls/effects, and compatible ownership. Emit a structured fused map candidate through the admission record and M1 extraction path. Keep source order for each element and preserve the reference fallback. Reject physical overreads, aliasing, allocation/cleanup changes, and unsupported indexing.

Positive cases: empty, singleton, odd, ragged, and large bounded arrays produce the same outputs and traps as serial execution; forced serial and candidate paths agree. Negative cases: overlap, out-of-range uncertainty, effectful callbacks, overflow-capable arithmetic, divergent bodies, and failed guards reject or fall back.

Acceptance interface and evidence: add `issue_m2_02_fused_map_extraction`; run `cargo test --locked issue_m2_02_fused_map_extraction`. Record loop-match reasons, admission facts, candidate/fallback outputs, source spans, and cost metadata. Do not use wall-clock speed as legality evidence.

Exclusions/stop: no reductions, scan algebra, scheduler, GPU, or speculative dependence solver. Stop when one canonical loop family is bounded, reproducible, and fallback-safe.

Keep matching syntactic and bounded: a near miss should produce a recorded rejection reason, not trigger a heuristic solver. Verify physical storage accesses after lowering, including padding and allocation reuse assumptions, before marking a candidate admissible.

Record the exact source region and rule revision with each match so a changed checker cannot reuse a candidate created from older facts.

Store rejected matches for review.

## Concrete integration requirements

Accept ordinary bootstrap wrapping loops with changed variable names, permuted independent statements and unequal array lengths under validated guards. Derive map regions from typed MIR and transform through typed nets. Reject xs[i]=xs[i-1]+1, output aliasing input, stateful calls and potential traps. Fresh-output fusion must remove an intermediate materialization; loop ordering alone is not sufficient if source failure/cleanup is observable.

Follow the shared acceptance interface in `docs/implementation-plan.md`: list and verify a nonzero test selection, run the issue suite in Debug and Release, and retain full milestone Cargo gates. Commands are required deliverables, not existing functionality in the planning checkout.
