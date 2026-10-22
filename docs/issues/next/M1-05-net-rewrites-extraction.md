# M1-05 — Executable net rewrites and extraction equivalence

Decision revision: 2026-09-23. This issue consumes the revised M1-04 typed
graph. It does not require a source-to-three-agent translation.

Dependencies: M1-02, M1-04.

Contract: implement only named, guarded graph rewrites. The first affine beta
rule rewires a closed, total root lambda application to an ordered `Let`; it declines
capturing/unknown-effect cases. The first map-fusion rule joins two total
wrapping-u32 maps into one extracted serial loop while preserving operation
order. Each successful rewrite changes topology, increments graph revision,
invalidates prior proofs, and re-verifies before replacing the original net.
An unavailable rule or resource guard leaves the original plan intact; an
internal verifier failure is an error.

Extract term and bulk serial plans from verified connections, and extract
public MIR from typed data wires. Compare original source, direct MIR/native
CPU, and graph-extracted MIR/native CPU on the same public programs. The map
fixture is an internal typed bulk region: ordinary source-loop recognition is
M2-02, so M1 does not claim automatic map compilation from source.

Positive cases: beta changes the extracted term body; fusion reduces two
serial loops to one on empty, singleton, and wrapping-boundary arrays;
direct and graph-extracted public MIR/native outcomes agree, including demand,
traps, steps, and cleanup where observable. Negative cases: stale proofs,
wrong graph connections, duplicate/missing ports, captured beta input,
unknown effect, and fusion resource limits reject or retain fallback.

Acceptance: `cargo test --locked issue_m1_05_rewrite_extraction_equivalence
-- --list` must select nonzero tests; run in Debug and Release, including the
`native` feature on a host with the pinned LLVM toolchain. Record accepted and
rejected rule IDs, graph revisions, extracted plan shapes, differential
outcomes, and direct-lowering controls. Keep full milestone gates from
`docs/validation.md` green.

Exclusions: no CPU scheduling, GPU, general source map recognition,
unrestricted beta reduction, or confluence claim for the typed extension.
