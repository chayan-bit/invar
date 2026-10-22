# M5-02 CPU adversarial qualification

Issue #34 at base `39ee676ae7e59317577d403668f371f038781f07`.
This is CPU semantic preparation before the M4/GNET integration, M5-01 freeze, and hardware qualification.
The issue remains open and M5-03 remains blocked.
The local contract and current benchmark binary use `invar-bench`.

## Verbatim acceptance classification

(a) CPU/semantic, executable now.
(b) Device paths and exact preflight after prerequisite merges.
(c) M4-06, M5-01 freeze, or hardware/performance acceptance.
Each quoted sentence below is verbatim from the local issue contract.

| Criterion | Class | CPU slice / remaining gate |
| --- | --- | --- |
| Contract and implementation: run source, independent evaluator, MIR/native serial, interaction extraction, admitted CPU, generated CUDA, and library-assisted references where applicable. | a + b | CPU engines now; device tracks pending. |
| Cover all manifest families, boundary shapes, aliases, malformed caches, invalid layouts, completion failures, unsupported devices, and every source trap/fallback rule. | a + b + c | CPU corpus, aliases, malformed admission/MIR/net, completion faults now; device layouts/failures and frozen full family denominator pending. |
| Require nonzero counts in each required class and report generated versus library-assisted outputs separately. | a + b | Nonzero CPU classes now; separate unavailable generated/library rows. |
| Device qualification requires the M3 `cuda` feature and exact preflight; CPU-only CI must still execute deterministic semantic tests. | a + b | CPU suite now; device feature/preflight pending. |
| Positive cases: qualified devices reproduce independent results, ownership/completion traces, numeric policy, source order, and required failure behavior across the full pipeline. | b | CPU counterpart now; actual devices pending. |
| Negative cases: pointer-chasing dependencies, overflow-sensitive regrouping, strict floating reassociation, malformed IR/cache, divergent barriers, device OOM, launch failure, timeout, and missing hardware reject or preserve CPU fallback. | a + b | CPU dependencies/numeric/IR/cache/worker timeout now; device failures pending. |
| Acceptance interface and evidence: add `issue_m5_02_full_pipeline_qualification`; run `cargo test --locked issue_m5_02_full_pipeline_qualification` and require a nonzero test count (zero-test success fails). | a | Implemented; Debug/Release evidence to be recorded. |
| Run the M5-01 CPU command and the separate `cargo run --locked --features cuda --bin invar-bench -- --manifest bench/manifest-v1.json --track generated --require-gpu --report target/m5-qualification.json` only after authorized device preflight. | a + b + c | CPU validate-only now; CUDA command waits for preflight and freeze. |
| Record commit, issue IDs, digests, toolchains, device identity, nonzero counts, failure injections, and pending hardware checks. | a + b | CPU receipt now; no invented device identity. |
| Exclusions/stop: no timing gate, family removal, silent tolerance change, or missing-hardware pass. | a + b + c | No timing claim; unavailable targets retained. |
| Complete only when every required pipeline stage is qualified with evidence. | b + c | Issue remains open. |
| A pending/failed hardware stage permits a checkpointed stop but leaves this issue open and blocks M5-03. | b + c | Draft CPU PR; full qualification pending. |
| Bound corpus size, device memory, compilation time, generated artifacts, and run deadline. | a + b | Bounded CPU inputs/fuel/artifacts/runtime now; device memory pending. |
| A partial qualification must list exact families and target conditions; it cannot be summarized as full-pipeline success. | a + b + c | CPU-only scope and explicit denominator. |

## Runnable interface

```sh
cargo test --locked issue_m5_02_full_pipeline_qualification -- --list
cargo test --locked issue_m5_02_full_pipeline_qualification -- --nocapture
cargo test --locked --release issue_m5_02_full_pipeline_qualification -- --nocapture
cargo test --locked --features native issue_m5_02_full_pipeline_qualification -- --nocapture
cargo test --locked --release --features native issue_m5_02_full_pipeline_qualification -- --nocapture
cargo run --locked --features native --bin invar-qualify -- --report NEW_REPORT.json
cargo run --locked --bin invar-bench -- --manifest bench/manifest-v1.json --track generated --validate-only --report NEW_MANIFEST_REPORT.json
```

The qualification CLI uses the same Rust harness as the acceptance tests.
`--require-gpu` returns a typed nonzero refusal until every required generated and library-assisted row qualifies on actual hardware.
Unavailable targets stay in the denominator, including native engines in a build without the native feature.
Frontend refusals are `blocked` for every engine and separately counted; they are never reported as backend executions.
The CPU suite compares source results, source trap categories and byte spans, source semantic fuel, and cleanup traces exactly.
Generated array programs check every output element and scan prefix against independently computed widened modular arithmetic.
Whole-region CPU fallback is recorded and establishes no device coverage.
A complete passing CPU report may publish with explicit pending device rows.
A disagreement, unexpected refusal, or `--require-gpu` failure never publishes a report path or overwrites an earlier report.

## Scope and limits

The corpus is reproducible from seed `0x4d35303220261014`.
Parser/checker fuzzing uses 128 deterministic mutations bounded to 256 bytes.
Semantic fuel is at most 100000, net transitions at most 1000000, corpus size at most 256, and each source at most 65536 bytes.
CPU builds and tests run with a bounded memory limit, no swap and a bounded runtime.
No device crate, M4/GNET implementation, Python, robotics, or frozen manifest changes are part of this slice.
All frozen manifest families, their expanded shapes, unsupported device/completion injections, device traces, and hardware performance stay pending.

## Evidence

The qualification record is kept in `evidence/m5-02-cpu/qualification.json`.

The final CPU corpus has 124 programs, including 9 typed frontend refusals and all four source trap categories.
Admitted map, scan and reduction controls at length 257 assert actual two- and four-worker dispatch.
The net oracle has three explicit unavailable rows: borrowed views and the two nominal unit-conversion programs.
These are current GNET oracle coverage gaps; MIR/net extraction and the other CPU engines still run those programs.
At the time of recording, the complete repository library gate failed in an existing GNET assembler-control test on GCC 15.2.
No compiler/runtime or GNET fix is claimed by this CPU harness PR.
