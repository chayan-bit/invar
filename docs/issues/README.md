# Implementation contracts

The nine M0 issues were completed with the initial Rust bootstrap and parser. Their original dependency order
remains the implementation record: bootstrap first, then front end, scalar
semantics, ownership, interpreter, adversarial corpus, and final acceptance.

```text
M0-01 -> M0-02 -> M0-03 -> M0-04 -> M0-05 -> M0-06 -> M0-07 -> M0-08 -> M0-09
```

For current consolidated priorities and open issue state, start with [Remaining work](../remaining-work.md). This page preserves the original acceptance contracts and dependency graph; its historical publication notes are not current task assignments.

Local issue bodies remain usable offline.

1. [M0-01 Rust Cargo bootstrap](01-rust-cargo-bootstrap.md)
2. [M0-02 lexer, parser, spans, diagnostics](02-lexer-parser-diagnostics.md)
3. [M0-03 names, types, scalar control flow](03-name-type-checker.md)
4. [M0-04 scalar reference evaluator](04-scalar-reference-evaluator.md)
5. [M0-05 arrays, moves, control-flow ownership](05-arrays-moves-ownership.md)
6. [M0-06 calls, loans, mutability conflicts](06-call-loans-mutability.md)
7. [M0-07 complete array interpreter and CLI JSON](07-array-interpreter-cli.md)
8. [M0-08 adversarial conformance and Rust tooling coverage](08-adversarial-conformance.md)
9. [M0-09 M0 acceptance and reproducible evidence](09-m0-acceptance.md)

The nine issues above are M0 only. The acceptance target is the offline reference interpreter described by [spec/bootstrap.md](../../spec/bootstrap.md), with syntax from [spec/grammar.ebnf](../../spec/grammar.ebnf). LLVM, GPU, networking, async execution, plugins, and native differential backends are out of this milestone.

## GitHub issues

| ID | Issue | Local acceptance contract |
| --- | --- | --- |
| M0-01 | done | [Rust Cargo bootstrap](01-rust-cargo-bootstrap.md) |
| M0-02 | done | [Lexer, parser, spans, and diagnostics](02-lexer-parser-diagnostics.md) |
| M0-03 | done | [Names, types, and scalar control flow](03-name-type-checker.md) |
| M0-04 | done | [Scalar reference evaluator with fuel and checked arithmetic](04-scalar-reference-evaluator.md) |
| M0-05 | done | [Arrays, moves, and control-flow ownership](05-arrays-moves-ownership.md) |
| M0-06 | done | [Calls, loans, mutability, and conflict checking](06-call-loans-mutability.md) |
| M0-07 | done | [Full array interpreter and CLI/JSON integration](07-array-interpreter-cli.md) |
| M0-08 | done | [Adversarial conformance, bounded corpus, and Rust tooling coverage](08-adversarial-conformance.md) |
| M0-09 | done | [Final M0 acceptance and reproducible evidence](09-m0-acceptance.md) |

The initial ordering is retained for history; do not restart completed M0 work.

## Full compiler and language plan

The following 32 additional contracts implement [the detailed plan](../implementation-plan.md). M1-01 and M1-03 landed with the initial Rust bootstrap; M1-02, revised M1-04, and M1-05 are implemented with [acceptance evidence](../m1-acceptance.md) in the M1 native and interaction foundations. The prerequisites below are authoritative for integration. The order is topological: M5-01 freezes the manifest after numeric semantics and before M4-06 tuning. Public source extensions require parser/reference/checker/native coverage, not IR-only fixtures.

M0/M1 acceptance commands now run against the Cargo implementation; later
issue commands remain implementation requirements. Require nonzero selected
tests; GPU hardware gates remain open until actually run.

| ID | Contract | Prerequisites | Issue |
| --- | --- | --- | --- |
| M1-01 | [Typed control-flow MIR and independent evaluator](next/M1-01-typed-mir-evaluator.md) | M0-09 | done |
| M1-02 | [LLVM native CPU and trap ABI](next/M1-02-llvm-native-cpu.md) | M1-01 | done |
| M1-03 | [Minimal interaction-combinator reducer](next/M1-03-minimal-interaction-reducer.md) | M0-09 | done |
| M1-04 | [Typed demand/ownership net verifier and source encoding](next/M1-04-typed-demand-net-verifier.md) | M1-01, M1-03 | done |
| M1-05 | [Executable net rewrites and extraction equivalence](next/M1-05-net-rewrites-extraction.md) | M1-02, M1-04 | done |
| M2-01 | [Dependence, effect, range analysis, and admission records](next/M2-01-effect-range-admission.md) | M1-05 | done |
| M2-02 | [Ordinary loop recognition and fused map extraction](next/M2-02-loop-recognition-fusion.md) | M2-01 | done |
| M2-03 | [Lawful reduction and ordered scan algebra](next/M2-03-lawful-reduction-scan.md) | M2-02 | done |
| M2-04 | [Compact affine recurrence inference](next/M2-04-affine-recurrence.md) | M2-03 | done |
| M2-05 | [CPU scheduler and serial fallback](next/M2-05-cpu-scheduler-fallback.md) | M2-04 | done |
| M3-01 | [Rust CUDA driver runtime ownership and completion](next/M3-01-cuda-driver-runtime.md) | M2-05 | done |
| M3-02 | [Device IR/verifier and generated CUDA toolchain](next/M3-02-device-ir-verifier.md) | M3-01 | done |
| M3-03 | [Coalesced/fused map kernels and transfers](next/M3-03-coalesced-fused-maps.md) | M3-02 | done |
| M3-04 | [Hierarchical reductions and ordered scans](next/M3-04-hierarchical-reductions-scans.md) | M3-03 | done |
| M3-05 | [Guarded GPU dispatch, cache, and failure integration](next/M3-05-guarded-gpu-cache.md) | M3-04 | done |
| M4-01 | [Versioned floating-point, tensor-shape, and layout semantics](next/M4-01-versioned-floating-point-tensor-shape-layout.md) | M3-05 | #1 |
| M4-02 | [Transpose and out-of-place stencil scheduling](next/M4-02-transpose-out-of-place-stencil.md) | M4-01 | #2 |
| M4-03 | [Generated matrix contraction and explicit Tensor Core schedules](next/M4-03-matrix-contraction-tensor-core.md) | M4-01, M4-02 | #3 |
| M4-04 | [Segmented/nested flattening and stable compaction](next/M4-04-segmented-nested-flattening.md) | M3-05, M4-01 | #4 |
| M4-05 | [Gather/scatter conflict semantics and bounded frontier work](next/M4-05-gather-scatter-frontier.md) | M4-04 | #5 |
| M4-07 | [Bounded finite-state and matrix recurrence summaries](next/M4-07-bounded-summary-families.md) | M2-04, M4-04 | #6 |
| M5-01 | [Rust benchmark manifest and independent CUDA baselines](next/M5-01-benchmark-manifest.md) | M3-05, M4-01 | #7 |
| M4-06 | [Bounded schedule tuning and residency-aware dispatch](next/M4-06-bounded-schedule-tuning-residency.md) | M4-03, M4-05, M5-01 | #8 |
| M5-02 | [Adversarial full-pipeline and device qualification](next/M5-02-full-pipeline-qualification.md) | M4-06, M4-07, M5-01 | #9 |
| M5-03 | [CUDA-class performance acceptance and report](next/M5-03-cuda-performance-acceptance.md) | M5-02 | #10 |
| L1-01 | [Modules, records, variants, Result, and pattern ownership](next/L1-01-modules-records-result.md) | M1-05 | done |
| L1-02 | [Precise borrowed views and lifetime-return rules](next/L1-02-borrowed-views.md) | L1-01, M2-01 | done |
| L1-03 | [Bounded generics and affine closures](next/L1-03-generics-closures.md) | L1-02 | done |
| L1-04 | [Bytes, strings, and explicit C ABI](next/L1-04-bytes-strings-cabi.md) | L1-03 | done |
| L1-05 | [Practical standard library, formatter, and interface tooling](next/L1-05-stdlib-tooling.md) | L1-04 | #11 |
| L2-01 | [Structured tasks, channels, and cancellation semantics](next/L2-01-structured-tasks.md) | L1-05, M2-05 | #12 |
| L2-02 | [Capability-based async IO and hosted acceptance](next/L2-02-capability-async-io.md) | L2-01 | #13 |

Issue numbers refer to this repository's tracker; `done` marks a completed contract.
