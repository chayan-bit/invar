# Roadmap and acceptance gates

The detailed contract is [implementation-plan.md](implementation-plan.md), with dependency-ordered work in the [issue catalog](issues/README.md). M0 and the revised M1 have implementation evidence; later work follows explicit assigned issue scope and prerequisites. A roadmap does not expand an issue's assigned scope.

| Milestone | Deliverable | Exit gate |
| --- | --- | --- |
| M0 | Exact Rust reference language, checker, interpreter and CLI | Nine original issues, Debug/Release conformance; no native or parallel claim |
| M1 | Typed MIR, native LLVM, minimal reference reducer, typed graph and extraction | Source/MIR/native and typed-net agreement on supported fragments, demand checks, beta/map rewrites, connectivity mutation rejection; no source-to-primitive claim |
| M2 | Dependence analysis, ordinary-loop recognition, lawful scans and inferred recurrences, CPU parallelism | Independent oracle, order-sensitive algebra, rejected unsafe near-matches, guarded fallback at several worker counts |
| M3 | Rust CUDA runtime, device IR/codegen, maps/reductions/scans, dispatch/cache | Actual authorized device correctness, lifetime/error tests, forced-target honesty; mocks alone cannot pass |
| M4 | Numeric/layout contract, stencil/matrix/Tensor Core schedules, segmented/irregular work, tuning and broader compact summaries | Each supported class transforms ordinary high-level code through interaction extraction and passes semantic/physical verification |
| M5 | Independent baselines, full-pipeline adversarial qualification and performance report | All required correctness and frozen [CUDA-class targets](performance-acceptance.md) pass on a named target |
| GNET | [General executable-net coverage](general-gpu-v01.md): frozen source-to-net boundary (GNET-01), CPU net oracle and native lowering (GNET-02), compiled CUDA/Metal general control (GNET-03), specialization, residency and calibrated automatic dispatch (GNET-04), robotics and held-out corpus (GNET-05) | Ordinary source executes nontrivial dynamic work on both devices against the net oracle; measured device coverage and frozen-baseline performance stay separate metrics. Required before final M5 qualification, AI-02 and release acceptance; open: GNET-04 part 2 (#196) and the GNET-05 device census (#188, #218) |
| L1 | Modules, records/variants/Result, borrowed views, generics/closures, bytes/strings/C ABI, practical library/tooling | Real multi-module application, ownership/cleanup/foreign-boundary regressions and consistent reference/native semantics |
| L2 | Structured tasks/channels, cancellation and capability-based async IO | Trace/lifetime contracts, retained loans until completion, bounded resources and hosted application acceptance |

Open v0.1 items, with their evidence on main, remaining gate and blocker, are in the [acceptance ledger](remaining-work.md).

GPU development now follows the shared correctness foundation directly, before hosted concurrency. General-purpose language work may proceed after its listed dependencies; it is not removed from the project. The old revision's M3/M4/M5 labels meant general-purpose/hosted/GPU respectively; the table above and new issue IDs supersede that ordering. Existing M0 issue IDs remain unchanged.

## What is fixed and what remains an implementation gate

Rust-owned implementation, required interaction foundations, source semantics and the independent reference path are fixed directions. Backend versions, native ABI details, typed graph templates, floating/tensor syntax, and device capability are frozen in their owning issues before use. A source language extension updates grammar, oracle, checker, native lowering and diagnostics together. No unexplained TODO can serve as a legality fact.

[Backend research](research/backend-options.md) chooses Rust-generated CUDA through NVRTC/Driver API first; [interaction research](research/interaction-design.md) defines the small calculus and obligations; the [summary probe](research/summary-evidence.md) supplies bounded recurrence evidence. These are design and test inputs, not achieved compiler milestones.

## Beyond the initial product

Multiple GPU vendors, public proof/recipe packages, generic dynamic GPU net execution and comprehensive mechanized verification require separate proposals and evidence. They are not hidden prerequisites for implementing the current issue graph. CPU fallback stays functional for unsupported transformations. A performance gate failure remains visible until corrected or explicitly revised before a new measurement cycle; do not quietly lower a target after seeing results.

No paid machine, GPU job, release, package publication, deployment, merge or visibility change is implied by this roadmap. Existing host and cost rules continue to apply.
