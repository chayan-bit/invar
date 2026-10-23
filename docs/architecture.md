# Invar architecture, design revision 0.4

Status: implementation direction, not a proved system. The implemented language is specified in [the v0.1 specification](spec/invar-v0.1.md) and the [bootstrap specification](../spec/bootstrap.md). Planned work is tracked in GitHub issues.

## One meaning, multiple admitted implementations

Ordinary source has sequential reference meaning: left-to-right evaluation, ordered mutation, defined traps, deterministic normal cleanup, and no spontaneous external concurrency. Optimizations must preserve the relevant behavior, including whether demanded computation terminates. Explicit concurrent source will later define allowed traces and progress obligations separately.

For deterministic supported regions, the design obligation is:

```text
admitted(contract, candidate, target, policy) and assumptions_hold(input)
    implies reference_observations(input) == candidate_observations(input)
```

This equation is a requirement. Initially, the implication relies on reviewed compiler rules, tests, LLVM, runtime code, and hardware. No Boolean field or test count turns it into a theorem.

## Semantic kernel

The kernel defines types, evaluation order, integer behavior, moves, loans, control-flow joins, initialization, trap provenance, and cleanup. The reference interpreter evaluates checked source before optimization and is kept separate from transformation and native lowering code. Parsing and immutable source metadata may be shared; optimized evaluation logic may not be used as the reference oracle.

Implement Invar-owned compiler, interpreter, runtime, interaction machinery, and tooling in Rust 2024 with Cargo. Follow [Rust engineering](rust-engineering.md): safe Rust by default, typed outcomes, explicit source semantics, narrowly scoped dependencies, and a thin CLI. External LLVM/device drivers remain dependencies behind Rust boundaries; they are not rewritten as part of this commitment.

The first native backend uses direct LLVM lowering for the bounded integer/array core. Add MLIR only when structured array transformations justify its build and maintenance cost. No LLVM fork, universal coroutine lowering, graph-per-element interpreter, solver, or custom task runtime is required for the reference milestone.

## Shared contract, progressively enriched

```text
RegionContract {
  semantic_version, source_identity, typed_inputs, typed_outputs,
  ownership_and_loans, reads, writes, lifetime_actions,
  observable_effect_order, numeric_policy,
  demand_summary, failure_summary, cleanup_obligations,
  termination_information, reference_body_identity
}
```

Unknown is an explicit state, never an empty footprint or a proof of termination. Contracts describe logical semantics; physical memory/transfer budgets live on candidates. A bounded-loop totality proof includes the loop body and its calls. A bound on the loop counter alone is insufficient.

Demand, failure, and cleanup have one canonical representation each; projections reference them rather than keeping competing copies. Later contract clauses must be classified as statically established, runtime-enforced, or documentary. An unestablished clause is never an optimizer fact. Initial algebraic witnesses are sealed built-ins.

All analyses use one authoritative typed representation initially. When multiple IRs exist, every transformation declares invalidated facts, and evidence binds to immutable region content, rule revision, target, and relevant policy. A candidate may not refer to permissions from a previous region revision.

## Admission and trust

The implementation introduces an opaque admitted-candidate value when a real alternative execution plan exists. Only the admission module can construct it; code generation consumes it. Serial lowering also passes its applicable type, memory, and numeric checks.

Evidence is categorized per obligation:

| Category | Meaning |
| --- | --- |
| Checked structural invariant | A verifier checked types, dominance, bounds representation, or well-formedness; scope is explicit. |
| Trusted compiler rule | A named, reviewed transformation establishes a documented implication; implementation remains trusted. |
| Checked proof artifact | A specific proof checker and theorem scope are named; not available at bootstrap. |
| Runtime guard | A total, nonobservable predicate is checked while its input facts remain protected. |
| Unknown | Candidate is ineligible. |

Bounded differential testing is validation evidence, not a substitute for a missing legality rule. Unknown evidence kinds and unknown schema versions fail closed. Runtime guard false chooses the admitted fallback. An internal invariant violation produces a compiler error; it is not papered over as an optimization miss. Corrupt caches are discarded, validated inputs are rebuilt, and persistent failures are reported.

## First useful optimization path

1. Build and validate ordinary native code for the exact core.
2. Establish the required reference interaction calculus, typed connectivity verifier, and extraction correspondence. Recognize one canonical counted loop over fixed arrays. Prove its bounds and physical nonoverlap, and reject hidden loop-carried dependencies.
3. Represent the region in the typed interaction layer with its bulk structure retained, and admit a total, nontrapping wrapping map whose callback has no visible effects, resource cleanup, or unknown calls.
4. Retain a serial loop. Add coarse scoped CPU chunks using an existing standard platform facility, with explicit publication and joins.
5. Add a modular `u32` reduction with a built-in law identity, preserving operand order unless commutativity is specifically established.
6. Measure threshold decisions with end-to-end cost. An estimated speedup never overrides a failed legality check.

No worker starts for plain scalar execution. Bootstrap itself has no automatic parallelism. SIMD is a later native optimization with exact numeric semantics; a backend vectorizer is not evidence that all language-level transformations are lawful.

## Safety boundaries

Ownership is responsibility for lifetime; a borrow temporarily constrains access. Whole-array call loans come first. Escaping views, subrange splitting, escaping/stored/subrange reborrows, closures, and task transfer each require separate rules before support. Call-scoped whole-array reborrows are part of the bootstrap contract. Borrow rejection must not trigger implicit cloning or reference counting.

Logical disjointness must survive physical lowering. Packed bits, vector overreads, cache-aligned padding, asynchronous device accesses, and allocation reuse can violate a superficially correct source footprint. The physical checker must account for actual accessed storage before such plans become eligible.

Normal cleanup follows source rules. A terminal trap aborts the whole program without language-level unwinding; an implementation must not reinterpret it as a task-local exception that allows siblings to keep producing external effects. Environmental failure has a separate status. Later asynchronous cleanup must retain borrowed storage until actual completion; a cancellation request or timeout does not free it.

For future failure-capable transformations, observations include the source-ordered prefix of visible writes/effects before failure. A fresh-output plan cannot erase earlier committed source writes. Until an explicit ordered publication/fault protocol is specified and validated, initial parallel rules require total, nontrapping regions without external effects.

Foreign code is outside the safe-language theorem boundary. Safe wrappers publish effects, lifetime, thread-affinity, and numeric-state assumptions. Checking a pointer length or alignment cannot generally establish that foreign storage is live; raw-pointer introduction remains an explicit trusted/unsafe obligation. Capabilities constrain safe code's access but are not an OS sandbox against malicious native dependencies.

## Required interaction foundations

The project's architectural direction makes interaction combinators a required foundation for automatic parallelization. The [parallelism contract](parallelism-foundation.md) defines its scope: a real minimal reference calculus, a typed demand-aware production representation with bulk operations, and validated extraction into conventional execution plans. This replaces the earlier recommendation that interaction analysis could be removed as an optional research pass.

All production automatic-parallel candidates, including GPU candidates, pass through that typed region and extraction path. Ordinary sequential native functions may compile directly; users are not required to execute a primitive graph reducer per scalar operation. The minimal reducer is a separate reference tool; source-to-primitive expansion remains a research question. The typed graph is not the physical representation of every array element.

Preserve demand, affine captures, cleanup, numeric behavior, and typed port connectivity. Bulk operators need reference definitions and rewrite/lowering laws; a decorative graph around opaque operations is insufficient. The exact encoding and extended rules must be frozen and validated before this stage is accepted. No source/compiler-wide proof is claimed.

Compare direct structured lowering with interaction-plus-extraction as an experimental control on identical programs and targets. Use evidence to improve granularity and lowering; removing the required subsystem needs an explicit architectural decision. A dynamic GPU net evaluator remains a backend option needing its own memory/reclamation and performance evidence.

## Later hosted and accelerated profiles

Add records, variants, `Result`, strings, and a small safe foreign boundary before claiming useful general-purpose programming. Structured CPU tasks precede async IO. Async scopes specify every exit edge, cold-future state transitions, drainage order, blocking/suspension effects, cancellation acknowledgment, and allowed nondeterminism before implementation.

GPU support follows CPU admission and physical-memory checks. Choose one target, pin its toolchain, include host/device transfers and device-completion lifetimes, and retain CPU fallback. No device replay after partial visible mutation without a specific replayability contract. Strict floating point is a separate conformance milestone.

Public optimization packages follow at least two useful built-in operations. Initially recipes are compiler-owned data/rules, not arbitrary executable plugins. Third-party kernels enter through an explicit trust boundary; content hashes establish identity, not correctness.

## Authoring experience

Provide stable diagnostics with UTF-8 byte spans from the first parser. Typed interface packets follow the checker. Unsupported syntax gets a clear rejection. A formatter and language server later reuse the parser and interface representation. The tools operate offline and never require a model. External coding tools may use them like a human would.

## Resource policy and performance

Compilation, generated variants, evaluator fuel, and generated-test counts are bounded. Tool limits report incomplete results rather than false proofs of divergence or safety. Hard runtime policies such as `no_alloc` include scheduler/device support and require admission or explicit enforcement; estimates are not bounds.

Performance comparisons freeze equivalent algorithms, arithmetic semantics, data residency, compiler versions, hardware, seeds, and timed boundaries. Publish per-workload results and uncertainty. Measure native baseline quality before claiming automatic-parallelism advantage, and measure authoring success independently of runtime speed.
