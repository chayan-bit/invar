# Invar architecture, design revision 0.4

Status: implementation direction, not a proved system. The implemented language is specified in [the v0.1 specification](spec/invar-v0.1.md) and the [grammar](../spec/grammar.ebnf). Planned work is tracked in GitHub issues.


This page has two parts.
The first part is a code tour: it follows one small program through the compiler and names the file and function at each step.
The second part, from "One meaning, multiple admitted implementations" onward, is the design direction and the trust model.
For unfamiliar words see the [glossary](glossary.md); for the other documents see the [documentation index](README.md).

## Code tour: one program end to end

The example is [examples/generated-map.inv](../examples/generated-map.inv), a zero-based counted loop that adds 5 to each element of an array.

```text
fn main() -> i32 {
    let xs: Array<u32> = [1u32, 2u32, 3u32];
    var ys: Array<u32> = filled_u32(len(xs), 0u32);
    for i in 0u32 .. len(xs) {
        ys[i] = wrapping_add(xs[i], 5u32);
    }
    if ys[2u32] == 8u32 {
        return 0;
    }
    return 1;
}
```

Each step below says what it consumes, what it produces and which command shows it.
The commands run from the repository root after `cargo build --locked`.

### 0. The command line

`src/main.rs` parses the arguments and dispatches: `check` and `explain` stop after analysis, and `run` picks an engine.
`execute` loads the source with `load_checked`, which runs steps 1 and 2, and then calls the reference evaluator, `execute_native` or `execute_scheduled`.
`src/cli.rs` and the `*_command.rs` files hold the larger subcommands.

### 1. Front end: text to AST

`front::parse` in [src/front/parser.rs](../src/front/parser.rs) takes the source bytes and returns a `Program`, or a `FrontError` with a byte span.
The lexer is [src/front/lexer.rs](../src/front/lexer.rs), and the input limits (1 MiB of source, nesting depth 32) are constants in [src/front.rs](../src/front.rs).
See [src/front/README.md](../src/front/README.md).

For the example this produces one function, `main`, with a `for` statement over `0u32 .. len(xs)` and an indexed store.

### 2. Checking: names, types, ownership

`check::check(program, require_main)` in [src/check.rs](../src/check.rs) resolves names, checks types and returns a `CheckedProgram`.
Its last steps call `ownership::check` ([src/ownership.rs](../src/ownership.rs)) for affine moves and call loans, and `views::check` for borrowed-view lifetimes.
`invar check examples/generated-map.inv` runs steps 1 and 2 and stops.

Here every operand of `wrapping_add` is `u32`, `xs` is a read-only owner, and `ys` is a separate fresh array, so no move or loan conflicts arise.
See [src/check/README.md](../src/check/README.md).

### 3. Reference evaluation: the meaning of the program

`eval::run` and `eval::observe` in [src/eval.rs](../src/eval.rs) interpret the `CheckedProgram` directly, charging fuel per step, trapping in source order and recording cleanup.
This evaluator shares no code with the optimizing paths, so it is the first oracle.
`invar run examples/generated-map.inv --engine reference` stops here and prints `Result: 0`, because `ys[2]` is 8.
See [src/eval/README.md](../src/eval/README.md).

### 4. MIR: a verified control-flow graph

`mir::lower(&checked)` in [src/mir.rs](../src/mir.rs) turns the checked program into typed basic blocks, and `mir::verify` ([src/mir/verify.rs](../src/mir/verify.rs)) checks dominance, types and terminators.
The `for` loop becomes a loop-head block with a `u32` induction parameter and a `Branch` on `induction < len(xs)`, plus a body block that reads, calls `wrapping_add` and stores.
`mir_eval::observe` ([src/mir_eval.rs](../src/mir_eval.rs)) is the second oracle: an independent block interpreter that must match the reference on value, trap, steps and cleanup.
See [src/mir/README.md](../src/mir/README.md), [src/mir_eval/README.md](../src/mir_eval/README.md) and [typed MIR v1](mir-v1.md).

### 5. Admission: may this loop run in parallel?

`schedule::regions` ([src/schedule/lowering.rs](../src/schedule/lowering.rs)) calls `admission::analyze(source, &program)` ([src/admission.rs](../src/admission.rs)).
`analyze` finds each counted `for` loop head in the MIR and builds a `Region` with a list of obligations; `analyze_region` is the matcher for the wrapping map shape.
Each obligation is `RANGE`, `TERMINATION`, `TOTALITY`, `OWNERSHIP`, `ALIAS`, `FRESH_OUTPUT`, `DEPENDENCE`, `EFFECT_ORDER`, `CLEANUP` or `PHYSICAL_BOUNDS`, with one of the evidence kinds from the [glossary](glossary.md#e).
A failure of any check rejects the region with the name of the first obligation that failed, such as `DEPENDENCE`, and the loop stays serial.

Then `plan::decide` ([src/plan.rs](../src/plan.rs)) classifies each region into an `Execution`.
For a map it calls `loops::single` ([src/loops.rs](../src/loops.rs)), which builds a typed interaction net, verifies it (`Net::verify` gives a `Proof`) and extracts a plan (`Net::extract`) in [src/typed_net/](../src/typed_net/README.md).
That round trip is the typed extraction that every parallel candidate must pass; if it fails, the region is rejected with `EXTRACTION`.
Adjacent maps may be fused here (`MAP_FUSION_RULE`).

The example's region is admitted with rule revision `counted-wrapping-map-v1`.

### 6. The independent check: GNET and the normal form

`schedule::families::dispatch` ([src/schedule/families.rs](../src/schedule/families.rs)) first calls `lowering::check`, which encodes the whole MIR as a general net (`Net::encode`), verifies it, extracts the law-directed normal form (`normal_form::extract`) and checks family admission against it.
No normal form means no regions: the families only see loops that this second, separately written analysis also accepts.
Then each admitted region is offered to the family registry in precedence order, and the first match wins.
The example's loop is taken by the `core` family ([src/schedule/families/core.rs](../src/schedule/families/core.rs)).
See [src/general_net/README.md](../src/general_net/README.md), [net normal form](net-normal-form-v01.md) and [PNF family admission](pnf-family-admission-v01.md).

### 7. Scheduling and execution

`invar run examples/generated-map.inv --engine cpu-parallel --workers 2` reaches `engine::run_mir` ([src/engine.rs](../src/engine.rs)), which calls `schedule::run_source_with_hooks` ([src/schedule/mod.rs](../src/schedule/mod.rs)).
The scheduler runs the MIR serially.
When it reaches the admitted loop and the runtime guard is true (the upper bound fits every array length and the byte budget), it splits the iteration range across workers (`parallel_chunks` in [src/schedule/workers.rs](../src/schedule/workers.rs)), stages the outputs and publishes them atomically.
If the guard is false, the serial MIR plan runs before any write, and the result is the same.
The run receipt lists a `choices` entry with backend `cpu-parallel` for this loop.
See [src/schedule/README.md](../src/schedule/README.md).

### 8. Other backends for the same admitted region

- **Device kernels**: `device_ir::from_region` ([src/device_ir.rs](../src/device_ir.rs)) lowers the region to the shared device IR, `invar_device::verify` ([crates/invar-device/](../crates/invar-device/Cargo.toml)) checks it, and only then is CUDA source (compiled with NVRTC by `invar-nvrtc`, driven from [src/schedule/cuda.rs](../src/schedule/cuda.rs) and [crates/invar-cuda/](../crates/invar-cuda/Cargo.toml)) or Metal source ([src/schedule/metal.rs](../src/schedule/metal.rs), [src/metal_aot/](../src/metal_aot/README.md)) emitted.
  Both need matching hardware and are not run in CI.
- **Native code**: with the `native` feature, `native::emit_mir` ([src/native.rs](../src/native.rs)) compiles the whole verified MIR through LLVM for `--engine native-serial`, which is serial and checks arithmetic exactly like the reference.
  See [src/native/README.md](../src/native/README.md).
- **Placement**: `--engine auto` chooses among these from measured costs; see [src/placement.rs](../src/placement.rs) and [placement profile v1](../spec/placement-v1.md).

### 9. The receipt

`invar explain examples/generated-map.inv --format json` runs steps 1 to 6 and prints one `invar-explain-v5` document built by `Report::to_json` in [src/admission.rs](../src/admission.rs).
For the example's region the document says:

| Field | Value |
| --- | --- |
| `function`, `status` | `main`, `admitted` |
| `first_rejected_obligation` | `null` |
| `rule_revision` | `counted-wrapping-map-v1` |
| `numeric_policy`, `target` | `wrapping-u32-v1`, `cpu` |
| `chosen_plan` | `wrapping-map-candidate-if-guard` |
| `fallback_plan` | `mir-serial` |
| `runtime_guard` | upper bound `length` of `xs`, inputs `xs`, output `ys` |
| `obligations` | the ten above: `RANGE`, `TERMINATION`, `OWNERSHIP`, `ALIAS`, `FRESH_OUTPUT`, `DEPENDENCE` and `EFFECT_ORDER` are `checked_invariant`; `TOTALITY` and `CLEANUP` are `trusted_rule`; `PHYSICAL_BOUNDS` is `runtime_guard` |

The test `tests/spec_examples.rs` runs this example and checks that its region is an admitted `map`.
Field meanings and the full schema are in [explain JSON](explain.md).
To see a rejection, change the store to `ys[0u32] = ...`.
The store index is no longer the induction variable, so `analyze_region` returns its `FOOTPRINT` rejection, `first_rejected_obligation` names it instead of `null`, and the loop runs as serial MIR.

## Source map

The compiler is one package, `invar`, whose library is `src/lib.rs` and whose CLI is `src/main.rs`.
Directories listed in the table have their own README.

| Stage | Modules |
| --- | --- |
| Front end | [front](../src/front/README.md), `diagnostics`, `fixes` (validated source fixes for `check --fix`) |
| Checking | [check](../src/check/README.md) (types, units, modules, generics, closures), `ownership`, `numeric_contract`, `math` (the cr-v1 library source) |
| Reference execution | [eval](../src/eval/README.md), `fsum_reference` |
| MIR | [mir](../src/mir/README.md), [mir_eval](../src/mir_eval/README.md), `mir_net` |
| Typed nets and laws | [typed_net](../src/typed_net/README.md), `term`, `laws`, `laws_synth`, `recurrence`, `counted`, [loops](../src/loops/README.md) |
| Admission and planning | `admission`, `plan`, `budget`, `array_limits`, `placement`, `device_cache`, `tensor_cache` |
| Source families (recognizers and bounded executors) | [compaction_source](../src/compaction_source/README.md), [histogram_source](../src/histogram_source/README.md), [indexed_source](../src/indexed_source/README.md), [scatter_proof](../src/scatter_proof/README.md), [stencil_source](../src/stencil_source/README.md), [summary_source](../src/summary_source/README.md), [tensor](../src/tensor/README.md), [transpose_source](../src/transpose_source/README.md), `segmented_source`, `contraction_source`, `frontier_source`, `pair_reduction`, `bounded_summary`, `f32_source`, `fsum`, `sorting`, `spatial` |
| Scheduling | [schedule](../src/schedule/README.md), `engine` |
| General nets | [general_net](../src/general_net/README.md) |
| Device plans | `device_ir`, `indexed_device`, `segmented_device`, `layout_contract`, [metal_aot](../src/metal_aot/README.md) |
| Native code | [native](../src/native/README.md), `exports` ([README](../src/exports/README.md)) |
| Robots and systems | [system](../src/system/README.md), [robotics](../src/robotics/README.md), [component_test](../src/component_test/README.md) |
| Helper binaries | [src/bin](../src/bin/README.md) |

Other packages in the workspace:

| Path | What it is |
| --- | --- |
| [crates/invar-device](../crates/invar-device/Cargo.toml) | Shared device plans, limits and CUDA/Metal source emitters |
| [crates/invar-cuda](../crates/invar-cuda/Cargo.toml), [crates/invar-metal](../crates/invar-metal/Cargo.toml) | Reviewed driver boundaries; hardware completion controls publication |
| [crates/invar-foreign](../crates/invar-foreign/Cargo.toml) | The C ABI boundary, buffers and dynamic libraries |
| [crates/invar-math](../crates/invar-math/Cargo.toml) | Vendored CORE-MATH sources behind the cr-v1 library |
| [crates/invar-python](../crates/invar-python/README.md) | The separate Python package |
| [tools](../tools/README.md) | Benchmark, held-out corpus and qualification binaries (`invar-tools`) |
| `tests/`, [test-support](../test-support/README.md), `bench/` | Integration tests, shared helpers and the benchmark corpus |

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

The project's architectural direction makes interaction combinators a required foundation for automatic parallelization. The [parallelism contract](parallelism-foundation.md) defines its scope: a typed demand-aware production representation with bulk operations, and validated extraction into conventional execution plans. This replaces the earlier recommendation that interaction analysis could be removed as an optional research pass.

All production automatic-parallel candidates, including GPU candidates, pass through that typed region and extraction path. Ordinary sequential native functions may compile directly; users are not required to execute a primitive graph reducer per scalar operation. Source-to-primitive expansion remains a research question. The typed graph is not the physical representation of every array element.

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
