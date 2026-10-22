# Implementation plan: ordinary Rust-built language to efficient GPU programs

The 2026-09-28 v0.1 amendment adds [general GPU compilation](general-gpu-v01.md) and [ordinary-source parity acceptance](cuda-parity-v01.md). After M4-06 and before M5-02, GNET-01..05 must establish complete v0.1 source-to-net and compiled-form lowering, a bounded CPU net semantic oracle, program-specific compiled CUDA/Metal dynamic control and worklists, lawful specialization and cost dispatch, then the frozen general/robotics corpus. The earlier recognized-family path remains necessary but no longer sufficient for v0.1. Production requires neither a generic GPU graph allocator nor a net interpreter; the net remains a checked compilation representation. Preserve all older semantic and numeric gates.

Status: implementation contract, written 22 September 2026 and reconciled with `main` for the AI-02 specification. The M0 reference core, typed MIR, interaction reducer, native CPU backend, CPU scheduling and the L1, R1, S1 and M4 increments are on `main`; the open GPU and acceptance work is tracked in [remaining work](remaining-work.md), and GPU performance results remain unclaimed. The standalone executable research probe validates one finite mathematical model. Later acceptance gates must be passed before claiming the full result.

## Product and architectural decisions

Invar source should express the computation, ownership, and numerical meaning. For admitted code, the compiler chooses CPU or GPU execution and generates the kernels, launch geometry, transfers, and synchronization. Users need not write CUDA, select thread indices, or manually tile their application. Sequential code remains usable when an algorithm has no supported parallel transformation.

The project-owned compiler, interpreter, runtime, testing tools, and backend adapters are Rust. CUDA source/PTX produced by a Rust code generator and third-party LLVM/CUDA libraries are allowed target artifacts and dependencies. This is not a proposal to implement a CUDA driver or LLVM in Rust.

Use a direct LLVM native CPU backend and a Rust-generated CUDA backend with a thin Rust driver/NVRTC adapter. Freeze tested versions when their issues land. Preserve a small typed device IR so scheduling is testable before emission. Do not add MLIR, a second GPU backend, a custom assembly compiler, or a generic GPU graph allocator initially. A bounded e-graph search beside the net is allowed instead of a global equality-saturation engine (see [net-parallel-laws-v01](design/net-parallel-laws-v01.md) section 6.4, decisions E1 and M1): the interaction net stays the IR of record and every extracted form is re-certified by the independent checker. The [backend comparison](research/backend-options.md) explains alternatives and experiments that could justify changing this decision.

Interaction combinators are required. Implement a real small reference reducer, a demand-aware typed interaction representation, and nontrivial rewrites with validated extraction into structured operations. Production GPU kernels execute the extracted schedule; they need not allocate a graph node for each scalar operation. Read the [interaction design](research/interaction-design.md) and [existing foundation contract](parallelism-foundation.md) together. A wrapper around an unrelated map optimizer does not satisfy this requirement.

The GPU vertical slice moves ahead of hosted async and the full general-purpose library. It begins with integer maps, reductions, scans, and transformed recurrences, then adds layouts, floating point, stencil and matrix schedules, segmented parallelism, and irregular work. This is an ordering decision, not removal of the rest of the language.

## What “CUDA-class” means here

The desired outcome is comparable performance to competent handwritten CUDA for a frozen, disclosed set of supported algorithm classes on named hardware. This is an engineering acceptance target, not a theorem about every source program or future GPU. Kernel generation alone is insufficient: memory layout, data residency, fusion, scheduling, and specialized instructions determine performance.

[Performance acceptance](performance-acceptance.md) fixes workload coverage, equivalent arithmetic, independent references, timing boundaries, and numerical targets before tuning. All core classes must pass; one fast matrix example cannot stand in for the language. Publish individual regressions, compiler overhead, and end-to-end results. A failed performance gate leaves the milestone incomplete; do not redefine the corpus afterward to remove slow cases.

## The mathematical framework for dependencies

There are two distinct transformations: rescheduling an existing computation and deriving a different equivalent algorithm. Both are needed.

For a region, define iteration instances `I`, their reads `R(i)`, writes `W(i)`, demanded effects and traps, and a dependence relation `D`. An edge `i -> j` records that the original semantics requires `i` before `j`. A schedule may run instances together only if required edges remain respected. Ownership helps prove that buffers cannot alias; it does not prove that two different indices address different physical bytes. Shape, layout, bounds, and access maps supply the latter facts. Unknown calls, accesses, or termination stay unknown.

Initially analyze one-dimensional counted loops, fixed arrays, affine indices, and read-only captures. Prove injective writes and absence of read/write overlap by simple interval and affine forms. Add restricted multidimensional affine relations with the layout milestone. Do not implement a general theorem prover to admit the first map. Polyhedral representations are an established way to express restricted affine dependencies; [MLIR's Affine dialect](https://mlir.llvm.org/docs/Dialects/Affine/) is a useful reference, not an initial dependency.

When a recurrence has genuine edges, look for a compact compositional summary. For independent inputs `x_i` and state transition `F`, require a carrier `S`, lift `h`, associative composition `⊗`, identity `e`, and action `A` satisfying:

```text
h(x_i) : S
A(s, e) = s
A(s, h(x_i)) = F(s, x_i)
A(A(s, a), b) = A(s, a ⊗ b)
(a ⊗ b) ⊗ c = a ⊗ (b ⊗ c)
```

An ordered prefix scan of the summaries yields every state; a reduction suffices when only the final state is observed. This changes the algorithm while preserving its result. Associativity allows regrouping, not permutation. A function-composition closure is not automatically a useful summary: if its size or application cost grows with the prefix, the sequential dependency has merely moved.

The first automatic recurrence transformation is modular affine state:

```text
s_(i+1) = a_i * s_i + b_i   modulo 2^32
h(i) = (a_i, b_i)
(a,b) ⊗ (c,d) = (c*a, c*b+d) modulo 2^32
e = (1,0)
A(s,(a,b)) = a*s+b modulo 2^32
```

The identity and action equations follow by substitution. Both associations of three summaries give `(e*c*a, e*c*b+e*d+f)` for the third pair `(e,f)`; the ring laws of modular arithmetic justify the equality. This short mathematical derivation is separate from testing the Rust implementation. Summaries have constant size and constant-cost composition. A work-efficient ordered tree has linear work and logarithmic span; actual speed still depends on overhead and memory traffic. [Blelloch's prefix-sum report](https://www.cs.cmu.edu/afs/cs.cmu.edu/project/scandal/public/papers/CMU-CS-90-190.html) establishes scan as a reusable parallel primitive.

Recognize the transition structurally after typed normalization, not from names or comments. Coefficients may depend on `i`, immutable inputs, and constants; they may not depend on the evolving state, visible mutation, trapping operations, or an unknown call. Both final-only and per-step-output forms need their own rule. Preserve the seed and exact prefix ordering. The [Rust probe](research/summary-evidence.md) checks small rings exhaustively and full-width generated cases, including an operand-order mutation.

Later bounded summary families include fixed small matrices and finite-state transition tables. Their cost is explicit: composing dense dimension-`d` matrices costs approximately `O(d^3)`, and state tables cost `O(K)` per composition. Do not enumerate a 32-bit state space. A stencil can use spatial independence or a legal time-space wavefront without falsely deleting dependencies. Data-dependent traversals may use frontier/worklist algorithms when a named equivalence rule exists. Unrecognized recurrences keep their sequential implementation with an explanation.

Checked arithmetic and strict floating-point loops do not inherit modular laws. Reassociation can change traps or numerical values. Range analysis may establish that all proposed checked intermediates are representable; otherwise retain source order. Floating-point relaxation must be an explicit source contract, never an optimizer flag silently enabled to win a benchmark.

## Compiler flow and trusted boundaries

```text
source -> parse/check/ownership -> typed control-flow MIR -> native serial LLVM
                  |                        |
                  v                        v
        independent source evaluator   regions + semantic contracts
                                           |
                               typed interaction encoding
                                           |
                            admitted rewrite + bulk extraction
                                           |
                         structured operations + lawful summaries
                                           |
                        schedule/layout candidates + physical verifier
                                     /                 \
                           scoped CPU chunks       device IR -> CUDA
                                     \                 /
                           guarded dispatch + completion + publication
```

The source evaluator never calls optimized lowering as its oracle. The MIR evaluator is a second useful differential boundary, not a replacement for the source reference. Keep common source IDs, immutable types and diagnostics; keep evaluation logic independent.

MIR owns typed values, blocks, terminators, explicit places, checks, calls, moves, and cleanup actions. It records source spans and numeric policy. A region contract identifies inputs/outputs, loans, logical memory effects, demand, failure behavior, termination evidence, and observable order. Analysis results attach to a region revision; mutation invalidates dependent facts. The first implementation uses concrete enums and IDs in existing modules, not a plugin hierarchy or repository full of empty crates.

The interaction verifier checks reciprocal ports, node arity, types, captures, activation, and permissions. An active pair is not permission to evaluate an unselected branch. Rules state a source fragment and an input/output relation. Extraction produces explicit `Map`, `OrderedReduce`, `OrderedScan`, and later layout/segmented/tiled operations, retaining their proof obligations. At least one topology-changing rewrite must change extracted IR. A connectivity mutation must alter output or be rejected.

Admission returns an opaque `AdmittedPlan`, constructible only by the admission module. Its obligations include type/shape consistency, preserved demand and observations, totality for initial parallel regions, lawful algebra, physical footprints, and valid target resources. Evidence kinds are structural check, trusted named compiler rule, runtime guard, checked proof artifact if one exists, or unknown. A test pass is never an evidence kind establishing a law.

A candidate's identity includes region content, rule revisions, semantic version, numerical policy, layout, backend/toolchain, device capability, and relevant options. Cache loading revalidates these fields and bounded payloads; stale or malformed entries are cache misses. Unknown schema/evidence versions reject. Runtime guards are total and nonobservable, and protected inputs stay unchanged from guard evaluation through execution. Guard false selects the retained lawful fallback. Internal verifier failure is a compiler defect, not an optimization miss to hide.

Persistent cache bytes are untrusted. Initially persist only schedule hints and descriptive manifests; reconstruct the current region, re-admit the schedule, regenerate CUDA from verified device IR and recompile with pinned NVRTC on a cold process. Never load disk/imported PTX, cubin or executable CUDA text merely because the hash that accompanies it agrees. Compiled-module reuse is in-process only and bound to the live context/current plan. Artifact hashes establish identity, not authenticity or legality. Preserve generated binaries as inspection evidence without treating them as trusted cache input. Authenticated persistent binary caching needs a later explicit contract.

## Device scheduling and runtime

Start with flat, fresh-output total regions. One stream and synchronous public completion are sufficient; use internal events to retain allocations until work actually completes. A Rust boundary owns the CUDA context, modules, streams, events, and buffers. Host-visible output is published only after success. Driver/NVRTC errors are environment errors with bounded diagnostics. A timeout or cancellation request does not free borrowed device storage.

The runtime states are `Prepared`, `Submitted`, `Completed`, `FailedQuiescent`, and `InFlightUnknown`. Bounded polling may return an owning pending token; dropping it transfers all resources to a bounded quarantine. Retain context, stream/events, device memory and owned pinned staging, prohibit replay/reuse, and poison new submissions on that context until quiescence is confirmed. The first transfer path must own staging rather than retain a caller buffer past its Rust lifetime. Reaching the quarantine budget disables further GPU dispatch. Only completion or a documented successful teardown that guarantees quiescence permits resource release; a hung device or cancellation request does not. The synchronous CLI reports the pending/environment failure. Test timeout followed by token drop and later completion, including exactly-once release and no premature CPU replay.

Native and reference execution must work without installed CUDA. GPU-specific tests are explicit and require a preflight; missing hardware is not a passing GPU test. Initialize the backend lazily. Use checked byte-size/index conversions, zero-length handling, bounds masks, typed scalar ABI, and context/device ownership. Default to one supported Linux NVIDIA configuration, with minimum capability selected from actual authorized hardware rather than guessed now.

First generated schedules:

| Operation | Initial schedule | Critical obligation |
| --- | --- | --- |
| Fused map | Grid-stride coalesced loads/stores; each lane owns outputs | No overlap, masks, total callback, no intermediate array materialization |
| Reduction | Per-block ordered tree, temporary block totals, separate merge launches | Identity and associativity; order preserved for noncommutative operations |
| Scan/recurrence | Block scan, ordered scan of block summaries, prefix propagation | Inclusive/exclusive conventions, seed, ragged blocks, operand orientation |
| Transpose | Shared-memory tile with padding and boundary masks | Unique physical writes and uniform barriers |
| Stencil | Interior tiles plus halo and explicit boundary policy | Reads come from correct time slice; no in-place cross-block race |
| Matrix multiply | Recognized contraction, shared/register tiling, ordered accumulation or declared relaxed contract | Layout, accumulation/rounding policy, edge masks, target instruction support |
| Segmented operations | Validated offsets, segment-aware scans/reductions and flattening | Empty segments, stable order, bounded temporary size |
| Gather/scatter | Gather or proved-injective writes; reduction conflicts only under a named law | Bounds, aliasing, deterministic conflict semantics |
| Irregular work | Bounded frontier rounds with compaction | Termination/resource outcome, ownership, ordering where observable |

Do not use a global spin barrier across arbitrary blocks. Inter-block phases use separate launches/events initially. All lanes in a block reach required barriers; boundary lanes mask memory operations rather than skip a barrier. Model physical accesses after vectorization, including padding and widened loads. Only use shuffles with a correct active-lane mask and defined participants.

Fuse elementwise producers/consumers before flattening when lawful. Retain bulk shapes to avoid a cubic materialized intermediate for matrix multiplication. Choose scalar execution inside a lane when exposing inner parallelism costs more than it saves. Futhark's [flattening discussion](https://futhark-lang.org/blog/2026-07-31-full-flattening.html) illustrates that segmented representation and fusion both matter; flattening alone does not guarantee speed.

Autotuning enumerates a small finite set of already-admitted schedules: block sizes, vector widths, tile sizes, fusion cutoffs, and CPU/GPU thresholds. Default bounds are at most 32 variants per region and no runtime tuning without an explicit budget. Compilation, launches, temporary memory, and tuning time all have limits. Never tune on production-visible output or change numeric policy. Cache tuned choices by hardware and semantic identity. Start with simple analytic estimates and benchmarked thresholds; no learned model service or unbounded search.

## Numerical and language growth

M0 retains its frozen integer/array syntax and semantics. M4 adds a separate versioned numeric/layout specification before frontend changes: `f32`, explicit fused operations, rounding/subnormals/signed zero/NaN behavior, tensor shapes and layouts, and explicit matrix/reduction policies. Strict ordered scalar floating-point remains the default. Tensor Core operations require an explicit operand/accumulator/rounding contract supported by the device; arbitrary approximate arithmetic is not the meaning of “fast.” Compare against CUDA implementing that same contract.

The language track adds modules, records, variants, pattern ownership, `Result`, strings/bytes, precise borrowed views, bounded monomorphization and affine closures, then a narrow C ABI and practical standard library/tooling. Each extension updates grammar, interpreter, checker, MIR/native semantics, diagnostics, and capability status before optimization support. Unsupported GPU constructs compile to CPU or receive an explicit forced-target rejection.

Hosted structured tasks, bounded channels, cancellation, capabilities and async IO follow explicit trace/lifetime contracts. A task retaining a loan must finish or acknowledge cancellation before storage is released. Do not infer a source concurrency memory model from Rust's implementation. Public recipe packages, a generic dynamic GPU net runtime, multiple GPU vendors, and full formal verification are separately gated future work rather than silent requirements of the first release.

## Early falsification and open implementation details

The M1-04 design revision requires demand-preserving encoding into the typed graph, with checked links and evidence for both demanded divergence and undemanded branches. Source-to-three-agent encoding is deferred research. M1-05 must show real typed rewrites and extraction; an opaque identity wrapper still fails acceptance. M2 owns automatic source-loop recognition.

The second risk is recognition coverage: rename variables, change lawful source structure and vary shapes to establish that compiler rules recognize programs rather than examples. The third is device scheduling quality: compare generated code with a same-schedule CUDA control before blaming NVRTC or adding another IR stack. The final independent CUDA baselines test algorithm/schedule quality, so passing the same-schedule control does not establish CUDA-class performance.

Exact source syntax for later numeric operations, ABI/toolchain pins and later typed bulk templates are deliverables of their named issues, not facts already validated by this plan. A failed experiment may require a reviewed design correction; the issue graph does not guarantee that every performance hypothesis will pass.

## Delivery sequence and ownership

The [issue catalog](issues/README.md) supplies exact dependencies and executable acceptance contracts. M0 remains nine existing issues. Subsequent phases are:

1. **M1:** typed MIR, native LLVM, real interaction reducer, typed encoding/verifier, nontrivial rewrites/extraction.
2. **M2:** dependence/effect analysis, ordinary-loop recognition, lawful reduction/scan, automatic affine recurrence transformation, CPU execution/fallback.
3. **M3:** Rust CUDA lifetime boundary, device verifier/toolchain, fused maps, hierarchical scans/reductions, integrated dispatch/cache/failure behavior.
4. **M4:** numerical/layout semantics; transpose/stencil; generated matrix/Tensor Core schedules; segmented/irregular work; deterministic scatter; cost model/tuning.
5. **M5:** independent benchmark harness, adversarial full-pipeline validation, and performance acceptance on a pinned target.
6. **L1/L2:** complete the practical general-purpose language and hosted concurrency with the same semantic discipline. These can proceed after their listed prerequisites; they do not gate the first GPU integer result.

Issue dependencies express integration order. Independent slices may proceed in parallel once their prerequisites are met. A completed issue has code, exact tests, semantic documentation, and reproducible evidence. A backend issue requiring hardware remains incomplete while only mocks pass.

## Validation strategy and failure criteria

Every new representation has a verifier and at least one adversarial mutation test. Every transformation has a hand-derived positive case, a near-match that must not transform, and differential execution through the independent source reference. Bounded generators vary source structure, identifiers, constants, shapes, aliasing, control flow, and numeric boundaries; do not match benchmark filenames. Use fixed seeds and minimize failures.

Compare source, MIR, native serial, extracted serial, forced CPU parallel at several worker counts, and forced GPU where available. Values alone are insufficient where semantics includes traps, demanded divergence, cleanup, or observable order. Initial total GPU regions deliberately avoid those extra observables; rejecting a region must preserve a functional CPU path. Compiler limits are tool errors, never false proofs of source divergence.

The existing Cargo Debug/Release/fmt/Clippy gates apply throughout. Later boundary crates justify and test each unsafe invariant; run Miri on pure Rust and device sanitizer checks on authorized CUDA hardware where supported. Pin toolchains and record coverage gaps rather than marking a skipped test green. Normal CI is bounded and CPU-only; GPU qualification is a distinct command/report.

Stop and revise a rule if any accepted candidate disagrees, if mutation tests cannot detect an incorrect connection, if an ordinary loop needs benchmark-specific annotation to transform, if generated code relies on undefined barriers/aliasing, or if tuning changes meaning. A slow but correct backend is a useful intermediate result, not completed CUDA-class acceptance.

## Shared executable acceptance interface

M0/M1 commands run against the current Cargo project; later commands are
implementation requirements. Each issue adds a named integration test
beginning with its lower-case ID, such as `issue_m1_04_`. Acceptance lists a
filtered suite, verifies a nonzero relevant selection, then runs it in Debug
and Release. Zero selected tests, ignored-only tests, and mock-only hardware
evidence cannot satisfy an issue. Full fmt/Clippy/Cargo gates still run at
milestone boundaries.

M1-02 adds `invar build INPUT --target native --output PATH` and `invar run INPUT --engine native-serial`. M2-01 adds `invar explain INPUT --format json`, with per-region status, first rejected obligation, rule identities and chosen/fallback plan. M2-05 adds `--engine cpu-parallel` and a bounded `--workers N`. M3-05 adds `--engine cuda` and `--engine auto`. Forced CUDA rejects unavailable hardware or ineligible code explicitly; auto may fall back before publication. The CLI must never quietly report forced GPU execution when it ran on CPU. `--engine reference` stays independent, and M0 exit/status meanings remain compatible with documented schema extensions.

M3-01 adds an opt-in Cargo feature `cuda` with no driver/toolkit dependency in the default build. Pure runtime-state/verifier tests run with default Cargo. Hardware suites use `cargo test --locked --features cuda issue_m3_01_gpu -- --ignored --nocapture` (corresponding issue prefix for later suites) after host preflight. GPU tests may be ignored in ordinary CI but their explicit qualification command must fail with an actionable prerequisite error on a missing device, not return success. Record nonzero test count and real device identity. Toolchain-only compilation can run separately where NVRTC exists; it does not establish runtime correctness.

M5-01 supplies a Rust benchmark interface, freezes its invocation/report schema and validates it against the protocol above. The issue catalog names exact added commands. No undocumented shell script or notebook is the sole way to reproduce acceptance.

