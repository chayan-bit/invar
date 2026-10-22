# General interaction-net compiled GPU execution: v0.1 scope addendum

Date: 2026-09-28. Versioned v0.1 design and acceptance contract; GNET-01 must freeze its rule details.

Performance amendment: the subsequent handwritten-CUDA requirement is specified in [cuda-parity-v01.md](cuda-parity-v01.md). It adds compiled-region optimization, robotics benchmark categories and stricter per-case performance gates; it does not replace the generality and semantic obligations here.

## Authority and status

v0.1 requires general automatic GPU acceleration through interaction nets and cost models, beyond recognized program families. This expands the old M4/M5 scope. The design below is a proposed implementation contract with proof obligations, not implemented or accepted functionality. The implementation plan incorporates this contract. Existing correctness, numeric, performance, hardware and release gates remain.

Generality means compositional lowering of every computational form in the declared v0.1 language into checked compiled execution, including dynamically discovered work. It does not mean every computation has independent work, every legal schedule is profitable, or an optimal algorithm can always be discovered. Missing a map/reduce/stencil pattern must not prevent a source-derived compiled device path for eligible pure work. Physical finite-memory and tool limits remain explicit. Compile-time declarations are checked/elaborated rather than executed as GPU work; actual foreign IO/device operations are explicit ordered host boundaries.

Current implementation gap: `interaction-reducer-v1.md` supplies untyped local combinator rules without source encoding. `typed-interaction-v1.md` supplies a finite acyclic graph and source-order demand rules; `mir_net.rs` retains a MIR control skeleton. Neither is a complete typed executable-net compilation representation or general compiled CUDA/Metal lowering. `plan.rs` currently preserves unrecognized work as serial MIR. Additional templates alone do not satisfy this addendum.

## Execution architecture

Ordinary source -> checked typed/ownership IR -> versioned demand/effect-aware executable-net IR -> verified rewrites and dependence analysis -> program-specific compiled native CPU executable and CUDA/Metal kernels with compiled orchestration -> guarded cost-based placement and ordered observation commit.

The net is a checked compilation representation with executable semantics and a bounded CPU reference evaluator for differential tests. Production need not interpret net nodes or run a universal GPU graph reducer. Runtime predicates become emitted branches, variable loops become emitted loops, recursion becomes native calls or checked frames/continuations, and irregular frontiers become compiled bounded worklists. Generated dispatch may evaluate workload size, guard results and device readiness at runtime; these input-dependent facts are not assumed known at compile time. Bounded allocation, queues, launches and fences are allowed where the compiled algorithm needs them, and their measured costs remain in the acceptance report.

Prefer ahead-of-time program-specific host and device artifacts for declared targets, with selected guarded variants precompiled. Report module loading, pipeline creation and any driver/PTX compilation separately; a PTX artifact that the driver compiles is not described as having zero runtime compilation. No runtime language compiler, optimizer or model is a production prerequisite. An optional later JIT cannot be required for the default v0.1 route.

Each core expression/control/data form needs explicit net semantics and a compositional compiled lowering, including direct and indirect calls where supported, recursive frames, loop/backedge activation, branches, early returns, closures, records/variants, owners, arrays/views and supported numeric operations. Reuse checked frontend/MIR facts and existing primitive kernels. Do not hide an arbitrary MIR interpreter inside an opaque device node or count whole-region CPU fallback as general GPU coverage. Freeze a coverage table over the actual v0.1 grammar and MIR variants; a missing entry is a release gap, not a new language restriction. Host-bound FFI must be identified precisely, not used as an escape hatch for missing pure-core rules. Physical numeric limitations require exact agreed implementations or explicit coverage gaps, never silent narrowing.

Use typed primitive/array agents where appropriate; their refinement and wire interface must be justified. Do not claim that Lafont's three-agent universality automatically proves an efficient or semantics-preserving encoding of Invar. Preserve frozen v1 artifacts; introduce a separately versioned net contract rather than silently changing them. All Invar-owned compiler/runtime logic remains Rust; existing narrowly reviewed platform bindings and emitted device source remain the backend boundaries.

## Semantic obligations

Let source configuration C = (control, store, ownership, ordered observation prefix), and executable-net state N include graph, demanded frontier, leases and ordered commit state. Define a relation R(C,N). For each logical net transition N -> N', establish either an invisible administrative step preserving R, or source steps C ->* C' with R(C',N'). Establish an equivalent simulation for compiled CPU and device steps against N. Progress under the specified fair schedule and divergence-sensitive behavior is required; forward simulation alone can hide starvation.

Observable agreement includes normal values, first source-ordered trap and source span, visible write/foreign-effect prefix, normal ownership cleanup, and the contract's logical Demand/Complete events. GPU physical work is not itself a logical Demand event. Do not equate evaluator fuel with net rewrite count. Precisely distinguish semantic limits from backend exhaustion or user cancellation; exhaustion is never proof of divergence.

Demand activates only the selected branch. Ownership/region tokens forbid duplication of affine owners and order conflicting memory operations. A device task holds input leases through the fence; FFI consumes the appropriate ordered host token. Unknown overlap serializes or uses a checked runtime guard; a failed guard cannot publish a speculative write. No unchecked foreign purity annotation authorizes speculation.

### Expose independent work without termination proofs

For `let a = f(x); let b = g(y); a + b`, independent observationally pure calls may execute concurrently even if their termination is unknown. The proposed rule is bounded speculation with source-ordered commit:

1. Run later work in finite preemptible compiled work quanta on stable inputs and isolated private scratch. Hold its result or trap.
2. Always give the oldest source frontier progress. If f returns, g may become the frontier; if f traps, discard g; if f diverges, never publish g.
3. No speculative writes, foreign effects, cleanup, or logical demand events become visible. No affine ownership is copied. Uncertain shared resources serialize.
4. Speculative scratch cannot deplete the resources reserved for the serial path. Scratch exhaustion suspends/discards speculative work rather than failing the source computation. Record semantic fuel/resource accounting in source order; any replay for accounting must itself be effect-free and explicitly specified.
5. Kernels have finite quanta so a speculative divergent call cannot prevent the oldest task progressing or cancellation. Reclamation waits for device completion.

This is a proof obligation, not a theorem established by this document. Counterexample to catch: f loops forever and g immediately traps or fills speculative memory; reporting g's trap/OOM as the source result is incorrect. Refusing all possibly-diverging pure calls solely for lack of a termination proof would miss the intended generality. Start from a sound sequential CPU-net reference and compiled CPU lowering, then prove and test this scheduling rule.

## Compiled dynamic work

Emit program-specific branch, loop, frame/continuation and worklist code for dynamic source behavior. A bounded worklist may use epochs when its algorithm needs global synchronization: discover candidates, check ownership/dependence and capacity, execute nonconflicting work, publish source-ordered observations, then retire state after all readers and device fences complete. It is not a generic net-opcode interpreter. Shared-edge writes and stale generations still require checked guards; allocation failure must leave a valid continuation and no partially published effects. Deferral cannot starve an older source frontier.

Completion requires resolved root/commit frontier, empty ready/deferred/host queues where used, no in-flight work and completed streams. An empty local queue alone is insufficient. CPU reference and compiled CPU/GPU paths share the net's logical rule semantics; their physical scheduling and storage strategies may differ. If a guard fails before execution, select the compiled alternative. Switching after progress requires transferring the exact live continuation, ownership and effect frontier, with no replay of visible effects.

## Mathematics for scheduling and profitability

For a realized finite dependency DAG, W is work, S is longest dependency-chain work and P is ideal equal-speed processors. In that ideal unit-cost model, T_P >= max(W/P,S), so speedup <= min(P,W/S). This constrains scheduling of that computation, not every possible equivalent algorithm. Lawful algorithmic transformations may change W and S and need separate equivalence proofs.

A starting empirical GPU estimator is:

    T_hat = T_dispatch + T_transfer + K * L
            + max(W / (P_eff * r), S / r, B / BW)
            + T_worklist + T_alloc + T_commit

K = launches/epochs, L = launch latency, r = calibrated per-lane compiled work rate, B = data/worklist bytes moved, BW = effective bandwidth. It is a fitted heuristic, not a timing theorem; avoid double counting when calibrating interacting terms. Include representation size, frontier skew, transfer/residency, cancellation and cost of moving live state. Unknown total recursion work can be estimated online from completed quanta and frontier statistics; no termination decision is needed. Record uncertainty and use a margin/hysteresis to avoid thrashing.

Choose argmin of estimated time among already semantically admissible schedules. Cost errors may hurt speed but must never legalize an invalid transformation, discard observations or change numeric semantics. GPU coverage testing needs an explicit forced compiled-device mode distinct from automatic profitable dispatch, so a CPU choice cannot conceal a missing GPU path.

For verified resource certificates, a potential Phi with Phi(N) >= c(rule) + Phi(N') telescopes to a sequential-work upper bound. Size/schedule annotations can support such certificates where available. Recursive programs without a provable finite potential still execute under runtime bounds; absence of a certificate does not mean the program is rejected. Parallel span and memory certificates need separate obligations; a work potential alone is not a GPU speed guarantee.

## Release acceptance and implementation slices

GNET-01: main freezes the versioned source-to-net and net-to-compiled boundaries, rule interfaces, observations and complete coverage table. Write the simulation/progress obligations and concrete counterexamples. Preserve existing source semantics and old formats.

GNET-02: main implements a bounded CPU net semantic oracle and full ordinary-source lowering, including loops and recursion, then validates compiled native CPU lowering against that oracle and the existing reference engine. Include randomized well-typed compositions and exact deterministic source cases. Test untaken traps, ordered traps/effects, borrowed mutation, cleanup, recursion and exhaustion; timed divergence tests establish bounded-prefix behavior, not a proof of nontermination.

GNET-03: main implements program-specific compiled CUDA/Metal general control and data paths, including bounded worklists for dynamic frontiers as needed. Validate contested endpoints where parallel worklists share state, stale generations, allocation failure, overflow, cancellation, host fences and quiescence. Ordinary source must actually execute nontrivial dynamic frontiers on both devices; emitted code alone is insufficient. Hardware gaps stay pending while independent implementation continues.

GNET-04: main integrates legal specialization, region layout/residency and calibrated compiled dispatch, including bounded pure speculation with ordered commit. Keep existing M4 family gates. Measure generated worklist/allocation/transfer overhead and serial controls; explain every dispatch/serialization choice using structured evidence.

GNET-05: benchmark owner prepares source-derived acceptance corpus and provenance in its exclusive paths; main wires the actual backend when ready. Use multiple held-out non-template compositions: irregular recursive divide-and-conquer, input-dependent branch trees and dynamically discovered graph frontiers. Inputs/depth/shape must vary at runtime. Include negative/serial cases, not just balanced profitable trees. Compiler must not match fixture names, source hashes or benchmark IDs. Handwritten net/parameterized-plan timing does not establish ordinary-source execution.

Keep three metrics separate: (1) complete source-form net and compiled-lowering coverage with declared host boundaries; (2) measured device-executed work/frontier width, including forced compiled-device mode; (3) transfer-inclusive performance against frozen semantically matched baselines, plus separately labeled warm/resident timing. Whole-region CPU fallback does not count as GPU coverage. Freeze workload choices and statistical/performance thresholds before timing, preserving current M5 methodology and gates; require reproducible gains on specified irregular families on both backends, report regressions and automatic CPU choices honestly. Passing only a handpicked family does not prove full lowering coverage; full coverage does not prove speedup.

These dependencies must be inserted before final M5 qualification/AI-02/release acceptance and update the roadmap denominator. GNET identifiers here are local slice labels, not existing GitHub issues. Do not fabricate a percent complete for the expanded scope.

## Research lineage and limits

- Lafont, Interaction Combinators (1997), https://dcreager.net/papers/Lafont1997/ : universal local rewriting substrate; not Invar source simulation or a performance theorem.
- HVM2, https://github.com/HigherOrderCO/HVM2 and https://github.com/HigherOrderCO/HVM2/blob/main/paper/HVM2.pdf : practical CPU/CUDA net execution. Paper section 13 documents eager-demand/divergence pitfalls and graph allocation overhead. Reuse insights, not an unreviewed foreign runtime or changed source semantics.
- Gimenez and Moser, The Complexity of Interaction, https://arxiv.org/abs/1511.01838 : sized/scheduled types and compositional resource analysis; no claim of automatic complete inference or calibrated GPU timing.
- Jiresch, Towards a GPU-based implementation of interaction nets, https://arxiv.org/abs/1404.0076 : implementation precedent and communication overhead; GPU-net feasibility alone is not speed acceptance.

