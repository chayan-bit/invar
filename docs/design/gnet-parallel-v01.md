# GNET parallel execution for v0.1 (design proposal)

Status: proposal for user approval, 2026-10-15.
Nothing here is implemented, accepted or a performance claim.
It changes no contract by itself; the contract changes it needs are listed as decisions in section 11.
Superseded in part: the P1, P3 and P4 plan of section 9 is superseded by the law-directed design in `docs/design/net-parallel-laws-v01.md` (#133), adopted on 2026-10-15; the rest of this document stands, with the strict-f32 corrections to sections 3 and 8 below.
Authority: `docs/general-gpu-v01.md`, `docs/cuda-parity-v01.md` and `src/general_net/CONTRACT.md` govern; this document proposes how to meet them.

Reconciliation, 2026-10-16: section 3 follows the GNET oracle at the external host-array boundary. Private source prefixes survive internally; external arrays commit only on normal return. No oracle behavior changes.

## 0. Recommendation in one paragraph

Keep the verified net as the single compilation input, and cut it into **regions** on the net-derived MIR.
Each region gets a typed legality certificate checked against the net proof, never a filename, hash or benchmark ID.
Regions that an existing M4 family admits (map, reduction, scan, segmented, compaction, gather/scatter/histogram, dense frontier, transpose/stencil, GEMM, summaries) lower to that family's verified device IR and emitters through stage "a"'s extension point.
GNET emits only two parallel kernel kinds of its own: a **body map** (one lane per independent iteration, with an arbitrary compiled loop body, the form robotics batches need) and a **fork-join task region** (independent recursive calls, many groups, children that fork again).
Serial control stays in a compiled frame, placed on the host as native code by default and on the device (today's ABI5 root) only when the control loop consumes device results every iteration.
Spikes on the M4 show the body map close to handwritten parity for batched FK (resident ratios 0.82 to 1.32 across sizes, host-to-host 0.95 to 1.10, bitwise-identical output), while a structure-preserving fork-join is 2 to 3x slower than an expert flat reduction on divide and conquer; parity there needs a lawful algebraic rewrite (recursion to reduction), and some general-dynamic cases may still fail the gate.

## 1. Current state (verified from code, 2026-10-15)

- `compiled/device.rs` emits one kernel `invar_gnet_children_v5`; `group_threads()` is 32 and `groups()` is 1, and the kernel writes status 8 for any other grid (`device.rs:241` CUDA, Metal equivalent).
- `crates/invar-metal/src/macos/gnet.rs` dispatches exactly one 1x32 threadgroup per submission.
- Thread 0 runs the root continuation for a finite quantum per round (`device.rs:191`); lanes 1..8 run speculative children in private slots, and children run with `spec = nullptr`, so **a child can never fork again**.
- `speculation.rs:40` caps slots at 8; the CLI default (`SpeculationLimits::default`) is 2 slots and quantum 32.
- `speculation_analysis.rs` extracts independent same-block calls conservatively (scalar signature, no owners, no dependent arguments).
- The device host boundary is scalar: entry parameters other than i32/u32/Bool are `unsupported_type`, and f32 programs are refused (probe below).
- Every device run writes the full Demand/Complete trace; the trace buffer is sized by `trace_events`.
- The M4 family recognizers on their branches (`segmented_source`, `indexed_source`, `transpose_source`, `stencil_source`, `contraction_source`, `summary_source`) are **whole-entry checked-source matchers** (`from_source(source, entry)` / `from_checked(source, entry)`); only `src/plan.rs` `classify` (via `loops::single`, `recurrence::candidate_in`, `laws::reduction_in`) recognizes MIR loop regions inside arbitrary functions.
- The GNET-05 corpus (draft #131, `evidence/gnet-05/EVIDENCE.md`): 60 generated programs in 5 families, 240 runtime inputs; on Metal all 51 admitted programs match the oracle exactly, 528 child lanes spawned, 411 committed, peak width 8 (the slot cap); 16 of the 51 have nothing to parallelize; 9 are refused `compiler_structural_budget` (8 compose, 1 branch).
- Cold Metal compile dominates: the first corpus run took 1884 s, mostly `MTLCompilerService` at 100 percent on single kernels, against 27 s with a warm shader cache; a compiled-ahead `.metallib` (#129) does not remove the backend step that runs at pipeline creation.
- The CLI's default speculation limits (2 slots) are smaller than the 8 lanes the tests use, and `invar gnet-device` refuses capacity 8 with `Compiler(Limit)`.
- The language has no `sin`, `cos` or `sqrt` intrinsic on main or any open branch (searched `src/` on main, #109, #108, #121).

Measured on the Apple M4 with the GNET-04 branch binary (section 8 has the method): the existing path computes the `divide_conquer` fixture at n = 4096 in about 1.3 to 1.4 s of device-blocked wall time over 95 submissions, against 0.002 to 0.009 ms resident and 0.15 to 0.19 ms host-to-host for a handwritten kernel.
That gap is the problem this design addresses, and it is not a tuning gap.

## 2. Architecture

```text
checked source
  -> Net::lower / Net::verify (unchanged, CONTRACT_VERSION 1)
  -> Net::compiled_mir(proof)                     (unchanged)
  -> region partition + RegionCertificate          (new, section 3)
       F: family region   -> stage-a extension -> family device IR -> family emitter
       M: body map        -> GNET body emitter (reuses compiled/emit.rs per-lane bodies)
       T: fork-join task  -> GNET task emitter (generalises ABI5 slots)
       S: serial control  -> compiled frame (host native, or device frame kernel)
  -> placement per region (section 6) with forced modes for coverage
  -> one ordered submission per epoch; publication at fences (section 5)
```

The frame is the residual sequential program.
A region boundary is an explicit continuation state in the frame, like today's call split points.
The frame never interprets graph opcodes; regions are program-specific emitted kernels.

## 3. (i) Which regions become parallel kernels, and how they are recognized and proven legal

Recognition runs on `Net::compiled_mir(proof)` plus the net's own annotations (call classification, effect tokens, loan facts, block routes).
It never reads the source AST, file names, source hashes or benchmark IDs.
A region is identified by `(function, head block or call site, net fingerprint)` and is invalidated by any net mutation, exactly as the proof is.

Every region carries a `RegionCertificate`:
`{ region, kind, footprint: read/write sets per array place, effect class, trap mode, fuel mode, law id (if any), guards }`.
A verifier re-derives the certificate from the current net and rejects any mismatch, the same pattern as `Net::verify` and `indexed::verify`.
Cost estimates never produce or alter a certificate.

| Region | Recognized on the net as | Legal when (certificate) | Lowering |
| --- | --- | --- | --- |
| Data-parallel array loop (map) | counted `ForNext` loop over index `i` in `[a, b)`; every `StoreIndex` target index is `i` (affine, stride 1) into a place not read at another index; body has no `Host` effect, no allocation that escapes the iteration, no call into an owner-returning function | the parallelism-foundation §3 condition `W(i) ∩ (R(j) ∪ W(j)) = ∅` for `i ≠ j` from affine footprints; read-only inputs leased for the region; traps allowed (resolved by minimum `i`); fuel by per-iteration step prefix | family map if the body is in its op set, else **M** |
| Reduction / scan | loop-carried scalar accumulator with a sealed law (`laws::reduction_in`, `recurrence::candidate_in`) | law id in the certificate; checked signed arithmetic regroups only under a guard that proves no overflow (interval analysis on the body plus a runtime length guard), otherwise the fold keeps source order | family collective |
| Segmented / nested loop | outer map whose body contains an inner loop with an outer-dependent trip count | outer map is legal as above; inner loop is either sequential in the lane (default) or a lawful segmented reduction or stable compaction | **M** (moderate flattening) or family segmented, chosen by placement among legal variants |
| Frontier worklist (array form) | loop over rounds with a dense mask/adjacency array, monotone `seen` update | M4-05 frontier certificate re-derived on MIR | family frontier |
| Independent recursive calls | call sites in one block whose arguments do not depend on earlier sibling results and whose callees are observationally pure (no `Host`, no owner escape), as `speculation_analysis` does today, extended to calls inside children | same `ForkPlan` serialization reasons; additionally a bounded arity per fork site and a static join continuation | **T** |
| Batched robotics math | no special recognizer: FK, Jacobian, mass (CRBA), rollouts and per-query distance are maps over the batch whose bodies loop over joints, links, horizon steps or spheres | map legality on the batch index; inner loops sequential | **M**, plus min-distance and argmin reductions under the NaN-lifted f32 min rule (NaN elements are the identity, the leftmost minimum wins ties, signed zero is preserved; #133 law L7); f32 cost sums stay in source order inside each lane, and a cost sum across rollouts is not a family reduction under strict-f32-v1 |

Two rules make this general rather than a template list.
First, a missing family never removes coverage: anything legal as a map becomes **M** with the full compiled body, including branches, inner loops, calls and traps.
Second, anything that is not a legal region stays in the frame and still runs (on the device frame in forced-device mode), so whole-program coverage is unchanged from GNET-03.

Trap and fuel semantics inside a region follow source order exactly.
A map records a per-lane status word; the first source-order trap is the minimum trapping index, found by a min-reduction after the fence.
Source fuel is charged as the prefix sum of per-iteration steps; if the prefix crosses the remaining fuel, the frame reports fuel exhaustion at the exact iteration and replays only that iteration serially to recover the span.
Region writes go to private scratch. After trap and fuel resolution, the internal frame retains the exact source-order prefix, including writes within a failing iteration recovered by serial replay. External host `mut` arrays are committed only when the whole entry returns normally. A trap, fuel or trace cut, or backend failure leaves those arrays unchanged, even after earlier regions completed successfully. This follows the semantic authority, `general_net::oracle::observe_call` (and HA-1.1); internal prefix reconstruction does not publish an external prefix. The P2 regression `p2_host_arrays_commit_only_on_normal_return` pins this boundary on the CPU model and physical backends.

## 4. (ii) Unification: lower to family IR, or emit GNET's own kernels?

**Answer: both, with a hard split.**
GNET lowers every region a family admits to that family's device IR and emitters through stage "a"'s extension point, and keeps serial control in the frame.
GNET emits its own kernels only for the two forms the families do not and should not admit: the body map **M** and the fork-join task region **T**.

| Option | For | Against |
| --- | --- | --- |
| Lower to family IR only | one backend per family, already verified and parity-gated by M4 on both targets; one place to tune; no duplicate emitters | families admit narrow op sets (device IR v1 ops are `WrappingAdd`/`WrappingMul`/`CheckedAdd` on u32); no recursion, no arbitrary loop bodies, no f32 body with control flow; robotics bodies and recursion would stay serial |
| GNET emits all its own parallel kernels | covers every form; one code path from net to kernel | re-implements and re-tunes seven families; two emitters per family to keep at parity; M4 gates would no longer cover what actually runs |
| **Hybrid (recommended)** | families keep their tuned emitters and gates; GNET adds exactly two kernel kinds that reuse its existing per-lane body emitter (`compiled/emit.rs`) and ABI5 slot protocol | needs a region-level MIR entry into every family (they are whole-entry AST matchers today); two kernel kinds that GNET must itself bring to parity |

Required interface from stage "a" (proposal; stage "a" owns `src/schedule/` and decides the shape):
a function that takes a MIR loop region (function, head block, the verified MIR from `compiled_mir`, and the region's footprint facts) and returns either a verified family plan with its emitters or a typed miss.
The existing `plan.rs` `classify` already has this shape for map, scan and reduction; each M4 family needs a MIR-region recognizer that produces the same family IR its AST recognizer produces today.
The AST recognizers remain as the M4 acceptance path; the MIR-region entry is additive.
Each family's MIR-region recognizer must be differentially tested to select the same family IR as its AST recognizer on every M4 corpus program, so the M4 parity evidence transfers.

## 5. (iii) Launch structure, and how the frame coordinates with regions

### Control placement

- **Host frame (default).** The frame is compiled native CPU code (the GNET-04 D2 recommendation: LLVM native).
  At a region state it binds resident buffers, encodes the region's kernels into the current command buffer (Metal) or stream (CUDA), and continues encoding until it needs a device value.
  This is the Futhark host-code model; robotics pipelines (FK, then collision, then cost) are straight-line region chains and never wait mid-chain.
- **Device frame.** When a loop in the frame reads a region result every iteration (convergence loops, rounds of a frontier), the frame runs on the device as today's ABI5 root, and each epoch is encoded as: frame quantum, then one indirect dispatch per region site whose group count the frame wrote (zero when that site is not pending), then the next frame quantum.
  Metal uses `dispatchThreadgroups(indirectBuffer:)`; CUDA launches each region with a grid sized for the site's static capacity and a device-side length guard.
  Epochs per submission are bounded; the host checks the frame's state word between submissions, as GNET-03 does now.

Placement picks between the two per program from the certificate facts (does the control consume device values per iteration) and cost (section 6), with both kept available in forced modes.

### Region kernels

- **Family regions** use their own launch geometry (grid-stride, multi-group) unchanged.
- **Body map M**: one lane per iteration, `ceil(n / 256)` groups of 256, grid-stride above the per-dispatch group cap; the lane runs the compiled body with private frames bounded by the body's static frame need; status and step words per lane; a min-reduction for traps and a prefix sum for fuel run only when a lane reports a trap or the fuel guard is tight.
- **Fork-join T** (generalises ABI5 from 1x32 with depth-1 children to many groups with nested forks):
  1. *Expansion.* For fork sites with a static arity, lane `t` replays the pure prologue down the path selected by the digits of `t` (no frontier memory, one dispatch).
     For data-dependent arity, a level-synchronous frontier with a bounded task/frame arena and indirect dispatch per level.
     Expansion stops at a target width from placement (a multiple of device lanes) or when the arena is full.
  2. *Execution.* Each frontier task runs its subtree with the existing compiled continuation machine, bounded private frames and a finite quantum per epoch.
     A full arena or frame stack serializes the call in its parent lane (`capacity_serialized`), never failing the program.
  3. *Join.* Post-fork continuations run bottom-up, in threadgroup memory for the lowest levels and one small dispatch per remaining level group.
  Lanes that idle while others still have deep subtrees may promote their next fork point to new tasks (heartbeat-style promotion) in the next epoch; v0.1 implements fixed-width expansion first and adds promotion only if GNET-05 skewed cases need it.

### Fences, quiescence and publication

- Ordering between frame, regions and the next frame quantum is command-buffer encoder order (Metal) or stream order (CUDA); no host round trip inside an epoch.
- No cross-threadgroup spin-waits or persistent kernels: Metal gives no inter-threadgroup forward-progress guarantee, and CUDA gives it only under cooperative launch; bounded epochs stay the progress mechanism, as the contract requires.
- A region's outputs become frame-visible only after its completion fence and its trap/fuel resolution; speculative T results commit to the frame only at demand, in source order, with ABI5's generation check.
- Reclaim of region scratch waits for command completion; unknown completion keeps the existing quarantine (`gnet.rs` `Drop` retention).
- Data stays resident across regions; host-to-device transfers happen only at entry and exit of the operation, which is what the warm host-to-host scope measures.

## 6. (iv) Speculation and placement (GNET-04)

Placement chooses only among already-certified variants: for each region, CPU native or GPU (family or M/T), and for each program, host frame or device frame.
Inputs are the runtime shape and length, residency, and calibrated constants per region kind (launch latency, per-element or per-step throughput, transfer bandwidth) measured by the M5-01 calibration on the gate hosts.
Before M5-01 freezes, the GNET-04 D1 option (a) applies: CPU by default with every legal device variant and serialization reason in explain and receipts, and forced modes for coverage.
After the freeze, option (b): the measured model with an uncertainty margin and hysteresis; option (c) (keyed online history) stays later.

Interaction with source-ordered commit:
- A map, reduction or family region that the frame has reached is **demanded work, not speculation**: every iteration would run in source order too.
  Its only ordering obligations are the first-trap and fuel rules of section 3.
- A fork-join T region **is speculation** for every sibling after the oldest.
  The ABI5 rules carry over unchanged: finite quanta per task, the oldest source-order task scheduled every epoch, speculative scratch separate from the reserved serial frame resources, commit only at demand in source order, and a later sibling's trap, fuel or allocation failure is never published while an older sibling has not completed.
  If the speculative work budget (a fixed multiple of remaining source fuel) is exhausted, the region cancels non-oldest tasks and continues serially on the oldest path, which preserves oldest-frontier progress.
- Switching a region between CPU and GPU happens only before submission or after confirmed quiescence (the M3-05 rule); the frame's live continuation is never replayed after a visible effect.

## 7. (v) Correctness

- **Oracle.** The CPU net oracle remains the semantic reference.
  Device results must agree word for word on status, value, first source-order trap and its span, source steps, cleanup prefix and the source-order Demand/Complete sequence, under strict-f32-v1 and the integer contracts.
  Physical transition counts necessarily differ in parallel execution and are reported separately (decision D3).
- **Tracing.** Event traces are produced only in verification mode; a parallel region reconstructs its source-order events from per-lane records after the fence.
  Performance runs execute trace-off variants, which are separate certified artifacts compared against the traced ones on the full corpus.
- **Differential tests.** Every runnable corpus program (332 today) and the 240 seeded GNET-02 generated programs run in forced serial-device, forced parallel-device, CPU native and oracle modes; the six `tests/fixtures/gnet_parallel` programs (#120) plus the GNET-05 generator families run at capacity 0 (serial control) and full width.
  Oracle comparisons run on the 64 MiB helper thread in `tests/support/gnet_agree.rs` (#131), because the reference evaluator overflows a 2 MiB Debug test thread before its 64-frame limit.
  Boundary sizes: 0, 1, lanes plus or minus 1, group size plus or minus 1, arena capacity plus or minus 1, and per-array limits.
- **Fuzzing.** Extend `tests/support/gnet_generator.rs` (#99) with region fragments: maps with a trap at a random index, reductions one step from overflow, nested loops with skewed trip counts, forks with random arity and depth, an older divergent sibling ahead of a trapping or allocating younger one, and fuel cuts inside a region.
- **Mutation plan** (each must be killed by a named test): take the last instead of the first trap index; regroup a checked signed sum without its guard; drop the generation check on T commit; remove a group barrier in a join; off-by-one in the grid length guard; ignore arena overflow; wrong fuel prefix; commit a younger sibling before an older one; publish a region output before trap resolution.
- **Fault injection.** Reuse #120's `gnet-fault-injection` feature and its barrier-bracketed hook discipline.
  New faults at region phase boundaries: stale region descriptor generation, poisoned partial results with a stale generation, forced arena exhaustion, cancellation at an epoch boundary.
  The CPU group model of every new kernel runs under ThreadSanitizer, with a barrier-removal mutation that TSan must report, as #120 did.
- **Numerics.** Strict f32 needs no contraction and no fast math on both targets (Metal `mathMode = safe` with precise functions; CUDA `--fmad=false`, no FTZ).
  The FK spike confirms bitwise-identical outputs between the generated-shape and handwritten kernels under these options (0 differing words of 12,582,912).

## 8. (vi) Performance model, with spike evidence

### Spike method (throwaway, outside the repository)

Code: `$HOME/.cache/invar-gnet-parallel-spike/` (`kernels.metal` sha256 `678a901a...febed1`, `spike.m` sha256 `8ab0372a...886190`, fixture `dc.inv` sha256 `311918f1...569303`).
Host: Apple M4 (10 GPU cores), macOS 27.0, MSL compiled at run time with `mathMode = safe` and precise math functions for both sides.
Each series: 10 warmups, 30 randomized interleaved pairs, **one session**, so the bound shown is a single-session paired bootstrap, not the three-session contract statistic.
"Resident" is command-buffer GPU time; "host-to-host" includes input copy into shared buffers, commit, wait and output copy.
The Mac was shared with other jobs during the runs; repeated runs moved individual ratios by up to about 0.2.

"Generated shape" kernels are hand-written to look like what this design would emit (runtime model arrays, checked bounds, per-lane status and fuel words, the compiled continuation machine), not output of an implemented compiler.

### Spike 1: irregular divide and conquer (`divide_conquer` fixture, checked i32)

| n | handwritten flat reduction, resident | level-synchronous T (73 dispatches), resident ratio | path-replay T (2 dispatches), resident ratio, best expansion depth D | path-replay T, host-to-host ratio |
| --- | --- | --- | --- | --- |
| 4,096 | 0.002 to 0.009 ms | 56x | 4.7x (D=14) | 1.12x |
| 65,536 | 0.004 to 0.018 ms | 30x to 33x | 2.6x (D=14) | 1.13x |
| 1,048,576 | 0.03 to 0.12 ms | 6.9x to 7.1x | 1.9x to 2.1x (D=14, 16) | 1.13x to 1.46x |
| 16,777,216 | 0.44 to 1.20 ms | 3.4x to 4.4x | 2.0x to 2.7x (D=14 to 18) | 1.7x to 2.1x |

All variants agree with the CPU reference on every size.
Expansion depth D was swept over 14, 16 and 18 (16,384 to 262,144 lanes); deeper expansion costs more than it gains below 2^24 elements.
The existing GNET 1x32 path at n = 4,096: 1.29 to 1.41 s device-blocked wall time (wall minus user minus sys over three runs, 95 submissions, full event trace, capacity 2 as the CLI allows), so roughly four to five orders of magnitude off.
Removing the checked add from the leaf and join (a guarded variant) recovered only about 15 percent; the cost is the per-lane recursive continuation machine against a flat strided loop.
Conclusion: structure-preserving fork-join gets within about 2x but **cannot reach parity against an expert flat algorithm**; parity requires the compiler to find the same algorithm, here the list-homomorphism rewrite `dc(lo, hi) = fold(+, map(mix, [lo, hi)))`, which is lawful for checked i32 only under a no-overflow guard.

### Spike 2: batched UR5e FK (6 joints, 3x4 pose, f32)

| configs | resident ratio (body map / handwritten) | host-to-host ratio | bitwise differences |
| --- | --- | --- | --- |
| 1,024 | 0.82 to 0.83 | 0.95 to 1.00 | 0 |
| 65,536 | 1.12 to 1.32 (upper 1.26 to 1.63) | 1.04 to 1.10 | 0 |
| 1,048,576 | 1.03 to 1.05 (upper 1.13 to 1.14) | 1.01 (upper 1.06) | 0 |

The handwritten kernel uses a constant model and an unrolled joint loop; the body map reads the model from a runtime array with checked bounds and status words.
The resident gap (3 to 5 percent at 1,048,576; 12 to 32 percent at 65,536, where run-to-run noise was largest) is most likely the runtime model array and its bounds checks, which this spike did not isolate; specialising robot-model constants into the emitted kernel (with a guarded generic variant) is the planned remedy (decision D7), to be confirmed in P2.
The existing GNET path refuses this workload (`unsupported_type`, probed with a `filled_f32` program).

### Per-category expectation

| Category | Mechanism | Expected vs handwritten | Main risk |
| --- | --- | --- | --- |
| Two-arm FK, full Jacobian | M over configurations, inner joint loop | parity (spike 2) once model constants specialise | trigonometry contract (D6); Jacobian output width favours cooperative per-element mappings (cuRobo uses 16 to 32 threads per batch element) |
| Mass matrix (CRBA) | M, inner loops over links | likely within 1.0 to 1.1 per element; parity depends on register pressure | Panda mass reference is a recorded failed cross-library gate, separate from compiler parity |
| Collision / distance | M over (query, sphere set) plus an f32 min-reduction under the NaN-lifted rule; sparse candidates as T | parity for dense sphere sets; at risk for sparse traversal | handwritten baselines use BVH or voxel structures the source may not express |
| Rollout / cost | M over rollouts with a sequential horizon loop; each rollout's f32 cost accumulates in source order inside its lane; argmin over rollouts under the NaN-lifted rule | parity expected | none beyond trig |
| Point cloud transform / filter / compact | M, then family stable compaction (M4-04) | parity expected for transform and compaction | 1,000,000-element source array limit (use SoA arrays); device caps raised separately (#123) |
| Point cloud neighborhood | grid hashing needs a sort or segmented family | **at risk**: no sort family exists | likely a v0.1 gap unless a radix-sort family lands |
| Composed robotics pipeline | host frame chaining resident M and family regions in one submission | parity expected if every stage is at parity and there are no host waits | launch count; fusion of adjacent maps (M3-03 laws) |
| Irregular recursive divide and conquer | T, plus recursion-to-reduction rewrite when a law and guard hold | 2x to 3x without the rewrite (spike 1); parity only with it | data-dependent splits defeat the rewrite |
| Input-dependent branch trees | T with data-dependent arity; divergence within SIMD groups | **at risk** | expert baselines flatten to a level-synchronous frontier with compaction; T must match that |
| Discovered frontiers | dense form: family frontier; recursive form: level-synchronous T | dense: per M4-05; recursive: **at risk** | per-level dispatch cost on small frontiers |

Strict-f32 correction (adopted with #133, decision F2).
Reassociating an f32 sum changes its result for almost every input, so under strict-f32-v1 an f32 cost sum is a source-order fold: it parallelizes only across independent accumulators (one per rollout lane), and a sum across rollouts is not a family reduction unless a separately declared f32 summation contract exists (deferred, decision F1 of #133).
An f32 min or argmin is exact under any tree shape only with the NaN-lifted f32 min rule (NaN elements are the identity, the leftmost minimum wins ties, signed zero is preserved; #133 law L7); the family reduction must implement that rule, and the baseline must match it.

### What each required robotics workload needs from the parallel path

The GNET-05 corpus does not yet contain these; the GNET agent is building them and their matched baselines engine-independently.
Each row lists what the workload's source must be able to express, which regions the parallel path must produce for it, and what the baseline must match.

| Workload | Source form the corpus should use | Regions needed | Prerequisites | Baseline (D) must match |
| --- | --- | --- | --- | --- |
| Two-arm FK (Panda 7, UR5e 6) | `fn fk(q: read Array<f32>, model: read Array<f32>, out: mut Array<f32>)`, loop over configurations, inner loop over joints computing angle-driven transforms | M over configurations; nothing else | P0a trig, P0b f32 array entry/`mut` output; D7 for constant models | same trig contract (D6), same op order, no FMA; all-body or tool-only output stated per case |
| Full Jacobian | as FK plus per-joint columns (6 x n per configuration) | M over configurations; later a team-per-configuration variant if output width hurts (cuRobo maps 16 to 32 threads per element) | as FK | same columns and frame convention |
| Mass matrix (CRBA) | M over configurations; inner loops over links with a backward pass; symmetric n x n output | M | as FK; records help readability but are not required | Panda mass stays a recorded failed cross-library gate; compiler parity compares against a baseline implementing the same source algorithm and contract |
| Collision / distance | per-query loop over a sphere set (robot spheres vs obstacle spheres/boxes), min-reduction over pairs, a boolean or distance per query | M over queries, f32 min-reduction under the NaN-lifted rule; sparse candidate lists later as T | trig not needed for precomputed-transform controls; `sqrt` for distances (P0a) | identical geometry and tolerance; no false negatives; precomputed-transform controls are separate cases |
| Rollout / cost | loop over rollouts, sequential horizon loop with a dynamics step and running cost, then argmin over rollouts | M over rollouts, f32 argmin under the NaN-lifted rule; no cross-rollout cost sum (strict-f32-v1 keeps source order) | P0a, P0b | same horizon, seed policy and stopping rule |
| Point-cloud transform / filter / compact | transform loop over points (SoA arrays), predicate loop, stable compaction into a fresh array | M, then family stable compaction (M4-04) | P0b; SoA arrays to stay under the 1,000,000-element source limit | same predicate and element order |
| Point-cloud neighborhood | voxel or grid binning plus per-point neighbor scan | needs a sort or segmented family; without one it runs as M with an inner scan (quadratic) | sort family (not planned in v0.1) | **at risk**; report as a gap rather than a slower algorithm |
| Composed pipeline | one entry calling FK, then collision, then cost on resident arrays | host frame chaining M and family regions in one submission, no host waits | all of the above | the complete operation, with equivalent residency |

None of these needs the fork-join T region; all of them need P0a/P0b, the body map and the host frame, which is why those come first in section 9.

Small sizes are dominated by command-buffer latency (about 0.15 ms host-to-host on the M4 even for the handwritten kernels); those cases rely on correct CPU placement and the existing dispatch-overhead gates, not on GPU parity.

### Compile cost, kernel shape and binary archives

The 1884 s cold run is a kernel-shape problem first: ABI5 emits one large kernel per program that contains every continuation state and every child projection, and the Metal backend compiles it as one function.
This design shrinks what the device compiles:
- with the host frame as the default, most programs no longer compile a device frame kernel at all;
- body maps and family kernels are small, one function per region;
- the device frame and T kernels contain only the functions reachable from their region, not the whole program;
- variants per region stay within the M4-06 bound of 32, and placement prefers fewer variants when they tie.
A compiled-ahead `.metallib` (#129) only removes the front end; the backend compile from AIR to GPU code still happens at pipeline creation.
`MTLBinaryArchive` stores those backend results, but an archive is specific to the GPU family and OS build, so it must be built on (or for) the exact gate device and is invalidated by OS updates.
**Binary archives are a v0.1 need** for honest cold-process numbers and for usable CI on Metal: P5b adds an archive cache keyed like the M4-06 cache (source, net, region and rule IDs, numeric/layout policy, target GPU family, OS build, toolchain), revalidated on load, with a cold-process report that states archive hit or miss.
The scored resident and warm host-to-host scopes are unaffected; cold-process and first-result costs are reported separately as the contract already requires.
CUDA has the analogous answer already: the compiled-ahead CUBIN (#122) contains machine code for the gate GPU, so no driver JIT runs.

The 9 `compiler_structural_budget` refusals (8 compose) are frame-size refusals.
P1 records which limit each one hits (values, places, continuation states or source bytes) before any budget changes; the host frame (LLVM native) does not carry the device frame's limits, so those programs become admissible in the default placement, while forced device-frame mode keeps the refusal visible.

## 9. (vii) Scope cut, phases, sizes, dependencies and parallel work

### v0.1 cut

In v0.1: region certificates, the body map M on CUDA and Metal, the family bridge for the families stage "a" exposes, the fork-join T with fixed-width expansion and nested forks, the recursion-to-reduction rewrite for range-splitting recursion with a sealed law, the host frame, the device frame with indirect epochs, placement option (a) then (b), and GNET-05 wiring.
Later: heartbeat promotion and work donation, persistent or cooperative kernels, CUDA graph conditional nodes, cooperative per-element (team) mappings for wide outputs, BVH traversal specialisation, a sort family for neighborhoods, online history placement, multi-device.

### Phases (sizes are estimates of new or changed Rust/emitted-template lines, tests separate)

| Phase | Content | Files (main) | Code / tests | Depends on |
| --- | --- | --- | --- | --- |
| P0a | strict-f32 `sin`/`cos`/`sqrt` contract and oracle | `numeric_contract.rs`, oracle, emitters | 800 / 600 | D6 |
| P0b | f32, records and host array arguments at the GNET device entry | `general_net/device_host`, `compiled/owned_f32.rs` | 1,000 / 800 | HA-1, #109 |
| P1 | region partition and `RegionCertificate` verifier on `compiled_mir` | new `general_net/regions/{mod,map,fork,cert}.rs` | 1,200 / 900 | #99, #114 merged |
| P2 | body map M emitter and host-frame launcher, CUDA and Metal | `compiled/body_map.rs`, `device_host`, `invar-metal`/`invar-cuda` glue | 1,300 / 900 | P1 |
| P3 | family bridge: MIR-region entries per family, AST-agreement tests | `general_net/regions/family.rs`; per-family recognizers coordinated with stage "a" | 1,800 / 900 | P1, stage "a" interface |
| P4 | fork-join T (ABI6): nested forks, many groups, arena, joins; recursion-to-reduction rewrite | `compiled/task.rs`, `compiled/task_layout.rs`, `regions/homomorphism.rs` | 1,700 / 1,200 | P1 |
| P5b | Metal binary archive cache keyed by the full cache identity; cold-process reporting with hit/miss | `invar-metal/src/macos/library.rs` (with #129), cache module | 400 / 300 | #129 |
| P5 | device frame epochs with indirect dispatch | `compiled/device.rs`, `device_fence.rs`, `invar-metal/src/macos/gnet.rs`, `invar-cuda` | 600 / 500 | P2 |
| P6 | placement (a), explain and receipts; then (b) after calibration; the `gnet-device` CLI takes the same certified limits as the tests (today it caps at 2 slots) | `general_net/placement.rs`, explain schema | 500 / 400 | P2, M5-01 freeze for (b) |
| P7 | GNET-05 wiring, parity runs on the RTX 5070 Ti and the Apple M4 | bench runner integration | 400 / evidence | P2 to P6, GNET-05 corpus, #122, #129 |

Total: about 9,700 lines of code and 6,500 of tests, plus evidence.

### Dependencies

- M5-01 freeze (#121): calibration constants, frozen baselines and the profitable size band, before placement (b) and any scored timing.
- #122 (CUDA AOT CUBIN) and #129 (Metal AOT `.metallib`): scored runs use ahead-of-time artifacts; runtime MSL/NVRTC compile is reported separately.
- Stage "a" (`feat/v01-family-dispatch`): the region-level entry for P3.
- GNET-05 corpus (#131, `feat/v01-gnet-05-corpus`), its robotics workloads (in progress) and the reviewed handwritten baselines for P7.
- #99, #114, #120 merged first: ABI5, the fault hook and the parallel slice are the base P4 generalises.

### What can run in parallel

- From day 0: P0a (numeric owner), P0b (GNET agent after #114/#120), P1 (GNET main), and stage "a" finishing its extension point.
- After P1: P2, P3 and P4 in parallel, each in its own files (`body_map.rs`; `regions/family.rs` plus coordinated recognizer files; `task.rs`); one writer per file.
- After P2: P5 and P6(a); P5b any time after #129, independent of the rest.
- P7 last, one timing job at a time per host.

## 10. Top risks

1. **General-dynamic parity.** Spike 1 shows structure-preserving fork-join at 2x to 3x of an expert flat kernel; the gate allows the baseline a different algorithm, so divide and conquer, branch trees and recursive frontiers pass only where the compiler finds the same algorithm under a law and guard.
   Some GNET-05 cases may remain failed, which blocks v0.1 under the current gate.
2. **Prerequisites outside GNET.** No trigonometry or sqrt intrinsics, a scalar-only device entry, f32 refused on the device route, and a 1,000,000-element source array limit; the robotics categories cannot run on GNET at all until P0a and P0b land.
3. **Launch and latency floors.** About 3 to 6 microseconds per mostly empty dispatch and about 0.15 ms per command-buffer round trip on the M4; deep level-synchronous structures and device-frame loops pay these per level or epoch, and Metal and CUDA differ in indirect-dispatch support, so the two targets need different epoch encodings.

Further risks: cold Metal backend compile (1884 s cold versus 27 s warm on the GNET-05 corpus) until smaller kernels and binary archives land, the observation contract change for parallel regions (D3), stage "a" interface churn, family-recognizer divergence between AST and MIR entries, and emitted-code size and compile time for many region variants.

## 11. Decisions needed before implementation

| # | Decision | Recommendation |
| --- | --- | --- |
| D1 | Unification: lower family-admitted regions to family IR and emitters, with GNET owning only the body map and fork-join kernels | Yes (section 4) |
| D2 | Control placement: host-native frame by default; device frame only for control loops that consume device results each iteration | Yes |
| D3 | GNET observation contract version 2 for parallel regions: compare source-level observations (status, value, first source-order trap and span, source steps, cleanup, source-order Demand/Complete reconstructed in verification mode); report physical transitions separately; production runs trace-off | Yes; it is required for any parallel execution |
| D4 | Admit the recursion-to-reduction (list homomorphism) rewrite in v0.1, only for range-splitting recursion with a sealed law and a no-overflow guard for checked integers | Yes |
| D5 | If a GNET-05 general-dynamic case cannot reach parity: keep the gate and report the case failed (v0.1 stays blocked), or amend the gate for that category | Keep the gate; revisit only with P7 data in hand |
| D6 | Strict-f32 trigonometry and sqrt: a frozen software algorithm (`invar-trig-f32-v1`, IEEE basic operations plus explicit `fma`, bit-identical on CPU, CUDA and Metal), with baselines required to implement the same contract | Frozen software algorithm; correctly rounded CORE-MATH style needs double precision that Metal lacks; vendor `sin`/`cos` differ across targets and cannot be the oracle |
| D7 | Allow compile-time specialisation of robot-model constants from checked robot declarations into emitted kernels, with a guarded generic variant | Yes |
| D8 | Epoch mechanism on CUDA: host-encoded epochs with static-capacity grids and device-side guards (portable with Metal's indirect dispatch), versus CUDA graph conditional nodes or cooperative persistent kernels | Host-encoded epochs for v0.1; the others later, only with profile evidence |
| D9 | Pending GNET-04 D1/D2 from #120: placement before calibration is CPU default plus explain; the compiled CPU engine is LLVM native | Confirm (a) and (a) |
| D10 | Metal binary archives in v0.1, as a validated cache keyed by GPU family and OS build, built on the gate device | Yes; smaller kernels first, archives for the remainder |

## Appendix A. Survey (reuse before writing)

Repository facts checked with `gh` on 2026-10-15 (stars, last push, license).

| Item | What to reuse | What to port | What does not fit |
| --- | --- | --- | --- |
| HVM2 (`HigherOrderCO/HVM2`, Apache-2.0, `src/hvm.cu`) | as optional comparison "A" only | its scheduling ideas: host-driven turns (`evaluator` plus `inbetween` kernels, host reads the redex count each turn), log-doubling SEED/GROW spawn inside a block (thread `i` hands work to `i ^ span`), and "parallel" flags on redexes that may be shared; these inform T's expansion | a generic reducer with 32-bit ports, no ownership, no source-order commit; the paper's section 13 documents eager-demand and allocation overheads |
| HVM3 (`HigherOrderCO/HVM3`, MIT) and Bend (`bendlang/bend`, Apache-2.0) | nothing directly | Bend's `fork`/`bend` shows source-level fork-join that maps to T | generic runtimes; Bend 2 states Bend 1/HVM programs do not carry over, and it restricts parallelism to balanced calls |
| Jiresch, GPU interaction nets (arXiv 1404.0076) | none | none | confirms communication overhead of generic GPU net reduction |
| Futhark (`diku-dk/futhark`, ISC) | host-code-plus-kernels split (our host frame) | moderate flattening (outer map parallel, inner sequential: our M) and incremental flattening's multi-versioned code with tuned thresholds (Henriksen et al., PPoPP 2019: our certified variants plus placement) | Futhark rejects recursion, so it offers nothing for T |
| NESL / Blelloch flattening; Data Parallel Haskell vectorisation | none | flattening of recursive data-parallel functions into level-by-level segmented operations: the level-synchronous T variant | full flattening's memory blow-up for deep nests; we flatten only fork-join recursion |
| Dex (`google-research/dex-lang`, BSD-3) | none | index-set `for` as parallel map, same as M | no recursion; research status |
| Halide (`halide/Halide`, MIT) / Taichi (`taichi-dev/taichi`, Apache-2.0) | none | Halide's algorithm/schedule split matches certificate plus placement; Taichi parallelises only outermost loops, exactly M | fixed-function or Python-embedded; no recursion or source-order traps |
| Merrill, Garland, Grimshaw BFS (PPoPP 2012); Gunrock (`gunrock/gunrock`, Apache-2.0) | none (CUDA-only C++ templates) | advance/filter with prefix-sum frontier compaction and warp/CTA neighbor expansion, for the level-synchronous T and any sparse frontier family | library dependency would be CUDA-only and graph-API shaped |
| Persistent threads (Gupta, Stuart, Owens 2012); Whippletree (Steinberger et al. 2014); Atos (`owensgroup/ATOS`, ICPP 2022) | none | queue designs for later work donation | rely on co-residency for progress; Metal gives no inter-threadgroup forward-progress guarantee (Sorensen et al., OOPSLA 2016 on occupancy-bound execution) |
| Kokkos tasking (`kokkos/kokkos`) | none | none | the tasking interface was deprecated (#7393) and removed (#8390) |
| CUDA dynamic parallelism | none | none | CDP2 removed device-side synchronization; per-launch cost and no Metal equivalent |
| Heartbeat scheduling (Acar et al., PLDI 2018); lazy task creation (Mohr, Kranz, Halstead 1991) | none | promote fork points to tasks only when parallelism is needed: the later T promotion | none |
| cuRobo (`NVlabs/curobo`, Apache-2.0) | as a reviewed baseline source for FK and sphere collision, pinned and labelled | its FK kernel layout (`kinematics_forward_kernel.cuh`: 32 threads per batch element for transforms, 16 for outputs, per-link 3x4 matrices in shared memory) for a later team-per-element variant of M | as a compiler dependency; calling it from Invar is the separately labelled integration track |
| GRiD (`A2R-Lab/GRiD`, MIT); Plancher et al., ICRA 2022 | as a baseline candidate for batched dynamics and gradients | its per-robot code generation (unrolled, constants folded) supports D7 | Python code generator targeting CUDA only |
| Pinocchio (`stack-of-tasks/pinocchio`, BSD-2) | already the CPU physical reference in the robotics oracles | none | CPU library; not a device baseline |
| NVIDIA Warp (`NVIDIA/warp`, Apache-2.0) | optional extra baseline for batched kernels | none | Python-embedded JIT; CUDA-only |
| stdgpu (`stotko/stdgpu`, Apache-2.0); BGHT (`owensgroup/BGHT`) | none | GPU hash tables for a later neighborhood family | CUDA-only containers |

Conclusion of the survey: no candidate is a drop-in production component, because each is either a generic runtime (contradicts compiled-first), CUDA-only (we need Metal parity too), or lacks source-order trap and fuel semantics.
The design reuses Invar's own verified family emitters and ABI5 protocol, and ports algorithms (moderate flattening, recursion flattening, Merrill frontier expansion, HVM2-style log-doubling spawn, heartbeat promotion) rather than code.
