# CUDA-class acceptance protocol

For v0.1 the [ordinary-source parity gate](#ordinary-source-parity) below applies: every scored case, size, device and both resident/warm timing scopes needs median Invar/reviewed handwritten ratio at most 1.00 and one-sided 95% upper bound at most 1.05.
Its general dynamic and robotics corpus and independent-session statistics are required, and existing numeric, library, bandwidth, dispatch, Metal and cold-process gates still apply.
General executable-net coverage (`src/general_net/CONTRACT.md`) is a separate prerequisite and cannot be inferred from fast templates.

This document defines future acceptance, not measured Invar performance. Version 1 of the benchmark manifest must be committed before schedule tuning. Changes after baseline collection require a explained manifest revision and rerunning both sides. Keep old results; never remove failing cases to pass a milestone.

M5-01 binds the named GPU model/compute-capability profile, exact large-case shapes, toolchain/driver/compiler flags, launch/workspace/reuse policy and target ratios before M4-06 tuning. Choose hardware through authorized preflight, not after inspecting favorable speed ratios. Qualifying another GPU creates a separate target report and cannot erase a failed first-target result.

## Claims and tracks

The primary claim is automatic generation of efficient kernels from ordinary high-level loops and declared numerical semantics, on one qualified NVIDIA GPU. Users may declare meaningful shapes, ownership and arithmetic policy; they do not provide kernel code, launch geometry, tile sizes, warp intrinsics, or a filename recognized by the compiler.

Use two reported tracks. **Generated** allows generated kernels and target intrinsics, with no dispatch to a pre-existing library implementation of the whole benchmark operation. **Library-assisted** allows explicit compiler dispatch to versioned CUB/cuBLAS/CUTLASS or other vetted implementations, with their licenses and trust boundary recorded. Library-assisted results do not satisfy the generated track. Handwritten CUDA baselines are external comparison fixtures, not Invar implementation code; pin upstream source revisions, review their semantics and preserve attribution. The Rust harness orchestrates them.

A baseline must pass a quality gate before Invar tuning: compare semantically matched vendor and reviewed handwritten implementations where available, keep all measurements, and use the fastest correct eligible reference. When a vendor contract does not match (for example a strict ordered FP32 dot product), document that mismatch and a profiler/resource review against achievable bandwidth/arithmetic throughput, including memory coalescing, materialization, occupancy and synchronization. Pin the reviewed source/configurations and rationale. An intentionally weak baseline or a same-schedule code-generator smoke control cannot establish competitive CUDA performance. Baseline quality review does not allow library dispatch to satisfy the generated track.

Report three timing boundaries separately:

1. **Device-resident steady state:** inputs already on device, output completion included, no compilation or host transfers. Charge all kernels and temporary buffers required by the operation.
2. **Host-to-host warm call:** host input to host output, including transfers, dispatch, launch overhead, synchronization, and required allocations. Both implementations use the same allocation/reuse policy.
3. **Cold process:** compiler/JIT, cache miss, driver initialization, loading and first execution. Report compilation, peak compiler memory and binary/cache sizes separately too.

No single number substitutes for all three. Small workloads should often select native CPU; measure the selection overhead and retain forced-target controls for diagnosis.

## Frozen workload families

The harness must include the following families. Exact random arrays use fixed seeds, plus independent adversarial inputs. CPU-sized correctness samples and larger performance samples are separate; the source interpreter need not execute billion-element cases to establish an oracle.

| Family | Required source/behavior | Shape coverage | CUDA comparison |
| --- | --- | --- | --- |
| Fused map | At least three elementwise stages, two read inputs, one fresh output, wrapping `u32` | 0, 1, 31, 32, 33, 255, 256, 257, 4093, then powers of two through memory budget | Reviewed coalesced fused CUDA kernel |
| Reduction | Modular sum and a noncommutative associative pair composition | Empty, singleton, odd, ragged-block and multi-block inputs | CUB reference and reviewed ordered CUDA tree with matching law |
| Scan/affine recurrence | Every prefix and final-state-only forms, nonidentity seed | Same boundary sizes plus long sequence | Ordered CUDA scan/custom affine operator; permutation forbidden |
| Stable filter/compaction | Predicate preserves source order and handles empty results | 0%, 1%, 50%, 99%, 100% selectivity | Stable CUDA/CUB selection with matching output order |
| Transpose/layout | Rectangular row-major transpose and strided read view | 31x33, 127x257, 1024x1024, larger bounded cases | Shared-memory tiled CUDA transpose |
| Stencil | Out-of-place radius-1 2D stencil, then bounded multi-step ping-pong | Tiny domains, halo boundaries, odd widths, wide/tall, larger square | Tiled CUDA with identical boundary and numeric policy |
| Matrix product | Recognized loop contraction, FP32 ordered track and explicit Tensor Core contract track | Rectangular, 1-sized dimensions, zero reduction extent, odd dimensions, 256/1024/4096 where budget permits | Semantically matched CUDA; cuBLAS/CUTLASS reported as additional reference where contracts match |
| Segmented nested work | Per-segment maps/reductions and a nested loop with data-dependent segment lengths | Empty segments, uniform, skewed, one dominant segment, ragged totals | Reviewed segmented CUDA/CUB implementation |
| Gather/scatter | Read gather, injective scatter, lawful modular histogram conflict mode | Sorted/random indices, duplicates, bounds negatives for rejection | CUDA gather/scatter/atomic histogram with identical conflict semantics |
| Irregular frontier | Bounded level-synchronous traversal or worklist transformation with declared set/order observations | Chains, stars, balanced and skewed small graphs, bounded sparse larger cases | Equivalent level-synchronous CUDA implementation |

The frontier comparison must not require a deterministic visitation order if the source program observes only a set or per-level result. If source order is observable, preserve it or reject transformation; do not silently weaken it in the benchmark. A pointer-chasing chain is a necessary negative case: expose the fallback and dependency explanation, not a fabricated speedup.

Keep performance classes distinct. Strict FP32 GEMM can parallelize independent output cells while preserving each dot-product order; it must not be compared as numerically equivalent to a reduced-precision Tensor Core GEMM. Explicit `fma` and reduced-precision contracts have separate baselines and checks. A tolerance accepts results only under a numerical policy that already permits them; tolerance cannot retroactively legalize a transformation.

## Correctness before timing

All integer outputs compare exactly. For declared floating contracts, compare required ordered bit patterns where applicable and separately specified NaN, subnormal, signed-zero and error criteria. Include cancellation, overflow, underflow, infinities, and ill-conditioned finite cases; do not report only random well-conditioned matrices.

For small inputs use the independent source interpreter and independently derived host/reference results. For large inputs use independently implemented CPU or reviewed CUDA references, identities/metamorphic checks and sampled exact positions, with scope reported. Cross-check reference implementations before treating disagreement as an Invar defect. Checking only checksums is insufficient.

Run boundary sizes for every block/tile selection. Test aliasing, invalid offsets, malformed caches, divergent barriers, widened overreads, unsupported devices, allocation limits, launch errors and incomplete completion using verifier mutations or injected runtime outcomes. Hardware tests add supported device memory/race/synchronization sanitizers. Distinguish a mock lifetime test from a real driver/device result.

## Reproducibility and measurement

A result packet contains repository commit, issue IDs, manifest digest, source and generated-code digests, compiler/toolchain/driver versions, GPU model/UUID or approved non-sensitive identifier, compute capability, CPU, OS, seeds, numeric policy, shapes/layouts, launch and tile configuration, stream policy, workspace size, transfer bytes, and timing boundaries. No credentials or unrelated host data belong in it.

Record shared-device occupancy, power/clock policy available without privileged changes, temperature and other visible sources of variance. Do not change global clocks or shared services. Preflight exact input/workspace sizes and cap total job allocations to the approved memory budget with headroom. A failed preflight is not permission to start another machine.

Warm each case at least 10 iterations, then collect at least 30 independent timed samples, alternating/randomizing baseline and Invar order with a fixed seed. For tiny kernels batch enough repetitions to overcome timer resolution and account for the same launch behavior on both sides. Synchronize the relevant work, not unrelated concurrent streams. Report median, interquartile range and a bootstrap confidence interval or other explicitly stated uncertainty method. CUDA events measure device work; a monotonic host clock includes transfers/host overhead. NVIDIA's [asynchronous execution guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html) describes streams/events, and its [Best Practices Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Best_Practices_Guide.pdf) discusses timing and bandwidth.

Record effective bandwidth, achieved arithmetic throughput, launch count, transfer count, temporary memory and register/shared-memory usage where available. Use profiler evidence to choose a specific correction: uncoalesced access, extra materialization, occupancy loss, excessive synchronization, launch overhead or inadequate arithmetic instructions. A blind increase in tuning budget is not a diagnosis.

## Version-1 engineering targets

For each generated-track family, predeclare representative **large, device-resident** cases and their competent equivalent CUDA baseline. Define `ratio = Invar median / baseline median` (lower is better).

- Every scored large case must pass the [frozen performance gate](#frozen-performance-gate); no geometric mean or family median can hide a failed case.
- If uncertainty overlaps a threshold, collect a bounded additional sample batch under the same protocol. Persistent ambiguity is inconclusive, not pass.
- Automatic placement must stay within the <=10% faster-forced-engine dispatch gate on the frozen dispatch corpus. For cases under 1 ms report absolute overhead, with an initial **100 microsecond** dispatch-overhead target; revise only through an explicit manifest/version decision before further tuning.
- No incorrect result, unsafe lifetime, leaked unbounded workspace, hidden benchmark special case, or silently relaxed numeric contract is acceptable regardless of timing.
- Report cold compilation and memory on every case. Bound variant search to 32 candidates per region by default. An exact per-case compilation time cap is frozen after the first generic toolchain smoke result, before kernel schedule tuning; report a missed cap as a failure rather than omitting it.

If one family fails, report partial capability precisely, for example “generated maps and scans qualified on target X; matrix and segmented targets not met.” Do not claim the whole CUDA-class milestone from a passing geometric mean with excluded families. CPU fallback correctness may pass earlier even while performance selection needs work.

## Hardware and execution policy

This repository preparation used local bounded CPU probes only. No CUDA compiler, device execution, or kernel benchmark was run. Before any hardware job, inspect occupancy, CUDA support, memory and runtime bounds on the target host. Do not launch hardware work merely because this document contains a command or benchmark requirement.

M5 introduces a Rust benchmark command and versioned JSON report with these fields. The issue's acceptance suite must fail on missing required cases or reports. GPU CI being unavailable must leave GPU acceptance pending and actionable, while deterministic CPU tests remain runnable.

## Research implications

Triton's [matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) demonstrates the importance of tiling, memory ordering and tuned configurations for competitive matrix kernels. Invar must infer/select these schedules from high-level computation. Exo-GPU's [paper](https://arxiv.org/abs/2609.16389) is further evidence that modern GPU performance involves explicit scheduling of storage and instructions; it does not establish that automatic discovery is solved. These references motivate acceptance work, not an Invar speed claim.

## Ordinary-source parity

### Requirement and honest claim

Invar aims to be the primary language for robotics, with general automatic GPU acceleration that can compete with handwritten CUDA. General-net execution alone is insufficient acceptance. Existing numeric, bandwidth, library, Metal, correctness, integration and release gates remain unless this section explicitly strengthens them.

The product target is automatic performance parity from ordinary Invar source, without user-written CUDA, fixture-specific rewrites or manual schedules being necessary for acceptance. A finite release corpus establishes measured parity on the stated contracts/devices/distributions, never universal parity with every possible expert implementation of every program. A cost model is not a proof of optimality. All computational forms still require the general execution coverage of `src/general_net/CONTRACT.md`, including the slow/serial cases.

Robotics remains heterogeneous: control, IO, lifecycle, foreign libraries and hardware interfaces have their appropriate CPU/host paths. General GPU support does not make hard real-time deadlines proved or place all robot code on the GPU. Native kernel generation, automatic placement, typed contracts and deterministic integration together support the language goal. Do not interpret this aspiration as authority to replace every external robotics library or add all later robotics roadmap features at once.

### Required compiler path

Ordinary source -> checked typed/ownership IR -> executable interaction-net compilation IR -> verified rewrites/dependence analysis -> program-specific compiled native CPU and CUDA/Metal control, data and worklist code -> residency-aware guarded execution.

1. Preserve a complete checked executable net as the general compilation representation and CPU semantic oracle. Production does not require a net interpreter, universal graph-opcode loop or GPU graph reducer. No template miss removes source-form coverage.
2. Compile dynamic predicates to branches, variable loops to loops, recursion to native calls or checked frames/continuations, and irregular frontiers to bounded program-specific worklists. Static structure and scalar/control state may be specialized, but dynamic input facts remain guards in emitted code.
3. Extract regions using dependence, ownership and effects, with guards for runtime facts. Fuse producers/consumers when lawful, batch similar ready work, and lower dense regions to direct GPU operations. A failed guard before execution selects a compiled alternative; switching after progress transfers the exact live continuation/ownership/effect frontier without replaying visible effects.
4. Select layouts and schedules with coalesced accesses, register/shared-memory tiling, occupancy/resource limits, reduction/scan laws and hardware capabilities. Persistent queues or more elaborate search are added only when profiles show the simpler compiled path misses a gate. Use existing Rust-owned backends, IR and tuning helpers before adding infrastructure; no mandatory new framework, LLVM/MLIR or AI-based optimizer; the exception is a bounded e-graph search beside the net, where the interaction net stays the IR of record and every extracted form is re-certified by the independent checker.
5. Keep data resident across composed robotics operations; eliminate avoidable allocation, transfers and launch boundaries. Prefer ahead-of-time host and device artifacts with precompiled guarded variants for declared targets. Key any optional cached specialization by complete source/IR/contract/shape/device/toolchain facts. Bound compilation/tuning budgets using the existing M4-06 limit unless the user changes it. Report code growth, loading, driver pipeline creation and any PTX/driver JIT separately; a runtime language compiler is not a production prerequisite.
6. Dispatch among compiled native CPU, compiled CUDA/Metal worklists and specialized kernels using measured costs and uncertainty. Explicitly explain serialization, specialization and placement. Forced source-derived compiled-device execution remains a separate coverage check even when automatic placement chooses CPU.

A region replacement must preserve its values, ordered observations, ownership/leases and trap/resource behavior at the boundary. Let a candidate schedule s belong to the checked legal set L(P,d). Choose s minimizing estimated time only over L. If the search space omits the expert schedule or the estimate is wrong, the compiler may lose: diagnose and improve the legal transformations/search rather than describing a cost estimate as a guarantee. In the ideal DAG model, T >= max(W/P,S); memory traffic, launches, allocation and synchronization add real costs. Algebraic reformulations can change work/span only under proved laws with the same numerical contract.

### Comparison tracks and semantics

- Automatic generated-code track: ordinary Invar source -> executable code. No external library doing the measured computation, embedded CUDA or user-supplied schedules in the scored Invar path. Handwritten baseline may use idiomatic CUDA intrinsics and equivalent parallel algorithms, preserving the exact operation's observable/numerical contract.
- Same-algorithm diagnostic track: compare equivalent algorithm/data structure and report compiled worklist, layout and scheduling overhead. This helps root-cause analysis; it is not the only product acceptance track.

The required ablations are B (production net-derived compiled code), D (independently reviewed handwritten CUDA) and native CPU placement controls. A generic GPU graph reducer A is optional diagnostic work and may be reported `not_built`; it is no release prerequisite. Matched-schedule direct structured lowering C is diagnostic where available. Never invent a timing for an unavailable path.
- Best available application track: compare the complete operation/pipeline with an independently optimized handwritten or library-assisted implementation, including its algorithm choice. Pin libraries/source and state precisely which party performs each operation. Calling cuRobo/cuBLAS from Invar does not prove automatic compiler parity; keep that integration track separately labeled.

Use bitwise/full-output agreement for deterministic exact contracts. For admitted approximate/relaxed floating contracts freeze error, stability, convergence, iteration budget and success/failure criteria before timing. No silent reduced precision, FMA/reassociation, dropped validation, weaker collision geometry or fewer planning iterations. A baseline implementing a different contract is informative only, not a scored denominator. Stochastic planning must match seed policy, success rate, solution quality and stopping rules as well as time.

### Frozen performance gate

For every scored workload/input size/device and timing scope, R = median(T_Invar) / median(T_reviewed_baseline). Engineering target and point-estimate gate: R <= 1.00. Require the one-sided upper 95% confidence bound on R <= 1.05. The 5% margin is only statistical tolerance, and the report must say 'matched within the declared 5% uncertainty margin'; exact equality is not proved. 'Faster' requires the upper bound below 1.00. A point estimate above 1.00 fails even if below 1.05. A confidence bound above 1.05 leaves the gate inconclusive/pending. This is a noninferiority bound plus a stricter point-estimate gate, not a two-sided equivalence interval that would reject large wins. No geometric mean, family median or fast case can hide a failed scored case.

Baseline source/algorithm/flags/contracts and test distributions are pinned and reviewed before measurements. Use independently maintained public implementations when compatible, otherwise transparently label a new baseline as author-written and require independent review/profiling. A deliberately weak baseline is not acceptance. If an independently reproduced stronger compatible baseline is found, retain both and use the stronger one; do not select a slower reference to pass. Don't call an unreviewed baseline 'expert CUDA'.

Choose the GPU-profitable size band using baseline CPU/GPU crossover evidence without observing Invar scores; freeze it before scoring. Small or dependency-limited controls remain in the corpus and require correct automatic placement, measured overhead and the existing <=10% faster-forced-engine dispatch gate, with typed exceptional reasons recorded but not counted as parity wins. They are not discarded as 'unsupported'. No semantic feature coverage gap may be excused by the profitable-band rule.

Apply parity to both device-resident complete-operation timing and warm host-input-to-host-output timing for scored GPU cases. End-to-end includes required allocations, generated worklist setup, dispatch, synchronization and transfers under equivalent residency. Baselines receive the same opportunities for pooling, overlap, precomputation and reused inputs. Profile complete pipelines as well as kernels.

Cold-process/JIT/compilation and deployment preparation are distinct scopes: report fresh-process first result, whether binaries are prebuilt, device module loading and any driver/PTX JIT, compilation/tuning cost, amortization count and peak host/device memory. Do not mix one precompiled executable with the other's JIT compilation and call that steady-state parity. Preserve the old cold-process methodology and immutable artifacts; no silent omission of first-use costs.

Retain at least 10 warmups and 30 raw samples per scoped series. Use randomized/interleaved paired measurement blocks, synchronized completed work, correctness outside timed regions with identical mandatory checks inside, and at least three independent process sessions for final parity. Bootstrap/resample independent blocks at their actual independence level; do not treat tightly correlated GPU launches as independent experiments. Freeze the statistical procedure, seeds and bounded follow-up batch policy before timing. Retain every valid run and negative result; don't repeat until a favorable sample appears. Noise that cannot be resolved within the bounded plan stays inconclusive.

Cuda parity is measured on the approved CUDA target; a different GPU is separate evidence. Metal compares Invar with reviewed native Metal/MPS equivalents on the same Apple hardware using the same contracts and new parity definition for new general/robotics cases; never compare absolute Apple latency to NVIDIA CUDA and call that compiler parity. All original Metal gates remain. Hardware availability and compute permissions are not relaxed by this section.

### Benchmark expansion

The benchmark owner freezes exact executable cases and reference contracts before tuning. Required categories span current M4 families plus non-template general programs and compositional robotics. Within each, include at least two materially different compositions/input structures, boundary/small controls, realistic size sweeps and held-out configurations generated by the same public workload definition. Keep structural holdouts separate from tuning; any exposure/use for tuning converts them to training and requires new holdouts under the frozen procedure, retaining failures.

| Category | Required representative operations | Measurements beyond kernel throughput |
| --- | --- | --- |
| General dynamic code | Irregular recursive divide-and-conquer, input-dependent branch trees, dynamically discovered frontiers | Expansion/work distribution, speculation waste, worklist bytes, scheduler overhead |
| Kinematics/dynamics | Batched Panda/UR5e forward transforms, Jacobians and a dynamics operation with independent physical reference | Numerical agreement, single/batched latency, allocation/residency |
| Collision and geometry | Batched distance/collision queries over pinned geometry, sparse candidate traversal | Same geometry/tolerance, false negatives forbidden, irregular density |
| Planning computation | Trajectory rollouts/costs and a bounded search/frontier operation | Same stopping/quality/seed policy; do not relabel a cost kernel as complete planning |
| Perception computation | Point-cloud transform/filter/compaction and neighborhood or reduction composition | Layout/transfer costs and full result agreement |
| Composed robotics operation | Source-written kinematics -> collision -> rollout/cost pipeline, plus integration with checked host boundaries | Warm full-operation latency, throughput, p50/p95/p99, peak memory and jitter |

Rows are concrete v0.1 benchmark obligations, not claims they are implemented. The manifest must identify actual language/math primitives needed (including missing floats or math intrinsics), expected outputs/reference lineage, provenance and prerequisite slices. Source literals used by current CLI smoke fixtures do not satisfy runtime-input/held-out GPU acceptance; use source-derived compiled entry points with real input buffers when the integration interface exists.

Hard real-time/WCET claims remain unproved without their separate later contracts. For v0.1 report measured tail latency, deadline observations where a deadline is already declared, cold spikes and contention conditions. Throughput alone cannot establish control-loop suitability. Do not infer collision safety or physical robot safety from benchmark agreement.

### Primary technical grounding

- HVM2 paper, section 13: graph runtime allocation and evaluation overhead, https://github.com/HigherOrderCO/HVM2/blob/main/paper/HVM2.pdf .
- Triton documentation: blocked computation and compiled GPU layout/scheduling as a productive performance route, https://triton-lang.org/main/programming-guide/chapter-1/introduction.html . This supports the approach, not general Invar parity or an instruction to depend on Triton.
- NVIDIA CUDA Programming Guide: memory coalescing and SIMT execution, https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html .
