# v0.1 amendment: general execution with handwritten-performance acceptance

Date: 2026-09-28. Scope amendment; implementation and acceptance pending.

## Requirement and honest claim

Invar aims to be the primary language for robotics, with general automatic GPU acceleration that can compete with handwritten CUDA. Add this to v0.1 now. General-net execution alone is insufficient acceptance. The old section-10 handwritten ratio ceiling of 1.25 is superseded for the CUDA comparisons by the stricter gate below. Existing numeric, bandwidth, library, Metal, correctness, integration and release gates remain unless this document explicitly strengthens them.

The product target is automatic performance parity from ordinary Invar source, without user-written CUDA, fixture-specific rewrites or manual schedules being necessary for acceptance. A finite release corpus establishes measured parity on the stated contracts/devices/distributions, never universal parity with every possible expert implementation of every program. A cost model is not a proof of optimality. All computational forms still require the general execution coverage in general-gpu-v01.md, including the slow/serial cases.

Robotics remains heterogeneous: control, IO, lifecycle, foreign libraries and hardware interfaces have their appropriate CPU/host paths. General GPU support does not make hard real-time deadlines proved or place all robot code on the GPU. Native kernel generation, automatic placement, typed contracts and deterministic integration together support the language goal. Do not interpret this aspiration as authority to replace every external robotics library or add all later robotics roadmap features at once.

## Required compiler path

Ordinary source -> checked typed/ownership IR -> executable interaction-net compilation IR -> verified rewrites/dependence analysis -> program-specific compiled native CPU and CUDA/Metal control, data and worklist code -> residency-aware guarded execution.

1. Preserve a complete checked executable net as the general compilation representation and CPU semantic oracle. Production does not require a net interpreter, universal graph-opcode loop or GPU graph reducer. No template miss removes source-form coverage.
2. Compile dynamic predicates to branches, variable loops to loops, recursion to native calls or checked frames/continuations, and irregular frontiers to bounded program-specific worklists. Static structure and scalar/control state may be specialized, but dynamic input facts remain guards in emitted code.
3. Extract regions using dependence, ownership and effects, with guards for runtime facts. Fuse producers/consumers when lawful, batch similar ready work, and lower dense regions to direct GPU operations. A failed guard before execution selects a compiled alternative; switching after progress transfers the exact live continuation/ownership/effect frontier without replaying visible effects.
4. Select layouts and schedules with coalesced accesses, register/shared-memory tiling, occupancy/resource limits, reduction/scan laws and hardware capabilities. Persistent queues or more elaborate search are added only when profiles show the simpler compiled path misses a gate. Use existing Rust-owned backends, IR and tuning helpers before adding infrastructure; no mandatory new framework, LLVM/MLIR or AI-based optimizer; the exception is a bounded e-graph search beside the net ([net-parallel-laws-v01](design/net-parallel-laws-v01.md) section 6.4, decisions E1 and M1), where the interaction net stays the IR of record and every extracted form is re-certified by the independent checker.
5. Keep data resident across composed robotics operations; eliminate avoidable allocation, transfers and launch boundaries. Prefer ahead-of-time host and device artifacts with precompiled guarded variants for declared targets. Key any optional cached specialization by complete source/IR/contract/shape/device/toolchain facts. Bound compilation/tuning budgets using the existing M4-06 limit unless the user changes it. Report code growth, loading, driver pipeline creation and any PTX/driver JIT separately; a runtime language compiler is not a production prerequisite.
6. Dispatch among compiled native CPU, compiled CUDA/Metal worklists and specialized kernels using measured costs and uncertainty. Explicitly explain serialization, specialization and placement. Forced source-derived compiled-device execution remains a separate coverage check even when automatic placement chooses CPU.

A region replacement must preserve its values, ordered observations, ownership/leases and trap/resource behavior at the boundary. Let a candidate schedule s belong to the checked legal set L(P,d). Choose s minimizing estimated time only over L. If the search space omits the expert schedule or the estimate is wrong, the compiler may lose: diagnose and improve the legal transformations/search rather than describing a cost estimate as a guarantee. In the ideal DAG model, T >= max(W/P,S); memory traffic, launches, allocation and synchronization add real costs. Algebraic reformulations can change work/span only under proved laws with the same numerical contract.

## Comparison tracks and semantics

- Automatic generated-code track: ordinary Invar source -> executable code. No external library doing the measured computation, embedded CUDA or user-supplied schedules in the scored Invar path. Handwritten baseline may use idiomatic CUDA intrinsics and equivalent parallel algorithms, preserving the exact operation's observable/numerical contract.
- Same-algorithm diagnostic track: compare equivalent algorithm/data structure and report compiled worklist, layout and scheduling overhead. This helps root-cause analysis; it is not the only product acceptance track.

The required ablations are B (production net-derived compiled code), D (independently reviewed handwritten CUDA) and native CPU placement controls. A generic GPU graph reducer A is optional diagnostic work and may be reported `not_built`; it is no release prerequisite. Matched-schedule direct structured lowering C is diagnostic where available. Never invent a timing for an unavailable path.
- Best available application track: compare the complete operation/pipeline with an independently optimized handwritten or library-assisted implementation, including its algorithm choice. Pin libraries/source and state precisely which party performs each operation. Calling cuRobo/cuBLAS from Invar does not prove automatic compiler parity; keep that integration track separately labeled.

Use bitwise/full-output agreement for deterministic exact contracts. For admitted approximate/relaxed floating contracts freeze error, stability, convergence, iteration budget and success/failure criteria before timing. No silent reduced precision, FMA/reassociation, dropped validation, weaker collision geometry or fewer planning iterations. A baseline implementing a different contract is informative only, not a scored denominator. Stochastic planning must match seed policy, success rate, solution quality and stopping rules as well as time.

## Frozen performance gate

For every scored workload/input size/device and timing scope, R = median(T_Invar) / median(T_reviewed_baseline). Engineering target and point-estimate gate: R <= 1.00. Require the one-sided upper 95% confidence bound on R <= 1.05. The 5% margin is only statistical tolerance, and the report must say 'matched within the declared 5% uncertainty margin'; exact equality is not proved. 'Faster' requires the upper bound below 1.00. A point estimate above 1.00 fails even if below 1.05. A confidence bound above 1.05 leaves the gate inconclusive/pending. This is a noninferiority bound plus a stricter point-estimate gate, not a two-sided equivalence interval that would reject large wins. No geometric mean, family median or fast case can hide a failed scored case.

Baseline source/algorithm/flags/contracts and test distributions are pinned and reviewed before measurements. Use independently maintained public implementations when compatible, otherwise transparently label a new baseline as author-written and require independent review/profiling. A deliberately weak baseline is not acceptance. If an independently reproduced stronger compatible baseline is found, retain both and use the stronger one; do not select a slower reference to pass. Don't call an unreviewed baseline 'expert CUDA'.

Choose the GPU-profitable size band using baseline CPU/GPU crossover evidence without observing Invar scores; freeze it before scoring. Small or dependency-limited controls remain in the corpus and require correct automatic placement, measured overhead and the existing <=10% faster-forced-engine dispatch gate, with typed exceptional reasons recorded but not counted as parity wins. They are not discarded as 'unsupported'. No semantic feature coverage gap may be excused by the profitable-band rule.

Apply parity to both device-resident complete-operation timing and warm host-input-to-host-output timing for scored GPU cases. End-to-end includes required allocations, generated worklist setup, dispatch, synchronization and transfers under equivalent residency. Baselines receive the same opportunities for pooling, overlap, precomputation and reused inputs. Profile complete pipelines as well as kernels.

Cold-process/JIT/compilation and deployment preparation are distinct scopes: report fresh-process first result, whether binaries are prebuilt, device module loading and any driver/PTX JIT, compilation/tuning cost, amortization count and peak host/device memory. Do not mix one precompiled executable with the other's JIT compilation and call that steady-state parity. Preserve the old cold-process methodology and immutable artifacts; no silent omission of first-use costs.

Retain at least 10 warmups and 30 raw samples per scoped series. Use randomized/interleaved paired measurement blocks, synchronized completed work, correctness outside timed regions with identical mandatory checks inside, and at least three independent process sessions for final parity. Bootstrap/resample independent blocks at their actual independence level; do not treat tightly correlated GPU launches as independent experiments. Freeze the statistical procedure, seeds and bounded follow-up batch policy before timing. Retain every valid run and negative result; don't repeat until a favorable sample appears. Noise that cannot be resolved within the bounded plan stays inconclusive.

Cuda parity is measured on the approved CUDA target; a different GPU is separate evidence. Metal compares Invar with reviewed native Metal/MPS equivalents on the same Apple hardware using the same contracts and new parity definition for new general/robotics cases; never compare absolute Apple latency to NVIDIA CUDA and call that compiler parity. All original Metal gates remain. Hardware availability and compute permissions are not relaxed by this amendment.

## Benchmark expansion

The benchmark owner freezes exact executable cases and reference contracts before tuning. Required categories span current M4 families plus non-template general programs and compositional robotics. Within each, include at least two materially different compositions/input structures, boundary/small controls, realistic size sweeps and held-out configurations generated by the same public workload definition. Keep structural holdouts separate from tuning; any exposure/use for tuning converts them to training and requires new holdouts under the frozen procedure, retaining failures.

| Category | Required representative operations | Measurements beyond kernel throughput |
| --- | --- | --- |
| General dynamic code | Irregular recursive divide-and-conquer, input-dependent branch trees, dynamically discovered frontiers | Expansion/work distribution, speculation waste, worklist bytes, scheduler overhead |
| Kinematics/dynamics | Batched Panda/UR5e forward transforms, Jacobians and a dynamics operation with independent physical reference | Numerical agreement, single/batched latency, allocation/residency |
| Collision and geometry | Batched distance/collision queries over pinned geometry, sparse candidate traversal | Same geometry/tolerance, false negatives forbidden, irregular density |
| Planning computation | Trajectory rollouts/costs and a bounded search/frontier operation | Same stopping/quality/seed policy; do not relabel a cost kernel as complete planning |
| Perception computation | Point-cloud transform/filter/compaction and neighborhood or reduction composition | Layout/transfer costs and full result agreement |
| Composed robotics operation | Source-written kinematics -> collision -> rollout/cost pipeline, plus integration with checked host boundaries | Warm full-operation latency, throughput, p50/p95/p99, peak memory and jitter |

Rows are concrete v0.1 benchmark obligations, not claims they are implemented. The manifest must identify actual language/math primitives needed (including missing floats or math intrinsics), expected outputs/reference lineage, provenance and prerequisite slices. A source form supported by later v0.1 must be a pending prerequisite, not permanently excluded because bootstrap.md is narrower. Source literals used by current CLI smoke fixtures do not satisfy runtime-input/held-out GPU acceptance; use source-derived compiled entry points with real input buffers when the integration interface exists.

Hard real-time/WCET claims remain unproved without their separate later contracts. For v0.1 report measured tail latency, deadline observations where a deadline is already declared, cold spikes and contention conditions. Throughput alone cannot establish control-loop suitability. Do not infer collision safety or physical robot safety from benchmark agreement.

## Primary technical grounding

- HVM2 paper, section 13: graph runtime allocation and evaluation overhead, https://github.com/HigherOrderCO/HVM2/blob/main/paper/HVM2.pdf .
- Triton documentation: blocked computation and compiled GPU layout/scheduling as a productive performance route, https://triton-lang.org/main/programming-guide/chapter-1/introduction.html . This supports the approach, not general Invar parity or an instruction to depend on Triton.
- NVIDIA CUDA Programming Guide: memory coalescing and SIMT execution, https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html .
