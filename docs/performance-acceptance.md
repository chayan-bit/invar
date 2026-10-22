# CUDA-class acceptance protocol

For v0.1, the later [ordinary-source parity contract](cuda-parity-v01.md) strengthens this protocol: every scored case, size, device and both resident/warm timing scopes needs median Invar/reviewed handwritten ratio at most 1.00 and one-sided 95% upper bound at most 1.05. Its general dynamic and robotics corpus and independent-session statistics are required. The older geometric-mean and 1.25 handwritten targets below are historical weaker thresholds where they conflict; existing numeric, library, bandwidth, dispatch, Metal and cold-process gates still apply. [General executable-net coverage](general-gpu-v01.md) is a separate prerequisite and cannot be inferred from fast templates.

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

- Target geometric mean ratio at most **1.20** across the frozen large cases, and at most **1.50** for every included large case. These are aspirational acceptance gates selected for this project, not observed results or literature-derived guarantees.
- If uncertainty overlaps a threshold, collect a bounded additional sample batch under the same protocol. Persistent ambiguity is inconclusive, not pass.
- The fully automatic host-to-host path must be within **1.25** of the fastest admitted measured CPU/GPU choice on the frozen dispatch corpus for cases lasting at least 1 ms. For smaller cases report absolute overhead, with an initial **100 microsecond** dispatch-overhead target; revise only through an explicit manifest/version decision before further tuning.
- No incorrect result, unsafe lifetime, leaked unbounded workspace, hidden benchmark special case, or silently relaxed numeric contract is acceptable regardless of timing.
- Report cold compilation and memory on every case. Bound variant search to 32 candidates per region by default. An exact per-case compilation time cap is frozen after the first generic toolchain smoke result, before kernel schedule tuning; report a missed cap as a failure rather than omitting it.

If one family fails, report partial capability precisely, for example “generated maps and scans qualified on target X; matrix and segmented targets not met.” Do not claim the whole CUDA-class milestone from a passing geometric mean with excluded families. CPU fallback correctness may pass earlier even while performance selection needs work.

## Hardware and execution policy

This repository preparation used local bounded CPU probes only. No CUDA compiler, device execution, or kernel benchmark was run. Before any hardware job, inspect occupancy, CUDA support, memory and runtime bounds on the target host. Do not launch hardware work merely because this document contains a command or benchmark requirement.

M5 introduces a Rust benchmark command and versioned JSON report with these fields. The issue's acceptance suite must fail on missing required cases or reports. GPU CI being unavailable must leave GPU acceptance pending and actionable, while deterministic CPU tests remain runnable.

## Research implications

Triton's [matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) demonstrates the importance of tiling, memory ordering and tuned configurations for competitive matrix kernels. Invar must infer/select these schedules from high-level computation. Exo-GPU's [paper](https://arxiv.org/abs/2609.16389) is further evidence that modern GPU performance involves explicit scheduling of storage and instructions; it does not establish that automatic discovery is solved. These references motivate acceptance work, not an Invar speed claim.
