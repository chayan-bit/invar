# v0.2 `fsum`: deterministic parallel float reductions

Status: draft, branch `feat/v02-fsum`.
Normative semantics live in [bootstrap.md](../../spec/bootstrap.md) ("v0.2 deterministic float reductions"); this document records the survey, the decisions and how each engine meets the contract.

## Goal

Robotics code (state estimation, kinematics, control) sums and dots float arrays constantly.
Invar never reassociates a source fold, so a `for` loop that folds `f32` stays serial and in source order.
This workstream adds explicit, opt-in reductions whose result bits are identical on `reference`, `native-serial`, `cpu-parallel` (any worker count), `cuda` and `metal`:

| Intrinsic | Result | Cost |
| --- | --- | --- |
| `sum_tree(xs)`, `dot_tree(a, b)` | fixed pairwise tree, shape depends only on the length | about one add per element |
| `sum_exact(xs)`, `dot_exact(a, b)` | exact real sum, rounded once to nearest-even | integer accumulation |

## Survey (done before implementation)

| Candidate | What it is | License | Maintenance | Verdict |
| --- | --- | --- | --- | --- |
| `xsum` crate 0.1.6 (Gumichocopengin8/xsum.rs), port of Neal's xsum | exact f64 summation, small and large superaccumulators | MIT (Neal 2015-2024, Nonaka 2025) | single maintainer, 2 stars, last push 2025-11 | Not used as a dependency: f64 only, its final rounding is to f64, so an f32 result would round twice; no dot; host floating-point in the rounding path; no GPU form. Design input: chunked integer accumulator with lazy carry propagation. |
| `accurate` 0.4.1 | Kahan, Neumaier, OnlineExactSum, iFastSum | MIT/Apache-2.0 | last release 2024-07 | Not used: f64-oriented, host-FP error-free transforms, no device story, not bit-identical across our engines by construction. |
| `fsum` 0.1.4 (bsuccinct-rs) | Shewchuk/Python `math.fsum` partials | MIT/Apache-2.0 | active | Not used: partial-list algorithm is sequential and data dependent; poor GPU fit. |
| `twofloat` 0.8.4 | double-double arithmetic | BSD-3-Clause | 2025-07 | Not used: not exact for arbitrary dynamic range. |
| `rug` 1.30 (GMP/MPFR) | arbitrary precision | LGPL-3.0+ | active | Not used even as a dev-dependency: heavy C build; `num-bigint`/`num-rational` are already in `Cargo.lock` and suffice for the test oracle. |
| ExBLAS (riakymch/exblas; Collange, Defour, Graillat, Iakymchuk 2015) | Kulisch superaccumulator plus floating-point-expansion cache, CPU/GPU | BSD-style "Software Development License" (KTH, UPMC, INRIA, UPVD) | last push 2021 | Design reference, not vendored: its GPU path is OpenCL f64 with shared-memory atomics; we need f32 now, CUDA NVRTC and Metal, and a host-side single rounding. Expected exact-sum cost of 1.5-3x a plain reduction comes from here. |
| ReproBLAS (Demmel, Ahrens, Nguyen) | binned reproducible sums | BSD-style | active | Not used: reproducible but not exact, and not equal to any CPU order. |
| CUB `DeviceReduce` with `cuda::execution::determinism::gpu_to_gpu` (CCCL) | RFA reproducible reduce | Apache-2.0 with LLVM exception | active | Not usable for bit identity with CPU engines; it is the CUDA speed baseline only. |
| Malcolm 1971, "On accurate floating-point summation" | exponent-indexed integer bins | algorithm | n/a | Ported for the CPU exact path: binary32 has only 254 exponents, so one i64 bin per exponent never overflows for u32 lengths. |
| Kulisch long accumulator / Neal small superaccumulator | fixed-point accumulator in 32-bit digits held in 64-bit words | algorithm (Neal: MIT) | n/a | Ported for the device exact path: per-thread digits, lazy carries, host merge. |

Pick: net-new code that ports two proven algorithms (Malcolm bins on CPU, Kulisch/Neal digits on devices) plus one integer-only rounding routine.
Every candidate crate lost on at least one of: f32 support, single rounding to f32, dot products, GPU form, or bit identity with our CPU engines.
The tree is a fixed specification, not an algorithm choice, so no library applies.

## Decisions

### Tree shape

Leaf block 8 summed left to right from its first element, then the aligned pairwise tree described in the spec (rulings C11, C12).
The key property: node `(L, j)` depends only on leaves `[j*2^L, (j+1)*2^L)`, and a missing right child carries its left sibling.
Hence any aligned chunk of `2^c` leaves reduces independently and the chunk roots combine by the same rule.
Three formulations are tested against each other: the reference reduces level by level with the rational oracle, the engine code uses a sequential stack (`fsum::fold_aligned`), and the test oracle splits top-down at the largest power of two.
Commutativity of IEEE addition makes the operand order inside a pair irrelevant, so devices may use shuffles in either direction.

`dot_tree` rounds each product and then uses the same tree; no FMA, so every engine can reproduce it with plain multiply and add.

### Exact sums

A finite binary32 value is `s * 2^(p - 149)` with `p` in `[0, 253]` and `s < 2^24`.
The CPU path adds `+-s` into `bins[p]` (Malcolm); `2^39` deposits per bin cannot overflow an i64.
A dot product of finite values is `sa * sb * 2^(pa + pb - 298)`; the 48-bit product splits into two 24-bit halves at bins `pa+pb+24` and `pa+pb`.
Normalization folds bins into 19 little-endian 32-bit digits (608 bits) through an i128 carry; the final carry is the sign.
Rounding uses integer bit tests only: keep 24 significant bits or the subnormal quantum, round half to even with a sticky bit, renormalize a carry-out, and overflow to infinity.
Special values come from four flags (NaN, +inf, -inf, "some term is not -0") that merge by OR, so any partition merges to the same result.

### Engines

| Engine | sum_tree / dot_tree | sum_exact / dot_exact |
| --- | --- | --- |
| reference | `fsum_reference`: level-by-level tree on the rational binary32 oracle | one `BigInt` sum, the oracle's single rounding |
| MIR (serial) | `fsum::fold_aligned` on host binary32 | Malcolm bins |
| native-serial | host runtime `invar_fsum_v1`, the same `fsum.rs` compiled into the generated host | same |
| cpu-parallel | aligned leaf chunks on scoped threads (`fsum::chunk_level`), eight leaves formed together as independent chains; one worker below the measured crossover | contiguous ranges, merged bins; one worker below the crossover |
| metal, cuda | SIMD-step aligned tree per group (below), host folds the group nodes | per-thread digit accumulators, group digit sums, host merges in i128 and rounds |

Host binary32 addition is IEEE round-to-nearest-even with gradual underflow on the supported hosts (aarch64, x86-64 SSE); Rust never contracts or reassociates.
The MIR interpreter gained one hook, `Reducer`, consulted when a reached call is evaluated; the hook may return bits or decline, and the length check and trap stay in MIR.
Fuel is engine independent: one step for the call plus one per operand.

### Device kernels

Both backends share one launch description (`invar_device::fsum::Launch`) and fixed sources (`msl()`, `cuda()`), so a kernel never depends on the program.
Tree kernel: 256 lanes per group, eight 32-lane SIMD groups.
SIMD group `g` owns the aligned run of `32 << t` leaves; in each of its `2^t` steps lane `l` folds one leaf of 8 elements (two aligned 16-byte loads, so 32 lanes read one contiguous kilobyte), then the lanes combine as an aligned shuffle tree with the existence rule.
The steps combine as an aligned tree, then the eight SIMD nodes, so each group writes the node at level `8 + t`; the host folds the group nodes with `fold_aligned`.
The first version gave each thread `2^t` adjacent leaves; its strided loads cost 1.35-2.5x the baseline, and the SIMD-step layout removed that while keeping the shape.
`t` is a launch parameter (0 to 3) and every value gives the same bits; the scheduler uses `t = 2`.
Exact kernel: a grid-stride loop deposits each term (or the exact 48-bit product, split in 24-bit halves) into 19 signed 64-bit digits of 32 bits; each group sums every digit in shared memory and writes one row of digits plus an OR of the flag words.
The host merges rows in i128 (1024 groups of at most `2^54` per digit cannot overflow) and rounds once with the same integer routine as the CPU.
CUDA compiles through the pinned NVRTC helper; the kernels use `__fadd_rn`/`__fmul_rn`, which never fuse, and NVRTC keeps IEEE subnormals by default.
Metal compiles with fast math off and `#pragma METAL fp contract(off)`.
`invar build --target metal` adds the fsum library (both kernels) to the compiled-ahead image of any program that calls one of the four reductions, so `invar run --artifact` never compiles fsum MSL at run time.

### Metal subnormals

The Apple GPU used here (M-series, Apple9 family) flushes binary32 subnormal inputs and results to zero even with fast math off; the first device run proved it (`dot_tree` of one subnormal returned `+0`).
The tree kernel therefore guards every operation whose IEEE result could differ: a subnormal input word, or a zero result from operands that do not cancel exactly (`r == 0 && x != -y` for sums, `r == 0 && x != 0 && y != 0` for products).
Once a subnormal is flushed, every later value is normal or zero, so these two checks cover all divergence.
A fired guard discards the device words (`SubnormalFallback`) and the CPU workers produce the result; the receipt says `metal-failed-before-publication`.
The exact kernel is integer-only and needs no guard.
This follows the M4-01 underflow-guard precedent instead of software-emulating gradual underflow on the GPU.

### Explain hint

When a loop body stores `acc + e` into the `f32` local `acc`, `invar explain` adds the obligation `FLOAT_FOLD_HINT` (evidence `hint`) to the rejected region, naming `sum_tree` and `sum_exact`.
It lives in the region's obligations so the explain schema needs no bump, and the code is registered in `schema/diagnostic-codes-fsum-v1.json` (ruling C14).
It is a hint only: the loop keeps its serial plan.
Only the direct body block is scanned; a fold under a nested `if` gets no hint yet.

### Placement below the crossovers

The tree shape depends only on the length, so the worker count and the device are free choices that never change the bits; the cost model routes losing shapes and never relaxes a gate.
CPU: `reductions::cpu_workers` splits a call over W workers only when `spawn_ns_per_worker * W + ns_per_term * n / W < ns_per_term * n`, else it runs on one worker.
Every engine that reaches the CPU uses it (cpu-parallel, Metal and CUDA fallbacks, `auto`); the receipt records the chosen worker count, and a call kept on one of several workers reports backend `cpu-serial` with reason `measured-cpu-cost`.
The `invar-explain-device-v5` schema has a closed reason enum and no worker field, so the refusal is carried by the backend name rather than a new reason; `kind` carries the family id.
Metal: `auto` places a call on Metal only from `metal_min_terms`.
End to end (upload into a new shared buffer, kernel, host finish), Metal on an Apple M4 lost to the better of one and ten CPU workers at every size from 2^10 to 2^26 for all four families (for example 34.95 ms against 6.09 ms for `sum_tree` at 2^26, where the resident kernel takes 2.6 ms), so the recorded crossover is `null`: never.
The copy into a fresh buffer, not the kernel, is what loses; a zero-copy upload would move the crossover and must be re-measured.
The coefficients live in `src/schedule/reductions.rs` and are recorded with their raw measurements under each family id in `calibration/fsum-v1.json`; a unit test keeps the two equal.
They are provisional: Apple M4 rows (load average 4.1) are measured, and the Linux CUDA-host rows are pending a quiet window.
On the M4 the model's CPU crossovers match the measured ones (2^18 for the trees, 2^16 for `dot_exact`) except `sum_exact`, where it splits from 2^15 and the measurement favours one worker until 2^16.

Metal 2^20 follow-up: two kernel variants were measured and dropped.
Issuing every step's leaf loads before any shuffle gave no gain (1.66x at shift 2).
One leaf per lane with the aligned combine in threadgroup memory (shift 0, 512 groups) gave 1.55x against the shipped shift-2 kernel's 1.58x in the same session, both bit-identical.
The 1.10x target is therefore missed at 2^20, and the placement above routes every size to the CPU end to end, so no `invar run` call pays it.

### Generic f64 shape (design now, implement after `f64math`)

- Checker: `float_element` in `src/check/reductions.rs` gains `Array<f64> -> f64`; spellings stay overloaded by argument type (ruling C15).
- Tree: unchanged shape and leaf block; `fsum::fold_aligned` becomes generic over an `Add` float.
- Exact CPU: Malcolm bins do not scale (2046 exponents and 53-bit significands, so 2^10 deposits per bin before overflow); use Neal's small superaccumulator (67 chunks of 32 bits in i64 with lazy carries) for sums, and split 106-bit exact products into three 53-bit pieces for dots.
- Exact devices: per-thread accumulators are too large for f64 (about 70 digits for sums, 140 for dots); use per-threadgroup digit arrays in shared memory with 64-bit atomics on CUDA, and 32-bit atomics on radix-2^24 digits on Metal (no 64-bit atomic add in MSL 3.1).
- Rounding: `fsum::round` takes the format's significand width and minimum exponent instead of the binary32 constants.

## Testing and acceptance

- `tests/v02_fsum_oracle.rs`: fast code and reference against an independent oracle (top-down tree on host binary32; exact `BigRational` sum rounded by binary search over encodings) on adversarial vectors (cancellation, huge dynamic range, subnormals, ties, overflow, infinities, NaN payloads, signed zeros, n = 0 and 1) and 300 random lengths; order independence of exact sums; chunk invariance at every aligned level.
- `tests/v02_fsum_oracle.rs` also runs every vector on exactly 2, 3 and 8 CPU workers (`reductions::cpu_on`), because the cost model now keeps small CLI cases on one worker.
- `tests/v02_fsum_cli.rs`: generated source with exact decimal literals runs through the real `invar` binary on every engine in the build (reference, cpu-parallel at 1 to 4 workers, native-serial, metal, cuda, auto) and must return the oracle bits for every case; dot length mismatch traps at the call span everywhere; checker and ownership rejections; reference and MIR agree on value, fuel at every prefix and cleanup.
- `tests/v02_fsum_metal.rs` and `tests/v02_fsum_cuda.rs`: the device partials, finished by the host, equal the CPU bits for every adversarial vector, for sizes 1 to 3,000,017 across group boundaries and leaf shifts 0 to 3, and for subnormal and contraction traps (a product such as `(1 + 2^-23)^2` whose fused multiply-add would round differently); the Metal suite also checks every placement receipt and the explain hint.
- Mutations, each applied to the source, run, and restored; each was caught by a failing test (the first failing test binary stops the run):

| Mutation | Meaning | Caught by |
| --- | --- | --- |
| `cpu-leaf-4` | leaf block 4 instead of 8 (wrong tree shape) | oracle |
| `cpu-leaf-zero-seed` | a leaf starts from `+0` (ruling C11) | CLI |
| `cpu-block-zero-seed` | the same, in the eight-leaf CPU block | CLI |
| `cpu-drop-limb-4` | the exact accumulator drops bin 4 | CLI |
| `metal-thread-serial-fold`, `cuda-thread-serial-fold` | step nodes folded serially instead of as an aligned tree | device group-boundary tests |
| `metal-drop-limb-4`, `cuda-drop-limb-4` | the group drops exact digit 4 | every device test |

## Benchmarks

Medians of 30 timed runs after 10 warmups (5 on the CPU), random finite binary32 inputs, ruling C13 (bench harnesses, not `invar run`).
Device numbers are resident kernel time (inputs already on the device; the host fold of at most a few thousand partials is excluded); `sum_tree` uses leaf shift 2.

**Status of these timings: unqualified.**
The Metal and CUDA rows below were measured without a host load receipt (the Mac load was not recorded; the CUDA host ran other jobs on a shared GPU), so under the load-under-8 rule they are history, not acceptance evidence.
They are kept because the bits and the kernel design they informed are unchanged; the 1.10x acceptance is pending a quiet-host rerun with the load printed beside every row.

Metal (Apple M-series GPU, 5 interleaved rounds, median round, host load not recorded), against a handwritten unordered `float4` + `simd_sum` reduction in the same harness:

| n | baseline ms | `sum_tree` ms | ratio | `sum_exact` ms | ratio |
| --- | --- | --- | --- | --- | --- |
| 2^20 | 0.061 | 0.096 | 1.57 | 0.332 | 5.40 |
| 2^24 | 0.730 | 0.750 | 1.03 | 1.442 | 1.93 |
| 2^26 | 2.595 | 2.609 | 1.01 | 3.840 | 1.32 |

CUDA (RTX 5070 Ti, shared GPU, host load not recorded), against `cub::DeviceReduce::Sum` (`bench/baselines/cuda-cub-f32sum.cu`, CUDA 12.4 compiled to `compute_90` PTX and JIT-compiled for `sm_120`; mean of the runs before and after ours):

| n | CUB ms | `sum_tree` ms | ratio | `sum_exact` ms | ratio |
| --- | --- | --- | --- | --- | --- |
| 2^20 | 0.0100 | 0.0085 | 0.85 | 0.0529 | 5.3 |
| 2^24 | 0.0913 | 0.0908 | 0.99 | 0.528 | 5.8 |
| 2^26 | 0.328 | 0.329 | 1.00 | 1.984 | 6.0 |

CPU against rayon `par_iter().sum()`: the earlier runs on a shared host were taken at load average 35 to 60 and do not count (rule: load under 8, recorded beside each number).
The rerun on a 32-core Linux host (`bench/fsum-cpu bench 4` and `bench 16`, every line printing its load average) is pending a quiet window: its load stayed between 12 and 28 on 18 October.

In these unqualified runs `sum_tree` was within 1.10x of the baseline on CUDA at every size and on Metal from 2^24; acceptance waits for the quiet-host rerun.
`sum_exact` costs 1.3x to 6x a plain sum: about 20 integer operations per element against one add.
On Metal it approaches the bandwidth bound at 2^26; on CUDA the 19-digit per-thread accumulator spills and stays at 6x.

## Ruling impacts

- C11: leaves start from their first element on every engine (CPU block path included); `sum_exact` keeps `-0` only for all-`-0` input; both have mutations.
- C12: the leaf block is 8 for `dot_tree` too; the device kernels hard-code it and the shape tests run dot vectors.
- C13: large sizes run only in `v02_fsum_metal_bench`, `v02_fsum_cuda_bench` and `bench/fsum-cpu`; `invar run` tests stay below a few thousand elements.
- C14: receipts carry `family` = `fsum.<op>.<type>`, explain choices carry it as `kind`, and the CPU and Metal crossovers are recorded under it in `calibration/fsum-v1.json`; `FLOAT_FOLD_HINT` is registered in `schema/diagnostic-codes-fsum-v1.json`. Kernel template placeholders were renamed to lowercase so the AI-01 literal scan does not mistake them for codes.
- C15: `sum_tree`, `dot_tree`, `sum_exact` and `dot_exact` are operations, overloaded by argument type, with no type suffix; no change was needed.

## Calibration and receipts

Calibration family ids are `fsum.<op>.<type>`, for example `fsum.sum_tree.f32` (ruling C14).
Run receipts (`schedule::reductions::Reduction`) record function, span, intrinsic, family, element count, backend, placement reason, the worker count the cost model chose, cost estimate and device time.
The placement coefficients and their measurements are in `calibration/fsum-v1.json` (see Placement).

## Open risks

- The CPU coefficients are provisional (Apple M4 only); one conservative set serves every host until the Linux rows land, and a host with cheaper thread start-up splits later than it could.
- Metal never wins end to end on the M4 because each call copies its operands into a new buffer; resident or zero-copy inputs would change that and need a new crossover.
- The CUDA end-to-end crossover is not measured yet; CUDA `auto` still uses the live bandwidth model only.
- Metal `sum_tree` kernels miss 1.10x at 2^20 (1.55x to 1.58x); no `invar run` placement reaches them.
- cpu-parallel spawns scoped threads per call; the cost model avoids the losing sizes, and the engine-wide persistent pool is a separate Phase B workstream.
- Metal falls back to the CPU whenever a subnormal appears in a tree reduction; inputs dense in subnormals never run on the GPU.
- The explain hint only scans the loop's direct body block; folds under a nested `if` get no hint.
- f64 is designed above but not implemented; it waits for the `f64math` workstream.
