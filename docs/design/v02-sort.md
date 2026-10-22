# Explicit stable sorting

Status: source surface frozen in `3af9743d`; CPU lowering in `686b6941`,
CUDA/Metal lowering in `2fde7feb`.
Reference, MIR, native CPU, joined CPU workers and both physical GPUs have
passing focused contracts. Extended acceptance is tracked in the sort test suites.

## Required surface

These are reserved builtin calls using ordinary call syntax, with implicit call-scoped array loans.

| Operation | Loans | Result |
| --- | --- | --- |
| `sort(xs)` | `mut xs: Array<T>` | `Unit` |
| `sort_by_key(keys, values)` | `mut keys: Array<T>, mut values: Array<U>` | `Unit` |
| `argsort(xs)` | `read xs: Array<T>` | owned `Array<u32>` |
| `segmented_sort(xs, offsets)` | `mut xs: Array<T>, read offsets: Array<u32>` | `Unit` |

`T` is exactly `u32`, `i32`, or `f32`.
`U` is any scalar element type; do not silently narrow this to the three existing numeric array variants.
All operations are stable and preserve input order within equal-key groups.
All floating keys are canonicalized to `0x7fc00000` when NaN, then compared using IEEE 754 totalOrder.
The resulting order is negative infinity, negative finite values, negative zero, positive zero, positive finite values, positive infinity, NaN.
An argsort does not mutate its input, including original NaN encodings.
No user loop is automatically replaced with sorting.

Length mismatch in `sort_by_key` traps `INDEX_OUT_OF_BOUNDS`.
Offsets are validated in increasing index order; the first descending or out-of-range offset traps `INDEX_OUT_OF_BOUNDS`.
Validation must finish before any sorting writes, and no invalid offset may reach host indexing or a device dispatch.
Offsets follow the binding C10/M4-04 convention: `segments + 1` entries,
first 0, last `len(xs)`, non-decreasing. There must be at least one offset;
`[0]` describes zero segments only for an empty array. Duplicate boundaries
are empty segments. Full validation precedes writes and dispatch. Empty lists,
partial coverage, descending and out-of-range entries trap `INDEX_OUT_OF_BOUNDS`.
This supersedes the initial frozen partial-boundary convention.

`sort_by_key` is a key/value **pair sort**, with no closure form (C16).
f64 is a follow-up after f32 lands, and i64 follows B3 (C5).
Calibration families are `sort.<op>.<type>`; for example
`sort.sort_by_key.f32`. Sort explain v3 adds this typed id while preserving
frozen v1/v2 schemas; `sort-cost-unmeasured` is registered in AI-01 (C14).
Large sizes use the bench binaries, never `invar run` (C13).
Physical CUDA runs take the host's GPU lock. Timing samples count only
with recorded one-minute host load below 8 before and after the call.
Earlier rows without those receipts remain historical correctness evidence.

## Reuse survey before implementation

At the input revision Cargo.lock had no sorting dependency. Rayon 1.11.0 is now a pinned development-only baseline; production reuses the existing worker runtime.
At input `8aa79242`, the front end had only `ArrayI32`, `ArrayU32`, and `ArrayF32`.
The checker/oracle slice adds `ArrayBool` to fulfill the scalar payload contract.
The source evaluator, MIR evaluator, native host, and borrowed host API mirror that numeric array set.
The reference allocator has a one-million-element per-array cap, distinct from the device admission limit.
Benchmarking 2^26 through that source allocator would require an explicit benchmark/host-buffer path, not an undocumented cap increase.

| Candidate | License and maintenance evidence | Fit and decision |
| --- | --- | --- |
| Rust slice `sort_by_key` | Rust standard library, existing toolchain | Use for the independent reference oracle and CPU baseline; not the requested production radix implementation. |
| [radsort](https://docs.rs/radsort/latest/radsort/) | Published Rust crate, 0.1.1 in surveyed documentation; license and current upstream revision still need pinning | Stable byte-digit LSD implementation worth inspecting before new CPU code; do not add until safe-allocation and no-unsafe-core compatibility are established. |
| [rdst](https://docs.rs/rdst/latest/rdst/) | Published maintained Rust crate, MIT/Apache-2.0 | Documentation explicitly calls it unstable; reject as the direct semantic implementation. |
| [Rayon](https://docs.rs/rayon/latest/rayon/slice/trait.ParallelSliceMut.html) | Published Rust crate, MIT/Apache-2.0 | Required `par_sort` benchmark baseline only; reuse Invar's existing worker mechanism for production. |
| [CUB block radix sort](https://github.com/NVIDIA/cccl/blob/main/cub/cub/block/block_radix_sort.cuh) | NVIDIA CCCL, BSD-3-Clause, current upstream source | Candidate for block ranking/scattering in NVRTC kernels; block sorting alone does not implement device-wide sorting. Compile on the actual installed CUDA headers before selecting. |
| [CUB device radix sort](https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/device_radix_sort.cuh) | NVIDIA CCCL, BSD-3-Clause, current upstream source | Required nvcc baseline. Current headers explicitly reject direct NVRTC inclusion; do not suppress that guard as an integration strategy. |
| [GPUSorting](https://github.com/b0nes164/GPUSorting) | MIT for its own source, with additional component notices in [LICENSE](https://github.com/b0nes164/GPUSorting/blob/main/LICENSE) | CUDA Onesweep is a candidate, but upstream describes its CUDA implementation as a benchmark/demo. Its chained-lookback path assumes forward progress and is explicitly cautioned against on Apple. |
| [FidelityFX Parallel Sort](https://gpuopen.com/fidelityfx-parallel-sort/) | AMD, MIT; SDK implementation | Preferred Metal port candidate: explicit reduce/scan/scatter passes avoid inter-workgroup spinning. Upstream is HLSL/DX12/Vulkan, not an existing Metal implementation. |

This is a source/documentation survey, not measured NVRTC compatibility or performance evidence.
Do not import the GPUSorting repository wholesale: its aggregate license also includes LGPL benchmark material.
Pin exact upstream revisions and retain the relevant notices before porting any source.

## Algorithm and integration plan

Start CPU LSD radix sorting with four stable 8-bit passes.
Unsigned integer keys need no transformation; signed keys use `bits ^ 0x80000000`.
Canonical floating bits use `!bits` for negative values and `bits ^ 0x80000000` otherwise.
Parallel histograms are per contiguous input chunk; prefix offsets include bucket order followed by chunk order.
Stable scattering must preserve input order inside each chunk, independent of worker completion order.
Do not implement parallel scatter by racing atomic bucket cursors.

The production device implementation is an independent staged LSD radix,
with eight 4-bit passes and four kernels per pass: histogram, chunk scan,
carry scan and stable scatter. It is not a FidelityFX source port. CUDA uses
warp ballots for local rank; Metal uses SIMD prefix sums and requires width32.
Every pass is ordered on one stream/command buffer. There is no inter-group
spin wait. CUB block inclusion failed the real bounded NVRTC helper's header
search; its guard was preserved. Chained-lookback Onesweep was not admitted
because its forward-progress assumption does not satisfy this bounded protocol.
The independent nvcc CUB device baseline remains the CUDA performance comparison.

Reuse call mode handling at every boundary: source checking, source ownership, view conflicts, MIR construction, MIR validation, and execution.
In particular, treating a sort call as an ordinary unknown builtin in view checking would miss mutation through a live read view.
Native lowering must distinguish floating arrays from unsigned payload arrays, even where both share an underlying word-buffer representation.
Unknown or losing measured costs select CPU; explain entries and execution receipts must name the actual selected implementation.
Apply the existing 2^28 device admission rule without relaxing source/host resource limits.

## Acceptance and current evidence

Tests include exact
reference/MIR/source-device result, fuel and cleanup observations; malformed
MIR loans; stable original-index payloads; and generated raw-bit key cases.
The reference oracle uses independent stdlib comparison sorting, not the
production radix transform.

The standalone native engine embeds the safe serial radix core. Shared-library
sort exports remain explicitly refused. Source programs containing these
intrinsics schedule the explicit calls and execute residual loops in serial MIR.
The general-net backend does not admit Bool arrays. Source array/live limits
are unchanged; raw word-buffer benchmarks have an explicit 2^28 admission cap,
plus real device capacity and allocation checks. Segmented offsets must cover the whole input; malformed coverage traps before mutation.

Automatic dispatch selects CPU with `sort-cost-unmeasured`. A generic loop
bandwidth model cannot authorize a radix cost estimate. Forced device explain
receipts name `cuda-lsd-radix-8x4` or `metal-lsd-radix-8x4`; CPU receipts distinguish
serial and genuinely joined radix work per call. No automatic loop-to-sort
recognition is introduced.

The initial implementation evidence includes Debug/Release library and focused
suites, strict Clippy, all three unstable-scatter mutations, coverage and the
88-row baseline matrix. These pre-ruling timing rows lack load receipts and
are excluded from accepted performance comparisons. Current C10/C14 conformity
checks, source lineage and any load-qualified reruns are recorded separately
in the evidence log. Auto remains CPU until sort-specific calibration is admitted.
