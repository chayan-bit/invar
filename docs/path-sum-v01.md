# P4a path-sum lowering v1

P4a lowers a checked L9 (`(sum c_j r_j + k) % P`) or L10 (checked `a + b`) recursion certificate into one program-specific kernel source (normal-form design section 10, design A' from the P4a kernel profile).
The module is `invar::general_net::path_sum`.

## Pipeline

1. `lower(net, proof, entry)` partitions the net, builds P1's normal form and runs `certificate::check_all`.
2. `Entry::find` accepts the recursion itself or a straight-line entry that ends in `return root(..)`.
3. The recursion certificate's `Fact::Combine` entries are paired, in `tree::walk` order, with the combine returns that mention self-call holes.
4. P8's `LawRepr::from_certificate` re-checks an L9 certificate, and its `ResidueSum` modulus must equal the kernel's modulus.
5. The interval analysis in `range.rs` searches a narrow licence: a box of root-argument ranges on which no demanded operator can leave its type.
6. The licence becomes `Narrow { bits: 32 }` and `Bounds` annotations on the certificate, with `Source::Analysis { name: "p4a-interval", version: 1 }`.
   `path_sum::rederive(partition, certificate)` recomputes them from the licence.
7. `emit::source` writes one source that compiles as MSL 3.1, as CUDA through NVRTC and as C++17 for the CPU model.

A program that does not lower gets a `Refusal`; it then runs only on the exact path.

## Kernel

* The path-replay kernel gives each lane one point of the root interval `[lo, hi)`, keyed `lo + gid`.
  The lane replays root to leaf through the unique child that contains its key.
  It owns a leaf, or a node constant, only when the node interval starts at its key.
* The children must partition the parent interval in order.
  `interval_pair` proves the chain syntactically: the first child starts at `lo`, the last ends at `hi`, and neighbours share a bound.
  `geometry.rs` then tries to prove every child nonempty under its path conditions.
  Its domain is `sum c_s * s + [L(len), U(len)]`: integer coefficients over `lo` and the root's `let` locals, plus affine rational bounds in `len = hi - lo`.
  Locals stay symbolic until a check, so `m2 - m1` cancels `m1`, and a product with a non-negative factor keeps its affine bounds.
  When the proof holds (`Lowering::geometry`), device lanes skip the per-node checks and the last child slot needs no key test.
  The CPU model keeps every check as `PROVEN(..)`, which aborts on failure.
  Otherwise every node is checked at run time, and a failed check sets `FLAG_GEOMETRY`.
* A recursion with no interval parameter pair runs serially in one lane, with an explicit stack.
* The weighted reduction multiplies a lane weight by the edge coefficient at each combine.
  Owned leaves and constants add `weight * value mod P`.
  The kernel reduces within the group and then across groups.
* Lane flags and raw words are rare, so they never ride the value reduction.
  On Metal they are ORed into two shared words that one barrier at kernel entry clears.
  On CUDA the first reduction barrier is `__syncthreads_or`, which also votes on whether any lane has a flag or raw word.
  Only a positive vote clears and ORs the shared words, so the common case pays no flag barrier and no shared flag traffic.
* The raw-root case covers a root that reaches a leaf through tail calls only.
  Such a root returns the leaf's raw value, flagged `FLAG_RAW`, and the host publishes that value unreduced.
* The L9 guards (`TableRange`, `NonNegativeConstants`, `NoOverflowCombine`) are checked per lane at run time.
  A failure sets `FLAG_GUARD`.
* Depth past the oracle's 64-frame limit sets `FLAG_DEPTH`, and an exhausted loop budget sets `FLAG_BUDGET`.

## Host

`Lowering::plan` evaluates the entry prologue with checked arithmetic, checks the licence box and bounds the lane count.
It runs on every call, so the prologue environment is a small vector rather than a map.
It then picks a shape with the `CostModel`:

* `Root`: one lane.
* `Fused`: one group loops over every lane and reduces once.
* `Parallel`: one lane per thread, followed by one reducing group.

`Lowering::route` adds the cost model's D5 refusal for a `Scope`.
A host call (`Scope::Host`: plan, launch, wait, publish) is refused with `Plan::Fallback("cost")`, and the native path answers, when `CostModel::host_refusal` lists its shape and lane count.
These are the launches that the D5 gate measured losing to the handwritten kernels, so the gate is not relaxed:

* CUDA refuses every host call, because every host-scope cell missed on an idle GPU.
* Metal refuses host-scope `Fused` launches above 32 lanes, because the 256-lane host cells of divide-02 and compose-02 missed in the exclusive runs.
  The model refuses by shape and lane count, not by program, so the 256-lane host cells of divide-00 and compose-00 (medians 0.97 to 0.998 across the six runs) are refused too.

A device-resident caller (`Scope::Resident`) is never refused, because every resident median passes.
The differential tests run at resident scope so that every kernel stays exercised.
`the_cost_model_refuses_every_measured_host_miss` reads the committed D5 evidence and pins the refused set to it.

`Lowering::finish` discards any flagged result.
`Lowering::run` publishes the fast value, or the oracle observation (the value or the first source-order trap) on every fallback.

## Interfaces

Exposed:

* `lower`
* `Lowering::{plan, execute, run, source, p8_reduce}`
* `Launcher`, implemented by `device::{CpuModel, Metal, Cuda}`
* `rederive`, and `RangeAnalysis`, the `p4a-interval` v1 range analysis as P1's named `certificate::Analysis`
* `invar_metal::PathSum`
* `invar_cuda::cuda::PathSum`

Consumed:

* P1 `normal_form::{from_partition, certificate::check_all}`, together with its typed `Annotation::{Bounds, Narrow}`.
  `lower` checks every attached annotation with `certificate::check_with_analyses` and `RangeAnalysis`: P1 re-derives its own annotations and asks the analysis for the bounds and narrowing under the licence the certificate states.
* P8 `LawRepr::from_certificate`, `msl::emit_reduce` and `cuda::emit_reduce`.
  P8 emits the ordered `ResidueSum` reduction, but the kernel keeps its own fused group reduction.
  That reduction also carries the flag and raw words, which P8's one-word element cannot.
  Routing the cross-group stage through P8 would also add a third dispatch and a 64-bit modulo per join.

## Checks

* `tests/gnet_p4a_path_sum.rs` runs a differential of every lowered program against the CPU net oracle.
  The inputs come from trap-heavy generators plus the 96 scored rows of the P4a kernel profile.
* The CPU model compiles with `-DPROOF_CHECK` under UBSan.
  Every narrow operator aborts on an out-of-type result, so an unsound licence fails loudly.
* Each `Mutation` must produce at least one mismatch.
  `AssumeGeometry` trusts an unproven partition and must be caught by the CPU model's `PROVEN` re-check.
  `DropLaneFlags` removes the device flag publication; the CPU model has no group reduction, so the Metal and CUDA differentials must catch it.
* `the_partition_is_proven_statically_only_where_it_holds` pins the proof to the four gate programs, `l9-pathsum` and `l10-range`, and refuses it for the `geometry` fixture.
* `metal_kernels_agree_with_the_net_oracle` and `cuda_kernels_agree_with_the_net_oracle` repeat the differential on hardware.
* `tools/src/bin/invar-pathsum-gate.rs` reruns the P4a profile protocol against the handwritten baselines in `tools/bench/p4a-gate/handwritten/`.
  Its `breakdown` subcommand times the host-only part of the guarded call (`plan`, `arg_bytes`, `finish`) that the baseline does not run.

## Known limits

* P8 has no L10 representation.
* A loop whose interval does not converge, or an operator that may leave its type on every candidate box, refuses the lowering (`compose-05`, `compose-09`).
* P8's `emit_reduce` ABI carries one value word per element, joins with a 64-bit modulo and needs its own dispatch, so the kernel keeps its fused group reduction (see Interfaces).
* D5 on Metal (exclusive Mac) scored 30 to 31 of 32 cells: the host-scope 256-lane cells of divide-02 and compose-02 missed by at most 0.32% on the median, although their measured device and host-only costs are lower than the baseline's (see the evidence README).
  The cost model routes those host calls to the native path.
  One resident cell (compose-02, 32 lanes) missed U95 in one run because of a single outlier session (median 0.980), and it stays launched.
* D5 on CUDA (idle RTX 5070 Ti): every resident cell passes, and every host-scope cell missed (medians 0.97 to 1.23), so the cost model routes every CUDA host call to the native path.
* Open for v0.2: the two-speed CUDA launch latency.
  Host wall time is bimodal for both variants (about 28 and 35.5 us), and the generated call lands in the slow mode more often, although its kernel is faster by event time and both variants share the same launch path.
  Lifting the CUDA host refusal needs this latency explained and the host cells rescored.
