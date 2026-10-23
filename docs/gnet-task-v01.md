# GNET P4b: residual fork-join task region T (`gnet-task-v1`)

This is the v0.1 lowering for programs whose certified parallel normal form keeps a fork-join after every law (law L20, `NormalForm::is_flat() == false`).
It implements the [region design](design/gnet-parallel-v01.md)'s fork-join T reduced to fixed-width expansion, nested forks, a compacted arena and source-ordered joins ([normal-form design](design/net-parallel-laws-v01.md) section 10, P4b row).

## Admission (`src/general_net/task/mod.rs`)

`task::admit(net, proof, form, entry)` re-derives the region partition, checks that the normal form is the net's own (fingerprint and `certificate::check_all`), and refuses a flat form.
Every function reachable from the entry must have a structured statement tree, be pure, and use only `i32`, `u32` and `bool` scalars; the only intrinsics are `wrapping_add` and `wrapping_mul`.
Mutual recursion is refused; self recursion is the task source.
A function is a *machine* function when it is recursive, holds an L20 fork site, or calls a machine function.
Machine calls must be whole statement values (`let`, assignment, expression statement, `return`) outside loops.
Each refusal has a stable reason code (`Refusal::code`).

## Lowering (`src/general_net/task/lower.rs`)

Machine functions become resumable state machines `gt_step_<i>` over a frame of 32-bit slots (parameters first, then every local).
A machine call suspends with `GT_R_CALL`; the resumed state reads the callee's result.
An L20 fork site in *fork mode* evaluates every sibling's arguments in source order, records the first failing argument as pending, and suspends with `GT_R_FORK`; its join state binds the sibling results and yields (`GT_R_YIELD`, a commit point).
Outside fork mode the same site runs the siblings one by one.
Other functions are plain checked C functions `gt_fn_<i>` with a step wrapper so a fork may spawn them.
All arithmetic is checked exactly as the oracle checks it (overflow, division by zero, `MIN / -1`, negation), and every trap carries its source span.

## Driver (`src/general_net/task/kernel.rs`)

The arena holds one node per spawned call: state, function, pc, parent, sibling slot, call depth, child count and done count, the first child, the pending argument failure, the frame, and one result slot per sibling that the finished child writes into its parent (so a resumed join reads only its own node).
A node runs until it returns, fails or reaches a call, spawning one child per sibling (nested forks fork again).
A node whose children are all done resumes: the first failing child in source order, then the pending argument failure, is its result; otherwise it binds the results and continues, and a continuation call spawns new children.
The arena is compacted (children are allocated with one atomic add), so no lane is padded and no leaf runs twice.
A full arena runs the node serially from its last commit point (`capacity_serialized` in the region design), never failing the program.
The call depth limit (64 live frames) is checked at every call and reported as the oracle's `CALL_DEPTH_LIMIT`.
A failing child sets the cancel flag of every later sibling; cancellation only stops work the oracle never does, because the join already takes the first failing child in source order, so a cancelled sibling's result is never read.
Earlier siblings are never cancelled (mutation `CancelEarlier` must fail).
Source fuel is not modelled: a lane step budget only bounds termination and is reported as a tool failure, never as a value.

Two schedules share the node machine (`cost::Schedule`):

- **Cascade** (CUDA, and Metal where MSL 3.2 is available): one persistent kernel `gt_cascade`, one dispatch per call.
  One lane per 32-lane SIMD group is a worker, because the step machines diverge and nodes sharing a SIMD group would run one after another.
  Workers claim arena indices in allocation order (state word 24, the device-wide ready list) and poll the claimed node's tag, one poll per loop iteration.
  A node is published by storing the call's tag (state word 25, a host-written sequence) after a device fence; the tag says READY, or TASK once the arena top reached `width`, and a TASK node runs serially in the worker.
  The last child to finish resumes its parent in the same worker (cascade join).
  Metal uses `coherent(device)` buffers and device-scope fences (MSL 3.2); CUDA uses volatile node words and `__threadfence`, and clamps the grid to the blocks the device keeps resident.
  A bounded poll count marks a call stalled (a defect, reported as an error, never a value).
- **Staged** (Metal MSL 3.1 fallback): a single-group scheduler kernel `gt_sched` expands ready nodes breadth first until a round holds `width` ready nodes, which become serial tasks of the multi-group run kernel `gt_run`; joins resume in later scheduler passes.
  The run kernel and every later scheduler pass are dispatched indirectly from group counts the scheduler writes; submission `i` encodes `min(2^i, epochs)` `(run, sched)` pairs, and the host submits again until the root is done.

The CPU model compiles the same source with a host harness for both schedules (the cascade as one worker).

## Runtime path selection (`src/general_net/task/host.rs`)

`host::metal::profile` probes the device: MSL 3.2 needs `MTLGPUFamily::Metal3` and a successful compile of a small 3.2 kernel (`MetalDevice::supports_msl_3_2`).
It returns `Profile::METAL` (cascade) when both hold, else `Profile::METAL_STAGED`; D13's MSL 3.1 / Apple7+ floor is unchanged.
Every `Outcome` records the path that ran: `metal-msl3.2-cascade`, `metal-msl3.1-staged` or `cuda-cascade`.

## Cost model (`src/general_net/task/cost.rs`)

`Profile` carries the schedule, cascade lanes, expansion width, group sizes, arena and the refused regions of one device class.
The bench pilot calibrates lanes (half, one and two times the profile's) and width over `task_bench::WIDTHS` (64 to 16384) on the C1 tuning split (programs 00, 02, 03, 04, 07, 08); the frozen values are folded into the profile, and `tests/net_task_region.rs` checks them against the committed pilot.
A width wider than the tree only costs scheduler work, never padded lanes, because the arena holds compacted nodes instead of `radix^depth` slots.
`task::route` refuses a region whose fingerprint the scored bench measured above the D5 residual tier on that device (`Refusal::CostModel`, code `task_cost_model`); the caller runs it on the native path.
The test also checks that each refused set equals the set of programs with a resident or host MISS in the committed report.
For contrast, `cost::digit_padding` reports the padded share of the P4a spike's digit form.

## Runners

`invar_metal::task` (staged and cascade) and `invar_cuda::cuda::task` (cascade) own the four buffers (entry arguments, state, nodes, lists) and the submissions.
`invar::general_net::task::host::{metal::MetalTask, cuda::CudaTask}` emit, prepare and decode.

## Tests

- `tests/net_task_region.rs`: CPU model (ASan and UBSan) against the CPU net oracle on the C1 family and generated trap-heavy fork-joins, under a wide plan and a tight plan that fills the arena; every named mutation (`task::Mutation`) must be caught; refusals.
- `tests/net_task_region_metal.rs` and `tests/net_task_region_cuda.rs` (ignored): the same corpora on the devices.
- `bench/net_task_region_bench.rs` and the bench tests in `tests/net_task_region_cuda.rs` (ignored): the D5 residual-tier bench against handwritten B-form kernels (`bench/task-region-v01/c1_handwritten.inc`).

## D5 residual tier (not met; refused to the native path)

Each run is scored with 3 sessions of 10 warmups and 30 interleaved pairs per program and input.
Each cell is the median of the 90 ratios, with a hierarchical bootstrap U95.
The pilot froze lanes and width first.
Metal GPU steps run with no other heavy job on the machine.
CUDA steps wait for an idle GPU (utilization at or below 5% for 5 samples) and record `nvidia-smi` before and after each step (`bench-cuda/nvidia-smi.log`).

| Device and path | Frozen lanes / width | Resident pass (<=3.0x, U95 <=3.15) | Resident median | Host pass (<=2.0x, U95 <=2.10) | Host median | Programs with no miss |
|---|---|---|---|---|---|---|
| Apple M4, MSL 3.2 cascade (`bench-metal`) | 2048 / 16384 | 23/60 | 3.61x | 57/60 | 1.28x | none |
| Apple M4, MSL 3.1 staged, information only (`bench-metal-staged`) | - / 64 | 17/60 | 4.27x | 43/60 | 1.69x | none |
| RTX 5070 Ti cascade (`bench-cuda`) | 8960 / 16384 | 47/60 | 2.51x | 30/60 | 2.00x | C1-01 |

The first cascade (one worker per lane) scored 10/60 resident on Metal.
Root cause: workers claim consecutive arena nodes, so up to 32 divergent step machines shared one SIMD group and ran one after another.
The effective parallelism was the node count divided by 32: C1-03 (185,6) ran in 110 us against 254 us fully serial, and growing the grid from 64 to 4096 lanes changed nothing.
One worker per SIMD group halved the medium-tree times (C1-03 (185,6) to 45 us).
Writing child results into the parent's slots and taking the serial decision from the published tag removed device round trips; together these changes gave the scores above.
Coherent memory and device fences were measured and are not the cost: dropping them changed under 10%.
A single-threadgroup cascade was slower.
The remaining gap is latency on medium trees.
Every fork and join hop is a chain of about 10 to 14 dependent device-memory round trips (about 0.3 us each on the M4).
On top of that comes a single-dispatch floor of about 10.5 us, against the handwritten kernels' 5 us for two dispatches.
The handwritten kernels replay the fork path in registers and join in group memory, so they pay neither.
Small inputs and large ones (where real parallel work hides the hops) pass; medium trees run 3.2x to 5.2x on Metal.
CUDA's resident tier mostly passes, but its host-to-host ratio misses because the handwritten baseline's host round trip is cheaper.

The cost model therefore refuses every C1 region on Metal (both paths) and every C1 region but C1-01 on CUDA (`Refusal::CostModel`).
Those regions run on the native path, and `tests/net_task_region.rs` keeps the refused sets equal to the committed reports' misses.
The gate is unchanged.
Lowering the hop latency further (for example a register replay of the top levels like the B form, or per-call routing by input size) is open.
## Open

- Source fuel and the speculative work budget of region design section 6 belong to the host and device frames (P2, P5, P6); T runs every sibling of a successful join to completion and cancels only the siblings after a failure.
- Heartbeat promotion of idle lanes is not implemented (fixed-width expansion only).
- The cost model refuses per region and device from the scored bench; a program outside the C1 family is admitted until a bench measures it.
- Arrays, f32 values and mutual recursion are refused.
