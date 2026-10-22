# Ordinary-source family dispatch (A0)

A0 introduced the registration boundary at `27002092`, preserving the existing
source admission, fusion, cost arithmetic, forced-engine errors and publication.

Draft A1 tensor registers strict scalar-accumulator and direct-cell f32
contractions for ordinary `invar run`, with real CompletedRegion publication.
Its qualification
passed Debug/Release on CUDA RTX 5070 Ti sm_120, including explicit TC adapter
execution and PTX/MMA proof. Metal and corrected transfer-cost checks
passed too. Those historical TC checks used the family-private checker.
Ordinary `tc_mul_add` checking and exact serial/auto fallback are now implemented;
refreshed hardware qualification remains open. Normal refresh merge 8eea8e69 includes the per-offer attribution fix
8ef0ffc1; fresh CPU/Metal and CUDA Debug library/CLI checks pass and the remaining CUDA suites are pending.
This integration also retains main's grid and segmented registrations. Current-head qualification remains pending.

## Registering a family

Add `src/schedule/families/<family>.rs` and one `Variant => module,` line in the
`families!` invocation in `src/schedule/families.rs`. The macro declares the module
and generates a closed typed enum and exhaustive matches for CPU, Metal and CUDA.
Payloads are boxed so different contract sizes do not require shared enum edits.
Use `core.rs` for the exact signatures, not a trait or a whole-program host runner.
Run rustfmt explicitly on each adapter: Cargo fmt cannot discover macro-declared modules.

- `Candidate` owns a source/MIR-bound contract; `recognize(source, program, region)`
  returns `Result<Option<Candidate>, admission::Error>`. `None` means no match;
  an invalid recognized contract returns a typed error. Every loop, including
  core-rejected loops, is offered to the registry. Function and span select the
  candidate when MIR actually reaches that loop.
- Registry order is precedence. Core is first and retains map extraction, then
  recurrence before sealed reduction, and the existing DAG/pair fusion decisions
  from `plan::decide`. Its producers and staged consumers stay together. Append
  M4 adapters in this order: grid, tensor, segmented, indexed, summary. Different
  functions/spans do not compete. At the same span the first match wins; a live
  refusal never retries a different recognizer. A completed outer-loop override
  bypasses nested loops; a serial outer loop can still reach nested candidates.
- `cpu` returns `Result<Option<Step>, schedule::Error>`; `Ok(None)` retains MIR for
  this loop only. M4 adapters may use this for their serial fallback. The M4
  `PlanKind` labels are reserved, so A1 adapters need no shared enum edits.
- Both backend submodules implement `cost(candidate, live, caps, model)` and
  `step(source, candidate, live, pending, device, [compiler,] fault)`. Cost receives
  live scalar dimensions and lengths, not just iteration count. Use checked byte,
  operation and launch counts with the existing measured `DeviceCostModel::estimate`;
  unknown legality/cost yields `None`, never an invented favorable estimate.
  Metal also supplies its existing `ineligible_reason` hook.
- A device step returns an owned, completed `Step`, cache-hit flag and `HintState`,
  or the backend's typed `Error`. It never mutates live source owners. Reuse the
  family's completion/receipt verification and populate its `LoopOverride` field.
  A compile/verification/completion failure cannot publish. Pending work remains
  a typed fatal error; auto fallback is allowed only before submission or after
  confirmed quiescence. Unreached and zero-trip loops submit nothing.
- Forced engines retain typed `METAL_UNAVAILABLE` / `CUDA_UNAVAILABLE` errors when
  there is no independent candidate or no completed publication. They do not
  silently switch to CPU. Existing forced live admission failures remain typed
  errors. In auto mode, a false live guard uses CPU admission then serial MIR for
  that loop; publication guards and source fuel remain enforced by MIR.

## Branch inventory

Read-only snapshots fetched on 2026-10-15: M4-02 `bd42b8a6`, M4-03 `509e0c1f`,
M4-04 `1a6a4ff6`, M4-05 `5522faff`, M4-07 `f8557c04`. Locations below refer to those
branch snapshots, not A0's base. `graphify query` was attempted first: the CLI
reported no `graphify-out/graph.json`. No graph build, install, or family-branch
write was performed; inventory uses `git show` and `git grep`.

| Family | Source recognizer and admitted forms | Metal / CUDA entry | Typed refusals | Publication / cost data / blockers |
| --- | --- | --- | --- | --- |
| Segmented and compaction, #100 | `segmented_device.rs:280` `MapContract::from_source`: nested offset ranges, u32 wrapping constant/segment-index map; `segmented_source.rs:173`: wrapping sum/product per segment; `compaction_source.rs:244`: stable comparison flags plus exclusive positions and scatter | `MetalDevice::run_segmented` (`crates/invar-metal/src/macos/segmented.rs:19`); `CudaDevice::run_segmented` (`crates/invar-cuda/src/cuda/segmented.rs:35`); IR via `reduce_ir:67`, `compact_ir:98`, `MapContract::ir:410` | Source/Net/Typed/Unrecognized/Admission; device `segmented::Error` at `crates/invar-device/src/segmented.rs:117`: version, source, graph, contract, bounds, shape, operations, capabilities, source limit, offsets; source contracts add offset order/coverage, budgets, fuel, descriptor/positions/allocation refusals | Completed private `SegmentedResult.output`; no reached-source publication permit. `Verified::{buffer_words,launches}`; result compile/kernel timing and transfer bytes (host elapsed can be sampled). **Blocker:** add reached-loop output, compaction's two-loop allocation/count/fuel protocol and role/span accessors. |
| Gather/scatter/histogram/frontier, #103 | `indexed_device.rs:26` `Plan::from_source`, in order: `indexed_source.rs:97` gather/injective scatter; `scatter_proof.rs:280` certified affine scatter; `histogram_source.rs:265` modular weighted histogram; `frontier_source.rs:226` bounded dense-mask frontier | `MetalDevice::run_indexed` (`crates/invar-metal/src/macos/indexed.rs:7`); `CudaDevice::run_indexed` (`crates/invar-cuda/src/cuda/indexed.rs:16`); `Plan::bind:62` | Source/Unrecognized/Binding/Device; `indexed::Error` (`crates/invar-device/src/indexed.rs:43`): version/source/binding/layout/alias/shape/law/limit/capabilities, ordered index/duplicate/capacity, allocation | Completed `Receipt.output`, not a MIR permit. Existing gather/scatter/histogram hosts have private completion fields; frontier needs its reached-source bridge. `Bound::{buffer_bytes:174,transfer_bytes:189}`, `schedule:509`, receipt bytes/phases; no per-call timing in receipt. **Blocker:** expose contract roles/spans and preserve first-failure/duplicate policy. |
| Transpose/stencil, #104 | `transpose_source.rs:137` `Contract::from_checked`: fresh row-major u32/f32 transpose with two counted loops; `stencil_source.rs:336`: fresh radius-1 five-point wrapping-u32/strict-f32, supported zero/clamp boundaries, repeated calls for time steps | Metal `grid.rs:17` `run_transpose`, `:53` `run_stencil_u32`; CUDA `cuda/transpose.rs:526` `run_transpose`, `cuda/stencil_u32.rs:97` `run_stencil_u32` (paths under respective backend crates) | Source/CheckedMismatch/Net/Unrecognized/Bounds/DescriptorMismatch/Device/DeviceU32; physical transpose rejects layout/alignment/bounds/alias/capabilities; u32 stencil rejects numeric/boundary/steps/geometry/index/target/source limit | Existing `transpose_source/host.rs:163` reached-step logic and `Completion` at `:24`; stencil host has its own completion protocol. `GridResult` bytes/phases/compile/host time; CUDA transpose compile timing. **Blocker:** reuse completed source tickets; word stencil runtime is u32, so strict-f32 stencil must be explicitly refused until a matching runtime exists. |
| GEMM/Tensor Core, #110 | `tensor.rs:347` `Contraction::from_checked`, contract recognition around `:152`: fresh rank-2 f32, nested i/j/k loops, zero seed, ordered multiply then add; `contraction_source.rs:259` also recognizes direct-cell strict updates and explicit `tc_mul_add` source; `source_device_command.rs:72,131,159` private qualification routes | Source-bound `tensor/host/metal.rs:21` `Metal::open` and `tensor/host/cuda.rs:16` `Cuda::open`; low-level `run_contraction`: Metal `macos/floating.rs:266`, CUDA `cuda/floating.rs:662`; tiled CUDA at `:674` and Metal at `:278`, both dispatch explicit TC plans | Source/Mir/Net/Unrecognized/Layout/Device/Cache (`tensor.rs:331`); strict probe/underflow, layout/alias, numeric policy, caps, resource and pending errors; `tc-f16-f32-v1` is separate from strict-f32 | `tensor/host.rs:129` reached-source tickets; packet hash/shape checks before response publication. Results expose transfer bytes, compilation and Metal kernel time; dimensions give M*K*N work. **Blocker:** preserve D1 probe/subnormal fallback, never silently choose relaxed Tensor Core arithmetic for strict source. Metal/CUDA strict and explicit TC emitters/runtimes exist; direct-cell TC still needs its reached-source completion bridge. |
| Bounded summaries, #117 | `summary_source.rs:277` `Contract::from_checked`: state tables K<=16 or modular matrices d<=4, final state or inclusive prefixes; `Prepared::device_plan` at `summary_source/device.rs:11` | `MetalDevice::run_summary` (`crates/invar-metal/src/macos/summary.rs:263`); `CudaDevice::run_summary` (`crates/invar-cuda/src/cuda/summary.rs:261`) | Source/CheckedMismatch/Net/Typed/Unrecognized/Bounds/Shape/Seed/Alias/Allocation/RuntimeMismatch/Budget/InteractionUnavailable/FamilyBound/StateDependent/Device (`summary_source.rs:53`); stable near-miss reasons `summary_family_bound`, `summary_state_dependent` | `summary_source/host.rs:20` CPU completion; GPU returns private `SummaryResult.output`. `Verified::{coefficient_words,output_words,transfer_bytes:170,blocks}`; compile/kernel/host timings. **Blocker:** bridge actual completed GPU output into the source ticket, preserving seed, source fuel and prefix layout. |

## A1 family adapters (one PR per family)

Every checklist includes: integrate A0 without rewriting its base branch; one new
adapter and one registration; both feature-gated backends; true live loans/fuel;
forced typed refusals; auto measured-cost/guard fallback; output equality and
no-publication failure tests; CLI `run --engine` evidence on both hardware targets.
Estimates below count adapter, family-owned accessors/ticket work and focused tests,
not the existing family implementation or imported A0 moves.

- [x] **Grid** (CPU, Metal and CUDA adapter qualification complete), `Grid => grid`, 250-450 lines: reuse transpose/stencil contracts,
  expose their reached-step helpers, bind live dimensions/layout, check completed
  words before constructing existing completion fields; test u32 and f32 transpose,
  stencil boundaries/time-step parity, and explicit unsupported f32 device refusal.
- [ ] **Tensor**, `Tensor => tensor`, 350-600 lines: reuse contraction packets and
  source ticket, keep strict D1 and underflow guard, use `contraction_source::Contract` for direct-cell/`tc_mul_add` forms, cost checked dimensions plus
  transfers/launches, gate relaxed TC on the explicit `tc_mul_add` policy and add its reached-source completion bridge; test strict errors and
  both completion outcomes. Do not call a whole-program host inside `step`.
- [ ] **Segmented**, `Segmented => segmented`, 450-750 lines: expose map roles/span;
  bind reduction offsets and output; implement the missing compaction completion
  bridge with source allocation/count/flags/positions and both loop spans. Test
  empty segments, malformed offsets, predicate stability and failed completion.
- [ ] **Indexed**, `Indexed => indexed`, 350-650 lines: expose bound contract
  identity/roles, reuse gather/scatter/histogram tickets, add frontier completion,
  cost `transfer_bytes` and phase schedule; test first bad index, duplicate scatter,
  histogram wrapping law, capacity and frontier ordering.
- [ ] **Summary**, `Summary => summary`, 250-450 lines: prepare from reached live
  coefficients/seed/count, lower `device_plan`, validate completed GPU words and
  publish through the existing source completion protocol; test final/prefix forms,
  family bounds/state dependence, budget, alias, pending and seed mismatch.

Adapter registration overlaps mechanically in `families.rs`. The missing segmented,
compaction, frontier, direct-cell TC and summary-GPU completion bridges publish
through the shared [publication bridge](#publication-bridge) below; no A1 adapter
adds a `LoopOverride` field or a MIR publication arm, and this registry does not
grant an absent publication permit. Keep family receipt validation in family-owned files and never
edit another adapter. Keep core first; A1 inserts in the order above. Core recurrence already wins over core reduction. A degenerate summary form
that also meets core affine-scan admission stays with core.
General indexed and affine-scatter recognizers overlap inside the indexed family:
retain `Plan::from_source`'s order. Tensor and summary matrix forms differ in
numeric type, state and loop structure. Grid/segmented outer loops may contain
core-shaped inner loops: outer completion suppresses those inner executions;
serial outer fallback leaves them available. Do not register the same outer span
twice or change another family's recognizer to resolve a conflict.

## Publication bridge

A completed device region publishes through one typed contract, `CompletedRegion` (`src/mir_eval/region.rs:109`).
An adapter's `step` returns `LoopOverride::completed(region)`, which sets the single generic `region` field (`src/mir_eval/bulk.rs:19`).
The law-directed parallel normal form (P1/P3) publishes through the same field later; nothing is added per family.

The contract describes everything the reached outer loop did, nested loops included:

- `function` and `span`: the outer loop, reached by MIR with a fresh induction.
- `branches`: outcome counts `(taken, skipped)` for every MIR branch inside the region, keyed by its source span.
  Keys are the AST spans the recognizer already holds: a `for`/`while`/`if` statement span, or the expression span of an `and`/`or`.
  An omitted branch claims zero outcomes.
- `writes`: each pre-existing owner the region stores into, with its live `length`, the published `range` and the values in the owner's layout (`U32`, `I32` or `F32`).
- `scalars`: each pre-existing scalar place the region assigns, with its seed and final value.
- `cleanup`: owners allocated and dropped inside the region, as `(name, length)` in source order.
- `fuel`: source fuel from the family's own source-level tick contract (its `loop_fuel`).
- `trap`: the first source-order trap (observation contract v2); the other fields then describe the source prefix ending at that trap.

MIR validates the whole claim before publishing anything (`validate_region`, `src/mir_eval/region.rs:460`):

- the head is this function's loop at `span` with a fresh induction, and its claim is `(trips, 1)`, or `(<= trips, 0)` with a trap;
- the region CFG is describable: no early return, no user call, no move of an outer owner, no rebinding of a live name, and every allocation is a named owner dropped inside the region;
- every branch key names one branch of this region, at most once, and the counts conserve flow through the MIR CFG;
- fuel re-derived from MIR block ticks under those counts equals the claimed fuel, which also fits the remaining budget, so the family's source arithmetic and MIR's block ticks must agree;
- the written owner set and the assigned scalar set equal the outer places the region's code can store to;
- each write targets a live array of the claimed layout and length, stays in bounds, fills exactly its range, and shares storage with no other live place or write;
- each scalar holds its claimed seed and keeps its type;
- cleanup events match the region's owner drops by name and count, and fit the live element limit;
- a trap names exactly one trapping instruction inside the region.

Any failure is a typed `Refusal`; MIR publishes nothing and runs the loop serially, which is exact because the device never touched live owners.
After validation, publication (fuel, writes, scalars, cleanup, then the trap if any) cannot fail part way.
A region next to any other receipt for the same head is `BULK_AMBIGUOUS_COMPLETION`.

How each A1 adapter fills it:

- **Segmented / compaction**: one region per pass, each returned at its own head.
  The count pass claims the counter scalar (`0 -> k`), the loop head `(n, 1)` and the predicate `if` `(k, n - k)`.
  MIR then runs `filled_u32(count, 0)` and the position seed itself, so allocation, its accounting and its traps stay MIR's.
  The copy pass writes `out` with `length: k`, `range: 0..k` and the compacted words, claims the position scalar (`0 -> k`) and the same branch counts.
  Flags and positions stay device-private: the count must equal the allocation length and the predicate counts must reproduce the fuel.
  Keep the device result on the family candidate between the two heads; if the copy head is not reached with matching state, MIR runs that pass serially.
  Segmented reductions claim the segment loop `(segments, 1)` and the nested offset loop `(sum of lengths, segments)`.
  Needs from #100: per-pass fuel accessors (the contract's `fuel` array mixes both passes and the intervening statements), the nested-loop and `if` spans, and role accessors for the segmented forms.
- **Frontier**: the `for step` rounds loop is the region.
  It writes `frontier` and `seen` whole, claims `("next", vertices)` cleanup once per round, and claims every loop, `if` and `and` count from the candidate's per-round records (processed vertices, edge reads, discoveries).
  The final count loop is an ordinary `i32` scalar region or stays serial.
  Needs from #103: the span of every branch in the rounds loop and a device completion that yields the same per-round records.
- **Direct-cell TC**: the `i` loop is the region, with `(rows, 1)`, `(rows * cols, rows)` and `(rows * cols * inner, rows * cols)` for the `i`, `j` and `k` heads.
  It writes `c` as `F32` words over `0..rows * cols`, with fuel from `contraction_source::Contract::loop_fuel`.
  Outer publication suppresses the nested loops; a serial outer loop still offers them.
  Needs from #110: the `j` and `k` loop spans (the contract records only the outer span) and the strict or `tc_mul_add` policy gate before the region is built.
- **Summary GPU**: the step loop is the region, with `(steps, 1)` and fuel from `summary_source::Contract::loop_fuel`.
  The prefix form writes `out` over `0..steps * width` in row-major prefix layout and claims each state scalar (`seed -> final`); the final-only form has no write.
  Needs from #117: the completed GPU words in place of the CPU candidate's prefixes and final state.
- **Core map/reduce/scan** keeps its `array`/`scalar` fields and is not migrated.
  It maps onto the contract as a write of `0..n`, a scalar seed and final value, the head `(n, 1)` and `n` times the body ticks.
  Migrating would change behaviour: core's fuel is the single-body-block formula and its fuel shortfall raises `EVALUATION_LIMIT` rather than refusing.
  It shares the live trip guard (`live_trips`, `src/mir_eval/bulk.rs:56`) with the bridge.


## Grid A1 stack

The grid adapter uses the existing source recognizers and reaches only their exact
outer loop spans. It admits fresh row-major u32/f32 bit transpose and one radius-1
wrapping-u32 stencil slice, with zero or clamp boundaries. Repeated source calls
retain ordinary source ping-pong and allocation semantics. The strict-f32 stencil
is a typed forced-device refusal; auto retains serial MIR. Empty domains submit
nothing, including a nonempty row axis with zero columns.

Both targets use live dimensions, checked lengths, retained read loans, source
fuel and the existing sealed layouts. Completed words are checked before building
`CompletedRegion`: one full output write, outer/inner loop counts, and four boundary
branch counts for stencil. Fuel comes from the source contract; there are no outer
scalar writes, region allocations or traps on admitted inputs. The device never
mutates live source storage. Cost uses actual input plus twice the output bytes,
logical cells, one transpose operation or four wrapping adds, and one launch.
No family schedule hint or cache hit is fabricated.

Grid qualification is complete on CPU, Metal and CUDA. Shared publication semantics are
unchanged; the now-used constructor loses its obsolete dead-code expectation.

Grid A1 qualification is recorded on draft #145: 79 distinct affected CPU tests per
Debug/Release profile, strict default/Metal/CUDA Clippy, eight actual Metal forms in
each profile, real CLI routing and three failing mutations with exact restoration.
After normal bridge merge `8de087b8` (including `8ef0ffc1`), the CPU selection
passes 80 tests per profile and CUDA Debug/Release each pass all eight forms on
RTX 5070 Ti with actual `sm_120` CUBIN, typed refusals, pending/reap,
CLI CUDA/auto equality and checked direct-host timings. See the qualification
record for exact revisions and the preserved earlier Metal receipt. No full
M4/M5 or performance acceptance is implied.
