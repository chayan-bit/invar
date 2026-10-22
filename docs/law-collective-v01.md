# P8 law collective interface v1

P8 supplies certificate-bound ordered collective source, not body lowering or
frame publication. Interface-first commit:
`f3b1f782a40a36c8a273a416e2462c3684948f21`.
It was pushed before kernels and deliberately returned `Unsupported` from the
emitters. Subsequent commits implement those APIs. Hardware and D5 acceptance
are independent gates.

## Rust boundary

`invar::general_net::law_collective` exports:

```rust,ignore
LawRepr::from_certificate(net, proof, certificate) -> Result<LawRepr, Error>
msl::emit_reduce(&repr, Layout { lanes: 256, items_per_lane: 4 })
msl::emit_scan(&repr, layout) -> Result<Emission, Error>
cuda::emit_reduce(&repr, layout) -> Result<Emission, Error>
cuda::emit_scan(&repr, layout) -> Result<Emission, Error>
emission.launch(length) -> Result<Launch, Error>
route(&repr, scan, length, Residency::Host, &calibration) -> Route
decide(&repr, scan, length, Residency::Host, &calibration) -> Decision
```

The constructor runs P1b's independent checker. `representation()` is typed
emitter IR for identity, ordered combine, apply and numeric guard; `guards()`
retains every original runtime obligation. Caller-supplied snippets are not
accepted. Whole-region admission also requires `certificate::check_all`;
one certificate does not establish legality of sibling updates. Check the
fingerprint exposed by `certificate()` before reusing a descriptor on a net.

L9 certificates produce `Representation::ResidueSum { modulus }` despite having
no P1b `Rep`. P4a supplies path-weighted residues in `[0,P)`, handles the raw-root
case, and checks `TableRange`, `NonNegativeConstants`, `NoOverflowCombine` before
publication. P8 does not reinterpret a failed guard as a modular computation.
A false uniform `admitted` argument performs no output or scratch stores.

`Layout` admits power-of-two lane counts 32 through 1024 and 1 through 16
consecutive elements per lane, subject to 32 KiB threadgroup storage including
carry and control words. Length is at most 2^26; `launch` checks aggregate input,
output and look-back tables against 8 GiB using checked arithmetic. Actual
pipeline, buffer and device limits remain the dispatcher's responsibility.
The qualification-only Metal and CUDA hooks cap total storage at 2 GiB.
CUDA reuses the existing bounded NVRTC compiler and owned-stream fence; it
requires an already available NVRTC supporting the physical architecture.

## Element and seed ABI

All buffers are AoS u32 words. Every 64-bit field is low word then high word.
Logical 64-bit arithmetic uses unsigned operations; signed comparisons read
those bits as two's-complement. The caller generates valid elements from checked
source facts, records independent body traps, and keeps all source writes in
scratch. Arbitrary words are not a certified source program.

| Representation | Physical words | Element fields | Seed fields |
| --- | --- | --- | --- |
| Sealed wrapping sum/product | 1 | contribution/factor | scalar |
| Residue sum | 1 | residue `[0,P)` | scalar |
| Lifted sum | 6 | i64 total, minimum prefix, maximum prefix; include empty prefix 0 | i64 scalar, then zero padding |
| Saturated product | 4 | signed product capped at magnitude 2^33, u64 maximum absolute prefix including empty prefix 1 | i64 scalar, zero padding |
| Affine | 2 or 4 | `a,b`, ring-width words each | scalar in first ring-width words, zero padding |
| Matrix, k <= 4 | `(k+1)^2` times ring width | homogeneous row-major matrix | k scalar ring words, zero padding |
| Tropical | 6 | i64 shift, i64 bound, has-shift, has-bound | i64 scalar, zero padding |
| Select | `2+companions`, at most 10 | valid, raw scalar bits, companion words | valid=1, raw seed bits, companion seeds |
| Last | 2 | valid, raw scalar bits | valid=1, scalar seed |
| Counter | 2 | u64 non-negative count | i64 seed |

Ring width is two u32 words for `Checked64`, one otherwise. Wrapping arithmetic
is modulo 2^32; checked affine/matrix arithmetic is modulo 2^64; residue products
use u64 multiplication and reduction at each operation. Lifted sum and tropical
inputs must respect the exact-i64 bound from their source operations and length;
the mapper must reject an element whose widening exceeds that bound.

The emitted helpers are `P8 p8_id()`, `P8 p8_then(P8 earlier, P8 later)`,
`P8 p8_apply(P8 prefix, P8 seed, uint count)` and
`bool p8_guard(P8 prefix, P8 seed)`. Apply writes scalar results at the seed
positions above; other words are padding. Count zero returns the raw seed.
This is required for empty residue prefixes and zero-trip loops: a modular
identity must not silently reduce a raw seed. Strict f32 selection only moves
bits, uses ordered-bit comparison including subnormals, ignores NaN elements,
retains a NaN seed, and preserves signed-zero and companion ties. There is no
f32 addition, multiplication, affine composition or reassociation.

## Dispatch ABI

Both targets expose `invar_law_reduce_v1` and `invar_law_scan_v1` with six
arguments. Metal binds each at the numbered buffer slot; CUDA takes pointers
for slots 0, 1, 3, 4, and by-value u32 for slots 2 and 5:

| Slot | Meaning |
| --- | --- |
| 0 | read-only input elements |
| 1 | output elements |
| 2 | element count n |
| 3 | aggregate table, scan only |
| 4 | inclusive-prefix table, scan only |
| 5 | uniform admitted flag, after all certificate guards |

Dispatch `Launch::groups` groups of `Layout::lanes`. Reduction writes one
ordered partial per tile; repeat with partials until one remains, then apply
to the source seed once. An empty reduction writes identity. Exclusive scan
writes n representations; an empty scan writes no output (its dummy group may
publish identity to scratch). Input/output/tables
must not alias. Keep nonempty dummy buffers for unused bindings on Metal.

Each scan table has `Launch::table_words` u32 words, initially zero, private to
one dispatch. Each payload word uses two atomic words, each encoding one u16
half and its ready bit. No other address's publication is assumed to order
that payload. Aggregate and prefix records are immutable, with one writer.
Every observed complete record is thus safe under Metal 3.1 relaxed atomics.
A partially published record is ignored. Reset both tables before reuse and
wait for completion before resetting, reading or reclaiming them.

Both targets share the ordered tile algorithm. The look-back takes at most
129 observations of each predecessor, then cooperatively recomputes that tile
if it is unavailable. Every round consumes a predecessor, so it never depends
on another group being scheduled. `P8_REVERSE`, `P8_USE_PREFIXES`, `P8_POLLS`
constants permit reverse-order, aggregate-only and immediate-fallback stress.
Fallback can do quadratic tile work in the worst schedule. This is a correctness
ceiling, not a D5 performance claim.

Within a tile, each lane folds its consecutive elements, then ordered 32-lane
shuffles (`simd_shuffle_*`, `__shfl_*_sync`) combine adjacent lanes and one warp
combines the warp totals; lane groups are therefore a multiple of 32, as
`Layout` already requires. Thread 0 walks predecessors without group barriers
and reads each prefix and aggregate record in one round; the group joins only
for a cooperative replay. CUDA reads and writes the self-validating slots with
relaxed `volatile` accesses instead of read-modify-write atomics.
Three exact fast paths are fixed at emission. When `items_per_lane*words` is a
multiple of four, a lane whose whole run is in bounds and whose buffer is
16-byte aligned loads and stores it as 16-byte vectors; ragged tails and other
layouts keep the scalar path. Word offsets use u32 when `words*(2^26+tile)`
fits, which holds for every in-bounds offset because `n <= 2^26`; otherwise u64.
Wrapping32 ring operations are emitted in u32, which is exact because truncation
mod 2^32 is a ring homomorphism. The hardware tests cover 32-, 256- and 64-lane
layouts (the last keeps scalar loads) and four named fast-path mutations.

Prior art reused: the adjacent ordered tree, consecutive lane ownership, simd
shuffle scan and 16-byte vector loads in
`crates/invar-device/src/msl.rs`; the packed atomic halves and cooperative,
bounded fallback in `crates/invar-device/src/summary.rs` on
`origin/feat/v01-m4-07-device` (not present on the P1b base).
The algorithm follows [Merrill and Garland's decoupled look-back](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back).
Metal's memory-order constraints are specified in sections 6.16.1 and 6.16.3 of
the [Metal Shading Language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).

## Trap recovery and publication

`trap_recovery()` distinguishes:

- `None`: no collective arithmetic trap; original body statuses still apply.
- `PrefixBounds`: reduce first and test `p8_guard(total,seed)`. If false, scan,
  then launch `invar_law_locate_v1` to locate the earliest suspect element.
  This shortcut additionally requires a source witness: exactly one direct
  primitive update with no aliases, nested loops or other carried-variable
  uses. Polynomial equality alone can hide checked intermediates such as
  `s*2-s+x`; these forms require source replay.
- `ReplayEveryLane`: apply each exclusive prefix to the original seed and replay
  the original lane body in its source demand order. Checked and residue affine/
  matrix updates, and tropical final states can hide overflowing intermediates, so a final-state
  numeric guard cannot establish legality. Positive residue inputs can still
  overflow a checked multiplication/addition before `% P`.

The locator takes input, exclusive prefixes, n, one seed element, an atomic
u32 earliest initialized to n, and admitted, in slots 0 through 5. Dispatch
ceil(n/lanes) groups. It uses atomic min on failing prefix bounds. Lifted sums
locate the exact first violating element when each element retains its internal
prefix extremes. Product bounds are conservative for signed minimum and
pre-aggregated products; replay suspects in source order to resolve the exact
first trap or establish success. A conservative suspect is never a source trap.

The caller replays the selected source element to recover its operator span
and intra-element demand order, reconciles source fuel, steps, cleanup and
Demand/Complete observations, and only then publishes the allowed source
prefix. This source-body replay is intentionally owned by the lowering/frame,
not approximated by a collective helper. P8 alone does not close observation
contract v2 for an arbitrary source region.

Certificates without a representation other than L9 return `Unsupported`.
In particular, L10 recursive range sums need the source join tree's trap
ordering; flat leaf-prefix bounds do not prove that each source subtree sum
fits. P4a must retain that checked combine evidence/fallback. L11 scatter
certificates likewise need their owning lowering to supply a carried combine
representation; P8 does not silently invent one from an array certificate.

### L10/L11 fallback acceptance

The constructor checks the certificate before returning a law-specific
`Unsupported` reason. No enum variant, field, signature, entry point or buffer
ABI from `f3b1f782` changes. P4a must interpret this refusal as native placement,
before allocating or dispatching a P8 collective.

- **L10:** `rep=None`; `Fact::Combine` records child coefficients but supplies
  neither source join schedule nor a checked tree-summary/replay ABI. A source
  tree `(MAX + 0) + (1 + -1)` succeeds, while a checked leaf-prefix fold traps.
  `(MAX + 1) + (-1 + 0)` traps even though its mathematical total fits. A
  lifted linear prefix cannot substitute for either source join behavior.
  Admission needs a certificate-bound source-join tree, ordered prologue/leaf/
  join trap keys, and checked replay. Those belong to path/frame lowering.
- **L11:** `rep=None`; array certificates are explicitly forbidden from
  carrying a `Rep`. They prove store/read shapes, but the six-slot flat ABI
  has no target indices, segment offsets, per-target initial values, or
  destination/writeback contract. The checked updates `MAX, 1, -1` to one
  target must trap at the second writer even though the final sum fits.
  Admission needs target grouping that preserves writer order, each target's
  seed, and source-index replay/publication. Wrapping scatter also requires
  target grouping; supplying a scalar `Sealed` representation alone is not
  sufficient. These additions require an additive lowering-owned interface.

`p8_l10_l11_refusal_selects_executable_native_fallback` checks real extracted
certificates for both laws, routes their explicit refusal into the existing
`compiled::emit_owned(Target::Cpu)` boundary, compiles and runs that source,
and compares two successful values and two first-trap spans to the GNET
oracle. This proves the P8 refusal/native execution contract; ordinary-source
placement in P4a remains the consuming owner's integration test.

### Cost-model refusal

`route` decides whether a placement should use the collective or refuse it to
the native path, the same exact fallback the L10/L11 refusals select.
Scans always stay collective.
A device-resident reduction stays collective: the native path would first
have to read every byte back, so it cannot beat the collective's floor.
A host-resident reduction is refused when the collective's lower bound is at
least 95% of the native fold time.
That bound stages the input at the calibrated link bandwidth, then reads it
once at the calibrated device peak, with no launch cost.
`Calibration` carries several independent passes of the peak, link and native
throughputs: the best (fastest) value of each figure decides the route, and
the worst (slowest) value breaks a marginal decision.
A busy machine can only slow a pass, so the best pass is the least disturbed
estimate.
When the best figures put the floor-to-native ratio within 5% of 0.95, the
decision is marginal: the path that is faster under the slowest passes is
taken, so one busy pass cannot flip it and a given file always gives the same
route.
`decide` returns the route, the ratio and whether it was marginal; sessions
record all three.
`Calibration::from_json` rejects a record with a missing, zero or non-finite
figure, without passes or a source, or whose best and worst figures are not
exactly the maximum and minimum of its passes.
No device figure is compiled into P8.
`p8_calibrate_metal` and `p8_calibrate_cuda` record one file per device.
Peak is a device-to-device copy counting read and write.
Link is the warm path's own staging: the pinned upload on CUDA, and on Metal the
no-copy wrap of page-aligned host pages plus the GPU's first read of the fresh
wrap, timed as the extra wall time over the same blit from resident memory.
Native is the single-thread host fold of a full-size input (2^26 elements, the
largest D5 length), with 256 MiB streamed before every sample so it starts
cache-cold.
Cache-cold is not paged out: the input stays resident during calibration, so the
figure does not include swap page-in, and a session under host memory pressure
can fold far slower than calibrated (Metal run D: 37 to 45 GB/s calibrated, 12
GB/s scored at 2^26).
The refusal holds only while the host input is resident.
Each pass takes each figure as the median of ten samples after three warmups;
a calibration records five passes.
`p8_cost_model_refuses_tied_host_reductions_to_native` checks the refusal, the
scan and device-resident non-refusals, and three named mutations.
`p8_cost_model_marginal_decisions_are_deterministic_per_file` places two files
either side of the threshold inside the margin, shows that both get the route
their slowest passes favour, every time, and that the `IgnoreMargin` mutation
splits them.

### Template timing protocol

`tests/net_law_collective_bench.rs` and `tests/support/law_bench.rs` freeze four
flat-template cases: checked-sum reduce, i32 min reduce, wrapping affine scan,
and integer max-plus scan at every power of two from 2^20 through 2^26.
The handwritten reference implements its own algebra and adjacent Blelloch
tree; its scan uses block aggregates, one aggregate-prefix group and a
writeback scan. It calls no P8 emitted helper. Both sides use 256 lanes,
four items per lane, strict integer semantics and identical AoS inputs.
Reduction timings include the complete partial cascade; generated scan
timings include a device scratch-reset dispatch. The reference overwrites
its scratch and needs no reset. All outputs are independently accumulated
on the CPU and checked before timing and on every warm call.

Each independent process performs ten warmups then thirty paired blocks,
alternating the first implementation per block. Allocations/modules are
pooled equally. Resident timing spans all device commands, including their
dependencies; warm timing includes input upload/copy, dispatch, completion
and output download/copy. Neither scope includes compilation or correctness
comparison. Each resident command follows an untimed copy of the same side
submitted back to back, so it never starts on a GPU that idled during host
copies. Every checked call first clears the output outside the timed window,
so it validates only its own side. CUDA reads back into pageable host memory:
the driver's pinned allocation is write-combined, and host reads of it ran at
uncached speed, which made scan warm cells measure that read alone. Metal
warm input is caller-owned, page-aligned host memory (`invar_metal::HostPages`):
Apple GPUs share it, so each warm call wraps it with
`newBufferWithBytesNoCopy` inside the timed window on both sides instead of
copying it; every kernel must declare slot 0 `device const`, and each command
retains the pages until it is reaped. The `ZeroedHost` mutation wraps other
pages and must fail the exactness check. The session count per device is
pre-registered: ten for Metal, three for CUDA. The frozen estimator is
the ratio of medians; the one-sided U95 uses 2,000 hierarchical bootstrap
resamples (sessions, then paired blocks) with seed `0x8d5`. Every cell keeps
its raw samples or exact unavailability reason. Flat limits remain 1.00/1.05.
Each session routes every cell with the recorded calibration, input on the
device for resident and on the host for warm. A refused warm cell times the
native host fold as its generated side, checks every call against the CPU
representation, and asserts that all 40 generated warm calls took that fold.
The report labels such a scope `warm (native)`.

These are template measurements on already lifted AoS data, not complete
ordinary-source performance: mapper, checked source replay and publication
costs remain with the lowering. The additive benchmark hooks use the emission
8 GiB storage ceiling and check actual device allocation/headroom. Exact host
capacities bound the largest case below 12 GiB RAM, including pinned staging
and readback; the original one-shot qualification hooks keep their 2 GiB cap.
Unavailable hardware allocations remain open cells. A handwritten hierarchical
reference is provided, but no CUB quality comparison or independent baseline
review is inferred from it. Keep that prerequisite separate from a measured
ratio, including a negative ratio.

### D5 bar for v0.1

P8 ships in v0.1 on the CUDA result plus the cost-model refusal (project ruling).
CUDA met the flat gate at 56 / 0 / 0 over 3 sessions in the clean refusal run,
on scan and reduction code byte-identical
to every later run.
The contended CUDA confirm run, 51 / 4 / 1, is recorded next to it; its misses
are warm scans at kernel parity on a loaded shared host.
Metal runs C (46 / 8 / 2) and D (41 / 7 / 8) are recorded as an environment
limit, not as passes: the unscored cells are host-resident warm reductions and
cells at kernel parity whose timings track swap on a shared Mac.
The flat limits stay at 1.00 / 1.05 and no cell that did not pass is counted.
Metal warm scoring on a swap-free machine, a run-time residency check before
the native sum fold, CUDA warm-scan parity in an uncontended window and wiring
`route` into P4a placement are deferred to v0.2 (#159).
