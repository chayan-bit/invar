# P2 compiled body capsules and host frame, protocol v1

This branch implements an additive `device_host::host_frame::Plan` entry. It
consumes the verified net's P1a partition and P1b PNF, checks every certificate
with `certificate::check_all`, and emits from the same net-derived MIR as the
existing continuation compiler. P1 extraction and laws are unchanged.

The adopted catalogue calls first-exit L15 and monotone frontier L14. L13 is
compaction. `compiled::body_map::Compiled` exposes sites, explicit admission
misses, the MIR contract fingerprint and both generated sources. `Site` names
its `(function, head)` region and Map, First or Frontier shape. No caller-supplied
certificate, task image or device address can construct a host-frame plan.

`Plan::run` owns native serial control, exclusive entry leases, region scratch,
fences, validation, ordered publication and replay. The frame is compiled
program-specific C++, through the existing bounded compiler child and reviewed
native loader. It does not evaluate tree nodes or graph opcodes. The ordinary
ABI5/13 `device_host::Plan` and CLI are unchanged; P6 owns placement/integration.

## Continuation boundary

A site pauses before its loop branch, after evaluating the first header. Lane
zero begins at that branch. Later lanes restart at the header with their own
induction value; each stops on the backedge before the next header. After a
successful chunk the native frame executes the final header or requests the
next chunk. Calls, branches, inner loops, integer traps, strict software f32,
and iteration-local owners use the existing continuation/operation emitter.
The native frame never hands its addresses to a shader.

Two device kernels form one capsule: `invar_gnet_body_map_v1` uses groups of
256 with grid-stride coverage beyond 65,535 groups; `invar_gnet_body_resolve_v1`
uses one 256-thread tree min reduction to find the first non-complete lane.
Both execute in encoder/stream order under one observed completion fence.
Neither uses cross-group waiting. Pending/unknown completion retains the
physical work; it cannot trigger CPU replay or host publication.

## Twelve-buffer capsule layout

All words are 64 bits. Sizes are checked before allocation. These are
compiler-private images, not a serialized user input format.

| Slot | Contents |
| --- | --- |
| 0 | Immutable ordinary entry packet |
| 1 | Ten terminal packet words per lane |
| 2 | Seven-word trace events per lane, bounded by config[3] |
| 3 | `[lanes, lane_fuel, lane_transitions, lane_trace_events, site+1, lo]` |
| 4 | Private owned heap per lane |
| 5 | Private cleanup transcript per lane |
| 6 | Private continuation task per lane |
| 7 | Untouched native task seed |
| 8 | Untouched native heap seed |
| 9 | Frontier prefix marker followed by per-lane carried sum seeds |
| 10 | Frontier round-start active-writer flags |
| 11 | Fourteen control words per lane; word 9 is the resolved boundary index |

Task layout remains the existing 64-frame continuation plus 18 counters.
Control words 0/1/2 are lane mode/site/depth; 6 is completion kind (1 backedge,
2 terminal, 3 quantum cut); 8 records branch entry. A completed capsule is
validated against the private seed and capacities before any image is merged.

Normal-map merge adjusts child frame IDs, root block activations, source fuel,
physical transitions and ordered Demand/Complete events in lane order. Local
allocation IDs are remapped as dead descriptors; live owners cannot escape a
certified iteration. Heap relocation is resolved through descriptors. A region
whose aggregate source budgets do not fit keeps the untouched native checkpoint.

The host entry commits mutable buffers only after normal return, exactly as
`oracle.rs::observe_entry` and HA-1.1 `eval/host_read.rs` do. Traps, fuel/trace cuts
and backend failures leave host buffers unchanged. Internal continuation state
still carries its exact source prefix, including writes in a failing iteration.
The reconciled section 3 of `gnet-parallel-v01.md` follows this oracle boundary;
P2 does not change the oracle or P1 semantics.

`First` uses the minimum boundary index. Successful preceding lanes may commit,
then the native frame replays the winning iteration, preserving traps before an
exit, writes within a trapping iteration and final cleanup. Source fuel or trace
cuts can select an earlier iteration; in that case the original checkpoint is
replayed. An early exit never publishes later iterations.

`Frontier` first maps against the round-start snapshot, then reconstructs earlier
active own-writers and carried sum prefixes for a second map. Only certified
constant monotone stores and non-scan lifted integer sums are admitted. Failed
probes or overflowing prefixes replay from the native checkpoint. Other carried
representations, runtime guards and compound first/frontier forms are explicit
misses whose serial control stays compiled. P8 can supply their collective bridge.

## Interfaces for P5 and P6

`RegionBackend::execute` consumes a compiler-issued capsule plus initialized
images and group count, and returns images only after confirmed completion.
The native frame and CPU lane model are explicit and cannot be confused with
physical Metal/CUDA execution. The physical adapters retain the existing ABI5
entry and add a separate region path; no old geometry is widened implicitly.
P5 may encode the same capsule and resolution in device epochs; P6 may select
CPU/native, M or a family among checked alternatives. Neither should re-infer
legality from an observed shape or a cost estimate.

## Current ceilings and acceptance

Private heap/task/trace snapshots deliberately bound each chunk by the configured
storage budget (eight simultaneous image copies), with at most 1,024 lanes.
Frontier reconstruction currently performs quadratic prefix materialization.
There is a host fence and full readback per capsule, and physical pipelines and
worksets are prepared per invocation. This is the exactness path, not a D5 parity
or residency claim. Resident scratch, trace-off specialization, narrowed frames,
collective prefixes and prepared work reuse require measured follow-up within P2
or integration with P5/P8. The hardware and performance gates remain separate
from compilation or CPU model tests.
