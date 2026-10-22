# Loaded numeric owned CPU host (ABI10)

This is a bounded, source-issued Rust host for the existing compiled ABI10 CPU
machine. It loads and calls generated `gnet_run`; the graph oracle is only an
independent test comparator. It does not complete all general-net or v0.1 work.

The runtime baseline is main `a0f4c96c13f3c25b496d2c33b47c20809e7387ea`.
The ARM corpus repair is separate: it keeps the numeric corpus portable across ARM and SSE hosts.
The implementation's changed-file hashes and check scopes were recorded in a
separate verification record. A local test snapshot is not a remote commit or a
hardware qualification receipt.

## Admission and ownership

`numeric_owned_host::Plan::from_source` accepts source bytes, a named entry,
provenance limits, fixed owned-storage limits, and a trusted local C++ compiler.
The existing `compile_owned_numeric` path must issue a CPU ABI10 artifact.
External arguments remain exact `i32`, `u32`, `Bool`, and `f32` scalar values.
Arrays are created by the program. Successful scalar and `Array<i32>`,
`Array<u32>`, and `Array<f32>` roots are supported within that existing admission.

The plan retains a namespaced hidden library, source provenance, checked MIR,
fixed heap/cleanup layout, and source-token map. Its driver and plan are neither
Send nor Sync. Each call has fresh initialized buffers. A successful array root
is copied into a fresh Rust owner only after the entire packet validates; no
native pointer or suspended work handle escapes.

The compiler executable bytes, version output, flags, source artifact, host
build, and exact buffer layout bind the private identity symbol. Library bytes
are checked before and after loading. This is a trusted generated-native-code
boundary, not a sandbox for a malicious compiler or arbitrary shared library.

## Bounds and returning failures

- Source and generated wrapper: at most 2 MiB each
- Library: a capped regular file of at most 64 MiB
- Compiler/version invocation: 30-second deadline and 64 KiB per output stream
- Complete compiled frame storage: at most 128 KiB
- At most 128 scalar parameters; fuel in 1..=1,000,000; transitions and trace
  events at most 1,000,000, additionally subject to total storage admission
- Initialized raw and decoded observation storage is conservatively bounded
  before native entry; excessive requests return an error
- A caller array-element cap tighter than the artifact's fixed native capacity
  is refused before execution. It is never silently ignored
- Semantic traps, source fuel exhaustion and native backend storage limits are
  returned distinctly. A resource-limited observation is not source completion
- A native wrapper protocol error or malformed output closes the plan and
  preserves original source, compiler logs, receipt and bounded failure buffers
- Ordinary successful destruction unloads before removing the private directory

Private directories are created atomically with owner-only permissions on Unix.
Timestamp and atomic nonce disambiguate reused process IDs. Existing directories
are never adopted or overwritten; historical failure evidence survives retries.

## Packet validation

The ten-word header is checked for status, typed root carrier, source trap span,
fuel/transitions/frames/trace bounds, cleanup count and allocation count. Trace
rows must name checked source/MIR tokens and rules, pair Demand/Complete events,
and end in a completed root Return for success. Unused trace storage is zero.

Heap descriptors require bounded extents, admitted kinds (0, 1, 258), valid live
flags, disjoint live storage and correctly typed carriers. Unallocated descriptor
slots and never-used pool tail must be zero. Stable-ID compaction may leave
initialized stale values in previously used pool storage; those values cannot
publish a root. A successful array root must identify the sole live owner.

Cleanup rows bind actual MIR function, declared array place, length and frame,
follow an ordered sequence of eligible cleanup sites, and reconcile with dead
allocation lengths. Unused cleanup slots must be zero. These consistency checks
are not a proof that arbitrary hostile native code executed the source program.

## Executed test scope

The dedicated integration tests exercise real loaded libraries:

- Twelve existing array fixtures and every positive small source-fuel prefix
- Literals, fills, empty/direct integer and floating roots, mutable/read/take
  calls, moves, replacement, compaction, element copies and original traps
- Runtime floating bit patterns including NaN payload copies and signed zero;
  repeated calls return independent mutable Rust owners
- Frame, trace, transition, heap, allocation and cleanup bounds, typed argument
  refusal, source-compiler budgets, and recovery after ordinary resource results
- Two simultaneously loaded artifacts, identity/symbol/digest failures,
  returning wrapper failures, original compiler diagnostics and bad libraries

Unit tests mutate real invocation headers, descriptors, carriers, cleanup and
trace data, shorten buffers, and inject returning malformed/protocol packets.
Compiler deadline/log-cap tests retain partial failure logs. Send/Sync
compile-fail checks cover both plan and driver. Public C-call packaging is
retested after stripping the additive private module from standalone embedding.

Scalar-root cases compare original-source and independent graph observations.
The existing named source API admits scalar roots only: array-root tests use the
independent owned graph oracle and explicit bit/ownership assertions. They do
not claim direct named-source array-return comparison.

## Remaining work and unchanged boundaries

Nominal-owned numeric extensions, continuation/resume, speculation, device
execution, external owner arguments and public C/Python array APIs remain outside
this host. CPU continuation/resume of the same direct owners is a separate ABI11
host; see [remaining work](remaining-work.md). Existing ABI8 and ABI9 hosts and their admission rules are unchanged.
No CUDA/Metal/robot hardware qualification or performance claim follows from
these CPU checks. Global allocator aborts, OS kills, and malicious native code
are not recoverable guarantees of this interface.

## Local qualification result

On Linux x86_64 with Rust 1.95.0, LLVM 21.1.8 and GCC 14.2.0, the
non-overlapping default workspace test commands pass **728 tests per profile**,
with **12 ignored** in each of Debug and Release. The dedicated new host suite
is nine integration and three unit tests. The final returning-fault fixture
expansion was separately rerun in both profiles after the full Debug run began;
that edit changed test coverage only.

Final workspace all-target/all-feature Clippy passes in both profiles. Public
C-call/native packaging tests pass 7/7 in both profiles. The full native-feature
workspace additionally passes **752 tests per profile**, with **13 ignored**, in
Debug and Release. The first full native Debug attempt observed a locally
replaced no-native CLI from a concurrent feature-changing rename check; the
original failure is retained and the full suite was rerun without that
interference, using a fresh temporary directory. No production source change was
needed for this verification-run failure. These are CPU checks, not device
qualification. The repaired private Python wheel passes all 253 cases in a clean
environment with both scalar engines and without workspace LLVM loader paths;
see the wheel document for platform limits.
