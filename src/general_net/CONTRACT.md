# GNET core boundary contract 1

This is the bounded implementation contract for `general_net`.
Its general semantic obligations are in [Semantic obligations](#semantic-obligations); its performance acceptance is `docs/performance-acceptance.md`.
The existing MIR, typed-net and MIR-net v1 formats remain unchanged.
This module has its own `CONTRACT_VERSION = 1`; it has no serialized input format yet.

## GNET-01 freeze (contract version 1)

The following interfaces are frozen at `CONTRACT_VERSION = 1`; any change to them requires a version bump and rerunning the GNET-01/02 suites.
- Source to net: `Net::lower(CheckedProgram, Limits)` and raw `Net::encode(Program, Limits)`.
- Net verification and the net to compiled boundary: `Net::verify` and `Net::compiled_mir(proof)`.
- Rule interface: the 33-rule inventory in `coverage.rs` (`ALL_RULES`, `operation`, `terminator`, `operands`, `Observation`), with `F64` appended last and refused by the oracle, pinned by `issue_gnet_01_all_current_mir_variants_have_explicit_inventory_entries`.
- Observations: `oracle::Event` (`Demand`/`Complete`, frame, activation, token, rule), `oracle::Observation` and the typed `ToolFailure` classes below.
- Coverage: the table below, which `tests/issue_gnet_02_corpus.rs` checks against every runnable fixture and example.
This freeze does not change any acceptance gate, and it is not GNET-03/04/05 evidence.

## Checked boundaries

`Net::lower(CheckedProgram, Limits)` rechecks ownership and uses the existing
compositional source-to-MIR lowering. `Net::encode(Program, Limits)` checks
structural budgets before cloning, verifies MIR, and reuses `mir_net::Net` typed
operand wires. It covers every current MIR operation and terminator with an
exhaustive Rust match in `coverage.rs`. A newly added MIR variant requires an
explicit rule, operands and observation classification. Variable record/variant
ports and terminator ports count against the wire budget, including the record
ports that the foundation's early limit estimator does not count.

Every instruction has an explicit agent, span, typed result, ordered operands,
call classification and one incoming/outgoing block-position token. Each block
has an explicit control route and typed incoming block parameters. Verification
reconstructs these annotations from the verified graph and compares them exactly;
forged effects, token skips, cleanup, calls and branch destinations fail. The
foundation checks missing/duplicate/extra wires, types, dominance, loan/owner
flow, call signatures, entry routes and cleanup. Foundation graph fingerprints
are in-process stale-proof checks, not cryptographic provenance or serialized
proofs. This module does not strengthen that trust boundary by claiming hashes
prove semantics.

`Net::verify` produces an opaque proof. `Net::compiled_mir(proof)` checks current
annotation snapshots and the foundation proof, then reconstructs typed MIR for
the existing native compiler or the program-specific scalar emitter.
Mutations invalidate the proof. Extracting verified MIR is a real compiler
boundary, not evidence of native execution, concurrency or GPU implementation.
A re-encoded modified graph can be structurally valid and compute a different
answer: structural validity is not a rewrite equivalence certificate.

## Activation and ordered observations

A dynamic activation consists of a call-frame identity, block identity and
activation generation. The block token advances only after the current agent
completes. A call suspends this advancement until its child returns. Goto resolves
all incoming arguments before assigning target block parameters. Branch gives
an activation token only to the selected target. Backedges create a new block
activation, not a graph-unrolling or totality requirement. Return resolves its
value, performs the existing normal cleanup sequence and resumes its caller.
Source traps stop the frontier and do not complete the trapping agent or perform
language cleanup. These are deliberately sequential rules; effects and all
potentially conflicting accesses retain their existing order.

The CPU oracle records `Demand` immediately before each non-Tick agent/control
transition and `Complete` immediately after its successful transition. Calls
complete after their child returns. Events identify the frame, activation,
static token and rule; the checked token identifies its retained source span.
`Tick` is an invisible source-fuel bookkeeping step and emits no logical event.
This is a separately versioned GNET event contract, not a change to typed-net v1
events. Physical device work never becomes a logical event by assumption.

Foreign declarations and their calls are explicit `Host`/`HostBoundary` entries.
Their source oracle is not treated as a purity assertion or device implementation.
Unknown intrinsic effects retain the ordered token. No owner duplication,
speculation, concurrent mutation, runtime placement or failed-guard publication
is implemented by this slice.

## Structural source coverage and remaining executable coverage

The source inventory below is over the actual current `spec/grammar.ebnf`, `front::ExprKind`, `front::StmtKind` and MIR.
Structural lowering preserves every form the existing checked source and MIR support.
This does not narrow the required v0.1 GPU coverage denominator.
`tests/issue_gnet_02_corpus.rs` runs every runnable source under `tests/fixtures` and `examples` through the reference evaluator, the MIR evaluator and the CPU oracle.

| Source form | Checked structural rules/elaboration | Independent CPU oracle |
| --- | --- | --- |
| i32/u32/Bool literals, scalar names and grouping | I32/U32/Bool/Read, grouping elaborated | implemented |
| Unit returns and direct Unit calls | Call/Return | implemented |
| unary, checked arithmetic, comparisons | Unary/Binary with exact operator spans | implemented |
| `and`/`or`, `if`/`else` and else-if | selected Branch/Goto routes | implemented |
| `let`/`var`, scalar assignment, expression statements | Bind/Store and Tick | implemented |
| counted `for`, `while`, early returns | Goto/Branch/ForNext/Return, block parameters | implemented |
| direct calls, recursive and mutually recursive functions | Call and explicit frame continuation | implemented |
| wrapping integer intrinsics | Call, retained argument order | implemented |
| fixed owned arrays, allocation intrinsics, index/len/store | Array/ArrayRef/Index/Length/StoreIndex/Call | implemented (i32/u32/f32) |
| moves, affine loans, normal cleanup | Move/MoveField/ArrayRef/RecordRef/Drop/DropValue/Return | implemented |
| records and field projection, including f32 members | Record/Field/MoveField/RecordRef | implemented |
| variants, Result constructors, match and `?` | Variant/VariantTag/Payload/TakePayload, Branch/Return | implemented |
| bytes/string literals, UTF-8 decode/encode/slices and their Result | Blob, typed Call and variant rules (`Ok = 0`, `Err = 1`, payload-free `Utf8Error`) | implemented |
| borrowed views, strided reborrows, view mutation and return-from | typed ArrayRef/Call/Index/StoreIndex/Length with MIR loan facts | implemented (integer views; f32 views do not check) |
| generic functions/records, closure captures and single calls | checked instantiation and lambda lifting into functions, owners and direct Call | implemented through the generic rules |
| modules/imports, visibility and qualified names | checked name/instance elaboration | no separate runtime agent |
| quantities/units/frames and nominal robotics values | checked elaboration; frame helpers are synthetic functions | implemented; a trap inside a frame helper reports its outermost call site |
| checked unit/angle conversion intrinsics | typed Call with checked scale/angle identity | implemented (exact scale; shared certified angle primitive) |
| f32 scalars, arrays and nominal members | strict-f32-v1 primitives | implemented |
| system/component/robot/cell/port/manifest declarations | checked compile-time system elaboration | no GPU work claim |
| actual C/foreign IO/device operations | explicit host call boundary and ordered token | explicit host boundary: refused before any demand |
| indirect calls/escaping closures | not in the current admitted grammar | must be added to coverage if v0.1 admits them |

All 32 current MIR rules have explicit inventory entries, and the CPU oracle supports all 32.
The only oracle refusal of checked source is a declared foreign call (a host boundary, not a pure-core rule).
A program over the structural net budget (`Limits`) is refused by `Net::lower` with `Error::Limit`.
The program-specific scalar emitter lowers the scalar/control subset to CPU/CUDA/Metal source; emitted CPU execution is tested, and GNET-03 device execution is described in its own section below.

## Bounded CPU oracle

`oracle::observe` accepts only a current proof and admits all functions before
the first demand. A declared foreign function, or a type without an oracle
representation, fails with typed `ToolFailure::Unsupported` and the existing
`UNSUPPORTED_FEATURE` tool diagnostic, including in uncalled functions. There is no
whole-region fallback. The machine executes explicit activation routes and
primitive agents using at most 64 heap frames, without recursive host calls or
calling either existing evaluator. Scalar operators use explicit checked or
wrapping arithmetic, including MIN%-1 = 0, and source operator spans.

Fuel charges retained MIR Tick agents. Per `spec/bootstrap.md`, evaluating a bare
owner name in a consuming return context charges its expression step before
transferring the owner slot. The source evaluator charges before taking the slot;
the MIR builder retains Tick before Move, and checked-source net lowering does
not strip it. If that Tick exhausts fuel, no owner transfer or return cleanup
occurs. Previously completed cleanup remains observable. Explicit move, scalar,
borrowed-view and Unit return charges are unchanged. Raw MIR encoding preserves
all supplied ticks and makes no source-step equivalence assertion.

This deliberately corrects the historical bare-owner return undercount without
amending the frozen specification: successful runs gain one source step per
executed bare owner-name return. The existing `[3,5]` maker golden in
`tests/issue_m1_01_owned_return_fuel.rs` changes from 11 to 12; its explicit-move
12 and borrowed-view 17 goldens remain unchanged. The new return-fuel regressions
pin the nested cleanup fixture at 22 (historically 21), and paired nested returns
at 19 (historically 17). Untaken paths add no runtime steps. Retaining the Tick
also changes derived MIR/net digests, instruction/token positions and transition
counts; rebuild and reverify source-bound artifacts. Existing supplied MIR Tick
semantics and wire/ABI versions do not change.

A separate finite transition budget stops malformed/expensive
administrative work; a finite trace budget bounds event storage. Exhaustion
reports a tool error or the exact source-fuel limit, never divergence. Trace/frame
reservation failures are tool errors. Defaults: 100,000 source steps, 1,000,000
transitions and 1,000,000 trace events. There is no semantic cleanup on a trap or
limit. Owned arrays have explicit arena handles and affine place slots. Borrowed read/mut
parameters do not acquire ownership. Move removes the owner slot; normal Drop,
reassignment and Return record source cleanup order, while traps and limits retain
only completed cleanup. Per-array length1,000,000 and source live-elements2,000,000
limits retain ARRAY_SIZE_LIMIT; a separately configured lower arena cap reports a
typed allocation/tool limit. IDs are never recycled in an observation. Named
scalar entries accept runtime arguments and retain the source entry fuel convention.
Max live frame count is independently observed.

The oracle's typed failure field distinguishes unsupported rules, transition
exhaustion, trace exhaustion, allocation failure and invalid internal state.
Compatibility errors reuse `UNSUPPORTED_FEATURE`, `RESOURCE_LIMIT` and
`MIR_INVALID`; no new public diagnostic codes or shared registry changes are
introduced by this private slice.

## Simulation, compiled execution and progress obligations

For the implemented scalar slice, relate source control/local values/call stack
and ordered observation prefix to the selected activation, scalar place/value
store, explicit frame continuations and completed-event prefix. Tick bookkeeping
preserves this relation except for the specified source-fuel counter. Primitive
agents correspond to source operations; branch and backedge routing correspond
to selected source control steps. Calls/returns relate frames and continuations.
Differential tests check values, exact trap categories/spans and source steps;
the event test checks nested demand/completion and selected-branch behavior.
These checks are evidence, not a completed mathematical simulation proof.

GNET-02 evidence for owners, loans, aggregates, views, text, units, cleanup and all ordinary source:
- `tests/issue_gnet_02_corpus.rs`: every runnable fixture and example agrees in all three engines on result, source steps and cleanup, at full fuel and one step short of its own use.
- `tests/issue_gnet_02_generated.rs`: 240 seeded well-typed compositions of 16 fragment kinds (traps of every integer category, the 64-frame limit, untaken traps, short-circuiting, borrowed mutation through views, scoped cleanup, early returns) agree in all three engines, plus every fuel prefix of a sampled subset.
- `tests/issue_gnet_02_native.rs` (`native` feature): the LLVM 21 native CPU lowering of `compiled_mir` agrees with the oracle on status, value, trap category and span, tool code, steps and cleanup for the corpus and the generated programs; a native-only resource bound (for example `NATIVE_TRACE_LIMIT`) is counted as an explicit refusal.
Foreign-visible prefixes remain outside the oracle (host boundary).
These are differential evidence, not a completed mathematical simulation proof. Before
GNET-03, establish program-specific compiled CPU/device simulation, live-frame
and worklist storage bounds, finite quanta, fence-held leases, completion and
progress. Before GNET-04 speculation, reserve serial-path resources, preserve
the oldest frontier's progress and source-ordered commit, and forbid later
trap/write/cleanup/resource failure publication while an older call diverges.

Concrete counterexamples pinned here: an untaken trap; two calls that trap in
different categories; an oldest infinite loop followed by a trapping call;
unselected short-circuit RHS; MIN%-1; frame-depth exhaustion; source fuel versus
administrative/trace exhaustion; branch/token/effect/cleanup/wire mutation; and
aggregate/host work that must not be silently admitted into the scalar oracle.
Future acceptance still requires contested endpoints, generation reuse,
allocation rollback, cancellation, fence/quiescence and speculative scratch
exhaustion. Sequential termination under finite tool bounds establishes a
bounded prefix, not source termination or a GPU fairness theorem.

## Program-specific compiled scalar execution

`compiled::emit` consumes a current proof and admits the entire checked candidate.
It rejects arrays, aggregates, views, foreign boundaries and unknown intrinsics,
including in unused functions. CPU/CUDA/Metal share checked scalar operations
and directly generated block bodies, branches and loop backedges. Only direct
calls split blocks into explicit continuation states. The switch selects those
program-specific control continuations; there is no runtime graph opcode buffer
or primitive opcode interpreter. Source recursion uses a private bounded64-frame
stack, not device recursion. No termination proof is required to admit recursion;
finite source/transition/event/depth limits report distinct statuses.

Value IDs and places map densely into bounded frame slots; sparse IDs never size
arrays. Default bounds:128 values/function,64 places/function,4096 continuation
states,128KiB full private frame stack/root,1MiB generated source. Explicit
profiles may request generated source up to the shared4MiB GNET ceiling; the
emitter, Metal admission and both CUDA compiler paths enforce that same bound.
Ordinary device-source limits remain unchanged. The current ABI
accepts positive fuel<=1M, transition and trace caps<=1M, including zero tool caps.
Four device buffers carry runtime input words,8-word observations,7-word events
and config[count,fuel,transitions,events]. Host code must bound root count and all
buffer sizes before dispatch. Observation words are status,value,span_start,
span_end,source_steps,physical_transitions,max_frames,event_count. Status ordinals:
0success,1overflow,2division-by-zero,3fuel,4depth,5transition,6trace,7invalid-state,
8invalid-input. Event words are kind(0Demand/1Complete),frame,activation,function,
block,position,Rule declaration ordinal (the explicit 32-rule inventory). This
private contract is version1 and is not a shared serialized IR/schema change.

The emitted C++17 CPU path runs under UBSan and matches every observation/event
against the independent net oracle for dynamic recursion, loops, phi routing,
selected branches, Unit calls, exact integer edges and ordered traps/limits.
This supplies executed compiler lowering evidence for the scalar slice only.
The minimal recursive GPU validation bundle has standalone CUDA/Metal runners,
fences,10s dispatch deadlines, retained leases/quarantine and exact oracle trace
comparison. Runtime compiler/module-load/transfer/fence costs remain separate.
CUDA runner builds and host comparison tests pass locally; actual NVRTC/Metal
compile and GPU dispatch are pending on authorized hosts. Nine roots can be
submitted together, but each recursive root is currently serial. There is no
automatic independent-child worklist extraction, placement/speculation optimizer,
general-device coverage, calibrated performance or completed GNET03/04 acceptance.

## GNET-03 compiled device execution (ABI5)

`device_host::Plan::from_source` compiles checked ordinary source to one program-specific ABI5 kernel per target (`invar_gnet_children_v5`) and runs it on the CPU group model, CUDA or Metal through the same fenced session.
There is no net interpreter on the device: the kernel switches only on the frame's own generated continuation states, and every independent call site has its own generated projection and discovery routine.
Thread 0 runs the root continuation for a finite quantum per round; up to 8 lanes run discovered children in private slots, heaps, traces and cleanup buffers; barriers separate root mutation, child execution and publication; a host fence separates submissions.
Children commit in source order only at their demand; an owner child, a stale generation, a failed guard or a child over the remaining budget is replayed by the root, never published.
A full worklist serializes the call (`capacity_serialized`), it never fails the source program.

GNET-03 admits i32/u32 owned arrays anywhere inside the program: the root owns the device heap and cleanup buffers, and the decoded observation includes the completed normal cleanup prefix.
P0b extends this boundary with ABI13 for f32 scalars, read/mut contiguous i32/u32/f32 arrays and read flat scalar records. Flat records and owned i32/u32/f32 arrays may also be returned. P5b validates returned allocation IDs, live descriptors, kinds, lengths, heap ranges and canonical words before copying the result or publishing mutable arguments. ABI5 retains its scalar packet and launch geometry.
Variants, nested/array-containing records, views, text, take entry parameters and unit intrinsics remain `unsupported_type`/`unsupported_call` refusals; programs over the compiler or net structural budgets are `compiler_structural_budget`/`net_structural_budget`.
`device_host::refusal_class` is the stable class; the `gnet-device` receipt records `placement` with `mode: forced-device`, `executed`, the refusal class and `fallback: none`.
Forced device mode never substitutes another engine. ABI13 executes exact integer-software binary32, including subnormals, after the live D1 preflight. Native float admission stays disabled; receipts record the probe and exact software fallback, with no CPU replay. Automatic placement is GNET-04. See [Entry ABI13](#entry-abi13).

Device statuses 9-13 decode as IndexOutOfBounds, ArraySizeLimit, HeapLimit, AllocationLimit and CleanupLimit.
HeapLimit equals the oracle's allocation failure when the oracle arena uses the device element cap.
AllocationLimit (allocation IDs are never recycled) and CleanupLimit are device storage exhaustion with no oracle analog: they publish no value and are counted separately, never as agreement.
The general-net CUDA module cap is 16 MiB (NVRTC 12.9 emits up to about 8.5 MiB for the adversarial set); other CUDA modules keep 1 MiB.
The general-net NVRTC subprocess deadline is 300 s; uncached NVRTC 12.9 (sm_120) takes 13-25 s for ordinary fixtures and 127 s for the widest adversarial program, while other CUDA modules keep 20 s.
NVRTC 12.9 reuses the CUDA ComputeCache, so compile-time evidence is recorded with `CUDA_CACHE_DISABLE=1`.

CUDA is compiled ahead: `invar build PATH --target cuda --entry ENTRY --sm SM --output FILE` runs NVRTC at build time (in the bounded `invar-nvrtc` helper) and writes one `device_host::cuda_aot` container.
The container is magic `INVGCUDA`, format version 1, header and image lengths, a SHA-256 over all of them, a canonical JSON header and one CUBIN.
The header records the program (entry, ABI, kernel, SHA-256 of the ordinary source and of the generated device source), the target (`sm`, `min_driver`) and the toolchain (NVRTC version, its exact options, the invar version), plus the image length and SHA-256.
A production run is `invar gnet-device PATH ENTRY cuda ARGS CONFIG --artifact FILE`: it opens the device through the Driver API only, reads the container as untrusted bytes and loads the CUBIN with `cuModuleLoadData`; the receipt's `image` block repeats the header with `runtime_compiler: none`.
Before the driver sees the image, the loader bounds the sizes, checks both digests and the canonical header, checks the CUDA ELF64 header (class, endianness, `EM_CUDA`, executable type, the SM in the ELF ABI 7 or 8 flags) and that every table and file-backed section and segment lies inside the buffer, because `cuModuleLoadData` takes no length.
It then requires the image to be the build of the program being run (entry, ABI, source and device-source digests), an exact SM match, and a driver no older than the image's NVRTC.
Each failure is a typed `image` refusal with `fallback: none`; a driver rejection is `Rejected`, and nothing is ever recompiled at run time.
The container carries no PTX, so the driver never JIT-compiles it; an image for another architecture is rebuilt with `invar build`, not translated at run time.
The digests are integrity checks, not signatures: authenticity is the deployer's responsibility.
Forced CUDA without `--artifact` refuses; the CUDA hardware suites build each plan's container in the harness (the build step) and load it through the same untrusted-bytes path.
`issue_gnet_cuda_aot_gpu` runs the production command with every NVRTC library name resolving to a tripwire that exits the process on load, checks the oracle's exact result and that `ldd` lists no NVRTC, and uses `invar build` under the same tripwire as the positive control.
Other CUDA routes (M3/M4 `run --engine cuda`, tensor and transpose source routes) still compile with NVRTC at run time; this section covers the general-net route only.

Evidence (`tests/issue_gnet_03_device.rs` by default, `issue_gnet_03_gpu` and `issue_gnet_03_metal_gpu` on hardware, all over `tests/support/gnet_device.rs`):
every one of the 326 runnable corpus programs is either device-executed with exact agreement with the CPU net oracle (status, value, trap span, source steps, transitions, frame peak, every Demand/Complete event and the cleanup prefix) at full fuel and one step short, or refused with a typed class.
Adversarial ordinary-source cases with runtime inputs cover data-dependent trip counts, divergent recursive frontiers up to width 8, a worklist wider than its bound, an early trap in one lane behind a slower or fuel-exhausted older call, fuel cut inside rounds and epochs, device owners with traps and heap exhaustion, allocation-identity exhaustion, cancellation after submission, and unfenced reclaim quarantine.
The general corpus is mostly sequential (only one of its admitted programs has an independent call site), so the parallel slice `tests/fixtures/gnet_parallel` adds six ordinary-source programs with four runtime inputs each: eight independent calls, divide and conquer, a four-way branch tree, a frontier discovered at run time, mixed owner/scalar fan-out and a trap behind slower siblings.
`tests/support/gnet_parallel.rs` runs every slice input at capacity 8 and as a capacity-0 serial control on the CPU group model, CUDA and Metal; every fixture must have fork candidates and spawn children, the slice must reach peak width 4 or more, and wide and serial runs must trap identically.
Device-side fault injection uses the test-only `gnet-fault-injection` Cargo feature: idle group thread 31 calls a non-inlined `gnet_fault_inject` device function that rewrites shared slot words.
Every hook call sits between two group barriers that every thread reaches, so no root or child thread accesses a slot while it runs; the injected state is adversarial, but the run stays data-race free under the CUDA, Metal and C++ memory models.
The CPU group model run of the injection suite is checked under ThreadSanitizer, and a mutation that drops the barriers (a plain write racing the child's read) is reported.
`StaleCompletion` (publication phase) stamps done slots with the previous generation just before the next root commit check.
`PoisonedStale` (child phase) takes only fully published slots, stamps the previous generation and corrupts the value word, so only a commit that ignores the generation could publish it.
`ContestedClaim` (child phase) claims free used lanes with an identity no call site matches and releases them one child phase later; `CancelRace` (child phase) cancels lane 0 while its child is unfinished.
Every slice run under every fault must agree exactly with the oracle, and its counters must move in the fault's direction against the same uninjected run (stale: fewer commits and more replays; contested: different spawn/commit/serialization counts; cancel: more spawns).
Without the feature `Fault` is uninhabited and the hook text is not compiled: the default suite checks that the `invar` binary and the emitted CUDA/Metal source of every admitted corpus program lack it, and the CUDA suite checks that every slice program's production compiled-ahead image lacks the hook symbol while every injected image keeps it.
Metal libraries are compiled from the emitted source at run time and the evidence Mac has no offline Metal toolchain, so Metal absence rests on the source and binary checks.
These are differential results, not a simulation proof; no performance, placement or parity claim is made.

## Semantic obligations

Let source configuration C = (control, store, ownership, ordered observation prefix), and executable-net state N include graph, demanded frontier, leases and ordered commit state.
Define a relation R(C,N).
For each logical net transition N -> N', establish either an invisible administrative step preserving R, or source steps C ->* C' with R(C',N').
Establish an equivalent simulation for compiled CPU and device steps against N.
Progress under the specified fair schedule and divergence-sensitive behavior is required; forward simulation alone can hide starvation.

Observable agreement includes normal values, first source-ordered trap and source span, visible write/foreign-effect prefix, normal ownership cleanup, and the contract's logical Demand/Complete events.
GPU physical work is not itself a logical Demand event.
Do not equate evaluator fuel with net rewrite count.
Precisely distinguish semantic limits from backend exhaustion or user cancellation; exhaustion is never proof of divergence.

Demand activates only the selected branch.
Ownership/region tokens forbid duplication of affine owners and order conflicting memory operations.
A device task holds input leases through the fence; FFI consumes the appropriate ordered host token.
Unknown overlap serializes or uses a checked runtime guard; a failed guard cannot publish a speculative write.
No unchecked foreign purity annotation authorizes speculation.

### Expose independent work without termination proofs

For `let a = f(x); let b = g(y); a + b`, independent observationally pure calls may execute concurrently even if their termination is unknown.
The proposed rule is bounded speculation with source-ordered commit:

1. Run later work in finite preemptible compiled work quanta on stable inputs and isolated private scratch. Hold its result or trap.
2. Always give the oldest source frontier progress. If f returns, g may become the frontier; if f traps, discard g; if f diverges, never publish g.
3. No speculative writes, foreign effects, cleanup, or logical demand events become visible. No affine ownership is copied. Uncertain shared resources serialize.
4. Speculative scratch cannot deplete the resources reserved for the serial path. Scratch exhaustion suspends/discards speculative work rather than failing the source computation. Record semantic fuel/resource accounting in source order; any replay for accounting must itself be effect-free and explicitly specified.
5. Kernels have finite quanta so a speculative divergent call cannot prevent the oldest task progressing or cancellation. Reclamation waits for device completion.

This is a proof obligation, not an established theorem.
Counterexample to catch: f loops forever and g immediately traps or fills speculative memory; reporting g's trap/OOM as the source result is incorrect.
Refusing all possibly-diverging pure calls solely for lack of a termination proof would miss the intended generality.

## Compiled dynamic work

Emit program-specific branch, loop, frame/continuation and worklist code for dynamic source behavior.
A bounded worklist may use epochs when its algorithm needs global synchronization: discover candidates, check ownership/dependence and capacity, execute nonconflicting work, publish source-ordered observations, then retire state after all readers and device fences complete.
It is not a generic net-opcode interpreter.
Shared-edge writes and stale generations still require checked guards; allocation failure must leave a valid continuation and no partially published effects.
Deferral cannot starve an older source frontier.

Completion requires resolved root/commit frontier, empty ready/deferred/host queues where used, no in-flight work and completed streams.
An empty local queue alone is insufficient.
CPU reference and compiled CPU/GPU paths share the net's logical rule semantics; their physical scheduling and storage strategies may differ.
If a guard fails before execution, select the compiled alternative.
Switching after progress requires transferring the exact live continuation, ownership and effect frontier, with no replay of visible effects.

## Entry ABI13

An expanded entry uses ABI13; scalar integer entries keep ABI5.
Parameters are i32/u32/Bool/f32, `read` or `mut Array<i32|u32|f32>`, or `read` flat records of i32/u32/Bool/f32.
Flat scalar records and owned `Array<i32|u32|f32>` results are supported.
`take` parameters, variants, nested records and records containing owners are typed refusals.
At most eight parameters and sixteen fields per record are admitted.
Field order is checked-source declaration order, never host struct padding.

The twelve physical buffers, task counters, group of 32 threads and kernel symbol remain unchanged.
Buffer 0 contains parameter words, allocation/live counters, then a canonical owned heap image.
Root initialization copies the image once before the first source step.
Each scalar/field/element occupies an eight-byte word: sign-extended i32, zero-extended u32/f32 bits, and Bool 0/1.
Record and array parameters are stable allocation IDs into the existing checked heap layout.
Records carry the existing per-field presence mask.

Host entry uses safe `eval::Value` containers.
Shared/exclusive RefCell leases prevent conflicting caller access while a session owns borrowed arrays.
Exact read aliases share a device owner.
Any duplicate identity involving `mut` is refused conservatively, including empty owners; partial or strided views are not admitted.
There is no raw-pointer, external-stride or caller-supplied record-padding ABI.
Wrong field count/types or a strided Value produces `wrong_layout`; capacity/length produces `oversize`; mutable alias produces `alias`; unsupported owners produce `unsupported_type`; an already borrowed container produces `borrowed`.
Each refusal precedes backend allocation.

Device writes remain private across continuation epochs.
Only a successful source return after the completion fence reaches publication.
Every destination descriptor, length, kind, live bit and scalar word is checked before the first mutable host write.
Trap, fuel exhaustion, cancellation, malformed completion and unknown completion publish no host array changes.
Unknown completion retains the existing quarantine and forbids replay.
Array transfers and readbacks are copying; native/Python zero-copy remains a separate gate.
Oracle receipts hash bounded inputs and alias topology before mutation, then retain an independent snapshot of flat record results.
Returned arrays are copied into fresh host storage only after their issued handle, kind, live flag, length, heap range and canonical element words validate.
A malformed result prevents mutable-argument publication as well.

Strict-f32-v1 uses the existing independently tested integer-software binary32 operations for every ABI13 operation on both targets.
This is the exact fallback path, including all subnormal inputs/results, signed zeros, NaNs and explicit fma.
Each physical backend runs the existing cached D1 live-device probe before preparing ABI13 work and records its completed report.
Native floating arithmetic is never admitted; every operation uses the exact software fallback, including when the probe detects flushing or contraction.
A failed or unknown probe completion refuses preparation through the existing poison/quarantine gate.
There is no hardware-float fast path or CPU replay.
Metal disables fast math and emits `#pragma METAL fp contract(off)`; CUDA disables contraction and FTZ and requests precise division/sqrt.
Receipts identify software arithmetic, absence of native admission and the subnormal path.
CPU net agreement compares exact words, steps, trap spans, transitions, Demand/Complete events and cleanup.

`array_limits` defines a 2^24 per-array cap and a 2^28 live-element ceiling.
A 2^24-element host f32/u32/i32 array is 64 MiB, or 128 MiB in GNET's eight-byte carriers.
The fixed packet plus heap requires about 256 MiB for one full-size input, before bounded frame/trace/scratch.
Callers must explicitly raise `Limits.owned.elements` and `Limits.buffer_bytes`; the default remains conservative.
The full-size test also sets `Limits.private.slots = 0`: unchanged speculation storage otherwise replicates the owner heap per private slot and can exceed its separate scratch budget.
Source fuel, trace and per-plan storage limits are unchanged.

The hard storage ceiling is 8 GiB, following the 2^28 device caps policy.
Before allocating the packet, host admission also checks four payload copies plus the retained input against that ceiling, covering adapter storage, byte capture, decoded checkpoints and readback overlap.
Metal checks live maxBufferLength and recommended working set minus current allocations; CUDA checks threefold concurrent storage and free memory less a ten-percent margin.
Checked arithmetic is retained for lengths, byte extents, heap sizes and totals.
Allocation failure remains explicit.
The CPU device model retains its smaller 64 MiB bound; full-size source acceptance and full-size Metal admission are separate tests.

The private `gnet-device` command accepts f32 as `{"bits":1065353216}`, arrays as JSON arrays of exact typed elements, and flat records as arrays in declaration order.
f32 array elements use the same bits object.
Expanded receipts include `arguments_after` after reclaim releases all leases.
The CLI's OS argument-size limit is separate from the Rust host API's supported array size.

P5b packages the exact D1 probe MSL as a second offline library for ABI13.
Build and load derive the same inventory from the admitted artifact and reject missing, stale, extra or wrong-entry libraries before installation.
ABI5 keeps its one-library container.
The container format and production prohibition on runtime source compilation are unchanged.

The compiled C++, CUDA and MSL numeric bodies admit `sin`, `cos` and `sqrt`, including ABI13 entries.
`trig_f32.rs` generates the frozen operation sequence over the existing exact integer f32 helpers; its integer square-root primitive rounds correctly without device float arithmetic.
NaNs, signed zeros, infinities, subnormals and huge trig arguments follow `invar-trig-f32-v1`.
The launch, speculation and scheduling structure is unchanged.

## Host-frame body capsules (protocol v1)

`device_host::host_frame::Plan` consumes the verified net's P1a partition and P1b PNF, checks every certificate with `certificate::check_all`, and emits from the same net-derived MIR as the continuation compiler.

The adopted catalogue calls first-exit L15 and monotone frontier L14.
L13 is compaction.
`compiled::body_map::Compiled` exposes sites, explicit admission misses, the MIR contract fingerprint and both generated sources.
`Site` names its `(function, head)` region and Map, First or Frontier shape.
No caller-supplied certificate, task image or device address can construct a host-frame plan.

`Plan::run` owns native serial control, exclusive entry leases, region scratch, fences, validation, ordered publication and replay.
The frame is compiled program-specific C++, through the existing bounded compiler child and reviewed native loader.
It does not evaluate tree nodes or graph opcodes.

### Continuation boundary

A site pauses before its loop branch, after evaluating the first header.
Lane zero begins at that branch.
Later lanes restart at the header with their own induction value; each stops on the backedge before the next header.
After a successful chunk the native frame executes the final header or requests the next chunk.
Calls, branches, inner loops, integer traps, strict software f32, and iteration-local owners use the existing continuation/operation emitter.
The native frame never hands its addresses to a shader.

Two device kernels form one capsule: `invar_gnet_body_map_v1` uses groups of 256 with grid-stride coverage beyond 65,535 groups; `invar_gnet_body_resolve_v1` uses one 256-thread tree min reduction to find the first non-complete lane.
Both execute in encoder/stream order under one observed completion fence.
Neither uses cross-group waiting.
Pending/unknown completion retains the physical work; it cannot trigger CPU replay or host publication.

### Twelve-buffer capsule layout

All words are 64 bits.
Sizes are checked before allocation.
These are compiler-private images, not a serialized user input format.

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
Control words 0/1/2 are lane mode/site/depth; 6 is completion kind (1 backedge, 2 terminal, 3 quantum cut); 8 records branch entry.
A completed capsule is validated against the private seed and capacities before any image is merged.

Normal-map merge adjusts child frame IDs, root block activations, source fuel, physical transitions and ordered Demand/Complete events in lane order.
Local allocation IDs are remapped as dead descriptors; live owners cannot escape a certified iteration.
Heap relocation is resolved through descriptors.
A region whose aggregate source budgets do not fit keeps the untouched native checkpoint.

The host entry commits mutable buffers only after normal return, exactly as `oracle.rs::observe_entry` and HA-1.1 `eval/host_read.rs` do.
Traps, fuel/trace cuts and backend failures leave host buffers unchanged.
Internal continuation state still carries its exact source prefix, including writes in a failing iteration.

`First` uses the minimum boundary index.
Successful preceding lanes may commit, then the native frame replays the winning iteration, preserving traps before an exit, writes within a trapping iteration and final cleanup.
Source fuel or trace cuts can select an earlier iteration; in that case the original checkpoint is replayed.
An early exit never publishes later iterations.

`Frontier` first maps against the round-start snapshot, then reconstructs earlier active own-writers and carried sum prefixes for a second map.
Only certified constant monotone stores and non-scan lifted integer sums are admitted.
Failed probes or overflowing prefixes replay from the native checkpoint.
Other carried representations, runtime guards and compound first/frontier forms are explicit misses whose serial control stays compiled.

### Region backend interface

`RegionBackend::execute` consumes a compiler-issued capsule plus initialized images and group count, and returns images only after confirmed completion.
The native frame and CPU lane model are explicit and cannot be confused with physical Metal/CUDA execution.
The physical adapters retain the existing ABI5 entry and add a separate region path; no old geometry is widened implicitly.
No consumer re-infers legality from an observed shape or a cost estimate.
