# M4-04 - Segmented/nested flattening and stable compaction

Problem/result: compile bounded nested maps/reductions and stable filtering over ragged segments without losing segment boundaries, source order, ownership, or output-size limits. The result must remain an extracted structured operation, not a decorative graph around an opaque loop.

Dependencies: M3-05, M4-01, M5-01.

Contract and implementation: define validated offsets or lengths, total element count, segment order, empty-segment behavior, and the callback's typed interaction contract. A segmented map/reduction may flatten only when offsets are nondecreasing, every access is in bounds, temporary storage is bounded, and the selected reduction law is already admitted. Stable compaction uses predicate flags and an ordered prefix position; retained elements keep source order. Preserve affine loans through extraction, publish fresh outputs only after completion, and retain a serial path. Fuse elementwise producers/consumers only when the M3/M2 rules permit it. The initial representation supports `SegmentedMap`, `SegmentedReduce`, and stable `Compact`; it does not allocate one primitive net per element. Any unknown segment shape, effect, cleanup, or conflict is an admission miss.

Positive cases: no segments, empty interior segments, uniform and highly skewed segments, one dominant segment, ragged totals, and 0/1/50/99/100 percent selection; per-segment maps and lawful reductions agree with the source interpreter, serial extraction, and M3 flat operations. Negative cases: decreasing or overflowing offsets, segment end beyond the input, unstable output order, effectful or trapping callbacks, missing reduction identity, aliasing, unbounded temporary allocation, stale shape/layout policy, and completion failure. Rejections must explain the failed contract and preserve CPU fallback.

Acceptance interface and evidence: add `issue_m4_04_segmented_flattening_compaction`; first require `cargo test --locked -- --list | rg -q 'issue_m4_04_segmented_flattening_compaction'` (a missing match fails), then run `cargo test --locked issue_m4_04_segmented_flattening_compaction`. Cover the performance protocol's `Segmented nested work` family and stable selection cases. When authorized NVIDIA hardware is available, first require `cargo test --locked --features cuda -- --list | rg -q 'issue_m4_04_gpu'`, then run `cargo test --locked --features cuda issue_m4_04_gpu -- --ignored --nocapture`; record device and residency evidence separately from deterministic CPU/mock checks. Record offsets, segment-to-flat maps, interaction extraction IDs, compaction positions, temporary bounds, output order, mutation tests, and fallback observations. No GPU run is performed now, and a mock cannot mark qualification complete.

Exclusions/stop: no gather/scatter conflict handling, irregular frontier scheduling, arbitrary dynamic nested parallelism, unstable compaction, or generic graph runtime. Stop when the segmented performance family has bounded semantic coverage and one verified extracted schedule with a legal fallback.

Shared acceptance: follow `docs/implementation-plan.md`, require a nonzero relevant test selection and run Debug/Release. Explicit GPU qualification must fail on missing hardware; ignored-only or mock-only success does not complete a hardware gate. Source-invalid programs remain errors in every engine; only optimization-specific unknown legality selects a lawful fallback.

## Independent baseline (from M5-01)

CUDA: CUB `DeviceSegmentedReduce` for segmented reduce and CUB `DeviceSelect` for stable compaction. Metal: a reviewed handwritten Metal kernel pinned by content hash for each, unless an MPS primitive with the same contract exists at the pinned macOS version. Its gate is measured through the M5-01 harness against the frozen `bench/manifest-v1.json` entry for this family and against the M5-01 held-out program set, on the RTX 5070 Ti (sm_120) and the Apple M4 (local Metal); the agent-written same-structure fixture is secondary and never sets the gate.

Amended 2026-10-14: the CUDA performance gates in this contract are measured on an NVIDIA RTX 5070 Ti (sm_120, driver 580.178.04) and, for Metal, on the Apple M4, replacing the earlier RTX A6000 target. No threshold value changes. Every fraction of peak is taken against a reference measured on that same GPU (device-to-device copy bandwidth, or the measured compute peak where this contract defines one), never a datasheet number.

## Device path status (October 13, 2026)

`invar_device::segmented` defines one versioned IR for `SegmentedMap`, `SegmentedReduce` (wrapping sum/product) and stable `Compact`, its verifier, host input admission, an independent reference model, and the CUDA and MSL emitters.
`segmented_device` lowers the source-bound contracts (`segmented_source`, `compaction_source`, and a new segmented-map recognizer) to that IR; any refusal keeps the CPU path.
`CudaDevice::run_segmented` and `MetalDevice::run_segmented` execute it and publish only after confirmed completion.
`issue_m4_04_segmented_device_ir` checks the reference model against the CPU net oracle on CPU; the ignored `issue_m4_04_gpu` (CUDA) and `issue_m4_04_metal_gpu` (Metal) suites compare device output bit for bit with the source expectation, including empty and all-empty segments, 0/1/50/99/100% selection and maximum sizes.
The CUB/Metal baseline measurements and the M5-01 gates remain open.

Continuation (October 14): the map callback is now a verified, extracted typed-net DAG, bounded by operation count; reduction and compaction lowering recheck their existing extracted plans. The source-runtime and performance gates remain open. f32 remains unavailable while PR #98 is open.

Size-cap continuation (October 15): the physical segmented IR admits 2^28 elements
and segments, with at most 2^20 compaction blocks and typed above-cap refusals.
Per-device buffer/memory/geometry preflights have typed resource reasons. The
shared CUDA transfer budget remains a blocker; ordinary-source/reference limits
remain unchanged and do not gate the device API. This is not M5-01 freeze
acceptance.

The cap continuation has 32 affected CPU passes per Debug/Release profile and
three explicit Metal Release passes, including every family at 2^28 against
independent sequential Rust test-support laws. CUDA remains pending connectivity
and the shared transfer budget. Ordinary-source execution is still bounded by
its reference limits. An unrelated Metal benchmark-binary build gap was recorded
separately from the successfully emitted and executed test harness.

Ordinary-source A1 adapter (15-16 October, draft PR #151): segmented maps,
wrapping sum/product and stable compaction now register through the shared family
bridge and publish only CompletedRegion. Compaction preserves MIR allocation
between count and copy regions. Native Metal Debug/Release parity, forced/auto
failure guards and three mutation checks pass. Linux CPU/default/CUDA-feature
checks and CUDA hardware remain pending, so the draft is not fully accepted.
