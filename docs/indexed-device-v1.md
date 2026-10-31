# M4-05 u32 device paths

`indexed_device::Plan::from_source` reuses the existing checked-source gather,
injective scatter, affine scatter certificate, weighted modular histogram and
dense-mask frontier recognizers. It verifies their interaction extraction again
before binding immutable inputs to `invar_device::indexed::Ir`. No caller proof
flag grants a histogram law or injectivity.

`invar_device::indexed::verify` checks exact freshly regenerated metadata,
contiguous u32 physical footprints, fresh output allocation identity, source
binding, family shapes, the named zero-identity modular law and target envelope.
`Verified::bind` checks every live index before copying or submitting anything;
scatter checks the RHS read first and refuses duplicate targets. Inputs may
alias read-only; all device outputs and worklists are fresh owned allocations.
The two emitters consume this one verified IR, with shared algorithm generation.

CUDA uses parallel gather/injective stores and u32 atomic histogram updates.
Metal uses the same operations and relaxed `atomic_uint` adds. Overflow wraps
modulo 2^32. Bounds refusals are typed candidate outcomes with no output; retained
source execution remains responsible for the exact source trap/fuel prefix.

Frontier expansion freezes entry seen/frontier masks and runs one 64-lane
cooperative group per source vertex, reading adjacent neighbors together. An
edge u->v contributes exactly when v was unseen at entry and
`frontier[v] == 0 || v > u`: this preserves the ascending source's discoveries
before a later same-round visit, including self/cross edges. Relaxed atomic stores
merge duplicate discoveries. The separate mask phase marks entry frontier
vertices seen, publishes the exact mask and clears next with one lane per vertex.
The owned bitmap is the worklist; completed decoding enumerates each mask in
ascending order for stable compaction maps. Exactly V lanes can contribute one
bit, so the admitted capacity V cannot be exceeded.

All phases are ordered encoders in one Metal command buffer or kernels in one
CUDA stream. There is one terminal host wait and no per-round host feedback or
compilation. Receipts retain seen bits, each round's mask and actual stable
compaction map. Exhausting the source's literal rounds truncates the result;
it does not assert full graph traversal or manufacture a divergence trap.

The device IR permits 2^28 indexed elements per values/indices/output buffer,
16,384 frontier vertices (2^28 dense adjacency cells) and 16 rounds, with capacity
exactly the vertex count. Frontier shape products use checked arithmetic.
Indexed launches use grid-stride loops with at most 65,535 blocks of 256 lanes.
Frontier expansion uses V groups of 64 lanes and mask publication uses ceil(V/64)
groups of 64 lanes.
Metal checks maxBufferLength, the recommended working set and pipeline lanes; CUDA checks free
device memory and launch dimensions. Resource refusals carry typed reasons.

Source observers (gather, scatter, histogram and frontier) admit by byte budget
(`ByteBudget`, #57): each computes the exact bytes of its candidate, plus the
host plans' staging bytes and visit limits, and refuses with a typed budget-exceeded error
only when that exceeds the budget; the former 4,096-element and 16-vertex
ceilings are gone. The language's per-array limit (2^24 elements, 2^28 live)
still applies to every source array, independently of the explicit device API,
and the device IR's own limits above are unchanged.
Large device correctness uses an independent sequential Rust oracle over the
same inputs; the existing small source/evaluator comparisons remain.
Arbitrary successor
functions and observable order changes remain unrecognized. Performance qualification and its exact measurement boundary were recorded
separately and are not retained in this repository.

The runtime owns every allocation through completion. Any post-submission CUDA
failure or unknown Metal completion quarantines private buffers and pipelines;
no partial receipt is returned and no automatic CPU replay occurs. Allocation
and pre-launch failures also return no output. `Receipt` names the actual target,
completion reason, source hash, completed phases and buffer traffic. CUDA traffic
counts pinned H2D/D2H bytes; Metal counts shared-buffer host copies, including
per-phase scalar buffers, rather than claiming PCIe transfers.

The A1 indexed adapter adds reached-loop ordinary `invar run` placement through
the generic CompletedRegion publication bridge. It admits through the byte-budgeted
source observers and the zero-based affine-scatter ticket; the device IR limits
above are not lifted by it.
The explicit library device observations remain independently available. Source fuel, traps and cleanup are still exercised by the
retained M4-05 CPU/source suites; a private device receipt alone is not source
publication authority. Full ordinary-source dispatch and M5 performance gates
remain separate acceptance work.

Metal f32 is unavailable in this slice. The M4-01 strict-f32 contraction work was open during qualification;
its strict-with-fallback path was neither copied nor bypassed. No owned f32,
segmented/compaction, Python or robotics implementation files were edited.

## Runnable checks

```sh
# CPU host, bounded memory and no swap:
cargo test --locked --test indexed device
cargo test --locked --release --test indexed device
cargo clippy --locked --all-targets -- -D warnings
cargo clippy --locked --features cuda --all-targets -- -D warnings
cargo clippy --locked --features metal --all-targets -- -D warnings
# NVIDIA host with NVRTC 12.9:
cargo test --locked --features cuda --test gpu_cuda indexed:: -- --ignored --nocapture --test-threads=1
# Apple Silicon host, no other GPU work running:
cargo test --locked -p invar-metal --test issue_m4_05_metal_gpu -- --ignored --nocapture --test-threads=1
```

Both hardware suites are ignored by default and fail when explicitly run without
hardware. The focused Metal suite resides in `invar-metal` to avoid the baseline
root benchmark binary's missing `benchmark_generated_map` and
`benchmark_generated_collective` methods. The root's Mac Metal all-target gate
remains unavailable until that independently owned baseline issue is fixed.

The four `caps_2_28` tests must run serially in both profiles. Frontier benchmark
sizes count adjacency cells, requiring conversion of the M5-01
vertex-based manifest/calibration. CUDA qualification at
the raised sizes remains pending.
