# M4 source increment

The compiler accepts primitive `f32`, affine `Array<f32>`, bounded exact decimal
literals, unary negation, ordered arithmetic and IEEE comparisons. `fma` is
explicit; `f32_from_i32`/`f32_from_u32` convert integers, and `filled_f32` creates
an owner. Arrays support literals, indexing, mutation, read/mut/take parameters,
owner returns and normal cleanup under existing bounds and source fuel.

The source and MIR evaluators use the integer/rational binary32 oracle. Native
LLVM carries f32 bits in internal scalar words, emits strict floating operations
without reassociation/contraction flags, and preserves subnormals. Only explicit
`fma` fuses multiplication/addition. Source main remains i32. Public scalar
exports and the L1-04 ABI remain i32/u32.

MIR carries strict numeric policy and layout version1; verification rejects stale
layout versions, and MIR/net/provenance/export identities bind both contracts.
The production layout checker validates bounded physical byte maps. Ordinary
ordered rank-2 matrix loops now produce sealed source annotations bound to the
exact verified MIR function and stable net identity. Concrete dimensions and
input lengths bind bounded row-major byte maps; unknown forms retain CPU
execution. `tensor::emit_cuda` checks actual source, ownership/net contracts and
layouts, then emits a strict program-specific contraction and canonical SHA256
cache identity from those exact inputs, options and emitted bytes.

The emitted CPU analogue is executed against the integer/rational binary32
oracle for zero, one and odd shapes, including subnormal/rounding-sensitive
values. Required CUDA numeric options prohibit FMA contraction and FTZ; CUDA
capability metadata is restricted to the implemented toolkit/SM set. Metal
contraction emission returns unsupported. A matching content key remains an
untrusted hint; these APIs grant no pointer lease, source-fuel/trap commit or
device-fence/launch permission.

`tensor::host::run_source` now admits only reached contraction loops, checks the
full remaining source fuel, and holds live read loans through private staging
and confirmed completion. Packets bind the artifact key, dimensions, lengths,
unique call generation and input-byte digest. Publication updates the fresh
source output atomically and charges its exact source ticks. Pending, failed,
stale, malformed or nonfinite work retains ordered source fallback and the
original spans and cleanup. Unconfirmed work remains quarantined. The actual
software `CpuDevice` uses ordered binary32 operations with a private work cap;
there is no CUDA adapter for this host path yet.

The device crate also verifies and emits out-of-place word transposes, including
masked 16x17 shared CUDA tiles and an executed CPU analogue. The bounded-summary
module implements source-ordered table and wrapping-matrix composition and
prefix scans. Source extraction and host publication for those two families
remain open. The parent reports ten floating CUDA goldens with guard words
qualified on the frozen Windows emitter bundle `988271b`, using NVRTC12.4 and
an A6000. That component receipt does not qualify this source-bound host path.
Its CUDA runtime integration, Tensor Core, measured performance and full M4
acceptance remain open.

Run the nonzero source/numeric/layout tests in both profiles:

```sh
cargo test --offline --locked --features native --test issue_m4_01_source_f32 \
  --test issue_m4_01_numeric_oracle --test issue_m4_01_decimal_literals \
  --test issue_m4_01_layout_oracle
cargo test --offline --locked --release --features native --test issue_m4_01_source_f32 \
  --test issue_m4_01_numeric_oracle --test issue_m4_01_decimal_literals \
  --test issue_m4_01_layout_oracle
```

The source/extraction/emission/cache integration suite is
`--test issue_m4_01_numeric_layout_contracts --test issue_m4_01_tensor_cache`.
The device package suite is `cargo test --offline --locked -p invar-device
--test m4_01_floating`; repeat both in Release.

Reached-loop fuel and completion tests are `--test issue_m4_01_tensor_host`;
bounded summary tests are `--test issue_m4_07_bounded_summary_families`. The
transpose package suite is `-p invar-device --test m4_02_transpose`. Repeat each
in Release.
