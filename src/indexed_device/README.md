# indexed_device

Device plans for the indexed families (gather, injective and affine scatter, histogram, frontier), derived from checked source.

## Where to start

- `Plan::from_source` in [../indexed_device.rs](../indexed_device.rs) re-verifies the family's extraction before binding inputs to the shared device IR; [affine.rs](affine.rs) expands affine indices within the device element limit.

## Position in the pipeline

Uses [indexed_source](../indexed_source/README.md), [histogram_source](../histogram_source/README.md) and [scatter_proof](../scatter_proof/README.md); feeds the device emitters in `crates/invar-device`.

## Tests

- `cargo nextest run --locked --test indexed` (`device`).
- Hardware: `tests/gpu_cuda/indexed.rs` and `crates/invar-metal/tests/indexed_gpu.rs`.

## Rules to keep

- A plan yields private completed observations; it never replaces the MIR fuel and publication protocol.

## Read next

- [indexed device paths](../../docs/indexed-device-v1.md)
