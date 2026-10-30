# invar-device

Versioned physical plans shared by the CUDA and Metal emitters.
A plan says what a device kernel must compute (buffers, geometry, operations, numeric mode) and carries the provenance that binds it to its source; the emitters turn a plan into CUDA C or Metal Shading Language text.
The crate is safe Rust (`#![forbid(unsafe_code)]`) and never touches a device.

## Where to start

- [src/lib.rs](src/lib.rs) defines `IR_VERSION`, the core types (`Op`, `Buffer`, `Geometry`, `ElementType`, `Provenance`) and re-exports the collective and quarantine types.
- One file per plan family: [indexed.rs](src/indexed.rs), [segmented.rs](src/segmented.rs), [summary.rs](src/summary.rs), [stencil.rs](src/stencil.rs), [transpose.rs](src/transpose.rs), [contraction.rs](src/contraction.rs), [spatial.rs](src/spatial.rs), [sorting.rs](src/sorting.rs), [fsum.rs](src/fsum.rs) and [collective.rs](src/collective.rs).
- [msl.rs](src/msl.rs) has `emit_msl` and `metal_functions`; [tc.rs](src/tc.rs) holds the tensor-core path; [target.rs](src/target.rs) and [resources.rs](src/resources.rs) describe targets and `CostFacts`.
- [quarantine.rs](src/quarantine.rs) defines `Quarantine`, the value that keeps device memory alive when completion is uncertain.

## Position in the pipeline

Built by the family adapters in [src/schedule](../../src/schedule/README.md) and [src/indexed_device](../../src/indexed_device/README.md); consumed by [invar-cuda](../invar-cuda/README.md) and [invar-metal](../invar-metal/README.md).

## Tests

- `cargo nextest run --locked -p invar-device` runs the unit tests and the three integration files in [tests](tests).
- Hardware-dependent checks are in `tests/gpu_cuda` and `tests/gpu_metal`, which are ignored by default.

## Rules to keep

- A plan is data; it grants no launch or publication authority.
- Any change to an emitted kernel changes the plan or cache identity; bump `IR_VERSION` when the schema changes.
