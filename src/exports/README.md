# exports

Selected scalar and array functions compiled as checked exports with retained source provenance, for the Rust and Python host boundaries.

## Where to start

- `compile_exports` in [../exports.rs](../exports.rs) returns `CompiledExports` or `ExportError`; `ExportSignature`, `ExportParameter` and `ExportEngine` describe the boundary.
- [arrays.rs](arrays.rs) admits `read` and `mut` array parameters (HA-1.2); [artifact_tests.rs](artifact_tests.rs) checks generated-library lineage.

## Position in the pipeline

Uses [check](../check/README.md), [eval](../eval/README.md) and [native](../native/README.md); consumed by [crates/invar-python](../../crates/invar-python/README.md).

## Tests

- `cargo nextest run --locked --test exports`; the Python side runs under `pytest` in `crates/invar-python`.

## Rules to keep

- The public wire format is frozen; failures are typed and never silently substituted.

## Read next

- [scalar exports](../../docs/scalar-exports-v1.md)
- [host array arguments](../../docs/host-array-arguments-ha1.md)
