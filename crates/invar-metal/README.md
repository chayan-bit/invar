# invar-metal

The owned Metal boundary for Apple Silicon.
Only a completed command buffer may publish output, and a library, pipeline or cache failure is a typed error, never a silent fallback.
On other hosts the crate builds but reports `Error::Unavailable`.

## Where to start

- [src/lib.rs](src/lib.rs) defines `MetalDevice`, `Error`, `LibraryError`, `Fault` and `WorkState`, plus `read_capped` and `write_atomic` for bounded file IO.
- [src/macos.rs](src/macos.rs) and [src/macos](src/macos) are the platform code: libraries, pipelines, work submission and the benchmark entry points (`CacheReport`, `Refusal`, per-family result types).
- [src/cache_file.rs](src/cache_file.rs) is the persistent pipeline-cache file format.

## Position in the pipeline

Consumes plans from [invar-device](../invar-device/README.md) and AOT containers from [src/metal_aot](../../src/metal_aot/README.md); used through the root crate's `metal` feature by `src/schedule/metal.rs`.

## Tests

- `cargo nextest run --locked -p invar-metal` runs what is host-independent.
- The integration files in [tests](tests) and `tests/gpu_metal` need Apple hardware; run them with `just test-metal` (see [CONTRIBUTING.md](../../CONTRIBUTING.md#running-the-gpu-tests-locally)).

## Rules to keep

- Creation of a pipeline is bounded by an asynchronous deadline; do not add an unbounded wait.
- Preserve the completion-before-publication order.
