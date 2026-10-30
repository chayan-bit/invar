# tensor

Rank-2 tensor contraction: source-derived contracts, physical layouts and CUDA and Metal emission.

## Where to start

- `annotate` and `verify` in [../tensor.rs](../tensor.rs) establish a `Contract`; `emit_cuda`, `emit_msl` and `emit` produce kernel source; [host.rs](host.rs) has `run_source` for CUDA, Metal or the explicit CPU model.

## Position in the pipeline

A source family used by [schedule](../schedule/README.md); cache keys are in [../tensor_cache.rs](../tensor_cache.rs).

## Tests

- `cargo nextest run --locked --test m4_contraction`, `--test m4_numeric_layout` and `--test contraction_emitter`.

## Rules to keep

- Recognition never changes source execution; unknown forms keep the CPU path.
