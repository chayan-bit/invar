# f32_source

Source-derived rank-1 strict `f32` regions over the typed net proof, with exact CPU execution and publication.

## Where to start

- `Region` and `Error` in [../f32_source.rs](../f32_source.rs); [host.rs](host.rs) has `run_source` and `run_cpu_source`.

## Position in the pipeline

A family adapter used by [schedule](../schedule/README.md) for `f32` programs.

## Tests

- `cargo nextest run --locked --test numeric_layout` (`source_f32`).

## Rules to keep

- Unknown forms and failed live guards keep the original source loop; results equal strict `strict-f32-v1`.

## Read next

- [numeric contracts](../../docs/numeric.md)
