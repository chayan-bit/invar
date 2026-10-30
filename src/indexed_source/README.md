# indexed_source

Bounded ordinary-source `u32` gather, injective scatter and dense-mask frontier candidates, with synchronous CPU host execution.

## Where to start

- `Family`, `Candidate` and `Contract` in [../indexed_source.rs](../indexed_source.rs); `host::prepare` and `host::run_source` in [host.rs](host.rs) validate and execute admitted work.

## Position in the pipeline

A source family used by [schedule](../schedule/README.md) and [indexed_device](../indexed_device/README.md); [../frontier_source.rs](../frontier_source.rs) holds the frontier oracle.

## Tests

- `cargo nextest run --locked --test m4_indexed` and `--test byte_budgets`.

## Rules to keep

- A candidate is an observation, not an execution, fuel or publication permit.
