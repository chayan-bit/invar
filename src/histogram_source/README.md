# histogram_source

Source-bound weighted modular histograms with bounded private CPU publication.

## Where to start

- `Candidate`, `Contract`, `Budget` and `Cost` in [../histogram_source.rs](../histogram_source.rs); [bounded.rs](bounded.rs) meters data production and receipts; [host.rs](host.rs) owns the execution boundary (`run_source`).

## Position in the pipeline

A source family used by [schedule](../schedule/README.md) and [indexed_device](../indexed_device/README.md).

## Tests

- `cargo nextest run --locked --test indexed` (histogram modules) and `--test byte_budgets`.

## Rules to keep

- No native or device publication here; publication authority stays with the MIR protocol.
