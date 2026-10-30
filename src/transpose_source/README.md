# transpose_source

Checked ordinary-source rank-2 transpose extraction and execution.

## Where to start

- `Contract` and `Emission` in [../transpose_source.rs](../transpose_source.rs); `host::contracts`, `provide` and `run_source` in [host.rs](host.rs) handle inputs and completion.

## Position in the pipeline

A source family used by [schedule](../schedule/README.md).

## Tests

- `cargo nextest run --locked --test m4_transpose_stencil`.

## Rules to keep

- The contract identifies a lawful loop; it does not authorize launch, fuel debit or publication.
