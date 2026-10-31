# summary_source

Checked recognition and preparation of bounded summary scans (finite-state and matrix), with CPU execution and a device plan.

## Where to start

- `Family`, `Contract`, `Observation` and `Prepared` in [../summary_source.rs](../summary_source.rs); [matcher.rs](matcher.rs) holds the matrix-scan helpers; [host.rs](host.rs) has `run_source`; [device.rs](device.rs) lowers a prepared scan to the device IR. [../bounded_summary.rs](../bounded_summary.rs) defines the summaries.

## Position in the pipeline

A source family used by [schedule](../schedule/README.md).

## Tests

- `cargo nextest run --locked --test summary`.

## Rules to keep

- Candidate extraction stays separate from fuel debit, loan admission and publication.
