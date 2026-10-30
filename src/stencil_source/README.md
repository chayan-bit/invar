# stencil_source

Source-bound extraction of fresh radius-1 strict `f32` five-point stencils, with synchronous fenced CPU tickets.

## Where to start

- `Contract` and `Emission` in [../stencil_source.rs](../stencil_source.rs) bind recognized work to generated code; [host.rs](host.rs) publishes a reached loop through a completed CPU ticket with an exact fuel debit; [device_u32.rs](device_u32.rs) covers the wrapping-`u32` device path.

## Position in the pipeline

A source family used by [schedule](../schedule/README.md).

## Tests

- `cargo nextest run --locked --test m4_transpose_stencil`.

## Rules to keep

- A completed, fenced ticket may replace only its reached loop.
