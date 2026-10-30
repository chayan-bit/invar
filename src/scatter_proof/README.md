# scatter_proof

Static injectivity and destination-bound certificates for literal-domain affine scatters, with bounded host execution.

## Where to start

- `Certificate` and `Contract` in [../scatter_proof.rs](../scatter_proof.rs); `host::prepare` and `host::run_source` in [host.rs](host.rs) enforce the admitted plan.

## Position in the pipeline

Used by the indexed families; see [indexed_source](../indexed_source/README.md).

## Tests

- `cargo nextest run --locked --test m4_indexed` (`static_scatter_proof`, `static_scatter_runtime`).

## Rules to keep

- The certificate carries no scheduling or publication authority.
