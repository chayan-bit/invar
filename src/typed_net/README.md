# typed_net

The typed, demand-ordered interaction graph with checked rewrites and plan extraction.
Every automatic parallel candidate passes through it: source region to `Net`, `verify` to a `Proof`, then `extract` to an executable plan.

## Where to start

- `Net`, `Node`, `Type`, `Effect`, `Proof` and the rule identifiers (`MAP_FUSION_RULE`, `BETA_RULE`) are in [mod.rs](mod.rs).
- [verify.rs](verify.rs) checks topology, types, affine uses, effects and limits and returns a `Proof`.
- [rewrite.rs](rewrite.rs) applies proof-carrying rewrites; [extract.rs](extract.rs) turns a verified graph into a `Plan`; [dag.rs](dag.rs) extracts total element graphs.
- [codec.rs](codec.rs) encodes source terms into the graph and back; [bounded_scan.rs](bounded_scan.rs) holds the finite-state and matrix scan candidates.

## Position in the pipeline

Used by [loops](../loops/README.md) and the source families; see the code tour step 5 in the [architecture code tour](../../docs/architecture.md#code-tour-one-program-end-to-end).

## Tests

- `cargo nextest run --locked --test mir` (`issue_m1_04` verifier, `issue_m1_05` rewrites and extraction).
- `cargo nextest run --locked --test summary` covers the bounded scans.

## Rules to keep

- Unknown or possibly failing work cannot satisfy a rewrite's guard.
- A bound exhausted is a tool result, never proof of divergence.

## Read next

- [typed interaction graph v1](../../docs/typed-interaction-v1.md)
- [parallelism foundations](../../docs/parallelism-foundation.md)
