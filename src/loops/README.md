# loops

Recognition of ordinary map loops and legal fusion of adjacent maps, built on the typed graph.

## Where to start

- `single(region)` in [../loops.rs](../loops.rs) builds, verifies and extracts one map region into a `LiveMap`; `extract_fused` and `fuse_pair` fuse two maps.
- [chain.rs](chain.rs) fuses longer chains (`fuse_chain`, `FusedChain`).
- `LiveMap`, `Capture` and `Fused` keep the source-derived inputs that the scheduler binds at run time.

## Position in the pipeline

Called from `plan::decide` in [../plan.rs](../plan.rs) after [admission](../admission/README.md); uses [typed_net](../typed_net/README.md).

## Tests

- `cargo nextest run --locked --test schedule_cpu` (`issue_m2_02` covers fusion).
- `cargo nextest run --locked --test byte_budgets` covers the graph limits.

## Rules to keep

- Fusion must not elide an observable intermediate; the unwritten tail of an output is preserved.
