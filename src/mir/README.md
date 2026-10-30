# mir

The typed control-flow IR: functions of basic blocks with typed values, one terminator per block, and a verifier.
Every execution path other than the reference evaluator starts from verified MIR.

## Where to start

- `lower(&checked)` in [../mir.rs](../mir.rs) builds a `Program`; [builder.rs](builder.rs) constructs blocks.
- `verify(&program)` in [verify.rs](verify.rs) checks dominance ([dominance.rs](dominance.rs)), types, ownership and terminators, and enforces the size limits (`MAX_FUNCTIONS`, `MAX_BLOCKS_PER_FUNCTION`, `MAX_INSTRUCTIONS`).

## Position in the pipeline

After [check](../check/README.md). Consumed by [mir_eval](../mir_eval/README.md), [admission](../admission/README.md), [native](../native/README.md) and [general_net](../general_net/README.md).

## Tests

- `cargo nextest run --locked --test mir` (`issue_m1_01` covers the verifier and its mutations).
- `cargo nextest run --locked --test limits` covers the size limits.

## Rules to keep

- MIR is an in-memory schema, not a serialized format; a serialized form would need its own bounded decoder.
- Verify at lowering and again before evaluation.

## Read next

- [typed MIR v1](../../docs/mir-v1.md)
