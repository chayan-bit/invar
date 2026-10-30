# eval

The reference evaluator: a direct interpreter over checked source that defines what a program means.
It shares no code with the optimizing paths, so it is the first oracle every other engine is compared with.

## Where to start

- `run` and `observe` in [../eval.rs](../eval.rs); `DEFAULT_FUEL` is the default step budget.
- [engine.rs](engine.rs) is the interpreter and charges fuel; [error.rs](error.rs) defines `TrapCategory`, `CleanupEvent` and `EvalError`; [value.rs](value.rs) defines `Value`.
- [named.rs](named.rs) is the reference entry for selected exports (`observe_named`); [host_read.rs](host_read.rs) is the reference for host array arguments.
- [floating.rs](floating.rs), [sorting.rs](sorting.rs) and [stack.rs](stack.rs) hold the strict float operations, the sort oracle and the dedicated evaluation stack.

## Position in the pipeline

Runs on the output of [check](../check/README.md); `invar run --engine reference` stops here. Compared against [mir_eval](../mir_eval/README.md) and the GNET oracle.

## Tests

- `cargo nextest run --locked --test language` (fuel, traps and cleanup order are in `issue_m0_04` and `issue_m0_05`).
- `cargo nextest run --locked --test limits` includes the 64-call depth limit.
- `cargo nextest run --locked --test exports` covers host array arguments.

## Rules to keep

- Never share optimized lowering code with this evaluator.
- Traps happen in source order; a trap runs no cleanup; fuel exhaustion is a tool limit, not a divergence proof.

## Read next

- [specification sections 3 and 4](../../docs/spec/invar-v0.1.md)
- [glossary: oracle](../../docs/glossary.md#o)
