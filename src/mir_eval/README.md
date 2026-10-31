# mir_eval

The MIR interpreter, independent of the reference evaluator, plus the atomic publication of completed family results.
With a scheduler attached it is the serial fallback that every parallel plan falls back to.

## Where to start

- `run` and `observe` in [../mir_eval.rs](../mir_eval.rs) execute verified MIR; [interpreter.rs](interpreter.rs), [operation.rs](operation.rs), [frame.rs](frame.rs) and [values.rs](values.rs) hold control flow, operations, frames and value access.
- [bulk.rs](bulk.rs) and [region.rs](region.rs) publish a staged result when a loop is reached with true guards.
- One file per family publishes a completed result: [gather.rs](gather.rs), [scatter.rs](scatter.rs), [histogram.rs](histogram.rs), [stencil.rs](stencil.rs), [transpose.rs](transpose.rs), [tensor.rs](tensor.rs), [summary.rs](summary.rs), [f32_source.rs](f32_source.rs); [reduce.rs](reduce.rs) runs the deterministic float reductions.

## Position in the pipeline

After [mir](../mir/README.md); driven by [schedule](../schedule/README.md).

## Tests

- `cargo nextest run --locked --test mir` and `--test schedule_cpu` compare it with the reference evaluator and exercise publication.
- The family binaries (`indexed`, `segmented`, `summary`, `transpose_stencil`) cover each publication path.

## Rules to keep

- It must agree with the reference evaluator on value, trap, step count and cleanup.
- Only a complete private result may write; a failed or partial result publishes nothing.

## Read next

- [typed MIR v1](../../docs/mir-v1.md)
