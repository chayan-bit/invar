# component_test

`invar component test`: the public component contract harness and the writer for immutable passing evidence.

## Where to start

- `command` in [../component_test.rs](../component_test.rs) takes the CLI arguments and returns an exit status; `planner` in [planner.rs](planner.rs) runs the planner-specific, hash-bound test path.

## Position in the pipeline

Uses [system](../system/README.md) manifests.

## Tests

- `cargo nextest run --locked --test robotics` (`foreign_manifests`, `planner_contract`, `planner_boundary`).

## Rules to keep

- Diagnostic fixtures never write a pass.

## Read next

- [specification section 10](../../docs/spec/invar-v0.1.md#10-foreign-component-manifests)
