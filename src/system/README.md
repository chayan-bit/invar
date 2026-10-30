# system

The system dialect: the checked `invar-system-v12` IR, graph checks, foreign components, robot models and the generators for ROS 2, MJCF and URDF.

## Where to start

- `SystemIr` in [../system.rs](../system.rs); `compile_manifest` and `compile` in [compiler.rs](compiler.rs) build it from source; `import_mjcf` and `import_urdf` ingest robot models.
- [graph.rs](graph.rs) runs the exact graph checks (rates, latency, frames, resources); [foreign.rs](foreign.rs) and [components.rs](components.rs) handle foreign component manifests.
- [ros2.rs](ros2.rs) with [ros2/](ros2) and [ros2_templates/](ros2_templates) generates ROS 2 workspaces; [generate.rs](generate.rs) writes MJCF and URDF; [sequencer.rs](sequencer.rs) is the task core behind `invar-sequencer`.

## Position in the pipeline

Parallel to the numeric pipeline: it uses [front](../front/README.md) and [check](../check/README.md) and shares nothing with the schedulers.

## Tests

- `cargo nextest run --locked --test robotics`; tests ending in `_env` need external tools.

## Rules to keep

- Generated output is deterministic and byte-identical for identical input.
- Partial runtime evidence is never reported as acceptance.

## Read next

- [system integration](../../docs/system-integration-v1.md)
- [specification sections 8 to 10](../../docs/spec/invar-v0.1.md#8-system-dialect)
