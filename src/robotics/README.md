# robotics

The shipped robotics vocabulary (joint states, poses, wire types) and the versioned planner wire layouts.

## Where to start

- `JointStateWire` and `ros_mapping` in [../robotics.rs](../robotics.rs) define the conversion data; [planner.rs](planner.rs) has `port_layout`, `config_layout` and `header` for the planner boundary. The Invar-side vocabulary is [lib/std/robotics](../../lib/std/robotics/README.md).

## Position in the pipeline

Used by [system](../system/README.md) and [component_test](../component_test/README.md).

## Tests

- `cargo nextest run --locked --test robotics` (`robotics_vocabulary`, `planner_*`).

## Rules to keep

- Wire layouts are versioned and independent of the private transport.

## Read next

- [planner wire](../../docs/planner-wire-v1.md)
