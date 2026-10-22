# S1-04 - Integration target: ROS 2 Lyrical

Problem/result: generate every piece of ROS 2 glue for a checked system (interfaces, parameters, launch, ros2_control and Zenoh configuration) deterministically from `invar-system-v1`, so no one hand-writes or hand-syncs ROS 2 configuration.

Dependencies: S1-01, S1-03.

Release: v0.1 (milestone S1).

Design: `docs/system-integration-v1.md` sections 5 and 10.

## Scope

`invar system gen ros2 PATH --out DIR` generates one colcon workspace fragment from the checked IR (checks must pass first):

- ROS IDL (`.msg`) for every port type without a standard mapping, plus the S1-11 standard mappings for the rest; each generated interface records its RIHS01 type hash, computed exactly as `rosidl_generator_type_description` computes it (SHA-256 over the hashable type-description JSON), and a test compares it with the hash that the ROS build produces;
- `generate_parameter_library` YAML per component from its `param` clauses (types, units in descriptions, bounds from declared ranges, read-only for compile-time constants);
- conversion code between Invar port layouts and ROS messages from the S1-11 mapping table (C++ for ROS nodes, calling the S1-02 generated headers);
- one ROS 2 node package per component whose `impl` is a foreign manifest, hosting the component through its C shim (in-process components share a component container; `separate-process` components get their own process);
- launch files (Python launch API) that start the system with the declared isolation, remappings derived from `connect`, QoS derived from the connection policy (`queue(n)` to `KEEP_LAST n` reliable, `sample-and-hold` to `KEEP_LAST 1`), and parameters;
- ros2_control configuration: the hardware interface declaration as a URDF `ros2_control` fragment (merged into the robot URDF that S1-05 generates; the S1-04 fixtures use a minimal joint-only URDF), the controller manager YAML (update rate from the controller component's rate), and controller configurations for the controllers the product enables (for v0.1: `joint_state_broadcaster` and `joint_trajectory_controller`);
- rmw_zenoh configuration (router and session config files) with `RMW_IMPLEMENTATION=rmw_zenoh_cpp` set in the generated launch environment;
- a generated notice file listing each component's license (from S1-03).

Target distribution: ROS 2 Lyrical Luth only.
In simulation the ros2_control hardware interface is `mujoco_ros2_control` (see verified facts); if that package cannot be used, the fallback is a thin generated MuJoCo hardware-interface component in this issue (a `hardware_interface::SystemInterface` that steps a MuJoCo model given by S1-05).

Section 7 decisions (2026-09-28): keep the checked S1-03 `cell-12.inv` graph as the deterministic golden-generation fixture. Its components are `impl invar run`, so the live ROS smoke uses a separate checked S1-04 fixture with manifest-backed nodes and `mock_components/GenericSystem` hardware. The S1-06 MuJoCo cell smoke remains a later acceptance gate. Add optional inclusive exact-decimal parameter bounds with `param name: Quantity<dec,U> = value U range min U .. max U;`; existing parameters and identity constants remain read-only, and a parameter without `range` emits no bound. Carry the range in a new IR version; preserve the existing v1-v3 schema bytes.

## Semantics

The generator is a pure function of the IR; output is byte-identical for identical input.
Generated files carry a header naming the generator version and the IR hash and a "generated, do not edit" line.
No generated file contains absolute host paths, user names, timestamps or environment values.
Failures use codes `S1500` to `S1599`.

## Acceptance tests

Add the integration test binary `issue_s1_04_ros2_target` (default CI: generation, golden files, determinism) and `issue_s1_04_ros2_target_env` (ignored by default, required for done; runs in the pinned ROS 2 environment).
Positive cases:
- golden-file tests per generated file kind for the S1-03 fixture graph, and a double-run hash test;
- in the ROS 2 environment: the generated workspace builds with `colcon build` with zero warnings from generated code; generated RIHS01 hashes equal those in the built type-description JSON; `ros2 launch <pkg> <launch> --show-args` and a dry-run launch parse succeed; a runtime smoke on the S1-03 fixture graph with the ros2_control `mock_components/GenericSystem` hardware (S1-05 does not exist yet at this point in the order) starts all nodes, the controller manager reaches active for both controllers, `ros2 topic hz` on the joint-state topic is within 5 percent of the declared rate over 10 s, and shutdown is clean;
- the same runtime smoke on the S1-06 cell with `mujoco_ros2_control` is part of S1-06 acceptance and must pass there before S1-04 is reported done.

## Negative cases

Each fails with a registered code and no partial output directory: a port type with no ROS representation (for example an unbounded collection); a name that is not a valid ROS name after mapping; two components mapping to the same node name; a controller component without a hardware interface; a QoS request that the connection policy cannot express.

## Done criteria

Generator, golden files and conversion code are committed; both test binaries pass with nonzero counts (the `_env` one with the environment recorded: image digest or Nix lock hash, host class); every code is registered.

## Out of scope

Other ROS distributions, ROS 1, DDS vendor tuning, MoveIt 2 configuration, Copper-rs (S1-07), the Nix image (S1-09), Jetson (R1-07).

## Verified facts (checked 2026-09-26)

- ROS 2 Lyrical Luth released 2026-05-22, LTS to May 2031; rmw_zenoh is Tier 1 but not the default RMW (https://docs.ros.org/en/kilted/Releases/Release-Lyrical-Luth.html).
- `ros-controls/mujoco_ros2_control` (Apache-2.0, 0.1.2) is released into Lyrical as 0.1.2-1 with status "developed", last pushed 2026-09-18, and is in nix-ros-overlay `lyrical` as `mujoco-ros2-control` (https://github.com/ros-controls/mujoco_ros2_control, https://github.com/ros/rosdistro/blob/master/lyrical/distribution.yaml, https://github.com/lopsided98/nix-ros-overlay/tree/develop/distros/lyrical); the fallback above is therefore not expected to be needed.
- Lyrical ships `generate_parameter_library` 1.3.0-1 and `rmw_zenoh` 0.10.6-1 (https://github.com/ros/rosdistro/blob/master/lyrical/distribution.yaml).
- Type hashes are the RIHS01 standard from REP-2011, with type descriptions per REP-2016 (draft, https://github.com/ros-infrastructure/rep/pull/381); `rosidl_generator_type_description` computes them with SHA-256 (https://github.com/ros2/rosidl/tree/rolling/rosidl_generator_type_description).
