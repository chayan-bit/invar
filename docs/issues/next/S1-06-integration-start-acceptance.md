# S1-06 - Integration-start acceptance: one Invar source, two arms, simulated reach-and-place

Problem/result: prove the integration base end to end: one Invar source describes a simulated arm cell, is checked, generates ROS 2 and MuJoCo artifacts, and runs with Pinocchio and cuRobo behind foreign-component manifests on two dissimilar arms, with a generated evidence report.
This issue is the integration-start checkpoint; when it passes the repository may be tagged `v0.1.0-alpha`.

Dependencies: S1-02, S1-04, S1-05.

Release: v0.1 (milestone S1); integration-start checkpoint.

Design: `docs/system-integration-v1.md` sections 3, 4, 6, 8 and 9.

## Scope

- The cell source `examples/cells/reach_place/cell.inv` (plus its component manifests, C shims and golden files under `examples/cells/reach_place/`) declares: the frame tree (world, table, robot base, tool, object), the cell (table, one 50 mm cube, a place target), the product `ReachPlace` (`distribution proprietary-binary`, `commercial true`, capacity), and components:
  - `mujoco_sim` (the `mujoco_ros2_control` hardware interface, Apache-2.0, separate process);
  - `pinocchio_kinematics` (S1-02 manifest, BSD-2-Clause, in-process, dynamic);
  - `curobo_planner` (S1-02 manifest, Apache-2.0, separate process, GPU memory declared);
  - `task_sequencer` (a small Invar or C component that issues the scripted task);
  - `joint_trajectory_controller` and `joint_state_broadcaster` (ros2_control, Apache-2.0);
  - `recorder` (MCAP via `rosbag2_storage_mcap`).
- Two products differ only in `robot`: `menagerie::franka_emika_panda` (7-DoF) and `menagerie::universal_robots_ur5e` (6-DoF); the test proves the two sources differ in that one line.
- Scripted task per arm: move from the home configuration to a pre-grasp pose above the cube, descend, attach the cube (a MuJoCo weld constraint toggled by the sequencer, no grasp physics), move to the place target, release, return home.
- Success criteria per arm, measured from the MuJoCo state recorded in MCAP: final cube position error at most 5 mm and orientation error at most 2 degrees against the place target; tool pose error at each waypoint under 2 mm and 1 degree (computed by Pinocchio from recorded joints and cross-checked against MuJoCo site poses within 1e-6 m); final joint home error at most 1e-6 rad; every recorded joint position and velocity inside declared limits; zero contacts between robot links and the table or between robot links themselves (MuJoCo contact list); task completes within 60 s of simulated time.
- The run produces an MCAP log with every port topic and the MuJoCo state, and `invar system report` generates the evidence report `examples/cells/reach_place/evidence/<arm>.md` from the diagnostics of S1-01 and S1-03, the S1-02 contract-test records, the S1-04 build and launch results and the run metrics, including the sentence that deadlines are declared, not measured.

## Environment

- Checking and generation run anywhere (default CI covers them).
- ROS 2 Lyrical, MuJoCo, Pinocchio and the non-GPU parts run in a pinned environment (nix-ros-overlay `lyrical` pinned by a flake lock, or the digest-pinned `ros:lyrical-ros-base` image) on an x86_64 Linux host with bounded memory and runtime.
- cuRobo needs an NVIDIA GPU (newer than Turing, driver 580.65.06 or newer), so the full run executes on a Linux host with such a GPU in the same pinned environment.

## Acceptance tests

Add `issue_s1_06_integration_start` (default CI: the cell checks, both products generate deterministically, the one-line difference test, golden reports for the check-only part) and `issue_s1_06_integration_start_env` (ignored by default, required for done; runs the simulation for both arms and asserts every success criterion).
Positive cases: both arms meet every success criterion in three consecutive runs with fixed seeds; repeat completion is accepted when every recorded tool waypoint is under 2 mm and 1 degree and final joint home error is at most 1e-6 rad. The report states the observed determinism class. Bit-exact final-state replay requires step-scheduled trajectory starts, attachment and initial-state barriers and belongs to v0.2. The S1-04 runtime smoke passes on this cell.

## Negative cases

Each fails with its registered code before any process starts: the cell with a GPL component linked in-process; a cuRobo manifest whose contract-test record is stale; a robot reference to a Menagerie model with an "All rights reserved" license; a flow budget that the declared budgets exceed; a place target in a frame not reachable from the root.
At runtime: a planner process killed mid-task yields a typed failure in the report and a nonzero exit, never a success claim.

## Done criteria

The cell, manifests, shims, golden files, evidence reports for both arms and MCAP logs (or their hashes and sizes when over 5 MB, with the logs archived as evidence off the repository) are committed; both test binaries pass with nonzero counts; the environment (image digest or lock hash, host, GPU, driver, cuRobo, Pinocchio, MuJoCo and mujoco_ros2_control versions) is recorded; the checkpoint completion comment is posted.

## Out of scope

Real hardware, grasp physics, perception, MoveIt 2, Invar-compiled kinematics replacing Pinocchio (S1-12), measured WCET (R1-06), the Nix image (S1-09).

## Verified facts (checked 2026-09-26)

- Arm licenses: Panda Apache-2.0, UR5e BSD-3-Clause (see S1-05 sources).
- cuRobo v0.8.0 (Apache-2.0) installs from source with `uv pip install .[cu12-torch]` or `.[cu13-torch]`, needs Python 3.10 to 3.13 and driver 580.65.06 or newer (https://github.com/NVlabs/curobo/blob/main/docs/getting-started/installation.rst).
- `mujoco_ros2_control` 0.1.2-1 is released for Lyrical (https://github.com/ros/rosdistro/blob/master/lyrical/distribution.yaml).
- nix-ros-overlay `lyrical` contains `mujoco-ros2-control`, `ros2-control`, `ros2-controllers`, `pinocchio`, `rmw-zenoh-cpp` and `rosbag2-storage-mcap` (https://github.com/lopsided98/nix-ros-overlay/tree/develop/distros/lyrical); whether it builds them reproducibly is recorded by this issue (an open item).
- `ros:lyrical-ros-base` digest `sha256:0c19f326a339ed770ef1d4c0646a8b53bdb49dd5ff74b6de41ebdb8ac21e1806` (amd64 and arm64, 2026-09-16) is the pinned container fallback (https://hub.docker.com/_/ros).
