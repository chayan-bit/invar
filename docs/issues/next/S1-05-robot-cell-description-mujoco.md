# S1-05 - Robot and cell description, URDF/MJCF import, MuJoCo scene target

Problem/result: make the robot and its work cell typed Invar declarations that can be imported from existing URDF and MJCF models and turned back into a MuJoCo scene, so switching arms or cells is a change of one declaration, not a hand-edited XML file.

Dependencies: S1-01, S1-11.

Release: v0.1 (milestone S1).

Design: `docs/system-integration-v1.md` sections 1, 5 and 8.

## Scope

- Typed robot model declarations (`robot`, `link`, `joint`, `actuator`, `sensor`): links with mass, inertia tensor (kg m^2) and origin frames; joints (revolute, prismatic, fixed) with axis, parent and child links, position, velocity and effort limits in units; actuators (position, velocity or motor torque with gear and control range); sensors (joint position and velocity, force-torque, camera with S1-11 intrinsics); visual and collision geometry by primitive or mesh reference (mesh files stay files, referenced by content hash).
- Cell model declarations (`cell`): fixtures and objects with poses in the cell frame, the robot mount pose, and digital I/O points as typed ports.
- Each link declares a frame, so the robot's kinematic tree is an R1-01 frame tree and S1-03 closure applies unchanged.
- Importers: `invar import urdf FILE` and `invar import mjcf FILE` emit Invar declarations (a formatted `.inv` module), through maintained parser crates chosen by a reuse survey (for example `urdf-rs` for URDF and `roxmltree` for MJCF XML), with every unsupported element reported as a warning code and never silently dropped.
- Generators: `invar system gen mjcf PATH --out DIR` emits the MuJoCo scene (robot, cell, actuators, sensors, cameras, and the `mujoco_ros2_control` plugin configuration S1-04 needs), and `invar system gen urdf` emits the URDF with the S1-04 `ros2_control` fragment merged in.
- The product's `robot` reference (S1-01) resolves to one of these robot declarations; changing it is the only edit needed to switch arms.
- Fixture set: the MuJoCo Menagerie `franka_emika_panda` and `universal_robots_ur5e` models at tag `v2026.9.2`, vendored under `examples/cells/reach_place/assets/menagerie/` with their LICENSE files and a pinned commit, plus two small hand-written fixtures (a planar two-link arm and a cell with one fixture and one object).

## Semantics

Import then generate is canonicalization: the generated MJCF equals the canonicalized source MJCF for the fixture set, where canonicalization (committed as code, applied to both sides) sorts attributes, expands MuJoCo defaults classes, normalizes numbers to shortest round-trip decimal and removes comments.
Numbers are exact decimals in the IR (design note "Numbers before M4-01"), so import and generation lose no digits.
Failures and warnings use codes `S1600` to `S1699`.

## Acceptance tests

Add the integration test binary `issue_s1_05_robot_cell_mujoco` (default CI) and `issue_s1_05_robot_cell_mujoco_env` (ignored by default, required for done; needs the MuJoCo Python package or the MuJoCo library in the pinned environment).
Positive cases:
- round trip for each fixture: import MJCF, generate MJCF, compare canonical forms byte for byte; same for URDF import against canonical URDF;
- the generated scene for each Menagerie arm loads in MuJoCo (`mujoco.MjModel.from_xml_path`) in the `_env` binary and `mj_step` runs 1,000 steps without warnings;
- joint count, joint limits, and body masses of the loaded model equal the Invar declarations exactly (as decimals);
- switching the product robot reference from Panda to UR5e changes only the robot part of the generated scene (a diff test).

## Negative cases

Each is rejected with a registered code: a joint with a missing parent or child link; a kinematic loop; a negative mass or a non-positive-definite inertia; a limit with lower above upper; a mesh reference whose content hash does not match; an MJCF element the importer does not support (warning, listed in the report, and a hard error if it affects kinematics or dynamics); a robot model whose license (from its asset LICENSE) S1-03 rejects.

## Done criteria

Declarations, importers, both generators, canonicalization, vendored fixtures with licenses and both test binaries are committed and pass with nonzero counts; every code is registered.

## Out of scope

SDF and OpenUSD import, contact and collision tuning, deformables, compile-time kinematics specialization (R1-02), Isaac Lab.

## Verified facts (checked 2026-09-26)

- `franka_emika_panda` in MuJoCo Menagerie is Apache-2.0 (https://github.com/google-deepmind/mujoco_menagerie/blob/main/franka_emika_panda/LICENSE).
- `universal_robots_ur5e` in MuJoCo Menagerie is BSD-3-Clause, copyright 2018 ROS Industrial Consortium (https://github.com/google-deepmind/mujoco_menagerie/blob/main/universal_robots_ur5e/LICENSE).
- Both permit commercial use, so the D4 rule "check each Menagerie model license" is satisfied for the acceptance arms; Menagerie tag `v2026.9.2` is commit `c96a32d28fb5da84da38c1da4d749e7a13212855` (https://github.com/google-deepmind/mujoco_menagerie/tags).
