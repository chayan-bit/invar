# R1-02 - Robot description as compiler input

Release: v0.2 (milestone R1); planning contract, to be expanded into a full contract before implementation.

Dependencies: R1-01, S1-05.

## Goal

Accept a typed robot model as compiler input and specialize kinematics and dynamics to it at compile time.

## Motivation

Invar's robotics direction is an OEM supplier model: each new manufacturer's hardware should be cheap to support, because robot-specific facts become typed compiler inputs instead of per-customer hand-written code.

## Scope

- A typed robot model: links, joints (type, axis, limits), inertias, sensors, compute board and timing budget, importable from URDF through a maintained parser crate selected by a reuse survey.
- Compile-time specialization of forward kinematics, Jacobians, inverse dynamics (RNEA) and mass matrix (CRBA) by constant folding and unrolled chains.
- Model validation with typed diagnostics (unknown frames, inconsistent inertias, limits out of range).

## Acceptance sketch

- Specialized kinematics and dynamics match a reference implementation within a declared tolerance on several published arms and a quadruped.
- Benchmarked against Pinocchio and RobCoGen generated code on the same models, reporting ratios per algorithm.
- Changing the model changes only the specialized output, never the application source.
- Integration test binary `issue_r1_02_robot_model`.

## Out of scope

No simulation, collision checking, contact dynamics, or motion planning.

## System integration impact (from S1)

Added 2026-09-26 by the integration plan (`docs/system-integration-v1.md`); existing content above is unchanged except where this section says so.

- The typed robot model comes from S1-05 (declarations and URDF/MJCF importers); R1-02 no longer defines its own model or URDF importer and is the compile-time kinematics and dynamics specialization over the S1-05 model.
- Specialized kinematics and dynamics are differential-tested against Pinocchio (the S1-06 reference component) under the declared numeric policy.
- R1-02 output also feeds S1-05 scene generation (derived quantities such as composite inertias) and S1-04 ros2_control generation (joint limits and interfaces), from the same model.
