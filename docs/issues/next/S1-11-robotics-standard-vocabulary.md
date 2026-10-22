# S1-11 - Robotics standard vocabulary

Problem/result: give every component, kernel and generator one shared set of typed robotics records, so a pose produced by one component and consumed by another is the same Invar type with the same unit and frame, and ROS 2 conversions are defined once.

Dependencies: R1-01, L1-03.

Release: v0.1 (milestone S1).

## Scope

Add a standard vocabulary module `std::robotics` (source under `lib/std/robotics/`, shipped with the compiler) of records over R1-01 quantities and frames, generic over the scalar `T`:

- `Point<T, F>`, `Vector3<T, U, F>`, `Quaternion<T, F>` (unit-norm as a declared property), `Pose<T, F, Child>` (translation plus rotation from `Child` to `F`), equivalent to `Transform<Child, F>`;
- `Twist<T, F>` (linear m/s, angular rad/s) and `Wrench<T, F>` (force N, torque N m);
- `JointPositions<T, N>` (rad or m per joint kind), `JointVelocities<T, N>`, `JointEfforts<T, N>` (N m or N) with `N` a compile-time joint count, plus `JointState<T, N>` bundling them with joint names (compile-time identifiers from the robot model, since runtime strings arrive with L1-04), and `JointTrajectory<T, N>` (a bounded sequence of positions and velocities with `time_from_start` durations);
- `CameraIntrinsics<T>` (width, height in px, fx, fy, cx, cy in px, distortion model `none | plumb_bob(k1, k2, p1, p2, k3) | equidistant(k1..k4)`) and `CameraExtrinsics<T, Camera, Parent>` as a `Pose`;
- `Stamped<X>` adding a time stamp (`Quantity<i64, ns>`) and the frame identity.

Each record declares its canonical C layout (used by S1-02 manifests) and its ROS 2 mapping (used by S1-04): `Pose` to `geometry_msgs/msg/Pose` (and `PoseStamped`, `TransformStamped`), `Twist` to `geometry_msgs/msg/Twist`, `Wrench` to `geometry_msgs/msg/Wrench`, `JointState` to `sensor_msgs/msg/JointState`, `JointTrajectory` to `trajectory_msgs/msg/JointTrajectory`, `CameraIntrinsics` to `sensor_msgs/msg/CameraInfo`.
The mapping is data (a table in the module), and S1-04 generates the conversion code from it.

## Semantics

Vocabulary records are ordinary Invar records; they add no language rules.
ROS messages are unit-free SI (REP-103: metres, radians, seconds, newtons) and right-handed; a conversion from an Invar record in a non-SI scale inserts the exact scale conversion, and a conversion that would lose frame identity (ROS `frame_id` strings) takes the frame name from the declared frame.
`sensor_msgs/JointState` with fewer names than `N`, or with unknown names, is a conversion failure with a registered `S17xx` code, never silent reordering.
Before M4-01 these records are used as port types and compile-time constants only (design note "Numbers before M4-01").

## Acceptance tests

Add the integration test binary `issue_s1_11_robotics_vocabulary`.
Run `cargo test --locked issue_s1_11_robotics_vocabulary` in Debug and Release and require a nonzero test count.
Positive cases:
- each record type checks with SI and non-SI scales, and its declared C layout matches a generated C header checked by a C compiler in the test (size, alignment, field offsets via `offsetof`);
- the ROS mapping table covers every record listed above, and a golden test pins it;
- composing `Pose<F, A>` with `Pose<A, B>` yields `Pose<F, B>`;
- `CameraIntrinsics` with each distortion model serializes to the IR and back unchanged.

## Negative cases

Each is rejected with a registered code: a `Twist` whose angular part is in a length unit, a `Pose` composed across mismatched frames, a joint vector of the wrong `N` connected to a port, a `CameraIntrinsics` with a non-pixel focal length, and a `JointState` mapping with a missing name.

## Done criteria

The module, its layouts and its ROS mapping table are committed; the test binary passes with a nonzero count; every diagnostic is registered under AI-01.

## Out of scope

No point clouds, images, occupancy grids or meshes (later vocabulary issues), no geometric algorithms beyond composition, no message generation (S1-04).

## Verified facts

- REP-103 fixes ROS units as SI and frames as right-handed (https://github.com/ros-infrastructure/rep/blob/master/rep-0103.rst, checked 2026-09-26).
