# Planner wire v1

Approved direction, 2026-09-30. This is a bounded S1-06 wire contract,
not executable floating-point language support or evidence of a planning run.
The source type/checker, C wire layouts, system-IR selection, checked foreign
lifecycle binding and INVP bridge are implemented. The transport is tested with
an explicit CPU child double; real cuRobo, its production contract harness and
generated cell runtime wiring remain separate acceptance work.

## Source types

- `std::robotics::PlannerRequest<f64,N,Base,Tool,Object>`: ordered start-joint
  positions in radians; ordered goal-joint positions in radians; target `Base <- Tool` pose; current `Base <- Object`
  pose; attachment `Tool <- Object` pose; and a u32 planner world phase.
  The phase selects the planner's world model (decision D2, 2026-10-15): 0 free (the object is an obstacle), 1 approach (the object obstacle is removed), 2 held (the object is attached with its full sphere cover), 3 held_contact (attached; only its table-contact slab is uncovered) and 4 home (the object is an obstacle; a joint-space plan to the goal joints, decision D4).
  Only phase 4 carries a joint goal; every other phase repeats the start joints bit for bit in the goal.
- `std::robotics::PlannerTrajectory<f64,N>`: dt in seconds, waypoint count, and
  storage for 256 ordered waypoints. Each waypoint contains time in seconds,
  positions in radians, and velocities in radians/second.
- Exactly N=6 or N=7 and binary64 are admitted. Frames use the existing nominal
  frame checker. Ordinary component ports may use `robot_dof` in N's position;
  it is resolved from the selected product before checking or serialization.
  Foreign manifests must state a concrete count.

The old `JointTrajectory` remains a 64-waypoint type, and the general
`FixedArray` limit remains 64. The new trajectory uses four contiguous
`PlannerSegment` records, each containing 64 `PlannerWaypoint` records.
Flattened index i means segment i/64, waypoint i%64; no header bytes are inserted
between segments. Source constructors do not gain executable f64 arithmetic.

`tests/fixtures/s1_06/planner-wire.inv` is an executed checker/IR test with an
explicit probe implementation. It is not the actual cuRobo cell.

## C representation

Headers come from `robotics::planner::header(N)`. They use explicit f64 arrays,
u32 count/flags and 8-byte alignment. Pose translation is xyz metres; quaternion
is xyzw. They are independent of the private INVP v3 transport structs.

| Public record | N=6 bytes | N=7 bytes |
|---|---:|---:|
| Request | 272 | 288 |
| Waypoint | 104 | 120 |
| Trajectory | 26,640 | 30,736 |

Request offsets: start=0, goal=8N, target=16N, object pose=16N+56, attachment pose=16N+112, phase=16N+168.
The C record is `inv_planner_request{N}_f64_v2` and the library version marker is `invar-planner-component-v2/{N}/`.
Trajectory offsets: dt=0, count=8, waypoints=16.
C11 compile-time assertions and independent offset tests check these layouts.
No six-joint source port is silently widened to seven joints. Any conversion to
the private transport's bounded storage must be explicit and validate N.

At the runtime boundary, require finite values, valid unit quaternions, a phase in 0..=4, a finite start and goal within the checked limits, and a goal equal to the start unless the phase is home. A successful
trajectory requires 2..256 waypoints, positive finite dt, valid monotonic times,
and all declared position/velocity limits. An error, timeout, child crash,
malformed result or cancellation must not publish partial output or a successful
task report. The prepared INVP v3 adapter remains the transport; no second JSON
RPC protocol or hidden runtime is introduced.

## Versioning and evidence

Graphs using these records select `invar-system-v7`, version 7, with required
`planner_contract: invar-planner-wire-v1`. The new schema permits standalone
wire/manifest graphs and graphs with robot/cell declarations. V1–v6 schema files
are unchanged, and graphs without planner records retain their prior version.
The existing source frame/flow/resource/license checks still apply.

Graphs containing the planner config select additive `invar-system-v8`, version
8. V7 is unchanged; only v8 admits the config descriptor's bounded hash, pose,
limit-array and borrowed-text fields. A v8 planner graph relabeled v7 is rejected.

## Checked lifecycle and transport

The generated foreign header requires separate-process isolation and exactly one
request/trajectory pair with matching six/seven-joint layouts. Config uses a
480-byte LP64 C record. Its string pointers are borrowed for the configure call
only. Position limits travel to the existing INVP adapter; velocity limits are
retained by the Rust wrapper for output validation. Hashes are supplied by the
caller: this layer alone does not establish model provenance.

The Rust boundary has Loaded, Initialized, Configured, Failed and Shutdown
states. A failed call or malformed output enters Failed and forbids subsequent
planning. Invalid requests fail before entering C. Output is private until every
active waypoint passes validation. Selected foreign libraries remain trusted to
uphold C pointer bounds and lifetimes; a version marker cannot prove this.

The thin C facade adapts generated types to the reviewed INVP v3 process client without adding planner algorithms.
INVP v3 differs from v2 only in the plan body: count, start joints, goal joints, target, object pose, phase, attachment pose.
The end-to-end CPU test compiles that facade and real client, then uses an
explicit fake child for normal, crash, malformed-reply and timeout scenarios
for both arm sizes. These are transport tests, not cuRobo planning results.
The manifest fixture is header/checker input, not a passing production manifest.

Real cuRobo, generated cell wiring, both robot runs, live failure injection,
MCAP metrics, and all original S1-06 criteria are still required. No mock result
can supply a passing real-planner manifest record.

## ROS message projection (generation slice)

Planner request and trajectory ports generate count-specific `PlannerRequest6V2`
/ `PlannerRequest7V2`, `PlannerWaypoint6V1` / `PlannerWaypoint7V1` and
`PlannerTrajectory6V1` / `PlannerTrajectory7V1` messages. Joint arrays keep their
exact count; pose arrays are xyz metres followed by quaternion xyzw. Nominal
frame relations remain checked in source. The trajectory uses a bounded ROS
sequence of at most 256 waypoints; its sequence length is the C wire count, and
runtime conversion requires at least two entries. This does not change the
legacy JointTrajectory mapping or its 64-element bound.

Generated C++ boundary conversion helpers reject invalid quaternions, flags,
array sizes, nonfinite values, joint/speed limits, invalid counts and nonmonotonic
times before assigning the destination. They are compiled and exercised with
explicit CPU message stand-ins, including the 256-waypoint success boundary and
unchanged destination on failure. These checks are not a ROS build or live
node/controller test.

Each interface includes its RIHS01 hash. Field identifiers and insertion-ordered
hashable JSON follow the official [ROS type-description generator](https://github.com/ros2/rosidl/blob/rolling/rosidl_generator_type_description/rosidl_generator_type_description/__init__.py)
and [FieldType definitions](https://github.com/ros2/rcl_interfaces/blob/rolling/type_description_interfaces/msg/FieldType.msg).
The six checked hashes agree with an independently constructed Python
json.dumps/SHA-256 reference. Matching against the pinned ROS build's generated
type-description JSON remains an environment acceptance check, not yet run.
Live ROS planner-node qualification and the scripted sequencer/controller
connection remain pending.

## Generated planner node (not live-qualified)

A checked foreign component using `inv_planner_config_v1` now selects a separate
planner node rather than the FK conversion path. The generated launch requires
`<instance>_config`, a ROS parameter YAML path, alongside the selected library
and child executable. Required values are the three SHA-256 strings, base/tool
frame names, ordered joint names, interleaved lower/upper joint limits, velocity
limits, table pose (xyz+xyzw), table half extents and absolute existing robot/scene
config paths. The cube remains exactly 0.05 m. Use floating-point YAML arrays for
floating-point parameters; there are no guessed robot/path/hash defaults.

The generated boundary validates these values before loading the selected
library or starting its child. Borrowed config text lives through configure only.
Request/result conversions run before calling/publishing. A planner status failure
or malformed trajectory marks the node failed, writes a non-success record with
an existing foreign diagnostic code and exits nonzero. Normal process shutdown
also checks the planner shutdown status. Destruction still performs best-effort
cleanup during exception unwinding. Supplied hashes must be bound to actual
artifacts by the production contract harness; checking hexadecimal syntax alone
is not a provenance check.

The default tests compile and execute the generated config helper with CPU values
and malformed cases, and syntax-check node/main against explicitly minimal ROS
API stand-ins. This is not evidence of ABI/API compatibility with the pinned ROS
installation, a successful colcon build, live publisher behavior, or a passing
planner process-death acceptance run. Those environment checks remain open.

All generated planner configuration parameters are declared read-only after
initialization, matching the configure-once foreign lifecycle. This includes
library/child paths, hashes, model paths and physical limits. The descriptor uses
the official [ROS read-only parameter contract](https://github.com/ros2/rcl_interfaces/blob/rolling/rcl_interfaces/msg/ParameterDescriptor.msg).
A live ROS parameter-service rejection test remains part of environment validation.

## Source-connected controller projection (not live-qualified)

When the checked controller component has one PlannerTrajectory input, the ROS
generator emits a separate controller bridge. Its subscription is remapped from
the actual source graph edge. Ordered joint names, position bounds and velocity
limits are derived from the selected checked robot's position joint actuators;
unsupported counts, missing limits and non-revolute joints fail generation.
Existing cells without the planner input keep their previous generated output.

The bridge validates the bounded planner message and publishes a standard
trajectory_msgs/JointTrajectory to ros2_control's controller topic. It retains
all waypoints and each joint's positions and velocities. ROS durations use nearest
integer nanoseconds with carry normalization; values outside the signed-seconds
range and distinct source timestamps that collapse to a non-increasing integer
time are rejected. The zero header stamp requests execution on receipt. This is
a transport projection, not evidence that the controller accepted/executed motion.

CPU conversion tests exercise 256 points, malformed inputs, duration overflow,
nanosecond collapse and unchanged destination on rejection. Generation tests use
the actual two checked robot definitions and an explicitly labelled probe
producer; generated C++ is syntax-checked against ROS API stand-ins. Production
cell sources still need the actual passing planner manifest and scripted
sequencer, followed by live controller/ROS/curobo qualification.

## Pinned ROS build/hash preparation

`issue_s1_06_planner_ros_build` adds a build-only probe using each real robot's
checked limits, generated planner node, controller bridge and all six planner
message definitions. The CPU test validates deterministic preparation and rejects
missing, duplicated, wrongly named and mismatched ROS hash entries. It does not
substitute constructed JSON for official ROS build evidence.

Run the ignored environment gate only on an authorized Linux CPU host with an
already provisioned ROS 2 Lyrical image. Supply its exact local `sha256:` image
ID and a new absolute evidence directory:

```sh
INVAR_S1_06_ROS_IMAGE_ID=sha256:<reviewed-image-id> \
INVAR_S1_06_ROS_BUILD_DIR=/absolute/new-evidence-directory \
cargo test --offline --locked --test issue_s1_06_planner_ros_build \
  issue_s1_06_planner_ros_build_env_both_counts_and_hash_parity -- --ignored --exact
```

Require one executed passing test, not a zero-test exit. The runner neither pulls images nor installs dependencies. It
records image metadata and package versions, uses no network or GPU, caps each
of two sequential builds at eight minutes, two CPUs, 10 GiB memory with no swap,
and 512 processes, and checks both installed executables. It matches ROS's
built type-description hashes by full type name and writes `hash-parity.json`
only after all six match. Generated source hashes and build logs remain in the
new evidence directory on failure.

This is a compile/hash gate only: no planner child, controller motion, sequencer,
or task acceptance runs. The probe intentionally excludes the existing FK node:
combining its measured pose with PlannerRequest currently exposes a duplicate
transform-source diagnostic; whether request poses are consumers needs a narrow
frame-policy decision before production wiring. Production cell files are
unchanged, and the existing FK/live ROS evidence is not superseded.
