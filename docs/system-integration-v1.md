# System integration v1: Invar as the robot system description

Status: design note, decided 2026-09-26; implemented by the S1 issues, R1-01 and REN-01.
Each section answers one integration design question and states what the checker proves and what is only measured.
Context: a robot platform integrates many external libraries (ROS 2, ros2_control, Zenoh, Pinocchio, cuRobo, MuJoCo, Nix and more), and one typed description, checked once and turned into all glue by deterministic generators, owns that integration.

## Vocabulary

- System description: Invar source that declares components, their ports and connections, frames and products.
- `invar-system-v12`: the single checked, versioned intermediate representation (IR) of a system description, serialized as canonical JSON.
- Integration target: a generator from `invar-system-v12` to a file set (ROS 2 packages, MJCF, later Nix modules, AADL).
- Foreign component: a component implemented outside Invar (C or C++ behind a C shim) and described by an `extern component` manifest.

## 1. Dialect boundary

The system description is ordinary Invar plus a small declaration set; there is no separate sub-language.
Ordinary Invar supplies modules, records, variants and generics (L1-01, L1-03) and the unit and frame types (R1-01).
The added declarations are:

- `component` with the clauses `port`, `param`, `resource`, `license`, `isolation` and `impl`;
- `extern component`, the foreign-component manifest (section 3);
- `connect`, joining an output port to an input port with a connection policy;
- `frame`, declaring a coordinate frame and its parent;
- `product`, declaring distribution, commercial use, capacity, the robot reference and enabled components;
- `system`, the root that names one product and the components and connections it instantiates.

For the v5 robot/cell extension, `cell NAME { robot R; ... }` checks `R` as the cell's standalone default. The selected product's `robot` overrides that default during generation, so changing one product reference selects a different arm while keeping the cell mount, objects and I/O unchanged. A generated scene accepts one cell; multiple cells without a selection are rejected.

S1-06 adds source clauses `tool LINK pose xyz (...) quat (...);` on a robot and `dynamic mass M kg attachable;` on a cell object. The selected robot's tool link must resolve, and an attachable object needs positive mass and collision geometry. MJCF generation adds a tool site, a freejoint and mass for the object, and a disabled site-to-site weld for the sequencer to activate.

The IR has an optional mount relation: `mount POSE on FIXTURE exclude (ROBOT::LINK, ...);`.
FIXTURE must be a cell fixture with exactly one collision box, and the mount must lie on that box's top face with its z axis along the face normal (exact rational check, S1608 otherwise).
Each excluded link is robot-qualified, so one cell source still serves both arms; it must resolve (S1601) and be mounted to the surface (S1608): the root link, or a link whose chain to the root has only fixed joints and revolute joints whose axis is parallel to the normal.
Such a link only rotates about the normal, so its distance to the fixture is the same in every configuration and the exclusion cannot hide a motion-dependent collision.
MJCF generation emits `<contact><exclude>` for the selected robot's excluded links, and the substep contact gate accepts exactly those pairs.
Planner preparation receives the exclusion through the runtime binding; cuRobo 0.8.0 has no per-link world-collision mask, so the excluded links get an exact-cap sphere cover and preparation refuses if that cover still intersects the fixture.

System declarations are evaluated at compile time by the reference evaluator into one `invar-system-v12` document.
They have no runtime semantics of their own; runtime behaviour comes from the generated targets and from component implementations.
Evaluation is bounded by the existing fuel and size limits, and a system that exceeds them is rejected with a diagnostic, never truncated.
Proven: the IR is well formed and every `connect` agrees on type, unit and frame.
Not claimed: anything about runtime behaviour beyond what sections 4 and 6 state.

Numbers before M4-01: the integration track lands before floating point (M4-01) in the v0.1 order.
Compile-time constants in system declarations (rates, budgets, limits, inertias, camera intrinsics, parameters) are therefore exact decimal literals with a unit, evaluated as exact rationals by the reference evaluator and serialized as canonical decimal strings in `invar-system-v12`.
Budget sums and rate ratios in S1-03 use exact rational arithmetic, so the proofs have no rounding.
Generators emit these decimals unchanged (shortest exact decimal form) into ROS 2 parameters, ros2_control YAML and MJCF.
An optional `range MIN UNIT .. MAX UNIT` on a `Quantity<dec,U>` parameter supplies inclusive ROS 2 bounds; parameters are read-only by default, and a parameter without a range emits no bound.
Vocabulary record types (S1-11) are generic over their scalar `T`; port types name `f64` or `f32` as the wire type, which the system dialect uses only as a type and never computes with before M4-01.

## 2. Units, frames, rates and latency in the core type system

Units and frames live in the core type system (R1-01), not in the dialect.
Quantities are `Quantity<T, Unit>` with SI dimension algebra at type level; frames are nominal types; transforms are `Transform<From, To>`.
A kernel signature and a component port type are the same type, so there is no translation layer between kernels and the system graph.
Rates (`Hz`) and durations (`s`, `ms`, `us`, `ns`) are ordinary unit types, so a rate is `Quantity<f64, Hz>` and a latency budget is `Quantity<f64, ms>`.
The frame tree is declared with `frame` declarations naming a parent (the root names none).
Frame-tree closure is a checker property: every frame used by a port or transform is reachable from the root, the parent relation has no cycles, and every edge used by a transform has a declared transform source (a component port or a constant).
Proven: dimension agreement, frame agreement across every connection, closure of the declared tree.
Not claimed: that a transform source publishes a physically correct transform.

## 3. Foreign components

An `extern component` manifest declares:

- the C ABI symbols it binds (L1-04);
- ports with Invar types and the C layout of each port type;
- the lifecycle `init`, `configure`, `step`, `shutdown`, called in that order;
- resources (CPU cores, memory, GPU memory, bus bandwidth);
- an SPDX license expression and the required isolation (`in-process`, `separate-process`, `separate-partition`);
- golden vectors or properties the implementation must satisfy.

C++ libraries (Pinocchio, ros2_control hardware interfaces) are wrapped by a thin hand-written C shim that lives next to the manifest.
Invar generates the C header, the binding and a contract-test harness from the manifest.
The harness checks layout round-trip, lifecycle order, golden vectors within the declared numeric policy, and rejection of out-of-contract inputs; it runs once per manifest version and its result is recorded as evidence.
A library that exposes no C++ entry point usable in-process (cuRobo is a PyTorch library) is declared `isolation = separate-process`, and its shim is the C client of that process; the manifest and harness do not change.
For S1-06, `import component "relative/path/manifest.inv";` includes one local manifest beneath the cell source directory. The optional `for robot menagerie::name` clause chooses exactly one of several same-name manifests for the product robot, so Panda and UR5e retain their distinct fixed joint layouts. The compiler checks the selected manifest's current transitive-input hash and passing S1-02 record before the component enters the existing IR; the source declares any frames named by its ports. A stale or absent selected record fails before generation or launch.
An ordinary source component port may use `std::robotics::JointPositions<f64,robot_dof>`; the checker substitutes the selected product robot's concrete joint count before checking connections or writing IR. Foreign ports retain their numeric count and C layout.
The S1-06 cell's `joint_states: joint_state_broadcaster` component maps its `q` output to ros2_control's `/joint_states` topic in the ROS generator. The generated Pinocchio node subscribes there using the selected robot's ordered position-joint names.
The approved additive planner wire types are `PlannerRequest<f64,N,base,tool,object>`
and `PlannerTrajectory<f64,N>` for N=6/7 and at most 256 waypoints, over the existing
INVP v2 process boundary. Planner graphs carry
`planner_frame_policy: request-consumer-v1`: request poses consume measured/static
frame sources and add no frame publishers. Pose/Transform output publishers
retain duplicate/source-closure checks.
`invar component test MANIFEST --planner-runtime PATH` adds the independently
checked planner endpoint harness. It calls a separately bound current Pinocchio
FK component on returned final joints and enforces Euclidean position <=2 mm,
sign-invariant rotation <=1 degree and per-joint start continuity <=1e-6 rad.
Production runtime inputs/versions/host and FK evidence are hash-bound; explicit
CPU diagnostic records cannot qualify a production manifest. Endpoint records
do not establish collision or complete S1 task acceptance. Preparation and the
deliberately unbound arm runtime files are documented in
`examples/cells/reach_place/components/curobo_planner/README.md`.
The approved S1-06 ROS-only tendon projection names a symmetric two-joint tendon
actuator after the primary joint of its unique active identity equality. This
maps the pinned Panda `actuator8` to `finger_joint1` without a robot-name heuristic.
Only the actuator's name changes in the ROS-packaged scene: tendon transmission,
equality, gains, ranges, model assets and canonical non-ROS MJCF stay unchanged.
Ambiguous coupling, actuator-name collisions and references that would become
stale reject with S1504 before packaging. This naming projection does not itself
establish numeric or live-controller acceptance.
Proven: the binding matches the manifest; the recorded contract tests passed for that manifest version.
Measured: correctness of the library beyond the golden vectors and properties.

## 4. License and isolation lattice

Each component carries an SPDX expression; the product declares `distribution` (`internal`, `proprietary-binary`, `open`) and `commercial` (bool).
Rules for `distribution = proprietary-binary`:

| License class | Allowed only as |
| --- | --- |
| GPL (any version) | separate unmodified process, no linking |
| AGPL | as GPL, and rejected if the component is network-exposed |
| LGPL | in-process only through dynamic linking with a replaceable library; static linking rejected |
| MPL-2.0 | in-process if its files are unmodified |
| Non-commercial (for example NVIDIA Source Code License NC, CC-BY-NC) | rejected when `commercial = true` |
| Proprietary third party | requires a declared grant id |
| Unknown or unparsable | rejected |

Permissive licenses (MIT, BSD, Apache-2.0, ISC, Zlib) are allowed with attribution recorded in the generated notice file.
`internal` distribution relaxes the redistribution rules but still rejects non-commercial terms when `commercial = true` and still rejects unknown licenses.
v0.1 checks component-level licenses in S1-03; extending the check to every derivation in the Nix image closure is S1-08 (v0.2).
Proven: the declared licenses and isolation satisfy the lattice.
Not claimed: that declared licenses are correct; this is an engineering check, not legal advice, and each red-flag result goes to counsel.

## 5. Integration targets

Every generator is a pure function from the checked `invar-system-v12` IR to a file set.
Output is byte-identical for identical input: each target has a double-run hashing test and a golden-file test.
Generators never read the source text, the environment, the clock or the network, and they order all output deterministically.
Failures and warnings go through `invar-diagnostics-v3` with stable `S1xxx` reason codes registered in the AI-01 registry:

| Range | Owner |
| --- | --- |
| S1000-S1099 | S1-01 dialect well-formedness and connection agreement |
| S1100-S1199 | S1-03 rates, latency, flows and queues |
| S1200-S1249 | S1-03 frame-tree closure |
| S1250-S1299 | S1-03 resource budgets |
| S1300-S1399 | S1-03 license and isolation |
| S1400-S1499 | S1-02 foreign manifests and contract tests |
| S1500-S1599 | S1-04 ROS 2 target |
| S1600-S1699 | S1-05 robot and cell model, importers and MJCF target |
| S1700-S1799 | S1-11 standard vocabulary conversions |
| S1800-S1899 | S1-06 acceptance run and evidence report |
| S1900-S1999 | reserved for later targets (S1-09, S1-10) |

S1-05 camera sensors use inline exact intrinsics: `intrinsics (width W px, height H px, fx FX px, fy FY px, cx CX px, cy CY px, distortion none);`. Width and height are positive integer pixels, focal lengths are positive, and the IR embeds the S1-11 camera model. MuJoCo `principalpixel` receives the exact center-relative offsets `cx - width/2` and `cy - height/2`. `sensorsize` uses the same two numbers as a normalized pixel pitch; the absolute physical sensor size is not declared. Other distortion models fail with S1607.

The standard URDF target emits the shared link, joint, inertia, and supported geometry subset alongside the checked `scene.xml` referenced by `mujoco_ros2_control`. Omitted sensor metadata is listed with S1604 in `urdf-warnings.txt`. MJCF tendon, equality, and keyframe semantics, capsule collision geometry, and actuator contracts that URDF cannot preserve fail with S1607.

## 6. Timing model

The dialect carries a subset of AADL: a periodic rate per component, a deadline, a latency budget per connection, a declared WCET budget per component, and end-to-end flow declarations with latency budgets.
Proven by S1-03:

- rate compatibility under the declared connection policy (`sample-and-hold`, `queue(depth)`, `decimate(n)`);
- for each flow, the sum of declared budgets along the flow is within the flow budget;
- no unbounded queue exists.

Measured later: actual WCET (R1-06) and runtime monitors (R1-05).
Every generated report states that deadlines are declared, not measured, until R1-06 evidence exists.

## 7. Product-line configuration

R2-04 depends on the S1-01 `product` declaration and adds variants over it.
The product declaration is the single source for distribution, commercial use, capacity and the robot reference; R2-04 does not redefine it.

## 8. Repository split

`invar` (this repository) holds the dialect, checker, generators, the standard robotics vocabulary (S1-11), the foreign-manifest mechanism, and small test fixtures.
The v0.1 acceptance cell lives under `examples/cells/` with the C shims it needs.
A separate platform repository later holds production components, adapters, skills and images; the acceptance cell moves there after v0.1.

## 9. Sequencing

v0.1 is one implementation run, integration first:

```text
REN-01 -> file splits -> AI-01 -> L1-01 -> L1-02 -> L1-03 -> R1-01 -> S1-11 -> S1-01 -> S1-03
  -> L1-04 -> S1-02 -> S1-04 -> S1-05 -> S1-06 (integration-start checkpoint) -> L1-06
  -> M4-01 -> M5-01 -> M4-02 -> M4-03 -> M4-04 -> M4-05 -> M4-07 -> M4-06 -> M5-02 -> M5-03 -> AI-02
```

The integration-start target is reached at S1-06, a checkpoint that may be tagged `v0.1.0-alpha`.
The GPU performance track follows, and AI-02 closes v0.1.

## 10. Real-time core

The real-time core target is open (ros2_control versus Copper-rs).
v0.1 generates ros2_control configuration as the interim target; the comparison spike is S1-07 (v0.2).
Generators keep target specifics behind the IR, so a Copper-rs target can be added without changing the dialect.

## Verified ecosystem facts (2026-09-26)

- MuJoCo Menagerie `franka_emika_panda` is Apache-2.0 and `universal_robots_ur5e` is BSD-3-Clause (copyright ROS Industrial Consortium); both permit commercial use; tag `v2026.9.2` (`c96a32d28fb5`). Sources: https://github.com/google-deepmind/mujoco_menagerie/blob/main/franka_emika_panda/LICENSE, https://github.com/google-deepmind/mujoco_menagerie/blob/main/universal_robots_ur5e/LICENSE.
- `ros-controls/mujoco_ros2_control` (Apache-2.0) is released into ROS 2 Lyrical as 0.1.2-1 (status developed). Sources: https://github.com/ros-controls/mujoco_ros2_control, https://github.com/ros/rosdistro/blob/master/lyrical/distribution.yaml.
- nix-ros-overlay has a `lyrical` distro containing `mujoco-ros2-control`, `ros2-control`, `ros2-controllers`, `controller-manager`, `generate-parameter-library`, `rmw-zenoh-cpp`, `rosbag2-storage-mcap` and `pinocchio`. Source: https://github.com/lopsided98/nix-ros-overlay/tree/develop/distros/lyrical.
- Official `ros:lyrical-ros-base` image digest `sha256:0c19f326a339ed770ef1d4c0646a8b53bdb49dd5ff74b6de41ebdb8ac21e1806` (amd64, arm64; updated 2026-09-16). Source: https://hub.docker.com/_/ros.
- Pinocchio v4.1.0 (2026-07-07) is BSD-2-Clause and released into Lyrical as 4.1.0-1. Sources: https://github.com/stack-of-tasks/pinocchio, https://github.com/ros/rosdistro/blob/master/lyrical/distribution.yaml.
- cuRobo v0.8.0 ("cuRoboV2", 2026-04-18) is Apache-2.0; it is a PyTorch/CUDA/Warp Python library installed from source with `uv pip install .[cu12-torch]` or `.[cu13-torch]`, needs an NVIDIA GPU newer than Turing with at least 4 GB, driver 580.65.06 or newer and Python 3.10 to 3.13. Sources: https://github.com/NVlabs/curobo/blob/main/LICENSE, https://github.com/NVlabs/curobo/blob/main/docs/getting-started/installation.rst.

The approved reference-to-cell tool adapter preserves the declared TCP/weld and
pinned FK/planner reference definitions. Generated cell FK output is composed
with the source-derived `reference_from_cell`; planner goals use its inverse and
attachment poses its forward direction. The adapters emit explicit pinned-source
metadata, check finite unit transforms and refuse unsupported anchors/models.
The endpoint harness's v3 runtime/golden/evidence formats bind the cell
source and projection. The IR,
C/INVP/ROS layouts and metric limits do not change. This CPU-tested adapter does
not establish live ROS/cuRobo, collision or complete integration-start acceptance.

The October 1 approved task binding uses the existing ordinary component syntax
`impl extern invar::s1_06_reach_place_v1;`. This explicit version selects the safe
Rust `invar-sequencer` process behind generated ROS glue; a component name alone
does not. It requires the selected 6/7-joint request/planned/approved ports,
100 Hz measured cube and diagnostic tool ports, process isolation and a checked
planner/FK/controller graph. Unknown versions and unsupported source bindings
fail generation. Planner C/INVP/ROS layouts and existing IR versions are unchanged.

Generated `CellSnapshot6V1`/`CellSnapshot7V1` messages add bounded per-physics-step
joints, velocities, world tool/cube poses, attachment and geometry-ID contacts.
Alongside the selected arm arrays, bounded full-robot arrays include Panda's
finger joints, with source units and their unchanged position/velocity limits.
The MuJoCo plugin forwards a private copy to align derived data with the captured
joint state, queues it without ROS publication in the physics callback, and
refuses queue overflow or unsupported external physics callbacks/plugins. Source
IR and joint-order digests identify the checked source selection; they are not a
byte attestation of the loaded compiled model. Runtime admission checks contiguous
steps, fixed geometry classes, every contact ID, q/dq limits and independent
same-state FK agreement within the unchanged 1e-6 m gate.

The sequencer checks each complete returned trajectory and performs independent
FK on its returned last joints before publishing the identical accepted message
to the controller. Existing endpoint/start thresholds persist. Attachment needs
both service acknowledgement and a later observed state. The explicit hash-bound
task input supplies only the pinned FK library/model; every grasp and pregrasp goal
comes from the checked binding. The private IVSQ/2
framing, watchdogs and evidence scope are documented in the binding evidence.

### Task phases (decisions D1 to D5, 2026-10-15)

The generated MuJoCo model compensates gravity on the arm (D1): `invar system gen mjcf` and every model built from a cell set `gravcomp="1"` on each body of the robot subtree (an existing `1` is kept, any other value is refused with S1607), and never on cell objects or the mount.
This changed both rendered model identities, which the substep policy requalified (Panda d1dd553c to 1cc33d8a, UR5e 635a4b0d to 557f4a32; the only rendered difference is that attribute), and the captured Panda home-hold replay was recaptured from the new model.
The source gripper hold 253 stays violation-free with gravity compensation; the hold selector now also admits 254.

Every motion is planned; there are no scripted segments.
Each move fixes the planner world phase it sends with its request:

| Move | Goal | Planner world phase |
|---|---|---|
| pregrasp | pregrasp tool pose | free: the cube is an obstacle |
| descend | grasp tool pose | approach: the cube obstacle is removed |
| lift | pregrasp tool pose | held_contact: cube attached, only its table-contact slab uncovered |
| transfer | preplace (place tool pose + 100 mm base Z) | held: cube attached with its full cover |
| place_descend | place tool pose (final cube times the settled grasp inverse) | held_contact |
| retreat | preplace | approach |
| home | the exact configured home joints (joint-space plan) | home: the cube is an obstacle |

Base Z is world Z because the binding requires an identity mount.
The planner child (cuRobo 0.8.0) rebuilds its world per request with cuRobo's attach/detach pattern: it detaches, updates the world (table, plus the cube for free and home), and for held phases attaches the cube to `attached_object` as 56 spheres at the FK-checked grasp.
The held cover is the circumsphere of every outer cell of a 4x4x4 split of the 50 mm cube: radius 10.83 mm, at most 4.58 mm beyond each face.
In held_contact the layer on the face resting toward the table is pulled inward until its lowest point is 0.5 mm above that face, which leaves only the bottom 5.0 mm slab of the side walls uncovered.
The contact face is the face whose outward normal points most nearly down the table normal.
The lift needs the same contact relaxation as the place descent, because the held cube starts the lift resting on the table.

After the attach acknowledgement and a later attached sample, the task waits for the held cube to settle: the measured tool-from-cube relation must stay within 0.1 mm and 0.1 degree of the window's first sample for 200 ms of simulation time (`SETTLE_WINDOW_NS`, `SETTLE_TRANSLATION_M`, `SETTLE_ROTATION_RAD` in `src/system/sequencer.rs`).
The settled relation is the attachment pose sent with every held request; the child's 1 mm FK attachment check is unchanged.
No settle within 2 s of simulation time from the acknowledgement (`SETTLE_TIMEOUT_NS`) is the typed failure `Error::Settle`.

The home move sends the configured home joints as its joint goal.
The plan gate requires the last waypoint within 1e-6 rad of those joints and its independent FK within 2 mm and 1 degree of the home tool pose; completion keeps the 1e-6 rad joint gate.

Every IVSQ/2 snapshot reply starts with the task phase label (`move/state/world=<phase>`, `attach`, `release`, `settle`, `complete`).
The task node logs a JSON line and publishes the label on its `phase` topic (`std_msgs/String`) whenever it changes, so the MCAP records each phase change; the request topic also carries the planner world phase.

The generated `config/sequencer-binding.json` is always `invar-sequencer-binding-v2`.
An object-relative `grasp pose ...;` sets the cube-from-grasp pose and the generated weld site.
A cube without one is grasped at its centre with the identity pose, the same pose as its generated weld.
Grasp is initial cube pose times cube-from-grasp; pregrasp adds 100 mm in world Z.
The task input carries no grasp goals.
The example cubes declare +25 mm Z and Rx(pi), preserving their initial and final poses and the TCP.
Source-certified geometry,
collision-free trajectories, measured attachment and every original metric remain
separate acceptance requirements.

Projection v2 additionally checks the pinned static model root. A single shared
base/tool convention maps FK as `B * FK * T`, planner targets as
`inverse(B) * goal * inverse(T)`, cube poses as `inverse(B) * cube` and attachments
as `T * tool_from_cube`. Declared frames/message layouts and joint arrays stay
unchanged. Runtime/golden/evidence v3 bind those coordinates. Legacy projection
v1 and runtime/evidence v1/v2 remain explicitly reference-base records; a version
relabel cannot qualify the new mapping. Unknown, moving or changed source roots
fail generation. The task currently requires its identity mount to agree with
the declared robot/world binding.

The generated launch records all topics using MCAP and shuts down the recorder
when the task exits. Failed task exits and unexpected recorder exits raise S1406
to make launch fail; orderly shutdown still allows recorder finalization.
The task admits no snapshot until the planner node's own request subscription
(created only after its configure succeeds) and the attachment service are live;
the 30 s first-telemetry watchdog bounds that wait. CPU
callback tests do not prove live startup readiness, ROS hash parity, transport,
actual automatic MCAP finalization or a full resource/deadline guarantee. Those
remain required pinned-host checks, together with useful-clearance supported Panda
geometry and every original recorded two-arm S1-06 metric. Returning to the exact
configured home joints is a joint-space plan to those joints (decision D4); a pose-only plan that reaches another IK branch is not accepted as home.

Generated robot home keys now explicitly include each added free object's source
pose. This preserves the vendor robot key prefix and avoids resetting the cube to
zero when hardware selects `home`; existing velocity prefixes get zero free-body
entries. A bounded actual CPU audit reproduces the cube initialization and fresh
FK/physics agreement using the physical root transform. It also identifies a
UR5e declared-base/reference-base mismatch, source-centred weld goals with terminal
geometry/table penetration, and alternate pose-only home solutions. The audit's
rigid terminal placements and three-seed IK results are diagnostics, not full
trajectory acceptance. No base/weld semantic remedy or geometry adoption is
implemented. The detailed observations are not retained in this repository; pinned ROS build/hash parity and
the original recorded task gates remain pending.
