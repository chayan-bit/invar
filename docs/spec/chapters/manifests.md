## 10. Foreign component manifests

### 10.1 Manifest clauses

[N10.1.1] An `extern component` declares `abi c "prefix"`, `lifecycle init, configure, step, shutdown`, typed ports with C layouts, resources, an SPDX license, isolation, `golden`, `property` and `shim` clauses. `tests/fixtures/s1_02/manifest-valid.inv`
[N10.1.2] `config c "name"` names a registered typed layout and `input "path"` declares an extra dependency, and the registered layouts are `inv_fk_config_v1` (model path, its length and model kind), `inv_planar2_config_v1` (two f64 link lengths) and the planner config. `src/system/foreign_layout.rs::inv_fk_config_v1`
[N10.1.3] A `golden` clause may state up to eight nonnegative exact-decimal absolute tolerances, and an unsupported layout or property text is `S1400`. `tests/robotics/foreign_manifests.rs::foreign_manifests_typed_config_and_tolerance_are_checked`
[N10.1.4] The foreign-component diagnostics are `S1400` to `S1499`. `docs/system-integration-v1.md`
[N10.1.5] A manifest file holds one `extern component` and its frames, its ports keep explicit `@` rates, and shim, golden and input paths are relative to the manifest and must resolve under the project root, and golden tolerances are in metres or radians. `src/system/foreign.rs`

### 10.2 Contract tests and evidence

[N10.2.1] `invar component generate MANIFEST --output NEW_DIRECTORY` writes a deterministic C header and binding. `tests/robotics/foreign_manifests.rs::foreign_manifests_header_is_deterministic_and_c11_layout_checked`
[N10.2.2] `invar component test MANIFEST` runs the generated contract-test harness and records evidence bound to a hash of the declared transitive inputs. `tests/robotics/foreign_manifests.rs::foreign_manifests_hash_tracks_declared_transitive_inputs`
[N10.2.3] The harness checks layout round trip, lifecycle order and golden vectors, and process crashes and malformed output are typed failures. `tests/robotics/foreign_manifests.rs::foreign_manifests_env_process_crash_is_typed`
[N10.2.4] The checker proves that the binding matches the manifest and that recorded tests passed, and it does not measure the library beyond its goldens. `docs/system-integration-v1.md`

### 10.3 Cell imports

[N10.3.1] `import component "relative/path/manifest.inv" [for robot menagerie::name];` inserts one checked manifest from beneath the source directory into the IR. `tests/robotics/integration_start.rs`
[N10.3.2] The import needs a current passing contract-test record, and a missing or changed record is `S1405` before generation. `tests/robotics/integration_start.rs`
[N10.3.3] A path that escapes the cell directory is `S1400`, and a project root is the nearest ancestor holding `.git`. `src/system.rs`
[N10.3.4] An ordinary port may use `std::robotics::JointPositions<f64,robot_dof>`, resolved from the selected Panda or UR5e product robot to a seven or six joint type, and a missing or unsupported robot is `S1001`. `src/system/compiler.rs::robot_dof`
[N10.3.5] With several same-name manifests `for robot` must select exactly one by the product robot, duplicate or mixed selectors fail, and only the selected one needs current evidence. `tests/robotics/integration_start.rs`
[N10.3.6] Frames used by an imported manifest's ports must be declared in the importing source, and an import loads no library and adds no transform. `tests/robotics/integration_start.rs`

### 10.4 Grasp poses and tool projection

[N10.4.1] An attachable dynamic object may declare one `grasp pose xyz(...) quat(...);`, its object-from-grasp pose, which needs positive dynamic mass and collision geometry, and a duplicate, wrong translation unit or zero quaternion is a source error; without one the object-from-grasp pose is the identity. `src/system/robot.rs`
[N10.4.2] MJCF places the weld site at the grasp, the tool goal is cube pose times object-from-grasp with a 100 mm world-Z pregrasp, and place uses the observed attachment and the unchanged final cube pose. `src/system/ros2/sequencer_node.rs::pregrasp`
[N10.4.3] `invar-tool-projection-v2` records source-base from reference-base `B` and reference-tool from declared-tool `T`, so FK is `B * FK * T`, a goal becomes `inverse(B) * goal * inverse(T)`, a cube pose `inverse(B) * cube` and attachment `T * tool_from_cube`. `src/system/tool_projection.rs`

