# S1-02 - Foreign-component manifests over the C ABI with generated contract tests

Problem/result: let external C and C++ libraries (Pinocchio, cuRobo, ros2_control hardware interfaces) enter a system as typed components, described once by a manifest and contract-tested once per manifest version, so that the rest of the graph can rely on the manifest.

Dependencies: L1-04, S1-01.

Release: v0.1 (milestone S1).

Design: `docs/system-integration-v1.md` section 3.

## Scope

The `extern component` declaration (parsed and recorded in the IR by S1-01; generation, contract tests and evidence here):

```text
frame world;
frame base: world;
frame tool: base;
extern component pinocchio_kinematics {
  abi c "invar_pinocchio_bridge_v1";     // generated adapter prefix
  lifecycle init, configure, step, shutdown;
  config c "inv_fk_config_v1";
  port in  q:    std::robotics::JointPositions<f64,7> @ 100 Hz layout c "inv_joint7_f64";
  port out pose: std::robotics::Pose<f64,base,tool> @ 100 Hz layout c "inv_pose_f64";
  resource cpu_cores = 1, memory = 64 MiB;
  license "BSD-2-Clause" link dynamic;
  isolation in-process;
  golden "golden/panda_fk.json" tolerance abs 0.000000000001 m, abs 0.000000000001 rad;
  property "pose.rotation is unit norm within 0.000000000001";
  shim "shim/pinocchio_bridge.cpp";
  input "abi_reference.h";
  input "abi_reference.cpp";
}
```

The runnable v0.1 subset keeps S1-01's explicit port `@` rate and uses exact decimal tolerance literals. `config c` names a checked typed layout; repeatable `input` clauses declare transitive local dependencies for the evidence hash. Generated adapter symbols may use a distinct versioned prefix from a hand-authored reference shim when their signatures differ; the adapter must preserve the reference shim's typed config and ownership semantics.

- Manifests live next to their C shim (`examples/cells/<cell>/components/<name>/`), and the shim is hand-written, small, and exports only C symbols named `<prefix>_init`, `_configure`, `_step`, `_shutdown` plus a version symbol.
- Invar generates from each manifest: the C header (`<name>.h`, types from S1-11 layouts), the Invar binding over the L1-04 C ABI, and a contract-test harness.
- The harness checks: layout round-trip (Invar value -> C struct -> Invar value, size, alignment and offsets); lifecycle order (calls out of order return the manifest's typed error, and `step` before `configure` never runs); golden vectors within the declared numeric tolerance; declared properties on randomized in-contract inputs with a fixed seed; rejection of out-of-contract inputs (non-finite values, wrong joint count, unnormalized quaternion) before the library is called.
- `isolation separate-process` components (cuRobo, a PyTorch library) run their implementation as a child process; the shim is a C client speaking a length-prefixed canonical encoding of the port types over a pipe or Unix socket; the harness then additionally checks process start, crash (typed error, no partial output) and shutdown.
- `invar component test MANIFEST` runs the harness and writes an evidence record `evidence/<name>-<manifest-hash>.json` (manifest hash, library version string from the shim, host, results); S1-03 and S1-06 accept a manifest only with a passing record for its current hash.

## Semantics

A manifest is a trust boundary: every value crossing it is validated against the port type and contract before use.
A foreign failure is a typed outcome (`S14xx` code), never a panic, abort or truncated output.
The manifest hash covers the manifest text, the shim source and the golden files; any change requires a new passing record.

## Acceptance tests

Add the integration test binary `issue_s1_02_foreign_manifests`.
Run `cargo test --locked issue_s1_02_foreign_manifests` in Debug and Release and require a nonzero test count; tests needing Pinocchio or Python run under the pinned `ros2` environment and are listed separately as `issue_s1_02_foreign_manifests_env` (ignored by default, required for done).
Positive cases:
- a self-contained fixture library written in C in the repository (a planar two-link arm) passes every harness check in default CI;
- the Pinocchio forward-kinematics manifest for the Panda passes golden vectors generated independently (Pinocchio's own Python binding in a separate run, committed with the command that produced them);
- a separate-process fixture passes start, step and shutdown, and a killed child yields the typed crash error.

## Negative cases

Each fails with a registered `S14xx` code: a C layout that disagrees with the port type; a missing lifecycle symbol; `step` before `configure`; a golden vector outside tolerance; a property violation; a non-finite input reaching the boundary; a manifest whose hash has no passing evidence record; a changed shim with a stale record.

## Done criteria

Manifest semantics, generated header, binding and harness are committed; the fixture library, Pinocchio and a separate-process fixture pass; the test binaries pass with nonzero counts; every code is registered.

## Out of scope

Automatic C++ binding generation (no bindgen of C++ headers), in-process Python embedding, remote components, sandboxing beyond process isolation.

## Verified facts

- Pinocchio v4.1.0 (2026-07-07) is BSD-2-Clause and packaged for ROS 2 Lyrical (4.1.0-1) and in nix-ros-overlay `lyrical` (https://github.com/stack-of-tasks/pinocchio, https://github.com/ros/rosdistro/blob/master/lyrical/distribution.yaml, checked 2026-09-26).
- cuRobo v0.8.0 is Apache-2.0 and is a Python library on PyTorch, CUDA and Warp with no C API, so its manifest uses `isolation separate-process` (https://github.com/NVlabs/curobo, checked 2026-09-26).
