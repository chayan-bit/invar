# Robotics and ROS 2

Users check a system description, print its IR, and generate a ROS 2 workspace. Generation writes a new directory of sources. It does not launch a controller or talk to a robot. Those steps are skipped when `ros2` is absent.

## Sub-features

- `robotics.two-components` accepts `examples/system/two-components.inv` with no diagnostics.
- `robotics.two-producers` rejects two producers on one input with `S1004`.
- `robotics.twelve-warning` accepts `examples/system/twelve-components.inv` and reports warning `S1011`.
- `robotics.two-components-ir` prints canonical IR whose stdout sha256 matches the example index.
- `robotics.ros2-gen-1`, `robotics.ros2-gen-2`, `robotics.ros2-tree-1`, `robotics.ros2-tree-2`, and `robotics.ros2-deterministic` generate the same 26-file ROS 2 tree twice.
- `robotics.ros2-runtime` and `robotics.component-test` are skipped without a ROS 2 runtime or a foreign component library.

## How to get to it (user POV)

- Run `invar system check PATH --format json`.
- Run `invar system ir PATH`.
- Run `invar system gen ros2 PATH --out NEW_DIRECTORY`.
- Run `invar system gen mjcf PATH --out NEW_DIRECTORY` or `urdf` for the model targets.
- Run `invar component test MANIFEST` when the foreign library and fixture server exist.
- Build and launch the generated workspace with ROS 2 on a machine that has it.

## Driving it with invar-verify

Preconditions:

- `invar-verify doctor --format json` reports `ready: true`.
- The output directory for `gen ros2` does not exist yet. Its parent must exist. The helper creates that parent under scratch.

- **Graph checks.** Run `invar-verify drive robotics --format json`. Two components: `invar system check examples/system/two-components.inv --format json` exits 0, `status` is `ok`, and `diagnostics` is empty. Two producers: the same command on `examples/system/two-producers.inv` exits 1 with code `S1004`. Twelve components: exit 0, `status` `ok`, and a diagnostic code `S1011`.
- **IR identity.** The drive runs `invar system ir examples/system/two-components.inv`. The sha256 of the raw stdout bytes is `3f88c8c774eb825214c8749cb485a2a734f83d9c5cfc754d42c61ec8368870e0`.
- **Generate twice.** The drive runs `invar system gen ros2 examples/system/twelve-components.inv --out <scratch>` twice, into two new directories. Each command prints `{"status":"ok","generated_files":26}`. Each tree hashes to `f629b8127c8a672478a2d556cfcb1f2935f4191018dfa51dfe094b19d08c77a9` with 26 files, and the two hashes are equal. The hash is sha256 over sorted relative paths of `path`, NUL, decimal length, NUL, and the file bytes.
- **Runtime skip.** `robotics.ros2-runtime` is skipped when `ros2` is not on PATH. The recorded command is `ros2 launch`, and the reason says generation already ran. `robotics.component-test` is skipped because `invar component test` on `examples/cells/reach_place/components/process_fixture/manifest.inv` needs the foreign library and fixture server, which this drive does not build.

## Gotchas

- `invar system gen ros2` refuses an existing output directory (`S1500`). Re-running into the same path fails. The helper uses a new directory per attempt.
- A warning does not fail `system check`. `twelve-components.inv` is `status` `ok` with `S1011`. Treat an empty diagnostic list as a different program (`two-components.inv`).
- Generation does not call `colcon`, `ros2`, or a device. A populated directory is not a running robot.
- `invar check PATH` on a system file is the system checker, not the language checker. Prefer `invar system check PATH --format json` when the file is a system description.
- The `cell-ur5e.inv` ROS 2 tree hash depends on libm `asin` and is not the cloud fixture. The drive uses `twelve-components.inv`, whose index hash is the contract.
- Cleanup deletes the generated tree. The evidence JSON keeps the hash and the command.
