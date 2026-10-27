# Invar verification map

This directory is the maintained source for verifying Invar's user-facing behavior. Read this index, then the feature file, then drive that feature with `invar-verify`.

The compiler is a short-lived CLI. Launch means building `target/debug/invar`. Each drive is its own process.

## Baseline preconditions

- The checkout is the Invar repo (`Cargo.toml` package name `invar`, `rust-toolchain.toml` channel 1.95.0).
- `invar-verify build --jobs 2` has produced `target/debug/invar` with default features (no `native`, `cuda`, or `metal`).
- `invar-verify doctor --format json` reports `ready: true` for that binary.
- `INVAR_VERIFY_ROOT` is outside the checkout. The default is `/tmp/invar-verify`.
- Do not start a second `cargo build` or `cargo test` against the same `target/` directory. The helper holds a lock and refuses the second one.
- Drive the binary this run built. `INVAR_VERIFY_BIN` overrides the path when a proof must name a different binary.
- `gnet-device` CPU models need a C++20 compiler that provides `<barrier>`. If `c++` cannot and `g++` can, the helper sets `CXX=g++` unless `CXX` is already set.

## Driving conventions

- Start from the baseline unless the feature file says otherwise.
- Treat every command as literal. Keep flags, engines, and worker counts unchanged.
- Run the bundled recipe with `invar-verify drive <feature> --format json`.
- Re-run one step with `invar-verify run -- <invar arguments>`.
- Assertions live in the drive. `run` only captures the process.
- Restore nothing in the checkout. Scratch programs and generated trees live under `$INVAR_VERIFY_ROOT/scratch`.
- Do not delete `$INVAR_VERIFY_ROOT/evidence` during cleanup.

## Proof and skip reporting

- Capture the command, the exit code, and the JSON document, not only a final "ok".
- A result check names `result.value` or the diagnostic code. A generation check names the file count and the tree sha256 from `examples/index.json`.
- Record the feature id and the check id with every artifact. The helper writes `evidence/checks/<id>.json`.
- A CPU-model backend is a pass only when `backend` is `cpu-model` or `cpu-model-cuda` / `cpu-model-metal` and the report has no `gpu_execution_claim`.
- A refusal (`CUDA_UNAVAILABLE`, `METAL_UNAVAILABLE`, `NATIVE_UNAVAILABLE`, `forced CUDA`, `forced Metal`) proves the default build does not pretend. It does not prove the device.
- Report a skipped entry with the command that was not run and the unmet host fact. Do not mark it verified through a different engine.

## Feature entry contract

Each feature file starts with an H1 and one paragraph of user-visible behavior, then exactly these H2 sections, in order:

1. `Sub-features`
2. `How to get to it (user POV)`
3. `Driving it with invar-verify`
4. `Gotchas`

## Features

- [Parser](parser.md) checks source structure and rejects a broken token.
- [Checker](checker.md) checks types, ownership, loans, views, and a foreign signature.
- [Reference evaluator](reference-evaluator.md) runs programs on `--engine reference` and reports values and traps.
- [Parallelism](parallelism.md) runs admitted CPU plans, explain, auto, and the interaction-net CPU model.
- [Codegen](codegen.md) runs device CPU models and records LLVM, CUDA, and Metal hardware as skipped when the host cannot run them.
- [Robotics and ROS 2](robotics.md) checks system descriptions and generates a ROS 2 workspace. Controller execution is skipped without ROS 2.
- [CLI](cli.md) covers help, usage, missing files, and the system-file routing of `invar check`.
- [Tests](tests.md) runs the file-size ratchet and one compiler test. The full nextest corpus is skipped in the default drive.
