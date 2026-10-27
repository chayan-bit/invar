---
name: verify-invar
description: "Build and drive the Invar compiler the way a user does: check and run .inv programs, CPU parallelism, CPU-model device kernels, and ROS 2 workspace generation. Use when changing Invar or proving a compiler behavior. Skips GPU, Metal, LLVM-native, and ROS 2 hardware when the host cannot run them."
---

# Verify Invar

Invar is a batch CLI (`invar`) plus a library. A user types a command, the process exits, and the proof is the exit code and the stdout document. There is no server, port, or shared data directory. Each drive is a new process. Two drives may run at once. Two `cargo build` or `cargo test` processes may not: they share `target/`, and this skill takes an exclusive lock for those.

The feature map is [`references/features/README.md`](references/features/README.md). `features/` is the same directory. Read the index, then the feature file, before driving.

## Launch

From the repository root, with Rust 1.95.0 (the `rust-toolchain.toml` pin):

```sh
.cursor/skills/verify-invar/scripts/invar-verify build --jobs 2 --format json
```

Ready means `target/debug/invar` exists and `invar --help` exits 0 with a first line `Usage: invar --help`. The build uses default Cargo features. It does not enable `native`, `cuda`, or `metal`. A missing binary is not ready, even if `cargo` itself runs.

The build writes `target/`. `--dry-run` prints the cargo command and does not run it.

## Doctor

```sh
.cursor/skills/verify-invar/scripts/invar-verify doctor --format json
```

Read-only. Exit 0 when `ready` is true: the binary answers `--help`, `rustc` is 1.95, and `cargo` is on `PATH`. Exit 1 when the report was produced but the tree is not worth driving (build first). The report also records LLVM 21, `nvidia-smi`, Metal (Darwin or not), `ros2`, `cargo-nextest`, and `just`. Those facts decide later skips. Doctor does not install anything and does not start a GPU.

## Drive

Run one feature, or every feature:

```sh
.cursor/skills/verify-invar/scripts/invar-verify drive parser --format json
.cursor/skills/verify-invar/scripts/invar-verify drive parallelism --format json
.cursor/skills/verify-invar/scripts/invar-verify suite --format json
```

`suite` builds, then drives every feature file, including the focused test. Feature ids: `parser`, `checker`, `reference`, `parallelism`, `codegen`, `robotics`, `cli`, `tests`.

A single invocation, without the assertions, is:

```sh
.cursor/skills/verify-invar/scripts/invar-verify run -- check examples/sum.inv --format json
```

Drives use the examples under `examples/` and scratch programs under `$INVAR_VERIFY_ROOT/scratch`. They do not edit the checkout. `cpu-parallel` always passes `--workers`. Device CPU models are `source-device ... cpu-model` and `gnet-device ... cpu-model-cuda|cpu-model-metal`. Those are not GPU runs. `gnet-device` compiles a C++20 model. If `c++` cannot include `<barrier>` and `g++` can, the drive sets `CXX=g++` for that child. A `CXX` already set in the environment is left alone.

`--format json` prints one `invar-verify-v1` object on stdout. A failed check names the command, the exit code, and the field that disagreed. Exit 0 when every check passed or was an explicit skip. Exit 1 when a check failed. Exit 2 when the tool itself could not run (wrong repo, missing binary, cargo lock held).

## Evidence

Transcripts go to `$INVAR_VERIFY_ROOT/evidence` (default `/tmp/invar-verify/evidence`):

- `doctor.json` — host probe and whether the binary is worth driving
- `suite.json` — the last suite summary
- `checks/<id>.json` — command, exit code, stdout, stderr, and the assertion

Override the root with `--root` or `INVAR_VERIFY_ROOT`. Do not point it at the repo. The summary on stdout is the proof to quote. The files are the full transcripts.

A passing check exercised the real `invar` binary on a real `.inv` file (or `cargo` / `scripts/check-file-size` for the test feature). A skip names the command that was not run and the host fact that blocked it. A CPU-model result is not a CUDA or Metal result. `NATIVE_UNAVAILABLE`, `CUDA_UNAVAILABLE`, and `METAL_UNAVAILABLE` from the default build are refusals, recorded as passes, and they do not stand in for a native or device run.

## Cleanup

```sh
.cursor/skills/verify-invar/scripts/invar-verify cleanup --dry-run --format json
.cursor/skills/verify-invar/scripts/invar-verify cleanup --format json
```

Cleanup deletes `$INVAR_VERIFY_ROOT/scratch` only (generated ROS 2 trees and scratch programs). It does not delete `evidence/`, `target/`, or anything in the checkout. It does not kill processes by name. `--dry-run` lists the directory it would remove. After cleanup, `evidence/suite.json` must still be there.

`drive robotics` and `build` and `test` are the other side-effecting commands. They honour `--dry-run`.

## Helpers

The helper is `.cursor/skills/verify-invar/scripts/invar-verify`. It is executable. Run it by that path. `--help` lists every subcommand, the evidence root, and which commands are destructive.

```sh
.cursor/skills/verify-invar/scripts/invar-verify --help
```

Useful environment:

- `INVAR_VERIFY_ROOT` — scratch and evidence parent. Default `/tmp/invar-verify`.
- `INVAR_VERIFY_BIN` — binary to drive. Default `<repo>/target/debug/invar`.
- `INVAR_VERIFY_JOBS` — cargo parallelism. Default 2.

## What this host cannot prove

The default suite records these as `skipped` when the probe says so:

- `codegen.native-serial-run` — LLVM 21.1.8 is not installed (`native` is not rustc's LLVM).
- `codegen.cuda-hardware` — no NVIDIA GPU, and the binary has no `cuda` feature. This covers `just test-gpu-cuda`.
- `codegen.metal-hardware` — not an Apple Silicon Mac. This covers `just test-metal`.
- `robotics.ros2-runtime` — `ros2` is not on PATH and there is no robot. Generation of the workspace is still checked.
- `robotics.component-test` — needs a foreign shared library and the fixture server.
- `tests.full-nextest` — `cargo-nextest` is absent, and the full Debug+Release corpus is the CI gate rather than the cloud proof.

CPU reference, `cpu-parallel`, explain, `source-device` CPU model, and `gnet-device` CPU model are not in that list. They run.

## Maintenance

When the compiler's user-facing commands change, run `/maintain-verification-skill`. It should re-read this file and `references/features/`, drive each feature, and edit only this skill directory. A product failure belongs in the report, not in a softened assertion.
