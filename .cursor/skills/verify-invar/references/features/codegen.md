# Codegen

Users can run an admitted source kernel on a CPU model of the device, or ask for LLVM, CUDA, or Metal. This checkout's verification binary is the default-feature build. CPU models run. LLVM 21, NVIDIA hardware, and Apple Metal are skipped when the host probe says they are absent, and the default build's refusals are checked so a missing backend cannot look like a successful device run.

## Sub-features

- `codegen.transpose-cpu-model` runs `examples/families/transpose.inv` on `cpu-model` and returns 7.
- `codegen.transpose-fuel-limit` stops that run at fuel 1 before a completed region.
- `codegen.tensor-cpu-model` runs the tensor contraction fixture on `cpu-model` and returns 7.
- `codegen.cuda-refuses-without-feature` reports `CUDA_UNAVAILABLE`.
- `codegen.metal-refuses-without-feature` reports `METAL_UNAVAILABLE`.
- `codegen.native-refuses-without-feature` reports `NATIVE_UNAVAILABLE`.
- `codegen.gnet-cuda-refuses` and `codegen.gnet-metal-refuses` refuse forced device backends.
- `codegen.native-serial-run`, `codegen.cuda-hardware`, and `codegen.metal-hardware` are skipped unless the host can actually run them.

## How to get to it (user POV)

- Run `invar source-device PATH transpose|tensor ENTRY cpu-model FUEL`.
- Run `invar run PATH --engine native-serial|cuda|metal --format json`.
- Run `invar build PATH --target native|cuda|metal ...` when the matching feature is compiled in.
- Run `just test-gpu-cuda` or `just test-metal` on matching hardware. Those are not the cloud proof.
- Run `just test-gpu-cuda-remote <ref> [filters]` with `REMOTE_CUDA_HOST` set to the ssh alias of a Windows CUDA host. It runs that ref's CUDA suites there and exits 0 pass, 1 test failures, 2 build failure, 3 GPU busy, 4 timeout, or 5 ssh dropped. It is not the cloud proof either.

## Driving it with invar-verify

Preconditions:

- `invar-verify doctor --format json` reports `ready: true` for the default-feature binary.
- Doctor's host probe is the source of skip reasons. Do not invent a GPU.

- **Transpose CPU model.** Run `invar-verify drive codegen --format json`. The transpose check runs `invar source-device examples/families/transpose.inv transpose transpose cpu-model 100000`. Exit 0, `schema` is `invar-private-source-device-v1`, `backend` is `cpu-model`, `result` is 7, `completed_regions` is 1, and `gpu_execution_claim` is absent.
- **Fuel limit.** The same command with fuel `1` fails. `completed_regions` is 0 and `error` contains `EvaluationLimit`.
- **Tensor CPU model.** The drive writes the contraction fixture from the source-device CLI test and runs `invar source-device <file> tensor mat cpu-model 100000`. `result` is 7 and `completed_regions` is 1.
- **Refusals.** The drive runs `invar run examples/sum.inv --engine cuda --format json`, the metal engine, and the native-serial engine. Each exits 2 with `tool_error` and code `CUDA_UNAVAILABLE`, `METAL_UNAVAILABLE`, or `NATIVE_UNAVAILABLE`. Forced `gnet-device ... cuda` and `... metal` fail and the error names the missing feature.
- **Skipped hardware.** The drive records `codegen.cuda-hardware` as skipped when `nvidia-smi` is missing or lists no device. It records `codegen.metal-hardware` as skipped when the host is not Darwin. It records `codegen.native-serial-run` as skipped when LLVM 21.1.8 is not installed and `LLVM_SYS_211_PREFIX` is unset. The skip names `just test-gpu-cuda` (its reason also names `just test-gpu-cuda-remote`), `just test-metal`, or `cargo build --locked --features native`.

## Gotchas

- `examples/families/matrix-f32.inv` is the reference-engine matrix example. The `source-device` tensor entry uses the contraction shape in the CLI test (a local `sum`, then one store). Do not point `source-device` at the family file and call a mismatch a hardware failure.
- A CPU-model report uses CUDA capability labels so the model matches the device contract. The backend field is still `cpu-model`. There is no GPU execution claim.
- `invar build --target native` refuses an existing output path and needs a filesystem that supports hard links. The default drive does not invoke it.
- Rustc 1.95 ships its own LLVM. That LLVM does not satisfy the `native` feature. The pinned backend is LLVM 21.1.8.
- Building `--features cuda` or `--features metal` on a host that cannot run them is not a hardware pass.
