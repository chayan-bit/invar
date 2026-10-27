# Parallelism

Users ask the compiler for a parallel CPU plan, an automatic plan, or an interaction-net CPU model. Admitted wrapping maps, reductions, and scans return the same value a serial run would. A checked sum that can trap stays a trap. Explain reports admission without executing the region.

## Sub-features

- `parallelism.map-workers-1` and `parallelism.map-workers-2` both return 120 for the wrapping map.
- `parallelism.reduction-workers-2` returns 7 for the wrapping reduction.
- `parallelism.scan-workers-2` returns 26 for the affine scan.
- `parallelism.checked-sum-still-traps` traps with `INTEGER_OVERFLOW` under `cpu-parallel`.
- `parallelism.sum-stays-serial` returns 10 for the checked i32 sum even when a parallel engine is requested.
- `parallelism.explain-map` admits `main` as a map.
- `parallelism.explain-sum` rejects `sum` as serial.
- `parallelism.auto-map` returns 120 on `--engine auto` in the default build.
- `parallelism.gnet-cpu-model-cuda` and `parallelism.gnet-cpu-model-metal` return 12 from the CPU group model.
- `parallelism.cpu-parallel-without-workers` is a usage error.

## How to get to it (user POV)

- Run `invar run PATH --engine cpu-parallel --workers N --format json` with `N` from 1 to 64.
- Run `invar explain PATH --format json`.
- Run `invar run PATH --engine auto --format json`.
- Run `invar gnet-device PATH ENTRY cpu-model-cuda|cpu-model-metal ARGS_JSON CONFIG_JSON`.

## Driving it with invar-verify

Preconditions:

- `invar-verify doctor --format json` reports `ready: true`.
- The parallel examples are `examples/parallel/map-wrapping.inv`, `reduction-wrapping-sum.inv`, `scan-affine.inv`, and `reduction-checked-sum.inv`.

- **Agree across worker counts.** Run `invar-verify drive parallelism --format json`. The map checks run `invar run examples/parallel/map-wrapping.inv --engine cpu-parallel --workers 1 --format json` and the same command with `--workers 2`. Both exit 0 with `result.value` 120.
- **Reduction and scan.** The drive runs the wrapping reduction and the affine scan with `--workers 2`. Values are 7 and 26.
- **Checked sum.** The drive runs `reduction-checked-sum.inv` with `--engine cpu-parallel --workers 2 --format json`. Exit 3 and `trap.category` is `INTEGER_OVERFLOW`. It also runs `examples/sum.inv` the same way and requires `result.value` 10.
- **Explain.** The drive runs `invar explain examples/parallel/map-wrapping.inv --format json` and requires region `main` to be `admitted` with execution kind `map`. It runs `invar explain examples/sum.inv --format json` and requires region `sum` to be `rejected` with kind `serial`.
- **Auto.** The drive runs `invar run examples/parallel/map-wrapping.inv --engine auto --format json`. `result.value` is 120. On a default build this is a CPU plan, not a GPU launch.
- **Interaction net.** The drive writes the two-call `entry` program and runs `invar gnet-device <file> entry cpu-model-cuda '[5]' '<config>'`, then the same with `cpu-model-metal`. Both receipts have `result` 12, `status` 0, and `execution.evidence_backend` `CpuGroupModel`. The config is `{"fuel":1000,"transitions":10000,"trace_events":2000,"quantum":2,"rounds":1,"capacity":2}`.
- **Missing workers.** The drive runs `cpu-parallel` without `--workers`. Exit 2 and the help text is on stderr.

## Gotchas

- `--workers` is required for `cpu-parallel` and rejected for other engines. Omitting it is exit 2, not a silent serial run.
- Worker counts outside 1..64 are a usage error.
- Small examples do not compile native region kernels. Those kernels need the `native` feature and a region of at least 2^23 elements. The value here is the scheduler's CPU plan.
- `explain` exits 0 when every region is rejected. Read `regions[].status`. Admission is not a speedup claim.
- `gnet-device` CPU models do not open a GPU. A forced `cuda` or `metal` backend is the codegen feature's refusal check.
- The CPU model compiles C++20 that includes `<barrier>`, using `CXX` or `c++`. If `c++` is clang and cannot see that header, the drive sets `CXX=g++` when `g++ -std=c++20` can compile it. If neither compiler can, the two gnet checks are skipped with that reason.
- The gnet config object must have exactly the six fields the command requires. An extra field is an error.
