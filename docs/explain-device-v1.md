# Device explanation v1

`invar explain PATH --engine cuda|metal|auto --format json` extends the M2 report with
`device_execution`. It runs the same guarded MIR plan once with the default
1,000,000 fuel limit and reports choices actually applied in that run. Invar
source has no external I/O. The plain `explain PATH --format json` interface
retains its static M2 report.

`device_execution.requested_engine` is `cuda`, `metal`, or `auto`; `result` is the reached
`main` i32. Each `choices` entry names the function and source span, backend,
reason, and (for `auto`) estimated CPU/device milliseconds. A forced device
choice is `cuda` or `metal` only after a completed result passes the live MIR
publication guard. `auto` may choose `cpu-parallel` from the measured cost model,
or `mir-serial` when a live guard is false. The model measures CPU cost from
the real cpu-parallel element executor on a one-operation
1,000,000-element map, device fixed cost from the submit-to-completion time of a
trivial one-thread kernel, and transfer bandwidth from a host/device buffer
round trip. It then compares transfer bytes, launch count and operation count
with a 10% device-win margin. Compilation is excluded because compiled modules
are reused in-process. Metal caches measured models per CPU worker count.
CPU workers use available parallelism, preferring the number of performance
cores on Apple Silicon, with a hard bound of 64.
`INVAR_CPU_WORKERS=1..64` overrides that preference within the OS affinity/quota
limit; invalid values return `CPU_WORKERS`. Explicit `cpu-parallel --workers`
requests accept 1..64. Calibration measures the actual selected CPU width.
`module_cache_hits`, `schedule_hint_hits` and
`schedule_hint_invalids` are counts for this process.

The execution path always reconstructs the region via `plan::decide`, lowers
the graph-extracted plan into the shared device IR, and reruns the verifier.
The CUDA module cache contains at most eight live-device entries keyed by
exact verified IR, capabilities, generated source and kernel names.
The Metal pipeline cache is one bounded LRU per device, shared by the map, scan and general-net routes.
It holds at most 16 pipelines keyed by the SHA-256 of the exact generated MSL and the function name, so two different kernels never share an entry and the three scan passes share one library.
Persistent
schedule hints contain only bounded geometry and descriptive identity: exact
source FNV-1a identity, typed contract, rule revisions, pinned toolchain,
device capability, layout, numeric policy, ABI and generated source hash.
Untrusted hints are compared with a freshly regenerated record before use.
No persisted source, PTX, CUBIN, Metal library or module is loaded from a hint.

Metal kernels are compiled ahead.
`invar build PATH --target metal --output FILE` runs Apple's offline compiler (`xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -mmacosx-version-min=14.0 -c`, then `metallib`) over every Metal region kernel and the calibration probe.
It writes a private, atomically published `INVGMETL` container; the general-net route adds `--entry NAME`.
`run`, `explain`, `gnet-device` and tensor `source-device ... metal` take `--artifact FILE` and load its libraries with `newLibraryWithData`; production never calls `newLibraryWithSource`.
The tensor route builds with `invar build PATH --target metal --entry ENTRY --family tensor --fuel FUEL --output FILE`: an exact CPU run on this Mac's Metal capabilities records the D1 probe and every shape-specific contraction the source reaches within `FUEL`.
A contraction missing from that container (another Mac's capabilities, or more fuel at run time) is a recorded `tensor_submit` refusal and that loop runs on the CPU.
Without an artifact, a forced Metal run fails with `METAL_UNAVAILABLE` and `auto` records `device-unavailable` and runs on the CPU, as on a machine without a GPU.
A corrupt, oversized, stale, wrong-route, wrong-entry, group- or world-writable or toolchain-rejected container, fails with `DEVICE_TOOLCHAIN_MISMATCH` under forced Metal and `auto` alike, and never compiles source.
A kernel missing from an accepted container is refused the same way when forced; under `auto` it is the recorded `metal-failed-before-publication` CPU result.
A build without Metal refuses `--artifact` with `METAL_UNAVAILABLE` before choosing a backend.
The container is opened once without following a final symlink or blocking on a FIFO, and every check and the read use that descriptor.
`INVAR_METAL_RUNTIME_SOURCE=1` is the developer-only runtime source path, off by default.
The container digest gives integrity, not authenticity.

A region whose Metal kernel would bind more than 31 buffer arguments (the
Apple GPU limit) is Metal-ineligible at planning time: `auto` records reason
`metal-buffer-slot-limit` and runs it on CPU.
Metal has no binary64, and Invar does not emulate it: under `--engine metal` or
`auto` on Metal, every function that computes an `f64` value is listed once, at
its first such instruction, with backend `mir-serial` and reason `metal-no-f64`,
and runs on the CPU.
The `metal-no-f64` reason was added to the frozen v1 enum with maintainer approval on 2026-10-19 ([frozen contract changes](contract-changes.md)).
Forced CUDA or Metal fails with `CUDA_UNAVAILABLE` or `METAL_UNAVAILABLE` for
missing hardware, ineligible regions or false live guards;
compilation/verification mismatch uses `DEVICE_TOOLCHAIN_MISMATCH`. Unknown
completion uses `CUDA_PENDING` or `METAL_PENDING` and permits no CPU replay.
`auto` can fall back to a fresh CPU result only before device
submission or after confirmed quiescence. All failures use the existing
`tool_error` document and exit code 2.

## Python binding

The CLI and Python binding share `invar::engine` for guarded whole-program selection.
`compile_source(..., engine="auto"|"cuda"|"metal").run_main(observe=True)` returns a receipt from that completed invocation.
`receipt()["backends"]` lists published region backends, or `mir-serial` when no region was published.
`explain()` includes the frozen `device_execution` document after a successful device/auto run and does not execute again.
Before execution it returns the static AI-01 explanation without a device execution claim.
The frozen device-explain schema is unchanged.
HA-1.2 NumPy host exports retain their separate reference/native-serial boundary.
