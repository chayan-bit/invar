# metal_aot

Compiled-ahead Metal libraries: the `INVGMETL` container written by `invar build --target metal`, loaded at run time without compiling any source.

## Where to start

- `encode` and `decode` in [mod.rs](mod.rs) define and validate the container (`Header`, `Image`, `Route`, `Error`); every field is bounded and checked.
- [build.rs](build.rs) runs Apple's offline compiler and provides `read` and `write_atomic`; [kernels.rs](kernels.rs) generates the run and region MSL; [../metal_build.rs](../metal_build.rs) owns the CLI build.

## Position in the pipeline

Used by [schedule](../schedule/README.md) (`schedule/metal.rs`) and `--artifact` on `run` and `explain`.

## Tests

- `cargo nextest run --locked --test device` (`issue_metal_aot_format` checks the container on any host).
- `tests/gpu_metal` (`metal_aot`, `metal_archives`) needs Apple Silicon and `--features metal`.

## Rules to keep

- Preserve source and toolchain identity and container bounds.
- Runtime loading must never silently compile source or accept a stale container; the digests give integrity, not authenticity.

## Read next

- [explain JSON](../../docs/explain.md#device-execution)
