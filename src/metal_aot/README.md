# metal_aot

Compiled-ahead Metal libraries and bounded, validated container IO.
mod.rs defines Header, Image, Route, Error and encode/decode entry points.
build.rs invokes the offline toolchain and provides read and write_atomic.
kernels.rs generates run and region MSL; src/metal_build.rs owns CLI builds.

See [the architecture overview](../../docs/architecture.md) for the full pipeline and [validation](../../docs/validation.md) for the checks.
Preserve source/toolchain identity, container bounds and typed refusals.
Runtime loading must not silently compile source or accept a stale container.
