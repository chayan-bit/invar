# invar-foreign

The reviewed `unsafe` boundary for loading and calling foreign and compiler-generated libraries: it loads a library with `libloading`, calls the narrow Invar C ABI and checks every buffer and release callback.
The core compiler stays `forbid(unsafe_code)`.

## Where to start

- [src/lib.rs](src/lib.rs) defines the ABI types (`Arg`, `Output`, `Mode`, `InvArg`, `InvResult`), `Error` and `call`, the scalar entry.
- [src/component.rs](src/component.rs) is the component host used by the system dialect and the robot planner; [src/host_array.rs](src/host_array.rs) implements host array arguments (HA-1); [src/exports.rs](src/exports.rs) is the export boundary.
- [src/numeric.rs](src/numeric.rs) and [src/numeric_owned_resume.rs](src/numeric_owned_resume.rs) are the loaded numeric hosts; [src/device.rs](src/device.rs) is the private CPU device-model driver and [src/region.rs](src/region.rs) runs the native region kernels; both load compiler-generated libraries, not the public C ABI.

## Position in the pipeline

Called from [src/system](../../src/system/README.md), [src/exports](../../src/exports/README.md) and the general-net numeric hosts; the fixtures under [fixtures](fixtures) are the C libraries the tests load.

## Tests

- `cargo nextest run --locked -p invar-foreign`, plus `cargo nextest run --locked --test robotics` and `--test exports`, which drive it end to end.

## Rules to keep

- Every `unsafe` block states the pointer-validity condition it relies on.
- A foreign failure is a typed error with a record; never trust a returned length, tag or pointer without validating it.
- See [docs/host-array-arguments-ha1.md](../../docs/host-array-arguments-ha1.md) for the frozen array ABI.
