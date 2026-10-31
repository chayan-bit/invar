# invar-math

The reviewed FFI boundary to the vendored CORE-MATH library, which supplies the correctly rounded `f32` and `f64` functions of `invar-math-cr-v1`.
`build.rs` compiles the unmodified sources with `-ffp-contract=off` into one static archive; the same C text is the source of the CUDA and Metal emitters, so every engine runs one algorithm.

## Where to start

- [src/lib.rs](src/lib.rs) defines `Function`, the raw-bit entry points `f32_bits` and `f64_bits` (callers never depend on host floating-point state), and `source` for the device emitters.
- [src/device.rs](src/device.rs) and [src/msl.rs](src/msl.rs) with [src/msl_prelude.h](src/msl_prelude.h) produce device source; [src/cxx.rs](src/cxx.rs) serves the C++ kernels; [src/sources.rs](src/sources.rs) lists the vendored sources.
- [core-math](core-math) is the vendored upstream; [oracle](oracle) holds the independent checkers (`driver.c`, `softdouble_check.cpp`).

## Position in the pipeline

The numeric library behind [src/math](../../src/math/README.md); evaluated by the reference, MIR, native and device engines.

## Tests

- `cargo nextest run --locked -p invar-math` and `cargo nextest run --locked --test fsum_math_sort_spatial`.

## Rules to keep

- Do not edit the vendored sources; record the upstream commit if you update them.
- Results must stay bit-identical to an independent CORE-MATH build on every engine.
