# math

The `invar-math-cr-v1` source library for `f32` and `f64`: correctly rounded `sqrt`, `sin`, `cos`, `exp` and the rest, evaluated through the vendored CORE-MATH sources.

## Where to start

- `MathFn`, `eval_f32`, `eval_f64` and `contracts` in [../math.rs](../math.rs); `cr_v1_device_sites` and `f64_sites` find the functions a device must keep on the CPU.
- [device.rs](device.rs) emits the CUDA elementwise kernels (`cuda_map_source`) and checks NVRTC admission.

## Position in the pipeline

Checked by [check](../check/README.md), evaluated by [eval](../eval/README.md) and the native and device engines; the C sources are in [crates/invar-math](../../crates/invar-math/Cargo.toml).

## Tests

- `cargo nextest run --locked --test fsum_math_sort_spatial` (`math_*`, `math_cr_default`).
- Hardware: `tests/gpu_cuda/math_cr_v1.rs`, `tests/gpu_metal/math_msl.rs`.

## Rules to keep

- Results are bit-identical to an independent build of CORE-MATH on every engine.

## Read next

- [specification Appendix B.4](../../docs/spec/invar-v0.1.md#b4-binary64-and-the-correctly-rounded-math-library)
- [numeric contracts](../../docs/numeric.md)
