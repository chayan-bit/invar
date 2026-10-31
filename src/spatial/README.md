# spatial

Spatial intrinsics: voxel grids, radius neighbors and k-nearest neighbors over sorted grids, on CPU, CUDA and Metal.

## Where to start

- [../spatial.rs](../spatial.rs) defines the signatures and built-in result records; [runtime.rs](runtime.rs) builds the `Grid`; [cuda.rs](cuda.rs) and [metal.rs](metal.rs) bridge to device source; [placement.rs](placement.rs) holds the stable calibration identities (`UNMEASURED`).

## Position in the pipeline

Checked by [check](../check/README.md); programs that use it keep their other regions on the CPU.

## Tests

- `cargo nextest run --locked --test fsum_math_sort_spatial` (`spatial_*`); hardware in `tests/gpu_cuda/spatial.rs` and `tests/gpu_metal/spatial.rs`.

## Rules to keep

- Unmeasured spatial choices stay on the CPU with no invented cost.

## Read next

- [specification Appendix B.3](../../docs/spec/invar-v0.1.md#b3-spatial-grid-queries)
