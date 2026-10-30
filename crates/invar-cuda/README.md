# invar-cuda

The CUDA ownership boundary.
A submitted `Work` owns its context, stream, event, allocations and host staging until completion is confirmed, and a failed or uncertain completion quarantines the memory rather than reusing it.
`unsafe` is confined to the driver code behind the `runtime` feature and the page probe, and `unsafe_op_in_unsafe_fn` is denied.

## Where to start

- [src/lib.rs](src/lib.rs) defines `Runtime`, `Prepared`, `Submitted`, the `Error` and `SubmitError` types and `validate_transfer_len`; the generic `Device` trait lets tests run the protocol without a GPU.
- [src/cuda.rs](src/cuda.rs) and [src/cuda/](src/cuda) are the real driver and NVRTC boundary, built only with the `runtime` feature (the root crate's `cuda` feature turns it on).
- [src/pages.rs](src/pages.rs) has `observe`, an advisory, racy snapshot of host-page residency that never reads or copies the input.
- `compile_cubin` and `compile_gnet_cubin` compile kernel text to a CUBIN under the limits in [docs/cuda-target-v1.md](../../docs/cuda-target-v1.md).

## Position in the pipeline

The last step of a CUDA run: `src/schedule/cuda.rs` and the GNET device host ([src/general_net/device_host](../../src/general_net/device_host)) submit plans from [invar-device](../invar-device/README.md) here.

## Tests

- `cargo nextest run --locked -p invar-cuda` runs the protocol tests on any host.
- `cargo build --locked --features cuda` is the compile-only check CI runs without a GPU.
- Suites that need a GPU live in `tests/gpu_cuda`, are `#[ignore]`d, and run with `just test-gpu-cuda` on an NVIDIA host, or `just test-gpu-cuda-remote <ref>` for a remote Windows CUDA host; see [CONTRIBUTING.md](../../CONTRIBUTING.md#running-the-gpu-tests-locally).

## Rules to keep

- Hardware completion controls publication; never publish before the completion event.
- Do not release device memory whose completion is unknown.
