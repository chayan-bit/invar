//! Every CUDA hardware test in one binary, one module per family.
//! Each test is `#[ignore]`d without a GPU; `just test-gpu-cuda` runs them with `--include-ignored`.
//! `cargo test --features cuda --test gpu_cuda -- --ignored <family>::` runs one family.
//! CI only compiles this binary (`cargo check --features cuda --all-targets`).
#![cfg(feature = "cuda")]

#[path = "../support/gnet_cuda_aot.rs"]
mod aot;
#[path = "../support/gnet_device.rs"]
mod device;
#[path = "../support/gnet_parallel.rs"]
mod parallel;

mod a1_grid;
mod a1_indexed;
mod a1_segmented;
mod a1_tensor_dispatch;
mod eng_03_tier;
mod gnet_03;
#[cfg(unix)]
mod gnet_03_nvrtc_pipes;
mod gnet_cuda_aot;
mod gnet_p0b;
mod law_bench;
mod law_collective;
mod m3_01;
mod m3_02;
mod m3_03;
mod m3_04;
mod m3_05;
mod m4_01;
mod m4_01_nvrtc_modes;
mod m4_02;
mod m4_03;
mod m4_04;
mod m4_05;
mod m4_07;
mod pair_reduction;
mod sort_fsum;
mod task_region;
mod v02_fsum;
mod v02_math;
mod v02_spatial;
