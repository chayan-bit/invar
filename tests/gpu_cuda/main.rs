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

mod cp_async_pipeline;
mod f32_contraction;
mod fold_tier;
mod fsum;
mod fused_maps;
mod general_net_abi5;
mod general_net_aot;
mod general_net_entry;
mod generated_map;
mod grid;
mod guarded_integration;
mod indexed;
mod indexed_routing;
mod law_bench;
mod law_collective;
mod math_cr_v1;
mod nvrtc_modes;
#[cfg(unix)]
mod nvrtc_pipes;
mod pair_reduction;
mod reductions_scans;
mod runtime_round_trip;
mod segmented;
mod segmented_publication;
mod sort_fsum;
mod spatial;
mod summary;
mod task_region;
mod tensor_dispatch;
mod tiled_contraction;
mod transpose_stencil;
