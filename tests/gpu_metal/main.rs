//! Every Apple Metal hardware test in one binary, one module per family.
//! Each test is `#[ignore]`d without a GPU; `just test-metal` runs them with `--include-ignored`.
//! `cargo test --features metal --test gpu_metal -- --ignored <family>::` runs one family.
//! CI only compiles this binary on a macOS runner (`cargo check --features metal --all-targets`).
#![cfg(all(feature = "metal", target_os = "macos"))]

#[path = "../support/gnet_device.rs"]
mod device;
#[path = "../support/gnet_parallel.rs"]
mod parallel;

mod backend_archives;
mod compiled_ahead;
mod dispatch_and_kernels;
mod f32_contraction;
mod fold_tier;
mod fsum;
mod general_net_abi5;
mod general_net_entry;
mod grid;
mod indexed_routing;
mod law_bench;
mod law_collective;
mod math_msl;
mod pair_reduction;
mod segmented;
mod segmented_publication;
mod sort_fsum;
mod spatial;
mod summary;
mod task_region;
mod tensor_dispatch;
mod tiled_contraction;
