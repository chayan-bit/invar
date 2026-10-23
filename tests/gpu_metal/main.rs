//! Every Apple Metal hardware test in one binary, one module per family.
//! Each test is `#[ignore]`d without a GPU; `just test-metal` runs them with `--include-ignored`.
//! `cargo test --features metal --test gpu_metal -- --ignored <family>::` runs one family.
//! CI only compiles this binary on a macOS runner (`cargo check --features metal --all-targets`).
#![cfg(all(feature = "metal", target_os = "macos"))]

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
mod gnet_p0b;
mod law_bench;
mod law_collective;
mod m3_06;
mod m4_01;
mod m4_03;
mod m4_04;
mod m4_07;
mod metal_aot;
mod metal_archives;
mod pair_reduction;
mod sort_fsum;
mod task_region;
mod v02_fsum;
mod v02_math_msl;
mod v02_spatial;
