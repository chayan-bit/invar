//! M3-03 fresh-process forced CUDA timing control (the in-process fused-map suite is
//! `tests/gpu_cuda/m3_03.rs` in the `invar` package).
#![cfg(feature = "cuda")]
#[path = "support/invar_bin.rs"]
mod invar_bin;
use std::{process::Command, time::Instant};

const CHAIN: &str = "fn main() -> i32 { \
    let xs: Array<u32> = [1u32,2u32,3u32]; \
    var a: Array<u32> = filled_u32(len(xs),0u32); \
    for i in 0u32 .. len(xs) { a[i] = wrapping_add(xs[i],3u32); } \
    let first: Array<u32> = move a; \
    var b: Array<u32> = filled_u32(len(first),0u32); \
    for j in 0u32 .. len(first) { b[j] = wrapping_mul(first[j],7u32); } \
    let second: Array<u32> = move b; \
    var c: Array<u32> = filled_u32(len(second),0u32); \
    for k in 0u32 .. len(second) { c[k] = wrapping_add(second[k],11u32); } \
    if c[0u32] != 39u32 { return 1; } \
    if c[1u32] != 46u32 { return 1; } \
    if c[2u32] != 53u32 { return 1; } return 0; }";

#[test]
#[ignore = "requires a CUDA GPU and NVRTC 12.4+"]
fn issue_m3_03_gpu_cold_process_fused_map_matches_reference() {
    let cold_started = Instant::now();
    let cold = Command::new(env!("CARGO_BIN_EXE_invar-m3-03-cold"))
        .arg(CHAIN)
        .arg(invar_bin::invar_bin("invar-nvrtc"))
        .output()
        .unwrap();
    assert!(
        cold.status.success(),
        "{}",
        String::from_utf8_lossy(&cold.stderr)
    );
    eprintln!(
        "cold_process_wall_ms={} {}",
        cold_started.elapsed().as_secs_f64() * 1000.0,
        String::from_utf8_lossy(&cold.stdout).trim()
    );
}
