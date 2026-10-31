//! CUDA adapters for the ENG-03 fold-tier device test shared with Metal.
use invar::{mir, schedule::cuda};
use std::path::Path;

const BACKEND: &str = "cuda";

fn run_forced(source: &[u8], program: &mir::Program, fuel: u64) -> cuda::Run {
    let compiler = Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    cuda::run_source(source, program, fuel, compiler).unwrap()
}

#[path = "../support/gpu_eng_03_tier.rs"]
mod body;
