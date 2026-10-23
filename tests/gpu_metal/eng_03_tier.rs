//! Metal adapters for the ENG-03 fold-tier device test shared with CUDA.
use invar::mir;
use invar::schedule::metal::{self, Kernels};

const BACKEND: &str = "metal";

fn run_forced(source: &[u8], program: &mir::Program, fuel: u64) -> metal::Run {
    metal::run_source(source, program, fuel, Kernels::RuntimeSource).unwrap()
}

#[path = "../support/gpu_eng_03_tier.rs"]
mod body;
