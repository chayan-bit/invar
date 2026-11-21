//! CUDA adapters for the #207 extremum device test shared with Metal.
use invar::{eval, mir, schedule::cuda};
use std::path::Path;

const BACKEND: &str = "cuda";

fn run_device(
    source: &[u8],
    program: &mir::Program,
    fuel: u64,
) -> Result<(eval::Observation, usize), String> {
    let compiler = Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    cuda::run_source(source, program, fuel, compiler)
        .map(|run| (run.observation, run.executed.len()))
        .map_err(|error| error.code.to_owned())
}

#[path = "../support/gpu_extremum.rs"]
mod body;
