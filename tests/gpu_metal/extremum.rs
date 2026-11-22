//! Metal adapters for the #207 extremum device test shared with CUDA.
use invar::schedule::metal::{self, Kernels};
use invar::{eval, mir};

const BACKEND: &str = "metal";

fn run_device(
    source: &[u8],
    program: &mir::Program,
    fuel: u64,
) -> Result<(eval::Observation, usize), String> {
    metal::run_source(source, program, fuel, Kernels::RuntimeSource)
        .map(|run| (run.observation, run.executed.len()))
        .map_err(|error| error.code.to_owned())
}

#[path = "../support/gpu_extremum.rs"]
mod body;
