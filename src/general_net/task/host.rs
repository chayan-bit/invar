//! Device execution of an admitted task region: emit for the target and the
//! plan's schedule, prepare the runner once, then run calls and decode the
//! root observation. The runners own buffers, submissions and timing
//! (`invar_metal::task`, `invar_cuda::cuda::task`).
//!
//! Metal probes MSL 3.2 at runtime ([`metal::profile`]): the cascade schedule
//! where the OS and GPU support it, else the staged MSL 3.1 fallback. CUDA
//! always runs the cascade. Every [`Outcome`] records the path that ran.

use crate::eval::{EvalError, Value};

use super::{Plan, Program, decode, encode_args};

/// Submissions after which an unfinished call is reported as stalled.
pub const MAX_SUBMISSIONS: u32 = 1 << 16;

/// One device call: the observation, the path that ran and the scheduler
/// counters.
#[derive(Debug)]
pub struct Outcome {
    pub result: Result<Value, EvalError>,
    /// Runner path: `metal-msl3.2-cascade`, `metal-msl3.1-staged` or
    /// `cuda-cascade`.
    pub path: &'static str,
    pub submissions: u32,
    pub gpu_seconds: f64,
    /// Staged: scheduler rounds, tasks run, scheduler passes, arena high
    /// water. Cascade: nodes claimed, serial runs, 1, arena top.
    pub rounds: u32,
    pub tasks: u32,
    pub passes: u32,
    pub arena: u32,
}

fn outcome(
    program: &Program,
    path: &'static str,
    state: &[u32],
    submissions: u32,
    gpu_seconds: f64,
) -> Outcome {
    let cascade = path.ends_with("cascade");
    Outcome {
        result: decode(program, [state[2], state[3], state[4], state[5]]),
        path,
        submissions,
        gpu_seconds,
        rounds: state[if cascade { 24 } else { 16 }],
        tasks: state[17],
        passes: if cascade { 1 } else { state[18] },
        arena: state[if cascade { 6 } else { 19 }],
    }
}

fn words(program: &Program, args: &[Value]) -> Result<Vec<u32>, String> {
    encode_args(program, args).ok_or_else(|| "argument types do not match the entry".to_owned())
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal {
    use invar_metal::{
        MetalDevice,
        task::{CascadeShape, CascadeWork, TaskShape, TaskWork},
    };

    use super::*;
    use crate::general_net::{
        compiled::Target,
        task::{CASCADE_KERNEL, Mutation, Profile, RUN_KERNEL, SCHED_KERNEL, Schedule, emit},
    };

    pub const CASCADE_PATH: &str = "metal-msl3.2-cascade";
    pub const STAGED_PATH: &str = "metal-msl3.1-staged";

    /// The profile this device runs: the MSL 3.2 cascade where the OS and GPU
    /// support it (runtime probe) on the measured device class, else the staged
    /// MSL 3.1 fallback.
    pub fn profile(device: &MetalDevice) -> Profile {
        if device.supports_msl_3_2()
            && crate::general_net::task::cost::metal_cascade_measured(&device.name())
        {
            Profile::METAL
        } else {
            Profile::METAL_STAGED
        }
    }

    enum Work {
        Staged(TaskWork),
        Cascade(CascadeWork),
    }

    pub struct MetalTask {
        work: Work,
        program: Program,
    }

    impl MetalTask {
        /// Prepare `program` under `plan`. A cascade plan on a device without
        /// MSL 3.2 is refused (use [`profile`] to pick the plan).
        pub fn prepare(
            device: &mut MetalDevice,
            program: &Program,
            plan: &Plan,
            mutation: Mutation,
        ) -> Result<Self, String> {
            let source =
                emit(program, Target::Metal, plan, mutation).map_err(|r| r.code().to_owned())?;
            let work = match plan.schedule {
                Schedule::Staged => Work::Staged(
                    device
                        .prepare_task(
                            &source.text,
                            SCHED_KERNEL,
                            RUN_KERNEL,
                            TaskShape {
                                words: source.words,
                                sched_threads: plan.sched_threads as usize,
                                run_threads: plan.run_threads as usize,
                                epochs: plan.epochs as usize,
                                max_submissions: MAX_SUBMISSIONS,
                            },
                        )
                        .map_err(|e| format!("{e:?}"))?,
                ),
                Schedule::Cascade => Work::Cascade(
                    device
                        .prepare_cascade(
                            &source.text,
                            CASCADE_KERNEL,
                            CascadeShape {
                                words: source.words,
                                threads: plan.run_threads as usize,
                                groups: plan.lanes.div_ceil(plan.run_threads.max(1)) as usize,
                            },
                        )
                        .map_err(|e| format!("{e:?}"))?,
                ),
            };
            Ok(Self {
                work,
                program: program.clone(),
            })
        }

        /// The runner path this task runs.
        pub fn path(&self) -> &'static str {
            match self.work {
                Work::Staged(_) => STAGED_PATH,
                Work::Cascade(_) => CASCADE_PATH,
            }
        }

        pub fn call(&mut self, args: &[Value]) -> Result<Outcome, String> {
            let params = words(&self.program, args)?;
            let path = self.path();
            let run = match &mut self.work {
                Work::Staged(work) => work.call(&params),
                Work::Cascade(work) => work.call(&params),
            }
            .map_err(|e| format!("{e:?}"))?;
            Ok(outcome(
                &self.program,
                path,
                &run.state,
                run.submissions,
                run.gpu_seconds,
            ))
        }
    }
}

#[cfg(feature = "cuda")]
pub mod cuda {
    use std::path::Path;

    use invar_cuda::cuda::{
        CudaDevice,
        task::{CascadeShape, CascadeWork},
    };

    use super::*;
    use crate::general_net::{
        compiled::Target,
        task::{CASCADE_KERNEL, Mutation, Schedule, emit},
    };

    pub const CASCADE_PATH: &str = "cuda-cascade";

    pub struct CudaTask {
        work: CascadeWork,
        program: Program,
    }

    impl CudaTask {
        /// Prepare `program` under a cascade `plan` (CUDA runs no other).
        pub fn prepare(
            device: &mut CudaDevice,
            program: &Program,
            plan: &Plan,
            mutation: Mutation,
            compiler: &Path,
        ) -> Result<Self, String> {
            if plan.schedule != Schedule::Cascade {
                return Err("CUDA task regions run the cascade schedule".to_owned());
            }
            let source =
                emit(program, Target::Cuda, plan, mutation).map_err(|r| r.code().to_owned())?;
            let shape = CascadeShape {
                words: source.words,
                threads: plan.run_threads,
                blocks: plan.lanes.div_ceil(plan.run_threads.max(1)).max(1),
            };
            let work = device
                .prepare_cascade(&source.text, CASCADE_KERNEL, shape, compiler)
                .map_err(|e| format!("{e:?}"))?;
            Ok(Self {
                work,
                program: program.clone(),
            })
        }

        /// Blocks of the persistent grid after the residency clamp.
        pub fn blocks(&self) -> u32 {
            self.work.blocks()
        }

        pub fn call(&mut self, args: &[Value]) -> Result<Outcome, String> {
            let params = words(&self.program, args)?;
            let run = self.work.call(&params).map_err(|e| format!("{e:?}"))?;
            Ok(outcome(
                &self.program,
                CASCADE_PATH,
                &run.state,
                run.submissions,
                run.gpu_seconds,
            ))
        }
    }
}
