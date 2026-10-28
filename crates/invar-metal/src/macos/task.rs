//! P4b task-region runner (ABI `gnet-task-v1`): one source with a single-group
//! scheduler kernel and a run kernel over four retained buffers. Submission
//! `i` encodes `min(2^i, epochs)` `(run, sched)` pairs, and the first one also
//! a leading scheduler pass (every later submission resumes after a scheduler
//! pass whose tasks are still pending), so a short call pays for no idle
//! passes and a long one for few submissions; the run kernel takes its
//! threadgroup count from the state buffer (indirect dispatch). The host
//! submits again until the root node is done. Source admission and the
//! observation decode live in root (`invar::general_net::task`).
//!
//! [`CascadeWork`] runs the cascade schedule instead: one persistent kernel,
//! compiled as MSL 3.2 (device-coherent buffers and device-scope fences), one
//! dispatch per call. It is offered only where [`MetalDevice::supports_msl_3_2`]
//! holds; the staged MSL 3.1 runner stays the exact fallback.
use super::*;

/// Launch shape of an emitted task region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaskShape {
    /// 32-bit words of the params, state, node and list buffers.
    pub words: [usize; 4],
    pub sched_threads: usize,
    pub run_threads: usize,
    pub epochs: usize,
    /// Submissions after which a call that has not finished is an error.
    pub max_submissions: u32,
}

/// The state buffer after the root finished, and timing.
#[derive(Clone, Debug)]
pub struct TaskResult {
    pub state: Vec<u32>,
    pub submissions: u32,
    /// Sum of GPU start-to-end times of the submissions, seconds.
    pub gpu_seconds: f64,
}

pub struct TaskWork {
    sched: PipelineRef,
    run: PipelineRef,
    buffers: Vec<BufferRef>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    shape: TaskShape,
}

const STATE_WORDS: usize = 32;
/// Byte offset of the run kernel's threadgroup count in the state buffer.
const RUN_GROUPS_OFFSET: usize = 13 * 4;
/// Byte offset of the next scheduler pass's threadgroup count (zero once the
/// root is done, so idle passes of a submission cost nothing).
const SCHED_GROUPS_OFFSET: usize = 20 * 4;

/// A trivial kernel using what the cascade needs from MSL 3.2.
const MSL_3_2_PROBE: &str = "#include <metal_stdlib>\nusing namespace metal;\nkernel void probe(coherent(device) device uint* p [[buffer(0)]],uint i [[thread_position_in_grid]]){p[i]=1u;atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);atomic_fetch_add_explicit((coherent(device) device atomic_uint*)p,1u,memory_order_relaxed);}\n";

fn compile(
    device: &ProtocolObject<dyn MTLDevice>,
    source: &str,
    version: MTLLanguageVersion,
    names: &[&str],
) -> Result<Vec<PipelineRef>, Error> {
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(version);
    #[allow(deprecated)]
    options.setFastMathEnabled(false);
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
        .map_err(|e| Error::Toolchain(format!("Metal source compilation failed: {e}")))?;
    names
        .iter()
        .map(|name| {
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| Error::Toolchain(format!("Metal function {name} missing")))?;
            device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| Error::Toolchain(format!("Metal pipeline compilation failed: {e}")))
        })
        .collect()
}

impl MetalDevice {
    /// Whether this OS and GPU support the cascade's MSL 3.2 features: the
    /// GPU reports the Metal 3 family and the OS's Metal compiler accepts and
    /// builds a kernel with device-coherent buffers and device-scope fences.
    pub fn supports_msl_3_2(&self) -> bool {
        self.device.supportsFamily(MTLGPUFamily::Metal3)
            && compile(
                &self.device,
                MSL_3_2_PROBE,
                MTLLanguageVersion::Version3_2,
                &["probe"],
            )
            .is_ok()
    }

    pub fn prepare_task(
        &mut self,
        source: &str,
        sched: &str,
        run: &str,
        shape: TaskShape,
    ) -> Result<TaskWork, Error> {
        self.admit()?;
        let bytes = shape.words.map(|w| w.saturating_mul(4));
        let total = bytes
            .iter()
            .try_fold(0usize, |a, n| a.checked_add(*n))
            .ok_or(Error::Limit)?;
        if source.len() > 4 * 1024 * 1024
            || bytes.contains(&0)
            || shape.words[1] < STATE_WORDS
            || shape.epochs == 0
            || total > invar_device::MAX_GNET_STORAGE_BYTES
        {
            return Err(Error::Limit);
        }
        let [sched, run]: [PipelineRef; 2] = compile(
            &self.device,
            source,
            MTLLanguageVersion::Version3_1,
            &[sched, run],
        )?
        .try_into()
        .map_err(|_| Error::Limit)?;
        if sched.maxTotalThreadsPerThreadgroup() < shape.sched_threads
            || run.maxTotalThreadsPerThreadgroup() < shape.run_threads
        {
            return Err(Error::Limit);
        }
        let buffers = bytes
            .iter()
            .map(|n| self.buffer(*n))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(TaskWork {
            sched,
            run,
            buffers,
            queue: self.queue.clone(),
            shape,
        })
    }
}

impl TaskWork {
    fn encode(
        &self,
        first: bool,
        epochs: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>, Error> {
        let command = command_buffer(&self.queue)?;
        let encoder = compute_encoder(&command)?;
        for (i, buffer) in self.buffers.iter().enumerate() {
            // SAFETY: the task ABI declares these four buffers at slots 0..4;
            // every capacity was checked at allocation; offset is 0.
            unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), 0, i) };
        }
        let one = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let sched = MTLSize {
            width: self.shape.sched_threads,
            height: 1,
            depth: 1,
        };
        let run = MTLSize {
            width: self.shape.run_threads,
            height: 1,
            depth: 1,
        };
        if first {
            encoder.setComputePipelineState(&self.sched);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(one, sched);
        }
        for _ in 0..epochs {
            encoder.setComputePipelineState(&self.run);
            // SAFETY: the scheduler writes three in-range threadgroup counts at
            // this aligned offset of the retained state buffer before the
            // serial encoder runs this dispatch.
            unsafe {
                encoder.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                    &self.buffers[1],
                    RUN_GROUPS_OFFSET,
                    run,
                )
            };
            encoder.setComputePipelineState(&self.sched);
            // SAFETY: as above; the previous scheduler pass wrote this count.
            unsafe {
                encoder.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                    &self.buffers[1],
                    SCHED_GROUPS_OFFSET,
                    sched,
                )
            };
        }
        encoder.endEncoding();
        command.commit();
        Ok(command)
    }

    fn state(&self) -> Vec<u32> {
        let mut state = vec![0u32; STATE_WORDS];
        // SAFETY: no command is in flight; the shared state buffer holds at
        // least STATE_WORDS initialized words.
        unsafe {
            ptr::copy_nonoverlapping(
                self.buffers[1].contents().as_ptr().cast::<u32>(),
                state.as_mut_ptr(),
                STATE_WORDS,
            )
        };
        state
    }

    /// Run one call with entry arguments `params` to completion.
    pub fn call(&mut self, params: &[u32]) -> Result<TaskResult, Error> {
        if params.len() > self.shape.words[0] {
            return Err(Error::Limit);
        }
        // SAFETY: no command is in flight (every submission below is awaited);
        // capacities were checked at allocation.
        unsafe {
            ptr::copy_nonoverlapping(
                params.as_ptr(),
                self.buffers[0].contents().as_ptr().cast::<u32>(),
                params.len(),
            );
            ptr::write_bytes(
                self.buffers[1].contents().as_ptr().cast::<u32>(),
                0,
                STATE_WORDS,
            );
        }
        let (mut submissions, mut gpu_seconds) = (0, 0.0);
        loop {
            let epochs = self.shape.epochs.min(1 << submissions.min(16));
            let command = self.encode(submissions == 0, epochs)?;
            command.waitUntilCompleted();
            if command.status() != MTLCommandBufferStatus::Completed {
                return Err(Error::Device(format!(
                    "Metal task command failed: {:?}",
                    command.error()
                )));
            }
            submissions += 1;
            gpu_seconds += command.GPUEndTime() - command.GPUStartTime();
            let state = self.state();
            if state[1] != 0 {
                return Ok(TaskResult {
                    state,
                    submissions,
                    gpu_seconds,
                });
            }
            if state[12] == 0 || submissions >= self.shape.max_submissions {
                return Err(Error::Device(format!(
                    "task region stalled after {submissions} submissions"
                )));
            }
        }
    }
}

/// Launch shape of an emitted cascade region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CascadeShape {
    /// 32-bit words of the params, state, node and list buffers.
    pub words: [usize; 4],
    pub threads: usize,
    pub groups: usize,
}

/// One cascade region: a persistent MSL 3.2 kernel over four retained
/// buffers, one dispatch per call.
pub struct CascadeWork {
    kernel: PipelineRef,
    buffers: Vec<BufferRef>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    shape: CascadeShape,
    /// Sequence of the last call (tags the nodes it publishes).
    seq: u32,
}

/// Call sequences stay below this so `seq << 3` keeps every bit.
const SEQ_LIMIT: u32 = 1 << 29;

impl MetalDevice {
    pub fn prepare_cascade(
        &mut self,
        source: &str,
        kernel: &str,
        shape: CascadeShape,
    ) -> Result<CascadeWork, Error> {
        self.admit()?;
        if !self.supports_msl_3_2() {
            return Err(Error::Unavailable(
                "the cascade schedule needs MSL 3.2 (OS or GPU lacks it)".into(),
            ));
        }
        let bytes = shape.words.map(|w| w.saturating_mul(4));
        let total = bytes
            .iter()
            .try_fold(0usize, |a, n| a.checked_add(*n))
            .ok_or(Error::Limit)?;
        if source.len() > 4 * 1024 * 1024
            || bytes.contains(&0)
            || shape.words[1] < STATE_WORDS
            || shape.threads == 0
            || shape.groups == 0
            || total > invar_device::MAX_GNET_STORAGE_BYTES
        {
            return Err(Error::Limit);
        }
        let [kernel]: [PipelineRef; 1] = compile(
            &self.device,
            source,
            MTLLanguageVersion::Version3_2,
            &[kernel],
        )?
        .try_into()
        .map_err(|_| Error::Limit)?;
        if kernel.maxTotalThreadsPerThreadgroup() < shape.threads {
            return Err(Error::Limit);
        }
        let buffers = bytes
            .iter()
            .map(|n| self.buffer(*n))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(CascadeWork {
            kernel,
            buffers,
            queue: self.queue.clone(),
            shape,
            seq: 0,
        })
    }
}

impl CascadeWork {
    /// Run one call with entry arguments `params` to completion. A stalled
    /// cascade (state word 1 = 2, a defect) is an error, never a result.
    pub fn call(&mut self, params: &[u32]) -> Result<TaskResult, Error> {
        if params.len() > self.shape.words[0] {
            return Err(Error::Limit);
        }
        self.seq = self.seq % (SEQ_LIMIT - 1) + 1;
        // SAFETY: no command is in flight (every call waits for completion);
        // capacities were checked at allocation.
        unsafe {
            ptr::copy_nonoverlapping(
                params.as_ptr(),
                self.buffers[0].contents().as_ptr().cast::<u32>(),
                params.len(),
            );
            let state = self.buffers[1].contents().as_ptr().cast::<u32>();
            ptr::write_bytes(state, 0, STATE_WORDS);
            state.add(25).write(self.seq);
            if self.seq == 1 {
                // A wrapped sequence could meet an old tag: clear the nodes.
                ptr::write_bytes(
                    self.buffers[2].contents().as_ptr().cast::<u8>(),
                    0,
                    self.shape.words[2] * 4,
                );
            }
        }
        let command = command_buffer(&self.queue)?;
        let encoder = compute_encoder(&command)?;
        for (i, buffer) in self.buffers.iter().enumerate() {
            // SAFETY: the task ABI declares these four buffers at slots 0..4;
            // every capacity was checked at allocation; offset is 0.
            unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), 0, i) };
        }
        encoder.setComputePipelineState(&self.kernel);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: self.shape.groups,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.shape.threads,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::Device(format!(
                "Metal cascade command failed: {:?}",
                command.error()
            )));
        }
        let gpu_seconds = command.GPUEndTime() - command.GPUStartTime();
        let mut state = vec![0u32; STATE_WORDS];
        // SAFETY: the command completed; the state buffer holds STATE_WORDS words.
        unsafe {
            ptr::copy_nonoverlapping(
                self.buffers[1].contents().as_ptr().cast::<u32>(),
                state.as_mut_ptr(),
                STATE_WORDS,
            )
        };
        if state[1] != 1 {
            return Err(Error::Device(format!(
                "task cascade stalled (state word 1 = {})",
                state[1]
            )));
        }
        Ok(TaskResult {
            state,
            submissions: 1,
            gpu_seconds,
        })
    }
}

/// Handwritten two-kernel baseline (bench only): a grid kernel then a
/// single-group top kernel over five buffers, one command buffer per call.
pub struct BaselineWork {
    grid: PipelineRef,
    top: PipelineRef,
    buffers: Vec<BufferRef>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

impl MetalDevice {
    pub fn prepare_baseline(
        &mut self,
        source: &str,
        grid: &str,
        top: &str,
        bytes: [usize; 5],
    ) -> Result<BaselineWork, Error> {
        self.admit()?;
        if bytes.contains(&0) || bytes[0] < 16 || bytes[4] < 8 {
            return Err(Error::Limit);
        }
        let [grid, top]: [PipelineRef; 2] = compile(
            &self.device,
            source,
            MTLLanguageVersion::Version3_1,
            &[grid, top],
        )?
        .try_into()
        .map_err(|_| Error::Limit)?;
        if grid.maxTotalThreadsPerThreadgroup() < 256 || top.maxTotalThreadsPerThreadgroup() < 256 {
            return Err(Error::Limit);
        }
        let buffers = bytes
            .iter()
            .map(|n| self.buffer(*n))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(BaselineWork {
            grid,
            top,
            buffers,
            queue: self.queue.clone(),
        })
    }
}

impl BaselineWork {
    /// Run with `params = [a, b, depth, groups]`; return the two output words
    /// (value, trap) and the GPU start-to-end time in seconds.
    pub fn call(&mut self, params: [i32; 4]) -> Result<([i32; 2], f64), Error> {
        let groups = usize::try_from(params[3]).map_err(|_| Error::Limit)?;
        if groups == 0 || groups > 256 {
            return Err(Error::Limit);
        }
        // SAFETY: no command is in flight (each call waits for completion);
        // the params buffer holds at least four words.
        unsafe {
            ptr::copy_nonoverlapping(
                params.as_ptr(),
                self.buffers[0].contents().as_ptr().cast::<i32>(),
                4,
            )
        };
        let command = command_buffer(&self.queue)?;
        let encoder = compute_encoder(&command)?;
        for (i, buffer) in self.buffers.iter().enumerate() {
            // SAFETY: the baseline source declares these five buffers at
            // slots 0..5; capacities were checked at allocation; offset is 0.
            unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), 0, i) };
        }
        let lanes = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.setComputePipelineState(&self.grid);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups,
                height: 1,
                depth: 1,
            },
            lanes,
        );
        encoder.setComputePipelineState(&self.top);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            lanes,
        );
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::Device(format!(
                "Metal baseline command failed: {:?}",
                command.error()
            )));
        }
        let mut out = [0i32; 2];
        // SAFETY: the command completed; the output buffer holds two words.
        unsafe {
            ptr::copy_nonoverlapping(
                self.buffers[4].contents().as_ptr().cast::<i32>(),
                out.as_mut_ptr(),
                2,
            )
        };
        Ok((out, command.GPUEndTime() - command.GPUStartTime()))
    }
}
