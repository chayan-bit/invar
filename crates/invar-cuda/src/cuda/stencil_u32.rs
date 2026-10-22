//! Owned multi-phase stencil. Each launch reads a completed stream time slice.
use super::transpose::{PendingWork, compile_bounded, finish, pinned};
use super::*;
use cudarc::driver::{CudaModule, DevicePtr};
use invar_device::stencil_u32 as ir;

/// Internal boundary fault injection, never selected by source or CLI inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StencilFault {
    None,
    OutOfMemory,
    LaunchFailure,
    PartialCompletion,
}

pub struct StencilResult {
    pub target: &'static str,
    pub output: Vec<u32>,
    pub source_hash: u64,
    pub transfer_bytes: usize,
    pub compile_ms: f64,
    pub host_to_host_ms: f64,
    pub phases: u32,
    pub final_buffer: usize,
}

fn admit_storage(
    bytes: usize,
    capacity: impl FnOnce(usize) -> Result<(), Error>,
) -> Result<usize, Error> {
    let owned_bytes = bytes.checked_mul(5).ok_or(Error::Limit)?;
    if owned_bytes as u64 > invar_device::transpose::MAX_OWNED_BYTES {
        return Err(Error::ResourceLimit("M4-02 owned storage exceeds 5 GiB"));
    }
    let transfer_bytes = bytes.checked_mul(3).ok_or(Error::Limit)?;
    crate::validate_transfer_budget(transfer_bytes, owned_bytes)?;
    capacity(bytes.checked_mul(2).ok_or(Error::Limit)?)?;
    Ok(transfer_bytes)
}

struct Work {
    context: Arc<CudaContext>,
    _module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    function: CudaFunction,
    input: PinnedHostSlice<u32>,
    output: PinnedHostSlice<u32>,
    buffers: [CudaSlice<u32>; 2],
    completed: CudaEvent,
    steps: u32,
    blocks: u32,
}
impl PendingWork for Work {
    fn submit(&mut self) -> Result<(), Error> {
        self.context.check_err().map_err(driver_error)?;
        self.stream
            .memcpy_htod(&self.input, &mut self.buffers[0])
            .map_err(driver_error)?;
        self.stream
            .memcpy_htod(&self.output, &mut self.buffers[1])
            .map_err(driver_error)?;
        for step in 0..self.steps {
            let [a, b] = &mut self.buffers;
            let (input, output) = if step.is_multiple_of(2) {
                (a, b)
            } else {
                (b, a)
            };
            let mut launch = self.stream.launch_builder(&self.function);
            launch.arg(&*input).arg(output);
            // SAFETY: the sealed IR fixes two u32 pointers and 16x16 geometry.
            // Both full capacities are initialized and disjoint. Same-stream
            // phases order all preceding writes before the next slice reads.
            // Every owner is held until the event after the final D2H completes.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (self.blocks, 1, 1),
                    block_dim: (16, 16, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(driver_error)?;
        }
        self.stream
            .memcpy_dtoh(&self.buffers[(self.steps % 2) as usize], &mut self.output)
            .map_err(driver_error)?;
        self.completed.record(&self.stream).map_err(driver_error)
    }
    fn complete(&self) -> Result<bool, Error> {
        self.context.bind_to_thread().map_err(driver_error)?;
        self.context.check_err().map_err(driver_error)?;
        // SAFETY: the live retained event follows all launches and D2H on the
        // retained stream; querying does not release work or publish output.
        match unsafe { driver_sys::cuEventQuery(self.completed.cu_event()) }.result() {
            Ok(()) => Ok(true),
            Err(DriverError(CUresult::CUDA_ERROR_NOT_READY)) => Ok(false),
            Err(e) => Err(driver_error(e)),
        }
    }
    fn read(&self) -> Result<Vec<u32>, Error> {
        let values = self.output.as_slice().map_err(driver_error)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(values.len())
            .map_err(|_| Error::OutOfMemory)?;
        result.extend_from_slice(values);
        Ok(result)
    }
}
impl CudaDevice {
    pub fn run_stencil_u32(
        &mut self,
        plan: &ir::Verified,
        input: &[u32],
        compiler: &Path,
    ) -> Result<StencilResult, Error> {
        self.run_stencil_u32_with_fault(plan, input, compiler, StencilFault::None)
    }
    pub fn run_stencil_u32_with_fault(
        &mut self,
        plan: &ir::Verified,
        input: &[u32],
        compiler: &Path,
        fault: StencilFault,
    ) -> Result<StencilResult, Error> {
        self.admit()?;
        match fault {
            StencilFault::OutOfMemory => return Err(Error::OutOfMemory),
            StencilFault::LaunchFailure => {
                return Err(Error::Driver("injected pre-submission rejection".into()));
            }
            _ => {}
        }
        let p = plan.ir();
        let count = usize::try_from(p.input.elements).map_err(|_| Error::Limit)?;
        let logical = u64::from(p.dimensions[0]) * u64::from(p.dimensions[1]);
        let bytes = count.checked_mul(4).ok_or(Error::Limit)?;
        if plan.capabilities() != self.caps || count != input.len() {
            return Err(Error::Limit);
        }
        let transfer_bytes =
            admit_storage(bytes, |device_bytes| self.check_submission(device_bytes))?;
        let source = ir::emit_cuda(plan).map_err(|e| Error::Compile(format!("{e:?}")))?;
        let hash = source_hash(&source);
        if logical == 0 || p.steps == 0 {
            let mut output = Vec::new();
            output
                .try_reserve_exact(count)
                .map_err(|_| Error::OutOfMemory)?;
            output.extend_from_slice(input);
            return Ok(StencilResult {
                target: "cuda",
                output,
                source_hash: hash,
                transfer_bytes: 0,
                compile_ms: 0.0,
                host_to_host_ms: 0.0,
                phases: 0,
                final_buffer: 0,
            });
        }
        let blocks = u32::try_from(
            u64::from(p.dimensions[0]).div_ceil(16) * u64::from(p.dimensions[1]).div_ceil(16),
        )
        .map_err(|_| Error::Limit)?
        .clamp(1, 65535);
        plan.validate_launch([16, 16, 1], [blocks, 1, 1])
            .map_err(|_| Error::Limit)?;
        let begun = Instant::now();
        let compiled = compile_bounded(&source, compiler, self.caps, COMPILE_DEADLINE)?;
        let compile_ms = begun.elapsed().as_secs_f64() * 1000.0;
        let module = self
            .context
            .load_module(Ptx::from_binary(compiled.image))
            .map_err(driver_error)?;
        let function = module
            .load_function(ir::KERNEL_NAME)
            .map_err(driver_error)?;
        let begun = Instant::now();
        let input_host = pinned(&self.context, Some(input), count)?;
        let output = pinned(&self.context, None, count)?;
        // SAFETY: private checked capacities, fully initialized by submit before
        // any device access, and held through every phase and final completion.
        let buffers = unsafe {
            [
                self.stream.alloc::<u32>(count).map_err(driver_error)?,
                self.stream.alloc::<u32>(count).map_err(driver_error)?,
            ]
        };
        for (buffer, metadata) in buffers.iter().zip([&p.input, &p.output]) {
            let (ptr, _lease) = buffer.device_ptr(&self.stream);
            if ptr == 0 || !ptr.is_multiple_of(metadata.alignment) {
                return Err(Error::Limit);
            }
        }
        let work = Box::new(Work {
            context: self.context.clone(),
            _module: module,
            stream: self.stream.clone(),
            function,
            input: input_host,
            output,
            buffers,
            completed: self.context.new_event(None).map_err(driver_error)?,
            steps: p.steps,
            blocks,
        });
        let output = finish(
            work,
            |work| self.hold(work),
            KERNEL_DEADLINE,
            || (fault == StencilFault::PartialCompletion).then_some(Error::Timeout),
        )?;
        let start = usize::try_from(p.output.offset).map_err(|_| Error::Limit)?;
        let end = start
            .checked_add(usize::try_from(logical).map_err(|_| Error::Limit)?)
            .ok_or(Error::Limit)?;
        // Initial slice padding survives even phases; fresh padding stays zero
        // on odd phases. No logical or padded overread is admitted by the IR.
        for (i, &word) in output.iter().enumerate() {
            if (i < start || i >= end)
                && word
                    != if p.steps.is_multiple_of(2) {
                        input[i]
                    } else {
                        0
                    }
            {
                return Err(Error::Driver("stencil padding modified".into()));
            }
        }
        Ok(StencilResult {
            target: "cuda",
            output,
            source_hash: hash,
            transfer_bytes,
            compile_ms,
            host_to_host_ms: begun.elapsed().as_secs_f64() * 1000.0,
            phases: p.steps,
            final_buffer: plan.final_buffer(),
        })
    }
}

#[cfg(test)]
mod capacity_tests {
    use super::*;

    #[test]
    fn stencil_shared_budget_and_live_capacity_precede_storage() {
        const GIB: usize = 1024 * 1024 * 1024;
        assert_eq!(
            admit_storage(GIB, |device_bytes| crate::validate_device_capacity(
                device_bytes,
                3 * GIB
            ))
            .unwrap(),
            3 * GIB
        );
        assert!(matches!(
            admit_storage(GIB, |device_bytes| crate::validate_device_capacity(
                device_bytes,
                2 * GIB
            )),
            Err(Error::ResourceLimit(_))
        ));
        let mut queried = false;
        assert!(matches!(
            admit_storage(GIB + 1, |_| {
                queried = true;
                Ok(())
            }),
            Err(Error::ResourceLimit("M4-02 owned storage exceeds 5 GiB"))
        ));
        assert!(!queried);
    }
}
