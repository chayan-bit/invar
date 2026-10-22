//! Owned multi-input map submission; no storage is released on unknown completion.

use super::*;

struct FusedWork {
    input_host: Vec<PinnedHostSlice<u32>>,
    output_host: Vec<PinnedHostSlice<u32>>,
    input_dev: Vec<CudaSlice<u32>>,
    output_dev: Vec<CudaSlice<u32>>,
    function: CudaFunction,
    h2d_start: CudaEvent,
    h2d_end: CudaEvent,
    kernel_end: CudaEvent,
    completed: CudaEvent,
    stream: Arc<CudaStream>,
    length: u32,
}

/// Internal qualification injection, never exposed as a CLI option.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusedFault {
    None,
    FalseProfitability,
    TransferOutOfMemory,
    LaunchFailure,
    PartialCompletion,
}

#[derive(Debug)]
pub struct FusedResult {
    pub outputs: Vec<Vec<u32>>,
    pub source_hash: u64,
    pub cubin_bytes: usize,
    pub transfer_bytes: usize,
    pub blocks: u32,
    pub lanes: u32,
    pub compile_ms: f64,
    pub host_to_host_ms: f64,
    pub h2d_ms: f32,
    pub kernel_ms: f32,
    pub d2h_ms: f32,
    pub nvrtc_log: String,
    pub cache_hit: bool,
}

impl CudaDevice {
    /// Execute one verified graph pass with every source-stage result retained.
    pub fn run_fused_map(
        &mut self,
        plan: &Verified,
        inputs: &[&[u32]],
        compiler: &Path,
    ) -> Result<FusedResult, Error> {
        self.run_fused_map_with_fault(plan, inputs, compiler, FusedFault::None)
    }

    #[doc(hidden)]
    pub fn run_fused_map_with_fault(
        &mut self,
        plan: &Verified,
        inputs: &[&[u32]],
        compiler: &Path,
        fault: FusedFault,
    ) -> Result<FusedResult, Error> {
        self.admit()?;
        let dag = plan.ir().dag.as_ref().ok_or(Error::Limit)?;
        let len = usize::try_from(plan.ir().length).map_err(|_| Error::Limit)?;
        let bytes = len
            .checked_mul(inputs.len() + dag.outputs.len())
            .and_then(|n| n.checked_mul(std::mem::size_of::<u32>()))
            .ok_or(Error::Limit)?;
        if plan.capabilities() != self.caps
            || inputs.len() != dag.inputs.len()
            || inputs.iter().any(|input| input.len() != len)
        {
            return Err(Error::Limit);
        }
        if bytes as u64 > invar_device::MAX_DAG_TRANSFER_BYTES {
            return Err(Error::ResourceLimit("map exceeds 8 GiB transfer budget"));
        }
        // These two failures occur before any enqueue; a caller may choose a
        // fresh CPU result without touching or reusing device output storage.
        match fault {
            FusedFault::FalseProfitability => return Err(Error::Unprofitable),
            FusedFault::TransferOutOfMemory => return Err(Error::OutOfMemory),
            FusedFault::LaunchFailure => return Err(Error::InvalidHandle),
            _ => {}
        }
        let source = emit_cuda(plan).map_err(|e| Error::Compile(format!("{e:?}")))?;
        let hash = source_hash(&source);
        let geometry = plan.ir().geometry;
        if len == 0 {
            return Ok(FusedResult {
                outputs: vec![Vec::new(); dag.outputs.len()],
                source_hash: hash,
                cubin_bytes: 0,
                transfer_bytes: 0,
                blocks: geometry.blocks,
                lanes: geometry.lanes,
                compile_ms: 0.0,
                host_to_host_ms: 0.0,
                h2d_ms: 0.0,
                kernel_ms: 0.0,
                d2h_ms: 0.0,
                nvrtc_log: String::new(),
                cache_hit: false,
            });
        }
        self.check_submission(bytes)?;
        let mut loaded = self.load_functions(plan, &source, &["invar_map_v1"], compiler)?;
        let transfer_started = Instant::now();
        let function = loaded.functions.remove(0);
        let mut input_host = Vec::new();
        let mut output_host = Vec::new();
        let mut input_dev = Vec::new();
        let mut output_dev = Vec::new();
        for input in inputs {
            // SAFETY: every slot is initialized by the disjoint host copy before
            // cudarc can read this pinned allocation for H2D transfer.
            let mut host =
                unsafe { self.context.alloc_pinned::<u32>(len) }.map_err(allocation_error)?;
            let ptr = host.as_mut_ptr().map_err(driver_error)?;
            // SAFETY: `ptr` owns `len` writable u32 elements; `input` has that
            // exact length and remains live for this synchronous copy.
            unsafe { std::ptr::copy_nonoverlapping(input.as_ptr(), ptr, len) };
            input_host.push(host);
            // SAFETY: the H2D copy on this stream initializes every device slot
            // before the kernel can read it; FusedWork retains the allocation.
            input_dev.push(unsafe { self.stream.alloc::<u32>(len) }.map_err(allocation_error)?);
        }
        for _ in &dag.outputs {
            // SAFETY: the zeroed pinned storage is fully initialized before the
            // D2H mutable-slice API sees it; the event gates every host read.
            let mut host =
                unsafe { self.context.alloc_pinned::<u32>(len) }.map_err(allocation_error)?;
            let ptr = host.as_mut_ptr().map_err(driver_error)?;
            // SAFETY: `ptr` owns `len` writable u32 elements of this allocation.
            unsafe { std::ptr::write_bytes(ptr, 0, len) };
            output_host.push(host);
            // SAFETY: the kernel writes every masked output slot before D2H,
            // and FusedWork retains it through the completion event.
            output_dev.push(unsafe { self.stream.alloc::<u32>(len) }.map_err(allocation_error)?);
        }
        let event = || {
            self.context
                .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(driver_error)
        };
        let mut work = Box::new(FusedWork {
            input_host,
            output_host,
            input_dev,
            output_dev,
            function,
            h2d_start: event()?,
            h2d_end: event()?,
            kernel_end: event()?,
            completed: event()?,
            stream: self.stream.clone(),
            length: plan.ir().length,
        });
        let submitted = (|| -> Result<(), Error> {
            work.h2d_start.record(&work.stream).map_err(driver_error)?;
            for (host, device) in work.input_host.iter().zip(&mut work.input_dev) {
                work.stream
                    .memcpy_htod(host, device)
                    .map_err(driver_error)?;
            }
            work.h2d_end.record(&work.stream).map_err(driver_error)?;
            let mut launch = work.stream.launch_builder(&work.function);
            for input in &work.input_dev {
                launch.arg(input);
            }
            for output in &mut work.output_dev {
                launch.arg(output);
            }
            launch.arg(&work.length);
            // SAFETY: Verified proves the generated argument order, disjoint
            // contiguous buffers, masked bounds, and legal grid-stride geometry.
            // FusedWork owns all arguments, module and stream until completion.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (geometry.blocks, 1, 1),
                    block_dim: (geometry.lanes, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(driver_error)?;
            work.kernel_end.record(&work.stream).map_err(driver_error)?;
            if fault == FusedFault::PartialCompletion {
                return Err(Error::Timeout);
            }
            for (device, host) in work.output_dev.iter().zip(&mut work.output_host) {
                work.stream
                    .memcpy_dtoh(device, host)
                    .map_err(driver_error)?;
            }
            work.completed.record(&work.stream).map_err(driver_error)?;
            Ok(())
        })();
        if let Err(error) = submitted {
            // Any failed enqueue may already be in flight. Neither pinned/device
            // buffers nor the compiled module may be freed until quiescence.
            self.hold(work);
            return Err(error);
        }
        let started = Instant::now();
        while !work.completed.is_complete() {
            if started.elapsed() >= KERNEL_DEADLINE {
                self.hold(work);
                return Err(Error::Timeout);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        let h2d_ms = work
            .h2d_start
            .elapsed_ms(&work.h2d_end)
            .map_err(driver_error)?;
        let kernel_ms = work
            .h2d_end
            .elapsed_ms(&work.kernel_end)
            .map_err(driver_error)?;
        let d2h_ms = work
            .kernel_end
            .elapsed_ms(&work.completed)
            .map_err(driver_error)?;
        let outputs = work
            .output_host
            .iter()
            .map(|host| {
                let slice = host.as_slice().map_err(driver_error)?;
                let mut values = Vec::new();
                values
                    .try_reserve_exact(slice.len())
                    .map_err(|_| Error::ResourceLimit("host output allocation failed"))?;
                values.extend_from_slice(slice);
                Ok(values)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let host_to_host_ms = transfer_started.elapsed().as_secs_f64() * 1000.0;
        Ok(FusedResult {
            outputs,
            source_hash: hash,
            cubin_bytes: loaded.cubin_bytes,
            transfer_bytes: bytes,
            blocks: geometry.blocks,
            lanes: geometry.lanes,
            compile_ms: loaded.compile_ms,
            host_to_host_ms,
            h2d_ms,
            kernel_ms,
            d2h_ms,
            nvrtc_log: loaded.nvrtc_log,
            cache_hit: loaded.cache_hit,
        })
    }
}
