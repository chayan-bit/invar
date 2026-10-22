//! One owned-stream dispatch for the bounded spatial raw ABI. Failed/incomplete
//! submissions retain all pinned/device storage until a terminal fence.
use super::*;
use invar_device::spatial;

#[derive(Debug)]
pub struct SpatialResult {
    pub counts: Vec<u32>,
    pub indices: Vec<u32>,
    pub distances: Vec<u32>,
    pub source_hash: u64,
    pub compile_ms: f64,
    pub kernel_ms: f64,
    pub transfer_bytes: usize,
}
struct SpatialWork {
    buffers: Vec<CudaSlice<u32>>,
    host: Vec<PinnedHostSlice<u32>>,
    _function: CudaFunction,
    start: CudaEvent,
    end: CudaEvent,
    fence: CudaEvent,
}
impl CudaDevice {
    pub fn run_spatial(
        &mut self,
        plan: &spatial::Plan,
        inputs: [&[u32]; 4],
        compiler: &Path,
    ) -> Result<SpatialResult, Error> {
        self.run_spatial_with_fault(plan, inputs, compiler, CollectiveFault::None)
    }
    pub fn run_spatial_with_fault(
        &mut self,
        plan: &spatial::Plan,
        inputs: [&[u32]; 4],
        compiler: &Path,
        fault: CollectiveFault,
    ) -> Result<SpatialResult, Error> {
        self.admit()?;
        if fault == CollectiveFault::OutOfMemory {
            return Err(Error::OutOfMemory);
        }
        if fault == CollectiveFault::LaunchFailure {
            return Err(Error::Driver("injected spatial launch rejection".into()));
        }
        let initial = plan.initial(inputs).map_err(|_| Error::Limit)?;
        let bytes = initial
            .iter()
            .try_fold(0usize, |sum, v| sum.checked_add(v.len().checked_mul(4)?))
            .ok_or(Error::Limit)?;
        crate::validate_transfer_budget(bytes, bytes.checked_mul(2).ok_or(Error::Limit)?)?;
        self.check_submission(bytes)?;
        let started = Instant::now();
        let compiled = compile_cubin_bounded(spatial::CUDA, compiler, self.caps)?;
        let module = self
            .context
            .load_module(Ptx::from_binary(compiled.image))
            .map_err(driver_error)?;
        let function = module
            .load_function(spatial::KERNELS[plan.kernel()])
            .map_err(driver_error)?;
        let compile_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut buffers = Vec::new();
        let mut host = Vec::new();
        for values in &initial {
            // SAFETY: initialize the complete pinned allocation before H2D;
            // SpatialWork retains it and every device buffer through the fence.
            let mut h =
                unsafe { self.context.alloc_pinned::<u32>(values.len()) }.map_err(driver_error)?;
            let target = h.as_mut_ptr().map_err(driver_error)?;
            // SAFETY: the new pinned allocation has exactly values.len() words.
            unsafe {
                std::ptr::copy_nonoverlapping(values.as_ptr(), target, values.len());
            }
            host.push(h);
            // SAFETY: stream-ordered H2D initializes all slots before launch.
            buffers.push(unsafe { self.stream.alloc::<u32>(values.len()) }.map_err(driver_error)?);
        }
        let timed = || {
            self.context
                .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(driver_error)
        };
        let mut work = Box::new(SpatialWork {
            buffers,
            host,
            _function: function.clone(),
            start: timed()?,
            end: timed()?,
            fence: timed()?,
        });
        let submitted = (|| -> Result<(), Error> {
            for (h, d) in work.host.iter().zip(&mut work.buffers) {
                self.stream.memcpy_htod(h, d).map_err(driver_error)?;
            }
            work.start.record(&self.stream).map_err(driver_error)?;
            if plan.work() != 0 {
                let mut launch = self.stream.launch_builder(&function);
                for buffer in &work.buffers {
                    launch.arg(buffer);
                }
                // SAFETY: Plan checks lengths, disjoint output slots and all
                // indirect range reads. Kernels guard their global row index;
                // the complete eight-pointer ABI is retained until completion.
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (plan.work().div_ceil(spatial::LANES), 1, 1),
                        block_dim: (spatial::LANES, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(driver_error)?;
            }
            work.end.record(&self.stream).map_err(driver_error)?;
            for i in 4..7 {
                self.stream
                    .memcpy_dtoh(&work.buffers[i], &mut work.host[i])
                    .map_err(driver_error)?;
            }
            work.fence.record(&self.stream).map_err(driver_error)?;
            Ok(())
        })();
        if let Err(error) = submitted {
            self.hold(work);
            return Err(error);
        }
        if fault == CollectiveFault::PartialCompletion {
            self.hold(work);
            return Err(Error::Timeout);
        }
        let waited = Instant::now();
        while !work.fence.is_complete() {
            if waited.elapsed() >= KERNEL_DEADLINE {
                self.hold(work);
                return Err(Error::Timeout);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        let mut output = Vec::new();
        for i in 4..7 {
            let src = work.host[i].as_slice().map_err(driver_error)?;
            let mut v = Vec::new();
            v.try_reserve_exact(plan.lengths()[i])
                .map_err(|_| Error::OutOfMemory)?;
            v.extend_from_slice(&src[..plan.lengths()[i]]);
            output.push(v);
        }
        let mut output = output.into_iter();
        Ok(SpatialResult {
            counts: output.next().ok_or(Error::Limit)?,
            indices: output.next().ok_or(Error::Limit)?,
            distances: output.next().ok_or(Error::Limit)?,
            source_hash: source_hash(spatial::CUDA),
            compile_ms,
            kernel_ms: f64::from(work.start.elapsed_ms(&work.end).map_err(driver_error)?),
            transfer_bytes: bytes
                .checked_add(
                    plan.lengths()[4..7]
                        .iter()
                        .sum::<usize>()
                        .checked_mul(4)
                        .ok_or(Error::Limit)?,
                )
                .ok_or(Error::Limit)?,
        })
    }
}
