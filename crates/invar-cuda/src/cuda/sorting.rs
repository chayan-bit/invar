//! Stable radix; ponytail: compile per call, cache for repeated small sorts.
//! until a terminal fence; partial/failed work never publishes host arrays.
use super::*;
use invar_device::sorting as sort;

#[derive(Debug)]
pub struct SortResult {
    pub keys: Vec<u32>,
    pub payload: Vec<u32>,
    pub source_hash: u64,
    pub compile_ms: f64,
    pub kernel_ms: f64,
    pub transfer_bytes: usize,
}
struct SortWork {
    buffers: Vec<CudaSlice<u32>>,
    keys: PinnedHostSlice<u32>,
    payload: PinnedHostSlice<u32>,
    _functions: Vec<CudaFunction>,
    start: CudaEvent,
    end: CudaEvent,
    event: CudaEvent,
}
impl CudaDevice {
    pub fn run_sort(
        &mut self,
        plan: &sort::Plan,
        keys: &[u32],
        payload: &[u32],
        compiler: &Path,
    ) -> Result<SortResult, Error> {
        self.run_sort_with_fault(plan, keys, payload, compiler, CollectiveFault::None)
    }
    pub fn run_sort_with_fault(
        &mut self,
        plan: &sort::Plan,
        keys: &[u32],
        payload: &[u32],
        compiler: &Path,
        fault: CollectiveFault,
    ) -> Result<SortResult, Error> {
        plan.validate_inputs(keys, payload)
            .map_err(|_| Error::Limit)?;
        self.admit()?;
        match fault {
            CollectiveFault::OutOfMemory => return Err(Error::OutOfMemory),
            CollectiveFault::LaunchFailure => {
                return Err(Error::Driver("injected sort launch rejection".into()));
            }
            _ => {}
        }
        if keys.is_empty() {
            return Ok(SortResult {
                keys: Vec::new(),
                payload: Vec::new(),
                source_hash: source_hash(sort::CUDA),
                compile_ms: 0.0,
                kernel_ms: 0.0,
                transfer_bytes: 0,
            });
        }
        let words = plan.buffer_words();
        let storage = words
            .iter()
            .try_fold(0usize, |sum, &n| sum.checked_add(n.max(1).checked_mul(4)?))
            .ok_or(Error::Limit)?;
        crate::validate_transfer_budget(storage, storage)?;
        self.check_submission(storage)?;
        let started = Instant::now();
        let compiled = compile_cubin_bounded(sort::CUDA, compiler, self.caps)?;
        let module = self
            .context
            .load_module(Ptx::from_binary(compiled.image))
            .map_err(driver_error)?;
        let functions = sort::KERNELS
            .iter()
            .map(|name| module.load_function(name).map_err(driver_error))
            .collect::<Result<Vec<_>, _>>()?;
        let compile_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut buffers = Vec::with_capacity(16);
        for (slot, &length) in words.iter().enumerate() {
            buffers.push(match slot {
                0 => self.stream.clone_htod(keys).map_err(driver_error)?,
                1 => self.stream.clone_htod(payload).map_err(driver_error)?,
                8..=15 => self
                    .stream
                    .clone_htod(&plan.parameters(slot - 8))
                    .map_err(driver_error)?,
                _ => self
                    .stream
                    .alloc_zeros::<u32>(length.max(1))
                    .map_err(driver_error)?,
            });
        }
        fn pinned(
            context: &Arc<CudaContext>,
            length: usize,
        ) -> Result<PinnedHostSlice<u32>, Error> {
            // SAFETY: initialize every pinned word before the safe D2H API can
            // borrow the buffer. SortWork retains it through the terminal fence.
            let mut host = unsafe { context.alloc_pinned::<u32>(length) }.map_err(driver_error)?;
            let target = host.as_mut_ptr().map_err(driver_error)?;
            // SAFETY: target owns exactly length writable, disjoint u32 slots.
            unsafe { std::ptr::write_bytes(target, 0, length) };
            Ok(host)
        }
        let timed = || {
            self.context
                .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(driver_error)
        };
        let mut work = Box::new(SortWork {
            buffers,
            keys: pinned(&self.context, keys.len())?,
            payload: pinned(&self.context, payload.len())?,
            _functions: functions.clone(),
            start: timed()?,
            end: timed()?,
            event: timed()?,
        });
        let stream = self.stream.clone();
        let submitted = (|| -> Result<(), Error> {
            work.start.record(&stream).map_err(driver_error)?;
            for dispatch in plan.launches() {
                let mut launch = stream.launch_builder(&functions[dispatch.kernel]);
                for slot in dispatch.buffers {
                    launch.arg(&work.buffers[slot]);
                }
                // SAFETY: Plan fixes pointer order, lengths and 256-lane
                // geometry for these immutable kernels. Histograms/scans precede
                // scatter on one stream; stable ranks yield disjoint writes.
                // Every argument is retained by SortWork through completion.
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (dispatch.blocks, 1, 1),
                        block_dim: (sort::LANES, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(driver_error)?;
            }
            work.end.record(&stream).map_err(driver_error)?;
            let SortWork {
                buffers,
                keys,
                payload,
                ..
            } = &mut *work;
            stream
                .memcpy_dtoh(&buffers[0], keys)
                .map_err(driver_error)?;
            stream
                .memcpy_dtoh(&buffers[1], payload)
                .map_err(driver_error)?;
            work.event.record(&stream).map_err(driver_error)?;
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
        while !work.event.is_complete() {
            if waited.elapsed() >= KERNEL_DEADLINE {
                self.hold(work);
                return Err(Error::Timeout);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        let kernel_ms = f64::from(work.start.elapsed_ms(&work.end).map_err(driver_error)?);
        fn copy(host: &PinnedHostSlice<u32>) -> Result<Vec<u32>, Error> {
            let source = host.as_slice().map_err(driver_error)?;
            let mut out = Vec::new();
            out.try_reserve_exact(source.len())
                .map_err(|_| Error::OutOfMemory)?;
            out.extend_from_slice(source);
            Ok(out)
        }
        Ok(SortResult {
            keys: copy(&work.keys)?,
            payload: copy(&work.payload)?,
            source_hash: source_hash(sort::CUDA),
            compile_ms,
            kernel_ms,
            transfer_bytes: keys.len().checked_mul(16).ok_or(Error::Limit)?,
        })
    }
}
