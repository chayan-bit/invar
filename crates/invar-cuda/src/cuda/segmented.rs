//! M4-04 segmented map, segmented reduction and stable compaction on the
//! owned stream. Inputs are admitted by `segmented::admit_inputs` before any
//! upload; outputs are read only after the stream event proves completion.
//! Work that misses its deadline or fails to enqueue is held, never freed.

use super::*;
use invar_device::segmented as seg;

#[derive(Debug)]
pub struct SegmentedResult {
    /// Map: one word per element. Reduce: one per segment. Compact: exactly
    /// the retained words, in source order.
    pub output: Vec<u32>,
    pub source_hash: u64,
    pub cubin_bytes: usize,
    pub compile_ms: f64,
    /// Device time from after uploads to the end of the last kernel.
    pub kernel_ms: f32,
    pub transfer_bytes: usize,
    pub nvrtc_log: String,
}

fn admit_storage(
    words: &[usize],
    capacity: impl FnOnce(usize) -> Result<(), Error>,
) -> Result<usize, Error> {
    let transfer_bytes = words
        .iter()
        .try_fold(0usize, |total, n| {
            total.checked_add(n.max(&1).checked_mul(4)?)
        })
        .ok_or(Error::Limit)?;
    crate::validate_transfer_budget(transfer_bytes, transfer_bytes)?;
    capacity(transfer_bytes)?;
    Ok(transfer_bytes)
}

/// Everything submitted work touches; dropped only after its event completes.
struct SegmentedWork {
    buffers: Vec<CudaSlice<u32>>,
    output_host: PinnedHostSlice<u32>,
    total_host: PinnedHostSlice<u32>,
    _functions: Vec<CudaFunction>,
    kernels_start: CudaEvent,
    kernels_end: CudaEvent,
    event: CudaEvent,
}

impl CudaDevice {
    pub fn run_segmented(
        &mut self,
        plan: &seg::Verified,
        input: &[u32],
        offsets: &[u32],
        compiler: &Path,
    ) -> Result<SegmentedResult, Error> {
        self.run_segmented_with_fault(plan, input, offsets, compiler, CollectiveFault::None)
    }

    pub fn run_segmented_with_fault(
        &mut self,
        plan: &seg::Verified,
        input: &[u32],
        offsets: &[u32],
        compiler: &Path,
        fault: CollectiveFault,
    ) -> Result<SegmentedResult, Error> {
        self.admit()?;
        match fault {
            CollectiveFault::OutOfMemory => return Err(Error::OutOfMemory),
            CollectiveFault::LaunchFailure => {
                return Err(Error::Driver("injected segmented launch rejection".into()));
            }
            _ => {}
        }
        if plan.capabilities() != self.caps {
            return Err(Error::Limit);
        }
        let launches = plan.launches(input, offsets).map_err(|_| Error::Limit)?;
        let words = plan.buffer_words();
        let transfer_bytes = admit_storage(&words, |bytes| self.check_submission(bytes))?;
        use cudarc::driver::sys::CUdevice_attribute as A;
        for (attribute, required) in [
            (
                A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                launches.iter().map(|l| l.blocks).max().unwrap_or(1),
            ),
            (A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, seg::LANES),
            (
                A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                seg::LANES * 8,
            ),
        ] {
            let available = self.context.attribute(attribute).map_err(driver_error)?;
            if i64::from(available) < i64::from(required) {
                return Err(Error::ResourceLimit(
                    "segmented geometry exceeds CUDA device limits",
                ));
            }
        }
        let source = seg::emit_cuda(plan).map_err(|e| Error::Compile(format!("{e:?}")))?;
        let started = Instant::now();
        let compiled = compile_cubin_bounded(&source, compiler, self.caps)?;
        let compile_ms = started.elapsed().as_secs_f64() * 1000.0;
        let cubin_bytes = compiled.image.len();
        let module = self
            .context
            .load_module(Ptx::from_binary(compiled.image))
            .map_err(driver_error)?;
        let functions = launches
            .iter()
            .map(|launch| {
                module
                    .load_function(plan.kernels()[launch.kernel])
                    .map_err(driver_error)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let compact = matches!(plan.ir().family, seg::Family::Compact { .. });
        // Uploaded inputs, then zeroed outputs/scratch; zero-length buffers
        // still get one word so every kernel argument is a valid pointer.
        let uploads: [&[u32]; 2] = [input, offsets];
        let mut buffers = Vec::with_capacity(words.len());
        for (slot, &n) in words.iter().enumerate() {
            let buffer = match uploads.get(slot) {
                Some(values) if !values.is_empty() => {
                    self.stream.clone_htod(*values).map_err(driver_error)?
                }
                _ => self
                    .stream
                    .alloc_zeros::<u32>(n.max(1))
                    .map_err(driver_error)?,
            };
            buffers.push(buffer);
        }
        let output_slot = if compact { 1 } else { 2 };
        let output_words = words[output_slot].max(1);
        // SAFETY: zero-filled below before the D2H API borrows it; the event
        // gates every host read and SegmentedWork retains the allocation.
        let mut output_host =
            unsafe { self.context.alloc_pinned::<u32>(output_words) }.map_err(driver_error)?;
        // SAFETY: as above, for the one-word compaction total.
        let mut total_host =
            unsafe { self.context.alloc_pinned::<u32>(1) }.map_err(driver_error)?;
        for host in [&mut output_host, &mut total_host] {
            let len = host.len();
            let ptr = host.as_mut_ptr().map_err(driver_error)?;
            // SAFETY: `ptr` owns `len` writable pinned u32 slots; the mutable
            // D2H slice API requires initialized host storage.
            unsafe { std::ptr::write_bytes(ptr, 0, len) };
        }
        let timed = || {
            self.context
                .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(driver_error)
        };
        let mut work = Box::new(SegmentedWork {
            buffers,
            output_host,
            total_host,
            _functions: functions.clone(),
            kernels_start: timed()?,
            kernels_end: timed()?,
            event: timed()?,
        });
        let stream = self.stream.clone();
        let submitted = (|| -> Result<(), Error> {
            work.kernels_start.record(&stream).map_err(driver_error)?;
            for (dispatch, function) in launches.iter().zip(&functions) {
                let mut launch = stream.launch_builder(function);
                for &slot in dispatch.buffers {
                    launch.arg(&work.buffers[slot]);
                }
                // SAFETY: `Verified` fixes each kernel's ABI (u32 pointers in
                // this order), the lane count and grid; `admit_inputs` proved
                // exact lengths and complete nondecreasing offsets, so every
                // access is in bounds. SegmentedWork retains all arguments.
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (dispatch.blocks, 1, 1),
                        block_dim: (seg::LANES, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(driver_error)?;
            }
            work.kernels_end.record(&stream).map_err(driver_error)?;
            let SegmentedWork {
                buffers,
                output_host,
                total_host,
                ..
            } = &mut *work;
            stream
                .memcpy_dtoh(&buffers[output_slot], output_host)
                .map_err(driver_error)?;
            if compact {
                stream
                    .memcpy_dtoh(&buffers[4], total_host)
                    .map_err(driver_error)?;
            }
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
        let kernel_ms = work
            .kernels_start
            .elapsed_ms(&work.kernels_end)
            .map_err(driver_error)?;
        let values = work.output_host.as_slice().map_err(driver_error)?;
        let published = if compact {
            let total = work.total_host.as_slice().map_err(driver_error)?[0] as usize;
            values
                .get(..total)
                .ok_or_else(|| Error::Driver("compaction total exceeds its input".into()))?
        } else {
            &values[..words[output_slot]]
        };
        let mut output = Vec::new();
        output
            .try_reserve_exact(published.len())
            .map_err(|_| Error::OutOfMemory)?;
        output.extend_from_slice(published);
        Ok(SegmentedResult {
            output,
            source_hash: source_hash(&source),
            cubin_bytes,
            compile_ms,
            kernel_ms,
            transfer_bytes,
            nvrtc_log: compiled.log,
        })
    }
}

#[cfg(test)]
mod capacity_tests {
    use super::*;

    #[test]
    fn segmented_shared_budget_and_live_capacity_precede_storage() {
        const GIB: usize = 1024 * 1024 * 1024;
        let words = [GIB / 4, GIB / 4];
        assert_eq!(
            admit_storage(&words, |bytes| crate::validate_device_capacity(
                bytes,
                3 * GIB
            ))
            .unwrap(),
            2 * GIB
        );
        assert!(matches!(
            admit_storage(&words, |bytes| crate::validate_device_capacity(
                bytes,
                2 * GIB
            )),
            Err(Error::ResourceLimit(_))
        ));
        let mut queried = false;
        assert!(matches!(
            admit_storage(&[2 * GIB / 4, 7 * GIB / 4], |_| {
                queried = true;
                Ok(())
            }),
            Err(Error::ResourceLimit(_))
        ));
        assert!(!queried);
    }
}
