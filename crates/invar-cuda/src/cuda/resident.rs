//! Retained M3 map buffers, with the same quarantine boundary as owned runs.
use super::*;
use crate::MAX_TRANSFER_BYTES;

pub struct ResidentMap {
    plan: Verified,
    inputs: Vec<CudaSlice<u32>>,
    outputs: Vec<CudaSlice<u32>>,
    input_host: Vec<PinnedHostSlice<u32>>,
    output_host: Vec<PinnedHostSlice<u32>>,
    function: CudaFunction,
    stream: Arc<CudaStream>,
    start: CudaEvent,
    end: CudaEvent,
    completed: bool,
}

impl ResidentMap {
    pub fn plan(&self) -> &Verified {
        &self.plan
    }
    pub fn is_completed(&self) -> bool {
        self.completed
    }
}

fn wait(event: &CudaEvent) -> Result<(), Error> {
    let started = Instant::now();
    while !event.is_complete() {
        if started.elapsed() >= KERNEL_DEADLINE {
            return Err(Error::Timeout);
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    Ok(())
}

impl CudaDevice {
    /// Setup transfers complete once, outside resident dispatch. Every host and
    /// device allocation is retained if submission/completion is uncertain.
    pub fn prepare_resident_map(
        &mut self,
        plan: Verified,
        inputs: &[&[u32]],
        compiler: &Path,
    ) -> Result<ResidentMap, Error> {
        self.admit()?;
        let ir = plan.ir();
        let input_count = ir.dag.as_ref().map_or(1, |d| d.inputs.len());
        let output_count = ir.dag.as_ref().map_or(1, |d| d.outputs.len());
        let bytes = (ir.length as usize)
            .checked_mul(input_count + output_count)
            .and_then(|n| n.checked_mul(4))
            .ok_or(Error::Limit)?;
        if plan.capabilities() != self.caps
            || ir.collective.is_some()
            || ir.length == 0
            || inputs.len() != input_count
            || inputs.iter().any(|a| a.len() != ir.length as usize)
            || bytes > MAX_TRANSFER_BYTES
        {
            return Err(Error::Limit);
        }
        let source = emit_cuda(&plan).map_err(|e| Error::Compile(format!("{e:?}")))?;
        let mut loaded = self.load_functions(&plan, &source, &["invar_map_v1"], compiler)?;
        let len = ir.length as usize;
        let mut input_host = Vec::new();
        let mut input_dev = Vec::new();
        let mut output_host = Vec::new();
        let mut output_dev = Vec::new();
        for input in inputs {
            // SAFETY: initialized before enqueue, with exact-length disjoint
            // host data. The map retains pinned storage through its end event.
            let mut host =
                unsafe { self.context.alloc_pinned::<u32>(len) }.map_err(driver_error)?;
            let ptr = host.as_mut_ptr().map_err(driver_error)?;
            // SAFETY: ptr owns len aligned writable u32s; input has exactly len.
            unsafe {
                std::ptr::copy_nonoverlapping(input.as_ptr(), ptr, len);
            }
            input_host.push(host);
            // SAFETY: H2D initializes every input before any kernel read.
            input_dev.push(unsafe { self.stream.alloc::<u32>(len) }.map_err(driver_error)?);
        }
        for _ in 0..output_count {
            // SAFETY: every pinned word is initialized before the D2H slice API.
            let mut host =
                unsafe { self.context.alloc_pinned::<u32>(len) }.map_err(driver_error)?;
            let ptr = host.as_mut_ptr().map_err(driver_error)?;
            // SAFETY: ptr owns exactly len writable u32 slots.
            unsafe {
                std::ptr::write_bytes(ptr, 0, len);
            }
            output_host.push(host);
            // SAFETY: Verified guarantees full output coverage before D2H;
            // completed=false forbids reading an output before that launch.
            output_dev.push(unsafe { self.stream.alloc::<u32>(len) }.map_err(driver_error)?);
        }
        let start = self
            .context
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(driver_error)?;
        let end = self
            .context
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(driver_error)?;
        let function = loaded.functions.remove(0);
        let mut map = ResidentMap {
            plan,
            inputs: input_dev,
            outputs: output_dev,
            input_host,
            output_host,
            function,
            stream: self.stream.clone(),
            start,
            end,
            completed: false,
        };
        let submitted = (|| -> Result<(), Error> {
            for (host, device) in map.input_host.iter().zip(&mut map.inputs) {
                map.stream.memcpy_htod(host, device).map_err(driver_error)?;
            }
            map.end.record(&map.stream).map_err(driver_error)?;
            wait(&map.end)
        })();
        if let Err(error) = submitted {
            self.hold(Box::new(map));
            return Err(error);
        }
        Ok(map)
    }

    /// Consumes the handle on failure so unknown work cannot be read or reused.
    /// Success returns the same allocations and the device-event interval, or
    /// `None` when the driver cannot report one for completed work.
    pub fn run_resident_map(
        &mut self,
        mut map: ResidentMap,
    ) -> Result<(ResidentMap, Option<f64>), Error> {
        self.admit()?;
        if !Arc::ptr_eq(&self.stream, &map.stream) {
            return Err(Error::InvalidHandle);
        }
        map.completed = false;
        let submitted = (|| -> Result<(), Error> {
            map.start.record(&map.stream).map_err(driver_error)?;
            let mut launch = map.stream.launch_builder(&map.function);
            for input in &map.inputs {
                launch.arg(input);
            }
            for output in &mut map.outputs {
                launch.arg(output);
            }
            launch.arg(&map.plan.ir().length);
            let g = map.plan.ir().geometry;
            // SAFETY: Verified fixes the ABI, disjoint allocations and masked
            // ranges. The consumed map retains every argument and the function;
            // any enqueue/completion failure moves it into device quarantine.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (g.blocks, 1, 1),
                    block_dim: (g.lanes, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(driver_error)?;
            map.end.record(&map.stream).map_err(driver_error)?;
            wait(&map.end)
        })();
        if let Err(error) = submitted {
            self.hold(Box::new(map));
            return Err(error);
        }
        // The end event was waited above, so the work is complete whether or
        // not the driver can report the interval.
        map.completed = true;
        let ms = interval_ms(map.start.elapsed_ms(&map.end));
        Ok((map, ms))
    }

    /// The returned arrays are host-owned only after the bounded end event.
    /// Failure consumes and quarantines the handle, including pinned staging.
    pub fn read_resident_map(
        &mut self,
        map: ResidentMap,
    ) -> Result<(ResidentMap, Vec<Vec<u32>>), Error> {
        if !map.completed {
            return Err(Error::StaleBorrow);
        }
        self.read_resident(map, true)
    }

    pub fn read_resident_inputs(
        &mut self,
        map: ResidentMap,
    ) -> Result<(ResidentMap, Vec<Vec<u32>>), Error> {
        self.read_resident(map, false)
    }

    fn read_resident(
        &mut self,
        mut map: ResidentMap,
        outputs: bool,
    ) -> Result<(ResidentMap, Vec<Vec<u32>>), Error> {
        self.admit()?;
        if !Arc::ptr_eq(&self.stream, &map.stream) {
            return Err(Error::InvalidHandle);
        }
        let submitted = (|| -> Result<(), Error> {
            let (device, host) = if outputs {
                (&map.outputs, &mut map.output_host)
            } else {
                (&map.inputs, &mut map.input_host)
            };
            for (device, host) in device.iter().zip(host) {
                map.stream.memcpy_dtoh(device, host).map_err(driver_error)?;
            }
            map.end.record(&map.stream).map_err(driver_error)?;
            wait(&map.end)
        })();
        if let Err(error) = submitted {
            self.hold(Box::new(map));
            return Err(error);
        }
        let host = if outputs {
            &map.output_host
        } else {
            &map.input_host
        };
        let values = host
            .iter()
            .map(|a| a.as_slice().map(|a| a.to_vec()).map_err(driver_error))
            .collect::<Result<Vec<_>, _>>()?;
        Ok((map, values))
    }
}

/// Event interval in milliseconds, or `None` when the driver reports none.
fn interval_ms<E>(elapsed: Result<f32, E>) -> Option<f64> {
    elapsed
        .ok()
        .map(f64::from)
        .filter(|ms| ms.is_finite() && *ms >= 0.0)
}

#[cfg(test)]
mod tests {
    use super::interval_ms;

    #[test]
    fn an_event_interval_the_driver_cannot_report_is_none_not_an_error() {
        assert_eq!(interval_ms::<()>(Err(())), None);
        assert_eq!(interval_ms::<()>(Ok(f32::NAN)), None);
        assert_eq!(interval_ms::<()>(Ok(-1.0)), None);
        assert_eq!(interval_ms::<()>(Ok(0.0)), Some(0.0));
        assert_eq!(interval_ms::<()>(Ok(1.5)), Some(1.5));
    }
}
