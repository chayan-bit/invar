//! Stable radix. Pipelines come from the device's compiled-ahead library
//! (`invar build --target metal` adds the sort kernels) through its cache.
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
impl MetalDevice {
    pub fn run_sort(
        &mut self,
        plan: &sort::Plan,
        keys: &[u32],
        payload: &[u32],
    ) -> Result<SortResult, Error> {
        self.run_sort_with_fault(plan, keys, payload, Fault::None)
    }
    pub fn run_sort_with_fault(
        &mut self,
        plan: &sort::Plan,
        keys: &[u32],
        payload: &[u32],
        fault: Fault,
    ) -> Result<SortResult, Error> {
        plan.validate_inputs(keys, payload)
            .map_err(|_| Error::Limit)?;
        self.admit()?;
        match fault {
            Fault::OutOfMemory => return Err(Error::OutOfMemory),
            Fault::LaunchFailure => {
                return Err(Error::Device("injected sort launch rejection".into()));
            }
            _ => {}
        }
        if keys.is_empty() {
            return Ok(SortResult {
                keys: Vec::new(),
                payload: Vec::new(),
                source_hash: source_hash(sort::METAL),
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
        crate::check_aggregate_limit(storage)?;
        self.check_submission(words.iter().copied().max().unwrap_or(1) * 4, storage)?;
        let started = Instant::now();
        let pipelines = sort::KERNELS
            .iter()
            .map(|name| {
                self.compile(sort::METAL, name)
                    .map(|(pipeline, _)| pipeline)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if pipelines.iter().any(|p| {
            p.maxTotalThreadsPerThreadgroup() < sort::LANES as usize
                || p.threadExecutionWidth() != 32
        }) {
            return Err(Error::ResourceLimit(
                "sort requires 256 lanes and 32-wide SIMD groups",
            ));
        }
        let compile_ms = started.elapsed().as_secs_f64() * 1000.0;
        let buffers = words
            .iter()
            .map(|&n| self.buffer(n.max(1) * 4))
            .collect::<Result<Vec<_>, _>>()?;
        for (values, buffer) in [keys, payload].into_iter().zip(&buffers) {
            // SAFETY: each retained shared buffer has values.len() initialized
            // word slots; no command has been submitted and copies do not alias.
            unsafe {
                ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    buffer.contents().as_ptr().cast(),
                    values.len(),
                )
            };
        }
        for pass in 0..8 {
            let p = plan.parameters(pass);
            // SAFETY: parameter buffers have exactly five writable u32 slots,
            // are private to this work, and are not in flight yet.
            unsafe {
                ptr::copy_nonoverlapping(
                    p.as_ptr(),
                    buffers[8 + pass].contents().as_ptr().cast(),
                    5,
                )
            };
        }
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Error::Unavailable("Metal sort command buffer unavailable".into()))?;
        for launch in plan.launches() {
            let bound = launch
                .buffers
                .iter()
                .map(|&slot| &*buffers[slot])
                .collect::<Vec<_>>();
            self.encode(
                &command,
                &pipelines[launch.kernel],
                &bound,
                launch.blocks,
                sort::LANES,
            )?;
        }
        let mut work = MetalWork::prepared(command.clone(), buffers.clone(), pipelines);
        work.submit()?;
        if fault == Fault::PartialCompletion {
            self.quarantine(work);
            return Err(Error::Pending);
        }
        let work = self.await_terminal(work, Instant::now())?;
        if work.state != WorkState::Completed {
            return Err(Error::Device(format!(
                "Metal sort failed: {:?}",
                command.error()
            )));
        }
        let kernel_ms = (command.GPUEndTime() - command.GPUStartTime()) * 1000.0;
        let read = |slot: usize| -> Result<Vec<u32>, Error> {
            let mut out = Vec::new();
            out.try_reserve_exact(keys.len())
                .map_err(|_| Error::OutOfMemory)?;
            out.resize(keys.len(), 0);
            // SAFETY: Completed is terminal and work retains all buffers. Slot
            // 0/1 each hold keys.len() initialized output words after eight passes.
            unsafe {
                ptr::copy_nonoverlapping(
                    buffers[slot].contents().as_ptr().cast(),
                    out.as_mut_ptr(),
                    out.len(),
                )
            };
            Ok(out)
        };
        Ok(SortResult {
            keys: read(0)?,
            payload: read(1)?,
            source_hash: source_hash(sort::METAL),
            compile_ms,
            kernel_ms,
            transfer_bytes: 0,
        })
    }
}
