//! Handwritten spatial grid kernels on one owned ordered command buffer.
use super::*;
use invar_device::{f32_probe, spatial};

#[derive(Debug)]
pub struct SpatialResult {
    pub counts: Vec<u32>,
    pub indices: Vec<u32>,
    pub distances: Vec<u32>,
    pub source_hash: u64,
    pub compile_ms: f64,
    pub kernel_ms: f64,
    pub transfer_bytes: usize,
    pub normal_window_enabled: bool,
}
impl MetalDevice {
    pub fn run_spatial(
        &mut self,
        plan: &spatial::Plan,
        inputs: [&[u32]; 4],
    ) -> Result<SpatialResult, Error> {
        self.run_spatial_with_fault(plan, inputs, Fault::None)
    }
    pub fn run_spatial_with_fault(
        &mut self,
        plan: &spatial::Plan,
        inputs: [&[u32]; 4],
        fault: Fault,
    ) -> Result<SpatialResult, Error> {
        self.admit()?;
        if fault == Fault::OutOfMemory {
            return Err(Error::OutOfMemory);
        }
        if fault == Fault::LaunchFailure {
            return Err(Error::Device("injected spatial launch rejection".into()));
        }
        let mut initial = plan.initial(inputs).map_err(|_| Error::Limit)?;
        let normal_window_enabled = if plan.kernel() == 2 {
            let report = self.strict_f32_probe()?;
            report.mismatches().iter().all(|&i| {
                let case = &f32_probe::CASES[i];
                !matches!(
                    case.op,
                    f32_probe::Op::Add
                        | f32_probe::Op::Sub
                        | f32_probe::Op::Mul
                        | f32_probe::Op::Accumulate
                ) || case.subnormal
            })
        } else {
            false
        };
        initial[7][7] = u32::from(normal_window_enabled);
        let total = initial
            .iter()
            .try_fold(0usize, |sum, v| sum.checked_add(v.len().checked_mul(4)?))
            .ok_or(Error::Limit)?;
        let largest = initial.iter().map(|v| v.len() * 4).max().unwrap_or(4);
        crate::check_aggregate_limit(total)?;
        self.check_submission(largest, total)?;
        let source = spatial::metal_source();
        let started = Instant::now();
        let (pipeline, _) = self.compile(&source, spatial::KERNELS[plan.kernel()])?;
        let compile_ms = started.elapsed().as_secs_f64() * 1000.0;
        let buffers = initial
            .iter()
            .map(|v| self.buffer(v.len() * 4))
            .collect::<Result<Vec<_>, _>>()?;
        for (values, buffer) in initial.iter().zip(&buffers) {
            // SAFETY: the private shared allocation has exactly values.len()
            // words, is not submitted, and is retained through terminal state.
            unsafe {
                ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    buffer.contents().as_ptr().cast(),
                    values.len(),
                );
            }
        }
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Error::Unavailable("Metal spatial command buffer unavailable".into()))?;
        if plan.work() != 0 {
            let bound = buffers.iter().map(|b| &**b).collect::<Vec<_>>();
            self.encode(
                &command,
                &pipeline,
                &bound,
                plan.work().div_ceil(spatial::LANES),
                spatial::LANES,
            )?;
        }
        let mut work = MetalWork::prepared(command.clone(), buffers.clone(), vec![pipeline]);
        work.submit()?;
        if fault == Fault::PartialCompletion {
            self.quarantine(work);
            return Err(Error::Pending);
        }
        let work = self.await_terminal(work, Instant::now())?;
        if work.state != WorkState::Completed {
            return Err(Error::Device(format!(
                "Metal spatial failed: {:?}",
                command.error()
            )));
        }
        let read = |slot: usize| -> Result<Vec<u32>, Error> {
            let mut out = Vec::new();
            out.try_reserve_exact(plan.lengths()[slot])
                .map_err(|_| Error::OutOfMemory)?;
            out.resize(plan.lengths()[slot], 0);
            // SAFETY: terminal completion fences every kernel write. Retained
            // buffers cover the checked logical output lengths, including zero.
            unsafe {
                ptr::copy_nonoverlapping(
                    buffers[slot].contents().as_ptr().cast(),
                    out.as_mut_ptr(),
                    out.len(),
                );
            }
            Ok(out)
        };
        Ok(SpatialResult {
            counts: read(4)?,
            indices: read(5)?,
            distances: read(6)?,
            source_hash: source_hash(&source),
            compile_ms,
            kernel_ms: (command.GPUEndTime() - command.GPUStartTime()) * 1000.0,
            transfer_bytes: 0,
            normal_window_enabled,
        })
    }
}
