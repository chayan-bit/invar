//! Retained M3 map buffers. Host publication is a separate, completed read.
use super::*;
use crate::MAX_DEVICE_BYTES;

pub struct ResidentMap {
    plan: Verified,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    pipeline: PipelineRef,
    buffers: Vec<BufferRef>,
    input_count: usize,
    output_count: usize,
    completed: bool,
}

impl ResidentMap {
    pub fn plan(&self) -> &Verified {
        &self.plan
    }

    /// Shared input storage is immutable after preparation. No host pointer
    /// escapes; output storage is disjoint and all submissions have terminated.
    pub fn inputs(&self) -> Vec<Vec<u32>> {
        self.buffers[..self.input_count]
            .iter()
            .map(|buffer| {
                // SAFETY: preparation initialized exactly length u32s and kernels
                // only read these retained shared inputs.
                unsafe {
                    std::slice::from_raw_parts(
                        buffer.contents().as_ptr().cast::<u32>(),
                        self.plan.ir().length as usize,
                    )
                    .to_vec()
                }
            })
            .collect()
    }

    pub fn read(&self) -> Result<Vec<Vec<u32>>, Error> {
        if !self.completed {
            return Err(Error::Pending);
        }
        Ok(
            self.buffers[self.input_count..self.input_count + self.output_count]
                .iter()
                .map(|buffer| {
                    // SAFETY: a Completed command wrote every masked output word;
                    // this handle owns all shared buffers, with no writable aliases.
                    unsafe {
                        std::slice::from_raw_parts(
                            buffer.contents().as_ptr().cast::<u32>(),
                            self.plan.ir().length as usize,
                        )
                        .to_vec()
                    }
                })
                .collect(),
        )
    }
}

impl MetalDevice {
    /// Upload once. This is setup, outside the resident timing/transfer boundary.
    pub fn prepare_resident_map(
        &mut self,
        plan: Verified,
        inputs: &[&[u32]],
    ) -> Result<ResidentMap, Error> {
        self.admit()?;
        let ir = plan.ir();
        let input_count = ir.dag.as_ref().map_or(1, |d| d.inputs.len());
        let output_count = ir.dag.as_ref().map_or(1, |d| d.outputs.len());
        let bytes = (ir.length as usize).checked_mul(4).ok_or(Error::Limit)?;
        if plan.capabilities() != self.caps
            || ir.collective.is_some()
            || ir.length == 0
            || inputs.len() != input_count
            || inputs.iter().any(|a| a.len() != ir.length as usize)
            || bytes
                .checked_mul(input_count + output_count)
                .is_none_or(|n| n > MAX_DEVICE_BYTES)
        {
            return Err(Error::Limit);
        }
        let source =
            emit_msl(&plan).map_err(|e| Error::Toolchain(format!("MSL emission: {e:?}")))?;
        let (pipeline, _) = self.compile(&source, "invar_map_v1")?;
        let mut buffers = (0..input_count + output_count)
            .map(|_| self.buffer(bytes))
            .collect::<Result<Vec<_>, _>>()?;
        for (input, buffer) in inputs.iter().zip(&buffers) {
            // SAFETY: fresh shared allocations hold exactly input.len() u32s,
            // aligned and not yet submitted to a command queue.
            unsafe {
                ptr::copy_nonoverlapping(
                    input.as_ptr(),
                    buffer.contents().as_ptr().cast(),
                    input.len(),
                );
            }
        }
        let n = self.buffer(4)?;
        // SAFETY: fresh n storage holds one u32 and has not been submitted.
        unsafe {
            ptr::write(n.contents().as_ptr().cast::<u32>(), ir.length);
        }
        buffers.push(n);
        Ok(ResidentMap {
            plan,
            device: self.device.clone(),
            pipeline,
            buffers,
            input_count,
            output_count,
            completed: false,
        })
    }

    /// No H2D/D2H or allocation here. Unknown work is retained by timed_command's
    /// existing quarantine; the handle loses permission to read old output.
    /// Completion is the command's terminal state, not the GPU clock: a
    /// completed command whose clock gave no interval publishes with `None`.
    pub fn run_resident_map(&mut self, map: &mut ResidentMap) -> Result<Option<f64>, Error> {
        if !std::ptr::eq(&*self.device, &*map.device) {
            return Err(Error::Limit);
        }
        map.completed = false;
        let g = map.plan.ir().geometry;
        let ms =
            self.timed_command_checked(&map.buffers, Some(&map.pipeline), g.blocks, g.lanes, 0)?;
        map.completed = true;
        Ok(ms)
    }
}
