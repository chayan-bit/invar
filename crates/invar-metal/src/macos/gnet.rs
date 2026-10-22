//! Retained twelve-buffer ABI5 command ownership; source admission lives in root.
use super::*;
use std::{marker::PhantomData, rc::Rc};

pub struct Work {
    buffers: Vec<BufferRef>,
    pipeline: PipelineRef,
    resolver: Option<PipelineRef>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    bytes: [usize; 12],
    groups: usize,
    threads: usize,
    // Queue, command, allocations and host access belong to one calling thread.
    _thread: PhantomData<Rc<()>>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Completion {
    Pending,
    Complete,
    FailedQuiescent,
}
impl Work {
    pub fn addresses(&self) -> [u64; 12] {
        std::array::from_fn(|i| self.buffers[i].contents().as_ptr() as usize as u64)
    }
    pub fn context_identity(&self) -> u64 {
        Retained::as_ptr(&self.device) as *const () as usize as u64
    }
    pub fn bytes(&self) -> [usize; 12] {
        self.bytes
    }
    /// Allocate and encode a fresh command while keeping the same physical state.
    pub fn submit(&mut self) -> Result<(), Error> {
        if self.command.is_some() && self.completion() != Completion::Complete {
            return Err(Error::Pending);
        }
        // Retire the previous fence before this epoch can be submitted.
        self.command = None;
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Error::Unavailable("Metal command buffer unavailable".into()))?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| Error::Unavailable("Metal compute encoder unavailable".into()))?;
        encoder.setComputePipelineState(&self.pipeline);
        for (i, buffer) in self.buffers.iter().enumerate() {
            // SAFETY: ABI5 source declares all twelve retained buffers at these
            // slots. Every capacity was checked before allocation; offset is0.
            unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), 0, i) };
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: self.groups,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threads,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        if let Some(resolver) = &self.resolver {
            let reduce = command
                .computeCommandEncoder()
                .ok_or_else(|| Error::Unavailable("P2 resolve encoder unavailable".into()))?;
            reduce.setComputePipelineState(resolver);
            for (i, buffer) in self.buffers.iter().enumerate() {
                // SAFETY: identical retained capsule slots, all producers
                // ordered before this separate encoder in the command buffer.
                unsafe {
                    reduce.setBuffer_offset_atIndex(Some(buffer), 0, i);
                }
            }
            reduce.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            reduce.endEncoding();
        }
        self.command = Some(command);
        self.command.as_ref().expect("retained command").commit();
        Ok(())
    }
    pub fn completion(&self) -> Completion {
        match self.command.as_ref().map(|command| command.status()) {
            Some(MTLCommandBufferStatus::Completed) => Completion::Complete,
            Some(MTLCommandBufferStatus::Error) => Completion::FailedQuiescent,
            _ => Completion::Pending,
        }
    }
    pub fn read(&self) -> Result<[Vec<u64>; 12], Error> {
        if self.completion() != Completion::Complete {
            return Err(Error::Pending);
        }
        let mut words: [Vec<u64>; 12] = std::array::from_fn(|_| Vec::new());
        for (i, out) in words.iter_mut().enumerate() {
            let count = self.bytes[i] / 8;
            out.try_reserve_exact(count)
                .map_err(|_| Error::OutOfMemory)?;
            out.resize(count, 0);
            // SAFETY: observed command completion precedes host reads. The
            // shared allocation owns this initialized byte extent, is retained
            // and cannot be submitted or mutated while &self is borrowed.
            unsafe {
                ptr::copy_nonoverlapping(
                    self.buffers[i].contents().as_ptr().cast::<u64>(),
                    out.as_mut_ptr(),
                    count,
                )
            };
        }
        Ok(words)
    }
}
impl Drop for Work {
    fn drop(&mut self) {
        if self.command.is_some() && self.completion() == Completion::Pending {
            // Direct boundary callers have the same conservative retention as
            // root's execution gate: unknown completion must not free resources.
            std::mem::forget(std::mem::take(&mut self.buffers));
            std::mem::forget(self.pipeline.clone());
            std::mem::forget(self.resolver.take());
            std::mem::forget(self.queue.clone());
            std::mem::forget(self.device.clone());
            std::mem::forget(self.command.take());
        }
    }
}
/// The ABI5 general-net entry point every generated net source exports.
pub const KERNEL: &str = "invar_gnet_children_v5";
impl MetalDevice {
    pub fn prepare_gnet(
        &mut self,
        source: &str,
        bytes: [usize; 12],
        input: &[u64],
        config: [u64; 6],
    ) -> Result<Work, Error> {
        self.prepare_gnet_launch(source, bytes, input, config, (1, 32), false)
    }
    fn prepare_gnet_launch(
        &mut self,
        source: &str,
        bytes: [usize; 12],
        input: &[u64],
        config: [u64; 6],
        geometry: (usize, usize),
        region: bool,
    ) -> Result<Work, Error> {
        self.admit()?;
        if source.len() > invar_device::MAX_GNET_SOURCE_BYTES
            || bytes.iter().any(|n| *n == 0 || n % 8 != 0)
            || bytes
                .iter()
                .try_fold(0usize, |a, n| a.checked_add(*n))
                .is_none_or(|n| n > invar_device::MAX_GNET_STORAGE_BYTES)
            || input.len() > bytes[0] / 8
            || bytes[3] != 48
        {
            return Err(Error::Limit);
        }
        let total = bytes
            .iter()
            .try_fold(0usize, |a, n| a.checked_add(*n))
            .ok_or(Error::Limit)?;
        let available = self
            .device
            .recommendedMaxWorkingSetSize()
            .saturating_sub(self.device.currentAllocatedSize() as u64);
        if bytes.iter().any(|n| *n > self.device.maxBufferLength()) || total as u64 > available {
            return Err(Error::Limit);
        }
        let (pipeline, _) = self.compile(
            source,
            if region {
                "invar_gnet_body_map_v1"
            } else {
                KERNEL
            },
        )?;
        if pipeline.maxTotalThreadsPerThreadgroup() < geometry.1 {
            return Err(Error::Limit);
        }
        let resolver = if region {
            let (resolve, _) = self.compile(source, "invar_gnet_body_resolve_v1")?;
            if resolve.maxTotalThreadsPerThreadgroup() < 256 {
                return Err(Error::Limit);
            }
            Some(resolve)
        } else {
            None
        };
        let mut buffers = Vec::new();
        for n in bytes {
            let buffer = self.buffer(n)?;
            // SAFETY: a newly allocated shared buffer owns n writable bytes;
            // no command exists yet and all storage is initialized before use.
            unsafe { ptr::write_bytes(buffer.contents().as_ptr().cast::<u8>(), 0, n) };
            buffers.push(buffer);
        }
        // SAFETY: typed scalar input and six-word configuration capacities are
        // checked above; these allocations remain idle and privately owned.
        unsafe {
            ptr::copy_nonoverlapping(
                input.as_ptr(),
                buffers[0].contents().as_ptr().cast(),
                input.len(),
            );
            ptr::copy_nonoverlapping(config.as_ptr(), buffers[3].contents().as_ptr().cast(), 6);
        }
        Ok(Work {
            buffers,
            pipeline,
            resolver,
            queue: self.queue.clone(),
            device: self.device.clone(),
            command: None,
            bytes,
            groups: geometry.0,
            threads: geometry.1,
            _thread: PhantomData,
        })
    }
    /// Compiler-owned P2 capsule. All twelve initialized images are retained
    /// until the observed completion; pending Drop quarantines the command.
    pub fn run_gnet_region(
        &mut self,
        source: &str,
        image: &[Vec<u64>; 12],
        groups: usize,
    ) -> Result<[Vec<u64>; 12], Error> {
        if groups == 0 || groups > 65_535 {
            return Err(Error::Limit);
        }
        let config: [u64; 6] = image[3].as_slice().try_into().map_err(|_| Error::Limit)?;
        let bytes = std::array::from_fn(|i| image[i].len().saturating_mul(8));
        let mut work =
            self.prepare_gnet_launch(source, bytes, &image[0], config, (groups, 256), true)?;
        for (buffer, words) in work.buffers.iter().zip(image) {
            // SAFETY: private idle allocations have exactly words.len()*8
            // initialized bytes; no command exists before this upload.
            unsafe {
                ptr::copy_nonoverlapping(
                    words.as_ptr(),
                    buffer.contents().as_ptr().cast(),
                    words.len(),
                );
            }
        }
        work.submit()?;
        let start = std::time::Instant::now();
        loop {
            match work.completion() {
                Completion::Complete => return work.read(),
                Completion::FailedQuiescent => {
                    return Err(Error::Device("P2 command failed".into()));
                }
                Completion::Pending => {}
            }
            if start.elapsed() > std::time::Duration::from_secs(120) {
                return Err(Error::Pending);
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
}

#[cfg(test)]
mod p5b_timing {
    include!("p5b_timing.rs");
}
