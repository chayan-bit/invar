//! Bounded P8 qualification dispatch using the existing quiescence boundary.
use super::*;
use objc2::{Message, msg_send, runtime::AnyObject};
use std::{
    alloc::{Layout, alloc_zeroed, dealloc},
    ffi::c_void,
    ops::Deref,
    ptr::NonNull,
    sync::Arc,
};

/// The native host fold a refused warm call runs instead of the kernel.
pub type NativeFold<'a> = Option<&'a dyn Fn(&[u32]) -> Vec<u32>>;

unsafe extern "C" {
    // libSystem; takes no arguments and cannot fail.
    fn getpagesize() -> i32;
}

/// Host-resident words in page-aligned, page-padded storage that the caller
/// owns. Apple GPUs share this memory, so a warm call wraps it without a copy.
#[derive(Clone)]
pub struct HostPages(Arc<PageAlloc>);

struct PageAlloc {
    ptr: NonNull<u32>,
    len: usize,
    layout: Layout,
}

// SAFETY: PageAlloc owns its allocation exclusively and only hands out shared
// reads through HostPages; the GPU reads it only through const slot 0.
unsafe impl Send for PageAlloc {}
unsafe impl Sync for PageAlloc {}

impl Drop for PageAlloc {
    fn drop(&mut self) {
        // SAFETY: allocated by alloc_zeroed with this exact layout.
        unsafe { dealloc(self.ptr.as_ptr().cast(), self.layout) }
    }
}

impl HostPages {
    pub fn copy_from(words: &[u32]) -> Result<Self, Error> {
        // SAFETY: getpagesize has no preconditions.
        let page = usize::try_from(unsafe { getpagesize() }).map_err(|_| Error::Limit)?;
        let bytes = words
            .len()
            .checked_mul(4)
            .ok_or(Error::Limit)?
            .max(1)
            .next_multiple_of(page);
        if bytes > crate::MAX_DEVICE_BYTES {
            return Err(Error::Limit);
        }
        let layout = Layout::from_size_align(bytes, page).map_err(|_| Error::Limit)?;
        // SAFETY: layout has nonzero size.
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) }.cast::<u32>())
            .ok_or(Error::OutOfMemory)?;
        // SAFETY: fresh allocation of at least words.len() u32, no overlap.
        unsafe { ptr::copy_nonoverlapping(words.as_ptr(), ptr.as_ptr(), words.len()) };
        Ok(Self(Arc::new(PageAlloc {
            ptr,
            len: words.len(),
            layout,
        })))
    }
}

impl Deref for HostPages {
    type Target = [u32];
    fn deref(&self) -> &[u32] {
        // SAFETY: len initialized words inside the live allocation.
        unsafe { std::slice::from_raw_parts(self.0.ptr.as_ptr(), self.0.len) }
    }
}

/// Warm-path mutation for qualification: `ZeroedHost` wraps other host pages
/// of the same size, all zero, instead of the input, and must be caught.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WarmMutation {
    None,
    ZeroedHost,
}

impl MetalDevice {
    /// Wrap host pages as a shared buffer without copying them.
    fn wrap(&self, host: &HostPages) -> Result<BufferRef, Error> {
        // SAFETY: the pointer is page aligned and the length is the whole
        // page-padded allocation, as Metal requires. A nil deallocator leaves
        // ownership with HostPages, which every command using the buffer
        // retains until it is reaped, so the pages outlive any GPU access.
        let buffer: Option<BufferRef> = unsafe {
            msg_send![
                &*self.device,
                newBufferWithBytesNoCopy: host.0.ptr.as_ptr().cast::<c_void>(),
                length: host.0.layout.size(),
                options: MTLResourceOptions::StorageModeShared,
                deallocator: Option::<&AnyObject>::None
            ]
        };
        buffer.ok_or(Error::OutOfMemory)
    }
}

impl MetalDevice {
    /// Paired P8 template benchmark, with pooled allocations on both sides.
    /// Each source implements the six-slot ABI and a scratch-reset entry.
    /// Reduction includes every partial level; scan includes scratch clearing.
    /// Returns 30 paired [generated resident, baseline resident, generated warm,
    /// baseline warm] samples in milliseconds, after 10 unrecorded warmups.
    /// Warm input is host-resident `HostPages`, wrapped without a copy inside
    /// the timed window on both sides; resident input is a pooled device copy.
    /// With `native`, the cost model refused the host-resident call: generated
    /// warm is that host fold of the same input, checked like every warm call.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn benchmark_law_pair(
        &mut self,
        sources: [&str; 2],
        entry: &str,
        input: &HostPages,
        expected: &[u32],
        words: u32,
        length: u32,
        scan: bool,
        native: NativeFold,
    ) -> Result<Vec<[f64; 4]>, Error> {
        self.benchmark_law_pair_with(
            sources,
            entry,
            input,
            expected,
            [words, length],
            scan,
            native,
            WarmMutation::None,
        )
    }

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn benchmark_law_pair_with(
        &mut self,
        sources: [&str; 2],
        entry: &str,
        host: &HostPages,
        expected: &[u32],
        [words, length]: [u32; 2],
        scan: bool,
        native: NativeFold,
        mutation: WarmMutation,
    ) -> Result<Vec<[f64; 4]>, Error> {
        let input: &[u32] = host;
        let w = words as usize;
        // Slot 0 may be host memory, so every kernel must declare it const.
        let read_only = |s: &&str| {
            s.matches("[[buffer(0)]]").count()
                == s.matches("device const uint* input [[buffer(0)]]").count()
        };
        if words == 0
            || words > 50
            || length == 0
            || length > 1 << 26
            || input.len() != length as usize * w
            || expected.len() != if scan { input.len() } else { w }
            || sources.iter().any(|s| s.len() > 1024 * 1024)
            || !sources.iter().all(read_only)
            || entry
                != if scan {
                    "invar_law_scan_v1"
                } else {
                    "invar_law_reduce_v1"
                }
        {
            return Err(Error::Limit);
        }
        self.admit()?;
        let pipelines = sources
            .iter()
            .map(|s| self.compile(s, entry).map(|(pipeline, _)| pipeline))
            .collect::<Result<Vec<_>, _>>()?;
        let reset = if scan {
            Some(
                self.compile(sources[0], "invar_law_reset_v1")
                    .map(|(pipeline, _)| pipeline)?,
            )
        } else {
            None
        };
        let baseline = if scan {
            Some([
                self.compile(sources[1], "invar_law_reduce_v1")
                    .map(|(pipeline, _)| pipeline)?,
                self.compile(sources[1], "invar_law_prefix_v1")
                    .map(|(pipeline, _)| pipeline)?,
            ])
        } else {
            None
        };
        if pipelines
            .iter()
            .any(|p| p.maxTotalThreadsPerThreadgroup() < 256)
        {
            return Err(Error::Limit);
        }
        let mut counts = vec![length];
        if !scan {
            while counts.last().copied().unwrap() > 1024 {
                counts.push(counts.last().copied().unwrap().div_ceil(1024));
            }
        }
        let groups = length.div_ceil(1024);
        let table_words = if scan { groups as usize * w * 2 } else { 1 };
        let mut sizes = vec![input.len()];
        sizes.extend(counts.iter().map(|n| {
            if scan {
                *n as usize * w
            } else {
                n.div_ceil(1024) as usize * w
            }
        }));
        let bytes = (sizes.iter().sum::<usize>() + 2 * table_words + counts.len() + 3) * 4;
        if bytes > invar_device::MAX_GNET_STORAGE_BYTES {
            return Err(Error::Limit);
        }
        let buffers = sizes
            .iter()
            .map(|n| self.buffer(n * 4))
            .collect::<Result<Vec<_>, _>>()?;
        let tables = [self.buffer(table_words * 4)?, self.buffer(table_words * 4)?];
        let mut params = counts.clone();
        params.extend([1, table_words as u32, groups]);
        let params = params
            .iter()
            .map(|x| {
                let b = self.buffer(4)?;
                // SAFETY: fresh four-byte shared buffer, no submitted command.
                unsafe { ptr::write(b.contents().as_ptr().cast::<u32>(), *x) };
                Ok(b)
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let output = buffers.last().unwrap();
        let retained = buffers
            .iter()
            .chain(&tables)
            .chain(&params)
            .cloned()
            .collect::<Vec<_>>();
        let all_pipelines = pipelines
            .iter()
            .cloned()
            .chain(reset.iter().cloned())
            .chain(baseline.iter().flat_map(|p| p.iter().cloned()))
            .collect::<Vec<_>>();
        // One command holds every dispatch of one side: scratch reset or
        // baseline aggregate/prefix passes, then each partial level.
        let wrapped_host = match mutation {
            WarmMutation::None => host.clone(),
            WarmMutation::ZeroedHost => HostPages::copy_from(&vec![0; input.len()])?,
        };
        let encode_side = |device: &Self,
                           side: usize,
                           slot0: &ProtocolObject<dyn MTLBuffer>|
         -> Result<_, Error> {
            let command = device
                .queue
                .commandBuffer()
                .ok_or_else(|| Error::Unavailable("Metal command unavailable".into()))?;
            if side == 0
                && let Some(reset) = &reset
            {
                let refs = [
                    slot0,
                    &*buffers[1],
                    &*params[counts.len() + 1],
                    &*tables[0],
                    &*tables[1],
                    &*params[counts.len()],
                ];
                device.encode(
                    &command,
                    reset,
                    &refs,
                    (table_words as u32).div_ceil(256),
                    256,
                )?;
            }
            if side == 1
                && let Some(baseline) = &baseline
            {
                let refs = [
                    slot0,
                    &*tables[0],
                    &*params[0],
                    &*tables[0],
                    &*tables[1],
                    &*params[counts.len()],
                ];
                device.encode(&command, &baseline[0], &refs, groups, 256)?;
                let refs = [
                    &*tables[0],
                    &*tables[1],
                    &*params[counts.len() + 2],
                    &*tables[0],
                    &*tables[1],
                    &*params[counts.len()],
                ];
                device.encode(&command, &baseline[1], &refs, 1, 256)?;
            }
            for (i, n) in counts.iter().enumerate() {
                let refs = [
                    if i == 0 { slot0 } else { &*buffers[i] },
                    &*buffers[i + 1],
                    &*params[i],
                    &*tables[0],
                    &*tables[1],
                    &*params[counts.len()],
                ];
                device.encode(&command, &pipelines[side], &refs, n.div_ceil(1024), 256)?;
            }
            let mut retained = retained.clone();
            retained.push(slot0.retain());
            let mut work = MetalWork::prepared(command, retained, all_pipelines.clone());
            // The pages stay alive while this command may still read them.
            work._host = Some(wrapped_host.clone());
            Ok(work)
        };
        let mut raw = Vec::new();
        // Fixed interleaving changes first side each block; independent processes
        // are the statistical resampling unit in the Rust report harness.
        for sample in 0..40 {
            let mut row = [0.0; 4];
            for side in [sample % 2, 1 - sample % 2] {
                for warm in [false, true] {
                    if warm
                        && side == 0
                        && let Some(native) = native
                    {
                        let begun = Instant::now();
                        let folded = native(input);
                        row[2] = begun.elapsed().as_secs_f64() * 1000.0;
                        if folded != expected {
                            return Err(Error::Device(
                                "P8 native fold differs from oracle representation".into(),
                            ));
                        }
                        continue;
                    }
                    let checked = warm || sample == 0;
                    if checked {
                        // Outside the timed window: poison the shared output so
                        // each checked call validates only its own side.
                        // SAFETY: all submissions reached Completed; exact extent.
                        unsafe {
                            ptr::write_bytes(
                                output.contents().as_ptr().cast::<u32>(),
                                0xcd,
                                expected.len(),
                            )
                        };
                    }
                    // Resident scope: an untimed copy of the same side runs
                    // immediately before, so the timed command starts on a busy
                    // GPU instead of one that idled during host copies.
                    let preroll = if warm {
                        None
                    } else {
                        if sample == 0 {
                            // SAFETY: all preceding submissions reached Completed;
                            // buffer 0 has exactly input.len() initialized slots.
                            unsafe {
                                ptr::copy_nonoverlapping(
                                    input.as_ptr(),
                                    buffers[0].contents().as_ptr().cast::<u32>(),
                                    input.len(),
                                )
                            }
                        }
                        let mut work = encode_side(self, side, &buffers[0])?;
                        work.submit()?;
                        Some(work)
                    };
                    let begun = Instant::now();
                    // Warm scope: wrap the host pages per call, no staging copy.
                    let slot0 = if warm {
                        self.wrap(&wrapped_host)?
                    } else {
                        buffers[0].clone()
                    };
                    let timed = encode_side(self, side, &slot0).and_then(|mut work| {
                        work.submit()?;
                        let command = work.command.clone();
                        Ok((self.await_terminal(work, Instant::now())?, command))
                    });
                    let (work, command) = match timed {
                        Ok(timed) => timed,
                        Err(error) => {
                            // Never release a possibly in-flight preroll.
                            if let Some(preroll) = preroll {
                                self.quarantine(preroll);
                            }
                            return Err(error);
                        }
                    };
                    if let Some(preroll) = preroll {
                        // Queue order completed it before the timed command.
                        let preroll = self.await_terminal(preroll, Instant::now())?;
                        if preroll.state != WorkState::Completed {
                            return Err(Error::Device("P8 benchmark preroll failed".into()));
                        }
                    }
                    if work.state != WorkState::Completed {
                        return Err(Error::Device("P8 benchmark command failed".into()));
                    }
                    let resident = (command.GPUEndTime() - command.GPUStartTime()) * 1000.0;
                    if !resident.is_finite() || resident <= 0.0 {
                        return Err(Error::Device("P8 GPU timing invalid".into()));
                    }
                    let mut result = Vec::new();
                    if checked {
                        result.resize(expected.len(), 0);
                        // SAFETY: Completed command and exact output allocation.
                        unsafe {
                            ptr::copy_nonoverlapping(
                                output.contents().as_ptr().cast::<u32>(),
                                result.as_mut_ptr(),
                                result.len(),
                            )
                        };
                    }
                    let host = begun.elapsed().as_secs_f64() * 1000.0;
                    if checked && result != expected {
                        return Err(Error::Device(format!(
                            "P8 benchmark side {side} differs from oracle representation"
                        )));
                    }
                    row[side + if warm { 2 } else { 0 }] = if warm { host } else { resident };
                }
            }
            if sample >= 10 {
                raw.push(row);
            }
        }
        Ok(raw)
    }

    /// Cost-model calibration over `words` u32: [blit copy bandwidth counting
    /// read and write, warm-path staging bandwidth] in bytes per millisecond,
    /// medians of ten after three warmups. Staging is what a warm call pays for
    /// host pages: the no-copy wrap plus the GPU's first access to freshly
    /// wrapped pages, measured as the extra wall time of a blit that reads a
    /// fresh wrap over the same blit reading resident memory.
    #[doc(hidden)]
    pub fn calibrate_law_transfer(&mut self, words: usize) -> Result<[f64; 2], Error> {
        if words == 0 || words * 8 > invar_device::MAX_GNET_STORAGE_BYTES {
            return Err(Error::Limit);
        }
        self.admit()?;
        let bytes = words * 4;
        let host = HostPages::copy_from(&vec![0x5a5a_5a5au32; words])?;
        let buffers = [self.buffer(bytes)?, self.buffer(bytes)?];
        let (mut copy, mut link) = (Vec::new(), Vec::new());
        for sample in 0..13 {
            let copied = self.timed_command(&buffers, None, 0, 0, bytes)?;
            let begun = Instant::now();
            self.timed_command(&buffers, None, 0, 0, bytes)?;
            let resident = begun.elapsed().as_secs_f64();
            let begun = Instant::now();
            let fresh = [self.wrap(&host)?, buffers[1].clone()];
            if let Err(error) = self.timed_command(&fresh, None, 0, 0, bytes) {
                // A quarantined blit may still read the pages: never free them.
                std::mem::forget(host.clone());
                return Err(error);
            }
            let staged = begun.elapsed().as_secs_f64() - resident;
            if sample >= 3 {
                copy.push(copied);
                link.push(staged * 1000.0);
            }
        }
        let median = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            (v[4] + v[5]) / 2.0
        };
        let [copy, link] = [median(copy), median(link)];
        if !(copy > 0.0 && link > 0.0 && copy.is_finite() && link.is_finite()) {
            return Err(Error::Device("P8 calibration timing invalid".into()));
        }
        Ok([2.0 * bytes as f64 / copy, bytes as f64 / link])
    }

    /// Qualification hook for generated P8 source. The six bindings follow the
    /// emitter ABI. Results are copied only after confirmed command completion.
    #[doc(hidden)]
    pub fn run_law_source(
        &mut self,
        source: &str,
        entry: &str,
        bindings: [&[u32]; 6],
        groups: u32,
        lanes: u32,
    ) -> Result<(Vec<Vec<u32>>, f64), Error> {
        let lengths: Vec<usize> = bindings.iter().map(|x| x.len()).collect();
        let bytes = lengths
            .iter()
            .try_fold(0usize, |sum, n| {
                n.checked_mul(4).and_then(|n| sum.checked_add(n))
            })
            .ok_or(Error::Limit)?;
        if source.len() > 1024 * 1024
            || lengths.contains(&0)
            || bytes > crate::MAX_DEVICE_BYTES
            || groups == 0
            || groups > (1 << 21)
            || !lanes.is_power_of_two()
            || !(32..=1024).contains(&lanes)
            || !matches!(
                entry,
                "invar_law_reduce_v1" | "invar_law_scan_v1" | "invar_law_locate_v1"
            )
        {
            return Err(Error::Limit);
        }
        self.admit()?;
        let pipeline = self.compile(source, entry).map(|(pipeline, _)| pipeline)?;
        if pipeline.maxTotalThreadsPerThreadgroup() < lanes as usize {
            return Err(Error::Limit);
        }
        let mut buffers = Vec::new();
        for binding in bindings {
            let buffer = self.buffer(binding.len() * 4)?;
            // SAFETY: fresh shared allocation has exactly binding.len()*4 bytes,
            // is private to this call and has no submitted command.
            unsafe {
                ptr::copy_nonoverlapping(
                    binding.as_ptr(),
                    buffer.contents().as_ptr().cast::<u32>(),
                    binding.len(),
                );
            }
            buffers.push(buffer);
        }
        let ms = self.timed_command(&buffers, Some(&pipeline), groups, lanes, bytes)?;
        let mut output = Vec::new();
        for (buffer, length) in buffers.iter().zip(lengths) {
            let mut words = vec![0u32; length];
            // SAFETY: timed_command confirmed completion, each retained shared
            // buffer contains length initialized words and words owns capacity.
            unsafe {
                ptr::copy_nonoverlapping(
                    buffer.contents().as_ptr().cast::<u32>(),
                    words.as_mut_ptr(),
                    length,
                );
            }
            output.push(words);
        }
        Ok((output, ms))
    }
}
