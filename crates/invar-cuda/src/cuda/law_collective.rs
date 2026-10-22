//! P8 qualification through the existing bounded NVRTC/owned-stream boundary.
use super::*;

struct LawWork {
    buffers: Vec<CudaSlice<u32>>,
    staging: Vec<PinnedHostSlice<u32>>,
    function: CudaFunction,
    event: CudaEvent,
}

struct LawBenchWork {
    buffers: Vec<CudaSlice<u32>>,
    host: Vec<PinnedHostSlice<u32>>,
    functions: Vec<CudaFunction>,
    start: CudaEvent,
    end: CudaEvent,
}

fn law_bench_launch(
    stream: &Arc<CudaStream>,
    work: &LawBenchWork,
    function: usize,
    indices: [usize; 4],
    n: u32,
    groups: u32,
) -> Result<(), Error> {
    let admitted = 1u32;
    let mut launch = stream.launch_builder(&work.functions[function]);
    launch
        .arg(&work.buffers[indices[0]])
        .arg(&work.buffers[indices[1]])
        .arg(&n)
        .arg(&work.buffers[indices[2]])
        .arg(&work.buffers[indices[3]])
        .arg(&admitted);
    // SAFETY: compiler-owned six-slot ABI, validated extents and fixed 256-lane
    // geometry; same-stream dependencies and ownership survive the final fence.
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (groups, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map(|_| ())
    .map_err(driver_error)
}

/// The native host fold a refused warm call runs instead of the kernel.
pub type NativeFold<'a> = Option<&'a dyn Fn(&[u32]) -> Vec<u32>>;

/// One timed native host fold, checked after the timed window: it must
/// publish exactly the oracle representation.
fn native_ms(
    native: &dyn Fn(&[u32]) -> Vec<u32>,
    input: &[u32],
    expected: &[u32],
) -> Result<f64, Error> {
    let begun = Instant::now();
    let output = native(input);
    let ms = begun.elapsed().as_secs_f64() * 1000.0;
    if output != expected {
        return Err(Error::Driver(
            "P8 native fold differs from oracle representation".into(),
        ));
    }
    Ok(ms)
}

fn median_ms(mut samples: Vec<f64>) -> Result<f64, Error> {
    if samples.is_empty() || samples.iter().any(|x| !x.is_finite() || *x <= 0.0) {
        return Err(Error::Driver("P8 calibration timing invalid".into()));
    }
    samples.sort_by(f64::total_cmp);
    Ok((samples[(samples.len() - 1) / 2] + samples[samples.len() / 2]) / 2.0)
}

impl CudaDevice {
    /// Paired complete P8 template timings. Allocations and compilation are
    /// pooled equally; scratch reset and all reduction levels are charged.
    /// Returns 30 [generated resident, baseline resident, generated warm,
    /// baseline warm] millisecond rows after ten warmups. With `native`, the
    /// cost model refused the host-resident call: generated warm is that
    /// host fold of the same input, checked like every other warm call.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn benchmark_law_pair(
        &mut self,
        sources: [&str; 2],
        entry: &str,
        input: &[u32],
        expected: &[u32],
        words: u32,
        length: u32,
        scan: bool,
        compiler: &Path,
        native: NativeFold,
    ) -> Result<Vec<[f64; 4]>, Error> {
        let w = words as usize;
        if words == 0
            || words > 50
            || length == 0
            || length > 1 << 26
            || input.len() != length as usize * w
            || expected.len() != if scan { input.len() } else { w }
            || sources.iter().any(|s| s.len() > 1024 * 1024)
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
        let mut functions = Vec::new();
        for source in sources {
            let image = compile_gnet_cubin_bounded(source, compiler, self.caps)?;
            let module = self
                .context
                .load_module(Ptx::from_binary(image.image))
                .map_err(driver_error)?;
            functions.push(module.load_function(entry).map_err(driver_error)?);
            if scan {
                let names = if functions.len() == 1 {
                    vec!["invar_law_reset_v1"]
                } else {
                    vec!["invar_law_reduce_v1", "invar_law_prefix_v1"]
                };
                for name in names {
                    functions.push(module.load_function(name).map_err(driver_error)?);
                }
            }
        }
        if functions
            .iter()
            .any(|f| f.max_threads_per_block().map_or(true, |n| n < 256))
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
        let table_index = sizes.len();
        sizes.extend([table_words, table_words]);
        let bytes = sizes.iter().sum::<usize>() * 4;
        let (free, _) = self.context.mem_get_info().map_err(driver_error)?;
        if bytes > invar_device::MAX_GNET_STORAGE_BYTES || bytes > free.saturating_sub(free / 10) {
            return Err(Error::Limit);
        }
        let event = || {
            self.context
                .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(driver_error)
        };
        let mut work = Box::new(LawBenchWork {
            buffers: Vec::new(),
            host: Vec::new(),
            functions,
            start: event()?,
            end: event()?,
        });
        let result = (|| -> Result<Vec<[f64; 4]>, Error> {
            for n in sizes {
                work.buffers
                    .push(self.stream.alloc_zeros::<u32>(n).map_err(driver_error)?);
            }
            // SAFETY: exact initialized extent retained with every launch.
            work.host.push(
                unsafe { self.context.alloc_pinned::<u32>(input.len()) }.map_err(driver_error)?,
            );
            // cudarc pins write-combined memory, which host reads at uncached
            // speed. Read back into ordinary pageable memory, as a host would.
            let mut readback = vec![0u32; expected.len()];
            // SAFETY: fresh pinned input has exactly input.len() disjoint slots.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    input.as_ptr(),
                    work.host[0].as_mut_ptr().map_err(driver_error)?,
                    input.len(),
                )
            };
            // Every dispatch of one side: scratch reset or baseline aggregate/
            // prefix passes, then each partial level.
            let launch_side = |work: &LawBenchWork, side: usize| -> Result<(), Error> {
                if scan && side == 0 {
                    law_bench_launch(
                        &self.stream,
                        work,
                        1,
                        [0, 1, table_index, table_index + 1],
                        table_words as u32,
                        (table_words as u32).div_ceil(256),
                    )?;
                }
                if scan && side == 1 {
                    law_bench_launch(
                        &self.stream,
                        work,
                        3,
                        [0, table_index, table_index, table_index + 1],
                        length,
                        groups,
                    )?;
                    law_bench_launch(
                        &self.stream,
                        work,
                        4,
                        [table_index, table_index + 1, table_index, table_index + 1],
                        groups,
                        1,
                    )?;
                }
                for (i, n) in counts.iter().enumerate() {
                    law_bench_launch(
                        &self.stream,
                        work,
                        if scan { side * 2 } else { side },
                        [i, i + 1, table_index, table_index + 1],
                        *n,
                        n.div_ceil(1024),
                    )?;
                }
                Ok(())
            };
            let upload = |work: &mut LawBenchWork| -> Result<(), Error> {
                // SAFETY: the previous call's output fence completed;
                // charge pageable-to-pinned staging in warm scope.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        input.as_ptr(),
                        work.host[0].as_mut_ptr().map_err(driver_error)?,
                        input.len(),
                    );
                }
                self.stream
                    .memcpy_htod(&work.host[0], &mut work.buffers[0])
                    .map_err(driver_error)
            };
            let mut raw = Vec::new();
            for sample in 0..40 {
                let mut row = [0.0; 4];
                for side in [sample % 2, 1 - sample % 2] {
                    for warm in [false, true] {
                        if warm
                            && side == 0
                            && let Some(native) = native
                        {
                            row[2] = native_ms(native, input, expected)?;
                            continue;
                        }
                        let checked = warm || sample == 0;
                        if checked {
                            // Outside the timed window: clear the output so each
                            // checked call validates only its own side.
                            self.stream
                                .memset_zeros(&mut work.buffers[table_index - 1])
                                .map_err(driver_error)?;
                            self.stream.synchronize().map_err(driver_error)?;
                        }
                        if !warm {
                            if sample == 0 {
                                upload(&mut work)?;
                            }
                            // Resident scope: an untimed copy of the same side runs
                            // back to back before the timed one on this stream.
                            launch_side(&work, side)?;
                        }
                        let begun = Instant::now();
                        if warm {
                            upload(&mut work)?;
                        }
                        work.start.record(&self.stream).map_err(driver_error)?;
                        launch_side(&work, side)?;
                        work.end.record(&self.stream).map_err(driver_error)?;
                        if checked {
                            // Pageable destination: returns after the copy and
                            // every prior command on this stream completed.
                            self.stream
                                .memcpy_dtoh(&work.buffers[table_index - 1], &mut readback)
                                .map_err(driver_error)?;
                        }
                        let fence = Instant::now();
                        while !work.end.is_complete() {
                            if fence.elapsed() > KERNEL_DEADLINE {
                                return Err(Error::Timeout);
                            }
                            std::thread::sleep(Duration::from_micros(50));
                        }
                        let resident =
                            f64::from(work.start.elapsed_ms(&work.end).map_err(driver_error)?);
                        let host = begun.elapsed().as_secs_f64() * 1000.0;
                        if checked && readback != expected {
                            return Err(Error::Driver(format!(
                                "P8 benchmark side {side} differs from oracle representation"
                            )));
                        }
                        if !resident.is_finite() || resident <= 0.0 {
                            return Err(Error::Driver("P8 invalid event timing".into()));
                        }
                        row[side + if warm { 2 } else { 0 }] = if warm { host } else { resident };
                    }
                }
                if sample >= 10 {
                    raw.push(row);
                }
            }
            Ok(raw)
        })();
        if result.is_err() {
            self.hold(work);
        }
        result
    }

    /// Cost-model calibration over `words` u32: [device-to-device copy
    /// bandwidth counting read and write, warm-path staging bandwidth
    /// (pageable to pinned, then host to device)] in bytes per millisecond,
    /// medians of ten samples after three warmups.
    #[doc(hidden)]
    pub fn calibrate_law_transfer(&mut self, words: usize) -> Result<[f64; 2], Error> {
        if words == 0 || words * 8 > invar_device::MAX_GNET_STORAGE_BYTES {
            return Err(Error::Limit);
        }
        self.admit()?;
        let input = vec![0x5a5a_5a5au32; words];
        let event = || {
            self.context
                .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(driver_error)
        };
        let mut work = Box::new(LawBenchWork {
            buffers: Vec::new(),
            host: Vec::new(),
            functions: Vec::new(),
            start: event()?,
            end: event()?,
        });
        let result = (|| -> Result<[f64; 2], Error> {
            for _ in 0..2 {
                work.buffers.push(
                    self.stream
                        .alloc_zeros::<u32>(words)
                        .map_err(driver_error)?,
                );
            }
            // SAFETY: fully overwritten before every transfer below.
            work.host
                .push(unsafe { self.context.alloc_pinned::<u32>(words) }.map_err(driver_error)?);
            let (mut copy, mut link) = (Vec::new(), Vec::new());
            for sample in 0..13 {
                work.start.record(&self.stream).map_err(driver_error)?;
                let (source, target) = work.buffers.split_at_mut(1);
                self.stream
                    .memcpy_dtod(&source[0], &mut target[0])
                    .map_err(driver_error)?;
                work.end.record(&self.stream).map_err(driver_error)?;
                work.end.synchronize().map_err(driver_error)?;
                let copied = f64::from(work.start.elapsed_ms(&work.end).map_err(driver_error)?);
                let begun = Instant::now();
                // SAFETY: the stream is idle; both sides hold `words` slots.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        input.as_ptr(),
                        work.host[0].as_mut_ptr().map_err(driver_error)?,
                        words,
                    )
                };
                self.stream
                    .memcpy_htod(&work.host[0], &mut work.buffers[0])
                    .map_err(driver_error)?;
                self.stream.synchronize().map_err(driver_error)?;
                if sample >= 3 {
                    copy.push(copied);
                    link.push(begun.elapsed().as_secs_f64() * 1000.0);
                }
            }
            let bytes = (words * 4) as f64;
            Ok([2.0 * bytes / median_ms(copy)?, bytes / median_ms(link)?])
        })();
        if result.is_err() {
            self.hold(work);
        }
        result
    }

    /// Qualification hook for compiler-owned P8 source and its six-slot ABI.
    /// Every asynchronous allocation and transfer is retained through the fence.
    #[doc(hidden)]
    pub fn run_law_source(
        &mut self,
        source: &str,
        entry: &str,
        bindings: [&[u32]; 6],
        geometry: (u32, u32),
        compiler: &Path,
    ) -> Result<Vec<Vec<u32>>, Error> {
        let (groups, lanes) = geometry;
        let bytes = bindings
            .iter()
            .try_fold(0usize, |sum, words| {
                words.len().checked_mul(4).and_then(|n| sum.checked_add(n))
            })
            .ok_or(Error::Limit)?;
        if source.len() > 1024 * 1024
            || bindings.iter().any(|b| b.is_empty())
            || bindings[2].len() != 1
            || bindings[5].len() != 1
            || bytes > 2 * 1024 * 1024 * 1024
            || groups == 0
            || groups > (1 << 21)
            || !lanes.is_power_of_two()
            || !(32..=1024).contains(&lanes)
            || !matches!(
                entry,
                "invar_law_scan_v1" | "invar_law_reduce_v1" | "invar_law_locate_v1"
            )
        {
            return Err(Error::Limit);
        }
        self.admit()?;
        let (free, _) = self.context.mem_get_info().map_err(driver_error)?;
        if bytes > free.saturating_sub((free / 10).max(1)) {
            return Err(Error::Limit);
        }
        let cubin = compile_gnet_cubin_bounded(source, compiler, self.caps)?;
        let module = self
            .context
            .load_module(Ptx::from_binary(cubin.image))
            .map_err(driver_error)?;
        let function = module.load_function(entry).map_err(driver_error)?;
        if function.max_threads_per_block().map_err(driver_error)? < lanes as i32 {
            return Err(Error::Limit);
        }
        let event = self.context.new_event(None).map_err(driver_error)?;
        let mut work = Box::new(LawWork {
            buffers: Vec::new(),
            staging: Vec::new(),
            function,
            event,
        });
        let n = bindings[2][0];
        let admitted = bindings[5][0];
        let prepared = (|| -> Result<(), Error> {
            for binding in bindings {
                // SAFETY: the exact initialized extent stays owned by LawWork
                // through every asynchronous copy and the final event.
                let mut host = unsafe { self.context.alloc_pinned::<u32>(binding.len()) }
                    .map_err(driver_error)?;
                let pointer = host.as_mut_ptr().map_err(driver_error)?;
                // SAFETY: pinned allocation and source both hold binding.len()
                // disjoint u32 slots; no transfer uses this allocation yet.
                unsafe {
                    std::ptr::copy_nonoverlapping(binding.as_ptr(), pointer, binding.len());
                }
                work.staging.push(host);
                work.buffers.push(
                    self.stream
                        .alloc_zeros::<u32>(binding.len())
                        .map_err(driver_error)?,
                );
            }
            for (host, buffer) in work.staging.iter().zip(&mut work.buffers) {
                self.stream
                    .memcpy_htod(host, buffer)
                    .map_err(driver_error)?;
            }
            let mut launch = self.stream.launch_builder(&work.function);
            for (index, buffer) in work.buffers.iter_mut().enumerate() {
                match index {
                    2 => {
                        launch.arg(&n);
                    }
                    5 => {
                        launch.arg(&admitted);
                    }
                    _ => {
                        launch.arg(buffer);
                    }
                }
            }
            // SAFETY: generated P8 source uses four distinct u32 buffer pointers
            // and by-value n/admitted. Qualification supplies the admitted sizes
            // and geometry; LawWork owns every argument through completion.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (groups, 1, 1),
                    block_dim: (lanes, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(driver_error)?;
            for (buffer, host) in work.buffers.iter().zip(&mut work.staging) {
                self.stream
                    .memcpy_dtoh(buffer, host)
                    .map_err(driver_error)?;
            }
            work.event.record(&self.stream).map_err(driver_error)?;
            Ok(())
        })();
        if let Err(error) = prepared {
            self.hold(work);
            return Err(error);
        }
        let begun = Instant::now();
        while !work.event.is_complete() {
            if begun.elapsed() > KERNEL_DEADLINE {
                self.hold(work);
                return Err(Error::Timeout);
            }
            std::thread::sleep(Duration::from_micros(50));
        }
        work.staging
            .iter()
            .map(|host| host.as_slice().map(|s| s.to_vec()).map_err(driver_error))
            .collect()
    }
}
