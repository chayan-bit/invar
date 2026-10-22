//! Three-launch collective execution with one owned completion boundary.

use super::*;
use invar_device::{CollectiveKind, PrefixMode};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollectiveFault {
    None,
    OutOfMemory,
    LaunchFailure,
    PartialCompletion,
}

struct CollectiveWork {
    input_host: Vec<PinnedHostSlice<u32>>,
    input_dev: Vec<CudaSlice<u32>>,
    output_host: PinnedHostSlice<u32>,
    output_dev: CudaSlice<u32>,
    summaries: CudaSlice<u32>,
    prefix: CudaSlice<u32>,
    block: CudaFunction,
    propagate: CudaFunction,
    writeback: CudaFunction,
    h2d_start: CudaEvent,
    h2d_end: CudaEvent,
    block_end: CudaEvent,
    propagate_end: CudaEvent,
    writeback_end: CudaEvent,
    completed: CudaEvent,
    stream: Arc<CudaStream>,
    length: u32,
    blocks: u32,
    seed: u32,
}

#[derive(Debug)]
pub struct CollectiveResult {
    pub final_state: u32,
    pub pair: Option<[u32; 2]>,
    pub prefixes: Vec<u32>,
    pub source_hash: u64,
    pub cubin_bytes: usize,
    pub transfer_bytes: usize,
    pub blocks: u32,
    pub lanes: u32,
    pub law_id: &'static str,
    pub identity: (u32, u32),
    pub combine_count: u64,
    pub scan_depth: u32,
    pub compile_ms: f64,
    pub host_to_host_ms: f64,
    pub h2d_ms: f32,
    pub block_ms: f32,
    pub propagate_ms: f32,
    pub writeback_ms: f32,
    pub d2h_ms: f32,
    pub nvrtc_log: String,
    pub cache_hit: bool,
}

impl CudaDevice {
    /// Execute a verified reduction or ordered affine scan from the graph plan.
    pub fn run_collective(
        &mut self,
        plan: &Verified,
        inputs: &[&[u32]],
        seed: u32,
        compiler: &Path,
    ) -> Result<CollectiveResult, Error> {
        self.run_collective_with_fault(plan, inputs, seed, compiler, CollectiveFault::None)
    }

    #[doc(hidden)]
    pub fn run_collective_with_fault(
        &mut self,
        plan: &Verified,
        inputs: &[&[u32]],
        seed: u32,
        compiler: &Path,
        fault: CollectiveFault,
    ) -> Result<CollectiveResult, Error> {
        self.admit()?;
        match fault {
            CollectiveFault::OutOfMemory => return Err(Error::OutOfMemory),
            CollectiveFault::LaunchFailure => return Err(Error::InvalidHandle),
            _ => {}
        }
        let ir = plan.ir();
        let collective = ir.collective.as_ref().ok_or(Error::Limit)?;
        let len = usize::try_from(ir.length).map_err(|_| Error::Limit)?;
        let blocks = usize::try_from(ir.geometry.blocks).map_err(|_| Error::Limit)?;
        let output_len = usize::try_from(collective.output.elements).map_err(|_| Error::Limit)?;
        if plan.capabilities() != self.caps
            || inputs.len() != collective.inputs.len()
            || inputs.iter().any(|values| values.len() != len)
        {
            return Err(Error::Limit);
        }
        let (law_id, identity, prefix_mode, scan) = match collective.kind {
            CollectiveKind::Reduction { law, identity } => (
                match law {
                    invar_device::ReduceLaw::WrappingSum => "wrapping-sum-v1",
                    invar_device::ReduceLaw::WrappingProduct => "wrapping-product-v1",
                },
                (identity, 0),
                PrefixMode::None,
                false,
            ),
            CollectiveKind::PairReduction {
                identity_a,
                identity_b,
            } => (
                "ordered-affine-pair-v1",
                (identity_a, identity_b),
                PrefixMode::None,
                true,
            ),
            CollectiveKind::AffineScan {
                prefix,
                identity_a,
                identity_b,
                ..
            } => ("ordered-affine-v1", (identity_a, identity_b), prefix, true),
        };
        let transfer_bytes = len
            .checked_mul(inputs.len())
            .and_then(|n| n.checked_add(output_len))
            .and_then(|n| n.checked_mul(4))
            .ok_or(Error::Limit)?;
        let summary_words = blocks
            .checked_mul(if scan { 2 } else { 1 })
            .ok_or(Error::ResourceLimit("summary size overflow"))?;
        let device_bytes = summary_words
            .checked_mul(8)
            .and_then(|n| n.checked_add(transfer_bytes))
            .ok_or(Error::ResourceLimit("collective byte count overflow"))?;
        let source = emit_cuda(plan).map_err(|e| Error::Compile(format!("{e:?}")))?;
        let hash = source_hash(&source);
        if len == 0 {
            return Ok(CollectiveResult {
                final_state: seed,
                pair: matches!(collective.kind, CollectiveKind::PairReduction { .. })
                    .then_some([1, 0]),
                prefixes: Vec::new(),
                source_hash: hash,
                cubin_bytes: 0,
                transfer_bytes: 0,
                blocks: ir.geometry.blocks,
                lanes: ir.geometry.lanes,
                law_id,
                identity,
                combine_count: 0,
                scan_depth: 0,
                compile_ms: 0.0,
                host_to_host_ms: 0.0,
                h2d_ms: 0.0,
                block_ms: 0.0,
                propagate_ms: 0.0,
                writeback_ms: 0.0,
                d2h_ms: 0.0,
                nvrtc_log: String::new(),
                cache_hit: false,
            });
        }
        self.check_submission(device_bytes)?;
        let mut loaded = self.load_functions(
            plan,
            &source,
            &["invar_block_v1", "invar_propagate_v1", "invar_writeback_v1"],
            compiler,
        )?;
        let host_started = Instant::now();
        let writeback = loaded.functions.pop().ok_or(Error::InvalidHandle)?;
        let propagate = loaded.functions.pop().ok_or(Error::InvalidHandle)?;
        let block = loaded.functions.pop().ok_or(Error::InvalidHandle)?;
        let mut input_host = Vec::new();
        let mut input_dev = Vec::new();
        for values in inputs {
            // SAFETY: every pinned slot is initialized by the exact-length host
            // copy before any asynchronous H2D submission can read it.
            let mut host =
                unsafe { self.context.alloc_pinned::<u32>(len) }.map_err(allocation_error)?;
            let ptr = host.as_mut_ptr().map_err(driver_error)?;
            // SAFETY: ptr owns len disjoint writable u32 slots; values has len.
            unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, len) };
            input_host.push(host);
            // SAFETY: H2D initializes every device slot before a kernel read.
            input_dev.push(unsafe { self.stream.alloc::<u32>(len) }.map_err(allocation_error)?);
        }
        // SAFETY: zero-initialization makes the pinned output valid for the D2H
        // mutable slice API. The completion event gates all later reads.
        let mut output_host =
            unsafe { self.context.alloc_pinned::<u32>(output_len) }.map_err(allocation_error)?;
        let ptr = output_host.as_mut_ptr().map_err(driver_error)?;
        // SAFETY: ptr owns exactly output_len writable u32 slots.
        unsafe { std::ptr::write_bytes(ptr, 0, output_len) };
        // SAFETY: the final kernel writes all visible output elements before D2H.
        let output_dev =
            unsafe { self.stream.alloc::<u32>(output_len) }.map_err(allocation_error)?;
        // SAFETY: the first kernel writes each summary, and the propagation
        // kernel writes every prefix before the final kernel accesses them.
        let summaries =
            unsafe { self.stream.alloc::<u32>(summary_words) }.map_err(allocation_error)?;
        // SAFETY: the propagation kernel writes every prefix slot before the
        // writeback kernel reads it on the same stream; the work retains it.
        let prefix =
            unsafe { self.stream.alloc::<u32>(summary_words.max(1)) }.map_err(allocation_error)?;
        let event = || {
            self.context
                .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(driver_error)
        };
        let mut work = Box::new(CollectiveWork {
            input_host,
            input_dev,
            output_host,
            output_dev,
            summaries,
            prefix,
            block,
            propagate,
            writeback,
            h2d_start: event()?,
            h2d_end: event()?,
            block_end: event()?,
            propagate_end: event()?,
            writeback_end: event()?,
            completed: event()?,
            stream: self.stream.clone(),
            length: ir.length,
            blocks: ir.geometry.blocks,
            seed,
        });
        let geometry = ir.geometry;
        let submitted = (|| -> Result<(), Error> {
            work.h2d_start.record(&work.stream).map_err(driver_error)?;
            for (host, device) in work.input_host.iter().zip(&mut work.input_dev) {
                work.stream
                    .memcpy_htod(host, device)
                    .map_err(driver_error)?;
            }
            work.h2d_end.record(&work.stream).map_err(driver_error)?;
            let mut first = work.stream.launch_builder(&work.block);
            for input in &work.input_dev {
                first.arg(input);
            }
            first.arg(&mut work.summaries).arg(&work.length);
            // SAFETY: Verified fixes the generated ABI, disjoint allocations,
            // uniform barriers, legal geometry and masked bounds. Work owns
            // every argument and module through confirmed completion.
            unsafe {
                first.launch(LaunchConfig {
                    grid_dim: (geometry.blocks, 1, 1),
                    block_dim: (geometry.lanes, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(driver_error)?;
            work.block_end.record(&work.stream).map_err(driver_error)?;
            if fault == CollectiveFault::PartialCompletion {
                return Err(Error::Timeout);
            }
            let mut second = work.stream.launch_builder(&work.propagate);
            second
                .arg(&work.summaries)
                .arg(&mut work.prefix)
                .arg(&work.blocks);
            // SAFETY: the prior launch is ordered on this stream; one block
            // propagates validated block summaries into owned prefix storage.
            unsafe {
                second.launch(LaunchConfig {
                    grid_dim: (
                        if scan {
                            geometry
                                .blocks
                                .div_ceil(invar_device::collective::CUDA_PREFIX_SEGMENT)
                        } else {
                            1
                        },
                        1,
                        1,
                    ),
                    block_dim: (geometry.lanes, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(driver_error)?;
            work.propagate_end
                .record(&work.stream)
                .map_err(driver_error)?;
            let mut third = work.stream.launch_builder(&work.writeback);
            if scan {
                for input in &work.input_dev {
                    third.arg(input);
                }
                third
                    .arg(&work.summaries)
                    .arg(&work.prefix)
                    .arg(&mut work.output_dev)
                    .arg(&work.seed)
                    .arg(&work.length)
                    .arg(&work.blocks);
            } else {
                third
                    .arg(&work.prefix)
                    .arg(&mut work.output_dev)
                    .arg(&work.seed);
            }
            // SAFETY: stream ordering gates all temporary reads and writes;
            // Verified guarantees the output allocation and launch bounds.
            unsafe {
                third.launch(LaunchConfig {
                    grid_dim: (geometry.blocks, 1, 1),
                    block_dim: (geometry.lanes, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(driver_error)?;
            work.writeback_end
                .record(&work.stream)
                .map_err(driver_error)?;
            work.stream
                .memcpy_dtoh(&work.output_dev, &mut work.output_host)
                .map_err(driver_error)?;
            work.completed.record(&work.stream).map_err(driver_error)?;
            Ok(())
        })();
        if let Err(error) = submitted {
            self.hold(work);
            return Err(error);
        }
        let deadline = Instant::now();
        while !work.completed.is_complete() {
            if deadline.elapsed() >= KERNEL_DEADLINE {
                self.hold(work);
                return Err(Error::Timeout);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        let h2d_ms = work
            .h2d_start
            .elapsed_ms(&work.h2d_end)
            .map_err(driver_error)?;
        let block_ms = work
            .h2d_end
            .elapsed_ms(&work.block_end)
            .map_err(driver_error)?;
        let propagate_ms = work
            .block_end
            .elapsed_ms(&work.propagate_end)
            .map_err(driver_error)?;
        let writeback_ms = work
            .propagate_end
            .elapsed_ms(&work.writeback_end)
            .map_err(driver_error)?;
        let d2h_ms = work
            .writeback_end
            .elapsed_ms(&work.completed)
            .map_err(driver_error)?;
        let output = work.output_host.as_slice().map_err(driver_error)?;
        let final_state = *output.last().ok_or(Error::Limit)?;
        let mut prefixes = Vec::new();
        if scan && prefix_mode != PrefixMode::None {
            prefixes
                .try_reserve_exact(len)
                .map_err(|_| Error::ResourceLimit("host prefix allocation failed"))?;
            prefixes.extend_from_slice(&output[..len]);
        }
        let host_to_host_ms = host_started.elapsed().as_secs_f64() * 1000.0;
        let combine_count = if scan {
            let blocks = u64::from(ir.geometry.blocks);
            let segments =
                blocks.div_ceil(u64::from(invar_device::collective::CUDA_PREFIX_SEGMENT));
            // Source-level upper bound including masked identity compositions.
            // Segment k rereads k preceding 512-summary groups, then scans
            // its own group without a cross-block barrier.
            blocks * (4096 + 526)
                + segments * 3199
                + 512 * segments * (segments - 1) / 2
                + blocks
                + if prefix_mode == PrefixMode::None {
                    1
                } else {
                    blocks * 8 * 2690 + 1
                }
        } else {
            u64::from(ir.length)
                + u64::from(ir.geometry.blocks) * 255
                + u64::from(ir.geometry.blocks)
                + 1
        };
        // Upper bound on dependent composition stages from the three launches.
        let scan_depth = if scan {
            let segments = ir
                .geometry
                .blocks
                .div_ceil(invar_device::collective::CUDA_PREFIX_SEGMENT);
            17 + (segments - 1)
                + 20
                + if prefix_mode == PrefixMode::None {
                    1
                } else {
                    8 * 11 + 1
                }
        } else {
            16 + 8 + ir.geometry.blocks.div_ceil(geometry.lanes) + 8 + 1
        };
        Ok(CollectiveResult {
            final_state,
            pair: matches!(collective.kind, CollectiveKind::PairReduction { .. })
                .then(|| [output[0], output[1]]),
            prefixes,
            source_hash: hash,
            cubin_bytes: loaded.cubin_bytes,
            transfer_bytes,
            blocks: ir.geometry.blocks,
            lanes: ir.geometry.lanes,
            law_id,
            identity,
            combine_count,
            scan_depth,
            compile_ms: loaded.compile_ms,
            host_to_host_ms,
            h2d_ms,
            block_ms,
            propagate_ms,
            writeback_ms,
            d2h_ms,
            nvrtc_log: loaded.nvrtc_log,
            cache_hit: loaded.cache_hit,
        })
    }
}
