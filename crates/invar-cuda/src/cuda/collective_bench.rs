//! M3-04 device-resident 64 Mi comparison with one reviewed fixture per law.

use super::*;
use invar_device::{Coefficient, CollectiveKind, PrefixMode, ReduceLaw};

const LARGE_LEN: usize = 64 * 1024 * 1024;
const WARMUPS: usize = 10;
const SAMPLES: usize = 30;
const REDUCTION: &str = include_str!("../../fixtures/handwritten-reduction.cu");
const SCAN: &str = include_str!("../../fixtures/handwritten-affine-scan.cu");

#[derive(Debug)]
pub struct CollectiveBenchResult {
    pub copy: BenchMetric,
    pub generated: BenchMetric,
    pub handwritten: BenchMetric,
    pub copy_gbps: f64,
    pub generated_gbps: f64,
    pub bandwidth_fraction: f64,
    pub generated_to_handwritten_ratio: f64,
    pub generated_hash: u64,
    pub handwritten_hash: u64,
    pub length: usize,
    pub blocks: u32,
    pub lanes: u32,
    pub workspace_bytes: usize,
    pub warmups: usize,
    pub samples: usize,
    pub outputs_correct: bool,
}

struct Kernels {
    block: CudaFunction,
    propagate: CudaFunction,
    writeback: CudaFunction,
}

struct Work {
    inputs: Vec<CudaSlice<u32>>,
    output: CudaSlice<u32>,
    summaries: CudaSlice<u32>,
    prefix: CudaSlice<u32>,
    copy: CudaSlice<u32>,
    generated: Kernels,
    handwritten: Kernels,
    start: CudaEvent,
    block_end: CudaEvent,
    propagate_end: CudaEvent,
    end: CudaEvent,
    stream: Arc<CudaStream>,
    length: u32,
    blocks: u32,
    lanes: u32,
    seed: u32,
    scan: bool,
}

fn load(device: &CudaDevice, source: &str, compiler: &Path) -> Result<Kernels, Error> {
    let image = compile_cubin_bounded(source, compiler, device.caps)?.image;
    let module = device
        .context
        .load_module(Ptx::from_binary(image))
        .map_err(driver_error)?;
    Ok(Kernels {
        block: module
            .load_function("invar_block_v1")
            .map_err(driver_error)?,
        propagate: module
            .load_function("invar_propagate_v1")
            .map_err(driver_error)?,
        writeback: module
            .load_function("invar_writeback_v1")
            .map_err(driver_error)?,
    })
}

fn launch(work: &mut Work, handwritten: bool, trace: bool) -> Result<(), Error> {
    let lanes = if handwritten && work.scan {
        256
    } else {
        work.lanes
    };
    let blocks = work.blocks;
    let kernels = if handwritten {
        &work.handwritten
    } else {
        &work.generated
    };
    let mut block = work.stream.launch_builder(&kernels.block);
    for input in &work.inputs {
        block.arg(input);
    }
    block.arg(&mut work.summaries).arg(&work.length);
    // SAFETY: the verifier fixes the exact collective ABI and all allocations
    // are distinct, long-lived, and fully initialized before this stream read.
    unsafe {
        block.launch(LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (lanes, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map_err(driver_error)?;
    if trace {
        work.block_end.record(&work.stream).map_err(driver_error)?;
    }
    let mut propagate = work.stream.launch_builder(&kernels.propagate);
    propagate
        .arg(&work.summaries)
        .arg(&mut work.prefix)
        .arg(&blocks);
    // SAFETY: stream order completes every block summary before propagation.
    unsafe {
        propagate.launch(LaunchConfig {
            grid_dim: (
                if work.scan && !handwritten {
                    work.blocks
                        .div_ceil(invar_device::collective::CUDA_PREFIX_SEGMENT)
                } else {
                    1
                },
                1,
                1,
            ),
            block_dim: (lanes, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map_err(driver_error)?;
    if trace {
        work.propagate_end
            .record(&work.stream)
            .map_err(driver_error)?;
    }
    let mut writeback = work.stream.launch_builder(&kernels.writeback);
    if work.scan {
        for input in &work.inputs {
            writeback.arg(input);
        }
        writeback
            .arg(&work.summaries)
            .arg(&work.prefix)
            .arg(&mut work.output)
            .arg(&work.seed)
            .arg(&work.length)
            .arg(&blocks);
    } else {
        writeback
            .arg(&work.prefix)
            .arg(&mut work.output)
            .arg(&work.seed);
    }
    // SAFETY: stream order completes propagation before writeback, and the
    // verified shape bounds every output write including the final state slot.
    unsafe {
        writeback.launch(LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (lanes, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map_err(driver_error)?;
    Ok(())
}

fn timed(work: &mut Work, kind: usize) -> Result<f32, Error> {
    work.start.record(&work.stream).map_err(driver_error)?;
    match kind {
        0 => work
            .stream
            .memcpy_dtod(&work.inputs[0], &mut work.copy)
            .map_err(driver_error)?,
        1 => launch(work, false, false)?,
        2 => launch(work, true, false)?,
        _ => return Err(Error::Limit),
    }
    work.end.record(&work.stream).map_err(driver_error)?;
    work.end.synchronize().map_err(driver_error)?;
    work.start.elapsed_ms(&work.end).map_err(driver_error)
}

fn compare(work: &mut Work) -> Result<(), Error> {
    let output_len = if work.scan { LARGE_LEN + 1 } else { 1 };
    let mut host = vec![0u32; output_len];
    for handwritten in [false, true] {
        launch(work, handwritten, false)?;
        work.stream
            .memcpy_dtoh(&work.output, &mut host)
            .map_err(driver_error)?;
        work.stream.synchronize().map_err(driver_error)?;
        if work.scan {
            if let Some((index, actual)) = host.iter().enumerate().find(|(i, value)| {
                **value != work.seed.wrapping_add((*i + 1).min(LARGE_LEN) as u32)
            }) {
                return Err(Error::Driver(format!(
                    "{} scan mismatch at {index}: expected {}, got {actual}",
                    if handwritten {
                        "handwritten"
                    } else {
                        "generated"
                    },
                    work.seed.wrapping_add((index + 1).min(LARGE_LEN) as u32)
                )));
            }
        } else if host[0] != work.seed.wrapping_add(LARGE_LEN as u32) {
            return Err(Error::Driver(format!(
                "{} reduction mismatch: expected {}, got {}",
                if handwritten {
                    "handwritten"
                } else {
                    "generated"
                },
                work.seed.wrapping_add(LARGE_LEN as u32),
                host[0]
            )));
        }
    }
    Ok(())
}

impl CudaDevice {
    /// Compare generated and reviewed kernels over identical resident buffers.
    pub fn benchmark_collective(
        &mut self,
        plan: &Verified,
        seed: u32,
        compiler: &Path,
    ) -> Result<CollectiveBenchResult, Error> {
        self.admit()?;
        let ir = plan.ir();
        let collective = ir.collective.as_ref().ok_or(Error::Limit)?;
        let scan = match collective.kind {
            CollectiveKind::Reduction {
                law: ReduceLaw::WrappingSum,
                identity: 0,
            } if collective.inputs.len() == 1 => false,
            CollectiveKind::AffineScan {
                a: Coefficient::Input(0),
                b: Coefficient::Input(1),
                prefix: PrefixMode::Inclusive,
                identity_a: 1,
                identity_b: 0,
            } if collective.inputs.len() == 2 => true,
            _ => return Err(Error::Limit),
        };
        if plan.capabilities() != self.caps
            || ir.length as usize != LARGE_LEN
            || ir.geometry.lanes != if scan { 512 } else { 256 }
            || ir.geometry.blocks != (LARGE_LEN / 4096) as u32
        {
            return Err(Error::Limit);
        }
        let generated_source = emit_cuda(plan).map_err(|e| Error::Compile(format!("{e:?}")))?;
        let handwritten_source = if scan { SCAN } else { REDUCTION };
        let generated_hash = source_hash(&generated_source);
        let handwritten_hash = source_hash(handwritten_source);
        let generated = load(self, &generated_source, compiler)?;
        let handwritten = load(self, handwritten_source, compiler)?;
        let host = vec![1u32; LARGE_LEN];
        let mut inputs = Vec::new();
        for _ in &collective.inputs {
            inputs.push(self.stream.clone_htod(&host).map_err(driver_error)?);
        }
        self.stream.synchronize().map_err(driver_error)?;
        drop(host);
        let output_len = if scan { LARGE_LEN + 1 } else { 1 };
        // SAFETY: every timed kernel fully writes the visible output before
        // D2H inspection or any subsequent reuse on the same stream.
        let output = unsafe { self.stream.alloc::<u32>(output_len) }.map_err(driver_error)?;
        let summary_words = ir.geometry.blocks as usize * if scan { 2 } else { 1 };
        // SAFETY: stream ordering initializes summaries and prefixes before
        // the dependent kernels read them; Work retains each allocation.
        let summaries = unsafe { self.stream.alloc::<u32>(summary_words) }.map_err(driver_error)?;
        // SAFETY: the propagation kernel writes every prefix before the
        // writeback kernel reads it on the same stream; Work retains it.
        let prefix = unsafe { self.stream.alloc::<u32>(summary_words) }.map_err(driver_error)?;
        // SAFETY: the measured D2D copy writes every slot before any reuse.
        let copy = unsafe { self.stream.alloc::<u32>(LARGE_LEN) }.map_err(driver_error)?;
        let event = || {
            self.context
                .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(driver_error)
        };
        let mut work = Box::new(Work {
            inputs,
            output,
            summaries,
            prefix,
            copy,
            generated,
            handwritten,
            start: event()?,
            block_end: event()?,
            propagate_end: event()?,
            end: event()?,
            stream: self.stream.clone(),
            length: ir.length,
            blocks: ir.geometry.blocks,
            lanes: ir.geometry.lanes,
            seed,
            scan,
        });
        let result = (|| -> Result<CollectiveBenchResult, Error> {
            compare(&mut work)?;
            if std::env::var_os("INVAR_PROFILE_PHASES").is_some() {
                work.start.record(&work.stream).map_err(driver_error)?;
                launch(&mut work, false, true)?;
                work.end.record(&work.stream).map_err(driver_error)?;
                work.end.synchronize().map_err(driver_error)?;
                eprintln!(
                    "collective_phases scan={} block_ms={} propagate_ms={} writeback_ms={}",
                    scan,
                    work.start
                        .elapsed_ms(&work.block_end)
                        .map_err(driver_error)?,
                    work.block_end
                        .elapsed_ms(&work.propagate_end)
                        .map_err(driver_error)?,
                    work.propagate_end
                        .elapsed_ms(&work.end)
                        .map_err(driver_error)?,
                );
            }
            for _ in 0..WARMUPS {
                for kind in [0, 1, 2] {
                    timed(&mut work, kind)?;
                }
            }
            let mut copy = Vec::with_capacity(SAMPLES);
            let mut generated = Vec::with_capacity(SAMPLES);
            let mut handwritten = Vec::with_capacity(SAMPLES);
            for index in 0..SAMPLES {
                for kind in if index % 2 == 0 { [2, 1] } else { [1, 2] } {
                    let ms = timed(&mut work, kind)?;
                    if kind == 1 {
                        generated.push(ms);
                    } else {
                        handwritten.push(ms);
                    }
                }
                copy.push(timed(&mut work, 0)?);
            }
            let copy = bench::metric(copy)?;
            let generated = bench::metric(generated)?;
            let handwritten = bench::metric(handwritten)?;
            let bytes = (LARGE_LEN * std::mem::size_of::<u32>()) as f64;
            let copy_gbps = 2.0 * bytes / f64::from(copy.median_ms) / 1_000_000.0;
            // Logical traffic: each scan reads two inputs and writes one
            // prefix array; temporary traffic is charged in elapsed time.
            let logical_bytes = if scan { 3.0 * bytes } else { bytes };
            let generated_gbps = logical_bytes / f64::from(generated.median_ms) / 1_000_000.0;
            Ok(CollectiveBenchResult {
                copy_gbps,
                generated_gbps,
                bandwidth_fraction: generated_gbps / copy_gbps,
                generated_to_handwritten_ratio: f64::from(generated.median_ms)
                    / f64::from(handwritten.median_ms),
                copy,
                generated,
                handwritten,
                generated_hash,
                handwritten_hash,
                length: LARGE_LEN,
                blocks: ir.geometry.blocks,
                lanes: ir.geometry.lanes,
                workspace_bytes: 2 * summary_words * 4,
                warmups: WARMUPS,
                samples: SAMPLES,
                outputs_correct: true,
            })
        })();
        if result.is_err() {
            // A failed asynchronous call can leave work in flight. Keep
            // modules/buffers alive and refuse future dispatch on this device.
            self.hold(work);
        }
        result
    }
}
