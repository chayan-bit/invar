//! Pinned, equivalent large resident comparison for the M3-03 map gate.

use super::*;
use invar_device::{DagOp, Geometry};

const LARGE_LEN: usize = 64 * 1024 * 1024;
const WARMUPS: usize = 10;
const SAMPLES: usize = 30;
const HANDWRITTEN: &str = include_str!("../../fixtures/handwritten-fused-map.cu");

#[derive(Debug)]
pub struct BenchMetric {
    pub median_ms: f32,
    pub q1_ms: f32,
    pub q3_ms: f32,
    /// Distribution-free 95.7% median interval for 30 samples (order stats 10,21).
    pub ci_low_ms: f32,
    pub ci_high_ms: f32,
}

#[derive(Debug)]
pub struct BenchResult {
    pub copy: BenchMetric,
    pub generated: BenchMetric,
    pub handwritten: BenchMetric,
    pub copy_gbps: f64,
    pub generated_gbps: f64,
    pub bandwidth_fraction: f64,
    pub generated_to_handwritten_ratio: f64,
    pub generated_hash: u64,
    pub handwritten_hash: u64,
    pub geometry: Geometry,
    pub length: usize,
    pub warmups: usize,
    pub samples: usize,
    pub outputs_correct: bool,
}

struct BenchWork {
    input: CudaSlice<u32>,
    outputs: Vec<CudaSlice<u32>>,
    host_outputs: Vec<Vec<u32>>,
    copy: CudaSlice<u32>,
    generated: CudaFunction,
    handwritten: CudaFunction,
    start: CudaEvent,
    end: CudaEvent,
    stream: Arc<CudaStream>,
    length: u32,
    geometry: Geometry,
}

fn launch(work: &mut BenchWork, handwritten: bool) -> Result<(), Error> {
    let function = if handwritten {
        &work.handwritten
    } else {
        &work.generated
    };
    let mut launch = work.stream.launch_builder(function);
    launch.arg(&work.input);
    for output in &mut work.outputs {
        launch.arg(output);
    }
    launch.arg(&work.length);
    // SAFETY: the verified graph and reviewed handwritten fixture share this
    // exact five-argument ABI, legal geometry and masked grid-stride indexing.
    // BenchWork owns all allocations and both modules until event completion.
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (work.geometry.blocks, 1, 1),
            block_dim: (work.geometry.lanes, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map_err(driver_error)?;
    Ok(())
}

fn timed(work: &mut BenchWork, kind: usize) -> Result<f32, Error> {
    work.start.record(&work.stream).map_err(driver_error)?;
    match kind {
        0 => work
            .stream
            .memcpy_dtod(&work.input, &mut work.copy)
            .map_err(driver_error)?,
        1 => launch(work, false)?,
        2 => launch(work, true)?,
        _ => return Err(Error::Limit),
    }
    work.end.record(&work.stream).map_err(driver_error)?;
    work.end.synchronize().map_err(driver_error)?;
    work.start.elapsed_ms(&work.end).map_err(driver_error)
}

pub(super) fn metric(mut samples: Vec<f32>) -> Result<BenchMetric, Error> {
    if samples.len() != SAMPLES || samples.iter().any(|n| !n.is_finite() || *n <= 0.0) {
        return Err(Error::Limit);
    }
    samples.sort_by(f32::total_cmp);
    Ok(BenchMetric {
        median_ms: (samples[14] + samples[15]) / 2.0,
        q1_ms: samples[7],
        q3_ms: samples[22],
        ci_low_ms: samples[9],
        ci_high_ms: samples[20],
    })
}

fn compare_outputs(work: &mut BenchWork) -> Result<bool, Error> {
    for handwritten in [false, true] {
        launch(work, handwritten)?;
        for (device, host) in work.outputs.iter().zip(&mut work.host_outputs) {
            work.stream
                .memcpy_dtoh(device, host)
                .map_err(driver_error)?;
        }
        work.stream.synchronize().map_err(driver_error)?;
        for (host, expected) in work.host_outputs.iter().zip([4u32, 28, 39]) {
            if host.iter().any(|value| *value != expected) {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

impl CudaDevice {
    /// Benchmark the reviewed three-stage map on exactly 64 Mi resident values.
    pub fn benchmark_fused_chain(
        &mut self,
        plan: &Verified,
        compiler: &Path,
    ) -> Result<BenchResult, Error> {
        self.admit()?;
        let dag = plan.ir().dag.as_ref().ok_or(Error::Limit)?;
        let shape = dag.nodes.iter().map(|node| node.op).collect::<Vec<_>>();
        if plan.capabilities() != self.caps
            || plan.ir().length as usize != LARGE_LEN
            || dag.inputs.len() != 1
            || dag.outputs.len() != 3
            || dag.stage_outputs != [1, 2, 3]
            || shape
                != [
                    DagOp::Input(0),
                    DagOp::WrappingAdd {
                        input: 0,
                        constant: 3,
                    },
                    DagOp::WrappingMul {
                        input: 1,
                        constant: 7,
                    },
                    DagOp::WrappingAdd {
                        input: 2,
                        constant: 11,
                    },
                ]
        {
            return Err(Error::Limit);
        }
        let generated_source = emit_cuda(plan).map_err(|e| Error::Compile(format!("{e:?}")))?;
        let generated_hash = source_hash(&generated_source);
        let handwritten_hash = source_hash(HANDWRITTEN);
        let generated_image = compile_cubin_bounded(&generated_source, compiler, self.caps)?.image;
        let handwritten_image = compile_cubin_bounded(HANDWRITTEN, compiler, self.caps)?.image;
        let generated_module = self
            .context
            .load_module(Ptx::from_binary(generated_image))
            .map_err(driver_error)?;
        let handwritten_module = self
            .context
            .load_module(Ptx::from_binary(handwritten_image))
            .map_err(driver_error)?;
        let generated = generated_module
            .load_function("invar_map_v1")
            .map_err(driver_error)?;
        let handwritten = handwritten_module
            .load_function("invar_map_v1")
            .map_err(driver_error)?;
        let mut host = Vec::new();
        host.try_reserve_exact(LARGE_LEN)
            .map_err(|_| Error::OutOfMemory)?;
        host.resize(LARGE_LEN, 1u32);
        let input = self.stream.clone_htod(&host).map_err(driver_error)?;
        self.stream.synchronize().map_err(driver_error)?;
        drop(host);
        let mut outputs = Vec::new();
        let mut host_outputs = Vec::new();
        for _ in 0..3 {
            // SAFETY: each output is fully written by every timed launch before
            // sampling or reuse; BenchWork retains it through synchronization.
            outputs.push(unsafe { self.stream.alloc::<u32>(LARGE_LEN) }.map_err(driver_error)?);
            let mut host = Vec::new();
            host.try_reserve_exact(LARGE_LEN)
                .map_err(|_| Error::OutOfMemory)?;
            host.resize(LARGE_LEN, 0u32);
            host_outputs.push(host);
        }
        // SAFETY: the D2D baseline writes every slot before any possible read.
        let copy = unsafe { self.stream.alloc::<u32>(LARGE_LEN) }.map_err(driver_error)?;
        let event = || {
            self.context
                .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(driver_error)
        };
        let mut work = Box::new(BenchWork {
            input,
            outputs,
            host_outputs,
            copy,
            generated,
            handwritten,
            start: event()?,
            end: event()?,
            stream: self.stream.clone(),
            length: plan.ir().length,
            geometry: plan.ir().geometry,
        });
        let run = (|| -> Result<BenchResult, Error> {
            let outputs_correct = compare_outputs(&mut work)?;
            if !outputs_correct {
                return Err(Error::Driver(
                    "resident output differs from reference".into(),
                ));
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
                // Fixed alternating order keeps drift from favoring one kernel.
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
            let copy = metric(copy)?;
            let generated = metric(generated)?;
            let handwritten = metric(handwritten)?;
            let bytes = (LARGE_LEN * std::mem::size_of::<u32>()) as f64;
            let copy_gbps = 2.0 * bytes / f64::from(copy.median_ms) / 1_000_000.0;
            let generated_gbps = 4.0 * bytes / f64::from(generated.median_ms) / 1_000_000.0;
            Ok(BenchResult {
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
                geometry: plan.ir().geometry,
                length: LARGE_LEN,
                warmups: WARMUPS,
                samples: SAMPLES,
                outputs_correct,
            })
        })();
        if run.is_err() {
            // A failed asynchronous benchmark can leave work in flight. Keep
            // context, modules and all allocations alive and poison dispatch.
            self.hold(work);
        }
        run
    }
}
