use super::{reached::*, *};
use crate::{check, eval, front};
use invar_cuda::{Device, Runtime, SubmitError};
use invar_device::Capabilities;
use std::{cell::Cell, rc::Rc};

const CAPS: Capabilities = Capabilities::Cuda {
    major: 12,
    minor: 9,
    sm: 89,
};
const SCALAR: &str = "fn mat(a:read Array<f32>,b:read Array<f32>,rows:u32,inner:u32,cols:u32)->Array<f32>{var c=filled_f32(rows*cols,0.0);for i in 0u32..rows{for j in 0u32..cols{var sum:f32=0.0;for k in 0u32..inner{sum=sum+a[i*inner+k]*b[k*cols+j];}c[i*cols+j]=sum;}}return c;}";
const DIRECT: &str = "fn mat(a:read Array<f32>,b:read Array<f32>,rows:u32,inner:u32,cols:u32)->Array<f32>{var c=filled_f32(rows*cols,0.0);for i in 0u32..rows{for j in 0u32..cols{for k in 0u32..inner{c[i*cols+j]=c[i*cols+j]+a[i*inner+k]*b[k*cols+j];}}}return c;}";
fn source(form: &str, [r, k, c]: [u32; 3]) -> String {
    format!(
        "{form} fn main()->i32{{let a=filled_f32({r}u32*{k}u32,1.5);let b=filled_f32({k}u32*{c}u32,2.0);let out=mat(a,b,{r}u32,{k}u32,{c}u32);for q in 0u32..len(out){{if out[q]!=f32_from_u32({k}u32)*3.0{{return 99;}}}}return 42;}}"
    )
}
fn nonuniform_source(form: &str, [rows, inner, cols]: [u32; 3]) -> String {
    use crate::numeric_contract::F32Bits;
    let a = (0..rows * inner)
        .map(|n| (n as i32 % 7) - 3)
        .collect::<Vec<_>>();
    let b = (0..inner * cols)
        .map(|n| (n as i32 % 5) - 2)
        .collect::<Vec<_>>();
    let mut expected = Vec::new();
    for i in 0..rows {
        for j in 0..cols {
            let mut sum = F32Bits::POS_ZERO;
            for k in 0..inner {
                sum = sum.add(
                    F32Bits::from_i32(a[(i * inner + k) as usize])
                        .mul(F32Bits::from_i32(b[(k * cols + j) as usize])),
                );
            }
            expected.push(sum);
        }
    }
    let list = |values: &[i32]| {
        values
            .iter()
            .map(|n| format!("{n}.0"))
            .collect::<Vec<_>>()
            .join(",")
    };
    let expected = expected
        .iter()
        .map(|n| format!("{}.0", n.trunc_i32().unwrap()))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{form} fn main()->i32{{let a:Array<f32> = [{}];let b:Array<f32> = [{}];let expected:Array<f32> = [{expected}];let out=mat(a,b,{rows}u32,{inner}u32,{cols}u32);for q in 0u32..len(out){{if out[q]!=expected[q]{{return 99;}}}}return 42;}}",
        list(&a),
        list(&b)
    )
}
fn program(source: &[u8]) -> mir::Program {
    mir::lower(&check::check(front::parse(source).unwrap(), true).unwrap()).unwrap()
}
fn candidates(source: &[u8], program: &mir::Program) -> Vec<Candidate> {
    admission::analyze(source, program)
        .unwrap()
        .regions()
        .iter()
        .filter_map(|r| bind(source, program, r).unwrap())
        .collect()
}
fn matching<'a>(cs: &'a [Candidate], live: &LiveLoop<'_>) -> Option<&'a Candidate> {
    cs.iter().find(|c| match &c.contract {
        Contract::Scalar(c) => c.function() == live.function && c.loop_span() == live.span,
        Contract::Direct(c) => c.function() == live.function && c.loop_span() == live.span,
    })
}
fn serial_words(p: &mir::Program) -> (Observation, Vec<crate::numeric_contract::F32Bits>) {
    let mut words = Vec::new();
    let mut provider = |live: &LiveLoop<'_>| {
        if live.function == "main"
            && let Some(out) = live.floating("out")
        {
            words = out.clone();
        }
        Ok(None)
    };
    let (observation, n) = mir_eval::observe_with_scheduler(p, 2_000_000, Some(&mut provider));
    assert_eq!(n, 0);
    (observation, words)
}
fn assert_words(step: &Step, expected: &[crate::numeric_contract::F32Bits]) {
    let crate::mir_eval::region::Elements::F32(actual) =
        &step.plan.region.as_ref().unwrap().writes[0].values
    else {
        panic!("f32 output required")
    };
    assert_eq!(
        actual, expected,
        "every completed word must equal serial MIR bits"
    );
}
fn parity(a: &Observation, b: &Observation) {
    assert_eq!(a.result, b.result);
    assert_eq!(a.steps_used, b.steps_used);
    assert_eq!(a.cleanup_trace, b.cleanup_trace);
}
#[test]
fn a1_tensor_completed_scalar_and_direct_regions_match_serial_mir() {
    for form in [SCALAR, DIRECT] {
        for dims in [
            [1, 1, 1],
            [3, 5, 2],
            [5, 0, 3],
            [0, 3, 2],
            [2, 3, 0],
            [17, 13, 19],
        ] {
            let s = if dims.contains(&0) {
                source(form, dims)
            } else {
                nonuniform_source(form, dims)
            };
            let p = program(s.as_bytes());
            let cs = candidates(s.as_bytes(), &p);
            assert_eq!(cs.len(), 1);
            let (serial, words) = serial_words(&p);
            assert_eq!(serial.result, Ok(eval::Value::I32(42)));
            let mut runtime = Runtime::new(tensor::host::CpuDevice, 1);
            let mut submissions = 0;
            let mut provider = |live: &LiveLoop<'_>| {
                let Some(c) = matching(&cs, live) else {
                    return Ok(None);
                };
                let Some(bound) = c.bind(live, CAPS) else {
                    return Ok(None);
                };
                submissions += 1;
                let step = finish(&bound, &mut runtime).unwrap();
                assert_words(&step, &words);
                Ok(Some(step.plan))
            };
            let (got, applied) =
                mir_eval::observe_with_scheduler(&p, 2_000_000, Some(&mut provider));
            parity(&serial, &got);
            let expected = usize::from(!dims.contains(&0));
            assert_eq!((applied, submissions), (expected, expected));
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Fault {
    Prepare,
    Submit,
    Pending,
    Unknown,
    Read,
    Truncated,
    Stale,
}
struct Probe {
    cpu: tensor::host::CpuDevice,
    fault: Fault,
    reads: Rc<Cell<usize>>,
}
impl Device for Probe {
    type Work = tensor::host::CpuWork;
    fn prepare(&mut self, b: &[u8]) -> Result<Self::Work, invar_cuda::Error> {
        if matches!(self.fault, Fault::Prepare) {
            return Err(invar_cuda::Error::Unavailable);
        }
        self.cpu.prepare(b)
    }
    fn submit(&mut self, w: &mut Self::Work) -> Result<(), SubmitError> {
        match self.fault {
            Fault::Submit => Err(SubmitError::FailedQuiescent(invar_cuda::Error::OutOfMemory)),
            Fault::Unknown => Err(SubmitError::InFlightUnknown(invar_cuda::Error::Timeout)),
            _ => self.cpu.submit(w),
        }
    }
    fn complete(&mut self, w: &mut Self::Work) -> Result<bool, invar_cuda::Error> {
        if matches!(self.fault, Fault::Pending | Fault::Unknown) {
            return Ok(false);
        }
        self.cpu.complete(w)
    }
    fn read(&mut self, w: &mut Self::Work) -> Result<Vec<u8>, invar_cuda::Error> {
        self.reads.set(self.reads.get() + 1);
        if matches!(self.fault, Fault::Read) {
            return Err(invar_cuda::Error::InvalidHandle);
        }
        let mut b = self.cpu.read(w)?;
        match self.fault {
            Fault::Truncated => {
                b.pop();
            }
            Fault::Stale => b[104] ^= 1,
            _ => {}
        }
        Ok(b)
    }
}
#[test]
fn a1_tensor_failed_or_unconfirmed_completions_never_publish() {
    for form in [SCALAR, DIRECT] {
        for fault in [
            Fault::Prepare,
            Fault::Submit,
            Fault::Pending,
            Fault::Unknown,
            Fault::Read,
            Fault::Truncated,
            Fault::Stale,
        ] {
            let s = source(form, [2, 3, 2]);
            let p = program(s.as_bytes());
            let cs = candidates(s.as_bytes(), &p);
            let reads = Rc::new(Cell::new(0));
            let mut runtime = Runtime::new(
                Probe {
                    cpu: tensor::host::CpuDevice,
                    fault,
                    reads: reads.clone(),
                },
                1,
            );
            let mut reached = false;
            let mut provider = |live: &LiveLoop<'_>| {
                let Some(c) = matching(&cs, live) else {
                    return Ok(None);
                };
                reached = true;
                let err = finish(&c.bind(live, CAPS).unwrap(), &mut runtime)
                    .err()
                    .expect("must refuse");
                if matches!(err, SubmitError::InFlightUnknown(_)) {
                    return Err(EvalError::Tool {
                        code: "CUDA_PENDING",
                        message: "pending".into(),
                    });
                }
                Ok(None)
            };
            let (got, applied) =
                mir_eval::observe_with_scheduler(&p, 100_000, Some(&mut provider));
            assert!(reached, "{fault:?}");
            assert_eq!(applied, 0, "{fault:?}");
            if matches!(fault, Fault::Pending | Fault::Unknown) {
                assert!(matches!(
                    got.result,
                    Err(EvalError::Tool {
                        code: "CUDA_PENDING",
                        ..
                    })
                ));
                assert_eq!(reads.get(), 0);
                assert!(runtime.blocks_dispatch());
            } else {
                parity(&got, &mir_eval::observe(&p, 100_000));
            }
        }
    }
}
#[test]
fn a1_tensor_costs_dimensions_transfers_launches_and_live_guards() {
    let s = source(DIRECT, [2, 3, 4]);
    let p = program(s.as_bytes());
    let cs = candidates(s.as_bytes(), &p);
    let model = crate::plan::DeviceCostModel {
        cpu_bytes_per_ms: 1.0,
        device_bytes_per_ms: 1e9,
        device_fixed_ms: 0.01,
    };
    let mut seen = false;
    let mut provider = |live: &LiveLoop<'_>| {
        let Some(c) = matching(&cs, live) else {
            return Ok(None);
        };
        seen = true;
        assert_eq!(
            cost(c, live, CAPS, model),
            Some(model.estimate((6 + 12 + 2 * 8) * 4, 24, 2, 1))
        );
        let threshold = crate::plan::DeviceCostModel {
            cpu_bytes_per_ms: 13.0,
            device_bytes_per_ms: 11.0,
            device_fixed_ms: 0.0,
        };
        assert!(threshold.estimate(104, 24, 2, 1).choose_device);
        assert!(!cost(c, live, CAPS, threshold).unwrap().choose_device);
        let expensive = crate::plan::DeviceCostModel {
            cpu_bytes_per_ms: 1e12,
            device_bytes_per_ms: 1.0,
            device_fixed_ms: 10.0,
        };
        assert!(!cost(c, live, CAPS, expensive).unwrap().choose_device);
        Ok(None)
    };
    let (got, n) = mir_eval::observe_with_scheduler(&p, 100_000, Some(&mut provider));
    assert!(seen);
    assert_eq!(n, 0);
    assert_eq!(got.result, Ok(eval::Value::I32(42)));
}
#[test]
fn a1_tensor_unreached_and_insufficient_fuel_submit_nothing() {
    for form in [SCALAR, DIRECT] {
        let s = source(form, [2, 3, 2]).replace(
            "let out=mat",
            "if false {let unused=mat(a,b,2u32,3u32,2u32);} let out=mat",
        );
        let s = s.replace(
            "let out=mat(a,b,2u32,3u32,2u32);",
            "let out=filled_f32(4u32,9.0);",
        );
        let p = program(s.as_bytes());
        let cs = candidates(s.as_bytes(), &p);
        let mut count = 0;
        let mut provider = |live: &LiveLoop<'_>| {
            if matching(&cs, live).is_some() {
                count += 1;
            }
            Ok(None)
        };
        let _ = mir_eval::observe_with_scheduler(&p, 100_000, Some(&mut provider));
        assert_eq!(count, 0);
        let s = source(form, [2, 3, 2]);
        let p = program(s.as_bytes());
        let cs = candidates(s.as_bytes(), &p);
        let mut submitted = 0;
        let mut provider = |live: &LiveLoop<'_>| {
            if let Some(c) = matching(&cs, live)
                && c.bind(live, CAPS).is_some()
            {
                submitted += 1;
            }
            Ok(None)
        };
        let (got, n) = mir_eval::observe_with_scheduler(&p, 50, Some(&mut provider));
        assert!(got.result.is_err());
        assert_eq!((n, submitted), (0, 0));
    }
}
#[test]
fn a1_tensor_packet_dispatch_and_missing_branch_count() {
    let s = source(DIRECT, [2, 3, 2]);
    let p = program(s.as_bytes());
    let cs = candidates(s.as_bytes(), &p);
    let mut provider = |live: &LiveLoop<'_>| {
        let Some(c) = matching(&cs, live) else {
            return Ok(None);
        };
        let bound = c.bind(live, CAPS).unwrap();
        let plan = &bound.plan;
        let mut runtime = Runtime::new(
            Dispatch(|inputs: [&[u32]; 2]| {
                let [r, k, n] = plan.ir().dimensions;
                let mut out = Vec::new();
                for i in 0..r as usize {
                    for j in 0..n as usize {
                        let mut sum = crate::numeric_contract::F32Bits::POS_ZERO;
                        for q in 0..k as usize {
                            sum = sum.add(
                                crate::numeric_contract::F32Bits::from_bits(
                                    inputs[0][i * k as usize + q],
                                )
                                .mul(
                                    crate::numeric_contract::F32Bits::from_bits(
                                        inputs[1][q * n as usize + j],
                                    ),
                                ),
                            );
                        }
                        out.push(sum.to_bits());
                    }
                }
                tensor::host::packet::logical_output(plan, &out)
                    .map_err(SubmitError::FailedQuiescent)
            }),
            1,
        );
        let mut step = finish(&bound, &mut runtime).unwrap();
        step.plan.region.as_mut().unwrap().branches.pop();
        Ok(Some(step.plan))
    };
    let (got, n) = mir_eval::observe_with_scheduler(&p, 100_000, Some(&mut provider));
    assert_eq!(n, 0);
    parity(&got, &mir_eval::observe(&p, 100_000));
}
#[test]
fn a1_tensor_source_mismatch_nonfinite_and_bounds_refuse() {
    for form in [SCALAR, DIRECT] {
        let s = source(form, [2, 3, 2]);
        let p = program(s.as_bytes());
        let changed = s.replace("1.5", "1.25");
        // Scalar contracts bind their own function; direct-cell contracts
        // additionally bind the complete verified MIR and source.
        if form == DIRECT {
            assert!(
                admission::analyze(changed.as_bytes(), &p)
                    .unwrap()
                    .regions()
                    .iter()
                    .any(|r| bind(changed.as_bytes(), &p, r)
                        .is_err_and(|e| e.code == "ADMISSION_STALE"))
            );
        }
        for bad in [
            s.replace("1.5", "(1.0/0.0)"),
            s.replace("filled_f32(2u32*3u32,1.5)", "filled_f32(1u32,1.5)"),
        ] {
            let p = program(bad.as_bytes());
            let cs = candidates(bad.as_bytes(), &p);
            let mut seen = false;
            let mut provider = |live: &LiveLoop<'_>| {
                if let Some(c) = matching(&cs, live) {
                    seen = true;
                    assert!(c.bind(live, CAPS).is_none());
                }
                Ok(None)
            };
            let (got, n) = mir_eval::observe_with_scheduler(&p, 100_000, Some(&mut provider));
            assert!(seen);
            assert_eq!(n, 0);
            parity(&got, &mir_eval::observe(&p, 100_000));
        }
    }
}
#[test]
fn a1_tensor_explicit_tc_policy_and_both_completion_outcomes() {
    let form = DIRECT.replace(
        "c[i*cols+j]+a[i*inner+k]*b[k*cols+j]",
        "tc_mul_add(a[i*inner+k],b[k*cols+j],c[i*cols+j])",
    );
    let s = source(&form, [2, 3, 2]);
    // Ordinary admission and the family-private extraction must bind the
    // same source/MIR before either completion outcome can be offered.
    let p = program(s.as_bytes());
    let contract = contraction_source::Contract::from_source(s.as_bytes(), "mat").unwrap();
    contract.verify_mir(&p).unwrap();
    let cs = candidates(s.as_bytes(), &p);
    assert_eq!(cs.len(), 1);
    for fail in [false, true] {
        let mut provider = |live: &LiveLoop<'_>| {
            let Some(c) = matching(&cs, live) else {
                return Ok(None);
            };
            let bound = c.bind(live, CAPS).unwrap();
            assert!(bound.plan.is_tc());
            let plan = &bound.plan;
            let mut runtime = Runtime::new(
                Dispatch(|inputs: [&[u32]; 2]| {
                    if fail {
                        return Err(SubmitError::FailedQuiescent(
                            invar_cuda::Error::OutOfMemory,
                        ));
                    }
                    invar_device::tc::reference(plan.ir().dimensions, inputs[0], inputs[1])
                        .map_err(|_| {
                            SubmitError::FailedQuiescent(invar_cuda::Error::InvalidHandle)
                        })
                }),
                1,
            );
            match finish(&bound, &mut runtime) {
                Ok(step) => Ok(Some(step.plan)),
                Err(_) => Err(EvalError::Tool {
                    code: "CUDA_UNAVAILABLE",
                    message: "TC completion refused".into(),
                }),
            }
        };
        let (got, n) = mir_eval::observe_with_scheduler(&p, 100_000, Some(&mut provider));
        if fail {
            assert_eq!(n, 0);
            assert!(matches!(
                got.result,
                Err(EvalError::Tool {
                    code: "CUDA_UNAVAILABLE",
                    ..
                })
            ));
        } else {
            assert_eq!(n, 1);
            assert_eq!(got.result, Ok(eval::Value::I32(42)));
        }
    }
}
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "requires an Apple GPU; verifies every completed word against serial MIR"]
fn a1_tensor_metal_output_words_match_serial_mir() {
    for form in [SCALAR, DIRECT] {
        for dims in [[1, 1, 1], [3, 5, 7], [17, 13, 19]] {
            let source = nonuniform_source(form, dims);
            let p = program(source.as_bytes());
            let mut device = built_device(source.as_bytes(), &p);
            let cs = candidates(source.as_bytes(), &p);
            let (serial, words) = serial_words(&p);
            let mut provider = |live: &LiveLoop<'_>| {
                let Some(c) = matching(&cs, live) else {
                    return Ok(None);
                };
                let (step, _, _) = metal::step(
                    source.as_bytes(),
                    c,
                    live,
                    &mut VecDeque::new(),
                    &mut device,
                    crate::schedule::metal::Fault::None,
                )
                .map_err(|e| EvalError::Tool {
                    code: e.code,
                    message: e.message,
                })?;
                assert_words(&step, &words);
                Ok(Some(step.plan))
            };
            let (got, n) = mir_eval::observe_with_scheduler(&p, 2_000_000, Some(&mut provider));
            assert_eq!(n, 1, "{:?}", got.result);
            parity(&serial, &got);
        }
    }
}
/// The production Metal device: compiled-ahead only, holding exactly the
/// libraries `invar build --target metal` builds for `source` (#194). A
/// kernel `step` loads that the build omits is refused, never compiled.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn built_device(source: &[u8], p: &mir::Program) -> invar_metal::MetalDevice {
    use crate::metal_aot;
    let mut device = invar_metal::MetalDevice::open().unwrap();
    let kernels = crate::schedule::metal::family_kernels(
        source,
        p,
        eval::DEFAULT_FUEL,
        device.capabilities(),
    )
    .unwrap();
    let bytes =
        metal_aot::build::build(metal_aot::Route::Run, "main", source, &kernels).unwrap();
    metal_aot::decode(&bytes)
        .unwrap()
        .install(&mut device)
        .unwrap();
    device
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA and INVAR_NVRTC_HELPER; verifies every completed word against serial MIR"]
fn a1_tensor_cuda_output_words_match_serial_mir() {
    let mut device = invar_cuda::cuda::CudaDevice::open().unwrap();
    let compiler = std::env::var_os("INVAR_NVRTC_HELPER").expect("NVRTC helper required");
    let compiler = std::path::Path::new(&compiler);
    println!(
        "device={} capabilities={:?}",
        device.name(),
        device.capabilities()
    );
    let tc_form = DIRECT.replace(
        "c[i*cols+j]+a[i*inner+k]*b[k*cols+j]",
        "tc_mul_add(a[i*inner+k],b[k*cols+j],c[i*cols+j])",
    );
    for form in [SCALAR, DIRECT, tc_form.as_str()] {
        for dims in [[1, 1, 1], [3, 5, 7], [17, 13, 19]] {
            let source = nonuniform_source(form, dims);
            let tc = form == tc_form;
            let p = if tc {
                contraction_source::Contract::from_source(source.as_bytes(), "mat")
                    .unwrap()
                    .verified_mir()
                    .unwrap()
            } else {
                program(source.as_bytes())
            };
            let cs = candidates(source.as_bytes(), &p);
            // This corpus has small integer operands, exactly representable
            // in f16, and exact sums: strict and TC must agree bit for bit.
            let reference =
                program(nonuniform_source(if tc { DIRECT } else { form }, dims).as_bytes());
            let (serial, words) = serial_words(&reference);
            let mut provider = |live: &LiveLoop<'_>| {
                let Some(c) = matching(&cs, live) else {
                    return Ok(None);
                };
                assert_eq!(
                    c.bind(live, device.capabilities()).unwrap().plan.is_tc(),
                    tc
                );
                let (step, _, _) = cuda::step(
                    source.as_bytes(),
                    c,
                    live,
                    &mut VecDeque::new(),
                    &mut device,
                    compiler,
                    crate::schedule::cuda::Fault::None,
                )
                .map_err(|e| EvalError::Tool {
                    code: e.code,
                    message: e.message,
                })?;
                assert_words(&step, &words);
                if tc {
                    let inputs = [live.floating("a").unwrap(), live.floating("b").unwrap()]
                        .map(|values| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                    let crate::mir_eval::region::Elements::F32(values) =
                        &step.plan.region.as_ref().unwrap().writes[0].values
                    else {
                        panic!("f32 output required")
                    };
                    let output = values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                    invar_device::tc::verify(dims, &inputs[0], &inputs[1], &output).unwrap();
                }
                Ok(Some(step.plan))
            };
            let (got, n) = mir_eval::observe_with_scheduler(&p, 2_000_000, Some(&mut provider));
            assert_eq!(n, 1);
            assert_eq!(got.result, serial.result);
            if !tc {
                parity(&serial, &got);
            }
        }
    }
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA; exercises the sm_120 binary16-subnormal regression through publication"]
fn a1_tensor_cuda_tc_subnormal_sm120_pair() {
    let mut device = invar_cuda::cuda::CudaDevice::open().unwrap();
    let compiler = std::env::var_os("INVAR_NVRTC_HELPER").expect("NVRTC helper required");
    let compiler = std::path::Path::new(&compiler);
    let a = [2f32.powi(-22).to_bits(), 2f32.powi(-23).to_bits()];
    let b = [0xc69c_a000, 0xbe6f_e000];
    let literals = |words: &[u32]| {
        words
            .iter()
            .map(|&w| format!("{:.149}", f32::from_bits(w)))
            .collect::<Vec<_>>()
            .join(",")
    };
    let form = DIRECT.replace(
        "c[i*cols+j]+a[i*inner+k]*b[k*cols+j]",
        "tc_mul_add(a[i*inner+k],b[k*cols+j],c[i*cols+j])",
    );
    let source = format!(
        "{form} fn main()->i32{{let a:Array<f32> = [{}];let b:Array<f32> = [{}];let out=mat(a,b,1u32,2u32,1u32);return 42;}}",
        literals(&a),
        literals(&b)
    );
    let p = contraction_source::Contract::from_source(source.as_bytes(), "mat")
        .unwrap()
        .verified_mir()
        .unwrap();
    let cs = candidates(source.as_bytes(), &p);
    let mut provider = |live: &LiveLoop<'_>| {
        let Some(c) = matching(&cs, live) else {
            return Ok(None);
        };
        assert!(c.bind(live, device.capabilities()).unwrap().plan.is_tc());
        let (step, _, _) = cuda::step(
            source.as_bytes(),
            c,
            live,
            &mut VecDeque::new(),
            &mut device,
            compiler,
            crate::schedule::cuda::Fault::None,
        )
        .map_err(|e| EvalError::Tool {
            code: e.code,
            message: e.message,
        })?;
        let crate::mir_eval::region::Elements::F32(values) =
            &step.plan.region.as_ref().unwrap().writes[0].values
        else {
            panic!("f32 output required")
        };
        let output = values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        let report = invar_device::tc::verify([1, 2, 1], &a, &b, &output).unwrap();
        println!(
            "sm120 subnormal pair: output={output:08x?} max_ratio={}",
            report.max_ratio
        );
        Ok(Some(step.plan))
    };
    let (got, n) = mir_eval::observe_with_scheduler(&p, 100_000, Some(&mut provider));
    assert_eq!(n, 1);
    assert_eq!(got.result, Ok(eval::Value::I32(42)));
}
#[test]
fn a1_tensor_completed_source_identity_matches_the_selected_recipe() {
    for form in [SCALAR, DIRECT] {
        let source = source(form, [2, 3, 2]);
        let p = program(source.as_bytes());
        let cs = candidates(source.as_bytes(), &p);
        let mut provider = |live: &LiveLoop<'_>| {
            let Some(c) = matching(&cs, live) else {
                return Ok(None);
            };
            let bound = c.bind(live, CAPS).unwrap();
            let emitted = match &c.contract {
                Contract::Scalar(_) => invar_device::floating::emit_cuda(&bound.plan).unwrap(),
                Contract::Direct(_) => invar_device::gemm::emit_cuda(&bound.plan).unwrap(),
            };
            let hash = invar_device::source_hash(&emitted);
            let words = vec![0; bound.plan.ir().output.elements as usize];
            assert_eq!(bound.completed_words(hash, &words).unwrap(), words);
            assert_eq!(
                bound.completed_words(hash ^ 1, &words),
                Err(invar_cuda::Error::StaleBorrow)
            );
            assert!(
                bound
                    .completed_words(hash, &words[..words.len() - 1])
                    .is_err()
            );
            Ok(None)
        };
        let (got, n) = mir_eval::observe_with_scheduler(&p, 100_000, Some(&mut provider));
        assert_eq!(n, 0);
        assert_eq!(got.result, Ok(eval::Value::I32(42)));
    }
}
