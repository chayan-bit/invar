//! Issue #2 (M4-02).
#![forbid(unsafe_code)]
use invar::{
    check, eval, front,
    transpose_source::host::{self, CpuDevice, CpuWork},
};
use invar_cuda::{Device, Error, Runtime, SubmitError};
use invar_device::Capabilities;
use std::{cell::Cell, rc::Rc};
fn source(floating: bool, rows: u32, cols: u32, short: bool) -> String {
    let (ty, fill, zero, one) = if floating {
        ("f32", "filled_f32", "0.0", "1.0")
    } else {
        ("u32", "filled_u32", "0u32", "1u32")
    };
    let n = if short { 1 } else { rows * cols };
    format!(
        "fn transpose(a:read Array<{ty}>,rows:u32,cols:u32)->Array<{ty}>{{var b={fill}(rows*cols,{zero});for r in 0u32..rows{{for c in 0u32..cols{{b[c*rows+r]=a[r*cols+c];}}}}return b;}} fn main()->i32{{let a={fill}({n}u32,{one});let b=transpose(a,{rows}u32,{cols}u32);for i in 0u32..len(b){{if b[i]!={one}{{return 0;}}}}return 7;}}"
    )
}
fn reference(s: &str, fuel: u64) -> eval::Observation {
    eval::observe(
        &check::check(front::parse(s.as_bytes()).unwrap(), true).unwrap(),
        fuel,
    )
}
fn parity(a: &eval::Observation, b: &eval::Observation) {
    assert_eq!(a.result, b.result);
    assert_eq!(a.steps_used, b.steps_used);
    assert_eq!(a.cleanup_trace, b.cleanup_trace);
}
#[test]
fn reached_transpose_publication_preserves_typed_results_fuel_and_cleanup() {
    for floating in [false, true] {
        for (rows, cols) in [(0, 33), (33, 0), (1, 1), (3, 5), (17, 31)] {
            let s = source(floating, rows, cols, false);
            let mut runtime = Runtime::new(CpuDevice, 2);
            let got = host::run_source(
                s.as_bytes(),
                1000000,
                &mut runtime,
                Capabilities::cuda_12_9_sm89(),
            )
            .unwrap();
            parity(&reference(&s, 1000000), &got.observation);
            assert_eq!(got.completed, usize::from(rows > 0), "{:?}", got.refusals);
        }
    }
}
#[test]
fn every_transpose_source_fuel_prefix_and_short_input_keep_original_spans() {
    for floating in [false, true] {
        for short in [false, true] {
            let s = source(floating, 2, 3, short);
            let full = reference(&s, 100000);
            let mut completions = 0;
            for fuel in 1..=full.steps_used + 1 {
                let mut runtime = Runtime::new(CpuDevice, 2);
                let got = host::run_source(
                    s.as_bytes(),
                    fuel,
                    &mut runtime,
                    Capabilities::cuda_12_9_sm89(),
                )
                .unwrap();
                parity(&reference(&s, fuel), &got.observation);
                completions += got.completed;
            }
            if short {
                assert_eq!(completions, 0);
            } else {
                assert!(completions > 0);
            }
        }
    }
}
#[derive(Clone, Copy)]
enum Fault {
    Pending,
    Unknown,
    Read,
    Stale,
}
struct Probe {
    cpu: CpuDevice,
    fault: Fault,
    reads: Rc<Cell<usize>>,
    prepares: Rc<Cell<usize>>,
    release: Rc<Cell<bool>>,
}
impl Device for Probe {
    type Work = CpuWork;
    fn prepare(&mut self, b: &[u8]) -> Result<CpuWork, Error> {
        self.prepares.set(self.prepares.get() + 1);
        self.cpu.prepare(b)
    }
    fn submit(&mut self, w: &mut CpuWork) -> Result<(), SubmitError> {
        if matches!(self.fault, Fault::Unknown) {
            Err(SubmitError::InFlightUnknown(Error::Timeout))
        } else {
            self.cpu.submit(w)
        }
    }
    fn complete(&mut self, w: &mut CpuWork) -> Result<bool, Error> {
        if self.release.get() {
            Ok(true)
        } else if matches!(self.fault, Fault::Pending | Fault::Unknown) {
            Ok(false)
        } else {
            self.cpu.complete(w)
        }
    }
    fn read(&mut self, w: &mut CpuWork) -> Result<Vec<u8>, Error> {
        self.reads.set(self.reads.get() + 1);
        if matches!(self.fault, Fault::Read) {
            return Err(Error::Driver("read".into()));
        }
        let mut b = self.cpu.read(w)?;
        if matches!(self.fault, Fault::Stale) {
            b[104] ^= 1;
        }
        Ok(b)
    }
}
#[test]
fn pending_transpose_retains_work_and_never_replays_or_publishes() {
    let s = source(false, 2, 3, false);
    for fault in [Fault::Pending, Fault::Unknown, Fault::Read, Fault::Stale] {
        let reads = Rc::new(Cell::new(0));
        let prepares = Rc::new(Cell::new(0));
        let release = Rc::new(Cell::new(false));
        let mut runtime = Runtime::new(
            Probe {
                cpu: CpuDevice,
                fault,
                reads: reads.clone(),
                prepares: prepares.clone(),
                release: release.clone(),
            },
            2,
        );
        let got = host::run_source(
            s.as_bytes(),
            100000,
            &mut runtime,
            Capabilities::cuda_12_9_sm89(),
        )
        .unwrap();
        assert_eq!(got.completed, 0);
        if matches!(fault, Fault::Pending | Fault::Unknown) {
            assert!(matches!(
                got.observation.result,
                Err(eval::EvalError::Tool {
                    code: "CUDA_UNAVAILABLE",
                    ..
                })
            ));
            assert_eq!(reads.get(), 0);
            assert_eq!(runtime.quarantine_len(), 1);
            let blocked = host::run_source(
                s.as_bytes(),
                100000,
                &mut runtime,
                Capabilities::cuda_12_9_sm89(),
            )
            .unwrap();
            assert!(blocked.observation.result.is_err());
            assert_eq!(prepares.get(), 1);
            release.set(true);
            runtime.reap();
            assert_eq!(runtime.quarantine_len(), 0);
        } else {
            parity(&reference(&s, 100000), &got.observation);
            assert_eq!(runtime.quarantine_len(), 0);
        }
    }
}
#[test]
fn floating_transpose_copies_nonfinite_and_signed_zero_source_words() {
    let s = source(true, 2, 3, false)
        .replace(
            "let a=filled_f32(6u32,1.0);",
            "let a:Array<f32> =[0.0,-0.0,1.0/0.0,0.0/0.0,-1.0/0.0,1.0];",
        )
        .replace(
            "for i in 0u32..len(b){if b[i]!=1.0{return 0;}}",
            "if b[0u32]!=0.0 or b[1u32]==b[1u32] or b[3u32]!=-1.0/0.0 or b[4u32]!=1.0/0.0 or b[5u32]!=1.0 {return 0;} if 1.0/b[0u32]!=1.0/0.0 or 1.0/b[2u32]!=-1.0/0.0 {return 0;}",
        );
    let mut runtime = Runtime::new(CpuDevice, 2);
    let got = host::run_source(
        s.as_bytes(),
        100000,
        &mut runtime,
        Capabilities::cuda_12_9_sm89(),
    )
    .unwrap();
    parity(&reference(&s, 100000), &got.observation);
    assert_eq!(got.observation.result, Ok(eval::Value::I32(7)));
    assert_eq!(got.completed, 1);
}

#[test]
fn transpose_publishes_every_nonuniform_u32_element() {
    let s=source(false,2,3,false).replace("let a=filled_u32(6u32,1u32);","let a:Array<u32> = [3u32,7u32,11u32,13u32,17u32,4294967295u32];").replace("for i in 0u32..len(b){if b[i]!=1u32{return 0;}}","if b[0u32]!=3u32 or b[1u32]!=13u32 or b[2u32]!=7u32 or b[3u32]!=17u32 or b[4u32]!=11u32 or b[5u32]!=4294967295u32 {return 0;}");
    let mut runtime = Runtime::new(CpuDevice, 2);
    let got = host::run_source(
        s.as_bytes(),
        100000,
        &mut runtime,
        Capabilities::cuda_12_9_sm89(),
    )
    .unwrap();
    parity(&reference(&s, 100000), &got.observation);
    assert_eq!(got.observation.result, Ok(eval::Value::I32(7)));
    assert_eq!(got.completed, 1);
}
