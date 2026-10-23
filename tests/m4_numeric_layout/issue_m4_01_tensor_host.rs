#![forbid(unsafe_code)]
use invar::{
    check, eval, front,
    tensor::host::{self, CpuDevice, CpuWork},
};
use invar_cuda::{Device, Error, Runtime, SubmitError};
use invar_device::Capabilities;
use std::{cell::Cell, rc::Rc};
const MAT: &str = "fn mat(a: read Array<f32>, b: read Array<f32>, rows: u32, inner: u32, cols: u32) -> Array<f32> { var c = filled_f32(rows * cols,0.0); for i in 0u32..rows { for j in 0u32..cols { var sum: f32 = 0.0; for k in 0u32..inner { sum = sum + a[i * inner + k] * b[k * cols + j]; } c[i * cols + j] = sum; } } return c; }";
fn source(r: u32, k: u32, c: u32) -> String {
    format!(
        "{MAT} fn main() -> i32 {{ let a=filled_f32({r}u32*{k}u32,1.0); let b=filled_f32({k}u32*{c}u32,2.0); let out=mat(a,b,{r}u32,{k}u32,{c}u32); for q in 0u32..len(out) {{ if out[q] != f32_from_u32({k}u32*2u32) {{ return 0; }} }} return 7; }}"
    )
}
fn reference(s: &str, f: u64) -> eval::Observation {
    eval::observe(
        &check::check(front::parse(s.as_bytes()).unwrap(), true).unwrap(),
        f,
    )
}
fn parity(a: &eval::Observation, b: &eval::Observation) {
    assert_eq!(a.result, b.result);
    assert_eq!(a.steps_used, b.steps_used);
    assert_eq!(a.cleanup_trace, b.cleanup_trace);
}
#[derive(Clone, Copy)]
enum Fault {
    None,
    Pending,
    UnknownSubmit,
    QuiescentSubmit,
    Poll,
    Read,
    Stale,
    Truncated,
    Replay,
}
struct Probe {
    cpu: CpuDevice,
    fault: Fault,
    prepares: Rc<Cell<usize>>,
    reads: Rc<Cell<usize>>,
    release: Rc<Cell<bool>>,
    previous: Option<Vec<u8>>,
}
impl Device for Probe {
    type Work = CpuWork;
    fn prepare(&mut self, b: &[u8]) -> Result<CpuWork, Error> {
        self.prepares.set(self.prepares.get() + 1);
        self.cpu.prepare(b)
    }
    fn submit(&mut self, w: &mut CpuWork) -> Result<(), SubmitError> {
        match self.fault {
            Fault::UnknownSubmit => Err(SubmitError::InFlightUnknown(Error::Timeout)),
            Fault::QuiescentSubmit => Err(SubmitError::FailedQuiescent(Error::Compile(
                "failed".into(),
            ))),
            _ => self.cpu.submit(w),
        }
    }
    fn complete(&mut self, w: &mut CpuWork) -> Result<bool, Error> {
        if self.release.get() {
            return Ok(true);
        }
        match self.fault {
            Fault::Pending | Fault::UnknownSubmit => Ok(false),
            Fault::Poll => Err(Error::Timeout),
            _ => self.cpu.complete(w),
        }
    }
    fn read(&mut self, w: &mut CpuWork) -> Result<Vec<u8>, Error> {
        self.reads.set(self.reads.get() + 1);
        if matches!(self.fault, Fault::Read) {
            return Err(Error::Driver("read".into()));
        }
        let mut b = self.cpu.read(w)?;
        if matches!(self.fault, Fault::Replay) {
            if let Some(previous) = &self.previous {
                return Ok(previous.clone());
            }
            self.previous = Some(b.clone());
        }
        match self.fault {
            Fault::Stale => b[104] ^= 1,
            Fault::Truncated => {
                b.pop();
            }
            _ => {}
        }
        Ok(b)
    }
}
type Probed = (
    Runtime<Probe>,
    Rc<Cell<usize>>,
    Rc<Cell<usize>>,
    Rc<Cell<bool>>,
);
fn probe(fault: Fault) -> Probed {
    let p = Rc::new(Cell::new(0));
    let r = Rc::new(Cell::new(0));
    let release = Rc::new(Cell::new(false));
    (
        Runtime::new(
            Probe {
                cpu: CpuDevice,
                fault,
                prepares: p.clone(),
                reads: r.clone(),
                release: release.clone(),
                previous: None,
            },
            2,
        ),
        p,
        r,
        release,
    )
}
#[test]
fn actual_source_completed_tensor_matches_ordered_eval() {
    for (r, k, c) in [(0, 3, 2), (2, 0, 3), (1, 1, 1), (3, 5, 3)] {
        let s = source(r, k, c);
        let (mut runtime, _, _, _) = probe(Fault::None);
        let got = host::run_source(
            s.as_bytes(),
            100_000,
            &mut runtime,
            Capabilities::cuda_12_9_sm89(),
        )
        .unwrap();
        parity(&reference(&s, 100_000), &got.observation);
        if r > 0 {
            assert_eq!(got.completed, 1, "{:?}", got.refusals)
        }
    }
}
#[test]
fn every_source_fuel_prefix_preserves_result_span_and_cleanup() {
    let s = source(2, 2, 2);
    let full = reference(&s, 100_000);
    let mut completions = 0;
    for fuel in 1..=full.steps_used + 1 {
        let (mut runtime, p, _, _) = probe(Fault::None);
        let got = host::run_source(
            s.as_bytes(),
            fuel,
            &mut runtime,
            Capabilities::cuda_12_9_sm89(),
        )
        .unwrap();
        parity(&reference(&s, fuel), &got.observation);
        if got.refusals.contains(&"tensor_source_fuel") {
            assert_eq!(p.get(), 0)
        }
        completions += got.completed;
    }
    assert!(completions > 0);
}
#[test]
fn device_failure_never_publishes_and_pending_work_is_retained() {
    let s = source(1, 1, 1);
    for fault in [
        Fault::Pending,
        Fault::UnknownSubmit,
        Fault::QuiescentSubmit,
        Fault::Poll,
        Fault::Read,
        Fault::Stale,
        Fault::Truncated,
    ] {
        let (mut runtime, p, reads, release) = probe(fault);
        let got = host::run_source(
            s.as_bytes(),
            100_000,
            &mut runtime,
            Capabilities::cuda_12_9_sm89(),
        )
        .unwrap();
        assert_eq!(got.completed, 0);
        assert_eq!(p.get(), 1);
        if matches!(fault, Fault::Pending | Fault::UnknownSubmit | Fault::Poll) {
            assert!(matches!(
                &got.observation.result,
                Err(eval::EvalError::Tool {
                    code: "CUDA_UNAVAILABLE",
                    ..
                })
            ));
            assert!(got.observation.steps_used < reference(&s, 100_000).steps_used);
            assert_eq!(reads.get(), 0);
            assert_eq!(runtime.quarantine_len(), 1);
            let blocked = host::run_source(
                s.as_bytes(),
                100_000,
                &mut runtime,
                Capabilities::cuda_12_9_sm89(),
            )
            .unwrap();
            assert!(matches!(
                blocked.observation.result,
                Err(eval::EvalError::Tool {
                    code: "CUDA_UNAVAILABLE",
                    ..
                })
            ));
            assert_eq!(p.get(), 1);
            release.set(true);
            runtime.reap();
            assert_eq!(runtime.quarantine_len(), 0);
        } else {
            parity(&reference(&s, 100_000), &got.observation);
            assert_eq!(runtime.quarantine_len(), 0);
        }
    }
}
#[test]
fn source_bounds_and_unreached_loops_do_not_start_device_work() {
    for s in [
        source(2, 2, 2).replace("filled_f32(2u32*2u32,1.0)", "filled_f32(1u32,1.0)"),
        format!("{MAT} fn main() -> i32 {{ return 7; }}"),
    ] {
        let (mut runtime, p, _, _) = probe(Fault::None);
        let got = host::run_source(
            s.as_bytes(),
            100_000,
            &mut runtime,
            Capabilities::cuda_12_9_sm89(),
        )
        .unwrap();
        parity(&reference(&s, 100_000), &got.observation);
        assert_eq!(got.completed, 0);
        assert_eq!(p.get(), 0);
    }
}

#[test]
fn nonfinite_input_and_overflowed_output_retain_ordered_source() {
    for (value, prepared) in [
        ("1.0 / 0.0", 0),
        ("0.0 / 0.0", 0),
        ("340282346638528859811704183484516925440.0", 1),
    ] {
        let s = source(1, 1, 1).replace(
            "filled_f32(1u32*1u32,1.0)",
            &format!("filled_f32(1u32*1u32,{value})"),
        );
        let (mut runtime, p, _, _) = probe(Fault::None);
        let got = host::run_source(
            s.as_bytes(),
            100_000,
            &mut runtime,
            Capabilities::cuda_12_9_sm89(),
        )
        .unwrap();
        parity(&reference(&s, 100_000), &got.observation);
        assert_eq!(got.completed, 0);
        assert_eq!(p.get(), prepared);
    }
}

#[test]
fn same_read_owner_and_grouped_source_preserve_source_ticks() {
    let s = source(2, 2, 2).replace("mat(a,b,", "mat(a,a,").replace(
        "sum = sum + a[i * inner + k] * b[k * cols + j]",
        "sum = (sum + (a[i * inner + k] * b[k * cols + j]))",
    );
    let (mut runtime, _, _, _) = probe(Fault::None);
    let got = host::run_source(
        s.as_bytes(),
        100_000,
        &mut runtime,
        Capabilities::cuda_12_9_sm89(),
    )
    .unwrap();
    parity(&reference(&s, 100_000), &got.observation);
    assert_eq!(got.completed, 1);
}

#[test]
fn prior_call_response_cannot_publish_for_changed_live_inputs() {
    let s = format!(
        "{MAT} fn main() -> i32 {{ var a=filled_f32(1u32,1.0); let b=filled_f32(1u32,2.0); let first=mat(a,b,1u32,1u32,1u32); a[0u32]=3.0; let second=mat(a,b,1u32,1u32,1u32); if first[0u32]==2.0 and second[0u32]==6.0 {{ return 7; }} return 0; }}"
    );
    let (mut runtime, p, reads, _) = probe(Fault::Replay);
    let got = host::run_source(
        s.as_bytes(),
        100_000,
        &mut runtime,
        Capabilities::cuda_12_9_sm89(),
    )
    .unwrap();
    parity(&reference(&s, 100_000), &got.observation);
    assert_eq!(got.completed, 1);
    assert_eq!(p.get(), 2);
    assert_eq!(reads.get(), 2);
    assert!(got.refusals.contains(&"tensor_stale_or_malformed_output"));
}
