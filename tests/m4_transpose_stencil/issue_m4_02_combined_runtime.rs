#![forbid(unsafe_code)]
//! One source run publishing reached transpose and stencil loops through a
//! single runtime quarantine. Every run is compared with the independent AST
//! evaluator; combined runs must also equal the two standalone runs.
use invar::{
    check, eval, front,
    stencil_source::host::{
        self as stencil_host,
        combined::{self, CpuDevice, CpuWork},
    },
    transpose_source,
};
use invar_cuda::{Device, Error, Runtime, SubmitError};
use invar_device::{Capabilities, stencil::Boundary};
use std::{cell::RefCell, rc::Rc};

fn caps() -> Capabilities {
    Capabilities::cuda_12_9_sm89()
}

/// Recognized transpose and stencil, plus never-recognized twins (one extra
/// leading statement) that always run as ordinary source.
fn functions(boundary: Boundary) -> String {
    let center = "xs[r*cols+c]";
    let edge = |name: &str| match boundary {
        Boundary::Clamp => format!(" else {{ {name} = {center}; }}"),
        _ => String::new(),
    };
    let stencil = format!(
        "for r in 0u32..rows {{ for c in 0u32..cols {{ \
var n = 0.0; if r > 0u32 {{ n = xs[(r-1u32)*cols+c]; }}{en} \
var s = 0.0; if r+1u32 < rows {{ s = xs[(r+1u32)*cols+c]; }}{es} \
var w = 0.0; if c > 0u32 {{ w = xs[r*cols+(c-1u32)]; }}{ew} \
var e = 0.0; if c+1u32 < cols {{ e = xs[r*cols+(c+1u32)]; }}{ee} \
out[r*cols+c] = ((({center}+n)+s)+w)+e; }} }} return out;",
        en = edge("n"),
        es = edge("s"),
        ew = edge("w"),
        ee = edge("e"),
    );
    let transpose = "for r in 0u32..rows{for c in 0u32..cols{b[c*rows+r]=a[r*cols+c];}}return b;";
    format!(
        "fn tr(a:read Array<f32>,rows:u32,cols:u32)->Array<f32>{{var b=filled_f32(rows*cols,0.0);{transpose}}}
fn tr_ref(a:read Array<f32>,rows:u32,cols:u32)->Array<f32>{{let unused=0u32;var b=filled_f32(rows*cols,0.0);{transpose}}}
fn step(xs: read Array<f32>, rows: u32, cols: u32) -> Array<f32> {{ var out = filled_f32(rows*cols, 0.0); {stencil} }}
fn step_ref(xs: read Array<f32>, rows: u32, cols: u32) -> Array<f32> {{ let unused = 0u32; var out = filled_f32(rows*cols, 0.0); {stencil} }}
fn negative(x: f32) -> Bool {{ return 1.0 / x < 0.0; }}
fn same(a: f32, b: f32) -> Bool {{ if a != a {{ return b != b; }} if a == 0.0 {{ return b == 0.0 and negative(a) == negative(b); }} return a == b; }}"
    )
}

fn array(n: u32) -> String {
    const POOL: [&str; 9] = [
        "1.5",
        "-0.0",
        "0.25",
        "-3.0",
        "1048576.0",
        "0.0",
        "-0.125",
        "7.0",
        "0.0/0.0",
    ];
    if n == 0 {
        "filled_f32(0u32, 0.0)".into()
    } else {
        format!(
            "[{}]",
            (0..n)
                .map(|i| POOL[(i as usize * 7 + 3) % POOL.len()])
                .collect::<Vec<_>>()
                .join(",")
        )
    }
}

#[derive(Clone, Copy, Debug)]
enum Order {
    TransposeThenStencil,
    StencilThenTranspose,
}

/// `steps` rounds over an R x C slice, in the chosen order, through the
/// recognized functions and their twins; 7 only if every element matches.
fn program(boundary: Boundary, order: Order, rows: u32, cols: u32, steps: u32) -> String {
    let (round, round_ref) = match order {
        Order::TransposeThenStencil => (
            format!(
                "let t = tr(cur, {rows}u32, {cols}u32); let s = step(t, {cols}u32, {rows}u32); let back = tr(s, {cols}u32, {rows}u32); cur = move back;"
            ),
            format!(
                "let t2 = tr_ref(other, {rows}u32, {cols}u32); let s2 = step_ref(t2, {cols}u32, {rows}u32); let back2 = tr_ref(s2, {cols}u32, {rows}u32); other = move back2;"
            ),
        ),
        Order::StencilThenTranspose => (
            format!(
                "let s = step(cur, {rows}u32, {cols}u32); let t = tr(s, {rows}u32, {cols}u32); let back = tr(t, {cols}u32, {rows}u32); cur = move back;"
            ),
            format!(
                "let s2 = step_ref(other, {rows}u32, {cols}u32); let t2 = tr_ref(s2, {rows}u32, {cols}u32); let back2 = tr_ref(t2, {cols}u32, {rows}u32); other = move back2;"
            ),
        ),
    };
    let lit = array(rows * cols);
    format!(
        "{}
fn main() -> i32 {{ var cur: Array<f32> = {lit}; var other: Array<f32> = {lit};
for k in 0u32..{steps}u32 {{ {round} {round_ref} }}
if len(cur) != len(other) {{ return 2; }}
for i in 0u32..len(cur) {{ if same(cur[i], other[i]) == false {{ return 1; }} }}
return 7; }}",
        functions(boundary)
    )
}

fn reference(source: &str, fuel: u64) -> eval::Observation {
    eval::observe(
        &check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap(),
        fuel,
    )
}

fn parity(a: &eval::Observation, b: &eval::Observation, label: &str) {
    assert_eq!(a.result, b.result, "{label}: result");
    assert_eq!(a.steps_used, b.steps_used, "{label}: steps");
    assert_eq!(a.cleanup_trace, b.cleanup_trace, "{label}: cleanup");
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Family {
    Stencil,
    Transpose,
}
#[derive(Clone, Copy, Debug, PartialEq)]
enum Fault {
    None,
    Quiescent(Family),
    Unknown(Family),
    Pending(Family),
    Header(Family, usize),
}

#[derive(Default)]
struct Log {
    stencil_prepares: usize,
    transpose_prepares: usize,
}

struct Probe {
    cpu: CpuDevice,
    fault: Fault,
    log: Rc<RefCell<Log>>,
    release: Rc<RefCell<bool>>,
}
fn family(w: &CpuWork) -> Family {
    match w {
        CpuWork::Stencil(_) => Family::Stencil,
        CpuWork::Transpose(_) => Family::Transpose,
    }
}
impl Device for Probe {
    type Work = CpuWork;
    fn prepare(&mut self, b: &[u8]) -> Result<CpuWork, Error> {
        let w = self.cpu.prepare(b)?;
        let mut log = self.log.borrow_mut();
        match family(&w) {
            Family::Stencil => log.stencil_prepares += 1,
            Family::Transpose => log.transpose_prepares += 1,
        }
        Ok(w)
    }
    fn submit(&mut self, w: &mut CpuWork) -> Result<(), SubmitError> {
        if *self.release.borrow() {
            return self.cpu.submit(w);
        }
        match self.fault {
            Fault::Quiescent(f) if f == family(w) => {
                Err(SubmitError::FailedQuiescent(Error::Driver("q".into())))
            }
            Fault::Unknown(f) if f == family(w) => {
                Err(SubmitError::InFlightUnknown(Error::Timeout))
            }
            _ => self.cpu.submit(w),
        }
    }
    fn complete(&mut self, w: &mut CpuWork) -> Result<bool, Error> {
        if *self.release.borrow() {
            return Ok(true);
        }
        match self.fault {
            Fault::Unknown(f) | Fault::Pending(f) if f == family(w) => Ok(false),
            _ => self.cpu.complete(w),
        }
    }
    fn read(&mut self, w: &mut CpuWork) -> Result<Vec<u8>, Error> {
        let f = family(w);
        let mut b = self.cpu.read(w)?;
        if let Fault::Header(target, i) = self.fault
            && target == f
        {
            b[i] ^= 1;
        }
        Ok(b)
    }
}

fn runtime(fault: Fault) -> (Runtime<Probe>, Rc<RefCell<Log>>, Rc<RefCell<bool>>) {
    let log = Rc::new(RefCell::new(Log::default()));
    let release = Rc::new(RefCell::new(false));
    let probe = Probe {
        cpu: CpuDevice::default(),
        fault,
        log: log.clone(),
        release: release.clone(),
    };
    (Runtime::new(probe, 4), log, release)
}

fn tagged<'a>(refusals: &'a [&'static str], prefix: &str) -> Vec<&'a str> {
    refusals
        .iter()
        .copied()
        .filter(|r| r.starts_with(prefix))
        .collect()
}

/// Combined run, plus both standalone runs on the same program and fault.
fn check_composition(source: &str, fuel: u64, fault: Fault, label: &str) -> stencil_host::Run {
    let expected = reference(source, fuel);
    let (mut rt, _, _) = runtime(fault);
    let both = combined::run_source(source.as_bytes(), fuel, &mut rt, caps()).unwrap();
    parity(&expected, &both.observation, label);
    // Composition only routes: it publishes what each family publishes alone
    // and records exactly each family's standalone refusals, in order.
    let (mut rt, _, _) = runtime(fault);
    let alone_s = stencil_host::run_source(source.as_bytes(), fuel, &mut rt, caps()).unwrap();
    let (mut rt, _, _) = runtime(fault);
    let alone_t =
        transpose_source::host::run_source(source.as_bytes(), fuel, &mut rt, caps()).unwrap();
    parity(&expected, &alone_s.observation, label);
    parity(&expected, &alone_t.observation, label);
    assert_eq!(
        both.completed,
        alone_s.completed + alone_t.completed,
        "{label}: tickets"
    );
    assert_eq!(
        tagged(&both.refusals, "stencil_"),
        alone_s.refusals,
        "{label}: stencil refusals"
    );
    assert_eq!(
        tagged(&both.refusals, "transpose_"),
        alone_t.refusals,
        "{label}: transpose refusals"
    );
    assert_eq!(
        both.refusals.len(),
        alone_s.refusals.len() + alone_t.refusals.len(),
        "{label}: no other refusals"
    );
    both
}

#[test]
fn both_orders_publish_both_families_including_empty_axes() {
    let mut tickets = 0;
    for boundary in [Boundary::ConstantZero, Boundary::Clamp] {
        for order in [Order::TransposeThenStencil, Order::StencilThenTranspose] {
            for (rows, cols) in [(0, 3), (3, 0), (1, 1), (1, 7), (7, 1), (2, 3), (4, 5)] {
                for steps in [1, 2] {
                    let label = format!("{boundary:?} {order:?} {rows}x{cols} steps {steps}");
                    let source = program(boundary, order, rows, cols, steps);
                    let run = check_composition(&source, 10_000_000, Fault::None, &label);
                    assert_eq!(run.observation.result, Ok(eval::Value::I32(7)), "{label}");
                    if rows > 0 && cols > 0 {
                        // Two transposes and one stencil reached per round.
                        assert_eq!(
                            run.completed,
                            3 * steps as usize,
                            "{label}: {:?}",
                            run.refusals
                        );
                        // Only a fuel shortfall refuses; source then runs out of fuel.
                        assert!(
                            run.refusals.iter().all(|r| matches!(
                                *r,
                                "stencil_source_fuel" | "transpose_source_fuel"
                            )),
                            "{label}: {:?}",
                            run.refusals
                        );
                    }
                    tickets += run.completed;
                }
            }
        }
    }
    eprintln!(
        "combined: {tickets} published tickets in single runs equal to standalone runs and source"
    );
}

// One test per order keeps each Debug run within the foreground budget.
#[test]
fn every_fuel_prefix_matches_source_and_standalone_runs_transpose_first() {
    fuel_prefixes(Order::TransposeThenStencil);
}

#[test]
fn every_fuel_prefix_matches_source_and_standalone_runs_stencil_first() {
    fuel_prefixes(Order::StencilThenTranspose);
}

fn fuel_prefixes(order: Order) {
    // Without the twins and comparison, so every prefix stays affordable;
    // the result folds a published element of each family into the value.
    let body = match order {
        Order::TransposeThenStencil => {
            "let t = tr(a, 2u32, 3u32); let s = step(t, 3u32, 2u32); let back = tr(s, 3u32, 2u32);"
        }
        Order::StencilThenTranspose => {
            "let s = step(a, 2u32, 3u32); let t = tr(s, 2u32, 3u32); let back = tr(t, 3u32, 2u32);"
        }
    };
    let source = format!(
        "{}\nfn main() -> i32 {{ let a: Array<f32> = [1.5,-0.0,0.25,-3.0,1048576.0,7.0]; {body} \
if back[4u32] != back[4u32] {{ return 1; }} if back[4u32] == 0.0 {{ return 2; }} return 7; }}",
        functions(Boundary::Clamp)
    );
    assert_eq!(
        reference(&source, 10_000_000).result,
        Ok(eval::Value::I32(7))
    );
    let full = reference(&source, 10_000_000).steps_used;
    let mut previous = 0;
    for fuel in 1..=full {
        let label = format!("{order:?} fuel {fuel}");
        // Every prefix against source; standalone equality on a stride
        // and at full fuel to bound the Debug run time.
        let run = if fuel % 8 == 0 || fuel == full {
            check_composition(&source, fuel, Fault::None, &label)
        } else {
            let (mut rt, _, _) = runtime(Fault::None);
            let run = combined::run_source(source.as_bytes(), fuel, &mut rt, caps()).unwrap();
            parity(&reference(&source, fuel), &run.observation, &label);
            run
        };
        // Only a fuel shortfall refuses; source then runs out of fuel.
        assert!(
            run.refusals
                .iter()
                .all(|r| matches!(*r, "stencil_source_fuel" | "transpose_source_fuel")),
            "{label}: {:?}",
            run.refusals
        );
        assert!(run.completed >= previous, "{label}");
        previous = run.completed;
    }
    assert_eq!(
        previous, 3,
        "{order:?}: full fuel publishes all three loops"
    );
    eprintln!("{order:?}: {full} fuel prefixes");
}

#[test]
fn one_family_refusing_falls_back_while_the_other_publishes() {
    for order in [Order::TransposeThenStencil, Order::StencilThenTranspose] {
        let source = program(Boundary::ConstantZero, order, 3, 4, 1);
        for (fault, refusal, published) in [
            (Fault::Quiescent(Family::Stencil), "stencil_submit", 2),
            (Fault::Quiescent(Family::Transpose), "transpose_submit", 1),
            (
                Fault::Header(Family::Stencil, 104),
                "stencil_stale_or_malformed_output",
                2,
            ),
            (
                Fault::Header(Family::Stencil, 12),
                "stencil_stale_or_malformed_output",
                2,
            ),
            (
                Fault::Header(Family::Transpose, 104),
                "transpose_stale_or_malformed_output",
                1,
            ),
            (
                Fault::Header(Family::Transpose, 40),
                "transpose_stale_or_malformed_output",
                1,
            ),
        ] {
            let label = format!("{order:?} {fault:?}");
            let run = check_composition(&source, 10_000_000, fault, &label);
            assert_eq!(run.observation.result, Ok(eval::Value::I32(7)), "{label}");
            assert_eq!(run.completed, published, "{label}: {:?}", run.refusals);
            assert!(
                run.refusals.iter().all(|r| *r == refusal),
                "{label}: {:?}",
                run.refusals
            );
        }
    }
    // A stencil short input refuses and traps exactly; the earlier transpose
    // in the same run still publishes.
    let short = format!(
        "{}\nfn main() -> i32 {{ let a: Array<f32> = {}; let t = tr(a, 2u32, 3u32); let bad: Array<f32> = {}; let s = step(bad, 2u32, 3u32); return 7; }}",
        functions(Boundary::Clamp),
        array(6),
        array(5)
    );
    let run = check_composition(&short, 10_000_000, Fault::None, "short");
    assert!(matches!(
        run.observation.result,
        Err(eval::EvalError::Trap { .. })
    ));
    assert_eq!(run.completed, 1);
    assert_eq!(run.refusals, vec!["stencil_contract_or_layout"]);
}

#[test]
fn unknown_or_pending_work_in_either_family_blocks_the_other_until_reaped() {
    for order in [Order::TransposeThenStencil, Order::StencilThenTranspose] {
        let source = program(Boundary::Clamp, order, 2, 3, 1);
        for fault in [
            Fault::Unknown(Family::Stencil),
            Fault::Unknown(Family::Transpose),
            Fault::Pending(Family::Stencil),
            Fault::Pending(Family::Transpose),
        ] {
            let label = format!("{order:?} {fault:?}");
            let (mut rt, log, release) = runtime(fault);
            let run = combined::run_source(source.as_bytes(), 10_000_000, &mut rt, caps()).unwrap();
            // Never a fallback: the run stops with the unconfirmed-work error.
            assert!(
                matches!(
                    run.observation.result,
                    Err(eval::EvalError::Tool {
                        code: "CUDA_UNAVAILABLE",
                        ..
                    })
                ),
                "{label}: {:?}",
                run.observation.result
            );
            assert_eq!(rt.quarantine_len(), 1, "{label}");
            assert!(run.refusals.is_empty(), "{label}: {:?}", run.refusals);
            let first = match order {
                Order::TransposeThenStencil => Family::Transpose,
                Order::StencilThenTranspose => Family::Stencil,
            };
            let faulted = match fault {
                Fault::Unknown(f) | Fault::Pending(f) => f,
                _ => unreachable!(),
            };
            assert_eq!(run.completed, usize::from(faulted != first), "{label}");
            let before = (
                log.borrow().stencil_prepares,
                log.borrow().transpose_prepares,
            );
            // Neither family dispatches again, through the combined path or
            // either standalone path, while the runtime holds unknown work.
            for attempt in 0..3 {
                let blocked = match attempt {
                    0 => combined::run_source(source.as_bytes(), 10_000_000, &mut rt, caps())
                        .unwrap(),
                    1 => stencil_host::run_source(source.as_bytes(), 10_000_000, &mut rt, caps())
                        .unwrap(),
                    _ => transpose_source::host::run_source(
                        source.as_bytes(),
                        10_000_000,
                        &mut rt,
                        caps(),
                    )
                    .map(|r| stencil_host::Run {
                        observation: r.observation,
                        completed: r.completed,
                        refusals: r.refusals,
                    })
                    .unwrap(),
                };
                assert!(
                    matches!(
                        blocked.observation.result,
                        Err(eval::EvalError::Tool {
                            code: "CUDA_UNAVAILABLE",
                            ..
                        })
                    ),
                    "{label} attempt {attempt}: {:?}",
                    blocked.observation.result
                );
                assert_eq!(blocked.completed, 0);
            }
            assert_eq!(
                (
                    log.borrow().stencil_prepares,
                    log.borrow().transpose_prepares
                ),
                before,
                "{label}: no dispatch"
            );
            // Reaping without confirmed completion keeps the block.
            rt.reap();
            assert_eq!(rt.quarantine_len(), 1, "{label}");
            assert!(rt.blocks_dispatch(), "{label}");
            // Confirmed quiescence and reaping restore dispatch for both
            // families on the same runtime.
            *release.borrow_mut() = true;
            rt.reap();
            assert_eq!(rt.quarantine_len(), 0, "{label}");
            let ok = combined::run_source(source.as_bytes(), 10_000_000, &mut rt, caps()).unwrap();
            assert_eq!(ok.observation.result, Ok(eval::Value::I32(7)), "{label}");
            assert_eq!(ok.completed, 3, "{label}: {:?}", ok.refusals);
            assert!(ok.refusals.is_empty(), "{label}");
        }
    }
}

#[test]
fn unconfirmed_work_in_the_last_reached_loop_is_never_replayed() {
    // The faulted family's loop is the last offloaded loop, so a replay as
    // ordinary source would complete the run with 7.
    for (body, fault) in [
        (
            "let t = tr(a, 2u32, 3u32); let s = step(t, 3u32, 2u32);",
            Family::Stencil,
        ),
        (
            "let s = step(a, 2u32, 3u32); let t = tr(s, 2u32, 3u32);",
            Family::Transpose,
        ),
    ] {
        let source = format!(
            "{}\nfn main() -> i32 {{ let a: Array<f32> = {}; {body} return 7; }}",
            functions(Boundary::Clamp),
            array(6)
        );
        assert_eq!(
            reference(&source, 10_000_000).result,
            Ok(eval::Value::I32(7))
        );
        for fault in [Fault::Unknown(fault), Fault::Pending(fault)] {
            let (mut rt, _, _) = runtime(fault);
            let run = combined::run_source(source.as_bytes(), 10_000_000, &mut rt, caps()).unwrap();
            assert!(
                matches!(
                    run.observation.result,
                    Err(eval::EvalError::Tool {
                        code: "CUDA_UNAVAILABLE",
                        ..
                    })
                ),
                "{fault:?}: {:?} completed {} refusals {:?} q {}",
                run.observation.result,
                run.completed,
                run.refusals,
                rt.quarantine_len()
            );
            assert_eq!(run.completed, 1, "{fault:?}");
            assert!(run.refusals.is_empty(), "{fault:?}: {:?}", run.refusals);
            assert_eq!(rt.quarantine_len(), 1, "{fault:?}");
        }
    }
}

#[test]
fn routed_cpu_device_keeps_family_domains_distinct() {
    let mut device = CpuDevice::default();
    assert!(device.prepare(b"xxxx").is_err());
    assert!(device.prepare(&[0u8; 200]).is_err());
    // A stencil-domain header must not be accepted as transpose work or vice versa.
    let mut stencil_header = vec![0u8; 144];
    stencil_header[..4].copy_from_slice(stencil_host::MAGIC);
    assert!(
        transpose_source::host::CpuDevice
            .prepare(&stencil_header)
            .is_err()
    );
    let mut transpose_header = vec![0u8; 144];
    transpose_header[..4].copy_from_slice(b"itr1");
    assert!(stencil_host::CpuDevice.prepare(&transpose_header).is_err());
}
