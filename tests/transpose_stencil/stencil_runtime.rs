#![forbid(unsafe_code)]
//! M4-02 bounded CPU stencil runtime publication (frozen slice). Every run is
//! compared with the independent AST evaluator on result, steps and cleanup;
//! published words are additionally checked against native f32 arithmetic
//! with explicit NaN canonicalization and hand-derived special values.
use invar::{
    check, eval, front,
    numeric_contract::F32Bits,
    stencil_source::{
        Contract,
        host::{self, CpuDevice, CpuWork},
    },
    transpose_source,
};
use invar_cuda::{Device, Error, Runtime, SubmitError};
use invar_device::{Capabilities, stencil::Boundary};
use std::{cell::RefCell, rc::Rc};

const NAN: u32 = 0x7fc0_0000;
const INF: u32 = 0x7f80_0000;
const NEG_INF: u32 = 0xff80_0000;
const NEG_ZERO: u32 = 0x8000_0000;

fn caps() -> Capabilities {
    Capabilities::cuda_12_9_sm89()
}

/// The recognized step, and `step_ref`: the same body behind one extra
/// statement, so it is never recognized and always runs as ordinary source.
fn functions(boundary: Boundary) -> String {
    let center = "xs[r*cols+c]";
    let edge = |name: &str| match boundary {
        Boundary::Clamp => format!(" else {{ {name} = {center}; }}"),
        _ => String::new(),
    };
    let body = format!(
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
    format!(
        "fn step(xs: read Array<f32>, rows: u32, cols: u32) -> Array<f32> {{ var out = filled_f32(rows*cols, 0.0); {body} }}
fn step_ref(xs: read Array<f32>, rows: u32, cols: u32) -> Array<f32> {{ let unused = 0u32; var out = filled_f32(rows*cols, 0.0); {body} }}
fn negative(x: f32) -> Bool {{ return 1.0 / x < 0.0; }}
fn same(a: f32, b: f32) -> Bool {{ if a != a {{ return b != b; }} if a == 0.0 {{ return b == 0.0 and negative(a) == negative(b); }} return a == b; }}"
    )
}

fn literal(bits: u32) -> String {
    let x = f32::from_bits(bits);
    if x.is_nan() {
        assert_eq!(bits, NAN, "source builds only the canonical NaN");
        "0.0/0.0".into()
    } else if x.is_infinite() {
        if x > 0.0 {
            "1.0/0.0".into()
        } else {
            "-1.0/0.0".into()
        }
    } else {
        let text = format!("{x}");
        if text.contains('.') {
            text
        } else {
            format!("{text}.0")
        }
    }
}

fn array(input: &[u32]) -> String {
    if input.is_empty() {
        "filled_f32(0u32, 0.0)".into()
    } else {
        format!(
            "[{}]",
            input
                .iter()
                .map(|b| literal(*b))
                .collect::<Vec<_>>()
                .join(",")
        )
    }
}

/// `main` runs `steps` ping-pong steps through the recognized function and its
/// unrecognized twin, then returns 7 only if every element matches exactly.
fn program(boundary: Boundary, rows: u32, cols: u32, input: &[u32], steps: u32) -> String {
    let lit = array(input);
    format!(
        "{}
fn main() -> i32 {{ var cur: Array<f32> = {lit}; var other: Array<f32> = {lit};
for t in 0u32..{steps}u32 {{ let next = step(cur, {rows}u32, {cols}u32); cur = move next; let again = step_ref(other, {rows}u32, {cols}u32); other = move again; }}
if len(cur) != len(other) {{ return 2; }}
for i in 0u32..len(cur) {{ if same(cur[i], other[i]) == false {{ return 1; }} }}
return 7; }}",
        functions(boundary)
    )
}

/// `steps` ping-pong steps through the recognized function only; used where
/// AST-evaluator parity is the check (every fuel prefix).
fn plain(boundary: Boundary, rows: u32, cols: u32, input: &[u32], steps: u32) -> String {
    format!(
        "{}\nfn main() -> i32 {{ var cur: Array<f32> = {}; for t in 0u32..{steps}u32 {{ let next = step(cur, {rows}u32, {cols}u32); cur = move next; }} return 7; }}",
        functions(boundary),
        array(input)
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
enum Fault {
    None,
    Prepare,
    SubmitQuiescent,
    SubmitUnknown,
    Pending,
    Read,
    Header(usize),
    Truncate,
    Payload,
    PayloadWithDigest,
    Stale,
}

#[derive(Default)]
struct Log {
    prepares: usize,
    reads: usize,
    responses: Vec<Vec<u8>>,
    previous: Option<Vec<u8>>,
}

struct Probe {
    cpu: CpuDevice,
    fault: Fault,
    log: Rc<RefCell<Log>>,
    release: Rc<RefCell<bool>>,
}
impl Device for Probe {
    type Work = CpuWork;
    fn prepare(&mut self, b: &[u8]) -> Result<CpuWork, Error> {
        self.log.borrow_mut().prepares += 1;
        if self.fault == Fault::Prepare {
            return Err(Error::OutOfMemory);
        }
        self.cpu.prepare(b)
    }
    fn submit(&mut self, w: &mut CpuWork) -> Result<(), SubmitError> {
        match self.fault {
            Fault::SubmitQuiescent => Err(SubmitError::FailedQuiescent(Error::Driver("q".into()))),
            Fault::SubmitUnknown => Err(SubmitError::InFlightUnknown(Error::Timeout)),
            _ => self.cpu.submit(w),
        }
    }
    fn complete(&mut self, w: &mut CpuWork) -> Result<bool, Error> {
        if *self.release.borrow() {
            Ok(true)
        } else if matches!(self.fault, Fault::Pending | Fault::SubmitUnknown) {
            Ok(false)
        } else {
            self.cpu.complete(w)
        }
    }
    fn read(&mut self, w: &mut CpuWork) -> Result<Vec<u8>, Error> {
        let mut log = self.log.borrow_mut();
        log.reads += 1;
        if self.fault == Fault::Read {
            return Err(Error::Driver("read".into()));
        }
        let fresh = self.cpu.read(w)?;
        log.responses.push(fresh.clone());
        let previous = log.previous.replace(fresh.clone());
        let mut b = fresh;
        let n = b.len();
        match self.fault {
            Fault::Header(i) => b[i] ^= 1,
            Fault::Truncate => b.truncate(n - 4),
            Fault::Payload => b[host::HEADER] ^= 1,
            Fault::PayloadWithDigest => {
                use sha2::{Digest, Sha256};
                b[host::HEADER] ^= 1;
                let words = n - host::HEADER - host::TRAILER;
                let digest = Sha256::digest(&b[host::HEADER..host::HEADER + words]);
                b[host::HEADER + words..].copy_from_slice(&digest);
            }
            Fault::Stale => {
                if let Some(p) = previous {
                    b = p;
                }
            }
            _ => {}
        }
        Ok(b)
    }
}

fn runtime(fault: Fault) -> (Runtime<Probe>, Rc<RefCell<Log>>, Rc<RefCell<bool>>) {
    let log = Rc::new(RefCell::new(Log::default()));
    let release = Rc::new(RefCell::new(false));
    let probe = Probe {
        cpu: CpuDevice,
        fault,
        log: log.clone(),
        release: release.clone(),
    };
    (Runtime::new(probe, 4), log, release)
}

fn values(n: usize) -> Vec<u32> {
    const POOL: [f32; 9] = [1.5, -0.0, 0.25, -3.0, 1048576.0, 0.0, -0.125, 7.0, 2.5];
    (0..n)
        .map(|i| POOL[(i * 7 + 3) % POOL.len()].to_bits())
        .collect()
}

const SHAPES: [(u32, u32); 13] = [
    (0, 0),
    (0, 5),
    (4, 0),
    (1, 1),
    (1, 7),
    (7, 1),
    (2, 2),
    (3, 5),
    (5, 3),
    (4, 33),
    (33, 4),
    (2, 61),
    (11, 13),
];

#[test]
fn reached_stencil_publication_matches_source_for_shapes_boundaries_and_steps() {
    let mut published = 0;
    for boundary in [Boundary::ConstantZero, Boundary::Clamp] {
        for (rows, cols) in SHAPES {
            let input = values((rows * cols) as usize);
            for steps in [1u32, 2, 3] {
                let label = format!("{boundary:?} {rows}x{cols} steps {steps}");
                let source = program(boundary, rows, cols, &input, steps);
                let (mut rt, log, _) = runtime(Fault::None);
                let got = host::run_source(source.as_bytes(), 10_000_000, &mut rt, caps()).unwrap();
                parity(&reference(&source, 10_000_000), &got.observation, &label);
                assert_eq!(got.observation.result, Ok(eval::Value::I32(7)), "{label}");
                // One fenced ticket per reached source loop. With zero rows the
                // loop body is never reached: no ticket, no refusal, no work.
                let tickets = if rows > 0 { steps as usize } else { 0 };
                assert_eq!(got.completed, tickets, "{label}: {:?}", got.refusals);
                assert_eq!(log.borrow().prepares, tickets, "{label}");
                assert!(got.refusals.is_empty(), "{label}: {:?}", got.refusals);
                assert_eq!(rt.quarantine_len(), 0);
                // The CPU device and source share F32Bits arithmetic, so parity
                // alone cannot catch an arithmetic defect. Every ticket must also
                // equal native binary32 additions applied to the previous slice.
                let responses = &log.borrow().responses;
                assert_eq!(responses.len(), tickets, "{label}");
                let mut slice = input.clone();
                for (ticket, response) in responses.iter().enumerate() {
                    let want = native(boundary, rows as usize, cols as usize, &slice);
                    assert_eq!(published_words(response), want, "{label} ticket {ticket}");
                    slice = want;
                }
                published += got.completed;
            }
        }
    }
    eprintln!("stencil runtime: {published} published tickets equal to ordinary source");
}

/// Independent checker: native binary32 additions in source order, with
/// explicit NaN canonicalization (host NaN payloads are not part of strict-f32).
fn native(boundary: Boundary, rows: usize, cols: usize, input: &[u32]) -> Vec<u32> {
    let canonical = |x: f32| if x.is_nan() { NAN } else { x.to_bits() };
    let at = |i: usize| f32::from_bits(input[i]);
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let i = r * cols + c;
            let center = at(i);
            let edge = if boundary == Boundary::Clamp {
                center
            } else {
                0.0
            };
            let n = if r > 0 { at(i - cols) } else { edge };
            let s = if r + 1 < rows { at(i + cols) } else { edge };
            let w = if c > 0 { at(i - 1) } else { edge };
            let e = if c + 1 < cols { at(i + 1) } else { edge };
            // Canonicalize after each rounding: a NaN operand yields the canonical NaN.
            let mut acc = f32::from_bits(canonical(center));
            for x in [n, s, w, e] {
                acc = f32::from_bits(canonical(acc + x));
            }
            out.push(canonical(acc));
        }
    }
    out
}

fn published_words(response: &[u8]) -> Vec<u32> {
    response[host::HEADER..response.len() - host::TRAILER]
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect()
}

#[test]
fn special_values_match_native_checker_and_hand_derived_bits() {
    let cases: Vec<(Boundary, u32, u32, Vec<u32>)> = vec![
        (
            Boundary::ConstantZero,
            2,
            3,
            vec![
                INF,
                NEG_INF,
                1.0f32.to_bits(),
                NAN,
                NEG_ZERO,
                2.0f32.to_bits(),
            ],
        ),
        (
            Boundary::Clamp,
            2,
            3,
            vec![
                INF,
                NEG_INF,
                1.0f32.to_bits(),
                NAN,
                NEG_ZERO,
                2.0f32.to_bits(),
            ],
        ),
        (Boundary::Clamp, 2, 2, vec![NEG_ZERO; 4]),
        (Boundary::ConstantZero, 2, 2, vec![NEG_ZERO; 4]),
        (
            Boundary::ConstantZero,
            3,
            1,
            vec![f32::MAX.to_bits(), f32::MAX.to_bits(), 1],
        ),
        (
            Boundary::ConstantZero,
            1,
            2,
            vec![16_777_216f32.to_bits(), 1.0f32.to_bits()],
        ),
        (
            Boundary::Clamp,
            1,
            3,
            vec![0.1f32.to_bits(), 0.2f32.to_bits(), 0.3f32.to_bits()],
        ),
    ];
    for (boundary, rows, cols, input) in cases {
        let label = format!("{boundary:?} {rows}x{cols} {input:x?}");
        let source = program(boundary, rows, cols, &input, 1);
        let (mut rt, log, _) = runtime(Fault::None);
        let got = host::run_source(source.as_bytes(), 10_000_000, &mut rt, caps()).unwrap();
        parity(&reference(&source, 10_000_000), &got.observation, &label);
        assert_eq!(got.observation.result, Ok(eval::Value::I32(7)), "{label}");
        let words = published_words(&log.borrow().responses[0]);
        assert_eq!(
            words,
            native(boundary, rows as usize, cols as usize, &input),
            "{label}"
        );
    }
    // Hand-derived: -0 + -0 + ... under clamp stays -0; under constant_zero the
    // first edge adds +0, giving +0 (round-to-nearest-even sign rule).
    let hand = |boundary, rows, cols, input: Vec<u32>| {
        let source = program(boundary, rows, cols, &input, 1);
        let (mut rt, log, _) = runtime(Fault::None);
        host::run_source(source.as_bytes(), 10_000_000, &mut rt, caps()).unwrap();
        published_words(&log.borrow().responses[0])
    };
    assert_eq!(hand(Boundary::Clamp, 1, 1, vec![NEG_ZERO]), vec![NEG_ZERO]);
    assert_eq!(hand(Boundary::ConstantZero, 1, 1, vec![NEG_ZERO]), vec![0]);
    // inf + 0 + 0 + 0 + -inf = NaN (canonical), both cells.
    assert_eq!(
        hand(Boundary::ConstantZero, 1, 2, vec![INF, NEG_INF]),
        vec![NAN, NAN]
    );
    // 2^24 + 1 ties to even: 16777216 (0x4b800000) in both orders.
    assert_eq!(
        hand(Boundary::ConstantZero, 1, 2, vec![0x4b80_0000, 0x3f80_0000]),
        vec![0x4b80_0000, 0x4b80_0000]
    );
    // Operation order: centre 2^24 with four 1.0 neighbours. In source order
    // every +1 ties back to 2^24; any reassociation that adds two 1s first
    // yields 2^24 + 4 (0x4b80_0002). Hand-derived, order-discriminating.
    let one = 0x3f80_0000;
    let order = hand(
        Boundary::ConstantZero,
        3,
        3,
        vec![0, one, 0, one, 0x4b80_0000, one, 0, one, 0],
    );
    assert_eq!(order[4], 0x4b80_0000, "centre keeps source order");
    // MAX + MAX overflows to +inf; +inf + 1ulp-subnormal stays +inf.
    assert_eq!(
        hand(Boundary::ConstantZero, 2, 1, vec![0x7f7f_ffff, 0x7f7f_ffff]),
        vec![INF, INF]
    );
}

#[test]
fn every_fuel_prefix_matches_source_and_publishes_only_with_enough_fuel() {
    for boundary in [Boundary::ConstantZero, Boundary::Clamp] {
        for (rows, cols, steps) in [(2u32, 3u32, 2u32), (1, 1, 3), (3, 0, 2)] {
            let source = plain(boundary, rows, cols, &values((rows * cols) as usize), steps);
            let full = reference(&source, 10_000_000).steps_used;
            let mut published_any = false;
            let mut previous = 0;
            for fuel in 1..=full {
                let (mut rt, _, _) = runtime(Fault::None);
                let got = host::run_source(source.as_bytes(), fuel, &mut rt, caps()).unwrap();
                parity(
                    &reference(&source, fuel),
                    &got.observation,
                    &format!("{boundary:?} {rows}x{cols} fuel {fuel}"),
                );
                published_any |= got.completed > 0;
                // More fuel never publishes fewer reached loops.
                assert!(
                    got.completed >= previous,
                    "{boundary:?} {rows}x{cols} fuel {fuel}"
                );
                previous = got.completed;
                if got.completed < steps as usize && fuel == full {
                    panic!("full fuel must publish every step");
                }
            }
            assert!(published_any);
        }
    }
}

#[test]
fn loop_fuel_derived_from_checked_structure_matches_ordinary_steps() {
    // Source steps of one `step` call with r x c, relative to the 0 x 0 call,
    // must equal the contract's derived loop fuel (which is 0 at 0 rows).
    for boundary in [Boundary::ConstantZero, Boundary::Clamp] {
        let one = |rows: u32, cols: u32| {
            let input = values((rows * cols) as usize);
            let with = |steps| {
                let src = format!(
                    "{}\nfn main() -> i32 {{ var cur: Array<f32> = {}; for t in 0u32..{steps}u32 {{ let next = step(cur, {rows}u32, {cols}u32); cur = move next; }} return 0; }}",
                    functions(boundary),
                    array(&input)
                );
                reference(&src, 10_000_000).steps_used
            };
            with(1) - with(0)
        };
        let checked = check::check(
            front::parse(program(boundary, 1, 1, &[0], 1).as_bytes()).unwrap(),
            true,
        )
        .unwrap();
        let contract = Contract::from_checked(&checked, "step").unwrap();
        let base = one(0, 0);
        let mut shapes: Vec<(u32, u32)> =
            (0..7).flat_map(|r| (0..7).map(move |c| (r, c))).collect();
        shapes.extend([(1, 40), (40, 1), (9, 13), (17, 3), (2, 61)]);
        for (rows, cols) in shapes {
            assert_eq!(
                one(rows, cols) - base,
                contract.loop_fuel([rows, cols]).unwrap(),
                "{boundary:?} {rows}x{cols}"
            );
        }
        assert_eq!(
            contract.loop_fuel([u32::MAX, u32::MAX]),
            None,
            "checked overflow"
        );
    }
}

#[test]
fn short_input_overflow_and_limits_fall_back_to_exact_source_failures() {
    for boundary in [Boundary::ConstantZero, Boundary::Clamp] {
        // Short input: refused; ordinary source traps at the original load span.
        let source = program(boundary, 2, 3, &values(5), 1);
        let (mut rt, log, _) = runtime(Fault::None);
        let got = host::run_source(source.as_bytes(), 10_000_000, &mut rt, caps()).unwrap();
        parity(&reference(&source, 10_000_000), &got.observation, "short");
        let Err(eval::EvalError::Trap { span, .. }) = &got.observation.result else {
            panic!("{:?}", got.observation)
        };
        assert!(source[span.start..span.end].starts_with("xs["));
        assert_eq!(got.refusals, vec!["stencil_contract_or_layout"]);
        assert_eq!((got.completed, log.borrow().prepares), (0, 0));
        // rows*cols overflow and over-limit allocation fail before the loop.
        for (rows, cols) in [(65536u32, 65536u32), (1001, 1000)] {
            let source = program(boundary, rows, cols, &[], 1);
            let (mut rt, log, _) = runtime(Fault::None);
            let got = host::run_source(source.as_bytes(), 10_000_000, &mut rt, caps()).unwrap();
            parity(
                &reference(&source, 10_000_000),
                &got.observation,
                "overflow/limit",
            );
            assert!(got.observation.result.is_err());
            assert_eq!((got.completed, log.borrow().prepares), (0, 0));
        }
    }
}

#[test]
fn quiescent_failures_and_tampered_responses_refuse_without_debit() {
    let source = program(Boundary::Clamp, 3, 4, &values(12), 2);
    let expected = reference(&source, 10_000_000);
    let cases = [
        (Fault::Prepare, "stencil_prepare"),
        (Fault::SubmitQuiescent, "stencil_submit"),
        (Fault::Read, "stencil_read"),
        (Fault::Header(0), "stencil_stale_or_malformed_output"), // magic
        (Fault::Header(12), "stencil_stale_or_malformed_output"), // boundary policy
        (Fault::Header(40), "stencil_stale_or_malformed_output"), // artifact key
        (Fault::Header(104), "stencil_stale_or_malformed_output"), // generation
        (Fault::Header(112), "stencil_stale_or_malformed_output"), // request digest
        (Fault::Truncate, "stencil_stale_or_malformed_output"),
        (Fault::Payload, "stencil_stale_or_malformed_output"),
        (Fault::Stale, "stencil_stale_or_malformed_output"),
    ];
    for (fault, refusal) in cases {
        let (mut rt, _, _) = runtime(fault);
        let got = host::run_source(source.as_bytes(), 10_000_000, &mut rt, caps()).unwrap();
        parity(&expected, &got.observation, &format!("{fault:?}"));
        let expected_published = if fault == Fault::Stale { 1 } else { 0 };
        assert_eq!(
            got.completed, expected_published,
            "{fault:?}: {:?}",
            got.refusals
        );
        assert!(
            got.refusals.contains(&refusal),
            "{fault:?}: {:?}",
            got.refusals
        );
        assert_eq!(rt.quarantine_len(), 0, "{fault:?}");
    }
    // A device that recomputes the digest over wrong words is inside the trust
    // boundary; the in-language exact comparison still exposes it.
    let (mut rt, _, _) = runtime(Fault::PayloadWithDigest);
    let got = host::run_source(source.as_bytes(), 10_000_000, &mut rt, caps()).unwrap();
    assert_eq!(got.observation.result, Ok(eval::Value::I32(1)));
}

#[test]
fn unknown_or_pending_work_blocks_dispatch_and_replay_until_reaped() {
    let source = program(Boundary::ConstantZero, 2, 2, &values(4), 1);
    for fault in [Fault::SubmitUnknown, Fault::Pending] {
        let (mut rt, log, release) = runtime(fault);
        let got = host::run_source(source.as_bytes(), 10_000_000, &mut rt, caps()).unwrap();
        assert!(
            matches!(
                got.observation.result,
                Err(eval::EvalError::Tool {
                    code: "CUDA_UNAVAILABLE",
                    ..
                })
            ),
            "{fault:?}: {:?}",
            got.observation.result
        );
        assert_eq!(
            (got.completed, log.borrow().reads, rt.quarantine_len()),
            (0, 0, 1),
            "{fault:?}"
        );
        // No replay and no new dispatch while unknown work is quarantined.
        let blocked = host::run_source(source.as_bytes(), 10_000_000, &mut rt, caps()).unwrap();
        assert!(matches!(
            blocked.observation.result,
            Err(eval::EvalError::Tool {
                code: "CUDA_UNAVAILABLE",
                ..
            })
        ));
        assert_eq!(log.borrow().prepares, 1, "{fault:?}");
        *release.borrow_mut() = true;
        rt.reap();
        assert_eq!(rt.quarantine_len(), 0);
    }
}

#[test]
fn staging_bound_and_unrecognized_functions_use_ordinary_source() {
    let source = program(Boundary::ConstantZero, 4, 4, &values(16), 1);
    let (mut rt, log, _) = runtime(Fault::None);
    let got =
        host::run_source_bounded(source.as_bytes(), 10_000_000, &mut rt, caps(), 200).unwrap();
    parity(&reference(&source, 10_000_000), &got.observation, "staging");
    assert_eq!(got.refusals, vec!["stencil_staging_limit"]);
    assert_eq!((got.completed, log.borrow().prepares), (0, 0));
    // Only step_ref present (one extra statement): no contract, no dispatch.
    let only_twin = program(Boundary::Clamp, 3, 3, &values(9), 1)
        .replace("let next = step(", "let next = step_ref(");
    let (mut rt, log, _) = runtime(Fault::None);
    let got = host::run_source(only_twin.as_bytes(), 10_000_000, &mut rt, caps()).unwrap();
    parity(&reference(&only_twin, 10_000_000), &got.observation, "twin");
    assert_eq!((got.completed, log.borrow().prepares), (0, 0));
}

/// Routes transpose and stencil packets to their own CPU devices through one
/// runtime; transpose submissions can be forced unknown.
struct Composite {
    transpose: transpose_source::host::CpuDevice,
    stencil: CpuDevice,
    transpose_unknown: bool,
    stencil_prepares: Rc<RefCell<usize>>,
}
enum Work {
    T(transpose_source::host::CpuWork),
    S(CpuWork),
}
impl Device for Composite {
    type Work = Work;
    fn prepare(&mut self, b: &[u8]) -> Result<Work, Error> {
        if b.starts_with(host::MAGIC) {
            *self.stencil_prepares.borrow_mut() += 1;
            self.stencil.prepare(b).map(Work::S)
        } else {
            self.transpose.prepare(b).map(Work::T)
        }
    }
    fn submit(&mut self, w: &mut Work) -> Result<(), SubmitError> {
        match w {
            Work::T(_) if self.transpose_unknown => {
                Err(SubmitError::InFlightUnknown(Error::Timeout))
            }
            Work::T(w) => self.transpose.submit(w),
            Work::S(w) => self.stencil.submit(w),
        }
    }
    fn complete(&mut self, w: &mut Work) -> Result<bool, Error> {
        match w {
            Work::T(_) if self.transpose_unknown => Ok(false),
            Work::T(w) => self.transpose.complete(w),
            Work::S(w) => self.stencil.complete(w),
        }
    }
    fn read(&mut self, w: &mut Work) -> Result<Vec<u8>, Error> {
        match w {
            Work::T(w) => self.transpose.read(w),
            Work::S(w) => self.stencil.read(w),
        }
    }
}

#[test]
fn transpose_and_stencil_share_one_quarantine_and_domains_are_distinct() {
    let transpose = "fn transpose(a:read Array<u32>,rows:u32,cols:u32)->Array<u32>{var b=filled_u32(rows*cols,0u32);for r in 0u32..rows{for c in 0u32..cols{b[c*rows+r]=a[r*cols+c];}}return b;} fn main()->i32{let a=filled_u32(6u32,1u32);let b=transpose(a,2u32,3u32);return 7;}";
    let stencil = program(Boundary::Clamp, 2, 2, &values(4), 1);
    let prepares = Rc::new(RefCell::new(0));
    let mut rt = Runtime::new(
        Composite {
            transpose: transpose_source::host::CpuDevice,
            stencil: CpuDevice,
            transpose_unknown: true,
            stencil_prepares: prepares.clone(),
        },
        4,
    );
    let t =
        transpose_source::host::run_source(transpose.as_bytes(), 100_000, &mut rt, caps()).unwrap();
    assert!(t.observation.result.is_err());
    assert_eq!(rt.quarantine_len(), 1);
    // The separate stencil path cannot bypass the transpose unknown operation.
    let s = host::run_source(stencil.as_bytes(), 10_000_000, &mut rt, caps()).unwrap();
    assert!(matches!(
        s.observation.result,
        Err(eval::EvalError::Tool {
            code: "CUDA_UNAVAILABLE",
            ..
        })
    ));
    assert_eq!(*prepares.borrow(), 0);
    // Distinct magic: each CPU device rejects the other family's packet.
    let (mut probe_rt, log, _) = runtime(Fault::None);
    host::run_source(stencil.as_bytes(), 10_000_000, &mut probe_rt, caps()).unwrap();
    let response = log.borrow().responses[0].clone();
    assert!(
        transpose_source::host::CpuDevice
            .prepare(&response[..host::HEADER])
            .is_err()
    );
    assert_eq!(&response[..4], host::MAGIC);
    assert_ne!(host::MAGIC, b"itr1");
    let _ = F32Bits::from_bits(0);
}
