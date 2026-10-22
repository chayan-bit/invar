#![forbid(unsafe_code)]
//! DRAFT (not frozen): ordinary-source and independent CPU reference
//! acceptance for the planned M4-02 stencil runtime publication slice.
//! Uses only existing public APIs; adds no runtime path. Every expectation is
//! derived from ordinary source execution or an independent safe-Rust model.
use invar::{
    check,
    eval::{self, EvalError, Value},
    front,
    numeric_contract::F32Bits,
    stencil_source::Contract,
};
use invar_device::stencil::Boundary;

/// The step function exactly as the reviewed recognizer accepts it.
fn step_source(boundary: Boundary) -> String {
    let center = "xs[r*cols+c]";
    let edge = |name: &str| match boundary {
        Boundary::Clamp => format!(" else {{ {name} = {center}; }}"),
        _ => String::new(),
    };
    format!(
        "fn step(xs: read Array<f32>, rows: u32, cols: u32) -> Array<f32> {{ var out = filled_f32(rows*cols, 0.0); \
for r in 0u32..rows {{ for c in 0u32..cols {{ \
var n = 0.0; if r > 0u32 {{ n = xs[(r-1u32)*cols+c]; }}{en} \
var s = 0.0; if r+1u32 < rows {{ s = xs[(r+1u32)*cols+c]; }}{es} \
var w = 0.0; if c > 0u32 {{ w = xs[r*cols+(c-1u32)]; }}{ew} \
var e = 0.0; if c+1u32 < cols {{ e = xs[r*cols+(c+1u32)]; }}{ee} \
out[r*cols+c] = ((({center}+n)+s)+w)+e; }} }} return out; }}",
        en = edge("n"),
        es = edge("s"),
        ew = edge("w"),
        ee = edge("e"),
    )
}

fn checked(source: &str) -> check::CheckedProgram {
    let c = check::check(front::parse(source.as_bytes()).unwrap(), false).unwrap();
    invar::ownership::check(&c).unwrap();
    c
}

/// Wrapper: build the input owner from exact bits, run `steps` ping-pong steps.
fn program(boundary: Boundary, rows: u32, cols: u32, input: &[u32], steps: u32) -> String {
    let lits = if input.is_empty() {
        "filled_f32(0u32, 0.0)".to_owned()
    } else {
        // Finite values print exactly; non-finite values use source expressions
        // (the only NaN source can build is the canonical one).
        format!(
            "[{}]",
            input
                .iter()
                .map(|b| literal(*b))
                .collect::<Vec<_>>()
                .join(",")
        )
    };
    format!(
        "{}\nfn run() -> Array<f32> {{ var cur: Array<f32> = {lits}; for t in 0u32..{steps}u32 {{ let next = step(cur, {rows}u32, {cols}u32); cur = move next; }} return cur; }}\nfn main() -> i32 {{ return 0; }}",
        step_source(boundary)
    )
}

fn literal(bits: u32) -> String {
    let x = f32::from_bits(bits);
    if x.is_nan() {
        assert_eq!(
            bits,
            F32Bits::NAN.to_bits(),
            "source can only build the canonical NaN"
        );
        "0.0/0.0".into()
    } else if x.is_infinite() {
        if x > 0.0 {
            "1.0/0.0".into()
        } else {
            "-1.0/0.0".into()
        }
    } else {
        // Plain decimal (no exponent), shortest round-trip digits.
        let text = format!("{x}");
        if text.contains('.') {
            text
        } else {
            format!("{text}.0")
        }
    }
}

/// Independent strict-FP32 model: ((center + n) + s) + w) + e, each a single
/// strict binary32 rounding (`F32Bits::add`: canonical NaN, no host payloads).
fn model(boundary: Boundary, rows: usize, cols: usize, input: &[u32]) -> Vec<u32> {
    let at = |r: usize, c: usize| F32Bits::from_bits(input[r * cols + c]);
    let zero = F32Bits::from_bits(0);
    let mut out = vec![0u32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let center = at(r, c);
            let pick = |inside: bool, v: F32Bits| match (inside, boundary) {
                (true, _) => v,
                (false, Boundary::Clamp) => center,
                (false, _) => zero,
            };
            let n = pick(r > 0, if r > 0 { at(r - 1, c) } else { zero });
            let s = pick(r + 1 < rows, if r + 1 < rows { at(r + 1, c) } else { zero });
            let w = pick(c > 0, if c > 0 { at(r, c - 1) } else { zero });
            let e = pick(c + 1 < cols, if c + 1 < cols { at(r, c + 1) } else { zero });
            out[r * cols + c] = center.add(n).add(s).add(w).add(e).to_bits();
        }
    }
    out
}

fn observe(source: &str, fuel: u64) -> eval::Observation {
    eval::named::observe_owned_numeric_named(&checked(source), "run", &[], fuel)
}

fn words(o: &eval::Observation) -> Vec<u32> {
    match &o.result {
        Ok(Value::ArrayF32(v)) => v.borrow().iter().map(|x| x.to_bits()).collect(),
        x => panic!("{x:?}"),
    }
}

/// Exactly representable, sign- and zero-sensitive values.
fn values(n: usize) -> Vec<u32> {
    const POOL: [f32; 9] = [1.5, -0.0, 0.25, -3.0, 1048576.0, 0.0, -0.125, 7.0, 2.5];
    (0..n)
        .map(|i| POOL[(i * 7 + 3) % POOL.len()].to_bits())
        .collect()
}

#[test]
fn draft_fixture_is_the_recognized_stencil_form() {
    for boundary in [Boundary::ConstantZero, Boundary::Clamp] {
        let c = checked(&program(boundary, 2, 3, &values(6), 1));
        let contract = Contract::from_checked(&c, "step").expect("recognized");
        assert_eq!(contract.boundary(), boundary);
        assert_eq!(
            contract.dimensions(),
            &["rows".to_owned(), "cols".to_owned()]
        );
    }
}

const NAN: u32 = 0x7fc0_0000;
const INF: u32 = 0x7f80_0000;
const NEG_INF: u32 = 0xff80_0000;
const NEG_ZERO: u32 = 0x8000_0000;

#[test]
fn draft_source_equals_independent_model_across_shapes_steps_and_boundaries() {
    let shapes = [
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
    ];
    let mut inputs: Vec<(usize, usize, Vec<u32>)> =
        shapes.iter().map(|&(r, c)| (r, c, values(r * c))).collect();
    // Non-finite and signed-zero propagation, including inf + -inf = NaN.
    inputs.push((2, 2, vec![NEG_ZERO, NEG_ZERO, NEG_ZERO, NEG_ZERO]));
    inputs.push((
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
    ));
    inputs.push((
        3,
        1,
        vec![f32::MAX.to_bits(), f32::MAX.to_bits(), 0x0000_0001],
    ));
    let mut compared = 0;
    for boundary in [Boundary::ConstantZero, Boundary::Clamp] {
        for (rows, cols, input) in &inputs {
            for steps in [1u32, 2, 3] {
                let o = observe(
                    &program(boundary, *rows as u32, *cols as u32, input, steps),
                    10_000_000,
                );
                let mut expect = input.clone();
                for _ in 0..steps {
                    expect = model(boundary, *rows, *cols, &expect);
                }
                assert_eq!(
                    words(&o),
                    expect,
                    "{boundary:?} {rows}x{cols} steps {steps}"
                );
                // Ping-pong: each step releases exactly the replaced slice.
                let released: Vec<_> = o
                    .cleanup_trace
                    .iter()
                    .map(|c| (c.name.as_str(), c.length))
                    .collect();
                assert_eq!(
                    released,
                    vec![("cur", rows * cols); steps as usize],
                    "{boundary:?} {rows}x{cols}"
                );
                compared += 1;
            }
        }
    }
    eprintln!("draft: {compared} source runs equal to the independent strict-FP32 model");
}

/// Measured source steps of one `step` call relative to the 0x0 call.
fn loop_delta(boundary: Boundary, rows: u32, cols: u32) -> u64 {
    let one = |r: u32, c: u32| {
        let input = values((r * c) as usize);
        let total = observe(&program(boundary, r, c, &input, 1), 10_000_000).steps_used;
        let none = observe(&program(boundary, r, c, &input, 0), 10_000_000).steps_used;
        total - none
    };
    one(rows, cols) - one(0, 0)
}

/// Proposed closed form for freezing (measured at e233951): 4 per row,
/// 50 per cell, 10 per in-range neighbour load, and under clamp 8 per edge copy.
fn proposed_delta(boundary: Boundary, rows: u64, cols: u64) -> u64 {
    let cells = rows * cols;
    let loads = 2 * rows.saturating_sub(1) * cols + 2 * rows * cols.saturating_sub(1);
    let copies = if boundary == Boundary::Clamp {
        4 * cells - loads
    } else {
        0
    };
    4 * rows + 50 * cells + 10 * loads + 8 * copies
}

#[test]
fn draft_loop_fuel_closed_form_matches_ordinary_source() {
    let mut shapes: Vec<(u32, u32)> = (0..7).flat_map(|r| (0..7).map(move |c| (r, c))).collect();
    shapes.extend([(1, 40), (40, 1), (9, 13), (17, 3), (2, 61)]);
    for boundary in [Boundary::ConstantZero, Boundary::Clamp] {
        for &(rows, cols) in &shapes {
            assert_eq!(
                loop_delta(boundary, rows, cols),
                proposed_delta(boundary, rows.into(), cols.into()),
                "{boundary:?} {rows}x{cols}"
            );
        }
    }
    eprintln!(
        "draft: closed-form loop fuel matches source over {} shapes per boundary",
        shapes.len()
    );
}

#[test]
fn draft_ordinary_failure_prefixes_are_the_publication_baseline() {
    // Every prefix below completion stops on the fuel limit at exactly `fuel`.
    let source = program(Boundary::Clamp, 2, 3, &values(6), 2);
    let done = observe(&source, 10_000_000);
    for fuel in 1..done.steps_used {
        let o = observe(&source, fuel);
        assert_eq!(
            o.result,
            Err(EvalError::EvaluationLimit {
                fuel_limit: fuel,
                steps_used: fuel
            })
        );
    }
    assert_eq!(observe(&source, done.steps_used).result, done.result);
    // Short input (6 cells, 5 elements) traps at the original load span; no publication may hide it.
    for boundary in [Boundary::ConstantZero, Boundary::Clamp] {
        let short = program(boundary, 2, 3, &values(5), 1);
        let o = observe(&short, 10_000_000);
        let Err(EvalError::Trap { span, .. }) = &o.result else {
            panic!("{o:?}")
        };
        assert!(
            short[span.start..span.end].starts_with("xs["),
            "{boundary:?}"
        );
        assert!(o.cleanup_trace.is_empty(), "traps record no cleanup");
    }
    // rows*cols overflow traps at the allocation multiply, before the loop.
    let overflow = program(Boundary::ConstantZero, 65536, 65536, &[], 1);
    let o = observe(&overflow, 10_000_000);
    assert!(matches!(o.result, Err(EvalError::Trap { .. })), "{o:?}");
}
