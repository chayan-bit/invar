//! Issue #262: pairs of 32-bit scalars. Literals, positional fields,
//! destructuring, parameters and results agree in value, trap and fuel on the
//! reference, MIR, cpu-parallel and native engines; GNET, the device route
//! and the export boundary refuse them.
use crate::loop_control::{agree, checked, parallel};
use crate::surface_sugar::first_code;
use invar::exports::{ExportEngine, ExportError, compile_exports};
use invar::general_net::{Limits, Net, oracle};
use invar::{eval, mir};

const SWAP: &str = "fn swap(p: (i32, f32)) -> (f32, i32) { return (p.1, p.0); }
fn main() -> i32 {
    let p = (-5, 2.5f32);
    let (x, y) = swap(p);
    var m = (true, -7);
    m = (m.0, m.1 * 2);
    if x == 2.5f32 and m.0 { return y * 100 + m.1; }
    return 0;
}";

#[test]
fn pairs_agree_on_every_cpu_engine() {
    // Negative i32 elements in both halves survive native packing.
    agree(SWAP, Ok(-514), false);
    // A pair is copied like a scalar: `q = p` leaves `p` usable.
    agree(
        "fn main() -> i32 {
            let p: (u32, Bool) = (4294967295u32, false);
            let q = p;
            var s = 0;
            if p.0 == q.0 and not q.1 { s += 1; }
            var f = (0, 1);
            for i in 0u32 .. 10u32 { f = (f.1, f.0 + f.1); s += f.0; }
            return s;
        }",
        Ok(144),
        false,
    );
    // Pairs cross closures as parameters and results.
    agree(
        "fn main() -> i32 {
            let k: i32 = 3;
            let f = closure [copy k: i32] (p: (i32, i32)) -> (i32, i32) { return (p.1 + k, p.0); };
            let r = f((4, 5));
            return r.0 * 10 + r.1;
        }",
        Ok(84),
        false,
    );
    // Unsuffixed elements take the expected pair's types, from an annotation
    // or a return type, and a literal's positional field needs no hint.
    agree(
        "fn flag() -> (u32, Bool) { return (7, false); }
        fn main() -> i32 {
            let p: (u32, Bool) = (1, false);
            let q = flag();
            let x = (1, 2.5).1;
            if p.0 == 1u32 and not q.1 and q.0 == 7u32 and x == 2.5f32 { return 3; }
            return 0;
        }",
        Ok(3),
        false,
    );
    // Element arithmetic keeps Invar's checked overflow trap.
    agree(
        "fn main() -> i32 { let p = (2147483647, 1); return p.0 + p.1; }",
        Err(eval::TrapCategory::IntegerOverflow),
        false,
    );
}

#[test]
fn malformed_pairs_are_refused_with_exact_codes() {
    for (source, code) in [
        (
            "fn main() -> i32 { let p = (1, 2, 3); return p.0; }",
            "TUPLE_SHAPE",
        ),
        (
            "fn main() -> i32 { let p = (1, (2, 3)); return p.0; }",
            "TUPLE_SHAPE",
        ),
        (
            "fn main() -> i32 { let p = (1, 2.5f64); return p.0; }",
            "TUPLE_SHAPE",
        ),
        (
            "fn main() -> i32 { let p = (1, 2); return p.2; }",
            "TUPLE_SHAPE",
        ),
        (
            "fn main() -> i32 { let p: (i32, Array<i32>) = (1, [2]); return p.0; }",
            "TUPLE_SHAPE",
        ),
        (
            "fn main() -> i32 { var p = (1, 2); p.0 = 5; return p.0; }",
            "PARSE_EXPECTED_TOKEN",
        ),
        (
            "fn main() -> i32 { let p = (1, 2); if p == (1, 2) { return 1; } return 0; }",
            "TYPE_MISMATCH",
        ),
        (
            "fn main() -> i32 { let (a, b) = 5; return a; }",
            "TYPE_MISMATCH",
        ),
        (
            "struct R { a: i32, b: i32 } fn main() -> i32 { let (x, y) = R { a: 1, b: 2 }; return x; }",
            "TYPE_MISMATCH",
        ),
        (
            "fn main() -> i32 { let p = (quantity::<m>(1), 2); return p.1; }",
            "TUPLE_SHAPE",
        ),
        (
            "fn main() -> i32 { let q: Quantity<i32,m> = quantity::<m>(1); let p = (2, q); return p.0; }",
            "TUPLE_SHAPE",
        ),
        (
            "struct R { p: (i32, i32) } fn main() -> i32 { return 0; }",
            "UNSUPPORTED_FEATURE",
        ),
        (
            "enum V { A((i32, i32)), B } fn main() -> i32 { return 0; }",
            "UNSUPPORTED_FEATURE",
        ),
        (
            "fn f(p: read (i32, i32)) -> i32 { return p.0; } fn main() -> i32 { return f((1, 2)); }",
            "TYPE_MISMATCH",
        ),
        (
            "fn id<T: Value>(x: T) -> T { return x; } fn main() -> i32 { let q = id((1, 2)); return q.0; }",
            "TYPE_MISMATCH",
        ),
        (
            "fn main() -> i32 { let p: (i32, u32) = (1, 2u32); let f = closure [copy p: (i32, u32)] () -> u32 { return p.1; }; return 0; }",
            "TYPE_MISMATCH",
        ),
    ] {
        assert_eq!(first_code(source), code, "{source}");
    }
}

#[test]
fn the_net_oracle_refuses_pairs() {
    let checked = checked(SWAP);
    let net = Net::lower(&checked, Limits::default()).unwrap();
    let proof = net.verify().unwrap();
    let run = oracle::observe(&net, &proof, oracle::Limits::default()).unwrap();
    // Refused at admission, before any step runs.
    assert_eq!(run.failure, Some(oracle::ToolFailure::Unsupported));
    assert_eq!(run.semantic.steps_used, 0);
    match run.semantic.result {
        Err(eval::EvalError::Tool { code, .. }) => assert_eq!(code, "UNSUPPORTED_FEATURE"),
        other => panic!("{other:?}"),
    }
    // The MIR the other engines run is unchanged by the refusal.
    assert!(mir::lower(&checked).is_ok());
}

#[test]
fn the_c_export_boundary_refuses_pairs() {
    let source = "pub fn split(x: i32) -> (i32, i32) { return (x, -x); }
pub fn first(p: (i32, i32)) -> i32 { return p.0; }
fn main() -> i32 { return first(split(3)); }";
    for name in ["split", "first"] {
        for engine in [ExportEngine::Reference, ExportEngine::NativeSerial] {
            match compile_exports(source.as_bytes(), "pairs.inv", &[name], engine) {
                Err(ExportError::Boundary { code, .. }) => assert_eq!(code, "NATIVE_ABI", "{name}"),
                Err(other) => panic!("{name}: {other:?}"),
                Ok(_) => panic!("{name}: a pair crossed the C boundary"),
            }
        }
    }
}

#[test]
fn a_loop_reading_a_pair_is_not_admitted_and_runs_serially() {
    let map = |step: &str, binding: &str| {
        format!(
            "fn main() -> i32 {{
                {binding}
                let xs: Array<u32> = [1u32, 0u32, 3u32, 4u32, 0u32, 6u32, 7u32, 8u32];
                var ys: Array<u32> = filled_u32(8u32, 0u32);
                for i in 0u32 .. len(xs) {{ ys[i] = wrapping_add(xs[i], {step}); }}
                if ys[1u32] == 1u32 and ys[7u32] == 9u32 {{ return 1; }}
                return 0;
            }}"
        )
    };
    let pair = map("p.1", "let p: (u32, u32) = (5u32, 1u32);");
    let plain = map("1u32", "");
    agree(&pair, Ok(1), false);
    agree(&plain, Ok(1), true);
    // The literal form runs as a cpu-parallel map; the pair form has no
    // admitted candidate and falls back to serial MIR.
    let receipt = |source: &str| parallel(source, 1_000_000).unwrap();
    assert_eq!(
        receipt(&plain)["backends"],
        serde_json::json!(["cpu-parallel"])
    );
    assert_eq!(
        receipt(&pair)["backends"],
        serde_json::json!(["mir-serial"])
    );
    assert_eq!(receipt(&pair)["fallback_reason"], "NO_ADMITTED_CANDIDATE");
}

#[test]
fn the_device_routes_refuse_pairs() {
    use invar::general_net::{
        compiled::Target,
        device_host::{Limits, Plan},
    };
    let source = "fn run(n: u32) -> u32 { let p = (n, 2u32); return wrapping_add(p.0, p.1); }";
    for target in [Target::Cuda, Target::Metal] {
        let refusal = Plan::from_source(source.as_bytes(), "run", target, Limits::default()).err();
        // Refused by the device compiler, which has no pair word layout.
        assert_eq!(
            format!("{refusal:?}"),
            "Some(Provenance(Compiler(UnsupportedType)))",
            "{target:?}"
        );
    }
}
