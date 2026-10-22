#![forbid(unsafe_code)]
use invar::budget::ByteBudget;
use invar::scatter_proof;
use invar::{check, eval, front, general_net, mir, mir_eval};
use scatter_proof::{Contract, Error, RULE_REVISION};

/// Domain size the boundary test exercises; no longer a refusal bound (#278 byte budgets).
const MAX_ELEMENTS: u32 = 4096;

fn checked(s: &str) -> check::CheckedProgram {
    check::check(front::parse(s.as_bytes()).unwrap(), false).unwrap()
}
fn function(lo: u32, hi: u32, output: u32, index: &str) -> String {
    format!(
        "fn scatter(a: read Array<u32>) -> Array<u32> {{ var out: Array<u32> = filled_u32({output}u32,0u32); for i in {lo}u32..{hi}u32 {{ out[{index}] = a[i]; }} return out; }}"
    )
}
fn contract(s: &str) -> Contract {
    Contract::from_checked(s.as_bytes(), &checked(s), "scatter").unwrap()
}
fn array(values: &[u32]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|v| format!("{v}u32"))
            .collect::<Vec<_>>()
            .join(",")
    )
}
fn source(f: &str, input: &[u32], output: &[u32]) -> String {
    format!(
        "{f} fn main() -> i32 {{ let a: Array<u32> = {}; let retained: Array<u32> = {}; let result = scatter(a); let expected: Array<u32> = {}; if len(result) != len(expected) {{ return 0; }} for output_index in 0u32..len(result) {{ if result[output_index] != expected[output_index] {{ return 0; }} }} for keep_index in 0u32..len(a) {{ if a[keep_index] != retained[keep_index] {{ return 0; }} }} return 7; }}",
        array(input),
        array(input),
        array(output)
    )
}
fn prefixes(s: &str, expected: Option<i32>) -> u64 {
    let p = checked(s);
    let m = match Contract::from_checked(s.as_bytes(), &p, "scatter") {
        Ok(c) => c.verified_mir().unwrap(),
        Err(
            Error::Unrecognized
            | Error::NonInjective
            | Error::Overflow
            | Error::DestinationBounds
            | Error::Budget,
        ) => {
            let fallback = general_net::Net::lower(&p, general_net::Limits::default()).unwrap();
            fallback.compiled_mir(&fallback.verify().unwrap()).unwrap()
        }
        Err(error) => panic!("unexpected proof boundary error: {error:?}"),
    };
    let net = general_net::Net::encode(&m, general_net::Limits::default()).unwrap();
    let proof = net.verify().unwrap();
    mir::verify(&m).unwrap();
    let complete = eval::observe(&p, 20_000);
    if let Some(value) = expected {
        assert_eq!(complete.result, Ok(eval::Value::I32(value)));
    } else {
        assert!(matches!(
            complete.result,
            Err(eval::EvalError::Trap { .. }) | Err(eval::EvalError::Tool { .. })
        ));
    }
    for fuel in 0..=complete.steps_used + 1 {
        let reference = eval::observe(&p, fuel);
        let extracted = mir_eval::observe(&m, fuel);
        let interaction = general_net::oracle::observe(
            &net,
            &proof,
            general_net::oracle::Limits {
                fuel,
                ..general_net::oracle::Limits::default()
            },
        )
        .unwrap();
        for actual in [&extracted, &interaction.semantic] {
            assert_eq!(actual.result, reference.result, "fuel={fuel}");
            assert_eq!(actual.steps_used, reference.steps_used, "fuel={fuel}");
            assert_eq!(actual.cleanup_trace, reference.cleanup_trace, "fuel={fuel}");
        }
    }
    complete.steps_used + 2
}

#[test]
fn issue_m4_05_gather_scatter_frontier_static_source_and_goldens() {
    let cases = [
        (0, 0, 0, "i", vec![], vec![]),
        (4, 2, 2, "4294967295u32*i+4294967295u32", vec![], vec![0, 0]),
        (0, 1, 1, "0u32*i", vec![11], vec![11]),
        (0, 1, 1, "4294967295u32*i", vec![u32::MAX], vec![u32::MAX]),
        (0, 3, 3, "i", vec![11, 22, 33], vec![11, 22, 33]),
        (0, 3, 5, "2u32*i", vec![11, 22, 33], vec![11, 0, 22, 0, 33]),
        (
            1,
            3,
            7,
            "2u32*i+1u32",
            vec![11, 22, 33],
            vec![0, 0, 0, 22, 0, 33, 0],
        ),
        (
            1,
            2,
            5,
            "0u32*i+4u32",
            vec![11, u32::MAX],
            vec![0, 0, 0, 0, u32::MAX],
        ),
        (
            0,
            3,
            8,
            "1u32+(i*3u32)",
            vec![11, 22, 33],
            vec![0, 11, 0, 0, 22, 0, 0, 33],
        ),
    ];
    let mut count = 0;
    for (lo, hi, n, index, input, wanted) in cases {
        let f = function(lo, hi, n, index);
        let c = contract(&f);
        let cert = c.certificate();
        assert_eq!(cert.revision(), RULE_REVISION);
        assert_eq!(cert.domain(), (lo, hi));
        assert_eq!(cert.output_len(), n);
        assert_eq!(cert.iterations(), hi.saturating_sub(lo));
        assert_eq!(
            cert.source_contract(),
            c.verified_mir().unwrap().contract_digest()
        );
        assert_eq!(
            &f[cert.loop_span().start..cert.loop_span().end],
            &f[f.find("for i").unwrap()..f.rfind(" return").unwrap()]
        );
        assert_eq!(
            &f[cert.destination_span().start..cert.destination_span().end],
            index
        );
        let original = input.clone();
        assert_eq!(c.candidate(&input).unwrap(), wanted);
        assert_eq!(input, original);
        count += prefixes(&source(&f, &input, &wanted), Some(7));
    }
    let c = contract(&function(0, 3, 6, "2u32*i+1u32"));
    assert_eq!(c.certificate().affine_map(), (2, 1));
    assert_eq!(c.certificate().endpoints(), Some((1, 5)));
    let f = function(0, 3, 5, "2u32*i")
        .replace("scatter", "reordered")
        .replace("out", "fresh")
        .replace("a:", "values:")
        .replace("a[i]", "values[i]")
        .replace("2u32*i", "((i)*(2u32))")
        .replace("return fresh;", "return move fresh;");
    assert_eq!(
        Contract::from_checked(f.as_bytes(), &checked(&f), "reordered")
            .unwrap()
            .candidate(&[7, 8, 9])
            .unwrap(),
        [7, 0, 8, 0, 9]
    );
    println!("static scatter positive source/MIR/net prefixes per engine: {count}");
}

#[test]
fn issue_m4_05_gather_scatter_frontier_static_noninjective_overflow_bounds() {
    let cases = [
        (
            function(0, 2, 1, "0u32*i"),
            vec![11, 22],
            vec![22],
            Error::NonInjective,
            true,
        ),
        (
            function(1, 3, 1, "0u32*i"),
            vec![0, 11, 22],
            vec![22],
            Error::NonInjective,
            true,
        ),
        (
            function(1, 2, 3, "4294967295u32*i+1u32"),
            vec![0, 11],
            vec![],
            Error::Overflow,
            false,
        ),
        (
            function(2, 3, 3, "4294967295u32*i"),
            vec![0, 0, 11],
            vec![],
            Error::Overflow,
            false,
        ),
        (
            function(2, 3, 1, "2147483648u32*i"),
            vec![0, 0, 11],
            vec![],
            Error::Overflow,
            false,
        ),
        (
            function(0, 1, 0, "i"),
            vec![11],
            vec![],
            Error::DestinationBounds,
            false,
        ),
        (
            function(0, 3, 4, "2u32*i"),
            vec![11, 22, 33],
            vec![],
            Error::DestinationBounds,
            false,
        ),
        // RHS-first source trap dominates the refused index overflow.
        (
            function(1, 2, 3, "4294967295u32*i+1u32"),
            vec![],
            vec![],
            Error::Overflow,
            false,
        ),
    ];
    let mut count = 0;
    for (f, input, wanted, error, success) in cases {
        assert_eq!(
            Contract::from_checked(f.as_bytes(), &checked(&f), "scatter").unwrap_err(),
            error
        );
        let s = source(&f, &input, &wanted);
        count += prefixes(&s, success.then_some(7));
        if !success {
            let Err(eval::EvalError::Trap { category, span }) =
                eval::observe(&checked(&s), 20_000).result
            else {
                panic!("original source trap");
            };
            if input.is_empty() {
                assert_eq!(category, eval::TrapCategory::IndexOutOfBounds);
                assert_eq!(&s[span.start..span.end], "a[i]");
            } else if error == Error::Overflow {
                assert_eq!(category, eval::TrapCategory::IntegerOverflow);
                assert!(matches!(&s[span.start..span.end], "+" | "*"));
            } else {
                assert_eq!(category, eval::TrapCategory::IndexOutOfBounds);
            }
        }
    }
    println!("static scatter refused arithmetic source/MIR/net prefixes per engine: {count}");
}

#[test]
fn issue_m4_05_gather_scatter_frontier_static_identity_and_unknown_forms() {
    let f = function(0, 3, 5, "2u32*i");
    let mut p = checked(&f);
    p.expression_types.clear();
    assert!(matches!(
        Contract::from_checked(f.as_bytes(), &p, "scatter"),
        Err(Error::CheckedMismatch)
    ));
    let mut p = checked(&f);
    p.function_expression_types
        .get_mut("scatter")
        .unwrap()
        .pop();
    assert!(matches!(
        Contract::from_checked(f.as_bytes(), &p, "scatter"),
        Err(Error::CheckedMismatch)
    ));
    let mut p = checked(&f);
    p.program.functions[0].span.end += 1;
    assert!(matches!(
        Contract::from_checked(f.as_bytes(), &p, "scatter"),
        Err(Error::CheckedMismatch)
    ));
    assert!(matches!(
        Contract::from_checked(
            function(0, 2, 5, "2u32*i").as_bytes(),
            &checked(&f),
            "scatter"
        ),
        Err(Error::CheckedMismatch)
    ));
    for changed in [
        f.replace("2u32*i", "wrapping_mul(2u32,i)"),
        f.replace("2u32*i", "i%2u32"),
        f.replace("2u32*i", "i/2u32"),
        f.replace("2u32*i", "a[i]"),
        f.replace("2u32*i", "1u32"),
        f.replace("2u32*i", "i*i"),
        f.replace("2u32*i", "4u32-i"),
        f.replace("2u32*i", "(i+1u32)*2u32"),
        f.replace(
            "out[2u32*i] = a[i];",
            "out[2u32*i] = a[i]; out[0u32] = a[i];",
        ),
        f.replace("a[i]", "out[i]"),
        f.replace("0u32..3u32", "0u32..len(a)"),
        f.replace("filled_u32(5u32,0u32)", "filled_u32(len(a),0u32)"),
        f.replace("for i in 0u32..3u32", "while false"),
    ] {
        // while-form still needs a defined i to be source-valid.
        let changed = if changed.contains("while") {
            changed.replace("while false", "let i = 0u32; while false")
        } else {
            changed
        };
        assert!(matches!(
            Contract::from_checked(changed.as_bytes(), &checked(&changed), "scatter"),
            Err(Error::Unrecognized)
        ));
    }
    let mut count = 0;
    for (index, wanted) in [
        ("i%2u32", vec![33, 22, 0]),
        ("wrapping_mul(0u32,i)", vec![33, 0, 0]),
        ("4u32-i", vec![0, 0, 33, 22, 11]),
    ] {
        let f = function(0, 3, wanted.len() as u32, index);
        assert!(matches!(
            Contract::from_checked(f.as_bytes(), &checked(&f), "scatter"),
            Err(Error::Unrecognized)
        ));
        count += prefixes(&source(&f, &[11, 22, 33], &wanted), Some(7));
    }
    println!("static scatter unknown-form source/MIR/net prefixes per engine: {count}");
}

#[test]
fn issue_m4_05_gather_scatter_frontier_static_input_alias_sign_negatives() {
    let f = function(0, 3, 5, "2u32*i");
    let c = contract(&f);
    assert_eq!(c.candidate(&[11, 22]).unwrap_err(), Error::InputBounds);
    let mut count = prefixes(&source(&f, &[11, 22], &[]), None);
    let good = checked(&f);
    let mutable = f.replace("read Array<u32>", "mut Array<u32>");
    assert!(matches!(
        Contract::from_checked(mutable.as_bytes(), &checked(&mutable), "scatter"),
        Err(Error::Unrecognized)
    ));
    for changed in [
        f.replace("2u32*i", "-1u32*i"),
        f.replace("2u32*i", "-1i32"),
        f.replace("0u32..3u32", "-1i32..3u32"),
        f.replace("read Array<u32>", "mut Array<u32>"),
    ] {
        let result = Contract::from_checked(changed.as_bytes(), &good, "scatter");
        if changed.contains("mut Array") {
            assert!(matches!(result, Err(Error::CheckedMismatch)));
        } else {
            let Err(Error::Source(error)) = result else {
                panic!("source-invalid signed index");
            };
            assert_eq!(error.code, "TYPE_MISMATCH");
        }
    }
    for (s, code) in [
        (
            "fn alias(a: mut Array<u32>,b: read Array<u32>)->i32 { a[0u32]=b[0u32]; return 0; } fn main()->i32 { var a: Array<u32> =[1u32]; return alias(a,a); }",
            "OWN_BORROW_CONFLICT",
        ),
        (
            "fn main()->i32 { let a: Array<u32> =[1u32]; let copied = a; return 0; }",
            "OWN_MOVE_REQUIRED",
        ),
    ] {
        assert_eq!(
            check::check(front::parse(s.as_bytes()).unwrap(), true)
                .unwrap_err()
                .code,
            code
        );
    }
    // Two calls borrowing the same input each receive a distinct fresh output.
    let s = format!(
        "{f} fn main()->i32 {{ let a: Array<u32> =[11u32,22u32,33u32]; var first = scatter(a); let second = scatter(a); first[0u32]=99u32; if second[0u32]==11u32 and a[0u32]==11u32 {{ return 7; }} return 0; }}"
    );
    count += prefixes(&s, Some(7));
    println!("static scatter input/alias source/MIR/net prefixes per engine: {count}");
}

#[test]
fn issue_m4_05_gather_scatter_frontier_static_empty_singleton_extrema_budget() {
    // These algebraic certificates are constructed without any array payload.
    let c = contract(&function(
        u32::MAX,
        u32::MAX,
        0,
        "4294967295u32*i+4294967295u32",
    ));
    assert_eq!(c.certificate().iterations(), 0);
    assert_eq!(c.certificate().endpoints(), None);
    assert!(c.candidate(&[]).unwrap().is_empty());
    let c = contract(&function(u32::MAX - 1, u32::MAX, 1, "0u32*i"));
    assert_eq!(c.certificate().iterations(), 1);
    assert_eq!(c.certificate().endpoints(), Some((0, 0)));
    assert_eq!(c.candidate(&[]).unwrap_err(), Error::InputBounds);
    let c = contract(&function(0, MAX_ELEMENTS, MAX_ELEMENTS, "i"));
    assert_eq!(c.certificate().endpoints(), Some((0, MAX_ELEMENTS - 1)));
    let input: Vec<_> = (0..MAX_ELEMENTS).collect();
    assert_eq!(c.candidate(&input).unwrap(), input);
    // Past the old cap the region is admitted under the default budget and refused only by bytes.
    assert_eq!(
        c.candidate(&vec![0; MAX_ELEMENTS as usize + 1]).unwrap(),
        vec![0; MAX_ELEMENTS as usize]
    );
    assert!(matches!(
        c.candidate_within(&input, &ByteBudget::new(2 * 4 * 4096 - 1, 0, 0)),
        Err(Error::Exceeded(e)) if e.needed == Some(2 * 4 * 4096)
    ));
    assert_eq!(
        c.candidate_within(&input, &ByteBudget::new(2 * 4 * 4096, 0, 0))
            .unwrap(),
        input
    );
    let mut count = 0;
    // Certificates no longer carry a size cap: past the old bound a domain is certified (or refused
    // for its own reason, here non-injectivity) and the byte budget decides at `candidate`. Only
    // the language's per-array limit still refuses at certification: the source would trap.
    for (f, certified) in [
        (function(0, 0, MAX_ELEMENTS + 1, "i"), Ok(())),
        (
            function(0, MAX_ELEMENTS + 1, 1, "0u32*i"),
            Err(Error::NonInjective),
        ),
        (
            function(0, 0, invar::array_limits::MAX_ARRAY_LEN as u32 + 1, "i"),
            Err(Error::Budget),
        ),
    ] {
        assert_eq!(
            Contract::from_checked(f.as_bytes(), &checked(&f), "scatter").map(|_| ()),
            certified
        );
        let s = format!(
            "{f} fn main()->i32 {{ let a: Array<u32> =[]; let result=scatter(a); return 7; }}"
        );
        let expected = if f.contains("0u32..0u32") && !f.contains("16777217u32") {
            Some(7)
        } else {
            None
        };
        count += prefixes(&s, expected);
    }
    let f = function(1, 2, 1, "4294967295u32*i+1u32");
    let s = format!(
        "{f} fn main()->i32 {{ let a: Array<u32> =[]; if false {{ let unused=scatter(a); }} return 7; }}"
    );
    count += prefixes(&s, Some(7));
    println!("static scatter budget/demand source/MIR/net prefixes per engine: {count}");
}
