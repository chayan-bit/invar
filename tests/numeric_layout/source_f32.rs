//! Issue #1 (M4-01).
#![forbid(unsafe_code)]
use invar::{check, eval, front, mir, mir_eval};
fn qualify(source: &str) {
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    invar::ownership::check(&checked).unwrap();
    let program = mir::lower(&checked).unwrap();
    let complete = eval::observe(&checked, 10000);
    assert_eq!(complete.result, Ok(eval::Value::I32(7)), "{source}");
    for fuel in 0..=complete.steps_used + 1 {
        let reference = eval::observe(&checked, fuel);
        let actual = mir_eval::observe(&program, fuel);
        assert_eq!(actual.result, reference.result, "fuel={fuel}: {source}");
        assert_eq!(actual.steps_used, reference.steps_used, "{source}");
        assert_eq!(actual.cleanup_trace, reference.cleanup_trace, "{source}");
    }
    #[cfg(feature = "native")]
    for fuel in [
        complete.steps_used - 1,
        complete.steps_used,
        complete.steps_used + 1,
    ] {
        let reference = eval::observe(&checked, fuel);
        let (status, doc) = invar::native::run_program(&program, fuel).unwrap();
        assert_eq!(doc["steps_used"], reference.steps_used, "{doc}: {source}");
        match reference.result {
            Ok(eval::Value::I32(value)) => {
                assert_eq!(status, 0, "{doc}");
                assert_eq!(doc["result"]["value"], value);
            }
            Err(eval::EvalError::EvaluationLimit { .. }) => {
                assert_eq!(status, 4, "{doc}");
            }
            other => panic!("{other:?}"),
        }
    }
}
#[test]
fn source_f32_strict_arithmetic_and_ieee_predicates() {
    for source in [
        "fn nudge(x: f32) -> f32 { return x + 0.2; } fn main() -> i32 { if nudge(0.1) == 0.3 { return 7; } return 0; }",
        "fn main() -> i32 { let x: f32 = -0.0; let n: f32 = 0.0 / 0.0; if x == 0.0 and 1.0 / x < 0.0 and n != n and not (n < 1.0) and not (n >= 1.0) { return 7; } return 0; }",
        "fn main() -> i32 { let x = 16777216.0; if (x + 1.0) - x == 0.0 and f32_from_u32(16777217u32) == x and f32_from_i32(-7i32) == -7.0 { return 7; } return 0; }",
        "fn main() -> i32 { let a = 1.00000011920928955078125; let b = 0.99999988079071044921875; if a * b - 1.0 == 0.0 and fma(a,b,-1.0) < 0.0 { return 7; } return 0; }",
        "fn main() -> i32 { let tiny = 0.00000000000000000000000000000000000000000000140129846432481707092372958328991613128026194187651577175706828388979108268586060148663818836212158203125; if tiny > 0.0 and tiny / 2.0 == 0.0 and tiny * 3.0 / 2.0 == tiny * 2.0 { return 7; } return 0; }",
    ] {
        qualify(source);
    }
}
#[test]
fn source_f32_affine_arrays_mutation_and_owner_return() {
    qualify(
        "fn make() -> Array<f32> { let a: Array<f32> = [1.0,2.0,3.0]; return a; } fn update(a: mut Array<f32>) -> Unit { a[1u32] = fma(a[0u32],3.0,a[2u32]); } fn main() -> i32 { var a = make(); update(a); let b = filled_f32(0u32,0.0); if a[1u32] == 6.0 and len(a) == 3u32 and len(b) == 0u32 { return 7; } return 0; }",
    );
}
#[test]
fn source_f32_rejects_implicit_conversion_and_remainder() {
    for source in [
        "fn main() -> i32 { let x: f32 = 1; return 0; }",
        "fn main() -> i32 { let x = 1.0 % 2.0; return 0; }",
        "fn main() -> i32 { let a: Array<f32> = [1.0]; let v = view(a,0u32,1u32); return 0; }",
        "fn main() -> i32 { let a: Array<f32> = [1.0]; let b = a; return 0; }",
    ] {
        let parsed = front::parse(source.as_bytes()).unwrap();
        match check::check(parsed, true) {
            Err(_) => {}
            Ok(checked) => assert!(invar::ownership::check(&checked).is_err(), "{source}"),
        }
    }
}

#[test]
fn source_f32_literal_bit_outcomes_and_contract_mutations() {
    for (literal, bits) in [
        ("0.1", 0x3dcc_cccd),
        ("0.3", 0x3e99_999a),
        ("4294967296.0", 0x4f80_0000),
    ] {
        let source = format!("fn main() -> i32 {{ let value = {literal}; return 7; }}");
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let mut program = mir::lower(&checked).unwrap();
        let constants: Vec<_> = program
            .functions
            .iter()
            .flat_map(|f| &f.blocks)
            .flat_map(|b| &b.instructions)
            .filter_map(|i| match i.kind {
                mir::Operation::F32(value) => Some(value.to_bits()),
                _ => None,
            })
            .collect();
        assert_eq!(constants, vec![bits]);
        for version in [0, 2, u32::MAX] {
            program.layout_version = version;
            assert_eq!(mir::verify(&program).unwrap_err().code, "MIR_INVALID");
        }
    }
}

#[test]
fn source_f32_ordinary_counted_matrix_zero_one_odd_axes() {
    let function = "fn mat(a: read Array<f32>, b: read Array<f32>, rows: u32, inner: u32, cols: u32) -> Array<f32> { var c = filled_f32(rows * cols,0.0); for i in 0u32..rows { for j in 0u32..cols { var sum: f32 = 0.0; for k in 0u32..inner { sum = sum + a[i * inner + k] * b[k * cols + j]; } c[i * cols + j] = sum; } } return c; }";
    for (rows, inner, cols) in [(0, 3, 2), (2, 0, 3), (1, 1, 1), (3, 5, 3)] {
        let source = format!(
            "{function} fn main() -> i32 {{ let a = filled_f32({rows}u32 * {inner}u32,1.0); let b = filled_f32({inner}u32 * {cols}u32,2.0); let c = mat(a,b,{rows}u32,{inner}u32,{cols}u32); for index in 0u32..len(c) {{ if c[index] != f32_from_u32({inner}u32 * 2u32) {{ return 0; }} }} return 7; }}"
        );
        qualify(&source);
    }
}

#[cfg(feature = "native")]
#[test]
fn source_f32_native_ir_keeps_strict_flags_and_explicit_fma() {
    let source = "fn compute(a: f32,b: f32,c: f32) -> f32 { return a * b + c + fma(a,b,c); } fn main() -> i32 { if compute(1.0,2.0,3.0) == 10.0 { return 7; } return 0; }";
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    let object = invar::native::emit_mir(&program).unwrap();
    assert!(object.ir.contains("fmul float"));
    assert!(object.ir.contains("fadd float"));
    assert!(object.ir.contains("call float @llvm.fma.f32"));
    assert!(object.ir.contains("\"denormal-fp-math-f32\"=\"ieee,ieee\""));
    for forbidden in [
        "fadd fast",
        "fmul fast",
        " reassoc ",
        " contract ",
        " nnan ",
        " ninf ",
        " nsz ",
    ] {
        assert!(!object.ir.contains(forbidden), "{forbidden}: {}", object.ir);
    }
    qualify(source);
}

#[test]
fn source_f32_does_not_expand_scalar_exports() {
    use invar::exports::{ExportEngine, compile_exports};
    let source = "fn entry(x: f32) -> f32 { return x + 1.0; }";
    assert!(
        compile_exports(
            source.as_bytes(),
            "original-m4.inv",
            &["entry"],
            ExportEngine::Reference
        )
        .is_err()
    );
    #[cfg(feature = "native")]
    assert!(
        compile_exports(
            source.as_bytes(),
            "original-m4.inv",
            &["entry"],
            ExportEngine::NativeSerial
        )
        .is_err()
    );
    // HA-1.3 admits read and mut Array<f32> host arrays and the f32 values the
    // exported code computes internally, so these exports are no longer
    // refusals. Only scalar f32 parameters and results stay outside the ABI.
    for source in [
        "fn helper() -> f32 { return 1.0; } fn entry() -> i32 { if helper() > 0.0 { return 7; } return 0; }",
        "fn entry(a: read Array<f32>) -> i32 { return 7; }",
    ] {
        assert!(
            compile_exports(
                source.as_bytes(),
                "original-m4.inv",
                &["entry"],
                ExportEngine::Reference
            )
            .is_ok(),
            "{source}"
        );
        #[cfg(feature = "native")]
        assert!(
            compile_exports(
                source.as_bytes(),
                "original-m4.inv",
                &["entry"],
                ExportEngine::NativeSerial
            )
            .is_ok(),
            "{source}"
        );
    }
}
