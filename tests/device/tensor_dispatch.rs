//! Ordinary CLI and owning-device tests. Hardware tests fail if Metal is absent.

use a1_tensor_common::{DIRECT, SCALAR, TC, program};
use invar::{check, front, mir, mir_eval};
use std::process::Command;

#[path = "../support/a1_tensor_common.rs"]
mod a1_tensor_common;

#[path = "../support/gnet_agree.rs"]
mod agree;

fn assert_net(checked: &check::CheckedProgram, fuel: u64, expected: &invar::eval::Observation) {
    use invar::general_net::{Limits, Net, oracle};
    let net = Net::lower(checked, Limits::default()).unwrap();
    let observed = oracle::observe(
        &net,
        &net.verify().unwrap(),
        oracle::Limits {
            fuel,
            ..oracle::Limits::default()
        },
    )
    .unwrap();
    assert_eq!(observed.failure, None, "{:?}", observed.semantic);
    agree::same(&observed.semantic, expected, "ordinary TC net oracle");
}

#[test]
fn tensor_ordinary_tc_rounds_inputs_and_serial_fallback() {
    let source = include_str!("../a1_tensor/tc.inv");
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    invar::ownership::check(&checked).unwrap();
    let p = mir::lower(&checked).unwrap();
    let reference = invar::eval::observe(&checked, 10_000);
    assert_eq!(reference.result, Ok(invar::eval::Value::I32(42)));
    for fuel in [1, reference.steps_used - 1, reference.steps_used, 10_000] {
        let expected = invar::eval::observe(&checked, fuel);
        let actual = mir_eval::observe(&p, fuel);
        assert_eq!(actual.result, expected.result);
        assert_eq!(actual.steps_used, expected.steps_used);
        assert_eq!(actual.cleanup_trace, expected.cleanup_trace);
        assert_net(&checked, fuel, &expected);
    }
    for engine in ["reference", "cpu-parallel", "auto"] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_invar"));
        command.args([
            "run",
            "tests/a1_tensor/tc.inv",
            "--engine",
            engine,
            "--format",
            "json",
        ]);
        if engine == "cpu-parallel" {
            command.args(["--workers", "2"]);
        }
        let got = command.output().unwrap();
        assert!(
            got.status.success(),
            "{engine}: {} {}",
            String::from_utf8_lossy(&got.stdout),
            String::from_utf8_lossy(&got.stderr)
        );
        let doc: serde_json::Value = serde_json::from_slice(&got.stdout).unwrap();
        assert_eq!(doc["result"]["value"], 42, "{engine}: {doc}");
    }
}

#[test]
fn tensor_ordinary_tc_exact_sum_range_and_demand() {
    // 65504^2 + 2^-24 - 65504^2: sequential f32 loses the middle
    // term, while D5's once-rounded exact reference returns it.
    let exact = TC
        .replace(
            "[1.00048828125]",
            "[65504.0,0.000000059604644775390625,-65504.0]",
        )
        .replace("[1.0]", "[65504.0,1.0,65504.0]")
        .replace("mat(a,b,1u32,1u32,1u32)", "mat(a,b,1u32,3u32,1u32)")
        .replace("out[0u32]==1.0", "out[0u32]==0.000000059604644775390625");
    for source in [
        exact,
        TC.replace("[1.00048828125]", "[1.00146484375]")
            .replace("out[0u32]==1.0", "out[0u32]==1.001953125"),
        TC.replace("[1.00048828125]", "[1.0,(1.0/0.0)]"),
        TC.replace("[1.0]", "[1.0,(1.0/0.0)]"),
    ] {
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let p = mir::lower(&checked).unwrap();
        let expected = invar::eval::observe(&checked, 10_000);
        let actual = mir_eval::observe(&p, 10_000);
        assert_eq!(expected.result, Ok(invar::eval::Value::I32(42)));
        assert_eq!(actual.result, expected.result);
        assert_eq!(actual.steps_used, expected.steps_used);
        assert_eq!(actual.cleanup_trace, expected.cleanup_trace);
        assert_net(&checked, 10_000, &expected);
        #[cfg(feature = "native")]
        {
            let (status, doc) = invar::native::run_program(&p, 10_000).unwrap();
            assert_eq!(status, 0, "{doc}");
            assert_eq!(doc["result"]["value"], 42, "{doc}");
            assert_eq!(doc["steps_used"], expected.steps_used);
        }
    }
    for bad in ["65520.0", "(1.0/0.0)", "(0.0/0.0)"] {
        let source = TC.replace("1.00048828125", bad);
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let p = mir::lower(&checked).unwrap();
        let expected = invar::eval::observe(&checked, 10_000);
        let actual = mir_eval::observe(&p, 10_000);
        assert!(matches!(
            expected.result,
            Err(invar::eval::EvalError::Tool {
                code: "TC_INPUT_RANGE",
                ..
            })
        ));
        assert_eq!(actual.result, expected.result);
        assert_eq!(actual.steps_used, expected.steps_used);
        assert_eq!(actual.cleanup_trace, expected.cleanup_trace);
        assert_net(&checked, 10_000, &expected);
        // An unreached contraction must not inspect or round its input.
        let unused = source.replace(
            "let out=mat(a,b,1u32,1u32,1u32);if out[0u32]==1.0{return 42;}return 99;",
            "if false{let out=mat(a,b,1u32,1u32,1u32);}return 42;",
        );
        assert_eq!(
            mir_eval::observe(&program(&unused), 10_000).result,
            Ok(invar::eval::Value::I32(42))
        );
        let checked = check::check(front::parse(unused.as_bytes()).unwrap(), true).unwrap();
        assert_net(&checked, 10_000, &invar::eval::observe(&checked, 10_000));
    }
    for dims in ["0u32,1u32,1u32", "1u32,0u32,1u32", "1u32,1u32,0u32"] {
        let source = format!(
            "{}fn main()->i32{{let a:Array<f32> = [1.0/0.0];let b:Array<f32> = [1.0];let out=mat(a,b,{dims});return 42;}}",
            TC.split("fn main").next().unwrap()
        );
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let expected = invar::eval::observe(&checked, 10_000);
        assert_eq!(expected.result, Ok(invar::eval::Value::I32(42)));
        assert_net(&checked, 10_000, &expected);
    }
    let short = TC.replace("mat(a,b,1u32,1u32,1u32)", "mat(a,b,1u32,2u32,1u32)");
    let checked = check::check(front::parse(short.as_bytes()).unwrap(), true).unwrap();
    let expected = invar::eval::observe(&checked, 10_000);
    assert!(matches!(
        expected.result,
        Err(invar::eval::EvalError::Trap {
            category: invar::eval::TrapCategory::IndexOutOfBounds,
            ..
        })
    ));
    assert_net(&checked, 10_000, &expected);
}

#[test]
fn tensor_ordinary_multiple_tc_functions_keep_independent_admission() {
    let source = format!(
        "{}{}",
        TC,
        TC.split("fn main")
            .next()
            .unwrap()
            .replace("fn mat(", "fn other(")
    );
    let p = program(&source);
    let serial = mir_eval::observe(&p, 10_000);
    let run = invar::schedule::run_source(source.as_bytes(), &p, 10_000, 2).unwrap();
    assert_eq!(run.observation.result, serial.result);
    assert_eq!(run.observation.steps_used, serial.steps_used);
    assert_eq!(run.observation.cleanup_trace, serial.cleanup_trace);
}

#[test]
fn tensor_serial_forms_and_forced_unavailable_cli() {
    for source in [SCALAR, DIRECT, TC] {
        assert_eq!(
            mir_eval::observe(&program(source), 1_000_000).result,
            Ok(invar::eval::Value::I32(42))
        );
    }
    #[cfg(not(feature = "cuda"))]
    {
        let got = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args([
                "run",
                "tests/a1_tensor/direct.inv",
                "--engine",
                "cuda",
                "--format",
                "json",
            ])
            .output()
            .unwrap();
        assert_eq!(got.status.code(), Some(2));
        let doc: serde_json::Value = serde_json::from_slice(&got.stdout).unwrap();
        assert_eq!(doc["diagnostics"][0]["code"], "CUDA_UNAVAILABLE");
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        let got = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args([
                "run",
                "tests/a1_tensor/direct.inv",
                "--engine",
                "metal",
                "--format",
                "json",
            ])
            .output()
            .unwrap();
        assert_eq!(got.status.code(), Some(2));
        let doc: serde_json::Value = serde_json::from_slice(&got.stdout).unwrap();
        assert_eq!(doc["diagnostics"][0]["code"], "METAL_UNAVAILABLE");
    }
}
