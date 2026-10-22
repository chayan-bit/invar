//! #48 C4 part A: `--numeric-contract trig-f32-v1` on `run`, `explain` and
//! `build`. Under it, f32 `sin` and `cos` take the names of their
//! invar-trig-f32-v1 helpers after typing, so MIR, explain and receipts carry
//! the contract; every CPU engine runs them with the same bits. The default
//! is unchanged until the C4 switch (part C).
//!
//! The selection is process-wide and one-way, so every in-process test in
//! this binary selects it, and the default runs only through the CLI binary.

use invar::numeric_contract::F32Bits as F;
use invar::{check, eval, front, math, mir, mir_eval, schedule};
use std::process::Command;

const FUEL: u64 = 1_000_000;

fn f(value: f32) -> F {
    F::from_bits(value.to_bits())
}

/// A finite `x` as a source literal with exactly its value (Display is the
/// shortest round-trip decimal and never uses an exponent).
fn literal(x: F) -> String {
    let text = format!("{}", f32::from_bits(x.to_bits()));
    if text.contains('.') {
        text
    } else {
        format!("{text}.0")
    }
}

/// f32 sin, cos and sqrt over 64 arguments, compared with the
/// invar-trig-f32-v1 oracle's bits, and one f64 `sin`, which stays cr-v1.
fn source() -> String {
    let mut s = f(0.0);
    for i in 0..64u32 {
        let x = F::from_u32(i).mul(f(0.37)).sub(f(11.0));
        s = s.add(x.sin().mul(x.cos())).add(F::from_u32(i).sqrt());
    }
    format!(
        "fn main() -> i32 {{ var s: f32 = 0.0; for i in 0u32..64u32 {{ \
         let x = f32_from_u32(i) * 0.37 - 11.0; s = s + sin(x) * cos(x) + sqrt(f32_from_u32(i)); }} \
         let d: f64 = sin(f64_from_u32(3u32)); \
         if s == {} and d > 0.14 and d < 0.15 {{ return 1; }} return 0; }}",
        literal(s)
    )
}

fn calls(program: &mir::Program) -> Vec<String> {
    let mut names: Vec<String> = program
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.instructions)
        .filter_map(|instruction| match &instruction.kind {
            mir::Operation::Call { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect();
    names.sort();
    names.dedup();
    names
}

#[test]
fn issue_210_trig_option_names_f32_calls_and_keeps_bits_on_every_cpu_engine() {
    math::select_trig_f32_v1();
    let source = source();
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true)
        .unwrap_or_else(|error| panic!("{error:?}\n{source}"));
    let program = mir::lower(&checked).unwrap();
    let names = calls(&program);
    for name in [math::TRIG_F32_SIN, math::TRIG_F32_COS, "sin", "sqrt"] {
        assert!(names.iter().any(|n| n == name), "{name} in {names:?}");
    }
    assert!(!names.iter().any(|n| n == "cos"), "{names:?}");
    // f64 `sin` keeps cr-v1, and both contracts are recorded.
    let contracts = math::contracts(&program).unwrap();
    assert_eq!(contracts["trig"], "invar-trig-f32-v1");
    assert_eq!(contracts["math"], math::CONTRACT);
    let reference = eval::observe(&checked, FUEL);
    assert_eq!(reference.result, Ok(eval::Value::I32(1)), "{source}");
    let lowered = mir_eval::observe(&program, FUEL);
    assert_eq!(lowered.result, reference.result);
    assert_eq!(lowered.steps_used, reference.steps_used);
    for workers in [1, 2, 7] {
        let parallel = schedule::run_source(source.as_bytes(), &program, FUEL, workers).unwrap();
        assert_eq!(parallel.observation.result, reference.result, "{workers}");
        assert_eq!(parallel.observation.steps_used, reference.steps_used);
    }
    #[cfg(feature = "native")]
    {
        let (status, doc) = invar::native::run_program(&program, FUEL).unwrap();
        assert_eq!(
            (status, &doc["result"]["value"]),
            (0, &serde_json::json!(1)),
            "{doc}"
        );
        assert_eq!(doc["steps_used"], reference.steps_used);
    }
}

#[test]
fn issue_210_trig_option_leaves_other_programs_unchanged() {
    math::select_trig_f32_v1();
    let source = "fn main() -> i32 { let a: f64 = cos(f64_from_u32(0u32)); \
                  let r = sqrt(4.0); if a == 1.0 and r == 2.0 { return 1; } return 0; }";
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    let names = calls(&program);
    assert!(names.iter().all(|n| !math::trig_f32_v1(n)), "{names:?}");
    assert_eq!(
        eval::observe(&checked, FUEL).result,
        Ok(eval::Value::I32(1))
    );
    // Source cannot name the helpers.
    let named = "fn main() -> i32 { let x = sin@trig-f32-v1(1.0); return 0; }";
    assert!(
        front::parse(named.as_bytes())
            .and_then(|p| check::check(p, true))
            .is_err()
    );
}

fn cli(args: &[&str]) -> (Option<i32>, serde_json::Value, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(args)
        .output()
        .expect("run invar binary");
    let json = serde_json::from_slice(&output.stdout).unwrap_or(serde_json::Value::Null);
    (
        output.status.code(),
        json,
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[test]
fn issue_210_trig_option_is_accepted_by_run_and_explain_and_recorded() {
    let path = std::env::temp_dir().join(format!("issue_210_trig_{}.inv", std::process::id()));
    std::fs::write(&path, source()).unwrap();
    let path = path.to_str().unwrap();
    // The default already runs trig-f32-v1, so both give the same receipt.
    for option in [&[][..], &["--numeric-contract", "trig-f32-v1"][..]] {
        for engine in [
            &["--engine", "reference"][..],
            &["--engine", "cpu-parallel", "--workers", "2"][..],
        ] {
            let mut args = vec!["run", path];
            args.extend_from_slice(option);
            args.extend_from_slice(engine);
            args.extend_from_slice(&["--format", "json"]);
            let (code, json, stderr) = cli(&args);
            assert_eq!(code, Some(0), "{args:?}: {json} {stderr}");
            assert_eq!(json["result"]["value"], 1, "{args:?}: {json}");
            assert_eq!(
                json["numeric_contracts"]["trig"], "invar-trig-f32-v1",
                "{json}"
            );
        }
        let mut args = vec!["explain", path];
        args.extend_from_slice(option);
        args.extend_from_slice(&["--format", "json"]);
        let (code, json, stderr) = cli(&args);
        assert_eq!(code, Some(0), "{args:?}: {stderr}");
        assert_eq!(
            json["numeric_contracts"]["trig"], "invar-trig-f32-v1",
            "{json}"
        );
    }
    // Any other value, or a missing one, is refused before the program runs.
    for value in [
        &["--numeric-contract", "math-cr-v1"][..],
        &["--numeric-contract"][..],
    ] {
        let mut args = vec!["run", path, "--engine", "reference"];
        args.extend_from_slice(value);
        let (code, _, stderr) = cli(&args);
        assert_eq!(code, Some(2), "{args:?}");
        assert!(stderr.contains("NUMERIC_CONTRACT"), "{stderr}");
    }
    std::fs::remove_file(path).ok();
}

#[test]
fn issue_210_trig_option_names_verify_only_on_f32() {
    math::select_trig_f32_v1();
    // f64 `sin` keeps `sin` (cr-v1); renamed to the f32 helper it must not verify.
    let source = "fn main() -> i32 { let d: f64 = sin(f64_from_u32(3u32)); if d > 0.14 { return 1; } return 0; }";
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let mut program = mir::lower(&checked).unwrap();
    assert!(mir::verify(&program).is_ok());
    let mut renamed = 0;
    for instruction in program
        .functions
        .iter_mut()
        .flat_map(|function| &mut function.blocks)
        .flat_map(|block| &mut block.instructions)
    {
        if let mir::Operation::Call { name, .. } = &mut instruction.kind
            && name == "sin"
        {
            *name = math::TRIG_F32_SIN.to_owned();
            renamed += 1;
        }
    }
    assert_eq!(renamed, 1);
    assert!(mir::verify(&program).is_err());
}
