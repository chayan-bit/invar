#![allow(dead_code)]
use invar::{check, front, mir};
use std::{path::Path, process::Command};

#[path = "m4_04_cases.rs"]
#[allow(dead_code)]
pub mod cases;

pub fn source(case: &cases::Case) -> String {
    let literals = |values: &[u32]| {
        values
            .iter()
            .map(|v| format!("{v}u32"))
            .collect::<Vec<_>>()
            .join(",")
    };
    let expected = cases::sequential_expected(case);
    let args = match case.kind {
        cases::Kind::Compact(_) => format!("xs, {}u32, {}u32", case.input.len(), case.cutoff),
        _ => format!("xs, offsets, {}u32", case.offsets.len() - 1),
    };
    format!(
        "{} fn main() -> i32 {{ let xs: Array<u32> = [{}]; let offsets: Array<u32> = [{}]; let expected: Array<u32> = [{}]; let out = run({args}); if len(out) != len(expected) {{ return 1; }} for verify in 0u32..len(out) {{ if out[verify] != expected[verify] {{ return 2; }} }} return 0; }}",
        cases::function(case.kind),
        literals(&case.input),
        literals(&case.offsets),
        literals(&expected)
    )
}

pub fn program(source: &str) -> mir::Program {
    mir::lower(&check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap()).unwrap()
}

pub fn cli(engine: &str, path: &Path) -> serde_json::Value {
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .env("INVAR_METAL_RUNTIME_SOURCE", "1")
        .arg("run")
        .arg(path)
        .args(["--engine", engine, "--format", "json"])
        .output()
        .unwrap();
    serde_json::from_slice(&output.stdout).unwrap_or_else(|_| {
        panic!(
            "CLI stdout: {} stderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    })
}
