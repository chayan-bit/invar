//! v0.2 reductions through the real `invar` CLI and the library run paths:
//! every engine built into this binary returns the oracle bits, traps and
//! fuel agree, and the checker rejects malformed calls.

#[path = "support/fsum_cases.rs"]
mod cases;
#[path = "support/fsum_source.rs"]
mod source;

use cases::{Rng, adversarial_vectors};
use invar::{check, eval, front, fsum, mir, mir_eval, schedule};
use std::{path::PathBuf, process::Command};

const FUEL: u64 = 50_000_000;

fn expected(call: &str, a: &[u32], b: &[u32]) -> u32 {
    match call {
        "sum_tree(A)" => fsum::sum_tree(a),
        "sum_exact(A)" => fsum::sum_exact(a),
        "dot_tree(A, B)" => fsum::dot_tree(a, b),
        _ => fsum::dot_exact(a, b),
    }
}

/// Every reduction of every adversarial vector up to `limit` elements.
fn corpus(limit: usize) -> Vec<(String, Vec<u32>, Vec<u32>, u32)> {
    let mut rng = Rng(0xc1);
    let mut vectors = adversarial_vectors()
        .into_iter()
        .filter(|(_, a, _)| a.len() <= limit)
        .map(|(_, a, b)| (a, b))
        .collect::<Vec<_>>();
    for n in [2usize, 33, 257] {
        vectors.push((
            (0..n).map(|_| rng.word()).collect(),
            (0..n).map(|_| rng.word()).collect(),
        ));
    }
    let calls = [
        "sum_tree(A)",
        "sum_exact(A)",
        "dot_tree(A, B)",
        "dot_exact(A, B)",
    ];
    vectors
        .iter()
        .flat_map(|(a, b)| {
            calls
                .iter()
                .map(|call| (call.to_string(), a.clone(), b.clone(), expected(call, a, b)))
        })
        .collect()
}

fn write(name: &str, text: &str) -> PathBuf {
    let directory = std::env::temp_dir().join(format!("v02-fsum-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join(name);
    std::fs::write(&path, text).unwrap();
    path
}

/// Metal runs only compiled-ahead libraries, so a forced Metal run
/// first builds the program's library with the real `invar build`.
fn run(path: &PathBuf, engine: &[&str]) -> serde_json::Value {
    let mut command = Command::new(env!("CARGO_BIN_EXE_invar"));
    command.arg("run").arg(path).args(engine);
    if engine.contains(&"metal") {
        let library = path.with_extension("metallib");
        let built = Command::new(env!("CARGO_BIN_EXE_invar"))
            .arg("build")
            .arg(path)
            .args(["--target", "metal", "--output"])
            .arg(&library)
            .output()
            .unwrap();
        assert!(
            built.status.success(),
            "{}",
            String::from_utf8_lossy(&built.stdout)
        );
        command.arg("--artifact").arg(&library);
    }
    let output = command
        .args(["--fuel", &FUEL.to_string(), "--format", "json"])
        .output()
        .unwrap();
    serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|_| panic!("{engine:?}: {}", String::from_utf8_lossy(&output.stderr)))
}

/// The engines this build of `invar` can run.
fn engines() -> Vec<Vec<&'static str>> {
    let mut engines = vec![vec!["--engine", "reference"]];
    for workers in ["1", "2", "3", "4"] {
        engines.push(vec!["--engine", "cpu-parallel", "--workers", workers]);
    }
    if cfg!(feature = "native") {
        engines.push(vec!["--engine", "native-serial"]);
    }
    if cfg!(all(feature = "metal", target_os = "macos")) {
        engines.push(vec!["--engine", "metal"]);
    }
    if cfg!(feature = "cuda") {
        engines.push(vec!["--engine", "cuda"]);
    }
    if cfg!(any(
        feature = "cuda",
        all(feature = "metal", target_os = "macos")
    )) {
        engines.push(vec!["--engine", "auto"]);
    }
    engines
}

#[test]
fn v02_fsum_cli_every_engine_returns_oracle_bits() {
    let text = source::program(&corpus(300));
    let path = write("corpus.inv", &text);
    for engine in engines() {
        let json = run(&path, &engine);
        assert_eq!(json["status"], "ok", "{engine:?}: {json}");
        assert_eq!(
            json["result"]["value"], 0,
            "{engine:?}: first mismatch {json}"
        );
    }
}

/// A program that also sorts runs through the sort engines, which pass
/// the fsum reducer on CPU. Its result must equal the fsum-only
/// program's on every engine.
#[test]
fn v02_fsum_cli_sorting_program_matches_the_fsum_only_result() {
    let fsum_only = source::program(&corpus(40));
    let prefix = "fn main() -> i32 {\n";
    assert!(fsum_only.starts_with(prefix));
    let sorting = fsum_only.replacen(
        prefix,
        "fn main() -> i32 {\n    var keys: Array<u32> = [3u32, 1u32, 2u32];\n    sort(keys);\n    if keys[0u32] != 1u32 { return 1000000; }\n",
        1,
    );
    let (fsum_path, sort_path) = (
        write("fsum-only.inv", &fsum_only),
        write("sort-fsum.inv", &sorting),
    );
    for engine in engines() {
        let (alone, with_sort) = (run(&fsum_path, &engine), run(&sort_path, &engine));
        assert_eq!(alone["status"], "ok", "{engine:?}: {alone}");
        assert_eq!(alone["result"]["value"], 0, "{engine:?}: {alone}");
        assert_eq!(
            with_sort["result"], alone["result"],
            "{engine:?}: {with_sort}"
        );
    }
}

#[test]
fn v02_fsum_cli_dot_length_mismatch_traps_at_the_call() {
    for call in ["dot_tree", "dot_exact"] {
        let text = format!(
            "fn main() -> i32 {{ let a: Array<f32> = [1.0, 2.0]; let b: Array<f32> = [1.0]; let r = {call}(a, b); return 0; }}"
        );
        let start = text.find(call).unwrap();
        let end = start + call.len() + "(a, b)".len();
        let path = write(&format!("{call}-mismatch.inv"), &text);
        for engine in engines() {
            let json = run(&path, &engine);
            assert_eq!(json["status"], "trap", "{engine:?}: {json}");
            assert_eq!(json["trap"]["category"], "INDEX_OUT_OF_BOUNDS", "{json}");
            assert_eq!(json["trap"]["span"]["start"], start, "{engine:?}: {json}");
            assert_eq!(json["trap"]["span"]["end"], end, "{engine:?}: {json}");
        }
    }
}

#[test]
fn v02_fsum_checker_rejects_malformed_calls() {
    for (body, code) in [
        (
            "let a: Array<i32> = [1]; let r = sum_tree(a);",
            "TYPE_MISMATCH",
        ),
        (
            "let a: Array<f32> = [1.0]; let r = sum_tree(a, a);",
            "TYPE_MISMATCH",
        ),
        (
            "let a: Array<f32> = [1.0]; let r = dot_exact(a);",
            "TYPE_MISMATCH",
        ),
        ("let x: f32 = 1.0; let r = sum_exact(x);", "TYPE_MISMATCH"),
        ("let r = sum_tree(filled_f32(2u32, 1.0));", "TYPE_MISMATCH"),
        (
            "let a: Array<f32> = [1.0]; let r: i32 = sum_tree(a);",
            "TYPE_MISMATCH",
        ),
    ] {
        let text = format!("fn main() -> i32 {{ {body} return 0; }}");
        let path = write("reject.inv", &text);
        let output = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args([
                "check".as_ref(),
                path.as_os_str(),
                "--format".as_ref(),
                "json".as_ref(),
            ])
            .output()
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(output.status.code(), Some(1), "{body}: {json}");
        assert_eq!(json["diagnostics"][0]["code"], code, "{body}: {json}");
    }
    let redefined = "fn sum_tree(x: i32) -> i32 { return x; } fn main() -> i32 { return 0; }";
    let parsed = front::parse(redefined.as_bytes()).unwrap();
    assert!(
        check::check(parsed, true).is_err(),
        "reduction names are reserved"
    );
}

fn lowered(text: &str) -> (check::CheckedProgram, mir::Program) {
    let checked = check::check(front::parse(text.as_bytes()).unwrap(), true).unwrap();
    invar::ownership::check(&checked).unwrap();
    let program = mir::lower(&checked).unwrap();
    (checked, program)
}

/// Reference and MIR agree on value, fuel at every prefix, and cleanup; a
/// read loan lets the owner be mutated and reduced again afterwards.
#[test]
fn v02_fsum_reference_and_mir_agree_on_fuel_and_cleanup() {
    let text = "fn both(xs: read Array<f32>) -> f32 { return sum_exact(xs) + sum_tree(xs); }
fn main() -> i32 {
    var a: Array<f32> = [1.0, 0.5, -0.25];
    let b: Array<f32> = [2.0, 2.0, 2.0];
    let first = both(a) + dot_tree(a, b) + dot_exact(a, a);
    a[0u32] = 3.0;
    let second = both(a) + dot_exact(a, b);
    if first == 6.3125 and second == 13.0 { return 1; }
    return 0;
}";
    let (checked, program) = lowered(text);
    let complete = eval::observe(&checked, FUEL);
    assert_eq!(complete.result, Ok(eval::Value::I32(1)));
    for fuel in 1..=complete.steps_used + 1 {
        let reference = eval::observe(&checked, fuel);
        let actual = mir_eval::observe(&program, fuel);
        assert_eq!(actual.result, reference.result, "fuel={fuel}");
        assert_eq!(actual.steps_used, reference.steps_used, "fuel={fuel}");
        assert_eq!(actual.cleanup_trace, reference.cleanup_trace, "fuel={fuel}");
    }
    let parallel = schedule::run_source(text.as_bytes(), &program, FUEL, 3).unwrap();
    assert_eq!(parallel.observation.result, complete.result);
    assert_eq!(parallel.observation.steps_used, complete.steps_used);
}

#[test]
fn v02_fsum_ownership_rejects_conflicting_loans() {
    for body in [
        "var a: Array<f32> = [1.0]; let b = move a; let r = sum_tree(a);",
        "let a: Array<f32> = [1.0]; let r = sum_tree(move a);",
    ] {
        let text = format!("fn main() -> i32 {{ {body} return 0; }}");
        let rejected = check::check(front::parse(text.as_bytes()).unwrap(), true)
            .and_then(|checked| invar::ownership::check(&checked).map(|_| ()));
        assert!(rejected.is_err(), "{body}");
    }
}
