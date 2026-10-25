#![forbid(unsafe_code)]
//! ENG-10 (#56): the held-out domain suites (`bench/domains`). Every program
//! checks, runs on the reference evaluator, on `cpu-parallel` with 1, 2, 7 and
//! 64 workers, and (with the `native` feature) on `native-serial`, and returns
//! 0 only when its outputs equal the Rust oracle bit for bit.
use invar::{
    engine::{self, Engine},
    eval::{self, Value as EvalValue},
};
use std::collections::BTreeMap;
#[path = "../../bench/domains/mod.rs"]
mod domains;
use serde_json::{Value, json};

const FUEL: u64 = 400_000_000;
const WORKERS: [usize; 4] = [1, 2, 7, 64];

fn checked(source: &str) -> invar::check::CheckedProgram {
    invar::front::parse(source.as_bytes())
        .and_then(|program| invar::check::check(program, true))
        .unwrap_or_else(|e| panic!("{source}: {e:?}"))
}

/// Every engine's result for `source`, keyed by engine label.
fn results(source: &str) -> BTreeMap<String, i32> {
    let program = checked(source);
    let mut out = BTreeMap::new();
    let reference = eval::run(&program, FUEL).unwrap_or_else(|e| panic!("reference: {e:?}"));
    let EvalValue::I32(value) = reference.value else {
        panic!("main returned a non-i32");
    };
    out.insert("reference".to_owned(), value);
    let mir = invar::mir::lower(&program).unwrap();
    for workers in WORKERS {
        let receipt = engine::run_mir(
            source.as_bytes(),
            &mir,
            Engine::CpuParallel,
            FUEL,
            workers,
            None,
        )
        .unwrap_or_else(|e| panic!("cpu-parallel {workers}: {e:?}"));
        out.insert(
            format!("cpu-parallel-{workers}"),
            receipt["result"].as_i64().unwrap() as i32,
        );
    }
    match engine::run_native(&mir, FUEL, "suite.inv", source.as_bytes(), None) {
        Ok((_, document)) => {
            out.insert(
                "native-serial".to_owned(),
                document["result"]["value"].as_i64().unwrap() as i32,
            );
        }
        Err(invar::eval::EvalError::Tool {
            code: "NATIVE_UNAVAILABLE",
            ..
        }) if !cfg!(feature = "native") => {}
        Err(e) => panic!("native-serial: {e:?}"),
    }
    out
}

fn on_big_stack(f: impl FnOnce() + Send + 'static) {
    // The reference evaluator recurses; Debug frames need more than a test
    // thread's default stack.
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(f)
        .unwrap()
        .join()
        .unwrap();
}

/// Half of a suite's programs (`shard` 0 or 1), so one test stays short: the
/// smallest and largest shape of each kernel in shard 0, the middle two in 1.
fn check_suite(suite: &'static str, shard: usize) {
    on_big_stack(move || {
        let engines = if cfg!(feature = "native") { 6 } else { 5 };
        let programs = domains::programs(suite).unwrap();
        assert!(programs.len() >= 10, "{suite}: {} programs", programs.len());
        let mut ids = std::collections::BTreeSet::new();
        for (at, program) in programs.iter().enumerate() {
            if usize::from(matches!(at % 4, 1 | 2)) != shard {
                assert!(
                    ids.insert(program.id.clone()),
                    "duplicate id {}",
                    program.id
                );
                continue;
            }
            assert!(
                ids.insert(program.id.clone()),
                "duplicate id {}",
                program.id
            );
            let got = results(&program.source);
            assert_eq!(got.len(), engines, "{}: {got:?}", program.id);
            assert!(
                got.values().all(|&value| value == 0),
                "{}: {got:?}",
                program.id
            );
        }
    });
}

#[test]
fn issue_eng_10_ml_programs_match_their_oracle_on_every_engine_0() {
    check_suite("ml", 0);
}

#[test]
fn issue_eng_10_ml_programs_match_their_oracle_on_every_engine_1() {
    check_suite("ml", 1);
}

#[test]
fn issue_eng_10_games_programs_match_their_oracle_on_every_engine_0() {
    check_suite("games", 0);
}

#[test]
fn issue_eng_10_games_programs_match_their_oracle_on_every_engine_1() {
    check_suite("games", 1);
}

#[test]
fn issue_eng_10_robotics_programs_match_their_oracle_on_every_engine_0() {
    check_suite("robotics", 0);
}

#[test]
fn issue_eng_10_robotics_programs_match_their_oracle_on_every_engine_1() {
    check_suite("robotics", 1);
}

/// `source` with the first expected value of its oracle array made wrong.
fn with_wrong_first_value(source: &str) -> String {
    let marker = "let wanted: Array<";
    let at = source.find(marker).unwrap();
    let open = at + source[at..].find("= [").unwrap() + 3;
    let len = source[open..].find([',', ']']).unwrap();
    let first = &source[open..open + len];
    let wrong = match first.strip_suffix("u32") {
        Some(n) => format!("{}u32", n.parse::<u32>().unwrap() + 1),
        None => domains::f32_literal(first.parse::<f32>().unwrap() + 1.0),
    };
    format!("{}{}{}", &source[..open], wrong, &source[open + len..])
}

#[test]
fn issue_eng_10_the_self_check_is_not_vacuous() {
    on_big_stack(|| {
        for suite in domains::SUITES {
            for program in domains::programs(suite).unwrap() {
                let broken = with_wrong_first_value(&program.source);
                assert_ne!(broken, program.source, "{}", program.id);
                let run = eval::run(&checked(&broken), FUEL).unwrap();
                assert_eq!(run.value, EvalValue::I32(1), "{}", program.id);
            }
        }
    });
}

#[test]
fn issue_eng_10_cr_exp_agrees_with_f64_libm_to_one_ulp() {
    for k in -2000i32..2000 {
        let x = k as f32 * 0.01;
        let got = domains::oracle::exp(x);
        let wide = (x as f64).exp() as f32;
        let ulps = (got.to_bits() as i64 - wide.to_bits() as i64).abs();
        assert!(ulps <= 1, "exp({x}): {got} against {wide}");
    }
}

fn entries<'a>(manifest: &'a mut Value, suite: &str) -> &'a mut Vec<Value> {
    manifest["suites"][suite]["programs"]
        .as_array_mut()
        .unwrap()
}

#[test]
fn issue_eng_10_trig_oracle_stays_within_the_contract_bound_of_f64_libm() {
    for k in -3000i32..3000 {
        let x = k as f32 * 0.01 + 0.003;
        for (got, wide) in [
            (domains::oracle::sin(x), (x as f64).sin() as f32),
            (domains::oracle::cos(x), (x as f64).cos() as f32),
        ] {
            let ulps = (got.to_bits() as i64 - wide.to_bits() as i64).abs();
            assert!(ulps <= 2, "trig({x}): {got} against {wide}");
        }
    }
}

#[test]
fn issue_eng_10_manifest_validation_refuses_every_tampering() {
    let good = domains::manifest::build().unwrap();
    domains::manifest::validate(&good).unwrap();
    assert_eq!(
        good["suites"].as_object().unwrap().len(),
        domains::SUITES.len()
    );
    for suite in domains::SUITES {
        let held = |m: &mut Value| {
            entries(m, suite)
                .iter()
                .position(|e| e["split"] == "held_out")
                .unwrap()
        };
        let counts = &good["suites"][suite]["program_counts"];
        assert!(counts["held_out"].as_u64().unwrap() > 0, "{suite}");
        assert!(counts["tuning"].as_u64().unwrap() > 0, "{suite}");
        let mut cases: Vec<(&str, Value)> = Vec::new();
        let mut m = good.clone();
        entries(&mut m, suite).pop();
        cases.push(("missing", m));
        let mut m = good.clone();
        let first = entries(&mut m, suite)[0].clone();
        entries(&mut m, suite).push(first);
        cases.push(("duplicate", m));
        let mut m = good.clone();
        entries(&mut m, suite).clear();
        cases.push(("zero", m));
        let mut m = good.clone();
        m["suites"][suite]["seed"] = json!(1);
        cases.push(("section seed", m));
        let mut m = good.clone();
        let at = held(&mut m);
        entries(&mut m, suite)[at]["inputs"][0]["seed"] = json!(999);
        cases.push(("held-out input seed", m));
        let mut m = good.clone();
        let at = held(&mut m);
        entries(&mut m, suite)[at]["source_sha256"] = json!("0".repeat(64));
        cases.push(("edited held-out entry", m));
        let mut m = good.clone();
        m["suites"][suite]["held_out_sha256"] = json!("0".repeat(64));
        cases.push(("hash mismatch", m));
        for (what, m) in cases {
            assert!(domains::manifest::validate(&m).is_err(), "{suite}: {what}");
        }
    }
    let mut frozen = good.clone();
    frozen["freeze_status"] = json!("frozen");
    assert!(domains::manifest::validate(&frozen).is_err());
    // The domain suites never enter the M5-01 held-out hash.
    let v1: Value = serde_json::from_str(include_str!("../../bench/manifest-v1.json")).unwrap();
    assert_eq!(good["v1_held_out_sha256"], v1["held_out_sha256"]);
}

fn reformatted(source: &str) -> String {
    source.split_whitespace().collect::<Vec<_>>().join("\n    ")
}

fn renamed(source: &str, from: &str, to: &str) -> String {
    let mut out = String::new();
    let mut rest = source;
    while let Some(at) = rest.find(from) {
        let before = rest[..at].chars().next_back();
        let after = rest[at + from.len()..].chars().next();
        let word = |c: Option<char>| c.is_some_and(|c| c.is_alphanumeric() || c == '_');
        out.push_str(&rest[..at]);
        out.push_str(if word(before) || word(after) {
            from
        } else {
            to
        });
        rest = &rest[at + from.len()..];
    }
    out + rest
}

#[test]
fn issue_eng_10_held_out_programs_are_refused_on_every_tuning_path() {
    for suite in domains::SUITES {
        let (index, tuning) = domains::manifest::tuning_programs(suite).unwrap();
        assert!(index.len() > 0 && !tuning.is_empty(), "{suite}");
        let held: Vec<_> = domains::manifest::split(suite)
            .unwrap()
            .into_iter()
            .filter(|(_, held_out)| *held_out)
            .collect();
        assert!(!held.is_empty(), "{suite}");
        for (program, _) in &held {
            let source = &program.source;
            let renamed_all = renamed(
                &renamed(source, program.kernel, "zz_kernel"),
                "fill",
                "zz_fill",
            );
            let wrapped = renamed(source, "main", "inner_main")
                + "fn main() -> i32 { return inner_main(); }\n";
            let embedded = format!(
                "fn extra_a() -> i32 {{ return 7; }}\n{source}\nfn extra_b(x: u32) -> u32 {{ return x + 1u32; }}\n"
            );
            for (what, candidate) in [
                ("verbatim", source.clone()),
                ("renamed", renamed_all.clone()),
                ("reformatted", reformatted(source)),
                ("renamed and reformatted", reformatted(&renamed_all)),
                ("wrapped", wrapped),
                ("embedded", embedded),
            ] {
                assert!(
                    index.admit(&candidate).is_err(),
                    "{}: {what} was admitted",
                    program.id
                );
            }
        }
        for program in &tuning {
            assert!(index.admit(&program.source).is_ok(), "{}", program.id);
        }
    }
}

#[test]
fn issue_eng_10_no_compiler_file_keys_on_a_suite() {
    let needles = domains::keying::needles().unwrap();
    // The scan covers the compiler (`src/`, `crates/`) next to this package, not the harness.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let found = domains::keying::scan(root, &needles).unwrap();
    assert!(found.is_empty(), "{found:?}");
    // A planted violation is detected, in each rule.
    let dir = std::env::temp_dir().join(format!("invar-eng-10-keying-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("src")).unwrap();
    let first = &domains::programs("ml").unwrap()[0];
    for (what, text) in [
        (
            "kernel name",
            format!(
                "fn f(n: &str) -> u32 {{ match n {{ \"{}\" => 1, _ => 0 }} }}",
                first.kernel
            ),
        ),
        ("program id", format!("// {}", first.id)),
        (
            "fixture path",
            "const P: &str = \"bench/domains/ml\";".to_owned(),
        ),
        (
            "domain word",
            "fn f(d: &str) -> bool { d == \"robotics\" }".to_owned(),
        ),
        (
            "domain word in matches!",
            "fn f(d: &str) -> bool { matches!(d, \"ml\") }".to_owned(),
        ),
        (
            "domain word split across lines",
            "fn f(d: &str) -> bool {\n    d ==\n        \"games\"\n}".to_owned(),
        ),
        (
            "domain word in a table",
            "const T: [&str; 1] = [\"Robotics\"];".to_owned(),
        ),
    ] {
        std::fs::write(dir.join("src/planted.rs"), text).unwrap();
        let found = domains::keying::scan(&dir, &needles).unwrap();
        assert!(!found.is_empty(), "{what} not detected");
    }
    std::fs::create_dir_all(dir.join("src/bin")).unwrap();
    std::fs::write(
        dir.join("src/bin/invar-bench.rs"),
        format!("const K: &str = \"{}\";", first.id),
    )
    .unwrap();
    assert!(!domains::keying::scan(&dir, &needles).unwrap().is_empty());
    std::fs::remove_file(dir.join("src/bin/invar-bench.rs")).unwrap();
    std::fs::write(dir.join("src/planted.rs"), "fn clean() {}").unwrap();
    assert!(domains::keying::scan(&dir, &needles).unwrap().is_empty());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn issue_eng_10_the_coverage_report_accounts_for_every_program() {
    let spawn = |suite: &str| {
        std::process::Command::new(env!("CARGO_BIN_EXE_invar-bench"))
            .args(["--domain-report", suite])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap()
    };
    // The three suites run as concurrent processes to keep the test short.
    let children: Vec<_> = domains::SUITES
        .iter()
        .map(|suite| (*suite, spawn(suite)))
        .collect();
    for (suite, child) in children {
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            report["device_timings"].as_str().unwrap().split(':').next(),
            Some("pending")
        );
        assert_eq!(report["suites"].as_object().unwrap().len(), 1);
        let section = &report["suites"][suite];
        let total = domains::programs(suite).unwrap().len() as u64;
        assert_eq!(section["programs"], json!(total), "{suite}");
        assert_eq!(
            section["correct_on_reference_and_cpu_parallel"],
            json!(total),
            "{suite}"
        );
        let coverage = &section["coverage"];
        let (parallel, serial) = (
            coverage["parallel"].as_u64().unwrap(),
            coverage["cpu_serial"].as_u64().unwrap(),
        );
        assert_eq!(parallel + serial, total, "{suite}");
        // Every serial program states why, so a refusal is never silent.
        for row in section["results"].as_array().unwrap() {
            if row["lowering"] == "cpu-serial" {
                assert!(
                    !row["refusal_reasons"].as_array().unwrap().is_empty(),
                    "{}",
                    row["id"]
                );
            }
        }
        let worst = section["worst_held_out"].as_array().unwrap();
        assert!(!worst.is_empty() && worst.len() <= 5, "{suite}");
        assert!(
            worst.iter().all(|row| row["split"] == "held_out"),
            "{suite}"
        );
    }
    let unknown = std::process::Command::new(env!("CARGO_BIN_EXE_invar-bench"))
        .args(["--domain-report", "nope"])
        .output()
        .unwrap();
    assert!(!unknown.status.success());
}
