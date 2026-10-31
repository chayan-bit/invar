//! v0.2 f64math acceptance: Panda forward kinematics in f64 reproduces the
//! S1-06 Pinocchio 4.1.0 golden poses within 1e-12 through the real CLI.
use std::process::Command;

const EXAMPLE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/fk_panda.inv");

fn run(engine: &[&str]) -> serde_json::Value {
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["run", EXAMPLE, "--format", "json", "--fuel", "100000000"])
        .args(engine)
        .output()
        .expect("run invar binary");
    let json: serde_json::Value = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|_| panic!("{}", String::from_utf8_lossy(&output.stderr)));
    assert_eq!(output.status.code(), Some(0), "{engine:?}: {json}");
    json
}

#[test]
fn fk_panda_matches_pinocchio_on_every_cpu_engine() {
    let mut engines = vec![
        vec!["--engine", "reference"],
        vec!["--engine", "cpu-parallel", "--workers", "2"],
    ];
    if cfg!(feature = "native") {
        engines.push(vec!["--engine", "native-serial"]);
    }
    for engine in engines {
        let json = run(&engine);
        assert_eq!(json["result"]["value"], 4, "{engine:?}: {json}");
        assert_eq!(json["numeric_contracts"], contracts(), "{engine:?}");
        valid("invar-diagnostics-v3.schema.json", &json);
    }
    let explain = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["explain", EXAMPLE, "--format", "json"])
        .output()
        .unwrap();
    let explain: serde_json::Value = serde_json::from_slice(&explain.stdout).unwrap();
    assert_eq!(explain["numeric_contracts"], contracts(), "{explain}");
    valid("explain-v5.schema.json", &explain);
}

/// The receipt fields this lane adds are part of the published schemas.
fn valid(schema: &str, document: &serde_json::Value) {
    let path = format!("{}/schema/{schema}", env!("CARGO_MANIFEST_DIR"));
    let schema: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let validator = jsonschema::validator_for(&schema).unwrap();
    let errors: Vec<String> = validator
        .iter_errors(document)
        .map(|e| e.to_string())
        .collect();
    assert!(errors.is_empty(), "{errors:?}\n{document}");
}

/// Metal has no binary64: auto placement names every f64 function with reason
/// `metal-no-f64` on `mir-serial` and returns the CPU result, while a forced
/// Metal run of a program with no f32 or integer region refuses, as every
/// forced device run without an eligible region does.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn f64_on_metal_runs_on_the_cpu_with_metal_no_f64() {
    let source = "fn grow(x: f64) -> f64 { return sqrt(x) * 3.0; }\n\
                  fn main() -> i32 { var s: f64 = 0.0; for i in 0u32..4u32 { s = s + grow(16.0); } return i32_from_f64(s); }\n";
    let path = std::env::temp_dir().join(format!("invar-metal-f64-{}.inv", std::process::id()));
    std::fs::write(&path, source).unwrap();
    let explain_json = |engine: &str| -> serde_json::Value {
        let output = Command::new(env!("CARGO_BIN_EXE_invar"))
            .arg("explain")
            .arg(&path)
            .args(["--engine", engine, "--format", "json"])
            .output()
            .unwrap();
        serde_json::from_slice(&output.stdout)
            .unwrap_or_else(|_| panic!("{}", String::from_utf8_lossy(&output.stderr)))
    };
    let forced = explain_json("metal");
    assert_eq!(
        forced["diagnostics"][0]["code"], "METAL_UNAVAILABLE",
        "{forced}"
    );
    {
        let explain = explain_json("auto");
        let device = &explain["device_execution"];
        assert_eq!(device["result"], 48, "{explain}");
        let sites: Vec<_> = device["choices"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|c| c["reason"] == "metal-no-f64")
            .collect();
        let functions: Vec<_> = sites
            .iter()
            .map(|c| c["function"].as_str().unwrap())
            .collect();
        assert_eq!(functions, ["grow", "main"], "{explain}");
        assert!(
            sites.iter().all(|c| c["backend"] == "mir-serial"),
            "{explain}"
        );
        valid("explain-v5.schema.json", &explain);
    }
    std::fs::remove_file(path).unwrap();
}

fn contracts() -> serde_json::Value {
    serde_json::json!({"binary64": "strict-f64-v1", "math": "invar-math-cr-v1"})
}

#[test]
fn fk_panda_detects_a_perturbed_link_offset() {
    let source = std::fs::read_to_string(EXAMPLE).unwrap();
    let mutated = source.replacen(
        "0.0825, 0.0, 0.0, q[3u32]",
        "0.0825000001, 0.0, 0.0, q[3u32]",
        1,
    );
    assert_ne!(mutated, source);
    let checked =
        invar::check::check(invar::front::parse(mutated.as_bytes()).unwrap(), true).unwrap();
    let result = invar::eval::run(&checked, 100_000_000).unwrap();
    assert!(
        matches!(result.value, invar::eval::Value::I32(0)),
        "{:?}",
        result.value
    );
}
