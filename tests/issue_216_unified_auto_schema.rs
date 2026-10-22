//! #216 parts 1 and 4: one auto version. Every `--engine auto` explain is
//! `invar-explain-v5` (with `invar-explain-device-v5`) and every auto run
//! receipt `invar-diagnostics-v3`; the established path records
//! `auto_path: "established"`, normal-form route included, and only
//! `auto_path: "placement"` carries the placement receipt.

use serde_json::{Value, json};
use std::{path::PathBuf, process::Command};

const SUM: &str = "fn f(xs: read Array<i32>, n: u32, s0: i32) -> i32 { var s = s0; for i in 0u32 .. n { s = s + xs[i]; } return s; } fn main() -> i32 { let xs: Array<i32> = [4, 9, (0 - 3), 12]; return f(xs, 4u32, 20); }";
/// A certified loop no family lowers: the typed `no-lowering` fallback (#263).
const GAP: &str = "fn run(a: read Array<u32>, n: u32) -> u32 { var s = 0u32; for i in 0u32..n { s = s + a[i]; } return s; } fn main() -> i32 { let a = filled_u32(8u32, 2u32); var out = filled_u32(8u32, 0u32); for i in 0u32..8u32 { out[i] = wrapping_add(a[i], 3u32); } if run(out, 8u32) != 40u32 { return 1; } return 42; }";
const SORT: &str =
    "fn main() -> i32 { var keys: Array<u32> = [2u32, 1u32]; sort(keys); return 42; }";
const TRAP: &str =
    "fn main() -> i32 { let xs: Array<i32> = [2147483647, 1]; return xs[0u32] + xs[1u32]; }";
const SPATIAL: &str = "fn main() -> i32 { let p: Array<f32> = [0.0]; let grid = voxel_grid_build(p, p, p, 1.0); let q = filled_f32(3u32, 0.0); let near = radius_neighbors(grid, q, q, q, 1.0, 1u32); return 42; }";

fn schema(name: &str) -> jsonschema::Validator {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("schema")
        .join(name);
    jsonschema::validator_for(&serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap())
        .unwrap()
}

/// `invar run|explain --engine auto --format json` from a directory with no
/// calibration profile, so the established path answers.
fn auto(name: &str, source: &str, command: &str) -> (Option<i32>, Value) {
    let dir = std::env::temp_dir().join(format!("invar-216-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(name), source).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .current_dir(&dir)
        .args([command, name, "--engine", "auto", "--format", "json"])
        .env_remove("INVAR_NO_NORMAL_FORM")
        .output()
        .unwrap();
    let document = serde_json::from_slice(&output.stdout).unwrap_or_else(|_| panic!("{output:?}"));
    (output.status.code(), document)
}

fn established(name: &str, source: &str, command: &str) -> Value {
    let (status, document) = auto(name, source, command);
    assert_eq!(status, Some(0), "{document}");
    assert_eq!(document["auto_path"], "established", "{document}");
    let (schema, version) = match command {
        "explain" => ("invar-explain-v5", 5),
        _ => ("invar-diagnostics-v3", 3),
    };
    assert_eq!(document["schema"], schema, "{document}");
    assert_eq!(document["schema_version"], version, "{document}");
    assert!(document.get("placement").is_none(), "{document}");
    document
}

#[test]
fn established_auto_documents_fit_the_unified_versions() {
    let (explain, device, run) = (
        schema("explain-v5.schema.json"),
        schema("invar-explain-device-v5.schema.json"),
        schema("invar-diagnostics-v3.schema.json"),
    );
    for (name, source) in [
        ("sum.inv", SUM),
        ("gap.inv", GAP),
        ("sort.inv", SORT),
        ("spatial.inv", SPATIAL),
    ] {
        let document = established(name, source, "explain");
        let errors: Vec<_> = explain
            .iter_errors(&document)
            .map(|e| e.to_string())
            .collect();
        assert!(errors.is_empty(), "{name}: {errors:?}\n{document}");
        assert_eq!(document["device_schema"], "invar-explain-device-v5");
        let errors: Vec<_> = device
            .iter_errors(&document["device_execution"])
            .map(|e| e.to_string())
            .collect();
        assert!(errors.is_empty(), "{name}: {errors:?}");
        let receipt = established(name, source, "run");
        let errors: Vec<_> = run.iter_errors(&receipt).map(|e| e.to_string()).collect();
        assert!(errors.is_empty(), "{name}: {errors:?}\n{receipt}");
        #[cfg(not(any(feature = "cuda", all(feature = "metal", target_os = "macos"))))]
        if name == "sum.inv" {
            assert_eq!(
                document["device_execution"]["normal_form"]["status"],
                "routed"
            );
            assert_eq!(receipt["normal_form"]["status"], "routed");
        }
        #[cfg(not(any(feature = "cuda", all(feature = "metal", target_os = "macos"))))]
        if name == "gap.inv" {
            let fallback = json!({"status": "fallback", "reason": "no-lowering:reduce/lifted-sum"});
            assert_eq!(document["device_execution"]["normal_form"], fallback);
            assert_eq!(receipt["normal_form"], fallback);
        }
    }
}

#[test]
fn auto_path_decides_whether_the_placement_receipt_is_present() {
    let (explain, run) = (
        schema("explain-v5.schema.json"),
        schema("invar-diagnostics-v3.schema.json"),
    );
    let document = established("sum.inv", SUM, "explain");
    let receipt = established("sum.inv", SUM, "run");
    let placement = json!({"profile_id":null,"profile_schema_version":null,"resource_identity":{},
        "trust_provisional":false,"engine":"cpu","choices":[]});
    let refused = |validator: &jsonschema::Validator, mut doc: Value, edit: &dyn Fn(&mut Value)| {
        edit(&mut doc);
        assert!(!validator.is_valid(&doc), "{doc}");
    };
    // An established document carries no placement receipt or execution.
    refused(&explain, document.clone(), &|d| {
        d["placement"] = placement.clone()
    });
    refused(&explain, document.clone(), &|d| {
        d["execution"] = receipt.clone()
    });
    refused(&run, receipt.clone(), &|d| {
        d["placement"] = placement.clone()
    });
    // A placement document must carry them.
    refused(&explain, document.clone(), &|d| {
        d["auto_path"] = json!("placement")
    });
    refused(&run, receipt.clone(), &|d| {
        d["auto_path"] = json!("placement")
    });
    // Only the two paths exist, and the field is required.
    refused(&explain, document.clone(), &|d| {
        d["auto_path"] = json!("cuda")
    });
    refused(&run, receipt.clone(), &|d| {
        d.as_object_mut().unwrap().remove("auto_path");
    });
}

#[test]
fn established_auto_failures_are_unified_receipts() {
    let run = schema("invar-diagnostics-v3.schema.json");
    for command in ["run", "explain"] {
        let (status, document) = auto("trap.inv", TRAP, command);
        assert_eq!(status, Some(3), "{command}: {document}");
        assert_eq!(document["schema"], "invar-diagnostics-v3", "{document}");
        assert_eq!(document["auto_path"], "established", "{document}");
        assert_eq!(
            document["trap"]["category"], "INTEGER_OVERFLOW",
            "{document}"
        );
        let errors: Vec<_> = run.iter_errors(&document).map(|e| e.to_string()).collect();
        assert!(errors.is_empty(), "{command}: {errors:?}\n{document}");
    }
}
