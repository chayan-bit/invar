//! #50 part 2: a placement run records the float reduction receipts and the
//! sort and spatial sites the established auto path records, in its placement
//! receipt and in the explain device choices, with the same value.

use serde_json::{Value, json};
use std::{path::Path, process::Command};

const FSUM: &str = "fn main() -> i32 { let a: Array<f32> = filled_f32(70000u32, 0.5); let b: Array<f32> = filled_f32(70000u32, 2.0); let s = sum_tree(a); let d = dot_exact(a, b); if s != 35000.0 or d != 70000.0 { return 1; } var xs: Array<u32> = [0u32, 1u32]; for i in 0u32..len(xs) { xs[i] = wrapping_add(xs[i], 1u32); } return 42; }";
const SORT: &str = "fn main() -> i32 { var keys: Array<u32> = [3u32, 1u32, 2u32]; sort(keys); let order = argsort(keys); if keys[0u32] != 1u32 or order[0u32] != 0u32 { return 1; } return 42; }";
const SPATIAL: &str = "fn main() -> i32 { let p: Array<f32> = [0.0, 1.0]; let grid = voxel_grid_build(p, p, p, 1.0); let q = filled_f32(3u32, 0.0); let near = radius_neighbors(grid, q, q, q, 1.0, 1u32); var keys: Array<u32> = [2u32, 1u32]; sort(keys); return 42; }";

/// Established sort and spatial choices on every build; trusted placement.
const TRUSTED: [&str; 5] = [
    "--host-class",
    "issue-216",
    "--profile",
    "profile.json",
    "--trust-provisional",
];

fn schema(name: &str) -> jsonschema::Validator {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("schema")
        .join(name);
    jsonschema::validator_for(&serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap())
        .unwrap()
}

fn auto(dir: &Path, command: &str, extra: &[&str]) -> Value {
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .current_dir(dir)
        .args([command, "main.inv", "--engine", "auto", "--format", "json"])
        .args(extra)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    serde_json::from_slice(&output.stdout).unwrap()
}

/// A directory holding `source` and a provisional CPU-only profile of this
/// host, trusted explicitly, so the profiled host runs on every build.
fn placed(name: &str, source: &str) -> std::path::PathBuf {
    let dir =
        std::env::temp_dir().join(format!("invar-50-placement-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("main.inv"), source).unwrap();
    let host = auto(&dir, "run", &["--host-class", "issue-216"]);
    let c = json!({"fixed_ms":0.0,"per_work_ms":1.0,"link_bytes_per_ms":1000.0,
        "link_latency_ms":0.0,"launch_floor_ms":0.0,"page_in_ms_per_byte":null});
    let profile = json!({"schema_version":1,"probe_version":"placement-v1","units":"ms,bytes,work",
        "identity":host["placement"]["resource_identity"],"provisional":true,
        "source_revision":"0123456789012345678901234567890123456789",
        "observations":{"before":{"load":null,"swap_bytes":null},"after":{"load":null,"swap_bytes":null}},
        "refusals":[],"measurements":[{"family":"placement.wrapping_add1.u32","numeric":"u32-wrapping",
            "numeric_contract_version":"wrapping-u32-v1","engine":"cpu","boundary":"llvm-host-array-v1",
            "region":null,"sizes":[1,10000],"passes":[c,c,c,c,c],"best":c}]});
    std::fs::write(dir.join("profile.json"), profile.to_string()).unwrap();
    dir
}

/// Reduction choices without costs, which the established path may measure.
fn fsum(choices: &Value) -> Vec<Value> {
    let fields = ["function", "span", "kind", "backend", "reason"];
    choices
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["kind"].as_str().is_some_and(|k| k.starts_with("fsum.")))
        .map(|c| fields.iter().map(|f| c[f].clone()).collect())
        .collect()
}

/// Sort and spatial site choices: they name an implementation or a family.
fn sites(choices: &Value) -> Vec<Value> {
    let fields = [
        "function",
        "span",
        "kind",
        "backend",
        "reason",
        "implementation",
        "family_id",
    ];
    choices
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["implementation"].is_string() || c["family_id"].is_string())
        .map(|c| fields.iter().map(|f| c[f].clone()).collect())
        .collect()
}

fn valid(run: &Value, explain: &Value, receipt: &str) {
    for (validator, document) in [
        (schema("invar-diagnostics-v3.schema.json"), run),
        (schema("explain-v5.schema.json"), explain),
    ] {
        let errors: Vec<_> = validator
            .iter_errors(document)
            .map(|e| e.to_string())
            .collect();
        assert!(errors.is_empty(), "{errors:?}\n{document}");
        let mut bad = document.clone();
        bad["placement"][receipt][0]["invented"] = json!(1);
        assert!(!validator.is_valid(&bad));
        // The reference route must carry both lists, even empty ones.
        let mut missing = document.clone();
        missing["placement"]
            .as_object_mut()
            .unwrap()
            .remove(receipt);
        assert!(!validator.is_valid(&missing));
    }
}

#[test]
fn placement_runs_record_the_established_reduction_receipts() {
    let dir = placed("fsum", FSUM);
    let (run, explain) = (auto(&dir, "run", &TRUSTED), auto(&dir, "explain", &TRUSTED));
    let established = auto(&dir, "explain", &[]);
    std::fs::remove_dir_all(dir).unwrap();
    assert_eq!(established["auto_path"], "established");
    assert_eq!(run["result"]["value"], 42, "{run}");
    assert_eq!(run["placement"]["route"]["host"], "reference", "{run}");
    let receipts = fsum(&run["placement"]["reductions"]);
    assert_eq!(receipts.len(), 2, "{run}");
    assert_eq!(fsum(&explain["placement"]["reductions"]), receipts);
    assert_eq!(fsum(&explain["device_execution"]["choices"]), receipts);
    // Placement does not price reductions yet (#49): they run on the
    // established CPU path's workers and cost model, never on a device.
    let reasons = [
        "measured-cpu-cost",
        "cuda-feature-unavailable",
        "device-unavailable",
    ];
    assert!(
        receipts
            .iter()
            .all(|r| r[3].as_str().unwrap().starts_with("cpu-")
                && reasons.contains(&r[4].as_str().unwrap())),
        "{receipts:?}"
    );
    // Without a device the receipts equal the established auto run's.
    if !cfg!(any(
        feature = "cuda",
        all(feature = "metal", target_os = "macos")
    )) {
        assert_eq!(fsum(&established["device_execution"]["choices"]), receipts);
    }
    valid(&run, &explain, "reductions");
}

#[test]
fn placement_runs_record_the_established_sort_and_spatial_sites() {
    for (name, source, expected) in [("sort", SORT, 2), ("spatial", SPATIAL, 3)] {
        let dir = placed(name, source);
        let (run, explain) = (auto(&dir, "run", &TRUSTED), auto(&dir, "explain", &TRUSTED));
        let established = auto(&dir, "explain", &[]);
        std::fs::remove_dir_all(dir).unwrap();
        assert_eq!(run["result"]["value"], 42, "{run}");
        assert_eq!(run["placement"]["route"]["host"], "reference", "{run}");
        let receipt = run["placement"]["intrinsics"].as_array().unwrap();
        assert_eq!(receipt.len(), expected, "{name}: {run}");
        // The explain lists the same sites as the established explain, with
        // the same backends, reasons, implementations and spatial families.
        let placed = sites(&explain["device_execution"]["choices"]);
        assert_eq!(placed.len(), expected, "{name}: {explain}");
        assert_eq!(
            placed,
            sites(&established["device_execution"]["choices"]),
            "{name}"
        );
        valid(&run, &explain, "intrinsics");
    }
}

/// The native route (an untrusted profile on a native build) runs compiled
/// code with no per-call hooks: its receipt names the route and carries
/// neither list, so it never reads as a run that reached no sites.
#[cfg(feature = "native")]
#[test]
fn native_placement_runs_name_their_route_and_record_no_site_lists() {
    let dir = placed("native", FSUM);
    let untrusted = ["--host-class", "issue-216", "--profile", "profile.json"];
    let run = auto(&dir, "run", &untrusted);
    std::fs::remove_dir_all(dir).unwrap();
    assert_eq!(run["result"]["value"], 42, "{run}");
    assert_eq!(run["placement"]["route"]["host"], "native", "{run}");
    for list in ["reductions", "intrinsics"] {
        assert!(run["placement"].get(list).is_none(), "{run}");
    }
    // The schema forbids either list on the native route.
    let validator = schema("invar-diagnostics-v3.schema.json");
    assert!(validator.is_valid(&run), "{run}");
    let mut listed = run.clone();
    listed["placement"]["reductions"] = json!([]);
    assert!(!validator.is_valid(&listed));
}

/// Metal has no binary64: placement names each f64 function `metal-no-f64` on
/// `mir-serial`, exactly as the established Metal explain does.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn placement_runs_record_metal_no_f64_like_the_established_path() {
    const F64: &str = "fn grow(x: f64) -> f64 { return x * 2.0f64; } fn main() -> i32 { let y = grow(1.5f64); if y > 2.0f64 { return 42; } return 9; }";
    let dir = placed("f64", F64);
    let explain = auto(&dir, "explain", &TRUSTED);
    let established = auto(&dir, "explain", &[]);
    std::fs::remove_dir_all(dir).unwrap();
    let f64_sites = |document: &Value| -> Vec<Value> {
        document["device_execution"]["choices"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|c| c["reason"] == "metal-no-f64")
            .cloned()
            .collect()
    };
    let placed = f64_sites(&explain);
    assert_eq!(placed.len(), 2, "{explain}");
    assert_eq!(placed, f64_sites(&established));
    valid(&explain["execution"], &explain, "intrinsics");
}
