use invar::placement::{self, Engine, Facts, PageState, Profile};
use invar::schedule::tuning::Residency;
use serde_json::{Value, json};

const DIAGNOSTICS_V3: &str = include_str!("../../schema/invar-diagnostics-v3.schema.json");

fn coefficients(_cpu: bool, work_ms: f64) -> Value {
    json!({"fixed_ms": 0.0, "per_work_ms": work_ms,
        "link_bytes_per_ms": json!(1000.0),
        "link_latency_ms": 0.0, "launch_floor_ms": 0.0, "page_in_ms_per_byte": null})
}
fn entry(engine: &str, work_ms: f64) -> Value {
    let c = coefficients(engine == "cpu", work_ms);
    json!({"family":"placement.wrapping_add1.u32", "numeric":"u32-wrapping", "numeric_contract_version":"wrapping-u32-v1", "engine":engine,
        "boundary":"retained-v1", "region":null, "sizes":[1, 10000],
        "passes":[c,c,c,c,c], "best":c})
}
fn document() -> Value {
    json!({"schema_version":1, "probe_version":"placement-v1", "units":"ms,bytes,work",
        "identity":{"host_class":"test", "cpu":"test-cpu", "device":"test-gpu",
            "driver":"test-driver", "toolchain":"rust-test"},
        "provisional":true, "source_revision":"0123456789012345678901234567890123456789",
        "observations":{"before":{"load":null,"swap_bytes":null},"after":{"load":null,"swap_bytes":null}},
        "refusals":[], "measurements":[entry("cpu", 1.0),entry("cuda", 0.5),entry("metal", 0.5)]})
}
fn profile(v: &Value) -> Profile {
    Profile::parse(&serde_json::to_vec(v).unwrap(), &v["identity"]).unwrap()
}
fn facts() -> Facts<'static> {
    Facts {
        family: "placement.wrapping_add1.u32",
        numeric: "u32-wrapping",
        numeric_contract_version: "wrapping-u32-v1",
        cpu_boundary: "retained-v1",
        device_boundary: "retained-v1",
        region: None,
        work: 100,
        input_bytes: 100,
        output_bytes: 100,
        launches: 1,
        temporary_bytes: 0,
        capacity_bytes: 10000,
        residency: Residency::HOST,
        pages: PageState::Known {
            resident: 1,
            total: 1,
        },
        cuda_legal: true,
        metal_legal: true,
    }
}
#[test]
fn profile_rejects_untrusted_figures_and_identity() {
    let v = document();
    profile(&v);
    let mut wrong = v["identity"].clone();
    wrong["cpu"] = json!("other");
    assert!(Profile::parse(&serde_json::to_vec(&v).unwrap(), &wrong).is_err());
    for (key, value) in [
        ("schema_version", json!(2)),
        ("probe_version", json!("future")),
        ("provisional", json!(false)),
        ("source_revision", json!("unknown")),
        ("units", json!("seconds")),
        ("observations", Value::Null),
    ] {
        let mut bad = v.clone();
        bad[key] = value;
        assert!(
            Profile::parse(&serde_json::to_vec(&bad).unwrap(), &v["identity"]).is_err(),
            "{key}"
        );
    }
    for (key, value) in [
        ("per_work_ms", json!(-1)),
        ("link_bytes_per_ms", json!(0)),
        ("fixed_ms", json!("NaN")),
    ] {
        let mut bad = v.clone();
        bad["measurements"][1]["passes"][0][key] = value;
        assert!(Profile::parse(&serde_json::to_vec(&bad).unwrap(), &v["identity"]).is_err());
    }
    let mut bad = v.clone();
    bad["measurements"][1]["best"]["per_work_ms"] = json!(0.1);
    assert!(Profile::parse(&serde_json::to_vec(&bad).unwrap(), &v["identity"]).is_err());
    assert!(
        Profile::parse(
            &vec![b' '; placement::MAX_PROFILE_BYTES + 1],
            &v["identity"]
        )
        .is_err()
    );
    assert!(Profile::parse(b"{", &v["identity"]).is_err());
}
#[test]
fn analytic_grid_margin_transfers_residency_launch_unknown_and_trust() {
    let v = document();
    let p = profile(&v);
    let mut f = facts();
    assert_eq!(placement::select(None, true, &f).engine, Engine::Cpu);
    assert_eq!(placement::select(Some(&p), false, &f).engine, Engine::Cpu);
    let receipt = placement::select(Some(&p), true, &f).receipt;
    let cpu = &receipt["cpu_cost"];
    assert_eq!(
        cpu["total_ms"].as_f64().unwrap(),
        cpu["fixed_ms"].as_f64().unwrap()
            + cpu["work_ms"].as_f64().unwrap()
            + cpu["transfer_ms"].as_f64().unwrap()
            + cpu["page_in_ms"].as_f64().unwrap()
    );
    for v in receipt["variants"].as_array().unwrap() {
        let sum = v["compute_ms"].as_f64().unwrap()
            + v["transfer_ms"].as_f64().unwrap()
            + v["page_in_ms"].as_f64().unwrap()
            + v["launch_ms"].as_f64().unwrap();
        assert!((sum - v["total_ms"].as_f64().unwrap()).abs() < 1e-12);
    }
    // Device ties use CUDA first, independently of profile row order.
    assert_eq!(placement::select(Some(&p), true, &f).engine, Engine::Cuda);
    f.capacity_bytes = 1_000_000;
    f.input_bytes = 100_000;
    f.output_bytes = 100_000;
    assert_eq!(placement::select(Some(&p), true, &f).engine, Engine::Cpu);
    f.residency = Residency::DEVICE;
    assert_eq!(placement::select(Some(&p), true, &f).engine, Engine::Cuda);
    f.cuda_legal = false;
    assert_eq!(placement::select(Some(&p), true, &f).engine, Engine::Metal);
    f.metal_legal = false;
    assert_eq!(placement::select(Some(&p), true, &f).engine, Engine::Cpu);
    let mut f = facts();
    f.work = 0;
    f.launches = 0;
    assert_eq!(placement::select(Some(&p), true, &f).engine, Engine::Cpu);
    f = facts();
    f.pages = PageState::Unknown;
    assert_eq!(placement::select(Some(&p), true, &f).engine, Engine::Cpu);
    f.pages = PageState::Known {
        resident: 0,
        total: 1,
    };
    assert_eq!(placement::select(Some(&p), true, &f).engine, Engine::Cpu);
    let mut v = v;
    let c = coefficients(true, 1.0);
    v["measurements"][0]["passes"] = json!([c, c, c, c, c]);
    for ratio in [0.949, 0.95, 0.951] {
        for i in 1..3 {
            let mut c = coefficients(false, ratio);
            c["link_bytes_per_ms"] = json!(1e30);
            v["measurements"][i]["passes"] = json!([c, c, c, c, c]);
            v["measurements"][i]["best"] = c;
        }
        let mut f = facts();
        f.input_bytes = 0;
        f.output_bytes = 0;
        assert_eq!(
            placement::select(Some(&profile(&v)), true, &f).engine,
            if ratio < 0.95 {
                Engine::Cuda
            } else {
                Engine::Cpu
            }
        );
    }
    let mut f = facts();
    f.temporary_bytes = 10001;
    assert_eq!(placement::select(Some(&p), true, &f).engine, Engine::Cpu);
    f = facts();
    f.cpu_boundary = "compiled-llvm-v1";
    assert_eq!(placement::select(Some(&p), true, &f).engine, Engine::Cpu);
}
#[test]
fn live_page_snapshot_never_modifies_the_slice() {
    assert_eq!(
        invar_cuda::pages::observe::<u32>(&[]),
        PageState::Known {
            resident: 0,
            total: 0
        }
    );
    let data = vec![0x12345678u32; 20000];
    let state = invar_cuda::pages::observe(&data[1..19999]);
    let live = cfg!(any(target_os = "linux", target_os = "macos")); // else Unknown
    assert!(matches!(state,PageState::Known{resident,total} if total>0 && resident<=total) == live);
    assert!(data.iter().all(|x| *x == 0x12345678));
}
#[test]
fn calibrate_cli_rejects_bad_options_without_creating_output() {
    for args in [
        vec!["calibrate", "--family", "made-up"],
        vec!["calibrate", "--family", "map"],
        vec!["calibrate", "--out"],
        vec![
            "calibrate",
            "--family",
            "placement.wrapping_add1.u32",
            "--family",
            "placement.wrapping_add1.u32",
        ],
    ] {
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_invar"))
            .args(args)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
        let v: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(v["status"], "tool_error");
    }
}

#[test]
fn page_cost_launch_floor_region_override_and_bounds() {
    let mut v = document();
    let mut cpu = coefficients(true, 0.1);
    cpu["page_in_ms_per_byte"] = json!(1.0);
    v["measurements"][0]["best"] = cpu.clone();
    v["measurements"][0]["passes"] = json!([cpu, cpu, cpu, cpu, cpu]);
    for i in 1..3 {
        let mut c = coefficients(false, 0.5);
        c["page_in_ms_per_byte"] = json!(0.0);
        v["measurements"][i]["best"] = c.clone();
        v["measurements"][i]["passes"] = json!([c, c, c, c, c]);
    }
    let mut f = facts();
    assert_eq!(
        placement::select(Some(&profile(&v)), true, &f).engine,
        Engine::Cpu
    );
    f.pages = PageState::Known {
        resident: 0,
        total: 1,
    };
    assert_eq!(
        placement::select(Some(&profile(&v)), true, &f).engine,
        Engine::Cuda
    );
    for i in 1..3 {
        let mut c = coefficients(false, 0.5);
        c["launch_floor_ms"] = json!(1000.0);
        v["measurements"][i]["best"] = c.clone();
        v["measurements"][i]["passes"] = json!([c, c, c, c, c]);
    }
    assert_eq!(
        placement::select(Some(&profile(&v)), true, &f).engine,
        Engine::Cpu
    );
    let mut v = document();
    let region = json!({"source_identity":42,"graph_revision":3,"region_identity":9});
    let mut row = entry("cpu", 0.01);
    row["region"] = region.clone();
    v["measurements"].as_array_mut().unwrap().push(row);
    let mut f = facts();
    f.region = Some(&region);
    assert_eq!(
        placement::select(Some(&profile(&v)), true, &f).engine,
        Engine::Cpu
    );
    f.region = None;
    assert_eq!(
        placement::select(Some(&profile(&v)), true, &f).engine,
        Engine::Cuda
    );
    f.work = 10001;
    assert_eq!(
        placement::select(Some(&profile(&v)), true, &f).engine,
        Engine::Cpu
    );
    let p = profile(&v);
    f = facts();
    f.input_bytes = usize::MAX;
    f.output_bytes = 1;
    assert_eq!(placement::select(Some(&p), true, &f).engine, Engine::Cpu);
}

#[test]
fn issue_v02_placement_bounded_loader_atomic_publish_and_no_clobber() {
    let dir = std::env::temp_dir().join(format!("invar-placement-test-{}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("profile.json");
    let v = document();
    assert!(
        Profile::load_default(&path, false, &v["identity"])
            .unwrap()
            .is_none()
    );
    assert!(Profile::load_default(&path, true, &v["identity"]).is_err());
    placement::calibration::publish(&path, &v).unwrap();
    let before = std::fs::read(&path).unwrap();
    assert!(Profile::load(&path, &v["identity"]).is_ok());
    assert!(placement::calibration::publish(&path, &v).is_err());
    assert_eq!(before, std::fs::read(&path).unwrap());
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
    std::fs::write(&path, b"{}").unwrap();
    assert!(Profile::load_default(&path, false, &v["identity"]).is_err());
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["calibrate", "--out"])
        .arg(&path)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(std::fs::read(&path).unwrap(), b"{}");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn calibrate_cli_exports_a_valid_provisional_profile() {
    let dir = std::env::temp_dir().join(format!("invar-calibrate-cli-{}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("test.json");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_invar"))
        .args([
            "calibrate",
            "--family",
            "placement.wrapping_add1.u32",
            "--host-class",
            "test",
            "--out",
        ])
        .arg(&path)
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(0),
        "{} {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let v: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(v["provisional"], true);
    // Map rows keep the placement family; native builds add the M5-01
    // family probes (#49), and every row keeps the wrapping-u32 contract.
    for row in v["measurements"].as_array().unwrap() {
        assert_eq!(
            row["table_family"] == "map",
            row["family"] == "placement.wrapping_add1.u32"
        );
        assert_eq!(row["numeric_contract_version"], "wrapping-u32-v1");
    }
    let registry: Value =
        serde_json::from_str(include_str!("../../schema/diagnostic-codes-v1.json")).unwrap();
    for refusal in v["refusals"].as_array().unwrap() {
        assert!(
            registry["explain_reason_codes"]
                .as_array()
                .unwrap()
                .contains(&refusal["reason"])
        );
    }
    assert_eq!(v["identity"]["host_class"], "test");
    assert!(Profile::load(&path, &v["identity"]).is_ok());
    assert_eq!(
        v["refusals"].as_array().unwrap().last().unwrap()["reason"],
        "unmeasured"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn published_schema_agrees_with_valid_fixture() {
    let schema: Value =
        serde_json::from_str(include_str!("../../schema/invar-placement-v1.schema.json")).unwrap();
    let validator = jsonschema::validator_for(&schema).unwrap();
    assert!(validator.is_valid(&document()));
    let mut bad = document();
    bad["measurements"][0]["passes"] = json!([]);
    assert!(!validator.is_valid(&bad));
    let mut bad = document();
    bad["measurements"][0]["family"] = json!("map-wrapping-add1");
    assert!(!validator.is_valid(&bad));
    bad = document();
    bad["measurements"][0]
        .as_object_mut()
        .unwrap()
        .remove("numeric_contract_version");
    assert!(!validator.is_valid(&bad));
}

#[test]
fn zero_work_does_not_dispatch_a_device() {
    let mut v = document();
    for row in v["measurements"].as_array_mut().unwrap() {
        row["sizes"][0] = json!(0);
    }
    let mut c = coefficients(true, 1.0);
    c["fixed_ms"] = json!(10.0);
    v["measurements"][0]["best"] = c.clone();
    v["measurements"][0]["passes"] = json!([c, c, c, c, c]);
    let mut f = facts();
    f.work = 0;
    f.input_bytes = 0;
    f.output_bytes = 0;
    f.launches = 0;
    assert_eq!(
        placement::select(Some(&profile(&v)), true, &f).engine,
        Engine::Cpu
    );
}

#[test]
fn cpu_download_and_output_upload_are_charged_for_resident_buffers() {
    let mut v = document();
    let c = coefficients(true, 0.4);
    v["measurements"][0]["best"] = c.clone();
    v["measurements"][0]["passes"] = json!([c, c, c, c, c]);
    let p = profile(&v);
    for (input, output) in [(100_000, 0), (0, 100_000)] {
        let mut f = facts();
        f.residency = Residency::DEVICE;
        f.capacity_bytes = 1_000_000;
        f.input_bytes = input;
        f.output_bytes = output;
        let d = placement::select(Some(&p), true, &f);
        // 40 ms compute + 100 ms retrieval/upload; device compute is 50 ms.
        assert_eq!(d.receipt["profile_schema_version"], 1);
        assert_eq!(d.receipt["profile_identity"], v["identity"]);
        assert_eq!(d.receipt["input_bytes"], input);
        assert_eq!(d.receipt["output_bytes"], output);
        assert_eq!(d.receipt["launches"], 1);
        assert_eq!(d.receipt["cpu_ms"], 140.0);
        assert_eq!(d.engine, Engine::Cuda);
    }
}

#[cfg(unix)]
#[test]
fn non_utf8_output_is_rejected_before_measurement_or_publication() {
    use std::os::unix::ffi::OsStringExt;
    let dir = std::env::temp_dir().join(format!("invar-placement-path-{}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join(std::ffi::OsString::from_vec(b"profile-\xff.json".to_vec()));
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["calibrate", "--out"])
        .arg(&path)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stderr.is_empty());
    let v: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(v["status"], "tool_error");
    assert!(String::from_utf8_lossy(&output.stdout).contains("TUNING_HINT_INVALID"));
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
    std::fs::remove_dir(dir).unwrap();
}

#[test]
fn binding_rulings_family_contract_and_registered_reasons() {
    let v = document();
    let p = profile(&v);
    let f = facts();
    let registry: Value =
        serde_json::from_str(include_str!("../../schema/diagnostic-codes-v1.json")).unwrap();
    let mut seen = std::collections::BTreeSet::new();
    for case in 0..7 {
        let mut f = facts();
        match case {
            0 => {
                f.cuda_legal = false;
                f.metal_legal = false;
            }
            1 => f.capacity_bytes = 0,
            4 => f.pages = PageState::Unknown,
            5 => f.device_boundary = "unmeasured-v1",
            _ => {}
        }
        let decision = placement::select(if case == 2 { None } else { Some(&p) }, case != 3, &f);
        assert_eq!(
            decision.receipt["numeric_contract_version"],
            "wrapping-u32-v1"
        );
        for variant in decision.receipt["variants"].as_array().unwrap() {
            assert!(
                registry["explain_reason_codes"]
                    .as_array()
                    .unwrap()
                    .contains(&variant["reason"])
            );
            seen.insert(variant["reason"].as_str().unwrap().to_owned());
        }
    }
    assert_eq!(seen.len(), 7);
    for bad_family in [
        "map-wrapping-add1",
        "map",
        "placement..u32",
        "placement.map.u32.extra",
    ] {
        let mut bad = v.clone();
        bad["measurements"][0]["family"] = json!(bad_family);
        assert!(Profile::parse(&serde_json::to_vec(&bad).unwrap(), &v["identity"]).is_err());
    }
    let mut bad = v.clone();
    bad["measurements"][0]
        .as_object_mut()
        .unwrap()
        .remove("numeric_contract_version");
    assert!(Profile::parse(&serde_json::to_vec(&bad).unwrap(), &v["identity"]).is_err());
    let mut mismatch = v.clone();
    for row in mismatch["measurements"].as_array_mut().unwrap() {
        row["numeric_contract_version"] = json!("wrapping-u32-v2");
    }
    assert_eq!(
        placement::select(Some(&profile(&mismatch)), true, &f).engine,
        Engine::Cpu
    );
}

fn auto_cli(dir: &std::path::Path, extra: &[&str], explain: bool) -> (i32, Value) {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_invar"))
        .current_dir(dir)
        .env("INVAR_METAL_SOURCE_TRIPWIRE", "1")
        .env_remove("INVAR_METAL_RUNTIME_SOURCE")
        .args([
            if explain { "explain" } else { "run" },
            "map.inv",
            "--engine",
            "auto",
            "--format",
            "json",
        ])
        .args(extra)
        .output()
        .unwrap();
    if out.stdout.is_empty() {
        assert_eq!(out.status.code(), Some(2));
        return (2, Value::Null);
    }
    let v = serde_json::from_slice(&out.stdout).unwrap_or_else(|e| {
        panic!(
            "{e}: {} {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    });
    (out.status.code().unwrap(), v)
}

#[test]
fn auto_stays_on_the_established_path_unless_placement_is_asked_for() {
    let dir = std::env::temp_dir().join(format!("invar-placement-default-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("calibration")).unwrap();
    std::fs::write(dir.join("map.inv"),b"fn main()->i32 {let xs:Array<u32> = [0u32,4294967295u32,7u32];var ys:Array<u32> = [0u32,0u32,0u32];for i in 0u32..len(xs) {ys[i]=wrapping_add(xs[i],1u32);} if ys[0u32]==1u32 and ys[1u32]==0u32 and ys[2u32]==8u32 {return 42;} return 9;}").unwrap();
    let host = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);
    // A provisional profile for the live host at the default path.
    let (_, placed) = auto_cli(&dir, &["--host-class", &host], false);
    let mut v = document();
    v["identity"] = placed["placement"]["resource_identity"].clone();
    std::fs::write(
        dir.join(format!("calibration/{host}.json")),
        serde_json::to_vec(&v).unwrap(),
    )
    .unwrap();
    for explain in [false, true] {
        // No placement flag: the provisional file alone does not switch auto.
        let (status, established) = auto_cli(&dir, &[], explain);
        assert_eq!(status, 0, "{established}");
        assert!(established.get("placement").is_none(), "{established}");
        assert_eq!(established["auto_path"], "established", "{established}");
        assert_eq!(
            established["schema"],
            if explain {
                "invar-explain-v5"
            } else {
                "invar-diagnostics-v3"
            },
            "{established}"
        );
        // Trusting it explicitly asks for placement.
        let (status, placement) = auto_cli(&dir, &["--trust-provisional"], explain);
        assert_eq!(status, 0, "{placement}");
        assert_eq!(placement["auto_path"], "placement");
        assert_eq!(placement["placement"]["profile_id"], profile(&v).id());
        if explain {
            assert_eq!(placement["schema"], "invar-explain-v5");
        }
    }
    // A frozen default is not silently skipped: placement reports the refusal
    // and keeps the CPU.
    let mut frozen = v.clone();
    frozen["provisional"] = json!(false);
    std::fs::write(
        dir.join(format!("calibration/{host}.json")),
        serde_json::to_vec(&frozen).unwrap(),
    )
    .unwrap();
    let (status, refused) = auto_cli(&dir, &[], false);
    assert_eq!(status, 0, "{refused}");
    assert_eq!(refused["auto_path"], "placement");
    assert_eq!(refused["placement"]["engine"], "cpu");
    assert_eq!(
        refused["placement"]["refusal_code"],
        "TUNING_FREEZE_REQUIRED"
    );
    // A malformed default stays a typed error.
    std::fs::write(dir.join(format!("calibration/{host}.json")), b"{").unwrap();
    let (status, malformed) = auto_cli(&dir, &[], false);
    assert_eq!(status, 2);
    assert_eq!(malformed["diagnostics"][0]["code"], "TUNING_HINT_INVALID");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn placement_documents_record_numeric_contracts() {
    let dir =
        std::env::temp_dir().join(format!("invar-placement-contracts-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("map.inv"),
        b"fn main() -> i32 { let e: f64 = exp(1.0f64); let s: f32 = sin(1.0); if e > 2.0f64 and s > 0.0 { return 42; } return 9; }",
    )
    .unwrap();
    let host = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);
    let expected = json!({"binary64": "strict-f64-v1", "math": "invar-math-cr-v1", "trig": "invar-trig-f32-v1"});
    for explain in [false, true] {
        let (status, document) = auto_cli(&dir, &["--host-class", &host], explain);
        assert_eq!(status, 0, "{document}");
        assert_eq!(document["auto_path"], "placement");
        assert_eq!(document["numeric_contracts"], expected, "{document}");
        let schema: Value = serde_json::from_str(if explain {
            include_str!("../../schema/explain-v5.schema.json")
        } else {
            DIAGNOSTICS_V3
        })
        .unwrap();
        assert!(jsonschema::is_valid(&schema, &document), "{document}");
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn real_cli_profile_trust_identity_contract_and_cpu_decision() {
    let dir = std::env::temp_dir().join(format!("invar-placement-auto-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("map.inv"),b"fn main()->i32 {let xs:Array<u32> = [0u32,4294967295u32,7u32];var ys:Array<u32> = [0u32,0u32,0u32];for i in 0u32..len(xs) {ys[i]=wrapping_add(xs[i],1u32);} if ys[0u32]==1u32 and ys[1u32]==0u32 and ys[2u32]==8u32 {return 42;} return 9;}").unwrap();
    let host = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);
    let (status, missing) = auto_cli(&dir, &["--host-class", &host], false);
    assert_eq!(status, 0, "{missing}");
    assert_eq!(missing["auto_path"], "placement");
    assert_eq!(missing["result"]["value"], 42);
    assert_eq!(missing["placement"]["engine"], "cpu");
    assert_eq!(missing["placement"]["refusal_code"], "TUNING_NO_CANDIDATES");
    let mut v = document();
    v["identity"] = missing["placement"]["resource_identity"].clone();
    for row in v["measurements"].as_array_mut().unwrap() {
        row["boundary"] = json!(if row["engine"] == "cpu" {
            "llvm-host-array-v1"
        } else {
            "host-map-v1"
        });
    }
    // CPU coefficients win even with trust. This also exercises LLVM bulk output.
    for row in v["measurements"].as_array_mut().unwrap().iter_mut().skip(1) {
        let c = coefficients(false, 100.0);
        row["best"] = c.clone();
        row["passes"] = json!([c, c, c, c, c]);
    }
    std::fs::write(dir.join("profile.json"), serde_json::to_vec(&v).unwrap()).unwrap();
    for explain in [false, true] {
        let (status, untrusted) = auto_cli(&dir, &["--profile", "profile.json"], explain);
        assert_eq!(status, 0, "{untrusted}");
        assert_eq!(untrusted["placement"]["engine"], "cpu");
        assert_eq!(
            untrusted["placement"]["refusal_code"],
            "TUNING_FREEZE_REQUIRED"
        );
        assert!(untrusted["placement"]["profile_id"].as_str().is_some());
        let (status, trusted) = auto_cli(
            &dir,
            &["--profile", "profile.json", "--trust-provisional"],
            explain,
        );
        assert_eq!(status, 0, "{trusted}");
        assert_eq!(trusted["auto_path"], "placement");
        if explain {
            assert_eq!(trusted["schema"], "invar-explain-v5");
        }
        let schema: Value = serde_json::from_str(if explain {
            include_str!("../../schema/explain-v5.schema.json")
        } else {
            DIAGNOSTICS_V3
        })
        .unwrap();
        let validator = jsonschema::validator_for(&schema).unwrap();
        assert!(
            validator.is_valid(&trusted),
            "{trusted}: {:?}",
            validator
                .iter_errors(&trusted)
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
        );
        if explain {
            let schema: Value = serde_json::from_str(DIAGNOSTICS_V3).unwrap();
            assert!(jsonschema::is_valid(&schema, &trusted["execution"]));
        }

        assert_eq!(trusted["placement"]["engine"], "cpu");
        let choices = trusted["placement"]["choices"].as_array().unwrap();
        assert!(
            choices.iter().any(|c| c["profile_id"] == profile(&v).id()
                && (!cfg!(feature = "native") || c["cpu_ms"].as_f64().is_some())),
            "{trusted}"
        );
        assert!(
            trusted["placement"]["choices"]
                .as_array()
                .is_some_and(|v| !v.is_empty()),
            "{trusted}"
        );
    }
    // The default discovery path uses the same identity and trust gate.
    let default_dir = dir.join("calibration");
    std::fs::create_dir(&default_dir).unwrap();
    let default_path = default_dir.join(format!(
        "{}-{}.json",
        std::env::consts::OS,
        std::env::consts::ARCH
    ));
    std::fs::write(&default_path, serde_json::to_vec(&v).unwrap()).unwrap();
    let (status, implicit) = auto_cli(&dir, &["--trust-provisional"], false);
    assert_eq!(status, 0);
    assert_eq!(implicit["placement"]["profile_id"], profile(&v).id());
    std::fs::remove_file(&default_path).unwrap();
    let saved = v.clone();
    for row in v["measurements"].as_array_mut().unwrap() {
        row["numeric_contract_version"] = json!("future-contract");
    }
    std::fs::write(dir.join("profile.json"), serde_json::to_vec(&v).unwrap()).unwrap();
    let (status, unknown) = auto_cli(
        &dir,
        &["--profile", "profile.json", "--trust-provisional"],
        false,
    );
    assert_eq!(status, 0);
    assert_eq!(unknown["placement"]["engine"], "cpu");
    assert!(
        unknown["placement"]["choices"]
            .as_array()
            .unwrap()
            .iter()
            .all(|c| c["cpu_ms"].is_null())
    );
    v = saved;
    let mut frozen = v.clone();
    frozen["provisional"] = json!(false);
    std::fs::write(
        dir.join("profile.json"),
        serde_json::to_vec(&frozen).unwrap(),
    )
    .unwrap();
    let (status, frozen) = auto_cli(
        &dir,
        &["--profile", "profile.json", "--trust-provisional"],
        false,
    );
    assert_eq!(status, 0);
    assert_eq!(
        frozen["placement"]["refusal_code"],
        "TUNING_FREEZE_REQUIRED"
    );
    v["identity"]["toolchain"] = json!("mismatched-toolchain");
    std::fs::write(dir.join("profile.json"), serde_json::to_vec(&v).unwrap()).unwrap();
    let (status, mismatched) = auto_cli(
        &dir,
        &["--profile", "profile.json", "--trust-provisional"],
        false,
    );
    assert_eq!(status, 0);
    assert_eq!(mismatched["placement"]["engine"], "cpu");
    assert_eq!(
        mismatched["placement"]["refusal_code"],
        "TUNING_HINT_INVALID"
    );
    std::fs::write(dir.join("profile.json"), b"{").unwrap();
    let (status, malformed) = auto_cli(&dir, &["--profile", "profile.json"], false);
    assert_eq!(status, 2);
    assert_eq!(malformed["diagnostics"][0]["code"], "TUNING_HINT_INVALID");
    for flags in [
        vec!["--trust-provisional", "--trust-provisional"],
        vec!["--profile"],
        vec!["--host-class", "../bad"],
    ] {
        assert_eq!(auto_cli(&dir, &flags, false).0, 2);
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[cfg(all(
    feature = "native",
    any(feature = "cuda", all(feature = "metal", target_os = "macos"))
))]
fn device_cli_profile(engine: &str) {
    let dir = std::env::temp_dir().join(format!(
        "invar-placement-auto-{engine}-{}",
        std::process::id()
    ));
    std::fs::create_dir(&dir).unwrap();
    for n in [1, 3, 257, 4096] {
        let source = format!(
            "fn main()->i32 {{let xs:Array<u32> = filled_u32({n}u32,4294967295u32);var ys:Array<u32> = filled_u32({n}u32,123u32);for i in 0u32..len(xs) {{ys[i]=wrapping_add(xs[i],1u32);}} for j in 0u32..len(ys) {{if ys[j]!=0u32 {{return 9;}}}} return 42;}}"
        );
        std::fs::write(dir.join("map.inv"), source).unwrap();
        // Ask for placement so the receipt carries the live identity.
        let host = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);
        let (_, missing) = auto_cli(&dir, &["--host-class", &host], false);
        let mut v = document();
        v["identity"] = missing["placement"]["resource_identity"].clone();
        for row in v["measurements"].as_array_mut().unwrap() {
            row["boundary"] = json!(if row["engine"] == "cpu" {
                "llvm-host-array-v1"
            } else {
                "host-map-v1"
            });
            let cost = if row["engine"] == engine {
                0.000001
            } else {
                100.0
            };
            let mut c = coefficients(false, cost);
            c["link_bytes_per_ms"] = json!(1e30);
            row["best"] = c.clone();
            row["passes"] = json!([c, c, c, c, c]);
        }
        std::fs::write(dir.join("profile.json"), serde_json::to_vec(&v).unwrap()).unwrap();
        let mut flags = vec!["--profile", "profile.json", "--trust-provisional"];
        if engine == "metal" {
            let build = std::process::Command::new(env!("CARGO_BIN_EXE_invar"))
                .current_dir(&dir)
                .args([
                    "build",
                    "map.inv",
                    "--target",
                    "metal",
                    "--output",
                    "map.metallib",
                ])
                .output()
                .unwrap();
            assert!(
                build.status.success(),
                "{} {}",
                String::from_utf8_lossy(&build.stdout),
                String::from_utf8_lossy(&build.stderr)
            );
            flags.extend(["--artifact", "map.metallib"]);
        }
        for explain in [false, true] {
            let (status, actual) = auto_cli(&dir, &flags, explain);
            assert_eq!(status, 0, "{actual}");
            assert_eq!(actual["placement"]["engine"], engine, "{actual}");
            assert_eq!(actual["placement"]["profile_id"], profile(&v).id());
            assert_eq!(actual["placement"]["profile_schema_version"], 1);
            let execution = if explain {
                &actual["execution"]
            } else {
                &actual
            };
            let reference = std::process::Command::new(env!("CARGO_BIN_EXE_invar"))
                .current_dir(&dir)
                .args([
                    "run",
                    "map.inv",
                    "--engine",
                    "reference",
                    "--format",
                    "json",
                ])
                .output()
                .unwrap();
            let reference: Value = serde_json::from_slice(&reference.stdout).unwrap();
            assert_eq!(execution["result"], reference["result"]);
            assert!(
                actual["placement"]["choices"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|c| c["engine"] == engine && c["published"] == true)
            );
        }
        if engine == "metal" {
            std::fs::remove_file(dir.join("map.metallib")).unwrap();
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}
#[test]
#[ignore = "requires a real CUDA device and built invar-nvrtc helper"]
#[cfg(all(feature = "native", feature = "cuda"))]
fn real_cli_cuda_profile_drives_device_and_matches_reference() {
    device_cli_profile("cuda");
}
#[test]
#[ignore = "requires Apple GPU and offline Metal compiler"]
#[cfg(all(feature = "native", feature = "metal", target_os = "macos"))]
fn real_cli_metal_profile_drives_device_and_matches_reference() {
    device_cli_profile("metal");
}

#[test]
fn profiled_cpu_preserves_reference_fuel_traps_and_cleanup() {
    use invar::{check, eval, front, mir, schedule::profiled};
    let v = document();
    let p = profile(&v);
    for tail in ["return 42;", "return 1/0;"] {
        let source = format!(
            "fn main()->i32 {{let xs:Array<u32> = [0u32,4294967295u32,7u32];var ys:Array<u32> = [0u32,0u32,0u32];for i in 0u32..len(xs) {{ys[i]=wrapping_add(xs[i],1u32);}} {tail}}}"
        );
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let program = mir::lower(&checked).unwrap();
        for fuel in [1, 20, 80, 1000] {
            let expected = eval::observe(&checked, fuel);
            let actual = profiled::run(
                source.as_bytes(),
                &program,
                fuel,
                Some(&p),
                true,
                profiled::Resources {
                    compiler: std::path::Path::new(""),
                    #[cfg(all(feature = "metal", target_os = "macos"))]
                    kernels: invar::schedule::metal::Kernels::Missing,
                },
            )
            .unwrap()
            .observation;
            assert_eq!(actual.result, expected.result, "fuel={fuel} tail={tail}");
            assert_eq!(actual.steps_used, expected.steps_used);
            assert_eq!(actual.cleanup_trace, expected.cleanup_trace);
        }
    }
}
