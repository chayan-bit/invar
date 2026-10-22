//! GNET-04 D1(a)/D2(a) (#196): a placement run records the legal engine set of
//! its general net, keeps the CPU route and says why (`uncalibrated`), and labels
//! which host ran: LLVM native, or the reference evaluator and why.
use invar::placement::Route;
use serde_json::{Value, json};

const FORKED: &str = "fn leaf(n:u32)->u32 { return wrapping_add(n,3u32); } fn main()->i32 { let a=leaf(1u32); let b=leaf(2u32); if wrapping_add(a,b)==9u32 {return 42;} return 9; }";
const SERIAL: &str = "fn leaf(n:u32)->u32 { return wrapping_add(n,3u32); } fn main()->i32 { let a=leaf(1u32); let b=leaf(a); if b==7u32 {return 42;} return 9; }";
// The device plans have no binary64 form, so both refuse the whole program.
const REFUSED: &str =
    "fn main()->i32 { let e: f64 = exp(1.0f64); if e > 2.0f64 {return 42;} return 9; }";

fn auto(dir: &std::path::Path, source: &str, explain: bool) -> Value {
    auto_with(dir, source, explain, &[])
}

fn auto_with(dir: &std::path::Path, source: &str, explain: bool, extra: &[&str]) -> Value {
    std::fs::write(dir.join("p.inv"), source).unwrap();
    let host = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_invar"))
        .current_dir(dir)
        .args([if explain { "explain" } else { "run" }, "p.inv"])
        .args([
            "--engine",
            "auto",
            "--format",
            "json",
            "--host-class",
            &host,
        ])
        .args(extra)
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(0), "{out:?}");
    serde_json::from_slice(&out.stdout).unwrap()
}

/// A named corruption of a valid legality or route object.
type Edit = (&'static str, fn(&mut Value));

fn schema(explain: bool) -> jsonschema::Validator {
    let schema: Value = serde_json::from_str(if explain {
        include_str!("../schema/explain-v5.schema.json")
    } else {
        include_str!("../schema/invar-diagnostics-v3.schema.json")
    })
    .unwrap();
    jsonschema::validator_for(&schema).unwrap()
}

#[test]
fn v01_gnet_auto_records_the_legal_set_and_keeps_the_cpu_route() {
    let dir = std::env::temp_dir().join(format!("invar-gnet-auto-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let native = cfg!(feature = "native");
    let cpu = json!({"engine":"cpu","legal":true,"compiled":native,
        "reason":if native {Value::Null} else {json!("native-feature-unavailable")}});
    for explain in [false, true] {
        for (source, devices) in [
            (
                FORKED,
                json!({"legal":true,"refusal":null,"forks":1,"serialized":[]}),
            ),
            (
                SERIAL,
                json!({"legal":true,"refusal":null,"forks":0,"serialized":["dependent_argument"]}),
            ),
            (
                REFUSED,
                json!({"legal":false,"refusal":"unsupported_type","forks":0,"serialized":[]}),
            ),
        ] {
            let doc = auto(&dir, source, explain);
            assert!(schema(explain).is_valid(&doc), "{doc}");
            assert_eq!(doc["auto_path"], "placement");
            let placement = &doc["placement"];
            // D1(a): the device legality is recorded, never acted on.
            assert_eq!(placement["engine"], "cpu", "{doc}");
            let result = if explain {
                &doc["execution"]["result"]
            } else {
                &doc["result"]
            };
            assert_eq!(result["value"], 42, "{doc}");
            let mut engines = vec![cpu.clone()];
            for engine in ["cuda", "metal"] {
                let mut d = devices.clone();
                d["engine"] = json!(engine);
                engines.push(d);
            }
            assert_eq!(
                placement["legality"],
                json!({"policy":"legal-set-cpu","reason":"uncalibrated","engines":engines}),
                "{source}"
            );
            if explain {
                assert_eq!(doc["execution"]["placement"], *placement);
            }
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn v01_gnet_auto_legality_schema_refuses_malformed_receipts() {
    let dir = std::env::temp_dir().join(format!("invar-gnet-schema-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // Run receipts (diagnostics-v3) and explain-v5 declare the same object.
    let valid = [false, true].map(|explain| (schema(explain), auto(&dir, FORKED, explain)));
    std::fs::remove_dir_all(dir).unwrap();
    let edits: [Edit; 10] = [
        ("measured reason", |l| l["reason"] = json!("measured")),
        ("unknown policy", |l| l["policy"] = json!("cost")),
        ("extra key", |l| l["chosen"] = json!("cuda")),
        ("compiled with a reason", |l| {
            l["engines"][0]["compiled"] = json!(true);
            l["engines"][0]["reason"] = json!("native-feature-unavailable")
        }),
        ("reference without a reason", |l| {
            l["engines"][0]["compiled"] = json!(false);
            l["engines"][0]["reason"] = Value::Null
        }),
        ("illegal cpu", |l| l["engines"][0]["legal"] = json!(false)),
        ("legal with refusal", |l| {
            l["engines"][1]["refusal"] = json!("source")
        }),
        ("refused without class", |l| {
            l["engines"][2]["legal"] = json!(false)
        }),
        ("unknown class", |l| {
            l["engines"][1]["legal"] = json!(false);
            l["engines"][1]["refusal"] = json!("busy");
            l["engines"][1]["forks"] = json!(0)
        }),
        ("reordered engines", |l| {
            l["engines"].as_array_mut().unwrap().swap(1, 2)
        }),
    ];
    for (validator, valid) in &valid {
        assert!(validator.is_valid(valid), "{valid}");
        for (name, edit) in edits {
            let mut doc = valid.clone();
            edit(&mut doc["placement"]["legality"]);
            assert!(!validator.is_valid(&doc), "{name}");
        }
    }
}

const MAP: &str = "fn main()->i32 {let xs:Array<u32> = [0u32,4294967295u32,7u32];var ys:Array<u32> = [0u32,0u32,0u32];for i in 0u32..len(xs) {ys[i]=wrapping_add(xs[i],1u32);} if ys[0u32]==1u32 and ys[1u32]==0u32 and ys[2u32]==8u32 {return 42;} return 9;}";

#[test]
fn v01_gnet_auto_route_labels_the_host_that_ran() {
    let dir = std::env::temp_dir().join(format!("invar-gnet-route-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let native = cfg!(feature = "native");
    // A provisional CPU-only profile for this host, trusted explicitly.
    let placed = auto(&dir, MAP, false);
    let c = json!({"fixed_ms":0.0,"per_work_ms":1.0,"link_bytes_per_ms":1000.0,
        "link_latency_ms":0.0,"launch_floor_ms":0.0,"page_in_ms_per_byte":null});
    let profile = json!({"schema_version":1,"probe_version":"placement-v1","units":"ms,bytes,work",
        "identity":placed["placement"]["resource_identity"],"provisional":true,
        "source_revision":"0123456789012345678901234567890123456789",
        "observations":{"before":{"load":null,"swap_bytes":null},"after":{"load":null,"swap_bytes":null}},
        "refusals":[],"measurements":[{"family":"placement.wrapping_add1.u32","numeric":"u32-wrapping",
            "numeric_contract_version":"wrapping-u32-v1","engine":"cpu","boundary":"llvm-host-array-v1",
            "region":null,"sizes":[1,10000],"passes":[c,c,c,c,c],"best":c}]});
    std::fs::write(dir.join("profile.json"), profile.to_string()).unwrap();
    let trusted = ["--profile", "profile.json", "--trust-provisional"];
    for explain in [false, true] {
        for (extra, expected) in [
            (&[][..], Route::select(native, false)),
            (&trusted[..], Route::select(native, true)),
        ] {
            let doc = auto_with(&dir, MAP, explain, extra);
            assert!(schema(explain).is_valid(&doc), "{doc}");
            let placement = &doc["placement"];
            assert_eq!(placement["route"], expected.receipt(), "{doc}");
            // The native route runs the whole program, so it offers no region;
            // the reference host offers the map loop to the placement policy.
            let choices = placement["choices"].as_array().unwrap();
            assert_eq!(choices.is_empty(), expected == Route::Native, "{doc}");
            assert_eq!(placement["engine"], "cpu", "{doc}");
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
    assert_eq!(Route::select(true, false), Route::Native);
    assert_eq!(
        Route::select(false, false).receipt(),
        json!({"host":"reference","reason":"native-feature-unavailable"})
    );
    assert_eq!(
        Route::select(false, true).receipt(),
        json!({"host":"reference","reason":"native-feature-unavailable"})
    );
    assert_eq!(
        Route::select(true, true).receipt(),
        json!({"host":"reference","reason":"profiled-regions"})
    );
}

#[test]
fn v01_gnet_auto_route_schema_refuses_compiled_claims() {
    let dir = std::env::temp_dir().join(format!("invar-gnet-route-schema-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let valid = [false, true].map(|explain| (schema(explain), auto(&dir, FORKED, explain)));
    std::fs::remove_dir_all(dir).unwrap();
    let edits: [Edit; 8] = [
        ("native with a reason", |p| {
            p["route"] = json!({"host":"native","reason":"profiled-regions"})
        }),
        ("reference without a reason", |p| {
            p["route"] = json!({"host":"reference","reason":null})
        }),
        ("unknown host", |p| {
            p["route"] = json!({"host":"compiled","reason":null})
        }),
        ("unknown reason", |p| {
            p["route"] = json!({"host":"reference","reason":"slow"})
        }),
        ("extra key", |p| p["route"]["engine"] = json!("cpu")),
        // The route and the CPU legality entry must agree on the native feature.
        ("native route without compiled cpu", |p| {
            p["route"] = json!({"host":"native","reason":null});
            cpu(p, false)
        }),
        ("missing feature with compiled cpu", |p| {
            p["route"] = json!({"host":"reference","reason":"native-feature-unavailable"});
            cpu(p, true)
        }),
        ("profiled regions without compiled cpu", |p| {
            p["route"] = json!({"host":"reference","reason":"profiled-regions"});
            cpu(p, false)
        }),
    ];
    let consistent: [Edit; 3] = [
        ("native", |p| {
            p["route"] = json!({"host":"native","reason":null});
            cpu(p, true)
        }),
        ("profiled", |p| {
            p["route"] = json!({"host":"reference","reason":"profiled-regions"});
            cpu(p, true)
        }),
        ("reference", |p| {
            p["route"] = json!({"host":"reference","reason":"native-feature-unavailable"});
            cpu(p, false)
        }),
    ];
    for (validator, valid) in &valid {
        assert!(validator.is_valid(valid), "{valid}");
        for (name, edit) in edits {
            let mut doc = valid.clone();
            edit(&mut doc["placement"]);
            assert!(!validator.is_valid(&doc), "{name}");
        }
        for (name, edit) in consistent {
            let mut doc = valid.clone();
            edit(&mut doc["placement"]);
            assert!(validator.is_valid(&doc), "{name}");
        }
    }
}

/// Sets the CPU legality entry to a compiled or a feature-less build.
fn cpu(placement: &mut Value, compiled: bool) {
    placement["legality"]["engines"][0] = json!({"engine":"cpu","legal":true,"compiled":compiled,
        "reason":if compiled {Value::Null} else {json!("native-feature-unavailable")}});
}
