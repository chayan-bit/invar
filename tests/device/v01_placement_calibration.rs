//! M5-01 calibration protocol in `invar-placement-v1` (#49): the loader
//! treats a profile as untrusted and refuses anything it cannot re-derive.
use invar::placement::{self, Engine, Error, Profile};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn samples() -> Vec<u64> {
    (1..=30).map(|i| i * 500).collect()
}
fn point(elements: Value, boundary: &str, ns: &[u64]) -> Value {
    json!({"elements":elements,"boundary":boundary,"samples_ns":ns,
        "summary":placement::summarize(ns).unwrap()})
}
fn row(engine: &str) -> Value {
    let boundary = if engine == "cpu" {
        "llvm-host-array-v1"
    } else {
        "host-map-v1"
    };
    let c = json!({"fixed_ms":0.0,"per_work_ms":1.0,"link_bytes_per_ms":1000.0,
        "link_latency_ms":0.0,"launch_floor_ms":0.0,"page_in_ms_per_byte":null});
    json!({"family":placement::MAP_FAMILY,"table_family":"map","numeric":"u32-wrapping",
        "numeric_contract_version":placement::MAP_NUMERIC_CONTRACT,"engine":engine,
        "boundary":boundary,"region":null,"sizes":[1, 10000],"passes":[c,c,c,c,c],"best":c,
        "points":[point(Value::Null, placement::FIRST_CALL, &[9000]),
            point(json!(1), boundary, &samples()), point(json!(10000), boundary, &samples())]})
}
/// CPU and CUDA measure `map`; everything else is pending.
fn document() -> Value {
    let mut pending = Vec::new();
    for engine in ["cpu", "cuda", "metal"] {
        for family in placement::FAMILIES {
            if engine == "metal" || family != "map" {
                pending.push(json!({"engine":engine,"family":family,"reason":"unmeasured"}));
            }
        }
    }
    json!({"schema_version":1,"probe_version":"placement-v1","units":"ms,bytes,work",
        "identity":{"host_class":"test","cpu":"test-cpu","device":"test-gpu",
            "driver":"test-driver","toolchain":"rust-test"},
        "provisional":true,"source_revision":"0123456789012345678901234567890123456789",
        "observations":{"before":{"load":null,"swap_bytes":null},"after":{"load":null,"swap_bytes":null}},
        "statistics":placement::statistics(),
        "coverage":{"families":placement::FAMILIES,"pending":pending},
        "refusals":[],"measurements":[row("cpu"),row("cuda")]})
}
fn parse(v: &Value) -> Result<Profile, Error> {
    Profile::parse(&serde_json::to_vec(v).unwrap(), &v["identity"])
}
/// The profile pinned to the digest of its own bytes.
fn pinned(v: &Value) -> Profile {
    let bytes = serde_json::to_vec(v).unwrap();
    Profile::parse_pinned(&bytes, &v["identity"], &hex(&bytes)).unwrap()
}
fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[test]
fn v01_placement_calibration_summary_uses_exact_order_statistics() {
    let s = placement::summarize(&samples()).unwrap();
    assert_eq!(s["n"], 30);
    assert_eq!(s["median_ns"], 7750.0);
    assert_eq!(s["ci95_ns"], json!([5000, 10500]));
    assert_eq!(s["q1_ns"], 4000);
    assert_eq!(s["q3_ns"], 11500);
    assert_eq!(
        (s["min_ns"].clone(), s["max_ns"].clone()),
        (json!(500), json!(15000))
    );
    // Order does not matter; the summary sorts.
    let mut reversed = samples();
    reversed.reverse();
    assert_eq!(placement::summarize(&reversed).unwrap(), s);
    let one = placement::summarize(&[9]).unwrap();
    assert_eq!(one["ci95_ns"], json!([9, 9]));
    for bad in [&samples()[..29], &[][..], &[0][..], &[1, 2][..]] {
        assert_eq!(placement::summarize(bad), Err(Error::Invalid), "{bad:?}");
    }
    assert_eq!(
        placement::statistics(),
        json!({"warmups":10,"samples":30,"center":placement::CENTER,"interval":placement::INTERVAL,
            "spread":placement::SPREAD,"policy":placement::POLICY})
    );
}

#[test]
fn v01_placement_calibration_valid_profile_loads_pinned_and_reports_coverage() {
    let v = document();
    // Without a pinned digest nothing vouches that passes and best came
    // from the samples, so an unpinned profile never claims calibration.
    let unpinned = parse(&v).unwrap();
    assert!(!unpinned.calibrated(Engine::Cpu, "map"));
    let mut swapped = document();
    let c = json!({"fixed_ms":0.0,"per_work_ms":1e-9,"link_bytes_per_ms":1e9,
        "link_latency_ms":0.0,"launch_floor_ms":0.0,"page_in_ms_per_byte":null});
    swapped["measurements"][0]["passes"] = json!([c, c, c, c, c]);
    swapped["measurements"][0]["best"] = c;
    assert!(!parse(&swapped).unwrap().calibrated(Engine::Cpu, "map"));
    let p = pinned(&v);
    assert!(p.calibrated(Engine::Cpu, "map"));
    assert!(p.calibrated(Engine::Cuda, "map"));
    assert!(!p.calibrated(Engine::Metal, "map"));
    assert!(!p.calibrated(Engine::Cpu, "scan"));
    let bytes = serde_json::to_vec(&v).unwrap();
    let exact = Profile::parse_pinned(&bytes, &v["identity"], &hex(&bytes)).unwrap();
    assert_eq!(exact.id(), hex(&bytes));
    assert_eq!(
        Profile::parse_pinned(&bytes, &v["identity"], &"0".repeat(64)).unwrap_err(),
        Error::Digest
    );
    // The size check comes first, before the digest is computed.
    let huge = vec![b' '; placement::MAX_PROFILE_BYTES + 1];
    assert_eq!(
        Profile::parse_pinned(&huge, &v["identity"], &hex(&huge)).unwrap_err(),
        Error::Invalid
    );
    // The cap is 4 MiB: a valid profile padded past the old 1 MiB still loads.
    let mut padded = bytes.clone();
    padded.resize(placement::MAX_PROFILE_BYTES, b' ');
    assert_eq!(placement::MAX_PROFILE_BYTES, 4 << 20);
    assert!(Profile::parse(&padded, &v["identity"]).is_ok());
    assert_eq!(
        Profile::parse(&huge, &v["identity"]).unwrap_err(),
        Error::Invalid
    );
    // A profile that does not declare the protocol keeps loading as before,
    // but can never claim a calibrated family.
    let mut legacy = document();
    let o = legacy.as_object_mut().unwrap();
    o.remove("statistics");
    o.remove("coverage");
    for row in legacy["measurements"].as_array_mut().unwrap() {
        let r = row.as_object_mut().unwrap();
        r.remove("points");
        r.remove("table_family");
    }
    assert!(!pinned(&legacy).calibrated(Engine::Cpu, "map"));
}

#[test]
fn v01_placement_calibration_loader_refuses_corrupt_or_incomplete_profiles() {
    type Corrupt = (&'static str, fn(&mut Value));
    let cases: [Corrupt; 30] = [
        ("summary edited", |v| {
            v["measurements"][0]["points"][1]["summary"]["median_ns"] = json!(1.0)
        }),
        ("sample edited", |v| {
            v["measurements"][0]["points"][1]["samples_ns"][3] = json!(99_000)
        }),
        ("29 samples", |v| {
            let p = &mut v["measurements"][0]["points"][1];
            p["samples_ns"].as_array_mut().unwrap().pop();
        }),
        ("zero sample", |v| {
            v["measurements"][0]["points"][0]["samples_ns"][0] = json!(0)
        }),
        ("float sample", |v| {
            v["measurements"][0]["points"][0]["samples_ns"][0] = json!(9000.5)
        }),
        ("first call with thirty samples", |v| {
            v["measurements"][0]["points"][1]["boundary"] = json!("first_call")
        }),
        ("timed point with one sample", |v| {
            v["measurements"][0]["points"][0]["boundary"] = json!("llvm-host-array-v1");
            v["measurements"][0]["points"][0]["elements"] = json!(1);
        }),
        ("size without samples", |v| {
            v["measurements"][0]["points"].as_array_mut().unwrap().pop();
        }),
        ("duplicate point", |v| {
            let p = v["measurements"][0]["points"][1].clone();
            v["measurements"][0]["points"]
                .as_array_mut()
                .unwrap()
                .push(p)
        }),
        ("point outside the row sizes", |v| {
            v["measurements"][0]["points"][1]["elements"] = json!(2)
        }),
        ("point under another boundary", |v| {
            v["measurements"][0]["points"][1]["boundary"] = json!("host-map-v1")
        }),
        ("no points", |v| v["measurements"][0]["points"] = json!([])),
        ("unknown table family", |v| {
            v["measurements"][0]["table_family"] = json!("sort")
        }),
        // A claim must follow from the row's own probe, not its label.
        ("map probe relabelled as scan", |v| {
            v["measurements"][0]["table_family"] = json!("scan");
            for p in v["coverage"]["pending"].as_array_mut().unwrap() {
                if p["engine"] == "cpu" && p["family"] == "scan" {
                    p["family"] = json!("map");
                }
            }
        }),
        ("unregistered probe family", |v| {
            v["measurements"][0]["family"] = json!("placement.wrapping_add2.u32")
        }),
        ("probe under another engine's boundary", |v| {
            v["measurements"][0]["boundary"] = json!("host-map-v1");
            for p in v["measurements"][0]["points"].as_array_mut().unwrap() {
                if p["boundary"] != "first_call" {
                    p["boundary"] = json!("host-map-v1");
                }
            }
        }),
        ("probe under another numeric contract", |v| {
            v["measurements"][0]["numeric_contract_version"] = json!("wrapping-u32-v2")
        }),
        (
            "region-bound probe row",
            |v| {
                v["measurements"][0]["region"] =
                    json!({"source_identity":1,"graph_revision":0,"region_identity":0})
            },
        ),
        ("no cold first call", |v| {
            v["measurements"][0]["points"]
                .as_array_mut()
                .unwrap()
                .remove(0);
        }),
        ("two cold first calls", |v| {
            let p = v["measurements"][0]["points"][0].clone();
            v["measurements"][0]["points"]
                .as_array_mut()
                .unwrap()
                .push(p)
        }),
        ("first call bound to a size", |v| {
            v["measurements"][0]["points"][0]["elements"] = json!(1)
        }),
        ("family neither measured nor pending", |v| {
            v["coverage"]["pending"].as_array_mut().unwrap().pop();
        }),
        ("family measured and pending", |v| {
            v["coverage"]["pending"]
                .as_array_mut()
                .unwrap()
                .push(json!({"engine":"cpu","family":"map","reason":"double counted"}))
        }),
        ("duplicate pending", |v| {
            let p = v["coverage"]["pending"][0].clone();
            v["coverage"]["pending"].as_array_mut().unwrap().push(p)
        }),
        ("pending engine unknown", |v| {
            v["coverage"]["pending"][0]["engine"] = json!("tpu")
        }),
        ("coverage families reordered", |v| {
            v["coverage"]["families"].as_array_mut().unwrap().swap(0, 1)
        }),
        ("statistics method changed", |v| {
            v["statistics"]["interval"] = json!("mean plus or minus two sigma")
        }),
        ("warmups changed", |v| v["statistics"]["warmups"] = json!(3)),
        ("statistics without coverage", |v| {
            v.as_object_mut().unwrap().remove("coverage");
        }),
        ("coverage without statistics", |v| {
            v.as_object_mut().unwrap().remove("statistics");
        }),
    ];
    for (name, corrupt) in cases {
        let mut v = document();
        corrupt(&mut v);
        assert_eq!(parse(&v).unwrap_err(), Error::Invalid, "{name}");
    }
    // Undeclared protocol fields are refused rather than silently ignored.
    let mut v = document();
    v.as_object_mut().unwrap().remove("statistics");
    v.as_object_mut().unwrap().remove("coverage");
    assert_eq!(
        parse(&v).unwrap_err(),
        Error::Invalid,
        "points without statistics"
    );
}

#[test]
fn v01_placement_calibration_published_schema_agrees() {
    let schema: Value =
        serde_json::from_str(include_str!("../../schema/invar-placement-v1.schema.json")).unwrap();
    let validator = jsonschema::validator_for(&schema).unwrap();
    assert!(validator.is_valid(&document()));
    let bad: [fn(&mut Value); 5] = [
        |v| {
            v["measurements"][0]
                .as_object_mut()
                .unwrap()
                .remove("points");
        },
        |v| {
            v.as_object_mut().unwrap().remove("coverage");
        },
        |v| {
            v["measurements"][0]["points"][1]["samples_ns"]
                .as_array_mut()
                .unwrap()
                .pop();
        },
        |v| v["statistics"]["samples"] = json!(5),
        |v| {
            v.as_object_mut().unwrap().remove("statistics");
            v.as_object_mut().unwrap().remove("coverage");
        },
    ];
    for (i, corrupt) in bad.into_iter().enumerate() {
        let mut v = document();
        corrupt(&mut v);
        assert!(!validator.is_valid(&v), "case {i}");
        assert!(parse(&v).is_err(), "case {i}");
    }
}

/// A named corruption of a profile's measurements.
type Edit = (&'static str, fn(&mut Value));

/// A CUDA `reduction` row from a forced-device source probe: whole-program
/// times on its own boundary, pricing no link.
fn source_device_row() -> Value {
    let boundary = placement::SOURCE_DEVICE_BOUNDARY;
    let c = json!({"fixed_ms":1.0,"per_work_ms":0.01,"link_bytes_per_ms":null,
        "link_latency_ms":0.0,"launch_floor_ms":0.0,"page_in_ms_per_byte":null});
    json!({"family":"placement.wrapping_sum.u32","table_family":"reduction","numeric":"u32-wrapping",
        "numeric_contract_version":placement::MAP_NUMERIC_CONTRACT,"engine":"cuda",
        "boundary":boundary,"region":null,"sizes":[1, 10000],"passes":[c,c,c,c,c],"best":c,
        "points":[point(Value::Null, placement::FIRST_CALL, &[9000]),
            point(json!(1), boundary, &samples()), point(json!(10000), boundary, &samples())]})
}

#[test]
fn v01_placement_calibration_source_device_rows_keep_their_own_boundary() {
    let mut v = document();
    v["measurements"]
        .as_array_mut()
        .unwrap()
        .push(source_device_row());
    v["coverage"]["pending"]
        .as_array_mut()
        .unwrap()
        .retain(|p| !(p["engine"] == "cuda" && p["family"] == "reduction"));
    let p = pinned(&v);
    assert!(p.calibrated(Engine::Cuda, "reduction"));
    assert!(p.calibrated(Engine::Cuda, "map"));
    let edits: [Edit; 4] = [
        // A family row may not claim the map probe's priced boundary...
        ("family row on host-map-v1", |m| {
            m[2]["boundary"] = json!("host-map-v1");
            for point in m[2]["points"].as_array_mut().unwrap().iter_mut().skip(1) {
                point["boundary"] = json!("host-map-v1");
            }
        }),
        ("family points off the row boundary", |m| {
            m[2]["points"][1]["boundary"] = json!("host-map-v1")
        }),
        // ...and the map probe may not shed its link pricing.
        ("map row on the source boundary", |m| {
            let boundary = placement::SOURCE_DEVICE_BOUNDARY;
            m[1]["boundary"] = json!(boundary);
            for point in m[1]["points"].as_array_mut().unwrap().iter_mut().skip(1) {
                point["boundary"] = json!(boundary);
            }
        }),
        ("map row without a link", |m| {
            let mut c = m[1]["best"].clone();
            c["link_bytes_per_ms"] = Value::Null;
            m[1]["passes"] = json!([c, c, c, c, c]);
            m[1]["best"] = c;
        }),
    ];
    for (name, edit) in edits {
        let mut bad = v.clone();
        edit(&mut bad["measurements"]);
        assert_eq!(parse(&bad).unwrap_err(), Error::Invalid, "{name}");
    }
}

#[test]
fn v01_placement_calibration_cli_profile_declares_the_protocol() {
    let dir = std::env::temp_dir().join(format!("invar-v01-calibrate-{}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("test.json");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["calibrate", "--host-class", "test", "--out"])
        .arg(&path)
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let receipt: Value = serde_json::from_slice(&output.stdout).unwrap();
    let v: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(receipt["coverage"], v["coverage"]);
    assert_eq!(v["statistics"], placement::statistics());
    assert!(Profile::load(&path, &v["identity"]).is_ok());
    let p = pinned(&v);
    let rows = v["measurements"].as_array().unwrap();
    for row in rows {
        assert_eq!(
            row["table_family"] == "map",
            row["family"] == placement::MAP_FAMILY
        );
        let points = row["points"].as_array().unwrap();
        assert_eq!(points.len(), 1 + row["sizes"].as_array().unwrap().len());
        for point in points {
            let n = point["samples_ns"].as_array().unwrap().len();
            assert_eq!(
                n,
                if point["boundary"] == "first_call" {
                    1
                } else {
                    30
                }
            );
        }
    }
    let pending = v["coverage"]["pending"].as_array().unwrap();
    assert_eq!(pending.len() + rows.len(), 30);
    for engine in [Engine::Cpu, Engine::Cuda, Engine::Metal] {
        for family in placement::FAMILIES {
            let measured = rows
                .iter()
                .any(|r| r["engine"] == engine.name() && r["table_family"] == family);
            assert_eq!(p.calibrated(engine, family), measured, "{family}");
        }
    }
    // With native code every CPU family is measured, each with its own ID.
    if cfg!(feature = "native") {
        assert!(!pending.iter().any(|p| p["engine"] == "cpu"));
        let ids = rows
            .iter()
            .filter(|r| r["engine"] == "cpu")
            .map(|r| r["family"].as_str().unwrap())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(ids.len(), 10);
    }
    // Beyond `map`, a device row is a forced-device source probe on its own
    // boundary, pricing no link; every other family is pending with its
    // refusal, never silently missing. The map probe can refuse on its own (a
    // shared device's noisy transfers), so a device ran when it has any row
    // or per-family refusal.
    for engine in ["cuda", "metal"] {
        let device_ran = rows.iter().any(|r| r["engine"] == engine)
            || v["refusals"]
                .as_array()
                .unwrap()
                .iter()
                .any(|r| r["engine"] == engine && r.get("family").is_some());
        if !device_ran {
            continue;
        }
        for family in placement::FAMILIES.iter().skip(1) {
            let row = rows
                .iter()
                .find(|r| r["engine"] == engine && r["table_family"] == *family);
            match row {
                Some(row) => {
                    assert_eq!(row["boundary"], placement::SOURCE_DEVICE_BOUNDARY);
                    assert!(row["best"]["link_bytes_per_ms"].is_null());
                }
                None => assert!(pending.iter().any(|p| p["engine"] == engine
                    && p["family"] == *family
                    && (p["reason"] == "device-plan-ineligible"
                        || p["reason"] == "calibration-probe-failed"))),
            }
        }
        // Both devices admit these two kernels (H100 in the CUDA family probes, M4 here).
        let kind = if engine == "cuda" {
            Engine::Cuda
        } else {
            Engine::Metal
        };
        for family in ["reduction", "scan"] {
            assert!(p.calibrated(kind, family), "{engine} {family}");
        }
    }
    // Without the native feature the CPU probe is refused, and every CPU
    // family is pending with that refusal rather than silently missing.
    if cfg!(not(feature = "native")) {
        assert!(pending.iter().any(|p| p["engine"] == "cpu"
            && p["family"] == "map"
            && p["reason"] == "calibration-probe-failed"));
    }
    std::fs::remove_dir_all(dir).unwrap();
}
