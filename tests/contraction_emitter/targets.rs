//! ENG-06 (#61) target descriptions: every file under `targets/` parses
//! strictly and against `schema/invar-target-v1.schema.json`, malformed files
//! are refused with the failing field, live capabilities select a file, and a
//! new target (sm_87) is a new file with no Rust change.

use std::path::{Path, PathBuf};

use invar_device::Capabilities;
use invar_device::target::{Backend, Error, Registry, SCHEMA, Target};
use serde_json::{Value, json};

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn files() -> Vec<(String, String)> {
    let mut files: Vec<_> = std::fs::read_dir(root().join("targets"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .map(|p| {
            let stem = p.file_stem().unwrap().to_str().unwrap().to_owned();
            (stem, std::fs::read_to_string(&p).unwrap())
        })
        .collect();
    files.sort();
    files
}

fn schema() -> jsonschema::Validator {
    let text = std::fs::read_to_string(root().join("schema/invar-target-v1.schema.json")).unwrap();
    jsonschema::validator_for(&serde_json::from_str(&text).unwrap()).unwrap()
}

fn sm90() -> Value {
    serde_json::from_str(&std::fs::read_to_string(root().join("targets/cuda-sm90.json")).unwrap())
        .unwrap()
}

fn parse(file: &str, doc: &Value) -> Result<Target, Error> {
    Target::parse(file, &doc.to_string())
}

/// A scratch directory unique to this process and label.
fn scratch(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("invar-eng-06-{label}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn builtin_targets_parse_and_match_the_files_on_disk() {
    let builtin = Registry::builtin().unwrap();
    let names: Vec<_> = builtin.targets().iter().map(|t| t.name.as_str()).collect();
    for expected in [
        "cuda-sm80",
        "cuda-sm86",
        "cuda-sm87",
        "cuda-sm89",
        "cuda-sm90",
        "cuda-sm120",
        "metal-apple7",
        "metal-apple8",
        "metal-apple9",
        "cpu-x86_64-zen4",
        "cpu-aarch64-apple-m4",
    ] {
        assert!(
            names.contains(&expected),
            "{expected} missing from {names:?}"
        );
    }
    // The embedded set is exactly the directory: nothing stale, nothing missed.
    assert_eq!(
        builtin,
        &Registry::from_dir(&root().join("targets")).unwrap()
    );
    let count = |backend| {
        builtin
            .targets()
            .iter()
            .filter(|t| t.device.backend == backend)
            .count()
    };
    assert!(count(Backend::Cuda) >= 6 && count(Backend::Metal) >= 3 && count(Backend::Cpu) >= 2);
    for t in builtin.targets() {
        assert_eq!(t.schema, SCHEMA);
        match t.device.backend {
            Backend::Cuda | Backend::Metal => {
                let gpu = t.gpu().unwrap();
                assert!(t.cpu().is_none());
                assert!(!gpu.copy_atoms.is_empty(), "{}", t.name);
                // Every target today's TC schedules run on (sm_80 and newer,
                // every Apple family) keeps one atom of their contract, with
                // its reference; older CUDA targets have none.
                let tc = gpu.mma_atoms.iter().any(|a| {
                    a.contract == "tc-f16-f32-v1"
                        && a.reference.as_deref() == Some("tc::reference::reference")
                });
                assert_eq!(tc, t.device.sm.is_none_or(|sm| sm >= 80), "{}", t.name);
            }
            Backend::Cpu => assert!(!t.cpu().unwrap().caches.is_empty()),
        }
    }
    // Hopper facts requested in #61 (2026-10-21).
    let hopper = builtin.get("cuda-sm90").unwrap().gpu().unwrap();
    assert!(
        hopper
            .mma_atoms
            .iter()
            .any(|a| a.name.starts_with("wgmma-"))
    );
    assert!(hopper.copy_atoms.iter().any(|c| c.name == "tma-tile"));
    assert!(hopper.copy_atoms.iter().any(|c| c.name == "cp-async-16"));
    assert!(hopper.warp_specialization.is_some());
}

#[test]
fn every_target_file_validates_against_the_schema() {
    let validator = schema();
    let files = files();
    assert!(files.len() >= 11);
    for (name, text) in &files {
        let doc: Value = serde_json::from_str(text).unwrap();
        let errors: Vec<_> = validator.iter_errors(&doc).map(|e| e.to_string()).collect();
        assert!(errors.is_empty(), "{name}: {errors:?}");
        Target::parse(name, text).unwrap();
    }
}

#[test]
fn malformed_and_unknown_field_files_are_refused() {
    let validator = schema();
    let base = sm90();
    let parse_refusals: [(&str, Value); 8] = [
        ("unknown top-level field", {
            let mut d = base.clone();
            d["vendor"] = json!("nvidia");
            d
        }),
        ("unknown gpu field", {
            let mut d = base.clone();
            d["gpu"]["l2_bytes"] = json!(1);
            d
        }),
        ("unknown atom field", {
            let mut d = base.clone();
            d["gpu"]["mma_atoms"][0]["sparse"] = json!(true);
            d
        }),
        ("unknown device field", {
            let mut d = base.clone();
            d["device"]["driver"] = json!(550);
            d
        }),
        ("wrong type", {
            let mut d = base.clone();
            d["gpu"]["subgroup_width"] = json!("32");
            d
        }),
        ("negative", {
            let mut d = base.clone();
            d["gpu"]["shared_bytes_per_group"] = json!(-1);
            d
        }),
        ("unknown element type", {
            let mut d = base.clone();
            d["gpu"]["mma_atoms"][0]["a_type"] = json!("f8");
            d
        }),
        ("missing required field", {
            let mut d = base.clone();
            d["gpu"].as_object_mut().unwrap().remove("mma_atoms");
            d
        }),
    ];
    for (label, doc) in &parse_refusals {
        assert!(
            matches!(parse("cuda-sm90", doc), Err(Error::Parse { .. })),
            "{label}"
        );
        assert!(!validator.is_valid(doc), "schema admits: {label}");
    }
    // Not JSON at all, and a duplicated key (which JSON Schema cannot see).
    for text in ["{", "[]", ""] {
        assert!(matches!(
            Target::parse("cuda-sm90", text),
            Err(Error::Parse { .. })
        ));
    }
    let duplicated = base
        .to_string()
        .replacen("\"schema\":", "\"source\":\"x\",\"schema\":", 1);
    assert!(matches!(
        Target::parse("cuda-sm90", &duplicated),
        Err(Error::Parse { detail, .. }) if detail.contains("duplicate")
    ));
}

#[test]
fn out_of_range_files_are_refused_with_the_field() {
    let validator = schema();
    let base = sm90();
    type Edit = fn(&mut Value);
    // (field reported, edit, whether JSON Schema can also express the rule)
    let cases: &[(&str, Edit, bool)] = &[
        ("schema", |d| d["schema"] = json!("invar-target-v2"), true),
        ("source", |d| d["source"] = json!(" "), false),
        ("device.sm", |d| d["device"]["sm"] = json!(7), true),
        (
            "gpu.subgroup_width",
            |d| d["gpu"]["subgroup_width"] = json!(48),
            true,
        ),
        (
            "gpu.subgroup_width",
            |d| d["gpu"]["subgroup_width"] = json!(0),
            true,
        ),
        (
            "gpu.max_threads_per_group",
            |d| d["gpu"]["max_threads_per_group"] = json!(1000),
            false,
        ),
        (
            "gpu.max_threads_per_group",
            |d| d["gpu"]["max_threads_per_group"] = json!(8192),
            true,
        ),
        (
            "gpu.shared_bytes_per_group",
            |d| d["gpu"]["shared_bytes_per_group"] = json!(0),
            true,
        ),
        (
            "gpu.shared_bytes_per_group",
            |d| d["gpu"]["shared_bytes_per_group"] = json!(300_000),
            false,
        ),
        (
            "gpu.shared_bytes_per_group_optin",
            |d| d["gpu"]["shared_bytes_per_group_optin"] = json!(2 << 20),
            true,
        ),
        (
            "gpu.registers_per_thread",
            |d| d["gpu"]["registers_per_thread"] = json!(8),
            true,
        ),
        (
            "gpu.memory_levels",
            |d| d["gpu"]["memory_levels"][1]["level"] = json!("global"),
            false,
        ),
        (
            "gpu.memory_levels.alignment_bytes",
            |d| d["gpu"]["memory_levels"][0]["alignment_bytes"] = json!(3),
            true,
        ),
        (
            "gpu.copy_atoms.name",
            |d| d["gpu"]["copy_atoms"][1]["name"] = json!("ld-global-4"),
            false,
        ),
        (
            "gpu.copy_atoms.name",
            |d| d["gpu"]["copy_atoms"][0]["name"] = json!("LD Global"),
            true,
        ),
        (
            "gpu.copy_atoms.kind",
            |d| d["gpu"]["copy_atoms"][6]["to"] = json!("registers"),
            false,
        ),
        (
            "gpu.copy_atoms.kind",
            |d| d["gpu"]["copy_atoms"][0]["to"] = json!("global"),
            false,
        ),
        (
            "gpu.copy_atoms.kind",
            |d| {
                drop(
                    d["gpu"]["copy_atoms"][9]
                        .as_object_mut()
                        .unwrap()
                        .remove("max_box_extent"),
                )
            },
            false,
        ),
        (
            "gpu.copy_atoms.bytes",
            |d| d["gpu"]["copy_atoms"][0]["bytes"] = json!(12),
            true,
        ),
        (
            "gpu.copy_atoms.alignment_bytes",
            |d| d["gpu"]["copy_atoms"][0]["alignment_bytes"] = json!(0),
            true,
        ),
        (
            "gpu.mma_atoms.name",
            |d| d["gpu"]["mma_atoms"][0]["name"] = json!("cp-async-16"),
            false,
        ),
        (
            "gpu.mma_atoms.instruction",
            |d| d["gpu"]["mma_atoms"][0]["instruction"] = json!(""),
            true,
        ),
        (
            "gpu.mma_atoms.m",
            |d| d["gpu"]["mma_atoms"][0]["m"] = json!(0),
            true,
        ),
        (
            "gpu.mma_atoms.k",
            |d| d["gpu"]["mma_atoms"][0]["k"] = json!(512),
            true,
        ),
        (
            "gpu.mma_atoms.threads",
            |d| d["gpu"]["mma_atoms"][0]["threads"] = json!(48),
            false,
        ),
        (
            "gpu.mma_atoms.threads",
            |d| d["gpu"]["mma_atoms"][0]["threads"] = json!(2048),
            false,
        ),
        (
            "gpu.mma_atoms.contract",
            |d| d["gpu"]["mma_atoms"][0]["contract"] = json!("tc-f16-f32"),
            true,
        ),
        (
            "gpu.mma_atoms.reference",
            |d| d["gpu"]["mma_atoms"][0]["reference"] = json!("tc::Reference"),
            true,
        ),
        (
            "gpu.warp_specialization",
            |d| d["gpu"]["warp_specialization"]["min_registers"] = json!(300),
            false,
        ),
        (
            "gpu.warp_specialization",
            |d| d["gpu"]["warp_specialization"]["max_registers"] = json!(250),
            false,
        ),
        (
            "gpu/cpu section",
            |d| d["cpu"] = json!({"simd_width_bits": 128, "isa": ["neon"], "caches": [{"level": 1, "bytes": 64, "line_bytes": 64, "shared_by_cores": 1, "core_class": null}], "numa_nodes": null}),
            true,
        ),
        (
            "gpu/cpu section",
            |d| drop(d.as_object_mut().unwrap().remove("gpu")),
            true,
        ),
        ("device", |d| d["device"]["apple_family"] = json!(9), true),
    ];
    let mut schema_checked = 0;
    for (field, edit, schema_expresses) in cases {
        let mut doc = base.clone();
        edit(&mut doc);
        match parse("cuda-sm90", &doc) {
            Err(Error::Range { field: got, .. }) => assert_eq!(&got, field),
            other => panic!("{field}: {other:?}"),
        }
        if *schema_expresses {
            assert!(!validator.is_valid(&doc), "schema admits bad {field}");
            schema_checked += 1;
        }
    }
    assert!(cases.len() >= 30 && schema_checked >= 15);
}

#[test]
fn names_must_match_the_file_and_the_device() {
    let base = sm90();
    assert!(matches!(parse("cuda-sm89", &base), Err(Error::Name { .. })));
    let mut renamed = base.clone();
    renamed["name"] = json!("cuda-sm89");
    assert!(matches!(
        parse("cuda-sm89", &renamed),
        Err(Error::Name { .. })
    ));
    let mut moved = base.clone();
    moved["device"]["sm"] = json!(89);
    assert!(matches!(
        parse("cuda-sm90", &moved),
        Err(Error::Name { .. })
    ));
    // Two files describing one target refuse the whole set.
    let text = base.to_string();
    assert!(matches!(
        Registry::parse([("cuda-sm90", text.as_str()), ("cuda-sm90", text.as_str())]),
        Err(Error::Duplicate { name }) if name == "cuda-sm90"
    ));
    // One bad file refuses the whole set.
    let files = files();
    let mut set: Vec<_> = files
        .iter()
        .map(|(n, t)| (n.as_str(), t.as_str()))
        .collect();
    set.push(("metal-apple10", "{}"));
    assert!(Registry::parse(set).is_err());
}

#[test]
fn live_capabilities_select_a_target_or_refuse() {
    let builtin = Registry::builtin().unwrap();
    let cuda = |sm| Capabilities::Cuda {
        major: 12,
        minor: 4,
        sm,
    };
    let metal = |apple_family| Capabilities::Metal {
        apple_family,
        msl_major: 3,
        msl_minor: 1,
    };
    let mut selected = 0;
    for (caps, name) in [
        (cuda(90), "cuda-sm90"),
        (cuda(86), "cuda-sm86"),
        (cuda(120), "cuda-sm120"),
        (metal(9), "metal-apple9"),
        (metal(7), "metal-apple7"),
    ] {
        assert_eq!(builtin.select(caps).unwrap().name, name);
        selected += 1;
    }
    for (caps, device) in [(cuda(30), "cuda-sm30"), (metal(6), "metal-apple6")] {
        assert_eq!(
            builtin.select(caps),
            Err(Error::Unknown {
                device: device.to_owned()
            })
        );
    }
    assert_eq!(
        builtin.select_cpu("x86_64", "zen4").unwrap().name,
        "cpu-x86_64-zen4"
    );
    assert_eq!(
        builtin.select_cpu("aarch64", "apple-m4").unwrap().name,
        "cpu-aarch64-apple-m4"
    );
    assert!(matches!(
        builtin.select_cpu("riscv64", "c910"),
        Err(Error::Unknown { .. })
    ));
    assert_eq!(selected, 5);
}

#[test]
fn a_new_target_is_a_new_file_with_no_rust_change() {
    // Jetson Orin (sm_87, #21) and the CPU targets: copy every other file,
    // confirm the device is unknown, add only its JSON file, and the same
    // compiled registry code now selects it.
    let files = files();
    let jetson = Capabilities::Cuda {
        major: 12,
        minor: 4,
        sm: 87,
    };
    for (new, select) in [
        ("cuda-sm87", None),
        ("cpu-x86_64-zen4", Some(("x86_64", "zen4"))),
        ("cpu-aarch64-apple-m4", Some(("aarch64", "apple-m4"))),
    ] {
        let dir = scratch(new);
        for (name, text) in files.iter().filter(|(n, _)| n != new) {
            std::fs::write(dir.join(format!("{name}.json")), text).unwrap();
        }
        // Non-JSON files are ignored.
        std::fs::write(dir.join("README.md"), "not a target").unwrap();
        let find = |r: &Registry| match select {
            None => r.select(jetson).map(|t| t.name.clone()),
            Some((arch, model)) => r.select_cpu(arch, model).map(|t| t.name.clone()),
        };
        let before = Registry::from_dir(&dir).unwrap();
        assert!(matches!(find(&before), Err(Error::Unknown { .. })), "{new}");
        let (_, text) = files.iter().find(|(n, _)| n == new).unwrap();
        std::fs::write(dir.join(format!("{new}.json")), text).unwrap();
        let after = Registry::from_dir(&dir).unwrap();
        assert_eq!(find(&after).unwrap(), new);
        assert_eq!(after.targets().len(), before.targets().len() + 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }
    assert!(matches!(
        Registry::from_dir(&root().join("no-such-targets-dir")),
        Err(Error::Io { .. })
    ));
}

#[test]
fn oversized_files_and_sets_are_refused_before_parsing() {
    use invar_device::target::{MAX_FILE_BYTES, MAX_FILES};
    // A valid target padded past the limit with whitespace is refused by size,
    // not parsed; one byte under the limit still parses.
    let text = sm90().to_string();
    let over = format!("{text}{}", " ".repeat(MAX_FILE_BYTES + 1 - text.len()));
    assert!(matches!(
        Target::parse("cuda-sm90", &over),
        Err(Error::Limit { .. })
    ));
    let under = format!("{text}{}", " ".repeat(MAX_FILE_BYTES - text.len()));
    Target::parse("cuda-sm90", &under).unwrap();
    // A huge file in a directory is refused without reading it in full.
    let dir = scratch("oversized");
    let file = std::fs::File::create(dir.join("cuda-sm90.json")).unwrap();
    file.set_len(1 << 30).unwrap();
    assert!(matches!(
        Registry::from_dir(&dir),
        Err(Error::Limit { what }) if what.contains("bytes")
    ));
    std::fs::remove_dir_all(&dir).unwrap();
    // More files than a registry holds are refused before any is read.
    let dir = scratch("many");
    for i in 0..=MAX_FILES {
        std::fs::write(dir.join(format!("cuda-sm{i}.json")), "").unwrap();
    }
    assert!(matches!(
        Registry::from_dir(&dir),
        Err(Error::Limit { what }) if what.contains("files")
    ));
    std::fs::remove_dir_all(&dir).unwrap();
    // Valid files, so the count, not a parse error, refuses the set.
    let many = vec![("cuda-sm90", text.as_str()); MAX_FILES + 1];
    assert!(matches!(Registry::parse(many), Err(Error::Limit { .. })));
}
