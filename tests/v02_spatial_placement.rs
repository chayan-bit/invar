use serde_json::Value;

fn schema(name: &str) -> Value {
    serde_json::from_slice(&std::fs::read(format!("schema/{name}")).unwrap()).unwrap()
}

#[test]
fn spatial_explain_uses_registered_families_and_strict_v4() {
    let registry = schema("diagnostic-codes-v1.json");
    assert!(
        registry["explain_reason_codes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == invar::spatial::placement::UNMEASURED)
    );
    for engine in [None, Some("auto")] {
        let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_invar"));
        command.args(["explain", "demo/spatial-checker.inv"]);
        if let Some(engine) = engine {
            command.args(["--engine", engine]);
        }
        command.args(["--format", "json"]);
        let output = command.output().unwrap();
        assert!(output.status.success(), "{:?}", output);
        assert!(output.stderr.is_empty());
        let document: Value = serde_json::from_slice(&output.stdout).unwrap();
        // #216: an auto explain is the unified v5, the same spatial sites as v4.
        let v4 = schema(if engine.is_some() {
            "explain-v5.schema.json"
        } else {
            "explain-v4.schema.json"
        });
        assert!(jsonschema::is_valid(&v4, &document), "{document}");
        assert!(!jsonschema::is_valid(
            &schema("explain-v3.schema.json"),
            &document
        ));
        let sites = document["explicit_spatial"].as_array().unwrap();
        assert_eq!(sites.len(), 4);
        for site in sites {
            let op = site["operation"].as_str().unwrap();
            assert_eq!(site["family_id"], format!("spatial.{op}.f32"));
            assert_eq!(site["reason"], invar::spatial::placement::UNMEASURED);
        }
        if engine.is_some() {
            assert!(jsonschema::is_valid(
                &schema("invar-explain-device-v5.schema.json"),
                &document["device_execution"]
            ));
            for choice in document["device_execution"]["choices"].as_array().unwrap() {
                assert_eq!(choice["backend"], "cpu-parallel");
                assert!(
                    choice["family_id"]
                        .as_str()
                        .unwrap()
                        .starts_with("spatial.")
                );
            }
        }
        for (key, wrong) in [
            ("family_id", "spatial.knn.f64"),
            ("reason", "unregistered-spatial-reason"),
            ("operation", "lbvh_build"),
        ] {
            let mut bad = document.clone();
            bad["explicit_spatial"][0][key] = wrong.into();
            assert!(!jsonschema::is_valid(&v4, &bad));
        }
        let mut mismatched = document.clone();
        mismatched["explicit_spatial"][0]["family_id"] = "spatial.knn.f32".into();
        assert!(!jsonschema::is_valid(&v4, &mismatched));
        let mut extra = document;
        extra["explicit_spatial"][0]["invented_cost"] = 0.0.into();
        assert!(!jsonschema::is_valid(&v4, &extra));
    }
    assert_eq!(invar::spatial::placement::family_id("sort"), None);
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
#[test]
#[ignore = "physical CUDA or Metal device"]
fn spatial_forced_explain_distinguishes_device_and_host_work() {
    let demo = std::fs::read_to_string("demo/spatial-checker.inv").unwrap();
    let source = demo.replace(
        "fn main() -> i32 {",
        "fn main() -> i32 { var keys: Array<u32> = [2u32,1u32]; sort(keys);",
    );
    let file =
        std::env::temp_dir().join(format!("spatial-forced-explain-{}.inv", std::process::id()));
    std::fs::write(&file, source).unwrap();
    let engines = [
        #[cfg(feature = "cuda")]
        "cuda",
        #[cfg(all(feature = "metal", target_os = "macos"))]
        "metal",
    ];
    for engine in engines {
        let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_invar"));
        command.arg("explain").arg(&file).args(["--engine", engine]);
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if engine == "metal" {
            command.arg("--artifact").arg(metal_library(&file));
        }
        command.args(["--format", "json"]);
        let output = command.output().unwrap();
        assert!(output.status.success(), "{:?}", output);
        let doc: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert!(jsonschema::is_valid(
            &schema("explain-v4.schema.json"),
            &doc
        ));
        assert_eq!(doc["device_execution"]["result"], 3);
        let choices = doc["device_execution"]["choices"].as_array().unwrap();
        assert_eq!(choices.len(), 5);
        let mut host = 0;
        let mut device = 0;
        for choice in choices {
            let implementation = choice["implementation"].as_str().unwrap();
            if implementation.starts_with("cpu-") || implementation.starts_with("host-") {
                host += 1;
                assert_eq!(choice["backend"], "cpu-serial");
                assert_eq!(choice["reason"], invar::spatial::placement::HOST_WORK);
            } else {
                device += 1;
                assert_eq!(choice["backend"], engine);
                assert_eq!(choice["reason"], "forced-device");
                assert!(choice["family_id"].is_string());
            }
        }
        assert_eq!((host, device), (2, 3)); // explicit sort + grid_excluded
    }
    std::fs::remove_file(file).unwrap();
}

#[cfg(feature = "cuda")]
#[test]
fn spatial_unmeasured_family_ignores_generic_bandwidth_model() {
    use invar::{plan::DeviceCostModel, schedule::cuda};
    let source = std::fs::read("demo/spatial-checker.inv").unwrap();
    let checked = invar::check::check(invar::front::parse(&source).unwrap(), true).unwrap();
    let program = invar::mir::lower(&checked).unwrap();
    let run = cuda::run_auto_source_with_fault(
        &source,
        &program,
        100000,
        std::path::Path::new("/unavailable/spatial-nvrtc"),
        DeviceCostModel {
            cpu_bytes_per_ms: 0.001,
            device_bytes_per_ms: 1e30,
            device_fixed_ms: 0.0,
        },
        cuda::Fault::None,
    )
    .unwrap();
    assert_eq!(run.choices.len(), 4);
    assert!(run.choices.iter().all(|c| c.backend == "cpu-parallel"
        && c.reason == invar::spatial::placement::UNMEASURED
        && c.cost.is_none()));
    assert_eq!(run.cache_hits, 0);
}

/// A production Metal device runs only compiled-ahead libraries.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_library(file: &std::path::Path) -> std::path::PathBuf {
    let library = file.with_extension("metallib");
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_invar"))
        .arg("build")
        .arg(file)
        .args(["--target", "metal", "--output"])
        .arg(&library)
        .status()
        .unwrap();
    assert!(status.success());
    library
}
