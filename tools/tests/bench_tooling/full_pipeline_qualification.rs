//! Issue #9 (M5-02).
#![forbid(unsafe_code)]
use crate::qualification;
use invar::{check, front, mir, schedule};
use serde_json::{Value, json};
use std::{fs, process::Command, sync::atomic::Ordering};

#[test]
fn full_pipeline_qualification_cpu_semantics_and_denominators() {
    let report = qualification::run(qualification::corpus::SEED).unwrap();
    assert_eq!(report["suite"], qualification::SUITE);
    let n = report["program_count"].as_u64().unwrap();
    assert!(n > 64);
    for count in report["engines"].as_object().unwrap().values() {
        assert_eq!(count["denominator"], n);
        assert_eq!(
            count["qualified"].as_u64().unwrap()
                + count["unavailable"].as_u64().unwrap()
                + count["blocked"].as_u64().unwrap(),
            n
        );
    }
    for engine in [
        "reference",
        "mir",
        "interaction-extracted",
        "net-extracted",
        "cpu-parallel-1",
        "cpu-parallel-2",
        "cpu-parallel-4",
        "net-oracle",
    ] {
        assert!(report["engines"][engine]["qualified"].as_u64().unwrap() > 0);
    }
    #[cfg(feature = "native")]
    for engine in ["native-serial", "native-net"] {
        assert!(report["engines"][engine]["qualified"].as_u64().unwrap() > 0);
    }
    for class in [
        "trap",
        "fuel",
        "resource",
        "refusal",
        "ownership",
        "map",
        "scan",
        "reduction",
        "floating",
        "generated-control",
    ] {
        assert!(report["class_counts"][class].as_u64().unwrap() > 0);
    }
    for family in ["map-257", "scan-257", "reduce-257"] {
        let case = report["programs"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["id"] == family)
            .unwrap();
        for workers in [2, 4] {
            let engine = format!("cpu-parallel-{workers}");
            let row = case["rows"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["engine"] == engine)
                .unwrap();
            assert_eq!(
                row["dispatch"]["workers_used"], workers,
                "{family} {engine}"
            );
            assert!(row["dispatch"]["published_regions"].as_u64().unwrap() > 0);
            assert!(row["dispatch"]["fallback_reason"].is_null());
        }
    }
    assert!(qualification::require_gpu(&report).is_err());
    println!(
        "M5-02 programs={n} engines={} corpus_sha256={}",
        report["engines"], report["corpus_sha256"]
    );
}

#[test]
fn full_pipeline_qualification_seeded_generation_and_bounded_front_fuzz() {
    let seed = qualification::corpus::SEED;
    assert_eq!(
        qualification::corpus::corpus(seed),
        qualification::corpus::corpus(seed)
    );
    assert_ne!(
        qualification::corpus::corpus(seed),
        qualification::corpus::corpus(seed + 1)
    );
    let fuzz = qualification::fuzz_boundary(seed).unwrap();
    assert_eq!(fuzz, qualification::fuzz_boundary(seed).unwrap());
    assert!(fuzz["accepted"].as_u64().unwrap() > 0);
    assert!(!fuzz["refused"].as_object().unwrap().is_empty());
    assert_eq!(
        fuzz["accepted"].as_u64().unwrap()
            + fuzz["refused"]
                .as_object()
                .unwrap()
                .values()
                .map(|x| x.as_u64().unwrap())
                .sum::<u64>(),
        128
    );
    println!("M5-02 boundary_fuzz={fuzz}");
}

#[test]
fn full_pipeline_qualification_mutations_are_detected() {
    let report = qualification::mutations().unwrap();
    assert_eq!(report["survived"], 0);
    assert!(report["detected"].as_u64().unwrap() >= 13);
    println!("M5-02 mutations={report}");
}

#[test]
fn full_pipeline_qualification_failure_injections_publish_nothing() {
    for body in [
        "ys[i] = wrapping_add(xs[i],7u32);",
        "acc = wrapping_add(acc,xs[i]);",
        "acc = wrapping_add(wrapping_mul(acc,3u32),xs[i]); ys[i] = acc;",
    ] {
        let source = format!(
            "fn main()->i32 {{ let xs = filled_u32(257u32,1u32); var ys = filled_u32(257u32,99u32); var acc = 11u32; for i in 0u32..len(xs) {{ {body} }} return 7; }}"
        );
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let program = mir::lower(&checked).unwrap();
        for (fault, code) in [
            (schedule::Fault::Spawn, "CPU_SPAWN"),
            (schedule::Fault::Join, "CPU_JOIN"),
            (schedule::Fault::Panic, "CPU_WORKER_PANIC"),
            (schedule::Fault::Allocation, "RESOURCE_LIMIT"),
            (schedule::Fault::Overlap, "CPU_CHUNK_OVERLAP"),
            (schedule::Fault::Timeout, "CPU_TIMEOUT"),
            (schedule::Fault::Cancel, "CPU_CANCELLED"),
        ] {
            let hooks = schedule::Hooks {
                fault,
                ..Default::default()
            };
            let error =
                schedule::run_source_with_hooks(source.as_bytes(), &program, 100_000, 4, &hooks)
                    .unwrap_err();
            assert_eq!(error.code(), code);
            assert_eq!(
                hooks.started.load(Ordering::SeqCst),
                hooks.joined.load(Ordering::SeqCst)
            );
            let live = hooks.live_arrays.lock().unwrap();
            let live = live.as_ref().unwrap();
            assert!(!live.before.is_empty());
            assert_eq!(live.before, live.after, "{fault:?}");
        }
    }
    println!("M5-02 failure_injections=21 unpublished=21");
}

#[test]
fn full_pipeline_qualification_missing_gpu_and_report_failure_are_atomic() {
    let directory = std::env::temp_dir().join(format!("invar-m502-report-{}", std::process::id()));
    fs::create_dir(&directory).unwrap();
    let path = directory.join("report.json");
    let failure = json!({"status":"failed"});
    assert!(qualification::publish(&path, &failure).is_err());
    assert!(!path.exists());
    let complete = json!({"status":"cpu_semantics_passed_device_qualification_pending","programs":[{"frontend":"accepted","id":"control","rows":[{"engine":"cuda-generated","status":"unavailable","diagnostic":"M5_DEVICE_NOT_REQUESTED"}]}]});
    assert_eq!(
        qualification::require_gpu(&complete).unwrap_err().code,
        "M5_GPU_UNAVAILABLE"
    );
    qualification::publish(&path, &complete).unwrap();
    let before = fs::read(&path).unwrap();
    assert!(qualification::publish(&path, &complete).is_err());
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);
    // Usage failure occurs before any work or report publication.
    let output = Command::new(env!("CARGO_BIN_EXE_invar-qualify"))
        .args(["--report", path.to_str().unwrap(), "--unknown"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let diagnostic: Value = serde_json::from_slice(&output.stderr).unwrap();
    assert_eq!(diagnostic["diagnostic"], "M5_USAGE");
    assert_eq!(fs::read(&path).unwrap(), before);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn full_pipeline_qualification_frontend_and_lowering_resource_limits() {
    let too_large = vec![b' '; 1_048_577];
    let error = front::parse(&too_large).unwrap_err();
    assert_eq!(error.code, "SOURCE_SIZE_LIMIT");
    assert_eq!(error.kind, front::ErrorKind::Tool);
    let checked =
        check::check(front::parse(b"fn main()->i32 { return 7; }").unwrap(), true).unwrap();
    let mut program = mir::lower(&checked).unwrap();
    let function = program.functions[0].clone();
    program.functions.resize(mir::MAX_FUNCTIONS + 1, function);
    assert_eq!(mir::verify(&program).unwrap_err().code, "RESOURCE_LIMIT");
    let checked =
        check::check(front::parse(b"fn main()->i32 { return 7; }").unwrap(), true).unwrap();
    let net = invar::general_net::Net::lower(&checked, Default::default()).unwrap();
    let proof = net.verify().unwrap();
    for (limits, steps) in [
        (
            invar::general_net::oracle::Limits {
                transitions: 0,
                ..Default::default()
            },
            0,
        ),
        (
            invar::general_net::oracle::Limits {
                trace_events: 0,
                ..Default::default()
            },
            2,
        ),
    ] {
        let actual = invar::general_net::oracle::observe(&net, &proof, limits).unwrap();
        assert!(matches!(
            actual.semantic.result,
            Err(invar::eval::EvalError::Tool {
                code: "RESOURCE_LIMIT",
                ..
            })
        ));
        assert!(actual.failure.is_some());
        // Tick instructions account for return and literal before the first Demand.
        assert_eq!(actual.semantic.steps_used, steps);
        assert!(actual.semantic.cleanup_trace.is_empty());
    }
    assert!(qualification::require_gpu(&json!({"programs":[]})).is_err());
    println!("M5-02 resource_injections=4 refused=4");
}
