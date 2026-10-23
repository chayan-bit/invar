//! Live MIR dispatch, auto choice, CLI explain and the M2 corpus on Metal.

use super::*;
use invar::schedule::metal::Kernels;

#[test]
#[ignore = "requires a real Apple7+ Metal GPU"]
fn issue_m3_06_metal_gpu_forced_live_mir_matches_reference() {
    let _gpu_lock = GPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for source in [
        MAP, ZIP_MAP, ZIP_DAG, SUM, PRODUCT, INCLUSIVE, EXCLUSIVE, FINAL,
    ] {
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let program = mir::lower(&checked).unwrap();
        let reference = invar::eval::observe(&checked, 1_000_000);
        let image = prebuilt(source.as_bytes());
        let run = invar::schedule::metal::run_source(
            source.as_bytes(),
            &program,
            1_000_000,
            Kernels::Prebuilt(&image),
        )
        .unwrap();
        assert_eq!(run.observation.result, reference.result, "{source}");
        assert_eq!(run.observation.steps_used, reference.steps_used, "{source}");
        assert_eq!(
            run.observation.cleanup_trace, reference.cleanup_trace,
            "{source}"
        );
        assert!(!run.executed.is_empty(), "{source}");
    }
}

#[test]
#[ignore = "requires a real Apple7+ Metal GPU"]
fn issue_m3_06_metal_gpu_auto_calibration_and_choice_match_execution() {
    let _gpu_lock = GPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let image = prebuilt(MAP.as_bytes());
    let model = calibrate(MAP.as_bytes(), &image);
    assert!(model.cpu_bytes_per_ms.is_finite() && model.device_bytes_per_ms.is_finite());
    let checked = check::check(front::parse(MAP.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    let run = invar::schedule::metal::run_auto_source(
        MAP.as_bytes(),
        &program,
        1_000_000,
        Kernels::Prebuilt(&image),
    )
    .unwrap();
    let reference = invar::eval::observe(&checked, 1_000_000);
    assert_eq!(run.observation.result, reference.result);
    assert_eq!(run.observation.steps_used, reference.steps_used);
    assert!(
        run.choices
            .iter()
            .all(|choice| matches!(choice.backend, "metal" | "cpu-parallel" | "mir-serial"))
    );
}

#[test]
#[ignore = "requires a real Apple7+ Metal GPU"]
fn issue_m3_06_metal_gpu_cli_explain_matches_forced_execution() {
    let _gpu_lock = GPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let path = std::env::temp_dir().join(format!("invar-m3-06-{}.inv", std::process::id()));
    std::fs::write(&path, MAP).unwrap();
    let library = path.with_extension("metallibc");
    let binary = env!("CARGO_BIN_EXE_invar");
    let built = std::process::Command::new(binary)
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "metal",
            "--output",
        ])
        .arg(&library)
        .output()
        .unwrap();
    assert!(
        built.status.success(),
        "{}",
        String::from_utf8_lossy(&built.stdout)
    );
    // Every command runs the compiled-ahead library with the tripwire armed.
    let command = |args: &[&str]| {
        std::process::Command::new(binary)
            .env(invar_metal::SOURCE_TRIPWIRE, "1")
            .args(args)
            .output()
            .unwrap()
    };
    let (source, artifact) = (path.to_str().unwrap(), library.to_str().unwrap());
    let run = command(&[
        "run",
        source,
        "--engine",
        "metal",
        "--artifact",
        artifact,
        "--format",
        "json",
    ]);
    let explain = command(&[
        "explain",
        source,
        "--engine",
        "metal",
        "--artifact",
        artifact,
        "--format",
        "json",
    ]);
    let auto = command(&[
        "explain",
        source,
        "--engine",
        "auto",
        "--artifact",
        artifact,
        "--format",
        "json",
    ]);
    std::fs::remove_file(&path).unwrap();
    std::fs::remove_file(&library).unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stdout)
    );
    assert!(
        explain.status.success(),
        "{}",
        String::from_utf8_lossy(&explain.stdout)
    );
    assert!(
        auto.status.success(),
        "{}",
        String::from_utf8_lossy(&auto.stdout)
    );
    let run: serde_json::Value = serde_json::from_slice(&run.stdout).unwrap();
    let explain: serde_json::Value = serde_json::from_slice(&explain.stdout).unwrap();
    let auto: serde_json::Value = serde_json::from_slice(&auto.stdout).unwrap();
    assert_eq!(
        run["result"]["value"],
        explain["device_execution"]["result"]
    );
    assert_eq!(
        explain["device_execution"]["choices"][0]["backend"],
        "metal"
    );
    assert_eq!(auto["device_execution"]["requested_engine"], "auto");
}

#[test]
#[ignore = "requires the exported 252-program M2 corpus and a real Apple7+ GPU"]
fn issue_m3_06_metal_gpu_full_m2_probe_corpus() {
    let _gpu_lock = GPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    use invar::{
        eval,
        schedule::{self, metal},
    };
    let directory = std::env::var_os("INVAR_M3_06_CORPUS_DIR")
        .expect("INVAR_M3_06_CORPUS_DIR must name the exported M2 probe corpus");
    let mut files = std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "invar"))
        .collect::<Vec<_>>();
    files.sort();
    assert_eq!(files.len(), 252);
    let mut counts = (0usize, 0usize, 0usize);
    for file in files {
        let source = std::fs::read(&file).unwrap();
        let checked = check::check(front::parse(&source).unwrap(), true).unwrap();
        let program = mir::lower(&checked).unwrap();
        let report = admission::analyze(&source, &program).unwrap();
        let eligible = report
            .regions()
            .iter()
            .any(|region| region.plan_kind() != invar::plan::PlanKind::Serial);
        let reference = eval::observe(&checked, 1_000_000);
        let image = prebuilt(&source);
        let model = calibrate(&source, &image);
        let mut device = metal::open_device(&source, Kernels::Prebuilt(&image)).unwrap();
        let cpu = schedule::run_source(&source, &program, 1_000_000, 4).unwrap();
        assert_eq!(cpu.observation.result, reference.result, "CPU {file:?}");
        assert_eq!(
            cpu.observation.steps_used, reference.steps_used,
            "CPU {file:?}"
        );
        assert_eq!(
            cpu.observation.cleanup_trace, reference.cleanup_trace,
            "CPU {file:?}"
        );
        let auto = metal::run_auto_source_with_fault(
            &source,
            &program,
            1_000_000,
            Kernels::Prebuilt(&image),
            model,
            metal::Fault::None,
        )
        .unwrap();
        assert_eq!(auto.observation.result, reference.result, "auto {file:?}");
        assert_eq!(
            auto.observation.steps_used, reference.steps_used,
            "auto {file:?}"
        );
        assert_eq!(
            auto.observation.cleanup_trace, reference.cleanup_trace,
            "auto {file:?}"
        );
        match metal::run_source_with_device(&source, &program, 1_000_000, &mut device) {
            Ok(run) => {
                assert!(eligible && !run.executed.is_empty(), "{file:?}");
                assert_eq!(run.observation.result, reference.result, "Metal {file:?}");
                assert_eq!(
                    run.observation.steps_used, reference.steps_used,
                    "Metal {file:?}"
                );
                assert_eq!(
                    run.observation.cleanup_trace, reference.cleanup_trace,
                    "Metal {file:?}"
                );
                counts.0 += 1;
            }
            Err(error) if !eligible && error.code == "METAL_UNAVAILABLE" => counts.2 += 1,
            Err(error)
                if eligible
                    // Only zero-trip regions were reached, so none could publish.
                    && cpu.executed.is_empty()
                    && error.code == "METAL_UNAVAILABLE"
                    && error.message == "no Metal result was published" =>
            {
                counts.1 += 1
            }
            Err(error) => panic!("unexpected forced Metal failure for {file:?}: {error:?}"),
        }
    }
    assert_eq!(counts, (120, 24, 108));
    eprintln!(
        "Metal M2 corpus 252/252: forced={}, empty={}, ineligible={}",
        counts.0, counts.1, counts.2
    );
}

/// Large eligible regions over filled inputs: each family's first region is
/// the device candidate, and every later statement is scalar.
fn large_auto_corpus(len: u32) -> Vec<(&'static str, String)> {
    let head =
        format!("fn main() -> i32 {{ let xs: Array<u32> = filled_u32({len}u32, 2654435761u32); ");
    let tail = "if out[1u32] == out[0u32] { return 1; } return 0; }";
    let mut chain = head.clone();
    let mut input = "xs".to_owned();
    for stage in 0..16 {
        let op = if stage % 2 == 0 {
            "wrapping_mul"
        } else {
            "wrapping_add"
        };
        chain.push_str(&format!(
            "var o{stage}: Array<u32> = filled_u32(len({input}), 0u32); \
             for i{stage} in 0u32 .. len({input}) {{ o{stage}[i{stage}] = {op}({input}[i{stage}], {}u32); }} \
             let s{stage}: Array<u32> = move o{stage}; ",
            2 * stage + 3
        ));
        input = format!("s{stage}");
    }
    chain.push_str(&format!("let out: Array<u32> = move {input}; {tail}"));
    vec![
        (
            "map",
            format!(
                "{head}var out: Array<u32> = filled_u32(len(xs), 0u32); \
                 for i in 0u32 .. len(xs) {{ out[i] = wrapping_mul(xs[i], 7u32); }} {tail}"
            ),
        ),
        ("chain16", chain),
        (
            "sum",
            format!(
                "{head}var s: u32 = 11u32; for i in 0u32 .. len(xs) {{ s = wrapping_add(s, xs[i]); }} \
                 if s == 0u32 {{ return 1; }} return 0; }}"
            ),
        ),
        (
            "scan",
            format!(
                "{head}var s: u32 = 5u32; var out: Array<u32> = filled_u32(len(xs), 0u32); \
                 for i in 0u32 .. len(xs) {{ s = wrapping_add(wrapping_mul(1664525u32, s), xs[i]); out[i] = s; }} {tail}"
            ),
        ),
    ]
}

#[test]
#[ignore = "requires a real Apple7+ Metal GPU"]
fn issue_m3_06_metal_gpu_auto_picks_device_for_large_regions() {
    let _gpu_lock = GPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for len in [65_536u32, 262_144, 1_000_000] {
        for (family, source) in large_auto_corpus(len) {
            let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
            let program = mir::lower(&checked).unwrap();
            let fuel = 100_000_000;
            let image = prebuilt(source.as_bytes());
            let model = calibrate(source.as_bytes(), &image);
            eprintln!("model={model:?}");
            let run = invar::schedule::metal::run_auto_source(
                source.as_bytes(),
                &program,
                fuel,
                Kernels::Prebuilt(&image),
            )
            .unwrap();
            let choices = run
                .choices
                .iter()
                .map(|c| (c.backend, c.reason, c.cost.map(|c| (c.cpu_ms, c.device_ms))))
                .collect::<Vec<_>>();
            eprintln!("len={len} {family}: {:?}", choices.first());
            let reference = invar::eval::observe(&checked, fuel);
            assert_eq!(run.observation.result, reference.result, "{family} {len}");
            assert_eq!(
                run.observation.steps_used, reference.steps_used,
                "{family} {len}"
            );
        }
    }
}

#[test]
#[ignore = "requires a real Apple7+ Metal GPU"]
fn issue_m3_06_metal_gpu_slot_overflow_is_ineligible_not_a_toolchain_error() {
    let _gpu_lock = GPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    // Sixteen distinct arrays zipped through fifteen stages: 16 inputs, 15
    // outputs and n need 32 buffer slots, one more than Apple GPUs provide.
    let mut source = String::from("fn main() -> i32 { ");
    for k in 0..16 {
        source.push_str(&format!("let in{k}: Array<u32> = [{k}u32, 1u32, 2u32]; "));
    }
    let mut input = "in0".to_owned();
    for k in 0..15 {
        source.push_str(&format!(
            "var o{k}: Array<u32> = filled_u32(len({input}), 0u32); \
             for i{k} in 0u32 .. len({input}) {{ o{k}[i{k}] = wrapping_add({input}[i{k}], in{}[i{k}]); }} \
             let s{k}: Array<u32> = move o{k}; ",
            k + 1
        ));
        input = format!("s{k}");
    }
    source.push_str(&format!(
        "if {input}[1u32] == 16u32 {{ return 0; }} return 1; }}"
    ));
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    let reference = invar::eval::observe(&checked, 1_000_000);
    assert_eq!(reference.result, Ok(invar::eval::Value::I32(0)));
    // A reached region's refusal surfaces as the run's tool error.
    let image = prebuilt(source.as_bytes());
    let forced = invar::schedule::metal::run_source(
        source.as_bytes(),
        &program,
        1_000_000,
        Kernels::Prebuilt(&image),
    )
    .expect("admission succeeds");
    assert!(forced.executed.is_empty());
    let Err(invar::eval::EvalError::Tool { code, message }) = forced.observation.result else {
        panic!(
            "forced Metal must refuse the region: {:?}",
            forced.observation.result
        );
    };
    assert_eq!(code, "METAL_UNAVAILABLE", "{message}");
    assert!(message.contains("32 Metal buffer slots"), "{message}");
    let auto = invar::schedule::metal::run_auto_source(
        source.as_bytes(),
        &program,
        1_000_000,
        Kernels::Prebuilt(&image),
    )
    .unwrap();
    assert_eq!(auto.observation.result, reference.result);
    assert_eq!(auto.observation.steps_used, reference.steps_used);
    let first = auto.choices.first().unwrap();
    assert_eq!(
        (first.backend, first.reason),
        ("cpu-parallel", "metal-buffer-slot-limit")
    );
}
