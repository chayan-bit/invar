//! GNET-03 CPU-side suite (runs by default): ordinary-source device admission
//! over the whole runnable corpus with typed refusals, program-specific emitted
//! CUDA/Metal structure, and the emitted device bodies executed by the CPU
//! group model against the bounded CPU net oracle. The model is never a GPU
//! claim; `gpu_cuda::general_net_abi5` and `gpu_metal::general_net_abi5` run the same
//! harness on actual devices.
#![forbid(unsafe_code)]

use crate::device;
use crate::executable;
use crate::parallel;

use device::{Denominator, corpus};
use invar::general_net::{
    compiled::Target,
    device_host::{Limits, Plan, cpu::CpuModel, refusal_class},
};
use std::collections::BTreeMap;

/// Typed classes a checked runnable program may be refused with. Anything else
/// (an internal compiler state, a host protocol error) is a defect.
const REFUSAL_CLASSES: [&str; 6] = [
    "unsupported_type",
    "unsupported_call",
    "unsupported_rule",
    "compiler_structural_budget",
    "net_structural_budget",
    "owner_entry_result",
];

#[test]
fn device_model_corpus_admission_is_typed_and_target_independent() {
    let mut tally: BTreeMap<&str, usize> = BTreeMap::new();
    let mut runnable = 0;
    for source in corpus() {
        runnable += 1;
        let [cuda, metal] = [Target::Cuda, Target::Metal]
            .map(|t| Plan::from_source(&source.bytes, "main", t, Limits::default()));
        match (&cuda, &metal) {
            (Ok(c), Ok(m)) => {
                assert!(
                    matches!(c.artifact().abi_version(), 5 | 13),
                    "{}",
                    source.name
                );
                assert_eq!(
                    m.artifact().abi_version(),
                    c.artifact().abi_version(),
                    "{}",
                    source.name
                );
                // Identical program-specific continuation states on both targets.
                assert_eq!(
                    c.artifact().states(),
                    m.artifact().states(),
                    "{}",
                    source.name
                );
                *tally.entry("admitted").or_default() += 1;
            }
            (Err(c), Err(m)) => {
                let class = refusal_class(c);
                assert_eq!(class, refusal_class(m), "{}", source.name);
                assert!(REFUSAL_CLASSES.contains(&class), "{}: {c:?}", source.name);
                *tally.entry(class).or_default() += 1;
            }
            _ => panic!("{}: targets disagree on admission", source.name),
        }
    }
    println!("general_net_abi5 admission runnable={runnable} {tally:?}");
    // Owners are device data paths: arrays no longer force a refusal.
    assert!(tally["admitted"] >= 250, "{tally:?}");
    assert_eq!(runnable, tally.values().sum::<usize>());
}

#[test]
fn device_model_owner_programs_and_validated_owner_results_are_admitted() {
    let owned = "fn f(n: u32) -> i32 { let xs = filled_i32(n, 4); return xs[0u32]; } fn main() -> i32 { return f(3u32); }";
    let plan =
        Plan::from_source(owned.as_bytes(), "main", Target::Metal, Limits::default()).unwrap();
    assert!(plan.artifact().owned_layout().unwrap().heap_words() > 0);
    // ABI13 publishes an owner handle into the completed, validated root heap.
    let result = "fn run(n: u32) -> Array<i32> { return filled_i32(n, 1); }";
    let returned =
        Plan::from_source(result.as_bytes(), "run", Target::Cuda, Limits::default()).unwrap();
    assert_eq!(returned.artifact().abi_version(), 13);
    assert!(returned.artifact().entry_layout().is_some());
    // ABI13 admits borrowed array parameters.
    let param = "fn run(xs: read Array<i32>) -> i32 { return xs[0u32]; }";
    let plan = Plan::from_source(param.as_bytes(), "run", Target::Cuda, Limits::default()).unwrap();
    assert_eq!(plan.artifact().abi_version(), 13);
    // Flat records and strict f32 use the extended checked entry packet.
    for source in [
        "struct P { x: i32 } fn main() -> i32 { let p = P { x: 1 }; return p.x; }",
        "fn main() -> i32 { let x: f32 = 1.5; if x > 1.0 { return 1; } return 0; }",
    ] {
        let plan =
            Plan::from_source(source.as_bytes(), "main", Target::Cuda, Limits::default()).unwrap();
        assert_eq!(plan.artifact().abi_version(), 13);
    }
}

fn count(text: &str, needle: &str) -> usize {
    text.matches(needle).count()
}

#[test]
fn device_model_emitted_code_is_program_specific_without_interpreter() {
    let sources = [
        "fn tree(n: u32) -> u32 { if n < 2u32 { return n; } let a = tree(n - 1u32); let b = tree(n - 2u32); return wrapping_add(a, b); } fn main() -> i32 { if tree(9u32) > 30u32 { return 1; } return 0; }",
        "fn main() -> i32 { var acc = 0; for i in 0u32..4u32 { let xs = filled_i32(2u32, 3); acc = acc + xs[0u32]; } return acc; }",
    ];
    let mut seen = Vec::new();
    for source in sources {
        for target in [Target::Cuda, Target::Metal] {
            let plan =
                Plan::from_source(source.as_bytes(), "main", target, Limits::default()).unwrap();
            let artifact = plan.artifact();
            let code = artifact.source();
            assert_eq!(artifact.abi_version(), 5);
            let kernel = artifact.device_layout().unwrap().kernel_name();
            assert_eq!(
                count(code, kernel),
                1,
                "{target:?}: exactly one entry kernel"
            );
            // One generated continuation per program-specific state; the only
            // dispatch is on the frame's own continuation, not an opcode stream.
            let states = (0..artifact.states())
                .filter(|pc| code.contains(&format!("case {pc}: {{")))
                .count();
            assert_eq!(states, artifact.states(), "{target:?}");
            for word in ["opcode", "interpret", "bytecode"] {
                assert!(!code.to_ascii_lowercase().contains(word), "{word}");
            }
            // Barriers separate root mutation, private children and publication.
            let barrier = if target == Target::Cuda {
                "__syncthreads()"
            } else {
                "threadgroup_barrier("
            };
            assert!(code.contains(barrier), "{target:?}");
            // Every discovered independent call site has its own generated
            // discovery routine; a program without candidates has none.
            let plan_sites = artifact.fork_plan().unwrap().candidates.len();
            assert_eq!(code.contains("gnet_discover_"), plan_sites > 0);
            seen.push(code.to_owned());
        }
    }
    seen.sort();
    seen.dedup();
    assert_eq!(
        seen.len(),
        4,
        "every program/target pair emits distinct code"
    );
}

/// Corpus parts CI runs as separate jobs (`INVAR_CORPUS_PART=1/2`, `2/2` in
/// .github/workflows/ci.yml).
const CI_CORPUS_PARTS: usize = 2;

#[test]
fn device_model_corpus_parts_cover_the_corpus() {
    // No device work: the CI parts are disjoint, non-empty, together exactly
    // the programs the unsplit test runs, and still above the corpus floor.
    let names = |part| -> Vec<String> {
        device::part_sources(part, true)
            .into_iter()
            .map(|source| source.name)
            .collect()
    };
    let mut whole = names(device::sample::WHOLE);
    let mut union = Vec::new();
    for index in 0..CI_CORPUS_PARTS {
        let part = names((index, CI_CORPUS_PARTS));
        assert!(
            !part.is_empty(),
            "corpus part {} of {CI_CORPUS_PARTS} is empty",
            index + 1
        );
        union.extend(part);
    }
    whole.sort();
    union.sort();
    assert_eq!(union, whole, "the corpus parts must partition the corpus");
    assert!(
        whole.len() >= device::sample::floor(250),
        "{} runnable programs",
        whole.len()
    );
}

#[test]
fn device_model_corpus_part_selector_reads_k_of_n() {
    assert_eq!(device::sample::parse_part(""), device::sample::WHOLE);
    assert_eq!(device::sample::parse_part("1/1"), (0, 1));
    assert_eq!(device::sample::parse_part("1/2"), (0, 2));
    assert_eq!(device::sample::parse_part("2/2"), (1, 2));
    for bad in ["0/2", "3/2", "2", "1/0", "a/2", "1/2/3", " 1/2"] {
        let parsed = std::panic::catch_unwind(|| device::sample::parse_part(bad));
        assert!(parsed.is_err(), "{bad:?} must be rejected");
    }
}

#[test]
fn device_model_corpus_cpu_model_agrees_with_oracle() {
    // Emitted CUDA bodies at full fuel; emitted Metal bodies at full fuel and
    // one source step short. Both run the same deterministic corpus sample
    // (every admitted runnable program under INVAR_FULL_CORPUS=1), or the part
    // of it that INVAR_CORPUS_PART selects.
    // Four shards per runner bound compiler contention across the shared CI pool.
    // The CUDA and Metal halves run at once: 2 x 4 shard threads fill the
    // test's 8-thread nextest reservation.
    let part = device::sample::corpus_part();
    let (cuda, metal): (Denominator, Denominator) = std::thread::scope(|scope| {
        let cuda = scope.spawn(|| {
            device::sharded(
                Target::Cuda,
                "cpu-model-cuda",
                CpuModel::default,
                false,
                part,
                4,
                true,
            )
        });
        let metal = scope.spawn(|| {
            device::sharded(
                Target::Metal,
                "cpu-model-metal",
                CpuModel::default,
                true,
                part,
                4,
                true,
            )
        });
        let join = |half: std::thread::ScopedJoinHandle<'_, Denominator>| {
            half.join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
        };
        (join(cuda), join(metal))
    });
    cuda.print("cpu-model-cuda");
    metal.print("cpu-model-metal");
    let mut expected: Vec<String> = device::part_sources(part, true)
        .into_iter()
        .map(|source| source.name)
        .collect();
    expected.sort();
    assert!(!expected.is_empty(), "corpus part {part:?} is empty");
    for d in [&cuda, &metal] {
        let mut ran = d.programs.clone();
        ran.sort();
        assert_eq!(ran, expected, "corpus part {part:?} ran other programs");
        assert_eq!(d.fallback, 0);
        // Each of the n parts executes at least ceil(floor / n), so all parts
        // together still execute at least the whole corpus's floor.
        let floor = device::sample::floor(250).div_ceil(part.1);
        assert!(d.executed_total() >= floor, "{d:?}");
        assert!(d.executed["agree_trap"] > 0 && d.executed["agree_value"] > 0);
    }
    assert_eq!(cuda.executed, metal.executed);
    assert_eq!(cuda.refused, metal.refused);
}

/// One adversarial case on one target, by name from `device::adversarial()`.
fn adversarial_case(target: Target, name: &str) {
    let label = format!("cpu-model-{target:?}");
    let mut backend = CpuModel::default;
    let case = device::adversarial()
        .into_iter()
        .find(|case| case.name == name)
        .unwrap_or_else(|| panic!("unknown adversarial case {name}"));
    let s = device::run_case(target, &label, &case, &mut backend);
    println!("general_net_abi5 {label} {}: {s:?}", case.name);
    device::check_case(case.name, &s);
}

/// Fuel cut inside rounds and across epochs of the recursive frontier on one target.
fn adversarial_fuel_prefixes(target: Target) {
    let label = format!("cpu-model-{target:?}");
    let mut backend = CpuModel::default;
    let n = device::fuel_prefixes(target, &label, &mut backend);
    assert!(n >= 10);
}

/// One test per (target, adversarial case) plus a fuel-prefix test per target,
/// so the runner can spread the cases across threads.
macro_rules! adversarial_tests {
    (cases: $($case:ident),+ $(,)?) => {
        const CASES: &[&str] = &[$(stringify!($case)),+];
        adversarial_tests!(@target cuda, Cuda, $($case),+);
        adversarial_tests!(@target metal, Metal, $($case),+);
    };
    (@target $module:ident, $target:ident, $($case:ident),+) => {
        mod $module {
            use super::super::{Target, adversarial_case, adversarial_fuel_prefixes};
            $(
                #[test]
                fn $case() {
                    adversarial_case(Target::$target, stringify!($case));
                }
            )+
            #[test]
            fn fuel_prefixes() {
                adversarial_fuel_prefixes(Target::$target);
            }
        }
    };
}

mod device_model_adversarial_cpu_model_agrees_with_oracle {
    use super::device;

    adversarial_tests!(cases:
        data_dependent_trip_counts,
        divergent_recursive_frontier,
        worklist_overflow_at_bound,
        early_trap_in_one_lane,
        device_owners_traps_and_heap_limit,
        allocation_identity_exhaustion,
    );

    /// Every adversarial case has its own test on both targets, in order.
    #[test]
    fn every_case_is_split() {
        let names: Vec<_> = device::adversarial().iter().map(|case| case.name).collect();
        assert_eq!(names, CASES);
    }
}

/// Replays of one program on one thread compile it once; another program
/// compiles again. Each run still agrees with the oracle on its own library,
/// and every compile runs under the CPU-seconds limit.
#[test]
#[cfg_attr(
    not(unix),
    ignore = "the compiler double is a POSIX shell script logging `ulimit -t`"
)]
fn device_model_cpu_model_compiles_each_program_once_per_thread() {
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("gnet03-model-cache-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (script, calls) = (dir.join("cxx"), dir.join("calls"));
    let log = calls.to_str().unwrap().replace('\'', "'\\''");
    let body = format!("#!/bin/sh\nulimit -t >> '{log}'\nexec \"${{CXX:-c++}}\" \"$@\"\n");
    executable::write(&script, body.as_bytes()).unwrap();
    let compiles = || {
        let log = std::fs::read_to_string(&calls).unwrap_or_default();
        assert!(log.lines().all(|l| l == "60"), "CPU limits {log:?}");
        log.lines().count()
    };
    let cases = device::adversarial();
    let mut backend = || CpuModel::with_compiler(&script);
    for (case, expected) in [(&cases[5], 1), (&cases[0], 2), (&cases[5], 3)] {
        let s = device::run_case(Target::Cuda, "cpu-model-cache", case, &mut backend);
        assert!(s.runs > 1, "{}", case.name);
        assert_eq!(compiles(), expected, "{}", case.name);
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

fn gnet_device(source: &str, entry: &str, backend: &str, input: &str) -> (bool, serde_json::Value) {
    let path = std::env::temp_dir().join(format!(
        "invar-gnet03-cli-{}-{}.inv",
        std::process::id(),
        source.len()
    ));
    std::fs::write(&path, source).unwrap();
    let config = r#"{"fuel":10000,"transitions":100000,"trace_events":4000,"quantum":8,"rounds":4,"capacity":2}"#;
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_invar"))
        .args([
            "gnet-device",
            path.to_str().unwrap(),
            entry,
            backend,
            input,
            config,
        ])
        .output()
        .unwrap();
    std::fs::remove_file(path).unwrap();
    (
        output.status.success(),
        serde_json::from_slice(&output.stdout).unwrap(),
    )
}

#[test]
fn device_model_receipt_records_placement_cleanup_and_typed_refusal() {
    let owned = "fn sum(n: u32) -> i32 { let xs = filled_i32(n, 3); var s = 0; for i in 0u32..n { s = s + xs[i]; } return s; } fn run(n: u32) -> i32 { let a = sum(n); let b = sum(n); return a + b; }";
    let (ok, report) = gnet_device(owned, "run", "cpu-model-metal", "[4]");
    assert!(ok, "{report}");
    assert_eq!(report["result"], 24);
    assert_eq!(report["placement"]["executed"], true);
    assert_eq!(report["placement"]["fallback"], "none");
    assert_eq!(report["execution"]["evidence_backend"], "CpuGroupModel");
    // Two normal owner drops, in source order, with their lengths.
    let cleanup = report["cleanup"].as_array().unwrap();
    assert_eq!(cleanup.len(), 2);
    assert!(cleanup.iter().all(|event| event[2] == 4));
    let (ok, report) = gnet_device(
        "struct P { x: i32 } fn run(n: i32) -> i32 { let p = P { x: n }; return p.x; }",
        "run",
        "cpu-model-cuda",
        "[1]",
    );
    assert!(ok, "{report}");
    assert_eq!(report["result"], 1);
    assert_eq!(report["placement"]["fallback"], "none");
    let (ok, report) = gnet_device(
        "fn run(a: take Array<i32>) -> i32 { return 0; }",
        "run",
        "cpu-model-cuda",
        "[1]",
    );
    assert!(!ok, "{report}");
    assert_eq!(report["placement"]["executed"], false);
    assert_eq!(report["placement"]["refusal"], "unsupported_type");
    assert_eq!(report["placement"]["fallback"], "none");
    assert!(report.get("result").is_none());
}

#[test]
fn device_model_parallel_slice_cpu_model_cuda_agrees_with_oracle() {
    parallel::denominator(Target::Cuda, "cpu-model-Cuda", CpuModel::default);
}

#[test]
fn device_model_parallel_slice_cpu_model_metal_agrees_with_oracle() {
    parallel::denominator(Target::Metal, "cpu-model-Metal", CpuModel::default);
}

/// Without `gnet-fault-injection` the shipped `invar` binary does not contain
/// the hook text, so no production emission can contain the hook; no admitted
/// corpus program's emitted CUDA/Metal source contains it either.
#[cfg(not(feature = "gnet-fault-injection"))]
#[test]
fn device_model_production_build_cannot_emit_the_fault_hook() {
    const HOOK: &[u8] = b"gnet_fault_inject";
    let binary = std::fs::read(env!("CARGO_BIN_EXE_invar")).unwrap();
    assert!(!binary.windows(HOOK.len()).any(|w| w == HOOK));
    let mut admitted = 0;
    for source in corpus() {
        for target in [Target::Cuda, Target::Metal] {
            if let Ok(plan) = Plan::from_source(&source.bytes, "main", target, Limits::default()) {
                assert!(
                    !plan
                        .artifact()
                        .source()
                        .as_bytes()
                        .windows(HOOK.len())
                        .any(|w| w == HOOK)
                );
                admitted += 1;
            }
        }
    }
    assert!(admitted >= 500, "{admitted}");
}
