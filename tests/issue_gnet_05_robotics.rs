//! GNET-05 robotics corpus (`bench/general-net-v01/gnet05/robotics`).
//!
//! Ordinary-source FK, Jacobian, mass, collision, rollout, point-cloud and
//! pipeline programs over the pinned Panda and UR5e MJCF models. The tests
//! check that the checked-in files are the seeded generator output; that the
//! CPU net oracle reproduces the reference evaluator's full output bit for
//! bit; that every output agrees with independent binary64 model math (and
//! the probes with the committed Pinocchio references) within tolerances
//! frozen below before any scoring; that held-out programs cannot reach a
//! tuning path; and that compiler decisions ignore names, hashes and IDs.
//! Regenerate with `INVAR_GNET05_WRITE=1` (validation digests need Release).
#![forbid(unsafe_code)]

#[path = "support/gnet_agree.rs"]
mod agree;
#[path = "support/gnet_device.rs"]
mod device;
#[path = "../bench/general-net-v01/gnet05/generate.rs"]
mod generate;
#[path = "support/gnet_corpus_guard.rs"]
mod guard;
#[path = "support/gnet_par.rs"]
mod par;
#[path = "support/gnet_parallel.rs"]
mod parallel;
#[path = "../bench/general-net-v01/gnet05/robotics/mod.rs"]
mod robotics;
#[path = "support/gnet_robotics_checks.rs"]
mod robotics_checks;

use generate::Split;
use invar::{
    check, eval, front,
    general_net::{
        self, Net, compiled,
        compiled::Target,
        device_host::{self, Plan},
        oracle, provenance,
    },
};
use robotics::corpus::{self, Program};
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// Frozen before scoring. Absolute tolerances (metres, radians, unit-free
/// rotation entries, kg m^2) for binary32 results against binary64 math.
pub const TOL_FRAME: f64 = 2e-5;
pub const TOL_JACOBIAN: f64 = 2e-5;
pub const TOL_MASS: f64 = 1e-4;
pub const TOL_DISTANCE: f64 = 5e-5;
pub const TOL_COST: f64 = 1e-4;
/// Binary64 model math against the committed Pinocchio 4.1.0 references.
pub const TOL_MODEL: f64 = 1e-9;
/// Reference evaluator fuel (never reached by any recorded input).
const FUEL: u64 = 1 << 40;

const TABLES: &str = "bench/baselines/robotics-tables.h";

fn repo() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}
fn root() -> PathBuf {
    repo().join("bench/general-net-v01/gnet05/robotics")
}
fn write() -> bool {
    std::env::var_os("INVAR_GNET05_WRITE").is_some()
}

#[test]
fn issue_gnet_05_robotics_files_are_exactly_the_seeded_generator_output() {
    let programs = corpus::generate(repo());
    assert_eq!(programs, corpus::generate(repo()), "deterministic");
    let manifest =
        serde_json::to_string_pretty(&corpus::manifest(repo(), &programs)).unwrap() + "\n";
    if write() {
        let dir = root().join("programs");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        for p in &programs {
            fs::write(dir.join(format!("{}.inv", p.id)), &p.source).unwrap();
        }
        fs::write(root().join("manifest.json"), &manifest).unwrap();
        fs::write(repo().join(TABLES), corpus::baseline_tables(repo())).unwrap();
    }
    assert_eq!(
        fs::read_to_string(repo().join(TABLES)).unwrap(),
        corpus::baseline_tables(repo()),
        "baseline tables are the generator's output"
    );
    assert_eq!(
        fs::read_to_string(root().join("manifest.json")).unwrap(),
        manifest
    );
    assert_eq!(
        fs::read_dir(root().join("programs")).unwrap().count(),
        programs.len()
    );
    for p in &programs {
        let on_disk = fs::read_to_string(root().join(format!("programs/{}.inv", p.id))).unwrap();
        assert_eq!(on_disk, p.source, "{}", p.id);
    }
    let identities: std::collections::BTreeSet<_> = programs
        .iter()
        .map(|p| generate::normalized_sha256(p.source.as_bytes()))
        .collect();
    assert_eq!(identities.len(), programs.len(), "no renamed duplicates");
    for family in robotics::families::FAMILIES {
        let held = programs
            .iter()
            .filter(|p| p.case.family == family && p.split == Split::HeldOut)
            .count();
        assert_eq!(held, corpus::HELD_OUT_PER_FAMILY, "{family}");
    }
}

fn checked(p: &Program) -> check::CheckedProgram {
    check::check(front::parse(p.source.as_bytes()).unwrap(), true)
        .unwrap_or_else(|e| panic!("{}: ordinary source must check: {e:?}", p.id))
}

fn values(input: &[u32]) -> Vec<eval::Value> {
    input.iter().map(|&x| eval::Value::U32(x)).collect()
}

/// Little-endian binary32 words of a successful array result.
pub fn words(o: &eval::Observation, context: &str) -> Vec<f32> {
    match &o.result {
        Ok(eval::Value::ArrayF32(v)) => v
            .borrow()
            .iter()
            .map(|x| f32::from_bits(x.to_bits()))
            .collect(),
        other => panic!("{context}: expected an f32 array, got {other:?}"),
    }
}
pub fn digest(out: &[f32]) -> String {
    let mut h = Sha256::new();
    for x in out {
        h.update(x.to_bits().to_le_bytes());
    }
    generate::hex(&h.finalize())
}

/// Full-output validation. Oracle inputs: the CPU net oracle equals the
/// reference evaluator bit for bit (value, steps, cleanup) and the output
/// passes the independent binary64 check. Scored inputs (Release only): the
/// reference output digest is frozen in `validation.json`.
#[test]
fn issue_gnet_05_robotics_oracle_and_independent_math_agree_on_full_outputs() {
    let mut programs = corpus::generate(repo());
    // Development filter (never with INVAR_GNET05_WRITE): program ID substring.
    if let Ok(only) = std::env::var("INVAR_GNET05R_ONLY") {
        assert!(!write(), "a filtered run cannot rewrite validation.json");
        programs.retain(|p| p.id.contains(&only));
    }
    let models = robotics_checks::Models::load(repo());
    let release = !cfg!(debug_assertions);
    let rows = par::map(&programs, |p| {
        let c = checked(p);
        let net = Net::lower(&c, general_net::Limits::default())
            .unwrap_or_else(|e| panic!("{}: net lowering {e:?}", p.id));
        let proof = net
            .verify()
            .unwrap_or_else(|e| panic!("{}: net proof {e:?}", p.id));
        let mut oracle_rows = Vec::new();
        for input in &p.oracle_inputs {
            let context = format!("{} {input:?}", p.id);
            let reference =
                eval::named::observe_owned_numeric_named(&c, "run", &values(input), FUEL);
            let out = words(&reference, &context);
            let graph = oracle::observe_call(
                &net,
                &proof,
                "run",
                &values(input),
                oracle::Limits {
                    fuel: FUEL,
                    transitions: u64::MAX,
                    trace_events: usize::MAX,
                    array_elements: 1 << 22,
                },
            )
            .unwrap_or_else(|e| panic!("{context}: oracle {e:?}"));
            assert_eq!(graph.failure, None, "{context}");
            agree::same(&graph.semantic, &reference, &context);
            let error = models.check(p, input, &out, &context);
            oracle_rows.push(json!({"input": input, "steps": reference.steps_used,
                "words": out.len(), "sha256": digest(&out), "oracle": "equal",
                "independent_max_abs_error": error}));
        }
        let mut scored_rows = Vec::new();
        if release {
            for input in &p.scored_inputs {
                let context = format!("{} {input:?}", p.id);
                let start = std::time::Instant::now();
                let reference =
                    eval::named::observe_owned_numeric_named(&c, "run", &values(input), FUEL);
                let out = words(&reference, &context);
                println!(
                    "gnet05r {context} steps={} in {:?}",
                    reference.steps_used,
                    start.elapsed()
                );
                scored_rows.push(json!({"input": input, "steps": reference.steps_used,
                    "words": out.len(), "sha256": digest(&out)}));
            }
        }
        json!({"id": p.id, "split": p.split.label(), "oracle_inputs": oracle_rows,
            "scored_inputs": scored_rows})
    });
    let report = json!({"schema": "invar-gnet05-robotics-validation-v1",
        "tolerances": {"frame": TOL_FRAME, "jacobian": TOL_JACOBIAN, "mass": TOL_MASS,
            "distance": TOL_DISTANCE, "cost": TOL_COST, "model_vs_pinocchio": TOL_MODEL},
        "programs": rows});
    let path = root().join("validation.json");
    assert_eq!(report["programs"].as_array().unwrap().len(), programs.len());
    if write() && release {
        fs::write(&path, serde_json::to_string_pretty(&report).unwrap() + "\n").unwrap();
    }
    let recorded: Json = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    assert_eq!(recorded["tolerances"], report["tolerances"]);
    for n in report["programs"].as_array().unwrap() {
        let r = recorded["programs"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["id"] == n["id"])
            .unwrap_or_else(|| panic!("{} is recorded", n["id"]));
        for (a, b) in r["oracle_inputs"]
            .as_array()
            .unwrap()
            .iter()
            .zip(n["oracle_inputs"].as_array().unwrap())
        {
            for key in ["input", "steps", "words", "sha256"] {
                assert_eq!(a[key], b[key], "{} {key}", n["id"]);
            }
        }
        if release {
            assert_eq!(r["scored_inputs"], n["scored_inputs"], "{}", n["id"]);
        }
    }
}

/// The three structural variants of a case perform the same operations in
/// the same order: on a shared input their full outputs are bit-identical
/// (one handwritten baseline per case therefore covers every variant).
#[test]
fn issue_gnet_05_robotics_variants_compute_identical_bits() {
    let programs = corpus::generate(repo());
    let groups = par::map(&programs, |p| {
        let first = programs
            .iter()
            .find(|q| {
                q.case.family == p.case.family
                    && q.case.arm == p.case.arm
                    && q.case.trig == p.case.trig
            })
            .unwrap();
        let input = &first.oracle_inputs[1];
        let o = eval::named::observe_owned_numeric_named(&checked(p), "run", &values(input), FUEL);
        (first.id.clone(), digest(&words(&o, &p.id)))
    });
    let mut by_case = std::collections::BTreeMap::<String, Vec<String>>::new();
    for (case, d) in groups {
        by_case.entry(case).or_default().push(d);
    }
    assert_eq!(by_case.len(), 16);
    for (case, digests) in &by_case {
        assert_eq!(digests.len(), 3, "{case}");
        assert!(
            digests.iter().all(|d| d == &digests[0]),
            "{case}: {digests:?}"
        );
    }
}

/// Probes run the program's own chain at the committed reference angles.
#[test]
fn issue_gnet_05_robotics_probes_and_model_match_pinocchio() {
    probes();
}

fn probes() {
    let models = robotics_checks::Models::load(repo());
    models.model_matches_pinocchio(TOL_MODEL);
    let programs = corpus::generate(repo());
    let mut probed = 0;
    for p in &programs {
        if !matches!(p.case.family, "fk" | "jacobian" | "mass") {
            continue;
        }
        let c = checked(p);
        probed += models.probe(p, &|args: &[f32]| {
            let values: Vec<eval::Value> = args
                .iter()
                .map(|&x| {
                    eval::Value::F32(invar::numeric_contract::F32Bits::from_bits(x.to_bits()))
                })
                .collect();
            let o = eval::named::observe_owned_numeric_named(&c, "probe", &values, FUEL);
            words(&o, &p.id)
        });
    }
    assert_eq!(
        probed,
        18 * 8,
        "every probe program at every reference case"
    );
}

#[test]
fn issue_gnet_05_robotics_held_out_programs_never_reach_tuning() {
    let programs = corpus::generate(repo());
    let held = corpus::held_out(&programs);
    assert_eq!(
        held.len(),
        robotics::families::FAMILIES.len() * corpus::HELD_OUT_PER_FAMILY
    );
    let manifest: Json =
        serde_json::from_slice(&fs::read(root().join("manifest.json")).unwrap()).unwrap();
    assert_eq!(
        manifest["held_out"]["normalized_identity_sha256"],
        generate::hex(&held.digest())
    );
    let tuning = corpus::tuning_programs(&programs, &held);
    assert_eq!(tuning.len(), programs.len() - held.len());
    // The dynamic corpus guard and this one are disjoint sets.
    let dynamic = generate::HeldOut::of(&generate::generate());
    for p in &programs {
        let admitted = held.admit_for_tuning(p.source.as_bytes()).is_ok();
        assert_eq!(admitted, p.split == Split::Tuning, "{}", p.id);
        assert!(dynamic.admit_for_tuning(p.source.as_bytes()).is_ok());
        if p.split == Split::HeldOut {
            for disguised in [guard::renamed(&p.source), guard::recommented(&p.source)] {
                assert!(
                    held.admit_for_tuning(disguised.as_bytes()).is_err(),
                    "{}: a disguised held-out program reached tuning",
                    p.id
                );
            }
        }
    }
}

/// Device admission and the CPU numeric artifact are functions of program
/// structure only: renaming identifiers and changing comments, whitespace
/// and the source hash leave both unchanged (names and spans removed).
#[test]
fn issue_gnet_05_robotics_compiler_decisions_ignore_names_hashes_and_ids() {
    let programs = corpus::generate(repo());
    let rows = par::map(&programs, |p| {
        let decide = |source: &str| {
            let mut out = Vec::new();
            for target in [Target::Cuda, Target::Metal] {
                out.push(
                    match Plan::from_source(source.as_bytes(), "run", target, parallel::limits()) {
                        Ok(_) => "admitted".to_owned(),
                        Err(e) => format!("refused {}", device_host::refusal_class(&e)),
                    },
                );
            }
            out.push(
                match provenance::compile_owned_numeric(
                    source.as_bytes(),
                    "run",
                    Target::Cpu,
                    // Wide compiler limits so the table variants compile and the
                    // comparison covers emitted code; `elements` stays within the
                    // owned-f32 cap (2,000,000), above which every program is refused.
                    provenance::Limits {
                        net: Default::default(),
                        compiler: compiled::Limits {
                            values: 1 << 16,
                            places: 4096,
                            states: 1 << 20,
                            frame_bytes: 1 << 26,
                            source_bytes: 1 << 26,
                        },
                    },
                    compiled::OwnedLimits {
                        elements: 2_000_000,
                        allocations: 64,
                        cleanup_events: 64,
                    },
                ) {
                    Ok(a) => {
                        let code = robotics_checks::without_names_and_spans(a.artifact().source());
                        format!("cpu {}", generate::hex(&Sha256::digest(code.as_bytes())))
                    }
                    Err(e) => format!("cpu refused {}", robotics_checks::class(&e)),
                },
            );
            out
        };
        let base = decide(&p.source);
        assert_eq!(decide(&guard::renamed(&p.source)), base, "{} renamed", p.id);
        assert_eq!(
            decide(&guard::recommented(&p.source)),
            base,
            "{} recommented",
            p.id
        );
        base
    });
    for (p, row) in programs.iter().zip(&rows) {
        println!("gnet05r decisions {} {row:?}", p.id);
    }
    let compiled = rows
        .iter()
        .filter(|r| !r[2].starts_with("cpu refused"))
        .count();
    println!(
        "gnet05r decisions cpu_compiled={compiled} of {}",
        rows.len()
    );
    assert!(compiled > 0, "the CPU comparison must cover emitted code");
}

#[test]
fn issue_gnet_05_robotics_compiler_sources_never_name_the_corpus() {
    let mut files = Vec::new();
    for dir in ["src/general_net", "src/front", "src/check", "src/mir"] {
        guard::walk(&repo().join(dir), &mut files);
    }
    assert!(files.len() > 20);
    let mut forbidden: Vec<String> = ["gnet05r", "robotics/programs", "gnet05-robotics"]
        .map(String::from)
        .to_vec();
    forbidden.extend(corpus::generate(repo()).into_iter().map(|p| p.id));
    for file in &files {
        let text = fs::read_to_string(file).unwrap();
        for word in &forbidden {
            assert!(
                !text.contains(word.as_str()),
                "{} mentions {word}",
                file.display()
            );
        }
    }
}

/// Compare every scored baseline output file with its frozen digest. A missing
/// file fails: a partial or interrupted run must never read as a pass.
fn check_baseline_outputs(dir: &Path, recorded: &Json) -> Result<usize, String> {
    let mut compared = 0;
    let mut missing = Vec::new();
    for row in recorded["programs"].as_array().unwrap() {
        for s in row["scored_inputs"].as_array().unwrap() {
            let input: Vec<String> = s["input"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.to_string())
                .collect();
            let file = dir.join(format!(
                "{}-{}.bin",
                row["id"].as_str().unwrap(),
                input.join("-")
            ));
            let Ok(bytes) = fs::read(&file) else {
                missing.push(file.display().to_string());
                continue;
            };
            if generate::hex(&Sha256::digest(&bytes)) != s["sha256"].as_str().unwrap() {
                return Err(format!(
                    "{}: digest differs from the frozen one",
                    file.display()
                ));
            }
            compared += 1;
        }
    }
    if let Some(first) = missing.first() {
        return Err(format!(
            "{} of {} scored baseline outputs are missing, first {first}",
            missing.len(),
            missing.len() + compared
        ));
    }
    Ok(compared)
}

/// Handwritten baseline (comparison D) output files must equal the frozen
/// reference digest bit for bit. Point `INVAR_ROBOTICS_BASELINE_OUT` at the
/// directory the runner wrote (`<program>-<input>.bin` per scored input).
#[test]
#[ignore = "consumes handwritten baseline outputs produced on a GPU host"]
fn issue_gnet_05_robotics_baseline_outputs_match_reference() {
    let dir = PathBuf::from(std::env::var("INVAR_ROBOTICS_BASELINE_OUT").expect("output dir"));
    let recorded: Json =
        serde_json::from_slice(&fs::read(root().join("validation.json")).unwrap()).unwrap();
    let compared = check_baseline_outputs(&dir, &recorded).unwrap_or_else(|e| panic!("{e}"));
    println!("gnet05r baseline outputs compared={compared}");
    assert!(compared > 0);
}

#[test]
fn issue_gnet_05_robotics_baseline_check_refuses_an_incomplete_output_directory() {
    let dir = std::env::temp_dir().join(format!("gnet05r-baseline-check-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let digest = |b: &[u8]| generate::hex(&Sha256::digest(b));
    let recorded = json!({"programs": [{
        "id": "p",
        "scored_inputs": [
            {"input": [1, 2], "sha256": digest(b"aa")},
            {"input": [3], "sha256": digest(b"bb")},
        ],
    }]});
    assert!(
        check_baseline_outputs(&dir, &recorded)
            .unwrap_err()
            .contains("2 of 2")
    );
    fs::write(dir.join("p-1-2.bin"), b"aa").unwrap();
    let partial = check_baseline_outputs(&dir, &recorded).unwrap_err();
    assert!(
        partial.contains("1 of 2") && partial.contains("p-3.bin"),
        "{partial}"
    );
    fs::write(dir.join("p-3.bin"), b"xx").unwrap();
    assert!(
        check_baseline_outputs(&dir, &recorded)
            .unwrap_err()
            .contains("digest differs")
    );
    fs::write(dir.join("p-3.bin"), b"bb").unwrap();
    assert_eq!(check_baseline_outputs(&dir, &recorded), Ok(2));
    fs::remove_dir_all(&dir).unwrap();
}
