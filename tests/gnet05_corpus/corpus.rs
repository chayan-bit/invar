//! GNET-05 source-derived acceptance corpus (`bench/general-net-v01/gnet05`).
//!
//! The corpus is generated ordinary Invar source with recorded seeds and
//! per-program provenance. These tests check that the checked-in files are
//! exactly what the generator produces, that the CPU net oracle agrees with
//! the reference evaluator on every runtime input (or refuses with a typed
//! reason), that held-out programs cannot reach a tuning path, and that
//! compiler decisions do not depend on names, source hashes or benchmark IDs.
//! Regenerate the files with `INVAR_GNET05_WRITE=1`.
#![forbid(unsafe_code)]

use crate::agree;
#[cfg(feature = "cuda")]
use crate::aot;
#[cfg(any(feature = "metal", feature = "cuda"))]
use crate::device;
use crate::generate;
use crate::guard;
use crate::parallel;

use generate::{HeldOut, Program, Split};
use guard::{RENAME_PREFIX, recommented, renamed, walk, without_spans};
use invar::{
    check, eval, front,
    general_net::{
        self, Net,
        compiled::Target,
        device_host::{self, Plan},
        oracle,
    },
};
use serde_json::{Value as Json, json};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

/// Whether input `i` of `p` runs in this profile: deep inputs (declared
/// budgets above the default, up to 10,000,000 steps) run in Release only.
fn runs_here(p: &Program, i: usize) -> bool {
    p.budgets[i] <= generate::DEFAULT_BUDGET || !cfg!(debug_assertions)
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("bench/general-net-v01/gnet05")
}

fn args(p: &Program, (a, b): (i64, i64)) -> Vec<eval::Value> {
    match p.param {
        "i32" => vec![eval::Value::I32(a as i32), eval::Value::I32(b as i32)],
        "u32" => vec![eval::Value::U32(a as u32), eval::Value::U32(b as u32)],
        other => panic!("{}: parameter type {other}", p.id),
    }
}

/// The generator's programs with their calibrated large inputs (the ladder
/// rung with the most reference steps within `generate::LARGE_STEPS`) and
/// their deep inputs.
fn calibrated() -> Vec<Program> {
    let large = generate::calibrate(generate::generate(), generate::LARGE_STEPS, |p, pair| {
        let checked = check::check(front::parse(p.source.as_bytes()).unwrap(), true).unwrap();
        let o =
            eval::named::observe_named(&checked, "run", &args(p, pair), generate::LARGE_STEPS + 1);
        o.result.is_ok().then_some(o.steps_used)
    });
    generate::deepen(large)
}

fn status(o: &eval::Observation) -> &'static str {
    match &o.result {
        Ok(_) => "value",
        Err(eval::EvalError::Trap { .. }) => "trap",
        Err(eval::EvalError::EvaluationLimit { .. }) => "fuel",
        Err(eval::EvalError::Tool { .. }) => "tool",
    }
}

/// The reference observation recorded as the expected baseline: the value,
/// the trap category or the tool diagnostic code.
fn outcome(o: &eval::Observation) -> String {
    match &o.result {
        Ok(value) => format!("{value:?}"),
        Err(eval::EvalError::Trap { category, .. }) => format!("trap {category:?}"),
        Err(eval::EvalError::EvaluationLimit { .. }) => "fuel".to_owned(),
        Err(eval::EvalError::Tool { code, .. }) => format!("tool {code}"),
    }
}

/// First word of a Debug rendering: the variant name of a typed error.
fn class(debug: String) -> String {
    debug
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .next()
        .unwrap_or_default()
        .to_owned()
}

#[test]
fn corpus_files_are_exactly_the_seeded_generator_output() {
    assert_eq!(
        generate::generate(),
        generate::generate(),
        "generation is deterministic"
    );
    let programs = calibrated();
    let manifest = serde_json::to_string_pretty(&generate::manifest(&programs)).unwrap() + "\n";
    if std::env::var_os("INVAR_GNET05_WRITE").is_some() {
        let dir = root().join("programs");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        for p in &programs {
            fs::write(dir.join(format!("{}.inv", p.id)), &p.source).unwrap();
        }
        fs::write(root().join("manifest.json"), &manifest).unwrap();
    }
    assert_eq!(
        fs::read_to_string(root().join("manifest.json")).unwrap(),
        manifest,
        "manifest.json is the generator's output"
    );
    let files = fs::read_dir(root().join("programs")).unwrap().count();
    assert_eq!(files, programs.len(), "no stray program files");
    for p in &programs {
        let on_disk = fs::read_to_string(root().join(format!("programs/{}.inv", p.id))).unwrap();
        assert_eq!(on_disk, p.source, "{}", p.id);
        assert_eq!(p.seed, generate::program_seed(family_index(p), p.ordinal));
        assert!(!p.inputs.is_empty() && p.inputs[0].0.abs() <= 3, "{}", p.id);
        // Every program whose work grows with its input records a large input,
        // and it is the largest rung of its ladder that fits the step budget.
        assert_eq!(p.large_input.is_some(), !p.ladder.is_empty(), "{}", p.id);
        if let Some(i) = p.large_input {
            assert_eq!(i, p.inputs.len() - 1 - usize::from(p.deep_input.is_some()));
            assert!(p.ladder.concat().contains(&p.inputs[i]), "{}", p.id);
        }
        // Budgets: one per input, the default except the deep input's, which
        // is a ladder rung with a larger budget within the corpus cap.
        assert_eq!(p.budgets.len(), p.inputs.len(), "{}", p.id);
        for (i, &budget) in p.budgets.iter().enumerate() {
            if Some(i) == p.deep_input {
                assert_eq!(i, p.inputs.len() - 1, "{}", p.id);
                assert!(p.ladder.concat().contains(&p.inputs[i]), "{}", p.id);
                assert!(
                    budget > generate::DEFAULT_BUDGET && budget <= generate::MAX_BUDGET,
                    "{}: deep budget {budget}",
                    p.id
                );
            } else {
                assert_eq!(budget, generate::DEFAULT_BUDGET, "{} input {i}", p.id);
            }
        }
    }
    let deep = programs.iter().filter(|p| p.deep_input.is_some()).count();
    assert_eq!(
        deep,
        generate::DEEP_INPUTS.len(),
        "every deep input names a program"
    );
    // Every family and split is populated, and no two programs share a
    // normalized identity (no family is a renamed copy of another program).
    let identities: std::collections::BTreeSet<_> = programs
        .iter()
        .map(|p| generate::normalized_sha256(p.source.as_bytes()))
        .collect();
    assert_eq!(identities.len(), programs.len());
    for family in generate::FAMILIES {
        let held = programs
            .iter()
            .filter(|p| p.family == family && p.split == Split::HeldOut)
            .count();
        assert_eq!(held, generate::HELD_OUT_PER_FAMILY, "{family}");
    }
}

fn family_index(p: &Program) -> usize {
    generate::FAMILIES
        .iter()
        .position(|f| *f == p.family)
        .unwrap()
}

/// Every runtime input: the CPU net oracle must agree with the reference
/// evaluator (status, value, trap, fuel, cleanup), or refuse with a typed
/// reason that is counted. Device admission refusals are counted per target.
/// The negative serial controls recurse to the 64-frame call-depth limit.
#[test]
fn corpus_oracle_agrees_with_reference_and_refusals_are_typed() {
    oracle_agrees();
}

fn oracle_agrees() {
    let programs = calibrated();
    let mut counts = BTreeMap::<String, usize>::new();
    let mut rows = Vec::new();
    let mut deep_rows = Vec::new();
    let mut bump = |key: String| *counts.entry(key).or_default() += 1;
    for p in &programs {
        let checked = check::check(front::parse(p.source.as_bytes()).unwrap(), true)
            .unwrap_or_else(|e| panic!("{}: ordinary source must check: {e:?}", p.id));
        let lowered = Net::lower(&checked, general_net::Limits::default())
            .map_err(|e| class(format!("{e:?}")))
            .and_then(|net| {
                let proof = net.verify().map_err(|e| class(format!("{e:?}")))?;
                Ok((net, proof))
            });
        let mut devices = json!({});
        for target in [Target::Cuda, Target::Metal] {
            let admission =
                match Plan::from_source(p.source.as_bytes(), "run", target, parallel::limits()) {
                    Ok(plan) => format!(
                        "admitted fork_candidates={}",
                        plan.artifact()
                            .fork_plan()
                            .map_or(0, |f| f.candidates.len())
                    ),
                    Err(e) => format!("refused {}", device_host::refusal_class(&e)),
                };
            bump(format!(
                "device {target:?} {}",
                admission.split(' ').next().unwrap()
            ));
            if let Some(reason) = admission.strip_prefix("refused ") {
                bump(format!("device {target:?} refused {reason}"));
            }
            devices[format!("{target:?}").to_lowercase()] = json!(admission);
        }
        let mut inputs = Vec::new();
        for (i, &pair) in p.inputs.iter().enumerate() {
            if !runs_here(p, i) {
                continue;
            }
            // Deep inputs are counted and recorded apart, so the Debug
            // denominators equal the Release ones.
            let deep = Some(i) == p.deep_input;
            let mut bump = |key: String| bump(if deep { format!("deep {key}") } else { key });
            let budget = p.budgets[i];
            let values = args(p, pair);
            let reference = eval::named::observe_named(&checked, "run", &values, budget);
            let negative = p.negative_inputs.contains(&i);
            let expected = status(&reference);
            if negative {
                assert!(
                    matches!(expected, "trap" | "fuel" | "tool"),
                    "{} input {i} {pair:?} must fail: {expected}",
                    p.id
                );
                bump(format!("negative {expected}"));
            } else {
                assert_eq!(expected, "value", "{} input {i} {pair:?}", p.id);
            }
            let oracle = match &lowered {
                Err(reason) => {
                    bump(format!("oracle refused {reason}"));
                    format!("net refused {reason}")
                }
                Ok((net, proof)) => {
                    let limits = if deep {
                        oracle::Limits {
                            fuel: budget,
                            transitions: u64::MAX,
                            trace_events: usize::MAX,
                            ..oracle::Limits::default()
                        }
                    } else {
                        oracle::Limits {
                            fuel: budget,
                            ..oracle::Limits::default()
                        }
                    };
                    let observed = oracle::observe_call(net, proof, "run", &values, limits)
                        .unwrap_or_else(|e| panic!("{}: oracle {e:?}", p.id));
                    match observed.failure {
                        Some(failure) => {
                            assert_eq!(observed.semantic.steps_used, 0, "{} {failure:?}", p.id);
                            bump(format!("oracle refused {failure:?}"));
                            format!("refused {failure:?}")
                        }
                        None => {
                            let context = format!("{} input {i} {pair:?}", p.id);
                            agree::same(&observed.semantic, &reference, &context);
                            bump(format!("oracle equal {expected}"));
                            "equal".to_owned()
                        }
                    }
                }
            };
            let row = json!({
                "input": [pair.0, pair.1], "reference": expected, "expected": outcome(&reference),
                "steps": reference.steps_used, "negative": negative, "oracle": oracle
            });
            if deep {
                println!("gnet05 deep {} {row}", p.id);
                deep_rows.push(json!({"id": p.id, "budget": budget, "row": row}));
            } else {
                inputs.push(row);
            }
        }
        rows.push(
            json!({"id": p.id, "family": p.family, "split": p.split.label(),
            "devices": devices, "inputs": inputs}),
        );
    }
    println!("gnet05 counts {counts:?}");
    let (deep_counts, counts): (BTreeMap<_, _>, BTreeMap<_, _>) = counts
        .into_iter()
        .partition(|(k, _)| k.starts_with("deep "));
    let total: usize = programs
        .iter()
        .map(|p| p.inputs.len() - usize::from(p.deep_input.is_some()))
        .sum();
    let equal: usize = counts
        .iter()
        .filter(|(k, _)| k.starts_with("oracle equal"))
        .map(|(_, n)| n)
        .sum();
    let refused: usize = counts
        .iter()
        .filter(|(k, _)| k.starts_with("oracle refused"))
        .map(|(_, n)| n)
        .sum();
    assert_eq!(equal + refused, total, "every input is compared or refused");
    assert!(
        equal * 2 > total,
        "the oracle compares most inputs: {counts:?}"
    );
    let release = !cfg!(debug_assertions);
    if release {
        let deep = programs.iter().filter(|p| p.deep_input.is_some()).count();
        assert_eq!(deep_rows.len(), deep);
        assert_eq!(
            deep_counts.get("deep oracle equal value"),
            Some(&deep),
            "every deep input is a value the oracle reproduces: {deep_counts:?}"
        );
    }
    let report = json!({"schema": "invar-gnet05-validation-v2",
        "budgets": {"default": generate::DEFAULT_BUDGET, "max": generate::MAX_BUDGET},
        "programs": programs.len(), "inputs": total,
        "counts": counts, "rows": rows, "deep_counts": deep_counts, "deep": deep_rows});
    let text = serde_json::to_string_pretty(&report).unwrap() + "\n";
    let path = root().join("validation.json");
    if std::env::var_os("INVAR_GNET05_WRITE").is_some() && release {
        fs::write(&path, &text).unwrap();
    }
    let recorded: Json = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    assert_eq!(
        recorded["counts"], report["counts"],
        "recorded denominators"
    );
    assert_eq!(
        recorded["rows"], report["rows"],
        "recorded per-input outcomes"
    );
    if release {
        assert_eq!(recorded["deep"], report["deep"], "recorded deep outcomes");
        assert_eq!(recorded["deep_counts"], report["deep_counts"]);
    }
}

/// Whether a self-recursive function multiplies two of its child results or
/// branches on them: the combines no linear law (path-sum) covers.
fn nonlinear_recursion(source: &str) -> bool {
    source.split("\nfn ").any(|function| {
        let name = function
            .trim_start_matches("fn ")
            .split('(')
            .next()
            .unwrap_or_default();
        let call = format!("{name}(");
        let children: Vec<&str> = function
            .lines()
            .filter_map(|l| l.trim().strip_prefix("let "))
            .filter_map(|l| l.split_once(" = "))
            .filter(|(_, rhs)| rhs.starts_with(&call))
            .map(|(child, _)| child)
            .collect();
        let is_child = |w: &str| children.contains(&w);
        function.lines().any(|line| {
            let words: Vec<&str> = line
                .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '*')
                .filter(|w| !w.is_empty())
                .collect();
            let product = words
                .windows(3)
                .any(|w| is_child(w[0]) && w[1] == "*" && is_child(w[2]));
            let condition = line.trim().starts_with("if ")
                && line
                    .split('{')
                    .next()
                    .unwrap()
                    .split(|c: char| !c.is_alphanumeric() && c != '_')
                    .any(is_child);
            product || condition
        })
    })
}

/// The forkjoin family is the residual fork-join tier: every program keeps a
/// combine that is not linear in its child results. Every other recursive
/// family combines children linearly modulo a prime (the bias the normal-form design measured).
#[test]
fn corpus_forkjoin_combines_are_not_linear() {
    for p in generate::generate() {
        assert_eq!(
            nonlinear_recursion(&p.source),
            p.family == "forkjoin",
            "{}",
            p.id
        );
    }
}

/// Held-out programs never reach a tuning path, including after renaming,
/// reformatting or recommenting, and the held-out set is pinned.
#[test]
fn corpus_held_out_programs_never_reach_tuning() {
    let programs = generate::generate();
    let held = HeldOut::of(&programs);
    assert_eq!(
        held.len(),
        generate::FAMILIES.len() * generate::HELD_OUT_PER_FAMILY
    );
    let manifest: Json =
        serde_json::from_slice(&fs::read(root().join("manifest.json")).unwrap()).unwrap();
    assert_eq!(
        manifest["held_out"]["normalized_identity_sha256"],
        generate::hex(&held.digest()),
        "pinned held-out set"
    );
    let tuning = generate::tuning_programs(&programs, &held);
    assert_eq!(tuning.len(), programs.len() - held.len());
    for p in &programs {
        let admitted = held.admit_for_tuning(p.source.as_bytes()).is_ok();
        assert_eq!(admitted, p.split == Split::Tuning, "{}", p.id);
        assert_eq!(tuning.iter().any(|t| t.id == p.id), admitted, "{}", p.id);
        if p.split == Split::HeldOut {
            for disguised in [renamed(&p.source), recommented(&p.source)] {
                assert!(
                    held.admit_for_tuning(disguised.as_bytes()).is_err(),
                    "{}: a disguised held-out program reached tuning",
                    p.id
                );
            }
        }
    }
}

/// Compiler decisions are a function of program structure only: renaming every
/// identifier, changing the source hash with comments and whitespace, and the
/// benchmark IDs (which never reach the compiler) leave admission, the fork
/// plan and the emitted device code unchanged. Emitted code carries source
/// function names as symbols and trap spans as source byte offsets, so it is
/// compared with the rename prefix and the span numbers removed.
#[test]
fn corpus_compiler_decisions_ignore_names_hashes_and_ids() {
    for p in generate::generate() {
        for target in [Target::Cuda, Target::Metal] {
            let decide = |source: &str| match Plan::from_source(
                source.as_bytes(),
                "run",
                target,
                parallel::limits(),
            ) {
                Ok(plan) => {
                    let fork = plan.artifact().fork_plan().map(|f| f.candidates.len());
                    let emitted = plan.artifact().source();
                    if !source.contains(RENAME_PREFIX) {
                        assert!(!emitted.contains(RENAME_PREFIX), "{}", p.id);
                    }
                    let code = without_spans(&emitted.replace(RENAME_PREFIX, ""));
                    let code = generate::normalized_sha256(code.as_bytes());
                    format!("admitted fork={fork:?} code={}", generate::hex(&code))
                }
                Err(e) => format!("refused {}", device_host::refusal_class(&e)),
            };
            let base = decide(&p.source);
            assert_eq!(
                decide(&renamed(&p.source)),
                base,
                "{} {target:?} renamed",
                p.id
            );
            assert_eq!(
                decide(&recommented(&p.source)),
                base,
                "{} {target:?} recommented",
                p.id
            );
        }
    }
}

/// No compiler, MIR, net or device source mentions the corpus, its paths,
/// program IDs or fixture names: decisions cannot key on them.
#[test]
fn corpus_compiler_sources_never_name_the_corpus() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    for dir in ["src/general_net", "src/front", "src/check", "src/mir"] {
        walk(&repo.join(dir), &mut files);
    }
    for file in [
        "src/front.rs",
        "src/check.rs",
        "src/mir.rs",
        "src/mir_net.rs",
        "src/gnet_device_command.rs",
    ] {
        files.push(repo.join(file));
    }
    let mut forbidden: Vec<String> = [
        "gnet05",
        "general-net-v01",
        "gnet_parallel",
        "fixtures/",
        "bench/",
        "held_out",
        "heldout",
        "manifest.json",
    ]
    .map(String::from)
    .to_vec();
    forbidden.extend(generate::generate().into_iter().map(|p| p.id));
    // The robotics corpus is held to the same rule.
    forbidden.extend(["gnet05r", "robotics/programs", "gnet05-robotics"].map(String::from));
    forbidden.extend(
        crate::robotics::corpus::generate(repo)
            .into_iter()
            .map(|p| p.id),
    );
    assert!(files.len() > 20, "scan scope");
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

/// Device execution of every admitted program and input, exact against the
/// oracle, wide and as a capacity-0 serial control. Returns
/// (programs executed, inputs executed, traps, spawned, committed, peak width).
/// Programs are independent: `INVAR_GNET05_DEVICE_JOBS` (default 1) runs that
/// many at once, each with its own backends; the totals do not depend on it.
#[cfg(any(feature = "metal", feature = "cuda"))]
fn device_corpus<B: device_host::Backend>(
    target: Target,
    label: &str,
    backend: impl Fn() -> B + Sync,
) -> [u64; 6] {
    let start = std::time::Instant::now();
    let programs = calibrated();
    let jobs = device_jobs();
    let runs = device::bounded_pool(&programs, jobs, |p| {
        let plan =
            Plan::from_source(p.source.as_bytes(), "run", target, parallel::limits()).ok()?;
        println!("gnet05 {label} {} at {:?}", p.id, start.elapsed());
        Some(device_program(target, label, &backend, p, &plan))
    });
    let mut totals = [0u64; 6];
    let mut deep = 0;
    for (program, program_deep) in runs.into_iter().flatten() {
        for i in 0..5 {
            totals[i] += program[i];
        }
        totals[5] = totals[5].max(program[5]);
        deep += program_deep;
    }
    println!("gnet05 {label} deep inputs executed={deep} jobs={jobs}");
    println!(
        "gnet05 {label} programs={} inputs={} traps={} spawned={} committed={} peak_width={}",
        totals[0], totals[1], totals[2], totals[3], totals[4], totals[5]
    );
    assert!(
        totals[1] > 0 && totals[3] > 0,
        "{label}: the corpus executed frontiers"
    );
    totals
}

/// Concurrent device programs: `INVAR_GNET05_DEVICE_JOBS`, default 1.
#[cfg(any(feature = "metal", feature = "cuda"))]
pub(crate) fn device_jobs() -> usize {
    std::env::var("INVAR_GNET05_DEVICE_JOBS")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(1)
}

/// Every input of one admitted program: its `device_corpus` totals and the
/// number of deep inputs it ran.
#[cfg(any(feature = "metal", feature = "cuda"))]
fn device_program<B: device_host::Backend>(
    target: Target,
    label: &str,
    backend: &impl Fn() -> B,
    p: &Program,
    plan: &Plan,
) -> ([u64; 6], u64) {
    let mut backend = || backend();
    let mut totals = [1, 0, 0, 0, 0, 0];
    let mut deep = 0;
    for (i, &pair) in p.inputs.iter().enumerate() {
        let values = args(p, pair);
        let context = format!("{label} {} {pair:?}", p.id);
        if Some(i) == p.deep_input {
            // Deep inputs run on the device on request, only within the
            // device configuration's 1,000,000-step fuel cap, and with a
            // larger admission buffer budget for their trace; they are
            // counted apart.
            if std::env::var_os("INVAR_GNET05_DEEP_DEVICE").is_some() && p.budgets[i] <= DEVICE_CAP
            {
                let limits = device_host::Limits {
                    buffer_bytes: DEEP_BUFFER_BYTES,
                    ..parallel::limits()
                };
                let plan = Plan::from_source(p.source.as_bytes(), "run", target, limits)
                    .unwrap_or_else(|e| panic!("{context}: deep plan {e:?}"));
                let budget = p.budgets[i];
                exact_with_budget(
                    &context,
                    &plan,
                    &mut backend,
                    &values,
                    parallel::SLOTS,
                    budget,
                );
                let (serial, _, _) =
                    exact_with_budget(&context, &plan, &mut backend, &values, 0, budget);
                assert_eq!(serial.spawned, 0, "{context}");
                deep += 1;
            }
            continue;
        }
        let (f, trapped, _) =
            parallel::exact(&context, plan, &mut backend, &values, parallel::SLOTS);
        let (serial, _, _) =
            parallel::exact(&format!("{context} serial"), plan, &mut backend, &values, 0);
        assert_eq!(serial.spawned, 0, "{context}");
        totals[1] += 1;
        totals[2] += u64::from(trapped);
        totals[3] += f.spawned;
        totals[4] += f.committed;
        totals[5] = totals[5].max(f.peak_width);
    }
    (totals, deep)
}

/// Device configuration cap on fuel, transitions and trace events
/// (`compiled::DeviceLayout::buffer_bytes`).
#[cfg(any(feature = "metal", feature = "cuda"))]
const DEVICE_CAP: u64 = 1_000_000;

/// Admission buffer budget of a deep input's plan: its device trace alone
/// can exceed the 64 MiB default (admission only; the emitted code is the
/// same).
#[cfg(any(feature = "metal", feature = "cuda"))]
const DEEP_BUFFER_BYTES: usize = 1 << 30;

/// `parallel::exact` with the input's declared step budget as device fuel
/// (deep inputs; the device code is unchanged).
#[cfg(any(feature = "metal", feature = "cuda"))]
fn exact_with_budget<B: device_host::Backend>(
    context: &str,
    plan: &Plan,
    backend: &mut impl FnMut() -> B,
    args: &[eval::Value],
    capacity: usize,
    budget: u64,
) -> (device::Frontier, bool, u64) {
    let started = std::time::Instant::now();
    let cfg = |events| invar::general_net::compiled::DeviceConfig {
        transitions: DEVICE_CAP,
        ..device::config(budget, events, 16, 64, capacity)
    };
    let sized = device::reference(plan, args, cfg(DEVICE_CAP as usize));
    let cfg = cfg(sized.events.len().max(1));
    let expected = device::reference(plan, args, cfg);
    let (got, epochs) = device::drive(plan, &mut backend(), args, cfg)
        .unwrap_or_else(|e| panic!("{context}: {e:?}"));
    device::agree(context, plan, &got, &expected);
    println!(
        "gnet05 deep device {context} capacity={capacity} epochs={epochs} in {:?}",
        started.elapsed()
    );
    (device::Frontier::of(&got), got.result.is_none(), epochs)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "requires an Apple Metal device"]
fn corpus_executes_on_metal_exactly() {
    use invar::general_net::device_host::metal::Metal;
    device_corpus(Target::Metal, "metal", || {
        Metal::open_runtime_source().expect("Metal device")
    });
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device and NVRTC"]
fn corpus_executes_on_cuda_exactly() {
    device_corpus(Target::Cuda, "cuda", aot::open);
}
