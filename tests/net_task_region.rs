#![forbid(unsafe_code)]
//! P4b soundness: the residual fork-join task region T against the CPU net
//! oracle (observation contract v2: value or first source-order trap; the
//! 64-frame depth limit as the oracle's `CALL_DEPTH_LIMIT` tool failure).
//!
//! The CPU model compiles the emitted scheduler, run kernel and state machines
//! as C++ under ASan and UBSan, with small widths so expansion, nested forks,
//! joins, continuation calls and arena-full fallbacks all run.
//!
//! Corpus: the C1 fork-join family of the GNET-05 robotics corpus (`tests/net_task_fixtures/c1`,
//! vendored from `bench/general-net-v01/gnet05/programs/gnet05-forkjoin-*.inv`
//! at `origin/feat/v01-gnet-05-robotics` 005e145f, with the GNET-05 v2
//! manifest inputs below), plus generated trap-heavy fork-join programs.

#[path = "support/task_model.rs"]
mod support;

use invar::general_net::task::{self, Mutation, Refusal, Schedule};
use support::{
    Rng, Tally, admit, c1, c1_cases, c1_source, cpu_plans, differential, differential_against,
    expected, generated, generated_cases,
};

fn report(label: &str, tally: &Tally) {
    eprintln!(
        "{label}: {} cases ({} values, {} traps, {} depth limits, {} fallbacks), {} skipped (oracle limit), {} skipped (lane budget)",
        tally.cases,
        tally.values,
        tally.traps,
        tally.depth_limits,
        tally.fallbacks,
        tally.skipped,
        tally.budget
    );
}

#[test]
fn c1_fork_join_family_agrees_with_the_net_oracle() {
    support::with_stack(|| {
        let mut rng = Rng(0x0c1_f0e5);
        let mut total = Tally::default();
        for index in 0..12 {
            let admitted = c1(index);
            let cases = c1_cases(index, &mut rng, 120);
            let oracle = expected(&admitted, &cases);
            for (label, plan) in cpu_plans(&admitted.program) {
                let name = format!("c1-{index:02}-{label}");
                let tally =
                    differential_against(&name, &admitted, &plan, Mutation::None, &cases, &oracle);
                report(&name, &tally);
                total.add(tally);
            }
        }
        report("C1 total", &total);
        assert!(total.mismatches.is_empty(), "{:#?}", total.mismatches);
        assert!(total.cases >= 5_600 && total.traps > 0 && total.fallbacks > 0);
        assert!(
            total.budget * 50 <= total.cases,
            "lane budget skips {}",
            total.budget
        );
    });
}

/// The cascade-liveness device tests' program stays a residual task region, and its CPU
/// model agrees with the net oracle on every plan (the device tests add a
/// 64-poll cascade limit on real hardware).
#[test]
fn long_serial_leaves_are_a_residual_region_that_agrees_with_the_net_oracle() {
    support::with_stack(|| {
        let admitted =
            admit(support::LONG_LEAVES).expect("LONG_LEAVES must stay a residual region");
        let cases = vec![vec![1, 5, 200_000], vec![1, 77, 200_000]];
        let oracle = expected(&admitted, &cases);
        // The reference the device tests use (past the oracle's limits) is the
        // oracle's value here.
        for (case, observed) in cases.iter().zip(&oracle) {
            let want = support::long_leaves_expected(case[0], case[1], case[2]);
            assert!(
                matches!(observed, Some(Ok(invar::eval::Value::I32(v))) if *v == want),
                "{case:?}: oracle {observed:?}, reference {want}"
            );
        }
        for (label, plan) in cpu_plans(&admitted.program) {
            let tally = differential_against(
                &format!("long-leaves-{label}"),
                &admitted,
                &plan,
                Mutation::None,
                &cases,
                &oracle,
            );
            assert!(
                tally.mismatches.is_empty(),
                "{label}: {:#?}",
                tally.mismatches
            );
            assert_eq!(tally.cases, 2, "{label}");
        }
    });
}

#[test]
fn generated_trap_heavy_fork_joins_agree_with_the_net_oracle() {
    support::with_stack(|| {
        let mut rng = Rng(0x7a5c_f04c);
        let mut total = Tally::default();
        let mut programs = 0;
        while programs < 24 {
            let source = generated(&mut rng);
            let admitted = match admit(&source) {
                Ok(admitted) => admitted,
                Err(refusal) => panic!("generated program refused ({refusal:?}):\n{source}"),
            };
            let cases = generated_cases(&mut rng, 160);
            let oracle = expected(&admitted, &cases);
            for (label, plan) in cpu_plans(&admitted.program) {
                let name = format!("gen-{programs:02}-{label}");
                let tally =
                    differential_against(&name, &admitted, &plan, Mutation::None, &cases, &oracle);
                if !tally.mismatches.is_empty() {
                    eprintln!("{source}");
                }
                total.add(tally);
            }
            programs += 1;
        }
        report("generated total", &total);
        assert!(total.mismatches.is_empty(), "{:#?}", total.mismatches);
        assert!(total.cases >= 12_000 && total.traps >= 2_000 && total.depth_limits > 0);
        assert!(
            total.budget * 50 <= total.cases,
            "lane budget skips {}",
            total.budget
        );
    });
}

/// Every named mutation of the driver or lowering is caught by the generated
/// differential on both schedules (each case is a soundness bug the oracle
/// comparison rejects).
#[test]
fn every_named_mutation_is_caught() {
    support::with_stack(|| {
        for mutation in [
            Mutation::LastFailure,
            Mutation::EagerArgumentFailure,
            Mutation::DepthOffByOne,
            Mutation::SwapBindings,
            Mutation::StaleFallback,
            Mutation::CancelEarlier,
            Mutation::EarlyJoin,
        ] {
            for schedule in [Schedule::Staged, Schedule::Cascade] {
                let mut rng = Rng(0x7a5c_f04c);
                let mut caught = 0;
                for program in 0..24 {
                    let source = generated(&mut rng);
                    let admitted = admit(&source).expect("admit");
                    let cases = generated_cases(&mut rng, 160);
                    let plans = cpu_plans(&admitted.program);
                    let (wide, tight) = match schedule {
                        Schedule::Staged => (&plans[0].1, &plans[1].1),
                        Schedule::Cascade => (&plans[2].1, &plans[3].1),
                    };
                    let plan = if program % 2 == 0 { wide } else { tight };
                    let name = format!("mut-{mutation:?}-{schedule:?}-{program:02}");
                    caught += differential(&name, &admitted, plan, mutation, &cases)
                        .mismatches
                        .len();
                    if caught > 0 {
                        break;
                    }
                }
                assert!(
                    caught > 0,
                    "{mutation:?} survived the generated differential ({schedule:?})"
                );
                eprintln!("{mutation:?} ({schedule:?}): caught");
            }
        }
    });
}

#[test]
fn refusals_name_their_reason() {
    let refused = |source: &str| admit(source).err().map(|r| r.code());
    // No residual fork-join: belongs to a flat lowering.
    assert_eq!(
        refused("fn run(a: i32, b: i32) -> i32 { return a + b; }"),
        Some(Refusal::Flat.code())
    );
    // A machine call nested in an expression.
    assert_eq!(
        refused(
            "fn t(n: i32, s: i32) -> i32 { if n <= 0 { return s; } let a = t(n - 1, s); let b = t(n - 2, s); return a + t(n - 1, b); }\n\
             fn run(a: i32, b: i32) -> i32 { return t(a, b); }"
        ),
        Some("task_nested_call")
    );
    // Mutual recursion.
    assert_eq!(
        refused(
            "fn f(n: i32) -> i32 { if n <= 0 { return 0; } let a = g(n - 1); let b = g(n - 2); return a + b; }\n\
             fn g(n: i32) -> i32 { if n <= 0 { return 1; } let a = f(n - 1); let b = f(n - 2); return a * b; }\n\
             fn run(a: i32, b: i32) -> i32 { return f(a + b); }"
        ),
        Some("task_mutual_recursion")
    );
}

#[test]
fn plan_widths_are_powers_of_two_and_continuations_take_epochs() {
    let admitted = admit(&c1_source(2)).expect("admit C1-02");
    assert!(admitted.program.continuation_calls());
    let odd = task::Profile {
        width: 1000,
        ..task::Profile::METAL_STAGED
    };
    let plan = task::plan(&admitted.program, &odd);
    assert_eq!(plan.width, 1024);
    assert!(plan.epochs > 1);
    let plain = admit(&c1_source(0)).expect("admit C1-00");
    assert!(!plain.program.continuation_calls());
    assert_eq!(
        task::plan(&plain.program, &task::Profile::METAL_STAGED).epochs,
        1
    );
}

/// C1 programs with a resident or host-to-host MISS in a committed scored
/// report (`tests/fixtures/archived-evidence/gnet-p4b/<dir>/report-<name>.json`).
fn measured_misses(dir: &str, name: &str) -> Vec<usize> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
        "tests/fixtures/archived-evidence/gnet-p4b/{dir}/report-{name}.json"
    ));
    let rows: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let rows = rows.as_array().unwrap();
    assert_eq!(rows.len(), 60, "{dir}: 12 programs x 5 inputs");
    let mut misses: Vec<usize> = rows
        .iter()
        .filter(|row| row["resident"][2] == "MISS" || row["host"][2] == "MISS")
        .map(|row| row["program"].as_u64().unwrap() as usize)
        .collect();
    misses.dedup();
    misses
}

#[test]
fn profiles_carry_the_scored_calibration_and_refuse_every_measured_miss() {
    support::with_stack(|| {
        let devices = [
            ("Metal", task::Profile::METAL, "bench-metal", "metal"),
            (
                "Metal 3.1",
                task::Profile::METAL_STAGED,
                "bench-metal-staged",
                "metal-staged",
            ),
            ("CUDA", task::Profile::CUDA, "bench-cuda", "cuda"),
        ];
        for (device, profile, dir, name) in devices {
            // The profile carries the scored pilot's frozen calibration.
            let pilot: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
                    format!("tests/fixtures/archived-evidence/gnet-p4b/{dir}/pilot-{name}.json"),
                ))
                .unwrap(),
            )
            .unwrap();
            assert_eq!(pilot["t_width"], profile.width, "{device} width");
            if profile.schedule == Schedule::Cascade {
                assert_eq!(pilot["t_lanes"], profile.lanes, "{device} lanes");
            }
            let misses = measured_misses(dir, name);
            for index in 0..12 {
                let program = c1(index).program;
                let routed = task::route(&program, &profile);
                if misses.contains(&index) {
                    assert_eq!(routed, Err(Refusal::CostModel), "{device} C1-{index:02}");
                } else {
                    assert_eq!(
                        routed,
                        Ok(task::plan(&program, &profile)),
                        "{device} C1-{index:02}"
                    );
                }
                // The CPU model never refuses.
                assert!(task::route(&program, &task::Profile::CPU_MODEL).is_ok());
            }
            eprintln!("{device}: refused {misses:?}");
        }
        assert_eq!(Refusal::CostModel.code(), "task_cost_model");
    });
}
