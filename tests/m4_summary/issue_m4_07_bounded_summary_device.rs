//! M4-07 device lowering and reason codes, selected by the acceptance name
//! `issue_m4_07_bounded_summary_families`. Hardware runs are in
//! `gpu_cuda::m4_07` and `gpu_metal::m4_07`.
#![forbid(unsafe_code)]
#[path = "../issue_m4_07_device/corpus.rs"]
mod corpus;

use corpus::{cases, contract, independent, prepare};
use invar::{
    eval,
    summary_source::{Family, host},
};
use invar_device::{
    Capabilities, Error, Provenance,
    summary::{self, InputError, Observation, Plan},
};
use std::collections::BTreeMap;

const CUDA: Capabilities = Capabilities::Cuda {
    major: 12,
    minor: 9,
    sm: 120,
};
const METAL: Capabilities = Capabilities::Metal {
    apple_family: 9,
    msl_major: 3,
    msl_minor: 1,
};

fn plan(family: summary::Family, observation: Observation, steps: u32) -> Plan {
    Plan {
        family,
        observation,
        steps,
        source_identity: 7,
        graph_revision: 0,
    }
}
const ORIGIN: Provenance = Provenance {
    source_identity: 7,
    graph_revision: 0,
};

#[test]
fn issue_m4_07_bounded_summary_families_device_plans_come_from_the_verified_source_scan() {
    let mut contracts = BTreeMap::new();
    let (mut checked, mut words) = (0, 0);
    for case in cases() {
        let c = contracts
            .entry((format!("{:?}", case.family), case.prefix))
            .or_insert_with(|| contract(case.family, case.prefix));
        let prepared = prepare(c, &case);
        let expected = independent(case.family, case.prefix, &case.coefficients, &case.seed);
        assert_eq!(prepared.device_words(), expected, "{}", case.name);
        for caps in [CUDA, METAL, Capabilities::cuda_12_9_sm89()] {
            let verified = prepared.device_plan(caps).unwrap();
            let p = verified.plan();
            let (family, dimension) = match case.family {
                Family::Table(k) => (summary::Family::Table(k), k),
                Family::Matrix(d) => (summary::Family::Matrix(d), d),
            };
            assert_eq!(p.family, family);
            assert_eq!(
                p.observation,
                if case.prefix {
                    Observation::InclusivePrefixes
                } else {
                    Observation::FinalState
                }
            );
            assert_eq!(p.steps, case.steps);
            assert_eq!(p.source_identity, prepared.source_identity());
            assert_eq!(verified.output_words(), expected.len());
            let (source, kernels) = match caps {
                Capabilities::Cuda { .. } => (
                    summary::emit_cuda(&verified).unwrap(),
                    &["invar_summary_scan_v1"][..],
                ),
                Capabilities::Metal { .. } => (
                    summary::emit_msl(&verified).unwrap(),
                    &[
                        "invar_summary_reduce_v1",
                        "invar_summary_final_v3",
                        "invar_summary_clear_v3",
                        "invar_summary_scan_v3",
                    ][..],
                ),
            };
            assert!(source.starts_with(&format!(
                "// {} family={} dimension={dimension} ",
                summary::SCHEDULE,
                family.id()
            )));
            for kernel in kernels {
                assert!(source.contains(kernel), "{kernel}");
            }
            // Composition order is fixed: no unordered shuffles or atomic
            // reductions. CUDA's only atomic is the tile ticket. Metal's only
            // read-modify-write atomics are the ticket and the fallback
            // counter; its look-back slots are relaxed atomic loads and stores
            // of self-flagged words, and its spin is bounded.
            for banned in ["__shfl_xor", "__shfl_down", "simd_shuffle_xor", "simd_sum"] {
                assert!(!source.contains(banned), "{banned}");
            }
            match caps {
                Capabilities::Cuda { .. } => {
                    assert_eq!(source.matches("atomic").count(), 1);
                    assert_eq!(source.matches("atomicAdd(ticket, 1u)").count(), 1);
                }
                Capabilities::Metal { .. } => {
                    // Both modes share one source; the preprocessor keeps
                    // the look-back for prefixes only.
                    assert_eq!(source.matches("atomic_fetch_").count(), 2);
                    assert_eq!(
                        source.matches("memory_order_").count(),
                        source.matches("memory_order_relaxed").count()
                    );
                    assert!(source.contains("spin < INVAR_SPINS"));
                    assert!(!source.contains("while"));
                }
            }
        }
        checked += 1;
        words += prepared.device_words().len();
    }
    println!("M4-07 device plans: {checked} source-bound cases, {words} oracle words");
}

#[test]
fn issue_m4_07_bounded_summary_families_device_verifier_refuses_out_of_contract_plans() {
    use summary::Family::{Matrix, Table};
    let ok = |p| summary::verify(p, ORIGIN, CUDA);
    for family in [Table(0), Table(17), Matrix(0), Matrix(5)] {
        assert_eq!(
            ok(plan(family, Observation::FinalState, 1)).err(),
            Some(Error::Operations)
        );
    }
    for family in [Table(1), Table(16), Matrix(1), Matrix(4)] {
        assert!(ok(plan(family, Observation::InclusivePrefixes, 4097)).is_ok());
    }
    let unsourced = Plan {
        source_identity: 0,
        ..plan(Table(2), Observation::FinalState, 1)
    };
    assert_eq!(
        summary::verify(
            unsourced,
            Provenance {
                source_identity: 0,
                graph_revision: 0
            },
            CUDA
        )
        .err(),
        Some(Error::Source)
    );
    for origin in [
        Provenance {
            source_identity: 8,
            graph_revision: 0,
        },
        Provenance {
            source_identity: 7,
            graph_revision: 1,
        },
    ] {
        assert_eq!(
            summary::verify(plan(Table(2), Observation::FinalState, 1), origin, CUDA).err(),
            Some(Error::Graph)
        );
    }
    // Transfer cap: (steps * step_words + width + output) * 4 bytes, exactly.
    let cap = summary::MAX_TRANSFER_BYTES / 4;
    for (family, observation, per_step, fixed) in [
        (Table(1), Observation::FinalState, 1, 2),
        (Table(16), Observation::InclusivePrefixes, 17, 1),
        (Matrix(4), Observation::FinalState, 16, 8),
        (Matrix(4), Observation::InclusivePrefixes, 20, 4),
    ] {
        let largest = ((cap - fixed) / per_step) as u32;
        let admitted = ok(plan(family, observation, largest)).unwrap();
        assert!(admitted.transfer_bytes() <= summary::MAX_TRANSFER_BYTES);
        assert_eq!(
            ok(plan(family, observation, largest + 1)).err(),
            Some(Error::Bounds),
            "{family:?} {observation:?}"
        );
    }
    assert_eq!(
        ok(plan(Table(1), Observation::FinalState, u32::MAX)).err(),
        Some(Error::Bounds)
    );
    let base = plan(Matrix(2), Observation::FinalState, 3);
    for caps in [
        Capabilities::Cuda {
            major: 12,
            minor: 3,
            sm: 89,
        },
        Capabilities::Cuda {
            major: 11,
            minor: 8,
            sm: 89,
        },
        Capabilities::Cuda {
            major: 12,
            minor: 9,
            sm: 35,
        },
        Capabilities::Metal {
            apple_family: 6,
            msl_major: 3,
            msl_minor: 1,
        },
        Capabilities::Metal {
            apple_family: 9,
            msl_major: 3,
            msl_minor: 0,
        },
    ] {
        assert_eq!(
            summary::verify(base, ORIGIN, caps).err(),
            Some(Error::Toolchain)
        );
    }
    let cuda = summary::verify(base, ORIGIN, CUDA).unwrap();
    let metal = summary::verify(base, ORIGIN, METAL).unwrap();
    assert_eq!(summary::emit_msl(&cuda).err(), Some(Error::Toolchain));
    assert_eq!(summary::emit_cuda(&metal).err(), Some(Error::Toolchain));
    assert_eq!(summary::emit_cuda(&cuda), summary::emit_cuda(&cuda));
}

#[test]
fn issue_m4_07_bounded_summary_families_device_inputs_are_checked_before_launch() {
    let table = summary::verify(
        plan(summary::Family::Table(3), Observation::FinalState, 2),
        ORIGIN,
        CUDA,
    )
    .unwrap();
    assert_eq!(
        summary::check_inputs(&table, &[0, 1, 2, 2, 1, 0], &[2]),
        Ok(())
    );
    assert_eq!(
        summary::check_inputs(&table, &[0, 1, 2, 2, 1], &[2]),
        Err(InputError::Shape)
    );
    assert_eq!(
        summary::check_inputs(&table, &[0, 1, 2, 2, 1, 0], &[]),
        Err(InputError::Shape)
    );
    let entry = summary::check_inputs(&table, &[0, 1, 2, 2, 3, 0], &[2]).unwrap_err();
    assert_eq!(entry, InputError::TableEntry { index: 4, value: 3 });
    assert_eq!(
        entry.to_string(),
        "invar-bounded-summary-scan-v3 table entry 4 = 3 is not a state"
    );
    assert_eq!(
        summary::check_inputs(&table, &[0, 1, 2, 2, 1, 0], &[3]),
        Err(InputError::Seed { value: 3 })
    );
    let matrix = summary::verify(
        plan(
            summary::Family::Matrix(2),
            Observation::InclusivePrefixes,
            2,
        ),
        ORIGIN,
        METAL,
    )
    .unwrap();
    assert_eq!(
        summary::check_inputs(&matrix, &[u32::MAX; 8], &[u32::MAX; 2]),
        Ok(())
    );
    assert_eq!(
        summary::check_inputs(&matrix, &[0; 8], &[0; 3]),
        Err(InputError::Shape)
    );
}

/// `main` calls `run` for two steps of all-zero tables or all-one d x d matrices.
fn caller(function: &str, family: Family) -> String {
    let call = match family {
        Family::Table(k) => format!(
            "let input: Array<u32> = filled_u32({}u32,0u32); let answer = run(input,0u32,2u32);",
            2 * u32::from(k)
        ),
        Family::Matrix(d) => format!(
            "let input: Array<u32> = filled_u32({}u32,1u32); let seed: Array<u32> = filled_u32({d}u32,1u32); let answer = run(input,seed,2u32);",
            2 * u32::from(d) * u32::from(d)
        ),
    };
    format!("{function} fn main() -> i32 {{ {call} return 7; }}")
}

/// K = 17, d = 5 and state-dependent coefficients stay serial: the program
/// computes exactly what the source does, and the run records why.
#[test]
fn issue_m4_07_bounded_summary_families_refusals_stay_serial_with_reason_codes() {
    let k17 = corpus::source(Family::Table(17), true);
    let d5 = corpus::source(Family::Matrix(5), false);
    let state = corpus::source(Family::Table(2), false)
        .replace("coeff[i * 2u32 + state]", "coeff[state * 2u32 + state]");
    for (program, reason) in [
        (caller(&k17, Family::Table(17)), "summary_family_bound"),
        (
            caller(&corpus::source(Family::Table(17), false), Family::Table(17)),
            "summary_family_bound",
        ),
        (caller(&d5, Family::Matrix(5)), "summary_family_bound"),
        (caller(&state, Family::Table(2)), "summary_state_dependent"),
    ] {
        let run = host::run_source(program.as_bytes(), 100_000, corpus::BUDGET).unwrap();
        let checked =
            invar::check::check(invar::front::parse(program.as_bytes()).unwrap(), true).unwrap();
        let serial = eval::observe(&checked, 100_000);
        assert_eq!(run.observation.result, Ok(eval::Value::I32(7)), "{program}");
        assert_eq!(run.observation.result, serial.result);
        assert_eq!(run.observation.steps_used, serial.steps_used);
        assert_eq!(run.observation.cleanup_trace, serial.cleanup_trace);
        assert_eq!(run.completed, 0);
        assert_eq!(run.refusals, [reason], "{program}");
    }
    // The bounded neighbours are admitted and complete on the CPU host.
    for family in [Family::Table(16), Family::Matrix(4)] {
        let program = caller(&corpus::source(family, true), family);
        let run = host::run_source(program.as_bytes(), 100_000, corpus::BUDGET).unwrap();
        assert_eq!(run.observation.result, Ok(eval::Value::I32(7)));
        assert_eq!((run.completed, run.refusals.len()), (1, 0), "{family:?}");
    }
}
