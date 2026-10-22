//! M4-07 bounded summary scans on CUDA hardware, compared bit for bit with the
//! source-bound CPU oracle and an independent sequential reference. Ignored by
//! default; a missing GPU fails rather than skips.
#![cfg(feature = "cuda")]
#[path = "issue_m4_07_device/corpus.rs"]
mod corpus;

use corpus::{Case, cases, contract, independent, prepare};
use invar_cuda::{Error, cuda::CudaDevice};
use invar_device::{Provenance, summary};
use std::{collections::BTreeMap, path::Path, sync::Mutex};

static GPU: Mutex<()> = Mutex::new(());

fn open() -> (CudaDevice, &'static Path) {
    let device = CudaDevice::open().expect("CUDA_UNAVAILABLE: requires a CUDA GPU and NVRTC 12.4+");
    (device, Path::new(env!("CARGO_BIN_EXE_invar-nvrtc")))
}

#[test]
#[ignore = "requires a CUDA GPU and NVRTC 12.4+"]
fn issue_m4_07_gpu_summaries_match_the_cpu_oracle_at_every_prefix() {
    let _gpu = GPU.lock().unwrap();
    let (mut device, compiler) = open();
    println!(
        "M4-07 CUDA device={} caps={:?}",
        device.name(),
        device.capabilities()
    );
    let mut contracts = BTreeMap::new();
    let (mut launched, mut words) = (0, 0usize);
    for case in cases() {
        let c = contracts
            .entry((format!("{:?}", case.family), case.prefix))
            .or_insert_with(|| contract(case.family, case.prefix));
        let prepared = prepare(c, &case);
        let plan = prepared.device_plan(device.capabilities()).unwrap();
        let result = device
            .run_summary(&plan, prepared.coefficients(), prepared.seed(), compiler)
            .unwrap_or_else(|e| panic!("{}: {e:?}", case.name));
        assert_eq!(result.output, prepared.device_words(), "{}", case.name);
        // Generated CUDA compiles without diagnostics.
        assert!(
            !result.nvrtc_log.contains("warning"),
            "{}: {}",
            case.name,
            result.nvrtc_log
        );
        assert_eq!(
            result.output,
            independent(case.family, case.prefix, &case.coefficients, &case.seed),
            "{}",
            case.name
        );
        launched += usize::from(case.steps > 0);
        words += result.output.len();
    }
    assert!(!device.is_poisoned());
    println!("M4-07 CUDA: {launched} launched cases, {words} words equal to the CPU oracle");
}

#[test]
#[ignore = "requires a CUDA GPU and NVRTC 12.4+"]
fn issue_m4_07_gpu_independent_instances_do_not_interfere() {
    let _gpu = GPU.lock().unwrap();
    let (mut device, compiler) = open();
    let all = cases();
    let picks: Vec<&Case> = all
        .iter()
        .filter(|c| c.prefix && [4097, 12289].contains(&c.steps))
        .collect();
    let mut contracts = BTreeMap::new();
    let prepared: Vec<_> = picks
        .iter()
        .map(|case| {
            let c = contracts
                .entry((format!("{:?}", case.family), case.prefix))
                .or_insert_with(|| contract(case.family, case.prefix));
            (case, prepare(c, case))
        })
        .collect();
    // Interleave instances of different families and lengths twice over.
    for round in 0..2 {
        for (case, p) in prepared.iter().rev().skip(round) {
            let plan = p.device_plan(device.capabilities()).unwrap();
            let result = device
                .run_summary(&plan, p.coefficients(), p.seed(), compiler)
                .unwrap();
            assert_eq!(
                result.output,
                p.device_words(),
                "{} round {round}",
                case.name
            );
        }
    }
    println!(
        "M4-07 CUDA: {} independent instances, two interleaved rounds",
        prepared.len()
    );
}

#[test]
#[ignore = "requires a CUDA GPU and NVRTC 12.4+"]
fn issue_m4_07_gpu_invalid_inputs_are_refused_before_launch() {
    let _gpu = GPU.lock().unwrap();
    let (mut device, compiler) = open();
    let caps = device.capabilities();
    let origin = Provenance {
        source_identity: 7,
        graph_revision: 0,
    };
    let verify = |family, observation, steps| {
        summary::verify(
            summary::Plan {
                family,
                observation,
                steps,
                source_identity: 7,
                graph_revision: 0,
            },
            origin,
            caps,
        )
        .unwrap()
    };
    let table = verify(
        summary::Family::Table(3),
        summary::Observation::InclusivePrefixes,
        2,
    );
    for (coefficients, seed, message) in [
        (
            vec![0, 1, 2, 2, 3, 0],
            vec![0],
            "table entry 4 = 3 is not a state",
        ),
        (vec![0, 1, 2, 2, 1, 0], vec![3], "seed 3 is not a state"),
        (vec![0, 1, 2], vec![0], "input shape differs"),
    ] {
        match device.run_summary(&table, &coefficients, &seed, compiler) {
            Err(Error::Driver(text)) => assert!(text.contains(message), "{text}"),
            other => panic!("{other:?}"),
        }
    }
    let metal = summary::verify(
        summary::Plan {
            family: summary::Family::Table(3),
            observation: summary::Observation::FinalState,
            steps: 2,
            source_identity: 7,
            graph_revision: 0,
        },
        origin,
        invar_device::Capabilities::Metal {
            apple_family: 9,
            msl_major: 3,
            msl_minor: 1,
        },
    )
    .unwrap();
    assert_eq!(
        device.run_summary(&metal, &[0; 6], &[0], compiler).err(),
        Some(Error::Limit)
    );
    // Refusals neither poison nor quarantine: the next valid scan runs.
    assert!(!device.is_poisoned());
    let result = device
        .run_summary(&table, &[1, 2, 0, 2, 0, 1], &[0], compiler)
        .unwrap();
    assert_eq!(result.output, [1, 0]);
}

/// Largest transfer-admitted K = 16 and d = 4 scans, resident against copies of
/// the same bytes on the same GPU. Recorded, not a performance gate.
#[test]
#[ignore = "requires a CUDA GPU and NVRTC 12.4+"]
fn issue_m4_07_gpu_largest_scans_are_recorded() {
    let _gpu = GPU.lock().unwrap();
    let (mut device, compiler) = open();
    let caps = device.capabilities();
    for (family, source_family) in [
        (
            summary::Family::Table(16),
            invar::summary_source::Family::Table(16),
        ),
        (
            summary::Family::Matrix(4),
            invar::summary_source::Family::Matrix(4),
        ),
    ] {
        let per_step = u64::from(family.step_words() + family.state_width());
        let steps =
            ((summary::MAX_TRANSFER_BYTES / 4 - u64::from(family.state_width())) / per_step) as u32;
        let (coefficients, seed) = corpus::random(source_family, steps, 99);
        for (observation, prefix) in [
            (summary::Observation::InclusivePrefixes, true),
            (summary::Observation::FinalState, false),
        ] {
            let plan = summary::verify(
                summary::Plan {
                    family,
                    observation,
                    steps,
                    source_identity: 7,
                    graph_revision: 0,
                },
                Provenance {
                    source_identity: 7,
                    graph_revision: 0,
                },
                caps,
            )
            .unwrap();
            let b = device
                .benchmark_summary(&plan, &coefficients, &seed, compiler)
                .unwrap();
            assert_eq!(
                b.output,
                independent(source_family, prefix, &coefficients, &seed),
                "{family:?} {observation:?}"
            );
            println!(
                "M4-07 CUDA bench {family:?} {observation:?} steps={steps} required_bytes={} summary_GBps={:.1} copy_GBps={:.1} fraction={:.3} summary_ms={:.3} copy_kernel_ms={:.3} copy_driver_ms={:.3}; not a gate",
                b.required_bytes,
                b.summary_gbps,
                b.copy_gbps,
                b.bandwidth_fraction,
                b.summary.median_ms,
                b.copy_kernel.median_ms,
                b.copy_driver.median_ms
            );
        }
    }
}
