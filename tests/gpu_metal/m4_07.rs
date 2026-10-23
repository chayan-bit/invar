//! M4-07 on Metal: the shared summary tests plus the look-back schedules, the Metal
//! refusals and cache report, and the largest scans.
use invar_device::{Provenance, summary};
use invar_metal::{Error, MetalDevice};
use std::{collections::BTreeMap, sync::Mutex};

#[path = "../issue_m4_07_device/corpus.rs"]
mod corpus;
use corpus::{cases, contract, independent, prepare};

type Device = MetalDevice;

static GPU: Mutex<()> = Mutex::new(());

fn open() -> Device {
    // Kernel qualification explicitly permits development-time MSL compilation.
    MetalDevice::open_runtime_source().expect("Metal Apple7+ GPU required")
}

fn run_summary(
    device: &mut Device,
    plan: &summary::Verified,
    coefficients: &[u32],
    seed: &[u32],
) -> Result<body::Out, String> {
    let r = device
        .run_summary(plan, coefficients, seed)
        .map_err(|e| format!("{e:?}"))?;
    Ok(body::Out {
        output: r.output,
        fallbacks: u64::from(r.fallbacks),
        compiler_warned: false,
    })
}

#[path = "../support/gpu_m4_07.rs"]
mod body;

/// The prefix look-back's result never depends on the schedule. With no polls every
/// predecessor is unpublished to the poller, so each threadgroup reduces all
/// of its predecessors itself (exactly tiles * (tiles - 1) / 2 fallbacks); with
/// tiles handed out last-first, the threadgroups that start first wait on
/// tiles no threadgroup has started yet, which only the fallback can resolve;
/// with published states ignored, every look-back walks whole windows of
/// aggregates back to the seed.
#[test]
#[ignore = "requires a real Apple7+ Metal GPU"]
fn issue_m4_07_metal_gpu_look_back_is_schedule_independent() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut device = open();
    let mut contracts = BTreeMap::new();
    let schedules = [
        summary::LookBack {
            spins: 0,
            reverse: false,
            states: true,
        },
        summary::LookBack {
            spins: 0,
            reverse: true,
            states: true,
        },
        summary::LookBack {
            spins: 1,
            reverse: true,
            states: true,
        },
        summary::LookBack {
            spins: summary::METAL_SPINS,
            reverse: true,
            states: true,
        },
        summary::LookBack {
            spins: summary::METAL_SPINS,
            reverse: false,
            states: false,
        },
    ];
    let (mut runs, mut fallbacks) = (0, [0u64; 5]);
    for case in cases() {
        let c = contracts
            .entry((format!("{:?}", case.family), case.prefix))
            .or_insert_with(|| contract(case.family, case.prefix));
        let prepared = prepare(c, &case);
        let plan = prepared.device_plan(device.capabilities()).unwrap();
        // Only prefix plans look back. Fallback work is quadratic in the
        // tile count, so keep to cases of 2..=64 tiles.
        if !case.prefix || !(2..=64).contains(&plan.blocks()) {
            continue;
        }
        let tiles = u64::from(plan.blocks());
        for (index, look_back) in schedules.into_iter().enumerate() {
            let result = device
                .run_summary_with(&plan, prepared.coefficients(), prepared.seed(), look_back)
                .unwrap_or_else(|e| panic!("{} {look_back:?}: {e:?}", case.name));
            assert_eq!(
                result.output,
                prepared.device_words(),
                "{} {look_back:?}",
                case.name
            );
            if look_back.spins == 0 {
                assert_eq!(
                    u64::from(result.fallbacks),
                    tiles * (tiles - 1) / 2,
                    "{}",
                    case.name
                );
            }
            fallbacks[index] += u64::from(result.fallbacks);
            runs += 1;
        }
    }
    assert!(runs >= 40, "{runs}");
    assert!(!device.is_poisoned());
    println!(
        "M4-07 Metal look-back: {runs} adversarial runs equal to the CPU oracle, fallbacks by schedule {schedules:?}: {fallbacks:?}"
    );
}

#[test]
#[ignore = "requires a real Apple7+ Metal GPU"]
fn issue_m4_07_metal_gpu_invalid_inputs_are_refused_before_launch() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut device = open();
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
        match device.run_summary(&table, &coefficients, &seed) {
            Err(Error::Device(text)) => assert!(text.contains(message), "{text}"),
            other => panic!("{other:?}"),
        }
    }
    let cuda = summary::verify(
        summary::Plan {
            family: summary::Family::Table(3),
            observation: summary::Observation::FinalState,
            steps: 2,
            source_identity: 7,
            graph_revision: 0,
        },
        origin,
        invar_device::Capabilities::cuda_12_9_sm89(),
    )
    .unwrap();
    assert!(matches!(
        device.run_summary(&cuda, &[0; 6], &[0]),
        Err(Error::Limit)
    ));
    // A production device never silently compiles a missing summary library.
    let mut production = MetalDevice::open().unwrap();
    assert!(matches!(
        production.run_summary(&table, &[1, 2, 0, 2, 0, 1], &[0]),
        Err(Error::Library(invar_metal::LibraryError::Missing { .. }))
    ));
    assert_eq!(production.cache_report().source_compiles, 0);

    // Refusals neither compile nor quarantine: the next valid scan runs.
    assert_eq!(device.cache_report().source_compiles, 0);
    assert!(!device.is_poisoned());
    let result = device
        .run_summary(&table, &[1, 2, 0, 2, 0, 1], &[0])
        .unwrap();
    assert_eq!(result.output, [1, 0]);
    let cached = device.cache_report();
    assert_eq!(cached.source_compiles, 1);
    assert_eq!(cached.pipeline_creations, 2);
    assert_eq!(
        device
            .run_summary(&table, &[1, 2, 0, 2, 0, 1], &[0])
            .unwrap()
            .output,
        result.output
    );
    // The repeat compiles and creates nothing and keeps every cumulative
    // counter. Since Metal binary archives, the archive report describes the last lookup, and
    // the reuse is an in-memory hit on the same key: no archive lookup, no
    // write (so no write refusal, even when the earlier write was refused,
    // e.g. a full cache) and no pipeline-creation time.
    let mut expected = cached;
    expected.archive.last = invar_metal::ArchiveState::MemoryHit;
    expected.archive.write_refusal = None;
    expected.archive.pipeline_creation_ns = 0;
    assert_eq!(device.cache_report(), expected);
}

/// Largest transfer-admitted K = 16 and d = 4 scans, resident against copies of
/// the same bytes on the same GPU. The section 10 verdict is read from these
/// records; the test itself asserts only the output.
#[test]
#[ignore = "requires a real Apple7+ Metal GPU"]
fn issue_m4_07_metal_gpu_largest_scans_are_recorded() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut device = open();
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
                .benchmark_summary(&plan, &coefficients, &seed)
                .unwrap();
            assert_eq!(
                b.output,
                independent(source_family, prefix, &coefficients, &seed),
                "{family:?} {observation:?}"
            );
            println!(
                "M4-07 Metal bench {family:?} {observation:?} steps={steps} required_bytes={} summary_GBps={:.1} copy_GBps={:.1} fraction={:.3} fallbacks={} summary_ms={:.3} copy_kernel_ms={:.3} copy_blit_ms={:.3}",
                b.required_bytes,
                b.summary_gbps,
                b.copy_gbps,
                b.bandwidth_fraction,
                b.fallbacks,
                b.summary_median_ms,
                b.copy_kernel_median_ms,
                b.copy_blit_median_ms
            );
        }
    }
}
