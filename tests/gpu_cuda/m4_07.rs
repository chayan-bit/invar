//! M4-07 on CUDA: the shared summary tests plus the CUDA refusals and the largest scans.
use invar_cuda::{Error, cuda::CudaDevice};
use invar_device::{Provenance, summary};
use std::{path::Path, sync::Mutex};

#[path = "../issue_m4_07_device/corpus.rs"]
mod corpus;
use corpus::independent;

type Device = CudaDevice;

static GPU: Mutex<()> = Mutex::new(());

fn compiler() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"))
}

fn open() -> Device {
    CudaDevice::open().expect("CUDA_UNAVAILABLE: requires a CUDA GPU and NVRTC 12.4+")
}

fn run_summary(
    device: &mut Device,
    plan: &summary::Verified,
    coefficients: &[u32],
    seed: &[u32],
) -> Result<body::Out, String> {
    let r = device
        .run_summary(plan, coefficients, seed, compiler())
        .map_err(|e| format!("{e:?}"))?;
    Ok(body::Out {
        output: r.output,
        fallbacks: 0,
        // Generated CUDA compiles without diagnostics.
        compiler_warned: r.nvrtc_log.contains("warning"),
    })
}

#[path = "../support/gpu_m4_07.rs"]
mod body;

#[test]
#[ignore = "requires a CUDA GPU and NVRTC 12.4+"]
fn issue_m4_07_gpu_invalid_inputs_are_refused_before_launch() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut device = open();
    let compiler = compiler();
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
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut device = open();
    let compiler = compiler();
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
