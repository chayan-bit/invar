//! M4-07 bounded summary scans on device hardware, compared bit for bit with the
//! source-bound CPU oracle and an independent sequential reference. Ignored by
//! default; a missing GPU fails rather than skips. Shared by the CUDA and Metal
//! binaries; the including module supplies `Device`, `open`, `run_summary` and `GPU`.
use super::corpus::{Case, cases, contract, independent, prepare};
use super::{Device, GPU, open, run_summary};
use std::collections::BTreeMap;

/// What each backend's summary result reports, device-independently.
pub struct Out {
    pub output: Vec<u32>,
    /// Look-back fallbacks (Metal only; CUDA has no look-back).
    pub fallbacks: u64,
    /// Generated CUDA compiles without diagnostics (CUDA only; Metal reports none).
    pub compiler_warned: bool,
}

#[test]
#[ignore = "requires a GPU (CUDA with NVRTC 12.4+, or real Apple7+ Metal)"]
fn issue_m4_07_gpu_summaries_match_the_cpu_oracle_at_every_prefix() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut device: Device = open();
    println!(
        "M4-07 device={} caps={:?}",
        device.name(),
        device.capabilities()
    );
    let mut contracts = BTreeMap::new();
    let (mut launched, mut words, mut fallbacks) = (0, 0usize, 0u64);
    for case in cases() {
        let c = contracts
            .entry((format!("{:?}", case.family), case.prefix))
            .or_insert_with(|| contract(case.family, case.prefix));
        let prepared = prepare(c, &case);
        let plan = prepared.device_plan(device.capabilities()).unwrap();
        let result = run_summary(&mut device, &plan, prepared.coefficients(), prepared.seed())
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        assert_eq!(result.output, prepared.device_words(), "{}", case.name);
        assert!(!result.compiler_warned, "{}", case.name);
        assert_eq!(
            result.output,
            independent(case.family, case.prefix, &case.coefficients, &case.seed),
            "{}",
            case.name
        );
        launched += usize::from(case.steps > 0);
        words += result.output.len();
        fallbacks += result.fallbacks;
    }
    assert!(!device.is_poisoned());
    println!(
        "M4-07: {launched} launched cases, {words} words equal to the CPU oracle, {fallbacks} look-back fallbacks"
    );
}

#[test]
#[ignore = "requires a GPU (CUDA with NVRTC 12.4+, or real Apple7+ Metal)"]
fn issue_m4_07_gpu_independent_instances_do_not_interfere() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut device: Device = open();
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
            let result = run_summary(&mut device, &plan, p.coefficients(), p.seed()).unwrap();
            assert_eq!(
                result.output,
                p.device_words(),
                "{} round {round}",
                case.name
            );
        }
    }
    println!(
        "M4-07: {} independent instances, two interleaved rounds",
        prepared.len()
    );
}
