#![forbid(unsafe_code)]
pub use invar::{
    check, front, general_net, hex, layout_contract, mir, numeric_contract, tensor_cache,
};
#[path = "../qualification/mod.rs"]
#[allow(dead_code)] // This target exercises the device half of the shared harness.
mod qualification;
use qualification::device::{self, Target};
use serde_json::json;
#[cfg(feature = "cuda")]
#[path = "support/invar_bin.rs"]
mod invar_bin;

#[test]
fn issue_m5_02_full_pipeline_qualification_device_boundaries_and_mutation_controls() {
    let checks = device::boundary_checks().unwrap();
    // A build without the device feature refuses that target before any work.
    #[cfg(not(feature = "cuda"))]
    assert_eq!(
        device::run(Target::Cuda, None).unwrap_err().code,
        "M5_DEVICE_FEATURE_REQUIRED"
    );
    assert_eq!(checks["status"], "passed");
    assert_eq!(
        device::compare_words(&[7], &[8]).unwrap_err().code,
        "M5_DEVICE_DISAGREEMENT"
    );
    assert_eq!(
        device::preflight(Target::Metal, "wrong-device", device::METAL)
            .unwrap_err()
            .code,
        "M5_PREFLIGHT_MISMATCH"
    );
    for target in [Target::Cuda, Target::Metal] {
        let report = device::pending(target, "host-not-run");
        assert_eq!(report["rows"].as_array().unwrap().len(), 20);
        assert!(
            report["rows"]
                .as_array()
                .unwrap()
                .iter()
                .all(|r| r["status"] == "pending")
        );
        assert!(device::require(&json!({"device_qualification":[report]})).is_err());
    }
}

#[test]
fn issue_m5_02_full_pipeline_qualification_device_corpus_all_families_prepare() {
    for target in [Target::Cuda, Target::Metal] {
        let inventory = device::inventory(target).unwrap();
        assert_eq!(inventory["families"].as_object().unwrap().len(), 10);
        assert!(inventory["case_count"].as_u64().unwrap() > 200);
        println!("{inventory}");
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "requires the exact Apple M4; run under the exclusive Mac wrapper"]
fn issue_m5_02_full_pipeline_qualification_metal_gpu() {
    hardware(Target::Metal, None);
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires the exact RTX 5070 Ti and a local NVRTC helper"]
fn issue_m5_02_full_pipeline_qualification_cuda_gpu() {
    hardware(Target::Cuda, Some(&invar_bin::invar_bin("invar-nvrtc")));
}
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
fn hardware(target: Target, compiler: Option<&std::path::Path>) {
    let mut report = match device::run(target, compiler) {
        Ok(report) => report,
        // On any GPU other than the pinned target (CI runs this set on a shared
        // H100), the only correct outcome is the typed preflight refusal before
        // any work; nothing is qualified. A refusal on the pinned device itself,
        // such as a toolchain mismatch, still fails.
        Err(failure)
            if failure.code == "M5_PREFLIGHT_MISMATCH"
                && !failure
                    .detail
                    .starts_with(&format!("{} ", device::pinned_device(target))) =>
        {
            println!(
                "device is not the pinned target, refused before any work: {}",
                failure.detail
            );
            return;
        }
        Err(failure) => panic!("{failure:?}"),
    };
    if let Some(directory) = std::env::var_os("M502_LIBRARY_DIRECTORY") {
        let libraries = device::libraries(target, Some(std::path::Path::new(&directory))).unwrap();
        assert!(libraries.iter().any(|r| r["status"] == "qualified"));
        report["rows"].as_array_mut().unwrap().extend(libraries);
    }
    let rows = report["rows"].as_array().unwrap();
    for family in device::FAMILIES {
        assert!(
            rows.iter().any(|r| r["family"] == family
                && r["track"] == "generated"
                && r["status"] == "qualified"),
            "{family}"
        );
    }
    assert_eq!(report["adversarial"]["checks"].as_array().unwrap().len(), 8);
    if let Some(path) = std::env::var_os("M502_DEVICE_REPORT") {
        qualification::publish(std::path::Path::new(&path), &report).unwrap();
    }
    println!(
        "device={} rows={} adversarial={} pending=criterion-c",
        target.name(),
        rows.len(),
        report["adversarial"]
    );
}
