#![forbid(unsafe_code)]
#[path = "support/gnet_entry.rs"]
mod entry;
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Metal hardware"]
fn metal_entry_matches_cpu_oracle_bit_for_bit() {
    entry::corpus(
        &mut invar::general_net::device_host::metal::Metal::open_runtime_source().unwrap(),
        invar::general_net::compiled::Target::Metal,
    );
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires physical CUDA hardware"]
fn cuda_entry_matches_cpu_oracle_bit_for_bit() {
    let compiler = nvrtc();
    entry::corpus(
        &mut invar::general_net::device_host::cuda_nvrtc::CudaNvrtc::open(&compiler).unwrap(),
        invar::general_net::compiled::Target::Cuda,
    );
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "requires Apple Metal hardware and a bounded 2^24-element allocation"]
fn metal_entry_accepts_two_to_24_host_elements() {
    entry::large(
        &mut invar::general_net::device_host::metal::Metal::open_runtime_source().unwrap(),
        invar::general_net::compiled::Target::Metal,
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires physical CUDA hardware"]
fn cuda_entry_accepts_two_to_24_host_elements() {
    let compiler = nvrtc();
    entry::large(
        &mut invar::general_net::device_host::cuda_nvrtc::CudaNvrtc::open(&compiler).unwrap(),
        invar::general_net::compiled::Target::Cuda,
    );
}

/// The NVRTC helper: `INVAR_NVRTC` when a host needs a newer NVRTC than the
/// linked one (an sm_120 host), else this build's `invar-nvrtc`.
#[cfg(feature = "cuda")]
fn nvrtc() -> std::path::PathBuf {
    std::env::var_os("INVAR_NVRTC")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| env!("CARGO_BIN_EXE_invar-nvrtc").into())
}
