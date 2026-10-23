//! General-net entry corpus on a real CUDA device against the CPU oracle.
#[path = "../support/gnet_entry.rs"]
mod entry;

#[test]
#[ignore = "requires physical CUDA hardware"]
fn cuda_entry_matches_cpu_oracle_bit_for_bit() {
    let compiler = nvrtc();
    entry::corpus(
        &mut invar::general_net::device_host::cuda_nvrtc::CudaNvrtc::open(&compiler).unwrap(),
        invar::general_net::compiled::Target::Cuda,
    );
}

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
fn nvrtc() -> std::path::PathBuf {
    std::env::var_os("INVAR_NVRTC")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| env!("CARGO_BIN_EXE_invar-nvrtc").into())
}
