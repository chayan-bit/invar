//! Physical CUDA law-collective calibration and sessions; the CPU statistics stay in `tests/net_law_collective_bench.rs`.
#[path = "../support/law_bench.rs"]
mod bench;
#[test]
#[ignore = "physical sm_120 CUDA, INVAR_GPU_EXCLUSIVE=1, records the cost-model calibration"]
fn p8_calibrate_cuda() {
    assert_eq!(std::env::var("INVAR_GPU_EXCLUSIVE").as_deref(), Ok("1"));
    let mut device = invar_cuda::cuda::CudaDevice::open().unwrap();
    let name = device.name().to_owned();
    bench::calibrate(false, &name, |words| {
        device
            .calibrate_law_transfer(words)
            .map_err(|e| format!("{e:?}"))
    });
}

#[test]
#[ignore = "physical sm_120 CUDA, INVAR_GPU_EXCLUSIVE=1, three independent sessions"]
fn p8_d5_cuda_session() {
    assert_eq!(std::env::var("INVAR_GPU_EXCLUSIVE").as_deref(), Ok("1"));
    let mut device = invar_cuda::cuda::CudaDevice::open().unwrap();
    println!("P8 D5 CUDA {} {:?}", device.name(), device.capabilities());
    bench::session(
        false,
        |sources, entry, input, expected, words, n, scan, native| {
            device
                .benchmark_law_pair(
                    sources,
                    entry,
                    &input,
                    expected,
                    words,
                    n,
                    scan,
                    std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc")),
                    native,
                )
                .map_err(|e| format!("{e:?}"))
        },
    );
}
