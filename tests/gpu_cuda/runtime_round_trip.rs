//! Live CUDA round trip; the mock-device runtime tests stay in `tests/device/cuda_runtime_mock.rs`.
//! Issue label M3-01.
use invar_cuda::Runtime;

#[test]
#[ignore = "requires a CUDA GPU and NVRTC 12.4+"]
fn gpu_round_trip() {
    use invar_cuda::cuda::CudaDevice;
    use std::time::{Duration, Instant};

    let device = CudaDevice::open().expect("CUDA_UNAVAILABLE: requires CUDA GPU and NVRTC 12.4+");
    eprintln!("device={} caps={:?}", device.name(), device.capabilities());
    let mut runtime = Runtime::new(device, 8);
    let bytes = (0u16..4096).map(|v| v as u8).collect::<Vec<_>>();
    let mut submitted = runtime.prepare(&bytes).unwrap().submit().unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(actual) = submitted.poll().unwrap() {
            assert_eq!(actual, bytes);
            break;
        }
        assert!(
            Instant::now() < deadline,
            "CUDA_PENDING: transfer did not complete"
        );
        std::thread::yield_now();
    }
}
