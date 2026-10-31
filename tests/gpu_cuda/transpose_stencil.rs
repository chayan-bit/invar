//! Issue #2 (M4-02).
#[allow(dead_code)]
#[path = "../support/m402.rs"]
mod support;
use invar_cuda::cuda::CudaDevice;
use invar_device::{source_hash, stencil::Boundary, transpose};
#[test]
#[ignore = "requires a real NVIDIA GPU and a local NVRTC"]
fn gpu_transpose() {
    let mut device = CudaDevice::open().expect("explicit CUDA gate requires hardware");
    let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    for &shape in support::SHAPES.iter().chain(&[[4096, 4096]]) {
        let plan = support::transpose_plan(shape, device.capabilities());
        let input = support::input(shape);
        let begun = std::time::Instant::now();
        let got = device.run_transpose(&plan, &input, compiler).unwrap();
        support::assert_transpose(shape, &input, &got.output);
        {
            let back = support::transpose_plan([shape[1], shape[0]], device.capabilities());
            assert_eq!(
                device
                    .run_transpose(&back, &got.output, compiler)
                    .unwrap()
                    .output,
                input
            );
        }
        println!(
            "{}",
            serde_json::json!({"target":"cuda","device":device.name(),"caps":format!("{:?}",device.capabilities()),"family":"transpose","shape":shape,"source_hash":source_hash(&transpose::emit_cuda(&plan).unwrap()),"transfer_bytes":got.transfer_bytes,"compile_ms":got.compile_ms,"host_to_host_including_compile_ms":begun.elapsed().as_secs_f64()*1000.0,"completed":true,"refusal":null,"scope":"source-extracted"})
        );
    }
}
#[test]
#[ignore = "requires a real NVIDIA GPU and a local NVRTC"]
fn gpu_stencil_ping_pong() {
    let mut device = CudaDevice::open().expect("explicit CUDA gate requires hardware");
    let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    for boundary in [Boundary::ConstantZero, Boundary::Clamp] {
        for &shape in support::SHAPES {
            for steps in [0, 1, 2, 3, 4] {
                let plan = support::stencil_plan(shape, steps, boundary, device.capabilities());
                let input = support::input(shape);
                let got = device.run_stencil_u32(&plan, &input, compiler).unwrap();
                assert_eq!(
                    got.output,
                    support::stencil_oracle(shape, steps, boundary, &input),
                    "{shape:?} {boundary:?} steps={steps}"
                );
                assert_eq!(
                    got.final_buffer,
                    if shape.contains(&0) {
                        0
                    } else {
                        (steps % 2) as usize
                    }
                );
                println!(
                    "{}",
                    serde_json::json!({"target":"cuda","device":device.name(),"family":"stencil-u32","shape":shape,"boundary":boundary.name(),"steps":steps,"phases":got.phases,"final_buffer":got.final_buffer,"source_hash":got.source_hash,"transfer_bytes":got.transfer_bytes,"compile_ms":got.compile_ms,"host_to_host_ms":got.host_to_host_ms,"completed":true,"reason":if got.phases==0 {"empty-or-zero-step-no-submission"}else{"verified-completed"}})
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a real NVIDIA GPU and a local NVRTC"]
fn gpu_failures_do_not_publish() {
    use invar_cuda::{Error, cuda::StencilFault};
    let mut device = CudaDevice::open().unwrap();
    let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    let plan = support::stencil_plan([31, 33], 3, Boundary::Clamp, device.capabilities());
    let input = support::input([31, 33]);
    assert!(
        device
            .run_stencil_u32(&plan, &input[..input.len() - 1], compiler)
            .is_err()
    );
    for fault in [StencilFault::OutOfMemory, StencilFault::LaunchFailure] {
        assert!(
            device
                .run_stencil_u32_with_fault(&plan, &input, compiler, fault)
                .is_err()
        );
        assert!(!device.is_poisoned());
    }
    assert_eq!(
        device
            .run_stencil_u32_with_fault(&plan, &input, compiler, StencilFault::PartialCompletion)
            .err(),
        Some(Error::Timeout)
    );
    assert_eq!(device.quarantine_len(), 1);
    for _ in 0..100 {
        device.reap();
        if !device.is_poisoned() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(!device.is_poisoned());
    assert_eq!(
        device
            .run_stencil_u32(&plan, &input, compiler)
            .unwrap()
            .output,
        support::stencil_oracle([31, 33], 3, Boundary::Clamp, &input)
    );
}

#[test]
#[ignore = "requires a real NVIDIA GPU and a local NVRTC"]
fn gpu_physical_padding() {
    let mut device = CudaDevice::open().unwrap();
    let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    for steps in [2, 3] {
        let plan = support::padded_stencil(steps, device.capabilities());
        let mut input = vec![0x12345678; 3];
        input.extend(support::input([31, 33]));
        input.extend([0x89abcdef; 3]);
        let got = device.run_stencil_u32(&plan, &input, compiler).unwrap();
        support::assert_padded_stencil(&got.output, steps, &input);
    }
}

#[test]
#[ignore = "requires real hardware, 2^28 elements; run alone; CUDA needs the shared transfer-cap change"]
fn gpu_cap_2_28_transpose() {
    let mut device = CudaDevice::open().expect("explicit hardware gate requires device");
    let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    let shape = [16384, 16384];
    let plan = support::transpose_plan(shape, device.capabilities());
    let input = support::input(shape);
    let got = device.run_transpose(&plan, &input, compiler).unwrap();
    support::assert_transpose(shape, &input, &got.output);
    println!(
        "device={} family=transpose elements=268435456 exact_words=268435456 completed=true transfer_bytes={} compile_ms={}",
        device.name(),
        got.transfer_bytes,
        got.compile_ms
    );
}

#[test]
#[ignore = "requires real hardware, 2^28 elements; run alone; CUDA needs the shared transfer-cap change"]
fn gpu_cap_2_28_stencil() {
    let mut device = CudaDevice::open().expect("explicit hardware gate requires device");
    let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    let shape = [16384, 16384];
    let input = support::input(shape);
    for boundary in [Boundary::ConstantZero, Boundary::Clamp] {
        let plan = support::stencil_plan(shape, 1, boundary, device.capabilities());
        let got = device.run_stencil_u32(&plan, &input, compiler).unwrap();
        assert_eq!(
            got.output,
            support::stencil_oracle(shape, 1, boundary, &input)
        );
        assert_eq!(got.final_buffer, 1);
        println!(
            "device={} family=stencil boundary={} elements=268435456 exact_words=268435456 completed=true transfer_bytes={} compile_ms={}",
            device.name(),
            boundary.name(),
            got.transfer_bytes,
            got.compile_ms
        );
    }
}
