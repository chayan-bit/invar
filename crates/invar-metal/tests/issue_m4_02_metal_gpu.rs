#![cfg(target_os = "macos")]
#[allow(dead_code)]
#[path = "../../../tests/support/m402.rs"]
mod support;
use invar_device::stencil::Boundary;
use invar_metal::{Error, Fault, MetalDevice};
#[test]
#[ignore = "requires a real Apple7 or newer Metal GPU"]
fn issue_m4_02_metal_gpu_transpose() {
    let mut device =
        MetalDevice::open_runtime_source().expect("explicit Metal gate requires hardware");
    for &shape in support::SHAPES.iter().chain(&[[4096, 4096]]) {
        let plan = support::transpose_plan(shape, device.capabilities());
        let input = support::input(shape);
        let got = device.run_transpose(&plan, &input).unwrap();
        support::assert_transpose(shape, &input, &got.output);
        {
            let back = support::transpose_plan([shape[1], shape[0]], device.capabilities());
            assert_eq!(
                device.run_transpose(&back, &got.output).unwrap().output,
                input
            );
        }
        println!(
            "{}",
            serde_json::json!({"target":got.target,"device":device.name(),"caps":format!("{:?}",device.capabilities()),"family":"transpose","shape":shape,"source_hash":got.source_hash,"phases":got.phases,"compile_ms":got.compile_ms,"host_to_host_ms":got.host_to_host_ms,"transfer_bytes":got.transfer_bytes,"completed":true,"scope":"source-extracted"})
        );
    }
}
#[test]
#[ignore = "requires a real Apple7 or newer Metal GPU"]
fn issue_m4_02_metal_gpu_stencil_ping_pong() {
    let mut device =
        MetalDevice::open_runtime_source().expect("explicit Metal gate requires hardware");
    for boundary in [Boundary::ConstantZero, Boundary::Clamp] {
        for &shape in support::SHAPES {
            for steps in [0, 1, 2, 3, 4] {
                let plan = support::stencil_plan(shape, steps, boundary, device.capabilities());
                let input = support::input(shape);
                let got = device.run_stencil_u32(&plan, &input).unwrap();
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
                    serde_json::json!({"target":got.target,"device":device.name(),"family":"stencil-u32","shape":shape,"boundary":boundary.name(),"steps":steps,"phases":got.phases,"final_buffer":got.final_buffer,"source_hash":got.source_hash,"compile_ms":got.compile_ms,"host_to_host_ms":got.host_to_host_ms,"transfer_bytes":got.transfer_bytes,"completed":true,"reason":if got.phases==0 {"empty-or-zero-step-no-submission"}else{"verified-completed"}})
                );
            }
        }
    }
    println!("Metal f32 stencil unavailable: the M4-01 strict-with-fallback path is not merged");
}
#[test]
#[ignore = "requires a real Apple7 or newer Metal GPU"]
fn issue_m4_02_metal_gpu_failures_do_not_publish() {
    let mut device = MetalDevice::open_runtime_source().unwrap();
    let plan = support::stencil_plan([31, 33], 3, Boundary::Clamp, device.capabilities());
    let input = support::input([31, 33]);
    assert!(
        device
            .run_stencil_u32(&plan, &input[..input.len() - 1])
            .is_err()
    );
    for fault in [Fault::OutOfMemory, Fault::LaunchFailure] {
        assert!(
            device
                .run_stencil_u32_with_fault(&plan, &input, fault)
                .is_err()
        );
        assert!(!device.is_poisoned());
    }
    assert!(matches!(
        device.run_stencil_u32_with_fault(&plan, &input, Fault::PartialCompletion),
        Err(Error::Pending)
    ));
    assert_eq!(device.quarantine_len(), 1);
    // Reap only actual terminal GPU state; no fault flips a fence to complete.
    for _ in 0..100 {
        device.reap();
        if !device.is_poisoned() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(!device.is_poisoned());
    assert_eq!(
        device.run_stencil_u32(&plan, &input).unwrap().output,
        support::stencil_oracle([31, 33], 3, Boundary::Clamp, &input)
    );
}

#[test]
#[ignore = "requires a real Apple7 or newer Metal GPU"]
fn issue_m4_02_metal_gpu_physical_padding() {
    let mut device = MetalDevice::open_runtime_source().unwrap();
    for steps in [2, 3] {
        let plan = support::padded_stencil(steps, device.capabilities());
        let mut input = vec![0x12345678; 3];
        input.extend(support::input([31, 33]));
        input.extend([0x89abcdef; 3]);
        let got = device.run_stencil_u32(&plan, &input).unwrap();
        support::assert_padded_stencil(&got.output, steps, &input);
    }
}

#[test]
#[ignore = "requires real hardware, 2^28 elements; run alone"]
fn issue_m4_02_metal_gpu_cap_2_28_transpose() {
    let mut device =
        MetalDevice::open_runtime_source().expect("explicit hardware gate requires device");
    let shape = [16384, 16384];
    let plan = support::transpose_plan(shape, device.capabilities());
    let input = support::input(shape);
    let got = device.run_transpose(&plan, &input).unwrap();
    support::assert_transpose(shape, &input, &got.output);
    println!(
        "device={} family=transpose elements=268435456 exact_words=268435456 completed=true transfer_bytes={} compile_ms={}",
        device.name(),
        got.transfer_bytes,
        got.compile_ms
    );
}

#[test]
#[ignore = "requires real hardware, 2^28 elements; run alone"]
fn issue_m4_02_metal_gpu_cap_2_28_stencil() {
    let mut device =
        MetalDevice::open_runtime_source().expect("explicit hardware gate requires device");
    let shape = [16384, 16384];
    let input = support::input(shape);
    for boundary in [Boundary::ConstantZero, Boundary::Clamp] {
        let plan = support::stencil_plan(shape, 1, boundary, device.capabilities());
        let got = device.run_stencil_u32(&plan, &input).unwrap();
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
