//! Live CUDA generated map and verifier rejections; the CPU verifier tests stay in `tests/issue_m3_02_device_ir_verifier.rs`.
use invar::{check::check, front::parse, mir};
use invar_device::Capabilities;

#[path = "../support/m3_02_cases.rs"]
mod m3_02_cases;
use m3_02_cases::{admitted, malformed_cases, observed_source, provenance};

#[test]
#[ignore = "requires a CUDA GPU and NVRTC 12.4+"]
fn issue_m3_02_gpu_generated_map_and_rejections() {
    use invar_cuda::{
        Error as CudaError,
        cuda::{CudaDevice, compile_cubin_bounded},
    };
    let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    let mut device =
        CudaDevice::open().expect("CUDA_UNAVAILABLE: requires CUDA GPU and NVRTC 12.4+");

    let base = admitted();
    for (label, expected, mutation) in malformed_cases() {
        let mut ir = base.clone();
        mutation(&mut ir);
        assert_eq!(
            invar_device::verify(ir, provenance(&base), device.capabilities()).err(),
            Some(expected),
            "{label} reached code generation"
        );
    }
    match compile_cubin_bounded(
        "extern \"C\" __global__ void broken(",
        compiler,
        device.capabilities(),
    ) {
        Err(error @ CudaError::Compile(_)) => eprintln!("compile_failure={error:?}"),
        _ => panic!("invalid CUDA source must return Compile"),
    }

    let verified =
        invar_device::verify(base.clone(), provenance(&base), device.capabilities()).unwrap();
    let Capabilities::Cuda { major, minor, sm } = device.capabilities() else {
        unreachable!()
    };
    let wrong_target = invar_device::verify(
        base.clone(),
        provenance(&base),
        Capabilities::Cuda {
            major,
            minor,
            sm: sm + 1,
        },
    )
    .unwrap();
    assert_eq!(
        device.run_map(&wrong_target, &[1, 2, 3], compiler).err(),
        Some(CudaError::Limit)
    );
    let result = device.run_map(&verified, &[1, 2, 3], compiler).unwrap();
    assert_eq!(result.values, [8, 9, 10]);
    eprintln!("completion=event-confirmed output={:?}", result.values);
    assert!(result.cubin_bytes > 0);
    assert_eq!(result.transfer_bytes, 24);
    assert!(result.compile_ms.is_finite() && result.compile_ms >= 0.0);
    assert!(result.h2d_ms.is_finite() && result.h2d_ms >= 0.0);
    assert!(result.kernel_ms.is_finite() && result.kernel_ms >= 0.0);
    assert!(result.d2h_ms.is_finite() && result.d2h_ms >= 0.0);
    eprintln!(
        "device={} caps={:?} source_hash={:016x} cubin_bytes={} geometry={}x{} transfer_bytes={} compile_ms={} h2d_ms={} kernel_ms={} d2h_ms={} residency=host-to-host nvrtc_options=--gpu-architecture=sm_{},--std=c++17 nvrtc_log={:?}\nsource:\n{}",
        device.name(),
        device.capabilities(),
        result.source_hash,
        result.cubin_bytes,
        result.blocks,
        result.lanes,
        result.transfer_bytes,
        result.compile_ms,
        result.h2d_ms,
        result.kernel_ms,
        result.d2h_ms,
        match device.capabilities() {
            Capabilities::Cuda { sm, .. } => sm,
            _ => unreachable!(),
        },
        result.nvrtc_log,
        invar_device::emit_cuda(&verified).unwrap()
    );

    let observed_source = observed_source();
    let checked = check(parse(observed_source.as_bytes()).unwrap(), true).unwrap();
    assert_eq!(
        invar::eval::observe(&checked, 100_000).result,
        Ok(invar::eval::Value::I32(0))
    );
    let program = mir::lower(&checked).unwrap();
    assert_eq!(
        invar::schedule::run_source(observed_source.as_bytes(), &program, 100_000, 4)
            .unwrap()
            .observation
            .result,
        Ok(invar::eval::Value::I32(0))
    );
}
