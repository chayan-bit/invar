use invar::{admission, check::check, device_ir, front::parse, mir, plan::Execution};
use invar_device::{
    AddressSpace, Barrier, Capabilities, Error as VerifyError, Geometry, Ir, LaneMask, Layout,
    Numeric, Op, Provenance, Transfer,
};

const SOURCE: &str = "fn main() -> i32 { let xs: Array<u32> = [1u32, 2u32, 3u32]; var ys: Array<u32> = filled_u32(len(xs), 0u32); for i in 0u32 .. len(xs) { ys[i] = wrapping_add(xs[i], 7u32); } return 0; }";

fn observed_source() -> String {
    SOURCE.replace("return 0;", "if ys[0u32] != 8u32 { return 1; } if ys[1u32] != 9u32 { return 1; } if ys[2u32] != 10u32 { return 1; } return 0;")
}

fn admitted() -> Ir {
    let checked = check(parse(SOURCE.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    let report = admission::analyze(SOURCE.as_bytes(), &program).unwrap();
    let region = &report.regions()[0];
    assert!(matches!(region.execution(), Execution::Map(_)));
    device_ir::from_region(region, 3).unwrap()
}

fn provenance(ir: &Ir) -> Provenance {
    Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    }
}

#[test]
fn issue_m3_02_device_ir_verifier_admitted_graph_map_is_deterministic() {
    let ir = admitted();
    eprintln!("device_ir={ir:?}");
    let origin = provenance(&ir);
    let verified = invar_device::verify(ir, origin, Capabilities::cuda_12_9_sm89()).unwrap();
    let source = invar_device::emit_cuda(&verified).unwrap();
    assert_eq!(source, invar_device::emit_cuda(&verified).unwrap());
    assert!(source.contains("extern \"C\" __global__ void invar_map_v1"));
    assert!(source.contains("+ 7u"));
    assert_eq!(invar_device::source_hash(&source), 0xdbbc_b16d_ac2c_5e72);
}

#[test]
fn issue_m3_02_device_ir_verifier_reference_and_cpu_agree() {
    let source = observed_source();
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    assert_eq!(
        invar::eval::observe(&checked, 100_000).result,
        Ok(invar::eval::Value::I32(0))
    );
    let program = mir::lower(&checked).unwrap();
    assert_eq!(
        invar::schedule::run_source(source.as_bytes(), &program, 100_000, 4)
            .unwrap()
            .observation
            .result,
        Ok(invar::eval::Value::I32(0))
    );
}

type Mutation = (&'static str, VerifyError, Box<dyn Fn(&mut Ir)>);

fn malformed_cases() -> Vec<Mutation> {
    vec![
        (
            "overread",
            VerifyError::Bounds,
            Box::new(|ir| ir.input.elements = 2),
        ),
        (
            "alias",
            VerifyError::Alias,
            Box::new(|ir| ir.output.id = ir.input.id),
        ),
        (
            "stale version",
            VerifyError::Version,
            Box::new(|ir| ir.version = 0),
        ),
        (
            "stale graph",
            VerifyError::Graph,
            Box::new(|ir| ir.graph_revision += 1),
        ),
        (
            "geometry",
            VerifyError::Geometry,
            Box::new(|ir| {
                ir.geometry = Geometry {
                    blocks: 0,
                    lanes: 256,
                }
            }),
        ),
        (
            "arithmetic",
            VerifyError::Operations,
            Box::new(|ir| ir.ops = vec![Op::CheckedAdd(1)]),
        ),
        (
            "missing barrier",
            VerifyError::Barrier,
            Box::new(|ir| {
                ir.requires_barrier = true;
                ir.barrier = Barrier::None;
            }),
        ),
        (
            "nonuniform barrier",
            VerifyError::Barrier,
            Box::new(|ir| {
                ir.requires_barrier = true;
                ir.barrier = Barrier::Conditional;
            }),
        ),
        (
            "lane mask",
            VerifyError::Mask,
            Box::new(|ir| ir.lane_mask = LaneMask::All),
        ),
        (
            "completion",
            VerifyError::Completion,
            Box::new(|ir| ir.completion = false),
        ),
        (
            "layout",
            VerifyError::Layout,
            Box::new(|ir| ir.input.layout = Layout::Unknown),
        ),
        (
            "address",
            VerifyError::Layout,
            Box::new(|ir| ir.input.address = AddressSpace::Host),
        ),
        (
            "transfer",
            VerifyError::Transfer,
            Box::new(|ir| ir.transfer = Transfer::Unknown),
        ),
        (
            "numeric",
            VerifyError::Numeric,
            Box::new(|ir| ir.numeric = Numeric::CheckedU32),
        ),
    ]
}

#[test]
fn issue_m3_02_device_ir_verifier_malformed_matrix() {
    let base = admitted();
    for (label, expected, mutation) in malformed_cases() {
        let mut ir = base.clone();
        mutation(&mut ir);
        let actual =
            invar_device::verify(ir, provenance(&base), Capabilities::cuda_12_9_sm89()).err();
        assert_eq!(actual, Some(expected), "{label}");
        eprintln!("verifier_reject={label}: {expected:?}");
    }
}

#[cfg(feature = "cuda")]
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

#[test]
fn issue_m3_02_device_ir_verifier_boundaries_and_no_device_fallback() {
    let mut ir = admitted();
    ir.length = 0;
    ir.geometry.blocks = 1;
    assert!(
        invar_device::verify(ir.clone(), provenance(&ir), Capabilities::cuda_12_9_sm89()).is_ok()
    );
    let mut ir = admitted();
    ir.length = 1_000_001;
    assert!(
        invar_device::verify(ir.clone(), provenance(&ir), Capabilities::cuda_12_9_sm89()).is_err()
    );
    let ir = admitted();
    assert!(
        invar_device::verify(
            ir.clone(),
            provenance(&ir),
            Capabilities::Cuda {
                major: 12,
                minor: 4,
                sm: 86,
            }
        )
        .is_ok()
    );
    assert!(
        invar_device::verify(
            ir.clone(),
            provenance(&ir),
            Capabilities::Cuda {
                major: 12,
                minor: 3,
                sm: 89
            }
        )
        .is_err()
    );
    assert!(
        invar_device::verify(
            ir.clone(),
            provenance(&ir),
            Capabilities::Cuda {
                major: 12,
                minor: 4,
                sm: 49,
            }
        )
        .is_err()
    );
}
