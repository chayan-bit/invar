//! Shared map/reduction/scan admission and opt-in, sequential 1 GiB device cases.
use invar::{admission, check::check, device_ir, front::parse, mir};
use invar_device::{Capabilities, Error, Provenance};

const N: u32 = 1 << 28;
const MAP: &str = "fn main() -> i32 { let xs: Array<u32> = [1u32]; var out: Array<u32> = filled_u32(len(xs),0u32); for i in 0u32 .. len(xs) { out[i] = wrapping_add(xs[i],11u32); } return 0; }";
const CHAIN: &str = "fn main() -> i32 { let xs: Array<u32> = [1u32]; var out: Array<u32> = filled_u32(len(xs),0u32); for i in 0u32 .. len(xs) { out[i] = wrapping_add(xs[i],3u32); } let first: Array<u32> = move out; var second: Array<u32> = filled_u32(len(first),0u32); for j in 0u32 .. len(first) { second[j] = wrapping_mul(first[j],7u32); } return 0; }";
const SUM: &str = "fn main() -> i32 { let xs: Array<u32> = [1u32]; var state: u32 = 17u32; for i in 0u32 .. len(xs) { state = wrapping_add(state,xs[i]); } return 0; }";
const SCAN: &str = "fn main() -> i32 { let xs: Array<u32> = [1u32]; var state: u32 = 17u32; var out: Array<u32> = filled_u32(len(xs),0u32); for i in 0u32 .. len(xs) { state = wrapping_add(wrapping_mul(3u32,state),xs[i]); out[i] = state; } return 0; }";
const TWO_INPUT_SCAN: &str = "fn main() -> i32 { let aa: Array<u32> = [2u32]; let bb: Array<u32> = [7u32]; var state: u32 = 17u32; var out: Array<u32> = filled_u32(len(aa),0u32); for i in 0u32 .. len(aa) { state = wrapping_add(wrapping_mul(aa[i],state),bb[i]); out[i] = state; } return 0; }";

fn plan(source: &str, n: u32, caps: Capabilities) -> invar_device::Verified {
    let program = mir::lower(&check(parse(source.as_bytes()).unwrap(), true).unwrap()).unwrap();
    let report = admission::analyze(source.as_bytes(), &program).unwrap();
    let mut ir = device_ir::from_region(&report.regions()[0], n)
        .unwrap_or_else(|error| panic!("{error:?}: {:?}", report.regions()[0].execution()));
    if matches!(caps, Capabilities::Metal { .. }) {
        device_ir::plan_metal(&mut ir).unwrap();
    }
    let origin = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    invar_device::verify(ir, origin, caps).unwrap()
}

#[test]
fn shared_device_cap_cpu_boundary_and_mutations() {
    for caps in [
        Capabilities::cuda_12_9_sm89(),
        Capabilities::Metal {
            apple_family: 9,
            msl_major: 3,
            msl_minor: 1,
        },
    ] {
        // The manifest's largest four-input/four-output map needs exactly 8 GiB.
        let mut maximum = plan(MAP, N, caps).ir().clone();
        let span = maximum.span;
        let buffer = |id| invar_device::Buffer {
            id,
            ..maximum.input
        };
        let dag = maximum.dag.as_mut().unwrap();
        dag.inputs = (0..4).map(buffer).collect();
        dag.outputs = (4..8).map(buffer).collect();
        dag.nodes = (0..4)
            .map(|i| invar_device::DagNode {
                op: invar_device::DagOp::Input(i),
                span,
            })
            .collect();
        dag.stage_outputs = (0..4).collect();
        maximum.output = dag.outputs[3];
        let origin = Provenance {
            source_identity: maximum.source_identity,
            graph_revision: maximum.graph_revision,
        };
        assert!(invar_device::verify(maximum.clone(), origin, caps).is_ok());
        maximum.dag.as_mut().unwrap().inputs.push(buffer(8));
        assert!(matches!(
            invar_device::verify(maximum, origin, caps),
            Err(Error::ResourceLimit("map exceeds 8 GiB transfer budget"))
        ));
        for source in [MAP, CHAIN, SUM, SCAN, TWO_INPUT_SCAN] {
            let verified = plan(source, N, caps);
            let ir = verified.ir();
            if ir.collective.is_some() {
                assert_eq!(ir.geometry.blocks, 65_536);
            } else {
                assert!(ir.dag.is_some(), "large standalone maps use the DAG path");
            }
            let program =
                mir::lower(&check(parse(source.as_bytes()).unwrap(), true).unwrap()).unwrap();
            let report = admission::analyze(source.as_bytes(), &program).unwrap();
            assert_eq!(
                device_ir::from_region(&report.regions()[0], N + 1).unwrap_err(),
                device_ir::Error::Length
            );
            let origin = Provenance {
                source_identity: ir.source_identity,
                graph_revision: ir.graph_revision,
            };
            let mut over = ir.clone();
            over.length = N + 1;
            assert!(
                matches!(invar_device::verify(over, origin, caps), Err(Error::ResourceLimit(reason)) if !reason.is_empty())
            );
            if ir.collective.is_some() {
                let mut padded = ir.clone();
                padded.collective.as_mut().unwrap().output.elements += 1;
                padded.output.elements += 1;
                assert!(matches!(
                    invar_device::verify(padded, origin, caps),
                    Err(Error::Bounds)
                ));
                let mut geometry = ir.clone();
                geometry.geometry.blocks += 1;
                assert!(invar_device::verify(geometry, origin, caps).is_err());
            }
            let mut short = ir.clone();
            if let Some(collective) = &mut short.collective {
                collective.inputs[0].elements = N - 1;
                short.input = collective.inputs[0];
            } else {
                short.dag.as_mut().unwrap().inputs[0].elements = N - 1;
                short.input.elements = N - 1;
            }
            assert!(matches!(
                invar_device::verify(short, origin, caps),
                Err(Error::Bounds)
            ));
            let emitted = if matches!(caps, Capabilities::Metal { .. }) {
                invar_device::emit_msl(&verified)
            } else {
                invar_device::emit_cuda(&verified)
            };
            assert!(!emitted.unwrap().is_empty());
        }
    }
    // Worst valid collective: two 1 GiB inputs, n+1 output, two 512 KiB
    // pair summaries. Arithmetic is u64/checked usize, never u32 bytes.
    let bytes = u64::from(N) * 4;
    assert_eq!(bytes, 1_073_741_824);
    assert_eq!(N.div_ceil(4096), 65_536);
    let total = bytes * 3 + 4 + 65_536 * 16;
    assert!(invar_device::check_memory(bytes + 4, total, bytes + 4, total).is_ok());
    assert!(invar_device::check_memory(bytes + 4, total, bytes, total).is_err());
    assert!(invar_device::check_memory(bytes + 4, total, bytes + 4, total - 1).is_err());
    assert!(invar_device::check_memory(0, u64::MAX, u64::MAX, 0).is_err());
}

#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "cuda"))]
fn input() -> Vec<u32> {
    (0..N).map(|i| i.wrapping_mul(17).wrapping_add(3)).collect()
}

#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "cuda"))]
fn check_scan(values: &[u32], prefixes: &[u32], final_state: u32) {
    assert_eq!(prefixes.len(), N as usize);
    let mut state = 17u32;
    for (i, (&value, &observed)) in values.iter().zip(prefixes).enumerate() {
        state = state.wrapping_mul(3).wrapping_add(value);
        assert_eq!(observed, state, "scan index={i}");
    }
    assert_eq!(final_state, state);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "requires Apple7+ Metal and sequential multi-GiB capacity"]
fn shared_device_cap_metal_2_28_bit_exact() {
    let mut device = invar_metal::MetalDevice::open_runtime_source().unwrap();
    eprintln!(
        "device={} caps={:?} n={N}",
        device.name(),
        device.capabilities()
    );
    let values = input();
    {
        let verified = plan(MAP, N, device.capabilities());
        let result = device.run_fused_map(&verified, &[&values]).unwrap();
        assert_eq!(result.outputs.len(), 1);
        for (&x, &y) in values.iter().zip(&result.outputs[0]) {
            assert_eq!(y, x.wrapping_add(11));
        }
        assert_eq!(result.outputs[0].len(), N as usize);
        eprintln!("map bit-exact n={N}");
    }
    {
        let verified = plan(SUM, N, device.capabilities());
        let result = device.run_collective(&verified, &[&values], 17).unwrap();
        assert_eq!(
            result.final_state,
            values.iter().fold(17u32, |s, x| s.wrapping_add(*x))
        );
        eprintln!("reduction bit-exact n={N}");
    }
    {
        let verified = plan(SCAN, N, device.capabilities());
        let result = device.run_collective(&verified, &[&values], 17).unwrap();
        check_scan(&values, &result.prefixes, result.final_state);
        eprintln!("scan bit-exact n={N}");
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA device and INVAR_NVRTC_COMPILER; sequential multi-GiB capacity"]
fn shared_device_cap_cuda_2_28_bit_exact() {
    use invar_cuda::cuda::CudaDevice;
    let compiler = std::path::PathBuf::from(
        std::env::var_os("INVAR_NVRTC_COMPILER").expect("INVAR_NVRTC_COMPILER"),
    );
    let mut device = CudaDevice::open().unwrap();
    let values = input();
    {
        let result = device
            .run_fused_map(&plan(MAP, N, device.capabilities()), &[&values], &compiler)
            .unwrap();
        assert_eq!(result.outputs[0].len(), N as usize);
        for (&x, &y) in values.iter().zip(&result.outputs[0]) {
            assert_eq!(y, x.wrapping_add(11));
        }
    }
    {
        let result = device
            .run_collective(
                &plan(SUM, N, device.capabilities()),
                &[&values],
                17,
                &compiler,
            )
            .unwrap();
        assert_eq!(
            result.final_state,
            values.iter().fold(17u32, |s, x| s.wrapping_add(*x))
        );
    }
    {
        let result = device
            .run_collective(
                &plan(SCAN, N, device.capabilities()),
                &[&values],
                17,
                &compiler,
            )
            .unwrap();
        check_scan(&values, &result.prefixes, result.final_state);
    }
}
