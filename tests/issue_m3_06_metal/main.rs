use invar::{admission, check, device_ir, front, mir};
use invar_device::{Capabilities, Provenance};

#[cfg(all(feature = "metal", not(target_os = "macos")))]
#[test]
fn issue_m3_06_metal_missing_device_is_an_actionable_error() {
    assert!(
        matches!(invar_metal::MetalDevice::open(), Err(invar_metal::Error::Unavailable(message)) if message.contains("Apple Silicon Mac"))
    );
}

const MAP: &str = "fn main() -> i32 { let xs: Array<u32> = [1u32,2u32,3u32]; \
    var ys: Array<u32> = filled_u32(len(xs),0u32); \
    for i in 0u32 .. len(xs) { ys[i] = wrapping_add(xs[i],5u32); } \
    if ys[2u32] == 8u32 { return 0; } return 1; }";

#[test]
fn issue_m3_06_metal_shared_verified_map_emits_deterministic_msl() {
    let checked = check::check(front::parse(MAP.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    let report = admission::analyze(MAP.as_bytes(), &program).unwrap();
    let ir = device_ir::from_region(&report.regions()[0], 3).unwrap();
    let origin = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    let verified = invar_device::verify(
        ir,
        origin,
        Capabilities::Metal {
            apple_family: 7,
            msl_major: 3,
            msl_minor: 1,
        },
    )
    .unwrap();
    let first = invar_device::emit_msl(&verified).unwrap();
    let second = invar_device::emit_msl(&verified).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        first,
        concat!(
            "#include <metal_stdlib>\nusing namespace metal;\n",
            "kernel void invar_map_v1(device const uint* input0 [[buffer(0)]], device uint* output0 [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {\n",
            "  if (i < n) {\n    uint v = input0[i];\n    v = v + 5u;\n    output0[i] = v;\n  }\n}\n",
        )
    );
    assert!(first.contains("kernel void invar_map_v1"));
    assert!(!first.contains("xs"));
    let mut alias = verified.ir().clone();
    alias.output.id = alias.input.id;
    assert_eq!(
        invar_device::verify(alias, origin, verified.capabilities()).err(),
        Some(invar_device::Error::Alias)
    );
    let cuda = invar_device::verify(
        verified.ir().clone(),
        origin,
        Capabilities::cuda_12_9_sm89(),
    )
    .unwrap();
    assert_eq!(
        invar_device::emit_msl(&cuda).err(),
        Some(invar_device::Error::Toolchain)
    );
}
