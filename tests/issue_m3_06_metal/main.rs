use invar::{admission, check, device_ir, front, mir};
use invar_device::{Capabilities, Provenance};

#[cfg(all(feature = "metal", target_os = "macos"))]
static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The program's compiled-ahead Metal container, built as `invar build
/// --target metal` builds it. These suites run the production path.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn prebuilt(source: &[u8]) -> invar::metal_aot::Image {
    let checked = check::check(front::parse(source).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    let kernels = invar::metal_aot::run_kernels(source, &program).unwrap();
    let bytes =
        invar::metal_aot::build::build(invar::metal_aot::Route::Run, "main", source, &kernels)
            .expect("the Metal Toolchain builds the container");
    invar::metal_aot::decode(&bytes).unwrap()
}

/// Calibrate placement with the probe from `image`.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn calibrate(source: &[u8], image: &invar::metal_aot::Image) -> invar::plan::DeviceCostModel {
    use invar::schedule::metal::{Kernels, calibrate, open_device};
    calibrate(&mut || open_device(source, Kernels::Prebuilt(image))).unwrap()
}

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

#[cfg(all(feature = "metal", target_os = "macos"))]
const ZIP_DAG: &str = "fn main() -> i32 { \
    let xs: Array<u32> = [1u32,2u32,3u32]; \
    let ys: Array<u32> = [4u32,5u32,6u32]; \
    let zs: Array<u32> = [7u32,8u32,9u32]; \
    var a: Array<u32> = filled_u32(len(xs),0u32); \
    for i in 0u32 .. len(xs) { a[i] = wrapping_add(xs[i],ys[i]); } \
    let first: Array<u32> = move a; \
    var b: Array<u32> = filled_u32(len(first),0u32); \
    for j in 0u32 .. len(first) { b[j] = wrapping_mul(first[j],zs[j]); } \
    return 0; }";

#[cfg(all(feature = "metal", target_os = "macos"))]
const ZIP_MAP: &str = "fn main() -> i32 { \
    let xs: Array<u32> = [1u32,2u32,3u32]; \
    let zs: Array<u32> = [7u32,8u32,4294967295u32]; \
    var ys: Array<u32> = filled_u32(len(xs),0u32); \
    for i in 0u32 .. len(xs) { ys[i] = wrapping_mul(zs[i],xs[i]); } \
    if ys[2u32] != 4294967293u32 { return 1; } return 0; }";

#[cfg(all(feature = "metal", target_os = "macos"))]
const SUM: &str = "fn main() -> i32 { let xs: Array<u32> = [1u32,2u32,3u32]; var acc: u32 = 7u32; for i in 0u32 .. len(xs) { acc = wrapping_add(acc,xs[i]); } if acc == 13u32 { return 0; } return 1; }";
#[cfg(all(feature = "metal", target_os = "macos"))]
const PRODUCT: &str = "fn main() -> i32 { let xs: Array<u32> = [2u32,3u32,5u32]; var acc: u32 = 3u32; for i in 0u32 .. len(xs) { acc = wrapping_mul(acc,xs[i]); } if acc == 90u32 { return 0; } return 1; }";
#[cfg(all(feature = "metal", target_os = "macos"))]
const INCLUSIVE: &str = "fn main() -> i32 { let aa: Array<u32> = [2u32,3u32,5u32]; let bb: Array<u32> = [7u32,11u32,13u32]; var state: u32 = 17u32; var out: Array<u32> = filled_u32(len(aa),0u32); for i in 0u32 .. len(aa) { state = wrapping_add(wrapping_mul(aa[i],state),bb[i]); out[i] = state; } if out[2u32] == 683u32 { return 0; } return 1; }";
#[cfg(all(feature = "metal", target_os = "macos"))]
const EXCLUSIVE: &str = "fn main() -> i32 { let aa: Array<u32> = [2u32,3u32,5u32]; let bb: Array<u32> = [7u32,11u32,13u32]; var state: u32 = 17u32; var out: Array<u32> = filled_u32(len(aa),0u32); for i in 0u32 .. len(aa) { out[i] = state; state = wrapping_add(wrapping_mul(aa[i],state),bb[i]); } if out[0u32] == 17u32 { return 0; } return 1; }";
#[cfg(all(feature = "metal", target_os = "macos"))]
const FINAL: &str = "fn main() -> i32 { let aa: Array<u32> = [2u32,3u32,5u32]; let bb: Array<u32> = [7u32,11u32,13u32]; var state: u32 = 17u32; for i in 0u32 .. len(aa) { state = wrapping_add(wrapping_mul(aa[i],state),bb[i]); } if state == 683u32 { return 0; } return 1; }";
#[cfg(all(feature = "metal", target_os = "macos"))]
const LARGE_CHAIN: &str = "fn main() -> i32 { \
    let xs: Array<u32> = filled_u32(67108864u32,1u32); \
    var a: Array<u32> = filled_u32(len(xs),0u32); \
    for i in 0u32 .. len(xs) { a[i] = wrapping_add(xs[i],3u32); } \
    let first: Array<u32> = move a; \
    var b: Array<u32> = filled_u32(len(first),0u32); \
    for j in 0u32 .. len(first) { b[j] = wrapping_mul(first[j],7u32); } \
    let second: Array<u32> = move b; \
    var c: Array<u32> = filled_u32(len(second),0u32); \
    for k in 0u32 .. len(second) { c[k] = wrapping_add(second[k],11u32); } return 0; }";
#[cfg(all(feature = "metal", target_os = "macos"))]
const CHAIN: &str = "fn main() -> i32 { \
    let xs: Array<u32> = [1u32,2u32,3u32]; \
    var a: Array<u32> = filled_u32(len(xs),0u32); \
    for i in 0u32 .. len(xs) { a[i] = wrapping_add(xs[i],3u32); } \
    let first: Array<u32> = move a; \
    var b: Array<u32> = filled_u32(len(first),0u32); \
    for j in 0u32 .. len(first) { b[j] = wrapping_mul(first[j],7u32); } \
    let second: Array<u32> = move b; \
    var c: Array<u32> = filled_u32(len(second),0u32); \
    for k in 0u32 .. len(second) { c[k] = wrapping_add(second[k],11u32); } return 0; }";

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

// Hardware suites, split by concern; each test keeps the issue_m3_06_metal_gpu prefix.
#[cfg(all(feature = "metal", target_os = "macos"))]
mod dispatch;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod kernels;
