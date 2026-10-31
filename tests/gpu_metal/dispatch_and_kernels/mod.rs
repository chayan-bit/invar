//! Live MIR dispatch, auto choice, CLI explain, the M2 corpus and device kernels on Metal;
//! the CPU emission check is `tests/issue_m3_06_metal/main.rs`.
//! Issue label M3-06.
use invar::{admission, check, device_ir, front, mir};
use invar_device::Provenance;

static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The program's compiled-ahead Metal container, built as `invar build
/// --target metal` builds it. These suites run the production path.
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
fn calibrate(source: &[u8], image: &invar::metal_aot::Image) -> invar::plan::DeviceCostModel {
    use invar::schedule::metal::{Kernels, calibrate, open_device};
    calibrate(&mut || open_device(source, Kernels::Prebuilt(image))).unwrap()
}

const MAP: &str = "fn main() -> i32 { let xs: Array<u32> = [1u32,2u32,3u32]; \
    var ys: Array<u32> = filled_u32(len(xs),0u32); \
    for i in 0u32 .. len(xs) { ys[i] = wrapping_add(xs[i],5u32); } \
    if ys[2u32] == 8u32 { return 0; } return 1; }";

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

const ZIP_MAP: &str = "fn main() -> i32 { \
    let xs: Array<u32> = [1u32,2u32,3u32]; \
    let zs: Array<u32> = [7u32,8u32,4294967295u32]; \
    var ys: Array<u32> = filled_u32(len(xs),0u32); \
    for i in 0u32 .. len(xs) { ys[i] = wrapping_mul(zs[i],xs[i]); } \
    if ys[2u32] != 4294967293u32 { return 1; } return 0; }";

const SUM: &str = "fn main() -> i32 { let xs: Array<u32> = [1u32,2u32,3u32]; var acc: u32 = 7u32; for i in 0u32 .. len(xs) { acc = wrapping_add(acc,xs[i]); } if acc == 13u32 { return 0; } return 1; }";
const PRODUCT: &str = "fn main() -> i32 { let xs: Array<u32> = [2u32,3u32,5u32]; var acc: u32 = 3u32; for i in 0u32 .. len(xs) { acc = wrapping_mul(acc,xs[i]); } if acc == 90u32 { return 0; } return 1; }";
const INCLUSIVE: &str = "fn main() -> i32 { let aa: Array<u32> = [2u32,3u32,5u32]; let bb: Array<u32> = [7u32,11u32,13u32]; var state: u32 = 17u32; var out: Array<u32> = filled_u32(len(aa),0u32); for i in 0u32 .. len(aa) { state = wrapping_add(wrapping_mul(aa[i],state),bb[i]); out[i] = state; } if out[2u32] == 683u32 { return 0; } return 1; }";
const EXCLUSIVE: &str = "fn main() -> i32 { let aa: Array<u32> = [2u32,3u32,5u32]; let bb: Array<u32> = [7u32,11u32,13u32]; var state: u32 = 17u32; var out: Array<u32> = filled_u32(len(aa),0u32); for i in 0u32 .. len(aa) { out[i] = state; state = wrapping_add(wrapping_mul(aa[i],state),bb[i]); } if out[0u32] == 17u32 { return 0; } return 1; }";
const FINAL: &str = "fn main() -> i32 { let aa: Array<u32> = [2u32,3u32,5u32]; let bb: Array<u32> = [7u32,11u32,13u32]; var state: u32 = 17u32; for i in 0u32 .. len(aa) { state = wrapping_add(wrapping_mul(aa[i],state),bb[i]); } if state == 683u32 { return 0; } return 1; }";
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

mod dispatch;
mod kernels;
