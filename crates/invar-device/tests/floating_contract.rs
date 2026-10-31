//! Issue #1 (M4-01).
#![forbid(unsafe_code)]
#[allow(dead_code)]
#[path = "../../../test-support/child_deadline.rs"]
mod child_deadline;
// Exercise the production module and its shared IR v2 verification.
use floating::{Buffer, Error, Ir, Verified, emit_cpu, emit_cuda, emit_msl, verify};
pub use invar_device::{Capabilities, MAX_SOURCE_BYTES, Provenance, Span, floating, source_hash};
use std::{
    fmt::Write,
    fs,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

fn plan(dimensions: [u32; 3]) -> Ir {
    let [rows, inner, cols] = dimensions;
    let make = |id, shape: [u32; 2]| Buffer {
        id,
        elements: u64::from(shape[0]) * u64::from(shape[1]) + 3,
        shape,
        strides: [u64::from(shape[1]), 1],
        offset: 1,
        alignment: 16,
    };
    Ir {
        version: 1,
        numeric_version: 1,
        numeric_policy: "strict-f32-v1".into(),
        layout_version: 1,
        span: Span { start: 7, end: 93 },
        source_identity: 77,
        graph_revision: 3,
        function_contract: [42; 32],
        dimensions,
        inputs: [make(1, [rows, inner]), make(2, [inner, cols])],
        output: make(3, [rows, cols]),
    }
}
fn origin() -> Provenance {
    Provenance {
        source_identity: 77,
        graph_revision: 3,
    }
}
fn admit(ir: Ir) -> Verified {
    verify(ir, Capabilities::cuda_12_9_sm89(), origin()).unwrap()
}
fn refused(ir: Ir, error: Error) {
    assert_eq!(
        verify(ir, Capabilities::cuda_12_9_sm89(), origin()).unwrap_err(),
        error
    );
}

#[test]
fn floating_versions_provenance_and_policy_mutations() {
    for (change, error) in [
        (0, Error::Version),
        (1, Error::Numeric),
        (2, Error::Numeric),
        (3, Error::Layout),
        (4, Error::Source),
        (5, Error::Graph),
        (6, Error::Graph),
        (7, Error::Contract),
    ] {
        let mut ir = plan([3, 5, 7]);
        match change {
            0 => ir.version = 2,
            1 => ir.numeric_version = 2,
            2 => ir.numeric_policy = "fast-f32".into(),
            3 => ir.layout_version = 2,
            4 => ir.span.end = ir.span.start,
            5 => ir.source_identity += 1,
            6 => ir.graph_revision += 1,
            7 => ir.function_contract = [0; 32],
            _ => unreachable!(),
        }
        refused(ir, error);
    }
    let mut ir = plan([3, 5, 7]);
    ir.source_identity = 0;
    refused(ir, Error::Source);
}

#[test]
fn floating_shapes_bounds_strides_alignment_and_alias_mutations() {
    let mut ir = plan([3, 5, 7]);
    ir.dimensions[0] += 1;
    refused(ir, Error::Shape);
    for slot in 0..3 {
        for change in 0..6 {
            let mut ir = plan([3, 5, 7]);
            let b = if slot == 2 {
                &mut ir.output
            } else {
                &mut ir.inputs[slot]
            };
            let error = match change {
                0 => {
                    b.strides[0] += 1;
                    Error::Layout
                }
                1 => {
                    b.strides[1] = 2;
                    Error::Layout
                }
                2 => {
                    b.alignment = 3;
                    Error::Alignment
                }
                3 => {
                    b.alignment = 2;
                    Error::Alignment
                }
                4 => {
                    b.elements = b.offset;
                    Error::Bounds
                }
                5 => {
                    b.offset = u64::MAX;
                    Error::Bounds
                }
                _ => unreachable!(),
            };
            refused(ir, error);
        }
    }
    for id in [1, 2] {
        let mut ir = plan([3, 5, 7]);
        ir.output.id = id;
        refused(ir, Error::Alias);
    }
    let mut ir = plan([2, 2, 2]);
    ir.inputs[1].id = 1;
    admit(ir.clone());
    ir.inputs[1].elements += 1;
    refused(ir, Error::Alias);
    let mut ir = plan([1, 1, 1]);
    ir.output.elements = floating::MAX_ELEMENTS + 1;
    refused(ir, Error::Bounds);
    let mut ir = plan([0, 0, 0]);
    ir.output.offset = ir.output.elements + 1;
    refused(ir, Error::Bounds);
    let mut ir = plan([0, u32::MAX, 0]);
    ir.inputs[0].strides = [u64::from(u32::MAX), 1];
    refused(ir, Error::Bounds);
    let ir = plan([1, 1, (1 << 28) + 1]);
    refused(ir, Error::MatrixLimit);
}

#[test]
fn floating_zero_one_odd_domains_and_physical_mapping() {
    for dimensions in [
        [0, 5, 3],
        [3, 0, 5],
        [3, 5, 0],
        [0, 0, 0],
        [1, 1, 1],
        [3, 5, 7],
    ] {
        let p = admit(plan(dimensions));
        for slot in 0..3 {
            let b = if slot == 2 {
                &p.ir().output
            } else {
                &p.ir().inputs[slot]
            };
            assert_eq!(p.physical_index(slot, b.shape), Err(Error::Index));
            if b.shape.contains(&0) {
                assert_eq!(p.physical_index(slot, [0, 0]), Err(Error::Index));
            } else {
                let at = [b.shape[0] - 1, b.shape[1] - 1];
                assert_eq!(
                    p.physical_index(slot, at).unwrap(),
                    b.offset + u64::from(at[0]) * b.strides[0] + u64::from(at[1])
                );
            }
        }
        assert_eq!(p.physical_index(3, [0, 0]), Err(Error::Index));
    }
    let mut ir = plan([0, 1, 0]);
    for b in ir.inputs.iter_mut().chain(std::iter::once(&mut ir.output)) {
        b.offset = b.elements;
    }
    admit(ir);
}

#[test]
fn floating_capabilities_and_exact_emitter_option_identity() {
    for caps in [
        Capabilities::Metal {
            apple_family: 6,
            msl_major: 3,
            msl_minor: 1,
        },
        Capabilities::Metal {
            apple_family: 9,
            msl_major: 3,
            msl_minor: 0,
        },
        Capabilities::Metal {
            apple_family: 11,
            msl_major: 3,
            msl_minor: 1,
        },
        // Blackwell needs the NVRTC release that introduced it.
        Capabilities::Cuda {
            major: 12,
            minor: 7,
            sm: 120,
        },
        Capabilities::Cuda {
            major: 12,
            minor: 8,
            sm: 121,
        },
        Capabilities::Cuda {
            major: 11,
            minor: 9,
            sm: 89,
        },
        Capabilities::Cuda {
            major: 12,
            minor: 3,
            sm: 89,
        },
        Capabilities::Cuda {
            major: 12,
            minor: 9,
            sm: 999,
        },
        Capabilities::Cuda {
            major: 13,
            minor: 0,
            sm: 89,
        },
    ] {
        assert_eq!(
            verify(plan([1, 1, 1]), caps, origin()).unwrap_err(),
            Error::Capabilities
        );
    }
    let p = admit(plan([3, 5, 7]));
    let cuda = emit_cuda(&p).unwrap();
    assert_eq!(p.capabilities(), Capabilities::cuda_12_9_sm89());
    assert!(cuda.contains(floating::KERNEL_NAME));
    assert!(cuda.contains("__fmul_rn(A[ai], B[bi])"));
    assert!(cuda.contains("__fadd_rn(sum, product)"));
    for option in p.nvrtc_options() {
        assert!(cuda.contains(option));
    }
    assert!(!cuda.contains("__fmaf"));
    assert!(!cuda.contains("--use_fast_math"));
    assert!(cuda.contains("unsigned long long"));
    let cpu = emit_cpu(&p).unwrap();
    for option in p.cpu_options() {
        assert!(cpu.contains(option));
    }
    let original = p.cuda_source_hash().unwrap();
    for change in 0..11 {
        let mut ir = plan([3, 5, 7]);
        match change {
            0 => ir.span.end += 1,
            1 => ir.function_contract[0] ^= 1,
            2 => ir.inputs[0].id += 10,
            3 => ir.inputs[0].alignment = 32,
            4 => ir.inputs[0].elements += 1,
            5 => ir.inputs[0].offset += 1,
            6 => ir.output.id += 1,
            7 => ir.output.alignment = 32,
            8 => ir.output.elements += 1,
            9 => ir.output.offset += 1,
            10 => ir = plan([3, 5, 8]),
            _ => unreachable!(),
        }
        assert_ne!(admit(ir).cuda_source_hash().unwrap(), original);
    }
    for (minor, sm) in [(8, 100), (8, 120), (9, 103), (9, 121)] {
        let caps = Capabilities::Cuda {
            major: 12,
            minor,
            sm,
        };
        let blackwell = verify(plan([3, 5, 7]), caps, origin()).unwrap();
        assert!(
            blackwell
                .nvrtc_options()
                .contains(&format!("--gpu-architecture=compute_{sm}"))
        );
        assert_ne!(blackwell.cuda_source_hash().unwrap(), original);
    }
    let caps = Capabilities::Cuda {
        major: 12,
        minor: 4,
        sm: 86,
    };
    let different = verify(plan([3, 5, 7]), caps, origin()).unwrap();
    assert_ne!(different.cuda_source_hash().unwrap(), original);
    assert!(
        different
            .nvrtc_options()
            .contains(&"--gpu-architecture=compute_86".to_owned())
    );
}

#[test]
fn floating_metal_admission_emits_strict_msl_from_the_same_plan() {
    let metal = Capabilities::Metal {
        apple_family: 9,
        msl_major: 3,
        msl_minor: 1,
    };
    let p = verify(plan([3, 5, 7]), metal, origin()).unwrap();
    assert!(p.nvrtc_options().is_empty());
    assert_eq!(emit_cuda(&p).unwrap_err(), Error::Capabilities);
    let cuda_plan = admit(plan([3, 5, 7]));
    assert_eq!(emit_msl(&cuda_plan).unwrap_err(), Error::Capabilities);
    let msl = emit_msl(&p).unwrap();
    assert_eq!(msl, emit_msl(&p).unwrap(), "deterministic emission");
    for option in p.msl_options() {
        assert!(msl.contains(option), "{option}");
    }
    assert!(msl.contains("#pragma METAL fp contract(off)\nusing namespace metal;"));
    assert!(msl.contains("kernel void invar_f32_contract_v1("));
    assert!(msl.contains("as_type<float>(aw) * as_type<float>(bw)"));
    assert!(msl.contains("const float next = sum + product;"));
    assert!(msl.contains("C[ci] = as_type<uint>(sum);"));
    // Every multiply and add feeds the dispatch underflow guard.
    assert!(msl.contains(&floating::guard_helpers(false)));
    assert!(msl.contains("device atomic_uint* F [[buffer(3)]]"));
    assert!(msl.contains("flag |= invar_operand(aw) | invar_operand(bw) | invar_mul(aw, bw, as_type<uint>(product)) | invar_add(as_type<uint>(sum), as_type<uint>(product), as_type<uint>(next));"));
    assert!(msl.contains("if (flag != 0u) atomic_fetch_or_explicit(F, 1u, memory_order_relaxed);"));
    // The dispatch flag is claimed by every lane before any other statement.
    assert!(msl.contains(&format!(
        "threads [[threads_per_grid]]) {{\n{}",
        floating::guard::msl_claim(floating::guard::GUARD)
    )));
    assert!(floating::guard_helpers(true).starts_with("__device__ static inline unsigned"));
    use floating::guard;
    assert!(guard::operand(1) && !guard::operand(0) && !guard::operand(0x0080_0000));
    assert!(guard::mul(0x0080_0000, 0x3f7f_ffff, 0) && guard::mul(1, 1, 0x0080_0000));
    assert!(!guard::mul(0, 0x0000_0001, 0) && !guard::mul(0x3f80_0000, 0x3f80_0000, 0x3f80_0000));
    assert!(guard::add(0x0080_0001, 0x8080_0000, 0) && guard::add(0, 0x0080_0000, 0x0080_0000));
    assert!(!guard::add(0x3f80_0000, 0xbf80_0000, 0) && !guard::add(0, 0x8000_0000, 0));
    assert!(!msl.contains("fma") && !msl.contains("long long"));
    // The emitted identity carries the plan exactly as CUDA's does.
    let cuda = emit_cuda(&cuda_plan).unwrap();
    let identity = |text: &str| {
        text.lines()
            .filter(|l| l.starts_with("// ") && !l.starts_with("// required-options"))
            .filter(|l| !l.contains("caps="))
            .map(str::to_owned)
            .collect::<Vec<_>>()
    };
    assert_eq!(identity(&msl), identity(&cuda));
    let original = p.msl_source_hash().unwrap();
    for change in 0..4 {
        let mut ir = plan([3, 5, 7]);
        match change {
            0 => ir.output.offset += 1,
            1 => ir.inputs[1].elements += 1,
            2 => ir.function_contract[3] ^= 1,
            3 => ir = plan([3, 5, 8]),
            _ => unreachable!(),
        }
        let changed = verify(ir, metal, origin()).unwrap();
        assert_ne!(changed.msl_source_hash().unwrap(), original);
    }
    let other = Capabilities::Metal {
        apple_family: 7,
        msl_major: 3,
        msl_minor: 1,
    };
    let other = verify(plan([3, 5, 7]), other, origin()).unwrap();
    assert_ne!(other.msl_source_hash().unwrap(), original);
    // Empty output domains emit no element access.
    let empty = verify(plan([0, 4, 3]), metal, origin()).unwrap();
    let text = emit_msl(&empty).unwrap();
    assert!(!text.contains("A[") && !text.contains("C["));
}

struct Temp(std::path::PathBuf);
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn cpu_run(p: &Verified, harness: &str, optimization: &str) -> String {
    static ID: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "invar-f32-device-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&dir).unwrap();
    let temp = Temp(dir);
    let source = format!("{}\n{harness}", emit_cpu(p).unwrap());
    fs::write(temp.0.join("contract.cpp"), &source).unwrap();
    let flags = [
        optimization,
        "-fsanitize=undefined",
        "-fno-sanitize-recover=all",
        "-Wall",
        "-Wextra",
        "-Werror",
    ];
    let flags = [p.cpu_options(), &flags].concat();
    child_deadline::cxx(
        &flags,
        &temp.0.join("contract.cpp"),
        &temp.0.join("contract"),
    );
    let run = Command::new(temp.0.join("contract")).output().unwrap();
    assert!(
        run.status.success(),
        "{}\n{}\n{source}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8(run.stdout).unwrap()
}
fn dot_harness(a: &[u32], b: &[u32], expected: u32) -> String {
    let list = |v: &[u32]| {
        v.iter()
            .map(|n| format!("0x{n:08x}u"))
            .collect::<Vec<_>>()
            .join(",")
    };
    format!(
        "#include <cstdio>\nint main() {{\n const std::uint32_t ab[]={{ {} }}, bb[]={{ {} }}; float A[{}],B[{}],C[4];\n std::memcpy(A,ab,sizeof(ab));std::memcpy(B,bb,sizeof(bb));for(auto &c:C)c=19.0f;\n if(invar_f32_contract_v1(A,B,C)!=0)return 10;\n std::uint32_t actual;std::memcpy(&actual,&C[1],4);\n if((0x{expected:08x}u & 0x7fffffffu)>0x7f800000u){{if((actual&0x7fffffffu)<=0x7f800000u)return 11;}}else if(actual!=0x{expected:08x}u)return 12;\n if(C[0]!=19.0f||C[2]!=19.0f||C[3]!=19.0f)return 13;\n std::printf(\"%08x\\n\",actual);return 0;}}",
        list(a),
        list(b),
        a.len(),
        b.len()
    )
}

#[test]
fn floating_emitted_cpu_exact_strict_edge_vectors() {
    // Frozen integer/rational numeric oracle vectors: separate rounding,
    // increasing-k order, nearest-even, gradual underflow, zeros and NaN class.
    let cases: &[(&[u32], &[u32], u32)] = &[
        (
            &[0, 0x3f800001, 0xbf800000, 0],
            &[0, 0x3f7ffffe, 0x3f800000, 0],
            0,
        ),
        (
            &[0, 0x4b800000, 0x3f800000, 0xcb800000, 0],
            &[0, 0x3f800000, 0x3f800000, 0x3f800000, 0],
            0,
        ),
        (
            &[0, 0x4b800000, 0xcb800000, 0x3f800000, 0],
            &[0, 0x3f800000, 0x3f800000, 0x3f800000, 0],
            0x3f800000,
        ),
        (&[0, 1, 1, 0], &[0, 0x3f000000, 0x3f800000, 0], 1),
        (&[0, 0x00800000, 0], &[0, 0x3f000000, 0], 0x00400000),
        (&[0, 3, 0], &[0, 0x3f000000, 0], 2),
        (&[0, 0x80000000, 0], &[0, 0x3f800000, 0], 0),
        (&[0, 0x7f7fffff, 0], &[0, 0x40000000, 0], 0x7f800000),
        (&[0, 0x7f800000, 0], &[0, 0, 0], 0x7fc00000),
        (&[0, 0x7f800001, 0], &[0, 0x3f800000, 0], 0x7fc00000),
    ];
    for &(a, b, expected) in cases {
        let mut ir = plan([1, u32::try_from(a.len() - 2).unwrap(), 1]);
        ir.inputs[0].elements = u64::try_from(a.len()).unwrap();
        ir.inputs[1].elements = u64::try_from(b.len()).unwrap();
        let p = admit(ir);
        for optimization in ["-O0", "-O2"] {
            let result = cpu_run(&p, &dot_harness(a, b, expected), optimization);
            eprintln!(
                "{optimization} inner={} expected={expected:08x} actual={}",
                p.ir().dimensions[1],
                result.trim()
            );
        }
    }
}

#[test]
fn floating_emitted_cpu_zero_one_odd_shapes_and_offsets() {
    for dimensions in [
        [0, 5, 3],
        [3, 5, 0],
        [0, 0, 0],
        [2, 0, 3],
        [1, 1, 1],
        [3, 5, 3],
    ] {
        let p = admit(plan(dimensions));
        let [rows, inner, cols] = dimensions;
        let mut harness = String::from("#include <vector>\nint main(){\n");
        writeln!(
            harness,
            "std::vector<float>A({},1.0f),B({},2.0f),C({},19.0f);",
            p.ir().inputs[0].elements,
            p.ir().inputs[1].elements,
            p.ir().output.elements
        )
        .unwrap();
        harness.push_str("if(invar_f32_contract_v1(A.data(),B.data(),C.data())!=0)return 1;\n");
        writeln!(harness,"for(unsigned long long i=0;i<C.size();++i){{float expected=(i>=1&&i<{}ull)?{}.0f:19.0f;if(C[i]!=expected)return 2;}}",u64::from(rows)*u64::from(cols)+1,inner*2).unwrap();
        if rows == 0 || cols == 0 {
            harness.push_str("if(invar_f32_contract_v1(nullptr,nullptr,nullptr)!=0)return 3;\n");
        } else if inner == 0 {
            harness.push_str("if(invar_f32_contract_v1(nullptr,nullptr,C.data())!=0)return 4;\n");
        }
        harness.push_str("return 0;}\n");
        cpu_run(&p, &harness, "-O2");
    }
}

#[test]
fn floating_emitted_cpu_refuses_bad_environment_and_runtime_aliases() {
    let p = admit(plan([1, 1, 1]));
    let harness = r#"
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
int main(){
float A[4]={0,1,0,0},B[4]={0,2,0,0},C[4]={19,19,19,19};
if(invar_f32_contract_v1(nullptr,B,C)!=2)return 1;
if(invar_f32_contract_v1(A,B,A)!=2)return 2;
if(invar_f32_contract_v1(A,B,A+1)!=2)return 3;
const int saved=std::fegetround();std::fesetround(FE_UPWARD);
if(invar_f32_contract_v1(A,B,C)!=1)return 4;
std::fesetround(saved);
std::feraiseexcept(FE_DIVBYZERO);const int flags=std::fetestexcept(FE_ALL_EXCEPT);
if(invar_f32_contract_v1(A,B,C)!=0)return 5;
if(std::fetestexcept(FE_ALL_EXCEPT)!=flags)return 6;
#if defined(__SSE__)
const auto csr=_mm_getcsr();
for(unsigned mask:{0x8000u,0x40u}){_mm_setcsr(csr|mask);if(invar_f32_contract_v1(A,B,C)!=1)return 7;_mm_setcsr(csr);}
#endif
return 0;}
"#;
    // The small host environment perturbations occur only in this child process.
    cpu_run(
        &p,
        &format!("#include <initializer_list>\n{harness}"),
        "-O2",
    );
}
