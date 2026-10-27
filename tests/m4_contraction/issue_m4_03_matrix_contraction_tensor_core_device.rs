//! M4-03 tiled strict-f32 device schedule, checked without a GPU: the CUDA and
//! MSL kernels come from the same verified plan as the CPU emission, their
//! identity and refusals are typed, and the emitted kernel text itself is
//! executed on the host with real barriers (256 threads per group) and
//! compared bit for bit with the independent F32Bits oracle.
#![forbid(unsafe_code)]

use crate::corpus;
use invar::contraction_source;

use contraction_source::{Contract, DeviceEmission, Error};
use corpus::{DIRECT, LARGEST, NONFINITE, NORMAL, SHAPES, SUBNORMAL};
use invar_device::{Capabilities, Provenance, floating, gemm};
use std::{
    fs,
    sync::atomic::{AtomicUsize, Ordering},
};

const METAL: Capabilities = Capabilities::Metal {
    apple_family: 9,
    msl_major: 3,
    msl_minor: 1,
};
const BLACKWELL: Capabilities = Capabilities::Cuda {
    major: 12,
    minor: 9,
    sm: 120,
};

fn contract() -> Contract {
    Contract::from_source(DIRECT.as_bytes(), "mat").unwrap()
}
fn lengths([r, k, n]: [u32; 3]) -> [u64; 2] {
    [u64::from(r) * u64::from(k), u64::from(k) * u64::from(n)]
}
fn device(dims: [u32; 3], caps: Capabilities) -> DeviceEmission {
    contract()
        .emit_device(DIRECT.as_bytes(), dims, lengths(dims), caps)
        .unwrap()
}

#[test]
fn issue_m4_03_matrix_contraction_tensor_core_device_one_plan_three_targets() {
    let c = contract();
    for dims in SHAPES.iter().chain(&LARGEST).copied() {
        let cpu = c
            .emit_cpu(DIRECT.as_bytes(), dims, lengths(dims), BLACKWELL)
            .unwrap();
        let cuda = device(dims, BLACKWELL);
        let metal = device(dims, METAL);
        // One verified device IR; only the target capabilities differ.
        assert_eq!(cuda.plan().ir(), cpu.plan().ir());
        assert_eq!(metal.plan().ir(), cpu.plan().ir());
        assert_eq!(cuda.plan().capabilities(), BLACKWELL);
        assert_eq!(metal.plan().capabilities(), METAL);
        let mut keys = [cpu.key(), cuda.key(), metal.key()].map(|k| k.to_hex());
        keys.sort();
        keys.windows(2).for_each(|w| assert_ne!(w[0], w[1]));
        assert_eq!(cuda.source(), gemm::emit_cuda(cuda.plan()).unwrap());
        assert_eq!(metal.source(), gemm::emit_msl(metal.plan()).unwrap());
        let schedule = format!("// schedule={}", gemm::SCHEDULE);
        for e in [&cuda, &metal] {
            assert!(e.source().contains(&schedule), "{dims:?}");
            assert!(e.source().len() <= invar_device::MAX_SOURCE_BYTES);
            for option in e.options() {
                assert!(e.source().contains(option.as_str()), "{option}");
            }
        }
        // Strict CUDA spelling: explicit RN multiply and add, never fused.
        assert!(cuda.options().iter().any(|o| o == "--fmad=false"));
        assert!(cuda.options().iter().any(|o| o == "--ftz=false"));
        assert!(
            cuda.options()
                .iter()
                .any(|o| o == "--gpu-architecture=compute_120")
        );
        let body = cuda.source();
        if dims[0] != 0 && dims[2] != 0 {
            assert!(body.contains("__launch_bounds__(256)"));
            // A zero inner extent omits the shared tiles and the k loop.
            let stepped = dims[1] != 0;
            for source in [body, metal.source()] {
                assert_eq!(source.contains("As[16][65]"), stepped, "{dims:?}");
                assert_eq!(source.contains("k0 += 16"), stepped, "{dims:?}");
            }
            assert_eq!(
                body.contains("__fadd_rn(acc[r][c], __fmul_rn(a[r], b[c]))"),
                stepped
            );
            assert_eq!(metal.source().contains("invar_mul(aw[r], bw[c]"), stepped);
        }
        for banned in ["fmaf", "__fma", "fast", "tf32", "half", "wmma", "mma.sync"] {
            assert!(!body.contains(banned), "{banned} in CUDA");
        }
        assert!(metal.source().contains("#pragma METAL fp contract(off)"));
        assert!(!metal.source().contains("fma("));
        // Identity is a pure function of the source and plan.
        assert_eq!(device(dims, BLACKWELL).key(), cuda.key());
        assert_ne!(
            device(dims, Capabilities::cuda_12_9_sm89()).key(),
            cuda.key()
        );
    }
}

#[test]
fn issue_m4_03_matrix_contraction_tensor_core_device_geometry_and_fallback() {
    for (dims, tiles, groups) in [
        ([1, 1, 1], 1, 1),
        ([64, 16, 64], 1, 1),
        ([65, 17, 63], 2, 2),
        ([129, 31, 130], 9, 9),
        ([256, 1024, 256], 16, 16),
        ([1000, 67, 1000], 256, 256),
        ([0, 3, 2], 0, 0),
        ([1, 0, 1 << 20], 16_384, 16_384),
        ([1, 1, 65_535 * 64 + 1], 65_536, 65_535),
    ] {
        let k = u64::from(dims[1]);
        let len = [u64::from(dims[0]) * k, k * u64::from(dims[2])];
        let plan = contract()
            .emit_device(DIRECT.as_bytes(), dims, len, BLACKWELL)
            .map(|e| e.geometry());
        let expected = gemm::Geometry {
            tiles,
            groups,
            threads: gemm::THREADS,
        };
        let geometry = plan.unwrap();
        assert_eq!(geometry, expected, "{dims:?}");
    }
    assert_eq!(gemm::cpu_fallback([0, 3, 2]), Some("empty-output"));
    assert_eq!(gemm::cpu_fallback([3, 5, 0]), Some("empty-output"));
    assert_eq!(gemm::cpu_fallback([2, 0, 3]), Some("zero-inner-extent"));
    assert_eq!(
        gemm::cpu_fallback([3, 5, 2]),
        Some("below-device-work-threshold")
    );
    assert_eq!(
        gemm::cpu_fallback([64, 16, 63]),
        Some("below-device-work-threshold")
    );
    assert_eq!(gemm::cpu_fallback([64, 16, 64]), None);
    assert_eq!(gemm::cpu_fallback([256, 1024, 256]), None);
    let receipt = device([3, 5, 2], METAL).receipt();
    assert_eq!(receipt["placement"], "cpu:below-device-work-threshold");
    let receipt = device([256, 1024, 256], BLACKWELL).receipt();
    assert_eq!(receipt["placement"], "device");
}

fn direct_plan(dims: [u32; 3]) -> floating::Verified {
    let [r, k, n] = dims;
    let buffer = |id, shape: [u32; 2]| floating::Buffer {
        id,
        elements: u64::from(shape[0]) * u64::from(shape[1]),
        shape,
        strides: [u64::from(shape[1]), 1],
        offset: 0,
        alignment: 4,
    };
    let ir = floating::Ir {
        version: floating::VERSION,
        numeric_version: floating::NUMERIC_VERSION,
        numeric_policy: floating::NUMERIC_POLICY.into(),
        layout_version: floating::LAYOUT_VERSION,
        span: invar_device::Span { start: 1, end: 2 },
        source_identity: 7,
        graph_revision: 1,
        function_contract: [1; 32],
        dimensions: dims,
        inputs: [buffer(0, [r, k]), buffer(1, [k, n])],
        output: buffer(2, [r, n]),
    };
    let origin = Provenance {
        source_identity: 7,
        graph_revision: 1,
    };
    floating::verify(ir, BLACKWELL, origin).unwrap()
}

#[test]
fn issue_m4_03_matrix_contraction_tensor_core_device_size_2_28() {
    let origin = Provenance {
        source_identity: 7,
        graph_revision: 1,
    };
    for source in [DIRECT, corpus::TC_DIRECT] {
        let c = Contract::from_source(source.as_bytes(), "mat").unwrap();
        for caps in [BLACKWELL, METAL] {
            // Every M5-01 square shape, without allocating any matrix storage.
            for exponent in (10..=28).step_by(2) {
                let side = 1u32 << (exponent / 2);
                let e = c
                    .emit_device(source.as_bytes(), [side; 3], [1 << exponent; 2], caps)
                    .unwrap();
                let n = (1u64 << exponent) - 1;
                for slot in 0..3 {
                    assert_eq!(e.plan().physical_index(slot, [side - 1; 2]), Ok(n));
                    assert_eq!((n + 1) * 4, 4u64 << exponent);
                }
                assert!(e.geometry().groups <= 65_535);
                assert!(e.source().contains(if caps == METAL {
                    "ulong"
                } else {
                    "unsigned long long"
                }));
            }
            // Exactly one cell above the cap, with tiny operands.
            assert!(matches!(
                c.emit_device(
                    source.as_bytes(),
                    [1, 1, (1 << 28) + 1],
                    [1, (1 << 28) + 1],
                    caps
                ),
                Err(Error::Bounds)
            ));
        }
    }
    let plan = direct_plan([1 << 14; 3]);
    assert_eq!(gemm::geometry(&plan).tiles, 65_536);
    assert_eq!(gemm::geometry(&plan).groups, 65_535);
    let mut ir = plan.ir().clone();
    ir.dimensions = [1, 1, (1 << 28) + 1];
    let buffer = |id, shape: [u32; 2]| floating::Buffer {
        id,
        elements: u64::from(shape[0]) * u64::from(shape[1]),
        shape,
        strides: [u64::from(shape[1]), 1],
        offset: 0,
        alignment: 4,
    };
    ir.inputs = [buffer(0, [1, 1]), buffer(1, [1, (1 << 28) + 1])];
    ir.output = buffer(2, [1, (1 << 28) + 1]);
    for policy in [floating::NUMERIC_POLICY, floating::TC_POLICY] {
        ir.numeric_policy = policy.into();
        assert_eq!(
            floating::verify(ir.clone(), BLACKWELL, origin).unwrap_err(),
            floating::Error::MatrixLimit
        );
    }
    let mut ir = plan.ir().clone();
    ir.dimensions = [u32::MAX; 3];
    ir.inputs[0].shape = [u32::MAX; 2];
    ir.inputs[1].shape = [u32::MAX; 2];
    ir.output.shape = [u32::MAX; 2];
    assert!(floating::verify(ir, BLACKWELL, origin).is_err());
}

#[test]
fn issue_m4_03_matrix_contraction_tensor_core_device_receipt_names_target_and_contract() {
    for (caps, target) in [
        (BLACKWELL, "cuda-nvrtc-c++17-gemm-tiled-v1"),
        (METAL, "metal-msl-3.1-gemm-tiled-v1"),
    ] {
        let e = device([65, 17, 63], caps);
        let r = e.receipt();
        assert_eq!(r["target"], target);
        assert_eq!(r["track"], "generated");
        assert_eq!(r["numeric_policy"], "strict-f32-v1");
        assert_eq!(r["numeric_version"], 1);
        assert_eq!(r["layout_version"], 1);
        assert_eq!(r["schedule"], gemm::SCHEDULE);
        assert_eq!(r["dimensions"], serde_json::json!([65, 17, 63]));
        assert_eq!(r["tile"], serde_json::json!([64, 64, 16]));
        assert_eq!(r["groups"], 2);
        assert_eq!(r["threads"], 256);
        assert_eq!(r["key"], e.key().to_hex());
        assert_eq!(r["capabilities"], format!("{caps:?}"));
        let hash = r["generated_sha256"].as_str().unwrap();
        assert_eq!(hash.len(), 64);
        assert_ne!(
            hash,
            device([65, 17, 64], caps).receipt()["generated_sha256"]
        );
        println!("{r}");
    }
}

#[test]
fn issue_m4_03_matrix_contraction_tensor_core_device_typed_refusals() {
    let c = contract();
    let refuse = |dims: [u32; 3], len: [u64; 2], caps| {
        c.emit_device(DIRECT.as_bytes(), dims, len, caps)
            .map(|_| ())
            .unwrap_err()
    };
    // Just above the matrix-owned source cap (output, then operands).
    let above = (1 << 28) + 1;
    assert!(matches!(
        refuse([1, 1, above], [1, u64::from(above)], METAL),
        Error::Bounds
    ));
    assert!(matches!(
        refuse([1, above, 1], [u64::from(above); 2], METAL),
        Error::Bounds
    ));
    // Unknown or short layouts, and toolchain/device capability mismatch.
    assert!(matches!(
        refuse([2, 3, 2], [5, 6], BLACKWELL),
        Error::Layout(_) | Error::Device(_)
    ));
    for caps in [
        Capabilities::Cuda {
            major: 12,
            minor: 4,
            sm: 120,
        },
        Capabilities::Cuda {
            major: 11,
            minor: 8,
            sm: 86,
        },
        Capabilities::Cuda {
            major: 12,
            minor: 9,
            sm: 35,
        },
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
    ] {
        assert!(
            matches!(
                refuse([2, 2, 2], [4, 4], caps),
                Error::Device(floating::Error::Capabilities)
            ),
            "{caps:?}"
        );
    }
    // A changed source is not this extraction.
    let other = DIRECT.replace("rows * cols", "cols * rows");
    assert!(matches!(
        c.emit_device(other.as_bytes(), [1, 1, 1], [1, 1], METAL),
        Err(Error::SourceMismatch)
    ));
    // Emitters refuse a plan for the other target family.
    let cuda = device([2, 2, 2], BLACKWELL);
    let metal = device([2, 2, 2], METAL);
    assert_eq!(
        gemm::emit_msl(cuda.plan()),
        Err(floating::Error::Capabilities)
    );
    assert_eq!(
        gemm::emit_cuda(metal.plan()),
        Err(floating::Error::Capabilities)
    );
    // A plan relabeled `tc-f16-f32-v1` verifies as a Tensor Core plan, and
    // then no strict emitter spells it; an unknown policy never verifies.
    let mut ir = cuda.plan().ir().clone();
    ir.numeric_policy = "tc-f16-f32-v1".into();
    let origin = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    let relabeled = floating::verify(ir.clone(), BLACKWELL, origin).unwrap();
    assert!(relabeled.is_tc());
    for emitted in [
        gemm::emit_cuda(&relabeled),
        floating::emit_cuda(&relabeled),
        floating::emit_cpu(&relabeled),
    ] {
        assert_eq!(emitted.unwrap_err(), floating::Error::Numeric);
    }
    ir.numeric_policy = "relaxed-f32-v1".into();
    assert_eq!(
        floating::verify(ir, BLACKWELL, origin).unwrap_err(),
        floating::Error::Numeric
    );
    // Output aliasing an input is refused before any emission.
    let mut ir = cuda.plan().ir().clone();
    ir.output.id = ir.inputs[1].id;
    assert_eq!(
        floating::verify(ir, BLACKWELL, origin).unwrap_err(),
        floating::Error::Alias
    );
}

// --- Host execution of the emitted kernel text -------------------------

const CUDA_SHIM: &str = r#"#include <barrier>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
struct InvarDim { unsigned x, y, z; };
static thread_local InvarDim threadIdx;
static InvarDim blockIdx, blockDim{256u, 1u, 1u}, gridDim{1u, 1u, 1u};
static std::barrier<>* invar_barrier;
#define __global__
#define __launch_bounds__(n)
#define __shared__ static
static inline void __syncthreads() { invar_barrier->arrive_and_wait(); }
static inline void __trap() { std::abort(); }
static inline float __fmul_rn(float a, float b) { volatile float r = a * b; return r; }
static inline float __fadd_rn(float a, float b) { volatile float r = a + b; return r; }
"#;

const MSL_SHIM: &str = r#"#include <atomic>
#include <barrier>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>
typedef unsigned int uint;
typedef __UINT64_TYPE__ ulong;
static_assert(sizeof(ulong) == 8, "64-bit ulong required");
#define kernel
#define device
#define threadgroup static
namespace metal {}
enum class mem_flags { mem_threadgroup };
static std::barrier<>* invar_barrier;
static inline void threadgroup_barrier(mem_flags) { invar_barrier->arrive_and_wait(); }
template <class T, class U> static inline T as_type(U u) { static_assert(sizeof(T) == sizeof(U)); T t; std::memcpy(&t, &u, sizeof t); return t; }
typedef std::atomic<uint> atomic_uint;
using std::atomic_fetch_and_explicit;
using std::atomic_fetch_or_explicit;
using std::atomic_store_explicit;
using std::memory_order_relaxed;
"#;

/// Reads `na nb nc groups` and the A/B words from stdin, runs every group in
/// order with 256 concurrent threads, prints the flag then the C words.
const CUDA_MAIN: &str = r#"
int main() {
  unsigned long na, nb, nc, groups; if (std::scanf("%lu %lu %lu %lu", &na, &nb, &nc, &groups) != 4) return 90;
  std::vector<unsigned> words(na + nb + 2); for (unsigned long i = 0; i < na + nb; ++i) if (std::scanf("%u", &words[i]) != 1) return 91;
  std::vector<float> A(na + 1), B(nb + 1), C(nc, 0.0f);
  std::memcpy(A.data(), words.data(), na * 4); std::memcpy(B.data(), words.data() + na, nb * 4);
  gridDim.x = static_cast<unsigned>(groups);
  for (unsigned g = 0; g < groups; ++g) {
    blockIdx.x = g; std::barrier<> bar(256); invar_barrier = &bar; std::vector<std::thread> threads;
    for (unsigned t = 0; t < 256; ++t) threads.emplace_back([&, t] { threadIdx = {t, 0u, 0u}; invar_f32_contract_v1(A.data(), B.data(), C.data()); });
    for (auto& t : threads) t.join();
  }
  std::printf("0\n"); for (float v : C) { unsigned w; std::memcpy(&w, &v, 4); std::printf("%u\n", w); }
}
"#;

const MSL_MAIN: &str = r#"
int main() {
  unsigned long na, nb, nc, groups; if (std::scanf("%lu %lu %lu %lu", &na, &nb, &nc, &groups) != 4) return 90;
  std::vector<uint> A(na + 1), B(nb + 1), C(nc, 0u); atomic_uint F(0x80000000u); // floating::guard::UNWRITTEN, as on the device
  for (unsigned long i = 0; i < na; ++i) if (std::scanf("%u", &A[i]) != 1) return 91;
  for (unsigned long i = 0; i < nb; ++i) if (std::scanf("%u", &B[i]) != 1) return 91;
  for (uint g = 0; g < groups; ++g) {
    std::barrier<> bar(256); invar_barrier = &bar; std::vector<std::thread> threads;
    for (uint t = 0; t < 256; ++t) threads.emplace_back([&, g, t] { invar_f32_contract_v1(A.data(), B.data(), C.data(), &F, g, uint(groups), t, 256u); });
    for (auto& t : threads) t.join();
  }
  std::printf("%u\n", F.load()); for (uint w : C) std::printf("%u\n", w);
}
"#;

static CHILD: AtomicUsize = AtomicUsize::new(0);

/// Compile `kernel` (emitted text, possibly mutated) with the host shim and
/// run `groups` groups over a zeroed C of `c_len` words; returns (flag, C).
fn host_run(
    metal: bool,
    groups: u32,
    c_len: u64,
    kernel: &str,
    a: &[u32],
    b: &[u32],
) -> (u32, Vec<u32>) {
    let (shim, main) = if metal {
        (MSL_SHIM, MSL_MAIN)
    } else {
        (CUDA_SHIM, CUDA_MAIN)
    };
    let text = format!(
        "{shim}{}{main}",
        kernel.replace("#include <metal_stdlib>", "")
    );
    let dir = std::env::temp_dir().join(format!(
        "invar-m4-03-host-{}-{}",
        std::process::id(),
        CHILD.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&dir).unwrap();
    fs::write(dir.join("kernel.cpp"), text).unwrap();
    let exe = dir.join(format!("run{}", std::env::consts::EXE_SUFFIX));
    let flags = [
        "-std=c++20",
        "-O1",
        "-pthread",
        "-ffp-contract=off",
        "-fno-fast-math",
        "-Wno-attributes",
        "-Wno-unknown-pragmas",
    ];
    crate::child_deadline::cxx(&flags, &dir.join("kernel.cpp"), &exe);
    let mut input = format!("{} {} {c_len} {groups}\n", a.len(), b.len());
    for w in a.iter().chain(b) {
        input.push_str(&format!("{w}\n"));
    }
    fs::write(dir.join("input"), input).unwrap();
    let ran = crate::child_deadline::command(120)
        .arg(&exe)
        .stdin(fs::File::open(dir.join("input")).unwrap())
        .output()
        .unwrap();
    assert!(
        ran.status.success(),
        "host run {:?}; preserved {dir:?}: {}",
        ran.status,
        String::from_utf8_lossy(&ran.stderr)
    );
    let mut words = String::from_utf8(ran.stdout)
        .unwrap()
        .lines()
        .map(|s| s.parse().unwrap())
        .collect::<Vec<u32>>();
    fs::remove_dir_all(dir).unwrap();
    let flag = words.remove(0);
    (flag, words)
}

fn run(e: &DeviceEmission, kernel: &str, a: &[u32], b: &[u32]) -> (u32, Vec<u32>) {
    let metal = e.target().starts_with("metal");
    let c_len = e.plan().ir().output.elements;
    host_run(metal, e.geometry().groups, c_len, kernel, a, b)
}

#[test]
fn issue_m4_03_matrix_contraction_tensor_core_device_emitted_kernels_match_oracle_on_host() {
    let mut cells = 0;
    for dims in SHAPES {
        let [r, k, n] = dims.map(|v| v as usize);
        for (pattern, seed) in [(&NORMAL[..], 3), (&SUBNORMAL[..], 5), (&NONFINITE[..], 7)] {
            let a = corpus::words(pattern, r * k, seed);
            let b = corpus::words(pattern, k * n, seed + 2);
            let expected = corpus::oracle(dims, &a, &b);
            for caps in [BLACKWELL, METAL] {
                let e = device(dims, caps);
                let (flag, c) = run(&e, e.source(), &a, &b);
                assert!(corpus::strict_equal(&c, &expected), "{dims:?} {caps:?}");
                // The host never flushes, so the Metal guard only reports
                // that a flushing device could differ: never on NORMAL words,
                // always when a subnormal operand is actually loaded.
                let used_subnormal = pattern == &SUBNORMAL[..]
                    && k > 0
                    && a.iter().chain(&b).any(|&w| corpus::is_subnormal(w));
                if caps == METAL && pattern == &NORMAL[..] {
                    assert_eq!(flag, 0, "{dims:?}");
                }
                if caps == METAL && used_subnormal {
                    assert_eq!(flag, 1, "{dims:?}");
                }
                cells += c.len();
            }
        }
    }
    println!("host-executed tiled CUDA/MSL cells bit-exact: {cells}");
}

#[test]
fn issue_m4_03_matrix_contraction_tensor_core_device_mutations_fail_strict_checks() {
    // Re-enabled FMA contraction keeps the -2^-46 residue in every cell.
    let (dims, a, b) = corpus::fma_discriminator(70, 67);
    let expected = corpus::oracle(dims, &a, &b);
    assert!(expected.iter().all(|&w| w == 0));
    let cuda = device(dims, BLACKWELL);
    assert_eq!(run(&cuda, cuda.source(), &a, &b).1, expected);
    let step = "__fadd_rn(acc[r][c], __fmul_rn(a[r], b[c]))";
    let fused = cuda
        .source()
        .replace(step, "std::fma(a[r], b[c], acc[r][c])");
    assert_ne!(fused, cuda.source());
    let (_, c) = run(&cuda, &fused, &a, &b);
    assert!(c.iter().all(|&w| w == 0xa880_0000));
    let metal = device(dims, METAL);
    assert_eq!(run(&metal, metal.source(), &a, &b), (0, expected));
    let fused = metal
        .source()
        .replace(
            "const float next = acc[r][c] + product;",
            "const float next = fma(as_type<float>(aw[r]), as_type<float>(bw[c]), acc[r][c]);",
        )
        .replace(
            "using namespace metal;",
            "using namespace metal; using std::fma;",
        );
    assert_ne!(fused, metal.source());
    let (_, c) = run(&metal, &fused, &a, &b);
    assert!(c.iter().all(|&w| w == 0xa880_0000));
    // Reassociation: (2^25 - 2^25) + 1 in source order is 1; reversed,
    // (1 - 2^25) rounds to -2^25 and the cell becomes +0.
    let dims = [64, 3, 64];
    let a = [0x4c00_0000, 0xcc00_0000, 0x3f80_0000].repeat(64);
    let b = vec![0x3f80_0000; 3 * 64];
    let expected = corpus::oracle(dims, &a, &b);
    assert!(expected.iter().all(|&w| w == 0x3f80_0000));
    let e = device(dims, BLACKWELL);
    assert_eq!(run(&e, e.source(), &a, &b).1, expected);
    let reversed = e.source().replace(
        "for (unsigned kk = 0u; kk < span; ++kk) {",
        "for (unsigned q = span; q-- > 0u;) { const unsigned kk = q;",
    );
    assert_ne!(reversed, e.source());
    let (_, c) = run(&e, &reversed, &a, &b);
    assert!(c.iter().all(|&w| w == 0));
    // A -0 accumulator start changes K = 0 cells from +0 to -0.
    let e = device([3, 0, 5], BLACKWELL);
    assert!(run(&e, e.source(), &[], &[]).1.iter().all(|&w| w == 0));
    let signed = e
        .source()
        .replace("acc[r][c] = 0.0f;", "acc[r][c] = -0.0f;");
    assert_ne!(signed, e.source());
    assert!(
        run(&e, &signed, &[], &[])
            .1
            .iter()
            .all(|&w| w == 0x8000_0000)
    );
}

/// The tiled Metal kernel claims its UNWRITTEN status word before anything
/// else; without the claim the host shim reads the sentinel back, which the
/// shared decoder refuses instead of publishing.
#[test]
fn issue_m4_03_matrix_contraction_tensor_core_device_metal_flag_is_claimed() {
    use floating::guard::{self, GUARD, InvalidWord, UNWRITTEN};
    let dims = [17, 9, 13];
    let a = corpus::words(&NORMAL, 17 * 9, 3);
    let b = corpus::words(&NORMAL, 9 * 13, 5);
    let e = device(dims, METAL);
    let claim = guard::msl_claim(GUARD);
    assert!(e.source().contains(&claim));
    assert_eq!(guard::decode(run(&e, e.source(), &a, &b).0), Ok(false));
    let (flag, _) = run(&e, &e.source().replace(&claim, ""), &a, &b);
    assert_eq!(guard::decode(flag), Err(InvalidWord(UNWRITTEN)));
}

#[test]
fn issue_m4_03_matrix_contraction_tensor_core_device_padded_output_keeps_padding() {
    let dims = [65, 17, 63];
    let e = device(dims, BLACKWELL);
    let mut ir = e.plan().ir().clone();
    ir.output.offset = 3;
    ir.output.elements = 65 * 63 + 9;
    let origin = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    let a = corpus::words(&NORMAL, 65 * 17, 3);
    let b = corpus::words(&NORMAL, 17 * 63, 5);
    let expected = corpus::oracle(dims, &a, &b);
    for caps in [BLACKWELL, METAL] {
        let plan = floating::verify(ir.clone(), caps, origin).unwrap();
        let metal = caps == METAL;
        let source = if metal {
            gemm::emit_msl(&plan)
        } else {
            gemm::emit_cuda(&plan)
        }
        .unwrap();
        let groups = gemm::geometry(&plan).groups;
        let (flag, c) = host_run(metal, groups, ir.output.elements, &source, &a, &b);
        assert_eq!(flag, 0);
        assert!(c[..3].iter().chain(&c[3 + 65 * 63..]).all(|&w| w == 0));
        assert_eq!(c[3..3 + 65 * 63], expected[..]);
    }
}
