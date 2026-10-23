//! The common GNET emitter's complete scalar state machine is batched only by
//! this qualification harness. Production ABI13 launch geometry is unchanged.
#![forbid(unsafe_code)]
#[path = "support/gnet_entry.rs"]
mod entry;
#[cfg(all(feature = "metal", target_os = "macos"))]
#[path = "issue_m4_01_device/ops.rs"]
mod ops;
use invar::{
    check, front,
    general_net::{
        self,
        compiled::{self, Target},
        device_host::{Backend, Plan},
    },
    numeric_contract::F32Bits,
};
use invar_device::trig_f32;

const FUNCTIONS: [&str; 3] = ["sin", "cos", "sqrt"];
const SPECIAL: &[u32] = &[
    0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000, 0x39800000,
    0x397fffff, 0x39800001, 0x3f000000, 0x3f800000, 0x3f800001, 0xbf800000, 0x40000000, 0x3fc90fda,
    0x3fc90fdb, 0x3fc90fdc, 0x40490fdb, 0x4096cbe4, 0x497fffff, 0x49800000, 0x49800001, 0xc9800001,
    0x6ff9be45, 0x6f79be45, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7f800001, 0xff800001,
    0x7fc00000, 0xffc00001, 0x7fffffff, 0xffffffff,
];
fn host(f: usize, x: u32) -> u32 {
    match f {
        0 => trig_f32::host_sin(x),
        1 => trig_f32::host_cos(x),
        _ => trig_f32::host_sqrt(x),
    }
}
fn exact(f: usize, x: u32) -> u32 {
    let x = F32Bits::from_bits(x);
    match f {
        0 => x.sin(),
        1 => x.cos(),
        _ => x.sqrt_sequence(),
    }
    .to_bits()
}
fn inputs(bits: u32) -> Vec<u32> {
    assert!((1..=24).contains(&bits));
    // One input in each of 2^bits equal bit-pattern strata. Fixed, injective.
    let shift = 32 - bits;
    let mut v: Vec<_> = (0u32..1 << bits)
        .map(|i| (i << shift) | (i.wrapping_mul(0x9e3779b1) >> bits))
        .collect();
    v.extend_from_slice(SPECIAL);
    for b in 0..23 {
        for s in [0, 0x80000000] {
            v.extend([s | (1 << b), s | 0x7f800000 | (1 << b)]);
        }
    }
    if bits == 24 {
        // Exhaust every NaN payload and both signs of every subnormal, in
        // addition to the stratified normal/huge-argument coverage.
        for m in 1..0x0080_0000 {
            v.extend([m, m | 0x8000_0000, m | 0x7f80_0000, m | 0xff80_0000]);
        }
    }
    v
}
fn scalar(f: usize) -> compiled::Artifact {
    let source = format!("fn run(x:f32)->f32{{return {}(x);}}", FUNCTIONS[f]);
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), false).unwrap();
    let net = general_net::Net::lower(&checked, Default::default()).unwrap();
    // Scalar programs need no array storage; one descriptor slot is the minimum.
    compiled::emit_resumable_owned_numeric(
        &net,
        &net.verify().unwrap(),
        "run",
        Target::Cpu,
        Default::default(),
        compiled::OwnedLimits {
            elements: 0,
            allocations: 1,
            cleanup_events: 1,
        },
    )
    .unwrap()
}
fn corpus<B: Backend>(backend: &mut B, target: Target) {
    let xs = inputs(8);
    for (f, name) in FUNCTIONS.iter().enumerate() {
        let source = format!(
            "fn run(a:read Array<f32>,b:mut Array<f32>)->u32{{for i in 0u32..len(a){{b[i]={name}(a[i]);}}return len(a);}} fn main()->i32{{return 0;}}"
        );
        let args = vec![entry::f(&xs), entry::f(&vec![0; xs.len()])];
        let invar::eval::Value::ArrayF32(out) = &args[1] else {
            panic!()
        };
        let out = std::rc::Rc::clone(out);
        assert_eq!(
            entry::agree(backend, target, &source, "run", args, 100000),
            compiled::Status::Success
        );
        for (&x, y) in xs.iter().zip(out.borrow().iter()) {
            assert_eq!(y.to_bits(), host(f, x), "{name} {x:#x}");
        }
        println!(
            "{target:?} ABI13 {name}: {} bit-identical outputs",
            xs.len()
        );
    }
    let fk = include_str!("fixtures/gnet_trig/fk.inv");
    let q = [
        0x3e800000, 0xbf000000, 0x3f800000, 0x3fc00000, 0xbe800000, 0x40000000,
    ];
    let tcp = entry::f(&[0; 3]);
    let invar::eval::Value::ArrayF32(out) = &tcp else {
        panic!()
    };
    let out = std::rc::Rc::clone(out);
    assert_eq!(
        entry::agree(backend, target, fk, "fk", vec![entry::f(&q), tcp], 100000),
        compiled::Status::Success
    );
    println!(
        "{target:?} six-joint FK: complete net observation and mutable outputs match; TCP bits {:?}",
        out.borrow().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

#[cfg(not(target_os = "macos"))]
#[test]
fn cpu_cpp_scalar_and_both_device_models_match_oracle() {
    use std::{fs, process::Command};
    let dir = std::env::temp_dir().join(format!("gnet-trig-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let xs = inputs(16);
    fs::write(
        dir.join("input"),
        xs.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
    .unwrap();
    for (f, name) in FUNCTIONS.iter().enumerate() {
        let source = scalar(f);
        let driver = r#"
#include <fstream>
int main(int argc,char** argv) {
  if(argc!=3)return 2;
  std::ifstream in(argv[1],std::ios::binary); std::ofstream out(argv[2],std::ios::binary);
  unsigned x; while(in.read((char*)&x,4)) {
    gnet_w a[1]={x},o[10]={},heap[4]={};gnet_u trace[224]={},cleanup[4]={};GnetTask task={};
    if(!gnet_resume(a,o,trace,1000,1000,32,heap,cleanup,&task,1000000))return 4;
    if(o[0]!=0)return 3;
    unsigned y=(unsigned)o[1];out.write((char*)&y,4);
  } return !in.eof() || !out.good();
}
"#;
        fs::write(
            dir.join("source.cpp"),
            format!("{}\n{driver}", source.source()),
        )
        .unwrap();
        let out = Command::new("c++")
            .args([
                "-std=c++17",
                "-O2",
                "-fsanitize=undefined",
                "-fno-sanitize-recover=all",
            ])
            .arg(dir.join("source.cpp"))
            .arg("-o")
            .arg(dir.join("run"))
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            Command::new(dir.join("run"))
                .arg(dir.join("input"))
                .arg(dir.join("output"))
                .status()
                .unwrap()
                .success()
        );
        let out = fs::read(dir.join("output")).unwrap();
        assert_eq!(out.len(), xs.len() * 4);
        for (&x, y) in xs.iter().zip(out.chunks_exact(4)) {
            let y = u32::from_le_bytes(y.try_into().unwrap());
            assert_eq!(y, host(f, x), "C++ {name} {x:#010x}");
            if SPECIAL.contains(&x) {
                assert_eq!(y, exact(f, x), "exact {name} {x:#010x}");
            }
        }
        println!(
            "C++ full scalar GNET {name}: {} bit-identical outputs",
            xs.len()
        );
    }
    corpus(
        &mut invar::general_net::device_host::cpu::CpuModel::default(),
        Target::Cuda,
    );
    corpus(
        &mut invar::general_net::device_host::cpu::CpuModel::default(),
        Target::Metal,
    );
    fs::remove_dir_all(dir).unwrap();
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "real Metal: 2^24 stratified inputs per op through the scalar GNET state machine"]
fn metal_stratified_and_entry_fk() {
    use invar_device::f32_ops::{Kind, Node, Program, Scalar};
    let mut metal = invar_metal::MetalDevice::open_runtime_source().unwrap();
    println!("Metal {}", metal.name());
    let bits = std::env::var("INVAR_GNET_TRIG_BITS")
        .map(|s| s.parse().unwrap())
        .unwrap_or(24);
    let xs = inputs(bits);
    for (f, name) in FUNCTIONS.iter().enumerate() {
        if std::env::var("INVAR_GNET_TRIG_OP").is_ok_and(|op| op != *name) {
            continue;
        }
        let a = scalar(f);
        // Only the test dispatch/storage spelling changes. The complete
        // generated state machine, event checks and operation bodies remain.
        let body = a
            .source()
            .replace("#define GNET_MEM\n", "#define GNET_MEM thread\n")
            .replace("#define GNET_PRIVATE\n", "#define GNET_PRIVATE thread\n")
            .replace(
                "typedef long long gnet_w; typedef unsigned long long gnet_u;",
                "typedef long gnet_w; typedef ulong gnet_u;",
            )
            .replace("#include <climits>", "")
            .replace("#include <cstddef>", "")
            .replace("CHAR_BIT", "8")
            .replace("ULLONG_MAX", "18446744073709551615ULL")
            .replace(
                "#define GNET_TRIG_CONSTANT static const",
                "#define GNET_TRIG_CONSTANT constant",
            );
        // Qualification-only emitted-source mutations. The ordinary comparison
        // below must fail with actual wrong output, after successful compilation.
        let body = match std::env::var("INVAR_GNET_TRIG_MUTATION").as_deref() {
            Ok("coefficient") => body.replace(&format!("{}u", 0xbe2a_aaa3u32), &format!("{}u", 0xbe2a_aaa4u32)),
            Ok("range") => [0x3fc9_0fdbu32,0xb33b_bd2e,0xa6f7_2ced].iter().fold(body, |s,c| s.replace(&format!("{c}u"),"0u")),
            Ok("fast-sqrt") => body.replace("GNET_FN unsigned gnet_f32_sqrt(unsigned a) {", "GNET_FN unsigned gnet_f32_sqrt(unsigned a) { return as_type<uint>(fast::sqrt(as_type<float>(a)));"),
            Err(_) => body,
            Ok(other) => panic!("unknown mutation {other}"),
        };
        for chunk in xs.chunks(1 << 18) {
            let len = chunk.len();
            let plan = ops::plan(
                Program {
                    length: len as u32,
                    inputs: vec![Scalar::F32],
                    nodes: vec![Node::Input(0)],
                    kind: Kind::Map,
                },
                metal.capabilities(),
            );
            let source = format!(
                r#"#include <metal_stdlib>
using namespace metal;
{body}
kernel void invar_f32_ops_v2(device const uint* A [[buffer(0)]],device const uint* B [[buffer(1)]],device uint* C [[buffer(2)]],device atomic_uint* F [[buffer(3)]],uint i [[thread_position_in_grid]],uint threads [[threads_per_grid]]) {{
  atomic_fetch_and_explicit(F,3u,memory_order_relaxed);
  for(;i<{len}u;i+=threads){{long input[1]={{A[i]}},output[10]={{}},heap[4]={{}};ulong trace[224]={{}},cleanup[4]={{}};GnetTask task={{}};
  if(!gnet_resume(input,output,trace,1000,1000,32,heap,cleanup,&task,1000000)||output[0]!=0)atomic_store_explicit(F,1u,memory_order_relaxed);
  C[i]=(uint)output[1];}}
}}"#
            );
            // Compact identifiers/whitespace to fit the existing diagnostic
            // transport cap; no expressions, branches or checks are removed.
            let source = source
                .replace("gnet_f32_", "gf_")
                .replace("invar_trig_f32_v1", "it")
                .replace("invar_sqrt_f32_v1", "is");
            // The host loader's offsetof layout assertion has no MSL spelling;
            // the sizeof(GnetTask) assertion still pins the task layout.
            let source = source
                .lines()
                .filter(|line| !line.trim_start().starts_with("//"))
                .filter(|line| !line.contains("GNET ABI11 layout mismatch"))
                .map(str::trim)
                .collect::<Vec<_>>()
                .join("\n");
            assert!(
                source.len() <= invar_device::MAX_SOURCE_BYTES,
                "{} bytes",
                source.len()
            );
            let out = metal
                .diagnose_f32_ops_source(&plan, chunk, &source)
                .unwrap();
            assert!(!out.flagged);
            for (&x, y) in chunk.iter().zip(out.output) {
                assert_eq!(y, host(f, x), "Metal {name} {x:#010x}");
                if SPECIAL.contains(&x) {
                    assert_eq!(y, exact(f, x));
                }
            }
        }
        println!(
            "Metal full scalar GNET {name}: {} bit-identical outputs",
            xs.len()
        );
    }
    corpus(
        &mut invar::general_net::device_host::metal::Metal::open_runtime_source().unwrap(),
        Target::Metal,
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "physical CUDA; no hardware required for --no-run"]
fn cuda_entry_fk() {
    let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    corpus(
        &mut invar::general_net::device_host::cuda_nvrtc::CudaNvrtc::open(compiler).unwrap(),
        Target::Cuda,
    );
}

#[test]
fn generated_device_sources_use_the_shared_sequence_and_software_sqrt() {
    for target in [Target::Cuda, Target::Metal] {
        let p = Plan::from_source(
            b"fn run(x:f32)->f32{return sin(x)+cos(x)+sqrt(x);}fn main()->i32{return 0;}",
            "run",
            target,
            entry::limits(),
        )
        .unwrap();
        let source = p.artifact().source();
        assert!(source.contains("generated from invar_device::trig_f32"));
        assert!(source.contains("invar_trig_f32_v1((unsigned)fr.v["));
        assert!(source.contains("invar_sqrt_f32_v1(fr.v["));
        assert!(source.contains("if (n-q*q > q) ++q;"));
        for forbidden in [
            "__sinf(",
            "__cosf(",
            "fast::sqrt",
            "__fsqrt_rn(",
            "as_type<float>",
        ] {
            assert!(!source.contains(forbidden));
        }
        if let Some(dir) = std::env::var_os("INVAR_GNET_TRIG_SOURCES") {
            std::fs::write(
                std::path::Path::new(&dir).join(format!("trig-{target:?}.txt")),
                source,
            )
            .unwrap();
        }
    }
}
