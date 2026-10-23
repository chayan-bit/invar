//! Scaffolding for the proposed CUDA strict-f32 owned-continuation contract.
//! The emitted strict binary32 helpers are integer-only software arithmetic.
//! This test compiles the exact emitted helper text behind the CUDA preamble's
//! `__device__` macro shape and checks every admitted f32 operation bit-for-bit
//! against the independent exact-rational oracle, under both strict and
//! deliberately hostile fast-math C++ flags. It is CPU host compilation only:
//! no NVRTC, CUDA driver or GPU, and no ABI or target admission change.
#![cfg(unix)]
#[allow(dead_code)]
#[path = "../test-support/child_deadline.rs"]
mod child_deadline;

use invar::{
    general_net::{compiled, provenance},
    numeric_contract::{Comparison, F32Bits},
};
use std::{
    ffi::OsString,
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    time::{Duration, Instant},
};

const WALL: Duration = Duration::from_secs(180);

/// Bounded subprocess: own process group, wall deadline kills the group, and a
/// CPU-seconds limit applied through `sh` before exec.
fn bounded(program: &Path, args: &[OsString]) -> Output {
    let wall = child_deadline::wall(WALL);
    use std::os::unix::process::CommandExt;
    let mut command = Command::new("sh");
    command
        .arg("-c")
        .arg("ulimit -t 180 && exec \"$0\" \"$@\"")
        .arg(program)
        .args(args)
        .process_group(0)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().unwrap();
    let drain = |mut pipe: Box<dyn Read + Send>| {
        std::thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = pipe.read_to_end(&mut bytes);
            bytes
        })
    };
    let out = drain(Box::new(child.stdout.take().unwrap()));
    let err = drain(Box::new(child.stderr.take().unwrap()));
    let begun = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if begun.elapsed() >= wall {
            invar_foreign::numeric::terminate_compiler_group(child.id());
            let _ = child.kill();
            let status = child.wait().unwrap();
            panic!("{} exceeded {wall:?} ({status:?})", program.display());
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    Output {
        status,
        stdout: out.join().unwrap(),
        stderr: err.join().unwrap(),
    }
}

/// The exact helper block emitted for strict-f32 artifacts, without its
/// `<climits>` include and `CHAR_BIT`/`ULLONG_MAX` assertion (NVRTC does not
/// supply standard headers by default; the proposal replaces that assertion).
fn emitted_helpers() -> String {
    let artifact = provenance::compile_resumable_owned_numeric(
        b"fn calc(a: f32, b: f32, c: f32, x: i32, y: u32) -> f32 { let sum = fma(a, b, c) + a * b - c / a; if a == b or a < c { return sum + f32_from_i32(x); } return sum + f32_from_u32(y); }",
        "calc",
        compiled::Target::Cpu,
        provenance::Limits::default(),
        compiled::OwnedLimits::default(),
    )
    .unwrap();
    let source = artifact.artifact().source();
    let start = source
        .find("struct GnetF32Big")
        .expect("helper block start");
    let end_marker = "GNET_FN gnet_u gnet_f32_from_i32";
    let end = source[start..].find(end_marker).expect("helper block end") + start;
    let end = end + source[end..].find('\n').unwrap();
    let block = source[start..end].to_owned();
    for forbidden in ["float", "double", "<cmath>", "std::"] {
        assert!(!block.contains(forbidden), "helper uses {forbidden}");
    }
    block
}

/// Mirrors the CUDA preamble shape (`GNET_FN __device__ inline`, 64-bit words)
/// with a CPU mapping for `__device__`.
const CUDA_SHAPED_SHIM: &str = r#"
#include <cstdio>
#include <cstdint>
#include <vector>
#define __device__
#define GNET_FN __device__ inline
#define GNET_MEM
#define GNET_PRIVATE
typedef long long gnet_w; typedef unsigned long long gnet_u;
static_assert(sizeof(gnet_u) == 8 && sizeof(gnet_w) == 8, "64-bit words");
"#;
const DRIVER: &str = r#"
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    FILE* in = std::fopen(argv[1], "rb"); FILE* out = std::fopen(argv[2], "wb");
    if (!in || !out) return 3;
    unsigned int op, a, b, c;
    while (std::fread(&op, 4, 1, in) == 1) {
        if (std::fread(&a, 4, 1, in) != 1 || std::fread(&b, 4, 1, in) != 1 || std::fread(&c, 4, 1, in) != 1) return 4;
        gnet_u r = 0;
        switch (op) {
            case 0: r = gnet_f32_add(a, b); break;
            case 1: r = gnet_f32_sub(a, b); break;
            case 2: r = gnet_f32_mul(a, b); break;
            case 3: r = gnet_f32_div(a, b); break;
            case 4: r = gnet_f32_fma(a, b, c); break;
            case 5: r = gnet_f32_eq(a, b); break;
            case 6: r = gnet_f32_lt(a, b); break;
            case 7: r = gnet_f32_from_i32((gnet_w)(int)a); break;
            case 8: r = gnet_f32_from_u32(a); break;
            default: return 5;
        }
        unsigned int word = (unsigned int)r;
        if (r > 0xffffffffULL) return 6;
        std::fwrite(&word, 4, 1, out);
    }
    return std::fclose(out) == 0 ? 0 : 7;
}
"#;

fn oracle(op: u32, a: u32, b: u32, c: u32) -> u32 {
    let (x, y, z) = (
        F32Bits::from_bits(a),
        F32Bits::from_bits(b),
        F32Bits::from_bits(c),
    );
    match op {
        0 => x.add(y).to_bits(),
        1 => x.sub(y).to_bits(),
        2 => x.mul(y).to_bits(),
        3 => x.div(y).to_bits(),
        4 => x.fma(y, z).to_bits(),
        5 => u32::from(x.compare(Comparison::Eq, y)),
        6 => u32::from(x.compare(Comparison::Lt, y)),
        7 => F32Bits::from_i32(a as i32).to_bits(),
        8 => F32Bits::from_u32(a).to_bits(),
        _ => unreachable!(),
    }
}

const SPECIAL: [u32; 34] = [
    0x0000_0000,
    0x8000_0000, // +-0
    0x0000_0001,
    0x8000_0001, // +-smallest subnormal
    0x007f_ffff,
    0x807f_ffff, // +-largest subnormal
    0x0080_0000,
    0x8080_0000, // +-smallest normal
    0x3f80_0000,
    0xbf80_0000, // +-1
    0x3f80_0001,
    0x3f7f_ffff, // 1 + ulp, 1 - ulp
    0x4b80_0000,
    0xcb80_0000, // +-2^24
    0x3fc0_0000,
    0x4040_0000, // 1.5, 3
    0x7f7f_ffff,
    0xff7f_ffff, // +-max finite
    0x7f80_0000,
    0xff80_0000, // +-inf
    0x7fc0_0000,
    0xffc0_0000, // canonical NaN, negative quiet NaN
    0x7f80_0001,
    0x7fa1_2345, // signaling NaN, payload NaN
    0x3400_0000,
    0x3380_0000, // 2^-23, 2^-24
    0x0040_0000,
    0x0080_0001, // subnormal midpoint, smallest normal + ulp
    0x7f00_0000,
    0x0100_0000, // 2^127, 2^-125
    0x3eaa_aaab,
    0xbeaa_aaab, // +-1/3
    0x4f00_0000,
    0xcf00_0000, // +-2^31
];

/// Vectors: all special pairs for binary ops, all special triples for fma,
/// seeded random pairs/triples and conversion inputs.
fn vectors() -> Vec<[u32; 4]> {
    let mut v = Vec::new();
    for op in [0, 1, 2, 3, 5, 6] {
        for &a in &SPECIAL {
            for &b in &SPECIAL {
                v.push([op, a, b, 0]);
            }
        }
    }
    for &a in &SPECIAL {
        for &b in &SPECIAL {
            for &c in &SPECIAL {
                v.push([4, a, b, c]);
            }
        }
    }
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state as u32
    };
    for _ in 0..12_000 {
        for op in [0, 1, 2, 3, 4, 5, 6] {
            v.push([op, next(), next(), next()]);
        }
    }
    let ints = [
        0u32,
        1,
        u32::MAX,
        i32::MIN as u32,
        i32::MAX as u32,
        16_777_217,
        33_554_431,
        0x8000_0001,
    ];
    for op in [7, 8] {
        for &a in &ints {
            v.push([op, a, 0, 0]);
        }
        for _ in 0..12_000 {
            v.push([op, next(), 0, 0]);
        }
    }
    v
}

#[test]
fn issue_gnet_f32_helpers_are_bit_exact_and_flag_independent() {
    let helpers = emitted_helpers();
    let dir = PathBuf::from(
        std::env::var_os("CARGO_TARGET_TMPDIR").unwrap_or_else(|| std::env::temp_dir().into()),
    )
    .join(format!("gnet-f32-flags-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let cpp = dir.join("helpers.cpp");
    fs::write(&cpp, format!("{CUDA_SHAPED_SHIM}\n{helpers}\n{DRIVER}")).unwrap();
    // Negative control: ties-to-even replaced by round-half-up in the shared
    // rounding helper must be detected (exact ties such as 1 + 2^-24 exist).
    let tie_rule = "(gnet_f32_below(a,shift-1)||(q&1))";
    assert_eq!(helpers.matches(tie_rule).count(), 1);
    let mutant = dir.join("mutant.cpp");
    fs::write(
        &mutant,
        format!(
            "{CUDA_SHAPED_SHIM}\n{}\n{DRIVER}",
            helpers.replace(tie_rule, "true")
        ),
    )
    .unwrap();
    let cases = vectors();
    let mut input = Vec::with_capacity(cases.len() * 16);
    for case in &cases {
        for word in case {
            input.extend_from_slice(&word.to_le_bytes());
        }
    }
    let input_path = dir.join("vectors.bin");
    fs::write(&input_path, &input).unwrap();
    let expected: Vec<u32> = cases
        .iter()
        .map(|&[op, a, b, c]| oracle(op, a, b, c))
        .collect();
    let compiler = PathBuf::from(std::env::var_os("CXX").unwrap_or_else(|| "c++".into()));
    let builds: [(&str, &[&str], &Path, bool); 3] = [
        (
            "strict-ubsan",
            &[
                "-std=c++17",
                "-O2",
                "-fno-fast-math",
                "-ffp-contract=off",
                "-fsanitize=undefined",
                "-fno-sanitize-recover=all",
            ],
            &cpp,
            true,
        ),
        (
            "hostile-fast-math",
            &[
                "-std=c++17",
                "-O3",
                "-ffast-math",
                "-ffp-contract=fast",
                "-funsafe-math-optimizations",
            ],
            &cpp,
            true,
        ),
        (
            "mutated-tie-rounding",
            &["-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=off"],
            &mutant,
            false,
        ),
    ];
    for (name, flags, source, exact) in builds {
        let exe = dir.join(name);
        let args: Vec<OsString> = flags
            .iter()
            .map(OsString::from)
            .chain([source.into(), "-o".into(), exe.clone().into()])
            .collect();
        let built = bounded(&compiler, &args);
        assert!(
            built.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&built.stderr)
        );
        let results = dir.join(format!("{name}.bin"));
        let run = bounded(&exe, &[input_path.clone().into(), results.clone().into()]);
        assert!(
            run.status.success(),
            "{name}: {:?} {}",
            run.status,
            String::from_utf8_lossy(&run.stderr)
        );
        let bytes = fs::read(&results).unwrap();
        assert_eq!(bytes.len(), expected.len() * 4, "{name}");
        let mut mismatches = 0usize;
        for (i, (chunk, want)) in bytes.chunks_exact(4).zip(&expected).enumerate() {
            let got = u32::from_le_bytes(chunk.try_into().unwrap());
            if got != *want {
                if exact && mismatches < 8 {
                    eprintln!(
                        "{name}: case {i} op {} inputs {:08x?}: got {got:08x}, oracle {want:08x}",
                        cases[i][0],
                        &cases[i][1..]
                    );
                }
                mismatches += 1;
            }
        }
        if exact {
            assert_eq!(
                mismatches,
                0,
                "{name}: {mismatches} of {} differ",
                expected.len()
            );
            eprintln!("{name}: {} f32 helper results bit-exact", expected.len());
        } else {
            assert!(mismatches > 0, "{name}: mutation was not detected");
            eprintln!(
                "{name}: mutation detected in {mismatches} of {} results",
                expected.len()
            );
        }
    }
    fs::remove_dir_all(dir).unwrap();
}
