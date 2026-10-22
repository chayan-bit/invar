//! Compiled-ahead CUDA on the actual device (ignored by default): `invar
//! build` runs NVRTC; a production `gnet-device --artifact` run loads the
//! image through the Driver API alone, stays exact, and never loads NVRTC,
//! proven by running it with every NVRTC library name resolving to a
//! tripwire that kills the process on load. Typed refusals cover another
//! program, another architecture, an old driver and a driver-rejected image.
//! Run with `cargo test --features cuda --test issue_gnet_cuda_aot_gpu -- --ignored`.
//! The NVRTC tripwire uses Linux shared objects; the refusals run on any host.
#![cfg(feature = "cuda")]
#![forbid(unsafe_code)]

#[path = "support/gnet_cuda_aot.rs"]
mod aot;
#[path = "support/gnet_device.rs"]
mod device;

#[cfg(target_os = "linux")]
use invar::eval::Value;
use invar::general_net::{
    compiled::Target,
    device_host::{Limits, Plan, cuda_aot},
};
use std::{
    path::{Path, PathBuf},
    process::{Command, Output},
};

const PROGRAM: &str = "fn collatz(x: i32) -> i32 { var n = x; var steps = 0; while n != 1 { if n % 2 == 0 { n = n / 2; } else { n = 3 * n + 1; } steps = steps + 1; } return steps; }
fn run(a: i32, b: i32) -> i32 { let p = collatz(a); let q = collatz(b); let r = collatz(a + b); return p + q + r; }";
const OTHER: &str = "fn run(a: i32, b: i32) -> i32 { return a - b; }";
const CONFIG: &str = r#"{"fuel":100000,"transitions":1000000,"trace_events":4000,"quantum":16,"rounds":64,"capacity":2}"#;
#[cfg(target_os = "linux")]
const TRIPWIRE_EXIT: i32 = 97;

struct Lane {
    dir: PathBuf,
}
impl Lane {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("invar-aot-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Self { dir }
    }
    fn write(&self, name: &str, text: &str) -> PathBuf {
        let path = self.dir.join(name);
        std::fs::write(&path, text).unwrap();
        path
    }
    /// A library directory where every name cudarc tries for NVRTC is a
    /// shared object whose constructor exits the process with 97.
    #[cfg(target_os = "linux")]
    fn tripwire(&self) -> PathBuf {
        let lib = self.dir.join("no-nvrtc");
        std::fs::create_dir_all(&lib).unwrap();
        let c = self.write(
            "tripwire.c",
            "#include <unistd.h>\n__attribute__((constructor)) static void tripwire(void) { static const char m[] = \"NVRTC_TRIPWIRE\\n\"; (void)write(2, m, sizeof m - 1); _exit(97); }\n",
        );
        let so = lib.join("libnvrtc.so");
        let status = Command::new("cc")
            .args(["-shared", "-fPIC", "-o"])
            .arg(&so)
            .arg(&c)
            .status()
            .unwrap();
        assert!(status.success(), "tripwire build");
        for name in [
            "libnvrtc64.so",
            "libnvrtc64_12.so",
            "libnvrtc64_124.so",
            "libnvrtc64_124_0.so",
            "libnvrtc64_120_4.so",
            "libnvrtc64_10.so",
            "libnvrtc64_11.so",
            "libnvrtc64_120_0.so",
            "libnvrtc64_9.so",
            "libnvrtc.so.12",
            "libnvrtc.so.11",
            "libnvrtc.so.10",
            "libnvrtc.so.9",
            "libnvrtc.so.1",
        ] {
            std::fs::copy(&so, lib.join(name)).unwrap();
        }
        lib
    }
}
impl Drop for Lane {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn invar(args: &[&Path], no_nvrtc: Option<&Path>) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_invar"));
    command.args(args);
    if let Some(lib) = no_nvrtc {
        command.env("LD_LIBRARY_PATH", lib);
    }
    command.output().unwrap()
}

fn json(output: &Output) -> serde_json::Value {
    serde_json::from_slice(&output.stdout).unwrap_or_else(|_| {
        panic!(
            "status {:?} stdout {} stderr {}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    })
}

fn build(source: &Path, sm: u32, out: &Path) -> serde_json::Value {
    let sm = sm.to_string();
    let output = invar(
        &[
            "build".as_ref(),
            source,
            "--target".as_ref(),
            "cuda".as_ref(),
            "--entry".as_ref(),
            "run".as_ref(),
            "--sm".as_ref(),
            sm.as_ref(),
            "--output".as_ref(),
            out,
        ],
        None,
    );
    assert!(output.status.success(), "{}", json(&output));
    json(&output)
}

fn run(source: &Path, args: &str, artifact: &Path, no_nvrtc: Option<&Path>) -> Output {
    invar(
        &[
            "gnet-device".as_ref(),
            source,
            "run".as_ref(),
            "cuda".as_ref(),
            args.as_ref(),
            CONFIG.as_ref(),
            "--artifact".as_ref(),
            artifact,
        ],
        no_nvrtc,
    )
}

fn sm() -> u32 {
    aot::open().cuda.sm()
}

/// Runs a production image that must be refused before execution with `why`.
fn refused(source: &Path, artifact: &Path, why: &str) {
    let output = run(source, "[3,4]", artifact, None);
    let report = json(&output);
    assert!(!output.status.success(), "{report}");
    assert_eq!(report["placement"]["executed"], false, "{report}");
    assert_eq!(report["placement"]["fallback"], "none");
    assert_eq!(report["placement"]["refusal"], "image");
    let error = report["error"].as_str().unwrap();
    assert!(error.contains(why), "{error}");
    println!("gnet cuda aot refusal: {error}");
}

/// The production run needs no NVRTC: with every NVRTC name a tripwire it
/// completes with the oracle's exact result, while `invar build` under the
/// same environment trips (positive control). The binary links no NVRTC.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires a CUDA device, NVRTC and a C compiler"]
fn issue_gnet_cuda_aot_production_run_never_loads_nvrtc() {
    let lane = Lane::new("tripwire");
    let source = lane.write("p.inv", PROGRAM);
    let artifact = lane.dir.join("p.invgcuda");
    let sm = sm();
    let built = build(&source, sm, &artifact);
    assert_eq!(built["image"]["target"]["sm"], sm);
    assert_eq!(built["image"]["toolchain"]["compiler"], "nvrtc");
    println!("gnet cuda aot build receipt {built}");
    let tripwire = lane.tripwire();

    let control = invar(
        &[
            "build".as_ref(),
            &source,
            "--target".as_ref(),
            "cuda".as_ref(),
            "--entry".as_ref(),
            "run".as_ref(),
            "--sm".as_ref(),
            sm.to_string().as_ref(),
            "--output".as_ref(),
            &lane.dir.join("control.invgcuda"),
        ],
        Some(&tripwire),
    );
    assert_eq!(control.status.code(), Some(TRIPWIRE_EXIT), "{control:?}");
    assert!(String::from_utf8_lossy(&control.stderr).contains("NVRTC_TRIPWIRE"));

    let plan =
        Plan::from_source(PROGRAM.as_bytes(), "run", Target::Cuda, Limits::certified()).unwrap();
    let config = device::config(100_000, 4000, 16, 64, 2);
    for (a, b) in [(6, 7), (7, 1), (3, 10)] {
        let output = run(&source, &format!("[{a},{b}]"), &artifact, Some(&tripwire));
        let report = json(&output);
        assert!(output.status.success(), "{report}");
        assert!(!String::from_utf8_lossy(&output.stderr).contains("NVRTC_TRIPWIRE"));
        let reference = device::reference(&plan, &[Value::I32(a), Value::I32(b)], config);
        let Ok(Value::I32(expected)) = reference.semantic.result else {
            panic!("reference {reference:?}");
        };
        assert_eq!(report["status"], 0, "({a},{b})");
        assert_eq!(report["result"], expected, "({a},{b})");
        assert_eq!(report["source_steps"], reference.semantic.steps_used);
        assert_eq!(report["transitions"], reference.transitions);
        assert_eq!(report["placement"]["target"], "cuda");
        assert_eq!(report["placement"]["fallback"], "none");
        assert_eq!(report["image"]["runtime_compiler"], "none");
        assert_eq!(report["image"]["loaded_via"], "cuModuleLoadData");
        assert_eq!(report["image"]["target"]["sm"], sm);
        assert_eq!(
            report["image"]["container_sha256"],
            built["image"]["container_sha256"]
        );
        println!("gnet cuda aot ({a},{b}) = {expected} with NVRTC unavailable");
    }

    let ldd = Command::new("ldd")
        .arg(env!("CARGO_BIN_EXE_invar"))
        .output()
        .unwrap();
    assert!(ldd.status.success());
    let linked = String::from_utf8_lossy(&ldd.stdout);
    assert!(!linked.contains("nvrtc"), "{linked}");
    assert!(!linked.contains("libcuda"), "{linked}");
    println!("gnet cuda aot: ldd lists no NVRTC or CUDA library; tripwire control exit 97");
}

/// Every refusal is typed, happens before execution, and substitutes nothing.
#[test]
#[ignore = "requires a CUDA device and NVRTC"]
fn issue_gnet_cuda_aot_refusals_are_typed_and_never_recompile() {
    let lane = Lane::new("refusals");
    let source = lane.write("p.inv", PROGRAM);
    let other = lane.write("q.inv", OTHER);
    let artifact = lane.dir.join("p.invgcuda");
    let sm = sm();
    build(&source, sm, &artifact);
    refused(&other, &artifact, "Program(\"source_sha256\")");

    let foreign_sm = if sm == 89 { 86 } else { 89 };
    let foreign = lane.dir.join("foreign.invgcuda");
    build(&source, foreign_sm, &foreign);
    refused(
        &source,
        &foreign,
        &format!("Sm {{ image: {foreign_sm}, device: {sm} }}"),
    );

    let bytes = std::fs::read(&artifact).unwrap();
    let image = cuda_aot::decode(&bytes).unwrap();
    let mut future = image.header().clone();
    future.nvrtc = (99, 0);
    let path = lane.dir.join("future.invgcuda");
    std::fs::write(&path, cuda_aot::encode(&future, image.cubin()).unwrap()).unwrap();
    refused(&source, &path, "Driver { required: 99000");

    let mut corrupt = bytes.clone();
    let last = corrupt.len() - 1;
    corrupt[last] ^= 1;
    std::fs::write(&path, &corrupt).unwrap();
    refused(&source, &path, "Image(Digest)");
}

/// A well-formed container whose CUDA ELF holds no kernel is refused by invar
/// before the driver sees it: the Windows driver faults inside
/// cuModuleLoadData on such an image instead of returning an error.
#[test]
#[ignore = "requires a CUDA device and NVRTC"]
fn issue_gnet_cuda_aot_refuses_a_kernel_less_image_before_the_driver() {
    let lane = Lane::new("kernel-less");
    let source = lane.write("p.inv", PROGRAM);
    let artifact = lane.dir.join("p.invgcuda");
    let sm = sm();
    build(&source, sm, &artifact);
    let bytes = std::fs::read(&artifact).unwrap();
    let image = cuda_aot::decode(&bytes).unwrap();
    let path = lane.dir.join("empty.invgcuda");

    // The same ELF cut down to its null section: every bounds check passes
    // and the architecture matches, but it holds no code.
    let plan =
        Plan::from_source(PROGRAM.as_bytes(), "run", Target::Cuda, Limits::certified()).unwrap();
    let mut empty = image.cubin().to_vec();
    let shoff = u64::from_le_bytes(empty[40..48].try_into().unwrap()) as usize;
    empty[56..58].copy_from_slice(&0u16.to_le_bytes());
    empty[60..62].copy_from_slice(&1u16.to_le_bytes());
    empty[62..64].copy_from_slice(&0u16.to_le_bytes());
    empty.truncate(shoff + 64);
    let header = cuda_aot::Header::of(&plan, sm, image.header().nvrtc, &empty);
    let container = cuda_aot::encode(&header, &empty).unwrap();
    std::fs::write(&path, &container).unwrap();
    assert!(matches!(
        cuda_aot::decode(&container),
        Err(cuda_aot::Error::Elf("no executable kernel section"))
    ));
    refused(&source, &path, "no executable kernel section");
}
