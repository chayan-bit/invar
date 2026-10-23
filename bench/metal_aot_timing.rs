//! Timing on a real Apple GPU, not a gate: the first dispatch of a kernel from
//! a compiled-ahead library against a runtime MSL compile, and warm repeated
//! calls, for the map, scan and general-net routes. Prints one JSON line per
//! row. Kernel constants are unique per run so a "first-ever" runtime compile
//! cannot be served by the macOS Metal shader cache.
//! Run: `cargo test --release --features metal --bench metal_aot_timing -- --ignored --nocapture --test-threads 1`.
#![cfg(all(feature = "metal", target_os = "macos"))]

use invar::{admission, check, device_ir, front, mir, plan::Execution};
use invar_device::{Provenance, Verified};
use invar_metal::MetalDevice;
use std::time::Instant;

const LENGTH: u32 = 1_000_000;
const COLD: u32 = 9;
const WARM: usize = 50;

fn map(k: u32) -> String {
    format!(
        "fn main() -> i32 {{ let xs: Array<u32> = [1u32,2u32,3u32]; \
         var ys: Array<u32> = filled_u32(len(xs),0u32); \
         for i in 0u32 .. len(xs) {{ ys[i] = wrapping_add(xs[i],{k}u32); }} \
         if ys[2u32] == 8u32 {{ return 0; }} return 1; }}"
    )
}

fn scan(k: u32) -> String {
    format!(
        "fn main() -> i32 {{ let aa: Array<u32> = [2u32,3u32,5u32]; var state: u32 = 17u32; \
         var out: Array<u32> = filled_u32(len(aa),0u32); \
         for i in 0u32 .. len(aa) {{ state = wrapping_add(wrapping_mul(aa[i],state),{k}u32); out[i] = state; }} \
         if out[2u32] == 683u32 {{ return 0; }} return 1; }}"
    )
}

/// The ABI5 general-net dispatch boundary with a one-pipeline kernel.
fn gnet(k: u32) -> String {
    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\
         kernel void invar_gnet_children_v5(device const uint* in [[buffer(0)]], device uint* out [[buffer(1)]], uint i [[thread_index_in_threadgroup]]) {{\n\
         if (i < 8u) {{ out[i] = in[i] + {k}u; }}\n}}\n"
    )
}

/// A constant no earlier run used, so the OS shader cache cannot have it;
/// `lane` keeps the two paths' kernels apart.
fn nonce(sample: u32, lane: u32) -> u32 {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as u32;
    (seconds % 1_000_000) * 100 + sample * 4 + lane + 3
}

/// A route's kernel `k`, planned once so calls time only the dispatch.
enum Kernel {
    Plan(Box<Verified>),
    Gnet(String),
}

fn kernel(route: &str, k: u32, caps: invar_device::Capabilities) -> Kernel {
    let source = match route {
        "gnet" => return Kernel::Gnet(gnet(k)),
        "map" => map(k),
        _ => scan(k),
    };
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    let report = admission::analyze(source.as_bytes(), &program).unwrap();
    let region = report
        .regions()
        .iter()
        .find(|r| {
            matches!(
                r.execution(),
                Execution::Map(_) | Execution::FusedProducer(_) | Execution::Scan(_)
            )
        })
        .expect("a device region");
    let mut ir = device_ir::from_region(region, LENGTH).unwrap();
    device_ir::plan_metal(&mut ir).unwrap();
    let origin = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    Kernel::Plan(Box::new(invar_device::verify(ir, origin, caps).unwrap()))
}

/// Dispatch `kernel` once on `device` and read the result back.
fn dispatch(device: &mut MetalDevice, kernel: &Kernel, values: &[u32]) {
    match kernel {
        Kernel::Gnet(msl) => {
            let mut bytes = [64usize; 12];
            bytes[0] = 128;
            bytes[1] = 128;
            bytes[3] = 48;
            let mut work = device.prepare_gnet(msl, bytes, &[0; 8], [0; 6]).unwrap();
            work.submit().unwrap();
            while work.completion() != invar_metal::gnet::Completion::Complete {
                std::thread::yield_now();
            }
            work.read().unwrap();
        }
        Kernel::Plan(plan) if plan.ir().collective.is_some() => {
            device.run_collective(plan, &[values], 1).unwrap();
        }
        Kernel::Plan(plan) => {
            device.run_map(plan, values).unwrap();
        }
    }
}

fn ms(begun: Instant) -> f64 {
    begun.elapsed().as_secs_f64() * 1000.0
}

fn row(route: &str, path: &str, phase: &str, mut samples: Vec<f64>) {
    samples.sort_by(f64::total_cmp);
    println!(
        "{}",
        serde_json::json!({"route": route, "path": path, "phase": phase,
            "median_ms": samples[samples.len() / 2], "min_ms": samples[0],
            "max_ms": samples[samples.len() - 1], "samples": samples.len()})
    );
}

/// Time one fresh device's first dispatch, then `WARM` cached calls on it.
fn first_then_warm(
    open: &dyn Fn() -> MetalDevice,
    kernel: &Kernel,
    values: &[u32],
) -> (f64, Vec<f64>, MetalDevice) {
    let begun = Instant::now();
    let mut device = open();
    dispatch(&mut device, kernel, values);
    let first = ms(begun);
    let warmed = (0..WARM)
        .map(|_| {
            let begun = Instant::now();
            dispatch(&mut device, kernel, values);
            ms(begun)
        })
        .collect();
    (first, warmed, device)
}

const ROUTES: [&str; 3] = ["map", "scan", "gnet"];

#[test]
#[ignore = "timing on a real Apple GPU"]
fn runtime_source_first_dispatch_and_warm_calls() {
    let values = (0..LENGTH).collect::<Vec<_>>();
    let open = || MetalDevice::open_runtime_source().unwrap();
    let caps = open().capabilities();
    for route in ROUTES {
        let (mut first, mut cached, mut warmed) = (Vec::new(), Vec::new(), Vec::new());
        for sample in 0..COLD {
            let kernel = kernel(route, nonce(sample, 0), caps);
            let (ever, calls, device) = first_then_warm(&open, &kernel, &values);
            drop(device);
            first.push(ever);
            warmed.extend(calls);
            cached.push(first_then_warm(&open, &kernel, &values).0);
        }
        row(route, "runtime_source", "first_dispatch_first_ever", first);
        row(route, "runtime_source", "first_dispatch_os_cached", cached);
        row(route, "runtime_source", "warm_call", warmed);
    }
}

#[test]
#[ignore = "timing on a real Apple GPU with the Metal Toolchain"]
fn compiled_ahead_first_dispatch_and_warm_calls() {
    use invar::metal_aot::{self, Route};
    let values = (0..LENGTH).collect::<Vec<_>>();
    let caps = MetalDevice::open().unwrap().capabilities();
    let image = |route: &str, k: u32| {
        let bytes = if route == "gnet" {
            let msl = gnet(k);
            let functions = metal_aot::msl_entry_points(&msl);
            metal_aot::build::build(Route::Gnet, "timing", b"timing", &[(msl, functions)])
        } else {
            let source = if route == "map" { map(k) } else { scan(k) };
            let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
            let program = mir::lower(&checked).unwrap();
            let kernels = metal_aot::run_kernels(source.as_bytes(), &program).unwrap();
            metal_aot::build::build(Route::Run, "main", source.as_bytes(), &kernels)
        };
        metal_aot::decode(&bytes.unwrap()).unwrap()
    };
    for route in ROUTES {
        let (mut first, mut cached, mut warmed) = (Vec::new(), Vec::new(), Vec::new());
        for sample in 0..COLD {
            let k = nonce(sample, 1);
            let (built, kernel) = (image(route, k), kernel(route, k, caps));
            let open = || {
                let mut device = MetalDevice::open().unwrap();
                built.install(&mut device).unwrap();
                device
            };
            let (ever, calls, device) = first_then_warm(&open, &kernel, &values);
            assert_eq!(device.cache_report().source_compiles, 0);
            drop(device);
            first.push(ever);
            warmed.extend(calls);
            cached.push(first_then_warm(&open, &kernel, &values).0);
        }
        row(route, "compiled_ahead", "first_dispatch_first_ever", first);
        row(route, "compiled_ahead", "first_dispatch_os_cached", cached);
        row(route, "compiled_ahead", "warm_call", warmed);
    }
}

/// Paired wall-clock samples: identical MSL/input/geometry, both pipelines
/// cached, AB/BA order reversed each round. Keep individual samples so the
/// independent unit for uncertainty is a round, rather than 450 correlated
/// calls. The original tests above retain unique-kernel first-dispatch timing.
#[test]
#[ignore = "paired timing on a real Apple GPU with the Metal Toolchain"]
fn paired_identical_source_warm_calls() {
    use invar::metal_aot;
    use sha2::{Digest, Sha256};
    use std::process::Command;

    assert_eq!(std::env::var("INVAR_BENCH_FOREGROUND").as_deref(), Ok("1"));
    let variants = std::env::var("INVAR_METAL_TIMING_VARIANTS").unwrap_or("pinned".into());
    let values = (0..LENGTH).collect::<Vec<_>>();
    let caps = MetalDevice::open().unwrap().capabilities();
    let scratch = std::env::temp_dir().join(format!("invar-metal-timing-{}", std::process::id()));
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new()
        .mode(0o700)
        .create(&scratch)
        .unwrap();
    let run_tool = |args: &[&str]| {
        let out = Command::new("xcrun").args(args).output().unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    };
    for route in ROUTES {
        for variant in variants.split(',') {
            for sample in 0..COLD {
                let kernel = kernel(route, nonce(sample, 2), caps);
                let msl = match &kernel {
                    Kernel::Plan(plan) => invar_device::emit_msl(plan).unwrap(),
                    Kernel::Gnet(msl) => msl.clone(),
                };
                let input = scratch.join("kernel.metal");
                let air = scratch.join("kernel.air");
                let lib = scratch.join("kernel.metallib");
                std::fs::write(&input, &msl).unwrap();
                let mut options = metal_aot::options().to_vec();
                match variant {
                    "pinned" => (),
                    "O2" => options.push("-O2".into()),
                    "O3" => options.push("-O3".into()),
                    "Os" => options.push("-Os".into()),
                    "record" => options.push("-frecord-sources".into()),
                    "debug" => options.push("-gline-tables-only".into()),
                    "host_os" => {
                        let version = Command::new("sw_vers")
                            .arg("-productVersion")
                            .output()
                            .unwrap();
                        options[2] = format!(
                            "-mmacosx-version-min={}",
                            String::from_utf8(version.stdout).unwrap().trim()
                        );
                    }
                    _ => panic!("unknown timing variant: {variant}"),
                }
                let mut args = vec!["-sdk", "macosx", "metal"];
                args.extend(options.iter().map(String::as_str));
                args.extend(["-c", input.to_str().unwrap(), "-o", air.to_str().unwrap()]);
                run_tool(&args);
                run_tool(&[
                    "-sdk",
                    "macosx",
                    "metallib",
                    air.to_str().unwrap(),
                    "-o",
                    lib.to_str().unwrap(),
                ]);
                if sample == 0
                    && let Some(dir) = std::env::var_os("INVAR_METAL_TIMING_ARTIFACTS")
                {
                    let dir = std::path::PathBuf::from(dir);
                    std::fs::create_dir_all(&dir).unwrap();
                    for (file, ext) in [(&input, "metal"), (&air, "air"), (&lib, "metallib")] {
                        std::fs::copy(file, dir.join(format!("{route}-{variant}.{ext}"))).unwrap();
                    }
                }
                let bytes = std::fs::read(&lib).unwrap();
                let digest: [u8; 32] = Sha256::digest(msl.as_bytes()).into();
                let digest_hex = digest
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>();
                let functions = metal_aot::msl_entry_points(&msl);
                let mut devices = [
                    MetalDevice::open_runtime_source().unwrap(),
                    MetalDevice::open().unwrap(),
                ];
                devices[1]
                    .load_library(
                        digest,
                        &bytes,
                        &functions.iter().map(String::as_str).collect::<Vec<_>>(),
                    )
                    .unwrap();
                // First dispatch is excluded: identical MSL may share the OS's
                // shader cache, so it is not independent first-ever evidence.
                for lane in [sample as usize % 2, 1 - sample as usize % 2] {
                    dispatch(&mut devices[lane], &kernel, &values);
                }
                for round in 0..10 {
                    let mut times: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
                    for lane in [
                        (sample as usize + round) % 2,
                        1 - (sample as usize + round) % 2,
                    ] {
                        for _ in 0..WARM {
                            let start = Instant::now();
                            dispatch(&mut devices[lane], &kernel, &values);
                            times[lane].push(ms(start));
                        }
                    }
                    println!(
                        "{}",
                        serde_json::json!({"route":route,"variant":variant,"sample":sample,"round":round,
                        "first_lane":(sample as usize + round)%2,"options":options,"msl_sha256":digest_hex,"runtime_ms":times[0],"aot_ms":times[1]})
                    );
                }
                assert_eq!(devices[0].cache_report().source_compiles, 1);
                assert_eq!(devices[1].cache_report().source_compiles, 0);
                assert_eq!(
                    devices[0].cache_report().pipeline_creations,
                    devices[1].cache_report().pipeline_creations
                );
            }
        }
    }
    std::fs::remove_dir_all(scratch).unwrap();
}
