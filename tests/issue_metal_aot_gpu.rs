//! Compiled-ahead Metal on a real Apple GPU: fresh processes run from built
//! libraries with the source-compiler tripwire armed, results match the
//! reference and the developer source path bit for bit, refusals never fall
//! back, and every route reuses pipelines from one bounded cache.
//! Needs the `metal` feature, an Apple7+ GPU and Apple's Metal Toolchain.
#![cfg(all(feature = "metal", target_os = "macos"))]

use invar::{
    admission, check, device_ir, eval, front,
    metal_aot::{self, Route},
    mir,
    plan::Execution,
    schedule::metal::{self as scheduler, Kernels},
};
use invar_device::{Provenance, Verified};
use invar_metal::{MAX_PIPELINES, MetalDevice, SOURCE_TRIPWIRE};
use std::{path::PathBuf, process::Command};

static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

const MAP: &str = "fn main() -> i32 { let xs: Array<u32> = [1u32,2u32,3u32]; \
    var ys: Array<u32> = filled_u32(len(xs),0u32); \
    for i in 0u32 .. len(xs) { ys[i] = wrapping_add(xs[i],5u32); } \
    if ys[2u32] == 8u32 { return 0; } return 1; }";
const ZIP_DAG: &str = "fn main() -> i32 { let xs: Array<u32> = [1u32,2u32,3u32]; \
    let ys: Array<u32> = [4u32,5u32,6u32]; let zs: Array<u32> = [7u32,8u32,9u32]; \
    var a: Array<u32> = filled_u32(len(xs),0u32); \
    for i in 0u32 .. len(xs) { a[i] = wrapping_add(xs[i],ys[i]); } \
    let first: Array<u32> = move a; var b: Array<u32> = filled_u32(len(first),0u32); \
    for j in 0u32 .. len(first) { b[j] = wrapping_mul(first[j],zs[j]); } \
    if b[2u32] == 81u32 { return 0; } return 1; }";
const ZIP_MAP: &str = "fn main() -> i32 { let xs: Array<u32> = [1u32,2u32,3u32]; \
    let zs: Array<u32> = [7u32,8u32,4294967295u32]; var ys: Array<u32> = filled_u32(len(xs),0u32); \
    for i in 0u32 .. len(xs) { ys[i] = wrapping_mul(zs[i],xs[i]); } \
    if ys[2u32] != 4294967293u32 { return 1; } return 0; }";
const SUM: &str = "fn main() -> i32 { let xs: Array<u32> = [1u32,2u32,3u32]; var acc: u32 = 7u32; for i in 0u32 .. len(xs) { acc = wrapping_add(acc,xs[i]); } if acc == 13u32 { return 0; } return 1; }";
/// A sort: auto runs it on the CPU alone, without opening the device.
const SORT: &str = "fn main() -> i32 { var xs: Array<u32> = [2u32,1u32]; sort(xs); if xs[0u32] == 1u32 { return 0; } return 1; }";
const PRODUCT: &str = "fn main() -> i32 { let xs: Array<u32> = [2u32,3u32,5u32]; var acc: u32 = 3u32; for i in 0u32 .. len(xs) { acc = wrapping_mul(acc,xs[i]); } if acc == 90u32 { return 0; } return 1; }";
const INCLUSIVE: &str = "fn main() -> i32 { let aa: Array<u32> = [2u32,3u32,5u32]; let bb: Array<u32> = [7u32,11u32,13u32]; var state: u32 = 17u32; var out: Array<u32> = filled_u32(len(aa),0u32); for i in 0u32 .. len(aa) { state = wrapping_add(wrapping_mul(aa[i],state),bb[i]); out[i] = state; } if out[2u32] == 683u32 { return 0; } return 1; }";
const EXCLUSIVE: &str = "fn main() -> i32 { let aa: Array<u32> = [2u32,3u32,5u32]; let bb: Array<u32> = [7u32,11u32,13u32]; var state: u32 = 17u32; var out: Array<u32> = filled_u32(len(aa),0u32); for i in 0u32 .. len(aa) { out[i] = state; state = wrapping_add(wrapping_mul(aa[i],state),bb[i]); } if out[0u32] == 17u32 { return 0; } return 1; }";
const PAIR: &str = include_str!("../examples/pair-reduction.inv");
const GNET: &str = "fn leaf(x:u32)->u32{return wrapping_add(x,1u32);} fn entry(x:u32)->u32{let a=leaf(x);let b=leaf(x);return wrapping_add(a,b);}";
const GNET_CONFIG: &str =
    r#"{"fuel":1000,"transitions":10000,"trace_events":2000,"quantum":2,"rounds":1,"capacity":2}"#;

fn lock() -> std::sync::MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

fn lowered(source: &str) -> (check::CheckedProgram, mir::Program) {
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    (checked, program)
}

/// The container `invar build --target metal` writes, built in process.
fn prebuilt(source: &str) -> metal_aot::Image {
    let (_, program) = lowered(source);
    let kernels = metal_aot::run_kernels(source.as_bytes(), &program).unwrap();
    let bytes = metal_aot::build::build(Route::Run, "main", source.as_bytes(), &kernels)
        .expect("Apple's Metal Toolchain builds the container");
    metal_aot::decode(&bytes).unwrap()
}

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("invar-metal-aot-gpu-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Run the CLI with the tripwire armed: any `newLibraryWithSource` aborts.
fn armed(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_invar"))
        .env(SOURCE_TRIPWIRE, "1")
        .env_remove(scheduler::RUNTIME_SOURCE_ENV)
        .args(args)
        .output()
        .unwrap()
}

fn json(output: &std::process::Output) -> serde_json::Value {
    serde_json::from_slice(&output.stdout).unwrap_or_else(|_| {
        panic!(
            "stdout {} stderr {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    })
}

#[test]
#[ignore = "requires a real Apple7+ Metal GPU and the Metal Toolchain"]
fn fresh_processes_run_built_libraries_with_the_source_tripwire_armed() {
    let _gpu = lock();
    let dir = scratch();
    for (index, source) in [
        MAP, ZIP_DAG, ZIP_MAP, SUM, PRODUCT, INCLUSIVE, EXCLUSIVE, PAIR,
    ]
    .into_iter()
    .enumerate()
    {
        let path = dir.join(format!("p{index}.inv"));
        let library = dir.join(format!("p{index}.metallibc"));
        std::fs::write(&path, source).unwrap();
        let (path, library) = (path.to_str().unwrap(), library.to_str().unwrap());
        let built = armed(&["build", path, "--target", "metal", "--output", library]);
        assert!(built.status.success(), "{}", json(&built));
        assert_eq!(json(&built)["library"]["runtime_compiler"], "none");
        let run = armed(&[
            "run",
            path,
            "--engine",
            "metal",
            "--artifact",
            library,
            "--format",
            "json",
        ]);
        assert!(run.status.success(), "{source}: {}", json(&run));
        let (checked, _) = lowered(source);
        let reference = match eval::observe(&checked, 1_000_000).result.unwrap() {
            eval::Value::I32(n) => n,
            other => panic!("{other:?}"),
        };
        assert_eq!(json(&run)["result"]["value"], reference, "{source}");
        let auto = armed(&[
            "run",
            path,
            "--engine",
            "auto",
            "--artifact",
            library,
            "--format",
            "json",
        ]);
        assert!(auto.status.success(), "{source}: {}", json(&auto));
    }
    // The general-net route under the same tripwire.
    let gnet = dir.join("gnet.inv");
    std::fs::write(&gnet, GNET).unwrap();
    let library = dir.join("gnet.metallibc");
    let (gnet, library) = (gnet.to_str().unwrap(), library.to_str().unwrap());
    let built = armed(&[
        "build", gnet, "--target", "metal", "--entry", "entry", "--output", library,
    ]);
    assert!(built.status.success(), "{}", json(&built));
    let device = armed(&[
        "gnet-device",
        gnet,
        "entry",
        "metal",
        "[5]",
        GNET_CONFIG,
        "--artifact",
        library,
    ]);
    assert!(device.status.success(), "{}", json(&device));
    // leaf(5) + leaf(5), as issue_gnet_04_host_cli checks on the CPU model.
    assert_eq!(json(&device)["result"], 12);
    assert_eq!(json(&device)["image"]["runtime_compiler"], "none");
    // The tripwire is live: the developer source path aborts under it.
    let path = dir.join("p0.inv");
    let tripped = Command::new(env!("CARGO_BIN_EXE_invar"))
        .env(SOURCE_TRIPWIRE, "1")
        .env(scheduler::RUNTIME_SOURCE_ENV, "1")
        .args(["run", path.to_str().unwrap(), "--engine", "metal"])
        .output()
        .unwrap();
    assert!(!tripped.status.success());
    assert!(String::from_utf8_lossy(&tripped.stderr).contains(SOURCE_TRIPWIRE));
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The run library also holds the A1 family kernels, whose MSL
/// depends on the dimensions the run reaches, so these programs run forced
/// Metal from a built library without the developer source flag.
#[test]
#[ignore = "requires a real Apple7+ Metal GPU and the Metal Toolchain"]
fn a1_family_programs_run_from_built_libraries_with_the_tripwire_armed() {
    let _gpu = lock();
    let dir = scratch();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for (index, path) in [
        "demo/a1-grid.inv",
        "demo/a1-segmented.inv",
        "tests/a1_tensor/scalar.inv",
        "tests/a1_tensor/direct.inv",
        "tests/a1_tensor/tc.inv",
        "tests/fixtures/a1-indexed/gather.inv",
        "tests/fixtures/a1-indexed/scatter.inv",
        "tests/fixtures/a1-indexed/histogram.inv",
        "tests/fixtures/a1-indexed/affine.inv",
    ]
    .into_iter()
    .enumerate()
    {
        let source = std::fs::read_to_string(root.join(path)).unwrap();
        let library = dir.join(format!("a1-{index}.metallibc"));
        let (file, library) = (root.join(path), library.to_str().unwrap().to_owned());
        let file = file.to_str().unwrap();
        let built = armed(&["build", file, "--target", "metal", "--output", &library]);
        assert!(built.status.success(), "{path}: {}", json(&built));
        for engine in ["metal", "auto"] {
            let run = armed(&[
                "run",
                file,
                "--engine",
                engine,
                "--artifact",
                &library,
                "--format",
                "json",
            ]);
            assert!(run.status.success(), "{path} {engine}: {}", json(&run));
            let (checked, _) = lowered(&source);
            let reference = match eval::observe(&checked, 1_000_000).result.unwrap() {
                eval::Value::I32(n) => n,
                other => panic!("{other:?}"),
            };
            assert_eq!(json(&run)["result"]["value"], reference, "{path} {engine}");
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[ignore = "requires a real Apple7+ Metal GPU and the Metal Toolchain"]
fn a_cached_calibration_never_hides_a_refused_library() {
    let _gpu = lock();
    let (_, program) = lowered(MAP);
    let (valid, stale) = (prebuilt(MAP), prebuilt(SUM));
    // A valid run caches the process-wide calibration.
    scheduler::run_auto_source(
        MAP.as_bytes(),
        &program,
        1_000_000,
        Kernels::Prebuilt(&valid),
    )
    .unwrap();
    // Every later refusal still surfaces, instead of a quiet CPU result.
    for _ in 0..2 {
        let refused = scheduler::run_auto_source(
            MAP.as_bytes(),
            &program,
            1_000_000,
            Kernels::Prebuilt(&stale),
        );
        assert_eq!(
            refused.err().map(|e| e.code),
            Some("DEVICE_TOOLCHAIN_MISMATCH")
        );
    }
}

#[test]
#[ignore = "requires a real Apple7+ Metal GPU and the Metal Toolchain"]
fn refusals_are_typed_and_never_compile_source() {
    let _gpu = lock();
    let dir = scratch();
    let path = dir.join("map.inv");
    std::fs::write(&path, MAP).unwrap();
    let path = path.to_str().unwrap();
    // No library: production refuses; nothing runs on the GPU.
    let missing = armed(&["run", path, "--engine", "metal", "--format", "json"]);
    let report = json(&missing);
    assert_eq!(
        report["diagnostics"][0]["code"], "METAL_UNAVAILABLE",
        "{report}"
    );
    assert!(report.to_string().contains("--artifact"));
    // Stale: a library built from another program.
    let other = dir.join("sum.inv");
    std::fs::write(&other, SUM).unwrap();
    let stale = dir.join("sum.metallibc");
    let stale = stale.to_str().unwrap();
    assert!(
        armed(&[
            "build",
            other.to_str().unwrap(),
            "--target",
            "metal",
            "--output",
            stale
        ])
        .status
        .success()
    );
    let refused = armed(&[
        "run",
        path,
        "--engine",
        "metal",
        "--artifact",
        stale,
        "--format",
        "json",
    ]);
    let report = json(&refused);
    assert_eq!(
        report["diagnostics"][0]["code"],
        "DEVICE_TOOLCHAIN_MISMATCH"
    );
    assert!(report.to_string().contains("source_sha256"), "{report}");
    // Auto refuses a supplied stale library too, instead of running on the CPU.
    let report = json(&armed(&[
        "run",
        path,
        "--engine",
        "auto",
        "--artifact",
        stale,
        "--format",
        "json",
    ]));
    assert_eq!(
        report["diagnostics"][0]["code"], "DEVICE_TOOLCHAIN_MISMATCH",
        "{report}"
    );
    // Also where auto runs on the CPU alone, never opening the device.
    let sort = dir.join("sort.inv");
    std::fs::write(&sort, SORT).unwrap();
    let report = json(&armed(&[
        "run",
        sort.to_str().unwrap(),
        "--engine",
        "auto",
        "--artifact",
        stale,
        "--format",
        "json",
    ]));
    assert_eq!(
        report["diagnostics"][0]["code"], "DEVICE_TOOLCHAIN_MISMATCH",
        "{report}"
    );
    assert!(report.to_string().contains("source_sha256"), "{report}");
    // Corrupt: one flipped byte inside a library.
    let bytes = std::fs::read(stale).unwrap();
    let mut corrupt = bytes.clone();
    let last = corrupt.len() - 9;
    corrupt[last] ^= 0x40;
    let corrupt_path = dir.join("corrupt.metallibc");
    metal_aot::build::write_atomic(&corrupt_path, &corrupt).unwrap();
    let report = json(&armed(&[
        "run",
        other.to_str().unwrap(),
        "--engine",
        "metal",
        "--artifact",
        corrupt_path.to_str().unwrap(),
        "--format",
        "json",
    ]));
    assert!(report.to_string().contains("Digest"), "{report}");
    // A library missing a reached kernel: refused at the region, no compile.
    let (_, program) = lowered(MAP);
    let probe_only = vec![(
        invar_device::PROBE_MSL.to_owned(),
        vec![invar_device::PROBE_FUNCTION.to_owned()],
    )];
    let partial = metal_aot::decode(
        &metal_aot::build::build(Route::Run, "main", MAP.as_bytes(), &probe_only).unwrap(),
    )
    .unwrap();
    let mut device = scheduler::open_device(MAP.as_bytes(), Kernels::Prebuilt(&partial)).unwrap();
    let run = scheduler::run_source_with_device(MAP.as_bytes(), &program, 1_000_000, &mut device)
        .unwrap();
    let Err(eval::EvalError::Tool { code, message }) = run.observation.result else {
        panic!("a missing kernel must refuse: {:?}", run.observation.result);
    };
    assert_eq!(code, "DEVICE_TOOLCHAIN_MISMATCH");
    assert!(message.contains("Missing"), "{message}");
    assert_eq!(device.cache_report().source_compiles, 0);
    // Missing entry point: the header claims functions the library lacks.
    let mut wrong = MetalDevice::open().unwrap();
    let (library, bytes) = partial.libraries().next().unwrap();
    let refused = wrong.load_library(library.msl_sha256, bytes, &["invar_map_v1"]);
    assert!(
        matches!(
            refused,
            Err(invar_metal::Error::Library(
                invar_metal::LibraryError::EntryPoint(_)
            ))
        ),
        "{refused:?}"
    );
    // Bytes Metal cannot parse behind a well-formed metallib header.
    let mut garbage = bytes.to_vec();
    for byte in &mut garbage[88..] {
        *byte = 0xa5;
    }
    let refused = wrong.load_library([7; 32], &garbage, &["invar_probe_v1"]);
    assert!(
        matches!(
            refused,
            Err(invar_metal::Error::Library(
                invar_metal::LibraryError::Rejected(_)
            ))
        ),
        "{refused:?}"
    );
    assert_eq!(wrong.cache_report().source_compiles, 0);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Verified plans for every device region of `source` at `length`.
fn plans(source: &str, length: u32, device: &MetalDevice) -> Vec<Verified> {
    let (_, program) = lowered(source);
    let report = admission::analyze(source.as_bytes(), &program).unwrap();
    report
        .regions()
        .iter()
        .filter(|r| {
            matches!(
                r.execution(),
                Execution::Map(_)
                    | Execution::FusedProducer(_)
                    | Execution::DagProducer(_)
                    | Execution::Reduction(_)
                    | Execution::PairReduction(_)
                    | Execution::Scan(_)
            )
        })
        .map(|region| {
            let mut ir = device_ir::from_region(region, length).unwrap();
            device_ir::plan_metal(&mut ir).unwrap();
            let origin = Provenance {
                source_identity: ir.source_identity,
                graph_revision: ir.graph_revision,
            };
            invar_device::verify(ir, origin, device.capabilities()).unwrap()
        })
        .collect()
}

/// Adversarial u32 input: wrap boundaries, then a fixed LCG.
fn input(length: usize, salt: u32) -> Vec<u32> {
    let edges = [0, 1, u32::MAX, u32::MAX - 1, 0x8000_0000, 0x7fff_ffff];
    let mut state = 0x9e37_79b9u32 ^ salt;
    (0..length)
        .map(|i| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            edges.get(i).copied().unwrap_or(state)
        })
        .collect()
}

/// Every output word, final state and prefix of one dispatch.
fn dispatch(device: &mut MetalDevice, plan: &Verified, length: usize) -> (Vec<Vec<u32>>, bool) {
    let ir = plan.ir();
    if let Some(collective) = &ir.collective {
        let inputs = (0..collective.inputs.len())
            .map(|k| input(length, k as u32))
            .collect::<Vec<_>>();
        let slices = inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let r = device.run_collective(plan, &slices, 0xdead_beef).unwrap();
        let pair = r.pair.map_or_else(Vec::new, Vec::from);
        (vec![vec![r.final_state], r.prefixes, pair], r.cache_hit)
    } else {
        let count = ir.dag.as_ref().map_or(1, |dag| dag.inputs.len());
        let inputs = (0..count)
            .map(|k| input(length, k as u32))
            .collect::<Vec<_>>();
        let slices = inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let r = device.run_fused_map(plan, &slices).unwrap();
        (r.outputs, r.cache_hit)
    }
}

#[test]
#[ignore = "requires a real Apple7+ Metal GPU and the Metal Toolchain"]
fn compiled_ahead_and_runtime_source_kernels_agree_bit_for_bit() {
    let _gpu = lock();
    let mut runtime = MetalDevice::open_runtime_source().unwrap();
    let mut compared = 0;
    for source in [
        MAP, ZIP_DAG, ZIP_MAP, SUM, PRODUCT, INCLUSIVE, EXCLUSIVE, PAIR,
    ] {
        let image = prebuilt(source);
        assert_eq!(
            prebuilt(source),
            image,
            "two private build directories must give byte-identical containers"
        );
        let mut aot = scheduler::open_device(source.as_bytes(), Kernels::Prebuilt(&image)).unwrap();
        for length in [1u32, 7, 256, 257, 4097, 100_003, 1_000_000] {
            for plan in plans(source, length, &aot) {
                let (ahead, _) = dispatch(&mut aot, &plan, length as usize);
                let (source_built, _) = dispatch(&mut runtime, &plan, length as usize);
                assert_eq!(ahead, source_built, "{source} at {length}");
                compared += 1;
            }
        }
        assert_eq!(aot.cache_report().source_compiles, 0);
    }
    assert_eq!(compared, 8 * 7);
    // The map oracle, independently of both devices.
    let image = prebuilt(MAP);
    let mut aot = scheduler::open_device(MAP.as_bytes(), Kernels::Prebuilt(&image)).unwrap();
    let plan = &plans(MAP, 4097, &aot)[0];
    let values = input(4097, 0);
    let out = aot.run_map(plan, &values).unwrap().values;
    let oracle = values.iter().map(|v| v.wrapping_add(5)).collect::<Vec<_>>();
    assert_eq!(out, oracle);
}

/// A strict-f32 probe through the ABI5 dispatch boundary: contraction-off
/// pragma, fast math off. Each lane computes `a*b+c` and `a*b`.
const F32_MSL: &str = "#include <metal_stdlib>\nusing namespace metal;\n#pragma METAL fp contract(off)\n\
kernel void invar_gnet_children_v5(device const uint* in [[buffer(0)]], device uint* out [[buffer(1)]], uint i [[thread_index_in_threadgroup]]) {\n\
  if (i < 8u) {\n    float a = as_type<float>(in[3u*i]); float b = as_type<float>(in[3u*i+1u]); float c = as_type<float>(in[3u*i+2u]);\n\
    out[2u*i] = as_type<uint>(a * b + c);\n    out[2u*i+1u] = as_type<uint>(a * b);\n  }\n}\n";

fn f32_cases() -> [[f32; 3]; 8] {
    let up = f32::from_bits(0x3f80_0001); // 1 + 2^-23
    let down = f32::from_bits(0x3f7f_fffe); // 1 - 2^-23
    let min_normal = f32::MIN_POSITIVE;
    [
        [up, down, -1.0],              // fused: -2^-46, separate: 0
        [min_normal, 0.5, 0.0],        // subnormal product
        [f32::from_bits(1), 1.0, 0.0], // subnormal input
        [3.0, 1.0 / 3.0, -1.0],        // rounding-sensitive
        [-0.0, 1.0, 0.0],              // signed zero
        [1.5, 2.5, 0.25],              // exact
        [1.0e30, 1.0e10, f32::NEG_INFINITY],
        [f32::INFINITY, 0.0, 1.0], // NaN
    ]
}

/// Dispatch the f32 probe on `device` and return its sixteen output words.
fn run_f32(device: &mut MetalDevice) -> Vec<u32> {
    let words = f32_cases()
        .iter()
        .flatten()
        .map(|v| v.to_bits())
        .collect::<Vec<_>>();
    let input = words
        .chunks(2)
        .map(|w| u64::from(w[0]) | (u64::from(w[1]) << 32))
        .collect::<Vec<_>>();
    let mut bytes = [64usize; 12];
    bytes[0] = 128;
    bytes[1] = 128;
    bytes[3] = 48;
    let mut work = device.prepare_gnet(F32_MSL, bytes, &input, [0; 6]).unwrap();
    work.submit().unwrap();
    let begun = std::time::Instant::now();
    while work.completion() != invar_metal::gnet::Completion::Complete {
        assert!(begun.elapsed().as_secs() < 10);
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let out = work.read().unwrap();
    out[1][..8]
        .iter()
        .flat_map(|w| [*w as u32, (*w >> 32) as u32])
        .collect()
}

#[test]
#[ignore = "requires a real Apple7+ Metal GPU and the Metal Toolchain"]
fn pinned_options_keep_strict_f32_contraction_off_and_subnormal_behaviour() {
    let _gpu = lock();
    let functions = metal_aot::msl_entry_points(F32_MSL);
    let bytes = metal_aot::build::build(
        Route::Gnet,
        "f32",
        b"strict-f32 probe",
        &[(F32_MSL.to_owned(), functions.clone())],
    )
    .unwrap();
    assert_eq!(
        bytes,
        metal_aot::build::build(
            Route::Gnet,
            "f32",
            b"strict-f32 probe",
            &[(F32_MSL.to_owned(), functions)],
        )
        .unwrap(),
        "strict-f32 containers must also be byte-reproducible"
    );
    let image = metal_aot::decode(&bytes).unwrap();
    let mut aot = MetalDevice::open().unwrap();
    image.install(&mut aot).unwrap();
    let mut runtime = MetalDevice::open_runtime_source().unwrap();
    let ahead = run_f32(&mut aot);
    let source_built = run_f32(&mut runtime);
    assert_eq!(ahead, source_built, "AOT and runtime source differ");
    assert_eq!(aot.cache_report().source_compiles, 0);
    for (case, [a, b, c]) in f32_cases().iter().enumerate() {
        let (sum, product) = (ahead[2 * case], ahead[2 * case + 1]);
        eprintln!("case {case}: a*b+c={sum:#010x} a*b={product:#010x}");
        // Normal-range, non-NaN results equal separately rounded CPU f32.
        if [0, 3, 4, 5].contains(&case) {
            assert_eq!(sum, (a * b + c).to_bits(), "case {case}");
            assert_eq!(product, (a * b).to_bits(), "case {case}");
        }
    }
    // Contraction off: the fused result would be -2^-46, not zero.
    assert_eq!(ahead[0], 0.0f32.to_bits());
    assert_ne!(f32_cases()[0][0].mul_add(f32_cases()[0][1], -1.0), 0.0);
    // Subnormals flush on this GPU on both paths; the strict-f32 subnormal
    // fallback stays in the D1 admission, which AOT does not change.
    let flushed = ahead[3] == 0;
    eprintln!("subnormal product flushed: {flushed}");
}

#[test]
#[ignore = "requires a real Apple7+ Metal GPU and the Metal Toolchain"]
fn pipelines_are_reused_bounded_and_keyed_by_exact_kernel() {
    let _gpu = lock();
    let image = prebuilt(MAP);
    let mut resident = scheduler::open_device(MAP.as_bytes(), Kernels::Prebuilt(&image)).unwrap();
    let values = input(4097, 3);
    let plan = plans(MAP, 4097, &resident).remove(0);
    let mut map = resident.prepare_resident_map(plan, &[&values]).unwrap();
    assert!(map.read().is_err());
    resident.run_resident_map(&mut map).unwrap();
    assert_eq!(
        map.read().unwrap(),
        vec![values.iter().map(|v| v.wrapping_add(5)).collect::<Vec<_>>()]
    );
    assert_eq!(resident.cache_report().source_compiles, 0);
    let mut device = MetalDevice::open_runtime_source().unwrap();
    // MAX_PIPELINES + 1 maps that differ only in their constant.
    let source = |k: u32| MAP.replace("5u32", &format!("{}u32", 100 + k));
    let values = input(4097, 3);
    let run = |device: &mut MetalDevice, k: u32| {
        let plan = &plans(&source(k), 4097, device)[0];
        let r = device.run_map(plan, &values).unwrap();
        let oracle = values
            .iter()
            .map(|v| v.wrapping_add(100 + k))
            .collect::<Vec<_>>();
        assert_eq!(r.values, oracle, "kernel {k} ran another kernel's pipeline");
        r.cache_hit
    };
    assert!(!run(&mut device, 0));
    assert!(
        run(&mut device, 0),
        "a repeated kernel must reuse its pipeline"
    );
    // Interleaved distinct kernels never collide.
    for _ in 0..3 {
        run(&mut device, 1);
        run(&mut device, 2);
    }
    for k in 3..=MAX_PIPELINES as u32 {
        run(&mut device, k);
    }
    let report = device.cache_report();
    assert_eq!(report.pipelines, MAX_PIPELINES, "{report:?}");
    // Kernel 0 was least recently used, so the seventeenth kernel evicted it.
    let created = report.pipeline_creations;
    assert!(!run(&mut device, 0));
    assert_eq!(device.cache_report().pipeline_creations, created + 1);
    assert_eq!(device.cache_report().pipelines, MAX_PIPELINES);

    // Scan: three passes from one library, compiled once, then all cached.
    let mut scan = MetalDevice::open_runtime_source().unwrap();
    let plan = &plans(INCLUSIVE, 4097, &scan)[0];
    let (_, hit) = dispatch(&mut scan, plan, 4097);
    assert!(!hit);
    let report = scan.cache_report();
    assert_eq!((report.source_compiles, report.pipeline_creations), (1, 3));
    let (_, hit) = dispatch(&mut scan, plan, 4097);
    assert!(hit);
    // Reuse preserves cumulative counters and records a memory hit rather
    // than repeating the prior archive lookup, its write refusal (a full cache
    // refuses the first write) or its pipeline-creation timing.
    let mut expected = report;
    expected.archive.last = invar_metal::ArchiveState::MemoryHit;
    expected.archive.write_refusal = None;
    expected.archive.pipeline_creation_ns = 0;
    assert_eq!(scan.cache_report(), expected);
    // The same from a compiled-ahead library: no source compile at all.
    let image = prebuilt(INCLUSIVE);
    let mut aot = scheduler::open_device(INCLUSIVE.as_bytes(), Kernels::Prebuilt(&image)).unwrap();
    let plan = &plans(INCLUSIVE, 4097, &aot)[0];
    assert!(!dispatch(&mut aot, plan, 4097).1);
    assert!(dispatch(&mut aot, plan, 4097).1);
    let report = aot.cache_report();
    assert_eq!((report.source_compiles, report.pipeline_creations), (0, 3));

    // General-net: preparing new work reuses the cached ABI5 pipeline.
    let mut gnet = MetalDevice::open_runtime_source().unwrap();
    run_f32(&mut gnet);
    run_f32(&mut gnet);
    let report = gnet.cache_report();
    assert_eq!((report.source_compiles, report.pipeline_creations), (1, 1));
}

/// Two shapes, so two shape-specific contraction libraries plus the probe.
const TENSOR: &str = include_str!("../examples/metal-tensor-two-shapes.inv");

#[test]
#[ignore = "requires a real Apple7+ Metal GPU and the Metal Toolchain"]
fn tensor_contractions_run_from_built_libraries_with_the_tripwire_armed() {
    let _gpu = lock();
    let dir = scratch();
    let (path, library, other, stale) = (
        dir.join("tensor.inv"),
        dir.join("tensor.metallibc"),
        dir.join("other.inv"),
        dir.join("other.metallibc"),
    );
    std::fs::write(&path, TENSOR).unwrap();
    std::fs::write(&other, TENSOR.replace("return 7;", "return 6;")).unwrap();
    let [path, library, other, stale] =
        [&path, &library, &other, &stale].map(|p| p.to_str().unwrap().to_owned());
    for (source, output) in [(&path, &library), (&other, &stale)] {
        let built = armed(&[
            "build", source, "--target", "metal", "--entry", "mat", "--family", "tensor", "--fuel",
            "100000", "--output", output,
        ]);
        assert!(built.status.success(), "{}", json(&built));
        let receipt = json(&built)["library"].clone();
        assert_eq!(receipt["route"], "tensor-f32-v1");
        assert_eq!(
            receipt["libraries"].as_array().unwrap().len(),
            3,
            "{receipt}"
        );
    }
    let device = ["source-device", &path, "tensor", "mat", "metal", "100000"];
    let aot = armed(&[&device[..], &["--artifact", &library]].concat());
    assert!(aot.status.success(), "{}", json(&aot));
    let aot = json(&aot);
    assert_eq!(aot["result"], 7);
    assert_eq!(aot["completed_regions"], 2, "{aot}");
    assert_eq!(aot["library"]["loaded_via"], "newLibraryWithData");

    let developer = Command::new(env!("CARGO_BIN_EXE_invar"))
        .env(scheduler::RUNTIME_SOURCE_ENV, "1")
        .env_remove(SOURCE_TRIPWIRE)
        .args(device)
        .output()
        .unwrap();
    let developer = json(&developer);
    for key in [
        "result",
        "completed_regions",
        "refusals",
        "source_steps",
        "numeric_probe",
    ] {
        assert_eq!(aot[key], developer[key], "{key}");
    }

    // A contraction missing from an accepted library is a recorded refusal;
    // that loop runs on the CPU and nothing is compiled.
    let caps = MetalDevice::open().unwrap().capabilities();
    let kernels =
        invar::tensor::host::metal::kernels(TENSOR.as_bytes(), "mat", 100_000, caps).unwrap();
    assert_eq!(kernels.len(), 3);
    let probe_only = dir.join("probe-only.metallibc");
    let bytes =
        metal_aot::build::build(Route::Tensor, "mat", TENSOR.as_bytes(), &kernels[..1]).unwrap();
    metal_aot::build::write_atomic(&probe_only, &bytes).unwrap();
    let partial = armed(&[&device[..], &["--artifact", probe_only.to_str().unwrap()]].concat());
    assert!(partial.status.success(), "{}", json(&partial));
    let partial = json(&partial);
    assert_eq!(partial["result"], 7);
    assert_eq!(partial["completed_regions"], 0);
    assert_eq!(
        partial["refusals"],
        serde_json::json!(["tensor_submit", "tensor_submit"])
    );

    // No library: refused with the build command, never compiled.
    let missing = armed(&device);
    assert_eq!(missing.status.code(), Some(1));
    assert!(
        json(&missing)["error"]
            .as_str()
            .unwrap()
            .contains("invar build")
    );
    // Another program's library: refused before Metal sees a byte.
    let refused = armed(&[&device[..], &["--artifact", &stale]].concat());
    assert_eq!(refused.status.code(), Some(1));
    assert!(
        json(&refused)["error"]
            .as_str()
            .unwrap()
            .contains("source_sha256"),
        "{}",
        json(&refused)
    );
}
