//! #48: f32 `sin`, `cos` and `sqrt` are `invar-math-cr-v1` on every engine.
//! The CPU engines compute the correctly rounded words; no device lowers cr-v1
//! `sin` or `cos` yet, so f32 source regions refuse them, CUDA and Metal
//! requests keep each calling function on the CPU with reason
//! `math-cr-v1-device-unavailable`, and `sqrt` keeps its exact device helpers.
//! tests/support/gnet_entry.rs covers the general net on every target.
use invar::{
    check, eval, front,
    math::{self, MathFn},
    mir, mir_eval,
    numeric_contract::F32Bits,
    schedule,
};

const FUEL: u64 = 1_000_000;

/// Normal, huge (large argument reduction) and subnormal arguments.
const ARGS: [u32; 10] = [
    0x3f00_0000,
    0x4040_0000,
    0x47c3_5000,
    0x3fc9_0fdb,
    0x4049_0fdb,
    0x6f79_be45,
    0x7f7f_ffff,
    0x0000_0001,
    0x0040_0000,
    0x3dcc_cccd,
];

fn checked(source: &str) -> check::CheckedProgram {
    check::check(front::parse(source.as_bytes()).unwrap(), true)
        .unwrap_or_else(|error| panic!("{error:?}\n{source}"))
}

/// A finite word as a source literal with exactly its value.
fn literal(bits: u32) -> String {
    let text = format!("{}", f32::from_bits(bits));
    if text.contains('.') {
        text
    } else {
        format!("{text}.0")
    }
}

/// `kernel` maps `f` over `ARGS` in an f32 region; `main` compares every
/// output with the cr-v1 word `math::eval_f32` gives and returns 7.
fn program(f: MathFn) -> String {
    let inputs = ARGS.map(literal).join(", ");
    let checks: String = ARGS
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let want = math::eval_f32(f, &[F32Bits::from_bits(x)]).to_bits();
            format!("if out[{i}u32] != {} {{ return 0; }} ", literal(want))
        })
        .collect();
    format!(
        "fn kernel(a: read Array<f32>, n: u32) -> Array<f32> {{ var out = filled_f32(n, 0.0); \
         for i in 0u32..n {{ out[i] = {}(a[i]); }} return out; }} \
         fn main() -> i32 {{ let a: Array<f32> = [{inputs}]; let out = kernel(a, {}u32); {checks}return 7; }}",
        f.name(),
        ARGS.len()
    )
}

#[test]
fn c4_f32_trig_regions_stay_on_the_cpu_with_cr_v1_words() {
    for f in [MathFn::Sin, MathFn::Cos] {
        let source = program(f);
        let checked = checked(&source);
        let reference = eval::observe(&checked, FUEL);
        assert_eq!(reference.result, Ok(eval::Value::I32(7)), "{source}");
        let lowered = mir::lower(&checked).unwrap();
        assert_eq!(mir_eval::observe(&lowered, FUEL).result, reference.result);
        assert_eq!(
            math::contracts(&lowered),
            Some(serde_json::json!({"math": "invar-math-cr-v1"}))
        );
        for workers in [1, 4] {
            let region =
                invar::f32_source::run_cpu_source(source.as_bytes(), FUEL, workers).unwrap();
            assert_eq!(region.completed, 0, "{} ran on the device plan", f.name());
            assert_eq!(region.observation.result, reference.result);
            assert_eq!(region.observation.steps_used, reference.steps_used);
            let parallel =
                schedule::run_source(source.as_bytes(), &lowered, FUEL, workers).unwrap();
            assert_eq!(parallel.observation.result, reference.result, "{workers}");
        }
        #[cfg(feature = "native")]
        {
            let (status, doc) = invar::native::run_program(&lowered, FUEL).unwrap();
            assert_eq!(
                (status, &doc["result"]["value"]),
                (0, &serde_json::json!(7)),
                "{doc}"
            );
        }
    }
}

#[test]
fn c4_f32_sqrt_regions_complete_on_the_device_plan_with_cr_v1_words() {
    let source = program(MathFn::Sqrt);
    let reference = eval::observe(&checked(&source), FUEL);
    assert_eq!(reference.result, Ok(eval::Value::I32(7)), "{source}");
    for workers in [1, 4] {
        let region = invar::f32_source::run_cpu_source(source.as_bytes(), FUEL, workers).unwrap();
        assert!(region.completed >= 1, "{:?}", region.refusals);
        assert_eq!(region.observation.result, reference.result);
    }
}

/// `wave` calls f32 `sin`, `turn` only f64 `cos` and `main` neither; `bump`
/// is a u32 map a forced device runs, so the device request has a region.
const SITES: &str = "fn wave(x: f32) -> f32 { let y = x * 2.0; return sin(y); }\n\
                     fn turn(x: f64) -> f64 { return cos(x); }\n\
                     fn bump(xs: read Array<u32>) -> Array<u32> { var ys = filled_u32(len(xs), 0u32); \
                     for i in 0u32..len(xs) { ys[i] = wrapping_add(xs[i], 3u32); } return ys; }\n\
                     fn main() -> i32 { let xs: Array<u32> = filled_u32(4096u32, 7u32); let ys = bump(xs); \
                     var s: f32 = 0.0; for k in 0u32..4u32 { s = s + wave(f32_from_u32(k)); } \
                     if turn(0.0f64) == 1.0 and s < 0.0 and ys[4095u32] == 10u32 { return 7; } return 0; }\n";

#[test]
fn c4_cpu_kept_sites_name_each_function_calling_f32_trig() {
    let program = mir::lower(&checked(SITES)).unwrap();
    let sites = math::cr_v1_device_sites(&program);
    let names: Vec<&str> = sites.iter().map(|(f, _)| f.as_str()).collect();
    assert_eq!(names, ["wave"]);
    let span = sites[0].1;
    assert_eq!(&SITES[span.start..span.end], "sin(y)");
}

fn explain(args: &[&str]) -> serde_json::Value {
    let dir = std::env::temp_dir().join(format!("invar-c4-{}-{}", std::process::id(), args.len()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("sites.inv"), SITES).unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_invar"))
        .current_dir(&dir)
        .env("INVAR_METAL_RUNTIME_SOURCE", "1")
        .args(["explain", "sites.inv"])
        .args(args)
        .args(["--format", "json"])
        .output()
        .unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    let explain: serde_json::Value = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|_| panic!("{}", String::from_utf8_lossy(&output.stderr)));
    let schema: serde_json::Value =
        serde_json::from_str(include_str!("../../schema/explain-v5.schema.json")).unwrap();
    let validator = jsonschema::validator_for(&schema).unwrap();
    let errors: Vec<String> = validator
        .iter_errors(&explain)
        .map(|e| e.to_string())
        .collect();
    assert!(errors.is_empty(), "{errors:?}\n{explain}");
    let device = &explain["device_execution"];
    assert_eq!(device["result"], 7, "{explain}");
    let kept: Vec<_> = device["choices"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["reason"] == math::MATH_CR_V1_DEVICE_UNAVAILABLE)
        .collect();
    assert_eq!(kept.len(), 1, "{explain}");
    assert_eq!(kept[0]["function"], "wave", "{explain}");
    assert_eq!(kept[0]["backend"], "mir-serial", "{explain}");
    explain
}

#[test]
fn c4_placement_explain_keeps_f32_trig_on_the_cpu() {
    let host = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);
    let explain = explain(&["--engine", "auto", "--host-class", &host]);
    assert_eq!(explain["auto_path"], "placement", "{explain}");
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
#[test]
#[ignore = "requires CUDA or Metal hardware"]
fn c4_device_explain_keeps_f32_trig_on_the_cpu() {
    let engine = if cfg!(feature = "cuda") {
        "cuda"
    } else {
        "metal"
    };
    explain(&["--engine", engine]);
    explain(&["--engine", "auto"]);
}
