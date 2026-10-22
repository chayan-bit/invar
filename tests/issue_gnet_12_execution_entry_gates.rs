//! ABI12 review3 A1: execution entry gates. For every artifact the corpus
//! compiles (ABI1-11 and ABI12), record the outcome of
//! `ProvenancedArtifact::fenced_execution` for each fence backend and of
//! `device_host::cpu::CpuModel::prepare` with a recording compiler stand-in.
//! ABI1-11 outcomes must equal a golden from pre-2b `a6786bf`; ABI12 must be
//! refused explicitly by both routes with zero compiler calls. An ABI5 positive
//! control is admitted by the fence and reaches the recorded compiler.
//! GNET-03 widened ABI5 device admission to owned i32/u32 arrays; its newly
//! admitted artifacts are pinned exactly in a separate golden.
use invar::general_net::{
    compiled,
    device_host::{Backend, cpu},
    provenance::{self, FenceBackend},
};
use serde_json::{Map, Value, json};
use std::path::{Path, PathBuf};

const SOURCES: [(&str, &str); 6] = [
    (
        "valid_owned_numeric",
        "fn run(a: f32, n: u32) -> f32 { var xs: Array<f32> = filled_f32(n + 1u32, a); xs[0u32] = xs[0u32] * a + a; return xs[0u32]; } fn main() -> i32 { return 0; }",
    ),
    (
        "valid_scalar",
        "fn run(n: u32) -> i32 { if n < 2u32 { return 1; } return run(n - 1u32) + run(n - 2u32); } fn main() -> i32 { return run(5u32); }",
    ),
    (
        "valid_numeric",
        "fn run(a: f32, b: f32) -> f32 { return a * b + a; } fn main() -> i32 { return 0; }",
    ),
    (
        "valid_int_loop",
        "fn run(n: u32) -> u32 { var i: u32 = 0u32; var s: u32 = 0u32; while i < n { s = s + i; i = i + 1u32; } return s; } fn main() -> i32 { return 0; }",
    ),
    (
        "valid_owned_int",
        "fn run(n: u32) -> i32 { var xs = filled_i32(n + 1u32, 3); xs[0u32] = xs[0u32] + 2; return xs[0u32]; } fn main() -> i32 { return 0; }",
    ),
    (
        "children",
        "fn branch(depth: u32) -> i32 { if depth<2u32 { return 1; } return branch(depth-1u32)+branch(depth-2u32); } fn run(n: u32) -> i32 { return branch(n); }",
    ),
];

const TARGETS: [compiled::Target; 3] = [
    compiled::Target::Cpu,
    compiled::Target::Cuda,
    compiled::Target::Metal,
];
const BACKENDS: [FenceBackend; 3] = [
    FenceBackend::CudaDriver,
    FenceBackend::MetalCommand,
    FenceBackend::CpuGroupModel,
];

fn storage() -> compiled::OwnedLimits {
    compiled::OwnedLimits {
        elements: 16,
        allocations: 8,
        cleanup_events: 16,
    }
}
fn speculation() -> compiled::SpeculationLimits {
    compiled::SpeculationLimits {
        slots: 2,
        fuel: 10000,
        transitions: 10000,
        trace_events: 1024,
        quantum: 32,
        lookahead: 128,
        scratch_bytes: 1024 * 1024,
    }
}
fn device_config() -> compiled::DeviceConfig {
    compiled::DeviceConfig {
        fuel: 10000,
        transitions: 10000,
        trace_events: 1024,
        quantum: 32,
        rounds: 64,
        capacity: 2,
    }
}

fn artifacts() -> Vec<(String, provenance::ProvenancedArtifact)> {
    let l = provenance::Limits::default();
    let mut out = Vec::new();
    for (name, source) in SOURCES {
        let s = source.as_bytes();
        for target in TARGETS {
            let routes = [
                ("compile", provenance::compile(s, "run", target, l)),
                (
                    "compile_numeric",
                    provenance::compile_numeric(s, "run", target, l),
                ),
                (
                    "compile_resumable_numeric",
                    provenance::compile_resumable_numeric(s, "run", target, l),
                ),
                (
                    "compile_owned",
                    provenance::compile_owned(s, "run", target, l, storage()),
                ),
                (
                    "compile_owned_numeric",
                    provenance::compile_owned_numeric(s, "run", target, l, storage()),
                ),
                (
                    "compile_resumable_owned_numeric",
                    provenance::compile_resumable_owned_numeric(s, "run", target, l, storage()),
                ),
                (
                    "compile_resumable",
                    provenance::compile_resumable(s, "run", target, l, storage()),
                ),
                (
                    "compile_speculative",
                    provenance::compile_speculative(s, "run", target, l, storage(), speculation()),
                ),
                (
                    "compile_device_speculative",
                    provenance::compile_device_speculative(
                        s,
                        "run",
                        target,
                        l,
                        storage(),
                        speculation(),
                    ),
                ),
            ];
            for (route, result) in routes {
                if let Ok(a) = result {
                    out.push((format!("{name}/{route}/{target:?}"), a));
                }
            }
        }
    }
    out
}

/// An admitted ABI5 workset: the artifact's own if it has a device layout,
/// otherwise the `children` ABI5 artifact's for the same device target.
fn workset(
    a: &provenance::ProvenancedArtifact,
    abi5: &[(compiled::Target, provenance::ProvenancedArtifact)],
) -> compiled::Workset {
    let owner = if a.artifact().device_layout().is_some() {
        a
    } else {
        let target = match a.artifact().target() {
            compiled::Target::Metal => compiled::Target::Metal,
            _ => compiled::Target::Cuda,
        };
        &abi5.iter().find(|(t, _)| *t == target).unwrap().1
    };
    let layout = owner.artifact().device_layout().unwrap();
    let bytes = layout.buffer_bytes(device_config()).unwrap();
    layout
        .workset(
            device_config(),
            std::array::from_fn(|i| compiled::BufferRange {
                device: 1,
                allocation: (i + 1) as u64,
                offset: 0,
                bytes: bytes[i],
            }),
        )
        .unwrap()
}

fn recorder(dir: &Path) -> (PathBuf, PathBuf) {
    use std::os::unix::fs::PermissionsExt;
    let calls = dir.join("calls");
    let script = dir.join("cxx");
    std::fs::write(
        &script,
        format!("#!/bin/sh\necho x >> '{}'\nexit 1\n", calls.display()),
    )
    .unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    (script, calls)
}

fn calls(path: &Path) -> usize {
    std::fs::read_to_string(path).map_or(0, |s| s.lines().count())
}

fn outcomes() -> Map<String, Value> {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("abi12-entry-gates-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (script, log) = recorder(&dir);
    let l = provenance::Limits::default();
    let abi5: Vec<_> = [compiled::Target::Cuda, compiled::Target::Metal]
        .into_iter()
        .map(|t| {
            let a = provenance::compile_device_speculative(
                SOURCES[5].1.as_bytes(),
                "run",
                t,
                l,
                storage(),
                speculation(),
            )
            .unwrap();
            (t, a)
        })
        .collect();
    let mut map = Map::new();
    for (key, a) in artifacts() {
        let w = workset(&a, &abi5);
        let mut fenced = Map::new();
        for backend in BACKENDS {
            let outcome = match a.fenced_execution(w.clone(), (), 1, backend) {
                Ok(_) => "Ok".to_owned(),
                Err(e) => format!("Err({e:?})"),
            };
            fenced.insert(format!("{backend:?}"), json!(outcome));
        }
        let before = calls(&log);
        let mut model = cpu::CpuModel::with_compiler(&script);
        let prepared = match model.prepare(a.artifact(), [8; 12], &[], [0; 6]) {
            Ok(_) => "Ok".to_owned(),
            Err(e) => {
                // A failed compile keeps an evidence directory; drop its path.
                if let Some(path) = e
                    .split("evidence ")
                    .nth(1)
                    .and_then(|p| p.split(':').next())
                {
                    let _ = std::fs::remove_dir_all(path);
                }
                e.split("; evidence").next().unwrap().to_owned()
            }
        };
        map.insert(
            key,
            json!({
                "abi": a.artifact().abi_version(),
                "fenced_execution": fenced,
                "cpu_model_prepare": prepared,
                "compiler_calls": calls(&log) - before,
            }),
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
    map
}

const GOLDEN: &str = include_str!("fixtures/gnet12/pre-2b-execution-entry-golden.json");
const GNET_03: &str = include_str!("fixtures/gnet12/gnet-03-device-admissions.json");

#[test]
fn issue_gnet_12_execution_entry_gates_refuse_abi12_and_preserve_abi1_11() {
    let current = outcomes();
    if let Some(path) = std::env::var_os("INVAR_GNET12_WRITE_ENTRY_GATES") {
        std::fs::write(
            path,
            serde_json::to_string_pretty(&Value::Object(current)).unwrap() + "\n",
        )
        .unwrap();
        return;
    }
    let golden: Map<String, Value> = serde_json::from_str(GOLDEN).unwrap();
    let p0b: Map<String, Value> = serde_json::from_str(include_str!(
        "fixtures/gnet_entry/abi13-execution-entries.json"
    ))
    .unwrap();
    for (key, want) in &p0b {
        assert!(!golden.contains_key(key));
        assert_eq!(want["abi"], 13);
        assert!(key.contains("/compile_device_speculative/"));
        assert_eq!(current.get(key), Some(want), "{key}: P0b entry changed");
    }
    // Every pre-2b artifact keeps byte-identical entry outcomes.
    for (key, want) in &golden {
        assert_ne!(want["abi"], 12, "{key}");
        assert_eq!(current.get(key), Some(want), "{key} differs from pre-2b");
    }
    // GNET-03 owned-array ABI5 admissions keep their exact outcomes.
    let gnet_03: Map<String, Value> = serde_json::from_str(GNET_03).unwrap();
    for (key, want) in &gnet_03 {
        assert!(!golden.contains_key(key), "{key}");
        assert_eq!(want["abi"], 5, "{key}");
        assert_eq!(current.get(key), Some(want), "{key} differs from GNET-03");
    }
    // Every other new artifact is an ABI12 compile receipt, all refused.
    let mut abi12 = 0;
    for (key, value) in &current {
        if golden.contains_key(key) || gnet_03.contains_key(key) || p0b.contains_key(key) {
            continue;
        }
        assert_eq!(value["abi"], 12, "{key} is new but not ABI12");
        assert!(
            key.ends_with("/compile_resumable_owned_numeric/Cuda"),
            "{key}"
        );
        for backend in BACKENDS {
            assert_eq!(
                value["fenced_execution"][format!("{backend:?}")],
                "Err(ExecutionMismatch)",
                "{key} {backend:?}"
            );
        }
        // Literal (the constant postdates the a6786bf golden tree).
        assert_eq!(
            value["cpu_model_prepare"], "CPU device model: unsupported ABI 12 artifact",
            "{key}"
        );
        assert_eq!(value["compiler_calls"], 0, "{key}");
        abi12 += 1;
    }
    // Exactly the CUDA counterparts of the pre-2b ABI11 CPU continuations.
    let mut expected: Vec<String> = golden
        .iter()
        .filter(|(k, v)| v["abi"] == 11 && k.ends_with("/compile_resumable_owned_numeric/Cpu"))
        .map(|(k, _)| k.replace("/Cpu", "/Cuda"))
        .collect();
    let mut admitted: Vec<String> = current
        .keys()
        .filter(|k| !golden.contains_key(*k) && !gnet_03.contains_key(*k) && !p0b.contains_key(*k))
        .cloned()
        .collect();
    expected.sort();
    admitted.sort();
    assert_eq!(admitted, expected);
    assert_eq!(abi12, expected.len());
    // Positive controls: an ABI5 artifact is admitted by its matching fence
    // backends and reaches the recorded compiler exactly once.
    let mut positive = 0;
    for (key, value) in &golden {
        if value["abi"] == 5 {
            assert!(
                value["fenced_execution"]
                    .as_object()
                    .unwrap()
                    .values()
                    .any(|v| v == "Ok"),
                "{key}"
            );
            assert_eq!(value["compiler_calls"], 1, "{key}");
            positive += 1;
        }
    }
    assert!(positive >= 2, "ABI5 controls: {positive}");
    eprintln!(
        "entry gates: {} pre-2b outcomes identical, {abi12} ABI12 refused, {positive} ABI5 controls",
        golden.len()
    );
}
