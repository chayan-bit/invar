//! ABI12 stage 1 (`cuda-owned-numeric-continuation-v1`): emitter/admission only.
//! Checks that every pre-existing emission is byte-identical to the recorded
//! pre-change golden, that CUDA strict-f32 owned continuations get a distinct
//! ABI and kernel symbol, that Metal and other CUDA numeric modes stay refused,
//! and that no runtime path admits ABI12. Emitted ABI12 text is checked by a CPU
//! execution model; there is no NVRTC compilation, device or host support here.
use invar::{
    check, front,
    general_net::{self, Net, compiled},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

pub fn lowered(source: &str) -> Net {
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), false).unwrap();
    invar::ownership::check(&checked).unwrap();
    Net::lower(&checked, general_net::Limits::default()).unwrap()
}
fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// (name, source, entry) — one program per emitter family.
const CORPUS: &[(&str, &str, &str)] = &[
    (
        "scalar",
        "fn run(n: u32) -> i32 { if n < 2u32 { return 1; } return run(n - 1u32) + run(n - 2u32); } fn main() -> i32 { return run(5u32); }",
        "run",
    ),
    (
        "owned",
        "fn run(n: u32) -> i32 { var xs: Array<i32> = filled_i32(n + 1u32, 1); xs[0u32] = xs[0u32] + 2; return xs[0u32]; } fn main() -> i32 { return run(2u32); }",
        "run",
    ),
    (
        "numeric",
        "fn run(a: f32, b: f32) -> f32 { return fma(a, b, a) / b; } fn main() -> i32 { return 0; }",
        "run",
    ),
    (
        "numeric_owned",
        "fn run(a: f32, n: u32) -> f32 { var xs: Array<f32> = filled_f32(n + 1u32, a); xs[0u32] = xs[0u32] * a + a; return xs[0u32]; } fn main() -> i32 { return 0; }",
        "run",
    ),
    (
        "aggregate",
        "fn make(ok: Bool) -> Result<i32,u32> { if ok { return Result::Ok(7); } return Result::Err(9u32); } fn run(ok: Bool) -> i32 { let x = make(ok); return match move x { Result::Ok(a) => a, Result::Err(e) => 9 }; } fn main() -> i32 { return run(true); }",
        "run",
    ),
];
const EMITTERS: [&str; 11] = [
    "emit",
    "emit_owned",
    "emit_resumable",
    "emit_speculative",
    "emit_device_speculative",
    "emit_aggregates",
    "emit_resumable_aggregates",
    "emit_numeric",
    "emit_resumable_numeric",
    "emit_owned_numeric",
    "emit_resumable_owned_numeric",
];
const TARGETS: [compiled::Target; 3] = [
    compiled::Target::Cpu,
    compiled::Target::Cuda,
    compiled::Target::Metal,
];

pub fn emit(
    net: &Net,
    emitter: &str,
    entry: &str,
    target: compiled::Target,
) -> Result<compiled::Artifact, compiled::Error> {
    let proof = net.verify().unwrap();
    let limits = compiled::Limits::default();
    let storage = compiled::OwnedLimits {
        elements: 64,
        allocations: 16,
        cleanup_events: 32,
    };
    let spec = compiled::SpeculationLimits::default();
    match emitter {
        "emit" => compiled::emit(net, &proof, entry, target, limits),
        "emit_owned" => compiled::emit_owned(net, &proof, entry, target, limits, storage),
        "emit_resumable" => compiled::emit_resumable(net, &proof, entry, target, limits, storage),
        "emit_speculative" => {
            compiled::emit_speculative(net, &proof, entry, target, limits, storage, spec)
        }
        "emit_device_speculative" => {
            compiled::emit_device_speculative(net, &proof, entry, target, limits, storage, spec)
        }
        "emit_aggregates" => compiled::emit_aggregates(net, &proof, entry, target, limits, storage),
        "emit_resumable_aggregates" => {
            compiled::emit_resumable_aggregates(net, &proof, entry, target, limits, storage)
        }
        "emit_numeric" => compiled::emit_numeric(net, &proof, entry, target, limits),
        "emit_resumable_numeric" => {
            compiled::emit_resumable_numeric(net, &proof, entry, target, limits)
        }
        "emit_owned_numeric" => {
            compiled::emit_owned_numeric(net, &proof, entry, target, limits, storage)
        }
        "emit_resumable_owned_numeric" => {
            compiled::emit_resumable_owned_numeric(net, &proof, entry, target, limits, storage)
        }
        _ => unreachable!(),
    }
}
fn record(result: &Result<compiled::Artifact, compiled::Error>) -> Value {
    match result {
        Ok(a) => json!({
            "abi": a.abi_version(),
            "target": format!("{:?}", a.target()),
            "source_sha256": hex(a.source().as_bytes()),
            "source_bytes": a.source().len(),
            "frame_bytes": a.frame_bytes(),
            "states": a.states(),
            "task_bytes": a.continuation_layout().map(|c| c.task_bytes()),
            "kernel": a.continuation_layout().map(|c| c.kernel_name()),
            "heap_words": a.owned_layout().map(|o| o.heap_words()),
        }),
        Err(e) => json!({"error": format!("{e:?}")}),
    }
}
/// Every (program, emitter, target) outcome, keyed for stable comparison.
fn outcomes() -> serde_json::Map<String, Value> {
    let mut map = serde_json::Map::new();
    for (name, source, entry) in CORPUS {
        let net = lowered(source);
        for emitter in EMITTERS {
            for target in TARGETS {
                map.insert(
                    format!("{name}/{emitter}/{target:?}"),
                    record(&emit(&net, emitter, entry, target)),
                );
            }
        }
    }
    map
}
const GOLDEN: &str = include_str!("fixtures/gnet12/pre-abi12-emission-golden.json");
/// The only outcomes ABI12 stage 1 may change are CUDA results of the resumable
/// owned-numeric emitter, previously refused with `NumericTarget` because the
/// target was checked first. The admitted subset must equal ABI11's: a program
/// that is ABI11 on CPU becomes ABI12 on CUDA with an identical layout, and a
/// program refused on CPU gets the identical refusal on CUDA.
fn abi12_emitter_on_cuda(key: &str) -> bool {
    key.ends_with("/emit_resumable_owned_numeric/Cuda")
}

#[test]
fn issue_gnet_12_stage1_emissions_preserve_layouts_and_pin_p0b_deltas() {
    let current = outcomes();
    if let Some(path) = std::env::var_os("INVAR_GNET12_WRITE_GOLDEN") {
        std::fs::write(
            path,
            serde_json::to_string_pretty(&Value::Object(current)).unwrap() + "\n",
        )
        .unwrap();
        return;
    }
    let golden = p5b_delta::apply(
        p0b_delta::apply(
            GOLDEN,
            include_str!("fixtures/gnet_entry/legacy-emission-delta.json"),
        ),
        include_str!("fixtures/gnet_p5b/emission-delta.json"),
    );
    assert_eq!(
        golden.keys().collect::<Vec<_>>(),
        current.keys().collect::<Vec<_>>()
    );
    let (mut same, mut admitted, mut refused_as_cpu) = (0, Vec::new(), Vec::new());
    let mut device_owners = Vec::new();
    for (key, want) in &golden {
        let got = &current[key];
        // GNET-03 admits owners inside ABI5 device children: the only other
        // permitted change is a former `DeviceSpeculationOwner` refusal that
        // becomes an ABI5 artifact with the same resumable task layout as CPU.
        if key.contains("/emit_device_speculative/")
            && want["error"] == "DeviceSpeculationOwner"
            && got.get("error").is_none()
        {
            assert_eq!(got["abi"], 5, "{key}");
            let resumable = &current[&key.replace("emit_device_speculative", "emit_resumable")];
            for field in ["frame_bytes", "task_bytes", "heap_words"] {
                assert_eq!(got[field], resumable[field], "{key}: {field}");
            }
            device_owners.push(key.clone());
            continue;
        }
        if !abi12_emitter_on_cuda(key) {
            assert_eq!(got, want, "{key} changed");
            same += 1;
            continue;
        }
        assert_eq!(
            want["error"], "NumericTarget",
            "{key}: golden must be a refusal"
        );
        let cpu = &current[&key.replace("/Cuda", "/Cpu")];
        if cpu.get("error").is_some() {
            assert_eq!(got, cpu, "{key}: refusal must equal the CPU target's");
            refused_as_cpu.push(format!("{key}: {}", got["error"]));
        } else {
            assert_eq!(cpu["abi"], 11, "{key}: CPU stays ABI11");
            assert_eq!(got["abi"], 12, "{key}");
            assert_eq!(got["target"], "Cuda", "{key}");
            for field in ["frame_bytes", "states", "task_bytes", "heap_words"] {
                assert_eq!(got[field], cpu[field], "{key}: {field} must equal ABI11");
            }
            assert_eq!(cpu["kernel"], "invar_gnet_resume_v3", "{key}");
            assert_eq!(got["kernel"], "invar_gnet_owned_numeric_resume_v1", "{key}");
            admitted.push(key.clone());
        }
    }
    eprintln!(
        "emission golden: {same} identical; ABI12 admitted: {admitted:?}; refused as on CPU: {refused_as_cpu:?}"
    );
    assert!(
        admitted.iter().any(|k| k.starts_with("numeric_owned/")),
        "{admitted:?}"
    );
    assert_eq!(
        device_owners,
        [
            "owned/emit_device_speculative/Cuda",
            "owned/emit_device_speculative/Metal"
        ]
    );
    assert_eq!(
        same + admitted.len() + refused_as_cpu.len() + device_owners.len(),
        golden.len()
    );
}

#[test]
fn issue_gnet_12_stage1_admission_refusals_and_header_free_text() {
    let net = lowered(CORPUS[3].1);
    let entry = CORPUS[3].2;
    // Metal stays refused for every numeric mode; CUDA only for ABI12.
    for emitter in [
        "emit_numeric",
        "emit_resumable_numeric",
        "emit_owned_numeric",
        "emit_resumable_owned_numeric",
    ] {
        assert!(
            matches!(
                emit(&net, emitter, entry, compiled::Target::Metal),
                Err(compiled::Error::NumericTarget)
            ),
            "Metal {emitter}"
        );
        let cuda = emit(&net, emitter, entry, compiled::Target::Cuda);
        if emitter == "emit_resumable_owned_numeric" {
            assert_eq!(cuda.as_ref().unwrap().abi_version(), 12);
        } else {
            assert!(
                matches!(cuda, Err(compiled::Error::NumericTarget)),
                "CUDA {emitter}"
            );
        }
    }
    let cpu = emit(
        &net,
        "emit_resumable_owned_numeric",
        entry,
        compiled::Target::Cpu,
    )
    .unwrap();
    let cuda = emit(
        &net,
        "emit_resumable_owned_numeric",
        entry,
        compiled::Target::Cuda,
    )
    .unwrap();
    assert_eq!(cpu.abi_version(), 11);
    assert_eq!(cuda.target(), compiled::Target::Cuda);
    assert_eq!(cuda.params(), cpu.params());
    assert_eq!(cuda.result(), cpu.result());
    let text = cuda.source();
    assert!(text.contains("extern \"C\" __global__ void invar_gnet_owned_numeric_resume_v1("));
    assert!(!text.contains("invar_gnet_resume_v3"));
    // Header-free numeric preamble; no host or device float type anywhere.
    assert!(
        !text.contains("#include"),
        "NVRTC has no standard headers by default"
    );
    assert!(text.contains("static_assert(sizeof(gnet_u)==8 && sizeof(gnet_w)==8"));
    for token in ["float", "double", "__f", "fmaf", "<cmath>"] {
        assert!(!text.contains(token), "unexpected {token} in ABI12 text");
    }
    // The CPU ABI11 text keeps its original header and assertion.
    assert!(cpu.source().contains("#include <climits>"));
    assert!(cpu.source().contains("CHAR_BIT==8"));
}

#[test]
fn issue_gnet_12_stage1_no_runtime_path_admits_abi12() {
    // P0b admits a distinct ABI13 entry. This must never enable ABI12 execution.
    use general_net::device_host;
    for target in [compiled::Target::Cuda, compiled::Target::Metal] {
        let plan = device_host::Plan::from_source(
            CORPUS[3].1.as_bytes(),
            CORPUS[3].2,
            target,
            device_host::Limits::default(),
        );
        assert_eq!(
            plan.unwrap().artifact().abi_version(),
            13,
            "{target:?}: P0b uses its separate entry ABI"
        );
    }
    // Stage 2b (named outcome update): provenance now admits the CUDA
    // owned-numeric continuation with a compile receipt only; every authority
    // is false and the ABI12 runtime refusals remain unchanged.
    let admitted = general_net::provenance::compile_resumable_owned_numeric(
        CORPUS[3].1.as_bytes(),
        CORPUS[3].2,
        compiled::Target::Cuda,
        Default::default(),
        compiled::OwnedLimits::default(),
    )
    .unwrap();
    assert_eq!(admitted.artifact().abi_version(), 12);
    assert_eq!(
        admitted.manifest()["cuda_owned_numeric_continuation"]["authority"],
        serde_json::json!({"execution": false, "module_load": false, "resume": false,
            "device_qualification": false})
    );
}

/// Name kept for test-ID continuity: since stage 2b the CUDA continuation
/// route yields an ABI12 compile receipt only; every other route still refuses.
#[test]
fn issue_gnet_12_stage1_every_source_bound_route_refuses_abi12() {
    use general_net::provenance::{self, Error};
    let refused = Error::Compiler(compiled::Error::NumericTarget);
    let storage = compiled::OwnedLimits::default();
    let spec = compiled::SpeculationLimits::default();
    let limits = provenance::Limits::default();
    let mut checked = 0;
    for (name, source, entry) in [CORPUS[2], CORPUS[3]] {
        let s = source.as_bytes();
        for target in [compiled::Target::Cuda, compiled::Target::Metal] {
            // Owned-numeric continuation and every numeric route: the exact
            // pre-stage-1 typed refusal.
            // Stage 2b (named outcome update): on CUDA the owned-numeric
            // continuation is admitted as an ABI12 compile receipt only.
            let continuation =
                provenance::compile_resumable_owned_numeric(s, entry, target, limits, storage);
            if target == compiled::Target::Cuda {
                let a = continuation.unwrap();
                assert_eq!(a.artifact().abi_version(), 12, "{name}");
                assert_eq!(
                    a.manifest()["cuda_owned_numeric_continuation"]["authority"]["execution"],
                    false
                );
            } else {
                assert_eq!(continuation.unwrap_err(), refused, "{name} {target:?}");
            }
            checked += 1;
            for (route, result) in [
                (
                    "compile_owned_numeric",
                    provenance::compile_owned_numeric(s, entry, target, limits, storage),
                ),
                (
                    "compile_numeric",
                    provenance::compile_numeric(s, entry, target, limits),
                ),
                (
                    "compile_resumable_numeric",
                    provenance::compile_resumable_numeric(s, entry, target, limits),
                ),
            ] {
                assert_eq!(result.unwrap_err(), refused, "{name} {route} {target:?}");
                checked += 1;
            }
            // Non-numeric routes never admit these f32 programs at all.
            for (route, result) in [
                ("compile", provenance::compile(s, entry, target, limits)),
                (
                    "compile_owned",
                    provenance::compile_owned(s, entry, target, limits, storage),
                ),
                (
                    "compile_resumable",
                    provenance::compile_resumable(s, entry, target, limits, storage),
                ),
                (
                    "compile_speculative",
                    provenance::compile_speculative(s, entry, target, limits, storage, spec),
                ),
                (
                    "compile_device_speculative",
                    provenance::compile_device_speculative(s, entry, target, limits, storage, spec),
                ),
            ] {
                if let Ok(artifact) = &result {
                    assert_eq!(
                        route, "compile_device_speculative",
                        "{name} {route} {target:?}"
                    );
                    assert_eq!(
                        artifact.artifact().abi_version(),
                        13,
                        "P0b must not alias ABI12"
                    );
                }
                checked += 1;
            }
        }
        // The CPU route still yields the ABI11 receipt and profile.
        let cpu = provenance::compile_resumable_owned_numeric(
            s,
            entry,
            compiled::Target::Cpu,
            limits,
            storage,
        )
        .unwrap();
        assert_eq!(cpu.artifact().abi_version(), 11);
        assert_eq!(
            cpu.manifest()["numeric_owned_continuation"]["profile"],
            "cpu-owned-numeric-continuation-v1"
        );
    }
    eprintln!("source-bound ABI12 refusals checked: {checked}");
    assert_eq!(checked, 2 * 2 * 9);
}

#[path = "support/gnet_p0b_delta.rs"]
mod p0b_delta;

#[path = "support/gnet_p5b_delta.rs"]
mod p5b_delta;
