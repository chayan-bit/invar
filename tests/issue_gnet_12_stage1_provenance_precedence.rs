//! ABI12 stage-1 review A1: provenance error precedence. Every source-bound
//! outcome of the owned-numeric routes must equal the pre-stage-1 outcome
//! recorded from `259d2fa` (before ABI12 existed): size/entry limits and source
//! parse/check errors keep their priority (net-limit errors are covered by
//! `issue_gnet_12_net_limit_precedence`). Metal and every CUDA emission refusal
//! stay `NumericTarget`; since stage 2b only the named valid CUDA
//! continuations become ABI12 compile receipts, with no execution authority.
use invar::general_net::{compiled, provenance};
use serde_json::{Value, json};

const VALID_OWNED: &str = "fn run(a: f32, n: u32) -> f32 { var xs: Array<f32> = filled_f32(n + 1u32, a); xs[0u32] = xs[0u32] * a + a; return xs[0u32]; } fn main() -> i32 { return 0; }";
const VALID_SCALAR: &str = "fn run(n: u32) -> i32 { if n < 2u32 { return 1; } return run(n - 1u32) + run(n - 2u32); } fn main() -> i32 { return run(5u32); }";
const AGGREGATE: &str = "fn make(ok: Bool) -> Result<i32,u32> { if ok { return Result::Ok(7); } return Result::Err(9u32); } fn run(ok: Bool) -> i32 { let x = make(ok); return match move x { Result::Ok(a) => a, Result::Err(e) => 9 }; } fn main() -> i32 { return run(true); }";

/// (case, source, entry) covering every precedence layer.
fn corpus() -> Vec<(&'static str, Vec<u8>, String)> {
    let max = 1024 * 1024;
    vec![
        ("valid_owned_numeric", VALID_OWNED.into(), "run".into()),
        ("valid_scalar", VALID_SCALAR.into(), "run".into()),
        ("valid_aggregate", AGGREGATE.into(), "run".into()),
        (
            "external_owner_param",
            b"fn run(xs: take Array<f32>) -> i32 { return 0; } fn main() -> i32 { return 0; }"
                .to_vec(),
            "run".into(),
        ),
        ("missing_entry", VALID_OWNED.into(), "nope".into()),
        ("empty_entry", VALID_OWNED.into(), String::new()),
        ("oversized_entry", VALID_OWNED.into(), "e".repeat(4097)),
        ("empty_source", Vec::new(), "run".into()),
        ("unparsable", b"fn (".to_vec(), "run".into()),
        (
            "type_error",
            b"fn run() -> i32 { return true; } fn main() -> i32 { return 0; }".to_vec(),
            "run".into(),
        ),
        (
            "invalid_utf8",
            vec![b'f', b'n', b' ', 0xff, 0xfe],
            "run".into(),
        ),
        ("oversized_source", vec![b' '; max + 1], "run".into()),
        (
            "max_size_unparsable",
            {
                let mut v = vec![b' '; max - 2];
                v.extend_from_slice(b"fn");
                v
            },
            "run".into(),
        ),
    ]
}

fn record(result: Result<provenance::ProvenancedArtifact, provenance::Error>) -> Value {
    match result {
        Ok(a) => json!({
            "abi": a.artifact().abi_version(),
            "target": format!("{:?}", a.artifact().target()),
            "source_bytes": a.artifact().source().len(),
        }),
        Err(e) => json!({ "error": format!("{e:?}") }),
    }
}

fn outcomes() -> serde_json::Map<String, Value> {
    let storage = compiled::OwnedLimits {
        elements: 64,
        allocations: 16,
        cleanup_events: 32,
    };
    let limits = provenance::Limits::default();
    let mut map = serde_json::Map::new();
    for (case, source, entry) in corpus() {
        for target in [
            compiled::Target::Cpu,
            compiled::Target::Cuda,
            compiled::Target::Metal,
        ] {
            map.insert(
                format!("{case}/compile_resumable_owned_numeric/{target:?}"),
                record(provenance::compile_resumable_owned_numeric(
                    &source, &entry, target, limits, storage,
                )),
            );
            map.insert(
                format!("{case}/compile_owned_numeric/{target:?}"),
                record(provenance::compile_owned_numeric(
                    &source, &entry, target, limits, storage,
                )),
            );
        }
    }
    map
}

const GOLDEN: &str = include_str!("fixtures/gnet12/pre-abi12-provenance-precedence-golden.json");

/// Stage 2b named outcome updates: the only keys that may differ from the
/// pre-ABI12 golden are valid sources admitted as ABI12 CUDA continuations
/// (their CPU counterparts are ABI11); every error outcome is unchanged.
const ADMITTED_CUDA: [&str; 2] = [
    "valid_owned_numeric/compile_resumable_owned_numeric/Cuda",
    "valid_scalar/compile_resumable_owned_numeric/Cuda",
];

#[test]
fn issue_gnet_12_stage1_precedence_with_p0b_source_sizes() {
    let current = outcomes();
    if let Some(path) = std::env::var_os("INVAR_GNET12_WRITE_PRECEDENCE") {
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
            include_str!("fixtures/gnet_entry/legacy-precedence-delta.json"),
        ),
        include_str!("fixtures/gnet_p5b/precedence-delta.json"),
    );
    assert_eq!(
        golden.keys().collect::<Vec<_>>(),
        current.keys().collect::<Vec<_>>()
    );
    for (key, want) in &golden {
        if ADMITTED_CUDA.contains(&key.as_str()) {
            assert_eq!(want["error"], "Compiler(NumericTarget)", "{key}");
            let cpu = key.replace("/Cuda", "/Cpu");
            assert_eq!(golden[&cpu]["abi"], 11, "{cpu}");
            assert_eq!(current[key]["abi"], 12, "{key}");
            assert_eq!(current[key]["target"], "Cuda", "{key}");
            continue;
        }
        assert_eq!(&current[key], want, "{key} differs from pre-ABI12");
    }
    // The precedence layers are all exercised.
    let errors: Vec<&str> = golden
        .values()
        .filter_map(|v| v["error"].as_str())
        .collect();
    for layer in [
        "Limit",
        "Source(",
        "Compiler(NumericTarget)",
        "Compiler(Entry)",
    ] {
        assert!(
            errors.iter().any(|e| e.starts_with(layer)),
            "layer {layer} not covered"
        );
    }
    eprintln!(
        "provenance precedence: {} outcomes equal to pre-ABI12 ({} errors)",
        golden.len(),
        errors.len()
    );
}

#[test]
fn issue_gnet_12_stage1_bad_sources_keep_source_errors_on_every_target() {
    // Explicit spot checks independent of the golden file.
    let storage = compiled::OwnedLimits::default();
    let limits = provenance::Limits::default();
    for target in [
        compiled::Target::Cpu,
        compiled::Target::Cuda,
        compiled::Target::Metal,
    ] {
        let oversized = vec![b' '; 1024 * 1024 + 1];
        assert_eq!(
            provenance::compile_resumable_owned_numeric(&oversized, "run", target, limits, storage)
                .unwrap_err(),
            provenance::Error::Limit,
            "{target:?}"
        );
        assert_eq!(
            provenance::compile_resumable_owned_numeric(
                VALID_OWNED.as_bytes(),
                "",
                target,
                limits,
                storage
            )
            .unwrap_err(),
            provenance::Error::Limit,
            "{target:?}"
        );
        assert!(
            matches!(
                provenance::compile_resumable_owned_numeric(
                    b"fn (", "run", target, limits, storage
                ),
                Err(provenance::Error::Source(_))
            ),
            "{target:?}"
        );
        let valid = provenance::compile_resumable_owned_numeric(
            VALID_OWNED.as_bytes(),
            "run",
            target,
            limits,
            storage,
        );
        match target {
            compiled::Target::Cpu => assert_eq!(valid.unwrap().artifact().abi_version(), 11),
            // Stage 2b named outcome update: ABI12 compile receipt only.
            compiled::Target::Cuda => assert_eq!(valid.unwrap().artifact().abi_version(), 12),
            _ => assert_eq!(
                valid.unwrap_err(),
                provenance::Error::Compiler(compiled::Error::NumericTarget),
                "{target:?}"
            ),
        }
    }
}

#[test]
fn issue_gnet_12_stage1_kernel_symbol_is_pinned_to_the_layout() {
    // A3: the device kernel symbol in emitted text and the continuation
    // layout's reported symbol must agree for every continuation artifact.
    use invar::{check, front, general_net};
    let checked = check::check(front::parse(VALID_OWNED.as_bytes()).unwrap(), false).unwrap();
    invar::ownership::check(&checked).unwrap();
    let net = general_net::Net::lower(&checked, general_net::Limits::default()).unwrap();
    let proof = net.verify().unwrap();
    let storage = compiled::OwnedLimits::default();
    let abi12 = compiled::emit_resumable_owned_numeric(
        &net,
        &proof,
        "run",
        compiled::Target::Cuda,
        compiled::Limits::default(),
        storage,
    )
    .unwrap();
    let scalar_checked =
        check::check(front::parse(VALID_SCALAR.as_bytes()).unwrap(), false).unwrap();
    invar::ownership::check(&scalar_checked).unwrap();
    let scalar_net =
        general_net::Net::lower(&scalar_checked, general_net::Limits::default()).unwrap();
    let scalar_proof = scalar_net.verify().unwrap();
    let mut checked_kernels = Vec::new();
    for target in [compiled::Target::Cuda, compiled::Target::Metal] {
        let abi3 = compiled::emit_resumable(
            &scalar_net,
            &scalar_proof,
            "run",
            target,
            compiled::Limits::default(),
            storage,
        )
        .unwrap();
        checked_kernels.push((abi3.abi_version(), abi3));
    }
    checked_kernels.push((abi12.abi_version(), abi12));
    for (abi, artifact) in &checked_kernels {
        let name = artifact.continuation_layout().unwrap().kernel_name();
        let declaration = match artifact.target() {
            compiled::Target::Metal => format!("kernel void {name}("),
            _ => format!("extern \"C\" __global__ void {name}("),
        };
        assert_eq!(
            artifact.source().matches(&declaration).count(),
            1,
            "ABI{abi}"
        );
        assert_eq!(
            name,
            if *abi == 12 {
                "invar_gnet_owned_numeric_resume_v1"
            } else {
                "invar_gnet_resume_v3"
            }
        );
    }
}

#[path = "support/gnet_p0b_delta.rs"]
mod p0b_delta;

#[path = "support/gnet_p5b_delta.rs"]
mod p5b_delta;
