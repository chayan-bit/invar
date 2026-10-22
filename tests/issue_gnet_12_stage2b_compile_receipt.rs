//! ABI12 stage 2b: the frozen `cuda_owned_numeric_continuation` compile
//! receipt (`invar.gnet.abi12.compile-receipt` v1) and source-bound admission.
//! The receipt grants no execution, module-load, resume or qualification
//! authority. ABI1-11 receipts keep their identity; every load, execute and
//! resume route stays typed-refused for ABI12. No device, NVRTC or loader.
use invar::general_net::{compiled, provenance};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const VALID_OWNED: &str = "fn run(a: f32, n: u32) -> f32 { var xs: Array<f32> = filled_f32(n + 1u32, a); xs[0u32] = xs[0u32] * a + a; return xs[0u32]; } fn main() -> i32 { return 0; }";
const VALID_SCALAR: &str = "fn run(n: u32) -> i32 { if n < 2u32 { return 1; } return run(n - 1u32) + run(n - 2u32); } fn main() -> i32 { return run(5u32); }";
const AGGREGATE: &str = "fn make(ok: Bool) -> Result<i32,u32> { if ok { return Result::Ok(7); } return Result::Err(9u32); } fn run(ok: Bool) -> i32 { let x = make(ok); return match move x { Result::Ok(a) => a, Result::Err(e) => 9 }; } fn main() -> i32 { return run(true); }";
const NUMERIC: &str =
    "fn run(a: f32, b: f32) -> f32 { return a * b + a; } fn main() -> i32 { return 0; }";
const INT_LOOP: &str = "fn run(n: u32) -> u32 { var i: u32 = 0u32; var s: u32 = 0u32; while i < n { s = s + i; i = i + 1u32; } return s; } fn main() -> i32 { return 0; }";

fn storage() -> compiled::OwnedLimits {
    compiled::OwnedLimits {
        elements: 64,
        allocations: 16,
        cleanup_events: 32,
    }
}

const TARGETS: [compiled::Target; 3] = [
    compiled::Target::Cpu,
    compiled::Target::Cuda,
    compiled::Target::Metal,
];

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The receipt id with the build-scoped `compiler_build` field (a digest of
/// every source file and the profile, so it differs per build by design)
/// replaced by a constant. Equal normalized ids mean byte-identical manifests
/// apart from that one field.
fn normalized_id(manifest: &Value) -> String {
    let mut m = manifest.clone();
    m["compiler_build"] = json!("normalized");
    sha(m.to_string().as_bytes())
}

fn routes(
    source: &[u8],
    target: compiled::Target,
) -> Vec<(
    &'static str,
    Result<provenance::ProvenancedArtifact, provenance::Error>,
)> {
    let l = provenance::Limits::default();
    let s = storage();
    let spec = compiled::SpeculationLimits::default();
    vec![
        ("compile", provenance::compile(source, "run", target, l)),
        (
            "compile_numeric",
            provenance::compile_numeric(source, "run", target, l),
        ),
        (
            "compile_resumable_numeric",
            provenance::compile_resumable_numeric(source, "run", target, l),
        ),
        (
            "compile_owned",
            provenance::compile_owned(source, "run", target, l, s),
        ),
        (
            "compile_owned_numeric",
            provenance::compile_owned_numeric(source, "run", target, l, s),
        ),
        (
            "compile_resumable_owned_numeric",
            provenance::compile_resumable_owned_numeric(source, "run", target, l, s),
        ),
        (
            "compile_resumable",
            provenance::compile_resumable(source, "run", target, l, s),
        ),
        (
            "compile_speculative",
            provenance::compile_speculative(source, "run", target, l, s, spec),
        ),
        (
            "compile_device_speculative",
            provenance::compile_device_speculative(source, "run", target, l, s, spec),
        ),
    ]
}

fn receipt_outcomes() -> serde_json::Map<String, Value> {
    let mut map = serde_json::Map::new();
    for (name, source) in [
        ("valid_owned_numeric", VALID_OWNED),
        ("valid_scalar", VALID_SCALAR),
        ("valid_aggregate", AGGREGATE),
        ("valid_numeric", NUMERIC),
        ("valid_int_loop", INT_LOOP),
    ] {
        for target in TARGETS {
            for (route, result) in routes(source.as_bytes(), target) {
                let value = match result {
                    Ok(a) => {
                        // The real id is the digest of the real manifest.
                        assert_eq!(a.id(), sha(a.manifest().to_string().as_bytes()));
                        json!({
                            "abi": a.artifact().abi_version(),
                            "target": format!("{:?}", a.artifact().target()),
                            "normalized_id": normalized_id(a.manifest()),
                        })
                    }
                    Err(e) => json!({ "error": format!("{e:?}") }),
                };
                map.insert(format!("{name}/{route}/{target:?}"), value);
            }
        }
    }
    map
}

const RECEIPT_GOLDEN: &str = include_str!("fixtures/gnet12/pre-2b-receipt-ids-golden.json");

/// The only outcomes 2b may change: valid sources whose resumable owned
/// numeric continuation the CUDA emitter admits (pre-2b `NumericTarget`); each
/// one's CPU counterpart is an ABI11 receipt in the same golden.
const ADMITTED_CUDA: [&str; 4] = [
    "valid_int_loop/compile_resumable_owned_numeric/Cuda",
    "valid_numeric/compile_resumable_owned_numeric/Cuda",
    "valid_owned_numeric/compile_resumable_owned_numeric/Cuda",
    "valid_scalar/compile_resumable_owned_numeric/Cuda",
];

#[test]
fn issue_gnet_12_stage2b_receipts_pin_p0b_changes() {
    let current = receipt_outcomes();
    if let Some(path) = std::env::var_os("INVAR_GNET12_WRITE_RECEIPT_IDS") {
        std::fs::write(
            path,
            serde_json::to_string_pretty(&Value::Object(current)).unwrap() + "\n",
        )
        .unwrap();
        return;
    }
    let golden = p5b_delta::apply(
        p0b_delta::apply(
            RECEIPT_GOLDEN,
            include_str!("fixtures/gnet_entry/legacy-receipt-delta.json"),
        ),
        include_str!("fixtures/gnet_p5b/receipt-delta.json"),
    );
    assert_eq!(
        golden.keys().collect::<Vec<_>>(),
        current.keys().collect::<Vec<_>>()
    );
    let mut abi1_11 = 0;
    for (key, want) in &golden {
        if ADMITTED_CUDA.contains(&key.as_str()) {
            assert_eq!(
                want,
                &json!({"error": "Compiler(NumericTarget)"}),
                "{key} before 2b"
            );
            assert_eq!(golden[&key.replace("/Cuda", "/Cpu")]["abi"], 11, "{key}");
            assert_eq!(current[key]["abi"], 12, "{key}");
            assert_eq!(current[key]["target"], "Cuda", "{key}");
            continue;
        }
        assert_eq!(&current[key], want, "{key} differs from pre-2b");
        abi1_11 += usize::from(want.get("abi").is_some());
    }
    let mut abis: Vec<u64> = golden.values().filter_map(|v| v["abi"].as_u64()).collect();
    abis.sort_unstable();
    abis.dedup();
    eprintln!(
        "receipt ids: {} outcomes, {abi1_11} ABI1-11 receipts byte-identical (abis {abis:?})",
        golden.len()
    );
}

const KEY: &str = "cuda_owned_numeric_continuation";
const ABI11_KEY: &str = "numeric_owned_continuation";
const BLOCK_V1: &str = include_str!("fixtures/gnet12/abi12-compile-receipt-block-v1.json");

fn admitted(source: &str) -> provenance::ProvenancedArtifact {
    provenance::compile_resumable_owned_numeric(
        source.as_bytes(),
        "run",
        compiled::Target::Cuda,
        provenance::Limits::default(),
        storage(),
    )
    .unwrap()
}

fn cpu(source: &str) -> provenance::ProvenancedArtifact {
    provenance::compile_resumable_owned_numeric(
        source.as_bytes(),
        "run",
        compiled::Target::Cpu,
        provenance::Limits::default(),
        storage(),
    )
    .unwrap()
}

#[test]
fn issue_gnet_12_stage2b_block_equals_frozen_schema_and_ids_are_exact() {
    let a = admitted(VALID_OWNED);
    let m = a.manifest();
    assert_eq!(a.artifact().abi_version(), 12);
    // The hand-derived frozen block (schema v1, independent of the builder).
    let frozen: Value = serde_json::from_str(BLOCK_V1).unwrap();
    assert_eq!(m[KEY], frozen);
    // Existing top-level identities are kept and bound to the live inputs.
    assert_eq!(m["version"], 3);
    assert_eq!(m["scope"], "checked_source_net_emitted_source");
    assert_eq!(m["target"], "Cuda");
    assert_eq!(m["abi_version"], 12);
    assert_eq!(m["source"]["sha256"], sha(VALID_OWNED.as_bytes()));
    assert_eq!(m["net"]["sha256"], sha(a.net_snapshot()));
    assert_eq!(
        m["artifact"]["sha256"],
        sha(a.artifact().source().as_bytes())
    );
    assert_eq!(
        m["layouts"]["continuation"]["kernel_name"],
        m[KEY]["kernel"]["symbol"]
    );
    assert_eq!(m["numeric"]["array_f32"]["kind"], 258);
    assert!(m["compiler_build"].as_str().is_some_and(|b| b.len() == 64));
    assert_eq!(a.id(), sha(m.to_string().as_bytes()));
    // The block's facts agree with the artifact's own layouts.
    let continuation = a.artifact().continuation_layout().unwrap();
    assert_eq!(m[KEY]["task"]["task_bytes"], continuation.task_bytes());
    assert_eq!(
        m[KEY]["workset"]["per_lane_words"]["tasks"],
        continuation.task_bytes() / 8
    );
    let owned = a.artifact().owned_layout().unwrap();
    assert_eq!(
        m[KEY]["workset"]["per_lane_words"]["heap"],
        owned.heap_words()
    );
    assert_eq!(
        m[KEY]["workset"]["per_lane_words"]["cleanup"],
        owned.cleanup_words()
    );
    // Deterministic: a fresh compile gives the same receipt and id.
    let again = admitted(VALID_OWNED);
    assert_eq!((again.id(), again.manifest()), (a.id(), m));
    // Exact revalidation against live objects.
    a.verify_manifest(m).unwrap();
    a.verify_source(VALID_OWNED.as_bytes()).unwrap();
    let (net, proof) = a.checked_net();
    a.verify_net(net, proof).unwrap();
    a.verify_artifact_bytes(a.artifact().source().as_bytes())
        .unwrap();
}

#[test]
fn issue_gnet_12_stage2b_forged_omitted_mixed_and_mutated_receipts_are_rejected() {
    use provenance::Error::{ArtifactMismatch, ManifestMismatch, NetMismatch, SourceMismatch};
    let a = admitted(VALID_OWNED);
    let base = a.manifest().clone();
    let abi11 = cpu(VALID_OWNED).manifest()[ABI11_KEY].clone();
    type Mutation = (&'static str, Box<dyn Fn(&mut Value)>);
    let set = |path: &'static [&'static str], v: Value| -> Box<dyn Fn(&mut Value)> {
        Box::new(move |m: &mut Value| {
            let mut at = m;
            for p in path {
                at = match p.parse::<usize>() {
                    Ok(i) if at.is_array() => &mut at[i],
                    _ => &mut at[*p],
                };
            }
            *at = v.clone();
        })
    };
    let abi11_block = abi11.clone();
    let abi11_mixed = abi11.clone();
    let mutations: Vec<Mutation> = vec![
        (
            "omitted block",
            Box::new(|m: &mut Value| {
                m.as_object_mut().unwrap().remove(KEY);
            }),
        ),
        (
            "mixed with ABI11 block",
            Box::new(move |m: &mut Value| m[ABI11_KEY] = abi11_mixed.clone()),
        ),
        (
            "ABI11 block forged under the ABI12 key",
            Box::new(move |m: &mut Value| m[KEY] = abi11_block.clone()),
        ),
        (
            "ABI12 block renamed to the ABI11 key",
            Box::new(|m: &mut Value| {
                let block = m.as_object_mut().unwrap().remove(KEY).unwrap();
                m[ABI11_KEY] = block;
            }),
        ),
        (
            "schema",
            set(&[KEY, "schema"], json!("invar.gnet.abi11.compile-receipt")),
        ),
        ("schema_version", set(&[KEY, "schema_version"], json!(2))),
        (
            "profile",
            set(
                &[KEY, "profile"],
                json!("cpu-owned-numeric-continuation-v1"),
            ),
        ),
        ("block abi", set(&[KEY, "abi_version"], json!(11))),
        ("block target", set(&[KEY, "target"], json!("Metal"))),
        (
            "authority.execution",
            set(&[KEY, "authority", "execution"], json!(true)),
        ),
        (
            "authority.module_load",
            set(&[KEY, "authority", "module_load"], json!(true)),
        ),
        (
            "authority.resume",
            set(&[KEY, "authority", "resume"], json!(true)),
        ),
        (
            "authority.device_qualification",
            set(&[KEY, "authority", "device_qualification"], json!(true)),
        ),
        (
            "device support claim in block",
            set(&[KEY, "device_support"], json!(true)),
        ),
        (
            "top-level qualification claim",
            set(&["device_qualification"], json!(true)),
        ),
        (
            "nvrtc invoked",
            set(&[KEY, "toolchain", "nvrtc"], json!("12.4")),
        ),
        (
            "cubin claimed",
            set(&[KEY, "toolchain", "cubin"], json!({"sha256": "00"})),
        ),
        (
            "kernel symbol",
            set(&[KEY, "kernel", "symbol"], json!("invar_gnet_resume_v3")),
        ),
        (
            "declarations",
            set(&[KEY, "kernel", "declarations"], json!(2)),
        ),
        (
            "signature",
            set(&[KEY, "kernel", "signature"], json!("void()")),
        ),
        (
            "block threads",
            set(&[KEY, "launch", "block_threads"], json!(64)),
        ),
        ("max roots", set(&[KEY, "launch", "max_roots"], json!(2048))),
        (
            "buffer order",
            set(
                &[KEY, "workset", "buffers"],
                json!([
                    "input", "output", "trace", "config", "heap", "tasks", "cleanup"
                ]),
            ),
        ),
        (
            "red zone",
            set(&[KEY, "workset", "red_zone_words"], json!(0)),
        ),
        (
            "per-lane tasks",
            set(&[KEY, "workset", "per_lane_words", "tasks"], json!(1491)),
        ),
        (
            "per-lane input",
            set(&[KEY, "workset", "per_lane_words", "input"], json!(1)),
        ),
        (
            "config bound",
            set(
                &[KEY, "workset", "config_bounds", "roots"],
                json!([1, 1025]),
            ),
        ),
        (
            "device cap +8",
            set(&[KEY, "workset", "max_device_bytes"], json!(67_108_872)),
        ),
        (
            "host snapshot cap",
            set(
                &[KEY, "workset", "max_host_snapshot_bytes"],
                json!(268_435_457),
            ),
        ),
        (
            "retained snapshots",
            set(&[KEY, "workset", "min_retained_snapshots"], json!(1)),
        ),
        (
            "task bytes",
            set(&[KEY, "task", "task_bytes"], json!(11928)),
        ),
        (
            "task field offset",
            set(&[KEY, "task", "task_fields", "0", "offset"], json!(0)),
        ),
        (
            "sentinel value",
            set(&[KEY, "output", "pending_sentinel", "value"], json!(0)),
        ),
        (
            "sentinel words",
            set(
                &[KEY, "output", "pending_sentinel", "words"],
                json!("all 9"),
            ),
        ),
        (
            "sentinel omitted",
            Box::new(|m: &mut Value| {
                m[KEY]["output"]
                    .as_object_mut()
                    .unwrap()
                    .remove("pending_sentinel");
            }),
        ),
        (
            "advance rule",
            set(&[KEY, "advance_rule"], json!("native-call")),
        ),
        (
            "numeric implementation",
            set(&[KEY, "numeric", "implementation"], json!("hardware-f32")),
        ),
        (
            "canonical NaN",
            set(
                &[KEY, "numeric", "canonical_nan_bits"],
                json!(0x7fc0_0001u32),
            ),
        ),
        (
            "semantic reference",
            set(&[KEY, "semantic_reference"], json!("abi12")),
        ),
        ("top-level abi", set(&["abi_version"], json!(11))),
        ("top-level target", set(&["target"], json!("Cpu"))),
        ("source digest", set(&["source", "sha256"], json!("00"))),
        ("net digest", set(&["net", "sha256"], json!("00"))),
        (
            "emitted-code digest",
            set(&["artifact", "sha256"], json!("00")),
        ),
        ("compiler build", set(&["compiler_build"], json!("00"))),
    ];
    for (name, mutate) in &mutations {
        let mut forged = base.clone();
        mutate(&mut forged);
        assert_ne!(forged, base, "{name} did not mutate");
        assert_eq!(a.verify_manifest(&forged), Err(ManifestMismatch), "{name}");
    }
    // A genuine receipt of a different admitted program is not this one's.
    let other = admitted(VALID_SCALAR);
    assert_eq!(a.verify_manifest(other.manifest()), Err(ManifestMismatch));
    assert_eq!(
        a.verify_manifest(cpu(VALID_OWNED).manifest()),
        Err(ManifestMismatch)
    );
    // Live-object revalidation.
    assert_eq!(
        a.verify_source(VALID_SCALAR.as_bytes()),
        Err(SourceMismatch)
    );
    let (net, proof) = other.checked_net();
    assert_eq!(a.verify_net(net, proof), Err(NetMismatch));
    let mut text = a.artifact().source().to_owned();
    text.push(' ');
    assert_eq!(
        a.verify_artifact_bytes(text.as_bytes()),
        Err(ArtifactMismatch)
    );
    eprintln!("forged receipts rejected: {}", mutations.len() + 2);
}

#[test]
fn issue_gnet_12_stage2b_blocks_are_exclusive_and_claim_no_device_support() {
    for source in [VALID_OWNED, VALID_SCALAR, NUMERIC, INT_LOOP] {
        let gpu = admitted(source);
        let host = cpu(source);
        let (g, h) = (gpu.manifest(), host.manifest());
        assert!(g.get(KEY).is_some() && g.get(ABI11_KEY).is_none());
        assert!(h.get(ABI11_KEY).is_some() && h.get(KEY).is_none());
        assert_ne!(gpu.id(), host.id());
        assert_ne!(g[KEY]["schema"], h[ABI11_KEY]["profile"]);
        // No compile metadata claims device support or qualification: every
        // boolean anywhere in the receipt is false, and no module/CUBIN exists.
        fn booleans(v: &Value, out: &mut Vec<(String, bool)>, at: String) {
            match v {
                Value::Bool(b) => out.push((at, *b)),
                Value::Array(xs) => xs
                    .iter()
                    .enumerate()
                    .for_each(|(i, x)| booleans(x, out, format!("{at}/{i}"))),
                Value::Object(o) => o
                    .iter()
                    .for_each(|(k, x)| booleans(x, out, format!("{at}/{k}"))),
                _ => {}
            }
        }
        let mut found = Vec::new();
        booleans(g, &mut found, String::new());
        assert!(found.iter().all(|(_, b)| !b), "{found:?}");
        assert_eq!(
            g[KEY]["authority"],
            json!({"execution": false, "module_load": false, "resume": false,
                "device_qualification": false})
        );
        assert_eq!(
            g[KEY]["toolchain"],
            json!({"nvrtc": "not-invoked", "cubin": null})
        );
        let text = g.to_string();
        for claim in ["qualified", "device_support", "cubin_sha", "module_receipt"] {
            assert!(!text.contains(claim), "{claim}");
        }
    }
    // Every ABI1-11 receipt in the corpus lacks the ABI12 block.
    for (name, source) in [("owned", VALID_OWNED), ("scalar", VALID_SCALAR)] {
        for target in TARGETS {
            for (route, result) in routes(source.as_bytes(), target) {
                if let Ok(a) = result
                    && a.artifact().abi_version() != 12
                {
                    assert!(a.manifest().get(KEY).is_none(), "{name} {route}");
                }
            }
        }
    }
}

#[test]
fn issue_gnet_12_stage2b_invalid_sources_keep_exact_errors_on_cuda() {
    use provenance::Error;
    let l = provenance::Limits::default();
    let s = storage();
    let route = |source: &[u8], entry: &str, target| {
        provenance::compile_resumable_owned_numeric(source, entry, target, l, s)
    };
    for target in TARGETS {
        assert_eq!(
            route(&vec![b' '; 1024 * 1024 + 1], "run", target).unwrap_err(),
            Error::Limit
        );
        assert_eq!(
            route(VALID_OWNED.as_bytes(), "", target).unwrap_err(),
            Error::Limit
        );
        assert!(matches!(
            route(b"fn (", "run", target),
            Err(Error::Source(_))
        ));
        assert!(matches!(
            route(
                b"fn run() -> i32 { return true; } fn main() -> i32 { return 0; }",
                "run",
                target
            ),
            Err(Error::Source(_))
        ));
    }
    // CUDA emission refusals keep the pre-ABI12 typed outcome; Metal refused.
    for (source, entry) in [(AGGREGATE, "run"), (VALID_OWNED, "nope")] {
        for target in [compiled::Target::Cuda, compiled::Target::Metal] {
            assert_eq!(
                route(source.as_bytes(), entry, target).unwrap_err(),
                Error::Compiler(compiled::Error::NumericTarget),
                "{entry} {target:?}"
            );
        }
    }
    assert_eq!(
        route(VALID_OWNED.as_bytes(), "run", compiled::Target::Metal).unwrap_err(),
        Error::Compiler(compiled::Error::NumericTarget)
    );
}

#[test]
fn issue_gnet_12_stage2b_every_execution_gate_refuses_abi12() {
    use invar::{eval, general_net::oracle, numeric_contract::F32Bits};
    use provenance::{Error, FenceBackend, NumericContinuationComponents};
    let a = admitted(VALID_OWNED);
    // ABI5 fence gate: an ABI12 artifact has no device layout. The ABI5
    // workset uses the issue_gnet_05 configuration.
    let abi5 = provenance::compile_device_speculative(
        b"fn branch(depth: u32) -> i32 { if depth<2u32 { return 1; } return branch(depth-1u32)+branch(depth-2u32); } fn calc(n: u32) -> i32 { return branch(n); }",
        "calc",
        compiled::Target::Cuda,
        provenance::Limits::default(),
        compiled::OwnedLimits {
            elements: 16,
            allocations: 8,
            cleanup_events: 16,
        },
        compiled::SpeculationLimits {
            slots: 2,
            fuel: 10000,
            transitions: 10000,
            trace_events: 1024,
            quantum: 32,
            lookahead: 128,
            scratch_bytes: 1024 * 1024,
        },
    )
    .unwrap();
    let layout = abi5.artifact().device_layout().unwrap();
    let config = compiled::DeviceConfig {
        fuel: 10000,
        transitions: 10000,
        trace_events: 1024,
        quantum: 32,
        rounds: 64,
        capacity: 2,
    };
    let bytes = layout.buffer_bytes(config).unwrap();
    let workset = layout
        .workset(
            config,
            std::array::from_fn(|i| compiled::BufferRange {
                device: 1,
                allocation: (i + 1) as u64,
                offset: 0,
                bytes: bytes[i],
            }),
        )
        .unwrap();
    for backend in [
        FenceBackend::CudaDriver,
        FenceBackend::MetalCommand,
        FenceBackend::CpuGroupModel,
    ] {
        assert!(
            matches!(
                a.fenced_execution(workset.clone(), (), 1, backend),
                Err(Error::ExecutionMismatch)
            ),
            "{backend:?}"
        );
    }
    // Continuation identity (B4): refused for every task length.
    let task_bytes = a.artifact().continuation_layout().unwrap().task_bytes();
    for len in [0, task_bytes, task_bytes + 8] {
        assert!(matches!(
            a.continuation_identity(&vec![0; len]),
            Err(Error::ContinuationMismatch)
        ));
    }
    // CPU component identities (ABI8 / ABI9 only).
    let args = [
        eval::Value::F32(F32Bits::from_bits(0x3f80_0000)),
        eval::Value::U32(3),
    ];
    assert!(matches!(
        a.numeric_cpu_identity(
            b"executable",
            &args,
            oracle::Limits::default(),
            &[0; compiled::OUTPUT_WORDS],
            &[]
        ),
        Err(Error::ExecutionMismatch)
    ));
    let task = vec![0; task_bytes];
    assert!(matches!(
        a.numeric_continuation_identity(NumericContinuationComponents {
            executable: b"executable",
            args: &args,
            limits: oracle::Limits::default(),
            quantum: 1,
            task: &task,
            output: None,
            trace: &[],
        }),
        Err(Error::ExecutionMismatch)
    ));
    // Device host: P0b uses ABI13; the CPU group model
    // backend refuses the ABI12 artifact explicitly before invoking any
    // compiler (follow-up named boundary change from "missing private layout").
    use invar::general_net::device_host::{self, Backend};
    assert_eq!(
        device_host::Plan::from_source(
            VALID_OWNED.as_bytes(),
            "run",
            compiled::Target::Cuda,
            device_host::Limits::default()
        )
        .unwrap()
        .artifact()
        .abi_version(),
        13
    );
    let mut model = device_host::cpu::CpuModel::with_compiler(std::path::Path::new(
        "/nonexistent/invar-stage2b-no-compiler",
    ));
    let refused = model.prepare(a.artifact(), [8; 12], &[], [0; 6]);
    assert_eq!(
        refused.err().as_deref(),
        Some(device_host::cpu::UNSUPPORTED_ABI12)
    );
    // The CPU net oracle stays available, labelled as oracle only.
    let observed = a.observe_oracle(&args, oracle::Limits::default()).unwrap();
    let m = observed.manifest();
    assert_eq!(m["backend"], "cpu_net_oracle");
    assert_eq!(m["continuation_execution"], false);
    assert_eq!(m["artifact_target"], "Cuda");
    assert_eq!(m["abi_version"], 12);
}

#[path = "support/gnet_p0b_delta.rs"]
mod p0b_delta;

#[path = "support/gnet_p5b_delta.rs"]
mod p5b_delta;
