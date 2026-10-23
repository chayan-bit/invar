#![forbid(unsafe_code)]
use invar::{
    eval,
    general_net::{
        compiled::Target,
        oracle,
        provenance::{self, Error, Limits},
    },
    numeric_contract::F32Bits,
};
const SOURCE: &[u8] = b"fn echo(x:f32)->f32{return x;} fn neg(x:f32)->f32{return -x;}";
/// The only CPU numeric profile: the ABI11 owned-numeric continuation.
fn compile_numeric(
    source: &[u8],
    entry: &str,
    target: Target,
    limits: Limits,
) -> Result<provenance::ProvenancedArtifact, Error> {
    provenance::compile_resumable_owned_numeric(
        source,
        entry,
        target,
        limits,
        invar::general_net::compiled::OwnedLimits::default(),
    )
}
#[test]
fn numeric_compile_receipt_binds_exact_source_policy_types_limits_and_emitted_bytes() {
    let a = compile_numeric(SOURCE, "echo", Target::Cpu, Limits::default()).unwrap();
    let b = compile_numeric(SOURCE, "echo", Target::Cpu, Limits::default()).unwrap();
    assert_eq!(a.id(), b.id());
    assert_eq!(a.manifest(), b.manifest());
    assert_eq!(a.artifact().abi_version(), 11);
    assert_eq!(a.manifest()["numeric"]["semantic_id"], "strict-f32-v1");
    assert_eq!(a.manifest()["numeric"]["policy"]["version"], 1);
    assert_eq!(a.manifest()["params"][0], "F32");
    assert_eq!(a.manifest()["result"], "F32");
    assert!(a.manifest().get("observed").is_none());
    a.verify_artifact_bytes(a.artifact().source().as_bytes())
        .unwrap();
    let (net, proof) = b.checked_net();
    a.verify_net(net, proof).unwrap();
    for path in [
        "numeric", "target", "params", "result", "source", "artifact", "limits",
    ] {
        let mut changed = a.manifest().clone();
        changed[path] = serde_json::json!("forged");
        assert_eq!(a.verify_manifest(&changed), Err(Error::ManifestMismatch));
    }
    let changed = [SOURCE, b" "].concat();
    assert_eq!(a.verify_source(&changed), Err(Error::SourceMismatch));
    assert_ne!(
        a.id(),
        compile_numeric(&changed, "echo", Target::Cpu, Limits::default())
            .unwrap()
            .id()
    );
    assert_ne!(
        a.id(),
        compile_numeric(SOURCE, "neg", Target::Cpu, Limits::default())
            .unwrap()
            .id()
    );
    let mut limits = Limits::default();
    limits.compiler.frame_bytes /= 2;
    assert_ne!(
        a.id(),
        compile_numeric(SOURCE, "echo", Target::Cpu, limits)
            .unwrap()
            .id()
    );
}
#[test]
fn numeric_oracle_receipt_snapshots_exact_f32_bits_and_source_failure_prefixes() {
    let a = compile_numeric(SOURCE, "echo", Target::Cpu, Limits::default()).unwrap();
    for bits in [
        0, 0x80000000, 1, 0x7f800000, 0xff800000, 0x7fc00123, 0xffc00123,
    ] {
        let execution = a
            .observe_oracle(
                &[eval::Value::F32(F32Bits::from_bits(bits))],
                oracle::Limits::default(),
            )
            .unwrap();
        assert_eq!(execution.manifest()["backend"], "cpu_net_oracle");
        assert_eq!(execution.manifest()["continuation_execution"], false);
        let eval::Value::F32(value) = execution.observation().semantic.result.unwrap() else {
            panic!("wrong type")
        };
        assert_eq!(value.to_bits(), bits);
        let mut changed = execution.manifest().clone();
        changed["backend"] = serde_json::json!("compiled_cpu");
        assert_eq!(
            execution.verify_manifest(&changed),
            Err(Error::ManifestMismatch)
        );
    }
    let limits = oracle::Limits {
        fuel: 1,
        ..oracle::Limits::default()
    };
    let execution = a
        .observe_oracle(&[eval::Value::F32(F32Bits::POS_ZERO)], limits)
        .unwrap();
    assert!(execution.observation().semantic.result.is_err());
    assert_eq!(execution.observation().semantic.steps_used, 1);
}
#[test]
fn numeric_provenance_keeps_legacy_and_unsupported_owner_device_gates_explicit() {
    assert!(provenance::compile(SOURCE, "echo", Target::Cpu, Limits::default()).is_err());
    assert!(matches!(
        compile_numeric(SOURCE, "echo", Target::Metal, Limits::default()),
        Err(Error::Compiler(
            invar::general_net::compiled::Error::NumericTarget
        ))
    ));
    let nominal = b"struct Numeric{x:f32} fn owner()->i32{return 7;}";
    assert!(compile_numeric(nominal, "owner", Target::Cpu, Limits::default()).is_err());
    let legacy = provenance::compile(
        b"fn id(x:u32)->u32{return x;}",
        "id",
        Target::Cpu,
        Limits::default(),
    )
    .unwrap();
    assert!(legacy.manifest().get("numeric").is_none());
    assert_eq!(legacy.artifact().abi_version(), 1);
}
