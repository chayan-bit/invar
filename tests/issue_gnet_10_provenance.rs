#![forbid(unsafe_code)]
use invar::{
    eval::Value,
    general_net::{
        compiled, oracle,
        provenance::{self, Error, Limits},
    },
    numeric_contract::F32Bits,
};
const SOURCE: &[u8] = b"fn make(x:f32,n:u32)->Array<f32>{return filled_f32(n,x);}";
fn storage() -> compiled::OwnedLimits {
    compiled::OwnedLimits {
        elements: 16,
        allocations: 4,
        cleanup_events: 8,
    }
}
fn compile(source: &[u8]) -> provenance::ProvenancedArtifact {
    provenance::compile_resumable_owned_numeric(
        source,
        "make",
        compiled::Target::Cpu,
        Limits::default(),
        storage(),
    )
    .unwrap()
}
#[test]
fn owned_numeric_provenance_binds_ten_word_layout_kind_source_and_storage() {
    let a = compile(SOURCE);
    let b = compile(SOURCE);
    assert_eq!(a.id(), b.id());
    assert_eq!(a.manifest(), b.manifest());
    assert_eq!(a.artifact().abi_version(), 11);
    assert_eq!(a.manifest()["numeric"]["semantic_id"], "strict-f32-v1");
    assert_eq!(a.manifest()["numeric"]["output_words"], 10);
    assert_eq!(a.manifest()["storage"]["output_words"], 10);
    assert_eq!(a.manifest()["numeric"]["array_f32"]["kind"], 258);
    assert_eq!(
        a.manifest()["numeric"]["array_f32"]["element_word_bytes"],
        8
    );
    assert_eq!(
        a.manifest()["numeric"]["array_f32"]["external_owner_inputs"],
        false
    );
    assert!(a.artifact().continuation_layout().is_some());
    assert!(a.artifact().device_layout().is_none());
    assert!(a.artifact().aggregate_layout().is_none());
    assert!(a.artifact().speculation_layout().is_none());
    for field in [
        "numeric", "layouts", "params", "result", "source", "artifact", "limits",
    ] {
        let mut changed = a.manifest().clone();
        changed[field] = serde_json::json!("forged");
        assert_eq!(a.verify_manifest(&changed), Err(Error::ManifestMismatch));
    }
    let changed = [SOURCE, b" "].concat();
    let other = compile(&changed);
    assert_ne!(a.id(), other.id());
    assert_eq!(a.verify_source(&changed), Err(Error::SourceMismatch));
    let mut smaller = storage();
    smaller.elements -= 1;
    let other = provenance::compile_resumable_owned_numeric(
        SOURCE,
        "make",
        compiled::Target::Cpu,
        Limits::default(),
        smaller,
    )
    .unwrap();
    assert_ne!(a.id(), other.id());
}
#[test]
fn owned_numeric_oracle_receipt_freezes_exact_array_bits_and_failure_prefixes() {
    let a = compile(SOURCE);
    for bits in [
        0, 0x80000000, 1, 0x7f800000, 0xff800000, 0x7fa12345, 0xffc12345,
    ] {
        for n in [0, 3] {
            let args = [Value::F32(F32Bits::from_bits(bits)), Value::U32(n)];
            let execution = a.observe_oracle(&args, oracle::Limits::default()).unwrap();
            assert_eq!(
                execution.manifest()["result_element_bits"],
                serde_json::json!(vec![bits; n as usize])
            );
            assert_eq!(execution.manifest()["continuation_execution"], false);
            assert_eq!(execution.manifest()["backend"], "cpu_net_oracle");
            let original_id = execution.id().to_owned();
            let manifest = execution.manifest().clone();
            let Value::ArrayF32(owner) = execution.observation().semantic.result.unwrap() else {
                panic!("array")
            };
            if n != 0 {
                owner.borrow_mut()[0] = F32Bits::from_bits(17);
            }
            let Value::ArrayF32(fresh) = execution.observation().semantic.result.unwrap() else {
                panic!("array")
            };
            assert_eq!(
                fresh
                    .borrow()
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                vec![bits; n as usize]
            );
            assert_eq!(execution.id(), original_id);
            assert_eq!(execution.manifest(), &manifest);
            let steps = execution.observation().semantic.steps_used;
            for fuel in 1..=steps + 1 {
                let observed = a
                    .observe_oracle(
                        &args,
                        oracle::Limits {
                            fuel,
                            ..oracle::Limits::default()
                        },
                    )
                    .unwrap();
                assert!(observed.observation().semantic.steps_used <= fuel);
                if fuel < steps {
                    assert!(observed.observation().semantic.result.is_err());
                }
            }
        }
    }
}
#[test]
fn owned_numeric_provenance_negative_admission() {
    assert!(matches!(
        provenance::compile_resumable_owned_numeric(
            SOURCE,
            "make",
            compiled::Target::Metal,
            Limits::default(),
            storage()
        ),
        Err(Error::Compiler(compiled::Error::NumericTarget))
    ));
    assert!(
        provenance::compile_owned(
            SOURCE,
            "make",
            compiled::Target::Cpu,
            Limits::default(),
            storage()
        )
        .is_err()
    );
    for source in [
        b"fn make(x:read Array<f32>)->f32{return x[0u32];}".as_slice(),
        b"struct Numeric{x:f32} fn make()->i32{return 7;}".as_slice(),
    ] {
        assert!(matches!(
            provenance::compile_resumable_owned_numeric(
                source,
                "make",
                compiled::Target::Cpu,
                Limits::default(),
                storage()
            ),
            Err(Error::Compiler(compiled::Error::UnsupportedType))
        ));
    }
}
