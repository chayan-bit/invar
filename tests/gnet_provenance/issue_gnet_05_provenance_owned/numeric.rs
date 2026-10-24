use super::*;

#[test]
fn issue_gnet_05_provenance_numeric_exact_policy_bits_layout_and_all_compile_budgets() {
    let a = numeric(NUMERIC);
    let b = numeric(NUMERIC);
    assert_eq!(a.id(), b.id());
    assert_eq!(a.manifest(), b.manifest());
    assert_eq!(a.artifact().abi_version(), 11);
    assert_eq!(
        a.artifact().numeric_policy(),
        Some(NumericPolicy::STRICT_F32)
    );
    assert_eq!(
        a.manifest()["numeric"]["policy"],
        NumericPolicy::STRICT_F32.to_json()
    );
    assert_eq!(a.manifest()["numeric"]["semantic_id"], "strict-f32-v1");
    assert_eq!(
        a.manifest()["numeric"]["layout_version"],
        invar::layout_contract::VERSION
    );
    assert_eq!(a.manifest()["numeric"]["f32"]["maximum"], u32::MAX);
    assert_eq!(a.manifest()["numeric"]["output_words"], 10);
    assert_eq!(
        a.manifest()["numeric"]["event_words"],
        compiled::EVENT_WORDS
    );
    assert_eq!(a.manifest()["params"], json!(["F32", "F32", "F32"]));
    assert_eq!(a.manifest()["result"], "F32");
    assert!(a.manifest().get("observed").is_none());
    a.verify_artifact_bytes(a.artifact().source().as_bytes())
        .unwrap();
    let (net, proof) = b.checked_net();
    a.verify_net(net, proof).unwrap();
    assert!(a.artifact().owned_layout().is_some());
    assert!(a.artifact().continuation_layout().is_some());
    assert!(a.artifact().speculation_layout().is_none());
    assert!(a.artifact().device_layout().is_none());
    assert!(a.artifact().aggregate_layout().is_none());
    for field in 0..10 {
        let mut l = Limits::default();
        match field {
            0 => l.net.functions += 1,
            1 => l.net.blocks += 1,
            2 => l.net.instructions += 1,
            3 => l.net.wires += 1,
            4 => l.net.literal_bytes += 1,
            5 => l.compiler.values += 1,
            6 => l.compiler.places += 1,
            7 => l.compiler.states += 1,
            8 => l.compiler.frame_bytes += 1,
            9 => l.compiler.source_bytes += 1,
            _ => unreachable!(),
        }
        let b = numeric_with(NUMERIC, compiled::Target::Cpu, l).unwrap();
        assert_ne!(a.id(), b.id(), "compile budget {field}");
        assert_eq!(a.artifact().source(), b.artifact().source());
    }
    // Non-numeric manifests carry no numeric block; numeric policy is added
    // only when the actual emitter identifies a numeric artifact.
    let legacy = provenance::compile(
        b"fn calc() -> i32 { return 7; }",
        "calc",
        compiled::Target::Cpu,
        Limits::default(),
    )
    .unwrap();
    assert_eq!(legacy.manifest()["version"], 3);
    assert!(legacy.manifest().get("numeric").is_none());
    assert!(
        legacy
            .observe_oracle(&[], numeric_limits())
            .unwrap()
            .manifest()
            .get("numeric")
            .is_none()
    );
}

#[test]
fn issue_gnet_05_provenance_numeric_stale_mutations_and_unsupported_modes_reject() {
    let a = numeric(NUMERIC);
    for forged in forged_leaves(a.manifest()) {
        assert_eq!(a.verify_manifest(&forged), Err(Error::ManifestMismatch));
    }
    let changed = numeric_separate(NUMERIC);
    let b = numeric(&changed);
    assert_ne!(a.id(), b.id());
    assert_eq!(a.verify_source(&changed), Err(Error::SourceMismatch));
    assert_eq!(
        a.verify_artifact_bytes(b.artifact().source().as_bytes()),
        Err(Error::ArtifactMismatch)
    );
    let (net, proof) = b.checked_net();
    assert_eq!(a.verify_net(net, proof), Err(Error::NetMismatch));
    let (net, proof) = a.checked_net();
    let mut stale = net.clone();
    stale.graph.revision += 1;
    assert!(matches!(a.verify_net(&stale, proof), Err(Error::Net(_))));
    assert!(matches!(
        numeric_with(NUMERIC, compiled::Target::Metal, Limits::default()),
        Err(Error::Compiler(compiled::Error::NumericTarget))
    ));
    assert!(matches!(
        provenance::compile(NUMERIC, "calc", compiled::Target::Cpu, Limits::default()),
        Err(Error::Compiler(compiled::Error::UnsupportedType))
    ));
    assert!(matches!(
        provenance::compile_owned(
            NUMERIC,
            "calc",
            compiled::Target::Cpu,
            Limits::default(),
            storage()
        ),
        Err(Error::Compiler(compiled::Error::UnsupportedType))
    ));
    assert!(
        numeric_with(
            b"struct Numeric{x:f32} fn calc() -> i32 { return 7; }",
            compiled::Target::Cpu,
            Limits::default()
        )
        .is_err()
    );
    assert!(matches!(
        numeric_with(
            b"fn calc() -> f32 { return 1u32; }",
            compiled::Target::Cpu,
            Limits::default()
        ),
        Err(Error::Source(_))
    ));
}

#[test]
fn issue_gnet_05_provenance_numeric_oracle_preserves_exact_nan_zero_and_runtime_budgets() {
    let echo = numeric(b"fn calc(a: f32) -> f32 { return a; }");
    let mut ids = std::collections::BTreeSet::new();
    for bits in [
        0,
        0x80000000,
        1,
        0x007fffff,
        0x00800000,
        0x3f800000,
        0x7f800000,
        0xff800000,
        0x7fc12345,
        0x7f812345,
        0xffc54321,
        u32::MAX,
    ] {
        let args = numeric_args(&[bits]);
        let e = echo.observe_oracle(&args, numeric_limits()).unwrap();
        assert!(ids.insert(e.id().to_owned()));
        assert_eq!(
            e.observation().semantic.result,
            Ok(eval::Value::F32(F32Bits::from_bits(bits)))
        );
        assert_eq!(e.manifest()["input_words"], json!([bits]));
        assert_eq!(e.manifest()["result_bits"], bits);
        assert_eq!(e.manifest()["backend"], "cpu_net_oracle");
        assert_eq!(e.manifest()["continuation_execution"], false);
        let Ok(eval::Value::F32(snapshot)) = e.observation().semantic.result else {
            panic!("expected an actual f32 snapshot");
        };
        assert_eq!(snapshot.to_bits(), bits);
        let mut observed = e.observation();
        observed.semantic.result = Ok(eval::Value::F32(F32Bits::POS_ZERO));
        assert_eq!(
            e.observation().semantic.result,
            Ok(eval::Value::F32(F32Bits::from_bits(bits)))
        );
        for forged in forged_leaves(e.manifest()) {
            assert_eq!(e.verify_manifest(&forged), Err(Error::ManifestMismatch));
        }
    }
    let args = numeric_args(&[0x80000000]);
    let full = echo.observe_oracle(&args, numeric_limits()).unwrap();
    for field in 0..4 {
        let mut l = numeric_limits();
        match field {
            0 => l.fuel += 1,
            1 => l.transitions += 1,
            2 => l.trace_events += 1,
            3 => l.array_elements += 1,
            _ => unreachable!(),
        }
        let changed = echo.observe_oracle(&args, l).unwrap();
        assert_ne!(full.id(), changed.id());
        assert_eq!(
            changed.observation().semantic.result,
            full.observation().semantic.result
        );
    }
    for fuel in 1..=full.observation().semantic.steps_used + 1 {
        let l = oracle::Limits {
            fuel,
            ..numeric_limits()
        };
        let e = echo.observe_oracle(&args, l).unwrap();
        assert_eq!(e.manifest()["limits"]["fuel"], fuel);
        if fuel < full.observation().semantic.steps_used {
            assert!(matches!(
                e.observation().semantic.result,
                Err(eval::EvalError::EvaluationLimit { .. })
            ));
        }
    }
    assert!(
        echo.observe_oracle(&[eval::Value::U32(0)], numeric_limits())
            .is_err()
    );
}
