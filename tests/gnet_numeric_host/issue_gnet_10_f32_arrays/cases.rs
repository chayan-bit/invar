use super::*;

#[test]
fn issue_gnet_10_source_mir_oracle_native_every_fuel_prefix() {
    let names = [
        "literals.inv",
        "filled_empty.inv",
        "update_fma.inv",
        "loans_moves.inv",
        "replacement.inv",
        "mixed_relocation.inv",
        "root_return.inv",
        "element_copy.inv",
        "index_trap.inv",
        "store_trap.inv",
        "older_integer_trap.inv",
        "array_size_trap.inv",
    ];
    let mut report_rows = Vec::new();
    let mut comparisons = 0;
    for name in names {
        let s = source(name);
        let c = checked(&s);
        let m = mir::lower(&c).unwrap();
        let n = net(&s);
        let done = eval::observe(&c, 1000);
        if name.contains("trap") {
            assert!(done.result.is_err())
        } else {
            assert_eq!(done.result, Ok(eval::Value::I32(7)))
        };
        let elements = if name == "mixed_relocation.inv" {
            3
        } else {
            128
        };
        let h = Cpu::new(
            &n,
            "main",
            compiled::OwnedLimits {
                elements,
                ..storage()
            },
        );
        let mut batch = Vec::new();
        let mut expected = Vec::new();
        for fuel in 1..=done.steps_used + 1 {
            let s = eval::observe(&c, fuel);
            equal(&s, &mir_eval::observe(&m, fuel));
            let l = oracle::Limits {
                fuel,
                array_elements: elements,
                ..limits()
            };
            let o = oracle::observe(&n, &n.verify().unwrap(), l).unwrap();
            assert_eq!(o.failure, None);
            equal(&s, &o.semantic);
            batch.push((l, vec![]));
            expected.push(o)
        }
        let mut observations = Vec::new();
        for (w, o) in h.batch(&batch, "normal").iter().zip(&expected) {
            observations.push(h.compare(&n, o, w));
            comparisons += 1
        }
        report_rows.push(json!({"source":name,"source_sha256":digest(s.as_bytes()),"source_complete":result(&done),"positive_fuel_prefixes":observations}));
    }
    eprintln!(
        "ABI10 source/MIR/oracle/native:{comparisons} exact positive fuel observations across12 source programs"
    );
    report(
        "source-prefixes",
        json!({"comparisons":comparisons,"programs":12,"cases":report_rows}),
    );
}
#[test]
fn issue_gnet_10_runtime_bits_root_returns_and_relocation() {
    let s = source("runtime.inv");
    let c = checked(&s);
    let n = net(&s);
    let values = [
        0, 0x80000000, 1, 0x007fffff, 0x00800000, 0x3f800000, 0x7f7fffff, 0xff7fffff, 0x7f800000,
        0xff800000, 0x7fa00001, 0xffc12345,
    ];
    let mut rows = Vec::new();
    let mut comparisons = 0;
    for entry in ["run", "make"] {
        let h = Cpu::new(&n, entry, storage());
        let mut cases = Vec::new();
        let mut expected = Vec::new();
        let mut inputs = Vec::new();
        for bits in values {
            for length in [1, 3, 7] {
                let args = vec![
                    eval::Value::U32(length),
                    eval::Value::F32(F32Bits::from_bits(bits)),
                ];
                let o =
                    oracle::observe_call(&n, &n.verify().unwrap(), entry, &args, limits()).unwrap();
                assert_eq!(o.failure, None);
                if entry == "run" {
                    let a = eval::named::observe_numeric_named(&c, entry, &args, 1000);
                    equal(&a, &o.semantic);
                    assert!(matches!(a.result,Ok(eval::Value::F32(x)) if x.to_bits()==bits))
                }
                cases.push((limits(), carriers(&args)));
                expected.push(o);
                inputs.push(json!({"length":length,"bits":bits}))
            }
        }
        if entry == "make" {
            let args = vec![eval::Value::U32(0), eval::Value::F32(F32Bits::NEG_ZERO)];
            cases.push((limits(), carriers(&args)));
            expected.push(
                oracle::observe_call(&n, &n.verify().unwrap(), entry, &args, limits()).unwrap(),
            );
            inputs.push(json!({"length":0,"bits":2147483648u32}))
        };
        for ((w, o), input) in h.batch(&cases, "normal").iter().zip(&expected).zip(&inputs) {
            let receipt = h.compare(&n, o, w);
            if entry == "run" {
                assert_eq!(w[1] as u64, input["bits"].as_u64().unwrap())
            } else {
                assert!(
                    receipt["root"]["bits"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .all(|x| x.as_u64() == input["bits"].as_u64())
                )
            };
            rows.push(json!({"entry":entry,"input":input,"source_named_reference":if entry=="run"{"actual numeric scalar source observation"}else{"array return signature unavailable; ordinary source root fixture qualifies AST/MIR returns"},"observation":receipt}));
            comparisons += 1
        }
    }
    let root_net = net(&source("root_return.inv"));
    let h = Cpu::new(&root_net, "make", storage());
    let o = oracle::observe_call(
        &root_net,
        &root_net.verify().unwrap(),
        "make",
        &[],
        limits(),
    )
    .unwrap();
    assert_eq!(
        result(&o.semantic),
        json!({"ArrayF32":[2147483648u32,2143289344u32,1065353216u32]})
    );
    let w = h.batch(&[(limits(), vec![])], "normal").pop().unwrap();
    let receipt = h.compare(&root_net, &o, &w);
    assert_eq!(
        receipt["root"]["bits"],
        json!([2147483648u32, 2143289344u32, 1065353216u32])
    );
    rows.push(json!({"entry":"make","source":"root_return.inv","source_reference":"source/MIR owner-producing function qualified through ordinary main at every fuel; direct owner root snapshot uses oracle and manual bits","observation":receipt}));
    comparisons += 1;
    eprintln!(
        "ABI10 runtime copy/roots:{comparisons} actual native observations including exact signaling-NaN payloads and empty owner"
    );
    report(
        "runtime-bits",
        json!({"comparisons":comparisons,"rows":rows}),
    );
}
#[test]
fn issue_gnet_10_private_storage_trace_transition_and_cleanup_caps() {
    let s = source("mixed_relocation.inv");
    let n = net(&s);
    let h = Cpu::new(&n, "main", storage());
    let mut rows = Vec::new();
    let mut comparisons = 0;
    for cap in [0, 1, 2, 7, 31] {
        for trace in [false, true] {
            let l = if trace {
                oracle::Limits {
                    trace_events: cap as usize,
                    ..limits()
                }
            } else {
                oracle::Limits {
                    transitions: cap,
                    ..limits()
                }
            };
            let o = oracle::observe(&n, &n.verify().unwrap(), l).unwrap();
            let w = h.batch(&[(l, vec![])], "normal").pop().unwrap();
            rows.push(h.compare(&n, &o, &w));
            comparisons += 1
        }
    }
    let heap = Cpu::new(
        &n,
        "main",
        compiled::OwnedLimits {
            elements: 2,
            ..storage()
        },
    );
    let l = oracle::Limits {
        array_elements: 2,
        ..limits()
    };
    let o = oracle::observe(&n, &n.verify().unwrap(), l).unwrap();
    assert_eq!(o.failure, Some(oracle::ToolFailure::Allocation));
    let w = heap.batch(&[(l, vec![])], "normal").pop().unwrap();
    rows.push(heap.compare(&n, &o, &w));
    comparisons += 1;
    let complete = oracle::observe(&n, &n.verify().unwrap(), limits()).unwrap();
    let p = n.compiled_mir(&n.verify().unwrap()).unwrap();
    let mut failures = Vec::new();
    for (s, code) in [
        (
            compiled::OwnedLimits {
                allocations: 2,
                ..storage()
            },
            12,
        ),
        (
            compiled::OwnedLimits {
                cleanup_events: 0,
                ..storage()
            },
            13,
        ),
        (
            compiled::OwnedLimits {
                cleanup_events: 1,
                ..storage()
            },
            13,
        ),
    ] {
        let h = Cpu::new(&n, "main", s);
        let w = h.batch(&[(limits(), vec![])], "normal").pop().unwrap();
        assert_eq!(w[0], code);
        let event_count = w[7] as usize;
        assert!(event_count < complete.events.len());
        for (actual, e) in w[10..10 + event_count * 7]
            .chunks_exact(7)
            .zip(&complete.events)
        {
            assert_eq!(actual, &gnet_words::event_i64(e))
        }
        let at = 10 + event_count * 7;
        for (q, c) in w[at..at + w[8] as usize * 4]
            .chunks_exact(4)
            .zip(&complete.semantic.cleanup_trace)
        {
            assert_eq!(
                p.functions[q[0] as usize].declaration_order[q[1] as usize],
                c.name
            );
            assert_eq!(q[2], c.length as i64)
        }
        failures.push(json!({"status":code,"header":w[..10],"full_source_and_oracle_status":"success; private backend storage failure is separate","exact_committed_trace_cleanup_prefix":true,"artifact":h.metadata}))
    }
    report(
        "budgets",
        json!({"exact_oracle_comparisons":comparisons,"private_storage_failures":failures,"observations":rows}),
    );
}
#[test]
fn issue_gnet_10_malformed_private_heap_and_bit_writes_fail_closed() {
    let s = source("runtime.inv");
    let n = net(&s);
    let h = Cpu::new(&n, "make", storage());
    let batch = [(limits(), vec![1, 1065353216])];
    let mut rows = Vec::new();
    for mode in [
        "kind",
        "offset",
        "length",
        "flag",
        "bits",
        "allocated",
        "handle",
        "bounds",
        "store",
    ] {
        let w = h.batch(&batch, mode).pop().unwrap();
        assert_eq!(w, [if mode == "bounds" { 9 } else { 7 }, 777]);
        rows.push(json!({"mode":mode,"status":w[0],"heap_and_output_sentinel_unchanged":true,"artifact":h.metadata}))
    }
    report(
        "malformed-heap",
        json!({"actual_native_fail_closed_cases":rows.len(),"rows":rows}),
    );
}
#[test]
fn issue_gnet_10_profile_types_proofs_limits_and_legacy_rejections() {
    let n = net(&source("literals.inv"));
    let proof = n.verify().unwrap();
    let numeric = |n: &Net, proof: &general_net::Proof, entry: &str, target, s| {
        compiled::emit_resumable_owned_numeric(
            n,
            proof,
            entry,
            target,
            compiled::Limits::default(),
            s,
        )
    };
    let a = numeric(&n, &proof, "main", compiled::Target::Cpu, storage()).unwrap();
    assert_eq!(a.abi_version(), 11);
    let mut rejections = 0;
    assert_eq!(
        numeric(&n, &proof, "main", compiled::Target::Metal, storage()).unwrap_err(),
        compiled::Error::NumericTarget
    );
    rejections += 1;
    for result in [
        compiled::emit(
            &n,
            &proof,
            "main",
            compiled::Target::Cpu,
            compiled::Limits::default(),
        ),
        compiled::emit_owned(
            &n,
            &proof,
            "main",
            compiled::Target::Cpu,
            compiled::Limits::default(),
            storage(),
        ),
        compiled::emit_resumable(
            &n,
            &proof,
            "main",
            compiled::Target::Cpu,
            compiled::Limits::default(),
            storage(),
        ),
        compiled::emit_speculative(
            &n,
            &proof,
            "main",
            compiled::Target::Cpu,
            compiled::Limits::default(),
            storage(),
            compiled::SpeculationLimits::default(),
        ),
        compiled::emit_aggregates(
            &n,
            &proof,
            "main",
            compiled::Target::Cpu,
            compiled::Limits::default(),
            storage(),
        ),
    ] {
        assert_eq!(result.unwrap_err(), compiled::Error::UnsupportedType);
        rejections += 1
    }
    for source in [
        "fn entry(xs:read Array<f32>)->i32{return 7;}",
        "struct Numeric{x:f32} fn main()->i32{return 7;}",
        "struct Old{x:i32} fn main()->i32{return 7;}",
    ] {
        let c = checked(source);
        let n = Net::lower(&c, general_net::Limits::default()).unwrap();
        assert_eq!(
            numeric(
                &n,
                &n.verify().unwrap(),
                if source.contains("entry") {
                    "entry"
                } else {
                    "main"
                },
                compiled::Target::Cpu,
                storage()
            )
            .unwrap_err(),
            compiled::Error::UnsupportedType
        );
        rejections += 1
    }
    let other = net("fn main()->i32{return 7;}");
    assert!(matches!(
        numeric(&other, &proof, "main", compiled::Target::Cpu, storage()),
        Err(compiled::Error::Boundary(_))
    ));
    rejections += 1;
    for s in [
        compiled::OwnedLimits {
            allocations: 0,
            ..storage()
        },
        compiled::OwnedLimits {
            elements: invar::array_limits::MAX_LIVE_ELEMENTS + 1,
            ..storage()
        },
        compiled::OwnedLimits {
            allocations: 10001,
            ..storage()
        },
        compiled::OwnedLimits {
            cleanup_events: 100001,
            ..storage()
        },
    ] {
        assert_eq!(
            numeric(&n, &proof, "main", compiled::Target::Cpu, s).unwrap_err(),
            compiled::Error::Limit
        );
        rejections += 1
    }
    let h = Cpu::new(&net(&source("runtime.inv")), "run", storage());
    for xs in [vec![1, -1], vec![1, u32::MAX as i64 + 1]] {
        let w = h.batch(&[(limits(), xs)], "normal").pop().unwrap();
        assert_eq!(w[0], 8);
        assert_eq!(w[4], 0);
        assert_eq!(w[7], 0);
        assert_eq!(w[8], 0);
        assert_eq!(w[9], 0);
        rejections += 1
    }
    report(
        "admission",
        json!({"actual_rejections":rejections,"abi11_metadata":h.metadata}),
    );
}
/// A record that owns an `Array<f32>` and is released by reassignment never reaches emitted
/// code: every compiled entry refuses it, so `gnet_leaf_length` (record array fields, kinds 0
/// and 1) never sees an f32 array (kind 258). The `Array<i32>` control shows the refusal comes
/// from the f32 field, not from the record shape.
#[test]
fn issue_gnet_10_record_owning_f32_array_is_refused_by_every_compiled_entry() {
    let shape = |ty: &str, a: &str, b: &str| {
        format!(
            "struct Signal {{ xs: Array<{ty}>, code: i32 }} fn main() -> i32 {{ var s = Signal {{ xs: [{a}], code: 7 }}; s = Signal {{ xs: [{b}], code: 9 }}; return s.code; }}"
        )
    };
    let control = net(&shape("i32", "1, 2", "3"));
    let proof = control.verify().unwrap();
    compiled::emit_aggregates(
        &control,
        &proof,
        "main",
        compiled::Target::Cpu,
        compiled::Limits::default(),
        storage(),
    )
    .unwrap();
    let n = net(&shape("f32", "1.0, 2.0", "3.0"));
    let proof = n.verify().unwrap();
    let limits = compiled::Limits::default;
    let mut results = vec![
        compiled::emit_aggregates(
            &n,
            &proof,
            "main",
            compiled::Target::Cpu,
            limits(),
            storage(),
        ),
        compiled::emit_resumable_aggregates(
            &n,
            &proof,
            "main",
            compiled::Target::Cpu,
            limits(),
            storage(),
        ),
        compiled::emit_resumable_owned_numeric(
            &n,
            &proof,
            "main",
            compiled::Target::Cpu,
            limits(),
            storage(),
        ),
    ];
    for target in [
        compiled::Target::Cpu,
        compiled::Target::Cuda,
        compiled::Target::Metal,
    ] {
        results.push(compiled::emit_resumable_owned_numeric(
            &n,
            &proof,
            "main",
            target,
            limits(),
            storage(),
        ));
        results.push(compiled::emit_device_speculative(
            &n,
            &proof,
            "main",
            target,
            limits(),
            storage(),
            compiled::SpeculationLimits::default(),
        ));
    }
    // Device targets may refuse the numeric target before any layout is built; either way no
    // code is emitted.
    for (i, result) in results.into_iter().enumerate() {
        let error = result.expect_err("a record owning Array<f32> must not compile");
        // The first four are the CPU aggregate and owned-numeric layouts the leaf helper serves.
        if i < 4 {
            assert_eq!(error, compiled::Error::UnsupportedType, "entry {i}");
        }
        assert!(
            matches!(
                error,
                compiled::Error::UnsupportedType | compiled::Error::NumericTarget
            ),
            "entry {i}: {error:?}"
        );
    }
}
