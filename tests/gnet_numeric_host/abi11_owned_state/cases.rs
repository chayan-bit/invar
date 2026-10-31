use super::*;

#[test]
fn every_fuel_prefix_matches_oracle_across_quanta() {
    let names = [
        "literals.inv",
        "filled_empty.inv",
        "update_fma.inv",
        "loans_moves.inv",
        "replacement.inv",
        "mixed_relocation.inv",
        "root_return.inv",
        "element_copy.inv",
        "nested_loans.inv",
        "index_trap.inv",
        "store_trap.inv",
        "older_integer_trap.inv",
        "division_trap.inv",
        "array_size_trap.inv",
        "call_depth_trap.inv",
    ];
    let mut comparisons = 0;
    for name in names {
        let n = net(&source(name));
        let proof = n.verify().unwrap();
        let elements = if name == "mixed_relocation.inv" {
            3
        } else {
            128
        };
        let pair = Pair::new(
            &n,
            "main",
            compiled::OwnedLimits {
                elements,
                ..storage()
            },
        );
        let l = |fuel| oracle::Limits {
            fuel,
            array_elements: elements,
            ..limits()
        };
        let complete = oracle::observe(&n, &proof, l(1000)).unwrap();
        if name.contains("trap") {
            assert!(complete.semantic.result.is_err(), "{name}")
        } else {
            assert_eq!(complete.semantic.result, Ok(eval::Value::I32(7)), "{name}")
        }
        let mut rows = Vec::new();
        let mut expected = Vec::new();
        for fuel in 1..=complete.semantic.steps_used + 1 {
            rows.push((l(fuel), vec![]));
            expected.push(oracle::observe(&n, &proof, l(fuel)).unwrap());
        }
        for (w, o) in pair.run(&rows).iter().zip(&expected) {
            compare(&pair.layout, o, w);
            comparisons += QUANTA.len();
        }
    }
    eprintln!(
        "ABI11: {comparisons} quantum/fuel observations equal across quanta and to the oracle"
    );
}

#[test]
fn scalar_arguments_bits_and_owner_roots() {
    let n = net(&source("runtime.inv"));
    let proof = n.verify().unwrap();
    let values = [
        0, 0x80000000, 1, 0x007fffff, 0x3f800000, 0x7f7fffff, 0x7f800000, 0xff800000, 0x7fa00001,
        0xffc12345,
    ];
    for entry in ["run", "make"] {
        let pair = Pair::new(&n, entry, storage());
        let mut rows = Vec::new();
        let mut expected = Vec::new();
        for bits in values {
            for length in [0, 1, 3] {
                if entry == "run" && length == 0 {
                    continue;
                }
                let args = [
                    eval::Value::U32(length),
                    eval::Value::F32(F32Bits::from_bits(bits)),
                ];
                rows.push((limits(), carriers(&args)));
                expected.push(oracle::observe_call(&n, &proof, entry, &args, limits()).unwrap());
            }
        }
        for ((w, o), (_, x)) in pair.run(&rows).iter().zip(&expected).zip(&rows) {
            compare(&pair.layout, o, w);
            if entry == "run" {
                // Signaling NaN payloads and signed zero copy exactly.
                assert_eq!(w[1], x[1])
            }
        }
    }
    // Invalid external carriers are refused before any owner is created.
    let pair = Pair::new(&n, "make", storage());
    for bad in [vec![-1, 0], vec![0, 1 << 32], vec![1 << 32, -1]] {
        for w in pair.run(&[(limits(), bad)]) {
            assert_eq!(&w[..10], &[8, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        }
    }
}

#[test]
fn storage_trace_and_transition_budgets_across_quanta() {
    let n = net(&source("mixed_relocation.inv"));
    let proof = n.verify().unwrap();
    let pair = Pair::new(&n, "main", storage());
    let mut rows = Vec::new();
    let mut expected = Vec::new();
    for cap in [0, 1, 2, 3, 7, 17, 31, 34] {
        for l in [
            oracle::Limits {
                trace_events: cap as usize,
                ..limits()
            },
            oracle::Limits {
                transitions: cap,
                ..limits()
            },
        ] {
            rows.push((l, vec![]));
            expected.push(oracle::observe(&n, &proof, l).unwrap());
        }
    }
    for (w, o) in pair.run(&rows).iter().zip(&expected) {
        compare(&pair.layout, o, w)
    }
    let heap_limited = oracle::Limits {
        array_elements: 2,
        ..limits()
    };
    let o = oracle::observe(&n, &proof, heap_limited).unwrap();
    assert_eq!(o.failure, Some(oracle::ToolFailure::Allocation));
    let pair = Pair::new(
        &n,
        "main",
        compiled::OwnedLimits {
            elements: 2,
            ..storage()
        },
    );
    compare(&pair.layout, &o, &pair.run(&[(heap_limited, vec![])])[0]);
    // Private backend storage failures stop identically at every quantum.
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
        let pair = Pair::new(&n, "main", s);
        assert_eq!(pair.run(&[(limits(), vec![])])[0][0], code);
    }
}

#[test]
fn fixed_budget_mutation_and_invalid_quantum_finish_with_status8() {
    let n = net(&source("mixed_relocation.inv"));
    let pair = Pair::new(&n, "main", storage());
    let full = pair.run(&[(limits(), vec![])]).remove(0);
    for q in [1u64, 3, 17] {
        let rows: Vec<_> = ["fuel", "transitions", "trace", "zero", "huge"]
            .into_iter()
            .map(|m| (m, limits(), q, vec![]))
            .collect();
        for w in pair.abi11.batch(&rows) {
            let w = w.expect("first quantum must yield");
            assert_eq!(w[0], 2);
            let w = &w[1..];
            // Terminal status 8 keeps the committed prefix of the fixed run.
            assert_eq!((w[0], w[1], w[5]), (8, 0, q as i64));
            let events = w[7] as usize * 7;
            assert_eq!(&w[10..10 + events], &full[10..10 + events]);
            assert!(w[4] <= full[4] && w[8] <= full[8] && w[9] <= full[9]);
        }
    }
    // Invalid first-call arguments finish an unstarted task with zero counters.
    for q in [0u64, 1_000_001] {
        let w = pair
            .abi11
            .batch(&[("run", limits(), q, vec![])])
            .remove(0)
            .unwrap();
        assert_eq!(&w[..11], &[1, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }
    for fuel in [0u64, 1_000_001] {
        let l = oracle::Limits { fuel, ..limits() };
        let w = pair
            .abi11
            .batch(&[("run", l, 1, vec![])])
            .remove(0)
            .unwrap();
        assert_eq!(w[1], 8);
    }
}

#[test]
fn identity_admission_and_legacy_profiles_unchanged() {
    let n = net(&source("literals.inv"));
    let proof = n.verify().unwrap();
    let cpu = compiled::Target::Cpu;
    let l = compiled::Limits::default();
    let abi = |r: Result<compiled::Artifact, compiled::Error>| r.unwrap().abi_version();
    assert_eq!(
        abi(compiled::emit_resumable_owned_numeric(
            &n,
            &proof,
            "main",
            cpu,
            l,
            storage()
        )),
        11
    );
    // Metal stays refused. Direct CUDA emission is the distinct ABI12 profile
    // (a compile receipt only since stage 2b; never loaded or executed, see the
    // cuda_owned_numeric_model tests).
    assert_eq!(
        compiled::emit_resumable_owned_numeric(
            &n,
            &proof,
            "main",
            compiled::Target::Metal,
            l,
            storage()
        )
        .unwrap_err(),
        compiled::Error::NumericTarget
    );
    assert_eq!(
        compiled::emit_resumable_owned_numeric(
            &n,
            &proof,
            "main",
            compiled::Target::Cuda,
            l,
            storage()
        )
        .unwrap()
        .abi_version(),
        12
    );
    // No other profile is widened to numeric owners.
    for r in [
        compiled::emit_resumable(&n, &proof, "main", cpu, l, storage()),
        compiled::emit_resumable_aggregates(&n, &proof, "main", cpu, l, storage()),
    ] {
        assert_eq!(r.unwrap_err(), compiled::Error::UnsupportedType);
    }
    for (s, entry) in [
        ("fn entry(xs:read Array<f32>)->i32{return 7;}", "entry"),
        ("fn entry(xs:take Array<i32>)->i32{return 7;}", "entry"),
        ("struct Numeric{x:f32} fn main()->i32{return 7;}", "main"),
        ("struct Old{x:i32} fn main()->i32{return 7;}", "main"),
    ] {
        let n = net(s);
        let proof = n.verify().unwrap();
        assert_eq!(
            compiled::emit_resumable_owned_numeric(&n, &proof, entry, cpu, l, storage())
                .unwrap_err(),
            compiled::Error::UnsupportedType,
            "{s}"
        );
    }
    // Legacy scalar profiles keep their numbers and emit no ABI11 layout check.
    let scalar = net("fn run(x:i32)->i32{return x;} fn main()->i32{return 7;}");
    let p = scalar.verify().unwrap();
    for (r, version) in [
        (compiled::emit(&scalar, &p, "run", cpu, l), 1),
        (
            compiled::emit_owned(&scalar, &p, "run", cpu, l, storage()),
            2,
        ),
        (
            compiled::emit_resumable(&scalar, &p, "run", cpu, l, storage()),
            3,
        ),
        (
            compiled::emit_aggregates(&scalar, &p, "run", cpu, l, storage()),
            6,
        ),
        (
            compiled::emit_resumable_aggregates(&scalar, &p, "run", cpu, l, storage()),
            7,
        ),
        (
            compiled::emit_resumable_owned_numeric(&scalar, &p, "run", cpu, l, storage()),
            11,
        ),
    ] {
        let a = r.unwrap();
        assert_eq!(a.abi_version(), version);
        assert_eq!(
            a.source().contains("GNET ABI11 layout mismatch"),
            version == 11
        );
    }
    // Storage limits are validated before emission.
    assert_eq!(
        compiled::emit_resumable_owned_numeric(
            &n,
            &proof,
            "main",
            cpu,
            l,
            compiled::OwnedLimits {
                allocations: 0,
                ..storage()
            }
        )
        .unwrap_err(),
        compiled::Error::Limit
    );
}

#[test]
fn cpu_worklist_independent_roots_slices_and_lane_isolation() {
    let n = net(&source("runtime.inv"));
    let proof = n.verify().unwrap();
    let eleven = compiled::emit_resumable_owned_numeric(
        &n,
        &proof,
        "run",
        compiled::Target::Cpu,
        compiled::Limits::default(),
        storage(),
    )
    .unwrap();
    let worklist = Cpu::build_driver(&eleven, WORKLIST, "");
    let pair = Pair::new(&n, "run", storage());
    // Success, signed zero, signaling NaN, an index trap (empty owner), a
    // private heap limit and infinity occupy adjacent independent lanes.
    let inputs: Vec<Vec<i64>> = vec![
        vec![3, 0x80000000],
        vec![0, 0x3f800000],
        vec![1, 0x7fa00001],
        vec![70, 0],
        vec![7, 0xff800000],
    ];
    let mut statuses = Vec::new();
    let mut observations = 0;
    for l in [
        limits(),
        oracle::Limits {
            trace_events: 5,
            ..limits()
        },
        oracle::Limits {
            transitions: 40,
            ..limits()
        },
    ] {
        let rows: Vec<_> = inputs.iter().map(|x| (l, x.clone())).collect();
        // Separate single-root runs (identical at every quantum) are the
        // reference for each lane.
        let single = pair.run(&rows);
        for q in [1u64, 3, 17, 1_000_000] {
            let resumed: Vec<_> = inputs.iter().map(|x| ("run", l, q, x.clone())).collect();
            let quanta: i64 = pair
                .abi11
                .batch(&resumed)
                .into_iter()
                .map(|w| w.unwrap()[0])
                .sum();
            for slices in [1u64, 2, 5, 1_000_000] {
                let w = worklist.worklist(l, inputs.len() as u64, q, slices, &inputs);
                assert_eq!(w.lanes, single, "quantum {q} slices {slices}");
                let (last, sliced) = w.calls.split_last().unwrap();
                assert_eq!(last[0], 0);
                assert_eq!(last[2], inputs.len() as i64);
                for c in sliced {
                    assert_eq!((c[0], c[1]), (14, slices as i64));
                }
                // Every submission is one exclusive root quantum.
                let submitted: i64 = w.calls.iter().map(|c| c[1]).sum();
                assert_eq!(submitted, quanta, "quantum {q} slices {slices}");
                observations += 1;
            }
        }
        statuses.extend(single.iter().map(|w| w[0]));
    }
    // Lanes reached success, index trap, heap limit, trace and transition limits.
    for code in [0, 9, 11, 6, 5] {
        assert!(statuses.contains(&code), "status {code}: {statuses:?}");
    }
    // Zero roots complete immediately; zero slices with pending roots yield 14
    // without submitting work, and a later call resumes them.
    let empty = worklist.worklist(limits(), 0, 1, 1, &[]);
    assert_eq!((empty.calls, empty.lanes.len()), (vec![[0, 0, 0]], 0));
    let stalled = worklist.worklist(limits(), 2, 1, 0, &inputs[..2]);
    assert_eq!(stalled.calls, vec![[14, 0, 0]]);
    // Invalid worklist arguments return 8 before touching any task or buffer.
    for (roots, fuel, quantum, slices) in [
        (1025, 1000, 1, 1),
        (2, 1000, 0, 1),
        (2, 1000, 1_000_001, 1),
        (2, 1000, 1, 1_000_001),
        (2, 0, 1, 1),
        (2, 1_000_001, 1, 1),
    ] {
        let lanes = usize::try_from(roots).unwrap().min(64);
        let filler = vec![vec![1, 0]; lanes];
        let l = oracle::Limits { fuel, ..limits() };
        let w = worklist.worklist(l, roots, quantum, slices, &filler);
        assert_eq!(w.calls, vec![[8, 0, 0]]);
        assert!(w.lanes.is_empty());
    }
    eprintln!("ABI11 worklist: {observations} multi-root executions equal to single roots");
}
