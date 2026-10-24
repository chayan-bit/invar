//! The full ABI11 denominator: every row is checked against its expectations;
//! only the sampled sources (`corpus_sample`) also compile and run the CPU backend.
use super::*;
use invar::general_net::numeric_owned_resume_host::{Error, Plan, Progress};

/// Rows whose observation fits the scalar output header run through the raw
/// C++ driver (which also covers its host fenv modes).
fn header_only(o: &oracle::Observation) -> bool {
    o.semantic.cleanup_trace.is_empty()
        && match &o.semantic.result {
            Ok(v) => matches!(
                v,
                eval::Value::F32(_)
                    | eval::Value::I32(_)
                    | eval::Value::U32(_)
                    | eval::Value::Bool(_)
                    | eval::Value::Unit
            ),
            Err(eval::EvalError::Trap { category, .. }) => matches!(
                category,
                eval::TrapCategory::IntegerOverflow | eval::TrapCategory::DivisionByZero
            ),
            Err(eval::EvalError::EvaluationLimit { .. }) => true,
            Err(eval::EvalError::Tool {
                code: "CALL_DEPTH_LIMIT",
                ..
            }) => true,
            Err(_) => matches!(
                o.failure,
                Some(oracle::ToolFailure::TransitionLimit | oracle::ToolFailure::TraceLimit)
            ),
        }
}

/// Owner/array rows run through the loaded ABI11 host, which decodes heap
/// results and cleanup events; its fixed host-buffer budget may refuse a row.
fn owned_host(
    bytes: &[u8],
    entry: &str,
    xs: &[eval::Value],
    l: oracle::Limits,
    o: &oracle::Observation,
) -> Json {
    let storage = compiled::OwnedLimits {
        elements: 128,
        allocations: 128,
        cleanup_events: 128,
    };
    let mut plan = Plan::from_source(
        bytes,
        entry,
        Default::default(),
        storage,
        std::path::Path::new("c++"),
    )
    .unwrap();
    let mut run = match plan.start(xs, l) {
        Err(Error::Limit) => return json!({"status":"host_budget_refused"}),
        started => started.unwrap(),
    };
    // Every row stays within 1e6 transitions, so two maximal quanta suffice.
    let got = (0..2)
        .find_map(|_| match run.resume(1_000_000).unwrap() {
            Progress::Complete(got) => Some(got),
            Progress::Pending(_) => None,
        })
        .expect("owned row finishes within two maximal quanta");
    equal(&got.semantic, &o.semantic);
    assert_eq!(got.failure, o.failure);
    assert_eq!(got.events, o.events);
    assert_eq!(
        (got.transitions, got.max_frames),
        (o.transitions, o.max_frames)
    );
    json!({"status":"executed","host":"numeric_owned_resume_host","events":got.events.len(),"cleanup":base::cleanup(&got.semantic),"gpu_execution":"not_performed"})
}

pub(super) fn run() {
    let ordinary = base::document("expanded-manifest.json");
    let runtime = base::document("runtime-vectors.json");
    let children = base::document("child-vectors.json");
    let numeric = base::document("numeric-vectors.json");
    let mut cache = BTreeMap::<(String, String), Cpu>::new();
    let mut rows = Vec::new();
    let mut comparisons = 0;
    let mut source_oracle = 0;
    let mut source_native = 0;
    let mut backend_failure_rows = 0;
    let mut budget_refused = Vec::<String>::new();
    let mut rules = BTreeSet::new();
    let documents = [
        ("ordinary", &ordinary),
        ("runtime", &runtime),
        ("child", &children),
        ("numeric", &numeric),
    ];
    let keep = corpus_sample::paths(documents.iter().flat_map(|(kind, cases)| {
        cases["cases"].as_array().unwrap().iter().map(move |case| {
            let path = case["source"].as_str().unwrap();
            (
                path.to_owned(),
                format!("{kind}:{}", corpus_sample::category(path)),
            )
        })
    }));
    for (kind, cases) in documents {
        for case in cases["cases"].as_array().unwrap() {
            let id = case["id"].as_str().unwrap();
            let path = case["source"].as_str().unwrap();
            let bytes = fs::read(base::root().join(path)).unwrap();
            let hash = case["source_sha256"]
                .as_str()
                .or_else(|| {
                    ordinary["cases"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|r| r["source"] == path)
                        .and_then(|r| r["source_sha256"].as_str())
                })
                .unwrap();
            assert_eq!(base::digest(&bytes), hash);
            let c = match checked(&bytes) {
                Ok(c) => c,
                Err(e) => {
                    if kind == "numeric" {
                        assert_eq!(case["expected_check"], e.code)
                    } else {
                        assert_eq!(case["expected_check"]["diagnostic_codes"][0], e.code)
                    }
                    let r = json!({"status":"source_rejected","reason":e.code});
                    rows.push(json!({"id":id,"kind":kind,"source":r,"oracle":r,"abi11_cpu":r,"source_path":path,"source_sha256":hash}));
                    continue;
                }
            };
            let entry = if kind == "ordinary" {
                "main"
            } else {
                case["entry"].as_str().unwrap()
            };
            let xs = if kind == "ordinary" {
                vec![]
            } else {
                args(&case["arguments"])
            };
            let l = oracle::Limits {
                fuel: case["fuel"].as_u64().unwrap(),
                transitions: case["limits"]["transitions"].as_u64().unwrap_or(100000),
                trace_events: case["limits"]["trace_events"].as_u64().unwrap_or(100000) as usize,
                array_elements: case["limits"]["array_elements"].as_u64().unwrap_or(128) as usize,
            };
            let s = observe(&c, entry, &xs, l.fuel);
            expectation(
                &s,
                if kind == "ordinary" {
                    &case["expected_reference_run"]
                } else {
                    &case["expected_source"]
                },
            );
            let net = Net::lower(&c, general_net::Limits::default()).unwrap();
            rules.extend(
                net.agents
                    .iter()
                    .map(|a| format!("{:?}", a.rule))
                    .chain(net.control.iter().map(|a| format!("{:?}", a.rule))),
            );
            let o = oracle::observe_call(&net, &net.verify().unwrap(), entry, &xs, l).unwrap();
            match o.failure {
                None => {
                    equal(&s, &o.semantic);
                    source_oracle += 1;
                }
                Some(oracle::ToolFailure::Unsupported) => {}
                Some(failure) => {
                    backend_failure_rows += 1;
                    assert_eq!(case["expected_oracle_failure"], format!("{failure:?}"))
                }
            }
            let cpu = match emit(&net, entry) {
                Err(e) => {
                    assert_eq!(e, compiled::Error::UnsupportedType);
                    json!({"status":"unsupported","reason":format!("{e:?}")})
                }
                Ok(_) if !keep.contains(path) => json!({"status":"sampled_out"}),
                Ok(_) if !header_only(&o) => {
                    let receipt = owned_host(&bytes, entry, &xs, l, &o);
                    if receipt["status"] == "executed" {
                        source_native += usize::from(o.failure.is_none());
                        comparisons += 1;
                    } else {
                        budget_refused.push(id.to_owned());
                    }
                    receipt
                }
                Ok(a) => {
                    let h = cache
                        .entry((path.to_owned(), entry.to_owned()))
                        .or_insert_with(|| Cpu::new(a));
                    let w = h.batch(&[(l, carriers(&xs))], 0, false).pop().unwrap();
                    let receipt = h.compare(&o, &w);
                    source_native += usize::from(o.failure.is_none());
                    if case["exact_nan_bits"] == true {
                        let bits = case["expected_source"]["result"]["bits"].as_u64().unwrap();
                        for result in [&s.result, &o.semantic.result] {
                            let Ok(eval::Value::F32(x)) = result else {
                                panic!("{result:?}")
                            };
                            assert_eq!(u64::from(x.to_bits()), bits);
                        }
                        assert_eq!(w[1] as u64, bits);
                    };
                    comparisons += 1;
                    receipt
                }
            };
            rows.push(json!({"id":id,"kind":kind,"source_path":path,"source_sha256":hash,"entry":entry,"arguments":case["arguments"],"source":recorded(&s),"oracle":oracle_record(&o),"abi11_cpu":cpu}));
        }
    }
    assert_eq!(rows.len(), 254);
    assert_eq!(rules.len(), 32);
    assert!(rules.contains("F32"));
    let counts = |key: &str| {
        let mut c = BTreeMap::<String, usize>::new();
        for r in &rows {
            *c.entry(r[key]["status"].as_str().unwrap().to_owned())
                .or_default() += 1
        }
        c
    };
    assert_eq!(
        json!(counts("source")),
        json!({"executed":250,"source_rejected":4})
    );
    assert_eq!(
        // GNET-02 closed the nominal f32 member oracle gap (formerly 6 unsupported).
        json!(counts("oracle")),
        json!({"executed":250,"source_rejected":4})
    );
    let mut cpu = counts("abi11_cpu");
    // Sampled-out rows were emitted but not compiled; INVAR_FULL_CORPUS=1 runs them all.
    let sampled_out = cpu.remove("sampled_out").unwrap_or(0);
    let refused = cpu.remove("host_budget_refused").unwrap_or(0);
    *cpu.entry("executed".to_owned()).or_default() += sampled_out + refused;
    assert_eq!(
        json!(cpu),
        json!({"executed":233,"unsupported":17,"source_rejected":4})
    );
    assert!(comparisons > 0);
    assert_eq!(comparisons + sampled_out + refused, 233);
    // Only rows asking for 1e6 trace events, or fewer live elements than the
    // artifact's fixed storage, exceed the ABI11 host's buffer budget.
    let known = [
        "array_runtime_0",
        "array_runtime_1",
        "array_runtime_5",
        "array_private_arena_limit",
    ];
    assert!(
        budget_refused.iter().all(|id| known.contains(&id.as_str())),
        "{budget_refused:?}"
    );
    assert_eq!(backend_failure_rows, 3);
    eprintln!(
        "ABI11:254 complete rows (158 retained+96 independently expected),{comparisons} native/reference comparisons;32 structural rules represented"
    );
    report(
        "numeric-full",
        json!({"profile":profile(),"case_count":rows.len(),"counts":{"source":counts("source"),"oracle":counts("oracle"),"abi11_cpu":counts("abi11_cpu")},"host_budget_refused":budget_refused,"actual_native_reference_comparisons":comparisons,"actual_source_observations":250,"actual_source_oracle_comparisons":source_oracle,"actual_source_native_comparisons":source_native,"source_backend_failure_rows":backend_failure_rows,"previously_unavailable_source_observations_closed":85,"retained_rows":158,"added_expected_rows":96,"structural_rules":rules,"rows":rows,"gpu_execution":"not_performed","numeric_policy":"strict-f32-v1"}),
    );
}
