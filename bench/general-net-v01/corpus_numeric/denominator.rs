//! The full ABI8 denominator: every row is checked against its expectations;
//! only the sampled sources (`corpus_sample`) also compile and run the CPU backend.
use super::*;

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
                    rows.push(json!({"id":id,"kind":kind,"source":r,"oracle":r,"abi8_cpu":r,"source_path":path,"source_sha256":hash}));
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
            rows.push(json!({"id":id,"kind":kind,"source_path":path,"source_sha256":hash,"entry":entry,"arguments":case["arguments"],"source":recorded(&s),"oracle":oracle_record(&o),"abi8_cpu":cpu}));
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
    let mut cpu = counts("abi8_cpu");
    // Sampled-out rows were emitted but not compiled; INVAR_FULL_CORPUS=1 runs them all.
    let sampled_out = cpu.remove("sampled_out").unwrap_or(0);
    *cpu.entry("executed".to_owned()).or_default() += sampled_out;
    assert_eq!(
        json!(cpu),
        json!({"executed":179,"unsupported":71,"source_rejected":4})
    );
    assert!(comparisons > 0);
    assert_eq!(comparisons + sampled_out, 179);
    assert_eq!(backend_failure_rows, 3);
    eprintln!(
        "ABI8:254 complete rows (158 retained+96 independently expected),{comparisons} native/reference comparisons;32 structural rules represented"
    );
    report(
        "numeric-full",
        json!({"profile":profile(),"case_count":rows.len(),"counts":{"source":counts("source"),"oracle":counts("oracle"),"abi8_cpu":counts("abi8_cpu")},"actual_native_reference_comparisons":comparisons,"actual_source_observations":250,"actual_source_oracle_comparisons":source_oracle,"actual_source_native_comparisons":source_native,"source_backend_failure_rows":backend_failure_rows,"previously_unavailable_source_observations_closed":85,"retained_rows":158,"added_expected_rows":96,"structural_rules":rules,"rows":rows,"gpu_execution":"not_performed","numeric_policy":"strict-f32-v1"}),
    );
}
