//! Corpus runner shared by the dedicated integration test. Expectations are
//! authored separately from executions; unsupported backends retain their rows.
use super::{child_deadline, gnet_words};

use general_net::{Net, compiled, oracle};
use invar::{check, eval, front, general_net, mir, mir_eval};
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

const DRIVER: &str = r#"
#include <iostream>
#include <vector>
#include <string>
int main(int argc, char** argv) {
    if (argc<4) return 2;
    gnet_u fuel=std::stoull(argv[1]), transitions=std::stoull(argv[2]), cap=std::stoull(argv[3]);
    if (cap>1000000) return 2;
    std::vector<gnet_w> input;
    for (int i=4;i<argc;++i) input.push_back(std::stoll(argv[i]));
    gnet_w output[8]={}; std::vector<gnet_u> trace(cap*7);
    gnet_run(input.data(),output,trace.data(),fuel,transitions,cap);
    for (auto word:output) std::cout<<word<<' ';
    for (gnet_u i=0;i<(gnet_u)output[7]*7;++i) std::cout<<trace[i]<<' ';
    std::cout<<'\n';
}
"#;

pub(super) fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("bench/general-net-v01")
}
pub(super) fn document(name: &str) -> Json {
    serde_json::from_slice(&fs::read(root().join(name)).unwrap()).unwrap()
}
pub(super) fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
fn structural_coverage(rows: &[Json]) -> Json {
    let covered = rows
        .iter()
        .flat_map(|row| row["structural_rules"].as_array().into_iter().flatten())
        .map(|rule| rule.as_str().unwrap().to_owned())
        .collect::<std::collections::BTreeSet<_>>();
    let inventory = general_net::coverage::ALL_RULES
        .iter()
        .map(|rule| format!("{rule:?}"))
        .collect::<std::collections::BTreeSet<_>>();
    assert!(
        covered.is_subset(&inventory),
        "corpus rules must belong to the current MIR inventory"
    );
    let required = inventory
        .iter()
        .filter(|name| !matches!(name.as_str(), "F32" | "F64"))
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        covered, required,
        "every integer/owner corpus rule must remain represented; F32 and F64 stay outside this source scope"
    );
    let missing = inventory.difference(&covered).cloned().collect::<Vec<_>>();
    json!({"scope":"integer_owner_source_corpus","covered_rule_count":covered.len(),"inventory_rule_count":inventory.len(),"covered_rules":covered,"not_represented_rules":missing,"rules":general_net::coverage::ALL_RULES.iter().map(|rule| {
        let name=format!("{rule:?}");
        json!({"rule":name,"represented":covered.contains(&name),"oracle_supported":oracle::supports(*rule),"corpus_exclusion_reason":if name=="F32" || name=="F64" {Some("outside the integer/owner source corpus; no numeric or device coverage claimed")}else{None}})
    }).collect::<Vec<_>>()})
}
fn report(name: &str, rows: Vec<Json>) {
    let Some(dir) = std::env::var_os("INVAR_GNET_CORPUS_REPORT_DIR") else {
        return;
    };
    let dir = PathBuf::from(dir);
    fs::create_dir_all(&dir).unwrap();
    let counts = |key: &str| {
        let mut counts = BTreeMap::<String, usize>::new();
        for row in &rows {
            *counts
                .entry(row[key]["status"].as_str().unwrap().to_owned())
                .or_default() += 1;
        }
        counts
    };
    let output = json!({"schema_version":1,"profile":if cfg!(debug_assertions){"debug"}else{"release"},"case_count":rows.len(),"counts":{"source":counts("source"),"oracle":counts("oracle"),"compiled_cpu":counts("compiled_cpu")},"gpu_execution":"not_performed","parsed_function_occurrences":rows.iter().filter_map(|r|r["function_count"].as_u64()).sum::<u64>(),"structural_rule_denominator":general_net::coverage::ALL_RULES.len(),"structural_coverage":if name=="ordinary" {structural_coverage(&rows)}else{Json::Null},"oracle_supported_rule_count":general_net::coverage::ALL_RULES.iter().filter(|r|oracle::supports(**r)).count(),"rows":rows});
    fs::write(
        dir.join(format!("{name}.json")),
        serde_json::to_vec_pretty(&output).unwrap(),
    )
    .unwrap();
}
pub(super) fn semantic(observation: &eval::Observation) -> Json {
    match &observation.result {
        Ok(eval::Value::I32(n)) => json!({"status":"ok","result":{"type":"i32","value":n}}),
        Ok(eval::Value::U32(n)) => json!({"status":"ok","result":{"type":"u32","value":n}}),
        Ok(eval::Value::ArrayI32(n)) => {
            json!({"status":"ok","result":{"type":"Array<i32>","value":&*n.borrow()}})
        }
        Ok(eval::Value::ArrayU32(n)) => {
            json!({"status":"ok","result":{"type":"Array<u32>","value":&*n.borrow()}})
        }
        Err(error @ eval::EvalError::Trap { .. }) => {
            json!({"status":"trap","trap":error.to_json("corpus")["trap"]})
        }
        Err(eval::EvalError::EvaluationLimit {
            fuel_limit,
            steps_used,
        }) => json!({"status":"evaluation_limit","fuel_limit":fuel_limit,"steps_used":steps_used}),
        Err(eval::EvalError::Tool { code, .. }) => json!({"status":"tool_error","code":code}),
        other => panic!("corpus requires scalar results: {other:?}"),
    }
}
pub(super) fn cleanup(observation: &eval::Observation) -> Json {
    json!(
        observation
            .cleanup_trace
            .iter()
            .map(|e| json!({"name":e.name,"length":e.length}))
            .collect::<Vec<_>>()
    )
}
pub(super) fn recorded(observation: &eval::Observation) -> Json {
    json!({"status":"executed","semantic":semantic(observation),"steps_used":observation.steps_used,"cleanup":cleanup(observation)})
}
pub(super) fn equal(actual: &eval::Observation, expected: &eval::Observation, id: &str) {
    assert_eq!(semantic(actual), semantic(expected), "{id}");
    assert_eq!(actual.steps_used, expected.steps_used, "{id}: fuel");
    assert_eq!(
        actual.cleanup_trace, expected.cleanup_trace,
        "{id}: cleanup"
    );
}
pub(super) fn expectation(run: &eval::Observation, expected: &Json, id: &str) {
    let mut expected = expected.clone();
    let object = expected.as_object_mut().unwrap();
    object.remove("exit");
    if let Some(codes) = object.remove("diagnostic_codes")
        && object["status"] == "tool_error"
    {
        object.insert("code".to_owned(), codes[0].clone());
    }
    assert_eq!(semantic(run), expected, "{id}: independent expectation");
}
struct Cpu {
    dir: PathBuf,
}
impl Cpu {
    fn new(artifact: &compiled::Artifact) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "invar-gnet-corpus-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        fs::write(
            dir.join("source.cpp"),
            format!("{}\n{DRIVER}", artifact.source()),
        )
        .unwrap();
        child_deadline::cxx(
            child_deadline::CXX17_UBSAN_O1,
            &dir.join("source.cpp"),
            &dir.join("run"),
        );
        Self { dir }
    }
    fn run(&self, args: &[eval::Value], limits: oracle::Limits) -> Vec<i64> {
        let mut command = child_deadline::command(10);
        command.arg(self.dir.join("run"));
        command.args([
            limits.fuel.to_string(),
            limits.transitions.to_string(),
            limits.trace_events.to_string(),
        ]);
        for arg in args {
            command.arg(match arg {
                eval::Value::I32(n) => n.to_string(),
                eval::Value::U32(n) => n.to_string(),
                _ => panic!("i32/u32 arguments only"),
            });
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "CPU run {:?}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout)
            .unwrap()
            .split_whitespace()
            .map(|n| n.parse().unwrap())
            .collect()
    }
}
impl Drop for Cpu {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.dir).unwrap();
    }
}
pub(super) fn words(observation: &oracle::Observation) -> Vec<i64> {
    let (status, start, end) = gnet_words::status(observation);
    let mut words = vec![
        status,
        gnet_words::result(observation),
        start,
        end,
        observation.semantic.steps_used as i64,
        observation.transitions as i64,
        observation.max_frames as i64,
        observation.events.len() as i64,
    ];
    for e in &observation.events {
        words.extend(gnet_words::event_i64(e));
    }
    words
}
fn oracle_record(run: &oracle::Observation) -> Json {
    if run.failure == Some(oracle::ToolFailure::Unsupported) {
        return json!({"status":"unsupported","reason":format!("{:?}",run.semantic.result),"steps_used":run.semantic.steps_used,"events":run.events.len()});
    }
    json!({"status":"executed","semantic":semantic(&run.semantic),"steps_used":run.semantic.steps_used,"cleanup":cleanup(&run.semantic),"failure":run.failure.map(|f|format!("{f:?}")),"transitions":run.transitions,"max_frames":run.max_frames,"event_count":run.events.len(),"full_observation_words_le_i64_sha256":words_if_scalar(run)})
}
fn words_if_scalar(run: &oracle::Observation) -> Json {
    // Array traps have no representation in the current scalar compiled ABI.
    if run.failure == Some(oracle::ToolFailure::Allocation) {
        return Json::Null;
    }
    if matches!(
        run.semantic.result,
        Err(eval::EvalError::Trap {
            category: eval::TrapCategory::IndexOutOfBounds,
            ..
        }) | Err(eval::EvalError::Tool {
            code: "ARRAY_SIZE_LIMIT",
            ..
        })
    ) {
        return Json::Null;
    }
    json!(digest(
        &words(run)
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>()
    ))
}
pub(super) fn compiled_rejection(error: compiled::Error) -> Json {
    assert!(
        matches!(
            error,
            compiled::Error::UnsupportedType
                | compiled::Error::UnsupportedRule(_)
                | compiled::Error::UnsupportedCall
        ),
        "unexpected backend failure {error:?}"
    );
    json!({"status":"unsupported","reason":format!("{error:?}")})
}
pub(super) fn artifact_record(artifact: &compiled::Artifact) -> Json {
    json!({"status":"emitted","source_sha256":digest(artifact.source().as_bytes()),"source_bytes":artifact.source().len(),"states":artifact.states(),"frame_bytes":artifact.frame_bytes(),"execution":"not_performed"})
}

pub fn ordinary_sources() {
    let manifest = document("expanded-manifest.json");
    let details = document("expanded-expectations.json");
    let cases = manifest["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 52);
    assert_eq!(manifest["case_count"], 52);
    let retained = document("manifest.json");
    assert_eq!(
        &cases[..15],
        retained["cases"].as_array().unwrap().as_slice(),
        "all initial cases and expectations are retained unchanged"
    );
    let mut expanded_spec = document("expanded-parity-spec.json");
    expanded_spec["source_corpus"] = json!("manifest.json");
    assert_eq!(
        expanded_spec,
        document("parity-spec.json"),
        "frozen parity gates are unchanged"
    );
    let mut rows = Vec::new();
    for case in cases {
        let id = case["id"].as_str().unwrap();
        let bytes = fs::read(root().join(case["source"].as_str().unwrap())).unwrap();
        assert_eq!(
            digest(&bytes),
            case["source_sha256"].as_str().unwrap(),
            "{id}"
        );
        assert_eq!(bytes.len() as u64, case["source_bytes"].as_u64().unwrap());
        let parsed = front::parse(&bytes).unwrap();
        let function_count = parsed.functions.len();
        let checked = check::check(parsed, true).and_then(|p| {
            invar::ownership::check(&p)?;
            Ok(p)
        });
        let checked = match checked {
            Ok(checked) => {
                assert_eq!(case["expected_check"]["status"], "ok", "{id}");
                checked
            }
            Err(error) => {
                assert_eq!(
                    case["expected_check"]["status"], "source_error",
                    "{id}: {error:?}"
                );
                assert_eq!(
                    error.code,
                    case["expected_check"]["diagnostic_codes"][0]
                        .as_str()
                        .unwrap(),
                    "{id}"
                );
                let rejected = json!({"status":"source_rejected","reason":error.code});
                rows.push(json!({"id":id,"source_sha256":digest(&bytes),"function_count":function_count,"source":rejected,"raw_mir":rejected,"checked_mir":rejected,"oracle":rejected,"compiled_cpu":rejected,"cuda":rejected,"metal":rejected}));
                continue;
            }
        };
        let fuel = case["fuel"].as_u64().unwrap();
        let reference = eval::observe(&checked, fuel);
        expectation(&reference, &case["expected_reference_run"], id);
        let detail = details["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|d| d["id"] == id);
        if let Some(detail) = detail {
            if let Some(steps) = detail["expected_source_steps"].as_u64() {
                assert_eq!(reference.steps_used, steps, "{id}: independent fuel");
            }
            if !detail["expected_cleanup"].is_null() {
                assert_eq!(
                    cleanup(&reference),
                    detail["expected_cleanup"],
                    "{id}: independent cleanup"
                );
            }
        }
        let raw = mir_eval::observe(&mir::lower(&checked).unwrap(), fuel);
        let extra = detail
            .and_then(|d| d["raw_mir_extra_steps"].as_u64())
            .unwrap_or(0);
        assert_eq!(semantic(&raw), semantic(&reference), "{id}: raw MIR result");
        assert_eq!(
            raw.cleanup_trace, reference.cleanup_trace,
            "{id}: raw MIR cleanup"
        );
        assert_eq!(
            raw.steps_used,
            reference.steps_used + extra,
            "{id}: declared raw MIR fuel delta"
        );
        let net = Net::lower(&checked, general_net::Limits::default()).unwrap();
        let proof = net.verify().unwrap();
        let extracted = mir_eval::observe(&net.compiled_mir(&proof).unwrap(), fuel);
        equal(&extracted, &reference, id);
        let limits = oracle::Limits {
            fuel,
            trace_events: 100000,
            ..oracle::Limits::default()
        };
        let observed = oracle::observe(&net, &proof, limits).unwrap();
        if observed.failure == Some(oracle::ToolFailure::Unsupported) {
            assert_eq!(observed.semantic.steps_used, 0, "{id}");
            assert!(observed.events.is_empty(), "{id}");
        } else {
            assert_eq!(observed.failure, None, "{id}: {observed:?}");
            equal(&observed.semantic, &reference, id);
        }
        let mut devices = BTreeMap::new();
        for (label, target) in [
            ("cpu", compiled::Target::Cpu),
            ("cuda", compiled::Target::Cuda),
            ("metal", compiled::Target::Metal),
        ] {
            let outcome =
                match compiled::emit(&net, &proof, "main", target, compiled::Limits::default()) {
                    Ok(artifact) => {
                        let mut result = artifact_record(&artifact);
                        if label == "cpu" {
                            assert_eq!(observed.failure, None, "{id}");
                            assert_eq!(
                                Cpu::new(&artifact).run(&[], limits),
                                words(&observed),
                                "{id}: emitted CPU full observation"
                            );
                            result["status"] = json!("executed");
                            result["execution"] = json!("c++17_ubsan");
                        }
                        result
                    }
                    Err(error) => compiled_rejection(error),
                };
            devices.insert(label, outcome);
        }
        rows.push(json!({"id":id,"source_sha256":digest(&bytes),"function_count":function_count,"structural_rules":net.agents.iter().map(|a|a.rule).chain(net.control.iter().map(|c|c.rule)).map(|r|format!("{r:?}")).collect::<std::collections::BTreeSet<_>>(),"fuel":fuel,"source":recorded(&reference),"raw_mir":recorded(&raw),"raw_mir_expected_extra_steps":extra,"checked_mir":recorded(&extracted),"oracle":oracle_record(&observed),"compiled_cpu":devices["cpu"],"cuda":devices["cuda"],"metal":devices["metal"]}));
    }
    let coverage = structural_coverage(&rows);
    eprintln!(
        "structural coverage {}/{}; not represented: {}",
        coverage["covered_rule_count"],
        coverage["inventory_rule_count"],
        coverage["not_represented_rules"]
    );
    eprintln!(
        "ordinary-source denominator {} (all original 15 retained)",
        rows.len()
    );
    report("ordinary", rows);
}

pub fn runtime_inputs() {
    let vectors = document("runtime-vectors.json");
    let cases = vectors["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 67);
    assert_eq!(vectors["case_count"], 67);
    let mut cpus = BTreeMap::<(String, String), Cpu>::new();
    let mut rows = Vec::new();
    for case in cases {
        let id = case["id"].as_str().unwrap();
        let source = case["source"].as_str().unwrap();
        let entry = case["entry"].as_str().unwrap();
        let bytes = fs::read(root().join(source)).unwrap();
        let checked = check::check(front::parse(&bytes).unwrap(), true).unwrap();
        let net = Net::lower(&checked, general_net::Limits::default()).unwrap();
        let proof = net.verify().unwrap();
        let args = case["arguments"]
            .as_array()
            .unwrap()
            .iter()
            .map(|arg| match arg["type"].as_str().unwrap() {
                "i32" => eval::Value::I32(i32::try_from(arg["value"].as_i64().unwrap()).unwrap()),
                "u32" => eval::Value::U32(u32::try_from(arg["value"].as_u64().unwrap()).unwrap()),
                _ => panic!("scalar type"),
            })
            .collect::<Vec<_>>();
        let fuel = case["fuel"].as_u64().unwrap();
        let reference = eval::named::observe_named(&checked, entry, &args, fuel);
        expectation(&reference, &case["expected_source"], id);
        if let Some(steps) = case["expected_source_steps"].as_u64() {
            assert_eq!(reference.steps_used, steps, "{id}");
        }
        if !case["expected_cleanup"].is_null() {
            assert_eq!(cleanup(&reference), case["expected_cleanup"], "{id}");
        }
        let limits = oracle::Limits {
            fuel,
            transitions: case["limits"]["transitions"].as_u64().unwrap(),
            trace_events: case["limits"]["trace_events"].as_u64().unwrap() as usize,
            array_elements: case["limits"]["array_elements"].as_u64().unwrap() as usize,
        };
        let observed = oracle::observe_call(&net, &proof, entry, &args, limits).unwrap();
        if let Some(failure) = case["expected_oracle_failure"].as_str() {
            assert_eq!(
                observed.failure.map(|f| format!("{f:?}")),
                Some(failure.to_owned()),
                "{id}"
            );
            assert!(observed.semantic.cleanup_trace.is_empty());
        } else if observed.failure == Some(oracle::ToolFailure::Unsupported) {
            assert_eq!(observed.semantic.steps_used, 0, "{id}");
            assert!(observed.events.is_empty());
        } else {
            assert_eq!(observed.failure, None, "{id}: {observed:?}");
            equal(&observed.semantic, &reference, id);
        }
        let key = (source.to_owned(), entry.to_owned());
        let execution = match compiled::emit(
            &net,
            &proof,
            entry,
            compiled::Target::Cpu,
            compiled::Limits::default(),
        ) {
            Ok(artifact) => {
                let cpu = cpus.entry(key).or_insert_with(|| Cpu::new(&artifact));
                assert_eq!(
                    cpu.run(&args, limits),
                    words(&observed),
                    "{id}: CPU status/span/fuel/events/frames"
                );
                json!({"status":"executed","execution":"c++17_ubsan","oracle_full_observation_equal":true})
            }
            Err(error) => compiled_rejection(error),
        };
        rows.push(json!({"id":id,"source_sha256":digest(&bytes),"entry":entry,"arguments":case["arguments"],"limits":case["limits"],"source":recorded(&reference),"oracle":oracle_record(&observed),"compiled_cpu":execution,"raw_mir":{"status":"unavailable","reason":"shared MIR evaluator exposes main-only entry; no synthetic runtime source specialization"}}));
    }
    eprintln!(
        "runtime scalar-input denominator {}, compiled once per source/entry",
        rows.len()
    );
    report("runtime", rows);
}

pub fn owner_fuel_prefixes() {
    let manifest = document("expanded-manifest.json");
    let ids = [
        "array_mutation",
        "array_owner_implicit_return",
        "array_owner_explicit_return",
        "array_take_reborrow",
        "array_replacement",
        "array_loop_cleanup",
        "array_empty_u32",
    ];
    let mut prefixes = 0;
    for id in ids {
        let case = manifest["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["id"] == id)
            .unwrap();
        let bytes = fs::read(root().join(case["source"].as_str().unwrap())).unwrap();
        let checked = check::check(front::parse(&bytes).unwrap(), true).unwrap();
        let complete = eval::observe(&checked, 1000);
        let net = Net::lower(&checked, general_net::Limits::default()).unwrap();
        let proof = net.verify().unwrap();
        let extracted = net.compiled_mir(&proof).unwrap();
        let raw = mir::lower(&checked).unwrap();
        if id == "array_owner_implicit_return" {
            assert_eq!(complete.steps_used, 12);
            assert_eq!(
                semantic(&mir_eval::observe(&raw, 12)),
                json!({"status":"ok","result":{"type":"i32","value":5}}),
                "canonical raw MIR fix preserves exact source boundary"
            );
            assert_eq!(
                semantic(&mir_eval::observe(&raw, 13)),
                json!({"status":"ok","result":{"type":"i32","value":5}})
            );
            assert_eq!(
                semantic(&eval::observe(&checked, 12)),
                json!({"status":"ok","result":{"type":"i32","value":5}})
            );
        }
        for fuel in 1..=complete.steps_used + 1 {
            let source = eval::observe(&checked, fuel);
            equal(&mir_eval::observe(&raw, fuel), &source, id);
            equal(&mir_eval::observe(&extracted, fuel), &source, id);
            let observed = oracle::observe(
                &net,
                &proof,
                oracle::Limits {
                    fuel,
                    ..oracle::Limits::default()
                },
            )
            .unwrap();
            assert_eq!(observed.failure, None);
            equal(&observed.semantic, &source, id);
            prefixes += 1;
        }
    }
    eprintln!(
        "{prefixes} original-source owner fuel prefixes match checked MIR and GNET, including cleanup"
    );
}
