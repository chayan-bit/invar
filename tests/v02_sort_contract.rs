#![forbid(unsafe_code)]

use invar::{check, eval, front, ownership};
use std::{fs, process::Command};

#[path = "support/gnet_agree.rs"]
mod agree;
#[path = "support/language_limits.rs"]
mod language_limits;

fn checked(source: &str) -> check::CheckedProgram {
    let program = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    ownership::check(&program).unwrap();
    program
}

fn returns_seven(source: &str) {
    let checked = checked(source);
    let observation = eval::observe(&checked, 100_000);
    assert_eq!(observation.result, Ok(eval::Value::I32(7)), "{source}");
    let program = invar::mir::lower(&checked).unwrap();
    let mut actual = vec![invar::mir_eval::observe(&program, 100_000)];
    for workers in [1, 2, 4] {
        actual.push(
            invar::schedule::run_source(source.as_bytes(), &program, 100_000, workers)
                .unwrap()
                .observation,
        );
    }
    for actual in actual {
        assert_eq!(actual.result, observation.result, "{source}");
        assert_eq!(actual.steps_used, observation.steps_used, "{source}");
        assert_eq!(actual.cleanup_trace, observation.cleanup_trace, "{source}");
    }
}

#[test]
fn sort_contract_explain_schema_and_registered_reasons() {
    let v5: serde_json::Value =
        serde_json::from_str(include_str!("../schema/explain-v5.schema.json")).unwrap();
    let device_schema: serde_json::Value = serde_json::from_str(include_str!(
        "../schema/invar-explain-device-v5.schema.json"
    ))
    .unwrap();
    let registry: serde_json::Value =
        serde_json::from_str(include_str!("../schema/diagnostic-codes-v1.json")).unwrap();
    for engine in [None, Some("auto")] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_invar"));
        command.args(["explain", "examples/sort.inv"]);
        if let Some(engine) = engine {
            command.args(["--engine", engine]);
        }
        command.args(["--format", "json"]);
        let output = command.output().unwrap();
        assert!(output.status.success(), "{output:?}");
        let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(document["schema"], "invar-explain-v5");
        assert_eq!(document["schema_version"], 5);
        assert!(jsonschema::is_valid(&v5, &document), "{document}");
        let mut relabelled = document.clone();
        relabelled["schema"] = "invar-explain-v1".into();
        relabelled["schema_version"] = 1.into();
        assert!(!jsonschema::is_valid(&v5, &relabelled));
        let sites = document["explicit_sorts"].as_array().unwrap();
        assert_eq!(sites.len(), 4);
        for site in sites {
            assert!(
                registry["explain_reason_codes"]
                    .as_array()
                    .unwrap()
                    .contains(&site["reason"])
            );
        }
        if engine.is_some() {
            assert!(jsonschema::is_valid(
                &device_schema,
                &document["device_execution"]
            ));
        }
        for bad in [
            "unknown reason",
            "unknown operation",
            "missing rule",
            "extra field",
        ] {
            let mut malformed = document.clone();
            let site = &mut malformed["explicit_sorts"][0];
            match bad {
                "unknown reason" => site["reason"] = "unknown".into(),
                "unknown operation" => site["operation"] = "unknown".into(),
                "missing rule" => {
                    site.as_object_mut().unwrap().remove("rule");
                }
                _ => site["extra"] = true.into(),
            }
            assert!(!jsonschema::is_valid(&v5, &malformed), "{bad}");
        }
    }
}

#[test]
fn sort_contract_general_net_oracle_preserves_values_traps_fuel_and_cleanup() {
    use invar::general_net::{Limits, Net, oracle};
    // The padding fills the live ceiling to one element short, so argsort's
    // result is the allocation that crosses it.
    let ceiling = format!(
        "fn main() -> i32 {{ {} let xs = filled_u32(1u32,0u32); let ids = argsort(xs); return 0; }}",
        language_limits::padding(invar::array_limits::MAX_LIVE_ELEMENTS - 1)
    );
    for source in [
        include_str!("../examples/sort.inv"),
        "fn main() -> i32 { let n = 0.0/0.0; var keys: Array<i32> = [2,1,1]; var payload: Array<f32> = [n,-0.0,0.0]; sort_by_key(keys,payload); if keys[0u32] == 1 and 1.0/payload[0u32] < 0.0 and 1.0/payload[1u32] > 0.0 and payload[2u32] != payload[2u32] { return 7; } return 0; }",
        "fn main() -> i32 { let n = 0.0/0.0; let xs: Array<f32> = [n,0.0,-0.0,-1.0]; let ids = argsort(xs); if ids[0u32] == 3u32 and ids[1u32] == 2u32 and ids[2u32] == 1u32 and ids[3u32] == 0u32 { return 7; } return 0; }",
        "fn main() -> i32 { let empty: Array<u32> = []; let ids = argsort(empty); var xs: Array<i32> = [2,1]; let offsets: Array<u32> = []; segmented_sort(xs,offsets); if len(ids) == 0u32 and xs[0u32] == 2 { return 7; } return 0; }",
        "fn main() -> i32 { var xs: Array<i32> = [2,1]; var ids: Array<u32> = [0u32]; sort_by_key(xs,ids); return 0; }",
        "fn main() -> i32 { var xs: Array<u32> = [2u32,1u32]; let offsets: Array<u32> = [0u32,2u32,1u32]; segmented_sort(xs,offsets); return 0; }",
        "fn main() -> i32 { var xs: Array<f32> = [2.0,1.0]; let offsets: Array<u32> = [0u32,3u32]; segmented_sort(xs,offsets); return 0; }",
        &ceiling,
    ] {
        let checked = checked(source);
        let complete = eval::observe(&checked, 100_000);
        let net = Net::lower(&checked, Limits::default()).unwrap();
        let proof = net.verify().unwrap();
        for fuel in [100_000, complete.steps_used - 1] {
            let expected = eval::observe(&checked, fuel);
            let actual = oracle::observe(
                &net,
                &proof,
                oracle::Limits {
                    fuel,
                    array_elements: invar::array_limits::MAX_LIVE_ELEMENTS,
                    ..oracle::Limits::default()
                },
            )
            .unwrap();
            assert_eq!(actual.failure, None, "{source}");
            agree::same(&actual.semantic, &expected, source);
        }
    }
}

#[test]
fn sort_contract_numeric_keys_and_stable_payloads() {
    for (ty, values, first, last) in [
        ("u32", "3u32,0u32,3u32,1u32", "0u32", "3u32"),
        ("i32", "3i32,-2i32,3i32,0i32", "-2i32", "3i32"),
        ("f32", "3.0,-2.0,3.0,0.0", "-2.0", "3.0"),
    ] {
        returns_seven(&format!(
            "fn main() -> i32 {{
                var keys: Array<{ty}> = [{values}];
                var ids: Array<u32> = [0u32,1u32,2u32,3u32];
                sort_by_key(keys,ids);
                if keys[0u32] == {first} and keys[3u32] == {last}
                    and ids[0u32] == 1u32 and ids[1u32] == 3u32
                    and ids[2u32] == 0u32 and ids[3u32] == 2u32 {{ return 7; }}
                return 0;
            }}"
        ));
    }
}

#[test]
fn sort_contract_argsort_total_order_and_input_preservation() {
    returns_seven(
        "fn main() -> i32 {
            let n = 0.0 / 0.0;
            let inf = 1.0 / 0.0;
            let xs: Array<f32> = [n,0.0,-0.0,inf,-inf,n,-1.0,1.0];
            let ids = argsort(xs);
            let expected: Array<u32> = [4u32,6u32,2u32,1u32,7u32,3u32,0u32,5u32];
            for i in 0u32..len(ids) { if ids[i] != expected[i] { return 0; } }
            if xs[0u32] == xs[0u32] or 1.0 / xs[1u32] < 0.0
                or 1.0 / xs[2u32] > 0.0 { return 0; }
            return 7;
        }",
    );
}

#[test]
fn sort_contract_empty_singleton_and_repeat_sort() {
    returns_seven(
        "fn main() -> i32 {
            var empty: Array<i32> = [];
            sort(empty);
            let ids = argsort(empty);
            var one: Array<u32> = [4294967295u32];
            sort(one);
            sort(one);
            if len(ids) == 0u32 and one[0u32] == 4294967295u32 { return 7; }
            return 0;
        }",
    );
}

#[test]
fn sort_contract_full_boundary_segments_and_duplicate_offsets() {
    returns_seven(
        "fn main() -> i32 {
            var xs: Array<i32> = [3i32,1i32,2i32,-1i32];
            let offsets: Array<u32> = [0u32,2u32,2u32,4u32];
            segmented_sort(xs,offsets);
            if xs[0u32] == 1i32 and xs[1u32] == 3i32
                and xs[2u32] == -1i32 and xs[3u32] == 2i32 { return 7; }
            return 0;
        }",
    );
}

#[test]
fn sort_contract_zero_segments_for_empty_array() {
    for offsets in ["0u32", "0u32,0u32"] {
        returns_seven(&format!(
            "fn main() -> i32 {{ var xs: Array<i32> = [];
             let offsets: Array<u32> = [{offsets}]; segmented_sort(xs,offsets);
             if len(xs) == 0u32 {{ return 7; }} return 0; }}"
        ));
    }
}

#[test]
fn sort_contract_unnamed_bool_array_indexes_like_a_named_one() {
    // A call result is not a bare name, so the reference indexes the owned value.
    returns_seven(
        "fn bits() -> Array<Bool> { return [false,true]; }
        fn main() -> i32 { if bits()[1u32] and not bits()[0u32] { return 7; } return 0; }",
    );
}

#[test]
fn sort_contract_bool_payload() {
    returns_seven(
        "fn main() -> i32 {
        var keys: Array<i32> = [2i32,1i32,2i32,1i32];
        var values: Array<Bool> = [true,false,false,true];
        sort_by_key(keys,values);
        if not values[0u32] and values[1u32] and values[2u32] and not values[3u32] { return 7; }
        return 0;
    }",
    );
}

#[test]
fn sort_contract_mismatch_and_invalid_offsets_trap() {
    for body in [
        "var xs: Array<i32> = [1i32]; var vs: Array<u32> = []; sort_by_key(xs,vs);",
        "var xs: Array<i32> = [1i32]; let off: Array<u32> = [0u32,2u32]; segmented_sort(xs,off);",
        "var xs: Array<i32> = [1i32]; let off: Array<u32> = [1u32,0u32]; segmented_sort(xs,off);",
        "var xs: Array<i32> = [3i32,1i32,2i32]; let off: Array<u32> = []; segmented_sort(xs,off);",
        "var xs: Array<i32> = [3i32,1i32,2i32]; let off: Array<u32> = [0u32]; segmented_sort(xs,off);",
        "var xs: Array<i32> = [3i32,1i32,2i32]; let off: Array<u32> = [1u32,3u32]; segmented_sort(xs,off);",
        "var xs: Array<i32> = [3i32,1i32,2i32]; let off: Array<u32> = [0u32,2u32]; segmented_sort(xs,off);",
        "var xs: Array<i32> = []; let off: Array<u32> = []; segmented_sort(xs,off);",
    ] {
        let source = format!("fn main() -> i32 {{ {body} return 0; }}");
        let checked = checked(&source);
        let reference = eval::observe(&checked, 100_000);
        let program = invar::mir::lower(&checked).unwrap();
        let mir = invar::mir_eval::observe(&program, 100_000);
        assert_eq!(mir.result, reference.result);
        assert_eq!(mir.steps_used, reference.steps_used);
        assert_eq!(mir.cleanup_trace, reference.cleanup_trace);
        for workers in [1, 2, 4] {
            let actual = invar::schedule::run_source(source.as_bytes(), &program, 100_000, workers)
                .unwrap()
                .observation;
            assert_eq!(actual.result, reference.result);
            assert_eq!(actual.steps_used, reference.steps_used);
            assert_eq!(actual.cleanup_trace, reference.cleanup_trace);
        }
        #[cfg(feature = "native")]
        {
            let path =
                std::env::temp_dir().join(format!("invar-sort-trap-{}.inv", std::process::id()));
            fs::write(&path, &source).unwrap();
            let run = |engine| {
                Command::new(env!("CARGO_BIN_EXE_invar"))
                    .arg("run")
                    .arg(&path)
                    .args(["--engine", engine, "--format", "json"])
                    .output()
                    .unwrap()
            };
            let expected = run("reference");
            let actual = run("native-serial");
            fs::remove_file(path).unwrap();
            assert!(!expected.status.success());
            assert_eq!(actual.status.code(), expected.status.code());
            let expected: serde_json::Value = serde_json::from_slice(&expected.stdout).unwrap();
            let actual: serde_json::Value = serde_json::from_slice(&actual.stdout).unwrap();
            assert_eq!(actual["status"], expected["status"]);
            assert_eq!(
                actual["diagnostics"][0]["message"],
                "native execution trapped"
            );
            assert_eq!(expected["diagnostics"][0]["message"], "evaluation trapped");
            let mut actual_diagnostic = actual["diagnostics"][0].clone();
            let mut expected_diagnostic = expected["diagnostics"][0].clone();
            actual_diagnostic.as_object_mut().unwrap().remove("message");
            expected_diagnostic
                .as_object_mut()
                .unwrap()
                .remove("message");
            assert_eq!(actual_diagnostic, expected_diagnostic);
        }
        assert!(matches!(
            reference.result,
            Err(eval::EvalError::Trap {
                category: eval::TrapCategory::IndexOutOfBounds,
                ..
            })
        ));
    }
}

#[test]
fn sort_contract_rejects_immutable_alias_and_live_view_mutation() {
    for body in [
        "let xs: Array<u32> = [1u32]; sort(xs);",
        "var xs: Array<u32> = [1u32]; sort_by_key(xs,xs);",
        "var xs: Array<u32> = [1u32]; segmented_sort(xs,xs);",
        "var xs: Array<u32> = [1u32]; let v = view(xs,0u32,1u32); sort(xs); let x = v[0u32];",
    ] {
        let source = format!("fn main() -> i32 {{ {body} return 0; }}");
        let parsed = front::parse(source.as_bytes()).unwrap();
        if let Ok(program) = check::check(parsed, true) {
            assert!(ownership::check(&program).is_err(), "{source}");
        }
    }
}

#[test]
fn sort_contract_rejects_invalid_signatures_and_borrowers() {
    for source in [
        "fn main() -> i32 { var keys: Array<u32> = [2u32,1u32]; var values: Array<f64> = [1.0f64,2.0f64]; sort_by_key(keys,values); return 0; }",
        "fn main() -> i32 { var xs: Array<Bool> = [true]; sort(xs); return 0; }",
        "fn main() -> i32 { var xs: Array<u32> = [1u32]; sort(xs,xs); return 0; }",
        "fn main() -> i32 { var xs: Array<u32> = [1u32]; let ids = argsort(move xs); return 0; }",
        "fn main() -> i32 { sort(filled_u32(1u32,0u32)); return 0; }",
        "fn run(xs: read Array<u32>) -> Unit { sort(xs); } fn main() -> i32 { return 0; }",
        "fn sort() -> Unit {} fn main() -> i32 { return 0; }",
    ] {
        let parsed = front::parse(source.as_bytes()).unwrap();
        if let Ok(program) = check::check(parsed, true) {
            assert!(ownership::check(&program).is_err(), "{source}");
        }
    }
}

#[test]
fn sort_contract_reference_cli() {
    let directory = std::env::temp_dir().join(format!("invar-sort-{}", std::process::id()));
    fs::create_dir(&directory).unwrap();
    let path = directory.join("sort.inv");
    fs::write(
        &path,
        "fn main() -> i32 { var xs: Array<i32> = [2i32,1i32]; sort(xs); return xs[0u32]; }",
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .arg("run")
        .arg(&path)
        .args(["--engine", "reference", "--format", "json"])
        .output()
        .unwrap();
    fs::remove_dir_all(directory).unwrap();
    assert!(output.status.success(), "{:?}", output);
    let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(document["status"], "ok");
    assert_eq!(document["result"]["value"], 1);
}

#[test]
fn sort_contract_mir_lowering_and_reference_agreement() {
    let source = "fn main() -> i32 { var keys: Array<i32> = [3i32,1i32,3i32];
        var payload: Array<Bool> = [true,false,false]; sort_by_key(keys,payload);
        let ids = argsort(keys); let offsets: Array<u32> = [0u32,3u32];
        segmented_sort(keys,offsets); sort(keys);
        if keys[0u32] == 1i32 and not payload[0u32] and payload[1u32]
            and not payload[2u32] and ids[2u32] == 2u32 { return 7; } return 0; }";
    let checked = checked(source);
    let program = invar::mir::lower(&checked).unwrap();
    let reference = eval::observe(&checked, 100_000);
    let actual = invar::mir_eval::observe(&program, 100_000);
    assert_eq!(actual.result, reference.result);
    assert_eq!(actual.result, Ok(eval::Value::I32(7)));
    assert_eq!(actual.steps_used, reference.steps_used);
    assert_eq!(actual.cleanup_trace, reference.cleanup_trace);
}

#[test]
fn sort_contract_malformed_mir_loans_are_rejected() {
    use invar::{
        front::Mode,
        mir::{self, Operation},
    };
    let checked = checked(
        "fn main() -> i32 { var keys: Array<u32> = [2u32,1u32];
        var ids: Array<u32> = [0u32,1u32]; sort_by_key(keys,ids); return 0; }",
    );
    let valid = mir::lower(&checked).unwrap();
    for replacement in [None, Some(Mode::Read), Some(Mode::Take)] {
        let mut malformed = valid.clone();
        let origin = malformed.functions[0]
            .blocks
            .iter_mut()
            .flat_map(|b| &mut b.instructions)
            .find(|i| {
                matches!(
                    i.kind,
                    Operation::ArrayRef {
                        loan: Some(Mode::Mut),
                        ..
                    }
                )
            })
            .unwrap();
        let Operation::ArrayRef { loan, .. } = &mut origin.kind else {
            panic!()
        };
        *loan = replacement;
        assert!(mir::verify(&malformed).is_err());
    }
    let mut malformed = valid.clone();
    let call = malformed.functions[0]
        .blocks
        .iter_mut()
        .flat_map(|b| &mut b.instructions)
        .find(|i| matches!(&i.kind,Operation::Call{name,..} if name=="sort_by_key"))
        .unwrap();
    let Operation::Call { args, .. } = &mut call.kind else {
        panic!()
    };
    args[1] = args[0];
    assert!(mir::verify(&malformed).is_err());
}

#[test]
fn sort_contract_cpu_parallel_reaches_workers_and_preserves_observation() {
    let source = "fn main() -> i32 { var xs = filled_u32(128u32,0u32);
        for i in 0u32..128u32 { xs[i] = 127u32-i; }
        sort(xs); var tiny: Array<u32> = [1u32,0u32]; sort(tiny);
        if tiny[0u32] == 0u32 and xs[0u32] == 0u32 and xs[127u32] == 127u32 { return 7; } return 0; }";
    let checked = checked(source);
    let program = invar::mir::lower(&checked).unwrap();
    let reference = eval::observe(&checked, 100_000);
    for workers in [1, 2, 4] {
        let run =
            invar::schedule::run_source(source.as_bytes(), &program, 100_000, workers).unwrap();
        assert_eq!(run.observation.result, reference.result);
        assert_eq!(run.observation.steps_used, reference.steps_used);
        assert_eq!(run.observation.cleanup_trace, reference.cleanup_trace);
        assert_eq!(run.plan, invar::plan::PlanKind::Sort);
        assert_eq!(run.executed.len(), 2);
        assert_eq!(run.joined > 0, workers > 1);
        assert_eq!(
            run.executed[0].implementation,
            Some(if workers == 1 {
                "cpu-lsd-radix-4x8"
            } else {
                "cpu-parallel-lsd-radix-4x8"
            })
        );
        assert_eq!(run.executed[1].implementation, Some("cpu-lsd-radix-4x8"));
    }
}

#[cfg(feature = "native")]
#[test]
fn sort_contract_native_cli() {
    let directory = std::env::temp_dir().join(format!("invar-sort-native-{}", std::process::id()));
    fs::create_dir(&directory).unwrap();
    let path = directory.join("sort.inv");
    fs::write(
        &path,
        "fn main() -> i32 {
        let n = 0.0 / 0.0;
        var xs: Array<f32> = [n,0.0,-0.0,-1.0,n];
        var payload: Array<Bool> = [true,false,true,false,false];
        sort_by_key(xs,payload);
        let ids = argsort(xs);
        let offsets: Array<u32> = [0u32,5u32]; segmented_sort(xs,offsets);
        var ints: Array<i32> = [3i32,-2i32,1i32]; sort(ints);
        var uints: Array<u32> = [4294967295u32,0u32]; sort(uints);
        if xs[0u32] == -1.0 and 1.0 / xs[1u32] < 0.0
            and 1.0 / xs[2u32] > 0.0 and xs[3u32] != xs[3u32]
            and not payload[0u32] and payload[1u32] and not payload[2u32]
            and payload[3u32] and not payload[4u32] and ids[4u32] == 4u32
            and ints[0u32] == -2i32 and uints[0u32] == 0u32 { return 7; } return 0;
    }",
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .arg("run")
        .arg(&path)
        .args(["--engine", "native-serial", "--format", "json"])
        .output()
        .unwrap();
    fs::remove_dir_all(directory).unwrap();
    assert!(output.status.success(), "{:?}", output);
    let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(document["result"]["value"], 7, "{document}");
}
