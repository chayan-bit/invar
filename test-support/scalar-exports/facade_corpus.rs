//! Qualification through the actual facade, without duplicating admission rules.
use invar::exports::{
    CompiledExports, ExportEngine, ExportError, ScalarType, ScalarValue, compile_exports,
};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

fn failure(error: &ExportError, source: &str, path: &str) -> Value {
    error.to_diagnostics_json(path, source.as_bytes())
}

pub fn qualify(inputs: Value) -> Value {
    let mut entries = Vec::new();
    let mut reasons = BTreeMap::new();
    let mut original_functions = 0;
    let mut accepted = 0;
    let mut excluded = 0;
    let mut accepted_inputs = 0;
    let mut unique_sources = BTreeSet::new();
    let mut unique_admitted_sources = BTreeSet::new();
    let mut unique_admitted_functions = BTreeSet::new();
    for entry in inputs["entries"].as_array().unwrap() {
        let source = entry["source"].as_str().unwrap();
        let path = entry["id"].as_str().unwrap();
        unique_sources.insert(source);
        let parsed = match invar::front::parse(source.as_bytes()) {
            Ok(program) => program,
            Err(_) => {
                let error =
                    match compile_exports(source.as_bytes(), path, &[], ExportEngine::Reference) {
                        Err(error) => error,
                        Ok(_) => panic!("{path}: unparseable source was exported"),
                    };
                let document = failure(&error, source, path);
                *reasons
                    .entry(
                        document["diagnostics"][0]["code"]
                            .as_str()
                            .unwrap()
                            .to_string(),
                    )
                    .or_insert(0) += 1;
                entries.push(json!({"id":path,"source":source,"functions":[],"status":"unparseable_input","diagnostics":document}));
                continue;
            }
        };
        let mut functions = Vec::new();
        let mut names = Vec::new();
        for declaration in &parsed.functions {
            original_functions += 1;
            let name = declaration.name.as_str();
            let result = compile_exports(source.as_bytes(), path, &[name], ExportEngine::Reference);
            match result {
                Ok(artifact) => {
                    accepted += 1;
                    unique_admitted_sources.insert(source);
                    unique_admitted_functions.insert((source, name.to_string()));
                    names.push(name);
                    let signatures = artifact.signatures();
                    assert_eq!(signatures.len(), 1, "{path}:{name}");
                    let signature = &signatures[0];
                    assert_eq!(signature.source_name, name);
                    assert_eq!(signature.span, declaration.span);
                    functions.push(json!({"name":name,"status":"admitted","result":format!("{:?}",signature.result),
                        "parameters":signature.params.iter().map(|p|json!({"name":p.name,"type":format!("{:?}",p.ty)})).collect::<Vec<_>>(),
                        "span":{"start":signature.span.start,"end":signature.span.end}}));
                }
                Err(error) => {
                    excluded += 1;
                    let diagnostics = failure(&error, source, path);
                    let code = diagnostics["diagnostics"][0]["code"].as_str().unwrap();
                    *reasons.entry(code.to_string()).or_insert(0) += 1;
                    functions
                        .push(json!({"name":name,"status":"excluded","diagnostics":diagnostics}));
                }
            }
        }
        if !names.is_empty() {
            accepted_inputs += 1;
            // Admission also succeeds with the exact complete admitted subset
            // selected together, in source declaration order.
            let combined =
                compile_exports(source.as_bytes(), path, &names, ExportEngine::Reference).unwrap();
            assert_eq!(combined.signatures().len(), names.len());
        }
        let diagnostics = if parsed.functions.is_empty() {
            let error = match compile_exports(source.as_bytes(), path, &[], ExportEngine::Reference)
            {
                Err(error) => error,
                Ok(_) => panic!("{path}: empty selection succeeded"),
            };
            let document = failure(&error, source, path);
            *reasons
                .entry(
                    document["diagnostics"][0]["code"]
                        .as_str()
                        .unwrap()
                        .to_string(),
                )
                .or_insert(0) += 1;
            Some(document)
        } else {
            None
        };
        entries.push(
            json!({"id":path,"source":source,"functions":functions,"diagnostics":diagnostics,
            "status":if names.is_empty(){"no_admitted_functions"}else{"admitted_subset"}}),
        );
    }
    assert_eq!(original_functions, accepted + excluded);
    json!({"schema_version":1,"core_api_input_commit":"6bbc111c8123e7ee64dd8a0652e297641949d5c8",
        "compiler_build_digest":env!("INVAR_BUILD_DIGEST"),
        "debug_assertions":cfg!(debug_assertions),
        "target_arch":std::env::consts::ARCH,"target_os":std::env::consts::OS,
        "scope":inputs["scope"],"scope_limitations":[
            "Rust literal occurrences include incomplete fragments and uninstantiated format templates; parse rejection does not classify their runtime expansions.",
            "Runtime-generated programs outside the replayed M2 corpus are not enumerated; this report qualifies the complete admitted subset of the explicitly enumerated input scope.",
            "Unparseable inputs have no parsed function count. Original source function occurrences are counted; foreign imports and synthetic checked functions are not original export declarations."],
        "admission":"actual_compile_exports_reference_engine_for_every_original_parsed_function",
        "summary":{"input_entries":entries.len(),"original_functions":original_functions,"accepted_functions":accepted,
            "excluded_functions":excluded,"inputs_with_admitted_functions":accepted_inputs,
            "unique_source_texts":unique_sources.len(),"unique_admitted_source_texts":unique_admitted_sources.len(),
            "unique_admitted_source_function_pairs":unique_admitted_functions.len(),"exclusion_reason_codes":reasons},"entries":entries})
}

fn result_value(result: Result<ScalarValue, ExportError>, artifact: &CompiledExports) -> Value {
    match result {
        Ok(ScalarValue::I32(value)) => json!({"status":"ok","result":{"type":"i32","value":value}}),
        Ok(ScalarValue::U32(value)) => json!({"status":"ok","result":{"type":"u32","value":value}}),
        Err(error) => artifact.diagnostics(&error),
    }
}

/// Fixed boundary vectors rather than a new random seed or guessed numeric bound.
fn arguments(parameters: &[invar::exports::ExportParameter], variant: usize) -> Vec<ScalarValue> {
    parameters
        .iter()
        .enumerate()
        .map(|(index, parameter)| match parameter.ty {
            ScalarType::F32 => unreachable!(
                "f32 array parameters require the host-array corpus, not scalar invoke"
            ),
            ScalarType::I32 => ScalarValue::I32(match variant {
                0 => 0,
                1 => 1,
                2 => -1,
                3 => i32::MIN,
                4 => i32::MAX,
                5 => {
                    if index % 2 == 0 {
                        i32::MIN
                    } else {
                        i32::MAX
                    }
                }
                6 => {
                    if index == 0 {
                        7
                    } else {
                        3
                    }
                }
                _ => unreachable!(),
            }),
            ScalarType::U32 => ScalarValue::U32(match variant {
                0 | 2 | 3 => 0,
                1 => 1,
                4 => u32::MAX,
                5 => {
                    if index % 2 == 0 {
                        u32::MAX
                    } else {
                        1
                    }
                }
                6 => {
                    if index == 0 {
                        7
                    } else {
                        3
                    }
                }
                _ => unreachable!(),
            }),
        })
        .collect()
}

// Arrays admitted by HA-1 must be replayed through the real host-array entry,
// rather than sending fabricated scalar arguments and comparing ABI refusals.
fn invoke_case(
    artifact: &mut CompiledExports,
    signature: &invar::exports::ExportSignature,
    variant: usize,
    fuel: u64,
) -> Value {
    use invar::{
        eval::host_read::{HostArgument as A, HostMutArray as M, HostReadArray as R},
        exports::ArrayMode,
    };
    if signature.params.iter().all(|p| p.array.is_none()) {
        let result = artifact.invoke(
            &signature.source_name,
            &arguments(&signature.params, variant),
            fuel,
        );
        return result_value(result, artifact);
    }
    let scalars = signature
        .params
        .iter()
        .filter(|p| p.array.is_none())
        .cloned()
        .collect::<Vec<_>>();
    let mut scalars = arguments(&scalars, variant).into_iter();
    let mut signed = vec![vec![variant as i32 - 3; 4]; signature.params.len()];
    let mut unsigned = vec![vec![variant as u32; 4]; signature.params.len()];
    let mut floats = vec![vec![variant as f32 - 3.0; 4]; signature.params.len()];
    let mut args = Vec::new();
    for (((p, i), u), f) in signature
        .params
        .iter()
        .zip(&mut signed)
        .zip(&mut unsigned)
        .zip(&mut floats)
    {
        args.push(match (p.array, p.ty) {
            (None, _) => match scalars.next().unwrap() {
                ScalarValue::I32(v) => A::I32(v),
                ScalarValue::U32(v) => A::U32(v),
            },
            (Some(ArrayMode::Read), ScalarType::I32) => A::Read(R::I32(i)),
            (Some(ArrayMode::Read), ScalarType::U32) => A::Read(R::U32(u)),
            (Some(ArrayMode::Read), ScalarType::F32) => A::Read(R::Float32(f)),
            (Some(ArrayMode::Mut), ScalarType::I32) => A::Mut(M::I32(i)),
            (Some(ArrayMode::Mut), ScalarType::U32) => A::Mut(M::U32(u)),
            (Some(ArrayMode::Mut), ScalarType::F32) => A::Mut(M::Float32(f)),
        });
    }
    let result = artifact
        .invoke_arrays(&signature.source_name, &mut args, fuel)
        .map(|outcome| outcome.value);
    drop(args);
    let mut result = result_value(result, artifact);
    result["array_outputs"] = json!({"i32": signed, "u32": unsigned,
        "f32_bits": floats.iter().map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>()).collect::<Vec<_>>()});
    result
}

pub fn replay(report: &Value, native: bool) -> Value {
    let mut invocations = 0;
    let mut functions = 0;
    let mut inputs = 0usize;
    let mut outcomes = BTreeMap::new();
    let mut results = Vec::new();
    for entry in report["entries"].as_array().unwrap() {
        let names = entry["functions"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|function| function["status"] == "admitted")
            .map(|function| function["name"].as_str().unwrap())
            .collect::<Vec<_>>();
        if names.is_empty() {
            continue;
        }
        inputs += 1;
        let source = entry["source"].as_str().unwrap().as_bytes();
        let path = entry["id"].as_str().unwrap();
        let mut reference = compile_exports(source, path, &names, ExportEngine::Reference).unwrap();
        let mut compiled = if native {
            Some(
                compile_exports(source, path, &names, ExportEngine::NativeSerial)
                    .unwrap_or_else(|e| panic!("{path}: native admission/build failed: {e:?}")),
            )
        } else {
            None
        };
        let signatures = reference.signatures().to_vec();
        if let Some(compiled) = &compiled {
            assert_ne!(compiled.artifact_id(), reference.artifact_id());
            assert_eq!(compiled.signatures().len(), signatures.len());
            for (a, b) in compiled.signatures().iter().zip(&signatures) {
                assert_eq!(a.source_name, b.source_name);
                assert_eq!(a.params, b.params);
                assert_eq!(a.result, b.result);
                assert_eq!(a.span, b.span);
            }
        }
        for signature in signatures {
            functions += 1;
            // No-argument functions have only one distinct argument vector.
            let variants = if signature.params.is_empty() { 1 } else { 7 };
            for variant in 0..variants {
                for fuel in [1, 64, 10_000] {
                    let expected = invoke_case(&mut reference, &signature, variant, fuel);
                    if let Some(compiled) = &mut compiled {
                        let actual = invoke_case(compiled, &signature, variant, fuel);
                        assert_eq!(
                            actual, expected,
                            "{path}:{} variant {variant}, fuel {fuel}",
                            signature.source_name
                        );
                    }
                    invocations += 1;
                    *outcomes
                        .entry(expected["status"].as_str().unwrap().to_string())
                        .or_insert(0) += 1;
                    results.push(json!({"input":path,"function":signature.source_name,"argument_variant":variant,"fuel":fuel,"outcome":expected}));
                }
            }
        }
        if native && inputs.is_multiple_of(25) {
            eprintln!(
                "native corpus progress: {inputs} input artifacts, {functions} functions, {invocations} equivalent calls"
            );
        }
    }
    assert_eq!(
        functions,
        report["summary"]["accepted_functions"].as_u64().unwrap()
    );
    json!({"schema_version":1,"engine":if native{"reference_vs_native_exports"}else{"reference_exports"},
        "summary":{"inputs":inputs,"functions":functions,"invocations":invocations,"outcomes":outcomes},"results":results})
}

pub fn run_vectors(vectors: Value, native: bool) {
    let cases = vectors["cases"].as_array().unwrap();
    let source = cases[0]["source"].as_str().unwrap();
    let mut names = Vec::new();
    for case in cases {
        let name = case["function"].as_str().unwrap();
        if !names.contains(&name) {
            names.push(name);
        }
    }
    let mut reference = compile_exports(
        source.as_bytes(),
        "scalar_exports.inv",
        &names,
        ExportEngine::Reference,
    )
    .unwrap();
    let mut compiled = if native {
        Some(
            compile_exports(
                source.as_bytes(),
                "scalar_exports.inv",
                &names,
                ExportEngine::NativeSerial,
            )
            .unwrap(),
        )
    } else {
        None
    };
    for case in cases {
        let args = case["arguments"]
            .as_array()
            .unwrap()
            .iter()
            .map(|argument| {
                if argument["type"] == "i32" {
                    ScalarValue::I32(i32::try_from(argument["value"].as_i64().unwrap()).unwrap())
                } else {
                    ScalarValue::U32(u32::try_from(argument["value"].as_u64().unwrap()).unwrap())
                }
            })
            .collect::<Vec<_>>();
        let name = case["function"].as_str().unwrap();
        let fuel = case["fuel"].as_u64().unwrap();
        for artifact in std::iter::once(&mut reference).chain(compiled.iter_mut()) {
            let actual = artifact.invoke(name, &args, fuel);
            match (&actual, case["expected"]["status"].as_str().unwrap()) {
                (Ok(value), "ok") => {
                    let actual = match value {
                        ScalarValue::I32(v) => json!({"type":"i32","value":v}),
                        ScalarValue::U32(v) => json!({"type":"u32","value":v}),
                    };
                    assert_eq!(actual, case["expected"]["result"], "{}", case["id"]);
                }
                (
                    Err(ExportError::Evaluation(invar::eval::EvalError::Trap { category, span })),
                    "trap",
                ) => {
                    let expected = &case["expected"]["trap"];
                    let category = match category {
                        invar::eval::TrapCategory::IntegerOverflow => "INTEGER_OVERFLOW",
                        invar::eval::TrapCategory::DivisionByZero => "DIVISION_BY_ZERO",
                        other => panic!("unexpected scalar trap {other:?}"),
                    };
                    assert_eq!(
                        json!({"category":category,"span":{"start":span.start,"end":span.end}}),
                        *expected,
                        "{}",
                        case["id"]
                    );
                }
                (
                    Err(ExportError::Evaluation(invar::eval::EvalError::EvaluationLimit {
                        fuel_limit,
                        steps_used,
                    })),
                    "evaluation_limit",
                ) => {
                    assert_eq!(*fuel_limit, fuel);
                    assert_eq!(*steps_used, fuel);
                }
                other => panic!("{}: unexpected vector outcome {other:?}", case["id"]),
            }
        }
    }
    eprintln!(
        "59 hand-derived scalar vectors passed through {} facade",
        if native {
            "reference/native compiled"
        } else {
            "reference"
        }
    );
}
