//! Direct reference entry for the frozen scalar-export interface.
//!
//! Selection and scalar argument validation happen before constructing runtime
//! state. The checked source body executes without a synthetic call expression,
//! retaining its original fuel convention, operator spans and cleanup trace.
use super::*;

fn refused(code: &'static str, message: &'static str) -> Observation {
    Observation {
        result: Err(EvalError::tool(code, message)),
        steps_used: 0,
        cleanup_trace: Vec::new(),
    }
}

/// Observe one named, checked i32/u32 function with at most eight scalar inputs.
///
/// Invalid names, signatures, arities and exact argument types report
/// `NATIVE_ABI` with zero source steps. Zero fuel reports `INVALID_FUEL`.
/// Export selection/MIR admission belongs to the compiler facade; this entry
/// invokes the original checked function without cloning or rewriting source.
pub fn observe_named(
    program: &CheckedProgram,
    name: &str,
    args: &[Value],
    fuel_limit: u64,
) -> Observation {
    observe_scalar(program, name, args, fuel_limit, false, false)
}

/// Direct source observation for private scalar numeric qualification. This
/// accepts exact I32/U32/Bool/Unit/F32 values without changing export signatures,
/// wire tags or the frozen observe_named admission. No synthetic call is added.
pub fn observe_numeric_named(
    program: &CheckedProgram,
    name: &str,
    args: &[Value],
    fuel_limit: u64,
) -> Observation {
    observe_scalar(program, name, args, fuel_limit, true, false)
}

/// Direct source observation for private numeric owner qualification.
/// Inputs retain exact scalar admission; only the result may be a direct
/// numeric array. This does not change public exports or scalar observers.
pub fn observe_owned_numeric_named(
    program: &CheckedProgram,
    name: &str,
    args: &[Value],
    fuel_limit: u64,
) -> Observation {
    observe_scalar(program, name, args, fuel_limit, true, true)
}

fn observe_scalar(
    program: &CheckedProgram,
    name: &str,
    args: &[Value],
    fuel_limit: u64,
    numeric: bool,
    owned_result: bool,
) -> Observation {
    if fuel_limit == 0 {
        return refused("INVALID_FUEL", "fuel must be positive");
    }
    let Some(function) = program
        .program
        .functions
        .iter()
        .find(|function| function.name == name)
    else {
        return refused("NATIVE_ABI", "unknown scalar function");
    };
    let admitted = |ty: &Type| {
        matches!(ty, Type::I32 | Type::U32)
            || numeric && matches!(ty, Type::Bool | Type::Unit | Type::F32)
    };
    let result_admitted = admitted(&function.result)
        || owned_result
            && matches!(
                function.result,
                Type::ArrayI32 | Type::ArrayU32 | Type::ArrayF32
            );
    if !result_admitted
        || function
            .params
            .iter()
            .any(|param| param.mode.is_some() || !admitted(&param.ty))
    {
        return refused(
            "NATIVE_ABI",
            if numeric {
                "function signature is outside numeric scalar reference admission"
            } else {
                "function signature is not scalar i32/u32"
            },
        );
    }
    if function.params.len() > 8 || args.len() != function.params.len() {
        return refused("NATIVE_ABI", "scalar function arity is invalid");
    }
    let mut values = Vec::with_capacity(args.len());
    for (param, value) in function.params.iter().zip(args) {
        let value = match (&param.ty, value) {
            (Type::I32, Value::I32(value)) => Value::I32(*value),
            (Type::U32, Value::U32(value)) => Value::U32(*value),
            (Type::F32, Value::F32(value)) if numeric => Value::F32(*value),
            (Type::Bool, Value::Bool(value)) if numeric => Value::Bool(*value),
            (Type::Unit, Value::Unit) if numeric => Value::Unit,
            _ => return refused("NATIVE_ABI", "scalar argument type does not match"),
        };
        values.push(value);
    }
    let values = super::stack::Values::new(values);
    super::stack::observe_on_own_stack(move || {
        let mut evaluator = Evaluator {
            program,
            types: program.types_for(name),
            fuel_limit,
            steps_used: 0,
            call_depth: 0,
            trace: Vec::new(),
            live_elements: 0,
            pending_result_error: None,
        };
        let result = evaluator.call_user(name, values.into_values());
        Observation {
            result,
            steps_used: evaluator.steps_used,
            cleanup_trace: evaluator.trace,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value as Json, json};

    fn checked(source: &[u8]) -> CheckedProgram {
        let checked = crate::check::check(crate::front::parse(source).unwrap(), false).unwrap();
        crate::ownership::check(&checked).unwrap();
        checked
    }

    #[test]
    fn issue_l1_06_scalar_export_corpus_named_exact_values_and_spans() {
        let cases: Json = serde_json::from_str(include_str!(
            "../../test-support/scalar-exports/scalar-vectors-v1.json"
        ))
        .unwrap();
        let source = include_bytes!("../../test-support/scalar-exports/scalar_exports.inv");
        let program = checked(source);
        let cases = cases["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 59);
        for case in cases {
            let args = case["arguments"]
                .as_array()
                .unwrap()
                .iter()
                .map(|arg| match arg["type"].as_str().unwrap() {
                    "i32" => Value::I32(i32::try_from(arg["value"].as_i64().unwrap()).unwrap()),
                    "u32" => Value::U32(u32::try_from(arg["value"].as_u64().unwrap()).unwrap()),
                    _ => panic!("invalid scalar vector type"),
                })
                .collect::<Vec<_>>();
            let run = observe_named(
                &program,
                case["function"].as_str().unwrap(),
                &args,
                case["fuel"].as_u64().unwrap(),
            );
            let actual = match run.result {
                Ok(Value::I32(value)) => {
                    json!({"status":"ok","result":{"type":"i32","value":value}})
                }
                Ok(Value::U32(value)) => {
                    json!({"status":"ok","result":{"type":"u32","value":value}})
                }
                Err(error) => {
                    let document = error.to_json("scalar_exports.inv");
                    match error {
                        EvalError::Trap { .. } => json!({"status":"trap","trap":document["trap"]}),
                        EvalError::EvaluationLimit {
                            fuel_limit,
                            steps_used,
                        } => {
                            json!({"status":"evaluation_limit","fuel_limit":fuel_limit,"steps_used":steps_used})
                        }
                        other => panic!("{}: {other:?}", case["id"]),
                    }
                }
                other => panic!("{}: {other:?}", case["id"]),
            };
            assert_eq!(actual, case["expected"], "{}", case["id"]);
            assert!(run.cleanup_trace.is_empty(), "{}", case["id"]);
        }
        eprintln!(
            "59 direct named reference vectors passed with original UTF-8 spans and exact scalar returns"
        );
    }

    #[test]
    fn issue_l1_06_scalar_export_corpus_named_fuel_has_no_wrapper_charge() {
        let program = checked(
            b"fn constant() -> i32 { return 42; } fn add(x: i32, y: i32) -> i32 { return x + y; }",
        );
        for (name, args, steps) in [
            ("constant", vec![], 2),
            ("add", vec![Value::I32(20), Value::I32(22)], 4),
        ] {
            let exhausted = observe_named(&program, name, &args, steps - 1);
            assert_eq!(
                exhausted.result,
                Err(EvalError::EvaluationLimit {
                    fuel_limit: steps - 1,
                    steps_used: steps - 1
                })
            );
            let completed = observe_named(&program, name, &args, steps);
            assert_eq!(completed.result, Ok(Value::I32(42)));
            assert_eq!(completed.steps_used, steps);
        }
    }

    #[test]
    fn issue_l1_06_scalar_export_corpus_named_refusals_use_zero_source_steps() {
        let program = checked(b"fn spin(x: i32) -> i32 { while true {} return x; } fn unsigned(x: u32) -> u32 { return x; } fn unit() -> Unit { return; } fn boolean(x: Bool) -> i32 { return 0; } fn nine(a: i32,b: i32,c: i32,d: i32,e: i32,f: i32,g: i32,h: i32,i: i32) -> i32 { return 0; }");
        for (name, args, fuel, code) in [
            ("spin", vec![Value::I32(0)], 0, "INVALID_FUEL"),
            ("missing", vec![], 10, "NATIVE_ABI"),
            ("spin", vec![], 10, "NATIVE_ABI"),
            ("spin", vec![Value::I32(0), Value::I32(0)], 10, "NATIVE_ABI"),
            ("spin", vec![Value::U32(0)], 10, "NATIVE_ABI"),
            ("spin", vec![Value::Bool(false)], 10, "NATIVE_ABI"),
            ("unsigned", vec![Value::I32(0)], 10, "NATIVE_ABI"),
            ("unit", vec![], 10, "NATIVE_ABI"),
            ("boolean", vec![Value::Bool(false)], 10, "NATIVE_ABI"),
            (
                "nine",
                (0..9).map(|_| Value::I32(0)).collect(),
                10,
                "NATIVE_ABI",
            ),
        ] {
            let run = observe_named(&program, name, &args, fuel);
            assert!(
                matches!(run.result, Err(EvalError::Tool { code: actual, .. }) if actual == code),
                "{name}: {:?}",
                run.result
            );
            assert_eq!(run.steps_used, 0, "{name}");
            assert!(run.cleanup_trace.is_empty());
        }
    }

    #[test]
    fn issue_l1_06_scalar_export_corpus_named_recursion_limit_preserves_next_call() {
        let program = checked(
            b"fn descend(n: u32) -> i32 { if n == 0u32 { return 7; } return descend(n - 1u32); }",
        );
        let exhausted = observe_named(&program, "descend", &[Value::U32(300)], 100_000);
        assert!(matches!(
            exhausted.result,
            Err(EvalError::Tool {
                code: "CALL_DEPTH_LIMIT",
                ..
            })
        ));
        assert!(exhausted.steps_used > 0);
        let completed = observe_named(&program, "descend", &[Value::U32(3)], 100_000);
        assert_eq!(completed.result, Ok(Value::I32(7)));
    }
}
