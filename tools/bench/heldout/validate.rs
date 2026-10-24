//! Oracle validation of generated held-out sources.

use super::*;

/// Validate a `tc-f16-f32-v1` program: it is ordinary checked source, its entry is the recognized explicit
/// Tensor Core contraction, the strict twin passes the source reference
/// oracle, and the exact expectation satisfies the amended D5 bound under the
/// device module's own checker.
pub fn validate_tc(source: &str, twin: &str, entry: &str, shape: &Value) -> Result<(), String> {
    let contract = contraction_source::Contract::from_source(source.as_bytes(), entry)
        .map_err(|e| format!("tc recognizer: {e:?}"))?;
    if contract.policy().id() != "tc-f16-f32-v1" {
        return Err("tc program recognized under another numeric policy".into());
    }
    // Ordinary checking admits explicit `tc_mul_add` source on main, so a
    // held-out tc program must also be an ordinary checked program.
    stored_oracle(source).map_err(|e| format!("stored tc program: {e}"))?;
    validate_source("matrix", twin).map_err(|e| format!("strict twin: {e}"))?;
    let dims: Vec<u32> = shape["dimensions"]
        .as_array()
        .ok_or("tc dimensions missing")?
        .iter()
        .map(|d| d.as_u64().map(|d| d as u32).ok_or("tc dimension"))
        .collect::<Result<_, _>>()?;
    let words = |key: &str| -> Result<Vec<u32>, String> {
        shape[key]
            .as_array()
            .ok_or("tc inputs missing")?
            .iter()
            .map(|v| {
                v.as_f64()
                    .map(|v| (v as f32).to_bits())
                    .ok_or_else(|| "tc input".to_string())
            })
            .collect()
    };
    let (a, b) = (words("a")?, words("b")?);
    let dims = [dims[0], dims[1], dims[2]];
    let reference =
        invar_device::tc::reference(dims, &a, &b).map_err(|e| format!("tc reference: {e:?}"))?;
    let [rows, inner, cols] = dims.map(|d| d as usize);
    let exact: Vec<u32> = (0..rows * cols)
        .map(|cell| {
            let (r, c) = (cell / cols, cell % cols);
            (0..inner)
                .map(|k| {
                    f64::from(f32::from_bits(a[r * inner + k]))
                        * f64::from(f32::from_bits(b[k * cols + c]))
                })
                .fold(0.0, |s, x| s + x) as f32
        })
        .map(f32::to_bits)
        .collect();
    if reference != exact {
        return Err("tc exact-sum reference differs from the independent host sum".into());
    }
    invar_device::tc::verify(dims, &a, &b, &reference).map_err(|e| format!("tc bound: {e:?}"))?;
    Ok(())
}

pub fn validate_source(family: &str, source: &str) -> Result<(), String> {
    let checked = check::check(
        front::parse(source.as_bytes()).map_err(|e| format!("parse: {e:?}"))?,
        true,
    )
    .map_err(|e| format!("check: {e:?}"))?;
    let compiled = mir::lower(&checked).map_err(|e| format!("MIR: {e:?}"))?;
    mir::verify(&compiled).map_err(|e| format!("MIR verify: {e:?}"))?;
    let entry = checked.program.functions[0].name.as_str();
    let pair = (family == "reduction")
        .then(|| invar::pair_reduction::extract(&compiled).ok())
        .flatten();
    match family {
        "reduction" if pair.is_some() => {
            let a = invar::admission::analyze(source.as_bytes(), &compiled)
                .map_err(|e| format!("admission: {e:?}"))?;
            if !a
                .regions()
                .iter()
                .any(|r| matches!(r.execution(), invar::plan::Execution::PairReduction(_)))
            {
                return Err("pair reduction extracted but not admitted".into());
            }
        }
        "map" | "reduction" | "scan" => {
            let a = invar::admission::analyze(source.as_bytes(), &compiled)
                .map_err(|e| format!("admission: {e:?}"))?;
            let kinds = match family {
                "map" => vec![invar::plan::PlanKind::Map, invar::plan::PlanKind::FusedMap],
                "reduction" => vec![invar::plan::PlanKind::Reduction],
                _ => vec![invar::plan::PlanKind::Scan],
            };
            // The separately evaluated oracle helper is not a workload kernel.
            let workload: Vec<_> = a
                .regions()
                .iter()
                .filter(|r| r.function() == "main")
                .collect();
            if workload.is_empty() || !workload.iter().all(|r| kinds.contains(&r.plan_kind())) {
                return Err("unexpected legacy family plan".into());
            }
        }
        "transpose" => {
            invar::transpose_source::Contract::from_checked(&checked, entry)
                .map_err(|e| format!("recognizer: {e:?}"))?;
        }
        "stencil" => {
            invar::stencil_source::Contract::from_checked(&checked, entry)
                .map_err(|e| format!("recognizer: {e:?}"))?;
        }
        "matrix" => {
            invar::tensor::Contraction::from_checked(&checked, entry)
                .map(|_| ())
                .or_else(|scalar| {
                    // M4-03's direct-cell strict form, `c[i,j] = c[i,j] + a*b`.
                    contraction_source::Contract::from_source(source.as_bytes(), entry)
                        .map_err(|cell| {
                            format!("recognizer: {scalar:?}; direct-cell recognizer: {cell:?}")
                        })
                        .and_then(|c| {
                            (c.policy().id() == "strict-f32-v1")
                                .then_some(())
                                .ok_or_else(|| "direct-cell form under another policy".to_string())
                        })
                })?;
        }
        "compaction" => {
            invar::compaction_source::Contract::from_source(source.as_bytes(), entry)
                .map_err(|e| format!("recognizer: {e:?}"))?;
        }
        "segmented" => {
            invar::segmented_source::Contract::from_source(source.as_bytes(), entry)
                .map(|_| ())
                .or_else(|reduce| {
                    invar::segmented_device::MapContract::from_source(source.as_bytes(), entry)
                        .map(|_| ())
                        .map_err(|map| {
                            format!("reduce recognizer: {reduce:?}; map recognizer: {map:?}")
                        })
                })?;
        }
        "gather_scatter" => {
            invar::indexed_source::Contract::from_checked(source.as_bytes(), &checked, entry)
                .map(|_| ())
                .or_else(|indexed_error| {
                    invar::histogram_source::Contract::from_checked(source.as_bytes(), &checked, entry)
                        .map(|_| ())
                        .map_err(|histogram_error| format!("indexed recognizer: {indexed_error:?}; histogram recognizer: {histogram_error:?}"))
                })?;
        }
        "frontier" => {
            frontier_source::Contract::from_checked(source.as_bytes(), &checked, entry)
                .map_err(|e| format!("recognizer: {e:?}"))?;
        }
        _ => return Err("unknown family".into()),
    }
    observe_oracle(&checked, &compiled)
}

/// Check, lower and verify a stored program, then run its own oracle.
pub fn stored_oracle(source: &str) -> Result<(), String> {
    let checked = check::check(
        front::parse(source.as_bytes()).map_err(|e| format!("parse: {e:?}"))?,
        true,
    )
    .map_err(|e| format!("check: {e:?}"))?;
    let compiled = mir::lower(&checked).map_err(|e| format!("MIR: {e:?}"))?;
    mir::verify(&compiled).map_err(|e| format!("MIR verify: {e:?}"))?;
    observe_oracle(&checked, &compiled)
}

/// The stored program's own oracle: `main` returns 0 on the reference
/// evaluator, and the MIR evaluator agrees on result, fuel and cleanup.
fn observe_oracle(checked: &check::CheckedProgram, compiled: &mir::Program) -> Result<(), String> {
    let reference = eval::observe(checked, FUEL);
    if reference.result != Ok(eval::Value::I32(0)) {
        return Err(format!("reference oracle: {:?}", reference.result));
    }
    let lowered = mir_eval::observe(compiled, FUEL);
    if reference.result != lowered.result
        || reference.steps_used != lowered.steps_used
        || reference.cleanup_trace != lowered.cleanup_trace
    {
        return Err("reference/MIR observation mismatch".into());
    }
    Ok(())
}
