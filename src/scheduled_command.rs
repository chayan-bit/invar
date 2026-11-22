//! `invar run --engine cpu-parallel|cuda|metal|auto`: one scheduled run and its
//! receipt, with the `--check-workers` invariance receipt (#209).

use std::ffi::OsString;

use invar::{eval::EvalError, front::FrontError};

use crate::{auto_command, diagnostic_source, report_eval, report_front};

#[allow(clippy::too_many_arguments)] // CLI flags map one-to-one onto the shared engine call.
pub(crate) fn execute(
    source: &[u8],
    checked: &invar::check::CheckedProgram,
    path: &OsString,
    json: bool,
    fuel: u64,
    engine: invar::engine::Engine,
    workers: usize,
    artifact: Option<&OsString>,
    check_workers: Option<&[usize]>,
) -> i32 {
    let program = match invar::mir::lower(checked) {
        Ok(program) => program,
        Err(error) => return report_front(FrontError::tool(error.code, error.message), path, json),
    };
    let invariance = match check_workers
        .map(|counts| invar::invariance::check_workers(checked, &program, source, fuel, counts))
        .transpose()
    {
        Ok(invariance) => invariance,
        Err(error) => return report_eval(error, path, json),
    };
    let image = match invar::engine::metal_image(artifact.map(std::path::Path::new)) {
        Ok(image) => image,
        Err(error) => return auto_command::report(error, path, json, engine),
    };
    match invar::engine::run_mir(source, &program, engine, fuel, workers, image.as_ref()) {
        Ok(receipt) => {
            if json {
                let mut diagnostics = Vec::new();
                if let Some(reason) = receipt["fallback_reason"].as_str() {
                    let mut note = invar::diagnostics::diagnostic(
                        reason,
                        "CPU execution used the serial fallback",
                        &path.to_string_lossy(),
                        source,
                        None,
                    );
                    note["severity"] = serde_json::json!("note");
                    diagnostics.push(note);
                }
                let mut document = serde_json::json!({
                    "schema_version":3, "status":"ok", "diagnostics":diagnostics,
                    "result":{"type":"i32", "value":receipt["result"]},
                });
                if let Some(route) = receipt.get("normal_form") {
                    document["normal_form"] = route.clone();
                }
                if let Some(invariance) = invariance {
                    document["invariance"] = invariance;
                }
                return auto_command::receipt(document, path, engine);
            }
            println!("Result: {}", receipt["result"]);
            print_invariance(invariance.as_ref());
            0
        }
        Err(error) => match invariance {
            // A trap or fuel limit every worker count reproduced is still checked evidence.
            Some(invariance) if json => {
                let mut document =
                    error.to_diagnostics_json(&path.to_string_lossy(), &diagnostic_source(path));
                document["invariance"] = invariance;
                println!("{document}");
                match error {
                    EvalError::Trap { .. } => 3,
                    EvalError::EvaluationLimit { .. } => 4,
                    EvalError::Tool { .. } => 2,
                }
            }
            invariance => {
                let status = auto_command::report(error, path, json, engine);
                print_invariance(invariance.as_ref());
                status
            }
        },
    }
}

/// The text form of a checked `invariance` receipt.
fn print_invariance(invariance: Option<&serde_json::Value>) {
    if let Some(invariance) = invariance {
        let workers: Vec<_> = invariance["workers"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|count| count.to_string())
            .collect();
        println!(
            "Invariance: result, trap and fuel identical on the reference evaluator and on cpu-parallel at workers {}",
            workers.join(", ")
        );
    }
}
