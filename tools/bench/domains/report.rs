//! The domain-suite coverage report (#56): per-program results, the worst
//! five held-out programs and how many programs reached a parallel lowering on
//! the CPU, with the engine's own refusal reasons. It runs every program on
//! the reference evaluator and on `cpu-parallel`; it times nothing, because
//! timings from a shared host are not evidence, and device runs and baseline
//! comparisons are reported pending until real hardware runs them.
use super::{Program, SUITES, manifest};
use invar::{
    engine::{self, Engine},
    eval::{self, Value as EvalValue},
};
use serde_json::{Value, json};
use std::collections::BTreeMap;

const FUEL: u64 = 400_000_000;
/// One worker count above 1 and below the 64-worker bound: enough to reach a
/// parallel lowering when the engine admits one.
const WORKERS: usize = 7;
const WORST: usize = 5;

fn reasons(receipt: &Value) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(text) = receipt["fallback_reason"].as_str() {
        out.push(format!("fallback: {text}"));
    }
    if receipt["normal_form"]["status"] == "fallback" {
        out.push(format!(
            "no lowering: {}",
            receipt["normal_form"]["reason"]
                .as_str()
                .unwrap_or("unspecified")
        ));
    }
    for choice in receipt["choices"].as_array().into_iter().flatten() {
        if choice["backend"] != "cpu-parallel" {
            out.push(format!("{}: {}", choice["backend"], choice["reason"]));
        }
    }
    out.sort();
    out.dedup();
    out
}

fn row(program: &Program, held_out: bool) -> Result<Value, String> {
    let checked = invar::front::parse(program.source.as_bytes())
        .and_then(|parsed| invar::check::check(parsed, true))
        .map_err(|e| format!("{}: {e:?}", program.id))?;
    let reference = match eval::run(&checked, FUEL)
        .map_err(|e| format!("{}: {e:?}", program.id))?
        .value
    {
        EvalValue::I32(value) => value,
        other => return Err(format!("{}: main returned {other:?}", program.id)),
    };
    let mir = invar::mir::lower(&checked).map_err(|e| format!("{}: {e:?}", program.id))?;
    let receipt = engine::run_mir(
        program.source.as_bytes(),
        &mir,
        Engine::CpuParallel,
        FUEL,
        WORKERS,
        None,
    )
    .map_err(|e| format!("{}: {e:?}", program.id))?;
    let backends: Vec<&str> = receipt["backends"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    let regions = receipt["choices"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|choice| choice["backend"] == "cpu-parallel")
        .count();
    let parallel = regions > 0 || backends.contains(&"normal-form");
    Ok(json!({
        "id": program.id,
        "kernel": program.kernel,
        "split": if held_out { "held_out" } else { "tuning" },
        "reference_result": reference,
        "cpu_parallel_result": receipt["result"],
        "backends": backends,
        "parallel_regions": regions,
        "lowering": if parallel { "parallel" } else { "cpu-serial" },
        "refusal_reasons": if parallel { Vec::new() } else { reasons(&receipt) },
    }))
}

fn count(rows: &[Value], key: &str, want: &str) -> usize {
    rows.iter().filter(|r| r[key] == want).count()
}

fn suite_report(suite: &'static str) -> Result<Value, String> {
    let rows = manifest::split(suite)?
        .iter()
        .map(|(program, held_out)| row(program, *held_out))
        .collect::<Result<Vec<_>, _>>()?;
    let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
    for r in rows.iter().filter(|r| r["lowering"] == "cpu-serial") {
        for reason in r["refusal_reasons"].as_array().into_iter().flatten() {
            *reasons
                .entry(reason.as_str().unwrap_or("").to_owned())
                .or_default() += 1;
        }
    }
    // Worst first: no parallel lowering, then the fewest parallel regions.
    let mut held: Vec<&Value> = rows.iter().filter(|r| r["split"] == "held_out").collect();
    held.sort_by_key(|r| {
        (
            r["parallel_regions"].as_u64(),
            r["id"].as_str().map(str::to_owned),
        )
    });
    let correct = rows
        .iter()
        .filter(|r| r["reference_result"] == 0 && r["cpu_parallel_result"] == 0)
        .count();
    Ok(json!({
        "suite": suite,
        "programs": rows.len(),
        "correct_on_reference_and_cpu_parallel": correct,
        "coverage": {
            "parallel": count(&rows, "lowering", "parallel"),
            "cpu_serial": count(&rows, "lowering", "cpu-serial"),
            "refusal_reasons": reasons,
        },
        "worst_held_out": held.iter().take(WORST).collect::<Vec<_>>(),
        "results": rows,
    }))
}

/// The report for one suite, or for all of them (`suite` = "all").
pub fn report(suite: &str) -> Result<Value, String> {
    let wanted: Vec<&'static str> =
        match suite {
            "all" => SUITES.to_vec(),
            other => vec![*SUITES.iter().find(|s| **s == other).ok_or_else(|| {
                format!("unknown suite {other}; choose one of {SUITES:?} or all")
            })?],
        };
    let mut suites = serde_json::Map::new();
    for name in wanted {
        suites.insert(name.to_owned(), suite_report(name)?);
    }
    Ok(json!({
        "schema_version": 1,
        "engine": "cpu-parallel",
        "workers": WORKERS,
        "ranking": "worst held-out: fewest parallel regions first, then id; nothing is timed",
        "device_timings": "pending: not run on real hardware, never estimated",
        "baselines": "pending: owned by #109 (ENG-26), reported separately",
        "suites": suites,
    }))
}
