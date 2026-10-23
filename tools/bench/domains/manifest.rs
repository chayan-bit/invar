//! The domain-suite manifest proposal (`bench/manifest-v3-proposal.json`) and
//! its validator. It is a separate document from the frozen-candidate v1 and
//! v2 manifests: the domain suites are a later set and are never merged into
//! the M5-01 held-out hash (#52). The proposal records no freeze.
use super::{Program, SEED, SUITES, held_out_kernels, programs};
use crate::{hash::sha256, heldout};
use serde_json::{Value, json};

fn kernel_names(list: &[Program]) -> Vec<&'static str> {
    let mut names: Vec<&'static str> = list.iter().map(|p| p.kernel).collect();
    names.sort_unstable();
    names.dedup();
    names
}

/// Every program of `suite` with its split: all cases of a kernel share it.
pub fn split(suite: &'static str) -> Result<Vec<(Program, bool)>, String> {
    let list = programs(suite)?;
    let held = held_out_kernels(suite, &kernel_names(&list));
    Ok(list
        .into_iter()
        .map(|p| {
            let held_out = held.contains(&p.kernel);
            (p, held_out)
        })
        .collect())
}

fn entry(program: &Program, held_out: bool) -> Value {
    let shape: serde_json::Map<String, Value> = program
        .shape
        .iter()
        .map(|(name, value)| ((*name).to_owned(), json!(value)))
        .collect();
    json!({
        "id": program.id,
        "kernel": program.kernel,
        "split": if held_out { "held_out" } else { "tuning" },
        "shape": shape,
        "inputs": program.inputs.iter().map(|(len, seed)| json!({"len": len, "seed": seed})).collect::<Vec<_>>(),
        "source_sha256": sha256(program.source.as_bytes()),
        "canonical_sha256": heldout::canonical_program(&program.source),
    })
}

fn held_out_hash(entries: &[Value]) -> Result<String, String> {
    let held: Vec<&Value> = entries
        .iter()
        .filter(|e| e["split"] == "held_out")
        .collect();
    serde_json::to_vec(&held)
        .map(|bytes| sha256(&bytes))
        .map_err(|e| e.to_string())
}

fn suite_section(suite: &'static str) -> Result<Value, String> {
    let entries: Vec<Value> = split(suite)?
        .iter()
        .map(|(program, held_out)| entry(program, *held_out))
        .collect();
    let count = |split: &str| entries.iter().filter(|e| e["split"] == split).count();
    Ok(json!({
        "seed": SEED,
        "program_counts": {"total": entries.len(), "held_out": count("held_out"), "tuning": count("tuning")},
        "held_out_sha256": held_out_hash(&entries)?,
        "programs": entries,
    }))
}

/// The proposal regenerated from the suites. Generation hashes every program,
/// so it is computed once per process.
pub fn build() -> Result<Value, String> {
    static BUILT: std::sync::OnceLock<Result<Value, String>> = std::sync::OnceLock::new();
    BUILT.get_or_init(generate).clone()
}

fn generate() -> Result<Value, String> {
    let v1: Value = serde_json::from_slice(include_bytes!("../manifest-v1.json"))
        .map_err(|e| format!("manifest-v1.json: {e}"))?;
    let mut suites = serde_json::Map::new();
    for suite in SUITES {
        suites.insert(suite.to_owned(), suite_section(suite)?);
    }
    Ok(json!({
        "schema_version": 1,
        "status": "proposal-pending-review",
        "generator": "domain-suites-v1",
        "freeze": "not frozen: the permanent held-out freeze is gated by #52 and needs maintainer confirmation",
        "split_policy": "seeded 25% of each suite's kernels, rounded up, all cases of a kernel on one side; independent of every result",
        "v1_held_out_sha256": v1["held_out_sha256"],
        "v1_held_out_sha256_note": "unchanged: the domain suites never enter the M5-01 held-out hash",
        "suites": suites,
        "pending": [{
            "program": "ml-quantized_matmul",
            "reason": "i8 storage with i32 accumulation needs the reduced-precision storage types of ENG-03; not approximated with wider types",
        }],
        "baselines": {"owner": "#109 (ENG-26)", "status": "pending", "reported_separately": true},
        "gpu_performance": "not_run",
    }))
}

/// Refuses a manifest with a missing, duplicate or empty case set, an altered
/// seed, an edited held-out entry or a hash that does not match its entries.
pub fn validate(manifest: &Value) -> Result<(), String> {
    if manifest["status"] != "proposal-pending-review" || manifest.get("freeze_status").is_some() {
        return Err("a domain manifest proposal must stay unfrozen".into());
    }
    let expected = build()?;
    for suite in SUITES {
        let section = &manifest["suites"][suite];
        let entries = section["programs"]
            .as_array()
            .ok_or_else(|| format!("{suite}: programs missing"))?;
        if entries.is_empty() {
            return Err(format!("{suite}: zero cases"));
        }
        if section["seed"] != json!(SEED) {
            return Err(format!("{suite}: altered seed"));
        }
        let mut ids = std::collections::BTreeSet::new();
        for e in entries {
            let id = e["id"]
                .as_str()
                .ok_or_else(|| format!("{suite}: entry without id"))?;
            if !ids.insert(id) {
                return Err(format!("{suite}: duplicate case {id}"));
            }
        }
        for e in expected["suites"][suite]["programs"]
            .as_array()
            .into_iter()
            .flatten()
        {
            if !ids.contains(e["id"].as_str().unwrap_or("")) {
                return Err(format!("{suite}: missing case {}", e["id"]));
            }
        }
        if section["held_out_sha256"] != json!(held_out_hash(entries)?) {
            return Err(format!("{suite}: held-out hash does not match its entries"));
        }
    }
    if manifest != &expected {
        return Err(
            "manifest differs from the regenerated suites (edited entry, seed or hash)".into(),
        );
    }
    Ok(())
}

/// The only program view a tuning path may use: the tuning kernels, each
/// admitted against the held-out index of its suite (`HeldOutIndex::admit`).
pub fn tuning_programs(
    suite: &'static str,
) -> Result<(heldout::HeldOutIndex, Vec<Program>), String> {
    let all = split(suite)?;
    let sources: Vec<Value> = all
        .iter()
        .map(|(p, held_out)| json!({"split": if *held_out { "held_out" } else { "tuning" }, "source": p.source}))
        .collect();
    let index = heldout::HeldOutIndex::from_programs(&Value::Array(sources))?;
    let mut tuning = Vec::new();
    for (program, held_out) in all {
        if held_out {
            continue;
        }
        index
            .admit(&program.source)
            .map_err(|e| format!("{}: {e}", program.id))?;
        tuning.push(program);
    }
    Ok((index, tuning))
}
