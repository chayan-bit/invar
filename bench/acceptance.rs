//! M5-03 offline report arithmetic. This module never executes a benchmark.
use super::hash::sha256;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

const FAMILIES: [&str; 17] = [
    "map",
    "reduction",
    "scan",
    "compaction",
    "transpose",
    "stencil",
    "matrix",
    "segmented",
    "gather_scatter",
    "frontier",
    "summaries",
    "general_dynamic",
    "kinematics_dynamics",
    "collision_geometry",
    "planning",
    "perception",
    "composed_robotics",
];

fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value[key]
        .as_str()
        .filter(|s| {
            !s.trim().is_empty()
                && ![
                    "unavailable",
                    "unknown",
                    "unrecorded",
                    "pending",
                    "not_recorded",
                    "not_run",
                    "provisional",
                ]
                .contains(&s.trim().to_ascii_lowercase().as_str())
        })
        .ok_or_else(|| format!("missing {key}"))
}
fn number(value: &Value, key: &str) -> Result<f64, String> {
    value[key]
        .as_f64()
        .filter(|n| n.is_finite() && *n >= 0.0)
        .ok_or_else(|| format!("invalid {key}"))
}
fn positive(value: &Value, key: &str) -> Result<f64, String> {
    let n = number(value, key)?;
    if n == 0.0 {
        return Err(format!("zero {key}"));
    }
    Ok(n)
}
fn samples(value: &Value) -> Result<Vec<f64>, String> {
    let values = value.as_array().ok_or("raw samples missing")?;
    if !(30..=1000).contains(&values.len()) {
        return Err("sample count outside 30..1000".into());
    }
    values
        .iter()
        .map(|v| {
            v.as_f64()
                .filter(|n| n.is_finite() && *n > 0.0)
                .ok_or_else(|| "nonpositive/nonfinite sample".into())
        })
        .collect()
}
fn quantile(values: &[f64], q: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let position = (sorted.len() - 1) as f64 * q;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    sorted[low] * (1.0 - (position - low as f64)) + sorted[high] * (position - low as f64)
}
fn summary(values: &[f64]) -> Value {
    json!({"median":quantile(values,0.5),"q1":quantile(values,0.25),"q3":quantile(values,0.75),"iqr":quantile(values,0.75)-quantile(values,0.25)})
}
fn timing(report: &Value, baseline: bool, scope: &str) -> Result<Vec<f64>, String> {
    let result = &report[if baseline {
        "baseline_result"
    } else {
        "generated_result"
    }];
    if scope == "cold_process" {
        if report["cold_process"]["status"] != "passed"
            || report["cold_process"]["contract_review"] != "accepted"
        {
            return Err("cold process scope unreviewed/unavailable".into());
        }
        return samples(&report["cold_process"]["samples_ms"]);
    }
    let key = match (baseline, scope) {
        (false, "device_resident") => "device_resident_ms",
        (false, "host_to_host") => "host_to_host_ms",
        (true, "device_resident") => "device_resident_samples_ms",
        (true, "host_to_host") => "host_to_host_samples_ms",
        _ => return Err("unknown timing scope".into()),
    };
    samples(&result[key])
}

// Resample independent process sessions, never correlated launches as independent runs.
// ponytail: bounded 4096 session-cluster draws; change only through a reviewed plan version.
fn ratio_statistics(pairs: &[(Vec<f64>, Vec<f64>)], seed: u64) -> Value {
    let a: Vec<_> = pairs.iter().flat_map(|p| p.0.iter().copied()).collect();
    let b: Vec<_> = pairs.iter().flat_map(|p| p.1.iter().copied()).collect();
    let ratio = quantile(&a, 0.5) / quantile(&b, 0.5);
    let mut state = seed;
    let mut draws = Vec::with_capacity(4096);
    for _ in 0..4096 {
        let mut aa = Vec::new();
        let mut bb = Vec::new();
        for _ in pairs {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let index = (state % pairs.len() as u64) as usize;
            aa.extend_from_slice(&pairs[index].0);
            bb.extend_from_slice(&pairs[index].1);
        }
        draws.push(quantile(&aa, 0.5) / quantile(&bb, 0.5));
    }
    json!({"ratio":ratio,"upper95":quantile(&draws,0.95),"generated_ms":summary(&a),"baseline_ms":summary(&b),"independent_sessions":pairs.len(),"method":"paired-session-cluster-bootstrap-v1","draws":4096,"seed":seed})
}

/// Foreign GPU activity that the timing wrapper recorded during the timed
/// phase: (max foreign utilization percent, foreign process samples). A packet
/// without the record cannot support any timing, shared or exclusive.
fn foreign_activity(report: &Value) -> Result<(f64, u64), String> {
    let activity = &report["environment"]["gpu_foreign_activity"];
    if activity["record"].is_null() {
        return Err("foreign GPU activity record missing".into());
    }
    let util = number(activity, "max_foreign_util_percent")?;
    if !(0.0..=100.0).contains(&util) {
        return Err("foreign GPU utilization out of range".into());
    }
    let processes = activity["foreign_processes"]
        .as_u64()
        .ok_or("foreign GPU process count missing")?;
    Ok((util, processes))
}

/// Measurement conditions of one result row over every timed packet. Only a
/// row whose every record shows no foreign activity is called exclusive.
fn gpu_conditions(row: &Value) -> Value {
    let mut packets = Vec::new();
    for session in row["sessions"].as_array().into_iter().flatten() {
        packets.push(&session["generated"]);
        for baseline in session["baselines"].as_array().into_iter().flatten() {
            packets.push(&baseline["report"]);
        }
    }
    let (mut max, mut processes, mut records) = (0.0_f64, 0_u64, Vec::new());
    for packet in &packets {
        match foreign_activity(packet) {
            Ok((util, count)) => {
                max = max.max(util);
                processes += count;
                records.push(packet["environment"]["gpu_foreign_activity"].clone());
            }
            Err(reason) => return json!({"label":"unrecorded","reason":reason}),
        }
    }
    if packets.is_empty() {
        return json!({"label":"unrecorded","reason":"no timed packets"});
    }
    let label = if max == 0.0 && processes == 0 {
        "exclusive GPU (no foreign activity recorded)".to_owned()
    } else {
        format!("shared GPU, max foreign util {max}%")
    };
    json!({"label":label,"max_foreign_util_percent":max,"foreign_process_samples":processes,"records":records})
}

fn lineage(
    report: &Value,
    manifest: &Value,
    digest: &str,
    target: &Value,
    baseline: bool,
) -> Result<(), String> {
    if report["manifest_sha256"] != digest
        || report["corpus_sha256"] != manifest["corpus_sha256"]
        || report["freeze_status"] != "frozen"
    {
        return Err("stale digest or provisional result".into());
    }
    if report["checkout_at_invocation"]["worktree"] != "clean" {
        return Err("unclean/unavailable checkout".into());
    }
    text(&report["checkout_at_invocation"], "head")?;
    let r = &report[if baseline {
        "baseline_result"
    } else {
        "generated_result"
    }];
    if r["build_review_status"] != "accepted" || r["build_record"]["worktree_at_build"] != "clean" {
        return Err("build review missing".into());
    }
    for key in ["source_head", "rustc_version", "executable_sha256"] {
        text(&r["build_record"], key)?;
    }
    if r["build_record"]["source_head"] != report["checkout_at_invocation"]["head"] {
        return Err("build source mismatch".into());
    }
    if baseline {
        if r["gate_status"] != "accepted"
            || r["target_compatibility"] != "accepted"
            || r["full_output_correctness"] != "exact_full_output_pass"
            || r["quality_review"] != "accepted"
            || r["semantic_review"] != "accepted"
        {
            return Err("baseline correctness/quality/semantics pending".into());
        }
        for key in [
            "fixture_source_sha256",
            "executable_sha256",
            "baseline_revision",
        ] {
            text(r, key)?;
        }
    } else {
        if r["measurement_scope"] != "checked_source_program"
            || r["full_output_correct"] != true
            || r["qualification_status"] != "passed"
            || r["safety_status"] != "passed"
            || r["numeric_policy_status"] != "passed"
        {
            return Err("source qualification/safety/numerics unavailable".into());
        }
        for key in [
            "source_sha256",
            "generated_code_hash",
            "runner_executable_sha256",
        ] {
            text(r, key)?;
        }
    }
    if r["device"] != target["name"] {
        return Err("wrong device".into());
    }
    let environment = &report["environment"];
    foreign_activity(report)?;
    if environment["host"] != target["host"]
        || environment["device_id"] != target["device_id"]
        || environment["compute_capability"] != target["compute_capability"]
    {
        return Err("non-named host/device".into());
    }
    for key in ["driver", "toolchain", "compiler_flags", "stream_policy"] {
        if environment[key].is_null() || environment[key] != target[key] {
            return Err(format!("unfrozen target {key}"));
        }
    }
    for key in [
        "driver",
        "toolchain",
        "cpu",
        "os",
        "clock_policy",
        "power_policy",
        "occupancy",
        "temperature",
        "memory_preflight",
        "runtime_bound",
    ] {
        text(environment, key)?;
    }
    if r["variants"].as_u64().is_none_or(|n| n == 0) {
        return Err("variant count invalid".into());
    }
    if r["warmups"].as_u64().is_none_or(|n| n < 10) {
        return Err("insufficient warmups".into());
    }
    for key in [
        "seed",
        "transfer_bytes",
        "launches",
        "workspace_bytes",
        "compile_ms",
        "peak_compiler_bytes",
        "binary_bytes",
        "cache_bytes",
        "variants",
        "tuning_ms",
    ] {
        number(r, key)?;
    }
    Ok(())
}

fn consecutive(row: &Value, gate: &str, threshold: f64, minimum: bool) -> Result<Value, String> {
    let sessions = row["sessions"].as_array().ok_or("sessions missing")?;
    let mut observed = Vec::new();
    for session in sessions {
        let values =
            session["generated"]["generated_result"]["acceptance_metrics"]["consecutive_runs"]
                [gate]
                .as_array()
                .ok_or("near-threshold gate requires at least 10 consecutive runs")?;
        if !(10..=1000).contains(&values.len()) {
            return Err("invalid consecutive run count".into());
        }
        for value in values {
            observed.push(
                value
                    .as_f64()
                    .filter(|n| n.is_finite() && *n > 0.0)
                    .ok_or("invalid consecutive-run metric")?,
            );
        }
    }
    let failed = observed.iter().any(|n| {
        if minimum {
            *n < threshold
        } else {
            *n > threshold
        }
    });
    Ok(
        json!({"status":if failed {"failed"} else {"passed"},"minimum":observed.iter().copied().min_by(f64::total_cmp),"median":quantile(&observed,0.5),"runs":observed.len()}),
    )
}

fn evaluate(
    expected: &Value,
    row: &Value,
    manifest: &Value,
    digest: &str,
    target: &Value,
    seed: u64,
    baseline_pins: &Value,
) -> Result<Value, String> {
    if row["status"] != "measured" {
        return Err("unavailable/ignored/pending row".into());
    }
    let sessions = row["sessions"].as_array().ok_or("sessions missing")?;
    if !(3..=32).contains(&sessions.len()) {
        return Err("need 3..32 independent process sessions".into());
    }
    let scope = text(expected, "scope")?;
    let gate = text(expected, "gate")?;
    let mut ids = BTreeSet::new();
    let mut pairs = Vec::new();
    let mut metrics = Vec::new();
    let mut typed_reasons = Vec::new();
    let case = manifest["cases"]
        .as_array()
        .and_then(|v| v.iter().find(|c| c["id"] == expected["case_id"]))
        .ok_or("case outside manifest")?;
    for session in sessions {
        if !ids.insert(text(session, "id")?) {
            return Err("duplicate process session".into());
        }
        if session["measurement_protocol"] != "randomized-interleaved-paired-blocks" {
            return Err("unrecorded measurement ordering".into());
        }
        number(session, "ordering_seed")?;
        let g = &session["generated"];
        lineage(g, manifest, digest, target, false)?;
        let r = &g["generated_result"];
        if r["case_id"] != case["id"]
            || r["numeric_policy"] != case["numeric_policy"]
            || r["shape"] != case["shape"]
            || r["layout"] != case["layout"]
            || r["required_bytes"] != case["required_bytes"]
            || r["seed"] != case["seed"]
        {
            return Err("shape/layout/numerics/bytes mismatch".into());
        }
        for key in ["shape", "layout", "numeric_policy", "seed"] {
            if case[key].is_null() {
                return Err(format!("missing frozen {key}"));
            }
        }
        if row["track"] != "generated" {
            return Err("library-assisted is informational, never generated acceptance".into());
        }
        let raw = timing(g, false, scope)?;
        if gate == "dispatch" && r["acceptance_metrics"]["typed_reason"].is_object() {
            let reason = &r["acceptance_metrics"]["typed_reason"];
            let code = text(reason, "code")?;
            if case["performance_class"] != "placement_control"
                || reason["review"] != "accepted"
                || r["placement_status"] != "passed"
                || !expected["admitted_typed_reasons"]
                    .as_array()
                    .is_some_and(|v| v.iter().any(|c| c == code))
            {
                return Err("unreviewed/non-control dispatch exception".into());
            }
            text(reason, "evidence_sha256")?;
            typed_reasons.push(reason.clone());
            continue;
        }

        if ["parity", "library_ratio", "tc_library_ratio", "cold"].contains(&gate) {
            let baselines = session["baselines"].as_array().ok_or("missing baseline")?;
            let kind = if gate == "library_ratio" || gate == "tc_library_ratio" {
                "library"
            } else {
                "handwritten"
            };
            let mut eligible = Vec::new();
            for item in baselines.iter().filter(|b| b["kind"] == kind) {
                let b = &item["report"];
                lineage(b, manifest, digest, target, true)?;
                let br = &b["baseline_result"];
                let pin = baseline_pins.as_array().and_then(|pins| {
                    pins.iter().find(|pin| {
                        pin["case_id"] == case["id"]
                            && pin["target"] == target["name"]
                            && pin["kind"] == kind
                            && pin["source_sha256"] == br["fixture_source_sha256"]
                            && pin["revision"] == br["baseline_revision"]
                            && pin["executable_sha256"] == br["executable_sha256"]
                    })
                });
                if pin.is_none() {
                    return Err("baseline not pinned in frozen report plan".into());
                }
                if br["candidate_case_id"] != case["id"] {
                    return Err("baseline case mismatch".into());
                }
                for key in [
                    "input_sha256",
                    "input_recipe",
                    "numeric_policy",
                    "allocation_policy",
                    "device_resident_boundary",
                    "host_to_host_boundary",
                ] {
                    if r[key].is_null() || br[key] != r[key] {
                        return Err(format!("baseline {key} mismatch"));
                    }
                }
                if br["output_sha256"] != r["source_observable_output_sha256"]
                    || br["output_sha256"].is_null()
                {
                    return Err("baseline full output digest mismatch".into());
                }
                if scope == "cold_process"
                    && b["cold_process"]["boundary"] != g["cold_process"]["boundary"]
                {
                    return Err("cold boundary mismatch".into());
                }
                eligible.push(timing(b, true, scope)?);
            }
            let best = eligible
                .into_iter()
                .min_by(|a, b| quantile(a, 0.5).total_cmp(&quantile(b, 0.5)))
                .ok_or("missing matching baseline")?;
            pairs.push((raw, best));
        } else {
            let median = quantile(&raw, 0.5);
            let m = &r["acceptance_metrics"];
            let value = match gate {
                "bandwidth" => {
                    positive(case, "required_bytes")? / median / positive(m, "copy_bytes_per_ms")?
                }
                "compute" => {
                    positive(case, "required_operations")?
                        / median
                        / positive(m, "measured_peak_ops_per_ms")?
                }
                "dispatch" => {
                    quantile(&samples(&m["auto_ms"])?, 0.5)
                        / quantile(&samples(&m["forced_cpu_ms"])?, 0.5)
                            .min(quantile(&samples(&m["forced_gpu_ms"])?, 0.5))
                }
                "dispatch_small" => quantile(&samples(&m["dispatch_overhead_ms"])?, 0.5),
                "resources" => {
                    for (key, cap) in [
                        ("variants", 32.0),
                        ("workspace_bytes", positive(expected, "workspace_max")?),
                        ("compile_ms", positive(expected, "compile_ms_max")?),
                        ("tuning_ms", positive(expected, "tuning_ms_max")?),
                    ] {
                        if number(r, key)? > cap {
                            return Ok(
                                json!({"status":"failed","reason":format!("{key} budget exceeded")}),
                            );
                        }
                    }
                    1.0
                }
                "held_out_median" | "held_out_p10" => {
                    if m["held_out_sha256"] != manifest["held_out_sha256"]
                        || m["held_out_sha256"].is_null()
                        || m["worst_five_profiler"]
                            .as_array()
                            .is_none_or(|v| v.len() != 5)
                    {
                        return Err("held-out lineage/profiler missing".into());
                    }
                    quantile(
                        &samples(&m["held_out_bandwidth_fractions"])?,
                        if gate == "held_out_median" { 0.5 } else { 0.1 },
                    )
                }
                _ => return Err("unknown gate".into()),
            };
            metrics.push(value);
        }
    }
    if !pairs.is_empty() {
        let stats = ratio_statistics(&pairs, seed);
        let ratio = positive(&stats, "ratio")?;
        let upper = positive(&stats, "upper95")?;
        let max = match gate {
            "parity" => 1.0,
            "library_ratio" => 1.25,
            "tc_library_ratio" => 1.5,
            "cold" => {
                return Ok(json!({"status":"passed","scope":"reported_only","statistics":stats}));
            }
            _ => return Err("unknown ratio gate".into()),
        };
        let bound = if gate == "parity" { 1.05 } else { max };
        let followup = if (ratio - max).abs() <= 0.05 && ratio <= max && upper <= bound {
            Some(consecutive(row, gate, max, false)?)
        } else {
            None
        };
        let status = if ratio > max || followup.as_ref().is_some_and(|v| v["status"] == "failed") {
            "failed"
        } else if upper > bound {
            "pending"
        } else {
            "passed"
        };
        return Ok(
            json!({"status":status,"statistics":stats,"point_max":max,"upper_max":bound,"consecutive_runs":followup,"interpretation":if gate=="parity" && status=="passed" {"matched within the declared 5% uncertainty margin; synthetic inputs establish no claim"} else {"no parity claim"}}),
        );
    }
    if !typed_reasons.is_empty() {
        if typed_reasons.len() != sessions.len() {
            return Err("dispatch reason inconsistent across sessions".into());
        }
        return Ok(
            json!({"status":"exception","parity_credit":false,"typed_reasons":typed_reasons}),
        );
    }
    let floor = match case["family"].as_str() {
        Some("map") => 0.80,
        Some("reduction" | "transpose") => 0.75,
        Some("scan" | "gather_scatter") => 0.60,
        Some("stencil") => 0.70,
        Some("segmented" | "compaction") => 0.55,
        Some("frontier") => 0.30,
        Some("summaries") => 0.50,
        _ => 0.0,
    };
    if floor == 0.0 && ["bandwidth", "held_out_median", "held_out_p10"].contains(&gate) {
        return Err("family has no frozen bandwidth threshold".into());
    }
    let (threshold, minimum) = match gate {
        "bandwidth" => (floor, true),
        "compute" => (0.50, true),
        "held_out_median" => (floor - 0.05, true),
        "held_out_p10" => (floor - 0.15, true),
        "dispatch" => (1.10, false),
        "dispatch_small" => (0.100, false),
        "resources" => (1.0, true),
        _ => return Err("unknown metric gate".into()),
    };
    // Conservative session extrema: ambiguity cannot be promoted by averaging.
    let failed = metrics.iter().any(|n| {
        if minimum {
            *n < threshold
        } else {
            *n > threshold
        }
    });
    let followup = if [
        "bandwidth",
        "compute",
        "held_out_median",
        "held_out_p10",
        "dispatch",
    ]
    .contains(&gate)
        && !failed
        && metrics.iter().any(|n| (*n - threshold).abs() <= 0.05)
    {
        Some(consecutive(row, gate, threshold, minimum)?)
    } else {
        None
    };
    let failed = failed || followup.as_ref().is_some_and(|v| v["status"] == "failed");
    Ok(
        json!({"status":if failed {"failed"} else {"passed"},"consecutive_runs":followup,"session_metrics":metrics,"threshold":threshold,"minimum_gate":minimum,"uncertainty":"conservative session-extrema envelope of the session-cluster percentile bootstrap","lower_bound":metrics.iter().copied().min_by(f64::total_cmp),"upper_bound":metrics.iter().copied().max_by(f64::total_cmp)}),
    )
}

/// Consume immutable manifest bytes, a reviewed expected-row plan and saved bench packets.
/// Provisional or incomplete inputs produce a pending report, never a passing verdict.
pub fn report(manifest_raw: &[u8], plan: &Value, bundle: &Value) -> Result<Value, String> {
    let manifest: Value = serde_json::from_slice(manifest_raw).map_err(|e| e.to_string())?;
    let digest = sha256(manifest_raw);
    let mut reasons = Vec::new();
    if manifest["freeze_status"] != "frozen" {
        reasons.push("M5-01 manifest provisional");
    }
    if plan["status"] != "frozen" || plan["manifest_sha256"] != digest {
        reasons.push("report plan provisional or stale manifest digest");
    }
    if plan["statistics"] != "paired-session-cluster-bootstrap-v1" {
        reasons.push("statistical procedure not frozen");
    }
    if bundle["qualification"]["status"] != "passed"
        || bundle["qualification"]["manifest_sha256"] != digest
    {
        reasons.push("M5-02 qualification unavailable/stale");
    }
    for prerequisite in ["m4_06", "gnet", "baseline_review", "held_out_freeze"] {
        if bundle["prerequisites"][prerequisite] != "passed" {
            reasons.push(prerequisite);
        }
    }
    let seed = plan["bootstrap_seed"]
        .as_u64()
        .ok_or("missing bootstrap seed")?;
    let expected = plan["expected"]
        .as_array()
        .ok_or("expected row plan missing")?;
    if expected.len() > 20000 {
        return Err("too many expected rows".into());
    }
    let rows = bundle["rows"].as_array().ok_or("result rows missing")?;
    let targets = manifest["targets"]
        .as_array()
        .ok_or("manifest targets missing")?;
    if manifest["pending_families"]
        .as_array()
        .is_some_and(|v| !v.is_empty())
    {
        reasons.push("manifest families pending");
    }
    if bundle["qualification"]["qualified_count"]
        .as_u64()
        .is_none_or(|n| n == 0)
        || bundle["qualification"]["unavailable_count"] != 0
        || bundle["qualification"]["failed_count"] != 0
    {
        reasons.push("zero/nonqualified/unavailable M5-02 count");
    }
    if !["cuda", "metal"]
        .iter()
        .all(|backend| targets.iter().any(|t| t["backend"] == *backend))
    {
        reasons.push("CUDA and Metal backend identities not frozen");
    }
    if targets.len() < 2 {
        reasons.push("CUDA and Metal named target coverage missing");
    }
    let mut seen = BTreeSet::new();
    let mut coverage = BTreeSet::new();
    let mut output = Vec::new();
    let mut counts: BTreeMap<String, usize> = [
        "passed",
        "failed",
        "pending",
        "unavailable",
        "ignored",
        "exception",
    ]
    .into_iter()
    .map(|s| (s.into(), 0))
    .collect();
    for item in expected {
        let id = text(item, "id")?;
        if !seen.insert(id) {
            return Err("duplicate expected id".into());
        }
        let matching: Vec<_> = rows.iter().filter(|r| r["id"] == id).collect();
        if matching.len() > 1 {
            return Err("duplicate result id".into());
        }
        let target = targets.iter().find(|t| t["name"] == item["target"]);
        let case = manifest["cases"]
            .as_array()
            .and_then(|v| v.iter().find(|c| c["id"] == item["case_id"]));
        let status = matching
            .first()
            .map_or("unavailable", |r| r["status"].as_str().unwrap_or("pending"));
        let evaluated = match (matching.first(), target, case) {
            (Some(row), Some(target), Some(case)) => {
                coverage.insert((
                    item["target"].to_string(),
                    case["family"].to_string(),
                    item["gate"].to_string(),
                    item["scope"].to_string(),
                ));
                evaluate(
                    item,
                    row,
                    &manifest,
                    &digest,
                    target,
                    seed,
                    &plan["baselines"],
                )
            }
            _ => Err("missing report, named target or frozen case".into()),
        };
        let result = evaluated.unwrap_or_else(|reason|json!({"status":if status=="unavailable" || status=="ignored" || status=="failed" {status} else {"pending"},"reason":reason}));
        *counts
            .entry(result["status"].as_str().unwrap_or("pending").into())
            .or_default() += 1;
        output.push(json!({"id":id,"case_id":item["case_id"],"family":case.map(|c|&c["family"]),"target":item["target"],"track":matching.first().map(|r|&r["track"]),"gate":item["gate"],"scope":item["scope"],"gpu_conditions":matching.first().copied().map(gpu_conditions),"result":result}));
    }
    if rows
        .iter()
        .any(|r| r["id"].as_str().is_none_or(|id| !seen.contains(id)))
    {
        reasons.push("unexpected result rows retained; revise frozen plan, never drop results");
    }
    // All obligations remain in the denominator, even before the freeze owner supplies rows.
    let mut missing_family_obligations = 0usize;
    let mut missing_case_scopes = 0usize;
    for target in targets {
        for family in FAMILIES {
            for (gate, scope) in [
                ("parity", "device_resident"),
                ("parity", "host_to_host"),
                ("library_ratio", "device_resident"),
                ("cold", "cold_process"),
                ("dispatch", "host_to_host"),
                ("resources", "device_resident"),
            ] {
                if !coverage.contains(&(
                    target["name"].to_string(),
                    json!(family).to_string(),
                    json!(gate).to_string(),
                    json!(scope).to_string(),
                )) {
                    missing_family_obligations += 1;
                    reasons.push("required family/target/gate coverage missing");
                }
            }
        }
    }
    let cases = manifest["cases"].as_array().ok_or("cases missing")?;
    for target in targets {
        for case in cases {
            for scope in ["device_resident", "host_to_host", "cold_process"] {
                if !expected.iter().any(|e| {
                    e["case_id"] == case["id"]
                        && e["target"] == target["name"]
                        && e["scope"] == scope
                        && e["gate"]
                            == if scope == "cold_process" {
                                "cold"
                            } else if case["performance_class"] == "placement_control" {
                                "dispatch"
                            } else {
                                "parity"
                            }
                }) {
                    missing_case_scopes += 1;
                    reasons.push("frozen case excluded");
                }
            }
        }
    }
    for target in targets {
        if ["name", "host", "device_id", "compute_capability"]
            .iter()
            .any(|key| target[key].as_str().is_none_or(str::is_empty))
        {
            reasons.push("named target facts missing");
        }
        for family in [
            "map",
            "reduction",
            "scan",
            "compaction",
            "transpose",
            "stencil",
            "segmented",
            "gather_scatter",
            "frontier",
            "summaries",
        ] {
            for gate in ["bandwidth", "held_out_median", "held_out_p10"] {
                if !expected.iter().any(|e| {
                    e["target"] == target["name"]
                        && e["gate"] == gate
                        && cases
                            .iter()
                            .any(|c| c["id"] == e["case_id"] && c["family"] == family)
                }) {
                    reasons.push("section-10 gate missing");
                }
            }
        }
        for numeric in ["strict_fp32", "tc_f16_f32"] {
            if !expected.iter().any(|e| {
                e["target"] == target["name"]
                    && e["gate"] == "compute"
                    && cases.iter().any(|c| {
                        c["id"] == e["case_id"]
                            && c["numeric_class"] == numeric
                            && c["shape"] == json!([4096, 4096, 4096])
                    })
            }) {
                reasons.push("GEMM numeric-class/shape gate missing");
            }
        }
        for (family, numeric) in [
            ("map", "f32"),
            ("reduction", "reassoc_add"),
            ("scan", "reassoc_add"),
        ] {
            if !expected.iter().any(|e| {
                e["target"] == target["name"]
                    && e["gate"] == "bandwidth"
                    && cases.iter().any(|c| {
                        c["id"] == e["case_id"]
                            && c["family"] == family
                            && c["numeric_class"] == numeric
                            && c["large_case_preflight"] == "accepted"
                    })
            }) {
                reasons.push("large f32 bandwidth gate missing");
            }
        }
        for gate in ["compute", "dispatch_small", "tc_library_ratio"] {
            if !expected
                .iter()
                .any(|e| e["target"] == target["name"] && e["gate"] == gate)
            {
                reasons.push("section-10 gate missing");
            }
        }
    }
    let mut library_assisted = Vec::new();
    let library_packets = bundle["library_assisted"].as_array();
    for target in targets {
        for case in cases {
            let packet = library_packets.and_then(|rows| {
                rows.iter()
                    .find(|r| r["target"] == target["name"] && r["case_id"] == case["id"])
            });
            let reviewed = packet
                .ok_or_else(|| "library-assisted report unavailable".to_string())
                .and_then(|packet| {
                    let saved = &packet["report"];
                    lineage(saved, &manifest, &digest, target, true)?;
                    let mut summaries = serde_json::Map::new();
                    for scope in ["device_resident", "host_to_host", "cold_process"] {
                        summaries.insert(scope.into(), summary(&timing(saved, true, scope)?));
                    }
                    Ok(json!({"status":"reported","timing":summaries,"acceptance_credit":false}))
                });
            let result=reviewed.unwrap_or_else(|reason|json!({"status":if packet.is_none(){"unavailable"}else{"pending"},"reason":reason,"acceptance_credit":false}));
            library_assisted
                .push(json!({"case_id":case["id"],"target":target["name"],"result":result}));
        }
    }
    let library_reported = library_assisted
        .iter()
        .filter(|r| r["result"]["status"] == "reported")
        .count();
    if library_reported != library_assisted.len() || library_assisted.is_empty() {
        reasons.push("library-assisted reporting incomplete");
    }
    reasons.sort_unstable();
    reasons.dedup();
    let passed = *counts.get("passed").unwrap_or(&0);
    let synthetic = bundle["synthetic"] == true;
    let unavailable_obligations = reasons
        .iter()
        .filter(|r| {
            [
                "required family/target/gate coverage missing",
                "frozen case excluded",
                "section-10 gate missing",
            ]
            .contains(r)
        })
        .count();
    let verdict = if synthetic {
        "synthetic_no_claim"
    } else if reasons.is_empty()
        && !expected.is_empty()
        && passed + counts.get("exception").copied().unwrap_or(0) == expected.len()
    {
        "passed"
    } else if counts.get("failed").is_some_and(|n| *n > 0) {
        "failed"
    } else {
        "pending"
    };
    let mut families = BTreeMap::<String, Value>::new();
    for row in &output {
        let family = row["family"].as_str().unwrap_or("unknown").to_string();
        let entry = families.entry(family).or_insert_with(||json!({"denominator":0,"passed":0,"failed":0,"pending":0,"unavailable":0,"ignored":0,"exception":0}));
        entry["denominator"] = json!(entry["denominator"].as_u64().unwrap_or(0) + 1);
        let key = row["result"]["status"].as_str().unwrap_or("pending");
        entry[key] = json!(entry[key].as_u64().unwrap_or(0) + 1);
    }
    let ratios: Vec<_> = output
        .iter()
        .filter(|r| r["gate"] == "parity" && r["scope"] == "device_resident")
        .filter_map(|r| r["result"]["statistics"]["ratio"].as_f64())
        .collect();
    Ok(
        json!({"schema_version":1,"kind":"m5_03_performance_acceptance","status":verdict,"synthetic":synthetic,"manifest_sha256":digest,"plan_sha256":sha256(&serde_json::to_vec(plan).map_err(|e|e.to_string())?),"bundle_sha256":sha256(&serde_json::to_vec(bundle).map_err(|e|e.to_string())?),"denominator":expected.len(),"unavailable_obligation_classes":unavailable_obligations,"coverage_counts":{"family_target_gate_denominator":targets.len()*FAMILIES.len()*6,"missing_family_target_gates":missing_family_obligations,"case_scope_denominator":targets.len()*cases.len()*3,"missing_case_scopes":missing_case_scopes},"passed":passed,"counts":counts,"families":families,"reasons":reasons,"rows":output,"historical_geomean_diagnostic":if ratios.is_empty(){Value::Null}else{json!((ratios.iter().map(|x|x.ln()).sum::<f64>()/ratios.len() as f64).exp())},"geomean_is_acceptance":false,"library_assisted":{"denominator":library_assisted.len(),"reported":library_reported,"unavailable":library_assisted.iter().filter(|r|r["result"]["status"]=="unavailable").count(),"pending":library_assisted.iter().filter(|r|r["result"]["status"]=="pending").count(),"rows":library_assisted},"input_packets":bundle}),
    )
}

/// Render the same verdict and denominators without inventing missing timings.
pub fn markdown(report: &Value) -> String {
    let mut out = format!(
        "# M5-03 performance acceptance\n\nStatus: `{}`.\nSynthetic: {}.\nPassed: {} / {}.\nCounts: `{}`.\nNo aggregate hides missing or failed rows.\n\n| Row | Target | Track | Gate | Scope | Status | GPU conditions | Detail |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n",
        report["status"].as_str().unwrap_or("pending"),
        report["synthetic"],
        report["passed"],
        report["denominator"],
        report["counts"]
    );
    if let Some(rows) = report["rows"].as_array() {
        for r in rows {
            let escape = |v: &Value| {
                v.to_string()
                    .replace('|', "&#124;")
                    .replace(['\n', '\r'], " ")
            };
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
                escape(&r["id"]),
                escape(&r["target"]),
                escape(&r["track"]),
                escape(&r["gate"]),
                escape(&r["scope"]),
                escape(&r["result"]["status"]),
                escape(&r["gpu_conditions"]["label"]),
                escape(&r["result"])
            ));
        }
    }
    out.push_str(&format!("\nLibrary-assisted reported: {} / {}.\nUnavailable: {}.\nPending: {}.\nGenerated acceptance credit: zero.\n",report["library_assisted"]["reported"],report["library_assisted"]["denominator"],report["library_assisted"]["unavailable"],report["library_assisted"]["pending"]));
    out.push_str(&format!("\nRemaining refusals: `{}`.\nManifest SHA-256: `{}`.\nPlan SHA-256: `{}`.\nBundle SHA-256: `{}`.\n",report["reasons"],report["manifest_sha256"].as_str().unwrap_or("missing"),report["plan_sha256"].as_str().unwrap_or("missing"),report["bundle_sha256"].as_str().unwrap_or("missing")));
    out
}
