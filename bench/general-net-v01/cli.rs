use super::{checkout_at_invocation, hash, read_bounded, write_json};
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    ffi::OsString,
    io::Read,
    path::{Component, Path, PathBuf},
    process::{Command, Stdio},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

const MAX_RAW: u64 = 2 * 1024 * 1024;
const MAX_SOURCE: u64 = 64 * 1024;
// 256 MiB: an all-features Debug `invar` executable is ~151 MB (user decision, ADDENDUM-C1-C4 C1).
pub(crate) const MAX_BINARY: u64 = 256 * 1024 * 1024;
const MAX_REPORT: u64 = 256 * 1024;
const MAX_CHILD_OUTPUT: usize = 128 * 1024;
const CHILD_TIMEOUT: Duration = Duration::from_secs(5);

pub fn run(args: Vec<OsString>) -> Result<(), String> {
    let mode = args
        .first()
        .and_then(|arg| arg.to_str())
        .ok_or("missing GNET mode")?;
    let mut options = HashMap::new();
    let mut rest = args.iter().skip(1);
    while let Some(key) = rest.next() {
        let key = key.to_str().ok_or("non-UTF8 option")?;
        if ![
            "--manifest",
            "--spec",
            "--invar",
            "--raw",
            "--report",
            "--source-revision",
        ]
        .contains(&key)
            || options
                .insert(
                    key,
                    PathBuf::from(rest.next().ok_or("missing option value")?),
                )
                .is_some()
        {
            return Err(format!("invalid or repeated GNET option: {key}"));
        }
    }
    let get = |key| options.get(key).ok_or_else(|| format!("missing {key}"));
    let manifest_path = get("--manifest")?;
    let spec_path = get("--spec")?;
    let report_path = get("--report")?;
    let source_revision = get("--source-revision")?
        .to_str()
        .ok_or("non-UTF8 source revision")?;
    if ![40, 64].contains(&source_revision.len())
        || !source_revision.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err("invalid source revision".into());
    }
    if report_path.exists() {
        return Err("report path exists; choose a new path to preserve it".into());
    }
    let (manifest_raw, manifest, spec_raw, spec) = load_contracts(manifest_path, spec_path)?;
    let report = match mode {
        "--gnet-validate" if !options.contains_key("--raw") => validate_corpus(
            manifest_path,
            &manifest_raw,
            &manifest,
            &spec_raw,
            &spec,
            get("--invar")?,
            source_revision,
        )?,
        "--gnet-evaluate" if !options.contains_key("--invar") => evaluate(
            &manifest_raw,
            &manifest,
            &spec_raw,
            &spec,
            get("--raw")?,
            source_revision,
        )?,
        _ => return Err("invalid GNET mode/options".into()),
    };
    write_json(report_path, &report, MAX_REPORT)
}

fn load_contracts(
    manifest_path: &PathBuf,
    spec_path: &PathBuf,
) -> Result<(Vec<u8>, Value, Vec<u8>, Value), String> {
    let manifest_raw = read_bounded(manifest_path, 256 * 1024)?;
    let spec_raw = read_bounded(spec_path, 256 * 1024)?;
    let manifest: Value = serde_json::from_slice(&manifest_raw).map_err(|e| e.to_string())?;
    let spec: Value = serde_json::from_slice(&spec_raw).map_err(|e| e.to_string())?;
    let spec_name = spec_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("invalid spec file name")?;
    let manifest_name = manifest_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("invalid manifest file name")?;
    if manifest["schema_version"] != 1
        || spec["schema_version"] != 1
        || manifest["parity_spec"].as_str() != Some(spec_name)
        || spec["source_corpus"].as_str() != Some(manifest_name)
        || spec["status"] != "proposed_unmeasured"
        || spec["scored_track"] != "automatic_generated"
        || spec["measurement_status"] != "none"
        || spec["gate"]["point_estimate_max_inclusive"] != 1.0
        || spec["gate"]["one_sided_95_upper_max_inclusive"] != 1.05
        || spec["sampling"]["warmups_per_series_per_session_min"] != 10
        || spec["sampling"]["raw_samples_per_series_per_session_min"] != 30
        || spec["sampling"]["independent_process_sessions_min"] != 3
        || spec["sampling"]["bootstrap_replicates"] != 10_000
        || spec["sampling"]["one_sided_upper_percentile"] != 95
        || spec["ablation_design"]["variant_status"]["A"] != "not_built_optional"
        || spec["ablation_design"]["release_required_variants"] != json!(["B", "D", "CPU"])
    {
        return Err("GNET manifest/spec contract mismatch".into());
    }
    let scopes = spec["scored_scopes"]
        .as_array()
        .ok_or("missing scored scopes")?;
    if scopes.len() != 2
        || !scopes
            .iter()
            .any(|s| s["id"] == "device_resident_complete_operation")
        || !scopes
            .iter()
            .any(|s| s["id"] == "warm_host_input_to_host_output")
    {
        return Err("GNET scored scopes mismatch".into());
    }
    let cases = manifest["cases"].as_array().ok_or("missing cases")?;
    if cases.is_empty() || cases.len() > 64 || manifest["case_count"] != cases.len() {
        return Err("invalid GNET case count".into());
    }
    let mut ids = HashSet::new();
    for case in cases {
        let id = case["id"].as_str().ok_or("missing case id")?;
        if id.len() > 80 || !ids.insert(id) || !digest(case["source_sha256"].as_str()) {
            return Err("duplicate/invalid GNET case identity".into());
        }
    }
    Ok((manifest_raw, manifest, spec_raw, spec))
}

fn digest(value: Option<&str>) -> bool {
    value.is_some_and(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
}

fn validate_corpus(
    manifest_path: &Path,
    manifest_raw: &[u8],
    manifest: &Value,
    spec_raw: &[u8],
    _spec: &Value,
    invar: &PathBuf,
    source_revision: &str,
) -> Result<Value, String> {
    let binary_raw = read_bounded(invar, MAX_BINARY)?;
    let binary_sha = hash::sha256(&binary_raw);
    let base = manifest_path.parent().unwrap_or(Path::new("."));
    let canonical_base = base.canonicalize().map_err(|e| e.to_string())?;
    let mut results = Vec::new();
    for case in manifest["cases"].as_array().ok_or("missing cases")? {
        let id = case["id"].as_str().ok_or("missing case id")?;
        let rel = case["source"].as_str().ok_or("missing source path")?;
        let rel_path = Path::new(rel);
        if !rel_path
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
        {
            return Err(format!("unsafe source path for {id}"));
        }
        let source = base.join(rel_path);
        if !source
            .canonicalize()
            .map_err(|e| e.to_string())?
            .starts_with(&canonical_base)
        {
            return Err(format!("source escapes corpus for {id}"));
        }
        let bytes = read_bounded(&source, MAX_SOURCE)?;
        if case["source_bytes"] != bytes.len() || case["source_sha256"] != hash::sha256(&bytes) {
            return Err(format!("source digest/size mismatch for {id}"));
        }
        let fuel = case["fuel"]
            .as_u64()
            .filter(|n| (1..=1_000_000).contains(n))
            .ok_or("invalid fuel")?;
        let check = observe(invar, &source, "check", fuel)?;
        match_expected(&check, &case["expected_check"], id, "check")?;
        let run = observe(invar, &source, "run", fuel)?;
        match_expected(&run, &case["expected_reference_run"], id, "run")?;
        results.push(json!({"id":id,"source_sha256":case["source_sha256"],"check":check,"reference_run":run}));
    }
    Ok(
        json!({"schema_version":1,"kind":"gnet_corpus_validation","status":"passed",
        "gpu_acceptance":"pending","timing":"not_recorded","case_count":results.len(),
        "validation_scope":"source_literal_corpus_semantics_only",
        "manifest_sha256":hash::sha256(manifest_raw),"parity_spec_sha256":hash::sha256(spec_raw),
        "invar_executable_sha256":binary_sha,"source_revision":source_revision,
        "source_revision_provenance":"caller_supplied_unverified",
        "executable_source_binding":"unverified",
        "checkout_at_invocation":checkout_at_invocation(),
        "source_input_kind":"source_literals_reference_smoke","results":results}),
    )
}

fn observe(invar: &Path, source: &Path, action: &str, fuel: u64) -> Result<Value, String> {
    let mut command = Command::new(invar);
    command.arg(action).arg(source);
    if action == "run" {
        command
            .args(["--engine", "reference", "--fuel"])
            .arg(fuel.to_string());
    }
    command.args(["--format", "json"]);
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| e.to_string())?;
    let (sender, receiver) = mpsc::channel();
    for (stream, pipe) in [
        (
            "stdout",
            Box::new(child.stdout.take().unwrap()) as Box<dyn Read + Send>,
        ),
        (
            "stderr",
            Box::new(child.stderr.take().unwrap()) as Box<dyn Read + Send>,
        ),
    ] {
        let sender = sender.clone();
        thread::spawn(move || {
            let mut bytes = Vec::new();
            let result = pipe
                .take(MAX_CHILD_OUTPUT as u64 + 1)
                .read_to_end(&mut bytes)
                .map(|_| bytes)
                .map_err(|e| e.to_string());
            let _ = sender.send((stream, result));
        });
    }
    drop(sender);
    let started = Instant::now();
    let mut stdout = None;
    let mut stderr = None;
    let status = loop {
        while let Ok((stream, result)) = receiver.try_recv() {
            let bytes = match result {
                Ok(bytes) => bytes,
                Err(error) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!("{action} {stream}: {error}"));
                }
            };
            if bytes.len() > MAX_CHILD_OUTPUT {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!(
                    "{action} emitted oversized output/stderr for {}",
                    source.display()
                ));
            }
            if stream == "stdout" {
                stdout = Some(bytes);
            } else {
                stderr = Some(bytes);
            }
        }
        let status = match child.try_wait() {
            Ok(status) => status,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error.to_string());
            }
        };
        if let (Some(stdout), Some(stderr), Some(status)) = (&stdout, &stderr, status) {
            break (stdout, stderr, status);
        }
        if started.elapsed() >= CHILD_TIMEOUT {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("{action} timed out for {}", source.display()));
        }
        thread::sleep(Duration::from_millis(10));
    };
    if !status.1.is_empty() {
        return Err(format!(
            "{action} emitted oversized output/stderr for {}",
            source.display()
        ));
    }
    let actual: Value =
        serde_json::from_slice(status.0).map_err(|e| format!("{action} JSON: {e}"))?;
    let diagnostics = actual["diagnostics"]
        .as_array()
        .ok_or("missing diagnostics")?
        .iter()
        .map(|d| d["code"].clone())
        .collect::<Vec<_>>();
    Ok(
        json!({"exit":status.2.code().unwrap_or(-1),"status":actual["status"],
        "result":actual["result"],"trap":actual["trap"],"fuel_limit":actual["fuel_limit"],
        "steps_used":actual["steps_used"],"diagnostic_codes":diagnostics}),
    )
}

fn match_expected(actual: &Value, expected: &Value, id: &str, action: &str) -> Result<(), String> {
    let expected = expected.as_object().ok_or("invalid expected observation")?;
    for (key, value) in expected {
        if actual.get(key) != Some(value) {
            return Err(format!(
                "{id} {action} {key}: expected {value}, got {}",
                actual[key]
            ));
        }
    }
    Ok(())
}

#[cfg(all(test, unix))]
mod observe_tests {
    use super::*;
    use std::{
        fs,
        sync::atomic::{AtomicU64, Ordering},
    };

    fn script_result(body: &str) -> Result<Value, String> {
        // A counter, not SystemTime (microseconds on macOS): parallel tests
        // must never publish over each other's script (#255).
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "invar-gnet-observe-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        // #119's writer (loaded once, in `baseline_families`) never leaves a writable
        // descriptor for a sibling test's fork to inherit (#245).
        crate::baseline_families::executable::write(
            &path,
            format!("#!/bin/sh\n{body}\n").as_bytes(),
        )
        .unwrap();
        let result = observe(&path, Path::new("ignored.invar"), "check", 1);
        fs::remove_file(path).unwrap();
        result
    }

    #[test]
    fn drains_both_pipes_and_enforces_output_limit_and_timeout() {
        let json = "printf '{\"status\":\"ok\",\"diagnostics\":[]}'";
        let padding = "awk 'BEGIN {for (i=0; i<110000; ++i) printf \" \"}'";
        assert_eq!(
            script_result(&format!("{json}\n{padding}")).unwrap()["status"],
            "ok"
        );
        let error = script_result(&format!("{json}\n{padding}\n{padding} >&2")).unwrap_err();
        assert!(error.contains("output/stderr"), "{error}");
        let error = script_result(&format!(
            "{json}\nawk 'BEGIN {{for (i=0; i<140000; ++i) printf \" \"}}'"
        ))
        .unwrap_err();
        assert!(error.contains("oversized"), "{error}");
        let error = script_result("exec sleep 10").unwrap_err();
        assert!(error.contains("timed out"), "{error}");
    }
}

fn evaluate(
    manifest_raw: &[u8],
    manifest: &Value,
    spec_raw: &[u8],
    spec: &Value,
    raw_path: &PathBuf,
    source_revision: &str,
) -> Result<Value, String> {
    let raw_bytes = read_bounded(raw_path, MAX_RAW)?;
    let raw: Value = serde_json::from_slice(&raw_bytes).map_err(|e| format!("raw JSON: {e}"))?;
    let manifest_sha = hash::sha256(manifest_raw);
    let spec_sha = hash::sha256(spec_raw);
    if raw["schema_version"] != 1
        || raw["kind"] != "gnet_paired_raw_v1"
        || raw["manifest_sha256"] != manifest_sha
        || raw["parity_spec_sha256"] != spec_sha
    {
        return Err("raw measurement manifest/spec lineage mismatch".into());
    }
    let cases = raw["cases"].as_array().ok_or("missing raw cases")?;
    if cases.is_empty() || cases.len() > 16 {
        return Err("raw case count outside 1..=16".into());
    }
    let manifest_cases = manifest["cases"]
        .as_array()
        .ok_or("missing manifest cases")?;
    let mut groups: HashMap<(String, String, String), (HashSet<String>, Value)> = HashMap::new();
    let mut output = Vec::new();
    for case in cases {
        let id = nonempty(case["case_id"].as_str(), 80, "case_id")?;
        let size = nonempty(case["size"].as_str(), 80, "size")?;
        let work_items = case["work_items"]
            .as_u64()
            .filter(|n| (1..=100_000_000).contains(n))
            .ok_or("invalid work_items")?;
        let device = case["device"].as_str().ok_or("missing device")?;
        if !["cuda", "metal"].contains(&device) || case["track"] != "B/D" {
            return Err("invalid device or scored track".into());
        }
        let scope = case["scope"].as_str().ok_or("missing scope")?;
        if ![
            "device_resident_complete_operation",
            "warm_host_input_to_host_output",
        ]
        .contains(&scope)
        {
            return Err("invalid scored scope".into());
        }
        let source = manifest_cases
            .iter()
            .find(|m| m["id"] == id)
            .ok_or("unknown source case")?;
        let binding = &case["binding"];
        if binding != &case["invar_binding"] || binding != &case["baseline_binding"] {
            return Err("paired measurement binding mismatch".into());
        }
        for key in [
            "manifest_sha256",
            "parity_spec_sha256",
            "source_sha256",
            "input_sha256",
            "baseline_sha256",
            "device_sha256",
            "numeric_contract_sha256",
            "scope_sha256",
        ] {
            if !digest(binding[key].as_str()) {
                return Err(format!("missing/invalid binding {key}"));
            }
        }
        if binding["manifest_sha256"] != manifest_sha
            || binding["parity_spec_sha256"] != spec_sha
            || binding["source_sha256"] != source["source_sha256"]
            || binding["scope_sha256"] != hash::sha256(scope.as_bytes())
        {
            return Err("paired measurement source/manifest/spec/scope digest mismatch".into());
        }
        let key = (id.to_owned(), size.to_owned(), device.to_owned());
        let mut shared = binding.clone();
        shared["scope_sha256"] = Value::Null;
        let (scopes, prior_binding) = groups
            .entry(key)
            .or_insert_with(|| (HashSet::new(), shared.clone()));
        if *prior_binding != shared || !scopes.insert(scope.to_owned()) {
            return Err("duplicate scope or cross-scope binding mismatch".into());
        }
        let sessions = case["sessions"].as_array().ok_or("missing sessions")?;
        if !(3..=5).contains(&sessions.len()) {
            return Err("sessions outside 3..=5".into());
        }
        let mut session_ids = HashSet::new();
        let mut parsed = Vec::new();
        for session in sessions {
            let session_id = nonempty(session["id"].as_str(), 80, "session id")?;
            if !session_ids.insert(session_id)
                || session["warmups_invar"]
                    .as_u64()
                    .is_none_or(|n| !(10..=100).contains(&n))
                || session["warmups_baseline"]
                    .as_u64()
                    .is_none_or(|n| !(10..=100).contains(&n))
                || session["order_seed"] != spec["sampling"]["order_seed"]
            {
                return Err("invalid/duplicate session or warmup/order record".into());
            }
            let pairs = session["pairs"].as_array().ok_or("missing pairs")?;
            if ![30, 40, 50].contains(&pairs.len()) {
                return Err("paired blocks must contain 30, 40, or 50 samples".into());
            }
            let mut blocks = vec![false; pairs.len()];
            let mut orders = HashSet::new();
            let mut timings = Vec::new();
            for pair in pairs {
                let block = pair["block_id"].as_u64().ok_or("missing block id")? as usize;
                let order = pair["order"].as_str().ok_or("missing pair order")?;
                if block >= blocks.len()
                    || blocks[block]
                    || !["invar_first", "baseline_first"].contains(&order)
                {
                    return Err("duplicate/missing block or invalid order".into());
                }
                blocks[block] = true;
                orders.insert(order);
                let invar = positive(pair["invar_ms"].as_f64(), "invar_ms")?;
                let baseline = positive(pair["baseline_ms"].as_f64(), "baseline_ms")?;
                timings.push((invar, baseline));
            }
            if blocks.iter().any(|seen| !seen) || orders.len() != 2 {
                return Err("missing paired block or interleaved order".into());
            }
            parsed.push(timings);
        }
        let (ratio, upper) = statistics(
            &parsed,
            spec["sampling"]["bootstrap_seed"]
                .as_u64()
                .ok_or("bootstrap seed")?,
            id,
            scope,
        )?;
        let statistical_status = if ratio > 1.0 {
            "fail"
        } else if upper <= 1.05 {
            "pass"
        } else {
            "inconclusive"
        };
        let evidence = &case["evidence"];
        let qualification_reason = if evidence["kind"] == "synthetic_fixture" {
            "synthetic_fixture_not_performance_evidence"
        } else if source["input_binding"] != "runtime_buffers" {
            "source_literals_not_runtime_input_GPU_acceptance"
        } else if spec["status"] != "frozen_for_scoring"
            || spec["baseline_prerequisites"]["status"] != "reviewed"
        {
            "spec_or_baseline_not_frozen_for_scoring"
        } else if [
            "device_execution",
            "numeric_contract",
            "oracle",
            "baseline_review",
            "compiled_source",
            "runtime_inputs",
        ]
        .iter()
        .any(|key| evidence[*key] != "verified")
        {
            "missing_device_numeric_oracle_or_baseline_evidence"
        } else {
            "eligible_for_external_hardware_review"
        };
        let qualification_status =
            if qualification_reason == "eligible_for_external_hardware_review" {
                statistical_status
            } else {
                "pending"
            };
        output.push(json!({"case_id":id,"size":size,"work_items":work_items,"device":device,"scope":scope,
            "binding":binding,"sessions":sessions.len(),"pairs_per_session":parsed[0].len(),
            "ratio_median":ratio,"one_sided_95_upper":upper,"statistical_status":statistical_status,
            "qualification_status":qualification_status,"qualification_reason":qualification_reason}));
    }
    if groups.values().any(|(scopes, _)| scopes.len() != 2) {
        return Err("each case/size/device needs both scored scopes".into());
    }
    Ok(json!({"schema_version":1,"kind":"gnet_paired_evaluation",
        "status":"pending_gpu_acceptance","manifest_sha256":manifest_sha,"parity_spec_sha256":spec_sha,
        "raw_sha256":hash::sha256(&raw_bytes),"source_revision":source_revision,
        "source_revision_provenance":"caller_supplied_unverified",
        "executable_source_binding":"unverified",
        "checkout_at_invocation":checkout_at_invocation(),
        "method":"hierarchical_session_then_paired_block_bootstrap_10000_seeded",
        "gate":{"point_max":1.0,"one_sided_95_upper_max":1.05},
        "cases":output,"hardware_timing_performed_by_evaluator":false}))
}

fn nonempty<'a>(value: Option<&'a str>, max: usize, name: &str) -> Result<&'a str, String> {
    value
        .filter(|s| !s.is_empty() && s.len() <= max)
        .ok_or_else(|| format!("invalid {name}"))
}

fn positive(value: Option<f64>, name: &str) -> Result<f64, String> {
    value
        .filter(|n| n.is_finite() && (1e-9..=1e9).contains(n))
        .ok_or_else(|| format!("invalid {name}"))
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

struct Rng(u64);
impl Rng {
    fn choose(&mut self, len: usize) -> usize {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)) % len as u64) as usize
    }
}

fn statistics(
    sessions: &[Vec<(f64, f64)>],
    seed: u64,
    id: &str,
    scope: &str,
) -> Result<(f64, f64), String> {
    let mut invar = sessions
        .iter()
        .flatten()
        .map(|pair| pair.0)
        .collect::<Vec<_>>();
    let mut baseline = sessions
        .iter()
        .flatten()
        .map(|pair| pair.1)
        .collect::<Vec<_>>();
    let ratio = median(&mut invar) / median(&mut baseline);
    let identity = hash::sha256(format!("{id}/{scope}").as_bytes());
    let salt = u64::from_str_radix(&identity[..16], 16).unwrap_or(1);
    let mut rng = Rng((seed ^ salt).max(1));
    let mut replicates = Vec::with_capacity(10_000);
    for _ in 0..10_000 {
        let mut invar = Vec::new();
        let mut baseline = Vec::new();
        for _ in sessions {
            let session = &sessions[rng.choose(sessions.len())];
            for _ in session {
                let (a, b) = session[rng.choose(session.len())];
                invar.push(a);
                baseline.push(b);
            }
        }
        replicates.push(median(&mut invar) / median(&mut baseline));
    }
    replicates.sort_by(f64::total_cmp);
    if !ratio.is_finite() || !replicates[9_499].is_finite() {
        return Err("timing ratio is not finite".into());
    }
    Ok((ratio, replicates[9_499]))
}
