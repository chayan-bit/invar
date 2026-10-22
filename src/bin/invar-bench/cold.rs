use super::*;

pub(crate) fn read_bounded(path: &PathBuf, limit: u64) -> Result<Vec<u8>, String> {
    let meta = fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if !meta.is_file() || meta.len() > limit {
        return Err(format!(
            "{}: invalid file or size limit exceeded",
            path.display()
        ));
    }
    let file = fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut bytes = Vec::new();
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > limit {
        return Err("file size limit exceeded during read".into());
    }
    Ok(bytes)
}

pub(crate) fn json_bytes(value: &Value, limit: u64) -> Result<Vec<u8>, String> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?;
    bytes.push(b'\n');
    if bytes.len() as u64 > limit {
        return Err("artifact size limit exceeded".into());
    }
    Ok(bytes)
}

pub(crate) fn write_json(path: &PathBuf, value: &Value, limit: u64) -> Result<(), String> {
    let bytes = json_bytes(value, limit)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .and_then(|mut file| file.write_all(&bytes))
        .map_err(|e| format!("report {}: {e}", path.display()))
}

pub(crate) fn wait_bounded(
    child: &mut Child,
    started: Instant,
    limit: Duration,
) -> Result<(ExitStatus, f64), String> {
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok((status, started.elapsed().as_secs_f64() * 1000.0)),
            Ok(None) if started.elapsed() < limit => thread::sleep(Duration::from_millis(10)),
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!(
                    "cold child timed out after {} seconds",
                    limit.as_secs()
                ));
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("cold child wait failed: {error}"));
            }
        }
    }
}

pub(crate) fn validate_cold_child(parent: &Value, child: &Value) -> Result<(), String> {
    if child["schema_version"] != parent["schema_version"]
        || child["kind"] != "generated_cold_child"
        || child["status"] != "passed"
        || child["generated_result"]["full_output_correct"] != true
        || child["generated_result"]["warmups"] != 0
        || child["generated_result"]["samples"] != 1
    {
        return Err("cold child protocol or full-output check failed".into());
    }
    for key in ["manifest_sha256", "corpus_sha256", "held_out_sha256"] {
        if !parent[key].is_string() || child[key] != parent[key] {
            return Err(format!("cold child {key} mismatch"));
        }
    }
    if child["checkout_at_invocation"] != parent["checkout_at_invocation"] {
        return Err("cold child checkout mismatch".into());
    }
    let prior = &parent["generated_result"];
    let current = &child["generated_result"];
    for key in [
        "case_id",
        "program_id",
        "source_sha256",
        "source_sha256_role",
        "measurement_scope",
        "measured_invocation",
        "measured_invocation_sha256",
        "generated_code_hash",
        "runner_executable_sha256",
        "device",
        "toolchain",
        "geometry",
        "launches",
        "input_recipe",
        "input_sha256",
        "measured_output_sha256",
        "device_resident_boundary",
        "host_to_host_boundary",
        "required_bytes",
        "transfer_bytes",
        "workspace_bytes",
    ] {
        if prior[key].is_null() || current[key] != prior[key] {
            return Err(format!("cold child {key} mismatch"));
        }
    }
    if prior["toolchain"]
        .as_str()
        .is_some_and(|name| name.starts_with("NVRTC"))
        && (prior["compiler_executable_sha256"].is_null()
            || current["compiler_executable_sha256"] != prior["compiler_executable_sha256"])
    {
        return Err("cold child compiler executable mismatch".into());
    }
    if ["device_resident_ms", "host_to_host_ms"].iter().any(|key| {
        current[*key].as_array().is_none_or(|values| {
            values.len() != 1
                || values[0]
                    .as_f64()
                    .is_none_or(|ms| !ms.is_finite() || ms <= 0.0)
        })
    }) {
        return Err("cold child timing missing or invalid".into());
    }
    Ok(())
}

pub(crate) fn validate_baseline_cold_child(parent: &Value, child: &Value) -> Result<(), String> {
    if child["schema_version"] != parent["schema_version"]
        || child["kind"] != "baseline_cold_child"
        || child["status"] != "passed"
        || child["baseline_result"]["full_output_correctness"] != "exact_full_output_pass"
        || child["baseline_result"]["warmups"] != 0
        || child["baseline_result"]["samples"] != 1
    {
        return Err("baseline cold child protocol or full-output check failed".into());
    }
    for key in ["manifest_sha256", "corpus_sha256", "held_out_sha256"] {
        if !parent[key].is_string() || child[key] != parent[key] {
            return Err(format!("baseline cold child {key} mismatch"));
        }
    }
    if child["checkout_at_invocation"] != parent["checkout_at_invocation"] {
        return Err("baseline cold child checkout mismatch".into());
    }
    let prior = &parent["baseline_result"];
    let current = &child["baseline_result"];
    for key in [
        "candidate_case_id",
        "family",
        "elements",
        "arithmetic_compatibility",
        "fixture_source_sha256",
        "executable_sha256",
        "build_record_sha256",
        "device_registry_id",
        "vendor_versions",
        "input_recipe",
        "input_sha256",
        "output_sha256",
        "required_transfer_bytes",
        "workspace_bytes",
        "device_resident_boundary",
        "host_to_host_boundary",
    ] {
        if prior[key].is_null() || current[key] != prior[key] {
            return Err(format!("baseline cold child {key} mismatch"));
        }
    }
    for key in ["device_resident_samples_ms", "host_to_host_samples_ms"] {
        if current[key].as_array().is_none_or(|samples| {
            samples.len() != 1
                || samples[0]
                    .as_f64()
                    .is_none_or(|ms| !ms.is_finite() || ms <= 0.0)
        }) {
            return Err(format!("baseline cold child {key} invalid"));
        }
    }
    Ok(())
}

pub(crate) fn read_cold_child(
    parent: &Value,
    path: &PathBuf,
    validate: impl Fn(&Value, &Value) -> Result<(), String>,
) -> Result<String, String> {
    let raw = read_bounded(path, MAX_REPORT)?;
    let child: Value = serde_json::from_slice(&raw).map_err(|e| format!("cold child JSON: {e}"))?;
    validate(parent, &child)?;
    Ok(hash::sha256(&raw))
}

pub(crate) fn cold_process_samples(
    report_path: &Path,
    case: &Value,
    parent: &Value,
    workload: &str,
    build_child: impl Fn(&Path) -> Command,
    validate: impl Fn(&Value, &Value) -> Result<(), String>,
) -> Result<Value, String> {
    let n = case["elements"]
        .as_u64()
        .ok_or("cold case length missing")?;
    let inputs = case["input_buffers"]
        .as_u64()
        .ok_or("cold input count missing")?;
    let host_budget = n
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_mul(inputs + 2))
        .ok_or("cold host byte count overflow")?;
    if host_budget > COLD_HOST_BUDGET {
        return Ok(
            json!({"status":"pending","reason":"cold host memory preflight exceeded 1 GiB"}),
        );
    }
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let directory = report_path.with_extension(format!("cold-{}-{nonce}", std::process::id()));
    if let Some(parent_dir) = directory.parent() {
        fs::create_dir_all(parent_dir).map_err(|e| e.to_string())?;
    }
    fs::create_dir(&directory).map_err(|e| e.to_string())?;
    let mut samples = Vec::with_capacity(30);
    let total_started = Instant::now();
    for index in 0..40 {
        if total_started.elapsed() >= COLD_TOTAL_TIMEOUT {
            return Ok(
                json!({"status":"pending","reason":"cold process sweep exceeded 300 seconds",
                "raw_directory":directory,"attempts_completed":index,"samples_ms":samples}),
            );
        }
        let child_report = directory.join(format!("attempt-{index:02}.json"));
        let stderr_path = directory.join(format!("attempt-{index:02}.stderr.txt"));
        let status_path = directory.join(format!("attempt-{index:02}.status.json"));
        let stderr = fs::File::create(&stderr_path).map_err(|e| e.to_string())?;
        let started = Instant::now();
        let remaining = COLD_TOTAL_TIMEOUT.saturating_sub(total_started.elapsed());
        let outcome = build_child(&child_report)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::from(stderr))
            .spawn()
            .map_err(|e| format!("cold child spawn failed: {e}"))
            .and_then(|mut child| wait_bounded(&mut child, started, COLD_TIMEOUT.min(remaining)));
        let validated = match outcome {
            Ok((status, elapsed_ms)) if status.success() => {
                read_cold_child(parent, &child_report, &validate).map(|digest| (elapsed_ms, digest))
            }
            Ok((status, _)) => Err(format!("cold child exited {status}")),
            Err(reason) => Err(reason),
        };
        let status = match &validated {
            Ok((elapsed_ms, digest)) => {
                json!({"status":"passed","kind":if index<10 {"warmup"} else {"sample"},
                "elapsed_ms":elapsed_ms,"child_report_sha256":digest,"child_report":child_report,
                "stderr":stderr_path,"manifest_sha256":parent["manifest_sha256"]})
            }
            Err(reason) => json!({"status":"failed","kind":if index<10 {"warmup"} else {"sample"},
                "reason":reason,"child_report":child_report,"stderr":stderr_path,
                "manifest_sha256":parent["manifest_sha256"]}),
        };
        write_json(&status_path, &status, MAX_REPORT)?;
        match validated {
            Ok((ms, _)) if index >= 10 => samples.push(ms),
            Ok(_) => {}
            Err(reason) => {
                return Ok(json!({"status":"pending","reason":reason,
                "raw_directory":directory,"attempts_completed":index,"samples_ms":samples}));
            }
        }
    }
    Ok(json!({"status":"passed","warmups":10,"samples_ms":samples,
        "boundary":"parent_before_spawn_through_verified_child_exit_and_report_write",
        "workload":workload,"contract_review":"pending_M5_cold_process_scope_review",
        "raw_directory":directory,"attempts_completed":40}))
}
