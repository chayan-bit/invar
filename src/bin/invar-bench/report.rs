use super::*;

pub(crate) fn baseline_eligibility(generated: &Value, baseline: &Value, case: &Value) -> Value {
    let result = &generated["generated_result"];
    let candidate = &baseline["baseline_result"];
    let mut reasons = Vec::new();
    if baseline["manifest_sha256"] != generated["manifest_sha256"]
        || baseline["corpus_sha256"] != generated["corpus_sha256"]
    {
        reasons.push("manifest_or_corpus_digest_mismatch");
    }
    if candidate["candidate_case_id"] != case["id"]
        || result["case_id"] != case["id"]
        || candidate["elements"] != case["elements"]
    {
        reasons.push("case_or_size_mismatch");
    }
    let family = match case["family"].as_str() {
        Some("reduction") => "sum",
        Some("scan") => "affine",
        Some("map") => "map",
        _ => "unknown",
    };
    if candidate["family"] != family || candidate["arithmetic_compatibility"] != "u32_wrapping_v1" {
        reasons.push("arithmetic_law_mismatch");
    }
    if result["full_output_correct"] != true
        || candidate["full_output_correctness"] != "exact_full_output_pass"
    {
        reasons.push("full_output_verification_missing");
    }
    if candidate["gate_status"] != "accepted"
        || candidate["target_compatibility"] != "accepted"
        || candidate["build_provenance"] != "pinned_source_and_executable"
    {
        reasons.push("baseline_semantic_target_or_build_review_pending");
    }
    if result["build_review_status"] != "accepted" {
        reasons.push("generated_build_review_pending");
    }
    if result["measurement_scope"] != "checked_source_program" {
        reasons.push("parameterized_extracted_plan_not_source_program");
    }
    if generated["checkout_at_invocation"]["worktree"] != "clean"
        || baseline["checkout_at_invocation"]["worktree"] != "clean"
        || result["build_record"]["worktree_at_build"] != "clean"
        || candidate["build_record"]["worktree_at_build"] != "clean"
    {
        reasons.push("clean_build_lineage_missing");
    }
    let device_match = baseline["gpu_preflight"].as_array().is_some_and(|checks| {
        checks
            .iter()
            .any(|check| check["status"] == "available" && check["device"] == result["device"])
    });
    if !device_match {
        reasons.push("device_identity_mismatch");
    }
    if candidate["required_transfer_bytes"] != result["required_bytes"]
        || result["required_bytes"] != case["required_bytes"]
    {
        reasons.push("required_byte_contract_mismatch");
    }
    if generated["freeze_status"] != "frozen" || baseline["freeze_status"] != "frozen" {
        reasons.push("provisional_manifest_not_ratio_eligible");
    }
    let measured_output = if result["measured_output_sha256"].is_string() {
        &result["measured_output_sha256"]
    } else {
        &result["source_observable_output_sha256"]
    };
    if candidate["input_recipe"] != result["input_recipe"]
        || candidate["input_sha256"].is_null()
        || result["input_sha256"].is_null()
        || candidate["input_sha256"] != result["input_sha256"]
        || measured_output.is_null()
        || &candidate["output_sha256"] != measured_output
    {
        reasons.push("matched_input_and_source_output_digests_missing");
    }
    if result["device_resident_boundary"].is_null()
        || candidate["device_resident_boundary"] != result["device_resident_boundary"]
        || result["host_to_host_boundary"].is_null()
        || candidate["host_to_host_boundary"] != result["host_to_host_boundary"]
    {
        reasons.push("device_or_host_timing_boundary_mismatch");
    }
    if generated["cold_process"]["status"] != "passed"
        || baseline["cold_process"]["status"] != "passed"
        || baseline["cold_process"]["samples_ms"]
            .as_array()
            .is_none_or(|samples| samples.len() != 30)
        || baseline["cold_process"]["boundary"] != generated["cold_process"]["boundary"]
    {
        reasons.push("cold_process_boundary_or_samples_mismatch");
    }
    // The fixture and generated children perform different setup work; an accepted
    // comparison contract needs a separate code review before this gate can open.
    reasons.push("cold_process_scope_review_pending");
    let medians = [
        median_30(&result["device_resident_ms"]),
        median_30(&candidate["device_resident_samples_ms"]),
        median_30(&result["host_to_host_ms"]),
        median_30(&candidate["host_to_host_samples_ms"]),
        median_30(&generated["cold_process"]["samples_ms"]),
        median_30(&baseline["cold_process"]["samples_ms"]),
    ];
    if medians[..4].iter().any(Option::is_none) {
        reasons.push("raw_hot_timing_samples_missing_or_invalid");
    }
    if medians[4..].iter().any(Option::is_none) {
        reasons.push("raw_cold_timing_samples_missing_or_invalid");
    }
    if reasons.is_empty() {
        let [
            Some(gpu),
            Some(gpu_base),
            Some(host),
            Some(host_base),
            Some(cold),
            Some(cold_base),
        ] = medians
        else {
            unreachable!();
        };
        return json!({"status":"eligible","candidate_case_id":candidate["candidate_case_id"],
            "baseline_name":candidate["baseline_name"],"reasons":[],
            "device_resident_ratio":gpu/gpu_base,"host_to_host_ratio":host/host_base,
            "cold_process_ratio":cold/cold_base});
    }
    json!({"status":"pending","candidate_case_id":candidate["candidate_case_id"],
        "baseline_name":candidate["baseline_name"],"reasons":reasons,
        "device_resident_ratio":null,"host_to_host_ratio":null,"cold_process_ratio":null})
}

pub(crate) fn median_30(samples: &Value) -> Option<f64> {
    let mut values = samples
        .as_array()?
        .iter()
        .map(|value| value.as_f64().filter(|n| n.is_finite() && *n > 0.0))
        .collect::<Option<Vec<_>>>()?;
    if values.len() != 30 {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some((values[14] + values[15]) / 2.0)
}

pub(crate) fn checkout_at_invocation() -> Value {
    let root = env!("CARGO_MANIFEST_DIR");
    let head = Command::new("git")
        .args(["-C", root, "rev-parse", "--verify", "HEAD"])
        .output();
    let Ok(head) = head else {
        return json!({"head":null,"worktree":"unavailable"});
    };
    let Ok(sha) = std::str::from_utf8(&head.stdout) else {
        return json!({"head":null,"worktree":"unavailable"});
    };
    let sha = sha.trim();
    if !head.status.success()
        || ![40, 64].contains(&sha.len())
        || !sha.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return json!({"head":null,"worktree":"unavailable"});
    }
    let worktree = match Command::new("git")
        .args([
            "-C",
            root,
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ])
        .output()
    {
        Ok(output) if output.status.success() && output.stdout.is_empty() => "clean",
        Ok(output) if output.status.success() => "dirty",
        _ => "unavailable",
    };
    json!({"head":sha,"worktree":worktree})
}

pub(crate) fn gpu_preflight() -> Value {
    let checks = Vec::new();
    #[cfg(feature = "cuda")]
    let checks = {
        let mut checks = checks;
        checks.push(match invar_cuda::cuda::CudaDevice::open() {
            Ok(device) => json!({"backend":"cuda","status":"available","device":device.name()}),
            Err(error) => {
                json!({"backend":"cuda","status":"unavailable","reason":format!("{error:?}")})
            }
        });
        checks
    };
    #[cfg(all(feature = "metal", target_os = "macos"))]
    let checks = {
        let mut checks = checks;
        checks.push(match invar_metal::MetalDevice::open() {
            Ok(device) => json!({"backend":"metal","status":"available","device":device.name()}),
            Err(error) => {
                json!({"backend":"metal","status":"unavailable","reason":format!("{error:?}")})
            }
        });
        checks
    };
    if checks.is_empty() {
        json!([{"status":"backend_feature_not_enabled"}])
    } else {
        Value::Array(checks)
    }
}
