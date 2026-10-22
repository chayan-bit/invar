use super::*;

fn generated_map_fixture() -> (Value, Value) {
    let source = "fn main() -> i32 { \
            let xs: Array<u32> = filled_u32(3u32,1u32); \
            let ys: Array<u32> = filled_u32(3u32,2u32); \
            var a: Array<u32> = filled_u32(3u32,0u32); \
            for i in 0u32 .. 3u32 { a[i] = wrapping_add(xs[i],ys[i]); } \
            let aa: Array<u32> = move a; \
            var b: Array<u32> = filled_u32(3u32,0u32); \
            for j in 0u32 .. 3u32 { b[j] = wrapping_mul(aa[j],ys[j]); } \
            let bb: Array<u32> = move b; \
            var c: Array<u32> = filled_u32(3u32,0u32); \
            for k in 0u32 .. 3u32 { c[k] = wrapping_add(bb[k],ys[k]); } \
            let cc: Array<u32> = move c; \
            var result0: Array<u32> = filled_u32(3u32,0u32); \
            for l in 0u32 .. 3u32 { result0[l] = wrapping_add(cc[l],1u32); } \
            if result0[0u32] != 9u32 { return 1; } return 0; }";
    (
        json!({"id":"map-n33","family":"map","elements":33,
                "stages":3,"input_buffers":2,"output_buffers":1,"required_bytes":396}),
        json!({"programs":[{"id":"map-test","family":"map","split":"tuning",
                "stages":3,"inputs":2,"outputs":1,"source":source,
                "source_sha256":hash::sha256(source.as_bytes())}]}),
    )
}

#[test]
fn generated_map_adapter_selects_final_stage() {
    let (case, corpus) = generated_map_fixture();
    let report = generated_case_with(
        &case,
        &corpus,
        invar_device::Capabilities::cuda_12_9_sm89(),
        "test double",
        "no hardware",
        (0, 1),
        |plan, inputs, reference, _| {
            let dag = plan.ir().dag.as_ref().unwrap();
            assert_eq!(dag.outputs.len(), 1);
            assert_eq!(dag.stage_outputs, [dag.nodes.len() - 1]);
            let GeneratedReference::Map(actual) = reference else {
                panic!("map expected")
            };
            // Independent source arithmetic, rather than the adapter's DAG oracle.
            let expected = inputs[0]
                .iter()
                .zip(inputs[1])
                .map(|(&x, &y)| {
                    x.wrapping_add(y)
                        .wrapping_mul(y)
                        .wrapping_add(y)
                        .wrapping_add(1)
                })
                .collect::<Vec<_>>();
            assert_eq!(*actual, [expected]);
            let code = invar_device::emit_cuda(plan).unwrap();
            assert!(code.contains("output0"));
            assert!(!code.contains("output1"));
            Ok(GeneratedRaw {
                source_hash: 1,
                compile_ms: 0.0,
                device_resident_ms: vec![1.0],
                host_to_host_ms: vec![2.0],
                transfer_bytes: 396,
                workspace_bytes: 0,
                launches: 1,
            })
        },
    )
    .unwrap();
    assert_eq!(report["transfer_bytes"], 396);
    assert_eq!(report["launches"], 1);
    assert_eq!(report["workspace_bytes"], 0);
    let inputs = measured_inputs(33, 2).unwrap();
    let expected = inputs[0]
        .iter()
        .zip(&inputs[1])
        .map(|(&x, &y)| {
            x.wrapping_add(y)
                .wrapping_mul(y)
                .wrapping_add(y)
                .wrapping_add(1)
        })
        .collect::<Vec<_>>();
    assert_eq!(report["measured_output_sha256"], u32_digest(&[&expected]));
}

#[test]
fn generated_map_projection_preserves_typed_refusals() {
    use invar::{admission, check::check, device_ir, front::parse, mir};
    use invar_device::{Capabilities, Error, Provenance};
    let (_, corpus) = generated_map_fixture();
    let source = corpus["programs"][0]["source"].as_str().unwrap();
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    let admission = admission::analyze(source.as_bytes(), &program).unwrap();
    let ir = device_ir::from_region(&admission.regions()[0], 33).unwrap();
    assert_eq!(ir.dag.as_ref().unwrap().outputs.len(), 4);
    let origin = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    for caps in [
        Capabilities::cuda_12_9_sm89(),
        Capabilities::Metal {
            apple_family: 9,
            msl_major: 3,
            msl_minor: 1,
        },
    ] {
        // Out-of-range counts, and a count that would publish an
        // intermediate stage (`c` is not a `result{k}` loop), are refused.
        for count in [0, 2, 5] {
            let plan = invar_device::verify(ir.clone(), origin, caps).unwrap();
            assert!(matches!(
                generated_map_plan(plan, count, source),
                Err(Error::Operations)
            ));
        }
        let plan = invar_device::verify(ir.clone(), origin, caps).unwrap();
        let projected = generated_map_plan(plan, 1, source).unwrap();
        assert_eq!(
            projected.ir().dag.as_ref().unwrap().stage_outputs,
            ir.dag.as_ref().unwrap().stage_outputs[3..]
        );
        let mut wrong_stage = projected.ir().clone();
        wrong_stage.dag.as_mut().unwrap().stage_outputs[0] = 0;
        assert!(matches!(
            invar_device::verify(wrong_stage, origin, caps),
            Err(Error::Operations)
        ));
    }
}

#[test]
fn generated_map_long_chain_over_the_fusion_cap_is_refused_not_projected() {
    // 16 chained stages plus four result loops exceed the 16-region fusion
    // cap, so no admitted region publishes result0..result3; the case is
    // refused rather than measured on intermediate stages.
    let corpus: Value =
        serde_json::from_str(include_str!("../../../bench/corpus-v1.json")).unwrap();
    let manifest: Value =
        serde_json::from_str(include_str!("../../../bench/manifest-v1.json")).unwrap();
    let case = manifest["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["id"] == "map-long-chain-n33")
        .unwrap();
    let result = generated_case_with(
        case,
        &corpus,
        invar_device::Capabilities::cuda_12_9_sm89(),
        "test double",
        "no hardware",
        (0, 1),
        |_, _, _, _| panic!("a long chain over the fusion cap must not be measured"),
    );
    assert!(result.is_err(), "{result:?}");
}

#[test]
fn report_path_is_immutable_across_kinds() {
    let dir = env::temp_dir().join(format!(
        "invar-measurement-write-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&dir).unwrap();
    let report = dir.join("report.json");
    let raw = dir.join("attempt-00.json");
    fs::write(&raw, b"original raw artifact").unwrap();
    let first =
        json!({"kind":"independent_library_baseline","manifest_sha256":"same","samples":[1]});
    write_json(&report, &first, MAX_REPORT).unwrap();
    let original = fs::read(&report).unwrap();
    let second =
        json!({"kind":"independent_library_baseline","manifest_sha256":"same","samples":[2]});
    assert!(write_json(&report, &second, MAX_REPORT).is_err());
    assert!(
        write_json(
            &report,
            &json!({"kind":"manifest_validation","manifest_sha256":"same"}),
            MAX_REPORT
        )
        .is_err()
    );
    assert_eq!(fs::read(&report).unwrap(), original);
    assert_eq!(fs::read(&raw).unwrap(), b"original raw artifact");

    let validation = dir.join("validation.json");
    write_json(
        &validation,
        &json!({"kind":"manifest_validation","manifest_sha256":"same","status":"first"}),
        MAX_REPORT,
    )
    .unwrap();
    let first_validation = fs::read(&validation).unwrap();
    assert!(
        write_json(
            &validation,
            &json!({"kind":"manifest_validation","manifest_sha256":"same","status":"second"}),
            MAX_REPORT,
        )
        .is_err()
    );
    assert_eq!(fs::read(&validation).unwrap(), first_validation);
    let artifact = dir.join("corpus.json");
    write_json(&artifact, &json!({"version":1}), MAX_CORPUS).unwrap();
    let first_artifact = fs::read(&artifact).unwrap();
    assert!(write_json(&artifact, &json!({"version":2}), MAX_CORPUS).is_err());
    assert_eq!(fs::read(&artifact).unwrap(), first_artifact);
    let latest: Value = serde_json::from_slice(&fs::read(&artifact).unwrap()).unwrap();
    assert_eq!(latest["version"], 1);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn measured_host_staging_refuses_allocation_overflow_and_preserves_digest() {
    assert!(matches!(
        measured_words(usize::MAX),
        Err(invar_device::Error::ResourceLimit(
            "benchmark host staging allocation failed"
        ))
    ));
    assert_eq!(u32_digest(&[]), hash::sha256(&[]));
    let words = (0..4097u32).collect::<Vec<_>>();
    let bytes = words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    assert_eq!(
        u32_digest(&[&words[..1], &words[1..]]),
        hash::sha256(&bytes)
    );
}

#[test]
fn measured_invocation_binds_shape_seed_and_inputs() {
    let recipe = measured_inputs(3, 2).unwrap();
    assert_eq!(recipe[0], [1_013_904_223, 1_015_568_748, 1_017_233_273]);
    assert_eq!(recipe[1][0], recipe[0][0] + 17);
    let corpus = corpus::generate().unwrap();
    let item = corpus["programs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["id"] == "reduction-082")
        .unwrap();
    assert!(
        item["source"]
            .as_str()
            .unwrap()
            .contains("[3u32,4u32,2u32]")
    );
    assert!(item["source"].as_str().unwrap().contains("acc: u32 = 5u32"));
    let id = item["id"].as_str().unwrap();
    let source = item["source_sha256"].as_str().unwrap();
    let values = [3u32, 4, 2];
    let inputs = [values.as_slice()];
    let (original, digest) = measured_invocation(id, source, 3, Some(5), 1, &inputs).unwrap();
    assert_eq!(original["initial_state_u32"], 5);
    assert_eq!(original["elements"], 3);
    assert_ne!(
        measured_invocation(id, source, 3, Some(17), 1, &inputs)
            .unwrap()
            .1,
        digest
    );
    let (map_invocation, map_digest) =
        measured_invocation(id, source, 3, None, 1, &inputs).unwrap();
    assert!(map_invocation["initial_state_u32"].is_null());
    assert_ne!(map_digest, digest);
    let changed = [3u32, 4, 3];
    assert_ne!(
        measured_invocation(id, source, 3, Some(5), 1, &[changed.as_slice()])
            .unwrap()
            .1,
        digest
    );
    assert!(measured_invocation(id, source, 33, Some(5), 1, &inputs).is_err());
    assert!(measured_invocation(id, source, 3, Some(5), 2, &inputs).is_err());
    assert_ne!(
        measured_invocation(id, "other", 3, Some(5), 1, &inputs)
            .unwrap()
            .1,
        digest
    );
}

#[test]
fn cold_child_rejects_malformed_and_mismatched_provenance() {
    let parent = json!({"schema_version":1,"manifest_sha256":"manifest","corpus_sha256":"corpus",
            "held_out_sha256":"held","checkout_at_invocation":{"head":"head"},
            "generated_result":{"case_id":"scan-n33","program_id":"scan-161",
                "source_sha256":"source","generated_code_hash":"code","runner_executable_sha256":"runner",
                "source_sha256_role":"three_element_corpus_template_only",
                "measurement_scope":"parameterized_extracted_plan_not_source_program",
                "measured_invocation":{"elements":33},"measured_invocation_sha256":"invocation",
                "device":"Apple M4","toolchain":"Metal MSL 3.1","geometry":{"blocks":1,"lanes":1024},
                "input_recipe":"lcg-v1","input_sha256":"input",
                "measured_output_sha256":"output",
                "device_resident_boundary":"metal-event-v1",
                "host_to_host_boundary":"metal-host-v1",
                "launches":3,"required_bytes":396,"transfer_bytes":400,"workspace_bytes":48}});
    let mut child = parent.clone();
    child["schema_version"] = json!(1);
    child["kind"] = json!("generated_cold_child");
    child["status"] = json!("passed");
    child["generated_result"]["full_output_correct"] = json!(true);
    child["generated_result"]["warmups"] = json!(0);
    child["generated_result"]["samples"] = json!(1);
    child["generated_result"]["device_resident_ms"] = json!([0.1]);
    child["generated_result"]["host_to_host_ms"] = json!([0.2]);
    assert!(validate_cold_child(&parent, &child).is_ok());
    child["generated_result"]["runner_executable_sha256"] = json!("different");
    assert!(
        validate_cold_child(&parent, &child)
            .unwrap_err()
            .contains("runner_executable")
    );
    child["generated_result"]["runner_executable_sha256"] = json!("runner");
    child["generated_result"]["measured_invocation_sha256"] = json!("different");
    assert!(
        validate_cold_child(&parent, &child)
            .unwrap_err()
            .contains("measured_invocation_sha256")
    );
    child["generated_result"]["measured_invocation_sha256"] = json!("invocation");
    child["manifest_sha256"] = json!("different");
    assert!(
        validate_cold_child(&parent, &child)
            .unwrap_err()
            .contains("manifest")
    );

    let path = env::temp_dir().join(format!("invar-cold-malformed-{}", std::process::id()));
    fs::write(&path, b"{").unwrap();
    assert!(
        read_cold_child(&parent, &path, validate_cold_child)
            .unwrap_err()
            .contains("cold child JSON")
    );
    fs::remove_file(path).unwrap();
}

#[test]
fn baseline_cold_child_rejects_bad_output_and_metadata() {
    let parent = json!({"schema_version":2,"manifest_sha256":"manifest",
            "corpus_sha256":"corpus","held_out_sha256":"held",
            "checkout_at_invocation":{"head":"head"},
            "baseline_result":{"candidate_case_id":"reduction-n33","family":"sum",
                "elements":33,"arithmetic_compatibility":"u32_wrapping_v1",
                "fixture_source_sha256":"source","executable_sha256":"exe",
                "build_record_sha256":"build","device_registry_id":1,
                "vendor_versions":{"framework":"MPSGraph"},"input_recipe":"recipe",
                "input_sha256":"input","output_sha256":"output",
                "required_transfer_bytes":136,"workspace_bytes":0,
                "device_resident_boundary":"gpu","host_to_host_boundary":"host"}});
    let mut child = parent.clone();
    child["kind"] = json!("baseline_cold_child");
    child["status"] = json!("passed");
    child["baseline_result"]["full_output_correctness"] = json!("exact_full_output_pass");
    child["baseline_result"]["warmups"] = json!(0);
    child["baseline_result"]["samples"] = json!(1);
    child["baseline_result"]["device_resident_samples_ms"] = json!([0.1]);
    child["baseline_result"]["host_to_host_samples_ms"] = json!([0.2]);
    assert!(validate_baseline_cold_child(&parent, &child).is_ok());
    child["baseline_result"]["output_sha256"] = json!("wrong");
    assert!(
        validate_baseline_cold_child(&parent, &child)
            .unwrap_err()
            .contains("output_sha256")
    );
    child["baseline_result"]["output_sha256"] = json!("output");
    child["baseline_result"]["full_output_correctness"] = json!("exact_full_output_fail");
    assert!(validate_baseline_cold_child(&parent, &child).is_err());
    let path = env::temp_dir().join(format!(
        "invar-baseline-cold-malformed-{}",
        std::process::id()
    ));
    fs::write(&path, b"{").unwrap();
    assert!(
        read_cold_child(&parent, &path, validate_baseline_cold_child)
            .unwrap_err()
            .contains("cold child JSON")
    );
    fs::remove_file(path).unwrap();
}

#[cfg(unix)]
#[test]
fn baseline_cold_process_preserves_failed_attempt_status() {
    let report = env::temp_dir().join(format!(
        "invar-baseline-failure-{}.json",
        std::process::id()
    ));
    let case = json!({"elements":1,"input_buffers":1});
    let parent = json!({"manifest_sha256":"manifest"});
    let result = cold_process_samples(
        &report,
        &case,
        &parent,
        "test",
        |_| {
            let mut child = Command::new("/bin/sh");
            child.args(["-c", "exit 7"]);
            child
        },
        validate_baseline_cold_child,
    )
    .unwrap();
    assert_eq!(result["status"], "pending");
    assert_eq!(result["attempts_completed"], 0);
    assert!(result["reason"].as_str().unwrap().contains("exited"));
    let directory = PathBuf::from(result["raw_directory"].as_str().unwrap());
    let status: Value =
        serde_json::from_slice(&fs::read(directory.join("attempt-00.status.json")).unwrap())
            .unwrap();
    assert_eq!(status["status"], "failed");
    fs::remove_dir_all(directory).unwrap();
}

#[cfg(unix)]
#[test]
fn cold_child_process_bounds_timeout_and_nonzero_exit() {
    let started = Instant::now();
    let mut sleeper = Command::new("/bin/sleep").arg("1").spawn().unwrap();
    assert!(
        wait_bounded(&mut sleeper, started, Duration::from_millis(20))
            .unwrap_err()
            .contains("timed out")
    );
    let started = Instant::now();
    let mut failed = Command::new("/bin/sh")
        .args(["-c", "exit 7"])
        .spawn()
        .unwrap();
    let (status, _) = wait_bounded(&mut failed, started, Duration::from_secs(1)).unwrap();
    assert_eq!(status.code(), Some(7));
}

#[test]
fn comparison_requires_matching_case_provenance_and_boundaries() {
    let samples = json!(vec![1.0; 30]);
    let generated = json!({"manifest_sha256":"manifest","corpus_sha256":"corpus",
            "freeze_status":"provisional-pre-m4-01","checkout_at_invocation":{"worktree":"dirty"},
            "generated_result":{"case_id":"scan-n33","required_bytes":396,"device":"Apple M4",
                "full_output_correct":true,"input_recipe":"lcg-v1","input_sha256":"input",
                "source_observable_output_sha256":"output",
                "device_resident_boundary":"metal-event-v1","host_to_host_boundary":"metal-host-v1",
                "device_resident_ms":samples,"host_to_host_ms":samples},
            "cold_process":{"status":"passed","boundary":"whole-process-v1","samples_ms":samples}});
    let baseline = json!({"manifest_sha256":"manifest","corpus_sha256":"corpus",
            "freeze_status":"provisional-pre-m4-01","gpu_preflight":[{"status":"available","device":"Apple M4"}],
            "baseline_result":{"candidate_case_id":"scan-n33","elements":33,"family":"affine",
                "arithmetic_compatibility":"u32_wrapping_v1","full_output_correctness":"exact_full_output_pass",
                "gate_status":"pending_semantic_quality_build_review","target_compatibility":"pending M4-01 policy",
                "build_provenance":"executable_hash_only","required_transfer_bytes":396,
                "input_recipe":"lcg-v1","input_sha256":"input","output_sha256":"output",
                "device_resident_boundary":"metal-event-v1","host_to_host_boundary":"metal-host-v1",
                "device_resident_samples_ms":samples,"host_to_host_samples_ms":samples}});
    let case = json!({"id":"scan-n33","family":"scan","elements":33,"required_bytes":396});
    let result = baseline_eligibility(&generated, &baseline, &case);
    assert_eq!(result["status"], "pending");
    assert!(result["device_resident_ratio"].is_null());
    assert!(!result["reasons"].as_array().unwrap().iter().any(|reason| {
        reason == "matched_input_and_source_output_digests_missing"
            || reason == "device_or_host_timing_boundary_mismatch"
    }));
    assert!(
        result["reasons"]
            .as_array()
            .unwrap()
            .iter()
            .any(|reason| reason == "provisional_manifest_not_ratio_eligible")
    );
    assert!(
        result["reasons"]
            .as_array()
            .unwrap()
            .iter()
            .any(|reason| reason == "parameterized_extracted_plan_not_source_program")
    );
    let mut claimed = generated.clone();
    let mut claimed_baseline = baseline.clone();
    claimed["cold_process"]["contract_review"] = json!("accepted");
    claimed_baseline["cold_process"]["contract_review"] = json!("accepted");
    let guarded = baseline_eligibility(&claimed, &claimed_baseline, &case);
    assert!(guarded["cold_process_ratio"].is_null());
    assert!(
        guarded["reasons"]
            .as_array()
            .unwrap()
            .iter()
            .any(|reason| reason == "cold_process_scope_review_pending")
    );
    let mut wrong = baseline;
    wrong["baseline_result"]["candidate_case_id"] = json!("scan-n34");
    let rejected = baseline_eligibility(&generated, &wrong, &case);
    assert!(
        rejected["reasons"]
            .as_array()
            .unwrap()
            .iter()
            .any(|reason| reason == "case_or_size_mismatch")
    );
    wrong["baseline_result"]["candidate_case_id"] = json!("scan-n33");
    wrong["baseline_result"]["input_sha256"] = json!("wrong");
    let rejected = baseline_eligibility(&generated, &wrong, &case);
    assert!(
        rejected["reasons"]
            .as_array()
            .unwrap()
            .iter()
            .any(|reason| { reason == "matched_input_and_source_output_digests_missing" })
    );
}

#[test]
fn recorded_metal_scan_and_mpsgraph_baseline_remain_ratio_ineligible() {
    let generated: Value = serde_json::from_slice(include_bytes!(
        "../../../tests/fixtures/archived-evidence/parallel-benchmark-baselines/generated-metal-scan-n33-cold.json"
    ))
    .unwrap();
    let baseline: Value = serde_json::from_slice(include_bytes!(
        "../../../tests/fixtures/archived-evidence/parallel-benchmark-baselines/metal-mpsgraph-affine-n33.report.json"
    ))
    .unwrap();
    let manifest: Value =
        serde_json::from_slice(include_bytes!("../../../bench/manifest-v1.json")).unwrap();
    let case = manifest["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["id"] == "scan-n33")
        .unwrap();
    let comparison = baseline_eligibility(&generated, &baseline, case);
    assert_eq!(comparison["candidate_case_id"], "scan-n33");
    assert_eq!(comparison["status"], "pending");
    assert!(comparison["device_resident_ratio"].is_null());
    assert!(
        comparison["reasons"]
            .as_array()
            .unwrap()
            .iter()
            .any(|reason| reason == "baseline_semantic_target_or_build_review_pending")
    );
}

#[test]
fn ordinary_three_stage_map_publishes_dead_intermediate_outputs() {
    use invar::{admission, check::check, device_ir, eval, front::parse, mir};
    let source = include_str!("../../../bench/map-output-elision-repro.invar");
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    assert_eq!(
        eval::observe(&checked, 100_000).result,
        Ok(eval::Value::I32(0))
    );
    let program = mir::lower(&checked).unwrap();
    let admitted = admission::analyze(source.as_bytes(), &program).unwrap();
    let dag = admitted
        .regions()
        .iter()
        .filter_map(|region| device_ir::from_region(region, 33).ok())
        .find_map(|ir| ir.dag)
        .expect("ordinary source must lower to a map DAG");
    assert_eq!(dag.inputs.len(), 2);
    assert_eq!(
        dag.outputs.len(),
        3,
        "current lowering publishes a and b as well as c"
    );
    assert_eq!(dag.stage_outputs.len(), 3);
}
