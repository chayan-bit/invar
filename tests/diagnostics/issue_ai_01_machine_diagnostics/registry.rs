fn valid_registry_entry(entry: &Value) -> bool {
    match entry["reachability"].as_str() {
        Some("source") => entry.get("justification").is_none() && entry.get("fault_test").is_none(),
        Some("invariant") => {
            entry["justification"]
                .as_str()
                .is_some_and(|sentence| sentence.len() > 20 && sentence.ends_with('.'))
                && entry["fault_test"]
                    .as_str()
                    .is_some_and(|name| name.starts_with("issue_ai_01_machine_diagnostics_"))
        }
        Some("external") => {
            entry["justification"]
                .as_str()
                .is_some_and(|sentence| sentence.len() > 20 && sentence.ends_with('.'))
                && entry["boundary_test"].as_str().is_some_and(|name| {
                    name.starts_with("issue_m4_06_bounded_schedule_tuning_")
                        || name.starts_with("issue_v02_placement_")
                        || name.starts_with("issue_s1_11_robotics_vocabulary_")
                        || name.starts_with("issue_s1_02_foreign_manifests_")
                        || name.starts_with("issue_l1_06_python_binding_")
                        || name.starts_with("issue_m5_02_full_pipeline_qualification_")
                        || name.starts_with("issue_ai_01_machine_diagnostics_qualification_")
                        || (name.starts_with("test_")
                            && include_str!("../../../crates/invar-python/tests/test_binding.py")
                                .contains(&format!("def {name}(")))
                })
                && entry.get("fault_test").is_none()
        }
        _ => false,
    }
}

fn document_codes(value: &Value, registered: &HashSet<String>, found: &mut HashSet<String>) {
    match value {
        Value::String(code) if registered.contains(code) => {
            found.insert(code.clone());
        }
        Value::Array(values) => {
            for value in values {
                document_codes(value, registered, found);
            }
        }
        Value::Object(values) => {
            for value in values.values() {
                document_codes(value, registered, found);
            }
        }
        _ => {}
    }
}

fn codes_registered(document: &Value, codes: &HashSet<String>) -> bool {
    document["diagnostics"]
        .as_array()
        .is_some_and(|diagnostics| {
            diagnostics.iter().all(|diagnostic| {
                diagnostic["code"]
                    .as_str()
                    .is_some_and(|code| codes.contains(code))
            })
        })
}

fn invariant_fixes_absent(document: &Value, invariants: &HashSet<String>) -> bool {
    document["diagnostics"]
        .as_array()
        .is_some_and(|diagnostics| {
            diagnostics.iter().all(|diagnostic| {
                !diagnostic["code"]
                    .as_str()
                    .is_some_and(|code| invariants.contains(code))
                    || diagnostic["suggested_fixes"] == json!([])
            })
        })
}

fn candidate_code_literals(source: &str) -> HashSet<String> {
    let source = source.replace("'\"'", "'\\u{22}'");
    let mut result = HashSet::new();
    let mut quoted = false;
    let mut offset = 0;
    for part in source.split('"') {
        if quoted {
            let prefix = source[..offset - 1].trim_end();
            let callee = prefix
                .strip_suffix('(')
                .map(str::trim_end)
                .and_then(|prefix| {
                    prefix
                        .rsplit(|c: char| {
                            !(c.is_ascii_alphanumeric() || matches!(c, '_' | '!' | ':'))
                        })
                        .next()
                });
            let callee = callee.map(|name| name.trim_start_matches(':'));
            let environment_key = matches!(
                callee,
                Some("std::env::var" | "std::env::var_os" | "env::var" | "env::var_os")
            ) || matches!(
                callee.and_then(|name| name.rsplit("::").next()),
                Some("env!" | "option_env!")
            ) || prefix.ends_with(".env(")
                || prefix.ends_with(".env_remove(");
            // Byte strings are binary data, not the &str diagnostic code ABI.
            // Include raw byte strings; keep ordinary/raw str codes covered.
            let literal_prefix = &source[..offset - 1];
            let byte_string = literal_prefix
                .strip_suffix('b')
                .or_else(|| literal_prefix.trim_end_matches('#').strip_suffix("br"))
                .is_some_and(|before| {
                    before
                        .chars()
                        .next_back()
                        .is_none_or(|c| !(c.is_alphanumeric() || c == '_'))
                });
            if !environment_key
                && !byte_string
                && part.len() >= 3
                // Match the diagnostic schema's leading-letter requirement.
                && part.bytes().next().is_some_and(|b| b.is_ascii_uppercase())
                && part
                    .bytes()
                    .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
            {
                result.insert(part.to_owned());
            }
        }
        quoted = !quoted;
        offset += part.len() + 1;
    }
    result
}

#[test]
fn issue_ai_01_machine_diagnostics_literal_scan_rejects_numeric_arguments() {
    // Numeric subprocess arguments are not diagnostic identifiers. Keep all
    // identifier-shaped candidates subject to the registry comparison.
    let source = r#"Command::new("head").args(["-c", "65537", "/dev/zero"]);
        boundary("S1600", "message");
        boundary("NEW_CODE9", "message");"#;
    assert_eq!(
        candidate_code_literals(source),
        HashSet::from(["S1600".to_owned(), "NEW_CODE9".to_owned()])
    );
    assert!(candidate_code_literals(r#""123" "0000" "___" "12_34" "12A""#).is_empty());
}

#[test]
fn issue_ai_01_machine_diagnostics_literal_scan_distinguishes_binary_magic() {
    let source = r####"let magic = b"BINARY_MAGIC";
        let raw_magic = br###"RAW_BINARY_MAGIC"###;
        boundary("BINARY_MAGIC", "message");
        boundary(r#"RAW_CODE"#, "message");
        boundary_tokens!(b "MACRO_CODE");"####;
    assert_eq!(
        candidate_code_literals(source),
        HashSet::from([
            "BINARY_MAGIC".to_owned(),
            "RAW_CODE".to_owned(),
            "MACRO_CODE".to_owned(),
        ])
    );
}

#[test]
fn issue_ai_01_machine_diagnostics_literal_scan_distinguishes_environment_macros() {
    let source = r#"let build = env!("INVAR_BUILD_DIGEST");
        let optional = std::option_env! ( "OPTIONAL_BUILD_KEY" );
        let directory = std::env::var_os("INVAR_TRANSPOSE_REPORT_DIR");
        let value = env::var ( "OTHER_ENVIRONMENT_KEY" );
        let record = json!({"host_build":env!("INVAR_BUILD_DIGEST")});
        command.env("INVAR_GNET9_CHILD_PID", "child.pid");
        boundary("NATIVE_ABI", "message");"#;
    assert_eq!(
        candidate_code_literals(source),
        HashSet::from(["NATIVE_ABI".to_owned()])
    );
    // The same spelling used as an actual error remains subject to registry
    // coverage. This is a syntactic macro distinction, not a code-name exemption.
    assert!(
        candidate_code_literals(r#"boundary("INVAR_BUILD_DIGEST", "message")"#)
            .contains("INVAR_BUILD_DIGEST")
    );
    assert!(
        candidate_code_literals(r#"boundary("INVAR_TRANSPOSE_REPORT_DIR", "message")"#)
            .contains("INVAR_TRANSPOSE_REPORT_DIR")
    );
}

#[test]
fn issue_ai_01_machine_diagnostics_registry_covers_emitted_literals() {
    fn visit(path: &std::path::Path, literals: &mut HashSet<String>) {
        for entry in std::fs::read_dir(path).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(&path, literals);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                let source = std::fs::read_to_string(path).unwrap();
                literals.extend(candidate_code_literals(&source));
            }
        }
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut literals = HashSet::new();
    visit(&root.join("src"), &mut literals);
    visit(&root.join("crates"), &mut literals);
    // M5-02 qualification surfaces its codes through `invar-qualify`: scan all of it.
    visit(&root.join("tools/src"), &mut literals);
    visit(&root.join("tools/qualification"), &mut literals);
    for environment_name in [
        "HOME",
        "CARGO_MANIFEST_DIR",
        "RUSTC",
        "CXX",
        "LOCALAPPDATA",
        "XDG_CACHE_HOME",
        "INVAR_CACHE_DIR",
        "INVAR_NVRTC_HELPER",
        "INVAR_PROFILE_PHASES",
        // SPDX identifiers used by the system license checker, not diagnostic codes.
        "MIT",
        "ISC",
        // Source asset license filename, not a diagnostic code.
        "LICENSE",
        // ROS QoS enum spellings in generated configuration.
        "KEEP_LAST",
        "RELIABLE",
        // Private sequencer framing magic, not a diagnostic code.
        "IVSQ",
        // ROS IDL scalar suffixes in generated message names.
        "I32",
        "U32",
        "F32",
        "F64",
        // Benchmark provenance uses Git's symbolic revision, not a diagnostic code.
        "HEAD",
        // Benchmark toolchain identification prefix, not an emitted diagnostic.
        "NVRTC",
        // Backend display name in shared core-family scheduling messages.
        "CUDA",
        // rustc error codes asserted by the host-array compile-fail harness
        // (crates/invar-foreign/src/host_array/tests.rs), and that harness's
        // marker for a rustc error without a code. Not Invar diagnostics.
        "E0277",
        "E0451",
        "E0505",
        "E0506",
        "E0507",
        "E0597",
        "E0599",
        "E0616",
        "E0624",
        "UNCODED",
        // Environment variable names passed to the ABI12 S2 CPU model's
        // subprocess (src/general_net/cuda_owned_numeric_host/cpu.rs).
        "PATH",
        "TMPDIR",
        // Address-space macro the MSL emitter writes into generated Metal
        // source (crates/invar-math/src/msl.rs), not a diagnostic code.
        "INVAR_CONSTANT",
        // Compiled-ahead Metal: container and metallib magics, and the
        // developer source-path flag and test tripwire environment names.
        "INVGMETL",
        "INVGMBA1",
        "MTLB",
        "INVAR_METAL_RUNTIME_SOURCE",
        "INVAR_METAL_SOURCE_TRIPWIRE",
        // Developer runtime-NVRTC flag for forced CUDA without an image.
        "INVAR_CUDA_RUNTIME_NVRTC",
    ] {
        literals.remove(environment_name);
    }
    let registry = code_registry();
    let fault_test_sources = [
        include_str!("../../../src/admission.rs"),
        include_str!("../../../src/engine.rs"),
        include_str!("../../../src/eval.rs"),
        include_str!("../../../src/eval/host_read.rs"),
        include_str!("../../../src/eval/host_read/tests.rs"),
        include_str!("../../../src/eval/floating.rs"),
        include_str!("../../../src/main.rs"),
        include_str!("../../../crates/invar-python/src/tests.rs"),
        include_str!("../../../crates/invar-python/src/tests/bridge_faults.rs"),
        include_str!("../../../src/mir_eval.rs"),
        include_str!("../../../src/recurrence.rs"),
        include_str!("../../../src/schedule/cuda.rs"),
        include_str!("../../../src/schedule/metal.rs"),
        include_str!("../../../src/schedule/workers.rs"),
        include_str!("../../../src/schedule/families/grid.rs"),
        include_str!("../../../src/summary_source/host.rs"),
        include_str!("../../../src/indexed_source/host.rs"),
        include_str!("../../../src/scatter_proof/host.rs"),
        include_str!("../../../src/histogram_source/host.rs"),
        include_str!("../../../src/mir_eval/bulk.rs"),
        include_str!("../../../src/mir_eval/gather.rs"),
        include_str!("../../../src/mir_eval/scatter.rs"),
        include_str!("../../../src/mir_eval/histogram.rs"),
        include_str!("../../../tools/tests/qualification_faults.rs"),
    ];
    for entry in registry["codes"].as_array().unwrap() {
        assert!(valid_registry_entry(entry), "invalid reachability: {entry}");
        if entry["reachability"] == "invariant" {
            let test = format!("fn {}(", entry["fault_test"].as_str().unwrap());
            assert!(
                fault_test_sources
                    .iter()
                    .chain(OWN_SOURCES.iter())
                    .any(|source| source.contains(&test)),
                "missing fault test: {entry}"
            );
            let diagnostic = diagnostics::diagnostic(
                entry["code"].as_str().unwrap(),
                "injected invariant failure",
                "fixture.inv",
                b"fn main() -> i32 { return 0; }",
                None,
            );
            assert_eq!(diagnostic["suggested_fixes"], json!([]), "{entry}");
        } else if entry["reachability"] == "external" {
            let test = format!("fn {}(", entry["boundary_test"].as_str().unwrap());
            assert!(
                include_str!("../../issue_m4_06_bounded_schedule_tuning.rs").contains(&test)
                    || include_str!("../../../tools/tests/issue_m5_02_device_qualification.rs")
                        .contains(&test)
                    || OWN_SOURCES.iter().any(|source| source.contains(&test))
                    || include_str!("../../../tools/tests/qualification_faults.rs").contains(&test)
                    || include_str!("../../v02/v02_placement.rs").contains(&test)
                    || include_str!("../../robotics/issue_s1_11_robotics_vocabulary.rs").contains(&test)
                    || include_str!("../../robotics/issue_s1_02_foreign_manifests.rs").contains(&test)
                    || include_str!("../../../crates/invar-python/src/tests.rs").contains(&test)
                    || include_str!("../../../crates/invar-python/src/tests/bridge_faults.rs")
                        .contains(&test)
                    || include_str!("../../../crates/invar-python/tests/test_binding.py").contains(
                        &format!("def {}(", entry["boundary_test"].as_str().unwrap())
                    ),
                "missing external boundary test: {entry}"
            );
            let diagnostic = diagnostics::diagnostic(
                entry["code"].as_str().unwrap(),
                "external conversion failure",
                "<external>",
                b"",
                None,
            );
            assert_eq!(diagnostic["suggested_fixes"], json!([]), "{entry}");
        }
    }
    let codes = registry["codes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["code"].as_str().unwrap().to_owned())
        .collect::<HashSet<_>>();
    assert_eq!(
        literals, codes,
        "update the versioned registry with source changes"
    );
}

#[test]
fn issue_ai_01_machine_diagnostics_indexed_admission_binding_fault() {
    let source = include_bytes!("../../fixtures/a1-indexed/gather.inv");
    let stale = std::str::from_utf8(source).unwrap().replace("7u32", "8u32");
    assert_ne!(source.as_slice(), stale.as_bytes());
    let program = invar::mir::lower(
        &invar::check::check(invar::front::parse(stale.as_bytes()).unwrap(), true).unwrap(),
    )
    .unwrap();
    let error = invar::schedule::run_source(source, &program, 1_000_000, 2).unwrap_err();
    assert_eq!(error, invar::schedule::Error::Candidate);
}
