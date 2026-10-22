const INDEXED_FAMILIES: [&str; 4] = ["GATHER_", "SCATTER_", "HISTOGRAM_", "BULK_"];

fn indexed_registry_codes() -> HashSet<String> {
    let registry = schema("diagnostic-codes-indexed-v1.json");
    assert_eq!(registry["schema"], "invar-diagnostic-codes-indexed-v1");
    assert_eq!(registry["extends"], "invar-diagnostic-codes-v1");
    registry["codes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| {
            assert_eq!(entry["deprecated"], false, "{entry}");
            assert_eq!(entry["reachability"], "invariant", "{entry}");
            assert!(valid_registry_entry(entry), "{entry}");
            entry["code"].as_str().unwrap().to_owned()
        })
        .collect()
}

#[test]
fn issue_ai_01_machine_diagnostics_indexed_registry_matches_family_sources() {
    fn visit(path: &std::path::Path, found: &mut Vec<(PathBuf, HashSet<String>)>) {
        for entry in std::fs::read_dir(path).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(&path, found);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                let literals = candidate_code_literals(&std::fs::read_to_string(&path).unwrap())
                    .into_iter()
                    .filter(|code| INDEXED_FAMILIES.iter().any(|f| code.starts_with(f)))
                    .collect::<HashSet<_>>();
                if !literals.is_empty() {
                    found.push((path, literals));
                }
            }
        }
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut found = Vec::new();
    visit(&root.join("src"), &mut found);
    visit(&root.join("crates"), &mut found);
    let mut files = found
        .iter()
        .map(|(path, _)| {
            path.strip_prefix(root)
                .unwrap()
                .to_str()
                .unwrap()
                .to_owned()
        })
        .collect::<Vec<_>>();
    files.sort_unstable();
    // Every indexed-family literal lives in a host or its publication module.
    assert_eq!(
        files,
        [
            "src/histogram_source/host.rs",
            "src/indexed_source/host.rs",
            "src/mir_eval/bulk.rs",
            "src/mir_eval/gather.rs",
            "src/mir_eval/histogram.rs",
            "src/mir_eval/scatter.rs",
            "src/scatter_proof/host.rs",
        ]
    );
    let literals = found
        .into_iter()
        .flat_map(|(_, literals)| literals)
        .collect::<HashSet<_>>();
    let registered = indexed_registry_codes();
    assert_eq!(literals, registered);
    let counts = INDEXED_FAMILIES.map(|family| {
        registered
            .iter()
            .filter(|code| code.starts_with(family))
            .count()
    });
    assert_eq!(counts, [12, 12, 15, 1]);
    let tests = schema("diagnostic-codes-indexed-v1.json")["codes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["fault_test"].as_str().unwrap().to_owned())
        .collect::<HashSet<_>>();
    assert_eq!(
        tests,
        [
            "issue_ai_01_machine_diagnostics_gather_publication_faults",
            "issue_ai_01_machine_diagnostics_gather_publication_harness_faults",
            "issue_ai_01_machine_diagnostics_scatter_publication_faults",
            "issue_ai_01_machine_diagnostics_scatter_publication_harness_faults",
            "issue_ai_01_machine_diagnostics_histogram_publication_faults",
            "issue_ai_01_machine_diagnostics_histogram_publication_harness_faults",
            "issue_ai_01_machine_diagnostics_bulk_ambiguous_completion",
        ]
        .map(str::to_owned)
        .into()
    );
    // The combined registry carries each indexed code exactly once.
    let combined = code_registry();
    for code in &registered {
        assert_eq!(
            combined["codes"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|entry| entry["code"] == code.as_str())
                .count(),
            1,
            "{code}"
        );
    }
}

#[test]
fn issue_ai_01_machine_diagnostics_indexed_unknown_codes_rejected() {
    let diagnostic_schema = schema("invar-diagnostics-v1.schema.json");
    let codes = registry_codes();
    let invariants = invariant_codes();
    let document_for = |code: &str| {
        diagnostics::document(
            "tool_error",
            vec![diagnostics::diagnostic(
                code,
                "indexed publication failure",
                "indexed.inv",
                b"",
                None,
            )],
        )
    };
    for code in indexed_registry_codes() {
        let document = document_for(&code);
        assert!(
            jsonschema::is_valid(&diagnostic_schema, &document),
            "{document}"
        );
        assert!(codes_registered(&document, &codes), "{document}");
        assert!(invariants.contains(&code), "{code}");
        assert!(invariant_fixes_absent(&document, &invariants), "{document}");
        let mut fixed = document.clone();
        fixed["diagnostics"][0]["suggested_fixes"] = json!([{"message": "retry", "edits": []}]);
        assert!(!invariant_fixes_absent(&fixed, &invariants), "{fixed}");
    }
    // Schema-valid but unregistered: plausible family spellings, cross-family
    // suffixes, truncations and extensions are all rejected by the registry.
    for code in [
        "GATHER_UNREGISTERED",
        "GATHER_EXTENT",
        "GATHER_METER",
        "SCATTER_FRESH_OUTPUT",
        "SCATTER_PRIVATE_ALLOCATION",
        "HISTOGRAM_STALE_COMPLETION",
        "HISTOGRAM_FUEL_RECIPE",
        "BULK_AMBIGUOUS",
        "BULK_AMBIGUOUS_COMPLETION_V2",
        "GATHER",
        "HISTOGRAM_BUDGET_",
    ] {
        let document = document_for(code);
        assert!(
            jsonschema::is_valid(&diagnostic_schema, &document),
            "{document}"
        );
        assert!(!codes_registered(&document, &codes), "{code}");
    }
    let mut mixed = document_for("GATHER_OWNER");
    mixed["diagnostics"]
        .as_array_mut()
        .unwrap()
        .push(document_for("SCATTER_UNREGISTERED")["diagnostics"][0].clone());
    assert!(!codes_registered(&mixed, &codes), "{mixed}");
    // Entries without an invariant justification or fault test are invalid.
    let entry = schema("diagnostic-codes-indexed-v1.json")["codes"][0].clone();
    for (field, value) in [
        ("fault_test", json!(null)),
        ("fault_test", json!("gather_publication_faults")),
        ("justification", json!("short")),
        ("reachability", json!("source")),
    ] {
        let mut changed = entry.clone();
        changed[field] = value;
        assert!(!valid_registry_entry(&changed), "{changed}");
    }
}

/// Schema and registration check of real CLI documents for indexed-shaped
/// programs. The CLI never runs the indexed hosts, so none of the 40 indexed
/// codes is exercised here; their real coverage is the named fault tests. The
/// `cpu-parallel` runs and the prefix scan guard the scheduler publication path
/// (review3 R1), where a valid program must not surface an indexed code.
#[test]
fn issue_ai_01_machine_diagnostics_indexed_cli_documents_validate() {
    let diagnostic_schema = schema("invar-diagnostics-v1.schema.json");
    let codes = registry_codes();
    let invariants = invariant_codes();
    for (name, source) in [
        (
            "gather",
            &b"fn gather(a: read Array<u32>, ix: read Array<u32>) -> Array<u32> { var out = filled_u32(len(ix),0u32); for i in 0u32..len(ix) { out[i] = a[ix[i]]; } return out; } fn main() -> i32 { let a: Array<u32> = [7u32,11u32]; let ix: Array<u32> = [1u32,0u32,1u32]; let out = gather(a,ix); return 7; }"[..],
        ),
        (
            "gather-trap",
            &b"fn gather(a: read Array<u32>, ix: read Array<u32>) -> Array<u32> { var out = filled_u32(len(ix),0u32); for i in 0u32..len(ix) { out[i] = a[ix[i]]; } return out; } fn main() -> i32 { let a: Array<u32> = [7u32,11u32]; let ix: Array<u32> = [1u32,2u32,1u32]; let out = gather(a,ix); return 7; }"[..],
        ),
        (
            "scatter",
            &b"fn scatter(a: read Array<u32>) -> Array<u32> { var out = filled_u32(5u32,0u32); for i in 0u32..3u32 { out[2u32*i] = a[i]; } return out; } fn main() -> i32 { let a: Array<u32> = [7u32,11u32,19u32]; let out = scatter(a); return 7; }"[..],
        ),
        (
            "histogram",
            &b"fn h(a: read Array<u32>, x: read Array<u32>, n: u32) -> Array<u32> { var o = filled_u32(n,0u32); for i in 0u32..len(x) { o[x[i]] = wrapping_add(o[x[i]],a[i]); } return o; } fn main() -> i32 { let a: Array<u32> = [0u32]; let o = h(a,a,1u32); return 7; }"[..],
        ),
        (
            "prefix-scan",
            &b"fn main() -> i32 { let aa: Array<u32> = [2u32,3u32,5u32]; let bb: Array<u32> = [7u32,11u32,13u32]; var state: u32 = 17u32; var out: Array<u32> = filled_u32(len(aa),0u32); for i in 0u32 .. len(aa) { state = wrapping_add(wrapping_mul(aa[i],state),bb[i]); out[i] = state; } if out[2u32] == 683u32 { return 0; } return 1; }"[..],
        ),
    ] {
        let path = fixture(name, source);
        let path_str = path.to_str().unwrap();
        let trap = name.ends_with("-trap");
        for (args, exit) in [
            (vec!["check", path_str, "--format", "json"], 0),
            (
                vec![
                    "run",
                    path_str,
                    "--engine",
                    "reference",
                    "--format",
                    "json",
                ],
                if trap { 3 } else { 0 },
            ),
            (
                vec![
                    "run",
                    path_str,
                    "--engine",
                    "cpu-parallel",
                    "--workers",
                    "2",
                    "--format",
                    "json",
                ],
                if trap { 3 } else { 0 },
            ),
            (
                vec![
                    "run",
                    path_str,
                    "--engine",
                    "reference",
                    "--fuel",
                    "1",
                    "--format",
                    "json",
                ],
                4,
            ),
        ] {
            let (status, document) = cli(&args);
            assert_eq!(status, exit, "{args:?}: {document}");
            assert!(
                jsonschema::is_valid(&diagnostic_schema, &document),
                "{args:?}: {document}"
            );
            assert!(codes_registered(&document, &codes), "{args:?}: {document}");
            assert!(
                invariant_fixes_absent(&document, &invariants),
                "{args:?}: {document}"
            );
            // A failing run reports errors; a successful run may carry only
            // notes (for example NO_ADMITTED_CANDIDATE on serial fallback).
            for diagnostic in document["diagnostics"].as_array().unwrap() {
                let severity = if exit == 0 { "note" } else { "error" };
                assert_eq!(diagnostic["severity"], severity, "{args:?}: {document}");
            }
            if document["diagnostics"].get(0).is_some() {
                // The same machine-readable document with an indexed-family
                // code that is not registered stays schema-valid but is refused.
                let mut forged = document.clone();
                forged["diagnostics"][0]["code"] = json!("GATHER_UNREGISTERED");
                assert!(jsonschema::is_valid(&diagnostic_schema, &forged), "{forged}");
                assert!(!codes_registered(&forged, &codes), "{forged}");
            }
            if trap && args[0] == "run" && exit == 3 {
                assert_eq!(document["status"], "trap", "{document}");
            }
            if args[0] == "run" && exit == 0 {
                assert_eq!(document["status"], "ok", "{args:?}: {document}");
            }
        }
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
fn issue_ai_01_machine_diagnostics_schema_digests_are_frozen() {
    // A digest changes only with an approved entry in docs/contract-changes.md
    // (2026-10-19: numeric_contracts and metal-no-f64 for strict-f64-v1, #173;
    // numeric_contracts.trig for invar-trig-f32-v1, #173 under the C4 interim;
    // 2026-10-20: normal_form for the P6(a) normal-form route, #263;
    // 2026-10-22: public-repository $id URLs).
    fn fnv1a(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        })
    }
    assert_eq!(
        fnv1a(include_bytes!(
            "../../schema/invar-diagnostics-v1.schema.json"
        )),
        0xdfa2_8d61_6759_b86d
    );
    assert_eq!(
        fnv1a(include_bytes!("../../schema/explain-v1.schema.json")),
        0xd64e_62b0_04f4_341c
    );
    assert_eq!(
        fnv1a(include_bytes!(
            "../../schema/invar-explain-device-v1.schema.json"
        )),
        0x3be8_b64f_24eb_50c9
    );
}

#[test]
fn issue_ai_01_machine_diagnostics_v2_conversion_trap_is_versioned() {
    let v1 = schema("invar-diagnostics-v1.schema.json");
    let v2 = schema("invar-diagnostics-v2.schema.json");
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/r1_01/inexact-conversion.inv");
    let path = path.to_str().unwrap();
    let (status, document) = cli(&["run", path, "--engine", "reference", "--format", "json"]);
    assert_eq!(status, 3, "{document}");
    assert_eq!(document["schema"], "invar-diagnostics-v2");
    assert_eq!(document["schema_version"], 2);
    assert_eq!(document["trap"]["category"], "UNIT_CONVERSION_INEXACT");
    assert!(jsonschema::is_valid(&v2, &document));
    assert!(!jsonschema::is_valid(&v1, &document));
    let mut old_identity = document.clone();
    old_identity["schema"] = "invar-diagnostics-v1".into();
    old_identity["schema_version"] = 1.into();
    assert!(!jsonschema::is_valid(&v1, &old_identity));
    let old_document = invar::diagnostics::document("ok", vec![]);
    assert!(jsonschema::is_valid(&v1, &old_document));
    #[cfg(feature = "native")]
    {
        let (status, native) = cli(&["run", path, "--engine", "native-serial", "--format", "json"]);
        assert_eq!(status, 3, "{native}");
        assert_eq!(native["schema"], "invar-diagnostics-v2");
        assert!(jsonschema::is_valid(&v2, &native));
    }
}

#[test]
fn issue_ai_01_machine_diagnostics_sort_calibration_and_reason_registry() {
    let v3 = schema("explain-v3.schema.json");
    let v2 = schema("explain-v2.schema.json");
    let old: Value =
        serde_json::from_str(include_str!("../fixtures/v02-sort/cpu-demo.json")).unwrap();
    assert!(jsonschema::is_valid(&v2, &old));
    let registry = code_registry();
    let reasons = registry["explain_reason_codes"].as_array().unwrap();
    for (ty, values) in [
        ("u32", "2u32,1u32"),
        ("i32", "2i32,-1i32"),
        ("f32", "2.0,-1.0"),
    ] {
        let path = std::env::temp_dir().join(format!(
            "invar-sort-calibration-{}-{ty}.inv",
            std::process::id()
        ));
        std::fs::write(&path, format!("fn main() -> i32 {{ var xs: Array<{ty}> = [{values}]; var ids: Array<u32> = [0u32,1u32]; sort(xs); sort_by_key(xs,ids); let order = argsort(xs); let offsets: Array<u32> = [0u32,2u32]; segmented_sort(xs,offsets); return 7; }}")).unwrap();
        let (status, document) = cli(&["explain", path.to_str().unwrap(), "--format", "json"]);
        std::fs::remove_file(path).unwrap();
        assert_eq!(status, 0, "{document}");
        assert_eq!(document["schema"], "invar-explain-v3");
        assert!(jsonschema::is_valid(&v3, &document), "{document}");
        assert!(!jsonschema::is_valid(&v2, &document));
        let sites = document["explicit_sorts"].as_array().unwrap();
        assert_eq!(sites.len(), 4);
        for (site, op) in sites
            .iter()
            .zip(["sort", "sort_by_key", "argsort", "segmented_sort"])
        {
            assert_eq!(site["calibration_id"], format!("sort.{op}.{ty}"));
            assert!(
                reasons.contains(&site["reason"]),
                "unregistered sort reason: {site}"
            );
        }
        let mut missing = document.clone();
        missing["explicit_sorts"][0]
            .as_object_mut()
            .unwrap()
            .remove("calibration_id");
        assert!(!jsonschema::is_valid(&v3, &missing));
        let mut wrong = document;
        wrong["explicit_sorts"][0]["calibration_id"] = "sort.sort.f64".into();
        assert!(!jsonschema::is_valid(&v3, &wrong));
    }
}

#[test]
fn issue_ai_01_machine_diagnostics_sorting_f64_program_records_numeric_contracts_in_v3() {
    let path = std::env::temp_dir().join(format!("invar-sort-f64-{}.inv", std::process::id()));
    std::fs::write(&path, "fn main() -> i32 { var xs: Array<u32> = [2u32,1u32]; sort(xs); let x = 1.5f64; if xs[0u32] == 1u32 and sqrt(x * x) == x { return 7; } return 0; }").unwrap();
    let (status, document) = cli(&["explain", path.to_str().unwrap(), "--format", "json"]);
    std::fs::remove_file(path).unwrap();
    assert_eq!(status, 0, "{document}");
    assert_eq!(document["schema"], "invar-explain-v3");
    assert_eq!(document["numeric_contracts"]["binary64"], "strict-f64-v1");
    assert_eq!(document["numeric_contracts"]["math"], "invar-math-cr-v1");
    assert!(
        jsonschema::is_valid(&schema("explain-v3.schema.json"), &document),
        "{document}"
    );
}
