#[test]
fn issue_ai_01_machine_diagnostics_positions_and_fix() {
    let text = "/* λ */\r\nfn main() -> i32 { return missing; }\r\n";
    let path = fixture("utf8-crlf", text.as_bytes());
    let (status, document) = cli(&["check", path.to_str().unwrap(), "--format", "json"]);
    assert_eq!(status, 1);
    let span = &document["diagnostics"][0]["span"];
    let start = text.find("missing").unwrap();
    assert_eq!(span["start"], start);
    assert_eq!(span["end"], start + "missing".len());
    assert_eq!(span["start_line"], 2);
    assert_eq!(span["start_column"], 27);
    assert_eq!(span["end_column"], 34);
    assert_eq!(span["path"], path.to_str().unwrap());
    std::fs::remove_file(path).unwrap();

    let source = b"fn library() -> i32 { return 7; }";
    let path = fixture("fix", source);
    let (status, document) = cli(&[
        "run",
        path.to_str().unwrap(),
        "--engine",
        "reference",
        "--format",
        "json",
    ]);
    assert_eq!(status, 1);
    assert_eq!(document["diagnostics"][0]["code"], "ENTRYPOINT_MISSING");
    assert!(jsonschema::is_valid(
        &schema("invar-diagnostics-v1.schema.json"),
        &document
    ));
    let edits = document["diagnostics"][0]["suggested_fixes"][0]["edits"]
        .as_array()
        .unwrap();
    assert_eq!(edits.len(), 1);
    assert!(diagnostics::valid_edit_spans(
        &[Span {
            start: source.len(),
            end: source.len(),
        }],
        source.len()
    ));
    assert!(!diagnostics::valid_edit_spans(
        &[Span { start: 0, end: 2 }, Span { start: 1, end: 3 }],
        source.len()
    ));
    let at = edits[0]["span"]["start"].as_u64().unwrap() as usize;
    assert_eq!(at, source.len());
    let mut edited = source.to_vec();
    edited.splice(
        at..at,
        edits[0]["replacement"]
            .as_str()
            .unwrap()
            .as_bytes()
            .iter()
            .copied(),
    );
    std::fs::write(&path, edited).unwrap();
    let (status, _) = cli(&["check", path.to_str().unwrap(), "--format", "json"]);
    assert_eq!(status, 0);
    let (status, document) = cli(&[
        "run",
        path.to_str().unwrap(),
        "--engine",
        "reference",
        "--format",
        "json",
    ]);
    assert_eq!(status, 0);
    assert_eq!(document["result"]["value"], 0);
    std::fs::remove_file(path).unwrap();

    let path = fixture("unfixable", b"fn helper() -> i32 { return missing; }");
    let (status, document) = cli(&[
        "run",
        path.to_str().unwrap(),
        "--engine",
        "reference",
        "--format",
        "json",
    ]);
    assert_eq!(status, 1);
    assert_eq!(document["diagnostics"][0]["code"], "ENTRYPOINT_MISSING");
    assert_eq!(document["diagnostics"][0]["suggested_fixes"], json!([]));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn issue_ai_01_machine_diagnostics_reject_invalid_contracts() {
    let registry = code_registry();
    let codes = registry["codes"].as_array().unwrap();
    assert_eq!(
        registry_codes().len(),
        codes.len(),
        "duplicate registry code"
    );
    let mut missing_reachability = codes[0].clone();
    missing_reachability
        .as_object_mut()
        .unwrap()
        .remove("reachability");
    assert!(!valid_registry_entry(&missing_reachability));
    let invariant = codes
        .iter()
        .find(|entry| entry["reachability"] == "invariant")
        .unwrap();
    for field in ["justification", "fault_test"] {
        let mut missing = invariant.clone();
        missing.as_object_mut().unwrap().remove(field);
        assert!(!valid_registry_entry(&missing), "missing {field}");
    }
    let mut invariant_with_fix = diagnostics::document(
        "tool_error",
        vec![diagnostics::diagnostic(
            invariant["code"].as_str().unwrap(),
            "injected invariant",
            "fixture.inv",
            b"",
            None,
        )],
    );
    invariant_with_fix["diagnostics"][0]["suggested_fixes"] = json!([{
        "description": "invalid invariant fix",
        "edits": [{"span": {"path":"fixture.inv","start":0,"end":0,"start_line":1,"start_column":1,"end_line":1,"end_column":1},"replacement":""}]
    }]);
    assert!(!invariant_fixes_absent(
        &invariant_with_fix,
        &invariant_codes()
    ));
    let unregistered = diagnostics::document(
        "tool_error",
        vec![diagnostics::diagnostic(
            "UNREGISTERED_EXAMPLE",
            "unregistered",
            "x.inv",
            b"",
            None,
        )],
    );
    assert!(!codes_registered(&unregistered, &registry_codes()));
    let diagnostic_schema = schema("invar-diagnostics-v1.schema.json");
    assert!(jsonschema::is_valid(
        &diagnostic_schema,
        &invariant_with_fix
    ));
    let mut invalid = diagnostics::document(
        "tool_error",
        vec![diagnostics::diagnostic(
            "FILE_READ",
            "unreadable",
            "x.inv",
            b"",
            None,
        )],
    );
    assert!(jsonschema::is_valid(&diagnostic_schema, &invalid));
    invalid["diagnostics"][0]["unexpected"] = json!(true);
    assert!(!jsonschema::is_valid(&diagnostic_schema, &invalid));
    assert!(diagnostics::location("x.inv", "λ".as_bytes(), Span { start: 1, end: 2 }).is_none());
    assert!(diagnostics::location("x.inv", b"a", Span { start: 0, end: 2 }).is_none());
    let bounded = diagnostics::document("source_error", vec![json!({}); 65]);
    assert_eq!(bounded["status"], "source_error");
    assert_eq!(bounded["diagnostics"][0]["code"], "DIAGNOSTIC_COUNT_LIMIT");
    let ordered = diagnostics::document(
        "source_error",
        vec![
            diagnostics::diagnostic(
                "TYPE_MISMATCH",
                "later",
                "x.inv",
                b"0123456789",
                Some(Span { start: 8, end: 9 }),
            ),
            diagnostics::diagnostic(
                "NAME_UNKNOWN",
                "earlier",
                "x.inv",
                b"0123456789",
                Some(Span { start: 2, end: 3 }),
            ),
        ],
    );
    assert_eq!(ordered["diagnostics"][0]["code"], "NAME_UNKNOWN");
    let mut explain = json!({"schema":"invar-explain-v1","schema_version":1,"status":"ok","diagnostics":[],"source_identity":"fnv1a64:0","regions":[]});
    let explain_schema = schema("explain-v1.schema.json");
    assert!(jsonschema::is_valid(&explain_schema, &explain));
    explain["new_field_without_version_bump"] = json!(1);
    assert!(!jsonschema::is_valid(&explain_schema, &explain));
}
