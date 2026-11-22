//! AI-01 fault fixtures of admission.

#[test]
fn machine_diagnostics_planner_extraction_fault() {
    let source = include_bytes!("../../examples/generated-map.inv");
    let checked = crate::check::check(crate::front::parse(source).unwrap(), true).unwrap();
    let mir = crate::mir::lower(&checked).unwrap();
    let mut report = super::analyze(source, &mir).unwrap();
    assert_eq!(report.regions[0].status, super::Status::Admitted);
    report.regions[0].shape = None;
    crate::plan::decide(&mir, &mut report.regions);
    let code = report.regions[0].first_rejected_obligation.unwrap();
    assert_eq!(code, "EXTRACTION");
    let document = crate::diagnostics::document(
        "tool_error",
        vec![crate::diagnostics::diagnostic(
            code,
            "forged admitted region",
            "fixture.inv",
            source,
            None,
        )],
    );
    let schema: serde_json::Value = serde_json::from_str(include_str!(
        "../../schema/invar-diagnostics-v3.schema.json"
    ))
    .unwrap();
    assert!(jsonschema::is_valid(&schema, &document), "{document}");
}
