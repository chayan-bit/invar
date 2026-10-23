#[test]
fn issue_ai_01_machine_diagnostics_qualification_retired_code_stays_in_evidence() {
    // The committed CPU report predates M5_DEVICE_NOT_REQUESTED: its device
    // rows still carry the retired spelling, so it stays registered, deprecated.
    let evidence = std::fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("evidence/m5-02-cpu/qualification.json"),
    )
    .unwrap();
    let retired = "M5_DEVICE_PREREQUISITES_PENDING";
    assert!(String::from_utf8(evidence).unwrap().contains(retired));
    let registry = code_registry();
    let entry = registry["codes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["code"] == retired)
        .unwrap();
    assert_eq!(entry["deprecated"], true);
}
