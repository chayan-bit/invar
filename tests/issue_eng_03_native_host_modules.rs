//! The standalone native host has no module files, so src/native.rs inlines
//! each file submodule of src/numeric_contract.rs. A new `mod x;` there must
//! join that list, or every native spatial link fails with E0583 (main went
//! red when #349 and #350 added `fold` and `storage`).

#[test]
fn issue_eng_03_native_host_inlines_every_numeric_contract_module() {
    let root = env!("CARGO_MANIFEST_DIR");
    let read = |path: &str| std::fs::read_to_string(format!("{root}/{path}")).unwrap();
    let (contract, native) = (read("src/numeric_contract.rs"), read("src/native.rs"));
    // rustfmt may wrap an entry and add a trailing comma, so compare the
    // pair's start with whitespace removed.
    let flat = |text: &str| text.split_whitespace().collect::<String>();
    let mut modules = Vec::new();
    for line in contract.lines().map(str::trim) {
        let declared = line
            .strip_prefix("pub mod ")
            .or_else(|| line.strip_prefix("mod "));
        if let Some(name) = declared.and_then(|rest| rest.strip_suffix(';')) {
            let entry = format!("(\"{line}\", include_str!(\"numeric_contract/{name}.rs\")");
            assert!(
                flat(&native).contains(&flat(&entry)),
                "the native host does not inline `{line}`"
            );
            modules.push(name.to_owned());
        }
    }
    assert_eq!(modules, ["binary64", "fold", "ieee", "storage"]);
    // fold.rs reaches crate::fsum_reference, so the host carries that file too.
    assert!(read("src/numeric_contract/fold.rs").contains("crate::fsum_reference"));
    assert!(native.contains("mod fsum_reference {{ {} }}"));
    assert!(native.contains("include_str!(\"fsum_reference.rs\")"));
}
