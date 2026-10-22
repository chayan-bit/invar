use invar::front::{self, Span};

/// The first error `check` reports for `source`.
pub fn first_error(source: &str) -> (&'static str, Span) {
    let error = front::parse(source.as_bytes())
        .and_then(|program| invar::check::check(program, false))
        .map(|_| ())
        .expect_err("source has an error");
    (error.code, error.span.expect("source error has a span"))
}

/// The source after the first emitted fix of its first diagnostic, with the
/// fix description. `None` when the diagnostic has no validated fix.
pub fn first_fix(source: &str) -> Option<(String, String)> {
    let (code, at) = first_error(source);
    let fixes = invar::fixes::suggested_fixes(code, "fixture.inv", source.as_bytes(), Some(at));
    let fix = fixes.first()?;
    let mut edited = source.to_owned();
    let mut edits: Vec<(usize, usize, &str)> = fix["edits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|edit| {
            (
                edit["span"]["start"].as_u64().unwrap() as usize,
                edit["span"]["end"].as_u64().unwrap() as usize,
                edit["replacement"].as_str().unwrap(),
            )
        })
        .collect();
    edits.sort_by_key(|edit| std::cmp::Reverse(edit.0));
    for (start, end, replacement) in edits {
        edited.replace_range(start..end, replacement);
    }
    Some((fix["description"].as_str().unwrap().to_owned(), edited))
}

/// Reference and MIR results of a program that has to check clean.
pub fn observe(source: &str) -> (String, String) {
    let checked = front::parse(source.as_bytes())
        .and_then(|program| invar::check::check(program, true))
        .expect("program checks clean");
    let reference = invar::eval::observe(&checked, 100_000);
    let mir = invar::mir::lower(&checked).expect("program lowers");
    let mir = invar::mir_eval::observe(&mir, 100_000);
    (
        format!("{:?}", reference.result),
        format!("{:?}", mir.result),
    )
}

/// A fixed program checks clean, and runs as the program the author meant on
/// the reference evaluator and on MIR. Native is checked where it is built.
pub fn assert_fixes_to(source: &str, intended: &str, description: &str) {
    let (found, fixed) = first_fix(source).expect("a validated fix");
    assert_eq!(found, description);
    assert_eq!(fixed, intended);
    let (reference, mir) = observe(&fixed);
    assert_eq!(
        reference, mir,
        "reference and MIR agree on the fixed program"
    );
    assert!(reference.starts_with("Ok("), "{reference}");
}

/// The diagnostic document entry `check` would print for the first error.
pub fn diagnostic_for(source: &str) -> serde_json::Value {
    let (code, at) = first_error(source);
    invar::diagnostics::diagnostic(code, "message", "fixture.inv", source.as_bytes(), Some(at))
}
