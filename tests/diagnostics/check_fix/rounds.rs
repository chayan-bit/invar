use invar::fixes::{self, MAX_ROUNDS};
use invar::front;

fn piece(kind: usize, n: usize) -> String {
    match kind {
        0 => format!(
            "fn name_{n}() -> i32 {{
    let total_{n}: i32 = 1i32;
    return totl_{n};
}}
"
        ),
        1 => format!(
            "fn lit_{n}() -> i32 {{
    let b: u32 = 7i32;
    return 0i32;
}}
"
        ),
        2 => format!(
            "fn mov_{n}() -> i32 {{
    let a: Array<i32> = [1i32, 2i32];
    let b: Array<i32> = a;
    return b[0u32];
}}
"
        ),
        3 => format!(
            "fn fill_{n}(a: mut Array<i32>) -> i32 {{
    a[0u32] = 9i32;
    return a[0u32];
}}

fn var_{n}() -> i32 {{
    let xs: Array<i32> = [1i32, 2i32];
    return fill_{n}(xs);
}}
"
        ),
        4 => format!(
            "fn head_{n}(xs: read Array<i32>) -> i32 {{
    return xs[0u32];
}}

fn hoist_{n}() -> i32 {{
    return head_{n}([3i32, 4i32]);
}}
"
        ),
        _ => format!(
            "fn eat_{n}(xs: take Array<i32>) -> i32 {{
    return xs[0u32];
}}

fn use_{n}() -> i32 {{
    let a: Array<i32> = [4i32, 5i32];
    let first: i32 = eat_{n}(move a);
    return first + a[1u32];
}}
"
        ),
    }
}

/// Six kinds of single-error function, combined in kind order: every pair and
/// five triples. The consuming-parameter fix needs the whole file clean, so
/// its piece goes last.
fn fixtures() -> Vec<(Vec<usize>, String)> {
    let mut combos: Vec<Vec<usize>> = Vec::new();
    for a in 0..6 {
        for b in a + 1..6 {
            combos.push(vec![a, b]);
        }
    }
    combos.extend([
        vec![0, 1, 2],
        vec![0, 2, 4],
        vec![1, 3, 5],
        vec![2, 3, 4],
        vec![0, 4, 5],
    ]);
    combos
        .into_iter()
        .enumerate()
        .map(|(n, kinds)| {
            let mut source: String = kinds.iter().map(|kind| piece(*kind, n)).collect();
            source.push_str("\nfn main() -> i32 {\n    return 0i32;\n}\n");
            (kinds, source)
        })
        .collect()
}

fn checks_clean(source: &[u8]) -> bool {
    front::parse(source)
        .and_then(|program| invar::check::check(program, true))
        .is_ok()
}

#[test]
fn fix_converges_on_twenty_multi_error_fixtures_and_is_idempotent() {
    let fixtures = fixtures();
    assert_eq!(fixtures.len(), 20);
    for (kinds, source) in fixtures {
        assert!(!checks_clean(source.as_bytes()), "{kinds:?} has errors");
        let outcome = fixes::fix(source.as_bytes());
        let remaining = outcome
            .remaining
            .as_ref()
            .map(|error| (error.code, &error.message));
        assert!(remaining.is_none(), "{kinds:?}: {remaining:?}\n{source}");
        assert_eq!(
            outcome.applied.len(),
            kinds.len(),
            "{kinds:?}: {:?}",
            outcome.applied
        );
        assert!(outcome.applied.len() <= MAX_ROUNDS);
        assert!(checks_clean(&outcome.fixed), "{kinds:?}");
        // Idempotent: a clean source is left alone.
        let again = fixes::fix(&outcome.fixed);
        assert!(
            again.applied.is_empty() && again.remaining.is_none(),
            "{kinds:?}"
        );
        assert_eq!(again.fixed, outcome.fixed, "{kinds:?}");
        // Deterministic: the same input gives the same fixes.
        let twice = fixes::fix(source.as_bytes());
        assert_eq!(
            (twice.fixed, twice.applied),
            (outcome.fixed, outcome.applied)
        );
    }
}

#[test]
fn fix_stops_at_the_round_limit_and_at_an_error_without_a_fix() {
    // Nine functions with an unknown name one edit from a binding: eight rounds run.
    let mut source = String::new();
    for n in 0..9 {
        source.push_str(&format!(
            "fn f{n}() -> i32 {{\n    let value_{n}: i32 = {n}i32;\n    return valu_{n};\n}}\n\n"
        ));
    }
    source.push_str("fn main() -> i32 {\n    return 0i32;\n}\n");
    let outcome = fixes::fix(source.as_bytes());
    assert_eq!(outcome.applied.len(), MAX_ROUNDS);
    assert_eq!(
        outcome.remaining.map(|error| error.code),
        Some("NAME_UNKNOWN")
    );
    let unfixable = "fn main() -> i32 {\n    return nothing_close;\n}\n";
    let outcome = fixes::fix(unfixable.as_bytes());
    assert!(outcome.applied.is_empty());
    assert_eq!(outcome.fixed, unfixable.as_bytes());
    assert_eq!(
        outcome.remaining.map(|error| error.code),
        Some("NAME_UNKNOWN")
    );
}

#[test]
fn fix_leaves_several_errors_of_one_function_to_the_author() {
    // `check` reports the first error only, so a second error in the same
    // function cannot be told from one the fix would create: no fix, no guess.
    let source = "fn main() -> i32 {
    let value_a: i32 = 1i32;
    let value_b: i32 = valu_a;
    return valu_b;
}
";
    let outcome = fixes::fix(source.as_bytes());
    assert!(outcome.applied.is_empty());
    assert_eq!(outcome.fixed, source.as_bytes());
    assert_eq!(
        outcome.remaining.map(|error| error.code),
        Some("NAME_UNKNOWN")
    );
}

#[test]
fn fix_gives_nothing_for_a_source_with_a_syntax_error() {
    let source = "fn main( -> i32 { return 0; }";
    let outcome = fixes::fix(source.as_bytes());
    assert!(outcome.applied.is_empty());
    assert_eq!(outcome.fixed, source.as_bytes());
    assert!(outcome.remaining.is_some());
}

#[test]
fn unified_diff_has_hunks_context_and_the_no_newline_marker() {
    assert_eq!(fixes::unified_diff("p.inv", "a\nb\n", "a\nb\n"), "");
    assert_eq!(
        fixes::unified_diff("p.inv", "a\nb\nc\n", "a\nB\nc\n"),
        "--- a/p.inv\n+++ b/p.inv\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n"
    );
    let old: String = (1..=30).map(|n| format!("line {n}\n")).collect();
    let new = old
        .replace("line 3\n", "THREE\n")
        .replace("line 28\n", "TWENTY-EIGHT\n");
    let diff = fixes::unified_diff("p.inv", &old, &new);
    assert_eq!(diff.matches("@@ -").count(), 2, "{diff}");
    assert!(diff.contains("@@ -1,6 +1,6 @@\n"), "{diff}");
    assert!(diff.contains("@@ -25,6 +25,6 @@\n"), "{diff}");
    assert_eq!(
        fixes::unified_diff("p.inv", "x", "x\ny"),
        "--- a/p.inv\n+++ b/p.inv\n@@ -1,1 +1,2 @@\n-x\n\\ No newline at end of file\n+x\n+y\n\\ No newline at end of file\n"
    );
}
