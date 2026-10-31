use super::common::{assert_fixes_to, diagnostic_for, first_error, first_fix};

fn code(source: &str) -> &'static str {
    first_error(source).0
}

const TAKE_IT: &str = "fn take_it(xs: take Array<i32>) -> i32 {
    return xs[0u32];
}

fn main() -> i32 {
    var xs: Array<i32> = [4i32, 5i32];
    let first: i32 = take_it(move xs);
    return first + xs[1u32];
}
";

#[test]
fn use_after_move_relaxes_a_take_parameter_to_read() {
    let intended = TAKE_IT
        .replace("xs: take", "xs: read")
        .replace("take_it(move xs)", "take_it(xs)");
    assert_eq!(code(TAKE_IT), "OWN_USE_AFTER_MOVE");
    assert_fixes_to(TAKE_IT, &intended, "Make 'take_it' borrow its argument");
}

#[test]
fn use_after_move_finds_the_take_mode_past_comments() {
    let source = TAKE_IT.replace("xs: take", "xs: /* owned */ // moved in\n    take");
    let intended = source
        .replace("take Array", "read Array")
        .replace("take_it(move xs)", "take_it(xs)");
    assert_eq!(code(&source), "OWN_USE_AFTER_MOVE");
    assert_fixes_to(&source, &intended, "Make 'take_it' borrow its argument");
}

#[test]
fn use_after_move_fixes_every_caller_of_the_relaxed_function() {
    let source = "fn sum2(xs: take Array<i32>) -> i32 {
    return xs[0u32] + xs[1u32];
}

fn first() -> i32 {
    let a: Array<i32> = [1i32, 2i32];
    let s: i32 = sum2(move a);
    return s + a[0u32];
}

fn main() -> i32 {
    let b: Array<i32> = [3i32, 4i32];
    let t: i32 = sum2(move b);
    return first() + t;
}
";
    let intended = source
        .replace("xs: take", "xs: read")
        .replace("sum2(move a)", "sum2(a)")
        .replace("sum2(move b)", "sum2(b)");
    assert_eq!(code(source), "OWN_USE_AFTER_MOVE");
    assert_fixes_to(source, &intended, "Make 'sum2' borrow its argument");
}

#[test]
fn use_after_move_gives_no_fix_when_the_callee_consumes_its_parameter() {
    let source = "fn pass(xs: take Array<i32>) -> Array<i32> {
    return xs;
}

fn main() -> i32 {
    let a: Array<i32> = [1i32];
    let b: Array<i32> = pass(move a);
    return a[0u32] + b[0u32];
}
";
    assert_eq!(code(source), "OWN_USE_AFTER_MOVE");
    assert_eq!(first_fix(source), None);
    let diagnostic = diagnostic_for(source);
    assert_eq!(diagnostic["suggested_fixes"], serde_json::json!([]));
    let notes = diagnostic["notes"].as_array().unwrap();
    assert_eq!(notes.len(), 1);
    assert!(
        notes[0]
            .as_str()
            .unwrap()
            .starts_with("moved here, line 7 column 30"),
        "{notes:?}"
    );
}

#[test]
fn use_after_move_names_an_earlier_move_that_is_not_a_call() {
    let source = "fn main() -> i32 {
    let a: Array<i32> = [1i32];
    let b: Array<i32> = move a;
    return a[0u32] + b[0u32];
}
";
    assert_eq!(code(source), "OWN_USE_AFTER_MOVE");
    assert_eq!(first_fix(source), None);
    let notes = diagnostic_for(source)["notes"].clone();
    assert!(
        notes[0]
            .as_str()
            .unwrap()
            .starts_with("moved here, line 3 column 25"),
        "{notes}"
    );
}

#[test]
fn return_borrow_has_no_fix_and_states_the_rule() {
    let source = "fn f(xs: read Array<i32>) -> Array<i32> {
    return xs;
}

fn main() -> i32 {
    return 0i32;
}
";
    assert_eq!(code(source), "OWN_RETURN_BORROW");
    assert_eq!(first_fix(source), None);
    let notes = diagnostic_for(source)["notes"].clone();
    let text = notes[0].as_str().unwrap();
    assert!(text.contains("return a copy"), "{text}");
    assert!(text.contains("-> View<T> from <parameter>"), "{text}");
}

fn note_count(source: &str) -> usize {
    diagnostic_for(source)["notes"]
        .as_array()
        .map_or(0, Vec::len)
}

#[test]
fn use_after_move_follows_the_evaluation_order_of_an_assignment() {
    // The store to `xs` is reported before the move in the source text, yet it
    // runs after the right-hand side that consumed `xs`.
    let source = "fn remake(xs: take Array<i32>) -> Array<i32> {
    return [xs[0u32], 2i32];
}

fn main() -> i32 {
    var xs: Array<i32> = [4i32, 5i32];
    xs = remake(move xs);
    return xs[0u32];
}
";
    let intended = source
        .replace("xs: take", "xs: read")
        .replace("remake(move xs)", "remake(xs)");
    assert_eq!(code(source), "OWN_USE_AFTER_MOVE");
    assert_fixes_to(source, &intended, "Make 'remake' borrow its argument");
    let notes = diagnostic_for(source)["notes"].clone();
    assert!(
        notes[0].as_str().unwrap().starts_with("moved here, line 7"),
        "{notes:?}"
    );
}

#[test]
fn use_after_move_name_identifies_the_binding_within_a_function() {
    // The note matches a move by the name's spelling. That is sound because a
    // function cannot declare a name twice, not even in a nested block.
    let source = "fn main() -> i32 {
    let a: Array<i32> = [1i32];
    let b: Array<i32> = move a;
    if true {
        let a: Array<i32> = [2i32];
        let c: Array<i32> = move a;
    }
    return a[0u32] + b[0u32];
}
";
    assert_eq!(code(source), "NAME_DUPLICATE");
}

#[test]
fn return_borrow_gives_no_note_for_a_view_from_the_wrong_source() {
    let source = "fn bad(xs: read Array<i32>) -> View<i32> from xs {
    let local: Array<i32> = [1i32];
    return view(local, 0u32, 1u32);
}

fn main() -> i32 {
    return 0i32;
}
";
    assert_eq!(code(source), "OWN_RETURN_BORROW");
    assert_eq!(note_count(source), 0);
}
