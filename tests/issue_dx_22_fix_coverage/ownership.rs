use crate::common::{assert_fixes_to, first_error, first_fix};

fn code(source: &str) -> &'static str {
    first_error(source).0
}

// OWN_MOVE_REQUIRED

const MOVE_INIT: &str = "fn main() -> i32 {
    let a: Array<i32> = [4i32, 5i32];
    let b: Array<i32> = a;
    return b[1u32];
}
";
const MOVE_INIT_FIXED: &str = "fn main() -> i32 {
    let a: Array<i32> = [4i32, 5i32];
    let b: Array<i32> = move a;
    return b[1u32];
}
";
const MOVE_ARG: &str = "fn eat(xs: take Array<i32>) -> i32 {
    return xs[0u32];
}

fn main() -> i32 {
    let a: Array<i32> = [7i32, 8i32];
    return eat(a);
}
";
const MOVE_ARG_FIXED: &str = "fn eat(xs: take Array<i32>) -> i32 {
    return xs[0u32];
}

fn main() -> i32 {
    let a: Array<i32> = [7i32, 8i32];
    return eat(move a);
}
";

#[test]
fn issue_dx_22_move_required_inserts_move_before_an_initializer() {
    assert_eq!(code(MOVE_INIT), "OWN_MOVE_REQUIRED");
    assert_fixes_to(MOVE_INIT, MOVE_INIT_FIXED, "Move 'a'");
}

#[test]
fn issue_dx_22_move_required_inserts_move_before_a_take_argument() {
    assert_eq!(code(MOVE_ARG), "OWN_MOVE_REQUIRED");
    assert_fixes_to(MOVE_ARG, MOVE_ARG_FIXED, "Move 'a'");
}

#[test]
fn issue_dx_22_move_required_gives_no_fix_when_the_move_would_break_a_later_use() {
    // `move a` fixes the first error and makes the later read a use after move.
    let source = "fn main() -> i32 {
    let a: Array<i32> = [4i32, 5i32];
    let b: Array<i32> = a;
    return a[0u32] + b[1u32];
}
";
    assert_eq!(code(source), "OWN_MOVE_REQUIRED");
    assert_eq!(first_fix(source), None);
}

#[test]
fn issue_dx_22_move_required_gives_no_fix_for_a_borrowed_owner_or_a_loop() {
    let borrowed = "fn f(xs: read Array<i32>) -> i32 {
    let b: Array<i32> = move xs;
    return b[0u32];
}

fn main() -> i32 {
    let a: Array<i32> = [1i32];
    return f(a);
}
";
    assert_eq!(code(borrowed), "OWN_MOVE_REQUIRED");
    assert_eq!(first_fix(borrowed), None);
    let in_loop = "fn main() -> i32 {
    let a: Array<i32> = [1i32];
    var i: i32 = 0i32;
    while i < 2i32 {
        let b: Array<i32> = a;
        i = i + 1i32;
    }
    return 0i32;
}
";
    assert_eq!(code(in_loop), "OWN_MOVE_REQUIRED");
    assert_eq!(first_fix(in_loop), None);
}

// OWN_MUT_REQUIRED

const FILL: &str = "fn fill(a: mut Array<i32>) -> i32 {
    a[0u32] = 9i32;
    return a[0u32];
}
";

#[test]
fn issue_dx_22_mut_required_turns_a_let_into_a_var() {
    let source = format!(
        "{FILL}
fn main() -> i32 {{
    let xs: Array<i32> = [1i32, 2i32];
    return fill(xs);
}}
"
    );
    let intended = source.replace("let xs", "var xs");
    assert_eq!(code(&source), "OWN_MUT_REQUIRED");
    assert_fixes_to(&source, &intended, "Declare 'xs' with var");
}

#[test]
fn issue_dx_22_mut_required_turns_a_read_parameter_into_mut() {
    let source = format!(
        "{FILL}
fn outer(xs: read Array<i32>) -> i32 {{
    return fill(xs);
}}

fn main() -> i32 {{
    var ys: Array<i32> = [1i32, 2i32];
    return outer(ys);
}}
"
    );
    let intended = source.replace("xs: read", "xs: mut");
    assert_eq!(code(&source), "OWN_MUT_REQUIRED");
    assert_fixes_to(&source, &intended, "Make parameter 'xs' mut");
}

#[test]
fn issue_dx_22_mut_required_finds_the_mode_past_comments() {
    let source = format!(
        "{FILL}
fn outer(xs: /* a /* nested */ note */ // owner
    read Array<i32>) -> i32 {{
    return fill(xs);
}}

fn main() -> i32 {{
    var ys: Array<i32> = [1i32, 2i32];
    return outer(ys);
}}
"
    );
    let intended = source.replace(
        "read Array<i32>) -> i32 {\n    return fill",
        "mut Array<i32>) -> i32 {\n    return fill",
    );
    assert_eq!(code(&source), "OWN_MUT_REQUIRED");
    assert_fixes_to(&source, &intended, "Make parameter 'xs' mut");
}

#[test]
fn issue_dx_22_mut_required_gives_no_fix_when_a_caller_would_break() {
    // `outer` would need a mutable argument, and `main` passes a `let`.
    let source = format!(
        "{FILL}
fn outer(xs: read Array<i32>) -> i32 {{
    return fill(xs);
}}

fn main() -> i32 {{
    let ys: Array<i32> = [1i32, 2i32];
    return outer(ys);
}}
"
    );
    assert_eq!(code(&source), "OWN_MUT_REQUIRED");
    assert_eq!(first_fix(&source), None);
}

#[test]
fn issue_dx_22_mut_required_gives_no_fix_for_a_take_parameter() {
    let source = format!(
        "{FILL}
fn outer(xs: take Array<i32>) -> i32 {{
    return fill(xs);
}}

fn main() -> i32 {{
    let ys: Array<i32> = [1i32, 2i32];
    return outer(move ys);
}}
"
    );
    assert_eq!(code(&source), "OWN_MUT_REQUIRED");
    assert_eq!(first_fix(&source), None);
}
