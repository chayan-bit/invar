use super::common::{assert_fixes_to, first_error, first_fix};

fn code(source: &str) -> &'static str {
    first_error(source).0
}

// OWN_BORROW_SOURCE

const HEAD_SUM: &str = "fn head_sum(xs: read Array<i32>) -> i32 {
    return xs[0u32] + xs[1u32];
}
";

#[test]
fn borrow_source_hoists_an_array_literal_argument() {
    let source = format!(
        "{HEAD_SUM}
fn main() -> i32 {{
    return head_sum([3i32, 4i32]);
}}
"
    );
    let intended = format!(
        "{HEAD_SUM}
fn main() -> i32 {{
    let tmp_1: Array<i32> = [3i32, 4i32];
    return head_sum(tmp_1);
}}
"
    );
    assert_eq!(code(&source), "OWN_BORROW_SOURCE");
    assert_fixes_to(&source, &intended, "Hoist the argument into 'tmp_1'");
}

#[test]
fn borrow_source_hoists_a_call_result_in_a_binding() {
    let source = format!(
        "{HEAD_SUM}
fn make() -> Array<i32> {{
    return [1i32, 2i32];
}}

fn main() -> i32 {{
    let n: i32 = head_sum(make());
    return n;
}}
"
    );
    let intended = format!(
        "{HEAD_SUM}
fn make() -> Array<i32> {{
    return [1i32, 2i32];
}}

fn main() -> i32 {{
    let tmp_1 = make();
    let n: i32 = head_sum(tmp_1);
    return n;
}}
"
    );
    assert_eq!(code(&source), "OWN_BORROW_SOURCE");
    assert_fixes_to(&source, &intended, "Hoist the argument into 'tmp_1'");
}

#[test]
fn borrow_source_picks_a_name_that_is_free() {
    let source = format!(
        "{HEAD_SUM}
fn main() -> i32 {{
    let tmp_1: i32 = 1i32;
    return head_sum([3i32, 4i32]) + tmp_1;
}}
"
    );
    let (description, fixed) = first_fix(&source).expect("a validated fix");
    assert_eq!(description, "Hoist the argument into 'tmp_2'");
    assert!(
        fixed.contains("let tmp_2: Array<i32> = [3i32, 4i32];"),
        "{fixed}"
    );
}

#[test]
fn borrow_source_gives_no_fix_inside_a_loop_condition() {
    // Hoisting would run the argument once instead of on every test.
    let source = format!(
        "{HEAD_SUM}
fn main() -> i32 {{
    var i: i32 = 0i32;
    while head_sum([1i32, 2i32]) < 9i32 {{
        i = i + 1i32;
    }}
    return i;
}}
"
    );
    assert_eq!(code(&source), "OWN_BORROW_SOURCE");
    assert_eq!(first_fix(&source), None);
}

// OWN_BORROW_CONFLICT

const LOANS: &str = "fn bump(a: mut Array<i32>, n: i32) -> i32 {
    a[0u32] = n;
    return n;
}

fn peek(a: read Array<i32>) -> i32 {
    return a[0u32];
}
";

#[test]
fn borrow_conflict_hoists_the_inner_call_of_a_return() {
    let source = format!(
        "{LOANS}
fn main() -> i32 {{
    var xs: Array<i32> = [1i32, 2i32];
    return bump(xs, peek(xs));
}}
"
    );
    let intended = format!(
        "{LOANS}
fn main() -> i32 {{
    var xs: Array<i32> = [1i32, 2i32];
    let tmp_1 = peek(xs);
    return bump(xs, tmp_1);
}}
"
    );
    assert_eq!(code(&source), "OWN_BORROW_CONFLICT");
    assert_fixes_to(&source, &intended, "Hoist the argument into 'tmp_1'");
}

#[test]
fn borrow_conflict_hoists_the_inner_call_of_a_binding() {
    let source = format!(
        "{LOANS}
fn main() -> i32 {{
    var xs: Array<i32> = [1i32, 2i32];
    let r: i32 = bump(xs, peek(xs));
    return r + xs[0u32];
}}
"
    );
    let intended = source.replace(
        "    let r: i32 = bump(xs, peek(xs));",
        "    let tmp_1 = peek(xs);\n    let r: i32 = bump(xs, tmp_1);",
    );
    assert_eq!(code(&source), "OWN_BORROW_CONFLICT");
    assert_fixes_to(&source, &intended, "Hoist the argument into 'tmp_1'");
}

#[test]
fn borrow_conflict_gives_no_fix_for_two_loans_of_one_name() {
    let source = "fn both(a: mut Array<i32>, b: read Array<i32>) -> i32 {
    return a[0u32] + b[0u32];
}

fn main() -> i32 {
    var xs: Array<i32> = [1i32, 2i32];
    return both(xs, xs);
}
";
    assert_eq!(code(source), "OWN_BORROW_CONFLICT");
    assert_eq!(first_fix(source), None);
}

#[test]
fn borrow_conflict_gives_no_fix_when_hoisting_would_reorder_a_call() {
    // `peek(ys)` runs before `bump(..)` today; a hoisted `peek(xs)` would run first.
    let source = format!(
        "{LOANS}
fn pair(a: i32, b: i32) -> i32 {{
    return a + b;
}}

fn main() -> i32 {{
    var xs: Array<i32> = [1i32, 2i32];
    let ys: Array<i32> = [3i32];
    return pair(peek(ys), bump(xs, peek(xs)));
}}
"
    );
    assert_eq!(code(&source), "OWN_BORROW_CONFLICT");
    assert_eq!(first_fix(&source), None);
}

#[test]
fn borrow_source_hoists_the_owner_of_a_view() {
    // The diagnostic for `view` spans the whole call, not its first argument.
    let source = "fn make() -> Array<i32> {
    return [1i32, 2i32];
}

fn main() -> i32 {
    let part = view(make(), 0u32, 1u32);
    return part[0u32];
}
";
    let intended = source.replace(
        "    let part = view(make(), 0u32, 1u32);",
        "    let tmp_1 = make();\n    let part = view(tmp_1, 0u32, 1u32);",
    );
    assert_eq!(code(source), "OWN_BORROW_SOURCE");
    assert_fixes_to(source, &intended, "Hoist the argument into 'tmp_1'");
}

#[test]
fn borrow_source_hoists_a_call_to_a_foreign_function() {
    // Foreign parameters are scalars or `read`/`take` Bytes and String, so the
    // temporary is a plain `let`.
    let source =
        "extern c fn c_len(data: read Bytes) -> Result<u32, ForeignError> oracle oracle_len;

fn oracle_len(data: read Bytes) -> Result<u32, ForeignError> {
    return Result::Ok(len(data));
}

fn make() -> Bytes {
    return b\"abc\";
}

fn main() -> i32 {
    let n: Result<u32, ForeignError> = c_len(make());
    return 0i32;
}
";
    let intended = source.replace(
        "    let n: Result<u32, ForeignError> = c_len(make());",
        "    let tmp_1 = make();\n    let n: Result<u32, ForeignError> = c_len(tmp_1);",
    );
    assert_eq!(code(source), "OWN_BORROW_SOURCE");
    assert_fixes_to(source, &intended, "Hoist the argument into 'tmp_1'");
}

#[test]
fn borrow_source_gives_no_fix_past_an_allocating_literal() {
    // `b"abc"` allocates and can hit a resource limit: it stays first.
    let source = "fn tagged(label: take Bytes, xs: read Array<i32>) -> i32 {
    return xs[0u32];
}

fn make() -> Array<i32> {
    return [1i32, 2i32];
}

fn main() -> i32 {
    return tagged(b\"abc\", make());
}
";
    assert_eq!(code(source), "OWN_BORROW_SOURCE");
    assert_eq!(first_fix(source), None);
}

#[test]
fn borrow_conflict_hoists_a_call_below_a_compound_argument() {
    let source = format!(
        "{LOANS}
fn main() -> i32 {{
    var xs: Array<i32> = [1i32, 2i32];
    return bump(xs, peek(xs) + 1i32);
}}
"
    );
    let intended = source.replace(
        "    return bump(xs, peek(xs) + 1i32);",
        "    let tmp_1 = peek(xs);\n    return bump(xs, tmp_1 + 1i32);",
    );
    assert_eq!(code(&source), "OWN_BORROW_CONFLICT");
    assert_fixes_to(&source, &intended, "Hoist the argument into 'tmp_1'");
}
