use super::common::{assert_fixes_to, first_error, first_fix};

fn code(source: &str) -> &'static str {
    first_error(source).0
}

// TYPE_MISMATCH on an integer literal

#[test]
fn issue_dx_22_type_mismatch_corrects_the_suffix_of_an_initializer() {
    let source = "fn main() -> i32 {
    let b: u32 = 2i32;
    return 0i32;
}
";
    assert_eq!(code(source), "TYPE_MISMATCH");
    assert_fixes_to(
        source,
        &source.replace("2i32", "2u32"),
        "Write the literal as u32",
    );
}

#[test]
fn issue_dx_22_type_mismatch_corrects_the_suffix_of_an_argument() {
    let source = "fn half(n: u32) -> u32 {
    return n / 2u32;
}

fn main() -> i32 {
    let x: u32 = half(8i32);
    return 0i32;
}
";
    assert_eq!(code(source), "TYPE_MISMATCH");
    assert_fixes_to(
        source,
        &source.replace("8i32", "8u32"),
        "Write the literal as u32",
    );
}

#[test]
fn issue_dx_22_type_mismatch_gives_no_fix_for_a_variable_or_an_out_of_range_literal() {
    // Two variables of different types: there is no literal to correct.
    let variables = "fn main() -> i32 {
    let a: i32 = 1i32;
    let b: u32 = 2u32;
    return a + b;
}
";
    assert_eq!(code(variables), "TYPE_MISMATCH");
    assert_eq!(first_fix(variables), None);
    // As i32 the literal is out of range: the fix would create another error.
    let range = "fn main() -> i32 {
    let a: i32 = 4000000000u32;
    return a;
}
";
    assert_eq!(code(range), "TYPE_MISMATCH");
    assert_eq!(first_fix(range), None);
}

// NAME_UNKNOWN

#[test]
fn issue_dx_22_name_unknown_uses_the_one_nearest_binding() {
    let source = "fn main() -> i32 {
    let total_sum: i32 = 41i32;
    return total_sun + 1i32;
}
";
    assert_eq!(code(source), "NAME_UNKNOWN");
    assert_fixes_to(
        source,
        &source.replace("total_sun", "total_sum"),
        "Use 'total_sum'",
    );
}

#[test]
fn issue_dx_22_name_unknown_uses_the_nearest_function_and_parameter() {
    let source = "fn helper(width: i32) -> i32 {
    return widht + 1i32;
}

fn main() -> i32 {
    return helpr(41i32);
}
";
    assert_eq!(code(source), "NAME_UNKNOWN");
    let (_, first) = first_fix(source).expect("a fix for the parameter");
    assert_eq!(first, source.replace("widht", "width"));
    let after_first = first;
    assert_eq!(code(&after_first), "NAME_UNKNOWN");
    assert_fixes_to(
        &after_first,
        &after_first.replace("helpr", "helper"),
        "Use 'helper'",
    );
}

#[test]
fn issue_dx_22_name_unknown_gives_no_fix_when_the_name_is_far_tied_or_short() {
    let far = "fn main() -> i32 {
    let alpha: i32 = 1i32;
    return zebra;
}
";
    assert_eq!(code(far), "NAME_UNKNOWN");
    assert_eq!(first_fix(far), None);
    let tied = "fn main() -> i32 {
    let count1: i32 = 1i32;
    let count2: i32 = 2i32;
    return count3;
}
";
    assert_eq!(code(tied), "NAME_UNKNOWN");
    assert_eq!(first_fix(tied), None);
    let short = "fn main() -> i32 {
    let j: i32 = 1i32;
    return i;
}
";
    assert_eq!(code(short), "NAME_UNKNOWN");
    assert_eq!(first_fix(short), None);
}

#[test]
fn issue_dx_22_name_unknown_sees_a_loop_variable_but_not_a_later_binding() {
    let looped = "fn main() -> i32 {
    var sum: i32 = 0i32;
    for index in 0u32 .. 3u32 {
        sum = sum + 1i32;
        let seen: u32 = indx;
    }
    return sum;
}
";
    assert_eq!(code(looped), "NAME_UNKNOWN");
    assert_fixes_to(looped, &looped.replace("indx", "index"), "Use 'index'");
    let later = "fn main() -> i32 {
    let early: i32 = erly;
    let early_two: i32 = 2i32;
    return early;
}
";
    assert_eq!(code(later), "NAME_UNKNOWN");
    assert_eq!(first_fix(later), None);
}

#[test]
fn issue_dx_22_name_unknown_gives_no_fix_when_an_intrinsic_is_as_near() {
    // `dos` is one edit from the function `bos` and from the intrinsic `cos`.
    let source = "fn bos(x: f32) -> f32 {
    return x;
}

fn run(q: f32) -> f32 {
    return dos(q);
}

fn main() -> i32 {
    return 0i32;
}
";
    assert_eq!(code(source), "NAME_UNKNOWN");
    assert_eq!(first_fix(source), None);
}

#[test]
fn issue_dx_22_name_unknown_still_fixes_a_call_no_intrinsic_is_near() {
    let source = "fn helper(x: i32) -> i32 {
    return x;
}

fn main() -> i32 {
    return helpr(1i32);
}
";
    assert_fixes_to(source, &source.replace("helpr", "helper"), "Use 'helper'");
}

#[test]
fn issue_dx_22_name_unknown_does_not_compare_very_long_identifiers() {
    // Two 100000-character names a few edits apart: no fix, and no quadratic
    // distance computation.
    let known = "a".repeat(100_000);
    let unknown = format!("{}b", &known[1..]);
    let source = format!(
        "fn main() -> i32 {{
    let {known}: i32 = 1i32;
    return {unknown};
}}
"
    );
    assert_eq!(code(&source), "NAME_UNKNOWN");
    assert_eq!(first_fix(&source), None);
}
