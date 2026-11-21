//! Issue #255: `for x in xs { ... }` over an array, a `View` or a `MutView`
//! is the counted loop over `0u32 .. len(xs)` on every engine.
use crate::surface_sugar::{assert_same_program_renamed, first_code};

/// `sugared` against `plain`, whose loops spell the hidden index of the
/// for-each over `element` as `index`.
fn same(sugared: &str, plain: &str, loops: &[(&str, &str)]) {
    let renames: Vec<(String, &str)> = loops
        .iter()
        .map(|(element, index)| {
            let at = sugared.find(&format!("for {element} in ")).unwrap() + 4;
            (format!("each@{at}"), *index)
        })
        .collect();
    assert_same_program_renamed(sugared, plain, &renames);
}

fn main_with(prelude: &str, body: &str) -> String {
    format!(
        "fn main() -> i32 {{\n    {prelude}\n    var sum: u32 = 0u32;\n    {body}\n    if sum == 14u32 {{ return 1; }}\n    return 0;\n}}\n"
    )
}

#[test]
fn for_each_by_value_is_the_counted_loop() {
    let full = "let xs: Array<u32> = [2u32, 4u32, 8u32];";
    for (prelude, sugared, plain) in [
        (
            full,
            "for x in xs { sum += x; }",
            "for i in 0u32 .. len(xs) { let x = xs[i]; sum += x; }",
        ),
        (
            "let xs: Array<u32> = [];",
            "for x in xs { sum += x; }",
            "for i in 0u32 .. len(xs) { let x = xs[i]; sum += x; }",
        ),
        (
            "let ys: Array<u32> = [2u32, 4u32, 8u32, 0u32]; let xs = view(ys, 1u32, 4u32);",
            "for x in xs { sum += x; }",
            "for i in 0u32 .. len(xs) { let x = xs[i]; sum += x; }",
        ),
        // The element's own trap, in source order, at the same fuel prefix.
        (
            "let xs: Array<u32> = [2u32, 0u32, 8u32];",
            "for x in xs { sum += 16u32 / x; }",
            "for i in 0u32 .. len(xs) { let x = xs[i]; sum += 16u32 / x; }",
        ),
    ] {
        same(
            &main_with(prelude, sugared),
            &main_with(prelude, plain),
            &[("x", "i")],
        );
    }
    let two = "let xs: Array<u32> = [1u32, 2u32]; let ys: Array<u32> = [3u32, 4u32];";
    same(
        &main_with(two, "for x in xs { for y in ys { sum += x * y; } }"),
        &main_with(
            two,
            "for i in 0u32 .. len(xs) { let x = xs[i]; for j in 0u32 .. len(ys) { let y = ys[j]; sum += x * y; } }",
        ),
        &[("x", "i"), ("y", "j")],
    );
}

#[test]
fn a_written_element_is_the_array_place() {
    let owner = "var xs: Array<u32> = [1u32, 2u32, 4u32];";
    // Written through, the index takes the element name.
    let read_back = "for i in 0u32 .. len(xs) { sum += xs[i]; }";
    for (sugared, plain) in [
        (
            "for x in xs { x *= 2u32; }",
            "for x in 0u32 .. len(xs) { xs[x] = xs[x] * 2u32; }",
        ),
        (
            "for x in xs { if x > 1u32 { x = x + x; } }",
            "for x in 0u32 .. len(xs) { if xs[x] > 1u32 { xs[x] = xs[x] + xs[x]; } }",
        ),
        (
            "if true { let w = view_mut(xs, 0u32, 3u32); for x in w { x += 1u32; } }",
            "if true { let w = view_mut(xs, 0u32, 3u32); for x in 0u32 .. len(w) { w[x] = w[x] + 1u32; } }",
        ),
    ] {
        same(
            &main_with(owner, &format!("{sugared} {read_back}")),
            &main_with(owner, &format!("{plain} {read_back}")),
            &[],
        );
    }
    let mut_parameter = |body: &str| {
        format!(
            "fn double(xs: mut Array<u32>) -> i32 {{ {body} return 0; }}\n{}",
            main_with(
                "var ys: Array<u32> = [7u32];",
                "let done: i32 = double(ys); sum = ys[0u32];"
            )
        )
    };
    same(
        &mut_parameter("for x in xs { x *= 2u32; }"),
        &mut_parameter("for x in 0u32 .. len(xs) { xs[x] = xs[x] * 2u32; }"),
        &[],
    );
}

#[test]
fn for_each_refuses_what_the_counted_loop_refuses() {
    for (prelude, sugared, plain) in [
        // Writing through an immutable owner or a read view.
        (
            "let xs: Array<u32> = [1u32];",
            "for x in xs { x = 2u32; }",
            "for i in 0u32 .. len(xs) { xs[i] = 2u32; }",
        ),
        (
            "let ys: Array<u32> = [1u32]; let xs = view(ys, 0u32, 1u32);",
            "for x in xs { x = 2u32; }",
            "for i in 0u32 .. len(xs) { xs[i] = 2u32; }",
        ),
        // A mutable iteration cannot alias a read of the same array.
        (
            "var ys: Array<u32> = [1u32, 2u32]; let xs = view_mut(ys, 0u32, 2u32);",
            "for x in xs { x += ys[0u32]; }",
            "for i in 0u32 .. len(xs) { xs[i] = xs[i] + ys[0u32]; }",
        ),
        // The element is an ordinary immutable binding unless it is written.
        (
            "let xs: Array<u32> = [1u32]; let x: u32 = 0u32;",
            "for x in xs { sum += x; }",
            "for i in 0u32 .. len(xs) { let x = xs[i]; sum += x; }",
        ),
        (
            "let xs: Array<i32> = [1i32];",
            "for x in xs { sum += x; }",
            "for i in 0u32 .. len(xs) { let x = xs[i]; sum += x; }",
        ),
    ] {
        let code = first_code(&main_with(prelude, sugared));
        assert_ne!(code, "OK", "{sugared}");
        assert_eq!(
            code,
            first_code(&main_with(prelude, plain)),
            "{prelude} {sugared}"
        );
    }
    for (body, code) in [
        ("for x in [1u32] { sum += x; }", "PARSE_EXPECTED_TOKEN"),
        ("for x in len(xs) { sum += x; }", "PARSE_EXPECTED_TOKEN"),
        (
            "for x in xs { var x: u32 = 1u32; x = 2u32; }",
            "NAME_DUPLICATE",
        ),
        (
            "for x in xs { x = 1u32; for x in 0u32 .. 1u32 {} }",
            "NAME_DUPLICATE",
        ),
        ("for x in xs { x = 1u32; let y = move x; }", "OWNERSHIP"),
        (
            "for x in xs { let c = closure [copy x: u32]() -> u32 { return x; }; x = c(); }",
            "OWNERSHIP",
        ),
        (
            "for x in xs { let v: u32 = x; let c = closure [copy v: u32]() -> u32 { return v; }; x = c(); }",
            "OK",
        ),
        // Written through, the element name is still reserved for the function.
        (
            "let x: u32 = 0u32; for x in xs { x = 2u32; }",
            "NAME_DUPLICATE",
        ),
        (
            "for x in xs { x = 2u32; } let x: u32 = 0u32;",
            "NAME_DUPLICATE",
        ),
        ("for main in xs { main = 2u32; }", "NAME_DUPLICATE"),
    ] {
        let source = main_with("var xs: Array<u32> = [1u32];", body);
        assert_eq!(first_code(&source), code, "{body}");
    }
}
