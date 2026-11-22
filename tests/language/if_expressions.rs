//! Issue #361: `if c { a } else { b }` as a scalar expression. One step, then the
//! condition, then only the selected arm, on every engine; an owner either arm moves
//! is moved afterwards, as with match arms; the MIR is a two-arm match without its
//! tag test, so a loop holding one gets the receipt of the equivalent match.
use crate::loop_control::{agree, checked};
use crate::surface_sugar::first_code;
use invar::{eval, mir, schedule};

const GRADES: &str = "fn grade(x: i32) -> i32 {
    return if x > 90 { 4 } else if x > 75 { 3 } else if x > 50 { 2 } else { 1 };
}
fn main() -> i32 {
    let a = grade(95) * 1000 + grade(80) * 100 + grade(60) * 10 + grade(3);
    let u: u32 = if a > 0 { 7u32 } else { 0u32 };
    let ok: Bool = if u == 7u32 { a == 4321 } else { false };
    return if ok { a + 1 } else { -1 };
}";

#[test]
fn if_expressions_agree_on_every_engine() {
    agree(GRADES, Ok(4322), true);
    let looped = "fn main() -> i32 {
    var acc = 0;
    for i in 0u32 .. 10u32 { acc = acc + if i % 2u32 == 0u32 { 1 } else { 100 }; }
    return acc;
}";
    agree(looped, Ok(505), true);
    let floats = "fn main() -> i32 {
    let x = 7;
    let f: f64 = if x == 7 { 1.5f64 } else { 2.0f64 };
    let g = if f > 1.0f64 { 0.25f32 } else { 0.5f32 };
    return if g == 0.25f32 { 1 } else { 0 };
}";
    // The net oracle has no f64 comparison, so floats run on the other engines.
    agree(floats, Ok(1), false);
}

const MOVES: &str = "fn both(a: take Array<u32>) -> u32 { return a[0u32] + a[1u32]; }
fn count(a: take Array<u32>) -> u32 { return len(a); }
fn main() -> i32 {
    let xs: Array<u32> = [3u32, 4u32];
    let v: u32 = if COND { both(move xs) } else { OTHER };
    return if v == 7u32 { 1 } else { 2 };
}";

#[test]
fn an_owner_moved_in_either_arm_is_moved_after_the_expression() {
    let program = |cond: &str, other: &str| MOVES.replace("COND", cond).replace("OTHER", other);
    // Both arms move `xs`, and one arm alone may: the other path cleans it up at return.
    agree(&program("true", "count(move xs)"), Ok(1), true);
    agree(&program("false", "count(move xs)"), Ok(2), true);
    agree(&program("true", "9u32"), Ok(1), true);
    agree(&program("false", "9u32"), Ok(2), true);
    let after =
        program("true", "9u32").replace("    return if", "    let n = len(xs);\n    return if");
    assert_eq!(first_code(&after), "OWN_USE_AFTER_MOVE");
}

/// `main` with `let t: Bool = true; let z = 0;` returning `value`.
fn program(value: &str) -> String {
    format!("fn main() -> i32 {{\n    let t: Bool = true;\n    let z = 0;\n    return {value};\n}}")
}

fn trap_at(source: &str, operator: &str) {
    agree(source, Err(eval::TrapCategory::DivisionByZero), true);
    let observed = eval::observe(&checked(source), 1_000_000);
    let Err(eval::EvalError::Trap { span, .. }) = observed.result else {
        panic!("{:?}", observed.result)
    };
    assert_eq!(span.start, source.find(operator).unwrap(), "{source}");
}

#[test]
fn a_trap_is_reported_at_its_operator_and_the_unselected_arm_never_runs() {
    trap_at(&program("if 10 / z > 1 { 1 } else { 2 }"), "/ z");
    trap_at(&program("if t { 10 % z } else { 2 }"), "% z");
    trap_at(&program("if not t { 2 } else { 10 % z }"), "% z");
    agree(&program("if t { 5 } else { 10 / z }"), Ok(5), true);
    agree(&program("if not t { 10 / z } else { 6 }"), Ok(6), true);
}

fn steps(value: &str) -> u64 {
    let observed = eval::observe(&checked(&program(value)), 1_000_000);
    assert!(observed.result.is_ok(), "{value}: {:?}", observed.result);
    observed.steps_used
}

#[test]
fn an_if_expression_costs_one_step_its_condition_and_its_selected_arm() {
    let plain = steps("5");
    assert_eq!(
        steps("if t { 5 } else { 0 }"),
        plain + 2,
        "the if and the name t"
    );
    assert_eq!(steps("if t { 5 } else { 10 / z * 3 + 1 }"), plain + 2);
    assert_eq!(steps("if not t { 10 / z * 3 + 1 } else { 5 }"), plain + 3);
}

#[test]
fn malformed_if_expressions_are_refused_with_exact_codes() {
    let main = |body: &str| {
        format!("fn main() -> i32 {{\n    let t: Bool = true;\n    {body}\n    return 0;\n}}")
    };
    for (body, code) in [
        ("let v = if 1 { 2 } else { 3 };", "TYPE_MISMATCH"),
        ("let v = if t { 2i32 } else { 3u32 };", "TYPE_MISMATCH"),
        ("let v = if t { 1.5f64 } else { 2.0f32 };", "TYPE_MISMATCH"),
        ("let v: u32 = if t { 2u32 } else { true };", "TYPE_MISMATCH"),
        ("let v = if t { 1 };", "PARSE_IF_WITHOUT_ELSE"),
        (
            "let v = if t { 1 } else if not t { 2 };",
            "PARSE_IF_WITHOUT_ELSE",
        ),
        (
            "let v = if t { return 1; } else { 2 };",
            "PARSE_EXPECTED_TOKEN",
        ),
        (
            "let a: Array<u32> = if t { [1u32] } else { [2u32] };",
            "UNSUPPORTED_FEATURE",
        ),
        ("let v = if t { 1 } else { 2 };", "OK"),
        ("if t { let v = 1; }", "OK"),
    ] {
        assert_eq!(first_code(&main(body)), code, "{body}");
    }
}

/// `sel`'s blocks in depth-first order from its entry, yes before no, skipping blocks
/// that only jump on; numbers (value ids, spans) are masked, and in a block that
/// branches the instructions after its last `Tick` (the condition or tag test) dropped.
fn canonical(source: &str) -> Vec<String> {
    let program = mir::lower(&checked(source)).unwrap();
    let function = program.functions.iter().find(|f| f.name == "sel").unwrap();
    let mask = |text: String| -> String {
        let mut masked = String::new();
        for c in text.chars() {
            if !c.is_ascii_digit() {
                masked.push(c);
            } else if !masked.ends_with('#') {
                masked.push('#');
            }
        }
        masked
    };
    let forward = |mut block: usize| loop {
        let b = &function.blocks[block];
        match &b.end {
            Some(mir::Terminator::Goto { target, args, .. })
                if b.params.is_empty() && b.instructions.is_empty() && args.is_empty() =>
            {
                block = *target
            }
            _ => return block,
        }
    };
    let (mut order, mut pending) = (Vec::new(), vec![forward(function.entry)]);
    while let Some(block) = pending.pop() {
        if order.contains(&block) {
            continue;
        }
        order.push(block);
        match &function.blocks[block].end {
            Some(mir::Terminator::Goto { target, .. }) => pending.push(forward(*target)),
            Some(mir::Terminator::Branch { yes, no, .. }) => {
                pending.extend([forward(*no), forward(*yes)])
            }
            _ => {}
        }
    }
    let at = |id: &usize| order.iter().position(|&o| o == forward(*id)).unwrap();
    order
        .iter()
        .map(|&block| {
            let b = &function.blocks[block];
            let mut instructions: Vec<_> = b
                .instructions
                .iter()
                .map(|i| format!("{:?}", i.kind))
                .collect();
            let end = match &b.end {
                Some(mir::Terminator::Branch { yes, no, .. }) => {
                    let last = instructions.iter().rposition(|i| i == "Tick").unwrap();
                    instructions.truncate(last + 1);
                    format!("branch {} {}", at(yes), at(no))
                }
                Some(mir::Terminator::Goto { target, args, .. }) => {
                    format!("goto {} with {}", at(target), args.len())
                }
                other => format!("{other:?}"),
            };
            let params: Vec<_> = b.params.iter().map(|(_, ty)| ty).collect();
            mask(format!("{params:?} {instructions:?} {end}"))
        })
        .collect()
}

#[test]
fn the_mir_of_an_if_expression_is_a_two_arm_match_without_its_tag_test() {
    let if_form = canonical(
        "fn sel(c: Bool, a: u32, b: u32) -> u32 { let v = if c { a } else { b }; return v; }
fn main() -> i32 { return 0; }",
    );
    let match_form = canonical(
        "enum Flag { Yes, No }
fn sel(c: read Flag, a: u32, b: u32) -> u32 { let v = match c { Flag::Yes => a, Flag::No => b }; return v; }
fn main() -> i32 { return 0; }",
    );
    assert_eq!(if_form.len(), 4, "{if_form:#?}");
    assert_eq!(if_form, match_form);
}

/// `(status, chosen_plan, first_rejected_obligation)` of every region `explain` reports.
fn receipt(source: &str) -> Vec<(String, String, String)> {
    let program = mir::lower(&checked(source)).unwrap();
    let report = schedule::regions(source.as_bytes(), &program)
        .unwrap()
        .to_json();
    let text = |value: &serde_json::Value| value.as_str().unwrap_or("").to_owned();
    report["regions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            (
                text(&r["status"]),
                text(&r["chosen_plan"]),
                text(&r["first_rejected_obligation"]),
            )
        })
        .collect()
}

const LOOPS: &str = "enum Pick { A(u32), B(u32) }
fn main() -> i32 {
    let xs: Array<u32> = filled_u32(64u32, 7u32);
    var ys: Array<u32> = filled_u32(64u32, 0u32);
    var acc: u32 = 0u32;
    for i in 0u32 .. len(xs) { BODY }
    var sum: u32 = 0u32;
    for j in 0u32 .. len(xs) { sum = wrapping_add(sum, xs[j]); }
    if sum == 448u32 and acc == 448u32 and ys[0u32] == 7u32 { return 1; }
    return 0;
}";

/// The if expression, the guarded match and the if statement forms of one loop body
/// agree everywhere; the if expression gets the match's receipts, and its loop the
/// statement form's. A value join still leaves its function without a region tree,
/// so the branch-free sibling loop is serial with either expression form.
fn same_receipts(forms: [&str; 3]) {
    let receipts = forms.map(|body| {
        let source = LOOPS.replace("BODY", body);
        agree(&source, Ok(1), true);
        let program = mir::lower(&checked(&source)).unwrap();
        for workers in [1, 4] {
            let run =
                schedule::run_source(source.as_bytes(), &program, 1_000_000, workers).unwrap();
            assert_eq!(run.observation.result, Ok(eval::Value::I32(1)), "{source}");
        }
        receipt(&source)
    });
    assert_eq!(
        receipts[0], receipts[1],
        "if expression and match\n{}",
        forms[0]
    );
    assert_eq!(
        receipts[0][0], receipts[2][0],
        "if expression and if statement\n{}",
        forms[0]
    );
}

#[test]
fn a_loop_holding_an_if_expression_has_the_receipt_of_the_equivalent_match() {
    same_receipts([
        "acc = wrapping_add(acc, if xs[i] > 3u32 { xs[i] } else { 0u32 }); ys[i] = xs[i];",
        "let o: Pick = Pick::A(xs[i]); acc = wrapping_add(acc, match o { Pick::A(a) if a > 3u32 => a, Pick::A(b) => 0u32, Pick::B(c) => c }); ys[i] = xs[i];",
        "if xs[i] > 3u32 { acc = wrapping_add(acc, xs[i]); } else { acc = wrapping_add(acc, 0u32); } ys[i] = xs[i];",
    ]);
    same_receipts([
        "ys[i] = if xs[i] > 3u32 { xs[i] } else { 0u32 }; acc = wrapping_add(acc, xs[i]);",
        "let o: Pick = Pick::A(xs[i]); ys[i] = match o { Pick::A(a) if a > 3u32 => a, Pick::A(b) => 0u32, Pick::B(c) => c }; acc = wrapping_add(acc, xs[i]);",
        "if xs[i] > 3u32 { ys[i] = xs[i]; } else { ys[i] = 0u32; } acc = wrapping_add(acc, xs[i]);",
    ]);
}
