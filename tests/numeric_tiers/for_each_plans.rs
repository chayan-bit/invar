//! #344: a for-each loop is its counted form with `let x = xs[i]` (NB.7.6),
//! so it admits the plan, receipt and results of that counted form and of the
//! form that reads `xs[i]` in place, for every sealed fold law, with and
//! without a `#[region]` declaration, on every engine and worker count.

use std::process::Command;

use invar::{check::check, front::parse, mir, mir_eval, schedule};
use serde_json::Value;

/// Strict, Tree and Exact folds of these give 16777216, 16777218 and 16777224.
const F32_ELEMENTS: &str = "16777216.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0";

/// (element type, seed, update of `s` by the element `{e}`, function attribute, elements).
const FOLDS: &[(&str, &str, &str, &str, &str)] = &[
    ("u32", "0u32", "s = wrapping_add(s, {e});", "", "4294967295u32, 2u32, 3u32"),
    ("u32", "0u32", "s = wrapping_add({e}, s);", "#[region(n = \"n1\")]", "7u32, 9u32"),
    ("u32", "1u32", "s = wrapping_mul(s, {e});", "", "3u32, 5u32, 7u32"),
    ("i32", "0", "s = s + {e};", "", "1, -2, 3"),
    ("i32", "0", "s += {e};", "#[region(n = \"n2\")]", "2147483647, 1"),
    ("u32", "0u32", "s = s + {e};", "", "4294967295u32, 1u32"),
    ("u32", "5u32", "s = s + {e};", "#[region(n = \"n1\")]", "1u32, 2u32"),
    ("f32", "0.0", "s = s + {e};", "", F32_ELEMENTS),
    ("f32", "0.0", "s = s + {e};", "#[region(contract = \"f32-tree-v2\")]", F32_ELEMENTS),
    ("f32", "0.0", "s = s + {e};", "#[region(contract = \"f32-exact-v2\")]", F32_ELEMENTS),
    ("f32", "0.0", "s = s + {e};", "#[numeric(fold = \"relaxed\")]", F32_ELEMENTS),
];

/// The for-each loop, its counted form (NB.7.6) and the in-place counted form.
fn forms(update: &str) -> [String; 3] {
    let at = |e: &str| update.replace("{e}", e);
    [
        format!("for x in xs {{ {} }}", at("x")),
        format!("for i in 0u32 .. len(xs) {{ let x = xs[i]; {} }}", at("x")),
        format!("for i in 0u32 .. len(xs) {{ {} }}", at("xs[i]")),
    ]
}

fn program(fold: &(&str, &str, &str, &str, &str), body: &str, empty: bool) -> String {
    let (ty, seed, _, attribute, elements) = *fold;
    let elements = if empty { "" } else { elements };
    let result = match ty {
        "i32" => "return r;",
        "u32" => "return code(r);",
        _ => "if r == 0.0 { return 4; } if r == 16777216.0 { return 1; } if r == 16777218.0 { return 2; } \
              if r == 16777224.0 { return 3; } return 0;",
    };
    format!(
        "{CODE}{attribute} fn fold_all(xs: read Array<{ty}>) -> {ty} {{ var s: {ty} = {seed}; {body} return s; }} \
         fn main() -> i32 {{ let xs: Array<{ty}> = [{elements}]; let r = fold_all(xs); {result} }}"
    )
}

/// `main` returns `i32`: a `u32` leaves through an injective code (low 31
/// bits, the top bit as the sign).
const CODE: &str = "fn code(s: u32) -> i32 { var r: i32 = 0; var p: i32 = 1; var q: u32 = s; \
    for b in 0u32 .. 31u32 { if q % 2u32 == 1u32 { r = r + p; } q = q / 2u32; if b < 30u32 { p = p * 2; } } \
    if q == 1u32 { return -1 - r; } return r; } ";

/// The `explain` region records of `source`, without the fields that name
/// source bytes: `span`, `source_identity` and `candidate_identity`.
fn regions(source: &str) -> Vec<Value> {
    let path = std::env::temp_dir().join(format!("invar-344-{}.inv", std::process::id()));
    std::fs::write(&path, source).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["explain", path.to_str().unwrap(), "--format", "json"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{source}");
    let document: Value = serde_json::from_slice(&output.stdout).unwrap();
    let mut regions = document["regions"].as_array().unwrap().clone();
    for region in &mut regions {
        let fields = region.as_object_mut().unwrap();
        fields.retain(|key, _| !["span", "source_identity", "candidate_identity"].contains(&key.as_str()));
    }
    regions
}

fn lowered(source: &str) -> mir::Program {
    let parsed = parse(source.as_bytes()).unwrap_or_else(|e| panic!("{source}: {e:?}"));
    mir::lower(&check(parsed, true).unwrap()).unwrap()
}

#[test]
fn a_for_each_fold_admits_the_plan_of_its_counted_form() {
    for fold in FOLDS {
        for empty in [false, true] {
            let [each, counted, in_place] = forms(fold.2).map(|body| program(fold, &body, empty));
            let records = regions(&each);
            assert!(!records.is_empty(), "{each}");
            assert_eq!(records, regions(&counted), "{each}");
            assert_eq!(records, regions(&in_place), "{each}");
        }
    }
}

/// `text` without its `span: Span { .. }` fields.
fn without_spans(text: &str) -> String {
    let mut rest = text;
    let mut kept = String::new();
    while let Some(at) = rest.find("span: Span {") {
        kept.push_str(&rest[..at]);
        rest = &rest[at..];
        rest = &rest[rest.find('}').map_or(rest.len(), |end| end + 1)..];
    }
    kept + rest
}

#[test]
fn a_for_each_fold_keeps_value_trap_and_fuel_on_every_engine() {
    for fold in FOLDS {
        for empty in [false, true] {
            let [each, counted, _] = forms(fold.2).map(|body| program(fold, &body, empty));
            let (program, twin) = (lowered(&each), lowered(&counted));
            let fuel = invar::eval::DEFAULT_FUEL;
            let serial = mir_eval::observe(&program, fuel);
            let twin = mir_eval::observe(&twin, fuel);
            let observed = |result: &dyn std::fmt::Debug, steps: u64| format!("{result:?} {steps}");
            // The two forms trap at the same update, at their own source bytes.
            assert_eq!(
                without_spans(&observed(&serial.result, serial.steps_used)),
                without_spans(&observed(&twin.result, twin.steps_used)),
                "{each}"
            );
            for workers in [1, 2, 16] {
                let run = schedule::run_source(each.as_bytes(), &program, fuel, workers).unwrap();
                assert_eq!(observed(&run.observation.result, run.observation.steps_used), observed(&serial.result, serial.steps_used), "{each} at {workers}");
            }
        }
    }
}

/// The wrapping and checked folds publish their parallel region, as the
/// counted form does.
#[test]
fn a_for_each_fold_runs_its_parallel_plan() {
    for fold in FOLDS.iter().filter(|fold| fold.0 != "f32") {
        let each = program(fold, &forms(fold.2)[0], false);
        let records = regions(&each);
        let record = records.iter().find(|r| r["function"] == "fold_all").unwrap();
        assert_eq!(record["status"], "admitted", "{each}: {record}");
        let run = schedule::run_source(each.as_bytes(), &lowered(&each), invar::eval::DEFAULT_FUEL, 2).unwrap();
        assert_eq!(run.executed.len(), 1, "{each}");
    }
}

/// A for-each loop over a `View` or a `MutView`, by value or by place, admits
/// the record of its counted form.
#[test]
fn a_for_each_over_a_view_admits_the_record_of_its_counted_form() {
    let shell = |setup: &str, body: &str| {
        format!(
            "{CODE}fn main() -> i32 {{ var ys: Array<u32> = [1u32, 2u32, 3u32, 4u32]; var s: u32 = 0u32; \
             if true {{ {setup} {body} }} return code(wrapping_add(s, ys[0])); }}"
        )
    };
    for (setup, each, counted) in [
        (
            "let w = view(ys, 1u32, 4u32);",
            "for x in w { s = wrapping_add(s, x); }",
            "for i in 0u32 .. len(w) { s = wrapping_add(s, w[i]); }",
        ),
        (
            "let w = view_mut(ys, 0u32, 3u32);",
            "for x in w { x = wrapping_add(x, 1u32); }",
            "for x in 0u32 .. len(w) { w[x] = wrapping_add(w[x], 1u32); }",
        ),
    ] {
        let (each, counted) = (shell(setup, each), shell(setup, counted));
        assert_eq!(regions(&each), regions(&counted), "{each}");
        let fuel = invar::eval::DEFAULT_FUEL;
        let (a, b) = (mir_eval::observe(&lowered(&each), fuel), mir_eval::observe(&lowered(&counted), fuel));
        assert_eq!(format!("{:?}", a.result), format!("{:?}", b.result), "{each}");
    }
}

/// An element is read in place only while the loop cannot change it: a loop
/// that writes its array keeps the binding, so it is never admitted as a fold
/// of the array it changes.
#[test]
fn an_element_of_an_array_the_loop_writes_is_not_forwarded() {
    let source = &format!("{CODE}fn main() -> i32 {{ var xs: Array<u32> = [1u32, 2u32, 3u32]; var s: u32 = 0u32; \
        for i in 0u32 .. len(xs) {{ let x = xs[i]; xs[i] = 9u32; s = wrapping_add(s, x); }} return code(s); }}");
    let records = regions(source);
    assert_ne!(records[0]["status"], "admitted", "{records:?}");
    let program = lowered(source);
    let fuel = invar::eval::DEFAULT_FUEL;
    let serial = mir_eval::observe(&program, fuel);
    let run = schedule::run_source(source.as_bytes(), &program, fuel, 2).unwrap();
    assert_eq!(format!("{:?}", run.observation.result), format!("{:?}", serial.result));
}

/// A for-each `f32` fold takes its tier's bits: strict 1, Tree 2, Exact 3.
#[test]
fn a_for_each_fold_takes_its_fold_tier() {
    for (fold, expected) in FOLDS.iter().filter(|fold| fold.0 == "f32").zip([1, 2, 3, 2]) {
        let each = program(fold, &forms(fold.2)[0], false);
        let path = std::env::temp_dir().join(format!("invar-344-run-{}.inv", std::process::id()));
        std::fs::write(&path, &each).unwrap();
        for engine in [&["--engine", "reference"][..], &["--engine", "cpu-parallel", "--workers", "2"]] {
            let output = Command::new(env!("CARGO_BIN_EXE_invar"))
                .args(["run", path.to_str().unwrap(), "--format", "json"])
                .args(engine)
                .output()
                .unwrap();
            let document: Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(document["result"]["value"], expected, "{each} {engine:?}: {document}");
        }
    }
}
