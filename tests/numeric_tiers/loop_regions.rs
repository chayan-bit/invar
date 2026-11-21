//! #355: `#[region(...)]` on a loop (`docs/spec/invar-v0.2-regions.md`
//! section 9). The same accept and refuse set as a function declaration, the
//! loop and its nested loops as the region, per-loop fold tiers on every
//! engine, the unchanged plan of a safe declaration, caps and the report.

use super::region_axes::{DECLARATIONS, cli, code, run, scratch};
use invar::axes::{Axes, C_AXIS};
use serde_json::{Value, json};

/// `fold_sum` with `attribute` on its loop instead of on the function: `main`
/// returns 2, 3 or 1 as the Tree, Exact or Strict fold ran.
fn loop_program(attribute: &str) -> String {
    format!(
        "fn fold_sum(xs: read Array<f32>) -> f32 {{ var acc: f32 = 0.0; \
         {attribute} for i in 0u32..len(xs) {{ acc = acc + xs[i]; }} return acc; }} \
         fn main() -> i32 {{ let xs: Array<f32> = [16777216.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; \
         let r = fold_sum(xs); if r == 16777218.0 {{ return 2; }} if r == 16777224.0 {{ return 3; }} \
         if r == 16777216.0 {{ return 1; }} return 0; }}"
    )
}

fn explain(source: &str) -> Value {
    let path = scratch("loop", source);
    let (status, stdout) = cli(&["explain", path.to_str().unwrap(), "--format", "json"]);
    assert_eq!(status, 0, "{stdout}");
    serde_json::from_str(&stdout).unwrap()
}

/// Every declaration refused on a function is refused on a loop with the same
/// code; `#[numeric]` stays function-only, and `#[region]` takes no other statement.
#[test]
fn a_loop_takes_the_declarations_and_refusals_of_a_function() {
    for &(attribute, on_function) in DECLARATIONS {
        let expected = if attribute.contains("#[numeric") {
            "UNSUPPORTED_FEATURE"
        } else {
            on_function
        };
        assert_eq!(code(&loop_program(attribute)), expected, "{attribute}");
    }
    let let_binding = "fn main() -> i32 { #[region(n = \"n1\")] let x: i32 = 1; return x; }";
    assert_eq!(code(let_binding), "UNSUPPORTED_FEATURE");
    let labelled = "fn main() -> i32 { var s: u32 = 0u32; #[region(n = \"n1\")] outer: while s < 3u32 { s = s + 1u32; } return 0; }";
    assert_eq!(code(labelled), "OK");
    let parsed = invar::front::parse(labelled.as_bytes()).unwrap();
    let loops = &parsed.functions[0].loop_axes;
    assert_eq!(loops.len(), 1);
    assert_eq!(&labelled[loops[0].0.start..loops[0].0.start + 5], "while");
}

/// A declared loop governs itself and the loops nested in it; an inner
/// declaration overrides it; other loops keep the function's levels.
#[test]
fn a_loop_declaration_governs_its_loop_and_the_loops_inside_it() {
    let source = "fn f(xs: read Array<u32>) -> u32 { var s: u32 = 0u32; \
        for a in 0u32..len(xs) { s = wrapping_add(s, xs[a]); } \
        #[region(contract = \"f32-relaxed-v2\", n = \"n1\")] for b in 0u32..2u32 { \
          for c in 0u32..len(xs) { s = wrapping_add(s, xs[c]); } \
          #[region(c = \"c0\")] for d in 0u32..len(xs) { s = wrapping_add(s, xs[d]); } } \
        return s; } \
        fn main() -> i32 { let xs: Array<u32> = [1u32, 2u32]; if f(xs) == 15u32 { return 1; } return 0; }";
    let document = explain(source);
    let levels: Vec<_> = document["regions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|region| {
            let start = region["span"]["start"].as_u64().unwrap() as usize;
            let head = &source[start..start + 5];
            (
                head.to_owned(),
                region["c_level"].clone(),
                region["n_level"].clone(),
            )
        })
        .collect();
    let expected = [
        ("for a", "c0", "n0"),
        ("for b", "c2", "n1"),
        ("for c", "c2", "n1"),
        ("for d", "c0", "n0"),
    ]
    .map(|(head, c, n)| (head.to_owned(), json!(c), json!(n)));
    assert_eq!(levels, expected);
    assert_eq!(
        document["numeric_contracts"]["levels"],
        json!({"c": ["c0", "c2"], "s": ["s0"], "n": ["n0", "n1"], "t": ["t0"]})
    );
    let bound = &document["regions"][1]["bound"];
    assert!(
        bound.as_str().is_some_and(|b| b.contains("2^-24")),
        "{bound}"
    );
    assert_eq!(run(source, &["--engine", "reference"]), 1);
}

/// A loop's fold contract selects its tier for that loop alone, with the bits
/// of the same contract on a function, on every engine and worker count; a C0
/// loop in a Tree function folds strict.
#[test]
fn a_declared_loop_folds_at_its_tier_on_every_engine() {
    let mixed = "#[numeric(fold = \"tree\")] fn mixed(xs: read Array<f32>) -> i32 { \
        var t: f32 = 0.0; for i in 0u32..len(xs) { t = t + xs[i]; } \
        var s: f32 = 0.0; #[region(c = \"c0\")] for j in 0u32..len(xs) { s = s + xs[j]; } \
        if t != 16777218.0 { return 5; } if s != 16777216.0 { return 6; } return 1; } \
        fn main() -> i32 { let xs: Array<f32> = [16777216.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; return mixed(xs); }";
    for (source, value) in [
        (loop_program(""), 1),
        (loop_program("#[region(n = \"n2\")]"), 1),
        (loop_program("#[region(contract = \"f32-tree-v2\")]"), 2),
        (loop_program("#[region(contract = \"f32-exact-v2\")]"), 3),
        (loop_program("#[region(contract = \"f32-relaxed-v2\")]"), 2),
        (mixed.to_owned(), 1),
    ] {
        assert_eq!(run(&source, &["--engine", "reference"]), value, "{source}");
        for workers in ["1", "2", "16"] {
            let parallel = ["--engine", "cpu-parallel", "--workers", workers];
            assert_eq!(run(&source, &parallel), value, "{source}");
        }
    }
    // The fold receipt (D12) names the loop's tier and the algorithm that ran.
    let receipt = |source: &str| {
        let parsed = invar::front::parse(source.as_bytes()).unwrap();
        let checked = invar::check::check(parsed, true).unwrap();
        invar::math::fold_receipt(&invar::mir::lower(&checked).unwrap())
    };
    let exact = loop_program("#[region(contract = \"f32-exact-v2\")]");
    assert_eq!(receipt(&exact), Some(json!({"exact": "exact"})));
    assert_eq!(receipt(mixed), Some(json!({"tree": "tree"})));
}

/// A safe loop declaration admits the same regions, plans and identities as
/// none, and a sugared and a sugar-free loop admit the same plan under one.
#[test]
fn a_safe_loop_declaration_admits_the_same_plan_and_identity() {
    let body = |attribute: &str, update: &str| {
        format!(
            "fn sum(xs: read Array<i32>) -> i32 {{ var s: i32 = 0; {attribute} \
             for i in 0u32..len(xs) {{ {update} }} return s; }} \
             fn main() -> i32 {{ let xs: Array<i32> = [1, 2, 3]; return sum(xs); }}"
        )
    };
    let declared = body(
        "#[region(c = \"c0\", s = \"s0\", n = \"n0\", t = \"t0\")]",
        "s = s + xs[i];",
    );
    let attribute = declared.find("#[region").unwrap();
    let end = declared[attribute..].find(']').unwrap() + attribute + 1;
    let blank = format!(
        "{}{}{}",
        &declared[..attribute],
        " ".repeat(end - attribute),
        &declared[end..]
    );
    let regions = |source: &str| explain(source)["regions"].clone();
    let plan = regions(&declared);
    assert_eq!(
        plan[0]["chosen_plan"], "checked-sum-carry-if-guard",
        "{plan}"
    );
    assert_eq!(plan, regions(&blank));
    let raised = |update: &str| {
        let mut region = regions(&body("#[region(n = \"n1\")]", update))[0].clone();
        let fields = region.as_object_mut().unwrap();
        fields.retain(|key, _| !["span", "candidate_identity"].contains(&key.as_str()));
        region
    };
    assert_eq!(raised("s = s + xs[i];"), raised("s += xs[i];"));
    assert_eq!(raised("s += xs[i];")["n_level"], "n1");
}

/// Caps refuse a loop above them at its attribute, and the region report lists
/// each declared loop under its function.
#[test]
fn caps_and_the_region_report_cover_declared_loops() {
    let path = scratch(
        "loop-cap",
        &loop_program("#[region(contract = \"f32-tree-v2\")]"),
    );
    let path = path.to_str().unwrap();
    let (status, stdout) = cli(&["check", path, "--max-c", "c0", "--format", "json"]);
    let document: Value = serde_json::from_str(&stdout).unwrap();
    assert_ne!(status, 0);
    assert_eq!(
        document["diagnostics"][0]["code"], "CONTRACT_CAP",
        "{document}"
    );
    let (status, stdout) = cli(&["check", path, "--max-c", "c1", "--format", "json"]);
    assert_eq!(status, 0, "{stdout}");
    let (status, stdout) = cli(&["check", path, "--region-report", "--format", "json"]);
    assert_eq!(status, 0, "{stdout}");
    let report: Value = serde_json::from_str(&stdout).unwrap();
    let rows = report["regions"].as_array().unwrap();
    assert_eq!(rows.len(), 3, "{report}");
    assert_eq!(rows[0]["function"], "fold_sum");
    assert_eq!(rows[0].get("loop"), None);
    assert_eq!(rows[1]["function"], "fold_sum");
    assert_eq!(
        (&rows[1]["c"], &rows[1]["contract"]),
        (&json!("c1"), &json!("f32-tree-v2"))
    );
    assert!(rows[1]["loop"]["start"].is_u64(), "{report}");
}

/// A loop declaration reaches no callee, a closure created in a declared loop
/// is safe-level, and a loop declared in a closure body stays with that closure.
#[test]
fn a_loop_declaration_reaches_neither_callee_nor_closure() {
    let source = "fn callee(x: u32) -> u32 { var s: u32 = 0u32; for i in 0u32..x { s = wrapping_add(s, i); } return s; } \
        fn main() -> i32 { var t: u32 = 0u32; \
        #[region(n = \"n1\")] for j in 0u32..2u32 { \
          let c = closure [copy j: u32] (y: u32) -> u32 { var u: u32 = y; \
            #[region(n = \"n2\")] for k in 0u32..2u32 { u = wrapping_add(u, k); } return u; }; \
          t = wrapping_add(t, callee(c(j))); } \
        if t == 1u32 { return 1; } return 0; }";
    let checked =
        invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap();
    for function in &checked.program.functions {
        let mut levels: Vec<_> = function
            .loop_axes
            .iter()
            .map(|(_, _, axes)| axes.level(&invar::axes::N_AXIS))
            .collect();
        levels.sort_unstable();
        let expected: &[&str] = match function.name.as_str() {
            "main" => &["n1", "n2"],
            "callee" => &[],
            _ => &["n2"],
        };
        assert_eq!(levels, expected, "{}", function.name);
        assert_eq!(function.axes, Axes::default(), "{}", function.name);
    }
    let lowered = invar::mir::lower(&checked).unwrap();
    assert!(!lowered.loop_axes.contains_key("callee"));
    let lifted = lowered
        .loop_axes
        .keys()
        .find(|name| !matches!(name.as_str(), "main"));
    assert!(lifted.is_some(), "{:?}", lowered.loop_axes);
    for function in lowered.functions.iter().filter(|f| f.name == "callee") {
        let axes = invar::axes::region_levels(&lowered, &function.name, function.span);
        assert_eq!(axes.level(&C_AXIS), "c0");
    }
    assert_eq!(run(source, &["--engine", "reference"]), 1);
}
