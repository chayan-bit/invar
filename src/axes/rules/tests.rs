//! #345: the rule table, its spec row for row, and the gate end to end. A rule
//! tagged above C0 on this thread stands in for a rule a later level adds.

use std::{cell::RefCell, collections::BTreeSet};

use serde_json::{Value, json};

use super::*;
use crate::schedule::normal_form::{Route, route};

thread_local! {
    static TAGGED: RefCell<Vec<(String, CLevel)>> = const { RefCell::new(Vec::new()) };
}

pub(super) fn tagged(rule: &str) -> Option<CLevel> {
    TAGGED.with(|tags| {
        let tags = tags.borrow();
        tags.iter()
            .find(|(name, _)| name == rule)
            .map(|(_, level)| *level)
    })
}

/// Run `body` with `rules` tagged at `level` on this thread only.
fn with_tags<T>(rules: &[String], level: CLevel, body: impl FnOnce() -> T) -> T {
    let tags = rules.iter().map(|rule| (rule.clone(), level)).collect();
    TAGGED.with(|cell| *cell.borrow_mut() = tags);
    let result = body();
    TAGGED.with(|cell| cell.borrow_mut().clear());
    result
}

/// One program per plan rule, each admitted today; the rule is the one its
/// plan adds over the others.
const PLANS: [(&str, &str); 7] = [
    (
        crate::admission::RULE_REVISION,
        "fn main() -> i32 { let a = filled_u32(64u32, 1u32); var out = filled_u32(64u32, 0u32); for i in 0u32..64u32 { out[i] = wrapping_add(a[i], 3u32); } if out[5u32] == 4u32 { return 1; } return 0; }",
    ),
    (
        crate::typed_net::MAP_FUSION_RULE,
        "fn main() -> i32 { let xs: Array<u32> = filled_u32(64u32,1u32); var a: Array<u32> = filled_u32(len(xs),0u32); for i in 0u32 .. len(xs) { a[i] = wrapping_add(xs[i],3u32); } let first: Array<u32> = move a; var b: Array<u32> = filled_u32(len(first),0u32); for j in 0u32 .. len(first) { b[j] = wrapping_mul(first[j],7u32); } if b[0u32] != 28u32 { return 1; } return 0; }",
    ),
    (
        crate::typed_net::DAG_FUSION_RULE,
        "fn main() -> i32 { let xs: Array<u32> = [1u32,2u32,3u32]; let ws: Array<u32> = [4u32,0u32,6u32]; var o0: Array<u32> = filled_u32(len(xs),0u32); for i in 0u32 .. len(xs) { o0[i] = wrapping_add(xs[i],ws[i]); } let s0: Array<u32> = move o0; var o1: Array<u32> = filled_u32(len(s0),0u32); for j in 0u32 .. len(s0) { o1[j] = wrapping_add(s0[j],ws[j]); } let s1: Array<u32> = move o1; var o2: Array<u32> = filled_u32(len(s1),0u32); for k in 0u32 .. len(s1) { o2[k] = wrapping_add(s1[k],ws[k]); } let s2: Array<u32> = move o2; if s2[0u32] != 13u32 { return 1; } return 0; }",
    ),
    (
        crate::laws::RULE_REVISION,
        "fn main() -> i32 { let xs = filled_u32(64u32, 7u32); var acc: u32 = 0u32; for i in 0u32..len(xs) { acc = wrapping_add(acc, xs[i]); } if acc == 448u32 { return 1; } return 0; }",
    ),
    (
        crate::pair_reduction::RULE_REVISION,
        "fn main() -> i32 { let xs: Array<u32> = [3u32, 5u32, 7u32]; let ys: Array<u32> = [1u32, 2u32, 4u32]; var a: u32 = 1u32; var b: u32 = 0u32; for i in 0u32..len(xs) { b = wrapping_add(wrapping_mul(xs[i], b), ys[i]); a = wrapping_mul(xs[i], a); } if a == 105u32 { return 1; } return 0; }",
    ),
    (
        crate::recurrence::RULE_REVISION,
        "fn main() -> i32 { let aa: Array<u32> = [2u32, 3u32, 5u32]; let bb: Array<u32> = [7u32, 11u32, 13u32]; var state: u32 = 17u32; var out: Array<u32> = filled_u32(len(aa), 0u32); for i in 0u32 .. len(aa) { state = wrapping_add(wrapping_mul(aa[i], state), bb[i]); out[i] = state; } if out[2u32] != 683u32 { return 1; } return 0; }",
    ),
    (
        crate::checked_fold::RULE_REVISION,
        "fn main() -> i32 { let a: Array<i32> = [2147483647, 1, -5]; var s: i32 = 0; for i in 0u32..len(a) { s = s + a[i]; } return s; }",
    ),
];

/// A region attribute that puts `main` at C1 and C2.
const C1: &str = "#[region(contract = \"f32-tree-v2\")]\n";
const C2: &str = "#[region(contract = \"f32-relaxed-v2\")]\n";

fn compile(source: &str) -> crate::mir::Program {
    let parsed = crate::front::parse(source.as_bytes()).unwrap();
    crate::mir::lower(&crate::check::check(parsed, true).unwrap()).unwrap()
}

/// The explain document of `source`, checked against the v6 schema.
fn explain(source: &str) -> Value {
    let program = compile(source);
    let mut document = crate::schedule::regions(source.as_bytes(), &program)
        .unwrap()
        .to_json();
    document["schema"] = json!(crate::admission::SCHEMA);
    let schema: Value =
        serde_json::from_str(include_str!("../../../schema/explain-v6.schema.json")).unwrap();
    assert!(jsonschema::is_valid(&schema, &document), "{document}");
    document
}

/// The regions of `document` whose plan or refusal names `rule`.
fn naming<'a>(document: &'a Value, rule: &str) -> Vec<&'a Value> {
    let regions = document["regions"].as_array().unwrap().iter();
    regions
        .filter(|region| {
            region["rule_identities"]
                .as_array()
                .unwrap()
                .contains(&json!(rule))
                || region["c_refused"]["rule"] == rule
        })
        .collect()
}

/// `RULES` is the set of rules the plans name, each once, and the spec's rule
/// table says the same, row for row; every normal-form law is C0.
#[test]
fn every_plan_rule_has_one_level_and_the_spec_table_says_the_same() {
    let names: BTreeSet<_> = RULES.iter().map(|(name, _)| *name).collect();
    assert_eq!(names.len(), RULES.len());
    let mut named = BTreeSet::new();
    for (_, source) in PLANS {
        for region in explain(source)["regions"].as_array().unwrap() {
            assert_eq!(region["status"], "admitted", "{region}");
            let rules = region["rule_identities"].as_array().unwrap();
            named.extend(rules.iter().map(|rule| rule.as_str().unwrap().to_owned()));
            assert_eq!(region["rule_levels"], json!(vec!["c0"; rules.len()]));
        }
    }
    assert_eq!(named, names.iter().map(|name| (*name).to_owned()).collect());
    let spec = include_str!("../../../docs/spec/invar-v0.2-regions.md");
    let table = spec.split("| Rule | Level | Plan |").nth(1).unwrap();
    let rows: Vec<(&str, &str)> = table
        .lines()
        .skip(2)
        .take_while(|line| line.starts_with('|'))
        .map(|line| {
            let cells: Vec<_> = line.split('|').map(str::trim).collect();
            (cells[1].trim_matches('`'), cells[2])
        })
        .collect();
    let mut expected: Vec<_> = RULES
        .iter()
        .map(|(name, level)| (*name, level.name()))
        .collect();
    let mut laws = LawId::ALL.to_vec();
    laws.push(LawId::L8FiniteState);
    assert!(laws.iter().all(|law| law_level(*law) == CLevel::C0));
    expected.push(("L1 to L20", CLevel::C0.name()));
    assert_eq!(rows, expected);
    let unknown = receipt(&["no-such-rule-v1"], Some("no-such-rule-v1"));
    assert_eq!(unknown["rule_levels"], json!([null]));
    assert_eq!(
        unknown["c_refused"],
        json!({"rule": "no-such-rule-v1", "c_level": null})
    );
}

/// The value, trap, fuel and cleanup of a run equal the serial reference on
/// the CPU scheduler at 1, 2 and 16 workers, and no region was published.
fn runs_serial_as_the_reference(source: &str) {
    let program = compile(source);
    let reference = crate::mir_eval::observe(&program, crate::eval::DEFAULT_FUEL);
    for workers in [1, 2, 16] {
        let run = crate::schedule::run_source(
            source.as_bytes(),
            &program,
            crate::eval::DEFAULT_FUEL,
            workers,
        )
        .unwrap();
        let observed = &run.observation;
        assert_eq!(
            format!("{:?}", observed.result),
            format!("{:?}", reference.result)
        );
        assert_eq!(observed.steps_used, reference.steps_used, "{source}");
        assert_eq!(observed.cleanup_trace, reference.cleanup_trace, "{source}");
        assert!(run.executed.is_empty(), "{source}");
    }
}

/// A rule tagged C2 is refused in a C0 and a C1 region with `CONTRACT_LEVEL`
/// and runs as the serial reference; a C2 region admits it again.
#[test]
fn machine_diagnostics_contract_level_refuses_a_rule_above_the_region_level() {
    for (rule, source) in PLANS {
        assert!(!naming(&explain(source), rule).is_empty(), "{rule}");
        with_tags(&[rule.to_owned()], CLevel::C2, || {
            for declared in ["", C1] {
                let source = format!("{declared}{source}");
                let document = explain(&source);
                let refused = naming(&document, rule);
                assert!(!refused.is_empty(), "{rule}");
                for region in refused {
                    assert_eq!(region["status"], "rejected", "{region}");
                    assert_eq!(region["chosen_plan"], "mir-serial", "{region}");
                    assert_eq!(region["first_rejected_obligation"], OBLIGATION);
                    assert_eq!(region["c_refused"], json!({"rule": rule, "c_level": "c2"}));
                    let serial = rule_level(crate::admission::RULE_REVISION);
                    assert_eq!(region["rule_levels"], json!([serial.map(CLevel::name)]));
                }
                runs_serial_as_the_reference(&source);
            }
            let document = explain(&format!("{C2}{source}"));
            for region in naming(&document, rule) {
                assert_eq!(region["status"], "admitted", "{region}");
                assert_eq!(region.get("c_refused"), None);
                let levels = region["rule_levels"].as_array().unwrap();
                assert!(levels.contains(&json!("c2")), "{region}");
            }
        });
    }
    let diagnostic = crate::diagnostics::diagnostic(
        OBLIGATION,
        "a rule tagged above the region's level",
        "fixture.inv",
        PLANS[0].1.as_bytes(),
        None,
    );
    let document = crate::diagnostics::document("tool_error", vec![diagnostic]);
    let schema: Value = serde_json::from_str(include_str!(
        "../../../schema/invar-diagnostics-v3.schema.json"
    ))
    .unwrap();
    assert!(jsonschema::is_valid(&schema, &document), "{document}");
}

/// The normal-form route never applies a law above the lowest level a
/// function declares, and falls back with its own reason.
#[test]
fn the_normal_form_route_refuses_a_law_above_the_program_level() {
    let sum = "fn f(xs: read Array<i32>, n: u32, s0: i32) -> i32 { var s = s0; for i in 0u32 .. n { s = s * xs[i]; } return s; } fn main() -> i32 { let xs: Array<i32> = [4, 9, (0 - 3), 12]; return f(xs, 4u32, 2) + 2634; }";
    let fuel = crate::eval::DEFAULT_FUEL;
    assert!(matches!(route(&compile(sum), fuel), Route::Routed(_)));
    let mut laws = LawId::ALL.to_vec();
    laws.push(LawId::L8FiniteState);
    let laws: Vec<_> = laws.iter().map(ToString::to_string).collect();
    with_tags(&laws, CLevel::C2, || {
        let refused = route(&compile(sum), fuel);
        assert!(matches!(refused, Route::Fallback("contract-level")));
        let both = sum.replace("fn ", &format!("{C2}fn "));
        assert!(matches!(route(&compile(&both), fuel), Route::Routed(_)));
        let one = format!("{C2}{sum}");
        assert!(matches!(
            route(&compile(&one), fuel),
            Route::Fallback("contract-level")
        ));
    });
}
