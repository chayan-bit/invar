use invar::fixes::{self, Candidate, Edit, Recheck};
use invar::front::{self, Span};

use super::common::first_error;

fn span(source: &str, needle: &str) -> Span {
    let start = source.find(needle).expect("needle in source");
    Span {
        start,
        end: start + needle.len(),
    }
}

fn replace(source: &str, needle: &str, replacement: &str) -> Candidate {
    Candidate {
        description: format!("Use {replacement}"),
        edits: vec![Edit {
            span: span(source, needle),
            replacement: replacement.to_owned(),
        }],
    }
}

const ONE_ERROR: &str = "fn main() -> i32 {\n    return nope;\n}\n";
const TWO_ERRORS: &str =
    "fn a() -> i32 {\n    return nope1;\n}\nfn main() -> i32 {\n    return nope2;\n}\n";

fn kept(source: &str, candidates: Vec<Candidate>, recheck: Recheck) -> Vec<Candidate> {
    fixes::validated(source.as_bytes(), candidates, recheck, false)
}

#[test]
fn issue_dx_22_engine_accepts_a_candidate_that_rechecks_clean() {
    assert_eq!(first_error(ONE_ERROR).0, "NAME_UNKNOWN");
    let good = replace(ONE_ERROR, "nope", "0");
    assert_eq!(
        kept(ONE_ERROR, vec![good.clone()], Recheck::Clean),
        vec![good]
    );
}

#[test]
fn issue_dx_22_engine_drops_a_candidate_that_creates_another_error() {
    // Fixes `nope` but names another unknown binding at the same place.
    let swap = replace(ONE_ERROR, "nope", "other");
    for recheck in [Recheck::Clean, Recheck::OtherFunctions] {
        assert!(kept(ONE_ERROR, vec![swap.clone()], recheck).is_empty());
    }
}

#[test]
fn issue_dx_22_engine_drops_a_candidate_that_leaves_the_original_error() {
    let at = first_error(ONE_ERROR).1;
    // A harmless insertion before the error shifts it but does not fix it.
    let shift = Candidate {
        description: "Add a space".to_owned(),
        edits: vec![Edit {
            span: Span {
                start: at.start,
                end: at.start,
            },
            replacement: " ".to_owned(),
        }],
    };
    assert!(kept(ONE_ERROR, vec![shift], Recheck::OtherFunctions).is_empty());
}

#[test]
fn issue_dx_22_engine_drops_a_candidate_that_adds_a_syntax_error() {
    // `return 0 nope;` no longer parses: the error sits at the edit boundary,
    // inside the edited function, so it is not a later pre-existing error.
    let at = first_error(ONE_ERROR).1;
    let broken = Candidate {
        description: "Insert a stray literal".to_owned(),
        edits: vec![Edit {
            span: Span {
                start: at.start,
                end: at.start,
            },
            replacement: "0 ".to_owned(),
        }],
    };
    for recheck in [Recheck::Clean, Recheck::OtherFunctions] {
        assert!(kept(ONE_ERROR, vec![broken.clone()], recheck).is_empty());
    }
}

#[test]
fn issue_dx_22_engine_tolerates_an_error_in_another_function_only_when_asked() {
    assert_eq!(first_error(TWO_ERRORS).1, span(TWO_ERRORS, "nope1"));
    let fix = replace(TWO_ERRORS, "nope1", "0");
    assert!(kept(TWO_ERRORS, vec![fix.clone()], Recheck::Clean).is_empty());
    assert_eq!(
        kept(TWO_ERRORS, vec![fix.clone()], Recheck::OtherFunctions),
        vec![fix]
    );
}

#[test]
fn issue_dx_22_engine_tolerates_an_error_in_an_earlier_function_too() {
    // Fixing the second function leaves the first one's error: not caused by the edit.
    let fix = replace(TWO_ERRORS, "nope2", "0");
    assert!(kept(TWO_ERRORS, vec![fix.clone()], Recheck::Clean).is_empty());
    assert_eq!(
        kept(TWO_ERRORS, vec![fix.clone()], Recheck::OtherFunctions),
        vec![fix]
    );
}

#[test]
fn issue_dx_22_engine_does_not_tolerate_an_edit_outside_one_function_body() {
    // The signature is outside any body: only a clean recheck may accept it.
    let signature = replace(TWO_ERRORS, "fn a() -> i32", "fn a() -> u32");
    assert!(kept(TWO_ERRORS, vec![signature], Recheck::OtherFunctions).is_empty());
}

#[test]
fn issue_dx_22_engine_rejects_overlapping_edits() {
    let overlapping = vec![
        Edit {
            span: Span { start: 0, end: 4 },
            replacement: "a".to_owned(),
        },
        Edit {
            span: Span { start: 2, end: 6 },
            replacement: "b".to_owned(),
        },
    ];
    assert_eq!(fixes::apply(ONE_ERROR.as_bytes(), &overlapping), None);
    let past_end = vec![Edit {
        span: Span {
            start: ONE_ERROR.len(),
            end: ONE_ERROR.len() + 1,
        },
        replacement: String::new(),
    }];
    assert_eq!(fixes::apply(ONE_ERROR.as_bytes(), &past_end), None);
    let candidate = Candidate {
        description: "overlap".to_owned(),
        edits: overlapping,
    };
    assert!(kept(ONE_ERROR, vec![candidate], Recheck::OtherFunctions).is_empty());
}

#[test]
fn issue_dx_22_engine_gives_no_fix_for_a_source_with_a_syntax_error() {
    let source = "fn helper( -> i32 { return 1; }";
    assert!(front::parse(source.as_bytes()).is_err());
    assert!(
        fixes::suggested_fixes("ENTRYPOINT_MISSING", "p.inv", source.as_bytes(), None).is_empty()
    );
    let at_end = Span {
        start: source.len(),
        end: source.len(),
    };
    let candidate = Candidate {
        description: "Append".to_owned(),
        edits: vec![Edit {
            span: at_end,
            replacement: "\nfn main() -> i32 { return 0; }\n".to_owned(),
        }],
    };
    for recheck in [Recheck::Clean, Recheck::OtherFunctions] {
        assert!(
            fixes::validated(source.as_bytes(), vec![candidate.clone()], recheck, true).is_empty()
        );
    }
    for code in ["OWN_MOVE_REQUIRED", "OWN_MUT_REQUIRED"] {
        let json = fixes::suggested_fixes(code, "p.inv", source.as_bytes(), Some(at_end));
        assert!(json.is_empty(), "{code}");
    }
}

#[test]
fn issue_dx_22_engine_bounds_candidates_and_keeps_their_order() {
    let candidates: Vec<Candidate> = (0..12)
        .map(|n| replace(ONE_ERROR, "nope", &n.to_string()))
        .collect();
    let first = kept(ONE_ERROR, candidates.clone(), Recheck::Clean);
    let second = kept(ONE_ERROR, candidates.clone(), Recheck::Clean);
    assert_eq!(first.len(), fixes::MAX_CANDIDATES);
    assert_eq!(first, candidates[..fixes::MAX_CANDIDATES]);
    assert_eq!(first, second);
}

#[test]
fn issue_dx_22_entrypoint_fix_is_emitted_through_the_engine() {
    let source = "fn helper() -> i32 { return 1; }\n";
    let json = fixes::suggested_fixes("ENTRYPOINT_MISSING", "p.inv", source.as_bytes(), None);
    assert_eq!(json.len(), 1);
    assert_eq!(json[0]["description"], "Add an entrypoint");
    let edit = &json[0]["edits"][0];
    assert_eq!(edit["span"]["start"], source.len());
    assert_eq!(edit["replacement"], "\nfn main() -> i32 { return 0; }\n");
    let none = fixes::suggested_fixes("NAME_UNKNOWN", "p.inv", source.as_bytes(), None);
    assert!(none.is_empty());
}
