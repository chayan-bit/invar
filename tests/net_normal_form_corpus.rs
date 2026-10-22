#![forbid(unsafe_code)]
//! P1 corpus classification: every GNET-05 program (#131) and every M5-01
//! held-out candidate gets a normal form, compared loop by loop with the #133
//! prototype classification recorded in `tests/net_normal_form_fixtures/prototype.json`.
//! The independent checker must accept every certificate and the completeness
//! of every normal form.

#[path = "support/normal_form.rs"]
mod support;

use invar::general_net::normal_form::{self, Recursion, certificate};
use serde_json::Value as Json;
use std::{fs, path::PathBuf};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/net_normal_form_fixtures")
}

fn prototype() -> Json {
    serde_json::from_str(&fs::read_to_string(fixtures().join("prototype.json")).expect("fixture"))
        .expect("json")
}

fn expected_loops(row: &Json) -> Vec<(Option<String>, Vec<String>)> {
    row["loops"]
        .as_array()
        .expect("loops")
        .iter()
        .map(|l| {
            (
                l["c"].as_str().map(str::to_owned),
                l["laws"]
                    .as_array()
                    .expect("laws")
                    .iter()
                    .map(|x| x.as_str().expect("law").to_owned())
                    .collect(),
            )
        })
        .collect()
}

fn diff(id: &str, source: &str, row: &Json, out: &mut Vec<String>) -> normal_form::NormalForm {
    let (net, proof) = support::compile(source);
    let form = normal_form::extract(&net, &proof).expect("extract");
    match certificate::check_all(&net, &proof, &form) {
        Ok(n) if n == form.certificates().count() => {}
        Ok(n) => out.push(format!(
            "{id}: checker covered {n} of {} certificates",
            form.certificates().count()
        )),
        Err(rejection) => out.push(format!(
            "{id}: checker rejects the normal form: {rejection:?}"
        )),
    }
    let ours = support::loop_labels(&form);
    let theirs = expected_loops(row);
    if ours.len() != theirs.len() {
        out.push(format!(
            "{id}: {} loops vs prototype {}",
            ours.len(),
            theirs.len()
        ));
        return form;
    }
    for (k, ((c, laws, why), (pc, plaws))) in ours.iter().zip(&theirs).enumerate() {
        if c.map(str::to_owned) != *pc || (c.is_some() && laws != plaws) {
            out.push(format!(
                "{id} loop {k}: ours {c:?} {laws:?} {why:?} vs prototype {pc:?} {plaws:?}"
            ));
        }
    }
    form
}

#[test]
fn gnet05_corpus_matches_prototype_classification() {
    let proto = prototype();
    let mut differences = Vec::new();
    let mut counted = 0;
    for row in proto["gnet05"].as_array().expect("rows") {
        let id = row["id"].as_str().expect("id");
        let source = fs::read_to_string(fixtures().join("gnet05").join(format!("{id}.inv")))
            .expect("source");
        let form = diff(id, &source, row, &mut differences);
        let fclass = row["fclass"].as_object().expect("fclass");
        for function in &form.functions {
            let ours = match &function.recursion {
                Some(Recursion::PathSum { modulus }) => {
                    Some(("pathsum", Some(u64::from(*modulus))))
                }
                Some(Recursion::RangeSum) => Some(("range_sum", None)),
                None => None,
            };
            let theirs = fclass
                .get(&function.name)
                .and_then(Json::as_str)
                .map(|c| (c, row["modulus"].get(&function.name).and_then(Json::as_u64)));
            let theirs = theirs.map(|(c, p)| {
                (
                    if c == "pathsum" {
                        "pathsum"
                    } else {
                        "range_sum"
                    },
                    p,
                )
            });
            if ours != theirs {
                differences.push(format!(
                    "{id} fn {}: ours {ours:?} vs prototype {theirs:?}",
                    function.name
                ));
            }
        }
        counted += 1;
    }
    assert_eq!(counted, 60);
    assert!(
        differences.is_empty(),
        "differences:\n{}",
        differences.join("\n")
    );
}

#[test]
fn m501_candidates_match_prototype_classification() {
    let proto = prototype();
    let mut differences = Vec::new();
    let mut counted = 0;
    for row in proto["m501"].as_array().expect("rows") {
        let id = row["id"].as_str().expect("id");
        diff(
            id,
            row["source"].as_str().expect("source"),
            row,
            &mut differences,
        );
        counted += 1;
    }
    assert_eq!(counted, 140);
    assert!(
        differences.is_empty(),
        "differences:\n{}",
        differences.join("\n")
    );
}
