//! Shared helpers for the P1 normal-form tests: compile a source to a verified
//! net and describe a normal form in the net-laws prototype's vocabulary.
#![allow(dead_code)]

use invar::{
    check, front,
    general_net::{
        self, Net, Proof,
        normal_form::{ArrayLaw, Node, NormalForm, Order, Rep, Ring},
    },
    laws::Law,
};

pub fn compile(source: &str) -> (Net, Proof) {
    let checked =
        check::check(front::parse(source.as_bytes()).expect("parse"), false).expect("check");
    invar::ownership::check(&checked).expect("ownership");
    let net = Net::lower(&checked, general_net::Limits::default()).expect("lower");
    let proof = net.verify().expect("verify");
    (net, proof)
}

/// Prototype law name of a carried representation.
pub fn rep_name(rep: &Rep) -> Vec<&'static str> {
    match rep {
        Rep::Sealed(Law::WrappingSum) => vec!["wadd"],
        Rep::Sealed(_) => vec!["wmul"],
        Rep::LiftedSum(_) | Rep::Counter(_) => vec!["add"],
        Rep::SaturatedProduct(_) => vec!["mul"],
        Rep::Affine(Ring::Wrapping32) => vec!["waff"],
        Rep::Affine(Ring::Checked64(_)) => vec!["affz"],
        Rep::Affine(Ring::Residue { .. }) => vec!["affmod"],
        Rep::Matrix {
            ring: Ring::Residue { .. },
            ..
        } => vec!["matmod"],
        Rep::Matrix { .. } => vec!["matz"],
        Rep::Tropical {
            order: Order::Max, ..
        } => vec!["maxplus"],
        Rep::Tropical { .. } => vec!["minplus"],
        Rep::Select {
            order, companions, ..
        } => {
            let mut out = vec![if *order == Order::Min { "min" } else { "max" }];
            if !companions.is_empty() {
                out.push("arg");
            }
            out
        }
        Rep::Last => vec!["last"],
        Rep::Lifted(_) => vec!["fsm"],
    }
}

/// `(c, laws)` per loop, functions in program order, loops in pre-order, in
/// the prototype's vocabulary (`c` is `Some("par")` or `None`).
pub fn loop_labels(form: &NormalForm) -> Vec<(Option<&'static str>, Vec<String>, String)> {
    let mut out = Vec::new();
    for function in &form.functions {
        for loop_form in &function.loops {
            match &loop_form.node {
                Node::Serial { reason } => out.push((None, Vec::new(), reason.clone())),
                Node::Parallel(node) => {
                    let mut laws: Vec<String> = node
                        .carried
                        .iter()
                        .flat_map(|carried| rep_name(&carried.rep))
                        .map(str::to_owned)
                        .collect();
                    for (_, law) in &node.arrays {
                        let name = match law {
                            ArrayLaw::Own => continue,
                            ArrayLaw::Combine { .. } => "histogram",
                            ArrayLaw::Ordered => "scatter",
                            ArrayLaw::Compaction { .. } => "compaction",
                            ArrayLaw::Monotone { .. } => "monotone",
                        };
                        laws.push(name.to_owned());
                    }
                    laws.sort();
                    laws.dedup();
                    if node.first_exit {
                        laws.push("first-exit".to_owned());
                    }
                    out.push((Some("par"), laws, String::new()));
                }
            }
        }
    }
    out
}
