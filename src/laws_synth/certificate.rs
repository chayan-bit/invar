//! Lifting certificates (#59): a file committed beside the source that
//! lifts one loop's carried scalars with auxiliary state so that chunks of
//! the iteration space combine with a join. A synthesizer only proposes; the
//! compile path trusts nothing the file states. [`LawCertificate::check`]
//! re-derives the loop's step from the current net ([`lift::loop_steps`]),
//! rebuilds the proof obligation, refuses any stored formula that differs,
//! and decides the obligation again with [`prove::decide`].
//!
//! With `F` the lifted step (the net's step for carried scalars, the
//! certificate's for auxiliaries), `e(y)` the lifted single iteration and
//! `join` the combine, the obligation is, for all states `a`, `b`, `c` and
//! inputs `y`:
//!
//! 1. `join(a, e(y)) = F(a, y)`;
//! 2. `join(a, F(b, y)) = F(join(a, b), y)`;
//! 3. `join(join(a, b), c) = join(a, join(b, c))`.
//!
//! By induction on a chunk `ys` (`h([y]) = e(y)`, `h(ys ++ [y]) = F(h(ys),
//! y)`), 1 and 2 give `join(a, h(ys)) = F^|ys|(a)` for every seed `a`, so a
//! split run equals the serial one; 3 lets the chunks combine as a tree.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Value, json};

use super::{
    SOLVER, SOLVER_VERSION,
    formula::{Cond, Formula, Term},
    prove::{self, Verdict},
};
use crate::general_net::{
    normal_form::lift::{self, LoopSteps},
    regions::Partition,
};

pub const FORMAT: &str = "invar-law-certificate-v1";

/// The trust root a certificate records (see [`prove`]).
pub const TRUST: &str = "invar-prove v1: exact polynomial normal forms over Z and Z/2^32, \
     max/min/ite case splits over Z refuted by Fourier-Motzkin over Q; no external solver";

/// One component of the lifted state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Component {
    pub name: String,
    /// The step of an auxiliary component; `None` for a carried scalar,
    /// whose step is re-derived from the net.
    pub step: Option<Term>,
    /// The component after a single iteration, over the inputs.
    pub element: Term,
    /// The combine of a left and a right chunk, over `l.<name>` and `r.<name>`.
    pub join: Term,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LawCertificate {
    pub function: String,
    /// Loop index in source pre-order.
    pub loop_index: usize,
    pub state: Vec<Component>,
    /// The obligation as SMT-LIB2 text.
    pub formula: String,
    pub solver: String,
    pub solver_version: u32,
    pub trust: String,
    /// `unsat`: the negated obligation has no model.
    pub result: String,
}

/// Why a certificate is refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    Malformed(String),
    Loop(String),
    State(String),
    FormulaDiffers,
    Solver,
    Result,
    Unproved(String),
}

impl LawCertificate {
    /// Certify `state` for a loop: derive the obligation and decide it.
    pub fn seal(
        partition: &Partition,
        function: &str,
        loop_index: usize,
        state: Vec<Component>,
    ) -> Result<Self, Refusal> {
        let steps = lift::loop_steps(partition, function, loop_index).map_err(Refusal::Loop)?;
        let formula = obligation(&state, &steps)?;
        if let Verdict::Unproved(why) = prove::decide(&formula) {
            return Err(Refusal::Unproved(why));
        }
        Ok(Self {
            function: function.to_owned(),
            loop_index,
            state,
            formula: formula.smtlib(),
            solver: SOLVER.to_owned(),
            solver_version: SOLVER_VERSION,
            trust: TRUST.to_owned(),
            result: "unsat".to_owned(),
        })
    }

    /// Accept the certificate only if the obligation re-derived from the
    /// current net is the stored one and is proved again.
    pub fn check(&self, partition: &Partition) -> Result<(), Refusal> {
        let steps =
            lift::loop_steps(partition, &self.function, self.loop_index).map_err(Refusal::Loop)?;
        let formula = obligation(&self.state, &steps)?;
        if formula.smtlib() != self.formula {
            return Err(Refusal::FormulaDiffers);
        }
        if self.solver != SOLVER || self.solver_version != SOLVER_VERSION || self.trust != TRUST {
            return Err(Refusal::Solver);
        }
        if self.result != "unsat" {
            return Err(Refusal::Result);
        }
        match prove::decide(&formula) {
            Verdict::Proved => Ok(()),
            Verdict::Unproved(why) => Err(Refusal::Unproved(why)),
        }
    }

    pub fn to_json(&self) -> String {
        let state: Vec<Value> = self
            .state
            .iter()
            .map(|c| {
                json!({
                    "name": c.name,
                    "step": c.step.as_ref().map(term_json),
                    "element": term_json(&c.element),
                    "join": term_json(&c.join),
                })
            })
            .collect();
        let value = json!({
            "format": FORMAT,
            "function": self.function,
            "loop": self.loop_index,
            "state": state,
            "formula": self.formula,
            "solver": self.solver,
            "solver_version": self.solver_version,
            "trust": self.trust,
            "result": self.result,
        });
        serde_json::to_string_pretty(&value).unwrap_or_default()
    }

    pub fn from_json(text: &str) -> Result<Self, Refusal> {
        let bad = |what: &str| Refusal::Malformed(what.to_owned());
        let value: Value = serde_json::from_str(text).map_err(|e| bad(&e.to_string()))?;
        let text_of = |key: &str| {
            value[key]
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| bad(key))
        };
        if text_of("format")? != FORMAT {
            return Err(bad("format"));
        }
        let number = |key: &str| value[key].as_u64().ok_or_else(|| bad(key));
        let mut state = Vec::new();
        for c in value["state"].as_array().ok_or_else(|| bad("state"))? {
            state.push(Component {
                name: c["name"].as_str().ok_or_else(|| bad("name"))?.to_owned(),
                step: match &c["step"] {
                    Value::Null => None,
                    step => Some(term_from(step)?),
                },
                element: term_from(&c["element"])?,
                join: term_from(&c["join"])?,
            });
        }
        Ok(Self {
            function: text_of("function")?,
            loop_index: usize::try_from(number("loop")?).map_err(|_| bad("loop"))?,
            state,
            formula: text_of("formula")?,
            solver: text_of("solver")?,
            solver_version: u32::try_from(number("solver_version")?)
                .map_err(|_| bad("solver_version"))?,
            trust: text_of("trust")?,
            result: text_of("result")?,
        })
    }
}

/// The obligation of `state` lifting the loop `steps`.
fn obligation(state: &[Component], steps: &LoopSteps) -> Result<Formula, Refusal> {
    let refuse = |why: String| Err(Refusal::State(why));
    let names: Vec<&str> = state.iter().map(|c| c.name.as_str()).collect();
    let unique: BTreeSet<&str> = names.iter().copied().collect();
    let carried: BTreeSet<&str> = state
        .iter()
        .filter(|c| c.step.is_none())
        .map(|c| c.name.as_str())
        .collect();
    if unique.len() != names.len() || names.iter().any(|n| n.is_empty() || n.contains('.')) {
        return refuse("state names must be distinct and undotted".to_owned());
    }
    if carried != steps.steps.keys().map(String::as_str).collect() {
        return refuse(format!(
            "carried components {carried:?} are not the loop's scalars"
        ));
    }
    let inputs: BTreeSet<String> = steps.inputs.iter().cloned().collect();
    let mut invariant: BTreeSet<String> = steps.steps.values().flat_map(Term::free_vars).collect();
    invariant.retain(|v| !unique.contains(v.as_str()) && !inputs.contains(v));
    let free = |extra: &mut dyn Iterator<Item = String>| -> BTreeSet<String> {
        inputs
            .iter()
            .chain(&invariant)
            .cloned()
            .chain(extra)
            .collect()
    };
    let in_state = free(&mut names.iter().map(|n| (*n).to_owned()));
    let in_join = free(
        &mut names
            .iter()
            .flat_map(|n| [format!("l.{n}"), format!("r.{n}")]),
    );
    let in_element = free(&mut std::iter::empty());
    for c in state {
        if inputs.contains(&c.name) || invariant.contains(&c.name) {
            return refuse(format!("{} names an input or invariant", c.name));
        }
        let terms = [(&c.element, &in_element), (&c.join, &in_join)];
        let step = c.step.as_ref().map(|s| (s, &in_state));
        for (term, allowed) in terms.into_iter().chain(step) {
            if let Some(stray) = term.free_vars().difference(allowed).next() {
                return refuse(format!("{} reads {stray}", c.name));
            }
        }
    }
    let at = |prefix: &str| -> BTreeMap<String, Term> {
        names
            .iter()
            .map(|n| ((*n).to_owned(), Term::var(&format!("{prefix}.{n}"))))
            .collect()
    };
    let step_of = |c: &Component| {
        c.step
            .clone()
            .unwrap_or_else(|| steps.steps[&c.name].clone())
    };
    let lifted = |x: &BTreeMap<String, Term>| -> BTreeMap<String, Term> {
        state
            .iter()
            .map(|c| (c.name.clone(), step_of(c).subst(x)))
            .collect()
    };
    let join = |x: &BTreeMap<String, Term>, y: &BTreeMap<String, Term>| {
        let sides: BTreeMap<String, Term> = names
            .iter()
            .flat_map(|n| {
                [
                    (format!("l.{n}"), x[*n].clone()),
                    (format!("r.{n}"), y[*n].clone()),
                ]
            })
            .collect();
        state
            .iter()
            .map(|c| (c.name.clone(), c.join.subst(&sides)))
            .collect::<BTreeMap<String, Term>>()
    };
    let (a, b, c) = (at("a"), at("b"), at("c"));
    let element: BTreeMap<String, Term> = state
        .iter()
        .map(|c| (c.name.clone(), c.element.clone()))
        .collect();
    let sides = [
        (join(&a, &element), lifted(&a)),
        (join(&a, &lifted(&b)), lifted(&join(&a, &b))),
        (join(&join(&a, &b), &c), join(&a, &join(&b, &c))),
    ];
    let equations = sides
        .into_iter()
        .flat_map(|(l, r)| names.iter().map(move |n| (l[*n].clone(), r[*n].clone())))
        .collect();
    Ok(Formula::new(steps.sort, equations))
}

fn term_json(term: &Term) -> Value {
    match term {
        Term::Var(name) => json!(["var", name]),
        Term::Lit(x) => json!(["lit", x]),
        Term::Add(a, b) => json!(["+", term_json(a), term_json(b)]),
        Term::Sub(a, b) => json!(["-", term_json(a), term_json(b)]),
        Term::Mul(a, b) => json!(["*", term_json(a), term_json(b)]),
        Term::Neg(a) => json!(["neg", term_json(a)]),
        Term::Max(a, b) => json!(["max", term_json(a), term_json(b)]),
        Term::Min(a, b) => json!(["min", term_json(a), term_json(b)]),
        Term::Ite(c, a, b) => json!(["ite", cond_json(c), term_json(a), term_json(b)]),
    }
}

fn cond_json(cond: &Cond) -> Value {
    match cond {
        Cond::Le(a, b) => json!(["<=", term_json(a), term_json(b)]),
        Cond::Lt(a, b) => json!(["<", term_json(a), term_json(b)]),
        Cond::Eq(a, b) => json!(["=", term_json(a), term_json(b)]),
        Cond::Not(c) => json!(["not", cond_json(c)]),
    }
}

fn parts(value: &Value) -> Result<(&str, &[Value]), Refusal> {
    let bad = || Refusal::Malformed(format!("term {value}"));
    let list = value.as_array().ok_or_else(bad)?;
    let (head, rest) = list.split_first().ok_or_else(bad)?;
    Ok((head.as_str().ok_or_else(bad)?, rest))
}

fn term_from(value: &Value) -> Result<Term, Refusal> {
    let bad = || Refusal::Malformed(format!("term {value}"));
    let (head, rest) = parts(value)?;
    let sub = |i: usize| rest.get(i).ok_or_else(bad).and_then(term_from);
    let two = |f: fn(Box<Term>, Box<Term>) -> Term| Ok(f(Box::new(sub(0)?), Box::new(sub(1)?)));
    let arity = match head {
        "var" | "lit" | "neg" => 1,
        "ite" => 3,
        _ => 2,
    };
    if rest.len() != arity {
        return Err(bad());
    }
    match head {
        "var" => Ok(Term::var(rest[0].as_str().ok_or_else(bad)?)),
        "lit" => Ok(Term::Lit(rest[0].as_i64().ok_or_else(bad)?)),
        "neg" => Ok(-sub(0)?),
        "+" => two(Term::Add),
        "-" => two(Term::Sub),
        "*" => two(Term::Mul),
        "max" => two(Term::Max),
        "min" => two(Term::Min),
        "ite" => Ok(Term::ite(cond_from(&rest[0])?, sub(1)?, sub(2)?)),
        _ => Err(bad()),
    }
}

fn cond_from(value: &Value) -> Result<Cond, Refusal> {
    let bad = || Refusal::Malformed(format!("condition {value}"));
    let (head, rest) = parts(value)?;
    let sub = |i: usize| rest.get(i).ok_or_else(bad).and_then(term_from);
    match (head, rest.len()) {
        ("not", 1) => Ok(Cond::Not(Box::new(cond_from(&rest[0])?))),
        ("<=", 2) => Ok(Cond::Le(sub(0)?, sub(1)?)),
        ("<", 2) => Ok(Cond::Lt(sub(0)?, sub(1)?)),
        ("=", 2) => Ok(Cond::Eq(sub(0)?, sub(1)?)),
        _ => Err(bad()),
    }
}
