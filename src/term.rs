//! Term syntax of the finite internal fragment that `typed_net` encodes and decodes.
//!
//! The call-by-value evaluator that serves as its independent oracle lives in
//! `test-support/term_oracle.rs`, not in the library.

use std::collections::BTreeMap;

pub type Binder = u32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Term {
    pub id: u32,
    pub kind: Kind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    Unit,
    Bool(bool),
    Nat(u32),
    Pair(Box<Term>, Box<Term>),
    Var(Binder),
    Lambda {
        binder: Binder,
        body: Box<Term>,
    },
    Apply(Box<Term>, Box<Term>),
    Let {
        binder: Binder,
        value: Box<Term>,
        body: Box<Term>,
    },
    If {
        condition: Box<Term>,
        yes: Box<Term>,
        no: Box<Term>,
    },
    Call {
        name: String,
        args: Vec<Term>,
    },
    /// Internal provenance fixture; no public Invar source syntax.
    Fail(&'static str),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Function {
    pub params: Vec<Binder>,
    pub body: Term,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Program {
    pub functions: BTreeMap<String, Function>,
}
