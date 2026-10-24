//! Quantifier-free law formulas: `forall vars. lhs_i = rhs_i` for every
//! equation, over exact integers or 32-bit words, and their SMT-LIB2 form
//! (the negation is asserted, so a valid formula is `unsat`). Terms are ring
//! terms plus `max`, `min` and `ite` over comparisons (unsigned for
//! [`Sort::Bv32`]).

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write,
    ops::{Add, Mul, Neg, Sub},
};

/// The domain the variables range over.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sort {
    /// Exact integers: checked arithmetic before its first trap (Lemma A).
    Int,
    /// `Z/2^32`: the sealed wrapping intrinsics.
    Bv32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Term {
    Var(String),
    Lit(i64),
    Add(Box<Term>, Box<Term>),
    Sub(Box<Term>, Box<Term>),
    Mul(Box<Term>, Box<Term>),
    Neg(Box<Term>),
    /// `a` if `b <= a`, else `b`.
    Max(Box<Term>, Box<Term>),
    /// `a` if `a <= b`, else `b`.
    Min(Box<Term>, Box<Term>),
    Ite(Box<Cond>, Box<Term>, Box<Term>),
}

/// A comparison between terms.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Cond {
    Le(Term, Term),
    Lt(Term, Term),
    Eq(Term, Term),
    Not(Box<Cond>),
}

impl Cond {
    fn subst(&self, map: &BTreeMap<String, Term>) -> Self {
        match self {
            Self::Le(a, b) => Self::Le(a.subst(map), b.subst(map)),
            Self::Lt(a, b) => Self::Lt(a.subst(map), b.subst(map)),
            Self::Eq(a, b) => Self::Eq(a.subst(map), b.subst(map)),
            Self::Not(c) => Self::Not(Box::new(c.subst(map))),
        }
    }

    fn vars(&self, out: &mut BTreeSet<String>) {
        match self {
            Self::Le(a, b) | Self::Lt(a, b) | Self::Eq(a, b) => {
                a.vars(out);
                b.vars(out);
            }
            Self::Not(c) => c.vars(out),
        }
    }

    fn smtlib(&self, sort: Sort, out: &mut String) {
        let (op, a, b) = match (self, sort) {
            (Self::Not(c), _) => {
                out.push_str("(not ");
                c.smtlib(sort, out);
                out.push(')');
                return;
            }
            (Self::Le(a, b), Sort::Int) => ("<=", a, b),
            (Self::Lt(a, b), Sort::Int) => ("<", a, b),
            (Self::Le(a, b), Sort::Bv32) => ("bvule", a, b),
            (Self::Lt(a, b), Sort::Bv32) => ("bvult", a, b),
            (Self::Eq(a, b), _) => ("=", a, b),
        };
        let _ = write!(out, "({op} ");
        a.smtlib(sort, out);
        out.push(' ');
        b.smtlib(sort, out);
        out.push(')');
    }
}

impl Term {
    pub fn var(name: &str) -> Self {
        Self::Var(name.to_owned())
    }

    /// Every variable the term reads.
    pub fn free_vars(&self) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        self.vars(&mut out);
        out
    }

    /// Simultaneous substitution of the variables `map` names.
    pub fn subst(&self, map: &BTreeMap<String, Term>) -> Self {
        let two = |a: &Term, b: &Term| (Box::new(a.subst(map)), Box::new(b.subst(map)));
        match self {
            Self::Var(name) => map.get(name).cloned().unwrap_or_else(|| self.clone()),
            Self::Lit(_) => self.clone(),
            Self::Add(a, b) => {
                let (a, b) = two(a, b);
                Self::Add(a, b)
            }
            Self::Sub(a, b) => {
                let (a, b) = two(a, b);
                Self::Sub(a, b)
            }
            Self::Mul(a, b) => {
                let (a, b) = two(a, b);
                Self::Mul(a, b)
            }
            Self::Max(a, b) => {
                let (a, b) = two(a, b);
                Self::Max(a, b)
            }
            Self::Min(a, b) => {
                let (a, b) = two(a, b);
                Self::Min(a, b)
            }
            Self::Neg(a) => Self::Neg(Box::new(a.subst(map))),
            Self::Ite(c, a, b) => {
                let (a, b) = two(a, b);
                Self::Ite(Box::new(c.subst(map)), a, b)
            }
        }
    }

    pub fn max(self, other: Self) -> Self {
        Self::Max(Box::new(self), Box::new(other))
    }

    pub fn min(self, other: Self) -> Self {
        Self::Min(Box::new(self), Box::new(other))
    }

    pub fn ite(cond: Cond, then: Self, other: Self) -> Self {
        Self::Ite(Box::new(cond), Box::new(then), Box::new(other))
    }

    fn vars(&self, out: &mut BTreeSet<String>) {
        match self {
            Self::Var(name) => {
                out.insert(name.clone());
            }
            Self::Lit(_) => {}
            Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::Mul(a, b)
            | Self::Max(a, b)
            | Self::Min(a, b) => {
                a.vars(out);
                b.vars(out);
            }
            Self::Neg(a) => a.vars(out),
            Self::Ite(c, a, b) => {
                c.vars(out);
                a.vars(out);
                b.vars(out);
            }
        }
    }

    fn smtlib(&self, sort: Sort, out: &mut String) {
        let (add, sub, mul, neg) = match sort {
            Sort::Int => ("+", "-", "*", "-"),
            Sort::Bv32 => ("bvadd", "bvsub", "bvmul", "bvneg"),
        };
        let pair = |op: &str, a: &Term, b: &Term, out: &mut String| {
            let _ = write!(out, "({op} ");
            a.smtlib(sort, out);
            out.push(' ');
            b.smtlib(sort, out);
            out.push(')');
        };
        match self {
            Self::Var(name) => out.push_str(&symbol(name)),
            Self::Lit(x) => match sort {
                Sort::Int if *x < 0 => {
                    let _ = write!(out, "(- {})", x.unsigned_abs());
                }
                Sort::Int => {
                    let _ = write!(out, "{x}");
                }
                Sort::Bv32 => {
                    let _ = write!(out, "(_ bv{} 32)", *x as u32);
                }
            },
            Self::Add(a, b) => pair(add, a, b, out),
            Self::Sub(a, b) => pair(sub, a, b, out),
            Self::Mul(a, b) => pair(mul, a, b, out),
            Self::Neg(a) => {
                let _ = write!(out, "({neg} ");
                a.smtlib(sort, out);
                out.push(')');
            }
            // Each operand is bound once, so nesting never duplicates text:
            // `(let ((|#l| a) (|#r| b)) (ite (<= |#r| |#l|) |#l| |#r|))`.
            // No variable prints as `|#l|` or `|#r|` (see [`symbol`]).
            Self::Max(a, b) | Self::Min(a, b) => {
                out.push_str("(let ((|#l| ");
                a.smtlib(sort, out);
                out.push_str(") (|#r| ");
                b.smtlib(sort, out);
                let le = if sort == Sort::Int { "<=" } else { "bvule" };
                let (low, high) = if matches!(self, Self::Max(..)) {
                    ("|#r|", "|#l|")
                } else {
                    ("|#l|", "|#r|")
                };
                let _ = write!(out, ")) (ite ({le} {low} {high}) |#l| |#r|))");
            }
            Self::Ite(c, a, b) => {
                out.push_str("(ite ");
                c.smtlib(sort, out);
                out.push(' ');
                a.smtlib(sort, out);
                out.push(' ');
                b.smtlib(sort, out);
                out.push(')');
            }
        }
    }
}

/// Simple names that SMT-LIB2 already gives a meaning to.
const RESERVED: [&str; 14] = [
    "and", "as", "distinct", "exists", "false", "forall", "ite", "let", "match", "not", "or",
    "par", "true", "xor",
];

/// An injective SMT-LIB2 symbol for a variable name: a simple, unreserved
/// name as it is; anything else quoted as `|#...|` with `#`, `|` and `\` coded
/// as `##`, `#p` and `#b`. A quoted symbol holds `#`, so it never equals a
/// simple name or a builtin.
fn symbol(name: &str) -> String {
    let simple = name.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !name.starts_with("bv")
        && !RESERVED.contains(&name);
    if simple {
        return name.to_owned();
    }
    let mut out = "|#".to_owned();
    for c in name.chars() {
        match c {
            '#' => out.push_str("##"),
            '|' => out.push_str("#p"),
            '\\' => out.push_str("#b"),
            c => out.push(c),
        }
    }
    out.push('|');
    out
}

macro_rules! binary {
    ($trait:ident, $method:ident, $variant:ident) => {
        impl $trait for Term {
            type Output = Term;
            fn $method(self, other: Term) -> Term {
                Term::$variant(Box::new(self), Box::new(other))
            }
        }
    };
}
binary!(Add, add, Add);
binary!(Sub, sub, Sub);
binary!(Mul, mul, Mul);

impl Neg for Term {
    type Output = Term;
    fn neg(self) -> Term {
        Term::Neg(Box::new(self))
    }
}

/// `forall vars. lhs_i = rhs_i` for every equation, over `sort`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Formula {
    pub sort: Sort,
    /// Every variable the equations read, sorted.
    pub vars: Vec<String>,
    pub equations: Vec<(Term, Term)>,
}

impl Formula {
    pub fn new(sort: Sort, equations: Vec<(Term, Term)>) -> Self {
        let mut vars = BTreeSet::new();
        for (lhs, rhs) in &equations {
            lhs.vars(&mut vars);
            rhs.vars(&mut vars);
        }
        Self {
            sort,
            vars: vars.into_iter().collect(),
            equations,
        }
    }

    /// The SMT-LIB2 script whose `check-sat` answers `unsat` iff the
    /// formula is valid.
    pub fn smtlib(&self) -> String {
        let (logic, ty) = match self.sort {
            Sort::Int => ("(set-logic QF_NIA)\n", "Int"),
            Sort::Bv32 => ("(set-logic QF_BV)\n", "(_ BitVec 32)"),
        };
        let mut out = logic.to_owned();
        for var in &self.vars {
            let _ = writeln!(out, "(declare-const {} {ty})", symbol(var));
        }
        let mut equations = Vec::with_capacity(self.equations.len());
        for (lhs, rhs) in &self.equations {
            let mut equation = "(= ".to_owned();
            lhs.smtlib(self.sort, &mut equation);
            equation.push(' ');
            rhs.smtlib(self.sort, &mut equation);
            equation.push(')');
            equations.push(equation);
        }
        let conjunction = match &equations[..] {
            [] => "true".to_owned(),
            [one] => one.clone(),
            all => format!("(and {})", all.join(" ")),
        };
        let _ = write!(out, "(assert (not {conjunction}))\n(check-sat)\n");
        out
    }
}
