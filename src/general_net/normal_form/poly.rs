//! Polynomial normal form for the certificate checker. It is independent of the
//! extractor's syntactic linear forms: an expression is normalised to a sum of
//! monomials over opaque atoms (locals, indexed reads, calls, lengths), with
//! integer coefficients reduced in the certificate's ring. Two expressions are
//! equal in the ring iff their normal forms are equal (atoms compared without
//! spans).

use std::collections::BTreeMap;

use crate::{
    front::{BinaryOp, Span, UnaryOp},
    general_net::regions::tree::Expr,
};

/// Coefficient ring of the normal form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Arith {
    /// Exact integers (checked arithmetic regrouped by Lemma A).
    Exact,
    /// `Z/2^32`: the sealed wrapping intrinsics.
    Wrap32,
}

/// Monomial: sorted multiset of atom keys.
type Monomial = Vec<String>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Poly {
    arith: Arith,
    terms: BTreeMap<Monomial, i128>,
}

/// Bound on coefficient magnitude in exact mode; larger values are refused.
const EXACT_LIMIT: i128 = 1 << 100;

impl Poly {
    fn new(arith: Arith) -> Self {
        Self {
            arith,
            terms: BTreeMap::new(),
        }
    }

    pub(super) fn constant(arith: Arith, value: i128) -> Option<Self> {
        let mut poly = Self::new(arith);
        poly.add_term(Vec::new(), value)?;
        Some(poly)
    }

    pub(super) fn atom(arith: Arith, key: String) -> Self {
        let mut poly = Self::new(arith);
        poly.terms.insert(vec![key], 1);
        poly
    }

    fn reduce(&self, value: i128) -> Option<i128> {
        match self.arith {
            Arith::Exact => (value.abs() < EXACT_LIMIT).then_some(value),
            Arith::Wrap32 => Some(value.rem_euclid(1 << 32)),
        }
    }

    fn add_term(&mut self, monomial: Monomial, value: i128) -> Option<()> {
        let sum = self
            .terms
            .get(&monomial)
            .copied()
            .unwrap_or(0)
            .checked_add(value)?;
        let sum = self.reduce(sum)?;
        if sum == 0 {
            self.terms.remove(&monomial);
        } else {
            self.terms.insert(monomial, sum);
        }
        Some(())
    }

    pub(super) fn add(&self, other: &Self) -> Option<Self> {
        let mut out = self.clone();
        for (monomial, value) in &other.terms {
            out.add_term(monomial.clone(), *value)?;
        }
        Some(out)
    }

    pub(super) fn neg(&self) -> Option<Self> {
        let mut out = Self::new(self.arith);
        for (monomial, value) in &self.terms {
            out.add_term(monomial.clone(), value.checked_neg()?)?;
        }
        Some(out)
    }

    pub(super) fn sub(&self, other: &Self) -> Option<Self> {
        self.add(&other.neg()?)
    }

    pub(super) fn mul(&self, other: &Self) -> Option<Self> {
        let mut out = Self::new(self.arith);
        for (a, x) in &self.terms {
            for (b, y) in &other.terms {
                let mut monomial: Monomial = a.iter().chain(b).cloned().collect();
                monomial.sort();
                out.add_term(monomial, x.checked_mul(*y)?)?;
            }
        }
        Some(out)
    }

    /// Monomials (sorted atom keys) with their non-zero coefficients.
    pub(super) fn terms(&self) -> impl Iterator<Item = (&[String], i128)> {
        self.terms
            .iter()
            .map(|(monomial, value)| (monomial.as_slice(), *value))
    }

    /// The atom key when the polynomial is exactly one atom.
    pub(super) fn single_atom(&self) -> Option<&str> {
        let mut terms = self.terms.iter();
        match (terms.next(), terms.next()) {
            (Some((monomial, &1)), None) if monomial.len() == 1 => {
                monomial.first().map(String::as_str)
            }
            _ => None,
        }
    }

    pub(super) fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// The constant value, when the polynomial has no atoms.
    pub(super) fn as_constant(&self) -> Option<i128> {
        match self.terms.len() {
            0 => Some(0),
            1 => self.terms.get(&Vec::new()).copied(),
            _ => None,
        }
    }

    /// Whether any monomial mentions an atom accepted by `pred`.
    pub(super) fn mentions(&self, pred: &dyn Fn(&str) -> bool) -> bool {
        self.terms.keys().flatten().any(|key| pred(key))
    }

    /// Split `self = c * x + rest` for the atom `x` (degree at most one in `x`).
    pub(super) fn split_linear(&self, x: &str) -> Option<(Self, Self)> {
        let mut coefficient = Self::new(self.arith);
        let mut rest = Self::new(self.arith);
        for (monomial, value) in &self.terms {
            let count = monomial.iter().filter(|key| *key == x).count();
            match count {
                0 => rest.add_term(monomial.clone(), *value)?,
                1 => {
                    let others: Monomial =
                        monomial.iter().filter(|key| *key != x).cloned().collect();
                    coefficient.add_term(others, *value)?;
                }
                _ => return None,
            }
        }
        Some((coefficient, rest))
    }
}

/// Atom key of a local name.
pub(super) fn local_key(name: &str) -> String {
    format!("{:?}", Expr::Local(name.to_owned()))
}

const NO_SPAN: Span = Span { start: 0, end: 0 };

/// The expression without source spans, for atom identity.
pub(super) fn strip(expr: &Expr) -> Expr {
    let boxed = |e: &Expr| Box::new(strip(e));
    match expr {
        Expr::Index { array, index, .. } => Expr::Index {
            array: array.clone(),
            index: boxed(index),
            span: NO_SPAN,
        },
        Expr::Unary { op, value, .. } => Expr::Unary {
            op: *op,
            value: boxed(value),
            span: NO_SPAN,
        },
        Expr::Binary {
            op, left, right, ..
        } => Expr::Binary {
            op: *op,
            left: boxed(left),
            right: boxed(right),
            span: NO_SPAN,
        },
        Expr::Logic { and, left, right } => Expr::Logic {
            and: *and,
            left: boxed(left),
            right: boxed(right),
        },
        Expr::Call { name, args, .. } => Expr::Call {
            name: name.clone(),
            args: args.iter().map(strip).collect(),
            span: NO_SPAN,
        },
        Expr::ArrayLit { elem, items } => Expr::ArrayLit {
            elem: elem.clone(),
            items: items.iter().map(strip).collect(),
        },
        other => other.clone(),
    }
}

fn atom_of(arith: Arith, expr: &Expr) -> Poly {
    Poly::atom(arith, format!("{:?}", strip(expr)))
}

/// Normalise `expr`. Checked `+ - *` and negation are ring operations in every
/// mode; the wrapping intrinsics are ring operations only in `Wrap32`, and in
/// other modes they are opaque atoms. Everything else is an atom.
pub(super) fn normalize(arith: Arith, expr: &Expr) -> Option<Poly> {
    match expr {
        Expr::I32(x) => Poly::constant(arith, i128::from(*x)),
        Expr::U32(x) => Poly::constant(arith, i128::from(*x)),
        Expr::Binary {
            op: op @ (BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul),
            left,
            right,
            ..
        } if arith != Arith::Wrap32 => {
            let (a, b) = (normalize(arith, left)?, normalize(arith, right)?);
            match op {
                BinaryOp::Add => a.add(&b),
                BinaryOp::Sub => a.sub(&b),
                _ => a.mul(&b),
            }
        }
        Expr::Unary {
            op: UnaryOp::Neg,
            value,
            ..
        } if arith != Arith::Wrap32 => normalize(arith, value)?.neg(),
        Expr::Call { name, args, .. } if arith == Arith::Wrap32 && args.len() == 2 => {
            match name.as_str() {
                "wrapping_add" => normalize(arith, &args[0])?.add(&normalize(arith, &args[1])?),
                "wrapping_mul" => normalize(arith, &args[0])?.mul(&normalize(arith, &args[1])?),
                _ => Some(atom_of(arith, expr)),
            }
        }
        _ => Some(atom_of(arith, expr)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn local(name: &str) -> Expr {
        Expr::Local(name.to_owned())
    }

    fn bin(op: BinaryOp, left: Expr, right: Expr) -> Expr {
        Expr::Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
            span: Span { start: 3, end: 4 },
        }
    }

    #[test]
    fn regrouped_sums_and_products_normalise_equal() {
        let left = bin(
            BinaryOp::Mul,
            bin(BinaryOp::Add, local("s"), local("x")),
            Expr::I32(3),
        );
        let right = bin(
            BinaryOp::Add,
            bin(BinaryOp::Mul, Expr::I32(3), local("x")),
            bin(BinaryOp::Mul, local("s"), Expr::I32(3)),
        );
        assert_eq!(
            normalize(Arith::Exact, &left),
            normalize(Arith::Exact, &right)
        );
        let wrong = bin(BinaryOp::Add, local("s"), local("x"));
        assert_ne!(
            normalize(Arith::Exact, &left),
            normalize(Arith::Exact, &wrong)
        );
    }

    #[test]
    fn split_linear_rejects_squares() {
        let square = bin(BinaryOp::Mul, local("s"), local("s"));
        let poly = normalize(Arith::Exact, &square).unwrap();
        assert!(poly.split_linear(&local_key("s")).is_none());
        let affine = bin(
            BinaryOp::Add,
            bin(BinaryOp::Mul, local("s"), local("a")),
            Expr::I32(7),
        );
        let (c, rest) = normalize(Arith::Exact, &affine)
            .unwrap()
            .split_linear(&local_key("s"))
            .unwrap();
        assert_eq!(c, normalize(Arith::Exact, &local("a")).unwrap());
        assert_eq!(rest.as_constant(), Some(7));
    }
}
