//! The in-repo decision procedure for law formulas (`invar-prove` v1).
//!
//! Trust root: this file and [`super::linear`]. No external solver is
//! trusted; a `z3` cross-check is optional and never gates. The fragment is
//! equations between terms over [`Sort::Int`] (`Z`) or [`Sort::Bv32`]
//! (`Z/2^32`): ring terms (variables, integer literals, `+`, `-`, unary `-`,
//! `*`) and, over `Z` only, `max`, `min` and `ite` over comparisons. Each side
//! is normalised to cases `(when, value)`: a conjunction of rows `p <= 0` and
//! a sum of monomials with exact `BigInt` coefficients. The rules, each sound
//! because it is a ring axiom, a ring homomorphism or a case analysis:
//!
//! * `Commute`: a monomial is the sorted multiset of its variables
//!   (commutativity and associativity of `*`);
//! * `Distribute`: a product multiplies every pair of monomials, multiplying
//!   coefficients (distributivity);
//! * `Negate`: `a - b` is `a + (-1) b` and `-a` is `(-1) a` (additive inverse);
//! * `Cancel`: like monomials are summed and zero coefficients dropped
//!   (`c x + d x = (c + d) x`, `0 x = 0`);
//! * `Wrap`: for [`Sort::Bv32`] only, coefficients are reduced into
//!   `[0, 2^32)`. The coefficient map `Z[x] -> (Z/2^n)[x]` is a ring
//!   homomorphism, so a polynomial identity over `Z` reduced mod `2^n` is an
//!   identity of functions on `Z/2^n`;
//! * `Split`: `max(a, b)` is `a` when `b <= a` and `b` when `a < b` (`min`
//!   and `ite` alike); the two cases cover every assignment;
//! * `Tighten`: over `Z`, `a < b` is the row `a - b + 1 <= 0` (no integer lies
//!   strictly between `a - b` and `a - b + 1`);
//! * `Relax` and `Eliminate`: a case whose rows have no rational solution,
//!   with every non-constant monomial an independent unknown, has no integer
//!   solution either (see [`super::linear`]);
//! * `Conjoin`: a formula is proved only if every equation is.
//!
//! An equation is proved when, for every pair of cases of its two sides, the
//! values have equal normal forms or the rows of both cases contradict
//! `lhs < rhs` and `lhs > rhs`. The procedure is sound and incomplete: any
//! other equation, or one that exceeds [`MAX_MONOMIALS`] or [`MAX_CASES`], is
//! [`Verdict::Unproved`]. `Unproved` is never a refutation (over `Z/2^32`,
//! `2^31 (x^2 - x)` is zero everywhere). [`Mutation`] breaks one rule each,
//! and the tests must notice every one.

use std::collections::BTreeMap;

use num_bigint::BigInt;

use super::{
    formula::{Cond, Formula, Sort, Term},
    linear,
};

/// Largest number of monomials a normal form may hold.
pub const MAX_MONOMIALS: usize = 4096;
/// Largest number of cases a term, or the case pairs an equation, may hold.
pub const MAX_CASES: usize = 4096;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    Proved,
    /// Not shown valid, with the first failing equation and why.
    Unproved(String),
}

/// A proof rule broken on purpose, by name; each must be caught by the tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mutation {
    None,
    /// `Commute`: keep a monomial's variables in product order.
    Unsorted,
    /// `Distribute`: add coefficients instead of multiplying them.
    AddCoefficients,
    /// `Negate`: read `a - b` as `a + b`.
    SubAsAdd,
    /// `Cancel`: keep monomials whose coefficient is zero.
    KeepZeros,
    /// `Wrap`: skip the reduction mod `2^32`.
    NoWrap,
    /// `Wrap`: reduce mod `2^32` over `Z` too.
    WrapInt,
    /// `Split`: keep only the first case of every comparison.
    OneCase,
    /// `Tighten`: read `a < b` as `a <= b`.
    NoTighten,
    /// `Relax`: drop the rows that mention a non-linear monomial.
    DropNonlinear,
    /// `Eliminate`: drop the combinations that cancel an unknown.
    DropCombinations,
    /// `Conjoin`: decide the first equation only.
    FirstOnly,
}

impl Mutation {
    pub const ALL: [Self; 11] = [
        Self::Unsorted,
        Self::AddCoefficients,
        Self::SubAsAdd,
        Self::KeepZeros,
        Self::NoWrap,
        Self::WrapInt,
        Self::OneCase,
        Self::NoTighten,
        Self::DropNonlinear,
        Self::DropCombinations,
        Self::FirstOnly,
    ];
}

/// A monomial is the sorted multiset of its variables; the constant term is
/// the empty monomial.
pub(super) type Poly = BTreeMap<Vec<String>, BigInt>;

/// `value` under the rows `when` (each `p <= 0`).
#[derive(Clone)]
struct Case {
    when: Vec<Poly>,
    value: Poly,
}

struct Prover {
    sort: Sort,
    mutation: Mutation,
}

impl Prover {
    fn reduce(&self, mut poly: Poly) -> Poly {
        let wrap = match self.mutation {
            Mutation::NoWrap => false,
            Mutation::WrapInt => true,
            _ => self.sort == Sort::Bv32,
        };
        if wrap {
            let modulus = BigInt::from(1u64 << 32);
            for coefficient in poly.values_mut() {
                *coefficient = ((&*coefficient % &modulus) + &modulus) % &modulus;
            }
        }
        if self.mutation != Mutation::KeepZeros {
            poly.retain(|_, c| *c != BigInt::from(0));
        }
        poly
    }

    fn add(&self, a: &Poly, b: &Poly, sign: i32) -> Option<Poly> {
        let mut out = a.clone();
        for (monomial, c) in b {
            *out.entry(monomial.clone()).or_default() += c * sign;
        }
        let out = self.reduce(out);
        (out.len() <= MAX_MONOMIALS).then_some(out)
    }

    fn mul(&self, a: &Poly, b: &Poly) -> Option<Poly> {
        if a.len().saturating_mul(b.len()) > MAX_MONOMIALS * 4 {
            return None;
        }
        let mut out = Poly::new();
        for (ma, ca) in a {
            for (mb, cb) in b {
                let mut monomial: Vec<String> = ma.iter().chain(mb).cloned().collect();
                if self.mutation != Mutation::Unsorted {
                    monomial.sort();
                }
                *out.entry(monomial).or_default() += if self.mutation == Mutation::AddCoefficients {
                    ca + cb
                } else {
                    ca * cb
                };
            }
        }
        let out = self.reduce(out);
        (out.len() <= MAX_MONOMIALS).then_some(out)
    }

    /// The row `p + 1 <= 0`, that is `p < 0` over `Z`.
    fn strict(&self, p: Poly) -> Poly {
        if self.mutation == Mutation::NoTighten {
            return p;
        }
        let mut p = p;
        *p.entry(Vec::new()).or_default() += 1;
        self.reduce(p)
    }

    /// `-p`, which never grows.
    fn negate(&self, p: &Poly) -> Poly {
        self.reduce(p.iter().map(|(m, c)| (m.clone(), -c)).collect())
    }

    /// Whether the rows have no solution.
    fn infeasible(&self, rows: &[Poly]) -> bool {
        let rows = rows.iter().filter(|row| {
            self.mutation != Mutation::DropNonlinear || row.keys().all(|m| m.len() <= 1)
        });
        linear::infeasible(rows.cloned(), self.mutation == Mutation::DropCombinations)
    }

    /// Every case pair of `a` and `b`, combined by `op`.
    fn cross(
        &self,
        a: &Term,
        b: &Term,
        op: impl Fn(&Poly, &Poly) -> Option<Poly>,
    ) -> Result<Vec<Case>, String> {
        let (a, b) = (self.normalize(a)?, self.normalize(b)?);
        if a.len().saturating_mul(b.len()) > MAX_CASES {
            return Err("too many cases".to_owned());
        }
        let mut out = Vec::new();
        for x in &a {
            for y in &b {
                out.push(Case {
                    when: x.when.iter().chain(&y.when).cloned().collect(),
                    value: op(&x.value, &y.value).ok_or("polynomial too large")?,
                });
            }
        }
        Ok(out)
    }

    /// The cases of `cond`, each with its truth value.
    fn cond(&self, cond: &Cond) -> Result<Vec<(Vec<Poly>, bool)>, String> {
        if self.sort == Sort::Bv32 {
            return Err("comparison over Z/2^32 is outside the fragment".to_owned());
        }
        if let Cond::Not(c) = cond {
            let cases = self.cond(c)?;
            return Ok(cases.into_iter().map(|(when, t)| (when, !t)).collect());
        }
        let (Cond::Le(a, b) | Cond::Lt(a, b) | Cond::Eq(a, b)) = cond else {
            unreachable!("Not is handled above")
        };
        let mut out = Vec::new();
        for case in self.cross(a, b, |x, y| self.add(x, y, -1))? {
            let (d, neg) = (case.value.clone(), self.negate(&case.value));
            let with = |rows: Vec<Poly>| case.when.iter().cloned().chain(rows).collect();
            match cond {
                Cond::Le(..) => {
                    out.push((with(vec![d]), true));
                    out.push((with(vec![self.strict(neg)]), false));
                }
                Cond::Lt(..) => {
                    out.push((with(vec![self.strict(d)]), true));
                    out.push((with(vec![neg]), false));
                }
                _ => {
                    out.push((with(vec![d.clone(), neg.clone()]), true));
                    out.push((with(vec![self.strict(d)]), false));
                    out.push((with(vec![self.strict(neg)]), false));
                }
            }
        }
        Ok(out)
    }

    /// `max(a, b)` (`a` when `b <= a`) or `min(a, b)` (`a` when `a <= b`),
    /// normalising each operand once.
    fn pick(&self, a: &Term, b: &Term, max: bool) -> Result<Vec<Case>, String> {
        if self.sort == Sort::Bv32 {
            return Err("comparison over Z/2^32 is outside the fragment".to_owned());
        }
        let (a, b) = (self.normalize(a)?, self.normalize(b)?);
        if a.len().saturating_mul(b.len()) > MAX_CASES {
            return Err("too many cases".to_owned());
        }
        let mut out = Vec::new();
        for x in &a {
            for y in &b {
                let (low, high) = if max { (y, x) } else { (x, y) };
                let le = self
                    .add(&low.value, &high.value, -1)
                    .ok_or("polynomial too large")?;
                let both = || x.when.iter().chain(&y.when).cloned();
                let mut options = vec![(both().chain([le.clone()]).collect::<Vec<_>>(), x)];
                if self.mutation != Mutation::OneCase {
                    options.push((both().chain([self.strict(self.negate(&le))]).collect(), y));
                }
                for (when, case) in options {
                    if !self.infeasible(&when) {
                        out.push(Case {
                            when,
                            value: case.value.clone(),
                        });
                    }
                }
            }
        }
        Ok(out)
    }

    fn ite(&self, cond: &Cond, then: &Term, other: &Term) -> Result<Vec<Case>, String> {
        let (then, other) = (self.normalize(then)?, self.normalize(other)?);
        let mut out = Vec::new();
        for (when, truth) in self.cond(cond)? {
            if !truth && self.mutation == Mutation::OneCase {
                continue;
            }
            for case in if truth { &then } else { &other } {
                let when: Vec<Poly> = when.iter().chain(&case.when).cloned().collect();
                if !self.infeasible(&when) {
                    out.push(Case {
                        when,
                        value: case.value.clone(),
                    });
                }
            }
            if out.len() > MAX_CASES {
                return Err("too many cases".to_owned());
            }
        }
        Ok(out)
    }

    fn normalize(&self, term: &Term) -> Result<Vec<Case>, String> {
        let minus = if self.mutation == Mutation::SubAsAdd {
            1
        } else {
            -1
        };
        let one = |value| {
            vec![Case {
                when: Vec::new(),
                value,
            }]
        };
        Ok(match term {
            Term::Var(name) => {
                one(self.reduce(Poly::from([(vec![name.clone()], BigInt::from(1))])))
            }
            Term::Lit(x) => one(self.reduce(Poly::from([(Vec::new(), BigInt::from(*x))]))),
            Term::Add(a, b) => self.cross(a, b, |x, y| self.add(x, y, 1))?,
            Term::Sub(a, b) => self.cross(a, b, |x, y| self.add(x, y, minus))?,
            Term::Mul(a, b) => self.cross(a, b, |x, y| self.mul(x, y))?,
            Term::Neg(a) => {
                let mut cases = self.normalize(a)?;
                for case in &mut cases {
                    case.value = self.negate(&case.value);
                }
                cases
            }
            Term::Max(a, b) => self.pick(a, b, true)?,
            Term::Min(a, b) => self.pick(a, b, false)?,
            Term::Ite(c, a, b) => self.ite(c, a, b)?,
        })
    }

    fn equation(&self, lhs: &Term, rhs: &Term) -> Result<(), String> {
        let (l, r) = (self.normalize(lhs)?, self.normalize(rhs)?);
        if l.len().saturating_mul(r.len()) > MAX_CASES {
            return Err("too many cases".to_owned());
        }
        for x in &l {
            for y in &r {
                let d = self
                    .add(&x.value, &y.value, -1)
                    .ok_or("polynomial too large")?;
                if d.is_empty() {
                    continue;
                }
                let when: Vec<Poly> = x.when.iter().chain(&y.when).cloned().collect();
                let below = when.iter().cloned().chain([self.strict(d.clone())]);
                let neg = self.negate(&d);
                let above = when.iter().cloned().chain([self.strict(neg)]);
                let decided = self.sort == Sort::Int
                    && self.infeasible(&below.collect::<Vec<_>>())
                    && self.infeasible(&above.collect::<Vec<_>>());
                if !decided {
                    return Err("normal forms differ".to_owned());
                }
            }
        }
        Ok(())
    }
}

/// Decide `formula`; [`Verdict::Proved`] only for a valid formula.
pub fn decide(formula: &Formula) -> Verdict {
    decide_mutated(formula, Mutation::None)
}

/// [`decide`] with one proof rule broken by `mutation`.
pub fn decide_mutated(formula: &Formula, mutation: Mutation) -> Verdict {
    let prover = Prover {
        sort: formula.sort,
        mutation,
    };
    let take = if mutation == Mutation::FirstOnly {
        1
    } else {
        usize::MAX
    };
    for (i, (lhs, rhs)) in formula.equations.iter().take(take).enumerate() {
        if let Err(why) = prover.equation(lhs, rhs) {
            return Verdict::Unproved(format!("equation {i}: {why}"));
        }
    }
    Verdict::Proved
}
