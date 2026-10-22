//! Static proof that an interval recursion's children partition its node.
//!
//! [`super::interval_pair`] proves the chain `lo .. m .. hi` syntactically:
//! the first child starts at `lo`, the last ends at `hi`, and neighbours
//! share a bound. What remains is that every child is nonempty, proven here
//! as `hi_j - lo_j > 0` for each self call under its path conditions.
//!
//! A value is `sum c_s * s + [L(len), U(len)]`: integer coefficients over
//! the symbols `lo` and the root's `let` locals, plus an affine lower and
//! upper bound in `len = hi - lo` with rational coefficients. Locals stay
//! symbolic until a check, so `m2 - m1` cancels `m1` before any bound is
//! taken, and separate affine bounds keep `x * k` correlated for `x >= 0`.
//! When every child is proven nonempty the device lanes skip the run-time
//! geometry checks; the CPU model keeps them and aborts if one ever fails.

use std::{cmp::Ordering, collections::BTreeMap};

use super::range::Iv;
use crate::{
    front::{BinaryOp, UnaryOp},
    general_net::regions::tree::{Expr, FunctionTree, Stmt},
};

/// Exact rational `n / d` with `d > 0`; arithmetic fails on overflow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Q(i128, i128);

const fn gcd(a: i128, b: i128) -> i128 {
    if b == 0 { a.abs() } else { gcd(b, a % b) }
}

impl Q {
    const ZERO: Self = Self(0, 1);

    const fn int(n: i128) -> Self {
        Self(n, 1)
    }
    fn new(n: i128, d: i128) -> Option<Self> {
        if d == 0 {
            return None;
        }
        let g = gcd(n, d).max(1).checked_mul(d.signum())?;
        Some(Self(n / g, d / g))
    }
    fn add(self, o: Self) -> Option<Self> {
        let n = self
            .0
            .checked_mul(o.1)?
            .checked_add(o.0.checked_mul(self.1)?)?;
        Self::new(n, self.1.checked_mul(o.1)?)
    }
    fn mul(self, o: Self) -> Option<Self> {
        Self::new(self.0.checked_mul(o.0)?, self.1.checked_mul(o.1)?)
    }
    fn cmp(self, o: Self) -> Option<Ordering> {
        Some(self.0.checked_mul(o.1)?.cmp(&o.0.checked_mul(self.1)?))
    }
    fn min(self, o: Self) -> Option<Self> {
        Some(if self.cmp(o)?.is_le() { self } else { o })
    }
    fn max(self, o: Self) -> Option<Self> {
        Some(if self.cmp(o)?.is_ge() { self } else { o })
    }
    /// Above `-1`: an integer bounded below by this is non-negative.
    fn nonneg(self) -> bool {
        self.cmp(Self::int(-1)).is_some_and(Ordering::is_gt)
    }
    /// Below `1`: an integer bounded above by this is non-positive.
    fn nonpos(self) -> bool {
        self.cmp(Self::int(1)).is_some_and(Ordering::is_lt)
    }
    const fn floor(self) -> i128 {
        self.0.div_euclid(self.1)
    }
    const fn ceil(self) -> i128 {
        -(-self.0).div_euclid(self.1)
    }
}

/// `a * len + b`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Aff(Q, Q);

impl Aff {
    const fn constant(b: Q) -> Self {
        Self(Q::ZERO, b)
    }
    fn add(self, o: Self) -> Option<Self> {
        Some(Self(self.0.add(o.0)?, self.1.add(o.1)?))
    }
    fn scale(self, k: Q) -> Option<Self> {
        Some(Self(self.0.mul(k)?, self.1.mul(k)?))
    }
    fn at(self, len: i128) -> Option<Q> {
        self.0.mul(Q::int(len))?.add(self.1)
    }
}

/// `sum c_s * s + [lo(len), hi(len)]` over symbols `s` (`lo` and locals).
#[derive(Clone, Debug, PartialEq, Eq)]
struct Form {
    syms: BTreeMap<String, i128>,
    lo: Aff,
    hi: Aff,
}

impl Form {
    fn constant(lo: Q, hi: Q) -> Self {
        Self {
            syms: BTreeMap::new(),
            lo: Aff::constant(lo),
            hi: Aff::constant(hi),
        }
    }
    fn int(n: i128) -> Self {
        Self::constant(Q::int(n), Q::int(n))
    }
    /// `name + len_coefficient * len`.
    fn symbol(name: &str, len_coefficient: i128) -> Self {
        let bound = Aff(Q::int(len_coefficient), Q::ZERO);
        Self {
            syms: BTreeMap::from([(name.to_owned(), 1)]),
            lo: bound,
            hi: bound,
        }
    }
    fn add(&self, o: &Self) -> Option<Self> {
        let mut syms = self.syms.clone();
        for (s, c) in &o.syms {
            let e = syms.entry(s.clone()).or_default();
            *e = e.checked_add(*c)?;
        }
        syms.retain(|_, c| *c != 0);
        Some(Self {
            syms,
            lo: self.lo.add(o.lo)?,
            hi: self.hi.add(o.hi)?,
        })
    }
    /// Multiply by the integer `n`.
    fn times(&self, n: i128) -> Option<Self> {
        let k = Q::int(n);
        let (lo, hi) = if n >= 0 {
            (self.lo.scale(k)?, self.hi.scale(k)?)
        } else {
            (self.hi.scale(k)?, self.lo.scale(k)?)
        };
        let syms = self
            .syms
            .iter()
            .filter(|_| n != 0)
            .map(|(s, c)| c.checked_mul(n).map(|v| (s.clone(), v)))
            .collect::<Option<_>>()?;
        Some(Self { syms, lo, hi })
    }
    fn sub(&self, o: &Self) -> Option<Self> {
        self.add(&o.times(-1)?)
    }
}

struct Ctx<'a> {
    lo: &'a str,
    hi: &'a str,
    /// Range of every root parameter over all nodes (the fixpoint box).
    params: BTreeMap<&'a str, Iv>,
    /// The root's `let` locals on this path, in definition order; each
    /// definition may name `lo` and earlier locals.
    lets: Vec<(String, Form)>,
    /// Integer range of `len = hi - lo` under the current path conditions.
    len: (i128, i128),
}

impl Ctx<'_> {
    /// Substitute locals, latest first, so shared terms cancel exactly.
    fn expand(&self, x: &Form) -> Option<Form> {
        let mut x = x.clone();
        for (name, def) in self.lets.iter().rev() {
            if let Some(c) = x.syms.remove(name) {
                x = x.add(&def.times(c)?)?;
            }
        }
        Some(x)
    }
    /// Bounds of `x` over every `lo`, `len` and local the path allows.
    fn range(&self, x: &Form) -> Option<(Q, Q)> {
        let x = self.expand(x)?;
        let (l0, l1) = self.len;
        let mut lo = x.lo.at(l0)?.min(x.lo.at(l1)?)?;
        let mut hi = x.hi.at(l0)?.max(x.hi.at(l1)?)?;
        for (s, c) in &x.syms {
            let iv = self.params.get(s.as_str())?;
            let a = Q::int(i128::from(iv.lo).checked_mul(*c)?);
            let b = Q::int(i128::from(iv.hi).checked_mul(*c)?);
            lo = lo.add(a.min(b)?)?;
            hi = hi.add(a.max(b)?)?;
        }
        Some((lo, hi))
    }
    /// `x` over `len` alone: symbols expanded, or flattened to its range.
    fn plain(&self, x: &Form) -> Option<Form> {
        let e = self.expand(x)?;
        if e.syms.is_empty() {
            return Some(e);
        }
        let (lo, hi) = self.range(&e)?;
        Some(Form::constant(lo, hi))
    }
    fn point(&self, x: &Form) -> Option<i128> {
        let (lo, hi) = self.range(x)?;
        (lo == hi && lo.1 == 1).then_some(lo.0)
    }
    fn product(&self, x: &Form, y: &Form) -> Option<Form> {
        if let Some(n) = self.point(y) {
            return x.times(n);
        }
        if let Some(n) = self.point(x) {
            return y.times(n);
        }
        let ((xl, xh), (yl, yh)) = (self.range(x)?, self.range(y)?);
        // For integers `f >= 0` and `g` in `[gl, gh]` with `gl >= 0`:
        // `gl * f <= f * g <= gh * f`, keeping `f`'s affine bounds.
        for (f, fl, (gl, gh)) in [(x, xl, (yl, yh)), (y, yl, (xl, xh))] {
            if fl.nonneg() && gl.nonneg() {
                let f = self.plain(f)?;
                return Some(Form {
                    syms: BTreeMap::new(),
                    lo: f.lo.scale(Q::int(gl.ceil()))?,
                    hi: f.hi.scale(Q::int(gh.floor()))?,
                });
            }
        }
        let c = [xl.mul(yl)?, xl.mul(yh)?, xh.mul(yl)?, xh.mul(yh)?];
        let lo = c[1..].iter().try_fold(c[0], |a, b| a.min(*b))?;
        let hi = c[1..].iter().try_fold(c[0], |a, b| a.max(*b))?;
        Some(Form::constant(lo, hi))
    }
    /// Truncating `x / k` for a nonzero integer `k`.
    fn quotient(&self, x: &Form, k: i128) -> Option<Form> {
        if k < 0 {
            return self.quotient(x, k.checked_neg()?)?.times(-1);
        }
        let x = self.plain(x)?;
        let (lo, hi) = self.range(&x)?;
        let inv = Q::new(1, k)?;
        // `trunc(v / k)` lies in `[v / k - s, v / k]` for `v >= 0` and in
        // `[v / k, v / k + s]` for `v <= 0`, with `s = (k - 1) / k`.
        let s = Q::new(k - 1, k)?;
        let below = if hi.nonpos() {
            Q::ZERO
        } else {
            Q::new(1 - k, k)?
        };
        let above = if lo.nonneg() { Q::ZERO } else { s };
        Some(Form {
            syms: BTreeMap::new(),
            lo: x.lo.scale(inv)?.add(Aff::constant(below))?,
            hi: x.hi.scale(inv)?.add(Aff::constant(above))?,
        })
    }
    /// `x / k`, first cancelling a common factor `g` of `k` and an integer
    /// factor `c` of `x = e * c`: `trunc(c e / k) = trunc((c/g) e / (k/g))`.
    fn divide(&self, left: &Expr, x: &Form, k: i128) -> Option<Form> {
        if k == 0 {
            return None;
        }
        if let Expr::Binary {
            op: BinaryOp::Mul,
            left: a,
            right: b,
            ..
        } = left
        {
            let (a, b) = (self.eval(a)?, self.eval(b)?);
            for (e, c) in [(&a, &b), (&b, &a)] {
                if let Some(c) = self.point(c).filter(|c| *c != 0) {
                    let g = gcd(c, k);
                    if g > 1 {
                        return self.quotient(&self.product(e, &Form::int(c / g))?, k / g);
                    }
                }
            }
        }
        self.quotient(x, k)
    }
    fn remainder(&self, x: &Form, k: i128) -> Option<Form> {
        let m = k.checked_abs()?.checked_sub(1)?;
        let (lo, hi) = self.range(x)?;
        let (lo, hi) = if lo.nonneg() {
            (0, m)
        } else if hi.nonpos() {
            (-m, 0)
        } else {
            (-m, m)
        };
        Some(Form::constant(Q::int(lo), Q::int(hi)))
    }

    fn eval(&self, expr: &Expr) -> Option<Form> {
        match expr {
            Expr::I32(_) | Expr::U32(_) => Some(Form::int(expr.literal()?.into())),
            Expr::Local(n) if n == self.lo => Some(Form::symbol(n, 0)),
            Expr::Local(n) if n == self.hi => Some(Form::symbol(self.lo, 1)),
            Expr::Local(n) if self.lets.iter().any(|(l, _)| l == n) => Some(Form::symbol(n, 0)),
            Expr::Local(n) => {
                let iv = self.params.get(n.as_str())?;
                Some(Form::constant(Q::int(iv.lo.into()), Q::int(iv.hi.into())))
            }
            Expr::Unary {
                op: UnaryOp::Neg,
                value,
                ..
            } => self.eval(value)?.times(-1),
            Expr::Binary {
                op, left, right, ..
            } => {
                let (x, y) = (self.eval(left)?, self.eval(right)?);
                match op {
                    BinaryOp::Add => x.add(&y),
                    BinaryOp::Sub => x.sub(&y),
                    BinaryOp::Mul => self.product(&x, &y),
                    BinaryOp::Div => self.divide(left, &x, self.point(&y)?),
                    BinaryOp::Rem => self.remainder(&x, self.point(&y)?),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// `x > 0` for every value the path allows.
    fn positive(&self, x: &Form) -> bool {
        self.range(x)
            .and_then(|(lo, _)| lo.cmp(Q::ZERO))
            .is_some_and(Ordering::is_gt)
    }

    /// `len` narrowed by `cond == polarity` when `cond` compares an exact
    /// `a * len + b` with zero; `None` when the branch is infeasible.
    fn refine(&self, cond: &Expr, polarity: bool) -> Option<(i128, i128)> {
        let len = self.narrow(cond, polarity).unwrap_or(self.len);
        (len.0 <= len.1).then_some(len)
    }
    fn narrow(&self, cond: &Expr, polarity: bool) -> Option<(i128, i128)> {
        let Expr::Binary {
            op, left, right, ..
        } = cond
        else {
            return None;
        };
        let d = self.expand(&self.eval(left)?.sub(&self.eval(right)?)?)?;
        if !d.syms.is_empty() || d.lo != d.hi {
            return None;
        }
        // Canonical `s * (a * len + b) + t <= 0` over integers.
        let (s, t) = match (op, polarity) {
            (BinaryOp::Le, true) | (BinaryOp::Gt, false) => (1, 0),
            (BinaryOp::Lt, true) | (BinaryOp::Ge, false) => (1, 1),
            (BinaryOp::Ge, true) | (BinaryOp::Lt, false) => (-1, 0),
            (BinaryOp::Gt, true) | (BinaryOp::Le, false) => (-1, 1),
            _ => return None,
        };
        let a = d.lo.0.mul(Q::int(s))?;
        let b = d.lo.1.mul(Q::int(s))?.add(Q::int(t))?;
        if a.0 == 0 {
            // Independent of `len`: the branch is taken always or never.
            return Some(if b.0 <= 0 { self.len } else { (1, 0) });
        }
        let bound = b.mul(Q::new(-a.1, a.0)?)?;
        Some(if a.0 > 0 {
            (self.len.0, self.len.1.min(bound.floor()))
        } else {
            (self.len.0.max(bound.ceil()), self.len.1)
        })
    }
}

/// Whether every self call of `tree` (interval parameters `lo`, `hi`) gets
/// a nonempty interval, given the parameter box `params` of every node.
pub(super) fn proven(tree: &FunctionTree, (lo, hi): (usize, usize), params: &[Iv]) -> bool {
    let names: Vec<&str> = tree.params.iter().map(|p| p.name.as_str()).collect();
    let (Some(lo_iv), Some(hi_iv)) = (params.get(lo), params.get(hi)) else {
        return false;
    };
    let mut ctx = Ctx {
        lo: names[lo],
        hi: names[hi],
        params: names.iter().copied().zip(params.iter().copied()).collect(),
        lets: Vec::new(),
        len: (
            i128::from(hi_iv.lo) - i128::from(lo_iv.hi),
            i128::from(hi_iv.hi) - i128::from(lo_iv.lo),
        ),
    };
    walk(&mut ctx, tree, &tree.body, (lo, hi)).is_some()
}

/// Walk every path; `None` when a self call is not provably nonempty or
/// the body leaves the supported shape (assignments, stores, loops).
fn walk(ctx: &mut Ctx<'_>, tree: &FunctionTree, body: &[Stmt], pair: (usize, usize)) -> Option<()> {
    for (i, stmt) in body.iter().enumerate() {
        match stmt {
            Stmt::Let { name, value } => {
                nonempty(ctx, tree, value, pair)?;
                let known = ctx.lets.iter().any(|(l, _)| l == name);
                if known || name == ctx.lo || name == ctx.hi {
                    return None;
                }
                // A local the domain cannot express is a plain unknown.
                let form = ctx.eval(value).unwrap_or_else(|| {
                    Form::constant(Q::int(i32::MIN.into()), Q::int(i32::MAX.into()))
                });
                ctx.lets.push((name.clone(), form));
            }
            Stmt::Eval(value) => nonempty(ctx, tree, value, pair)?,
            Stmt::Return { value, .. } => {
                if let Some(value) = value {
                    nonempty(ctx, tree, value, pair)?;
                }
                return Some(());
            }
            Stmt::If { cond, yes, no } => {
                nonempty(ctx, tree, cond, pair)?;
                // Each branch continues with the rest of this body.
                let rest = &body[i + 1..];
                let (len, depth) = (ctx.len, ctx.lets.len());
                for (branch, polarity) in [(yes, true), (no, false)] {
                    if let Some(narrowed) = ctx.refine(cond, polarity) {
                        ctx.len = narrowed;
                        let path: Vec<Stmt> = branch.iter().chain(rest).cloned().collect();
                        let ok = walk(ctx, tree, &path, pair);
                        ctx.len = len;
                        ctx.lets.truncate(depth);
                        ok?;
                    }
                }
                return Some(());
            }
            Stmt::Assign { .. } | Stmt::Store { .. } | Stmt::Loop(_) => return None,
        }
    }
    Some(())
}

/// Every self call inside `expr` gets a nonempty interval.
fn nonempty(
    ctx: &Ctx<'_>,
    tree: &FunctionTree,
    expr: &Expr,
    (lo, hi): (usize, usize),
) -> Option<()> {
    let mut calls = Vec::new();
    self_calls(&tree.name, expr, &mut calls);
    calls.into_iter().try_for_each(|args| {
        let width = ctx.eval(args.get(hi)?)?.sub(&ctx.eval(args.get(lo)?)?)?;
        ctx.positive(&width).then_some(())
    })
}

fn self_calls<'e>(name: &str, expr: &'e Expr, out: &mut Vec<&'e [Expr]>) {
    match expr {
        Expr::Call { name: n, args, .. } => {
            if n == name {
                out.push(args);
            }
            for a in args {
                self_calls(name, a, out);
            }
        }
        Expr::Unary { value, .. } => self_calls(name, value, out),
        Expr::Index { index, .. } => self_calls(name, index, out),
        Expr::Binary { left, right, .. } | Expr::Logic { left, right, .. } => {
            self_calls(name, left, out);
            self_calls(name, right, out);
        }
        Expr::ArrayLit { items, .. } => {
            for item in items {
                self_calls(name, item, out);
            }
        }
        _ => {}
    }
}
