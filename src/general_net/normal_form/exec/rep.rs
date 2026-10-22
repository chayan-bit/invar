//! Law representations as executable elements: identity, lane element,
//! associative combine and application to the loop-entry seed (#133 section 3).

use super::{
    Stop,
    value::{Sym, Val},
};
use crate::{
    general_net::normal_form::{Guard, IntTy, Order, Rep, RepExpr, Ring, ScalarTy},
    laws::Law,
    numeric_contract::Comparison,
};

/// Saturation magnitude for trap-lifted products: anything above the i32/u32
/// range already implies a trap once multiplied by a non-zero seed.
const SATURATE: i128 = 1 << 40;

/// `sum_j coeffs[j] x_j + constant` in `ring`; missing coefficients are zero.
#[derive(Clone, Debug, PartialEq)]
pub(in crate::general_net::normal_form) struct Lin {
    pub ring: Ring,
    pub coeffs: Vec<u64>,
    pub constant: u64,
}

fn modulus(ring: Ring) -> Option<u64> {
    match ring {
        Ring::Wrapping32 => Some(1 << 32),
        Ring::Checked64(_) => None,
        Ring::Residue { modulus, .. } => Some(u64::from(modulus)),
    }
}

pub(super) fn ring_add(ring: Ring, a: u64, b: u64) -> u64 {
    match modulus(ring) {
        None => a.wrapping_add(b),
        Some(m) => ((u128::from(a) + u128::from(b)) % u128::from(m)) as u64,
    }
}

pub(super) fn ring_mul(ring: Ring, a: u64, b: u64) -> u64 {
    match modulus(ring) {
        None => a.wrapping_mul(b),
        Some(m) => ((u128::from(a) * u128::from(b)) % u128::from(m)) as u64,
    }
}

impl Lin {
    pub(super) fn unit(ring: Ring, k: usize, j: usize) -> Self {
        let mut coeffs = vec![0; k];
        if let Some(slot) = coeffs.get_mut(j) {
            *slot = 1;
        }
        Self {
            ring,
            coeffs,
            constant: 0,
        }
    }

    pub(super) fn constant(ring: Ring, k: usize, value: u64) -> Self {
        Self {
            ring,
            coeffs: vec![0; k],
            constant: value,
        }
    }

    pub(super) fn coefficient(&self, j: usize) -> u64 {
        self.coeffs.get(j).copied().unwrap_or(0)
    }

    pub(super) fn is_constant(&self) -> bool {
        self.coeffs.iter().all(|c| *c == 0)
    }

    pub(super) fn add(&self, other: &Self) -> Self {
        let k = self.coeffs.len().max(other.coeffs.len());
        Self {
            ring: self.ring,
            coeffs: (0..k)
                .map(|j| ring_add(self.ring, self.coefficient(j), other.coefficient(j)))
                .collect(),
            constant: ring_add(self.ring, self.constant, other.constant),
        }
    }

    pub(super) fn scale(&self, factor: u64) -> Self {
        Self {
            ring: self.ring,
            coeffs: self
                .coeffs
                .iter()
                .map(|c| ring_mul(self.ring, *c, factor))
                .collect(),
            constant: ring_mul(self.ring, self.constant, factor),
        }
    }

    /// Additive inverse; only exact 64-bit rings have one that matches source
    /// subtraction (Lemma B forbids it under a modulus).
    pub(super) fn neg(&self) -> Option<Self> {
        match self.ring {
            Ring::Checked64(_) | Ring::Wrapping32 => Some(self.scale(match self.ring {
                Ring::Wrapping32 => u64::from(u32::MAX),
                _ => u64::MAX,
            })),
            Ring::Residue { .. } => None,
        }
    }
}

/// Embed a concrete scalar into `ring`. Under a modulus, a negative value
/// violates Lemma B: `guard` is the runtime guard that fails.
pub(super) fn embed(
    ring: Ring,
    value: &Val,
    guard: Option<Guard>,
    checked: bool,
) -> Result<u64, Stop> {
    let x = value
        .int()
        .ok_or_else(|| Stop::violation("non-integer operand of a linear law"))?;
    match ring {
        Ring::Wrapping32 => match value {
            Val::U32(x) => Ok(u64::from(*x)),
            _ => Err(Stop::violation("wrapping law on a non-u32 operand")),
        },
        Ring::Checked64(_) => Ok(x as i64 as u64),
        Ring::Residue { modulus, .. } => {
            if x < 0 && checked {
                return Err(Stop::Guard(guard.unwrap_or(Guard::NonNegativeConstants)));
            }
            Ok(x.rem_euclid(i128::from(modulus)) as u64)
        }
    }
}

/// Lemma B entry guard: `seed` lies in `[0, P)` of the residue ring of `rep`,
/// so its ring word is its source value. A seed at or above `P` would replay
/// from `seed mod P` and miss the overflow of `a s + b` (#266).
pub(super) fn reduced(rep: Option<&Rep>, seed: Option<i128>) -> bool {
    match (rep.and_then(super::sym::ring_of), seed) {
        (Some(Ring::Residue { modulus, .. }), Some(x)) => (0..i128::from(modulus)).contains(&x),
        _ => false,
    }
}

/// The scalar a ring word denotes in type `ty`, if representable.
pub(super) fn denote(ring: Ring, word: u64) -> Option<Val> {
    match ring {
        Ring::Wrapping32 => u32::try_from(word).ok().map(Val::U32),
        Ring::Checked64(ty) => Val::from_int(ty, i128::from(word as i64)),
        Ring::Residue { ty, .. } => Val::from_int(ty, i128::from(word)),
    }
}

/// One lane's (or segment's) element of a carried representation.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Elem {
    /// L2: one u32 word.
    Word(u32),
    /// L3 sum / L13 counter: total and the extreme running partial sums.
    Sum { total: i128, low: i128, high: i128 },
    /// L3 product, saturated.
    Prod(i128),
    /// L4/L5: `(k+1) x (k+1)` row-major matrix in the ring.
    Mat { ring: Ring, k: usize, m: Vec<u64> },
    /// L6: `x -> order(x + a, b)`; `None` is the identity's infinity.
    Trop { a: Option<i128>, b: Option<i128> },
    /// L7: best candidate value and its companions.
    Select(Option<(Val, Vec<Val>)>),
    /// L12: last value written.
    Last(Option<Val>),
    /// L8: the state each state `s` maps to (`None`: the run from `s` ended
    /// in an event); the empty map is the identity.
    Table(Vec<Option<u32>>),
}

pub(super) fn identity(rep: &Rep) -> Elem {
    match rep {
        Rep::Sealed(law) => Elem::Word(law.identity()),
        Rep::LiftedSum(_) | Rep::Counter(_) => Elem::Sum {
            total: 0,
            low: 0,
            high: 0,
        },
        Rep::SaturatedProduct(_) => Elem::Prod(1),
        Rep::Affine(ring) => mat_identity(*ring, 1),
        Rep::Matrix { ring, vars } => mat_identity(*ring, vars.len()),
        Rep::Tropical { .. } => Elem::Trop {
            a: Some(0),
            b: None,
        },
        Rep::Select { .. } => Elem::Select(None),
        Rep::Last => Elem::Last(None),
        Rep::Lifted(RepExpr::FunctionTable { .. }) => Elem::Table(Vec::new()),
    }
}

fn mat_identity(ring: Ring, k: usize) -> Elem {
    let n = k + 1;
    let mut m = vec![0; n * n];
    for i in 0..n {
        m[i * n + i] = 1;
    }
    Elem::Mat { ring, k, m }
}

fn saturate(x: i128) -> i128 {
    x.clamp(-SATURATE, SATURATE)
}

fn tropical(order: Order, a: Option<i128>, b: Option<i128>) -> Option<i128> {
    match (a, b) {
        (Some(a), Some(b)) => Some(if order == Order::Max {
            a.max(b)
        } else {
            a.min(b)
        }),
        (x, None) | (None, x) => x,
    }
}

/// Whether candidate `c` replaces the current best `cur` (source order: `cur`
/// came first).
pub(super) fn replaces(order: Order, strict: bool, c: &Val, cur: &Val) -> bool {
    match (c, cur) {
        (Val::F32(c), Val::F32(cur)) => {
            let predicate = match (order, strict) {
                (Order::Min, true) => Comparison::Lt,
                (Order::Min, false) => Comparison::Le,
                (Order::Max, true) => Comparison::Gt,
                (Order::Max, false) => Comparison::Ge,
            };
            c.compare(predicate, *cur)
        }
        _ => match (c.int(), cur.int()) {
            (Some(c), Some(cur)) => match (order, strict) {
                (Order::Min, true) => c < cur,
                (Order::Min, false) => c <= cur,
                (Order::Max, true) => c > cur,
                (Order::Max, false) => c >= cur,
            },
            _ => false,
        },
    }
}

/// Associative combine: `left` covers earlier iterations than `right`.
pub(super) fn combine(rep: &Rep, left: &Elem, right: &Elem) -> Result<Elem, Stop> {
    Ok(match (left, right) {
        (Elem::Word(a), Elem::Word(b)) => Elem::Word(match rep {
            Rep::Sealed(Law::WrappingProduct) => a.wrapping_mul(*b),
            _ => a.wrapping_add(*b),
        }),
        (
            Elem::Sum { total, low, high },
            Elem::Sum {
                total: t2,
                low: l2,
                high: h2,
            },
        ) => Elem::Sum {
            total: total + t2,
            low: (*low).min(total + l2),
            high: (*high).max(total + h2),
        },
        (Elem::Prod(a), Elem::Prod(b)) => Elem::Prod(saturate(a * b)),
        (Elem::Mat { ring, k, m: f }, Elem::Mat { m: g, .. }) => {
            // Apply `left` first: the product `right * left`.
            let n = k + 1;
            let mut m = vec![0; n * n];
            for i in 0..n {
                for j in 0..n {
                    let mut acc = 0;
                    for l in 0..n {
                        acc = ring_add(*ring, acc, ring_mul(*ring, g[i * n + l], f[l * n + j]));
                    }
                    m[i * n + j] = acc;
                }
            }
            Elem::Mat {
                ring: *ring,
                k: *k,
                m,
            }
        }
        (Elem::Trop { a, b }, Elem::Trop { a: a2, b: b2 }) => {
            let Rep::Tropical { order, .. } = rep else {
                return Err(Stop::internal("tropical element without tropical rep"));
            };
            let shift = |x: Option<i128>, y: Option<i128>| x.zip(y).map(|(x, y)| x + y);
            Elem::Trop {
                a: shift(*a, *a2),
                b: tropical(*order, shift(*b, *a2), *b2),
            }
        }
        (Elem::Select(l), Elem::Select(r)) => {
            let Rep::Select { order, strict, .. } = rep else {
                return Err(Stop::internal("select element without select rep"));
            };
            Elem::Select(match (l, r) {
                (x, None) | (None, x) => x.clone(),
                (Some(l), Some(r)) => {
                    if replaces(*order, *strict, &r.0, &l.0) {
                        Some(r.clone())
                    } else {
                        Some(l.clone())
                    }
                }
            })
        }
        (Elem::Last(l), Elem::Last(r)) => Elem::Last(r.clone().or_else(|| l.clone())),
        (Elem::Table(f), Elem::Table(g)) if f.is_empty() || g.is_empty() => {
            Elem::Table(if f.is_empty() { g.clone() } else { f.clone() })
        }
        // Apply `left` first: `g` after `f`.
        (Elem::Table(f), Elem::Table(g)) => Elem::Table(
            f.iter()
                .map(|s| s.and_then(|s| g.get(s as usize).copied().flatten()))
                .collect(),
        ),
        _ => return Err(Stop::internal("mismatched elements")),
    })
}

/// Exclusive prefixes and the total of `elems`, computed by a balanced tree
/// (so the result relies on associativity, not on source order). `wrong`
/// is the wrong-combine mutation: keep only the right operand.
pub(super) fn scan(rep: &Rep, elems: &[Elem], wrong: bool) -> Result<(Vec<Elem>, Elem), Stop> {
    let join = |l: &Elem, r: &Elem| {
        if wrong {
            Ok(r.clone())
        } else {
            combine(rep, l, r)
        }
    };
    fn go(
        elems: &[Elem],
        offset: &Elem,
        join: &dyn Fn(&Elem, &Elem) -> Result<Elem, Stop>,
        out: &mut Vec<Elem>,
    ) -> Result<Elem, Stop> {
        match elems {
            [] => Ok(offset.clone()),
            [one] => {
                out.push(offset.clone());
                join(offset, one)
            }
            _ => {
                let (left, right) = elems.split_at(elems.len() / 2);
                let total_left = reduce(left, join)?;
                let mut tail = Vec::new();
                let mid = join(offset, &total_left)?;
                go(left, offset, join, out)?;
                let end = go(right, &mid, join, &mut tail)?;
                out.extend(tail);
                Ok(end)
            }
        }
    }
    fn reduce(
        elems: &[Elem],
        join: &dyn Fn(&Elem, &Elem) -> Result<Elem, Stop>,
    ) -> Result<Elem, Stop> {
        match elems {
            [] => Err(Stop::internal("empty reduce")),
            [one] => Ok(one.clone()),
            _ => {
                let (left, right) = elems.split_at(elems.len() / 2);
                join(&reduce(left, join)?, &reduce(right, join)?)
            }
        }
    }
    let mut out = Vec::with_capacity(elems.len());
    let total = go(elems, &identity(rep), &join, &mut out)?;
    Ok((out, total))
}

/// The lane element of `rep` from the group state at the end of a lane.
pub(super) fn element(
    rep: &Rep,
    state: &[Val],
    select: &Option<(Val, Vec<Val>)>,
    range: (i128, i128),
) -> Result<Elem, Stop> {
    let lin = |v: &Val| match v.sym() {
        Some(Sym::Lin(lin)) => Ok(lin.clone()),
        _ => Err(Stop::violation("linear law state is not linear")),
    };
    let first = state.first().ok_or_else(|| Stop::internal("empty group"))?;
    Ok(match rep {
        Rep::Sealed(Law::WrappingSum) => {
            let l = lin(first)?;
            if l.coefficient(0) != 1 {
                return Err(Stop::violation("sealed sum with a non-unit coefficient"));
            }
            Elem::Word(u32::try_from(l.constant).map_err(|_| Stop::internal("word"))?)
        }
        Rep::Sealed(_) => {
            let l = lin(first)?;
            if l.constant != 0 {
                return Err(Stop::violation("sealed product with an offset"));
            }
            Elem::Word(u32::try_from(l.coefficient(0)).map_err(|_| Stop::internal("word"))?)
        }
        Rep::LiftedSum(_) | Rep::Counter(_) => {
            let l = lin(first)?;
            if l.coefficient(0) != 1 {
                return Err(Stop::violation("lifted sum with a non-unit coefficient"));
            }
            let total = i128::from(l.constant as i64);
            Elem::Sum {
                total,
                low: range.0.min(total),
                high: range.1.max(total),
            }
        }
        Rep::SaturatedProduct(_) => {
            let l = lin(first)?;
            if l.constant != 0 {
                return Err(Stop::violation("lifted product with an offset"));
            }
            Elem::Prod(saturate(i128::from(l.coefficient(0) as i64)))
        }
        Rep::Affine(ring) | Rep::Matrix { ring, .. } => {
            let k = state.len();
            let n = k + 1;
            let mut m = vec![0; n * n];
            for (i, value) in state.iter().enumerate() {
                let l = lin(value)?;
                for j in 0..k {
                    m[i * n + j] = l.coefficient(j);
                }
                m[i * n + k] = l.constant;
            }
            m[k * n + k] = 1;
            Elem::Mat { ring: *ring, k, m }
        }
        Rep::Tropical { .. } => match first.sym() {
            Some(Sym::Trop { a, b }) => Elem::Trop { a: *a, b: *b },
            _ => return Err(Stop::violation("tropical state is not tropical")),
        },
        Rep::Select { .. } => Elem::Select(select.clone()),
        Rep::Last => match first.sym() {
            Some(Sym::Last(v)) => Elem::Last(v.as_deref().cloned()),
            _ => return Err(Stop::violation("last-writer state")),
        },
        Rep::Lifted(_) => return Err(Stop::internal("table elements are enumerated")),
    })
}

/// Apply a prefix element to the seed values of the group. A value that is
/// not representable becomes poison: its first use is a lane event that an
/// earlier, exact lane must precede.
pub(super) fn apply(
    rep: &Rep,
    elem: &Elem,
    seeds: &[Val],
    companions: &[Val],
) -> Result<(Vec<Val>, Vec<Val>), Stop> {
    let seed = seeds.first().ok_or_else(|| Stop::internal("empty seed"))?;
    let int = |ty: IntTy, x: i128| Val::from_int(ty, x).unwrap_or_else(Val::poison);
    let seed_int = || seed.int().ok_or_else(|| Stop::internal("integer seed"));
    let values = match (rep, elem) {
        (Rep::Sealed(law), Elem::Word(w)) => {
            let Val::U32(s) = seed else {
                return Err(Stop::internal("sealed seed"));
            };
            vec![Val::U32(match law {
                Law::WrappingSum => s.wrapping_add(*w),
                Law::WrappingProduct => s.wrapping_mul(*w),
            })]
        }
        (Rep::LiftedSum(ty) | Rep::Counter(ty), Elem::Sum { total, .. }) => {
            vec![int(*ty, seed_int()? + total)]
        }
        (Rep::SaturatedProduct(ty), Elem::Prod(p)) => vec![int(*ty, seed_int()? * p)],
        (Rep::Affine(_) | Rep::Matrix { .. }, Elem::Mat { ring, k, m }) => {
            let n = k + 1;
            let words = seeds
                .iter()
                .map(|s| embed(*ring, s, None, false))
                .collect::<Result<Vec<_>, _>>()?;
            (0..*k)
                .map(|i| {
                    let mut acc = m[i * n + k];
                    for (j, word) in words.iter().enumerate() {
                        acc = ring_add(*ring, acc, ring_mul(*ring, m[i * n + j], *word));
                    }
                    denote(*ring, acc).unwrap_or_else(Val::poison)
                })
                .collect()
        }
        (Rep::Tropical { order, ty }, Elem::Trop { a, b }) => {
            let shifted = a.map(|a| seed_int().map(|s| s + a)).transpose()?;
            match tropical(*order, shifted, *b) {
                Some(x) => vec![int(*ty, x)],
                None => return Err(Stop::internal("tropical infinity")),
            }
        }
        (Rep::Select { order, strict, .. }, Elem::Select(best)) => {
            return Ok(match best {
                Some((v, comps)) if replaces(*order, *strict, v, seed) => {
                    (vec![v.clone()], comps.clone())
                }
                _ => (vec![seed.clone()], companions.to_vec()),
            });
        }
        (Rep::Last, Elem::Last(v)) => vec![v.clone().unwrap_or_else(|| seed.clone())],
        (Rep::Lifted(_), Elem::Table(map)) => vec![match seed {
            Val::U32(s) if map.is_empty() => Val::U32(*s),
            Val::U32(s) => map
                .get(*s as usize)
                .copied()
                .flatten()
                .map_or_else(Val::poison, Val::U32),
            _ => Val::poison(),
        }],
        _ => return Err(Stop::internal("element does not match its representation")),
    };
    Ok((values, companions.to_vec()))
}

/// The scalar type a select element ranges over.
pub(super) fn is_nan(value: &Val, elem: ScalarTy) -> bool {
    matches!((value, elem), (Val::F32(x), ScalarTy::F32) if x.is_nan())
}
