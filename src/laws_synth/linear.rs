//! Fourier-Motzkin elimination for the case splits of [`super::prove`]: is a
//! conjunction of rows `p <= 0` unsatisfiable over `Q`? Every non-constant
//! monomial of a row is an independent unknown (`Relax`), so an
//! unsatisfiable relaxation means the rows have no integer solution either.
//! Eliminating an unknown keeps the rows free of it and adds every positive
//! combination that cancels it, which preserves satisfiability over `Q`
//! (`Eliminate`). A row with no unknowns is then decided by its constant.

use std::collections::BTreeSet;

use num_bigint::BigInt;

use super::prove::Poly;

/// Largest number of rows kept while eliminating; beyond it the rows are
/// reported satisfiable (never shown unsatisfiable: incomplete, not unsound).
const MAX_ROWS: usize = 2048;

/// Drop zero coefficients and divide by the positive gcd (same half-space).
fn scaled(mut row: Poly) -> Poly {
    row.retain(|_, c| *c != BigInt::from(0));
    let gcd = row.values().fold(BigInt::from(0), |mut a, c| {
        let mut b = if *c < BigInt::from(0) { -c } else { c.clone() };
        while b != BigInt::from(0) {
            (a, b) = (b.clone(), a % b);
        }
        a
    });
    if gcd > BigInt::from(1) {
        for c in row.values_mut() {
            *c /= &gcd;
        }
    }
    row
}

/// Whether `rows` (each `p <= 0`) is unsatisfiable. `drop_combinations`
/// is the `Eliminate` mutation: elimination keeps only the rows free of the
/// unknown.
pub(super) fn infeasible(rows: impl IntoIterator<Item = Poly>, drop_combinations: bool) -> bool {
    let mut rows: BTreeSet<Poly> = rows.into_iter().map(scaled).collect();
    loop {
        let mut unknowns = BTreeSet::new();
        for row in &rows {
            if row.keys().all(Vec::is_empty) {
                if row.values().any(|c| *c > BigInt::from(0)) {
                    return true;
                }
            } else {
                unknowns.extend(row.keys().filter(|m| !m.is_empty()).cloned());
            }
        }
        rows.retain(|row| row.keys().any(|m| !m.is_empty()));
        let sign = |row: &Poly, m: &Vec<String>| {
            row.get(m).map_or(0, |c| {
                i8::from(*c > BigInt::from(0)) - i8::from(*c < BigInt::from(0))
            })
        };
        let Some(unknown) = unknowns.into_iter().min_by_key(|m| {
            let count = |s| rows.iter().filter(|row| sign(row, m) == s).count();
            count(1) * count(-1)
        }) else {
            return false;
        };
        let (mut pos, mut neg, mut next) = (Vec::new(), Vec::new(), BTreeSet::new());
        for row in rows {
            match sign(&row, &unknown) {
                1 => pos.push(row),
                -1 => neg.push(row),
                _ => {
                    next.insert(row);
                }
            }
        }
        if !drop_combinations {
            for p in &pos {
                for n in &neg {
                    let (a, b) = (&p[&unknown], -&n[&unknown]);
                    let mut row = Poly::new();
                    for (m, c) in p {
                        *row.entry(m.clone()).or_default() += c * &b;
                    }
                    for (m, c) in n {
                        *row.entry(m.clone()).or_default() += c * a;
                    }
                    next.insert(scaled(row));
                }
                if next.len() > MAX_ROWS {
                    return false;
                }
            }
        }
        rows = next;
    }
}
