//! The matrix-scan matcher's helpers: the analysis footprint and admission of its state-name list,
//! and the row matcher that peels a dot product iteratively.
use super::{Error, binding, call, flat, index, integer, mentions};
use crate::{
    budget::{ByteBudget, Footprint, array_bytes},
    front::{Expr, Stmt, named},
};

/// Analysis bytes the matcher holds for a `d`-dimensional matrix scan: the state-name list (an
/// exactly reserved `Vec<String>`) plus the `name_bytes` of the identifiers it clones into it.
/// The matcher itself is iterative, so the size of the form is bounded by this storage and not by
/// the stack; what it reads is never refused for its size alone, only for exceeding the budget
/// or the family (`FamilyBound`).
pub fn matcher_footprint(d: usize, name_bytes: usize) -> Footprint {
    Footprint::ZERO
        .analysis(array_bytes::<String>(d))
        .analysis(Some(name_bytes))
}
/// The `d` state names, admitted by their exact owned bytes before any is cloned.
pub(super) fn state_names(
    stmts: &[Stmt],
    d: usize,
    seed: &str,
    budget: &ByteBudget,
) -> Result<Vec<String>, Error> {
    let mut name_bytes = 0usize;
    for s in &stmts[..d] {
        let (n, _) = binding(s, true).ok_or(Error::Unrecognized)?;
        name_bytes = name_bytes.saturating_add(n.len());
    }
    budget
        .admit("summary_matcher", &matcher_footprint(d, name_bytes))
        .map_err(Error::Exceeded)?;
    let mut states = Vec::with_capacity(d);
    for (j, s) in stmts[..d].iter().enumerate() {
        let (n, e) = binding(s, true).ok_or(Error::Unrecognized)?;
        if index(e, seed).and_then(integer) != Some(j as u32) {
            return Err(Error::Unrecognized);
        }
        states.push(n.to_owned());
    }
    Ok(states)
}
pub(super) fn dot(
    e: &Expr,
    coeff: &str,
    iter: &str,
    states: &[String],
    row: usize,
    terms: usize,
) -> Result<(), Error> {
    // Peel `wrapping_add(prior, wrapping_mul(coeff[row*d+col], state[col]))` from the last column
    // down; a loop, so a long row costs no stack.
    let d = states.len();
    let (mut e, mut terms) = (e, terms);
    while terms != 0 {
        let [prior, term] = call(e, "wrapping_add").ok_or(Error::Unrecognized)? else {
            return Err(Error::Unrecognized);
        };
        let [coefficient, state] = call(term, "wrapping_mul").ok_or(Error::Unrecognized)? else {
            return Err(Error::Unrecognized);
        };
        if mentions(coefficient, states) {
            return Err(Error::StateDependent);
        }
        let col = terms - 1;
        if !index(coefficient, coeff)
            .is_some_and(|i| flat(i, iter, (d * d) as u32, (row * d + col) as u32))
            || !named(state, &states[col])
        {
            return Err(Error::Unrecognized);
        }
        (e, terms) = (prior, col);
    }
    (integer(e) == Some(0))
        .then_some(())
        .ok_or(Error::Unrecognized)
}
