//! L8 finite-state map (#59, normal-form design section 2): a carried `u32` state updated
//! only by `s := table[stride * s + c]` from one table the loop never writes.
//! Each iteration is then a map `[0, S) -> [0, S)` with `S = len(table) /
//! stride`, and maps compose, so the loop is a scan over function tables.
//! Work grows by `S` per iteration, so the law is cost-gated (`gate.rs`).

use std::collections::BTreeSet;

use super::{
    Guard, LawId, Rep, RepExpr,
    linear::{const_val, linform},
    loops::Context,
};
use crate::{front::Type, general_net::regions::tree::Expr};

/// Largest state count the executor enumerates per lane; a larger table fails
/// [`Guard::StateTable`] and the loop runs in source order.
pub(super) const MAX_STATES: u32 = 64;

/// The L8 law of carried `var` whose every update is a table read.
pub(super) fn law<'e>(
    cx: &Context<'_>,
    var: &str,
    values: impl IntoIterator<Item = &'e Expr>,
) -> Result<(LawId, Rep, Vec<Guard>), String> {
    if cx.tree.locals.get(var) != Some(&Type::U32) {
        return Err(format!("finite-state map on non-u32 {var}"));
    }
    let own = BTreeSet::from([var.to_owned()]);
    let mut shape = BTreeSet::new();
    for value in values {
        let Expr::Index { array, index, .. } = value else {
            return Err(format!("finite-state update of {var} is not a table read"));
        };
        let form = linform(index, &own).ok_or("finite-state index not linear")?;
        let stride = form
            .get(&Some(var.to_owned()))
            .and_then(const_val)
            .and_then(|k| u32::try_from(k).ok())
            .filter(|&k| k > 0)
            .ok_or("finite-state stride is not a positive literal")?;
        shape.insert((array.clone(), stride));
    }
    let Ok([(table, stride)]) = <[_; 1]>::try_from(shape.into_iter().collect::<Vec<_>>()) else {
        return Err("finite-state updates read different tables or strides".to_owned());
    };
    if cx.tree.locals.get(&table) != Some(&Type::ArrayU32) {
        return Err("finite-state table is not Array<u32>".to_owned());
    }
    let guard = Guard::StateTable {
        table: table.clone(),
        stride,
        seed: var.to_owned(),
    };
    let rep = Rep::Lifted(RepExpr::FunctionTable { table, stride });
    Ok((LawId::L8FiniteState, rep, vec![guard]))
}
