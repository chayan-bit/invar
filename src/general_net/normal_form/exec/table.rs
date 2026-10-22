//! L8 lane elements (#281): an iteration's transition map over the states
//! `[0, S)`, found by running the lane once from every state. A run that ends
//! in an event maps its state to `None`; the lane replay from the scanned
//! prefix then meets that event in source order if the prefix reaches it.

use std::collections::HashMap;

use super::{
    Stop,
    rep::Elem,
    value::{Elems, Val},
};
use crate::general_net::normal_form::{Guard, Rep, RepExpr, fsm::MAX_STATES};

/// `S = len(table) / stride` of a table in `vars`.
fn state_count(vars: &HashMap<String, Val>, table: &str, stride: u32) -> Option<usize> {
    match vars.get(table) {
        Some(Val::Arr(array)) => Some(array.borrow().len() / usize::try_from(stride).ok()?),
        _ => None,
    }
    .filter(|&states| (1..=MAX_STATES as usize).contains(&states))
}

/// `S` of an L8 group at loop entry; `None` for other representations.
pub(super) fn states(rep: &Rep, entry: &HashMap<String, Val>) -> Result<Option<u32>, Stop> {
    let Rep::Lifted(RepExpr::FunctionTable { table, stride }) = rep else {
        return Ok(None);
    };
    let states = state_count(entry, table, *stride)
        .ok_or_else(|| Stop::violation("finite-state count out of range"))?;
    Ok(Some(states as u32))
}

/// [`Guard::StateTable`] at loop entry: the seed and every entry are states.
pub(super) fn admits(vars: &HashMap<String, Val>, guard: &Guard) -> bool {
    let Guard::StateTable {
        table,
        stride,
        seed,
    } = guard
    else {
        return true;
    };
    let (Some(states), Some(Val::Arr(array)), Some(Val::U32(seed))) = (
        state_count(vars, table, *stride),
        vars.get(table),
        vars.get(seed),
    ) else {
        return false;
    };
    let Elems::U32(entries) = &*array.borrow() else {
        return false;
    };
    (*seed as usize) < states && entries.iter().all(|&entry| (entry as usize) < states)
}

/// The lane element: `run(s)` is the state after the iteration entered in
/// state `s`, or `None` when that run ended in an event.
pub(super) fn element(
    states: u32,
    mut run: impl FnMut(u32) -> Result<Option<Val>, Stop>,
) -> Result<Elem, Stop> {
    let mut map = Vec::with_capacity(states as usize);
    for state in 0..states {
        map.push(match run(state)? {
            None => None,
            Some(Val::U32(next)) if next < states => Some(next),
            Some(_) => return Err(Stop::violation("finite-state transition out of range")),
        });
    }
    Ok(Elem::Table(map))
}
