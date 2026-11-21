//! Reached `min_of`, `max_of`, `argmin` and `argmax` calls (#207) on CPU
//! workers: each chunk's extremum, then the ordered combine, so the bits and
//! the index equal the serial fold at every worker count, and an empty input
//! traps at the call as it does serially.

use super::{Error, Fault, Hooks, workers::parallel_chunks_with_limit};
use crate::{
    array_limits::MAX_LIVE_ELEMENTS,
    eval::{EvalError, Value},
    extremum::{self, Elem, Kind},
    front::Span,
    mir_eval::SelectCall,
};

/// One reached call and how many workers joined it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Selection {
    pub function: String,
    pub span: Span,
    pub kind: Kind,
    pub elements: usize,
    pub workers: usize,
}

fn chunked<T: Elem>(
    kind: Kind,
    values: &[T],
    workers: usize,
    hooks: &Hooks,
) -> Result<(Option<extremum::Outcome<T>>, usize), Error> {
    let max = kind.is_max();
    let (parts, _, joined) =
        parallel_chunks_with_limit(values.len(), workers, hooks, MAX_LIVE_ELEMENTS, |range| {
            Ok(extremum::extremum(max, &values[range.clone()], range.start))
        })?;
    let found = extremum::combine(max, &parts, hooks.fault == Fault::RightmostTie);
    Ok((extremum::outcome(kind, found), joined))
}

/// The call's result (or its empty-input trap) and its receipt.
pub(crate) fn cpu(
    call: &SelectCall<'_>,
    workers: usize,
    hooks: &Hooks,
) -> Result<(Result<Value, EvalError>, Selection), Error> {
    let borrowed = || Error::Candidate;
    let (result, elements, joined) = match call.operand {
        Value::ArrayI32(v) => {
            let v = v.try_borrow().map_err(|_| borrowed())?;
            let (found, joined) = chunked(call.kind, &v, workers, hooks)?;
            (
                extremum::value(found, Value::I32, call.span),
                v.len(),
                joined,
            )
        }
        Value::ArrayU32(v) => {
            let v = v.try_borrow().map_err(|_| borrowed())?;
            let (found, joined) = chunked(call.kind, &v, workers, hooks)?;
            (
                extremum::value(found, Value::U32, call.span),
                v.len(),
                joined,
            )
        }
        Value::ArrayF32(v) => {
            let v = v.try_borrow().map_err(|_| borrowed())?;
            let (found, joined) = chunked(call.kind, &v, workers, hooks)?;
            (
                extremum::value(found, Value::F32, call.span),
                v.len(),
                joined,
            )
        }
        Value::ArrayF64(v) => {
            let v = v.try_borrow().map_err(|_| borrowed())?;
            let (found, joined) = chunked(call.kind, &v, workers, hooks)?;
            (
                extremum::value(found, Value::F64, call.span),
                v.len(),
                joined,
            )
        }
        _ => return Err(Error::Candidate),
    };
    let receipt = Selection {
        function: call.function.to_owned(),
        span: call.span,
        kind: call.kind,
        elements,
        workers: joined,
    };
    Ok((result, receipt))
}
