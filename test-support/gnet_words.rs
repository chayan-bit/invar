//! The compiled GNET ABIs' oracle words, shared by the host and CPU-model
//! tests: the status and trap span an oracle result maps to, its scalar
//! result word, and the seven words of one trace event.

use invar::eval::{EvalError, TrapCategory, Value};
use invar::general_net::{compiled::Status, oracle};

/// The ABI status and trap span `(status, start, end)` of the oracle `o`;
/// panics on a result no compiled ABI encodes.
pub fn status(o: &oracle::Observation) -> (i64, i64, i64) {
    let status = match &o.semantic.result {
        Ok(_) => Status::Success,
        Err(EvalError::Trap { category, span }) => {
            let status = match category {
                TrapCategory::IntegerOverflow => Status::IntegerOverflow,
                TrapCategory::DivisionByZero => Status::DivisionByZero,
                TrapCategory::IndexOutOfBounds => Status::IndexOutOfBounds,
                other => panic!("trap {other:?} has no compiled status"),
            };
            return (status as i64, span.start as i64, span.end as i64);
        }
        Err(EvalError::EvaluationLimit { .. }) => Status::FuelLimit,
        Err(EvalError::Tool {
            code: "CALL_DEPTH_LIMIT",
            ..
        }) => Status::CallDepthLimit,
        Err(EvalError::Tool {
            code: "ARRAY_SIZE_LIMIT",
            ..
        }) => Status::ArraySizeLimit,
        Err(EvalError::Tool {
            code: "ENTRYPOINT_SIGNATURE" | "INVALID_FUEL",
            ..
        }) => Status::InvalidInput,
        Err(_) => match o.failure {
            Some(oracle::ToolFailure::TransitionLimit) => Status::TransitionLimit,
            Some(oracle::ToolFailure::TraceLimit) => Status::TraceLimit,
            Some(oracle::ToolFailure::Allocation) => Status::HeapLimit,
            _ => panic!("oracle result {o:?} has no compiled status"),
        },
    };
    (status as i64, 0, 0)
}

/// The scalar result word of `o`: the value of an integer or `bool` result,
/// the bits of an `f32` result, and 0 for a unit result or a failure.
pub fn result(o: &oracle::Observation) -> i64 {
    match &o.semantic.result {
        Ok(Value::I32(n)) => i64::from(*n),
        Ok(Value::U32(n)) => i64::from(*n),
        Ok(Value::Bool(n)) => i64::from(*n),
        Ok(Value::F32(x)) => i64::from(x.to_bits()),
        Ok(Value::Unit) | Err(_) => 0,
        Ok(other) => panic!("{other:?} is not a scalar compiled result"),
    }
}

/// The seven ABI words of one trace event: kind, frame, activation, function,
/// block, position and rule.
pub fn event(e: &oracle::Event) -> [u64; 7] {
    [
        u64::from(e.kind == oracle::EventKind::Complete),
        e.frame,
        e.activation,
        e.token.block.function as u64,
        e.token.block.block as u64,
        e.token.position as u64,
        e.rule as u64,
    ]
}

/// [`event`] as signed words, for the ABIs that read results as `i64`.
pub fn event_i64(e: &oracle::Event) -> [i64; 7] {
    event(e).map(|w| w as i64)
}
