//! Intrinsics the CPU net oracle evaluates directly from their operands.

use super::{Frame, OracleError, Scalar, arrays, internal};
use crate::{
    eval::{EvalError, TrapCategory},
    front::Span,
    mir,
};

/// Intrinsics evaluated by [`direct`] or by the binary `u32` rule of `operation`.
pub(super) fn is_direct(name: &str) -> bool {
    matches!(
        name,
        "wrapping_add" | "wrapping_mul" | "filled_i32" | "filled_u32" | "filled_f32" | "assert"
    )
}

/// A filled array, or `assert` (#258) on its evaluated condition; `None`
/// for every other intrinsic.
pub(super) fn direct(
    name: &str,
    args: &[mir::ValueId],
    frame: &Frame,
    arena: &mut arrays::Arena,
    span: Span,
) -> Option<Result<Scalar, OracleError>> {
    Some(match (name, args) {
        ("filled_i32" | "filled_u32" | "filled_f32", _) => arrays::filled(name, args, frame, arena),
        ("assert", [condition]) => match frame.get(*condition) {
            Ok(Scalar::Bool(true)) => Ok(Scalar::Unit),
            Ok(Scalar::Bool(false)) => Err(OracleError::from(EvalError::Trap {
                category: TrapCategory::AssertionFailed,
                span,
            })),
            Ok(_) => Err(internal()),
            Err(error) => Err(error),
        },
        ("assert", _) => Err(internal()),
        _ => return None,
    })
}
