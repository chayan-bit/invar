//! The reserved intrinsic names and the `assert` rule (#258).

use super::*;

pub(crate) fn intrinsic(name: &str) -> bool {
    crate::spatial::result(name).is_some()
        || sort_modes(name).is_some()
        || matches!(
            name,
            "len"
                | "assert"
                | "view"
                | "view_mut"
                | "fma"
                | "sin"
                | "cos"
                | "sqrt"
                | "tc_mul_add"
                | "wrapping_add"
                | "wrapping_mul"
                | "bytes_slice"
                | "utf8_slice"
                | "utf8_decode"
                | "utf8_encode"
                | "quantity"
                | "to"
                | "sum_tree"
                | "dot_tree"
                | "sum_exact"
                | "dot_exact"
                | "reassoc_add"
        )
        || conversion(name).is_some()
        || storage::filled(name).is_some()
        || crate::math::MathFn::from_name(name).is_some()
}

/// Call-scoped loans for the reserved sort intrinsics, in argument order.
pub(crate) fn sort_modes(name: &str) -> Option<&'static [Mode]> {
    match name {
        "sort" => Some(&[Mode::Mut]),
        "sort_by_key" => Some(&[Mode::Mut, Mode::Mut]),
        "argsort" => Some(&[Mode::Read]),
        "segmented_sort" => Some(&[Mode::Mut, Mode::Read]),
        _ => None,
    }
}

impl FunctionChecker<'_> {
    /// `assert(condition, "message")`: a `Bool` condition and a string-literal
    /// message, which every engine reports from the call span; `Unit`.
    pub(super) fn assert_call(&mut self, args: &[Expr], span: Span) -> Result<Type, FrontError> {
        let [condition, message] = args else {
            return Err(mismatch("assert expects a condition and a message", span));
        };
        if self.expr(condition, Some(&Type::Bool))? != Type::Bool {
            return Err(mismatch("assert condition must be Bool", condition.span));
        }
        if !matches!(message.kind, ExprKind::Text(_)) {
            return Err(mismatch(
                "assert message must be a string literal",
                message.span,
            ));
        }
        self.expr(message, None)?;
        Ok(Type::Unit)
    }
}

/// Fixed-signature float conversions: (input, result).
pub(crate) fn conversion(name: &str) -> Option<(Type, Type)> {
    Some(match name {
        "f32_from_i32" => (Type::I32, Type::F32),
        "f32_from_u32" => (Type::U32, Type::F32),
        "f64_from_i32" => (Type::I32, Type::F64),
        "f64_from_u32" => (Type::U32, Type::F64),
        "f64_from_f32" => (Type::F32, Type::F64),
        "f32_from_f64" => (Type::F64, Type::F32),
        "i32_from_f32" => (Type::F32, Type::I32),
        "u32_from_f32" => (Type::F32, Type::U32),
        "i32_from_f64" => (Type::F64, Type::I32),
        "u32_from_f64" => (Type::F64, Type::U32),
        _ => return storage::store(name),
    })
}
