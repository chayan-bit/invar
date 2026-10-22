//! The type a numeric literal's suffix names.

use super::*;

pub(super) fn literal_type(raw: &str) -> Option<Type> {
    match crate::numeric_contract::decimal_digits(raw).1 {
        Some("f32") => Some(Type::F32),
        Some("f64") => Some(Type::F64),
        _ => None,
    }
}

pub(super) fn split_literal(raw: &str) -> (&str, Option<Type>) {
    if let Some(digits) = raw.strip_suffix("i32") {
        (digits, Some(Type::I32))
    } else if let Some(digits) = raw.strip_suffix("u32") {
        (digits, Some(Type::U32))
    } else {
        (raw, None)
    }
}
