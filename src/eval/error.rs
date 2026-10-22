//! Evaluator errors, trap categories, cleanup events and checked unit conversions.

use super::*;

#[derive(Debug, PartialEq, Eq)]
pub struct CleanupEvent {
    pub name: String,
    pub length: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrapCategory {
    IntegerOverflow,
    DivisionByZero,
    IndexOutOfBounds,
    UnitConversionInexact,
}

impl TrapCategory {
    fn code(self) -> &'static str {
        match self {
            Self::IntegerOverflow => "INTEGER_OVERFLOW",
            Self::DivisionByZero => "DIVISION_BY_ZERO",
            Self::IndexOutOfBounds => "INDEX_OUT_OF_BOUNDS",
            Self::UnitConversionInexact => "UNIT_CONVERSION_INEXACT",
        }
    }
}

pub(crate) fn checked_scale(
    value: Value,
    numerator: i64,
    denominator: i64,
    span: Span,
) -> Result<Value, EvalError> {
    let (raw, signed) = match value {
        Value::I32(value) => (i128::from(value), true),
        Value::U32(value) => (i128::from(value), false),
        _ => {
            return Err(EvalError::tool(
                "INTERNAL_TYPE",
                "checked scale input is not integer",
            ));
        }
    };
    let product = raw
        .checked_mul(i128::from(numerator))
        .ok_or(EvalError::Trap {
            category: TrapCategory::IntegerOverflow,
            span,
        })?;
    if denominator <= 0 {
        return Err(EvalError::tool(
            "INTERNAL_TYPE",
            "invalid checked scale denominator",
        ));
    }
    if product % i128::from(denominator) != 0 {
        return Err(EvalError::Trap {
            category: TrapCategory::UnitConversionInexact,
            span,
        });
    }
    let result = product / i128::from(denominator);
    if signed {
        i32::try_from(result).map(Value::I32)
    } else {
        u32::try_from(result).map(Value::U32)
    }
    .map_err(|_| EvalError::Trap {
        category: TrapCategory::IntegerOverflow,
        span,
    })
}

pub(crate) fn checked_angle(
    value: Value,
    degrees_to_radians: bool,
    span: Span,
) -> Result<Value, EvalError> {
    let (input, signed) = match value {
        Value::I32(value) => (i64::from(value), true),
        Value::U32(value) => (i64::from(value), false),
        _ => {
            return Err(EvalError::tool(
                "INTERNAL_TYPE",
                "checked angle input is not integer",
            ));
        }
    };
    let output =
        crate::angle::convert(input, signed, degrees_to_radians).map_err(|error| match error {
            crate::angle::AngleError::Overflow => EvalError::Trap {
                category: TrapCategory::IntegerOverflow,
                span,
            },
            crate::angle::AngleError::Ambiguous => {
                EvalError::tool("INTERNAL_TYPE", "certified angle interval did not resolve")
            }
        })?;
    if signed {
        i32::try_from(output).map(Value::I32)
    } else {
        u32::try_from(output).map(Value::U32)
    }
    .map_err(|_| EvalError::Trap {
        category: TrapCategory::IntegerOverflow,
        span,
    })
}

#[derive(Debug, PartialEq, Eq)]
pub enum EvalError {
    Trap { category: TrapCategory, span: Span },
    EvaluationLimit { fuel_limit: u64, steps_used: u64 },
    Tool { code: &'static str, message: String },
}

impl EvalError {
    pub(super) fn tool(code: &'static str, message: impl Into<String>) -> Self {
        Self::Tool {
            code,
            message: message.into(),
        }
    }

    pub fn to_json(&self, path: &str) -> JsonValue {
        match self {
            Self::Trap { category, span } => json!({
                "schema_version": if *category == TrapCategory::UnitConversionInexact { 2 } else { 1 },
                "status": "trap",
                "diagnostics": [],
                "trap": {"category": category.code(), "span": {"start": span.start, "end": span.end}}
            }),
            Self::EvaluationLimit {
                fuel_limit,
                steps_used,
            } => json!({
                "schema_version": 1,
                "status": "evaluation_limit",
                "diagnostics": [],
                "fuel_limit": fuel_limit,
                "steps_used": steps_used
            }),
            Self::Tool { code, message } => json!({
                "schema_version": 1,
                "status": "tool_error",
                "diagnostics": [{"code": code, "message": message, "path": path, "span": null, "notes": []}]
            }),
        }
    }

    /// CLI representation; preserves the established status and exit codes.
    pub fn to_diagnostics_json(&self, path: &str, source: &[u8]) -> JsonValue {
        match self {
            Self::Trap { category, span } => {
                let mut document = crate::diagnostics::document(
                    "trap",
                    vec![crate::diagnostics::diagnostic(
                        category.code(),
                        "evaluation trapped",
                        path,
                        source,
                        Some(*span),
                    )],
                );
                document["trap"] = json!({"category": category.code(), "span": {"start": span.start, "end": span.end}});
                if *category == TrapCategory::UnitConversionInexact {
                    document["schema"] = json!("invar-diagnostics-v2");
                    document["schema_version"] = json!(2);
                }
                document
            }
            Self::EvaluationLimit {
                fuel_limit,
                steps_used,
            } => {
                let mut document = crate::diagnostics::document(
                    "evaluation_limit",
                    vec![crate::diagnostics::diagnostic(
                        "EVALUATION_LIMIT",
                        "evaluation fuel limit exceeded",
                        path,
                        source,
                        None,
                    )],
                );
                document["fuel_limit"] = json!(fuel_limit);
                document["steps_used"] = json!(steps_used);
                document
            }
            Self::Tool { code, message } => crate::diagnostics::document(
                "tool_error",
                vec![crate::diagnostics::diagnostic(
                    code, message, path, source, None,
                )],
            ),
        }
    }
}
