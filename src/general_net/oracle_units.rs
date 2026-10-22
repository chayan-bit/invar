//! Checked unit-scale and angle conversion rules for the CPU net oracle.
//! Scale is exact rational arithmetic implemented here; angle conversion reuses
//! the shared certified-interval primitive (`crate::angle`), the same arithmetic
//! module every engine embeds, not either evaluator.

use super::{OracleError, Scalar, internal};
use crate::{
    eval::{EvalError, TrapCategory},
    front::Span,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Rule {
    /// `$scale$<numerator|denominator>`, both positive and checked by the frontend.
    Scale(i64, i64),
    /// `$angle$<d2r>` (true) or `$angle$<r2d>` (false).
    Angle(bool),
}

pub(super) fn rule(name: &str) -> Option<Rule> {
    crate::check::scale_call(name)
        .map(|(numerator, denominator)| Rule::Scale(numerator, denominator))
        .or_else(|| crate::check::angle_call(name).map(Rule::Angle))
}

pub(super) fn call(rule: Rule, value: Scalar, span: Span) -> Result<Scalar, OracleError> {
    let trap = |category| OracleError::from(EvalError::Trap { category, span });
    let (input, signed) = match value {
        Scalar::I32(x) => (i64::from(x), true),
        Scalar::U32(x) => (i64::from(x), false),
        _ => return Err(internal()),
    };
    let output = match rule {
        Rule::Scale(numerator, denominator) => {
            if numerator <= 0 || denominator <= 0 {
                return Err(internal());
            }
            // |input| < 2^32 and numerator <= 10^9: the product fits in i128.
            let product = i128::from(input) * i128::from(numerator);
            if product % i128::from(denominator) != 0 {
                return Err(trap(TrapCategory::UnitConversionInexact));
            }
            product / i128::from(denominator)
        }
        Rule::Angle(degrees_to_radians) => {
            match crate::angle::convert(input, signed, degrees_to_radians) {
                Ok(value) => i128::from(value),
                Err(crate::angle::AngleError::Overflow) => {
                    return Err(trap(TrapCategory::IntegerOverflow));
                }
                Err(crate::angle::AngleError::Ambiguous) => return Err(internal()),
            }
        }
    };
    let narrowed = if signed {
        i32::try_from(output).ok().map(Scalar::I32)
    } else {
        u32::try_from(output).ok().map(Scalar::U32)
    };
    narrowed.ok_or_else(|| trap(TrapCategory::IntegerOverflow))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span() -> Span {
        Span { start: 3, end: 9 }
    }

    fn trapped(result: Result<Scalar, OracleError>) -> Option<TrapCategory> {
        match result {
            Err(OracleError::Semantic(EvalError::Trap { category, span: s })) if s == span() => {
                Some(category)
            }
            _ => None,
        }
    }

    #[test]
    fn issue_gnet_02_unit_scale_is_exact_inexact_or_overflowing() {
        assert!(matches!(
            call(Rule::Scale(1000, 1), Scalar::I32(-7), span()),
            Ok(Scalar::I32(-7000))
        ));
        assert!(matches!(
            call(Rule::Scale(1, 1000), Scalar::U32(4000), span()),
            Ok(Scalar::U32(4))
        ));
        assert_eq!(
            trapped(call(Rule::Scale(1, 1000), Scalar::U32(4001), span())),
            Some(TrapCategory::UnitConversionInexact)
        );
        assert_eq!(
            trapped(call(Rule::Scale(1000, 1), Scalar::I32(i32::MAX), span())),
            Some(TrapCategory::IntegerOverflow)
        );
        assert_eq!(
            trapped(call(Rule::Scale(1000, 1), Scalar::U32(u32::MAX), span())),
            Some(TrapCategory::IntegerOverflow)
        );
        assert!(call(Rule::Scale(0, 1), Scalar::I32(1), span()).is_err());
        assert!(call(Rule::Scale(1, 1), Scalar::Bool(true), span()).is_err());
    }

    #[test]
    fn issue_gnet_02_angle_rules_are_selected_by_name_only() {
        assert_eq!(rule("$angle$<d2r>"), Some(Rule::Angle(true)));
        assert_eq!(rule("$angle$<r2d>"), Some(Rule::Angle(false)));
        assert_eq!(rule("$scale$<1000|1>"), Some(Rule::Scale(1000, 1)));
        assert_eq!(rule("$scale$<0|1>"), None);
        assert_eq!(rule("wrapping_add"), None);
    }
}
