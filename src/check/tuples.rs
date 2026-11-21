//! Pairs of 32-bit scalars (#262): the literal's type and positional fields.

use super::*;

impl FunctionChecker<'_> {
    /// `(first, second)`, each element taking the expected pair's element as a hint.
    pub(super) fn pair(
        &mut self,
        first: &Expr,
        second: &Expr,
        expected: Option<&Type>,
        span: Span,
    ) -> Result<Type, FrontError> {
        let (left, right) = match expected {
            Some(Type::Pair(pair)) => (Some(&pair.0), Some(&pair.1)),
            _ => (None, None),
        };
        let pair = (self.expr(first, left)?, self.expr(second, right)?);
        if !crate::front::pair_element(&pair.0) || !crate::front::pair_element(&pair.1) {
            return Err(crate::front::tuple_shape(span));
        }
        Ok(Type::Pair(Box::new(pair)))
    }

    /// `base.0` or `base.1` of a pair-typed `base`.
    pub(super) fn pair_field(
        &mut self,
        base: &Expr,
        name: &str,
        span: Span,
    ) -> Result<Type, FrontError> {
        let Type::Pair(pair) = self.expr(base, None)? else {
            return Err(mismatch("positional field requires a pair", base.span));
        };
        match name {
            "0" => Ok(pair.0),
            "1" => Ok(pair.1),
            _ => Err(crate::front::tuple_shape(span)),
        }
    }

    /// Whether `base` is known, without checking it, to have a pair type.
    pub(super) fn is_pair(&self, base: &Expr) -> bool {
        matches!(self.hint(base), Some(Type::Pair(_)))
    }

    /// The type of `base.name` for a pair-typed `base`, if known without checking.
    pub(super) fn pair_field_hint(&self, base: &Expr, name: &str) -> Option<Type> {
        let Some(Type::Pair(pair)) = self.hint(base) else {
            return None;
        };
        match name {
            "0" => Some(pair.0),
            "1" => Some(pair.1),
            _ => None,
        }
    }
}
