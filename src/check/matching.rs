//! Checking `match`: arm types, payload bindings, guards (#260) and exhaustiveness,
//! and the `if` expression (#361), whose two arms return the same scalar subset.

use super::*;

impl FunctionChecker<'_> {
    pub(super) fn match_expr(
        &mut self,
        value: &Expr,
        arms: &[MatchArm],
        expected: Option<&Type>,
        span: Span,
    ) -> Result<Type, FrontError> {
        let Type::Named(nominal) = self.expr(value, None)? else {
            return Err(mismatch("match scrutinee must be a variant", value.span));
        };
        let declared = self
            .checker
            .variants
            .get(&nominal)
            .cloned()
            .ok_or_else(|| mismatch("match scrutinee must be a variant", value.span))?;
        let first = arms
            .first()
            .ok_or_else(|| mismatch("match needs arms", span))?;
        let result_ty = expected
            .cloned()
            .or_else(|| self.hint(&first.value))
            .unwrap_or(Type::I32);
        scalar_result(&result_ty, "this match subset returns a scalar", span)?;
        let mut seen = HashSet::new();
        for arm in arms {
            let (prefix, case) = arm.variant.rsplit_once("::").ok_or_else(|| {
                FrontError::source(
                    "NAME_UNKNOWN",
                    "match arm needs a qualified variant",
                    arm.span,
                )
            })?;
            if prefix != nominal
                && !(crate::front::result_parts(&nominal).is_some() && prefix == "Result")
            {
                return Err(mismatch("match arm has the wrong variant type", arm.span));
            }
            let payload = declared
                .iter()
                .find(|(name, _)| name == case)
                .ok_or_else(|| {
                    FrontError::source(
                        "NAME_UNKNOWN",
                        format!("unknown variant arm '{}'", arm.variant),
                        arm.span,
                    )
                })?
                .1
                .clone();
            // A guarded arm never covers its case, and no arm may follow its case's unguarded arm.
            if seen.contains(case) || (arm.guard.is_none() && !seen.insert(case)) {
                return Err(FrontError::source(
                    "NAME_DUPLICATE",
                    format!("duplicate match arm '{case}'"),
                    arm.span,
                ));
            }
            if payload.is_some() != arm.binding.is_some() {
                return Err(mismatch(
                    "match payload binding does not match variant arm",
                    arm.span,
                ));
            }
            if payload
                .as_ref()
                .is_some_and(|ty| matches!(ty, Type::Named(_)))
                && !matches!(value.kind, ExprKind::Move(_))
            {
                return Err(FrontError::source(
                    "OWN_MOVE_REQUIRED",
                    "owned pattern payload requires match move",
                    arm.span,
                ));
            }
            if let Some(guard) = &arm.guard
                && payload
                    .as_ref()
                    .is_some_and(|ty| matches!(ty, Type::Named(_)))
            {
                return Err(FrontError::source(
                    "UNSUPPORTED_FEATURE",
                    "a match guard needs a scalar payload",
                    guard.span,
                ));
            }
            self.scopes.push(HashMap::new());
            if let (Some(binding), Some(ty)) = (&arm.binding, payload) {
                self.declare(
                    binding,
                    Binding {
                        ty,
                        mutable: false,
                        mode: None,
                    },
                    arm.span,
                )?;
            }
            if let Some(guard) = &arm.guard
                && self.expr(guard, Some(&Type::Bool))? != Type::Bool
            {
                return Err(mismatch("a match guard must be Bool", guard.span));
            }
            self.expr(&arm.value, Some(&result_ty))?;
            self.scopes.pop();
        }
        if seen.len() != declared.len() {
            return Err(FrontError::source(
                "MATCH_NONEXHAUSTIVE",
                "match does not cover every variant arm",
                span,
            ));
        }
        Ok(result_ty)
    }

    /// The type the arms of a `match` or an `if` expression suggest.
    pub(super) fn branch_hint(&self, expr: &Expr) -> Option<Type> {
        match &expr.kind {
            ExprKind::Match { arms, .. } => arms.first().and_then(|arm| self.hint(&arm.value)),
            ExprKind::If {
                then, otherwise, ..
            } => self.hint(then).or_else(|| self.hint(otherwise)),
            _ => None,
        }
    }

    pub(super) fn branch_expr(
        &mut self,
        expr: &Expr,
        expected: Option<&Type>,
    ) -> Result<Type, FrontError> {
        match &expr.kind {
            ExprKind::Match { value, arms } => self.match_expr(value, arms, expected, expr.span),
            ExprKind::If {
                condition,
                then,
                otherwise,
            } => self.if_expr(condition, [then, otherwise], expected, expr.span),
            _ => Err(mismatch("match or if expression required", expr.span)),
        }
    }

    /// `if c { a } else { b }` (#361): a `Bool` condition and two arms of one scalar
    /// type, taken from the expected type, else from the first arm whose type is known.
    fn if_expr(
        &mut self,
        condition: &Expr,
        arms: [&Expr; 2],
        expected: Option<&Type>,
        span: Span,
    ) -> Result<Type, FrontError> {
        self.expr(condition, Some(&Type::Bool))?;
        let result_ty = expected
            .cloned()
            .or_else(|| arms.iter().find_map(|arm| self.hint(arm)))
            .unwrap_or(Type::I32);
        scalar_result(&result_ty, "an if expression returns a scalar", span)?;
        for arm in arms {
            self.expr(arm, Some(&result_ty))?;
        }
        Ok(result_ty)
    }
}

fn scalar_result(ty: &Type, message: &str, span: Span) -> Result<(), FrontError> {
    if matches!(
        ty,
        Type::I32 | Type::U32 | Type::F32 | Type::F64 | Type::Bool
    ) {
        Ok(())
    } else {
        Err(FrontError::source("UNSUPPORTED_FEATURE", message, span))
    }
}
