//! v0.2 deterministic float reductions: `sum_tree`, `dot_tree`, `sum_exact`
//! and `dot_exact`. Each argument read-borrows a bare named float array for
//! the call; the result is the array's element type.

use super::*;

/// The element types a reduction admits. `f64` joins once lane `f64math`
/// adds `Array<f64>`; the result type is always the element type.
fn float_element(ty: &Type) -> Option<Type> {
    match ty {
        Type::ArrayF32 => Some(Type::F32),
        _ => None,
    }
}

pub(crate) fn is_reduction(name: &str) -> bool {
    crate::fsum::Kind::from_name(name).is_some()
}

impl FunctionChecker<'_> {
    pub(super) fn reduction(
        &mut self,
        name: &str,
        args: &[Expr],
        span: Span,
    ) -> Result<Type, FrontError> {
        let arity = if name.starts_with("dot_") { 2 } else { 1 };
        if args.len() != arity {
            return Err(mismatch(
                format!("{name} expects {arity} named float array argument(s)"),
                span,
            ));
        }
        let mut element = None;
        for arg in args {
            if !matches!(arg.kind, ExprKind::Name(_)) {
                return Err(mismatch(
                    format!("{name} borrows a named float array"),
                    arg.span,
                ));
            }
            let ty = self.expr(arg, None)?;
            let item = float_element(&ty)
                .ok_or_else(|| mismatch(format!("{name} expects Array<f32>"), arg.span))?;
            if element.as_ref().is_some_and(|first| *first != item) {
                return Err(mismatch(
                    format!("{name} operands must share one element type"),
                    arg.span,
                ));
            }
            element = Some(item);
        }
        element.ok_or_else(|| mismatch("reduction without operands", span))
    }
}
