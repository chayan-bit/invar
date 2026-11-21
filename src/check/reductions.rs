//! v0.2 deterministic float reductions: `sum_tree`, `dot_tree`, `sum_exact`
//! and `dot_exact`. Each argument read-borrows a bare named float array for
//! the call; the result is the array's element type. Also `min_of`, `max_of`,
//! `argmin` and `argmax` (#207), which read-borrow one named `i32`, `u32`,
//! `f32` or `f64` array and return its element type or the `u32` index.

use super::*;

/// The element types a reduction admits. `f64` joins once lane `f64math`
/// adds `Array<f64>`; the result type is always the element type.
fn float_element(ty: &Type) -> Option<Type> {
    match ty {
        Type::ArrayF32 => Some(Type::F32),
        _ => None,
    }
}

/// A float reduction or an extremum intrinsic.
pub(crate) fn is_reduction(name: &str) -> bool {
    crate::fsum::Kind::from_name(name).is_some() || extremum_kind(name).is_some()
}

impl FunctionChecker<'_> {
    pub(super) fn reduction(
        &mut self,
        name: &str,
        args: &[Expr],
        span: Span,
    ) -> Result<Type, FrontError> {
        if extremum_kind(name).is_some() {
            return self.extremum(name, args, span);
        }
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

pub(crate) fn extremum_kind(name: &str) -> Option<crate::extremum::Kind> {
    crate::extremum::Kind::from_name(name)
}

impl FunctionChecker<'_> {
    /// The expected result type of a reduction or extremum call.
    pub(super) fn reduction_hint(&self, name: &str, args: &[Expr]) -> Option<Type> {
        if extremum_kind(name).is_some_and(crate::extremum::Kind::is_index) {
            return Some(Type::U32);
        }
        args.first()
            .and_then(|arg| self.hint(arg))
            .and_then(|ty| element(&ty))
    }

    fn extremum(&mut self, name: &str, args: &[Expr], span: Span) -> Result<Type, FrontError> {
        let [arg] = args else {
            return Err(mismatch(format!("{name} expects one named array"), span));
        };
        if !matches!(arg.kind, ExprKind::Name(_)) {
            return Err(mismatch(format!("{name} borrows a named array"), arg.span));
        }
        let ty = self.expr(arg, None)?;
        let item = match ty {
            Type::ArrayI32 => Type::I32,
            Type::ArrayU32 => Type::U32,
            Type::ArrayF32 => Type::F32,
            Type::ArrayF64 => Type::F64,
            _ => {
                return Err(mismatch(
                    format!("{name} expects Array<i32>, Array<u32>, Array<f32> or Array<f64>"),
                    arg.span,
                ));
            }
        };
        let is_index = extremum_kind(name).is_some_and(crate::extremum::Kind::is_index);
        Ok(if is_index { Type::U32 } else { item })
    }
}
