//! Iterative checking of left-associative binary chains.

use super::*;

pub(super) fn check_binary_chain(
    expr: &mut Expr,
    scope: &HashMap<String, PhysicalType>,
    context: &PhysicalContext,
    expected: Option<&PhysicalType>,
) -> Result<PhysicalType, FrontError> {
    struct Step {
        op: BinaryOp,
        op_span: Span,
        span: Span,
        right: Expr,
        expected: Option<PhysicalType>,
    }

    // The parser bounds tree height, but a left-associative chain still needs
    // iterative checking to fit its documented 2 MiB checker stack.
    let placeholder = Expr::new(ExprKind::Bool(false), expr.span)?;
    let mut current = std::mem::replace(expr, placeholder);
    let mut steps = Vec::new();
    let mut hint = expected.cloned();
    loop {
        let Expr { kind, span, .. } = current;
        match kind {
            ExprKind::Binary {
                left,
                op,
                op_span,
                right,
            } => {
                let left_hint = if matches!(op, BinaryOp::Add | BinaryOp::Sub)
                    && hint.as_ref().is_none_or(|ty| ty.frame.is_none())
                {
                    hint.clone()
                } else {
                    None
                };
                steps.push(Step {
                    op,
                    op_span,
                    span,
                    right: *right,
                    expected: hint,
                });
                current = *left;
                hint = left_hint;
            }
            kind => {
                current = Expr::new(kind, span)?;
                break;
            }
        }
    }
    let mut left_ty = check_expr(&mut current, scope, context, hint.as_ref())?;
    for step in steps.into_iter().rev() {
        let mut right = step.right;
        let right_hint = if left_ty.unit.is_some()
            || matches!(step.op, BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem)
        {
            None
        } else {
            Some(&left_ty)
        };
        let right_ty = check_expr(&mut right, scope, context, right_hint)?;
        let right_span = right.span;
        let involved = left_ty.unit.is_some() || right_ty.unit.is_some();
        if left_ty.frame.is_some() || right_ty.frame.is_some() {
            let (helper, result) = frame_binary(step.op, &left_ty, &right_ty, step.span)?;
            current = Expr::new(
                ExprKind::Call {
                    name: helper.to_owned(),
                    args: vec![current, right],
                },
                step.span,
            )?;
            context.frame_helpers_used.set(true);
            left_ty = result;
        } else {
            left_ty = match step.op {
                BinaryOp::Add
                | BinaryOp::Sub
                | BinaryOp::Eq
                | BinaryOp::Ne
                | BinaryOp::Lt
                | BinaryOp::Le
                | BinaryOp::Gt
                | BinaryOp::Ge => {
                    if involved && left_ty != right_ty {
                        let mut error = FrontError::source(
                            "UNIT_MISMATCH",
                            "quantity unit or scalar does not match; use an explicit conversion",
                            step.span,
                        );
                        if let (Some(left_unit), Some(right_unit)) = (left_ty.unit, right_ty.unit)
                            && left_ty.scalar == right_ty.scalar
                            && left_unit.dimension == right_unit.dimension
                            && let Some(target) = left_unit.builtin_name()
                        {
                            error = error.with_unit_conversion_fix(right_span, target);
                        }
                        return Err(error);
                    }
                    if matches!(
                        step.op,
                        BinaryOp::Eq
                            | BinaryOp::Ne
                            | BinaryOp::Lt
                            | BinaryOp::Le
                            | BinaryOp::Gt
                            | BinaryOp::Ge
                    ) {
                        PhysicalType::plain(Type::Bool)
                    } else {
                        left_ty
                    }
                }
                BinaryOp::Mul | BinaryOp::Div if involved => {
                    if left_ty.scalar != right_ty.scalar {
                        return Err(mismatch(
                            "quantity arithmetic requires matching numeric scalars",
                            step.span,
                        ));
                    }
                    let lhs = left_ty.unit.unwrap_or(Unit::builtin("one").unwrap());
                    let rhs = right_ty.unit.unwrap_or(Unit::builtin("one").unwrap());
                    let unit = if step.op == BinaryOp::Mul {
                        lhs.mul(rhs)
                    } else {
                        lhs.div(rhs)
                    }
                    .ok_or_else(|| {
                        FrontError::source(
                            "RESOURCE_LIMIT",
                            "unit dimension or scale exceeds its bound",
                            step.span,
                        )
                    })?;
                    PhysicalType {
                        scalar: left_ty.scalar,
                        unit: Some(unit),
                        frame: None,
                    }
                }
                BinaryOp::Rem if involved => {
                    return Err(mismatch("remainder has no quantity operation", step.span));
                }
                _ => PhysicalType::plain(Type::I32),
            };
            current = Expr::new(
                ExprKind::Binary {
                    left: Box::new(current),
                    op: step.op,
                    op_span: step.op_span,
                    right: Box::new(right),
                },
                step.span,
            )?;
        }
        if let Some(expected) = &step.expected {
            left_ty.clone().require(expected, step.span)?;
        }
    }
    *expr = current;
    Ok(left_ty)
}
