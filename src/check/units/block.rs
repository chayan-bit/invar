//! Statement and block checking with erased unit tags.

use super::*;

pub(super) fn check_block(
    block: &mut Block,
    scope: &mut HashMap<String, PhysicalType>,
    result: &PhysicalType,
    context: &PhysicalContext,
) -> Result<(), FrontError> {
    let mut local = scope.clone();
    for stmt in &mut block.statements {
        match &mut stmt.kind {
            StmtKind::Binding {
                name,
                annotation,
                value,
                ..
            } => {
                let expected = annotation
                    .as_ref()
                    .map(|ty| PhysicalType::from_source(ty, stmt.span, &context.frames))
                    .transpose()?;
                let actual = check_expr(value, &local, context, expected.as_ref())?;
                if let Some(expected) = &expected {
                    actual.clone().require(expected, value.span)?;
                }
                local.insert(name.clone(), expected.clone().unwrap_or(actual));
                if let Some(expected) = expected {
                    *annotation = Some(expected.scalar);
                }
            }
            StmtKind::Assign { place, value } => {
                if let Place::Index { index, .. } = place
                    && {
                        let index_ty = check_expr(index, &local, context, None)?;
                        index_ty.unit.is_some() || index_ty.frame.is_some()
                    }
                {
                    return Err(FrontError::source(
                        "UNIT_MISMATCH",
                        "array index cannot be a quantity or geometric value",
                        index.span,
                    ));
                }
                let expected = match place {
                    Place::Name(name, _) | Place::Index { name, .. } => local.get(name),
                };
                let actual = check_expr(value, &local, context, expected)?;
                if let Some(expected) = expected {
                    actual.require(expected, value.span)?;
                }
            }
            StmtKind::Return(value) => {
                if let Some(value) = value {
                    check_expr(value, &local, context, Some(result))?
                        .require(result, value.span)?;
                }
            }
            StmtKind::If {
                condition,
                then_block,
                else_arm,
            } => {
                check_expr(condition, &local, context, None)?;
                check_block(then_block, &mut local.clone(), result, context)?;
                if let Some(arm) = else_arm {
                    check_else(arm, &local, result, context)?;
                }
            }
            StmtKind::While { condition, body } => {
                check_expr(condition, &local, context, None)?;
                check_block(body, &mut local.clone(), result, context)?;
            }
            StmtKind::For {
                name,
                lower,
                upper,
                body,
            } => {
                let lower_ty = check_expr(lower, &local, context, None)?;
                let upper_ty = check_expr(upper, &local, context, None)?;
                if lower_ty.unit.is_some()
                    || lower_ty.frame.is_some()
                    || upper_ty.unit.is_some()
                    || upper_ty.frame.is_some()
                {
                    return Err(FrontError::source(
                        "UNIT_MISMATCH",
                        "loop bounds cannot be quantities or geometric values",
                        stmt.span,
                    ));
                }
                let mut loop_scope = local.clone();
                loop_scope.insert(name.clone(), PhysicalType::plain(Type::I32));
                check_block(body, &mut loop_scope, result, context)?;
            }
            StmtKind::Expr(value) => {
                check_expr(value, &local, context, None)?;
            }
        }
    }
    Ok(())
}

fn check_else(
    arm: &mut ElseArm,
    scope: &HashMap<String, PhysicalType>,
    result: &PhysicalType,
    context: &PhysicalContext,
) -> Result<(), FrontError> {
    match arm {
        ElseArm::Block(block) => check_block(block, &mut scope.clone(), result, context),
        ElseArm::If(stmt) => {
            let StmtKind::If {
                condition,
                then_block,
                else_arm,
            } = &mut stmt.kind
            else {
                return Ok(());
            };
            check_expr(condition, scope, context, None)?;
            check_block(then_block, &mut scope.clone(), result, context)?;
            if let Some(arm) = else_arm {
                check_else(arm, scope, result, context)?;
            }
            Ok(())
        }
    }
}
