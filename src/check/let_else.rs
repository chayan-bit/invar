//! Completes the matches the parser emits for `let V::P(x) = s else { ... };` (#260,
//! `src/front/parser/let_else.rs`) once modules are resolved and every variant, `Result`
//! instances included, is declared: the check `match s { V::P(..) => true }` gains
//! `V::Q(..) => false` for every other arm, and the extraction
//! `let x = match s { V::P(v) => v }` gains a typed zero for every other arm, never
//! evaluated because the else block diverges, and the payload type as `x`'s annotation.

use super::*;
use crate::front::VariantDecl;

pub(super) fn complete(program: &mut Program) -> Result<(), FrontError> {
    let variants = program.variants.clone();
    let signatures: HashMap<String, Type> = program
        .functions
        .iter()
        .map(|f| (f.name.clone(), f.result.clone()))
        .collect();
    for function in &mut program.functions {
        let owners = owner_types(function, &signatures);
        complete_block(&mut function.body, &variants, &owners)?;
    }
    Ok(())
}

/// The declared type of every parameter and annotated or call-initialized binding; names
/// are unique within a function, so no scoping is needed.
fn owner_types(function: &Function, signatures: &HashMap<String, Type>) -> HashMap<String, Type> {
    let mut types: HashMap<String, Type> = function
        .params
        .iter()
        .map(|p| (p.name.clone(), p.ty.clone()))
        .collect();
    let mut pending = vec![&function.body];
    while let Some(block) = pending.pop() {
        for stmt in &block.statements {
            match &stmt.kind {
                StmtKind::Binding {
                    name,
                    annotation,
                    value,
                    ..
                } => {
                    let ty = annotation.clone().or_else(|| match &value.kind {
                        ExprKind::Call { name, .. } => signatures.get(name).cloned(),
                        ExprKind::Move(source) => types.get(source).cloned(),
                        _ => None,
                    });
                    types.extend(ty.map(|ty| (name.clone(), ty)));
                }
                StmtKind::If {
                    then_block,
                    else_arm,
                    ..
                } => {
                    pending.push(then_block);
                    let mut arm = else_arm.as_ref();
                    while let Some(next) = arm {
                        match next {
                            ElseArm::Block(block) => {
                                pending.push(block);
                                arm = None;
                            }
                            ElseArm::If(stmt) => match &stmt.kind {
                                StmtKind::If {
                                    then_block,
                                    else_arm,
                                    ..
                                } => {
                                    pending.push(then_block);
                                    arm = else_arm.as_ref();
                                }
                                _ => arm = None,
                            },
                        }
                    }
                }
                StmtKind::While { body, .. } | StmtKind::For { body, .. } => pending.push(body),
                _ => {}
            }
        }
    }
    types
}

fn complete_block(
    block: &mut Block,
    variants: &[VariantDecl],
    owners: &HashMap<String, Type>,
) -> Result<(), FrontError> {
    for index in 0..block.statements.len() {
        let (head, tail) = block.statements.split_at_mut(index + 1);
        let stmt = &mut head[index];
        match &mut stmt.kind {
            StmtKind::Binding { name, value, .. } if name.starts_with("else@") => {
                let hidden = name.clone();
                let extraction = tail.get_mut(1).filter(|next| extraction_of(next, &hidden));
                complete_let_else(value, extraction, &hidden, variants, owners)?;
            }
            StmtKind::If {
                then_block,
                else_arm,
                ..
            } => {
                complete_block(then_block, variants, owners)?;
                let mut arm = else_arm.as_mut();
                while let Some(next) = arm {
                    arm = match next {
                        ElseArm::Block(block) => {
                            complete_block(block, variants, owners)?;
                            None
                        }
                        ElseArm::If(stmt) => match &mut stmt.kind {
                            StmtKind::If {
                                then_block,
                                else_arm,
                                ..
                            } => {
                                complete_block(then_block, variants, owners)?;
                                else_arm.as_mut()
                            }
                            _ => None,
                        },
                    };
                }
            }
            StmtKind::While { body, .. } | StmtKind::For { body, .. } => {
                complete_block(body, variants, owners)?
            }
            _ => {}
        }
    }
    Ok(())
}

fn extraction_of(stmt: &Stmt, hidden: &str) -> bool {
    let payload = format!("{hidden}@x0");
    matches!(&stmt.kind, StmtKind::Binding { value: Expr { kind: ExprKind::Match { arms, .. }, .. }, .. }
        if arms.len() == 1 && arms[0].binding.as_deref() == Some(payload.as_str()))
}

fn complete_let_else(
    check: &mut Expr,
    extraction: Option<&mut Stmt>,
    hidden: &str,
    variants: &[VariantDecl],
    owners: &HashMap<String, Type>,
) -> Result<(), FrontError> {
    let ExprKind::Match { value, arms } = &mut check.kind else {
        return Ok(());
    };
    let (Some(first), 1) = (arms.first().cloned(), arms.len()) else {
        return Ok(());
    };
    let Some((prefix, case)) = first.variant.rsplit_once("::") else {
        return Ok(());
    };
    let declaration = if prefix == "Result" {
        let ExprKind::Name(owner) = &value.kind else {
            return Ok(());
        };
        match owners.get(owner) {
            Some(Type::Named(nominal)) => variants.iter().find(|v| v.name == *nominal),
            _ => {
                return Err(FrontError::source(
                    "TYPE_MISMATCH",
                    "a let-else over Result needs an owner with a declared Result type",
                    value.span,
                ));
            }
        }
    } else {
        variants.iter().find(|v| v.name == prefix)
    };
    // An unknown variant or arm is left for the match checker to report.
    let Some(declaration) = declaration else {
        return Ok(());
    };
    let Some(selected) = declaration.arms.iter().find(|arm| arm.name == case) else {
        return Ok(());
    };
    let others = declaration.arms.iter().filter(|arm| arm.name != case);
    for (index, other) in others.clone().enumerate() {
        arms.push(MatchArm {
            variant: format!("{prefix}::{}", other.name),
            binding: other
                .payload
                .as_ref()
                .map(|_| format!("{hidden}@{}", index + 1)),
            guard: None,
            value: Expr::new(ExprKind::Bool(false), first.value.span)?,
            span: first.span,
        });
    }
    let Some(StmtKind::Binding {
        annotation, value, ..
    }) = extraction.map(|stmt| &mut stmt.kind)
    else {
        return Ok(());
    };
    let ExprKind::Match { arms, .. } = &mut value.kind else {
        return Ok(());
    };
    let Some(payload) = selected.payload.clone() else {
        return Ok(());
    };
    let zero = match payload {
        Type::I32 => ExprKind::Integer("0i32".to_owned()),
        Type::U32 => ExprKind::Integer("0u32".to_owned()),
        Type::F32 => ExprKind::Float("0.0f32".to_owned()),
        Type::F64 => ExprKind::Float("0.0f64".to_owned()),
        Type::Bool => ExprKind::Bool(false),
        _ => {
            return Err(FrontError::source(
                "UNSUPPORTED_FEATURE",
                "a let-else pattern binds a scalar payload",
                first.span,
            ));
        }
    };
    let filler_span = arms[0].value.span;
    for (index, other) in others.enumerate() {
        arms.push(MatchArm {
            variant: format!("{prefix}::{}", other.name),
            binding: other
                .payload
                .as_ref()
                .map(|_| format!("{hidden}@x{}", index + 1)),
            guard: None,
            value: Expr::new(zero.clone(), filler_span)?,
            span: first.span,
        });
    }
    *annotation = Some(payload);
    Ok(())
}
