use std::collections::BTreeMap;

use crate::front::{RecordField, VariantArmDecl, VariantDecl, result_parts};

use super::*;

fn collect_type(types: &mut BTreeMap<String, (Type, Type, Span)>, ty: &Type, span: Span) {
    if let Type::Named(name) = ty
        && let Some((ok, err)) = result_parts(name)
    {
        collect_type(types, &ok, span);
        collect_type(types, &err, span);
        types.entry(name.clone()).or_insert((ok, err, span));
    }
}

fn collect_expr(types: &mut BTreeMap<String, (Type, Type, Span)>, expr: &Expr) {
    let mut pending = vec![expr];
    while let Some(expr) = pending.pop() {
        match &expr.kind {
            ExprKind::Call { name, args } => {
                if matches!(name.as_str(), "utf8_decode" | "utf8_slice") {
                    collect_type(
                        types,
                        &Type::Named("Result<String,Utf8Error>".to_owned()),
                        expr.span,
                    );
                }
                pending.extend(args);
            }
            ExprKind::MoveField { base, .. }
            | ExprKind::Group(base)
            | ExprKind::Try(base)
            | ExprKind::Field { base, .. }
            | ExprKind::Unary { value: base, .. } => pending.push(base),
            ExprKind::Array(items) => pending.extend(items),
            ExprKind::Record { fields, .. } => {
                pending.extend(fields.iter().map(|field| &field.value))
            }
            ExprKind::Match { value, arms } => {
                pending.push(value);
                pending.extend(arms.iter().map(|arm| &arm.value));
            }
            ExprKind::Index { base, index } => {
                pending.push(base);
                pending.push(index);
            }
            ExprKind::Binary { left, right, .. } => {
                pending.push(left);
                pending.push(right);
            }
            ExprKind::Integer(_)
            | ExprKind::Float(_)
            | ExprKind::Bool(_)
            | ExprKind::Text(_)
            | ExprKind::Bytes(_)
            | ExprKind::Name(_)
            | ExprKind::Move(_) => {}
        }
    }
}

fn collect_block(types: &mut BTreeMap<String, (Type, Type, Span)>, block: &Block) {
    let mut pending: Vec<&Stmt> = block.statements.iter().collect();
    while let Some(statement) = pending.pop() {
        match &statement.kind {
            StmtKind::Binding {
                annotation: Some(ty),
                value,
                ..
            } => {
                collect_type(types, ty, statement.span);
                collect_expr(types, value);
            }
            StmtKind::Binding { value, .. } | StmtKind::Expr(value) => collect_expr(types, value),
            StmtKind::Assign { place, value } => {
                collect_expr(types, value);
                if let Place::Index { index, .. } = place {
                    collect_expr(types, index);
                }
            }
            StmtKind::Return(Some(value)) => collect_expr(types, value),
            StmtKind::Return(None) => {}
            StmtKind::If {
                condition,
                then_block,
                else_arm,
                ..
            } => {
                collect_expr(types, condition);
                pending.extend(&then_block.statements);
                if let Some(else_arm) = else_arm {
                    match else_arm {
                        ElseArm::Block(block) => pending.extend(&block.statements),
                        ElseArm::If(statement) => pending.push(statement),
                    }
                }
            }
            StmtKind::While { condition, body } => {
                collect_expr(types, condition);
                pending.extend(&body.statements);
            }
            StmtKind::For {
                lower, upper, body, ..
            } => {
                collect_expr(types, lower);
                collect_expr(types, upper);
                pending.extend(&body.statements);
            }
        }
    }
}

pub(super) fn synthesize_results(program: &mut Program) -> Result<(), FrontError> {
    if !program.foreign_functions.is_empty() {
        if program
            .variants
            .iter()
            .any(|variant| variant.name == "ForeignError")
        {
            return Err(FrontError::source(
                "NAME_DUPLICATE",
                "ForeignError is a built-in type",
                Span { start: 0, end: 1 },
            ));
        }
        let span = Span { start: 0, end: 1 };
        program.variants.push(VariantDecl {
            name: "ForeignError".to_owned(),
            module: None,
            public: true,
            arms: vec![VariantArmDecl {
                name: "Failure".to_owned(),
                payload: None,
                span,
            }],
            span,
        });
    }
    if program.records.iter().any(|record| record.name == "Result")
        || program
            .variants
            .iter()
            .any(|variant| variant.name == "Result")
    {
        return Err(FrontError::source(
            "NAME_DUPLICATE",
            "Result is a built-in type",
            Span { start: 0, end: 1 },
        ));
    }
    let mut types = BTreeMap::new();
    for record in &program.records {
        for RecordField { ty, span, .. } in &record.fields {
            collect_type(&mut types, ty, *span);
        }
    }
    for variant in &program.variants {
        for arm in &variant.arms {
            if let Some(ty) = &arm.payload {
                collect_type(&mut types, ty, arm.span);
            }
        }
    }
    for function in &program.functions {
        for param in &function.params {
            collect_type(&mut types, &param.ty, param.span);
        }
        collect_type(&mut types, &function.result, function.span);
        collect_block(&mut types, &function.body);
    }
    for foreign in &program.foreign_functions {
        for param in &foreign.params {
            collect_type(&mut types, &param.ty, param.span);
        }
        collect_type(&mut types, &foreign.result, foreign.span);
    }
    if program.variants.len().saturating_add(types.len()) > 128 {
        return Err(FrontError::source(
            "VARIANT_LIMIT",
            "variant and Result specialization count exceeds 128",
            Span { start: 0, end: 1 },
        ));
    }
    for (name, (ok, err, span)) in types {
        if program.variants.iter().any(|variant| variant.name == name) {
            return Err(FrontError::source(
                "NAME_DUPLICATE",
                "Result specialization conflicts with a declared variant",
                span,
            ));
        }
        program.variants.push(VariantDecl {
            name,
            module: None,
            public: false,
            arms: vec![
                VariantArmDecl {
                    name: "Ok".to_owned(),
                    payload: Some(ok),
                    span,
                },
                VariantArmDecl {
                    name: "Err".to_owned(),
                    payload: Some(err),
                    span,
                },
            ],
            span,
        });
    }
    Ok(())
}
