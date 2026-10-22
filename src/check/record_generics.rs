//! Materialize only the concrete record types used by the checked program.

use super::*;
use crate::front::{RecordDecl, RecordGenericKind};

const MAX_RECORD_INSTANCES: usize = 128;

fn parts(name: &str) -> Option<(&str, Vec<&str>)> {
    let (base, tail) = name.split_once('<')?;
    let inside = tail.strip_suffix('>')?;
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut args = Vec::new();
    for (index, byte) in inside.bytes().enumerate() {
        match byte {
            b'<' => depth = depth.checked_add(1)?,
            b'>' => depth = depth.checked_sub(1)?,
            b',' if depth == 0 => {
                args.push(&inside[start..index]);
                start = index + 1;
            }
            _ => {}
        }
    }
    if depth != 0 {
        return None;
    }
    args.push(&inside[start..]);
    Some((base, args))
}

fn substitute(name: &str, args: &HashMap<&str, &str>) -> String {
    let mut result = String::with_capacity(name.len());
    let mut start = 0usize;
    let bytes = name.as_bytes();
    for index in 0..=bytes.len() {
        if index < bytes.len() && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_') {
            continue;
        }
        if start < index {
            let word = &name[start..index];
            result.push_str(args.get(word).copied().unwrap_or(word));
        }
        if index < bytes.len() {
            result.push(bytes[index] as char);
        }
        start = index + 1;
    }
    result
}

fn concrete_type(ty: &Type, args: &HashMap<&str, &str>) -> Type {
    let name = match ty {
        Type::Named(name) => name,
        _ => return ty.clone(),
    };
    match substitute(name, args).as_str() {
        "i32" => Type::I32,
        "u32" => Type::U32,
        "Bool" => Type::Bool,
        "Array<i32>" => Type::ArrayI32,
        "Array<u32>" => Type::ArrayU32,
        name => Type::Named(name.to_owned()),
    }
}

fn collect_expr(expr: &Expr, names: &mut Vec<String>) {
    match &expr.kind {
        ExprKind::Record { name, fields } => {
            names.push(name.clone());
            for field in fields {
                collect_expr(&field.value, names);
            }
        }
        ExprKind::Array(items) => {
            for item in items {
                collect_expr(item, names);
            }
        }
        ExprKind::Call { args, .. } => {
            for arg in args {
                collect_expr(arg, names);
            }
        }
        ExprKind::Binary { left, right, .. }
        | ExprKind::Index {
            base: left,
            index: right,
        } => {
            collect_expr(left, names);
            collect_expr(right, names);
        }
        ExprKind::Unary { value, .. }
        | ExprKind::Group(value)
        | ExprKind::Try(value)
        | ExprKind::Field { base: value, .. } => collect_expr(value, names),
        ExprKind::Match { value, arms } => {
            collect_expr(value, names);
            for arm in arms {
                collect_expr(&arm.value, names);
            }
        }
        ExprKind::Integer(_)
        | ExprKind::Float(_)
        | ExprKind::Bool(_)
        | ExprKind::Text(_)
        | ExprKind::Bytes(_)
        | ExprKind::Name(_)
        | ExprKind::Move(_)
        | ExprKind::MoveField { .. } => {}
    }
}

fn collect_block(block: &Block, names: &mut Vec<String>) {
    for statement in &block.statements {
        collect_statement(statement, names);
    }
}

fn collect_statement(statement: &Stmt, names: &mut Vec<String>) {
    match &statement.kind {
        StmtKind::Binding {
            annotation, value, ..
        } => {
            if let Some(Type::Named(name)) = annotation {
                names.push(name.clone());
            }
            collect_expr(value, names);
        }
        StmtKind::Assign { place, value } => {
            if let Place::Index { index, .. } = place {
                collect_expr(index, names);
            }
            collect_expr(value, names);
        }
        StmtKind::Expr(value) => collect_expr(value, names),
        StmtKind::Return(value) => {
            if let Some(value) = value {
                collect_expr(value, names);
            }
        }
        StmtKind::If {
            condition,
            then_block,
            else_arm,
        } => {
            collect_expr(condition, names);
            collect_block(then_block, names);
            if let Some(else_arm) = else_arm {
                match else_arm {
                    ElseArm::Block(block) => collect_block(block, names),
                    ElseArm::If(statement) => collect_statement(statement, names),
                }
            }
        }
        StmtKind::While { condition, body } => {
            collect_expr(condition, names);
            collect_block(body, names);
        }
        StmtKind::For {
            lower, upper, body, ..
        } => {
            collect_expr(lower, names);
            collect_expr(upper, names);
            collect_block(body, names);
        }
    }
}

pub(super) fn expand(program: &mut Program) -> Result<(), FrontError> {
    let mut templates = HashMap::<String, RecordDecl>::new();
    program.records.retain(|record| {
        if record.generic_params.is_empty() {
            true
        } else {
            templates.insert(record.name.clone(), record.clone());
            false
        }
    });
    if templates.is_empty() {
        return Ok(());
    }
    let mut pending = Vec::new();
    for function in &program.functions {
        for param in &function.params {
            if let Type::Named(name) = &param.ty {
                pending.push(name.clone());
            }
        }
        if let Type::Named(name) = &function.result {
            pending.push(name.clone());
        }
        collect_block(&function.body, &mut pending);
    }
    let frames: HashSet<&str> = program
        .frames
        .iter()
        .map(|frame| frame.name.as_str())
        .collect();
    let mut seen = HashSet::new();
    while let Some(name) = pending.pop() {
        let Some((base, values)) = parts(&name) else {
            continue;
        };
        if matches!(base, "Array" | "FixedArray") {
            if let Some(element) = values.first() {
                pending.push((*element).to_owned());
            }
            continue;
        }
        let Some(template) = templates.get(base) else {
            continue;
        };
        if !seen.insert(name.clone()) {
            continue;
        }
        if seen.len() > MAX_RECORD_INSTANCES {
            return Err(FrontError::source(
                "RECORD_LIMIT",
                "generic record instance limit exceeded",
                template.span,
            ));
        }
        if values.len() != template.generic_params.len() {
            return Err(FrontError::source(
                "TYPE_MISMATCH",
                "generic record argument count mismatch",
                template.span,
            ));
        }
        if crate::robotics::is_planner_record(base)
            && crate::robotics::planner_joint_count(&name).is_none()
        {
            return Err(FrontError::source(
                "TYPE_MISMATCH",
                "planner-v1 records require binary64 and exactly six or seven joints",
                template.span,
            ));
        }
        let mut replacements = HashMap::new();
        for (param, value) in template.generic_params.iter().zip(values) {
            let valid = match param.kind {
                RecordGenericKind::Value => {
                    matches!(value, "i32" | "u32" | "f32" | "f64" | "i64")
                        || parts(value).is_some_and(|(base, _)| templates.contains_key(base))
                        || program.records.iter().any(|record| record.name == value)
                }
                RecordGenericKind::Frame => frames.contains(value),
                RecordGenericKind::Unit => super::unit_math::Unit::parse(value).is_some(),
                RecordGenericKind::Count => value
                    .parse::<u32>()
                    .is_ok_and(|count| (1..=64).contains(&count)),
            };
            if !valid {
                return Err(FrontError::source(
                    if param.kind == RecordGenericKind::Frame {
                        "FRAME_UNKNOWN"
                    } else {
                        "TYPE_MISMATCH"
                    },
                    format!("invalid {} record argument '{value}'", param.name),
                    template.span,
                ));
            }
            replacements.insert(param.name.as_str(), value);
        }
        let mut concrete = template.clone();
        concrete.name = name.clone();
        concrete.generic_params.clear();
        for field in &mut concrete.fields {
            field.ty = concrete_type(&field.ty, &replacements);
            if let Type::Named(name) = &field.ty {
                pending.push(name.clone());
            }
        }
        program.records.push(concrete);
    }
    Ok(())
}

fn type_only(
    ty: &Type,
    records: &HashMap<&str, &RecordDecl>,
    visiting: &mut HashSet<String>,
) -> bool {
    let Type::Named(name) = ty else {
        return false;
    };
    if matches!(name.as_str(), "f32" | "f64" | "i64")
        || name.starts_with("FixedArray<")
        || name.starts_with("Quantity<f32,")
        || name.starts_with("Quantity<f64,")
        || name.starts_with("Quantity<i64,")
    {
        return true;
    }
    if !visiting.insert(name.clone()) {
        return false;
    }
    let result = records.get(name.as_str()).is_some_and(|record| {
        record.fields.iter().any(|field| {
            matches!(&field.ty, Type::Named(nested) if record.module.as_deref() == Some("std::robotics") && records.contains_key(nested.as_str()))
                || type_only(&field.ty, records, visiting)
        })
    });
    visiting.remove(name);
    result
}

pub(super) fn reject_type_only_runtime(program: &Program) -> Result<(), FrontError> {
    let records: HashMap<&str, &RecordDecl> = program
        .records
        .iter()
        .map(|record| (record.name.as_str(), record))
        .collect();
    for function in &program.functions {
        let mut names = Vec::new();
        collect_block(&function.body, &mut names);
        if function
            .params
            .iter()
            .any(|param| type_only(&param.ty, &records, &mut HashSet::new()))
            || type_only(&function.result, &records, &mut HashSet::new())
            || names
                .iter()
                .any(|name| type_only(&Type::Named(name.clone()), &records, &mut HashSet::new()))
        {
            return Err(FrontError::source(
                "UNSUPPORTED_FEATURE",
                "type-only robotics wire values cannot be runtime function parameters or results before M4-01",
                function.span,
            ));
        }
    }
    for record in &program.records {
        if record.module.as_deref() != Some("std::robotics")
            && record
                .fields
                .iter()
                .any(|field| type_only(&field.ty, &records, &mut HashSet::new()))
        {
            return Err(FrontError::source(
                "UNSUPPORTED_FEATURE",
                "type-only robotics wire values cannot be runtime record fields before M4-01",
                record.span,
            ));
        }
    }
    Ok(())
}
