use crate::front::{Import, ModuleDecl};

const MAX_MODULES: usize = 32;
const MAX_IMPORTS: usize = 256;
const MAX_RECORDS: usize = 128;
const MAX_RECORD_FIELDS: usize = 16;
const MAX_VARIANTS: usize = 128;
const MAX_VARIANT_ARMS: usize = 16;

fn resolve_modules(program: &mut Program) -> Result<(), FrontError> {
    let user_modules: Vec<&ModuleDecl> = program.modules.iter().filter(|module| module.name != "std::robotics").collect();
    if user_modules.len() > MAX_MODULES || program.imports.len() > MAX_IMPORTS {
        return Err(FrontError::source(
            "MODULE_LIMIT",
            "module or import count exceeds its limit",
            user_modules
                .get(MAX_MODULES)
                .map(|module| module.span)
                .or_else(|| program.imports.get(MAX_IMPORTS).map(|import| import.span))
                .unwrap_or(Span { start: 0, end: 0 }),
        ));
    }
    let mut declared = HashMap::<String, Span>::new();
    for ModuleDecl { name, span } in &program.modules {
        if declared.insert(name.clone(), *span).is_some() {
            return Err(FrontError::source(
                "NAME_DUPLICATE",
                format!("duplicate module '{name}'"),
                *span,
            ));
        }
    }
    let mut functions = HashMap::<Option<String>, HashMap<String, bool>>::new();
    for function in &program.functions {
        let short = function.name.rsplit("::").next().unwrap_or(&function.name);
        let scope = function.module.clone();
        if functions
            .entry(scope)
            .or_default()
            .insert(short.to_owned(), function.public)
            .is_some()
        {
            return Err(FrontError::source(
                "NAME_DUPLICATE",
                format!("duplicate function or export '{short}'"),
                function.span,
            ));
        }
    }
    let user_records: Vec<&crate::front::RecordDecl> = program.records.iter().filter(|record| record.module.as_deref() != Some("std::robotics")).collect();
    if user_records.len() > MAX_RECORDS {
        return Err(FrontError::source(
            "RECORD_LIMIT",
            "record count exceeds 128",
            user_records[MAX_RECORDS].span,
        ));
    }
    let mut records = HashMap::<Option<String>, HashMap<String, bool>>::new();
    for record in &program.records {
        if record.name.rsplit("::").next().is_some_and(crate::spatial::is_builtin) {
            return Err(FrontError::source("NAME_DUPLICATE", "spatial builtin types cannot be redefined", record.span));
        }
        if record.fields.is_empty() || record.fields.len() > MAX_RECORD_FIELDS {
            return Err(FrontError::source(
                "RECORD_LIMIT",
                "record field count must be between 1 and 16",
                record.span,
            ));
        }
        let short = record.name.rsplit("::").next().unwrap_or(&record.name);
        if functions.get(&record.module).is_some_and(|items| items.contains_key(short))
            || records.entry(record.module.clone()).or_default().insert(short.to_owned(), record.public).is_some()
        {
            return Err(FrontError::source(
                "NAME_DUPLICATE",
                format!("duplicate record or value name '{short}'"),
                record.span,
            ));
        }
        let mut fields = HashSet::new();
        for field in &record.fields {
            if !fields.insert(field.name.as_str()) {
                return Err(FrontError::source(
                    "NAME_DUPLICATE",
                    format!("duplicate record field '{}'", field.name),
                    field.span,
                ));
            }
        }
    }
    if program.variants.len() > MAX_VARIANTS {
        return Err(FrontError::source("VARIANT_LIMIT", "variant count exceeds 128", program.variants[MAX_VARIANTS].span));
    }
    for variant in &program.variants {
        if variant.name.rsplit("::").next().is_some_and(crate::spatial::is_builtin) {
            return Err(FrontError::source("NAME_DUPLICATE", "spatial builtin types cannot be redefined", variant.span));
        }
        if variant.arms.is_empty() || variant.arms.len() > MAX_VARIANT_ARMS {
            return Err(FrontError::source("VARIANT_LIMIT", "variant arm count must be between 1 and 16", variant.span));
        }
        let short = variant.name.rsplit("::").next().unwrap_or(&variant.name);
        if functions.get(&variant.module).is_some_and(|items| items.contains_key(short))
            || records.entry(variant.module.clone()).or_default().insert(short.to_owned(), variant.public).is_some()
        {
            return Err(FrontError::source("NAME_DUPLICATE", format!("duplicate variant or value name '{short}'"), variant.span));
        }
        let mut arms = HashSet::new();
        for arm in &variant.arms {
            if !arms.insert(arm.name.as_str()) {
                return Err(FrontError::source("NAME_DUPLICATE", format!("duplicate variant arm '{}'", arm.name), arm.span));
            }
        }
    }
    let mut aliases = HashMap::<Option<String>, HashMap<String, String>>::new();
    for import in &program.imports {
        let target = imported_target(import, &declared, &functions, &records)?;
        if functions
            .get(&import.owner)
            .is_some_and(|local| local.contains_key(&import.item))
            || records.get(&import.owner).is_some_and(|local| local.contains_key(&import.item))
            || intrinsic(&import.item)
            || aliases
                .entry(import.owner.clone())
                .or_default()
                .insert(import.item.clone(), target)
                .is_some()
        {
            return Err(FrontError::source(
                "NAME_DUPLICATE",
                format!("duplicate imported name '{}'", import.item),
                import.span,
            ));
        }
    }
    for module in &program.modules {
        let mut stack = vec![module.name.as_str()];
        let mut seen = HashSet::new();
        while let Some(current) = stack.pop() {
            if !seen.insert(current) {
                continue;
            }
            for import in program
                .imports
                .iter()
                .filter(|import| import.owner.as_deref() == Some(current))
            {
                if import.module == module.name {
                    return Err(FrontError::source(
                        "MODULE_IMPORT_CYCLE",
                        format!("import cycle reaches module '{}'", module.name),
                        import.span,
                    ));
                }
                stack.push(&import.module);
            }
        }
    }
    for record in &mut program.records {
        let params: Vec<String> = record.generic_params.iter().map(|param| param.name.clone()).collect();
        for field in &mut record.fields {
            resolve_record_type(&mut field.ty, &record.module, field.span, &records, &aliases, &params)?;
        }
    }
    for variant in &mut program.variants {
        for arm in &mut variant.arms {
            if let Some(payload) = &mut arm.payload {
                resolve_record_type(payload, &variant.module, arm.span, &records, &aliases, &[])?;
            }
        }
    }
    for function in &mut program.functions {
        for param in &mut function.params {
            resolve_record_type(&mut param.ty, &function.module, param.span, &records, &aliases, &function.generic_params)?;
        }
        resolve_record_type(&mut function.result, &function.module, function.span, &records, &aliases, &function.generic_params)?;
        resolve_block(&mut function.body, &function.module, &functions, &aliases)?;
        resolve_record_block(&mut function.body, &function.module, &records, &aliases, &function.generic_params)?;
    }
    Ok(())
}

fn imported_target(
    import: &Import,
    declared: &HashMap<String, Span>,
    functions: &HashMap<Option<String>, HashMap<String, bool>>,
    records: &HashMap<Option<String>, HashMap<String, bool>>,
) -> Result<String, FrontError> {
    if !declared.contains_key(&import.module) {
        return Err(FrontError::source(
            "NAME_UNKNOWN",
            format!("unknown module '{}'", import.module),
            import.span,
        ));
    }
    let visible = functions
        .get(&Some(import.module.clone()))
        .and_then(|items| items.get(&import.item))
        .or_else(|| records.get(&Some(import.module.clone())).and_then(|items| items.get(&import.item)));
    match visible {
        Some(true) => Ok(format!("{}::{}", import.module, import.item)),
        Some(false) if import.owner.as_deref() == Some(import.module.as_str()) => {
            Ok(format!("{}::{}", import.module, import.item))
        }
        _ => Err(FrontError::source(
            "NAME_UNKNOWN",
            format!("unknown or private import '{}::{}'", import.module, import.item),
            import.span,
        )),
    }
}

fn resolve_record_type(
    ty: &mut Type,
    scope: &Option<String>,
    span: Span,
    records: &HashMap<Option<String>, HashMap<String, bool>>,
    aliases: &HashMap<Option<String>, HashMap<String, String>>,
    generic_params: &[String],
) -> Result<(), FrontError> {
    let Type::Named(name) = ty else { return Ok(()); };
    if crate::spatial::is_builtin(name) {
        return Ok(());
    }
    if generic_params.iter().any(|param| param == name || name == &format!("Array<{param}>")) {
        return Ok(());
    }
    if storage::is_array(name) {
        return Ok(());
    }
    if let Some(element) = name.strip_prefix("Array<").and_then(|name| name.strip_suffix('>')) {
        let mut inner = Type::Named(element.to_owned());
        resolve_record_type(&mut inner, scope, span, records, aliases, generic_params)?;
        let Type::Named(element) = inner else { unreachable!("named array element") };
        *name = format!("Array<{element}>");
        return Ok(());
    }
    if let Some(inside) = name.strip_prefix("FixedArray<").and_then(|name| name.strip_suffix('>')) {
        let (element, count) = inside.rsplit_once(',').ok_or_else(|| FrontError::source("TYPE_MISMATCH", "fixed array needs element and count", span))?;
        if !generic_params.iter().any(|param| param == count)
            && !count.parse::<u32>().is_ok_and(|count| (1..=64).contains(&count)) {
            return Err(FrontError::source("RECORD_LIMIT", "fixed array count must be 1..64", span));
        }
        let element = if matches!(element, "i32" | "u32" | "f32" | "f64" | "i64") {
            element.to_owned()
        } else {
            let mut inner = Type::Named(element.to_owned());
            resolve_record_type(&mut inner, scope, span, records, aliases, generic_params)?;
            let Type::Named(element) = inner else { unreachable!("named fixed array element") };
            element
        };
        *name = format!("FixedArray<{element},{count}>");
        return Ok(());
    }
    if matches!(name.as_str(), "f32" | "f64" | "i64" | "Bytes" | "String" | "Utf8Error" | "ForeignError") {
        return Ok(());
    }
    if crate::front::result_parts(name).is_some() || crate::front::view_parts(name).is_some() || ["Quantity<", "Vector3<", "Point<", "Transform<"].iter().any(|prefix| name.starts_with(prefix)) {
        return Ok(());
    }
    let base = name.split_once('<').map_or(name.as_str(), |(base, _)| base);
    let suffix = name[base.len()..].to_owned();
    if let Some((module, item)) = base.rsplit_once("::") {
        let visible = records.get(&Some(module.to_owned())).and_then(|items| items.get(item));
        if visible.is_some_and(|public| *public || scope.as_deref() == Some(module)) {
            return Ok(());
        }
    } else if records.get(scope).is_some_and(|items| items.contains_key(base)) {
        if let Some(module) = scope {
            *name = format!("{module}::{base}{suffix}");
        }
        return Ok(());
    } else if let Some(target) = aliases.get(scope).and_then(|items| items.get(base))
        && target.rsplit_once("::").is_some_and(|(module, item)| {
            records.get(&Some(module.to_owned())).is_some_and(|items| items.contains_key(item))
        }) {
        *name = format!("{target}{suffix}");
        return Ok(());
    }
    Err(FrontError::source("NAME_UNKNOWN", format!("unknown or private record type '{name}'"), span))
}

fn resolve_block(
    block: &mut Block,
    scope: &Option<String>,
    functions: &HashMap<Option<String>, HashMap<String, bool>>,
    aliases: &HashMap<Option<String>, HashMap<String, String>>,
) -> Result<(), FrontError> {
    for statement in &mut block.statements {
        resolve_statement(statement, scope, functions, aliases)?;
    }
    Ok(())
}

fn resolve_statement(
    statement: &mut Stmt,
    scope: &Option<String>,
    functions: &HashMap<Option<String>, HashMap<String, bool>>,
    aliases: &HashMap<Option<String>, HashMap<String, String>>,
) -> Result<(), FrontError> {
    match &mut statement.kind {
            StmtKind::Binding { value, .. } | StmtKind::Expr(value) => {
                resolve_expr(value, scope, functions, aliases)?;
            }
            StmtKind::Assign { place, value } => {
                match place {
                    Place::Index { index, .. } => resolve_expr(index, scope, functions, aliases)?,
                    Place::Name(..) => {}
                }
                resolve_expr(value, scope, functions, aliases)?;
            }
            StmtKind::Return(value) => {
                if let Some(value) = value {
                    resolve_expr(value, scope, functions, aliases)?;
                }
            }
            StmtKind::If { condition, then_block, else_arm } => {
                resolve_expr(condition, scope, functions, aliases)?;
                resolve_block(then_block, scope, functions, aliases)?;
                if let Some(else_arm) = else_arm {
                    match else_arm {
                        ElseArm::Block(block) => resolve_block(block, scope, functions, aliases)?,
                        ElseArm::If(statement) => resolve_statement(statement, scope, functions, aliases)?,
                    }
                }
            }
            StmtKind::While { condition, body } => {
                resolve_expr(condition, scope, functions, aliases)?;
                resolve_block(body, scope, functions, aliases)?;
            }
            StmtKind::For { lower, upper, body, .. } => {
                resolve_expr(lower, scope, functions, aliases)?;
                resolve_expr(upper, scope, functions, aliases)?;
                resolve_block(body, scope, functions, aliases)?;
            }
    }
    Ok(())
}

fn resolve_expr(
    expr: &mut Expr,
    scope: &Option<String>,
    functions: &HashMap<Option<String>, HashMap<String, bool>>,
    aliases: &HashMap<Option<String>, HashMap<String, String>>,
) -> Result<(), FrontError> {
    match &mut expr.kind {
        ExprKind::Call { name, args } => {
            for arg in args {
                resolve_expr(arg, scope, functions, aliases)?;
            }
            let base = name.split_once("$<").map_or(name.as_str(), |(base, _)| base).to_owned();
            let suffix = name[base.len()..].to_owned();
            if intrinsic(&base) {
                return Ok(());
            }
            if let Some((module, item)) = base.rsplit_once("::") {
                let visible = functions
                    .get(&Some(module.to_owned()))
                    .and_then(|items| items.get(item));
                if visible.is_some_and(|public| !*public && scope.as_deref() != Some(module)) {
                    return Err(FrontError::source(
                        "NAME_UNKNOWN",
                        format!("unknown or private function '{name}'"),
                        expr.span,
                    ));
                }
            } else if functions.get(scope).is_some_and(|items| items.contains_key(&base)) {
                if let Some(module) = scope {
                    *name = format!("{module}::{base}{suffix}");
                }
            } else if let Some(target) = aliases.get(scope).and_then(|items| items.get(&base)) {
                *name = format!("{target}{suffix}");
            }
        }
        ExprKind::Array(items) => {
            for item in items {
                resolve_expr(item, scope, functions, aliases)?;
            }
        }
        ExprKind::Record { fields, .. } => {
            for field in fields {
                resolve_expr(&mut field.value, scope, functions, aliases)?;
            }
        }
        ExprKind::Field { base, .. } => resolve_expr(base, scope, functions, aliases)?,
        ExprKind::Match { value, arms } => {
            resolve_expr(value, scope, functions, aliases)?;
            for arm in arms { resolve_expr(&mut arm.value, scope, functions, aliases)?; }
        }
        ExprKind::Index { base, index } | ExprKind::Binary { left: base, right: index, .. } => {
            resolve_expr(base, scope, functions, aliases)?;
            resolve_expr(index, scope, functions, aliases)?;
        }
        ExprKind::Unary { value, .. } | ExprKind::Group(value) | ExprKind::Try(value) => {
            resolve_expr(value, scope, functions, aliases)?;
        }
        ExprKind::Integer(_) | ExprKind::Float(_) | ExprKind::Bool(_) | ExprKind::Text(_) | ExprKind::Bytes(_) | ExprKind::Name(_) | ExprKind::Move(_) | ExprKind::MoveField { .. } => {}
    }
    Ok(())
}

fn resolve_record_block(
    block: &mut Block,
    scope: &Option<String>,
    records: &HashMap<Option<String>, HashMap<String, bool>>,
    aliases: &HashMap<Option<String>, HashMap<String, String>>,
    generic_params: &[String],
) -> Result<(), FrontError> {
    for statement in &mut block.statements {
        resolve_record_statement(statement, scope, records, aliases, generic_params)?;
    }
    Ok(())
}

fn resolve_record_statement(
    statement: &mut Stmt,
    scope: &Option<String>,
    records: &HashMap<Option<String>, HashMap<String, bool>>,
    aliases: &HashMap<Option<String>, HashMap<String, String>>,
    generic_params: &[String],
) -> Result<(), FrontError> {
    match &mut statement.kind {
        StmtKind::Binding { annotation, value, .. } => {
            if let Some(annotation) = annotation {
                resolve_record_type(annotation, scope, statement.span, records, aliases, generic_params)?;
            }
            resolve_record_expr(value, scope, records, aliases, generic_params)?;
        }
        StmtKind::Assign { place, value } => {
            if let Place::Index { index, .. } = place {
                resolve_record_expr(index, scope, records, aliases, generic_params)?;
            }
            resolve_record_expr(value, scope, records, aliases, generic_params)?;
        }
        StmtKind::Return(value) => {
            if let Some(value) = value {
                resolve_record_expr(value, scope, records, aliases, generic_params)?;
            }
        }
        StmtKind::If { condition, then_block, else_arm } => {
            resolve_record_expr(condition, scope, records, aliases, generic_params)?;
            resolve_record_block(then_block, scope, records, aliases, generic_params)?;
            if let Some(else_arm) = else_arm {
                match else_arm {
                    ElseArm::Block(block) => resolve_record_block(block, scope, records, aliases, generic_params)?,
                    ElseArm::If(statement) => resolve_record_statement(statement, scope, records, aliases, generic_params)?,
                }
            }
        }
        StmtKind::While { condition, body } => {
            resolve_record_expr(condition, scope, records, aliases, generic_params)?;
            resolve_record_block(body, scope, records, aliases, generic_params)?;
        }
        StmtKind::For { lower, upper, body, .. } => {
            resolve_record_expr(lower, scope, records, aliases, generic_params)?;
            resolve_record_expr(upper, scope, records, aliases, generic_params)?;
            resolve_record_block(body, scope, records, aliases, generic_params)?;
        }
        StmtKind::Expr(value) => resolve_record_expr(value, scope, records, aliases, generic_params)?,
    }
    Ok(())
}

fn resolve_record_expr(
    expr: &mut Expr,
    scope: &Option<String>,
    records: &HashMap<Option<String>, HashMap<String, bool>>,
    aliases: &HashMap<Option<String>, HashMap<String, String>>,
    generic_params: &[String],
) -> Result<(), FrontError> {
    match &mut expr.kind {
        ExprKind::Record { name, fields } => {
            let mut ty = Type::Named(name.clone());
            resolve_record_type(&mut ty, scope, expr.span, records, aliases, generic_params)?;
            if let Type::Named(resolved) = ty {
                *name = resolved;
            }
            for field in fields {
                resolve_record_expr(&mut field.value, scope, records, aliases, generic_params)?;
            }
        }
        ExprKind::Match { value, arms } => {
            resolve_record_expr(value, scope, records, aliases, generic_params)?;
            for arm in arms {
                if let Some((prefix, item)) = arm.variant.rsplit_once("::") {
                    let mut ty = Type::Named(prefix.to_owned());
                    if resolve_record_type(&mut ty, scope, arm.span, records, aliases, generic_params).is_ok()
                        && let Type::Named(resolved) = ty {
                        arm.variant = format!("{resolved}::{item}");
                    }
                }
                resolve_record_expr(&mut arm.value, scope, records, aliases, generic_params)?;
            }
        }
        ExprKind::Field { base, .. } | ExprKind::Group(base) | ExprKind::Unary { value: base, .. } | ExprKind::Try(base) => {
            resolve_record_expr(base, scope, records, aliases, generic_params)?;
        }
        ExprKind::Call { name, args } => {
            if let Some((prefix, item)) = name.rsplit_once("::") {
                let mut ty = Type::Named(prefix.to_owned());
                if resolve_record_type(&mut ty, scope, expr.span, records, aliases, generic_params).is_ok()
                    && let Type::Named(resolved) = ty {
                    *name = format!("{resolved}::{item}");
                }
            }
            for item in args { resolve_record_expr(item, scope, records, aliases, generic_params)?; }
        }
        ExprKind::Array(items) => {
            for item in items {
                resolve_record_expr(item, scope, records, aliases, generic_params)?;
            }
        }
        ExprKind::Index { base, index } | ExprKind::Binary { left: base, right: index, .. } => {
            resolve_record_expr(base, scope, records, aliases, generic_params)?;
            resolve_record_expr(index, scope, records, aliases, generic_params)?;
        }
        ExprKind::Integer(_) | ExprKind::Float(_) | ExprKind::Bool(_) | ExprKind::Text(_) | ExprKind::Bytes(_) | ExprKind::Name(_) | ExprKind::Move(_) | ExprKind::MoveField { .. } => {}
    }
    Ok(())
}
