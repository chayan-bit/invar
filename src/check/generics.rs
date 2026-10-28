//! Bounded source monomorphization. Instances retain source spans; type facts are per function.

use super::*;

pub(super) const MAX_FUNCTIONS: usize = 128;
const MAX_GENERIC_DEPTH: usize = 16;

pub(super) fn validate_template(function: &Function) -> Result<(), FrontError> {
    if function.name == "main" || function.generic_params.len() > 2 {
        return Err(FrontError::source(
            "UNSUPPORTED_FEATURE",
            "generic functions need one or two Value parameters and cannot be main",
            function.span,
        ));
    }
    let mut seen = HashSet::new();
    for name in &function.generic_params {
        if !seen.insert(name) || function.params.iter().any(|param| param.name == *name) {
            return Err(FrontError::source(
                "NAME_DUPLICATE",
                "duplicate or shadowed generic parameter",
                function.span,
            ));
        }
    }
    Ok(())
}

fn value_type(ty: &Type) -> bool {
    match ty {
        Type::I32
        | Type::U32
        | Type::F32
        | Type::F64
        | Type::Bool
        | Type::ArrayI32
        | Type::ArrayU32
        | Type::ArrayF32
        | Type::ArrayF64
        | Type::ArrayBool => true,
        Type::Named(name) => {
            crate::front::view_parts(name).is_none()
                && !name.starts_with("Array<")
                && name != "Unit"
        }
        Type::Unit => false,
    }
}

fn split_call(name: &str) -> (&str, Option<&str>) {
    match name.split_once("$<") {
        Some((base, rest)) => (base, rest.strip_suffix('>')),
        None => (name, None),
    }
}

fn substitute(
    ty: &Type,
    arguments: &HashMap<String, Type>,
    span: Span,
) -> Result<Type, FrontError> {
    if let Type::Named(name) = ty {
        if let Some(concrete) = arguments.get(name) {
            return Ok(concrete.clone());
        }
        if let Some(element) = name
            .strip_prefix("Array<")
            .and_then(|tail| tail.strip_suffix('>'))
        {
            return match arguments.get(element) {
                Some(Type::I32) => Ok(Type::ArrayI32),
                Some(Type::U32) => Ok(Type::ArrayU32),
                _ => Err(mismatch("Array<T> requires an integer type argument", span)),
            };
        }
    }
    Ok(ty.clone())
}

fn infer(
    formal: &Type,
    actual: &Type,
    arguments: &mut HashMap<String, Type>,
    parameters: &[String],
    span: Span,
) -> Result<(), FrontError> {
    let variable = match formal {
        Type::Named(name) if parameters.iter().any(|param| param == name) => Some(name.as_str()),
        Type::Named(name) => name
            .strip_prefix("Array<")
            .and_then(|tail| tail.strip_suffix('>'))
            .filter(|name| parameters.iter().any(|param| param == name)),
        _ => None,
    };
    if let Some(variable) = variable {
        let concrete = if matches!(formal, Type::Named(name) if name.starts_with("Array<")) {
            match actual {
                Type::ArrayI32 => Type::I32,
                Type::ArrayU32 => Type::U32,
                _ => {
                    return Err(mismatch(
                        "generic array argument must be an integer array",
                        span,
                    ));
                }
            }
        } else {
            actual.clone()
        };
        if !value_type(&concrete) {
            return Err(mismatch("generic type argument is not a Value", span));
        }
        if arguments
            .insert(variable.to_owned(), concrete.clone())
            .is_some_and(|previous| previous != concrete)
        {
            return Err(mismatch("conflicting generic type inference", span));
        }
    } else if formal != actual {
        return Err(mismatch("generic argument type mismatch", span));
    }
    Ok(())
}

fn replace_explicit_types(name: &mut String, arguments: &HashMap<String, Type>) {
    let (base, Some(types)) = split_call(name) else {
        return;
    };
    let base = base.to_owned();
    let types = types.to_owned();
    let changed = types
        .split('|')
        .map(|part| {
            if let Some(ty) = arguments.get(part) {
                ty.source_name().to_owned()
            } else if let Some(element) = part
                .strip_prefix("Array<")
                .and_then(|tail| tail.strip_suffix('>'))
            {
                arguments.get(element).map_or_else(
                    || part.to_owned(),
                    |ty| format!("Array<{}>", ty.source_name()),
                )
            } else {
                part.to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join("|");
    *name = format!("{base}$<{changed}>");
}

fn substitute_expr(expr: &mut Expr, arguments: &HashMap<String, Type>) {
    match &mut expr.kind {
        ExprKind::Call { name, args } => {
            replace_explicit_types(name, arguments);
            for arg in args {
                substitute_expr(arg, arguments);
            }
        }
        ExprKind::Array(items) => {
            for item in items {
                substitute_expr(item, arguments);
            }
        }
        ExprKind::Record { fields, .. } => {
            for field in fields {
                substitute_expr(&mut field.value, arguments);
            }
        }
        ExprKind::Match { value, arms } => {
            substitute_expr(value, arguments);
            for arm in arms {
                substitute_expr(&mut arm.value, arguments);
            }
        }
        ExprKind::Index { base, index }
        | ExprKind::Binary {
            left: base,
            right: index,
            ..
        } => {
            substitute_expr(base, arguments);
            substitute_expr(index, arguments);
        }
        ExprKind::MoveField { base, .. }
        | ExprKind::Field { base, .. }
        | ExprKind::Unary { value: base, .. }
        | ExprKind::Group(base)
        | ExprKind::Try(base) => substitute_expr(base, arguments),
        ExprKind::Integer(_)
        | ExprKind::Float(_)
        | ExprKind::Bool(_)
        | ExprKind::Text(_)
        | ExprKind::Bytes(_)
        | ExprKind::Name(_)
        | ExprKind::Move(_) => {}
    }
}

fn substitute_block(
    block: &mut Block,
    arguments: &HashMap<String, Type>,
) -> Result<(), FrontError> {
    for statement in &mut block.statements {
        substitute_statement(statement, arguments)?;
    }
    Ok(())
}

fn substitute_statement(
    statement: &mut Stmt,
    arguments: &HashMap<String, Type>,
) -> Result<(), FrontError> {
    match &mut statement.kind {
        StmtKind::Binding {
            annotation, value, ..
        } => {
            if let Some(annotation) = annotation {
                *annotation = substitute(annotation, arguments, statement.span)?;
            }
            substitute_expr(value, arguments);
        }
        StmtKind::Assign { place, value } => {
            if let Place::Index { index, .. } = place {
                substitute_expr(index, arguments);
            }
            substitute_expr(value, arguments);
        }
        StmtKind::Return(Some(value)) | StmtKind::Expr(value) => substitute_expr(value, arguments),
        StmtKind::If {
            condition,
            then_block,
            else_arm,
        } => {
            substitute_expr(condition, arguments);
            substitute_block(then_block, arguments)?;
            match else_arm {
                Some(ElseArm::Block(block)) => substitute_block(block, arguments)?,
                Some(ElseArm::If(statement)) => substitute_statement(statement, arguments)?,
                None => {}
            }
        }
        StmtKind::While { condition, body } => {
            substitute_expr(condition, arguments);
            substitute_block(body, arguments)?;
        }
        StmtKind::For {
            lower, upper, body, ..
        } => {
            substitute_expr(lower, arguments);
            substitute_expr(upper, arguments);
            substitute_block(body, arguments)?;
        }
        StmtKind::Return(None) => {}
    }
    Ok(())
}

fn instantiate(
    template: &Function,
    arguments: &HashMap<String, Type>,
    name: String,
) -> Result<Function, FrontError> {
    let mut function = template.clone();
    function.name = name;
    function.generic_params.clear();
    for param in &mut function.params {
        param.ty = substitute(&param.ty, arguments, param.span)?;
    }
    function.result = substitute(&function.result, arguments, function.span)?;
    substitute_block(&mut function.body, arguments)?;
    Ok(function)
}

impl FunctionChecker<'_> {
    pub(super) fn generic_call(
        &mut self,
        name: &str,
        args: &[Expr],
        span: Span,
    ) -> Result<Option<Type>, FrontError> {
        let (base, explicit) = split_call(name);
        let Some(template) = self.checker.generic_templates.get(base).cloned() else {
            if name.contains("$<") {
                return Err(FrontError::source(
                    "NAME_UNKNOWN",
                    "explicit type arguments require a generic function",
                    span,
                ));
            }
            return Ok(None);
        };
        if args.len() != template.params.len() {
            return Err(mismatch("wrong generic call arity", span));
        }
        let mut mapping = HashMap::new();
        if let Some(explicit) = explicit {
            let parts = explicit.split('|').collect::<Vec<_>>();
            if parts.len() != template.generic_params.len() {
                return Err(mismatch("wrong generic type argument arity", span));
            }
            for (param, part) in template.generic_params.iter().zip(parts) {
                let ty = match part {
                    "i32" => Type::I32,
                    "u32" => Type::U32,
                    "Bool" => Type::Bool,
                    "Array<i32>" => Type::ArrayI32,
                    "Array<u32>" => Type::ArrayU32,
                    other
                        if self.checker.records.contains_key(other)
                            || self.checker.variants.contains_key(other) =>
                    {
                        Type::Named(other.to_owned())
                    }
                    other => {
                        let qualified = self
                            .checker
                            .current_module
                            .as_ref()
                            .map(|module| format!("{module}::{other}"));
                        match qualified {
                            Some(name)
                                if self.checker.records.contains_key(&name)
                                    || self.checker.variants.contains_key(&name) =>
                            {
                                Type::Named(name)
                            }
                            _ => {
                                return Err(mismatch(
                                    "unsupported or unknown generic type argument",
                                    span,
                                ));
                            }
                        }
                    }
                };
                if !value_type(&ty) {
                    return Err(mismatch("generic type argument is not a Value", span));
                }
                mapping.insert(param.clone(), ty);
            }
        } else {
            for (arg, param) in args.iter().zip(&template.params) {
                let actual = self.expr(arg, None)?;
                infer(
                    &param.ty,
                    &actual,
                    &mut mapping,
                    &template.generic_params,
                    arg.span,
                )?;
            }
            if template
                .generic_params
                .iter()
                .any(|param| !mapping.contains_key(param))
            {
                return Err(mismatch(
                    "generic type argument cannot be inferred from arguments",
                    span,
                ));
            }
        }
        let types = template
            .generic_params
            .iter()
            .map(|param| mapping[param].clone())
            .collect::<Vec<_>>();
        let instance_name = format!(
            "{base}$<{}>",
            types
                .iter()
                .map(Type::source_name)
                .collect::<Vec<_>>()
                .join("|")
        );
        let instance = instantiate(&template, &mapping, instance_name.clone())?;
        let instance_result = instance.result.clone();
        if explicit.is_some() {
            for (arg, param) in args.iter().zip(&instance.params) {
                self.expr(arg, Some(&param.ty))?;
            }
        }
        if !self.checker.functions.contains_key(&instance_name) {
            if self.checker.functions.len() >= MAX_FUNCTIONS {
                return Err(FrontError::source(
                    "RESOURCE_LIMIT",
                    "lowered function or monomorphization limit exceeded",
                    span,
                ));
            }
            self.checker.functions.insert(
                instance_name.clone(),
                Signature {
                    params: instance
                        .params
                        .iter()
                        .map(|param| param.ty.clone())
                        .collect(),
                    result: instance.result.clone(),
                },
            );
            self.checker.generic_instances.insert(instance_name.clone());
            self.checker.pending_instances.push(instance);
        }
        self.checker.resolved_calls.insert(
            (self.checker.current_function.clone(), span),
            instance_name.clone(),
        );
        self.checker
            .call_edges
            .entry(self.checker.current_function.clone())
            .or_default()
            .insert(instance_name);
        Ok(Some(instance_result))
    }
}

pub(super) fn rewrite_calls(
    block: &mut Block,
    function: &str,
    calls: &HashMap<(String, Span), String>,
) {
    fn rewrite_expr(expr: &mut Expr, function: &str, calls: &HashMap<(String, Span), String>) {
        if let ExprKind::Call { name, args } = &mut expr.kind {
            if let Some(resolved) = calls.get(&(function.to_owned(), expr.span)) {
                *name = resolved.clone();
            }
            for arg in args {
                rewrite_expr(arg, function, calls);
            }
            return;
        }
        match &mut expr.kind {
            ExprKind::Array(items) => {
                for item in items {
                    rewrite_expr(item, function, calls);
                }
            }
            ExprKind::Record { fields, .. } => {
                for field in fields {
                    rewrite_expr(&mut field.value, function, calls);
                }
            }
            ExprKind::Match { value, arms } => {
                rewrite_expr(value, function, calls);
                for arm in arms {
                    rewrite_expr(&mut arm.value, function, calls);
                }
            }
            ExprKind::Index { base, index }
            | ExprKind::Binary {
                left: base,
                right: index,
                ..
            } => {
                rewrite_expr(base, function, calls);
                rewrite_expr(index, function, calls);
            }
            ExprKind::MoveField { base, .. }
            | ExprKind::Field { base, .. }
            | ExprKind::Unary { value: base, .. }
            | ExprKind::Group(base)
            | ExprKind::Try(base) => rewrite_expr(base, function, calls),
            ExprKind::Integer(_)
            | ExprKind::Float(_)
            | ExprKind::Bool(_)
            | ExprKind::Text(_)
            | ExprKind::Bytes(_)
            | ExprKind::Name(_)
            | ExprKind::Move(_)
            | ExprKind::Call { .. } => {}
        }
    }
    fn rewrite_statement(
        statement: &mut Stmt,
        function: &str,
        calls: &HashMap<(String, Span), String>,
    ) {
        match &mut statement.kind {
            StmtKind::Binding { value, .. }
            | StmtKind::Expr(value)
            | StmtKind::Return(Some(value)) => rewrite_expr(value, function, calls),
            StmtKind::Assign { place, value } => {
                if let Place::Index { index, .. } = place {
                    rewrite_expr(index, function, calls);
                }
                rewrite_expr(value, function, calls);
            }
            StmtKind::If {
                condition,
                then_block,
                else_arm,
            } => {
                rewrite_expr(condition, function, calls);
                rewrite_calls(then_block, function, calls);
                match else_arm {
                    Some(ElseArm::Block(block)) => rewrite_calls(block, function, calls),
                    Some(ElseArm::If(statement)) => rewrite_statement(statement, function, calls),
                    None => {}
                }
            }
            StmtKind::While { condition, body } => {
                rewrite_expr(condition, function, calls);
                rewrite_calls(body, function, calls);
            }
            StmtKind::For {
                lower, upper, body, ..
            } => {
                rewrite_expr(lower, function, calls);
                rewrite_expr(upper, function, calls);
                rewrite_calls(body, function, calls);
            }
            StmtKind::Return(None) => {}
        }
    }
    for statement in &mut block.statements {
        rewrite_statement(statement, function, calls);
    }
}

impl Checker {
    pub(super) fn check_generic_graph(&self) -> Result<(), FrontError> {
        fn visit(
            name: &str,
            checker: &Checker,
            stack: &mut Vec<String>,
            seen: &mut HashSet<(String, usize)>,
            depth: usize,
        ) -> Result<(), FrontError> {
            if let Some(start) = stack.iter().position(|item| item == name) {
                if stack[start..]
                    .iter()
                    .any(|item| checker.generic_instances.contains(item))
                {
                    return Err(FrontError::source(
                        "UNSUPPORTED_FEATURE",
                        "recursive generic instantiation",
                        checker.generic_span(name),
                    ));
                }
                return Ok(());
            }
            let depth = depth + usize::from(checker.generic_instances.contains(name));
            if depth > MAX_GENERIC_DEPTH {
                return Err(FrontError::source(
                    "RESOURCE_LIMIT",
                    "generic expansion depth exceeds 16",
                    checker.generic_span(name),
                ));
            }
            if !seen.insert((name.to_owned(), depth)) {
                return Ok(());
            }
            stack.push(name.to_owned());
            if let Some(edges) = checker.call_edges.get(name) {
                for next in edges {
                    visit(next, checker, stack, seen, depth)?;
                }
            }
            stack.pop();
            Ok(())
        }
        let mut seen = HashSet::new();
        for name in self.call_edges.keys() {
            visit(name, self, &mut Vec::new(), &mut seen, 0)?;
        }
        Ok(())
    }

    fn generic_span(&self, name: &str) -> Span {
        let (base, _) = split_call(name);
        self.generic_templates
            .get(base)
            .map_or(Span { start: 0, end: 0 }, |template| template.span)
    }
}
