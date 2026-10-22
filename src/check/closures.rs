//! Local affine closures lower to ordinary functions plus explicit capture bindings.

use super::*;
use crate::front::{ClosureCapture, ClosureDecl, Param};

const MAX_CAPTURES: usize = 8;
const MAX_PARAMS: usize = 8;
const MAX_CLOSURE_BYTES: usize = 65_536;
const MAX_FUNCTIONS: usize = 128;

#[derive(Clone)]
struct LiveClosure {
    hidden: String,
    captures: Vec<ClosureCapture>,
    called: bool,
}

fn unsupported(message: &str, span: Span) -> FrontError {
    FrontError::source("UNSUPPORTED_FEATURE", message, span)
}

fn affine(ty: &Type) -> bool {
    matches!(
        ty,
        Type::ArrayI32 | Type::ArrayU32 | Type::ArrayF32 | Type::ArrayF64 | Type::ArrayBool
    ) || matches!(ty, Type::Named(name) if crate::front::view_parts(name).is_none())
}

fn borrowed(active: &HashMap<String, LiveClosure>, inherited: &HashSet<String>) -> HashSet<String> {
    let mut roots = inherited.clone();
    for closure in active.values().filter(|closure| !closure.called) {
        roots.extend(
            closure
                .captures
                .iter()
                .filter(|capture| capture.mode == Some(Mode::Read))
                .map(|capture| capture.name.clone()),
        );
    }
    roots
}

fn scan_expr(expr: &Expr, roots: &HashSet<String>) -> Result<(), FrontError> {
    match &expr.kind {
        ExprKind::Name(name) | ExprKind::Move(name) if roots.contains(name) => {
            Err(FrontError::source(
                "OWN_BORROW_CONFLICT",
                "captured read owner is unavailable until closure invocation or scope exit",
                expr.span,
            ))
        }
        ExprKind::Call { args, .. } | ExprKind::Array(args) => {
            for arg in args {
                scan_expr(arg, roots)?;
            }
            Ok(())
        }
        ExprKind::Record { fields, .. } => {
            for field in fields {
                scan_expr(&field.value, roots)?;
            }
            Ok(())
        }
        ExprKind::Match { value, arms } => {
            scan_expr(value, roots)?;
            for arm in arms {
                scan_expr(&arm.value, roots)?;
            }
            Ok(())
        }
        ExprKind::Index { base, index }
        | ExprKind::Binary {
            left: base,
            right: index,
            ..
        } => {
            scan_expr(base, roots)?;
            scan_expr(index, roots)
        }
        ExprKind::MoveField { base, .. }
        | ExprKind::Field { base, .. }
        | ExprKind::Unary { value: base, .. }
        | ExprKind::Group(base)
        | ExprKind::Try(base) => scan_expr(base, roots),
        ExprKind::Integer(_)
        | ExprKind::Float(_)
        | ExprKind::Bool(_)
        | ExprKind::Text(_)
        | ExprKind::Bytes(_)
        | ExprKind::Name(_)
        | ExprKind::Move(_) => Ok(()),
    }
}

fn scan_statement(statement: &Stmt, roots: &HashSet<String>) -> Result<(), FrontError> {
    match &statement.kind {
        StmtKind::Binding { value, .. } | StmtKind::Expr(value) | StmtKind::Return(Some(value)) => {
            scan_expr(value, roots)
        }
        StmtKind::Assign { place, value } => {
            let name = match place {
                Place::Name(name, _) | Place::Index { name, .. } => name,
            };
            if roots.contains(name) {
                return Err(FrontError::source(
                    "OWN_BORROW_CONFLICT",
                    "captured read owner cannot be assigned",
                    statement.span,
                ));
            }
            if let Place::Index { index, .. } = place {
                scan_expr(index, roots)?;
            }
            scan_expr(value, roots)
        }
        StmtKind::If { condition, .. } | StmtKind::While { condition, .. } => {
            scan_expr(condition, roots)
        }
        StmtKind::For { lower, upper, .. } => {
            scan_expr(lower, roots)?;
            scan_expr(upper, roots)
        }
        StmtKind::Return(None) => Ok(()),
    }
}

fn rewrite_expr(
    expr: &mut Expr,
    active: &mut HashMap<String, LiveClosure>,
    inherited_names: &HashSet<String>,
    allow_call: bool,
) -> Result<(), FrontError> {
    match &mut expr.kind {
        ExprKind::Call { name, args } => {
            for arg in args.iter_mut() {
                rewrite_expr(arg, active, inherited_names, allow_call)?;
            }
            if let Some(closure) = active.get_mut(name) {
                if !allow_call {
                    return Err(unsupported(
                        "closure invocation is limited to straight-line statements",
                        expr.span,
                    ));
                }
                if closure.called {
                    return Err(FrontError::source(
                        "OWN_USE_AFTER_MOVE",
                        "affine closure was already invoked",
                        expr.span,
                    ));
                }
                closure.called = true;
                let mut captured = Vec::new();
                for (index, capture) in closure.captures.iter().enumerate() {
                    let kind = match capture.mode {
                        None => ExprKind::Name(format!("$cap@{}@{index}", closure.hidden)),
                        Some(Mode::Read) => ExprKind::Name(capture.name.clone()),
                        Some(Mode::Take) => {
                            ExprKind::Move(format!("$cap@{}@{index}", closure.hidden))
                        }
                        Some(Mode::Mut) => {
                            return Err(unsupported(
                                "mutable closure capture is unsupported",
                                capture.span,
                            ));
                        }
                    };
                    captured.push(Expr::new(kind, capture.span)?);
                }
                captured.append(args);
                *args = captured;
                *name = closure.hidden.clone();
            } else if inherited_names.contains(name) {
                return Err(unsupported(
                    "closure must be invoked in its declaration block",
                    expr.span,
                ));
            } else if name.starts_with("$closure@") {
                return Err(unsupported(
                    "closure expression must bind directly to a local let",
                    expr.span,
                ));
            }
            Ok(())
        }
        ExprKind::Name(name) | ExprKind::Move(name)
            if active.contains_key(name) || inherited_names.contains(name) =>
        {
            Err(unsupported(
                "closure cannot escape or be used as a value",
                expr.span,
            ))
        }
        ExprKind::Array(items) => {
            for item in items {
                rewrite_expr(item, active, inherited_names, allow_call)?;
            }
            Ok(())
        }
        ExprKind::Record { fields, .. } => {
            for field in fields {
                rewrite_expr(&mut field.value, active, inherited_names, allow_call)?;
            }
            Ok(())
        }
        ExprKind::Match { value, arms } => {
            rewrite_expr(value, active, inherited_names, allow_call)?;
            for arm in arms {
                rewrite_expr(&mut arm.value, active, inherited_names, allow_call)?;
            }
            Ok(())
        }
        ExprKind::Index { base, index }
        | ExprKind::Binary {
            left: base,
            right: index,
            ..
        } => {
            rewrite_expr(base, active, inherited_names, allow_call)?;
            rewrite_expr(index, active, inherited_names, allow_call)
        }
        ExprKind::MoveField { base, .. }
        | ExprKind::Field { base, .. }
        | ExprKind::Unary { value: base, .. }
        | ExprKind::Group(base)
        | ExprKind::Try(base) => rewrite_expr(base, active, inherited_names, allow_call),
        ExprKind::Integer(_)
        | ExprKind::Float(_)
        | ExprKind::Bool(_)
        | ExprKind::Text(_)
        | ExprKind::Bytes(_)
        | ExprKind::Name(_)
        | ExprKind::Move(_) => Ok(()),
    }
}

fn contains_view_call(block: &Block) -> bool {
    fn expr(value: &Expr) -> bool {
        match &value.kind {
            ExprKind::Call { name, args } => {
                matches!(name.as_str(), "view" | "view_mut") || args.iter().any(expr)
            }
            ExprKind::Array(items) => items.iter().any(expr),
            ExprKind::Record { fields, .. } => fields.iter().any(|field| expr(&field.value)),
            ExprKind::Match { value, arms } => {
                expr(value) || arms.iter().any(|arm| expr(&arm.value))
            }
            ExprKind::Index { base, index }
            | ExprKind::Binary {
                left: base,
                right: index,
                ..
            } => expr(base) || expr(index),
            ExprKind::MoveField { base, .. }
            | ExprKind::Field { base, .. }
            | ExprKind::Unary { value: base, .. }
            | ExprKind::Group(base)
            | ExprKind::Try(base) => expr(base),
            _ => false,
        }
    }
    fn statement(stmt: &Stmt) -> bool {
        match &stmt.kind {
            StmtKind::Binding { value, .. }
            | StmtKind::Expr(value)
            | StmtKind::Return(Some(value)) => expr(value),
            StmtKind::Assign { place, value } => {
                expr(value) || matches!(place, Place::Index { index, .. } if expr(index))
            }
            StmtKind::If {
                condition,
                then_block,
                else_arm,
            } => {
                expr(condition)
                    || contains_view_call(then_block)
                    || match else_arm {
                        Some(ElseArm::Block(block)) => contains_view_call(block),
                        Some(ElseArm::If(next)) => statement(next),
                        None => false,
                    }
            }
            StmtKind::While { condition, body } => expr(condition) || contains_view_call(body),
            StmtKind::For {
                lower, upper, body, ..
            } => expr(lower) || expr(upper) || contains_view_call(body),
            StmtKind::Return(None) => false,
        }
    }
    block.statements.iter().any(statement)
}

fn create_closure(
    declaration: &ClosureDecl,
    source: &Function,
    environment: &HashMap<String, Option<Type>>,
    borrowed_roots: &HashSet<String>,
    view_calls: bool,
    produced: &mut Vec<Function>,
) -> Result<(LiveClosure, Vec<Stmt>), FrontError> {
    if !source.generic_params.is_empty() {
        return Err(unsupported(
            "generic closures are outside L1-03",
            declaration.span,
        ));
    }
    if declaration.captures.len() > MAX_CAPTURES
        || declaration.params.len() > MAX_PARAMS
        || declaration.span.end.saturating_sub(declaration.span.start) > MAX_CLOSURE_BYTES
    {
        return Err(FrontError::source(
            "RESOURCE_LIMIT",
            "closure size, capture, or parameter limit exceeded",
            declaration.span,
        ));
    }
    if produced.len() >= MAX_FUNCTIONS {
        return Err(FrontError::source(
            "RESOURCE_LIMIT",
            "closure function limit exceeded",
            declaration.span,
        ));
    }
    let hidden = source.module.as_ref().map_or_else(
        || declaration.name.clone(),
        |module| format!("{module}::{}", declaration.name),
    );
    let mut names = HashSet::new();
    let mut setup = Vec::new();
    for (index, capture) in declaration.captures.iter().enumerate() {
        // ponytail: explicit binding types avoid a second inference pass; widen when closures need inferred locals.
        if !names.insert(capture.name.as_str())
            || environment.get(&capture.name) != Some(&Some(capture.ty.clone()))
        {
            return Err(mismatch(
                "capture name or declared type does not match an active binding",
                capture.span,
            ));
        }
        match capture.mode {
            Some(Mode::Mut) => {
                return Err(unsupported(
                    "mutable closure capture is unsupported",
                    capture.span,
                ));
            }
            None if !matches!(capture.ty, Type::I32 | Type::U32 | Type::Bool) => {
                return Err(mismatch(
                    "copy capture requires an integer or Bool",
                    capture.span,
                ));
            }
            Some(Mode::Read | Mode::Take) if !affine(&capture.ty) => {
                return Err(mismatch(
                    "borrowed or taken capture requires an affine owner",
                    capture.span,
                ));
            }
            Some(Mode::Read) if view_calls => {
                // ponytail: captured loans stay separate from view alias facts until both share one checker.
                return Err(unsupported(
                    "read closure capture and views cannot share a function in this subset",
                    capture.span,
                ));
            }
            Some(Mode::Take) if borrowed_roots.contains(&capture.name) => {
                return Err(FrontError::source(
                    "OWN_BORROW_CONFLICT",
                    "cannot take an owner borrowed by a live closure",
                    capture.span,
                ));
            }
            _ => {}
        }
        if capture.mode != Some(Mode::Read) {
            let local = format!("$cap@{hidden}@{index}");
            let value = Expr::new(
                if capture.mode == Some(Mode::Take) {
                    ExprKind::Move(capture.name.clone())
                } else {
                    ExprKind::Name(capture.name.clone())
                },
                capture.span,
            )?;
            setup.push(Stmt::new(
                StmtKind::Binding {
                    mutable: false,
                    name: local,
                    annotation: Some(capture.ty.clone()),
                    value,
                },
                capture.span,
            )?);
        }
    }
    for param in &declaration.params {
        if !names.insert(param.name.as_str()) {
            return Err(FrontError::source(
                "NAME_DUPLICATE",
                "closure parameter shadows a capture",
                param.span,
            ));
        }
    }
    let mut params = declaration
        .captures
        .iter()
        .map(|capture| Param {
            name: capture.name.clone(),
            mode: capture.mode,
            ty: capture.ty.clone(),
            span: capture.span,
        })
        .collect::<Vec<_>>();
    params.extend(declaration.params.iter().cloned());
    produced.push(Function {
        name: hidden.clone(),
        module: source.module.clone(),
        public: false,
        generic_params: Vec::new(),
        params,
        result: declaration.result.clone(),
        return_from: None,
        body: declaration.body.clone(),
        span: declaration.span,
        // A closure's folds stay strict: no tier reaches it from its encloser.
        fold: crate::numeric_contract::FoldTier::Strict,
    });
    Ok((
        LiveClosure {
            hidden,
            captures: declaration.captures.clone(),
            called: false,
        },
        setup,
    ))
}

#[allow(clippy::too_many_arguments)]
fn rewrite_block(
    block: &mut Block,
    source: &Function,
    declarations: &HashMap<String, ClosureDecl>,
    environment: &HashMap<String, Option<Type>>,
    inherited_roots: &HashSet<String>,
    inherited_names: &HashSet<String>,
    view_calls: bool,
    produced: &mut Vec<Function>,
    used: &mut HashSet<String>,
) -> Result<(), FrontError> {
    let mut environment = environment.clone();
    let mut active = HashMap::<String, LiveClosure>::new();
    let mut rewritten = Vec::new();
    for mut statement in std::mem::take(&mut block.statements) {
        if let StmtKind::Binding {
            mutable: false,
            name,
            annotation: None,
            value:
                Expr {
                    kind:
                        ExprKind::Call {
                            name: declaration,
                            args,
                        },
                    ..
                },
        } = &statement.kind
            && args.is_empty()
            && let Some(declaration) = declarations.get(declaration)
        {
            if active.contains_key(name)
                || inherited_names.contains(name)
                || environment.contains_key(name)
            {
                return Err(FrontError::source(
                    "NAME_DUPLICATE",
                    "closure binding shadows another name",
                    statement.span,
                ));
            }
            let (closure, setup) = create_closure(
                declaration,
                source,
                &environment,
                &borrowed(&active, inherited_roots),
                view_calls,
                produced,
            )?;
            rewritten.extend(setup);
            active.insert(name.clone(), closure);
            used.insert(declaration.name.clone());
            continue;
        }
        let roots = borrowed(&active, inherited_roots);
        scan_statement(&statement, &roots)?;
        match &mut statement.kind {
            StmtKind::Binding {
                name,
                annotation,
                value,
                ..
            } => {
                if active.contains_key(name) || inherited_names.contains(name) {
                    return Err(FrontError::source(
                        "NAME_DUPLICATE",
                        "binding shadows a closure",
                        statement.span,
                    ));
                }
                rewrite_expr(value, &mut active, inherited_names, true)?;
                environment.insert(name.clone(), annotation.clone());
            }
            StmtKind::Expr(value) | StmtKind::Return(Some(value)) => {
                rewrite_expr(value, &mut active, inherited_names, true)?
            }
            StmtKind::Assign { place, value } => {
                let name = match place {
                    Place::Name(name, _) | Place::Index { name, .. } => name,
                };
                if active.contains_key(name) || inherited_names.contains(name) {
                    return Err(unsupported(
                        "closure binding cannot be assigned",
                        statement.span,
                    ));
                }
                if let Place::Index { index, .. } = place {
                    rewrite_expr(index, &mut active, inherited_names, false)?;
                }
                rewrite_expr(value, &mut active, inherited_names, true)?;
            }
            StmtKind::If {
                condition,
                then_block,
                else_arm,
            } => {
                rewrite_expr(condition, &mut active, inherited_names, false)?;
                let roots = borrowed(&active, inherited_roots);
                let names = inherited_names
                    .iter()
                    .cloned()
                    .chain(active.keys().cloned())
                    .collect();
                rewrite_block(
                    then_block,
                    source,
                    declarations,
                    &environment,
                    &roots,
                    &names,
                    view_calls,
                    produced,
                    used,
                )?;
                match else_arm {
                    Some(ElseArm::Block(block)) => rewrite_block(
                        block,
                        source,
                        declarations,
                        &environment,
                        &roots,
                        &names,
                        view_calls,
                        produced,
                        used,
                    )?,
                    Some(ElseArm::If(statement)) => {
                        let mut fake = Block::new(vec![*statement.clone()], statement.span)?;
                        rewrite_block(
                            &mut fake,
                            source,
                            declarations,
                            &environment,
                            &roots,
                            &names,
                            view_calls,
                            produced,
                            used,
                        )?;
                        **statement = fake.statements.remove(0);
                    }
                    None => {}
                }
            }
            StmtKind::While { condition, body } => {
                rewrite_expr(condition, &mut active, inherited_names, false)?;
                let roots = borrowed(&active, inherited_roots);
                let names = inherited_names
                    .iter()
                    .cloned()
                    .chain(active.keys().cloned())
                    .collect();
                rewrite_block(
                    body,
                    source,
                    declarations,
                    &environment,
                    &roots,
                    &names,
                    view_calls,
                    produced,
                    used,
                )?;
            }
            StmtKind::For {
                name,
                lower,
                upper,
                body,
            } => {
                rewrite_expr(lower, &mut active, inherited_names, false)?;
                rewrite_expr(upper, &mut active, inherited_names, false)?;
                let roots = borrowed(&active, inherited_roots);
                let names = inherited_names
                    .iter()
                    .cloned()
                    .chain(active.keys().cloned())
                    .collect();
                let mut inner = environment.clone();
                inner.insert(name.clone(), Some(Type::U32));
                rewrite_block(
                    body,
                    source,
                    declarations,
                    &inner,
                    &roots,
                    &names,
                    view_calls,
                    produced,
                    used,
                )?;
            }
            StmtKind::Return(None) => {}
        }
        rewritten.push(statement);
    }
    block.statements = rewritten;
    Ok(())
}

pub(super) fn desugar(program: &mut Program) -> Result<(), FrontError> {
    if program.closures.is_empty() {
        return Ok(());
    }
    let declarations = program
        .closures
        .iter()
        .map(|closure| (closure.name.clone(), closure.clone()))
        .collect::<HashMap<_, _>>();
    let mut produced = Vec::new();
    let mut used = HashSet::new();
    let functions = program
        .functions
        .iter()
        .map(|function| {
            (
                function.module.clone(),
                function
                    .name
                    .rsplit("::")
                    .next()
                    .unwrap_or(&function.name)
                    .to_owned(),
            )
        })
        .collect::<Vec<_>>();
    let imports = program
        .imports
        .iter()
        .map(|import| (import.owner.clone(), import.item.clone()))
        .collect::<Vec<_>>();
    for function in &mut program.functions {
        let source = function.clone();
        let mut environment = functions
            .iter()
            .chain(&imports)
            .filter(|(module, _)| module.as_ref() == function.module.as_ref())
            .map(|(_, name)| (name.clone(), None))
            .collect::<HashMap<_, _>>();
        environment.extend(
            [
                "len",
                "view",
                "view_mut",
                "filled_i32",
                "filled_u32",
                "wrapping_add",
                "wrapping_mul",
            ]
            .map(|name| (name.to_owned(), None)),
        );
        environment.extend(
            function
                .params
                .iter()
                .map(|param| (param.name.clone(), Some(param.ty.clone()))),
        );
        let view_calls = contains_view_call(&function.body);
        rewrite_block(
            &mut function.body,
            &source,
            &declarations,
            &environment,
            &HashSet::new(),
            &HashSet::new(),
            view_calls,
            &mut produced,
            &mut used,
        )?;
    }
    if let Some(unused) = program
        .closures
        .iter()
        .find(|declaration| !used.contains(&declaration.name))
    {
        return Err(unsupported(
            "closure expression must bind directly in an ordinary function",
            unused.span,
        ));
    }
    if program.functions.len().saturating_add(produced.len()) > MAX_FUNCTIONS {
        return Err(FrontError::source(
            "RESOURCE_LIMIT",
            "closure or function count exceeds 128",
            program
                .closures
                .last()
                .map_or(Span { start: 0, end: 0 }, |item| item.span),
        ));
    }
    program.functions.extend(produced);
    program.closures.clear();
    Ok(())
}
