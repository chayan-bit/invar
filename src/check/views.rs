//! Lexical view facts: roots, physical intervals, aliases, and return provenance.

use std::collections::HashMap;

use crate::{
    check::CheckedProgram,
    front::{
        Block, ElseArm, Expr, ExprKind, FrontError, Function, Mode, Place, Span, Stmt, StmtKind,
        Type,
    },
};

#[derive(Clone)]
struct Fact {
    root: String,
    // Half-open conservative physical interval. Unknown bounds cannot prove disjointness.
    interval: Option<(u64, u64)>,
    stride: Option<u64>,
    mutable: bool,
    parent: Option<String>,
}

struct Scan<'a> {
    program: &'a CheckedProgram,
    function: &'a Function,
    types: HashMap<Span, Type>,
    bindings: HashMap<String, (Type, bool, Option<Mode>)>,
    views: HashMap<String, Fact>,
}

fn fail(code: &'static str, message: &str, span: Span) -> FrontError {
    FrontError::source(code, message, span)
}

fn literal(expr: &Expr) -> Option<u64> {
    match &expr.kind {
        ExprKind::Integer(raw) => raw.strip_suffix("u32").unwrap_or(raw).parse().ok(),
        ExprKind::Group(inner) => literal(inner),
        _ => None,
    }
}

pub(super) fn check(program: &CheckedProgram) -> Result<(), FrontError> {
    for function in &program.program.functions {
        let mut scan = Scan {
            program,
            function,
            types: program.types_for(&function.name),
            bindings: function
                .params
                .iter()
                .map(|p| (p.name.clone(), (p.ty.clone(), false, p.mode)))
                .collect(),
            views: HashMap::new(),
        };
        scan.block(&function.body)?;
    }
    Ok(())
}

impl Scan<'_> {
    fn is_view(&self, expr: &Expr) -> bool {
        matches!(self.types.get(&expr.span), Some(Type::Named(name)) if crate::front::view_parts(name).is_some())
    }

    fn active<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Fact> + 'a {
        self.views.values().filter(move |view| view.root == name)
    }

    fn source(&self, name: &str, span: Span) -> Result<Fact, FrontError> {
        if let Some(view) = self.views.get(name) {
            if self
                .views
                .values()
                .any(|other| other.parent.as_deref() == Some(name))
            {
                return Err(fail(
                    "OWN_BORROW_CONFLICT",
                    "view is reborrowed in this scope",
                    span,
                ));
            }
            return Ok(view.clone());
        }
        let Some((ty, mutable, mode)) = self.bindings.get(name) else {
            return Err(fail(
                "OWN_BORROW_SOURCE",
                "view source must be a named owner",
                span,
            ));
        };
        if !matches!(ty, Type::ArrayI32 | Type::ArrayU32) {
            return Err(fail(
                "OWN_BORROW_SOURCE",
                "view source must be an integer array or view",
                span,
            ));
        }
        Ok(Fact {
            root: name.to_owned(),
            interval: None,
            stride: None,
            mutable: *mutable || *mode == Some(Mode::Mut),
            parent: None,
        })
    }

    fn conflict(&self, fact: &Fact, span: Span) -> Result<(), FrontError> {
        for existing in self.active(&fact.root) {
            if existing.parent == fact.parent
                && existing.interval == fact.interval
                && !fact.mutable
                && !existing.mutable
            {
                continue;
            }
            // A reborrow suspends its whole ancestor chain; siblings still need disjoint bounds.
            let mut ancestor = fact.parent.as_deref();
            let mut suspended = false;
            while let Some(name) = ancestor {
                let parent = self.views.get(name).expect("view parent is active");
                if std::ptr::eq(parent, existing) {
                    suspended = true;
                    break;
                }
                ancestor = parent.parent.as_deref();
            }
            if suspended {
                continue;
            }
            let disjoint = matches!((fact.interval, existing.interval), (Some((a,b)), Some((c,d))) if a >= d || c >= b || a == b || c == d);
            if !disjoint && (fact.mutable || existing.mutable) {
                return Err(fail(
                    "OWN_BORROW_CONFLICT",
                    "overlapping or unproved mutable views",
                    span,
                ));
            }
        }
        Ok(())
    }

    fn derive(&mut self, expr: &Expr) -> Result<Fact, FrontError> {
        match &expr.kind {
            ExprKind::Name(name) => self.views.get(name).cloned().ok_or_else(|| {
                fail(
                    "OWN_BORROW_SOURCE",
                    "view value needs a named view",
                    expr.span,
                )
            }),
            ExprKind::Call { name, args } if matches!(name.as_str(), "view" | "view_mut") => {
                let ExprKind::Name(source) = &args[0].kind else {
                    return Err(fail(
                        "OWN_BORROW_SOURCE",
                        "view needs a named owner",
                        args[0].span,
                    ));
                };
                let base = self.source(source, args[0].span)?;
                let mutable = name == "view_mut";
                if mutable && !base.mutable {
                    return Err(fail(
                        "OWN_MUT_REQUIRED",
                        "mutable view needs a mutable owner",
                        expr.span,
                    ));
                }
                for bound in args.iter().skip(1) {
                    self.expr(bound)?;
                }
                let start = literal(&args[1]);
                let end = literal(&args[2]);
                let step =
                    args.get(3)
                        .and_then(literal)
                        .or(if args.len() == 3 { Some(1) } else { None });
                let interval = match (base.interval, base.stride, start, end, step) {
                    (Some((lo, _)), Some(parent_stride), Some(start), Some(end), Some(step))
                        if step > 0 && end >= start =>
                    {
                        let first = lo
                            .checked_add(start.checked_mul(parent_stride).ok_or_else(|| {
                                fail(
                                    "TYPE_MISMATCH",
                                    "view range arithmetic exceeds u32",
                                    expr.span,
                                )
                            })?)
                            .ok_or_else(|| {
                                fail(
                                    "TYPE_MISMATCH",
                                    "view range arithmetic exceeds u32",
                                    expr.span,
                                )
                            })?;
                        let count = (end - start).div_ceil(step);
                        let upper = if count == 0 {
                            first
                        } else {
                            first
                                .checked_add(
                                    (count - 1)
                                        .checked_mul(parent_stride.checked_mul(step).ok_or_else(
                                            || {
                                                fail(
                                                    "TYPE_MISMATCH",
                                                    "view stride arithmetic exceeds u32",
                                                    expr.span,
                                                )
                                            },
                                        )?)
                                        .ok_or_else(|| {
                                            fail(
                                                "TYPE_MISMATCH",
                                                "view range arithmetic exceeds u32",
                                                expr.span,
                                            )
                                        })?,
                                )
                                .and_then(|last| last.checked_add(1))
                                .ok_or_else(|| {
                                    fail(
                                        "TYPE_MISMATCH",
                                        "view range arithmetic exceeds u32",
                                        expr.span,
                                    )
                                })?
                        };
                        Some((first, upper))
                    }
                    (None, _, Some(start), Some(end), Some(step))
                        if !self.views.contains_key(source) && step > 0 && end >= start =>
                    {
                        let count = (end - start).div_ceil(step);
                        Some((
                            start,
                            if count == 0 {
                                start
                            } else {
                                start + (count - 1) * step + 1
                            },
                        ))
                    }
                    _ => None,
                };
                let stride = match (base.stride, step) {
                    (Some(parent), Some(step)) => parent.checked_mul(step),
                    (None, Some(step)) if !self.views.contains_key(source) => Some(step),
                    _ => None,
                };
                let fact = Fact {
                    root: base.root,
                    interval,
                    stride,
                    mutable,
                    parent: self.views.contains_key(source).then(|| source.clone()),
                };
                self.conflict(&fact, expr.span)?;
                Ok(fact)
            }
            ExprKind::Call { name, args } => {
                let callee = self.program.program.function(name).ok_or_else(|| {
                    fail(
                        "OWN_BORROW_SOURCE",
                        "view-producing call is unknown",
                        expr.span,
                    )
                })?;
                let source = callee.return_from.as_ref().ok_or_else(|| {
                    fail(
                        "OWN_RETURN_BORROW",
                        "callee does not declare view return source",
                        expr.span,
                    )
                })?;
                let index = callee
                    .params
                    .iter()
                    .position(|p| p.name == *source)
                    .ok_or_else(|| {
                        fail("OWN_RETURN_BORROW", "invalid view return source", expr.span)
                    })?;
                for (position, arg) in args.iter().enumerate() {
                    if position != index {
                        self.expr(arg)?;
                    }
                }
                let ExprKind::Name(actual) = &args[index].kind else {
                    return Err(fail(
                        "OWN_BORROW_SOURCE",
                        "view return call needs a named owner",
                        args[index].span,
                    ));
                };
                let mut fact = self.source(actual, args[index].span)?;
                fact.mutable = matches!(&callee.result, Type::Named(name) if crate::front::view_parts(name).is_some_and(|(_, mutable)| mutable));
                fact.interval = None;
                fact.stride = None;
                fact.parent = self.views.contains_key(actual).then(|| actual.clone());
                self.conflict(&fact, expr.span)?;
                Ok(fact)
            }
            ExprKind::Group(inner) => self.derive(inner),
            _ => Err(fail(
                "OWN_BORROW_SOURCE",
                "view needs a named source or declared return",
                expr.span,
            )),
        }
    }

    fn read(&self, name: &str, span: Span) -> Result<(), FrontError> {
        if self.views.contains_key(name) {
            self.source(name, span)?;
        } else if self.active(name).any(|view| view.mutable) {
            return Err(fail(
                "OWN_BORROW_CONFLICT",
                "mutable view excludes owner access",
                span,
            ));
        }
        Ok(())
    }

    fn write(&self, name: &str, span: Span) -> Result<(), FrontError> {
        if self.views.contains_key(name) {
            self.source(name, span)?;
        } else if self.active(name).next().is_some() {
            return Err(fail(
                "OWN_BORROW_CONFLICT",
                "live view excludes owner mutation or move",
                span,
            ));
        }
        Ok(())
    }

    /// Kept out of [`Self::expr`] so the recursive scan's unoptimized frame
    /// stays within the 2 MiB height budget.
    #[inline(never)]
    fn sort_args(&self, args: &[Expr], modes: &[Mode]) -> Result<(), FrontError> {
        for (arg, mode) in args.iter().zip(modes) {
            let ExprKind::Name(actual) = &arg.kind else {
                return Err(fail(
                    "OWN_BORROW_SOURCE",
                    "sort requires named arrays",
                    arg.span,
                ));
            };
            if *mode == Mode::Read {
                self.read(actual, arg.span)?;
            } else {
                self.write(actual, arg.span)?;
            }
        }
        Ok(())
    }

    fn expr(&mut self, expr: &Expr) -> Result<(), FrontError> {
        if self.is_view(expr) && matches!(&expr.kind, ExprKind::Call { .. }) {
            return Err(fail(
                "UNSUPPORTED_FEATURE",
                "view results must bind or return directly",
                expr.span,
            ));
        }
        match &expr.kind {
            ExprKind::Name(name) => self.read(name, expr.span),
            ExprKind::Move(name) => self.write(name, expr.span),
            ExprKind::Call { name, args } => {
                if let Some(modes) = crate::check::sort_modes(name) {
                    return self.sort_args(args, modes);
                }
                if let Some(callee) = self.program.program.function(name) {
                    for (arg, param) in args.iter().zip(&callee.params) {
                        if let ExprKind::Name(actual) = &arg.kind {
                            if param.mode == Some(Mode::Read) {
                                self.read(actual, arg.span)?;
                            } else if param.mode.is_some() {
                                self.write(actual, arg.span)?;
                            } else {
                                self.expr(arg)?;
                            }
                        } else {
                            self.expr(arg)?;
                        }
                    }
                } else {
                    for arg in args {
                        self.expr(arg)?;
                    }
                }
                Ok(())
            }
            ExprKind::Index { base, index } => {
                self.expr(base)?;
                self.expr(index)
            }
            ExprKind::Field { base, .. }
            | ExprKind::MoveField { base, .. }
            | ExprKind::Group(base)
            | ExprKind::Try(base)
            | ExprKind::Unary { value: base, .. } => self.expr(base),
            ExprKind::Binary { left, right, .. } => {
                self.expr(left)?;
                self.expr(right)
            }
            ExprKind::Array(items) => {
                for item in items {
                    self.expr(item)?;
                }
                Ok(())
            }
            ExprKind::Record { fields, .. } => {
                for field in fields {
                    self.expr(&field.value)?;
                }
                Ok(())
            }
            ExprKind::Match { value, arms } => {
                self.expr(value)?;
                for arm in arms {
                    self.expr(&arm.value)?;
                }
                Ok(())
            }
            ExprKind::Integer(_)
            | ExprKind::Float(_)
            | ExprKind::Bool(_)
            | ExprKind::Text(_)
            | ExprKind::Bytes(_) => Ok(()),
        }
    }

    fn block(&mut self, block: &Block) -> Result<(), FrontError> {
        let mut declared = Vec::new();
        for statement in &block.statements {
            self.statement(statement, &mut declared)?;
        }
        for name in declared {
            self.views.remove(&name);
            self.bindings.remove(&name);
        }
        Ok(())
    }

    fn statement(
        &mut self,
        statement: &Stmt,
        declared: &mut Vec<String>,
    ) -> Result<(), FrontError> {
        match &statement.kind {
            StmtKind::Binding {
                mutable,
                name,
                value,
                ..
            } => {
                let ty = self.types.get(&value.span).cloned().ok_or_else(|| {
                    fail("TYPE_MISMATCH", "missing view binding type", value.span)
                })?;
                if matches!(&ty, Type::Named(view) if crate::front::view_parts(view).is_some()) {
                    if *mutable {
                        return Err(fail(
                            "UNSUPPORTED_FEATURE",
                            "view bindings cannot be reassigned",
                            statement.span,
                        ));
                    }
                    if matches!(value.kind, ExprKind::Name(_)) {
                        return Err(fail(
                            "UNSUPPORTED_FEATURE",
                            "view aliases require an explicit reborrow",
                            value.span,
                        ));
                    }
                    if self.views.len() >= 4096 {
                        return Err(FrontError::tool(
                            "RESOURCE_LIMIT",
                            "view fact limit exceeded",
                        ));
                    }
                    let fact = self.derive(value)?;
                    self.views.insert(name.clone(), fact);
                } else {
                    self.expr(value)?;
                }
                self.bindings.insert(name.clone(), (ty, *mutable, None));
                declared.push(name.clone());
            }
            StmtKind::Assign { place, value } => {
                self.expr(value)?;
                match place {
                    Place::Name(name, span) => self.write(name, *span)?,
                    Place::Index { name, index, span } => {
                        self.write(name, *span)?;
                        self.expr(index)?;
                    }
                }
            }
            StmtKind::Return(Some(value)) if self.is_view(value) => {
                let fact = self.derive(value)?;
                if self.function.return_from.as_deref() != Some(fact.root.as_str()) {
                    return Err(fail(
                        "OWN_RETURN_BORROW",
                        "returned view must come from declared borrowing parameter",
                        value.span,
                    ));
                }
            }
            StmtKind::Return(Some(value)) | StmtKind::Expr(value) => self.expr(value)?,
            StmtKind::Return(None) => {}
            StmtKind::If {
                condition,
                then_block,
                else_arm,
            } => {
                self.expr(condition)?;
                self.block(then_block)?;
                match else_arm {
                    Some(ElseArm::Block(block)) => self.block(block)?,
                    Some(ElseArm::If(stmt)) => self.statement(stmt, &mut Vec::new())?,
                    None => {}
                }
            }
            StmtKind::While { condition, body } => {
                self.expr(condition)?;
                self.block(body)?;
            }
            StmtKind::For {
                name,
                lower,
                upper,
                body,
            } => {
                self.expr(lower)?;
                self.expr(upper)?;
                self.bindings.insert(name.clone(), (Type::U32, false, None));
                self.block(body)?;
                self.bindings.remove(name);
            }
        }
        Ok(())
    }
}
