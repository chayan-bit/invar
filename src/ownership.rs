//! Local array ownership and immediate call loans.

use std::collections::{HashMap, HashSet};

use crate::{
    check::CheckedProgram,
    front::{
        Block, ElseArm, Expr, ExprKind, FrontError, Function, Mode, Place, Span, Stmt, StmtKind,
        Type,
    },
};

#[derive(Clone)]
struct Owner {
    available: bool,
    owned: bool,
    mutable: bool,
    mode: Option<Mode>,
    declared_loop: usize,
    read_loans: usize,
    mut_loan: bool,
    protected: bool,
    moved_fields: HashSet<String>,
}

type State = HashMap<String, Owner>;

#[derive(Clone, Copy)]
enum Context {
    Value,
    Read,
    Return,
}

fn array(ty: &Type) -> bool {
    matches!(
        ty,
        Type::ArrayI32 | Type::ArrayU32 | Type::ArrayF32 | Type::ArrayF64 | Type::ArrayBool
    )
}

fn affine(ty: &Type) -> bool {
    array(ty) || matches!(ty, Type::Named(name) if crate::front::view_parts(name).is_none())
}

fn error(code: &'static str, message: impl Into<String>, span: Span) -> FrontError {
    FrontError::source(code, message, span)
}

/// Check local array moves, uses, immediate call loans, and continuing-path joins.
pub fn check(program: &CheckedProgram) -> Result<(), FrontError> {
    for function in &program.program.functions {
        let mut state = State::new();
        for param in &function.params {
            if affine(&param.ty) {
                state.insert(
                    param.name.clone(),
                    Owner {
                        available: true,
                        owned: param.mode == Some(Mode::Take),
                        mutable: false,
                        mode: param.mode,
                        declared_loop: 0,
                        read_loans: 0,
                        mut_loan: false,
                        protected: false,
                        moved_fields: HashSet::new(),
                    },
                );
            }
        }
        block(program, function, &function.body, &mut state, 0)?;
    }
    Ok(())
}

fn available(state: &State, name: &str, span: Span) -> Result<(), FrontError> {
    if let Some(owner) = state.get(name)
        && !owner.available
    {
        return Err(error(
            "OWN_USE_AFTER_MOVE",
            format!("array '{name}' was moved"),
            span,
        ));
    }
    Ok(())
}

fn read(state: &State, name: &str, span: Span) -> Result<(), FrontError> {
    available(state, name, span)?;
    if state.get(name).is_some_and(|owner| owner.mut_loan) {
        return Err(error(
            "OWN_BORROW_CONFLICT",
            "mutable loan excludes read access",
            span,
        ));
    }
    Ok(())
}

fn store(state: &State, name: &str, span: Span) -> Result<(), FrontError> {
    available(state, name, span)?;
    if state
        .get(name)
        .is_some_and(|owner| !owner.moved_fields.is_empty())
    {
        return Err(error(
            "OWN_USE_AFTER_MOVE",
            "partially moved record cannot be overwritten",
            span,
        ));
    }
    if state
        .get(name)
        .is_some_and(|owner| owner.read_loans > 0 || owner.mut_loan)
    {
        return Err(error(
            "OWN_BORROW_CONFLICT",
            "active loan excludes store",
            span,
        ));
    }
    Ok(())
}

fn loan(state: &mut State, name: &str, mode: Mode, span: Span) -> Result<(), FrontError> {
    available(state, name, span)?;
    let owner = state
        .get_mut(name)
        .ok_or_else(|| error("OWN_BORROW_SOURCE", "loan requires a named array", span))?;
    if !owner.moved_fields.is_empty() {
        return Err(error(
            "OWN_USE_AFTER_MOVE",
            "partially moved record cannot be loaned whole",
            span,
        ));
    }
    if owner.mut_loan || (mode == Mode::Mut && (owner.read_loans > 0 || owner.protected)) {
        return Err(error(
            "OWN_BORROW_CONFLICT",
            "incompatible active array loan",
            span,
        ));
    }
    if mode == Mode::Mut && !owner.mutable && owner.mode != Some(Mode::Mut) {
        return Err(error(
            "OWN_MUT_REQUIRED",
            "mutable loan requires a mutable owner or mut parameter",
            span,
        ));
    }
    match mode {
        Mode::Read => owner.read_loans += 1,
        Mode::Mut => owner.mut_loan = true,
        Mode::Take => unreachable!(),
    }
    Ok(())
}

fn consume(
    state: &mut State,
    name: &str,
    span: Span,
    loop_depth: usize,
    returning: bool,
) -> Result<(), FrontError> {
    let owner = state
        .get_mut(name)
        .ok_or_else(|| error("OWN_MOVE_REQUIRED", "move requires an array owner", span))?;
    if !owner.available {
        return Err(error(
            "OWN_USE_AFTER_MOVE",
            format!("array '{name}' was moved"),
            span,
        ));
    }
    if !owner.moved_fields.is_empty() {
        return Err(error(
            "OWN_USE_AFTER_MOVE",
            "partially moved record cannot be moved whole",
            span,
        ));
    }
    if !owner.owned {
        return Err(error(
            if returning {
                "OWN_RETURN_BORROW"
            } else {
                "OWN_MOVE_REQUIRED"
            },
            "borrowed array cannot be moved",
            span,
        ));
    }
    if owner.read_loans > 0 || owner.mut_loan || owner.protected {
        return Err(error(
            "OWN_BORROW_CONFLICT",
            "active loan excludes move",
            span,
        ));
    }
    if owner.declared_loop < loop_depth {
        return Err(error(
            "OWN_MOVE_REQUIRED",
            "cannot move an outer array binding from a loop",
            span,
        ));
    }
    owner.available = false;
    Ok(())
}

fn consume_field(
    state: &mut State,
    name: &str,
    field: &str,
    span: Span,
    loop_depth: usize,
) -> Result<(), FrontError> {
    let owner = state.get_mut(name).ok_or_else(|| {
        error(
            "OWN_MOVE_REQUIRED",
            "field move requires an owned record",
            span,
        )
    })?;
    if !owner.available || owner.moved_fields.contains(field) {
        return Err(error(
            "OWN_USE_AFTER_MOVE",
            format!("record field '{name}.{field}' was moved"),
            span,
        ));
    }
    if !owner.owned
        || owner.read_loans > 0
        || owner.mut_loan
        || owner.protected
        || owner.declared_loop < loop_depth
    {
        return Err(error(
            "OWN_BORROW_CONFLICT",
            "record field move conflicts with ownership or a loan",
            span,
        ));
    }
    owner.moved_fields.insert(field.to_owned());
    Ok(())
}

fn block(
    program: &CheckedProgram,
    function: &Function,
    block: &Block,
    state: &mut State,
    loop_depth: usize,
) -> Result<bool, FrontError> {
    let mut locals = Vec::new();
    let mut returns = false;
    for statement in &block.statements {
        let statement_returns =
            statement_check(program, function, statement, state, loop_depth, &mut locals)?;
        returns |= statement_returns;
    }
    for name in locals {
        state.remove(&name);
    }
    Ok(returns)
}

fn statement_check(
    program: &CheckedProgram,
    function: &Function,
    statement: &Stmt,
    state: &mut State,
    loop_depth: usize,
    locals: &mut Vec<String>,
) -> Result<bool, FrontError> {
    match &statement.kind {
        StmtKind::Binding {
            mutable,
            name,
            value,
            annotation,
        } => {
            expr(program, function, value, state, loop_depth, Context::Value)?;
            let is_owned = annotation.as_ref().is_some_and(affine)
                || type_of(program, function, value).is_some_and(affine);
            if is_owned {
                state.insert(
                    name.clone(),
                    Owner {
                        available: true,
                        owned: true,
                        mutable: *mutable,
                        mode: None,
                        declared_loop: loop_depth,
                        read_loans: 0,
                        mut_loan: false,
                        protected: false,
                        moved_fields: HashSet::new(),
                    },
                );
                locals.push(name.clone());
            }
            Ok(false)
        }
        StmtKind::Assign { place, value } => {
            expr(program, function, value, state, loop_depth, Context::Value)?;
            match place {
                Place::Name(name, span) => store(state, name, *span)?,
                Place::Index { name, index, span } => {
                    store(state, name, *span)?;
                    if let Some(owner) = state.get_mut(name) {
                        owner.protected = true;
                    }
                    expr(program, function, index, state, loop_depth, Context::Value)?;
                    if let Some(owner) = state.get_mut(name) {
                        owner.protected = false;
                    }
                    store(state, name, *span)?;
                }
            }
            Ok(false)
        }
        StmtKind::Return(value) => {
            if let Some(value) = value {
                let context = if affine(&function.result) {
                    Context::Return
                } else {
                    Context::Value
                };
                expr(program, function, value, state, loop_depth, context)?;
            }
            Ok(true)
        }
        StmtKind::If {
            condition,
            then_block,
            else_arm,
        } => {
            expr(
                program,
                function,
                condition,
                state,
                loop_depth,
                Context::Value,
            )?;
            let mut then_state = state.clone();
            let then_returns = block(program, function, then_block, &mut then_state, loop_depth)?;
            let mut else_state = state.clone();
            let else_returns = match else_arm {
                Some(ElseArm::Block(else_block)) => {
                    block(program, function, else_block, &mut else_state, loop_depth)?
                }
                Some(ElseArm::If(statement)) => statement_check(
                    program,
                    function,
                    statement,
                    &mut else_state,
                    loop_depth,
                    &mut Vec::new(),
                )?,
                None => false,
            };
            if !then_returns && !else_returns {
                for (name, owner) in state.iter_mut() {
                    if let (Some(left), Some(right)) = (then_state.get(name), else_state.get(name))
                    {
                        owner.available = left.available && right.available;
                        owner.moved_fields = left
                            .moved_fields
                            .union(&right.moved_fields)
                            .cloned()
                            .collect();
                    }
                }
            } else if !then_returns {
                *state = then_state;
            } else if !else_returns {
                *state = else_state;
            }
            Ok(then_returns && else_arm.is_some() && else_returns)
        }
        StmtKind::While { condition, body } => {
            let mut loop_state = state.clone();
            expr(
                program,
                function,
                condition,
                &mut loop_state,
                loop_depth + 1,
                Context::Value,
            )?;
            block(program, function, body, &mut loop_state, loop_depth + 1)?;
            Ok(false)
        }
        StmtKind::For {
            lower, upper, body, ..
        } => {
            expr(program, function, lower, state, loop_depth, Context::Value)?;
            expr(program, function, upper, state, loop_depth, Context::Value)?;
            let mut loop_state = state.clone();
            block(program, function, body, &mut loop_state, loop_depth + 1)?;
            Ok(false)
        }
        StmtKind::Expr(value) => {
            expr(program, function, value, state, loop_depth, Context::Value)?;
            if type_of(program, function, value).is_some_and(affine) {
                return Err(error(
                    "UNSUPPORTED_FEATURE",
                    "discarding an owned array result is not specified yet",
                    value.span,
                ));
            }
            Ok(false)
        }
    }
}

fn type_of<'a>(program: &'a CheckedProgram, function: &Function, expr: &Expr) -> Option<&'a Type> {
    program
        .function_expression_types
        .get(&function.name)?
        .iter()
        .find(|(span, _)| *span == expr.span)
        .map(|(_, ty)| ty)
}

fn expr(
    program: &CheckedProgram,
    function: &Function,
    value: &Expr,
    state: &mut State,
    loop_depth: usize,
    context: Context,
) -> Result<(), FrontError> {
    match &value.kind {
        ExprKind::Integer(_)
        | ExprKind::Float(_)
        | ExprKind::Bool(_)
        | ExprKind::Text(_)
        | ExprKind::Bytes(_) => Ok(()),
        ExprKind::Name(name) => {
            if state.contains_key(name) {
                match context {
                    Context::Read => read(state, name, value.span),
                    Context::Return => consume(state, name, value.span, loop_depth, true),
                    Context::Value => {
                        available(state, name, value.span)?;
                        Err(error(
                            "OWN_MOVE_REQUIRED",
                            format!("owned value '{name}' requires move"),
                            value.span,
                        ))
                    }
                }
            } else {
                Ok(())
            }
        }
        ExprKind::Move(name) => consume(
            state,
            name,
            value.span,
            loop_depth,
            matches!(context, Context::Return),
        ),
        ExprKind::MoveField { base, field } => {
            let ExprKind::Name(name) = &base.kind else {
                return Err(error(
                    "OWN_MOVE_REQUIRED",
                    "field move requires a named record",
                    base.span,
                ));
            };
            consume_field(state, name, field, value.span, loop_depth)
        }
        ExprKind::Try(inner) => expr(program, function, inner, state, loop_depth, Context::Value),
        ExprKind::Group(inner) => expr(
            program,
            function,
            inner,
            state,
            loop_depth,
            if matches!(context, Context::Return) {
                Context::Value
            } else {
                context
            },
        ),
        ExprKind::Array(items) => {
            for item in items {
                expr(program, function, item, state, loop_depth, Context::Value)?;
            }
            Ok(())
        }
        ExprKind::Record { fields, .. } => {
            for field in fields {
                expr(
                    program,
                    function,
                    &field.value,
                    state,
                    loop_depth,
                    Context::Value,
                )?;
            }
            Ok(())
        }
        ExprKind::Match { value, arms } => {
            let moving = matches!(value.kind, ExprKind::Move(_));
            if !moving && !matches!(value.kind, ExprKind::Name(_)) {
                return Err(error(
                    "UNSUPPORTED_FEATURE",
                    "this match subset uses a named variant owner",
                    value.span,
                ));
            }
            expr(
                program,
                function,
                value,
                state,
                loop_depth,
                if moving {
                    Context::Value
                } else {
                    Context::Read
                },
            )?;
            let mut joined = state.clone();
            if let ExprKind::Name(name) = &value.kind
                && let Some(owner) = joined.get_mut(name)
            {
                owner.read_loans += 1;
            }
            let mut states = Vec::new();
            for arm in arms {
                let mut arm_state = joined.clone();
                if let Some(binding) = &arm.binding {
                    let payload = type_of(program, function, value)
                        .and_then(|ty| match ty {
                            Type::Named(name) => program
                                .program
                                .variants
                                .iter()
                                .find(|variant| variant.name == *name),
                            _ => None,
                        })
                        .and_then(|variant| {
                            arm.variant.rsplit_once("::").and_then(|(_, case)| {
                                variant.arms.iter().find(|item| item.name == case)
                            })
                        })
                        .and_then(|case| case.payload.as_ref());
                    if payload.is_some_and(affine) {
                        arm_state.insert(
                            binding.clone(),
                            Owner {
                                available: true,
                                owned: moving,
                                mutable: false,
                                mode: None,
                                declared_loop: loop_depth,
                                read_loans: 0,
                                mut_loan: false,
                                protected: false,
                                moved_fields: HashSet::new(),
                            },
                        );
                    }
                }
                expr(
                    program,
                    function,
                    &arm.value,
                    &mut arm_state,
                    loop_depth,
                    Context::Value,
                )?;
                if let Some(binding) = &arm.binding {
                    arm_state.remove(binding);
                }
                states.push(arm_state);
            }
            for (name, owner) in state.iter_mut() {
                owner.available = states
                    .iter()
                    .all(|arm_state| arm_state.get(name).is_none_or(|item| item.available));
                owner.moved_fields = states
                    .iter()
                    .filter_map(|arm_state| arm_state.get(name))
                    .flat_map(|item| item.moved_fields.iter().cloned())
                    .collect();
            }
            Ok(())
        }
        ExprKind::Field { base, name } => {
            expr(program, function, base, state, loop_depth, Context::Read)?;
            if let ExprKind::Name(owner) = &base.kind
                && state
                    .get(owner)
                    .is_some_and(|record| record.moved_fields.contains(name))
            {
                return Err(error(
                    "OWN_USE_AFTER_MOVE",
                    format!("record field '{owner}.{name}' was moved"),
                    value.span,
                ));
            }
            Ok(())
        }
        ExprKind::Index { base, index } => {
            expr(program, function, base, state, loop_depth, Context::Read)?;
            expr(program, function, index, state, loop_depth, Context::Value)
        }
        ExprKind::Unary { value, .. } => {
            expr(program, function, value, state, loop_depth, Context::Value)
        }
        ExprKind::Binary { left, right, .. } => {
            expr(program, function, left, state, loop_depth, Context::Value)?;
            expr(program, function, right, state, loop_depth, Context::Value)
        }
        ExprKind::Call { name, args } => {
            if let Some(modes) = crate::check::sort_modes(name) {
                return call_arguments(
                    program,
                    function,
                    args,
                    modes.iter().copied().map(Some),
                    state,
                    loop_depth,
                );
            }
            let read_count = crate::spatial::read_count(name);
            if read_count > 0 {
                return call_arguments(
                    program,
                    function,
                    args,
                    (0..args.len()).map(|index| (index < read_count).then_some(Mode::Read)),
                    state,
                    loop_depth,
                );
            }
            if matches!(name.as_str(), "view" | "view_mut") {
                let Some(Expr {
                    kind: ExprKind::Name(_),
                    ..
                }) = args.first()
                else {
                    return Err(error(
                        "OWN_BORROW_SOURCE",
                        "view needs a named owner",
                        value.span,
                    ));
                };
                expr(
                    program,
                    function,
                    &args[0],
                    state,
                    loop_depth,
                    Context::Read,
                )?;
                for arg in args.iter().skip(1) {
                    expr(program, function, arg, state, loop_depth, Context::Value)?;
                }
                return Ok(());
            }
            if crate::fsum::Kind::from_name(name).is_some() {
                // Every operand is a read loan for the call; both dot operands
                // may name the same array.
                for arg in args {
                    if !matches!(arg.kind, ExprKind::Name(_)) {
                        return Err(error(
                            "UNSUPPORTED_FEATURE",
                            "reduction requires named arrays",
                            arg.span,
                        ));
                    }
                    expr(program, function, arg, state, loop_depth, Context::Read)?;
                }
                return Ok(());
            }
            if matches!(
                name.as_str(),
                "len" | "bytes_slice" | "utf8_slice" | "utf8_decode" | "utf8_encode"
            ) {
                let Some(Expr {
                    kind: ExprKind::Name(_),
                    ..
                }) = args.first()
                else {
                    return Err(error(
                        "UNSUPPORTED_FEATURE",
                        "len requires a named array",
                        value.span,
                    ));
                };
                expr(
                    program,
                    function,
                    &args[0],
                    state,
                    loop_depth,
                    Context::Read,
                )?;
                for arg in args.iter().skip(1) {
                    expr(program, function, arg, state, loop_depth, Context::Value)?;
                }
                return Ok(());
            }
            if let Some(params) = program
                .program
                .function(name)
                .map(|function| function.params.as_slice())
                .or_else(|| {
                    program
                        .program
                        .foreign_functions
                        .iter()
                        .find(|function| function.name == *name)
                        .map(|function| function.params.as_slice())
                })
            {
                let modes = params
                    .iter()
                    .map(|param| if affine(&param.ty) { param.mode } else { None });
                return call_arguments(program, function, args, modes, state, loop_depth);
            }
            for arg in args {
                expr(program, function, arg, state, loop_depth, Context::Value)?;
            }
            Ok(())
        }
    }
}

// Keep loan temporaries outside the recursive expression walker's stack frame.
fn call_arguments(
    program: &CheckedProgram,
    function: &Function,
    args: &[Expr],
    modes: impl IntoIterator<Item = Option<Mode>>,
    state: &mut State,
    loop_depth: usize,
) -> Result<(), FrontError> {
    let mut loans = Vec::new();
    for (arg, mode) in args.iter().zip(modes) {
        if let Some(mode @ (Mode::Read | Mode::Mut)) = mode {
            let ExprKind::Name(actual) = &arg.kind else {
                return Err(error(
                    "OWN_BORROW_SOURCE",
                    "borrowed argument requires a bare owner name",
                    arg.span,
                ));
            };
            loan(state, actual, mode, arg.span)?;
            loans.push((actual.as_str(), mode, arg.span));
        } else {
            expr(program, function, arg, state, loop_depth, Context::Value)?;
        }
    }
    for (actual, mode, span) in loans {
        let owner = state
            .get_mut(actual)
            .ok_or_else(|| error("OWN_BORROW_SOURCE", "borrowed owner missing", span))?;
        match mode {
            Mode::Read => owner.read_loans -= 1,
            Mode::Mut => owner.mut_loan = false,
            Mode::Take => unreachable!(),
        }
    }
    Ok(())
}
