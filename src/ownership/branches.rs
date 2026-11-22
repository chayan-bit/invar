//! Ownership through value branches: `match` arms with their guards (#260) and the
//! two arms of an `if` expression (#361), joined so an owner any path moves is moved.

use super::*;

pub(super) fn match_expr(
    program: &CheckedProgram,
    function: &Function,
    value: &Expr,
    arms: &[crate::front::MatchArm],
    state: &mut State,
    loop_depth: usize,
) -> Result<(), FrontError> {
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
    // A guard runs only for its own case, so its effects reach its arm and
    // the later arms of that case, never another case's arms (#260).
    let mut fallthrough = HashMap::new();
    for arm in arms {
        let case = arm.variant.rsplit("::").next();
        let mut arm_state = fallthrough.get(&case).unwrap_or(&joined).clone();
        if let Some(guard) = &arm.guard {
            expr(
                program,
                function,
                guard,
                &mut arm_state,
                loop_depth,
                Context::Value,
            )?;
            fallthrough.insert(case, arm_state.clone());
        }
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
                    arm.variant
                        .rsplit_once("::")
                        .and_then(|(_, case)| variant.arms.iter().find(|item| item.name == case))
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
    join(state, &states);
    Ok(())
}

/// `if c { a } else { b }`: the condition, then each arm from the state after it.
pub(super) fn if_expr(
    program: &CheckedProgram,
    function: &Function,
    [condition, then, otherwise]: [&Expr; 3],
    state: &mut State,
    loop_depth: usize,
) -> Result<(), FrontError> {
    expr(
        program,
        function,
        condition,
        state,
        loop_depth,
        Context::Value,
    )?;
    let mut states = vec![state.clone(), state.clone()];
    for (arm, arm_state) in [then, otherwise].into_iter().zip(&mut states) {
        expr(
            program,
            function,
            arm,
            arm_state,
            loop_depth,
            Context::Value,
        )?;
    }
    join(state, &states);
    Ok(())
}

/// The state after branches that each ran from `state`: an owner any branch moved is moved.
fn join(state: &mut State, states: &[State]) {
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
}
