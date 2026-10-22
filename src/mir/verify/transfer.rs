fn transfer(
    op: &Instruction,
    state: &mut FlowState,
    signatures: &HashMap<&str, &Function>,
    function: &Function,
    definitions: &HashMap<ValueId, Definition>,
) -> Result<(), Error> {
    let require = |name: &str, state: &FlowState| {
        if state.available.contains(name) {
            Ok(())
        } else {
            Err(invalid("uninitialized or moved MIR place"))
        }
    };
    let conflict = |name: &str, state: &FlowState, mutating: bool| {
        state
            .loans
            .values()
            .any(|(actual, mode)| actual == name && (mutating || *mode == Mode::Mut))
    };
    match &op.kind {
        Operation::Read(name) | Operation::Length(name) => {
            require(name, state)?;
            if conflict(name, state, false) {
                return Err(invalid("MIR read during mutable loan"));
            }
        }
        Operation::ArrayRef { name, loan } | Operation::RecordRef { name, loan } => {
            require(name, state)?;
            if loan.is_some() && state.partial.get(name).is_some_and(|fields| !fields.is_empty()) {
                return Err(invalid("MIR loan of partially moved record"));
            }
            if conflict(name, state, *loan == Some(Mode::Mut)) {
                return Err(invalid("MIR loan conflict"));
            }
            if let Some(mode) = loan {
                let id = op
                    .output
                    .as_ref()
                    .ok_or_else(|| invalid("loan lacks output"))?
                    .0;
                state.loans.insert(id, (name.clone(), *mode));
            }
        }
        Operation::Move(name) => {
            require(name, state)?;
            if state.partial.get(name).is_some_and(|fields| !fields.is_empty()) {
                return Err(invalid("MIR move of partially moved record"));
            }
            if conflict(name, state, true) {
                return Err(invalid("MIR move during loan"));
            }
            state.available.remove(name);
        }
        Operation::MoveField { name, index } => {
            require(name, state)?;
            if conflict(name, state, true) || !state.partial.entry(name.clone()).or_default().insert(*index) {
                return Err(invalid("MIR field moved twice or during loan"));
            }
        }
        Operation::VariantTakePayload { value, .. }
            if !state.taken_payloads.insert(*value) || state.dropped_values.contains(value) =>
        {
            return Err(invalid("MIR Result payload taken twice"));
        }
        Operation::DropValue(value)
            if !state.taken_payloads.contains(value) || !state.dropped_values.insert(*value) =>
        {
            return Err(invalid("MIR Result temporary cleanup mismatch"));
        }
        Operation::VariantTakePayload { .. } | Operation::DropValue(_) => {}
        Operation::Field { record, index } => {
            let definition = definitions.get(record).ok_or_else(|| invalid("MIR field base missing"))?;
            let origin = definition.position.and_then(|position| function.blocks[definition.block].instructions.get(position));
            let Some(Operation::RecordRef { name, .. }) = origin.map(|instruction| &instruction.kind) else {
                return Err(invalid("MIR field base must be a named record reference"));
            };
            if state.partial.get(name).is_some_and(|fields| fields.contains(index)) {
                return Err(invalid("MIR read of moved record field"));
            }
        }
        Operation::Bind { name, .. } => {
            if state.available.contains(name) {
                return Err(invalid("MIR binding initialized twice"));
            }
            state.available.insert(name.clone());
            state.partial.remove(name);
        }
        Operation::Store { name, .. } => {
            require(name, state)?;
            if state.partial.get(name).is_some_and(|fields| !fields.is_empty()) {
                return Err(invalid("MIR store to partially moved record"));
            }
            if conflict(name, state, true) {
                return Err(invalid("MIR store during loan"));
            }
        }
        Operation::StoreIndex { name, .. } => {
            require(name, state)?;
            if conflict(name, state, true) {
                return Err(invalid("MIR indexed store during loan"));
            }
        }
        Operation::Drop(names) => {
            for name in names {
                if conflict(name, state, true) {
                    return Err(invalid("MIR drop during loan"));
                }
                state.available.remove(name);
                state.partial.remove(name);
            }
        }
        Operation::Call { name, args } => {
            let modes = signatures
                .get(name.as_str())
                .map(|callee| {
                    callee
                        .params
                        .iter()
                        .map(|param| param.mode)
                        .collect::<Vec<_>>()
                })
                .or_else(|| crate::check::sort_modes(name).map(|modes| modes.iter().copied().map(Some).collect()))
                .or_else(|| crate::spatial::result(name).map(|_| (0..args.len()).map(|i| crate::spatial::mode(name, i)).collect()))
                .unwrap_or_default();
            for (index, arg) in args.iter().enumerate() {
                if matches!(modes.get(index), Some(Some(Mode::Read | Mode::Mut)))
                    && !state.loans.contains_key(arg)
                {
                    return Err(invalid("inactive MIR loan reused"));
                }
                state.loans.remove(arg);
            }
        }
        _ => {}
    }
    match &op.kind {
        Operation::Read(name) | Operation::Length(name) => view_access(state, name, false)?,
        Operation::ArrayRef { name, loan: Some(mode) } => view_access(state, name, *mode != Mode::Read)?,
        Operation::Index { array, .. } => {
            if let Operation::ArrayRef { name, .. } = instruction(function, definitions, *array)? {
                if !state.available.contains(name) {
                    return Err(invalid("MIR index uses unavailable array or view"));
                }
                view_access(state, name, false)?;
            }
        }
        Operation::Move(name) => {
            view_access(state, name, true)?;
            if let Some(fact) = state.views.remove(name) {
                let id = op.output.as_ref().ok_or_else(|| invalid("moved view has no output"))?.0;
                state.view_values.insert(id, fact);
            }
        }
        Operation::MoveField { name, .. } | Operation::Store { name, .. } | Operation::StoreIndex { name, .. } => view_access(state, name, true)?,
        Operation::Bind { name, value } => {
            if matches!(&function.locals.get(name).ok_or_else(|| invalid("view binding missing"))?.ty, Type::Named(view) if crate::front::view_parts(view).is_some()) {
                let fact = state.view_values.remove(value).ok_or_else(|| invalid("view binding lacks proven source"))?;
                view_conflict(state, &fact)?;
                state.views.insert(name.clone(), fact);
            }
        }
        Operation::Drop(names) => {
            for name in names {
                view_access(state, name, true)?;
                state.views.remove(name);
            }
        }
        Operation::Call { name, args } if matches!(name.as_str(), "view" | "view_mut") => {
            let fact = derive_view(state, function, definitions, args, name == "view_mut", false)?;
            let id = op.output.as_ref().ok_or_else(|| invalid("view constructor has no output"))?.0;
            state.view_values.insert(id, fact);
        }
        Operation::Call { name, args } if signatures.get(name.as_str()).is_some_and(|callee| matches!(&callee.result, Type::Named(view) if crate::front::view_parts(view).is_some())) => {
            let callee = signatures[name.as_str()];
            let source = callee.return_from.as_ref().ok_or_else(|| invalid("view callee has no return source"))?;
            let position = callee.params.iter().position(|param| param.name == *source).ok_or_else(|| invalid("view callee source missing"))?;
            let fact = derive_view(state, function, definitions, &args[position..=position], matches!(&callee.result, Type::Named(view) if crate::front::view_parts(view).is_some_and(|(_, mutable)| mutable)), true)?;
            let id = op.output.as_ref().ok_or_else(|| invalid("view call has no output"))?.0;
            state.view_values.insert(id, fact);
        }
        _ => {}
    }
    Ok(())
}
