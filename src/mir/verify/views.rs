fn instruction<'a>(function: &'a Function, definitions: &HashMap<ValueId, Definition>, id: ValueId) -> Result<&'a Operation, Error> {
    let definition = definitions.get(&id).ok_or_else(|| invalid("view operand is undefined"))?;
    definition.position.and_then(|position| function.blocks[definition.block].instructions.get(position))
        .map(|item| &item.kind).ok_or_else(|| invalid("view operand has no defining operation"))
}

fn view_source<'a>(function: &'a Function, definitions: &HashMap<ValueId, Definition>, id: ValueId) -> Result<&'a str, Error> {
    match instruction(function, definitions, id)? {
        Operation::ArrayRef { name, .. } => Ok(name),
        _ => Err(invalid("view source must be a named array reference")),
    }
}

fn known_u32(function: &Function, definitions: &HashMap<ValueId, Definition>, id: ValueId) -> Option<u64> {
    match instruction(function, definitions, id).ok()? {
        Operation::U32(value) => Some(u64::from(*value)),
        _ => None,
    }
}

fn base_view(state: &FlowState, function: &Function, name: &str) -> Result<ViewFact, Error> {
    if let Some(view) = state.views.get(name) {
        if state.views.values().any(|other| other.parent.as_deref() == Some(name)) {
            return Err(invalid("reborrowed view used while child is live"));
        }
        return Ok(view.clone());
    }
    let binding = function.locals.get(name).ok_or_else(|| invalid("view root is not a place"))?;
    if !matches!(binding.ty, Type::ArrayI32 | Type::ArrayU32 | Type::ArrayF32 | Type::ArrayF64 | Type::ArrayBool) {
        return Err(invalid("view root is not an array"));
    }
    Ok(ViewFact { root: name.to_owned(), interval: None, stride: None, mutable: binding.mutable || binding.mode == Some(Mode::Mut), parent: None })
}

fn view_conflict(state: &FlowState, fact: &ViewFact) -> Result<(), Error> {
    for (name, other) in &state.views {
        if fact.root != other.root { continue; }
        let mut ancestor = fact.parent.as_deref();
        let mut suspended = false;
        while let Some(parent) = ancestor {
            if parent == name {
                suspended = true;
                break;
            }
            ancestor = state.views.get(parent).and_then(|view| view.parent.as_deref());
        }
        if suspended { continue; }
        let disjoint = matches!((fact.interval, other.interval), (Some((a,b)), Some((c,d))) if a >= d || c >= b || a == b || c == d);
        if !disjoint && (fact.mutable || other.mutable) { return Err(invalid("overlapping or unproved mutable MIR views")); }
    }
    Ok(())
}

fn derive_view(state: &FlowState, function: &Function, definitions: &HashMap<ValueId, Definition>, args: &[ValueId], mutable: bool, from_function: bool) -> Result<ViewFact, Error> {
    let name = view_source(function, definitions, args[0])?;
    if !state.available.contains(name) { return Err(invalid("view source is unavailable")); }
    let base = base_view(state, function, name)?;
    if mutable && !base.mutable { return Err(invalid("mutable MIR view lacks permission")); }
    let parent = state.views.contains_key(name).then(|| name.to_owned());
    let (interval, stride) = if from_function {
        (None, None)
    } else {
        let start = known_u32(function, definitions, args[1]);
        let end = known_u32(function, definitions, args[2]);
        let step = args.get(3).and_then(|id| known_u32(function, definitions, *id)).or(if args.len() == 3 { Some(1) } else { None });
        let parent_stride = base.stride.or(if parent.is_none() { Some(1) } else { None });
        let origin = base.interval.map(|(lo, _)| lo).or(if parent.is_none() { Some(0) } else { None });
        let interval = match (origin, parent_stride, start, end, step) {
            (Some(origin), Some(parent_stride), Some(start), Some(end), Some(step)) if step > 0 && end >= start => {
                let first = origin.checked_add(start.checked_mul(parent_stride).ok_or_else(|| invalid("view interval overflow"))?).ok_or_else(|| invalid("view interval overflow"))?;
                let count = (end - start).div_ceil(step);
                let upper = if count == 0 { first } else {
                    first.checked_add((count - 1).checked_mul(parent_stride.checked_mul(step).ok_or_else(|| invalid("view stride overflow"))?).ok_or_else(|| invalid("view interval overflow"))?)
                        .and_then(|last| last.checked_add(1)).ok_or_else(|| invalid("view interval overflow"))?
                };
                Some((first, upper))
            }
            _ => None,
        };
        let stride = parent_stride.and_then(|parent| step.and_then(|step| parent.checked_mul(step)));
        (interval, stride)
    };
    let fact = ViewFact { root: base.root, interval, stride, mutable, parent };
    view_conflict(state, &fact)?;
    Ok(fact)
}

fn view_access(state: &FlowState, name: &str, writing: bool) -> Result<(), Error> {
    if state.views.contains_key(name) {
        if state.views.values().any(|view| view.parent.as_deref() == Some(name)) {
            return Err(invalid("MIR parent view accessed during reborrow"));
        }
    } else if state.views.values().any(|view| view.root == name && (writing || view.mutable)) {
        return Err(invalid("MIR owner access conflicts with live view"));
    }
    Ok(())
}
