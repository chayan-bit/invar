include!("verify/types.rs");
include!("verify/operation.rs");
include!("verify/views.rs");
include!("verify/transfer.rs");

fn check_limits(program: &Program) -> Result<(), Error> {
    let instructions = program
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .fold(0usize, |total, block| {
            total.saturating_add(block.instructions.len())
        });
    let user_records = program.records.iter().filter(|record|
        !matches!(record.name.as_str(), crate::spatial::NEIGHBORS | crate::spatial::KNN)
    ).count();
    // The two canonical builtin result records do not consume the existing
    // 128 user-record slots; their exact layouts are checked below.
    if program.functions.len() > MAX_FUNCTIONS
        || program.records.len() > 130
        || user_records > 128
        || program.variants.len() > 128
        || instructions > MAX_INSTRUCTIONS
        || program.functions.iter().any(|function| {
            function.blocks.len() > MAX_BLOCKS_PER_FUNCTION
                || function.locals.len() > MAX_LOCALS_PER_FUNCTION
                || function.declaration_order.len() > MAX_LOCALS_PER_FUNCTION
        })
    {
        return Err(Error {
            code: "RESOURCE_LIMIT",
            message: "MIR resource limit exceeded".into(),
        });
    }
    Ok(())
}

/// Reject malformed structure, types, dominance, places and active-loan joins.
///
/// Programs over [`MAX_FUNCTIONS`], [`MAX_BLOCKS_PER_FUNCTION`],
/// [`MAX_LOCALS_PER_FUNCTION`] or [`MAX_INSTRUCTIONS`] are rejected with
/// `RESOURCE_LIMIT` before any other analysis.
pub fn verify(program: &Program) -> Result<(), Error> {
    if program.schema_version != SCHEMA_VERSION {
        return Err(invalid("unknown MIR schema"));
    }
    if program.numeric_policy != crate::numeric_contract::NumericPolicy::STRICT_F32 || program.layout_version != crate::layout_contract::VERSION {
        return Err(invalid("unknown MIR numeric or layout contract"));
    }
    check_limits(program)?;
    crate::tensor::verify(program)?;
    let mut signatures = HashMap::new();
    for function in &program.functions {
        if crate::spatial::result(&function.name).is_some() { return Err(invalid("spatial builtin function redefined")); }
        if signatures
            .insert(function.name.as_str(), function)
            .is_some()
        {
            return Err(invalid("duplicate MIR function"));
        }
    }
    let mut records = HashMap::new();
    for record in &program.records {
        if record.name.rsplit("::").next().is_some_and(crate::spatial::is_builtin) {
            let expected = crate::spatial::records().into_iter().find(|(name, _)| name == &record.name);
            if expected.is_none_or(|(_, fields)| record.fields.iter().map(|f| (&f.name, &f.ty)).ne(fields.iter().map(|(name, ty)| (name, ty)))) {
                return Err(invalid("opaque or malformed spatial record declaration"));
            }
        }
        if record.fields.is_empty()
            || record.fields.len() > 16
            || record.fields.iter().any(|field| !matches!(field.ty, Type::I32 | Type::U32 | Type::F32 | Type::F64 | Type::Bool | Type::ArrayI32 | Type::ArrayU32 | Type::ArrayF32 | Type::ArrayF64 | Type::ArrayBool))
            || record.fields.iter().map(|field| field.name.as_str()).collect::<HashSet<_>>().len() != record.fields.len()
            || records.insert(record.name.as_str(), record).is_some()
        {
            return Err(invalid("invalid MIR record declaration"));
        }
    }
    let mut variants = HashMap::new();
    for variant in &program.variants {
        if variant.name.rsplit("::").next().is_some_and(crate::spatial::is_builtin) { return Err(invalid("spatial builtin variant redefined")); }
        let result_payloads = crate::front::result_parts(&variant.name);
        if variant.arms.is_empty()
            || variant.arms.len() > 16
            || records.contains_key(variant.name.as_str())
            || (variant.name.starts_with("Result<") && result_payloads.is_none())
            || variant.arms.iter().any(|arm| arm.payload.as_ref().is_some_and(|ty| !matches!(ty, Type::I32 | Type::U32 | Type::F32 | Type::F64 | Type::Bool) && result_payloads.is_none()))
            || result_payloads.as_ref().is_some_and(|(ok, err)| variant.arms.len() != 2 || variant.arms[0].name != "Ok" || variant.arms[0].payload.as_ref() != Some(ok) || variant.arms[1].name != "Err" || variant.arms[1].payload.as_ref() != Some(err))
            || variant.arms.iter().map(|arm| arm.name.as_str()).collect::<HashSet<_>>().len() != variant.arms.len()
            || variants.insert(variant.name.as_str(), variant).is_some()
        {
            return Err(invalid("invalid MIR variant declaration"));
        }
    }
    for function in &program.functions {
        if let Some(oracle) = &function.foreign_oracle {
            let Some(target) = signatures.get(oracle.as_str()) else { return Err(invalid("foreign oracle missing")); };
            if target.foreign_oracle.is_some() { return Err(invalid("foreign oracle cannot be foreign")); }
            if !function.blocks.is_empty() || !function.locals.is_empty()
                || function.declaration_order.iter().map(String::as_str).ne(function.params.iter().map(|param| param.name.as_str()))
                || function.result != target.result
                || function.params.iter().map(|param| (&param.ty, param.mode)).ne(target.params.iter().map(|param| (&param.ty, param.mode))) {
                return Err(invalid("foreign declaration differs from oracle"));
            }
        } else {
            verify_function(function, &signatures, &records, &variants)?;
        }
    }
    Ok(())
}

fn verify_function(
    function: &Function,
    signatures: &HashMap<&str, &Function>,
    records: &HashMap<&str, &crate::front::RecordDecl>,
    variants: &HashMap<&str, &crate::front::VariantDecl>,
) -> Result<(), Error> {
    let count = function.blocks.len();
    if function.entry >= count {
        return Err(invalid("missing entry block"));
    }
    if !function.blocks[function.entry].params.is_empty() {
        return Err(invalid("entry block has parameters"));
    }
    match (&function.result, &function.return_from) {
        (Type::Named(view), Some(source)) if crate::front::view_parts(view).is_some() => {
            let (_, mutable) = crate::front::view_parts(view).ok_or_else(|| invalid("invalid MIR view result"))?;
            let param = function.params.iter().find(|param| param.name == *source).ok_or_else(|| invalid("MIR view return source missing"))?;
            if !matches!(param.ty, Type::ArrayI32 | Type::ArrayU32 | Type::ArrayF32 | Type::ArrayF64 | Type::ArrayBool)
                || (mutable && param.mode != Some(Mode::Mut))
                || (!mutable && !matches!(param.mode, Some(Mode::Read | Mode::Mut))) {
                return Err(invalid("MIR view return source has wrong mode"));
            }
        }
        (Type::Named(view), None) if crate::front::view_parts(view).is_some() => return Err(invalid("MIR view result has no source")),
        (_, Some(_)) => return Err(invalid("MIR nonview result declares source")),
        _ => {}
    }
    for param in &function.params {
        if param.ty == Type::Unit
            || affine(&param.ty) != param.mode.is_some()
            || (crate::spatial::is_grid(&param.ty) && param.mode == Some(Mode::Mut))
            || matches!(&param.ty, Type::Named(view) if crate::front::view_parts(view).is_some())
        {
            return Err(invalid("invalid MIR parameter mode or type"));
        }
        let local = function
            .locals
            .get(&param.name)
            .ok_or_else(|| invalid("missing MIR parameter place"))?;
        if local.ty != param.ty || local.mode != param.mode || local.mutable {
            return Err(invalid("MIR parameter place mismatch"));
        }
    }
    let unique: HashSet<&String> = function.declaration_order.iter().collect();
    if unique.len() != function.declaration_order.len()
        || unique.len() != function.locals.len()
        || unique
            .iter()
            .any(|name| !function.locals.contains_key(*name))
    {
        return Err(invalid("MIR declaration order mismatch"));
    }
    if function.locals.values().any(|local| local.ty == Type::Unit) {
        return Err(invalid("Unit MIR place is unsupported"));
    }
    let order: HashMap<&str, usize> = function
        .declaration_order
        .iter()
        .enumerate()
        .map(|(index, name)| (name.as_str(), index))
        .collect();
    let mut predecessors = vec![Vec::new(); count];
    let mut defs = HashMap::new();
    for (block_id, block) in function.blocks.iter().enumerate() {
        let Some(end) = &block.end else {
            return Err(invalid("missing terminator"));
        };
        if !contained(end.span(), function.span) {
            return Err(invalid("stale terminator span"));
        }
        for (target, args) in end.targets() {
            let successor = function
                .blocks
                .get(target)
                .ok_or_else(|| invalid("missing successor block"))?;
            if args.len() != successor.params.len() {
                return Err(invalid("successor parameter arity mismatch"));
            }
            predecessors[target].push(block_id);
        }
        for (id, ty) in &block.params {
            if defs
                .insert(
                    *id,
                    Definition {
                        block: block_id,
                        position: None,
                        ty: ty.clone(),
                    },
                )
                .is_some()
            {
                return Err(invalid("duplicate MIR value"));
            }
        }
        for (position, instruction) in block.instructions.iter().enumerate() {
            if !contained(instruction.span, function.span) {
                return Err(invalid("stale source span"));
            }
            if let Some((id, ty)) = &instruction.output
                && defs
                    .insert(
                        *id,
                        Definition {
                            block: block_id,
                            position: Some(position),
                            ty: ty.clone(),
                        },
                    )
                    .is_some()
            {
                return Err(invalid("duplicate MIR value"));
            }
        }
    }
    let dominance = Dominance::new(function, &predecessors);
    let types = defs;
    for block_id in &dominance.order {
        let block = &function.blocks[*block_id];
        for (position, instruction) in block.instructions.iter().enumerate() {
            for id in instruction.kind.uses() {
                check_use(id, *block_id, position, &types, &dominance)?;
            }
            if let Operation::Unary { operator, .. } | Operation::Binary { operator, .. } =
                &instruction.kind
                && !contained(*operator, instruction.span)
            {
                return Err(invalid("stale operator span"));
            }
            check_operation(
                &instruction.kind,
                instruction.output.as_ref().map(|(_, ty)| ty),
                function,
                signatures,
                records,
                variants,
                &types,
            )?;
            if let Operation::Drop(names) = &instruction.kind {
                let positions = names
                    .iter()
                    .map(|name| {
                        order
                            .get(name.as_str())
                            .copied()
                            .ok_or_else(|| invalid("unknown cleanup place"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if positions.windows(2).any(|pair| pair[0] <= pair[1]) {
                    return Err(invalid("cleanup order mismatch"));
                }
            }
        }
        let end = block.end.as_ref().expect("terminator checked");
        for id in end.uses() {
            check_use(id, *block_id, block.instructions.len(), &types, &dominance)?;
        }
        match end {
            Terminator::Goto { target, args, .. } => {
                for (id, (_, ty)) in args.iter().zip(&function.blocks[*target].params) {
                    if &types[id].ty != ty {
                        return Err(invalid("successor argument type mismatch"));
                    }
                }
            }
            Terminator::Branch { condition, .. } if types[condition].ty != Type::Bool => {
                return Err(invalid("branch condition must be Bool"));
            }
            Terminator::Return { value, cleanup, .. } => {
                if value.map(|id| &types[&id].ty)
                    != (function.result != Type::Unit).then_some(&function.result)
                {
                    return Err(invalid("return type mismatch"));
                }
                for name in cleanup {
                    if !function.locals.contains_key(name) {
                        return Err(invalid("unknown cleanup place"));
                    }
                }
                if let Some(id) = value
                    && affine(&types[id].ty)
                {
                    let definition = &types[id];
                    if definition
                        .position
                        .and_then(|position| {
                            function.blocks[definition.block].instructions.get(position)
                        })
                        .is_some_and(|instruction| {
                            reference(&instruction.kind).is_some()
                        })
                    {
                        return Err(invalid("borrowed array returned as owner"));
                    }
                }
            }
            _ => {}
        }
    }
    let mut consumed = HashSet::new();
    let mut loan_uses = HashMap::<ValueId, usize>::new();
    for block_id in &dominance.order {
        let block = &function.blocks[*block_id];
        for instruction in &block.instructions {
            for id in instruction.kind.uses() {
                let definition = &types[&id];
                if definition
                    .position
                    .and_then(|position| {
                        function.blocks[definition.block].instructions.get(position)
                    })
                    .is_some_and(|origin| {
                        reference(&origin.kind).flatten().is_some()
                    })
                {
                    if !matches!(instruction.kind, Operation::Call { .. }) {
                        return Err(invalid("loan value escapes its call"));
                    }
                    *loan_uses.entry(id).or_default() += 1;
                }
            }
            let ownership = match &instruction.kind {
                Operation::Bind { name, value } | Operation::Store { name, value }
                    if affine(&function.locals[name].ty) =>
                {
                    vec![*value]
                }
                Operation::Call { name, args } => signatures
                    .get(name.as_str())
                    .map(|callee| {
                        args.iter()
                            .zip(&callee.params)
                            .filter_map(|(arg, param)| {
                                (param.mode == Some(Mode::Take)).then_some(*arg)
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
                _ => Vec::new(),
            };
            for id in ownership {
                if !consumed.insert(id) {
                    return Err(invalid("owned MIR value reused"));
                }
            }
        }
        let end = block.end.as_ref().expect("terminator checked");
        for id in end.uses() {
            let definition = &types[&id];
            if definition
                .position
                .and_then(|position| function.blocks[definition.block].instructions.get(position))
                .is_some_and(|origin| {
                    reference(&origin.kind).flatten().is_some()
                })
            {
                return Err(invalid("loan value escapes its call"));
            }
        }
        if let Terminator::Return {
            value: Some(id), ..
        } = end
            && affine(&types[id].ty)
            && !consumed.insert(*id)
        {
            return Err(invalid("owned MIR value reused"));
        }
    }
    for block_id in &dominance.order {
        for instruction in &function.blocks[*block_id].instructions {
            if let Some((id, _)) = &instruction.output
                && reference(&instruction.kind).flatten().is_some()
                && loan_uses.get(id) != Some(&1)
            {
                return Err(invalid("loan must be used once by a call"));
            }
        }
    }
    let initial = FlowState {
        available: function
            .params
            .iter()
            .map(|param| param.name.clone())
            .collect(),
        loans: HashMap::new(),
        views: HashMap::new(),
        view_values: HashMap::new(),
        partial: HashMap::new(),
        taken_payloads: HashSet::new(),
        dropped_values: HashSet::new(),
    };
    let mut incoming = vec![None; count];
    incoming[function.entry] = Some(initial);
    let mut pending = VecDeque::from([function.entry]);
    while let Some(block_id) = pending.pop_front() {
        let mut state = incoming[block_id]
            .clone()
            .ok_or_else(|| invalid("missing flow state"))?;
        let block = &function.blocks[block_id];
        for instruction in &block.instructions {
            transfer(instruction, &mut state, signatures, function, &types)?;
        }
        let end = block.end.as_ref().expect("terminator checked");
        if state.view_values.keys().any(|id| match end {
            Terminator::Return { value, .. } => value != &Some(*id),
            _ => true,
        }) {
            return Err(invalid("temporary MIR view must bind or return once"));
        }
        if let Terminator::Return { value, .. } = end
            && matches!(&function.result, Type::Named(view) if crate::front::view_parts(view).is_some()) {
            let id = value.ok_or_else(|| invalid("MIR view return lacks value"))?;
            let fact = state.view_values.get(&id).ok_or_else(|| invalid("MIR view return lacks provenance"))?;
            if function.return_from.as_deref() != Some(fact.root.as_str()) {
                return Err(invalid("MIR view return escapes source lifetime"));
            }
        }
        if let Terminator::Return { .. } = end
            && !state.loans.is_empty()
        {
            return Err(invalid("loan escapes MIR return"));
        }
        if let Terminator::Return { cleanup, .. } = end {
            let expected: Vec<&str> = function
                .declaration_order
                .iter()
                .rev()
                .map(String::as_str)
                .filter(|name| {
                    state.available.contains(*name)
                        && function.locals.get(*name).is_some_and(|local| {
                            affine(&local.ty)
                                && !matches!(local.mode, Some(Mode::Read | Mode::Mut))
                        })
                })
                .collect();
            let actual: Vec<&str> = cleanup
                .iter()
                .map(String::as_str)
                .filter(|name| expected.contains(name))
                .collect();
            if actual != expected {
                return Err(invalid("return cleanup mismatch"));
            }
        }
        for (successor, _) in end.targets() {
            let next = match &incoming[successor] {
                Some(prior) => {
                    if prior.loans != state.loans || prior.views != state.views {
                        return Err(invalid("incompatible loan or view join"));
                    }
                    FlowState {
                        available: prior
                            .available
                            .intersection(&state.available)
                            .cloned()
                            .collect(),
                        loans: state.loans.clone(),
                        views: state.views.clone(),
                        view_values: prior.view_values.iter().filter(|(id, fact)| state.view_values.get(id) == Some(*fact)).map(|(id, fact)| (*id, fact.clone())).collect(),
                        partial: {
                            let mut fields = prior.partial.clone();
                            for (name, moved) in &state.partial {
                                fields.entry(name.clone()).or_default().extend(moved.iter().copied());
                            }
                            fields
                        },
                        taken_payloads: prior.taken_payloads.union(&state.taken_payloads).copied().collect(),
                        dropped_values: prior.dropped_values.union(&state.dropped_values).copied().collect(),
                    }
                }
                None => state.clone(),
            };
            if incoming[successor].as_ref() != Some(&next) {
                incoming[successor] = Some(next);
                pending.push_back(successor);
            }
        }
    }
    Ok(())
}
