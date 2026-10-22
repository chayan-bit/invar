fn check_operation(
    op: &Operation,
    output: Option<&Type>,
    function: &Function,
    signatures: &HashMap<&str, &Function>,
    records: &HashMap<&str, &crate::front::RecordDecl>,
    variants: &HashMap<&str, &crate::front::VariantDecl>,
    types: &HashMap<ValueId, Definition>,
) -> Result<(), Error> {
    let ty = |id: &ValueId| {
        types
            .get(id)
            .map(|definition| &definition.ty)
            .ok_or_else(|| invalid("undefined MIR value"))
    };
    let local = |name: &str| {
        function
            .locals
            .get(name)
            .ok_or_else(|| invalid("unknown MIR place"))
    };
    let expected = match op {
        Operation::Tick
        | Operation::Bind { .. }
        | Operation::Store { .. }
        | Operation::StoreIndex { .. }
        | Operation::Drop(_) => None,
        Operation::DropValue(value) => {
            let Type::Named(name) = ty(value)? else { return Err(invalid("temporary cleanup requires Result")); };
            if crate::front::result_parts(name).is_none() { return Err(invalid("temporary cleanup requires Result")); }
            let definition = types.get(value).ok_or_else(|| invalid("temporary Result value missing"))?;
            if definition.position.and_then(|position| function.blocks[definition.block].instructions.get(position))
                .is_some_and(|origin| matches!(origin.kind, Operation::RecordRef { .. } | Operation::ArrayRef { .. })) {
                return Err(invalid("borrowed Result cannot be cleaned as a temporary"));
            }
            None
        }
        Operation::I32(_) => Some(Type::I32),
        Operation::F32(_) => Some(Type::F32),
        Operation::F64(_) => Some(Type::F64),
        Operation::U32(_) | Operation::ForNext(_) | Operation::Length(_) => Some(Type::U32),
        Operation::Bool(_) => Some(Type::Bool),
        Operation::Blob { bytes, text } => {
            if bytes.len() > 1_048_576 || (*text && std::str::from_utf8(bytes).is_err()) {
                return Err(invalid("blob literal violates encoding or size limit"));
            }
            Some(Type::Named(if *text { "String" } else { "Bytes" }.to_owned()))
        }
        Operation::Read(name) => {
            let ty = &local(name)?.ty;
            if matches!(ty, Type::ArrayI32 | Type::ArrayU32 | Type::ArrayF32 | Type::ArrayF64 | Type::ArrayBool | Type::Named(_)) {
                return Err(invalid("array read must carry ownership mode"));
            }
            Some(ty.clone())
        }
        Operation::Move(name) => {
            let binding = local(name)?;
            if !matches!(binding.ty, Type::ArrayI32 | Type::ArrayU32 | Type::ArrayF32 | Type::ArrayF64 | Type::ArrayBool | Type::Named(_))
                || matches!(binding.mode, Some(Mode::Read | Mode::Mut))
            {
                return Err(invalid("invalid array move"));
            }
            Some(binding.ty.clone())
        }
        Operation::MoveField { name, index } => {
            let Type::Named(record) = &local(name)?.ty else { return Err(invalid("field move base is not a record")); };
            let declaration = records.get(record.as_str()).ok_or_else(|| invalid("field move record missing"))?;
            let field = declaration.fields.get(*index).ok_or_else(|| invalid("field move index out of bounds"))?;
            if !matches!(field.ty, Type::ArrayI32 | Type::ArrayU32 | Type::ArrayF32 | Type::ArrayF64 | Type::ArrayBool)
                || matches!(local(name)?.mode, Some(Mode::Read | Mode::Mut)) {
                return Err(invalid("invalid field move ownership"));
            }
            Some(field.ty.clone())
        }
        Operation::ArrayRef { name, loan } => {
            let binding = local(name)?;
            if !matches!(&binding.ty, Type::ArrayI32 | Type::ArrayU32 | Type::ArrayF32 | Type::ArrayF64 | Type::ArrayBool)
                && !matches!(&binding.ty, Type::Named(name) if crate::front::view_parts(name).is_some())
                || *loan == Some(Mode::Take)
                || loan.is_some() && matches!(&binding.ty, Type::Named(view) if crate::front::view_parts(view).is_some()) {
                return Err(invalid("invalid array reference"));
            }
            if *loan == Some(Mode::Mut) && !binding.mutable && binding.mode != Some(Mode::Mut) {
                return Err(invalid("mutable loan lacks permission"));
            }
            Some(binding.ty.clone())
        }
        Operation::RecordRef { name, loan } => {
            let binding = local(name)?;
            if !matches!(&binding.ty, Type::Named(record) if crate::front::view_parts(record).is_none()) || *loan == Some(Mode::Take) {
                return Err(invalid("invalid record reference"));
            }
            if *loan == Some(Mode::Mut) && (crate::spatial::is_grid(&binding.ty) || (!binding.mutable && binding.mode != Some(Mode::Mut))) {
                return Err(invalid("mutable record loan lacks permission"));
            }
            Some(binding.ty.clone())
        }
        Operation::Array(items) => {
            let first = items.first().map(&ty).transpose()?;
            if items.iter().any(|item| ty(item).ok() != first) {
                return Err(invalid("mixed array elements"));
            }
            match first {
                Some(Type::I32) => Some(Type::ArrayI32),
                Some(Type::U32) => Some(Type::ArrayU32),
                Some(Type::F32) => Some(Type::ArrayF32),
                Some(Type::F64) => Some(Type::ArrayF64),
                Some(Type::Bool) => Some(Type::ArrayBool),
                None => output
                    .cloned()
                    .filter(|ty| matches!(ty, Type::ArrayI32 | Type::ArrayU32 | Type::ArrayF32 | Type::ArrayF64 | Type::ArrayBool)),
                _ => return Err(invalid("invalid array element type")),
            }
        }
        Operation::Record { name, fields } => {
            let declaration = records.get(name.as_str()).ok_or_else(|| invalid("unknown MIR record"))?;
            if fields.len() != declaration.fields.len() {
                return Err(invalid("record initializer arity mismatch"));
            }
            let mut seen = HashSet::new();
            for (index, value) in fields {
                let field = declaration.fields.get(*index).ok_or_else(|| invalid("record field index out of bounds"))?;
                if !seen.insert(*index) || ty(value)? != &field.ty {
                    return Err(invalid("record initializer field mismatch"));
                }
            }
            Some(Type::Named(name.clone()))
        }
        Operation::Field { record, index } => {
            let Type::Named(name) = ty(record)? else { return Err(invalid("field base is not a record")); };
            let declaration = records.get(name.as_str()).ok_or_else(|| invalid("unknown MIR record"))?;
            Some(declaration.fields.get(*index).ok_or_else(|| invalid("record field index out of bounds"))?.ty.clone())
        }
        Operation::Variant { name, tag, payload } => {
            let declaration = variants.get(name.as_str()).ok_or_else(|| invalid("unknown MIR variant"))?;
            let expected = declaration.arms.get(*tag).ok_or_else(|| invalid("variant tag out of bounds"))?.payload.as_ref();
            if expected != payload.map(|id| ty(&id)).transpose()? {
                return Err(invalid("variant constructor payload mismatch"));
            }
            Some(Type::Named(name.clone()))
        }
        Operation::VariantTag(value) => {
            let Type::Named(name) = ty(value)? else { return Err(invalid("variant tag base is not nominal")); };
            if !variants.contains_key(name.as_str()) { return Err(invalid("variant tag base is not variant")); }
            Some(Type::U32)
        }
        Operation::VariantPayload { value, tag } | Operation::VariantTakePayload { value, tag } => {
            let Type::Named(name) = ty(value)? else { return Err(invalid("variant payload base is not nominal")); };
            if matches!(op, Operation::VariantTakePayload { .. }) && crate::front::result_parts(name).is_none() {
                return Err(invalid("owned variant payload take requires Result"));
            }
            if matches!(op, Operation::VariantTakePayload { .. }) {
                let definition = types.get(value).ok_or_else(|| invalid("Result owner missing"))?;
                if let Some(Operation::RecordRef { name, .. }) = definition.position
                    .and_then(|position| function.blocks[definition.block].instructions.get(position))
                    .map(|origin| &origin.kind)
                    && matches!(local(name)?.mode, Some(Mode::Read | Mode::Mut)) {
                    return Err(invalid("borrowed Result payload cannot be taken"));
                }
            }
            let declaration = variants.get(name.as_str()).ok_or_else(|| invalid("variant payload base is not variant"))?;
            Some(declaration.arms.get(*tag).and_then(|arm| arm.payload.clone()).ok_or_else(|| invalid("variant payload arm mismatch"))?)
        }
        Operation::Unary { op, value, .. } => match (op, ty(value)?) {
            (UnaryOp::Not, Type::Bool) => Some(Type::Bool),
            (UnaryOp::Neg, Type::I32) => Some(Type::I32),
            (UnaryOp::Neg, Type::F32) => Some(Type::F32),
            (UnaryOp::Neg, Type::F64) => Some(Type::F64),
            _ => return Err(invalid("invalid unary operand")),
        },
        Operation::Binary {
            op, left, right, ..
        } => result_type(*op, ty(left)?, ty(right)?),
        Operation::Index { array, index } => {
            if ty(index)? != &Type::U32 {
                return Err(invalid("index must be u32"));
            }
            match ty(array)? {
                Type::ArrayI32 => Some(Type::I32),
                Type::ArrayU32 => Some(Type::U32),
                Type::ArrayF32 => Some(Type::F32),
                Type::ArrayF64 => Some(Type::F64),
                Type::ArrayBool => Some(Type::Bool),
                Type::Named(name) if name == "Bytes" => Some(Type::U32),
                Type::Named(name) => crate::front::view_parts(name).map(|(element, _)| element),
                _ => return Err(invalid("indexed nonarray")),
            }
        }
        Operation::Call { name, args } if crate::check::scale_call(name).is_some() || crate::check::angle_call(name).is_some() => {
            if args.len() != 1 { return Err(invalid("checked scale arity mismatch")); }
            match ty(&args[0])? {
                Type::I32 => Some(Type::I32),
                Type::U32 => Some(Type::U32),
                _ => return Err(invalid("checked scale requires integer")),
            }
        }
        Operation::Call { name, args } if crate::spatial::result(name).is_some() => {
            let (params, result) = crate::spatial::signature(name).ok_or_else(|| invalid("spatial signature missing"))?;
            if args.len() != params.len() { return Err(invalid("spatial call arity mismatch")); }
            for (position, (arg, param)) in args.iter().zip(&params).enumerate() {
                if ty(arg)? != param { return Err(invalid("spatial argument type mismatch")); }
                if let Some(mode) = crate::spatial::mode(name, position) {
                    let definition = types.get(arg).ok_or_else(|| invalid("spatial argument missing"))?;
                    let origin = definition.position.and_then(|position| function.blocks[definition.block].instructions.get(position));
                    if origin.and_then(|op| reference(&op.kind)) != Some(Some(mode)) {
                        return Err(invalid("spatial argument loan mode mismatch"));
                    }
                }
            }
            if let Type::Named(name) = &result
                && name != crate::spatial::GRID && !records.contains_key(name.as_str()) {
                return Err(invalid("spatial result record declaration missing"));
            }
            Some(result)
        }
        Operation::Call { name, args } if crate::check::sort_modes(name).is_some() => {
            let modes = crate::check::sort_modes(name).ok_or_else(|| invalid("sort modes missing"))?;
            if args.len() != modes.len() { return Err(invalid("sort call arity mismatch")); }
            for (index, (arg, mode)) in args.iter().zip(modes).enumerate() {
                let valid = if index == 0 {
                    matches!(ty(arg)?, Type::ArrayI32 | Type::ArrayU32 | Type::ArrayF32)
                } else if name == "segmented_sort" { ty(arg)? == &Type::ArrayU32 }
                else { matches!(ty(arg)?, Type::ArrayI32 | Type::ArrayU32 | Type::ArrayF32 | Type::ArrayBool) };
                let definition = types.get(arg).ok_or_else(|| invalid("sort argument missing"))?;
                let origin = definition.position.and_then(|position| function.blocks[definition.block].instructions.get(position));
                if !valid || origin.and_then(|op| reference(&op.kind)) != Some(Some(*mode)) {
                    return Err(invalid("sort argument type or loan mode mismatch"));
                }
            }
            Some(if name == "argsort" { Type::ArrayU32 } else { Type::Unit })
        }
        Operation::Call { name, args } => match name.as_str() {
            float if crate::eval::floating::handles(float) => {
                let types = args.iter().map(&ty).collect::<Result<Vec<_>, _>>()?;
                Some(float_intrinsic(float, &types).ok_or_else(|| invalid("float intrinsic operand mismatch"))?)
            }
            "bytes_slice" | "utf8_slice" | "utf8_decode" | "utf8_encode" => {
                let sliced = matches!(name.as_str(), "bytes_slice" | "utf8_slice");
                if args.len() != if sliced { 3 } else { 1 }
                    || args.iter().skip(1).any(|arg| ty(arg).ok() != Some(&Type::U32)) {
                    return Err(invalid("text intrinsic arity or bounds mismatch"));
                }
                let input = if matches!(name.as_str(), "bytes_slice" | "utf8_decode") { "Bytes" } else { "String" };
                if ty(&args[0])? != &Type::Named(input.to_owned()) {
                    return Err(invalid("text intrinsic input type mismatch"));
                }
                let source = types.get(&args[0]).ok_or_else(|| invalid("text intrinsic source missing"))?;
                let Some(Operation::RecordRef { name: owner, loan: None }) = source.position
                    .and_then(|position| function.blocks[source.block].instructions.get(position))
                    .map(|instruction| &instruction.kind) else {
                    return Err(invalid("text intrinsic needs a named immutable owner"));
                };
                if &local(owner)?.ty != ty(&args[0])? {
                    return Err(invalid("text intrinsic owner type mismatch"));
                }
                Some(Type::Named(if matches!(name.as_str(), "bytes_slice" | "utf8_encode") { "Bytes" } else { "Result<String,Utf8Error>" }.to_owned()))
            }
            "view" | "view_mut" if (3..=4).contains(&args.len()) && args.iter().skip(1).all(|arg| ty(arg).ok() == Some(&Type::U32)) => {
                let source = ty(&args[0])?;
                let element = match source {
                    Type::ArrayI32 => Type::I32,
                    Type::ArrayU32 => Type::U32,
                Type::ArrayF32 => Type::F32,
                Type::ArrayF64 => Type::F64,
                Type::ArrayBool => Type::Bool,
                    Type::Named(name) => crate::front::view_parts(name).map(|(element, _)| element).ok_or_else(|| invalid("view source must be array or view"))?,
                    _ => return Err(invalid("view source must be array or view")),
                };
                let definition = types.get(&args[0]).ok_or_else(|| invalid("view source missing"))?;
                let Some(Operation::ArrayRef { name: local_name, loan: None }) = definition.position
                    .and_then(|position| function.blocks[definition.block].instructions.get(position))
                    .map(|item| &item.kind) else { return Err(invalid("view source must be a named reference")); };
                let binding = local(local_name)?;
                if name == "view_mut" && !binding.mutable && binding.mode != Some(Mode::Mut)
                    && !matches!(&binding.ty, Type::Named(view) if crate::front::view_parts(view).is_some_and(|(_, mutable)| mutable)) {
                    return Err(invalid("mutable view lacks permission"));
                }
                let kind = if name == "view_mut" { "MutView" } else { "View" };
                let item = if element == Type::I32 { "i32" } else { "u32" };
                Some(Type::Named(format!("{kind}<{item}>")))
            }
            reduction if crate::fsum::Kind::from_name(reduction).is_some() => {
                let arity = if reduction.starts_with("dot_") { 2 } else { 1 };
                if args.len() != arity {
                    return Err(invalid("reduction arity mismatch"));
                }
                for arg in args {
                    let source = types.get(arg).ok_or_else(|| invalid("reduction operand missing"))?;
                    let named = source.position
                        .and_then(|position| function.blocks[source.block].instructions.get(position))
                        .is_some_and(|item| matches!(&item.kind, Operation::ArrayRef { loan: None, .. }));
                    if ty(arg)? != &Type::ArrayF32 || !named {
                        return Err(invalid("reduction operand must be a named Array<f32>"));
                    }
                }
                Some(Type::F32)
            }
            "filled_i32"
                if args.len() == 2
                    && ty(&args[0])? == &Type::U32
                    && ty(&args[1])? == &Type::I32 =>
            {
                Some(Type::ArrayI32)
            }
            "filled_u32"
                if args.len() == 2
                    && ty(&args[0])? == &Type::U32
                    && ty(&args[1])? == &Type::U32 =>
            {
                Some(Type::ArrayU32)
            }
            "wrapping_add" | "wrapping_mul"
                if args.len() == 2 && args.iter().all(|arg| ty(arg).ok() == Some(&Type::U32)) =>
            {
                Some(Type::U32)
            }
            _ => {
                let callee = signatures
                    .get(name.as_str())
                    .ok_or_else(|| invalid("unknown MIR call"))?;
                if args.len() != callee.params.len() {
                    return Err(invalid("MIR call arity mismatch"));
                }
                for (arg, param) in args.iter().zip(&callee.params) {
                    if ty(arg)? != &param.ty {
                        return Err(invalid("MIR call argument type mismatch"));
                    }
                    let definition = types
                        .get(arg)
                        .ok_or_else(|| invalid("missing call argument definition"))?;
                    let instruction = definition.position.and_then(|position| {
                        function.blocks[definition.block].instructions.get(position)
                    });
                    if matches!(param.mode, Some(Mode::Read | Mode::Mut)) {
                        if instruction.and_then(|instruction| reference(&instruction.kind)) != Some(param.mode)
                        {
                            return Err(invalid("call loan mode mismatch"));
                        }
                    } else if param.mode == Some(Mode::Take)
                        && instruction.is_some_and(|instruction| {
                            reference(&instruction.kind).is_some()
                        })
                    {
                        return Err(invalid("borrowed array passed to take"));
                    }
                }
                Some(callee.result.clone())
            }
        },
    };
    if let Operation::ForNext(value) = op
        && ty(value)? != &Type::U32
    {
        return Err(invalid("for increment must be u32"));
    }
    if let Operation::Length(name) = op
        && !matches!(&local(name)?.ty, Type::ArrayI32 | Type::ArrayU32 | Type::ArrayF32 | Type::ArrayF64 | Type::ArrayBool)
        && !matches!(&local(name)?.ty, Type::Named(view) if view == "Bytes" || view == "String" || crate::front::view_parts(view).is_some())
    {
        return Err(invalid("len requires array"));
    }
    match op {
        Operation::Bind { name, value } | Operation::Store { name, value } => {
            let binding = local(name)?;
            if ty(value)? != &binding.ty {
                return Err(invalid("place/value type mismatch"));
            }
            if affine(&binding.ty) {
                let definition = &types[value];
                if definition
                    .position
                    .and_then(|position| {
                        function.blocks[definition.block].instructions.get(position)
                    })
                    .is_some_and(|instruction| {
                        reference(&instruction.kind).is_some()
                    })
                {
                    return Err(invalid("borrowed array stored as owner"));
                }
            }
            if matches!(op, Operation::Store { .. }) && !binding.mutable {
                return Err(invalid("store to immutable place"));
            }
            if matches!(op, Operation::Store { .. })
                && matches!(&binding.ty, Type::Named(view) if crate::front::view_parts(view).is_some()) {
                return Err(invalid("view place cannot be rebound"));
            }
        }
        Operation::StoreIndex { name, index, value } => {
            let binding = local(name)?;
            let element = match &binding.ty {
                Type::ArrayI32 => Type::I32,
                Type::ArrayU32 => Type::U32,
                Type::ArrayF32 => Type::F32,
                Type::ArrayF64 => Type::F64,
                Type::ArrayBool => Type::Bool,
                Type::Named(view) => crate::front::view_parts(view).filter(|(_, mutable)| *mutable).map(|(element, _)| element).ok_or_else(|| invalid("indexed store requires mutable view"))?,
                _ => return Err(invalid("indexed store to nonarray")),
            };
            if ty(index)? != &Type::U32
                || ty(value)? != &element
                || !(binding.mutable || binding.mode == Some(Mode::Mut) || matches!(&binding.ty, Type::Named(view) if crate::front::view_parts(view).is_some_and(|(_, mutable)| mutable)))
            {
                return Err(invalid("invalid indexed store"));
            }
        }
        Operation::Drop(names) => {
            for name in names {
                local(name)?;
            }
        }
        _ => {}
    }
    if output != expected.as_ref() {
        return Err(invalid("MIR output type mismatch"));
    }
    Ok(())
}
