impl Evaluator<'_> {
    fn charge(&mut self) -> Result<(), EvalError> {
        if self.steps_used >= self.fuel_limit {
            return Err(EvalError::EvaluationLimit {
                fuel_limit: self.fuel_limit,
                steps_used: self.steps_used,
            });
        }
        self.steps_used += 1;
        Ok(())
    }

    fn call_user(&mut self, name: &str, args: Vec<Value>) -> Result<Value, EvalError> {
        if self.call_depth >= MAX_CALL_DEPTH {
            return Err(EvalError::tool(
                "CALL_DEPTH_LIMIT",
                "reference call depth limit exceeded",
            ));
        }
        let function = self
            .program
            .program
            .function(name)
            .ok_or_else(|| EvalError::tool("INTERNAL_CALL", "missing checked function"))?;
        if args.len() != function.params.len() {
            return Err(EvalError::tool(
                "INTERNAL_CALL",
                "invalid checked call arity",
            ));
        }
        let tc = crate::contraction_source::is_tc(&self.program.program, name)
            .then(|| floating::TcInputs::capture(&args)).transpose()?;
        self.call_depth += 1;
        let previous_types = std::mem::replace(&mut self.types, self.program.types_for(name));
        let mut env = Env::new(function, args);
        let result = self.block(&function.body, &mut env);
        self.types = previous_types;
        self.call_depth -= 1;
        env.pop(&mut self.trace, &mut self.live_elements, result.is_ok() || self.pending_result_error.is_some());
        if let Some(error) = self.pending_result_error.take() {
            admit_live(&mut self.live_elements, 2)?;
            return Ok(Value::Variant(Rc::new(RefCell::new((1, Some(error))))));
        }
        match result? {
            Flow::Return(value) => match tc { Some(tc) => tc.finish(value), None => Ok(value) },
            Flow::Continue if function.result == Type::Unit => Ok(Value::Unit),
            Flow::Continue => Err(EvalError::tool(
                "INTERNAL_RETURN",
                "checked function fell through",
            )),
        }
    }

    fn block(&mut self, block: &Block, env: &mut Env) -> Result<Flow, EvalError> {
        env.scopes.push(Scope::new());
        let result = (|| {
            for statement in &block.statements {
                match self.statement(statement, env)? {
                    Flow::Continue => {}
                    returned => return Ok(returned),
                }
            }
            Ok(Flow::Continue)
        })();
        env.pop(&mut self.trace, &mut self.live_elements, result.is_ok() || self.pending_result_error.is_some());
        result
    }

    fn statement(&mut self, statement: &Stmt, env: &mut Env) -> Result<Flow, EvalError> {
        self.charge()?;
        match &statement.kind {
            StmtKind::Binding { name, value, .. } => {
                let value = self.expr(value, env)?;
                env.bind(name, value)?;
                Ok(Flow::Continue)
            }
            StmtKind::Assign { place, value } => {
                let value = self.expr(value, env)?;
                match place {
                    Place::Name(name, _) => {
                        let old = env.set(name, value)?;
                        record_cleanup(&mut self.trace, &mut self.live_elements, name, &old);
                    }
                    Place::Index { name, index, span } => {
                        let index = self.expr(index, env)?;
                        let Value::U32(index) = index else {
                            return Err(EvalError::tool("INTERNAL_TYPE", "invalid checked index"));
                        };
                        env.set_index(name, index, value, *span)?;
                    }
                }
                Ok(Flow::Continue)
            }
            StmtKind::Return(expr) => {
                let value = match expr {
                    Some(Expr {
                        kind: ExprKind::Name(name),
                        ..
                    }) if matches!(env.slot(name)?, Value::ArrayI32(_) | Value::ArrayU32(_) | Value::ArrayF32(_) | Value::ArrayF64(_) | Value::ArrayBool(_) | Value::Bytes(_) | Value::Text(_) | Value::VoxelGrid(_) | Value::Record(_) | Value::Variant(_)) => {
                        // The consuming return context still evaluates a name.
                        // Charge before transferring its owner or exiting scopes.
                        self.charge()?;
                        env.take(name)?
                    }
                    Some(expr) => self.expr(expr, env)?,
                    None => Value::Unit,
                };
                Ok(Flow::Return(value))
            }
            StmtKind::If {
                condition,
                then_block,
                else_arm,
            } => {
                if self.expr(condition, env)? == Value::Bool(true) {
                    self.block(then_block, env)
                } else {
                    match else_arm {
                        Some(ElseArm::Block(block)) => self.block(block, env),
                        Some(ElseArm::If(statement)) => self.statement(statement, env),
                        None => Ok(Flow::Continue),
                    }
                }
            }
            StmtKind::While { condition, body } => {
                loop {
                    if self.expr(condition, env)? != Value::Bool(true) {
                        break;
                    }
                    if let Flow::Return(value) = self.block(body, env)? {
                        return Ok(Flow::Return(value));
                    }
                }
                Ok(Flow::Continue)
            }
            StmtKind::For {
                name,
                lower,
                upper,
                body,
            } => {
                let lo = self.expr(lower, env)?;
                let hi = self.expr(upper, env)?;
                let (Value::U32(mut i), Value::U32(hi)) = (lo, hi) else {
                    return Err(EvalError::tool(
                        "INTERNAL_TYPE",
                        "invalid checked for bounds",
                    ));
                };
                env.scopes.push(Scope::new());
                let result = (|| {
                    while i < hi {
                        self.charge()?;
                        env.bind(name, Value::U32(i))?;
                        if let Flow::Return(value) = self.block(body, env)? {
                            return Ok(Flow::Return(value));
                        }
                        // i < hi <= u32::MAX, so the increment cannot overflow.
                        i += 1;
                    }
                    Ok(Flow::Continue)
                })();
                env.pop(&mut self.trace, &mut self.live_elements, result.is_ok() || self.pending_result_error.is_some());
                result
            }
            StmtKind::Expr(expr) => {
                self.expr(expr, env)?;
                Ok(Flow::Continue)
            }
        }
    }

    fn expr(&mut self, expr: &Expr, env: &mut Env) -> Result<Value, EvalError> {
        // Parentheses carry a span for checking but are not a separate semantic step.
        if let ExprKind::Group(inner) = &expr.kind {
            return self.expr(inner, env);
        }
        self.charge()?;
        match &expr.kind {
            ExprKind::Integer(raw) => self.integer(raw, expr.span),
            ExprKind::Float(raw) => floating::literal(raw, self.types.get(&expr.span)),
            ExprKind::Bool(value) => Ok(Value::Bool(*value)),
            ExprKind::Text(value) => {
                admit_live(&mut self.live_elements, value.len())?;
                Ok(Value::Text(Rc::new(value.clone())))
            }
            ExprKind::Bytes(value) => {
                admit_live(&mut self.live_elements, value.len())?;
                Ok(Value::Bytes(Rc::new(value.clone())))
            }
            ExprKind::Name(name) => env.get_scalar(name),
            ExprKind::Move(name) => env.take(name),
            ExprKind::Try(inner) => {
                let Value::Variant(storage) = self.expr(inner, env)? else {
                    return Err(EvalError::tool("INTERNAL_TYPE", "checked ? input was not Result"));
                };
                let (tag, payload) = {
                    let mut storage = storage.borrow_mut();
                    (storage.0, storage.1.take())
                };
                record_cleanup(&mut self.trace, &mut self.live_elements, "<result>", &Value::Variant(storage));
                let payload = payload.ok_or_else(|| EvalError::tool("INTERNAL_TYPE", "checked Result payload missing"))?;
                if tag == 0 { Ok(payload) } else {
                    self.pending_result_error = Some(payload);
                    Err(EvalError::tool("INTERNAL_TYPE", "Result propagation is handled by the current function"))
                }
            }
            ExprKind::MoveField { base, field } => {
                let ExprKind::Name(binding) = &base.kind else {
                    return Err(EvalError::tool("INTERNAL_TYPE", "checked field move base missing"));
                };
                self.charge()?;
                let Type::Named(name) = self.types.get(&base.span)
                    .ok_or_else(|| EvalError::tool("INTERNAL_TYPE", "checked field owner type missing"))? else {
                    return Err(EvalError::tool("INTERNAL_TYPE", "checked field owner not nominal"));
                };
                let declaration = self.program.program.records.iter().find(|record| record.name == *name)
                    .ok_or_else(|| EvalError::tool("INTERNAL_TYPE", "checked record declaration missing"))?;
                let index = declaration.fields.iter().position(|item| item.name == *field)
                    .ok_or_else(|| EvalError::tool("INTERNAL_TYPE", "checked field index missing"))?;
                env.move_record_field(binding, index)
            }
            ExprKind::Array(items) => self.array_literal(expr, items, env),
            ExprKind::Record { name, fields } => {
                let declaration = self.program.program.records.iter().find(|record| record.name == *name)
                    .ok_or_else(|| EvalError::tool("INTERNAL_TYPE", "missing checked record declaration"))?;
                let order = declaration.fields.iter().map(|field| field.name.as_str()).collect::<Vec<_>>();
                let mut values = std::iter::repeat_with(|| Value::Unit).take(order.len()).collect::<Vec<_>>();
                for field in fields {
                    let value = self.expr(&field.value, env)?;
                    let index = order.iter().position(|name| *name == field.name)
                        .ok_or_else(|| EvalError::tool("INTERNAL_TYPE", "unknown checked record field"))?;
                    values[index] = value;
                }
                if values.iter().any(|value| matches!(value, Value::Unit)) {
                    return Err(EvalError::tool("INTERNAL_TYPE", "missing checked record field"));
                }
                admit_live(&mut self.live_elements, values.len())?;
                Ok(Value::Record(Rc::new(RefCell::new(values))))
            }
            ExprKind::Match { value, arms } => {
                let (name, moving) = match &value.kind {
                    ExprKind::Name(name) => (name, false),
                    ExprKind::Move(name) => (name, true),
                    _ => return Err(EvalError::tool("INTERNAL_TYPE", "checked match needs a named owner")),
                };
                self.charge()?;
                let Value::Variant(storage) = env.borrow_owned(name)? else {
                    return Err(EvalError::tool("INTERNAL_TYPE", "checked match owner is not a variant"));
                };
                let (tag, payload) = if moving {
                    let mut storage = storage.borrow_mut();
                    (storage.0, storage.1.take())
                } else {
                    let storage = storage.borrow();
                    let payload = storage.1.as_ref().map(|value| match value {
                        Value::I32(v) => Ok(Value::I32(*v)),
                        Value::U32(v) => Ok(Value::U32(*v)),
                        Value::Bool(v) => Ok(Value::Bool(*v)),
                        Value::F32(v) => Ok(Value::F32(*v)),
                        Value::F64(v) => Ok(Value::F64(*v)),
                        _ => Err(EvalError::tool("INTERNAL_TYPE", "borrowed owned variant payload")),
                    }).transpose()?;
                    (storage.0, payload)
                };
                let nominal = self.types.get(&value.span).and_then(|ty| match ty { Type::Named(name) => Some(name.as_str()), _ => None })
                    .ok_or_else(|| EvalError::tool("INTERNAL_TYPE", "checked match nominal type missing"))?;
                let declaration = self.program.program.variants.iter().find(|variant| variant.name == nominal)
                    .ok_or_else(|| EvalError::tool("INTERNAL_TYPE", "checked variant declaration missing"))?;
                let case = declaration.arms.get(tag).ok_or_else(|| EvalError::tool("INTERNAL_TYPE", "checked variant tag missing"))?;
                let selected = arms.iter().find(|arm| arm.variant.rsplit_once("::").is_some_and(|(owner, name)| (owner == nominal || (crate::front::result_parts(nominal).is_some() && owner == "Result")) && name == case.name))
                    .ok_or_else(|| EvalError::tool("INTERNAL_TYPE", "checked match arm missing"))?;
                env.scopes.push(Scope::new());
                if let (Some(binding), Some(payload)) = (&selected.binding, payload) {
                    env.bind(binding, payload)?;
                }
                let result = self.expr(&selected.value, env);
                env.pop(&mut self.trace, &mut self.live_elements, result.is_ok() || self.pending_result_error.is_some());
                if moving && (result.is_ok() || self.pending_result_error.is_some()) {
                    let owned = env.take(name)?;
                    record_cleanup(&mut self.trace, &mut self.live_elements, name, &owned);
                }
                result
            }
            ExprKind::Field { base, name } => {
                let ExprKind::Name(binding) = &base.kind else {
                    return Err(EvalError::tool("INTERNAL_TYPE", "invalid checked field base"));
                };
                self.charge()?;
                let Type::Named(record) = self.types.get(&base.span).ok_or_else(|| EvalError::tool("INTERNAL_TYPE", "missing field base type"))? else {
                    return Err(EvalError::tool("INTERNAL_TYPE", "invalid checked field base type"));
                };
                let declaration = self.program.program.records.iter().find(|declaration| declaration.name == *record)
                    .ok_or_else(|| EvalError::tool("INTERNAL_TYPE", "missing checked record declaration"))?;
                let index = declaration.fields.iter().position(|field| field.name == *name)
                    .ok_or_else(|| EvalError::tool("INTERNAL_TYPE", "missing checked field"))?;
                env.record_field(binding, index)
            }
            ExprKind::Index { base, index } => {
                if let Some(name) = named_array(base) {
                    self.charge()?;
                    env.slot(name)?;
                    let Value::U32(index) = self.expr(index, env)? else {
                        return Err(EvalError::tool("INTERNAL_TYPE", "invalid checked index"));
                    };
                    env.index(name, index, expr.span)
                } else {
                    let array = self.expr(base, env)?;
                    let Value::U32(index) = self.expr(index, env)? else {
                        return Err(EvalError::tool("INTERNAL_TYPE", "invalid checked index"));
                    };
                    index_owned(&array, index, expr.span)
                }
            }
            ExprKind::Group(_) => unreachable!("group handled before charging"),
            ExprKind::Call { name, args } => {
                if crate::spatial::result(name).is_some() {
                    let reads = crate::spatial::read_count(name);
                    let mut values = Vec::with_capacity(args.len());
                    for (i, arg) in args.iter().enumerate() {
                        values.push(if i < reads {
                            let ExprKind::Name(actual) = &arg.kind else {
                                return Err(EvalError::tool(
                                    "INTERNAL_BORROW",
                                    "spatial source is not named",
                                ));
                            };
                            self.charge()?;
                            env.borrow_owned(actual)?
                        } else {
                            self.expr(arg, env)?
                        });
                    }
                    return crate::spatial::invoke(
                        name,
                        &values,
                        expr.span,
                        &mut self.live_elements,
                    );
                }
                if crate::check::sort_modes(name).is_some() {
                    let mut values = Vec::with_capacity(args.len());
                    for arg in args {
                        let ExprKind::Name(actual) = &arg.kind else {
                            return Err(EvalError::tool(
                                "INTERNAL_BORROW",
                                "sort source is not named",
                            ));
                        };
                        self.charge()?;
                        values.push(env.borrow_owned(actual)?);
                    }
                    return sorting::invoke(
                        name,
                        &values,
                        expr.span,
                        &mut self.live_elements,
                    );
                }
                if matches!(name.as_str(), "view" | "view_mut") {
                    let ExprKind::Name(source) = &args[0].kind else {
                        return Err(EvalError::tool("INTERNAL_BORROW", "checked view source missing"));
                    };
                    self.charge()?;
                    let mut bounds = Vec::with_capacity(args.len() - 1);
                    for arg in args.iter().skip(1) {
                        let Value::U32(value) = self.expr(arg, env)? else {
                            return Err(EvalError::tool("INTERNAL_TYPE", "checked view bound is not u32"));
                        };
                        bounds.push(value);
                    }
                    return make_view(env.slot(source)?, bounds[0], bounds[1], *bounds.get(2).unwrap_or(&1), name == "view_mut", expr.span);
                }
                if let Some(kind) = crate::fsum::Kind::from_name(name) {
                    let mut operands = Vec::with_capacity(args.len());
                    for arg in args {
                        let ExprKind::Name(actual) = &arg.kind else {
                            return Err(EvalError::tool(
                                "INTERNAL_BORROW",
                                "checked reduction operand is not named",
                            ));
                        };
                        self.charge()?;
                        operands.push(env.borrow_owned(actual)?);
                    }
                    return floating::reduction(kind, &operands, expr.span);
                }
                if name == "len" {
                    let Some(Expr {
                        kind: ExprKind::Name(array),
                        ..
                    }) = args.first()
                    else {
                        return Err(EvalError::tool(
                            "INTERNAL_TYPE",
                            "len requires a checked array binding",
                        ));
                    };
                    self.charge()?;
                    return Ok(Value::U32(env.length(array)?));
                }
                if matches!(name.as_str(), "bytes_slice" | "utf8_slice" | "utf8_decode" | "utf8_encode") {
                    let ExprKind::Name(source) = &args[0].kind else {
                        return Err(EvalError::tool("INTERNAL_TYPE", "checked text source is not named"));
                    };
                    self.charge()?;
                    let mut bounds = Vec::new();
                    for arg in args.iter().skip(1) {
                        let Value::U32(bound) = self.expr(arg, env)? else {
                            return Err(EvalError::tool("INTERNAL_TYPE", "checked slice bound is not u32"));
                        };
                        bounds.push(bound as usize);
                    }
                    let source = env.slot(source)?;
                    let invalid_range = || EvalError::Trap { category: TrapCategory::IndexOutOfBounds, span: expr.span };
                    return match (name.as_str(), source) {
                        ("bytes_slice", Value::Bytes(value)) => {
                            let [start, end] = bounds.as_slice() else { return Err(EvalError::tool("INTERNAL_TYPE", "checked byte slice arity")); };
                            let slice = value.get(*start..*end).ok_or_else(invalid_range)?;
                            admit_live(&mut self.live_elements, slice.len())?;
                            Ok(Value::Bytes(Rc::new(slice.to_vec())))
                        }
                        ("utf8_encode", Value::Text(value)) => {
                            admit_live(&mut self.live_elements, value.len())?;
                            Ok(Value::Bytes(Rc::new(value.as_bytes().to_vec())))
                        }
                        ("utf8_decode", Value::Bytes(value)) => {
                            let decoded = std::str::from_utf8(value).ok();
                            admit_live(&mut self.live_elements, 2 + decoded.map_or(2, str::len))?;
                            let payload = decoded.map_or_else(
                                || Value::Variant(Rc::new(RefCell::new((0, None)))),
                                |text| Value::Text(Rc::new(text.to_owned())),
                            );
                            Ok(Value::Variant(Rc::new(RefCell::new((usize::from(decoded.is_none()), Some(payload))))))
                        }
                        ("utf8_slice", Value::Text(value)) => {
                            let [start, end] = bounds.as_slice() else { return Err(EvalError::tool("INTERNAL_TYPE", "checked UTF-8 slice arity")); };
                            if start > end || *end > value.len() { return Err(invalid_range()); }
                            let part = value.get(*start..*end);
                            admit_live(&mut self.live_elements, 2 + part.map_or(2, str::len))?;
                            let payload = part.map_or_else(
                                || Value::Variant(Rc::new(RefCell::new((0, None)))),
                                |text| Value::Text(Rc::new(text.to_owned())),
                            );
                            Ok(Value::Variant(Rc::new(RefCell::new((usize::from(part.is_none()), Some(payload))))))
                        }
                        _ => Err(EvalError::tool("INTERNAL_TYPE", "checked text intrinsic type mismatch")),
                    };
                }
                let mut values = Vec::with_capacity(args.len());
                let modes = self
                    .program
                    .program
                    .function(name)
                    .map(|function| function.params.as_slice())
                    .or_else(|| self.program.program.foreign_functions.iter().find(|function| function.name == *name).map(|function| function.params.as_slice()))
                    .map(|params| {
                        params
                            .iter()
                            .map(|param| param.mode)
                            .collect::<Vec<_>>()
                    });
                for (position, arg) in args.iter().enumerate() {
                    if matches!(
                        modes.as_ref().and_then(|modes| modes.get(position)),
                        Some(Some(Mode::Read | Mode::Mut))
                    ) {
                        let ExprKind::Name(actual) = &arg.kind else {
                            return Err(EvalError::tool(
                                "INTERNAL_BORROW",
                                "checked loan requires bare name",
                            ));
                        };
                        self.charge()?;
                        values.push(env.borrow_owned(actual)?);
                    } else {
                        values.push(self.expr(arg, env)?);
                    }
                }
                if let Some((numerator, denominator)) = crate::check::scale_call(name) {
                    let [value] = <[Value; 1]>::try_from(values).map_err(|_| EvalError::tool("INTERNAL_TYPE", "checked scale arity mismatch"))?;
                    return checked_scale(value, numerator, denominator, expr.span);
                }
                if let Some(degrees_to_radians) = crate::check::angle_call(name) {
                    let [value] = <[Value; 1]>::try_from(values).map_err(|_| EvalError::tool("INTERNAL_TYPE", "checked angle arity mismatch"))?;
                    return checked_angle(value, degrees_to_radians, expr.span);
                }
                match name.as_str() {
                    "wrapping_add" => match values.as_slice() {
                        [Value::U32(a), Value::U32(b)] => Ok(Value::U32(a.wrapping_add(*b))),
                        _ => Err(EvalError::tool(
                            "INTERNAL_TYPE",
                            "invalid checked wrapping_add",
                        )),
                    },
                    "wrapping_mul" => match values.as_slice() {
                        [Value::U32(a), Value::U32(b)] => Ok(Value::U32(a.wrapping_mul(*b))),
                        _ => Err(EvalError::tool(
                            "INTERNAL_TYPE",
                            "invalid checked wrapping_mul",
                        )),
                    },
                    float if floating::handles(float) => floating::intrinsic(float, &values, &mut self.live_elements, expr.span),
                    "filled_i32" => match values.as_slice() {
                        [Value::U32(length), Value::I32(value)] => {
                            filled_i32(&mut self.live_elements, *length, *value)
                        }
                        _ => Err(EvalError::tool(
                            "INTERNAL_TYPE",
                            "invalid checked filled_i32",
                        )),
                    },
                    "filled_u32" => match values.as_slice() {
                        [Value::U32(length), Value::U32(value)] => {
                            filled_u32(&mut self.live_elements, *length, *value)
                        }
                        _ => Err(EvalError::tool(
                            "INTERNAL_TYPE",
                            "invalid checked filled_u32",
                        )),
                    },
                    _ => {
                        if matches!(name.as_str(), "Result::Ok" | "Result::Err") {
                            admit_live(&mut self.live_elements, 2)?;
                            return Ok(Value::Variant(Rc::new(RefCell::new((usize::from(name == "Result::Err"), values.into_iter().next())))));
                        }
                        if let Some((nominal, arm)) = name.rsplit_once("::")
                            && let Some(declaration) = self.program.program.variants.iter().find(|variant| variant.name == nominal)
                            && let Some(tag) = declaration.arms.iter().position(|case| case.name == arm) {
                            admit_live(&mut self.live_elements, 2)?;
                            Ok(Value::Variant(Rc::new(RefCell::new((tag, values.into_iter().next())))))
                        } else {
                            let target = self.program.program.foreign_functions.iter().find(|function| function.name == *name).map_or(name.as_str(), |function| function.oracle.as_str());
                            self.call_user(target, values).map_err(|error| match error {
                                EvalError::Trap { category, .. }
                                    if crate::check::is_frame_helper(name) =>
                                {
                                    EvalError::Trap {
                                        category,
                                        span: expr.span,
                                    }
                                }
                                error => error,
                            })
                        }
                    },
                }
            }
            ExprKind::Unary {
                op: UnaryOp::Not,
                value,
                ..
            } => match self.expr(value, env)? {
                Value::Bool(value) => Ok(Value::Bool(!value)),
                _ => Err(EvalError::tool(
                    "INTERNAL_TYPE",
                    "invalid checked not operand",
                )),
            },
            ExprKind::Unary {
                op: UnaryOp::Neg,
                op_span,
                value,
            } => {
                if let ExprKind::Integer(raw) = &value.kind {
                    self.charge()?;
                    let digits = raw.strip_suffix("i32").unwrap_or(raw);
                    let magnitude = digits.parse::<u64>().map_err(|_| {
                        EvalError::tool("INTERNAL_LITERAL", "invalid checked negative literal")
                    })?;
                    let signed = i64::try_from(magnitude).map_err(|_| {
                        EvalError::tool("INTERNAL_LITERAL", "invalid checked negative literal")
                    })?;
                    let result = i32::try_from(-signed).map_err(|_| {
                        EvalError::tool("INTERNAL_LITERAL", "invalid checked negative literal")
                    })?;
                    return Ok(Value::I32(result));
                }
                match self.expr(value, env)? {
                    Value::F32(value) => Ok(Value::F32(value.negate())),
                    Value::F64(value) => Ok(Value::F64(value.negate())),
                    Value::I32(value) => {
                        value.checked_neg().map(Value::I32).ok_or(EvalError::Trap {
                            category: TrapCategory::IntegerOverflow,
                            span: *op_span,
                        })
                    }
                    _ => Err(EvalError::tool("INTERNAL_TYPE", "invalid checked negation")),
                }
            }
            ExprKind::Binary {
                op: BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem,
                ..
            } => self.arithmetic_chain(expr, env),
            ExprKind::Binary {
                left,
                op,
                op_span,
                right,
            } => {
                let left = self.expr(left, env)?;
                if *op == BinaryOp::And && left == Value::Bool(false) {
                    return Ok(Value::Bool(false));
                }
                if *op == BinaryOp::Or && left == Value::Bool(true) {
                    return Ok(Value::Bool(true));
                }
                let right = self.expr(right, env)?;
                binary(left, *op, right, *op_span)
            }
        }
    }

    fn arithmetic_chain(&mut self, expr: &Expr, env: &mut Env) -> Result<Value, EvalError> {
        let mut chain = Vec::new();
        let mut leaf = expr;
        while let ExprKind::Binary {
            op: op @ (BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem),
            left,
            right,
            op_span,
        } = &leaf.kind
        {
            if !chain.is_empty() {
                self.charge()?;
            }
            chain.push((*op, *op_span, right.as_ref()));
            leaf = left;
        }
        let mut value = self.expr(leaf, env)?;
        for (op, span, right) in chain.into_iter().rev() {
            let other = self.expr(right, env)?;
            value = binary(value, op, other, span)?;
        }
        Ok(value)
    }

    fn integer(&self, raw: &str, span: Span) -> Result<Value, EvalError> {
        let ty = self
            .types
            .get(&span)
            .ok_or_else(|| EvalError::tool("INTERNAL_TYPE", "missing checked expression type"))?;
        let digits = raw
            .strip_suffix("i32")
            .or_else(|| raw.strip_suffix("u32"))
            .unwrap_or(raw);
        match ty {
            Type::I32 => digits
                .parse::<i32>()
                .map(Value::I32)
                .map_err(|_| EvalError::tool("INTERNAL_LITERAL", "invalid checked i32 literal")),
            Type::U32 => digits
                .parse::<u32>()
                .map(Value::U32)
                .map_err(|_| EvalError::tool("INTERNAL_LITERAL", "invalid checked u32 literal")),
            _ => Err(EvalError::tool(
                "INTERNAL_TYPE",
                "invalid checked literal type",
            )),
        }
    }
}

fn named_array(expr: &Expr) -> Option<&str> {
    match &expr.kind {
        ExprKind::Name(name) => Some(name),
        ExprKind::Group(inner) => named_array(inner),
        _ => None,
    }
}

fn index_owned(array: &Value, index: u32, span: Span) -> Result<Value, EvalError> {
    let index = host_index(index, usize::MAX as u64)?;
    match array {
        Value::ArrayI32(values) => values.borrow().get(index).copied().map(Value::I32),
        Value::ArrayU32(values) => values.borrow().get(index).copied().map(Value::U32),
        Value::ArrayBool(values) => values.borrow().get(index).copied().map(Value::Bool),
        Value::Bytes(values) => values.get(index).copied().map(|byte| Value::U32(u32::from(byte))),
        Value::ViewI32(values, range) => values.borrow().get(view_index(*range, index as u32, span)?).copied().map(Value::I32),
        Value::ViewU32(values, range) => values.borrow().get(view_index(*range, index as u32, span)?).copied().map(Value::U32),
        _ => {
            return Err(EvalError::tool(
                "INTERNAL_TYPE",
                "indexing requires an array",
            ));
        }
    }
    .ok_or(EvalError::Trap {
        category: TrapCategory::IndexOutOfBounds,
        span,
    })
}

fn checked_length(length: u32) -> Result<usize, EvalError> {
    let length = usize::try_from(length)
        .map_err(|_| EvalError::tool("ARRAY_SIZE_LIMIT", "array length exceeds host size"))?;
    if length > MAX_ARRAY_LEN {
        return Err(EvalError::tool(
            "ARRAY_SIZE_LIMIT",
            "array length limit exceeded",
        ));
    }
    Ok(length)
}

fn filled_i32(live: &mut usize, length: u32, value: i32) -> Result<Value, EvalError> {
    let length = checked_length(length)?;
    admit_live(live, length)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| EvalError::tool("RESOURCE_LIMIT", "array allocation failed"))?;
    values.resize(length, value);
    Ok(Value::ArrayI32(Rc::new(RefCell::new(values))))
}

fn filled_u32(live: &mut usize, length: u32, value: u32) -> Result<Value, EvalError> {
    let length = checked_length(length)?;
    admit_live(live, length)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| EvalError::tool("RESOURCE_LIMIT", "array allocation failed"))?;
    values.resize(length, value);
    Ok(Value::ArrayU32(Rc::new(RefCell::new(values))))
}

fn binary(left: Value, op: BinaryOp, right: Value, span: Span) -> Result<Value, EvalError> {
    use BinaryOp as Op;
    use TrapCategory as Trap;
    let overflow = || EvalError::Trap {
        category: Trap::IntegerOverflow,
        span,
    };
    let zero = || EvalError::Trap {
        category: Trap::DivisionByZero,
        span,
    };
    match (left, right) {
        (Value::F32(a), Value::F32(b)) => crate::eval::floating::binary(a, op, b),
        (Value::F64(a), Value::F64(b)) => crate::eval::floating::binary64(a, op, b),
        (Value::I32(a), Value::I32(b)) => match op {
            Op::Add => a.checked_add(b).map(Value::I32).ok_or_else(overflow),
            Op::Sub => a.checked_sub(b).map(Value::I32).ok_or_else(overflow),
            Op::Mul => a.checked_mul(b).map(Value::I32).ok_or_else(overflow),
            Op::Div if b == 0 => Err(zero()),
            Op::Div if a == i32::MIN && b == -1 => Err(overflow()),
            Op::Div => Ok(Value::I32(a / b)),
            Op::Rem if b == 0 => Err(zero()),
            Op::Rem if a == i32::MIN && b == -1 => Ok(Value::I32(0)),
            Op::Rem => Ok(Value::I32(a % b)),
            Op::Eq => Ok(Value::Bool(a == b)),
            Op::Ne => Ok(Value::Bool(a != b)),
            Op::Lt => Ok(Value::Bool(a < b)),
            Op::Le => Ok(Value::Bool(a <= b)),
            Op::Gt => Ok(Value::Bool(a > b)),
            Op::Ge => Ok(Value::Bool(a >= b)),
            _ => Err(EvalError::tool(
                "INTERNAL_TYPE",
                "invalid checked i32 operation",
            )),
        },
        (Value::U32(a), Value::U32(b)) => match op {
            Op::Add => a.checked_add(b).map(Value::U32).ok_or_else(overflow),
            Op::Sub => a.checked_sub(b).map(Value::U32).ok_or_else(overflow),
            Op::Mul => a.checked_mul(b).map(Value::U32).ok_or_else(overflow),
            Op::Div if b == 0 => Err(zero()),
            Op::Div => Ok(Value::U32(a / b)),
            Op::Rem if b == 0 => Err(zero()),
            Op::Rem => Ok(Value::U32(a % b)),
            Op::Eq => Ok(Value::Bool(a == b)),
            Op::Ne => Ok(Value::Bool(a != b)),
            Op::Lt => Ok(Value::Bool(a < b)),
            Op::Le => Ok(Value::Bool(a <= b)),
            Op::Gt => Ok(Value::Bool(a > b)),
            Op::Ge => Ok(Value::Bool(a >= b)),
            _ => Err(EvalError::tool(
                "INTERNAL_TYPE",
                "invalid checked u32 operation",
            )),
        },
        (Value::Bool(a), Value::Bool(b)) => match op {
            Op::And => Ok(Value::Bool(a && b)),
            Op::Or => Ok(Value::Bool(a || b)),
            Op::Eq => Ok(Value::Bool(a == b)),
            Op::Ne => Ok(Value::Bool(a != b)),
            _ => Err(EvalError::tool(
                "INTERNAL_TYPE",
                "invalid checked Bool operation",
            )),
        },
        _ => Err(EvalError::tool(
            "INTERNAL_TYPE",
            "invalid checked binary operand types",
        )),
    }
}
