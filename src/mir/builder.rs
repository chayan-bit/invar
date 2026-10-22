impl Builder<'_> {
    fn new_block(&mut self) -> BlockId {
        let id = self.blocks.len();
        self.blocks.push(BasicBlock {
            params: Vec::new(),
            instructions: Vec::new(),
            end: None,
        });
        id
    }

    fn param(&mut self, block: BlockId, ty: Type) -> ValueId {
        let id = self.next_value;
        self.next_value += 1;
        self.blocks[block].params.push((id, ty));
        id
    }

    fn emit(&mut self, kind: Operation, ty: Option<Type>, span: Span) -> Option<ValueId> {
        let output = ty.map(|ty| {
            let id = self.next_value;
            self.next_value += 1;
            (id, ty)
        });
        self.blocks[self.current].instructions.push(Instruction {
            output: output.clone(),
            kind,
            span,
        });
        output.map(|(id, _)| id)
    }

    fn value(&mut self, kind: Operation, ty: Type, span: Span) -> ValueId {
        self.emit(kind, Some(ty), span)
            .expect("typed operation produces a value")
    }

    fn tick(&mut self, span: Span) {
        self.emit(Operation::Tick, None, span);
    }

    fn end(&mut self, term: Terminator) {
        self.blocks[self.current].end = Some(term);
    }

    fn ty(&self, expr: &Expr) -> Result<Type, Error> {
        self.types
            .get(&expr.span)
            .cloned()
            .ok_or_else(|| invalid("checked expression type missing"))
    }

    fn cleanup(&self) -> Vec<String> {
        self.scopes
            .iter()
            .rev()
            .flat_map(|scope| scope.iter().rev().cloned())
            .collect()
    }

    fn block(&mut self, block: &Block) -> Result<(), Error> {
        self.scopes.push(Vec::new());
        for stmt in &block.statements {
            if self.blocks[self.current].end.is_none() {
                self.statement(stmt)?;
            }
        }
        let names = self
            .scopes
            .pop()
            .ok_or_else(|| invalid("missing lowering scope"))?;
        if self.blocks[self.current].end.is_none() {
            self.emit(
                Operation::Drop(names.into_iter().rev().collect()),
                None,
                block.span,
            );
        }
        Ok(())
    }

    fn statement(&mut self, stmt: &Stmt) -> Result<(), Error> {
        self.tick(stmt.span);
        match &stmt.kind {
            StmtKind::Binding {
                mutable,
                name,
                annotation,
                value,
            } => {
                let result = self.expr(value, false)?;
                let ty = annotation.clone().unwrap_or(self.ty(value)?);
                self.locals.insert(
                    name.clone(),
                    Local {
                        ty,
                        mutable: *mutable,
                        mode: None,
                    },
                );
                self.declaration_order.push(name.clone());
                self.emit(
                    Operation::Bind {
                        name: name.clone(),
                        value: result,
                    },
                    None,
                    stmt.span,
                );
                self.scopes
                    .last_mut()
                    .ok_or_else(|| invalid("missing lowering scope"))?
                    .push(name.clone());
            }
            StmtKind::Assign { place, value } => {
                let result = self.expr(value, false)?;
                match place {
                    Place::Name(name, _) => {
                        self.emit(
                            Operation::Store {
                                name: name.clone(),
                                value: result,
                            },
                            None,
                            stmt.span,
                        );
                    }
                    Place::Index { name, index, span } => {
                        let offset = self.expr(index, false)?;
                        self.emit(
                            Operation::StoreIndex {
                                name: name.clone(),
                                index: offset,
                                value: result,
                            },
                            None,
                            *span,
                        );
                    }
                }
            }
            StmtKind::Return(expr) => {
                let value = expr
                    .as_ref()
                    .map(|expr| self.expr(expr, true))
                    .transpose()?;
                self.end(Terminator::Return {
                    value,
                    cleanup: self.cleanup(),
                    span: stmt.span,
                });
            }
            StmtKind::Expr(expr) => {
                self.expr(expr, false)?;
            }
            StmtKind::If {
                condition,
                then_block,
                else_arm,
            } => {
                let test = self.expr(condition, false)?;
                let yes = self.new_block();
                let no = self.new_block();
                let join = self.new_block();
                self.end(Terminator::Branch {
                    condition: test,
                    yes,
                    no,
                    span: stmt.span,
                });
                self.current = yes;
                self.block(then_block)?;
                if self.blocks[self.current].end.is_none() {
                    self.end(Terminator::Goto {
                        target: join,
                        args: Vec::new(),
                        span: stmt.span,
                    });
                }
                self.current = no;
                match else_arm {
                    Some(ElseArm::Block(block)) => self.block(block)?,
                    Some(ElseArm::If(stmt)) => self.statement(stmt)?,
                    None => {}
                }
                if self.blocks[self.current].end.is_none() {
                    self.end(Terminator::Goto {
                        target: join,
                        args: Vec::new(),
                        span: stmt.span,
                    });
                }
                self.current = join;
            }
            StmtKind::While { condition, body } => {
                let head = self.new_block();
                let loop_body = self.new_block();
                let exit = self.new_block();
                self.end(Terminator::Goto {
                    target: head,
                    args: Vec::new(),
                    span: stmt.span,
                });
                self.current = head;
                let test = self.expr(condition, false)?;
                self.end(Terminator::Branch {
                    condition: test,
                    yes: loop_body,
                    no: exit,
                    span: stmt.span,
                });
                self.current = loop_body;
                self.block(body)?;
                if self.blocks[self.current].end.is_none() {
                    self.end(Terminator::Goto {
                        target: head,
                        args: Vec::new(),
                        span: stmt.span,
                    });
                }
                self.current = exit;
            }
            StmtKind::For {
                name,
                lower,
                upper,
                body,
            } => {
                let lo = self.expr(lower, false)?;
                let hi = self.expr(upper, false)?;
                let head = self.new_block();
                let loop_body = self.new_block();
                let exit = self.new_block();
                let induction = self.param(head, Type::U32);
                self.end(Terminator::Goto {
                    target: head,
                    args: vec![lo],
                    span: stmt.span,
                });
                self.current = head;
                let less = self.value(
                    Operation::Binary {
                        op: BinaryOp::Lt,
                        left: induction,
                        right: hi,
                        operator: stmt.span,
                    },
                    Type::Bool,
                    stmt.span,
                );
                self.end(Terminator::Branch {
                    condition: less,
                    yes: loop_body,
                    no: exit,
                    span: stmt.span,
                });
                self.current = loop_body;
                self.tick(stmt.span); // for-loop control step
                self.locals.insert(
                    name.clone(),
                    Local {
                        ty: Type::U32,
                        mutable: false,
                        mode: None,
                    },
                );
                self.declaration_order.push(name.clone());
                self.emit(
                    Operation::Bind {
                        name: name.clone(),
                        value: induction,
                    },
                    None,
                    stmt.span,
                );
                self.scopes.push(vec![name.clone()]);
                self.block(body)?;
                let names = self
                    .scopes
                    .pop()
                    .ok_or_else(|| invalid("missing for scope"))?;
                if self.blocks[self.current].end.is_none() {
                    self.emit(Operation::Drop(names), None, stmt.span);
                    let next = self.value(Operation::ForNext(induction), Type::U32, stmt.span);
                    self.end(Terminator::Goto {
                        target: head,
                        args: vec![next],
                        span: stmt.span,
                    });
                }
                self.current = exit;
            }
        }
        Ok(())
    }

    fn expr(&mut self, expr: &Expr, returning: bool) -> Result<ValueId, Error> {
        if let ExprKind::Group(inner) = &expr.kind {
            return self.expr(inner, false);
        }
        let ty = self.ty(expr)?;
        // A consuming bare owner name is an expression too. Its Tick must
        // precede the Move, so exhausted fuel cannot transfer or clean it up.
        self.tick(expr.span);
        Ok(match &expr.kind {
            ExprKind::Try(inner) => {
                let Type::Named(input_name) = self.ty(inner)? else { return Err(invalid("checked ? input type missing")); };
                let (ok, err) = crate::front::result_parts(&input_name).ok_or_else(|| invalid("checked ? input not Result"))?;
                let Type::Named(return_name) = &self.result else { return Err(invalid("checked ? return type missing")); };
                let return_name = return_name.clone();
                let input = self.expr(inner, false)?;
                let tag = self.value(Operation::VariantTag(input), Type::U32, expr.span);
                let zero = self.value(Operation::U32(0), Type::U32, expr.span);
                let condition = self.value(Operation::Binary { op: BinaryOp::Eq, left: tag, right: zero, operator: expr.span }, Type::Bool, expr.span);
                let success = self.new_block();
                let failure = self.new_block();
                self.end(Terminator::Branch { condition, yes: success, no: failure, span: expr.span });
                self.current = failure;
                let error = self.value(Operation::VariantTakePayload { value: input, tag: 1 }, err, expr.span);
                self.emit(Operation::DropValue(input), None, expr.span);
                self.emit(Operation::Drop(self.cleanup()), None, expr.span);
                let wrapped = self.value(Operation::Variant { name: return_name.clone(), tag: 1, payload: Some(error) }, Type::Named(return_name), expr.span);
                self.end(Terminator::Return { value: Some(wrapped), cleanup: Vec::new(), span: expr.span });
                self.current = success;
                let payload = self.value(Operation::VariantTakePayload { value: input, tag: 0 }, ok.clone(), expr.span);
                self.emit(Operation::DropValue(input), None, expr.span);
                payload
            }
            ExprKind::Float(raw) => {
                let invalid_literal = |_| invalid("invalid checked float literal");
                if self.types.get(&expr.span) == Some(&Type::F64) {
                    self.value(Operation::F64(crate::numeric_contract::F64Bits::from_decimal(raw).map_err(invalid_literal)?), Type::F64, expr.span)
                } else {
                    self.value(Operation::F32(crate::numeric_contract::F32Bits::from_decimal(raw).map_err(invalid_literal)?), Type::F32, expr.span)
                }
            }
            ExprKind::Integer(raw) => {
                let digits = raw
                    .strip_suffix("i32")
                    .or_else(|| raw.strip_suffix("u32"))
                    .unwrap_or(raw);
                match ty {
                    Type::I32 => self.value(
                        Operation::I32(
                            digits
                                .parse()
                                .map_err(|_| invalid("invalid checked i32 literal"))?,
                        ),
                        Type::I32,
                        expr.span,
                    ),
                    Type::U32 => self.value(
                        Operation::U32(
                            digits
                                .parse()
                                .map_err(|_| invalid("invalid checked u32 literal"))?,
                        ),
                        Type::U32,
                        expr.span,
                    ),
                    _ => return Err(invalid("integer has noninteger type")),
                }
            }
            ExprKind::Bool(value) => self.value(Operation::Bool(*value), Type::Bool, expr.span),
            ExprKind::Text(value) => self.value(Operation::Blob { bytes: value.as_bytes().to_vec(), text: true }, ty, expr.span),
            ExprKind::Bytes(value) => self.value(Operation::Blob { bytes: value.clone(), text: false }, ty, expr.span),
            ExprKind::Name(name) => {
                if matches!(ty, Type::ArrayI32 | Type::ArrayU32 | Type::ArrayF32 | Type::ArrayF64 | Type::ArrayBool | Type::Named(_)) {
                    if returning {
                        self.value(Operation::Move(name.clone()), ty, expr.span)
                    } else {
                        let reference = if matches!(&ty, Type::Named(name) if crate::front::view_parts(name).is_none()) {
                            Operation::RecordRef { name: name.clone(), loan: None }
                        } else {
                            Operation::ArrayRef { name: name.clone(), loan: None }
                        };
                        self.value(
                            reference,
                            ty,
                            expr.span,
                        )
                    }
                } else {
                    self.value(Operation::Read(name.clone()), ty, expr.span)
                }
            }
            ExprKind::Move(name) => self.value(Operation::Move(name.clone()), ty, expr.span),
            ExprKind::MoveField { base, field } => {
                let ExprKind::Name(name) = &base.kind else { return Err(invalid("checked move field base missing")); };
                let Type::Named(record) = self.ty(base)? else { return Err(invalid("checked move field type missing")); };
                let declaration = self.source.program.records.iter().find(|item| item.name == record)
                    .ok_or_else(|| invalid("checked move field record missing"))?;
                let index = declaration.fields.iter().position(|item| item.name == *field)
                    .ok_or_else(|| invalid("checked move field index missing"))?;
                self.tick(base.span);
                self.value(Operation::MoveField { name: name.clone(), index }, ty, expr.span)
            }
            ExprKind::Array(items) => {
                let items = items
                    .iter()
                    .map(|item| self.expr(item, false))
                    .collect::<Result<Vec<_>, _>>()?;
                self.value(Operation::Array(items), ty, expr.span)
            }
            ExprKind::Record { name, fields } => {
                let declaration = self.source.program.records.iter().find(|record| record.name == *name)
                    .ok_or_else(|| invalid("checked record declaration missing"))?;
                let order = declaration.fields.iter().map(|field| field.name.as_str()).collect::<Vec<_>>();
                let mut values = Vec::with_capacity(fields.len());
                for field in fields {
                    let index = order.iter().position(|name| *name == field.name)
                        .ok_or_else(|| invalid("checked record field missing"))?;
                    values.push((index, self.expr(&field.value, false)?));
                }
                self.value(Operation::Record { name: name.clone(), fields: values }, ty, expr.span)
            }
            ExprKind::Match { value, arms } => {
                let nominal = self.ty(value)?;
                let Type::Named(name) = nominal else { return Err(invalid("checked match type missing")); };
                let declaration = self.source.program.variants.iter().find(|variant| variant.name == name)
                    .ok_or_else(|| invalid("checked variant declaration missing"))?;
                let moving = matches!(value.kind, ExprKind::Move(_));
                let owner = if let ExprKind::Move(owner_name) = &value.kind {
                    self.tick(value.span);
                    self.value(Operation::RecordRef { name: owner_name.clone(), loan: None }, Type::Named(name.clone()), value.span)
                } else {
                    self.expr(value, false)?
                };
                let tag = self.value(Operation::VariantTag(owner), Type::U32, value.span);
                let join = self.new_block();
                let result = self.param(join, ty);
                for (position, arm) in arms.iter().enumerate() {
                    let case = arm.variant.rsplit_once("::").map(|(_, case)| case)
                        .ok_or_else(|| invalid("checked variant arm missing"))?;
                    let index = declaration.arms.iter().position(|item| item.name == case)
                        .ok_or_else(|| invalid("checked variant tag missing"))?;
                    let body = self.new_block();
                    let next = if position + 1 == arms.len() {
                        self.end(Terminator::Goto { target: body, args: Vec::new(), span: arm.span });
                        None
                    } else {
                        let expected = self.value(Operation::U32(index as u32), Type::U32, arm.span);
                        let condition = self.value(Operation::Binary { op: BinaryOp::Eq, left: tag, right: expected, operator: arm.span }, Type::Bool, arm.span);
                        let next = self.new_block();
                        self.end(Terminator::Branch { condition, yes: body, no: next, span: arm.span });
                        Some(next)
                    };
                    self.current = body;
                    if let Some(binding) = &arm.binding {
                        let payload_ty = declaration.arms[index].payload.clone().ok_or_else(|| invalid("checked payload type missing"))?;
                        let take = matches!(payload_ty, Type::Named(_));
                        let payload = self.value(if take { Operation::VariantTakePayload { value: owner, tag: index } } else { Operation::VariantPayload { value: owner, tag: index } }, payload_ty.clone(), arm.span);
                        self.locals.insert(binding.clone(), Local { ty: payload_ty, mutable: false, mode: None });
                        self.declaration_order.push(binding.clone());
                        self.emit(Operation::Bind { name: binding.clone(), value: payload }, None, arm.span);
                    }
                    let arm_value = self.expr(&arm.value, false)?;
                    if let Some(binding) = &arm.binding
                        && declaration.arms[index].payload.as_ref().is_some_and(|ty| matches!(ty, Type::Named(_))) {
                        self.emit(Operation::Drop(vec![binding.clone()]), None, arm.span);
                    }
                    if moving && let ExprKind::Move(owner_name) = &value.kind {
                        self.emit(Operation::Drop(vec![owner_name.clone()]), None, arm.span);
                    }
                    self.end(Terminator::Goto { target: join, args: vec![arm_value], span: arm.span });
                    if let Some(next) = next {
                        self.current = next;
                    }
                }
                self.current = join;
                result
            }
            ExprKind::Field { base, name } => {
                let Type::Named(record_name) = self.ty(base)? else {
                    return Err(invalid("checked field base type missing"));
                };
                let declaration = self.source.program.records.iter().find(|record| record.name == record_name)
                    .ok_or_else(|| invalid("checked field declaration missing"))?;
                let index = declaration.fields.iter().position(|field| field.name == *name)
                    .ok_or_else(|| invalid("checked field index missing"))?;
                let record = self.expr(base, false)?;
                self.value(Operation::Field { record, index }, ty, expr.span)
            }
            ExprKind::Index { base, index } => {
                let array = self.expr(base, false)?;
                let index = self.expr(index, false)?;
                self.value(Operation::Index { array, index }, ty, expr.span)
            }
            ExprKind::Unary { op, op_span, value } => {
                if *op == UnaryOp::Neg
                    && let ExprKind::Integer(raw) = &value.kind
                {
                    self.tick(value.span);
                    let digits = raw.strip_suffix("i32").unwrap_or(raw);
                    let magnitude: i64 = digits
                        .parse()
                        .map_err(|_| invalid("invalid checked negative literal"))?;
                    let signed = i32::try_from(-magnitude)
                        .map_err(|_| invalid("invalid checked negative literal"))?;
                    self.value(Operation::I32(signed), Type::I32, expr.span)
                } else {
                    let input = self.expr(value, false)?;
                    self.value(
                        Operation::Unary {
                            op: *op,
                            value: input,
                            operator: *op_span,
                        },
                        ty,
                        expr.span,
                    )
                }
            }
            ExprKind::Binary {
                op: BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem,
                ..
            } => return self.arithmetic_chain(expr),
            ExprKind::Binary {
                left,
                op,
                op_span,
                right,
            } if matches!(op, BinaryOp::And | BinaryOp::Or) => {
                let lhs = self.expr(left, false)?;
                let demand = self.new_block();
                let skip = self.new_block();
                let join = self.new_block();
                let result = self.param(join, Type::Bool);
                let (yes, no) = if *op == BinaryOp::And {
                    (demand, skip)
                } else {
                    (skip, demand)
                };
                self.end(Terminator::Branch {
                    condition: lhs,
                    yes,
                    no,
                    span: expr.span,
                });
                self.current = skip;
                self.end(Terminator::Goto {
                    target: join,
                    args: vec![lhs],
                    span: expr.span,
                });
                self.current = demand;
                let rhs = self.expr(right, false)?;
                self.end(Terminator::Goto {
                    target: join,
                    args: vec![rhs],
                    span: expr.span,
                });
                self.current = join;
                result
            }
            ExprKind::Binary {
                left,
                op,
                op_span,
                right,
            } => {
                let left = self.expr(left, false)?;
                let right = self.expr(right, false)?;
                self.value(
                    Operation::Binary {
                        op: *op,
                        left,
                        right,
                        operator: *op_span,
                    },
                    ty,
                    expr.span,
                )
            }
            ExprKind::Call { name, args } if name == "len" => {
                let arg = args
                    .first()
                    .ok_or_else(|| invalid("checked len argument missing"))?;
                self.tick(arg.span);
                let ExprKind::Name(array) = &arg.kind else {
                    return Err(invalid("checked len requires name"));
                };
                self.value(Operation::Length(array.clone()), Type::U32, expr.span)
            }
            ExprKind::Call { name, args } => {
                if matches!(name.as_str(), "Result::Ok" | "Result::Err") {
                    let Type::Named(nominal) = &ty else { return Err(invalid("checked Result type missing")); };
                    let payload = self.expr(&args[0], false)?;
                    return Ok(self.value(Operation::Variant { name: nominal.clone(), tag: usize::from(name == "Result::Err"), payload: Some(payload) }, ty, expr.span));
                }
                if let Some((nominal, case)) = name.rsplit_once("::")
                    && let Some(declaration) = self.source.program.variants.iter().find(|variant| variant.name == nominal)
                    && let Some(tag) = declaration.arms.iter().position(|arm| arm.name == case) {
                    let payload = args.first().map(|arg| self.expr(arg, false)).transpose()?;
                    return Ok(self.value(Operation::Variant { name: nominal.to_owned(), tag, payload }, ty, expr.span));
                }
                let function = self
                    .source
                    .program
                    .functions
                    .iter()
                    .find(|function| function.name == *name);
                let foreign = self.source.program.foreign_functions.iter().find(|function| function.name == *name);
                let mut values = Vec::with_capacity(args.len());
                for (position, arg) in args.iter().enumerate() {
                    if let Some(mode @ (Mode::Read | Mode::Mut)) = function
                        .and_then(|function| function.params.get(position))
                        .or_else(|| foreign.and_then(|function| function.params.get(position)))
                        .and_then(|param| param.mode)
                        .or_else(|| crate::check::sort_modes(name).and_then(|modes| modes.get(position).copied()))
                        .or_else(|| crate::spatial::mode(name, position))
                    {
                        let ExprKind::Name(actual) = &arg.kind else {
                            return Err(invalid("checked loan requires bare name"));
                        };
                        self.tick(arg.span);
                        let ty = self.ty(arg)?;
                        let reference = if matches!(&ty, Type::Named(name) if crate::front::view_parts(name).is_none()) {
                            Operation::RecordRef { name: actual.clone(), loan: Some(mode) }
                        } else {
                            Operation::ArrayRef { name: actual.clone(), loan: Some(mode) }
                        };
                        values.push(self.value(reference, ty, arg.span));
                    } else {
                        values.push(self.expr(arg, false)?);
                    }
                }
                self.value(
                    Operation::Call {
                        name: name.clone(),
                        args: values,
                    },
                    ty,
                    expr.span,
                )
            }
            ExprKind::Group(_) => unreachable!(),
        })
    }

    fn arithmetic_chain(&mut self, expr: &Expr) -> Result<ValueId, Error> {
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
                self.tick(leaf.span);
            }
            chain.push((leaf, *op, *op_span, right.as_ref()));
            leaf = left;
        }
        let mut value = self.expr(leaf, false)?;
        for (node, op, operator, right) in chain.into_iter().rev() {
            let other = self.expr(right, false)?;
            let ty = self.ty(node)?;
            value = self.value(
                Operation::Binary {
                    op,
                    left: value,
                    right: other,
                    operator,
                },
                ty,
                node.span,
            );
        }
        Ok(value)
    }
}
