impl Parser {
    fn expression(&mut self) -> Result<Expr, FrontError> {
        self.nested(|parser| parser.or_expr())
    }

    fn or_expr(&mut self) -> Result<Expr, FrontError> {
        let mut left = self.and_expr()?;
        while let Some(op_span) = self.take_keyword("or") {
            let right = self.and_expr()?;
            let span = Span {
                start: left.span.start,
                end: right.span.end,
            };
            left = Expr::new(
                ExprKind::Binary {
                    left: Box::new(left),
                    op: BinaryOp::Or,
                    op_span,
                    right: Box::new(right),
                },
                span,
            )?;
        }
        Ok(left)
    }

    fn and_expr(&mut self) -> Result<Expr, FrontError> {
        let mut left = self.compare_expr()?;
        while let Some(op_span) = self.take_keyword("and") {
            let right = self.compare_expr()?;
            let span = Span {
                start: left.span.start,
                end: right.span.end,
            };
            left = Expr::new(
                ExprKind::Binary {
                    left: Box::new(left),
                    op: BinaryOp::And,
                    op_span,
                    right: Box::new(right),
                },
                span,
            )?;
        }
        Ok(left)
    }

    fn compare_expr(&mut self) -> Result<Expr, FrontError> {
        let left = self.add_expr()?;
        let op = match &self.current().kind {
            TokenKind::Symbol("==") => BinaryOp::Eq,
            TokenKind::Symbol("!=") => BinaryOp::Ne,
            TokenKind::Symbol("<") => BinaryOp::Lt,
            TokenKind::Symbol("<=") => BinaryOp::Le,
            TokenKind::Symbol(">") => BinaryOp::Gt,
            TokenKind::Symbol(">=") => BinaryOp::Ge,
            _ => return Ok(left),
        };
        let op_span = self.bump().span;
        let right = self.add_expr()?;
        let span = Span {
            start: left.span.start,
            end: right.span.end,
        };
        Expr::new(
            ExprKind::Binary {
                left: Box::new(left),
                op,
                op_span,
                right: Box::new(right),
            },
            span,
        )
    }

    fn add_expr(&mut self) -> Result<Expr, FrontError> {
        let mut left = self.mul_expr()?;
        loop {
            let op = if self.is_symbol("+") {
                BinaryOp::Add
            } else if self.is_symbol("-") {
                BinaryOp::Sub
            } else {
                break;
            };
            let op_span = self.bump().span;
            let right = self.mul_expr()?;
            let span = Span {
                start: left.span.start,
                end: right.span.end,
            };
            left = Expr::new(
                ExprKind::Binary {
                    left: Box::new(left),
                    op,
                    op_span,
                    right: Box::new(right),
                },
                span,
            )?;
        }
        Ok(left)
    }

    fn mul_expr(&mut self) -> Result<Expr, FrontError> {
        let mut left = self.unary_expr()?;
        loop {
            let op = if self.is_symbol("*") {
                BinaryOp::Mul
            } else if self.is_symbol("/") {
                BinaryOp::Div
            } else if self.is_symbol("%") {
                BinaryOp::Rem
            } else {
                break;
            };
            let op_span = self.bump().span;
            let right = self.unary_expr()?;
            let span = Span {
                start: left.span.start,
                end: right.span.end,
            };
            left = Expr::new(
                ExprKind::Binary {
                    left: Box::new(left),
                    op,
                    op_span,
                    right: Box::new(right),
                },
                span,
            )?;
        }
        Ok(left)
    }

    fn unary_expr(&mut self) -> Result<Expr, FrontError> {
        let op = if self.is_keyword("not") {
            Some(UnaryOp::Not)
        } else if self.is_symbol("-") {
            Some(UnaryOp::Neg)
        } else {
            None
        };
        if let Some(op) = op {
            let op_span = self.bump().span;
            let value = self.nested(|parser| parser.unary_expr())?;
            let span = Span {
                start: op_span.start,
                end: value.span.end,
            };
            return Expr::new(
                ExprKind::Unary {
                    op,
                    op_span,
                    value: Box::new(value),
                },
                span,
            );
        }
        if let Some(start) = self.take_keyword("move") {
            let (name, span) = self.expect_name()?;
            if self.take_symbol(".").is_some() {
                let (field, end) = self.expect_name()?;
                let base = Expr::new(ExprKind::Name(name), span)?;
                return Expr::new(ExprKind::MoveField { base: Box::new(base), field }, Span { start: start.start, end: end.end });
            }
            return Expr::new(
                ExprKind::Move(name),
                Span {
                    start: start.start,
                    end: span.end,
                },
            );
        }
        self.postfix_expr()
    }

    fn postfix_expr(&mut self) -> Result<Expr, FrontError> {
        let mut base = self.primary()?;
        loop {
            if self.take_symbol("[").is_some() {
                let index = self.expression()?;
                let end = self.expect_symbol("]")?.end;
                let span = Span { start: base.span.start, end };
                base = Expr::new(ExprKind::Index { base: Box::new(base), index: Box::new(index) }, span)?;
            } else if self.take_symbol(".").is_some() {
                let (name, field_span) = self.expect_name()?;
                let span = Span { start: base.span.start, end: field_span.end };
                base = Expr::new(ExprKind::Field { base: Box::new(base), name }, span)?;
            } else if let Some(end) = self.take_symbol("?") {
                let span = Span { start: base.span.start, end: end.end };
                base = Expr::new(ExprKind::Try(Box::new(base)), span)?;
            } else {
                break;
            }
        }
        Ok(base)
    }

    fn primary(&mut self) -> Result<Expr, FrontError> {
        let token = self.bump();
        match token.kind {
            TokenKind::Integer(raw) => Expr::new(ExprKind::Integer(raw), token.span),
            TokenKind::StringLit(value) => Expr::new(ExprKind::Text(value), token.span),
            TokenKind::BytesLit(value) => Expr::new(ExprKind::Bytes(value), token.span),
            TokenKind::Decimal(raw) => {
                crate::numeric_contract::F64Bits::from_decimal(crate::numeric_contract::decimal_digits(&raw).0).map_err(|error| {
                    let code = match error {
                        crate::numeric_contract::DecimalError::DigitLimit => "RESOURCE_LIMIT",
                        crate::numeric_contract::DecimalError::Malformed => "LEX_INVALID_LITERAL",
                    };
                    FrontError::source(code, "invalid or oversized decimal literal", token.span)
                })?;
                Expr::new(ExprKind::Float(raw), token.span)
            },
            TokenKind::Keyword("true") => Expr::new(ExprKind::Bool(true), token.span),
            TokenKind::Keyword("false") => Expr::new(ExprKind::Bool(false), token.span),
            TokenKind::Keyword("closure") => {
                self.expect_symbol("[")?;
                let captures = self.comma_list("]", |parser| {
                    let mode = if parser.take_keyword("copy").is_some() {
                        None
                    } else if parser.take_keyword("read").is_some() {
                        Some(Mode::Read)
                    } else if parser.take_keyword("take").is_some() {
                        Some(Mode::Take)
                    } else {
                        return Err(parser.expected("copy, read, or take capture"));
                    };
                    let (name, start) = parser.expect_name()?;
                    parser.expect_symbol(":")?;
                    let ty = parser.ty()?;
                    Ok(ClosureCapture { name, ty, mode, span: Span { start: start.start, end: parser.tokens[parser.pos - 1].span.end } })
                })?;
                self.expect_symbol("]")?;
                self.expect_symbol("(")?;
                let params = self.comma_list(")", |parser| parser.param())?;
                self.expect_symbol(")")?;
                self.expect_symbol("->")?;
                let result = self.ty()?;
                let body = self.block()?;
                let span = Span { start: token.span.start, end: body.span.end };
                let name = format!("$closure@{}", token.span.start);
                self.closures.push(ClosureDecl { name: name.clone(), captures, params, result, body, span });
                Expr::new(ExprKind::Call { name, args: Vec::new() }, span)
            }
            TokenKind::Keyword("match") => {
                let value = self.expression()?;
                self.expect_symbol("{")?;
                let arms = self.comma_list("}", |parser| {
                    let (first, start) = parser.expect_name()?;
                    let mut variant = first;
                    while parser.take_symbol("::").is_some() {
                        let (segment, _) = parser.expect_name()?;
                        variant.push_str("::");
                        variant.push_str(&segment);
                    }
                    let binding = if parser.take_symbol("(").is_some() {
                        let (name, _) = parser.expect_name()?;
                        parser.expect_symbol(")")?;
                        Some(name)
                    } else {
                        None
                    };
                    parser.expect_symbol("=>")?;
                    let arm_value = parser.expression()?;
                    Ok(MatchArm { variant, binding, span: Span { start: start.start, end: arm_value.span.end }, value: arm_value })
                })?;
                let end = self.expect_symbol("}")?.end;
                Expr::new(ExprKind::Match { value: Box::new(value), arms }, Span { start: token.span.start, end })
            }
            TokenKind::Name(name) => {
                let mut name = name;
                while self.is_symbol("::") && !matches!(self.tokens.get(self.pos + 1).map(|token| &token.kind), Some(TokenKind::Symbol("<"))) {
                    self.bump();
                    let (item, _) = self.expect_name()?;
                    name.push_str("::");
                    name.push_str(&item);
                }
                if self.is_symbol("<") && self.generic_record_initializer_follows() {
                    self.bump();
                    let args = self.comma_list(">", |parser| {
                        if matches!(parser.current().kind, TokenKind::Integer(_)) {
                            let token = parser.bump();
                            let TokenKind::Integer(value) = token.kind else { unreachable!() };
                            Ok(value)
                        } else {
                            Ok(Self::type_name(&parser.nested(|parser| parser.ty())?))
                        }
                    })?;
                    self.expect_symbol(">")?;
                    name.push('<');
                    name.push_str(&args.join(","));
                    name.push('>');
                }
                if self.take_symbol("::").is_some() {
                    self.expect_symbol("<")?;
                    let types = self.comma_list(">", |parser| parser.ty())?;
                    self.expect_symbol(">")?;
                    if types.is_empty() {
                        return Err(self.expected("generic type argument"));
                    }
                    name.push_str("$<");
                    for (index, ty) in types.iter().enumerate() {
                        if index != 0 { name.push('|'); }
                        name.push_str(&match ty {
                            Type::I32 => "i32".to_owned(),
                            Type::U32 => "u32".to_owned(),
                            Type::F32 => "f32".to_owned(),
                            Type::F64 => "f64".to_owned(),
                            Type::Bool => "Bool".to_owned(),
                            Type::Unit => "Unit".to_owned(),
                            Type::ArrayI32 => "Array<i32>".to_owned(),
                            Type::ArrayU32 => "Array<u32>".to_owned(),
                            Type::ArrayF32 => "Array<f32>".to_owned(),
                            Type::ArrayF64 => "Array<f64>".to_owned(),
                            Type::ArrayBool => "Array<Bool>".to_owned(),
                            Type::Named(name) => name.clone(),
                        });
                    }
                    name.push('>');
                }
                // A control-flow body also begins with `{`; only a field label
                // followed by `:` starts a record initializer.
                if self.is_symbol("{")
                    && matches!(self.tokens.get(self.pos + 1).map(|token| &token.kind), Some(TokenKind::Name(_)))
                    && matches!(self.tokens.get(self.pos + 2).map(|token| &token.kind), Some(TokenKind::Symbol(":")))
                {
                    self.bump();
                    let fields = self.comma_list("}", |parser| {
                        let (name, name_span) = parser.expect_name()?;
                        parser.expect_symbol(":")?;
                        let value = parser.expression()?;
                        let span = Span { start: name_span.start, end: value.span.end };
                        Ok(FieldInit { name, value, span })
                    })?;
                    let end = self.expect_symbol("}")?.end;
                    Expr::new(ExprKind::Record { name, fields }, Span { start: token.span.start, end })
                } else if self.take_symbol("(").is_some() {
                    let args = self.comma_list(")", |parser| parser.expression())?;
                    let end = self.expect_symbol(")")?.end;
                    Expr::new(
                        ExprKind::Call { name, args },
                        Span {
                            start: token.span.start,
                            end,
                        },
                    )
                } else {
                    Expr::new(ExprKind::Name(name), token.span)
                }
            }
            TokenKind::Symbol("(") => {
                let expr = self.expression()?;
                let end = self.expect_symbol(")")?.end;
                Expr::new(
                    ExprKind::Group(Box::new(expr)),
                    Span {
                        start: token.span.start,
                        end,
                    },
                )
            }
            TokenKind::Symbol("[") => {
                let values = self.comma_list("]", |parser| parser.expression())?;
                let end = self.expect_symbol("]")?.end;
                Expr::new(
                    ExprKind::Array(values),
                    Span {
                        start: token.span.start,
                        end,
                    },
                )
            }
            _ => Err(FrontError::source(
                "PARSE_EXPECTED_TOKEN",
                "expected expression",
                token.span,
            )),
        }
    }

    fn comma_list<T>(
        &mut self,
        closing: &str,
        mut parse: impl FnMut(&mut Self) -> Result<T, FrontError>,
    ) -> Result<Vec<T>, FrontError> {
        let mut items = Vec::new();
        if self.is_symbol(closing) {
            return Ok(items);
        }
        loop {
            items.push(parse(self)?);
            if self.take_symbol(",").is_none() {
                break;
            }
            if self.is_symbol(closing) {
                break;
            }
        }
        Ok(items)
    }
}
