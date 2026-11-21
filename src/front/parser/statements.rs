impl Parser {
    fn compound_operator(&mut self) -> Option<(BinaryOp, Span)> {
        let op = match self.current().kind {
            TokenKind::Symbol("+=") => BinaryOp::Add,
            TokenKind::Symbol("-=") => BinaryOp::Sub,
            TokenKind::Symbol("*=") => BinaryOp::Mul,
            TokenKind::Symbol("/=") => BinaryOp::Div,
            TokenKind::Symbol("%=") => BinaryOp::Rem,
            _ => return None,
        };
        Some((op, self.bump().span))
    }

    /// `place op= value;` is `place = place op (value);` (#254). A name or a
    /// name- or literal-indexed place is repeated as written; any other index
    /// is bound once to a hidden immutable `u32` local placed before the
    /// statement, so it evaluates exactly once.
    fn compound_assignment(
        &mut self,
        target: Expr,
        op: BinaryOp,
        op_span: Span,
    ) -> Result<Stmt, FrontError> {
        let span = target.span;
        let mut hoisted = None;
        let (place, read) = match target.kind {
            ExprKind::Name(name) => (
                Place::Name(name.clone(), span),
                Expr::new(ExprKind::Name(name), span)?,
            ),
            ExprKind::Index { base, index } if matches!(base.kind, ExprKind::Name(_)) => {
                let ExprKind::Name(name) = base.kind.clone() else {
                    unreachable!("matched name")
                };
                let index = if matches!(index.kind, ExprKind::Name(_) | ExprKind::Integer(_)) {
                    *index
                } else {
                    let at = index.span;
                    let hidden = format!("index@{}", at.start);
                    let binding = StmtKind::Binding {
                        mutable: false,
                        name: hidden.clone(),
                        annotation: Some(Type::U32),
                        value: *index,
                    };
                    hoisted = Some(Stmt::new(binding, at)?);
                    Expr::new(ExprKind::Name(hidden), at)?
                };
                let read = Expr::new(
                    ExprKind::Index {
                        base,
                        index: Box::new(index.clone()),
                    },
                    span,
                )?;
                (Place::Index { name, index, span }, read)
            }
            _ => {
                return Err(FrontError::source(
                    "PARSE_EXPECTED_TOKEN",
                    "expected assignment place",
                    span,
                ));
            }
        };
        let value = self.expression()?;
        let value = if matches!(value.kind, ExprKind::Binary { .. }) {
            let at = value.span;
            Expr::new(ExprKind::Group(Box::new(value)), at)?
        } else {
            value
        };
        let end = self.expect_symbol(";")?.end;
        self.hoisted = hoisted;
        let value_span = Span {
            start: span.start,
            end: value.span.end,
        };
        let value = Expr::new(
            ExprKind::Binary {
                left: Box::new(read),
                op,
                op_span,
                right: Box::new(value),
            },
            value_span,
        )?;
        Stmt::new(
            StmtKind::Assign { place, value },
            Span {
                start: span.start,
                end,
            },
        )
    }

    fn block(&mut self) -> Result<Block, FrontError> {
        self.nested(|parser| {
            let start = parser.expect_symbol("{")?.start;
            let mut statements = Vec::new();
            while !parser.is_symbol("}") {
                if matches!(parser.current().kind, TokenKind::Eof) {
                    return Err(parser.expected("}"));
                }
                let statement = parser.statement()?;
                statements.extend(parser.hoisted.take());
                statements.push(statement);
            }
            let end = parser.expect_symbol("}")?.end;
            Block::new(statements, Span { start, end })
        })
    }

    fn statement(&mut self) -> Result<Stmt, FrontError> {
        if self.is_keyword("let") || self.is_keyword("var") {
            let mutable = self.is_keyword("var");
            let start = self.bump().span.start;
            let (name, _) = self.expect_name()?;
            let annotation = if self.take_symbol(":").is_some() {
                Some(self.ty()?)
            } else {
                None
            };
            self.expect_symbol("=")?;
            let value = self.expression()?;
            let end = self.expect_symbol(";")?.end;
            return Stmt::new(
                StmtKind::Binding {
                    mutable,
                    name,
                    annotation,
                    value,
                },
                Span { start, end },
            );
        }
        if let Some(start) = self.take_keyword("return") {
            let value = if self.is_symbol(";") {
                None
            } else {
                Some(self.expression()?)
            };
            let end = self.expect_symbol(";")?.end;
            return Stmt::new(
                StmtKind::Return(value),
                Span {
                    start: start.start,
                    end,
                },
            );
        }
        if let Some(start) = self.take_keyword("if") {
            // `else if` chains are parsed iteratively and nested afterwards, so
            // the parser's own stack does not grow with the chain length.
            let condition = self.expression()?;
            let then_block = self.block()?;
            let mut chain = Vec::new();
            let mut tail = None;
            while self.take_keyword("else").is_some() {
                if let Some(link) = self.take_keyword("if") {
                    chain.push((link.start, self.expression()?, self.block()?));
                } else {
                    tail = Some(ElseArm::Block(self.block()?));
                    break;
                }
            }
            for (link, condition, then_block) in chain.into_iter().rev() {
                tail = Some(ElseArm::If(Box::new(if_statement(
                    link, condition, then_block, tail,
                )?)));
            }
            return if_statement(start.start, condition, then_block, tail);
        }
        if let Some(start) = self.take_keyword("while") {
            let condition = self.expression()?;
            let body = self.block()?;
            let span = Span {
                start: start.start,
                end: body.span.end,
            };
            return Stmt::new(StmtKind::While { condition, body }, span);
        }
        if let Some(start) = self.take_keyword("for") {
            let (name, _) = self.expect_name()?;
            self.expect_keyword("in")?;
            let lower = self.expression()?;
            self.expect_symbol("..")?;
            let upper = self.expression()?;
            let body = self.block()?;
            let span = Span {
                start: start.start,
                end: body.span.end,
            };
            return Stmt::new(
                StmtKind::For {
                    name,
                    lower,
                    upper,
                    body,
                },
                span,
            );
        }
        let starts_with_name = matches!(self.current().kind, TokenKind::Name(_));
        let expr = self.expression()?;
        if let Some((op, op_span)) = self.compound_operator() {
            return self.compound_assignment(expr, op, op_span);
        }
        if self.take_symbol("=").is_some() {
            let span = expr.span;
            if !starts_with_name {
                return Err(FrontError::source(
                    "PARSE_EXPECTED_TOKEN",
                    "expected assignment place",
                    span,
                ));
            }
            let place = match expr.kind {
                ExprKind::Name(name) => Place::Name(name, span),
                ExprKind::Index { base, index } => match base.kind {
                    ExprKind::Name(name) => Place::Index {
                        name,
                        index: *index,
                        span,
                    },
                    _ => {
                        return Err(FrontError::source(
                            "PARSE_EXPECTED_TOKEN",
                            "expected assignment place",
                            span,
                        ));
                    }
                },
                _ => {
                    return Err(FrontError::source(
                        "PARSE_EXPECTED_TOKEN",
                        "expected assignment place",
                        span,
                    ));
                }
            };
            let value = self.expression()?;
            let end = self.expect_symbol(";")?.end;
            Stmt::new(
                StmtKind::Assign { place, value },
                Span {
                    start: span.start,
                    end,
                },
            )
        } else {
            let end = self.expect_symbol(";")?.end;
            let span = Span {
                start: expr.span.start,
                end,
            };
            Stmt::new(StmtKind::Expr(expr), span)
        }
    }
}
