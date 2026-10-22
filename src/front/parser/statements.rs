impl Parser {
    fn block(&mut self) -> Result<Block, FrontError> {
        self.nested(|parser| {
            let start = parser.expect_symbol("{")?.start;
            let mut statements = Vec::new();
            while !parser.is_symbol("}") {
                if matches!(parser.current().kind, TokenKind::Eof) {
                    return Err(parser.expected("}"));
                }
                statements.push(parser.statement()?);
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
