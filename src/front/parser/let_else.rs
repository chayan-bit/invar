// `let V::P(x) = s else { ... };` (#260). The parser emits existing statements:
//
//     let else@N: Bool = match s { V::P(else@N@0) => true };
//     if not else@N { ... }
//     let x = match s { V::P(else@N@x0) => else@N@x0 };
//
// `check::let_else` completes both matches once the variant's arms are known: the first
// gains `=> false` for every other arm, the second a never-evaluated typed zero and the
// payload type as `x`'s annotation. The else block must end in a diverging statement.

impl Parser {
    /// After `let`, when the next tokens are a qualified variant path.
    fn let_else(&mut self, start: usize) -> Result<Stmt, FrontError> {
        let (mut variant, path) = self.expect_name()?;
        let mut pattern = path;
        while self.take_symbol("::").is_some() {
            let (segment, at) = self.expect_name()?;
            variant.push_str("::");
            variant.push_str(&segment);
            pattern.end = at.end;
        }
        let path_end = pattern.end;
        let binding = if self.take_symbol("(").is_some() {
            let name = self.expect_name()?;
            pattern.end = self.expect_symbol(")")?.end;
            Some(name)
        } else {
            None
        };
        self.expect_symbol("=")?;
        let value = self.expression()?;
        let owner = match &value.kind {
            ExprKind::Name(name) | ExprKind::Move(name) => name.clone(),
            _ => {
                return Err(FrontError::source(
                    "PARSE_EXPECTED_TOKEN",
                    "expected a named variant owner or move of one",
                    value.span,
                ));
            }
        };
        let else_span = self.expect_keyword("else")?;
        let else_block = self.block()?;
        let end = self.expect_symbol(";")?.end;
        if !else_block.statements.last().is_some_and(diverging) {
            return Err(FrontError::source(
                "PARSE_LET_ELSE_FALLTHROUGH",
                "a let-else block must end in return, break, continue, assert(false, ...) or an if whose arms all do",
                else_block.span,
            ));
        }
        // Synthesized nodes take spans no parsed expression has, one per type.
        let hidden = format!("else@{start}");
        let name = |text: &str, span: Span| Expr::new(ExprKind::Name(text.to_owned()), span);
        let check_arm = MatchArm {
            variant: variant.clone(),
            binding: binding.as_ref().map(|_| format!("{hidden}@0")),
            guard: None,
            value: Expr::new(ExprKind::Bool(true), Span { start: pattern.start, end: path_end })?,
            span: pattern,
        };
        let check = Expr::new(
            ExprKind::Match { value: Box::new(name(&owner, value.span)?), arms: vec![check_arm] },
            Span { start, end: else_span.end },
        )?;
        let flag = StmtKind::Binding { mutable: false, name: hidden.clone(), annotation: Some(Type::Bool), value: check };
        self.hoisted.push(Stmt::new(flag, Span { start, end })?);
        let condition = Expr::new(
            ExprKind::Unary {
                op: UnaryOp::Not,
                op_span: else_span,
                value: Box::new(name(&hidden, Span { start: else_span.start, end: else_span.end - 1 })?),
            },
            else_span,
        )?;
        let guard = if_statement(else_span.start, condition, else_block, None)?;
        let Some((target, at)) = binding else {
            return Ok(guard);
        };
        self.hoisted.push(guard);
        let payload = format!("{hidden}@x0");
        let extract_arm = MatchArm {
            variant,
            binding: Some(payload.clone()),
            guard: None,
            value: name(&payload, Span { start: at.start, end: pattern.end })?,
            span: pattern,
        };
        let extract = Expr::new(ExprKind::Match { value: Box::new(value), arms: vec![extract_arm] }, at)?;
        Stmt::new(StmtKind::Binding { mutable: false, name: target, annotation: None, value: extract }, at)
    }
}

/// Whether a statement ends its path: `return`, `break`, `continue`, `assert(false, ...)`
/// with a literal `false`, or an `if` with an `else` whose blocks all end in one.
fn diverging(stmt: &Stmt) -> bool {
    let ends = |block: &Block| block.statements.last().is_some_and(diverging);
    match &stmt.kind {
        StmtKind::Return(_) | StmtKind::Break(_) | StmtKind::Continue(_) => true,
        StmtKind::If { then_block, else_arm: Some(arm), .. } => {
            ends(then_block)
                && match arm {
                    ElseArm::Block(block) => ends(block),
                    ElseArm::If(stmt) => diverging(stmt),
                }
        }
        StmtKind::Expr(Expr { kind: ExprKind::Call { name, args }, .. }) if name == "assert" => {
            let mut condition = args.first().map(|arg| &arg.kind);
            while let Some(ExprKind::Group(inner)) = condition {
                condition = Some(&inner.kind);
            }
            matches!(condition, Some(ExprKind::Bool(false)))
        }
        _ => false,
    }
}
