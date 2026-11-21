// Pairs of 32-bit scalars (#262): the type `(A, B)`, the literal `(a, b)`,
// positional fields `p.0` and `p.1`, and `let (a, b) = e;`, which binds a
// hidden pair and copies each field out, in order.

/// Whether `ty` may be a pair element: a scalar that fits 32 bits.
pub(crate) fn pair_element(ty: &Type) -> bool {
    matches!(ty, Type::I32 | Type::U32 | Type::F32 | Type::Bool)
}

pub(crate) fn tuple_shape(span: Span) -> FrontError {
    FrontError::source(
        "TUPLE_SHAPE",
        "a tuple is a pair of i32, u32, f32 or Bool values",
        span,
    )
}

impl Parser {
    /// `(A, B)` after its `(`.
    fn pair_type(&mut self, start: Span) -> Result<Type, FrontError> {
        let types = self.comma_list(")", |parser| parser.nested(|parser| parser.ty()))?;
        let end = self.expect_symbol(")")?;
        let span = Span { start: start.start, end: end.end };
        match <[Type; 2]>::try_from(types) {
            Ok([first, second]) if pair_element(&first) && pair_element(&second) => {
                Ok(Type::Pair(Box::new((first, second))))
            }
            _ => Err(tuple_shape(span)),
        }
    }

    /// `(first, second)` once `first` and the `,` after it are reached.
    fn pair_literal(&mut self, open: Span, first: Expr) -> Result<Expr, FrontError> {
        self.expect_symbol(",")?;
        let second = self.expression()?;
        if self.take_symbol(",").is_some() && !self.is_symbol(")") {
            return Err(tuple_shape(Span { start: open.start, end: self.current().span.end }));
        }
        let span = Span { start: open.start, end: self.expect_symbol(")")?.end };
        Expr::new(ExprKind::Pair(Box::new(first), Box::new(second)), span)
    }

    /// A field name after `.`: an identifier, or `0` or `1` spelled as one digit.
    fn field_name(&mut self) -> Result<(String, Span), FrontError> {
        let token = self.current();
        match &token.kind {
            TokenKind::Integer(digit) if token.span.end - token.span.start == 1 => {
                let field = (digit.clone(), token.span);
                self.bump();
                Ok(field)
            }
            _ => self.expect_name(),
        }
    }

    /// `let (a, b) = e;` after `let` or `var`: `let pair@N = e;` then
    /// `let a = pair@N.0;` and `let b = pair@N.1;`.
    fn pair_binding(&mut self, mutable: bool, start: usize) -> Result<Stmt, FrontError> {
        let open = self.expect_symbol("(")?;
        let names = self.comma_list(")", |parser| parser.expect_name())?;
        let close = self.expect_symbol(")")?;
        let Ok([first, second]) = <[(String, Span); 2]>::try_from(names) else {
            return Err(tuple_shape(Span { start: open.start, end: close.end }));
        };
        self.expect_symbol("=")?;
        let value = self.expression()?;
        let end = self.expect_symbol(";")?.end;
        let hidden = format!("pair@{start}");
        let pair = StmtKind::Binding { mutable: false, name: hidden.clone(), annotation: None, value };
        self.hoisted.push(Stmt::new(pair, Span { start, end })?);
        let mut fields = Vec::new();
        for (index, (name, at)) in [first, second].into_iter().enumerate() {
            // Synthesized nodes take spans no parsed expression has: the
            // binding name, and from `let` to the end of that name.
            let base = Expr::new(ExprKind::Name(hidden.clone()), at)?;
            let field = ExprKind::Field { base: Box::new(base), name: index.to_string() };
            let value = Expr::new(field, Span { start, end: at.end })?;
            fields.push(Stmt::new(StmtKind::Binding { mutable, name, annotation: None, value }, at)?);
        }
        let last = fields.pop().expect("two fields");
        self.hoisted.extend(fields);
        Ok(last)
    }
}
