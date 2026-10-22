struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    depth: usize,
    closures: Vec<ClosureDecl>,
}

impl Parser {
    fn current(&self) -> &Token {
        // The lexer always appends EOF and parser methods never advance past it.
        &self.tokens[self.pos]
    }

    fn bump(&mut self) -> Token {
        let span = self.tokens[self.pos].span;
        let token = std::mem::replace(
            &mut self.tokens[self.pos],
            Token {
                kind: TokenKind::Eof,
                span,
            },
        );
        if !matches!(token.kind, TokenKind::Eof) {
            self.pos += 1;
        }
        token
    }

    fn is_symbol(&self, symbol: &str) -> bool {
        matches!(&self.current().kind, TokenKind::Symbol(actual) if *actual == symbol)
    }

    fn is_keyword(&self, keyword: &str) -> bool {
        matches!(&self.current().kind, TokenKind::Keyword(actual) if *actual == keyword)
    }

    fn take_symbol(&mut self, symbol: &str) -> Option<Span> {
        self.is_symbol(symbol).then(|| self.bump().span)
    }

    fn take_keyword(&mut self, keyword: &str) -> Option<Span> {
        self.is_keyword(keyword).then(|| self.bump().span)
    }

    fn expected(&self, expected: &str) -> FrontError {
        FrontError::source(
            "PARSE_EXPECTED_TOKEN",
            format!("expected {expected}"),
            self.current().span,
        )
    }

    fn expect_symbol(&mut self, symbol: &str) -> Result<Span, FrontError> {
        self.take_symbol(symbol)
            .ok_or_else(|| self.expected(symbol))
    }

    fn expect_keyword(&mut self, keyword: &str) -> Result<Span, FrontError> {
        self.take_keyword(keyword)
            .ok_or_else(|| self.expected(keyword))
    }

    fn expect_name(&mut self) -> Result<(String, Span), FrontError> {
        if let TokenKind::Keyword(word @ ("module" | "import" | "pub" | "struct" | "enum" | "match")) = self.current().kind {
            return Err(FrontError::source("UNSUPPORTED_FEATURE", format!("reserved keyword '{word}' cannot be an identifier"), self.current().span));
        }
        if !matches!(self.current().kind, TokenKind::Name(_)) {
            return Err(self.expected("identifier"));
        }
        let token = self.bump();
        let TokenKind::Name(name) = token.kind else {
            unreachable!("checked token kind")
        };
        Ok((name, token.span))
    }

    fn nested<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<T, FrontError>,
    ) -> Result<T, FrontError> {
        if self.depth >= MAX_NESTING {
            return Err(FrontError::tool(
                "SOURCE_NESTING_LIMIT",
                "parser nesting limit exceeded",
            ));
        }
        self.depth += 1;
        let result = f(self);
        self.depth -= 1;
        result
    }
}
