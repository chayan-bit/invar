impl Parser {
    fn ty(&mut self) -> Result<Type, FrontError> {
        if matches!(&self.current().kind, TokenKind::Name(name) if matches!(name.as_str(), "Vector3" | "Point" | "Transform"))
            && matches!(self.tokens.get(self.pos + 1).map(|token| &token.kind), Some(TokenKind::Symbol("<")))
        {
            let kind = self.expect_name()?.0;
            self.expect_symbol("<")?;
            let name = if kind == "Transform" {
                let from = self.expect_name()?.0;
                self.expect_symbol(",")?;
                let to = self.expect_name()?.0;
                format!("Transform<{from},{to}>")
            } else {
                let scalar = if self.take_keyword("i32").is_some() {
                    "i32".to_owned()
                } else if self.take_keyword("u32").is_some() {
                    "u32".to_owned()
                } else if matches!(self.current().kind, TokenKind::Name(_)) {
                    self.expect_name()?.0
                } else {
                    return Err(self.expected("coordinate scalar"));
                };
                self.expect_symbol(",")?;
                if kind == "Vector3" {
                    let unit = self.nested(|parser| parser.unit_type())?;
                    self.expect_symbol(",")?;
                    let frame = self.expect_name()?.0;
                    format!("Vector3<{scalar},{unit},{frame}>")
                } else {
                    let frame = self.expect_name()?.0;
                    format!("Point<{scalar},{frame}>")
                }
            };
            self.expect_symbol(">")?;
            return Ok(Type::Named(name));
        }
        if matches!(&self.current().kind, TokenKind::Name(name) if name == "Quantity")
            && matches!(self.tokens.get(self.pos + 1).map(|token| &token.kind), Some(TokenKind::Symbol("<")))
        {
            self.bump();
            self.expect_symbol("<")?;
            let scalar = if self.take_keyword("i32").is_some() {
                "i32".to_owned()
            } else if self.take_keyword("u32").is_some() {
                "u32".to_owned()
            } else if self.take_keyword("f32").is_some() {
                "f32".to_owned()
            } else if matches!(self.current().kind, TokenKind::Name(_)) {
                self.expect_name()?.0
            } else {
                return Err(self.expected("quantity scalar"));
            };
            self.expect_symbol(",")?;
            let unit = self.nested(|parser| parser.unit_type())?;
            self.expect_symbol(">")?;
            return Ok(Type::Named(format!("Quantity<{scalar},{unit}>")));
        }
        if matches!(&self.current().kind, TokenKind::Name(name) if matches!(name.as_str(), "Mul" | "Div" | "Pow"))
            && matches!(self.tokens.get(self.pos + 1).map(|token| &token.kind), Some(TokenKind::Symbol("<")))
        {
            return Ok(Type::Named(self.nested(|parser| parser.unit_type())?));
        }
        let kind = if self.take_keyword("View").is_some() {
            Some("View")
        } else if self.take_keyword("MutView").is_some() {
            Some("MutView")
        } else {
            None
        };
        if let Some(kind) = kind {
            self.expect_symbol("<")?;
            let element = if self.take_keyword("i32").is_some() {
                "i32"
            } else if self.take_keyword("u32").is_some() {
                "u32"
            } else {
                return Err(self.expected("i32 or u32 view element"));
            };
            self.expect_symbol(">")?;
            return Ok(Type::Named(format!("{kind}<{element}>")));
        }
        if matches!(&self.current().kind, TokenKind::Name(name) if name == "Result")
            && matches!(self.tokens.get(self.pos + 1).map(|token| &token.kind), Some(TokenKind::Symbol("<"))) {
            let start = self.bump().span;
            self.expect_symbol("<")?;
            let ok = self.nested(|parser| parser.ty())?;
            self.expect_symbol(",")?;
            let err = self.nested(|parser| parser.ty())?;
            let end = self.expect_symbol(">")?;
            let key = |ty: &Type| match ty {
                Type::I32 => Some("i32".to_owned()), Type::U32 => Some("u32".to_owned()), Type::F32 => Some("f32".to_owned()), Type::F64 => Some("f64".to_owned()), Type::Bool => Some("Bool".to_owned()),
                Type::Named(name) if crate::front::result_parts(name).is_some() || matches!(name.as_str(), "Bytes" | "String" | "Utf8Error" | "ForeignError") => Some(name.clone()),
                _ => None,
            };
            let (Some(ok), Some(err)) = (key(&ok), key(&err)) else {
                return Err(FrontError::source("UNSUPPORTED_FEATURE", "Result payloads must be scalar or bounded nested Result", Span { start: start.start, end: end.end }));
            };
            let name = format!("Result<{ok},{err}>");
            if crate::front::result_parts(&name).is_none() {
                return Err(FrontError::source("UNSUPPORTED_FEATURE", "Result nesting exceeds four levels", Span { start: start.start, end: end.end }));
            }
            return Ok(Type::Named(name));
        }
        if self.take_keyword("i32").is_some() {
            return Ok(Type::I32);
        }
        if self.take_keyword("u32").is_some() {
            return Ok(Type::U32);
        }
        if matches!(&self.current().kind, TokenKind::Name(name) if name == "f32") {
            self.bump();
            return Ok(Type::F32);
        }
        if matches!(&self.current().kind, TokenKind::Name(name) if name == "f64") {
            self.bump();
            return Ok(Type::F64);
        }
        if self.take_keyword("Bool").is_some() {
            return Ok(Type::Bool);
        }
        if self.take_keyword("Unit").is_some() {
            return Ok(Type::Unit);
        }
        if self.take_keyword("Array").is_some() {
            self.expect_symbol("<")?;
            let ty = if self.take_keyword("i32").is_some() {
                Type::ArrayI32
            } else if self.take_keyword("u32").is_some() {
                Type::ArrayU32
            } else if matches!(&self.current().kind, TokenKind::Name(name) if name == "f32") {
                self.bump();
                Type::ArrayF32
            } else if matches!(&self.current().kind, TokenKind::Name(name) if name == "f64") {
                self.bump();
                Type::ArrayF64
            } else if self.take_keyword("Bool").is_some() {
                Type::ArrayBool
            } else if let TokenKind::Name(_) = &self.current().kind {
                Type::Named(format!("Array<{}>", self.expect_name()?.0))
            } else {
                return Err(self.expected("integer array element type"));
            };
            self.expect_symbol(">")?;
            return Ok(ty);
        }
        if matches!(self.current().kind, TokenKind::Name(_)) {
            let (mut name, _) = self.expect_name()?;
            while self.take_symbol("::").is_some() {
                let (item, _) = self.expect_name()?;
                name.push_str("::");
                name.push_str(&item);
            }
            if self.take_symbol("<").is_some() {
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
            return Ok(Type::Named(name));
        }
        Err(self.expected("type"))
    }

    fn type_name(ty: &Type) -> String {
        match ty {
            Type::I32 => "i32".into(),
            Type::U32 => "u32".into(),
            Type::F32 => "f32".into(),
            Type::F64 => "f64".into(),
            Type::Bool => "Bool".into(),
            Type::Unit => "Unit".into(),
            Type::ArrayI32 => "Array<i32>".into(),
            Type::ArrayU32 => "Array<u32>".into(),
            Type::ArrayF32 => "Array<f32>".into(),
            Type::ArrayF64 => "Array<f64>".into(),
            Type::ArrayBool => "Array<Bool>".into(),
            Type::Named(name) => name.clone(),
        }
    }

    fn generic_record_initializer_follows(&self) -> bool {
        let mut depth = 0usize;
        for (index, token) in self.tokens.iter().enumerate().skip(self.pos).take(256) {
            match token.kind {
                TokenKind::Symbol("<") => {
                    depth += 1;
                    if depth > MAX_NESTING { return false; }
                }
                TokenKind::Symbol(">") => {
                    if depth == 0 { return false; }
                    depth -= 1;
                    if depth == 0 {
                        return matches!(self.tokens.get(index + 1).map(|next| &next.kind), Some(TokenKind::Symbol("{")))
                            && matches!(self.tokens.get(index + 2).map(|next| &next.kind), Some(TokenKind::Name(_)))
                            && matches!(self.tokens.get(index + 3).map(|next| &next.kind), Some(TokenKind::Symbol(":")));
                    }
                }
                TokenKind::Eof | TokenKind::Symbol(";") => return false,
                _ => {}
            }
        }
        false
    }

    fn unit_type(&mut self) -> Result<String, FrontError> {
        let (name, _) = self.expect_name()?;
        if !matches!(name.as_str(), "Mul" | "Div" | "Pow") {
            let mut unit = name;
            while self.is_symbol("/") || self.is_symbol("*") {
                let operation = self.bump();
                let rhs = self.expect_name()?.0;
                unit = format!("{}<{unit},{rhs}>", if matches!(operation.kind, TokenKind::Symbol("/")) { "Div" } else { "Mul" });
            }
            return Ok(unit);
        }
        self.expect_symbol("<")?;
        let left = self.nested(|parser| parser.unit_type())?;
        self.expect_symbol(",")?;
        let right = if name == "Pow" {
            let negative = self.take_symbol("-").is_some();
            let token = self.bump();
            let TokenKind::Integer(raw) = token.kind else {
                return Err(self.expected("integer unit exponent"));
            };
            let exponent = raw.parse::<i8>().map_err(|_| {
                FrontError::source("RESOURCE_LIMIT", "unit exponent exceeds 127", token.span)
            })?;
            format!("{}{}", if negative { "-" } else { "" }, exponent)
        } else {
            self.nested(|parser| parser.unit_type())?
        };
        self.expect_symbol(">")?;
        Ok(format!("{name}<{left},{right}>"))
    }
}
