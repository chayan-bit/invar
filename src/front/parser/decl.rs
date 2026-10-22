impl Parser {
    fn program(&mut self) -> Result<Program, FrontError> {
        let mut functions = Vec::new();
        let mut foreign_functions = Vec::new();
        let mut modules = Vec::new();
        let mut frames = Vec::new();
        let mut imports = Vec::new();
        let mut records = Vec::new();
        let mut variants = Vec::new();
        let mut system_items = Vec::new();
        while !matches!(self.current().kind, TokenKind::Eof) {
            if let Some(start) = self.take_keyword("module") {
                let (mut name, _) = self.expect_name()?;
                if self.take_symbol("::").is_some() {
                    name.push_str("::");
                    name.push_str(&self.expect_name()?.0);
                }
                self.expect_symbol("{")?;
                while !self.is_symbol("}") {
                    if matches!(self.current().kind, TokenKind::Eof) {
                        return Err(self.expected("}"));
                    }
                    if self.is_component_import() {
                        system_items.push(self.component_import(Some(&name))?);
                    } else if self.is_keyword("import") {
                        imports.push(self.import(Some(&name))?);
                    } else if self.is_foreign_fn() {
                        return Err(FrontError::source("UNSUPPORTED_FEATURE", "foreign functions must be top-level", self.current().span));
                    } else if self.is_system_item() {
                        system_items.push(self.system_item(Some(&name))?);
                    } else {
                        let (fold, public) = self.item_prefix()?;
                        if self.is_keyword("struct") {
                            records.push(self.record(Some(&name), public)?);
                        } else if self.is_keyword("enum") {
                            variants.push(self.variant(Some(&name), public)?);
                        } else {
                            functions.push(Function { fold, ..self.function(Some(&name), public)? });
                        }
                    }
                }
                let end = self.expect_symbol("}")?.end;
                modules.push(ModuleDecl {
                    name,
                    span: Span { start: start.start, end },
                });
            } else if self.is_component_import() {
                system_items.push(self.component_import(None)?);
            } else if self.is_keyword("import") {
                imports.push(self.import(None)?);
            } else if self.is_keyword("frame") {
                frames.push(self.frame()?);
            } else if self.is_foreign_fn() {
                foreign_functions.push(self.foreign_function()?);
            } else if self.is_system_item() {
                system_items.push(self.system_item(None)?);
            } else {
                let (fold, public) = self.item_prefix()?;
                if self.is_keyword("struct") {
                    records.push(self.record(None, public)?);
                } else if self.is_keyword("enum") {
                    variants.push(self.variant(None, public)?);
                } else {
                    functions.push(Function { fold, ..self.function(None, public)? });
                }
            }
        }
        Ok(Program { functions, foreign_functions, closures: std::mem::take(&mut self.closures), frames, modules, imports, records, variants, system_items })
    }

    fn is_component_import(&self) -> bool {
        self.is_keyword("import")
            && matches!(self.tokens.get(self.pos + 1).map(|token| &token.kind), Some(TokenKind::Keyword("component")))
    }

    fn component_import(&mut self, module: Option<&str>) -> Result<SystemItem, FrontError> {
        let start = self.expect_keyword("import")?.start;
        self.expect_keyword("component")?;
        let path = self.system_string()?;
        let robot = if self.take_keyword("for").is_some() {
            self.expect_word("robot")?;
            Some(self.system_path()?)
        } else {
            None
        };
        let end = self.expect_symbol(";")?.end;
        Ok(SystemItem {
            module: module.map(str::to_owned),
            kind: SystemItemKind::ComponentImport { path, robot },
            span: Span { start, end },
        })
    }

    fn foreign_function(&mut self) -> Result<crate::front::ForeignFunction, FrontError> {
        let start = self.expect_keyword("extern")?.start;
        let (abi, abi_span) = self.expect_name()?;
        if abi != "c" {
            return Err(FrontError::source("UNSUPPORTED_FEATURE", "only the c foreign ABI is supported", abi_span));
        }
        self.expect_keyword("fn")?;
        let (name, _) = self.expect_name()?;
        self.expect_symbol("(")?;
        let params = self.comma_list(")", |parser| parser.param())?;
        self.expect_symbol(")")?;
        self.expect_symbol("->")?;
        let result = self.ty()?;
        let (oracle_keyword, oracle_span) = self.expect_name()?;
        if oracle_keyword != "oracle" {
            return Err(FrontError::source("UNSUPPORTED_FEATURE", "foreign function requires an oracle binding", oracle_span));
        }
        let (oracle, _) = self.expect_name()?;
        let end = self.expect_symbol(";")?.end;
        Ok(crate::front::ForeignFunction { name, params, result, oracle, span: Span { start, end } })
    }

    fn frame(&mut self) -> Result<FrameDecl, FrontError> {
        let start = self.expect_keyword("frame")?.start;
        let (name, _) = self.expect_name()?;
        let parent = if self.take_symbol(":").is_some() {
            Some(self.expect_name()?.0)
        } else {
            None
        };
        let end = self.expect_symbol(";")?.end;
        Ok(FrameDecl { name, parent, span: Span { start, end } })
    }

    fn record(&mut self, module: Option<&str>, public: bool) -> Result<RecordDecl, FrontError> {
        let start = self.expect_keyword("struct")?.start;
        let (name, _) = self.expect_name()?;
        let generic_params = if self.take_symbol("<").is_some() {
            let params = self.comma_list(">", |parser| {
                let count = parser.take_keyword("const").is_some();
                let (name, _) = parser.expect_name()?;
                parser.expect_symbol(":")?;
                let (kind, span) = if count && parser.take_keyword("u32").is_some() {
                    (crate::front::RecordGenericKind::Count, parser.tokens[parser.pos - 1].span)
                } else if !count && parser.take_keyword("Unit").is_some() {
                    (crate::front::RecordGenericKind::Unit, parser.tokens[parser.pos - 1].span)
                } else {
                    let (kind, span) = parser.expect_name()?;
                    let kind = match (count, kind.as_str()) {
                        (false, "Value") => crate::front::RecordGenericKind::Value,
                        (false, "Frame") => crate::front::RecordGenericKind::Frame,
                        (false, "Unit") => crate::front::RecordGenericKind::Unit,
                        _ => return Err(FrontError::source("UNSUPPORTED_FEATURE", "record parameter kind must be Value, Frame, Unit, or const u32", span)),
                    };
                    (kind, span)
                };
                let _ = span;
                Ok(crate::front::RecordGenericParam { name, kind })
            })?;
            self.expect_symbol(">")?;
            if params.len() > 8 || params.iter().enumerate().any(|(index, param)| params[..index].iter().any(|earlier: &crate::front::RecordGenericParam| earlier.name == param.name)) {
                return Err(FrontError::source("RECORD_LIMIT", "record parameters must be distinct and at most eight", self.tokens[self.pos - 1].span));
            }
            params
        } else {
            Vec::new()
        };
        self.expect_symbol("{")?;
        let fields = self.comma_list("}", |parser| {
            let (name, span) = parser.expect_name()?;
            parser.expect_symbol(":")?;
            let ty = parser.ty()?;
            Ok(RecordField {
                name,
                ty,
                span: Span { start: span.start, end: parser.tokens[parser.pos - 1].span.end },
            })
        })?;
        let end = self.expect_symbol("}")?.end;
        Ok(RecordDecl {
            name: module.map_or_else(|| name.clone(), |module| format!("{module}::{name}")),
            module: module.map(str::to_owned),
            public,
            generic_params,
            fields,
            span: Span { start, end },
        })
    }

    fn variant(&mut self, module: Option<&str>, public: bool) -> Result<VariantDecl, FrontError> {
        let start = self.expect_keyword("enum")?.start;
        let (name, _) = self.expect_name()?;
        self.expect_symbol("{")?;
        let arms = self.comma_list("}", |parser| {
            let (name, span) = parser.expect_name()?;
            let payload = if parser.take_symbol("(").is_some() {
                let ty = parser.ty()?;
                parser.expect_symbol(")")?;
                Some(ty)
            } else {
                None
            };
            Ok(VariantArmDecl { name, payload, span: Span { start: span.start, end: parser.tokens[parser.pos - 1].span.end } })
        })?;
        let end = self.expect_symbol("}")?.end;
        Ok(VariantDecl {
            name: module.map_or_else(|| name.clone(), |module| format!("{module}::{name}")),
            module: module.map(str::to_owned),
            public,
            arms,
            span: Span { start, end },
        })
    }

    fn import(&mut self, owner: Option<&str>) -> Result<Import, FrontError> {
        let start = self.expect_keyword("import")?.start;
        let (mut module, _) = self.expect_name()?;
        self.expect_symbol("::")?;
        let (mut item, _) = self.expect_name()?;
        while self.take_symbol("::").is_some() {
            module.push_str("::");
            module.push_str(&item);
            item = self.expect_name()?.0;
        }
        let end = self.expect_symbol(";")?.end;
        Ok(Import {
            owner: owner.map(str::to_owned),
            module,
            item,
            span: Span { start, end },
        })
    }

    fn function(&mut self, module: Option<&str>, public: bool) -> Result<Function, FrontError> {
        let start = self.expect_keyword("fn")?.start;
        let (name, _) = self.expect_name()?;
        let generic_params = if self.take_symbol("<").is_some() {
            let params = self.comma_list(">", |parser| {
                let (name, _) = parser.expect_name()?;
                parser.expect_symbol(":")?;
                let (bound, span) = parser.expect_name()?;
                if bound != "Value" {
                    return Err(FrontError::source("UNSUPPORTED_FEATURE", "generic bound must be Value", span));
                }
                Ok(name)
            })?;
            self.expect_symbol(">")?;
            params
        } else {
            Vec::new()
        };
        self.expect_symbol("(")?;
        let params = self.comma_list(")", |parser| parser.param())?;
        self.expect_symbol(")")?;
        self.expect_symbol("->")?;
        let result = self.ty()?;
        let return_from = if self.take_keyword("from").is_some() {
            Some(self.expect_name()?.0)
        } else {
            None
        };
        let body = self.block()?;
        Ok(Function {
            name: module.map_or_else(|| name.clone(), |module| format!("{module}::{name}")),
            module: module.map(str::to_owned),
            public,
            generic_params,
            params,
            result,
            return_from,
            span: Span {
                start,
                end: body.span.end,
            },
            body,
            fold: crate::numeric_contract::FoldTier::Strict,
        })
    }

    fn param(&mut self) -> Result<Param, FrontError> {
        let (name, name_span) = self.expect_name()?;
        self.expect_symbol(":")?;
        let mode = if self.take_keyword("read").is_some() {
            Some(Mode::Read)
        } else if self.take_keyword("mut").is_some() {
            Some(Mode::Mut)
        } else if self.take_keyword("take").is_some() {
            Some(Mode::Take)
        } else {
            None
        };
        let ty = self.ty()?;
        Ok(Param {
            name,
            mode,
            ty,
            span: Span {
                start: name_span.start,
                end: self.tokens[self.pos - 1].span.end,
            },
        })
    }
}
