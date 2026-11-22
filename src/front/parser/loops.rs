// Surface forms of `for` desugared to the counted loop before checking and
// family recognition, which reparses source (N11.4.3): for-each (#255)
// and stepped ranges (#256).
//
// Checking keys expression types by span, and MIR spans must be non-empty,
// so every synthesized node takes a span no parsed expression has: a range
// starting at the `for` keyword or at the element name, which no
// expression starts at.

/// The element of a for-each loop and the hidden index of its counted loop.
struct Element<'a> {
    name: &'a str,
    array: &'a str,
    index: &'a str,
    /// Where the `for` keyword and the element name start.
    origin: (usize, usize),
}

impl Element<'_> {
    fn name_at(&self, name: &str, start: usize, end: usize) -> Result<Expr, FrontError> {
        Expr::new(ExprKind::Name(name.to_owned()), Span { start, end })
    }

    /// `array[index]` at `span`, its base and index ending with it.
    fn place(&self, span: Span) -> Result<Expr, FrontError> {
        let base = self.name_at(self.array, self.origin.0, span.end)?;
        let index = self.name_at(self.index, self.origin.1, span.end)?;
        Expr::new(
            ExprKind::Index {
                base: Box::new(base),
                index: Box::new(index),
            },
            span,
        )
    }
}

impl Parser {
    /// `for x in xs { body }`. The element is bound by value with
    /// `let x = xs[each@N];`, except that a body assigning `x` iterates by
    /// place: the index is named `x` and every use of `x` becomes `xs[x]`, so
    /// a `MutView` (or a `var` array) is written through and reads see the
    /// current element.
    fn for_each(
        &mut self,
        start: usize,
        element: String,
        element_span: Span,
        iterable: Expr,
    ) -> Result<Stmt, FrontError> {
        let ExprKind::Name(array) = iterable.kind else {
            return Err(FrontError::source(
                "PARSE_EXPECTED_TOKEN",
                "expected `..` or a named array or view to iterate",
                iterable.span,
            ));
        };
        let at = iterable.span;
        let first_closure = self.closures.len();
        let body = self.block()?;
        let by_place = assigns(&body.statements, &element);
        // Closure captures resolve by name outside the rewritten body, where
        // `x` names the index, so a by-place element cannot be captured.
        let captured = self.closures[first_closure..]
            .iter()
            .flat_map(|closure| &closure.captures)
            .find(|capture| capture.name == element);
        if let (true, Some(capture)) = (by_place, captured) {
            return Err(FrontError::source(
                "OWNERSHIP",
                "an element written in its for-each body is an array place and cannot be captured; capture a `let` copy of it",
                capture.span,
            ));
        }
        // By place, the index takes the element name, which every `x` reads
        // through, so the checker reserves the name as `let x` does by value.
        let index = if by_place {
            element.clone()
        } else {
            format!("each@{}", element_span.start)
        };
        let site = Element {
            name: &element,
            array: &array,
            index: &index,
            origin: (start, element_span.start),
        };
        let mut statements = Vec::with_capacity(body.statements.len() + 1);
        if by_place {
            for statement in body.statements {
                statements.push(place_stmt(statement, &site)?);
            }
        } else {
            let value = site.place(Span {
                start: element_span.end,
                end: at.end,
            })?;
            let binding = StmtKind::Binding {
                mutable: false,
                name: element.clone(),
                annotation: None,
                value,
            };
            statements.push(Stmt::new(binding, element_span)?);
            statements.extend(body.statements);
        }
        let body = Block::new(statements, body.span)?;
        let lower = Span {
            start,
            end: at.start,
        };
        let lower = Expr::new(ExprKind::Integer("0u32".to_owned()), lower)?;
        let length = vec![site.name_at(&array, start, element_span.end)?];
        let upper = Expr::new(
            ExprKind::Call {
                name: "len".to_owned(),
                args: length,
            },
            at,
        )?;
        let span = Span {
            start,
            end: body.span.end,
        };
        Stmt::new(
            StmtKind::For {
                name: index,
                lower,
                upper,
                body,
            },
            span,
        )
    }
}

/// Whether any statement, at any depth, assigns the name `element`.
fn assigns(statements: &[Stmt], element: &str) -> bool {
    statements.iter().any(|statement| match &statement.kind {
        StmtKind::Assign {
            place: Place::Name(name, _),
            ..
        } => name == element,
        StmtKind::If {
            then_block,
            else_arm,
            ..
        } => {
            assigns(&then_block.statements, element)
                || match else_arm {
                    Some(ElseArm::Block(block)) => assigns(&block.statements, element),
                    Some(ElseArm::If(nested)) => assigns(std::slice::from_ref(nested), element),
                    None => false,
                }
        }
        StmtKind::While { body, .. } | StmtKind::For { body, .. } => {
            assigns(&body.statements, element)
        }
        _ => false,
    })
}

/// The element name is bound by the for-each loop for its whole body.
fn rebound(span: Span) -> FrontError {
    FrontError::source(
        "NAME_DUPLICATE",
        "the for-each element name is already bound in this function",
        span,
    )
}

fn place_block(block: Block, site: &Element) -> Result<Block, FrontError> {
    let mut statements = Vec::with_capacity(block.statements.len());
    for statement in block.statements {
        statements.push(place_stmt(statement, site)?);
    }
    Block::new(statements, block.span)
}

fn place_stmt(stmt: Stmt, site: &Element) -> Result<Stmt, FrontError> {
    let e = site.name;
    let expr = |value: Expr| place_expr(value, site);
    let block = |value: Block| place_block(value, site);
    let kind = match stmt.kind {
        StmtKind::Binding { name, .. } | StmtKind::For { name, .. } if name == e => {
            return Err(rebound(stmt.span));
        }
        StmtKind::Binding {
            mutable,
            name,
            annotation,
            value,
        } => StmtKind::Binding {
            mutable,
            name,
            annotation,
            value: expr(value)?,
        },
        StmtKind::Assign { place, value } => {
            let place = match place {
                Place::Name(name, span) if name == e => Place::Index {
                    name: site.array.to_owned(),
                    index: site.name_at(site.index, site.origin.1, span.end)?,
                    span,
                },
                Place::Index {
                    name,
                    index: at,
                    span,
                } => Place::Index {
                    name,
                    index: expr(at)?,
                    span,
                },
                place => place,
            };
            StmtKind::Assign {
                place,
                value: expr(value)?,
            }
        }
        StmtKind::Return(value) => StmtKind::Return(value.map(expr).transpose()?),
        kind @ (StmtKind::Break(_) | StmtKind::Continue(_)) => kind,
        StmtKind::If {
            condition,
            then_block,
            else_arm,
        } => StmtKind::If {
            condition: expr(condition)?,
            then_block: block(then_block)?,
            else_arm: match else_arm {
                Some(ElseArm::Block(arm)) => Some(ElseArm::Block(block(arm)?)),
                Some(ElseArm::If(nested)) => {
                    Some(ElseArm::If(Box::new(place_stmt(*nested, site)?)))
                }
                None => None,
            },
        },
        StmtKind::While { condition, body } => StmtKind::While {
            condition: expr(condition)?,
            body: block(body)?,
        },
        StmtKind::For {
            name,
            lower,
            upper,
            body,
        } => StmtKind::For {
            name,
            lower: expr(lower)?,
            upper: expr(upper)?,
            body: block(body)?,
        },
        StmtKind::Expr(value) => StmtKind::Expr(expr(value)?),
    };
    Stmt::new(kind, stmt.span)
}

fn place_expr(expr: Expr, site: &Element) -> Result<Expr, FrontError> {
    let e = site.name;
    let go = |value: Expr| place_expr(value, site);
    let boxed = |value: Box<Expr>| place_expr(*value, site).map(Box::new);
    let all = |values: Vec<Expr>| values.into_iter().map(go).collect::<Result<Vec<_>, _>>();
    let span = expr.span;
    let kind = match expr.kind {
        ExprKind::Name(name) if name == e => return site.place(span),
        ExprKind::Move(name) if name == e => {
            return Err(FrontError::source(
                "OWNERSHIP",
                "an element written in its for-each body is an array place and cannot be moved",
                span,
            ));
        }
        ExprKind::Group(value) => ExprKind::Group(boxed(value)?),
        ExprKind::Try(value) => ExprKind::Try(boxed(value)?),
        ExprKind::Call { name, args } => ExprKind::Call {
            name,
            args: all(args)?,
        },
        ExprKind::Array(items) => ExprKind::Array(all(items)?),
        ExprKind::Record { name, fields } => ExprKind::Record {
            name,
            fields: fields
                .into_iter()
                .map(|field| {
                    Ok(FieldInit {
                        value: go(field.value)?,
                        ..field
                    })
                })
                .collect::<Result<_, FrontError>>()?,
        },
        ExprKind::Match { value, arms } => ExprKind::Match {
            value: boxed(value)?,
            arms: arms
                .into_iter()
                .map(|arm| {
                    if arm.binding.as_deref() == Some(e) {
                        return Err(rebound(arm.span));
                    }
                    Ok(MatchArm {
                        guard: arm.guard.map(&go).transpose()?,
                        value: go(arm.value)?,
                        ..arm
                    })
                })
                .collect::<Result<_, FrontError>>()?,
        },
        ExprKind::Field { base, name } => ExprKind::Field {
            base: boxed(base)?,
            name,
        },
        ExprKind::Index { base, index: at } => ExprKind::Index {
            base: boxed(base)?,
            index: boxed(at)?,
        },
        ExprKind::Unary { op, op_span, value } => ExprKind::Unary {
            op,
            op_span,
            value: boxed(value)?,
        },
        ExprKind::Binary {
            left,
            op,
            op_span,
            right,
        } => ExprKind::Binary {
            left: boxed(left)?,
            op,
            op_span,
            right: boxed(right)?,
        },
        kind => kind,
    };
    Expr::new(kind, span)
}

/// Distinct spans for the synthesized nodes of a stepped loop: ranges from
/// the `for` keyword, then from the loop variable, to successive ends within
/// the header, so no parsed expression has one (see the top of this file).
struct Fresh<I: Iterator<Item = Span>>(I, Span);

impl<I: Iterator<Item = Span>> Fresh<I> {
    fn span(&mut self) -> Result<Span, FrontError> {
        self.0.next().ok_or_else(|| {
            FrontError::source(
                "PARSE_EXPECTED_TOKEN",
                "a stepped `for` header is too short to desugar",
                self.1,
            )
        })
    }

    fn name(&mut self, name: &str) -> Result<Expr, FrontError> {
        Expr::new(ExprKind::Name(name.to_owned()), self.span()?)
    }

    fn int(&mut self, value: &str) -> Result<Expr, FrontError> {
        Expr::new(ExprKind::Integer(value.to_owned()), self.span()?)
    }

    fn binary(&mut self, left: Expr, op: BinaryOp, right: Expr) -> Result<Expr, FrontError> {
        let span = self.span()?;
        binary(left, op, right, span)
    }
}

fn binary(left: Expr, op: BinaryOp, right: Expr, span: Span) -> Result<Expr, FrontError> {
    let kind = ExprKind::Binary {
        left: Box::new(left),
        op,
        op_span: span,
        right: Box::new(right),
    };
    Expr::new(kind, span)
}

impl Parser {
    /// `for i in lo .. hi step k { body }` (#256): the bounds and step are
    /// bound once, in that order, to hidden `u32` locals placed before the
    /// loop, which then counts `ceil((hi - lo) / k)` trips, the view stride
    /// rule (N5.5.2), with `i = lo + trip * k`. The trip count is zero when
    /// `hi <= lo`, so neither `hi - lo - 1` nor `lo + trip * k` can overflow,
    /// and a zero step traps `DIVISION_BY_ZERO`, over the header, only on a
    /// non-empty range.
    fn stepped_for(
        &mut self,
        start: usize,
        (element, element_span): (String, Span),
        lower: Expr,
        upper: Expr,
    ) -> Result<Stmt, FrontError> {
        let step = self.expression()?;
        let mut literal = &step;
        while let ExprKind::Group(inner) = &literal.kind {
            literal = inner;
        }
        if let ExprKind::Integer(value) = &literal.kind
            && ["u32", "i32", ""]
                .iter()
                .find_map(|suffix| value.strip_suffix(suffix))
                .is_some_and(|digits| digits.bytes().all(|digit| digit == b'0'))
        {
            return Err(FrontError::source(
                "PARSE_ZERO_STEP",
                "a `for` step must be a positive `u32`",
                step.span,
            ));
        }
        let header = Span {
            start,
            end: step.span.end,
        };
        let body = self.block()?;
        let hidden = |role: &str| format!("{role}@{}", element_span.start);
        let (low, high, stride, trips, trip) = (
            hidden("lo"),
            hidden("hi"),
            hidden("step"),
            hidden("trips"),
            hidden("trip"),
        );
        let (variable, header_end) = (element_span.start, body.span.start);
        let mut fresh = Fresh(
            (start + 1..=header_end)
                .filter(|end| *end != header.end)
                .map(|end| Span { start, end })
                .chain((variable + 1..=header_end).map(|end| Span {
                    start: variable,
                    end,
                })),
            header,
        );
        let bind = |mutable: bool, name: &str, value: Expr, span: Span| {
            let kind = StmtKind::Binding {
                mutable,
                name: name.to_owned(),
                annotation: Some(Type::U32),
                value,
            };
            Stmt::new(kind, span)
        };
        let (lower_span, upper_span, step_span) = (lower.span, upper.span, step.span);
        let mut hoisted = vec![
            bind(false, &low, lower, lower_span)?,
            bind(false, &high, upper, upper_span)?,
            bind(false, &stride, step, step_span)?,
            bind(true, &trips, fresh.int("0u32")?, element_span)?,
        ];
        // if hi > lo { trips = (hi - lo - 1u32) / step + 1u32; }
        let (left, right) = (fresh.name(&high)?, fresh.name(&low)?);
        let nonempty = fresh.binary(left, BinaryOp::Gt, right)?;
        let (left, right) = (fresh.name(&high)?, fresh.name(&low)?);
        let gap = fresh.binary(left, BinaryOp::Sub, right)?;
        let one = fresh.int("1u32")?;
        let last = fresh.binary(gap, BinaryOp::Sub, one)?;
        let divisor = fresh.name(&stride)?;
        let count = binary(last, BinaryOp::Div, divisor, header)?;
        let one = fresh.int("1u32")?;
        let count = fresh.binary(count, BinaryOp::Add, one)?;
        let assign = StmtKind::Assign {
            place: Place::Name(trips.clone(), element_span),
            value: count,
        };
        let then_block = Block::new(vec![Stmt::new(assign, header)?], header)?;
        let branch = StmtKind::If {
            condition: nonempty,
            then_block,
            else_arm: None,
        };
        hoisted.push(Stmt::new(branch, header)?);
        // let i: u32 = lo + trip * step;
        let (left, right) = (fresh.name(&trip)?, fresh.name(&stride)?);
        let offset = fresh.binary(left, BinaryOp::Mul, right)?;
        let base = fresh.name(&low)?;
        let value = fresh.binary(base, BinaryOp::Add, offset)?;
        let mut statements = vec![bind(false, &element, value, element_span)?];
        statements.extend(body.statements);
        let end = body.span.end;
        let body = Block::new(statements, body.span)?;
        let kind = StmtKind::For {
            name: trip,
            lower: fresh.int("0u32")?,
            upper: fresh.name(&trips)?,
            body,
        };
        // Placed before the loop by the enclosing block, after the body has
        // taken its own hoisted statements.
        self.hoisted.extend(hoisted);
        Stmt::new(kind, Span { start, end })
    }
}

// `break` and `continue`, resolved here to how many loops out they act (#261).
impl Parser {
    /// `name:` before `for` or `while` labels that loop.
    fn loop_label(&mut self) -> Result<Option<String>, FrontError> {
        let TokenKind::Name(label) = &self.current().kind else {
            return Ok(None);
        };
        if !matches!(self.tokens.get(self.pos + 1).map(|t| &t.kind), Some(TokenKind::Symbol(":"))) {
            return Ok(None);
        }
        let (label, span) = (label.clone(), self.bump().span);
        self.bump();
        if self.loops.iter().any(|outer| outer.as_deref() == Some(label.as_str())) {
            return Err(FrontError::source(
                "PARSE_LOOP_CONTROL",
                format!("loop label `{label}` already names an enclosing loop"),
                span,
            ));
        }
        Ok(Some(label))
    }

    /// `break [label];` or `continue [label];`.
    fn loop_control(&mut self) -> Result<Option<Stmt>, FrontError> {
        let next = self.is_keyword("continue");
        if !next && !self.is_keyword("break") {
            return Ok(None);
        }
        let start = self.bump().span.start;
        let label = match &self.current().kind {
            TokenKind::Name(label) => Some(label.clone()),
            _ => None,
        };
        if label.is_some() {
            self.bump();
        }
        let span = Span { start, end: self.expect_symbol(";")?.end };
        let depth = match &label {
            None if self.loops.is_empty() => None,
            None => Some(0),
            Some(label) => self.loops.iter().rev().position(|outer| outer.as_deref() == Some(label.as_str())),
        };
        let Some(depth) = depth else {
            let word = if next { "continue" } else { "break" };
            let message = match label {
                Some(label) => format!("`{word} {label}` names no enclosing loop"),
                None => format!("`{word}` outside a loop"),
            };
            return Err(FrontError::source("PARSE_LOOP_CONTROL", message, span));
        };
        let kind = if next { StmtKind::Continue(depth) } else { StmtKind::Break(depth) };
        Stmt::new(kind, span).map(Some)
    }
}
