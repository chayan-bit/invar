//! Tree and Exact folds (`docs/numeric.md`). In a function declared
//! `#[numeric(fold = "tree" | "exact")]`, an update `acc = acc + term` directly
//! in a counted `for` body, the only use there of an f32 accumulator declared
//! at the loop's own depth, becomes a term buffer and one `sum_tree` or
//! `sum_exact` call. Every engine then computes the same bits through the
//! existing deterministic reductions.
//!
//! Iteration `i` writes slot `i - lower` (Exact: slot 0 holds `init`). The
//! generated statements are parsed from fixed text and re-spanned to unique
//! nonempty spans inside the function that no source node has, so typed lookups
//! by span never collide. Every other self-referencing sum
//! inside a loop is kept as a suspect, and after typing an f32 suspect is
//! refused, so no fold in such a function silently stays strict.
//!
//! A Relaxed fold, and a `reassoc_add` fold in a Strict function, takes the
//! Tree lowering (ENG-03 step 5'), which the Relaxed contract admits; a shape the
//! lowering cannot take stays strict, which it admits too.

use super::visit::{Visit, walk_block, walk_expr};
use super::*;
use crate::numeric_contract::FoldTier;

fn refuse(tier: FoldTier, message: &str, span: Span) -> FrontError {
    FrontError::source(
        "NUMERIC_POLICY",
        format!("fold tier \"{}\": {message}", tier.name()),
        span,
    )
}

/// Counts the uses of one name, as a value or an assignment target.
struct Uses<'a>(&'a str, usize);

impl Visit for Uses<'_> {
    fn expr(&mut self, e: &mut Expr) {
        if matches!(&e.kind, ExprKind::Name(n) | ExprKind::Move(n) if n == self.0) {
            self.1 += 1;
        }
    }
    fn place(&mut self, name: &str) {
        self.1 += usize::from(name == self.0);
    }
}

/// Gives every node a fresh span inside the function, skipping the spans the
/// source already uses. MIR verification wants nonempty spans inside the
/// function and an operator span inside its expression's. Spans go row by
/// row: `start` stays fixed while `end` counts down from the function end, and
/// a unary or binary node starts only where its operator, allocated next, still
/// fits in the same row, so it nests. A function of L bytes holds about
/// L * L / 2 spans, far more than the about 70 nodes a fold of at least 25
/// source bytes generates, so `full` is a guard, not a length limit.
struct Respan {
    start: usize,
    end: usize,
    function: Span,
    taken: HashSet<Span>,
    full: bool,
}

impl Respan {
    fn new(function: Span, taken: HashSet<Span>) -> Self {
        Respan {
            start: function.start,
            end: function.end,
            function,
            taken,
            full: false,
        }
    }

    /// Moves to the first row with `room` free spans left; false when none.
    fn row_with(&mut self, room: usize) -> bool {
        while self.start < self.function.end {
            let free = (self.start + 1..=self.end)
                .rev()
                .filter(|&end| {
                    !self.taken.contains(&Span {
                        start: self.start,
                        end,
                    })
                })
                .take(room)
                .count();
            if free == room {
                return true;
            }
            self.start += 1;
            self.end = self.function.end;
        }
        false
    }

    fn assign(&mut self, span: &mut Span, room: usize) {
        if !self.row_with(room) {
            self.full = true;
            return;
        }
        loop {
            let fresh = Span {
                start: self.start,
                end: self.end,
            };
            self.end -= 1;
            if !self.taken.contains(&fresh) {
                *span = fresh;
                return;
            }
        }
    }
}

impl Visit for Respan {
    fn expr(&mut self, e: &mut Expr) {
        let room = if matches!(e.kind, ExprKind::Unary { .. } | ExprKind::Binary { .. }) {
            2
        } else {
            1
        };
        self.assign(&mut e.span, room);
    }
    fn span(&mut self, span: &mut Span) {
        self.assign(span, 1);
    }
}

/// Every span of the source, which fresh spans avoid, and every name, which
/// generated names avoid.
#[derive(Default)]
struct Taken {
    spans: HashSet<Span>,
    names: HashSet<String>,
}

impl Visit for Taken {
    fn expr(&mut self, e: &mut Expr) {
        self.spans.insert(e.span);
        if let ExprKind::Name(name) | ExprKind::Move(name) = &e.kind {
            self.names.insert(name.clone());
        }
    }
    fn span(&mut self, span: &mut Span) {
        self.spans.insert(*span);
    }
    fn place(&mut self, name: &str) {
        self.names.insert(name.to_owned());
    }
    fn bind(&mut self, name: &str) {
        self.names.insert(name.to_owned());
    }
}

/// A prefix no name starts with, so generated locals never collide with a
/// source binding, parameter or function.
fn fresh_prefix(names: &HashSet<String>) -> String {
    let mut prefix = String::from("__fold");
    while names.iter().any(|name| name.starts_with(&prefix)) {
        prefix.push('_');
    }
    prefix
}

/// A float initializer the checker types as `f32` without context.
fn float_literal(value: &Expr) -> bool {
    match &crate::front::ungroup(value).kind {
        ExprKind::Float(raw) => !raw.ends_with("f64"),
        ExprKind::Unary {
            op: UnaryOp::Neg,
            value,
            ..
        } => float_literal(value),
        _ => false,
    }
}

/// The term of `acc = acc + term` or `acc = term + acc`.
fn fold_term<'a>(acc: &str, value: &'a Expr) -> Option<&'a Expr> {
    let ExprKind::Binary {
        left,
        op: BinaryOp::Add,
        right,
        ..
    } = &value.kind
    else {
        return None;
    };
    if crate::front::named(left, acc) {
        Some(right)
    } else if crate::front::named(right, acc) {
        Some(left)
    } else {
        None
    }
}

/// The sum `e` (through `+`, `-`, unary `-` and grouping) has `target` as a term.
fn terms_contain(e: &Expr, target: &str) -> bool {
    match &crate::front::ungroup(e).kind {
        ExprKind::Binary {
            left,
            op: BinaryOp::Add | BinaryOp::Sub,
            right,
            ..
        } => terms_contain(left, target) || terms_contain(right, target),
        ExprKind::Unary {
            op: UnaryOp::Neg,
            value,
            ..
        } => terms_contain(value, target),
        ExprKind::Name(name) => name == target,
        _ => false,
    }
}

/// A fold the rewrite did not take: its tier and the span of its sum.
pub(super) type Suspect = (FoldTier, Span);

/// Rewrite the Tree and Exact folds of every declaring function.
pub(super) fn rewrite(
    program: &mut Program,
    reassoc: &HashMap<String, HashSet<Span>>,
    lowered: &mut crate::numeric_contract::LoweredFolds,
) -> Result<Vec<Suspect>, FrontError> {
    let mut suspects = Vec::new();
    let functions: Vec<String> = program.functions.iter().map(|f| f.name.clone()).collect();
    for function in &mut program.functions {
        let only = match function.fold {
            FoldTier::Tree | FoldTier::Exact | FoldTier::Relaxed => None,
            FoldTier::Strict => match reassoc.get(&function.name) {
                Some(sites) if !sites.is_empty() => Some(sites.clone()),
                _ => continue,
            },
        };
        let mut taken = Taken::default();
        taken.spans.insert(function.span);
        taken.names.extend(functions.iter().cloned());
        taken
            .names
            .extend(function.params.iter().map(|p| p.name.clone()));
        walk_block(&mut function.body, &mut taken);
        let mut rewriter = Rewriter {
            tier: match function.fold {
                FoldTier::Exact => FoldTier::Exact,
                _ => FoldTier::Tree,
            },
            relaxed: matches!(function.fold, FoldTier::Relaxed | FoldTier::Strict),
            only,
            floats: function
                .params
                .iter()
                .filter(|p| p.ty == Type::F32)
                .map(|p| p.name.clone())
                .collect(),
            declared: function
                .params
                .iter()
                .map(|p| (p.name.clone(), 0))
                .collect(),
            depth: 0,
            folds: 0,
            prefix: fresh_prefix(&taken.names),
            respan: Respan::new(function.span, taken.spans),
            suspects: Vec::new(),
        };
        let body = std::mem::replace(&mut function.body, Block::new(Vec::new(), function.span)?);
        function.body = rewriter.block(body)?;
        if rewriter.respan.full {
            return Err(refuse(
                function.fold,
                "function too short for its generated folds",
                function.span,
            ));
        }
        if rewriter.folds > 0 {
            let declared = match function.fold {
                FoldTier::Strict => FoldTier::Relaxed,
                tier => tier,
            };
            lowered.insert((declared, rewriter.tier));
        }
        suspects.extend(
            rewriter
                .suspects
                .into_iter()
                .map(|span| (function.fold, span)),
        );
    }
    Ok(suspects)
}

/// After typing: an f32 suspect is a fold no tier lowering took.
pub(super) fn reject_unrewritten(
    suspects: &[Suspect],
    types: &[(Span, Type)],
) -> Result<(), FrontError> {
    for &(tier, site) in suspects {
        if types
            .iter()
            .any(|(span, ty)| *span == site && *ty == Type::F32)
        {
            return Err(refuse(
                tier,
                "an f32 fold must be one update `acc = acc + term` directly in a for loop that reads `acc` nowhere else",
                site,
            ));
        }
    }
    Ok(())
}

struct Rewriter {
    /// The lowering: Tree or Exact.
    tier: FoldTier,
    /// Relaxed (or a Strict function's `reassoc_add` sites): a fold the
    /// lowering cannot take stays strict instead of being refused.
    relaxed: bool,
    /// In a Strict function, only the `reassoc_add` updates (by span) lower.
    only: Option<HashSet<Span>>,
    /// Names declared `f32` (annotation, float literal or parameter).
    floats: HashSet<String>,
    /// The loop depth of every declaration; shadowing is refused, and a later
    /// sibling declaration replaces an earlier one in traversal order.
    declared: HashMap<String, usize>,
    /// Loops enclosing the statement being rewritten.
    depth: usize,
    folds: usize,
    /// Starts every generated name; no source name starts with it.
    prefix: String,
    respan: Respan,
    suspects: Vec<Span>,
}

impl Rewriter {
    fn block(&mut self, block: Block) -> Result<Block, FrontError> {
        let mut statements = Vec::with_capacity(block.statements.len());
        for statement in block.statements {
            self.statement(statement, &mut statements)?;
        }
        Block::new(statements, block.span)
    }

    fn nested(&mut self, body: Block) -> Result<Block, FrontError> {
        self.depth += 1;
        let body = self.block(body);
        self.depth -= 1;
        body
    }

    fn statement(&mut self, statement: Stmt, out: &mut Vec<Stmt>) -> Result<(), FrontError> {
        let span = statement.span;
        let kind = match statement.kind {
            StmtKind::Binding {
                mutable,
                name,
                annotation,
                value,
            } => {
                if annotation.as_ref() == Some(&Type::F32)
                    || (annotation.is_none() && float_literal(&value))
                {
                    self.floats.insert(name.clone());
                }
                self.declared.insert(name.clone(), self.depth);
                StmtKind::Binding {
                    mutable,
                    name,
                    annotation,
                    value,
                }
            }
            StmtKind::Assign {
                place: Place::Name(target, place),
                value,
            } => {
                if !self.relaxed
                    && self
                        .declared
                        .get(&target)
                        .is_some_and(|&at| at < self.depth)
                    && terms_contain(&value, &target)
                {
                    self.suspects.push(value.span);
                }
                StmtKind::Assign {
                    place: Place::Name(target, place),
                    value,
                }
            }
            StmtKind::If {
                condition,
                then_block,
                else_arm,
            } => {
                let then_block = self.block(then_block)?;
                let else_arm = match else_arm {
                    Some(ElseArm::Block(arm)) => Some(ElseArm::Block(self.block(arm)?)),
                    // An `if` statement rewrites to exactly one statement.
                    Some(ElseArm::If(arm)) => self
                        .block(Block::new(vec![*arm], span)?)?
                        .statements
                        .pop()
                        .map(|arm| ElseArm::If(Box::new(arm))),
                    None => None,
                };
                StmtKind::If {
                    condition,
                    then_block,
                    else_arm,
                }
            }
            StmtKind::While { condition, body } => StmtKind::While {
                condition,
                body: self.nested(body)?,
            },
            StmtKind::For {
                name,
                lower,
                upper,
                body,
            } => {
                let body = self.nested(body)?;
                return self.fold_loop(name, lower, upper, body, span, out);
            }
            other => other,
        };
        out.push(Stmt::new(kind, span)?);
        Ok(())
    }

    /// Statements parsed from fixed text, re-spanned inside the function.
    fn parse(&mut self, text: &str) -> Result<Vec<Stmt>, FrontError> {
        let source = format!("fn fold() -> Unit {{ {text} }}");
        let mut body = crate::front::parse(source.as_bytes())?
            .functions
            .remove(0)
            .body;
        walk_block(&mut body, &mut self.respan);
        Ok(body.statements)
    }

    /// Why `acc` cannot take the tier lowering in this loop, if it cannot.
    fn misfit(&self, acc: &str, body: &mut Block, lower: &Expr, upper: &Expr) -> Option<String> {
        let mut uses = Uses(acc, 0);
        walk_block(body, &mut uses);
        let (mut low, mut high) = (lower.clone(), upper.clone());
        walk_expr(&mut low, &mut uses);
        walk_expr(&mut high, &mut uses);
        if uses.1 != 2 {
            return Some(format!(
                "the accumulator `{acc}` must be updated once, by `{acc} = {acc} + term`, and read nowhere else inside its for loop"
            ));
        }
        (self.declared[acc] < self.depth).then(|| format!(
            "the accumulator `{acc}` is declared outside an enclosing loop; tiered folds over nested loops are not supported, so fold each loop into its own accumulator"
        ))
    }

    fn fold_loop(
        &mut self,
        counter: String,
        lower: Expr,
        upper: Expr,
        mut body: Block,
        span: Span,
        out: &mut Vec<Stmt>,
    ) -> Result<(), FrontError> {
        // An accumulator declared inside the body restarts every iteration.
        let mut accs: Vec<String> = body
            .statements
            .iter()
            .filter_map(|s| match &s.kind {
                StmtKind::Assign {
                    place: Place::Name(acc, _),
                    value,
                } if self.floats.contains(acc)
                    && self.declared.get(acc).is_some_and(|&at| at <= self.depth)
                    && fold_term(acc, value).is_some()
                    && self
                        .only
                        .as_ref()
                        .is_none_or(|only| only.contains(&value.span)) =>
                {
                    Some(acc.clone())
                }
                _ => None,
            })
            .collect();
        accs.sort();
        accs.dedup();
        let mut kept = Vec::new();
        for acc in accs {
            match self.misfit(&acc, &mut body, &lower, &upper) {
                None => kept.push(acc),
                Some(_) if self.relaxed => {}
                Some(message) => return Err(refuse(self.tier, &message, span)),
            }
        }
        let accs = kept;
        if accs.is_empty() {
            out.push(Stmt::new(
                StmtKind::For {
                    name: counter,
                    lower,
                    upper,
                    body,
                },
                span,
            )?);
            return Ok(());
        }
        let id = self.folds;
        self.folds += 1;
        let id = format!("{}{id}", self.prefix);
        let (lo, hi, n) = (format!("{id}_lo"), format!("{id}_hi"), format!("{id}_n"));
        let exact = self.tier == FoldTier::Exact;
        // Exact adds the init slot, saturating: a count past the array limit
        // traps ARRAY_SIZE_LIMIT in `filled_f32`, never INTEGER_OVERFLOW.
        let (first, plus, slot_init) = if exact {
            (
                "1u32",
                " + 1u32",
                format!(" if {n} < 4294967295u32 {{ {n} = {n} + 1u32; }}"),
            )
        } else {
            ("0u32", "", String::new())
        };
        // The bounds are evaluated once, in order, as the loop would.
        let mut prelude = self.parse(&format!(
            "let {lo}: u32 = 0u32; let {hi}: u32 = 0u32; var {n}: u32 = {first}; if {hi} > {lo} {{ {n} = {hi} - {lo};{slot_init} }}"
        ))?;
        for (statement, bound) in prelude.iter_mut().zip([lower, upper]) {
            if let StmtKind::Binding { value, .. } = &mut statement.kind {
                *value = bound;
            }
        }
        out.append(&mut prelude);
        let mut after = Vec::new();
        for acc in &accs {
            let terms = format!("{id}_{acc}");
            let init = if exact {
                format!(" {terms}[0u32] = {acc};")
            } else {
                String::new()
            };
            out.append(&mut self.parse(&format!("var {terms} = filled_f32({n}, -0.0);{init}"))?);
            let slot = self
                .parse(&format!("{terms}[{counter} - {lo}{plus}] = 0.0;"))?
                .remove(0);
            for statement in &mut body.statements {
                if let StmtKind::Assign {
                    place: Place::Name(target, _),
                    value,
                } = &statement.kind
                    && target == acc
                    && let Some(term) = fold_term(acc, value)
                {
                    self.suspects.retain(|&site| site != value.span);
                    let mut write = slot.clone();
                    if let StmtKind::Assign { value, .. } = &mut write.kind {
                        *value = term.clone();
                    }
                    *statement = write;
                }
            }
            // A loop that never runs leaves `init` itself, -0 included.
            after.append(&mut self.parse(&if exact {
                format!("{acc} = sum_exact({terms});")
            } else {
                format!("if {hi} > {lo} {{ {acc} = {acc} + sum_tree({terms}); }}")
            })?);
        }
        let StmtKind::For { lower, upper, .. } = self
            .parse(&format!("for {counter} in {lo}..{hi} {{ }}"))?
            .remove(0)
            .kind
        else {
            unreachable!("parsed a for statement")
        };
        let body = Block::new(body.statements, body.span)?;
        out.push(Stmt::new(
            StmtKind::For {
                name: counter,
                lower,
                upper,
                body,
            },
            span,
        )?);
        out.append(&mut after);
        Ok(())
    }
}
