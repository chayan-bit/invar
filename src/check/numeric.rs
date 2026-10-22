//! Numeric policy v2 declarations (`docs/numeric-v2.md`): fold tiers and the
//! `reassoc_add` intrinsic. `acc = reassoc_add(acc, x)` in a loop that uses
//! `acc` nowhere else becomes `acc + x`, with its span kept for the f32 check
//! after typing. Tree, Exact and Relaxed folds, and the `reassoc_add` folds of
//! a Strict function, lower in `fold_tiers`.

mod fold_tiers;
mod trig;
pub(super) mod visit;
use super::*;
use visit::{Visit, walk_stmt};

/// How often each name is used, as a value or an assignment target, in the
/// innermost enclosing loop (its condition or bounds included).
type Uses = HashMap<String, usize>;

struct Count(Uses);

impl Visit for Count {
    fn expr(&mut self, e: &mut Expr) {
        if let ExprKind::Name(name) | ExprKind::Move(name) = &e.kind {
            self.place(name);
        }
    }
    fn place(&mut self, name: &str) {
        *self.0.entry(name.to_string()).or_default() += 1;
    }
}

fn misuse(span: Span) -> FrontError {
    FrontError::source(
        "NUMERIC_POLICY",
        "reassoc_add relaxes only a fold update `acc = reassoc_add(acc, x)` in a loop that uses `acc` nowhere else",
        span,
    )
}

/// Rewrite every fold update and refuse every other `reassoc_add`, then lower
/// the tiered folds. A rewritten update folds by its function's tier, and as
/// Tree in a Strict function.
pub(super) fn desugar(program: &mut Program) -> Result<Sites, FrontError> {
    let mut relaxed = Vec::new();
    let mut by_function = HashMap::new();
    for function in &mut program.functions {
        let first = relaxed.len();
        block(&mut function.body, &mut relaxed, None)?;
        let sites: HashSet<Span> = relaxed[first..].iter().copied().collect();
        by_function.insert(function.name.clone(), sites);
    }
    let mut lowered = Default::default();
    let folds = fold_tiers::rewrite(program, &by_function, &mut lowered)?;
    Ok(Sites {
        relaxed,
        folds,
        lowered,
    })
}

/// What the checks after typing look up by span: the rewritten `reassoc_add`
/// updates, and the sums no tier lowering took. `lowered` is the fold receipt.
pub(super) struct Sites {
    relaxed: Vec<Span>,
    folds: Vec<fold_tiers::Suspect>,
    pub(super) lowered: crate::numeric_contract::LoweredFolds,
}

fn block(block: &mut Block, sites: &mut Vec<Span>, uses: Option<&Uses>) -> Result<(), FrontError> {
    block
        .statements
        .iter_mut()
        .try_for_each(|statement| stmt(statement, sites, uses))
}

fn stmt(
    statement: &mut Stmt,
    sites: &mut Vec<Span>,
    uses: Option<&Uses>,
) -> Result<(), FrontError> {
    let mut count = Count(Uses::new());
    if matches!(
        statement.kind,
        StmtKind::While { .. } | StmtKind::For { .. }
    ) {
        walk_stmt(statement, &mut count);
    }
    match &mut statement.kind {
        StmtKind::Assign {
            place: Place::Name(acc, _),
            value,
        } if matches!(&value.kind, ExprKind::Call { name, .. } if name == "reassoc_add") => {
            let span = value.span;
            let ExprKind::Call { args, .. } = &mut value.kind else {
                unreachable!("matched call")
            };
            let [left, right] = std::mem::take(args).try_into().map_err(|_| misuse(span))?;
            // The target and the first argument are the loop's only uses.
            if !crate::front::named(&left, acc) || uses.and_then(|u| u.get(acc)) != Some(&2) {
                return Err(misuse(span));
            }
            expr(&right)?;
            let kind = ExprKind::Binary {
                left: Box::new(left),
                op: BinaryOp::Add,
                op_span: span,
                right: Box::new(right),
            };
            *value = Expr::new(kind, span)?;
            sites.push(span);
            Ok(())
        }
        StmtKind::Assign { place, value } => {
            if let Place::Index { index, .. } = place {
                expr(index)?;
            }
            expr(value)
        }
        StmtKind::Binding { value, .. } | StmtKind::Return(Some(value)) | StmtKind::Expr(value) => {
            expr(value)
        }
        StmtKind::Return(None) => Ok(()),
        StmtKind::If {
            condition,
            then_block,
            else_arm,
        } => {
            expr(condition)?;
            block(then_block, sites, uses)?;
            match else_arm {
                Some(ElseArm::Block(arm)) => block(arm, sites, uses),
                Some(ElseArm::If(arm)) => stmt(arm, sites, uses),
                None => Ok(()),
            }
        }
        StmtKind::While { condition, body } => {
            expr(condition)?;
            block(body, sites, Some(&count.0))
        }
        StmtKind::For {
            lower, upper, body, ..
        } => {
            expr(lower)?;
            expr(upper)?;
            block(body, sites, Some(&count.0))
        }
    }
}

/// Any `reassoc_add` inside an expression is not a fold update.
fn expr(e: &Expr) -> Result<(), FrontError> {
    match &e.kind {
        ExprKind::Call { name, .. } if name == "reassoc_add" => Err(misuse(e.span)),
        ExprKind::Call { args, .. } | ExprKind::Array(args) => args.iter().try_for_each(expr),
        ExprKind::Record { fields, .. } => fields.iter().try_for_each(|field| expr(&field.value)),
        ExprKind::Match { value, arms } => {
            expr(value)?;
            arms.iter().try_for_each(|arm: &MatchArm| expr(&arm.value))
        }
        ExprKind::MoveField { base, .. }
        | ExprKind::Field { base, .. }
        | ExprKind::Group(base)
        | ExprKind::Try(base)
        | ExprKind::Unary { value: base, .. } => expr(base),
        ExprKind::Index { base, index }
        | ExprKind::Binary {
            left: base,
            right: index,
            ..
        } => {
            expr(base)?;
            expr(index)
        }
        ExprKind::Integer(_)
        | ExprKind::Float(_)
        | ExprKind::Bool(_)
        | ExprKind::Text(_)
        | ExprKind::Bytes(_)
        | ExprKind::Name(_)
        | ExprKind::Move(_) => Ok(()),
    }
}

/// After typing: every rewritten update is an f32 addition, and the selected
/// trig contract names its f32 calls.
pub(super) fn finish(
    sites: &Sites,
    program: &mut Program,
    checker: &Checker,
) -> Result<(), FrontError> {
    trig::select(program, &checker.function_expression_types);
    let types = &checker.expression_types;
    for &site in &sites.relaxed {
        if types
            .iter()
            .any(|(span, ty)| *span == site && *ty != Type::F32)
        {
            return Err(FrontError::source(
                "NUMERIC_POLICY",
                "reassoc_add relaxes an f32 fold; its operands must be f32",
                site,
            ));
        }
    }
    fold_tiers::reject_unrewritten(&sites.folds, types)
}
