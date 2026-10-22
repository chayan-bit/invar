//! Structured statement tree recovered from one net-derived MIR function.
//!
//! The verified MIR builder emits a reducible CFG with three control shapes:
//! `if` (a branch whose arms re-merge or return), loops (`while` heads without
//! parameters, `for` heads with one u32 induction parameter) and short-circuit
//! `and`/`or` (a branch whose skip arm forwards the left operand to a join with
//! one Bool parameter). Arm extents follow dominance: an arm ends at the first
//! block its entry does not dominate. When the other arm returns, the code after
//! the `if` is dominated by the surviving arm and is placed inside it, which is
//! the same program. Anything else (match joins, records, views, `?`) is
//! reported as [`Unstructured`]; such functions stay serial frames.

use std::collections::{BTreeMap, HashMap, HashSet};

use super::cfg::Cfg;
use crate::{
    front::{BinaryOp, Mode, Span, Type, UnaryOp},
    general_net::coverage,
    mir::{BlockId, Function, Instruction, Operation, Terminator, ValueId},
    numeric_contract::F32Bits,
};

/// Maximum expression nesting; deeper expressions are refused, not recursed.
const MAX_DEPTH: usize = 256;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expr {
    I32(i32),
    U32(u32),
    F32(F32Bits),
    Bool(bool),
    /// `Read(name)` of a scalar local or parameter.
    Local(String),
    /// A named array owner used as a value: a call argument or a binding.
    Owner {
        name: String,
        loan: Option<Mode>,
    },
    Move(String),
    ArrayLit {
        elem: Type,
        items: Vec<Expr>,
    },
    /// `array[index]`; `span` is the trap span of the index operation.
    Index {
        array: String,
        index: Box<Expr>,
        span: Span,
    },
    Len(String),
    Unary {
        op: UnaryOp,
        value: Box<Expr>,
        span: Span,
    },
    /// Checked or f32 arithmetic and comparisons; `span` is the operator span.
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
        span: Span,
    },
    /// Short-circuit `and` (`and == true`) or `or`.
    Logic {
        and: bool,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Call {
        name: String,
        args: Vec<Expr>,
        span: Span,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoopKind {
    /// `for var in lo .. hi`; `lo` and `hi` are evaluated once at entry.
    For { var: String, lo: Expr, hi: Expr },
    /// `while cond`; the condition is evaluated before every iteration.
    While { cond: Expr },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Loop {
    /// Loop head block: the region key together with the function index.
    pub head: BlockId,
    pub kind: LoopKind,
    pub body: Vec<Stmt>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stmt {
    Let {
        name: String,
        value: Expr,
    },
    Assign {
        name: String,
        value: Expr,
    },
    /// `array[index] = value`; the value is evaluated before the index.
    Store {
        array: String,
        index: Expr,
        value: Expr,
        span: Span,
    },
    /// An expression evaluated for its traps or effects only.
    Eval(Expr),
    If {
        cond: Expr,
        yes: Vec<Stmt>,
        no: Vec<Stmt>,
    },
    Loop(Box<Loop>),
    Return {
        value: Option<Expr>,
        span: Span,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Param {
    pub name: String,
    pub ty: Type,
    pub mode: Option<Mode>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionTree {
    pub index: usize,
    pub name: String,
    pub params: Vec<Param>,
    pub result: Type,
    /// Declared type of every local and parameter.
    pub locals: BTreeMap<String, Type>,
    pub body: Vec<Stmt>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unstructured {
    Foreign,
    Cfg,
    JoinParameter,
    Operation,
    Depth,
    Shape,
}

impl Unstructured {
    pub const fn reason(self) -> &'static str {
        match self {
            Self::Foreign => "foreign_function",
            Self::Cfg => "invalid_cfg",
            Self::JoinParameter => "value_join",
            Self::Operation => "unsupported_operation",
            Self::Depth => "expression_depth",
            Self::Shape => "control_shape",
        }
    }
}

enum Flow {
    Returned,
    Reached(BlockId),
}

struct Decompiler<'a> {
    function: &'a Function,
    cfg: Cfg,
    defs: HashMap<ValueId, &'a Instruction>,
    uses: HashMap<ValueId, usize>,
    env: HashMap<ValueId, Expr>,
    heads: HashSet<BlockId>,
    inductions: HashSet<ValueId>,
    depth: usize,
}

/// Recover the statement tree of `function` (index `index` in its program).
pub fn decompile(function: &Function, index: usize) -> Result<FunctionTree, Unstructured> {
    if function.foreign_oracle.is_some() || function.blocks.is_empty() {
        return Err(Unstructured::Foreign);
    }
    let cfg = Cfg::new(function).ok_or(Unstructured::Cfg)?;
    let heads = cfg.loop_heads().into_iter().map(|(head, _)| head).collect();
    let mut defs = HashMap::new();
    let mut uses: HashMap<ValueId, usize> = HashMap::new();
    for block in &function.blocks {
        for instruction in &block.instructions {
            if let Some((id, _)) = &instruction.output {
                defs.insert(*id, instruction);
            }
            for operand in coverage::operands(&instruction.kind) {
                *uses.entry(operand).or_default() += 1;
            }
        }
        let operands: Vec<ValueId> = match &block.end {
            Some(Terminator::Goto { args, .. }) => args.clone(),
            Some(Terminator::Branch { condition, .. }) => vec![*condition],
            Some(Terminator::Return { value, .. }) => value.iter().copied().collect(),
            None => return Err(Unstructured::Cfg),
        };
        for operand in operands {
            *uses.entry(operand).or_default() += 1;
        }
    }
    let mut decompiler = Decompiler {
        function,
        cfg,
        defs,
        uses,
        env: HashMap::new(),
        heads,
        inductions: HashSet::new(),
        depth: 0,
    };
    let mut body = Vec::new();
    match decompiler.seq(function.entry, function.entry, &mut body)? {
        Flow::Returned => {}
        Flow::Reached(_) => return Err(Unstructured::Shape),
    }
    let mut locals: BTreeMap<String, Type> = function
        .locals
        .iter()
        .map(|(name, local)| (name.clone(), local.ty.clone()))
        .collect();
    for param in &function.params {
        locals.insert(param.name.clone(), param.ty.clone());
    }
    Ok(FunctionTree {
        index,
        name: function.name.clone(),
        params: function
            .params
            .iter()
            .map(|param| Param {
                name: param.name.clone(),
                ty: param.ty.clone(),
                mode: param.mode,
            })
            .collect(),
        result: function.result.clone(),
        locals,
        body,
    })
}

impl<'a> Decompiler<'a> {
    fn block(&self, id: BlockId) -> Result<&'a crate::mir::BasicBlock, Unstructured> {
        self.function.blocks.get(id).ok_or(Unstructured::Cfg)
    }

    fn end(&self, id: BlockId) -> Result<&'a Terminator, Unstructured> {
        self.block(id)?.end.as_ref().ok_or(Unstructured::Cfg)
    }

    fn seq(
        &mut self,
        start: BlockId,
        scope: BlockId,
        out: &mut Vec<Stmt>,
    ) -> Result<Flow, Unstructured> {
        let mut block = start;
        loop {
            if block != scope && !self.cfg.dominates(scope, block) {
                return Ok(Flow::Reached(block));
            }
            self.statements(block, out)?;
            match self.end(block)?.clone() {
                Terminator::Return { value, span, .. } => {
                    let value = value.map(|id| self.expr(id)).transpose()?;
                    out.push(Stmt::Return { value, span });
                    return Ok(Flow::Returned);
                }
                Terminator::Goto { target, args, .. } => {
                    if self.heads.contains(&target) {
                        if self.cfg.dominates(target, block) {
                            return Ok(Flow::Reached(target));
                        }
                        block = self.parse_loop(target, &args, out)?;
                        continue;
                    }
                    if !args.is_empty() {
                        return Err(Unstructured::JoinParameter);
                    }
                    block = target;
                }
                Terminator::Branch {
                    condition, yes, no, ..
                } => {
                    if let Some(join) = self.short_circuit(condition, yes, no)? {
                        block = join;
                        continue;
                    }
                    let cond = self.expr(condition)?;
                    let (mut then, mut other) = (Vec::new(), Vec::new());
                    let first = self.seq(yes, yes, &mut then)?;
                    let second = self.seq(no, no, &mut other)?;
                    out.push(Stmt::If {
                        cond,
                        yes: then,
                        no: other,
                    });
                    block = match (first, second) {
                        (Flow::Returned, Flow::Returned) => return Ok(Flow::Returned),
                        (Flow::Reached(next), Flow::Returned)
                        | (Flow::Returned, Flow::Reached(next)) => next,
                        (Flow::Reached(a), Flow::Reached(b)) if a == b => a,
                        _ => return Err(Unstructured::Shape),
                    };
                }
            }
        }
    }

    /// Parse the loop headed by `head`, entered with `args`; returns its exit.
    fn parse_loop(
        &mut self,
        head: BlockId,
        args: &[ValueId],
        out: &mut Vec<Stmt>,
    ) -> Result<BlockId, Unstructured> {
        let header = self.block(head)?;
        let (kind, body_entry, exit) = match (header.params.as_slice(), args) {
            ([], []) => {
                let (cond, yes, no) = self.condition_chain(head)?;
                (LoopKind::While { cond }, yes, no)
            }
            ([(induction, Type::U32)], [lower]) => {
                let induction = *induction;
                let [compare] = header.instructions.as_slice() else {
                    return Err(Unstructured::Shape);
                };
                let (
                    Operation::Binary {
                        op: BinaryOp::Lt,
                        left,
                        right,
                        ..
                    },
                    Some((test, _)),
                ) = (&compare.kind, &compare.output)
                else {
                    return Err(Unstructured::Shape);
                };
                let Terminator::Branch {
                    condition, yes, no, ..
                } = self.end(head)?.clone()
                else {
                    return Err(Unstructured::Shape);
                };
                if *left != induction || condition != *test {
                    return Err(Unstructured::Shape);
                }
                let var = self
                    .block(yes)?
                    .instructions
                    .iter()
                    .find_map(|instruction| match &instruction.kind {
                        Operation::Bind { name, value } if *value == induction => {
                            Some(name.clone())
                        }
                        _ => None,
                    })
                    .ok_or(Unstructured::Shape)?;
                let lo = self.expr(*lower)?;
                let hi = self.expr(*right)?;
                self.inductions.insert(induction);
                self.env.insert(induction, Expr::Local(var.clone()));
                (LoopKind::For { var, lo, hi }, yes, no)
            }
            _ => return Err(Unstructured::Shape),
        };
        let mut body = Vec::new();
        match self.seq(body_entry, body_entry, &mut body)? {
            Flow::Returned => {}
            Flow::Reached(target) if target == head => {}
            Flow::Reached(_) => return Err(Unstructured::Shape),
        }
        out.push(Stmt::Loop(Box::new(Loop { head, kind, body })));
        Ok(exit)
    }

    /// A `while` condition: expression blocks and short circuits ending in the
    /// deciding branch. Returns `(condition, body, exit)`.
    fn condition_chain(&mut self, head: BlockId) -> Result<(Expr, BlockId, BlockId), Unstructured> {
        let mut block = head;
        loop {
            self.expression_block(block)?;
            let Terminator::Branch {
                condition, yes, no, ..
            } = self.end(block)?.clone()
            else {
                return Err(Unstructured::Shape);
            };
            match self.short_circuit(condition, yes, no)? {
                Some(join) => block = join,
                None => return Ok((self.expr(condition)?, yes, no)),
            }
        }
    }

    fn expression_block(&mut self, block: BlockId) -> Result<(), Unstructured> {
        let mut stray = Vec::new();
        self.statements(block, &mut stray)?;
        if stray.is_empty() {
            Ok(())
        } else {
            Err(Unstructured::Shape)
        }
    }

    /// Fold a short-circuit branch into a `Logic` value for its join parameter.
    fn short_circuit(
        &mut self,
        condition: ValueId,
        yes: BlockId,
        no: BlockId,
    ) -> Result<Option<BlockId>, Unstructured> {
        let skip_to = |this: &Self, arm: BlockId| -> Option<BlockId> {
            let block = this.function.blocks.get(arm)?;
            match &block.end {
                Some(Terminator::Goto { target, args, .. })
                    if block.instructions.is_empty() && args.as_slice() == [condition] =>
                {
                    Some(*target)
                }
                _ => None,
            }
        };
        let (demand, join, and) = if let Some(join) = skip_to(self, no) {
            (yes, join, true)
        } else if let Some(join) = skip_to(self, yes) {
            (no, join, false)
        } else {
            return Ok(None);
        };
        let [(param, Type::Bool)] = self.block(join)?.params.as_slice() else {
            return Err(Unstructured::JoinParameter);
        };
        let param = *param;
        let mut block = demand;
        let right = loop {
            self.expression_block(block)?;
            match self.end(block)?.clone() {
                Terminator::Goto { target, args, .. } if target == join => {
                    let [value] = args.as_slice() else {
                        return Err(Unstructured::Shape);
                    };
                    break self.expr(*value)?;
                }
                Terminator::Branch {
                    condition, yes, no, ..
                } => match self.short_circuit(condition, yes, no)? {
                    Some(next) => block = next,
                    None => return Err(Unstructured::Shape),
                },
                _ => return Err(Unstructured::Shape),
            }
        };
        let left = self.expr(condition)?;
        self.env.insert(
            param,
            Expr::Logic {
                and,
                left: Box::new(left),
                right: Box::new(right),
            },
        );
        Ok(Some(join))
    }

    fn statements(&mut self, block: BlockId, out: &mut Vec<Stmt>) -> Result<(), Unstructured> {
        let instructions = &self.block(block)?.instructions;
        for instruction in instructions {
            match &instruction.kind {
                Operation::Tick
                | Operation::Drop(_)
                | Operation::DropValue(_)
                | Operation::ForNext(_) => {}
                Operation::Bind { name, value } => {
                    if !self.inductions.contains(value) {
                        let value = self.expr(*value)?;
                        out.push(Stmt::Let {
                            name: name.clone(),
                            value,
                        });
                    }
                }
                Operation::Store { name, value } => {
                    let value = self.expr(*value)?;
                    out.push(Stmt::Assign {
                        name: name.clone(),
                        value,
                    });
                }
                Operation::StoreIndex { name, index, value } => {
                    let value = self.expr(*value)?;
                    let index = self.expr(*index)?;
                    out.push(Stmt::Store {
                        array: name.clone(),
                        index,
                        value,
                        span: instruction.span,
                    });
                }
                kind => {
                    let pure = matches!(
                        kind,
                        Operation::I32(_)
                            | Operation::U32(_)
                            | Operation::F32(_)
                            | Operation::Bool(_)
                            | Operation::Read(_)
                            | Operation::ArrayRef { .. }
                            | Operation::Length(_)
                    );
                    if let Some((id, _)) = &instruction.output
                        && !pure
                        && self.uses.get(id).copied().unwrap_or(0) == 0
                    {
                        out.push(Stmt::Eval(self.expr(*id)?));
                    }
                }
            }
        }
        Ok(())
    }

    fn expr(&mut self, id: ValueId) -> Result<Expr, Unstructured> {
        if let Some(expr) = self.env.get(&id) {
            return Ok(expr.clone());
        }
        if self.depth >= MAX_DEPTH {
            return Err(Unstructured::Depth);
        }
        self.depth += 1;
        let result = self.expr_inner(id);
        self.depth -= 1;
        result
    }

    fn boxed(&mut self, id: ValueId) -> Result<Box<Expr>, Unstructured> {
        self.expr(id).map(Box::new)
    }

    fn expr_inner(&mut self, id: ValueId) -> Result<Expr, Unstructured> {
        let instruction = *self.defs.get(&id).ok_or(Unstructured::Shape)?;
        Ok(match &instruction.kind {
            Operation::I32(x) => Expr::I32(*x),
            Operation::U32(x) => Expr::U32(*x),
            Operation::F32(x) => Expr::F32(*x),
            Operation::Bool(x) => Expr::Bool(*x),
            Operation::Read(name) => Expr::Local(name.clone()),
            Operation::ArrayRef { name, loan } => Expr::Owner {
                name: name.clone(),
                loan: *loan,
            },
            Operation::Move(name) => Expr::Move(name.clone()),
            Operation::Length(name) => Expr::Len(name.clone()),
            Operation::Array(items) => {
                let elem = match instruction.output.as_ref().map(|(_, ty)| ty) {
                    Some(Type::ArrayI32) => Type::I32,
                    Some(Type::ArrayU32) => Type::U32,
                    Some(Type::ArrayF32) => Type::F32,
                    _ => return Err(Unstructured::Operation),
                };
                let items = items
                    .iter()
                    .map(|item| self.expr(*item))
                    .collect::<Result<_, _>>()?;
                Expr::ArrayLit { elem, items }
            }
            Operation::Index { array, index } => {
                let Expr::Owner { name, .. } = self.expr(*array)? else {
                    return Err(Unstructured::Operation);
                };
                Expr::Index {
                    array: name,
                    index: self.boxed(*index)?,
                    span: instruction.span,
                }
            }
            Operation::Unary {
                op,
                value,
                operator,
            } => Expr::Unary {
                op: *op,
                value: self.boxed(*value)?,
                span: *operator,
            },
            Operation::Binary {
                op,
                left,
                right,
                operator,
            } => Expr::Binary {
                op: *op,
                left: self.boxed(*left)?,
                right: self.boxed(*right)?,
                span: *operator,
            },
            Operation::Call { name, args } => Expr::Call {
                name: name.clone(),
                args: args
                    .iter()
                    .map(|arg| self.expr(*arg))
                    .collect::<Result<_, _>>()?,
                span: instruction.span,
            },
            _ => return Err(Unstructured::Operation),
        })
    }
}

impl Expr {
    /// Every scalar local read and array owner named by this expression.
    pub fn names(&self, out: &mut Vec<String>) {
        match self {
            Self::I32(_) | Self::U32(_) | Self::F32(_) | Self::Bool(_) => {}
            Self::Local(name) | Self::Move(name) | Self::Len(name) => out.push(name.clone()),
            Self::Owner { name, .. } => out.push(name.clone()),
            Self::ArrayLit { items, .. } => items.iter().for_each(|item| item.names(out)),
            Self::Index { array, index, .. } => {
                out.push(array.clone());
                index.names(out);
            }
            Self::Unary { value, .. } => value.names(out),
            Self::Binary { left, right, .. } | Self::Logic { left, right, .. } => {
                left.names(out);
                right.names(out);
            }
            Self::Call { args, .. } => args.iter().for_each(|arg| arg.names(out)),
        }
    }

    pub fn mentions(&self, name: &str) -> bool {
        let mut names = Vec::new();
        self.names(&mut names);
        names.iter().any(|candidate| candidate == name)
    }

    /// Calls made by this expression, in evaluation order.
    pub fn calls<'a>(&'a self, out: &mut Vec<&'a Expr>) {
        match self {
            Self::Call { args, .. } => {
                args.iter().for_each(|arg| arg.calls(out));
                out.push(self);
            }
            Self::ArrayLit { items, .. } => items.iter().for_each(|item| item.calls(out)),
            Self::Index { index, .. } => index.calls(out),
            Self::Unary { value, .. } => value.calls(out),
            Self::Binary { left, right, .. } | Self::Logic { left, right, .. } => {
                left.calls(out);
                right.calls(out);
            }
            _ => {}
        }
    }

    /// Integer literal value, if this is one.
    pub fn literal(&self) -> Option<i64> {
        match self {
            Self::I32(x) => Some(i64::from(*x)),
            Self::U32(x) => Some(i64::from(*x)),
            _ => None,
        }
    }
}

/// Structural equality that ignores source spans: two occurrences of the same
/// source expression at different positions compare equal.
pub fn same(a: &Expr, b: &Expr) -> bool {
    match (a, b) {
        (
            Expr::Index { array, index, .. },
            Expr::Index {
                array: x, index: y, ..
            },
        ) => array == x && same(index, y),
        (
            Expr::Unary { op, value, .. },
            Expr::Unary {
                op: o, value: v, ..
            },
        ) => op == o && same(value, v),
        (
            Expr::Binary {
                op, left, right, ..
            },
            Expr::Binary {
                op: o,
                left: l,
                right: r,
                ..
            },
        ) => op == o && same(left, l) && same(right, r),
        (
            Expr::Logic { and, left, right },
            Expr::Logic {
                and: a2,
                left: l,
                right: r,
            },
        ) => and == a2 && same(left, l) && same(right, r),
        (
            Expr::Call { name, args, .. },
            Expr::Call {
                name: n, args: a2, ..
            },
        ) => name == n && args.len() == a2.len() && args.iter().zip(a2).all(|(x, y)| same(x, y)),
        (Expr::ArrayLit { elem, items }, Expr::ArrayLit { elem: e, items: i }) => {
            elem == e && items.len() == i.len() && items.iter().zip(i).all(|(x, y)| same(x, y))
        }
        _ => a == b,
    }
}

/// Visit every loop in `body` in source pre-order with its nesting path.
pub fn for_each_loop<'a>(body: &'a [Stmt], visit: &mut dyn FnMut(&'a Loop)) {
    for stmt in body {
        match stmt {
            Stmt::Loop(inner) => {
                visit(inner);
                for_each_loop(&inner.body, visit);
            }
            Stmt::If { yes, no, .. } => {
                for_each_loop(yes, visit);
                for_each_loop(no, visit);
            }
            _ => {}
        }
    }
}

/// The loop headed by `head` inside `body`, if any.
pub fn find_loop(body: &[Stmt], head: BlockId) -> Option<&Loop> {
    let mut found = None;
    for_each_loop(body, &mut |candidate| {
        if candidate.head == head && found.is_none() {
            found = Some(candidate);
        }
    });
    found
}

/// Visitor of [`walk`]: a statement, its guards and its loop depth.
pub type Visit<'v, 'a> = dyn FnMut(&'a Stmt, &[(&'a Expr, bool)], usize) + 'v;

/// Statements in `body`, recursively, with the conditions guarding them
/// (`(condition, polarity)`, outermost first) and their loop depth relative
/// to `body`.
pub fn walk<'a>(
    body: &'a [Stmt],
    guards: &mut Vec<(&'a Expr, bool)>,
    depth: usize,
    visit: &mut Visit<'_, 'a>,
) {
    for stmt in body {
        visit(stmt, guards, depth);
        match stmt {
            Stmt::If { cond, yes, no } => {
                guards.push((cond, true));
                walk(yes, guards, depth, visit);
                guards.pop();
                guards.push((cond, false));
                walk(no, guards, depth, visit);
                guards.pop();
            }
            Stmt::Loop(inner) => {
                if let LoopKind::While { cond } = &inner.kind {
                    guards.push((cond, true));
                    walk(&inner.body, guards, depth + 1, visit);
                    guards.pop();
                } else {
                    walk(&inner.body, guards, depth + 1, visit);
                }
            }
            _ => {}
        }
    }
}

/// Expressions evaluated directly by `stmt` (not by nested statements).
pub fn stmt_exprs(stmt: &Stmt) -> Vec<&Expr> {
    match stmt {
        Stmt::Let { value, .. } | Stmt::Assign { value, .. } | Stmt::Eval(value) => vec![value],
        Stmt::Store { index, value, .. } => vec![value, index],
        Stmt::If { cond, .. } => vec![cond],
        Stmt::Loop(inner) => match &inner.kind {
            LoopKind::For { lo, hi, .. } => vec![lo, hi],
            LoopKind::While { cond } => vec![cond],
        },
        Stmt::Return { value, .. } => value.iter().collect(),
    }
}
