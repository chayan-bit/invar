//! Source-order interpretation of the statement tree, with the hooks the law
//! executor needs (lane array models, symbolic values, path-mode self calls).

use std::collections::HashMap;

use super::{
    Machine, Mutation, Stop, lanes, paths, sym,
    value::{self, Val, array_id},
};
use crate::{
    eval::TrapCategory,
    general_net::{
        normal_form::{Node, NormalForm},
        regions::{
            RegionId,
            tree::{Expr, Loop, LoopKind, Stmt},
        },
    },
};

pub(super) struct Frame {
    pub function: usize,
    pub vars: HashMap<String, Val>,
    /// Set while this frame executes one lane of a law loop.
    pub lane: Option<Box<lanes::Lane>>,
    /// L19: the first loop's lanes, waiting for the fused second loop.
    pub pending: Option<lanes::Pending>,
    /// Set while this frame executes one node of a path-sum call tree.
    pub path: Option<paths::Node>,
}

impl Frame {
    pub(super) fn new(function: usize, vars: HashMap<String, Val>) -> Self {
        Self {
            function,
            vars,
            lane: None,
            pending: None,
            path: None,
        }
    }

    /// Symbolic values are legal (and poison absorbs) in this frame.
    pub(super) fn symbolic(&self) -> bool {
        self.lane
            .as_ref()
            .is_some_and(|lane| lane.phase == lanes::Phase::Symbolic)
    }

    pub(super) fn get(&self, name: &str) -> Result<Val, Stop> {
        self.vars
            .get(name)
            .cloned()
            .ok_or_else(|| Stop::internal("unbound local"))
    }
}

/// Poison in a symbolic lane or a path-sum node; a poisoned lane event
/// otherwise (a replayed lane used an unrepresentable prefix).
pub(super) fn poison(frame: &Frame) -> Result<Val, Stop> {
    if frame.symbolic() || frame.path.is_some() {
        Ok(Val::poison())
    } else {
        Err(Stop::Poison)
    }
}

/// A condition's truth: `None` when it is unknown in a symbolic lane.
pub(super) fn truth(frame: &Frame, value: &Val) -> Result<Option<bool>, Stop> {
    match value {
        Val::Bool(b) => Ok(Some(*b)),
        Val::Sym(_) if frame.path.is_some() && !frame.symbolic() => {
            Err(Stop::violation("path-sum hole decides control"))
        }
        Val::Sym(_) => poison(frame).map(|_| None),
        _ => Err(Stop::internal("condition type")),
    }
}

fn fork_at(form: &NormalForm, function: usize, stmt: &Stmt) -> Option<usize> {
    let Stmt::Let {
        value: Expr::Call { span, .. },
        ..
    } = stmt
    else {
        return None;
    };
    form.functions
        .get(function)?
        .forks
        .iter()
        .find(|site| site.start == *span)
        .map(|site| site.siblings)
}

impl Machine<'_> {
    pub(super) fn call(&mut self, index: usize, args: Vec<Val>) -> Result<Val, Stop> {
        self.depth += 1;
        self.receipt.max_depth = self.receipt.max_depth.max(self.depth);
        let result = if self.depth > super::DEPTH_LIMIT {
            Err(Stop::tool("call depth limit"))
        } else if self
            .form
            .functions
            .get(index)
            .is_some_and(|f| f.recursion.is_some())
            && !self.serial.contains(&index)
        {
            paths::call(self, index, args)
        } else {
            self.call_serial(index, args)
        };
        self.depth -= 1;
        result
    }

    pub(super) fn call_serial(&mut self, index: usize, args: Vec<Val>) -> Result<Val, Stop> {
        let tree = self.function(index)?;
        let vars = tree
            .params
            .iter()
            .map(|p| p.name.clone())
            .zip(args)
            .collect();
        let mut frame = Frame::new(index, vars);
        match self.block(&mut frame, &tree.body) {
            Ok(()) => Ok(Val::Unit),
            Err(Stop::Return(value)) => Ok(value),
            Err(other) => Err(other),
        }
    }

    pub(super) fn block(&mut self, frame: &mut Frame, stmts: &[Stmt]) -> Result<(), Stop> {
        let mut i = 0;
        while i < stmts.len() {
            self.tick()?;
            if frame.lane.is_none()
                && frame.path.is_none()
                && let Some(n) = fork_at(self.form, frame.function, &stmts[i])
                && let Some(run) = stmts.get(i..i + n)
            {
                self.fork(frame, run)?;
                i += n;
                continue;
            }
            let stmt = &stmts[i];
            if frame.pending.is_some() && !lanes::keeps_pending(frame, stmt) {
                lanes::flush(frame)?;
            }
            self.stmt(frame, stmt)?;
            i += 1;
        }
        if frame.pending.is_some() {
            lanes::flush(frame)?;
        }
        Ok(())
    }

    /// L20: siblings evaluated in reverse (any order is allowed), committed in
    /// source order so the first source-order trap wins.
    fn fork(&mut self, frame: &mut Frame, run: &[Stmt]) -> Result<(), Stop> {
        let mut results = Vec::with_capacity(run.len());
        for stmt in run.iter().rev() {
            let Stmt::Let { name, value } = stmt else {
                return Err(Stop::internal("fork sibling"));
            };
            results.push((name, self.eval(frame, value)));
        }
        results.reverse();
        self.receipt
            .applied([crate::general_net::normal_form::LawId::L20ForkJoin]);
        self.receipt.lanes += run.len() as u64;
        // The first failure in source order is the one the serial program meets.
        let mut failures = results.iter().filter_map(|(_, r)| r.as_ref().err());
        let chosen = if self.mutation == Mutation::LastTrap {
            failures.rev().find(|stop| stop.is_event())
        } else {
            failures.next()
        };
        if let Some(stop) = chosen {
            return Err(stop.clone());
        }
        for (name, result) in results {
            frame.vars.insert(name.clone(), result?);
        }
        Ok(())
    }

    fn stmt(&mut self, frame: &mut Frame, stmt: &Stmt) -> Result<(), Stop> {
        match stmt {
            Stmt::Let { name, value } => {
                if frame.path.is_some() && paths::bind_child(self, frame, name, value)? {
                    return Ok(());
                }
                let value = self.eval(frame, value)?;
                frame.vars.insert(name.clone(), value);
                Ok(())
            }
            Stmt::Assign { name, value } => {
                let value = self.eval(frame, value)?;
                sym::assign(self, frame, name, value)
            }
            Stmt::Store {
                array,
                index,
                value,
                span,
            } => {
                let value = self.eval(frame, value)?;
                let index = self.eval(frame, index)?;
                let target = frame.get(array)?;
                let target = target.array()?;
                let Val::U32(index) = index else {
                    if index.is_sym() {
                        return sym::poison_store(frame, target);
                    }
                    return Err(Stop::internal("index type"));
                };
                let index = usize::try_from(index)
                    .map_err(|_| Stop::Trap(TrapCategory::IndexOutOfBounds, *span))?;
                lanes::store(frame, target, index, value, *span)
            }
            Stmt::Eval(expr) => self.eval(frame, expr).map(|_| ()),
            Stmt::If { cond, yes, no } => sym::branch(self, frame, cond, yes, no),
            Stmt::Loop(inner) => self.run_loop(frame, inner),
            Stmt::Return { value, .. } => {
                if frame.path.is_some() {
                    return paths::ret(self, frame, value.as_ref());
                }
                let value = match value {
                    Some(expr) => self.eval(frame, expr)?,
                    None => Val::Unit,
                };
                if value.is_sym() {
                    poison(frame)?;
                    return Err(Stop::violation("symbolic value returned"));
                }
                Err(Stop::Return(value))
            }
        }
    }

    fn run_loop(&mut self, frame: &mut Frame, inner: &Loop) -> Result<(), Stop> {
        if frame.lane.is_none() {
            let region = RegionId {
                function: frame.function,
                head: Some(inner.head),
            };
            if let Some(form) = self.form.loop_form(region)
                && let Node::Parallel(node) = &form.node
            {
                return lanes::run(self, frame, inner, form, node);
            }
        }
        self.serial_loop(frame, inner)
    }

    pub(super) fn serial_loop(&mut self, frame: &mut Frame, inner: &Loop) -> Result<(), Stop> {
        match &inner.kind {
            LoopKind::For { var, lo, hi } => {
                let lo = self.eval(frame, lo)?;
                let hi = self.eval(frame, hi)?;
                let (Val::U32(lo), Val::U32(hi)) = (&lo, &hi) else {
                    if lo.is_sym() || hi.is_sym() {
                        poison(frame)?;
                        return sym::skip(frame, &[&inner.body]);
                    }
                    return Err(Stop::internal("for bounds"));
                };
                for x in *lo..*hi {
                    self.tick()?;
                    frame.vars.insert(var.clone(), Val::U32(x));
                    self.block(frame, &inner.body)?;
                }
                Ok(())
            }
            LoopKind::While { cond } => loop {
                self.tick()?;
                let value = self.eval(frame, cond)?;
                match truth(frame, &value)? {
                    Some(true) => self.block(frame, &inner.body)?,
                    Some(false) => return Ok(()),
                    None => return sym::skip(frame, &[&inner.body]),
                }
            },
        }
    }

    pub(super) fn eval(&mut self, frame: &mut Frame, expr: &Expr) -> Result<Val, Stop> {
        match expr {
            Expr::I32(x) => Ok(Val::I32(*x)),
            Expr::U32(x) => Ok(Val::U32(*x)),
            Expr::F32(x) => Ok(Val::F32(*x)),
            Expr::Bool(x) => Ok(Val::Bool(*x)),
            Expr::Local(name) | Expr::Owner { name, .. } | Expr::Move(name) => frame.get(name),
            Expr::ArrayLit { elem, items } => {
                let items = items
                    .iter()
                    .map(|item| self.eval(frame, item))
                    .collect::<Result<Vec<_>, _>>()?;
                self.allocate(Some(items.len() as u64))?;
                if items.iter().any(Val::is_sym) {
                    return poison(frame);
                }
                value::array_literal(elem, items)
            }
            Expr::Index { array, index, span } => {
                let target = frame.get(array)?;
                let index = self.eval(frame, index)?;
                let Val::U32(index) = index else {
                    if index.is_sym() {
                        return poison(frame);
                    }
                    return Err(Stop::internal("index type"));
                };
                let index = usize::try_from(index)
                    .map_err(|_| Stop::Trap(TrapCategory::IndexOutOfBounds, *span))?;
                lanes::read(self, frame, target.array()?, index, *span)
            }
            Expr::Len(name) => {
                let target = frame.get(name)?;
                let length = target.array()?.borrow().len();
                Ok(Val::U32(
                    u32::try_from(length).map_err(|_| Stop::internal("length"))?,
                ))
            }
            Expr::Unary { op, value, span } => {
                let value = self.eval(frame, value)?;
                if value.is_sym() {
                    return sym::unary(self, frame, *op, &value);
                }
                value::unary(*op, &value, *span)
            }
            Expr::Binary {
                op,
                left,
                right,
                span,
            } => {
                let left = self.eval(frame, left)?;
                let right = self.eval(frame, right)?;
                if left.is_sym() || right.is_sym() {
                    return sym::binary(self, frame, *op, &left, &right);
                }
                value::binary(*op, &left, &right, *span)
            }
            Expr::Logic { and, left, right } => {
                let left = self.eval(frame, left)?;
                match truth(frame, &left)? {
                    None => Ok(Val::poison()),
                    Some(x) if x != *and => Ok(Val::Bool(x)),
                    Some(_) => {
                        let right = self.eval(frame, right)?;
                        Ok(match truth(frame, &right)? {
                            Some(x) => Val::Bool(x),
                            None => Val::poison(),
                        })
                    }
                }
            }
            Expr::Call { name, args, .. } => {
                let values = args
                    .iter()
                    .map(|arg| self.eval(frame, arg))
                    .collect::<Result<Vec<_>, _>>()?;
                if name.starts_with("filled_") {
                    self.allocate(match values.first() {
                        Some(Val::U32(n)) => Some(u64::from(*n)),
                        _ => None,
                    })?;
                }
                if values.iter().any(Val::is_sym) {
                    return sym::call(self, frame, name, &values);
                }
                if let Some(result) = value::intrinsic(name, &values) {
                    return result;
                }
                let index = *self
                    .names
                    .get(name)
                    .ok_or_else(|| Stop::internal("unknown callee"))?;
                if frame
                    .path
                    .as_ref()
                    .is_some_and(|node| node.function == index)
                {
                    return Err(Stop::violation("self call outside a let or return"));
                }
                self.call(index, values)
            }
        }
    }
}

/// Read `target[index]` outside any lane.
pub(super) fn read_base(
    target: &value::Array,
    index: usize,
    span: crate::front::Span,
) -> Result<Val, Stop> {
    target
        .borrow()
        .get(index)
        .ok_or(Stop::Trap(TrapCategory::IndexOutOfBounds, span))
}

/// Identity of the array a local names (for lane overlays).
pub(super) fn local_array_id(frame: &Frame, name: &str) -> Option<usize> {
    match frame.vars.get(name) {
        Some(Val::Arr(array)) => Some(array_id(array)),
        _ => None,
    }
}
