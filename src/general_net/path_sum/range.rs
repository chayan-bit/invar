//! Interval range inference for P4a narrow licences (P4a profile item 1).
//!
//! Given a box of root parameter ranges (the licence), the analysis proves
//! that every operator demanded by a path-sum lane stays inside its source
//! type, so the lane may run unchecked 32-bit arithmetic: no source trap can
//! occur under the licence. Interval-owned parameters keep the licence range at
//! every node because the lane validates the split geometry at run time; the
//! other root parameters, nested recursion levels and loops are fixpoints. A
//! failure (possible overflow, an unbounded loop or recursion, arrays) means no
//! licence, never an unchecked kernel.

use std::collections::{BTreeMap, HashMap};

use crate::{
    front::{BinaryOp, Span, Type, UnaryOp},
    general_net::regions::tree::{Expr, FunctionTree, LoopKind, Stmt},
};

/// Name and version of this analysis in annotation provenance.
pub const NAME: &str = "p4a-interval";
pub const VERSION: u32 = 1;

/// Fixpoint rounds before a loop or parameter range is declared unbounded.
const ROUNDS: usize = 64;
/// Nested recursion levels before its depth is declared unbounded.
const MAX_LEVELS: usize = 32;
/// Statements interpreted per analysis (a tool budget, not a semantic one).
const BUDGET: usize = 4_000_000;

/// A closed integer interval; empty (bottom) when `lo > hi`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Iv {
    pub lo: i64,
    pub hi: i64,
}

impl Iv {
    pub const BOTTOM: Self = Self { lo: 1, hi: 0 };
    pub const fn new(lo: i64, hi: i64) -> Self {
        Self { lo, hi }
    }
    pub const fn point(x: i64) -> Self {
        Self { lo: x, hi: x }
    }
    pub const fn is_bottom(self) -> bool {
        self.lo > self.hi
    }
    pub fn join(self, other: Self) -> Self {
        match (self.is_bottom(), other.is_bottom()) {
            (true, _) => other,
            (_, true) => self,
            _ => Self::new(self.lo.min(other.lo), self.hi.max(other.hi)),
        }
    }
    fn meet(self, other: Self) -> Self {
        Self::new(self.lo.max(other.lo), self.hi.min(other.hi))
    }
    pub fn within(self, outer: Self) -> bool {
        self.is_bottom() || (!outer.is_bottom() && outer.lo <= self.lo && self.hi <= outer.hi)
    }
}

pub fn type_range(ty: &Type) -> Option<Iv> {
    match ty {
        Type::I32 => Some(Iv::new(i64::from(i32::MIN), i64::from(i32::MAX))),
        Type::U32 => Some(Iv::new(0, i64::from(u32::MAX))),
        Type::Bool => Some(Iv::new(0, 1)),
        _ => None,
    }
}

/// What the lowering asks the analysis to prove.
pub struct Spec<'a> {
    /// Statement trees by program function index (`None`: unstructured).
    pub trees: &'a [Option<&'a FunctionTree>],
    /// The enumerated (lane-replayed) recursion.
    pub root: usize,
    /// Interval-owned `(lo, hi)` parameter indices, validated per lane.
    pub owned: Option<(usize, usize)>,
    /// Range of every root parameter at the root call.
    pub licence: &'a [Iv],
    /// Certified combine returns of the root, keyed by return span.
    pub combines: &'a Combines,
}

/// Coefficients and constant of each certified combine return of the root,
/// keyed by return span (certificate `Fact::Combine`, in `tree::walk` order).
pub type Combines = BTreeMap<(usize, usize), (Vec<(String, i64)>, Expr)>;

/// Everything the analysis proved under one licence.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Facts {
    /// Bounds of every operator site, keyed by `(function, span)`.
    pub ops: BTreeMap<(usize, (usize, usize)), Iv>,
    /// Leaf values the root recursion returns (no self-call holes).
    pub leaves: Option<Iv>,
    /// Combine constants of the root recursion.
    pub constants: Option<Iv>,
    /// Range of every root parameter over all nodes (the fixpoint box).
    pub params: Vec<Iv>,
    /// Nested recursive functions: number of depth levels a lane evaluates.
    pub levels: BTreeMap<usize, usize>,
    /// Most user-call frames a root node pushes below itself.
    pub frames: usize,
}

type Env = BTreeMap<String, Iv>;
type Fail = &'static str;

/// How self calls of the function being interpreted are treated.
#[derive(Clone, Copy)]
enum Mode {
    /// Not recursive.
    Plain,
    /// The enumerated root: self calls are holes, combines are not evaluated.
    Root,
    /// Level `k` of a nested recursion: self calls read level `k + 1`.
    Nested { below: Iv },
}

struct Call {
    function: usize,
    mode: Mode,
    returns: Iv,
    /// Joined argument box of the self calls made at this level.
    children: Option<Vec<Iv>>,
    frames: usize,
}

struct Engine<'a> {
    spec: &'a Spec<'a>,
    facts: Facts,
    active: Vec<usize>,
    memo: HashMap<(usize, Vec<Iv>), (Iv, usize)>,
    nested: HashMap<(usize, Vec<Iv>), (Iv, usize)>,
    budget: usize,
}

/// Prove the lane semantics of `spec` free of source traps.
pub fn analyse(spec: &Spec<'_>) -> Result<Facts, Fail> {
    let tree = tree(spec.trees, spec.root)?;
    if spec.licence.len() != tree.params.len() {
        return Err("licence arity");
    }
    let mut engine = Engine {
        spec,
        facts: Facts::default(),
        active: vec![spec.root],
        memo: HashMap::new(),
        nested: HashMap::new(),
        budget: BUDGET,
    };
    let mut params = spec.licence.to_vec();
    for _ in 0..ROUNDS {
        let mut call = Call::new(spec.root, Mode::Root);
        let env = bind(tree, &params)?;
        engine.block(&mut call, env, &tree.body)?;
        engine.facts.frames = engine.facts.frames.max(call.frames);
        let mut next = params.clone();
        if let Some(children) = &call.children {
            for (i, (slot, child)) in next.iter_mut().zip(children).enumerate() {
                if spec.owned.is_some_and(|(lo, hi)| i == lo || i == hi) {
                    continue;
                }
                *slot = slot.join(*child);
            }
        }
        if next == params {
            engine.facts.params = params;
            return Ok(engine.facts);
        }
        params = next;
    }
    Err("root parameter range does not converge")
}

fn tree<'a>(trees: &[Option<&'a FunctionTree>], index: usize) -> Result<&'a FunctionTree, Fail> {
    trees
        .get(index)
        .copied()
        .flatten()
        .ok_or("unstructured function")
}

fn bind(tree: &FunctionTree, args: &[Iv]) -> Result<Env, Fail> {
    if tree.params.len() != args.len() {
        return Err("call arity");
    }
    let mut env = Env::new();
    for (param, arg) in tree.params.iter().zip(args) {
        let range = type_range(&param.ty).ok_or("non-scalar parameter")?;
        if !arg.within(range) {
            return Err("argument outside its type");
        }
        env.insert(param.name.clone(), *arg);
    }
    Ok(env)
}

fn join_env(a: Option<Env>, b: Option<Env>) -> Option<Env> {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some(mut a), Some(b)) => {
            for (name, value) in b {
                let slot = a.entry(name).or_insert(Iv::BOTTOM);
                *slot = slot.join(value);
            }
            Some(a)
        }
    }
}

pub(super) fn recursive(tree: &FunctionTree) -> bool {
    let mut found = false;
    crate::general_net::regions::tree::walk(&tree.body, &mut Vec::new(), 0, &mut |stmt, _, _| {
        for expr in crate::general_net::regions::tree::stmt_exprs(stmt) {
            let mut calls = Vec::new();
            expr.calls(&mut calls);
            found |= calls
                .iter()
                .any(|c| matches!(c, Expr::Call { name, .. } if *name == tree.name));
        }
    });
    found
}

impl Call {
    fn new(function: usize, mode: Mode) -> Self {
        Self {
            function,
            mode,
            returns: Iv::BOTTOM,
            children: None,
            frames: 0,
        }
    }
    fn child(&mut self, args: Vec<Iv>) {
        self.children = Some(match self.children.take() {
            None => args,
            Some(old) => old.iter().zip(&args).map(|(a, b)| a.join(*b)).collect(),
        });
    }
}

impl<'a> Engine<'a> {
    fn tick(&mut self) -> Result<(), Fail> {
        self.budget = self.budget.checked_sub(1).ok_or("analysis budget")?;
        Ok(())
    }

    fn tree(&self, index: usize) -> Result<&'a FunctionTree, Fail> {
        tree(self.spec.trees, index)
    }

    fn index_of(&self, name: &str) -> Option<usize> {
        self.spec
            .trees
            .iter()
            .position(|t| t.is_some_and(|t| t.name == name))
    }

    fn block(&mut self, call: &mut Call, mut env: Env, body: &[Stmt]) -> Result<Option<Env>, Fail> {
        for stmt in body {
            self.tick()?;
            match stmt {
                Stmt::Let { name, value } | Stmt::Assign { name, value } => {
                    let v = self.bound(call, &env, value)?;
                    env.insert(name.clone(), v);
                }
                Stmt::Eval(value) => {
                    self.eval(call, &env, value)?;
                }
                Stmt::Store { .. } => return Err("array store"),
                Stmt::If { cond, yes, no } => {
                    let c = self.eval(call, &env, cond)?;
                    let a = match (c != Iv::point(0))
                        .then(|| refine(&env, cond, true))
                        .flatten()
                    {
                        Some(e) => self.block(call, e, yes)?,
                        None => None,
                    };
                    let b = match (c != Iv::point(1))
                        .then(|| refine(&env, cond, false))
                        .flatten()
                    {
                        Some(e) => self.block(call, e, no)?,
                        None => None,
                    };
                    match join_env(a, b) {
                        Some(e) => env = e,
                        None => return Ok(None),
                    }
                }
                Stmt::Loop(inner) => match self.looped(call, env, inner)? {
                    Some(e) => env = e,
                    None => return Ok(None),
                },
                Stmt::Return { value, span } => {
                    let value = value.as_ref().ok_or("return without value")?;
                    self.ret(call, &env, value, *span)?;
                    return Ok(None);
                }
            }
        }
        Ok(Some(env))
    }

    fn looped(
        &mut self,
        call: &mut Call,
        env: Env,
        inner: &crate::general_net::regions::tree::Loop,
    ) -> Result<Option<Env>, Fail> {
        let (cond, mut head) = match &inner.kind {
            LoopKind::While { cond } => (Some(cond), env),
            LoopKind::For { var, lo, hi } => {
                let lo = self.eval(call, &env, lo)?;
                let hi = self.eval(call, &env, hi)?;
                let mut env = env;
                env.insert(var.clone(), Iv::new(lo.lo, hi.hi.max(lo.lo)));
                (None, env)
            }
        };
        for _ in 0..ROUNDS {
            let entry = match cond {
                Some(cond) => {
                    let c = self.eval(call, &head, cond)?;
                    if c == Iv::point(0) {
                        None
                    } else {
                        refine(&head, cond, true)
                    }
                }
                None => Some(head.clone()),
            };
            let Some(entry) = entry else {
                return Ok(Some(head));
            };
            let out = self.block(call, entry, &inner.body)?;
            let next = join_env(Some(head.clone()), out).ok_or("loop state")?;
            if next == head {
                return Ok(match cond {
                    Some(cond) if self.eval(call, &head, cond)? == Iv::point(1) => None,
                    Some(cond) => refine(&head, cond, false),
                    None => Some(head),
                });
            }
            head = next;
        }
        Err("loop range does not converge")
    }

    /// `let name = value`: a self call binds a hole (root) or a level result.
    fn bound(&mut self, call: &mut Call, env: &Env, value: &Expr) -> Result<Iv, Fail> {
        if matches!(call.mode, Mode::Root) && self.self_call(call, value).is_some() {
            let args = self.args(call, env, value)?;
            call.child(args);
            return Ok(Iv::BOTTOM);
        }
        self.eval(call, env, value)
    }

    fn self_call<'e>(&self, call: &Call, value: &'e Expr) -> Option<&'e [Expr]> {
        let own = &self.tree(call.function).ok()?.name;
        match value {
            Expr::Call { name, args, .. } if name == own => Some(args),
            _ => None,
        }
    }

    fn args(&mut self, call: &mut Call, env: &Env, value: &Expr) -> Result<Vec<Iv>, Fail> {
        let Expr::Call { args, .. } = value else {
            return Err("call shape");
        };
        args.iter().map(|a| self.eval(call, env, a)).collect()
    }

    fn ret(&mut self, call: &mut Call, env: &Env, value: &Expr, span: Span) -> Result<(), Fail> {
        if let Mode::Root = call.mode {
            if self.self_call(call, value).is_some() {
                let args = self.args(call, env, value)?;
                call.child(args);
                return Ok(());
            }
            let spec = self.spec;
            if let Some((_, constant)) = spec.combines.get(&(span.start, span.end)) {
                // The lane never evaluates a combine; its constant term it does.
                let k = self.eval(call, env, constant)?;
                self.facts.constants = Some(self.facts.constants.unwrap_or(Iv::BOTTOM).join(k));
                return Ok(());
            }
            let v = self.eval(call, env, value)?;
            self.facts.leaves = Some(self.facts.leaves.unwrap_or(Iv::BOTTOM).join(v));
            return Ok(());
        }
        let v = self.eval(call, env, value)?;
        call.returns = call.returns.join(v);
        Ok(())
    }

    fn record(&mut self, call: &Call, span: Span, v: Iv) {
        if span.start == span.end {
            return;
        }
        let slot = self
            .facts
            .ops
            .entry((call.function, (span.start, span.end)))
            .or_insert(Iv::BOTTOM);
        *slot = slot.join(v);
    }

    fn ty(&self, call: &Call, expr: &Expr) -> Result<Type, Fail> {
        let tree = self.tree(call.function)?;
        Ok(match expr {
            Expr::I32(_) => Type::I32,
            Expr::U32(_) => Type::U32,
            Expr::Bool(_) | Expr::Logic { .. } => Type::Bool,
            Expr::Local(name) => tree.locals.get(name).cloned().ok_or("unknown local")?,
            Expr::Unary {
                op: UnaryOp::Not, ..
            } => Type::Bool,
            Expr::Unary { value, .. } => self.ty(call, value)?,
            Expr::Binary { op, left, .. } => match op {
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem => {
                    self.ty(call, left)?
                }
                _ => Type::Bool,
            },
            Expr::Call { name, .. } => match name.as_str() {
                "wrapping_add" | "wrapping_mul" => Type::U32,
                _ => {
                    let index = self.index_of(name).ok_or("unknown callee")?;
                    self.tree(index)?.result.clone()
                }
            },
            _ => return Err("non-scalar expression"),
        })
    }

    fn eval(&mut self, call: &mut Call, env: &Env, expr: &Expr) -> Result<Iv, Fail> {
        Ok(match expr {
            Expr::I32(x) => Iv::point(i64::from(*x)),
            Expr::U32(x) => Iv::point(i64::from(*x)),
            Expr::Bool(b) => Iv::point(i64::from(*b)),
            Expr::Local(name) => *env.get(name).ok_or("unbound local")?,
            Expr::Unary { op, value, span } => {
                let v = self.eval(call, env, value)?;
                let r = match op {
                    UnaryOp::Not => not(v),
                    UnaryOp::Neg => Iv::new(-v.hi, -v.lo),
                };
                let r = if v.is_bottom() { Iv::BOTTOM } else { r };
                self.checked(call, expr, *span, r)?
            }
            Expr::Binary {
                op,
                left,
                right,
                span,
            } => {
                let a = self.eval(call, env, left)?;
                let b = self.eval(call, env, right)?;
                let r = binary(*op, a, b)?;
                self.checked(call, expr, *span, r)?
            }
            Expr::Logic { and, left, right } => {
                let l = self.eval(call, env, left)?;
                if l == Iv::point(i64::from(!*and)) {
                    return Ok(l);
                }
                match refine(env, left, *and) {
                    Some(inner) => {
                        self.eval(call, &inner, right)?;
                    }
                    None => return Ok(l),
                }
                Iv::new(0, 1)
            }
            Expr::Call { name, args, .. } => {
                let values = args
                    .iter()
                    .map(|a| self.eval(call, env, a))
                    .collect::<Result<Vec<_>, _>>()?;
                self.invoke(call, name, values)?
            }
            _ => return Err("arrays or owners"),
        })
    }

    /// Record an operator's bounds and refuse any result outside its type.
    fn checked(&mut self, call: &Call, expr: &Expr, span: Span, r: Iv) -> Result<Iv, Fail> {
        if r.is_bottom() {
            return Ok(r);
        }
        let range = type_range(&self.ty(call, expr)?).ok_or("non-scalar operator")?;
        if !r.within(range) {
            return Err("operator may leave its type");
        }
        self.record(call, span, r);
        Ok(r)
    }

    fn invoke(&mut self, call: &mut Call, name: &str, args: Vec<Iv>) -> Result<Iv, Fail> {
        if matches!(name, "wrapping_add" | "wrapping_mul") {
            return Ok(Iv::new(0, i64::from(u32::MAX)));
        }
        if args.iter().any(|a| a.is_bottom()) {
            return Ok(Iv::BOTTOM);
        }
        let index = self.index_of(name).ok_or("unknown callee")?;
        if index == call.function {
            return match call.mode {
                Mode::Nested { below } => {
                    call.child(args);
                    Ok(below)
                }
                Mode::Root => Err("self call outside a let or return"),
                Mode::Plain => Err("unexpected self call"),
            };
        }
        if self.active.contains(&index) {
            return Err("mutual recursion");
        }
        let tree = self.tree(index)?;
        let (result, frames) = if recursive(tree) {
            self.nested_call(index, args)?
        } else {
            self.plain_call(index, args)?
        };
        call.frames = call.frames.max(frames);
        Ok(result)
    }

    fn plain_call(&mut self, index: usize, args: Vec<Iv>) -> Result<(Iv, usize), Fail> {
        let key = (index, args);
        if let Some(hit) = self.memo.get(&key) {
            return Ok(*hit);
        }
        let tree = self.tree(index)?;
        let env = bind(tree, &key.1)?;
        let body = &tree.body;
        let mut call = Call::new(index, Mode::Plain);
        self.active.push(index);
        let done = self.block(&mut call, env, body);
        self.active.pop();
        if done?.is_some() {
            return Err("function falls off its end");
        }
        let out = (call.returns, call.frames + 1);
        self.memo.insert(key, out);
        Ok(out)
    }

    /// Depth levels of a nested recursion, joined per level, to a fixpoint.
    fn nested_call(&mut self, index: usize, args: Vec<Iv>) -> Result<(Iv, usize), Fail> {
        let key = (index, args);
        if let Some(hit) = self.nested.get(&key) {
            return Ok(*hit);
        }
        let tree = self.tree(index)?;
        let mut boxes = vec![key.1.clone()];
        let mut results = vec![Iv::BOTTOM];
        let mut frames = vec![0usize];
        self.active.push(index);
        let mut stable = false;
        for _ in 0..ROUNDS {
            let mut changed = false;
            let mut k = 0;
            while k < boxes.len() {
                let below = results.get(k + 1).copied().unwrap_or(Iv::BOTTOM);
                let mut call = Call::new(index, Mode::Nested { below });
                let env = match bind(tree, &boxes[k]) {
                    Ok(env) => env,
                    Err(e) => {
                        self.active.pop();
                        return Err(e);
                    }
                };
                let done = self.block(&mut call, env, &tree.body);
                match done {
                    Ok(None) => {}
                    Ok(Some(_)) => {
                        self.active.pop();
                        return Err("function falls off its end");
                    }
                    Err(e) => {
                        self.active.pop();
                        return Err(e);
                    }
                }
                let r = results[k].join(call.returns);
                changed |= r != results[k];
                results[k] = r;
                let deeper = frames[k].max(call.frames);
                changed |= deeper != frames[k];
                frames[k] = deeper;
                if let Some(children) = call.children {
                    if k + 1 == boxes.len() {
                        if boxes.len() == MAX_LEVELS {
                            self.active.pop();
                            return Err("nested recursion depth is not bounded");
                        }
                        boxes.push(children);
                        results.push(Iv::BOTTOM);
                        frames.push(0);
                        changed = true;
                    } else {
                        let joined: Vec<Iv> = boxes[k + 1]
                            .iter()
                            .zip(&children)
                            .map(|(a, b)| a.join(*b))
                            .collect();
                        changed |= joined != boxes[k + 1];
                        boxes[k + 1] = joined;
                    }
                }
                k += 1;
            }
            if !changed {
                stable = true;
                break;
            }
        }
        self.active.pop();
        if !stable {
            return Err("nested recursion does not converge");
        }
        let levels = boxes.len();
        let slot = self.facts.levels.entry(index).or_insert(0);
        *slot = (*slot).max(levels);
        let depth = frames
            .iter()
            .enumerate()
            .map(|(k, f)| k + 1 + f)
            .max()
            .unwrap_or(1);
        let out = (results[0], depth);
        self.nested.insert(key, out);
        Ok(out)
    }
}

fn not(v: Iv) -> Iv {
    match (v.lo, v.hi) {
        (1, 1) => Iv::point(0),
        (0, 0) => Iv::point(1),
        _ => Iv::new(0, 1),
    }
}

fn corners(a: Iv, b: Iv, f: impl Fn(i64, i64) -> i64) -> Iv {
    let xs = [f(a.lo, b.lo), f(a.lo, b.hi), f(a.hi, b.lo), f(a.hi, b.hi)];
    Iv::new(
        *xs.iter().min().unwrap_or(&0),
        *xs.iter().max().unwrap_or(&0),
    )
}

fn compare(op: BinaryOp, a: Iv, b: Iv) -> Iv {
    let (always, never) = match op {
        BinaryOp::Lt => (a.hi < b.lo, a.lo >= b.hi),
        BinaryOp::Le => (a.hi <= b.lo, a.lo > b.hi),
        BinaryOp::Gt => (a.lo > b.hi, a.hi <= b.lo),
        BinaryOp::Ge => (a.lo >= b.hi, a.hi < b.lo),
        BinaryOp::Eq => (
            a.lo == a.hi && b.lo == b.hi && a.lo == b.lo,
            a.hi < b.lo || b.hi < a.lo,
        ),
        BinaryOp::Ne => (
            a.hi < b.lo || b.hi < a.lo,
            a.lo == a.hi && b.lo == b.hi && a.lo == b.lo,
        ),
        _ => (false, false),
    };
    if always {
        Iv::point(1)
    } else if never {
        Iv::point(0)
    } else {
        Iv::new(0, 1)
    }
}

/// Interval of a checked binary operator; a divisor range containing zero, or
/// `MIN / -1` and `MIN % -1`, is refused (the lane would need a trap check).
fn binary(op: BinaryOp, a: Iv, b: Iv) -> Result<Iv, Fail> {
    if a.is_bottom() || b.is_bottom() {
        return Ok(Iv::BOTTOM);
    }
    Ok(match op {
        BinaryOp::Add => corners(a, b, |x, y| x + y),
        BinaryOp::Sub => corners(a, b, |x, y| x - y),
        BinaryOp::Mul => corners(a, b, |x, y| x * y),
        BinaryOp::Div | BinaryOp::Rem => {
            if b.lo <= 0 && 0 <= b.hi {
                return Err("divisor range contains zero");
            }
            if a.lo <= i64::from(i32::MIN) && b.lo <= -1 && -1 <= b.hi {
                return Err("MIN by -1");
            }
            if op == BinaryOp::Div {
                corners(a, b, |x, y| x / y)
            } else {
                let m = b.lo.abs().max(b.hi.abs()) - 1;
                Iv::new(
                    if a.lo >= 0 { 0 } else { a.lo.max(-m) },
                    if a.hi <= 0 { 0 } else { a.hi.min(m) },
                )
            }
        }
        BinaryOp::Or | BinaryOp::And => Iv::new(0, 1),
        _ => compare(op, a, b),
    })
}

/// Narrow `env` by `cond == polarity`; `None` when that is infeasible.
fn refine(env: &Env, cond: &Expr, polarity: bool) -> Option<Env> {
    match cond {
        Expr::Unary {
            op: UnaryOp::Not,
            value,
            ..
        } => refine(env, value, !polarity),
        Expr::Logic { and, left, right } if *and == polarity => {
            let env = refine(env, left, polarity)?;
            refine(&env, right, polarity)
        }
        Expr::Binary {
            op, left, right, ..
        } => {
            let op = if polarity { *op } else { negate(*op)? };
            let mut out = env.clone();
            let value = |e: &Expr| quick(env, e);
            if let (Expr::Local(x), Some(r)) = (left.as_ref(), value(right)) {
                let v = out.get(x)?.meet(constraint(op, r));
                if v.is_bottom() {
                    return None;
                }
                out.insert(x.clone(), v);
            }
            if let (Some(l), Expr::Local(y)) = (value(left), right.as_ref()) {
                let v = out.get(y)?.meet(constraint(flip(op)?, l));
                if v.is_bottom() {
                    return None;
                }
                out.insert(y.clone(), v);
            }
            Some(out)
        }
        _ => Some(env.clone()),
    }
}

/// Interval of a call-free arithmetic expression (for branch refinement).
fn quick(env: &Env, expr: &Expr) -> Option<Iv> {
    match expr {
        Expr::Local(n) => env.get(n).copied(),
        Expr::Unary {
            op: UnaryOp::Neg,
            value,
            ..
        } => quick(env, value).map(|v| Iv::new(-v.hi, -v.lo)),
        Expr::Binary {
            op: op @ (BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem),
            left,
            right,
            ..
        } => binary(*op, quick(env, left)?, quick(env, right)?).ok(),
        _ => expr.literal().map(Iv::point),
    }
}

fn negate(op: BinaryOp) -> Option<BinaryOp> {
    Some(match op {
        BinaryOp::Lt => BinaryOp::Ge,
        BinaryOp::Le => BinaryOp::Gt,
        BinaryOp::Gt => BinaryOp::Le,
        BinaryOp::Ge => BinaryOp::Lt,
        BinaryOp::Eq => BinaryOp::Ne,
        BinaryOp::Ne => BinaryOp::Eq,
        _ => return None,
    })
}

fn flip(op: BinaryOp) -> Option<BinaryOp> {
    Some(match op {
        BinaryOp::Lt => BinaryOp::Gt,
        BinaryOp::Le => BinaryOp::Ge,
        BinaryOp::Gt => BinaryOp::Lt,
        BinaryOp::Ge => BinaryOp::Le,
        BinaryOp::Eq | BinaryOp::Ne => op,
        _ => return None,
    })
}

/// Values `x` with `x op r` possible for some `r` in the interval.
fn constraint(op: BinaryOp, r: Iv) -> Iv {
    const LOW: i64 = i64::MIN / 4;
    const HIGH: i64 = i64::MAX / 4;
    match op {
        BinaryOp::Lt => Iv::new(LOW, r.hi - 1),
        BinaryOp::Le => Iv::new(LOW, r.hi),
        BinaryOp::Gt => Iv::new(r.lo + 1, HIGH),
        BinaryOp::Ge => Iv::new(r.lo, HIGH),
        BinaryOp::Eq => r,
        _ => Iv::new(LOW, HIGH),
    }
}
