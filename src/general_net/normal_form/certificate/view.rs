//! The checker's own view of a loop region, re-derived from the statement
//! tree: the counted shape, body-declared names, carried updates (with
//! body-local definitions substituted in program order, never across a
//! reassignment of a name they read), stores, indexed reads and effects.

use std::collections::{BTreeMap, BTreeSet};

use super::Rejection;
use crate::general_net::regions::{
    self, Partition, RegionId,
    tree::{Expr, FunctionTree, Loop, LoopKind, Stmt},
};

/// Guards of a site, outermost first: `(condition, polarity)`. A condition is
/// `None` when it reads a body local whose value cannot be expressed.
pub(super) type Guards = Vec<(Option<Expr>, bool)>;

/// One assignment to a carried variable, in program order. Values may read
/// earlier versions of carried names (through definitions captured before a
/// reassignment): closed-composition laws (sums, products, affine and matrix
/// maps) are insensitive to which version is read, comparisons are not, so
/// select shapes also require [`Update::exact`].
pub(super) struct Update {
    /// The assigned value with body-local definitions substituted; `None`
    /// when it reads a carried-dependent local that cannot be expressed.
    pub value: Option<Expr>,
    pub guards: Guards,
    /// Identity of the enclosing `if` arms and loops, outermost first.
    pub scope: Vec<(usize, bool)>,
    /// The value and guards read only current versions of every name (no
    /// definition captured before a reassignment of a name it reads).
    pub exact: bool,
}

/// One store to an array declared outside the loop (raw expressions).
pub(super) struct Store<'a> {
    pub index: &'a Expr,
    pub value: &'a Expr,
    pub guards: Vec<&'a Expr>,
    pub depth: usize,
}

pub(super) struct View<'a> {
    pub tree: &'a FunctionTree,
    pub id: RegionId,
    pub induction: String,
    /// `(lo, hi)` of a `for` loop.
    pub bounds: Option<(&'a Expr, &'a Expr)>,
    /// The body without a counted `while`'s final increment.
    pub body: &'a [Stmt],
    pub declared: BTreeSet<String>,
    /// Nested `for` loops: `var -> (lo, hi)`.
    pub nested: BTreeMap<String, Vec<(&'a Expr, &'a Expr)>>,
    /// Every name assigned anywhere in the body.
    pub assigned: BTreeSet<String>,
    pub updates: BTreeMap<String, Vec<Update>>,
    pub stores: BTreeMap<String, Vec<Store<'a>>>,
    pub reads: Vec<(&'a str, &'a Expr)>,
    /// Owners named as values (call arguments, moves) in the body.
    pub owners: BTreeSet<String>,
    pub returns: bool,
}

impl<'a> View<'a> {
    pub fn new(tree: &'a FunctionTree, inner: &'a Loop) -> Result<Self, Rejection> {
        let (induction, bounds, body) = match &inner.kind {
            LoopKind::For { var, lo, hi } => (var.clone(), Some((lo, hi)), inner.body.as_slice()),
            LoopKind::While { cond } => {
                let induction = regions::counted_while(cond, &inner.body)
                    .ok_or(Rejection::Law("uncounted loop"))?;
                let (_, rest) = inner
                    .body
                    .split_last()
                    .ok_or(Rejection::Law("uncounted loop"))?;
                (induction.to_owned(), None, rest)
            }
        };
        let mut view = Self {
            tree,
            id: RegionId {
                function: tree.index,
                head: Some(inner.head),
            },
            induction,
            bounds,
            body,
            declared: BTreeSet::new(),
            nested: BTreeMap::new(),
            assigned: BTreeSet::new(),
            updates: BTreeMap::new(),
            stores: BTreeMap::new(),
            reads: Vec::new(),
            owners: BTreeSet::new(),
            returns: false,
        };
        view.scan(body, &mut Vec::new(), 0);
        let declared = view.declared.clone();
        view.stores.retain(|array, _| !declared.contains(array));
        let carried: BTreeSet<String> = view.carried().into_iter().collect();
        let tainted = taint(body, &view.declared, &carried);
        let mut walker = Walker {
            carried: &carried,
            tainted: &tainted,
            updates: BTreeMap::new(),
            scope: Vec::new(),
            exact_guards: Vec::new(),
        };
        walker.list(body, &mut Env::new(), &mut Vec::new());
        view.updates = walker.updates;
        Ok(view)
    }

    fn scan(&mut self, list: &'a [Stmt], guards: &mut Vec<&'a Expr>, depth: usize) {
        for stmt in list {
            for expr in regions::tree::stmt_exprs(stmt) {
                reads_of(expr, &mut self.reads, &mut self.owners);
            }
            match stmt {
                Stmt::Let { name, .. } => {
                    self.declared.insert(name.clone());
                }
                Stmt::Assign { name, .. } => {
                    self.assigned.insert(name.clone());
                }
                Stmt::Store {
                    array,
                    index,
                    value,
                    ..
                } => self.stores.entry(array.clone()).or_default().push(Store {
                    index,
                    value,
                    guards: guards.clone(),
                    depth,
                }),
                Stmt::If { cond, yes, no } => {
                    guards.push(cond);
                    self.scan(yes, guards, depth);
                    self.scan(no, guards, depth);
                    guards.pop();
                }
                Stmt::Loop(nested) => {
                    if let LoopKind::For { var, lo, hi } = &nested.kind {
                        self.declared.insert(var.clone());
                        self.nested.entry(var.clone()).or_default().push((lo, hi));
                    }
                    self.scan(&nested.body, guards, depth + 1);
                }
                Stmt::Return { .. } => self.returns = true,
                Stmt::Eval(_) => {}
            }
        }
    }

    /// Scalars declared outside the loop and assigned in it.
    pub fn carried(&self) -> Vec<String> {
        self.assigned
            .iter()
            .filter(|name| !self.declared.contains(*name) && **name != self.induction)
            .cloned()
            .collect()
    }

    /// Names whose value differs between iterations.
    pub fn varying(&self) -> BTreeSet<String> {
        let mut out = self.declared.clone();
        out.extend(self.assigned.iter().cloned());
        out.insert(self.induction.clone());
        out
    }

    pub fn reads_of(&self, array: &str) -> Vec<&'a Expr> {
        self.reads
            .iter()
            .filter(|(a, _)| *a == array)
            .map(|(_, ix)| *ix)
            .collect()
    }
}

fn reads_of<'a>(expr: &'a Expr, out: &mut Vec<(&'a str, &'a Expr)>, owners: &mut BTreeSet<String>) {
    match expr {
        Expr::Index { array, index, .. } => {
            out.push((array, index));
            reads_of(index, out, owners);
        }
        Expr::Owner { name, .. } | Expr::Move(name) => {
            owners.insert(name.clone());
        }
        Expr::ArrayLit { items, .. } | Expr::Call { args: items, .. } => {
            items.iter().for_each(|item| reads_of(item, out, owners));
        }
        Expr::Unary { value, .. } => reads_of(value, out, owners),
        Expr::Binary { left, right, .. } | Expr::Logic { left, right, .. } => {
            reads_of(left, out, owners);
            reads_of(right, out, owners);
        }
        _ => {}
    }
}

pub(super) fn names(expr: &Expr) -> BTreeSet<String> {
    let mut out = Vec::new();
    expr.names(&mut out);
    out.into_iter().collect()
}

/// Body-declared names whose value may depend on carried state, through data
/// or through the conditions of their assignments.
fn taint(
    body: &[Stmt],
    declared: &BTreeSet<String>,
    carried: &BTreeSet<String>,
) -> BTreeSet<String> {
    let mut tainted = BTreeSet::new();
    let mut changed = true;
    while changed {
        changed = false;
        regions::tree::walk(body, &mut Vec::new(), 0, &mut |stmt, guards, _| {
            let (name, sources) = match stmt {
                Stmt::Let { name, value } => (name, names(value)),
                Stmt::Assign { name, value } => {
                    let mut sources = names(value);
                    guards.iter().for_each(|(g, _)| sources.extend(names(g)));
                    (name, sources)
                }
                _ => return,
            };
            if declared.contains(name)
                && !tainted.contains(name)
                && sources
                    .iter()
                    .any(|s| carried.contains(s) || tainted.contains(s))
            {
                tainted.insert(name.clone());
                changed = true;
            }
        });
    }
    tainted
}

/// Definitions of body locals at a program point. `Some((expr, stale))` is
/// substitutable; `stale` marks a definition reading a name reassigned since,
/// so it denotes an earlier version of that name. `None` is not expressible
/// (an atom if untainted, unusable if tainted).
type Env = BTreeMap<String, Option<(Expr, bool)>>;

struct Walker<'w> {
    carried: &'w BTreeSet<String>,
    tainted: &'w BTreeSet<String>,
    updates: BTreeMap<String, Vec<Update>>,
    scope: Vec<(usize, bool)>,
    exact_guards: Vec<bool>,
}

impl Walker<'_> {
    fn list(&mut self, list: &[Stmt], env: &mut Env, guards: &mut Guards) {
        for stmt in list {
            match stmt {
                Stmt::Let { name, value } => {
                    let expanded = self.expand(value, env);
                    assign(env, name, expanded);
                }
                Stmt::Assign { name, value } => {
                    let expanded = self.expand(value, env);
                    if self.carried.contains(name) {
                        let exact = expanded.as_ref().is_some_and(|(_, stale)| !stale)
                            && self.exact_guards.iter().all(|e| *e);
                        self.updates.entry(name.clone()).or_default().push(Update {
                            value: expanded.map(|(e, _)| e),
                            guards: guards.clone(),
                            scope: self.scope.clone(),
                            exact,
                        });
                        invalidate(env, name);
                    } else {
                        assign(env, name, expanded);
                    }
                }
                Stmt::If { cond, yes, no } => self.branch(stmt, cond, yes, no, env, guards),
                Stmt::Loop(nested) => self.nested(stmt, nested, env, guards),
                Stmt::Store { .. } | Stmt::Eval(_) | Stmt::Return { .. } => {}
            }
        }
    }

    fn branch(
        &mut self,
        stmt: &Stmt,
        cond: &Expr,
        yes: &[Stmt],
        no: &[Stmt],
        env: &mut Env,
        guards: &mut Guards,
    ) {
        let id = std::ptr::from_ref(stmt).addr();
        let cond = self.expand(cond, env);
        self.exact_guards
            .push(cond.as_ref().is_some_and(|(_, stale)| !stale));
        let cond = cond.map(|(e, _)| e);
        let mut arms = Vec::with_capacity(2);
        for (arm, polarity) in [(yes, true), (no, false)] {
            let mut inner = env.clone();
            guards.push((cond.clone(), polarity));
            self.scope.push((id, polarity));
            self.list(arm, &mut inner, guards);
            self.scope.pop();
            guards.pop();
            arms.push(inner);
        }
        self.exact_guards.pop();
        let (Some(right), Some(left)) = (arms.pop(), arms.pop()) else {
            return;
        };
        let names: BTreeSet<String> = left.keys().chain(right.keys()).cloned().collect();
        for name in names {
            let merged = match (left.get(&name), right.get(&name)) {
                (Some(a), Some(b)) if a == b => a.clone(),
                _ => None,
            };
            env.insert(name, merged);
        }
    }

    fn nested(&mut self, stmt: &Stmt, nested: &Loop, env: &mut Env, guards: &mut Guards) {
        let mut assigned = BTreeSet::new();
        regions::tree::walk(&nested.body, &mut Vec::new(), 0, &mut |s, _, _| {
            if let Stmt::Assign { name, .. } | Stmt::Let { name, .. } = s {
                assigned.insert(name.clone());
            }
        });
        if let LoopKind::For { var, .. } = &nested.kind {
            assigned.insert(var.clone());
        }
        for name in &assigned {
            invalidate(env, name);
            if !self.carried.contains(name) {
                env.insert(name.clone(), None);
            }
        }
        let mut inside = env.clone();
        let cond = match &nested.kind {
            LoopKind::While { cond } => Some(self.expand(cond, &inside)),
            LoopKind::For { .. } => None,
        };
        if let Some(cond) = &cond {
            self.exact_guards
                .push(cond.as_ref().is_some_and(|(_, stale)| !stale));
            guards.push((cond.as_ref().map(|(e, _)| e.clone()), true));
        }
        self.scope.push((std::ptr::from_ref(stmt).addr(), true));
        self.list(&nested.body, &mut inside, guards);
        self.scope.pop();
        if cond.is_some() {
            guards.pop();
            self.exact_guards.pop();
        }
    }

    /// `expr` with definitions inlined, and whether a stale one was used;
    /// `None` when it reads a tainted local that is not expressible here.
    fn expand(&self, expr: &Expr, env: &Env) -> Option<(Expr, bool)> {
        let mut stale = false;
        let out = self.inline(expr, env, &mut stale)?;
        Some((out, stale))
    }

    fn inline(&self, expr: &Expr, env: &Env, stale: &mut bool) -> Option<Expr> {
        let boxed = |e: &Expr, stale: &mut bool| self.inline(e, env, stale).map(Box::new);
        Some(match expr {
            Expr::Local(name) => match env.get(name) {
                Some(Some((definition, old))) => {
                    *stale |= *old;
                    definition.clone()
                }
                Some(None) | None if self.tainted.contains(name) => return None,
                _ => expr.clone(),
            },
            Expr::Index { array, index, span } => Expr::Index {
                array: array.clone(),
                index: boxed(index, stale)?,
                span: *span,
            },
            Expr::Unary { op, value, span } => Expr::Unary {
                op: *op,
                value: boxed(value, stale)?,
                span: *span,
            },
            Expr::Binary {
                op,
                left,
                right,
                span,
            } => Expr::Binary {
                op: *op,
                left: boxed(left, stale)?,
                right: boxed(right, stale)?,
                span: *span,
            },
            Expr::Logic { and, left, right } => Expr::Logic {
                and: *and,
                left: boxed(left, stale)?,
                right: boxed(right, stale)?,
            },
            Expr::Call { name, args, span } => Expr::Call {
                name: name.clone(),
                args: args
                    .iter()
                    .map(|a| self.inline(a, env, stale))
                    .collect::<Option<_>>()?,
                span: *span,
            },
            Expr::ArrayLit { elem, items } => Expr::ArrayLit {
                elem: elem.clone(),
                items: items
                    .iter()
                    .map(|a| self.inline(a, env, stale))
                    .collect::<Option<_>>()?,
            },
            other => other.clone(),
        })
    }
}

/// Definitions reading `name` denote its earlier version once it is reassigned.
fn invalidate(env: &mut Env, name: &str) {
    for (value, stale) in env.values_mut().flatten() {
        if value.mentions(name) {
            *stale = true;
        }
    }
}

fn assign(env: &mut Env, name: &str, value: Option<(Expr, bool)>) {
    invalidate(env, name);
    env.insert(name.to_owned(), value);
}

/// The loop with this region id, and the region itself, in a checked tree.
pub(super) fn loop_of<'t>(
    partition: &Partition,
    tree: &'t FunctionTree,
    id: RegionId,
) -> Result<&'t Loop, Rejection> {
    let head = id.head.ok_or(Rejection::Region)?;
    partition.region(id).ok_or(Rejection::Region)?;
    regions::tree::find_loop(&tree.body, head).ok_or(Rejection::Region)
}
