//! One loop iteration as law-formula terms (#59): the step of every scalar
//! the loop carries, over the pre-iteration values (same names) and the
//! iteration's inputs. This is the net side of a lifting certificate: the
//! checker re-derives these steps from the current net, so a certificate can
//! never state a different loop. It walks the region tree on its own and
//! shares nothing with the extractor in [`super::loops`].
//!
//! The fragment: a counted `for` loop whose body assigns scalars and branches
//! with `if`. An array read at the induction variable is an input named
//! `array[i]`; loop-invariant locals and parameters are free variables.
//! Checked `i32`/`u32` arithmetic is exact over `Z` (Lemma A: before its
//! first trap); `wrapping_add` and `wrapping_mul` on `u32`
//! are `Z/2^32`, whose ordered comparisons are unsigned, so a wrapping loop
//! that orders an `i32` value is refused. A loop mixing checked and wrapping
//! arithmetic, or using anything else, is refused.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    front::{BinaryOp, Type, UnaryOp},
    general_net::regions::{
        Partition,
        tree::{self, Expr, FunctionTree, Loop, LoopKind, Stmt},
    },
    laws_synth::formula::{Cond, Sort, Term},
};

/// The iteration of one loop.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoopSteps {
    pub sort: Sort,
    pub induction: String,
    /// Every scalar declared outside the loop and assigned in it, with its
    /// value after one iteration.
    pub steps: BTreeMap<String, Term>,
    /// The per-iteration inputs the steps read: the induction variable and
    /// array reads at it.
    pub inputs: BTreeSet<String>,
}

/// The steps of the `index`-th loop (source pre-order) of `function`.
pub fn loop_steps(
    partition: &Partition,
    function: &str,
    index: usize,
) -> Result<LoopSteps, String> {
    let tree = partition
        .functions
        .iter()
        .find_map(|f| f.tree.as_ref().ok().filter(|t| t.name == function))
        .ok_or_else(|| format!("no structured function {function}"))?;
    let mut loops = Vec::new();
    tree::for_each_loop(&tree.body, &mut |inner| loops.push(inner));
    let inner = loops
        .get(index)
        .ok_or_else(|| format!("{function} has no loop {index}"))?;
    steps(tree, inner)
}

struct Walk<'a> {
    induction: &'a str,
    locals: &'a BTreeMap<String, Type>,
    /// Names declared inside the body (not carried).
    local: BTreeSet<String>,
    inputs: BTreeSet<String>,
    checked: bool,
    wrapping: bool,
    /// An ordered comparison reads an `i32` value.
    signed: bool,
}

fn steps(tree: &FunctionTree, inner: &Loop) -> Result<LoopSteps, String> {
    let LoopKind::For { var, .. } = &inner.kind else {
        return Err("only counted for loops lift".to_owned());
    };
    let mut walk = Walk {
        induction: var,
        locals: &tree.locals,
        local: BTreeSet::new(),
        inputs: BTreeSet::new(),
        checked: false,
        wrapping: false,
        signed: false,
    };
    let mut env = BTreeMap::new();
    walk.block(&inner.body, &mut env)?;
    let steps: BTreeMap<String, Term> = env
        .into_iter()
        .filter(|(name, _)| !walk.local.contains(name))
        .collect();
    if steps.is_empty() {
        return Err("the loop carries no scalar".to_owned());
    }
    let words = steps
        .keys()
        .all(|name| tree.locals.get(name) == Some(&Type::U32));
    let sort = match (walk.checked, walk.wrapping) {
        (_, false) => Sort::Int,
        (false, true) if words && !walk.signed => Sort::Bv32,
        _ => {
            return Err(
                "checked and wrapping arithmetic mix, wrap a non-u32, or order an i32".to_owned(),
            );
        }
    };
    Ok(LoopSteps {
        sort,
        induction: var.clone(),
        steps,
        inputs: walk.inputs,
    })
}

impl Walk<'_> {
    fn block(&mut self, body: &[Stmt], env: &mut BTreeMap<String, Term>) -> Result<(), String> {
        for stmt in body {
            match stmt {
                Stmt::Let { name, value } => {
                    self.local.insert(name.clone());
                    let value = self.term(value, env)?;
                    env.insert(name.clone(), value);
                }
                Stmt::Assign { name, value } => {
                    if name == self.induction {
                        return Err("the loop assigns its induction variable".to_owned());
                    }
                    let value = self.term(value, env)?;
                    env.insert(name.clone(), value);
                }
                Stmt::If { cond, yes, no } => {
                    // Every merge reads the condition before either branch ran.
                    let before = env.clone();
                    let (mut then, mut other) = (env.clone(), env.clone());
                    self.block(yes, &mut then)?;
                    self.block(no, &mut other)?;
                    let names: BTreeSet<String> =
                        then.keys().chain(other.keys()).cloned().collect();
                    for name in names {
                        let old = || {
                            before
                                .get(&name)
                                .cloned()
                                .unwrap_or_else(|| Term::var(&name))
                        };
                        let y = then.get(&name).cloned().unwrap_or_else(old);
                        let n = other.get(&name).cloned().unwrap_or_else(old);
                        let merged = if y == n {
                            y
                        } else {
                            self.select(cond, y, n, &before)?
                        };
                        env.insert(name, merged);
                    }
                }
                _ => return Err("the body stores, evaluates, nests a loop or returns".to_owned()),
            }
        }
        Ok(())
    }

    /// `if cond { yes } else { no }` as a term.
    fn select(
        &mut self,
        cond: &Expr,
        yes: Term,
        no: Term,
        env: &BTreeMap<String, Term>,
    ) -> Result<Term, String> {
        Ok(match cond {
            Expr::Bool(true) => yes,
            Expr::Bool(false) => no,
            Expr::Unary {
                op: UnaryOp::Not,
                value,
                ..
            } => self.select(value, no, yes, env)?,
            Expr::Logic {
                and: true,
                left,
                right,
            } => {
                let inner = self.select(right, yes, no.clone(), env)?;
                self.select(left, inner, no, env)?
            }
            Expr::Logic {
                and: false,
                left,
                right,
            } => {
                let inner = self.select(right, yes.clone(), no, env)?;
                self.select(left, yes, inner, env)?
            }
            Expr::Binary {
                op, left, right, ..
            } => {
                let (l, r) = (self.term(left, env)?, self.term(right, env)?);
                if !matches!(op, BinaryOp::Eq | BinaryOp::Ne) {
                    self.signed |= self.reads_i32(left) || self.reads_i32(right);
                }
                let cond = match op {
                    BinaryOp::Lt => Cond::Lt(l, r),
                    BinaryOp::Le => Cond::Le(l, r),
                    BinaryOp::Gt => Cond::Lt(r, l),
                    BinaryOp::Ge => Cond::Le(r, l),
                    BinaryOp::Eq => Cond::Eq(l, r),
                    BinaryOp::Ne => Cond::Not(Box::new(Cond::Eq(l, r))),
                    _ => return Err(format!("condition operator {op:?} does not lift")),
                };
                Term::ite(cond, yes, no)
            }
            _ => return Err("a condition outside the fragment".to_owned()),
        })
    }

    /// `expr` reads an `i32` literal, local or array element.
    fn reads_i32(&self, expr: &Expr) -> bool {
        match expr {
            Expr::I32(_) => true,
            Expr::Local(name) => self.locals.get(name) == Some(&Type::I32),
            Expr::Index { array, .. } => self.locals.get(array) == Some(&Type::ArrayI32),
            Expr::Unary { value, .. } => self.reads_i32(value),
            Expr::Binary { left, right, .. } => self.reads_i32(left) || self.reads_i32(right),
            Expr::Call { args, .. } => args.iter().any(|a| self.reads_i32(a)),
            _ => false,
        }
    }

    fn term(&mut self, expr: &Expr, env: &BTreeMap<String, Term>) -> Result<Term, String> {
        Ok(match expr {
            Expr::I32(x) => Term::Lit(i64::from(*x)),
            Expr::U32(x) => Term::Lit(i64::from(*x)),
            Expr::Local(name) => {
                if name == self.induction {
                    self.inputs.insert(name.clone());
                }
                env.get(name).cloned().unwrap_or_else(|| Term::var(name))
            }
            Expr::Index { array, index, .. } => {
                if !matches!(index.as_ref(), Expr::Local(i) if i == self.induction) {
                    return Err(format!("{array} is read away from the induction variable"));
                }
                let name = format!("{array}[{}]", self.induction);
                self.inputs.insert(name.clone());
                Term::Var(name)
            }
            Expr::Unary {
                op: UnaryOp::Neg,
                value,
                ..
            } => {
                self.checked = true;
                -self.term(value, env)?
            }
            Expr::Binary {
                op, left, right, ..
            } => {
                let (l, r) = (self.term(left, env)?, self.term(right, env)?);
                self.checked = true;
                match op {
                    BinaryOp::Add => l + r,
                    BinaryOp::Sub => l - r,
                    BinaryOp::Mul => l * r,
                    _ => return Err(format!("operator {op:?} does not lift")),
                }
            }
            Expr::Call { name, args, .. } if args.len() == 2 => {
                let (l, r) = (self.term(&args[0], env)?, self.term(&args[1], env)?);
                self.wrapping = true;
                match name.as_str() {
                    "wrapping_add" => l + r,
                    "wrapping_mul" => l * r,
                    _ => return Err(format!("call to {name} does not lift")),
                }
            }
            _ => return Err("an expression outside the fragment".to_owned()),
        })
    }
}
