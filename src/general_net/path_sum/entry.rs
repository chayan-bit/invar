//! The entry that tail-calls a certified recursion, evaluated exactly on the
//! host: its straight-line prologue and the root call's arguments use checked
//! arithmetic, and any trap (or anything else) sends the call to the exact
//! fallback, which then owns the source-order trap.

use std::collections::BTreeMap;

use super::Refusal;
use crate::{
    eval::Value,
    front::{BinaryOp, Type, UnaryOp},
    general_net::regions::{
        Partition,
        tree::{Expr, Stmt},
    },
};

#[derive(Clone, Debug)]
pub struct Entry {
    /// The entry function (the root itself when `direct`).
    pub function: usize,
    pub name: String,
    pub root: usize,
    pub direct: bool,
    params: Vec<(String, Type)>,
    prologue: Vec<Stmt>,
    args: Vec<Expr>,
    types: BTreeMap<String, Type>,
}

impl Entry {
    /// `entry` is the recursion itself, or `let`s followed by `return root(..)`
    /// where `certified(root)` holds.
    pub fn find(
        partition: &Partition,
        entry: &str,
        certified: impl Fn(usize) -> bool,
    ) -> Result<Self, Refusal> {
        let function = partition
            .functions
            .iter()
            .find(|f| f.name == entry)
            .ok_or(Refusal::Shape("unknown entry"))?;
        let tree = function
            .tree
            .as_ref()
            .map_err(|_| Refusal::Shape("unstructured entry"))?;
        let params: Vec<_> = tree
            .params
            .iter()
            .map(|p| (p.name.clone(), p.ty.clone()))
            .collect();
        if params
            .iter()
            .any(|(_, ty)| !matches!(ty, Type::I32 | Type::U32))
        {
            return Err(Refusal::Shape("entry parameters must be i32 or u32"));
        }
        if certified(function.index) {
            return Ok(Self {
                function: function.index,
                name: function.name.clone(),
                root: function.index,
                direct: true,
                args: params.iter().map(|(n, _)| Expr::Local(n.clone())).collect(),
                params,
                prologue: Vec::new(),
                types: tree.locals.clone(),
            });
        }
        let (last, prologue) = tree
            .body
            .split_last()
            .ok_or(Refusal::Shape("empty entry"))?;
        let Stmt::Return {
            value: Some(Expr::Call { name, args, .. }),
            ..
        } = last
        else {
            return Err(Refusal::Shape("entry must return the root call"));
        };
        if prologue
            .iter()
            .any(|s| !matches!(s, Stmt::Let { .. } | Stmt::Assign { .. }))
        {
            return Err(Refusal::Shape("entry prologue must be straight-line"));
        }
        let root = partition
            .functions
            .iter()
            .find(|f| f.name == *name)
            .map(|f| f.index)
            .filter(|i| certified(*i))
            .ok_or(Refusal::Shape("the root call has no L9/L10 law"))?;
        Ok(Self {
            function: function.index,
            name: function.name.clone(),
            root,
            direct: false,
            params,
            prologue: prologue.to_vec(),
            args: args.clone(),
            types: tree.locals.clone(),
        })
    }

    /// The root call's arguments, or `None` when the exact path must decide
    /// (a prologue trap, an out-of-type argument or an unsupported form).
    pub fn root_args(&self, args: &[i64]) -> Option<Vec<i64>> {
        if args.len() != self.params.len() {
            return None;
        }
        // A few names per entry: a vector beats a map on the per-call host path;
        // the latest binding of a name wins.
        let mut env: Vec<(&str, i64)> = Vec::with_capacity(self.params.len() + self.prologue.len());
        for ((name, ty), value) in self.params.iter().zip(args) {
            fits(ty, *value)?;
            env.push((name.as_str(), *value));
        }
        for stmt in &self.prologue {
            let (Stmt::Let { name, value } | Stmt::Assign { name, value }) = stmt else {
                return None;
            };
            let v = self.eval(&env, value)?;
            env.push((name.as_str(), v));
        }
        self.args.iter().map(|a| self.eval(&env, a)).collect()
    }

    /// Oracle values of entry arguments, if each fits its parameter type.
    pub fn values(&self, args: &[i64]) -> Option<Vec<Value>> {
        if args.len() != self.params.len() {
            return None;
        }
        self.params
            .iter()
            .zip(args)
            .map(|((_, ty), v)| match ty {
                Type::I32 => i32::try_from(*v).ok().map(Value::I32),
                Type::U32 => u32::try_from(*v).ok().map(Value::U32),
                _ => None,
            })
            .collect()
    }

    fn ty(&self, expr: &Expr) -> Option<Type> {
        Some(match expr {
            Expr::I32(_) => Type::I32,
            Expr::U32(_) => Type::U32,
            Expr::Local(n) => self.types.get(n)?.clone(),
            Expr::Unary { value, .. } | Expr::Binary { left: value, .. } => self.ty(value)?,
            _ => return None,
        })
    }

    fn eval(&self, env: &[(&str, i64)], expr: &Expr) -> Option<i64> {
        let v = match expr {
            Expr::I32(x) => i64::from(*x),
            Expr::U32(x) => i64::from(*x),
            Expr::Local(n) => env.iter().rev().find(|(k, _)| *k == n.as_str())?.1,
            Expr::Unary {
                op: UnaryOp::Neg,
                value,
                ..
            } => self.eval(env, value)?.checked_neg()?,
            Expr::Binary {
                op, left, right, ..
            } => {
                let (a, b) = (self.eval(env, left)?, self.eval(env, right)?);
                match op {
                    // Operands fit i32/u32, but u32 * u32 can pass i64::MAX.
                    BinaryOp::Add => a.checked_add(b)?,
                    BinaryOp::Sub => a.checked_sub(b)?,
                    BinaryOp::Mul => a.checked_mul(b)?,
                    BinaryOp::Div => a.checked_div(b).filter(|_| b != 0)?,
                    BinaryOp::Rem => a.checked_rem(b).filter(|_| b != 0)?,
                    _ => return None,
                }
            }
            _ => return None,
        };
        fits(&self.ty(expr)?, v)
    }
}

/// `value` when it lies inside `ty` (a checked operator's result).
fn fits(ty: &Type, value: i64) -> Option<i64> {
    let ok = match ty {
        Type::I32 => i32::try_from(value).is_ok(),
        Type::U32 => u32::try_from(value).is_ok(),
        _ => false,
    };
    ok.then_some(value)
}
