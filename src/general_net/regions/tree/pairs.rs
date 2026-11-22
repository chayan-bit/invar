//! Pairs (#262) in the statement tree (#362). A pair is two 32-bit scalars, so a pair
//! local `p` is tracked as the scalar locals `p.0` and `p.1`: `let p = (a, b)` becomes
//! two lets in evaluation order and `p.0` a read of the local `p.0`. A pair that
//! crosses a call or return boundary, or a pair value used any other way, still has no
//! tree form, so only the function holding it stays serial, as before. A loop that
//! touches a component stays serial ([`in_loop`]).

use super::*;

/// Component `index` of pair local `name`; `.` never occurs in a MIR local name.
pub(super) fn component(name: &str, index: usize) -> String {
    format!("{name}.{index}")
}

/// Whether the loop whose head is `head` reads or writes a pair component in its
/// header or body. Such a loop has no admitted candidate and stays serial (NB.7.13);
/// the components exist only so that the function's other loops keep their form.
pub fn in_loop(body: &[Stmt], head: Option<BlockId>) -> bool {
    let is_component = |name: &String| name.contains('.');
    let mut found = false;
    for_each_loop(body, &mut |inner| {
        if Some(inner.head) != head {
            return;
        }
        let mut names = Vec::new();
        match &inner.kind {
            LoopKind::For { lo, hi, .. } => [lo, hi].iter().for_each(|e| e.names(&mut names)),
            LoopKind::While { cond } => cond.names(&mut names),
        }
        found |= names.iter().any(is_component);
        walk(&inner.body, &mut Vec::new(), 0, &mut |stmt, _, _| {
            let mut names = Vec::new();
            stmt_exprs(stmt)
                .iter()
                .for_each(|expr| expr.names(&mut names));
            if let Stmt::Let { name, .. } | Stmt::Assign { name, .. } = stmt {
                names.push(name.clone());
            }
            found |= names.iter().any(is_component);
        });
    });
    found
}

/// The tree locals standing for MIR local `name`: itself, or its two components.
pub(super) fn scalars(name: &str, ty: &Type) -> Vec<(String, Type)> {
    match ty {
        Type::Pair(pair) => vec![
            (component(name, 0), pair.0.clone()),
            (component(name, 1), pair.1.clone()),
        ],
        _ => vec![(name.to_owned(), ty.clone())],
    }
}

/// Whether a pair crosses the function boundary as a parameter or the result.
pub(super) fn crosses_boundary(function: &Function) -> bool {
    matches!(function.result, Type::Pair(_))
        || function
            .params
            .iter()
            .any(|p| matches!(p.ty, Type::Pair(_)))
}

impl Decompiler<'_> {
    pub(super) fn is_pair(&self, name: &str) -> bool {
        self.function
            .locals
            .get(name)
            .is_some_and(|local| matches!(local.ty, Type::Pair(_)))
    }

    /// `let p = value` (`bind`) or `p = value` for pair local `p`, one statement per
    /// component. A store whose second component reads `p.0` would see the first
    /// component's new value, so it has no tree form.
    pub(super) fn pair_stmts(
        &mut self,
        name: &str,
        value: ValueId,
        bind: bool,
    ) -> Result<[Stmt; 2], Unstructured> {
        let instruction = *self.defs.get(&value).ok_or(Unstructured::Shape)?;
        let [first, second] = match &instruction.kind {
            Operation::Pair(first, second) => [self.expr(*first)?, self.expr(*second)?],
            Operation::Read(source) | Operation::Move(source) if self.is_pair(source) => {
                [0, 1].map(|index| Expr::Local(component(source, index)))
            }
            _ => return Err(Unstructured::Operation),
        };
        let (zero, one) = (component(name, 0), component(name, 1));
        if !bind && second.mentions(&zero) {
            return Err(Unstructured::Operation);
        }
        Ok(if bind {
            [
                Stmt::Let {
                    name: zero,
                    value: first,
                },
                Stmt::Let {
                    name: one,
                    value: second,
                },
            ]
        } else {
            [
                Stmt::Assign {
                    name: zero,
                    value: first,
                },
                Stmt::Assign {
                    name: one,
                    value: second,
                },
            ]
        })
    }

    /// Field `index` of a pair local. A field of any other pair value has no tree
    /// form: dropping the other component would drop its traps.
    pub(super) fn pair_get(&self, pair: ValueId, index: usize) -> Result<Expr, Unstructured> {
        match &self.defs.get(&pair).ok_or(Unstructured::Shape)?.kind {
            Operation::Read(name) if self.is_pair(name) => Ok(Expr::Local(component(name, index))),
            _ => Err(Unstructured::Operation),
        }
    }
}
