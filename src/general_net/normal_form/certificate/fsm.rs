//! L8 certificates (#59): a finite-state map re-derived on the checker's
//! view. The state is a `u32` carried scalar, every update reads one table the
//! loop neither declares, assigns nor stores to, at an index whose polynomial
//! normal form is `stride * s + rest` with `rest` free of `s`, and no guard of
//! an update reads `s`.

use super::{
    Rejection,
    loops::{expr_mentions, guards_are, guards_free, mentions, one},
    view::View,
};
use crate::{
    front::Type,
    general_net::{
        normal_form::{
            Certificate, Fact, Guard, Rep, RepExpr,
            poly::{Arith, local_key, normalize},
        },
        regions::tree::{Expr, same},
    },
};

pub(super) fn check(view: &View<'_>, var: &str, cert: &Certificate) -> Result<(), Rejection> {
    if !view.carried().iter().any(|c| c == var) {
        return Err(Rejection::Subject);
    }
    let Some(Rep::Lifted(RepExpr::FunctionTable { table, stride })) = &cert.rep else {
        return Err(Rejection::Rep("not a function table"));
    };
    let locals = &view.tree.locals;
    if locals.get(var) != Some(&Type::U32) || locals.get(table) != Some(&Type::ArrayU32) {
        return Err(Rejection::Rep("state or table type"));
    }
    if *stride == 0 {
        return Err(Rejection::Rep("zero stride"));
    }
    if view.stores.contains_key(table)
        || view.assigned.contains(table)
        || view.declared.contains(table)
    {
        return Err(Rejection::Law("table varies in the loop"));
    }
    let own = one(var);
    let updates = view.updates.get(var).ok_or(Rejection::Subject)?;
    if updates.is_empty() || cert.facts.len() != updates.len() {
        return Err(Rejection::Facts);
    }
    for (update, fact) in updates.iter().zip(&cert.facts) {
        let value = update
            .value
            .as_ref()
            .ok_or(Rejection::Rep("inexpressible update"))?;
        let Expr::Index { array, index, .. } = value else {
            return Err(Rejection::Rep("update is not a table read"));
        };
        if array != table || !guards_free(update, &own, false) {
            return Err(Rejection::Rep("update is not a transition of the table"));
        }
        let stated = normalize(Arith::Exact, index)
            .and_then(|poly| poly.split_linear(&local_key(var)))
            .is_some_and(|(c, rest)| {
                c.as_constant() == Some(i128::from(*stride)) && !mentions(&rest, &own)
            });
        if !stated {
            return Err(Rejection::Rep("index is not stride * state + offset"));
        }
        if !matches!(fact, Fact::Transition(e) if same(e, value) && expr_mentions(e, &own)) {
            return Err(Rejection::Facts);
        }
    }
    guards_are(
        cert,
        &[Guard::StateTable {
            table: table.clone(),
            stride: *stride,
            seed: var.to_owned(),
        }],
    )
}
