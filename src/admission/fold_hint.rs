//! v0.2 explain hint for serial binary32 folds (`acc = acc + e` in a counted
//! loop). The hint names the explicit reductions; it never rewrites the loop,
//! because a parallel order would change the rounded result.

use std::collections::HashMap;

use crate::{
    front::{BinaryOp, Type},
    mir::{BlockId, Function, Operation, ValueId},
};

use super::{Obligation, fact};

pub const CODE: &str = "FLOAT_FOLD_HINT";

/// The hint for a loop whose body block folds into an f32 local, if any.
// ponytail: only the direct body block is scanned; folds under a nested `if`
// get no hint until a lane needs that.
pub(super) fn hint(
    function: &Function,
    defs: &HashMap<ValueId, (&Operation, BlockId)>,
    body: BlockId,
) -> Option<Obligation> {
    let block = function.blocks.get(body)?;
    let reads = |id: &ValueId, target: &str| matches!(defs.get(id), Some((Operation::Read(name), _)) if name == target);
    let folds = block.instructions.iter().any(|instruction| {
        let Operation::Store { name, value } = &instruction.kind else {
            return false;
        };
        block.instructions.iter().any(|sum| {
            matches!(
                (&sum.output, &sum.kind),
                (Some((id, Type::F32)), Operation::Binary { op: BinaryOp::Add, left, right, .. })
                    if id == value && (reads(left, name) || reads(right, name))
            )
        })
    });
    folds.then(|| {
        fact(
            CODE,
            "hint",
            "serial binary32 fold kept in source order; sum_tree(xs) gives a fixed-shape \
             parallel sum and sum_exact(xs) the correctly rounded sum, with no rewrite",
        )
    })
}
