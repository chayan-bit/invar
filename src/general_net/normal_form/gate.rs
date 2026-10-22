//! Cost-gated admission (#59, normal-form design section 6.4). L8 multiplies the work of
//! every iteration by its state count and L5 by `(k + 1)^2`; both are legal,
//! but only the calibrated cost model (#52) may choose them over the serial
//! loop. Until then an extracted loop that uses either stays serial with
//! [`super::COST_UNCALIBRATED`], unless the caller admits uncalibrated laws.

use super::{LoopForm, Node, Rep, RepExpr};
use crate::general_net::regions::Partition;

/// Matrix groups up to this size are the v0.1 L5 cut and need no cost model.
pub(super) const CALIBRATED_MATRIX: usize = 4;

fn serial(form: &LoopForm, reason: &str) -> LoopForm {
    LoopForm {
        region: form.region,
        node: Node::Serial {
            reason: reason.to_owned(),
        },
        certificates: Vec::new(),
    }
}

/// Admission of an extracted loop: an L8 loop whose table the loop declares or
/// writes is not a finite-state map, and a cost-gated loop stays serial unless
/// `uncalibrated`.
pub(super) fn admit(partition: &Partition, form: LoopForm, uncalibrated: bool) -> LoopForm {
    let Node::Parallel(node) = &form.node else {
        return form;
    };
    let tables: Vec<&String> = node
        .carried
        .iter()
        .filter_map(|c| match &c.rep {
            Rep::Lifted(RepExpr::FunctionTable { table, .. }) => Some(table),
            _ => None,
        })
        .collect();
    if node.arrays.iter().any(|(name, _)| tables.contains(&name)) {
        return serial(&form, "finite-state table written in the loop");
    }
    let declared = partition
        .region(form.region)
        .is_none_or(|region| tables.iter().any(|t| region.ownership.declared.contains(t)));
    if !tables.is_empty() && declared {
        return serial(&form, "finite-state table declared in the loop");
    }
    let large = node
        .carried
        .iter()
        .any(|c| matches!(&c.rep, Rep::Matrix { vars, .. } if vars.len() > CALIBRATED_MATRIX));
    if (tables.is_empty() && !large) || uncalibrated {
        form
    } else {
        serial(&form, super::COST_UNCALIBRATED)
    }
}
