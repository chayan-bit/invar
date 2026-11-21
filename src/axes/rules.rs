//! The lowest C level each rule a plan or the normal-form route names needs
//! (`docs/spec/invar-v0.2-axes.md` NA3.13 to NA3.15), and the admission gate
//! that keeps a region from running a rule above its declared level.

use super::{C_AXIS, CLevel};
use crate::{
    admission::{Region, Status, fact},
    general_net::normal_form::LawId,
    plan::{Execution, PlanKind},
};

/// Every rule identity a plan names in `rule_identities`, with the lowest C
/// level that admits it. A contract named at a call site (NA3.10) is not a
/// region rule; its level is in `CONTRACTS`.
pub const RULES: [(&str, CLevel); 7] = [
    (crate::admission::RULE_REVISION, CLevel::C0),
    (crate::typed_net::MAP_FUSION_RULE, CLevel::C0),
    (crate::typed_net::DAG_FUSION_RULE, CLevel::C0),
    (crate::laws::RULE_REVISION, CLevel::C0),
    (crate::pair_reduction::RULE_REVISION, CLevel::C0),
    (crate::recurrence::RULE_REVISION, CLevel::C0),
    (crate::checked_fold::RULE_REVISION, CLevel::C0),
];

/// The obligation a region fails when its plan names a rule above its level.
pub const OBLIGATION: &str = "CONTRACT_LEVEL";

/// The lowest level that admits `rule`; `None` for a rule this table lacks,
/// which no level admits.
pub fn rule_level(rule: &str) -> Option<CLevel> {
    #[cfg(test)]
    if let Some(level) = tests::tagged(rule) {
        return Some(level);
    }
    RULES
        .iter()
        .find(|(name, _)| *name == rule)
        .map(|(_, level)| *level)
}

/// The lowest level of a normal-form law: every law of the v0.1 set is exact,
/// so C0. A new law must be given its level here before it can compile.
pub fn law_level(law: LawId) -> CLevel {
    #[cfg(test)]
    if let Some(level) = tests::tagged(&law.to_string()) {
        return level;
    }
    match law {
        LawId::L1Map
        | LawId::L2SealedFold
        | LawId::L3TrapLiftedFold
        | LawId::L4Affine
        | LawId::L5Matrix
        | LawId::L6Tropical
        | LawId::L7Select
        | LawId::L8FiniteState
        | LawId::L9PathSum
        | LawId::L10RangeSum
        | LawId::L11ScatterCombine
        | LawId::L12OrderedWrite
        | LawId::L13Compaction
        | LawId::L14MonotoneFrontier
        | LawId::L15FirstExit
        | LawId::L19MapFusion
        | LawId::L20ForkJoin => CLevel::C0,
    }
}

/// The declared C level of `function` in `program`, C0 when undeclared.
fn declared(program: &crate::mir::Program, function: &str) -> CLevel {
    program
        .region_axes
        .get(function)
        .and_then(|axes| CLevel::from_name(axes.level(&C_AXIS)))
        .unwrap_or(CLevel::C0)
}

/// Whether the normal-form route may apply `laws`: it runs the whole program,
/// so every law must be admitted at the lowest level any function declares.
pub fn laws_admitted(program: &crate::mir::Program, laws: impl IntoIterator<Item = LawId>) -> bool {
    let floor = program
        .functions
        .iter()
        .map(|function| declared(program, &function.name))
        .min()
        .unwrap_or(CLevel::C0);
    laws.into_iter().all(|law| law_level(law) <= floor)
}

/// Refuse a region whose plan names a rule above its declared level, or one
/// `RULES` lacks: unknown legality is serial (NA3.14). Serial is the fallback
/// and is never refused. Fused partners share a function, so a refusal of one
/// is a refusal of the other.
pub(crate) fn gate(region: &mut Region) {
    if region.plan_kind() == PlanKind::Serial {
        return;
    }
    let level = CLevel::from_name(region.axes.level(&C_AXIS)).unwrap_or(CLevel::C0);
    let refused = region
        .execution()
        .rules()
        .into_iter()
        .find(|rule| rule_level(rule).is_none_or(|needed| needed > level));
    let Some(rule) = refused else {
        return;
    };
    region.obligations.push(fact(
        OBLIGATION,
        "unknown",
        "a rule of the candidate plan needs a C level above the region's",
    ));
    region.status = Status::Rejected;
    region.first_rejected_obligation = Some(OBLIGATION);
    (region.guard, region.shape) = (None, None);
    region.execution = Execution::Serial;
    region.c_refused = Some(rule);
}

/// The receipt fields of the gate: `rule_levels`, the lowest level of each
/// rule the chosen plan names, and on a refusal `c_refused`, the rule and its
/// level (null when the table lacks it).
pub fn receipt(
    rules: &[&str],
    refused: Option<&str>,
) -> serde_json::Map<String, serde_json::Value> {
    let level = |rule: &str| rule_level(rule).map(CLevel::name);
    let mut fields = serde_json::Map::new();
    let levels: Vec<_> = rules.iter().map(|rule| level(rule)).collect();
    fields.insert("rule_levels".into(), serde_json::json!(levels));
    if let Some(rule) = refused {
        fields.insert(
            "c_refused".into(),
            serde_json::json!({"rule": rule, "c_level": level(rule)}),
        );
    }
    fields
}

#[cfg(test)]
mod tests;
