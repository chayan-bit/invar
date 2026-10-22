//! Exhaustive structural rule inventory for the current checked MIR.
//!
//! Every variant must choose a rule explicitly. This inventory records a
//! checked compiler boundary, not a device implementation or a simulation proof.

use crate::{front::Span, mir};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rule {
    Tick,
    I32,
    U32,
    F32,
    Bool,
    Blob,
    Read,
    Move,
    MoveField,
    ArrayRef,
    RecordRef,
    Array,
    Record,
    Field,
    Variant,
    VariantTag,
    VariantPayload,
    VariantTakePayload,
    DropValue,
    Unary,
    Binary,
    Index,
    Length,
    Call,
    Bind,
    Store,
    StoreIndex,
    Drop,
    ForNext,
    Goto,
    Branch,
    Return,
    // Appended last so the discriminants that emitted sources and receipts
    // encode stay identical to the pre-f64 inventory.
    F64,
}

pub const ALL_RULES: [Rule; 33] = [
    Rule::Tick,
    Rule::I32,
    Rule::U32,
    Rule::F32,
    Rule::Bool,
    Rule::Blob,
    Rule::Read,
    Rule::Move,
    Rule::MoveField,
    Rule::ArrayRef,
    Rule::RecordRef,
    Rule::Array,
    Rule::Record,
    Rule::Field,
    Rule::Variant,
    Rule::VariantTag,
    Rule::VariantPayload,
    Rule::VariantTakePayload,
    Rule::DropValue,
    Rule::Unary,
    Rule::Binary,
    Rule::Index,
    Rule::Length,
    Rule::Call,
    Rule::Bind,
    Rule::Store,
    Rule::StoreIndex,
    Rule::Drop,
    Rule::ForNext,
    Rule::Goto,
    Rule::Branch,
    Rule::Return,
    Rule::F64,
];

/// Conservative observations requiring the selected block's ordered token.
/// Even value-only agents remain sequential until a separate legality rule
/// permits speculation. This flag is never a profitability authorization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Observation {
    Administrative,
    Value,
    MayTrap,
    ReadPlace,
    TransferOwner,
    Borrow,
    Allocate,
    WritePlace,
    Cleanup,
    Call,
    HostBoundary,
}

/// MIR operands in their existing port order. StoreIndex's RHS and index
/// evaluation order comes from preceding agents, not from permuting ports.
pub fn operands(op: &mir::Operation) -> Vec<mir::ValueId> {
    use mir::Operation as O;
    match op {
        O::Tick
        | O::I32(_)
        | O::U32(_)
        | O::F32(_)
        | O::F64(_)
        | O::Bool(_)
        | O::Blob { .. }
        | O::Read(_)
        | O::Move(_)
        | O::MoveField { .. }
        | O::ArrayRef { .. }
        | O::RecordRef { .. }
        | O::Length(_)
        | O::Drop(_) => Vec::new(),
        O::Array(items) | O::Call { args: items, .. } => items.clone(),
        O::Record { fields, .. } => fields.iter().map(|(_, value)| *value).collect(),
        O::Unary { value, .. }
        | O::Bind { value, .. }
        | O::Store { value, .. }
        | O::ForNext(value)
        | O::VariantTag(value)
        | O::VariantPayload { value, .. }
        | O::VariantTakePayload { value, .. }
        | O::DropValue(value) => vec![*value],
        O::Binary { left, right, .. } => vec![*left, *right],
        O::Index { array, index } => vec![*array, *index],
        O::Field { record, .. } => vec![*record],
        O::Variant { payload, .. } => payload.iter().copied().collect(),
        O::StoreIndex { index, value, .. } => vec![*index, *value],
    }
}

pub fn operation(op: &mir::Operation) -> (Rule, Observation) {
    use Observation as E;
    use Rule as R;
    use mir::Operation as O;
    match op {
        O::Tick => (R::Tick, E::Administrative),
        O::I32(_) => (R::I32, E::Value),
        O::U32(_) => (R::U32, E::Value),
        O::F32(_) => (R::F32, E::Value),
        O::F64(_) => (R::F64, E::Value),
        O::Bool(_) => (R::Bool, E::Value),
        O::Blob { .. } => (R::Blob, E::Allocate),
        O::Read(_) => (R::Read, E::ReadPlace),
        O::Move(_) => (R::Move, E::TransferOwner),
        O::MoveField { .. } => (R::MoveField, E::TransferOwner),
        O::ArrayRef { .. } => (R::ArrayRef, E::Borrow),
        O::RecordRef { .. } => (R::RecordRef, E::Borrow),
        O::Array(_) => (R::Array, E::Allocate),
        O::Record { .. } => (R::Record, E::Allocate),
        O::Field { .. } => (R::Field, E::ReadPlace),
        O::Variant { .. } => (R::Variant, E::Allocate),
        O::VariantTag(_) => (R::VariantTag, E::Value),
        O::VariantPayload { .. } => (R::VariantPayload, E::ReadPlace),
        O::VariantTakePayload { .. } => (R::VariantTakePayload, E::TransferOwner),
        O::DropValue(_) => (R::DropValue, E::Cleanup),
        O::Unary { .. } => (R::Unary, E::MayTrap),
        O::Binary { .. } => (R::Binary, E::MayTrap),
        O::Index { .. } => (R::Index, E::MayTrap),
        O::Length(_) => (R::Length, E::ReadPlace),
        O::Call { .. } => (R::Call, E::Call),
        O::Bind { .. } => (R::Bind, E::WritePlace),
        O::Store { .. } => (R::Store, E::WritePlace),
        O::StoreIndex { .. } => (R::StoreIndex, E::WritePlace),
        O::Drop(_) => (R::Drop, E::Cleanup),
        O::ForNext(_) => (R::ForNext, E::MayTrap),
    }
}

pub fn terminator(term: &mir::Terminator) -> (Rule, Vec<mir::ValueId>, Span) {
    match term {
        mir::Terminator::Goto { args, span, .. } => (Rule::Goto, args.clone(), *span),
        mir::Terminator::Branch {
            condition, span, ..
        } => (Rule::Branch, vec![*condition], *span),
        mir::Terminator::Return { value, span, .. } => {
            (Rule::Return, value.iter().copied().collect(), *span)
        }
    }
}
