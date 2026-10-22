//! Source allocation requirements, not authenticated live state or permission.
//! The runtime ledger is private and includes caller owners absent from LiveLoop.
use crate::front::{ExprKind, Span, Stmt, StmtKind};

#[derive(Debug)]
pub(super) struct Site {
    statement: Span,
    call: Span,
    arguments: [Span; 2],
    ordinal: usize,
    output: String,
    length: String,
    pub(super) ticks: u64,
}
impl Site {
    pub(super) fn from_statement(s: &Stmt, ordinal: usize, ticks: u64) -> Option<Self> {
        let StmtKind::Binding { name, value, .. } = &s.kind else {
            return None;
        };
        let call = super::ungroup(value);
        let ExprKind::Call { args, .. } = &call.kind else {
            return None;
        };
        let [length, fill] = args.as_slice() else {
            return None;
        };
        let ExprKind::Name(length_name) = &super::ungroup(length).kind else {
            return None;
        };
        Some(Self {
            statement: s.span,
            call: call.span,
            arguments: [length.span, fill.span],
            ordinal,
            output: name.clone(),
            length: length_name.clone(),
            ticks,
        })
    }
}

/// A diagnostic copy, checked against the private source-bound requirement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Descriptor<'a> {
    pub source_hash: [u8; 32],
    pub mir_hash: [u8; 32],
    pub function: &'a str,
    /// Zero-based position in the recognized function's top-level statements.
    pub statement_index: usize,
    pub statement_span: Span,
    pub call_span: Span,
    /// Source argument order: selected length, then fill value.
    pub argument_spans: [Span; 2],
    pub output_binding: &'a str,
    pub length_binding: &'a str,
    pub loop_spans: [Span; 2],
    pub requested_elements: u32,
    /// From first-loop dispatch/bounds completion through allocation call/args
    /// charges, immediately before the intrinsic checks its live ledger.
    pub ticks_to_intrinsic: u64,
    pub per_array_limit: usize,
    pub live_limit: usize,
}

/// Immutable descriptive obligation. No constructor accepts a caller's ledger.
#[derive(Debug)]
pub struct Description<'a> {
    descriptor: Descriptor<'a>,
}
impl<'a> Description<'a> {
    pub(super) fn new(
        site: &'a Site,
        function: &'a str,
        source_hash: [u8; 32],
        mir_hash: [u8; 32],
        loop_spans: [Span; 2],
        requested_elements: u32,
        ticks_to_intrinsic: u64,
    ) -> Self {
        Self {
            descriptor: Descriptor {
                source_hash,
                mir_hash,
                function,
                statement_index: site.ordinal,
                statement_span: site.statement,
                call_span: site.call,
                argument_spans: site.arguments,
                output_binding: &site.output,
                length_binding: &site.length,
                loop_spans,
                requested_elements,
                ticks_to_intrinsic,
                per_array_limit: crate::eval::MAX_ARRAY_LEN,
                live_limit: crate::eval::MAX_LIVE_ELEMENTS,
            },
        }
    }
    pub fn descriptor(&self) -> Descriptor<'a> {
        self.descriptor
    }
    /// Required additional units at this source point, excluding candidate
    /// temporaries. Those units must be charged to the actual interpreter ledger.
    pub fn required_live_elements(&self) -> usize {
        self.descriptor.requested_elements as usize
    }
    pub fn maximum_live_before_allocation(&self) -> Option<usize> {
        self.descriptor
            .live_limit
            .checked_sub(self.required_live_elements())
    }
    /// Hypothetical diagnostics only; even a reported zero ledger cannot create
    /// a token permitting execution, source allocation or publication.
    pub fn diagnose_reported_live(&self, live: Option<usize>) -> BudgetStatus {
        diagnose_reported_budget(self.descriptor.requested_elements, live)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    MissingLiveState,
    PerArray,
    LiveElements,
    ArithmeticOverflow,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BudgetStatus {
    Refused(Refusal),
    /// Reported arithmetic fits; authentic source/runtime checks are still owed.
    RequiresRuntimeCheck,
}
/// Evaluate reported arithmetic without granting any runtime admission. Per-array
/// validation precedes live accounting, matching the source intrinsic's order.
pub fn diagnose_reported_budget(requested: u32, live: Option<usize>) -> BudgetStatus {
    let Ok(requested) = usize::try_from(requested) else {
        return BudgetStatus::Refused(Refusal::PerArray);
    };
    if requested > crate::eval::MAX_ARRAY_LEN {
        return BudgetStatus::Refused(Refusal::PerArray);
    }
    let Some(live) = live else {
        return BudgetStatus::Refused(Refusal::MissingLiveState);
    };
    let Some(total) = live.checked_add(requested) else {
        return BudgetStatus::Refused(Refusal::ArithmeticOverflow);
    };
    if total > crate::eval::MAX_LIVE_ELEMENTS {
        return BudgetStatus::Refused(Refusal::LiveElements);
    }
    BudgetStatus::RequiresRuntimeCheck
}
