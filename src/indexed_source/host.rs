//! Isolated synchronous CPU execution of reached checked-source gathers.
//! Ordinary MIR retains allocation, return, traps and cleanup. No native/device ABI.
use super::{Access, Candidate, Contract, Error, Family};
use crate::{
    check,
    eval::{EvalError, Observation},
    front::{self, Expr, ExprKind, Place, Span, StmtKind},
    general_net::compiled::FenceObservation,
    mir,
    mir_eval::{self, LiveLoop, LoopOverride},
};
use sha2::{Digest, Sha256};
use std::{cell::RefCell, rc::Rc};

pub const MAX_STAGING_BYTES: usize = 160 * 1024;
pub const MAX_VISITS: usize = 65_536;
const METADATA_BYTES: usize = 4096;

/// Logical retained payload/pass limits; allocator overhead/RSS and checked
/// source/AST/net/MIR storage/verification work are outside this accounting.
/// Caller limits may tighten, never increase, the frozen limits.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub bytes: usize,
    pub visits: usize,
}
impl Default for Budget {
    fn default() -> Self {
        Self {
            bytes: MAX_STAGING_BYTES,
            visits: MAX_VISITS,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cost {
    pub bytes: usize,
    pub visits: usize,
}

pub(crate) fn cost(a: usize, ix: usize, out: usize, budget: Budget) -> Option<Cost> {
    if ix != out {
        return None;
    }
    let inputs = a.checked_add(ix)?;
    let bytes = inputs
        .checked_add(out)?
        .checked_mul(size_of::<u32>())?
        .checked_add(out.checked_mul(size_of::<Access>())?)?
        .checked_add(METADATA_BYTES)?;
    let visits = inputs.checked_mul(2)?.checked_add(out.checked_mul(6)?)?;
    (bytes <= budget.bytes.min(MAX_STAGING_BYTES) && visits <= budget.visits.min(MAX_VISITS))
        .then_some(Cost { bytes, visits })
}

pub(crate) fn failure(code: &'static str) -> EvalError {
    EvalError::Tool {
        code,
        message: "checked-source CPU gather failed; no publication or replay".into(),
    }
}
fn ticks(expr: &Expr, output: &mut Vec<Span>) -> Result<(), Error> {
    if let ExprKind::Group(inner) = &expr.kind {
        return ticks(inner, output);
    }
    output.push(expr.span);
    match &expr.kind {
        ExprKind::Name(_) => Ok(()),
        ExprKind::Index { base, index } => {
            ticks(base, output)?;
            ticks(index, output)
        }
        _ => Err(Error::Unrecognized),
    }
}

pub(crate) struct Plan {
    source: Rc<[u8]>,
    source_hash: [u8; 32],
    pub(crate) contract: Contract,
    pub(crate) function: String,
    pub(crate) values: String,
    pub(crate) indices: String,
    pub(crate) output: String,
    pub(crate) program_digest: [u8; 32],
    pub(crate) function_digest: [u8; 32],
    tick_spans: Vec<Span>,
}
impl Plan {
    pub(crate) fn bind(
        source: Rc<[u8]>,
        checked: &check::CheckedProgram,
        entry: &str,
        program: &mir::Program,
    ) -> Result<Self, Error> {
        let contract = Contract::from_checked(&source, checked, entry)?;
        if contract.family() != Family::Gather {
            return Err(Error::Unrecognized);
        }
        let f = checked
            .program
            .functions
            .iter()
            .find(|f| f.name == entry)
            .ok_or(Error::Unrecognized)?;
        let [allocation, region, _] = f.body.statements.as_slice() else {
            return Err(Error::Unrecognized);
        };
        let StmtKind::Binding { name: output, .. } = &allocation.kind else {
            return Err(Error::Unrecognized);
        };
        let StmtKind::For { body, .. } = &region.kind else {
            return Err(Error::Unrecognized);
        };
        let [store] = body.statements.as_slice() else {
            return Err(Error::Unrecognized);
        };
        let StmtKind::Assign {
            place: Place::Index { index, .. },
            value,
        } = &store.kind
        else {
            return Err(Error::Unrecognized);
        };
        let mut tick_spans = vec![region.span, store.span];
        ticks(value, &mut tick_spans)?;
        ticks(index, &mut tick_spans)?;
        let function = program
            .functions
            .iter()
            .find(|f| f.name == entry)
            .ok_or(Error::Unrecognized)?;
        let p = Self {
            source_hash: Sha256::digest(&source).into(),
            source,
            function: entry.into(),
            values: f.params[0].name.clone(),
            indices: f.params[1].name.clone(),
            output: output.clone(),
            program_digest: program.contract_digest(),
            function_digest: crate::tensor::function_fingerprint(function),
            contract,
            tick_spans,
        };
        p.verify().map_err(|_| Error::CheckedMismatch)?;
        Ok(p)
    }
    pub(crate) fn loop_ticks(&self, n: usize) -> Option<u64> {
        u64::try_from(n)
            .ok()?
            .checked_mul(u64::try_from(self.tick_spans.len()).ok()?)
    }
    pub(crate) fn matches_source(&self, source_hash: [u8; 32]) -> bool {
        self.source_hash == source_hash
    }
    pub(crate) fn verify(&self) -> Result<(), EvalError> {
        let source_hash: [u8; 32] = Sha256::digest(&self.source).into();
        let m = self
            .contract
            .verified_mir()
            .map_err(|_| failure("GATHER_SOURCE_BINDING"))?;
        if source_hash != self.source_hash
            || m.contract_digest() != self.program_digest
            || self.contract.source_contract() != self.program_digest
        {
            return Err(failure("GATHER_SOURCE_BINDING"));
        }
        let function = m
            .functions
            .iter()
            .find(|f| f.name == self.function)
            .ok_or_else(|| failure("GATHER_SOURCE_BINDING"))?;
        self.verify_function(function)
    }
    pub(crate) fn verify_function(&self, function: &mir::Function) -> Result<(), EvalError> {
        if crate::tensor::function_fingerprint(function) != self.function_digest {
            return Err(failure("GATHER_SOURCE_BINDING"));
        }
        let (head_id, head, body) = function
            .blocks
            .iter()
            .enumerate()
            .find_map(|(i, b)| match &b.end {
                Some(mir::Terminator::Branch { yes, span, .. })
                    if *span == self.contract.loop_span() =>
                {
                    function.blocks.get(*yes).map(|body| (i, b, body))
                }
                _ => None,
            })
            .ok_or_else(|| failure("GATHER_FUEL_RECIPE"))?;
        if head
            .instructions
            .iter()
            .any(|i| matches!(i.kind, mir::Operation::Tick))
            || !matches!(body.end, Some(mir::Terminator::Goto { target, .. }) if target == head_id)
            || !body
                .instructions
                .iter()
                .filter(|i| matches!(i.kind, mir::Operation::Tick))
                .map(|i| i.span)
                .eq(self.tick_spans.iter().copied())
        {
            return Err(failure("GATHER_FUEL_RECIPE"));
        }
        Ok(())
    }
}

pub(crate) struct Payload {
    pub(crate) source_hash: [u8; 32],
    pub(crate) plan: Rc<Plan>,
    pub(crate) values: Vec<u32>,
    pub(crate) indices: Vec<u32>,
    pub(crate) identities: [usize; 3],
    pub(crate) candidate: Candidate,
    pub(crate) cost: Cost,
    pub(crate) budget: Budget,
    pub(crate) ticks: u64,
}
pub(crate) type Quarantine = Rc<RefCell<Option<Pending>>>;
/// Opaque and crate-private. Only the synchronous producer can issue it;
/// source/runtime checks supply authority, not a public status or proof flag.
pub(crate) struct Completion {
    pub(crate) payload: Rc<Payload>,
    pub(crate) status: FenceObservation,
    pub(crate) quarantine: Quarantine,
}

/// An unknown-completion token retains the reached frame and private staging.
/// This CPU-only route has no asynchronous submission/recovery API. Dropping an
/// unresolved token retains its resources, following the existing fence rule.
pub struct Pending {
    pub(crate) resources: Option<crate::mir_eval::gather::Quarantined>,
}
impl Pending {
    pub fn retained_arrays(&self) -> usize {
        self.resources.as_ref().map_or(0, |r| r.retained_arrays())
    }
}
impl Drop for Pending {
    fn drop(&mut self) {
        if let Some(resources) = self.resources.take() {
            std::mem::forget(resources);
        }
    }
}
pub struct Run {
    pub observation: Observation,
    pub begun: usize,
    pub completed: usize,
    pub refusals: Vec<&'static str>,
    pub pending: Option<Pending>,
}

fn copy(values: &[u32]) -> Option<Vec<u32>> {
    let mut v = Vec::new();
    v.try_reserve_exact(values.len()).ok()?;
    v.extend_from_slice(values);
    Some(v)
}
pub(crate) fn distinct(live: &LiveLoop<'_>, plan: &Plan) -> bool {
    !live.same_object(&plan.values, &plan.output) && !live.same_object(&plan.indices, &plan.output)
}
type Producer = fn(&Contract, &[u32], &[u32]) -> Result<Candidate, Error>;
pub(crate) fn produce_sync(
    contract: &Contract,
    values: &[u32],
    indices: &[u32],
) -> Result<Candidate, Error> {
    contract.candidate(values, indices, None)
}
pub(crate) fn prepare(
    plan: Rc<Plan>,
    live: &LiveLoop<'_>,
    budget: Budget,
    quarantine: Quarantine,
    begun: &mut usize,
    refusals: &mut Vec<&'static str>,
    producer: Producer,
) -> Result<Option<Completion>, EvalError> {
    plan.verify()?;
    let refused = |reason, refusals: &mut Vec<&'static str>| {
        if refusals.len() < 64 {
            refusals.push(reason);
        }
        Ok(None)
    };
    if plan.function != live.function
        || plan.contract.loop_span() != live.span
        || !distinct(live, &plan)
    {
        return refused("gather_owner_or_region", refusals);
    }
    let Some(values) = live.array(&plan.values) else {
        return refused("gather_input_loan", refusals);
    };
    let Some(indices) = live.array(&plan.indices) else {
        return refused("gather_index_loan", refusals);
    };
    let Some(output) = live.array(&plan.output) else {
        return refused("gather_output_loan", refusals);
    };
    let Some(cost) = cost(values.len(), indices.len(), output.len(), budget) else {
        return refused("gather_budget_or_extent", refusals);
    };
    let Some(ticks) = plan
        .loop_ticks(indices.len())
        .filter(|t| *t <= live.fuel_remaining)
    else {
        return refused("gather_source_fuel", refusals);
    };
    if indices.len() != live.iterations
        || indices
            .iter()
            .zip(output.iter())
            .any(|(&i, &zero)| zero != 0 || !usize::try_from(i).is_ok_and(|i| i < values.len()))
    {
        return refused("gather_bounds_or_fresh_output", refusals);
    }
    let Some(values_copy) = copy(&values) else {
        return refused("gather_private_allocation", refusals);
    };
    let Some(indices_copy) = copy(&indices) else {
        return refused("gather_private_allocation", refusals);
    };
    let identities = [
        std::ptr::from_ref(&*values).addr(),
        std::ptr::from_ref(&*indices).addr(),
        std::ptr::from_ref(&*output).addr(),
    ];
    let next = begun
        .checked_add(1)
        .ok_or_else(|| failure("GATHER_COUNT"))?;
    *begun = next; // Candidate entry: all later failures are typed, with no replay.
    let candidate =
        producer(&plan.contract, &values, &indices).map_err(|_| failure("GATHER_POST_ENTRY"))?;
    Ok(Some(Completion {
        payload: Rc::new(Payload {
            source_hash: plan.source_hash,
            plan,
            values: values_copy,
            indices: indices_copy,
            identities,
            candidate,
            cost,
            budget,
            ticks,
        }),
        status: FenceObservation::Complete,
        quarantine,
    }))
}

/// Execute ordinary source with only reached verified gather loops replaced.
/// No GPU, native emission, global dispatch or performance qualification.
pub fn run_source(source: &[u8], fuel: u64, budget: Budget) -> Result<Run, Error> {
    run_with(source, fuel, budget, |_| {})
}
fn run_with(
    source: &[u8],
    fuel: u64,
    budget: Budget,
    receipt: impl FnMut(&mut Completion),
) -> Result<Run, Error> {
    run_using(source, fuel, budget, receipt, produce_sync)
}
fn run_using(
    source: &[u8],
    fuel: u64,
    budget: Budget,
    mut receipt: impl FnMut(&mut Completion),
    producer: Producer,
) -> Result<Run, Error> {
    let checked = front::parse(source)
        .and_then(|p| check::check(p, true))
        .map_err(Error::Source)?;
    let program =
        mir::lower(&checked).map_err(|e| Error::Net(crate::general_net::Error::Mir(e)))?;
    let source: Rc<[u8]> = source.into();
    let mut plans = Vec::new();
    plans
        .try_reserve_exact(checked.program.functions.len())
        .map_err(|_| Error::Allocation)?;
    for f in &checked.program.functions {
        match Plan::bind(source.clone(), &checked, &f.name, &program) {
            Ok(p) => plans.push(Rc::new(p)),
            Err(Error::Unrecognized) => {}
            Err(e) => return Err(e),
        }
    }
    let quarantine = Rc::new(RefCell::new(None));
    let mut begun = 0;
    let mut refusals = Vec::new();
    refusals
        .try_reserve_exact(64)
        .map_err(|_| Error::Allocation)?;
    let mut provider = |live: &LiveLoop<'_>| {
        let Some(plan) = plans
            .iter()
            .find(|p| p.function == live.function && p.contract.loop_span() == live.span)
        else {
            return Ok(None);
        };
        let Some(mut completion) = prepare(
            plan.clone(),
            live,
            budget,
            quarantine.clone(),
            &mut begun,
            &mut refusals,
            producer,
        )?
        else {
            return Ok(None);
        };
        receipt(&mut completion);
        Ok(Some(LoopOverride {
            gather: Some(completion),
            iterations: live.iterations,
            ..Default::default()
        }))
    };
    let (observation, completed) =
        mir_eval::observe_with_scheduler(&program, fuel, Some(&mut provider));
    let pending = quarantine.borrow_mut().take();
    Ok(Run {
        observation,
        begun,
        completed,
        refusals,
        pending,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    const SOURCE: &[u8] = b"fn gather(a: read Array<u32>, ix: read Array<u32>) -> Array<u32> { var out = filled_u32(len(ix),0u32); for i in 0u32..len(ix) { out[i] = a[ix[i]]; } return out; } fn main() -> i32 { let a: Array<u32> = [7u32,11u32]; let ix: Array<u32> = [1u32,0u32,1u32]; let out = gather(a,ix); return 7; }";
    #[test]
    fn issue_m4_05_gather_scatter_frontier_runtime_post_entry_error_no_replay() {
        let run = run_using(
            SOURCE,
            1000,
            Budget::default(),
            |_| {},
            |_, _, _| Err(Error::Allocation),
        )
        .unwrap();
        assert_eq!(run.begun, 1);
        assert_eq!(run.completed, 0);
        assert!(run.pending.is_none());
        assert!(matches!(
            run.observation.result,
            Err(EvalError::Tool {
                code: "GATHER_POST_ENTRY",
                ..
            })
        ));
        assert!(run.observation.cleanup_trace.is_empty());
    }
    #[test]
    fn issue_m4_05_gather_scatter_frontier_runtime_bad_receipts_no_replay() {
        for fault in 0..7 {
            let run = run_with(SOURCE, 1000, Budget::default(), |c| {
                let p = Rc::get_mut(&mut c.payload).unwrap();
                match fault {
                    0 => {
                        p.candidate.output.pop();
                    }
                    1 => p.candidate.output[0] ^= 1,
                    2 => p.values[0] ^= 1,
                    3 => p.identities[2] ^= 1,
                    4 => p.ticks += 1,
                    5 => c.status = FenceObservation::FailedQuiescent,
                    _ => p.source_hash[0] ^= 1,
                }
            })
            .unwrap();
            assert_eq!(run.begun, 1);
            assert_eq!(run.completed, 0);
            assert!(run.pending.is_none());
            assert!(matches!(
                run.observation.result,
                Err(EvalError::Tool { .. })
            ));
            assert!(run.observation.cleanup_trace.is_empty());
        }
    }
    #[test]
    fn issue_m4_05_gather_scatter_frontier_runtime_unknown_receipt_retains_frame() {
        for status in [FenceObservation::Unknown, FenceObservation::Pending] {
            let run = run_with(SOURCE, 1000, Budget::default(), |c| c.status = status).unwrap();
            assert_eq!(run.begun, 1);
            assert_eq!(run.completed, 0);
            assert!(matches!(
                run.observation.result,
                Err(EvalError::Tool {
                    code: "GATHER_UNKNOWN_COMPLETION",
                    ..
                })
            ));
            assert!(run.observation.cleanup_trace.is_empty());
            let mut pending = run.pending.unwrap();
            assert_eq!(pending.retained_arrays(), 3);
            let resources = pending.resources.take().unwrap();
            let weak = resources.output_weak();
            pending.resources = Some(resources);
            drop(pending);
            assert!(weak.upgrade().is_some()); // unknown token drop retains, never declares quiescence
        }
    }
    #[test]
    fn issue_m4_05_gather_scatter_frontier_runtime_cost_and_metadata_bound() {
        let cost = cost(4096, 4096, 4096, Budget::default()).unwrap();
        assert_eq!(
            cost,
            Cost {
                bytes: 151552,
                visits: 40960
            }
        );
        assert!(
            size_of::<Payload>()
                + size_of::<Completion>()
                + size_of::<Pending>()
                + size_of::<crate::mir_eval::gather::Quarantined>()
                <= METADATA_BYTES
        );
        assert!(super::cost(usize::MAX, 0, 0, Budget::default()).is_none());
        assert!(
            super::cost(
                4096,
                4096,
                4096,
                Budget {
                    bytes: cost.bytes - 1,
                    visits: MAX_VISITS
                }
            )
            .is_none()
        );
        assert!(
            super::cost(
                4096,
                4096,
                4096,
                Budget {
                    bytes: MAX_STAGING_BYTES,
                    visits: cost.visits - 1
                }
            )
            .is_none()
        );
    }
}

#[cfg(test)]
mod diagnostic_tests {
    use super::*;
    const SOURCE: &[u8] = b"fn gather(a: read Array<u32>, ix: read Array<u32>) -> Array<u32> { var out = filled_u32(len(ix),0u32); for i in 0u32..len(ix) { out[i] = a[ix[i]]; } return out; } fn main() -> i32 { let a: Array<u32> = [7u32,11u32]; let ix: Array<u32> = [1u32,0u32,1u32]; let out = gather(a,ix); return 7; }";

    #[test]
    fn issue_ai_01_machine_diagnostics_gather_publication_faults() {
        let checked = check::check(front::parse(SOURCE).unwrap(), true).unwrap();
        let program = mir::lower(&checked).unwrap();
        let mut injected = Vec::new();
        for fault in 0..9 {
            let producer: Producer = if fault == 8 {
                |_, _, _| Err(Error::Allocation)
            } else {
                produce_sync
            };
            let run = run_using(
                SOURCE,
                1000,
                Budget::default(),
                |c| match fault {
                    0 => c.status = FenceObservation::Unknown,
                    1 => c.status = FenceObservation::FailedQuiescent,
                    2 => Rc::get_mut(&mut c.payload).unwrap().source_hash[0] ^= 1,
                    3 => {
                        let mut stale =
                            Plan::bind(SOURCE.into(), &checked, "gather", &program).unwrap();
                        stale.tick_spans.push(Span { start: 0, end: 1 });
                        Rc::get_mut(&mut c.payload).unwrap().plan = Rc::new(stale);
                    }
                    4 => Rc::get_mut(&mut c.payload).unwrap().ticks += 1,
                    5 => Rc::get_mut(&mut c.payload).unwrap().values[0] ^= 1,
                    6 => Rc::get_mut(&mut c.payload).unwrap().candidate.output[0] ^= 1,
                    _ => {}
                },
                producer,
            )
            .unwrap();
            let Err(error @ EvalError::Tool { code, .. }) = &run.observation.result else {
                assert_eq!(
                    (fault, run.completed),
                    (7, 1),
                    "fault {fault} must fail closed"
                );
                continue;
            };
            assert_eq!(run.completed, 0);
            assert!(run.observation.cleanup_trace.is_empty());
            injected.push((*code, error.to_diagnostics_json("gather.inv", SOURCE)));
        }
        let injected_codes = injected.iter().map(|(code, _)| *code).collect::<Vec<_>>();
        assert_eq!(
            injected_codes,
            [
                "GATHER_UNKNOWN_COMPLETION",
                "GATHER_FAILED_COMPLETION",
                "GATHER_SOURCE_BINDING",
                "GATHER_FUEL_RECIPE",
                "GATHER_LOOP_BINDING",
                "GATHER_STALE_COMPLETION",
                "GATHER_BAD_COMPLETION",
                "GATHER_POST_ENTRY",
            ]
        );
        // Only GATHER_FUEL remains constructor-only: it is dead code, because
        // GATHER_LOOP_BINDING first requires ticks <= fuel_limit - steps_used, so
        // steps_used + ticks cannot overflow. GATHER_OWNER, GATHER_LOAN and GATHER_COUNT are injected for
        // real by `mir_eval::gather::tests::issue_ai_01_machine_diagnostics_gather_publication_harness_faults`.
        let constructed = ["GATHER_FUEL"].map(|code| {
            (
                code,
                failure(code).to_diagnostics_json("gather.inv", SOURCE),
            )
        });
        let registry: serde_json::Value = serde_json::from_str(include_str!(
            "../../schema/diagnostic-codes-indexed-v1.json"
        ))
        .unwrap();
        let mut registered = registry["codes"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|e| {
                e["fault_test"] == "issue_ai_01_machine_diagnostics_gather_publication_faults"
            })
            .map(|e| e["code"].as_str().unwrap())
            .collect::<Vec<_>>();
        registered.sort_unstable();
        let mut covered = injected_codes.clone();
        covered.extend(constructed.iter().map(|(code, _)| *code));
        covered.sort_unstable();
        assert_eq!(covered, registered);
        for (code, document) in injected.iter().chain(&constructed) {
            assert_eq!(document["status"], "tool_error");
            assert_eq!(document["diagnostics"][0]["code"], *code);
            assert_eq!(document["diagnostics"][0]["severity"], "error");
            assert_eq!(
                document["diagnostics"][0]["suggested_fixes"],
                serde_json::json!([])
            );
        }
    }
}
