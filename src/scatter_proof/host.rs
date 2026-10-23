//! Isolated reached-source synchronous CPU scatter. No native/device ABI.
use super::{Contract, Error};
pub use crate::indexed_source::host::{Budget, MAX_STAGING_BYTES, MAX_VISITS};
use crate::{
    check,
    eval::{EvalError, Observation},
    front::{self, BinaryOp, Expr, ExprKind, Place, Span, StmtKind},
    general_net::compiled::FenceObservation,
    indexed_source::Access,
    mir,
    mir_eval::{self, LiveLoop, LoopOverride},
};
use sha2::{Digest, Sha256};
use std::{cell::RefCell, rc::Rc};

const METADATA_BYTES: usize = 4096;
/// Logical payload/pass cost, excluding allocator overhead/RSS, retained source,
/// AST/net/MIR and compiler verification work. Not a machine instruction count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cost {
    pub bytes: usize,
    pub visits: usize,
}
pub(crate) fn cost(a: usize, n: usize, i: usize, budget: Budget) -> Option<Cost> {
    if i > a {
        return None;
    }
    let bytes = a
        .checked_add(n)?
        .checked_mul(size_of::<u32>())?
        .checked_add(i.checked_mul(size_of::<Access>())?)?
        .checked_add(METADATA_BYTES)?;
    let visits = a
        .checked_mul(2)?
        .checked_add(n.checked_mul(4)?)?
        .checked_add(i.checked_mul(3)?)?;
    (bytes <= budget.bytes.min(MAX_STAGING_BYTES) && visits <= budget.visits.min(MAX_VISITS))
        .then_some(Cost { bytes, visits })
}
pub(crate) fn failure(code: &'static str) -> EvalError {
    EvalError::Tool {
        code,
        message: "checked-source CPU scatter failed; no publication or replay".into(),
    }
}
fn ticks(expr: &Expr, spans: &mut Vec<Span>) -> Result<(), Error> {
    if let ExprKind::Group(inner) = &expr.kind {
        return ticks(inner, spans);
    }
    spans.push(expr.span);
    match &expr.kind {
        ExprKind::Name(_) | ExprKind::Integer(_) => Ok(()),
        ExprKind::Index { base, index } => {
            ticks(base, spans)?;
            ticks(index, spans)
        }
        ExprKind::Binary {
            op: BinaryOp::Add | BinaryOp::Mul,
            left,
            right,
            ..
        } => {
            ticks(left, spans)?;
            ticks(right, spans)
        }
        _ => Err(Error::Unrecognized),
    }
}
pub(crate) struct Plan {
    source: Rc<[u8]>,
    pub(crate) source_hash: [u8; 32],
    pub(crate) contract: Contract,
    pub(crate) function: String,
    pub(crate) input: String,
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
        if contract.certificate().domain().0 != 0 {
            return Err(Error::Unrecognized);
        }
        let f = checked
            .program
            .functions
            .iter()
            .find(|f| f.name == entry)
            .ok_or(Error::Unrecognized)?;
        let [allocate, region, _] = f.body.statements.as_slice() else {
            return Err(Error::Unrecognized);
        };
        let StmtKind::Binding { name: output, .. } = &allocate.kind else {
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
        let plan = Self {
            source_hash: Sha256::digest(&source).into(),
            source,
            contract,
            function: entry.into(),
            input: f.params[0].name.clone(),
            output: output.clone(),
            program_digest: program.contract_digest(),
            function_digest: crate::tensor::function_fingerprint(function),
            tick_spans,
        };
        plan.verify().map_err(|_| Error::CertificateMismatch)?;
        Ok(plan)
    }
    pub(crate) fn iterations(&self) -> usize {
        self.contract.certificate().iterations() as usize
    }
    pub(crate) fn output_len(&self) -> usize {
        self.contract.certificate().output_len() as usize
    }
    pub(crate) fn destination(&self, i: usize) -> Option<usize> {
        let (a, b) = self.contract.certificate().affine_map();
        let i = u32::try_from(i).ok()?;
        usize::try_from(a.checked_mul(i)?.checked_add(b)?).ok()
    }
    pub(crate) fn loop_ticks(&self) -> Option<u64> {
        u64::try_from(self.iterations())
            .ok()?
            .checked_mul(u64::try_from(self.tick_spans.len()).ok()?)
    }
    pub(crate) fn verify(&self) -> Result<(), EvalError> {
        let m = self
            .contract
            .verified_mir()
            .map_err(|_| failure("SCATTER_SOURCE_BINDING"))?;
        let hash: [u8; 32] = Sha256::digest(&self.source).into();
        if hash != self.source_hash
            || m.contract_digest() != self.program_digest
            || self.contract.certificate().source_contract() != self.program_digest
            || self.contract.certificate().domain().0 != 0
        {
            return Err(failure("SCATTER_SOURCE_BINDING"));
        }
        self.verify_function(
            m.functions
                .iter()
                .find(|f| f.name == self.function)
                .ok_or_else(|| failure("SCATTER_SOURCE_BINDING"))?,
        )
    }
    pub(crate) fn verify_function(&self, function: &mir::Function) -> Result<(), EvalError> {
        if crate::tensor::function_fingerprint(function) != self.function_digest {
            return Err(failure("SCATTER_SOURCE_BINDING"));
        }
        let (head_id, head, body) = function
            .blocks
            .iter()
            .enumerate()
            .find_map(|(id, b)| match &b.end {
                Some(mir::Terminator::Branch { yes, span, .. })
                    if *span == self.contract.certificate().loop_span() =>
                {
                    function.blocks.get(*yes).map(|body| (id, b, body))
                }
                _ => None,
            })
            .ok_or_else(|| failure("SCATTER_FUEL_RECIPE"))?;
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
            return Err(failure("SCATTER_FUEL_RECIPE"));
        }
        Ok(())
    }
}
pub(crate) struct Payload {
    pub(crate) source_hash: [u8; 32],
    pub(crate) plan: Rc<Plan>,
    pub(crate) input: Vec<u32>,
    pub(crate) identities: [usize; 2],
    pub(crate) output: Vec<u32>,
    pub(crate) accesses: Vec<Access>,
    pub(crate) cost: Cost,
    pub(crate) budget: Budget,
    pub(crate) ticks: u64,
}
pub(crate) type Quarantine = Rc<RefCell<Option<Pending>>>;
/// Private synchronous producer receipt; no public authority constructor/flag.
pub(crate) struct Completion {
    pub(crate) payload: Rc<Payload>,
    pub(crate) status: FenceObservation,
    pub(crate) quarantine: Quarantine,
}
/// Conservative retention for an unknown private test receipt. The production
/// route is synchronous CPU only and exposes no asynchronous recovery API.
pub struct Pending {
    pub(crate) resources: Option<crate::mir_eval::scatter::Quarantined>,
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
pub(crate) fn distinct(live: &LiveLoop<'_>, plan: &Plan) -> bool {
    !live.same_object(&plan.input, &plan.output)
}
type Producer = fn(&Contract, &[u32]) -> Result<Vec<u32>, Error>;
pub(crate) fn produce_sync(contract: &Contract, input: &[u32]) -> Result<Vec<u32>, Error> {
    contract.candidate(input)
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
    let refused = |reason, refusals: &mut Vec<&'static str>| {
        if refusals.len() < 64 {
            refusals.push(reason);
        }
        Ok(None)
    };
    if plan.verify().is_err() {
        return refused("scatter_proof_binding", refusals);
    }
    if plan.function != live.function
        || plan.contract.certificate().loop_span() != live.span
        || !distinct(live, &plan)
    {
        return refused("scatter_owner_or_region", refusals);
    }
    let Some(input) = live.array(&plan.input) else {
        return refused("scatter_input_loan", refusals);
    };
    let Some(output) = live.array(&plan.output) else {
        return refused("scatter_output_loan", refusals);
    };
    let i = plan.iterations();
    let n = plan.output_len();
    let Some(cost) = cost(input.len(), output.len(), i, budget) else {
        return refused("scatter_budget_or_input_bounds", refusals);
    };
    let Some(ticks) = plan.loop_ticks().filter(|t| *t <= live.fuel_remaining) else {
        return refused("scatter_source_fuel", refusals);
    };
    if live.iterations != i || output.len() != n || output.iter().any(|&v| v != 0) {
        return refused("scatter_extent_or_fresh_output", refusals);
    }
    let mut snapshot = Vec::new();
    let mut accesses = Vec::new();
    if snapshot.try_reserve_exact(input.len()).is_err() || accesses.try_reserve_exact(i).is_err() {
        return refused("scatter_private_allocation", refusals);
    }
    snapshot.extend_from_slice(&input);
    for k in 0..i {
        let Some(d) = plan.destination(k).filter(|d| *d < n) else {
            return refused("scatter_physical_map", refusals);
        };
        if k >= input.len()
            || k.checked_mul(size_of::<u32>()).is_none()
            || d.checked_mul(size_of::<u32>()).is_none()
        {
            return refused("scatter_physical_map", refusals);
        }
        accesses.push(Access {
            iteration: k,
            read_index: k,
            write_index: d,
        });
    }
    let identities = [
        std::ptr::from_ref(&*input).addr(),
        std::ptr::from_ref(&*output).addr(),
    ];
    let next = begun
        .checked_add(1)
        .ok_or_else(|| failure("SCATTER_COUNT"))?;
    *begun = next; // Entry boundary: all later failures are typed, never replayed.
    let output = producer(&plan.contract, &input).map_err(|_| failure("SCATTER_POST_ENTRY"))?;
    Ok(Some(Completion {
        payload: Rc::new(Payload {
            source_hash: plan.source_hash,
            plan,
            input: snapshot,
            identities,
            output,
            accesses,
            cost,
            budget,
            ticks,
        }),
        status: FenceObservation::Complete,
        quarantine,
    }))
}
/// Ordinary MIR with only reached verified L=0 affine scatter loops replaced.
/// No automatic global dispatch, native emission, device or performance claim.
pub fn run_source(source: &[u8], fuel: u64, budget: Budget) -> Result<Run, Error> {
    run_using(source, fuel, budget, |_| {}, produce_sync)
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
    let program = mir::lower(&checked).map_err(Error::Mir)?;
    let source: Rc<[u8]> = source.into();
    let mut plans = Vec::new();
    plans
        .try_reserve_exact(checked.program.functions.len())
        .map_err(|_| Error::Allocation)?;
    for f in &checked.program.functions {
        match Plan::bind(source.clone(), &checked, &f.name, &program) {
            Ok(p) => plans.push(Rc::new(p)),
            Err(
                Error::Unrecognized
                | Error::NonInjective
                | Error::Overflow
                | Error::DestinationBounds
                | Error::Budget,
            ) => {}
            Err(e) => return Err(e),
        }
    }
    let mut begun = 0;
    let mut refusals = Vec::new();
    refusals
        .try_reserve_exact(64)
        .map_err(|_| Error::Allocation)?;
    let quarantine = Rc::new(RefCell::new(None));
    let mut provider = |live: &LiveLoop<'_>| {
        let Some(plan) = plans.iter().find(|p| {
            p.function == live.function && p.contract.certificate().loop_span() == live.span
        }) else {
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
            scatter: Some(completion),
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
    const SOURCE: &[u8] = b"fn scatter(a: read Array<u32>) -> Array<u32> { var out = filled_u32(5u32,0u32); for i in 0u32..3u32 { out[2u32*i] = a[i]; } return out; } fn main() -> i32 { let a: Array<u32> = [7u32,11u32,19u32]; let out = scatter(a); return 7; }";
    #[test]
    fn issue_m4_05_gather_scatter_frontier_scatter_runtime_bad_receipts_no_replay() {
        for fault in 0..17 {
            let run = run_using(
                SOURCE,
                1000,
                Budget::default(),
                |c| {
                    let p = Rc::get_mut(&mut c.payload).unwrap();
                    match fault {
                        0 => {
                            p.output.pop();
                        }
                        1 => p.output[0] ^= 1,
                        2 => p.output[1] = 42,
                        3 => p.input[0] ^= 1,
                        4 => p.identities[1] ^= 1,
                        5 => p.ticks += 1,
                        6 => p.accesses[1].write_index = p.accesses[0].write_index,
                        7 => p.accesses[1].read_index = 0,
                        8 => p.accesses[1].iteration = 0,
                        9 => p.source_hash[0] ^= 1,
                        10 => c.status = FenceObservation::FailedQuiescent,
                        11 => p.cost.bytes += 1,
                        12 => p.accesses.swap(0, 1),
                        _ => {
                            let checked =
                                check::check(front::parse(SOURCE).unwrap(), true).unwrap();
                            let program = mir::lower(&checked).unwrap();
                            let mut stale =
                                Plan::bind(SOURCE.into(), &checked, "scatter", &program).unwrap();
                            match fault {
                                13 => stale.contract.certificate.shape.coefficient = 1,
                                14 => stale.contract.certificate.shape.upper = 2,
                                15 => stale.contract.certificate.shape.offset = 1,
                                _ => stale.contract.net.agents[0].span.end += 1,
                            }
                            p.plan = Rc::new(stale);
                        }
                    }
                },
                produce_sync,
            )
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
    fn issue_m4_05_gather_scatter_frontier_scatter_runtime_post_entry_error_no_replay() {
        let run = run_using(
            SOURCE,
            1000,
            Budget::default(),
            |_| {},
            |_, _| Err(Error::Allocation),
        )
        .unwrap();
        assert_eq!(run.begun, 1);
        assert_eq!(run.completed, 0);
        assert!(run.pending.is_none());
        assert!(matches!(
            run.observation.result,
            Err(EvalError::Tool {
                code: "SCATTER_POST_ENTRY",
                ..
            })
        ));
        assert!(run.observation.cleanup_trace.is_empty());
    }
    #[test]
    fn issue_m4_05_gather_scatter_frontier_scatter_runtime_unknown_retains_frame() {
        for status in [FenceObservation::Unknown, FenceObservation::Pending] {
            let run = run_using(
                SOURCE,
                1000,
                Budget::default(),
                |c| c.status = status,
                produce_sync,
            )
            .unwrap();
            assert_eq!(run.begun, 1);
            assert_eq!(run.completed, 0);
            assert!(matches!(
                run.observation.result,
                Err(EvalError::Tool {
                    code: "SCATTER_UNKNOWN_COMPLETION",
                    ..
                })
            ));
            assert!(run.observation.cleanup_trace.is_empty());
            let mut pending = run.pending.unwrap();
            assert_eq!(pending.retained_arrays(), 2);
            let resources = pending.resources.take().unwrap();
            let weak = resources.output_weak();
            pending.resources = Some(resources);
            drop(pending);
            assert!(weak.upgrade().is_some());
        }
    }
    #[test]
    fn issue_m4_05_gather_scatter_frontier_scatter_runtime_cost_and_recipe() {
        let maximum = cost(4096, 4096, 4096, Budget::default()).unwrap();
        assert_eq!(
            maximum,
            Cost {
                bytes: 135168,
                visits: 36864
            }
        );
        assert!(
            size_of::<Payload>()
                + size_of::<Completion>()
                + size_of::<Pending>()
                + size_of::<crate::mir_eval::scatter::Quarantined>()
                <= METADATA_BYTES
        );
        assert!(cost(usize::MAX, 0, 0, Budget::default()).is_none());
        assert!(
            cost(
                4096,
                4096,
                4096,
                Budget {
                    bytes: maximum.bytes - 1,
                    visits: MAX_VISITS
                }
            )
            .is_none()
        );
        assert!(
            cost(
                4096,
                4096,
                4096,
                Budget {
                    bytes: MAX_STAGING_BYTES,
                    visits: maximum.visits - 1
                }
            )
            .is_none()
        );
        for (destination, count) in [
            ("i", 6),
            ("2u32*i", 8),
            ("2u32*i+1u32", 10),
            ("1u32+(i*2u32)", 10),
            ("(i)", 6),
        ] {
            let source = String::from_utf8(SOURCE.into())
                .unwrap()
                .replace("2u32*i", destination)
                .replace("5u32,0u32", "6u32,0u32");
            let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
            let m = mir::lower(&checked).unwrap();
            let plan = Plan::bind(source.as_bytes().into(), &checked, "scatter", &m).unwrap();
            assert_eq!(plan.tick_spans.len(), count);
            assert_eq!(plan.loop_ticks(), Some(3 * count as u64));
        }
    }
}

#[cfg(test)]
mod diagnostic_tests {
    use super::*;
    const SOURCE: &[u8] = b"fn scatter(a: read Array<u32>) -> Array<u32> { var out = filled_u32(5u32,0u32); for i in 0u32..3u32 { out[2u32*i] = a[i]; } return out; } fn main() -> i32 { let a: Array<u32> = [7u32,11u32,19u32]; let out = scatter(a); return 7; }";

    #[test]
    fn issue_ai_01_machine_diagnostics_scatter_publication_faults() {
        let checked = check::check(front::parse(SOURCE).unwrap(), true).unwrap();
        let program = mir::lower(&checked).unwrap();
        let mut injected = Vec::new();
        for fault in 0..9 {
            let producer: Producer = if fault == 8 {
                |_, _| Err(Error::Allocation)
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
                            Plan::bind(SOURCE.into(), &checked, "scatter", &program).unwrap();
                        stale.tick_spans.push(Span { start: 0, end: 1 });
                        Rc::get_mut(&mut c.payload).unwrap().plan = Rc::new(stale);
                    }
                    4 => Rc::get_mut(&mut c.payload).unwrap().ticks += 1,
                    5 => Rc::get_mut(&mut c.payload).unwrap().input[0] ^= 1,
                    6 => Rc::get_mut(&mut c.payload).unwrap().output[1] = 42,
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
            injected.push((*code, error.to_diagnostics_json("scatter.inv", SOURCE)));
        }
        let injected_codes = injected.iter().map(|(code, _)| *code).collect::<Vec<_>>();
        assert_eq!(
            injected_codes,
            [
                "SCATTER_UNKNOWN_COMPLETION",
                "SCATTER_FAILED_COMPLETION",
                "SCATTER_SOURCE_BINDING",
                "SCATTER_FUEL_RECIPE",
                "SCATTER_LOOP_BINDING",
                "SCATTER_STALE_COMPLETION",
                "SCATTER_BAD_COMPLETION",
                "SCATTER_POST_ENTRY",
            ]
        );
        // Only SCATTER_FUEL remains constructor-only: it is dead code, because
        // SCATTER_LOOP_BINDING first requires ticks <= fuel_limit - steps_used, so
        // steps_used + ticks cannot overflow. SCATTER_OWNER, SCATTER_LOAN and SCATTER_COUNT are injected for
        // real by `mir_eval::scatter::tests::issue_ai_01_machine_diagnostics_scatter_publication_harness_faults`.
        let constructed = ["SCATTER_FUEL"].map(|code| {
            (
                code,
                failure(code).to_diagnostics_json("scatter.inv", SOURCE),
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
                e["fault_test"] == "issue_ai_01_machine_diagnostics_scatter_publication_faults"
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
