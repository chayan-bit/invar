//! Synchronous, source-bound CPU histograms. No native/device publication.
use super::{Budget, Contract, Cost, Error, MAX_VISITS, bounded};
use crate::{
    check,
    eval::{EvalError, Observation},
    front::{self, Expr, ExprKind, Place, Span, StmtKind},
    general_net::compiled::FenceObservation,
    mir,
    mir_eval::{self, LiveLoop, LoopOverride},
};
use std::{cell::RefCell, rc::Rc};

pub(crate) fn failure(code: &'static str) -> EvalError {
    EvalError::Tool {
        code,
        message: "source-bound CPU histogram failed; zero publication and no replay".into(),
    }
}
fn ticks(e: &Expr, out: &mut Vec<Span>) -> Result<(), Error> {
    if let ExprKind::Group(inner) = &e.kind {
        return ticks(inner, out);
    }
    out.push(e.span);
    match &e.kind {
        ExprKind::Name(_) => Ok(()),
        ExprKind::Index { base, index } => {
            ticks(base, out)?;
            ticks(index, out)
        }
        ExprKind::Call { name, args } if name == "wrapping_add" && args.len() == 2 => {
            for arg in args {
                ticks(arg, out)?;
            }
            Ok(())
        }
        _ => Err(Error::Unrecognized),
    }
}

pub(crate) struct Plan {
    pub(crate) contract: Contract,
    pub(crate) program: Rc<mir::Program>,
    pub(crate) function_index: usize,
    pub(crate) head_index: usize,
    pub(crate) condition: mir::ValueId,
    pub(crate) left: mir::ValueId,
    pub(crate) right: mir::ValueId,
    pub(crate) values: String,
    pub(crate) indices: String,
    pub(crate) bins: String,
    pub(crate) output: String,
    tick_spans: Vec<Span>,
}
impl Plan {
    pub(crate) fn bind(
        source: &[u8],
        checked: &check::CheckedProgram,
        entry: &str,
        program: Rc<mir::Program>,
    ) -> Result<Self, Error> {
        let contract = Contract::from_checked(source, checked, entry)?;
        // Cold ordinary compiler work retains its frontend resource limits.
        let verified = contract.verified_mir()?;
        if verified.contract_digest() != program.contract_digest() {
            return Err(Error::Binding);
        }
        let f = checked.program.function(entry).ok_or(Error::Unrecognized)?;
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
        let function_index = program
            .functions
            .iter()
            .position(|f| f.name == entry)
            .ok_or(Error::Binding)?;
        let function = &program.functions[function_index];
        let (head_index, head, body, condition) = function
            .blocks
            .iter()
            .enumerate()
            .find_map(|(id, b)| match b.end {
                Some(mir::Terminator::Branch {
                    yes,
                    condition,
                    span,
                    ..
                }) if span == contract.loop_span() => function
                    .blocks
                    .get(yes)
                    .map(|body| (id, b, body, condition)),
                _ => None,
            })
            .ok_or(Error::Binding)?;
        if head
            .instructions
            .iter()
            .any(|i| matches!(i.kind, mir::Operation::Tick))
            || !matches!(body.end,Some(mir::Terminator::Goto{target,..}) if target==head_index)
            || !body
                .instructions
                .iter()
                .filter(|i| matches!(i.kind, mir::Operation::Tick))
                .map(|i| i.span)
                .eq(tick_spans.iter().copied())
        {
            return Err(Error::Binding);
        }
        let (left, right) = head
            .instructions
            .iter()
            .find_map(|i| match i.kind {
                mir::Operation::Binary {
                    op: front::BinaryOp::Lt,
                    left,
                    right,
                    ..
                } if i.output.as_ref().is_some_and(|(id, _)| *id == condition) => {
                    Some((left, right))
                }
                _ => None,
            })
            .ok_or(Error::Binding)?;
        Ok(Self {
            contract,
            program,
            function_index,
            head_index,
            condition,
            left,
            right,
            values: f.params[0].name.clone(),
            indices: f.params[1].name.clone(),
            bins: f.params[2].name.clone(),
            output: output.clone(),
            tick_spans,
        })
    }
    pub(crate) fn loop_ticks(&self, i: usize) -> Option<u64> {
        u64::try_from(i)
            .ok()?
            .checked_mul(u64::try_from(self.tick_spans.len()).ok()?)
    }
    pub(crate) fn function(&self) -> &mir::Function {
        &self.program.functions[self.function_index]
    }
}

// The ledger owns all plans strongly throughout the run. Their source, checked
// AST, net, proof and ordinary MIR have no interior mutability. Completion may
// neither replace this authority nor mutate it using Rc::get_mut. Pointer
// equality here refers to retained live objects; it is never an activation ID.
pub(crate) struct Ledger {
    pub(crate) plans: Vec<Rc<Plan>>,
    pub(crate) reserved: usize,
    pub(crate) decision_visits: usize,
    limit: usize,
    pub(crate) attempts: Vec<Rc<RefCell<bounded::Meter>>>,
}
impl Ledger {
    pub(crate) fn new(plans: Vec<Rc<Plan>>, budget: Budget) -> Self {
        Self {
            plans,
            reserved: 0,
            decision_visits: 0,
            limit: budget.visits.min(MAX_VISITS),
            attempts: Vec::new(),
        }
    }
    // Charge before discovery, even if no plan or a later guard refuses. Never
    // refund either decision work or the complete conservative reservation.
    fn charge(&mut self, n: usize) -> bool {
        let Some(next) = self.reserved.checked_add(n).filter(|v| *v <= self.limit) else {
            return false;
        };
        self.reserved = next;
        true
    }
    fn discover(&mut self, live: &LiveLoop<'_>) -> Option<Rc<Plan>> {
        for index in 0..self.plans.len() {
            let p = &self.plans[index];
            let count = p
                .function()
                .name
                .len()
                .checked_add(live.function.len())?
                .checked_add(8)?;
            if !self.charge(count) {
                return None;
            }
            self.decision_visits += count;
            let p = &self.plans[index];
            if p.function().name == live.function && p.contract.loop_span() == live.span {
                return Some(p.clone());
            }
        }
        None
    }
    pub(crate) fn permits(&self, plan: &Rc<Plan>) -> bool {
        // Runtime admits at most one plan: immutable cold authority comparison
        // is constant work, not an unmetered search through arbitrary context.
        matches!(self.plans.as_slice(), [original] if Rc::ptr_eq(original, plan))
    }
    pub(crate) fn permits_attempt(
        &self,
        plan: &Rc<Plan>,
        meter: &Rc<RefCell<bounded::Meter>>,
        cost: Cost,
    ) -> bool {
        self.permits(plan)
            && matches!(self.attempts.as_slice(), [original] if Rc::ptr_eq(original, meter))
            && self.reserved >= cost.visits
            && self.reserved <= self.limit
    }
}

pub(crate) struct Payload {
    pub(crate) plan: Rc<Plan>,
    pub(crate) ledger: Rc<RefCell<Ledger>>,
    pub(crate) meter: Rc<RefCell<bounded::Meter>>,
    pub(crate) owners: [Rc<RefCell<Vec<u32>>>; 3],
    pub(crate) data: bounded::RuntimeData,
    pub(crate) cost: Cost,
    pub(crate) ticks: u64,
}
pub(crate) struct Completion {
    pub(crate) payload: Rc<Payload>,
    pub(crate) status: FenceObservation,
    pub(crate) quarantine: Rc<RefCell<Option<Pending>>>,
}
pub struct Pending {
    pub(crate) resources: Option<mir_eval::histogram::Quarantined>,
}
impl Pending {
    pub fn retained_arrays(&self) -> usize {
        self.resources.as_ref().map_or(0, |r| r.retained_arrays())
    }
}
impl Drop for Pending {
    fn drop(&mut self) {
        if let Some(r) = self.resources.take() {
            std::mem::forget(r);
        }
    }
}
pub struct Run {
    pub observation: Observation,
    pub begun: usize,
    pub completed: usize,
    pub refusals: Vec<&'static str>,
    pub pending: Option<Pending>,
    /// Complete conservative reservations across all reached attempts, including failures.
    pub reserved_visits: usize,
    /// Work actually metered before execution (unused reservations are not refunded).
    pub visited: usize,
}
fn refuse(
    refusals: &mut Vec<&'static str>,
    reason: &'static str,
) -> Result<Option<Completion>, EvalError> {
    if refusals.len() < 64 {
        refusals.push(reason);
    }
    Ok(None)
}
type Producer = fn(
    &Contract,
    bounded::RuntimePrepared,
    &mut bounded::Meter,
) -> Result<bounded::RuntimeData, Error>;
pub(crate) struct Progress<'a> {
    pub(crate) begun: &'a mut usize,
    pub(crate) refusals: &'a mut Vec<&'static str>,
}
pub(crate) fn prepare(
    plan: Rc<Plan>,
    ledger: Rc<RefCell<Ledger>>,
    live: &LiveLoop<'_>,
    budget: Budget,
    quarantine: Rc<RefCell<Option<Pending>>>,
    progress: Progress<'_>,
    producer: Producer,
) -> Result<Option<Completion>, EvalError> {
    let Progress { begun, refusals } = progress;
    // Prepaid header covers scalar lookups, extent/cost arithmetic, names,
    // RefCell/Rc ownership/loan checks, and fixed receipt metadata.
    let precheck = 64usize
        .checked_add(plan.values.len())
        .and_then(|v| v.checked_add(plan.indices.len()))
        .and_then(|v| v.checked_add(plan.output.len()))
        .and_then(|v| v.checked_add(plan.bins.len()))
        .and_then(|v| v.checked_add(live.function.len()))
        .ok_or_else(|| failure("HISTOGRAM_BUDGET"))?;
    if !ledger.borrow_mut().charge(precheck) {
        return refuse(refusals, "histogram_run_budget");
    }
    ledger.borrow_mut().decision_visits += precheck;
    if !ledger.borrow().permits(&plan) {
        return refuse(refusals, "histogram_source_authority");
    }
    let Some(a) = live.lengths.get(&plan.values).copied() else {
        return refuse(refusals, "histogram_values_extent");
    };
    let Some(i) = live.lengths.get(&plan.indices).copied() else {
        return refuse(refusals, "histogram_indices_extent");
    };
    let Some(n) = live.scalars.get(&plan.bins).copied() else {
        return refuse(refusals, "histogram_bins");
    };
    let cost = match plan.contract.runtime_cost(a, i, n as usize, budget) {
        Ok(c) => c,
        Err(_) => return refuse(refusals, "histogram_complete_budget"),
    };
    // Full remainder, including post-entry proof/copy, is reserved before any
    // proof stream, snapshot, candidate work or publication. Failed attempts
    // keep this reservation for the rest of the run.
    if !ledger.borrow_mut().charge(cost.visits) {
        return refuse(refusals, "histogram_run_budget");
    }
    let meter = Rc::new(RefCell::new(bounded::Meter {
        used: 0,
        limit: cost.visits,
    }));
    ledger
        .borrow_mut()
        .attempts
        .try_reserve(1)
        .map_err(|_| failure("HISTOGRAM_PRIVATE_ALLOCATION"))?;
    ledger.borrow_mut().attempts.push(meter.clone());
    let mut m = meter.borrow_mut();
    m.visit(
        plan.contract
            .runtime_header()
            .map_err(|_| failure("HISTOGRAM_BUDGET"))?,
    )
    .map_err(|_| failure("HISTOGRAM_BUDGET"))?;
    if plan.function().name != live.function
        || plan.contract.loop_span() != live.span
        || i != live.iterations
        || live.owned(&plan.values) != Some(false)
        || live.owned(&plan.indices) != Some(false)
        || live.owned(&plan.output) != Some(true)
        || live.lengths.get(&plan.output) != Some(&(n as usize))
        || live.same_object(&plan.values, &plan.output)
        || live.same_object(&plan.indices, &plan.output)
    {
        return refuse(refusals, "histogram_owner_or_region");
    }
    let Some(ticks) = plan.loop_ticks(i).filter(|t| *t <= live.fuel_remaining) else {
        return refuse(refusals, "histogram_source_fuel");
    };
    let (Some(values), Some(indices), Some(output)) = (
        live.array(&plan.values),
        live.array(&plan.indices),
        live.array(&plan.output),
    ) else {
        return refuse(refusals, "histogram_loan");
    };
    for &zero in output.iter() {
        m.visit(1).map_err(|_| failure("HISTOGRAM_BUDGET"))?;
        if zero != 0 {
            return refuse(refusals, "histogram_fresh_output");
        }
    }
    let prepared = match plan.contract.runtime_prepare(&values, &indices, n, &mut m) {
        Ok(p) => p,
        Err(_) => return refuse(refusals, "histogram_proof_bounds_or_snapshot"),
    };
    let (Some(v), Some(ix), Some(out)) = (
        live.array_handle(&plan.values),
        live.array_handle(&plan.indices),
        live.array_handle(&plan.output),
    ) else {
        return refuse(refusals, "histogram_owner");
    };
    *begun = begun
        .checked_add(1)
        .ok_or_else(|| failure("HISTOGRAM_COUNT"))?;
    // Entry precedes every trace/group/output allocation and all grouped work.
    let data =
        producer(&plan.contract, prepared, &mut m).map_err(|_| failure("HISTOGRAM_POST_ENTRY"))?;
    drop(m);
    Ok(Some(Completion {
        payload: Rc::new(Payload {
            plan,
            ledger,
            meter,
            owners: [v, ix, out],
            data,
            cost,
            ticks,
        }),
        status: FenceObservation::Complete,
        quarantine,
    }))
}

/// Scoped ordinary-source execution. At most one recognized histogram plan is
/// admitted. The stronger run-wide work cap never resets at calls or retries.
/// Initial ordinary compilation/storage uses existing compiler resource limits.
pub fn run_source(source: &[u8], fuel: u64, budget: Budget) -> Result<Run, Error> {
    run_using(source, fuel, budget, |_| {}, Contract::runtime_produce)
}
fn run_using(
    source: &[u8],
    fuel: u64,
    budget: Budget,
    mut receipt: impl FnMut(&mut Completion),
    producer: Producer,
) -> Result<Run, Error> {
    let checked =
        check::check(front::parse(source).map_err(Error::Source)?, true).map_err(Error::Source)?;
    let program =
        Rc::new(mir::lower(&checked).map_err(|e| Error::Net(crate::general_net::Error::Mir(e)))?);
    let mut plans = Vec::new();
    for f in &checked.program.functions {
        match Plan::bind(source, &checked, &f.name, program.clone()) {
            Ok(p) => {
                plans.try_reserve(1).map_err(|_| Error::Allocation)?;
                plans.push(Rc::new(p));
            }
            Err(Error::Unrecognized) => {}
            Err(e) => return Err(e),
        }
    }
    // Multiple plans remain ordinary source. Plan discovery context is cold
    // compiler storage; dynamic attempts/receipts are limited to one retained
    // attempt by the complete-proof reservation (>= 32769 visits).
    if plans.len() != 1 {
        plans.clear();
    }
    let ledger = Rc::new(RefCell::new(Ledger::new(plans, budget)));
    let quarantine = Rc::new(RefCell::new(None));
    let mut begun = 0;
    let mut refusals = Vec::new();
    refusals
        .try_reserve_exact(64)
        .map_err(|_| Error::Allocation)?;
    let mut provider = |live: &LiveLoop<'_>| {
        let Some(plan) = ledger.borrow_mut().discover(live) else {
            return Ok(None);
        };
        let Some(mut completion) = prepare(
            plan,
            ledger.clone(),
            live,
            budget,
            quarantine.clone(),
            Progress {
                begun: &mut begun,
                refusals: &mut refusals,
            },
            producer,
        )?
        else {
            return Ok(None);
        };
        receipt(&mut completion);
        Ok(Some(LoopOverride {
            histogram: Some(completion),
            iterations: live.iterations,
            ..Default::default()
        }))
    };
    let (observation, completed) =
        mir_eval::observe_with_scheduler(&program, fuel, Some(&mut provider));
    let l = ledger.borrow();
    let visited = l
        .attempts
        .iter()
        .try_fold(l.decision_visits, |n, m| n.checked_add(m.borrow().used))
        .ok_or(Error::Budget)?;
    let pending = quarantine.borrow_mut().take();
    Ok(Run {
        observation,
        begun,
        completed,
        refusals,
        pending,
        reserved_visits: l.reserved,
        visited,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    const SOURCE: &str = "fn h(a: read Array<u32>, x: read Array<u32>, n: u32) -> Array<u32> { var o = filled_u32(n,0u32); for i in 0u32..len(x) { o[x[i]] = wrapping_add(o[x[i]],a[i]); } return o; }";
    #[test]
    fn issue_m4_05_gather_scatter_frontier_histogram_runtime_stale_source_proof_before_entry() {
        for fault in 0..7 {
            let checked = check::check(front::parse(SOURCE.as_bytes()).unwrap(), false).unwrap();
            let program = Rc::new(mir::lower(&checked).unwrap());
            let mut p = Plan::bind(SOURCE.as_bytes(), &checked, "h", program).unwrap();
            match fault {
                0 => p.contract.source_hash[0] ^= 1,
                1 => p.contract.source_contract[0] ^= 1,
                2 => p.contract.source[0] = b' ',
                3 => p.contract.law = crate::laws::Law::WrappingProduct,
                4 => p.contract.seal.visits += 1,
                5 => p.contract.runtime_names += 1,
                _ => {
                    let changed = SOURCE.replace("wrapping_add", "wrapping_mul");
                    let other =
                        check::check(front::parse(changed.as_bytes()).unwrap(), false).unwrap();
                    p.contract.net =
                        crate::general_net::Net::lower(&other, Default::default()).unwrap();
                    p.contract.proof = p.contract.net.verify().unwrap();
                }
            }
            let mut meter = bounded::Meter {
                used: 0,
                limit: 65536,
            };
            assert!(
                p.contract
                    .runtime_prepare(&[7], &[0], 1, &mut meter)
                    .is_err()
            );
            assert!(meter.used <= 65536);
        }
    }
    #[test]
    fn issue_m4_05_gather_scatter_frontier_histogram_runtime_main_context_gate_is_explicit() {
        {
            let main =
                "fn main() -> i32 { let a: Array<u32> = [0u32]; let o = h(a,a,1u32); return 7; }";
            let s = format!("{SOURCE} {main}");
            let checked = check::check(front::parse(s.as_bytes()).unwrap(), true).unwrap();
            let program = Rc::new(mir::lower(&checked).unwrap());
            let p = Plan::bind(s.as_bytes(), &checked, "h", program).unwrap();
            println!(
                "ordinary-main source bytes={}, proof visits={}, reserved remainder result={:?}",
                s.len(),
                p.contract.seal.visits,
                p.contract.runtime_cost(1, 1, 1, Budget::default())
            );
            assert!(p.contract.runtime_cost(1, 1, 1, Budget::default()).is_ok());
            let run = run_source(s.as_bytes(), 10000, Budget::default()).unwrap();
            assert_eq!(run.completed, 1);
            assert_eq!(run.begun, 1);
        }
    }
}
#[cfg(test)]
mod source_failures {
    use super::*;
    const SOURCE:&[u8]=b"fn h(a: read Array<u32>, x: read Array<u32>, n: u32) -> Array<u32> { var o = filled_u32(n,0u32); for i in 0u32..len(x) { o[x[i]] = wrapping_add(o[x[i]],a[i]); } return o; } fn main() -> i32 { let a: Array<u32> = [0u32]; let o = h(a,a,1u32); return 7; }";
    #[test]
    fn issue_m4_05_gather_scatter_frontier_histogram_runtime_reached_source_errors_never_replay() {
        let ordinary = run_source(
            SOURCE,
            10000,
            Budget {
                bytes: 0,
                visits: 0,
            },
        )
        .unwrap();
        assert_eq!(ordinary.completed, 0);
        let error = run_using(
            SOURCE,
            10000,
            Default::default(),
            |_| {},
            |_, _, _| Err(Error::Allocation),
        )
        .unwrap();
        assert_eq!(error.begun, 1);
        assert_eq!(error.completed, 0);
        assert!(matches!(
            error.observation.result,
            Err(EvalError::Tool {
                code: "HISTOGRAM_POST_ENTRY",
                ..
            })
        ));
        assert!(error.observation.steps_used < ordinary.observation.steps_used);
        assert!(error.observation.cleanup_trace.is_empty());
        assert!(error.pending.is_none());
        for status in [
            FenceObservation::FailedQuiescent,
            FenceObservation::Unknown,
            FenceObservation::Pending,
        ] {
            let run = run_using(
                SOURCE,
                10000,
                Default::default(),
                |c| {
                    c.status = status;
                    Rc::get_mut(&mut c.payload).unwrap().ticks += 1;
                },
                Contract::runtime_produce,
            )
            .unwrap();
            assert_eq!(run.begun, 1);
            assert_eq!(run.completed, 0);
            assert_eq!(run.observation.steps_used, error.observation.steps_used);
            assert!(run.observation.cleanup_trace.is_empty());
            assert!(matches!(
                run.observation.result,
                Err(EvalError::Tool { .. })
            ));
            assert_eq!(
                run.pending.is_some(),
                matches!(
                    status,
                    FenceObservation::Unknown | FenceObservation::Pending
                )
            );
        }
    }
}

#[cfg(test)]
mod diagnostic_tests {
    use super::*;
    const SOURCE: &[u8] = b"fn h(a: read Array<u32>, x: read Array<u32>, n: u32) -> Array<u32> { var o = filled_u32(n,0u32); for i in 0u32..len(x) { o[x[i]] = wrapping_add(o[x[i]],a[i]); } return o; } fn main() -> i32 { let a: Array<u32> = [0u32]; let o = h(a,a,1u32); return 7; }";

    #[test]
    fn issue_ai_01_machine_diagnostics_histogram_publication_faults() {
        let checked = check::check(front::parse(SOURCE).unwrap(), true).unwrap();
        let mut injected = Vec::new();
        for fault in 0..14 {
            let producer: Producer = if fault == 10 {
                |_, _, _| Err(Error::Allocation)
            } else {
                Contract::runtime_produce
            };
            let run = run_using(
                SOURCE,
                10000,
                Budget::default(),
                |c| match fault {
                    0 => c.status = FenceObservation::Unknown,
                    1 => c.status = FenceObservation::FailedQuiescent,
                    2 => {
                        let mut meter = c.payload.meter.borrow_mut();
                        meter.used = meter.limit;
                    }
                    3 => {
                        // A plan bound to another lowering is not the admitted plan.
                        let program = Rc::new(mir::lower(&checked).unwrap());
                        let stale = Plan::bind(SOURCE, &checked, "h", program).unwrap();
                        Rc::get_mut(&mut c.payload).unwrap().plan = Rc::new(stale);
                    }
                    4 => Rc::get_mut(&mut c.payload).unwrap().ticks += 1,
                    5 => {
                        Rc::get_mut(&mut c.payload).unwrap().owners[0] =
                            Rc::new(RefCell::new(vec![0]));
                    }
                    6 => Rc::get_mut(&mut c.payload).unwrap().cost.bytes += 1,
                    7 => Rc::get_mut(&mut c.payload).unwrap().data.proof_visits += 1,
                    8 => c.payload.owners[2].borrow_mut()[0] = 1,
                    11 => {
                        // A meter already mutably borrowed. It is not the
                        // ledger's attempt meter, so post-run accounting still
                        // reads the real one.
                        let held = Rc::new(RefCell::new(bounded::Meter { used: 0, limit: 0 }));
                        std::mem::forget(held.borrow_mut());
                        Rc::get_mut(&mut c.payload).unwrap().meter = held;
                    }
                    // Live borrows of the entered output and values owners.
                    12 => std::mem::forget(c.payload.owners[2].borrow()),
                    13 => std::mem::forget(c.payload.owners[0].borrow_mut()),
                    _ => {}
                },
                producer,
            )
            .unwrap();
            let Err(error @ EvalError::Tool { code, .. }) = &run.observation.result else {
                assert_eq!(
                    (fault, run.completed),
                    (9, 1),
                    "fault {fault} must fail closed"
                );
                continue;
            };
            assert_eq!(run.completed, 0);
            assert!(run.observation.cleanup_trace.is_empty());
            injected.push((*code, error.to_diagnostics_json("histogram.inv", SOURCE)));
        }
        let injected_codes = injected.iter().map(|(code, _)| *code).collect::<Vec<_>>();
        assert_eq!(
            injected_codes,
            [
                "HISTOGRAM_UNKNOWN_COMPLETION",
                "HISTOGRAM_FAILED_COMPLETION",
                "HISTOGRAM_BUDGET",
                "HISTOGRAM_SOURCE_BINDING",
                "HISTOGRAM_LOOP_BINDING",
                "HISTOGRAM_OWNER",
                "HISTOGRAM_EXTENT",
                "HISTOGRAM_BAD_COMPLETION",
                "HISTOGRAM_FRESH_OUTPUT",
                "HISTOGRAM_POST_ENTRY",
                "HISTOGRAM_METER",
                "HISTOGRAM_LOAN",
                "HISTOGRAM_LOAN",
            ]
        );
        // Constructor-only, each with its reason:
        // - HISTOGRAM_FUEL is dead code, because HISTOGRAM_LOOP_BINDING first
        //   requires ticks <= fuel_limit - steps_used;
        // - HISTOGRAM_PRIVATE_ALLOCATION needs an allocator failure of the
        //   bounded private attempt reservation.
        // HISTOGRAM_COUNT (both counters) is injected for real by
        // `mir_eval::histogram::atomic_tests::issue_ai_01_machine_diagnostics_histogram_publication_harness_faults`.
        let constructed = ["HISTOGRAM_FUEL", "HISTOGRAM_PRIVATE_ALLOCATION"].map(|code| {
            (
                code,
                failure(code).to_diagnostics_json("histogram.inv", SOURCE),
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
                e["fault_test"] == "issue_ai_01_machine_diagnostics_histogram_publication_faults"
            })
            .map(|e| e["code"].as_str().unwrap())
            .collect::<Vec<_>>();
        registered.sort_unstable();
        let mut covered = injected_codes.clone();
        covered.extend(constructed.iter().map(|(code, _)| *code));
        covered.sort_unstable();
        covered.dedup();
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
