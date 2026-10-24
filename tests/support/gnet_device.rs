//! GNET-03 shared device harness: drive the ordinary-source ABI5 route on one
//! backend (CPU group model, CUDA or Metal) and compare every observation word
//! with the bounded CPU net oracle. The corpus runs through `Plan::from_source`
//! exactly as the `gnet-device` command does; nothing is selected by file name.
#![allow(dead_code)]

use invar::{
    check,
    eval::{self, EvalError, TrapCategory, Value},
    front,
    general_net::{
        compiled::{DeviceConfig, Status, Target},
        device_host::{self, Backend, Limits, Observation, Plan, Progress},
        oracle,
    },
};
use std::{
    collections::BTreeMap,
    path::Path,
    time::{Duration, Instant},
};

#[path = "../../test-support/corpus_sample.rs"]
pub mod sample;

pub const FUEL: u64 = 100_000;
const TRANSITIONS: u64 = 1_000_000;
/// One submission is at most 64 rounds of this root quantum; children advance
/// their own private quantum between the same barriers.
const QUANTUM: u64 = 4096;
const DEADLINE: Duration = Duration::from_secs(10);

pub struct Source {
    pub name: String,
    pub bytes: Vec<u8>,
    pub checked: check::CheckedProgram,
}

/// The GNET-02 runnable corpus: every source under tests/fixtures and examples
/// that parses and checks as an executable program, with its exact bytes.
pub fn corpus() -> Vec<Source> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    sample::walk(&root.join("tests/fixtures"), &mut files);
    sample::walk(&root.join("examples"), &mut files);
    files
        .into_iter()
        .filter_map(|path| {
            let bytes = std::fs::read(&path).ok()?;
            let checked = check::check(front::parse(&bytes).ok()?, true).ok()?;
            // Fixture names use `/` on every host so name filters match on Windows.
            let name = path
                .strip_prefix(root)
                .unwrap()
                .display()
                .to_string()
                .replace('\\', "/");
            Some(Source {
                name,
                bytes,
                checked,
            })
        })
        .collect()
}

pub fn config(
    fuel: u64,
    trace_events: usize,
    quantum: u64,
    rounds: u64,
    capacity: usize,
) -> DeviceConfig {
    DeviceConfig {
        fuel,
        transitions: TRANSITIONS,
        trace_events,
        quantum,
        rounds,
        capacity,
    }
}

pub fn oracle_limits(plan: &Plan, config: DeviceConfig) -> oracle::Limits {
    oracle::Limits {
        fuel: config.fuel,
        transitions: config.transitions,
        trace_events: config.trace_events,
        array_elements: plan.artifact().owned_layout().unwrap().elements(),
    }
}

pub fn reference(plan: &Plan, args: &[Value], config: DeviceConfig) -> oracle::Observation {
    oracle::observe_call(
        plan.net(),
        plan.proof(),
        plan.entry(),
        args,
        oracle_limits(plan, config),
    )
    .unwrap()
}

/// Every submission is fenced before its checkpoint is read; a missed
/// deadline cancels, never replays, and fails the case.
pub fn drive<B: Backend>(
    plan: &Plan,
    backend: &mut B,
    args: &[Value],
    config: DeviceConfig,
) -> Result<(Box<Observation>, u64), device_host::Error> {
    let mut session = plan.prepare(backend, args, config)?;
    let observation = loop {
        session.submit()?;
        let begun = Instant::now();
        let progress = loop {
            match session.poll()? {
                Progress::Pending if begun.elapsed() < DEADLINE => {
                    std::thread::sleep(Duration::from_micros(200))
                }
                Progress::Pending => {
                    session.cancel();
                    panic!("device completion deadline exceeded");
                }
                progress => break progress,
            }
        };
        if let Progress::Complete(observation) = progress {
            break observation;
        }
    };
    let epochs = session.submitted();
    // Quiescence: every submission completed and nothing further is accepted.
    assert_eq!(epochs, session.completed());
    assert!(session.submit().is_err());
    drop(session.reclaim()?);
    Ok((observation, epochs))
}

/// Device storage exhaustion that the oracle's live-element arena does not
/// model (total allocation IDs, cleanup records). It is a typed backend limit,
/// never a source result, and is counted separately from agreement.
pub fn device_storage_limit(status: Status) -> bool {
    matches!(status, Status::AllocationLimit | Status::CleanupLimit)
}

fn status(reference: &oracle::Observation) -> (Status, Option<(usize, usize)>) {
    match &reference.semantic.result {
        Ok(_) => (Status::Success, None),
        Err(EvalError::Trap { category, span }) => (
            match category {
                TrapCategory::IntegerOverflow => Status::IntegerOverflow,
                TrapCategory::DivisionByZero => Status::DivisionByZero,
                TrapCategory::IndexOutOfBounds => Status::IndexOutOfBounds,
                other => panic!("trap {other:?} has no ABI5 status"),
            },
            Some((span.start, span.end)),
        ),
        Err(EvalError::EvaluationLimit { .. }) => (Status::FuelLimit, Some((0, 0))),
        Err(EvalError::Tool {
            code: "CALL_DEPTH_LIMIT",
            ..
        }) => (Status::CallDepthLimit, Some((0, 0))),
        Err(EvalError::Tool {
            code: "ARRAY_SIZE_LIMIT",
            ..
        }) => (Status::ArraySizeLimit, Some((0, 0))),
        Err(_) => (
            match reference.failure {
                Some(oracle::ToolFailure::TransitionLimit) => Status::TransitionLimit,
                Some(oracle::ToolFailure::TraceLimit) => Status::TraceLimit,
                Some(oracle::ToolFailure::Allocation) => Status::HeapLimit,
                other => panic!("oracle failure {other:?} has no ABI5 status"),
            },
            Some((0, 0)),
        ),
    }
}

pub fn encode(event: &oracle::Event) -> [u64; 7] {
    [
        u64::from(event.kind == oracle::EventKind::Complete),
        event.frame,
        event.activation,
        event.token.block.function as u64,
        event.token.block.block as u64,
        event.token.position as u64,
        event.rule as u64,
    ]
}

/// Exact agreement: status, value, first trap span, source steps, physical
/// transitions, frame peak, every ordered Demand/Complete event and the
/// completed normal cleanup prefix. A failed root publishes no value.
pub fn agree(context: &str, plan: &Plan, got: &Observation, reference: &oracle::Observation) {
    let (expected, span) = status(reference);
    assert_eq!(got.status, expected, "{context}: status");
    match (&reference.semantic.result, span) {
        (Ok(value), _) => {
            assert_eq!(got.result.as_ref(), Some(value), "{context}: value");
            assert_eq!(got.span, None, "{context}");
        }
        (Err(_), Some((start, end))) => {
            assert_eq!(got.result, None, "{context}: no publication on failure");
            let span = got.span.expect("failure span");
            assert_eq!((span.start, span.end), (start, end), "{context}: span");
        }
        _ => unreachable!(),
    }
    assert_eq!(got.steps, reference.semantic.steps_used, "{context}: steps");
    assert_eq!(
        got.transitions, reference.transitions,
        "{context}: transitions"
    );
    assert_eq!(got.max_frames, reference.max_frames, "{context}: frames");
    assert_eq!(
        got.events.len(),
        reference.events.len(),
        "{context}: event count"
    );
    for (i, (a, e)) in got.events.iter().zip(&reference.events).enumerate() {
        assert_eq!(*a, encode(e), "{context}: event {i}");
    }
    let mir = plan.net().compiled_mir(plan.proof()).unwrap();
    let cleanup: Vec<(String, usize)> = got
        .cleanup
        .iter()
        .map(|[function, place, length, _frame]| {
            (
                mir.functions[*function as usize].declaration_order[*place as usize].clone(),
                *length as usize,
            )
        })
        .collect();
    let expected: Vec<(String, usize)> = reference
        .semantic
        .cleanup_trace
        .iter()
        .map(|e| (e.name.clone(), e.length))
        .collect();
    assert_eq!(cleanup, expected, "{context}: cleanup");
}

/// Typed frontier counters from the device state words.
#[derive(Default, Debug, Clone, Copy)]
pub struct Frontier {
    pub spawned: u64,
    pub committed: u64,
    pub replayed: u64,
    pub capacity_serialized: u64,
    pub owned_replays: u64,
    pub peak_width: u64,
}
impl Frontier {
    pub fn of(o: &Observation) -> Self {
        let s = o.private_state;
        Self {
            spawned: s[2],
            committed: s[3],
            replayed: s[5],
            capacity_serialized: s[7],
            owned_replays: s[9],
            peak_width: s[13],
        }
    }
}

/// Explicit denominator: every runnable source is device-executed with exact
/// agreement, refused before any submission with a typed class, or ended in a
/// typed device storage limit. Forced device mode has no fallback row.
#[derive(Default, Debug)]
pub struct Denominator {
    pub runnable: usize,
    pub executed: BTreeMap<&'static str, usize>,
    pub refused: BTreeMap<&'static str, usize>,
    pub storage_limit: usize,
    pub fallback: usize,
    /// Executed programs whose fork plan has at least one independent call site.
    pub candidate_programs: usize,
    pub frontier_programs: usize,
    pub committed_children: u64,
    pub peak_width: u64,
    pub epochs: u64,
}
impl Denominator {
    pub fn absorb(&mut self, other: Self) {
        self.runnable += other.runnable;
        for (k, v) in other.executed {
            *self.executed.entry(k).or_default() += v;
        }
        for (k, v) in other.refused {
            *self.refused.entry(k).or_default() += v;
        }
        self.storage_limit += other.storage_limit;
        self.fallback += other.fallback;
        self.candidate_programs += other.candidate_programs;
        self.frontier_programs += other.frontier_programs;
        self.committed_children += other.committed_children;
        self.peak_width = self.peak_width.max(other.peak_width);
        self.epochs += other.epochs;
    }
    pub fn executed_total(&self) -> usize {
        self.executed.values().sum()
    }
    pub fn refused_total(&self) -> usize {
        self.refused.values().sum()
    }
    pub fn print(&self, label: &str) {
        println!(
            "gnet_03 {label}: runnable={} device_executed={} {:?} refused_with_reason={} {:?} device_storage_limit={} fallback={} candidate_programs={} frontier_programs={} committed_children={} peak_private_width={} epochs={}",
            self.runnable,
            self.executed_total(),
            self.executed,
            self.refused_total(),
            self.refused,
            self.storage_limit,
            self.fallback,
            self.candidate_programs,
            self.frontier_programs,
            self.committed_children,
            self.peak_width,
            self.epochs
        );
        assert_eq!(
            self.runnable,
            self.executed_total() + self.refused_total() + self.storage_limit + self.fallback
        );
    }
}

fn outcome(reference: &oracle::Observation) -> &'static str {
    match &reference.semantic.result {
        Ok(_) => "agree_value",
        Err(EvalError::Trap { .. }) => "agree_trap",
        Err(EvalError::EvaluationLimit { .. }) => "agree_fuel_limit",
        Err(_) => "agree_tool_limit",
    }
}

/// Run the whole corpus on one backend kind. `fuel_short` additionally replays
/// every admitted program one source step short of its own use, which must end
/// in the exact fuel limit with only completed cleanup observable.
/// Shard `(index, count)` of the corpus, or of its `sample` if `sampled`.
pub fn corpus_denominator<B: Backend>(
    target: Target,
    label: &str,
    mut backend: impl FnMut() -> B,
    fuel_short: bool,
    shard: (usize, usize),
    sampled: bool,
) -> Denominator {
    let mut d = Denominator::default();
    for (_, source) in sample::keep(corpus(), sampled, |source| &source.name)
        .into_iter()
        .enumerate()
        .filter(|(i, _)| i % shard.1 == shard.0)
    {
        d.runnable += 1;
        let plan = match Plan::from_source(&source.bytes, "main", target, Limits::default()) {
            Ok(plan) => plan,
            Err(error) => {
                *d.refused
                    .entry(device_host::refusal_class(&error))
                    .or_default() += 1;
                continue;
            }
        };
        let sized = reference(&plan, &[], config(FUEL, 1_000_000, QUANTUM, 64, 2));
        // The trace buffer is sized to the oracle's own event count; a device
        // that emitted more would end in TraceLimit and fail agreement.
        let events = sized.events.len().max(1);
        let mut runs = vec![FUEL];
        if fuel_short && sized.semantic.steps_used > 1 {
            runs.push(sized.semantic.steps_used - 1);
        }
        let mut storage = false;
        let mut key = "";
        for fuel in runs {
            let cfg = config(fuel, events, QUANTUM, 64, 2);
            let expected = reference(&plan, &[], cfg);
            let context = format!("{label} {} fuel={fuel}", source.name);
            let (got, epochs) = drive(&plan, &mut backend(), &[], cfg)
                .unwrap_or_else(|e| panic!("{context}: {e:?}"));
            if device_storage_limit(got.status) {
                assert_eq!(got.result, None, "{context}");
                storage = true;
                break;
            }
            agree(&context, &plan, &got, &expected);
            if fuel == FUEL {
                // The oracle is itself checked against the reference evaluator.
                let eval = eval::observe(&source.checked, FUEL);
                assert_eq!(eval.steps_used, expected.semantic.steps_used, "{context}");
                let frontier = Frontier::of(&got);
                if !plan.artifact().fork_plan().unwrap().candidates.is_empty() {
                    d.candidate_programs += 1;
                }
                if frontier.spawned > 0 {
                    d.frontier_programs += 1;
                }
                d.committed_children += frontier.committed;
                d.peak_width = d.peak_width.max(frontier.peak_width);
                d.epochs += epochs;
                key = outcome(&expected);
            }
        }
        if storage {
            d.storage_limit += 1;
        } else {
            *d.executed.entry(key).or_default() += 1;
        }
    }
    d
}

/// One adversarial device run: runtime inputs and the exact device budgets.
pub struct Run {
    pub inputs: Vec<Value>,
    pub fuel: u64,
    pub capacity: usize,
    pub quantum: u64,
    pub rounds: u64,
}
pub struct Case {
    pub name: &'static str,
    pub source: &'static str,
    pub entry: &'static str,
    pub slots: usize,
    pub storage: invar::general_net::compiled::OwnedLimits,
    pub runs: Vec<Run>,
}
#[derive(Default, Debug)]
pub struct CaseSummary {
    pub runs: usize,
    pub statuses: BTreeMap<String, usize>,
    pub spawned: u64,
    pub committed: u64,
    pub replayed: u64,
    pub capacity_serialized: u64,
    pub owned_replays: u64,
    pub peak_width: u64,
    pub max_epochs: u64,
    /// Device storage exhaustion with the oracle's own successful result.
    pub storage_limited: usize,
}

fn run(inputs: Vec<Value>, fuel: u64, capacity: usize, quantum: u64, rounds: u64) -> Run {
    Run {
        inputs,
        fuel,
        capacity,
        quantum,
        rounds,
    }
}

fn default_storage() -> invar::general_net::compiled::OwnedLimits {
    invar::general_net::compiled::OwnedLimits::default()
}

/// Ordinary-source adversarial programs with runtime inputs: data-dependent
/// trip counts, divergent recursive frontiers, a worklist wider than its bound,
/// an early trap in one lane behind a slow or divergent older call, fuel cut
/// mid-round, device owners with traps and storage exhaustion.
pub fn adversarial() -> Vec<Case> {
    use Value::{I32, U32};
    let trips = "fn work(n: u32) -> u32 { var s = 0u32; var i = 0u32; while i < n { s = wrapping_add(s, wrapping_mul(i, i)); i = i + 1u32; } return s; }
fn run(a: u32, b: u32) -> u32 { let x = work(a); let y = work(b); return wrapping_add(x, y); }";
    let tree = "fn tree(n: u32, seed: u32) -> u32 {
    if n < 2u32 { return seed; }
    if seed % 3u32 == 0u32 { return tree(n - 1u32, wrapping_add(wrapping_mul(seed, 7u32), 1u32)); }
    let a = tree(n - 1u32, wrapping_add(seed, 1u32));
    let b = tree(n - 2u32, wrapping_mul(seed, 5u32));
    return wrapping_add(a, b);
}
fn run(n: u32, seed: u32) -> u32 { return tree(n, seed); }";
    let wide = "fn leaf(x: u32, k: u32) -> u32 { var s = x; for i in 0u32..k { s = wrapping_add(wrapping_mul(s, 33u32), i); } return s; }
fn run(x: u32) -> u32 {
    let a = leaf(x, 40u32); let b = leaf(x, 41u32); let c = leaf(x, 42u32); let d = leaf(x, 43u32); let e = leaf(x, 44u32);
    let f = leaf(x, 45u32); let g = leaf(x, 46u32); let h = leaf(x, 47u32); let i = leaf(x, 48u32); let j = leaf(x, 49u32);
    return wrapping_add(wrapping_add(wrapping_add(wrapping_add(a, b), wrapping_add(c, d)), wrapping_add(wrapping_add(e, f), wrapping_add(g, h))), wrapping_add(i, j));
}";
    let ordered = "fn slow(n: u32) -> i32 { var i = 0u32; while i < n { i = i + 1u32; } return 1; }
fn bad(d: i32) -> i32 { return 100 / d; }
fn run(n: u32, d: i32) -> i32 { return slow(n) + bad(d); }";
    let owners = "fn fill(n: u32, v: i32) -> i32 { let xs = filled_i32(n, v); var s = 0; for i in 0u32..n { s = s + xs[i]; } return s; }
fn pick(n: u32, at: u32) -> i32 { let xs = filled_i32(n, 7); return xs[at]; }
fn run(n: u32, at: u32) -> i32 { let a = fill(n, 2); let b = pick(n, at); return a + b; }";
    let churn = "fn churn(n: u32) -> i32 { var acc = 0; for i in 0u32..n { let xs = filled_i32(2u32, 3); acc = acc + xs[0u32]; } return acc; }
fn run(n: u32) -> i32 { return churn(n); }";
    vec![
        Case {
            name: "data_dependent_trip_counts",
            source: trips,
            entry: "run",
            slots: 2,
            storage: default_storage(),
            runs: [(0, 0), (3, 500), (500, 3), (40, 40), (1000, 999)]
                .iter()
                .map(|&(a, b)| run(vec![U32(a), U32(b)], FUEL, 2, 16, 64))
                .collect(),
        },
        Case {
            name: "divergent_recursive_frontier",
            source: tree,
            entry: "run",
            slots: 8,
            storage: default_storage(),
            runs: vec![
                run(vec![U32(0), U32(9)], FUEL, 8, 4, 64),
                run(vec![U32(8), U32(1)], FUEL, 8, 4, 64),
                run(vec![U32(10), U32(2)], FUEL, 8, 8, 64),
                run(vec![U32(12), U32(4)], FUEL, 8, 64, 64),
                run(vec![U32(7), U32(4)], FUEL, 2, 1, 3),
                // Depth exhaustion at 64 frames is a typed tool limit.
                run(vec![U32(70), U32(1)], FUEL, 8, 64, 64),
            ],
        },
        Case {
            name: "worklist_overflow_at_bound",
            source: wide,
            entry: "run",
            slots: 8,
            storage: default_storage(),
            runs: [0, 1, 3, 8]
                .iter()
                .map(|&c| run(vec![U32(77)], FUEL, c, 8, 64))
                .collect(),
        },
        Case {
            name: "early_trap_in_one_lane",
            source: ordered,
            entry: "run",
            slots: 2,
            storage: default_storage(),
            runs: vec![
                run(vec![U32(2000), I32(0)], FUEL, 2, 8, 64),
                run(vec![U32(2000), I32(5)], FUEL, 2, 8, 64),
                run(vec![U32(0), I32(0)], FUEL, 2, 1, 1),
                // The older call exhausts fuel first: its limit, never the
                // younger lane's held trap, is the observation.
                run(vec![U32(60000), I32(0)], 1000, 2, 8, 64),
                run(vec![U32(4000000), I32(0)], FUEL, 2, 4096, 64),
            ],
        },
        Case {
            name: "device_owners_traps_and_heap_limit",
            source: owners,
            entry: "run",
            slots: 2,
            storage: default_storage(),
            runs: vec![
                run(vec![U32(4), U32(1)], FUEL, 2, 4, 64),
                run(vec![U32(4), U32(4)], FUEL, 2, 4, 64),
                run(vec![U32(0), U32(0)], FUEL, 2, 1, 2),
                run(vec![U32(300), U32(299)], FUEL, 2, 32, 64),
                // Over the 65,536 live-element device heap: typed HeapLimit,
                // equal to the oracle arena configured with the same cap.
                run(vec![U32(70000), U32(0)], FUEL, 2, 64, 64),
            ],
        },
        Case {
            name: "allocation_identity_exhaustion",
            source: churn,
            entry: "run",
            slots: 2,
            storage: invar::general_net::compiled::OwnedLimits {
                elements: 64,
                allocations: 16,
                cleanup_events: 64,
            },
            runs: vec![
                run(vec![U32(10)], FUEL, 2, 16, 64),
                run(vec![U32(15)], FUEL, 2, 16, 64),
                run(vec![U32(40)], FUEL, 2, 16, 64),
            ],
        },
    ]
}

/// Fuel cut points spread over the whole run, landing inside rounds and
/// epochs: 1, every tenth of the use, one short, exact and one over.
pub fn fuel_cuts(steps: u64) -> Vec<u64> {
    let mut cuts: Vec<u64> = (1..=10).map(|k| (steps * k / 10).max(1)).collect();
    cuts.extend([1, steps.saturating_sub(1).max(1), steps + 1]);
    cuts.sort_unstable();
    cuts.dedup();
    cuts
}

pub fn run_case<B: Backend>(
    target: Target,
    label: &str,
    case: &Case,
    backend: &mut impl FnMut() -> B,
) -> CaseSummary {
    let limits = Limits {
        owned: case.storage,
        private: invar::general_net::compiled::SpeculationLimits {
            slots: case.slots,
            ..Default::default()
        },
        ..Limits::default()
    };
    let plan = Plan::from_source(case.source.as_bytes(), case.entry, target, limits)
        .unwrap_or_else(|e| panic!("{label} {}: admission {e:?}", case.name));
    let mut summary = CaseSummary::default();
    for r in &case.runs {
        let cfg = config(r.fuel, 1_000_000, r.quantum, r.rounds, r.capacity);
        let expected = reference(&plan, &r.inputs, cfg);
        let cfg = config(
            r.fuel,
            expected.events.len().max(1),
            r.quantum,
            r.rounds,
            r.capacity,
        );
        let context = format!(
            "{label} {} {:?} fuel={} cap={} q={} rounds={}",
            case.name, r.inputs, r.fuel, r.capacity, r.quantum, r.rounds
        );
        let (got, epochs) = drive(&plan, &mut backend(), &r.inputs, cfg)
            .unwrap_or_else(|e| panic!("{context}: {e:?}"));
        summary.runs += 1;
        *summary
            .statuses
            .entry(format!("{:?}", got.status))
            .or_default() += 1;
        if device_storage_limit(got.status) {
            assert_eq!(got.result, None, "{context}: no publication");
            assert!(
                expected.semantic.result.is_ok(),
                "{context}: oracle {:?}",
                expected.semantic.result
            );
            summary.storage_limited += 1;
        } else {
            agree(&context, &plan, &got, &expected);
        }
        let f = Frontier::of(&got);
        summary.spawned += f.spawned;
        summary.committed += f.committed;
        summary.replayed += f.replayed;
        summary.capacity_serialized += f.capacity_serialized;
        summary.owned_replays += f.owned_replays;
        summary.peak_width = summary.peak_width.max(f.peak_width);
        summary.max_epochs = summary.max_epochs.max(epochs);
    }
    summary
}

/// Fuel exhausted at cut points inside rounds and across epochs of a
/// recursive frontier: each prefix is exactly the oracle's, with no value.
pub fn fuel_prefixes<B: Backend>(
    target: Target,
    label: &str,
    backend: &mut impl FnMut() -> B,
) -> usize {
    let case = &adversarial()[1];
    let limits = Limits {
        private: invar::general_net::compiled::SpeculationLimits {
            slots: 8,
            ..Default::default()
        },
        ..Limits::default()
    };
    let plan = Plan::from_source(case.source.as_bytes(), "run", target, limits).unwrap();
    let inputs = [Value::U32(7), Value::U32(1)];
    let full = reference(&plan, &inputs, config(FUEL, 1_000_000, 3, 2, 8));
    let mut n = 0;
    for fuel in fuel_cuts(full.semantic.steps_used) {
        let cfg = config(fuel, full.events.len().max(1), 3, 2, 8);
        let expected = reference(&plan, &inputs, cfg);
        let (got, _) = drive(&plan, &mut backend(), &inputs, cfg).unwrap();
        agree(
            &format!("{label} fuel prefix {fuel}"),
            &plan,
            &got,
            &expected,
        );
        n += 1;
    }
    n
}

/// Run `job` over `items` on at most `workers` threads (for example bounded
/// compiler children); the results keep the input order.
pub fn bounded_pool<T: Sync, R: Send>(
    items: &[T],
    workers: usize,
    job: impl Fn(&T) -> R + Sync,
) -> Vec<R> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut done: Vec<(usize, R)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers.clamp(1, items.len().max(1)))
            .map(|_| {
                scope.spawn(|| {
                    let mut mine = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(item) = items.get(i) else {
                            break mine;
                        };
                        mine.push((i, job(item)));
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().unwrap())
            .collect()
    });
    done.sort_unstable_by_key(|(i, _)| *i);
    done.into_iter().map(|(_, result)| result).collect()
}

/// Run `shards` corpus shards concurrently (each with a 64 MiB stack for the
/// recursive reference evaluator) and merge their denominators. `sampled`
/// runs the deterministic `sample` instead of the whole corpus.
pub fn sharded<B: Backend>(
    target: Target,
    label: &str,
    backend: fn() -> B,
    fuel_short: bool,
    shards: usize,
    sampled: bool,
) -> Denominator {
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..shards)
            .map(|i| {
                std::thread::Builder::new()
                    .stack_size(64 * 1024 * 1024)
                    .spawn_scoped(scope, move || {
                        let shard = (i, shards);
                        corpus_denominator(target, label, backend, fuel_short, shard, sampled)
                    })
                    .unwrap()
            })
            .collect();
        let mut total = Denominator::default();
        for handle in handles {
            total.absorb(handle.join().unwrap());
        }
        total
    })
}

/// Case-specific frontier facts, identical for every backend.
pub fn check_case(name: &str, s: &CaseSummary) {
    match name {
        "data_dependent_trip_counts" => assert!(s.committed > 0, "{s:?}"),
        "divergent_recursive_frontier" => {
            assert!(s.committed > 0 && s.peak_width >= 2, "{s:?}");
            assert_eq!(s.statuses["CallDepthLimit"], 1, "{s:?}");
        }
        "worklist_overflow_at_bound" => {
            assert!(s.capacity_serialized > 0 && s.committed > 0, "{s:?}");
        }
        "early_trap_in_one_lane" => {
            assert_eq!(s.statuses["DivisionByZero"], 2, "{s:?}");
            assert_eq!(s.statuses["FuelLimit"], 2, "{s:?}");
        }
        "device_owners_traps_and_heap_limit" => {
            assert!(s.statuses["IndexOutOfBounds"] >= 2, "{s:?}");
            assert_eq!(s.statuses["HeapLimit"], 1, "{s:?}");
            assert!(s.owned_replays > 0, "{s:?}");
        }
        "allocation_identity_exhaustion" => {
            assert_eq!(s.storage_limited, 1, "{s:?}");
            assert_eq!(s.statuses["AllocationLimit"], 1, "{s:?}");
        }
        _ => panic!("unknown case {name}"),
    }
}

/// Host fence, cancellation and quiescence on one backend: a cancelled epoch
/// publishes nothing and is reclaimed only after its completion fence; the
/// same backend then runs an exact fresh session; reclaiming unfenced work
/// quarantines it and poisons the backend, refusing further preparation.
pub fn lifecycle<B: Backend>(target: Target, label: &str, mut open: impl FnMut() -> B) {
    use invar::general_net::compiled::ExecutionState;
    let case = &adversarial()[3];
    let plan = Plan::from_source(
        case.source.as_bytes(),
        case.entry,
        target,
        Limits::default(),
    )
    .unwrap();
    let long = [Value::U32(4_000_000), Value::I32(1)];
    let cfg = config(1_000_000, 64, 4096, 64, 2);
    let mut backend = open();
    let mut session = plan.prepare(&mut backend, &long, cfg).unwrap();
    session.submit().unwrap();
    session.cancel();
    let begun = Instant::now();
    loop {
        match session.poll() {
            Ok(Progress::Pending) if begun.elapsed() < DEADLINE => {
                std::thread::sleep(Duration::from_micros(200))
            }
            Ok(progress) => panic!("{label}: cancelled epoch published {progress:?}"),
            Err(_) => break,
        }
    }
    assert_eq!(session.state(), ExecutionState::Cancelled, "{label}");
    assert_eq!(session.submitted(), session.completed(), "{label}");
    assert!(session.submit().is_err(), "{label}");
    drop(session.reclaim().unwrap());
    assert!(!backend.poisoned(), "{label}");
    let short = [Value::U32(10), Value::I32(5)];
    let cfg = config(FUEL, 1_000, 8, 64, 2);
    let (got, _) = drive(&plan, &mut backend, &short, cfg).unwrap();
    agree(
        &format!("{label} after cancel"),
        &plan,
        &got,
        &reference(&plan, &short, cfg),
    );
    let mut session = plan
        .prepare(&mut backend, &long, config(1_000_000, 64, 4096, 64, 2))
        .unwrap();
    session.submit().unwrap();
    assert!(session.reclaim().is_err(), "{label}: unfenced reclaim");
    assert!(backend.poisoned(), "{label}");
    assert!(matches!(
        plan.prepare(&mut backend, &short, cfg),
        Err(device_host::Error::Poisoned)
    ));
}
