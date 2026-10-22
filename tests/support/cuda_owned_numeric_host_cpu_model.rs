//! ABI12 stage 2c S2 focused tests of the subprocess CPU group model, as a
//! child module of `cuda_owned_numeric_host::cpu`. Positive epochs are
//! compared with the ABI11 CPU host at counter level only (its 8 public
//! per-epoch counters; ABI11 raw slices are private, so full device-state
//! equality per epoch is NOT established here) and at the terminal
//! observation, and with the independent net oracle (exact bits). CPU
//! modelling only: never device execution or device qualification.
use crate::child_deadline;

use super::super::{Config, Error, Plan, Progress, State, root};
use super::*;
use crate::{
    eval,
    general_net::{numeric_owned_resume_host as abi11, oracle, provenance},
    numeric_contract::F32Bits,
};
use std::time::{Duration, Instant};

const SOURCE: &str = "fn run(a: f32, n: u32) -> f32 { var xs: Array<f32> = filled_f32(n + 1u32, a); xs[0u32] = xs[0u32] * a + a; return xs[0u32]; } fn main() -> i32 { return 0; }";

fn compiler() -> PathBuf {
    std::env::var_os("CXX").map_or_else(|| PathBuf::from("c++"), PathBuf::from)
}
fn storage() -> compiled::OwnedLimits {
    compiled::OwnedLimits {
        elements: 64,
        allocations: 16,
        cleanup_events: 32,
    }
}
fn plan() -> Plan {
    Plan::from_source(
        SOURCE.as_bytes(),
        "run",
        provenance::Limits::default(),
        storage(),
    )
    .unwrap()
}
fn args(r: usize) -> Vec<eval::Value> {
    vec![
        eval::Value::F32(F32Bits::from_bits(0x3f80_0000 + r as u32)),
        eval::Value::U32((r % 4) as u32),
    ]
}
fn roots(n: usize) -> Vec<Vec<eval::Value>> {
    (0..n).map(args).collect()
}
/// Budgets under which every `SOURCE` root completes successfully (status 0).
fn config(roots: usize, quantum: u64) -> Config {
    Config {
        roots,
        fuel: 1000,
        transitions: 200,
        trace_events: 64,
        quantum,
        max_submissions: 200u64.div_ceil(quantum) + 1,
    }
}
fn limits(c: Config) -> oracle::Limits {
    oracle::Limits {
        fuel: c.fuel,
        transitions: c.transitions,
        trace_events: c.trace_events,
        array_elements: storage().elements,
    }
}
fn model() -> CpuGroupModel {
    CpuGroupModel::new(&compiler(), Duration::from_secs(60)).unwrap()
}
/// Poll one submitted epoch to a non-pending outcome within a test bound.
fn settle(s: &mut super::super::Session<'_, CpuGroupModel>) -> Result<Progress, Error> {
    let begun = Instant::now();
    loop {
        match s.poll() {
            Ok(Progress::Pending) => {
                assert!(
                    begun.elapsed() < child_deadline::wall(Duration::from_secs(150)),
                    "test bound"
                );
                std::thread::sleep(Duration::from_millis(2));
            }
            other => return other,
        }
    }
}

/// The ABI11 counters published per epoch: (steps, transitions, events,
/// depth, max_frames, allocations, live_elements, cleanup_events).
type Counters = [u64; 8];
fn ours(c: [u64; 18]) -> Counters {
    [c[4], c[5], c[6], c[2], c[3], c[8], c[9], c[10]]
}
fn theirs(p: &abi11::Pending) -> Counters {
    [
        p.steps,
        p.transitions,
        p.events as u64,
        p.depth as u64,
        p.max_frames as u64,
        p.allocations as u64,
        p.live_elements as u64,
        p.cleanup_events as u64,
    ]
}
fn same(a: &oracle::Observation, b: &oracle::Observation) -> bool {
    format!("{:?}", a.semantic.result) == format!("{:?}", b.semantic.result)
        && a.semantic.steps_used == b.semantic.steps_used
        && format!("{:?}", a.semantic.cleanup_trace) == format!("{:?}", b.semantic.cleanup_trace)
        && a.transitions == b.transitions
        && format!("{:?}", a.events) == format!("{:?}", b.events)
}

/// Status triple as the ABI11 owned-state tests derive it from an observation.
fn status(o: &oracle::Observation) -> (i64, i64, i64) {
    match &o.semantic.result {
        Ok(_) => (0, 0, 0),
        Err(eval::EvalError::Trap { category, span }) => (
            match category {
                eval::TrapCategory::IntegerOverflow => 1,
                eval::TrapCategory::DivisionByZero => 2,
                eval::TrapCategory::IndexOutOfBounds => 9,
                _ => panic!("trap"),
            },
            span.start as i64,
            span.end as i64,
        ),
        Err(eval::EvalError::EvaluationLimit { .. }) => (3, 0, 0),
        Err(eval::EvalError::Tool {
            code: "ARRAY_SIZE_LIMIT",
            ..
        }) => (10, 0, 0),
        Err(eval::EvalError::Tool {
            code: "CALL_DEPTH_LIMIT",
            ..
        }) => (4, 0, 0),
        Err(_) => (
            match o.failure {
                Some(oracle::ToolFailure::TransitionLimit) => 5,
                Some(oracle::ToolFailure::TraceLimit) => 6,
                Some(oracle::ToolFailure::Allocation) => 11,
                _ => panic!("{o:?}"),
            },
            0,
            0,
        ),
    }
}
fn event_rows(o: &oracle::Observation) -> Vec<[u64; 7]> {
    o.events
        .iter()
        .map(|e| {
            [
                u64::from(e.kind == oracle::EventKind::Complete),
                e.frame,
                e.activation,
                e.token.block.function as u64,
                e.token.block.block as u64,
                e.token.position as u64,
                e.rule as u64,
            ]
        })
        .collect()
}
/// The independent net oracle comparison used by the ABI11 owned-state
/// tests: status, counters, event rows, cleanup lengths and exact result bits.
fn against_oracle(d: &oracle::Observation, o: &oracle::Observation, r: usize) {
    assert_eq!(status(d), status(o), "root {r}");
    assert_eq!(
        (d.semantic.steps_used, d.transitions, d.max_frames),
        (o.semantic.steps_used, o.transitions, o.max_frames),
        "root {r}"
    );
    assert_eq!(event_rows(d), event_rows(o), "root {r}");
    let lengths = |x: &oracle::Observation| {
        x.semantic
            .cleanup_trace
            .iter()
            .map(|c| c.length)
            .collect::<Vec<_>>()
    };
    assert_eq!(lengths(d), lengths(o), "root {r}");
    match (&d.semantic.result, &o.semantic.result) {
        (Ok(eval::Value::F32(a)), Ok(eval::Value::F32(b))) => {
            assert_eq!(a.to_bits(), b.to_bits(), "root {r} exact bits")
        }
        (Ok(a), Ok(b)) => assert_eq!(format!("{a:?}"), format!("{b:?}"), "root {r}"),
        (Err(_), Err(_)) => {}
        (a, b) => panic!("root {r}: {a:?} vs {b:?}"),
    }
}

/// One source-issued run: source, storage, config and per-root arguments.
struct Case<'a> {
    source: &'a str,
    storage: compiled::OwnedLimits,
    config: Config,
    args: Vec<Vec<eval::Value>>,
}
struct Outcome {
    epochs: usize,
    compared: usize,
    /// Exact status word of each root's published terminal packet.
    statuses: Vec<i64>,
    /// Roots whose storage-capacity status (12/13) the net oracle cannot model.
    oracle_skipped: usize,
}

/// Run a full session; every epoch's 8 per-root ABI11 counters equal the
/// ABI11 CPU host's (counter-level, not full state), terminal roots stay
/// terminal, and every published result equals the ABI11 terminal
/// observation and the net oracle exactly.
fn run_case(case: &Case<'_>) -> Outcome {
    let n = case.config.roots;
    let c = case.config;
    let plan = Plan::from_source(
        case.source.as_bytes(),
        "run",
        provenance::Limits::default(),
        case.storage,
    )
    .unwrap();
    let mut s = plan.prepare(model(), &case.args, c).ok().unwrap();
    let frame_bytes = s.lane.frame_bytes;
    let mut epochs: Vec<Vec<[u64; 18]>> = Vec::new();
    loop {
        s.submit().unwrap();
        let progress = settle(&mut s).unwrap();
        let prev = &s.previous;
        epochs.push(
            (0..n)
                .map(|r| root(prev, s.lane, r).counters(frame_bytes))
                .collect(),
        );
        if progress == Progress::Published {
            break;
        }
        assert_eq!(progress, Progress::Advanced);
    }
    assert_eq!(s.state(), State::Published);
    assert!(epochs.len() as u64 <= c.max_submissions);
    let statuses = (0..n)
        .map(|r| root(&s.previous, s.lane, r).output[0])
        .collect();
    let l = oracle::Limits {
        fuel: c.fuel,
        transitions: c.transitions,
        trace_events: c.trace_events,
        array_elements: case.storage.elements,
    };
    // ABI11 reference, one root at a time, same budgets and quantum.
    let mut reference = abi11::Plan::from_source(
        case.source.as_bytes(),
        "run",
        Default::default(),
        case.storage,
        &compiler(),
    )
    .unwrap();
    let mut compared = 0;
    let mut oracle_skipped = 0;
    for r in 0..n {
        let mut run = reference.start(&case.args[r], l).unwrap();
        let mut done = None;
        for (k, epoch) in epochs.iter().enumerate() {
            let counters = epoch[r];
            if let Some(at) = done {
                assert_eq!(counters[1], 1, "root {r} terminal since epoch {at}");
                continue;
            }
            match run.resume(c.quantum).unwrap() {
                abi11::Progress::Pending(p) => {
                    assert_eq!(counters[1], 0, "root {r} epoch {k}");
                    assert_eq!(ours(counters), theirs(&p), "root {r} epoch {k}");
                }
                abi11::Progress::Complete(o11) => {
                    assert_eq!(counters[1], 1, "root {r} epoch {k}");
                    let o12 = s.result(r).unwrap();
                    assert!(same(&o12, &o11), "root {r}: ABI12 {o12:?} vs ABI11 {o11:?}");
                    // The net oracle has no compiled descriptor (12) or
                    // cleanup (13) capacity; those statuses are checked
                    // against the compiled ABI11 reference only (as the
                    // ABI11 owned-state tests do).
                    if matches!(counters[14], 12 | 13) {
                        oracle_skipped += 1;
                    } else {
                        let oracle = plan.artifact().observe_oracle(&case.args[r], l).unwrap();
                        against_oracle(&o12, &oracle.observation(), r);
                    }
                    done = Some(k);
                }
            }
            compared += 1;
        }
        assert!(done.is_some(), "root {r} never completed in ABI11");
    }
    let used = s.ledger().used();
    assert!(used > 0 && used <= s.ledger().reserved());
    assert!(s.host_bytes() <= 256 * 1024 * 1024);
    let (_, work) = s.reclaim().ok().unwrap();
    assert_eq!(work.record().lanes, n.div_ceil(32) * 32);
    Outcome {
        epochs: epochs.len(),
        compared,
        statuses,
        oracle_skipped,
    }
}

fn positive(n: usize, quantum: u64) -> (usize, usize) {
    let o = run_case(&Case {
        source: SOURCE,
        storage: storage(),
        config: config(n, quantum),
        args: roots(n),
    });
    assert!(o.statuses.iter().all(|s| *s == 0));
    (o.epochs, o.compared)
}

#[test]
fn s2_positive_epochs_roots_1_32_33_1024_quanta_1_3_t() {
    for (n, q) in [(1, 3), (32, 3), (33, 1), (33, 3), (33, 200), (1024, 3)] {
        let (epochs, compared) = positive(n, q);
        eprintln!(
            "roots {n} quantum {q}: {epochs} epochs, {compared} root-epochs counter-equal to ABI11"
        );
        assert!(compared >= n);
    }
}

#[test]
fn s2_failed_child_paths_never_read_back() {
    let plan = plan();
    // Reaped non-zero exit: FailedQuiescent, no epoch output, reclaim allowed.
    let mut m = model();
    m.substitute = Some((PathBuf::from("/bin/false"), vec![]));
    let mut s = plan.prepare(m, &roots(1), config(1, 3)).ok().unwrap();
    s.submit().unwrap();
    assert_eq!(settle(&mut s), Err(Error::FailedQuiescent));
    assert_eq!(s.state(), State::FailedQuiescent);
    assert_eq!(s.submit(), Err(Error::State(State::FailedQuiescent)));
    assert!(s.reclaim().is_ok());

    // Deadline: the child's group is killed and reaped, FailedQuiescent.
    let mut m = CpuGroupModel::new(&compiler(), Duration::from_millis(200)).unwrap();
    m.substitute = Some((PathBuf::from("/bin/sleep"), vec!["30".into()]));
    let mut s = plan.prepare(m, &roots(1), config(1, 3)).ok().unwrap();
    s.submit().unwrap();
    let begun = Instant::now();
    assert_eq!(settle(&mut s), Err(Error::FailedQuiescent));
    assert!(begun.elapsed() < child_deadline::wall(Duration::from_secs(10)));
    assert!(s.reclaim().is_ok());

    // A program that cannot be spawned is NotSubmitted.
    let mut m = model();
    m.substitute = Some((PathBuf::from("/nonexistent/invar-model"), vec![]));
    let mut s = plan.prepare(m, &roots(1), config(1, 3)).ok().unwrap();
    assert_eq!(s.submit(), Err(Error::NotSubmitted));
    assert!(s.reclaim().is_ok());

    // An unconfirmable child is the only Unknown: one poison, quarantine,
    // no further poll; dropping forgets the work and backend.
    let mut m = model();
    m.unconfirmable = true;
    let mut s = plan.prepare(m, &roots(1), config(1, 3)).ok().unwrap();
    s.submit().unwrap();
    assert_eq!(s.poll(), Err(Error::Unknown));
    assert_eq!(s.state(), State::Quarantined);
    assert!(s.backend.as_ref().unwrap().poisoned());
    assert_eq!(s.poll(), Err(Error::State(State::Quarantined)));
    assert!(s.reclaim().is_err());
}

#[test]
fn s2_refusals_never_invoke_the_compiler() {
    use std::os::unix::fs::PermissionsExt;
    let dir = std::env::temp_dir().join(format!("s2-recording-compiler-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let calls = dir.join("calls");
    let script = dir.join("cxx");
    fs::write(
        &script,
        format!("#!/bin/sh\necho x >> '{}'\nexit 1\n", calls.display()),
    )
    .unwrap();
    fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
    let plan = plan();
    // A workset whose host total exceeds 256 MiB (device 37 MiB < 64 MiB,
    // but 2 snapshots + child cap > 256 MiB) is refused before the backend.
    let big = Config {
        roots: 1024,
        trace_events: 400,
        ..config(1024, 3)
    };
    let m = CpuGroupModel::new(&script, Duration::from_secs(5)).unwrap();
    let refused = plan.prepare(m, &roots(1024), big).err().unwrap();
    assert_eq!(refused.error, Error::HostBytes);
    assert!(!calls.exists(), "compiler invoked on a refused workset");
    // A poisoned model refuses before any compiler call.
    let mut m = CpuGroupModel::new(&script, Duration::from_secs(5)).unwrap();
    m.poisoned = true;
    let refused = plan.prepare(m, &roots(1), config(1, 3)).err().unwrap();
    assert!(matches!(refused.error, Error::Backend(_)));
    assert!(!calls.exists());
    // Epoch deadlines are bounded.
    assert!(CpuGroupModel::new(&script, Duration::ZERO).is_err());
    assert!(CpuGroupModel::new(&script, MAX_EPOCH_DEADLINE + Duration::from_millis(1)).is_err());
    // A real refusal from the compiler itself is a Backend error after one
    // version probe and one compile attempt (positive control for the
    // recorder: probes and the compile reach it).
    let m = CpuGroupModel::new(&script, Duration::from_secs(5)).unwrap();
    let refused = plan.prepare(m, &roots(1), config(1, 3)).err().unwrap();
    assert!(matches!(refused.error, Error::Backend(_)));
    assert!(calls.exists());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn s2_host_cost_includes_child_cap_and_is_bounded() {
    let plan = plan();
    let c = config(1024, 3);
    let sizes = plan.sizes(c).unwrap();
    let d = sizes.device_bytes;
    let host = plan.host_bytes::<CpuGroupModel>(c, sizes).unwrap();
    assert_eq!(
        child_cap(d),
        Some(d + 33 * STACK_BYTES + CHILD_OVERHEAD_BYTES)
    );
    assert!(host >= 2 * d + child_cap(d).unwrap());
    assert!(host <= 256 * 1024 * 1024, "{host}");
}

const RECURSE: &str = "fn run(n: u32) -> u32 { if n == 0u32 { return 0u32; } return run(n - 1u32) + 1u32; } fn main() -> i32 { return 0; }";
const ARRAY: &str = "fn run(a: f32, n: u32) -> f32 { var xs: Array<f32> = filled_f32(n, a); return xs[0u32]; } fn main() -> i32 { return 0; }";
const INDEX: &str = "fn run(a: f32, n: u32) -> f32 { var xs: Array<f32> = filled_f32(2u32, a); return xs[n]; } fn main() -> i32 { return 0; }";
const LOOP: &str = "fn run(a: f32, n: u32) -> f32 { var i: u32 = 0u32; var s: f32 = a; while i < n { var xs: Array<f32> = filled_f32(1u32, a); s = s + xs[0u32]; i = i + 1u32; } return s; } fn main() -> i32 { return 0; }";

fn f(bits: u32, n: u32) -> Vec<eval::Value> {
    vec![
        eval::Value::F32(F32Bits::from_bits(bits)),
        eval::Value::U32(n),
    ]
}
fn budget(fuel: u64, transitions: u64, trace_events: usize, quantum: u64) -> Config {
    Config {
        roots: 1,
        fuel,
        transitions,
        trace_events,
        quantum,
        max_submissions: transitions.div_ceil(quantum) + 1,
    }
}
fn store(elements: usize, allocations: usize, cleanup_events: usize) -> compiled::OwnedLimits {
    compiled::OwnedLimits {
        elements,
        allocations,
        cleanup_events,
    }
}

/// Each reachable terminal status from an admitted source, through real
/// subprocess epochs, counter-equal to ABI11 per epoch and terminal-equal to
/// ABI11 and the net oracle.
#[test]
fn s2_status_corpus_reaches_every_device_status() {
    let i32s = |xs: &[i32]| xs.iter().map(|x| eval::Value::I32(*x)).collect::<Vec<_>>();
    let cases: Vec<(i64, Case<'_>)> = vec![
        (
            0,
            Case {
                source: SOURCE,
                storage: storage(),
                config: budget(1000, 200, 64, 10),
                args: vec![f(0x3f80_0000, 1)],
            },
        ),
        (
            1,
            Case {
                source: "fn run(a: i32) -> i32 { return a + 2147483647; } fn main() -> i32 { return 0; }",
                storage: storage(),
                config: budget(1000, 30, 16, 3),
                args: vec![i32s(&[1])],
            },
        ),
        (
            2,
            Case {
                source: "fn run(a: i32, b: i32) -> i32 { return a / b; } fn main() -> i32 { return 0; }",
                storage: storage(),
                config: budget(1000, 30, 16, 3),
                args: vec![i32s(&[1, 0])],
            },
        ),
        (
            3,
            Case {
                source: SOURCE,
                storage: storage(),
                config: budget(1, 30, 16, 3),
                args: vec![f(0x3f80_0000, 1)],
            },
        ),
        (
            4,
            Case {
                source: RECURSE,
                storage: storage(),
                config: budget(100_000, 20_000, 5000, 100),
                args: vec![vec![eval::Value::U32(100)]],
            },
        ),
        (
            5,
            Case {
                source: SOURCE,
                storage: storage(),
                config: budget(1000, 5, 16, 1),
                args: vec![f(0x3f80_0000, 1)],
            },
        ),
        (
            6,
            Case {
                source: SOURCE,
                storage: storage(),
                config: budget(1000, 30, 1, 3),
                args: vec![f(0x3f80_0000, 1)],
            },
        ),
        (
            9,
            Case {
                source: INDEX,
                storage: storage(),
                config: budget(1000, 30, 16, 3),
                args: vec![f(0x3f80_0000, 5)],
            },
        ),
        (
            10,
            Case {
                source: ARRAY,
                storage: storage(),
                config: budget(1000, 30, 16, 3),
                args: vec![f(0x3f80_0000, 4_000_000_000)],
            },
        ),
        (
            11,
            Case {
                source: ARRAY,
                storage: storage(),
                config: budget(1000, 30, 16, 3),
                args: vec![f(0x3f80_0000, 100)],
            },
        ),
        (
            12,
            Case {
                source: LOOP,
                storage: store(64, 4, 1000),
                config: budget(10_000, 2000, 400, 10),
                args: vec![f(0x3f80_0000, 20)],
            },
        ),
        (
            13,
            Case {
                source: LOOP,
                storage: store(64, 1000, 2),
                config: budget(10_000, 2000, 400, 10),
                args: vec![f(0x3f80_0000, 20)],
            },
        ),
    ];
    let mut reached = Vec::new();
    for (want, case) in &cases {
        let o = run_case(case);
        eprintln!(
            "status {want}: got {:?}, {} epochs, {} root-epochs counter-equal to ABI11",
            o.statuses, o.epochs, o.compared
        );
        assert_eq!(o.statuses, vec![*want], "status {want}");
        assert_eq!(o.oracle_skipped, usize::from(matches!(want, 12 | 13)));
        reached.push(*want);
    }
    assert_eq!(reached, [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13]);
}

/// Status 8 (budget change / invalid quantum) is host-only. A real published
/// run's genuine terminal transition re-labelled as status 8 is refused by
/// the shared device checker; the unforged transition is accepted (control).
/// Status 7 has a single emission site, the `gnet_valid` owner-handle guard,
/// which verified ownership rules out; no admitted source reaches it.
#[test]
fn s2_forged_status_8_is_refused_by_the_shared_checker() {
    use super::super::task;
    let plan = plan();
    let c = config(1, 10);
    let mut s = plan.prepare(model(), &roots(1), c).ok().unwrap();
    loop {
        s.submit().unwrap();
        if settle(&mut s).unwrap() == Progress::Published {
            break;
        }
    }
    let lane = s.lane;
    let shape = plan.shape();
    let l = plan.oracle_limits(c);
    // After publication, `previous` is the final epoch and `candidate` the
    // one before it, where the root was still pending.
    let old = root(&s.candidate, lane, 0);
    assert_eq!(
        old.counters(lane.frame_bytes)[1],
        0,
        "needs a pending predecessor"
    );
    let new = root(&s.previous, lane, 0);
    assert_eq!(new.counters(lane.frame_bytes)[14], 0);
    assert_eq!(
        task::validate_device_epoch(
            &shape,
            l,
            &old.words(lane.frame_bytes),
            &new.words(lane.frame_bytes),
            c.quantum
        ),
        Ok(())
    );
    let mut forged = s.previous.clone_fallible().unwrap();
    let at = lane.frame_bytes + 14 * 8;
    forged.tasks[at..at + 8].copy_from_slice(&8u64.to_ne_bytes());
    forged.output[0] = 8;
    let bad = root(&forged, lane, 0);
    assert_eq!(
        task::validate_device_epoch(
            &shape,
            l,
            &old.words(lane.frame_bytes),
            &bad.words(lane.frame_bytes),
            c.quantum
        ),
        Err(task::TaskError::Malformed)
    );
}

fn sha_hex(bytes: &[u8]) -> String {
    sha256(bytes).iter().map(|b| format!("{b:02x}")).collect()
}
fn le<T: Copy>(words: &[T], to: impl Fn(T) -> u64) -> Vec<u8> {
    words.iter().flat_map(|w| to(*w).to_le_bytes()).collect()
}

#[test]
fn s2_execution_receipt_binds_identity_and_refuses_forgery() {
    use crate::general_net::provenance::owned_numeric_execution as receipt;
    let plan = plan();
    let c = config(33, 10);
    let mut s = plan.prepare(model(), &roots(33), c).ok().unwrap();
    // Only a published session yields a receipt.
    assert_eq!(
        s.execution_receipt().err(),
        Some(Error::State(State::Ready))
    );
    let mut epochs = 0u64;
    loop {
        s.submit().unwrap();
        epochs += 1;
        if settle(&mut s).unwrap() == Progress::Published {
            break;
        }
    }
    let r = s.execution_receipt().unwrap();
    let m = r.manifest().clone();
    assert_eq!(r.id(), sha_hex(m.to_string().as_bytes()));
    // Fixed labels: CPU modelling, no module, no library, no authority.
    assert_eq!(m["schema"], "invar.gnet.abi12.execution-receipt");
    assert_eq!(m["evidence_backend"], "CpuGroupModel");
    assert_eq!(m["module_receipt_id"], serde_json::Value::Null);
    assert_eq!(m["resume_authorized"], false);
    assert_eq!(m["device_qualification"], false);
    assert_eq!(m["execution"]["loader"], "subprocess");
    assert_eq!(m["execution"]["library"], serde_json::Value::Null);
    assert!(!m.to_string().contains("library_sha256"));
    // Compile receipt and emitted-text binding.
    let compile = plan.artifact();
    assert_eq!(m["compile_receipt"]["id"], compile.id());
    let artifact_sha = &compile.manifest()["artifact"]["sha256"];
    assert_eq!(&m["compile_receipt"]["artifact_sha256"], artifact_sha);
    assert_eq!(&m["model"]["kernel_sha256"], artifact_sha);
    // Model identity from independently recomputed bytes.
    assert_eq!(m["model"]["flags"], serde_json::json!(FLAGS_V2));
    assert_eq!(m["model"]["shim_sha256"], sha_hex(SHIM_V1.as_bytes()));
    assert_eq!(m["model"]["driver_sha256"], sha_hex(DRIVER_V2.as_bytes()));
    let resolved =
        toolchain::resolve(&compiler(), &std::env::var_os("PATH").unwrap_or_default()).unwrap();
    assert_eq!(
        m["model"]["compiler_sha256"],
        sha_hex(&fs::read(&resolved).unwrap())
    );
    let triple = Command::new(&resolved)
        .arg("-dumpmachine")
        .output()
        .unwrap();
    assert_eq!(
        m["model"]["host_triple"],
        String::from_utf8_lossy(&triple.stdout).trim()
    );
    // Run facts: config, input, epochs (last state recomputed from the
    // published snapshot), terminal packets and ledger.
    assert_eq!(
        m["config"],
        serde_json::json!({"roots": 33, "fuel": 1000, "transitions": 200,
            "trace_events": 64, "quantum": 10, "max_submissions": 21})
    );
    assert_eq!(
        m["input_sha256"],
        sha_hex(&le(&s.previous.input, |w| w as u64))
    );
    assert_eq!(m["epochs"].as_array().unwrap().len() as u64, epochs);
    let words: [usize; 7] = serde_json::from_value(m["model"]["words"].clone()).unwrap();
    let mut state = Vec::new();
    state.extend(MAGIC.to_le_bytes());
    state.extend(epochs.to_le_bytes());
    for w in words {
        state.extend((w as u64).to_le_bytes());
    }
    let p = &s.previous;
    state.extend(le(&p.input, |w| w as u64));
    state.extend(le(&p.output, |w| w as u64));
    state.extend(le(&p.trace, |w| w));
    state.extend(le(&p.config, |w| w));
    state.extend(le(&p.heap, |w| w as u64));
    state.extend(le(&p.cleanup, |w| w));
    state.extend(&p.tasks);
    assert_eq!(
        m["epochs"][epochs as usize - 1]["state_sha256"],
        sha_hex(&state)
    );
    for root_index in 0..33 {
        let out = root(&s.previous, s.lane, root_index).output;
        let t = &m["terminals"][root_index];
        assert_eq!(t["status"], 0);
        assert_eq!(t["packet_sha256"], sha_hex(&le(out, |w| w as u64)));
    }
    assert_eq!(m["ledger"]["instrumented"], false);
    assert!(m["ledger"]["work_used"].as_u64() <= m["ledger"]["work_reserved"].as_u64());
    // Exact revalidation; every forged or tampered field is refused.
    r.verify_manifest(&m).unwrap();
    type Tamper = (&'static str, fn(&mut serde_json::Value));
    let tampers: Vec<Tamper> = vec![
        ("backend", |m| m["evidence_backend"] = "CudaDriver".into()),
        ("module receipt", |m| {
            m["module_receipt_id"] = "module".into()
        }),
        ("resume", |m| m["resume_authorized"] = true.into()),
        ("qualification", |m| m["device_qualification"] = true.into()),
        ("device support", |m| m["device_support"] = true.into()),
        ("library", |m| m["execution"]["library"] = "model.so".into()),
        ("loader", |m| m["execution"]["loader"] = "dlopen".into()),
        ("compiler", |m| m["model"]["compiler_sha256"] = "00".into()),
        ("executable", |m| {
            m["model"]["executable_sha256"] = "00".into()
        }),
        ("flags", |m| {
            m["model"]["flags"] = serde_json::json!(["-O3"])
        }),
        ("triple", |m| {
            m["model"]["host_triple"] = "x86_64-pc-linux-gnu".into()
        }),
        ("shim", |m| m["model"]["shim_sha256"] = "00".into()),
        ("compile receipt", |m| {
            m["compile_receipt"]["id"] = "00".into()
        }),
        ("nonce", |m| {
            m["plan_nonce"] = (m["plan_nonce"].as_u64().unwrap() + 1).into()
        }),
        ("quantum", |m| m["config"]["quantum"] = 3.into()),
        ("input", |m| m["input_sha256"] = "00".into()),
        ("initial", |m| m["initial_state_sha256"] = "00".into()),
        ("epoch digest", |m| {
            m["epochs"][0]["state_sha256"] = "00".into()
        }),
        ("epoch dropped", |m| {
            m["epochs"].as_array_mut().unwrap().pop();
        }),
        ("status", |m| m["terminals"][0]["status"] = 8.into()),
        ("packet", |m| {
            m["terminals"][0]["packet_sha256"] = "00".into()
        }),
        ("ledger", |m| m["ledger"]["instrumented"] = true.into()),
    ];
    for (name, tamper) in &tampers {
        let mut forged = m.clone();
        tamper(&mut forged);
        assert_ne!(forged, m, "{name} did not change the receipt");
        assert_eq!(
            r.verify_manifest(&forged),
            Err(provenance::Error::ManifestMismatch),
            "{name}"
        );
    }
    // The receipt is a deterministic function of the published session.
    assert_eq!(s.execution_receipt().unwrap().id(), r.id());
    // The builder itself refuses a kernel digest that is not the compile
    // receipt's emitted text, and an empty epoch list.
    let record = {
        let (_, work) = s.reclaim().ok().unwrap();
        assert_eq!(
            m["model"]["executable_sha256"],
            sha_hex(&fs::read(&work.exe).unwrap())
        );
        work.record.clone()
    };
    let identity = |kernel: [u8; 32]| receipt::ModelIdentity {
        compiler_path: &record.compiler_path,
        compiler_sha256: record.compiler_sha256,
        compiler_version_sha256: record.compiler_version_sha256,
        host_triple: &record.host_triple,
        flags: record.flags,
        shim_sha256: record.shim_sha256,
        driver_sha256: record.driver_sha256,
        model_source_sha256: record.model_source_sha256,
        executable_sha256: record.executable_sha256,
        kernel_sha256: kernel,
        stack_bytes: record.stack_bytes,
        child_address_space_bytes: record.child_address_space_bytes,
        lanes: record.lanes,
        words: record.words,
        toolchain: record
            .toolchain
            .iter()
            .map(|p| receipt::ProgramIdentity {
                role: p.role,
                path: &p.path,
                sha256: p.sha256,
                libraries: &p.libraries,
            })
            .collect(),
        runtime_static: record.runtime_static,
        runtime_libraries: &record.runtime_libraries,
        compile_stage: record.compile_stage,
        steady_state_disk_bytes: record.steady_state_disk_bytes,
        steady_state_disk_cap: SESSION_DISK_BYTES,
        toolchain_reverified: record.toolchain_reverified,
    };
    let terminals = vec![(0u64, [0u8; 32]); 33];
    let build = |kernel: [u8; 32], epochs: &[[u8; 32]]| {
        receipt::cpu_group_model(
            super::super::ReceiptSeal(()),
            receipt::Run {
                compile,
                plan_nonce: plan.nonce(),
                config: [33, 1000, 200, 64, 10, 21],
                input_sha256: [0; 32],
                initial_sha256: [0; 32],
                epochs,
                terminals: &terminals,
                work_reserved: 1,
                work_used: 0,
                host_bytes: 0,
                model: identity(kernel),
            },
        )
        .err()
    };
    assert_eq!(
        build([0; 32], &[[0; 32]]),
        Some(provenance::Error::ArtifactMismatch)
    );
    assert_eq!(
        build(record.kernel_sha256, &[]),
        Some(provenance::Error::ExecutionMismatch)
    );
    assert_eq!(build(record.kernel_sha256, &[[0; 32]]), None);

    // A run that failed quiescently has no receipt.
    let mut m2 = model();
    m2.substitute = Some((PathBuf::from("/bin/false"), vec![]));
    let mut f = plan.prepare(m2, &roots(1), config(1, 10)).ok().unwrap();
    f.submit().unwrap();
    assert_eq!(settle(&mut f), Err(Error::FailedQuiescent));
    assert_eq!(
        f.execution_receipt().err(),
        Some(Error::State(State::FailedQuiescent))
    );
}

/// Sanitizer profiles (test-only, recorded separately from the product
/// profile). UBSan runs under the product address-space cap; ASan reserves
/// shadow memory far beyond `RLIMIT_AS` and cannot run under it, which is
/// recorded as a limitation, not hidden by raising the cap.
#[test]
fn s2_sanitizer_profile_limits() {
    let plan = plan();
    // UBSan: positive runs, including the overflow and division traps.
    let mut m = model();
    m.profile = Some((UBSAN_FLAGS_V1, false));
    let mut s = plan.prepare(m, &roots(33), config(33, 10)).ok().unwrap();
    loop {
        s.submit().unwrap();
        if settle(&mut s).unwrap() == Progress::Published {
            break;
        }
    }
    let r = s.execution_receipt().unwrap();
    assert_eq!(
        r.manifest()["model"]["flags"],
        serde_json::json!(UBSAN_FLAGS_V1)
    );
    assert_ne!(r.manifest()["model"]["flags"], serde_json::json!(FLAGS_V2));
    for (source, args, status) in [
        (
            "fn run(a: i32) -> i32 { return a + 2147483647; } fn main() -> i32 { return 0; }",
            vec![eval::Value::I32(1)],
            1,
        ),
        (
            "fn run(a: i32, b: i32) -> i32 { return a / b; } fn main() -> i32 { return 0; }",
            vec![eval::Value::I32(1), eval::Value::I32(0)],
            2,
        ),
    ] {
        let p = Plan::from_source(
            source.as_bytes(),
            "run",
            provenance::Limits::default(),
            storage(),
        )
        .unwrap();
        let mut m = model();
        m.profile = Some((UBSAN_FLAGS_V1, false));
        let mut s = p.prepare(m, &[args], config(1, 10)).ok().unwrap();
        loop {
            s.submit().unwrap();
            if settle(&mut s).unwrap() == Progress::Published {
                break;
            }
        }
        assert_eq!(root(&s.previous, s.lane, 0).output[0], status);
    }
    // ASan: the build succeeds, but the first epoch fails under the product
    // address-space cap: FailedQuiescent, with no readback.
    let mut m = model();
    m.profile = Some((ASAN_FLAGS_V1, false));
    let mut s = plan.prepare(m, &roots(1), config(1, 10)).ok().unwrap();
    s.submit().unwrap();
    assert_eq!(settle(&mut s), Err(Error::FailedQuiescent));
    assert_eq!(s.state(), State::FailedQuiescent);
    let (_, work) = s.reclaim().ok().unwrap();
    let stderr = fs::read_to_string(work.dir.0.join("epoch.stderr")).unwrap();
    // The ASan runtime's shadow-memory mapping fails under the cap.
    let cause = stderr
        .lines()
        .find(|l| l.starts_with("ERROR:"))
        .unwrap_or("");
    eprintln!("ASan under RLIMIT_AS: {cause}");
    assert_eq!(cause, "ERROR: Failed to mmap", "{stderr}");
}

fn recording_compiler(tag: &str) -> (PathBuf, PathBuf, PathBuf) {
    use std::os::unix::fs::PermissionsExt;
    let dir = std::env::temp_dir().join(format!("s2f-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let calls = dir.join("calls");
    let script = dir.join("cxx");
    fs::write(
        &script,
        format!("#!/bin/sh\necho x >> '{}'\nexit 1\n", calls.display()),
    )
    .unwrap();
    fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
    (dir, script, calls)
}

/// F1: the model identity binds the whole toolchain (each program and every
/// library its interpreter maps) and a static executable.
#[test]
fn s2f_toolchain_identity_is_complete_and_runtime_static() {
    let plan = plan();
    let mut s = plan
        .prepare(model(), &roots(1), config(1, 10))
        .ok()
        .unwrap();
    loop {
        s.submit().unwrap();
        if settle(&mut s).unwrap() == Progress::Published {
            break;
        }
    }
    let r = s.execution_receipt().unwrap();
    let m = r.manifest().clone();
    assert_eq!(m["schema_version"], 2);
    let (_, work) = s.reclaim().ok().unwrap();
    let rec = work.record().clone();
    let roles: Vec<&str> = rec.toolchain.iter().map(|p| p.role).collect();
    assert_eq!(
        roles,
        ["shell", "driver", "cc1plus", "collect2", "as", "ld"]
    );
    for p in &rec.toolchain {
        assert_eq!(p.sha256, sha256(&fs::read(&p.path).unwrap()), "{}", p.role);
        assert!(!p.libraries.is_empty(), "{}: no libraries", p.role);
        for (lib, h) in &p.libraries {
            assert_eq!(*h, sha256(&fs::read(lib).unwrap()), "{}", lib.display());
        }
    }
    assert_eq!(rec.toolchain[1].path, rec.compiler_path);
    let named = Command::new(&rec.compiler_path)
        .arg("-print-prog-name=cc1plus")
        .output()
        .unwrap();
    assert_eq!(
        rec.toolchain[2].path,
        fs::canonicalize(String::from_utf8_lossy(&named.stdout).trim()).unwrap()
    );
    // Static executable: no interpreter (checked independently by readelf).
    assert!(rec.runtime_static && rec.runtime_libraries.is_empty());
    let readelf = Command::new("readelf")
        .arg("-l")
        .arg(&work.exe)
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&readelf.stdout);
    assert!(
        readelf.status.success() && text.contains("Program Headers"),
        "{text}"
    );
    assert!(!text.contains("INTERP"), "{text}");
    assert_eq!(
        m["model"]["runtime"],
        serde_json::json!({"static": true, "libraries": []})
    );
    assert_eq!(
        m["execution"]["compile_stage"]["in_runtime_host_cap"],
        false
    );
    for (name, path) in [
        ("cc1plus", "/model/toolchain/2/sha256"),
        ("ld library", "/model/toolchain/5/libraries/0/sha256"),
        ("runtime", "/model/runtime/static"),
        (
            "compile cap",
            "/execution/compile_stage/hard/address_space_bytes_per_process",
        ),
        ("disk bound", "/ledger/disk/steady_state_session_bytes"),
    ] {
        let mut forged = m.clone();
        *forged.pointer_mut(path).unwrap() = serde_json::json!("00");
        assert_eq!(
            r.verify_manifest(&forged),
            Err(provenance::Error::ManifestMismatch),
            "{name}"
        );
    }
}

/// F1: a dynamically linked executable under a static-required profile fails
/// closed before any epoch.
#[test]
fn s2f_dynamic_product_executable_fails_closed() {
    let plan = plan();
    let mut m = model();
    m.profile = Some((UBSAN_FLAGS_V1, true));
    let refused = plan.prepare(m, &roots(1), config(1, 10)).err().unwrap();
    assert_eq!(
        refused.error,
        Error::Backend("model executable is dynamically linked".into())
    );
}

/// F2: the compile stage is a separate stage with finite per-process
/// address-space and per-file caps (not part of the runtime 256 MiB); a
/// compile that cannot fit fails closed.
#[test]
fn s2f_compile_stage_runs_under_a_finite_cap_and_fails_closed() {
    assert_eq!(toolchain::COMPILE_ADDRESS_SPACE_BYTES, 1 << 30);
    assert_eq!(toolchain::COMPILE_FILE_BYTES, 64 << 20);
    assert_eq!(toolchain::COMPILE_MAX_PROCESSES, 3);
    assert_eq!(toolchain::COMPILE_DISK_BYTES, 256 << 20);
    let (dir, _, _) = recording_compiler("guard");
    let (ok, out, _) = toolchain::guarded(
        Path::new("/bin/sh"),
        &["-c".as_ref(), "ulimit -v; ulimit -f".as_ref()],
        &[],
        &dir,
        "limits",
        toolchain::Guard {
            address_space_bytes: 1 << 30,
            file_bytes: 64 << 20,
            deadline: Duration::from_secs(10),
        },
    )
    .unwrap();
    assert!(ok);
    assert_eq!(String::from_utf8_lossy(&out), "1048576\n131072\n");
    let _ = fs::remove_dir_all(&dir);
    let plan = plan();
    let mut m = model();
    m.compile_address_space = Some(64 * 1024 * 1024);
    let refused = plan.prepare(m, &roots(1), config(1, 10)).err().unwrap();
    let Error::Backend(message) = &refused.error else {
        panic!("{:?}", refused.error);
    };
    eprintln!(
        "compile stage under a 64 MiB cap: {}",
        message.lines().next().unwrap_or("")
    );
}

/// F4: host bytes include the epoch digests and the record bound; the real
/// scratch footprint stays within the recorded disk bound with one stderr
/// file; a disk cap below the bound is refused before any compiler call.
#[test]
fn s2f_host_and_disk_accounting() {
    let plan = plan();
    let c = config(33, 10);
    let d = plan.sizes(c).unwrap().device_bytes;
    assert_eq!(
        CpuGroupModel::host_bytes(d),
        Some(child_cap(d).unwrap() + 2 * 64 * 1024 + 1024 * 32 + 4 * 1024 * 1024)
    );
    let mut s = plan.prepare(model(), &roots(33), c).ok().unwrap();
    loop {
        s.submit().unwrap();
        if settle(&mut s).unwrap() == Progress::Published {
            break;
        }
    }
    let (_, work) = s.reclaim().ok().unwrap();
    let bound = work.record().steady_state_disk_bytes;
    assert_eq!(bound, COMPILE_DISK_BYTES + 3 * (d as u64 + 72));
    let used = toolchain::dir_bytes(&work.dir.0).unwrap();
    assert!(used <= bound, "{used} > {bound}");
    let names: Vec<String> = fs::read_dir(&work.dir.0)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(names.iter().any(|n| n == "epoch.stderr"));
    assert!(!names.iter().any(|n| n.starts_with("epoch-")), "{names:?}");
    drop(work);
    let (dir, script, calls) = recording_compiler("disk");
    let mut m = CpuGroupModel::new(&script, Duration::from_secs(5)).unwrap();
    m.disk_cap = Some(COMPILE_DISK_BYTES + 1);
    let refused = plan.prepare(m, &roots(33), c).err().unwrap();
    assert_eq!(refused.error, Error::Backend("session disk bound".into()));
    assert!(!calls.exists());
    let _ = fs::remove_dir_all(&dir);
}

/// F5: lanes and every buffer length come from the plan facts; any mismatch
/// is refused before any compiler call.
#[test]
fn s2f_workset_layout_comes_from_plan_facts() {
    let plan = plan();
    let c = config(33, 10);
    let image = plan.image(c, plan.sizes(c).unwrap(), &roots(33)).unwrap();
    let (dir, script, calls) = recording_compiler("layout");
    let mut m = CpuGroupModel::new(&script, Duration::from_secs(5)).unwrap();
    type Tamper = (&'static str, fn(&mut Buffers));
    let tampers: [Tamper; 7] = [
        ("input", |b| b.input.push(0)),
        ("output", |b| {
            b.output.pop();
        }),
        ("trace", |b| {
            b.trace.pop();
        }),
        ("config", |b| b.config.push(0)),
        ("heap", |b| {
            b.heap.pop();
        }),
        ("cleanup", |b| b.cleanup.push(0)),
        ("tasks", |b| {
            b.tasks.pop();
        }),
    ];
    for (name, tamper) in tampers {
        let mut forged = image.clone();
        tamper(&mut forged);
        assert_eq!(
            m.prepare(plan.artifact().artifact(), &forged).err(),
            Some("workset layout".to_owned()),
            "{name}"
        );
    }
    assert!(!calls.exists());
    // The genuine image passes the layout check and reaches the compiler.
    assert!(m.prepare(plan.artifact().artifact(), &image).is_err());
    assert!(calls.exists());
    let _ = fs::remove_dir_all(&dir);
}

/// R1/R2/R4: the receipt classifies every compile-stage bound (hard vs
/// modelled vs post-compile check), labels the disk bound as steady-state,
/// and lists every identity exclusion; relabelling any of them is refused.
#[test]
fn s2r_receipt_classifies_bounds_and_identity_scope() {
    use crate::general_net::provenance::owned_numeric_execution as receipt;
    let plan = plan();
    let mut s = plan
        .prepare(model(), &roots(1), config(1, 10))
        .ok()
        .unwrap();
    loop {
        s.submit().unwrap();
        if settle(&mut s).unwrap() == Progress::Published {
            break;
        }
    }
    let r = s.execution_receipt().unwrap();
    let m = r.manifest().clone();
    assert_eq!(
        m["execution"]["compile_stage"],
        serde_json::json!({
            "hard": {"address_space_bytes_per_process": 1u64 << 30,
                "file_bytes_per_file": 64u64 << 20, "probe_deadline_ms": 30_000,
                "compile_deadline_ms": 120_000,
                "enforcement": "rlimits inherited by every compile-stage process; process-group kill and reap on deadline"},
            "modelled": {"max_concurrent_processes": 3, "aggregate_address_space_bytes": 3u64 << 30,
                "basis": "gcc driver runs its pipeline sequentially (no -pipe)", "enforced": false},
            "post_compile_check": {"scratch_bytes": 256u64 << 20, "peak_bounded": false},
            "in_runtime_host_cap": false
        })
    );
    let scope = &m["model"]["identity_scope"];
    assert_eq!(scope["reproducible_build"], false);
    assert_eq!(
        scope["excluded"],
        serde_json::json!([
            "kernel",
            "vDSO",
            "system headers",
            "static archives (libc.a, libstdc++.a, libgcc.a, libgcc_eh.a)",
            "crt objects",
            "gcc spec files"
        ])
    );
    assert!(
        scope["toolchain_hash_window"]
            .as_str()
            .unwrap()
            .contains("no atomic pinning")
    );
    let disk = &m["ledger"]["disk"];
    assert_eq!(disk["steady_state_cap"], 512u64 << 20);
    assert!(
        disk["class"]
            .as_str()
            .unwrap()
            .contains("not a whole-session peak bound")
    );
    for (name, path, value) in [
        (
            "modelled enforced",
            "/execution/compile_stage/modelled/enforced",
            serde_json::json!(true),
        ),
        (
            "peak bounded",
            "/execution/compile_stage/post_compile_check/peak_bounded",
            serde_json::json!(true),
        ),
        (
            "aggregate",
            "/execution/compile_stage/modelled/aggregate_address_space_bytes",
            serde_json::json!(1),
        ),
        (
            "reproducible",
            "/model/identity_scope/reproducible_build",
            serde_json::json!(true),
        ),
        (
            "exclusions",
            "/model/identity_scope/excluded",
            serde_json::json!(["kernel", "vDSO"]),
        ),
        (
            "hash window",
            "/model/identity_scope/toolchain_hash_window",
            serde_json::json!("pinned"),
        ),
        (
            "disk class",
            "/ledger/disk/class",
            serde_json::json!("peak"),
        ),
        ("schema", "/schema_version", serde_json::json!(1)),
    ] {
        let mut forged = m.clone();
        *forged.pointer_mut(path).unwrap() = value;
        assert_eq!(
            r.verify_manifest(&forged),
            Err(provenance::Error::ManifestMismatch),
            "{name}"
        );
    }
    // The builder refuses an unverified toolchain, a steady-state disk bound
    // over its cap and a zero modelled bound.
    let (_, work) = s.reclaim().ok().unwrap();
    let rec = work.record().clone();
    let terminals = vec![(0u64, [0u8; 32]); 1];
    let build = |reverified: bool, disk: u64, processes: u64| {
        receipt::cpu_group_model(
            super::super::ReceiptSeal(()),
            receipt::Run {
                compile: plan.artifact(),
                plan_nonce: plan.nonce(),
                config: [1, 1000, 200, 64, 10, 21],
                input_sha256: [0; 32],
                initial_sha256: [0; 32],
                epochs: &[[0; 32]],
                terminals: &terminals,
                work_reserved: 1,
                work_used: 0,
                host_bytes: 0,
                model: receipt::ModelIdentity {
                    compiler_path: &rec.compiler_path,
                    compiler_sha256: rec.compiler_sha256,
                    compiler_version_sha256: rec.compiler_version_sha256,
                    host_triple: &rec.host_triple,
                    flags: rec.flags,
                    shim_sha256: rec.shim_sha256,
                    driver_sha256: rec.driver_sha256,
                    model_source_sha256: rec.model_source_sha256,
                    executable_sha256: rec.executable_sha256,
                    kernel_sha256: rec.kernel_sha256,
                    stack_bytes: rec.stack_bytes,
                    child_address_space_bytes: rec.child_address_space_bytes,
                    lanes: rec.lanes,
                    words: rec.words,
                    toolchain: rec
                        .toolchain
                        .iter()
                        .map(|p| receipt::ProgramIdentity {
                            role: p.role,
                            path: &p.path,
                            sha256: p.sha256,
                            libraries: &p.libraries,
                        })
                        .collect(),
                    runtime_static: rec.runtime_static,
                    runtime_libraries: &rec.runtime_libraries,
                    compile_stage: receipt::CompileStage {
                        modelled_max_processes: processes,
                        ..rec.compile_stage
                    },
                    steady_state_disk_bytes: disk,
                    steady_state_disk_cap: SESSION_DISK_BYTES,
                    toolchain_reverified: reverified,
                },
            },
        )
        .err()
    };
    let disk = rec.steady_state_disk_bytes;
    assert_eq!(build(true, disk, 3), None);
    assert_eq!(
        build(false, disk, 3),
        Some(provenance::Error::ExecutionMismatch)
    );
    assert_eq!(
        build(true, SESSION_DISK_BYTES + 1, 3),
        Some(provenance::Error::ExecutionMismatch)
    );
    assert_eq!(
        build(true, disk, 0),
        Some(provenance::Error::ExecutionMismatch)
    );
}

/// R3: every recorded toolchain file is re-hashed after the compile; a change
/// (here to a PATH-resolved copy of `as` that the compile really uses) is
/// refused. A file changed and restored between the checks is not detected.
#[test]
fn s2r_toolchain_change_after_compile_is_refused() {
    use std::os::unix::fs::PermissionsExt;
    let original = std::env::var_os("PATH").unwrap_or_default();
    // Copy the assembler the driver really runs, under the name it searches
    // PATH for: `as` on most hosts, `x86_64-linux-gnu-as` on Debian/Ubuntu
    // x86_64. Even an absolute query result can change with PATH; verify
    // below that this driver actually selects the private copy.
    let driver = toolchain::resolve(&compiler(), &original).unwrap();
    let named = Command::new(&driver)
        .arg("-print-prog-name=as")
        .env_clear()
        .env("PATH", &original)
        .output()
        .unwrap();
    assert!(named.status.success());
    let named = PathBuf::from(String::from_utf8(named.stdout).unwrap().trim());
    let real_as = toolchain::resolve(&named, &original).unwrap();
    let dir = scratch().unwrap();
    let copy = dir.0.join(named.file_name().unwrap());
    fs::copy(&real_as, &copy).unwrap();
    fs::set_permissions(&copy, fs::Permissions::from_mode(0o755)).unwrap();
    let mut paths = vec![dir.0.clone()];
    paths.extend(std::env::split_paths(&original));
    let path_env = std::env::join_paths(paths).unwrap();
    let selected = Command::new(&driver)
        .arg("-print-prog-name=as")
        .env_clear()
        .env("PATH", &path_env)
        .output()
        .unwrap();
    assert!(selected.status.success());
    let selected = PathBuf::from(String::from_utf8(selected.stdout).unwrap().trim());
    assert_eq!(
        toolchain::resolve(&selected, &path_env).unwrap(),
        fs::canonicalize(&copy).unwrap(),
        "compiler did not select the private assembler"
    );
    let plan = plan();
    // Control: the copy is the recorded and used `as`; the session admits.
    let mut m = model();
    m.path_env = Some(path_env.clone());
    let mut s = plan.prepare(m, &roots(1), config(1, 10)).ok().unwrap();
    let (_, work) = s.reclaim().ok().unwrap();
    let recorded = work
        .record()
        .toolchain
        .iter()
        .find(|p| p.role == "as")
        .unwrap();
    assert_eq!(recorded.path, fs::canonicalize(&copy).unwrap());
    assert!(work.record().toolchain_reverified);
    drop(work);
    // Changing that file after the compile is refused.
    let mut m = model();
    m.path_env = Some(path_env);
    let target = copy.clone();
    m.after_compile = Some(Box::new(move || {
        use std::io::Write as _;
        let mut f = fs::OpenOptions::new().append(true).open(&target).unwrap();
        f.write_all(&[0]).unwrap();
    }));
    let refused = plan.prepare(m, &roots(1), config(1, 10)).err().unwrap();
    assert_eq!(
        refused.error,
        Error::Backend(format!(
            "toolchain changed during compile: {}",
            fs::canonicalize(&copy).unwrap().display()
        ))
    );
}
