//! ABI12 stage 2c S1 lifecycle tests, compiled as a child module of
//! `general_net::cuda_owned_numeric_host` (`#[cfg(test)] #[path]`). Only
//! test-only fault injectors implement the sealed backend: there is no
//! compiler, loader, CUDA or NVRTC, and no positive epoch acceptance (that
//! needs the S2 CPU group model). Every counter below is the injector's own.
use super::*;
use crate::numeric_contract::F32Bits;
use std::{cell::RefCell, collections::VecDeque, rc::Rc};

const SOURCE: &str = "fn run(a: f32, n: u32) -> f32 { var xs: Array<f32> = filled_f32(n + 1u32, a); xs[0u32] = xs[0u32] * a + a; return xs[0u32]; } fn main() -> i32 { return 0; }";
const MIN: i64 = i64::MIN;
const S: u64 = 0x5a5a_5a5a_5a5a_5a5a;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Calls {
    prepare: usize,
    submit: usize,
    poll: usize,
    read: usize,
    poison: usize,
    work_drops: usize,
    backend_drops: usize,
}

type Forge = Box<dyn Fn(&mut Slots<'_>)>;

/// Scripted fault injector. `HOST` is the declared host-byte multiplier of
/// the device workset (`0`: cannot be bounded).
struct Injector<const HOST: usize> {
    calls: Rc<RefCell<Calls>>,
    submits: VecDeque<Submission>,
    polls: VecDeque<FenceObservation>,
    /// `None`: `read_into` writes nothing. `Some`: copy the work's image,
    /// then apply the forge.
    forge: Option<Forge>,
    captured: Rc<RefCell<Option<Buffers>>>,
}
struct Work {
    image: Buffers,
    calls: Rc<RefCell<Calls>>,
}
impl Drop for Work {
    fn drop(&mut self) {
        self.calls.borrow_mut().work_drops += 1;
    }
}
impl<const H: usize> Drop for Injector<H> {
    fn drop(&mut self) {
        self.calls.borrow_mut().backend_drops += 1;
    }
}
impl<const H: usize> sealed::Sealed for Injector<H> {}
impl<const H: usize> Backend for Injector<H> {
    type Work = Work;
    const KIND: BackendKind = BackendKind::TestInjector;
    fn host_bytes(device_bytes: usize) -> Option<usize> {
        if H == 0 {
            None
        } else {
            device_bytes.checked_mul(H)
        }
    }
    fn prepare(&mut self, _artifact: &compiled::Artifact, image: &Buffers) -> Result<Work, String> {
        self.calls.borrow_mut().prepare += 1;
        *self.captured.borrow_mut() = Some(image.clone());
        Ok(Work {
            image: image.clone(),
            calls: self.calls.clone(),
        })
    }
    fn submit(&mut self, _work: &mut Work) -> Submission {
        self.calls.borrow_mut().submit += 1;
        self.submits.pop_front().unwrap_or(Submission::Submitted)
    }
    fn poll(&mut self, _work: &mut Work) -> FenceObservation {
        self.calls.borrow_mut().poll += 1;
        self.polls.pop_front().unwrap_or(FenceObservation::Complete)
    }
    fn read_into(&self, work: &Work, mut slots: Slots<'_>) -> Result<(), String> {
        self.calls.borrow_mut().read += 1;
        if let Some(forge) = &self.forge {
            slots.input.copy_from_slice(&work.image.input);
            slots.output.copy_from_slice(&work.image.output);
            slots.trace.copy_from_slice(&work.image.trace);
            slots.config.copy_from_slice(&work.image.config);
            slots.heap.copy_from_slice(&work.image.heap);
            slots.cleanup.copy_from_slice(&work.image.cleanup);
            slots.tasks.copy_from_slice(&work.image.tasks);
            forge(&mut slots);
        }
        Ok(())
    }
    fn poison(&mut self) {
        self.calls.borrow_mut().poison += 1;
    }
}

type Captured = Rc<RefCell<Option<Buffers>>>;

fn injector<const H: usize>(
    submits: &[Submission],
    polls: &[FenceObservation],
    forge: Option<Forge>,
) -> (Injector<H>, Rc<RefCell<Calls>>, Captured) {
    let calls = Rc::new(RefCell::new(Calls::default()));
    let captured = Rc::new(RefCell::new(None));
    (
        Injector {
            calls: calls.clone(),
            submits: submits.iter().copied().collect(),
            polls: polls.iter().copied().collect(),
            forge,
            captured: captured.clone(),
        },
        calls,
        captured,
    )
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
fn config() -> Config {
    Config {
        roots: 33,
        fuel: 1000,
        transitions: 30,
        trace_events: 16,
        quantum: 3,
        max_submissions: 11,
    }
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
fn calls(c: &Rc<RefCell<Calls>>) -> Calls {
    c.borrow().clone()
}

#[test]
fn s1_initial_image_is_exact() {
    let plan = plan();
    let (b, c, captured) = injector::<1>(&[], &[], None);
    let session = plan.prepare(b, &roots(33), config()).ok().unwrap();
    assert_eq!(session.state(), State::Ready);
    assert_eq!(
        calls(&c),
        Calls {
            prepare: 1,
            ..Default::default()
        }
    );
    let image = captured.borrow().clone().unwrap();
    // Independent layout literals for this program and storage: 2 params,
    // heap 16 * 4 + 64, cleanup 32 * 4, task 64 * 8 * 23 + 144 bytes;
    // 33 roots -> 64 lanes; 64-word red zones.
    let (lanes, red, heap, cleanup, task) = (64, 64, 128, 128, 64 * 8 * 23 + 144);
    let trace = 7 * 16;
    assert_eq!(image.input.len(), 2 * lanes + red);
    assert_eq!(image.output.len(), 10 * lanes + red);
    assert_eq!(image.trace.len(), trace * lanes + red);
    assert_eq!(image.config.len(), 5 + red);
    assert_eq!(image.heap.len(), heap * lanes + red);
    assert_eq!(image.cleanup.len(), cleanup * lanes + red);
    assert_eq!(image.tasks.len(), (task * lanes) + red * 8);
    let s = S as i64;
    for lane in 0..lanes {
        let live = lane < 33;
        let input = &image.input[lane * 2..lane * 2 + 2];
        if live {
            assert_eq!(input, [0x3f80_0000 + lane as i64, (lane % 4) as i64]);
        } else {
            assert_eq!(input, [0, 0]);
        }
        let want = if live { MIN } else { s };
        assert!(
            image.output[lane * 10..lane * 10 + 10]
                .iter()
                .all(|w| *w == want)
        );
        let want_u = if live { 0 } else { S };
        let want_i = if live { 0 } else { s };
        assert!(
            image.trace[lane * trace..(lane + 1) * trace]
                .iter()
                .all(|w| *w == want_u)
        );
        assert!(
            image.heap[lane * heap..(lane + 1) * heap]
                .iter()
                .all(|w| *w == want_i)
        );
        assert!(
            image.cleanup[lane * cleanup..(lane + 1) * cleanup]
                .iter()
                .all(|w| *w == want_u)
        );
        assert!(
            image.tasks[lane * task..(lane + 1) * task]
                .iter()
                .all(|b| *b == 0)
        );
    }
    assert!(image.input[2 * lanes..].iter().all(|w| *w == s));
    assert!(image.output[10 * lanes..].iter().all(|w| *w == s));
    assert!(image.trace[trace * lanes..].iter().all(|w| *w == S));
    assert_eq!(image.config[..5], [33, 1000, 30, 16, 3]);
    assert!(image.config[5..].iter().all(|w| *w == S));
    assert!(image.heap[heap * lanes..].iter().all(|w| *w == s));
    assert!(image.cleanup[cleanup * lanes..].iter().all(|w| *w == S));
    assert!(image.tasks[task * lanes..].iter().all(|b| *b == 0x5a));
    // D5: the session's host total is within 256 MiB and includes two full
    // snapshots plus the injector's declared copy.
    let device: usize = 8
        * (image.input.len()
            + image.output.len()
            + image.trace.len()
            + image.config.len()
            + image.heap.len()
            + image.cleanup.len())
        + image.tasks.len();
    assert!(session.host_bytes() >= 3 * device);
    assert!(session.host_bytes() <= 256 * 1024 * 1024);
    assert_eq!(session.epochs_bound(), 11);
    assert!(session.ledger().reserved() <= 1 << 30);
    assert_eq!(session.ledger().used(), 0);
}

#[test]
fn s1_every_bound_refuses_before_any_backend_call() {
    let plan = plan();
    let c = config();
    let typed = {
        let mut r = roots(33);
        r[4][0] = eval::Value::I32(1);
        r
    };
    let short = {
        let mut r = roots(33);
        r[2].pop();
        r
    };
    let cases: Vec<(&str, Config, Vec<Vec<eval::Value>>, Error)> = vec![
        ("root count", c, roots(32), Error::Input),
        ("arity", c, short, Error::Input),
        ("argument type", c, typed, Error::Input),
        ("roots 0", Config { roots: 0, ..c }, vec![], Error::Limit),
        (
            "roots 1025",
            Config { roots: 1025, ..c },
            vec![],
            Error::Limit,
        ),
        ("fuel 0", Config { fuel: 0, ..c }, roots(33), Error::Limit),
        (
            "fuel max+1",
            Config {
                fuel: 1_000_001,
                ..c
            },
            roots(33),
            Error::Limit,
        ),
        (
            "transitions max+1",
            Config {
                transitions: 1_000_001,
                ..c
            },
            roots(33),
            Error::Limit,
        ),
        (
            "trace max+1",
            Config {
                trace_events: 1_000_001,
                ..c
            },
            roots(33),
            Error::Limit,
        ),
        (
            "quantum 0",
            Config { quantum: 0, ..c },
            roots(33),
            Error::Limit,
        ),
        (
            "quantum max+1",
            Config {
                quantum: 1_000_001,
                ..c
            },
            roots(33),
            Error::Limit,
        ),
        (
            "D4: ceil(30/3)+1 = 11 > 10",
            Config {
                max_submissions: 10,
                ..c
            },
            roots(33),
            Error::Limit,
        ),
        (
            "D4: zero submissions",
            Config {
                max_submissions: 0,
                ..c
            },
            roots(33),
            Error::Limit,
        ),
        (
            "D4: 1025 submissions",
            Config {
                max_submissions: 1025,
                ..c
            },
            roots(33),
            Error::Limit,
        ),
        (
            "D4: q=1, T=1e6 needs 1e6+1",
            Config {
                transitions: 1_000_000,
                quantum: 1,
                max_submissions: 1024,
                ..c
            },
            roots(33),
            Error::Limit,
        ),
        (
            "device > 64 MiB",
            Config {
                roots: 1024,
                trace_events: 2000,
                ..c
            },
            roots(1024),
            Error::Limit,
        ),
        (
            "work > 2^30",
            Config {
                roots: 1024,
                transitions: 1_000_000,
                quantum: 1_000_000,
                trace_events: 0,
                max_submissions: 2,
                ..c
            },
            roots(1024),
            Error::Work,
        ),
    ];
    for (name, config, roots, error) in cases {
        let (b, calls_, _) = injector::<1>(&[], &[], None);
        let refused = plan.prepare(b, &roots, config).err().unwrap();
        assert_eq!(refused.error, error, "{name}");
        assert_eq!(
            calls(&calls_),
            Calls::default(),
            "{name}: backend was called"
        );
    }
    // D5: a backend that cannot bound its host bytes, and one over 256 MiB.
    // (The refusal hands the untouched backend back; it is dropped only after
    // its counters are read.)
    let (b, c0, _) = injector::<0>(&[], &[], None);
    let refused = plan.prepare(b, &roots(33), c).err().unwrap();
    assert_eq!(refused.error, Error::HostBytes);
    assert_eq!(calls(&c0), Calls::default());
    drop(refused);
    let (b, c1, _) = injector::<1000>(&[], &[], None);
    let refused = plan.prepare(b, &roots(33), c).err().unwrap();
    assert_eq!(refused.error, Error::HostBytes);
    assert_eq!(calls(&c1), Calls::default());
    drop(refused);
    // Positive control at the exact D4 boundary: 11 submissions admit.
    let (b, ok, _) = injector::<1>(&[], &[], None);
    assert!(plan.prepare(b, &roots(33), c).is_ok());
    assert_eq!(calls(&ok).prepare, 1);
}

fn quarantined_drop_frees_nothing(c: &Rc<RefCell<Calls>>) {
    let after = calls(c);
    assert_eq!((after.work_drops, after.backend_drops), (0, 0));
    assert_eq!(after.poison, 1);
    assert_eq!(after.read, 0);
}

#[test]
fn s1_unknown_submit_poisons_once_and_never_touches_the_work_again() {
    let plan = plan();
    let (b, c, _) = injector::<1>(&[Submission::Unknown], &[], None);
    let mut s = plan.prepare(b, &roots(33), config()).ok().unwrap();
    assert_eq!(s.submit(), Err(Error::Unknown));
    assert_eq!(s.state(), State::Quarantined);
    let before = calls(&c);
    assert_eq!((before.submit, before.poison), (1, 1));
    assert_eq!(s.submit(), Err(Error::State(State::Quarantined)));
    assert_eq!(s.poll(), Err(Error::State(State::Quarantined)));
    assert_eq!(s.result(0).err(), Some(Error::State(State::Quarantined)));
    s.cancel();
    assert_eq!(s.state(), State::Quarantined);
    assert_eq!(s.reclaim().err(), Some(Error::State(State::Quarantined)));
    assert_eq!(calls(&c), before, "no backend call after quarantine");
    drop(s);
    quarantined_drop_frees_nothing(&c);
}

#[test]
fn s1_unknown_fence_poisons_once_and_never_polls_again() {
    let plan = plan();
    for polls in [
        &[FenceObservation::Unknown][..],
        &[FenceObservation::Pending, FenceObservation::Unknown][..],
    ] {
        let (b, c, _) = injector::<1>(&[], polls, None);
        let mut s = plan.prepare(b, &roots(33), config()).ok().unwrap();
        s.submit().unwrap();
        if polls.len() == 2 {
            assert_eq!(s.poll(), Ok(Progress::Pending));
        }
        assert_eq!(s.poll(), Err(Error::Unknown));
        let before = calls(&c);
        assert_eq!(before.poll, polls.len());
        // A later Complete must never be observed (the 2a gate would accept
        // it and move Quarantined -> Cancelled): the session never polls.
        assert_eq!(s.poll(), Err(Error::State(State::Quarantined)));
        assert_eq!(s.submit(), Err(Error::State(State::Quarantined)));
        assert!(s.reclaim().is_err());
        assert_eq!(calls(&c), before);
        drop(s);
        quarantined_drop_frees_nothing(&c);
    }
    // Unknown while cancel is pending quarantines the same way.
    let (b, c, _) = injector::<1>(&[], &[FenceObservation::Unknown], None);
    let mut s = plan.prepare(b, &roots(33), config()).ok().unwrap();
    s.submit().unwrap();
    s.cancel();
    assert_eq!(s.state(), State::CancelPending);
    assert_eq!(s.poll(), Err(Error::Unknown));
    assert_eq!(s.poll(), Err(Error::State(State::Quarantined)));
    drop(s);
    quarantined_drop_frees_nothing(&c);
}

#[test]
fn s1_cancel_reclaim_and_drop_follow_quiescence() {
    let plan = plan();
    // Cancel between epochs: no backend call, reclaim returns both.
    let (b, c, _) = injector::<1>(&[], &[], None);
    let mut s = plan.prepare(b, &roots(33), config()).ok().unwrap();
    s.cancel();
    assert_eq!(s.state(), State::Cancelled);
    assert_eq!(s.submit(), Err(Error::State(State::Cancelled)));
    assert_eq!(s.poll(), Err(Error::State(State::Cancelled)));
    let (backend, work) = s.reclaim().ok().unwrap();
    assert_eq!(s.state(), State::Reclaimed);
    assert!(s.reclaim().is_err());
    assert_eq!(s.submit(), Err(Error::State(State::Reclaimed)));
    assert_eq!(calls(&c).submit, 0);
    drop(work);
    drop(backend);
    assert_eq!((calls(&c).work_drops, calls(&c).backend_drops), (1, 1));

    // Cancel in flight: poll to quiescence, no readback, then reclaim.
    let (b, c, _) = injector::<1>(
        &[],
        &[FenceObservation::Pending, FenceObservation::Complete],
        None,
    );
    let mut s = plan.prepare(b, &roots(33), config()).ok().unwrap();
    s.submit().unwrap();
    // Reclaim is refused while in flight and hands the session back.
    assert_eq!(s.reclaim().err(), Some(Error::State(State::InFlight)));
    s.cancel();
    assert_eq!(s.state(), State::CancelPending);
    assert_eq!(s.poll(), Ok(Progress::Pending));
    assert_eq!(s.poll(), Ok(Progress::Cancelled));
    assert_eq!(s.state(), State::Cancelled);
    assert_eq!(calls(&c).read, 0);
    assert_eq!(s.submit(), Err(Error::State(State::Cancelled)));
    assert!(s.reclaim().is_ok());
    assert_eq!(calls(&c).work_drops, 1);

    // Dropping in-flight work forgets both work and backend (D2).
    let (b, c, _) = injector::<1>(&[], &[], None);
    let mut s = plan.prepare(b, &roots(33), config()).ok().unwrap();
    s.submit().unwrap();
    drop(s);
    let after = calls(&c);
    assert_eq!(
        (after.work_drops, after.backend_drops, after.poison),
        (0, 0, 0)
    );

    // Quiescent drop frees the work before the backend.
    let (b, c, _) = injector::<1>(&[], &[], None);
    drop(plan.prepare(b, &roots(33), config()).ok().unwrap());
    assert_eq!((calls(&c).work_drops, calls(&c).backend_drops), (1, 1));
}

#[test]
fn s1_failed_quiescent_and_not_submitted_never_read() {
    let plan = plan();
    let (b, c, _) = injector::<1>(&[], &[FenceObservation::FailedQuiescent], None);
    let mut s = plan.prepare(b, &roots(33), config()).ok().unwrap();
    s.submit().unwrap();
    assert_eq!(s.poll(), Err(Error::FailedQuiescent));
    assert_eq!(s.state(), State::FailedQuiescent);
    assert_eq!(s.submit(), Err(Error::State(State::FailedQuiescent)));
    assert_eq!(calls(&c).read, 0);
    assert!(s.reclaim().is_ok());

    let (b, c, _) = injector::<1>(&[Submission::NotSubmitted], &[], None);
    let mut s = plan.prepare(b, &roots(33), config()).ok().unwrap();
    assert_eq!(s.submit(), Err(Error::NotSubmitted));
    assert_eq!(s.state(), State::NotSubmitted);
    assert_eq!(s.submit(), Err(Error::State(State::NotSubmitted)));
    assert_eq!(calls(&c).submit, 1);
    assert!(s.reclaim().is_ok());
}

/// Run one epoch whose readback is the work image changed by `forge`.
fn forged(forge: Option<Forge>, prior_done: bool) -> (Error, Calls) {
    let plan = plan();
    let done = |tasks: &mut [u8]| {
        let frame_bytes = plan.artifact().artifact().frame_bytes();
        tasks[frame_bytes + 8..frame_bytes + 16].copy_from_slice(&1u64.to_ne_bytes());
    };
    let (b, c, _) = injector::<1>(&[], &[], forge);
    let mut s = plan.prepare(b, &roots(33), config()).ok().unwrap();
    if prior_done {
        // Synthetic prior: root 0 recorded as terminal in the last validated
        // snapshot (positive checker acceptance is S2).
        done(&mut s.previous.tasks);
    }
    s.submit().unwrap();
    let error = s.poll().err().unwrap();
    assert_eq!(s.state(), State::Failed);
    let before = calls(&c);
    assert_eq!(s.submit(), Err(Error::State(State::Failed)));
    assert_eq!(s.result(0).err(), Some(Error::State(State::Failed)));
    assert_eq!(calls(&c), before);
    let (_, failing) = s.evidence();
    assert_eq!(failing.config.len(), 5 + 64);
    assert!(s.reclaim().is_ok());
    (error, before)
}

#[test]
fn s1_forged_readbacks_are_refused() {
    use Validation as V;
    let v = |e: Validation| Error::Validation(e);
    // Nothing written: unread words never pass as unchanged.
    let (e, c) = forged(None, false);
    assert_eq!(e, v(V::Config));
    assert_eq!(c.read, 1);
    let cases: Vec<(&str, Forge, Error)> = vec![
        ("config word", Box::new(|s| s.config[2] += 1), v(V::Config)),
        (
            "config red zone",
            Box::new(|s| s.config[70 - 2] = 0),
            v(V::Config),
        ),
        (
            "input red zone",
            Box::new(|s| *s.input.last_mut().unwrap() = 0),
            v(V::RedZone { buffer: 0 }),
        ),
        (
            "output red zone",
            Box::new(|s| *s.output.last_mut().unwrap() = 0),
            v(V::RedZone { buffer: 1 }),
        ),
        (
            "trace red zone",
            Box::new(|s| *s.trace.last_mut().unwrap() = 0),
            v(V::RedZone { buffer: 2 }),
        ),
        (
            "heap red zone",
            Box::new(|s| *s.heap.last_mut().unwrap() = 0),
            v(V::RedZone { buffer: 4 }),
        ),
        (
            "cleanup red zone",
            Box::new(|s| *s.cleanup.last_mut().unwrap() = 0),
            v(V::RedZone { buffer: 5 }),
        ),
        (
            "task red zone",
            Box::new(|s| *s.tasks.last_mut().unwrap() = 0),
            v(V::RedZone { buffer: 6 }),
        ),
        (
            "tail output",
            Box::new(|s| s.output[40 * 10] = 0),
            v(V::Tail { lane: 40 }),
        ),
        (
            "tail input",
            Box::new(|s| s.input[63 * 2] = 1),
            v(V::Tail { lane: 63 }),
        ),
        (
            "tail task",
            Box::new(|s| {
                let task = s.tasks.len() / 64 - 8;
                s.tasks[33 * task] = 1;
            }),
            v(V::Tail { lane: 33 }),
        ),
        (
            "root input",
            Box::new(|s| s.input[5 * 2] ^= 1),
            v(V::Input { root: 5 }),
        ),
        (
            "pending output zero",
            Box::new(|s| s.output[..10].fill(0)),
            v(V::PendingOutput { root: 0 }),
        ),
        ("no advance", Box::new(|_| {}), v(V::Checker { root: 0 })),
    ];
    for (name, forge, want) in cases {
        let (e, c) = forged(Some(forge), false);
        assert_eq!(e, want, "{name}");
        assert_eq!((c.submit, c.poll, c.read), (1, 1, 1), "{name}");
    }
}

#[test]
fn s1_finished_root_must_stay_byte_identical() {
    let frame_bytes = plan().artifact().artifact().frame_bytes();
    let task = 64 * 8 * 23 + 144;
    let mark = move |s: &mut Slots<'_>| {
        s.tasks[frame_bytes + 8..frame_bytes + 16].copy_from_slice(&1u64.to_ne_bytes());
    };
    // Identical terminal root 0 passes its identity check; root 1 (pending,
    // unchanged) then fails the checker.
    let (e, _) = forged(Some(Box::new(move |s| mark(s))), true);
    assert_eq!(e, Error::Validation(Validation::Checker { root: 1 }));
    for (name, forge) in [
        (
            "heap",
            Box::new(move |s: &mut Slots<'_>| {
                mark(s);
                s.heap[0] = 7;
            }) as Forge,
        ),
        (
            "trace",
            Box::new(move |s: &mut Slots<'_>| {
                mark(s);
                s.trace[0] = 1;
            }),
        ),
        (
            "output",
            Box::new(move |s: &mut Slots<'_>| {
                mark(s);
                s.output[0] = 0;
            }),
        ),
        (
            "task frame",
            Box::new(move |s: &mut Slots<'_>| {
                mark(s);
                s.tasks[0] = 1;
            }),
        ),
        (
            "task counter",
            Box::new(move |s: &mut Slots<'_>| {
                mark(s);
                s.tasks[task - 8] = 1;
            }),
        ),
    ] {
        let (e, _) = forged(Some(forge), true);
        assert_eq!(
            e,
            Error::Validation(Validation::Terminal { root: 0 }),
            "{name}"
        );
    }
}

#[test]
fn s1_plan_authority_nonce_and_ledger() {
    // The only constructor is source-bound and keeps 2b precedence.
    let l = provenance::Limits::default();
    let err = |src: &[u8], entry: &str| Plan::from_source(src, entry, l, storage()).err().unwrap();
    assert_eq!(
        err(&vec![b' '; 1024 * 1024 + 1], "run"),
        Error::Provenance(provenance::Error::Limit)
    );
    assert!(matches!(
        err(b"fn (", "run"),
        Error::Provenance(provenance::Error::Source(_))
    ));
    assert_eq!(
        err(SOURCE.as_bytes(), "nope"),
        Error::Provenance(provenance::Error::Compiler(compiled::Error::NumericTarget))
    );
    // The plan holds an ABI12 compile receipt with no authority.
    let (a, b) = (plan(), plan());
    assert_eq!(a.artifact().artifact().abi_version(), 12);
    assert_eq!(
        a.artifact().manifest()["cuda_owned_numeric_continuation"]["authority"],
        serde_json::json!({"execution": false, "module_load": false, "resume": false,
            "device_qualification": false})
    );
    assert_ne!(a.nonce(), b.nonce());
    let (backend, c, _) = injector::<1>(&[], &[], None);
    let mut s = b.prepare(backend, &roots(33), config()).ok().unwrap();
    assert_eq!(s.plan_nonce(), b.nonce());
    assert_eq!(s.backend_kind(), BackendKind::TestInjector);
    // Results are unreadable before atomic publication.
    assert_eq!(s.result(0).err(), Some(Error::State(State::Ready)));
    assert_eq!(calls(&c).read, 0);
    // The session reserved exactly the plan's bound, before any backend call.
    let sizes = b.sizes(config()).unwrap();
    assert_eq!(
        Some(s.ledger().reserved()),
        b.work_units(config(), sizes, 11)
    );
    // Monotonic ledger: charges never refund or exceed the reservation.
    let mut ledger = Ledger {
        reserved: 10,
        used: 0,
    };
    ledger.charge(4).unwrap();
    assert_eq!(ledger.charge(7), Err(Error::Work));
    assert_eq!(ledger.used(), 4);
    ledger.charge(6).unwrap();
    assert_eq!(ledger.charge(1), Err(Error::Work));
    assert_eq!(ledger.charge(u64::MAX), Err(Error::Work));
    assert_eq!(ledger.used(), 10);
    s.cancel();
}

/// A backend whose submit unwinds after the fence marked the work in flight.
struct Panicking(Rc<RefCell<Calls>>);
impl Drop for Panicking {
    fn drop(&mut self) {
        self.0.borrow_mut().backend_drops += 1;
    }
}
impl sealed::Sealed for Panicking {}
impl Backend for Panicking {
    type Work = Work;
    const KIND: BackendKind = BackendKind::TestInjector;
    fn host_bytes(device_bytes: usize) -> Option<usize> {
        Some(device_bytes)
    }
    fn prepare(&mut self, _: &compiled::Artifact, image: &Buffers) -> Result<Work, String> {
        Ok(Work {
            image: image.clone(),
            calls: self.0.clone(),
        })
    }
    fn submit(&mut self, _: &mut Work) -> Submission {
        panic!("backend submit unwinds");
    }
    fn poll(&mut self, _: &mut Work) -> FenceObservation {
        unreachable!()
    }
    fn read_into(&self, _: &Work, _: Slots<'_>) -> Result<(), String> {
        unreachable!()
    }
    fn poison(&mut self) {}
}

#[test]
fn s1_unwinding_submit_forgets_work_and_backend_together() {
    let plan = plan();
    let calls = Rc::new(RefCell::new(Calls::default()));
    let mut s = plan
        .prepare(Panicking(calls.clone()), &roots(33), config())
        .ok()
        .unwrap();
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = s.submit();
    }));
    assert!(unwound.is_err());
    // The session never recorded the submission, but the fence did: drop must
    // follow the fence and free neither the work nor the backend.
    drop(s);
    let after = calls.borrow().clone();
    assert_eq!((after.work_drops, after.backend_drops), (0, 0));
}
