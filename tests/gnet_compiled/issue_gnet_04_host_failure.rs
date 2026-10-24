#![forbid(unsafe_code)]

use invar::{
    eval::Value,
    general_net::{
        compiled::{
            self, BufferRange, DeviceConfig, ExecutionState, FenceObservation, Submission, Target,
        },
        device_host::{Backend, Error, Limits, Plan, Progress, cpu::CpuModel},
        provenance,
    },
};
use std::{cell::Cell, rc::Rc};
fn plan() -> Plan {
    Plan::from_source(b"fn leaf(x:u32)->u32{return wrapping_add(x,1u32);} fn run(x:u32)->u32{let a=leaf(x);let b=leaf(x);return wrapping_add(a,b);}","run",Target::Cuda,Limits { owned: compiled::OwnedLimits { elements:0, allocations:1, cleanup_events:0 }, private: compiled::SpeculationLimits { slots:2, ..Default::default() }, ..Default::default() }).unwrap()
}
fn config() -> DeviceConfig {
    DeviceConfig {
        fuel: 1000,
        transitions: 10000,
        trace_events: 2000,
        quantum: 1,
        rounds: 1,
        capacity: 2,
    }
}
struct Counters {
    prepare: Cell<usize>,
    reads: Cell<usize>,
    launches: Cell<usize>,
    drops: Cell<usize>,
    observation: Cell<FenceObservation>,
    submission: Cell<Submission>,
}
impl Default for Counters {
    fn default() -> Self {
        Self {
            prepare: Cell::new(0),
            reads: Cell::new(0),
            launches: Cell::new(0),
            drops: Cell::new(0),
            observation: Cell::new(FenceObservation::Pending),
            submission: Cell::new(Submission::Submitted),
        }
    }
}
struct Work {
    data: [Vec<u64>; 12],
    counts: Rc<Counters>,
}
impl Drop for Work {
    fn drop(&mut self) {
        self.counts.drops.set(self.counts.drops.get() + 1);
    }
}
struct FaultBackend {
    counts: Rc<Counters>,
    poisoned: bool,
}
impl Default for FaultBackend {
    fn default() -> Self {
        Self {
            counts: Rc::new(Counters::default()),
            poisoned: false,
        }
    }
}
impl Backend for FaultBackend {
    type Work = Work;
    fn label() -> provenance::FenceBackend {
        provenance::FenceBackend::CpuGroupModel
    }
    fn poisoned(&self) -> bool {
        self.poisoned
    }
    fn poison(&mut self) {
        self.poisoned = true;
    }
    fn prepare(
        &mut self,
        _: &compiled::Artifact,
        bytes: [usize; 12],
        _: &[u64],
        _: [u64; 6],
    ) -> Result<Work, String> {
        self.counts.prepare.set(self.counts.prepare.get() + 1);
        Ok(Work {
            data: std::array::from_fn(|i| vec![0; bytes[i] / 8]),
            counts: self.counts.clone(),
        })
    }
    fn ranges(work: &Work) -> [BufferRange; 12] {
        std::array::from_fn(|i| BufferRange {
            device: 1,
            allocation: work.data[i].as_ptr() as usize as u64,
            offset: 0,
            bytes: work.data[i].len() * 8,
        })
    }
    fn submit(work: &mut Work) -> Submission {
        work.counts.launches.set(work.counts.launches.get() + 1);
        work.counts.submission.get()
    }
    fn fence(work: &mut Work) -> FenceObservation {
        work.counts.observation.get()
    }
    fn read(work: &Work) -> Result<[Vec<u64>; 12], String> {
        work.counts.reads.set(work.counts.reads.get() + 1);
        Err("fault backend cannot certify a result".into())
    }
}
#[test]
fn pending_and_unknown_retain_entire_work_and_never_read_or_replay() {
    let plan = plan();
    let args = [Value::U32(3)];
    for unknown in [false, true] {
        let mut backend = FaultBackend::default();
        let counts = backend.counts.clone();
        if unknown {
            counts.submission.set(Submission::Unknown);
        }
        let mut session = plan.prepare(&mut backend, &args, config()).unwrap();
        let launched = session.submit();
        if unknown {
            assert!(launched.is_err());
            assert_eq!(session.state(), ExecutionState::Quarantined);
        } else {
            launched.unwrap();
            assert_eq!(session.poll().unwrap(), Progress::Pending);
        }
        assert!(session.submit().is_err());
        assert_eq!(counts.launches.get(), 1);
        assert_eq!(counts.reads.get(), 0);
        drop(session);
        assert!(backend.poisoned());
        assert_eq!(counts.drops.get(), 0);
        assert!(matches!(
            plan.prepare(&mut backend, &args, config()),
            Err(Error::Poisoned)
        ));
    }
}
#[test]
fn cancellation_and_failed_quiescence_only_release_after_real_terminal_observation() {
    let plan = plan();
    let args = [Value::U32(3)];
    for failed in [false, true] {
        let mut backend = FaultBackend::default();
        let counts = backend.counts.clone();
        let mut session = plan.prepare(&mut backend, &args, config()).unwrap();
        session.submit().unwrap();
        if failed {
            counts.observation.set(FenceObservation::FailedQuiescent);
        } else {
            session.cancel();
            assert_eq!(session.poll().unwrap(), Progress::Pending);
            counts.observation.set(FenceObservation::Complete);
        }
        assert!(session.poll().is_err());
        assert_eq!(counts.reads.get(), 0);
        assert!(session.checkpoint().is_none());
        assert_eq!(counts.drops.get(), 0);
        drop(session.reclaim().unwrap());
        assert_eq!(counts.drops.get(), 1);
        assert!(!backend.poisoned());
    }
}
#[test]
fn input_and_private_admission_fail_before_backend_and_unsent_attempt_has_no_receipt() {
    let plan = plan();
    let mut backend = FaultBackend::default();
    let counts = backend.counts.clone();
    for args in [
        vec![],
        vec![Value::I32(3)],
        vec![Value::U32(3), Value::U32(4)],
    ] {
        assert!(matches!(
            plan.prepare(&mut backend, &args, config()),
            Err(Error::Input)
        ));
    }
    let args = [Value::U32(3)];
    let mut cfg = config();
    cfg.capacity = 3;
    assert!(plan.prepare(&mut backend, &args, cfg).is_err());
    assert_eq!(counts.prepare.get(), 0);
    counts.submission.set(Submission::NotSubmitted);
    let mut session = plan.prepare(&mut backend, &args, config()).unwrap();
    assert!(session.submit().is_err());
    // The core host budget counts attempts, including confirmed unsent work.
    assert_eq!(session.submitted(), 1);
    assert_eq!(session.completed(), 0);
    assert!(session.checkpoint().is_none());
    assert!(session.poll().is_err());
    drop(session.reclaim().unwrap());
    assert_eq!(counts.drops.get(), 1);
}
struct Tamper {
    inner: CpuModel,
}
struct TamperWork {
    inner: <CpuModel as Backend>::Work,
    reads: Cell<usize>,
}
impl Backend for Tamper {
    type Work = TamperWork;
    fn label() -> provenance::FenceBackend {
        CpuModel::label()
    }
    fn poisoned(&self) -> bool {
        self.inner.poisoned()
    }
    fn poison(&mut self) {
        self.inner.poison();
    }
    fn prepare(
        &mut self,
        a: &compiled::Artifact,
        b: [usize; 12],
        i: &[u64],
        c: [u64; 6],
    ) -> Result<Self::Work, String> {
        Ok(TamperWork {
            inner: self.inner.prepare(a, b, i, c)?,
            reads: Cell::new(0),
        })
    }
    fn ranges(work: &Self::Work) -> [BufferRange; 12] {
        CpuModel::ranges(&work.inner)
    }
    fn submit(work: &mut Self::Work) -> Submission {
        CpuModel::submit(&mut work.inner)
    }
    fn fence(work: &mut Self::Work) -> FenceObservation {
        CpuModel::fence(&mut work.inner)
    }
    fn read(work: &Self::Work) -> Result<[Vec<u64>; 12], String> {
        let mut words = CpuModel::read(&work.inner)?;
        let reads = work.reads.get();
        work.reads.set(reads + 1);
        if reads > 0 {
            words[11][1] ^= 1;
        }
        Ok(words)
    }
}
#[test]
fn actual_completed_continuation_generation_is_checked_again_before_resubmission() {
    let plan = plan();
    let args = [Value::U32(3)];
    let mut backend = Tamper {
        inner: CpuModel::default(),
    };
    let mut session = plan.prepare(&mut backend, &args, config()).unwrap();
    session.submit().unwrap();
    assert!(matches!(
        session.poll().unwrap(),
        Progress::Continuation { .. }
    ));
    assert!(session.checkpoint().is_some());
    assert!(matches!(session.submit(), Err(Error::Provenance(_))));
    assert_eq!(session.submitted(), 1);
    assert_eq!(session.completed(), 1);
    assert!(matches!(session.submit(), Err(Error::Closed)));
    drop(session.reclaim().unwrap());
}
