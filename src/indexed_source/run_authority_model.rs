//! Standalone preparatory model, included only by its owned test harness.
//! No production registration, source verification, backend entry or buffer ABI.
//! Supplied identities/costs and completion events are model inputs, not proofs.
#![forbid(unsafe_code)]

use std::{
    cell::RefCell,
    rc::{Rc, Weak},
    sync::atomic::{AtomicU64, Ordering},
};

pub(super) const MAX_VISITS: usize = 65_536;
pub(super) const MAX_BYTES: usize = 160 * 1024;
pub(super) const MAX_WITNESSES: usize = 16;
const METADATA_BYTES: usize = 4096;
const HEADER: usize = 8;
const KEY_BYTES: usize = 32;
const STEP: usize = 4;
const MIN_RESERVATION: usize = 3 * STEP;
static RUN_SEQUENCE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Family {
    Gather,
    Scatter,
    Histogram,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Cost {
    pub visits: usize,
    pub bytes: usize,
}
#[derive(Clone, Copy, Debug)]
pub(super) struct Limits {
    pub visits: usize,
    pub bytes: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Refusal {
    Exhausted,
    Overflow,
    NoClaim,
    Ambiguous,
    WitnessLimit,
    DuplicateWitness,
    StaleRun,
    StaleWitness,
    StalePermit,
    Consumed,
    Busy,
    Terminal,
    Unfunded,
    EntryFailed,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Phase {
    Idle,
    Selected,
    Refused,
    Abandoned,
    Entered,
    Ready,
    Publishing,
    Published,
    FailedQuiescent,
    Quarantined,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Completion {
    Complete,
    FailedQuiescent,
    Unknown,
    Pending,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Snapshot {
    pub run_id: u64,
    pub reserved_visits: usize,
    pub actual_visits: usize,
    pub reserved_bytes: usize,
    pub visits_limit: usize,
    pub bytes_limit: usize,
    pub attempts: u64,
    pub entries: u64,
    pub publications: u64,
    pub phase: Phase,
    pub closed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Authority {
    run_id: u64,
    attempt: u64,
    witness: usize,
    family: Family,
    key: [u8; KEY_BYTES],
}
#[derive(Clone, Copy, Debug)]
struct Record {
    family: Family,
    key: [u8; KEY_BYTES],
    required: Cost,
    active: bool,
}
#[derive(Clone, Copy, Debug)]
struct Reservation {
    authority: Authority,
    required: Cost,
    spent: usize,
}
#[derive(Debug)]
struct Ledger {
    snapshot: Snapshot,
    witnesses: [Option<Record>; MAX_WITNESSES],
    witness_count: usize,
    current: Option<Reservation>,
}
const _: () = assert!(size_of::<Ledger>() <= METADATA_BYTES);

impl Ledger {
    fn add(&mut self, visits: usize, bytes: usize, actual: bool) -> Result<(), Refusal> {
        let totals = (|| {
            let v = self
                .snapshot
                .reserved_visits
                .checked_add(visits)
                .ok_or(Refusal::Overflow)?;
            let b = self
                .snapshot
                .reserved_bytes
                .checked_add(bytes)
                .ok_or(Refusal::Overflow)?;
            let used = self
                .snapshot
                .actual_visits
                .checked_add(if actual { visits } else { 0 })
                .ok_or(Refusal::Overflow)?;
            if v > self.snapshot.visits_limit || b > self.snapshot.bytes_limit {
                return Err(Refusal::Exhausted);
            }
            Ok((v, b, used))
        })();
        match totals {
            Ok((v, b, used)) => {
                self.snapshot.reserved_visits = v;
                self.snapshot.reserved_bytes = b;
                self.snapshot.actual_visits = used;
                Ok(())
            }
            Err(e) => {
                self.snapshot.closed = true;
                Err(e)
            }
        }
    }
    fn witness(
        &self,
        run_id: u64,
        slot: usize,
        family: Family,
        key: [u8; KEY_BYTES],
    ) -> Result<Record, Refusal> {
        if run_id != self.snapshot.run_id {
            return Err(Refusal::StaleRun);
        }
        let r = self
            .witnesses
            .get(slot)
            .and_then(|r| *r)
            .ok_or(Refusal::StaleWitness)?;
        if !r.active || r.family != family || r.key != key {
            return Err(Refusal::StaleWitness);
        }
        Ok(r)
    }
    fn authority(&self, a: Authority, phase: Phase) -> Result<(), Refusal> {
        self.witness(a.run_id, a.witness, a.family, a.key)?;
        if self.current.is_none_or(|r| r.authority != a) {
            return Err(Refusal::StalePermit);
        }
        if self.snapshot.phase != phase {
            return Err(Refusal::Consumed);
        }
        Ok(())
    }
    fn debit(&mut self, visits: usize, floor: usize) -> Result<(), Refusal> {
        let result = (|| {
            let r = self.current.as_ref().ok_or(Refusal::StalePermit)?;
            let used = r.spent.checked_add(visits).ok_or(Refusal::Overflow)?;
            let required = used.checked_add(floor).ok_or(Refusal::Overflow)?;
            if required > r.required.visits {
                return Err(Refusal::Exhausted);
            }
            let actual = self
                .snapshot
                .actual_visits
                .checked_add(visits)
                .ok_or(Refusal::Overflow)?;
            if actual > self.snapshot.reserved_visits {
                return Err(Refusal::Exhausted);
            }
            Ok((used, actual))
        })();
        match result {
            Ok((used, actual)) => {
                self.current.as_mut().unwrap().spent = used;
                self.snapshot.actual_visits = actual;
                Ok(())
            }
            Err(e) => {
                self.snapshot.closed = true;
                Err(e)
            }
        }
    }
    fn open(&self) -> Result<(), Refusal> {
        if self.snapshot.closed {
            return Err(Refusal::Terminal);
        }
        if matches!(
            self.snapshot.phase,
            Phase::Selected | Phase::Entered | Phase::Ready | Phase::Publishing
        ) {
            return Err(Refusal::Busy);
        }
        Ok(())
    }
}

// Unique for this process, checked at exhaustion; no address, reset or reuse API.
fn fresh_run(sequence: &AtomicU64) -> Result<u64, Refusal> {
    sequence
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map_err(|_| Refusal::Overflow)
}

#[derive(Debug)]
pub(super) struct Run {
    ledger: Rc<RefCell<Ledger>>,
}
#[derive(Debug)]
pub(super) struct Witness {
    run_id: u64,
    slot: usize,
    family: Family,
    key: [u8; KEY_BYTES],
}
#[derive(Debug)]
pub(super) struct Permit {
    ledger: Rc<RefCell<Ledger>>,
    authority: Option<Authority>,
}
#[derive(Debug)]
pub(super) struct Entered {
    ledger: Rc<RefCell<Ledger>>,
    authority: Option<Authority>,
}
#[derive(Debug)]
pub(super) struct PublicationPermit {
    ledger: Rc<RefCell<Ledger>>,
    authority: Option<Authority>,
}
#[derive(Debug)]
pub(super) struct Quarantine {
    ledger: Option<Rc<RefCell<Ledger>>>,
}
#[derive(Debug)]
pub(super) struct Inspection {
    ledger: Weak<RefCell<Ledger>>,
}
#[derive(Debug)]
pub(super) enum Finish {
    Ready(PublicationPermit),
    Failed,
    Quarantined(Quarantine),
}

impl Run {
    pub fn new(limits: Limits) -> Result<Self, Refusal> {
        let run_id = fresh_run(&RUN_SEQUENCE)?;
        let limits = Limits {
            visits: limits.visits.min(MAX_VISITS),
            bytes: limits.bytes.min(MAX_BYTES),
        };
        if limits.visits < HEADER || limits.bytes < METADATA_BYTES {
            return Err(Refusal::Exhausted);
        }
        Ok(Self {
            ledger: Rc::new(RefCell::new(Ledger {
                snapshot: Snapshot {
                    run_id,
                    reserved_visits: HEADER,
                    actual_visits: HEADER,
                    reserved_bytes: METADATA_BYTES,
                    visits_limit: limits.visits,
                    bytes_limit: limits.bytes,
                    attempts: 0,
                    entries: 0,
                    publications: 0,
                    phase: Phase::Idle,
                    closed: false,
                },
                witnesses: [None; MAX_WITNESSES],
                witness_count: 0,
                current: None,
            })),
        })
    }
    pub fn snapshot(&self) -> Snapshot {
        self.ledger.borrow().snapshot
    }
    pub fn inspection(&self) -> Inspection {
        Inspection {
            ledger: Rc::downgrade(&self.ledger),
        }
    }
    /// A symbolic pre-work charge, not measurement of an opaque verifier.
    pub fn charge(&mut self, cost: Cost) -> Result<(), Refusal> {
        self.ledger.borrow_mut().add(cost.visits, cost.bytes, true)
    }
    pub fn register(
        &mut self,
        family: Family,
        key: [u8; KEY_BYTES],
        required: Cost,
    ) -> Result<Witness, Refusal> {
        let mut s = self.ledger.borrow_mut();
        s.add(HEADER + KEY_BYTES, 0, true)?;
        s.open()?;
        if required.visits < MIN_RESERVATION {
            return Err(Refusal::Unfunded);
        }
        if s.witnesses[..s.witness_count]
            .iter()
            .flatten()
            .any(|r| r.family == family && r.key == key)
        {
            return Err(Refusal::DuplicateWitness);
        }
        if s.witness_count == MAX_WITNESSES {
            return Err(Refusal::WitnessLimit);
        }
        let slot = s.witness_count;
        s.witnesses[slot] = Some(Record {
            family,
            key,
            required,
            active: true,
        });
        s.witness_count += 1;
        Ok(Witness {
            run_id: s.snapshot.run_id,
            slot,
            family,
            key,
        })
    }
    pub fn revoke(&mut self, witness: &Witness) -> Result<(), Refusal> {
        let mut s = self.ledger.borrow_mut();
        s.add(HEADER, 0, true)?;
        if s.snapshot.closed {
            return Err(Refusal::Terminal);
        }
        if matches!(
            s.snapshot.phase,
            Phase::Entered | Phase::Ready | Phase::Publishing
        ) {
            return Err(Refusal::Busy);
        }
        s.witness(witness.run_id, witness.slot, witness.family, witness.key)?;
        s.witnesses[witness.slot].as_mut().unwrap().active = false;
        Ok(())
    }
    pub fn select(&mut self, claims: &[&Witness]) -> Result<Permit, Refusal> {
        let mut s = self.ledger.borrow_mut();
        // Reject overlong claim lists after a fixed header; do not scan them.
        s.add(HEADER, 0, true)?;
        s.open()?;
        if claims.len() > 3 {
            return Err(Refusal::Ambiguous);
        }
        s.add(claims.len() * (HEADER + KEY_BYTES), 0, true)?;
        if claims.is_empty() {
            return Err(Refusal::NoClaim);
        }
        if claims.len() != 1 {
            return Err(Refusal::Ambiguous);
        }
        let w = claims[0];
        let record = s.witness(w.run_id, w.slot, w.family, w.key)?;
        let attempt = s.snapshot.attempts.checked_add(1).ok_or_else(|| {
            s.snapshot.closed = true;
            Refusal::Overflow
        })?;
        s.add(record.required.visits, record.required.bytes, false)?;
        let authority = Authority {
            run_id: s.snapshot.run_id,
            attempt,
            witness: w.slot,
            family: w.family,
            key: w.key,
        };
        s.current = Some(Reservation {
            authority,
            required: record.required,
            spent: 0,
        });
        s.snapshot.attempts = attempt;
        s.snapshot.phase = Phase::Selected;
        Ok(Permit {
            ledger: self.ledger.clone(),
            authority: Some(authority),
        })
    }
    pub fn prepare_work(&mut self, permit: &Permit, visits: usize) -> Result<(), Refusal> {
        let a = permit.authority.ok_or(Refusal::Consumed)?;
        let mut s = self.ledger.borrow_mut();
        s.authority(a, Phase::Selected)?;
        if s.snapshot.closed {
            return Err(Refusal::Terminal);
        }
        s.debit(visits, MIN_RESERVATION)
    }
    pub fn reject(&mut self, permit: &mut Permit) -> Result<(), Refusal> {
        let a = permit.authority.ok_or(Refusal::Consumed)?;
        let mut s = self.ledger.borrow_mut();
        s.authority(a, Phase::Selected)?;
        s.debit(STEP, 0)?;
        permit.authority.take();
        s.snapshot.phase = Phase::Refused;
        Ok(())
    }
    pub fn enter(
        &mut self,
        permit: &mut Permit,
        entry: impl FnOnce() -> Result<(), ()>,
    ) -> Result<Entered, Refusal> {
        let a = match permit.authority {
            Some(a) => a,
            None => {
                self.charge(Cost {
                    visits: STEP,
                    bytes: 0,
                })?;
                return Err(Refusal::Consumed);
            }
        };
        {
            let mut s = self.ledger.borrow_mut();
            if a.run_id != s.snapshot.run_id {
                s.add(STEP, 0, true)?;
                return Err(Refusal::StaleRun);
            }
            // Same-run failed entry consumes the permit and keeps its reservation.
            permit.authority.take();
            if let Err(e) = s.authority(a, Phase::Selected) {
                s.add(STEP, 0, true)?;
                if s.current.is_some_and(|r| r.authority == a)
                    && s.snapshot.phase == Phase::Selected
                {
                    s.snapshot.phase = Phase::Refused;
                }
                return Err(e);
            }
            if s.snapshot.closed {
                s.snapshot.phase = Phase::Refused;
                return Err(Refusal::Terminal);
            }
            s.debit(STEP, 2 * STEP)?;
            let entries = s.snapshot.entries.checked_add(1).ok_or_else(|| {
                s.snapshot.closed = true;
                Refusal::Overflow
            })?;
            s.snapshot.entries = entries;
            s.snapshot.phase = Phase::Entered;
        }
        // The local token quarantines the run if the simulator callback unwinds.
        let mut entered = Entered {
            ledger: self.ledger.clone(),
            authority: Some(a),
        };
        if entry().is_err() {
            let mut s = self.ledger.borrow_mut();
            s.snapshot.closed = true;
            s.snapshot.phase = Phase::FailedQuiescent;
            entered.authority.take();
            return Err(Refusal::EntryFailed);
        }
        Ok(entered)
    }
    pub fn work(&mut self, entered: &Entered, visits: usize) -> Result<(), Refusal> {
        let a = entered.authority.ok_or(Refusal::Consumed)?;
        let mut s = self.ledger.borrow_mut();
        s.authority(a, Phase::Entered)?;
        if s.snapshot.closed {
            return Err(Refusal::Terminal);
        }
        s.debit(visits, 2 * STEP)
    }
    pub fn finish(
        &mut self,
        entered: &mut Entered,
        completion: Completion,
    ) -> Result<Finish, Refusal> {
        if matches!(completion, Completion::Unknown | Completion::Pending) {
            entered.authority.take().ok_or(Refusal::Consumed)?;
            // Retention precedes accounting/authority validation, even for a wrong caller run.
            let hold = entered.ledger.clone();
            if let Ok(mut s) = hold.try_borrow_mut() {
                let _ = s.debit(STEP, STEP);
                s.snapshot.closed = true;
                s.snapshot.phase = Phase::Quarantined;
            }
            let q = Quarantine { ledger: Some(hold) };
            if entered.ledger.borrow().snapshot.run_id != self.snapshot().run_id {
                drop(q);
                return Err(Refusal::StaleRun);
            }
            return Ok(Finish::Quarantined(q));
        }
        let a = entered.authority.ok_or(Refusal::Consumed)?;
        let mut s = self.ledger.borrow_mut();
        s.authority(a, Phase::Entered)?;
        if s.snapshot.closed && completion != Completion::FailedQuiescent {
            return Err(Refusal::Terminal);
        }
        s.debit(STEP, STEP)?;
        entered.authority.take();
        if completion == Completion::FailedQuiescent {
            s.snapshot.closed = true;
            s.snapshot.phase = Phase::FailedQuiescent;
            return Ok(Finish::Failed);
        }
        s.snapshot.phase = Phase::Ready;
        Ok(Finish::Ready(PublicationPermit {
            ledger: self.ledger.clone(),
            authority: Some(a),
        }))
    }
    pub fn validate_work(
        &mut self,
        ready: &PublicationPermit,
        visits: usize,
    ) -> Result<(), Refusal> {
        let a = ready.authority.ok_or(Refusal::Consumed)?;
        let mut s = self.ledger.borrow_mut();
        s.authority(a, Phase::Ready)?;
        if s.snapshot.closed {
            return Err(Refusal::Terminal);
        }
        s.debit(visits, STEP)
    }
    pub fn publish(
        &mut self,
        ready: &mut PublicationPermit,
        publish: impl FnOnce(),
    ) -> Result<(), Refusal> {
        let a = match ready.authority {
            Some(a) => a,
            None => {
                self.charge(Cost {
                    visits: STEP,
                    bytes: 0,
                })?;
                return Err(Refusal::Consumed);
            }
        };
        let publications;
        {
            let mut s = self.ledger.borrow_mut();
            s.authority(a, Phase::Ready)?;
            if s.snapshot.closed {
                return Err(Refusal::Terminal);
            }
            s.debit(STEP, 0)?;
            publications = s.snapshot.publications.checked_add(1).ok_or_else(|| {
                s.snapshot.closed = true;
                Refusal::Overflow
            })?;
            ready.authority.take();
            s.snapshot.phase = Phase::Publishing;
        }
        let mut guard = PublicationPermit {
            ledger: self.ledger.clone(),
            authority: Some(a),
        };
        publish();
        let mut s = self.ledger.borrow_mut();
        s.snapshot.publications = publications;
        s.snapshot.phase = Phase::Published;
        guard.authority.take();
        Ok(())
    }
}

impl Inspection {
    pub fn snapshot(&self) -> Option<Snapshot> {
        self.ledger.upgrade().map(|s| s.borrow().snapshot)
    }
}
impl Quarantine {
    pub fn snapshot(&self) -> Snapshot {
        self.ledger.as_ref().unwrap().borrow().snapshot
    }
}

fn retain_unresolved(ledger: &Rc<RefCell<Ledger>>) {
    if let Ok(mut s) = ledger.try_borrow_mut() {
        s.snapshot.closed = true;
        s.snapshot.phase = Phase::Quarantined;
    }
    std::mem::forget(ledger.clone());
}
impl Drop for Permit {
    fn drop(&mut self) {
        if let Some(a) = self.authority.take()
            && let Ok(mut s) = self.ledger.try_borrow_mut()
            && s.current.is_some_and(|r| r.authority == a)
            && s.snapshot.phase == Phase::Selected
        {
            s.snapshot.phase = Phase::Abandoned;
        }
    }
}
impl Drop for Entered {
    fn drop(&mut self) {
        if self.authority.take().is_some() {
            retain_unresolved(&self.ledger);
        }
    }
}
impl Drop for PublicationPermit {
    fn drop(&mut self) {
        if self.authority.take().is_some() {
            let publishing = self
                .ledger
                .try_borrow()
                .map_or(true, |s| s.snapshot.phase == Phase::Publishing);
            if publishing {
                retain_unresolved(&self.ledger);
            } else if let Ok(mut s) = self.ledger.try_borrow_mut() {
                s.snapshot.closed = true;
                s.snapshot.phase = Phase::Abandoned;
            }
        }
    }
}
impl Drop for Quarantine {
    fn drop(&mut self) {
        if let Some(ledger) = self.ledger.take() {
            std::mem::forget(ledger);
        }
    }
}

#[cfg(test)]
mod internal_tests {
    use super::*;
    #[test]
    fn issue_m4_05_run_authority_model_sequence_overflow_is_sticky() {
        let sequence = AtomicU64::new(u64::MAX - 1);
        assert_eq!(fresh_run(&sequence), Ok(u64::MAX - 1));
        for _ in 0..3 {
            assert_eq!(fresh_run(&sequence), Err(Refusal::Overflow));
        }
        assert_eq!(sequence.load(Ordering::Relaxed), u64::MAX);
    }
    #[test]
    fn issue_m4_05_run_authority_model_attempt_overflow_no_entry() {
        let mut run = Run::new(Limits {
            visits: MAX_VISITS,
            bytes: MAX_BYTES,
        })
        .unwrap();
        let witness = run
            .register(
                Family::Gather,
                [1; KEY_BYTES],
                Cost {
                    visits: 32,
                    bytes: 8,
                },
            )
            .unwrap();
        run.ledger.borrow_mut().snapshot.attempts = u64::MAX;
        assert_eq!(run.select(&[&witness]).unwrap_err(), Refusal::Overflow);
        assert_eq!(run.snapshot().entries, 0);
        assert!(run.snapshot().closed);
    }

    #[test]
    fn issue_m4_05_run_authority_model_effect_counter_overflow_precedes_callbacks() {
        for publication in [false, true] {
            let mut run = Run::new(Limits {
                visits: MAX_VISITS,
                bytes: MAX_BYTES,
            })
            .unwrap();
            let w = run
                .register(
                    Family::Gather,
                    [1; KEY_BYTES],
                    Cost {
                        visits: 64,
                        bytes: 8,
                    },
                )
                .unwrap();
            let mut p = run.select(&[&w]).unwrap();
            if !publication {
                run.ledger.borrow_mut().snapshot.entries = u64::MAX;
                assert_eq!(
                    run.enter(&mut p, || panic!("overflow entered"))
                        .unwrap_err(),
                    Refusal::Overflow
                );
            } else {
                let mut entry = run.enter(&mut p, || Ok(())).unwrap();
                let Finish::Ready(mut ready) =
                    run.finish(&mut entry, Completion::Complete).unwrap()
                else {
                    panic!()
                };
                run.ledger.borrow_mut().snapshot.publications = u64::MAX;
                assert_eq!(
                    run.publish(&mut ready, || panic!("overflow published")),
                    Err(Refusal::Overflow)
                );
            }
            assert!(run.snapshot().closed);
        }
    }

    #[test]
    fn issue_m4_05_run_authority_model_old_private_ticket_cannot_replace_current_attempt() {
        let mut run = Run::new(Limits {
            visits: MAX_VISITS,
            bytes: MAX_BYTES,
        })
        .unwrap();
        let w = run
            .register(
                Family::Gather,
                [1; KEY_BYTES],
                Cost {
                    visits: 64,
                    bytes: 8,
                },
            )
            .unwrap();
        let mut p = run.select(&[&w]).unwrap();
        let old = p.authority.unwrap();
        let mut entry = run.enter(&mut p, || Ok(())).unwrap();
        let Finish::Ready(mut ready) = run.finish(&mut entry, Completion::Complete).unwrap() else {
            panic!()
        };
        run.publish(&mut ready, || {}).unwrap();
        let mut current = run.select(&[&w]).unwrap();
        // Internal fault injection deliberately bypasses the client's private-field boundary.
        let mut stale = Permit {
            ledger: run.ledger.clone(),
            authority: Some(old),
        };
        assert_eq!(
            run.enter(&mut stale, || panic!("stale attempt entered"))
                .unwrap_err(),
            Refusal::StalePermit
        );
        let _current = run.enter(&mut current, || Ok(())).unwrap();
        assert_eq!(run.snapshot().attempts, 2);
        assert_eq!(run.snapshot().entries, 2);
    }
}
