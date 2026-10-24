use super::*;
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Rad(pub(super) f64);
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Arm {
    Seven([Rad; 7]),
    Six([Rad; 6]),
}
impl Arm {
    fn values(&self) -> &[Rad] {
        match self {
            Self::Seven(v) => v,
            Self::Six(v) => v,
        }
    }
    fn same(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
            && self
                .values()
                .iter()
                .zip(other.values())
                .all(|(a, b)| a.0.to_bits() == b.0.to_bits())
    }
}
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Controls {
    Panda { arm: [Rad; 7], tendon: u8 },
    Ur5e { arm: [Rad; 6] },
}
impl Controls {
    pub(super) fn raw(&self) -> Vec<f64> {
        match self {
            Self::Panda { arm, tendon } => arm
                .iter()
                .map(|v| v.0)
                .chain([f64::from(*tendon)])
                .collect(),
            Self::Ur5e { arm } => arm.iter().map(|v| v.0).collect(),
        }
    }
    pub(super) fn from_raw(layout: Layout, values: &[f64]) -> Result<Self, Fault> {
        match layout {
            Layout::SevenArmTendon => {
                let [a, b, c, d, e, f, g, t] = values else {
                    return Err(Fault::Capacity);
                };
                if !t.is_finite() || !in_bounds(*t, [0.0, 255.0]) || t.fract() != 0.0 {
                    return Err(Fault::Bounds);
                }
                Ok(Self::Panda {
                    arm: [*a, *b, *c, *d, *e, *f, *g].map(Rad),
                    tendon: *t as u8,
                })
            }
            Layout::SixArm => {
                let [a, b, c, d, e, f] = values else {
                    return Err(Fault::Capacity);
                };
                Ok(Self::Ur5e {
                    arm: [*a, *b, *c, *d, *e, *f].map(Rad),
                })
            }
        }
    }
    pub(super) fn arm(&self) -> Arm {
        match self {
            Self::Panda { arm, .. } => Arm::Seven(*arm),
            Self::Ur5e { arm } => Arm::Six(*arm),
        }
    }
    pub(super) fn tendon(&self) -> Option<u8> {
        match self {
            Self::Panda { tendon, .. } => Some(*tendon),
            Self::Ur5e { .. } => None,
        }
    }
    fn same(&self, other: &Self) -> bool {
        self.arm().same(&other.arm()) && self.tendon() == other.tendon()
    }
}
#[derive(Clone, Debug, PartialEq)]
pub(super) struct State {
    pub(super) step: u64,
    pub(super) time: f64,
    pub(super) q: Vec<f64>,
    pub(super) dq: Vec<f64>,
    pub(super) forces: Vec<f64>,
    pub(super) contacts: Vec<(usize, usize)>,
    pub(super) applied: Controls,
    pub(super) attached: bool,
    // Owned, artificial hidden model state for clone/leak tests. A captured
    // native integration digest is only a fixture witness, not reconstruction.
    pub(super) latent: [u64; 8],
    pub(super) witness: Digest,
}
impl State {
    pub(super) fn home(binding: &Binding) -> Self {
        Self {
            step: 0,
            time: 0.0,
            q: binding
                .home_q
                .iter()
                .take(binding.joint_bounds.len())
                .copied()
                .collect(),
            dq: vec![0.0; binding.joint_bounds.len()],
            forces: vec![0.0; binding.force_bounds.len()],
            contacts: Vec::new(),
            applied: Controls::from_raw(binding.layout, &binding.home_controls)
                .expect("checked home controls"),
            attached: false,
            latent: [0; 8],
            witness: Digest([0; 32]),
        }
    }
    fn same(&self, other: &Self) -> bool {
        let bits = |a: &[f64], b: &[f64]| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        };
        self.step == other.step
            && self.time.to_bits() == other.time.to_bits()
            && bits(&self.q, &other.q)
            && bits(&self.dq, &other.dq)
            && bits(&self.forces, &other.forces)
            && self.contacts == other.contacts
            && self.applied.same(&other.applied)
            && self.attached == other.attached
            && self.latent == other.latent
            && self.witness == other.witness
    }
}
impl Binding {
    /// Gripper candidates step down from the source-derived home hold, so
    /// the policy never commands more than the source declares.
    pub(super) fn tendon_start(&self) -> Option<u8> {
        match self.layout {
            Layout::SevenArmTendon => self.home_controls.last().map(|t| *t as u8),
            Layout::SixArm => None,
        }
    }
    pub(super) fn candidates(&self) -> u32 {
        self.tendon_start().map_or(1, |start| u32::from(start) + 1)
    }
    fn dimensions(&self, state: &State) -> Result<(), Fault> {
        let controls = state.applied.raw();
        if state.q.len() != self.joint_bounds.len()
            || state.dq.len() != self.speeds.len()
            || state.forces.len() != self.force_bounds.len()
            || controls.len() != self.control_bounds.len()
            || state.contacts.len() > MAX_CONTACTS
        {
            return Err(Fault::Capacity);
        }
        Ok(())
    }
    pub(super) fn gate(&self, state: &State) -> Result<(), Fault> {
        self.dimensions(state)?;
        let controls = state.applied.raw();
        if !state.time.is_finite()
            || state.time < 0.0
            || state
                .q
                .iter()
                .chain(&state.dq)
                .chain(&state.forces)
                .chain(&controls)
                .any(|v| !v.is_finite())
        {
            return Err(Fault::Nonfinite);
        }
        if state
            .q
            .iter()
            .zip(&self.joint_bounds)
            .any(|(v, b)| !in_bounds(*v, *b))
            || state.dq.iter().zip(&self.speeds).any(|(v, s)| v.abs() > *s)
            || controls
                .iter()
                .zip(&self.control_bounds)
                .any(|(v, b)| !in_bounds(*v, *b))
            || state
                .forces
                .iter()
                .zip(&self.force_bounds)
                .any(|(v, b)| !in_bounds(*v, *b))
        {
            return Err(Fault::Bounds);
        }
        for (a, b) in &state.contacts {
            let a = self.geoms.get(*a).ok_or(Fault::Capacity)?;
            let b = self.geoms.get(*b).ok_or(Fault::Capacity)?;
            if (*a == GeomClass::Robot
                && matches!(
                    b,
                    GeomClass::Robot | GeomClass::Table | GeomClass::Environment
                ))
                || (*b == GeomClass::Robot
                    && matches!(
                        a,
                        GeomClass::Robot | GeomClass::Table | GeomClass::Environment
                    ))
            {
                return Err(Fault::Contact);
            }
        }
        Ok(())
    }
    fn nominal(&self, arm: &Arm) -> Result<(), Fault> {
        if !matches!(
            (self.layout, arm),
            (Layout::SevenArmTendon, Arm::Seven(_)) | (Layout::SixArm, Arm::Six(_))
        ) {
            return Err(Fault::Capacity);
        }
        if arm.values().iter().any(|v| !v.0.is_finite()) {
            return Err(Fault::Nonfinite);
        }
        if arm
            .values()
            .iter()
            .zip(&self.control_bounds)
            .any(|(v, b)| !in_bounds(v.0, *b))
        {
            return Err(Fault::Bounds);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RunId(pub(super) [u8; 16]);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Equality {
    pub(super) sequence: u64,
    pub(super) attached: bool,
}
#[derive(Clone, Debug)]
pub(super) struct Request {
    pub(super) run: RunId,
    pub(super) identity: Identity,
    pub(super) next_step: u64,
    // Supplied test identity; current PlannerV1/JTC transport has no such ID.
    pub(super) arm_sequence: u64,
    pub(super) arm: Arm,
    pub(super) equality: Equality,
}
impl Request {
    fn same(&self, other: &Self) -> bool {
        self.run == other.run
            && self.identity == other.identity
            && self.next_step == other.next_step
            && self.arm_sequence == other.arm_sequence
            && self.arm.same(&other.arm)
            && self.equality == other.equality
    }
}
#[derive(Debug)]
pub(super) struct Ticket {
    run: RunId,
    identity: Identity,
    pub(super) request: Request,
    pub(super) controls: Controls,
    pub(super) predicted: State,
    rank: u32,
}
#[derive(Clone, Debug)]
pub(super) enum Kind {
    Candidate {
        step: u64,
        evaluation: u64,
        rank: u32,
        controls: Controls,
        prospective: State,
        predicted: Box<State>,
        result: Result<(), Fault>,
    },
    Command {
        step: u64,
        request: Request,
        controls: Controls,
    },
    Snapshot(State),
    Halt {
        step: u64,
        reason: Fault,
        evaluations: u64,
    },
    Final {
        step: u64,
        witness: Digest,
        outcome: Result<(), Fault>,
    },
}
#[derive(Clone, Debug)]
pub(super) struct Record {
    pub(super) sequence: u64,
    pub(super) kind: Kind,
}
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Phase {
    Active,
    Prepared,
    Frozen,
    Complete,
    Halted(Fault),
}
#[derive(Debug)]
pub(super) struct Completion {
    pub(super) commands: u64,
    pub(super) snapshots: u64,
    pub(super) completed_steps: u64,
    pub(super) final_sequence: u64,
}
pub(super) struct Run<'a> {
    pub(super) binding: &'a Binding,
    pub(super) id: RunId,
    pub(super) live: State,
    pub(super) phase: Phase,
    required: u64,
    capacity: u32,
    pub(super) available: u32,
    pub(super) held: u32,
    pub(super) queue: VecDeque<Record>,
    pub(super) last_record: u64,
    last_ack: u64,
    pub(super) commands: u64,
    pub(super) snapshots: u64,
    last_request: Option<Request>,
    pub(super) candidate_evaluations: u64,
    pending: Option<(u64, u32, Controls, State, Request)>,
    frozen_fault: Option<Fault>,
}
impl<'a> Run<'a> {
    pub(super) fn new(
        binding: &'a Binding,
        id: RunId,
        initial: State,
        required: u64,
        capacity: u32,
    ) -> Result<Self, Fault> {
        if id.0 == [0; 16] {
            return Err(Fault::Identity);
        }
        if required == 0
            || required > MAX_STEPS
            || initial.step != 0
            || initial.time.to_bits() != 0.0f64.to_bits()
        {
            return Err(Fault::Order);
        }
        binding.gate(&initial)?;
        let bytes = u64::from(capacity)
            .checked_add(3)
            .and_then(|v| v.checked_mul(RECORD_BYTES))
            .ok_or(Fault::Overflow)?;
        if capacity == 0 || bytes > binding.memory_bytes {
            return Err(Fault::Capacity);
        }
        let mut queue = VecDeque::new();
        queue
            .try_reserve_exact(usize::try_from(capacity).map_err(|_| Fault::Capacity)?)
            .map_err(|_| Fault::Capacity)?;
        let mut run = Self {
            binding,
            id,
            live: initial.clone(),
            phase: Phase::Active,
            required,
            capacity,
            available: capacity,
            held: 0,
            queue,
            last_record: 0,
            last_ack: 0,
            commands: 0,
            snapshots: 1,
            candidate_evaluations: 0,
            last_request: None,
            pending: None,
            frozen_fault: None,
        };
        run.reserve(1)?;
        run.emit(Kind::Snapshot(initial))?;
        Ok(run)
    }
    fn reserve(&mut self, count: u32) -> Result<(), Fault> {
        self.last_record
            .checked_add(u64::from(count))
            .ok_or(Fault::Overflow)?;
        if self.held != 0 || count > self.available {
            return Err(Fault::Credits);
        }
        self.available = self.available.checked_sub(count).ok_or(Fault::Credits)?;
        self.held = count;
        Ok(())
    }
    fn emit(&mut self, kind: Kind) -> Result<(), Fault> {
        if self.held == 0
            || self.queue.len() >= usize::try_from(self.capacity).map_err(|_| Fault::Capacity)?
        {
            return Err(Fault::Credits);
        }
        let next = self.last_record.checked_add(1).ok_or(Fault::Overflow)?;
        self.queue.push_back(Record {
            sequence: next,
            kind,
        });
        self.last_record = next;
        self.held -= 1;
        Ok(())
    }
    fn release(&mut self) -> Result<(), Fault> {
        self.available = self
            .available
            .checked_add(self.held)
            .filter(|v| *v <= self.capacity)
            .ok_or(Fault::Credits)?;
        self.held = 0;
        Ok(())
    }
    fn halt<T>(&mut self, fault: Fault) -> Result<T, Fault> {
        self.pending = None;
        if self.held > 0 {
            self.emit(Kind::Halt {
                step: self.live.step,
                reason: fault,
                evaluations: self.candidate_evaluations,
            })?;
        }
        self.release()?;
        self.phase = Phase::Halted(fault);
        Err(fault)
    }
    pub(super) fn prepare(
        &mut self,
        request: Request,
        predictor: &mut Predictor,
    ) -> Result<Ticket, Fault> {
        if self.phase != Phase::Active {
            return self.halt(Fault::Lifecycle);
        }
        if request.run != self.id || request.identity != self.binding.identity {
            return self.halt(Fault::Identity);
        }
        let next = match self.live.step.checked_add(1) {
            Some(v) => v,
            None => return self.halt(Fault::Overflow),
        };
        if request.next_step != next || next > self.required {
            return self.halt(Fault::Order);
        }
        if let Some(last) = &self.last_request
            && (request.arm_sequence < last.arm_sequence
                || (request.arm_sequence == last.arm_sequence && !request.arm.same(&last.arm))
                || request.equality.sequence < last.equality.sequence
                || (request.equality.sequence == last.equality.sequence
                    && request.equality.attached != last.equality.attached))
        {
            return self.halt(Fault::Order);
        }
        if let Err(fault) = self.binding.nominal(&request.arm) {
            return self.halt(fault);
        }
        if let Err(fault) = self.binding.gate(&self.live) {
            return self.halt(fault);
        }
        if self.candidate_evaluations >= MAX_RUN_CANDIDATES {
            return self.halt(Fault::CandidateBudget);
        }
        // Worst-case candidate records plus command/state/halt. Unused slots
        // are released exactly; accepted commands hold three through commit.
        if let Err(fault) = self.reserve(self.binding.candidates() + 3) {
            return self.halt(fault);
        }
        let mut prospective = self.live.clone();
        prospective.attached = request.equality.attached;
        for rank in 0..self.binding.candidates() {
            if self.candidate_evaluations >= MAX_RUN_CANDIDATES {
                return self.halt(Fault::CandidateBudget);
            }
            self.candidate_evaluations = self
                .candidate_evaluations
                .checked_add(1)
                .ok_or(Fault::Overflow)?;
            let controls = match request.arm {
                Arm::Seven(arm) => Controls::Panda {
                    arm,
                    tendon: self
                        .binding
                        .tendon_start()
                        .and_then(|start| start.checked_sub(u8::try_from(rank).ok()?))
                        .ok_or(Fault::Overflow)?,
                },
                Arm::Six(arm) => Controls::Ur5e { arm },
            };
            let predicted = match predictor.predict(prospective.clone(), &controls) {
                Ok(v) => v,
                Err(fault) => return self.halt(fault),
            };
            if let Err(fault) = self.binding.dimensions(&predicted) {
                return self.halt(fault);
            }
            let verdict = if predicted.step != next
                || predicted.time.to_bits() != (self.live.time + DT).to_bits()
                || !predicted.applied.same(&controls)
                || predicted.attached != prospective.attached
            {
                Err(Fault::PredictionMismatch)
            } else {
                self.binding.gate(&predicted)
            };
            self.emit(Kind::Candidate {
                step: next,
                evaluation: self.candidate_evaluations,
                rank,
                controls: controls.clone(),
                prospective: prospective.clone(),
                predicted: Box::new(predicted.clone()),
                result: verdict,
            })?;
            match verdict {
                Ok(()) => {
                    let unused = self.held.checked_sub(3).ok_or(Fault::Credits)?;
                    self.held -= unused;
                    self.available = self.available.checked_add(unused).ok_or(Fault::Credits)?;
                    self.pending = Some((
                        next,
                        rank,
                        controls.clone(),
                        predicted.clone(),
                        request.clone(),
                    ));
                    self.phase = Phase::Prepared;
                    return Ok(Ticket {
                        run: self.id,
                        identity: self.binding.identity,
                        request,
                        controls,
                        predicted,
                        rank,
                    });
                }
                Err(Fault::Bounds | Fault::Contact) => (),
                Err(fault) => return self.halt(fault),
            }
        }
        self.halt(Fault::NoAdmissibleFiniteGrid)
    }
    pub(super) fn commit(&mut self, ticket: Ticket, actual: State) -> Result<(), Fault> {
        let Some((step, rank, controls, predicted, frozen_request)) = self.pending.take() else {
            return self.halt(Fault::Lifecycle);
        };
        if self.phase != Phase::Prepared
            || ticket.run != self.id
            || ticket.identity != self.binding.identity
            || !ticket.request.same(&frozen_request)
            || ticket.request.next_step != step
            || ticket.rank != rank
            || !controls.same(&ticket.controls)
            || !predicted.same(&ticket.predicted)
        {
            return self.halt(Fault::Identity);
        }
        if actual.step != step {
            return self.halt(Fault::Order);
        }
        if let Err(fault) = self.binding.dimensions(&actual) {
            return self.halt(fault);
        }
        // Commit is a supplied actual-model observation, never native execution.
        // Preserve the completed observation on mismatch; do not undo a step.
        self.commands = self.commands.checked_add(1).ok_or(Fault::Overflow)?;
        self.snapshots = self.snapshots.checked_add(1).ok_or(Fault::Overflow)?;
        self.emit(Kind::Command {
            step,
            request: ticket.request.clone(),
            controls: actual.applied.clone(),
        })?;
        self.emit(Kind::Snapshot(actual.clone()))?;
        self.live = actual;
        if !self.live.same(&predicted) {
            return self.halt(Fault::PredictionMismatch);
        }
        if let Err(fault) = self.binding.gate(&self.live) {
            return self.halt(fault);
        }
        self.release()?;
        self.last_request = Some(frozen_request);
        self.phase = Phase::Active;
        Ok(())
    }
    pub(super) fn acknowledge(&mut self, last: u64) -> Result<(), Fault> {
        if last <= self.last_ack || last > self.last_record {
            return Err(Fault::Order);
        }
        let count = last.checked_sub(self.last_ack).ok_or(Fault::Overflow)?;
        if count > u64::try_from(self.queue.len()).map_err(|_| Fault::Overflow)? {
            return Err(Fault::Order);
        }
        let count = usize::try_from(count).map_err(|_| Fault::Overflow)?;
        if self
            .queue
            .iter()
            .take(count)
            .zip(self.last_ack + 1..=last)
            .any(|(r, expected)| r.sequence != expected)
        {
            return Err(Fault::Order);
        }
        let available = self
            .available
            .checked_add(u32::try_from(count).map_err(|_| Fault::Overflow)?)
            .filter(|v| *v <= self.capacity)
            .ok_or(Fault::Credits)?;
        for _ in 0..count {
            self.queue.pop_front();
        }
        self.available = available;
        self.last_ack = last;
        Ok(())
    }
    pub(super) fn quiesce(&mut self, final_capture: State) -> Result<(), Fault> {
        let reason = match self.phase {
            Phase::Active => {
                if self.live.step == self.required {
                    None
                } else {
                    Some(Fault::Incomplete)
                }
            }
            Phase::Halted(fault) => Some(fault),
            _ => return self.halt(Fault::Lifecycle),
        };
        if !final_capture.same(&self.live) {
            return self.halt(Fault::PredictionMismatch);
        }
        if let Err(fault) = self.reserve(1) {
            return self.halt(fault);
        }
        self.emit(Kind::Final {
            step: self.live.step,
            witness: final_capture.witness,
            outcome: reason.map_or(Ok(()), Err),
        })?;
        self.frozen_fault = reason;
        self.phase = Phase::Frozen;
        Ok(())
    }
    pub(super) fn finish(&mut self) -> Result<Completion, Fault> {
        if self.phase != Phase::Frozen {
            return Err(Fault::Lifecycle);
        }
        if !self.queue.is_empty() || self.held != 0 || self.available != self.capacity {
            return Err(Fault::Credits);
        }
        if let Some(fault) = self.frozen_fault {
            return Err(fault);
        }
        if self.live.step != self.required
            || self.commands != self.required
            || self.snapshots != self.required.checked_add(1).ok_or(Fault::Overflow)?
        {
            return Err(Fault::Incomplete);
        }
        self.phase = Phase::Complete;
        Ok(Completion {
            commands: self.commands,
            snapshots: self.snapshots,
            completed_steps: self.live.step,
            final_sequence: self.last_ack,
        })
    }
    pub(super) fn credits_consistent(&self) -> bool {
        u64::from(self.available) + u64::from(self.held) + self.queue.len() as u64
            == u64::from(self.capacity)
    }
}

mod sealed {
    pub trait Sealed {}
}
trait TestPrediction: sealed::Sealed {
    fn predict(&mut self, state: State, controls: &Controls) -> Result<State, Fault>;
}
#[derive(Clone, Debug)]
pub(super) struct Recorded {
    pub(super) step: u64,
    pub(super) controls: Controls,
    pub(super) state: State,
}
pub(super) enum Behavior {
    Pass,
    RejectAbove(u8),
    Exhaust,
    Recorded(Vec<Recorded>),
    Nonfinite,
    WrongTransition,
}
pub(super) struct Predictor {
    pub(super) behavior: Behavior,
    pub(super) fuel: u32,
    pub(super) inputs: Vec<State>,
    pub(super) calls: u64,
}
impl Predictor {
    pub(super) fn new(behavior: Behavior, fuel: u32) -> Self {
        Self {
            behavior,
            fuel,
            inputs: Vec::new(),
            calls: 0,
        }
    }
}
impl sealed::Sealed for Predictor {}
impl TestPrediction for Predictor {
    fn predict(&mut self, mut state: State, controls: &Controls) -> Result<State, Fault> {
        if self.fuel == 0 {
            return Err(Fault::Fuel);
        }
        if self.inputs.first().is_some_and(|s| s.step != state.step) {
            self.inputs.clear();
        }
        if self.inputs.len() >= 256 {
            return Err(Fault::Capacity);
        }
        self.calls = self.calls.checked_add(1).ok_or(Fault::Overflow)?;
        self.fuel -= 1;
        self.inputs.push(state.clone());
        let next = state.step.checked_add(1).ok_or(Fault::Overflow)?;
        if let Behavior::Recorded(records) = &self.behavior {
            return records
                .iter()
                .find(|r| r.step == next && r.controls.same(controls))
                .map(|r| r.state.clone())
                .ok_or(Fault::Predictor);
        }
        state.step = next;
        state.time += DT;
        state.applied = controls.clone();
        state.latent[0] = state.latent[0].checked_add(1).ok_or(Fault::Overflow)?;
        match self.behavior {
            Behavior::RejectAbove(limit) if controls.tendon().is_some_and(|t| t > limit) => {
                *state.q.last_mut().ok_or(Fault::Capacity)? = 0.041;
                state.latent[1] = 999;
            }
            Behavior::Exhaust => {
                *state.q.last_mut().ok_or(Fault::Capacity)? = 100.0;
                state.latent[1] = 999;
            }
            Behavior::Nonfinite => {
                *state.q.first_mut().ok_or(Fault::Capacity)? = f64::NAN;
            }
            Behavior::WrongTransition => state.attached = !state.attached,
            _ => (),
        }
        Ok(state)
    }
}
