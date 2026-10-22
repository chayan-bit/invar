//! Ordinary-source ABI5 invocation. A backend owns the entire physical workset;
//! observed completion precedes all reads, publication and continuation epochs.
use super::{Net, Proof, compiled, provenance};
use crate::{
    eval::Value,
    front::{self, Span, Type},
};
use compiled::{
    BufferRange, DeviceConfig, ExecutionState, FenceError, FenceObservation, Submission, Target,
};
use sha2::{Digest, Sha256};

pub mod host_frame;
mod marshal;
pub use marshal::Refusal as EntryRefusal;
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod cuda_aot;
#[cfg(feature = "cuda")]
pub mod cuda_nvrtc;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub net: super::Limits,
    pub compiler: compiled::Limits,
    pub owned: compiled::OwnedLimits,
    pub private: compiled::SpeculationLimits,
    /// Logical physical buffer payload; admission only, not a source limit.
    pub buffer_bytes: usize,
    pub submissions: u64,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            net: super::Limits::default(),
            compiler: compiled::Limits::default(),
            owned: compiled::OwnedLimits::default(),
            private: compiled::SpeculationLimits::default(),
            buffer_bytes: 64 * 1024 * 1024,
            submissions: 1024,
        }
    }
}
impl Limits {
    /// Speculative child slots the device test suites certify against the
    /// CPU net oracle.
    pub const CERTIFIED_SLOTS: usize = 8;

    /// The certified device limits: the defaults with
    /// [`Self::CERTIFIED_SLOTS`] slots. `invar gnet-device` and `invar build
    /// --target cuda|metal --entry` use them, so the CLI runs what the tests
    /// cover and a built image matches the plan that runs it.
    pub fn certified() -> Self {
        Self {
            private: compiled::SpeculationLimits {
                slots: Self::CERTIFIED_SLOTS,
                ..compiled::SpeculationLimits::default()
            },
            ..Self::default()
        }
    }
}
#[derive(Debug)]
pub enum Error {
    Source(front::FrontError),
    Net(super::Error),
    Compiler(compiled::Error),
    Provenance(provenance::Error),
    Input,
    Entry(EntryRefusal),
    Limit,
    Backend(String),
    Fence(FenceError),
    Malformed,
    Closed,
    Poisoned,
    /// A compiled-ahead image was malformed, built for another program, or
    /// not loadable on this device or driver.
    Image(cuda_aot::Error),
}
impl From<FenceError> for Error {
    fn from(e: FenceError) -> Self {
        Self::Fence(e)
    }
}

/// Stable class of a forced-device refusal for receipts and corpus denominators.
/// Forced device execution never substitutes another engine: a refusal is the
/// whole outcome, and its receipt records `fallback: none`.
pub fn refusal_class(error: &Error) -> &'static str {
    let compiler = match error {
        Error::Compiler(c) | Error::Provenance(provenance::Error::Compiler(c)) => c,
        Error::Net(super::Error::Limit)
        | Error::Provenance(provenance::Error::Net(super::Error::Limit)) => {
            return "net_structural_budget";
        }
        Error::Net(_) | Error::Provenance(provenance::Error::Net(_)) => return "net_boundary",
        Error::Source(_) | Error::Provenance(provenance::Error::Source(_)) => return "source",
        Error::Limit | Error::Provenance(provenance::Error::Limit) => return "admission_budget",
        Error::Input => return "input",
        Error::Entry(e) => {
            return match e {
                EntryRefusal::Oversize => "oversize",
                EntryRefusal::Alias => "alias",
                EntryRefusal::WrongLayout => "wrong_layout",
                EntryRefusal::UnsupportedType => "unsupported_type",
                EntryRefusal::Borrowed => "borrowed",
            };
        }
        Error::Backend(_) => return "backend",
        Error::Image(_) => return "image",
        Error::Fence(_) | Error::Provenance(provenance::Error::Fence(_)) => return "fence",
        _ => return "host_protocol",
    };
    match compiler {
        compiled::Error::UnsupportedType => "unsupported_type",
        compiled::Error::UnsupportedRule(_) => "unsupported_rule",
        compiled::Error::UnsupportedCall => "unsupported_call",
        compiled::Error::DeviceSpeculationOwner => "owner_entry_result",
        compiled::Error::Limit => "compiler_structural_budget",
        compiled::Error::Entry => "entry",
        _ => "compiler_boundary",
    }
}

/// No externally supplied digest or descriptor can construct this source plan.
#[derive(Debug)]
pub struct Plan {
    artifact: provenance::ProvenancedArtifact,
    entry: String,
    source_digest: [u8; 32],
    source_bytes: usize,
    limits: Limits,
}
impl Plan {
    pub fn from_source(
        source: &[u8],
        entry: &str,
        target: Target,
        limits: Limits,
    ) -> Result<Self, Error> {
        if source.len() > 1024 * 1024
            || entry.is_empty()
            || entry.len() > 4096
            || limits.buffer_bytes == 0
            || limits.buffer_bytes > crate::array_limits::MAX_DEVICE_BYTES
            || limits.submissions == 0
            || limits.submissions > 1024
        {
            return Err(Error::Limit);
        }
        let artifact = provenance::compile_device_speculative(
            source,
            entry,
            target,
            provenance::Limits {
                net: limits.net,
                compiler: limits.compiler,
            },
            limits.owned,
            limits.private,
        )
        .map_err(Error::Provenance)?;
        Ok(Self {
            artifact,
            entry: entry.into(),
            source_digest: Sha256::digest(source).into(),
            source_bytes: source.len(),
            limits,
        })
    }
    /// Test-only: the same admission with a racing device fault hook.
    #[cfg(feature = "gnet-fault-injection")]
    pub fn from_source_with_fault(
        source: &[u8],
        entry: &str,
        target: Target,
        limits: Limits,
        fault: compiled::Fault,
    ) -> Result<Self, Error> {
        let mut plan = Self::from_source(source, entry, target, limits)?;
        plan.artifact = provenance::compile_device_speculative_with_fault(
            source,
            entry,
            target,
            provenance::Limits {
                net: limits.net,
                compiler: limits.compiler,
            },
            limits.owned,
            limits.private,
            fault,
        )
        .map_err(Error::Provenance)?;
        Ok(plan)
    }
    pub fn artifact(&self) -> &compiled::Artifact {
        self.artifact.artifact()
    }
    pub fn net(&self) -> &Net {
        self.artifact.checked_net().0
    }
    pub fn proof(&self) -> &Proof {
        self.artifact.checked_net().1
    }
    pub fn entry(&self) -> &str {
        &self.entry
    }
    pub fn compile_receipt(&self) -> &serde_json::Value {
        self.artifact.manifest()
    }
    pub fn source_digest(&self) -> [u8; 32] {
        self.source_digest
    }
    pub fn prepare<'a, B: Backend>(
        &'a self,
        backend: &'a mut B,
        args: &'a [Value],
        config: DeviceConfig,
    ) -> Result<Session<'a, B>, Error> {
        if backend.poisoned() {
            return Err(Error::Poisoned);
        }
        backend.admit(self).map_err(Error::Backend)?;
        let layout = self.artifact().device_layout().ok_or(Error::Malformed)?;
        let bytes = layout.buffer_bytes(config).map_err(Error::Compiler)?;
        let total = bytes
            .iter()
            .try_fold(0usize, |n, v| n.checked_add(*v))
            .ok_or(Error::Limit)?;
        // Readback, byte capture, decoded checkpoint and adapter storage may
        // coexist. Admit that peak before allocating the entry packet.
        let retained = total
            .checked_mul(4)
            .and_then(|n| n.checked_add(bytes[0]))
            .ok_or(Error::Limit)?;
        if retained > crate::array_limits::MAX_DEVICE_BYTES || total > self.limits.buffer_bytes {
            return Err(Error::Limit);
        }
        let packet = marshal::encode(self.artifact(), args)?;
        let input = packet.words;
        let work = backend
            .prepare(self.artifact(), bytes, &input, config.words())
            .map_err(Error::Backend)?;
        let ranges = B::ranges(&work);
        let workset = layout.workset(config, ranges).map_err(Error::Compiler)?;
        let execution = self
            .artifact
            .fenced_execution(workset, work, self.limits.submissions, B::label())
            .map_err(Error::Provenance)?;
        Ok(Session {
            plan: self,
            backend,
            execution: Some(execution),
            _input_lease: args,
            leases: packet.leases,
            input,
            rounds: 0,
            generation: 0,
            closed: false,
            checkpoint: None,
        })
    }
}
fn encode(types: &[Type], args: &[Value]) -> Result<Vec<u64>, Error> {
    if types.len() != args.len() {
        return Err(Error::Input);
    }
    types
        .iter()
        .zip(args)
        .map(|(ty, value)| match (ty, value) {
            (Type::I32, Value::I32(n)) => Ok(i64::from(*n) as u64),
            (Type::U32, Value::U32(n)) => Ok(u64::from(*n)),
            (Type::Bool, Value::Bool(b)) => Ok(u64::from(*b)),
            (Type::Unit, Value::Unit) => Ok(0),
            _ => Err(Error::Input),
        })
        .collect()
}

/// Implementations must issue identities from real retained allocations and
/// report actual completion. Metadata alone is never a fence observation.
pub trait Backend {
    type Work;
    fn label() -> provenance::FenceBackend;
    fn poisoned(&self) -> bool;
    fn poison(&mut self);
    /// Refuse a plan this backend has no executable for. Backends that run
    /// compiled-ahead images bind the matching image here; others admit all.
    fn admit(&mut self, _plan: &Plan) -> Result<(), String> {
        Ok(())
    }
    /// Completed live-device numeric preflight, never source execution proof.
    /// CPU group models have no physical floating-point admission.
    fn numeric_receipt(&self) -> Option<&serde_json::Value> {
        None
    }
    fn prepare(
        &mut self,
        artifact: &compiled::Artifact,
        bytes: [usize; 12],
        input: &[u64],
        config: [u64; 6],
    ) -> Result<Self::Work, String>;
    fn ranges(work: &Self::Work) -> [BufferRange; 12];
    fn submit(work: &mut Self::Work) -> Submission;
    fn fence(work: &mut Self::Work) -> FenceObservation;
    fn read(work: &Self::Work) -> Result<[Vec<u64>; 12], String>;
}

#[derive(Debug, PartialEq)]
pub struct Observation {
    pub status: compiled::Status,
    pub result: Option<Value>,
    pub span: Option<Span>,
    pub steps: u64,
    pub transitions: u64,
    pub max_frames: usize,
    pub events: Vec<[u64; 7]>,
    /// Completed normal cleanup in source order: function, place, length, frame.
    pub cleanup: Vec<[u64; 4]>,
    pub private_state: [u64; 14],
}
#[derive(Debug, PartialEq)]
pub enum Progress {
    Pending,
    Continuation { steps: u64, transitions: u64 },
    Complete(Box<Observation>),
}

pub struct Session<'a, B: Backend> {
    plan: &'a Plan,
    backend: &'a mut B,
    execution: Option<provenance::DeviceExecution<B::Work>>,
    _input_lease: &'a [Value],
    input: Vec<u64>,
    leases: Vec<marshal::Argument<'a>>,
    rounds: u64,
    generation: u64,
    closed: bool,
    checkpoint: Option<provenance::DeviceCheckpoint>,
}
impl<B: Backend> Session<'_, B> {
    pub fn workset(&self) -> &compiled::Workset {
        self.execution.as_ref().expect("live execution").workset()
    }
    pub fn submitted(&self) -> u64 {
        self.execution.as_ref().expect("live execution").submitted()
    }
    pub fn completed(&self) -> u64 {
        self.execution.as_ref().expect("live execution").completed()
    }
    pub fn numeric_receipt(&self) -> Option<&serde_json::Value> {
        self.plan
            .artifact()
            .entry_layout()
            .and_then(|_| self.backend.numeric_receipt())
    }
    pub fn checkpoint(&self) -> Option<&provenance::DeviceCheckpoint> {
        self.checkpoint.as_ref()
    }
    pub fn state(&self) -> ExecutionState {
        self.execution.as_ref().expect("live execution").state()
    }
    pub fn submit(&mut self) -> Result<(), Error> {
        if self.closed {
            return Err(Error::Closed);
        }
        if let Some(checkpoint) = &self.checkpoint {
            let mut read_error = None;
            let result = self
                .execution
                .as_ref()
                .expect("live execution")
                .verify_checkpoint(checkpoint, |work| match B::read(work) {
                    Ok(words) => std::array::from_fn(|i| {
                        words[i]
                            .iter()
                            .flat_map(|word| word.to_le_bytes())
                            .collect()
                    }),
                    Err(error) => {
                        read_error = Some(error);
                        std::array::from_fn(|_| Vec::new())
                    }
                });
            if let Some(error) = read_error {
                self.closed = true;
                return Err(Error::Backend(error));
            }
            if let Err(error) = result {
                self.closed = true;
                return Err(Error::Provenance(error));
            }
        }
        let result = self
            .execution
            .as_mut()
            .expect("live execution")
            .submit(B::submit);
        if matches!(result, Err(provenance::Error::Fence(FenceError::Unknown))) {
            self.backend.poison()
        }
        result.map_err(Error::Provenance)
    }
    pub fn cancel(&mut self) {
        self.closed = true;
        self.execution.as_mut().expect("live execution").cancel()
    }
    pub fn poll(&mut self) -> Result<Progress, Error> {
        let execution = self.execution.as_mut().expect("live execution");
        match execution.poll(B::fence) {
            Ok(()) => {}
            Err(provenance::Error::Fence(FenceError::Pending)) => return Ok(Progress::Pending),
            Err(e) => {
                if e == provenance::Error::Fence(FenceError::Unknown) {
                    self.backend.poison()
                }
                return Err(Error::Provenance(e));
            }
        }
        if self.closed {
            return Err(Error::Closed);
        }
        let mut read_error = None;
        let checkpoint = execution.capture(|work| match B::read(work) {
            Ok(words) => std::array::from_fn(|i| {
                words[i]
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect()
            }),
            Err(error) => {
                read_error = Some(error);
                std::array::from_fn(|_| Vec::new())
            }
        });
        if let Some(error) = read_error {
            self.closed = true;
            return Err(Error::Backend(error));
        }
        let checkpoint = match checkpoint {
            Ok(value) => value,
            Err(error) => {
                self.closed = true;
                return Err(Error::Provenance(error));
            }
        };
        let buffers = std::array::from_fn(|i| {
            checkpoint.buffers()[i]
                .chunks_exact(8)
                .map(|word| u64::from_le_bytes(word.try_into().expect("eight bytes")))
                .take(execution.workset().buffer_bytes()[i] / 8)
                .collect()
        });
        self.checkpoint = Some(checkpoint);
        let decoded = decode(
            self.plan,
            execution.workset(),
            &self.input,
            &buffers,
            self.rounds,
            self.generation,
        );
        match decoded {
            Ok((progress, rounds, generation)) => {
                self.rounds = rounds;
                self.generation = generation;
                if let Progress::Complete(observation) = &progress {
                    self.closed = true;
                    if observation.status == compiled::Status::Success {
                        marshal::publish(
                            &mut self.leases,
                            self.plan
                                .artifact()
                                .owned_layout()
                                .ok_or(Error::Malformed)?,
                            &buffers[4],
                        )?;
                    }
                }
                Ok(progress)
            }
            Err(e) => {
                self.closed = true;
                Err(e)
            }
        }
    }
    pub fn reclaim(mut self) -> Result<B::Work, Error> {
        let execution = self.execution.take().expect("live execution");
        if matches!(
            execution.state(),
            ExecutionState::InFlight | ExecutionState::CancelPending | ExecutionState::Quarantined
        ) {
            self.backend.poison()
        }
        execution.reclaim().map_err(Error::Provenance)
    }
}
impl<B: Backend> Drop for Session<'_, B> {
    fn drop(&mut self) {
        if self.execution.as_ref().is_some_and(|e| {
            matches!(
                e.state(),
                ExecutionState::InFlight
                    | ExecutionState::CancelPending
                    | ExecutionState::Quarantined
            )
        }) {
            self.backend.poison()
        }
    }
}

fn decode(
    plan: &Plan,
    workset: &compiled::Workset,
    input: &[u64],
    b: &[Vec<u64>; 12],
    previous_rounds: u64,
    previous_generation: u64,
) -> Result<(Progress, u64, u64), Error> {
    if b.iter()
        .zip(workset.buffer_bytes())
        .any(|(v, n)| v.len() != n / 8)
        || b[0].get(..input.len()) != Some(input)
        || b[3].as_slice() != workset.config().words()
    {
        return Err(Error::Malformed);
    }
    let config = workset.config();
    let task = plan
        .artifact()
        .continuation_layout()
        .ok_or(Error::Malformed)?
        .task_bytes()
        / 8
        - 18;
    let t = &b[6][task..];
    let h = &b[1];
    let s: [u64; 14] = b[11].as_slice().try_into().map_err(|_| Error::Malformed)?;
    // Owner counters: total allocations (IDs never recycled), live elements and
    // completed cleanup events, each within the compiled storage layout.
    let owned = plan.artifact().owned_layout().ok_or(Error::Malformed)?;
    if t[0] != 1
        || t[1] > 1
        || t[2] > 64
        || t[3] > 64
        || t[2] > t[3]
        || t[8] > owned.allocations() as u64
        || t[9] > owned.elements() as u64
        || t[10] > owned.cleanup_events() as u64
        || t[4] > config.fuel
        || t[5] > config.transitions
        || t[6] > config.trace_events as u64
        || t[11] != config.fuel
        || t[12] != config.transitions
        || t[13] != config.trace_events as u64
        || s[0] != config.capacity as u64
        || s[1] < previous_generation
        || s[3] > s[2]
        || s[11] <= previous_rounds
        || s[11]
            > previous_rounds
                .checked_add(config.rounds)
                .ok_or(Error::Malformed)?
        || s[13] > config.capacity as u64
    {
        return Err(Error::Malformed);
    }
    if t[1] == 0 {
        if s[12] != 14 {
            return Err(Error::Malformed);
        }
        return Ok((
            Progress::Continuation {
                steps: t[4],
                transitions: t[5],
            },
            s[11],
            s[1],
        ));
    }
    if s[12] != 0
        || h[0] != t[14]
        || h[1] != t[15]
        || h[2] != t[16]
        || h[3] != t[17]
        || h[2] > h[3]
        || h[3] > plan.source_bytes as u64
        || h[4] != t[4]
        || h[5] != t[5]
        || h[6] != t[3]
        || h[7] != t[6]
        || h[8] != t[10]
        || h[9] != t[8]
        || b[5].len() < t[10] as usize * 4
    {
        return Err(Error::Malformed);
    }
    let private = plan
        .artifact()
        .speculation_layout()
        .ok_or(Error::Malformed)?;
    if b[7]
        .chunks_exact(private.slot_bytes() / 8)
        .any(|slot| slot[slot.len() - 8] != 0)
    {
        return Err(Error::Malformed);
    }
    let status = match h[0] {
        0 => compiled::Status::Success,
        1 => compiled::Status::IntegerOverflow,
        2 => compiled::Status::DivisionByZero,
        3 => compiled::Status::FuelLimit,
        4 => compiled::Status::CallDepthLimit,
        5 => compiled::Status::TransitionLimit,
        6 => compiled::Status::TraceLimit,
        7 => compiled::Status::InvalidState,
        8 => compiled::Status::InvalidInput,
        9 => compiled::Status::IndexOutOfBounds,
        10 => compiled::Status::ArraySizeLimit,
        11 => compiled::Status::HeapLimit,
        12 => compiled::Status::AllocationLimit,
        13 => compiled::Status::CleanupLimit,
        _ => return Err(Error::Malformed),
    };
    let result = if status == compiled::Status::Success {
        Some(marshal::result(
            plan.artifact(),
            h[1],
            &b[4],
            t[8] as usize,
        )?)
    } else {
        None
    };
    let span = if status == compiled::Status::Success {
        None
    } else {
        Some(Span {
            start: usize::try_from(h[2]).map_err(|_| Error::Malformed)?,
            end: usize::try_from(h[3]).map_err(|_| Error::Malformed)?,
        })
    };
    let events = b[2][..t[6] as usize * 7]
        .chunks_exact(7)
        .map(|event| event.try_into().expect("seven words"))
        .collect();
    Ok((
        Progress::Complete(Box::new(Observation {
            status,
            result,
            span,
            steps: t[4],
            transitions: t[5],
            max_frames: t[3] as usize,
            events,
            cleanup: b[5][..t[10] as usize * 4]
                .chunks_exact(4)
                .map(|event| event.try_into().expect("four words"))
                .collect(),
            private_state: s,
        })),
        s[11],
        s[1],
    ))
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
fn numeric_probe_receipt(report: &invar_device::f32_probe::Report) -> serde_json::Value {
    serde_json::json!({"policy":"strict-f32-v1","probe":report.summary(),"native_strict":report.admits_strict(),"flushes_subnormals":report.flushes_subnormals(),"implicit_contraction":report.contracts_implicitly(),"native_float_admission":false,"arithmetic":"integer-software-binary32","subnormal_fallback":"integer-software-binary32"})
}
