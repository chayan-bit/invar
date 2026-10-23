//! ABI12 stage 2c S1: source-issued plan, bounded initial image and sealed
//! backend session lifecycle for `cuda-owned-numeric-continuation-v1`.
//!
//! Only test injectors implement the sealed backend in this slice: there is
//! no compiler, loader, CUDA, NVRTC or receipt here, and nothing here is
//! device evidence. Frozen review decisions:
//! - D1: the first `Unknown` (submit or fence) poisons the backend exactly once;
//!   afterwards no poll, read or reclaim reaches the backend or the work.
//! - D2: a session borrows its plan; non-quiescent work and its backend are
//!   forgotten together on drop, so neither is freed under unknown use.
//! - D3: the backend trait is sealed; its kind is a sealed constant.
//! - D4: `ceil(T/q) + 1 <= max_submissions <= 1024` is required before entry.
//! - D5: every host byte the session can hold (initial image as the first
//!   snapshot, the candidate snapshot, backend host buffers, checker scratch,
//!   decode scratch and one decoded result) is bounded by 256 MiB before entry,
//!   together with a monotonic per-session work budget of 2^30 units.
//!
//! Publication is atomic: results are readable only after every root is
//! terminal and validated. Every terminal, failed, cancelled or quarantined
//! state refuses further backend calls.
#![allow(
    dead_code,
    reason = "ABI12 stage 2c S1: consumed by the S2 CPU group model backend; exercised by owned lifecycle tests now"
)]
use super::{
    compiled, coverage, oracle, owned_numeric_decode,
    owned_numeric_task::{self as task, PENDING, TaskShape, TaskWords, TokenKey},
    provenance,
};
use crate::{
    eval,
    front::{Span, Type},
    mir,
};
use compiled::{ExecutionState, FenceError, FenceObservation, FencedExecution, Submission};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::atomic::{AtomicU64, Ordering},
};

/// Tail-lane and red-zone sentinel, as in the stage-1 and ABI3 device models.
pub(crate) const SENTINEL: u64 = 0x5a5a_5a5a_5a5a_5a5a;
/// Fill for the candidate snapshot before each readback: a word the backend
/// does not write cannot pass as an unchanged one.
const UNREAD: u64 = 0xa5a5_a5a5_a5a5_a5a5;
pub(crate) const MAX_SUBMISSIONS: u64 = 1024;
pub(crate) const MAX_WORK_UNITS: u64 = 1 << 30;
const OUTPUT_WORDS: usize = 10;
const EVENT_WORDS: usize = 7;
const CONFIG_WORDS: usize = 5;
const COUNTERS: usize = 18;
const BUFFERS: usize = 7;
/// Conservative per-entry bound for the checker's `BTreeMap` of dead-owner
/// lengths, including node overhead (entries <= allocations).
const BTREE_ENTRY_BYTES: usize = 256;
/// Conservative per-entry bound for decoded events and result elements.
const DECODED_ENTRY_BYTES: usize = 256;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Error {
    Provenance(provenance::Error),
    Layout(compiled::Error),
    /// Plan facts disagree with the frozen ABI12 profile.
    Malformed,
    /// Root count, arity or argument types.
    Input,
    /// Config bounds or the D4 submission budget.
    Limit,
    /// The D5 host total exceeds 256 MiB or cannot be bounded.
    HostBytes,
    /// The work budget exceeds 2^30 units, or a charge exceeds it.
    Work,
    Allocation,
    Backend(String),
    /// The operation is refused in the current state; no backend call is made.
    State(State),
    NotSubmitted,
    Unknown,
    FailedQuiescent,
    Cancelled,
    Validation(Validation),
    Decode {
        root: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Validation {
    Config,
    RedZone { buffer: usize },
    Tail { lane: usize },
    Input { root: usize },
    Terminal { root: usize },
    PendingOutput { root: usize },
    Checker { root: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum State {
    /// Prepared or validated with pending roots: quiescent, may submit.
    Ready,
    InFlight,
    CancelPending,
    /// Every root terminal, validated and decodable: results readable.
    Published,
    /// A readback, validation or decode failure after a completed epoch.
    Failed,
    /// The backend reported failure at quiescence; no readback was made.
    FailedQuiescent,
    /// The backend refused the submission; nothing was launched.
    NotSubmitted,
    Cancelled,
    Quarantined,
    /// The backend and its work were returned to the caller; nothing remains.
    Reclaimed,
}
impl State {
    fn quiescent(self) -> bool {
        !matches!(
            self,
            State::InFlight | State::CancelPending | State::Quarantined
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Progress {
    Pending,
    /// A validated epoch with roots still pending.
    Advanced,
    Published,
    Cancelled,
}

/// Run configuration: the five kernel config words plus the fence budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Config {
    pub roots: usize,
    pub fuel: u64,
    pub transitions: u64,
    pub trace_events: usize,
    pub quantum: u64,
    pub max_submissions: u64,
}

/// Facts copied from the 2b read-only accessor (the 2a layout types are
/// private to `compiled`).
#[derive(Clone, Copy, Debug)]
struct Facts {
    input_words: usize,
    heap_words: usize,
    cleanup_words: usize,
    task_words: usize,
    red_zone_words: usize,
    max_roots: usize,
    max_fuel: u64,
    max_transitions: u64,
    max_trace_events: usize,
    max_quantum: u64,
    max_device_bytes: usize,
    max_host_bytes: usize,
}

mod sealed {
    pub trait Sealed {}
}

/// Who executes the emitted text. Sealed: no other kind can be forged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BackendKind {
    /// Reserved for the S2 CPU group model (never device evidence).
    CpuGroupModel,
    #[cfg(test)]
    TestInjector,
}

/// Sealed ABI12 seven-buffer backend; only this module can implement it.
pub(crate) trait Backend: sealed::Sealed {
    type Work;
    const KIND: BackendKind;
    /// Worst-case host bytes the backend allocates for a workset of
    /// `device_bytes`, or `None` if it cannot bound them. Not an instance call.
    fn host_bytes(device_bytes: usize) -> Option<usize>;
    fn prepare(
        &mut self,
        artifact: &compiled::Artifact,
        image: &Buffers,
    ) -> Result<Self::Work, String>;
    fn submit(&mut self, work: &mut Self::Work) -> Submission;
    fn poll(&mut self, work: &mut Self::Work) -> FenceObservation;
    /// Write every word of every buffer into the preallocated slots.
    fn read_into(&self, work: &Self::Work, slots: Slots<'_>) -> Result<(), String>;
    fn poison(&mut self);
}

/// The seven buffers in kernel order, each with its trailing red zone.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Buffers {
    pub input: Vec<i64>,
    pub output: Vec<i64>,
    pub trace: Vec<u64>,
    pub config: Vec<u64>,
    pub heap: Vec<i64>,
    pub cleanup: Vec<u64>,
    pub tasks: Vec<u8>,
}
/// Fixed-length views of a preallocated snapshot; no buffer can be resized.
pub(crate) struct Slots<'a> {
    pub input: &'a mut [i64],
    pub output: &'a mut [i64],
    pub trace: &'a mut [u64],
    pub config: &'a mut [u64],
    pub heap: &'a mut [i64],
    pub cleanup: &'a mut [u64],
    pub tasks: &'a mut [u8],
}
impl Buffers {
    fn slots(&mut self) -> Slots<'_> {
        Slots {
            input: &mut self.input,
            output: &mut self.output,
            trace: &mut self.trace,
            config: &mut self.config,
            heap: &mut self.heap,
            cleanup: &mut self.cleanup,
            tasks: &mut self.tasks,
        }
    }
}

fn reserve<T: Clone>(n: usize, fill: T) -> Result<Vec<T>, Error> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| Error::Allocation)?;
    v.resize(n, fill);
    Ok(v)
}

/// Exact per-run sizes, all computed with checked arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Sizes {
    lanes: usize,
    words: [usize; BUFFERS],
    task_bytes: usize,
    device_bytes: usize,
}

/// Monotonic work ledger: charges never refund and never exceed the
/// reservation made before entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Ledger {
    reserved: u64,
    used: u64,
}
impl Ledger {
    fn charge(&mut self, units: u64) -> Result<(), Error> {
        let next = self.used.checked_add(units).ok_or(Error::Work)?;
        if next > self.reserved {
            return Err(Error::Work);
        }
        self.used = next;
        Ok(())
    }
    pub(crate) fn reserved(&self) -> u64 {
        self.reserved
    }
    pub(crate) fn used(&self) -> u64 {
        self.used
    }
}

/// A source-issued ABI12 plan. The only constructor compiles the source
/// through provenance (2b precedence and receipt); no artifact, manifest or
/// receipt can be supplied from outside.
pub(crate) struct Plan {
    artifact: provenance::ProvenancedArtifact,
    program: mir::Program,
    sites: BTreeMap<TokenKey, (coverage::Rule, Span)>,
    state_functions: Vec<usize>,
    frame_values: usize,
    frame_places: usize,
    facts: Facts,
    nonce: u64,
}

/// Seal for the execution-receipt builder: only this module family (the
/// session and its sealed backends) can construct it.
pub(crate) struct ReceiptSeal(());

pub(crate) struct Refusal<B> {
    pub error: Error,
    pub backend: B,
}

impl Plan {
    pub(crate) fn from_source(
        source: &[u8],
        entry: &str,
        limits: provenance::Limits,
        storage: compiled::OwnedLimits,
    ) -> Result<Self, Error> {
        let artifact = provenance::compile_resumable_owned_numeric(
            source,
            entry,
            compiled::Target::Cuda,
            limits,
            storage,
        )
        .map_err(Error::Provenance)?;
        let a = artifact.artifact();
        if a.abi_version() != 12 || a.target() != compiled::Target::Cuda {
            return Err(Error::Malformed);
        }
        let f = a.owned_numeric_cuda_facts().map_err(Error::Layout)?;
        if f.output_words != OUTPUT_WORDS
            || f.event_words != EVENT_WORDS
            || f.config_words != CONFIG_WORDS
            || f.lane_block != 32
            || f.min_retained_snapshots != 2
        {
            return Err(Error::Malformed);
        }
        let facts = Facts {
            input_words: f.input_words,
            heap_words: f.heap_words,
            cleanup_words: f.cleanup_words,
            task_words: f.task_words,
            red_zone_words: f.red_zone_words,
            max_roots: f.max_roots,
            max_fuel: f.max_fuel,
            max_transitions: f.max_transitions,
            max_trace_events: f.max_trace_events,
            max_quantum: f.max_quantum,
            max_device_bytes: f.max_device_bytes,
            max_host_bytes: f.max_host_snapshot_bytes,
        };
        let (net, proof) = artifact.checked_net();
        let program = net
            .compiled_mir(proof)
            .map_err(|e| Error::Provenance(provenance::Error::Net(e)))?;
        // As the ABI11 host: one PC per instruction and terminator in MIR
        // order; frame = GnetFrame{v[V], p[P], pc, ret, id, activation}.
        let mut state_functions = Vec::new();
        state_functions
            .try_reserve_exact(a.states())
            .map_err(|_| Error::Allocation)?;
        let (mut frame_values, mut frame_places) = (1, 1);
        for (function_id, function) in program.functions.iter().enumerate() {
            let mut values = BTreeSet::new();
            for block in &function.blocks {
                for (id, _) in block
                    .params
                    .iter()
                    .chain(block.instructions.iter().filter_map(|i| i.output.as_ref()))
                {
                    values.insert(*id);
                }
                for _ in 0..=block.instructions.len() {
                    if state_functions.len() == a.states() {
                        return Err(Error::Malformed);
                    }
                    state_functions.push(function_id);
                }
            }
            frame_values = frame_values.max(values.len());
            frame_places = frame_places.max(function.locals.len());
        }
        let task_bytes = a
            .continuation_layout()
            .ok_or(Error::Malformed)?
            .task_bytes();
        let frame_len = (frame_values + frame_places + 4) * 8;
        if state_functions.len() != a.states()
            || frame_len * 64 != a.frame_bytes()
            || task_bytes != a.frame_bytes() + COUNTERS * 8
            || facts.task_words * 8 != task_bytes
        {
            return Err(Error::Malformed);
        }
        let mut sites = BTreeMap::new();
        for (token, rule, span) in net
            .agents
            .iter()
            .map(|x| (x.token, x.rule, x.span))
            .chain(net.control.iter().map(|x| (x.token, x.rule, x.span)))
        {
            let operator = program
                .functions
                .get(token.block.function)
                .and_then(|f| f.blocks.get(token.block.block))
                .and_then(|b| b.instructions.get(token.position))
                .and_then(|i| match i.kind {
                    mir::Operation::Binary { operator, .. }
                    | mir::Operation::Unary { operator, .. } => Some(operator),
                    _ => None,
                });
            sites.insert(
                (token.block.function, token.block.block, token.position),
                (rule, operator.unwrap_or(span)),
            );
        }
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let nonce = NEXT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| Error::Limit)?;
        Ok(Self {
            artifact,
            program,
            sites,
            state_functions,
            frame_values,
            frame_places,
            facts,
            nonce,
        })
    }

    pub(crate) fn nonce(&self) -> u64 {
        self.nonce
    }
    pub(crate) fn artifact(&self) -> &provenance::ProvenancedArtifact {
        &self.artifact
    }

    fn shape(&self) -> TaskShape<'_> {
        let a = self.artifact.artifact();
        TaskShape {
            frame_bytes: a.frame_bytes(),
            task_bytes: self.facts.task_words * 8,
            states: a.states(),
            frame_values: self.frame_values,
            frame_places: self.frame_places,
            state_functions: &self.state_functions,
            sites: &self.sites,
            program: &self.program,
            storage: a.owned_layout().expect("ABI12 owned layout"),
        }
    }
    fn storage(&self) -> &compiled::OwnedLayout {
        self.artifact
            .artifact()
            .owned_layout()
            .expect("ABI12 owned layout")
    }
    fn oracle_limits(&self, c: Config) -> oracle::Limits {
        oracle::Limits {
            fuel: c.fuel,
            transitions: c.transitions,
            trace_events: c.trace_events,
            array_elements: self.storage().elements(),
        }
    }

    /// Config bounds (as the 2a workset check, via the 2b facts) and D4.
    fn check_config(&self, c: Config) -> Result<u64, Error> {
        let f = self.facts;
        if c.roots == 0
            || c.roots > f.max_roots
            || c.fuel == 0
            || c.fuel > f.max_fuel
            || c.transitions > f.max_transitions
            || c.trace_events > f.max_trace_events
            || c.quantum == 0
            || c.quantum > f.max_quantum
        {
            return Err(Error::Limit);
        }
        let epochs = c
            .transitions
            .div_ceil(c.quantum)
            .checked_add(1)
            .ok_or(Error::Limit)?;
        if c.max_submissions == 0
            || c.max_submissions > MAX_SUBMISSIONS
            || epochs > c.max_submissions
        {
            return Err(Error::Limit);
        }
        Ok(epochs)
    }

    fn sizes(&self, c: Config) -> Result<Sizes, Error> {
        let f = self.facts;
        let lanes = c
            .roots
            .checked_add(31)
            .map(|n| n / 32 * 32)
            .ok_or(Error::Limit)?;
        let trace = c
            .trace_events
            .checked_mul(EVENT_WORDS)
            .ok_or(Error::Limit)?;
        let red = f.red_zone_words;
        let per_lane = |words: usize| {
            words
                .checked_mul(lanes)
                .and_then(|n| n.checked_add(red))
                .ok_or(Error::Limit)
        };
        let words = [
            per_lane(f.input_words)?,
            per_lane(OUTPUT_WORDS)?,
            per_lane(trace)?,
            CONFIG_WORDS + red,
            per_lane(f.heap_words)?,
            per_lane(f.cleanup_words)?,
            per_lane(f.task_words)?,
        ];
        let device_bytes = words
            .iter()
            .try_fold(0usize, |n, w| {
                w.checked_mul(8).and_then(|b| n.checked_add(b))
            })
            .ok_or(Error::Limit)?;
        if device_bytes > f.max_device_bytes {
            return Err(Error::Limit);
        }
        Ok(Sizes {
            lanes,
            words,
            task_bytes: f.task_words * 8,
            device_bytes,
        })
    }

    /// Words of one root's slices (input, output, trace, heap, cleanup, task).
    fn root_words(&self, c: Config) -> Option<usize> {
        let f = self.facts;
        f.input_words
            .checked_add(OUTPUT_WORDS)?
            .checked_add(c.trace_events.checked_mul(EVENT_WORDS)?)?
            .checked_add(f.heap_words)?
            .checked_add(f.cleanup_words)?
            .checked_add(f.task_words)
    }

    /// Peak checker scratch for one `validate_device_epoch` call, from its
    /// reserve sites (`owned_numeric_task.rs`: active/stack 64 each, history
    /// <= transitions + 1, rows <= trace events, owners/live <= allocations,
    /// dead-length map <= allocations), doubled for allocator overhead.
    fn checker_bytes(&self, c: Config) -> Option<usize> {
        use std::mem::size_of;
        let s = self.storage();
        let history = usize::try_from(c.transitions.checked_add(1)?).ok()?;
        let bytes = 64usize
            .checked_mul(size_of::<(usize, u64, u64, usize)>())?
            .checked_add(64usize.checked_mul(size_of::<(u64, u64, TokenKey, coverage::Rule)>())?)?
            .checked_add(history.checked_mul(size_of::<Option<task::History>>())?)?
            .checked_add(c.trace_events.checked_mul(size_of::<task::Row>())?)?
            .checked_add(s.allocations().checked_mul(
                size_of::<Option<i64>>() + size_of::<(usize, usize)>() + BTREE_ENTRY_BYTES,
            )?)?;
        bytes.checked_mul(2)
    }

    /// Decode scratch (one root's trace, heap and cleanup copies) plus one
    /// decoded result (events and result elements).
    fn decode_bytes(&self, c: Config) -> Option<usize> {
        let f = self.facts;
        let scratch = c
            .trace_events
            .checked_mul(EVENT_WORDS)?
            .checked_add(f.heap_words)?
            .checked_add(f.cleanup_words)?
            .checked_mul(8)?;
        let decoded = c
            .trace_events
            .checked_add(self.storage().elements())?
            .checked_add(64)?
            .checked_mul(DECODED_ENTRY_BYTES)?;
        scratch.checked_add(decoded)
    }

    /// D5: total host bytes; `None` if any term cannot be bounded.
    fn host_bytes<B: Backend>(&self, c: Config, sizes: Sizes) -> Option<usize> {
        sizes
            .device_bytes
            .checked_mul(2)?
            .checked_add(B::host_bytes(sizes.device_bytes)?)?
            .checked_add(self.checker_bytes(c)?)?
            .checked_add(self.decode_bytes(c)?)
    }

    /// Work units: per epoch, the candidate fill and the full comparison of
    /// every device word, plus each root's checker bound; then one decode
    /// pass to publish and one to read every result.
    fn work_units(&self, c: Config, sizes: Sizes, epochs: u64) -> Option<u64> {
        let device_words = u64::try_from(sizes.device_bytes / 8).ok()?;
        let roots = u64::try_from(c.roots).ok()?;
        let per_epoch = device_words
            .checked_mul(2)?
            .checked_add(roots.checked_mul(self.work_per_root(c)?)?)?;
        epochs
            .checked_mul(per_epoch)?
            .checked_add(roots.checked_mul(self.decode_units(c)?)?.checked_mul(2)?)
    }

    /// Encoded input words of every root, after count, arity and type checks.
    fn encode(&self, c: Config, roots: &[Vec<eval::Value>]) -> Result<(), Error> {
        if roots.len() != c.roots {
            return Err(Error::Input);
        }
        let types = self.artifact.artifact().params();
        for args in roots {
            if args.len() != types.len() {
                return Err(Error::Input);
            }
            for (ty, arg) in types.iter().zip(args) {
                word(ty, arg)?;
            }
        }
        Ok(())
    }

    /// C3 initial image: live roots equal the ABI11 creation state
    /// (input = arguments, output = 10 x `i64::MIN`, everything else zero);
    /// tail lanes have zero input and task and `SENTINEL` elsewhere; every red
    /// zone and the config red zone hold `SENTINEL`.
    fn image(&self, c: Config, s: Sizes, roots: &[Vec<eval::Value>]) -> Result<Buffers, Error> {
        let f = self.facts;
        let red = f.red_zone_words;
        let sentinel = SENTINEL as i64;
        let mut b = Buffers {
            input: reserve(s.words[0], 0i64)?,
            output: reserve(s.words[1], sentinel)?,
            trace: reserve(s.words[2], SENTINEL)?,
            config: reserve(s.words[3], SENTINEL)?,
            heap: reserve(s.words[4], sentinel)?,
            cleanup: reserve(s.words[5], SENTINEL)?,
            tasks: reserve(s.words[6] * 8, 0u8)?,
        };
        let types = self.artifact.artifact().params();
        let (iw, tw, hw, cw) = (
            f.input_words,
            c.trace_events * EVENT_WORDS,
            f.heap_words,
            f.cleanup_words,
        );
        for (r, args) in roots.iter().enumerate() {
            for (i, (ty, arg)) in types.iter().zip(args).enumerate() {
                b.input[r * iw + i] = word(ty, arg)?;
            }
            b.output[r * OUTPUT_WORDS..(r + 1) * OUTPUT_WORDS].copy_from_slice(&PENDING);
            b.trace[r * tw..(r + 1) * tw].fill(0);
            b.heap[r * hw..(r + 1) * hw].fill(0);
            b.cleanup[r * cw..(r + 1) * cw].fill(0);
        }
        // Input red zone and task red zone hold the sentinel; tail input and
        // tail tasks stay zero.
        let input_red = s.words[0] - red;
        b.input[input_red..].fill(sentinel);
        let task_red = s.lanes * s.task_bytes;
        b.tasks[task_red..].fill(0x5a);
        for (slot, value) in b.config.iter_mut().zip([
            c.roots as u64,
            c.fuel,
            c.transitions,
            c.trace_events as u64,
            c.quantum,
        ]) {
            *slot = value;
        }
        Ok(b)
    }

    /// Admit a run: every check and reservation precedes the first backend
    /// call; refusals return the untouched backend.
    pub(crate) fn prepare<B: Backend>(
        &self,
        mut backend: B,
        roots: &[Vec<eval::Value>],
        config: Config,
    ) -> Result<Session<'_, B>, Refusal<B>> {
        let admitted = (|| {
            let epochs = self.check_config(config)?;
            self.encode(config, roots)?;
            let sizes = self.sizes(config)?;
            let host = self
                .host_bytes::<B>(config, sizes)
                .filter(|n| *n <= self.facts.max_host_bytes)
                .ok_or(Error::HostBytes)?;
            let work = self
                .work_units(config, sizes, epochs)
                .filter(|n| *n <= MAX_WORK_UNITS)
                .ok_or(Error::Work)?;
            // N2: the 2a checker's dead-owner `BTreeMap` allocates
            // infallibly (abort on OOM). Its bound is in the D5 total; this
            // one-time fallible preflight of that bound fails early instead.
            let mut preflight = Vec::<u8>::new();
            preflight
                .try_reserve_exact(self.checker_bytes(config).ok_or(Error::HostBytes)?)
                .map_err(|_| Error::Allocation)?;
            drop(preflight);
            let previous = self.image(config, sizes, roots)?;
            let candidate = previous.clone_fallible()?;
            let f = self.facts;
            let decode = (
                reserve(config.trace_events * EVENT_WORDS, 0u64)?,
                reserve(f.heap_words, 0i64)?,
                reserve(f.cleanup_words, 0u64)?,
            );
            Ok((epochs, sizes, host, work, previous, candidate, decode))
        })();
        let (epochs, sizes, host, work, previous, candidate, decode) = match admitted {
            Ok(v) => v,
            Err(error) => return Err(Refusal { error, backend }),
        };
        let resources = match backend.prepare(self.artifact.artifact(), &previous) {
            Ok(w) => w,
            Err(e) => {
                return Err(Refusal {
                    error: Error::Backend(e),
                    backend,
                });
            }
        };
        let Ok(fence) = FencedExecution::new(sizes, resources, config.max_submissions) else {
            return Err(Refusal {
                error: Error::Limit,
                backend,
            });
        };
        let lane = LaneLayout {
            input: self.facts.input_words,
            trace: config.trace_events * EVENT_WORDS,
            heap: self.facts.heap_words,
            cleanup: self.facts.cleanup_words,
            task: self.facts.task_words * 8,
            red: self.facts.red_zone_words,
            frame_bytes: self.artifact.artifact().frame_bytes(),
        };
        Ok(Session {
            plan: self,
            lane,
            backend: Some(backend),
            fence: Some(fence),
            state: State::Ready,
            poisoned: false,
            config,
            epochs,
            host_bytes: host,
            ledger: Ledger {
                reserved: work,
                used: 0,
            },
            previous,
            candidate,
            decode,
        })
    }
}

impl Buffers {
    fn clone_fallible(&self) -> Result<Self, Error> {
        let copy = |v: &[i64]| -> Result<Vec<i64>, Error> {
            let mut out = reserve(v.len(), 0)?;
            out.copy_from_slice(v);
            Ok(out)
        };
        let copy_u = |v: &[u64]| -> Result<Vec<u64>, Error> {
            let mut out = reserve(v.len(), 0)?;
            out.copy_from_slice(v);
            Ok(out)
        };
        let mut tasks = reserve(self.tasks.len(), 0u8)?;
        tasks.copy_from_slice(&self.tasks);
        Ok(Self {
            input: copy(&self.input)?,
            output: copy(&self.output)?,
            trace: copy_u(&self.trace)?,
            config: copy_u(&self.config)?,
            heap: copy(&self.heap)?,
            cleanup: copy_u(&self.cleanup)?,
            tasks,
        })
    }
    fn fill_unread(&mut self) {
        self.input.fill(UNREAD as i64);
        self.output.fill(UNREAD as i64);
        self.trace.fill(UNREAD);
        self.config.fill(UNREAD);
        self.heap.fill(UNREAD as i64);
        self.cleanup.fill(UNREAD);
        self.tasks.fill(0xa5);
    }
}

fn word(ty: &Type, arg: &eval::Value) -> Result<i64, Error> {
    Ok(match (ty, arg) {
        (Type::F32, eval::Value::F32(x)) => i64::from(x.to_bits()),
        (Type::I32, eval::Value::I32(x)) => i64::from(*x),
        (Type::U32, eval::Value::U32(x)) => i64::from(*x),
        (Type::Bool, eval::Value::Bool(x)) => i64::from(*x),
        _ => return Err(Error::Input),
    })
}

/// One root's slices in a snapshot.
struct Root<'a> {
    input: &'a [i64],
    output: &'a [i64],
    trace: &'a [u64],
    heap: &'a [i64],
    cleanup: &'a [u64],
    task: &'a [u8],
}
impl Root<'_> {
    fn counters(&self, frame_bytes: usize) -> [u64; COUNTERS] {
        std::array::from_fn(|i| {
            let at = frame_bytes + i * 8;
            u64::from_ne_bytes(self.task[at..at + 8].try_into().expect("8 bytes"))
        })
    }
    fn words(&self, frame_bytes: usize) -> TaskWords<'_> {
        TaskWords {
            input: self.input,
            output: self.output.try_into().expect("10 output words"),
            trace: self.trace,
            task: self.task,
            counters: self.counters(frame_bytes),
            heap: self.heap,
            cleanup: self.cleanup,
        }
    }
    fn identical(&self, other: &Root<'_>) -> bool {
        self.input == other.input
            && self.output == other.output
            && self.trace == other.trace
            && self.heap == other.heap
            && self.cleanup == other.cleanup
            && self.task == other.task
    }
}

/// An exclusive run over one sealed backend. Borrows its plan, so the
/// artifact and layout outlive every validation.
pub(crate) struct Session<'p, B: Backend> {
    plan: &'p Plan,
    lane: LaneLayout,
    backend: Option<B>,
    fence: Option<FencedExecution<B::Work, Sizes>>,
    state: State,
    poisoned: bool,
    config: Config,
    epochs: u64,
    host_bytes: usize,
    ledger: Ledger,
    /// Last validated snapshot (the initial image before the first epoch).
    previous: Buffers,
    /// Preallocated readback slot; the failing snapshot after a failure.
    candidate: Buffers,
    decode: (Vec<u64>, Vec<i64>, Vec<u64>),
}

impl<'p, B: Backend> Session<'p, B> {
    pub(crate) fn state(&self) -> State {
        self.state
    }
    pub(crate) fn plan_nonce(&self) -> u64 {
        self.plan.nonce
    }
    pub(crate) fn backend_kind(&self) -> BackendKind {
        B::KIND
    }
    pub(crate) fn host_bytes(&self) -> usize {
        self.host_bytes
    }
    pub(crate) fn ledger(&self) -> Ledger {
        self.ledger
    }
    pub(crate) fn epochs_bound(&self) -> u64 {
        self.epochs
    }
    /// Last validated and, after a failure, the failing snapshot.
    pub(crate) fn evidence(&self) -> (&Buffers, &Buffers) {
        (&self.previous, &self.candidate)
    }

    fn quarantine(&mut self) -> Error {
        self.state = State::Quarantined;
        if !self.poisoned {
            self.poisoned = true;
            if let Some(b) = self.backend.as_mut() {
                b.poison();
            }
        }
        Error::Unknown
    }

    pub(crate) fn submit(&mut self) -> Result<(), Error> {
        if self.state != State::Ready {
            return Err(Error::State(self.state));
        }
        let (Some(backend), Some(fence)) = (self.backend.as_mut(), self.fence.as_mut()) else {
            return Err(Error::State(self.state));
        };
        match fence.submit(|work| backend.submit(work)) {
            Ok(()) => {
                self.state = State::InFlight;
                Ok(())
            }
            Err(FenceError::Unknown) => Err(self.quarantine()),
            Err(FenceError::Submission) => {
                self.state = State::NotSubmitted;
                Err(Error::NotSubmitted)
            }
            Err(_) => {
                self.state = State::Failed;
                Err(Error::Limit)
            }
        }
    }

    pub(crate) fn cancel(&mut self) {
        match self.state {
            State::Ready => self.state = State::Cancelled,
            State::InFlight => {
                if let Some(fence) = self.fence.as_mut() {
                    fence.cancel();
                }
                self.state = State::CancelPending;
            }
            _ => {}
        }
    }

    pub(crate) fn poll(&mut self) -> Result<Progress, Error> {
        if !matches!(self.state, State::InFlight | State::CancelPending) {
            return Err(Error::State(self.state));
        }
        let (Some(backend), Some(fence)) = (self.backend.as_mut(), self.fence.as_mut()) else {
            return Err(Error::State(self.state));
        };
        match fence.poll(|work| backend.poll(work)) {
            Err(FenceError::Pending) => Ok(Progress::Pending),
            Err(FenceError::Unknown) => Err(self.quarantine()),
            Err(FenceError::Failed) => {
                self.state = State::FailedQuiescent;
                Err(Error::FailedQuiescent)
            }
            Err(FenceError::Cancelled) => {
                self.state = State::Cancelled;
                Ok(Progress::Cancelled)
            }
            Err(_) => {
                self.state = State::Failed;
                Err(Error::State(State::Failed))
            }
            Ok(()) => match self.observe() {
                Ok(progress) => Ok(progress),
                Err(e) => {
                    self.state = State::Failed;
                    Err(e)
                }
            },
        }
    }

    /// Read the completed epoch into the preallocated candidate, validate it
    /// against the last validated snapshot, then swap (no allocation).
    fn observe(&mut self) -> Result<Progress, Error> {
        let words = (self.previous.input.len()
            + self.previous.output.len()
            + self.previous.trace.len()
            + self.previous.config.len()
            + self.previous.heap.len()
            + self.previous.cleanup.len()
            + self.previous.tasks.len() / 8) as u64;
        self.ledger
            .charge(words.checked_mul(2).ok_or(Error::Work)?)?;
        self.candidate.fill_unread();
        let (Some(backend), Some(fence)) = (self.backend.as_ref(), self.fence.as_ref()) else {
            return Err(Error::State(self.state));
        };
        let candidate = &mut self.candidate;
        fence
            .observe(|work| backend.read_into(work, candidate.slots()))
            .map_err(|_| Error::State(State::Failed))?
            .map_err(Error::Backend)?;
        self.validate()?;
        std::mem::swap(&mut self.previous, &mut self.candidate);
        let lane = self.lane;
        let all_terminal = (0..self.config.roots)
            .all(|r| root(&self.previous, lane, r).counters(lane.frame_bytes)[1] == 1);
        if !all_terminal {
            self.state = State::Ready;
            return Ok(Progress::Advanced);
        }
        // Atomic publication: every root must also decode. On a decode
        // failure, keep the last good epoch as `previous` and the failing
        // (checker-valid but undecodable) epoch as `candidate` (N3).
        for r in 0..self.config.roots {
            if let Err(e) = self.decode_root(r) {
                std::mem::swap(&mut self.previous, &mut self.candidate);
                return Err(e);
            }
        }
        self.state = State::Published;
        Ok(Progress::Published)
    }

    fn validate(&mut self) -> Result<(), Error> {
        let p = self.plan;
        let lane = self.lane;
        let red = lane.red;
        let (old, new) = (&self.previous, &self.candidate);
        if new.config != old.config {
            return Err(Error::Validation(Validation::Config));
        }
        let sentinel = SENTINEL as i64;
        let red_ok = [
            red_i(&new.input, red, sentinel),
            red_i(&new.output, red, sentinel),
            red_u(&new.trace, red),
            red_u(&new.config, red),
            red_i(&new.heap, red, sentinel),
            red_u(&new.cleanup, red),
            new.tasks[new.tasks.len() - red * 8..]
                .iter()
                .all(|b| *b == 0x5a),
        ];
        if let Some(buffer) = red_ok.iter().position(|ok| !ok) {
            return Err(Error::Validation(Validation::RedZone { buffer }));
        }
        let lanes = (new.output.len() - red) / OUTPUT_WORDS;
        for lane in self.config.roots..lanes {
            let t = root(new, self.lane, lane);
            if t.input.iter().any(|w| *w != 0)
                || t.task.iter().any(|b| *b != 0)
                || t.output.iter().any(|w| *w != sentinel)
                || t.trace.iter().any(|w| *w != SENTINEL)
                || t.heap.iter().any(|w| *w != sentinel)
                || t.cleanup.iter().any(|w| *w != SENTINEL)
            {
                return Err(Error::Validation(Validation::Tail { lane }));
            }
        }
        let shape = p.shape();
        let limits = p.oracle_limits(self.config);
        let frame_bytes = lane.frame_bytes;
        // Structural checks for every root precede any checker call, so a
        // forged readback reports its most specific cause.
        for r in 0..self.config.roots {
            let (o, n) = (root(old, lane, r), root(new, lane, r));
            if o.input != n.input {
                return Err(Error::Validation(Validation::Input { root: r }));
            }
            if o.counters(frame_bytes)[1] == 1 {
                if !o.identical(&n) {
                    return Err(Error::Validation(Validation::Terminal { root: r }));
                }
            } else if n.counters(frame_bytes)[1] == 0 && n.output != PENDING {
                return Err(Error::Validation(Validation::PendingOutput { root: r }));
            }
        }
        let mut units = 0u64;
        for r in 0..self.config.roots {
            let (o, n) = (root(old, lane, r), root(new, lane, r));
            if o.counters(frame_bytes)[1] == 1 {
                continue;
            }
            units = units
                .checked_add(p.work_per_root(self.config).ok_or(Error::Work)?)
                .ok_or(Error::Work)?;
            match task::validate_device_epoch(
                &shape,
                limits,
                &o.words(frame_bytes),
                &n.words(frame_bytes),
                self.config.quantum,
            ) {
                Ok(()) => {}
                Err(task::TaskError::Allocation) => return Err(Error::Allocation),
                Err(task::TaskError::Malformed) => {
                    return Err(Error::Validation(Validation::Checker { root: r }));
                }
            }
        }
        self.ledger.charge(units)
    }

    fn decode_root(&mut self, r: usize) -> Result<oracle::Observation, Error> {
        let p = self.plan;
        let decode_units = p.decode_units(self.config).ok_or(Error::Work)?;
        self.ledger.charge(decode_units)?;
        let s = root(&self.previous, self.lane, r);
        let (trace, heap, cleanup) = &mut self.decode;
        trace.clear();
        trace.extend_from_slice(s.trace);
        heap.clear();
        heap.extend_from_slice(s.heap);
        cleanup.clear();
        cleanup.extend_from_slice(s.cleanup);
        let raw = invar_foreign::numeric_owned_resume::Invocation {
            output: s.output.try_into().expect("10 output words"),
            trace: std::mem::take(trace),
            heap: std::mem::take(heap),
            cleanup: std::mem::take(cleanup),
        };
        let decoded = owned_numeric_decode::decode(
            p.artifact.artifact().result(),
            p.artifact.source().len(),
            &p.sites,
            &p.program,
            p.storage(),
            p.oracle_limits(self.config),
            &raw,
        );
        self.decode = (raw.trace, raw.heap, raw.cleanup);
        decoded.map_err(|e| match e {
            owned_numeric_decode::Error::Allocation => Error::Allocation,
            _ => Error::Decode { root: r },
        })
    }

    /// A published result, decoded freshly (one decoded result at a time).
    /// N4: a result the caller keeps is caller memory, outside the session's
    /// 256 MiB total; the session holds at most one decoded result.
    pub(crate) fn result(&mut self, r: usize) -> Result<oracle::Observation, Error> {
        if self.state != State::Published || r >= self.config.roots {
            return Err(Error::State(self.state));
        }
        self.decode_root(r)
    }

    /// Return the backend and its work only when no submission can still
    /// reference them; the session is then `Reclaimed`. Otherwise nothing
    /// changes and no backend call is made.
    pub(crate) fn reclaim(&mut self) -> Result<(B, B::Work), Error> {
        if !self.quiescent() || self.state == State::Reclaimed {
            return Err(Error::State(self.state));
        }
        let (Some(fence), Some(backend)) = (self.fence.take(), self.backend.take()) else {
            return Err(Error::State(self.state));
        };
        self.state = State::Reclaimed;
        match fence.reclaim() {
            Ok(work) => Ok((backend, work)),
            Err(_) => {
                // Unreachable for a quiescent fence; if it ever refused, the
                // fence has forgotten the work, so the backend is kept too.
                std::mem::forget(backend);
                self.state = State::Quarantined;
                Err(Error::State(State::Quarantined))
            }
        }
    }
}

impl Plan {
    /// Modelled (not instrumented) per-root checker work: twice the root's
    /// words, the history (T + 1), the trace rows, the cleanup reconciliation
    /// and a conservative `L * ceil(log2(L + 1))` charge for the checker's
    /// `BTreeMap` of dead-owner lengths (L = allocations) (N1).
    fn work_per_root(&self, c: Config) -> Option<u64> {
        let root = u64::try_from(self.root_words(c)?).ok()?;
        let l = u64::try_from(self.storage().allocations()).ok()?;
        let log = u64::from(u64::BITS - l.checked_add(1)?.leading_zeros());
        root.checked_mul(2)?
            .checked_add(c.transitions.checked_add(1)?)?
            .checked_add(
                u64::try_from(c.trace_events)
                    .ok()?
                    .checked_mul(EVENT_WORDS as u64)?,
            )?
            .checked_add(
                u64::try_from(self.storage().allocations() + self.storage().cleanup_events())
                    .ok()?
                    .checked_mul(8)?,
            )?
            .checked_add(l.checked_mul(log)?.checked_mul(8)?)?
            .checked_add(64)
    }
    fn decode_units(&self, c: Config) -> Option<u64> {
        let root = u64::try_from(self.root_words(c)?).ok()?;
        root.checked_add(u64::try_from(c.trace_events).ok()?.checked_mul(8)?)
    }
}

fn red_i(v: &[i64], red: usize, s: i64) -> bool {
    v[v.len() - red..].iter().all(|w| *w == s)
}
fn red_u(v: &[u64], red: usize) -> bool {
    v[v.len() - red..].iter().all(|w| *w == SENTINEL)
}

/// Per-lane slice widths of one run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LaneLayout {
    input: usize,
    trace: usize,
    heap: usize,
    cleanup: usize,
    task: usize,
    red: usize,
    frame_bytes: usize,
}

/// Lane `r`'s slices in snapshot `b`.
fn root(b: &Buffers, l: LaneLayout, r: usize) -> Root<'_> {
    Root {
        input: &b.input[r * l.input..(r + 1) * l.input],
        output: &b.output[r * OUTPUT_WORDS..(r + 1) * OUTPUT_WORDS],
        trace: &b.trace[r * l.trace..(r + 1) * l.trace],
        heap: &b.heap[r * l.heap..(r + 1) * l.heap],
        cleanup: &b.cleanup[r * l.cleanup..(r + 1) * l.cleanup],
        task: &b.tasks[r * l.task..(r + 1) * l.task],
    }
}

impl<B: Backend> Session<'_, B> {
    /// Quiescent by both the session state and the owning fence (the fence
    /// marks a submission in flight before invoking the backend, so this also
    /// holds if a backend call unwinds).
    fn quiescent(&self) -> bool {
        self.state.quiescent()
            && !self.fence.as_ref().is_some_and(|f| {
                matches!(
                    f.state(),
                    ExecutionState::InFlight
                        | ExecutionState::CancelPending
                        | ExecutionState::Quarantined
                )
            })
    }
}

impl<B: Backend> Drop for Session<'_, B> {
    /// Non-quiescent work and its backend are forgotten together (D1/D2):
    /// the fence forgets the work and the backend is never dropped. Quiescent
    /// sessions drop the work before the backend.
    fn drop(&mut self) {
        if self.quiescent() {
            drop(self.fence.take());
            drop(self.backend.take());
        } else if let Some(backend) = self.backend.take() {
            std::mem::forget(backend);
        }
    }
}

#[cfg(unix)]
mod cpu;

#[cfg(test)]
#[path = "../../tests/support/cuda_owned_numeric_host_lifecycle.rs"]
mod lifecycle;
