//! Program-specific integer/array control/frame compiler with bounded CPU child
//! extraction. Generated code contains no runtime graph opcodes or interpreter.

use super::{Net, Proof, coverage};
use crate::{
    front::{Span, Type},
    mir,
};
use std::collections::BTreeMap;

mod aggregate;
pub mod body_map;
mod device;
mod device_fence;
mod emit;
mod entry;
mod f32;
mod fault;
mod owned;
mod owned_emit;
mod owned_f32;
mod owned_numeric_device;
mod reachable;
mod resume;
mod speculation;
mod speculation_analysis;
pub(crate) use aggregate::MAX_RECORD_FIELDS;
pub use aggregate::{AggregateForm, AggregateKind, AggregateLayout};
pub use device::{BufferRange, DeviceConfig, DeviceLayout, Workset};
pub use device_fence::{ExecutionState, FenceError, FenceObservation, FencedExecution, Submission};
pub use entry::EntryLayout;
#[cfg(feature = "gnet-fault-injection")]
pub use fault::{Fault, HOOK as FAULT_HOOK};
pub use owned::{OwnedLayout, OwnedLimits};
pub use resume::ContinuationLayout;
pub use speculation::{SpeculationLayout, SpeculationLimits};
pub use speculation_analysis::{CallSite, Candidate, ForkPlan, Serialization, SerializationReason};

impl SerializationReason {
    /// Stable label for placement legality receipts (GNET-04, #47).
    pub fn label(self) -> &'static str {
        match self {
            Self::AffineArgument => "affine_argument",
            Self::OwnerResult => "owner_result",
            Self::DependentArgument => "dependent_argument",
            Self::OrderedPrelude => "ordered_prelude",
            Self::ProjectionLimit => "projection_limit",
        }
    }
}

pub const OUTPUT_WORDS: usize = 8;
pub const EVENT_WORDS: usize = 7;
pub const MAX_FRAMES: usize = 64;
pub const KERNEL_NAME: &str = "invar_gnet_v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Cpu,
    Cuda,
    Metal,
}

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub values: usize,
    pub places: usize,
    pub states: usize,
    pub frame_bytes: usize,
    pub source_bytes: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            values: 128,
            places: 64,
            states: 4096,
            frame_bytes: 128 * 1024,
            source_bytes: 1024 * 1024,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Boundary(super::Error),
    UnsupportedType,
    UnsupportedRule(coverage::Rule),
    UnsupportedCall,
    Entry,
    Limit,
    State,
    SpeculationTarget,
    DeviceSpeculationOwner,
    AggregateTarget,
    NumericTarget,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Status {
    Success = 0,
    IntegerOverflow = 1,
    DivisionByZero = 2,
    FuelLimit = 3,
    CallDepthLimit = 4,
    TransitionLimit = 5,
    TraceLimit = 6,
    InvalidState = 7,
    InvalidInput = 8,
    IndexOutOfBounds = 9,
    ArraySizeLimit = 10,
    HeapLimit = 11,
    AllocationLimit = 12,
    CleanupLimit = 13,
    WorklistLimit = 14,
}

#[derive(Debug)]
pub struct Artifact {
    source: String,
    target: Target,
    params: Vec<Type>,
    result: Type,
    states: usize,
    emitted_states: usize,
    /// Conservative bound for the full 64-frame private stack per invocation.
    frame_bytes: usize,
    owned: Option<OwnedLayout>,
    continuation: Option<ContinuationLayout>,
    speculation: Option<SpeculationLayout>,
    fork_plan: Option<ForkPlan>,
    device: Option<DeviceLayout>,
    aggregate: Option<AggregateLayout>,
    numeric: Option<crate::numeric_contract::NumericPolicy>,
    entry_layout: Option<EntryLayout>,
}
impl Artifact {
    pub fn source(&self) -> &str {
        &self.source
    }
    pub fn target(&self) -> Target {
        self.target
    }
    pub fn params(&self) -> &[Type] {
        &self.params
    }
    pub fn result(&self) -> &Type {
        &self.result
    }
    /// Stable logical cursor slots, including slots whose code is unreachable.
    pub fn states(&self) -> usize {
        self.states
    }
    /// Reachable switch arms emitted from the stable logical cursor table.
    pub fn emitted_states(&self) -> usize {
        self.emitted_states
    }
    pub fn frame_bytes(&self) -> usize {
        self.frame_bytes
    }
    pub fn abi_version(&self) -> u32 {
        if self.entry_layout.is_some() {
            13
        } else if self.numeric.is_some() {
            // CPU numeric artifacts are always ABI11 owner continuations; ABI12
            // `cuda-owned-numeric-continuation-v1` is the same source on CUDA.
            if self.target == Target::Cuda { 12 } else { 11 }
        } else if self.aggregate.is_some() {
            if self.continuation.is_some() { 7 } else { 6 }
        } else if self.device.is_some() {
            5
        } else if self.speculation.is_some() {
            4
        } else if self.continuation.is_some() {
            3
        } else if self.owned.is_some() {
            2
        } else {
            1
        }
    }
    pub fn entry_layout(&self) -> Option<&EntryLayout> {
        self.entry_layout.as_ref()
    }
    pub fn numeric_policy(&self) -> Option<crate::numeric_contract::NumericPolicy> {
        self.numeric
    }
    pub fn aggregate_layout(&self) -> Option<&AggregateLayout> {
        self.aggregate.as_ref()
    }
    pub fn device_layout(&self) -> Option<&DeviceLayout> {
        self.device.as_ref()
    }
    pub fn speculation_layout(&self) -> Option<&SpeculationLayout> {
        self.speculation.as_ref()
    }
    pub fn fork_plan(&self) -> Option<&ForkPlan> {
        self.fork_plan.as_ref()
    }
    pub fn continuation_layout(&self) -> Option<&ContinuationLayout> {
        self.continuation.as_ref()
    }
    pub fn owned_layout(&self) -> Option<&OwnedLayout> {
        self.owned.as_ref()
    }
}

struct Layout {
    values: BTreeMap<mir::ValueId, (usize, Type)>,
    places: BTreeMap<String, usize>,
}

#[derive(Clone, Copy)]
struct State {
    function: usize,
    block: usize,
    start: usize,
    pc: u32,
    reachable: bool,
}

struct Program<'a> {
    mir: &'a mir::Program,
    layout: Vec<Layout>,
    states: Vec<State>,
    pc: BTreeMap<(usize, usize, usize), u32>,
    calls: BTreeMap<(usize, usize, usize), usize>,
    entry: usize,
    vsize: usize,
    psize: usize,
    bytes: usize,
    owned: Option<OwnedLayout>,
    resumable: bool,
    speculation: Option<(SpeculationLayout, ForkPlan)>,
    device: bool,
    aggregate: Option<AggregateLayout>,
    numeric: Option<crate::numeric_contract::NumericPolicy>,
    entry_layout: Option<EntryLayout>,
    /// Emission target; only target-specific preambles consult it.
    target: Target,
    /// Test-only device fault; uninhabited without `gnet-fault-injection`.
    fault: Option<fault::Fault>,
    body_regions: Vec<body_map::Site>,
}

fn scalar(ty: &Type) -> bool {
    matches!(ty, Type::I32 | Type::U32 | Type::Bool | Type::Unit)
}

impl<'a> Program<'a> {
    #[allow(clippy::too_many_arguments)] // Private builder binds independent immutable profiles.
    fn new(
        mir: &'a mir::Program,
        entry: &str,
        limits: Limits,
        owned: Option<OwnedLayout>,
        resumable: bool,
        aggregate: Option<AggregateLayout>,
        numeric: Option<crate::numeric_contract::NumericPolicy>,
    ) -> Result<Self, Error> {
        let entry = mir
            .functions
            .iter()
            .position(|f| f.name == entry)
            .ok_or(Error::Entry)?;
        let reachable = reachable::blocks(mir, entry)?;
        let mut program = Self {
            mir,
            layout: Vec::new(),
            states: Vec::new(),
            pc: BTreeMap::new(),
            calls: BTreeMap::new(),
            entry,
            vsize: 1,
            psize: 1,
            bytes: 0,
            owned,
            resumable,
            speculation: None,
            device: false,
            aggregate,
            numeric,
            entry_layout: None,
            target: Target::Cpu,
            fault: None,
            body_regions: Vec::new(),
        };
        if numeric.is_some()
            && program.aggregate.is_none()
            && (!mir.records.is_empty() || !mir.variants.is_empty())
        {
            return Err(Error::UnsupportedType);
        }
        for (f, function) in mir.functions.iter().enumerate() {
            if function.foreign_oracle.is_some() {
                return Err(Error::UnsupportedCall);
            }
            let admitted = |ty: &Type| {
                scalar(ty)
                    || program.numeric.is_some() && ty == &Type::F32
                    || program.owned.is_some() && program.handle(ty)
            };
            if !admitted(&function.result)
                || function.locals.values().any(|local| !admitted(&local.ty))
                || function.params.iter().any(|param| !admitted(&param.ty))
            {
                return Err(Error::UnsupportedType);
            }
            let mut values = BTreeMap::new();
            for block in &function.blocks {
                for (id, ty) in block
                    .params
                    .iter()
                    .chain(block.instructions.iter().filter_map(|i| i.output.as_ref()))
                {
                    if !admitted(ty) {
                        return Err(Error::UnsupportedType);
                    }
                    let index = values.len();
                    values.insert(*id, (index, ty.clone()));
                }
            }
            // IDs are mapped densely; hostile sparse IDs never size an allocation.
            if values.len() > limits.values || function.locals.len() > limits.places {
                return Err(Error::Limit);
            }
            program.vsize = program.vsize.max(values.len());
            program.psize = program.psize.max(function.locals.len());
            program.layout.push(Layout {
                values,
                places: function
                    .declaration_order
                    .iter()
                    .enumerate()
                    .map(|(i, n)| (n.clone(), i))
                    .collect(),
            });
            for (b, block) in function.blocks.iter().enumerate() {
                let mut starts = vec![0];
                for (i, instruction) in block.instructions.iter().enumerate() {
                    match &instruction.kind {
                        mir::Operation::Call { name, .. } => {
                            if let Some(target) = mir.functions.iter().position(|f| f.name == *name)
                            {
                                if reachable.contains(&(f, b)) {
                                    program.calls.insert((f, b, i), target);
                                }
                                starts.push(i + 1);
                            } else if !(matches!(name.as_str(), "wrapping_add" | "wrapping_mul")
                                || program.numeric.is_some()
                                    && matches!(
                                        name.as_str(),
                                        "fma"
                                            | "f32_from_i32"
                                            | "f32_from_u32"
                                            | "sin"
                                            | "cos"
                                            | "sqrt"
                                    )
                                || program.owned.is_some()
                                    && (matches!(name.as_str(), "filled_i32" | "filled_u32")
                                        || program.numeric.is_some() && name == "filled_f32"))
                            {
                                return Err(Error::UnsupportedCall);
                            }
                        }
                        op if !(program.numeric.is_some()
                            && matches!(op, mir::Operation::F32(_))
                            || matches!(
                                op,
                                mir::Operation::Tick
                                    | mir::Operation::I32(_)
                                    | mir::Operation::U32(_)
                                    | mir::Operation::Bool(_)
                                    | mir::Operation::Read(_)
                                    | mir::Operation::Unary { .. }
                                    | mir::Operation::Binary { .. }
                                    | mir::Operation::Bind { .. }
                                    | mir::Operation::Store { .. }
                                    | mir::Operation::Drop(_)
                                    | mir::Operation::ForNext(_)
                            )
                            || program.owned.is_some() && owned::operation(op)
                            || program.aggregate.is_some()
                                && aggregate::operation_supported(op)) =>
                        {
                            return Err(Error::UnsupportedRule(coverage::operation(op).0));
                        }
                        _ => {}
                    }
                    for span in [
                        instruction.span,
                        match &instruction.kind {
                            mir::Operation::Unary { operator, .. }
                            | mir::Operation::Binary { operator, .. } => *operator,
                            _ => instruction.span,
                        },
                    ] {
                        if span.start > span.end {
                            return Err(Error::State);
                        }
                        i64::try_from(span.start).map_err(|_| Error::Limit)?;
                        i64::try_from(span.end).map_err(|_| Error::Limit)?;
                    }
                }
                if resumable {
                    starts = (0..=block.instructions.len()).collect();
                }
                for start in starts {
                    if program.states.len() >= limits.states {
                        return Err(Error::Limit);
                    }
                    let pc = u32::try_from(program.states.len()).map_err(|_| Error::Limit)?;
                    program.pc.insert((f, b, start), pc);
                    program.states.push(State {
                        function: f,
                        block: b,
                        start,
                        pc,
                        reachable: reachable.contains(&(f, b)),
                    });
                }
            }
        }
        program.bytes = program
            .vsize
            .checked_add(program.psize)
            .and_then(|n| n.checked_add(4))
            .and_then(|n| n.checked_mul(8 * MAX_FRAMES))
            .ok_or(Error::Limit)?;
        if program.bytes > limits.frame_bytes {
            return Err(Error::Limit);
        }
        Ok(program)
    }
    fn handle(&self, ty: &Type) -> bool {
        owned::array(ty)
            || self.numeric.is_some() && self.owned.is_some() && ty == &Type::ArrayF32
            || self.aggregate.as_ref().is_some_and(|a| a.accepts(ty))
    }
    fn value(&self, f: usize, id: mir::ValueId) -> Result<usize, Error> {
        self.layout
            .get(f)
            .and_then(|l| l.values.get(&id))
            .map(|(slot, _)| *slot)
            .ok_or(Error::State)
    }
    fn ty(&self, f: usize, id: mir::ValueId) -> Result<&Type, Error> {
        self.layout
            .get(f)
            .and_then(|l| l.values.get(&id))
            .map(|(_, ty)| ty)
            .ok_or(Error::State)
    }
    fn place(&self, f: usize, name: &str) -> Result<usize, Error> {
        self.layout
            .get(f)
            .and_then(|l| l.places.get(name))
            .copied()
            .ok_or(Error::State)
    }
    fn pc(&self, f: usize, b: usize, start: usize) -> Result<u32, Error> {
        self.pc.get(&(f, b, start)).copied().ok_or(Error::State)
    }
}

/// Emit only from a current verified net. The whole candidate is admitted;
/// aggregate/host forms fail explicitly even in an uncalled function.
pub fn emit(
    net: &Net,
    proof: &Proof,
    entry: &str,
    target: Target,
    limits: Limits,
) -> Result<Artifact, Error> {
    let mir = net.compiled_mir(proof).map_err(Error::Boundary)?;
    artifact(
        &mir, entry, target, limits, None, false, None, false, false, false,
    )
}

/// Program-specific owned i32/u32 arrays with separate bounded storage and
/// cleanup ABI. External entry parameters remain scalar; owned results and
/// array parameters between compiled functions are supported.
pub fn emit_owned(
    net: &Net,
    proof: &Proof,
    entry: &str,
    target: Target,
    limits: Limits,
    storage: OwnedLimits,
) -> Result<Artifact, Error> {
    let mir = net.compiled_mir(proof).map_err(Error::Boundary)?;
    let layout = OwnedLayout::new(storage)?;
    artifact(
        &mir,
        entry,
        target,
        limits,
        Some(layout),
        false,
        None,
        false,
        false,
        false,
    )
}

/// Compile continuations that yield after a finite instruction quantum. Each
/// root owns its task, trace and heap. Calls suspend their parent in source order.
/// This does not speculate or extract independent children.
pub fn emit_resumable(
    net: &Net,
    proof: &Proof,
    entry: &str,
    target: Target,
    limits: Limits,
    storage: OwnedLimits,
) -> Result<Artifact, Error> {
    let mir = net.compiled_mir(proof).map_err(Error::Boundary)?;
    artifact(
        &mir,
        entry,
        target,
        limits,
        Some(OwnedLayout::new(storage)?),
        true,
        None,
        false,
        false,
        false,
    )
}

/// Compiled CPU child speculation over ABI3 continuations. Private scratch
/// never consumes serial-path storage; logical publication stays source ordered.
/// This initial backend does not dispatch CUDA/Metal speculation.
#[allow(clippy::too_many_arguments)]
pub fn emit_speculative(
    net: &Net,
    proof: &Proof,
    entry: &str,
    target: Target,
    limits: Limits,
    storage: OwnedLimits,
    speculation: SpeculationLimits,
) -> Result<Artifact, Error> {
    let mir = net.compiled_mir(proof).map_err(Error::Boundary)?;
    artifact(
        &mir,
        entry,
        target,
        limits,
        Some(OwnedLayout::new(storage)?),
        true,
        Some(speculation),
        false,
        false,
        false,
    )
}

/// Fixed32-lane single-group scalar child execution with source-ordered commit.
/// Host dispatch must validate worksets and fence every epoch before publication.
#[allow(clippy::too_many_arguments)]
pub fn emit_device_speculative(
    net: &Net,
    proof: &Proof,
    entry: &str,
    target: Target,
    limits: Limits,
    storage: OwnedLimits,
    speculation: SpeculationLimits,
) -> Result<Artifact, Error> {
    let mir = net.compiled_mir(proof).map_err(Error::Boundary)?;
    artifact(
        &mir,
        entry,
        target,
        limits,
        Some(OwnedLayout::new(storage)?),
        true,
        Some(speculation),
        true,
        false,
        false,
    )
}

/// `emit_device_speculative` plus one test-only device fault hook. The hook is
/// a separate non-inlined function raced by an idle group thread.
#[cfg(feature = "gnet-fault-injection")]
#[allow(clippy::too_many_arguments)]
pub fn emit_device_speculative_with_fault(
    net: &Net,
    proof: &Proof,
    entry: &str,
    target: Target,
    limits: Limits,
    storage: OwnedLimits,
    speculation: SpeculationLimits,
    fault: Fault,
) -> Result<Artifact, Error> {
    let mir = net.compiled_mir(proof).map_err(Error::Boundary)?;
    artifact_with(
        &mir,
        entry,
        target,
        limits,
        Some(OwnedLayout::new(storage)?),
        true,
        Some(speculation),
        true,
        false,
        false,
        Some(fault),
    )
}

/// CPU ABI6 records with scalar/integer-array fields and scalar variants.
/// Nested nominal/blob/f32 members and device execution fail explicitly.
pub fn emit_aggregates(
    net: &Net,
    proof: &Proof,
    entry: &str,
    target: Target,
    limits: Limits,
    storage: OwnedLimits,
) -> Result<Artifact, Error> {
    let mir = net.compiled_mir(proof).map_err(Error::Boundary)?;
    artifact(
        &mir,
        entry,
        target,
        limits,
        Some(OwnedLayout::new(storage)?),
        false,
        None,
        false,
        true,
        false,
    )
}

/// CPU ABI7 aggregate continuations over the ABI6 affine heap. Each exclusively
/// owned task resumes at its generated state after a finite instruction quantum.
/// This does not admit device or speculative aggregate execution.
pub fn emit_resumable_aggregates(
    net: &Net,
    proof: &Proof,
    entry: &str,
    target: Target,
    limits: Limits,
    storage: OwnedLimits,
) -> Result<Artifact, Error> {
    let mir = net.compiled_mir(proof).map_err(Error::Boundary)?;
    artifact(
        &mir,
        entry,
        target,
        limits,
        Some(OwnedLayout::new(storage)?),
        true,
        None,
        false,
        true,
        false,
    )
}

/// CPU ABI11 strict-f32-v1 numeric continuations, the only CPU numeric ABI:
/// f32 scalars plus direct i32/u32/f32 array owners, a ten-word terminal packet
/// and the scalar continuation task/frame layout. The CPU
/// entry `gnet_resume` takes a zero-initialized task plus the same zeroed heap,
/// cleanup, trace and output buffers on every quantum; output is written only
/// when it returns true. Fixed semantic budgets bind at first start; invalid
/// later arguments finish the task with status 8. The quantum counts
/// transitions, not native work.
///
/// CUDA emits ABI12 `cuda-owned-numeric-continuation-v1` (stage 1): the same
/// admitted source, task layout and ten-word terminal packet, the integer
/// software binary32 helpers, a header-free numeric preamble and the distinct
/// kernel `invar_gnet_owned_numeric_resume_v1`. Emission is not executable host
/// support: no device workset, fence, receipt or loader exists for ABI12 yet.
/// Metal returns `NumericTarget`.
pub fn emit_resumable_owned_numeric(
    net: &Net,
    proof: &Proof,
    entry: &str,
    target: Target,
    limits: Limits,
    storage: OwnedLimits,
) -> Result<Artifact, Error> {
    let mir = net.compiled_mir(proof).map_err(Error::Boundary)?;
    owned_f32::validate(&mir, entry, target)?;
    artifact(
        &mir,
        entry,
        target,
        limits,
        Some(OwnedLayout::new(storage)?),
        true,
        None,
        false,
        false,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn artifact(
    mir: &mir::Program,
    entry: &str,
    target: Target,
    limits: Limits,
    owned: Option<OwnedLayout>,
    resumable: bool,
    speculation: Option<SpeculationLimits>,
    device: bool,
    aggregates: bool,
    numeric: bool,
) -> Result<Artifact, Error> {
    artifact_with(
        mir,
        entry,
        target,
        limits,
        owned,
        resumable,
        speculation,
        device,
        aggregates,
        numeric,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn artifact_with(
    mir: &mir::Program,
    entry: &str,
    target: Target,
    limits: Limits,
    owned: Option<OwnedLayout>,
    resumable: bool,
    speculation: Option<SpeculationLimits>,
    device: bool,
    aggregates: bool,
    numeric: bool,
    fault: Option<fault::Fault>,
) -> Result<Artifact, Error> {
    let floating = |ty: &Type| matches!(ty, Type::F32 | Type::ArrayF32);
    let extended = device
        && (mir.functions.iter().any(|f| {
            floating(&f.result)
                || f.params.iter().any(|p| floating(&p.ty))
                || f.locals.values().any(|l| floating(&l.ty))
                || f.blocks.iter().any(|b| {
                    b.params.iter().any(|(_, t)| floating(t))
                        || b.instructions
                            .iter()
                            .any(|i| i.output.as_ref().is_some_and(|(_, t)| floating(t)))
                })
        }) || !mir.records.is_empty()
            || mir
                .functions
                .iter()
                .find(|f| f.name == entry)
                .is_some_and(|f| {
                    entry::owner(&f.result) || f.params.iter().any(|p| entry::owner(&p.ty))
                }));
    let numeric = numeric || extended;
    // Numeric artifacts are CPU-only except ABI12: CUDA resumable owners.
    let owned_numeric_cuda =
        numeric && target == Target::Cuda && resumable && owned.is_some() && !device;
    if numeric && target != Target::Cpu && !owned_numeric_cuda && !extended {
        return Err(Error::NumericTarget);
    }
    let numeric = numeric.then_some(crate::numeric_contract::NumericPolicy::STRICT_F32);
    if aggregates && target != Target::Cpu {
        return Err(Error::AggregateTarget);
    }
    let aggregate = if extended && !mir.records.is_empty() {
        Some(AggregateLayout::device(mir)?)
    } else if aggregates {
        Some(AggregateLayout::new(mir)?)
    } else {
        None
    };
    let mut program = Program::new(mir, entry, limits, owned, resumable, aggregate, numeric)?;
    program.device = device;
    program.target = target;
    program.fault = fault;
    if extended {
        program.entry_layout = Some(EntryLayout::new(&program)?);
    }
    if let Some(config) = speculation {
        if device == (target == Target::Cpu) {
            return Err(Error::SpeculationTarget);
        }
        // GNET-03: the root continuation owns the device heap/cleanup buffers
        // and owner children replay privately (never commit), exactly as on
        // the CPU ABI4 path. ABI5 retains a scalar host boundary; ABI13's
        // terminal packet may publish a validated owner handle. The
        // completed root heap is fenced and decoded before host publication.
        if device
            && !(scalar(&mir.functions[program.entry].result)
                || extended
                    && (program.handle(&mir.functions[program.entry].result)
                        || mir.functions[program.entry].result == Type::F32
                        || program
                            .aggregate
                            .as_ref()
                            .is_some_and(|a| a.accepts(&mir.functions[program.entry].result))))
        {
            return Err(Error::DeviceSpeculationOwner);
        }
        let layout = SpeculationLayout::new(config, &program)?;
        let mut plan = speculation_analysis::analyze(mir, config.lookahead)?;
        if config.slots == 0 {
            plan.candidates.clear();
        }
        let reachable = reachable::blocks(mir, program.entry)?;
        plan.candidates
            .retain(|c| reachable.contains(&(c.source.function, c.source.block)));
        plan.serialized
            .retain(|c| reachable.contains(&(c.source.function, c.source.block)));
        program.speculation = Some((layout, plan));
    }
    if mir.functions[program.entry].params.iter().any(|p| {
        !(matches!(p.ty, Type::I32 | Type::U32 | Type::Bool)
            || numeric.is_some() && p.ty == Type::F32
            || extended && program.handle(&p.ty))
    }) {
        return Err(Error::UnsupportedType);
    }
    let source = emit::source(&program, target, limits.source_bytes)?;
    let device_layout = if device {
        Some(DeviceLayout::new(&program, target)?)
    } else {
        None
    };
    Ok(Artifact {
        source,
        entry_layout: program.entry_layout.take(),
        aggregate: program.aggregate.take(),
        numeric: program.numeric,
        device: device_layout,
        target,
        params: mir.functions[program.entry]
            .params
            .iter()
            .map(|p| p.ty.clone())
            .collect(),
        result: mir.functions[program.entry].result.clone(),
        states: program.states.len(),
        emitted_states: program
            .states
            .iter()
            .filter(|state| state.reachable)
            .count(),
        frame_bytes: program.bytes,
        continuation: program.resumable.then(|| {
            if owned_numeric_cuda {
                ContinuationLayout::owned_numeric_cuda(program.bytes)
            } else {
                ContinuationLayout::new(program.bytes)
            }
        }),
        speculation: program
            .speculation
            .as_ref()
            .map(|(layout, _)| layout.clone()),
        fork_plan: program.speculation.map(|(_, plan)| plan),
        owned: program.owned,
    })
}

fn fail(status: Status, span: Option<Span>) -> String {
    let (start, end) = span.map(|s| (s.start, s.end)).unwrap_or((0, 0));
    format!(
        "status={}; start={}; end={}; GNET_STOP;",
        status as u32, start, end
    )
}
