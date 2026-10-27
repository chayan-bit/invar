//! Bounded CPU net semantic oracle with declared floating contracts.
//! Explicit heap frames and selected-block tokens preserve source order; this
//! test oracle is not a production interpreter or scheduler. Unsupported owners
//! and host rules remain explicit coverage gaps.

use std::collections::BTreeMap;

use super::{CallTarget, Net, Proof, Route, Token, coverage::Rule};
use crate::{
    eval::{self, EvalError, TrapCategory, Value},
    front::{BinaryOp, Mode, Span, Type, UnaryOp},
    mir,
};

const MAX_CALL_DEPTH: usize = 64;
#[path = "oracle_aggregates.rs"]
mod aggregates;
#[path = "oracle_arrays.rs"]
mod arrays;
#[path = "oracle_f32.rs"]
mod floating;
#[path = "oracle_units.rs"]
mod units;

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub fuel: u64,
    /// Physical oracle steps are independent of source semantic fuel.
    pub transitions: u64,
    pub trace_events: usize,
    /// Shared live-unit budget for arrays, record slots, variant shells and blobs.
    pub array_elements: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            fuel: 100_000,
            transitions: 1_000_000,
            trace_events: 1_000_000,
            array_elements: 2_000_000,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventKind {
    Demand,
    Complete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Event {
    pub kind: EventKind,
    pub frame: u64,
    pub activation: u64,
    pub token: Token,
    pub rule: Rule,
}

#[derive(Debug)]
pub struct Observation {
    pub semantic: eval::Observation,
    pub failure: Option<ToolFailure>,
    pub events: Vec<Event>,
    pub transitions: u64,
    pub max_frames: usize,
}

#[derive(Clone, Copy, Debug)]
enum Scalar {
    F32(crate::numeric_contract::F32Bits),
    I32(i32),
    U32(u32),
    Bool(bool),
    Unit,
    Array(usize),
    View(arrays::View),
    Aggregate(usize),
}

impl Scalar {
    fn value(
        self,
        arena: &arrays::Arena,
        aggregates: &aggregates::Arena,
    ) -> Result<Value, OracleError> {
        Ok(match self {
            Self::F32(x) => Value::F32(x),
            Self::I32(x) => Value::I32(x),
            Self::U32(x) => Value::U32(x),
            Self::Bool(x) => Value::Bool(x),
            Self::Unit => Value::Unit,
            Self::Array(id) => return arena.value(id),
            Self::View(view) => return arena.view_value(view),
            Self::Aggregate(id) => return aggregates.value(id, arena),
        })
    }
}

/// Oracle-specific failure distinctions reuse the registered public diagnostic
/// categories at the compatibility boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolFailure {
    Unsupported,
    TransitionLimit,
    TraceLimit,
    Allocation,
    InvalidState,
}

#[derive(Debug)]
enum OracleError {
    Semantic(EvalError),
    Tool { kind: ToolFailure, message: String },
}

impl From<EvalError> for OracleError {
    fn from(error: EvalError) -> Self {
        Self::Semantic(error)
    }
}

fn tool(kind: ToolFailure, message: impl Into<String>) -> OracleError {
    OracleError::Tool {
        kind,
        message: message.into(),
    }
}

fn semantic_tool(code: &'static str, message: impl Into<String>) -> OracleError {
    EvalError::Tool {
        code,
        message: message.into(),
    }
    .into()
}

fn internal() -> OracleError {
    tool(
        ToolFailure::InvalidState,
        "inconsistent verified oracle state",
    )
}

pub fn supports(rule: Rule) -> bool {
    match rule {
        Rule::F32
        | Rule::Tick
        | Rule::I32
        | Rule::U32
        | Rule::Bool
        | Rule::Read
        | Rule::Unary
        | Rule::Binary
        | Rule::Call
        | Rule::Bind
        | Rule::Store
        | Rule::Drop
        | Rule::ForNext
        | Rule::Array
        | Rule::ArrayRef
        | Rule::Move
        | Rule::Index
        | Rule::Length
        | Rule::StoreIndex
        | Rule::Goto
        | Rule::Branch
        | Rule::Return
        | Rule::Blob
        | Rule::MoveField
        | Rule::RecordRef
        | Rule::Record
        | Rule::Field
        | Rule::Variant
        | Rule::VariantTag
        | Rule::VariantPayload
        | Rule::VariantTakePayload
        | Rule::DropValue => true,
        // GNET has no binary64 encoding; f64 programs stay on the CPU engines.
        Rule::F64 => false,
    }
}

fn admit(net: &Net, program: &mir::Program) -> Result<(), OracleError> {
    let accepts = |ty: &Type| aggregates::accepts_type(ty, program);
    for function in &program.functions {
        if function.foreign_oracle.is_some() {
            return Err(tool(
                ToolFailure::Unsupported,
                "foreign call is an explicit host boundary, not a CPU oracle rule",
            ));
        }
        if !accepts(&function.result)
            || function.params.iter().any(|p| !accepts(&p.ty))
            || function.locals.values().any(|p| !accepts(&p.ty))
        {
            return Err(tool(
                ToolFailure::Unsupported,
                "type without a CPU oracle representation",
            ));
        }
        for block in &function.blocks {
            if block.params.iter().any(|(_, ty)| !accepts(ty)) {
                return Err(tool(ToolFailure::Unsupported, "nonscalar block parameter"));
            }
        }
    }
    for agent in &net.agents {
        if !supports(agent.rule) || agent.output.as_ref().is_some_and(|(_, ty)| !accepts(ty)) {
            return Err(tool(
                ToolFailure::Unsupported,
                format!("CPU oracle rule {:?} pending", agent.rule),
            ));
        }
        if let Some(CallTarget::Intrinsic(name)) = &agent.call
            && !matches!(
                name.as_str(),
                "wrapping_add" | "wrapping_mul" | "filled_i32" | "filled_u32" | "filled_f32"
            )
            && !floating::supports_intrinsic(name)
            && units::rule(name).is_none()
            && !aggregates::is_text_intrinsic(name)
            && !arrays::is_view_intrinsic(name)
            && crate::check::sort_modes(name).is_none()
            && crate::fsum::Kind::from_name(name).is_none()
        {
            return Err(tool(
                ToolFailure::Unsupported,
                format!("CPU oracle intrinsic {name} pending"),
            ));
        }
    }
    Ok(())
}

struct Frame {
    id: u64,
    activation: u64,
    function: usize,
    block: usize,
    position: usize,
    places: BTreeMap<String, Slot>,
    values: BTreeMap<mir::ValueId, Scalar>,
    waiting: Option<(Option<mir::ValueId>, Event)>,
    tc_inputs: Option<([usize; 2], [u32; 3])>,
}

struct Slot {
    value: Option<Scalar>,
    owned: bool,
}

impl Frame {
    fn new(
        program: &mir::Program,
        function: usize,
        args: Vec<Scalar>,
        id: u64,
    ) -> Result<Self, OracleError> {
        let body = program.functions.get(function).ok_or_else(internal)?;
        if body.params.len() != args.len() {
            return Err(internal());
        }
        let tc_inputs = if crate::contraction_source::mir_is_tc(body) {
            let [
                Scalar::Array(a),
                Scalar::Array(b),
                Scalar::U32(r),
                Scalar::U32(k),
                Scalar::U32(n),
            ] = args.as_slice()
            else {
                return Err(internal());
            };
            Some(([*a, *b], [*r, *k, *n]))
        } else {
            None
        };
        Ok(Self {
            id,
            activation: 0,
            function,
            block: body.entry,
            position: 0,
            places: body
                .params
                .iter()
                .zip(args)
                .map(|(p, v)| {
                    (
                        p.name.clone(),
                        Slot {
                            value: Some(v),
                            owned: !matches!(p.mode, Some(Mode::Read | Mode::Mut)),
                        },
                    )
                })
                .collect(),
            values: BTreeMap::new(),
            waiting: None,
            tc_inputs,
        })
    }

    fn get(&self, value: mir::ValueId) -> Result<Scalar, OracleError> {
        self.values.get(&value).copied().ok_or_else(internal)
    }
}

struct Machine<'a> {
    external: Vec<Scalar>,
    net: &'a Net,
    program: &'a mir::Program,
    limits: Limits,
    steps: u64,
    transitions: u64,
    max_frames: usize,
    events: Vec<Event>,
    arrays: arrays::Arena,
    aggregates: aggregates::Arena,
    cleanup: Vec<eval::CleanupEvent>,
    agent_index: BTreeMap<(usize, usize, usize), usize>,
    control_index: BTreeMap<(usize, usize), usize>,
}

impl Machine<'_> {
    fn emit(&mut self, event: Event) -> Result<(), OracleError> {
        if self.events.len() >= self.limits.trace_events {
            return Err(tool(
                ToolFailure::TraceLimit,
                "logical observation trace budget exhausted",
            ));
        }
        self.events
            .try_reserve(1)
            .map_err(|_| tool(ToolFailure::Allocation, "trace allocation failed"))?;
        self.events.push(event);
        Ok(())
    }

    fn run(&mut self, name: &str, args: &[Value], main_entry: bool) -> Result<Scalar, OracleError> {
        if self.limits.fuel == 0 {
            return Err(semantic_tool("INVALID_FUEL", "fuel must be positive"));
        }
        admit(self.net, self.program)?;
        let main = self
            .program
            .functions
            .iter()
            .position(|f| f.name == name)
            .ok_or_else(|| semantic_tool("ENTRYPOINT_MISSING", "CPU oracle entry missing"))?;
        if main_entry
            && (!self.program.functions[main].params.is_empty()
                || self.program.functions[main].result != Type::I32)
        {
            return Err(semantic_tool(
                "ENTRYPOINT_SIGNATURE",
                "CPU oracle requires fn main() -> i32",
            ));
        }
        let mut frames = Vec::new();
        frames
            .try_reserve(MAX_CALL_DEPTH)
            .map_err(|_| tool(ToolFailure::Allocation, "frame allocation failed"))?;
        let params = &self.program.functions[main].params;
        if params.len() != args.len() {
            return Err(semantic_tool(
                "ENTRYPOINT_SIGNATURE",
                "oracle entry arity mismatch",
            ));
        }
        let mut imported = Vec::new();
        for (i, (param, arg)) in params.iter().zip(args).enumerate() {
            let value = if let Some((leader, value)) = args[..i]
                .iter()
                .zip(&imported)
                .enumerate()
                .find_map(|(j, (old, value))| external_alias(old, arg).then_some((j, *value)))
            {
                if params[leader].ty != param.ty
                    || !matches!(param.mode, Some(Mode::Read | Mode::Mut))
                    || matches!(param.ty, Type::Named(_)) && param.mode != Some(Mode::Read)
                {
                    return Err(semantic_tool(
                        "ENTRYPOINT_SIGNATURE",
                        "oracle alias requires matching types and borrowed modes",
                    ));
                }
                if param.mode == Some(Mode::Mut)
                    || params[..i]
                        .iter()
                        .zip(&args[..i])
                        .any(|(p, v)| p.mode == Some(Mode::Mut) && external_alias(v, arg))
                {
                    return Err(semantic_tool("HOST_ARRAY_ALIAS", "mutable host alias"));
                }
                value
            } else {
                match (&param.ty, arg) {
                    (Type::ArrayI32, Value::ArrayI32(_))
                    | (Type::ArrayU32, Value::ArrayU32(_))
                    | (Type::ArrayF32, Value::ArrayF32(_))
                        if matches!(param.mode, Some(Mode::Read | Mode::Mut)) =>
                    {
                        self.arrays.import(arg)?
                    }
                    (Type::Named(name), Value::Record(fields))
                        if param.mode == Some(Mode::Read) =>
                    {
                        let definition = self
                            .program
                            .records
                            .iter()
                            .find(|r| r.name == *name)
                            .ok_or_else(|| {
                                semantic_tool(
                                    "ENTRYPOINT_SIGNATURE",
                                    "oracle entry requires a flat record",
                                )
                            })?;
                        let fields = fields.try_borrow().map_err(|_| internal())?;
                        if fields.len() != definition.fields.len() {
                            return Err(semantic_tool(
                                "ENTRYPOINT_SIGNATURE",
                                "oracle record field count mismatch",
                            ));
                        }
                        let fields = definition
                            .fields
                            .iter()
                            .zip(fields.iter())
                            .map(|(f, v)| floating::argument(&f.ty, v))
                            .collect::<Result<Vec<_>, _>>()?;
                        self.aggregates.import_record(fields, &mut self.arrays)?
                    }
                    _ => floating::argument(&param.ty, arg)?,
                }
            };
            imported.push(value);
        }
        self.external = imported.clone();
        frames.push(Frame::new(self.program, main, imported, 0)?);
        self.max_frames = 1;
        let outcome = self.drive(&mut frames);
        outcome.map_err(|error| frame_helper_trap(error, &frames, self.program))
    }

    fn drive(&mut self, frames: &mut Vec<Frame>) -> Result<Scalar, OracleError> {
        let mut next_frame = 1u64;
        loop {
            if self.transitions == self.limits.transitions {
                return Err(tool(
                    ToolFailure::TransitionLimit,
                    "physical oracle transition budget exhausted",
                ));
            }
            self.transitions += 1;
            let depth = frames.len();
            let frame = frames.last_mut().ok_or_else(internal)?;
            let function = self
                .program
                .functions
                .get(frame.function)
                .ok_or_else(internal)?;
            let block = function.blocks.get(frame.block).ok_or_else(internal)?;
            if let Some(instruction) = block.instructions.get(frame.position) {
                let index = *self
                    .agent_index
                    .get(&(frame.function, frame.block, frame.position))
                    .ok_or_else(internal)?;
                let agent = self.net.agents.get(index).ok_or_else(internal)?;
                if agent.rule == Rule::Tick {
                    if self.steps == self.limits.fuel {
                        return Err(EvalError::EvaluationLimit {
                            fuel_limit: self.limits.fuel,
                            steps_used: self.steps,
                        }
                        .into());
                    }
                    self.steps += 1;
                    frame.position = agent.next.position;
                    continue;
                }
                let event = Event {
                    kind: EventKind::Demand,
                    frame: frame.id,
                    activation: frame.activation,
                    token: agent.token,
                    rule: agent.rule,
                };
                self.emit(event)?;
                if let Some(CallTarget::Function(target)) = &agent.call {
                    if depth == MAX_CALL_DEPTH {
                        return Err(semantic_tool(
                            "CALL_DEPTH_LIMIT",
                            "CPU oracle call depth limit exceeded",
                        ));
                    }
                    let args = agent
                        .inputs
                        .iter()
                        .map(|id| frame.get(*id))
                        .collect::<Result<Vec<_>, _>>()?;
                    frame.waiting = Some((agent.output.as_ref().map(|(id, _)| *id), event));
                    let child = Frame::new(self.program, *target, args, next_frame)?;
                    next_frame = next_frame.checked_add(1).ok_or_else(internal)?;
                    frames.push(child);
                    self.max_frames = self.max_frames.max(frames.len());
                    continue;
                }
                let result = operation(
                    &instruction.kind,
                    frame,
                    instruction.output.as_ref().map(|(_, ty)| ty),
                    instruction.span,
                    self.program,
                    &mut self.aggregates,
                    &mut self.arrays,
                    &mut self.cleanup,
                )?;
                match (&agent.output, result) {
                    (Some((id, _)), Some(value)) => {
                        frame.values.insert(*id, value);
                    }
                    (None, None) => {}
                    _ => return Err(internal()),
                }
                self.emit(Event {
                    kind: EventKind::Complete,
                    ..event
                })?;
                frame.position = agent.next.position;
            } else {
                let index = *self
                    .control_index
                    .get(&(frame.function, frame.block))
                    .ok_or_else(internal)?;
                let control = self.net.control.get(index).ok_or_else(internal)?;
                let event = Event {
                    kind: EventKind::Demand,
                    frame: frame.id,
                    activation: frame.activation,
                    token: control.token,
                    rule: control.rule,
                };
                self.emit(event)?;
                let (target, args) = match &control.route {
                    Route::Goto { target, args } => (
                        target.block,
                        args.iter()
                            .map(|id| frame.get(*id))
                            .collect::<Result<Vec<_>, _>>()?,
                    ),
                    Route::Branch { condition, yes, no } => {
                        let Scalar::Bool(condition) = frame.get(*condition)? else {
                            return Err(internal());
                        };
                        (if condition { yes.block } else { no.block }, Vec::new())
                    }
                    Route::Return { value, cleanup } => {
                        let value = value
                            .map(|id| frame.get(id))
                            .transpose()?
                            .unwrap_or(Scalar::Unit);
                        aggregates::cleanup(
                            frame,
                            cleanup,
                            &mut self.arrays,
                            &mut self.aggregates,
                            &mut self.cleanup,
                        )?;
                        if let Some((inputs, dims)) = frame.tc_inputs {
                            self.arrays.finish_tc(inputs, dims, value)?;
                        }
                        self.emit(Event {
                            kind: EventKind::Complete,
                            ..event
                        })?;
                        frames.pop();
                        if let Some(parent) = frames.last_mut() {
                            let (output, call_event) =
                                parent.waiting.take().ok_or_else(internal)?;
                            if let Some(output) = output {
                                parent.values.insert(output, value);
                            }
                            self.emit(Event {
                                kind: EventKind::Complete,
                                ..call_event
                            })?;
                            parent.position += 1;
                            continue;
                        }
                        return Ok(value);
                    }
                };
                let next = function.blocks.get(target).ok_or_else(internal)?;
                if next.params.len() != args.len() {
                    return Err(internal());
                }
                // Resolve all incoming values before replacing any phi value.
                for ((id, _), value) in next.params.iter().zip(args) {
                    frame.values.insert(*id, value);
                }
                self.emit(Event {
                    kind: EventKind::Complete,
                    ..event
                })?;
                frame.block = target;
                frame.position = 0;
                frame.activation = frame.activation.checked_add(1).ok_or_else(internal)?;
            }
        }
    }
}

/// Checked frame helpers (`$vec_*`, `$xf_*`) carry a synthetic span; a source
/// trap inside one is reported at its call site, and an enclosing helper's call
/// site wins, exactly as each evaluator remaps at every helper call boundary.
fn frame_helper_trap(error: OracleError, frames: &[Frame], program: &mir::Program) -> OracleError {
    let OracleError::Semantic(EvalError::Trap { category, mut span }) = error else {
        return error;
    };
    for pair in frames.windows(2).rev() {
        let [caller, callee] = pair else { continue };
        let helper = program
            .functions
            .get(callee.function)
            .is_some_and(|f| crate::check::is_frame_helper(&f.name));
        let call = program
            .functions
            .get(caller.function)
            .and_then(|f| f.blocks.get(caller.block))
            .and_then(|b| b.instructions.get(caller.position));
        if let (true, Some(call)) = (helper, call) {
            span = call.span;
        }
    }
    EvalError::Trap { category, span }.into()
}

// Shared verified-program/storage contexts match the aggregate dispatch hook.
#[allow(clippy::too_many_arguments)]
fn operation(
    op: &mir::Operation,
    frame: &mut Frame,
    ty: Option<&Type>,
    span: Span,
    program: &mir::Program,
    aggregates: &mut aggregates::Arena,
    arena: &mut arrays::Arena,
    cleanup: &mut Vec<eval::CleanupEvent>,
) -> Result<Option<Scalar>, OracleError> {
    use mir::Operation as O;
    Ok(match op {
        O::F32(x) => Some(Scalar::F32(*x)),
        O::I32(x) => Some(Scalar::I32(*x)),
        O::U32(x) => Some(Scalar::U32(*x)),
        O::Bool(x) => Some(Scalar::Bool(*x)),
        O::Read(name) => Some(
            frame
                .places
                .get(name)
                .and_then(|slot| slot.value)
                .ok_or_else(internal)?,
        ),
        O::Bind { name, value } | O::Store { name, value } => {
            let value = frame.get(*value)?;
            aggregates::cleanup(
                frame,
                std::slice::from_ref(name),
                arena,
                aggregates,
                cleanup,
            )?;
            frame.places.insert(
                name.clone(),
                Slot {
                    value: Some(value),
                    owned: true,
                },
            );
            None
        }
        O::Drop(names) => {
            aggregates::cleanup(frame, names, arena, aggregates, cleanup)?;
            None
        }
        O::ForNext(value) => {
            let Scalar::U32(value) = frame.get(*value)? else {
                return Err(internal());
            };
            Some(Scalar::U32(value.checked_add(1).ok_or_else(internal)?))
        }
        O::Unary {
            op,
            value,
            operator,
        } => Some(match (op, frame.get(*value)?) {
            (UnaryOp::Neg, Scalar::F32(x)) => Scalar::F32(x.negate()),
            (UnaryOp::Not, Scalar::Bool(x)) => Scalar::Bool(!x),
            (UnaryOp::Neg, Scalar::I32(x)) => {
                Scalar::I32(x.checked_neg().ok_or(EvalError::Trap {
                    category: TrapCategory::IntegerOverflow,
                    span: *operator,
                })?)
            }
            _ => return Err(internal()),
        }),
        O::Binary {
            op,
            left,
            right,
            operator,
        } => Some(binary(
            *op,
            frame.get(*left)?,
            frame.get(*right)?,
            *operator,
        )?),
        O::Call { name, args } => {
            if let Some(kind) = crate::fsum::Kind::from_name(name) {
                return arrays::reduce(kind, args, frame, span, arena).map(Some);
            }
            if crate::check::sort_modes(name).is_some() {
                return arrays::sort(name, args, frame, span, arena).map(Some);
            }
            if let Some(rule) = units::rule(name) {
                let [value] = args.as_slice() else {
                    return Err(internal());
                };
                return units::call(rule, frame.get(*value)?, span).map(Some);
            }
            if aggregates::is_text_intrinsic(name) {
                return aggregates::text_intrinsic(name, args, frame, span, arena, aggregates)
                    .map(Some);
            }
            if arrays::is_view_intrinsic(name) {
                return arrays::make_view(name, args, frame, span, arena).map(Some);
            }
            if floating::supports_intrinsic(name) {
                return floating::call(name, args, frame).map(Some);
            }
            if matches!(name.as_str(), "filled_i32" | "filled_u32" | "filled_f32") {
                return arrays::filled(name, args, frame, arena).map(Some);
            }
            let [left, right] = args.as_slice() else {
                return Err(internal());
            };
            let (Scalar::U32(a), Scalar::U32(b)) = (frame.get(*left)?, frame.get(*right)?) else {
                return Err(internal());
            };
            Some(Scalar::U32(match name.as_str() {
                "wrapping_add" => a.wrapping_add(b),
                "wrapping_mul" => a.wrapping_mul(b),
                _ => return Err(internal()),
            }))
        }
        O::Tick | O::F64(_) => return Err(internal()),
        O::Length(name)
            if frame
                .places
                .get(name)
                .is_some_and(|slot| matches!(slot.value, Some(Scalar::Aggregate(_)))) =>
        {
            return aggregates::operation(op, frame, ty, span, program, arena, aggregates, cleanup);
        }
        O::Index { array, .. } if matches!(frame.get(*array)?, Scalar::Aggregate(_)) => {
            return aggregates::operation(op, frame, ty, span, program, arena, aggregates, cleanup);
        }
        O::Blob { .. }
        | O::MoveField { .. }
        | O::RecordRef { .. }
        | O::Record { .. }
        | O::Field { .. }
        | O::Variant { .. }
        | O::VariantTag(_)
        | O::VariantPayload { .. }
        | O::VariantTakePayload { .. }
        | O::DropValue(_) => {
            return aggregates::operation(op, frame, ty, span, program, arena, aggregates, cleanup);
        }
        O::Move(_)
        | O::ArrayRef { .. }
        | O::Array(_)
        | O::Index { .. }
        | O::Length(_)
        | O::StoreIndex { .. } => return arrays::operation(op, frame, ty, span, arena),
    })
}

fn binary(op: BinaryOp, left: Scalar, right: Scalar, span: Span) -> Result<Scalar, OracleError> {
    use BinaryOp as B;
    let overflow = || EvalError::Trap {
        category: TrapCategory::IntegerOverflow,
        span,
    };
    let zero = || EvalError::Trap {
        category: TrapCategory::DivisionByZero,
        span,
    };
    macro_rules! integer {
        ($a:expr, $b:expr, $variant:ident, $remainder:expr) => {{
            let (a, b) = ($a, $b);
            match op {
                B::Eq => Scalar::Bool(a == b),
                B::Ne => Scalar::Bool(a != b),
                B::Lt => Scalar::Bool(a < b),
                B::Le => Scalar::Bool(a <= b),
                B::Gt => Scalar::Bool(a > b),
                B::Ge => Scalar::Bool(a >= b),
                B::Add => Scalar::$variant(a.checked_add(b).ok_or_else(overflow)?),
                B::Sub => Scalar::$variant(a.checked_sub(b).ok_or_else(overflow)?),
                B::Mul => Scalar::$variant(a.checked_mul(b).ok_or_else(overflow)?),
                B::Div => {
                    if b == 0 {
                        return Err(zero().into());
                    }
                    Scalar::$variant(a.checked_div(b).ok_or_else(overflow)?)
                }
                B::Rem => {
                    if b == 0 {
                        return Err(zero().into());
                    }
                    Scalar::$variant($remainder)
                }
                B::And | B::Or => return Err(internal()),
            }
        }};
    }
    Ok(match (left, right) {
        (Scalar::F32(a), Scalar::F32(b)) => return floating::binary(op, a, b),
        (Scalar::I32(a), Scalar::I32(b)) => integer!(
            a,
            b,
            I32,
            if a == i32::MIN && b == -1 {
                0
            } else {
                a.checked_rem(b).ok_or_else(overflow)?
            }
        ),
        (Scalar::U32(a), Scalar::U32(b)) => {
            integer!(a, b, U32, a.checked_rem(b).ok_or_else(overflow)?)
        }
        (Scalar::Bool(a), Scalar::Bool(b)) => Scalar::Bool(match op {
            B::Eq => a == b,
            B::Ne => a != b,
            B::And => a && b,
            B::Or => a || b,
            _ => return Err(internal()),
        }),
        _ => return Err(internal()),
    })
}

/// Execute only a current verified candidate. A source trap leaves the final
/// Demand unmatched and emits no following Complete/cleanup. Trace/transition
/// exhaustion is a tool limit, never source divergence or a source trap.
pub fn observe(net: &Net, proof: &Proof, limits: Limits) -> Result<Observation, super::Error> {
    observe_entry(net, proof, "main", &[], true, limits)
}

/// Named scalar entry has no artificial call-expression or function-entry fuel.
pub fn observe_call(
    net: &Net,
    proof: &Proof,
    name: &str,
    args: &[Value],
    limits: Limits,
) -> Result<Observation, super::Error> {
    observe_entry(net, proof, name, args, false, limits)
}

fn observe_entry(
    net: &Net,
    proof: &Proof,
    name: &str,
    args: &[Value],
    main_entry: bool,
    limits: Limits,
) -> Result<Observation, super::Error> {
    let program = net.compiled_mir(proof)?;
    let mut machine = Machine {
        external: Vec::new(),
        net,
        program: &program,
        limits,
        steps: 0,
        transitions: 0,
        max_frames: 0,
        events: Vec::new(),
        arrays: arrays::Arena::new(limits.array_elements),
        aggregates: aggregates::Arena::new(),
        cleanup: Vec::new(),
        agent_index: net
            .agents
            .iter()
            .enumerate()
            .map(|(index, agent)| {
                (
                    (
                        agent.token.block.function,
                        agent.token.block.block,
                        agent.token.position,
                    ),
                    index,
                )
            })
            .collect(),
        control_index: net
            .control
            .iter()
            .enumerate()
            .map(|(index, control)| {
                (
                    (control.token.block.function, control.token.block.block),
                    index,
                )
            })
            .collect(),
    };
    let outcome = machine
        .run(name, args, main_entry)
        .and_then(|value| value.value(&machine.arrays, &machine.aggregates))
        .and_then(|result| {
            let function = program.function(name).ok_or_else(internal)?;
            let mut commits = Vec::new();
            for ((param, arg), value) in function.params.iter().zip(args).zip(&machine.external) {
                if param.mode != Some(Mode::Mut) {
                    continue;
                }
                let staged = value.value(&machine.arrays, &machine.aggregates)?;
                commits.push(OracleCommit::prepare(arg, staged)?);
            }
            for commit in commits {
                commit.apply();
            }
            Ok(result)
        });
    let (result, failure) = match outcome {
        Ok(value) => (Ok(value), None),
        Err(OracleError::Semantic(error)) => (Err(error), None),
        Err(OracleError::Tool { kind, message }) => {
            let code = match kind {
                ToolFailure::Unsupported => "UNSUPPORTED_FEATURE",
                ToolFailure::TransitionLimit
                | ToolFailure::TraceLimit
                | ToolFailure::Allocation => "RESOURCE_LIMIT",
                ToolFailure::InvalidState => "MIR_INVALID",
            };
            (Err(EvalError::Tool { code, message }), Some(kind))
        }
    };
    Ok(Observation {
        semantic: eval::Observation {
            result,
            steps_used: machine.steps,
            cleanup_trace: machine.cleanup,
        },
        failure,
        events: machine.events,
        transitions: machine.transitions,
        max_frames: machine.max_frames,
    })
}

pub(super) fn external_alias(a: &Value, b: &Value) -> bool {
    use std::rc::Rc;
    match (a, b) {
        (Value::ArrayI32(a), Value::ArrayI32(b)) => Rc::ptr_eq(a, b),
        (Value::ArrayU32(a), Value::ArrayU32(b)) => Rc::ptr_eq(a, b),
        (Value::ArrayF32(a), Value::ArrayF32(b)) => Rc::ptr_eq(a, b),
        (Value::Record(a), Value::Record(b)) => Rc::ptr_eq(a, b),
        _ => false,
    }
}
enum OracleCommit<'a> {
    I32(std::cell::RefMut<'a, Vec<i32>>, Vec<i32>),
    U32(std::cell::RefMut<'a, Vec<u32>>, Vec<u32>),
    F32(
        std::cell::RefMut<'a, Vec<crate::numeric_contract::F32Bits>>,
        Vec<crate::numeric_contract::F32Bits>,
    ),
}
impl<'a> OracleCommit<'a> {
    fn prepare(to: &'a Value, from: Value) -> Result<Self, OracleError> {
        Ok(match (to, from) {
            (Value::ArrayI32(a), Value::ArrayI32(b)) => Self::I32(
                a.try_borrow_mut().map_err(|_| internal())?,
                b.borrow().clone(),
            ),
            (Value::ArrayU32(a), Value::ArrayU32(b)) => Self::U32(
                a.try_borrow_mut().map_err(|_| internal())?,
                b.borrow().clone(),
            ),
            (Value::ArrayF32(a), Value::ArrayF32(b)) => Self::F32(
                a.try_borrow_mut().map_err(|_| internal())?,
                b.borrow().clone(),
            ),
            _ => return Err(internal()),
        })
    }
    fn apply(self) {
        match self {
            Self::I32(mut a, b) => *a = b,
            Self::U32(mut a, b) => *a = b,
            Self::F32(mut a, b) => *a = b,
        }
    }
}
