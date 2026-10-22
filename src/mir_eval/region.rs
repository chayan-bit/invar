//! One typed publication contract for a completed device region.
//!
//! An adapter that finished a reached loop nest off the interpreter (a device
//! kernel today, a law-directed parallel form later) states the nest's whole
//! source effect as a [`CompletedRegion`]. MIR re-derives everything it can
//! from the verified CFG and the live frame: the reached head, flow-conserving
//! branch counts, exact source fuel from block ticks, the written owner and
//! scalar sets, owner identity, aliasing, bounds, counts, cleanup and the trap
//! site. Validation is pure; publication starts only after every check passes
//! and cannot fail part way. A refusal publishes nothing and leaves the loop to
//! serial MIR, which is exact because the device never touched live owners.
use std::collections::{HashMap, HashSet};
use std::ops::Range;

use super::*;
use crate::numeric_contract::F32Bits;

/// Elements in the written owner's own layout.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "constructed by the A1 family adapters")
)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Elements {
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<F32Bits>),
}

impl Elements {
    fn len(&self) -> usize {
        match self {
            Self::U32(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::F32(v) => v.len(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Scalar {
    U32(u32),
    I32(i32),
    F32(F32Bits),
    Bool(bool),
}

impl Scalar {
    fn of(value: &Value) -> Option<Self> {
        match value {
            Value::U32(v) => Some(Self::U32(*v)),
            Value::I32(v) => Some(Self::I32(*v)),
            Value::F32(v) => Some(Self::F32(*v)),
            Value::Bool(v) => Some(Self::Bool(*v)),
            _ => None,
        }
    }
    fn value(self) -> Value {
        match self {
            Self::U32(v) => Value::U32(v),
            Self::I32(v) => Value::I32(v),
            Self::F32(v) => Value::F32(v),
            Self::Bool(v) => Value::Bool(v),
        }
    }
}

/// Outcome counts of one MIR branch inside the region, keyed by source span:
/// the outer loop head, nested loop heads, `if`s and short-circuit operators.
/// An omitted branch claims zero outcomes; flow conservation catches a lie.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Branch {
    pub span: Span,
    pub taken: u64,
    pub skipped: u64,
}

/// Final elements `range` of one existing live owner of `length` elements.
/// Elements outside `range` are untouched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Write {
    pub owner: String,
    pub length: usize,
    pub range: Range<usize>,
    pub values: Elements,
}

/// A pre-existing scalar place the region assigns: its seed and final value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ScalarState {
    pub name: String,
    pub before: Scalar,
    pub after: Scalar,
}

/// The first source-order trap (observation contract v2). The writes, scalars,
/// cleanup and fuel then describe the source prefix ending at that trap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FirstTrap {
    pub category: TrapCategory,
    pub span: Span,
}

/// Everything a completed outer loop at its reached head did to the source
/// frame. Nested loops are part of the region: publication skips them.
/// Sequential loops (compaction's count and copy passes) are one region each,
/// published at their own heads; MIR runs the code between them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CompletedRegion {
    pub function: String,
    /// The outer loop this region replaces, as reached with a fresh induction.
    pub span: Span,
    pub branches: Vec<Branch>,
    pub writes: Vec<Write>,
    pub scalars: Vec<ScalarState>,
    /// Owners allocated and dropped inside the region, in source order.
    pub cleanup: Vec<(String, usize)>,
    pub fuel: u64,
    pub trap: Option<FirstTrap>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// Another function or loop, or not a fresh live entry of this loop.
    Head,
    /// The region's CFG has a shape this contract cannot describe.
    Shape,
    /// A branch claim names no branch of this region, or one twice.
    Span,
    /// Branch counts violate flow conservation or the live trip count.
    Flow,
    /// Claimed fuel differs from block ticks under the claimed counts.
    Fuel,
    /// Fuel or live allocations exceed what remains.
    Budget,
    /// Written owner set, identity, layout or allocation length mismatch.
    Owner,
    /// A written owner shares storage with another live place or write.
    Alias,
    Bounds,
    /// Values do not fill exactly the claimed range.
    Count,
    Scalar,
    Cleanup,
    Trap,
}

/// Static facts of the loop nest under one head.
struct Shape {
    head: usize,
    /// Head first, then every block reachable from its body without the head.
    blocks: Vec<usize>,
    bound: HashSet<String>,
    stored: HashSet<String>,
    assigned: HashSet<String>,
}

fn shape(program: &mir::Program, function: &Function, head: usize) -> Result<Shape, Refusal> {
    let Some(Terminator::Branch { yes, no, .. }) = function.blocks[head].end else {
        return Err(Refusal::Head);
    };
    let mut blocks = vec![head];
    let mut seen = HashSet::from([head]);
    let mut queue = vec![yes];
    while let Some(id) = queue.pop() {
        if !seen.insert(id) {
            continue;
        }
        if id == no {
            return Err(Refusal::Shape); // the exit is reachable around the head
        }
        blocks.push(id);
        match &function.blocks.get(id).ok_or(Refusal::Shape)?.end {
            Some(Terminator::Goto { target, .. }) => queue.push(*target),
            Some(Terminator::Branch { yes, no, .. }) => queue.extend([*yes, *no]),
            _ => return Err(Refusal::Shape), // early return or unterminated
        }
    }
    let (mut bound, mut stored, mut assigned) = (HashSet::new(), HashSet::new(), HashSet::new());
    let mut dropped = HashSet::new();
    for instruction in blocks
        .iter()
        .flat_map(|&b| &function.blocks[b].instructions)
    {
        match &instruction.kind {
            Operation::Bind { name, .. } => {
                bound.insert(name.clone());
            }
            Operation::Store { name, .. } => {
                assigned.insert(name.clone());
            }
            Operation::StoreIndex { name, .. } => {
                stored.insert(name.clone());
            }
            Operation::Drop(names) => dropped.extend(names.iter().cloned()),
            Operation::Call { name, .. } if program.functions.iter().any(|f| f.name == *name) => {
                return Err(Refusal::Shape);
            }
            Operation::Move(_) | Operation::MoveField { .. } => return Err(Refusal::Shape),
            _ => {}
        }
    }
    if !dropped.is_subset(&bound) {
        return Err(Refusal::Shape); // an outer owner would leave the frame
    }
    // Live accounting stays exact only if every allocation is a named owner
    // whose drop the cleanup claim pays back; an unnamed temporary is refused.
    let instructions = || {
        blocks
            .iter()
            .flat_map(|&b| &function.blocks[b].instructions)
    };
    let named: HashSet<ValueId> = instructions()
        .filter_map(|i| match &i.kind {
            Operation::Bind { value, .. } => Some(*value),
            _ => None,
        })
        .collect();
    if instructions().any(|i| {
        matches!(
            i.kind,
            Operation::Array(_)
                | Operation::Record { .. }
                | Operation::Variant { .. }
                | Operation::Blob { .. }
                | Operation::Call { .. }
        ) && i
            .output
            .as_ref()
            .is_some_and(|(id, ty)| !scalar_type(ty) && !named.contains(id))
    }) {
        return Err(Refusal::Shape);
    }
    stored.retain(|name| !bound.contains(name));
    assigned.retain(|name| !bound.contains(name));
    Ok(Shape {
        head,
        blocks,
        bound,
        stored,
        assigned,
    })
}

fn ticks(instructions: &[mir::Instruction]) -> u64 {
    instructions
        .iter()
        .filter(|i| matches!(i.kind, Operation::Tick))
        .count() as u64
}

/// The unique instruction that can raise a trap at `span`.
fn trap_site(function: &Function, shape: &Shape, span: Span) -> Result<(usize, usize), Refusal> {
    let mut sites = shape.blocks.iter().flat_map(|&b| {
        function.blocks[b]
            .instructions
            .iter()
            .enumerate()
            .filter(move |(_, i)| match &i.kind {
                Operation::Binary { operator, .. } | Operation::Unary { operator, .. } => {
                    *operator == span
                }
                Operation::Index { .. } | Operation::StoreIndex { .. } | Operation::Call { .. } => {
                    i.span == span
                }
                _ => false,
            })
            .map(move |(at, _)| (b, at))
    });
    match (sites.next(), sites.next()) {
        (Some(site), None) => Ok(site),
        _ => Err(Refusal::Trap),
    }
}

/// Block executions implied by the claimed branch outcomes. Branch blocks take
/// their claim, goto blocks the sum of their incoming edges; every branch block
/// must then receive exactly the flow it claims (the head also gets its entry).
/// `trap` is the block whose last execution stopped part way and left no edge.
struct Flow<'f> {
    function: &'f Function,
    preds: HashMap<usize, Vec<usize>>,
    outcomes: HashMap<usize, (u64, u64)>,
    exec: HashMap<usize, u64>,
    trap: Option<usize>,
}

impl Flow<'_> {
    /// Executions that left `b` through an edge; a trapped block never runs
    /// zero times once `executions` returns.
    fn out(&self, b: usize) -> u64 {
        self.exec[&b].saturating_sub(u64::from(self.trap == Some(b)))
    }

    /// Edges into `b` from region blocks, or `None` while a goto source is unknown.
    fn inflow(&self, b: usize) -> Result<Option<u64>, Refusal> {
        let mut total = 0u64;
        for &p in self.preds.get(&b).into_iter().flatten() {
            let edge = match (&self.function.blocks[p].end, self.outcomes.get(&p)) {
                (Some(Terminator::Branch { yes, no, .. }), Some(&(taken, skipped))) => {
                    u64::from(*yes == b) * taken + u64::from(*no == b) * skipped
                }
                _ if self.exec.contains_key(&p) => self.out(p),
                _ => return Ok(None),
            };
            total = total.checked_add(edge).ok_or(Refusal::Flow)?;
        }
        Ok(Some(total))
    }
}

fn executions<'f>(
    function: &'f Function,
    shape: &Shape,
    claims: &HashMap<Span, (u64, u64)>,
    trap: Option<usize>,
) -> Result<Flow<'f>, Refusal> {
    let inside: HashSet<usize> = shape.blocks.iter().copied().collect();
    let mut flow = Flow {
        function,
        preds: HashMap::new(),
        outcomes: HashMap::new(),
        exec: HashMap::new(),
        trap,
    };
    let mut spans = HashSet::new();
    for &b in &shape.blocks {
        let targets = match &function.blocks[b].end {
            Some(Terminator::Goto { target, .. }) => vec![*target],
            Some(Terminator::Branch { yes, no, span, .. }) => {
                if !spans.insert(*span) {
                    return Err(Refusal::Span); // two branches would share one claim
                }
                let (taken, skipped) = claims.get(span).copied().unwrap_or((0, 0));
                let total = taken
                    .checked_add(skipped)
                    .and_then(|n| n.checked_add(u64::from(trap == Some(b))))
                    .ok_or(Refusal::Flow)?;
                flow.outcomes.insert(b, (taken, skipped));
                flow.exec.insert(b, total);
                if yes == no {
                    vec![*yes]
                } else {
                    vec![*yes, *no]
                }
            }
            _ => return Err(Refusal::Shape),
        };
        for target in targets.into_iter().filter(|t| inside.contains(t)) {
            flow.preds.entry(target).or_default().push(b);
        }
    }
    if claims.keys().any(|span| !spans.contains(span)) {
        return Err(Refusal::Span);
    }
    // Every cycle of the structured CFG passes a branch block, so the goto
    // blocks settle; a round without progress is a cycle of gotos.
    let mut pending: Vec<usize> = shape
        .blocks
        .iter()
        .copied()
        .filter(|b| !flow.exec.contains_key(b))
        .collect();
    while !pending.is_empty() {
        let mut unresolved = Vec::new();
        for &b in &pending {
            match flow.inflow(b)? {
                Some(n) => {
                    flow.exec.insert(b, n);
                }
                None => unresolved.push(b),
            }
        }
        if unresolved.len() == pending.len() {
            return Err(Refusal::Shape);
        }
        pending = unresolved;
    }
    for &b in flow.outcomes.keys() {
        let incoming = flow
            .inflow(b)?
            .and_then(|n| n.checked_add(u64::from(b == shape.head)))
            .ok_or(Refusal::Flow)?;
        if incoming != flow.exec[&b] {
            return Err(Refusal::Flow);
        }
    }
    if trap.is_some_and(|t| flow.exec[&t] == 0) {
        return Err(Refusal::Trap);
    }
    Ok(flow)
}

fn scalar_type(ty: &Type) -> bool {
    matches!(
        ty,
        Type::I32 | Type::U32 | Type::F32 | Type::Bool | Type::Unit
    )
}

fn is_heap(function: &Function, name: &str) -> bool {
    function
        .locals
        .get(name)
        .is_some_and(|local| !scalar_type(&local.ty))
}

/// Address of the array storage behind a place, views included.
fn storage(value: &Value) -> Option<*const ()> {
    match value {
        Value::ArrayU32(a) | Value::ViewU32(a, _) => Some(Rc::as_ptr(a).cast()),
        Value::ArrayI32(a) | Value::ViewI32(a, _) => Some(Rc::as_ptr(a).cast()),
        Value::ArrayF32(a) => Some(Rc::as_ptr(a).cast()),
        _ => None,
    }
}

/// A live owner handle, taken during validation so publication cannot fail.
enum Target {
    U32(Rc<RefCell<Vec<u32>>>),
    I32(Rc<RefCell<Vec<i32>>>),
    F32(Rc<RefCell<Vec<F32Bits>>>),
}

pub(super) struct Validated {
    fuel: u64,
    targets: Vec<Target>,
}

fn target(value: &Value, values: &Elements) -> Option<(Target, usize)> {
    Some(match (value, values) {
        (Value::ArrayU32(a), Elements::U32(_)) => {
            (Target::U32(a.clone()), a.try_borrow_mut().ok()?.len())
        }
        (Value::ArrayI32(a), Elements::I32(_)) => {
            (Target::I32(a.clone()), a.try_borrow_mut().ok()?.len())
        }
        (Value::ArrayF32(a), Elements::F32(_)) => {
            (Target::F32(a.clone()), a.try_borrow_mut().ok()?.len())
        }
        _ => return None,
    })
}

/// Each name once, and exactly the names the region's code can change.
fn same_names<'a>(
    claimed: impl Iterator<Item = &'a String>,
    expected: &HashSet<String>,
) -> Option<()> {
    let mut seen = HashSet::new();
    for name in claimed {
        if !seen.insert(name) {
            return None;
        }
    }
    (seen.len() == expected.len() && seen.iter().all(|name| expected.contains(*name))).then_some(())
}

impl Interpreter<'_> {
    pub(super) fn validate_region(
        &self,
        function: &Function,
        head: &mir::BasicBlock,
        condition: ValueId,
        frame: &Frame,
        region: &CompletedRegion,
    ) -> Result<Validated, Refusal> {
        let head_id = function
            .blocks
            .iter()
            .position(|block| std::ptr::eq(block, head))
            .ok_or(Refusal::Head)?;
        let trips = super::bulk::live_trips(head, condition, frame).ok_or(Refusal::Head)?;
        if region.function != function.name
            || !matches!(&head.end, Some(Terminator::Branch { span, .. }) if *span == region.span)
        {
            return Err(Refusal::Head);
        }
        let shape = shape(self.program, function, head_id)?;
        if shape
            .bound
            .iter()
            .any(|name| frame.slots.contains_key(name))
        {
            return Err(Refusal::Shape); // a region binding would shadow a live place
        }
        let mut claims = HashMap::new();
        for branch in &region.branches {
            if claims
                .insert(branch.span, (branch.taken, branch.skipped))
                .is_some()
            {
                return Err(Refusal::Span);
            }
        }
        let site = region
            .trap
            .map(|trap| trap_site(function, &shape, trap.span))
            .transpose()?;
        let flow = executions(function, &shape, &claims, site.map(|(b, _)| b))?;
        let (taken, skipped) = claims.get(&region.span).copied().unwrap_or((0, 0));
        let complete = taken == u64::from(trips) && skipped == 1;
        let trapped = taken <= u64::from(trips) && skipped == 0;
        if !(if site.is_some() { trapped } else { complete }) {
            return Err(Refusal::Flow);
        }

        // Fuel and cleanup follow from the executions of every instruction.
        let mut fuel = 0u64;
        let mut drops: HashMap<&str, u64> = HashMap::new();
        for &b in &shape.blocks {
            let block = &function.blocks[b];
            let out = flow.out(b);
            let partial = site.filter(|(t, _)| *t == b).map(|(_, at)| at);
            fuel = ticks(&block.instructions)
                .checked_mul(out)
                .and_then(|n| n.checked_add(fuel))
                .and_then(|n| {
                    n.checked_add(partial.map_or(0, |at| ticks(&block.instructions[..at])))
                })
                .ok_or(Refusal::Fuel)?;
            for (at, instruction) in block.instructions.iter().enumerate() {
                let times = out + u64::from(partial.is_some_and(|stop| at < stop));
                let names: &[String] = match &instruction.kind {
                    Operation::Drop(names) => names,
                    Operation::Store { name, .. } => std::slice::from_ref(name),
                    _ => &[],
                };
                for name in names
                    .iter()
                    .filter(|n| shape.bound.contains(*n) && is_heap(function, n))
                {
                    *drops.entry(name.as_str()).or_default() += times;
                }
            }
        }
        if fuel != region.fuel {
            return Err(Refusal::Fuel);
        }
        if fuel > self.fuel_limit.saturating_sub(self.steps_used) {
            return Err(Refusal::Budget);
        }
        let mut claimed: HashMap<&str, u64> = HashMap::new();
        let mut transient = 0usize;
        for (name, length) in &region.cleanup {
            *claimed.entry(name.as_str()).or_default() += 1;
            transient = transient.checked_add(*length).ok_or(Refusal::Budget)?;
        }
        drops.retain(|_, n| *n > 0);
        if claimed != drops {
            return Err(Refusal::Cleanup);
        }
        // ponytail: every transient counted live at once, an upper bound of the
        // source peak; track per-iteration peaks if this refuses real regions.
        if self.live_elements.saturating_add(transient) > crate::eval::MAX_LIVE_ELEMENTS {
            return Err(Refusal::Budget);
        }

        same_names(region.scalars.iter().map(|s| &s.name), &shape.assigned)
            .ok_or(Refusal::Scalar)?;
        for state in &region.scalars {
            let live = frame.slot(&state.name).ok().and_then(Scalar::of);
            if live != Some(state.before)
                || std::mem::discriminant(&state.before) != std::mem::discriminant(&state.after)
            {
                return Err(Refusal::Scalar);
            }
        }

        let mut owners = HashSet::new();
        if !region.writes.iter().all(|w| owners.insert(&w.owner)) {
            return Err(Refusal::Alias);
        }
        same_names(region.writes.iter().map(|w| &w.owner), &shape.stored).ok_or(Refusal::Owner)?;
        let mut targets = Vec::new();
        for write in &region.writes {
            let slot = frame.slot(&write.owner).map_err(|_| Refusal::Owner)?;
            let (handle, length) = target(slot, &write.values).ok_or(Refusal::Owner)?;
            if length != write.length {
                return Err(Refusal::Owner);
            }
            if write.range.start > write.range.end || write.range.end > length {
                return Err(Refusal::Bounds);
            }
            if write.values.len() != write.range.len() {
                return Err(Refusal::Count);
            }
            let address = storage(slot);
            if frame.slots.iter().any(|(name, other)| {
                *name != write.owner && other.value.as_ref().and_then(storage) == address
            }) {
                return Err(Refusal::Alias);
            }
            targets.push(handle);
        }
        Ok(Validated { fuel, targets })
    }

    /// Infallible after validation: every place, handle and count is checked.
    pub(super) fn publish_region(
        &mut self,
        frame: &mut Frame,
        region: &CompletedRegion,
        validated: Validated,
    ) -> Result<bool, EvalError> {
        self.steps_used += validated.fuel;
        for (target, write) in validated.targets.iter().zip(&region.writes) {
            let range = write.range.clone();
            match (target, &write.values) {
                (Target::U32(a), Elements::U32(v)) => a.borrow_mut()[range].copy_from_slice(v),
                (Target::I32(a), Elements::I32(v)) => a.borrow_mut()[range].copy_from_slice(v),
                (Target::F32(a), Elements::F32(v)) => a.borrow_mut()[range].copy_from_slice(v),
                _ => unreachable!("validated layout"),
            }
        }
        for state in &region.scalars {
            if let Some(slot) = frame.slots.get_mut(&state.name) {
                slot.value = Some(state.after.value());
            }
        }
        self.trace
            .extend(region.cleanup.iter().map(|(name, length)| CleanupEvent {
                name: name.clone(),
                length: *length,
            }));
        self.used_overrides += 1;
        match region.trap {
            Some(first) => Err(trap(first.category, first.span)),
            None => Ok(true),
        }
    }

    pub(super) fn apply_region(
        &mut self,
        function: &Function,
        head: &mir::BasicBlock,
        condition: ValueId,
        frame: &mut Frame,
        region: &CompletedRegion,
    ) -> Result<bool, EvalError> {
        match self.validate_region(function, head, condition, frame, region) {
            Ok(validated) => self.publish_region(frame, region, validated),
            Err(_) => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests;
