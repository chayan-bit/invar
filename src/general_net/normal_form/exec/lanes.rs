//! Law execution of a parallel loop: lanes, symbolic elements, tree scans,
//! per-lane replay from scanned prefixes, array models and ordered merge.

use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
};

use super::{
    Machine, Mutation, Stop,
    interp::{Frame, read_base},
    rep, sym, table,
    value::{Array, Elems, Sym, Val, array_id},
};
use crate::{
    eval::TrapCategory,
    front::Span,
    general_net::{
        normal_form::{ArrayLaw, Guard, LawId, LoopForm, LoopNode, Rep},
        regions::{
            Counting, RegionId,
            tree::{self, Expr, Loop, LoopKind, Stmt},
        },
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Phase {
    /// Carried group symbolic: compute lane elements.
    Symbolic,
    /// Combine arrays read as zero: collect per-lane contributions.
    Contribute,
    /// Everything concrete from scanned prefixes: the lane's real execution.
    Replay,
}

#[derive(Clone, Debug)]
pub(super) struct Group {
    pub rep: Rep,
    pub vars: Vec<String>,
    pub companions: Vec<String>,
}

/// How a lane sees an array the loop writes.
#[derive(Clone, Debug)]
pub(super) enum Model {
    /// L1: round-start value unless this lane wrote it.
    Own,
    /// L12, L13, L14 without reads: never read in the loop.
    WriteOnly,
    /// L11: per-target prefix of earlier lanes' contributions.
    /// The table is filled after the contribution phase.
    Combine { table: Option<Rc<CombineTable>> },
    /// L14 with reads: `old[w] == c || (active(w) && w < u)`.
    Monotone {
        value: Val,
        active: Rc<Vec<Option<bool>>>,
        lo: i128,
    },
    /// L19: the fused first loop's writes of the same lane.
    Fused { table: Rc<Vec<HashMap<usize, Val>>> },
}

pub(super) struct Lane {
    pub phase: Phase,
    pub position: usize,
    pub models: Rc<HashMap<usize, Model>>,
    pub overlay: HashMap<(usize, usize), Val>,
    pub writes: Vec<(usize, usize, Val)>,
    pub arrays: HashMap<usize, Array>,
    pub poisoned: HashSet<usize>,
    pub group: Option<Group>,
    pub select: Option<(Val, Vec<Val>)>,
    pub range: (i128, i128),
    pub capturing: bool,
    pub wrong: bool,
}

/// One replayed lane: its first event and its writes in program order.
pub(super) struct LaneOut {
    event: Option<Stop>,
    writes: Vec<(usize, usize, Val)>,
    arrays: HashMap<usize, Array>,
}

/// L11: round-start values and, per target index, the running value after
/// each contributing lane (ascending lane order).
#[derive(Debug)]
pub(super) struct CombineTable {
    base: Elems,
    history: HashMap<usize, Vec<(usize, Val)>>,
}

impl CombineTable {
    /// `target[index]` before lane `k`'s contribution.
    fn before(&self, k: usize, index: usize, wrong: bool) -> Option<Val> {
        let entries = self.history.get(&index).filter(|_| !wrong);
        let earlier = entries.map_or(0, |e| e.partition_point(|(j, _)| *j < k));
        match entries.and_then(|e| e.get(earlier.checked_sub(1)?)) {
            Some((_, value)) => Some(value.clone()),
            None => self.base.get(index),
        }
    }
}

/// Whether lane `k` starts from an unrepresentable carried prefix: an earlier
/// lane has the first event, so this lane's element is never used.
fn poisoned(prefixes: &HashMap<String, Vec<Val>>, k: usize) -> bool {
    prefixes
        .values()
        .any(|lanes| lanes.get(k).is_some_and(Val::is_sym))
}

/// L19: the first loop's lanes, published only with the fused second loop.
pub(super) struct Pending {
    region: RegionId,
    outs: Vec<LaneOut>,
    reverse: bool,
}

fn merge(outs: &[LaneOut], reverse: bool) -> Result<(), Stop> {
    let order: Vec<&LaneOut> = if reverse {
        outs.iter().rev().collect()
    } else {
        outs.iter().collect()
    };
    for out in order {
        for (id, index, value) in &out.writes {
            let array = out
                .arrays
                .get(id)
                .ok_or_else(|| Stop::internal("merge target"))?;
            array
                .borrow_mut()
                .set(*index, value)?
                .ok_or_else(|| Stop::internal("merge bounds"))?;
        }
    }
    Ok(())
}

/// Statements allowed between fused loops keep the first loop pending.
pub(super) fn keeps_pending(frame: &Frame, stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Let { value, .. } => match value {
            Expr::Move(_) => true,
            Expr::Call { name, .. } => name.starts_with("filled_"),
            other => other.literal().is_some(),
        },
        Stmt::Loop(inner) => frame
            .pending
            .as_ref()
            .is_some_and(|p| p.region.head == Some(inner.head)),
        _ => false,
    }
}

pub(super) fn flush(frame: &mut Frame) -> Result<(), Stop> {
    match frame.pending.take() {
        Some(pending) => merge(&pending.outs, pending.reverse),
        None => Ok(()),
    }
}

fn zero(elems: &Elems) -> Val {
    match elems {
        Elems::I32(_) => Val::I32(0),
        Elems::U32(_) => Val::U32(0),
        Elems::F32(_) => Val::F32(crate::numeric_contract::F32Bits::from_bits(0)),
    }
}

/// `target[index]` as this lane sees it.
pub(super) fn read(
    machine: &Machine<'_>,
    frame: &Frame,
    target: &Array,
    index: usize,
    span: Span,
) -> Result<Val, Stop> {
    let base = read_base(target, index, span)?;
    let Some(lane) = frame.lane.as_deref() else {
        return Ok(base);
    };
    let id = array_id(target);
    let symbolic = lane.phase == Phase::Symbolic;
    let unknown = || {
        if symbolic {
            Ok(Val::poison())
        } else {
            Err(Stop::Poison)
        }
    };
    if lane.poisoned.contains(&id) {
        return unknown();
    }
    let model = lane.models.get(&id);
    let own_ignored = lane.wrong && matches!(model, Some(Model::Own));
    if let Some(value) = lane.overlay.get(&(id, index))
        && !own_ignored
    {
        return Ok(value.clone());
    }
    let wrong = machine.mutation == Mutation::WrongCombine;
    match model {
        None | Some(Model::Own) => Ok(base),
        Some(Model::WriteOnly) => Err(Stop::violation("read of a write-only array")),
        Some(Model::Combine { table }) => match lane.phase {
            Phase::Symbolic => Ok(Val::poison()),
            Phase::Contribute => Ok(zero(&target.borrow())),
            Phase::Replay => match table
                .as_ref()
                .and_then(|t| t.before(lane.position, index, wrong))
            {
                Some(value) if value.is_sym() => Err(Stop::Poison),
                Some(value) => Ok(value),
                None => Err(Stop::internal("combine bounds")),
            },
        },
        Some(Model::Monotone { value, active, lo }) => {
            if &base == value {
                return Ok(base);
            }
            let writer = i128::try_from(index).map_err(|_| Stop::internal("index"))? - lo;
            let earlier = usize::try_from(writer).ok().filter(|w| *w < lane.position);
            match earlier.map(|w| active.get(w).copied().flatten()) {
                Some(Some(true)) if !wrong => Ok(value.clone()),
                Some(None) if !wrong => unknown(),
                _ => Ok(base),
            }
        }
        Some(Model::Fused { table }) => {
            match table.get(lane.position).and_then(|t| t.get(&index)) {
                Some(value) if !wrong => Ok(value.clone()),
                _ => Ok(base),
            }
        }
    }
}

/// `target[index] = value`: into the lane overlay, or the array itself.
pub(super) fn store(
    frame: &mut Frame,
    target: &Array,
    index: usize,
    value: Val,
    span: Span,
) -> Result<(), Stop> {
    if index >= target.borrow().len() {
        return Err(Stop::Trap(TrapCategory::IndexOutOfBounds, span));
    }
    let Some(lane) = frame.lane.as_deref_mut() else {
        if value.is_sym() {
            super::interp::poison(frame)?;
            return Err(Stop::violation("symbolic value stored"));
        }
        target.borrow_mut().set(index, &value)?;
        return Ok(());
    };
    if value.is_sym() && lane.phase != Phase::Symbolic {
        return Err(Stop::Poison);
    }
    let id = array_id(target);
    lane.arrays.entry(id).or_insert_with(|| target.clone());
    lane.overlay.insert((id, index), value.clone());
    lane.writes.push((id, index, value));
    Ok(())
}

struct Plan<'b> {
    body: &'b [Stmt],
    induction: String,
    lanes: Vec<Val>,
    final_induction: Option<Val>,
}

fn plan<'b>(
    machine: &mut Machine<'_>,
    frame: &mut Frame,
    inner: &'b Loop,
    node: &LoopNode,
) -> Result<Plan<'b>, Stop> {
    match (&inner.kind, &node.counting) {
        (LoopKind::For { var, lo, hi }, _) => {
            let (Val::U32(lo), Val::U32(hi)) = (machine.eval(frame, lo)?, machine.eval(frame, hi)?)
            else {
                return Err(Stop::internal("for bounds"));
            };
            machine.charge(u64::from(hi.saturating_sub(lo)))?;
            Ok(Plan {
                body: &inner.body,
                induction: var.clone(),
                lanes: (lo..hi).map(Val::U32).collect(),
                final_induction: None,
            })
        }
        (
            LoopKind::While {
                cond: Expr::Binary { right, .. },
            },
            Counting::CountedWhile { induction },
        ) => {
            let start = frame.get(induction)?;
            let bound = machine.eval(frame, right)?;
            let count = match (&start, &bound) {
                (Val::I32(a), Val::I32(b)) => (i64::from(*b) - i64::from(*a)).max(0) as u64,
                (Val::U32(a), Val::U32(b)) => u64::from(b.saturating_sub(*a)),
                _ => 0,
            };
            machine.charge(count)?;
            let lanes: Vec<Val> = match (&start, &bound) {
                (Val::I32(a), Val::I32(b)) => (*a..*b).map(Val::I32).collect(),
                (Val::U32(a), Val::U32(b)) => (*a..*b).map(Val::U32).collect(),
                _ => return Err(Stop::internal("counted while bounds")),
            };
            let body = inner
                .body
                .split_last()
                .map(|(_, rest)| rest)
                .ok_or_else(|| Stop::internal("counted while body"))?;
            Ok(Plan {
                body,
                induction: induction.clone(),
                final_induction: (!lanes.is_empty()).then_some(bound),
                lanes,
            })
        }
        _ => Err(Stop::violation("law form on an uncounted loop")),
    }
}

fn check_entry(frame: &Frame, guard: &Guard, law: Option<&Rep>) -> Result<(), Stop> {
    let int = |name: &str| frame.get(name).map(|v| v.int());
    let ok = match guard {
        Guard::ReducedSeed(name) => rep::reduced(law, int(name)?),
        Guard::StrideNonZero(name) => int(name)?.is_some_and(|x| x != 0),
        Guard::StateTable { .. } => table::admits(&frame.vars, guard),
        _ => true,
    };
    ok.then_some(()).ok_or_else(|| Stop::Guard(guard.clone()))
}

/// Execute a certified parallel loop through its laws, falling back to source
/// order when a guard fails or the law state cannot be formed.
pub(super) fn run(
    machine: &mut Machine<'_>,
    frame: &mut Frame,
    inner: &Loop,
    form: &LoopForm,
    node: &LoopNode,
) -> Result<(), Stop> {
    let laws: Vec<LawId> = form.certificates.iter().map(|c| c.law).collect();
    let pending = match frame.pending.take() {
        Some(p) if p.region == form.region => Some(p),
        other => {
            frame.pending = other;
            flush(frame)?;
            None
        }
    };
    match law_path(machine, frame, inner, form, node, pending.as_ref()) {
        Ok(()) => {
            machine.receipt.applied(laws);
            Ok(())
        }
        Err(stop) if stop.is_event() => {
            machine.receipt.applied(laws);
            Err(stop)
        }
        Err(stop @ Stop::Tool(..)) => Err(stop),
        Err(cause) => {
            machine.receipt.fallback(laws, &cause);
            if let Some(p) = pending {
                merge(&p.outs, p.reverse)?;
            }
            machine.serial_loop(frame, inner)
        }
    }
}

struct Ctx<'c> {
    entry: &'c HashMap<String, Val>,
    function: usize,
    induction: &'c str,
    lanes: &'c [Val],
    body: &'c [Stmt],
    models: Rc<HashMap<usize, Model>>,
    wrong: bool,
}

impl Ctx<'_> {
    fn frame(
        &self,
        k: usize,
        phase: Phase,
        values: &HashMap<String, Vec<Val>>,
        group: Option<Group>,
    ) -> Frame {
        let mut vars = self.entry.clone();
        for (name, per_lane) in values {
            if let Some(v) = per_lane.get(k) {
                vars.insert(name.clone(), v.clone());
            }
        }
        if let Some(v) = self.lanes.get(k) {
            vars.insert(self.induction.to_owned(), v.clone());
        }
        let mut frame = Frame::new(self.function, vars);
        frame.lane = Some(Box::new(Lane {
            phase,
            position: k,
            models: self.models.clone(),
            overlay: HashMap::new(),
            writes: Vec::new(),
            arrays: HashMap::new(),
            poisoned: HashSet::new(),
            group,
            select: None,
            range: (0, 0),
            capturing: false,
            wrong: self.wrong,
        }));
        frame
    }
}

/// Run one lane; events end it (and are returned), fallbacks propagate.
fn run_lane(
    machine: &mut Machine<'_>,
    frame: &mut Frame,
    body: &[Stmt],
) -> Result<Option<Stop>, Stop> {
    match machine.block(frame, body) {
        Ok(()) => Ok(None),
        Err(stop) if stop.is_event() || stop == Stop::Poison => Ok(Some(stop)),
        Err(stop) => Err(stop),
    }
}

fn group_of(rep: &Rep, var: &str) -> Group {
    Group {
        rep: rep.clone(),
        vars: match rep {
            Rep::Matrix { vars, .. } => vars.clone(),
            _ => vec![var.to_owned()],
        },
        companions: match rep {
            Rep::Select { companions, .. } => companions.clone(),
            _ => Vec::new(),
        },
    }
}

fn law_path(
    machine: &mut Machine<'_>,
    frame: &mut Frame,
    inner: &Loop,
    form: &LoopForm,
    node: &LoopNode,
    pending: Option<&Pending>,
) -> Result<(), Stop> {
    let plan = plan(machine, frame, inner, node)?;
    if machine.guards_on() {
        for certificate in &form.certificates {
            for guard in &certificate.guards {
                check_entry(frame, guard, certificate.rep.as_ref())?;
            }
        }
    }
    if plan.lanes.is_empty() {
        if let Some(p) = pending {
            merge(&p.outs, p.reverse)?;
        }
        return Ok(());
    }
    let wrong = machine.mutation == Mutation::WrongCombine;
    let entry = frame.vars.clone();
    let mut models: HashMap<usize, Model> = HashMap::new();
    let mut monotone = Vec::new();
    for (name, law) in &node.arrays {
        let id = array_id(frame.get(name)?.array()?);
        let model = match law {
            ArrayLaw::Own => Model::Own,
            ArrayLaw::Combine { .. } => Model::Combine { table: None },
            ArrayLaw::Monotone { own_writer: true } => {
                monotone.push((id, name.clone()));
                Model::Own
            }
            ArrayLaw::Ordered | ArrayLaw::Compaction { .. } | ArrayLaw::Monotone { .. } => {
                Model::WriteOnly
            }
        };
        models.insert(id, model);
    }
    if let Some(p) = pending {
        for (id, table) in fused_tables(p) {
            models.entry(id).or_insert(Model::Fused {
                table: Rc::new(table),
            });
        }
    }
    let mut ctx = Ctx {
        entry: &entry,
        function: frame.function,
        induction: &plan.induction,
        lanes: &plan.lanes,
        body: plan.body,
        models: Rc::new(models.clone()),
        wrong,
    };
    for (id, name) in monotone {
        let model = monotone_model(machine, &ctx, &name)?;
        models.insert(id, model);
    }
    ctx.models = Rc::new(models.clone());
    let (prefixes, finals) = carried(machine, &ctx, node)?;
    if node
        .arrays
        .iter()
        .any(|(_, law)| matches!(law, ArrayLaw::Combine { .. }))
    {
        let tables = combine_tables(machine, &ctx, &prefixes, node, frame)?;
        for (id, table) in tables {
            models.insert(
                id,
                Model::Combine {
                    table: Some(Rc::new(table)),
                },
            );
        }
        ctx.models = Rc::new(models);
    }
    let mut outs = Vec::with_capacity(plan.lanes.len());
    for k in 0..plan.lanes.len() {
        let mut lane_frame = ctx.frame(k, Phase::Replay, &prefixes, None);
        let event = run_lane(machine, &mut lane_frame, plan.body)?;
        let lane = lane_frame
            .lane
            .take()
            .ok_or_else(|| Stop::internal("lane"))?;
        outs.push(LaneOut {
            event,
            writes: lane.writes,
            arrays: lane.arrays,
        });
    }
    machine.receipt.lanes += plan.lanes.len() as u64;
    let mut events = outs.iter().enumerate().filter(|(_, o)| o.event.is_some());
    let first = if machine.mutation == Mutation::LastTrap {
        events.next_back()
    } else {
        events.next()
    };
    let reverse = wrong;
    if let Some((k, out)) = first {
        let event = out.event.clone().ok_or_else(|| Stop::internal("event"))?;
        return match event {
            Stop::Return(value) => {
                if let Some(p) = pending {
                    merge(&p.outs, p.reverse)?;
                }
                merge(outs.get(..=k).unwrap_or(&[]), reverse)?;
                Err(Stop::Return(value))
            }
            other => Err(other),
        };
    }
    if let Some(p) = pending {
        merge(&p.outs, p.reverse)?;
    }
    match node.fused_with {
        Some(region) => {
            frame.pending = Some(Pending {
                region,
                outs,
                reverse,
            })
        }
        None => merge(&outs, reverse)?,
    }
    if finals.iter().any(|(_, value)| value.is_sym()) {
        return Err(Stop::violation(
            "final carried value unrepresentable without an event",
        ));
    }
    for (name, value) in finals {
        frame.vars.insert(name, value);
    }
    if let Some(bound) = plan.final_induction {
        frame.vars.insert(plan.induction.clone(), bound);
    }
    Ok(())
}

/// Per array id written by the pending first loop: its writes per lane.
fn fused_tables(pending: &Pending) -> HashMap<usize, Vec<HashMap<usize, Val>>> {
    let mut tables: HashMap<usize, Vec<HashMap<usize, Val>>> = HashMap::new();
    let lanes = pending.outs.len();
    for (k, out) in pending.outs.iter().enumerate() {
        for (id, index, value) in &out.writes {
            let table = tables
                .entry(*id)
                .or_insert_with(|| vec![HashMap::new(); lanes]);
            if let Some(slot) = table.get_mut(k) {
                slot.insert(*index, value.clone());
            }
        }
    }
    tables
}

/// Carried groups in evaluation order: symbolic elements, tree scan, prefixes.
#[allow(clippy::type_complexity)]
fn carried(
    machine: &mut Machine<'_>,
    ctx: &Ctx<'_>,
    node: &LoopNode,
) -> Result<(HashMap<String, Vec<Val>>, Vec<(String, Val)>), Stop> {
    let mut all = Vec::new();
    for c in &node.carried {
        let group = group_of(&c.rep, &c.var);
        all.extend(group.vars.iter().cloned());
        all.extend(group.companions.iter().cloned());
    }
    let mut prefixes: HashMap<String, Vec<Val>> = HashMap::new();
    let mut finals = Vec::new();
    for c in &node.carried {
        let group = group_of(&c.rep, &c.var);
        let mut values = prefixes.clone();
        for name in &all {
            if !prefixes.contains_key(name) {
                values.insert(name.clone(), vec![Val::poison(); ctx.lanes.len()]);
            }
        }
        for (j, name) in group.vars.iter().enumerate() {
            values.insert(name.clone(), vec![sym::start(&group, j); ctx.lanes.len()]);
        }
        for name in &group.companions {
            values.insert(
                name.clone(),
                vec![Val::Sym(Box::new(Sym::Companion)); ctx.lanes.len()],
            );
        }
        let states = table::states(&group.rep, ctx.entry)?;
        let mut elems = Vec::with_capacity(ctx.lanes.len());
        for k in 0..ctx.lanes.len() {
            if poisoned(&prefixes, k) {
                // Unrepresentable prefix: an earlier lane has the first event
                // (the replay confirms it), so this element is never used.
                elems.push(rep::identity(&group.rep));
                continue;
            }
            if let Some(states) = states {
                elems.push(table::element(states, |state| {
                    let mut lane_frame = ctx.frame(k, Phase::Symbolic, &values, None);
                    lane_frame.vars.insert(c.var.clone(), Val::U32(state));
                    let event = run_lane(machine, &mut lane_frame, ctx.body)?;
                    event.map_or_else(|| lane_frame.get(&c.var).map(Some), |_| Ok(None))
                })?);
                continue;
            }
            let mut lane_frame = ctx.frame(k, Phase::Symbolic, &values, Some(group.clone()));
            run_lane(machine, &mut lane_frame, ctx.body)?;
            let state = group
                .vars
                .iter()
                .map(|v| lane_frame.get(v))
                .collect::<Result<Vec<_>, _>>()?;
            let lane = lane_frame
                .lane
                .as_deref()
                .ok_or_else(|| Stop::internal("lane"))?;
            elems.push(rep::element(&group.rep, &state, &lane.select, lane.range)?);
        }
        let (pre, total) = rep::scan(&group.rep, &elems, ctx.wrong)?;
        let seed = |v: &String| {
            ctx.entry
                .get(v)
                .cloned()
                .ok_or_else(|| Stop::internal("seed"))
        };
        let seeds = group.vars.iter().map(seed).collect::<Result<Vec<_>, _>>()?;
        let companion_seeds = group
            .companions
            .iter()
            .map(seed)
            .collect::<Result<Vec<_>, _>>()?;
        let names: Vec<&String> = group.vars.iter().chain(&group.companions).collect();
        for name in &names {
            prefixes.insert((*name).clone(), Vec::with_capacity(ctx.lanes.len()));
        }
        for elem in &pre {
            let (vals, comps) = rep::apply(&group.rep, elem, &seeds, &companion_seeds)?;
            for (name, value) in names.iter().zip(vals.into_iter().chain(comps)) {
                if let Some(list) = prefixes.get_mut(*name) {
                    list.push(value);
                }
            }
        }
        let (vals, comps) = rep::apply(&group.rep, &total, &seeds, &companion_seeds)?;
        for (name, value) in names.iter().zip(vals.into_iter().chain(comps)) {
            finals.push(((*name).clone(), value));
        }
    }
    Ok((prefixes, finals))
}

/// L11: contributions per lane (combine arrays read as zero), then the
/// running value of each target after each contributing lane.
fn combine_tables(
    machine: &mut Machine<'_>,
    ctx: &Ctx<'_>,
    prefixes: &HashMap<String, Vec<Val>>,
    node: &LoopNode,
    frame: &Frame,
) -> Result<HashMap<usize, CombineTable>, Stop> {
    let mut arrays: HashMap<usize, Array> = HashMap::new();
    for (name, law) in &node.arrays {
        if matches!(law, ArrayLaw::Combine { .. }) {
            let array = frame.get(name)?.array()?.clone();
            arrays.insert(array_id(&array), array);
        }
    }
    let mut tables: HashMap<usize, CombineTable> = arrays
        .iter()
        .map(|(id, array)| {
            let table = CombineTable {
                base: array.borrow().clone(),
                history: HashMap::new(),
            };
            (*id, table)
        })
        .collect();
    for k in 0..ctx.lanes.len() {
        if poisoned(prefixes, k) {
            continue;
        }
        let mut lane_frame = ctx.frame(k, Phase::Contribute, prefixes, None);
        run_lane(machine, &mut lane_frame, ctx.body)?;
        let lane = lane_frame
            .lane
            .take()
            .ok_or_else(|| Stop::internal("lane"))?;
        let mut mine: Vec<(usize, usize, Val)> = lane
            .overlay
            .into_iter()
            .filter(|((id, _), _)| tables.contains_key(id))
            .map(|((id, index), value)| (id, index, value))
            .collect();
        mine.sort_by_key(|(id, index, _)| (*id, *index));
        for (id, index, value) in mine {
            let table = tables
                .get_mut(&id)
                .ok_or_else(|| Stop::internal("combine table"))?;
            let current = table
                .before(k + 1, index, false)
                .ok_or_else(|| Stop::internal("combine bounds"))?;
            let next = accumulate(&current, &value)?;
            table.history.entry(index).or_default().push((k, next));
        }
    }
    Ok(tables)
}

/// `current + contribution`: exact for checked i32/u32 (poison when out of
/// range: an earlier lane traps first), modular for wrapping u32.
fn accumulate(current: &Val, contribution: &Val) -> Result<Val, Stop> {
    Ok(match (current, contribution) {
        (Val::Sym(_), _) => Val::poison(),
        (Val::I32(a), Val::I32(b)) => a.checked_add(*b).map_or_else(Val::poison, Val::I32),
        (Val::U32(a), Val::U32(b)) => Val::U32(a.wrapping_add(*b)),
        _ => return Err(Stop::internal("combine element type")),
    })
}

/// L14 with reads: which lanes write their own flag, from round-start state.
fn monotone_model(machine: &mut Machine<'_>, ctx: &Ctx<'_>, array: &str) -> Result<Model, Stop> {
    let mut site = None;
    tree::walk(ctx.body, &mut Vec::new(), 0, &mut |stmt, guards, depth| {
        if let Stmt::Store {
            array: a, value, ..
        } = stmt
            && a == array
            && depth == 0
        {
            site = Some((
                value.clone(),
                guards
                    .iter()
                    .map(|(g, p)| ((*g).clone(), *p))
                    .collect::<Vec<_>>(),
            ));
        }
    });
    let (value, guards) = site.ok_or_else(|| Stop::violation("monotone writer not found"))?;
    let no_values = HashMap::new();
    let mut probe = ctx.frame(0, Phase::Replay, &no_values, None);
    let value = machine.eval(&mut probe, &value)?;
    let mut active = Vec::with_capacity(ctx.lanes.len());
    for k in 0..ctx.lanes.len() {
        let mut lane_frame = ctx.frame(k, Phase::Replay, &no_values, None);
        let mut state = Some(true);
        for (guard, polarity) in &guards {
            match machine.eval(&mut lane_frame, guard) {
                Ok(Val::Bool(b)) if b == *polarity => {}
                Ok(Val::Bool(_)) => {
                    state = Some(false);
                    break;
                }
                Ok(_) => return Err(Stop::internal("guard type")),
                Err(stop) if stop.is_event() || stop == Stop::Poison => {
                    state = None;
                    break;
                }
                Err(stop) => return Err(stop),
            }
        }
        active.push(state);
    }
    let lo = ctx
        .lanes
        .first()
        .and_then(Val::int)
        .ok_or_else(|| Stop::internal("monotone lanes"))?;
    Ok(Model::Monotone {
        value,
        active: Rc::new(active),
        lo,
    })
}
