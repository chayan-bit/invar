//! Extract admitted public-source map regions through the typed graph.
//!
//! Graph inputs are runtime slots bound to the live source arrays when the plan
//! runs; no element value is taken from source text.

use std::ops::Range;

use crate::{
    admission::{self, MapShape, Region, Upper},
    front::Span,
    mir::{Function, Operation, Terminator, ValueId},
    plan::Execution,
    typed_net::{self, Effect, Limits, Net, Node, NodeId, Op, Plan, Type},
};

mod chain;
pub use chain::{ChainLink, FusedChain, fuse_chain};

/// Analysis bytes one graph node occupies: the node slot plus its (at most two) input ports.
const NODE_BYTES: usize = size_of::<Option<Node>>() + 2 * size_of::<NodeId>();
/// Recursion depth of the verify, observe and extract walks and of the proof walks below. It
/// bounds native stack, not memory, so the byte budget cannot replace it. It is far above any
/// chain the byte budget admits in practice; a deeper graph is refused and runs serially.
const STACK_DEPTH: usize = 256;

/// The graph limits an analysis budget implies (#278): as many nodes as `analysis_bytes` holds,
/// two ports per node, as many map operations as nodes, and the language's per-array limit.
/// Only `depth` is not a byte budget: it is the stack guard `STACK_DEPTH`.
pub const fn graph_limits(analysis_bytes: usize) -> Limits {
    let nodes = analysis_bytes / NODE_BYTES;
    Limits {
        nodes,
        ports: nodes.saturating_mul(2),
        depth: STACK_DEPTH,
        array_len: crate::array_limits::MAX_ARRAY_LEN,
        map_ops: nodes,
    }
}

pub const GRAPH_LIMITS: Limits = graph_limits(crate::budget::DEFAULT_ANALYSIS_BYTES);

/// Analysis bytes a typed graph holds, in the unit `graph_limits` divides the budget by. Plans
/// extracted from a graph hold at most one operation per node, so they ride on the node charge.
pub(crate) fn net_bytes(net: &Net) -> usize {
    net.nodes
        .len()
        .saturating_mul(NODE_BYTES)
        .saturating_add(net.rule_history.len().saturating_mul(size_of::<&str>()))
}

/// Heap bytes of a list of owner names: the slots plus each name's capacity.
pub(crate) fn names_bytes(names: &[String]) -> usize {
    names
        .iter()
        .map(|name| name.capacity())
        .fold(size_of_val(names), usize::saturating_add)
}

impl LiveMap {
    /// Analysis bytes this map retains, its header included.
    pub(crate) fn held_bytes(&self) -> usize {
        size_of::<Self>()
            .saturating_add(names_bytes(&self.inputs))
            .saturating_add(self.output.capacity())
    }
}

impl Fused {
    /// Analysis bytes this fused pair retains, its header included.
    pub(crate) fn held_bytes(&self) -> usize {
        size_of::<Self>()
            .saturating_add(net_bytes(&self.unfused))
            .saturating_add(net_bytes(&self.graph))
            .saturating_add(names_bytes(&self.inputs))
            .saturating_add(self.intermediate.capacity())
            .saturating_add(self.consumer_input.capacity())
            .saturating_add(self.consumer_other.as_ref().map_or(0, String::capacity))
            .saturating_add(self.output.capacity())
    }
}

/// Longest run of map regions one fused plan holds. It is the typed-graph extractor's own plan
/// size (`typed_net::MAX_DAG_MAPS`), not an analysis budget: `extract_dag` refuses a longer
/// chain, and `plan::decide` then fuses the run in segments of at most this many stages and
/// leaves the rest as ordinary maps. Lifting it means resizing the device plan, so it stays.
pub const MAX_FUSED_STAGES: usize = typed_net::MAX_DAG_MAPS;
/// Longest `move` chain the proofs follow. Each step rescans the function, so the walk is
/// quadratic in the chain: this bounds analysis time, not memory.
const MAX_OWNER_CHAIN: usize = 128;

#[derive(Debug)]
pub enum Error {
    Admission(admission::Error),
    Graph(typed_net::Error),
}

impl From<admission::Error> for Error {
    fn from(value: admission::Error) -> Self {
        Self::Admission(value)
    }
}
impl From<typed_net::Error> for Error {
    fn from(value: typed_net::Error) -> Self {
        Self::Graph(value)
    }
}

/// A graph-derived plan for one admitted map region over live arrays.
#[derive(Clone, Debug)]
pub struct LiveMap {
    pub plan: Plan,
    /// Source owners bound to the plan's input slots, in slot order.
    pub inputs: Vec<String>,
    pub output: String,
}

/// Which intermediate value a fused pass also yields for publication.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Capture {
    /// Each element after its first `k` operations in stage order.
    AfterOps(usize),
    /// Operand `k` of the combining zip after its own operations.
    Operand(usize),
}

/// Two adjacent map regions fused into one element pass. The producer's output
/// (`intermediate`) stays observable, so the pass also yields `capture` for it.
#[derive(Clone, Debug)]
pub struct Fused {
    /// The region graph before fusion: input slots, producer stage, consumer map.
    pub unfused: Net,
    /// The same graph after the map-fusion rewrite; `plan` is extracted from it.
    pub graph: Net,
    pub plan: Plan,
    pub source_spans: [Span; 2],
    pub capture: Capture,
    /// Owners bound to the plan's array slots, in slot order.
    pub inputs: Vec<String>,
    pub intermediate: String,
    pub consumer_input: String,
    /// The consumer's other operand when it is a two-input map.
    pub consumer_other: Option<String>,
    pub output: String,
    /// The intermediate's element values are never observed (only the consumer
    /// reads them, and its output is always published), so they are not written.
    pub elide_intermediate: bool,
}

impl Plan {
    /// Evaluate `range` of a live plan over `arrays` (indexed by input slot),
    /// also returning each element's `capture` value when asked.
    pub fn live_values(
        &self,
        arrays: &[&[u32]],
        range: Range<usize>,
        capture: Option<Capture>,
    ) -> Result<(Vec<u32>, Vec<u32>), typed_net::Error> {
        let Self::LiveArray {
            inputs,
            pre,
            zip,
            stages,
            ..
        } = self
        else {
            return Err(typed_net::Error::Type(0));
        };
        let operations = stages.iter().flatten().copied().collect::<Vec<_>>();
        let slices = inputs
            .iter()
            .map(|slot| {
                usize::try_from(*slot)
                    .ok()
                    .and_then(|slot| arrays.get(slot))
                    .filter(|values| range.end <= values.len())
                    .copied()
                    .ok_or(typed_net::Error::ArrayLimit)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let valid_capture = match capture {
            None => true,
            Some(Capture::AfterOps(k)) => k <= operations.len(),
            Some(Capture::Operand(k)) => k < slices.len(),
        };
        if range.start > range.end
            || !valid_capture
            || pre.len() != slices.len()
            || slices.len() != 1 + usize::from(zip.is_some())
        {
            return Err(typed_net::Error::Type(0));
        }
        let mut captured = Vec::new();
        let mut values = Vec::new();
        captured
            .try_reserve_exact(if capture.is_some() { range.len() } else { 0 })
            .and_then(|()| values.try_reserve_exact(range.len()))
            .map_err(|_| typed_net::Error::ArrayLimit)?;
        let mut operands = [0u32; 2];
        for index in range {
            for (k, (slice, ops)) in slices.iter().zip(pre).enumerate() {
                operands[k] = ops.iter().fold(slice[index], |v, op| op.apply(v));
                if capture == Some(Capture::Operand(k)) {
                    captured.push(operands[k]);
                }
            }
            let mut value = match zip {
                Some(op) => op.apply(operands[0], operands[1]),
                None => operands[0],
            };
            for (position, operation) in operations.iter().enumerate() {
                if capture == Some(Capture::AfterOps(position)) {
                    captured.push(value);
                }
                value = operation.apply(value);
            }
            if capture == Some(Capture::AfterOps(operations.len())) {
                captured.push(value);
            }
            values.push(value);
        }
        Ok((captured, values))
    }

    /// Run a live plan over the first `len` elements of `arrays`.
    pub fn run_live(&self, arrays: &[&[u32]], len: usize) -> Result<Vec<u32>, typed_net::Error> {
        Ok(self.live_values(arrays, 0..len, None)?.1)
    }
}

fn push(graph: &mut Net, op: Op, inputs: Vec<NodeId>) -> NodeId {
    let source_id = u32::try_from(graph.nodes.len() + 1).unwrap_or(u32::MAX);
    graph.push(Node {
        source_id,
        ty: Type::ArrayU32,
        effect: Effect::Total,
        inputs,
        op,
    })
}

/// Copy the element tree rooted at `id`, so a later stage can reuse an earlier
/// stage's value while the verified graph stays a tree (one parent per node).
fn copy_tree(graph: &mut Net, id: NodeId) -> Result<NodeId, Error> {
    // Repeated reuse can grow copies geometrically; stop at the graph bound.
    if graph.nodes.len() >= GRAPH_LIMITS.nodes {
        return Err(typed_net::Error::NodeLimit.into());
    }
    let node = graph
        .nodes
        .get(id)
        .and_then(Option::as_ref)
        .ok_or(typed_net::Error::InvalidNode(id))?
        .clone();
    let inputs = node
        .inputs
        .iter()
        .map(|input| copy_tree(graph, *input))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(push(graph, node.op, inputs))
}

/// Push input slots and the region's element stage; returns the stage node.
fn push_stage(graph: &mut Net, shape: MapShape, first_slot: u32) -> NodeId {
    match shape {
        MapShape::Unary(op) => {
            let input = push(graph, Op::InputU32(first_slot), Vec::new());
            push(graph, Op::Map(op), vec![input])
        }
        MapShape::Zip(op) => {
            let left = push(graph, Op::InputU32(first_slot), Vec::new());
            let right = push(graph, Op::InputU32(first_slot + 1), Vec::new());
            let len = GRAPH_LIMITS.array_len;
            push(graph, Op::ZipMap { op, len }, vec![left, right])
        }
    }
}

fn region_parts(region: &Region) -> Result<(MapShape, &admission::Guard), Error> {
    match (region.shape(), region.guard()) {
        (Some(shape), Some(guard)) => Ok((shape, guard)),
        _ => Err(typed_net::Error::Type(0).into()),
    }
}

/// Extract the live plan of one admitted map region.
pub(crate) fn single(region: &Region) -> Result<LiveMap, Error> {
    let (shape, guard) = region_parts(region)?;
    let mut graph = Net::empty(0);
    graph.entry = push_stage(&mut graph, shape, 0);
    graph.seal_source();
    let proof = graph.verify(GRAPH_LIMITS)?;
    Ok(LiveMap {
        plan: graph.extract(&proof)?,
        inputs: guard.inputs().to_vec(),
        output: guard.output().to_string(),
    })
}

fn binding(function: &Function, name: &str) -> Option<ValueId> {
    function
        .blocks
        .iter()
        .flat_map(|block| &block.instructions)
        .find_map(|instruction| match &instruction.kind {
            Operation::Bind { name: local, value } if local == name => Some(*value),
            _ => None,
        })
}

fn definition(function: &Function, id: ValueId) -> Option<&Operation> {
    function
        .blocks
        .iter()
        .flat_map(|block| &block.instructions)
        .find(|instruction| {
            instruction
                .output
                .as_ref()
                .is_some_and(|(out, _)| *out == id)
        })
        .map(|instruction| &instruction.kind)
}

fn reassigned(function: &Function, name: &str) -> bool {
    function
        .blocks
        .iter()
        .flat_map(|block| &block.instructions)
        .any(|instruction| matches!(&instruction.kind, Operation::Store { name: target, .. } if target == name))
}

/// Owners from `name` back through `move` bindings; ends at the original owner.
fn move_chain(function: &Function, name: &str) -> Option<Vec<String>> {
    let mut chain = vec![name.to_string()];
    for _ in 0..MAX_OWNER_CHAIN {
        let last = chain.last()?;
        match binding(function, last).and_then(|id| definition(function, id)) {
            Some(Operation::Move(previous)) if !chain.contains(previous) => {
                chain.push(previous.clone())
            }
            Some(Operation::Move(_)) => return None,
            _ => return Some(chain),
        }
    }
    None
}

#[derive(Debug, PartialEq, Eq)]
enum Length {
    Constant(usize),
    Of(String),
}

/// A symbolic array length that no statement can change; `None` if unknown.
fn owner_length(function: &Function, name: &str, depth: usize) -> Option<Length> {
    if depth > STACK_DEPTH {
        return None;
    }
    let chain = move_chain(function, name)?;
    if chain.iter().any(|owner| reassigned(function, owner)) {
        return None;
    }
    let root = chain.last()?;
    let Some(value) = binding(function, root) else {
        return function
            .params
            .iter()
            .any(|param| param.name == *root)
            .then(|| Length::Of(root.clone()));
    };
    match definition(function, value)? {
        Operation::Array(items) => Some(Length::Constant(items.len())),
        Operation::Call { name, args } if name == "filled_u32" || name == "filled_i32" => {
            match definition(function, *args.first()?)? {
                Operation::U32(n) => Some(Length::Constant(usize::try_from(*n).ok()?)),
                Operation::Length(other) => owner_length(function, other, depth + 1),
                _ => None,
            }
        }
        _ => Some(Length::Of(root.clone())),
    }
}

fn trip_count(function: &Function, upper: &Upper) -> Option<Length> {
    match upper {
        Upper::Constant(n) => Some(Length::Constant(usize::try_from(*n).ok()?)),
        Upper::Length(name) => owner_length(function, name, 0),
    }
}

fn successors(end: &Terminator) -> Vec<usize> {
    match end {
        Terminator::Goto { target, .. } => vec![*target],
        Terminator::Branch { yes, no, .. } => vec![*yes, *no],
        Terminator::Return { .. } => Vec::new(),
    }
}

/// Blocks from the first loop's exit to the second loop's preheader, if control
/// can only flow straight from one to the other.
fn straight_segment(function: &Function, first: usize, second: usize) -> Option<Vec<usize>> {
    let mut predecessors = vec![Vec::new(); function.blocks.len()];
    for (id, block) in function.blocks.iter().enumerate() {
        for target in block.end.as_ref().map(successors).unwrap_or_default() {
            predecessors.get_mut(target)?.push(id);
        }
    }
    let Some(Terminator::Branch { no: exit, .. }) = &function.blocks.get(first)?.end else {
        return None;
    };
    if predecessors.get(*exit)? != &[first] {
        return None;
    }
    let mut segment = Vec::new();
    let mut current = *exit;
    for _ in 0..function.blocks.len() {
        segment.push(current);
        let Some(Terminator::Goto { target, .. }) = &function.blocks.get(current)?.end else {
            return None;
        };
        if *target == second {
            return Some(segment);
        }
        if predecessors.get(*target)? != &[current] {
            return None;
        }
        current = *target;
    }
    None
}

fn segment_writes(function: &Function, segment: &[usize], chain: &[String]) -> bool {
    segment
        .iter()
        .filter_map(|id| function.blocks.get(*id))
        .flat_map(|block| &block.instructions)
        .any(|instruction| match &instruction.kind {
            Operation::Store { name, .. } | Operation::StoreIndex { name, .. } => {
                chain.contains(name)
            }
            Operation::Call { name, .. } => !matches!(
                name.as_str(),
                "len" | "filled_u32" | "filled_i32" | "wrapping_add" | "wrapping_mul"
            ),
            _ => false,
        })
}

/// Whether any segment block declares or may write one of `names`.
fn segment_binds_or_writes(function: &Function, segment: &[usize], names: &[String]) -> bool {
    segment_writes(function, segment, names)
        || segment
            .iter()
            .filter_map(|id| function.blocks.get(*id))
            .flat_map(|block| &block.instructions)
            .any(|instruction| {
                matches!(&instruction.kind, Operation::Bind { name, .. } if names.contains(name))
            })
}

/// Build and fuse the region graph for a producer stage feeding the consumer.
/// Returns the unfused graph, the fused graph, its plan, and the capture.
fn fused_graph(
    producer: MapShape,
    consumer: MapShape,
    consumer_side: usize,
) -> Result<Option<(Net, Net, Plan, Capture)>, Error> {
    let mut graph = Net::empty(0);
    let stage = push_stage(&mut graph, producer, 0);
    let zip_consumer = match consumer {
        MapShape::Unary(op) => {
            graph.entry = push(&mut graph, Op::Map(op), vec![stage]);
            false
        }
        MapShape::Zip(op) => {
            let MapShape::Unary(_) = producer else {
                return Ok(None);
            };
            let other = push(&mut graph, Op::InputU32(1), Vec::new());
            let mut operands = vec![stage, other];
            if consumer_side == 1 {
                operands.swap(0, 1);
            }
            let len = GRAPH_LIMITS.array_len;
            graph.entry = push(&mut graph, Op::ZipMap { op, len }, operands);
            true
        }
    };
    graph.seal_source();
    graph.verify(GRAPH_LIMITS)?;
    let unfused = graph.clone();
    if graph.rewrite_map_fusion(GRAPH_LIMITS)?.is_none() {
        return Ok(None);
    }
    let plan = graph.extract(&graph.verify(GRAPH_LIMITS)?)?;
    let Plan::LiveArray { stages: after, .. } = &plan else {
        return Err(typed_net::Error::Type(graph.entry).into());
    };
    if after.len() != 1 {
        return Err(typed_net::Error::RuleRejected.into());
    }
    let capture = if zip_consumer {
        Capture::Operand(consumer_side)
    } else {
        let before = unfused.extract(&unfused.verify(GRAPH_LIMITS)?)?;
        let Plan::LiveArray { stages: before, .. } = before else {
            return Err(typed_net::Error::Type(unfused.entry).into());
        };
        if after[0] != before.concat() {
            return Err(typed_net::Error::RuleRejected.into());
        }
        Capture::AfterOps(before.first().map_or(0, Vec::len))
    };
    Ok(Some((unfused, graph, plan, capture)))
}

fn loop_body(function: &Function, head: usize) -> Option<usize> {
    match function.blocks.get(head)?.end {
        Some(Terminator::Branch { yes, .. }) => Some(yes),
        _ => None,
    }
}

/// Whether no statement observes the intermediate's elements except the consumer
/// body: the only other uses are the producer's stores, the `move` between chain
/// owners, lengths, and cleanup.
fn intermediate_unobservable(
    function: &Function,
    chain: &[String],
    producer: usize,
    consumer: usize,
) -> bool {
    let chain_values = function
        .blocks
        .iter()
        .flat_map(|block| &block.instructions)
        .filter_map(|instruction| match &instruction.kind {
            Operation::Bind { name, value } if chain.contains(name) => Some(*value),
            _ => None,
        })
        .collect::<Vec<_>>();
    function.blocks.iter().enumerate().all(|(id, block)| {
        block.instructions.iter().all(|instruction| {
            let output = instruction.output.as_ref().map(|(value, _)| *value);
            match &instruction.kind {
                Operation::StoreIndex { name, .. } if chain.contains(name) => id == producer,
                Operation::ArrayRef { name, .. } if chain.contains(name) => id == consumer,
                Operation::Move(name) if chain.contains(name) => {
                    output.is_some_and(|value| chain_values.contains(&value))
                }
                Operation::Read(name) | Operation::Store { name, .. } => !chain.contains(name),
                _ => true,
            }
        })
    })
}

/// Fuse two admitted map regions when the second always runs straight after the
/// first, reads the first's unmodified output as one operand, and has the same
/// trip count. A two-input consumer requires a one-input producer.
pub(crate) fn fuse_pair(
    function: &Function,
    first: &Region,
    second: &Region,
) -> Result<Option<Fused>, Error> {
    let ((first_shape, g1), (second_shape, g2)) = (region_parts(first)?, region_parts(second)?);
    let chains = g2
        .inputs()
        .iter()
        .map(|name| move_chain(function, name))
        .collect::<Option<Vec<_>>>();
    let Some(chains) = chains else {
        return Ok(None);
    };
    let feeds = |chain: &Vec<String>| chain.last().map(String::as_str) == Some(g1.output());
    let sides = chains.iter().filter(|chain| feeds(chain)).count();
    let Some(side) = chains.iter().position(feeds) else {
        return Ok(None);
    };
    let other = (g2.inputs().len() == 2).then(|| &g2.inputs()[1 - side]);
    let Some(segment) = straight_segment(function, first.head, second.head) else {
        return Ok(None);
    };
    let mut watched = chains[side].clone();
    watched.extend(other.cloned());
    let first_trips = trip_count(function, g1.upper());
    if sides != 1
        || segment_writes(function, &segment, &watched)
        || first_trips.is_none()
        || first_trips != trip_count(function, g2.upper())
    {
        return Ok(None);
    }
    let Some((unfused, graph, plan, capture)) = fused_graph(first_shape, second_shape, side)?
    else {
        return Ok(None);
    };
    let mut inputs = g1.inputs().to_vec();
    inputs.extend(other.cloned());
    // Elision needs the consumer's publication to be certain once the producer ran:
    // a one-input consumer whose output length is the shared trip count.
    let elide_intermediate = other.is_none()
        && owner_length(function, g2.output(), 0) == first_trips
        && match (
            loop_body(function, first.head),
            loop_body(function, second.head),
        ) {
            (Some(producer), Some(consumer)) => {
                intermediate_unobservable(function, &chains[side], producer, consumer)
            }
            _ => false,
        };
    Ok(Some(Fused {
        unfused,
        graph,
        plan,
        source_spans: [first.span(), second.span()],
        capture,
        inputs,
        intermediate: g1.output().to_string(),
        consumer_input: g2.inputs()[side].clone(),
        consumer_other: other.cloned(),
        output: g2.output().to_string(),
        elide_intermediate,
    }))
}

/// The first fused region pair chosen by the shared planner, if any.
pub fn extract_fused(source: &[u8], program: &crate::mir::Program) -> Result<Option<Fused>, Error> {
    let report = admission::analyze(source, program)?;
    Ok(report
        .regions()
        .iter()
        .find_map(|region| match region.execution() {
            Execution::FusedProducer(fused) => Some((**fused).clone()),
            _ => None,
        }))
}
