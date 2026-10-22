//! Structural recognition of zero-based counted `for` loops in verified MIR.

use std::collections::HashMap;

use crate::{
    front::{BinaryOp, Span, Type},
    mir::{BasicBlock, Function, Operation, Terminator, ValueId},
    typed_net::{Effect, Net, Node, NodeId, Op},
};

pub(crate) type Defs<'a> = HashMap<ValueId, &'a Operation>;

pub(crate) fn definitions(function: &Function) -> Defs<'_> {
    function
        .blocks
        .iter()
        .flat_map(|block| &block.instructions)
        .filter_map(|instruction| {
            instruction
                .output
                .as_ref()
                .map(|(id, _)| (*id, &instruction.kind))
        })
        .collect()
}

/// A loop `for i in 0u32 .. hi { body }` whose body is one block.
pub(crate) struct CountedLoop<'a> {
    pub body: &'a BasicBlock,
    pub span: Span,
    pub induction_name: &'a str,
}

/// The counted loop whose head is block `head`, recognized from MIR alone.
pub(crate) fn counted_loop<'a>(
    function: &'a Function,
    defs: &Defs<'_>,
    head: usize,
) -> Option<CountedLoop<'a>> {
    let block = function.blocks.get(head)?;
    let (
        Some((induction, Type::U32)),
        Some(Terminator::Branch {
            condition,
            yes,
            span,
            ..
        }),
    ) = (block.params.first(), &block.end)
    else {
        return None;
    };
    if !matches!(defs.get(condition), Some(Operation::Binary { op: BinaryOp::Lt, left, .. }) if left == induction)
    {
        return None;
    }
    let incoming = function
        .blocks
        .iter()
        .enumerate()
        .filter_map(|(id, candidate)| match &candidate.end {
            Some(Terminator::Goto { target, args, .. }) if *target == head && id != *yes => {
                Some(args.first().copied())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    if !matches!(incoming.as_slice(), [Some(lower)] if matches!(defs.get(lower), Some(Operation::U32(0))))
    {
        return None;
    }
    let body = function.blocks.get(*yes)?;
    if !matches!(&body.end, Some(Terminator::Goto { target, args, .. }) if *target == head
        && matches!(args.as_slice(), [next] if matches!(defs.get(next), Some(Operation::ForNext(id)) if id == induction)))
    {
        return None;
    }
    let induction_name = body.instructions.iter().find_map(|i| match &i.kind {
        Operation::Bind { name, value } if value == induction => Some(name.as_str()),
        _ => None,
    })?;
    Some(CountedLoop {
        body,
        span: *span,
        induction_name,
    })
}

/// The array owner of `id` when it is `name[i]` for the loop induction `i`.
pub(crate) fn element_of<'a>(
    defs: &Defs<'a>,
    id: ValueId,
    induction_name: &str,
) -> Option<&'a str> {
    let Some(Operation::Index { array, index }) = defs.get(&id) else {
        return None;
    };
    if !matches!(defs.get(index), Some(Operation::Read(name)) if name == induction_name) {
        return None;
    }
    match defs.get(array) {
        Some(Operation::ArrayRef { name, .. }) => Some(name),
        _ => None,
    }
}

/// Whether `name` is a local or parameter of type `ty` (and mutable, if asked).
pub(crate) fn local_is(function: &Function, name: &str, ty: Type, mutable: bool) -> bool {
    function
        .locals
        .get(name)
        .is_some_and(|local| local.ty == ty && (!mutable || local.mutable))
}

/// Push a graph node whose source id is its position.
pub(crate) fn push_node(
    graph: &mut Net,
    ty: crate::typed_net::Type,
    op: Op,
    inputs: Vec<NodeId>,
) -> NodeId {
    let source_id = u32::try_from(graph.nodes.len() + 1).unwrap_or(u32::MAX);
    graph.push(Node {
        source_id,
        ty,
        effect: Effect::Total,
        inputs,
        op,
    })
}
