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

/// The head of a zero-based counted loop: the body block its branch enters,
/// the loop span and the induction parameter.
pub(crate) fn counted_head(
    function: &Function,
    defs: &Defs<'_>,
    head: usize,
) -> Option<(usize, Span, ValueId)> {
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
        .filter_map(|(id, candidate)| {
            match &candidate.end {
            // The back edge `goto head(i + 1)` may leave any body block.
            Some(Terminator::Goto { target, args, .. })
                if *target == head
                    && id != *yes
                    && !args.first().is_some_and(|next| {
                        matches!(defs.get(next), Some(Operation::ForNext(id)) if id == induction)
                    }) =>
            {
                Some(args.first().copied())
            }
            _ => None,
        }
        })
        .collect::<Vec<_>>();
    if !matches!(incoming.as_slice(), [Some(lower)] if matches!(defs.get(lower), Some(Operation::U32(0))))
    {
        return None;
    }
    Some((*yes, *span, *induction))
}

/// Whether `block` closes one iteration: `goto head(i + 1)`.
pub(crate) fn steps_back(
    block: &BasicBlock,
    defs: &Defs<'_>,
    head: usize,
    induction: ValueId,
) -> bool {
    matches!(&block.end, Some(Terminator::Goto { target, args, .. }) if *target == head
        && matches!(args.as_slice(), [next] if matches!(defs.get(next), Some(Operation::ForNext(id)) if *id == induction)))
}

/// The name bound to the induction parameter in `block`.
pub(crate) fn induction_name(block: &BasicBlock, induction: ValueId) -> Option<&str> {
    block.instructions.iter().find_map(|i| match &i.kind {
        Operation::Bind { name, value } if *value == induction => Some(name.as_str()),
        _ => None,
    })
}

/// The counted loop whose head is block `head`, recognized from MIR alone.
pub(crate) fn counted_loop<'a>(
    function: &'a Function,
    defs: &Defs<'_>,
    head: usize,
) -> Option<CountedLoop<'a>> {
    let (yes, span, induction) = counted_head(function, defs, head)?;
    let body = function.blocks.get(yes)?;
    if !steps_back(body, defs, head, induction) {
        return None;
    }
    Some(CountedLoop {
        body,
        span,
        induction_name: induction_name(body, induction)?,
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
