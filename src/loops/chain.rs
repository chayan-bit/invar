//! Linear source-region chains lowered through one verified element graph.

use super::*;

#[derive(Clone, Debug)]
pub struct ChainLink {
    pub span: Span,
    pub consumer_input: String,
    pub consumer_other: Option<String>,
    /// Earlier stage whose output the zip reads as `consumer_other`.
    pub other_stage: Option<usize>,
    pub output: String,
}

#[derive(Clone, Debug)]
pub struct FusedChain {
    pub graph: Net,
    pub plan: Plan,
    pub source_spans: Vec<Span>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub links: Vec<ChainLink>,
}

impl FusedChain {
    /// Analysis bytes this chain retains, its header included.
    pub(crate) fn held_bytes(&self) -> usize {
        let links = self.links.iter().map(|link| {
            size_of::<ChainLink>()
                + link.consumer_input.capacity()
                + link.consumer_other.as_ref().map_or(0, String::capacity)
                + link.output.capacity()
        });
        links
            .fold(size_of::<Self>(), usize::saturating_add)
            .saturating_add(super::net_bytes(&self.graph))
            .saturating_add(
                self.source_spans
                    .capacity()
                    .saturating_mul(size_of::<Span>()),
            )
            .saturating_add(super::names_bytes(&self.inputs))
            .saturating_add(super::names_bytes(&self.outputs))
    }
    /// Values stage `index + 1` must still observe at its loop head: the prior
    /// stage under its consumer name and, for a zip, the other operand, both
    /// from the same producer pass. `inputs` follow `self.inputs`.
    pub fn link_operands(
        &self,
        index: usize,
        stages: &[Vec<u32>],
        inputs: &[&[u32]],
        len: usize,
    ) -> Option<Vec<(String, Vec<u32>)>> {
        let link = self.links.get(index)?;
        let mut operands = vec![(link.consumer_input.clone(), stages.get(index)?.clone())];
        if let Some(other) = &link.consumer_other {
            let values = match link.other_stage {
                Some(stage) => stages.get(stage)?.clone(),
                None => {
                    let slot = self.inputs.iter().position(|name| name == other)?;
                    inputs.get(slot)?.get(..len)?.to_vec()
                }
            };
            operands.push((other.clone(), values));
        }
        Some(operands)
    }
}

/// Fuse a straight run of admitted source maps. Every stage remains a separate
/// publication point; only element computation moves to the first reached loop.
pub fn fuse_chain(function: &Function, regions: &[Region]) -> Result<Option<FusedChain>, Error> {
    if !(2..=MAX_FUSED_STAGES).contains(&regions.len()) {
        return Ok(None);
    }
    let (first_shape, first_guard) = region_parts(&regions[0])?;
    let mut graph = Net::empty(0);
    let mut root = push_stage(&mut graph, first_shape, 0);
    let mut roots = vec![root];
    let mut inputs = first_guard.inputs().to_vec();
    let mut outputs = vec![first_guard.output().to_owned()];
    let mut source_spans = vec![regions[0].span()];
    let mut links = Vec::new();
    let mut between = Vec::new();
    let first_trips = trip_count(function, first_guard.upper());
    if first_trips.is_none() {
        return Ok(None);
    }
    for pair in regions.windows(2) {
        let (producer, consumer) = (&pair[0], &pair[1]);
        if producer.function() != consumer.function() || producer.function() != function.name {
            return Ok(None);
        }
        let (_, previous) = region_parts(producer)?;
        let (shape, next) = region_parts(consumer)?;
        let chains = next
            .inputs()
            .iter()
            .map(|name| move_chain(function, name))
            .collect::<Option<Vec<_>>>();
        let Some(chains) = chains else {
            return Ok(None);
        };
        let feeds =
            |chain: &Vec<String>| chain.last().map(String::as_str) == Some(previous.output());
        let sides = chains.iter().filter(|chain| feeds(chain)).count();
        let Some(side) = chains.iter().position(feeds) else {
            return Ok(None);
        };
        let other = (next.inputs().len() == 2).then(|| next.inputs()[1 - side].clone());
        let Some(segment) = straight_segment(function, producer.head, consumer.head) else {
            return Ok(None);
        };
        let mut watched = chains[side].clone();
        watched.extend(other.clone());
        between.extend(segment.iter().copied());
        if sides != 1
            || segment_writes(function, &segment, &watched)
            || trip_count(function, next.upper()) != first_trips
        {
            return Ok(None);
        }
        let mut other_stage = None;
        root = match shape {
            MapShape::Unary(op) => push(&mut graph, Op::Map(op), vec![root]),
            MapShape::Zip(op) => {
                let other = other.as_ref().ok_or(typed_net::Error::Type(0))?;
                let owners = &chains[1 - side];
                other_stage = outputs
                    .iter()
                    .position(|output| owners.last() == Some(output));
                let leaf = if let Some(stage) = other_stage {
                    // An earlier stage's value is recomputed in-graph; it is not
                    // live at the producer head, so it cannot be a device input.
                    copy_tree(&mut graph, roots[stage])?
                } else {
                    // The producer head must already see the operand unchanged.
                    if segment_binds_or_writes(function, &between, owners) {
                        return Ok(None);
                    }
                    let slot = match inputs.iter().position(|name| name == other) {
                        Some(slot) => slot,
                        None => {
                            inputs.push(other.clone());
                            inputs.len() - 1
                        }
                    };
                    let slot = u32::try_from(slot).map_err(|_| typed_net::Error::MapLimit)?;
                    push(&mut graph, Op::InputU32(slot), Vec::new())
                };
                let mut operands = vec![root, leaf];
                if side == 1 {
                    operands.swap(0, 1);
                }
                push(
                    &mut graph,
                    Op::ZipMap {
                        op,
                        len: GRAPH_LIMITS.array_len,
                    },
                    operands,
                )
            }
        };
        roots.push(root);
        outputs.push(next.output().to_owned());
        source_spans.push(consumer.span());
        links.push(ChainLink {
            span: consumer.span(),
            consumer_input: next.inputs()[side].clone(),
            consumer_other: other,
            other_stage,
            output: next.output().to_owned(),
        });
    }
    graph.entry = root;
    graph.seal_source();
    let proof = graph.verify(GRAPH_LIMITS)?;
    let plan = graph.extract_dag(&proof, &roots)?;
    Ok(Some(FusedChain {
        graph,
        plan,
        source_spans,
        inputs,
        outputs,
        links,
    }))
}
