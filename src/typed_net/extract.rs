//! Executable plans extracted from verified graph connections.

use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Plan {
    Term {
        program: term::Program,
        entry: term::Term,
        source_identity: u64,
        graph_revision: u64,
    },
    Array {
        input: Vec<u32>,
        /// One entry is one serial loop. A fused entry performs all operations
        /// per element before writing the result.
        stages: Vec<Vec<MapOp>>,
        source_identity: u64,
        graph_revision: u64,
    },
    ZipArray {
        left: Vec<u32>,
        right: Vec<u32>,
        len: usize,
        op: ZipOp,
        suffix: Vec<MapOp>,
        fused: bool,
        source_identity: u64,
        graph_revision: u64,
    },
    /// An element pipeline over runtime input slots. Element `i` reads every slot
    /// at `i` (combining two with `zip`), then applies each stage in order; stage 0
    /// runs in the same loop as the read.
    LiveArray {
        inputs: Vec<u32>,
        /// Operations applied to each input's element before it is combined.
        pre: Vec<Vec<MapOp>>,
        zip: Option<ZipOp>,
        stages: Vec<Vec<MapOp>>,
        source_identity: u64,
        graph_revision: u64,
    },
    /// A verified element DAG evaluated once per index, with one output for each
    /// source map region in a fused chain.
    LiveDag {
        inputs: Vec<u32>,
        ops: Vec<DagOp>,
        outputs: Vec<usize>,
        source_identity: u64,
        graph_revision: u64,
    },
    /// Bounded ordered action; slot namespaces depend on the seed family.
    /// Public plan fields describe a candidate, never a source publication permit.
    BoundedScan {
        family: BoundedFamily,
        prefix: Prefix,
        seed: u32,
        input: u32,
        steps: u32,
        source_identity: u64,
        graph_revision: u64,
    },
    /// Fold array slot `input` from scalar slot `seed`, left to right.
    Reduce {
        law: crate::laws::Law,
        seed: u32,
        input: u32,
        source_identity: u64,
        graph_revision: u64,
    },
    /// Full affine pair from two scalar seeds and two coefficient arrays.
    PairReduce {
        seeds: [u32; 2],
        inputs: [u32; 2],
        source_identity: u64,
        graph_revision: u64,
    },
    /// Affine states from scalar slot `seed`; each coefficient is a constant or
    /// element `i` of an array slot.
    AffineScan {
        prefix: Prefix,
        seed: u32,
        a: Source,
        b: Source,
        source_identity: u64,
        graph_revision: u64,
    },
}

/// Where a live affine coefficient comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Constant(u32),
    Slot(u32),
}

impl Plan {
    pub fn run_term(&self, limits: term::Limits) -> Result<term::Observation, Error> {
        let Self::Term { program, entry, .. } = self else {
            return Err(Error::Type(0));
        };
        Ok(term::observe(program, entry, limits))
    }

    pub fn run_array(&self) -> Result<Vec<u32>, Error> {
        match self {
            Self::Array { input, stages, .. } => {
                let mut values = input.clone();
                for stage in stages {
                    for value in &mut values {
                        for &op in stage {
                            *value = op.apply(*value);
                        }
                    }
                }
                Ok(values)
            }
            Self::ZipArray {
                left,
                right,
                len,
                op,
                suffix,
                fused,
                ..
            } => {
                let mut values = left
                    .iter()
                    .copied()
                    .zip(right.iter().copied())
                    .take(*len)
                    .map(|(a, b)| {
                        let mut value = op.apply(a, b);
                        if *fused {
                            for &operation in suffix {
                                value = operation.apply(value);
                            }
                        }
                        value
                    })
                    .collect::<Vec<_>>();
                if !*fused {
                    for value in &mut values {
                        for &operation in suffix {
                            *value = operation.apply(*value);
                        }
                    }
                }
                Ok(values)
            }
            Self::Term { .. }
            | Self::LiveArray { .. }
            | Self::LiveDag { .. }
            | Self::PairReduce { .. }
            | Self::Reduce { .. }
            | Self::AffineScan { .. }
            | Self::BoundedScan { .. } => Err(Error::Type(0)),
        }
    }

    pub fn loop_count(&self) -> usize {
        match self {
            Self::Array { stages, .. } => stages.len(),
            Self::ZipArray { suffix, fused, .. } => 1 + usize::from(!suffix.is_empty() && !*fused),
            Self::LiveArray { stages, .. } => stages.len(),
            Self::LiveDag { .. } => 1,
            Self::PairReduce { .. }
            | Self::Reduce { .. }
            | Self::AffineScan { .. }
            | Self::BoundedScan { .. } => 1,
            Self::Term { .. } => 0,
        }
    }
}

/// Input slots, per-input operations, optional two-input combiner, and element
/// stages of a live plan.
type LiveParts = (Vec<u32>, Vec<Vec<MapOp>>, Option<ZipOp>, Vec<Vec<MapOp>>);

impl Net {
    fn array_plan(&self, id: NodeId) -> Result<(Vec<u32>, Vec<Vec<MapOp>>), Error> {
        let node = self.node(id)?;
        match &node.op {
            Op::ArrayU32(values) => Ok((values.clone(), Vec::new())),
            Op::Map(op) => {
                let (input, mut stages) = self.array_plan(node.inputs[0])?;
                stages.push(vec![*op]);
                Ok((input, stages))
            }
            Op::FusedMap(ops) => {
                let (input, mut stages) = self.array_plan(node.inputs[0])?;
                stages.push(ops.clone());
                Ok((input, stages))
            }
            _ => Err(Error::Type(id)),
        }
    }

    fn live_plan(&self, id: NodeId) -> Result<LiveParts, Error> {
        let node = self.node(id)?;
        let slots = || {
            node.inputs
                .iter()
                .map(|input| match self.node(*input)?.op {
                    Op::InputU32(slot) => Ok(slot),
                    _ => Err(Error::Type(*input)),
                })
                .collect::<Result<Vec<_>, _>>()
        };
        match &node.op {
            Op::InputU32(slot) => Ok((vec![*slot], vec![Vec::new()], None, Vec::new())),
            Op::ZipMap { op, .. } => Ok((
                slots()?,
                vec![Vec::new(), Vec::new()],
                Some(*op),
                vec![Vec::new()],
            )),
            Op::FusedZipMap {
                op,
                left,
                right,
                suffix,
                ..
            } => Ok((
                slots()?,
                vec![left.clone(), right.clone()],
                Some(*op),
                vec![suffix.clone()],
            )),
            Op::Map(_) | Op::FusedMap(_) => {
                let (inputs, pre, zip, mut stages) = self.live_plan(node.inputs[0])?;
                let ops = match &node.op {
                    Op::Map(op) => vec![*op],
                    Op::FusedMap(ops) => ops.clone(),
                    _ => return Err(Error::Type(id)),
                };
                stages.push(ops);
                Ok((inputs, pre, zip, stages))
            }
            _ => Err(Error::Type(id)),
        }
    }

    pub(super) fn scalar_slot(&self, id: NodeId) -> Result<u32, Error> {
        match self.node(id)?.op {
            Op::InputScalar(slot) => Ok(slot),
            _ => Err(Error::Type(id)),
        }
    }

    pub(super) fn array_slot(&self, id: NodeId) -> Result<u32, Error> {
        match self.node(id)?.op {
            Op::InputU32(slot) => Ok(slot),
            _ => Err(Error::Type(id)),
        }
    }

    /// Plans whose root is a live reduction or affine scan node.
    fn summary_plan(&self) -> Result<Option<Plan>, Error> {
        let root = self.node(self.entry)?;
        let (source_identity, graph_revision) = (self.source_identity, self.revision);
        match &root.op {
            Op::BoundedScan { family, prefix, .. } => Ok(Some(Plan::BoundedScan {
                family: *family,
                prefix: *prefix,
                seed: match family {
                    BoundedFamily::Table(_) => self.scalar_slot(root.inputs[0])?,
                    BoundedFamily::Matrix(_) => self.array_slot(root.inputs[0])?,
                },
                input: self.array_slot(root.inputs[1])?,
                steps: self.scalar_slot(root.inputs[2])?,
                source_identity,
                graph_revision,
            })),
            Op::PairReduce { .. } => Ok(Some(Plan::PairReduce {
                seeds: [
                    self.scalar_slot(root.inputs[0])?,
                    self.scalar_slot(root.inputs[1])?,
                ],
                inputs: [
                    self.array_slot(root.inputs[2])?,
                    self.array_slot(root.inputs[3])?,
                ],
                source_identity,
                graph_revision,
            })),
            Op::Reduce { law, .. } => Ok(Some(Plan::Reduce {
                law: *law,
                seed: self.scalar_slot(root.inputs[0])?,
                input: self.array_slot(root.inputs[1])?,
                source_identity,
                graph_revision,
            })),
            Op::AffineScan { prefix, a, b, .. } => {
                let mut arrays = root.inputs.iter().skip(1);
                let mut source = |coefficient: Coefficient| match coefficient {
                    Coefficient::Constant(n) => Ok(Source::Constant(n)),
                    Coefficient::Input => {
                        let id = arrays.next().ok_or(Error::Arity(self.entry))?;
                        Ok(Source::Slot(self.array_slot(*id)?))
                    }
                };
                Ok(Some(Plan::AffineScan {
                    prefix: *prefix,
                    seed: self.scalar_slot(root.inputs[0])?,
                    a: source(*a)?,
                    b: source(*b)?,
                    source_identity,
                    graph_revision,
                }))
            }
            _ => Ok(None),
        }
    }

    /// Extract a conventional serial plan from the verified graph connections.
    pub fn extract(&self, proof: &Proof) -> Result<Plan, Error> {
        self.check_proof(proof)?;
        if let Some(plan) = self.summary_plan()? {
            return Ok(plan);
        }
        if self
            .nodes
            .iter()
            .flatten()
            .any(|node| matches!(node.op, Op::InputU32(_) | Op::InputScalar(_)))
        {
            let (inputs, pre, zip, stages) = self.live_plan(self.entry)?;
            if stages.is_empty() {
                return Err(Error::Type(self.entry));
            }
            return Ok(Plan::LiveArray {
                inputs,
                pre,
                zip,
                stages,
                source_identity: self.source_identity,
                graph_revision: self.revision,
            });
        }
        if self.node(self.entry)?.ty == Type::ArrayU32 {
            let root = self.node(self.entry)?;
            let zip = match &root.op {
                Op::FusedZipMap {
                    op,
                    len,
                    left,
                    right,
                    suffix,
                } if left.is_empty() && right.is_empty() => {
                    Some((*op, *len, suffix.clone(), true, root.inputs.as_slice()))
                }
                Op::FusedZipMap { .. } => return Err(Error::Type(self.entry)),
                Op::Map(op) => match self.node(root.inputs[0])? {
                    Node {
                        op: Op::ZipMap { op: zip_op, len },
                        inputs,
                        ..
                    } => Some((*zip_op, *len, vec![*op], false, inputs.as_slice())),
                    _ => None,
                },
                _ => None,
            };
            if let Some((op, len, suffix, fused, inputs)) = zip {
                let (Op::ArrayU32(left), Op::ArrayU32(right)) =
                    (&self.node(inputs[0])?.op, &self.node(inputs[1])?.op)
                else {
                    return Err(Error::Type(self.entry));
                };
                return Ok(Plan::ZipArray {
                    left: left.clone(),
                    right: right.clone(),
                    len,
                    op,
                    suffix,
                    fused,
                    source_identity: self.source_identity,
                    graph_revision: self.revision,
                });
            }
            let (input, stages) = self.array_plan(self.entry)?;
            Ok(Plan::Array {
                input,
                stages,
                source_identity: self.source_identity,
                graph_revision: self.revision,
            })
        } else {
            let (program, entry) = self.decode(proof)?;
            Ok(Plan::Term {
                program,
                entry,
                source_identity: self.source_identity,
                graph_revision: self.revision,
            })
        }
    }
}
