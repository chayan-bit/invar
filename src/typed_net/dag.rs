//! One-pass extraction of a verified, total element graph.

use std::ops::Range;

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DagOp {
    Input(u32),
    Map {
        input: usize,
        op: MapOp,
    },
    Zip {
        left: usize,
        right: usize,
        op: ZipOp,
    },
}

impl Net {
    /// Extract all source-region roots from one verified tree into element SSA.
    /// Graph connections, rather than source spelling, determine every operand.
    pub fn extract_dag(&self, proof: &Proof, roots: &[NodeId]) -> Result<Plan, Error> {
        self.check_proof(proof)?;
        if roots.is_empty() || roots.len() > MAX_DAG_MAPS {
            return Err(Error::MapLimit);
        }
        let mut lowered = vec![None; self.nodes.len()];
        let mut ops = Vec::new();
        let mut slots = Vec::new();
        let mut map_count = 0;
        fn visit(
            net: &Net,
            id: NodeId,
            lowered: &mut [Option<usize>],
            ops: &mut Vec<DagOp>,
            slots: &mut Vec<u32>,
            map_count: &mut usize,
        ) -> Result<usize, Error> {
            if let Some(index) = lowered.get(id).copied().flatten() {
                return Ok(index);
            }
            let node = net.node(id)?;
            if node.effect != Effect::Total || node.ty != Type::ArrayU32 {
                return Err(Error::Type(id));
            }
            let children = node
                .inputs
                .iter()
                .map(|child| visit(net, *child, lowered, ops, slots, map_count))
                .collect::<Result<Vec<_>, _>>()?;
            let mut emit = |op| {
                let index = ops.len();
                ops.push(op);
                index
            };
            let output = match &node.op {
                Op::InputU32(slot) => {
                    if !slots.contains(slot) {
                        slots.push(*slot);
                    }
                    emit(DagOp::Input(*slot))
                }
                Op::Map(op) => {
                    *map_count += 1;
                    emit(DagOp::Map {
                        input: children[0],
                        op: *op,
                    })
                }
                Op::FusedMap(maps) => {
                    *map_count += maps.len();
                    maps.iter()
                        .fold(children[0], |input, op| emit(DagOp::Map { input, op: *op }))
                }
                Op::ZipMap { op, .. } => emit(DagOp::Zip {
                    left: children[0],
                    right: children[1],
                    op: *op,
                }),
                Op::FusedZipMap {
                    op,
                    left,
                    right,
                    suffix,
                    ..
                } => {
                    *map_count += left.len() + right.len() + suffix.len();
                    let lhs = left
                        .iter()
                        .fold(children[0], |input, op| emit(DagOp::Map { input, op: *op }));
                    let rhs = right
                        .iter()
                        .fold(children[1], |input, op| emit(DagOp::Map { input, op: *op }));
                    let zipped = emit(DagOp::Zip {
                        left: lhs,
                        right: rhs,
                        op: *op,
                    });
                    suffix
                        .iter()
                        .fold(zipped, |input, op| emit(DagOp::Map { input, op: *op }))
                }
                _ => return Err(Error::Type(id)),
            };
            if *map_count > MAX_DAG_MAPS || ops.len() > 64 {
                return Err(Error::MapLimit);
            }
            *lowered.get_mut(id).ok_or(Error::InvalidNode(id))? = Some(output);
            Ok(output)
        }
        visit(
            self,
            self.entry,
            &mut lowered,
            &mut ops,
            &mut slots,
            &mut map_count,
        )?;
        let outputs = roots
            .iter()
            .map(|root| {
                lowered
                    .get(*root)
                    .copied()
                    .flatten()
                    .ok_or(Error::InvalidNode(*root))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Plan::LiveDag {
            inputs: slots,
            ops,
            outputs,
            source_identity: self.source_identity,
            graph_revision: self.revision,
        })
    }
}

impl Plan {
    /// Evaluate every source-stage result without materializing graph edges.
    pub fn run_dag(&self, arrays: &[&[u32]], range: Range<usize>) -> Result<Vec<Vec<u32>>, Error> {
        let Self::LiveDag {
            inputs,
            ops,
            outputs,
            ..
        } = self
        else {
            return Err(Error::Type(0));
        };
        if range.start > range.end || outputs.is_empty() || ops.len() > 64 {
            return Err(Error::ArrayLimit);
        }
        for slot in inputs {
            let slice = arrays
                .get(usize::try_from(*slot).map_err(|_| Error::ArrayLimit)?)
                .ok_or(Error::ArrayLimit)?;
            if range.end > slice.len() {
                return Err(Error::ArrayLimit);
            }
        }
        let mut results = Vec::new();
        results
            .try_reserve_exact(outputs.len())
            .map_err(|_| Error::ArrayLimit)?;
        for _ in outputs {
            let mut values = Vec::new();
            values
                .try_reserve_exact(range.len())
                .map_err(|_| Error::ArrayLimit)?;
            results.push(values);
        }
        let mut row = Vec::new();
        row.try_reserve_exact(ops.len())
            .map_err(|_| Error::ArrayLimit)?;
        for index in range {
            row.clear();
            for op in ops {
                let value = match *op {
                    DagOp::Input(slot) => *arrays
                        .get(usize::try_from(slot).map_err(|_| Error::ArrayLimit)?)
                        .and_then(|slice| slice.get(index))
                        .ok_or(Error::ArrayLimit)?,
                    DagOp::Map { input, op } => op.apply(*row.get(input).ok_or(Error::Type(0))?),
                    DagOp::Zip { left, right, op } => op.apply(
                        *row.get(left).ok_or(Error::Type(0))?,
                        *row.get(right).ok_or(Error::Type(0))?,
                    ),
                };
                row.push(value);
            }
            for (result, source) in results.iter_mut().zip(outputs) {
                result.push(*row.get(*source).ok_or(Error::Type(0))?);
            }
        }
        Ok(results)
    }
}
