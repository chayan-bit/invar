//! Checked M3 resource facts shared by record producers and runtimes. These
//! descriptive counts confer no admission, execution or publication authority.
use crate::{Capabilities, CollectiveKind, Ir, Verified};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CostFacts {
    pub input_bytes: usize,
    pub output_bytes: usize,
    pub temporary_bytes: usize,
    pub launches: usize,
    pub operations: usize,
}

/// Exact temporary device storage for the existing collective implementations.
/// Metal allocates length/block/seed scalars and chunk bases/carry pairs in
/// addition to the two summary arrays. CUDA passes its scalars by value.
pub fn collective_workspace(ir: &Ir, metal: bool) -> Option<usize> {
    let c = ir.collective.as_ref()?;
    // Pair reductions use the same ordered two-word summaries as affine scans.
    let paired = matches!(
        c.kind,
        CollectiveKind::AffineScan { .. } | CollectiveKind::PairReduction { .. }
    );
    let summaries = (ir.geometry.blocks as usize).checked_mul(if paired { 16 } else { 8 })?;
    if !metal {
        return Some(summaries);
    }
    let chunks = ir
        .geometry
        .blocks
        .div_ceil(crate::METAL_SCAN_CHUNK_BLOCKS)
        .max(1) as usize;
    summaries
        .checked_add(12)?
        .checked_add(chunks.checked_mul(4)?)?
        .checked_add(chunks.checked_add(1)?.checked_mul(8)?)
}

impl Verified {
    pub fn cost_facts(&self) -> Option<CostFacts> {
        let ir = self.ir();
        let n = ir.length as usize;
        let metal = matches!(self.capabilities(), Capabilities::Metal { .. });
        if n == 0 {
            return Some(CostFacts {
                input_bytes: 0,
                output_bytes: 0,
                temporary_bytes: 0,
                launches: 0,
                operations: 0,
            });
        }
        let (inputs, output_words, operations, temporary_bytes, launches) =
            if let Some(c) = &ir.collective {
                let chunks = if metal
                    && matches!(
                        c.kind,
                        CollectiveKind::AffineScan { .. } | CollectiveKind::PairReduction { .. }
                    ) {
                    ir.geometry
                        .blocks
                        .div_ceil(crate::METAL_SCAN_CHUNK_BLOCKS)
                        .max(1) as usize
                } else {
                    1
                };
                (
                    c.inputs.len(),
                    c.output.elements as usize,
                    2,
                    collective_workspace(ir, metal)?,
                    chunks.checked_mul(3)?,
                )
            } else if let Some(dag) = &ir.dag {
                (
                    dag.inputs.len(),
                    n.checked_mul(dag.outputs.len())?,
                    dag.nodes.len(),
                    if metal { 4 } else { 0 },
                    1,
                )
            } else {
                (1, n, ir.ops.len(), if metal { 4 } else { 0 }, 1)
            };
        Some(CostFacts {
            input_bytes: n.checked_mul(inputs)?.checked_mul(4)?,
            output_bytes: output_words.checked_mul(4)?,
            temporary_bytes,
            launches,
            operations,
        })
    }
}
