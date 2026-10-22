//! Lower an admitted, graph-extracted map to the shared physical device IR.

use invar_device::{
    AddressSpace, Barrier, Buffer, Coefficient, Collective, CollectiveKind, DagNode, DagOp,
    Geometry, Ir, LaneMask, Layout, MapDag, Numeric, Op, PrefixMode, ReduceLaw, Span, Transfer,
};

use crate::{
    admission::Region,
    laws,
    plan::Execution,
    typed_net::{DagOp as GraphDagOp, MapOp, Plan, Prefix, Source, ZipOp},
};

/// Bump when typed-to-physical extraction rules change independently of the IR schema.
pub const EXTRACTION_RULE: &str = "typed-u32-physical-extraction-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Ineligible,
    Length,
    /// The region's kernel needs more MSL buffer arguments than Apple GPUs bind.
    MetalBufferSlots {
        needed: usize,
        limit: usize,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ineligible => f.write_str("no device plan for this region"),
            Self::Length => f.write_str("region length exceeds the device plan bound"),
            Self::MetalBufferSlots { needed, limit } => write!(
                f,
                "region needs {needed} Metal buffer slots; Apple GPUs bind at most {limit}"
            ),
        }
    }
}

/// Metal planning for a lowered region: retile for Apple GPUs, then refuse a
/// region whose kernel would bind more buffer slots than the target allows.
pub fn plan_metal(ir: &mut Ir) -> Result<(), Error> {
    retile_metal(ir);
    let needed = invar_device::metal_buffer_slots(ir);
    let limit = invar_device::METAL_MAX_BUFFERS;
    if needed > limit {
        return Err(Error::MetalBufferSlots { needed, limit });
    }
    Ok(())
}

/// Metal keeps more independent map/DAG threadgroups resident. The measured
/// M4 64 Mi chain reached 91.8% of copy bandwidth at 32,768 groups versus
/// 65.4% at the generic 1,024-group geometry.
pub fn retile_metal(ir: &mut Ir) {
    if ir.dag.is_some() {
        ir.geometry.blocks = ir.length.div_ceil(ir.geometry.lanes).clamp(1, 32_768);
    } else if let Some(collective) = &ir.collective
        && matches!(
            collective.kind,
            CollectiveKind::AffineScan { .. } | CollectiveKind::PairReduction { .. }
        )
    {
        ir.geometry.lanes = 1024;
    }
}

pub fn from_region(region: &Region, length: u32) -> Result<Ir, Error> {
    if let Execution::PairReduction(candidate) = region.execution() {
        let Plan::PairReduce {
            seeds: [0, 1],
            inputs: [0, 1],
            source_identity,
            graph_revision,
        } = &candidate.plan
        else {
            return Err(Error::Ineligible);
        };
        return collective_ir(
            region,
            length,
            *source_identity,
            *graph_revision,
            2,
            CollectiveKind::PairReduction {
                identity_a: 1,
                identity_b: 0,
            },
            2,
        );
    }
    if let Execution::Reduction(reduction) = region.execution() {
        let Plan::Reduce {
            law,
            seed,
            input,
            source_identity,
            graph_revision,
        } = &reduction.plan
        else {
            return Err(Error::Ineligible);
        };
        if *seed != 0 || *input != 0 {
            return Err(Error::Ineligible);
        }
        let law = match law {
            laws::Law::WrappingSum => ReduceLaw::WrappingSum,
            laws::Law::WrappingProduct => ReduceLaw::WrappingProduct,
        };
        return collective_ir(
            region,
            length,
            *source_identity,
            *graph_revision,
            1,
            CollectiveKind::Reduction {
                law,
                identity: law.identity(),
            },
            1,
        );
    }
    if let Execution::Scan(candidate) = region.execution() {
        let Plan::AffineScan {
            prefix,
            seed,
            a,
            b,
            source_identity,
            graph_revision,
        } = &candidate.plan
        else {
            return Err(Error::Ineligible);
        };
        if *seed != 0 {
            return Err(Error::Ineligible);
        }
        let coefficient = |value: Source| match value {
            Source::Constant(n) => Coefficient::Constant(n),
            Source::Slot(slot) => Coefficient::Input(slot),
        };
        let prefix = match prefix {
            Prefix::None => PrefixMode::None,
            Prefix::Inclusive => PrefixMode::Inclusive,
            Prefix::Exclusive => PrefixMode::Exclusive,
        };
        let output_len = if prefix == PrefixMode::None {
            1
        } else {
            length.checked_add(1).ok_or(Error::Length)?
        };
        return collective_ir(
            region,
            length,
            *source_identity,
            *graph_revision,
            candidate.arrays.len(),
            CollectiveKind::AffineScan {
                a: coefficient(*a),
                b: coefficient(*b),
                prefix,
                identity_a: 1,
                identity_b: 0,
            },
            output_len,
        );
    }
    if let Execution::DagProducer(chain) = region.execution() {
        return from_chain(chain, length);
    }
    if let Execution::FusedProducer(pair) = region.execution() {
        let consumer = pair.unfused.entry;
        let node = pair
            .unfused
            .nodes
            .get(consumer)
            .and_then(Option::as_ref)
            .ok_or(Error::Ineligible)?;
        let side = match pair.capture {
            crate::loops::Capture::AfterOps(_) => 0,
            crate::loops::Capture::Operand(side) => side,
        };
        let producer = *node.inputs.get(side).ok_or(Error::Ineligible)?;
        let proof = pair
            .unfused
            .verify(crate::loops::GRAPH_LIMITS)
            .map_err(|_| Error::Ineligible)?;
        let plan = pair
            .unfused
            .extract_dag(&proof, &[producer, consumer])
            .map_err(|_| Error::Ineligible)?;
        return from_dag(&plan, pair.inputs.len(), 2, &pair.source_spans, length);
    }
    let Execution::Map(map) = region.execution() else {
        return Err(Error::Ineligible);
    };
    let Plan::LiveArray {
        inputs,
        pre,
        zip,
        stages,
        source_identity,
        graph_revision,
    } = &map.plan
    else {
        return Err(Error::Ineligible);
    };
    if zip.is_some() || length > invar_device::MAX_ELEMENTS {
        return from_live_map(region, &map.plan, length);
    }
    if inputs.as_slice() != [0]
        || pre.as_slice() != [Vec::<MapOp>::new()]
        || length > invar_device::MAX_ELEMENTS
    {
        return Err(Error::Ineligible);
    }
    let mut ops = Vec::new();
    for stage in stages {
        for op in stage {
            ops.push(match op {
                MapOp::WrappingAdd(n) => Op::WrappingAdd(*n),
                MapOp::WrappingMul(n) => Op::WrappingMul(*n),
            });
        }
    }
    let blocks = length.div_ceil(256).max(1);
    let span = region.span();
    Ok(Ir {
        version: invar_device::IR_VERSION,
        span: Span {
            start: span.start,
            end: span.end,
        },
        source_identity: *source_identity,
        graph_revision: *graph_revision,
        input: Buffer {
            id: 0,
            elements: length,
            address: AddressSpace::Device,
            layout: Layout::ContiguousU32,
        },
        output: Buffer {
            id: 1,
            elements: length,
            address: AddressSpace::Device,
            layout: Layout::ContiguousU32,
        },
        length,
        geometry: Geometry { blocks, lanes: 256 },
        lane_mask: LaneMask::IndexLtLength,
        barrier: Barrier::None,
        requires_barrier: false,
        transfer: Transfer::Owned,
        numeric: Numeric::WrappingU32,
        completion: true,
        ops,
        dag: None,
        collective: None,
    })
}

fn collective_ir(
    region: &Region,
    length: u32,
    source_identity: u64,
    graph_revision: u64,
    inputs: usize,
    kind: CollectiveKind,
    output_len: u32,
) -> Result<Ir, Error> {
    if length > invar_device::collective::MAX_ELEMENTS || inputs > 2 {
        return Err(Error::Length);
    }
    let make_buffer = |id: usize, elements| -> Result<Buffer, Error> {
        Ok(Buffer {
            id: u32::try_from(id).map_err(|_| Error::Length)?,
            elements,
            address: AddressSpace::Device,
            layout: Layout::ContiguousU32,
        })
    };
    let input_buffers = (0..inputs)
        .map(|i| make_buffer(i, length))
        .collect::<Result<Vec<_>, _>>()?;
    let output = make_buffer(inputs, output_len)?;
    let span = region.span();
    Ok(Ir {
        version: invar_device::IR_VERSION,
        span: Span {
            start: span.start,
            end: span.end,
        },
        source_identity,
        graph_revision,
        input: *input_buffers.first().unwrap_or(&output),
        output,
        length,
        geometry: Geometry {
            blocks: length
                .div_ceil(invar_device::collective::BLOCK_ITEMS)
                .max(1),
            lanes: if matches!(
                kind,
                CollectiveKind::AffineScan { .. } | CollectiveKind::PairReduction { .. }
            ) {
                invar_device::collective::SCAN_LANES
            } else {
                invar_device::collective::LANES
            },
        },
        lane_mask: LaneMask::IndexLtLength,
        barrier: Barrier::Uniform,
        requires_barrier: true,
        transfer: Transfer::Owned,
        numeric: Numeric::WrappingU32,
        completion: true,
        ops: Vec::new(),
        dag: None,
        collective: Some(Collective {
            kind,
            inputs: input_buffers,
            output,
            block_items: invar_device::collective::BLOCK_ITEMS,
        }),
    })
}

/// A zip or large standalone map uses the shared DAG contract;
/// the single-input M3-02 map encoding stays frozen.
fn from_live_map(region: &Region, plan: &Plan, length: u32) -> Result<Ir, Error> {
    let Plan::LiveArray {
        inputs,
        pre,
        zip,
        stages,
        source_identity,
        graph_revision,
    } = plan
    else {
        return Err(Error::Ineligible);
    };
    if inputs.is_empty() || inputs.len() > 2 || pre.len() != inputs.len() {
        return Err(Error::Ineligible);
    }
    let mut ops = Vec::new();
    let mut operands = Vec::new();
    for (slot, maps) in inputs.iter().zip(pre) {
        ops.push(GraphDagOp::Input(*slot));
        for op in maps {
            ops.push(GraphDagOp::Map {
                input: ops.len() - 1,
                op: *op,
            });
        }
        operands.push(ops.len() - 1);
    }
    if let Some(zip) = zip {
        ops.push(GraphDagOp::Zip {
            left: operands[0],
            right: *operands.get(1).ok_or(Error::Ineligible)?,
            op: *zip,
        });
    } else if operands.len() != 1 {
        return Err(Error::Ineligible);
    }
    for op in stages.iter().flatten() {
        ops.push(GraphDagOp::Map {
            input: ops.len() - 1,
            op: *op,
        });
    }
    let dag = Plan::LiveDag {
        inputs: inputs.to_vec(),
        outputs: vec![ops.len() - 1],
        ops,
        source_identity: *source_identity,
        graph_revision: *graph_revision,
    };
    from_dag(&dag, inputs.len(), 1, &[region.span()], length)
}

fn from_chain(chain: &crate::loops::FusedChain, length: u32) -> Result<Ir, Error> {
    from_dag(
        &chain.plan,
        chain.inputs.len(),
        chain.outputs.len(),
        &chain.source_spans,
        length,
    )
}

/// Lower a graph-extracted element DAG whose slots are `0..input_count`, with
/// one output per source stage span.
fn from_dag(
    plan: &Plan,
    input_count: usize,
    output_count: usize,
    source_spans: &[crate::front::Span],
    length: u32,
) -> Result<Ir, Error> {
    let Plan::LiveDag {
        inputs,
        ops,
        outputs,
        source_identity,
        graph_revision,
    } = plan
    else {
        return Err(Error::Ineligible);
    };
    if length > invar_device::MAX_DAG_ELEMENTS {
        return Err(Error::Length);
    }
    let mut slots = inputs.clone();
    slots.sort_unstable();
    if inputs.len() != input_count
        || outputs.len() != output_count
        || slots.iter().enumerate().any(|(i, slot)| *slot != i as u32)
    {
        return Err(Error::Ineligible);
    }
    let make_buffer = |id: usize| -> Result<Buffer, Error> {
        Ok(Buffer {
            id: u32::try_from(id).map_err(|_| Error::Length)?,
            elements: length,
            address: AddressSpace::Device,
            layout: Layout::ContiguousU32,
        })
    };
    let input_buffers = (0..inputs.len())
        .map(make_buffer)
        .collect::<Result<Vec<_>, _>>()?;
    let output_buffers = (inputs.len()..inputs.len() + outputs.len())
        .map(make_buffer)
        .collect::<Result<Vec<_>, _>>()?;
    let nodes = ops
        .iter()
        .enumerate()
        .map(|(index, op)| {
            let stage = outputs
                .iter()
                .position(|output| index <= *output)
                .ok_or(Error::Ineligible)?;
            let span = source_spans.get(stage).ok_or(Error::Ineligible)?;
            let op = match *op {
                GraphDagOp::Input(slot) => DagOp::Input(slot),
                GraphDagOp::Map {
                    input,
                    op: MapOp::WrappingAdd(constant),
                } => DagOp::WrappingAdd { input, constant },
                GraphDagOp::Map {
                    input,
                    op: MapOp::WrappingMul(constant),
                } => DagOp::WrappingMul { input, constant },
                GraphDagOp::Zip {
                    left,
                    right,
                    op: ZipOp::WrappingAdd,
                } => DagOp::ZipAdd { left, right },
                GraphDagOp::Zip {
                    left,
                    right,
                    op: ZipOp::WrappingMul,
                } => DagOp::ZipMul { left, right },
            };
            Ok(DagNode {
                op,
                span: Span {
                    start: span.start,
                    end: span.end,
                },
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let first = source_spans.first().ok_or(Error::Ineligible)?;
    let last = source_spans.last().ok_or(Error::Ineligible)?;
    let blocks = length.div_ceil(256).clamp(1, 1024);
    Ok(Ir {
        version: invar_device::IR_VERSION,
        span: Span {
            start: first.start,
            end: last.end,
        },
        source_identity: *source_identity,
        graph_revision: *graph_revision,
        input: *input_buffers.first().ok_or(Error::Ineligible)?,
        output: *output_buffers.last().ok_or(Error::Ineligible)?,
        length,
        geometry: Geometry { blocks, lanes: 256 },
        lane_mask: LaneMask::IndexLtLength,
        barrier: Barrier::None,
        requires_barrier: false,
        transfer: Transfer::Owned,
        numeric: Numeric::WrappingU32,
        completion: true,
        ops: Vec::new(),
        dag: Some(MapDag {
            inputs: input_buffers,
            outputs: output_buffers,
            nodes,
            stage_outputs: outputs.clone(),
        }),
        collective: None,
    })
}
