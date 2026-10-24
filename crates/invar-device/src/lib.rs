#![forbid(unsafe_code)]

//! Versioned physical plans shared by the CUDA and Metal emitters.

pub const IR_VERSION: u32 = 1;

/// Two lowercase hex digits per byte.
pub fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        result.push(DIGITS[(byte >> 4) as usize] as char);
        result.push(DIGITS[(byte & 15) as usize] as char);
    }
    result
}

/// GNET entry storage ceiling from the 2^28 device caps policy.
pub const MAX_GNET_STORAGE_BYTES: usize = 8 * 1024 * 1024 * 1024;
/// Generated GNET source only; ordinary device source keeps its smaller cap.
pub const MAX_GNET_SOURCE_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_ELEMENTS: u32 = 1_000_000;
pub const MAX_OPS: usize = 16;
pub const MAX_SOURCE_BYTES: usize = 16 * 1024;
pub const MAX_DAG_ELEMENTS: u32 = 1 << 28;
pub const MAX_DAG_OPS: usize = 64;
pub const MAX_DAG_TRANSFER_BYTES: u64 = 8 * 1024 * 1024 * 1024;
/// MSL buffer argument indices are 0..=30 on Apple GPUs.
pub const METAL_MAX_BUFFERS: usize = 31;

pub mod collective;
pub mod contraction;
pub mod f32_ops;
pub mod f32_probe;
pub mod floating;
pub mod fsum;
pub mod gemm;
pub mod indexed;
pub mod moves;
mod msl;
mod quarantine;
mod resources;
pub mod segmented;
pub mod sorting;
pub mod spatial;
pub mod stencil;
pub mod stencil_u32;
pub mod summary;
pub mod target;
pub mod tc;
pub mod transpose;
pub mod v2;
pub use collective::{Coefficient, Collective, CollectiveKind, PrefixMode, ReduceLaw};
pub use msl::{METAL_SCAN_CHUNK_BLOCKS, PROBE_FUNCTION, PROBE_MSL, emit_msl, metal_functions};
pub use quarantine::Quarantine;
pub use resources::{CostFacts, collective_workspace};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddressSpace {
    Device,
    Host,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    ContiguousU32,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transfer {
    Owned,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Numeric {
    WrappingU32,
    CheckedU32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneMask {
    IndexLtLength,
    All,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Barrier {
    None,
    Uniform,
    Conditional,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    WrappingAdd(u32),
    WrappingMul(u32),
    CheckedAdd(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DagOp {
    Input(u32),
    WrappingAdd { input: usize, constant: u32 },
    WrappingMul { input: usize, constant: u32 },
    ZipAdd { left: usize, right: usize },
    ZipMul { left: usize, right: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DagNode {
    pub op: DagOp,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MapDag {
    pub inputs: Vec<Buffer>,
    pub outputs: Vec<Buffer>,
    pub nodes: Vec<DagNode>,
    pub stage_outputs: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Buffer {
    pub id: u32,
    pub elements: u32,
    pub address: AddressSpace,
    pub layout: Layout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub blocks: u32,
    pub lanes: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Provenance {
    pub source_identity: u64,
    pub graph_revision: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Capabilities {
    Cuda {
        major: u32,
        minor: u32,
        sm: u32,
    },
    Metal {
        apple_family: u32,
        msl_major: u32,
        msl_minor: u32,
    },
}

impl Capabilities {
    pub const fn cuda_12_9_sm89() -> Self {
        Self::Cuda {
            major: 12,
            minor: 9,
            sm: 89,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ir {
    pub version: u32,
    pub span: Span,
    pub source_identity: u64,
    pub graph_revision: u64,
    pub input: Buffer,
    pub output: Buffer,
    pub length: u32,
    pub geometry: Geometry,
    pub lane_mask: LaneMask,
    pub barrier: Barrier,
    pub requires_barrier: bool,
    pub transfer: Transfer,
    pub numeric: Numeric,
    pub completion: bool,
    pub ops: Vec<Op>,
    /// The same physical map contract for multi-input and multi-output graphs.
    /// `None` retains the frozen M3-02 single-input map encoding.
    pub dag: Option<MapDag>,
    /// A hierarchical scalar reduction or ordered affine scan over the same
    /// verified source graph. Mutually exclusive with map operations.
    pub collective: Option<Collective>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Version,
    Source,
    Graph,
    Layout,
    Bounds,
    Alias,
    Geometry,
    Mask,
    Barrier,
    Transfer,
    Numeric,
    Completion,
    Operations,
    Toolchain,
    SourceLimit,
    ResourceLimit(&'static str),
}

/// Only `verify` can construct this capability to emit and launch device code.
pub struct Verified {
    ir: Ir,
    caps: Capabilities,
    device_ir: v2::Verified,
}

impl Verified {
    pub fn ir(&self) -> &Ir {
        &self.ir
    }
    pub fn capabilities(&self) -> Capabilities {
        self.caps
    }
    pub fn device_ir(&self) -> &v2::Verified {
        &self.device_ir
    }
}

/// MSL buffer arguments the widest emitted kernel for `ir` binds.
pub fn metal_buffer_slots(ir: &Ir) -> usize {
    if let Some(dag) = &ir.dag {
        // Every input and output, then the length.
        return dag.inputs.len() + dag.outputs.len() + 1;
    }
    match &ir.collective {
        // Scan writeback: inputs, summaries, prefix, output, seed, n, blocks, base.
        Some(Collective {
            kind: CollectiveKind::AffineScan { .. } | CollectiveKind::PairReduction { .. },
            inputs,
            ..
        }) => inputs.len() + 7,
        // Block pass: input, summaries, n.
        Some(_) => 3,
        // Frozen single map: input, output, n.
        None => 3,
    }
}

pub fn verify(ir: Ir, origin: Provenance, caps: Capabilities) -> Result<Verified, Error> {
    if ir.version != IR_VERSION {
        return Err(Error::Version);
    }
    if ir.span.start >= ir.span.end || ir.source_identity == 0 {
        return Err(Error::Source);
    }
    if ir.source_identity != origin.source_identity || ir.graph_revision != origin.graph_revision {
        return Err(Error::Graph);
    }
    if let Some(collective) = &ir.collective {
        collective::verify_collective(&ir, collective, caps)?;
    } else if let Some(dag) = &ir.dag {
        verify_dag(&ir, dag)?;
        // Defence in depth: Metal planning refuses these regions first.
        if matches!(caps, Capabilities::Metal { .. }) && metal_buffer_slots(&ir) > METAL_MAX_BUFFERS
        {
            return Err(Error::Operations);
        }
    } else {
        if ir.input.address != AddressSpace::Device
            || ir.output.address != AddressSpace::Device
            || ir.input.layout != Layout::ContiguousU32
            || ir.output.layout != Layout::ContiguousU32
        {
            return Err(Error::Layout);
        }
        if ir.input.id == ir.output.id {
            return Err(Error::Alias);
        }
        if ir.length > MAX_ELEMENTS
            || ir.input.elements < ir.length
            || ir.output.elements < ir.length
        {
            return Err(Error::Bounds);
        }
    }
    if ir.geometry.blocks == 0
        || ir.geometry.blocks
            > if ir.collective.is_some() {
                65_536
            } else {
                65_535
            }
        || ir.geometry.lanes == 0
        || ir.geometry.lanes > 1024
        || !ir.geometry.lanes.is_power_of_two()
        || (ir.dag.is_none()
            && ir.collective.is_none()
            && u64::from(ir.geometry.blocks) * u64::from(ir.geometry.lanes) < u64::from(ir.length))
    {
        return Err(Error::Geometry);
    }
    if ir.lane_mask != LaneMask::IndexLtLength {
        return Err(Error::Mask);
    }
    if (ir.collective.is_some() && (!ir.requires_barrier || ir.barrier != Barrier::Uniform))
        || (ir.collective.is_none() && (ir.requires_barrier || ir.barrier != Barrier::None))
    {
        return Err(Error::Barrier);
    }
    if ir.transfer != Transfer::Owned {
        return Err(Error::Transfer);
    }
    if ir.numeric != Numeric::WrappingU32 {
        return Err(Error::Numeric);
    }
    if !ir.completion {
        return Err(Error::Completion);
    }
    if ir.dag.is_none()
        && ir.collective.is_none()
        && (ir.ops.is_empty()
            || ir.ops.len() > MAX_OPS
            || ir.ops.iter().any(|op| matches!(op, Op::CheckedAdd(_))))
    {
        return Err(Error::Operations);
    }
    match caps {
        Capabilities::Cuda {
            major: 12,
            minor,
            sm,
        } if minor >= 4 && sm >= 50 => {}
        Capabilities::Metal {
            apple_family,
            msl_major,
            msl_minor,
        } if apple_family >= 7 && (msl_major, msl_minor) == (3, 1) => {}
        _ => return Err(Error::Toolchain),
    }
    let device_ir =
        v2::verify_metadata(v2::legacy(&ir), origin, caps).map_err(|_| Error::Layout)?;
    Ok(Verified {
        ir,
        caps,
        device_ir,
    })
}

fn verify_dag(ir: &Ir, dag: &MapDag) -> Result<(), Error> {
    if !ir.ops.is_empty()
        || dag.inputs.is_empty()
        || dag.inputs.len() > 16
        || dag.outputs.is_empty()
        || dag.outputs.len() > 16
        || dag.outputs.len() != dag.stage_outputs.len()
        || dag.nodes.is_empty()
        || dag.nodes.len() > MAX_DAG_OPS
        || ir.input != dag.inputs[0]
        || ir.output != *dag.outputs.last().ok_or(Error::Operations)?
    {
        return Err(Error::Operations);
    }
    if ir.length > MAX_DAG_ELEMENTS {
        return Err(Error::ResourceLimit("map exceeds 2^28 elements"));
    }
    let transfer = u64::from(ir.length)
        .checked_mul(
            u64::try_from(dag.inputs.len() + dag.outputs.len()).map_err(|_| Error::Bounds)?,
        )
        .and_then(|n| n.checked_mul(4))
        .ok_or(Error::ResourceLimit("map byte count overflow"))?;
    if transfer > MAX_DAG_TRANSFER_BYTES {
        return Err(Error::ResourceLimit("map exceeds 8 GiB transfer budget"));
    }
    let mut ids = std::collections::BTreeSet::new();
    for buffer in dag.inputs.iter().chain(&dag.outputs) {
        if buffer.address != AddressSpace::Device || buffer.layout != Layout::ContiguousU32 {
            return Err(Error::Layout);
        }
        if buffer.elements < ir.length {
            return Err(Error::Bounds);
        }
        if !ids.insert(buffer.id) {
            return Err(Error::Alias);
        }
    }
    for (index, node) in dag.nodes.iter().enumerate() {
        if node.span.start >= node.span.end
            || node.span.start < ir.span.start
            || node.span.end > ir.span.end
        {
            return Err(Error::Source);
        }
        let prior = |value: usize| value < index;
        match node.op {
            DagOp::Input(slot) if usize::try_from(slot).is_ok_and(|s| s < dag.inputs.len()) => {}
            DagOp::WrappingAdd { input, .. } | DagOp::WrappingMul { input, .. } if prior(input) => {
            }
            DagOp::ZipAdd { left, right } | DagOp::ZipMul { left, right }
                if prior(left) && prior(right) => {}
            _ => return Err(Error::Operations),
        }
    }
    if dag
        .stage_outputs
        .iter()
        .any(|index| *index >= dag.nodes.len())
        || !dag.stage_outputs.windows(2).all(|pair| pair[0] < pair[1])
        || dag.stage_outputs.last() != Some(&(dag.nodes.len() - 1))
    {
        return Err(Error::Operations);
    }
    Ok(())
}

pub fn emit_cuda(plan: &Verified) -> Result<String, Error> {
    if !matches!(plan.caps, Capabilities::Cuda { .. }) {
        return Err(Error::Toolchain);
    }
    if let Some(collective) = &plan.ir.collective {
        return collective::emit_cuda(&plan.ir, collective);
    }
    if let Some(dag) = &plan.ir.dag {
        return emit_dag_cuda(dag);
    }
    let mut source = String::from(
        "extern \"C\" __global__ void invar_map_v1(const unsigned int* input0, unsigned int* output0, unsigned int n) {\n  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n  if (i < n) {\n    unsigned int v = input0[i];\n",
    );
    for op in &plan.ir.ops {
        match op {
            Op::WrappingAdd(n) => source.push_str(&format!("    v = v + {n}u;\n")),
            Op::WrappingMul(n) => source.push_str(&format!("    v = v * {n}u;\n")),
            Op::CheckedAdd(_) => return Err(Error::Operations),
        }
    }
    source.push_str("    output0[i] = v;\n  }\n}\n");
    if source.len() > MAX_SOURCE_BYTES {
        return Err(Error::SourceLimit);
    }
    Ok(source)
}

fn emit_dag_cuda(dag: &MapDag) -> Result<String, Error> {
    use std::fmt::Write;

    let mut source = String::from("extern \"C\" __global__ void invar_map_v1(");
    for index in 0..dag.inputs.len() {
        if index > 0 {
            source.push_str(", ");
        }
        write!(source, "const unsigned int* input{index}").map_err(|_| Error::SourceLimit)?;
    }
    for index in 0..dag.outputs.len() {
        source.push_str(", ");
        write!(source, "unsigned int* output{index}").map_err(|_| Error::SourceLimit)?;
    }
    source.push_str(", unsigned int n) {\n  unsigned int base = blockIdx.x * blockDim.x + threadIdx.x;\n  unsigned int stride = gridDim.x * blockDim.x;\n  for (unsigned int i = base; i < n; i += stride) {\n");
    for (index, node) in dag.nodes.iter().enumerate() {
        match node.op {
            DagOp::Input(slot) => {
                writeln!(source, "    unsigned int v{index} = input{slot}[i];")
                    .map_err(|_| Error::SourceLimit)?;
            }
            DagOp::WrappingAdd { input, constant } => {
                writeln!(
                    source,
                    "    unsigned int v{index} = v{input} + {constant}u;"
                )
                .map_err(|_| Error::SourceLimit)?;
            }
            DagOp::WrappingMul { input, constant } => {
                writeln!(
                    source,
                    "    unsigned int v{index} = v{input} * {constant}u;"
                )
                .map_err(|_| Error::SourceLimit)?;
            }
            DagOp::ZipAdd { left, right } => {
                writeln!(source, "    unsigned int v{index} = v{left} + v{right};")
                    .map_err(|_| Error::SourceLimit)?;
            }
            DagOp::ZipMul { left, right } => {
                writeln!(source, "    unsigned int v{index} = v{left} * v{right};")
                    .map_err(|_| Error::SourceLimit)?;
            }
        }
    }
    for (index, stage) in dag.stage_outputs.iter().enumerate() {
        writeln!(source, "    output{index}[i] = v{stage};").map_err(|_| Error::SourceLimit)?;
    }
    source.push_str("  }\n}\n");
    if source.len() > MAX_SOURCE_BYTES {
        return Err(Error::SourceLimit);
    }
    Ok(source)
}

/// Stable FNV-1a identity for an emitted artifact, independent of Rust hashing.
pub fn source_hash(source: &str) -> u64 {
    source
        .as_bytes()
        .iter()
        .fold(0xcbf29ce484222325u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        })
}

/// Check a complete submission before allocating any device storage. Capacity
/// is a snapshot; backends must also handle allocation failure after this check.
pub fn check_memory(
    buffer_bytes: u64,
    total_bytes: u64,
    max_buffer: u64,
    available: u64,
) -> Result<(), &'static str> {
    if buffer_bytes > max_buffer {
        Err("buffer exceeds device maxBufferLength")
    } else if total_bytes > available {
        Err("submission exceeds available device memory or recommended working set")
    } else {
        Ok(())
    }
}

/// Shared bounded integer binary32 primitives for emitted CPU/CUDA/Metal text.
pub const EXACT_F32: &str = include_str!("f32-exact.inc");
