//! Bounded u32 irregular device plans. Physical verification is distinct from
//! source recognition, semantic fuel and source-state publication authority.
//! Bound owns immutable guarded inputs; no imported injectivity flag is trusted.
use crate::{AddressSpace, Buffer, Capabilities, Layout, Span};

pub const VERSION: u32 = 1;
pub const MAX_ELEMENTS: u32 = 1 << 28;
/// Dense adjacency cells, rather than vertices, are the benchmark size.
pub const MAX_VERTICES: u32 = 1 << 14;
pub const MAX_ROUNDS: u32 = 16;
pub const HISTOGRAM_LAW: &str = "wrapping-u32-histogram-v1";
pub const INDEXED_KERNEL: &str = "invar_indexed_v1";
pub const EXPAND_KERNEL: &str = "invar_frontier_expand_v1";
pub const COMPACT_KERNEL: &str = "invar_frontier_compact_v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    Gather,
    InjectiveScatter,
    /// Identity zero, duplicates combine by addition modulo 2^32.
    Histogram,
    /// Dense source adjacency; exact ordered masks preserve source observations.
    Frontier {
        vertices: u32,
        rounds: u32,
        capacity: u32,
    },
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ir {
    pub version: u32,
    pub source_identity: u64,
    pub graph_revision: u64,
    pub source_contract: [u8; 32],
    pub span: Span,
    pub operation: Operation,
    pub values: Buffer,
    pub indices: Buffer,
    pub output: Buffer,
    pub law: Option<String>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Version,
    Source,
    Binding,
    Layout,
    Alias,
    Shape,
    Law,
    Limit,
    Capabilities,
    Index {
        iteration: usize,
        operand: &'static str,
    },
    Duplicate {
        iteration: usize,
        index: u32,
    },
    Capacity {
        round: u32,
        required: u32,
        capacity: u32,
    },
    Allocation,
}
#[derive(Debug)]
pub struct Verified {
    ir: Ir,
    caps: Capabilities,
}
impl Verified {
    pub fn ir(&self) -> &Ir {
        &self.ir
    }
    pub fn capabilities(&self) -> Capabilities {
        self.caps
    }
    /// Validate the entire indirect footprint before copying or submitting.
    /// Failed guards publish nothing; a source host must retain its trap order.
    pub fn bind(&self, values: &[u32], indices: &[u32], start: u32) -> Result<Bound, Error> {
        if values.len() != self.ir.values.elements as usize
            || indices.len() != self.ir.indices.elements as usize
        {
            return Err(Error::Shape);
        }
        let out = self.ir.output.elements as usize;
        match self.ir.operation {
            Operation::Frontier { vertices, .. } => {
                if start >= vertices {
                    return Err(Error::Index {
                        iteration: 0,
                        operand: "start",
                    });
                }
            }
            op => {
                let mut seen = if op == Operation::InjectiveScatter {
                    zeros(out)?
                } else {
                    Vec::new()
                };
                for (iteration, &index) in indices.iter().enumerate() {
                    // Scatter RHS is evaluated before its destination index.
                    if op == Operation::InjectiveScatter && iteration >= values.len() {
                        return Err(Error::Index {
                            iteration,
                            operand: "values",
                        });
                    }
                    let limit = if op == Operation::Gather {
                        values.len()
                    } else {
                        out
                    };
                    if index as usize >= limit {
                        return Err(Error::Index {
                            iteration,
                            operand: "indices",
                        });
                    }
                    // Histogram's RHS reads out[ix[i]] before a[i].
                    if op == Operation::Histogram && iteration >= values.len() {
                        return Err(Error::Index {
                            iteration,
                            operand: "values",
                        });
                    }
                    if op == Operation::InjectiveScatter {
                        let slot = seen.get_mut(index as usize).ok_or(Error::Shape)?;
                        if *slot != 0 {
                            return Err(Error::Duplicate { iteration, index });
                        }
                        *slot = 1;
                    }
                }
            }
        }
        Ok(Bound {
            plan: Verified {
                ir: self.ir.clone(),
                caps: self.caps,
            },
            values: copy(values)?,
            indices: copy(indices)?,
            start,
        })
    }
}
/// Owning immutable live guards. Device output is always a fresh allocation.
#[derive(Debug)]
pub struct Bound {
    plan: Verified,
    values: Vec<u32>,
    indices: Vec<u32>,
    start: u32,
}
impl Bound {
    pub fn plan(&self) -> &Verified {
        &self.plan
    }
    pub fn values(&self) -> &[u32] {
        &self.values
    }
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }
    pub fn start(&self) -> u32 {
        self.start
    }
    /// Per-buffer allocation sizes, including initialized empty ABI dummies.
    /// Verification bounds every product to at most one GiB.
    pub fn buffer_bytes(&self) -> [usize; 7] {
        let n = match self.plan.ir.operation {
            Operation::Frontier { vertices, .. } => vertices as usize,
            _ => 0,
        };
        [
            self.values.len().max(1) * 4,
            self.indices.len().max(1) * 4,
            (self.plan.ir.output.elements as usize).max(1) * 4,
            n.max(1) * 4,
            n.max(1) * 4,
            n.max(1) * 4,
            8,
        ]
    }
    pub fn transfer_bytes(&self) -> usize {
        let n = match self.plan.ir.operation {
            Operation::Frontier { vertices, .. } => vertices as usize,
            _ => 0,
        };
        let output = self.plan.ir.output.elements as usize;
        let metal = matches!(self.plan.caps, Capabilities::Metal { .. });
        let round_copies = if metal {
            match self.plan.ir.operation {
                Operation::Frontier { rounds, .. } => rounds as usize * 2,
                _ => 1,
            }
        } else {
            0
        };
        // Initialized ABI buffers and readback. Metal also copies a scalar
        // into each phase's constant buffer, but reads no empty output dummy.
        (self.values.len().max(1)
            + self.indices.len().max(1)
            + output.max(1)
            + if metal { output } else { output.max(1) }
            + n.max(1) * 3
            + 4
            + round_copies)
            * 4
    }
}
/// Fallible owning copy for guarded inputs, staging and completed readback.
pub fn copy(v: &[u32]) -> Result<Vec<u32>, Error> {
    let mut out = Vec::new();
    out.try_reserve_exact(v.len())
        .map_err(|_| Error::Allocation)?;
    out.extend_from_slice(v);
    Ok(out)
}
pub fn zeros(n: usize) -> Result<Vec<u32>, Error> {
    let mut out = Vec::new();
    out.try_reserve_exact(n).map_err(|_| Error::Allocation)?;
    out.resize(n, 0);
    Ok(out)
}
/// `expected` is independently regenerated by source lowering, not a saved
/// proof flag. This catches any mutation of the physical/source binding.
pub fn verify(ir: Ir, expected: &Ir, caps: Capabilities) -> Result<Verified, Error> {
    if ir.version != VERSION {
        return Err(Error::Version);
    }
    if ir.source_identity == 0 || ir.source_contract == [0; 32] || ir.span.start >= ir.span.end {
        return Err(Error::Source);
    }
    if &ir != expected {
        return Err(Error::Binding);
    }
    for b in [&ir.values, &ir.indices, &ir.output] {
        if b.address != AddressSpace::Device || b.layout != Layout::ContiguousU32 {
            return Err(Error::Layout);
        }
        if b.elements > MAX_ELEMENTS {
            return Err(Error::Limit);
        }
    }
    if ir.output.id == ir.values.id || ir.output.id == ir.indices.id {
        return Err(Error::Alias);
    }
    if ir.values.id == ir.indices.id && ir.values.elements != ir.indices.elements {
        return Err(Error::Shape);
    }
    if (ir.operation == Operation::Histogram && ir.law.as_deref() != Some(HISTOGRAM_LAW))
        || (ir.operation != Operation::Histogram && ir.law.is_some())
    {
        return Err(Error::Law);
    }
    match ir.operation {
        Operation::Gather if ir.output.elements != ir.indices.elements => return Err(Error::Shape),
        Operation::Frontier {
            vertices,
            rounds,
            capacity,
        } => {
            if vertices > MAX_VERTICES || rounds > MAX_ROUNDS {
                return Err(Error::Limit);
            }
            if capacity != vertices {
                return Err(Error::Shape);
            }
            let cells = vertices.checked_mul(vertices).ok_or(Error::Limit)?;
            let words = rounds
                .checked_add(1)
                .and_then(|r| vertices.checked_mul(r))
                .ok_or(Error::Limit)?;
            if ir.values.elements < cells || ir.indices.elements != 0 || ir.output.elements != words
            {
                return Err(Error::Shape);
            }
        }
        _ => {}
    }
    match caps {
        Capabilities::Cuda {
            major: 12,
            minor: 4..=9,
            sm,
        } if matches!(
            sm,
            50 | 52 | 53 | 60 | 61 | 62 | 70 | 72 | 75 | 80 | 86 | 87 | 89 | 90
        ) || minor_supports_blackwell(caps) => {}
        Capabilities::Metal {
            apple_family: 7..=10,
            msl_major: 3,
            msl_minor: 1,
        } => {}
        _ => return Err(Error::Capabilities),
    }
    Ok(Verified { ir, caps })
}
fn minor_supports_blackwell(caps: Capabilities) -> bool {
    matches!(
        caps,
        Capabilities::Cuda {
            minor: 9,
            sm: 100 | 101 | 120,
            ..
        }
    )
}

/// One generated algorithm and ABI, with only target syntax and atomic spelling
/// differing. Frontier phase ordering is enforced by the runtime, never spins.
fn emit(p: &Verified, metal: bool) -> Result<String, Error> {
    if metal != matches!(p.caps, Capabilities::Metal { .. }) {
        return Err(Error::Capabilities);
    }
    let ir = &p.ir;
    let mut text = format!(
        "// invar-indexed-u32-v1 caps={:?} source={} graph={} contract={:?} span={:?} operation={:?} law={:?}\n",
        p.caps,
        ir.source_identity,
        ir.graph_revision,
        ir.source_contract,
        ir.span,
        ir.operation,
        ir.law
    );
    text.push_str(&format!(
        "// values={:?} indices={:?} output={:?} numeric=wrapping-u32\n",
        ir.values, ir.indices, ir.output
    ));
    if metal {
        text.push_str("#include <metal_stdlib>\nusing namespace metal;\n");
    }
    let signature = |name: &str, histogram: bool| -> String {
        if metal {
            format!(
                "kernel void {name}(device const uint* values [[buffer(0)]], device const uint* indices [[buffer(1)]], device {}* output [[buffer(2)]], device uint* frontier [[buffer(3)]], device uint* next [[buffer(4)]], device uint* queue [[buffer(5)]], device uint* control [[buffer(6)]], constant uint& round [[buffer(7)]], uint tid [[thread_position_in_grid]]) {{\n",
                if histogram { "atomic_uint" } else { "uint" }
            )
        } else {
            format!(
                "extern \"C\" __global__ void {name}(const unsigned int* values, const unsigned int* indices, unsigned int* output, unsigned int* frontier, unsigned int* next, unsigned int* queue, unsigned int* control, unsigned int round) {{\n  const unsigned int tid=blockIdx.x*blockDim.x+threadIdx.x;\n"
            )
        }
    };
    match ir.operation {
        Operation::Frontier {
            vertices: n,
            rounds,
            capacity: _,
        } => {
            // Freeze seen/frontier during expansion. In ascending source order,
            // v has been marked before u expands iff frontier[v] && v <= u
            // (or it was seen at round entry). This predicate preserves even
            // self/cross edges and the observable next masks, without a race.
            text.push_str(&signature(EXPAND_KERNEL, false));
            text.push_str(&format!("  if (round>={rounds}u || control[1]!=0) return;\n  if (tid==0) control[0]=0;\n  unsigned int u=tid/64u;\n  if (u>={n}u || frontier[u]!=1 || output[u]!=0) return;\n  for (unsigned int v=tid%64u;v<{n}u;v+=64u) {{\n    if (values[u*{n}u+v]==1 && output[v]==0 && (frontier[v]==0 || v>u)) {{\n"));
            if metal {
                text.push_str("      atomic_store_explicit((device atomic_uint*)&next[v],1u,memory_order_relaxed);\n");
            } else {
                text.push_str("      atomicExch(&next[v],1u);\n");
            }
            text.push_str("    }\n  }\n}\n");
            text.push_str(&signature(COMPACT_KERNEL, false));
            // A bitmap is the bounded owned worklist. decode enumerates it in
            // ascending order, so a device queue/prefix scan is unnecessary.
            // Exactly n lanes can contribute one bit: capacity == n is proved
            // by verify, and the counter cannot overflow or exceed capacity.
            text.push_str(&format!("  if (tid>={n}u || round>={rounds}u || control[1]!=0) return;\n  unsigned int bit=next[tid];\n  if (frontier[tid]==1) output[tid]=1;\n  frontier[tid]=bit;\n  output[{n}u*(round+1u)+tid]=bit;\n  next[tid]=0;\n  if (bit==1) {{\n"));
            if metal {
                text.push_str("    atomic_fetch_add_explicit((device atomic_uint*)&control[0],1u,memory_order_relaxed);\n");
            } else {
                text.push_str("    atomicAdd(&control[0],1u);\n");
            }
            text.push_str("  }\n}\n");
        }
        op => {
            text.push_str(&signature(INDEXED_KERNEL, op == Operation::Histogram));
            let (blocks, lanes) = launch_dimensions(p);
            text.push_str(&format!(
                "  for (unsigned int i=tid;i<{}u;i+={}u) {{\n",
                ir.indices.elements,
                blocks * lanes
            ));
            match op {
                Operation::Gather => text.push_str("  output[i]=values[indices[i]];\n"),
                Operation::InjectiveScatter => text.push_str("  output[indices[i]]=values[i];\n"),
                Operation::Histogram if metal => text.push_str("  atomic_fetch_add_explicit(&output[indices[i]],values[i],memory_order_relaxed);\n"),
                Operation::Histogram => text.push_str("  atomicAdd(&output[indices[i]],values[i]);\n"),
                _ => return Err(Error::Shape),
            }
            text.push_str("  }\n}\n");
        }
    }
    if text.len() > crate::MAX_SOURCE_BYTES {
        return Err(Error::Limit);
    }
    Ok(text)
}
pub fn emit_cuda(p: &Verified) -> Result<String, Error> {
    emit(p, false)
}
pub fn emit_msl(p: &Verified) -> Result<String, Error> {
    emit(p, true)
}

/// Grid-stride loops keep each dimension within the supported targets' minimum
/// grid limit. No shared/threadgroup storage is used; runtimes check lane limits.
pub fn launch_dimensions(p: &Verified) -> (u32, u32) {
    match p.ir.operation {
        Operation::Frontier { vertices, .. } => (vertices.max(1), 64),
        _ => (p.ir.indices.elements.div_ceil(256).clamp(1, 65_535), 256),
    }
}
/// Expansion uses a cooperative group per adjacency row; mask publication
/// needs one lane per vertex. Both targets consume these same phase shapes.
pub fn phase_launch_dimensions(p: &Verified, kernel: usize) -> (u32, u32) {
    if let Operation::Frontier { vertices, .. } = p.ir.operation
        && kernel == 1
    {
        (vertices.div_ceil(64).max(1), 64)
    } else {
        launch_dimensions(p)
    }
}

/// Decode only completed private output. Frontier receipts expose stable masks
/// and ascending compaction maps; bounded rounds may intentionally truncate.
#[derive(Debug, PartialEq, Eq)]
pub struct Output {
    pub values: Vec<u32>,
    pub frontier_masks: Vec<Vec<u32>>,
    pub compaction_maps: Vec<Vec<u32>>,
}
pub fn decode(p: &Verified, words: Vec<u32>, status: &[u32]) -> Result<Output, Error> {
    if words.len() != p.ir.output.elements as usize || status.len() != 2 {
        return Err(Error::Shape);
    }
    if status[1] != 0 {
        return Err(Error::Limit);
    }
    if let Operation::Frontier {
        vertices,
        rounds,
        capacity,
    } = p.ir.operation
    {
        if status[0] > capacity || words.iter().any(|&x| x > 1) {
            return Err(Error::Shape);
        }
        let n = vertices as usize;
        if n == 0 {
            return Ok(Output {
                values: vec![],
                frontier_masks: vec![vec![]; rounds as usize],
                compaction_maps: vec![vec![]; rounds as usize],
            });
        }
        let values = copy(&words[..n])?;
        let frontier_masks: Vec<_> = words[n..]
            .chunks_exact(n)
            .map(|mask| mask.to_vec())
            .collect();
        let compaction_maps = frontier_masks
            .iter()
            .map(|mask| {
                mask.iter()
                    .enumerate()
                    .filter_map(|(i, &bit)| (bit == 1).then_some(i as u32))
                    .collect()
            })
            .collect();
        return Ok(Output {
            values,
            frontier_masks,
            compaction_maps,
        });
    }
    Ok(Output {
        values: words,
        frontier_masks: vec![],
        compaction_maps: vec![],
    })
}

/// Completed device observation, not a semantic fuel/publication permit.
#[derive(Debug)]
pub struct Receipt {
    pub target: &'static str,
    pub reason: &'static str,
    pub output: Output,
    pub source_hash: u64,
    pub transfer_bytes: usize,
    pub transfer_mode: &'static str,
    pub completed_phases: u32,
}
pub fn kernel_names(p: &Verified) -> Vec<&'static str> {
    if matches!(p.ir.operation, Operation::Frontier { .. }) {
        vec![EXPAND_KERNEL, COMPACT_KERNEL]
    } else {
        vec![INDEXED_KERNEL]
    }
}
pub fn schedule(p: &Verified) -> Vec<(u32, usize)> {
    match p.ir.operation {
        Operation::Frontier { rounds, .. } => (0..rounds).flat_map(|r| [(r, 0), (r, 1)]).collect(),
        _ => vec![(0, 0)],
    }
}
/// Fresh initialized ABI buffers. Slots 3/4 are owned frontier bitmaps, slot
/// 5 retains the original queue ABI allocation, and slot 6 is [length, error].
/// The parallel schedule uses bitmaps and decodes stable maps after completion;
/// no device access depends on host drops.
pub fn initial_buffers(b: &Bound) -> Result<[Vec<u32>; 7], Error> {
    let n = match b.plan.ir.operation {
        Operation::Frontier { vertices, .. } => vertices as usize,
        _ => 0,
    };
    let mut frontier = zeros(n.max(1))?;
    let mut queue = zeros(n.max(1))?;
    let mut control = vec![0, 0];
    if n != 0 {
        frontier[b.start as usize] = 1;
        queue[0] = b.start;
        control[0] = 1;
    }
    Ok([
        if b.values.is_empty() {
            zeros(1)?
        } else {
            copy(&b.values)?
        },
        if b.indices.is_empty() {
            zeros(1)?
        } else {
            copy(&b.indices)?
        },
        zeros((b.plan.ir.output.elements as usize).max(1))?,
        frontier,
        zeros(n.max(1))?,
        queue,
        control,
    ])
}

#[cfg(test)]
mod tests {
    #[test]
    fn frontier_frozen_seen_predicate_preserves_all_three_vertex_rounds() {
        // Exhaust every directed graph (including self edges), entry seen set
        // and frontier. This checks the source-order equivalence used by both
        // emitters, including discoveries of later same-round active vertices.
        for edges in 0u32..512 {
            for seen in 0u32..8 {
                for front in 0u32..8 {
                    let mut serial_seen = seen;
                    let mut serial_next = 0;
                    let mut parallel_next = 0;
                    for u in 0..3 {
                        if front & (1 << u) != 0 && serial_seen & (1 << u) == 0 {
                            serial_seen |= 1 << u;
                            for v in 0..3 {
                                if edges & (1 << (u * 3 + v)) != 0 && serial_seen & (1 << v) == 0 {
                                    serial_next |= 1 << v;
                                }
                            }
                        }
                        if front & (1 << u) != 0 && seen & (1 << u) == 0 {
                            for v in 0..3 {
                                if edges & (1 << (u * 3 + v)) != 0
                                    && seen & (1 << v) == 0
                                    && (front & (1 << v) == 0 || v > u)
                                {
                                    parallel_next |= 1 << v;
                                }
                            }
                        }
                    }
                    assert_eq!(
                        parallel_next, serial_next,
                        "edges={edges} seen={seen} frontier={front}"
                    );
                    assert_eq!(serial_seen, seen | front);
                }
            }
        }
    }
}
