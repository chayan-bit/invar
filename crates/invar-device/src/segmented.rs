//! Verified segmented map, segmented reduction and stable compaction (M4-04).
//!
//! One immutable IR drives the CUDA source, the Metal source and the Rust
//! reference model. All arithmetic is exact wrapping u32. Sizes, operations,
//! the reduction law and the compaction predicate and cutoff are baked into the
//! emitted source; offsets and inputs are runtime buffers that host admission
//! ([`admit_inputs`]) checks before any upload. Verification grants no source
//! extraction, loan, fuel or publication authority, and no launch by itself.

use crate::{Capabilities, Provenance, Span};
use std::fmt::Write;

pub const VERSION: u32 = 1;
/// Threads per block (CUDA) or threadgroup (Metal) for every kernel.
pub const LANES: u32 = 256;
/// Grid cap for the grid-stride map and reduction-carry kernels.
pub const MAX_BLOCKS: u32 = 4096;
/// Physical IR bounds for the M5-01 size sweep. Shared source/runtime
/// budgets remain independent admission gates.
pub const MAX_ELEMENTS: u32 = 1 << 28;
pub const MAX_SEGMENTS: u32 = 1 << 28;
pub const MAX_MAP_OPS: usize = crate::MAX_OPS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapOp {
    WrappingAdd(u32),
    WrappingMul(u32),
    /// Wrapping add of the element's segment index.
    AddSegment,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Law {
    WrappingSum,
    WrappingProduct,
}

impl Law {
    pub fn identity(self) -> u32 {
        match self {
            Self::WrappingSum => 0,
            Self::WrappingProduct => 1,
        }
    }
    pub fn combine(self, a: u32, b: u32) -> u32 {
        match self {
            Self::WrappingSum => a.wrapping_add(b),
            Self::WrappingProduct => a.wrapping_mul(b),
        }
    }
    fn operator(self) -> &'static str {
        match self {
            Self::WrappingSum => "+",
            Self::WrappingProduct => "*",
        }
    }
}

/// Total unsigned comparisons of an element against the cutoff.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Predicate {
    Less,
    LessEqual,
    Equal,
    NotEqual,
    GreaterEqual,
    Greater,
}

impl Predicate {
    pub fn apply(self, value: u32, cutoff: u32) -> bool {
        match self {
            Self::Less => value < cutoff,
            Self::LessEqual => value <= cutoff,
            Self::Equal => value == cutoff,
            Self::NotEqual => value != cutoff,
            Self::GreaterEqual => value >= cutoff,
            Self::Greater => value > cutoff,
        }
    }
    fn operator(self) -> &'static str {
        match self {
            Self::Less => "<",
            Self::LessEqual => "<=",
            Self::Equal => "==",
            Self::NotEqual => "!=",
            Self::GreaterEqual => ">=",
            Self::Greater => ">",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Family {
    /// `output[j] = ops(input[j], segment(j))` for every element.
    Map(Vec<MapOp>),
    /// `output[s] = fold(law, identity, input[offsets[s]..offsets[s + 1]])`.
    Reduce(Law),
    /// Stable selection of `input[j]` with `predicate(input[j], cutoff)`.
    Compact { predicate: Predicate, cutoff: u32 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ir {
    pub version: u32,
    pub span: Span,
    pub source_identity: u64,
    pub graph_revision: u64,
    pub function_contract: [u8; 32],
    pub family: Family,
    pub elements: u32,
    /// Segment count for map/reduce; zero for compaction (no offsets).
    pub segments: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Version,
    Source,
    Graph,
    Contract,
    Bounds,
    Shape,
    Operations,
    Capabilities,
    SourceLimit,
    /// Runtime offsets or input lengths fail host admission.
    Offsets,
}

/// Static proof of one IR for one target. Inputs still need [`admit_inputs`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Verified {
    ir: Ir,
    caps: Capabilities,
    blocks: u32,
    nvrtc_options: Vec<String>,
}

impl Verified {
    pub fn ir(&self) -> &Ir {
        &self.ir
    }
    pub fn capabilities(&self) -> Capabilities {
        self.caps
    }
    /// Primary grid bound. Use `launches` for the actual admitted dispatches.
    pub fn blocks(&self) -> u32 {
        self.blocks
    }
    pub fn nvrtc_options(&self) -> &[String] {
        &self.nvrtc_options
    }
    /// All entry points. `launches` selects the admitted execution schedule.
    pub fn kernels(&self) -> &'static [&'static str] {
        match self.ir.family {
            Family::Map(_) => &["invar_segmented_map_v1", "invar_segmented_map_short_v1"],
            Family::Reduce(_) => &[
                "invar_segmented_reduce_v1",
                "invar_segmented_reduce_tiles_v1",
                "invar_segmented_reduce_carry_v1",
                "invar_segmented_reduce_uniform_v1",
            ],
            Family::Compact { .. } => &[
                "invar_compact_count_v1",
                "invar_compact_scan_v1",
                "invar_compact_carry_v1",
                "invar_compact_scatter_v1",
            ],
        }
    }
    /// Words of each runtime buffer. Reduction adds tile tails; compaction
    /// uses input, output, counts, bases, total, superblock sums and carries.
    pub fn buffer_words(&self) -> Vec<usize> {
        let n = self.ir.elements as usize;
        let s = self.ir.segments as usize;
        let b = self.blocks as usize;
        match self.ir.family {
            Family::Map(_) => vec![n, s + 1, n],
            Family::Reduce(_) => vec![n, s + 1, s, n.div_ceil(LANES as usize).max(1)],
            Family::Compact { .. } => vec![
                n,
                n,
                b,
                b,
                1,
                b.div_ceil(LANES as usize),
                b.div_ceil(LANES as usize),
            ],
        }
    }
    /// The same checked schedule is consumed by CUDA and Metal. The short path
    /// is legal only after inspecting every admitted runtime segment length.
    pub fn launches(&self, input: &[u32], offsets: &[u32]) -> Result<Vec<Launch>, Error> {
        admit_inputs(self, input, offsets)?;
        let short = offsets.windows(2).all(|p| p[1] - p[0] <= LANES);
        let segments = self.ir.segments.div_ceil(LANES).max(1);
        let launch = |kernel, buffers, blocks| Launch {
            kernel,
            buffers,
            blocks,
        };
        Ok(match self.ir.family {
            Family::Map(_) if short => vec![launch(1, &[0, 1, 2][..], segments)],
            Family::Map(_) => vec![launch(0, &[0, 1, 2][..], self.blocks)],
            Family::Reduce(_)
                if short
                    && self.ir.segments != 0
                    && self.ir.elements.is_multiple_of(self.ir.segments)
                    && offsets
                        .windows(2)
                        .all(|p| p[1] - p[0] == self.ir.elements / self.ir.segments) =>
            {
                vec![launch(3, &[0, 2][..], segments)]
            }
            Family::Reduce(_) if short => vec![launch(0, &[0, 1, 2][..], segments)],
            Family::Reduce(_) => vec![
                launch(
                    1,
                    &[0, 1, 2, 3][..],
                    self.ir.elements.div_ceil(LANES).max(1),
                ),
                launch(2, &[1, 3, 2][..], self.ir.segments.clamp(1, MAX_BLOCKS)),
            ],
            Family::Compact { .. } => vec![
                launch(0, &[0, 2][..], self.blocks),
                launch(1, &[2, 3, 5][..], self.blocks.div_ceil(LANES)),
                launch(2, &[5, 6, 4][..], 1),
                launch(3, &[0, 3, 6, 1][..], self.blocks),
            ],
        })
    }
}

/// A verified kernel index, buffer ABI and launch geometry. No host round trip
/// is needed between these ordered dispatches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Launch {
    pub kernel: usize,
    pub buffers: &'static [usize],
    pub blocks: u32,
}

/// CUDA 12.4-12.9 with a listed architecture; Blackwell (sm_100/120) needs 12.8+.
/// Metal needs an Apple7+ GPU and MSL 3.1, as the Metal runtime reports.
fn target(caps: Capabilities) -> Result<Option<u32>, Error> {
    match caps {
        Capabilities::Cuda {
            major: 12,
            minor: minor @ 4..=9,
            sm,
        } if matches!(
            sm,
            50 | 52 | 53 | 60 | 61 | 62 | 70 | 72 | 75 | 80 | 86 | 87 | 89 | 90
        ) || (minor >= 8 && matches!(sm, 100 | 120)) =>
        {
            Ok(Some(sm))
        }
        Capabilities::Metal {
            apple_family: 7..,
            msl_major: 3,
            msl_minor: 1,
        } => Ok(None),
        _ => Err(Error::Capabilities),
    }
}

pub fn verify(ir: Ir, caps: Capabilities, origin: Provenance) -> Result<Verified, Error> {
    if ir.version != VERSION {
        return Err(Error::Version);
    }
    if ir.span.start >= ir.span.end || ir.source_identity == 0 {
        return Err(Error::Source);
    }
    if ir.source_identity != origin.source_identity || ir.graph_revision != origin.graph_revision {
        return Err(Error::Graph);
    }
    if ir.function_contract == [0; 32] {
        return Err(Error::Contract);
    }
    if ir.elements > MAX_ELEMENTS || ir.segments > MAX_SEGMENTS {
        return Err(Error::Bounds);
    }
    let chunks = ir.elements.div_ceil(LANES);
    let blocks = match &ir.family {
        Family::Map(ops) => {
            if ops.is_empty() || ops.len() > MAX_MAP_OPS {
                return Err(Error::Operations);
            }
            // Complete coverage: no elements without segments.
            if ir.segments == 0 && ir.elements != 0 {
                return Err(Error::Shape);
            }
            chunks.min(MAX_BLOCKS)
        }
        Family::Reduce(_) => {
            if ir.segments == 0 && ir.elements != 0 {
                return Err(Error::Shape);
            }
            ir.segments.min(MAX_BLOCKS)
        }
        Family::Compact { .. } => {
            if ir.segments != 0 {
                return Err(Error::Shape);
            }
            // One chunk per block, at most 2^20 blocks. Supported CUDA
            // architectures admit this x grid; Metal uses threadgroups.
            chunks
        }
    }
    .max(1);
    let nvrtc_options = match target(caps)? {
        // The options of the shared bounded NVRTC child (`compile_cubin`), which
        // builds a CUBIN for the matching real `sm_N`.
        Some(sm) => vec![
            format!("--gpu-architecture=compute_{sm}"),
            "--std=c++17".to_owned(),
        ],
        None => Vec::new(),
    };
    Ok(Verified {
        ir,
        caps,
        blocks,
        nvrtc_options,
    })
}

/// Host admission of runtime buffers against the verified IR: exact lengths,
/// offsets that start at zero, never decrease and end at the element count.
/// Incomplete or decreasing offsets can be valid serial source; they are an
/// admission miss (CPU fallback), never a device fault.
pub fn admit_inputs(plan: &Verified, input: &[u32], offsets: &[u32]) -> Result<(), Error> {
    let ir = &plan.ir;
    if input.len() != ir.elements as usize {
        return Err(Error::Offsets);
    }
    match ir.family {
        Family::Compact { .. } => {
            if !offsets.is_empty() {
                return Err(Error::Offsets);
            }
        }
        Family::Map(_) | Family::Reduce(_) => {
            if offsets.len() != ir.segments as usize + 1
                || offsets.first() != Some(&0)
                || offsets.last() != Some(&ir.elements)
                || offsets.windows(2).any(|pair| pair[0] > pair[1])
            {
                return Err(Error::Offsets);
            }
        }
    }
    Ok(())
}

/// Independent sequential model of the IR, after [`admit_inputs`]. Compaction
/// returns exactly the retained words.
pub fn reference(plan: &Verified, input: &[u32], offsets: &[u32]) -> Result<Vec<u32>, Error> {
    admit_inputs(plan, input, offsets)?;
    Ok(match &plan.ir.family {
        Family::Map(ops) => {
            let mut output = Vec::with_capacity(input.len());
            for (segment, pair) in offsets.windows(2).enumerate() {
                for &value in &input[pair[0] as usize..pair[1] as usize] {
                    output.push(ops.iter().fold(value, |v, op| match op {
                        MapOp::WrappingAdd(k) => v.wrapping_add(*k),
                        MapOp::WrappingMul(k) => v.wrapping_mul(*k),
                        MapOp::AddSegment => v.wrapping_add(segment as u32),
                    }));
                }
            }
            output
        }
        Family::Reduce(law) => offsets
            .windows(2)
            .map(|pair| {
                input[pair[0] as usize..pair[1] as usize]
                    .iter()
                    .fold(law.identity(), |acc, &v| law.combine(acc, v))
            })
            .collect(),
        Family::Compact { predicate, cutoff } => input
            .iter()
            .copied()
            .filter(|&v| predicate.apply(v, *cutoff))
            .collect(),
    })
}

fn header(plan: &Verified, dialect: &str) -> Result<String, Error> {
    let ir = &plan.ir;
    let mut text = format!(
        "// invar-segmented-v1 version={} dialect={dialect}\n// source={} graph={} span={}:{} caps={:?}\n// family={:?} elements={} segments={} blocks={} lanes={LANES}\n// function-contract=",
        ir.version,
        ir.source_identity,
        ir.graph_revision,
        ir.span.start,
        ir.span.end,
        plan.caps,
        ir.family,
        ir.elements,
        ir.segments,
        plan.blocks
    );
    for byte in ir.function_contract {
        write!(text, "{byte:02x}").map_err(|_| Error::SourceLimit)?;
    }
    text.push_str("\n// required-options:");
    for option in &plan.nvrtc_options {
        write!(text, " {option}").map_err(|_| Error::SourceLimit)?;
    }
    text.push('\n');
    Ok(text)
}

fn finish(text: String) -> Result<String, Error> {
    if text.len() > crate::MAX_SOURCE_BYTES {
        Err(Error::SourceLimit)
    } else {
        Ok(text)
    }
}

fn map_body(ops: &[MapOp]) -> String {
    ops.iter()
        .map(|op| match op {
            MapOp::WrappingAdd(k) => format!("    v = v + {k}u;\n"),
            MapOp::WrappingMul(k) => format!("    v = v * {k}u;\n"),
            MapOp::AddSegment => "    v = v + lo;\n".to_owned(),
        })
        .collect()
}

/// The device source shared by both dialects, with dialect-specific words.
struct Dialect {
    prelude: &'static str,
    word: &'static str,
    barrier: &'static str,
    shared: &'static str,
}

const CUDA: Dialect = Dialect {
    prelude: "static_assert(sizeof(unsigned int) == 4, \"32-bit words required\");\n",
    word: "unsigned int",
    barrier: "__syncthreads()",
    shared: "__shared__",
};
const MSL: Dialect = Dialect {
    prelude: "#include <metal_stdlib>\nusing namespace metal;\n",
    word: "uint",
    barrier: "threadgroup_barrier(mem_flags::mem_threadgroup)",
    shared: "threadgroup",
};

/// Kernel signature and geometry guard. CUDA reads geometry from builtins;
/// Metal from attributes. An unexpected geometry exits uniformly per block.
fn signature(
    dialect: &Dialect,
    cuda: bool,
    name: &str,
    buffers: &[(&str, bool)],
    grid: Option<u32>,
) -> String {
    let mut text = String::new();
    let grid_guard = grid.map_or(String::new(), |g| format!(" || groups != {g}u"));
    if cuda {
        let params = buffers
            .iter()
            .map(|(name, writes)| {
                if *writes {
                    format!("{} * __restrict__ {name}", dialect.word)
                } else {
                    format!("const {} * __restrict__ {name}", dialect.word)
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        let _ = write!(
            text,
            "extern \"C\" __global__ void {name}({params}) {{\n  const unsigned int group = blockIdx.x, lane = threadIdx.x, groups = gridDim.x;\n  if (blockDim.x != {LANES}u || blockDim.y != 1u || blockDim.z != 1u || gridDim.y != 1u || gridDim.z != 1u{grid_guard}) return;\n"
        );
    } else {
        let params = buffers
            .iter()
            .enumerate()
            .map(|(slot, (name, writes))| {
                let constness = if *writes { "" } else { "const " };
                format!("device {constness}uint* {name} [[buffer({slot})]]")
            })
            .collect::<Vec<_>>()
            .join(", ");
        let _ = write!(
            text,
            "kernel void {name}({params}, uint group [[threadgroup_position_in_grid]], uint lane [[thread_position_in_threadgroup]], uint groups [[threadgroups_per_grid]], uint width [[threads_per_threadgroup]]) {{\n  if (width != {LANES}u{grid_guard}) return;\n"
        );
    }
    text
}

fn emit(plan: &Verified, dialect: &Dialect, cuda: bool) -> Result<String, Error> {
    let ir = &plan.ir;
    let n = ir.elements;
    let s = ir.segments;
    let w = dialect.word;
    let barrier = dialect.barrier;
    let shared = dialect.shared;
    let mut text = header(plan, if cuda { "cuda" } else { "msl" })?;
    text.push_str(dialect.prelude);
    match &ir.family {
        Family::Map(ops) => {
            text.push_str(&signature(
                dialect,
                cuda,
                "invar_segmented_map_v1",
                &[("input", false), ("offsets", false), ("output", true)],
                None,
            ));
            let _ = write!(
                text,
                "  for ({w} j = group * {LANES}u + lane; j < {n}u; j += groups * {LANES}u) {{\n    {w} lo = 0u, hi = {s}u;\n    while (lo < hi) {{\n      const {w} mid = hi - (hi - lo) / 2u;\n      if (offsets[mid] <= j) lo = mid; else hi = mid - 1u;\n    }}\n    {w} v = input[j];\n{}    output[j] = v;\n  }}\n}}\n",
                map_body(ops)
            );
            text.push_str(&signature(
                dialect,
                cuda,
                "invar_segmented_map_short_v1",
                &[("input", false), ("offsets", false), ("output", true)],
                None,
            ));
            let _ = write!(
                text,
                "  for ({w} lo = group * {LANES}u + lane; lo < {s}u; lo += groups * {LANES}u) {{\n    for ({w} j = offsets[lo]; j < offsets[lo + 1u]; ++j) {{\n    {w} v = input[j];\n{}    output[j] = v;\n    }}\n  }}\n}}\n",
                map_body(ops)
            );
        }
        Family::Reduce(law) => {
            let op = law.operator();
            let identity = law.identity();
            // Many short segments: one lane per segment, no barriers or atomics.
            text.push_str(&signature(
                dialect,
                cuda,
                "invar_segmented_reduce_v1",
                &[("input", false), ("offsets", false), ("output", true)],
                None,
            ));
            let _ = write!(
                text,
                "  for ({w} s = group * {LANES}u + lane; s < {s}u; s += groups * {LANES}u) {{\n    {w} acc = {identity}u;\n    for ({w} k = offsets[s]; k < offsets[s + 1u]; ++k) acc = acc {op} input[k];\n    output[s] = acc;\n  }}\n}}\n"
            );
            // Admission proves uniform short offsets before this fixed-span
            // loop can omit their loads and expose its bound to the compiler.
            text.push_str(&signature(
                dialect,
                cuda,
                "invar_segmented_reduce_uniform_v1",
                &[("input", false), ("output", true)],
                None,
            ));
            let length = n.checked_div(s).unwrap_or(0);
            let _ = write!(
                text,
                "  const {w} s = group * {LANES}u + lane;\n  if (s < {s}u) {{\n    {w} acc = {identity}u;\n    for ({w} k = 0u; k < {length}u; ++k) acc = acc {op} input[s * {length}u + k];\n    output[s] = acc;\n  }}\n}}\n"
            );
            // Inclusive segmented block scan: a head resets the reduction.
            // Only the last run in a tile can carry into the next tile.
            text.push_str(&signature(
                dialect,
                cuda,
                "invar_segmented_reduce_tiles_v1",
                &[
                    ("input", false),
                    ("offsets", false),
                    ("output", true),
                    ("tails", true),
                ],
                Some(n.div_ceil(LANES).max(1)),
            ));
            let _ = write!(
                text,
                "  {shared} {w} partial[{LANES}], head[{LANES}];\n  const {w} j = group * {LANES}u + lane;\n  {w} lo = 0u, hi = {s}u;\n  while (lo < hi) {{\n    const {w} mid = hi - (hi - lo) / 2u;\n    if (offsets[mid] <= j) lo = mid; else hi = mid - 1u;\n  }}\n  partial[lane] = j < {n}u ? input[j] : {identity}u;\n  head[lane] = (lane == 0u || j >= {n}u || j == offsets[lo]) ? 1u : 0u;\n  {barrier};\n  for ({w} delta = 1u; delta < {LANES}u; delta <<= 1u) {{\n    const {w} reset = head[lane];\n    const {w} left = lane >= delta ? partial[lane - delta] : {identity}u;\n    const {w} left_head = lane >= delta ? head[lane - delta] : 1u;\n    {barrier};\n    if (lane >= delta) {{\n      if (reset == 0u) partial[lane] = left {op} partial[lane];\n      head[lane] = reset | left_head;\n    }}\n    {barrier};\n  }}\n  if (lane == {LANES}u - 1u) tails[group] = partial[lane];\n  if (j < {n}u && j + 1u == offsets[lo + 1u]) output[lo] = partial[lane];\n}}\n"
            );
            // The final tile already wrote its local segment result. Reduce all
            // preceding tile tails and combine once. Empty segments get identity.
            text.push_str(&signature(
                dialect,
                cuda,
                "invar_segmented_reduce_carry_v1",
                &[("offsets", false), ("tails", false), ("output", true)],
                None,
            ));
            let _ = write!(
                text,
                "  {shared} {w} partial[{LANES}];\n  for ({w} s = group; s < {s}u; s += groups) {{\n    const {w} begin = offsets[s], end = offsets[s + 1u];\n    const {w} first = begin / {LANES}u, last = end > begin ? (end - 1u) / {LANES}u : first;\n    {w} acc = {identity}u;\n    for ({w} b = first + lane; b < last; b += {LANES}u) acc = acc {op} tails[b];\n    partial[lane] = acc;\n    {barrier};\n    for ({w} width2 = {}u; width2 > 0u; width2 >>= 1u) {{\n      if (lane < width2) partial[lane] = partial[lane] {op} partial[lane + width2];\n      {barrier};\n    }}\n    if (lane == 0u) output[s] = end == begin ? {identity}u : partial[0] {op} output[s];\n    {barrier};\n  }}\n}}\n",
                LANES / 2
            );
        }
        Family::Compact { predicate, cutoff } => {
            let op = predicate.operator();
            let blocks = plan.blocks;
            let supers = blocks.div_ceil(LANES);
            text.push_str(&signature(
                dialect,
                cuda,
                "invar_compact_count_v1",
                &[("input", false), ("counts", true)],
                Some(blocks),
            ));
            let _ = write!(
                text,
                "  {shared} {w} partial[{LANES}];\n  const {w} j = group * {LANES}u + lane;\n  partial[lane] = (j < {n}u && input[j] {op} {cutoff}u) ? 1u : 0u;\n  {barrier};\n  for ({w} width2 = {}u; width2 > 0u; width2 >>= 1u) {{\n    if (lane < width2) partial[lane] += partial[lane + width2];\n    {barrier};\n  }}\n  if (lane == 0u) counts[group] = partial[0];\n}}\n",
                LANES / 2
            );
            text.push_str(&signature(
                dialect,
                cuda,
                "invar_compact_scan_v1",
                &[("counts", false), ("bases", true), ("sums", true)],
                Some(supers),
            ));
            let _ = write!(
                text,
                "  {shared} {w} scan[{LANES}];\n  const {w} b = group * {LANES}u + lane;\n  const {w} value = b < {blocks}u ? counts[b] : 0u;\n  scan[lane] = value;\n  {barrier};\n  for ({w} delta = 1u; delta < {LANES}u; delta <<= 1u) {{\n    const {w} add = lane >= delta ? scan[lane - delta] : 0u;\n    {barrier};\n    scan[lane] += add;\n    {barrier};\n  }}\n  if (b < {blocks}u) bases[b] = scan[lane] - value;\n  if (lane == {LANES}u - 1u) sums[group] = scan[lane];\n}}\n"
            );
            text.push_str(&signature(
                dialect,
                cuda,
                "invar_compact_carry_v1",
                &[("sums", false), ("carries", true), ("total", true)],
                Some(1),
            ));
            let chunk = supers.div_ceil(LANES);
            let _ = write!(
                text,
                "  {shared} {w} scan[{LANES}];\n  const {w} begin = lane * {chunk}u, end = min(begin + {chunk}u, {supers}u);\n  {w} value = 0u;\n  for ({w} b = begin; b < end; ++b) value += sums[b];\n  scan[lane] = value;\n  {barrier};\n  for ({w} delta = 1u; delta < {LANES}u; delta <<= 1u) {{\n    const {w} add = lane >= delta ? scan[lane - delta] : 0u;\n    {barrier};\n    scan[lane] += add;\n    {barrier};\n  }}\n  {w} run = scan[lane] - value;\n  for ({w} b = begin; b < end; ++b) {{ carries[b] = run; run += sums[b]; }}\n  if (lane == {LANES}u - 1u) total[0] = scan[lane];\n}}\n"
            );
            text.push_str(&signature(
                dialect,
                cuda,
                "invar_compact_scatter_v1",
                &[
                    ("input", false),
                    ("bases", false),
                    ("carries", false),
                    ("output", true),
                ],
                Some(blocks),
            ));
            let _ = write!(
                text,
                "  {shared} {w} scan[{LANES}];\n  const {w} j = group * {LANES}u + lane;\n  const {w} v = j < {n}u ? input[j] : 0u;\n  const {w} keep = (j < {n}u && v {op} {cutoff}u) ? 1u : 0u;\n  scan[lane] = keep;\n  {barrier};\n  for ({w} delta = 1u; delta < {LANES}u; delta <<= 1u) {{\n    const {w} add = lane >= delta ? scan[lane - delta] : 0u;\n    {barrier};\n    scan[lane] += add;\n    {barrier};\n  }}\n  if (keep != 0u) output[carries[group / {LANES}u] + bases[group] + scan[lane] - 1u] = v;\n}}\n"
            );
        }
    }
    finish(text)
}

/// CUDA C++ for NVRTC; the verified options are part of artifact identity.
pub fn emit_cuda(plan: &Verified) -> Result<String, Error> {
    if !matches!(plan.caps, Capabilities::Cuda { .. }) {
        return Err(Error::Capabilities);
    }
    emit(plan, &CUDA, true)
}

/// MSL 3.1 for the Metal runtime, from the same verified IR.
pub fn emit_msl(plan: &Verified) -> Result<String, Error> {
    if !matches!(plan.caps, Capabilities::Metal { .. }) {
        return Err(Error::Capabilities);
    }
    emit(plan, &MSL, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ir(family: Family, elements: u32, segments: u32) -> Ir {
        Ir {
            version: VERSION,
            span: Span { start: 3, end: 9 },
            source_identity: 7,
            graph_revision: 11,
            function_contract: [5; 32],
            family,
            elements,
            segments,
        }
    }
    const ORIGIN: Provenance = Provenance {
        source_identity: 7,
        graph_revision: 11,
    };
    const METAL: Capabilities = Capabilities::Metal {
        apple_family: 9,
        msl_major: 3,
        msl_minor: 1,
    };

    type Mutation = (fn(&mut Ir), Error);

    #[test]
    fn issue_m4_04_device_size_cap_2_28_and_refusal() {
        // Literal contract boundary: restoring the old cap must fail this test.
        const N: u32 = 1 << 28;
        for caps in [METAL, Capabilities::cuda_12_9_sm89()] {
            for (family, segments, blocks, words) in [
                (
                    Family::Map(vec![MapOp::AddSegment]),
                    N,
                    MAX_BLOCKS,
                    vec![N as usize, N as usize + 1, N as usize],
                ),
                (
                    Family::Reduce(Law::WrappingSum),
                    N,
                    MAX_BLOCKS,
                    vec![N as usize, N as usize + 1, N as usize, 1 << 20],
                ),
                (
                    Family::Reduce(Law::WrappingProduct),
                    N,
                    MAX_BLOCKS,
                    vec![N as usize, N as usize + 1, N as usize, 1 << 20],
                ),
                (
                    Family::Compact {
                        predicate: Predicate::Less,
                        cutoff: 7,
                    },
                    0,
                    1 << 20,
                    vec![N as usize, N as usize, 1 << 20, 1 << 20, 1, 4096, 4096],
                ),
            ] {
                let good = ir(family, N, segments);
                let plan = verify(good.clone(), caps, ORIGIN).unwrap();
                assert_eq!(plan.blocks(), blocks);
                assert_eq!(plan.buffer_words(), words);
                let bytes: u64 = plan.buffer_words().iter().map(|&n| n as u64 * 4).sum();
                assert!(bytes <= 3 * (1u64 << 30) + (1 << 22) + 4);
                let source = match caps {
                    Capabilities::Cuda { .. } => emit_cuda(&plan).unwrap(),
                    Capabilities::Metal { .. } => emit_msl(&plan).unwrap(),
                };
                assert!(source.contains("268435456u"));
                let mut over = good.clone();
                over.elements = N + 1;
                assert_eq!(verify(over, caps, ORIGIN), Err(Error::Bounds));
                if segments != 0 {
                    let mut over = good;
                    over.segments = N + 1;
                    assert_eq!(verify(over, caps, ORIGIN), Err(Error::Bounds));
                }
            }
        }
    }

    #[test]
    fn issue_m4_04_device_verifier_rejects_every_field_mutation() {
        let good = ir(Family::Reduce(Law::WrappingSum), 10, 3);
        let caps = Capabilities::cuda_12_9_sm89();
        assert!(verify(good.clone(), caps, ORIGIN).is_ok());
        let cases: [Mutation; 9] = [
            (|i| i.version = 2, Error::Version),
            (|i| i.span = Span { start: 9, end: 9 }, Error::Source),
            (|i| i.graph_revision = 12, Error::Graph),
            (|i| i.function_contract = [0; 32], Error::Contract),
            (|i| i.elements = MAX_ELEMENTS + 1, Error::Bounds),
            (|i| i.segments = MAX_SEGMENTS + 1, Error::Bounds),
            (|i| i.segments = 0, Error::Shape),
            (|i| i.family = Family::Map(Vec::new()), Error::Operations),
            (
                |i| {
                    i.family = Family::Compact {
                        predicate: Predicate::Less,
                        cutoff: 1,
                    }
                },
                Error::Shape,
            ),
        ];
        for (mutate, error) in cases {
            let mut changed = good.clone();
            mutate(&mut changed);
            assert_eq!(verify(changed, caps, ORIGIN).unwrap_err(), error);
        }
        for caps in [
            Capabilities::Cuda {
                major: 12,
                minor: 3,
                sm: 89,
            },
            Capabilities::Cuda {
                major: 12,
                minor: 7,
                sm: 120,
            },
            Capabilities::Cuda {
                major: 11,
                minor: 8,
                sm: 86,
            },
            Capabilities::Metal {
                apple_family: 6,
                msl_major: 3,
                msl_minor: 1,
            },
        ] {
            assert_eq!(
                verify(good.clone(), caps, ORIGIN).unwrap_err(),
                Error::Capabilities
            );
        }
        let sm120 = Capabilities::Cuda {
            major: 12,
            minor: 9,
            sm: 120,
        };
        assert!(verify(good.clone(), sm120, ORIGIN).is_ok());
        let compact = ir(
            Family::Compact {
                predicate: Predicate::Less,
                cutoff: 1,
            },
            MAX_ELEMENTS,
            0,
        );
        assert_eq!(verify(compact, METAL, ORIGIN).unwrap().blocks(), 1 << 20);
    }

    #[test]
    fn issue_m4_04_device_admission_and_reference_model() {
        let caps = Capabilities::cuda_12_9_sm89();
        let map = verify(
            ir(
                Family::Map(vec![MapOp::WrappingMul(3), MapOp::AddSegment]),
                5,
                4,
            ),
            caps,
            ORIGIN,
        )
        .unwrap();
        let input = [1, 2, u32::MAX, 4, 5];
        assert_eq!(
            reference(&map, &input, &[0, 2, 2, 3, 5]).unwrap(),
            [3, 6, u32::MAX.wrapping_mul(3) + 2, 15, 18]
        );
        for offsets in [
            &[0, 2, 2, 3][..],
            &[1, 2, 2, 3, 5],
            &[0, 3, 2, 3, 5],
            &[0, 2, 2, 3, 6],
        ] {
            assert_eq!(
                reference(&map, &input, offsets).unwrap_err(),
                Error::Offsets
            );
        }
        let product = verify(ir(Family::Reduce(Law::WrappingProduct), 5, 4), caps, ORIGIN).unwrap();
        assert_eq!(
            reference(&product, &input, &[0, 2, 2, 3, 5]).unwrap(),
            [2, 1, u32::MAX, 20]
        );
        let compact = verify(
            ir(
                Family::Compact {
                    predicate: Predicate::Greater,
                    cutoff: 2,
                },
                5,
                0,
            ),
            caps,
            ORIGIN,
        )
        .unwrap();
        assert_eq!(reference(&compact, &input, &[]).unwrap(), [u32::MAX, 4, 5]);
        assert_eq!(
            reference(&compact, &input, &[0]).unwrap_err(),
            Error::Offsets
        );
    }

    #[test]
    fn issue_m4_04_device_admitted_schedule_and_scratch() {
        for caps in [METAL, Capabilities::cuda_12_9_sm89()] {
            let plan = verify(ir(Family::Reduce(Law::WrappingSum), 600, 4), caps, ORIGIN).unwrap();
            let input = vec![7; 600];
            let short = plan.launches(&input, &[0, 200, 200, 400, 600]).unwrap();
            assert_eq!(
                short,
                vec![Launch {
                    kernel: 0,
                    buffers: &[0, 1, 2],
                    blocks: 1
                }]
            );
            let tiled = plan.launches(&input, &[0, 257, 257, 599, 600]).unwrap();
            assert_eq!(
                tiled
                    .iter()
                    .map(|l| (l.kernel, l.blocks))
                    .collect::<Vec<_>>(),
                [(1, 3), (2, 4)]
            );
            assert_eq!(plan.buffer_words(), [600, 5, 4, 3]);
            assert_eq!(
                plan.launches(&input, &[0, 150, 300, 450, 600]).unwrap(),
                vec![Launch {
                    kernel: 3,
                    buffers: &[0, 2],
                    blocks: 1
                }]
            );
            // Same N/S, but a ragged shape must keep the offset-reading path.
            assert_eq!(
                plan.launches(&input, &[0, 149, 300, 450, 600]).unwrap()[0].kernel,
                0
            );
            assert_eq!(
                plan.launches(&input, &[0, 257, 256, 599, 600]),
                Err(Error::Offsets)
            );
            assert_eq!(
                plan.launches(&input[..599], &[0, 200, 200, 400, 600]),
                Err(Error::Offsets)
            );
            for n in [0, 1, 65_537, MAX_ELEMENTS] {
                let compact = verify(
                    ir(
                        Family::Compact {
                            predicate: Predicate::Less,
                            cutoff: 9,
                        },
                        n,
                        0,
                    ),
                    caps,
                    ORIGIN,
                )
                .unwrap();
                let words = compact.buffer_words();
                assert_eq!(words[2], n.div_ceil(LANES).max(1) as usize);
                assert_eq!(words[5], words[2].div_ceil(LANES as usize));
                assert_eq!(words[5], words[6]);
            }
        }
    }

    #[test]
    fn issue_m4_04_device_emitters_are_deterministic_and_bound_to_identity() {
        let families = [
            Family::Map(vec![MapOp::WrappingAdd(7), MapOp::AddSegment]),
            Family::Reduce(Law::WrappingSum),
            Family::Compact {
                predicate: Predicate::NotEqual,
                cutoff: 9,
            },
        ];
        for family in families {
            let segments = if matches!(family, Family::Compact { .. }) {
                0
            } else {
                2
            };
            let cuda = verify(
                ir(family.clone(), 600, segments),
                Capabilities::cuda_12_9_sm89(),
                ORIGIN,
            )
            .unwrap();
            let metal = verify(ir(family, 600, segments), METAL, ORIGIN).unwrap();
            let cuda_source = emit_cuda(&cuda).unwrap();
            let msl_source = emit_msl(&metal).unwrap();
            assert_eq!(cuda_source, emit_cuda(&cuda).unwrap());
            assert!(
                cuda_source
                    .contains("// required-options: --gpu-architecture=compute_89 --std=c++17")
            );
            for kernel in cuda.kernels() {
                assert!(
                    cuda_source.contains(kernel) && msl_source.contains(kernel),
                    "{kernel}"
                );
            }
            assert_eq!(emit_msl(&cuda).unwrap_err(), Error::Capabilities);
            assert_eq!(emit_cuda(&metal).unwrap_err(), Error::Capabilities);
        }
    }
}
