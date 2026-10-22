//! Verified out-of-place rank-2 transpose of 32-bit element bits.
//!
//! Static metadata does not grant source extraction, pointer ownership, source
//! fuel/trap commit, completion/fences or launch permission. CUDA geometry is a
//! fixed schedule, not a measured tuning result.

use crate::{Capabilities, Provenance, Span, floating};
use floating::Buffer;
use std::fmt::Write;

pub const VERSION: u32 = 1;
pub const KERNEL_NAME: &str = "invar_transpose_bits_v1";
pub const TILE_SIDE: u32 = 16;
/// Logical domain ceiling; physical buffers may retain checked padding.
pub const MAX_LOGICAL_ELEMENTS: u64 = 1_u64 << 28;
/// Two host/device copies and one returned output, for two full-size buffers.
pub const MAX_OWNED_BYTES: u64 = 5 * MAX_LOGICAL_ELEMENTS * 4;
const CPU_OPTIONS: &[&str] = &["-std=c++17"];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ir {
    pub version: u32,
    pub layout_version: u32,
    pub span: Span,
    pub source_identity: u64,
    pub graph_revision: u64,
    pub function_contract: [u8; 32],
    pub dimensions: [u32; 2],
    pub input: Buffer,
    pub output: Buffer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Version,
    Layout,
    Source,
    Graph,
    Contract,
    Shape,
    Alignment,
    Bounds,
    Alias,
    Capabilities,
    SourceLimit,
}

/// Immutable static physical proof. Dynamic pointer and completion admission
/// remain separate host obligations, even after successful emission.
#[derive(Debug)]
pub struct Verified {
    ir: Ir,
    caps: Capabilities,
    nvrtc_options: Vec<String>,
}

impl Verified {
    pub fn ir(&self) -> &Ir {
        &self.ir
    }
    pub fn capabilities(&self) -> Capabilities {
        self.caps
    }
    /// Required options are part of artifact identity; host caching and launch
    /// admission must bind them without unverified changes.
    pub fn nvrtc_options(&self) -> &[String] {
        &self.nvrtc_options
    }
    pub fn cpu_options(&self) -> &'static [&'static str] {
        CPU_OPTIONS
    }
}

pub(crate) fn buffer(b: &Buffer) -> Result<(), Error> {
    if b.shape.iter().any(|&n| n > floating::MAX_EXTENT)
        || b.elements > floating::MAX_ELEMENTS
        || b.offset > b.elements
    {
        return Err(Error::Bounds);
    }
    if b.strides != [u64::from(b.shape[1]), 1] {
        return Err(Error::Layout);
    }
    if !b.alignment.is_power_of_two() || b.alignment < 4 {
        return Err(Error::Alignment);
    }
    // Empty axes permit a one-past offset, but no physical element access.
    if !b.shape.contains(&0) {
        let count = u64::from(b.shape[0])
            .checked_mul(u64::from(b.shape[1]))
            .ok_or(Error::Bounds)?;
        if count > MAX_LOGICAL_ELEMENTS {
            return Err(Error::Bounds);
        }
        let end = b.offset.checked_add(count).ok_or(Error::Bounds)?;
        if end > b.elements {
            return Err(Error::Bounds);
        }
    }
    Ok(())
}

pub fn verify(ir: Ir, caps: Capabilities, origin: Provenance) -> Result<Verified, Error> {
    if ir.version != VERSION {
        return Err(Error::Version);
    }
    if ir.layout_version != floating::LAYOUT_VERSION {
        return Err(Error::Layout);
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
    let [rows, cols] = ir.dimensions;
    if ir.input.shape != [rows, cols] || ir.output.shape != [cols, rows] {
        return Err(Error::Shape);
    }
    buffer(&ir.input)?;
    buffer(&ir.output)?;
    if ir.input.id == ir.output.id {
        return Err(Error::Alias);
    }
    let nvrtc_options = target_options(caps)?;
    Ok(Verified {
        ir,
        caps,
        nvrtc_options,
    })
}

fn header(plan: &Verified, cpu: bool) -> Result<String, Error> {
    let ir = &plan.ir;
    let mut text = format!(
        "// invar-transpose-bits-v1 version={} layout-version={} element-bytes=4\n// source={} graph={} span={}:{} caps={:?}\n// function-contract=",
        ir.version,
        ir.layout_version,
        ir.source_identity,
        ir.graph_revision,
        ir.span.start,
        ir.span.end,
        plan.caps
    );
    for byte in ir.function_contract {
        write!(text, "{byte:02x}").map_err(|_| Error::SourceLimit)?;
    }
    writeln!(
        text,
        "\n// dimensions={:?} tile=16x17 block=16x16x1 grid=Gx1x1 G>=1",
        ir.dimensions
    )
    .map_err(|_| Error::SourceLimit)?;
    for (slot, b) in [&ir.input, &ir.output].into_iter().enumerate() {
        writeln!(
            text,
            "// buffer{slot} id={} elements={} shape={:?} strides={:?} offset={} alignment={}",
            b.id, b.elements, b.shape, b.strides, b.offset, b.alignment
        )
        .map_err(|_| Error::SourceLimit)?;
    }
    text.push_str("// required-options:");
    if cpu {
        for option in CPU_OPTIONS {
            write!(text, " {option}").map_err(|_| Error::SourceLimit)?;
        }
    } else {
        for option in &plan.nvrtc_options {
            write!(text, " {option}").map_err(|_| Error::SourceLimit)?;
        }
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

/// Padded shared-memory bit transpose. Nonempty plans require a 16x16x1 block
/// and a Gx1x1 grid, G>=1. Every block strides over whole tiles. Invalid geometry
/// exits uniformly before any barrier/access; host admission must validate it.
/// Both barriers are unconditional within a block's uniform tile loop. The
/// second barrier prevents a subsequent tile load from clobbering prior reads.
pub fn emit_cuda(plan: &Verified) -> Result<String, Error> {
    if !matches!(plan.caps, Capabilities::Cuda { .. }) {
        return Err(Error::Capabilities);
    }
    let mut text = header(plan, false)?;
    text.push_str("static_assert(sizeof(unsigned int) == 4, \"32-bit words required\");\nextern \"C\" __global__ void invar_transpose_bits_v1(const unsigned int* input, unsigned int* output) {\n");
    let [rows, cols] = plan.ir.dimensions;
    if rows == 0 || cols == 0 {
        text.push_str("  (void)input; (void)output;\n}\n");
        return finish(text);
    }
    let tile_cols = u64::from(cols).div_ceil(u64::from(TILE_SIDE));
    let tiles = u64::from(rows).div_ceil(u64::from(TILE_SIDE)) * tile_cols;
    text.push_str("  if (blockDim.x != 16 || blockDim.y != 16 || blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1 || gridDim.x == 0) return;\n  __shared__ unsigned int tile[16][17];\n  for (unsigned long long t = static_cast<unsigned long long>(blockIdx.x); ");
    text.push_str(&format!("t < {tiles}ull; t += static_cast<unsigned long long>(gridDim.x)) {{\n    const unsigned long long tile_row = t / {tile_cols}ull, tile_col = t % {tile_cols}ull;\n    const unsigned long long x = tile_col * 16ull + threadIdx.x;\n    const unsigned long long y = tile_row * 16ull + threadIdx.y;\n    if (y < {rows}ull && x < {cols}ull) {{\n      const unsigned long long at = {}ull + y * {}ull + x;\n      tile[threadIdx.y][threadIdx.x] = input[at];\n    }}\n    __syncthreads();\n    const unsigned long long out_row = tile_col * 16ull + threadIdx.y;\n    const unsigned long long out_col = tile_row * 16ull + threadIdx.x;\n    if (out_row < {cols}ull && out_col < {rows}ull) {{\n      const unsigned long long at = {}ull + out_row * {}ull + out_col;\n      output[at] = tile[threadIdx.x][threadIdx.y];\n    }}\n    __syncthreads();\n  }}\n}}\n", plan.ir.input.offset, plan.ir.input.strides[0], plan.ir.output.offset, plan.ir.output.strides[0]));
    finish(text)
}

/// Fixed-pointer C++17 analogue, copying u32 or binary32 bits without arithmetic.
/// Status0 succeeds; status2 rejects null/misaligned/overflowing/overlapping
/// allocation ranges before writing. Empty domains require no valid pointer.
/// Buffer sizes/ownership and synchronization still need host admission.
pub fn emit_cpu(plan: &Verified) -> Result<String, Error> {
    let mut text = header(plan, true)?;
    text.push_str("#include <cstdint>\nstatic_assert(sizeof(std::uint32_t) == 4, \"32-bit words required\");\nextern \"C\" int invar_transpose_bits_v1(const std::uint32_t* input, std::uint32_t* output) {\n");
    let [rows, cols] = plan.ir.dimensions;
    if rows == 0 || cols == 0 {
        text.push_str("  (void)input; (void)output; return 0;\n}\n");
        return finish(text);
    }
    let a = &plan.ir.input;
    let b = &plan.ir.output;
    text.push_str("  if (!input || !output) return 2;\n  const auto ip = reinterpret_cast<std::uintptr_t>(input), op = reinterpret_cast<std::uintptr_t>(output);\n");
    text.push_str(&format!("  if ((ip & {}ull) != 0 || (op & {}ull) != 0) return 2;\n  if (ip > UINTPTR_MAX - {}ull || op > UINTPTR_MAX - {}ull) return 2;\n  if (ip < op + {}ull && op < ip + {}ull) return 2;\n  for (unsigned long long r = 0; r < {rows}ull; ++r) {{\n    for (unsigned long long c = 0; c < {cols}ull; ++c) {{\n      output[{}ull + c * {}ull + r] = input[{}ull + r * {}ull + c];\n    }}\n  }}\n  return 0;\n}}\n", a.alignment - 1, b.alignment - 1, a.elements * 4, b.elements * 4, b.elements * 4, a.elements * 4, b.offset, b.strides[0], a.offset, a.strides[0]));
    finish(text)
}

/// Exact target options for the M4-02 integer/bit-copy schedules.
pub fn target_options(caps: Capabilities) -> Result<Vec<String>, Error> {
    match caps {
        Capabilities::Cuda {
            major: 12,
            minor: 4..=9,
            sm,
        } if matches!(
            sm,
            50 | 52 | 53 | 60 | 61 | 62 | 70 | 72 | 75 | 80 | 86 | 87 | 89 | 90
        ) || (matches!(sm, 100 | 101 | 120)
            && matches!(caps, Capabilities::Cuda { minor: 8..=9, .. })) =>
        {
            Ok(vec![
                "--std=c++17".into(),
                format!("--gpu-architecture=compute_{sm}"),
                "--modify-stack-limit=false".into(),
            ])
        }
        Capabilities::Metal {
            apple_family: 7..=10,
            msl_major: 3,
            msl_minor: 1,
        } => Ok(Vec::new()),
        _ => Err(Error::Capabilities),
    }
}

#[path = "transpose_msl.rs"]
mod metal;
pub use metal::emit_msl;
