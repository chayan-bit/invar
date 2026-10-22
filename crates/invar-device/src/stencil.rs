//! Static physical proof and fixed schedules for one fresh radius-1 FP32 slice.
//! Source extraction, dynamic ownership, launch/completion and publication remain
//! separate obligations. No in-place or multi-step device execution is admitted.
use crate::{Capabilities, Provenance, Span, floating};
use floating::Buffer;
use std::fmt::Write;

pub const VERSION: u32 = 1;
pub const BOUNDARY_VERSION: u32 = 1;
pub const KERNEL_NAME: &str = "invar_stencil_f32_v1";
pub const TILE_SIDE: u32 = 16;
const CPU_OPTIONS: &[&str] = &[
    "-std=c++17",
    "-fno-fast-math",
    "-ffp-contract=off",
    "-frounding-math",
    "-fno-associative-math",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boundary {
    ConstantZero,
    Clamp,
}
impl Boundary {
    pub fn name(self) -> &'static str {
        match self {
            Self::ConstantZero => "constant_zero",
            Self::Clamp => "clamp",
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ir {
    pub version: u32,
    pub numeric_version: u32,
    pub numeric_policy: String,
    pub layout_version: u32,
    pub boundary_version: u32,
    pub boundary: Boundary,
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
    Numeric,
    Layout,
    Boundary,
    Source,
    Graph,
    Contract,
    Shape,
    Alignment,
    Bounds,
    Alias,
    Capabilities,
    Geometry,
    Index,
    SourceLimit,
}
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
    pub fn nvrtc_options(&self) -> &[String] {
        &self.nvrtc_options
    }
    pub fn cpu_options(&self) -> &'static [&'static str] {
        CPU_OPTIONS
    }
    /// A whole block traverses each tile. All 256 lanes participate in both
    /// barriers, including masked edge lanes. Geometry is part of host admission.
    pub fn validate_launch(&self, block: [u32; 3], grid: [u32; 3]) -> Result<(), Error> {
        if block != [16, 16, 1] || grid[0] == 0 || grid[1..] != [1, 1] {
            Err(Error::Geometry)
        } else {
            Ok(())
        }
    }
    /// Physical read map for center, north, south, west, east, in that order.
    /// None denotes the constant +0 boundary; no out-of-range load is performed.
    pub fn reads(&self, [r, c]: [u32; 2]) -> Result<[Option<u64>; 5], Error> {
        let [rows, cols] = self.ir.dimensions;
        if r >= rows || c >= cols {
            return Err(Error::Index);
        }
        let mut result = [None; 5];
        for (slot, (y, x)) in [
            (i64::from(r), i64::from(c)),
            (i64::from(r) - 1, i64::from(c)),
            (i64::from(r) + 1, i64::from(c)),
            (i64::from(r), i64::from(c) - 1),
            (i64::from(r), i64::from(c) + 1),
        ]
        .into_iter()
        .enumerate()
        {
            let inside = y >= 0 && x >= 0 && y < i64::from(rows) && x < i64::from(cols);
            if inside || self.ir.boundary == Boundary::Clamp {
                let y = y.clamp(0, i64::from(rows) - 1) as u64;
                let x = x.clamp(0, i64::from(cols) - 1) as u64;
                result[slot] = Some(self.ir.input.offset + y * u64::from(cols) + x);
            }
        }
        Ok(result)
    }
}
fn buffer(b: &Buffer) -> Result<(), Error> {
    if b.shape.iter().any(|&n| n > floating::MAX_EXTENT)
        || b.elements > floating::MAX_ELEMENTS
        || b.offset > b.elements
        || u64::from(b.shape[0]) * u64::from(b.shape[1]) > crate::transpose::MAX_LOGICAL_ELEMENTS
    {
        return Err(Error::Bounds);
    }
    if b.strides != [u64::from(b.shape[1]), 1] {
        return Err(Error::Layout);
    }
    if !b.alignment.is_power_of_two() || b.alignment < 4 {
        return Err(Error::Alignment);
    }
    if !b.shape.contains(&0)
        && b.offset
            .checked_add(u64::from(b.shape[0]) * u64::from(b.shape[1]))
            .is_none_or(|n| n > b.elements)
    {
        return Err(Error::Bounds);
    }
    Ok(())
}
pub fn verify(ir: Ir, caps: Capabilities, origin: Provenance) -> Result<Verified, Error> {
    if ir.version != VERSION {
        return Err(Error::Version);
    }
    if ir.numeric_version != floating::NUMERIC_VERSION
        || ir.numeric_policy != floating::NUMERIC_POLICY
    {
        return Err(Error::Numeric);
    }
    if ir.layout_version != floating::LAYOUT_VERSION {
        return Err(Error::Layout);
    }
    if ir.boundary_version != BOUNDARY_VERSION {
        return Err(Error::Boundary);
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
    if ir.input.shape != ir.dimensions || ir.output.shape != ir.dimensions {
        return Err(Error::Shape);
    }
    buffer(&ir.input)?;
    buffer(&ir.output)?;
    if ir.input.id == ir.output.id {
        return Err(Error::Alias);
    }
    let sm = match caps {
        Capabilities::Cuda {
            major: 12,
            minor: 4..=9,
            sm,
        } if matches!(
            sm,
            50 | 52 | 53 | 60 | 61 | 62 | 70 | 72 | 75 | 80 | 86 | 87 | 89 | 90
        ) =>
        {
            sm
        }
        _ => return Err(Error::Capabilities),
    };
    Ok(Verified {
        ir,
        caps,
        nvrtc_options: vec![
            "--std=c++17".into(),
            format!("--gpu-architecture=compute_{sm}"),
            "--fmad=false".into(),
            "--ftz=false".into(),
            "--prec-div=true".into(),
            "--prec-sqrt=true".into(),
            "--modify-stack-limit=false".into(),
        ],
    })
}
fn header(p: &Verified, cpu: bool) -> Result<String, Error> {
    let mut s = format!(
        "// invar-stencil-f32-v1 version={} numeric={}:{} layout={} boundary={}:{}\n// source={} graph={} span={}:{} caps={:?}\n// function-contract=",
        p.ir.version,
        p.ir.numeric_version,
        p.ir.numeric_policy,
        p.ir.layout_version,
        p.ir.boundary_version,
        p.ir.boundary.name(),
        p.ir.source_identity,
        p.ir.graph_revision,
        p.ir.span.start,
        p.ir.span.end,
        p.caps
    );
    for byte in p.ir.function_contract {
        write!(s, "{byte:02x}").map_err(|_| Error::SourceLimit)?;
    }
    writeln!(s,"\n// dimensions={:?} block=16x16x1 tile=18x19 grid=Gx1x1 G>=1 order=center,north,south,west,east",p.ir.dimensions).map_err(|_| Error::SourceLimit)?;
    for b in [&p.ir.input, &p.ir.output] {
        writeln!(
            s,
            "// buffer id={} elements={} shape={:?} strides={:?} offset={} alignment={}",
            b.id, b.elements, b.shape, b.strides, b.offset, b.alignment
        )
        .map_err(|_| Error::SourceLimit)?;
    }
    writeln!(
        s,
        "// required-options: {}",
        if cpu {
            CPU_OPTIONS.join(" ")
        } else {
            p.nvrtc_options.join(" ")
        }
    )
    .map_err(|_| Error::SourceLimit)?;
    Ok(s)
}
fn finish(s: String) -> Result<String, Error> {
    if s.len() > crate::MAX_SOURCE_BYTES {
        Err(Error::SourceLimit)
    } else {
        Ok(s)
    }
}

/// Padded halo tile, cooperative masked loads, unconditional phase barriers.
/// Every cell performs four separately rounded additions, with canonical NaNs.
pub fn emit_cuda(p: &Verified) -> Result<String, Error> {
    let mut s = header(p, false)?;
    s.push_str("__device__ float invar_stencil_add(float a, float b) { float v=__fadd_rn(a,b); unsigned int bits=__float_as_uint(v); return ((bits & 0x7fffffffU)>0x7f800000U) ? __uint_as_float(0x7fc00000U) : v; }\nextern \"C\" __global__ void invar_stencil_f32_v1(const float* input, float* output) {\n");
    let [rows, cols] = p.ir.dimensions;
    if rows == 0 || cols == 0 {
        s.push_str("  (void)input; (void)output;\n}\n");
        return finish(s);
    }
    let tile_cols = u64::from(cols).div_ceil(16);
    let tiles = u64::from(rows).div_ceil(16) * tile_cols;
    s.push_str("  if (blockDim.x!=16 || blockDim.y!=16 || blockDim.z!=1 || gridDim.x==0 || gridDim.y!=1 || gridDim.z!=1) return;\n  __shared__ float tile[18][19];\n  const unsigned int lane=threadIdx.y*16+threadIdx.x;\n");
    s.push_str(&format!("  for (unsigned long long t=blockIdx.x; t<{tiles}ull; t+=static_cast<unsigned long long>(gridDim.x)) {{\n    const long long tr=static_cast<long long>(t/{tile_cols}ull)*16, tc=static_cast<long long>(t%{tile_cols}ull)*16;\n    for (unsigned int h=lane; h<324; h+=256) {{\n      long long y=tr+static_cast<long long>(h/18)-1, x=tc+static_cast<long long>(h%18)-1;\n      float v=0.0f;\n"));
    if p.ir.boundary == Boundary::Clamp {
        s.push_str(&format!("      y=y<0 ? 0 : (y>={rows}ll ? {rows}ll-1 : y); x=x<0 ? 0 : (x>={cols}ll ? {cols}ll-1 : x);\n"));
    }
    s.push_str(&format!("      if (y>=0 && x>=0 && y<{rows}ll && x<{cols}ll) v=input[{}ull+static_cast<unsigned long long>(y)*{cols}ull+static_cast<unsigned long long>(x)];\n      tile[h/18][h%18]=v;\n    }}\n    __syncthreads();\n    const unsigned long long y=static_cast<unsigned long long>(tr)+threadIdx.y, x=static_cast<unsigned long long>(tc)+threadIdx.x;\n    if (y<{rows}ull && x<{cols}ull) {{\n      const unsigned int ly=threadIdx.y+1, lx=threadIdx.x+1;\n      float v=invar_stencil_add(tile[ly][lx],tile[ly-1][lx]);\n      v=invar_stencil_add(v,tile[ly+1][lx]);\n      v=invar_stencil_add(v,tile[ly][lx-1]);\n      v=invar_stencil_add(v,tile[ly][lx+1]);\n      output[{}ull+y*{cols}ull+x]=v;\n    }}\n    __syncthreads();\n  }}\n}}\n",p.ir.input.offset,p.ir.output.offset));
    finish(s)
}

/// Fixed-pointer CPU analogue. Refuses unsafe ranges/FP environments before
/// stores. Allocation size and lifetime still require host validation.
pub fn emit_cpu(p: &Verified) -> Result<String, Error> {
    let mut s = header(p, true)?;
    s.push_str("#include <cfenv>\n#include <cstdint>\n#include <cstring>\n#include <limits>\nstatic_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559 && std::numeric_limits<float>::digits==24, \"binary32 required\");\nstatic float invar_stencil_add(float a,float b) { volatile float v=a+b; float n=v; std::uint32_t bits; std::memcpy(&bits,&n,4); if ((bits & 0x7fffffffU)>0x7f800000U) { bits=0x7fc00000U; std::memcpy(&n,&bits,4); } return n; }\nextern \"C\" int invar_stencil_f32_v1(const float* input,float* output) {\n");
    let [rows, cols] = p.ir.dimensions;
    if rows == 0 || cols == 0 {
        s.push_str("  (void)input; (void)output; return 0;\n}\n");
        return finish(s);
    }
    let a = &p.ir.input;
    let b = &p.ir.output;
    s.push_str("  if (!input || !output) return 2;\n  const auto ip=reinterpret_cast<std::uintptr_t>(input), op=reinterpret_cast<std::uintptr_t>(output);\n");
    s.push_str(&format!("  if ((ip & {}ull)!=0 || (op & {}ull)!=0 || ip>UINTPTR_MAX-{}ull || op>UINTPTR_MAX-{}ull) return 2;\n  if (ip<op+{}ull && op<ip+{}ull) return 2;\n",a.alignment-1,b.alignment-1,a.elements*4,b.elements*4,b.elements*4,a.elements*4));
    s.push_str("  if (std::fegetround()!=FE_TONEAREST) return 1;\n  std::fenv_t saved; if (std::feholdexcept(&saved)!=0) return 1;\n  struct Restore { std::fenv_t* s; ~Restore() { std::fesetenv(s); } } restore{&saved};\n  std::uint32_t tiny_bits=1; float tiny; std::memcpy(&tiny,&tiny_bits,4); volatile float t=tiny, z=0.0f; volatile float probe=t+z; float n=probe; std::uint32_t bits; std::memcpy(&bits,&n,4); if (bits!=1) return 1;\n");
    s.push_str("  auto load=[&](long long y,long long x)->float {\n");
    if p.ir.boundary == Boundary::Clamp {
        s.push_str(&format!("    y=y<0 ? 0 : (y>={rows}ll ? {rows}ll-1 : y); x=x<0 ? 0 : (x>={cols}ll ? {cols}ll-1 : x);\n"));
    }
    s.push_str(&format!("    if (y<0 || x<0 || y>={rows}ll || x>={cols}ll) return 0.0f;\n    return input[{}ull+static_cast<unsigned long long>(y)*{cols}ull+static_cast<unsigned long long>(x)];\n  }};\n  for (long long r=0;r<{rows}ll;++r) for (long long c=0;c<{cols}ll;++c) {{\n    float v=invar_stencil_add(load(r,c),load(r-1,c));\n    v=invar_stencil_add(v,load(r+1,c));\n    v=invar_stencil_add(v,load(r,c-1));\n    v=invar_stencil_add(v,load(r,c+1));\n    output[{}ull+static_cast<unsigned long long>(r)*{cols}ull+static_cast<unsigned long long>(c)]=v;\n  }}\n  return 0;\n}}\n",a.offset,b.offset));
    finish(s)
}
