//! Source-ordered wrapping-u32 five-point stencil, with one device phase per step.
//! Physical proof is separate from source extraction and completion publication.
use crate::{Capabilities, Provenance, Span, floating::Buffer, stencil::Boundary, transpose};

pub const VERSION: u32 = 1;
pub const NUMERIC_POLICY: &str = "wrapping-u32-v1";
pub const KERNEL_NAME: &str = "invar_stencil_u32_v1";
pub const MAX_STEPS: u32 = 64;
pub const MAX_CELL_STEPS: u64 = transpose::MAX_LOGICAL_ELEMENTS * MAX_STEPS as u64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ir {
    pub version: u32,
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
    pub steps: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Physical(transpose::Error),
    Numeric,
    Boundary,
    Steps,
    Geometry,
    Index,
    Target,
    SourceLimit,
}

#[derive(Debug)]
pub struct Verified {
    ir: Ir,
    caps: Capabilities,
    options: Vec<String>,
}
impl Verified {
    pub fn ir(&self) -> &Ir {
        &self.ir
    }
    pub fn capabilities(&self) -> Capabilities {
        self.caps
    }
    pub fn nvrtc_options(&self) -> &[String] {
        &self.options
    }
    pub fn validate_launch(&self, block: [u32; 3], grid: [u32; 3]) -> Result<(), Error> {
        if block != [16, 16, 1] || grid[0] == 0 || grid[0] > 65535 || grid[1..] != [1, 1] {
            Err(Error::Geometry)
        } else {
            Ok(())
        }
    }
    /// Slot 0 initially holds the input, slot 1 is fresh. Each phase swaps them.
    pub fn phase(&self, step: u32) -> Result<[usize; 2], Error> {
        if step >= self.ir.steps {
            return Err(Error::Steps);
        }
        let read = (step % 2) as usize;
        Ok([read, 1 - read])
    }
    pub fn final_buffer(&self) -> usize {
        (self.ir.steps % 2) as usize
    }
    /// Center, north, south, west, east; None denotes a zero boundary load.
    pub fn reads(&self, [r, c]: [u32; 2]) -> Result<[Option<u64>; 5], Error> {
        let [rows, cols] = self.ir.dimensions;
        if r >= rows || c >= cols {
            return Err(Error::Index);
        }
        let mut result = [None; 5];
        for (i, (y, x)) in [
            (i64::from(r), i64::from(c)),
            (i64::from(r) - 1, i64::from(c)),
            (i64::from(r) + 1, i64::from(c)),
            (i64::from(r), i64::from(c) - 1),
            (i64::from(r), i64::from(c) + 1),
        ]
        .into_iter()
        .enumerate()
        {
            if (y >= 0 && x >= 0 && y < i64::from(rows) && x < i64::from(cols))
                || self.ir.boundary == Boundary::Clamp
            {
                result[i] = Some(
                    self.ir.input.offset
                        + y.clamp(0, i64::from(rows) - 1) as u64 * u64::from(cols)
                        + x.clamp(0, i64::from(cols) - 1) as u64,
                );
            }
        }
        Ok(result)
    }
}

pub fn verify(ir: Ir, caps: Capabilities, origin: Provenance) -> Result<Verified, Error> {
    if ir.numeric_policy != NUMERIC_POLICY {
        return Err(Error::Numeric);
    }
    if ir.boundary_version != crate::stencil::BOUNDARY_VERSION {
        return Err(Error::Boundary);
    }
    if ir.steps > MAX_STEPS
        || u64::from(ir.dimensions[0])
            .checked_mul(u64::from(ir.dimensions[1]))
            .and_then(|n| n.checked_mul(u64::from(ir.steps)))
            .is_none_or(|n| n > MAX_CELL_STEPS)
    {
        return Err(Error::Steps);
    }
    // Ping-pong uses two identically mapped private slices. Reject a different
    // time-slice layout instead of silently remapping later steps.
    if ir.output.shape != ir.dimensions
        || ir.input.elements != ir.output.elements
        || ir.input.offset != ir.output.offset
        || ir.output.strides != ir.input.strides
    {
        return Err(Error::Physical(transpose::Error::Layout));
    }
    transpose::buffer(&ir.output).map_err(Error::Physical)?;
    let mut output = ir.output.clone();
    output.shape = [ir.dimensions[1], ir.dimensions[0]];
    output.strides = [u64::from(ir.dimensions[0]), 1];
    // Reuse the full existing version/provenance/shape/alias/capacity verifier.
    let proof = transpose::verify(
        transpose::Ir {
            version: ir.version,
            layout_version: ir.layout_version,
            span: ir.span,
            source_identity: ir.source_identity,
            graph_revision: ir.graph_revision,
            function_contract: ir.function_contract,
            dimensions: ir.dimensions,
            input: ir.input.clone(),
            output,
        },
        caps,
        origin,
    )
    .map_err(Error::Physical)?;
    let options = proof.nvrtc_options().to_vec();
    Ok(Verified { ir, caps, options })
}

fn emit(p: &Verified, metal: bool) -> Result<String, Error> {
    if metal != matches!(p.caps, Capabilities::Metal { .. }) {
        return Err(Error::Target);
    }
    let ir = &p.ir;
    let mut s = format!(
        "// invar-stencil-u32-v1 version=1 numeric=wrapping-u32-v1\n// source={} graph={} span={}:{} caps={:?}\n// required-options: {}\n// boundary={}:{} dimensions={:?} steps={} phase=separate final-buffer={} tile=18x19 block=16x16x1\n// physical-input={:?} physical-output={:?} function-contract={:?}\n",
        ir.source_identity,
        ir.graph_revision,
        ir.span.start,
        ir.span.end,
        p.caps,
        p.options.join(" "),
        ir.boundary_version,
        ir.boundary.name(),
        ir.dimensions,
        ir.steps,
        p.final_buffer(),
        ir.input,
        ir.output,
        ir.function_contract
    );
    if metal {
        s.push_str("#include <metal_stdlib>\nusing namespace metal;\nkernel void invar_stencil_u32_v1(device const uint* input [[buffer(0)]], device uint* output [[buffer(1)]], uint3 lane [[thread_position_in_threadgroup]], uint3 group [[threadgroup_position_in_grid]], uint3 groups [[threadgroups_per_grid]], uint3 threads [[threads_per_threadgroup]]) {\n");
    } else {
        s.push_str("extern \"C\" __global__ void invar_stencil_u32_v1(const unsigned int* input, unsigned int* output) {\n");
    }
    let [rows, cols] = ir.dimensions;
    if rows == 0 || cols == 0 {
        s.push_str("}\n");
        return Ok(s);
    }
    let (word, unsigned, signed, shared, tx, ty, group, groups, barrier) = if metal {
        (
            "uint",
            "ulong",
            "long",
            "threadgroup",
            "lane.x",
            "lane.y",
            "group.x",
            "groups.x",
            "threadgroup_barrier(mem_flags::mem_threadgroup)",
        )
    } else {
        (
            "unsigned int",
            "unsigned long long",
            "long long",
            "__shared__",
            "threadIdx.x",
            "threadIdx.y",
            "blockIdx.x",
            "gridDim.x",
            "__syncthreads()",
        )
    };
    let guard = if metal {
        "threads.x!=16 || threads.y!=16 || threads.z!=1 || groups.x==0 || groups.y!=1 || groups.z!=1"
    } else {
        "blockDim.x!=16 || blockDim.y!=16 || blockDim.z!=1 || gridDim.x==0 || gridDim.y!=1 || gridDim.z!=1"
    };
    let tile_cols = u64::from(cols).div_ceil(16);
    let tiles = u64::from(rows).div_ceil(16) * tile_cols;
    s.push_str(&format!("  if ({guard}) return;\n  {shared} {word} tile[18][19];\n  const {word} id={ty}*16+{tx};\n  for ({unsigned} t={group}; t<{tiles}ull; t+=({unsigned}){groups}) {{\n    const {signed} tr=({signed})(t/{tile_cols}ull)*16, tc=({signed})(t%{tile_cols}ull)*16;\n    for ({word} h=id; h<324; h+=256) {{\n      {signed} y=tr+({signed})(h/18)-1, x=tc+({signed})(h%18)-1;\n      {word} v=0;\n"));
    if ir.boundary == Boundary::Clamp {
        s.push_str(&format!("      y=y<0 ? 0 : (y>={rows}ll ? {rows}ll-1 : y); x=x<0 ? 0 : (x>={cols}ll ? {cols}ll-1 : x);\n"));
    }
    s.push_str(&format!("      if (y>=0 && x>=0 && y<{rows}ll && x<{cols}ll) v=input[{}ull+({unsigned})y*{cols}ull+({unsigned})x];\n      tile[h/18][h%18]=v;\n    }}\n    {barrier};\n    const {unsigned} y=({unsigned})tr+{ty}, x=({unsigned})tc+{tx};\n    if (y<{rows}ull && x<{cols}ull) {{\n      {word} v=tile[{ty}+1][{tx}+1];\n      v=v+tile[{ty}][{tx}+1];\n      v=v+tile[{ty}+2][{tx}+1];\n      v=v+tile[{ty}+1][{tx}];\n      v=v+tile[{ty}+1][{tx}+2];\n      output[{}ull+y*{cols}ull+x]=v;\n    }}\n    {barrier};\n  }}\n}}\n",ir.input.offset,ir.output.offset));
    if s.len() > crate::MAX_SOURCE_BYTES {
        Err(Error::SourceLimit)
    } else {
        Ok(s)
    }
}
pub fn emit_cuda(p: &Verified) -> Result<String, Error> {
    emit(p, false)
}
pub fn emit_msl(p: &Verified) -> Result<String, Error> {
    emit(p, true)
}
