//! Metal bit transpose emitted from the same sealed physical IR as CUDA.
use super::*;

pub fn emit_msl(plan: &Verified) -> Result<String, Error> {
    if !matches!(plan.capabilities(), Capabilities::Metal { .. }) {
        return Err(Error::Capabilities);
    }
    let mut text = header(plan, false)?;
    text.push_str("#include <metal_stdlib>\nusing namespace metal;\nkernel void invar_transpose_bits_v1(device const uint* input [[buffer(0)]], device uint* output [[buffer(1)]], uint3 lane [[thread_position_in_threadgroup]], uint3 group [[threadgroup_position_in_grid]], uint3 groups [[threadgroups_per_grid]], uint3 threads [[threads_per_threadgroup]]) {\n");
    let [rows, cols] = plan.ir().dimensions;
    if rows == 0 || cols == 0 {
        text.push_str("}\n");
        return finish(text);
    }
    let tile_cols = u64::from(cols).div_ceil(16);
    let tiles = u64::from(rows).div_ceil(16) * tile_cols;
    text.push_str("  if (threads.x != 16 || threads.y != 16 || threads.z != 1 || groups.x == 0 || groups.y != 1 || groups.z != 1) return;\n  threadgroup uint tile[16][17];\n");
    text.push_str(&format!("  for (ulong t=group.x; t<{tiles}ul; t+=groups.x) {{\n    ulong tr=(t/{tile_cols}ul)*16ul, tc=(t%{tile_cols}ul)*16ul;\n    ulong x=tc+lane.x, y=tr+lane.y;\n    if (y<{rows}ul && x<{cols}ul) tile[lane.y][lane.x]=input[{}ul+y*{cols}ul+x];\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    ulong r=tc+lane.y, c=tr+lane.x;\n    if (r<{cols}ul && c<{rows}ul) output[{}ul+r*{rows}ul+c]=tile[lane.x][lane.y];\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n  }}\n}}\n", plan.ir().input.offset, plan.ir().output.offset));
    finish(text)
}
