//! Deterministic Metal source from the same verified physical device IR.

use std::fmt::Write;

use crate::{
    Capabilities, Coefficient, Collective, CollectiveKind, DagOp, Error, MAX_SOURCE_BYTES, MapDag,
    Op, PrefixMode, ReduceLaw, Verified,
};

/// Blocks per Metal scan chunk. Each chunk runs block, propagate and writeback
/// back to back, so the writeback's second input read hits the system cache
/// (measured on Apple M4: 64 blocks = 2 MiB of two-input data per chunk).
pub const METAL_SCAN_CHUNK_BLOCKS: u32 = 64;

/// Calibration kernel: one thread writes one word. It is compiled ahead with
/// every Metal program, so placement calibration needs no runtime compiler.
pub const PROBE_MSL: &str = "#include <metal_stdlib>\nusing namespace metal;\nkernel void invar_probe_v1(device uint* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {\n  if (i == 0u) out[0] = 1u;\n}\n";
pub const PROBE_FUNCTION: &str = "invar_probe_v1";

/// The functions `emit_msl` exports for `ir`, in dispatch order.
pub fn metal_functions(ir: &crate::Ir) -> &'static [&'static str] {
    if ir.collective.is_some() {
        &["invar_block_v1", "invar_propagate_v1", "invar_writeback_v1"]
    } else {
        &["invar_map_v1"]
    }
}

pub fn emit_msl(plan: &Verified) -> Result<String, Error> {
    if !matches!(plan.capabilities(), Capabilities::Metal { .. }) {
        return Err(Error::Toolchain);
    }
    let ir = plan.ir();
    let source = if let Some(collective) = &ir.collective {
        match collective.kind {
            CollectiveKind::Reduction { law, identity } => {
                let operator = match law {
                    ReduceLaw::WrappingSum => "+",
                    ReduceLaw::WrappingProduct => "*",
                };
                REDUCTION_MSL
                    .replace("@OP@", operator)
                    .replace("@IDENTITY@", &format!("{identity}u"))
            }
            CollectiveKind::PairReduction { .. } => emit_scan(
                collective,
                Coefficient::Input(0),
                Coefficient::Input(1),
                PrefixMode::None,
                ir.geometry.lanes,
                true,
            )?,
            CollectiveKind::AffineScan { a, b, prefix, .. } => {
                emit_scan(collective, a, b, prefix, ir.geometry.lanes, false)?
            }
        }
    } else if let Some(dag) = &ir.dag {
        emit_dag(dag)?
    } else {
        let mut source = String::from(
            "#include <metal_stdlib>\nusing namespace metal;\nkernel void invar_map_v1(device const uint* input0 [[buffer(0)]], device uint* output0 [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {\n  if (i < n) {\n    uint v = input0[i];\n",
        );
        for op in &ir.ops {
            match op {
                Op::WrappingAdd(n) => {
                    writeln!(source, "    v = v + {n}u;").map_err(|_| Error::SourceLimit)?;
                }
                Op::WrappingMul(n) => {
                    writeln!(source, "    v = v * {n}u;").map_err(|_| Error::SourceLimit)?;
                }
                Op::CheckedAdd(_) => return Err(Error::Operations),
            }
        }
        source.push_str("    output0[i] = v;\n  }\n}\n");
        source
    };
    if source.len() > MAX_SOURCE_BYTES {
        return Err(Error::SourceLimit);
    }
    Ok(source)
}

fn emit_scan(
    plan: &Collective,
    a: Coefficient,
    b: Coefficient,
    prefix: PrefixMode,
    lanes: u32,
    pair: bool,
) -> Result<String, Error> {
    let input_count = plan.inputs.len();
    let inputs = (0..input_count)
        .map(|index| format!("device const uint* input{index} [[buffer({index})]], "))
        .collect::<String>();
    let coefficient = |value: Coefficient| match value {
        Coefficient::Constant(n) => format!("{n}u"),
        Coefficient::Input(slot) => format!("input{slot}[i]"),
    };
    let coefficient_vector = |value: Coefficient| match value {
        Coefficient::Constant(n) => format!("uint4({n}u)"),
        Coefficient::Input(slot) => format!("*((device const uint4*)(input{slot} + base))"),
    };
    let write = match prefix {
        PrefixMode::None => "",
        PrefixMode::Inclusive => "if (i < n) output[i] = state;",
        PrefixMode::Exclusive => "if (i < n) output[i] = before;",
    };
    let final_index = if prefix == PrefixMode::None {
        "0u"
    } else {
        "n"
    };
    let source = SCAN_MSL
        .replace("@ITEMS@", &plan.block_items.to_string())
        .replace("@LANES@", &lanes.to_string())
        .replace("@WARPS@", &(lanes / 32).to_string())
        .replace("@INPUTS@", &inputs)
        .replace("@CHUNK@", &METAL_SCAN_CHUNK_BLOCKS.to_string())
        .replace("@BLOCK_BASE_SLOT@", &(input_count + 2).to_string())
        .replace("@WRITE_BASE_SLOT@", &(input_count + 6).to_string())
        .replace("@SUMMARY_SLOT@", &input_count.to_string())
        .replace("@LENGTH_SLOT@", &(input_count + 1).to_string())
        .replace("@PREFIX_SLOT@", &(input_count + 1).to_string())
        .replace("@OUTPUT_SLOT@", &(input_count + 2).to_string())
        .replace("@SEED_SLOT@", &(input_count + 3).to_string())
        .replace("@N_SLOT@", &(input_count + 4).to_string())
        .replace("@BLOCKS_SLOT@", &(input_count + 5).to_string())
        .replace("@A@", &coefficient(a))
        .replace("@B@", &coefficient(b))
        .replace("@AVEC@", &coefficient_vector(a))
        .replace("@BVEC@", &coefficient_vector(b))
        .replace(
            "@PREFIX_VALUE@",
            if prefix == PrefixMode::Exclusive {
                "before"
            } else {
                "state"
            },
        )
        .replace(
            "@VECTOR_STORE@",
            if prefix == PrefixMode::None {
                ""
            } else {
                "*((device uint4*)(output + base)) = out4;"
            },
        )
        .replace("@WRITE@", write)
        .replace("@FINAL_WRITE@", if pair {
            "{ InvarPair total = invar_then(prefix[blocks - 1u], summaries[blocks - 1u]); output[0] = total.a; output[1] = total.b; }"
        } else {
            "output[@FINAL_INDEX@] = invar_apply(invar_then(prefix[blocks - 1u], summaries[blocks - 1u]), seed);"
        })
        .replace("@FINAL_INDEX@", final_index);
    if source.len() > MAX_SOURCE_BYTES {
        return Err(Error::SourceLimit);
    }
    Ok(source)
}

const SCAN_MSL: &str = r#"#include <metal_stdlib>
using namespace metal;
#define INVAR_ITEMS @ITEMS@u
#define INVAR_LANES @LANES@u
#define INVAR_WARPS @WARPS@u
#define INVAR_CHUNK @CHUNK@u
struct InvarPair { uint a; uint b; };
inline InvarPair invar_identity() { return {1u, 0u}; }
inline InvarPair invar_then(InvarPair left, InvarPair right) {
  return {right.a * left.a, right.a * left.b + right.b};
}
inline uint invar_apply(InvarPair pair, uint seed) { return pair.a * seed + pair.b; }
inline InvarPair invar_before(InvarPair value, uint offset) {
  return {simd_shuffle_up(value.a, offset), simd_shuffle_up(value.b, offset)};
}
kernel void invar_block_v1(@INPUTS@device InvarPair* summaries [[buffer(@SUMMARY_SLOT@)]], constant uint& n [[buffer(@LENGTH_SLOT@)]], device const uint* chunk_base [[buffer(@BLOCK_BASE_SLOT@)]], uint t [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
  uint block = chunk_base[0] + group;
  uint lane = t & 31u;
  uint warp = t >> 5u;
  uint base = block * INVAR_ITEMS + t * (INVAR_ITEMS / INVAR_LANES);
  InvarPair value = invar_identity();
#if INVAR_LANES == 1024u
  // Full groups start at a four-element boundary; ragged tails use scalar loads.
  if (base + 4u <= n) {
    uint4 av = @AVEC@;
    uint4 bv = @BVEC@;
    value = invar_then(value, InvarPair{av.x, bv.x});
    value = invar_then(value, InvarPair{av.y, bv.y});
    value = invar_then(value, InvarPair{av.z, bv.z});
    value = invar_then(value, InvarPair{av.w, bv.w});
  } else
#endif
  {
  for (uint j = 0u; j < INVAR_ITEMS / INVAR_LANES; ++j) {
    uint i = base + j;
    InvarPair item = (i < n) ? InvarPair{@A@, @B@} : invar_identity();
    value = invar_then(value, item);
  }
  }
  for (uint offset = 1u; offset < 32u; offset <<= 1u) {
    InvarPair before = invar_before(value, offset);
    if (lane >= offset) value = invar_then(before, value);
  }
  threadgroup InvarPair warp_totals[INVAR_WARPS];
  if (lane == 31u) warp_totals[warp] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (warp == 0u) {
    InvarPair total = (lane < INVAR_WARPS) ? warp_totals[lane] : invar_identity();
    for (uint offset = 1u; offset < INVAR_WARPS; offset <<= 1u) {
      InvarPair before = invar_before(total, offset);
      if (lane >= offset) total = invar_then(before, total);
    }
    if (lane == INVAR_WARPS - 1u) summaries[block] = total;
  }
}
// One dispatch per chunk of at most INVAR_CHUNK blocks. carries[0] is the
// composition of every earlier chunk; carries[1] receives this chunk's result.
kernel void invar_propagate_v1(device const InvarPair* summaries [[buffer(0)]], device InvarPair* prefix [[buffer(1)]], constant uint& blocks [[buffer(2)]], device const uint* chunk_base [[buffer(3)]], device InvarPair* carries [[buffer(4)]], uint t [[thread_index_in_threadgroup]]) {
  const uint prop_lanes = 1024u;
  uint lane = t & 31u;
  uint warp = t >> 5u;
  uint first = chunk_base[0];
  uint count = min(INVAR_CHUNK, blocks - first);
  uint chunk = (count + prop_lanes - 1u) / prop_lanes;
  uint base = first + t * chunk;
  uint end = first + count;
  threadgroup InvarPair carry_in;
  if (t == 0u) carry_in = carries[0];
  InvarPair local = invar_identity();
  for (uint j = 0u; j < chunk; ++j) {
    uint i = base + j;
    if (i < end) local = invar_then(local, summaries[i]);
  }
  InvarPair within = local;
  for (uint offset = 1u; offset < 32u; offset <<= 1u) {
    InvarPair before = invar_before(within, offset);
    if (lane >= offset) within = invar_then(before, within);
  }
  threadgroup InvarPair warp_totals[32];
  threadgroup InvarPair warp_prefix[32];
  if (lane == 31u) warp_totals[warp] = within;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (warp == 0u) {
    InvarPair total = warp_totals[lane];
    for (uint offset = 1u; offset < 32u; offset <<= 1u) {
      InvarPair before = invar_before(total, offset);
      if (lane >= offset) total = invar_then(before, total);
    }
    warp_prefix[lane] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  InvarPair before_lane = invar_before(within, 1u);
  InvarPair before_warp = (warp == 0u) ? invar_identity() : warp_prefix[warp - 1u];
  InvarPair carry = (lane == 0u) ? before_warp : invar_then(before_warp, before_lane);
  carry = invar_then(carry_in, carry);
  for (uint j = 0u; j < chunk; ++j) {
    uint i = base + j;
    if (i < end) {
      prefix[i] = carry;
      carry = invar_then(carry, summaries[i]);
    }
  }
  if (t == 0u) carries[1] = invar_then(carry_in, warp_prefix[31]);
}
kernel void invar_writeback_v1(@INPUTS@device const InvarPair* summaries [[buffer(@SUMMARY_SLOT@)]], device const InvarPair* prefix [[buffer(@PREFIX_SLOT@)]], device uint* output [[buffer(@OUTPUT_SLOT@)]], constant uint& seed [[buffer(@SEED_SLOT@)]], constant uint& n [[buffer(@N_SLOT@)]], constant uint& blocks [[buffer(@BLOCKS_SLOT@)]], device const uint* chunk_base [[buffer(@WRITE_BASE_SLOT@)]], uint t [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
  uint block = chunk_base[0] + group;
  uint lane = t & 31u;
  uint warp = t >> 5u;
  uint base = block * INVAR_ITEMS + t * (INVAR_ITEMS / INVAR_LANES);
  InvarPair value = invar_identity();
#if INVAR_LANES == 1024u
  if (base + 4u <= n) {
    uint4 av = @AVEC@;
    uint4 bv = @BVEC@;
    value = invar_then(value, InvarPair{av.x, bv.x});
    value = invar_then(value, InvarPair{av.y, bv.y});
    value = invar_then(value, InvarPair{av.z, bv.z});
    value = invar_then(value, InvarPair{av.w, bv.w});
  } else
#endif
  {
  for (uint j = 0u; j < INVAR_ITEMS / INVAR_LANES; ++j) {
    uint i = base + j;
    InvarPair item = (i < n) ? InvarPair{@A@, @B@} : invar_identity();
    value = invar_then(value, item);
  }
  }
  for (uint offset = 1u; offset < 32u; offset <<= 1u) {
    InvarPair before = invar_before(value, offset);
    if (lane >= offset) value = invar_then(before, value);
  }
  threadgroup InvarPair warp_totals[INVAR_WARPS];
  threadgroup InvarPair warp_prefix[INVAR_WARPS];
  if (lane == 31u) warp_totals[warp] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (warp == 0u) {
    InvarPair total = (lane < INVAR_WARPS) ? warp_totals[lane] : invar_identity();
    for (uint offset = 1u; offset < INVAR_WARPS; offset <<= 1u) {
      InvarPair before = invar_before(total, offset);
      if (lane >= offset) total = invar_then(before, total);
    }
    if (lane < INVAR_WARPS) warp_prefix[lane] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  InvarPair before_lane = invar_before(value, 1u);
  InvarPair before_warp = (warp == 0u) ? invar_identity() : warp_prefix[warp - 1u];
  InvarPair previous = (lane == 0u) ? before_warp : invar_then(before_warp, before_lane);
  InvarPair start = invar_then(prefix[block], previous);
  uint state = invar_apply(start, seed);
#if INVAR_LANES == 1024u
  if (base + 4u <= n) {
    uint4 av = @AVEC@;
    uint4 bv = @BVEC@;
    uint4 out4;
    uint before = state;
    state = av.x * state + bv.x; out4.x = @PREFIX_VALUE@;
    before = state;
    state = av.y * state + bv.y; out4.y = @PREFIX_VALUE@;
    before = state;
    state = av.z * state + bv.z; out4.z = @PREFIX_VALUE@;
    before = state;
    state = av.w * state + bv.w; out4.w = @PREFIX_VALUE@;
    @VECTOR_STORE@
  } else
#endif
  {
  for (uint j = 0u; j < INVAR_ITEMS / INVAR_LANES; ++j) {
    uint i = base + j;
    if (i < n) {
      uint before = state;
      state = @A@ * state + @B@;
      @WRITE@
    }
  }
  }
  // The last chunk's propagation has completed every prefix before this dispatch.
  if (block == blocks - 1u && t == 0u)
    @FINAL_WRITE@
}
"#;

const REDUCTION_MSL: &str = r#"#include <metal_stdlib>
using namespace metal;
#define INVAR_ITEMS 4096u
#define INVAR_LANES 256u
inline uint invar_combine(uint a, uint b) { return a @OP@ b; }
kernel void invar_block_v1(device const uint* input [[buffer(0)]], device uint* summaries [[buffer(1)]], constant uint& n [[buffer(2)]], uint t [[thread_index_in_threadgroup]], uint block [[threadgroup_position_in_grid]]) {
  uint value = @IDENTITY@;
  uint base = block * INVAR_ITEMS + t;
  for (uint row = 0u; row < INVAR_ITEMS / INVAR_LANES; ++row) {
    uint i = base + row * INVAR_LANES;
    if (i < n) value = invar_combine(value, input[i]);
  }
  threadgroup uint lanes[INVAR_LANES];
  lanes[t] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint offset = INVAR_LANES / 2u; offset > 0u; offset >>= 1u) {
    if (t < offset) lanes[t] = invar_combine(lanes[t], lanes[t + offset]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (t == 0u) summaries[block] = lanes[0];
}
kernel void invar_propagate_v1(device const uint* summaries [[buffer(0)]], device uint* total [[buffer(1)]], constant uint& blocks [[buffer(2)]], uint t [[thread_index_in_threadgroup]]) {
  uint value = @IDENTITY@;
  for (uint i = t; i < blocks; i += INVAR_LANES) value = invar_combine(value, summaries[i]);
  threadgroup uint lanes[INVAR_LANES];
  lanes[t] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint offset = INVAR_LANES / 2u; offset > 0u; offset >>= 1u) {
    if (t < offset) lanes[t] = invar_combine(lanes[t], lanes[t + offset]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (t == 0u) total[0] = lanes[0];
}
kernel void invar_writeback_v1(device const uint* total [[buffer(0)]], device uint* output [[buffer(1)]], constant uint& seed [[buffer(2)]], uint t [[thread_index_in_threadgroup]]) {
  if (t == 0u) output[0] = invar_combine(seed, total[0]);
}
"#;

/// The grid stride comes from `threads_per_grid`, which equals the dispatched
/// `blocks * lanes`, so the source does not depend on the input length and a
/// compiled-ahead library serves every length of the same region.
fn emit_dag(dag: &MapDag) -> Result<String, Error> {
    let mut source =
        String::from("#include <metal_stdlib>\nusing namespace metal;\nkernel void invar_map_v1(");
    let mut index = 0;
    for input in 0..dag.inputs.len() {
        if index > 0 {
            source.push_str(", ");
        }
        write!(
            source,
            "device const uint* input{input} [[buffer({index})]]"
        )
        .map_err(|_| Error::SourceLimit)?;
        index += 1;
    }
    for output in 0..dag.outputs.len() {
        if index > 0 {
            source.push_str(", ");
        }
        write!(source, "device uint* output{output} [[buffer({index})]]")
            .map_err(|_| Error::SourceLimit)?;
        index += 1;
    }
    writeln!(
        source,
        ", constant uint& n [[buffer({index})]], uint base [[thread_position_in_grid]], uint stride [[threads_per_grid]]) {{\n  for (uint i = base; i < n; i += stride) {{"
    )
    .map_err(|_| Error::SourceLimit)?;
    for (index, node) in dag.nodes.iter().enumerate() {
        match node.op {
            DagOp::Input(slot) => writeln!(source, "    uint v{index} = input{slot}[i];"),
            DagOp::WrappingAdd { input, constant } => {
                writeln!(source, "    uint v{index} = v{input} + {constant}u;")
            }
            DagOp::WrappingMul { input, constant } => {
                writeln!(source, "    uint v{index} = v{input} * {constant}u;")
            }
            DagOp::ZipAdd { left, right } => {
                writeln!(source, "    uint v{index} = v{left} + v{right};")
            }
            DagOp::ZipMul { left, right } => {
                writeln!(source, "    uint v{index} = v{left} * v{right};")
            }
        }
        .map_err(|_| Error::SourceLimit)?;
    }
    for (index, stage) in dag.stage_outputs.iter().enumerate() {
        writeln!(source, "    output{index}[i] = v{stage};").map_err(|_| Error::SourceLimit)?;
    }
    source.push_str("  }\n}\n");
    Ok(source)
}
