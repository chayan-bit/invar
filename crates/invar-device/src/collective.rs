//! Shared physical contract for hierarchical wrapping reductions and ordered scans.

use super::*;

pub const BLOCK_ITEMS: u32 = 4096;
pub const LANES: u32 = 256;
pub const SCAN_LANES: u32 = 512;
pub const CUDA_PREFIX_SEGMENT: u32 = 512;
pub const MAX_ELEMENTS: u32 = super::MAX_DAG_ELEMENTS;
pub const MAX_TRANSFER_BYTES: u64 = 4 * 1024 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceLaw {
    WrappingSum,
    WrappingProduct,
}

impl ReduceLaw {
    pub const fn identity(self) -> u32 {
        match self {
            Self::WrappingSum => 0,
            Self::WrappingProduct => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coefficient {
    Constant(u32),
    Input(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixMode {
    None,
    Inclusive,
    Exclusive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollectiveKind {
    Reduction {
        law: ReduceLaw,
        identity: u32,
    },
    PairReduction {
        identity_a: u32,
        identity_b: u32,
    },
    AffineScan {
        a: Coefficient,
        b: Coefficient,
        prefix: PrefixMode,
        identity_a: u32,
        identity_b: u32,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Collective {
    pub kind: CollectiveKind,
    pub inputs: Vec<Buffer>,
    pub output: Buffer,
    pub block_items: u32,
}

pub(super) fn verify_collective(
    ir: &Ir,
    plan: &Collective,
    caps: crate::Capabilities,
) -> Result<(), Error> {
    let scan_lanes_valid = match caps {
        crate::Capabilities::Metal { .. } => matches!(ir.geometry.lanes, 512 | 1024),
        _ => ir.geometry.lanes == SCAN_LANES,
    };
    if ir.length > MAX_ELEMENTS {
        return Err(Error::ResourceLimit("collective exceeds 2^28 elements"));
    }
    if ir.dag.is_some()
        || !ir.ops.is_empty()
        || plan.block_items != BLOCK_ITEMS
        || if matches!(
            plan.kind,
            CollectiveKind::AffineScan { .. } | CollectiveKind::PairReduction { .. }
        ) {
            !scan_lanes_valid
        } else {
            ir.geometry.lanes != LANES
        }
        || ir.geometry.blocks != ir.length.div_ceil(BLOCK_ITEMS).max(1)
        || plan.inputs.len() > 2
        || ir.input != *plan.inputs.first().unwrap_or(&plan.output)
        || ir.output != plan.output
    {
        return Err(Error::Operations);
    }
    let output_len = match plan.kind {
        CollectiveKind::PairReduction {
            identity_a,
            identity_b,
        } => {
            if (identity_a, identity_b) != (1, 0) || plan.inputs.len() != 2 {
                return Err(Error::Operations);
            }
            2
        }
        CollectiveKind::Reduction { law, identity } => {
            if plan.inputs.len() != 1 || identity != law.identity() {
                return Err(Error::Operations);
            }
            1
        }
        CollectiveKind::AffineScan {
            a,
            b,
            prefix,
            identity_a,
            identity_b,
        } => {
            if (identity_a, identity_b) != (1, 0) {
                return Err(Error::Operations);
            }
            for source in [a, b] {
                if let Coefficient::Input(slot) = source
                    && !usize::try_from(slot).is_ok_and(|n| n < plan.inputs.len())
                {
                    return Err(Error::Operations);
                }
            }
            if prefix == PrefixMode::None {
                1
            } else {
                ir.length.checked_add(1).ok_or(Error::Bounds)?
            }
        }
    };
    // Runners publish the last output word as final state; reject unwritten padding.
    // Keep this separate from the lower bound so the PairReduction lane composes.
    if plan.output.elements > output_len {
        return Err(Error::Bounds);
    }
    if plan.output.address != AddressSpace::Device
        || plan.output.layout != Layout::ContiguousU32
        || plan
            .inputs
            .iter()
            .any(|b| b.address != AddressSpace::Device || b.layout != Layout::ContiguousU32)
    {
        return Err(Error::Layout);
    }
    if plan.output.elements < output_len
        || (matches!(plan.kind, CollectiveKind::PairReduction { .. }) && plan.output.elements != 2)
        || plan.inputs.iter().any(|b| b.elements < ir.length)
    {
        return Err(Error::Bounds);
    }
    let mut ids = std::collections::BTreeSet::new();
    if !plan
        .inputs
        .iter()
        .chain([&plan.output])
        .all(|b| ids.insert(b.id))
    {
        return Err(Error::Alias);
    }
    let transfer = u64::from(ir.length)
        .checked_mul(u64::try_from(plan.inputs.len()).map_err(|_| Error::Bounds)? + 2)
        .and_then(|n| n.checked_mul(4))
        .ok_or(Error::Bounds)?;
    if transfer > MAX_TRANSFER_BYTES {
        return Err(Error::ResourceLimit(
            "collective exceeds 4 GiB transfer budget",
        ));
    }
    Ok(())
}

pub(super) fn emit_cuda(ir: &Ir, plan: &Collective) -> Result<String, Error> {
    let source = match plan.kind {
        CollectiveKind::Reduction { law, identity } => {
            let operator = match law {
                ReduceLaw::WrappingSum => "+",
                ReduceLaw::WrappingProduct => "*",
            };
            REDUCTION_CUDA
                .replace("@IDENTITY@", &format!("{identity}u"))
                .replace("@OP@", operator)
        }
        CollectiveKind::PairReduction { .. } => {
            // Same ordered block summaries/propagation and launch ABI as final scan.
            SCAN_CUDA
                .replace("@PREFIX_SEGMENT@", &CUDA_PREFIX_SEGMENT.to_string())
                .replace("@WRITEBACK@", PAIR_FINAL_CUDA)
                .replace(
                    "@INPUTS@",
                    "const unsigned int* input0, const unsigned int* input1, ",
                )
                .replace("@A@", "input0[i]")
                .replace("@B@", "input1[i]")
        }
        CollectiveKind::AffineScan { a, b, prefix, .. } => {
            let inputs = (0..plan.inputs.len())
                .map(|i| format!("const unsigned int* input{i}, "))
                .collect::<String>();
            let coefficient = |value: Coefficient| match value {
                Coefficient::Constant(n) => format!("{n}u"),
                Coefficient::Input(slot) => format!("input{slot}[i]"),
            };
            let (writeback, write) = match prefix {
                PrefixMode::None => (SCAN_FINAL_CUDA, ""),
                PrefixMode::Inclusive => (
                    SCAN_PREFIX_CUDA,
                    "if (i < n) output[i] = invar_apply(value, invar_apply(before_warp, invar_apply(row_carry, seed)));",
                ),
                PrefixMode::Exclusive => (
                    SCAN_PREFIX_CUDA,
                    "InvarPair previous_lane = {__shfl_up_sync(0xffffffffu, row_prefix.a, 1u), __shfl_up_sync(0xffffffffu, row_prefix.b, 1u)}; InvarPair previous = (t == 0u) ? invar_identity() : ((lane == 0u) ? warp_prefix[warp - 1u] : previous_lane); if (i < n) output[i] = invar_apply(previous, invar_apply(row_carry, seed));",
                ),
            };
            let final_index = if prefix == PrefixMode::None {
                "0u"
            } else {
                "n"
            };
            SCAN_CUDA
                .replace("@PREFIX_SEGMENT@", &CUDA_PREFIX_SEGMENT.to_string())
                .replace("@WRITEBACK@", writeback)
                .replace("@INPUTS@", &inputs)
                .replace("@A@", &coefficient(a))
                .replace("@B@", &coefficient(b))
                .replace("@WRITE@", write)
                .replace("@FINAL_INDEX@", final_index)
        }
    };
    if source.len() > super::MAX_SOURCE_BYTES || ir.length > MAX_ELEMENTS {
        return Err(Error::SourceLimit);
    }
    Ok(source)
}

const REDUCTION_CUDA: &str = r#"#define INVAR_ITEMS 4096u
#define INVAR_LANES 256u
__device__ __forceinline__ unsigned int invar_combine(unsigned int a, unsigned int b) { return a @OP@ b; }
extern "C" __global__ void invar_block_v1(const unsigned int* input, unsigned int* summaries, unsigned int n) {
  unsigned int t = threadIdx.x;
  unsigned int base = blockIdx.x * INVAR_ITEMS + t;
  unsigned int value = @IDENTITY@;
  for (unsigned int row = 0u; row < INVAR_ITEMS / INVAR_LANES; ++row) {
    unsigned int i = base + row * INVAR_LANES;
    if (i < n) value = invar_combine(value, input[i]);
  }
  __shared__ unsigned int lanes[INVAR_LANES];
  lanes[t] = value;
  __syncthreads();
  for (unsigned int offset = INVAR_LANES / 2u; offset > 0u; offset >>= 1u) {
    if (t < offset) lanes[t] = invar_combine(lanes[t], lanes[t + offset]);
    __syncthreads();
  }
  if (t == 0u) summaries[blockIdx.x] = lanes[0];
}
extern "C" __global__ void invar_propagate_v1(const unsigned int* summaries, unsigned int* total, unsigned int blocks) {
  if (blockIdx.x == 0u) {
    unsigned int t = threadIdx.x;
    unsigned int value = @IDENTITY@;
    for (unsigned int i = t; i < blocks; i += INVAR_LANES) value = invar_combine(value, summaries[i]);
    __shared__ unsigned int lanes[INVAR_LANES];
    lanes[t] = value;
    __syncthreads();
    for (unsigned int offset = INVAR_LANES / 2u; offset > 0u; offset >>= 1u) {
      if (t < offset) lanes[t] = invar_combine(lanes[t], lanes[t + offset]);
      __syncthreads();
    }
    if (t == 0u) total[0] = lanes[0];
  }
}
extern "C" __global__ void invar_writeback_v1(const unsigned int* total, unsigned int* output, unsigned int seed) {
  if (threadIdx.x == 0u && blockIdx.x == 0u) output[0] = invar_combine(seed, total[0]);
}
"#;

const SCAN_CUDA: &str = r#"#define INVAR_ITEMS 4096u
#define INVAR_LANES 512u
#define INVAR_PREFIX_SEGMENT @PREFIX_SEGMENT@u
typedef struct { unsigned int a; unsigned int b; } InvarPair;
__device__ __forceinline__ InvarPair invar_identity() { return {1u, 0u}; }
__device__ __forceinline__ InvarPair invar_then(InvarPair left, InvarPair right) {
  return {right.a * left.a, right.a * left.b + right.b};
}
__device__ __forceinline__ unsigned int invar_apply(InvarPair value, unsigned int seed) { return value.a * seed + value.b; }
__device__ __forceinline__ InvarPair invar_shuffle_down(InvarPair value, unsigned int offset) {
  return {__shfl_down_sync(0xffffffffu, value.a, offset), __shfl_down_sync(0xffffffffu, value.b, offset)};
}
extern "C" __global__ void invar_block_v1(@INPUTS@InvarPair* summaries, unsigned int n) {
  unsigned int t = threadIdx.x;
  unsigned int warp = t >> 5u;
  unsigned int lane = t & 31u;
  unsigned int base = blockIdx.x * INVAR_ITEMS + t * (INVAR_ITEMS / INVAR_LANES);
  InvarPair aggregate = invar_identity();
  for (unsigned int j = 0u; j < INVAR_ITEMS / INVAR_LANES; ++j) {
    unsigned int i = base + j;
    InvarPair item = (i < n) ? InvarPair{@A@, @B@} : invar_identity();
    aggregate = invar_then(aggregate, item);
  }
  for (unsigned int offset = 1u; offset < 32u; offset <<= 1u) {
    InvarPair right = invar_shuffle_down(aggregate, offset);
    if ((lane & (2u * offset - 1u)) == 0u) aggregate = invar_then(aggregate, right);
  }
  __shared__ InvarPair warp_totals[16];
  if (lane == 0u) warp_totals[warp] = aggregate;
  __syncthreads();
  if (warp == 0u) {
    InvarPair total = (lane < 16u) ? warp_totals[lane] : invar_identity();
    for (unsigned int offset = 1u; offset < 16u; offset <<= 1u) {
      InvarPair right = invar_shuffle_down(total, offset);
      if ((lane & (2u * offset - 1u)) == 0u) total = invar_then(total, right);
    }
    if (lane == 0u) summaries[blockIdx.x] = total;
  }
}
extern "C" __global__ void invar_propagate_v1(const InvarPair* summaries, InvarPair* prefix, unsigned int blocks) {
  unsigned int t = threadIdx.x;
  unsigned int lane = t & 31u;
  unsigned int warp = t >> 5u;
  unsigned int segment_start = blockIdx.x * INVAR_PREFIX_SEGMENT;
  InvarPair aggregate = invar_identity();
  for (unsigned int j = 0u; j < blockIdx.x; ++j) {
    unsigned int i = t * blockIdx.x + j;
    aggregate = invar_then(aggregate, summaries[i]);
  }
  for (unsigned int offset = 1u; offset < 32u; offset <<= 1u) {
    InvarPair right = invar_shuffle_down(aggregate, offset);
    if ((lane & (2u * offset - 1u)) == 0u) aggregate = invar_then(aggregate, right);
  }
  __shared__ InvarPair warp_totals[16];
  __shared__ InvarPair warp_prefix[16];
  __shared__ InvarPair segment_carry;
  if (lane == 0u) warp_totals[warp] = aggregate;
  __syncthreads();
  if (warp == 0u) {
    InvarPair total = (lane < 16u) ? warp_totals[lane] : invar_identity();
    for (unsigned int offset = 1u; offset < 16u; offset <<= 1u) {
      InvarPair right = invar_shuffle_down(total, offset);
      if ((lane & (2u * offset - 1u)) == 0u) total = invar_then(total, right);
    }
    if (lane == 0u) segment_carry = total;
  }
  __syncthreads();
  InvarPair carry = segment_carry;
  unsigned int i = segment_start + t;
  InvarPair value = (i < blocks) ? summaries[i] : invar_identity();
  for (unsigned int offset = 1u; offset < 32u; offset <<= 1u) {
    InvarPair before = {__shfl_up_sync(0xffffffffu, value.a, offset), __shfl_up_sync(0xffffffffu, value.b, offset)};
    if (lane >= offset) value = invar_then(before, value);
  }
  if (lane == 31u) warp_totals[warp] = value;
  __syncthreads();
  if (warp == 0u) {
    InvarPair total = (lane < 16u) ? warp_totals[lane] : invar_identity();
    for (unsigned int offset = 1u; offset < 16u; offset <<= 1u) {
      InvarPair before = {__shfl_up_sync(0xffffffffu, total.a, offset), __shfl_up_sync(0xffffffffu, total.b, offset)};
      if (lane >= offset) total = invar_then(before, total);
    }
    if (lane < 16u) warp_prefix[lane] = total;
  }
  __syncthreads();
  InvarPair before_lane = {__shfl_up_sync(0xffffffffu, value.a, 1u), __shfl_up_sync(0xffffffffu, value.b, 1u)};
  InvarPair before_warp = (warp == 0u) ? invar_identity() : warp_prefix[warp - 1u];
  InvarPair local = (lane == 0u) ? before_warp : invar_then(before_warp, before_lane);
  if (i < blocks) prefix[i] = invar_then(carry, local);
}
@WRITEBACK@
"#;

const SCAN_FINAL_CUDA: &str = r#"extern "C" __global__ void invar_writeback_v1(@INPUTS@const InvarPair* summaries, const InvarPair* prefix, unsigned int* output, unsigned int seed, unsigned int n, unsigned int blocks) {
  if (blockIdx.x == 0u && threadIdx.x == 0u)
    output[@FINAL_INDEX@] = invar_apply(invar_then(prefix[blocks - 1u], summaries[blocks - 1u]), seed);
}"#;

const SCAN_PREFIX_CUDA: &str = r#"extern "C" __global__ void invar_writeback_v1(@INPUTS@const InvarPair* summaries, const InvarPair* prefix, unsigned int* output, unsigned int seed, unsigned int n, unsigned int blocks) {
  unsigned int t = threadIdx.x;
  unsigned int warp = t >> 5u;
  unsigned int lane = t & 31u;
  unsigned int base = blockIdx.x * INVAR_ITEMS;
  __shared__ InvarPair warp_totals[16];
  __shared__ InvarPair warp_prefix[16];
  __shared__ InvarPair carry;
  if (t == 0u) carry = prefix[blockIdx.x];
  __syncthreads();
  for (unsigned int row = 0u; row < INVAR_ITEMS / INVAR_LANES; ++row) {
    InvarPair row_carry = carry;
    unsigned int i = base + row * INVAR_LANES + t;
    InvarPair value = (i < n) ? InvarPair{@A@, @B@} : invar_identity();
    for (unsigned int offset = 1u; offset < 32u; offset <<= 1u) {
      InvarPair before = {__shfl_up_sync(0xffffffffu, value.a, offset), __shfl_up_sync(0xffffffffu, value.b, offset)};
      if (lane >= offset) value = invar_then(before, value);
    }
    if (lane == 31u) warp_totals[warp] = value;
    __syncthreads();
    if (warp == 0u) {
      InvarPair whole_warp = (lane < 16u) ? warp_totals[lane] : invar_identity();
      for (unsigned int offset = 1u; offset < 16u; offset <<= 1u) {
        InvarPair before = {__shfl_up_sync(0xffffffffu, whole_warp.a, offset), __shfl_up_sync(0xffffffffu, whole_warp.b, offset)};
        if (lane >= offset) whole_warp = invar_then(before, whole_warp);
      }
      if (lane < 16u) warp_prefix[lane] = whole_warp;
    }
    __syncthreads();
    InvarPair before_warp = (warp == 0u) ? invar_identity() : warp_prefix[warp - 1u];
    InvarPair row_prefix = invar_then(before_warp, value);
    @WRITE@
    if (t == 0u) carry = invar_then(row_carry, warp_prefix[15]);
    __syncthreads();
  }
  if (blockIdx.x == 0u && t == 0u)
    output[@FINAL_INDEX@] = invar_apply(invar_then(prefix[blocks - 1u], summaries[blocks - 1u]), seed);
}"#;

const PAIR_FINAL_CUDA: &str = r#"extern "C" __global__ void invar_writeback_v1(@INPUTS@const InvarPair* summaries, const InvarPair* prefix, unsigned int* output, unsigned int seed, unsigned int n, unsigned int blocks) {
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    InvarPair total = invar_then(prefix[blocks - 1u], summaries[blocks - 1u]);
    output[0] = total.a;
    output[1] = total.b;
  }
}"#;
