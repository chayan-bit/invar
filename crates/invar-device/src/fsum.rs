//! v0.2 deterministic float reductions on devices: launch geometry and the
//! fixed CUDA and Metal kernel sources. Both backends compute the same
//! partials; the host finishes them with the shared `fsum` code.
//!
//! Tree kernels: each 32-lane SIMD group reduces an aligned run of `32 << t`
//! leaves in `2^t` steps of 32 adjacent leaves (one leaf per lane, then an
//! aligned shuffle tree), the steps as an aligned tree, and the group's eight
//! SIMD nodes likewise; missing right children carry. One output word per
//! group is the node at level `8 + t`; the host folds those with the same rule.
//!
//! Exact kernels: each thread accumulates binary32 values (or exact products)
//! as 32-bit digits held in 64-bit words, the group sums each digit, and one
//! row of `DIGITS + 1` words (digits, then flags) per group is the output.

/// Threads per group; also the CUDA block size. Both targets use 32-lane SIMD.
pub const LANES: u32 = 256;
/// Leaf block of the tree shape (spec: frozen at 8).
pub const LEAF: u32 = 8;
/// 32-bit digits of an exact dot accumulator; sums use the first ten.
pub const DIGITS: usize = 19;
/// Output words per exact group row: the digits, then the flag word.
pub const ROW: usize = DIGITS + 1;
/// Upper bound on exact groups; more groups only add host merging.
pub const MAX_EXACT_GROUPS: u32 = 1024;
/// Target elements per exact thread before another group is added.
pub const EXACT_PER_THREAD: u32 = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    SumTree,
    DotTree,
    SumExact,
    DotExact,
}

impl Op {
    pub fn is_dot(self) -> bool {
        matches!(self, Self::DotTree | Self::DotExact)
    }

    pub fn is_exact(self) -> bool {
        matches!(self, Self::SumExact | Self::DotExact)
    }

    pub fn kernel(self) -> &'static str {
        if self.is_exact() {
            "invar_fsum_exact_v1"
        } else {
            "invar_fsum_tree_v1"
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// Zero elements: the host returns `+0` without a launch.
    Empty,
    /// Above the u32 element bound or the leaf-chunk exponent bound.
    Limit,
}

/// One verified launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    pub op: Op,
    pub elements: u32,
    pub groups: u32,
    /// Tree: each SIMD group reduces `2^leaf_shift` steps of 32 leaves.
    pub leaf_shift: u32,
    /// Kernel parameter words: elements, dot flag, leaf shift, digits.
    pub params: [u32; 4],
}

impl Launch {
    pub fn new(op: Op, elements: usize, leaf_shift: u32) -> Result<Self, Error> {
        let n = u32::try_from(elements).map_err(|_| Error::Limit)?;
        if n == 0 {
            return Err(Error::Empty);
        }
        if leaf_shift > 3 {
            return Err(Error::Limit);
        }
        let groups = if op.is_exact() {
            n.div_ceil(LANES * EXACT_PER_THREAD)
                .clamp(1, MAX_EXACT_GROUPS)
        } else {
            n.div_ceil(LEAF).div_ceil(LANES << leaf_shift)
        };
        let digits = if op.is_dot() { DIGITS as u32 } else { 10 };
        Ok(Self {
            op,
            elements: n,
            groups,
            leaf_shift,
            params: [n, u32::from(op.is_dot()), leaf_shift, digits],
        })
    }

    /// Output words: one f32 word per tree group, or one i64 row per exact group.
    pub fn output_words(&self) -> usize {
        if self.op.is_exact() {
            self.groups as usize * ROW
        } else {
            self.groups as usize
        }
    }

    /// Level of each tree output node above the leaves.
    pub fn node_level(&self) -> u32 {
        8 + self.leaf_shift
    }
}

/// Shared deposit logic, written once in the common C subset of CUDA and MSL.
/// `fsum_i64`/`fsum_u64` are the backend's 64-bit integer spellings.
const DEPOSIT: &str = r#"
static fsum_inline void fsum_add_at(fsum_thread fsum_i64* d, uint pos, uint value, bool neg) {
    fsum_u64 wide = ((fsum_u64)value) << (pos & 31u);
    uint q = pos >> 5;
    fsum_i64 lo = (fsum_i64)(wide & 0xffffffffull);
    fsum_i64 hi = (fsum_i64)(wide >> 32);
    if (neg) { d[q] -= lo; d[q + 1] -= hi; } else { d[q] += lo; d[q + 1] += hi; }
}

static fsum_inline void fsum_deposit(fsum_thread fsum_i64* d, fsum_thread uint* flags, uint x) {
    uint e = (x >> 23) & 0xffu;
    uint f = x & 0x7fffffu;
    bool neg = (x >> 31) != 0u;
    if (x != 0x80000000u) { *flags |= 8u; }
    if (e == 255u) { *flags |= (f != 0u) ? 1u : (neg ? 4u : 2u); return; }
    uint sig = (e != 0u) ? (f | 0x800000u) : f;
    uint pos = (e != 0u) ? (e - 1u) : 0u;
    fsum_add_at(d, pos, sig, neg);
}

static fsum_inline void fsum_deposit_product(fsum_thread fsum_i64* d, fsum_thread uint* flags, uint x, uint y) {
    uint mx = x & 0x7fffffffu;
    uint my = y & 0x7fffffffu;
    bool neg = ((x ^ y) >> 31) != 0u;
    bool zero = (mx == 0u) || (my == 0u);
    bool inf = (mx == 0x7f800000u) || (my == 0x7f800000u);
    if (!(zero && neg)) { *flags |= 8u; }
    if (mx > 0x7f800000u || my > 0x7f800000u || (zero && inf)) { *flags |= 1u; return; }
    if (inf) { *flags |= neg ? 4u : 2u; return; }
    if (zero) { return; }
    uint ex = mx >> 23;
    uint ey = my >> 23;
    fsum_u64 sx = (fsum_u64)((ex != 0u) ? ((mx & 0x7fffffu) | 0x800000u) : mx);
    fsum_u64 sy = (fsum_u64)((ey != 0u) ? ((my & 0x7fffffu) | 0x800000u) : my);
    uint pos = ((ex != 0u) ? ex - 1u : 0u) + ((ey != 0u) ? ey - 1u : 0u);
    fsum_u64 product = sx * sy;
    fsum_add_at(d, pos, (uint)(product & 0xffffffull), neg);
    fsum_add_at(d, pos + 24u, (uint)(product >> 24), neg);
}
"#;

/// Metal Shading Language 3.1 source with both kernels.
pub fn msl() -> String {
    let deposit = DEPOSIT
        .replace("fsum_inline ", "")
        .replace("fsum_thread ", "thread ")
        .replace("fsum_i64", "long")
        .replace("fsum_u64", "ulong");
    format!("{MSL_HEAD}{deposit}{MSL_KERNELS}")
}

/// CUDA C++ source for NVRTC with both kernels (no FMA: explicit `_rn`).
pub fn cuda() -> String {
    let deposit = DEPOSIT
        .replace("static fsum_inline ", "__device__ __forceinline__ ")
        .replace("fsum_thread ", "")
        .replace("fsum_i64", "long long")
        .replace("fsum_u64", "unsigned long long");
    format!("{CUDA_HEAD}{deposit}{CUDA_KERNELS}")
}

const MSL_HEAD: &str = r#"#include <metal_stdlib>
#pragma METAL fp contract(off)
using namespace metal;
#define LANES 256u
"#;

const MSL_KERNELS: &str = r#"
// Apple GPUs flush binary32 subnormals. The guard records every operation
// whose IEEE result could differ: a subnormal input word, or a zero result
// from nonzero operands that do not cancel exactly. The host then discards
// the device words and computes the call on the CPU.
static bool fsum_subnormal(uint w) {
    return (w & 0x7f800000u) == 0u && (w & 0x7fffffu) != 0u;
}

static float fsum_add(float x, float y, thread uint* bad) {
    float r = x + y;
    if (r == 0.0f && x != -y) { *bad = 1u; }
    return r;
}

static float fsum_term(uint wx, uint wy, uint dot, thread uint* bad) {
    if (fsum_subnormal(wx)) { *bad = 1u; }
    float x = as_type<float>(wx);
    if (dot == 0u) { return x; }
    if (fsum_subnormal(wy)) { *bad = 1u; }
    float y = as_type<float>(wy);
    float r = x * y;
    if (r == 0.0f && x != 0.0f && y != 0.0f) { *bad = 1u; }
    return r;
}

// A full leaf loads as two aligned 16-byte vectors; a short tail leaf loads
// word by word. Either way the fold runs left to right from the first term.
static float fsum_leaf(device const uint* a, device const uint* b, uint dot, uint n, uint leaf, thread uint* bad) {
    uint start = leaf * 8u;
    if (start + 8u <= n) {
        device const uint4* a4 = (device const uint4*)(a + start);
        device const uint4* b4 = (device const uint4*)(b + start);
        uint4 x0 = a4[0], x1 = a4[1];
        uint4 y0 = (dot != 0u) ? b4[0] : x0, y1 = (dot != 0u) ? b4[1] : x1;
        float acc = fsum_term(x0.x, y0.x, dot, bad);
        acc = fsum_add(acc, fsum_term(x0.y, y0.y, dot, bad), bad);
        acc = fsum_add(acc, fsum_term(x0.z, y0.z, dot, bad), bad);
        acc = fsum_add(acc, fsum_term(x0.w, y0.w, dot, bad), bad);
        acc = fsum_add(acc, fsum_term(x1.x, y1.x, dot, bad), bad);
        acc = fsum_add(acc, fsum_term(x1.y, y1.y, dot, bad), bad);
        acc = fsum_add(acc, fsum_term(x1.z, y1.z, dot, bad), bad);
        return fsum_add(acc, fsum_term(x1.w, y1.w, dot, bad), bad);
    }
    float acc = fsum_term(a[start], b[start], dot, bad);
    for (uint i = start + 1u; i < n; ++i) { acc = fsum_add(acc, fsum_term(a[i], b[i], dot, bad), bad); }
    return acc;
}

kernel void invar_fsum_tree_v1(device const uint* a [[buffer(0)]],
                               device const uint* b [[buffer(1)]],
                               device uint* out [[buffer(2)]],
                               constant uint* p [[buffer(3)]],
                               device atomic_uint* guard [[buffer(4)]],
                               uint group [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint simd [[simdgroup_index_in_threadgroup]]) {
    uint n = p[0], dot = p[1], t = p[2];
    uint m = (n + 7u) / 8u;
    uint steps = 1u << t;
    // Simdgroup `g` owns the aligned run of `32 << t` leaves from `base`. Each
    // step, lane `l` folds leaf `base + 32k + l` (32 adjacent leaves load as
    // one contiguous kilobyte), then the lanes reduce them as an aligned tree.
    uint base = (group * 8u + simd) * (32u << t);
    uint bad = 0u;
    float nodes[8];
    for (uint k = 0u; k < 8u; ++k) {
        if (k < steps) {
            uint leaf = base + k * 32u + lane;
            float v = leaf < m ? fsum_leaf(a, b, dot, n, leaf, &bad) : 0.0f;
            for (uint s = 1u; s < 32u; s <<= 1) {
                float other = simd_shuffle_down(v, (ushort)s);
                if ((lane & (2u * s - 1u)) == 0u && leaf + s < m) { v = fsum_add(v, other, &bad); }
            }
            nodes[k] = v;
        }
    }
    for (uint s = 1u; s < 8u; s <<= 1) {
        for (uint i = 0u; i < 8u; i += 2u * s) {
            if (i + s < steps && base + (i + s) * 32u < m) { nodes[i] = fsum_add(nodes[i], nodes[i + s], &bad); }
        }
    }
    threadgroup float partial[8];
    if (lane == 0u) { partial[simd] = nodes[0]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u) {
        float v = partial[lane & 7u];
        uint first = group * 8u * (32u << t);
        for (uint s = 1u; s < 8u; s <<= 1) {
            float other = simd_shuffle_down(v, (ushort)s);
            if (lane < 8u && (lane & (2u * s - 1u)) == 0u && first + (lane + s) * (32u << t) < m) { v = fsum_add(v, other, &bad); }
        }
        if (lane == 0u) { out[group] = as_type<uint>(v); }
    }
    if (bad != 0u) { atomic_fetch_or_explicit(guard, 1u, memory_order_relaxed); }
}

kernel void invar_fsum_exact_v1(device const uint* a [[buffer(0)]],
                                device const uint* b [[buffer(1)]],
                                device long* out [[buffer(2)]],
                                constant uint* p [[buffer(3)]],
                                device atomic_uint* guard [[buffer(4)]],
                                uint tid [[thread_position_in_threadgroup]],
                                uint group [[threadgroup_position_in_grid]],
                                uint gid [[thread_position_in_grid]],
                                uint total [[threads_per_grid]]) {
    uint n = p[0], dot = p[1], digits = p[3];
    long d[20];
    for (uint k = 0u; k < 20u; ++k) { d[k] = 0; }
    uint flags = 0u;
    for (uint i = gid; i < n; i += total) {
        if (dot != 0u) { fsum_deposit_product(d, &flags, a[i], b[i]); }
        else { fsum_deposit(d, &flags, a[i]); }
    }
    threadgroup long buf[LANES];
    threadgroup atomic_uint group_flags;
    if (tid == 0u) { atomic_store_explicit(&group_flags, 0u, memory_order_relaxed); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (flags != 0u) { atomic_fetch_or_explicit(&group_flags, flags, memory_order_relaxed); }
    for (uint k = 0u; k < digits; ++k) {
        buf[tid] = d[k];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = LANES / 2u; s > 0u; s >>= 1) {
            if (tid < s) { buf[tid] += buf[tid + s]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) { out[group * 20u + k] = buf[0]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        for (uint k = digits; k < 19u; ++k) { out[group * 20u + k] = 0; }
        out[group * 20u + 19u] = (long)atomic_load_explicit(&group_flags, memory_order_relaxed);
    }
}
"#;

const CUDA_HEAD: &str = r#"#define LANES 256u
typedef unsigned int uint;
"#;

const CUDA_KERNELS: &str = r#"
__device__ __forceinline__ float fsum_term(uint wx, uint wy, uint dot) {
    float x = __uint_as_float(wx);
    return dot != 0u ? __fmul_rn(x, __uint_as_float(wy)) : x;
}

// A full leaf loads as two aligned 16-byte vectors; a short tail leaf loads
// word by word. Either way the fold runs left to right from the first term.
__device__ __forceinline__ float fsum_leaf(const uint* a, const uint* b, uint dot, uint n, uint leaf) {
    uint start = leaf * 8u;
    if (start + 8u <= n) {
        const uint4* a4 = reinterpret_cast<const uint4*>(a + start);
        const uint4* b4 = reinterpret_cast<const uint4*>(b + start);
        uint4 x0 = a4[0], x1 = a4[1];
        uint4 y0 = dot != 0u ? b4[0] : x0, y1 = dot != 0u ? b4[1] : x1;
        float acc = fsum_term(x0.x, y0.x, dot);
        acc = __fadd_rn(acc, fsum_term(x0.y, y0.y, dot));
        acc = __fadd_rn(acc, fsum_term(x0.z, y0.z, dot));
        acc = __fadd_rn(acc, fsum_term(x0.w, y0.w, dot));
        acc = __fadd_rn(acc, fsum_term(x1.x, y1.x, dot));
        acc = __fadd_rn(acc, fsum_term(x1.y, y1.y, dot));
        acc = __fadd_rn(acc, fsum_term(x1.z, y1.z, dot));
        return __fadd_rn(acc, fsum_term(x1.w, y1.w, dot));
    }
    float acc = fsum_term(a[start], b[start], dot);
    for (uint i = start + 1u; i < n; ++i) { acc = __fadd_rn(acc, fsum_term(a[i], b[i], dot)); }
    return acc;
}

extern "C" __global__ void __launch_bounds__(256) invar_fsum_tree_v1(
        const uint* __restrict__ a, const uint* __restrict__ b, uint* __restrict__ out,
        uint n, uint dot, uint t) {
    uint lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    uint m = (n + 7u) / 8u;
    uint steps = 1u << t;
    // Warp `g` owns the aligned run of `32 << t` leaves from `base` (see MSL).
    uint base = (blockIdx.x * 8u + warp) * (32u << t);
    float nodes[8];
    #pragma unroll
    for (uint k = 0u; k < 8u; ++k) {
        if (k < steps) {
            uint leaf = base + k * 32u + lane;
            float v = leaf < m ? fsum_leaf(a, b, dot, n, leaf) : 0.0f;
            #pragma unroll
            for (uint s = 1u; s < 32u; s <<= 1) {
                float other = __shfl_down_sync(0xffffffffu, v, s);
                if ((lane & (2u * s - 1u)) == 0u && leaf + s < m) { v = __fadd_rn(v, other); }
            }
            nodes[k] = v;
        }
    }
    #pragma unroll
    for (uint s = 1u; s < 8u; s <<= 1) {
        #pragma unroll
        for (uint i = 0u; i < 8u; i += 2u * s) {
            if (i + s < steps && base + (i + s) * 32u < m) { nodes[i] = __fadd_rn(nodes[i], nodes[i + s]); }
        }
    }
    __shared__ float partial[8];
    if (lane == 0u) { partial[warp] = nodes[0]; }
    __syncthreads();
    if (warp == 0u) {
        float v = partial[lane & 7u];
        uint first = blockIdx.x * 8u * (32u << t);
        for (uint s = 1u; s < 8u; s <<= 1) {
            float other = __shfl_down_sync(0xffffffffu, v, s);
            if (lane < 8u && (lane & (2u * s - 1u)) == 0u && first + (lane + s) * (32u << t) < m) { v = __fadd_rn(v, other); }
        }
        if (lane == 0u) { out[blockIdx.x] = __float_as_uint(v); }
    }
}

extern "C" __global__ void __launch_bounds__(256) invar_fsum_exact_v1(
        const uint* __restrict__ a, const uint* __restrict__ b, long long* __restrict__ out,
        uint n, uint dot, uint digits) {
    uint tid = threadIdx.x;
    long long d[20];
    for (uint k = 0u; k < 20u; ++k) { d[k] = 0; }
    uint flags = 0u;
    uint total = gridDim.x * LANES;
    for (uint i = blockIdx.x * LANES + tid; i < n; i += total) {
        if (dot != 0u) { fsum_deposit_product(d, &flags, a[i], b[i]); }
        else { fsum_deposit(d, &flags, a[i]); }
    }
    __shared__ long long buf[LANES];
    __shared__ uint group_flags;
    if (tid == 0u) { group_flags = 0u; }
    __syncthreads();
    if (flags != 0u) { atomicOr(&group_flags, flags); }
    for (uint k = 0u; k < digits; ++k) {
        buf[tid] = d[k];
        __syncthreads();
        for (uint s = LANES / 2u; s > 0u; s >>= 1) {
            if (tid < s) { buf[tid] += buf[tid + s]; }
            __syncthreads();
        }
        if (tid == 0u) { out[blockIdx.x * 20u + k] = buf[0]; }
        __syncthreads();
    }
    if (tid == 0u) {
        for (uint k = digits; k < 19u; ++k) { out[blockIdx.x * 20u + k] = 0; }
        out[blockIdx.x * 20u + 19u] = (long long)group_flags;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_covers_every_leaf_and_bounds_exact_groups() {
        assert_eq!(Launch::new(Op::SumTree, 0, 2), Err(Error::Empty));
        assert_eq!(Launch::new(Op::SumTree, 1, 4), Err(Error::Limit));
        let launch = Launch::new(Op::SumTree, 8 * 1024 + 1, 2).unwrap();
        assert_eq!((launch.groups, launch.node_level()), (2, 10));
        assert_eq!(launch.output_words(), 2);
        let exact = Launch::new(Op::DotExact, 1 << 30, 0).unwrap();
        assert_eq!(exact.groups, MAX_EXACT_GROUPS);
        assert_eq!(exact.output_words(), MAX_EXACT_GROUPS as usize * ROW);
        assert_eq!(exact.params, [1 << 30, 1, 0, 19]);
    }

    #[test]
    fn sources_name_both_kernels_and_avoid_contraction() {
        for source in [msl(), cuda()] {
            assert!(source.contains("invar_fsum_tree_v1"));
            assert!(source.contains("invar_fsum_exact_v1"));
            assert!(!source.contains("fma"));
        }
        assert!(cuda().contains("__fmul_rn"));
        assert!(!msl().contains("fsum_thread") && !cuda().contains("fsum_inline"));
    }
}
