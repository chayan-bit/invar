//! Matrix-atom schedules of the generic contraction emitter (#283): the
//! `tc-f16-f32-v1` contraction through a target's `mma.sync` m16n8k16 or
//! `simdgroup_matrix` 8x8 atom. The atom, its instruction text, the tile, the
//! tiles per group and the warp width come from the verified moves and the
//! target file; each atom's fragment layout is a lowering of its own. `tc.rs`
//! emits through the sequences `tc_cuda_v1` and `tc_metal_v1`, whose text is
//! the former fixed templates' byte for byte.
//! Semantics are those of `tc.rs` (binary16 inputs rounded on the host,
//! subnormal products corrected on CUDA cores); see its module comment.

use super::{Error, describes, language, stage, tile};
use crate::floating::{self, Target as Lang, finish, header};
use crate::gemm::Geometry;
use crate::moves::{self, Axis, Move, Operand, Region, Schedule};
use crate::target::{Element, Target};
use crate::{Capabilities, tc};

/// `invar-tc-f16-f32-mma-m16n8k16-v1` (tc.rs): one warp per 16x8 tile on
/// `mma.sync` m16n8k16, four independent tiles per group, fragments loaded
/// from global memory straight into registers.
pub fn tc_cuda_v1() -> Schedule {
    let [m, n, k] = tc::CUDA_TILE;
    Schedule {
        name: tc::CUDA_SCHEDULE.to_owned(),
        moves: vec![
            tile(m, n, k),
            Move::UseAtom {
                atom: "mma-sync-m16n8k16-f16-f32".to_owned(),
            },
            Move::Split {
                axis: Axis::Tiles,
                factor: 4,
            },
            Move::Privatize { buffer: Operand::C },
        ],
        guards: Vec::new(),
    }
}

/// `invar-tc-f16-f32-simdgroup8x8-v1` (tc.rs): one simdgroup per 8x8 tile,
/// with A, B and the stored C staged in threadgroup memory.
pub fn tc_metal_v1() -> Schedule {
    let [m, n, k] = tc::METAL_TILE;
    Schedule {
        name: tc::METAL_SCHEDULE.to_owned(),
        moves: vec![
            tile(m, n, k),
            Move::UseAtom {
                atom: "simdgroup-float8x8".to_owned(),
            },
            stage(Operand::A),
            stage(Operand::B),
            stage(Operand::C),
            Move::Privatize { buffer: Operand::C },
        ],
        guards: Vec::new(),
    }
}

/// Verify a matrix-atom schedule for a `tc-f16-f32-v1` plan and accept only
/// the atom lowerings this emitter has: `mma.sync` m16n8k16 (f16 inputs, f32
/// accumulator, one warp per atom tile, fragments in registers) and the
/// `simdgroup_matrix` 8x8x8 multiply-accumulate (one simdgroup per group, A,
/// B and C staged).
pub fn verify(
    plan: &floating::Verified,
    schedule: &Schedule,
    target: &Target,
) -> Result<moves::Verified, Error> {
    if !plan.is_tc() {
        return Err(Error::Plan(floating::Error::Numeric));
    }
    describes(plan, target)?;
    let region = Region {
        contract: plan.ir().numeric_policy.clone(),
        element_bytes: 4,
    };
    let v = moves::verify(schedule, target, &region).map_err(Error::Schedule)?;
    let Some(atom) = &v.atom else {
        return Err(Error::Unsupported("scalar-schedule-for-a-matrix-contract"));
    };
    let shape = [atom.m, atom.n, atom.k];
    let staged: Vec<Operand> = v.staged.iter().map(|(o, _)| *o).collect();
    let lowered = match plan.capabilities() {
        Capabilities::Cuda { .. } => {
            shape == [16, 8, 16]
                && (atom.a_type, atom.b_type, atom.acc_type)
                    == (Element::F16, Element::F16, Element::F32)
                && atom
                    .instruction
                    .starts_with("mma.sync.aligned.m16n8k16.row.col.")
                && staged.is_empty()
        }
        Capabilities::Metal { .. } => {
            shape == [8, 8, 8]
                && atom
                    .instruction
                    .starts_with("simdgroup_multiply_accumulate")
                && staged == [Operand::A, Operand::B, Operand::C]
                && v.tiles_per_group == 1
        }
    };
    // One atom per warp tile, one thread set per atom, scalar loads.
    let one_atom = v.tile == shape && atom.threads * v.tiles_per_group == v.threads;
    if !lowered || !one_atom || !v.privatized || v.stages != 1 || v.vector != 1 {
        return Err(Error::Unsupported("atom-lowering"));
    }
    Ok(v)
}

/// Launch geometry: `tiles_per_group` atom tiles per group.
pub fn geometry(plan: &floating::Verified, v: &moves::Verified) -> Geometry {
    let [rows, _, cols] = plan.ir().dimensions;
    let tiles = u64::from(rows.div_ceil(v.tile[0])) * u64::from(cols.div_ceil(v.tile[1]));
    let groups = tiles.div_ceil(u64::from(v.tiles_per_group));
    Geometry {
        tiles,
        groups: u32::try_from(groups.min(u64::from(tc::MAX_GROUPS))).unwrap_or(tc::MAX_GROUPS),
        threads: v.threads,
    }
}

fn substitute(
    plan: &floating::Verified,
    v: &moves::Verified,
    target: &Target,
    text: String,
) -> String {
    let atom = v.atom.as_ref().map_or("", |a| a.instruction.as_str());
    let warp = target.gpu().map_or(32, |g| g.subgroup_width);
    let [tm, tn, tk] = v.tile;
    let schedule = [
        ("$MMA$", atom.to_owned()),
        ("$T$", v.threads.to_string()),
        ("$W$", warp.to_string()),
        ("$PG$", v.tiles_per_group.to_string()),
        ("$AE$", (tm * tk).to_string()),
        ("$CE$", (tm * tn).to_string()),
        ("$TM$", tm.to_string()),
        ("$TN$", tn.to_string()),
        ("$TK$", tk.to_string()),
    ];
    let text = schedule
        .into_iter()
        .fold(text, |text, (token, value)| text.replace(token, &value));
    let ir = plan.ir();
    let [rows, inner, cols] = ir.dimensions;
    let [a, b] = &ir.inputs;
    let c = &ir.output;
    let data = [
        ("$TILES_N$", u64::from(cols.div_ceil(tn))),
        ("$TILES$", geometry(plan, v).tiles),
        ("$ROWS$", u64::from(rows)),
        ("$INNER$", u64::from(inner)),
        ("$COLS$", u64::from(cols)),
        ("$A_OFFSET$", a.offset),
        ("$A_STRIDE$", a.strides[0]),
        ("$A_ELEMENTS$", a.elements),
        ("$B_OFFSET$", b.offset),
        ("$B_STRIDE$", b.strides[0]),
        ("$B_ELEMENTS$", b.elements),
        ("$C_OFFSET$", c.offset),
        ("$C_STRIDE$", c.strides[0]),
        ("$C_ELEMENTS$", c.elements),
    ];
    data.iter().fold(text, |text, (token, value)| {
        text.replace(token, &value.to_string())
    })
}

fn schedule_line(plan: &floating::Verified, v: &moves::Verified) -> String {
    let g = geometry(plan, v);
    let [m, n, k] = v.tile;
    format!(
        "// schedule={} tile={m}x{n}x{k} threads={} tiles={} groups={}\n",
        v.schedule, g.threads, g.tiles, g.groups
    )
}

fn empty(plan: &floating::Verified) -> bool {
    let [rows, _, cols] = plan.ir().dimensions;
    rows == 0 || cols == 0
}

/// CUDA C++ with inline-PTX `mma.sync` for a verified `tc-f16-f32-v1`
/// schedule. A zero inner extent omits the k loop and its helpers.
pub fn emit_cuda(
    plan: &floating::Verified,
    schedule: &Schedule,
    target: &Target,
) -> Result<String, Error> {
    let v = verify(plan, schedule, target)?;
    language(plan, Lang::Cuda)?;
    let mut text = header(plan, Lang::Cuda)?;
    text.push_str(&schedule_line(plan, &v));
    if empty(plan) {
        text.push_str("extern \"C\" __global__ void invar_f32_contract_v1(const float* A, const float* B, float* C) { (void)A; (void)B; (void)C; }\n");
        return Ok(finish(text)?);
    }
    let body = if plan.ir().dimensions[1] == 0 {
        format!("{CUDA_HEAD}{CUDA_TAIL}")
    } else {
        format!("{CUDA_HELPERS}{CUDA_HEAD}{CUDA_K_LOOP}{CUDA_TAIL}")
    };
    text.push_str(&substitute(plan, &v, target, body));
    Ok(finish(text)?)
}

/// MSL 3.1 with `simdgroup_matrix` multiply-accumulate for a verified
/// `tc-f16-f32-v1` schedule: buffers 0/1/2 = A/B/C words, 3 = the zeroed
/// dispatch flag, raised only by a launch or bounds violation.
pub fn emit_msl(
    plan: &floating::Verified,
    schedule: &Schedule,
    target: &Target,
) -> Result<String, Error> {
    let v = verify(plan, schedule, target)?;
    language(plan, Lang::Metal)?;
    let mut text = header(plan, Lang::Metal)?;
    text.push_str(&schedule_line(plan, &v));
    text.push_str(
        "#include <metal_stdlib>\n#pragma METAL fp contract(off)\nusing namespace metal;\n",
    );
    if empty(plan) {
        text.push_str("kernel void invar_f32_contract_v1(device const uint* A [[buffer(0)]], device const uint* B [[buffer(1)]], device uint* C [[buffer(2)]], device atomic_uint* F [[buffer(3)]]) { (void)A; (void)B; (void)C; atomic_fetch_and_explicit(F, 1u, memory_order_relaxed); }\n");
        return Ok(finish(text)?);
    }
    let body = if plan.ir().dimensions[1] == 0 {
        format!("{MSL_HEAD}{MSL_TAIL}")
    } else {
        format!("{MSL_HEAD}{MSL_K_LOOP}{MSL_TAIL}")
    };
    text.push_str(&substitute(plan, &v, target, body));
    Ok(finish(text)?)
}

const CUDA_HELPERS: &str = r#"__device__ static inline float invar_a(const float* A, unsigned long long i, unsigned long long k) {
  if (i >= $ROWS$ull || k >= $INNER$ull) return 0.0f;
  const unsigned long long ai = $A_OFFSET$ull + i * $A_STRIDE$ull + k;
  if (ai >= $A_ELEMENTS$ull) __trap();
  return A[ai];
}
__device__ static inline float invar_b(const float* B, unsigned long long k, unsigned long long j) {
  if (k >= $INNER$ull || j >= $COLS$ull) return 0.0f;
  const unsigned long long bi = $B_OFFSET$ull + k * $B_STRIDE$ull + j;
  if (bi >= $B_ELEMENTS$ull) __trap();
  return B[bi];
}
// A nonzero binary16 subnormal (below 2^-14). The mma sees such an operand as
// zero; the correction pass adds its products exactly (second D5 amendment).
__device__ static inline bool invar_sub(float x) {
  const unsigned m = __float_as_uint(x) & 0x7fffffffu;
  return m != 0u && m < 0x38800000u;
}
__device__ static inline unsigned invar_pack(float lo, float hi) {
  unsigned d;
  const float l = invar_sub(lo) ? 0.0f : lo, h = invar_sub(hi) ? 0.0f : hi;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(h), "f"(l));
  return d;
}
// One correction term: a product with a subnormal operand is exact in f32
// (at most 22 significant bits, at least 2^-48), added in increasing k.
__device__ static inline float invar_fix(float s, float x, float y) {
  return (invar_sub(x) || invar_sub(y)) ? __fadd_rn(s, __fmul_rn(x, y)) : s;
}
"#;

const CUDA_HEAD: &str = r#"__device__ static inline void invar_c(float* C, unsigned long long i, unsigned long long j, float v) {
  if (i >= $ROWS$ull || j >= $COLS$ull) return;
  const unsigned long long ci = $C_OFFSET$ull + i * $C_STRIDE$ull + j;
  if (ci >= $C_ELEMENTS$ull) __trap();
  C[ci] = v;
}
extern "C" __global__ void __launch_bounds__($T$) invar_f32_contract_v1(const float* A, const float* B, float* C) {
  if (blockDim.x != $T$u || blockDim.y != 1u || blockDim.z != 1u || gridDim.y != 1u || gridDim.z != 1u) __trap();
  const unsigned lane = threadIdx.x % $W$u, g = lane / 4u, q = lane % 4u;
  for (unsigned long long tile = blockIdx.x * $PG$ull + threadIdx.x / $W$u; tile < $TILES$ull; tile += gridDim.x * $PG$ull) {
    const unsigned long long row0 = tile / $TILES_N$ull * $TM$ull, col0 = tile % $TILES_N$ull * $TN$ull;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
"#;

// Fragment layouts of mma.sync m16n8k16 (PTX ISA): A row-major 16x16 in four
// f16x2 registers, B column-major 16x8 in two, C/D 16x8 f32 in four. Every
// lane of a warp shares `tile`, so the warp converges at each mma.
const CUDA_K_LOOP: &str = r#"    for (unsigned long long k0 = 0ull; k0 < $INNER$ull; k0 += $TK$ull) {
      const unsigned long long r0 = row0 + g, r1 = row0 + g + 8u, ka = k0 + 2u * q, n = col0 + g;
      const float x0 = invar_a(A, r0, ka), x1 = invar_a(A, r0, ka + 1u);
      const float x2 = invar_a(A, r1, ka), x3 = invar_a(A, r1, ka + 1u);
      const float x4 = invar_a(A, r0, ka + 8u), x5 = invar_a(A, r0, ka + 9u);
      const float x6 = invar_a(A, r1, ka + 8u), x7 = invar_a(A, r1, ka + 9u);
      const float y0 = invar_b(B, ka, n), y1 = invar_b(B, ka + 1u, n);
      const float y2 = invar_b(B, ka + 8u, n), y3 = invar_b(B, ka + 9u, n);
      const unsigned a0 = invar_pack(x0, x1), a1 = invar_pack(x2, x3);
      const unsigned a2 = invar_pack(x4, x5), a3 = invar_pack(x6, x7);
      const unsigned b0 = invar_pack(y0, y1), b1 = invar_pack(y2, y3);
      asm volatile("$MMA$ {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                   : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      // The warp's fragments cover the whole k-tile, so one vote decides
      // whether any of its products needs the CUDA-core correction.
      const bool sub = invar_sub(x0) || invar_sub(x1) || invar_sub(x2) || invar_sub(x3) ||
                       invar_sub(x4) || invar_sub(x5) || invar_sub(x6) || invar_sub(x7) ||
                       invar_sub(y0) || invar_sub(y1) || invar_sub(y2) || invar_sub(y3);
      if (__any_sync(0xffffffffu, sub)) {
        const unsigned long long j = col0 + 2u * q;
        for (unsigned long long k = k0; k < k0 + $TK$ull; ++k) {
          const float u0 = invar_a(A, r0, k), u1 = invar_a(A, r1, k);
          const float v0 = invar_b(B, k, j), v1 = invar_b(B, k, j + 1u);
          s0 = invar_fix(s0, u0, v0);
          s1 = invar_fix(s1, u0, v1);
          s2 = invar_fix(s2, u1, v0);
          s3 = invar_fix(s3, u1, v1);
        }
      }
    }
"#;

const CUDA_TAIL: &str = r#"    const unsigned long long r0 = row0 + g, r1 = row0 + g + 8u, j = col0 + 2u * q;
    invar_c(C, r0, j, __fadd_rn(c0, s0));
    invar_c(C, r0, j + 1u, __fadd_rn(c1, s1));
    invar_c(C, r1, j, __fadd_rn(c2, s2));
    invar_c(C, r1, j + 1u, __fadd_rn(c3, s3));
  }
}
"#;

// Words carry binary32 bits of binary16 values (rounded on the host), which
// are normal binary32 numbers, as are their products and nonzero partial sums,
// so a flushing device cannot change them. A bounds failure raises the flag.
const MSL_HEAD: &str = r#"kernel void invar_f32_contract_v1(device const uint* A [[buffer(0)]], device const uint* B [[buffer(1)]], device uint* C [[buffer(2)]], device atomic_uint* F [[buffer(3)]], uint group [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]], uint t [[thread_position_in_threadgroup]], uint lanes [[threads_per_threadgroup]]) {
  threadgroup float As[$AE$];
  threadgroup float Bs[$AE$];
  threadgroup float Cs[$CE$];
  // Claim the UNWRITTEN status word first (floating::guard encoding).
  atomic_fetch_and_explicit(F, 1u, memory_order_relaxed);
  if (lanes != $T$u) { atomic_fetch_or_explicit(F, 1u, memory_order_relaxed); return; }
  uint flag = 0u;
  for (ulong tile = group; tile < $TILES$ul; tile += groups) {
    const ulong row0 = tile / $TILES_N$ul * $TM$ul, col0 = tile % $TILES_N$ul * $TN$ul;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
"#;

const MSL_K_LOOP: &str = r#"    for (ulong k0 = 0ul; k0 < $INNER$ul; k0 += $TK$ul) {
      for (uint e = t; e < $AE$u; e += $T$u) {
        const ulong i = row0 + e / $TK$u, k = k0 + e % $TK$u;
        uint w = 0u;
        if (i < $ROWS$ul && k < $INNER$ul) {
          const ulong ai = $A_OFFSET$ul + i * $A_STRIDE$ul + k;
          if (ai < $A_ELEMENTS$ul) { w = A[ai]; } else { flag = 1u; }
        }
        As[e] = as_type<float>(w);
        const ulong kb = k0 + e / $TN$u, j = col0 + e % $TN$u;
        uint v = 0u;
        if (kb < $INNER$ul && j < $COLS$ul) {
          const ulong bi = $B_OFFSET$ul + kb * $B_STRIDE$ul + j;
          if (bi < $B_ELEMENTS$ul) { v = B[bi]; } else { flag = 1u; }
        }
        Bs[e] = as_type<float>(v);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      simdgroup_float8x8 a;
      simdgroup_float8x8 b;
      simdgroup_load(a, As, $TK$);
      simdgroup_load(b, Bs, $TN$);
      simdgroup_multiply_accumulate(acc, a, b, acc);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"#;

const MSL_TAIL: &str = r#"    simdgroup_store(acc, Cs, $TN$);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = t; e < $CE$u; e += $T$u) {
      const ulong i = row0 + e / $TN$u, j = col0 + e % $TN$u;
      if (i < $ROWS$ul && j < $COLS$ul) {
        const ulong ci = $C_OFFSET$ul + i * $C_STRIDE$ul + j;
        if (ci < $C_ELEMENTS$ul) { C[ci] = as_type<uint>(Cs[e]); } else { flag = 1u; }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (flag != 0u) atomic_fetch_or_explicit(F, 1u, memory_order_relaxed);
}
"#;
