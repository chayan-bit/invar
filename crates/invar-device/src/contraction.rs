//! ENG-06 generic contraction emitter (#283): CUDA C++ and MSL for a strict
//! binary32 contraction from a move sequence verified against a target
//! description, instead of one fixed template per schedule. The kernel shape
//! follows the verified moves: the group tile, the cells per thread, each
//! staged operand's layout and pad. `gemm.rs` emits through the
//! `gemm_tiled_v1` sequence, whose text is the former fixed template's byte
//! for byte, so the emitted kernels, their identity hashes and their cache
//! keys are unchanged.

use crate::floating::{self, Target as Lang, finish, guard_helpers, header, strict_only};
use crate::gemm::Geometry;
use crate::moves::{self, Axis, Layout, Move, Operand, Region, Schedule};
use crate::target::{Backend, Level, Registry, Target};
use crate::{Capabilities, gemm};

pub mod atoms;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Plan(floating::Error),
    Schedule(moves::Refusal),
    /// The target does not describe the plan's device.
    Target,
    /// A verified move this emitter does not lower yet.
    Unsupported(&'static str),
}

impl From<floating::Error> for Error {
    fn from(e: floating::Error) -> Self {
        Self::Plan(e)
    }
}

fn tile(m: u32, n: u32, k: u32) -> Move {
    Move::Tile {
        axes: vec![Axis::M, Axis::N, Axis::K],
        sizes: vec![m, n, k],
    }
}

fn stage(buffer: Operand) -> Move {
    Move::Stage {
        buffer,
        level: Level::Shared,
    }
}

/// `invar-f32-gemm-tiled-v1` (gemm.rs): a 64x64 group tile over 16-deep K
/// slices, 4x4 strided cells per thread (256 threads), A staged K-major with
/// one pad element per row, B staged row-major, accumulators in registers.
pub fn gemm_tiled_v1() -> Schedule {
    let split = |axis| Move::Split { axis, factor: 4 };
    Schedule {
        name: gemm::SCHEDULE.to_owned(),
        moves: vec![
            tile(gemm::TILE, gemm::TILE, gemm::TILE_K),
            split(Axis::M),
            split(Axis::N),
            stage(Operand::A),
            Move::Permute {
                buffer: Operand::A,
                layout: Layout {
                    k_major: true,
                    pad: 1,
                },
            },
            stage(Operand::B),
            Move::Privatize { buffer: Operand::C },
        ],
        guards: Vec::new(),
    }
}

/// The target must describe the plan's device exactly.
fn describes(plan: &floating::Verified, target: &Target) -> Result<(), Error> {
    let d = &target.device;
    let ok = match plan.capabilities() {
        Capabilities::Cuda { sm, .. } => d.backend == Backend::Cuda && d.sm == Some(sm),
        Capabilities::Metal { apple_family, .. } => {
            d.backend == Backend::Metal && d.apple_family == Some(apple_family)
        }
    };
    if ok { Ok(()) } else { Err(Error::Target) }
}

impl Error {
    /// The plan error a fixed schedule reports: a refusal of its own moves
    /// on a built-in target means the device is not supported.
    pub fn into_plan(self) -> floating::Error {
        match self {
            Self::Plan(e) => e,
            Self::Schedule(_) | Self::Target | Self::Unsupported(_) => {
                floating::Error::Capabilities
            }
        }
    }
}

/// The target `registry` holds for `caps`; a device no target description
/// selects refuses device execution with `TARGET_UNKNOWN`.
pub fn select_target(registry: &Registry, caps: Capabilities) -> Result<&Target, floating::Error> {
    registry
        .select(caps)
        .map_err(|_| floating::Error::TargetUnknown)
}

/// The built-in target file for the plan's live capabilities. Every
/// capability the device gate admits has one.
pub(crate) fn builtin_target(
    plan: &floating::Verified,
) -> Result<&'static Target, floating::Error> {
    let registry = Registry::builtin().map_err(|_| floating::Error::TargetUnknown)?;
    select_target(registry, plan.capabilities())
}

/// The plan's backend is the emitted language (`describes` ties the target to
/// the plan, so a Metal plan and target cannot reach the CUDA template).
fn language(plan: &floating::Verified, lang: Lang) -> Result<(), Error> {
    match (plan.capabilities(), lang) {
        (Capabilities::Cuda { .. }, Lang::Cuda) | (Capabilities::Metal { .. }, Lang::Metal) => {
            Ok(())
        }
        _ => Err(Error::Plan(floating::Error::Capabilities)),
    }
}

/// Verify `schedule` for the plan's device and accept only what this
/// emitter lowers: scalar cells, one tile per group, A and B staged in
/// shared memory, accumulators in registers, scalar loads, one stage.
pub fn verify(
    plan: &floating::Verified,
    schedule: &Schedule,
    target: &Target,
) -> Result<moves::Verified, Error> {
    strict_only(plan)?;
    describes(plan, target)?;
    let region = Region {
        contract: plan.ir().numeric_policy.clone(),
        element_bytes: 4,
    };
    let v = moves::verify(schedule, target, &region).map_err(Error::Schedule)?;
    let staged: Vec<Operand> = v.staged.iter().map(|(o, _)| *o).collect();
    if v.atom.is_some() {
        return Err(Error::Unsupported("matrix-atom"));
    }
    if v.tiles_per_group != 1 || v.stages != 1 || v.vector != 1 {
        return Err(Error::Unsupported(
            "tiles-per-group, pipeline or vector loads",
        ));
    }
    if staged != [Operand::A, Operand::B] || !v.privatized {
        return Err(Error::Unsupported(
            "staging other than A and B in shared memory",
        ));
    }
    Ok(v)
}

/// Launch geometry: one-dimensional groups of the verified thread count.
pub fn geometry(plan: &floating::Verified, v: &moves::Verified) -> Geometry {
    let [rows, _, cols] = plan.ir().dimensions;
    let tiles = u64::from(rows.div_ceil(v.tile[0])) * u64::from(cols.div_ceil(v.tile[1]));
    Geometry {
        tiles,
        groups: u32::try_from(tiles.min(u64::from(gemm::MAX_GROUPS))).unwrap_or(gemm::MAX_GROUPS),
        threads: v.threads,
    }
}

fn layout(v: &moves::Verified, operand: Operand) -> Layout {
    v.staged.iter().find(|(o, _)| *o == operand).map_or(
        Layout {
            k_major: false,
            pad: 0,
        },
        |(_, l)| *l,
    )
}

/// The schedule tokens of the templates, in substitution order.
fn schedule_tokens(v: &moves::Verified) -> Vec<(&'static str, String)> {
    let [tm, tn, tk] = v.tile;
    let [fm, fnn] = v.micro;
    let (rm, rn) = (tm / fm, tn / fnn);
    let (a, b) = (layout(v, Operand::A), layout(v, Operand::B));
    let (a_decl, a_store, a_read) = if a.k_major {
        (
            format!("[{tk}][{}]", tm + a.pad),
            format!("[e % {tk}u][e / {tk}u]"),
            format!("[kk][ty + {rm}u * r]"),
        )
    } else {
        (
            format!("[{tm}][{}]", tk + a.pad),
            format!("[e / {tk}u][e % {tk}u]"),
            format!("[ty + {rm}u * r][kk]"),
        )
    };
    let (b_decl, b_store, b_read) = if b.k_major {
        (
            format!("[{tn}][{}]", tk + b.pad),
            format!("[e % {tn}u][e / {tn}u]"),
            format!("[tx + {rn}u * c][kk]"),
        )
    } else {
        (
            format!("[{tk}][{}]", tn + b.pad),
            format!("[e / {tn}u][e % {tn}u]"),
            format!("[kk][tx + {rn}u * c]"),
        )
    };
    vec![
        ("$AS$", a_decl),
        ("$BS$", b_decl),
        ("$A_ST$", a_store),
        ("$B_ST$", b_store),
        ("$A_RD$", a_read),
        ("$B_RD$", b_read),
        ("$AE$", (tm * tk).to_string()),
        ("$BE$", (tk * tn).to_string()),
        ("$T$", v.threads.to_string()),
        ("$TM$", tm.to_string()),
        ("$TN$", tn.to_string()),
        ("$TK$", tk.to_string()),
        ("$FM$", fm.to_string()),
        ("$FN$", fnn.to_string()),
        ("$RM$", rm.to_string()),
        ("$RN$", rn.to_string()),
    ]
}

fn body(plan: &floating::Verified, v: &moves::Verified, parts: [&str; 5]) -> String {
    let [signature, tiles, head, k_loop, tail] = parts;
    let ir = plan.ir();
    let [rows, inner, cols] = ir.dimensions;
    let template = if inner == 0 {
        format!("{signature}{head}{tail}")
    } else {
        format!("{signature}{tiles}{head}{k_loop}{tail}")
    };
    let text = schedule_tokens(v)
        .into_iter()
        .fold(template, |text, (token, value)| text.replace(token, &value));
    let [a, b] = &ir.inputs;
    let c = &ir.output;
    let data = [
        ("$TILES_N$", u64::from(cols.div_ceil(v.tile[1]))),
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
    let ([tm, tn, tk], [fm, fnn]) = (v.tile, v.micro);
    format!(
        "// schedule={} tile={tm}x{tn}x{tk} threads={} micro={fm}x{fnn} tiles={} groups={}\n",
        v.schedule, g.threads, g.tiles, g.groups
    )
}

fn empty(plan: &floating::Verified) -> bool {
    let [rows, _, cols] = plan.ir().dimensions;
    rows == 0 || cols == 0
}

/// CUDA C++ for a verified strict schedule, under the strict contraction
/// header; launch with `geometry` only.
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
    } else {
        text.push_str(&body(plan, &v, CUDA_BODY));
    }
    Ok(finish(text)?)
}

/// MSL 3.1 for a verified strict schedule, with the strict pragma and the
/// shared underflow guard: buffers 0/1/2 = A/B/C words, 3 = the zeroed
/// dispatch flag. A raised flag means the dispatch must run on the CPU.
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
    text.push_str(&guard_helpers(false));
    if empty(plan) {
        text.push_str("kernel void invar_f32_contract_v1(device const uint* A [[buffer(0)]], device const uint* B [[buffer(1)]], device uint* C [[buffer(2)]], device atomic_uint* F [[buffer(3)]]) { (void)A; (void)B; (void)C; atomic_fetch_and_explicit(F, 1u, memory_order_relaxed); }\n");
    } else {
        text.push_str(&body(plan, &v, MSL_BODY));
    }
    Ok(finish(text)?)
}

const CUDA_BODY: [&str; 5] = [
    r#"extern "C" __global__ void __launch_bounds__($T$) invar_f32_contract_v1(const float* A, const float* B, float* C) {
"#,
    r#"  __shared__ float As$AS$;
  __shared__ float Bs$BS$;
"#,
    r#"  if (blockDim.x != $T$u || blockDim.y != 1u || blockDim.z != 1u || gridDim.y != 1u || gridDim.z != 1u) __trap();
  const unsigned t = threadIdx.x, ty = t / $RN$u, tx = t % $RN$u;
  for (unsigned long long tile = blockIdx.x; tile < $TILES$ull; tile += gridDim.x) {
    const unsigned long long row0 = tile / $TILES_N$ull * $TM$ull, col0 = tile % $TILES_N$ull * $TN$ull;
    float acc[$FM$][$FN$];
#pragma unroll
    for (unsigned r = 0u; r < $FM$u; ++r)
#pragma unroll
      for (unsigned c = 0u; c < $FN$u; ++c) acc[r][c] = 0.0f;
"#,
    r#"    for (unsigned long long k0 = 0ull; k0 < $INNER$ull; k0 += $TK$ull) {
      for (unsigned e = t; e < $AE$u; e += $T$u) {
        const unsigned long long i = row0 + e / $TK$u, k = k0 + e % $TK$u;
        float v = 0.0f;
        if (i < $ROWS$ull && k < $INNER$ull) {
          const unsigned long long ai = $A_OFFSET$ull + i * $A_STRIDE$ull + k;
          if (ai >= $A_ELEMENTS$ull) __trap();
          v = A[ai];
        }
        As$A_ST$ = v;
      }
      for (unsigned e = t; e < $BE$u; e += $T$u) {
        const unsigned long long k = k0 + e / $TN$u, j = col0 + e % $TN$u;
        float v = 0.0f;
        if (k < $INNER$ull && j < $COLS$ull) {
          const unsigned long long bi = $B_OFFSET$ull + k * $B_STRIDE$ull + j;
          if (bi >= $B_ELEMENTS$ull) __trap();
          v = B[bi];
        }
        Bs$B_ST$ = v;
      }
      __syncthreads();
      const unsigned span = $INNER$ull - k0 < $TK$ull ? static_cast<unsigned>($INNER$ull - k0) : $TK$u;
      for (unsigned kk = 0u; kk < span; ++kk) {
        float a[$FM$], b[$FN$];
#pragma unroll
        for (unsigned r = 0u; r < $FM$u; ++r) a[r] = As$A_RD$;
#pragma unroll
        for (unsigned c = 0u; c < $FN$u; ++c) b[c] = Bs$B_RD$;
#pragma unroll
        for (unsigned r = 0u; r < $FM$u; ++r)
#pragma unroll
          for (unsigned c = 0u; c < $FN$u; ++c) acc[r][c] = __fadd_rn(acc[r][c], __fmul_rn(a[r], b[c]));
      }
      __syncthreads();
    }
"#,
    r#"    for (unsigned r = 0u; r < $FM$u; ++r)
      for (unsigned c = 0u; c < $FN$u; ++c) {
        const unsigned long long i = row0 + ty + $RM$u * r, j = col0 + tx + $RN$u * c;
        if (i < $ROWS$ull && j < $COLS$ull) {
          const unsigned long long ci = $C_OFFSET$ull + i * $C_STRIDE$ull + j;
          if (ci >= $C_ELEMENTS$ull) __trap();
          C[ci] = acc[r][c];
        }
      }
  }
}
"#,
];

// Words carry binary32 bits. A defensive bounds failure raises the dispatch
// flag instead of returning, so no thread leaves a barrier early; any raised
// flag discards the whole dispatch.
const MSL_BODY: [&str; 5] = [
    r#"kernel void invar_f32_contract_v1(device const uint* A [[buffer(0)]], device const uint* B [[buffer(1)]], device uint* C [[buffer(2)]], device atomic_uint* F [[buffer(3)]], uint group [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]], uint t [[thread_position_in_threadgroup]], uint lanes [[threads_per_threadgroup]]) {
"#,
    r#"  threadgroup uint As$AS$;
  threadgroup uint Bs$BS$;
"#,
    r#"  // Claim the UNWRITTEN status word first (floating::guard encoding).
  atomic_fetch_and_explicit(F, 1u, memory_order_relaxed);
  if (lanes != $T$u) { atomic_fetch_or_explicit(F, 1u, memory_order_relaxed); return; }
  const uint ty = t / $RN$u, tx = t % $RN$u;
  uint flag = 0u;
  for (ulong tile = group; tile < $TILES$ul; tile += groups) {
    const ulong row0 = tile / $TILES_N$ul * $TM$ul, col0 = tile % $TILES_N$ul * $TN$ul;
    float acc[$FM$][$FN$];
    for (uint r = 0u; r < $FM$u; ++r)
      for (uint c = 0u; c < $FN$u; ++c) acc[r][c] = 0.0f;
"#,
    r#"    for (ulong k0 = 0ul; k0 < $INNER$ul; k0 += $TK$ul) {
      for (uint e = t; e < $AE$u; e += $T$u) {
        const ulong i = row0 + e / $TK$u, k = k0 + e % $TK$u;
        uint w = 0u;
        if (i < $ROWS$ul && k < $INNER$ul) {
          const ulong ai = $A_OFFSET$ul + i * $A_STRIDE$ul + k;
          if (ai < $A_ELEMENTS$ul) { w = A[ai]; flag |= invar_operand(w); } else { flag = 1u; }
        }
        As$A_ST$ = w;
      }
      for (uint e = t; e < $BE$u; e += $T$u) {
        const ulong k = k0 + e / $TN$u, j = col0 + e % $TN$u;
        uint w = 0u;
        if (k < $INNER$ul && j < $COLS$ul) {
          const ulong bi = $B_OFFSET$ul + k * $B_STRIDE$ul + j;
          if (bi < $B_ELEMENTS$ul) { w = B[bi]; flag |= invar_operand(w); } else { flag = 1u; }
        }
        Bs$B_ST$ = w;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const uint span = $INNER$ul - k0 < $TK$ul ? uint($INNER$ul - k0) : $TK$u;
      for (uint kk = 0u; kk < span; ++kk) {
        uint aw[$FM$], bw[$FN$];
        for (uint r = 0u; r < $FM$u; ++r) aw[r] = As$A_RD$;
        for (uint c = 0u; c < $FN$u; ++c) bw[c] = Bs$B_RD$;
        for (uint r = 0u; r < $FM$u; ++r)
          for (uint c = 0u; c < $FN$u; ++c) {
            const float product = as_type<float>(aw[r]) * as_type<float>(bw[c]);
            const float next = acc[r][c] + product;
            flag |= invar_mul(aw[r], bw[c], as_type<uint>(product)) | invar_add(as_type<uint>(acc[r][c]), as_type<uint>(product), as_type<uint>(next));
            acc[r][c] = next;
          }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"#,
    r#"    for (uint r = 0u; r < $FM$u; ++r)
      for (uint c = 0u; c < $FN$u; ++c) {
        const ulong i = row0 + ty + $RM$u * r, j = col0 + tx + $RN$u * c;
        if (i < $ROWS$ul && j < $COLS$ul) {
          const ulong ci = $C_OFFSET$ul + i * $C_STRIDE$ul + j;
          if (ci < $C_ELEMENTS$ul) { C[ci] = as_type<uint>(acc[r][c]); } else { flag = 1u; }
        }
      }
  }
  if (flag != 0u) atomic_fetch_or_explicit(F, 1u, memory_order_relaxed);
}
"#,
];
