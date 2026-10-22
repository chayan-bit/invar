//! M4-07 bounded summary scans: finite-state tables (K <= 16) and wrapping-u32
//! matrices (d <= 4) as ordered scans on CUDA and Metal.
//!
//! Both families are exact: table composition is function composition and
//! matrix composition is multiplication in the ring of u32 modulo 2^32, so any
//! bracketing of the ordered composition equals the source's sequential loop
//! bit for bit. Order is preserved; composition is never commuted. Root owns
//! source recognition, runtime input validation and publication.
//!
//! Prefix scans run one single-pass kernel with a decoupled look-back on both
//! targets: each block or threadgroup takes a tile ticket, publishes its tile
//! aggregate, then the inclusive state after its tile once its predecessors
//! resolve, so every coefficient is read once. Metal has no inter-threadgroup
//! forward-progress guarantee, so its look-back polls each predecessor a
//! bounded number of times and otherwise reduces that predecessor's
//! coefficients itself (decoupled fallback); no threadgroup ever waits for
//! another to be scheduled. Metal final-state scans, which need no prefixes,
//! reduce tiles and compose the aggregates in a second dispatch instead.

use super::{Capabilities, Error, Provenance};

pub const SCHEDULE: &str = "invar-bounded-summary-scan-v3";
/// Threads per block or threadgroup, as 8 SIMD groups of 32.
pub const LANES: u32 = 256;
/// Coefficient words each thread keeps in registers per round, by target and
/// family (measured on the RTX 5070 Ti and Apple M4; see the M4-07 evidence).
/// Longer runs amortize the block scan, which matters most for O(d^3) matrix
/// compositions; shared-memory staging of a round lost on both targets. Metal
/// table prefixes take longer runs than final states, which amortize their
/// look-back over more steps; final states keep short runs and more rounds.
pub const CUDA_TABLE_ITEM_WORDS: u32 = 64;
pub const METAL_TABLE_ITEM_WORDS: u32 = 128;
pub const METAL_TABLE_FINAL_ITEM_WORDS: u32 = 32;
pub const MATRIX_ITEM_WORDS: u32 = 128;
/// CUDA look-back slot per tile: flag, inclusive state at 1..5, sum at 8..24.
pub const STATUS_WORDS: u32 = 24;
/// Rounds of `LANES * items` steps per Metal final-state reduce threadgroup.
pub const METAL_ROUNDS: u32 = 4;
/// Metal look-back slot per tile: a 16-word aggregate and a 4-word state, each
/// u32 split into two atomic words that carry their own published bit.
pub const METAL_SLOT_WORDS: u32 = 40;
/// Polls of one unpublished predecessor before a Metal threadgroup reduces
/// that tile itself (measured on the Apple M4; see the M4-07 evidence).
pub const METAL_SPINS: u32 = 128;

/// Metal look-back schedule: `spins` bounds the polls of each predecessor and
/// `reverse` hands tiles out last-first, and `states: false` ignores published
/// inclusive states so every look-back walks aggregates back to the seed.
/// Results never depend on any of them; the adversarial tests use all three.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LookBack {
    pub spins: u32,
    pub reverse: bool,
    pub states: bool,
}

impl LookBack {
    pub const DEFAULT: Self = Self {
        spins: METAL_SPINS,
        reverse: false,
        states: true,
    };
}
/// Coefficients, seed and output together; matches the CUDA transfer cap.
pub const MAX_TRANSFER_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Table(u8),
    Matrix(u8),
}

impl Family {
    pub fn id(self) -> &'static str {
        match self {
            Self::Table(_) => "finite-state-table-v1",
            Self::Matrix(_) => "u32-matrix-mod2p32-v1",
        }
    }
    fn valid(self) -> bool {
        match self {
            Self::Table(k) => (1..=16).contains(&k),
            Self::Matrix(d) => (1..=4).contains(&d),
        }
    }
    /// u32 words of state: one table index, or d matrix-vector entries.
    pub fn state_width(self) -> u32 {
        match self {
            Self::Table(_) => 1,
            Self::Matrix(d) => u32::from(d),
        }
    }
    /// u32 coefficient words per step: K table entries or d*d row-major entries.
    pub fn step_words(self) -> u32 {
        match self {
            Self::Table(k) => u32::from(k),
            Self::Matrix(d) => u32::from(d) * u32::from(d),
        }
    }
}

/// Consecutive steps each thread composes per round on `caps`.
pub fn items(family: Family, observation: Observation, caps: Capabilities) -> u32 {
    let words = match (family, caps, observation) {
        (Family::Matrix(_), _, _) => MATRIX_ITEM_WORDS,
        (Family::Table(_), Capabilities::Cuda { .. }, _) => CUDA_TABLE_ITEM_WORDS,
        (Family::Table(_), Capabilities::Metal { .. }, Observation::FinalState) => {
            METAL_TABLE_FINAL_ITEM_WORDS
        }
        (Family::Table(_), Capabilities::Metal { .. }, Observation::InclusivePrefixes) => {
            METAL_TABLE_ITEM_WORDS
        }
    };
    (words / family.step_words()).max(1)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Observation {
    FinalState,
    InclusivePrefixes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub family: Family,
    pub observation: Observation,
    pub steps: u32,
    pub source_identity: u64,
    pub graph_revision: u64,
}

/// Only `verify` constructs this capability to emit and launch a summary scan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Verified {
    plan: Plan,
    caps: Capabilities,
}

impl Verified {
    pub fn plan(&self) -> Plan {
        self.plan
    }
    pub fn capabilities(&self) -> Capabilities {
        self.caps
    }
    /// Steps per block (CUDA) or threadgroup (Metal) tile.
    pub fn tile_steps(&self) -> u32 {
        let rounds = match (self.caps, self.plan.observation) {
            (Capabilities::Metal { .. }, Observation::FinalState) => METAL_ROUNDS,
            _ => 1,
        };
        LANES * items(self.plan.family, self.plan.observation, self.caps) * rounds
    }
    pub fn blocks(&self) -> u32 {
        self.plan.steps.div_ceil(self.tile_steps()).max(1)
    }
    pub fn coefficient_words(&self) -> usize {
        self.plan.steps as usize * self.plan.family.step_words() as usize
    }
    pub fn output_words(&self) -> usize {
        let width = self.plan.family.state_width() as usize;
        match self.plan.observation {
            Observation::FinalState => width,
            Observation::InclusivePrefixes => self.plan.steps as usize * width,
        }
    }
    /// Required bytes: every coefficient and seed word read once, every output
    /// word written once. Look-back slots and fallback re-reads are not
    /// charged.
    pub fn transfer_bytes(&self) -> u64 {
        (self.coefficient_words() as u64
            + u64::from(self.plan.family.state_width())
            + self.output_words() as u64)
            * 4
    }
}

/// Runtime input refusal, checked on the host before any device work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputError {
    /// Coefficient or seed word count differs from the verified plan.
    Shape,
    /// A finite-state table entry is not a state of the K-state machine.
    TableEntry { index: usize, value: u32 },
    /// The finite-state seed is not a state of the K-state machine.
    Seed { value: u32 },
}

impl std::fmt::Display for InputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shape => write!(f, "{SCHEDULE} input shape differs from the verified plan"),
            Self::TableEntry { index, value } => {
                write!(f, "{SCHEDULE} table entry {index} = {value} is not a state")
            }
            Self::Seed { value } => write!(f, "{SCHEDULE} seed {value} is not a state"),
        }
    }
}

/// Validate caller inputs against the verified plan; the kernels rely on it.
pub fn check_inputs(plan: &Verified, coefficients: &[u32], seed: &[u32]) -> Result<(), InputError> {
    if coefficients.len() != plan.coefficient_words()
        || seed.len() != plan.plan.family.state_width() as usize
    {
        return Err(InputError::Shape);
    }
    if let Family::Table(k) = plan.plan.family {
        let k = u32::from(k);
        if let Some((index, &value)) = coefficients.iter().enumerate().find(|(_, v)| **v >= k) {
            return Err(InputError::TableEntry { index, value });
        }
        if seed[0] >= k {
            return Err(InputError::Seed { value: seed[0] });
        }
    }
    Ok(())
}

pub fn verify(plan: Plan, origin: Provenance, caps: Capabilities) -> Result<Verified, Error> {
    if plan.source_identity == 0 {
        return Err(Error::Source);
    }
    if (plan.source_identity, plan.graph_revision)
        != (origin.source_identity, origin.graph_revision)
    {
        return Err(Error::Graph);
    }
    if !plan.family.valid() {
        return Err(Error::Operations);
    }
    let width = u64::from(plan.family.state_width());
    let output = match plan.observation {
        Observation::FinalState => width,
        Observation::InclusivePrefixes => u64::from(plan.steps) * width,
    };
    let bytes = (u64::from(plan.steps) * u64::from(plan.family.step_words()) + width + output) * 4;
    if bytes > MAX_TRANSFER_BYTES {
        return Err(Error::Bounds);
    }
    match caps {
        Capabilities::Cuda {
            major: 12,
            minor,
            sm,
        } if minor >= 4 && sm >= 50 => {}
        Capabilities::Metal {
            apple_family,
            msl_major: 3,
            msl_minor: 1,
        } if apple_family >= 7 => {}
        _ => return Err(Error::Toolchain),
    }
    Ok(Verified { plan, caps })
}

fn header(plan: &Verified) -> String {
    let p = plan.plan;
    let dimension = match p.family {
        Family::Table(n) | Family::Matrix(n) => n,
    };
    format!(
        "// {SCHEDULE} family={} dimension={dimension} observation={:?} caps={:?}\n",
        p.family.id(),
        p.observation,
        plan.caps
    )
}

/// Packed table: entry `s` occupies bits `4s..4s+3`; entries `s >= K` stay the
/// identity. Entries are validated below K before launch; the 4-bit fields
/// keep every lookup inside the table regardless.
const TABLE_OPS: &str = r#"#define INVAR_W 1u
#define INVAR_S @D@u
struct InvarSum { ulong p; };
INVAR_FN InvarSum invar_identity() { InvarSum r; r.p = ((ulong)0xFEDCBA98u << 32) | (ulong)0x76543210u; return r; }
#define INVAR_HIGH_SHIFT ((4u * INVAR_S) & 63u)
INVAR_FN ulong invar_high() { return INVAR_S >= 16u ? (ulong)0 : (invar_identity().p >> INVAR_HIGH_SHIFT) << INVAR_HIGH_SHIFT; }
INVAR_FN InvarSum invar_from_words(thread const uint* w, uint base) {
  InvarSum r; r.p = invar_high();
  INVAR_UNROLL for (uint s = 0u; s < INVAR_S; ++s) r.p |= (ulong)(w[base + s] & 15u) << (4u * s);
  return r;
}
/* First a, then b: entry s becomes b[a[s]]. */
INVAR_FN InvarSum invar_then(InvarSum a, InvarSum b) {
  InvarSum r; r.p = invar_high();
  INVAR_UNROLL for (uint s = 0u; s < INVAR_S; ++s)
    r.p |= ((b.p >> (4u * (uint)((a.p >> (4u * s)) & (ulong)15))) & (ulong)15) << (4u * s);
  return r;
}
INVAR_FN void invar_apply(InvarSum a, thread uint* x) { x[0] = (uint)((a.p >> (4u * (x[0] & 15u))) & (ulong)15); }
INVAR_FN InvarSum invar_shuffle_up(InvarSum a, uint d) {
  const uint lo = INVAR_SHFL_UP((uint)a.p, d);
  const uint hi = INVAR_SHFL_UP((uint)(a.p >> 32), d);
  InvarSum r; r.p = ((ulong)hi << 32) | (ulong)lo; return r;
}
INVAR_FN void invar_store(INVAR_STATUS uint* p, InvarSum a) { p[0] = (uint)a.p; p[1] = (uint)(a.p >> 32); }
INVAR_FN InvarSum invar_fetch(INVAR_STATUS const uint* p) { InvarSum r; r.p = ((ulong)p[1] << 32) | (ulong)p[0]; return r; }
"#;

const TABLE_WORDS: &str = r#"#define INVAR_SUM_WORDS 2u
INVAR_FN void invar_to_words(InvarSum a, thread uint* w) { w[0] = (uint)a.p; w[1] = (uint)(a.p >> 32); }
INVAR_FN InvarSum invar_of_words(thread const uint* w) { InvarSum r; r.p = ((ulong)w[1] << 32) | (ulong)w[0]; return r; }
"#;

const MATRIX_OPS: &str = r#"#define INVAR_W @D@u
#define INVAR_D @D@u
#define INVAR_S (INVAR_D * INVAR_D)
struct InvarSum { uint v[16]; };
INVAR_FN InvarSum invar_identity() {
  InvarSum r;
  INVAR_UNROLL for (uint e = 0u; e < 16u; ++e) r.v[e] = 0u;
  INVAR_UNROLL for (uint k = 0u; k < INVAR_D; ++k) r.v[k * INVAR_D + k] = 1u;
  return r;
}
INVAR_FN InvarSum invar_from_words(thread const uint* w, uint base) {
  InvarSum r = invar_identity();
  INVAR_UNROLL for (uint e = 0u; e < INVAR_S; ++e) r.v[e] = w[base + e];
  return r;
}
/* First a, then b: the product b * a in the ring of u32 modulo 2^32. */
INVAR_FN InvarSum invar_then(InvarSum a, InvarSum b) {
  InvarSum r = invar_identity();
  INVAR_UNROLL for (uint row = 0u; row < INVAR_D; ++row)
    INVAR_UNROLL for (uint col = 0u; col < INVAR_D; ++col) {
      uint value = 0u;
      INVAR_UNROLL for (uint k = 0u; k < INVAR_D; ++k) value += b.v[row * INVAR_D + k] * a.v[k * INVAR_D + col];
      r.v[row * INVAR_D + col] = value;
    }
  return r;
}
INVAR_FN void invar_apply(InvarSum a, thread uint* x) {
  uint y[4];
  INVAR_UNROLL for (uint row = 0u; row < INVAR_D; ++row) {
    uint value = 0u;
    INVAR_UNROLL for (uint k = 0u; k < INVAR_D; ++k) value += a.v[row * INVAR_D + k] * x[k];
    y[row] = value;
  }
  INVAR_UNROLL for (uint row = 0u; row < INVAR_D; ++row) x[row] = y[row];
}
INVAR_FN InvarSum invar_shuffle_up(InvarSum a, uint d) {
  InvarSum r = a;
  INVAR_UNROLL for (uint e = 0u; e < INVAR_S; ++e) r.v[e] = INVAR_SHFL_UP(a.v[e], d);
  return r;
}
INVAR_FN void invar_store(INVAR_STATUS uint* p, InvarSum a) { INVAR_UNROLL for (uint e = 0u; e < INVAR_S; ++e) p[e] = a.v[e]; }
INVAR_FN InvarSum invar_fetch(INVAR_STATUS const uint* p) {
  InvarSum r = invar_identity();
  INVAR_UNROLL for (uint e = 0u; e < INVAR_S; ++e) r.v[e] = p[e];
  return r;
}
"#;

const MATRIX_WORDS: &str = r#"#define INVAR_SUM_WORDS INVAR_S
INVAR_FN void invar_to_words(InvarSum a, thread uint* w) { INVAR_UNROLL for (uint e = 0u; e < INVAR_S; ++e) w[e] = a.v[e]; }
INVAR_FN InvarSum invar_of_words(thread const uint* w) { return invar_from_words(w, 0u); }
"#;

/// Shared by both targets: per-thread steps in registers and the ordered
/// block scan (SIMD shuffle-up within each group of 32, then group prefixes).
const COMMON: &str = r#"#define INVAR_LANES 256u
#define INVAR_WARPS 8u
#define INVAR_ITEMS @ITEMS@u
#define INVAR_TILE (INVAR_LANES * INVAR_ITEMS)
#define INVAR_PREFIX @PREFIX@
/* Compose this thread's INVAR_ITEMS consecutive steps from `first` in order,
   keeping their words in registers for the write pass. Steps of a multiple of
   4 words are 16-byte aligned, so a full run loads each step as uint4s. */
INVAR_FN InvarSum invar_thread_load(INVAR_GLOBAL const uint* c, ulong first, uint n, thread uint* w) {
#if INVAR_S % 4u == 0u
  if (first + INVAR_ITEMS <= (ulong)n) {
    INVAR_GLOBAL const uint4* q = (INVAR_GLOBAL const uint4*)(c + first * INVAR_S);
    InvarSum acc = invar_identity();
    INVAR_UNROLL for (uint j = 0u; j < INVAR_ITEMS; ++j) {
      INVAR_UNROLL for (uint i = 0u; i < INVAR_S / 4u; ++i) {
        const uint4 v = q[j * (INVAR_S / 4u) + i];
        const uint b = j * INVAR_S + 4u * i;
        w[b] = v.x; w[b + 1u] = v.y; w[b + 2u] = v.z; w[b + 3u] = v.w;
      }
      const InvarSum step = invar_from_words(w, j * INVAR_S);
      acc = j == 0u ? step : invar_then(acc, step);
    }
    return acc;
  }
#endif
  InvarSum acc = invar_identity();
  INVAR_UNROLL for (uint j = 0u; j < INVAR_ITEMS; ++j) {
    if (first + j < (ulong)n) {
      INVAR_UNROLL for (uint s = 0u; s < INVAR_S; ++s) w[j * INVAR_S + s] = c[(first + j) * INVAR_S + s];
      const InvarSum step = invar_from_words(w, j * INVAR_S);
      acc = j == 0u ? step : invar_then(acc, step);
    }
  }
  return acc;
}
#if INVAR_PREFIX
/* Apply this thread's steps to x in order, writing each inclusive state. */
INVAR_FN void invar_thread_write(thread const uint* w, ulong first, uint n, thread uint* x, INVAR_GLOBAL uint* out) {
  INVAR_UNROLL for (uint j = 0u; j < INVAR_ITEMS; ++j) {
    if (first + j < (ulong)n) {
      invar_apply(invar_from_words(w, j * INVAR_S), x);
      INVAR_UNROLL for (uint r = 0u; r < INVAR_W; ++r) out[(first + j) * INVAR_W + r] = x[r];
    }
  }
}
#endif
/* Ordered exclusive scan of one value per thread; *total receives the
   composition of all INVAR_LANES values. `sums` has INVAR_WARPS + 1 slots. */
INVAR_FN InvarSum invar_block_scan(InvarSum mine, uint t, uint lane, uint warp, INVAR_SHARED InvarSum* sums, thread InvarSum* total) {
  InvarSum inc = mine;
  for (uint d = 1u; d < 32u; d <<= 1u) {
    const InvarSum up = invar_shuffle_up(inc, d);
    if (lane >= d) inc = invar_then(up, inc);
  }
  InvarSum before = invar_shuffle_up(inc, 1u);
  if (lane == 0u) before = invar_identity();
  if (lane == 31u) sums[warp] = inc;
  INVAR_BARRIER();
  InvarSum prefix = invar_identity();
  for (uint k = 0u; k < warp; ++k) { const InvarSum s = sums[k]; prefix = invar_then(prefix, s); }
  if (t == INVAR_LANES - 1u) sums[INVAR_WARPS] = invar_then(prefix, inc);
  INVAR_BARRIER();
  *total = sums[INVAR_WARPS];
  return invar_then(prefix, before);
}
"#;

const CUDA_PRELUDE: &str = r#"typedef unsigned int uint;
typedef unsigned long long ulong;
#define thread
#define INVAR_FN __device__ static inline
#define INVAR_GLOBAL
#define INVAR_STATUS volatile
#define INVAR_SHARED
#define INVAR_UNROLL _Pragma("unroll")
#define INVAR_BARRIER() __syncthreads()
#define INVAR_SHFL_UP(v, d) __shfl_up_sync(0xffffffffu, (v), (d))
"#;

/// One launch of `blocks` blocks; `status` (blocks * 24 words) and `ticket`
/// are zeroed on the stream first. The ticket orders tiles by start, so every
/// predecessor a block waits on is already resident.
const CUDA: &str = r#"#define INVAR_STATUS_WORDS 24u
extern "C" __global__ void __launch_bounds__(256) invar_summary_scan_v1(const uint* coefficients, const uint* seed,
    uint* output, uint* status, uint* ticket, uint n) {
  __shared__ InvarSum sums[INVAR_WARPS + 1u];
  __shared__ uint shared_tile;
#if INVAR_PREFIX
  __shared__ uint shared_state[4];
#endif
  const uint t = threadIdx.x, lane = t & 31u, warp = t >> 5u;
  if (t == 0u) shared_tile = atomicAdd(ticket, 1u);
  __syncthreads();
  const uint tile = shared_tile;
  uint w[INVAR_ITEMS * INVAR_S];
  InvarSum total;
  const InvarSum before = invar_block_scan(invar_thread_load(coefficients, (ulong)tile * INVAR_TILE + (ulong)t * INVAR_ITEMS, n, w),
      t, lane, warp, sums, &total);
  if (t == 0u) {
    volatile uint* entry = status + (ulong)tile * INVAR_STATUS_WORDS;
    uint x[INVAR_W];
    for (uint r = 0u; r < INVAR_W; ++r) x[r] = seed[r];
    if (tile > 0u) {
      invar_store(entry + 8u, total);
      __threadfence();
      entry[0] = 1u;
      /* Look back: compose published aggregates, newest last, until a
         predecessor's inclusive state is available. Tile 0 always is. */
      InvarSum acc = invar_identity();
      for (uint j = tile - 1u; j < tile; --j) {
        volatile uint* other = status + (ulong)j * INVAR_STATUS_WORDS;
        uint flag;
        do { flag = other[0]; } while (flag == 0u);
        __threadfence();
        if (flag == 2u) {
          for (uint r = 0u; r < INVAR_W; ++r) x[r] = other[1u + r];
          break;
        }
        acc = invar_then(invar_fetch(other + 8u), acc);
      }
      invar_apply(acc, x);
    }
#if INVAR_PREFIX
    for (uint r = 0u; r < INVAR_W; ++r) shared_state[r] = x[r];
#endif
    invar_apply(total, x);
    for (uint r = 0u; r < INVAR_W; ++r) entry[1u + r] = x[r];
    __threadfence();
    entry[0] = 2u;
#if !INVAR_PREFIX
    if ((ulong)(tile + 1u) * INVAR_TILE >= (ulong)n) for (uint r = 0u; r < INVAR_W; ++r) output[r] = x[r];
#endif
  }
#if INVAR_PREFIX
  __syncthreads();
  uint x[INVAR_W];
  for (uint r = 0u; r < INVAR_W; ++r) x[r] = shared_state[r];
  invar_apply(before, x);
  invar_thread_write(w, (ulong)tile * INVAR_TILE + (ulong)t * INVAR_ITEMS, n, x, output);
#else
  (void)before;
#endif
}
"#;

const MSL_PRELUDE: &str = r#"#include <metal_stdlib>
using namespace metal;
#define INVAR_FN static inline
#define INVAR_GLOBAL device
#define INVAR_STATUS device
#define INVAR_SHARED threadgroup
#define INVAR_UNROLL _Pragma("unroll")
#define INVAR_BARRIER() threadgroup_barrier(mem_flags::mem_threadgroup)
#define INVAR_SHFL_UP(v, d) simd_shuffle_up((v), (ushort)(d))
"#;

/// Final-state plans (which read coefficients once already) run reduce over
/// `METAL_ROUNDS` rounds per threadgroup, then one threadgroup composes the
/// aggregates. Prefix plans run `invar_summary_clear_v3`, which zeroes `status`
/// (tiles * 40 + 2 words), then one single-pass `invar_summary_scan_v3`
/// dispatch; word tiles * 40 hands out tiles and word tiles * 40 + 1 counts
/// fallbacks. `counts` holds [steps, threadgroups].
const MSL: &str = r#"#define INVAR_IDS uint t [[thread_position_in_threadgroup]], uint group [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_simdgroup]], uint warp [[simdgroup_index_in_threadgroup]]
#if !INVAR_PREFIX
#define INVAR_ROUNDS @ROUNDS@u
kernel void invar_summary_reduce_v1(device const uint* coefficients [[buffer(0)]], device uint* aggregates [[buffer(1)]],
    device const uint* counts [[buffer(2)]], INVAR_IDS) {
  const uint n = counts[0];
  threadgroup InvarSum sums[INVAR_WARPS + 1u];
  uint w[INVAR_ITEMS * INVAR_S];
  InvarSum acc = invar_identity();
  for (uint round = 0u; round < INVAR_ROUNDS; ++round) {
    const ulong start = ((ulong)group * INVAR_ROUNDS + round) * INVAR_TILE;
    if (start >= (ulong)n) break;
    InvarSum total;
    invar_block_scan(invar_thread_load(coefficients, start + (ulong)t * INVAR_ITEMS, n, w), t, lane, warp, sums, &total);
    acc = invar_then(acc, total);
  }
  if (t == 0u) invar_store(aggregates + (ulong)group * 16u, acc);
}
kernel void invar_summary_final_v3(device const uint* aggregates [[buffer(0)]], device const uint* seed [[buffer(1)]],
    device uint* output [[buffer(2)]], device const uint* counts [[buffer(3)]], INVAR_IDS) {
  const uint groups = counts[1];
  threadgroup InvarSum sums[INVAR_WARPS + 1u];
  uint carry[INVAR_W];
  for (uint r = 0u; r < INVAR_W; ++r) carry[r] = seed[r];
  for (uint base = 0u; base < groups; base += INVAR_LANES) {
    const uint b = base + t;
    InvarSum total;
    invar_block_scan(b < groups ? invar_fetch(aggregates + (ulong)b * 16u) : invar_identity(), t, lane, warp, sums, &total);
    invar_apply(total, carry);
  }
  if (t == 0u) for (uint r = 0u; r < INVAR_W; ++r) output[r] = carry[r];
}
#else
#define INVAR_SPINS @SPINS@u
#define INVAR_REVERSE @REVERSE@
#define INVAR_STATES @STATES@
#define INVAR_SLOT 40u
#define INVAR_STATE_AT 32u
#define INVAR_WINDOW 4u
/* Each u32 is published as two atomic words (1 << 16) | 16 payload bits. A
   reader that sees bit 16 of a word sees that word's payload (coherence of one
   atomic location), so no ordering between words or slots is assumed: a slot
   is used only when every word is published, and every writer of a word
   writes the same value. */
INVAR_FN void invar_publish(device atomic_uint* slot, thread const uint* w, uint words) {
  for (uint i = 0u; i < words; ++i) {
    atomic_store_explicit(slot + 2u * i, 0x10000u | (w[i] & 0xFFFFu), memory_order_relaxed);
    atomic_store_explicit(slot + 2u * i + 1u, 0x10000u | (w[i] >> 16), memory_order_relaxed);
  }
}
INVAR_FN bool invar_poll(device atomic_uint* slot, thread uint* w, uint words) {
  for (uint i = 0u; i < words; ++i) {
    const uint lo = atomic_load_explicit(slot + 2u * i, memory_order_relaxed);
    const uint hi = atomic_load_explicit(slot + 2u * i + 1u, memory_order_relaxed);
    if ((lo & hi & 0x10000u) == 0u) return false;
    w[i] = (lo & 0xFFFFu) | (hi << 16);
  }
  return true;
}
/* Fallback: compose this thread's run of a tile without keeping it. */
INVAR_FN InvarSum invar_thread_reduce(device const uint* c, ulong first, uint n) {
  InvarSum acc = invar_identity();
  for (uint j = 0u; j < INVAR_ITEMS; ++j) {
    if (first + j < (ulong)n) {
      uint s[INVAR_S];
      INVAR_UNROLL for (uint e = 0u; e < INVAR_S; ++e) s[e] = c[(first + j) * INVAR_S + e];
      const InvarSum step = invar_from_words(s, 0u);
      acc = j == 0u ? step : invar_then(acc, step);
    }
  }
  return acc;
}
/* invar_thread_write with one 16-byte store per 4-wide state. */
INVAR_FN void invar_thread_store(thread const uint* w, ulong first, uint n, thread uint* x, device uint* out) {
  INVAR_UNROLL for (uint j = 0u; j < INVAR_ITEMS; ++j) {
    if (first + j < (ulong)n) {
      invar_apply(invar_from_words(w, j * INVAR_S), x);
#if INVAR_W == 4u
      ((device uint4*)out)[first + j] = uint4(x[0], x[1], x[2], x[3]);
#else
      INVAR_UNROLL for (uint r = 0u; r < INVAR_W; ++r) out[(first + j) * INVAR_W + r] = x[r];
#endif
    }
  }
}
kernel void invar_summary_clear_v3(device uint* status [[buffer(0)]], device const uint* counts [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
  if ((ulong)i < (ulong)counts[1] * INVAR_SLOT + 2u) status[i] = 0u;
}
/* Lanes of SIMD group 0 poll the INVAR_WINDOW nearest unresolved predecessors
   at once, each at most INVAR_SPINS times; lanes before tile 0 read the seed as
   a published state. Thread 0 composes the window's aggregates newest last up
   to the first lane that is not an aggregate: a published inclusive state ends
   the look-back, and an unpublished tile is reduced from its own coefficients
   by the whole threadgroup, which never depends on another threadgroup, and its
   aggregate published for other waiters. Every round consumes at least one
   tile, so at most tile + 1 rounds run whatever the schedule. */
kernel void invar_summary_scan_v3(device const uint* coefficients [[buffer(0)]], device const uint* seed [[buffer(1)]],
    device uint* output [[buffer(2)]], device atomic_uint* status [[buffer(3)]], device const uint* counts [[buffer(4)]], INVAR_IDS) {
  const uint n = counts[0], tiles = counts[1];
  threadgroup InvarSum sums[INVAR_WARPS + 1u];
  threadgroup uint shared_tile, shared_kind, shared_skip, shared_state[4], window_kind[INVAR_WINDOW], window_words[INVAR_WINDOW * 16u];
  if (t == 0u) {
    const uint ticket = atomic_fetch_add_explicit(status + (ulong)tiles * INVAR_SLOT, 1u, memory_order_relaxed);
    shared_tile = INVAR_REVERSE ? tiles - 1u - ticket : ticket;
  }
  INVAR_BARRIER();
  const uint tile = shared_tile;
  uint w[INVAR_ITEMS * INVAR_S];
  InvarSum total;
  const InvarSum before = invar_block_scan(invar_thread_load(coefficients, (ulong)tile * INVAR_TILE + (ulong)t * INVAR_ITEMS, n, w),
      t, lane, warp, sums, &total);
  device atomic_uint* entry = status + (ulong)tile * INVAR_SLOT;
  uint x[INVAR_W];
  for (uint r = 0u; r < INVAR_W; ++r) x[r] = seed[r];
  if (tile > 0u) {
    if (t == 0u) { uint words[16]; invar_to_words(total, words); invar_publish(entry, words, INVAR_SUM_WORDS); }
    InvarSum acc = invar_identity();
    uint next = tile;
    for (uint round = 0u; round <= tile; ++round) {
      if (warp == 0u && lane < INVAR_WINDOW) {
        uint kind = 0u, words[16];
        for (uint i = 0u; i < 16u; ++i) words[i] = 0u;
        if (lane >= next) {
          kind = 2u;
          for (uint r = 0u; r < INVAR_W; ++r) words[r] = seed[r];
        } else {
          device atomic_uint* other = status + (ulong)(next - 1u - lane) * INVAR_SLOT;
          for (uint spin = 0u; spin < INVAR_SPINS && kind == 0u; ++spin)
            kind = INVAR_STATES && invar_poll(other + INVAR_STATE_AT, words, INVAR_W) ? 2u : invar_poll(other, words, INVAR_SUM_WORDS) ? 1u : 0u;
        }
        window_kind[lane] = kind;
        for (uint i = 0u; i < 16u; ++i) window_words[lane * 16u + i] = words[i];
      }
      INVAR_BARRIER();
      if (t == 0u) {
        uint stop = 0u;
        for (; stop < INVAR_WINDOW && window_kind[stop] == 1u; ++stop) {
          uint words[16];
          for (uint i = 0u; i < 16u; ++i) words[i] = window_words[stop * 16u + i];
          acc = invar_then(invar_of_words(words), acc);
        }
        const uint kind = stop < INVAR_WINDOW ? window_kind[stop] : 1u;
        if (kind == 2u) for (uint r = 0u; r < INVAR_W; ++r) x[r] = window_words[stop * 16u + r];
        if (kind == 0u) atomic_fetch_add_explicit(status + (ulong)tiles * INVAR_SLOT + 1u, 1u, memory_order_relaxed);
        shared_kind = kind;
        shared_skip = stop;
      }
      INVAR_BARRIER();
      const uint kind = shared_kind, skip = shared_skip;
      if (kind == 2u) break;
      if (kind == 0u) {
        InvarSum aggregate;
        invar_block_scan(invar_thread_reduce(coefficients, (ulong)(next - 1u - skip) * INVAR_TILE + (ulong)t * INVAR_ITEMS, n),
            t, lane, warp, sums, &aggregate);
        if (t == 0u) {
          /* The same words the owner publishes, so other waiters can use them. */
          uint words[16];
          invar_to_words(aggregate, words);
          invar_publish(status + (ulong)(next - 1u - skip) * INVAR_SLOT, words, INVAR_SUM_WORDS);
          acc = invar_then(aggregate, acc);
        }
        next -= skip + 1u;
      } else {
        next -= INVAR_WINDOW;
      }
      INVAR_BARRIER();
    }
    if (t == 0u) invar_apply(acc, x);
  }
  if (t == 0u) {
    uint after[INVAR_W];
    for (uint r = 0u; r < INVAR_W; ++r) { shared_state[r] = x[r]; after[r] = x[r]; }
    invar_apply(total, after);
    invar_publish(entry + INVAR_STATE_AT, after, INVAR_W);
  }
  INVAR_BARRIER();
  for (uint r = 0u; r < INVAR_W; ++r) x[r] = shared_state[r];
  invar_apply(before, x);
  invar_thread_store(w, (ulong)tile * INVAR_TILE + (ulong)t * INVAR_ITEMS, n, x, output);
}
#endif
"#;

fn emit(plan: &Verified, prelude: &str, words: bool, kernels: &str) -> Result<String, Error> {
    let p = plan.plan;
    let (ops, conversions, dimension) = match p.family {
        Family::Table(k) => (TABLE_OPS, TABLE_WORDS, k),
        Family::Matrix(d) => (MATRIX_OPS, MATRIX_WORDS, d),
    };
    let ops = ops.to_owned() + if words { conversions } else { "" };
    let common = COMMON
        .replace(
            "@ITEMS@",
            &items(p.family, p.observation, plan.caps).to_string(),
        )
        .replace(
            "@PREFIX@",
            match p.observation {
                Observation::FinalState => "0",
                Observation::InclusivePrefixes => "1",
            },
        );
    let source =
        header(plan) + prelude + &ops.replace("@D@", &dimension.to_string()) + &common + kernels;
    if source.len() > super::MAX_SOURCE_BYTES {
        return Err(Error::SourceLimit);
    }
    Ok(source)
}

pub fn emit_cuda(plan: &Verified) -> Result<String, Error> {
    if !matches!(plan.caps, Capabilities::Cuda { .. }) {
        return Err(Error::Toolchain);
    }
    emit(plan, CUDA_PRELUDE, false, CUDA)
}

pub fn emit_msl(plan: &Verified) -> Result<String, Error> {
    emit_msl_with(plan, LookBack::DEFAULT)
}

/// `emit_msl` with an explicit look-back schedule, for adversarial tests.
pub fn emit_msl_with(plan: &Verified, look_back: LookBack) -> Result<String, Error> {
    if !matches!(plan.caps, Capabilities::Metal { .. }) {
        return Err(Error::Toolchain);
    }
    let kernels = MSL
        .replace("@ROUNDS@", &METAL_ROUNDS.to_string())
        .replace("@SPINS@", &look_back.spins.to_string())
        .replace("@REVERSE@", if look_back.reverse { "1" } else { "0" })
        .replace("@STATES@", if look_back.states { "1" } else { "0" });
    emit(plan, MSL_PRELUDE, true, &kernels)
}
