//! Verified strict, row-major rank-2 binary32 contractions.
//!
//! This descriptor proves static physical bounds, not source extraction,
//! dynamic pointer ownership, source fuel/trap commit, fences or launch legality.
//! Those remain checked-source/host gates. The only relaxed policy admitted is
//! the explicit `tc-f16-f32-v1` contract (D5 as amended), whose plans only the
//! `tc` emitters accept; every strict emitter refuses them.
//! One verified plan emits CUDA, Metal and C++ from the same IR. Static
//! verification never proves device numerics: each runtime must also pass the
//! per-process D1 probe in `f32_probe` before submitting strict work.

use crate::{Capabilities, Provenance, Span};
use std::fmt::Write;

pub const VERSION: u32 = 1;
pub const NUMERIC_VERSION: u32 = 1;
pub const LAYOUT_VERSION: u32 = 1;
pub const NUMERIC_POLICY: &str = "strict-f32-v1";
/// The explicit Tensor Core contract (D5 as amended on 2026-10-14).
pub const TC_POLICY: &str = "tc-f16-f32-v1";
/// Match the frozen layout contract's 2 GiB allocation and extent bounds.
pub const MAX_ELEMENTS: u64 = 1 << 29;
pub const MAX_EXTENT: u32 = (1 << 31) - 1;
/// Logical cells per matrix, including the M5-01 16384-square case.
pub const MAX_MATRIX_ELEMENTS: u64 = 1 << 28;
/// Three pairwise matrix sizes <= 2^28 bound M*K*N by 2^42.
pub const MAX_MULTIPLY_ADDS: u64 = 1 << 42;
pub const KERNEL_NAME: &str = "invar_f32_contract_v1";
/// Pinned Metal contract: MSL 3.1 with fast math disabled, plus the in-source
/// `#pragma METAL fp contract(off)`. Without the pragma the Metal compiler
/// fuses `sum + a * b` into one FMA even with fast math off (observed on Apple
/// M4, macOS 27), which strict-f32-v1 forbids.
pub const MSL_OPTIONS: &[&str] = &[
    "-std=metal3.1",
    "-fno-fast-math",
    "#pragma METAL fp contract(off)",
];
const CPU_OPTIONS: &[&str] = &[
    "-std=c++17",
    "-fno-fast-math",
    "-ffp-contract=off",
    "-frounding-math",
    "-fno-associative-math",
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Buffer {
    pub id: u32,
    pub elements: u64,
    pub shape: [u32; 2],
    pub strides: [u64; 2],
    /// Offset in binary32 elements from an allocation base.
    pub offset: u64,
    /// Guaranteed byte alignment of that allocation base.
    pub alignment: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ir {
    pub version: u32,
    pub numeric_version: u32,
    pub numeric_policy: String,
    pub layout_version: u32,
    pub span: Span,
    pub source_identity: u64,
    pub graph_revision: u64,
    pub function_contract: [u8; 32],
    pub dimensions: [u32; 3],
    pub inputs: [Buffer; 2],
    pub output: Buffer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Version,
    Numeric,
    Layout,
    Source,
    Graph,
    Contract,
    Shape,
    Alignment,
    Bounds,
    Alias,
    Capabilities,
    Index,
    WorkLimit,
    MatrixLimit,
    SourceLimit,
    /// A `tc-f16-f32-v1` inner extent above `tc::MAX_INNER`, where the
    /// amended D5 bound is no longer proven.
    InnerLimit,
    /// No target description (`targets/*.json`) selects the plan's device.
    TargetUnknown,
}

impl Error {
    /// The diagnostic code of a refusal that has its own.
    pub fn code(&self) -> Option<&'static str> {
        match self {
            Self::TargetUnknown => Some("TARGET_UNKNOWN"),
            _ => None,
        }
    }
}

/// Immutable verified metadata, never a permit to launch unvalidated pointers.
#[derive(Debug)]
pub struct Verified {
    ir: Ir,
    caps: Capabilities,
    nvrtc_options: Vec<String>,
    device_ir: crate::v2::Verified,
}

impl Verified {
    pub fn ir(&self) -> &Ir {
        &self.ir
    }
    pub fn device_ir(&self) -> &crate::v2::Verified {
        &self.device_ir
    }
    pub fn capabilities(&self) -> Capabilities {
        self.caps
    }
    /// True for an explicit `tc-f16-f32-v1` plan, which only `tc` emits.
    pub fn is_tc(&self) -> bool {
        self.ir.numeric_policy == TC_POLICY
    }
    /// Required ordered options; host admission must reject overrides/additions
    /// that change this numeric contract. They are embedded in emitted identity.
    /// Empty for a Metal plan.
    pub fn nvrtc_options(&self) -> &[String] {
        &self.nvrtc_options
    }
    /// Required Metal compile contract, embedded in emitted MSL identity.
    pub fn msl_options(&self) -> &'static [&'static str] {
        MSL_OPTIONS
    }
    pub fn cpu_options(&self) -> &'static [&'static str] {
        CPU_OPTIONS
    }
    /// Slots 0/1/2 select A/B/C; empty domains never admit element access.
    pub fn physical_index(&self, slot: usize, index: [u32; 2]) -> Result<u64, Error> {
        let buffer = match slot {
            0 | 1 => &self.ir.inputs[slot],
            2 => &self.ir.output,
            _ => return Err(Error::Index),
        };
        physical_index(buffer, index)
    }
    /// Stable artifact identity following the existing device source-hash API;
    /// the root's SHA256 tensor cache binds this source and exact option bytes.
    pub fn cuda_source_hash(&self) -> Result<u64, Error> {
        Ok(crate::source_hash(&emit_cuda(self)?))
    }
    pub fn msl_source_hash(&self) -> Result<u64, Error> {
        Ok(crate::source_hash(&emit_msl(self)?))
    }
}

/// Exact ordered NVRTC options for strict binary32 on a CUDA target, or `None`
/// when the toolkit/architecture pair is outside the pinned set. Blackwell
/// targets need the NVRTC release that introduced them (12.8: sm_100, sm_101,
/// sm_120; 12.9: sm_103, sm_121).
pub fn cuda_strict_options(caps: Capabilities) -> Option<Vec<String>> {
    let Capabilities::Cuda {
        major: 12,
        minor: minor @ 4..=9,
        sm,
    } = caps
    else {
        return None;
    };
    let known = matches!(
        sm,
        50 | 52 | 53 | 60 | 61 | 62 | 70 | 72 | 75 | 80 | 86 | 87 | 89 | 90
    ) || (minor >= 8 && matches!(sm, 100 | 101 | 120))
        || (minor >= 9 && matches!(sm, 103 | 121));
    known.then(|| {
        vec![
            "--std=c++17".to_owned(),
            format!("--gpu-architecture=compute_{sm}"),
            "--fmad=false".to_owned(),
            "--ftz=false".to_owned(),
            "--prec-div=true".to_owned(),
            "--prec-sqrt=true".to_owned(),
            "--modify-stack-limit=false".to_owned(),
        ]
    })
}

fn physical_index(buffer: &Buffer, index: [u32; 2]) -> Result<u64, Error> {
    if index.iter().zip(buffer.shape).any(|(&i, n)| i >= n) {
        return Err(Error::Index);
    }
    let position = index
        .iter()
        .zip(buffer.strides)
        .try_fold(buffer.offset, |v, (&i, s)| {
            u64::from(i)
                .checked_mul(s)
                .and_then(|delta| v.checked_add(delta))
                .ok_or(Error::Bounds)
        })?;
    if position >= buffer.elements {
        return Err(Error::Bounds);
    }
    Ok(position)
}

fn buffer(buffer: &Buffer) -> Result<(), Error> {
    if buffer.shape.iter().any(|&n| n > MAX_EXTENT) {
        return Err(Error::Bounds);
    }
    if buffer.strides != [u64::from(buffer.shape[1]), 1] {
        return Err(Error::Layout);
    }
    if !buffer.alignment.is_power_of_two() || buffer.alignment < 4 {
        return Err(Error::Alignment);
    }
    if buffer.elements > MAX_ELEMENTS || buffer.offset > buffer.elements {
        return Err(Error::Bounds);
    }
    // Empty layouts admit a one-past offset but no physical load/store.
    if !buffer.shape.contains(&0) {
        physical_index(buffer, [buffer.shape[0] - 1, buffer.shape[1] - 1])?;
    }
    Ok(())
}

pub fn verify(ir: Ir, caps: Capabilities, origin: Provenance) -> Result<Verified, Error> {
    if ir.version != VERSION {
        return Err(Error::Version);
    }
    let tc = ir.numeric_policy == TC_POLICY;
    if ir.numeric_version != NUMERIC_VERSION || !(tc || ir.numeric_policy == NUMERIC_POLICY) {
        return Err(Error::Numeric);
    }
    if ir.layout_version != LAYOUT_VERSION {
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
    let [rows, inner, cols] = ir.dimensions;
    if ir.inputs[0].shape != [rows, inner]
        || ir.inputs[1].shape != [inner, cols]
        || ir.output.shape != [rows, cols]
    {
        return Err(Error::Shape);
    }
    for b in ir.inputs.iter().chain(std::iter::once(&ir.output)) {
        buffer(b)?;
    }
    if ir.output.id == ir.inputs[0].id || ir.output.id == ir.inputs[1].id {
        return Err(Error::Alias);
    }
    // Read aliases are legal, but an allocation ID cannot have conflicting size
    // or base-alignment claims. Different offsets/shapes may select read regions.
    if ir.inputs[0].id == ir.inputs[1].id
        && (ir.inputs[0].elements != ir.inputs[1].elements
            || ir.inputs[0].alignment != ir.inputs[1].alignment)
    {
        return Err(Error::Alias);
    }
    let work = u64::from(rows)
        .checked_mul(u64::from(inner))
        .and_then(|n| n.checked_mul(u64::from(cols)))
        .ok_or(Error::WorkLimit)?;
    if work > MAX_MULTIPLY_ADDS {
        return Err(Error::WorkLimit);
    }
    if tc && inner > crate::tc::MAX_INNER {
        return Err(Error::InnerLimit);
    }
    for shape in [[rows, inner], [inner, cols], [rows, cols]] {
        if u64::from(shape[0]) * u64::from(shape[1]) > MAX_MATRIX_ELEMENTS {
            return Err(Error::MatrixLimit);
        }
    }
    // mma.sync m16n8k16 with f16 inputs needs sm_80 or newer.
    if tc && matches!(caps, Capabilities::Cuda { sm, .. } if sm < 80) {
        return Err(Error::Capabilities);
    }
    let nvrtc_options = match caps {
        Capabilities::Cuda { .. } => cuda_strict_options(caps).ok_or(Error::Capabilities)?,
        Capabilities::Metal {
            apple_family: 7..=10,
            msl_major: 3,
            msl_minor: 1,
        } => Vec::new(),
        Capabilities::Metal { .. } => return Err(Error::Capabilities),
    };
    let device_ir =
        crate::v2::verify(crate::v2::contraction(&ir), origin, caps).map_err(|_| Error::Layout)?;
    Ok(Verified {
        ir,
        caps,
        nvrtc_options,
        device_ir,
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Target {
    Cpu,
    Cuda,
    Metal,
}

pub(crate) fn header(plan: &Verified, target: Target) -> Result<String, Error> {
    let ir = &plan.ir;
    let mut text = format!(
        "// invar-f32-contract-v1 numeric={} numeric-version={} layout-version={}\n// source={} graph={} span={}:{} caps={:?}\n// function-contract=",
        ir.numeric_policy,
        ir.numeric_version,
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
        "\n// device-ir-version={}\n// dimensions={:?}",
        plan.device_ir.ir().version,
        ir.dimensions
    )
    .map_err(|_| Error::SourceLimit)?;
    for (slot, b) in ir
        .inputs
        .iter()
        .chain(std::iter::once(&ir.output))
        .enumerate()
    {
        writeln!(
            text,
            "// buffer{slot} id={} elements={} shape={:?} strides={:?} offset={} alignment={}",
            b.id, b.elements, b.shape, b.strides, b.offset, b.alignment
        )
        .map_err(|_| Error::SourceLimit)?;
    }
    text.push_str("// required-options:");
    let options: Vec<&str> = match target {
        Target::Cpu => CPU_OPTIONS.to_vec(),
        Target::Cuda => plan.nvrtc_options.iter().map(String::as_str).collect(),
        Target::Metal => MSL_OPTIONS.to_vec(),
    };
    for option in options {
        write!(text, " {option}").map_err(|_| Error::SourceLimit)?;
    }
    text.push('\n');
    Ok(text)
}

fn indexes(plan: &Verified, target: Target) -> String {
    let [_, inner, cols] = plan.ir.dimensions;
    let [a, b] = &plan.ir.inputs;
    let c = &plan.ir.output;
    // MSL has 64-bit `ulong` but no `long long`; C++ and CUDA spell it out.
    let (word, suffix) = match target {
        Target::Metal => ("ulong", "ul"),
        Target::Cpu | Target::Cuda => ("unsigned long long", "ull"),
    };
    let fail = match target {
        Target::Cpu => "return 3;",
        Target::Cuda => "return;",
        // A defensive bounds failure also discards the whole dispatch.
        Target::Metal => "atomic_fetch_or_explicit(F, 1u, memory_order_relaxed); return;",
    };
    let mut text = format!(
        "    const {word} i = q / {cols}{suffix}, j = q % {cols}{suffix};\n    const {word} ci = {}{suffix} + i * {}{suffix} + j;\n    if (ci >= {}{suffix}) {{ {fail} }}\n    float sum = 0.0f;\n",
        c.offset, c.strides[0], c.elements
    );
    if inner != 0 {
        text.push_str(&format!("    for ({word} k = 0; k < {inner}{suffix}; ++k) {{\n      const {word} ai = {}{suffix} + i * {}{suffix} + k;\n      const {word} bi = {}{suffix} + k * {}{suffix} + j;\n      if (ai >= {}{suffix} || bi >= {}{suffix}) {{ {fail} }}\n", a.offset, a.strides[0], b.offset, b.strides[0], a.elements, b.elements));
        text.push_str(match target {
            Target::Cpu => "      volatile float product = A[ai] * B[bi];\n      volatile float rounded_sum = sum + product;\n      sum = rounded_sum;\n",
            Target::Cuda => "      const float product = __fmul_rn(A[ai], B[bi]);\n      sum = __fadd_rn(sum, product);\n",
            // Words carry binary32 bits; the file-scope pragma forbids
            // contraction. The guard raises the dispatch flag whenever a
            // flushing device could differ from IEEE gradual underflow.
            Target::Metal => "      const uint aw = A[ai], bw = B[bi];\n      const float product = as_type<float>(aw) * as_type<float>(bw);\n      const float next = sum + product;\n      flag |= invar_operand(aw) | invar_operand(bw) | invar_mul(aw, bw, as_type<uint>(product)) | invar_add(as_type<uint>(sum), as_type<uint>(product), as_type<uint>(next));\n      sum = next;\n",
        });
        text.push_str("    }\n");
    }
    text.push_str(if target == Target::Metal {
        "    C[ci] = as_type<uint>(sum);\n"
    } else {
        "    C[ci] = sum;\n"
    });
    text
}

pub(crate) fn finish(text: String) -> Result<String, Error> {
    if text.len() > crate::MAX_SOURCE_BYTES {
        Err(Error::SourceLimit)
    } else {
        Ok(text)
    }
}

/// Strict emitters never spell a `tc-f16-f32-v1` plan as strict code.
pub(crate) fn strict_only(plan: &Verified) -> Result<(), Error> {
    if plan.is_tc() {
        Err(Error::Numeric)
    } else {
        Ok(())
    }
}

pub fn emit_cuda(plan: &Verified) -> Result<String, Error> {
    strict_only(plan)?;
    if !matches!(plan.caps, Capabilities::Cuda { .. }) {
        return Err(Error::Capabilities);
    }
    let mut text = header(plan, Target::Cuda)?;
    text.push_str("extern \"C\" __global__ void invar_f32_contract_v1(const float* A, const float* B, float* C) {\n");
    let [rows, _, cols] = plan.ir.dimensions;
    if rows == 0 || cols == 0 {
        text.push_str("  (void)A; (void)B; (void)C;\n}\n");
    } else {
        let count = u64::from(rows) * u64::from(cols);
        text.push_str("  const unsigned long long start = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;\n  const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;\n  if (stride == 0) return;\n");
        text.push_str(&format!(
            "  for (unsigned long long q = start; q < {count}ull;) {{\n"
        ));
        text.push_str(&indexes(plan, Target::Cuda));
        text.push_str(&format!(
            "    if ({count}ull - q <= stride) break;\n    q += stride;\n  }}\n}}\n"
        ));
    }
    finish(text)
}

/// Sound underflow guard over binary32 bit patterns for a device that may
/// flush subnormal operands and results (DAZ/FTZ) but rounds normal results to
/// nearest even. Each predicate returns 1 when the device result could differ
/// from IEEE 754 gradual underflow, so the dispatch must be discarded:
/// - `invar_operand`: a subnormal input word (a DAZ device reads it as zero);
/// - `invar_mul`: nonzero factors whose device product is zero, subnormal or
///   exactly FLT_MIN (an exact product below FLT_MIN is rounded at subnormal
///   spacing by IEEE, but flushed or rounded differently by the device);
/// - `invar_add`: a nonzero sum with magnitude at most FLT_MIN, or a zero sum
///   that is not an exact cancellation (`x == -y` or both zero). Sums of
///   normals are multiples of 2^-149, so a nonzero exact sum below FLT_MIN is
///   an exact subnormal the device flushes to zero.
///
/// Unflagged products and sums have normal or exactly-zero results, for which
/// round-to-nearest-even agrees with IEEE; `guard` mirrors this argument in Rust.
pub fn guard_helpers(cuda: bool) -> String {
    let (q, u) = if cuda {
        ("__device__ static inline", "unsigned")
    } else {
        ("static inline", "uint")
    };
    format!(
        "{q} {u} invar_operand({u} w) {{ return ((w & 0x7f800000u) == 0u && (w & 0x007fffffu) != 0u) ? 1u : 0u; }}\n\
{q} {u} invar_mul({u} a, {u} b, {u} p) {{ return ((a & 0x7fffffffu) != 0u && (b & 0x7fffffffu) != 0u && (p & 0x7fffffffu) <= 0x00800000u) ? 1u : 0u; }}\n\
{q} {u} invar_add({u} x, {u} y, {u} r) {{ const {u} m = r & 0x7fffffffu; if (m != 0u) return m <= 0x00800000u ? 1u : 0u; return (((x | y) & 0x7fffffffu) == 0u || x == (y ^ 0x80000000u)) ? 0u : 1u; }}\n"
    )
}

/// Bit-for-bit Rust mirror of `guard_helpers`, used by the probe verdict,
/// property tests and evidence. Each returns true when a dispatch must be
/// discarded.
///
/// One encoding for every device-written guard, status or flag word: D1
/// probe case flags, strict-contraction and tiled dispatch flags, and the
/// rank-1 f32 status word. The host writes `UNWRITTEN` before launch. A
/// kernel claims an atomic word by a fetch-and with the word's legal mask
/// before anything else (order-independent: it never clears another
/// lane's legal bits) and sets only legal bits with fetch-or; a probe
/// stores each case flag outright. The host decodes against the word's
/// legal mask, so an untouched sentinel or any stray bit is a typed error
/// and never reads as success or "no underflow".
pub mod guard {
    /// A device word with a bit outside its legal mask, such as an untouched
    /// `UNWRITTEN` sentinel.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct InvalidWord(pub u32);

    /// Host-written before launch; outside every legal mask.
    pub const UNWRITTEN: u32 = 1 << 31;
    /// The underflow guard (or a tiled kernel's launch or bounds check)
    /// fired: discard the device result.
    pub const GUARD: u32 = 1;
    /// A checked f32 to i32 conversion was out of range.
    pub const OVERFLOW: u32 = 2;

    /// The word's bits, if every set bit is in `legal`.
    pub fn decode_bits(word: u32, legal: u32) -> Result<u32, InvalidWord> {
        if word & !legal == 0 {
            Ok(word)
        } else {
            Err(InvalidWord(word))
        }
    }

    /// A guard flag: legal values are 0 (exact) and `GUARD` (discard).
    pub fn decode(word: u32) -> Result<bool, InvalidWord> {
        decode_bits(word, GUARD).map(|w| w == GUARD)
    }

    /// MSL statement claiming status word `F` (first statement of a kernel).
    pub fn msl_claim(legal: u32) -> String {
        format!("  atomic_fetch_and_explicit(F, {legal}u, memory_order_relaxed);\n")
    }

    const MAGNITUDE: u32 = 0x7fff_ffff;
    const FLT_MIN: u32 = 0x0080_0000;
    pub fn operand(w: u32) -> bool {
        w & 0x7f80_0000 == 0 && w & 0x007f_ffff != 0
    }
    pub fn mul(a: u32, b: u32, p: u32) -> bool {
        a & MAGNITUDE != 0 && b & MAGNITUDE != 0 && p & MAGNITUDE <= FLT_MIN
    }
    pub fn add(x: u32, y: u32, r: u32) -> bool {
        let m = r & MAGNITUDE;
        if m != 0 {
            return m <= FLT_MIN;
        }
        !((x | y) & MAGNITUDE == 0 || x == y ^ 0x8000_0000)
    }
}

/// MSL 3.1 kernel over u32 words holding binary32 bits, buffers 0/1/2 = A/B/C
/// and 3 = a `guard::UNWRITTEN` dispatch flag that every lane claims first,
/// grid-strided over the logical output. A flag of `guard::GUARD` means the result may differ from strict IEEE on a
/// flushing device and must be discarded, never published. Compile with
/// `msl_options()` only.
pub fn emit_msl(plan: &Verified) -> Result<String, Error> {
    strict_only(plan)?;
    if !matches!(plan.caps, Capabilities::Metal { .. }) {
        return Err(Error::Capabilities);
    }
    let mut text = header(plan, Target::Metal)?;
    text.push_str(
        "#include <metal_stdlib>\n#pragma METAL fp contract(off)\nusing namespace metal;\n",
    );
    text.push_str(&guard_helpers(false));
    text.push_str("kernel void invar_f32_contract_v1(device const uint* A [[buffer(0)]], device const uint* B [[buffer(1)]], device uint* C [[buffer(2)]], device atomic_uint* F [[buffer(3)]], uint start [[thread_position_in_grid]], uint threads [[threads_per_grid]]) {\n");
    text.push_str(&guard::msl_claim(guard::GUARD));
    let [rows, _, cols] = plan.ir.dimensions;
    if rows == 0 || cols == 0 {
        text.push_str("  (void)A; (void)B; (void)C; (void)start; (void)threads;\n}\n");
    } else {
        let count = u64::from(rows) * u64::from(cols);
        text.push_str(
            "  const ulong stride = threads;\n  if (stride == 0) return;\n  uint flag = 0u;\n",
        );
        text.push_str(&format!("  for (ulong q = start; q < {count}ul;) {{\n"));
        text.push_str(&indexes(plan, Target::Metal));
        text.push_str(&format!(
            "    if ({count}ul - q <= stride) break;\n    q += stride;\n  }}\n  if (flag != 0u) atomic_fetch_or_explicit(F, 1u, memory_order_relaxed);\n}}\n"
        ));
    }
    finish(text)
}

/// C++17 analogue with fixed A/B/C arguments. Status 0 succeeds; 1 refuses a
/// non-nearest/FTZ/DAZ environment; 2 refuses null/overlapping writable allocations;
/// 3 is a defensive physical-bounds failure. Other pointer/fence validation
/// remains host admission. Compile with cpu_options(), never fast-math.
pub fn emit_cpu(plan: &Verified) -> Result<String, Error> {
    strict_only(plan)?;
    let mut text = header(plan, Target::Cpu)?;
    text.push_str("#include <cfenv>\n#include <cstdint>\n#include <cstring>\n#include <limits>\nstatic_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559 && std::numeric_limits<float>::digits == 24, \"binary32 required\");\nextern \"C\" int invar_f32_contract_v1(const float* A, const float* B, float* C) {\n");
    let [rows, inner, cols] = plan.ir.dimensions;
    if rows == 0 || cols == 0 {
        text.push_str("  (void)A; (void)B; (void)C; return 0;\n}\n");
        return finish(text);
    }
    text.push_str("  if (!C) return 2;\n");
    if inner != 0 {
        text.push_str("  if (!A || !B || C == A || C == B) return 2;\n  if (std::fegetround() != FE_TONEAREST) return 1;\n  std::fenv_t saved; if (std::feholdexcept(&saved) != 0) return 1;\n  struct Restore { std::fenv_t* saved; ~Restore() { std::fesetenv(saved); } } restore{&saved};\n  const std::uint32_t tiny_bits = 1; float tiny; std::memcpy(&tiny, &tiny_bits, 4);\n  volatile float input_tiny = tiny, one = 1.0f, zero = 0.0f;\n  volatile float tiny_product = input_tiny * one;\n  volatile float tiny_sum = tiny_product + zero;\n  const float rounded_tiny = tiny_sum; std::uint32_t probe; std::memcpy(&probe, &rounded_tiny, 4);\n  if (probe != 1) return 1;\n");
    } else {
        text.push_str("  (void)A; (void)B;\n");
    }
    if inner != 0 {
        let [a, b] = &plan.ir.inputs;
        let c = &plan.ir.output;
        text.push_str("  const auto ap = reinterpret_cast<std::uintptr_t>(A), bp = reinterpret_cast<std::uintptr_t>(B), cp = reinterpret_cast<std::uintptr_t>(C);\n");
        text.push_str(&format!("  if (ap > UINTPTR_MAX - {}ull || bp > UINTPTR_MAX - {}ull || cp > UINTPTR_MAX - {}ull) return 2;\n  if ((cp < ap + {}ull && ap < cp + {}ull) || (cp < bp + {}ull && bp < cp + {}ull)) return 2;\n", a.elements * 4, b.elements * 4, c.elements * 4, a.elements * 4, c.elements * 4, b.elements * 4, c.elements * 4));
    }
    let count = u64::from(rows) * u64::from(cols);
    text.push_str(&format!(
        "  for (unsigned long long q = 0; q < {count}ull; ++q) {{\n"
    ));
    text.push_str(&indexes(plan, Target::Cpu));
    text.push_str("  }\n  return 0;\n}\n");
    finish(text)
}
