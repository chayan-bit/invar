//! Strict f32 maps and source-order folds through the canonical device IR v2.
//! The fixed two-input runtime ABI packs logical inputs as planes in buffer 0.
use crate::{Capabilities, floating, v2};
use std::fmt::Write;

pub const MAX_LENGTH: u32 = 1_000_000;
pub const MAX_INPUTS: usize = 4;
pub const MAX_NODES: usize = 64;
pub const KERNEL_NAME: &str = "invar_f32_ops_v2";
/// Legal bits of the status word (`floating::guard` encoding): CUDA appends it
/// to C and can only refuse a conversion; Metal binds it as F and also carries
/// the underflow guard.
pub const CUDA_STATUS: u32 = floating::guard::OVERFLOW;
pub const METAL_STATUS: u32 = floating::guard::GUARD | floating::guard::OVERFLOW;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scalar {
    F32,
    U32,
    I32,
    Bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compare {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Node {
    Input(usize),
    Constant(u32, Scalar),
    Neg(usize),
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Fma(usize, usize, usize),
    Compare(Compare, usize, usize),
    FromU32(usize),
    FromI32(usize),
    TruncI32(usize),
    /// Correctly rounded `sqrt` (`invar-math-cr-v1`), see [`sqrt_helper`].
    Sqrt(usize),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Map,
    Reduce { initial: u32 },
    Scan { initial: u32 },
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub length: u32,
    pub inputs: Vec<Scalar>,
    pub nodes: Vec<Node>,
    pub kind: Kind,
}
impl Program {
    pub fn output_len(&self) -> usize {
        if matches!(self.kind, Kind::Reduce { .. }) {
            1
        } else {
            self.length as usize
        }
    }
    pub fn packed_len(&self) -> usize {
        self.length as usize * self.inputs.len()
    }
    pub fn output_type(&self) -> Result<Scalar, v2::Error> {
        self.types()?.last().copied().ok_or(v2::Error::Numeric)
    }
    fn types(&self) -> Result<Vec<Scalar>, v2::Error> {
        if self.inputs.is_empty()
            || self.inputs.len() > MAX_INPUTS
            || self.nodes.is_empty()
            || self.nodes.len() > MAX_NODES
            || self.length > MAX_LENGTH
        {
            return Err(v2::Error::Bounds);
        }
        let mut types = Vec::new();
        for node in &self.nodes {
            let get = |i: usize| types.get(i).copied().ok_or(v2::Error::Numeric);
            let float = |i| {
                if get(i)? == Scalar::F32 {
                    Ok(())
                } else {
                    Err(v2::Error::Numeric)
                }
            };
            let ty = match *node {
                Node::Input(slot) => *self.inputs.get(slot).ok_or(v2::Error::Bounds)?,
                Node::Constant(_, ty) => ty,
                Node::Neg(a) | Node::Sqrt(a) => {
                    float(a)?;
                    Scalar::F32
                }
                Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => {
                    float(a)?;
                    float(b)?;
                    Scalar::F32
                }
                Node::Fma(a, b, c) => {
                    float(a)?;
                    float(b)?;
                    float(c)?;
                    Scalar::F32
                }
                Node::Compare(_, a, b) => {
                    float(a)?;
                    float(b)?;
                    Scalar::Bool
                }
                Node::FromU32(a) => {
                    if get(a)? != Scalar::U32 {
                        return Err(v2::Error::Numeric);
                    }
                    Scalar::F32
                }
                Node::FromI32(a) => {
                    if get(a)? != Scalar::I32 {
                        return Err(v2::Error::Numeric);
                    }
                    Scalar::F32
                }
                Node::TruncI32(a) => {
                    float(a)?;
                    Scalar::I32
                }
            };
            types.push(ty);
        }
        if !matches!(self.kind, Kind::Map) && types.last() != Some(&Scalar::F32) {
            return Err(v2::Error::Numeric);
        }
        Ok(types)
    }
}

pub fn descriptor(
    program: Program,
    span: crate::Span,
    origin: crate::Provenance,
) -> Result<v2::Ir, v2::Error> {
    let output = program.output_type()?;
    let n = u64::from(program.length);
    let planes = program.inputs.len() as u64;
    let buffer = |id, element, shape, strides, bytes, writable| v2::Buffer {
        id,
        element,
        allocation_bytes: bytes,
        alignment: 4,
        writable,
        layout: v2::Layout::Strided {
            shape,
            strides_in_elements: strides,
            offset: 0,
        },
    };
    Ok(v2::Ir {
        version: v2::VERSION,
        layout_version: v2::LAYOUT_VERSION,
        numeric: v2::NumericPolicy::STRICT_F32,
        span,
        origin,
        buffers: vec![
            buffer(
                0,
                v2::ElemType::U32,
                vec![planes, n],
                vec![n, 1],
                n * planes * 4,
                false,
            ),
            buffer(1, v2::ElemType::U32, vec![0], vec![1], 0, false),
            buffer(
                2,
                if output == Scalar::F32 {
                    v2::ElemType::F32
                } else {
                    v2::ElemType::U32
                },
                vec![program.output_len() as u64],
                vec![1],
                (program.output_len() as u64 + 1) * 4,
                true,
            ),
        ],
        kernel: v2::Kernel::F32(program),
    })
}
pub(crate) fn validate(ir: &v2::Ir, p: &Program) -> Result<(), v2::Error> {
    let expected = descriptor(p.clone(), ir.span, ir.origin)?;
    if ir.numeric != expected.numeric || ir.buffers != expected.buffers {
        return Err(v2::Error::Bounds);
    }
    Ok(())
}
pub fn program(plan: &v2::Verified) -> Result<&Program, v2::Error> {
    match &plan.ir().kernel {
        v2::Kernel::F32(p) => Ok(p),
        _ => Err(v2::Error::Numeric),
    }
}
pub fn probe_ops(p: &Program) -> Vec<crate::f32_probe::Op> {
    use crate::f32_probe::Op;
    let mut ops = vec![Op::Accumulate]; // Always test the non-contraction pragma.
    for node in &p.nodes {
        let op = match node {
            Node::Add(..) => Some(Op::Add),
            Node::Sub(..) => Some(Op::Sub),
            Node::Mul(..) => Some(Op::Mul),
            Node::Fma(..) => Some(Op::Fma),
            Node::FromU32(..) => Some(Op::FromU32),
            Node::FromI32(..) => Some(Op::FromI32),
            _ => None, // Integer implementations need no floating probe.
        };
        if let Some(op) = op
            && !ops.contains(&op)
        {
            ops.push(op);
        }
    }
    if !matches!(p.kind, Kind::Map) && !ops.contains(&Op::Add) {
        ops.push(Op::Add);
    }
    ops
}

/// Device admission for the `sqrt` helper of `p`: it reads a subnormal only
/// through exact integer paths, so every probe mismatch of the operations it
/// uses (`Mul`, `FromU32`, `Sqrt`) must depend on gradual underflow. A program
/// without `sqrt` is admitted trivially.
pub fn admits_sqrt(p: &Program, report: &crate::f32_probe::Report) -> bool {
    use crate::f32_probe::Op;
    !p.nodes.iter().any(|n| matches!(n, Node::Sqrt(_)))
        || report.mismatches().iter().all(|&i| {
            let case = &crate::f32_probe::CASES[i];
            !matches!(case.op, Op::Mul | Op::FromU32 | Op::Sqrt) || case.subnormal
        })
}

/// Correctly rounded binary32 `sqrt` over words, `invar_sqrt_f32(x)`. A
/// subnormal operand is scaled exactly by 2^150 into an integer, whose root
/// rounds once and scales back exactly by 2^-75, so flushing never changes
/// the bits; NaN results are the canonical quiet NaN.
pub fn sqrt_helper(cuda: bool) -> String {
    let (q, u, root, twice, product) = if cuda {
        (
            "__device__ static inline",
            "unsigned",
            "__float_as_uint(__fsqrt_rn(__uint_as_float(x)))",
            "__fsqrt_rn(__uint2float_rn(a << 1))",
            "__float_as_uint(__fmul_rn(r, __uint_as_float(0x1a000000u)))",
        )
    } else {
        (
            "static inline",
            "uint",
            "as_type<uint>(sqrt(as_type<float>(x)))",
            "sqrt(float(a << 1))",
            "as_type<uint>(r * as_type<float>(0x1a000000u))",
        )
    };
    format!(
        "{q} {u} invar_sqrt_f32({u} x) {{ const {u} a = x & 0x7fffffffu; \
if (a > 0x7f800000u || ((x >> 31) != 0u && a != 0u)) return 0x7fc00000u; \
if (a == 0u) return x; \
if (a < 0x00800000u) {{ const float r = {twice}; return {product}; }} \
return {root}; }}\n"
    )
}

/// Integer-only IEEE comparisons and checked truncation cannot read a
/// subnormal float, so DAZ/FTZ cannot affect either operation.
pub fn word_helpers(cuda: bool) -> String {
    let q = if cuda {
        "__device__ static inline"
    } else {
        "static inline"
    };
    let u = if cuda { "unsigned" } else { "uint" };
    format!(
        "{q} {u} invar_nan({u} a) {{ return (a & 0x7fffffffu) > 0x7f800000u; }}\n\
{q} {u} invar_key({u} a) {{ return (a & 0x80000000u) ? ~a : (a | 0x80000000u); }}\n\
{q} {u} invar_eq({u} a, {u} b) {{ return !invar_nan(a) && !invar_nan(b) && (a == b || ((a | b) & 0x7fffffffu) == 0u); }}\n\
{q} {u} invar_lt({u} a, {u} b) {{ return !invar_nan(a) && !invar_nan(b) && !invar_eq(a,b) && invar_key(a) < invar_key(b); }}\n\
{q} {u} invar_trunc({u} a) {{ const {u} m = a & 0x7fffffffu; if (m < 0x3f800000u) return 0u; const {u} e = (m >> 23) - 127u; const {u} s = (m & 0x007fffffu) | 0x00800000u; const {u} v = e >= 23u ? s << (e - 23u) : s >> (23u - e); return (a & 0x80000000u) ? 0u - v : v; }}\n"
    )
}

pub fn emit(plan: &v2::Verified) -> Result<String, v2::Error> {
    let p = program(plan)?;
    let cuda = matches!(plan.capabilities(), Capabilities::Cuda { .. });
    let u = if cuda { "unsigned" } else { "uint" };
    let bits = |x: &str| {
        if cuda {
            format!("__float_as_uint({x})")
        } else {
            format!("as_type<uint>({x})")
        }
    };
    let float = |i| {
        if cuda {
            format!("__uint_as_float(v{i})")
        } else {
            format!("as_type<float>(v{i})")
        }
    };
    // The existing bounded NVRTC child uses the strict-f32-v1 compiler mode.
    let mut s = format!(
        "// invar-f32-contract-v1 numeric=strict-f32-v1 numeric-version=1 layout-version=1\n// caps={:?}\n// device-ir-version=2 kernel={:?}\n",
        plan.capabilities(),
        p
    );
    if cuda {
        let opts =
            floating::cuda_strict_options(plan.capabilities()).ok_or(v2::Error::Capability)?;
        writeln!(s, "// required-options: {}", opts.join(" ")).map_err(|_| v2::Error::Bounds)?;
    } else {
        s.push_str(
            "#include <metal_stdlib>\n#pragma METAL fp contract(off)\nusing namespace metal;\n",
        );
    }
    s.push_str(&floating::guard_helpers(cuda));
    s.push_str(&word_helpers(cuda));
    if p.nodes.iter().any(|n| matches!(n, Node::Sqrt(_))) {
        s.push_str(&sqrt_helper(cuda));
    }
    if cuda {
        writeln!(s, "extern \"C\" __global__ void {KERNEL_NAME}(const unsigned* A, const unsigned* B, unsigned* C) {{ unsigned start = blockIdx.x * blockDim.x + threadIdx.x; unsigned threads = gridDim.x * blockDim.x; unsigned flag = 0u;").map_err(|_| v2::Error::Bounds)?;
    } else {
        writeln!(s, "kernel void {KERNEL_NAME}(device const uint* A [[buffer(0)]], device const uint* B [[buffer(1)]], device uint* C [[buffer(2)]], device atomic_uint* F [[buffer(3)]], uint start [[thread_position_in_grid]], uint threads [[threads_per_grid]]) {{ uint flag = 0u;").map_err(|_| v2::Error::Bounds)?;
    }
    // Claim the status word before any early return (see `floating::guard`).
    if cuda {
        writeln!(s, "  atomicAnd(C + {}u, {CUDA_STATUS}u);", p.output_len())
            .map_err(|_| v2::Error::Bounds)?;
    } else {
        s.push_str(&floating::guard::msl_claim(METAL_STATUS));
    }
    let (begin, step) = if matches!(p.kind, Kind::Map) {
        ("start", "threads")
    } else {
        s.push_str("  if (start != 0u) return;\n");
        let initial = match p.kind {
            Kind::Reduce { initial } | Kind::Scan { initial } => initial,
            _ => 0,
        };
        writeln!(s, "  {u} sum = {initial}u;").map_err(|_| v2::Error::Bounds)?;
        ("0u", "1u")
    };
    writeln!(
        s,
        "  for ({u} i = {begin}; i < {}u; i += {step}) {{",
        p.length
    )
    .map_err(|_| v2::Error::Bounds)?;
    for (i, node) in p.nodes.iter().enumerate() {
        let mut guard = None;
        let expr = match *node {
            Node::Input(slot) => format!("A[{}u + i]", slot * p.length as usize),
            Node::Constant(w, _) => format!("{w}u"),
            Node::Neg(a) => format!("v{a} ^ 0x80000000u"),
            Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => {
                let sub = matches!(node, Node::Sub(..));
                let mul = matches!(node, Node::Mul(..));
                let right = if sub {
                    if cuda {
                        format!("__uint_as_float(v{b} ^ 0x80000000u)")
                    } else {
                        format!("as_type<float>(v{b} ^ 0x80000000u)")
                    }
                } else {
                    float(b)
                };
                guard = Some(format!(
                    "invar_operand(v{a}) | invar_operand(v{b}) | invar_{}(v{a}, {}, v{i})",
                    if mul { "mul" } else { "add" },
                    if sub {
                        format!("v{b} ^ 0x80000000u")
                    } else {
                        format!("v{b}")
                    }
                ));
                let op = if mul { "*" } else { "+" };
                let calc = if cuda {
                    format!(
                        "{}({}, {right})",
                        if mul { "__fmul_rn" } else { "__fadd_rn" },
                        float(a)
                    )
                } else {
                    format!("{} {op} {right}", float(a))
                };
                bits(&calc)
            }
            Node::Fma(a, b, c) => {
                // ponytail: conservatively discard every Metal fma dispatch;
                // add a narrower guard only with an independent soundness proof.
                guard = Some("1u".to_owned());
                bits(&format!(
                    "{}({}, {}, {})",
                    if cuda { "__fmaf_rn" } else { "fma" },
                    float(a),
                    float(b),
                    float(c)
                ))
            }
            Node::Compare(op, a, b) => match op {
                Compare::Eq => format!("invar_eq(v{a},v{b})"),
                Compare::Ne => format!("!invar_eq(v{a},v{b})"),
                Compare::Lt => format!("invar_lt(v{a},v{b})"),
                Compare::Gt => format!("invar_lt(v{b},v{a})"),
                Compare::Le => format!("invar_lt(v{a},v{b}) | invar_eq(v{a},v{b})"),
                Compare::Ge => format!("invar_lt(v{b},v{a}) | invar_eq(v{a},v{b})"),
            },
            // No dispatch guard: the helper never reads a subnormal float (see
            // `sqrt_helper`). Later nodes still guard their own operands.
            Node::Sqrt(a) => format!("invar_sqrt_f32(v{a})"),
            Node::FromU32(a) => bits(&if cuda {
                format!("__uint2float_rn(v{a})")
            } else {
                format!("float(v{a})")
            }),
            Node::FromI32(a) => bits(&if cuda {
                format!("__int2float_rn(__float_as_int(__uint_as_float(v{a})))")
            } else {
                format!("float(as_type<int>(v{a}))")
            }),
            Node::TruncI32(a) => {
                // Reject before any shift or cast. INT_MAX rounds upward as f32,
                // so the positive bound is exclusive; INT_MIN is representable.
                writeln!(s, "    if ((v{a} & 0x7fffffffu) > 0x4f000000u || ((v{a} & 0x7fffffffu) == 0x4f000000u && (v{a} & 0x80000000u) == 0u)) {{ {} }}", if cuda { format!("atomicOr(C + {}u, {}u); return;", p.output_len(), floating::guard::OVERFLOW) } else { format!("atomic_fetch_or_explicit(F, {}u, memory_order_relaxed); return;", floating::guard::OVERFLOW) }).map_err(|_| v2::Error::Bounds)?;
                format!("invar_trunc(v{a})")
            }
        };
        writeln!(s, "    {u} v{i} = {expr};").map_err(|_| v2::Error::Bounds)?;
        if !cuda && let Some(guard) = guard {
            writeln!(s, "    flag |= {guard};").map_err(|_| v2::Error::Bounds)?;
        }
    }
    let last = p.nodes.len() - 1;
    if matches!(p.kind, Kind::Map) {
        writeln!(s, "    C[i] = v{last};").map_err(|_| v2::Error::Bounds)?;
    } else {
        writeln!(
            s,
            "    {u} previous = sum; sum = {};",
            bits(&if cuda {
                format!(
                    "__fadd_rn({}, {})",
                    if cuda {
                        "__uint_as_float(sum)"
                    } else {
                        "as_type<float>(sum)"
                    },
                    float(last)
                )
            } else {
                format!("as_type<float>(sum) + {}", float(last))
            })
        )
        .map_err(|_| v2::Error::Bounds)?;
        if !cuda {
            writeln!(s, "    flag |= invar_operand(previous) | invar_operand(v{last}) | invar_add(previous,v{last},sum);").map_err(|_| v2::Error::Bounds)?;
        }
        if matches!(p.kind, Kind::Scan { .. }) {
            s.push_str("    C[i] = sum;\n");
        }
    }
    s.push_str("  }\n");
    if matches!(p.kind, Kind::Reduce { .. }) {
        s.push_str("  C[0] = sum;\n");
    }
    if !cuda {
        writeln!(
            s,
            "  if (flag != 0u) atomic_fetch_or_explicit(F, {}u, memory_order_relaxed);",
            floating::guard::GUARD
        )
        .map_err(|_| v2::Error::Bounds)?;
    }
    s.push_str("}\n");
    if s.len() > crate::MAX_SOURCE_BYTES {
        return Err(v2::Error::Bounds);
    }
    Ok(s)
}
