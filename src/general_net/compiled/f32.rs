//! Exact binary32 primitives over bounded unsigned integer storage, for CPU and
//! the compiled numeric profiles. No host or device float,
//! floating environment, heap BigInt or executable net interpreter.
use super::{
    Error, Program, Target,
    emit::{Writer, value},
};
use crate::{
    front::{BinaryOp, Type, UnaryOp},
    mir,
};

/// Correctly rounded binary32 `sqrt` (`invar-math-cr-v1`) over words, in
/// exact integer arithmetic: no host or device float. A subnormal operand is
/// scaled exactly by 2^150 into an integer whose root rounds once and scales
/// back exactly by 2^-75; NaN results are the canonical quiet NaN.
const SQRT: &str = r#"
static_assert(sizeof(unsigned)==4,"GNET sqrt words require32 bits");
GNET_FN unsigned gnet_f32_root(unsigned a) {
  if (a == 0u || a == 2139095040u) return a;
  int e = (int)(a >> 23) - 127;
  gnet_u s = (a & 8388607u) | 8388608u;
  if (e % 2 != 0) { s <<= 1; --e; }
  const gnet_u n = s << 23;
  gnet_u q = 0;
  for (int bit = 23; bit >= 0; --bit) {
    const gnet_u t = q | (1ULL << bit);
    if (t*t <= n) q = t;
  }
  if (n-q*q > q) ++q;
  return (unsigned)gnet_f32_pack(q, e/2, 0);
}
GNET_FN unsigned gnet_f32_sqrt(unsigned x) {
  const unsigned a = x & 2147483647u;
  if (a > 2139095040u || ((x >> 31) != 0u && a != 0u)) return 2143289344u;
  if (a == 0u) return x;
  if (a < 8388608u) return (unsigned)gnet_f32_mul(gnet_f32_root((unsigned)gnet_f32_from_u32(a << 1)), 436207616u);
  return gnet_f32_root(x);
}
"#;

/// CUDA keeps the soft-float arithmetic entry points out of line: inlined at
/// each of hundreds of call sites they dominate NVRTC time (#278). The
/// arithmetic is exact integer code, so where it is compiled changes no bit.
/// A host compiler (the CPU model of CUDA sources) still inlines them.
const OUTLINED: [&str; 4] = ["add", "mul", "fma", "div"];
const CUDA_CALL: &str = r#"
#ifdef __CUDACC__
#define GNET_CALL __device__ __noinline__
#else
#define GNET_CALL GNET_FN
#endif"#;

fn outlined(text: &str) -> String {
    OUTLINED.iter().fold(text.to_owned(), |text, op| {
        let head = format!(" gnet_u gnet_f32_{op}(gnet_u a");
        text.replacen(&format!("GNET_FN{head}"), &format!("GNET_CALL{head}"), 1)
    })
}

pub(super) fn preamble(w: &mut Writer, p: &Program<'_>) -> Result<(), Error> {
    if p.numeric.is_none() {
        return Ok(());
    }
    // NVRTC supplies no standard headers by default: the CUDA assertion is
    // header-free. The CPU text is unchanged byte for byte.
    w.push(if p.target != Target::Cpu {
        r#"
static_assert(sizeof(gnet_u)==8 && sizeof(gnet_w)==8 && (gnet_u)0-1ULL==18446744073709551615ULL,"GNET exact numeric words require64 bits");"#
    } else {
        r#"
#include <climits>
static_assert(CHAR_BIT==8 && sizeof(gnet_u)==8 && sizeof(gnet_w)==8 && ULLONG_MAX==18446744073709551615ULL,"GNET exact numeric words require64 bits");"#
    })?;
    if p.target == Target::Cuda {
        w.push(CUDA_CALL)?;
        w.push(&outlined(invar_device::EXACT_F32))?;
    } else {
        w.push(invar_device::EXACT_F32)?;
    }
    let calls = p
        .mir
        .functions
        .iter()
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.instructions)
        .filter_map(|i| match &i.kind {
            mir::Operation::Call { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    if calls.contains(&"sqrt") {
        w.push(SQRT)?;
    }
    Ok(())
}

pub(super) fn operation(
    w: &mut Writer,
    p: &Program<'_>,
    f: usize,
    i: &mir::Instruction,
) -> Result<bool, Error> {
    if p.numeric.is_none() {
        return Ok(false);
    }
    use mir::Operation as O;
    let expr = match &i.kind {
        O::F32(x) => format!("{}ULL", x.to_bits()),
        O::Unary {
            op: UnaryOp::Neg,
            value: id,
            ..
        } if p.ty(f, *id)? == &Type::F32 => format!("(gnet_u){}^2147483648ULL", value(p, f, *id)?),
        O::Binary {
            op, left, right, ..
        } if p.ty(f, *left)? == &Type::F32 => {
            let a = value(p, f, *left)?;
            let b = value(p, f, *right)?;
            match op {
                BinaryOp::Add => format!("gnet_f32_add({a},{b})"),
                BinaryOp::Sub => format!("gnet_f32_sub({a},{b})"),
                BinaryOp::Mul => format!("gnet_f32_mul({a},{b})"),
                BinaryOp::Div => format!("gnet_f32_div({a},{b})"),
                BinaryOp::Eq => format!("gnet_f32_eq({a},{b})"),
                BinaryOp::Ne => format!("!gnet_f32_eq({a},{b})"),
                BinaryOp::Lt => format!("gnet_f32_lt({a},{b})"),
                BinaryOp::Le => format!("(gnet_f32_lt({a},{b})||gnet_f32_eq({a},{b}))"),
                BinaryOp::Gt => format!("gnet_f32_lt({b},{a})"),
                BinaryOp::Ge => format!("(gnet_f32_lt({b},{a})||gnet_f32_eq({a},{b}))"),
                _ => {
                    return Err(Error::UnsupportedRule(
                        super::coverage::operation(&i.kind).0,
                    ));
                }
            }
        }
        O::Call { name, args }
            if matches!(
                name.as_str(),
                "fma" | "f32_from_i32" | "f32_from_u32" | "sqrt"
            ) =>
        {
            let name = name.as_str();
            let helper = match name {
                "fma" => "gnet_f32_fma",
                "f32_from_i32" => "gnet_f32_from_i32",
                "f32_from_u32" => "gnet_f32_from_u32",
                "sqrt" => "gnet_f32_sqrt",
                _ => return Err(Error::UnsupportedCall),
            };
            match args.as_slice() {
                [a] if name == "sqrt" => format!("{helper}((unsigned){})", value(p, f, *a)?),
                [a] => format!("{helper}({})", value(p, f, *a)?),
                [a, b, c] if name == "fma" => format!(
                    "{helper}({},{},{})",
                    value(p, f, *a)?,
                    value(p, f, *b)?,
                    value(p, f, *c)?
                ),
                _ => return Err(Error::State),
            }
        }
        _ => return Ok(false),
    };
    let (id, _) = i.output.as_ref().ok_or(Error::State)?;
    w.push(&format!("{}=(gnet_w)({expr});", value(p, f, *id)?))?;
    Ok(true)
}

/// The numeric template's functions are deliberately single-line definitions.
/// Retain their transitive call closure from the emitted program, rather than
/// sending every exact numeric primitive to each target compiler.
pub(super) fn trim_unused(source: String) -> String {
    use std::collections::{BTreeMap, BTreeSet};
    let helpers: BTreeMap<_, _> = source
        .lines()
        .filter_map(|line| {
            if !(line.starts_with("GNET_FN ") || line.starts_with("GNET_CALL "))
                || !line.contains(" gnet_f32_")
                || !line.ends_with(" }")
            {
                return None;
            }
            let name = line.split_once('(')?.0.split_whitespace().last()?;
            Some((name, line))
        })
        .collect();
    let identifiers = |line: &str| -> Vec<String> {
        line.split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .filter(|word| helpers.contains_key(word))
            .map(str::to_owned)
            .collect()
    };
    let mut used = BTreeSet::new();
    let mut pending: Vec<_> = source
        .lines()
        .filter(|line| !helpers.values().any(|body| body == line))
        .flat_map(identifiers)
        .collect();
    while let Some(name) = pending.pop() {
        if used.insert(name.clone()) {
            pending.extend(identifiers(helpers[name.as_str()]));
        }
    }
    let mut result = String::with_capacity(source.len());
    for line in source.lines() {
        if let Some((name, _)) = helpers.iter().find(|(_, body)| **body == line)
            && !used.contains(*name)
        {
            continue;
        }
        result.push_str(line);
        result.push('\n');
    }
    result
}

#[cfg(test)]
mod tests {
    #[test]
    fn cuda_outlines_every_soft_float_entry_point_once() {
        let text = super::outlined(invar_device::EXACT_F32);
        for op in super::OUTLINED {
            let head = format!(" gnet_u gnet_f32_{op}(gnet_u a");
            assert_eq!(text.matches(&format!("GNET_CALL{head}")).count(), 1, "{op}");
            assert!(!text.contains(&format!("GNET_FN{head}")), "{op}");
        }
        assert_eq!(text.matches("GNET_CALL ").count(), super::OUTLINED.len());
    }
}
