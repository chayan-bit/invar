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
    w.push(invar_device::EXACT_F32)?;
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
    let trig = calls
        .iter()
        .any(|n| matches!(crate::math::f32_trig_base(n), "sin" | "cos"));
    let root = calls.contains(&"sqrt");
    if trig || root {
        w.push(match p.target {
            Target::Cpu => "#define GNET_TRIG_CONSTANT static const",
            Target::Cuda => "#define GNET_TRIG_CONSTANT __device__ static const",
            Target::Metal => "#define GNET_TRIG_CONSTANT constant",
        })?;
        w.push(&invar_device::trig_f32::helpers(
            invar_device::trig_f32::Lang::Gnet,
            trig,
            root,
        ))?;
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
                crate::math::f32_trig_base(name),
                "fma" | "f32_from_i32" | "f32_from_u32" | "sin" | "cos" | "sqrt"
            ) =>
        {
            let name = crate::math::f32_trig_base(name);
            let helper = match name {
                "fma" => "gnet_f32_fma",
                "f32_from_i32" => "gnet_f32_from_i32",
                "f32_from_u32" => "gnet_f32_from_u32",
                "sin" | "cos" => invar_device::trig_f32::SIN_HELPER,
                "sqrt" => invar_device::trig_f32::SQRT_HELPER,
                _ => return Err(Error::UnsupportedCall),
            };
            match args.as_slice() {
                [a] if matches!(name, "sin" | "cos") => format!(
                    "{helper}((unsigned){},{}u)",
                    value(p, f, *a)?,
                    u8::from(name == "cos")
                ),
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
            if !line.starts_with("GNET_FN ")
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
