//! Scalar strict-f32-v1 and explicit TC step semantics. Control, demand, fuel and spans belong to the
//! net machine; the frozen exact bit oracle supplies only primitive arithmetic.
use super::{OracleError, Scalar, internal};
use crate::{
    front::BinaryOp,
    numeric_contract::{Comparison, F32Bits},
};

pub(super) fn supports_intrinsic(name: &str) -> bool {
    matches!(
        name,
        "fma" | "tc_mul_add" | "sin" | "cos" | "sqrt" | "f32_from_i32" | "f32_from_u32"
    )
}
pub(super) fn call(
    name: &str,
    args: &[crate::mir::ValueId],
    frame: &super::Frame,
) -> Result<Scalar, OracleError> {
    Ok(match (name, args) {
        ("fma" | "tc_mul_add", [a, b, c]) => {
            let (Scalar::F32(a), Scalar::F32(b), Scalar::F32(c)) =
                (frame.get(*a)?, frame.get(*b)?, frame.get(*c)?)
            else {
                return Err(internal());
            };
            Scalar::F32(if name == "fma" {
                a.fma(b, c)
            } else {
                let round = |x: F32Bits| {
                    invar_device::tc::f16_round(x.to_bits())
                        .map(F32Bits::from_bits)
                        .ok_or_else(|| {
                            super::semantic_tool(
                                "TC_INPUT_RANGE",
                                "tc-f16-f32-v1 requires finite binary16 inputs",
                            )
                        })
                };
                round(a)?.mul(round(b)?).add(c)
            })
        }
        ("sin" | "cos" | "sqrt", [a]) => {
            let Scalar::F32(a) = frame.get(*a)? else {
                return Err(internal());
            };
            let f = crate::math::MathFn::from_name(name).ok_or_else(internal)?;
            Scalar::F32(crate::math::eval_f32(f, &[a]))
        }
        ("f32_from_i32", [a]) => {
            let Scalar::I32(a) = frame.get(*a)? else {
                return Err(internal());
            };
            Scalar::F32(F32Bits::from_i32(a))
        }
        ("f32_from_u32", [a]) => {
            let Scalar::U32(a) = frame.get(*a)? else {
                return Err(internal());
            };
            Scalar::F32(F32Bits::from_u32(a))
        }
        _ => return Err(internal()),
    })
}
pub(super) fn binary(op: BinaryOp, a: F32Bits, b: F32Bits) -> Result<Scalar, OracleError> {
    use BinaryOp as O;
    Ok(match op {
        O::Add => Scalar::F32(a.add(b)),
        O::Sub => Scalar::F32(a.sub(b)),
        O::Mul => Scalar::F32(a.mul(b)),
        O::Div => Scalar::F32(a.div(b)),
        O::Eq => Scalar::Bool(a.compare(Comparison::Eq, b)),
        O::Ne => Scalar::Bool(a.compare(Comparison::Ne, b)),
        O::Lt => Scalar::Bool(a.compare(Comparison::Lt, b)),
        O::Le => Scalar::Bool(a.compare(Comparison::Le, b)),
        O::Gt => Scalar::Bool(a.compare(Comparison::Gt, b)),
        O::Ge => Scalar::Bool(a.compare(Comparison::Ge, b)),
        _ => return Err(internal()),
    })
}

pub(super) fn argument(
    ty: &crate::front::Type,
    arg: &crate::eval::Value,
) -> Result<Scalar, OracleError> {
    use crate::{eval::Value, front::Type};
    match (ty, arg) {
        (Type::F32, Value::F32(x)) => Ok(Scalar::F32(*x)),
        (Type::I32, Value::I32(x)) => Ok(Scalar::I32(*x)),
        (Type::U32, Value::U32(x)) => Ok(Scalar::U32(*x)),
        (Type::Bool, Value::Bool(x)) => Ok(Scalar::Bool(*x)),
        _ => Err(super::semantic_tool(
            "ENTRYPOINT_SIGNATURE",
            "oracle external entry requires matching scalar arguments",
        )),
    }
}
