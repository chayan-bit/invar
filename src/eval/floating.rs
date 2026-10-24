//! Strict binary32 and binary64 source operations, conversions and the
//! `invar-math-cr-v1` library; demand and fuel belong to each evaluator.
use super::{EvalError, MAX_ARRAY_LEN, Value, admit_live};
use crate::{
    front::{BinaryOp, Span, Type},
    math::MathFn,
    numeric_contract::{
        Comparison, F32Bits, F64Bits, NumericTrap,
        storage::{Bf16Bits, F16Bits, narrow_i8, narrow_u8},
    },
};
use std::{cell::RefCell, rc::Rc};

fn comparison(op: BinaryOp) -> Option<Comparison> {
    use BinaryOp as O;
    Some(match op {
        O::Eq => Comparison::Eq,
        O::Ne => Comparison::Ne,
        O::Lt => Comparison::Lt,
        O::Le => Comparison::Le,
        O::Gt => Comparison::Gt,
        O::Ge => Comparison::Ge,
        _ => return None,
    })
}

fn invalid_operation() -> EvalError {
    EvalError::tool("INTERNAL_TYPE", "invalid checked float operation")
}

pub(crate) fn binary(a: F32Bits, op: BinaryOp, b: F32Bits) -> Result<Value, EvalError> {
    if let Some(predicate) = comparison(op) {
        return Ok(Value::Bool(a.compare(predicate, b)));
    }
    Ok(Value::F32(match op {
        BinaryOp::Add => a.add(b),
        BinaryOp::Sub => a.sub(b),
        BinaryOp::Mul => a.mul(b),
        BinaryOp::Div => a.div(b),
        _ => return Err(invalid_operation()),
    }))
}

pub(crate) fn binary64(a: F64Bits, op: BinaryOp, b: F64Bits) -> Result<Value, EvalError> {
    if let Some(predicate) = comparison(op) {
        return Ok(Value::Bool(a.compare(predicate, b)));
    }
    Ok(Value::F64(match op {
        BinaryOp::Add => a.add(b),
        BinaryOp::Sub => a.sub(b),
        BinaryOp::Mul => a.mul(b),
        BinaryOp::Div => a.div(b),
        _ => return Err(invalid_operation()),
    }))
}

/// The checked literal `raw` at its checked type (`f32` unless typed `f64`).
pub(crate) fn literal(raw: &str, ty: Option<&Type>) -> Result<Value, EvalError> {
    let invalid = |_| EvalError::tool("INTERNAL_LITERAL", "invalid checked float literal");
    match ty {
        Some(Type::F64) => F64Bits::from_decimal(raw).map(Value::F64).map_err(invalid),
        _ => F32Bits::from_decimal(raw).map(Value::F32).map_err(invalid),
    }
}

/// Whether `name` is a float intrinsic this module evaluates.
pub(crate) fn handles(name: &str) -> bool {
    matches!(name, "filled_f32" | "filled_f64" | "fma" | "tc_mul_add")
        || crate::check::conversion(name).is_some()
        || MathFn::from_name(name).is_some()
}

/// Evaluate a float intrinsic. A float-to-integer conversion that is NaN or
/// out of range after truncation traps `INTEGER_OVERFLOW` at `span`.
pub(crate) fn intrinsic(
    name: &str,
    values: &[Value],
    live: &mut usize,
    span: Span,
) -> Result<Value, EvalError> {
    let trap = |_: NumericTrap| EvalError::Trap {
        category: crate::eval::TrapCategory::IntegerOverflow,
        span,
    };
    if let Some(f) = MathFn::from_name(name) {
        return math(f, values);
    }
    match (name, values) {
        ("tc_mul_add", [Value::F32(a), Value::F32(b), Value::F32(c)]) => {
            let round = |x: F32Bits| {
                invar_device::tc::f16_round(x.to_bits())
                    .map(F32Bits::from_bits)
                    .ok_or_else(|| {
                        EvalError::tool(
                            "TC_INPUT_RANGE",
                            "tc-f16-f32-v1 requires finite binary16 inputs",
                        )
                    })
            };
            Ok(Value::F32(round(*a)?.mul(round(*b)?).add(*c)))
        }
        ("fma", [Value::F32(a), Value::F32(b), Value::F32(c)]) => Ok(Value::F32(a.fma(*b, *c))),
        ("fma", [Value::F64(a), Value::F64(b), Value::F64(c)]) => Ok(Value::F64(a.fma(*b, *c))),
        ("f32_from_i32", [Value::I32(a)]) => Ok(Value::F32(F32Bits::from_i32(*a))),
        ("f32_from_u32", [Value::U32(a)]) => Ok(Value::F32(F32Bits::from_u32(*a))),
        ("f64_from_i32", [Value::I32(a)]) => Ok(Value::F64(F64Bits::from_i32(*a))),
        ("f64_from_u32", [Value::U32(a)]) => Ok(Value::F64(F64Bits::from_u32(*a))),
        ("f64_from_f32", [Value::F32(a)]) => Ok(Value::F64(F64Bits::from_f32(*a))),
        ("f32_from_f64", [Value::F64(a)]) => Ok(Value::F32(a.to_f32())),
        ("i32_from_f32", [Value::F32(a)]) => a.trunc_i32().map(Value::I32).map_err(trap),
        ("u32_from_f32", [Value::F32(a)]) => a.trunc_u32().map(Value::U32).map_err(trap),
        ("i32_from_f64", [Value::F64(a)]) => a.trunc_i32().map(Value::I32).map_err(trap),
        ("u32_from_f64", [Value::F64(a)]) => a.trunc_u32().map(Value::U32).map_err(trap),
        // Stores into storage arrays (`check::storage`, ENG-03).
        ("store-f16", [Value::F32(a)]) => Ok(Value::F32(F16Bits::round(*a).widen())),
        ("store-bf16", [Value::F32(a)]) => Ok(Value::F32(Bf16Bits::round(*a).widen())),
        ("store-i8", [Value::I32(a)]) => narrow_i8(*a).map(|_| Value::I32(*a)).map_err(trap),
        ("store-u8", [Value::U32(a)]) => narrow_u8(*a).map(|_| Value::U32(*a)).map_err(trap),
        ("filled_f32", [Value::U32(len), Value::F32(value)]) => {
            filled(live, *len, *value).map(|items| Value::ArrayF32(Rc::new(RefCell::new(items))))
        }
        ("filled_f64", [Value::U32(len), Value::F64(value)]) => {
            filled(live, *len, *value).map(|items| Value::ArrayF64(Rc::new(RefCell::new(items))))
        }
        _ => Err(EvalError::tool(
            "INTERNAL_TYPE",
            "invalid checked float intrinsic",
        )),
    }
}

fn math(f: MathFn, values: &[Value]) -> Result<Value, EvalError> {
    if values.len() != f.arity() {
        return Err(invalid_operation());
    }
    let singles: Option<Vec<F32Bits>> = values
        .iter()
        .map(|value| match value {
            Value::F32(x) => Some(*x),
            _ => None,
        })
        .collect();
    if let Some(args) = singles {
        return Ok(Value::F32(crate::math::eval_f32(f, &args)));
    }
    let doubles: Option<Vec<F64Bits>> = values
        .iter()
        .map(|value| match value {
            Value::F64(x) => Some(*x),
            _ => None,
        })
        .collect();
    doubles
        .map(|args| Value::F64(crate::math::eval_f64(f, &args)))
        .ok_or_else(invalid_operation)
}

fn filled<T: Copy>(live: &mut usize, len: u32, value: T) -> Result<Vec<T>, EvalError> {
    let length = usize::try_from(len)
        .map_err(|_| EvalError::tool("ARRAY_SIZE_LIMIT", "array length exceeds host size"))?;
    if length > MAX_ARRAY_LEN {
        return Err(EvalError::tool(
            "ARRAY_SIZE_LIMIT",
            "array length limit exceeded",
        ));
    }
    admit_live(live, length)?;
    let mut items = Vec::new();
    items
        .try_reserve_exact(length)
        .map_err(|_| EvalError::tool("RESOURCE_LIMIT", "array allocation failed"))?;
    items.resize(length, value);
    Ok(items)
}

/// Reference `sum_tree`/`dot_tree`/`sum_exact`/`dot_exact`; a dot length
/// mismatch is the only trap.
pub(crate) fn reduction(
    kind: crate::fsum::Kind,
    operands: &[Value],
    span: crate::front::Span,
) -> Result<Value, EvalError> {
    use crate::{fsum::Kind, fsum_reference as reference};
    let arrays = operands
        .iter()
        .map(|operand| match operand {
            Value::ArrayF32(values) => values
                .try_borrow()
                .map_err(|_| EvalError::tool("INTERNAL_BORROW", "reduction borrow conflict")),
            _ => Err(EvalError::tool(
                "INTERNAL_TYPE",
                "invalid checked reduction",
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let a = &arrays[0];
    let b = arrays.get(1).unwrap_or(a);
    if a.len() != b.len() {
        return Err(EvalError::Trap {
            category: super::TrapCategory::IndexOutOfBounds,
            span,
        });
    }
    Ok(Value::F32(match kind {
        Kind::SumTree => reference::sum_tree(a),
        Kind::DotTree => reference::dot_tree(a, b),
        Kind::SumExact => reference::sum_exact(a),
        Kind::DotExact => reference::dot_exact(a, b),
    }))
}

use super::{Env, Evaluator};
use crate::front::Expr;
impl Evaluator<'_> {
    pub(super) fn array_literal(
        &mut self,
        expr: &Expr,
        items: &[Expr],
        env: &mut Env,
    ) -> Result<Value, EvalError> {
        if items.len() > MAX_ARRAY_LEN {
            return Err(EvalError::tool(
                "ARRAY_SIZE_LIMIT",
                "array length limit exceeded",
            ));
        }
        let ty = self
            .types
            .get(&expr.span)
            .ok_or_else(|| EvalError::tool("INTERNAL_TYPE", "missing array expression type"))?;
        match ty {
            Type::ArrayI32 => {
                let mut values = Vec::new();
                values
                    .try_reserve_exact(items.len())
                    .map_err(|_| EvalError::tool("RESOURCE_LIMIT", "array allocation failed"))?;
                for item in items {
                    let Value::I32(value) = self.expr(item, env)? else {
                        return Err(EvalError::tool("INTERNAL_TYPE", "invalid array element"));
                    };
                    values.push(value);
                }
                admit_live(&mut self.live_elements, values.len())?;
                Ok(Value::ArrayI32(Rc::new(RefCell::new(values))))
            }
            Type::ArrayU32 => {
                let mut values = Vec::new();
                values
                    .try_reserve_exact(items.len())
                    .map_err(|_| EvalError::tool("RESOURCE_LIMIT", "array allocation failed"))?;
                for item in items {
                    let Value::U32(value) = self.expr(item, env)? else {
                        return Err(EvalError::tool("INTERNAL_TYPE", "invalid array element"));
                    };
                    values.push(value);
                }
                admit_live(&mut self.live_elements, values.len())?;
                Ok(Value::ArrayU32(Rc::new(RefCell::new(values))))
            }
            Type::ArrayF32 => {
                let mut values = Vec::new();
                values
                    .try_reserve_exact(items.len())
                    .map_err(|_| EvalError::tool("RESOURCE_LIMIT", "array allocation failed"))?;
                for item in items {
                    let Value::F32(value) = self.expr(item, env)? else {
                        return Err(EvalError::tool("INTERNAL_TYPE", "invalid array element"));
                    };
                    values.push(value);
                }
                admit_live(&mut self.live_elements, values.len())?;
                Ok(Value::ArrayF32(Rc::new(RefCell::new(values))))
            }
            Type::ArrayF64 => {
                let mut values = Vec::new();
                values
                    .try_reserve_exact(items.len())
                    .map_err(|_| EvalError::tool("RESOURCE_LIMIT", "array allocation failed"))?;
                for item in items {
                    let Value::F64(value) = self.expr(item, env)? else {
                        return Err(EvalError::tool("INTERNAL_TYPE", "invalid array element"));
                    };
                    values.push(value);
                }
                admit_live(&mut self.live_elements, values.len())?;
                Ok(Value::ArrayF64(Rc::new(RefCell::new(values))))
            }
            Type::ArrayBool => {
                let mut values = Vec::new();
                values
                    .try_reserve_exact(items.len())
                    .map_err(|_| EvalError::tool("RESOURCE_LIMIT", "array allocation failed"))?;
                for item in items {
                    let Value::Bool(value) = self.expr(item, env)? else {
                        return Err(EvalError::tool("INTERNAL_TYPE", "invalid array element"));
                    };
                    values.push(value);
                }
                admit_live(&mut self.live_elements, values.len())?;
                Ok(Value::ArrayBool(Rc::new(RefCell::new(values))))
            }
            _ => Err(EvalError::tool(
                "INTERNAL_TYPE",
                "invalid checked array type",
            )),
        }
    }
}

/// The recognized function exposes only its final fresh output. Run its source
/// control flow first so fuel, bounds failures and cleanup keep their meaning;
/// normalize a successful serial completion to D5's once-rounded exact sum.
/// ponytail: repeats the dot products; use a hidden exact accumulator if this
/// reference/fallback path becomes performance sensitive.
pub(crate) struct TcInputs {
    a: Rc<RefCell<Vec<F32Bits>>>,
    b: Rc<RefCell<Vec<F32Bits>>>,
    dims: [u32; 3],
}
impl TcInputs {
    pub(crate) fn capture(args: &[Value]) -> Result<Self, EvalError> {
        let [
            Value::ArrayF32(a),
            Value::ArrayF32(b),
            Value::U32(r),
            Value::U32(k),
            Value::U32(n),
        ] = args
        else {
            return Err(EvalError::tool(
                "TC_CONTRACT",
                "invalid TC contraction signature",
            ));
        };
        Ok(Self {
            a: Rc::clone(a),
            b: Rc::clone(b),
            dims: [*r, *k, *n],
        })
    }
    pub(crate) fn finish(self, result: Value) -> Result<Value, EvalError> {
        let Value::ArrayF32(output) = &result else {
            return Err(EvalError::tool(
                "TC_CONTRACT",
                "invalid TC contraction output",
            ));
        };
        if Rc::ptr_eq(output, &self.a) || Rc::ptr_eq(output, &self.b) {
            return Err(EvalError::tool("TC_CONTRACT", "TC output aliases an input"));
        }
        // No arithmetic is reached by an empty contraction. Unused input
        // elements (including nonfinite ones) must not introduce new failures.
        if self.dims.contains(&0) {
            return Ok(result);
        }
        let [r, k, n] = self.dims.map(u64::from);
        let a = self.a.borrow();
        let b = self.b.borrow();
        let words = |v: &[F32Bits], count: u64| -> Result<Vec<u32>, EvalError> {
            let slice = usize::try_from(count)
                .ok()
                .and_then(|n| v.get(..n))
                .ok_or_else(|| {
                    EvalError::tool("TC_CONTRACT", "invalid completed TC input extent")
                })?;
            Ok(slice.iter().map(|v| v.to_bits()).collect())
        };
        let values = invar_device::tc::reference(self.dims, &words(&a, r * k)?, &words(&b, k * n)?)
            .map_err(|_| EvalError::tool("TC_INPUT_RANGE", "invalid TC reference inputs"))?;
        if output.borrow().len() != values.len() {
            return Err(EvalError::tool(
                "TC_CONTRACT",
                "invalid completed TC output extent",
            ));
        }
        *output.borrow_mut() = values.into_iter().map(F32Bits::from_bits).collect();
        Ok(result)
    }
}

#[cfg(test)]
#[test]
fn issue_ai_01_machine_diagnostics_tc_serial_completion_refuses_output_aliases() {
    let a = Rc::new(RefCell::new(vec![F32Bits::from_bits(0x3f80_0000)]));
    let b = Rc::new(RefCell::new(vec![F32Bits::from_bits(0x4000_0000)]));
    for output in [&a, &b] {
        let inputs = TcInputs::capture(&[
            Value::ArrayF32(Rc::clone(&a)),
            Value::ArrayF32(Rc::clone(&b)),
            Value::U32(1),
            Value::U32(1),
            Value::U32(1),
        ])
        .unwrap();
        assert!(matches!(
            inputs.finish(Value::ArrayF32(Rc::clone(output))),
            Err(EvalError::Tool {
                code: "TC_CONTRACT",
                ..
            })
        ));
    }
    assert_eq!(a.borrow()[0].to_bits(), 0x3f80_0000);
    assert_eq!(b.borrow()[0].to_bits(), 0x4000_0000);
}
