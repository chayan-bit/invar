//! Values of the PNF reference executor and the source scalar semantics
//! (checked i32/u32, strict f32, wrapping intrinsics) they follow.

use std::{cell::RefCell, rc::Rc};

use super::Stop;
use crate::{
    eval::{self, TrapCategory, floating},
    front::{BinaryOp, Span, Type, UnaryOp},
    numeric_contract::F32Bits,
};

/// Largest array the reference executor allocates (matches the test oracle
/// budgets; larger requests are a tool limit, not a source trap).
pub(super) const MAX_ELEMENTS: usize = 1 << 22;

#[derive(Clone, Debug, PartialEq)]
pub(in crate::general_net::normal_form) enum Elems {
    I32(Vec<i32>),
    U32(Vec<u32>),
    F32(Vec<F32Bits>),
}

impl Elems {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::I32(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::F32(v) => v.len(),
        }
    }

    pub(super) fn get(&self, index: usize) -> Option<Val> {
        Some(match self {
            Self::I32(v) => Val::I32(*v.get(index)?),
            Self::U32(v) => Val::U32(*v.get(index)?),
            Self::F32(v) => Val::F32(*v.get(index)?),
        })
    }

    /// Store `value`; `None` when out of bounds, `Err` on a type mismatch.
    pub(super) fn set(&mut self, index: usize, value: &Val) -> Result<Option<()>, Stop> {
        let slot = match (self, value) {
            (Self::I32(v), Val::I32(x)) => v.get_mut(index).map(|slot| *slot = *x),
            (Self::U32(v), Val::U32(x)) => v.get_mut(index).map(|slot| *slot = *x),
            (Self::F32(v), Val::F32(x)) => v.get_mut(index).map(|slot| *slot = *x),
            _ => return Err(Stop::internal("array store type")),
        };
        Ok(slot)
    }
}

pub(in crate::general_net::normal_form) type Array = Rc<RefCell<Elems>>;

pub(super) fn array_id(array: &Array) -> usize {
    Rc::as_ptr(array).cast::<u8>() as usize
}

/// Symbolic values of the law executor's element phase (never escape a loop).
#[derive(Clone, Debug, PartialEq)]
pub(in crate::general_net::normal_form) enum Sym {
    /// Unknown in this phase: any use that matters selects the fallback.
    Poison,
    /// `sum_j c_j x_j + k` over the current group, in the group's ring.
    Lin(super::rep::Lin),
    /// `x -> max(x + a, b)` (or `min`); `None` is the order's infinity.
    Trop { a: Option<i128>, b: Option<i128> },
    /// The select variable; its lane state lives in the lane.
    Select,
    /// A select companion; its lane state lives in the lane.
    Companion,
    /// A last-writer scalar: the last value written in this lane, if any.
    Last(Option<Box<Val>>),
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::general_net::normal_form) enum Val {
    I32(i32),
    U32(u32),
    F32(F32Bits),
    Bool(bool),
    Unit,
    Arr(Array),
    Sym(Box<Sym>),
}

impl Val {
    pub(super) fn poison() -> Self {
        Self::Sym(Box::new(Sym::Poison))
    }

    pub(super) fn is_sym(&self) -> bool {
        matches!(self, Self::Sym(_))
    }

    pub(super) fn sym(&self) -> Option<&Sym> {
        match self {
            Self::Sym(sym) => Some(sym),
            _ => None,
        }
    }

    pub(super) fn array(&self) -> Result<&Array, Stop> {
        match self {
            Self::Arr(array) => Ok(array),
            _ => Err(Stop::internal("array expected")),
        }
    }

    /// Exact integer value of an i32 or u32 scalar.
    pub(super) fn int(&self) -> Option<i128> {
        match self {
            Self::I32(x) => Some(i128::from(*x)),
            Self::U32(x) => Some(i128::from(*x)),
            _ => None,
        }
    }

    pub(super) fn from_int(ty: super::super::IntTy, value: i128) -> Option<Self> {
        match ty {
            super::super::IntTy::I32 => i32::try_from(value).ok().map(Self::I32),
            super::super::IntTy::U32 => u32::try_from(value).ok().map(Self::U32),
        }
    }

    pub(super) fn from_eval(value: &eval::Value) -> Result<Self, Stop> {
        Ok(match value {
            eval::Value::I32(x) => Self::I32(*x),
            eval::Value::U32(x) => Self::U32(*x),
            eval::Value::F32(x) => Self::F32(*x),
            eval::Value::Bool(x) => Self::Bool(*x),
            eval::Value::Unit => Self::Unit,
            eval::Value::ArrayI32(v) => {
                Self::Arr(Rc::new(RefCell::new(Elems::I32(v.borrow().clone()))))
            }
            eval::Value::ArrayU32(v) => {
                Self::Arr(Rc::new(RefCell::new(Elems::U32(v.borrow().clone()))))
            }
            eval::Value::ArrayF32(v) => {
                Self::Arr(Rc::new(RefCell::new(Elems::F32(v.borrow().clone()))))
            }
            _ => return Err(Stop::unsupported("argument kind")),
        })
    }

    pub(super) fn to_eval(&self) -> Result<eval::Value, Stop> {
        Ok(match self {
            Self::I32(x) => eval::Value::I32(*x),
            Self::U32(x) => eval::Value::U32(*x),
            Self::F32(x) => eval::Value::F32(*x),
            Self::Bool(x) => eval::Value::Bool(*x),
            Self::Unit => eval::Value::Unit,
            Self::Arr(array) => match &*array.borrow() {
                Elems::I32(v) => eval::Value::ArrayI32(Rc::new(RefCell::new(v.clone()))),
                Elems::U32(v) => eval::Value::ArrayU32(Rc::new(RefCell::new(v.clone()))),
                Elems::F32(v) => eval::Value::ArrayF32(Rc::new(RefCell::new(v.clone()))),
            },
            Self::Sym(_) => return Err(Stop::internal("symbolic value escaped")),
        })
    }
}

fn trap(category: TrapCategory, span: Span) -> Stop {
    Stop::Trap(category, span)
}

/// Checked source binary operation on concrete scalars.
pub(super) fn binary(op: BinaryOp, left: &Val, right: &Val, span: Span) -> Result<Val, Stop> {
    use BinaryOp as B;
    let overflow = || trap(TrapCategory::IntegerOverflow, span);
    let zero = || trap(TrapCategory::DivisionByZero, span);
    macro_rules! integer {
        ($a:expr, $b:expr, $variant:ident, $rem:expr) => {{
            let (a, b) = ($a, $b);
            match op {
                B::Eq => Val::Bool(a == b),
                B::Ne => Val::Bool(a != b),
                B::Lt => Val::Bool(a < b),
                B::Le => Val::Bool(a <= b),
                B::Gt => Val::Bool(a > b),
                B::Ge => Val::Bool(a >= b),
                B::Add => Val::$variant(a.checked_add(b).ok_or_else(overflow)?),
                B::Sub => Val::$variant(a.checked_sub(b).ok_or_else(overflow)?),
                B::Mul => Val::$variant(a.checked_mul(b).ok_or_else(overflow)?),
                B::Div => {
                    if b == 0 {
                        return Err(zero());
                    }
                    Val::$variant(a.checked_div(b).ok_or_else(overflow)?)
                }
                B::Rem => {
                    if b == 0 {
                        return Err(zero());
                    }
                    Val::$variant($rem)
                }
                B::And | B::Or => return Err(Stop::internal("logic on integers")),
            }
        }};
    }
    Ok(match (left, right) {
        (Val::F32(a), Val::F32(b)) => match floating::binary(*a, op, *b) {
            Ok(eval::Value::F32(x)) => Val::F32(x),
            Ok(eval::Value::Bool(x)) => Val::Bool(x),
            _ => return Err(Stop::internal("f32 operation")),
        },
        (Val::I32(a), Val::I32(b)) => integer!(
            *a,
            *b,
            I32,
            if *a == i32::MIN && *b == -1 {
                0
            } else {
                a.checked_rem(*b).ok_or_else(overflow)?
            }
        ),
        (Val::U32(a), Val::U32(b)) => {
            integer!(*a, *b, U32, a.checked_rem(*b).ok_or_else(overflow)?)
        }
        (Val::Bool(a), Val::Bool(b)) => Val::Bool(match op {
            B::Eq => a == b,
            B::Ne => a != b,
            B::And => *a && *b,
            B::Or => *a || *b,
            _ => return Err(Stop::internal("bool operation")),
        }),
        _ => return Err(Stop::internal("binary operand types")),
    })
}

pub(super) fn unary(op: UnaryOp, value: &Val, span: Span) -> Result<Val, Stop> {
    Ok(match (op, value) {
        (UnaryOp::Neg, Val::F32(x)) => Val::F32(x.negate()),
        (UnaryOp::Not, Val::Bool(x)) => Val::Bool(!x),
        (UnaryOp::Neg, Val::I32(x)) => Val::I32(
            x.checked_neg()
                .ok_or(trap(TrapCategory::IntegerOverflow, span))?,
        ),
        _ => return Err(Stop::internal("unary operand type")),
    })
}

fn length(value: &Val) -> Result<usize, Stop> {
    let Val::U32(n) = value else {
        return Err(Stop::internal("length type"));
    };
    let n = usize::try_from(*n).map_err(|_| Stop::tool("array length exceeds host size"))?;
    if n > MAX_ELEMENTS {
        return Err(Stop::tool("array length limit exceeded"));
    }
    Ok(n)
}

/// Built-in intrinsics; `None` when `name` is not one.
pub(super) fn intrinsic(name: &str, args: &[Val]) -> Option<Result<Val, Stop>> {
    let array = |elems: Elems| Ok(Val::Arr(Rc::new(RefCell::new(elems))));
    Some(match (name, args) {
        ("wrapping_add", [Val::U32(a), Val::U32(b)]) => Ok(Val::U32(a.wrapping_add(*b))),
        ("wrapping_mul", [Val::U32(a), Val::U32(b)]) => Ok(Val::U32(a.wrapping_mul(*b))),
        ("fma", [Val::F32(a), Val::F32(b), Val::F32(c)]) => Ok(Val::F32(a.fma(*b, *c))),
        ("f32_from_i32", [Val::I32(a)]) => Ok(Val::F32(F32Bits::from_i32(*a))),
        ("f32_from_u32", [Val::U32(a)]) => Ok(Val::F32(F32Bits::from_u32(*a))),
        ("filled_i32", [n, Val::I32(x)]) => length(n).and_then(|n| array(Elems::I32(vec![*x; n]))),
        ("filled_u32", [n, Val::U32(x)]) => length(n).and_then(|n| array(Elems::U32(vec![*x; n]))),
        ("filled_f32", [n, Val::F32(x)]) => length(n).and_then(|n| array(Elems::F32(vec![*x; n]))),
        ("wrapping_add" | "wrapping_mul" | "fma" | "f32_from_i32" | "f32_from_u32", _)
        | ("filled_i32" | "filled_u32" | "filled_f32", _) => {
            Err(Stop::internal("intrinsic arguments"))
        }
        _ => return None,
    })
}

/// A literal array of element type `elem`.
pub(super) fn array_literal(elem: &Type, items: Vec<Val>) -> Result<Val, Stop> {
    let elems = match elem {
        Type::I32 => Elems::I32(
            items
                .iter()
                .map(|v| match v {
                    Val::I32(x) => Ok(*x),
                    _ => Err(Stop::internal("array literal type")),
                })
                .collect::<Result<_, _>>()?,
        ),
        Type::U32 => Elems::U32(
            items
                .iter()
                .map(|v| match v {
                    Val::U32(x) => Ok(*x),
                    _ => Err(Stop::internal("array literal type")),
                })
                .collect::<Result<_, _>>()?,
        ),
        Type::F32 => Elems::F32(
            items
                .iter()
                .map(|v| match v {
                    Val::F32(x) => Ok(*x),
                    _ => Err(Stop::internal("array literal type")),
                })
                .collect::<Result<_, _>>()?,
        ),
        _ => return Err(Stop::unsupported("array literal element type")),
    };
    Ok(Val::Arr(Rc::new(RefCell::new(elems))))
}
