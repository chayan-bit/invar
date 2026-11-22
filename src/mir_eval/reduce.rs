//! MIR execution of the deterministic float reductions. An engine may supply
//! a [`Reducer`] that computes a reached call on its own workers or device;
//! the serial result is the same bits by contract, so the hook only selects
//! where the work runs. The length check and its trap stay here.

use super::*;
use crate::{
    fsum::{self, Kind},
    numeric_contract::F32Bits,
};

/// One reached reduction call with its borrowed operands.
pub struct ReduceCall<'a> {
    pub function: &'a str,
    pub span: Span,
    pub kind: Kind,
    pub a: &'a [F32Bits],
    /// The second dot operand; equal in length to `a`.
    pub b: Option<&'a [F32Bits]>,
}

/// Returns `Some(bits)` when the engine computed the call, `None` to keep
/// the serial MIR result. Errors end the run like any other tool failure.
pub type Reducer<'a> = dyn FnMut(&ReduceCall<'_>) -> Result<Option<F32Bits>, EvalError> + 'a;

fn word(value: F32Bits) -> f32 {
    f32::from_bits(value.to_bits())
}

/// The serial engine result of one call, from the shared fast code.
pub fn serial_reduction(call: &ReduceCall<'_>) -> F32Bits {
    let a = call.a;
    let b = call.b.unwrap_or(a);
    let bits = match call.kind {
        Kind::SumTree => fsum::tree(a.len(), |i| word(a[i])),
        Kind::DotTree => fsum::tree(a.len(), |i| word(a[i]) * word(b[i])),
        Kind::SumExact => {
            let mut exact = fsum::Exact::new(false);
            exact.add_words(a, F32Bits::to_bits);
            exact.finish()
        }
        Kind::DotExact => {
            let mut exact = fsum::Exact::new(true);
            exact.add_products(a, b, F32Bits::to_bits);
            exact.finish()
        }
    };
    F32Bits::from_bits(bits)
}

fn floats(value: &Value) -> Result<Rc<RefCell<Vec<F32Bits>>>, EvalError> {
    match value {
        Value::ArrayF32(values) => Ok(Rc::clone(values)),
        _ => Err(tool("MIR_TYPE", "reduction operand is not Array<f32>")),
    }
}

impl Interpreter<'_> {
    pub(super) fn reduce(
        &mut self,
        function: &Function,
        name: &str,
        args: &[ValueId],
        span: Span,
        frame: &Frame,
    ) -> Result<Value, EvalError> {
        let kind = Kind::from_name(name).ok_or_else(|| tool("MIR_CALL", "unknown reduction"))?;
        let operands = args
            .iter()
            .map(|id| frame.value(*id).and_then(|value| floats(&value)))
            .collect::<Result<Vec<_>, _>>()?;
        if operands.len() != if kind.is_dot() { 2 } else { 1 } {
            return Err(tool("MIR_CALL", "invalid reduction arity"));
        }
        let conflict = |_| tool("MIR_LOAN", "array borrow conflict");
        let a = operands[0].try_borrow().map_err(conflict)?;
        let b = match operands.get(1) {
            Some(values) => Some(values.try_borrow().map_err(conflict)?),
            None => None,
        };
        if b.as_ref().is_some_and(|b| b.len() != a.len()) {
            return Err(trap(TrapCategory::IndexOutOfBounds, span));
        }
        let call = ReduceCall {
            function: &function.name,
            span,
            kind,
            a: &a,
            b: b.as_deref().map(Vec::as_slice),
        };
        if let Some(reducer) = self.calls.reducer.as_mut()
            && let Some(bits) = reducer(&call)?
        {
            return Ok(Value::F32(bits));
        }
        Ok(Value::F32(serial_reduction(&call)))
    }
}

impl Interpreter<'_> {
    /// `min_of`, `max_of`, `argmin`, `argmax` (#207): the engine's selector
    /// may compute a reached call on its workers; the serial result is the
    /// same bits and index by the fold's associativity.
    pub(super) fn extremum(
        &mut self,
        function: &Function,
        name: &str,
        args: &[ValueId],
        span: Span,
        frame: &Frame,
    ) -> Result<Value, EvalError> {
        let kind = crate::extremum::Kind::from_name(name)
            .ok_or_else(|| tool("MIR_CALL", "unknown extremum"))?;
        let [arg] = args else {
            return Err(tool("MIR_CALL", "invalid extremum arity"));
        };
        let operand = frame.value(*arg)?;
        if let Some(selector) = self.calls.selector.as_mut() {
            let call = SelectCall {
                function: &function.name,
                span,
                kind,
                operand: &operand,
            };
            if let Some(value) = selector(&call)? {
                return Ok(value);
            }
        }
        crate::extremum::evaluate(kind, &operand, span)
    }
}

/// The interpreter's engine hooks for reached calls.
#[derive(Default)]
pub(super) struct CallHooks<'a> {
    /// `sum_tree`/`dot_tree`/`sum_exact`/`dot_exact`.
    pub(super) reducer: Option<&'a mut Reducer<'a>>,
    /// `min_of`/`max_of`/`argmin`/`argmax` (#207).
    pub(super) selector: Option<&'a mut Selector<'a>>,
}

/// One reached `min_of`/`max_of`/`argmin`/`argmax` call (#207).
pub struct SelectCall<'a> {
    pub function: &'a str,
    pub span: Span,
    pub kind: crate::extremum::Kind,
    /// The borrowed array operand.
    pub operand: &'a Value,
}

/// Returns `Some(result)` when the engine computed the call (or its
/// empty-input trap), `None` to keep the serial MIR result.
pub type Selector<'a> = dyn FnMut(&SelectCall<'_>) -> Result<Option<Value>, EvalError> + 'a;
