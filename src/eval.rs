//! Independent source evaluator. This does not share optimized lowering.

use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
};

use serde_json::{Value as JsonValue, json};

use crate::{
    check::CheckedProgram,
    front::{
        BinaryOp, Block, ElseArm, Expr, ExprKind, Function, Mode, Place, Span, Stmt, StmtKind,
        Type, UnaryOp,
    },
};

mod error;
mod value;
pub use error::{CleanupEvent, EvalError, TrapCategory};
pub(crate) use error::{checked_angle, checked_scale};
pub use value::{Value, ViewRange};
pub(crate) use value::{make_view, view_index, view_len};

const MAX_CALL_DEPTH: usize = 64;
/// The fuel `invar run` uses without `--fuel`.
pub const DEFAULT_FUEL: u64 = 1_000_000;
pub(crate) const MAX_ARRAY_LEN: usize = crate::array_limits::MAX_ARRAY_LEN;
/// Aggregate cap on array elements allocated and not yet cleaned up. Shared by
/// the MIR evaluator and injected into the separately built native host.
pub(crate) const MAX_LIVE_ELEMENTS: usize = crate::array_limits::MAX_LIVE_ELEMENTS;

/// Account for a new array of `length` elements against [`MAX_LIVE_ELEMENTS`].
pub(crate) fn admit_live(live: &mut usize, length: usize) -> Result<(), EvalError> {
    *live = live
        .checked_add(length)
        .filter(|total| *total <= MAX_LIVE_ELEMENTS)
        .ok_or_else(|| EvalError::tool("ARRAY_SIZE_LIMIT", "live array element limit exceeded"))?;
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
pub struct Outcome {
    pub value: Value,
    pub steps_used: u64,
    /// Internal lifetime observation; Invar source has no cleanup IO.
    pub cleanup_trace: Vec<CleanupEvent>,
}

/// Test observation, including cleanup already completed before an error.
#[derive(Debug)]
pub struct Observation {
    pub result: Result<Value, EvalError>,
    pub steps_used: u64,
    pub cleanup_trace: Vec<CleanupEvent>,
}

enum Flow {
    Continue,
    Return(Value),
}

struct Scope {
    values: HashMap<String, Option<Value>>,
    order: Vec<String>,
    borrowed: HashSet<String>,
}

impl Scope {
    fn new() -> Self {
        Self {
            values: HashMap::new(),
            order: Vec::new(),
            borrowed: HashSet::new(),
        }
    }

    fn bind(&mut self, name: &str, value: Value, borrowed: bool) {
        if !self.values.contains_key(name) {
            self.order.push(name.to_owned());
        }
        self.values.insert(name.to_owned(), Some(value));
        if borrowed {
            self.borrowed.insert(name.to_owned());
        }
    }
}

fn array_len_u32(length: usize) -> Result<u32, EvalError> {
    u32::try_from(length)
        .map_err(|_| EvalError::tool("INTERNAL_ARRAY_SIZE", "array length exceeds u32"))
}

fn host_index(index: u32, host_max: u64) -> Result<usize, EvalError> {
    if u64::from(index) > host_max {
        return Err(EvalError::tool("INTERNAL_INDEX", "index exceeds host size"));
    }
    usize::try_from(index).map_err(|_| EvalError::tool("INTERNAL_INDEX", "index exceeds host size"))
}

struct Env {
    scopes: Vec<Scope>,
}

impl Env {
    fn new(function: &Function, args: Vec<Value>) -> Self {
        let mut params = Scope::new();
        for (param, value) in function.params.iter().zip(args) {
            params.bind(
                &param.name,
                value,
                matches!(param.mode, Some(Mode::Read | Mode::Mut)),
            );
        }
        Self {
            scopes: vec![params],
        }
    }

    fn slot(&self, name: &str) -> Result<&Value, EvalError> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.values.get(name))
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                EvalError::tool(
                    "INTERNAL_BINDING",
                    format!("unavailable checked binding '{name}'"),
                )
            })
    }

    fn get_scalar(&self, name: &str) -> Result<Value, EvalError> {
        match self.slot(name)? {
            Value::I32(value) => Ok(Value::I32(*value)),
            Value::U32(value) => Ok(Value::U32(*value)),
            Value::F32(value) => Ok(Value::F32(*value)),
            Value::F64(value) => Ok(Value::F64(*value)),
            Value::Bool(value) => Ok(Value::Bool(*value)),
            Value::Unit => Ok(Value::Unit),
            Value::ViewI32(values, range) => Ok(Value::ViewI32(Rc::clone(values), *range)),
            Value::ViewU32(values, range) => Ok(Value::ViewU32(Rc::clone(values), *range)),
            Value::ArrayI32(_)
            | Value::ArrayU32(_)
            | Value::ArrayF32(_)
            | Value::ArrayF64(_)
            | Value::ArrayBool(_)
            | Value::Bytes(_)
            | Value::Text(_)
            | Value::VoxelGrid(_)
            | Value::Record(_)
            | Value::Variant(_) => Err(EvalError::tool(
                "INTERNAL_ARRAY_COPY",
                "array value requires an explicit move",
            )),
        }
    }

    fn borrow_owned(&self, name: &str) -> Result<Value, EvalError> {
        match self.slot(name)? {
            Value::ArrayI32(values) => Ok(Value::ArrayI32(Rc::clone(values))),
            Value::ArrayU32(values) => Ok(Value::ArrayU32(Rc::clone(values))),
            Value::ArrayF32(values) => Ok(Value::ArrayF32(Rc::clone(values))),
            Value::ArrayF64(values) => Ok(Value::ArrayF64(Rc::clone(values))),
            Value::ArrayBool(values) => Ok(Value::ArrayBool(Rc::clone(values))),
            Value::Bytes(values) => Ok(Value::Bytes(Rc::clone(values))),
            Value::Text(value) => Ok(Value::Text(Rc::clone(value))),
            Value::VoxelGrid(grid) => Ok(Value::VoxelGrid(Rc::clone(grid))),
            Value::Record(values) => Ok(Value::Record(Rc::clone(values))),
            Value::Variant(values) => Ok(Value::Variant(Rc::clone(values))),
            _ => Err(EvalError::tool(
                "INTERNAL_TYPE",
                "loan requires array binding",
            )),
        }
    }

    fn take(&mut self, name: &str) -> Result<Value, EvalError> {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(slot) = scope.values.get_mut(name) {
                return slot
                    .take()
                    .ok_or_else(|| EvalError::tool("INTERNAL_BINDING", "moved binding was used"));
            }
        }
        Err(EvalError::tool(
            "INTERNAL_BINDING",
            format!("missing checked binding '{name}'"),
        ))
    }

    fn set(&mut self, name: &str, value: Value) -> Result<Value, EvalError> {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(slot) = scope.values.get_mut(name) {
                let old = slot.as_mut().ok_or_else(|| {
                    EvalError::tool("INTERNAL_BINDING", "moved binding was assigned")
                })?;
                return Ok(std::mem::replace(old, value));
            }
        }
        Err(EvalError::tool(
            "INTERNAL_BINDING",
            format!("missing checked binding '{name}'"),
        ))
    }

    fn bind(&mut self, name: &str, value: Value) -> Result<(), EvalError> {
        let scope = self
            .scopes
            .last_mut()
            .ok_or_else(|| EvalError::tool("INTERNAL_SCOPE", "missing runtime scope"))?;
        scope.bind(name, value, false);
        Ok(())
    }

    fn length(&self, name: &str) -> Result<u32, EvalError> {
        view_len(self.slot(name)?)
            .ok_or_else(|| EvalError::tool("INTERNAL_TYPE", "len requires an array or view"))
    }

    fn index(&self, name: &str, index: u32, span: Span) -> Result<Value, EvalError> {
        let index = host_index(index, usize::MAX as u64)?;
        match self.slot(name)? {
            Value::ArrayI32(values) => values.borrow().get(index).copied().map(Value::I32),
            Value::ArrayU32(values) => values.borrow().get(index).copied().map(Value::U32),
            Value::ArrayF32(values) => values.borrow().get(index).copied().map(Value::F32),
            Value::ArrayF64(values) => values.borrow().get(index).copied().map(Value::F64),
            Value::ArrayBool(values) => values.borrow().get(index).copied().map(Value::Bool),
            Value::Bytes(values) => values
                .get(index)
                .copied()
                .map(|byte| Value::U32(u32::from(byte))),
            Value::ViewI32(values, range) => values
                .borrow()
                .get(view_index(*range, index as u32, span)?)
                .copied()
                .map(Value::I32),
            Value::ViewU32(values, range) => values
                .borrow()
                .get(view_index(*range, index as u32, span)?)
                .copied()
                .map(Value::U32),
            _ => {
                return Err(EvalError::tool(
                    "INTERNAL_TYPE",
                    "indexing requires an array",
                ));
            }
        }
        .ok_or(EvalError::Trap {
            category: TrapCategory::IndexOutOfBounds,
            span,
        })
    }

    fn record_field(&self, name: &str, index: usize) -> Result<Value, EvalError> {
        let Value::Record(fields) = self.slot(name)? else {
            return Err(EvalError::tool(
                "INTERNAL_TYPE",
                "field base is not a record",
            ));
        };
        match fields.borrow().get(index) {
            Some(Value::I32(value)) => Ok(Value::I32(*value)),
            Some(Value::U32(value)) => Ok(Value::U32(*value)),
            Some(Value::Bool(value)) => Ok(Value::Bool(*value)),
            // A checked `f32` member is a copyable scalar like the integers.
            Some(Value::F32(value)) => Ok(Value::F32(*value)),
            Some(Value::F64(value)) => Ok(Value::F64(*value)),
            _ => Err(EvalError::tool(
                "INTERNAL_TYPE",
                "invalid checked record field",
            )),
        }
    }

    fn move_record_field(&mut self, name: &str, index: usize) -> Result<Value, EvalError> {
        let Value::Record(fields) = self.slot(name)? else {
            return Err(EvalError::tool(
                "INTERNAL_TYPE",
                "field move base is not a record",
            ));
        };
        let mut fields = fields.borrow_mut();
        let field = fields
            .get_mut(index)
            .ok_or_else(|| EvalError::tool("INTERNAL_TYPE", "field move index missing"))?;
        if matches!(field, Value::Unit) {
            return Err(EvalError::tool("INTERNAL_TYPE", "field already moved"));
        }
        Ok(std::mem::replace(field, Value::Unit))
    }

    fn set_index(
        &mut self,
        name: &str,
        index: u32,
        value: Value,
        span: Span,
    ) -> Result<(), EvalError> {
        let index = host_index(index, usize::MAX as u64)?;
        for scope in self.scopes.iter_mut().rev() {
            if let Some(slot) = scope.values.get_mut(name) {
                return match (slot.as_mut(), value) {
                    (Some(Value::ArrayI32(values)), Value::I32(value)) => values
                        .borrow_mut()
                        .get_mut(index)
                        .map(|slot| *slot = value)
                        .ok_or(EvalError::Trap {
                            category: TrapCategory::IndexOutOfBounds,
                            span,
                        }),
                    (Some(Value::ArrayU32(values)), Value::U32(value)) => values
                        .borrow_mut()
                        .get_mut(index)
                        .map(|slot| *slot = value)
                        .ok_or(EvalError::Trap {
                            category: TrapCategory::IndexOutOfBounds,
                            span,
                        }),
                    (Some(Value::ArrayF32(values)), Value::F32(value)) => values
                        .borrow_mut()
                        .get_mut(index)
                        .map(|slot| *slot = value)
                        .ok_or(EvalError::Trap {
                            category: TrapCategory::IndexOutOfBounds,
                            span,
                        }),
                    (Some(Value::ArrayF64(values)), Value::F64(value)) => values
                        .borrow_mut()
                        .get_mut(index)
                        .map(|slot| *slot = value)
                        .ok_or(EvalError::Trap {
                            category: TrapCategory::IndexOutOfBounds,
                            span,
                        }),
                    (Some(Value::ArrayBool(values)), Value::Bool(value)) => values
                        .borrow_mut()
                        .get_mut(index)
                        .map(|slot| *slot = value)
                        .ok_or(EvalError::Trap {
                            category: TrapCategory::IndexOutOfBounds,
                            span,
                        }),
                    (Some(Value::ViewI32(values, range)), Value::I32(value)) if range.mutable => {
                        values
                            .borrow_mut()
                            .get_mut(view_index(*range, index as u32, span)?)
                            .map(|slot| *slot = value)
                            .ok_or(EvalError::Trap {
                                category: TrapCategory::IndexOutOfBounds,
                                span,
                            })
                    }
                    (Some(Value::ViewU32(values, range)), Value::U32(value)) if range.mutable => {
                        values
                            .borrow_mut()
                            .get_mut(view_index(*range, index as u32, span)?)
                            .map(|slot| *slot = value)
                            .ok_or(EvalError::Trap {
                                category: TrapCategory::IndexOutOfBounds,
                                span,
                            })
                    }
                    _ => Err(EvalError::tool(
                        "INTERNAL_TYPE",
                        "invalid checked array store",
                    )),
                };
            }
        }
        Err(EvalError::tool(
            "INTERNAL_BINDING",
            format!("missing checked binding '{name}'"),
        ))
    }

    fn pop(&mut self, trace: &mut Vec<CleanupEvent>, live: &mut usize, normal: bool) {
        if let Some(mut scope) = self.scopes.pop()
            && normal
        {
            for name in scope.order.into_iter().rev() {
                if let Some(Some(value)) = scope.values.remove(&name)
                    && !scope.borrowed.contains(&name)
                {
                    record_cleanup(trace, live, &name, &value);
                }
            }
        }
    }
}

fn record_cleanup(trace: &mut Vec<CleanupEvent>, live: &mut usize, name: &str, value: &Value) {
    fn size(value: &Value) -> usize {
        match value {
            Value::ArrayI32(values) => values.borrow().len(),
            Value::ArrayU32(values) => values.borrow().len(),
            Value::ArrayF32(values) => values.borrow().len(),
            Value::ArrayF64(values) => values.borrow().len(),
            Value::ArrayBool(values) => values.borrow().len(),
            Value::Bytes(values) => values.len(),
            Value::Text(value) => value.len(),
            Value::VoxelGrid(grid) => grid.storage_elements(),
            Value::Record(values) => {
                values.borrow().len() + values.borrow().iter().map(size).sum::<usize>()
            }
            Value::Variant(storage) => 2 + storage.borrow().1.as_ref().map(size).unwrap_or(0),
            _ => 0,
        }
    }
    if !matches!(
        value,
        Value::ArrayI32(_)
            | Value::ArrayU32(_)
            | Value::ArrayF32(_)
            | Value::ArrayF64(_)
            | Value::ArrayBool(_)
            | Value::Bytes(_)
            | Value::Text(_)
            | Value::VoxelGrid(_)
            | Value::Record(_)
            | Value::Variant(_)
    ) {
        return;
    }
    let length = size(value);
    // Every cleaned owner was admitted exactly once; moves transfer, never re-admit.
    *live = live.saturating_sub(length);
    trace.push(CleanupEvent {
        name: name.to_owned(),
        length,
    });
}

struct Evaluator<'a> {
    program: &'a CheckedProgram,
    types: HashMap<Span, Type>,
    fuel_limit: u64,
    steps_used: u64,
    call_depth: usize,
    trace: Vec<CleanupEvent>,
    live_elements: usize,
    pending_result_error: Option<Value>,
}

/// Execute a checked program from `main` with an exact fuel boundary.
/// Borrowed array parameters share storage with their caller for the call duration.
pub fn run(program: &CheckedProgram, fuel_limit: u64) -> Result<Outcome, EvalError> {
    let observation = observe(program, fuel_limit);
    observation.result.map(|value| Outcome {
        value,
        steps_used: observation.steps_used,
        cleanup_trace: observation.cleanup_trace,
    })
}

/// Run with a test-only lifetime observation; no trace is emitted by source programs.
pub fn observe(program: &CheckedProgram, fuel_limit: u64) -> Observation {
    if fuel_limit == 0 {
        return Observation {
            result: Err(EvalError::tool("INVALID_FUEL", "fuel must be positive")),
            steps_used: 0,
            cleanup_trace: Vec::new(),
        };
    }
    stack::observe_on_own_stack(|| {
        let mut evaluator = Evaluator {
            program,
            types: program.types_for("main"),
            fuel_limit,
            steps_used: 0,
            call_depth: 0,
            trace: Vec::new(),
            live_elements: 0,
            pending_result_error: None,
        };
        let result = evaluator.call_user("main", Vec::new());
        Observation {
            result,
            steps_used: evaluator.steps_used,
            cleanup_trace: evaluator.trace,
        }
    })
}

pub mod host_read;
pub mod named;
pub(crate) mod stack;

include!("eval/engine.rs");

#[cfg(test)]
mod tests {
    #[test]
    fn machine_diagnostics_eval_faults() {
        let mut errors = vec![(
            super::host_index(65_536, u16::MAX as u64).unwrap_err(),
            "INTERNAL_INDEX",
        )];
        #[cfg(target_pointer_width = "64")]
        errors.push((
            super::array_len_u32(u32::MAX as usize + 1).unwrap_err(),
            "INTERNAL_ARRAY_SIZE",
        ));
        let schema: serde_json::Value =
            serde_json::from_str(include_str!("../schema/invar-diagnostics-v3.schema.json"))
                .unwrap();
        for (error, code) in errors {
            let document = error.to_diagnostics_json("fixture.inv", b"");
            assert_eq!(document["diagnostics"][0]["code"], code);
            assert!(jsonschema::is_valid(&schema, &document), "{document}");
        }
    }

    #[test]
    fn machine_diagnostics_missing_runtime_scope() {
        let mut env = super::Env { scopes: Vec::new() };
        let error = env.bind("x", super::Value::I32(1)).unwrap_err();
        let document = error.to_diagnostics_json("fixture.inv", b"");
        assert_eq!(document["diagnostics"][0]["code"], "INTERNAL_SCOPE");
        let schema: serde_json::Value =
            serde_json::from_str(include_str!("../schema/invar-diagnostics-v3.schema.json"))
                .unwrap();
        assert!(jsonschema::is_valid(&schema, &document));
    }
}

pub(crate) mod floating;
mod sorting;

#[cfg(test)]
mod allocation_limit_tests {
    #[test]
    fn live_element_ceiling_is_checked_without_allocating() {
        let mut live = super::MAX_LIVE_ELEMENTS;
        assert!(super::admit_live(&mut live, 1).is_err());
        let mut live = usize::MAX;
        assert!(super::admit_live(&mut live, 1).is_err());
    }
}
