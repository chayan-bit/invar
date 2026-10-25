//! A dedicated, documented evaluation stack for the recursive reference engine.
//!
//! The evaluator walks source trees recursively: each Invar call costs several
//! host frames (about 50 KiB in an unoptimized build, about 2.6 KiB optimized),
//! and an expression of the checked maximum height inside one call costs up to
//! [`PER_CALL_STACK_BYTES`]. Running on the caller's thread let a default 2 MiB
//! thread overflow (a process abort) before the 64-call `CALL_DEPTH_LIMIT`.
//! Every root evaluation therefore runs on its own thread whose stack is sized
//! from the documented limits, so the typed limit is always reached first.
//! `mir_eval::observe` runs its recursive MIR interpreter on the same stack.

use super::{EvalError, MAX_CALL_DEPTH, Observation, Value, ViewRange};
use crate::numeric_contract::{F32Bits, F64Bits};
use std::{cell::RefCell, rc::Rc};

/// Upper bound for one call frame plus a full-height expression tree; checked
/// trees of the maximum height are stack-safe on 2 MiB (`robustness_limits`).
pub(crate) const PER_CALL_STACK_BYTES: usize = 2 * 1024 * 1024;
/// The root frame plus `MAX_CALL_DEPTH` nested calls. This is a virtual
/// reservation; only touched pages are committed.
pub(crate) const STACK_BYTES: usize = (MAX_CALL_DEPTH + 1) * PER_CALL_STACK_BYTES;

/// A thread-transferable copy of a [`Value`]. Shared storage is copied: a
/// root observation is returned by value, so no caller observes aliasing.
enum Portable {
    I32(i32),
    U32(u32),
    F32(F32Bits),
    F64(F64Bits),
    Bool(bool),
    Unit,
    ArrayI32(Vec<i32>),
    ArrayU32(Vec<u32>),
    ArrayF32(Vec<F32Bits>),
    ArrayF64(Vec<F64Bits>),
    ArrayBool(Vec<bool>),
    Bytes(Vec<u8>),
    Text(String),
    ViewI32(Vec<i32>, ViewRange),
    ViewU32(Vec<u32>, ViewRange),
    VoxelGrid(crate::spatial::Grid),
    Record(Vec<Portable>),
    Variant(usize, Option<Box<Portable>>),
}

impl Portable {
    /// Iterative, so a deeply nested value never recurses on the host stack.
    fn from_value(value: Value) -> Self {
        enum Task {
            Visit(Value),
            Record(usize),
            Variant(usize, bool),
        }
        let mut pending = vec![Task::Visit(value)];
        let mut done: Vec<Portable> = Vec::new();
        while let Some(task) = pending.pop() {
            match task {
                Task::Visit(value) => done.push(match value {
                    Value::I32(x) => Self::I32(x),
                    Value::U32(x) => Self::U32(x),
                    Value::F32(x) => Self::F32(x),
                    Value::F64(x) => Self::F64(x),
                    Value::Bool(x) => Self::Bool(x),
                    Value::Unit => Self::Unit,
                    Value::ArrayI32(v) => Self::ArrayI32(v.borrow().clone()),
                    Value::ArrayU32(v) => Self::ArrayU32(v.borrow().clone()),
                    Value::ArrayF32(v) => Self::ArrayF32(v.borrow().clone()),
                    Value::ArrayF64(v) => Self::ArrayF64(v.borrow().clone()),
                    Value::ArrayBool(v) => Self::ArrayBool(v.borrow().clone()),
                    Value::Bytes(v) => Self::Bytes(v.as_ref().clone()),
                    Value::Text(v) => Self::Text(v.as_ref().clone()),
                    Value::ViewI32(v, range) => Self::ViewI32(v.borrow().clone(), range),
                    Value::ViewU32(v, range) => Self::ViewU32(v.borrow().clone(), range),
                    Value::VoxelGrid(grid) => Self::VoxelGrid(grid.as_ref().clone()),
                    Value::Record(fields) => {
                        let fields = fields.borrow();
                        pending.push(Task::Record(fields.len()));
                        pending.extend(fields.iter().rev().map(|f| Task::Visit(share(f))));
                        continue;
                    }
                    Value::Variant(cell) => {
                        let (tag, payload) = &*cell.borrow();
                        pending.push(Task::Variant(*tag, payload.is_some()));
                        pending.extend(payload.as_ref().map(|p| Task::Visit(share(p))));
                        continue;
                    }
                }),
                Task::Record(length) => {
                    let fields = done.split_off(done.len() - length);
                    done.push(Self::Record(fields));
                }
                Task::Variant(tag, has_payload) => {
                    let payload = has_payload.then(|| done.pop().map(Box::new)).flatten();
                    done.push(Self::Variant(tag, payload));
                }
            }
        }
        done.pop().unwrap_or(Self::Unit)
    }

    fn into_value(self) -> Value {
        enum Task {
            Visit(Portable),
            Record(usize),
            Variant(usize, bool),
        }
        let mut pending = vec![Task::Visit(self)];
        let mut done: Vec<Value> = Vec::new();
        while let Some(task) = pending.pop() {
            match task {
                Task::Visit(value) => done.push(match value {
                    Self::I32(x) => Value::I32(x),
                    Self::U32(x) => Value::U32(x),
                    Self::F32(x) => Value::F32(x),
                    Self::F64(x) => Value::F64(x),
                    Self::Bool(x) => Value::Bool(x),
                    Self::Unit => Value::Unit,
                    Self::ArrayI32(v) => Value::ArrayI32(Rc::new(RefCell::new(v))),
                    Self::ArrayU32(v) => Value::ArrayU32(Rc::new(RefCell::new(v))),
                    Self::ArrayF32(v) => Value::ArrayF32(Rc::new(RefCell::new(v))),
                    Self::ArrayF64(v) => Value::ArrayF64(Rc::new(RefCell::new(v))),
                    Self::ArrayBool(v) => Value::ArrayBool(Rc::new(RefCell::new(v))),
                    Self::Bytes(v) => Value::Bytes(Rc::new(v)),
                    Self::Text(v) => Value::Text(Rc::new(v)),
                    Self::ViewI32(v, range) => Value::ViewI32(Rc::new(RefCell::new(v)), range),
                    Self::ViewU32(v, range) => Value::ViewU32(Rc::new(RefCell::new(v)), range),
                    Self::VoxelGrid(grid) => Value::VoxelGrid(Rc::new(grid)),
                    Self::Record(fields) => {
                        pending.push(Task::Record(fields.len()));
                        pending.extend(fields.into_iter().rev().map(Task::Visit));
                        continue;
                    }
                    Self::Variant(tag, payload) => {
                        pending.push(Task::Variant(tag, payload.is_some()));
                        pending.extend(payload.map(|p| Task::Visit(*p)));
                        continue;
                    }
                }),
                Task::Record(length) => {
                    let fields = done.split_off(done.len() - length);
                    done.push(Value::Record(Rc::new(RefCell::new(fields))));
                }
                Task::Variant(tag, has_payload) => {
                    let payload = has_payload.then(|| done.pop()).flatten();
                    done.push(Value::Variant(Rc::new(RefCell::new((tag, payload)))));
                }
            }
        }
        done.pop().unwrap_or(Value::Unit)
    }
}

/// Root arguments carried onto the evaluation thread.
pub(crate) struct Values(Vec<Portable>);

impl Values {
    pub(crate) fn new(values: Vec<Value>) -> Self {
        Self(values.into_iter().map(Portable::from_value).collect())
    }

    pub(crate) fn into_values(self) -> Vec<Value> {
        self.0.into_iter().map(Portable::into_value).collect()
    }
}

/// A shallow handle copy: the same storage, never a deep clone.
fn share(value: &Value) -> Value {
    match value {
        Value::I32(x) => Value::I32(*x),
        Value::U32(x) => Value::U32(*x),
        Value::F32(x) => Value::F32(*x),
        Value::F64(x) => Value::F64(*x),
        Value::Bool(x) => Value::Bool(*x),
        Value::Unit => Value::Unit,
        Value::ArrayI32(v) => Value::ArrayI32(Rc::clone(v)),
        Value::ArrayU32(v) => Value::ArrayU32(Rc::clone(v)),
        Value::ArrayF32(v) => Value::ArrayF32(Rc::clone(v)),
        Value::ArrayF64(v) => Value::ArrayF64(Rc::clone(v)),
        Value::ArrayBool(v) => Value::ArrayBool(Rc::clone(v)),
        Value::Bytes(v) => Value::Bytes(Rc::clone(v)),
        Value::Text(v) => Value::Text(Rc::clone(v)),
        Value::ViewI32(v, range) => Value::ViewI32(Rc::clone(v), *range),
        Value::ViewU32(v, range) => Value::ViewU32(Rc::clone(v), *range),
        Value::VoxelGrid(grid) => Value::VoxelGrid(Rc::clone(grid)),
        Value::Record(v) => Value::Record(Rc::clone(v)),
        Value::Variant(v) => Value::Variant(Rc::clone(v)),
    }
}

/// Run one root evaluation on a dedicated thread with [`STACK_BYTES`] of stack.
/// A panic inside resumes on the caller with its original payload. If the
/// operating system cannot create the thread, the evaluation never starts and
/// a typed `RESOURCE_LIMIT` with zero steps is reported.
pub(crate) fn observe_on_own_stack(evaluate: impl FnOnce() -> Observation + Send) -> Observation {
    let outcome = std::thread::scope(|scope| {
        let worker = std::thread::Builder::new()
            .name("invar-eval".into())
            .stack_size(STACK_BYTES)
            .spawn_scoped(scope, || {
                let observation = evaluate();
                (
                    observation.result.map(Portable::from_value),
                    observation.steps_used,
                    observation.cleanup_trace,
                )
            });
        worker.map(|handle| handle.join())
    });
    match outcome {
        Ok(Ok((result, steps_used, cleanup_trace))) => Observation {
            result: result.map(Portable::into_value),
            steps_used,
            cleanup_trace,
        },
        Ok(Err(panic)) => std::panic::resume_unwind(panic),
        Err(_) => Observation {
            result: Err(EvalError::tool(
                "RESOURCE_LIMIT",
                "reference evaluator stack thread unavailable",
            )),
            steps_used: 0,
            cleanup_trace: Vec::new(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn issue_eval_stack_portable_round_trip_preserves_nested_values() {
        let value = Value::Record(Rc::new(RefCell::new(vec![
            Value::I32(-3),
            Value::ArrayU32(Rc::new(RefCell::new(vec![1, 2]))),
            Value::Variant(Rc::new(RefCell::new((
                1,
                Some(Value::Text(Rc::new("é".into()))),
            )))),
            Value::ViewI32(
                Rc::new(RefCell::new(vec![4, 5, 6])),
                ViewRange {
                    start: 1,
                    len: 2,
                    stride: 1,
                    mutable: false,
                },
            ),
            Value::Bytes(Rc::new(vec![255])),
            Value::F32(F32Bits::NEG_ZERO),
        ])));
        let expected = format!("{value:?}");
        assert_eq!(
            format!("{:?}", Portable::from_value(value).into_value()),
            expected
        );
        let mut deep = Value::Unit;
        for tag in 0..100_000 {
            deep = Value::Variant(Rc::new(RefCell::new((tag % 3, Some(deep)))));
        }
        let back = Portable::from_value(share(&deep)).into_value();
        let mut cursor = share(&back);
        let mut depth = 0usize;
        while let Value::Variant(cell) = cursor {
            let (tag, payload) = &*cell.borrow();
            assert_eq!(*tag, (99_999 - depth) % 3);
            let next = payload.as_ref().map(share).unwrap_or(Value::Unit);
            cursor = next;
            depth += 1;
        }
        assert_eq!((depth, cursor), (100_000, Value::Unit));
        // Dropping a 100k-deep Rc chain is recursive in Rust itself; keep the
        // test about conversion, not about host drop depth.
        std::mem::forget((back, deep));
    }

    #[test]
    fn issue_eval_stack_panics_resume_on_the_caller() {
        let caught =
            std::panic::catch_unwind(|| observe_on_own_stack(|| panic!("evaluator panic payload")))
                .unwrap_err();
        assert_eq!(
            caught.downcast_ref::<&str>(),
            Some(&"evaluator panic payload")
        );
    }
}
