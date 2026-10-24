//! Continuation identities, oracle executions and frozen oracle results.

use super::*;

/// Opaque identity for captured task bytes; no execution claim or resume method.
#[derive(Debug)]
pub struct ContinuationIdentity {
    pub(super) artifact_id: String,
    pub(super) task: Vec<u8>,
    pub(super) manifest: Value,
    pub(super) id: String,
}
impl ContinuationIdentity {
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn manifest(&self) -> &Value {
        &self.manifest
    }
    pub fn verify_manifest(&self, manifest: &Value) -> Result<(), Error> {
        if self.manifest == *manifest {
            Ok(())
        } else {
            Err(Error::ManifestMismatch)
        }
    }
}

/// Receipt created only by an observed oracle call; compile() creates none.
#[derive(Debug)]
pub struct OracleExecution {
    pub(super) observation: oracle::Observation,
    pub(super) result: FrozenResult,
    pub(super) manifest: Value,
    pub(super) id: String,
}
impl OracleExecution {
    /// Defensive snapshot: returned array storage never aliases this receipt.
    /// Copies the bounded trace, cleanup, owned arrays and flat record result.
    pub fn observation(&self) -> oracle::Observation {
        oracle::Observation {
            semantic: eval::Observation {
                result: self.result.materialize(),
                steps_used: self.observation.semantic.steps_used,
                cleanup_trace: self
                    .observation
                    .semantic
                    .cleanup_trace
                    .iter()
                    .map(|e| eval::CleanupEvent {
                        name: e.name.clone(),
                        length: e.length,
                    })
                    .collect(),
            },
            failure: self.observation.failure,
            events: self.observation.events.clone(),
            transitions: self.observation.transitions,
            max_frames: self.observation.max_frames,
        }
    }
    pub fn manifest(&self) -> &Value {
        &self.manifest
    }
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn verify_manifest(&self, manifest: &Value) -> Result<(), Error> {
        if self.manifest == *manifest {
            Ok(())
        } else {
            Err(Error::ManifestMismatch)
        }
    }
}

#[derive(Debug)]
pub(super) enum FrozenResult {
    I32(i32),
    U32(u32),
    Bool(bool),
    F32(crate::numeric_contract::F32Bits),
    Unit,
    ArrayI32(Vec<i32>),
    ArrayU32(Vec<u32>),
    ArrayF32(Vec<crate::numeric_contract::F32Bits>),
    Record(Vec<FrozenResult>),
    Trap(eval::TrapCategory, front::Span),
    Limit(u64, u64),
    Tool(&'static str, String),
}
impl FrozenResult {
    pub(super) fn capture(result: &Result<eval::Value, eval::EvalError>) -> Result<Self, Error> {
        Ok(match result {
            Ok(eval::Value::I32(n)) => Self::I32(*n),
            Ok(eval::Value::U32(n)) => Self::U32(*n),
            Ok(eval::Value::Bool(n)) => Self::Bool(*n),
            Ok(eval::Value::F32(n)) => Self::F32(*n),
            Ok(eval::Value::Unit) => Self::Unit,
            Ok(eval::Value::ArrayI32(xs)) => Self::ArrayI32(xs.borrow().clone()),
            Ok(eval::Value::ArrayU32(xs)) => Self::ArrayU32(xs.borrow().clone()),
            Ok(eval::Value::ArrayF32(xs)) => {
                let values = xs.borrow();
                let mut frozen = Vec::new();
                frozen
                    .try_reserve_exact(values.len())
                    .map_err(|_| Error::Limit)?;
                frozen.extend_from_slice(&values);
                Self::ArrayF32(frozen)
            }
            Ok(eval::Value::Record(fields)) => {
                let fields = fields.try_borrow().map_err(|_| Error::ExecutionMismatch)?;
                if fields.len() > compiled::MAX_RECORD_FIELDS {
                    return Err(Error::Limit);
                }
                let mut frozen = Vec::new();
                frozen
                    .try_reserve_exact(fields.len())
                    .map_err(|_| Error::Limit)?;
                for value in fields.iter() {
                    frozen.push(match value {
                        eval::Value::I32(v) => Self::I32(*v),
                        eval::Value::U32(v) => Self::U32(*v),
                        eval::Value::F32(v) => Self::F32(*v),
                        eval::Value::Bool(v) => Self::Bool(*v),
                        _ => return Err(Error::Compiler(compiled::Error::UnsupportedType)),
                    });
                }
                Self::Record(frozen)
            }
            // Nested/owning aggregates and borrowed views remain unsupported.
            Ok(_) => return Err(Error::Compiler(compiled::Error::UnsupportedType)),
            Err(eval::EvalError::Trap { category, span }) => Self::Trap(*category, *span),
            Err(eval::EvalError::EvaluationLimit {
                fuel_limit,
                steps_used,
            }) => Self::Limit(*fuel_limit, *steps_used),
            Err(eval::EvalError::Tool { code, message }) => Self::Tool(code, message.clone()),
        })
    }
    pub(super) fn materialize(&self) -> Result<eval::Value, eval::EvalError> {
        use std::{cell::RefCell, rc::Rc};
        Ok(match self {
            Self::I32(n) => eval::Value::I32(*n),
            Self::U32(n) => eval::Value::U32(*n),
            Self::Bool(n) => eval::Value::Bool(*n),
            Self::F32(n) => eval::Value::F32(*n),
            Self::Unit => eval::Value::Unit,
            Self::ArrayI32(xs) => eval::Value::ArrayI32(Rc::new(RefCell::new(xs.clone()))),
            Self::ArrayU32(xs) => eval::Value::ArrayU32(Rc::new(RefCell::new(xs.clone()))),
            Self::ArrayF32(xs) => {
                let mut values = Vec::new();
                values
                    .try_reserve_exact(xs.len())
                    .map_err(|_| eval::EvalError::Tool {
                        code: "RESOURCE_LIMIT",
                        message: "numeric owner observation allocation failed".into(),
                    })?;
                values.extend_from_slice(xs);
                eval::Value::ArrayF32(Rc::new(RefCell::new(values)))
            }
            Self::Record(fields) => eval::Value::Record(Rc::new(RefCell::new(
                fields
                    .iter()
                    .map(Self::materialize)
                    .collect::<Result<Vec<_>, _>>()?,
            ))),
            Self::Trap(category, span) => {
                return Err(eval::EvalError::Trap {
                    category: *category,
                    span: *span,
                });
            }
            Self::Limit(fuel_limit, steps_used) => {
                return Err(eval::EvalError::EvaluationLimit {
                    fuel_limit: *fuel_limit,
                    steps_used: *steps_used,
                });
            }
            Self::Tool(code, message) => {
                return Err(eval::EvalError::Tool {
                    code,
                    message: message.clone(),
                });
            }
        })
    }
}
