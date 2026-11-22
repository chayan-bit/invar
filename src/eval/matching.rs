//! Evaluating `match`: the selected arm of the owner's case, trying guarded arms in order (#260).

use super::*;

impl Evaluator<'_> {
    pub(super) fn match_expr(
        &mut self,
        value: &Expr,
        arms: &[crate::front::MatchArm],
        env: &mut Env,
    ) -> Result<Value, EvalError> {
        let (name, moving) = match &value.kind {
            ExprKind::Name(name) => (name, false),
            ExprKind::Move(name) => (name, true),
            _ => {
                return Err(EvalError::tool(
                    "INTERNAL_TYPE",
                    "checked match needs a named owner",
                ));
            }
        };
        self.charge()?;
        let Value::Variant(storage) = env.borrow_owned(name)? else {
            return Err(EvalError::tool(
                "INTERNAL_TYPE",
                "checked match owner is not a variant",
            ));
        };
        let (tag, payload) = if moving {
            let mut storage = storage.borrow_mut();
            (storage.0, storage.1.take())
        } else {
            let storage = storage.borrow();
            let payload = storage
                .1
                .as_ref()
                .map(|value| match value {
                    Value::I32(v) => Ok(Value::I32(*v)),
                    Value::U32(v) => Ok(Value::U32(*v)),
                    Value::Bool(v) => Ok(Value::Bool(*v)),
                    Value::F32(v) => Ok(Value::F32(*v)),
                    Value::F64(v) => Ok(Value::F64(*v)),
                    _ => Err(EvalError::tool(
                        "INTERNAL_TYPE",
                        "borrowed owned variant payload",
                    )),
                })
                .transpose()?;
            (storage.0, payload)
        };
        let nominal = self
            .types
            .get(&value.span)
            .and_then(|ty| match ty {
                Type::Named(name) => Some(name.as_str()),
                _ => None,
            })
            .ok_or_else(|| {
                EvalError::tool("INTERNAL_TYPE", "checked match nominal type missing")
            })?;
        let declaration = self
            .program
            .program
            .variants
            .iter()
            .find(|variant| variant.name == nominal)
            .ok_or_else(|| {
                EvalError::tool("INTERNAL_TYPE", "checked variant declaration missing")
            })?;
        let case = declaration
            .arms
            .get(tag)
            .ok_or_else(|| EvalError::tool("INTERNAL_TYPE", "checked variant tag missing"))?;
        let candidates: Vec<&crate::front::MatchArm> = arms
            .iter()
            .filter(|arm| {
                arm.variant.rsplit_once("::").is_some_and(|(owner, name)| {
                    (owner == nominal
                        || (crate::front::result_parts(nominal).is_some() && owner == "Result"))
                        && name == case.name
                })
            })
            .collect();
        let (mut payload, mut chosen) = (payload, None);
        // Arms of the case in order; a false guard (#260) falls through to the next one.
        for selected in candidates {
            env.scopes.push(Scope::new());
            // Guarded arms have scalar payloads (checked), so each attempt binds a copy.
            let bound = if selected.guard.is_some() {
                payload.as_ref().and_then(copy_scalar)
            } else {
                payload.take()
            };
            if let (Some(binding), Some(bound)) = (&selected.binding, bound) {
                env.bind(binding, bound)?;
            }
            let pass = match &selected.guard {
                None => Ok(true),
                Some(guard) => self.expr(guard, env).and_then(|value| match value {
                    Value::Bool(pass) => Ok(pass),
                    _ => Err(EvalError::tool(
                        "INTERNAL_TYPE",
                        "checked match guard is not Bool",
                    )),
                }),
            };
            let result = match pass {
                Ok(false) => {
                    env.pop(&mut self.trace, &mut self.live_elements, true);
                    continue;
                }
                Ok(true) => self.expr(&selected.value, env),
                Err(error) => Err(error),
            };
            env.pop(
                &mut self.trace,
                &mut self.live_elements,
                result.is_ok() || self.pending_result_error.is_some(),
            );
            chosen = Some(result);
            break;
        }
        let result =
            chosen.ok_or_else(|| EvalError::tool("INTERNAL_TYPE", "checked match arm missing"))?;
        if moving && (result.is_ok() || self.pending_result_error.is_some()) {
            let owned = env.take(name)?;
            record_cleanup(&mut self.trace, &mut self.live_elements, name, &owned);
        }
        result
    }
}

/// A copy of a scalar value, for binding a guarded arm's payload once per attempt (#260).
fn copy_scalar(value: &Value) -> Option<Value> {
    match value {
        Value::I32(v) => Some(Value::I32(*v)),
        Value::U32(v) => Some(Value::U32(*v)),
        Value::Bool(v) => Some(Value::Bool(*v)),
        Value::F32(v) => Some(Value::F32(*v)),
        Value::F64(v) => Some(Value::F64(*v)),
        _ => None,
    }
}
