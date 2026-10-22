//! Demand-ordered observation of verified graphs.

use std::collections::{BTreeMap, BTreeSet};

use super::*;

#[derive(Clone, Debug)]
enum RuntimeValue {
    Unit,
    Bool(bool),
    Nat(u32),
    Pair(Box<Self>, Box<Self>),
    Closure {
        binder: term::Binder,
        body: NodeId,
        captures: BTreeMap<term::Binder, Self>,
    },
    ArrayU32(Vec<u32>),
}

impl RuntimeValue {
    fn copyable(&self) -> bool {
        match self {
            Self::Unit | Self::Bool(_) | Self::Nat(_) => true,
            Self::Pair(a, b) => a.copyable() && b.copyable(),
            Self::Closure { .. } | Self::ArrayU32(_) => false,
        }
    }

    fn into_term(self, net: &Net) -> Result<term::Value, Error> {
        Ok(match self {
            Self::Unit => term::Value::Unit,
            Self::Bool(value) => term::Value::Bool(value),
            Self::Nat(value) => term::Value::Nat(value),
            Self::Pair(a, b) => {
                term::Value::Pair(Box::new(a.into_term(net)?), Box::new(b.into_term(net)?))
            }
            Self::Closure {
                binder,
                body,
                captures,
            } => term::Value::Closure {
                binder,
                body: Box::new(net.decode_term(body)?),
                captures: captures
                    .into_iter()
                    .map(|(id, value)| value.into_term(net).map(|value| (id, value)))
                    .collect::<Result<_, _>>()?,
            },
            Self::ArrayU32(_) => return Err(Error::Type(net.entry)),
        })
    }
}

struct Machine<'a> {
    net: &'a Net,
    limits: term::Limits,
    events: Vec<term::Event>,
    steps: usize,
}

impl Machine<'_> {
    fn evaluate(
        &mut self,
        id: NodeId,
        env: &mut BTreeMap<term::Binder, RuntimeValue>,
        depth: usize,
    ) -> Result<RuntimeValue, term::Error> {
        let node = self.net.node(id).expect("verified node");
        self.events.push(term::Event::Demand(node.source_id));
        if depth > self.limits.depth {
            return Err(term::Error::DepthLimit(node.source_id));
        }
        if self.steps == self.limits.steps {
            return Err(term::Error::StepLimit(node.source_id));
        }
        self.steps += 1;
        let value = match &node.op {
            Op::Unit => RuntimeValue::Unit,
            Op::Bool(value) => RuntimeValue::Bool(*value),
            Op::Nat(value) => RuntimeValue::Nat(*value),
            Op::ArrayU32(values) => RuntimeValue::ArrayU32(values.clone()),
            Op::InputU32(_)
            | Op::InputScalar(_)
            | Op::PairReduce { .. }
            | Op::Reduce { .. }
            | Op::BoundedScan { .. }
            | Op::AffineScan { .. } => {
                return Err(term::Error::Type(node.source_id));
            }
            Op::Pair => RuntimeValue::Pair(
                Box::new(self.evaluate(node.inputs[0], env, depth + 1)?),
                Box::new(self.evaluate(node.inputs[1], env, depth + 1)?),
            ),
            Op::Var(binder) => {
                let value = env.get(binder).ok_or(term::Error::MissingBinder {
                    at: node.source_id,
                    binder: *binder,
                })?;
                if value.copyable() {
                    value.clone()
                } else {
                    env.remove(binder).ok_or(term::Error::MissingBinder {
                        at: node.source_id,
                        binder: *binder,
                    })?
                }
            }
            Op::Lambda { binder } => {
                let mut free = BTreeSet::new();
                self.net
                    .free_variables(node.inputs[0], &mut BTreeSet::from([*binder]), &mut free);
                let mut captures = BTreeMap::new();
                for name in free {
                    let value = env.get(&name).ok_or(term::Error::MissingBinder {
                        at: node.source_id,
                        binder: name,
                    })?;
                    let value = if value.copyable() {
                        value.clone()
                    } else {
                        env.remove(&name).ok_or(term::Error::MissingBinder {
                            at: node.source_id,
                            binder: name,
                        })?
                    };
                    captures.insert(name, value);
                }
                RuntimeValue::Closure {
                    binder: *binder,
                    body: node.inputs[0],
                    captures,
                }
            }
            Op::Apply => {
                let function = self.evaluate(node.inputs[0], env, depth + 1)?;
                let argument = self.evaluate(node.inputs[1], env, depth + 1)?;
                let RuntimeValue::Closure {
                    binder,
                    body,
                    mut captures,
                } = function
                else {
                    return Err(term::Error::Type(node.source_id));
                };
                captures.insert(binder, argument);
                self.evaluate(body, &mut captures, depth + 1)?
            }
            Op::Let { binder } => {
                let value = self.evaluate(node.inputs[0], env, depth + 1)?;
                let previous = env.insert(*binder, value);
                let result = self.evaluate(node.inputs[1], env, depth + 1);
                env.remove(binder);
                if let Some(previous) = previous {
                    env.insert(*binder, previous);
                }
                result?
            }
            Op::Select => {
                let condition = self.evaluate(node.inputs[0], env, depth + 1)?;
                let RuntimeValue::Bool(condition) = condition else {
                    return Err(term::Error::Type(node.source_id));
                };
                self.evaluate(node.inputs[if condition { 1 } else { 2 }], env, depth + 1)?
            }
            Op::Call { name } => {
                let mut values = Vec::new();
                for &input in &node.inputs {
                    values.push(self.evaluate(input, env, depth + 1)?);
                }
                let function =
                    self.net
                        .functions
                        .get(name)
                        .ok_or_else(|| term::Error::UnknownFunction {
                            at: node.source_id,
                            name: name.clone(),
                        })?;
                if values.len() != function.signature.params.len() {
                    return Err(term::Error::Arity {
                        at: node.source_id,
                        name: name.clone(),
                    });
                }
                let mut callee = function
                    .signature
                    .params
                    .iter()
                    .map(|(binder, _)| *binder)
                    .zip(values)
                    .collect();
                self.evaluate(function.body, &mut callee, depth + 1)?
            }
            Op::Fail(label) => {
                return Err(term::Error::Trap {
                    at: node.source_id,
                    label,
                });
            }
            Op::Map(op) => {
                let RuntimeValue::ArrayU32(values) =
                    self.evaluate(node.inputs[0], env, depth + 1)?
                else {
                    return Err(term::Error::Type(node.source_id));
                };
                RuntimeValue::ArrayU32(values.into_iter().map(|v| op.apply(v)).collect())
            }
            Op::FusedMap(ops) => {
                let RuntimeValue::ArrayU32(values) =
                    self.evaluate(node.inputs[0], env, depth + 1)?
                else {
                    return Err(term::Error::Type(node.source_id));
                };
                RuntimeValue::ArrayU32(
                    values
                        .into_iter()
                        .map(|mut v| {
                            for op in ops {
                                v = op.apply(v);
                            }
                            v
                        })
                        .collect(),
                )
            }
            Op::ZipMap { op, len } | Op::FusedZipMap { op, len, .. } => {
                let RuntimeValue::ArrayU32(left) = self.evaluate(node.inputs[0], env, depth + 1)?
                else {
                    return Err(term::Error::Type(node.source_id));
                };
                let RuntimeValue::ArrayU32(right) =
                    self.evaluate(node.inputs[1], env, depth + 1)?
                else {
                    return Err(term::Error::Type(node.source_id));
                };
                let (before_left, before_right, suffix) = match &node.op {
                    Op::FusedZipMap {
                        left,
                        right,
                        suffix,
                        ..
                    } => (left.as_slice(), right.as_slice(), suffix.as_slice()),
                    _ => (&[][..], &[][..], &[][..]),
                };
                let prepare = |value: u32, ops: &[MapOp]| {
                    ops.iter()
                        .fold(value, |value, operation| operation.apply(value))
                };
                RuntimeValue::ArrayU32(
                    left.into_iter()
                        .zip(right)
                        .take(*len)
                        .map(|(a, b)| {
                            let mut value =
                                op.apply(prepare(a, before_left), prepare(b, before_right));
                            for operation in suffix {
                                value = operation.apply(value);
                            }
                            value
                        })
                        .collect(),
                )
            }
        };
        self.events.push(term::Event::Complete(node.source_id));
        Ok(value)
    }
}

impl Net {
    fn preflight_observation(&self, limits: term::Limits) -> Option<term::Error> {
        let mut pending = vec![(self.entry, 1usize)];
        pending.extend(self.functions.values().map(|function| (function.body, 1)));
        let mut count = 0usize;
        while let Some((id, depth)) = pending.pop() {
            let node = self.node(id).expect("verified node");
            if depth > limits.depth {
                return Some(term::Error::DepthLimit(node.source_id));
            }
            count += 1;
            if count > limits.nodes {
                return Some(term::Error::NodeLimit);
            }
            pending.extend(node.inputs.iter().map(|&child| (child, depth + 1)));
        }
        None
    }

    pub fn observe(&self, proof: &Proof, limits: term::Limits) -> Result<term::Observation, Error> {
        self.check_proof(proof)?;
        if let Some(error) = self.preflight_observation(limits) {
            return Ok(term::Observation {
                result: Err(error),
                events: Vec::new(),
                steps: 0,
            });
        }
        let mut machine = Machine {
            net: self,
            limits,
            events: Vec::new(),
            steps: 0,
        };
        let result = machine.evaluate(self.entry, &mut BTreeMap::new(), 1);
        let result = match result {
            Ok(value) => Ok(value.into_term(self)?),
            Err(error) => Err(error),
        };
        Ok(term::Observation {
            result,
            events: machine.events,
            steps: machine.steps,
        })
    }

    pub fn observe_array(
        &self,
        proof: &Proof,
        limits: term::Limits,
    ) -> Result<ArrayObservation, Error> {
        self.check_proof(proof)?;
        if let Some(error) = self.preflight_observation(limits) {
            return Ok(ArrayObservation {
                result: Err(error),
                events: Vec::new(),
                steps: 0,
            });
        }
        let mut machine = Machine {
            net: self,
            limits,
            events: Vec::new(),
            steps: 0,
        };
        let result = machine.evaluate(self.entry, &mut BTreeMap::new(), 1);
        let result = match result {
            Ok(RuntimeValue::ArrayU32(values)) => Ok(values),
            Ok(_) => return Err(Error::Type(self.entry)),
            Err(error) => Err(error),
        };
        Ok(ArrayObservation {
            result,
            events: machine.events,
            steps: machine.steps,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArrayObservation {
    pub result: Result<Vec<u32>, term::Error>,
    pub events: Vec<term::Event>,
    pub steps: usize,
}
