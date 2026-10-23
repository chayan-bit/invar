//! Test support, included by path: the independent call-by-value oracle for the finite term
//! fragment (`invar::term` syntax) and the demand-ordered observation of verified typed graphs.
//! Never use this module in production code.
#![allow(dead_code, unused_imports)]

use std::collections::{BTreeMap, BTreeSet};

pub use invar::term::{Binder, Function, Kind, Program, Term};
use invar::typed_net::{self, MapOp, Net, NodeId, Op, Plan, Proof};

type Environment = BTreeMap<Binder, Value>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value {
    Unit,
    Bool(bool),
    Nat(u32),
    Pair(Box<Value>, Box<Value>),
    Closure {
        binder: Binder,
        body: Box<Term>,
        captures: Environment,
    },
}

impl Value {
    fn copyable(&self) -> bool {
        match self {
            Self::Unit | Self::Bool(_) | Self::Nat(_) => true,
            Self::Pair(a, b) => a.copyable() && b.copyable(),
            Self::Closure { .. } => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    pub nodes: usize,
    pub depth: usize,
    pub steps: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    NodeLimit,
    DepthLimit(u32),
    StepLimit(u32),
    MissingBinder { at: u32, binder: Binder },
    UnknownFunction { at: u32, name: String },
    Arity { at: u32, name: String },
    Type(u32),
    Trap { at: u32, label: &'static str },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    Demand(u32),
    Complete(u32),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Observation {
    pub result: Result<Value, Error>,
    pub events: Vec<Event>,
    pub steps: usize,
}

fn children(term: &Term) -> Vec<&Term> {
    match &term.kind {
        Kind::Pair(a, b) | Kind::Apply(a, b) => vec![a, b],
        Kind::Lambda { body, .. } => vec![body],
        Kind::Let { value, body, .. } => vec![value, body],
        Kind::If { condition, yes, no } => vec![condition, yes, no],
        Kind::Call { args, .. } => args.iter().collect(),
        _ => Vec::new(),
    }
}

fn validate_size(program: &Program, entry: &Term, limits: Limits) -> Result<(), Error> {
    let mut stack = vec![(entry, 1usize)];
    stack.extend(program.functions.values().map(|f| (&f.body, 1)));
    let mut count = 0usize;
    while let Some((term, depth)) = stack.pop() {
        if depth > limits.depth {
            return Err(Error::DepthLimit(term.id));
        }
        count += 1;
        if count > limits.nodes {
            return Err(Error::NodeLimit);
        }
        stack.extend(children(term).into_iter().map(|child| (child, depth + 1)));
    }
    Ok(())
}

fn free_variables(term: &Term, bound: &mut BTreeSet<Binder>, free: &mut BTreeSet<Binder>) {
    match &term.kind {
        Kind::Var(id) => {
            if !bound.contains(id) {
                free.insert(*id);
            }
        }
        Kind::Lambda { binder, body } => {
            bound.insert(*binder);
            free_variables(body, bound, free);
            bound.remove(binder);
        }
        Kind::Let {
            binder,
            value,
            body,
        } => {
            free_variables(value, bound, free);
            bound.insert(*binder);
            free_variables(body, bound, free);
            bound.remove(binder);
        }
        _ => {
            for child in children(term) {
                free_variables(child, bound, free);
            }
        }
    }
}

struct Machine<'a> {
    program: &'a Program,
    limits: Limits,
    events: Vec<Event>,
    steps: usize,
}

impl Machine<'_> {
    fn evaluate(
        &mut self,
        term: &Term,
        env: &mut Environment,
        depth: usize,
    ) -> Result<Value, Error> {
        self.events.push(Event::Demand(term.id));
        if depth > self.limits.depth {
            return Err(Error::DepthLimit(term.id));
        }
        if self.steps == self.limits.steps {
            return Err(Error::StepLimit(term.id));
        }
        self.steps += 1;
        let result = match &term.kind {
            Kind::Unit => Value::Unit,
            Kind::Bool(value) => Value::Bool(*value),
            Kind::Nat(value) => Value::Nat(*value),
            Kind::Pair(left, right) => Value::Pair(
                Box::new(self.evaluate(left, env, depth + 1)?),
                Box::new(self.evaluate(right, env, depth + 1)?),
            ),
            Kind::Var(binder) => {
                let value = env.get(binder).ok_or(Error::MissingBinder {
                    at: term.id,
                    binder: *binder,
                })?;
                if value.copyable() {
                    value.clone()
                } else {
                    env.remove(binder).ok_or(Error::MissingBinder {
                        at: term.id,
                        binder: *binder,
                    })?
                }
            }
            Kind::Lambda { binder, body } => {
                let mut free = BTreeSet::new();
                let mut bound = BTreeSet::from([*binder]);
                free_variables(body, &mut bound, &mut free);
                let mut captures = Environment::new();
                for id in free {
                    let value = env.get(&id).ok_or(Error::MissingBinder {
                        at: term.id,
                        binder: id,
                    })?;
                    let captured = if value.copyable() {
                        value.clone()
                    } else {
                        env.remove(&id).ok_or(Error::MissingBinder {
                            at: term.id,
                            binder: id,
                        })?
                    };
                    captures.insert(id, captured);
                }
                Value::Closure {
                    binder: *binder,
                    body: body.clone(),
                    captures,
                }
            }
            Kind::Apply(function, argument) => {
                let function = self.evaluate(function, env, depth + 1)?;
                let argument = self.evaluate(argument, env, depth + 1)?;
                let Value::Closure {
                    binder,
                    body,
                    mut captures,
                } = function
                else {
                    return Err(Error::Type(term.id));
                };
                captures.insert(binder, argument);
                self.evaluate(&body, &mut captures, depth + 1)?
            }
            Kind::Let {
                binder,
                value,
                body,
            } => {
                let value = self.evaluate(value, env, depth + 1)?;
                let previous = env.insert(*binder, value);
                let result = self.evaluate(body, env, depth + 1);
                env.remove(binder);
                if let Some(previous) = previous {
                    env.insert(*binder, previous);
                }
                result?
            }
            Kind::If { condition, yes, no } => {
                let condition = self.evaluate(condition, env, depth + 1)?;
                let Value::Bool(condition) = condition else {
                    return Err(Error::Type(term.id));
                };
                self.evaluate(if condition { yes } else { no }, env, depth + 1)?
            }
            Kind::Call { name, args } => {
                let mut values = Vec::new();
                for arg in args {
                    values.push(self.evaluate(arg, env, depth + 1)?);
                }
                let function =
                    self.program
                        .functions
                        .get(name)
                        .ok_or_else(|| Error::UnknownFunction {
                            at: term.id,
                            name: name.clone(),
                        })?;
                if values.len() != function.params.len() {
                    return Err(Error::Arity {
                        at: term.id,
                        name: name.clone(),
                    });
                }
                let mut callee = function.params.iter().copied().zip(values).collect();
                self.evaluate(&function.body, &mut callee, depth + 1)?
            }
            Kind::Fail(label) => {
                return Err(Error::Trap { at: term.id, label });
            }
        };
        self.events.push(Event::Complete(term.id));
        Ok(result)
    }
}

/// Evaluates the bounded internal fragment and records every attempted demand.
pub fn observe(program: &Program, entry: &Term, limits: Limits) -> Observation {
    if let Err(error) = validate_size(program, entry, limits) {
        return Observation {
            result: Err(error),
            events: Vec::new(),
            steps: 0,
        };
    }
    let mut machine = Machine {
        program,
        limits,
        events: Vec::new(),
        steps: 0,
    };
    let result = machine.evaluate(entry, &mut Environment::new(), 1);
    Observation {
        result,
        events: machine.events,
        steps: machine.steps,
    }
}

#[derive(Clone, Debug)]
enum RuntimeValue {
    Unit,
    Bool(bool),
    Nat(u32),
    Pair(Box<Self>, Box<Self>),
    Closure {
        binder: Binder,
        body: NodeId,
        captures: BTreeMap<Binder, Self>,
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

    fn into_term(self, net: &Net) -> Result<Value, typed_net::Error> {
        Ok(match self {
            Self::Unit => Value::Unit,
            Self::Bool(value) => Value::Bool(value),
            Self::Nat(value) => Value::Nat(value),
            Self::Pair(a, b) => {
                Value::Pair(Box::new(a.into_term(net)?), Box::new(b.into_term(net)?))
            }
            Self::Closure {
                binder,
                body,
                captures,
            } => Value::Closure {
                binder,
                body: Box::new(net.decode_term(body)?),
                captures: captures
                    .into_iter()
                    .map(|(id, value)| value.into_term(net).map(|value| (id, value)))
                    .collect::<Result<_, _>>()?,
            },
            Self::ArrayU32(_) => return Err(typed_net::Error::Type(net.entry)),
        })
    }
}

struct NetMachine<'a> {
    net: &'a Net,
    limits: Limits,
    events: Vec<Event>,
    steps: usize,
}

impl NetMachine<'_> {
    fn evaluate(
        &mut self,
        id: NodeId,
        env: &mut BTreeMap<Binder, RuntimeValue>,
        depth: usize,
    ) -> Result<RuntimeValue, Error> {
        let node = node(self.net, id);
        self.events.push(Event::Demand(node.source_id));
        if depth > self.limits.depth {
            return Err(Error::DepthLimit(node.source_id));
        }
        if self.steps == self.limits.steps {
            return Err(Error::StepLimit(node.source_id));
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
                return Err(Error::Type(node.source_id));
            }
            Op::Pair => RuntimeValue::Pair(
                Box::new(self.evaluate(node.inputs[0], env, depth + 1)?),
                Box::new(self.evaluate(node.inputs[1], env, depth + 1)?),
            ),
            Op::Var(binder) => {
                let value = env.get(binder).ok_or(Error::MissingBinder {
                    at: node.source_id,
                    binder: *binder,
                })?;
                if value.copyable() {
                    value.clone()
                } else {
                    env.remove(binder).ok_or(Error::MissingBinder {
                        at: node.source_id,
                        binder: *binder,
                    })?
                }
            }
            Op::Lambda { binder } => {
                let mut free = BTreeSet::new();
                free_variables_of(
                    self.net,
                    node.inputs[0],
                    &mut BTreeSet::from([*binder]),
                    &mut free,
                );
                let mut captures = BTreeMap::new();
                for name in free {
                    let value = env.get(&name).ok_or(Error::MissingBinder {
                        at: node.source_id,
                        binder: name,
                    })?;
                    let value = if value.copyable() {
                        value.clone()
                    } else {
                        env.remove(&name).ok_or(Error::MissingBinder {
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
                    return Err(Error::Type(node.source_id));
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
                    return Err(Error::Type(node.source_id));
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
                        .ok_or_else(|| Error::UnknownFunction {
                            at: node.source_id,
                            name: name.clone(),
                        })?;
                if values.len() != function.signature.params.len() {
                    return Err(Error::Arity {
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
                return Err(Error::Trap {
                    at: node.source_id,
                    label,
                });
            }
            Op::Map(op) => {
                let RuntimeValue::ArrayU32(values) =
                    self.evaluate(node.inputs[0], env, depth + 1)?
                else {
                    return Err(Error::Type(node.source_id));
                };
                RuntimeValue::ArrayU32(values.into_iter().map(|v| op.apply(v)).collect())
            }
            Op::FusedMap(ops) => {
                let RuntimeValue::ArrayU32(values) =
                    self.evaluate(node.inputs[0], env, depth + 1)?
                else {
                    return Err(Error::Type(node.source_id));
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
                    return Err(Error::Type(node.source_id));
                };
                let RuntimeValue::ArrayU32(right) =
                    self.evaluate(node.inputs[1], env, depth + 1)?
                else {
                    return Err(Error::Type(node.source_id));
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
        self.events.push(Event::Complete(node.source_id));
        Ok(value)
    }
}

fn node(net: &Net, id: NodeId) -> &typed_net::Node {
    net.nodes
        .get(id)
        .and_then(Option::as_ref)
        .expect("verified node")
}

/// Independent free-variable scan over graph nodes (the library has its own for beta rewriting).
fn free_variables_of(
    net: &Net,
    id: NodeId,
    bound: &mut BTreeSet<Binder>,
    free: &mut BTreeSet<Binder>,
) {
    let n = node(net, id);
    match n.op {
        Op::Var(binder) => {
            if !bound.contains(&binder) {
                free.insert(binder);
            }
        }
        Op::Lambda { binder } => {
            bound.insert(binder);
            free_variables_of(net, n.inputs[0], bound, free);
            bound.remove(&binder);
        }
        Op::Let { binder } => {
            free_variables_of(net, n.inputs[0], bound, free);
            bound.insert(binder);
            free_variables_of(net, n.inputs[1], bound, free);
            bound.remove(&binder);
        }
        _ => {
            for &child in &n.inputs {
                free_variables_of(net, child, bound, free);
            }
        }
    }
}

fn preflight_observation(net: &Net, limits: Limits) -> Option<Error> {
    let mut pending = vec![(net.entry, 1usize)];
    pending.extend(net.functions.values().map(|function| (function.body, 1)));
    let mut count = 0usize;
    while let Some((id, depth)) = pending.pop() {
        let n = node(net, id);
        if depth > limits.depth {
            return Some(Error::DepthLimit(n.source_id));
        }
        count += 1;
        if count > limits.nodes {
            return Some(Error::NodeLimit);
        }
        pending.extend(n.inputs.iter().map(|&child| (child, depth + 1)));
    }
    None
}

/// Demand-ordered observation of a verified term graph.
pub fn observe_net(
    net: &Net,
    proof: &Proof,
    limits: Limits,
) -> Result<Observation, typed_net::Error> {
    net.check_proof(proof)?;
    if let Some(error) = preflight_observation(net, limits) {
        return Ok(Observation {
            result: Err(error),
            events: Vec::new(),
            steps: 0,
        });
    }
    let mut machine = NetMachine {
        net,
        limits,
        events: Vec::new(),
        steps: 0,
    };
    let result = machine.evaluate(net.entry, &mut BTreeMap::new(), 1);
    let result = match result {
        Ok(value) => Ok(value.into_term(net)?),
        Err(error) => Err(error),
    };
    Ok(Observation {
        result,
        events: machine.events,
        steps: machine.steps,
    })
}

/// Demand-ordered observation of a verified array graph.
pub fn observe_array_net(
    net: &Net,
    proof: &Proof,
    limits: Limits,
) -> Result<ArrayObservation, typed_net::Error> {
    net.check_proof(proof)?;
    if let Some(error) = preflight_observation(net, limits) {
        return Ok(ArrayObservation {
            result: Err(error),
            events: Vec::new(),
            steps: 0,
        });
    }
    let mut machine = NetMachine {
        net,
        limits,
        events: Vec::new(),
        steps: 0,
    };
    let result = machine.evaluate(net.entry, &mut BTreeMap::new(), 1);
    let result = match result {
        Ok(RuntimeValue::ArrayU32(values)) => Ok(values),
        Ok(_) => return Err(typed_net::Error::Type(net.entry)),
        Err(error) => Err(error),
    };
    Ok(ArrayObservation {
        result,
        events: machine.events,
        steps: machine.steps,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArrayObservation {
    pub result: Result<Vec<u32>, Error>,
    pub events: Vec<Event>,
    pub steps: usize,
}

/// Runs the term carried by an extracted `Plan::Term`.
pub fn run_term(plan: &Plan, limits: Limits) -> Result<Observation, typed_net::Error> {
    let Plan::Term { program, entry, .. } = plan else {
        return Err(typed_net::Error::Type(0));
    };
    Ok(observe(program, entry, limits))
}
