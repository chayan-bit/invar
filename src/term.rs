//! Independent call-by-value oracle for the finite internal interaction fragment.
//!
//! This evaluates terms directly. It does not encode or reduce a net.

use std::collections::{BTreeMap, BTreeSet};

pub type Binder = u32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Term {
    pub id: u32,
    pub kind: Kind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    Unit,
    Bool(bool),
    Nat(u32),
    Pair(Box<Term>, Box<Term>),
    Var(Binder),
    Lambda {
        binder: Binder,
        body: Box<Term>,
    },
    Apply(Box<Term>, Box<Term>),
    Let {
        binder: Binder,
        value: Box<Term>,
        body: Box<Term>,
    },
    If {
        condition: Box<Term>,
        yes: Box<Term>,
        no: Box<Term>,
    },
    Call {
        name: String,
        args: Vec<Term>,
    },
    /// Internal provenance fixture; no public Invar source syntax.
    Fail(&'static str),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Function {
    pub params: Vec<Binder>,
    pub body: Term,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Program {
    pub functions: BTreeMap<String, Function>,
}

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

#[cfg(test)]
mod tests {
    use super::{Error, Event, Function, Kind, Limits, Program, Term, Value, observe};

    const LIMITS: Limits = Limits {
        nodes: 64,
        depth: 16,
        steps: 32,
    };

    fn t(id: u32, kind: Kind) -> Term {
        Term { id, kind }
    }

    fn call(id: u32, name: &str, args: Vec<Term>) -> Term {
        t(
            id,
            Kind::Call {
                name: name.into(),
                args,
            },
        )
    }

    fn spin() -> Program {
        Program {
            functions: [(
                "spin".into(),
                Function {
                    params: vec![],
                    body: call(90, "spin", vec![]),
                },
            )]
            .into(),
        }
    }

    #[test]
    fn issue_m1_04_term_oracle_preserves_demand_and_provenance() {
        // A discarded argument remains demanded under call-by-value.
        let discarded = t(
            0,
            Kind::Apply(
                Box::new(t(
                    1,
                    Kind::Lambda {
                        binder: 10,
                        body: Box::new(t(2, Kind::Nat(7))),
                    },
                )),
                Box::new(call(3, "spin", vec![])),
            ),
        );
        let observation = observe(&spin(), &discarded, Limits { steps: 7, ..LIMITS });
        assert_eq!(observation.result, Err(Error::StepLimit(90)));
        assert_eq!(
            &observation.events[..4],
            &[
                Event::Demand(0),
                Event::Demand(1),
                Event::Complete(1),
                Event::Demand(3),
            ]
        );
        assert!(!observation.events.contains(&Event::Demand(2)));

        // The unselected branch is never demanded, even if divergent.
        let branch = t(
            4,
            Kind::If {
                condition: Box::new(t(5, Kind::Bool(true))),
                yes: Box::new(t(6, Kind::Nat(5))),
                no: Box::new(call(7, "spin", vec![])),
            },
        );
        let observation = observe(&spin(), &branch, LIMITS);
        assert_eq!(observation.result, Ok(Value::Nat(5)));
        assert_eq!(observation.steps, 3);
        assert!(!observation.events.contains(&Event::Demand(7)));
        assert!(!observation.events.contains(&Event::Demand(90)));

        // Function position is evaluated before the argument, including traps.
        let first = t(
            8,
            Kind::Apply(
                Box::new(t(9, Kind::Fail("function"))),
                Box::new(t(10, Kind::Fail("argument"))),
            ),
        );
        let observation = observe(&Program::default(), &first, LIMITS);
        assert_eq!(
            observation.result,
            Err(Error::Trap {
                at: 9,
                label: "function"
            })
        );
        assert!(!observation.events.contains(&Event::Demand(10)));

        let function_diverges = t(
            20,
            Kind::Apply(
                Box::new(call(21, "spin", vec![])),
                Box::new(t(22, Kind::Fail("argument"))),
            ),
        );
        let observation = observe(&spin(), &function_diverges, Limits { steps: 7, ..LIMITS });
        assert_eq!(observation.result, Err(Error::StepLimit(90)));
        assert!(!observation.events.contains(&Event::Demand(22)));

        // Named-call arguments are left-to-right; a later trap is hidden.
        let named = call(
            11,
            "missing",
            vec![
                t(12, Kind::Unit),
                t(13, Kind::Fail("second")),
                t(14, Kind::Fail("third")),
            ],
        );
        let observation = observe(&Program::default(), &named, LIMITS);
        assert_eq!(
            observation.result,
            Err(Error::Trap {
                at: 13,
                label: "second"
            })
        );
        assert!(!observation.events.contains(&Event::Demand(14)));
    }

    #[test]
    fn issue_m1_04_term_oracle_affine_binding_and_limits() {
        let identity = t(
            1,
            Kind::Lambda {
                binder: 20,
                body: Box::new(t(2, Kind::Var(20))),
            },
        );
        let applied = t(
            0,
            Kind::Apply(Box::new(identity), Box::new(t(3, Kind::Nat(9)))),
        );
        assert_eq!(
            observe(&Program::default(), &applied, LIMITS).result,
            Ok(Value::Nat(9))
        );

        let duplicate = t(
            4,
            Kind::Let {
                binder: 21,
                value: Box::new(t(
                    5,
                    Kind::Lambda {
                        binder: 22,
                        body: Box::new(t(6, Kind::Var(22))),
                    },
                )),
                body: Box::new(t(
                    7,
                    Kind::Pair(Box::new(t(8, Kind::Var(21))), Box::new(t(9, Kind::Var(21)))),
                )),
            },
        );
        assert_eq!(
            observe(&Program::default(), &duplicate, LIMITS).result,
            Err(Error::MissingBinder { at: 9, binder: 21 })
        );

        // Capturing an affine closure consumes it from the surrounding scope.
        let capture = t(
            24,
            Kind::Let {
                binder: 25,
                value: Box::new(t(
                    26,
                    Kind::Lambda {
                        binder: 27,
                        body: Box::new(t(28, Kind::Var(27))),
                    },
                )),
                body: Box::new(t(
                    29,
                    Kind::Let {
                        binder: 30,
                        value: Box::new(t(
                            31,
                            Kind::Lambda {
                                binder: 32,
                                body: Box::new(t(33, Kind::Var(25))),
                            },
                        )),
                        body: Box::new(t(34, Kind::Var(25))),
                    },
                )),
            },
        );
        assert_eq!(
            observe(&Program::default(), &capture, LIMITS).result,
            Err(Error::MissingBinder { at: 34, binder: 25 })
        );

        let copyable = t(
            10,
            Kind::Let {
                binder: 23,
                value: Box::new(t(11, Kind::Nat(4))),
                body: Box::new(t(
                    12,
                    Kind::Pair(
                        Box::new(t(13, Kind::Var(23))),
                        Box::new(t(14, Kind::Var(23))),
                    ),
                )),
            },
        );
        assert_eq!(
            observe(&Program::default(), &copyable, LIMITS).result,
            Ok(Value::Pair(
                Box::new(Value::Nat(4)),
                Box::new(Value::Nat(4))
            ))
        );

        let recursive = call(15, "spin", vec![]);
        assert_eq!(
            observe(&spin(), &recursive, Limits { depth: 3, ..LIMITS }).result,
            Err(Error::DepthLimit(90))
        );
        assert_eq!(
            observe(
                &Program::default(),
                &copyable,
                Limits { nodes: 2, ..LIMITS }
            )
            .result,
            Err(Error::NodeLimit)
        );
    }
}
