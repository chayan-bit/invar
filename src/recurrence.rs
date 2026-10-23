//! Conservative MIR inference for one wrapping affine recurrence family.
//!
//! `s = wrapping_add(wrapping_mul(a_i, s), b_i)` in any commuted operand order,
//! where each coefficient is a constant or `array[i]` of a live `Array<u32>`, is
//! represented as a typed `AffineScan` graph over live inputs. Scalar slot 0 is
//! the state; array slots follow in `a`, `b` order.

use std::sync::Arc;

use crate::{
    counted::{Defs, counted_loop, definitions, element_of, local_is, push_node},
    front::{Span, Type},
    laws::{self, Affine},
    mir::{self, Function, Operation, ValueId},
    typed_net::{Coefficient, Net, Op, Plan, Prefix, Source, Type as GraphType},
};

pub const RULE_REVISION: &str = "wrapping-affine-recurrence-v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    InvalidMir,
    InvalidPlan,
    ResourceLimit,
}

#[derive(Clone, Debug)]
pub struct Rejection {
    pub span: Span,
    pub reason: &'static str,
}

#[derive(Clone, Debug)]
pub struct Report {
    pub candidate: Option<Candidate>,
    pub rejected: Vec<Rejection>,
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub graph: Net,
    pub plan: Plan,
    /// The state owner bound to scalar slot 0.
    pub state: String,
    /// Coefficient array owners bound to array slots, in slot order.
    pub arrays: Vec<String>,
    /// The array that receives one state per step, if any.
    pub output: Option<String>,
    pub publishes_prefixes: bool,
    /// The published store precedes the state update, so element `i` is the state before step `i`.
    pub exclusive_prefix: bool,
    pub source_span: Span,
    pub rule_revision: &'static str,
    /// The unchanged MIR program, shared by every candidate of one analysis.
    pub fallback: Arc<mir::Program>,
}

impl Plan {
    /// The first `len` affine steps of an `AffineScan` plan over live array slots.
    pub fn affine_coefficients(&self, arrays: &[&[u32]], len: usize) -> Result<Vec<Affine>, Error> {
        let Plan::AffineScan { a, b, .. } = self else {
            return Err(Error::InvalidPlan);
        };
        if len > laws::MAX_ELEMENTS {
            return Err(Error::ResourceLimit);
        }
        let column = |source: &Source| -> Result<Option<&[u32]>, Error> {
            match source {
                Source::Constant(_) => Ok(None),
                Source::Slot(slot) => usize::try_from(*slot)
                    .ok()
                    .and_then(|slot| arrays.get(slot))
                    .and_then(|values| values.get(..len))
                    .map(Some)
                    .ok_or(Error::InvalidPlan),
            }
        };
        let (a_values, b_values) = (column(a)?, column(b)?);
        let value = |source: &Source, values: Option<&[u32]>, i: usize| match (source, values) {
            (Source::Constant(n), _) => *n,
            (_, Some(values)) => values[i],
            (_, None) => 0,
        };
        Ok((0..len)
            .map(|i| Affine {
                a: value(a, a_values, i),
                b: value(b, b_values, i),
            })
            .collect())
    }
}

impl Candidate {
    pub fn prefix(&self) -> Prefix {
        match &self.plan {
            Plan::AffineScan { prefix, .. } => *prefix,
            _ => Prefix::None,
        }
    }

    /// Every state after each step, from the graph-extracted plan.
    pub fn prefix_states(
        &self,
        arrays: &[&[u32]],
        seed: u32,
        len: usize,
    ) -> Result<Vec<u32>, Error> {
        let coefficients = self.plan.affine_coefficients(arrays, len)?;
        Ok(laws::scan_affine(&coefficients)
            .map_err(|_| Error::ResourceLimit)?
            .prefixes
            .into_iter()
            .map(|f| f.apply(seed))
            .collect())
    }

    /// The values the loop body stores into its output, in element order.
    pub fn published_states(
        &self,
        arrays: &[&[u32]],
        seed: u32,
        len: usize,
    ) -> Result<Vec<u32>, Error> {
        let states = self.prefix_states(arrays, seed, len)?;
        if self.prefix() != Prefix::Exclusive {
            return Ok(states);
        }
        Ok(std::iter::once(seed).chain(states).take(len).collect())
    }

    pub fn final_state(&self, arrays: &[&[u32]], seed: u32, len: usize) -> Result<u32, Error> {
        Ok(self
            .prefix_states(arrays, seed, len)?
            .last()
            .copied()
            .unwrap_or(seed))
    }

    pub fn scan_cost(&self, arrays: &[&[u32]], len: usize) -> Result<(usize, usize), Error> {
        let coefficients = self.plan.affine_coefficients(arrays, len)?;
        let scan = laws::scan_affine(&coefficients).map_err(|_| Error::ResourceLimit)?;
        Ok((scan.combines, scan.depth))
    }
}

enum Term<'a> {
    Constant(u32),
    Element(&'a str),
}

fn coefficient<'a>(
    defs: &Defs<'a>,
    id: ValueId,
    state: &str,
    induction: &str,
) -> Result<Term<'a>, &'static str> {
    match defs.get(&id) {
        Some(Operation::U32(n)) => Ok(Term::Constant(*n)),
        Some(Operation::Index { .. }) => match element_of(defs, id, induction) {
            Some(name) if name != state => Ok(Term::Element(name)),
            _ => Err("COEFFICIENT"),
        },
        _ => Err("STATE_DEPENDENCE"),
    }
}

/// Split a commutative two-operand call into (the operand matching `pick`, the other).
fn split(
    args: &[ValueId],
    pick: impl Fn(&ValueId) -> bool,
) -> Result<(ValueId, ValueId), &'static str> {
    match args {
        [left, right] if pick(left) && !pick(right) => Ok((*left, *right)),
        [left, right] if pick(right) && !pick(left) => Ok((*right, *left)),
        [left, right] if pick(left) && pick(right) => Err("STATE_DEPENDENCE"),
        _ => Err("SHAPE"),
    }
}

struct Body<'a> {
    state: &'a str,
    add: ValueId,
    store_position: usize,
    published: Option<(&'a str, ValueId, ValueId, usize)>,
    calls: Vec<ValueId>,
    positions: std::collections::HashMap<ValueId, usize>,
    reads: Vec<&'a str>,
    refs: usize,
    indexes: usize,
    constants: usize,
}

fn scan_body<'a>(body: &'a mir::BasicBlock, induction: &str) -> Result<Body<'a>, &'static str> {
    let mut stores = Vec::new();
    let mut published = Vec::new();
    let mut calls = Vec::new();
    let mut positions = std::collections::HashMap::new();
    let (mut reads, mut refs, mut indexes, mut constants) = (Vec::new(), 0, 0, 0);
    for (position, instruction) in body.instructions.iter().enumerate() {
        let output = instruction.output.as_ref().map(|(id, _)| *id);
        if let Some(id) = output {
            positions.insert(id, position);
        }
        match &instruction.kind {
            Operation::Tick | Operation::ForNext(_) => {}
            Operation::Drop(names) if names.iter().all(|name| name == induction) => {}
            Operation::Bind { name, .. } if name == induction => {}
            Operation::Read(name) => reads.push(name.as_str()),
            Operation::ArrayRef { .. } => refs += 1,
            Operation::U32(_) => constants += 1,
            Operation::Index { .. } => indexes += 1,
            Operation::Call { name, .. }
                if matches!(name.as_str(), "wrapping_add" | "wrapping_mul") =>
            {
                calls.push(output.ok_or("SHAPE")?)
            }
            Operation::Store { name, value } => stores.push((name.as_str(), *value, position)),
            Operation::StoreIndex { name, index, value } => {
                published.push((name.as_str(), *index, *value, position))
            }
            Operation::Call { .. } => return Err("UNKNOWN_EFFECT"),
            Operation::Binary { .. } | Operation::Unary { .. } => return Err("NUMERIC_POLICY"),
            _ => return Err("EFFECT_ORDER"),
        }
    }
    let ([(state, add, store_position)], true, 2) =
        (stores.as_slice(), published.len() <= 1, calls.len())
    else {
        return Err("SHAPE");
    };
    Ok(Body {
        state,
        add: *add,
        store_position: *store_position,
        published: published.first().copied(),
        calls,
        positions,
        reads,
        refs,
        indexes,
        constants,
    })
}

fn graph(
    prefix: Prefix,
    a: &Term<'_>,
    b: &Term<'_>,
) -> Result<(Net, Plan, Vec<String>), &'static str> {
    let mut graph = Net::empty(0);
    let mut inputs = vec![push_node(
        &mut graph,
        GraphType::Nat,
        Op::InputScalar(0),
        Vec::new(),
    )];
    let mut arrays = Vec::new();
    let mut source = |term: &Term<'_>, graph: &mut Net| match term {
        Term::Constant(n) => Coefficient::Constant(*n),
        Term::Element(name) => {
            let slot = u32::try_from(arrays.len()).unwrap_or(u32::MAX);
            arrays.push((*name).to_string());
            inputs.push(push_node(
                graph,
                GraphType::ArrayU32,
                Op::InputU32(slot),
                Vec::new(),
            ));
            Coefficient::Input
        }
    };
    let (a, b) = (source(a, &mut graph), source(b, &mut graph));
    let ty = if prefix == Prefix::None {
        GraphType::Nat
    } else {
        GraphType::ArrayU32
    };
    let op = Op::AffineScan {
        prefix,
        a,
        b,
        rule: RULE_REVISION,
    };
    graph.entry = push_node(&mut graph, ty, op, inputs);
    graph.seal_source();
    let plan = extract_graph(&graph)?;
    Ok((graph, plan, arrays))
}

fn extract_graph(graph: &Net) -> Result<Plan, &'static str> {
    let checked = graph
        .verify(crate::loops::GRAPH_LIMITS)
        .map_err(|_| "GRAPH")?;
    graph.extract(&checked).map_err(|_| "GRAPH")
}

fn publication(body: &Body<'_>, defs: &Defs<'_>, induction: &str) -> Result<Prefix, &'static str> {
    let Some((_, index, value, position)) = body.published else {
        return Ok(Prefix::None);
    };
    let exclusive = position < body.store_position;
    // The published read and its store sit wholly before or wholly after the update.
    let consistent = body.positions.get(&value).is_some_and(|read| {
        if exclusive {
            *read < body.store_position
        } else {
            *read > body.store_position
        }
    });
    if !consistent
        || !matches!(defs.get(&value), Some(Operation::Read(name)) if name == body.state)
        || !matches!(defs.get(&index), Some(Operation::Read(name)) if name == induction)
    {
        return Err("PUBLICATION");
    }
    Ok(if exclusive {
        Prefix::Exclusive
    } else {
        Prefix::Inclusive
    })
}

/// The recurrence candidate whose counted loop head is block `head`.
pub(crate) fn candidate_in(
    fallback: &Arc<mir::Program>,
    function: &Function,
    defs: &Defs<'_>,
    head: usize,
) -> Result<Candidate, &'static str> {
    let counted = counted_loop(function, defs, head).ok_or("RANGE")?;
    let induction = counted.induction_name;
    let body = scan_body(counted.body, induction)?;
    let state = body.state;
    if !local_is(function, state, Type::U32, true) {
        return Err("OWNERSHIP");
    }
    let Some(Operation::Call { name, args }) = defs.get(&body.add) else {
        return Err("SHAPE");
    };
    if name != "wrapping_add" {
        return Err("SHAPE");
    }
    let is_mul = |id: &ValueId| matches!(defs.get(id), Some(Operation::Call { name, .. }) if name == "wrapping_mul");
    let (mul, b_id) = split(args, is_mul)?;
    let Some(Operation::Call { args: mul_args, .. }) = defs.get(&mul) else {
        return Err("SHAPE");
    };
    let is_state =
        |id: &ValueId| matches!(defs.get(id), Some(Operation::Read(name)) if name == state);
    let (_, a_id) = split(mul_args, is_state).map_err(|_| "STATE_DEPENDENCE")?;
    if body.calls != [mul, body.add] {
        return Err("EFFECT_ORDER");
    }
    let a = coefficient(defs, a_id, state, induction)?;
    let b = coefficient(defs, b_id, state, induction)?;
    let prefix = publication(&body, defs, induction)?;
    let elements = [&a, &b]
        .into_iter()
        .filter(|term| matches!(term, Term::Element(_)))
        .count();
    let publishes = usize::from(prefix != Prefix::None);
    if body.indexes != elements
        || body.refs != elements
        || body.constants != 2 - elements
        || body.reads.len() != elements + 1 + 2 * publishes
        || body.reads.iter().filter(|name| **name == state).count() != 1 + publishes
        || body.reads.iter().filter(|name| **name == induction).count() != elements + publishes
    {
        return Err("EFFECT_ORDER");
    }
    let output = body.published.map(|(name, ..)| name);
    let coefficient_arrays = [&a, &b].into_iter().filter_map(|term| match term {
        Term::Element(name) => Some(*name),
        Term::Constant(_) => None,
    });
    if let Some(output) = output
        && (!local_is(function, output, Type::ArrayU32, true)
            || output == state
            || coefficient_arrays.clone().any(|name| name == output))
    {
        return Err("PUBLICATION");
    }
    if coefficient_arrays
        .clone()
        .any(|name| !local_is(function, name, Type::ArrayU32, false))
    {
        return Err("COEFFICIENT");
    }
    let (graph, plan, arrays) = graph(prefix, &a, &b)?;
    Ok(Candidate {
        graph,
        plan,
        state: state.to_string(),
        arrays,
        output: output.map(str::to_string),
        publishes_prefixes: prefix != Prefix::None,
        exclusive_prefix: prefix == Prefix::Exclusive,
        source_span: counted.span,
        rule_revision: RULE_REVISION,
        fallback: Arc::clone(fallback),
    })
}

pub fn infer(program: &mir::Program) -> Result<Report, Error> {
    mir::verify(program).map_err(|_| Error::InvalidMir)?;
    let mut rejected = Vec::new();
    let fallback = Arc::new(program.clone());
    for function in &program.functions {
        let defs = definitions(function);
        for head in 0..function.blocks.len() {
            let Some(counted) = counted_loop(function, &defs, head) else {
                continue;
            };
            match candidate_in(&fallback, function, &defs, head) {
                Ok(candidate) => {
                    return Ok(Report {
                        candidate: Some(candidate),
                        rejected,
                    });
                }
                Err(reason) => rejected.push(Rejection {
                    span: counted.span,
                    reason,
                }),
            }
        }
    }
    Ok(Report {
        candidate: None,
        rejected,
    })
}

#[cfg(test)]
mod ai_01_fault_tests {
    #[test]
    fn issue_ai_01_machine_diagnostics_recurrence_graph_fault() {
        let forged = super::Net::empty(0);
        let code = super::extract_graph(&forged).unwrap_err();
        assert_eq!(code, "GRAPH");
        let document = crate::diagnostics::document(
            "tool_error",
            vec![crate::diagnostics::diagnostic(
                code,
                "forged recurrence graph",
                "fixture.inv",
                b"",
                None,
            )],
        );
        let schema: serde_json::Value =
            serde_json::from_str(include_str!("../schema/invar-diagnostics-v3.schema.json"))
                .unwrap();
        assert!(jsonschema::is_valid(&schema, &document), "{document}");
    }
}
