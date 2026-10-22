//! Source-derived ordered wrapping-u32 affine-pair reduction.
//! The sealed law is `left.then(right)`, with identity `(1, 0)`.

use crate::{
    counted::{Defs, counted_loop, definitions, element_of, local_is, push_node},
    front::{Span, Type},
    laws::{self, Affine},
    mir::{self, Function, Operation, ValueId},
    typed_net::{Net, Op, Plan, Type as GraphType},
};

pub const RULE_REVISION: &str = "ordered-affine-pair-reduction-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    InvalidMir,
    Shape,
    UnlawfulOperator,
    InvalidPlan,
    ResourceLimit,
}

impl Error {
    pub const fn code(self) -> &'static str {
        match self {
            Self::InvalidMir => "MIR_INVALID",
            Self::Shape => "SHAPE",
            Self::UnlawfulOperator => "NUMERIC_POLICY",
            Self::InvalidPlan => "CPU_CANDIDATE",
            Self::ResourceLimit => "RESOURCE_LIMIT",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Reduction {
    pub graph: Net,
    pub plan: Plan,
    /// Live scalar seeds in coefficient a/b order.
    pub states: [String; 2],
    /// Live coefficient arrays in a/b order; read/read aliasing is legal.
    pub inputs: [String; 2],
    pub source_span: Span,
}

impl Reduction {
    pub fn run(&self, a: &[u32], b: &[u32], seed: Affine, len: usize) -> Result<Affine, Error> {
        let proof = self
            .graph
            .verify(crate::loops::GRAPH_LIMITS)
            .map_err(|_| Error::InvalidPlan)?;
        if self.graph.extract(&proof).map_err(|_| Error::InvalidPlan)? != self.plan {
            return Err(Error::InvalidPlan);
        }
        self.plan.run_pair_reduce(&[a, b], &[seed.a, seed.b], len)
    }
}

impl Plan {
    /// Execute the graph-extracted ordered pair fold with exact slot/bounds checks.
    pub fn run_pair_reduce(
        &self,
        arrays: &[&[u32]],
        scalars: &[u32],
        len: usize,
    ) -> Result<Affine, Error> {
        let Self::PairReduce { seeds, inputs, .. } = self else {
            return Err(Error::InvalidPlan);
        };
        if len > laws::MAX_ELEMENTS {
            return Err(Error::ResourceLimit);
        }
        let scalar = |slot: u32| {
            scalars
                .get(slot as usize)
                .copied()
                .ok_or(Error::InvalidPlan)
        };
        let array = |slot: u32| {
            arrays
                .get(slot as usize)
                .and_then(|a| a.get(..len))
                .ok_or(Error::InvalidPlan)
        };
        let (a, b) = (array(inputs[0])?, array(inputs[1])?);
        Ok(a.iter().zip(b).fold(
            Affine {
                a: scalar(seeds[0])?,
                b: scalar(seeds[1])?,
            },
            |acc, (&a, &b)| acc.then(Affine { a, b }),
        ))
    }
}

fn graph() -> Result<(Net, Plan), Error> {
    let mut graph = Net::empty(0);
    let mut inputs = Vec::new();
    for slot in 0..2 {
        inputs.push(push_node(
            &mut graph,
            GraphType::Nat,
            Op::InputScalar(slot),
            Vec::new(),
        ));
    }
    for slot in 0..2 {
        inputs.push(push_node(
            &mut graph,
            GraphType::ArrayU32,
            Op::InputU32(slot),
            Vec::new(),
        ));
    }
    graph.entry = push_node(
        &mut graph,
        GraphType::Pair(Box::new(GraphType::Nat), Box::new(GraphType::Nat)),
        Op::PairReduce {
            rule: RULE_REVISION,
        },
        inputs,
    );
    graph.seal_source();
    let proof = graph
        .verify(crate::loops::GRAPH_LIMITS)
        .map_err(|_| Error::InvalidPlan)?;
    let plan = graph.extract(&proof).map_err(|_| Error::InvalidPlan)?;
    Ok((graph, plan))
}

fn operands(defs: &Defs<'_>, id: ValueId, operator: &str) -> Result<[ValueId; 2], Error> {
    match defs.get(&id) {
        Some(Operation::Call { name, args }) if name == operator => {
            args.as_slice().try_into().map_err(|_| Error::Shape)
        }
        _ => Err(Error::UnlawfulOperator),
    }
}

fn state_product<'a>(
    defs: &Defs<'a>,
    id: ValueId,
    state: &str,
    induction: &str,
) -> Result<&'a str, Error> {
    let [left, right] = operands(defs, id, "wrapping_mul")?;
    let is_state = |id| matches!(defs.get(&id), Some(Operation::Read(name)) if name == state);
    let element = match (is_state(left), is_state(right)) {
        (true, false) => right,
        (false, true) => left,
        _ => return Err(Error::UnlawfulOperator),
    };
    element_of(defs, element, induction).ok_or(Error::Shape)
}

/// Recognize only the sealed combined law, never a user assertion of associativity.
/// Each component reads only its own old state, so either store order is valid.
pub(crate) fn reduction_in(
    function: &Function,
    defs: &Defs<'_>,
    head: usize,
) -> Result<Reduction, Error> {
    let counted = counted_loop(function, defs, head).ok_or(Error::Shape)?;
    let induction = counted.induction_name;
    let mut stores = Vec::new();
    let (mut calls, mut reads, mut indexes, mut refs) = (0, 0, 0, 0);
    for (position, instruction) in counted.body.instructions.iter().enumerate() {
        match &instruction.kind {
            Operation::Tick | Operation::ForNext(_) => {}
            Operation::Bind { name, .. } if name == induction => {}
            Operation::Drop(names) if names.iter().all(|name| name == induction) => {}
            Operation::Read(_) => reads += 1,
            Operation::ArrayRef { .. } => refs += 1,
            Operation::Index { .. } => indexes += 1,
            Operation::Call { name, .. }
                if matches!(name.as_str(), "wrapping_add" | "wrapping_mul") =>
            {
                calls += 1
            }
            Operation::Store { name, value } => stores.push((name, *value, position)),
            Operation::Call { .. } | Operation::Binary { .. } | Operation::Unary { .. } => {
                return Err(Error::UnlawfulOperator);
            }
            _ => return Err(Error::Shape),
        }
    }
    if stores.len() != 2
        || stores[0].0 == stores[1].0
        || (calls, reads, indexes, refs) != (3, 5, 3, 3)
    {
        return Err(Error::Shape);
    }
    // The multiplicative state is a, and the add-of-product state is b.
    if matches!(defs.get(&stores[0].1), Some(Operation::Call { name, .. }) if name == "wrapping_add")
    {
        stores.swap(0, 1);
    }
    let (state_a, value_a, position_a) = stores[0];
    let (state_b, value_b, position_b) = stores[1];
    let input_a = state_product(defs, value_a, state_a, induction)?;
    let [left, right] = operands(defs, value_b, "wrapping_add")?;
    let (product, offset) = if state_product(defs, left, state_b, induction).is_ok() {
        (left, right)
    } else {
        (right, left)
    };
    if state_product(defs, product, state_b, induction)? != input_a {
        return Err(Error::UnlawfulOperator);
    }
    let input_b = element_of(defs, offset, induction).ok_or(Error::UnlawfulOperator)?;
    if !local_is(function, state_a, Type::U32, true) || !local_is(function, state_b, Type::U32, true)
        || !local_is(function, input_a, Type::ArrayU32, false) || !local_is(function, input_b, Type::ArrayU32, false)
        || counted.body.instructions.iter().enumerate().any(|(position, instruction)| matches!(&instruction.kind, Operation::Read(name) if (name == state_a && position >= position_a) || (name == state_b && position >= position_b))) {
        return Err(Error::Shape);
    }
    let (graph, plan) = graph()?;
    Ok(Reduction {
        graph,
        plan,
        states: [state_a.clone(), state_b.clone()],
        inputs: [input_a.to_string(), input_b.to_string()],
        source_span: counted.span,
    })
}

/// Find a pair reduction in verified MIR, with a typed refusal when no law matches.
pub fn extract(program: &mir::Program) -> Result<Reduction, Error> {
    mir::verify(program).map_err(|_| Error::InvalidMir)?;
    let mut reason = Error::Shape;
    for function in &program.functions {
        let defs = definitions(function);
        for head in 0..function.blocks.len() {
            match reduction_in(function, &defs, head) {
                Ok(candidate) => return Ok(candidate),
                Err(Error::UnlawfulOperator) => reason = Error::UnlawfulOperator,
                Err(_) => {}
            }
        }
    }
    Err(reason)
}
