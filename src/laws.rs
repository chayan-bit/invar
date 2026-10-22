//! Sealed wrapping-u32 laws and left-to-right ordered scans.

use crate::{
    counted::{Defs, counted_loop, definitions, element_of, local_is, push_node},
    front::{Span, Type},
    mir::{self, Function, Operation, ValueId},
    typed_net::{self, Net, Op, Plan, Type as GraphType},
};

pub const RULE_REVISION: &str = "sealed-wrapping-u32-laws-v1";
pub const MAX_ELEMENTS: usize = 1_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Law {
    WrappingSum,
    WrappingProduct,
}

impl Law {
    pub const fn identity(self) -> u32 {
        match self {
            Self::WrappingSum => 0,
            Self::WrappingProduct => 1,
        }
    }

    pub const fn combine(self, left: u32, right: u32) -> u32 {
        match self {
            Self::WrappingSum => left.wrapping_add(right),
            Self::WrappingProduct => left.wrapping_mul(right),
        }
    }

    pub const fn revision(self) -> &'static str {
        RULE_REVISION
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Affine {
    pub a: u32,
    pub b: u32,
}

impl Affine {
    pub const IDENTITY: Self = Self { a: 1, b: 0 };

    /// Apply `left`, then `right`. This order matters.
    pub const fn then(self, right: Self) -> Self {
        Self {
            a: right.a.wrapping_mul(self.a),
            b: right.a.wrapping_mul(self.b).wrapping_add(right.b),
        }
    }

    pub const fn apply(self, seed: u32) -> u32 {
        self.a.wrapping_mul(seed).wrapping_add(self.b)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scan<T> {
    pub prefixes: Vec<T>,
    pub combines: usize,
    pub depth: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    ResourceLimit,
    InvalidMir,
    InvalidPlan,
}

fn ordered_scan<T: Copy>(
    values: &[T],
    identity: T,
    combine: impl Fn(T, T) -> T,
) -> Result<Scan<T>, Error> {
    if values.len() > MAX_ELEMENTS {
        return Err(Error::ResourceLimit);
    }
    if values.is_empty() {
        return Ok(Scan {
            prefixes: Vec::new(),
            combines: 0,
            depth: 0,
        });
    }
    let size = values.len().next_power_of_two();
    let mut tree = vec![identity; size];
    tree[..values.len()].copy_from_slice(values);
    let mut combines = 0;
    let mut depth = 0;
    let mut stride = 1;
    while stride < size {
        for base in (0..size).step_by(2 * stride) {
            let left = base + stride - 1;
            let right = base + 2 * stride - 1;
            tree[right] = combine(tree[left], tree[right]);
            combines += 1;
        }
        stride *= 2;
        depth += 1;
    }
    tree[size - 1] = identity;
    stride = size / 2;
    while stride > 0 {
        for base in (0..size).step_by(2 * stride) {
            let left = base + stride - 1;
            let right = base + 2 * stride - 1;
            let total = tree[left];
            tree[left] = tree[right];
            tree[right] = combine(tree[right], total);
            combines += 1;
        }
        stride /= 2;
        depth += 1;
    }
    let prefixes = values
        .iter()
        .enumerate()
        .map(|(i, &value)| {
            combines += 1;
            combine(tree[i], value)
        })
        .collect();
    Ok(Scan {
        prefixes,
        combines,
        depth: depth + 1,
    })
}

pub fn scan_u32(law: Law, values: &[u32]) -> Result<Scan<u32>, Error> {
    ordered_scan(values, law.identity(), |left, right| {
        law.combine(left, right)
    })
}

pub fn scan_affine(values: &[Affine]) -> Result<Scan<Affine>, Error> {
    ordered_scan(values, Affine::IDENTITY, Affine::then)
}

/// A lawful counted reduction `acc = law(acc, xs[i])` (either operand order),
/// represented as a typed graph over live inputs: scalar slot 0 is `state`, array
/// slot 0 is `input`.
#[derive(Clone, Debug)]
pub struct Reduction {
    pub law: Law,
    pub graph: Net,
    pub plan: Plan,
    pub input: String,
    pub state: String,
    pub source_span: Span,
    pub rule_revision: &'static str,
    pub fallback_reason: &'static str,
}

impl Reduction {
    /// Run the graph-extracted plan over `input` from `seed`.
    pub fn run(&self, input: &[u32], seed: u32) -> Result<u32, Error> {
        self.plan.run_reduce(&[input], &[seed], input.len())
    }
}

impl Plan {
    /// Fold the first `len` elements of the reduction's array slot from its seed slot.
    pub fn run_reduce(&self, arrays: &[&[u32]], scalars: &[u32], len: usize) -> Result<u32, Error> {
        let Plan::Reduce {
            law, seed, input, ..
        } = self
        else {
            return Err(Error::InvalidPlan);
        };
        let slot = |index: u32| usize::try_from(index).map_err(|_| Error::InvalidPlan);
        let seed = *scalars.get(slot(*seed)?).ok_or(Error::InvalidPlan)?;
        let values = arrays
            .get(slot(*input)?)
            .and_then(|values| values.get(..len))
            .ok_or(Error::InvalidPlan)?;
        if len > MAX_ELEMENTS {
            return Err(Error::ResourceLimit);
        }
        Ok(values
            .iter()
            .fold(seed, |acc, value| law.combine(acc, *value)))
    }
}

pub fn extract_reduction(program: &mir::Program) -> Result<Option<Reduction>, Error> {
    mir::verify(program).map_err(|_| Error::InvalidMir)?;
    for function in &program.functions {
        let defs = definitions(function);
        for head in 0..function.blocks.len() {
            if let Some(reduction) = reduction_in(function, &defs, head) {
                return Ok(Some(reduction));
            }
        }
    }
    Ok(None)
}

fn graph(law: Law) -> Result<(Net, Plan), typed_net::Error> {
    let mut graph = Net::empty(0);
    let seed = push_node(&mut graph, GraphType::Nat, Op::InputScalar(0), Vec::new());
    let input = push_node(&mut graph, GraphType::ArrayU32, Op::InputU32(0), Vec::new());
    graph.entry = push_node(
        &mut graph,
        GraphType::Nat,
        Op::Reduce {
            law,
            rule: RULE_REVISION,
        },
        vec![seed, input],
    );
    graph.seal_source();
    let plan = graph.extract(&graph.verify(crate::loops::GRAPH_LIMITS)?)?;
    Ok((graph, plan))
}

/// The lawful reduction whose loop head is block `head`, recognized from MIR alone.
pub(crate) fn reduction_in(function: &Function, defs: &Defs<'_>, head: usize) -> Option<Reduction> {
    let counted = counted_loop(function, defs, head)?;
    let induction = counted.induction_name;
    let (mut store, mut call) = (None, None);
    let (mut reads, mut refs, mut indexes) = (0, 0, 0);
    for instruction in &counted.body.instructions {
        match &instruction.kind {
            Operation::Tick | Operation::ForNext(_) | Operation::Bind { .. } => {}
            Operation::Drop(names) if names.iter().all(|n| n == induction) => {}
            Operation::Read(_) => reads += 1,
            Operation::ArrayRef { .. } => refs += 1,
            Operation::Index { .. } => indexes += 1,
            Operation::Call { name, args } if call.is_none() => {
                call = Some((name.as_str(), args, instruction.output.as_ref()?.0))
            }
            Operation::Store { name, value } if store.is_none() => store = Some((name, *value)),
            _ => return None,
        }
    }
    let ((state, value), (name, args, result)) = (store?, call?);
    let law = match name {
        "wrapping_add" => Law::WrappingSum,
        "wrapping_mul" => Law::WrappingProduct,
        _ => return None,
    };
    let [left, right] = args.as_slice() else {
        return None;
    };
    let is_state =
        |id: &ValueId| matches!(defs.get(id), Some(Operation::Read(name)) if name == state);
    // Both sealed laws are commutative, so the accumulator may be either operand.
    let element = match (is_state(left), is_state(right)) {
        (true, false) => right,
        (false, true) => left,
        _ => return None,
    };
    let input = element_of(defs, *element, induction)?;
    let binds_only_induction = counted
        .body
        .instructions
        .iter()
        .all(|i| !matches!(&i.kind, Operation::Bind { name, .. } if name != induction));
    if value != result
        || (reads, refs, indexes) != (2, 1, 1)
        || !binds_only_induction
        || input == state
        || !local_is(function, state, Type::U32, true)
        || !local_is(function, input, Type::ArrayU32, false)
    {
        return None;
    }
    let (graph, plan) = graph(law).ok()?;
    Some(Reduction {
        law,
        graph,
        plan,
        input: input.to_string(),
        state: state.clone(),
        source_span: counted.span,
        rule_revision: RULE_REVISION,
        fallback_reason: "mir-serial",
    })
}
