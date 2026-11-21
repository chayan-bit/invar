//! L7 select loops on CPU workers (#207, PNF law L7).
//!
//! `for i in 0u32 .. n { if a[i] < m { m = a[i]; at = i; } }` (also `<=`, `>`
//! and `>=`, either operand order, any number of `c = i` companions) keeps the
//! running extremum `m` and the companions of its last update. A chunk's local
//! records are its elements that beat every earlier element of the chunk. In
//! serial order an element updates `m` iff it is a local record and beats the
//! best entering its chunk. Local records are monotone (strictly, for a strict
//! compare), so the ones that beat the incoming best are a suffix, found by
//! binary search. The ordered combine over chunks yields the final value and
//! companions (the last such record's) and how often the `if` was taken, which
//! the completed region states so MIR re-derives the exact fuel. A NaN is
//! never a record and nothing beats a NaN seed; ties go to the left for a
//! strict compare and to the right for a non-strict one, as in the serial
//! loop. Value bits, companions and fuel equal the serial loop (level C0).

use crate::{
    counted::{Defs, counted_head, element_of, induction_name, local_is, steps_back},
    front::{BinaryOp, Span, Type},
    general_net::normal_form::Order,
    mir::{Function, Instruction, Operation, Terminator},
    numeric_contract::{Comparison, F32Bits},
};

pub const RULE_REVISION: &str = "pnf-l7-select-records-v1";

/// Type of the selected value and of the scanned elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Elem {
    I32,
    U32,
    F32,
}

/// An element type with the source compare.
pub trait Key: Copy + Send + Sync {
    fn holds(self, op: Comparison, other: Self) -> bool;
    /// False only for NaN, which no compare accepts.
    fn ordered(self) -> bool {
        true
    }
}

fn int_holds<T: Ord>(a: T, op: Comparison, b: T) -> bool {
    match op {
        Comparison::Eq => a == b,
        Comparison::Ne => a != b,
        Comparison::Lt => a < b,
        Comparison::Le => a <= b,
        Comparison::Gt => a > b,
        Comparison::Ge => a >= b,
    }
}

impl Key for i32 {
    fn holds(self, op: Comparison, other: Self) -> bool {
        int_holds(self, op, other)
    }
}

impl Key for u32 {
    fn holds(self, op: Comparison, other: Self) -> bool {
        int_holds(self, op, other)
    }
}

impl Key for F32Bits {
    fn holds(self, op: Comparison, other: Self) -> bool {
        self.compare(op, other)
    }
    fn ordered(self) -> bool {
        self.compare(Comparison::Eq, self)
    }
}

/// Whether `x` replaces the running best under the loop's compare.
pub fn beats<T: Key>(order: Order, strict: bool, x: T, best: T) -> bool {
    let op = match (order, strict) {
        (Order::Min, true) => Comparison::Lt,
        (Order::Min, false) => Comparison::Le,
        (Order::Max, true) => Comparison::Gt,
        (Order::Max, false) => Comparison::Ge,
    };
    x.holds(op, best)
}

/// A chunk's local records in source order, and the index of the last one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Records<T> {
    pub values: Vec<T>,
    pub last: Option<usize>,
}

/// The local records of `values`, whose first element has index `start`.
pub fn records<T: Key>(order: Order, strict: bool, values: &[T], start: usize) -> Records<T> {
    let mut out = Records {
        values: Vec::new(),
        last: None,
    };
    for (k, &x) in values.iter().enumerate() {
        let record = match out.values.last() {
            None => x.ordered(),
            Some(&best) => beats(order, strict, x, best),
        };
        if record {
            out.values.push(x);
            out.last = Some(start + k);
        }
    }
    out
}

/// The serial loop's effect: how often the `if` was taken, and the index and
/// value of the last update.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Outcome<T> {
    pub taken: u64,
    pub last: Option<(usize, T)>,
}

/// Ordered combine of the chunks' records from `seed`. `right_ties` is the
/// injected mutation that lets a later chunk win a tie of a strict compare.
pub fn combine<T: Key>(
    order: Order,
    strict: bool,
    seed: T,
    chunks: &[Records<T>],
    right_ties: bool,
) -> Outcome<T> {
    let incoming = strict && !right_ties;
    let mut outcome = Outcome {
        taken: 0,
        last: None,
    };
    let mut best = seed;
    for chunk in chunks {
        let first = chunk
            .values
            .partition_point(|&r| !beats(order, incoming, r, best));
        if let (Some(&value), Some(index)) =
            (chunk.values.get(first..).and_then(<[T]>::last), chunk.last)
        {
            outcome.taken += (chunk.values.len() - first) as u64;
            outcome.last = Some((index, value));
            best = value;
        }
    }
    outcome
}

/// A counted select loop recognized from MIR.
#[derive(Clone, Debug)]
pub struct Select {
    pub elem: Elem,
    pub order: Order,
    pub strict: bool,
    /// The selected value `m`.
    pub best: String,
    /// Places assigned the induction when `m` is replaced.
    pub companions: Vec<String>,
    pub input: String,
    pub source_span: Span,
    /// The `if` statement's span (its MIR branch).
    pub test_span: Span,
    /// Ticks of the head, test, taken arm, empty arm and join blocks.
    pub ticks: [u64; 5],
}

impl Select {
    /// Source fuel of `n` completed iterations of which `taken` replaced `m`;
    /// the count MIR derives from the block ticks.
    pub fn fuel(&self, n: u64, taken: u64) -> Option<u64> {
        let [head, test, arm, other, join] = self.ticks;
        let per = [
            (head, n.checked_add(1)?),
            (test, n),
            (arm, taken),
            (other, n.checked_sub(taken)?),
            (join, n),
        ];
        per.iter().try_fold(0u64, |fuel, (ticks, times)| {
            fuel.checked_add(ticks.checked_mul(*times)?)
        })
    }
}

fn ticks(instructions: &[Instruction]) -> u64 {
    instructions
        .iter()
        .filter(|i| matches!(i.kind, Operation::Tick))
        .count() as u64
}

fn goto(function: &Function, block: usize) -> Option<usize> {
    match function.blocks.get(block)?.end {
        Some(Terminator::Goto { target, .. }) => Some(target),
        _ => None,
    }
}

/// `(order, strict)` of `x op m` with the element on the `element_left` side.
fn compare(op: BinaryOp, element_left: bool) -> Option<(Order, bool)> {
    let (order, strict) = match op {
        BinaryOp::Lt => (Order::Min, true),
        BinaryOp::Le => (Order::Min, false),
        BinaryOp::Gt => (Order::Max, true),
        BinaryOp::Ge => (Order::Max, false),
        _ => return None,
    };
    let flipped = match order {
        Order::Min => Order::Max,
        Order::Max => Order::Min,
    };
    Some((if element_left { order } else { flipped }, strict))
}

/// The select loop whose head is block `head`, recognized from MIR alone.
pub(crate) fn select_in(function: &Function, defs: &Defs<'_>, head: usize) -> Option<Select> {
    let (test_id, source_span, induction) = counted_head(function, defs, head)?;
    let test = function.blocks.get(test_id)?;
    let Some(Terminator::Branch {
        condition,
        yes: arm_id,
        no: other_id,
        span: test_span,
    }) = test.end
    else {
        return None;
    };
    let name = induction_name(test, induction)?;
    let join_id = goto(function, arm_id)?;
    if goto(function, other_id)? != join_id || arm_id == other_id {
        return None;
    }
    let (arm, other, join) = (
        function.blocks.get(arm_id)?,
        function.blocks.get(other_id)?,
        function.blocks.get(join_id)?,
    );
    // Test block: bind i, read a[i] and m, compare.
    let mut counts = (0, 0, 0, 0);
    for instruction in &test.instructions {
        match &instruction.kind {
            Operation::Tick => {}
            Operation::Bind { name: bound, .. } if bound == name => {}
            Operation::Read(_) => counts.0 += 1,
            Operation::ArrayRef { .. } => counts.1 += 1,
            Operation::Index { .. } => counts.2 += 1,
            Operation::Binary { .. } => counts.3 += 1,
            _ => return None,
        }
    }
    let Some(Operation::Binary {
        op, left, right, ..
    }) = defs.get(&condition)
    else {
        return None;
    };
    let read = |id| match defs.get(id) {
        Some(Operation::Read(place)) => Some(place.as_str()),
        _ => None,
    };
    let (input, best, element_left) = match (element_of(defs, *left, name), read(right)) {
        (Some(input), Some(best)) => (input, best, true),
        _ => (element_of(defs, *right, name)?, read(left)?, false),
    };
    let (order, strict) = compare(*op, element_left)?;
    if counts != (2, 1, 1, 1) {
        return None;
    }
    // Taken arm: m = a[i] and c = i for each companion.
    let (mut replaced, mut companions, mut arm_counts) = (false, Vec::new(), (0, 0, 0));
    for instruction in &arm.instructions {
        match &instruction.kind {
            Operation::Tick => {}
            Operation::Drop(names) if names.is_empty() => {}
            Operation::Read(_) => arm_counts.0 += 1,
            Operation::ArrayRef { .. } => arm_counts.1 += 1,
            Operation::Index { .. } => arm_counts.2 += 1,
            Operation::Store { name: place, value } if place == best && !replaced => {
                replaced = element_of(defs, *value, name) == Some(input);
            }
            Operation::Store { name: place, value }
                if read(value) == Some(name) && place != best && place != input =>
            {
                if companions.contains(place) {
                    return None;
                }
                companions.push(place.clone());
            }
            _ => return None,
        }
    }
    if !replaced || arm_counts != (1 + companions.len(), 1, 1) {
        return None;
    }
    // Empty arm and join: nothing but ticks, the induction's drop and `i + 1`.
    let only = |block: &crate::mir::BasicBlock| {
        block.instructions.iter().all(|i| match &i.kind {
            Operation::Tick => true,
            Operation::Drop(names) => names.iter().all(|n| n == name),
            Operation::ForNext(id) => *id == induction,
            _ => false,
        })
    };
    if !only(other) || !only(join) || !steps_back(join, defs, head, induction) {
        return None;
    }
    let (elem, scalar, array) = match function.locals.get(best)?.ty {
        Type::I32 => (Elem::I32, Type::I32, Type::ArrayI32),
        Type::U32 => (Elem::U32, Type::U32, Type::ArrayU32),
        Type::F32 => (Elem::F32, Type::F32, Type::ArrayF32),
        _ => return None,
    };
    if !local_is(function, best, scalar, true)
        || !local_is(function, input, array, false)
        || !companions
            .iter()
            .all(|c| local_is(function, c, Type::U32, true))
    {
        return None;
    }
    Some(Select {
        elem,
        order,
        strict,
        best: best.to_owned(),
        companions,
        input: input.to_owned(),
        source_span,
        test_span,
        ticks: [
            ticks(&function.blocks.get(head)?.instructions),
            ticks(&test.instructions),
            ticks(&arm.instructions),
            ticks(&other.instructions),
            ticks(&join.instructions),
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn serial<T: Key>(order: Order, strict: bool, seed: T, values: &[T]) -> Outcome<T> {
        let (mut best, mut outcome) = (
            seed,
            Outcome {
                taken: 0,
                last: None,
            },
        );
        for (k, &x) in values.iter().enumerate() {
            if beats(order, strict, x, best) {
                best = x;
                outcome.taken += 1;
                outcome.last = Some((k, x));
            }
        }
        outcome
    }

    fn split<T: Key>(order: Order, strict: bool, values: &[T], width: usize) -> Vec<Records<T>> {
        values
            .chunks(width)
            .enumerate()
            .map(|(c, chunk)| records(order, strict, chunk, c * width))
            .collect()
    }

    #[test]
    fn issue_207_records_combine_equals_the_serial_select_at_every_split() {
        let values = [5, 3, 3, 9, 1, 1, 7, 1, 0, 0, 4, -2, -2, 8];
        for order in [Order::Min, Order::Max] {
            for strict in [true, false] {
                for seed in [i32::MIN, -2, 0, 3, i32::MAX] {
                    let expected = serial(order, strict, seed, &values);
                    for width in 1..=values.len() {
                        let chunks = split(order, strict, &values, width);
                        assert_eq!(
                            combine(order, strict, seed, &chunks, false),
                            expected,
                            "{order:?} strict={strict} seed={seed} width={width}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn issue_207_nan_signed_zero_and_nan_seed_follow_the_serial_compare() {
        let f = |x: f32| F32Bits::from_bits(x.to_bits());
        let nan = F32Bits::from_bits(0x7fc0_0000);
        let values = [f(1.0), nan, f(0.0), f(-0.0), nan, f(-0.0), f(0.0), nan];
        for order in [Order::Min, Order::Max] {
            for strict in [true, false] {
                for seed in [f(f32::INFINITY), f(f32::NEG_INFINITY), f(0.0), nan] {
                    let expected = serial(order, strict, seed, &values);
                    for width in 1..=values.len() {
                        let chunks = split(order, strict, &values, width);
                        assert_eq!(combine(order, strict, seed, &chunks, false), expected);
                    }
                }
            }
        }
        // A strict min keeps the first of +0 and -0; a non-strict one the last.
        let zeros = [f(0.0), f(-0.0)];
        let first = serial(Order::Min, true, f(1.0), &zeros).last;
        let last = serial(Order::Min, false, f(1.0), &zeros).last;
        assert_eq!(first, Some((0, f(0.0))));
        assert_eq!(last, Some((1, f(-0.0))));
    }

    #[test]
    fn issue_207_the_rightmost_tie_mutation_changes_a_strict_select() {
        let values = [4, 1, 6, 1, 5];
        let chunks = split(Order::Min, true, &values, 3);
        assert_eq!(
            combine(Order::Min, true, i32::MAX, &chunks, false).last,
            Some((1, 1))
        );
        assert_eq!(
            combine(Order::Min, true, i32::MAX, &chunks, true).last,
            Some((3, 1))
        );
    }
}
