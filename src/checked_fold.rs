//! Checked integer sums in parallel without per-lane replay (#204, PNF law L3).
//!
//! `s = s + a[i]` over a counted loop traps at the first prefix that leaves
//! the integer range. Each chunk carries a [`Carry`]: its total and the least
//! and greatest of its prefix sums (the empty prefix included), exact in 64
//! bits. The carry combine is associative, so the combined carries decide
//! whether any prefix leaves the range and which chunk holds the first such
//! prefix. Only that chunk is re-run serially, from its exact seed, to find
//! the trap index. Value, trap kind, trap position and fuel prefix equal the
//! serial fold (contract level C0).

use crate::{
    counted::{Defs, counted_loop, element_of, local_is},
    front::{BinaryOp, Span, Type},
    mir::{Function, Instruction, Operation},
};

pub const RULE_REVISION: &str = "pnf-l3-checked-sum-carry-v1";

/// Integer type of the accumulator and of the summed elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Int {
    I32,
    U32,
}

impl Int {
    pub const fn range(self) -> (i64, i64) {
        match self {
            Self::I32 => (i32::MIN as i64, i32::MAX as i64),
            Self::U32 => (0, u32::MAX as i64),
        }
    }

    fn contains(self, value: i64) -> bool {
        let (min, max) = self.range();
        (min..=max).contains(&value)
    }
}

/// A chunk's total and its least and greatest prefix sum (empty prefix
/// included). Exact while a fold has at most `2^28` elements of at most
/// `2^32` magnitude, which the scheduler checks before it starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Carry {
    pub total: i64,
    pub low: i64,
    pub high: i64,
}

impl Carry {
    pub const IDENTITY: Self = Self {
        total: 0,
        low: 0,
        high: 0,
    };

    pub fn of(values: impl IntoIterator<Item = i64>) -> Self {
        values.into_iter().fold(Self::IDENTITY, |carry, x| {
            let total = carry.total + x;
            Self {
                total,
                low: carry.low.min(total),
                high: carry.high.max(total),
            }
        })
    }

    /// `self` followed by `right`. Associative, with [`Self::IDENTITY`].
    pub fn then(self, right: Self) -> Self {
        Self {
            total: self.total + right.total,
            low: self.low.min(self.total + right.low),
            high: self.high.max(self.total + right.high),
        }
    }

    /// Whether a fold started at `seed` leaves the range of `ty` inside this carry.
    pub fn leaves(self, seed: i64, ty: Int) -> bool {
        !ty.contains(seed + self.low) || !ty.contains(seed + self.high)
    }
}

/// The first element whose checked addition leaves the range of `ty` when the
/// fold starts at `seed`, with the accumulator value before it.
pub fn first_trap(
    seed: i64,
    values: impl IntoIterator<Item = i64>,
    ty: Int,
) -> Option<(usize, i64)> {
    let mut state = seed;
    for (k, x) in values.into_iter().enumerate() {
        if !ty.contains(state + x) {
            return Some((k, state));
        }
        state += x;
    }
    None
}

/// A counted loop `s = s + a[i]` (either operand order) with checked `+`,
/// `s` an `i32` or `u32` place and `a` an array of the same element type.
#[derive(Clone, Debug)]
pub struct Fold {
    pub ty: Int,
    pub state: String,
    pub input: String,
    pub source_span: Span,
    /// The `+` operator span, the loop's only trap site.
    pub operator: Span,
    pub head_ticks: u64,
    pub body_ticks: u64,
    /// Body ticks charged before the `+` executes.
    pub trap_ticks: u64,
}

impl Fold {
    /// Source fuel of `taken` body entries, the last one stopped at the `+`
    /// when `trapped`; the same count MIR derives from the block ticks.
    pub fn fuel(&self, taken: u64, trapped: bool) -> Option<u64> {
        let heads = taken.checked_add(u64::from(!trapped))?;
        let bodies = taken.checked_sub(u64::from(trapped))?;
        self.head_ticks
            .checked_mul(heads)?
            .checked_add(self.body_ticks.checked_mul(bodies)?)?
            .checked_add(if trapped { self.trap_ticks } else { 0 })
    }
}

fn ticks(instructions: &[Instruction]) -> u64 {
    instructions
        .iter()
        .filter(|i| matches!(i.kind, Operation::Tick))
        .count() as u64
}

/// The checked sum whose loop head is block `head`, recognized from MIR alone.
pub(crate) fn fold_in(function: &Function, defs: &Defs<'_>, head: usize) -> Option<Fold> {
    let counted = counted_loop(function, defs, head)?;
    let induction = counted.induction_name;
    let body = &counted.body.instructions;
    let (mut store, mut add) = (None, None);
    let (mut reads, mut refs, mut indexes) = (0, 0, 0);
    for (at, instruction) in body.iter().enumerate() {
        match &instruction.kind {
            Operation::Tick | Operation::ForNext(_) => {}
            Operation::Bind { name, .. } if name == induction => {}
            Operation::Drop(names) if names.iter().all(|n| n == induction) => {}
            Operation::Read(_) => reads += 1,
            Operation::ArrayRef { .. } => refs += 1,
            Operation::Index { .. } => indexes += 1,
            Operation::Binary {
                op: BinaryOp::Add,
                left,
                right,
                operator,
            } if add.is_none() => {
                let (id, ty) = instruction.output.as_ref()?;
                add = Some((at, *left, *right, *operator, *id, ty));
            }
            Operation::Store { name, value } if store.is_none() => store = Some((name, *value)),
            _ => return None,
        }
    }
    let ((state, value), (at, left, right, operator, result, result_ty)) = (store?, add?);
    let is_state = |id| matches!(defs.get(id), Some(Operation::Read(name)) if name == state);
    let element = match (is_state(&left), is_state(&right)) {
        (true, false) => right,
        (false, true) => left,
        _ => return None,
    };
    let input = element_of(defs, element, induction)?;
    let (ty, scalar, array) = match result_ty {
        Type::I32 => (Int::I32, Type::I32, Type::ArrayI32),
        Type::U32 => (Int::U32, Type::U32, Type::ArrayU32),
        _ => return None,
    };
    if value != result
        || (reads, refs, indexes) != (2, 1, 1)
        || input == state
        || !local_is(function, state, scalar, true)
        || !local_is(function, input, array, false)
    {
        return None;
    }
    Some(Fold {
        ty,
        state: state.clone(),
        input: input.to_owned(),
        source_span: counted.span,
        operator,
        head_ticks: ticks(&function.blocks.get(head)?.instructions),
        body_ticks: ticks(body),
        trap_ticks: ticks(&body[..at]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn carry_of(values: &[i64]) -> Carry {
        Carry::of(values.iter().copied())
    }

    #[test]
    fn issue_204_carry_combine_is_associative_and_matches_the_serial_prefixes() {
        let values = [5, -9, 3, i64::from(i32::MAX), -7, 2, -40, 11];
        let whole = carry_of(&values);
        for split in 0..=values.len() {
            let (left, right) = values.split_at(split);
            assert_eq!(carry_of(left).then(carry_of(right)), whole, "split {split}");
        }
        let (a, b, c) = (
            carry_of(&values[..2]),
            carry_of(&values[2..5]),
            carry_of(&values[5..]),
        );
        assert_eq!(a.then(b).then(c), a.then(b.then(c)));
        assert_eq!(Carry::IDENTITY.then(a), a);
        assert_eq!(a.then(Carry::IDENTITY), a);
    }

    #[test]
    fn issue_204_the_carry_finds_the_first_trap_in_source_order() {
        // Positive overflow at index 1, negative at index 3: the first wins.
        let values = [i64::from(i32::MAX), 1, i64::from(i32::MIN), -10];
        assert!(carry_of(&values).leaves(0, Int::I32));
        assert_eq!(
            first_trap(0, values, Int::I32),
            Some((1, i64::from(i32::MAX)))
        );
        // A dip below zero that recovers still traps for u32.
        let dip = [3, -5, 10];
        assert_eq!(carry_of(&dip).total, 8);
        assert!(carry_of(&dip).leaves(0, Int::U32));
        assert!(!carry_of(&dip).leaves(2, Int::U32));
        assert_eq!(first_trap(0, dip, Int::U32), Some((1, 3)));
    }
}
