//! Fallible bounded scan candidates. Graph and source admission remain separate.
#![forbid(unsafe_code)]

use std::mem::size_of;

use crate::bounded_summary::{self, Budget, SummaryFamily};

use super::{MATRIX_SCAN_RULE, Plan, Prefix, TABLE_SCAN_RULE};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BoundedFamily {
    Table(u8),
    Matrix(u8),
}
impl BoundedFamily {
    pub(super) fn valid(self) -> bool {
        match self {
            Self::Table(k) => (1..=16).contains(&k),
            Self::Matrix(d) => (1..=4).contains(&d),
        }
    }
    pub fn rule(self) -> &'static str {
        match self {
            Self::Table(_) => TABLE_SCAN_RULE,
            Self::Matrix(_) => MATRIX_SCAN_RULE,
        }
    }
    fn dimension(self) -> usize {
        match self {
            Self::Table(d) | Self::Matrix(d) => usize::from(d),
        }
    }
    fn width(self) -> usize {
        match self {
            Self::Table(_) => 1,
            Self::Matrix(d) => usize::from(d),
        }
    }
    fn coefficients(self) -> usize {
        match self {
            Self::Table(k) => usize::from(k),
            Self::Matrix(d) => usize::from(d) * usize::from(d),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundedScanError {
    Plan,
    Slot,
    Shape,
    Seed,
    Alias,
    Bounds,
    Allocation,
    Budget(bounded_summary::Error),
}

/// Candidate state vectors, with immutable input snapshots and explicit output mode.
/// This observation is not an ordinary-source execution/publication permit.
#[derive(Debug)]
pub struct BoundedObservation {
    family: BoundedFamily,
    prefix: Prefix,
    seed: [u32; 4],
    coefficients: Vec<u32>,
    transitions: Vec<SummaryFamily>,
    states: Vec<Vec<u32>>,
    required_bytes: usize,
    required_work: u64,
}
impl BoundedObservation {
    pub fn family(&self) -> BoundedFamily {
        self.family
    }
    pub fn prefix(&self) -> Prefix {
        self.prefix
    }
    pub fn seed(&self) -> &[u32] {
        &self.seed[..self.family.width()]
    }
    pub fn coefficients(&self) -> &[u32] {
        &self.coefficients
    }
    pub fn transitions(&self) -> &[SummaryFamily] {
        &self.transitions
    }
    pub fn prefixes(&self) -> &[Vec<u32>] {
        &self.states
    }
    pub fn final_state(&self) -> &[u32] {
        self.states.last().map_or(self.seed(), Vec::as_slice)
    }
    pub fn required_bytes(&self) -> usize {
        self.required_bytes
    }
    pub fn required_work(&self) -> u64 {
        self.required_work
    }
}

// Same frozen logical scan/snapshot admission as ordinary-source preparation.
// These are requested payload and arithmetic costs, not machine/RSS estimates.
fn cost(family: BoundedFamily, n: usize) -> Result<(usize, usize, u64), BoundedScanError> {
    let snapshot = n
        .checked_mul(family.coefficients())
        .and_then(|n| n.checked_mul(size_of::<u32>()))
        .and_then(|bytes| {
            n.checked_mul(size_of::<SummaryFamily>())
                .and_then(|t| bytes.checked_add(t))
        })
        .ok_or(BoundedScanError::Bounds)?;
    if n == 0 {
        return Ok((0, 0, 0));
    }
    let p = n
        .checked_next_power_of_two()
        .ok_or(BoundedScanError::Bounds)?;
    let combines = p
        .checked_sub(1)
        .and_then(|x| x.checked_mul(2))
        .and_then(|x| x.checked_add(n))
        .ok_or(BoundedScanError::Bounds)?;
    let d = family.dimension() as u64;
    let (compose, action) = match family {
        BoundedFamily::Table(_) => (3 * d, 2),
        BoundedFamily::Matrix(_) => (4 * d * d * d + d * d, 4 * d * d + d),
    };
    let work = (combines as u64)
        .checked_mul(compose)
        .and_then(|x| {
            (n as u64)
                .checked_mul(action)
                .and_then(|a| x.checked_add(a))
        })
        .ok_or(BoundedScanError::Bounds)?;
    let tree = p
        .checked_add(2)
        .and_then(|n| n.checked_mul(size_of::<SummaryFamily>()));
    let output = n.checked_mul(size_of::<Vec<u32>>() + family.width() * size_of::<u32>());
    let bytes = tree
        .and_then(|t| output.and_then(|o| t.checked_add(o)))
        .and_then(|b| b.checked_add(snapshot))
        .ok_or(BoundedScanError::Bounds)?;
    Ok((snapshot, bytes, work))
}
fn overlap(left: &[u32], right: &[u32]) -> Result<bool, BoundedScanError> {
    if left.is_empty() || right.is_empty() {
        return Ok(false);
    }
    let end = |s: &[u32]| {
        s.len()
            .checked_mul(size_of::<u32>())
            .and_then(|n| (s.as_ptr() as usize).checked_add(n))
            .ok_or(BoundedScanError::Bounds)
    };
    Ok((left.as_ptr() as usize) < end(right)? && (right.as_ptr() as usize) < end(left)?)
}
impl Plan {
    /// Execute a bounded candidate using the sealed ordered summary algebra.
    /// Validate shape/domain and all budgets before allocating observations.
    /// Scalar and array slot numbers name independent input namespaces.
    pub fn run_bounded_scan(
        &self,
        arrays: &[&[u32]],
        scalars: &[u32],
        budget: Budget,
    ) -> Result<BoundedObservation, BoundedScanError> {
        let Self::BoundedScan {
            family,
            prefix,
            seed,
            input,
            steps,
            source_identity,
            ..
        } = self
        else {
            return Err(BoundedScanError::Plan);
        };
        if !family.valid() || *prefix == Prefix::Exclusive || *source_identity == 0 {
            return Err(BoundedScanError::Plan);
        }
        let coefficients = *arrays.get(*input as usize).ok_or(BoundedScanError::Slot)?;
        let steps = *scalars.get(*steps as usize).ok_or(BoundedScanError::Slot)?;
        let mut fixed_seed = [0; 4];
        match family {
            BoundedFamily::Table(k) => {
                fixed_seed[0] = *scalars.get(*seed as usize).ok_or(BoundedScanError::Slot)?;
                if fixed_seed[0] >= u32::from(*k) {
                    return Err(BoundedScanError::Seed);
                }
            }
            BoundedFamily::Matrix(_) => {
                if seed == input {
                    return Err(BoundedScanError::Alias);
                }
                let state = *arrays.get(*seed as usize).ok_or(BoundedScanError::Slot)?;
                if state.len() != family.width() {
                    return Err(BoundedScanError::Seed);
                }
                if overlap(coefficients, state)? {
                    return Err(BoundedScanError::Alias);
                }
                fixed_seed[..state.len()].copy_from_slice(state);
            }
        }
        let count = steps
            .checked_mul(family.coefficients() as u32)
            .ok_or(BoundedScanError::Bounds)?;
        if coefficients.len() != count as usize || coefficients.len() > crate::eval::MAX_ARRAY_LEN {
            return Err(BoundedScanError::Shape);
        }
        // All inclusive vectors are retained even when final-state output is selected.
        if steps
            .checked_mul(family.width() as u32)
            .is_none_or(|n| n as usize > crate::eval::MAX_ARRAY_LEN)
        {
            return Err(BoundedScanError::Bounds);
        }
        if let BoundedFamily::Table(k) = family
            && coefficients.iter().any(|&n| n >= u32::from(*k))
        {
            return Err(BoundedScanError::Shape);
        }
        let n = steps as usize;
        let (snapshot, required_bytes, required_work) = cost(*family, n)?;
        if n > budget.max_steps {
            return Err(BoundedScanError::Budget(bounded_summary::Error::StepLimit));
        }
        if required_bytes > budget.max_bytes {
            return Err(BoundedScanError::Budget(bounded_summary::Error::ByteLimit));
        }
        if required_work > budget.max_composition_ops {
            return Err(BoundedScanError::Budget(
                bounded_summary::Error::CompositionLimit,
            ));
        }
        let mut transitions = Vec::new();
        transitions
            .try_reserve_exact(n)
            .map_err(|_| BoundedScanError::Allocation)?;
        for row in coefficients.chunks_exact(family.coefficients()) {
            let transition = match family {
                BoundedFamily::Table(k) => {
                    let mut t = [0u8; 16];
                    for (to, &from) in t.iter_mut().zip(row) {
                        *to = from as u8;
                    }
                    SummaryFamily::table(&t[..usize::from(*k)])
                }
                BoundedFamily::Matrix(d) => SummaryFamily::matrix(*d, row),
            }
            .map_err(BoundedScanError::Budget)?;
            transitions.push(transition);
        }
        let states = bounded_summary::prefixes(
            &transitions,
            &fixed_seed[..family.width()],
            Budget {
                max_bytes: budget.max_bytes - snapshot,
                ..budget
            },
        )
        .map_err(BoundedScanError::Budget)?;
        let mut immutable = Vec::new();
        immutable
            .try_reserve_exact(coefficients.len())
            .map_err(|_| BoundedScanError::Allocation)?;
        immutable.extend_from_slice(coefficients);
        Ok(BoundedObservation {
            family: *family,
            prefix: *prefix,
            seed: fixed_seed,
            coefficients: immutable,
            transitions,
            states,
            required_bytes,
            required_work,
        })
    }
}
