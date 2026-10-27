//! Source-bound weighted modular histograms and private synchronous CPU publication.
#![forbid(unsafe_code)]

use crate::{
    budget::{BudgetExceeded, ByteBudget, Footprint, array_bytes},
    check::{self, CheckedProgram},
    front::{self, Expr, ExprKind, Mode, Place, Span, StmtKind, Type, named, ungroup},
    general_net,
    laws::Law,
    mir,
};
use sha2::{Digest, Sha256};

#[path = "histogram_source/bounded.rs"]
mod bounded;
pub mod host;

/// Staging-protocol limits of the optimized producer, shared with the indexed hosts; the
/// element and contribution counts it used to carry are byte budgets now (#57).
pub const MAX_STAGING_BYTES: usize = 160 * 1024;
pub const MAX_VISITS: usize = 65_536;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    pub bytes: usize,
    pub visits: usize,
}
impl Default for Budget {
    fn default() -> Self {
        Self {
            bytes: MAX_STAGING_BYTES,
            visits: MAX_VISITS,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cost {
    pub bytes: usize,
    pub visits: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Source(front::FrontError),
    CheckedMismatch,
    Unrecognized,
    Net(general_net::Error),
    /// Work or staging limit of a host plan; byte budgets refuse as `Exceeded`.
    Budget,
    /// The region needs more bytes than the [`ByteBudget`] holds (`BUDGET_EXCEEDED`).
    /// The caller keeps the original source path.
    Exceeded(BudgetExceeded),
    Bounds {
        iteration: usize,
    },
    Allocation,
    Binding,
    Receipt,
}

/// Source iteration reads values/indices at `iteration`, and reads/writes `bin`.
/// Inputs are immutable contiguous arrays; histogram storage is a fresh owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Access {
    pub iteration: usize,
    pub bin: usize,
}

#[derive(Debug)]
pub struct Candidate {
    output: Vec<u32>,
    source_accesses: Vec<Access>,
    grouped_accesses: Vec<Access>,
    sort_visits: usize,
}
impl Candidate {
    pub fn output(&self) -> &[u32] {
        &self.output
    }
    pub fn source_accesses(&self) -> &[Access] {
        &self.source_accesses
    }
    /// Actual candidate schedule, grouped by destination under the sealed law.
    pub fn grouped_accesses(&self) -> &[Access] {
        &self.grouped_accesses
    }
    pub fn sort_visits(&self) -> usize {
        self.sort_visits
    }
}

/// Host bytes one region holds: the resident `u32` values, indices and fresh output, and the
/// source-order and bin-grouped access records, all of which `candidate_within` allocates but
/// the first two. Device bytes are the caller's to add.
pub fn footprint(values: usize, indices: usize, bins: usize) -> Footprint {
    let words = values
        .checked_add(indices)
        .and_then(|n| n.checked_add(bins))
        .and_then(array_bytes::<u32>);
    Footprint::ZERO
        .host(words)
        .host(array_bytes::<Access>(indices))
        .host(array_bytes::<Access>(indices))
}

/// This law is derived from checked source, never accepted as a caller argument.
#[derive(Debug)]
pub struct Contract {
    checked: CheckedProgram,
    seal: bounded::Seal,
    runtime_names: usize,
    source: Vec<u8>,
    source_hash: [u8; 32],
    entry: String,
    net: general_net::Net,
    proof: general_net::Proof,
    source_contract: [u8; 32],
    loop_span: Span,
    law: Law,
}

fn zero(e: &Expr) -> bool {
    matches!(&ungroup(e).kind, ExprKind::Integer(n) if n == "0" || n == "0u32")
}
fn index<'a>(e: &'a Expr, name: &str) -> Option<&'a Expr> {
    match &ungroup(e).kind {
        ExprKind::Index { base, index } if named(base, name) => Some(index),
        _ => None,
    }
}
fn indirect(e: &Expr, indices: &str, induction: &str) -> bool {
    index(e, indices).is_some_and(|i| named(i, induction))
}
fn reserve<T>(v: &mut Vec<T>, count: usize) -> Result<(), Error> {
    v.try_reserve_exact(count).map_err(|_| Error::Allocation)
}

fn source_law(checked: &CheckedProgram, entry: &str) -> Result<(Span, Law), Error> {
    let f = checked.program.function(entry).ok_or(Error::Unrecognized)?;
    let [values, indices, bins] = f.params.as_slice() else {
        return Err(Error::Unrecognized);
    };
    if [values, indices]
        .iter()
        .any(|p| p.ty != Type::ArrayU32 || p.mode != Some(Mode::Read))
        || bins.ty != Type::U32
        || bins.mode.is_some()
        || f.result != Type::ArrayU32
        || f.return_from.is_some()
        || !f.generic_params.is_empty()
    {
        return Err(Error::Unrecognized);
    }
    let [allocation, region, ret] = f.body.statements.as_slice() else {
        return Err(Error::Unrecognized);
    };
    let StmtKind::Binding {
        mutable: true,
        name: output,
        value,
        ..
    } = &allocation.kind
    else {
        return Err(Error::Unrecognized);
    };
    let ExprKind::Call { name, args } = &ungroup(value).kind else {
        return Err(Error::Unrecognized);
    };
    if name != "filled_u32"
        || !matches!(args.as_slice(), [n, fill] if named(n, &bins.name) && zero(fill))
    {
        return Err(Error::Unrecognized);
    }
    let StmtKind::Return(Some(owner)) = &ret.kind else {
        return Err(Error::Unrecognized);
    };
    if !named(owner, output) && !matches!(&ungroup(owner).kind, ExprKind::Move(n) if n == output) {
        return Err(Error::Unrecognized);
    }
    let StmtKind::For {
        name: induction,
        lower,
        upper,
        body,
    } = &region.kind
    else {
        return Err(Error::Unrecognized);
    };
    if !zero(lower)
        || !matches!(&ungroup(upper).kind, ExprKind::Call { name, args } if name == "len" && matches!(args.as_slice(), [input] if named(input, &indices.name)))
    {
        return Err(Error::Unrecognized);
    }
    let [store] = body.statements.as_slice() else {
        return Err(Error::Unrecognized);
    };
    let StmtKind::Assign {
        place:
            Place::Index {
                name: destination,
                index: dst,
                ..
            },
        value,
    } = &store.kind
    else {
        return Err(Error::Unrecognized);
    };
    if destination != output || !indirect(dst, &indices.name, induction) {
        return Err(Error::Unrecognized);
    }
    let ExprKind::Call { name, args } = &ungroup(value).kind else {
        return Err(Error::Unrecognized);
    };
    let [left, right] = args.as_slice() else {
        return Err(Error::Unrecognized);
    };
    let accumulator =
        |e: &Expr| index(e, output).is_some_and(|i| indirect(i, &indices.name, induction));
    let weight = |e: &Expr| index(e, &values.name).is_some_and(|i| named(i, induction));
    if name != "wrapping_add"
        || !((accumulator(left) && weight(right)) || (weight(left) && accumulator(right)))
    {
        return Err(Error::Unrecognized);
    }
    // Existing compiler-sealed law, identity and revision. No supplied flag
    // or test outcome is used as algebraic evidence.
    let law = Law::WrappingSum;
    Ok((region.span, law))
}

impl Contract {
    /// Recognize a zero-initialized weighted histogram whose only loop update
    /// adds one source weight to its indexed bin using sealed wrapping-u32 sum.
    /// Every duplicate occurrence contributes once, as specified by that source.
    pub fn from_checked(
        source: &[u8],
        supplied: &CheckedProgram,
        entry: &str,
    ) -> Result<Self, Error> {
        let checked = check::check(front::parse(source).map_err(Error::Source)?, false)
            .map_err(Error::Source)?;
        if checked.program != supplied.program
            || checked.expression_types != supplied.expression_types
            || checked.function_expression_types != supplied.function_expression_types
        {
            return Err(Error::CheckedMismatch);
        }
        let (loop_span, law) = source_law(&checked, entry)?;
        let runtime_names = bounded::role_bytes(&checked, entry)?;
        let net = general_net::Net::lower(&checked, general_net::Limits::default())
            .map_err(Error::Net)?;
        let proof = net.verify().map_err(Error::Net)?;
        let m = net.compiled_mir(&proof).map_err(Error::Net)?;
        let seal = bounded::Seal::new(source, &checked, &net, &proof, m.contract_digest(), entry)?;
        Ok(Self {
            checked,
            seal,
            runtime_names,
            source: source.to_vec(),
            source_hash: Sha256::digest(source).into(),
            entry: entry.into(),
            source_contract: m.contract_digest(),
            loop_span,
            law,
            net,
            proof,
        })
    }
    pub fn law(&self) -> Law {
        self.law
    }
    pub fn law_revision(&self) -> &'static str {
        self.law.revision()
    }
    pub fn loop_span(&self) -> Span {
        self.loop_span
    }
    pub fn source_contract(&self) -> [u8; 32] {
        self.source_contract
    }
    pub fn verified_mir(&self) -> Result<mir::Program, Error> {
        self.verified().map(|(m, _)| m)
    }
    fn verified(&self) -> Result<(mir::Program, Law), Error> {
        let hash: [u8; 32] = Sha256::digest(&self.source).into();
        if hash != self.source_hash {
            return Err(Error::Binding);
        }
        let m = self.net.compiled_mir(&self.proof).map_err(Error::Net)?;
        let checked = check::check(front::parse(&self.source).map_err(Error::Source)?, false)
            .map_err(Error::Source)?;
        // Re-recognize actual checked source, including identity and operand
        // roles; a saved law/revision or a newly valid different net is not proof.
        let derived = Self::from_checked(&self.source, &checked, &self.entry)?;
        if self.law != derived.law
            || self.law != Law::WrappingSum
            || self.source_contract != m.contract_digest()
            || self.source_contract != derived.source_contract
            || self.loop_span != derived.loop_span
        {
            return Err(Error::Binding);
        }
        Ok((m, derived.law))
    }

    /// [`Self::candidate_within`] under the default [`ByteBudget`].
    pub fn candidate(
        &self,
        values: &[u32],
        indices: &[u32],
        bins: u32,
    ) -> Result<Candidate, Error> {
        self.candidate_within(values, indices, bins, &ByteBudget::default())
    }

    /// Guard all source reads and writes, then observe a bin-grouped CPU result.
    /// Borrowed inputs stay immutable throughout guarding and observation; they
    /// may alias each other. No externally supplied output can be mutated.
    /// Refusals retain source execution for exact trap/fuel/cleanup behavior.
    /// This operation does not debit semantic fuel or publish source state.
    /// A region whose [`footprint`] exceeds `budget` is refused as `Exceeded`.
    pub fn candidate_within(
        &self,
        values: &[u32],
        indices: &[u32],
        bins: u32,
        budget: &ByteBudget,
    ) -> Result<Candidate, Error> {
        self.verified_mir()?;
        let n = usize::try_from(bins).map_err(|_| Error::Budget)?;
        budget
            .admit("histogram", &footprint(values.len(), indices.len(), n))
            .map_err(Error::Exceeded)?;
        // The language's per-array limit is independent of bytes: the source traps on a larger array.
        if values.len().max(indices.len()).max(n) > crate::array_limits::MAX_ARRAY_LEN {
            return Err(Error::Budget);
        }
        let mut accesses = Vec::new();
        reserve(&mut accesses, indices.len())?;
        for (iteration, &raw) in indices.iter().enumerate() {
            let bin = usize::try_from(raw).map_err(|_| Error::Bounds { iteration })?;
            if iteration >= values.len() || bin >= n {
                return Err(Error::Bounds { iteration });
            }
            accesses.push(Access { iteration, bin });
        }
        let mut grouped = Vec::new();
        reserve(&mut grouped, accesses.len())?;
        grouped.extend_from_slice(&accesses);
        // Only the sealed commutative sum on a fresh, otherwise unobserved
        // accumulator permits this regrouping. Unknown source operations refuse.
        let sort_visits = bounded::sort_observation(&mut grouped)?;
        let mut output = Vec::new();
        reserve(&mut output, n)?;
        output.resize(n, self.law.identity());
        for access in &grouped {
            let value = *values.get(access.iteration).ok_or(Error::Bounds {
                iteration: access.iteration,
            })?;
            let destination = output.get_mut(access.bin).ok_or(Error::Bounds {
                iteration: access.iteration,
            })?;
            *destination = self.law.combine(*destination, value);
        }
        Ok(Candidate {
            output,
            source_accesses: accesses,
            grouped_accesses: grouped,
            sort_visits,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn issue_m4_05_gather_scatter_frontier_histogram_stale_net_refuses_candidate() {
        let s = b"fn hist(a: read Array<u32>, ix: read Array<u32>, n: u32) -> Array<u32> { var out = filled_u32(n,0u32); for i in 0u32..len(ix) { out[ix[i]] = wrapping_add(out[ix[i]],a[i]); } return out; }";
        for mutation in 0..4 {
            let p = check::check(front::parse(s).unwrap(), false).unwrap();
            let mut c = Contract::from_checked(s, &p, "hist").unwrap();
            assert_eq!(c.law(), Law::WrappingSum);
            assert_eq!(c.law_revision(), crate::laws::RULE_REVISION);
            match mutation {
                0 => c.net.agents[0].next.position += 1,
                1 => c.net.agents[0].span.end += 1,
                2 => c.net.agents[0].rule = general_net::coverage::Rule::Binary,
                _ => c.net.graph.revision += 1,
            }
            assert!(c.verified_mir().is_err());
            assert!(matches!(c.candidate(&[1], &[0], 1), Err(Error::Net(_))));
        }
    }
}
