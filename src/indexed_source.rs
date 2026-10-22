//! Bounded ordinary-source u32 gather/injective-scatter candidate observations.
//! This is not an execution, fuel-debit, ABI or device-publication permit.
#![forbid(unsafe_code)]

pub mod host;

use crate::{
    budget::{BudgetExceeded, ByteBudget, Footprint, array_bytes},
    check::{self, CheckedProgram},
    front::{self, Expr, ExprKind, Mode, Place, Span, StmtKind, Type, named, ungroup},
    general_net, mir,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Gather,
    InjectiveScatter,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Source(front::FrontError),
    CheckedMismatch,
    Net(general_net::Error),
    Unrecognized,
    Shape,
    /// Work or staging limit of a host plan; byte budgets refuse as `Exceeded`.
    Budget,
    /// The region needs more bytes than the [`ByteBudget`] holds (`BUDGET_EXCEEDED`).
    /// The caller keeps the original source path.
    Exceeded(BudgetExceeded),
    Bounds {
        iteration: usize,
    },
    Duplicate {
        iteration: usize,
        index: u32,
    },
    Allocation,
}

/// Scalar element accesses, in original iteration order, to distinct fresh output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Access {
    pub iteration: usize,
    pub read_index: usize,
    pub write_index: usize,
}

/// Private candidate payload: immutable inputs were guarded before any copy.
#[derive(Debug)]
pub struct Candidate {
    output: Vec<u32>,
    accesses: Vec<Access>,
}
impl Candidate {
    pub fn output(&self) -> &[u32] {
        &self.output
    }
    pub fn accesses(&self) -> &[Access] {
        &self.accesses
    }
}

/// Source-owned identity and verified extraction, with no imported proof flags.
#[derive(Debug)]
pub struct Contract {
    net: general_net::Net,
    proof: general_net::Proof,
    family: Family,
    loop_span: Span,
    source_contract: [u8; 32],
}

fn zero(e: &Expr) -> bool {
    matches!(&ungroup(e).kind, ExprKind::Integer(n) if n == "0" || n == "0u32")
}
fn length(e: &Expr, input: &str) -> bool {
    matches!(&ungroup(e).kind, ExprKind::Call { name, args }
        if name == "len" && matches!(args.as_slice(), [a] if named(a, input)))
}
fn index<'a>(e: &'a Expr, input: &str) -> Option<&'a Expr> {
    match &ungroup(e).kind {
        ExprKind::Index { base, index } if named(base, input) => Some(index),
        _ => None,
    }
}
/// Host bytes one region holds: the resident inputs and fresh output (u32), the
/// per-iteration access records, and for scatter the one-byte-per-slot written
/// map. `candidate_within` allocates exactly the last three. Device bytes are
/// the caller's to add: this observer never touches a device.
pub fn footprint(family: Family, values: usize, indices: usize, output: usize) -> Footprint {
    let words = values
        .checked_add(indices)
        .and_then(|n| n.checked_add(output))
        .and_then(array_bytes::<u32>);
    let written = match family {
        Family::Gather => Some(0),
        Family::InjectiveScatter => array_bytes::<bool>(output),
    };
    Footprint::ZERO
        .host(words)
        .host(array_bytes::<Access>(indices))
        .host(written)
}
fn reserve<T>(v: &mut Vec<T>, n: usize) -> Result<(), Error> {
    v.try_reserve_exact(n).map_err(|_| Error::Allocation)
}

impl Contract {
    /// Reparse exact source bytes and compare all checked annotations before
    /// lowering. Public CheckedProgram fields alone cannot authenticate source.
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
        let f = checked
            .program
            .functions
            .iter()
            .find(|f| f.name == entry)
            .ok_or(Error::Unrecognized)?;
        if !(2..=3).contains(&f.params.len())
            || f.params[..2]
                .iter()
                .any(|p| p.mode != Some(Mode::Read) || p.ty != Type::ArrayU32)
            || f.result != Type::ArrayU32
            || f.return_from.is_some()
            || !f.generic_params.is_empty()
        {
            return Err(Error::Unrecognized);
        }
        let family = if f.params.len() == 2 {
            Family::Gather
        } else if f.params[2].mode.is_none() && f.params[2].ty == Type::U32 {
            Family::InjectiveScatter
        } else {
            return Err(Error::Unrecognized);
        };
        let [allocate, region, ret] = f.body.statements.as_slice() else {
            return Err(Error::Unrecognized);
        };
        let StmtKind::Binding {
            mutable: true,
            name: output,
            value,
            ..
        } = &allocate.kind
        else {
            return Err(Error::Unrecognized);
        };
        let input = &f.params[0].name;
        let indices = &f.params[1].name;
        let ExprKind::Call { name, args } = &ungroup(value).kind else {
            return Err(Error::Unrecognized);
        };
        let [count, fill] = args.as_slice() else {
            return Err(Error::Unrecognized);
        };
        if name != "filled_u32"
            || !zero(fill)
            || match family {
                Family::Gather => !length(count, indices),
                Family::InjectiveScatter => !named(count, &f.params[2].name),
            }
        {
            return Err(Error::Unrecognized);
        }
        let StmtKind::Return(Some(owner)) = &ret.kind else {
            return Err(Error::Unrecognized);
        };
        if !named(owner, output)
            && !matches!(&ungroup(owner).kind, ExprKind::Move(n) if n == output)
        {
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
        if !zero(lower) || !length(upper, indices) {
            return Err(Error::Unrecognized);
        }
        let [store] = body.statements.as_slice() else {
            return Err(Error::Unrecognized);
        };
        let StmtKind::Assign {
            place: Place::Index {
                name, index: dst, ..
            },
            value,
        } = &store.kind
        else {
            return Err(Error::Unrecognized);
        };
        let src = index(value, input).ok_or(Error::Unrecognized)?;
        let matches = match family {
            Family::Gather => {
                named(dst, induction) && index(src, indices).is_some_and(|i| named(i, induction))
            }
            Family::InjectiveScatter => {
                named(src, induction) && index(dst, indices).is_some_and(|i| named(i, induction))
            }
        };
        if name != output || !matches {
            return Err(Error::Unrecognized);
        }
        let net = general_net::Net::lower(&checked, general_net::Limits::default())
            .map_err(Error::Net)?;
        let proof = net.verify().map_err(Error::Net)?;
        let m = net.compiled_mir(&proof).map_err(Error::Net)?;
        Ok(Self {
            family,
            loop_span: region.span,
            source_contract: m.contract_digest(),
            net,
            proof,
        })
    }
    pub fn family(&self) -> Family {
        self.family
    }
    pub fn loop_span(&self) -> Span {
        self.loop_span
    }
    pub fn source_contract(&self) -> [u8; 32] {
        self.source_contract
    }
    pub fn verified_mir(&self) -> Result<mir::Program, Error> {
        self.net.compiled_mir(&self.proof).map_err(Error::Net)
    }

    /// [`Self::candidate_within`] under the default [`ByteBudget`].
    pub fn candidate(
        &self,
        values: &[u32],
        indices: &[u32],
        scatter_len: Option<u32>,
    ) -> Result<Candidate, Error> {
        self.candidate_within(values, indices, scatter_len, &ByteBudget::default())
    }

    /// Observe the recognized scalar loop on immutable contiguous u32 inputs.
    /// Gather takes no output length; scatter requires the source scalar length.
    /// Failed guards are candidate refusals, not source traps. Run the retained
    /// source path to preserve RHS-first trap order, fuel and language cleanup.
    /// Read inputs may alias each other lawfully; output is always a fresh owner.
    /// A region whose [`footprint`] exceeds `budget` is refused as `Exceeded`.
    pub fn candidate_within(
        &self,
        values: &[u32],
        indices: &[u32],
        scatter_len: Option<u32>,
        budget: &ByteBudget,
    ) -> Result<Candidate, Error> {
        self.verified_mir()?;
        let output_len = match (self.family, scatter_len) {
            (Family::Gather, None) => indices.len(),
            (Family::InjectiveScatter, Some(n)) => usize::try_from(n).map_err(|_| Error::Budget)?,
            _ => return Err(Error::Shape),
        };
        let family = match self.family {
            Family::Gather => "gather",
            Family::InjectiveScatter => "injective_scatter",
        };
        budget
            .admit(
                family,
                &footprint(self.family, values.len(), indices.len(), output_len),
            )
            .map_err(Error::Exceeded)?;
        // The language's per-array limit is independent of bytes: the source traps on a larger array.
        if values.len().max(indices.len()).max(output_len) > crate::array_limits::MAX_ARRAY_LEN {
            return Err(Error::Budget);
        }
        let mut accesses = Vec::new();
        reserve(&mut accesses, indices.len())?;
        let mut written = Vec::new();
        if self.family == Family::InjectiveScatter {
            reserve(&mut written, output_len)?;
            written.resize(output_len, false);
        }
        for (iteration, &raw) in indices.iter().enumerate() {
            let indirect = usize::try_from(raw).map_err(|_| Error::Bounds { iteration })?;
            let (read_index, write_index) = match self.family {
                Family::Gather => (indirect, iteration),
                Family::InjectiveScatter => (iteration, indirect),
            };
            if read_index >= values.len() || write_index >= output_len {
                return Err(Error::Bounds { iteration });
            }
            if self.family == Family::InjectiveScatter {
                let seen = written
                    .get_mut(write_index)
                    .ok_or(Error::Bounds { iteration })?;
                if *seen {
                    return Err(Error::Duplicate {
                        iteration,
                        index: raw,
                    });
                }
                *seen = true;
            }
            accesses.push(Access {
                iteration,
                read_index,
                write_index,
            });
        }
        let mut output = Vec::new();
        reserve(&mut output, output_len)?;
        output.resize(output_len, 0);
        for access in &accesses {
            let value = values.get(access.read_index).ok_or(Error::Bounds {
                iteration: access.iteration,
            })?;
            let destination = output.get_mut(access.write_index).ok_or(Error::Bounds {
                iteration: access.iteration,
            })?;
            *destination = *value;
        }
        Ok(Candidate { output, accesses })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn issue_m4_05_gather_scatter_frontier_stale_net_rejects_extraction_and_candidate() {
        let source = b"fn gather(a: read Array<u32>, ix: read Array<u32>) -> Array<u32> { var out = filled_u32(len(ix),0u32); for i in 0u32..len(ix) { out[i] = a[ix[i]]; } return out; }";
        for mutation in 0..4 {
            let checked = check::check(front::parse(source).unwrap(), false).unwrap();
            let mut contract = Contract::from_checked(source, &checked, "gather").unwrap();
            match mutation {
                0 => contract.net.agents[0].next.position += 1,
                1 => contract.net.agents[0].span.end += 1,
                2 => contract.net.agents[0].rule = general_net::coverage::Rule::Binary,
                _ => contract.net.graph.revision += 1,
            }
            assert!(contract.verified_mir().is_err(), "mutation={mutation}");
            assert!(matches!(
                contract.candidate(&[7], &[0], None),
                Err(Error::Net(_))
            ));
        }
    }
}
