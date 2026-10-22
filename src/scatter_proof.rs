//! Static injectivity and destination-bound certificates for literal-domain
//! checked-u32 affine scatters. No scheduling or source-publication authority.
#![forbid(unsafe_code)]

pub mod host;

use crate::{
    budget::{BudgetExceeded, ByteBudget, Footprint, array_bytes},
    check::{self, CheckedProgram},
    front::{self, BinaryOp, Expr, ExprKind, Mode, Place, Span, StmtKind, Type, named, ungroup},
    general_net, mir,
};

pub const RULE_REVISION: &str = "checked-u32-affine-scatter-v1";

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Source(front::FrontError),
    CheckedMismatch,
    Unrecognized,
    NonInjective,
    Overflow,
    DestinationBounds,
    /// Work or staging limit of a host plan; byte budgets refuse as `Exceeded`.
    Budget,
    /// The region needs more bytes than the [`ByteBudget`] holds (`BUDGET_EXCEEDED`).
    /// The caller keeps the original source path.
    Exceeded(BudgetExceeded),
    InputBounds,
    Allocation,
    CertificateMismatch,
    Mir(mir::Error),
    Net(general_net::Error),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Shape {
    lower: u32,
    upper: u32,
    coefficient: u32,
    offset: u32,
    output_len: u32,
    loop_span: Span,
    destination_span: Span,
}

/// Host bytes one region holds: the resident `u32` input and the fresh `u32`
/// output, which `candidate_within` allocates. Device bytes are the caller's to add.
pub fn footprint(input: usize, output: usize) -> Footprint {
    Footprint::ZERO
        .host(array_bytes::<u32>(input))
        .host(array_bytes::<u32>(output))
}

/// Opaque, source-derived proof. No constructor, deserializer or proof flag.
/// For distinct i<j, positive a and checked endpoints imply
/// (a*j+b)-(a*i+b)=a*(j-i)>0 in mathematical integers. Domains with at most
/// one member have no distinct pair. Bounds are proved against fresh output.
#[derive(Debug, PartialEq, Eq)]
pub struct Certificate {
    revision: &'static str,
    shape: Shape,
    iterations: u32,
    endpoints: Option<(u32, u32)>,
    source_contract: [u8; 32],
}

impl Certificate {
    fn derive(shape: Shape, source_contract: [u8; 32]) -> Result<Self, Error> {
        let iterations = shape.upper.saturating_sub(shape.lower);
        // The certificate is O(1) checked arithmetic on the two endpoints, so it needs no size
        // cap; `candidate_within` admits the region's bytes against a `ByteBudget`. Only the
        // language's per-array limit applies: a larger output makes the source trap.
        if shape.output_len as usize > crate::array_limits::MAX_ARRAY_LEN {
            return Err(Error::Budget);
        }
        if iterations > 1 && shape.coefficient == 0 {
            return Err(Error::NonInjective);
        }
        let endpoints = if iterations == 0 {
            None
        } else {
            let at = |i: u32| {
                shape
                    .coefficient
                    .checked_mul(i)
                    .and_then(|product| product.checked_add(shape.offset))
                    .ok_or(Error::Overflow)
            };
            let first = at(shape.lower)?;
            let last = at(shape.upper - 1)?;
            if last >= shape.output_len {
                return Err(Error::DestinationBounds);
            }
            Some((first, last))
        };
        Ok(Self {
            revision: RULE_REVISION,
            shape,
            iterations,
            endpoints,
            source_contract,
        })
    }
    pub fn revision(&self) -> &'static str {
        self.revision
    }
    pub fn domain(&self) -> (u32, u32) {
        (self.shape.lower, self.shape.upper)
    }
    pub fn affine_map(&self) -> (u32, u32) {
        (self.shape.coefficient, self.shape.offset)
    }
    pub fn output_len(&self) -> u32 {
        self.shape.output_len
    }
    pub fn iterations(&self) -> u32 {
        self.iterations
    }
    /// None means the source body is never demanded.
    pub fn endpoints(&self) -> Option<(u32, u32)> {
        self.endpoints
    }
    pub fn loop_span(&self) -> Span {
        self.shape.loop_span
    }
    pub fn destination_span(&self) -> Span {
        self.shape.destination_span
    }
    pub fn source_contract(&self) -> [u8; 32] {
        self.source_contract
    }
}

#[derive(Debug)]
pub struct Contract {
    source: Vec<u8>,
    entry: String,
    net: general_net::Net,
    net_proof: general_net::Proof,
    certificate: Certificate,
}

fn number(e: &Expr) -> Option<u32> {
    let ExprKind::Integer(n) = &ungroup(e).kind else {
        return None;
    };
    n.strip_suffix("u32").unwrap_or(n).parse().ok()
}
fn binary(e: &Expr, op: BinaryOp) -> Option<(&Expr, &Expr)> {
    match &ungroup(e).kind {
        ExprKind::Binary {
            left,
            op: actual,
            right,
            ..
        } if *actual == op => Some((left, right)),
        _ => None,
    }
}
// Deliberately narrow: i, a*i (either operand order), optionally plus b
// (either operand order). Nested/distributed expressions remain unrecognized.
fn affine(e: &Expr, induction: &str) -> Option<(u32, u32)> {
    let product = |e: &Expr| {
        if named(e, induction) {
            Some(1)
        } else {
            let (left, right) = binary(e, BinaryOp::Mul)?;
            if named(left, induction) {
                number(right)
            } else if named(right, induction) {
                number(left)
            } else {
                None
            }
        }
    };
    if let Some(a) = product(e) {
        return Some((a, 0));
    }
    let (left, right) = binary(e, BinaryOp::Add)?;
    product(left)
        .zip(number(right))
        .or_else(|| product(right).zip(number(left)))
}

fn shape(checked: &CheckedProgram, entry: &str) -> Result<Shape, Error> {
    let f = checked
        .program
        .functions
        .iter()
        .find(|f| f.name == entry)
        .ok_or(Error::Unrecognized)?;
    let [input] = f.params.as_slice() else {
        return Err(Error::Unrecognized);
    };
    if input.mode != Some(Mode::Read)
        || input.ty != Type::ArrayU32
        || f.result != Type::ArrayU32
        || f.return_from.is_some()
        || !f.generic_params.is_empty()
    {
        return Err(Error::Unrecognized);
    }
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
    let ExprKind::Call { name, args } = &ungroup(value).kind else {
        return Err(Error::Unrecognized);
    };
    let [count, fill] = args.as_slice() else {
        return Err(Error::Unrecognized);
    };
    if name != "filled_u32" || number(fill) != Some(0) {
        return Err(Error::Unrecognized);
    }
    let output_len = number(count).ok_or(Error::Unrecognized)?;
    let StmtKind::For {
        name: induction,
        lower,
        upper,
        body,
    } = &region.kind
    else {
        return Err(Error::Unrecognized);
    };
    let lower = number(lower).ok_or(Error::Unrecognized)?;
    let upper = number(upper).ok_or(Error::Unrecognized)?;
    let [store] = body.statements.as_slice() else {
        return Err(Error::Unrecognized);
    };
    let StmtKind::Assign {
        place:
            Place::Index {
                name,
                index: destination,
                ..
            },
        value,
    } = &store.kind
    else {
        return Err(Error::Unrecognized);
    };
    let ExprKind::Index { base, index: read } = &ungroup(value).kind else {
        return Err(Error::Unrecognized);
    };
    let StmtKind::Return(Some(owner)) = &ret.kind else {
        return Err(Error::Unrecognized);
    };
    if name != output
        || !named(base, &input.name)
        || !named(read, induction)
        || (!named(owner, output)
            && !matches!(&ungroup(owner).kind, ExprKind::Move(n) if n == output))
    {
        return Err(Error::Unrecognized);
    }
    let (coefficient, offset) = affine(destination, induction).ok_or(Error::Unrecognized)?;
    Ok(Shape {
        lower,
        upper,
        coefficient,
        offset,
        output_len,
        loop_span: region.span,
        destination_span: destination.span,
    })
}

fn recheck(source: &[u8]) -> Result<CheckedProgram, Error> {
    check::check(front::parse(source).map_err(Error::Source)?, false).map_err(Error::Source)
}

impl Contract {
    /// Prove from checked source structure and literal bounds, without reading
    /// array payloads or enumerating concrete destination indices.
    pub fn from_checked(
        source: &[u8],
        supplied: &CheckedProgram,
        entry: &str,
    ) -> Result<Self, Error> {
        let checked = recheck(source)?;
        if checked.program != supplied.program
            || checked.expression_types != supplied.expression_types
            || checked.function_expression_types != supplied.function_expression_types
        {
            return Err(Error::CheckedMismatch);
        }
        let shape = shape(&checked, entry)?;
        let net = general_net::Net::lower(&checked, general_net::Limits::default())
            .map_err(Error::Net)?;
        let net_proof = net.verify().map_err(Error::Net)?;
        let m = net.compiled_mir(&net_proof).map_err(Error::Net)?;
        let certificate = Certificate::derive(shape, m.contract_digest())?;
        let mut retained = Vec::new();
        retained
            .try_reserve_exact(source.len())
            .map_err(|_| Error::Allocation)?;
        retained.extend_from_slice(source);
        let mut entry_name = String::new();
        entry_name
            .try_reserve_exact(entry.len())
            .map_err(|_| Error::Allocation)?;
        entry_name.push_str(entry);
        Ok(Self {
            source: retained,
            entry: entry_name,
            net,
            net_proof,
            certificate,
        })
    }
    pub fn certificate(&self) -> &Certificate {
        &self.certificate
    }
    /// Recheck current retained source, regenerate arithmetic obligations and
    /// compare exact source-lowered MIR with verified extraction. Hash equality
    /// binds identity; the algebraic check supplies the injectivity evidence.
    pub fn verified_mir(&self) -> Result<mir::Program, Error> {
        let m = self.net.compiled_mir(&self.net_proof).map_err(Error::Net)?;
        let checked = recheck(&self.source)?;
        let source_mir = mir::lower(&checked).map_err(Error::Mir)?;
        mir::verify(&source_mir).map_err(Error::Mir)?;
        if source_mir.contract_digest() != m.contract_digest() {
            return Err(Error::CertificateMismatch);
        }
        let expected = Certificate::derive(shape(&checked, &self.entry)?, m.contract_digest())?;
        if expected != self.certificate {
            return Err(Error::CertificateMismatch);
        }
        Ok(m)
    }
    /// Supplementary independent CPU observation. Input length is guarded, not
    /// statically proved. Arbitrary Rust slices are not live source-owner handles;
    /// this returns no fuel, execution, scheduling or publication permission.
    pub fn candidate(&self, input: &[u32]) -> Result<Vec<u32>, Error> {
        self.candidate_within(input, &ByteBudget::default())
    }
    /// [`Self::candidate`] under `budget`; a region whose [`footprint`] exceeds it
    /// is refused as `Exceeded`.
    pub fn candidate_within(&self, input: &[u32], budget: &ByteBudget) -> Result<Vec<u32>, Error> {
        self.verified_mir()?;
        let c = &self.certificate;
        let required = if c.iterations == 0 { 0 } else { c.shape.upper };
        budget
            .admit(
                "affine_scatter",
                &footprint(input.len(), c.shape.output_len as usize),
            )
            .map_err(Error::Exceeded)?;
        if input.len() > crate::array_limits::MAX_ARRAY_LEN {
            return Err(Error::Budget);
        }
        if input.len() < required as usize {
            return Err(Error::InputBounds);
        }
        let mut output = Vec::new();
        output
            .try_reserve_exact(c.shape.output_len as usize)
            .map_err(|_| Error::Allocation)?;
        output.resize(c.shape.output_len as usize, 0);
        for i in c.shape.lower..c.shape.upper {
            let destination = c
                .shape
                .coefficient
                .checked_mul(i)
                .and_then(|v| v.checked_add(c.shape.offset))
                .ok_or(Error::Overflow)?;
            let slot = output
                .get_mut(destination as usize)
                .ok_or(Error::DestinationBounds)?;
            *slot = *input.get(i as usize).ok_or(Error::InputBounds)?;
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const SOURCE: &[u8] = b"fn scatter(a: read Array<u32>) -> Array<u32> { var out = filled_u32(5u32,0u32); for i in 0u32..3u32 { out[2u32*i] = a[i]; } return out; }";
    #[test]
    fn issue_m4_05_gather_scatter_frontier_static_certificate_tamper() {
        for mutation in 0..10 {
            let p = recheck(SOURCE).unwrap();
            let mut c = Contract::from_checked(SOURCE, &p, "scatter").unwrap();
            match mutation {
                0 => c.certificate.shape.coefficient = 0,
                1 => c.certificate.shape.upper = 2,
                2 => c.certificate.shape.output_len = 6,
                3 => c.certificate.endpoints = Some((0, 3)),
                4 => c.certificate.iterations = 2,
                5 => c.certificate.revision = "untrusted",
                6 => c.certificate.source_contract[0] ^= 1,
                7 => c.certificate.shape.destination_span.end += 1,
                8 => c.certificate.shape.offset = 1,
                _ => c.certificate.shape.lower = 1,
            }
            assert!(matches!(c.verified_mir(), Err(Error::CertificateMismatch)));
            assert!(c.candidate(&[1, 2, 3]).is_err());
        }
    }

    #[test]
    fn issue_m4_05_gather_scatter_frontier_static_source_net_tamper() {
        for mutation in 0..5 {
            let p = recheck(SOURCE).unwrap();
            let mut c = Contract::from_checked(SOURCE, &p, "scatter").unwrap();
            match mutation {
                0 => c.net.agents[0].span.end += 1,
                1 => c.net.graph.revision += 1,
                2 => {
                    c.source = String::from_utf8(c.source)
                        .unwrap()
                        .replace("2u32*i", "1u32*i")
                        .into_bytes()
                }
                3 => c.entry = "missing".into(),
                _ => {
                    // A structurally valid but semantically changed, re-verified
                    // graph cannot inherit this source's injectivity certificate.
                    let mut m = c.net.compiled_mir(&c.net_proof).unwrap();
                    let instruction = m.functions[0]
                        .blocks
                        .iter_mut()
                        .flat_map(|b| &mut b.instructions)
                        .find(|i| matches!(i.kind, mir::Operation::U32(2)))
                        .unwrap();
                    instruction.kind = mir::Operation::U32(1);
                    mir::verify(&m).unwrap();
                    c.net = general_net::Net::encode(&m, general_net::Limits::default()).unwrap();
                    c.net_proof = c.net.verify().unwrap();
                }
            }
            assert!(c.verified_mir().is_err());
            assert!(c.candidate(&[1, 2, 3]).is_err());
        }
    }

    #[test]
    fn issue_m4_05_gather_scatter_frontier_static_independent_algebra_oracle() {
        let mut count = 0;
        for lower in 0..=3 {
            for upper in 0..=4 {
                for coefficient in 0..=4 {
                    for offset in 0..=3 {
                        for output_len in 0..=12 {
                            let shape = Shape {
                                lower,
                                upper,
                                coefficient,
                                offset,
                                output_len,
                                loop_span: Span { start: 0, end: 0 },
                                destination_span: Span { start: 0, end: 0 },
                            };
                            // Enumeration is an independent TEST oracle only.
                            // Production derivation uses endpoints and algebra.
                            let destinations: Vec<u64> = (lower..upper)
                                .map(|i| u64::from(coefficient) * u64::from(i) + u64::from(offset))
                                .collect();
                            let bounded = destinations.iter().all(|d| *d < u64::from(output_len));
                            let mut unique = destinations.clone();
                            unique.sort_unstable();
                            unique.dedup();
                            let lawful = bounded && unique.len() == destinations.len();
                            let result = Certificate::derive(shape, [0; 32]);
                            assert_eq!(result.is_ok(), lawful, "{shape:?}");
                            if let Ok(certificate) = result {
                                let endpoints = destinations
                                    .first()
                                    .zip(destinations.last())
                                    .map(|(first, last)| (*first as u32, *last as u32));
                                assert_eq!(certificate.endpoints(), endpoints, "{shape:?}");
                            }
                            count += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(count, 5200);
        println!("static scatter independent bounded algebra comparisons: {count}");
    }
}
