//! Source-derived M4-05 device plans. These APIs produce private completed
//! observations, never replace the MIR source fuel/publication protocol.
use crate::{check, front, frontier_source, histogram_source, indexed_source, scatter_proof};
mod affine;
use invar_device::{AddressSpace, Buffer, Capabilities, Layout, Span, indexed as device};

#[derive(Debug)]
pub enum Error {
    Source(front::FrontError),
    Unrecognized,
    Binding,
    Device(device::Error),
}
enum Contract {
    Indexed(indexed_source::Contract),
    Scatter(scatter_proof::Contract),
    Histogram(Box<histogram_source::Contract>),
    Frontier(frontier_source::Contract),
}
pub struct Plan {
    contract: Contract,
    source_identity: u64,
}
impl Plan {
    /// Actual checked source recognition and verified interaction extraction;
    /// histogram laws and affine injectivity are derived, never caller flags.
    pub fn from_source(source: &[u8], entry: &str) -> Result<Self, Error> {
        let checked = front::parse(source)
            .and_then(|p| check::check(p, false))
            .map_err(Error::Source)?;
        let contract = match indexed_source::Contract::from_checked(source, &checked, entry) {
            Ok(c) => Contract::Indexed(c),
            Err(indexed_source::Error::Unrecognized) => {
                match scatter_proof::Contract::from_checked(source, &checked, entry) {
                    Ok(c) => Contract::Scatter(c),
                    Err(scatter_proof::Error::Unrecognized) => {
                        match histogram_source::Contract::from_checked(source, &checked, entry) {
                            Ok(c) => Contract::Histogram(Box::new(c)),
                            Err(histogram_source::Error::Unrecognized) => Contract::Frontier(
                                frontier_source::Contract::from_checked(source, &checked, entry)
                                    .map_err(|e| match e {
                                        frontier_source::Error::Unrecognized => Error::Unrecognized,
                                        _ => Error::Binding,
                                    })?,
                            ),
                            Err(_) => return Err(Error::Binding),
                        }
                    }
                    Err(
                        scatter_proof::Error::NonInjective
                        | scatter_proof::Error::Overflow
                        | scatter_proof::Error::DestinationBounds
                        | scatter_proof::Error::Budget,
                    ) => return Err(Error::Unrecognized),
                    Err(_) => return Err(Error::Binding),
                }
            }
            Err(_) => return Err(Error::Binding),
        };
        Ok(Self {
            contract,
            source_identity: invar_device::source_hash(
                std::str::from_utf8(source).map_err(|_| Error::Binding)?,
            ),
        })
    }
    /// Scalar is scatter/histogram output length or frontier start. Gather uses
    /// None. Input copies and all guards precede device submission.
    pub fn bind(
        &self,
        values: &[u32],
        indices: &[u32],
        scalar: Option<u32>,
        caps: Capabilities,
    ) -> Result<device::Bound, Error> {
        let (operation, span, digest, output_len, start);
        let mut live_values = values;
        let mut live_indices = indices;
        let affine_indices;
        match &self.contract {
            Contract::Indexed(c) => {
                c.verified_mir().map_err(|_| Error::Binding)?;
                let gather = c.family() == indexed_source::Family::Gather;
                if gather != scalar.is_none() {
                    return Err(Error::Device(device::Error::Shape));
                }
                operation = if gather {
                    device::Operation::Gather
                } else {
                    device::Operation::InjectiveScatter
                };
                span = c.loop_span();
                digest = c.source_contract();
                output_len = if gather {
                    u32::try_from(live_indices.len())
                        .map_err(|_| Error::Device(device::Error::Limit))?
                } else {
                    scalar.ok_or(Error::Binding)?
                };
                start = 0;
            }
            Contract::Scatter(c) => {
                c.verified_mir().map_err(|_| Error::Binding)?;
                if !indices.is_empty() || scalar.is_some() {
                    return Err(Error::Device(device::Error::Shape));
                }
                let proof = c.certificate();
                let (lo, hi) = proof.domain();
                let (a, b) = proof.affine_map();
                live_values = values
                    .get(if hi <= lo {
                        0..0
                    } else {
                        lo as usize..hi as usize
                    })
                    .ok_or(Error::Device(device::Error::Index {
                        iteration: 0,
                        operand: "values",
                    }))?;
                affine_indices = affine::expand(lo, hi, a, b)?;
                live_indices = &affine_indices;
                operation = device::Operation::InjectiveScatter;
                span = proof.loop_span();
                digest = proof.source_contract();
                output_len = proof.output_len();
                start = 0;
            }
            Contract::Histogram(c) => {
                c.verified_mir().map_err(|_| Error::Binding)?;
                operation = device::Operation::Histogram;
                span = c.loop_span();
                digest = c.source_contract();
                output_len = scalar.ok_or(Error::Device(device::Error::Shape))?;
                start = 0;
            }
            Contract::Frontier(c) => {
                c.verified_mir().map_err(|_| Error::Binding)?;
                if !indices.is_empty()
                    || c.vertices() > device::MAX_VERTICES
                    || c.round_count() > device::MAX_ROUNDS
                {
                    return Err(Error::Device(device::Error::Limit));
                }
                operation = device::Operation::Frontier {
                    vertices: c.vertices(),
                    rounds: c.round_count(),
                    capacity: c.vertices(),
                };
                span = c.loop_span();
                digest = c.source_contract();
                output_len = c
                    .round_count()
                    .checked_add(1)
                    .and_then(|r| c.vertices().checked_mul(r))
                    .ok_or(Error::Device(device::Error::Limit))?;
                start = scalar.ok_or(Error::Device(device::Error::Shape))?;
            }
        }
        // The exact extracted-MIR digest seals all source roles. Revision zero
        // means no subsequent graph rewrite, not an imported live-graph claim.
        let buffer = |id, elements| Buffer {
            id,
            elements,
            address: AddressSpace::Device,
            layout: Layout::ContiguousU32,
        };
        let ir = device::Ir {
            version: device::VERSION,
            source_identity: self.source_identity,
            graph_revision: 0,
            source_contract: digest,
            span: Span {
                start: span.start,
                end: span.end,
            },
            operation,
            values: buffer(
                0,
                u32::try_from(live_values.len())
                    .map_err(|_| Error::Device(device::Error::Limit))?,
            ),
            indices: buffer(
                1,
                u32::try_from(live_indices.len())
                    .map_err(|_| Error::Device(device::Error::Limit))?,
            ),
            output: buffer(2, output_len),
            law: (operation == device::Operation::Histogram)
                .then(|| device::HISTOGRAM_LAW.to_owned()),
        };
        device::verify(ir.clone(), &ir, caps)
            .and_then(|v| v.bind(live_values, live_indices, start))
            .map_err(Error::Device)
    }
}

// Reached-source metadata stays with the family. The scheduler only selects a
// backend; it cannot manufacture source roles, branch counts or completion.
use crate::{
    front::{BinaryOp, Expr, ExprKind, Place, Stmt, StmtKind},
    mir,
    mir_eval::{
        LiveLoop,
        region::{Branch, CompletedRegion, Elements, Write},
    },
};
use std::rc::Rc;

pub(crate) struct SourcePlan {
    plan: Plan,
    function: String,
    region: Stmt,
    values: String,
    indices: Option<String>,
    scalar: Option<String>,
    output: String,
    seen: Option<String>,
    next: Option<String>,
    ticket: Ticket,
}
enum Ticket {
    Gather(indexed_source::host::Plan),
    Scatter(scatter_proof::host::Plan),
    Histogram(Box<histogram_source::host::Plan>),
    IndexedScatter,
    Frontier,
}

#[cfg_attr(
    not(any(test, feature = "cuda", all(feature = "metal", target_os = "macos"))),
    expect(dead_code, reason = "used by indexed device adapters")
)]
impl SourcePlan {
    pub(crate) fn recognize(
        source: &[u8],
        program: &mir::Program,
        entry: &str,
        span: crate::front::Span,
    ) -> Result<Option<Self>, Error> {
        let plan = match Plan::from_source(source, entry) {
            Ok(p) => p,
            Err(Error::Unrecognized) => return Ok(None),
            Err(e) => return Err(e),
        };
        let (own_span, verified) = match &plan.contract {
            Contract::Indexed(c) => (c.loop_span(), c.verified_mir().map_err(|_| Error::Binding)?),
            Contract::Scatter(c) => (
                c.certificate().loop_span(),
                c.verified_mir().map_err(|_| Error::Binding)?,
            ),
            Contract::Histogram(c) => {
                (c.loop_span(), c.verified_mir().map_err(|_| Error::Binding)?)
            }
            Contract::Frontier(c) => (c.loop_span(), c.verified_mir().map_err(|_| Error::Binding)?),
        };
        if own_span != span {
            return Ok(None);
        }
        if verified.contract_digest() != program.contract_digest() {
            return Err(Error::Binding);
        }
        let checked = front::parse(source)
            .and_then(|p| check::check(p, false))
            .map_err(Error::Source)?;
        let f = checked
            .program
            .functions
            .iter()
            .find(|f| f.name == entry)
            .ok_or(Error::Binding)?;
        let binding = |s: &Stmt| match &s.kind {
            StmtKind::Binding { name, .. } => Ok(name.clone()),
            _ => Err(Error::Binding),
        };
        let region = f
            .body
            .statements
            .iter()
            .find(|s| s.span == span)
            .ok_or(Error::Binding)?
            .clone();
        let mut indices = f.params.get(1).map(|p| p.name.clone());
        let mut scalar = f.params.get(2).map(|p| p.name.clone());
        let mut seen = None;
        let mut next = None;
        let ticket = match &plan.contract {
            Contract::Indexed(c) if c.family() == indexed_source::Family::Gather => Ticket::Gather(
                indexed_source::host::Plan::bind(Rc::from(source), &checked, entry, program)
                    .map_err(|_| Error::Binding)?,
            ),
            Contract::Indexed(_) => Ticket::IndexedScatter,
            Contract::Scatter(c) => {
                // The existing reached-source scatter ticket and bridge both
                // require a zero-based fresh induction. Never renumber it.
                if c.certificate().domain().0 != 0 {
                    return Ok(None);
                }
                indices = None;
                scalar = None;
                Ticket::Scatter(
                    scatter_proof::host::Plan::bind(Rc::from(source), &checked, entry, program)
                        .map_err(|_| Error::Binding)?,
                )
            }
            Contract::Histogram(_) => Ticket::Histogram(Box::new(
                histogram_source::host::Plan::bind(
                    source,
                    &checked,
                    entry,
                    Rc::new(program.clone()),
                )
                .map_err(|_| Error::Binding)?,
            )),
            Contract::Frontier(_) => {
                indices = None;
                scalar = f.params.get(1).map(|p| p.name.clone());
                seen = Some(binding(f.body.statements.get(1).ok_or(Error::Binding)?)?);
                let StmtKind::For { body, .. } = &region.kind else {
                    return Err(Error::Binding);
                };
                next = Some(binding(body.statements.first().ok_or(Error::Binding)?)?);
                Ticket::Frontier
            }
        };
        Ok(Some(Self {
            plan,
            function: entry.into(),
            region,
            values: f.params[0].name.clone(),
            indices,
            scalar,
            output: binding(f.body.statements.first().ok_or(Error::Binding)?)?,
            seen,
            next,
            ticket,
        }))
    }

    /// The PNF family this plan implements; the #263 lowering requires the
    /// normal form to admit exactly this family at the plan's loop.
    pub(crate) fn family(&self) -> crate::general_net::regions::family::Family {
        use crate::general_net::regions::family::Family;
        match self.ticket {
            Ticket::Gather(_) => Family::Gather,
            Ticket::IndexedScatter => Family::Scatter,
            Ticket::Scatter(_) => Family::AffineScatter,
            Ticket::Histogram(_) => Family::Histogram,
            Ticket::Frontier => Family::Frontier,
        }
    }

    /// All live guards and private source checks precede submission. Existing
    /// source observers retain their own budgets; no device cap lifts them.
    pub(crate) fn prepare(
        &self,
        live: &LiveLoop<'_>,
        caps: Capabilities,
    ) -> Result<Prepared, Error> {
        if live.function != self.function || live.span != self.region.span || live.iterations == 0 {
            return Err(Error::Binding);
        }
        let values = live.array(&self.values).ok_or(Error::Binding)?;
        let indices = self
            .indices
            .as_ref()
            .map(|n| live.array(n).ok_or(Error::Binding))
            .transpose()?;
        let ix = indices.as_deref().map_or(&[][..], |v| v.as_slice());
        let scalar = self
            .scalar
            .as_ref()
            .map(|n| live.scalars.get(n).copied().ok_or(Error::Binding))
            .transpose()?;
        let writes = [&self.output]
            .into_iter()
            .chain(self.seen.iter())
            .collect::<Vec<_>>();
        for name in &writes {
            if live.owned(name) != Some(true)
                || live.same_object(name, &self.values)
                || self
                    .indices
                    .as_ref()
                    .is_some_and(|i| live.same_object(name, i))
                || writes
                    .iter()
                    .any(|other| *other != *name && live.same_object(name, other))
            {
                return Err(Error::Binding);
            }
        }
        let output = live.array(&self.output).ok_or(Error::Binding)?;
        let mut branches = vec![Branch {
            span: self.region.span,
            taken: live.iterations as u64,
            skipped: 1,
        }];
        let mut cleanup = Vec::new();
        let (expected, final_frontier) = match &self.plan.contract {
            Contract::Indexed(c) => (
                c.candidate(&values, ix, scalar)
                    .map_err(|_| Error::Binding)?
                    .output()
                    .to_vec(),
                None,
            ),
            Contract::Scatter(c) => (c.candidate(&values).map_err(|_| Error::Binding)?, None),
            Contract::Histogram(c) => (
                c.candidate(&values, ix, scalar.ok_or(Error::Binding)?)
                    .map_err(|_| Error::Binding)?
                    .output()
                    .to_vec(),
                None,
            ),
            Contract::Frontier(c) => {
                let n = c.vertices() as usize;
                let start = scalar.ok_or(Error::Binding)? as usize;
                let seen = live
                    .array(self.seen.as_ref().ok_or(Error::Binding)?)
                    .ok_or(Error::Binding)?;
                if live.iterations != c.round_count() as usize
                    || output.len() != n
                    || seen.len() != n
                    || seen.iter().any(|&v| v != 0)
                    || output
                        .iter()
                        .enumerate()
                        .any(|(i, &v)| v != u32::from(i == start))
                {
                    return Err(Error::Binding);
                }
                let candidate = c
                    .candidate(&values, scalar.ok_or(Error::Binding)?, Default::default())
                    .map_err(|_| Error::Binding)?;
                let (mut processed, mut scheduled, mut reads, mut edges, mut discoveries) =
                    (0, 0, 0, 0, 0);
                for r in candidate.rounds() {
                    processed += r.processed().len() as u64;
                    scheduled += r.scheduled().len() as u64;
                    reads += r.edge_reads().len() as u64;
                    edges += r.edge_reads().iter().filter(|&&i| values[i] == 1).count() as u64;
                    discoveries += r.discoveries().len() as u64;
                    cleanup.push((self.next.as_ref().ok_or(Error::Binding)?.clone(), n));
                }
                let vertices = n as u64 * live.iterations as u64;
                let counts = [
                    (vertices, live.iterations as u64),
                    (processed, vertices - processed),
                    (scheduled, vertices - scheduled),
                    (reads, processed),
                    (discoveries, reads - discoveries),
                    (edges, reads - edges),
                    (vertices, live.iterations as u64),
                ];
                branches.extend(c.branch_spans().iter().zip(counts).map(
                    |(&span, (taken, skipped))| Branch {
                        span,
                        taken,
                        skipped,
                    },
                ));
                let masks = candidate
                    .rounds()
                    .iter()
                    .map(|r| r.next().to_vec())
                    .collect::<Vec<_>>();
                (candidate.seen().to_vec(), Some(masks))
            }
        };
        let bound = self.plan.bind(&values, ix, scalar, caps)?;
        let trips = match bound.plan().ir().operation {
            device::Operation::Frontier { rounds, .. } => rounds as usize,
            _ => bound.indices().len(),
        };
        if trips != live.iterations {
            return Err(Error::Binding);
        }
        let fuel = match &self.ticket {
            Ticket::Gather(p) => p.loop_ticks(trips),
            Ticket::Scatter(p) => p.loop_ticks(),
            Ticket::Histogram(p) => p.loop_ticks(trips),
            _ => loop_fuel(&self.region, &branches),
        }
        .filter(|&n| n <= live.fuel_remaining)
        .ok_or(Error::Binding)?;
        let mut region = CompletedRegion {
            function: self.function.clone(),
            span: self.region.span,
            branches,
            writes: vec![],
            scalars: vec![],
            cleanup,
            fuel,
            trap: None,
        };
        let expected = if let Some(masks) = final_frontier {
            let frontier = masks.last().ok_or(Error::Binding)?.clone();
            region.writes.push(Write {
                owner: self.output.clone(),
                length: output.len(),
                range: 0..output.len(),
                values: Elements::U32(frontier),
            });
            region.writes.push(Write {
                owner: self.seen.as_ref().ok_or(Error::Binding)?.clone(),
                length: expected.len(),
                range: 0..expected.len(),
                values: Elements::U32(expected.clone()),
            });
            let compaction_maps = masks
                .iter()
                .map(|m| {
                    m.iter()
                        .enumerate()
                        .filter_map(|(i, &v)| (v == 1).then_some(i as u32))
                        .collect()
                })
                .collect();
            device::Output {
                values: expected,
                frontier_masks: masks,
                compaction_maps,
            }
        } else {
            if output.len() != expected.len() || output.iter().any(|&v| v != 0) {
                return Err(Error::Binding);
            }
            region.writes.push(Write {
                owner: self.output.clone(),
                length: output.len(),
                range: 0..output.len(),
                values: Elements::U32(expected.clone()),
            });
            device::Output {
                values: expected,
                frontier_masks: vec![],
                compaction_maps: vec![],
            }
        };
        Ok(Prepared {
            bound,
            region,
            expected,
        })
    }
}

pub(crate) struct Prepared {
    pub(crate) bound: device::Bound,
    region: CompletedRegion,
    expected: device::Output,
}
#[cfg_attr(
    not(any(test, feature = "cuda", all(feature = "metal", target_os = "macos"))),
    expect(dead_code, reason = "used by indexed device adapters")
)]
impl Prepared {
    pub(crate) fn cost(
        &self,
        model: crate::plan::DeviceCostModel,
    ) -> Option<crate::plan::DeviceCost> {
        let launches = device::schedule(self.bound.plan()).len();
        let (elements, operations) = match self.bound.plan().ir().operation {
            device::Operation::Frontier {
                vertices, rounds, ..
            } => (
                (vertices as usize)
                    .checked_mul(vertices as usize)?
                    .checked_mul(rounds as usize)?,
                2,
            ),
            device::Operation::Histogram => (self.bound.indices().len(), 3),
            _ => (self.bound.indices().len(), 1),
        };
        elements.checked_mul(operations)?.checked_mul(4)?;
        let cost = model.estimate(self.bound.transfer_bytes(), elements, operations, launches);
        (cost.cpu_ms.is_finite() && cost.device_ms.is_finite()).then_some(cost)
    }
    pub(crate) fn complete(
        mut self,
        receipt: device::Receipt,
        target: &str,
        kernel_hash: u64,
    ) -> Result<CompletedRegion, Error> {
        if receipt.target != target
            || receipt.reason != "verified-u32-completed"
            || receipt.source_hash != kernel_hash
            || receipt.completed_phases as usize != device::schedule(self.bound.plan()).len()
            || receipt.transfer_bytes != self.bound.transfer_bytes()
            || receipt.output != self.expected
        {
            return Err(Error::Binding);
        }
        // Publish actual completed device words, after every word and all round
        // records agree. No compiler result or incomplete fence reaches here.
        if self.region.writes.len() == 2 {
            self.region.writes[0].values = Elements::U32(
                receipt
                    .output
                    .frontier_masks
                    .last()
                    .ok_or(Error::Binding)?
                    .clone(),
            );
            self.region.writes[1].values = Elements::U32(receipt.output.values);
        } else {
            self.region.writes[0].values = Elements::U32(receipt.output.values);
        }
        Ok(self.region)
    }
}

// Source ticks, independently of MIR's publication-side block-tick derivation.
// Only the already-recognized indexed/frontier syntax is accepted here.
fn expression_fuel(e: &Expr, times: u64, branches: &[Branch]) -> Option<u64> {
    let child = match &e.kind {
        ExprKind::Group(inner) => return expression_fuel(inner, times, branches),
        ExprKind::Name(_) | ExprKind::Integer(_) => 0,
        ExprKind::Index { base, index } => expression_fuel(base, times, branches)?
            .checked_add(expression_fuel(index, times, branches)?)?,
        ExprKind::Call { args, .. } => args.iter().try_fold(0u64, |n, a| {
            n.checked_add(expression_fuel(a, times, branches)?)
        })?,
        ExprKind::Binary {
            left, op, right, ..
        } => {
            let rhs = if *op == BinaryOp::And {
                branches.iter().find(|b| b.span == e.span)?.taken
            } else {
                times
            };
            expression_fuel(left, times, branches)?
                .checked_add(expression_fuel(right, rhs, branches)?)?
        }
        _ => return None,
    };
    times.checked_add(child)
}
fn statement_fuel(s: &Stmt, times: u64, branches: &[Branch]) -> Option<u64> {
    let child = match &s.kind {
        StmtKind::Binding { value, .. } => expression_fuel(value, times, branches)?,
        StmtKind::Assign {
            place: Place::Index { index, .. },
            value,
        } => expression_fuel(value, times, branches)?
            .checked_add(expression_fuel(index, times, branches)?)?,
        StmtKind::If {
            condition,
            then_block,
            else_arm: None,
        } => {
            let taken = branches.iter().find(|b| b.span == s.span)?.taken;
            expression_fuel(condition, times, branches)?.checked_add(
                then_block.statements.iter().try_fold(0u64, |n, s| {
                    n.checked_add(statement_fuel(s, taken, branches)?)
                })?,
            )?
        }
        StmtKind::For { lower, upper, .. } => expression_fuel(lower, times, branches)?
            .checked_add(expression_fuel(upper, times, branches)?)?
            .checked_add(loop_fuel(s, branches)?)?,
        _ => return None,
    };
    times.checked_add(child)
}
fn loop_fuel(s: &Stmt, branches: &[Branch]) -> Option<u64> {
    let StmtKind::For { body, .. } = &s.kind else {
        return None;
    };
    let taken = branches.iter().find(|b| b.span == s.span)?.taken;
    body.statements.iter().try_fold(taken, |n, s| {
        n.checked_add(statement_fuel(s, taken, branches)?)
    })
}

#[cfg(test)]
mod source_tests;
