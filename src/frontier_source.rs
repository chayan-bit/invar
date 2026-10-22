//! Independent CPU observation of the frozen dense-mask frontier source form.
//! Source order is retained; this is not a parallel conflict or runtime permit.
#![forbid(unsafe_code)]

use crate::{
    budget::{BudgetExceeded, ByteBudget, Footprint, array_bytes},
    check::{self, CheckedProgram},
    front::{
        self, BinaryOp, Block, Expr, ExprKind, Mode, Place, Span, Stmt, StmtKind, Type, named,
        ungroup,
    },
    general_net, mir,
};

/// Observer resource limits, never a source duplicate/order policy. The default is unbounded:
/// a frontier never holds more than the vertex count and the work is bounded by vertices and
/// rounds, both of which the [`ByteBudget`] bounds through [`footprint`] (#278).
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub capacity: usize,
    pub work: u64,
}
impl Default for Budget {
    fn default() -> Self {
        Self {
            capacity: usize::MAX,
            work: u64::MAX,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Source(front::FrontError),
    CheckedMismatch,
    Unrecognized,
    Net(general_net::Error),
    /// Counter overflow; byte budgets refuse as `Exceeded`.
    Budget,
    /// The region needs more bytes than the [`ByteBudget`] holds (`BUDGET_EXCEEDED`).
    /// The caller keeps the original source path.
    Exceeded(BudgetExceeded),
    Bounds,
    Capacity {
        round: u32,
        required: usize,
        capacity: usize,
    },
    Work,
    Allocation,
}

/// One source-ordered literal `next[to] = 1` update. Duplicates stay recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Discovery {
    pub from: u32,
    pub to: u32,
    pub edge_index: usize,
}

/// An owning record is appended only after the entire mask-copy phase finishes.
#[derive(Debug)]
pub struct Round {
    number: u32,
    frontier_before: Vec<u32>,
    scheduled: Vec<u32>,
    processed: Vec<u32>,
    edge_reads: Vec<usize>,
    discoveries: Vec<Discovery>,
    next: Vec<u32>,
    seen_after: Vec<u32>,
}
impl Round {
    pub fn number(&self) -> u32 {
        self.number
    }
    pub fn frontier_before(&self) -> &[u32] {
        &self.frontier_before
    }
    /// Ascending positions of ones, including already-seen vertices.
    pub fn scheduled(&self) -> &[u32] {
        &self.scheduled
    }
    pub fn processed(&self) -> &[u32] {
        &self.processed
    }
    pub fn edge_reads(&self) -> &[usize] {
        &self.edge_reads
    }
    pub fn discoveries(&self) -> &[Discovery] {
        &self.discoveries
    }
    pub fn next(&self) -> &[u32] {
        &self.next
    }
    pub fn seen_after(&self) -> &[u32] {
        &self.seen_after
    }
}

#[derive(Debug)]
pub struct Candidate {
    count: i32,
    seen: Vec<u32>,
    rounds: Vec<Round>,
    work: u64,
}
impl Candidate {
    pub fn count(&self) -> i32 {
        self.count
    }
    pub fn seen(&self) -> &[u32] {
        &self.seen
    }
    pub fn rounds(&self) -> &[Round] {
        &self.rounds
    }
    pub fn work(&self) -> u64 {
        self.work
    }
}

#[derive(Debug)]
pub struct Contract {
    net: general_net::Net,
    proof: general_net::Proof,
    vertices: u32,
    rounds: u32,
    loop_span: Span,
    source_contract: [u8; 32],
    branches: [Span; 7],
}

fn number(e: &Expr) -> Option<u32> {
    let ExprKind::Integer(raw) = &ungroup(e).kind else {
        return None;
    };
    raw.strip_suffix("u32")
        .or_else(|| raw.strip_suffix("i32"))
        .unwrap_or(raw)
        .parse()
        .ok()
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
fn index<'a>(e: &'a Expr, name: &str) -> Option<&'a Expr> {
    match &ungroup(e).kind {
        ExprKind::Index { base, index } if named(base, name) => Some(index),
        _ => None,
    }
}
fn test_index(e: &Expr, array: &str, slot: &str, value: u32) -> bool {
    binary(e, BinaryOp::Eq).is_some_and(|(left, right)| {
        index(left, array).is_some_and(|i| named(i, slot)) && number(right) == Some(value)
    })
}
fn allocation(s: &Stmt) -> Option<(&str, u32)> {
    let StmtKind::Binding {
        mutable: true,
        name,
        value,
        ..
    } = &s.kind
    else {
        return None;
    };
    let ExprKind::Call { name: call, args } = &ungroup(value).kind else {
        return None;
    };
    let [count, fill] = args.as_slice() else {
        return None;
    };
    if call != "filled_u32" || number(fill) != Some(0) {
        return None;
    }
    Some((name.as_str(), number(count)?))
}
fn loop_at(s: &Stmt) -> Option<(&str, u32, &Block)> {
    let StmtKind::For {
        name,
        lower,
        upper,
        body,
    } = &s.kind
    else {
        return None;
    };
    if number(lower) != Some(0) {
        return None;
    }
    Some((name.as_str(), number(upper)?, body))
}
fn if_at(s: &Stmt) -> Option<(&Expr, &Block)> {
    match &s.kind {
        StmtKind::If {
            condition,
            then_block,
            else_arm: None,
        } => Some((condition, then_block)),
        _ => None,
    }
}
fn assign_index<'a>(s: &'a Stmt, array: &str, slot: &str) -> Option<&'a Expr> {
    match &s.kind {
        StmtKind::Assign {
            place: Place::Index { name, index, .. },
            value,
        } if name == array && named(index, slot) => Some(value),
        _ => None,
    }
}

impl Contract {
    /// Recognize the dense-mask/ascending-order form already frozen in the
    /// frontier_graph source fixtures, with source-constant vertex/round bounds.
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
        let [edges, start] = f.params.as_slice() else {
            return Err(Error::Unrecognized);
        };
        if edges.ty != Type::ArrayU32
            || edges.mode != Some(Mode::Read)
            || start.ty != Type::U32
            || start.mode.is_some()
            || f.result != Type::I32
            || f.return_from.is_some()
            || !f.generic_params.is_empty()
        {
            return Err(Error::Unrecognized);
        }
        let [
            frontier_alloc,
            seen_alloc,
            seed,
            round_loop,
            count_init,
            count_loop,
            ret,
        ] = f.body.statements.as_slice()
        else {
            return Err(Error::Unrecognized);
        };
        let (frontier, n) = allocation(frontier_alloc).ok_or(Error::Unrecognized)?;
        let (seen, seen_n) = allocation(seen_alloc).ok_or(Error::Unrecognized)?;
        if seen_n != n || assign_index(seed, frontier, &start.name).and_then(number) != Some(1) {
            return Err(Error::Unrecognized);
        }
        let (_, rounds, round_body) = loop_at(round_loop).ok_or(Error::Unrecognized)?;
        let [next_alloc, vertex_loop, copy_loop] = round_body.statements.as_slice() else {
            return Err(Error::Unrecognized);
        };
        let (next, next_n) = allocation(next_alloc).ok_or(Error::Unrecognized)?;
        let (u, vertex_n, vertex_body) = loop_at(vertex_loop).ok_or(Error::Unrecognized)?;
        let [vertex_if] = vertex_body.statements.as_slice() else {
            return Err(Error::Unrecognized);
        };
        let (condition, process) = if_at(vertex_if).ok_or(Error::Unrecognized)?;
        let vertex_and = ungroup(condition).span;
        let (active, unseen) = binary(condition, BinaryOp::And).ok_or(Error::Unrecognized)?;
        if next_n != n
            || vertex_n != n
            || !test_index(active, frontier, u, 1)
            || !test_index(unseen, seen, u, 0)
        {
            return Err(Error::Unrecognized);
        }
        let [mark, neighbor_loop] = process.statements.as_slice() else {
            return Err(Error::Unrecognized);
        };
        if assign_index(mark, seen, u).and_then(number) != Some(1) {
            return Err(Error::Unrecognized);
        }
        let (v, neighbor_n, neighbor_body) = loop_at(neighbor_loop).ok_or(Error::Unrecognized)?;
        let [neighbor_if] = neighbor_body.statements.as_slice() else {
            return Err(Error::Unrecognized);
        };
        let (condition, discover) = if_at(neighbor_if).ok_or(Error::Unrecognized)?;
        let neighbor_and = ungroup(condition).span;
        let (edge, unseen) = binary(condition, BinaryOp::And).ok_or(Error::Unrecognized)?;
        let (edge_value, one) = binary(edge, BinaryOp::Eq).ok_or(Error::Unrecognized)?;
        let offset = index(edge_value, &edges.name).ok_or(Error::Unrecognized)?;
        let (product, column) = binary(offset, BinaryOp::Add).ok_or(Error::Unrecognized)?;
        let (row, stride) = binary(product, BinaryOp::Mul).ok_or(Error::Unrecognized)?;
        let [store] = discover.statements.as_slice() else {
            return Err(Error::Unrecognized);
        };
        if neighbor_n != n
            || number(one) != Some(1)
            || !named(row, u)
            || number(stride) != Some(n)
            || !named(column, v)
            || !test_index(unseen, seen, v, 0)
            || assign_index(store, next, v).and_then(number) != Some(1)
        {
            return Err(Error::Unrecognized);
        }
        let (copy_i, copy_n, copy_body) = loop_at(copy_loop).ok_or(Error::Unrecognized)?;
        let [copy] = copy_body.statements.as_slice() else {
            return Err(Error::Unrecognized);
        };
        if copy_n != n
            || !assign_index(copy, frontier, copy_i)
                .and_then(|e| index(e, next))
                .is_some_and(|e| named(e, copy_i))
        {
            return Err(Error::Unrecognized);
        }
        let StmtKind::Binding {
            mutable: true,
            name: count,
            value,
            ..
        } = &count_init.kind
        else {
            return Err(Error::Unrecognized);
        };
        let (count_i, count_n, count_body) = loop_at(count_loop).ok_or(Error::Unrecognized)?;
        let [count_if] = count_body.statements.as_slice() else {
            return Err(Error::Unrecognized);
        };
        let (condition, body) = if_at(count_if).ok_or(Error::Unrecognized)?;
        let [increment] = body.statements.as_slice() else {
            return Err(Error::Unrecognized);
        };
        let StmtKind::Assign {
            place: Place::Name(destination, _),
            value: increment_value,
        } = &increment.kind
        else {
            return Err(Error::Unrecognized);
        };
        let (previous, one) = binary(increment_value, BinaryOp::Add).ok_or(Error::Unrecognized)?;
        if number(value) != Some(0)
            || count_n != n
            || !test_index(condition, seen, count_i, 1)
            || destination != count
            || !named(previous, count)
            || number(one) != Some(1)
            || !matches!(&ret.kind, StmtKind::Return(Some(e)) if named(e, count))
        {
            return Err(Error::Unrecognized);
        }
        let net = general_net::Net::lower(&checked, general_net::Limits::default())
            .map_err(Error::Net)?;
        let proof = net.verify().map_err(Error::Net)?;
        let m = net.compiled_mir(&proof).map_err(Error::Net)?;
        Ok(Self {
            vertices: n,
            rounds,
            loop_span: round_loop.span,
            source_contract: m.contract_digest(),
            branches: [
                vertex_loop.span,
                vertex_if.span,
                vertex_and,
                neighbor_loop.span,
                neighbor_if.span,
                neighbor_and,
                copy_loop.span,
            ],
            net,
            proof,
        })
    }
    pub fn vertices(&self) -> u32 {
        self.vertices
    }
    pub fn round_count(&self) -> u32 {
        self.rounds
    }
    pub fn loop_span(&self) -> Span {
        self.loop_span
    }
    /// Vertex loop, vertex if/and, neighbor loop, neighbor if/and, copy loop.
    /// Together with `loop_span`, these identify every branch of the rounds region.
    pub fn branch_spans(&self) -> &[Span; 7] {
        &self.branches
    }
    pub fn source_contract(&self) -> [u8; 32] {
        self.source_contract
    }
    pub fn verified_mir(&self) -> Result<mir::Program, Error> {
        self.net.compiled_mir(&self.proof).map_err(Error::Net)
    }

    /// Observe original source order with owned masks and ascending active lists.
    /// A refusal publishes neither a result nor partial round state. Capacity
    /// and work bounds limit this observer, not the original program's semantics.
    /// Work counts vertex, edge, mask-copy and final-count loop iterations; it is
    /// unrelated to semantic fuel, GPU events or native instruction counts.
    pub fn candidate(&self, edges: &[u32], start: u32, budget: Budget) -> Result<Candidate, Error> {
        self.candidate_within(edges, start, budget, &ByteBudget::default())
    }

    /// [`Self::candidate`] under `bytes`; a region whose [`footprint`] exceeds it is refused as
    /// `Exceeded` before anything is allocated.
    pub fn candidate_within(
        &self,
        edges: &[u32],
        start: u32,
        budget: Budget,
        bytes: &ByteBudget,
    ) -> Result<Candidate, Error> {
        self.verified_mir()?;
        let n = usize::try_from(self.vertices).map_err(|_| Error::Budget)?;
        let rounds = usize::try_from(self.rounds).map_err(|_| Error::Budget)?;
        bytes
            .admit("frontier", &footprint(n, rounds, edges.len()))
            .map_err(Error::Exceeded)?;
        // The language's per-array limit is independent of bytes: the source traps on a larger array.
        if edges.len().max(n) > crate::array_limits::MAX_ARRAY_LEN {
            return Err(Error::Budget);
        }
        let cells = n.checked_mul(n).ok_or(Error::Bounds)?;
        let start = usize::try_from(start).map_err(|_| Error::Bounds)?;
        if start >= n || edges.len() < cells {
            return Err(Error::Bounds);
        }
        let mut frontier = zeros(n)?;
        *frontier.get_mut(start).ok_or(Error::Bounds)? = 1;
        let mut seen = zeros(n)?;
        let mut records = Vec::new();
        reserve(
            &mut records,
            usize::try_from(self.rounds).map_err(|_| Error::Budget)?,
        )?;
        let mut work = Work {
            limit: budget.work,
            used: 0,
        };
        for round in 0..self.rounds {
            let before = snapshot(&frontier)?;
            let mut scheduled = Vec::new();
            reserve(&mut scheduled, n)?;
            for (u, bit) in frontier.iter().enumerate() {
                if *bit == 1 {
                    scheduled.push(u32::try_from(u).map_err(|_| Error::Bounds)?);
                }
            }
            capacity(round, scheduled.len(), budget.capacity)?;
            let mut next = zeros(n)?;
            let mut next_len = 0;
            let mut processed = Vec::new();
            reserve(&mut processed, n)?;
            let mut reads = Vec::new();
            reserve(&mut reads, cells)?;
            let mut discoveries = Vec::new();
            reserve(&mut discoveries, cells)?;
            for u in 0..n {
                work.tick()?;
                if *frontier.get(u).ok_or(Error::Bounds)? != 1
                    || *seen.get(u).ok_or(Error::Bounds)? != 0
                {
                    continue;
                }
                *seen.get_mut(u).ok_or(Error::Bounds)? = 1;
                processed.push(u32::try_from(u).map_err(|_| Error::Bounds)?);
                for v in 0..n {
                    work.tick()?;
                    let offset = u
                        .checked_mul(n)
                        .and_then(|i| i.checked_add(v))
                        .ok_or(Error::Bounds)?;
                    reads.push(offset);
                    if *edges.get(offset).ok_or(Error::Bounds)? == 1
                        && *seen.get(v).ok_or(Error::Bounds)? == 0
                    {
                        let slot = next.get_mut(v).ok_or(Error::Bounds)?;
                        if *slot == 0 {
                            next_len += 1;
                            capacity(round, next_len, budget.capacity)?;
                        }
                        *slot = 1;
                        discoveries.push(Discovery {
                            from: u32::try_from(u).map_err(|_| Error::Bounds)?,
                            to: u32::try_from(v).map_err(|_| Error::Bounds)?,
                            edge_index: offset,
                        });
                    }
                }
            }
            for i in 0..n {
                work.tick()?;
                *frontier.get_mut(i).ok_or(Error::Bounds)? = *next.get(i).ok_or(Error::Bounds)?;
            }
            records.push(Round {
                number: round,
                frontier_before: before,
                scheduled,
                processed,
                edge_reads: reads,
                discoveries,
                next,
                seen_after: snapshot(&seen)?,
            });
        }
        let mut count = 0i32;
        for bit in &seen {
            work.tick()?;
            if *bit == 1 {
                count = count.checked_add(1).ok_or(Error::Budget)?;
            }
        }
        Ok(Candidate {
            count,
            seen,
            rounds: records,
            work: work.used,
        })
    }
}

/// Host bytes one region holds: the resident `u32` adjacency matrix, the frontier and seen masks,
/// and for each round the retained record (before, scheduled, processed, next and seen-after
/// lists, one edge read and one discovery per matrix cell). `candidate_within` allocates all but
/// the matrix. Device bytes are the caller's to add.
pub fn footprint(vertices: usize, rounds: usize, edges: usize) -> Footprint {
    let cells = vertices.checked_mul(vertices);
    let lists = array_bytes::<u32>(vertices).and_then(|w| w.checked_mul(5));
    let per_round = lists
        .zip(cells.and_then(array_bytes::<usize>))
        .and_then(|(a, b)| a.checked_add(b))
        .zip(cells.and_then(array_bytes::<Discovery>))
        .and_then(|(a, b)| a.checked_add(b))
        .and_then(|a| a.checked_add(size_of::<Round>()))
        .and_then(|a| a.checked_mul(rounds));
    Footprint::ZERO
        .host(array_bytes::<u32>(edges))
        .host(vertices.checked_mul(2).and_then(array_bytes::<u32>))
        .host(per_round)
}

fn reserve<T>(v: &mut Vec<T>, n: usize) -> Result<(), Error> {
    v.try_reserve_exact(n).map_err(|_| Error::Allocation)
}
fn zeros(n: usize) -> Result<Vec<u32>, Error> {
    let mut v = Vec::new();
    reserve(&mut v, n)?;
    v.resize(n, 0);
    Ok(v)
}
fn snapshot<T: Copy>(source: &[T]) -> Result<Vec<T>, Error> {
    let mut v = Vec::new();
    reserve(&mut v, source.len())?;
    v.extend_from_slice(source);
    Ok(v)
}
fn capacity(round: u32, required: usize, limit: usize) -> Result<(), Error> {
    if required > limit {
        Err(Error::Capacity {
            round,
            required,
            capacity: limit,
        })
    } else {
        Ok(())
    }
}
struct Work {
    limit: u64,
    used: u64,
}
impl Work {
    fn tick(&mut self) -> Result<(), Error> {
        if self.used == self.limit {
            return Err(Error::Work);
        }
        self.used = self.used.checked_add(1).ok_or(Error::Work)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn issue_m4_05_gather_scatter_frontier_frozen_round_stale_net_rejects() {
        let source = include_bytes!("../bench/general-net-v01/current/frontier_graph_a.inv");
        for mutation in 0..4 {
            let p = check::check(front::parse(source).unwrap(), false).unwrap();
            let mut c = Contract::from_checked(source, &p, "reach").unwrap();
            match mutation {
                0 => c.net.agents[0].next.position += 1,
                1 => c.net.agents[0].span.end += 1,
                2 => c.net.agents[0].rule = general_net::coverage::Rule::Binary,
                _ => c.net.graph.revision += 1,
            }
            assert!(c.verified_mir().is_err());
            assert!(matches!(
                c.candidate(&[0; 25], 0, Budget::default()),
                Err(Error::Net(_))
            ));
        }
    }
}
