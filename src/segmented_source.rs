//! Ordinary-source segmented wrapping-u32 reductions, with a bounded CPU candidate.
//! This module never publishes source state or debits fuel. Runtime loans, reached
//! loop admission and source completion must be supplied by a separate host.
#![forbid(unsafe_code)]

use crate::{
    check,
    front::{
        self, BinaryOp, Expr, ExprKind, Mode, Place, Span, Stmt, StmtKind, Type, named, ungroup,
    },
    general_net,
    laws::{self, Law},
    mir,
    typed_net::{self, Effect, Net, Node, Op, Plan},
};
use sha2::{Digest, Sha256};

pub const RULE: &str = "source-segmented-wrapping-u32-v1";

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Source(front::FrontError),
    Net(general_net::Error),
    Typed(typed_net::Error),
    Reduction(laws::Error),
    Unrecognized,
    SourceMismatch,
    MirMismatch,
    ExtractionMismatch,
    OffsetShape,
    OffsetOrder,
    OffsetBounds,
    IncompleteCoverage,
    Budget,
    Allocation,
    FuelStructure,
}

/// Logical candidate limits, including the fresh output, but not borrowed inputs.
/// These are neither RSS limits nor device residency measurements.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub max_elements: usize,
    pub max_segments: usize,
    pub max_output_bytes: u64,
    pub max_work: u64,
}

/// Private checked-source/verified-MIR binding for one exact reduction region.
#[derive(Debug)]
pub struct Contract {
    net: general_net::Net,
    proof: general_net::Proof,
    source_hash: [u8; 32],
    mir_hash: [u8; 32],
    entry: String,
    names: [String; 4], // input, offsets, segment count, fresh output
    span: Span,
    inner_span: Span,
    law: Law,
    graph: Net,
    plan: Plan,
    loop_ticks: [u64; 2], // per segment, per demanded element, derived from checked AST
}

fn integer(e: &Expr) -> Option<u32> {
    match &ungroup(e).kind {
        ExprKind::Integer(n) => n.strip_suffix("u32").unwrap_or(n).parse().ok(),
        _ => None,
    }
}
fn index<'a>(e: &'a Expr, name: &str) -> Option<&'a Expr> {
    match &ungroup(e).kind {
        ExprKind::Index { base, index } if named(base, name) => Some(index),
        _ => None,
    }
}
fn call<'a>(e: &'a Expr, name: &str) -> Option<&'a [Expr]> {
    match &ungroup(e).kind {
        ExprKind::Call { name: n, args } if n == name => Some(args),
        _ => None,
    }
}
fn successor(e: &Expr, name: &str) -> bool {
    matches!(&ungroup(e).kind, ExprKind::Binary { left, op: BinaryOp::Add, right, .. }
        if named(left, name) && integer(right) == Some(1))
}
// Match the source tick convention used by the transpose extraction: groups
// carry spans only; every admitted expression and statement dispatch charges
// once. Unknown expressions are refused instead of assigned an estimated cost.
pub(crate) fn expr_ticks(e: &Expr) -> Option<u64> {
    match &e.kind {
        ExprKind::Group(inner) => expr_ticks(inner),
        ExprKind::Name(_) | ExprKind::Integer(_) => Some(1),
        ExprKind::Binary { left, right, .. }
        | ExprKind::Index {
            base: left,
            index: right,
        } => expr_ticks(left)?
            .checked_add(expr_ticks(right)?)?
            .checked_add(1),
        ExprKind::Call { args, .. } => args
            .iter()
            .try_fold(1u64, |total, arg| total.checked_add(expr_ticks(arg)?)),
        _ => None,
    }
}
pub(crate) fn stmt_ticks(s: &Stmt) -> Option<u64> {
    match &s.kind {
        StmtKind::Binding { value, .. } => expr_ticks(value)?.checked_add(1),
        StmtKind::Assign { place, value } => {
            expr_ticks(value)?.checked_add(1)?.checked_add(match place {
                Place::Name(..) => 0,
                Place::Index { index, .. } => expr_ticks(index)?,
            })
        }
        // This is loop entry only. Per-iteration control/body ticks are added
        // separately after the entire narrow loop shape has been recognized.
        StmtKind::For { lower, upper, .. } => expr_ticks(lower)?
            .checked_add(expr_ticks(upper)?)?
            .checked_add(1),
        _ => None,
    }
}
fn loop_ticks(initialize: &Stmt, inner: &Stmt, update: &Stmt, store: &Stmt) -> Option<[u64; 2]> {
    let per_segment = 1u64
        .checked_add(stmt_ticks(initialize)?)?
        .checked_add(stmt_ticks(inner)?)?
        .checked_add(stmt_ticks(store)?)?;
    let per_element = 1u64.checked_add(stmt_ticks(update)?)?;
    Some([per_segment, per_element])
}
fn graph(law: Law) -> Result<(Net, Plan), Error> {
    let mut graph = Net::empty(2);
    for (source_id, ty, op, inputs) in [
        (0, typed_net::Type::Nat, Op::InputScalar(0), vec![]),
        (1, typed_net::Type::ArrayU32, Op::InputU32(0), vec![]),
        (
            2,
            typed_net::Type::Nat,
            Op::Reduce {
                law,
                rule: laws::RULE_REVISION,
            },
            vec![0, 1],
        ),
    ] {
        graph.nodes.push(Some(Node {
            source_id,
            ty,
            effect: Effect::Total,
            inputs,
            op,
        }));
    }
    graph.seal_source();
    let proof = graph
        .verify(crate::loops::GRAPH_LIMITS)
        .map_err(Error::Typed)?;
    let plan = graph.extract(&proof).map_err(Error::Typed)?;
    Ok((graph, plan))
}

impl Contract {
    /// Parse and check actual source, then recognize a narrow, rename-independent
    /// nested loop. Unknown shapes/callbacks are explicit admission misses.
    pub fn from_source(source: &[u8], entry: &str) -> Result<Self, Error> {
        let checked = check::check(front::parse(source).map_err(Error::Source)?, false)
            .map_err(Error::Source)?;
        let f = checked.program.function(entry).ok_or(Error::Unrecognized)?;
        if f.params.len() != 3
            || f.result != Type::ArrayU32
            || f.return_from.is_some()
            || !f.generic_params.is_empty()
            || f.params[..2]
                .iter()
                .any(|p| p.ty != Type::ArrayU32 || p.mode != Some(Mode::Read))
            || f.params[2].ty != Type::U32
            || f.params[2].mode.is_some()
        {
            return Err(Error::Unrecognized);
        }
        let input = &f.params[0].name;
        let offsets = &f.params[1].name;
        let count = &f.params[2].name;
        let [allocate, outer, ret] = f.body.statements.as_slice() else {
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
        let Some([length, fill]) = call(value, "filled_u32") else {
            return Err(Error::Unrecognized);
        };
        if !named(length, count) || integer(fill) != Some(0) {
            return Err(Error::Unrecognized);
        }
        if !matches!(&ret.kind, StmtKind::Return(Some(e))
            if named(e, output) || matches!(&ungroup(e).kind, ExprKind::Move(n) if n == output))
        {
            return Err(Error::Unrecognized);
        }
        let StmtKind::For {
            name: segment,
            lower,
            upper,
            body,
        } = &outer.kind
        else {
            return Err(Error::Unrecognized);
        };
        if integer(lower) != Some(0) || !named(upper, count) {
            return Err(Error::Unrecognized);
        }
        let [initialize, inner, store] = body.statements.as_slice() else {
            return Err(Error::Unrecognized);
        };
        let StmtKind::Binding {
            mutable: true,
            name: state,
            value: seed,
            ..
        } = &initialize.kind
        else {
            return Err(Error::Unrecognized);
        };
        let StmtKind::For {
            name: element,
            lower,
            upper,
            body,
        } = &inner.kind
        else {
            return Err(Error::Unrecognized);
        };
        if !index(lower, offsets).is_some_and(|e| named(e, segment))
            || !index(upper, offsets).is_some_and(|e| successor(e, segment))
        {
            return Err(Error::Unrecognized);
        }
        let [update] = body.statements.as_slice() else {
            return Err(Error::Unrecognized);
        };
        let StmtKind::Assign {
            place: Place::Name(n, _),
            value,
        } = &update.kind
        else {
            return Err(Error::Unrecognized);
        };
        let law = [
            ("wrapping_add", Law::WrappingSum),
            ("wrapping_mul", Law::WrappingProduct),
        ]
        .into_iter()
        .find_map(|(name, law)| {
            let [left, right] = call(value, name)? else {
                return None;
            };
            (n == state
                && integer(seed) == Some(law.identity())
                && named(left, state)
                && index(right, input).is_some_and(|e| named(e, element)))
            .then_some(law)
        })
        .ok_or(Error::Unrecognized)?;
        if !matches!(&store.kind, StmtKind::Assign { place: Place::Index { name, index, .. }, value }
            if name == output && named(index, segment) && named(value, state))
        {
            return Err(Error::Unrecognized);
        }
        let loop_ticks =
            loop_ticks(initialize, inner, update, store).ok_or(Error::FuelStructure)?;
        let names = [
            input.clone(),
            offsets.clone(),
            count.clone(),
            output.clone(),
        ];
        let span = outer.span;
        let net = general_net::Net::lower(&checked, general_net::Limits::default())
            .map_err(Error::Net)?;
        let proof = net.verify().map_err(Error::Net)?;
        let m = net.compiled_mir(&proof).map_err(Error::Net)?;
        let (graph, plan) = graph(law)?;
        Ok(Self {
            net,
            proof,
            source_hash: Sha256::digest(source).into(),
            mir_hash: m.contract_digest(),
            entry: entry.to_owned(),
            names,
            span,
            inner_span: inner.span,
            law,
            graph,
            plan,
            loop_ticks,
        })
    }
    pub fn function(&self) -> &str {
        &self.entry
    }
    pub fn names(&self) -> &[String; 4] {
        &self.names
    }
    pub fn loop_span(&self) -> Span {
        self.span
    }
    pub fn inner_span(&self) -> Span {
        self.inner_span
    }
    pub fn law(&self) -> Law {
        self.law
    }
    pub fn mir_contract(&self) -> [u8; 32] {
        self.mir_hash
    }
    pub fn source_hash(&self) -> [u8; 32] {
        self.source_hash
    }
    pub fn graph(&self) -> &Net {
        &self.graph
    }
    pub fn plan(&self) -> &Plan {
        &self.plan
    }
    pub fn verified_mir(&self) -> Result<mir::Program, Error> {
        self.net.compiled_mir(&self.proof).map_err(Error::Net)
    }
    /// Exact source bytes, rather than a caller-supplied identity, bind this candidate.
    pub fn verify_source(&self, source: &[u8]) -> Result<(), Error> {
        if <[u8; 32]>::from(Sha256::digest(source)) != self.source_hash {
            return Err(Error::SourceMismatch);
        }
        Ok(())
    }
    /// Reverify before comparing the entire source MIR, including numeric/layout policy.
    pub fn verify_mir(&self, m: &mir::Program) -> Result<(), Error> {
        mir::verify(m).map_err(|_| Error::MirMismatch)?;
        if m.contract_digest() != self.mir_hash {
            return Err(Error::MirMismatch);
        }
        Ok(())
    }
    /// Public graph/plan copies remain candidates. Even resealed semantic mutations
    /// must extract exactly this privately source-bound reduction.
    pub fn verify_extraction(&self, graph: &Net, plan: &Plan) -> Result<(), Error> {
        let proof = graph
            .verify(crate::loops::GRAPH_LIMITS)
            .map_err(Error::Typed)?;
        let extracted = graph.extract(&proof).map_err(Error::Typed)?;
        if extracted != self.plan || plan != &self.plan {
            return Err(Error::ExtractionMismatch);
        }
        Ok(())
    }
    /// Admission validates full offsets and all logical limits before allocation.
    /// Decreasing/partial offsets can still be valid serial source; these errors
    /// are optimization misses and must not be converted into source traps.
    /// This prepares a local candidate only. It authenticates no reached runtime
    /// instance, Invar owner/loan, source fuel debit or publication authority.
    pub fn prepare<'a>(
        &'a self,
        input: &'a [u32],
        offsets: &'a [u32],
        segments: u32,
        budget: Budget,
    ) -> Result<Prepared<'a>, Error> {
        let segments = usize::try_from(segments).map_err(|_| Error::Budget)?;
        if input.len() > laws::MAX_ELEMENTS
            || segments > laws::MAX_ELEMENTS
            || input.len() > budget.max_elements
            || segments > budget.max_segments
        {
            return Err(Error::Budget);
        }
        if offsets.len() != segments.checked_add(1).ok_or(Error::OffsetShape)? {
            return Err(Error::OffsetShape);
        }
        let elements = u32::try_from(input.len()).map_err(|_| Error::Budget)?;
        if offsets.iter().any(|&n| n > elements) {
            return Err(Error::OffsetBounds);
        }
        if offsets.windows(2).any(|p| p[0] > p[1]) {
            return Err(Error::OffsetOrder);
        }
        if offsets.first() != Some(&0) || offsets.last() != Some(&elements) {
            return Err(Error::IncompleteCoverage);
        }
        let output_bytes = u64::try_from(segments)
            .ok()
            .and_then(|n| n.checked_mul(4))
            .ok_or(Error::Budget)?;
        let work = u64::from(elements)
            .checked_add(u64::try_from(segments).map_err(|_| Error::Budget)?)
            .ok_or(Error::Budget)?;
        if output_bytes > budget.max_output_bytes || work > budget.max_work {
            return Err(Error::Budget);
        }
        self.verify_extraction(&self.graph, &self.plan)?;
        Ok(Prepared {
            contract: self,
            input,
            offsets,
            output_bytes,
            work,
        })
    }
}

/// Immutable borrowed inputs and the source-bound plan. No source owner, loan
/// token or completion permit is reconstructed from this object.
#[derive(Debug)]
pub struct Prepared<'a> {
    contract: &'a Contract,
    input: &'a [u32],
    offsets: &'a [u32],
    output_bytes: u64,
    work: u64,
}
impl Prepared<'_> {
    pub fn offsets(&self) -> &[u32] {
        self.offsets
    }
    pub fn output_bytes(&self) -> u64 {
        self.output_bytes
    }
    pub fn work(&self) -> u64 {
        self.work
    }
    pub fn rule(&self) -> &'static str {
        RULE
    }
    /// Remaining original source ticks after outer-loop dispatch and both bounds,
    /// computed from this contract's recognized checked-AST tick structure.
    /// Allocation before the loop, return after it and source publication are
    /// outside this count; a number of ticks alone grants no execution authority.
    pub fn remaining_loop_ticks(&self) -> Option<u64> {
        let segments = u64::try_from(self.offsets.len().checked_sub(1)?).ok()?;
        let [per_segment, per_element] = self.contract.loop_ticks;
        segments.checked_mul(per_segment)?.checked_add(
            u64::try_from(self.input.len())
                .ok()?
                .checked_mul(per_element)?,
        )
    }
    /// Execute one verified ordered Reduce plan per segment; the same three-node
    /// typed graph serves every segment. Publish only the fully completed fresh
    /// candidate Vec. This is local CPU execution, without native/device emission.
    pub fn execute(&self) -> Result<Vec<u32>, Error> {
        let segments = self
            .offsets
            .len()
            .checked_sub(1)
            .ok_or(Error::OffsetShape)?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(segments)
            .map_err(|_| Error::Allocation)?;
        for pair in self.offsets.windows(2) {
            let begin = usize::try_from(pair[0]).map_err(|_| Error::OffsetBounds)?;
            let end = usize::try_from(pair[1]).map_err(|_| Error::OffsetBounds)?;
            let values = self.input.get(begin..end).ok_or(Error::OffsetBounds)?;
            output.push(
                self.contract
                    .plan
                    .run_reduce(&[values], &[self.contract.law.identity()], values.len())
                    .map_err(Error::Reduction)?,
            );
        }
        Ok(output)
    }
}
